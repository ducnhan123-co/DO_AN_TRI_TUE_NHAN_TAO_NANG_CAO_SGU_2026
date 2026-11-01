"""
c4env.py - Môi trường huấn luyện Connect 4 (self-play + bể đối thủ) cho MaskablePPO.

So với env cũ:
  * Quan sát TƯƠNG ĐỐI: kênh 0 = quân của mình, kênh 1 = quân đối thủ  (2xROWSxCOLS, float32).
    -> không còn phụ thuộc mình là quân 1 hay 2, và mạng chỉ cần học MỘT bài toán thay vì hai.
  * Action masking (action_masks()) -> không còn nước phạm quy, không cần phạt -100.
  * Reward sạch, đối xứng: thắng +1 / thua -1 / hoà 0 (không còn hoà = +1 hay -1 tuỳ ai đi cuối).
  * Kiểm tra thắng TRƯỚC kiểm tra hoà (env cũ báo "hoà" khi nước thứ 42 đồng thời tạo 4 quân thẳng hàng).
  * Đối thủ lấy từ OpponentMixture: random / tactical / minimax nông / các bản chụp (snapshot) của chính mình.
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from c4core import (
    COLS,
    ROWS,
    MinimaxPlayer,
    RandomPlayer,
    TacticalPlayer,
    drop_bit,
    encode,
    has_won,
    is_full,
    legal_mask,
    sensible_mask,
    minimax_scores,
    mirror_cols,
    safe_cols,
    tactical_override,
    legal_cols,
)


# --------------------------------------------------------------------------- mạng CNN (để ở module cố định)
# Lưu ý: class trong policy_kwargs được pickle THEO TÊN MODULE. Nếu để trong train_v2.py (chạy như __main__)
# thì file model chỉ nạp được khi chạy đúng script đó. Đặt ở đây => nạp được từ mọi nơi.
class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class C4Extractor(BaseFeaturesExtractor):
    """CNN nhỏ cho bàn 2xROWSxCOLS. Tích chập 3x3 hợp với bài toán vì 'bốn quân thẳng hàng'
    là mẫu cục bộ lặp lại ở mọi vị trí."""

    def __init__(self, observation_space: spaces.Box, channels=64, n_res=2, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.stem = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[_Res(channels) for _ in range(n_res)])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * ROWS * COLS, features_dim), nn.ReLU())

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# --------------------------------------------------------------------------- lưu / nạp model an toàn
def _custom_objects(lr=3e-4, clip=0.2):
    # learning_rate / lr_schedule / clip_range bị SB3 pickle thành hàm (cloudpickle) vào file zip; nạp lại bằng
    # Python khác phiên bản có thể segfault. Ghi đè bằng hằng số khi nạp => không bao giờ giải mã các hàm đó.
    return {"learning_rate": lr, "lr_schedule": (lambda _: lr), "clip_range": (lambda _: clip)}


def load_model(path, env=None, device="cpu", lr=3e-4, **kwargs):
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(path, env=env, device=device, custom_objects=_custom_objects(lr), **kwargs)


def check_board(model):
    """Báo lỗi sớm nếu model được train cho kích thước bàn khác bàn hiện tại."""
    shape, n = tuple(model.observation_space.shape), int(model.action_space.n)
    if shape != (2, ROWS, COLS) or n != COLS:
        raise ValueError(f"model được train cho bàn {shape[1]}x{shape[2]}, nhưng bàn hiện tại là {ROWS}x{COLS}")
    return model


def save_model(model, path):
    """Lưu KHÔNG kèm lr_schedule / clip_range (hàm) -> file zip không chứa lambda."""
    model.save(path, exclude=["lr_schedule", "clip_range"])


# --------------------------------------------------------------------------- người chơi dùng mạng nơ-ron
class TorchPlayer:
    """Bản sao 'đóng băng' của policy MaskablePPO (đối thủ self-play, đánh giá, hoặc chơi thật).

    Luôn chạy trên CPU (bản sao nhỏ, suy luận 1 ván cờ trên CPU nhanh hơn GPU) nên dùng được cho cả
    model train bằng CUDA/MPS.

    deterministic=True : luôn chọn nước xác suất cao nhất (đánh giá/chơi).
    flip_avg=True      : trung bình dự đoán của bàn cờ và bản lật gương trái-phải.
    shield=True        : mask 1-ply (thắng ngay > chặn ngay > không tặng thắng).
    search_depth=D>0   : chế độ LAI - loại các nước mà minimax độ sâu D cho là kém hơn nước tốt nhất quá
                         `search_margin` điểm (tức tránh mọi thế thua cưỡng bức trong D nước), rồi để policy
                         chọn trong các nước còn lại.
    """

    def __init__(self, policy, deterministic=False, temperature=1.0, eps=0.0, flip_avg=False, shield=True,
                 search_depth=0, search_margin=16, seed=None, name="net"):
        self.fe = copy.deepcopy(policy.features_extractor).cpu().eval()
        self.mlp = copy.deepcopy(policy.mlp_extractor).cpu().eval()
        self.head = copy.deepcopy(policy.action_net).cpu().eval()
        for m in (self.fe, self.mlp, self.head):
            for p in m.parameters():
                p.requires_grad_(False)
        self.deterministic, self.temperature, self.eps = deterministic, temperature, eps
        self.flip_avg, self.shield = flip_avg, shield
        self.search_depth, self.search_margin = search_depth, search_margin
        self.rng = np.random.default_rng(seed)
        self.name = name

    def with_options(self, **kw):
        """Bản dùng chung trọng số nhưng khác tuỳ chọn. Luôn có generator RIÊNG: theo `seed` nếu có,
        nếu không thì lấy hạt giống từ generator gốc (không chia sẻ cùng một dòng số ngẫu nhiên)."""
        new = copy.copy(self)
        seed = kw.pop("seed", None)
        for k, v in kw.items():
            setattr(new, k, v)
        new.rng = np.random.default_rng(seed if seed is not None else int(self.rng.integers(2**31)))
        return new

    # lưu bể đối thủ chỉ gồm tensor + chuỗi => torch.load(weights_only=True) an toàn
    def state(self):
        return dict(fe=self.fe.state_dict(), mlp=self.mlp.state_dict(), head=self.head.state_dict(), name=self.name)

    @classmethod
    def from_state(cls, policy_template, st, **kw):
        p = cls(policy_template, name=st.get("name", "net"), **kw)
        p.fe.load_state_dict(st["fe"])
        p.mlp.load_state_dict(st["mlp"])
        p.head.load_state_dict(st["head"])
        return p

    @torch.inference_mode()
    def probs(self, me, opp):
        x = encode(me, opp)
        legal = sensible_mask(me, opp) if self.shield else legal_mask(me | opp)
        if self.flip_avg:
            batch = torch.from_numpy(np.stack([x, mirror_cols(x)]))
            lg = self.head(self.mlp.forward_actor(self.fe(batch))).numpy().astype(np.float64)
            logits = 0.5 * (lg[0] + lg[1][::-1])
        else:
            logits = self.head(self.mlp.forward_actor(self.fe(torch.from_numpy(x)[None]))).numpy()[0].astype(np.float64)
        logits[~legal] = -np.inf
        z = logits / max(self.temperature, 1e-6)
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def act(self, me, opp):
        mask = me | opp
        if self.eps and self.rng.random() < self.eps:
            cols = legal_cols(mask)
            if not cols:
                raise ValueError("Bàn cờ đã đầy, không còn nước đi hợp lệ")
            return int(cols[self.rng.integers(len(cols))])
        p = self.probs(me, opp)
        if self.search_depth:
            sc = minimax_scores(me, opp, self.search_depth)
            top = max(sc.values())
            ok = [c for c, v in sc.items() if v >= top - self.search_margin]
            q = np.zeros(COLS)
            q[ok] = p[ok]
            if q.sum() <= 0:  # policy không ưa nước nào trong tập an toàn -> chọn đều trong tập đó
                q[ok] = 1.0
            p = q / q.sum()
        if self.deterministic:
            return int(p.argmax())
        return int(self.rng.choice(COLS, p=p))


# --------------------------------------------------------------------------- bể đối thủ
class OpponentMixture:
    """Mỗi ván lấy ngẫu nhiên một đối thủ. Phần lớn là các bản chụp quá khứ của chính agent
    (tránh 'đánh vòng tròn' kéo-búa-bao), phần còn lại là đối thủ có chiến thuật cố định
    (để agent buộc phải học chặn/tạo đe doạ thay vì chỉ khai thác một đối thủ ngốc)."""

    def __init__(self, w_random=0.05, w_tactical=0.15, w_minimax=0.20, w_pool=0.60,
                 pool_size=12, latest_prob=0.5, seed=0):
        self.w = dict(random=w_random, tactical=w_tactical, minimax=w_minimax, pool=w_pool)
        self.pool = []
        self.pool_size, self.latest_prob = pool_size, latest_prob
        self.rng = np.random.default_rng(seed)

    def add_snapshot(self, player):
        self.pool.append(player)
        if len(self.pool) > self.pool_size:
            # giữ bản mới nhất + bản cũ nhất (đa dạng), bỏ một bản ở giữa
            self.pool.pop(1 if self.pool_size > 2 else 0)

    def __call__(self, rng):
        w = dict(self.w)
        if not self.pool:
            extra = w.pop("pool")
            w["tactical"] += extra * 0.5
            w["minimax"] += extra * 0.5
            w["pool"] = 0.0
        keys = list(w)
        p = np.array([w[k] for k in keys], dtype=float)
        kind = keys[int(rng.choice(len(keys), p=p / p.sum()))]
        seed = int(rng.integers(2**31))
        if kind == "random":
            return RandomPlayer(seed)
        if kind == "tactical":
            return TacticalPlayer(eps=float(rng.uniform(0, 0.15)), seed=seed)
        if kind == "minimax":
            depth = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
            return MinimaxPlayer(depth, eps=float(rng.uniform(0, 0.10)), seed=seed)
        snap = self.pool[-1] if rng.random() < self.latest_prob else self.pool[int(rng.integers(len(self.pool)))]
        return snap.with_options(eps=float(rng.uniform(0, 0.05)), seed=seed, deterministic=False)


# --------------------------------------------------------------------------- môi trường
class Connect4Env(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, opponent_sampler=None, tactical_mask=True):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, (2, ROWS, COLS), dtype=np.float32)
        self.action_space = spaces.Discrete(COLS)
        self.sampler = opponent_sampler or (lambda rng: RandomPlayer(int(rng.integers(2**31))))
        self.me = self.opp = 0
        self.opponent = None
        self.tactical_mask = tactical_mask

    # SB3 gọi qua env_method khi dùng DummyVecEnv
    def set_sampler(self, sampler):
        self.sampler = sampler

    def action_masks(self):
        if self.tactical_mask:
            return sensible_mask(self.me, self.opp)
        return legal_mask(self.me | self.opp)

    def _obs(self):
        return encode(self.me, self.opp)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.me = self.opp = 0
        self.opponent = self.sampler(self.np_random)
        self.agent_first = bool(self.np_random.random() < 0.5)
        if not self.agent_first:
            self._opp_move()
        return self._obs(), {"opponent": self.opponent.name, "agent_first": self.agent_first}

    def _opp_move(self):
        a = int(self.opponent.act(self.opp, self.me))
        mv = drop_bit(self.me | self.opp, a) if 0 <= a < COLS else 0
        if not mv:  # đối thủ lỡ đi sai -> đi ngẫu nhiên hợp lệ (không thưởng cho agent)
            cols = legal_cols(self.me | self.opp)
            a = int(cols[self.np_random.integers(len(cols))])
            mv = drop_bit(self.me | self.opp, a)
        self.opp |= mv

    def step(self, action):
        action = int(action)
        mv = drop_bit(self.me | self.opp, action)
        if not mv:  # chỉ xảy ra nếu KHÔNG dùng mask -> xử thua để tương thích PPO thường
            return self._obs(), -1.0, True, False, {"outcome": -1, "illegal": True, "is_success": False}
        self.me |= mv
        if has_won(self.me):
            return self._obs(), 1.0, True, False, {"outcome": 1, "is_success": True}
        if is_full(self.me | self.opp):
            return self._obs(), 0.0, True, False, {"outcome": 0, "is_success": False}
        self._opp_move()
        if has_won(self.opp):
            return self._obs(), -1.0, True, False, {"outcome": -1, "is_success": False}
        if is_full(self.me | self.opp):
            return self._obs(), 0.0, True, False, {"outcome": 0, "is_success": False}
        return self._obs(), 0.0, False, False, {}

    def render(self):
        from c4core import bb_to_grid
        g = bb_to_grid(self.me, self.opp, 1)
        return "\n".join("".join(".XO"[v] for v in row) for row in g)

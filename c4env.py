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
from gymnasium import spaces

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
    mirror_cols,
    safe_cols,
    tactical_override,
    legal_cols,
)


# --------------------------------------------------------------------------- người chơi dùng mạng nơ-ron
class TorchPlayer:
    """Bản sao 'đóng băng' của policy MaskablePPO (dùng làm đối thủ hoặc để đánh giá / chơi thật).

    deterministic=True : luôn chọn nước có xác suất cao nhất (dùng khi đánh giá/chơi).
    flip_avg=True      : trung bình dự đoán của bàn cờ và bản lật gương (Connect4 đối xứng trái-phải).
    shield=True        : lá chắn chiến thuật 1-ply (thắng ngay > chặn ngay > không tặng thắng).
    """

    def __init__(self, policy, deterministic=False, temperature=1.0, eps=0.0,
                 flip_avg=False, shield=True, seed=None, name="net"):
        self.fe = copy.deepcopy(policy.features_extractor).eval()
        self.mlp = copy.deepcopy(policy.mlp_extractor).eval()
        self.head = copy.deepcopy(policy.action_net).eval()
        for m in (self.fe, self.mlp, self.head):
            for p in m.parameters():
                p.requires_grad_(False)
        self.deterministic, self.temperature, self.eps = deterministic, temperature, eps
        self.flip_avg, self.shield = flip_avg, shield
        self.rng = np.random.default_rng(seed)
        self.name = name

    def with_options(self, **kw):
        """Tạo bản dùng chung trọng số nhưng khác tuỳ chọn (không copy lại mạng)."""
        new = copy.copy(self)
        for k, v in kw.items():
            setattr(new, k, v)
        return new

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
            return int(cols[self.rng.integers(len(cols))])
        p = self.probs(me, opp)
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

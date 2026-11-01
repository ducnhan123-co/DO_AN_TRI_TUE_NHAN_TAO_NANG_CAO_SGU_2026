"""
train_v2.py - Huấn luyện agent Connect 4 mạnh hơn.

Chạy thử nhanh :  python train_v2.py --steps 200000 --out runs/quick
Chạy đầy đủ    :  python train_v2.py --steps 3000000 --arch cnn --channels 64 --out runs/main
Tiếp tục       :  python train_v2.py --steps 3000000 --out runs/main --resume runs/main/last.zip
Xem log        :  tensorboard --logdir runs
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from c4core import ROWS, COLS, MinimaxPlayer, RandomPlayer, TacticalPlayer, match
from c4env import C4Extractor, Connect4Env, OpponentMixture, TorchPlayer, load_model, save_model


# --------------------------------------------------------------------------- callback
class SelfPlayCallback(BaseCallback):
    """Định kỳ: (1) chụp policy hiện tại vào bể đối thủ, (2) đánh giá với các đối thủ CỐ ĐỊNH
    (không phụ thuộc vào chính agent nên so sánh được qua thời gian), (3) lưu checkpoint tốt nhất."""

    def __init__(self, mixture, out_dir, warmup=25_000, snap_every=25_000, eval_every=50_000,
                 ckpt_every=250_000, eval_games=(40, 60, 60, 30), verbose=1):
        super().__init__(verbose)
        self.mix, self.out = mixture, out_dir
        self.warmup, self.snap_every, self.eval_every, self.ckpt_every = warmup, snap_every, eval_every, ckpt_every
        self.eval_games = eval_games
        self.best = -1.0
        self._last_snap = self._last_eval = self._last_ckpt = 0
        self.t0 = time.time()

    def _on_training_start(self):
        os.makedirs(os.path.join(self.out, "snapshots"), exist_ok=True)
        # khôi phục điểm tốt nhất khi --resume để KHÔNG ghi đè best.zip bằng model tệ hơn
        bp = os.path.join(self.out, "best_score.json")
        if os.path.exists(bp):
            try:
                self.best = float(json.load(open(bp, encoding="utf-8"))["hard_score"])
                print(f"[resume] khôi phục best hard_score = {self.best:.3f}")
            except (OSError, ValueError, KeyError):
                pass

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n >= self.warmup and n - self._last_snap >= self.snap_every:
            self._last_snap = n
            self.mix.add_snapshot(TorchPlayer(self.model.policy, name=f"snap{n // 1000}k"))
            torch.save([p.state() for p in self.mix.pool], os.path.join(self.out, "pool.pt"))
        if n - self._last_eval >= self.eval_every:
            self._last_eval = n
            self.evaluate(n)
        if n - self._last_ckpt >= self.ckpt_every:
            self._last_ckpt = n
            save_model(self.model, os.path.join(self.out, "snapshots", f"step_{n}"))
        return True

    def evaluate(self, n):
        # cùng cấu hình với lúc chơi thật (flip_avg=True) để best.zip được chọn bằng đúng policy sẽ dùng
        me = TorchPlayer(self.model.policy, deterministic=True, flip_avg=True, name="eval")
        gr, gt, g2, g4 = self.eval_games
        r = {
            "random": match(me, RandomPlayer(1), gr, 0, seed=1)["score"],
            "tactical": match(me, TacticalPlayer(0.0, 2), gt, 2, seed=2)["score"],
            "minimax2": match(me, MinimaxPlayer(2, seed=3), g2, 2, seed=3)["score"],
            "minimax4": match(me, MinimaxPlayer(4, seed=4), g4, 2, seed=4)["score"],
        }
        hard = 0.25 * r["tactical"] + 0.35 * r["minimax2"] + 0.40 * r["minimax4"]
        for k, v in r.items():
            self.logger.record(f"eval/{k}", v)
        self.logger.record("eval/hard_score", hard)
        if hard > self.best:
            self.best = hard
            save_model(self.model, os.path.join(self.out, "best"))
            with open(os.path.join(self.out, "best_score.json"), "w", encoding="utf-8") as f:
                json.dump({"hard_score": hard, "timesteps": n}, f)
        save_model(self.model, os.path.join(self.out, "last"))
        # lưu kết quả đánh giá ra CSV (mỗi lần eval một dòng) để vẽ biểu đồ / đưa vào báo cáo
        path = os.path.join(self.out, "eval_log.csv")
        new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write("timesteps,seconds,board,random,tactical,minimax2,minimax4,hard_score\n")
            f.write(f"{n},{time.time() - self.t0:.0f},{ROWS}x{COLS},{r['random']:.3f},{r['tactical']:.3f},"
                    f"{r['minimax2']:.3f},{r['minimax4']:.3f},{hard:.3f}\n")
        if self.verbose:
            print(f"[eval {n:>9,d} | {time.time() - self.t0:6.0f}s] "
                  f"rand {r['random']:.2f} tac {r['tactical']:.2f} mm2 {r['minimax2']:.2f} "
                  f"mm4 {r['minimax4']:.2f} | hard {hard:.3f} (best {self.best:.3f}) | pool {len(self.mix.pool)}",
                  flush=True)


# --------------------------------------------------------------------------- main
def lin(start, end=None):
    end = start * 0.1 if end is None else end
    return lambda progress_remaining: end + (start - end) * progress_remaining


def build_model(args, vec_env):
    if args.arch == "cnn":
        pk = dict(
            features_extractor_class=C4Extractor,
            features_extractor_kwargs=dict(channels=args.channels, n_res=args.n_res, features_dim=256),
            net_arch=dict(pi=[128], vf=[128]),
            activation_fn=nn.ReLU,
        )
    else:
        pk = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=nn.ReLU)
    return MaskablePPO(
        "MlpPolicy", vec_env, policy_kwargs=pk,
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        learning_rate=args.lr, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        tensorboard_log=os.path.join(args.out, "tb"), seed=args.seed, verbose=1, device=args.device,
    )


def init_from_supervised(model, npz_path):
    """Khởi tạo MaskablePPO (MLP 256x256) từ mạng học có giám sát (cấp 5): chép thân mạng vào cả policy lẫn value,
    đầu policy chép từ đầu ra của mạng giám sát; đầu value giữ khởi tạo ngẫu nhiên."""
    z = np.load(npz_path, allow_pickle=False)
    if (int(z["rows"]), int(z["cols"])) != (ROWS, COLS):
        raise ValueError("mạng giám sát khác kích thước bàn")
    sd = model.policy.state_dict()
    f = lambda k, tr=False: torch.from_numpy(np.ascontiguousarray(z[k].T if tr else z[k]))  # noqa: E731
    with torch.no_grad():
        for net in ("policy_net", "value_net"):
            sd[f"mlp_extractor.{net}.0.weight"].copy_(f("w0", True))
            sd[f"mlp_extractor.{net}.0.bias"].copy_(f("b0"))
            sd[f"mlp_extractor.{net}.2.weight"].copy_(f("w1", True))
            sd[f"mlp_extractor.{net}.2.bias"].copy_(f("b1"))
        sd["action_net.weight"].copy_(f("wp", True))
        sd["action_net.bias"].copy_(f("bp"))
    model.policy.load_state_dict(sd)
    print(f"Đã khởi tạo policy từ mạng giám sát {npz_path}")


def transfer_from_smaller(old_path, new_model, old_rc):
    """Khởi tạo model bàn lớn từ model MLP train trên bàn nhỏ hơn.
    - Lớp ẩn: copy nguyên.
    - Lớp vào / lớp đầu ra policy: ô (r,c) của bàn nhỏ -> ô (r+dr, c+dc) của bàn lớn, căn đáy + căn giữa;
      các ô/cột mới giữ khởi tạo ngẫu nhiên. Nhờ vậy policy cũ ban đầu chơi ở vùng giữa-đáy bàn mới."""
    old = load_model(old_path, device="cpu")
    R0, C0 = old_rc
    dr, dc = ROWS - R0, (COLS - C0) // 2
    osd, nsd = old.policy.state_dict(), new_model.policy.state_dict()
    cell_map = [(p * ROWS * COLS + (r + dr) * COLS + (c + dc), p * R0 * C0 + r * C0 + c)
                for p in range(2) for r in range(R0) for c in range(C0)]
    new_idx = torch.tensor([a for a, _ in cell_map]); old_idx = torch.tensor([b for _, b in cell_map])
    with torch.no_grad():
        for k, v in osd.items():
            nv = nsd[k]
            if v.shape == nv.shape:
                nv.copy_(v)
            elif k.endswith(("policy_net.0.weight", "value_net.0.weight")):  # lớp vào
                nv[:, new_idx] = v[:, old_idx]
            elif k == "action_net.weight":
                nv[dc:dc + C0] = v
            elif k == "action_net.bias":
                nv[dc:dc + C0] = v
            else:
                raise RuntimeError(f"không ánh xạ được {k} {tuple(v.shape)} -> {tuple(nv.shape)}")
    new_model.policy.load_state_dict(nsd)
    print(f"Đã chuyển tri thức từ {old_path} ({R0}x{C0}) sang bàn {ROWS}x{COLS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--arch", choices=["mlp", "cnn"], default="cnn")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--n-res", type=int, default=2)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-sl", default=None, help="file .npz của mạng học có giám sát (cấp 5) để khởi tạo (chỉ mlp)")
    ap.add_argument("--init-from", default=None, help="model MLP train trên bàn nhỏ hơn để khởi tạo")
    ap.add_argument("--init-board", default="6x7", help="kích thước bàn của --init-from (hàng x cột)")
    ap.add_argument("--warmup", type=int, default=25_000)
    ap.add_argument("--snap-every", type=int, default=25_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--threads", type=int, default=0, help="0 = để torch tự chọn")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    mixture = OpponentMixture(seed=args.seed)
    vec_env = DummyVecEnv([lambda: Connect4Env(mixture) for _ in range(args.n_envs)])
    vec_env.seed(args.seed)

    if args.resume:
        model = load_model(args.resume, env=vec_env, device=args.device, lr=args.lr,
                           tensorboard_log=os.path.join(args.out, "tb"))
        pp = os.path.join(args.out, "pool.pt")
        if os.path.exists(pp):
            try:  # weights_only=True: chỉ nạp tensor/chuỗi, không chạy mã pickle tuỳ ý
                mixture.pool = [TorchPlayer.from_state(model.policy, st) for st in torch.load(pp, weights_only=True)]
            except Exception as e:
                print(f"[cảnh báo] không nạp được {pp} ({type(e).__name__}); bỏ qua bể cũ")
        mixture.add_snapshot(TorchPlayer(model.policy, name="resume"))
    else:
        model = build_model(args, vec_env)
        if args.init_sl:
            assert args.arch == "mlp", "--init-sl chỉ hỗ trợ --arch mlp"
            init_from_supervised(model, args.init_sl)
        if args.init_from:
            assert args.arch == "mlp", "chuyển tri thức hiện chỉ hỗ trợ --arch mlp"
            transfer_from_smaller(args.init_from, model, tuple(int(x) for x in args.init_board.split("x")))
    model.lr_schedule = lin(args.lr)  # giảm tuyến tính; chỉ nằm trong RAM, KHÔNG được lưu vào file model
    print(model.policy)
    print("tham số:", sum(p.numel() for p in model.policy.parameters()))

    cb = SelfPlayCallback(mixture, args.out, warmup=args.warmup, snap_every=args.snap_every,
                          eval_every=args.eval_every)
    model.learn(total_timesteps=args.steps, callback=cb, reset_num_timesteps=not args.resume,
                log_interval=20, tb_log_name="ppo")
    cb.evaluate(model.num_timesteps)
    save_model(model, os.path.join(args.out, "final"))
    print("Đã lưu:", os.path.join(args.out, "final.zip"), "| tốt nhất:", os.path.join(args.out, "best.zip"))


if __name__ == "__main__":
    main()

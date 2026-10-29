"""
train_v2.py - Huấn luyện agent Connect 4 mạnh hơn.

Chạy thử nhanh :  python train_v2.py --steps 200000 --out runs/quick
Chạy đầy đủ    :  python train_v2.py --steps 3000000 --arch cnn --channels 64 --out runs/main
Tiếp tục       :  python train_v2.py --steps 3000000 --out runs/main --resume runs/main/last.zip
Xem log        :  tensorboard --logdir runs
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from c4core import ROWS, COLS, MinimaxPlayer, RandomPlayer, TacticalPlayer, match
from c4env import Connect4Env, OpponentMixture, TorchPlayer


# --------------------------------------------------------------------------- mạng
class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class C4Extractor(BaseFeaturesExtractor):
    """CNN nhỏ cho bàn 2x6x7. Tích chập 3x3 hợp với bài toán này vì 'bốn quân thẳng hàng'
    là mẫu cục bộ, lặp lại ở mọi vị trí -> học nhanh hơn và tổng quát hơn MLP trên 42 số."""

    def __init__(self, observation_space: spaces.Box, channels=64, n_res=2, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.stem = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[_Res(channels) for _ in range(n_res)])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * ROWS * COLS, features_dim), nn.ReLU())

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# --------------------------------------------------------------------------- callback
class SelfPlayCallback(BaseCallback):
    """Định kỳ: (1) chụp policy hiện tại vào bể đối thủ, (2) đánh giá với các đối thủ CỐ ĐỊNH
    (không phụ thuộc vào chính agent nên so sánh được qua thời gian), (3) lưu checkpoint tốt nhất."""

    def __init__(self, mixture, out_dir, warmup=25_000, snap_every=25_000, eval_every=50_000,
                 ckpt_every=250_000, eval_games=(40, 40, 40, 20), verbose=1):
        super().__init__(verbose)
        self.mix, self.out = mixture, out_dir
        self.warmup, self.snap_every, self.eval_every, self.ckpt_every = warmup, snap_every, eval_every, ckpt_every
        self.eval_games = eval_games
        self.best = -1.0
        self._last_snap = self._last_eval = self._last_ckpt = 0
        self.t0 = time.time()

    def _on_training_start(self):
        os.makedirs(os.path.join(self.out, "snapshots"), exist_ok=True)

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n >= self.warmup and n - self._last_snap >= self.snap_every:
            self._last_snap = n
            self.mix.add_snapshot(TorchPlayer(self.model.policy, name=f"snap{n // 1000}k"))
            torch.save(self.mix.pool, os.path.join(self.out, "pool.pt"))
        if n - self._last_eval >= self.eval_every:
            self._last_eval = n
            self.evaluate(n)
        if n - self._last_ckpt >= self.ckpt_every:
            self._last_ckpt = n
            self.model.save(os.path.join(self.out, "snapshots", f"step_{n}"))
        return True

    def evaluate(self, n):
        me = TorchPlayer(self.model.policy, deterministic=True, name="eval")
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
            self.model.save(os.path.join(self.out, "best"))
        self.model.save(os.path.join(self.out, "last"))
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
        learning_rate=lin(args.lr), gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        tensorboard_log=os.path.join(args.out, "tb"), seed=args.seed, verbose=1,
    )


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
    ap.add_argument("--warmup", type=int, default=25_000)
    ap.add_argument("--snap-every", type=int, default=25_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--threads", type=int, default=0, help="0 = để torch tự chọn")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    mixture = OpponentMixture(seed=args.seed)
    vec_env = DummyVecEnv([lambda: Connect4Env(mixture) for _ in range(args.n_envs)])
    vec_env.seed(args.seed)

    if args.resume:
        model = MaskablePPO.load(args.resume, env=vec_env, tensorboard_log=os.path.join(args.out, "tb"))
        pp = os.path.join(args.out, "pool.pt")
        if os.path.exists(pp):
            mixture.pool = torch.load(pp, weights_only=False)
        mixture.add_snapshot(TorchPlayer(model.policy, name="resume"))
    else:
        model = build_model(args, vec_env)
    print(model.policy)
    print("tham số:", sum(p.numel() for p in model.policy.parameters()))

    cb = SelfPlayCallback(mixture, args.out, warmup=args.warmup, snap_every=args.snap_every,
                          eval_every=args.eval_every)
    model.learn(total_timesteps=args.steps, callback=cb, reset_num_timesteps=not args.resume,
                log_interval=20, tb_log_name="ppo")
    cb.evaluate(model.num_timesteps)
    model.save(os.path.join(args.out, "final"))
    print("Đã lưu:", os.path.join(args.out, "final.zip"), "| tốt nhất:", os.path.join(args.out, "best.zip"))


if __name__ == "__main__":
    main()

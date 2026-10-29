"""Đánh giá một model MaskablePPO với các đối thủ cố định.
   python evaluate_v2.py models/agent_8x9.zip [--games 100]"""
import argparse, warnings
warnings.filterwarnings("ignore")
import torch
from sb3_contrib import MaskablePPO
from c4core import *
from c4env import TorchPlayer

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--games", type=int, default=100)
a = ap.parse_args()
torch.set_num_threads(1)
m = MaskablePPO.load(a.model)
me = TorchPlayer(m.policy, deterministic=True, flip_avg=True)   # BẮT BUỘC dùng mask khi chơi
print(f"Bàn {ROWS}x{COLS} | model {a.model}")
for name, o, n in [("random", RandomPlayer(1), a.games), ("tactical", TacticalPlayer(0, 2), a.games),
                   ("minimax2", MinimaxPlayer(2, seed=3), a.games), ("minimax4", MinimaxPlayer(4, seed=4), a.games),
                   ("minimax6", MinimaxPlayer(6, seed=5), max(20, a.games // 3))]:
    r = match(me, o, n, 2, seed=7)
    print(f"vs {name:9s} thắng {r['win']:3d} hoà {r['draw']:3d} thua {r['loss']:3d}  (điểm {r['score']:.2f})", flush=True)

"""Đánh giá một model MaskablePPO mới (và so với agent cũ nếu có thư mục code cũ).
   python evaluate_v2.py models/agent_v2.zip [--old ../drmkn_connect4_src]"""
import argparse, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from sb3_contrib import MaskablePPO
from c4core import *
from c4env import TorchPlayer

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--old", default=None)
ap.add_argument("--games", type=int, default=100); a = ap.parse_args()
torch.set_num_threads(1)
m = MaskablePPO.load(a.model)
me = TorchPlayer(m.policy, deterministic=True, flip_avg=True)   # BẮT BUỘC dùng mask khi chơi
opps = [("random", RandomPlayer(1)), ("tactical", TacticalPlayer(0, 2)),
        ("minimax2", MinimaxPlayer(2, seed=3)), ("minimax4", MinimaxPlayer(4, seed=4)), ("minimax6", MinimaxPlayer(6, seed=5))]
if a.old:
    sys.path.insert(0, a.old)
    from stable_baselines3 import PPO
    from heuristic import HeuristicAgent
    opps += [("heuristic(cũ)", LegacyGridPlayer(HeuristicAgent())),
             ("agent15(cũ)", LegacyGridPlayer(PPO.load(f"{a.old}/saved_agents/agent15")))]
for name, o in opps:
    n = a.games if "6" not in name else max(20, a.games // 3)
    r = match(me, o, n, 2, seed=7)
    print(f"vs {name:14s} thắng {r['win']:3d} hoà {r['draw']:3d} thua {r['loss']:3d}  (điểm {r['score']:.2f})")

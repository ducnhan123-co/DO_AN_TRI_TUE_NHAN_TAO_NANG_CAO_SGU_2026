"""Đánh giá một model MaskablePPO với các đối thủ cố định.
   python evaluate_v2.py models/agent_8x9.zip [--games 100]"""
import argparse, warnings
warnings.filterwarnings("ignore")
import torch
from c4core import *
from c4env import TorchPlayer, check_board, load_model

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--games", type=int, default=100)
ap.add_argument("--search", type=int, default=0, help="chế độ lai: minimax độ sâu N loại nước tệ (0 = tắt)")
ap.add_argument("--save", default=None, help="ghi kết quả vào file CSV (nối thêm nếu đã có)")
a = ap.parse_args()
if a.games < 1:
    ap.error("--games phải >= 1")
torch.set_num_threads(1)
m = check_board(load_model(a.model))
me = TorchPlayer(m.policy, deterministic=True, flip_avg=True, search_depth=a.search)   # BẮT BUỘC dùng mask khi chơi
print(f"Bàn {ROWS}x{COLS} | model {a.model}" + (f" | lai minimax d{a.search}" if a.search else ""))
rows = []
for name, o, n in [("random", RandomPlayer(1), a.games), ("tactical", TacticalPlayer(0, 2), a.games),
                   ("minimax2", MinimaxPlayer(2, seed=3), a.games), ("minimax4", MinimaxPlayer(4, seed=4), a.games),
                   ("minimax6", MinimaxPlayer(6, seed=5), max(20, a.games // 3))]:
    r = match(me, o, n, 2, seed=7)
    rows.append((a.model, f"{ROWS}x{COLS}", name, n, r["win"], r["draw"], r["loss"], r["score"]))
    print(f"vs {name:9s} thắng {r['win']:3d} hoà {r['draw']:3d} thua {r['loss']:3d}  (điểm {r['score']:.2f})", flush=True)

if a.save:
    import os
    new = not os.path.exists(a.save)
    with open(a.save, "a", encoding="utf-8") as f:
        if new:
            f.write("model,board,opponent,games,win,draw,loss,score\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print("Đã lưu:", a.save)

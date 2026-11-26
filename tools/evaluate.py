"""Đánh giá một model MaskablePPO với các đối thủ cố định.

   python tools/evaluate.py models/agent_8x9.zip [--games 100]
   python tools/evaluate.py models/agent_8x9.zip --no-mask                 # bỏ mask chiến thuật 1 nước
   python tools/evaluate.py models/agent_8x9.zip --no-flip                 # bỏ lấy trung bình với bản lật gương
   python tools/evaluate.py models/agent_8x9.zip --opponents random,minimax6 --games 60
   python tools/evaluate.py models/agent_8x9.zip --opponents minimax2,minimax4,minimax6 --d6-games 33

Đối thủ: random | tactical | minimaxN (N = độ sâu). Mặc định: random,tactical,minimax2,minimax4,minimax6.
Minimax độ sâu >= 6 chậm nên mặc định chỉ chơi max(20, games//3) ván; ghi đè bằng --d6-games.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # thư mục gốc dự án
import argparse, warnings
warnings.filterwarnings("ignore")
import torch
from c4.agent_eval import DEFAULT_OPPONENTS, evaluate_player, parse_opponents
from c4.core import *
from c4.env import TorchPlayer, check_board, load_model

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--games", type=int, default=100)
ap.add_argument("--search", type=int, default=0, help="chế độ lai: minimax độ sâu N loại nước tệ (0 = tắt)")
ap.add_argument("--no-mask", action="store_true",
                help="ABLATION: tắt mask chiến thuật 1 nước (thắng ngay/chặn ngay/không tặng thắng) khi chơi; "
                     "chỉ còn mask nước hợp lệ")
ap.add_argument("--no-flip", action="store_true", help="ABLATION: tắt lấy trung bình với bản lật gương trái-phải")
ap.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS),
                help="danh sách đối thủ, phân tách bằng dấu phẩy: random,tactical,minimaxN (hoặc 'all')")
ap.add_argument("--d6-games", type=int, default=None, help="số ván với minimax độ sâu >= 6 (mặc định max(20, games//3))")
ap.add_argument("--opening-plies", type=int, default=2, help="số nước khai cuộc ngẫu nhiên")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--save", default=None, help="ghi kết quả vào file CSV (nối thêm nếu đã có)")
a = ap.parse_args()
if a.games < 1:
    ap.error("--games phải >= 1")
if a.d6_games is not None and a.d6_games < 1:
    ap.error("--d6-games phải >= 1")
try:
    opps = parse_opponents(a.opponents)
except ValueError as e:
    ap.error(str(e))
torch.set_num_threads(1)
m = check_board(load_model(a.model))
shield, flip = not a.no_mask, not a.no_flip
me = TorchPlayer(m.policy, deterministic=True, flip_avg=flip, shield=shield, search_depth=a.search)
tags = [t for t, on in (("KHÔNG mask", not shield), ("KHÔNG lật gương", not flip),
                        (f"lai minimax d{a.search}", bool(a.search))) if on]
variant = ("full" if not tags else "+".join(tags)).replace(" ", "_")
print(f"Bàn {ROWS}x{COLS} | model {a.model}" + (" | " + ", ".join(tags) if tags else " | cấu hình đầy đủ (mask + gương)"))
if not shield:
    print("  CHÚ Ý: tắt mask => tác tử có thể bỏ lỡ nước thắng/chặn 1 nước; đây là thí nghiệm ablation, không phải cấu hình chơi.")

rows = []


def show(r):
    rows.append((a.model, f"{ROWS}x{COLS}", r["opponent"], r["games"], r["win"], r["draw"], r["loss"], round(r["score"], 4),
                 variant))
    print(f"vs {r['opponent']:9s} thắng {r['win']:3d} hoà {r['draw']:3d} thua {r['loss']:3d}  (điểm {r['score']:.2f} "
          f"[{r['wilson_lo']:.2f}; {r['wilson_hi']:.2f}], đi trước {r['score_first']:.2f} / đi sau {r['score_second']:.2f})",
          flush=True)


evaluate_player(me, opps, a.games, a.opening_plies, a.seed, progress=show, slow_games=a.d6_games)

if a.save:
    new = not os.path.exists(a.save)
    with open(a.save, "a", encoding="utf-8") as f:
        if new:
            f.write("model,board,opponent,games,win,draw,loss,score,variant\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print("Đã lưu:", a.save)

"""
evaluate_levels.py - Đo sức mạnh thang 8 cấp độ.

    python evaluate_levels.py                       # mặc định 20 ván mỗi phép đo
    python evaluate_levels.py --games 40 --levels 3,4,8 --save results/levels_ladder.csv

Mỗi cấp đấu với: Random, Minimax d2, Minimax d4 (mốc cố định) và với cấp ngay dưới nó (so sánh bậc thang).
Điểm = (thắng + 0.5*hoà)/số ván, đổi bên luân phiên, 2 nước mở đầu ngẫu nhiên.
"""
import argparse
import csv
import os
import time
import warnings

warnings.filterwarnings("ignore")

from c4core import COLS, ROWS, MinimaxPlayer, RandomPlayer, match
from levels import LEVEL_NAMES, is_ready, make_level
from train_levels import parse_levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="all")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--save", default=None, help="ghi CSV (nối thêm)")
    a = ap.parse_args()
    if a.games < 1:
        ap.error("--games phải >= 1")
    rows = []
    print(f"Bàn {ROWS}x{COLS} | {a.games} ván mỗi phép đo | điểm = (thắng + 0.5*hoà)/ván\n")
    print(f"{'Cấp':<30}{'vs Random':>11}{'vs Mm d2':>10}{'vs Mm d4':>10}{'vs cấp dưới':>13}")
    for n in parse_levels(a.levels):
        if not is_ready(n):
            print(f"Cấp {n}: chưa train, bỏ qua")
            continue
        t0 = time.time()
        p = make_level(n, seed=1)
        res = {}
        for tag, opp in (("random", RandomPlayer(11)), ("minimax2", MinimaxPlayer(2, seed=12)),
                         ("minimax4", MinimaxPlayer(4, seed=13))):
            res[tag] = match(p, opp, a.games, 2, seed=100 + n)
        res["cap_duoi"] = match(p, make_level(n - 1, seed=2), a.games, 2, seed=200 + n) if n > 1 else None
        fmt = lambda r: "   -" if r is None else f"{r['score']:.2f}"  # noqa: E731
        print(f"{n}. {LEVEL_NAMES[n]:<27}{fmt(res['random']):>11}{fmt(res['minimax2']):>10}{fmt(res['minimax4']):>10}"
              f"{fmt(res['cap_duoi']):>13}   ({time.time() - t0:.0f}s)", flush=True)
        for tag, r in res.items():
            if r is not None:
                rows.append((f"{ROWS}x{COLS}", n, LEVEL_NAMES[n], tag, a.games, r["win"], r["draw"], r["loss"],
                             round(r["score"], 3)))
    if a.save and rows:
        new = not os.path.exists(a.save)
        with open(a.save, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["board", "level", "name", "opponent", "games", "win", "draw", "loss", "score"])
            w.writerows(rows)
        print("Đã lưu:", a.save)


if __name__ == "__main__":
    main()

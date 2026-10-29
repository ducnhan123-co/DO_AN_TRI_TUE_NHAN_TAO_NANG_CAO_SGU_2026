"""Kiểm thử lõi luật với kích thước bàn hiện tại (C4_ROWS x C4_COLS)."""
import random, time
import numpy as np
from c4core import *
import c4core


def brute_win(grid, s):
    R, C = grid.shape
    for r in range(R):
        for c in range(C):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if all(0 <= r + dr * i < R and 0 <= c + dc * i < C and grid[r + dr * i, c + dc * i] == s for i in range(4)):
                    return True
    return False


print(f"Bàn {ROWS} hàng x {COLS} cột, bitboard {H * COLS} bit")
rng = random.Random(1)
n = 0
for g in range(1500):
    grid = np.zeros((ROWS, COLS), dtype=np.int8); heights = [0] * COLS; bbs = [0, 0]
    for ply in range(ROWS * COLS):
        cols = [c for c in range(COLS) if heights[c] < ROWS]
        if not cols:
            break
        c = rng.choice(cols); r = ROWS - 1 - heights[c]; heights[c] += 1
        p = ply % 2; grid[r, c] = p + 1; bbs[p] |= drop_bit(bbs[0] | bbs[1], c)
        for pp in (0, 1):
            assert has_won(bbs[pp]) == brute_win(grid, pp + 1)
        assert grid_to_bb(grid, 1) == (bbs[0], bbs[1])
        assert (bb_to_grid(bbs[0], bbs[1], 1) == grid).all()
        assert (legal_mask(bbs[0] | bbs[1]) == np.array([h < ROWS for h in heights])).all()
        n += 1
        if has_won(bbs[0]) or has_won(bbs[1]):
            break
print("bitboard OK trên", n, "vị trí")


def play_moves(seq):
    me = opp = 0
    for c in seq:
        mv = drop_bit(me | opp, c); me |= mv; me, opp = opp, me
    return me, opp  # me = người sắp đi

L = COLS - 1
# X: 0,1,2 ở đáy, O chồng 3 quân ở cột cuối -> X thắng ngay ở cột 3 (và phải ưu tiên thắng hơn chặn)
me, opp = play_moves([0, L, 1, L, 2, L])
assert minimax_move(me, opp, 2) == 3
# O chồng 3 quân ở giữa -> X phải chặn
mid = COLS // 2
me, opp = play_moves([0, mid, 1, mid, L, mid])
assert minimax_move(me, opp, 2) == mid
assert sensible_mask(me, opp).sum() == 1
print("minimax / sensible_mask OK")

# alpha-beta == minimax thuần
def plain(me, opp, depth, ply):
    mask = me | opp; moves = []
    for c in COL_ORDER:
        mv = drop_bit(mask, c)
        if mv:
            if has_won(me | mv):
                return WIN - ply
            moves.append(mv)
    if not moves:
        return 0
    if depth == 0:
        return evaluate(me, opp)
    return max(-plain(opp, me | mv, depth - 1, ply + 1) for mv in moves)

rg = np.random.default_rng(5); bad = tot = 0
for _ in range(40):
    me = opp = 0
    for _ in range(int(rg.integers(2, 20))):
        c = int(rg.choice(legal_cols(me | opp))); mv = drop_bit(me | opp, c); me |= mv; me, opp = opp, me
    if has_won(me) or has_won(opp):
        continue
    for d in (2, 3):
        tot += 1; bad += c4core._negamax(me, opp, d, -INF, INF, 0) != plain(me, opp, d, 0)
print(f"alpha-beta vs minimax thuần: {tot - bad}/{tot} khớp")
assert bad == 0

me = opp = 0
for d in (2, 4, 6):
    t = time.time(); minimax_move(me, opp, d, rg); print(f"depth {d} (thế mở đầu): {time.time() - t:.3f}s")
print(match(MinimaxPlayer(4, seed=1), RandomPlayer(2), 20))

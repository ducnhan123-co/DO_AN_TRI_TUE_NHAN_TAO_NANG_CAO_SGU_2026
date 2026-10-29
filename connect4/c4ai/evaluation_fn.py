"""Hàm lượng giá tĩnh dựa trên "cửa sổ 4 ô" (dùng cho Heuristic và Alpha-Beta).

Có 69 cửa sổ 4 ô liên tiếp trên bàn 7x6 (24 ngang + 21 dọc + 12 + 12 chéo).
Với mỗi cửa sổ chỉ chứa quân của một bên:
    3 quân + 1 ô trống : +5 (cho mình) / -4 (cho đối thủ: phải coi trọng phòng thủ)
    2 quân + 2 ô trống : +2 / -2
Cộng thêm điều khiển cột giữa: +3 điểm mỗi quân của mình ở cột 3, trừ cho đối thủ.
"""

from __future__ import annotations

from .game import COLS, ROWS, H1

# -- trọng số (có thể chỉnh để ablation) ------------------------------------
W_THREE_MINE = 5
W_TWO_MINE = 2
W_THREE_OPP = -4
W_TWO_OPP = -2
W_CENTER = 3


def _build_windows() -> list[int]:
    wins = []
    for c in range(COLS):
        for r in range(ROWS):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                ec, er = c + 3 * dc, r + 3 * dr
                if 0 <= ec < COLS and 0 <= er < ROWS:
                    m = 0
                    for k in range(4):
                        m |= 1 << ((c + k * dc) * H1 + (r + k * dr))
                    wins.append(m)
    return wins


WINDOWS: list[int] = _build_windows()
assert len(WINDOWS) == 69
CENTER_MASK = sum(1 << (3 * H1 + r) for r in range(ROWS))


def evaluate_bitboards(mine: int, opp: int) -> int:
    """Điểm của thế cờ theo góc nhìn bên có bitboard `mine` (cao = tốt cho mình)."""
    score = W_CENTER * ((mine & CENTER_MASK).bit_count() - (opp & CENTER_MASK).bit_count())
    for w in WINDOWS:
        a = (mine & w).bit_count()
        o = (opp & w).bit_count()
        if a and o:
            continue
        if a == 3:
            score += W_THREE_MINE
        elif a == 2:
            score += W_TWO_MINE
        elif o == 3:
            score += W_THREE_OPP
        elif o == 2:
            score += W_TWO_OPP
    return score


def evaluate(game, player: int) -> int:
    return evaluate_bitboards(game.bb[player], game.bb[3 - player])

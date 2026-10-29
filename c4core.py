"""
c4core.py - Luật Connect 4 bằng bitboard (thuần Python int) + các "người chơi" dùng chung.

Quy ước
-------
* Mỗi cột dùng 7 bit (6 ô + 1 bit đệm phía trên) -> bit = col * 7 + row_from_bottom.
  Bit đệm luôn = 0 nên các phép dịch bit không bao giờ "nối" nhầm giữa hai cột.
* "me" / "opp" là bitboard quân của người sắp đi / người còn lại (góc nhìn của người đi).
* Mọi Player có hàm  act(me, opp) -> cột (0..COLS-1).
* Khi chuyển sang mảng numpy (obs / grid) thì hàng 0 là hàng TRÊN CÙNG (giống code cũ).
"""
from __future__ import annotations

import os

import numpy as np

# Kích thước bàn: mặc định 8 hàng x 9 cột. Đặt biến môi trường C4_ROWS / C4_COLS để đổi
# (vd C4_ROWS=6 C4_COLS=7 là bàn cổ điển). Model train ở kích thước nào phải dùng đúng kích thước đó.
ROWS = int(os.environ.get("C4_ROWS", 8))
COLS = int(os.environ.get("C4_COLS", 9))
H = ROWS + 1
NBYTES = (H * COLS + 7) // 8
# ưu tiên cột giữa -> alpha-beta cắt tỉa tốt hơn
COL_ORDER = tuple(sorted(range(COLS), key=lambda c: (abs(c - (COLS - 1) / 2), c)))

BOT = [1 << (c * H) for c in range(COLS)]
COLMASK = [((1 << ROWS) - 1) << (c * H) for c in range(COLS)]
BOARD_MASK = sum(COLMASK)
CENTER = COLMASK[COLS // 2]

# chỉ số bit của từng ô theo toạ độ numpy (hàng 0 = trên cùng)
_IDX = np.array(
    [[c * H + (ROWS - 1 - r) for c in range(COLS)] for r in range(ROWS)], dtype=np.int64
)


# --------------------------------------------------------------------------- luật
def has_won(bb: int) -> bool:
    m = bb & (bb >> H)  # ngang
    if m & (m >> 2 * H):
        return True
    m = bb & (bb >> (H - 1))  # chéo \
    if m & (m >> 2 * (H - 1)):
        return True
    m = bb & (bb >> (H + 1))  # chéo /
    if m & (m >> 2 * (H + 1)):
        return True
    m = bb & (bb >> 1)  # dọc
    return bool(m & (m >> 2))


def drop_bit(mask: int, col: int) -> int:
    """Bit của ô mà quân sẽ rơi vào; 0 nếu cột đầy."""
    return (mask + BOT[col]) & COLMASK[col]


def legal_cols(mask: int):
    return [c for c in range(COLS) if (mask >> (c * H + ROWS - 1)) & 1 == 0]


def legal_mask(mask: int) -> np.ndarray:
    return np.array([(mask >> (c * H + ROWS - 1)) & 1 == 0 for c in range(COLS)])


def is_full(mask: int) -> bool:
    return mask & BOARD_MASK == BOARD_MASK


# --------------------------------------------------------------------------- chuyển đổi
def to_plane(bb: int) -> np.ndarray:
    # bitboard có thể > 64 bit (bàn lớn) nên đi qua bytes thay vì numpy int64
    bits = np.unpackbits(np.frombuffer(bb.to_bytes(NBYTES, "little"), dtype=np.uint8), bitorder="little")
    return bits[_IDX].astype(np.float32)


def encode(me: int, opp: int) -> np.ndarray:
    """Quan sát (2, ROWS, COLS): kênh 0 = quân của người đi, kênh 1 = quân đối thủ.
    Biểu diễn theo góc nhìn người đi => agent không cần biết mình là quân 1 hay 2."""
    return np.stack([to_plane(me), to_plane(opp)])


def mirror_cols(x: np.ndarray) -> np.ndarray:
    return x[..., ::-1].copy()


def grid_to_bb(grid: np.ndarray, symbol: int):
    """grid (6x7 gồm 0/1/2, hàng 0 = trên) -> (bitboard của `symbol`, bitboard của đối thủ)."""
    a = b = 0
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r, c]
            if v == 0:
                continue
            bit = 1 << int(_IDX[r, c])
            if v == symbol:
                a |= bit
            else:
                b |= bit
    return a, b


def bb_to_grid(me: int, opp: int, my_symbol: int) -> np.ndarray:
    g = np.zeros((ROWS, COLS), dtype=np.int8)
    g[to_plane(me) > 0] = my_symbol
    g[to_plane(opp) > 0] = 3 - my_symbol
    return g


# --------------------------------------------------------------------------- chiến thuật 1-ply
def winning_cols(me: int, mask: int):
    out = []
    for c in range(COLS):
        mv = drop_bit(mask, c)
        if mv and has_won(me | mv):
            out.append(c)
    return out


def safe_cols(me: int, opp: int):
    """Các cột không tạo điều kiện cho đối thủ thắng ngay ở ô ngay phía trên."""
    mask = me | opp
    out = []
    for c in range(COLS):
        mv = drop_bit(mask, c)
        if not mv:
            continue
        above = mv << 1
        if above & COLMASK[c] and has_won(opp | above):
            continue
        out.append(c)
    return out


def sensible_mask(me: int, opp: int) -> np.ndarray:
    """Mask 'có lý' (kiến thức 1-ply đưa vào mask): nếu có nước thắng ngay -> chỉ nước đó;
    elif đối thủ sắp thắng -> chỉ các nước chặn; else loại các nước tặng đối thủ thắng ngay."""
    mask = me | opp
    out = np.zeros(COLS, dtype=bool)
    cols = winning_cols(me, mask) or winning_cols(opp, mask) or safe_cols(me, opp) or legal_cols(mask)
    out[cols] = True
    return out


def tactical_override(me: int, opp: int, probs: np.ndarray | None = None):
    """'Lá chắn chiến thuật': thắng ngay > chặn ngay > tránh nước tặng đối thủ thắng.
    Trả về cột, hoặc None nếu không có gì đặc biệt (để dùng policy)."""
    mask = me | opp
    w = winning_cols(me, mask)
    if w:
        return w[0]
    b = winning_cols(opp, mask)
    if b:
        return b[0]
    return None


# --------------------------------------------------------------------------- minimax nhanh
WIN = 1_000_000
INF = 10**9


def _threes(me: int, empty: int, d: int) -> int:
    a = me & (me >> d) & (me >> 2 * d) & (empty >> 3 * d)
    b = me & (me >> d) & (empty >> 2 * d) & (me >> 3 * d)
    c = me & (empty >> d) & (me >> 2 * d) & (me >> 3 * d)
    e = empty & (me >> d) & (me >> 2 * d) & (me >> 3 * d)
    return a.bit_count() + b.bit_count() + c.bit_count() + e.bit_count()


def evaluate(me: int, opp: int) -> int:
    empty = BOARD_MASK & ~(me | opp)
    s = 3 * ((me & CENTER).bit_count() - (opp & CENTER).bit_count())
    for d in (1, H, H - 1, H + 1):
        s += 8 * (_threes(me, empty, d) - _threes(opp, empty, d))
    return s


def _negamax(me: int, opp: int, depth: int, alpha: int, beta: int, ply: int) -> int:
    mask = me | opp
    moves = []
    for c in COL_ORDER:
        mv = (mask + BOT[c]) & COLMASK[c]
        if mv:
            if has_won(me | mv):
                return WIN - ply
            moves.append(mv)
    if not moves:
        return 0
    if depth == 0:
        return evaluate(me, opp)
    # nước ép buộc: nếu đối thủ có ô thắng ngay thì phải chặn
    must = [mv for mv in moves if has_won(opp | mv)]
    if len(must) >= 2:
        return -(WIN - ply - 1)
    if must:
        moves = must
    best = -INF
    for mv in moves:
        s = -_negamax(opp, me | mv, depth - 1, -beta, -alpha, ply + 1)
        if s > best:
            best = s
            if s > alpha:
                alpha = s
                if alpha >= beta:
                    break
    return best


def minimax_move(me: int, opp: int, depth: int, rng: np.random.Generator | None = None) -> int:
    """Chọn cột tốt nhất theo negamax + alpha-beta; hoà điểm thì chọn ngẫu nhiên."""
    mask = me | opp
    best, best_cols, alpha = -INF, [], -INF
    for c in COL_ORDER:
        mv = drop_bit(mask, c)
        if not mv:
            continue
        if has_won(me | mv):
            return c
        s = -_negamax(opp, me | mv, depth - 1, -INF, -alpha, 1)
        if s > best:
            best, best_cols, alpha = s, [c], s - 1
        elif s == best:
            best_cols.append(c)
    if rng is None or len(best_cols) == 1:
        return best_cols[0]
    return int(best_cols[rng.integers(len(best_cols))])


# --------------------------------------------------------------------------- người chơi
class RandomPlayer:
    name = "random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, me, opp):
        cols = legal_cols(me | opp)
        return int(cols[self.rng.integers(len(cols))])


class TacticalPlayer:
    """Thắng ngay > chặn ngay > tránh tặng thắng > ưu tiên giữa. Đối thủ 'có mắt' nhưng nông."""

    name = "tactical"

    def __init__(self, eps=0.0, seed=None):
        self.eps = eps
        self.rng = np.random.default_rng(seed)

    def act(self, me, opp):
        mask = me | opp
        if self.rng.random() < self.eps:
            cols = legal_cols(mask)
            return int(cols[self.rng.integers(len(cols))])
        o = tactical_override(me, opp)
        if o is not None:
            return o
        cols = safe_cols(me, opp) or legal_cols(mask)
        mid = (COLS - 1) / 2
        w = np.array([COLS // 2 + 1 - abs(mid - c) for c in cols], dtype=float)
        return int(cols[self.rng.choice(len(cols), p=w / w.sum())])


class MinimaxPlayer:
    def __init__(self, depth=4, eps=0.0, seed=None):
        self.depth, self.eps = depth, eps
        self.rng = np.random.default_rng(seed)
        self.name = f"minimax{depth}"

    def act(self, me, opp):
        if self.eps and self.rng.random() < self.eps:
            cols = legal_cols(me | opp)
            return int(cols[self.rng.integers(len(cols))])
        return minimax_move(me, opp, self.depth, self.rng)


class LegacyGridPlayer:
    """Bọc agent CŨ (predict(grid 6x7 gồm 0/1/2, quân 1 đi trước) -> (action, _)).
    Dùng cho PPO agent0..15, HeuristicAgent, EnsembleAgent của code cũ."""

    def __init__(self, agent, name="legacy", deterministic=True):
        self.agent, self.name, self.deterministic = agent, name, deterministic

    def act(self, me, opp):
        my_symbol = 1 if (me | opp).bit_count() % 2 == 0 else 2
        grid = bb_to_grid(me, opp, my_symbol)
        return int(self.agent.predict(grid, deterministic=self.deterministic)[0])


# --------------------------------------------------------------------------- đấu thử
def play_game(p_first, p_second, rng=None, opening_plies=0):
    """Trả về (kết quả cho người đi trước: +1/0/-1, nước phạm quy của người đi trước?, của người đi sau?)."""
    rng = rng or np.random.default_rng()
    bbs = [0, 0]
    players = (p_first, p_second)
    for ply in range(ROWS * COLS):
        cur = ply % 2
        me, opp = bbs[cur], bbs[1 - cur]
        mask = me | opp
        if ply < opening_plies:
            cols = legal_cols(mask)
            a = int(cols[rng.integers(len(cols))])
        else:
            a = int(players[cur].act(me, opp))
        mv = drop_bit(mask, a) if 0 <= a < COLS else 0
        if not mv:  # nước phạm quy => thua ngay (giống env cũ)
            ill = [cur == 0, cur == 1]
            return (-1 if cur == 0 else 1), ill[0], ill[1]
        bbs[cur] |= mv
        if has_won(bbs[cur]):
            return (1 if cur == 0 else -1), False, False
    return 0, False, False


def match(a, b, n_games=40, opening_plies=2, seed=0):
    """a đấu b, đổi bên luân phiên. Trả về dict thống kê từ góc nhìn của a."""
    rng = np.random.default_rng(seed)
    w = d = l = ia = ib = 0
    for g in range(n_games):
        if g % 2 == 0:
            r, i1, i2 = play_game(a, b, rng, opening_plies)
            ia += i1
            ib += i2
        else:
            r, i1, i2 = play_game(b, a, rng, opening_plies)
            r = -r
            ia += i2
            ib += i1
        w += r == 1
        d += r == 0
        l += r == -1
    return dict(win=w, draw=d, loss=l, illegal_a=ia, illegal_b=ib, score=(w + 0.5 * d) / n_games)

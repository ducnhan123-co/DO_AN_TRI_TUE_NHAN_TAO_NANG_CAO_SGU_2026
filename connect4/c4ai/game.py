"""Lõi game Connect Four 7x6 (tự cài đặt, thuần Python, bitboard).

Mô hình hóa bài toán
--------------------
* Hai người chơi: 1 (đi trước) và 2 (đi sau). Quan sát đầy đủ, xác định,
  tổng bằng không, không có yếu tố ngẫu nhiên.
* Trạng thái: bàn 7 cột x 6 hàng + người đến lượt.
* Hành động: số cột 0..6 (chỉ hợp lệ khi cột chưa đầy).
* Chuyển trạng thái: quân rơi xuống ô trống thấp nhất của cột.
* Kết thúc: một bên có 4 quân liên tiếp (ngang/dọc/chéo) hoặc bàn đầy (hòa).
* Reward theo góc nhìn người chơi p: +1 thắng, 0 hòa, -1 thua.

Biểu diễn bitboard (mỗi cột 7 bit: 6 bit quân + 1 bit đệm ở trên cùng):

    .  .  .  .  .  .  .      bit 5 | 12 | 19 | 26 | 33 | 40 | 47   (hàng trên)
    ...                       ...
    0  7 14 21 28 35 42       (hàng dưới cùng)

API: reset(), legal_actions(), step(action), is_terminal(), result(),
render(), clone(), undo() (+ reward(player), key()).
"""

from __future__ import annotations

COLS = 7
ROWS = 6
H1 = ROWS + 1  # số bit mỗi cột (có 1 bit đệm)
N_CELLS = COLS * ROWS

# Thứ tự duyệt cột: từ giữa ra hai bên (dùng cho move ordering / tie-break)
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)

_BOTTOM = [1 << (c * H1) for c in range(COLS)]
_TOP = [1 << (c * H1 + ROWS - 1) for c in range(COLS)]  # ô trên cùng của cột


def _is_win(b: int) -> bool:
    """True nếu bitboard b có 4 bit liên tiếp theo 1 trong 4 hướng."""
    m = b & (b >> H1)  # ngang
    if m & (m >> (2 * H1)):
        return True
    m = b & (b >> (H1 + 1))  # chéo /
    if m & (m >> (2 * (H1 + 1))):
        return True
    m = b & (b >> (H1 - 1))  # chéo \
    if m & (m >> (2 * (H1 - 1))):
        return True
    m = b & (b >> 1)  # dọc
    return bool(m & (m >> 2))


def is_win_bitboard(b: int) -> bool:
    return _is_win(b)


class Connect4:
    """Môi trường Connect Four. Người chơi là 1 và 2; người 1 đi trước."""

    __slots__ = ("bb", "mask", "heights", "moves", "winner", "to_move")

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> "Connect4":
        self.bb = [0, 0, 0]  # bb[1], bb[2]: bitboard của từng người chơi
        self.mask = 0  # tất cả ô đã có quân
        self.heights = [0] * COLS
        self.moves: list[int] = []  # lịch sử nước đi (để undo)
        self.winner = 0  # 0: chưa có người thắng
        self.to_move = 1
        return self

    def legal_actions(self) -> list[int]:
        """Danh sách cột còn trống (rỗng nếu ván đã kết thúc)."""
        if self.winner:
            return []
        h = self.heights
        return [c for c in range(COLS) if h[c] < ROWS]

    def is_legal(self, action: int) -> bool:
        return (
            not self.winner
            and isinstance(action, int)
            and 0 <= action < COLS
            and self.heights[action] < ROWS
        )

    def step(self, action: int) -> "Connect4":
        """Thả quân của người đến lượt vào cột `action`.

        Ném ValueError nếu nước đi bất hợp lệ (cột đầy / ngoài phạm vi /
        ván đã kết thúc) -> cấm nước đi bất hợp lệ.
        """
        if not self.is_legal(action):
            raise ValueError(f"Nước đi bất hợp lệ: {action!r}")
        p = self.to_move
        bit = 1 << (action * H1 + self.heights[action])
        self.bb[p] |= bit
        self.mask |= bit
        self.heights[action] += 1
        self.moves.append(action)
        if _is_win(self.bb[p]):
            self.winner = p
        self.to_move = 3 - p
        return self

    def undo(self) -> int:
        """Hoàn tác nước đi gần nhất, trả về cột của nước đó."""
        if not self.moves:
            raise ValueError("Không còn nước đi để hoàn tác")
        action = self.moves.pop()
        self.to_move = 3 - self.to_move
        self.heights[action] -= 1
        bit = 1 << (action * H1 + self.heights[action])
        self.bb[self.to_move] ^= bit
        self.mask ^= bit
        self.winner = 0  # chỉ nước cuối mới có thể tạo ra người thắng
        return action

    def is_terminal(self) -> bool:
        return bool(self.winner) or len(self.moves) == N_CELLS

    def result(self):
        """1 / 2 nếu có người thắng, 0 nếu hòa, None nếu chưa kết thúc."""
        if self.winner:
            return self.winner
        if len(self.moves) == N_CELLS:
            return 0
        return None

    def reward(self, player: int) -> int:
        """Reward theo góc nhìn `player` khi ván kết thúc: +1 / 0 / -1."""
        r = self.result()
        if r is None:
            raise ValueError("Ván chưa kết thúc")
        if r == 0:
            return 0
        return 1 if r == player else -1

    def clone(self) -> "Connect4":
        g = Connect4.__new__(Connect4)
        g.bb = self.bb[:]
        g.mask = self.mask
        g.heights = self.heights[:]
        g.moves = self.moves[:]
        g.winner = self.winner
        g.to_move = self.to_move
        return g

    # -------------------------------------------------------------- tiện ích
    def key(self) -> tuple:
        """Khóa duy nhất của thế cờ (dùng cho bảng chuyển vị / loại trùng)."""
        return (self.bb[1], self.bb[2])

    def cell(self, row: int, col: int) -> int:
        """Quân tại (row, col); row 0 là hàng dưới cùng. 0 nếu trống."""
        bit = 1 << (col * H1 + row)
        if self.bb[1] & bit:
            return 1
        if self.bb[2] & bit:
            return 2
        return 0

    def to_grid(self) -> list[list[int]]:
        """Ma trận 6x7, hàng 0 là hàng TRÊN CÙNG (giống cách hiển thị)."""
        return [[self.cell(ROWS - 1 - r, c) for c in range(COLS)] for r in range(ROWS)]

    @classmethod
    def from_moves(cls, moves) -> "Connect4":
        g = cls()
        for m in moves:
            g.step(m)
        return g

    @classmethod
    def from_grid(cls, rows: list[str]) -> "Connect4":
        """Tạo thế cờ từ 6 chuỗi (hàng trên -> dưới), ký tự 'X'=1, 'O'=2, '.'=trống.

        Chỉ dùng cho test: chèn thẳng quân, không kiểm tra lượt hợp lệ; người
        đến lượt được suy ra từ số quân (X đi trước).
        """
        assert len(rows) == ROWS and all(len(r) == COLS for r in rows)
        g = cls()
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch == ".":
                    continue
                p = 1 if ch.upper() == "X" else 2
                row = ROWS - 1 - r
                bit = 1 << (c * H1 + row)
                g.bb[p] |= bit
                g.mask |= bit
                g.heights[c] = max(g.heights[c], row + 1)
        n1 = bin(g.bb[1]).count("1")
        n2 = bin(g.bb[2]).count("1")
        g.to_move = 1 if n1 == n2 else 2
        for p in (1, 2):
            if _is_win(g.bb[p]):
                g.winner = p
        # lịch sử giả để is_terminal() đúng khi bàn đầy
        g.moves = [0] * (n1 + n2)
        return g

    def render(self, file=None) -> str:
        """Trả về (và in nếu file != None) bàn cờ dạng text."""
        sym = {0: ".", 1: "X", 2: "O"}
        lines = [" ".join(sym[v] for v in row) for row in self.to_grid()]
        lines.append(" ".join(str(c) for c in range(COLS)))
        text = "\n".join(lines)
        if file is not None:
            print(text, file=file)
        return text

    def __repr__(self) -> str:  # pragma: no cover
        return self.render()

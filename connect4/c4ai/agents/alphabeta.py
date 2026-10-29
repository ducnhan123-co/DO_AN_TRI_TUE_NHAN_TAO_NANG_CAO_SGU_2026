"""Cấp 3 - Minimax / Alpha-Beta (dạng negamax).

* Giới hạn độ sâu `depth`; nếu đặt `time_limit` (giây/nước) thì dùng iterative
  deepening: lặp depth = 1, 2, ... và dùng kết quả của độ sâu hoàn thành cuối cùng.
* `prune=False` tắt cắt tỉa (minimax thuần) để so sánh số nút và thời gian.
* `use_eval=False` thì nút lá chưa kết thúc có giá trị 0 (không có hàm lượng giá).
* Điểm thắng = WIN - ply  -> ưu tiên thắng nhanh, kéo dài khi sắp thua.
* Duyệt cột từ giữa ra biên (move ordering) để cắt tỉa hiệu quả; khi nhiều nước
  cùng điểm thì chọn nước đứng trước trong thứ tự đó (xác định, không ngẫu nhiên).
* Đếm số nút (mỗi lần gọi negamax = 1 nút) vào last_stats["nodes"].
"""

from __future__ import annotations

import time

from ..evaluation_fn import evaluate
from ..game import CENTER_ORDER, COLS, N_CELLS, ROWS
from .base import Agent

WIN = 1_000_000
INF = 10**9


class _Timeout(Exception):
    pass


class AlphaBetaAgent(Agent):
    name = "alphabeta"
    level = 3

    def __init__(self, depth=4, prune=True, use_eval=True, time_limit=None, max_depth=42, seed=None):
        super().__init__(seed)
        self.depth = depth
        self.prune = prune
        self.use_eval = use_eval
        self.time_limit = time_limit
        self.max_depth = max_depth
        tag = f"alphabeta(d={depth}" if time_limit is None else f"alphabeta(t={time_limit}s"
        if not prune:
            tag += ",noprune"
        if not use_eval:
            tag += ",noeval"
        self.name = tag + ")"
        self._nodes = 0
        self._deadline = None

    # ---------------------------------------------------------------- search
    def _negamax(self, g, depth, alpha, beta, ply):
        self._nodes += 1
        if self._deadline is not None and (self._nodes & 255) == 0 and time.perf_counter() > self._deadline:
            raise _Timeout
        if g.winner:  # người vừa đi đã thắng -> bên đến lượt thua
            return -(WIN - ply)
        if len(g.moves) == N_CELLS:
            return 0
        if depth == 0:
            return evaluate(g, g.to_move) if self.use_eval else 0
        best = -INF
        h = g.heights
        for c in CENTER_ORDER:
            if h[c] >= ROWS:
                continue
            g.step(c)
            v = -self._negamax(g, depth - 1, -beta, -alpha, ply + 1)
            g.undo()
            if v > best:
                best = v
                if self.prune:
                    if best > alpha:
                        alpha = best
                    if alpha >= beta:
                        break
        return best

    def _search_root(self, g, depth, first=None):
        order = [c for c in CENTER_ORDER if g.heights[c] < ROWS]
        if first is not None and first in order:
            order.remove(first)
            order.insert(0, first)
        best_v, best_c = -INF, order[0]
        alpha = -INF
        for c in order:
            g.step(c)
            v = -self._negamax(g, depth - 1, -INF, -alpha if self.prune else INF, 1)
            g.undo()
            if v > best_v:
                best_v, best_c = v, c
                if self.prune and best_v > alpha:
                    alpha = best_v
        return best_c, best_v

    def select_action(self, game) -> int:
        g = game.clone()
        self._nodes = 0
        reached = 0
        if self.time_limit is None:
            self._deadline = None
            move, val = self._search_root(g, self.depth)
            reached = self.depth
        else:
            t0 = time.perf_counter()
            move, val, prev = None, 0, None
            for d in range(1, min(self.max_depth, N_CELLS - len(g.moves)) + 1):
                self._deadline = None if d == 1 else t0 + self.time_limit
                try:
                    m, v = self._search_root(g, d, prev)
                except _Timeout:
                    # trạng thái g có thể đang "lệch" do ngắt giữa chừng
                    g = game.clone()
                    break
                move, val, prev, reached = m, v, m, d
                if abs(v) >= WIN - 100:  # đã thấy thắng/thua cưỡng bức
                    break
                if time.perf_counter() - t0 > self.time_limit:
                    break
        self._deadline = None
        self.last_stats = {"nodes": self._nodes, "sims": 0, "depth": reached, "value": val}
        return move

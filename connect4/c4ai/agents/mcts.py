"""Cấp 4 - MCTS với UCT.

Bốn bước chuẩn của mỗi lần mô phỏng:
  1. Selection      : từ gốc đi xuống theo UCB1 = W/N + c * sqrt(ln(N_cha) / N)
                      cho tới nút còn nước chưa thử hoặc nút kết thúc.
  2. Expansion      : thêm 1 con ứng với một nước chưa thử (chọn ngẫu nhiên).
  3. Simulation     : chơi ngẫu nhiên (rollout) tới hết ván.
                      rollout="light": ưu tiên thắng ngay / chặn thắng ngay, còn lại ngẫu nhiên.
  4. Backpropagation: cập nhật N, W lên tới gốc. W của nút được tính theo góc nhìn
                      của NGƯỜI VỪA ĐI vào nút đó (thắng 1, hòa 0.5, thua 0).
Nước đi cuối cùng: con của gốc có số lần thăm lớn nhất ("robust child").

Ngân sách: `n_sims` (số mô phỏng) hoặc `time_limit` (giây/nước) - nếu có
time_limit thì dùng thời gian (n_sims là trần).
"""

from __future__ import annotations

import math
import time

from ..game import H1, ROWS, COLS, CENTER_ORDER, is_win_bitboard
from .base import Agent


class _Node:
    __slots__ = ("parent", "action", "player", "children", "untried", "N", "W")

    def __init__(self, parent, action, player, untried):
        self.parent = parent
        self.action = action
        self.player = player  # người vừa đi để tới nút này
        self.children = []
        self.untried = untried
        self.N = 0
        self.W = 0.0


class MCTSAgent(Agent):
    name = "mcts"
    level = 4

    def __init__(self, n_sims=1000, c=math.sqrt(2), time_limit=None, rollout="random", seed=None):
        super().__init__(seed)
        assert rollout in ("random", "light")
        self.n_sims = n_sims
        self.c = c
        self.time_limit = time_limit
        self.rollout = rollout
        tag = f"mcts(n={n_sims}" if time_limit is None else f"mcts(t={time_limit}s"
        if abs(c - math.sqrt(2)) > 1e-9:
            tag += f",c={c:g}"
        if rollout != "random":
            tag += f",{rollout}"
        self.name = tag + ")"

    # ------------------------------------------------------------------ utils
    def _rollout(self, g):
        rng = self.rng
        h = g.heights
        if self.rollout == "random":
            while not g.is_terminal():
                cols = [c for c in range(COLS) if h[c] < ROWS]
                g.step(rng.choice(cols))
            return
        while not g.is_terminal():
            cols = [c for c in range(COLS) if h[c] < ROWS]
            me, opp = g.to_move, 3 - g.to_move
            move = None
            for c in cols:  # thắng ngay
                if is_win_bitboard(g.bb[me] | (1 << (c * H1 + h[c]))):
                    move = c
                    break
            if move is None:
                for c in cols:  # chặn
                    if is_win_bitboard(g.bb[opp] | (1 << (c * H1 + h[c]))):
                        move = c
                        break
            g.step(move if move is not None else rng.choice(cols))

    def _uct_child(self, node):
        logn = math.log(node.N)
        c = self.c
        best, best_child = -1.0, None
        for ch in node.children:
            u = ch.W / ch.N + c * math.sqrt(logn / ch.N)
            if u > best:
                best, best_child = u, ch
        return best_child

    # ------------------------------------------------------------------ main
    def select_action(self, game) -> int:
        g = game.clone()
        base = len(g.moves)
        legal = [c for c in CENTER_ORDER if g.heights[c] < ROWS]
        if len(legal) == 1:
            self.last_stats = {"nodes": 1, "sims": 0, "depth": 0}
            return legal[0]
        untried = legal[:]
        self.rng.shuffle(untried)
        root = _Node(None, None, 3 - g.to_move, untried)
        deadline = None if self.time_limit is None else time.perf_counter() + self.time_limit
        sims, nodes, max_depth = 0, 1, 0
        while sims < self.n_sims:
            if deadline is not None and (sims & 7) == 0 and time.perf_counter() > deadline:
                break
            node = root
            # 1. selection
            while not node.untried and node.children:
                node = self._uct_child(node)
                g.step(node.action)
            # 2. expansion
            if node.untried and not g.is_terminal():
                a = node.untried.pop()
                mover = g.to_move
                g.step(a)
                ch_untried = [] if g.is_terminal() else [c for c in g.legal_actions()]
                self.rng.shuffle(ch_untried)
                child = _Node(node, a, mover, ch_untried)
                node.children.append(child)
                node = child
                nodes += 1
            depth = len(g.moves) - base
            if depth > max_depth:
                max_depth = depth
            # 3. simulation
            self._rollout(g)
            winner = g.result()
            # 4. backpropagation
            n = node
            while n is not None:
                n.N += 1
                if winner == 0:
                    n.W += 0.5
                elif winner == n.player:
                    n.W += 1.0
                n = n.parent
            while len(g.moves) > base:
                g.undo()
            sims += 1
        best = max(root.children, key=lambda ch: (ch.N, ch.W / ch.N))
        self.last_stats = {
            "nodes": nodes,
            "sims": sims,
            "depth": max_depth,
            "visits": {ch.action: ch.N for ch in root.children},
            "win_rate": best.W / best.N,
        }
        return best.action

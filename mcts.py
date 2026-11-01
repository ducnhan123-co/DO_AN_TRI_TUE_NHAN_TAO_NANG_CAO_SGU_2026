"""
mcts.py - Các tác nhân dùng tìm kiếm cây / mạng nhỏ chạy bằng numpy (không cần torch khi chơi):

  * MCTSPlayer        : Cấp 4 - Monte Carlo Tree Search thuần (UCT + ván giả lập ngẫu nhiên).
  * NpPV              : mạng policy(+value) chạy bằng numpy, nạp từ file .npz (không dùng pickle).
  * SupervisedPlayer  : Cấp 5 - mạng học có giám sát (bắt chước minimax), chọn nước bằng policy.
  * AZSearch/AZPlayer : Cấp 7-8 - AlphaZero: MCTS dạng PUCT dẫn đường bằng mạng policy+value.

Các tác nhân học (5, 7, 8) dùng chung "mask 1 nước" (thắng ngay / chặn ngay / không tặng thắng) như agent PPO.
"""
from __future__ import annotations

import math
import random

import numpy as np

from c4core import (BOT, COLMASK, COLS, H, ROWS, _require_moves, drop_bit, encode, has_won, is_full,
                    legal_mask, mirror_cols, safe_cols, sensible_mask, tactical_override)


# =========================================================================== Cấp 4: Monte Carlo thuần
class _N:
    __slots__ = ("me", "opp", "parent", "col", "kids", "untried", "n", "w", "term")

    def __init__(self, me, opp, parent, col, untried, term=None):
        # me = bitboard người SẮP đi ở nút này; w = tổng kết quả theo góc nhìn người VỪA đi vào nút (cha)
        self.me, self.opp, self.parent, self.col = me, opp, parent, col
        self.kids, self.untried, self.n, self.w, self.term = [], untried, 0, 0.0, term


def _legal(mask):
    return [c for c in range(COLS) if not (mask >> (c * H + ROWS - 1)) & 1]


def _rollout(me, opp, rnd):
    """Chơi ngẫu nhiên tới hết ván. Trả về kết quả theo góc nhìn người sắp đi ở trạng thái xuất phát
    (1 thắng, 0.5 hoà, 0 thua)."""
    cur, oth, turn = me, opp, 0
    left = ROWS * COLS - (me | opp).bit_count()
    while left:
        mask = cur | oth
        while True:
            c = int(rnd() * COLS)
            mv = (mask + BOT[c]) & COLMASK[c]
            if mv:
                break
        cur |= mv
        if has_won(cur):
            return 1.0 if turn == 0 else 0.0
        cur, oth, turn = oth, cur, turn ^ 1
        left -= 1
    return 0.5


class MCTSPlayer:
    """Cấp 4 - Monte Carlo Tree Search thuần: không dùng mạng, không dùng hàm đánh giá; chỉ dựa vào tỉ lệ thắng
    của các ván giả lập ngẫu nhiên. Ở gốc có kiểm tra thắng-ngay / chặn-ngay / tránh-tặng-thắng (rẻ)."""

    def __init__(self, sims=800, c=1.2, seed=None):
        self.sims, self.c = sims, c
        self.rnd = random.Random(seed).random
        self.name = f"mcts{sims}"

    def act(self, me, opp):
        mask = me | opp
        cols = _require_moves(mask)
        o = tactical_override(me, opp)
        if o is not None:
            return o
        root_cols = safe_cols(me, opp) or cols
        if len(root_cols) == 1:
            return int(root_cols[0])
        root = _N(me, opp, None, -1, list(root_cols))
        rnd, c_uct = self.rnd, self.c
        for _ in range(self.sims):
            node = root
            while not node.untried and node.kids and node.term is None:
                lg = math.log(node.n)
                node = max(node.kids, key=lambda k: k.w / k.n + c_uct * math.sqrt(lg / k.n))
            if node.term is not None:
                r = node.term
            elif node.untried:
                col = node.untried.pop(int(rnd() * len(node.untried)))
                m = node.me | node.opp
                mv = drop_bit(m, col)
                new_me = node.me | mv
                if has_won(new_me):
                    child = _N(node.opp, new_me, node, col, [], term=1.0)
                elif is_full(m | mv):
                    child = _N(node.opp, new_me, node, col, [], term=0.5)
                else:
                    child = _N(node.opp, new_me, node, col, _legal(m | mv))
                node.kids.append(child)
                node = child
                r = node.term if node.term is not None else 1.0 - _rollout(node.me, node.opp, rnd)
            else:
                r = 0.5
            while node is not None:
                node.n += 1
                node.w += r
                r = 1.0 - r
                node = node.parent
        return int(max(root.kids, key=lambda k: k.n).col)


# =========================================================================== mạng chạy bằng numpy
class NpPV:
    """Mạng MLP policy (+ value) suy luận bằng numpy. Trọng số lưu dạng (vào, ra); file .npz không chứa pickle."""

    def __init__(self, arrs):
        self.rows, self.cols = int(arrs["rows"]), int(arrs["cols"])
        f = lambda k: np.ascontiguousarray(arrs[k], dtype=np.float32)  # noqa: E731
        self.w0, self.b0, self.w1, self.b1 = f("w0"), f("b0"), f("w1"), f("b1")
        self.wp, self.bp = f("wp"), f("bp")
        self.has_value = "wv" in arrs
        if self.has_value:
            self.wv, self.bv = f("wv"), f("bv")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            net = cls({k: z[k] for k in z.files})
        if (net.rows, net.cols) != (ROWS, COLS):
            raise ValueError(f"model được train cho bàn {net.rows}x{net.cols}, nhưng bàn hiện tại là {ROWS}x{COLS}")
        return net

    def __call__(self, x):
        h = np.maximum(x @ self.w0 + self.b0, 0.0)
        h = np.maximum(h @ self.w1 + self.b1, 0.0)
        logits = h @ self.wp + self.bp
        v = float(np.tanh(h @ self.wv + self.bv)[0]) if self.has_value else 0.0
        return logits.astype(np.float64), v


# =========================================================================== Cấp 5: học có giám sát
class SupervisedPlayer:
    """Cấp 5 - mạng được huấn luyện có giám sát để BẮT CHƯỚC nước đi của minimax; khi chơi chọn nước có
    xác suất cao nhất (trung bình hoá với bản lật gương) trong các nước 'có lý'."""

    def __init__(self, net, flip_avg=True, shield=True, name="supervised"):
        self.net, self.flip_avg, self.shield, self.name = net, flip_avg, shield, name

    def act(self, me, opp):
        mask = me | opp
        _require_moves(mask)
        allowed = sensible_mask(me, opp) if self.shield else legal_mask(mask)
        x = encode(me, opp)
        lg, _ = self.net(x.reshape(-1))
        if self.flip_avg:
            lg2, _ = self.net(mirror_cols(x).reshape(-1))
            lg = 0.5 * (lg + lg2[::-1])
        lg = np.where(allowed, lg, -np.inf)
        return int(lg.argmax())


# =========================================================================== Cấp 7-8: AlphaZero (PUCT)
class _AZNode:
    __slots__ = ("me", "opp", "P", "N", "W", "kids", "allowed")

    def __init__(self, me, opp, P, allowed):
        self.me, self.opp, self.P, self.allowed = me, opp, P, allowed
        self.N, self.W, self.kids = np.zeros(COLS), np.zeros(COLS), {}


class _Term:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v  # giá trị theo góc nhìn người sắp đi ở nút kết thúc


class AZSearch:
    """MCTS kiểu AlphaZero: chọn nước theo Q + c_puct * P * sqrt(N_cha) / (1 + N), lá được đánh giá bằng mạng
    (policy làm xác suất tiên nghiệm, value thay cho ván giả lập). Giá trị lưu theo kiểu negamax."""

    def __init__(self, net, sims=100, c_puct=1.5, dir_alpha=1.0, dir_eps=0.25, rng=None):
        self.net, self.sims, self.c = net, sims, c_puct
        self.alpha, self.eps = dir_alpha, dir_eps
        self.rng = rng or np.random.default_rng()

    def _evaluate(self, me, opp):
        allowed = sensible_mask(me, opp)
        lg, v = self.net(encode(me, opp).reshape(-1))
        z = np.where(allowed, lg, -1e30)
        p = np.exp(z - z.max()) * allowed
        return _AZNode(me, opp, p / p.sum(), allowed), v

    def run(self, me, opp, noise=False):
        """Trả về (số lượt thăm của từng cột ở gốc, nút gốc)."""
        root, _ = self._evaluate(me, opp)
        if noise:
            idx = np.flatnonzero(root.allowed)
            if len(idx) > 1:
                P = root.P.copy()
                P[idx] = (1 - self.eps) * P[idx] + self.eps * self.rng.dirichlet([self.alpha] * len(idx))
                root.P = P
        c = self.c
        for _ in range(self.sims):
            node, path = root, []
            while True:
                tot = node.N.sum()
                Q = np.divide(node.W, node.N, out=np.zeros(COLS), where=node.N > 0)
                score = Q + c * node.P * math.sqrt(tot + 1.0) / (1.0 + node.N)
                score[~node.allowed] = -np.inf
                a = int(score.argmax())
                path.append((node, a))
                child = node.kids.get(a)
                if child is None:
                    m = node.me | node.opp
                    mv = drop_bit(m, a)
                    new_me = node.me | mv
                    if has_won(new_me):
                        child, v = _Term(-1.0), -1.0
                    elif is_full(m | mv):
                        child, v = _Term(0.0), 0.0
                    else:
                        child, v = self._evaluate(node.opp, new_me)
                    node.kids[a] = child
                    break
                if isinstance(child, _Term):
                    v = child.v
                    break
                node = child
            for nd, a in reversed(path):
                v = -v
                nd.N[a] += 1
                nd.W[a] += v
        return root.N, root


class AZPlayer:
    """Cấp 7/8 - AlphaZero: cùng một mạng, khác nhau ở số mô phỏng MCTS lúc chơi."""

    def __init__(self, net, sims=100, seed=None, name=None):
        self.search = AZSearch(net, sims, rng=np.random.default_rng(seed))
        self.sims = sims
        self.name = name or f"alphazero{sims}"

    def act(self, me, opp):
        mask = me | opp
        _require_moves(mask)
        allowed = sensible_mask(me, opp)
        if allowed.sum() == 1:
            return int(allowed.argmax())
        N, root = self.search.run(me, opp, noise=False)
        return int((np.where(allowed, N, -1) + 1e-3 * root.P).argmax())

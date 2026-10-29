"""Giao diện chung của mọi tác tử."""

from __future__ import annotations

import random
import time


class Agent:
    """Tác tử chơi Connect Four.

    select_action(game) trả về một cột hợp lệ cho người đang đến lượt
    (game.to_move). Tác tử KHÔNG được thay đổi `game` (nếu có do/undo thì phải
    trả về đúng trạng thái ban đầu). Sau mỗi lần gọi, `self.last_stats` chứa số
    liệu chi phí tính toán: {"nodes": ..., "sims": ..., "depth": ...}.
    """

    name = "agent"
    level = 0

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.last_stats: dict = {}

    def select_action(self, game) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    def timed_action(self, game):
        """Trả về (action, seconds, stats)."""
        t0 = time.perf_counter()
        a = self.select_action(game)
        dt = time.perf_counter() - t0
        return a, dt, dict(self.last_stats)

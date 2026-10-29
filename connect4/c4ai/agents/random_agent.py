"""Cấp 1 - Random: chọn đều trong các nước đi hợp lệ (baseline)."""

from __future__ import annotations

from .base import Agent


class RandomAgent(Agent):
    name = "random"
    level = 1

    def select_action(self, game) -> int:
        self.last_stats = {"nodes": 0, "sims": 0}
        return self.rng.choice(game.legal_actions())

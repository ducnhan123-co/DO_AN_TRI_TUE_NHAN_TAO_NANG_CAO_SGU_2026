import numpy as np


def _check_win(grid, r, c, piece):
    # horizontal
    for sc in range(max(0, c - 3), min(7, c + 4) - 3):
        if (grid[r, sc : sc + 4] == piece).all():
            return True

    # vertical
    if r <= 2:
        if (grid[r : r + 4, c] == piece).all():
            return True

    # diagonal (top-left to bottom-right)
    for i in range(-3, 1):
        sr, sc = r + i, c + i
        if 0 <= sr <= 2 and 0 <= sc <= 3:
            if all(grid[sr + j, sc + j] == piece for j in range(4)):
                return True

    # diagonal (bottom-left to top-right)
    for i in range(-3, 1):
        sr, sc = r - i, c + i
        if 3 <= sr < 6 and 0 <= sc <= 3:
            if all(grid[sr - j, sc + j] == piece for j in range(4)):
                return True

    return False


def _get_target_row(grid, col):
    for r in range(5, -1, -1):
        if grid[r, col] == 0:
            return r
    return -1


class EnsembleAgent:
    def __init__(self, agents):
        self.agents = agents

    def predict(self, obs, deterministic=True):
        grid = obs.copy()
        count_1 = np.sum(grid == 1)
        count_2 = np.sum(grid == 2)
        my_symb = 1 if count_1 == count_2 else 2
        opp_symb = 2 if my_symb == 1 else 1

        valid_cols = [c for c in range(7) if grid[0, c] == 0]
        if not valid_cols:
            return (0,)

        # 1. Tactical Guard: Immediate Win
        for c in valid_cols:
            r = _get_target_row(grid, c)
            grid[r, c] = my_symb
            if _check_win(grid, r, c, my_symb):
                return (c,)
            grid[r, c] = 0

        # 2. Tactical Guard: Immediate Block
        for c in valid_cols:
            r = _get_target_row(grid, c)
            grid[r, c] = opp_symb
            if _check_win(grid, r, c, opp_symb):
                return (c,)
            grid[r, c] = 0

        # 3. Ensemble Voting with Action Masking
        predictions = [
            agent.predict(obs, deterministic)[0] for agent in self.agents
        ]
        valid_predictions = [p for p in predictions if p in valid_cols]

        if valid_predictions:
            values, counts = np.unique(valid_predictions, return_counts=True)
            return (int(values[np.argmax(counts)]),)

        # Fallback to center-most valid column
        best_col = min(valid_cols, key=lambda c: abs(c - 3))
        return (best_col,)

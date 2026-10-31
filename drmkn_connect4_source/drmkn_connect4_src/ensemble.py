import numpy as np


def _check_win(grid, r, c, piece):
    rows, cols = grid.shape

    # horizontal
    for sc in range(max(0, c - 3), min(cols, c + 4) - 3):
        if (grid[r, sc : sc + 4] == piece).all():
            return True

    # vertical
    if r <= rows - 4:
        if (grid[r : r + 4, c] == piece).all():
            return True

    # diagonal \ (top-left to bottom-right)
    for i in range(-3, 1):
        sr, sc = r + i, c + i
        if 0 <= sr <= rows - 4 and 0 <= sc <= cols - 4:
            if all(grid[sr + j, sc + j] == piece for j in range(4)):
                return True

    # diagonal / (bottom-left to top-right)
    for i in range(-3, 1):
        sr, sc = r - i, c + i
        if 3 <= sr < rows and 0 <= sc <= cols - 4:
            if all(grid[sr - j, sc + j] == piece for j in range(4)):
                return True

    return False


def _get_target_row(grid, col):
    rows = grid.shape[0]
    for r in range(rows - 1, -1, -1):
        if grid[r, col] == 0:
            return r
    return -1


class EnsembleAgent:
    def __init__(self, agents, learner=None):
        self.agents = agents
        self.learner = learner

    def set_learner(self, learner):
        self.learner = learner

    def predict(self, obs, deterministic=True):
        grid = obs.copy()
        rows, cols = grid.shape
        count_1 = np.sum(grid == 1)
        count_2 = np.sum(grid == 2)
        my_symb = 1 if count_1 == count_2 else 2
        opp_symb = 2 if my_symb == 1 else 1

        valid_cols = [c for c in range(cols) if grid[0, c] == 0]
        if not valid_cols:
            return (0,)

        # 1. Tactical Guard: Immediate Win
        for c in valid_cols:
            r = _get_target_row(grid, c)
            if r != -1:
                grid[r, c] = my_symb
                if _check_win(grid, r, c, my_symb):
                    return (c,)
                grid[r, c] = 0

        # 2. Tactical Guard: Immediate Block
        for c in valid_cols:
            r = _get_target_row(grid, c)
            if r != -1:
                grid[r, c] = opp_symb
                if _check_win(grid, r, c, opp_symb):
                    return (c,)
                grid[r, c] = 0

        # 3. Adaptive Learned Counter-move (Bắt bài bẫy của người chơi)
        if self.learner is not None:
            learned_col = self.learner.lookup_memory(grid)
            if learned_col is not None and learned_col in valid_cols:
                return (int(learned_col),)

        # 4. Ensemble Voting with Action Masking
        predictions = []
        for agent in self.agents:
            try:
                obs_input = obs
                if hasattr(agent, "observation_space"):
                    sp = getattr(agent.observation_space, "shape", None)
                    if sp == (2, rows, cols):
                        obs_input = np.stack(
                            [(grid == my_symb).astype(np.float32), (grid == opp_symb).astype(np.float32)],
                            axis=0,
                        )
                p = agent.predict(obs_input, deterministic)[0]
                if isinstance(p, (np.ndarray, list, tuple)):
                    p = int(p[0])
                else:
                    p = int(p)
                predictions.append(p)
            except Exception:
                pass

        valid_predictions = [p for p in predictions if p in valid_cols]

        if valid_predictions:
            values, counts = np.unique(valid_predictions, return_counts=True)
            return (int(values[np.argmax(counts)]),)

        # Fallback to center-most valid column
        center_col = cols // 2
        best_col = min(valid_cols, key=lambda c: abs(c - center_col))
        return (best_col,)

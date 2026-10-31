import numpy as np


class RandomAgent:
    """
    Cấp 1: Random Agent - Nước đi ngẫu nhiên hợp lệ.
    Lựa chọn ngẫu nhiên đồng đều (Uniform Random) trong các cột chưa đầy.
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def predict(self, obs, deterministic=False):
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            grid = obs[0] if obs.shape[0] == 2 else obs
        else:
            grid = obs

        cols = grid.shape[1]
        valid_cols = [c for c in range(cols) if grid[0, c] == 0]
        if not valid_cols:
            return (0,)
        return (int(np.random.choice(valid_cols)),)

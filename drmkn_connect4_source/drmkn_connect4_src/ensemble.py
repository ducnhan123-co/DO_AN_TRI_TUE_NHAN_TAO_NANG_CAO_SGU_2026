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


def _is_suicide_move(grid, col, my_symb, opp_symb):
    """
    Kiểm tra xem thả quân vào (r, col) có tạo điều kiện cho đối thủ thả đè lên (r-1, col) để thắng ngay không.
    """
    r = _get_target_row(grid, col)
    if r <= 0:
        return False
    grid[r, col] = my_symb
    grid[r - 1, col] = opp_symb
    is_opp_win = _check_win(grid, r - 1, col, opp_symb)
    grid[r - 1, col] = 0
    grid[r, col] = 0
    return is_opp_win


def _find_fork_moves(grid, piece):
    """
    Tìm các cột mà nếu thả quân `piece` vào sẽ tạo ra từ 2 cơ hội thắng trở lên (bẫy 2 đầu / Fork).
    """
    cols = grid.shape[1]
    fork_cols = []
    for c in range(cols):
        r = _get_target_row(grid, c)
        if r != -1:
            grid[r, c] = piece
            win_count = 0
            for c2 in range(cols):
                r2 = _get_target_row(grid, c2)
                if r2 != -1:
                    grid[r2, c2] = piece
                    if _check_win(grid, r2, c2, piece):
                        win_count += 1
                    grid[r2, c2] = 0
            grid[r, c] = 0
            if win_count >= 2:
                fork_cols.append(c)
    return fork_cols


class EnsembleAgent:
    def __init__(self, agents, learner=None):
        self.agents = agents
        self.learner = learner

    def set_learner(self, learner):
        self.learner = learner

    def predict(self, obs, deterministic=True):
        grid = obs.copy()
        rows, cols = grid.shape
        center_col = cols // 2
        count_1 = np.sum(grid == 1)
        count_2 = np.sum(grid == 2)
        my_symb = 1 if count_1 == count_2 else 2
        opp_symb = 2 if my_symb == 1 else 1

        valid_cols = [c for c in range(cols) if grid[0, c] == 0]
        if not valid_cols:
            return (0,)

        # -------------------------------------------------------------
        # 1. TACTICAL GUARD (CẤP 1): NƯỚC ĐI THẮNG NGAY (4 QUÂN)
        # -------------------------------------------------------------
        for c in valid_cols:
            r = _get_target_row(grid, c)
            if r != -1:
                grid[r, c] = my_symb
                if _check_win(grid, r, c, my_symb):
                    return (c,)
                grid[r, c] = 0

        # -------------------------------------------------------------
        # 2. TACTICAL GUARD (CẤP 2): CHẶN ĐỐI THỦ THẮNG NGAY (4 QUÂN)
        # -------------------------------------------------------------
        for c in valid_cols:
            r = _get_target_row(grid, c)
            if r != -1:
                grid[r, c] = opp_symb
                if _check_win(grid, r, c, opp_symb):
                    return (c,)
                grid[r, c] = 0

        # -------------------------------------------------------------
        # 3. TACTICAL GUARD (CẤP 3): TỰ TẠO THẾ BẪY 2 ĐẦU (FORK)
        # -------------------------------------------------------------
        my_forks = _find_fork_moves(grid, my_symb)
        safe_my_forks = [c for c in my_forks if not _is_suicide_move(grid, c, my_symb, opp_symb)]
        if safe_my_forks:
            best_fork = min(safe_my_forks, key=lambda c: abs(c - center_col))
            return (best_fork,)

        # -------------------------------------------------------------
        # 4. TACTICAL GUARD (CẤP 4): CHẶN TRƯỚC BẪY 2 ĐẦU CỦA ĐỐI THỦ
        #    (Đoán trước nước cờ nguy hiểm của người chơi để dập tắt ngay)
        # -------------------------------------------------------------
        opp_forks = _find_fork_moves(grid, opp_symb)
        if opp_forks:
            safe_opp_blocks = [c for c in opp_forks if not _is_suicide_move(grid, c, my_symb, opp_symb)]
            target_blocks = safe_opp_blocks if safe_opp_blocks else opp_forks
            best_block = min(target_blocks, key=lambda c: abs(c - center_col))
            return (best_block,)

        # -------------------------------------------------------------
        # 5. ADAPTIVE MEMORY LOOKUP: BẮT BÀI THẾ CỜ ĐÃ HỌC TỪ NGƯỜI CHƠI
        # -------------------------------------------------------------
        if self.learner is not None:
            learned_col = self.learner.lookup_memory(grid)
            if learned_col is not None and learned_col in valid_cols:
                if not _is_suicide_move(grid, learned_col, my_symb, opp_symb):
                    return (int(learned_col),)

        # -------------------------------------------------------------
        # 6. SUICIDE FILTER: LOẠI BỎ NƯỚC TỰ SÁT (BẮC CẦU CHO ĐỐI PHƯƠNG)
        # -------------------------------------------------------------
        safe_cols = [c for c in valid_cols if not _is_suicide_move(grid, c, my_symb, opp_symb)]
        candidate_cols = safe_cols if safe_cols else valid_cols

        # -------------------------------------------------------------
        # 7. ENSEMBLE VOTING: 16 MÔ HÌNH PPO BỎ PHIẾU TRÊN CỘT AN TOÀN
        # -------------------------------------------------------------
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

        valid_predictions = [p for p in predictions if p in candidate_cols]

        if valid_predictions:
            values, counts = np.unique(valid_predictions, return_counts=True)
            return (int(values[np.argmax(counts)]),)

        # Dự phòng: Chọn cột gần trung tâm nhất trong các cột an toàn
        best_col = min(candidate_cols, key=lambda c: abs(c - center_col))
        return (best_col,)

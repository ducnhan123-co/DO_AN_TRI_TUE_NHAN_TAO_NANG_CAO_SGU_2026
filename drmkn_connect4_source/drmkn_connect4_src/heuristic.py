import random
import numpy as np


class HeuristicAgent:
    def __init__(self, c=np.sqrt(2)):
        self.action_counts = None
        self.timesteps = 0
        self.c = c

    def _determine_turn(self, grid):
        count_1 = np.sum(grid == 1)
        count_2 = np.sum(grid == 2)
        if count_1 == count_2:
            return 1  # Player 1's turn
        elif count_1 == count_2 + 1:
            return 2  # Player 2's turn
        else:
            return 1

    def predict(self, obs, deterministic=True):
        self.grid = obs.copy()
        rows, cols = self.grid.shape
        if self.action_counts is None or len(self.action_counts) != cols:
            self.action_counts = np.ones(cols)
            self.timesteps = 0

        self.self_symb = self._determine_turn(self.grid)
        self.opp_symb = 2 if self.self_symb == 1 else 1

        ucb_scores = np.array([float("-inf")] * cols)
        valid_columns = self._get_valid_columns(self.grid)
        if not valid_columns:
            return (0,)

        scores = list()

        for action in valid_columns:
            grid = self.grid.copy()
            grid_copy = self.grid.copy()
            score = 1

            # Winning move for self
            row = self._drop_piece(grid, action, self.self_symb)
            if row is not None and self._check_game_over(grid, row, action):
                score += 10000

            # Evaluated board on other heuristics
            if row is not None:
                score += self._evaluate_board(grid, row, action)

            # Blocking winning move for opponent
            row_opp = self._drop_piece(grid_copy, action, self.opp_symb)
            if row_opp is not None and self._check_game_over(grid_copy, row_opp, action):
                score += 9000

            scores.append(score)

        ucb_scores[valid_columns] = np.array(scores)

        self.timesteps += 1
        if self.c > 0:
            ucb_scores += self.c * np.sqrt(np.log(max(2, self.timesteps)) / self.action_counts)

        chosen_action = int(np.argmax(ucb_scores))
        self.action_counts[chosen_action] += 1

        return (chosen_action,)

    def _get_valid_columns(self, grid):
        cols = grid.shape[1]
        return [col for col in range(cols) if grid[0, col] == 0]

    def _drop_piece(self, grid, action, player):
        rows = grid.shape[0]
        for row in range(rows - 1, -1, -1):
            if grid[row, action] == 0:
                grid[row, action] = player
                return row
        return None

    def _check_game_over(self, grid, row, col):
        if row is None or col is None:
            return False

        rows, cols = grid.shape
        player_symb = grid[row, col]

        # Horizontal
        for sc in range(max(0, col - 3), min(cols, col + 4) - 3):
            if all(grid[row, sc + i] == player_symb for i in range(4)):
                return True

        # Vertical
        if row <= rows - 4:
            if all(grid[row + i, col] == player_symb for i in range(4)):
                return True

        # Diagonal \ (positive slope top-left to bottom-right)
        for i in range(-3, 1):
            sr, sc = row + i, col + i
            if 0 <= sr <= rows - 4 and 0 <= sc <= cols - 4:
                if all(grid[sr + j, sc + j] == player_symb for j in range(4)):
                    return True

        # Diagonal / (negative slope bottom-left to top-right)
        for i in range(-3, 1):
            sr, sc = row - i, col + i
            if 3 <= sr < rows and 0 <= sc <= cols - 4:
                if all(grid[sr - j, sc + j] == player_symb for j in range(4)):
                    return True

        return False

    def _evaluate_board(self, grid, row, col):
        score = 0
        rows, cols = grid.shape
        center_col = cols // 2

        # Preference for center columns
        center_dist = abs(col - center_col)
        score += max(0, (cols // 2 + 1 - center_dist) * 6)

        # Check for potential chains (2-in-a-row, 3-in-a-row)
        score += self._count_chains(grid, row, col, length=2) * 5
        score += self._count_chains(grid, row, col, length=3) * 25

        return score

    def _count_chains(self, grid, row, col, length):
        count = 0
        rows, cols = grid.shape
        player = grid[row, col]

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            consecutive = 1
            # Forward
            for step in range(1, 4):
                r, c = row + dr * step, col + dc * step
                if 0 <= r < rows and 0 <= c < cols and grid[r, c] == player:
                    consecutive += 1
                else:
                    break
            # Backward
            for step in range(1, 4):
                r, c = row - dr * step, col - dc * step
                if 0 <= r < rows and 0 <= c < cols and grid[r, c] == player:
                    consecutive += 1
                else:
                    break

            if consecutive >= length:
                count += 1

        return count

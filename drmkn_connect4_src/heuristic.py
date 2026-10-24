import random
import numpy as np


class HeuristicAgent:
    def __init__(self,c=np.sqrt(2)):
        self.action_counts = np.ones(7)
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
            raise ValueError(
                "Invalid grid state: counts of 1s and 2s are not consistent."
            )

    def predict(self, obs, deterministic=True):
        self.self_symb = self._determine_turn(obs)
        self.opp_symb = 2 if self.self_symb == 1 else 1
        self.grid = obs.copy()

        ucb_scores = np.array([float("-inf")] * 7)
        valid_columns = self._get_valid_columns(self.grid)
        scores = list()

        for action in valid_columns:
            grid = self.grid.copy()
            grid_copy = self.grid.copy()  # used only for checking opponent's win
            score = 1

            # Winning move for self
            row = self._drop_piece(grid, action, self.self_symb)
            if self._check_game_over(grid, row, action):
                score += 10000

            # Evaluated board on other heuristics
            score += self._evaluate_board(grid, row, action)

            # Blocking winning move for opponent
            row = self._drop_piece(grid_copy, action, self.opp_symb)
            if self._check_game_over(grid_copy, row, action):
                score += 9000

            scores.append(score)

        ucb_scores[valid_columns] = np.array(scores)

        self.timesteps += 1
        ucb_scores += self.c * np.sqrt(np.log(self.timesteps) / self.action_counts)

        chosen_action = np.argmax(ucb_scores)
        self.action_counts[chosen_action] += 1

        return (chosen_action,)

    def _get_valid_columns(self, grid):
        return [col for col in range(7) if grid[0, col] == 0]

    def _drop_piece(self, grid, action, player):
        for row in range(5, -1, -1):  # Find the lowest empty row
            if grid[row, action] == 0:
                grid[row, action] = player
                return row

    def _check_game_over(self, grid, row, col):
        """Checks whether placing a piece at (row, col) results in a win."""
        player_symb = grid[row, col]

        # Horizontal check
        for start_col in range(max(0, col - 3), min(7, col + 4) - 3):
            if all(grid[row, start_col + i] == player_symb for i in range(4)):
                return True

        # Vertical check
        if row <= 2:  # Check only if there are at least 4 rows below
            if all(grid[row + i, col] == player_symb for i in range(4)):
                return True

        # Diagonal (positive slope) check
        for i in range(-3, 1):
            start_row, start_col = row + i, col + i
            if (
                0 <= start_row <= 2
                and 0 <= start_col <= 3
                and all(
                    grid[start_row + j, start_col + j] == player_symb for j in range(4)
                )
            ):
                return True

        # Diagonal (negative slope) check
        for i in range(-3, 1):
            start_row, start_col = row - i, col + i
            if (
                3 <= start_row <= 5
                and 0 <= start_col <= 3
                and all(
                    grid[start_row - j, start_col + j] == player_symb for j in range(4)
                )
            ):
                return True

        return False

    def _evaluate_board(self, grid, row, col):
        """A simple heuristic to evaluate board strength."""
        score = 0

        # Central column preference
        center_col = 3
        score += 3 - abs(center_col - col)

        # Connected pieces heuristic
        score += self._count_connections(grid, row, col, self.self_symb) * 10

        return score

    def _count_connections(self, grid, row, col, player):
        """Counts connected pieces around the placed piece."""
        directions = [
            (0, 1),  # Horizontal
            (1, 0),  # Vertical
            (1, 1),  # Diagonal positive slope
            (1, -1),  # Diagonal negative slope
        ]

        total_connections = 0
        for dr, dc in directions:
            count = 1  # Count the piece just placed
            # Check in the positive direction
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < 6 and 0 <= c < 7 and grid[r, c] == player:
                    count += 1
                else:
                    break
            # Check in the negative direction
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < 6 and 0 <= c < 7 and grid[r, c] == player:
                    count += 1
                else:
                    break
            total_connections += count - 1  # Subtract 1 to avoid double-counting

        return total_connections

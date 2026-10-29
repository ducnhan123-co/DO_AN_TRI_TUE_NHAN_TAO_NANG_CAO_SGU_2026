import numpy as np
import gymnasium as gym
import pygame
import time

WIN_REWARD = 10
BLOCK_REWARD = 3
DRAW_REWARD = 1
CONTINUE_REWARD = 0
ILLEGAL_MOVE_PENALTY = -100


# Define the Connect4 environment with dynamic board size
class Connect4(gym.Env):
    def __init__(self, rows=8, cols=9):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.action_space = gym.spaces.Discrete(self.cols)
        self.observation_space = gym.spaces.Box(
            low=0, high=2, shape=(self.rows, self.cols), dtype=np.int8
        )
        self.opp_agent = None
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        # Pygame variables
        self.screen = None
        self.cell_size = 90
        self.win_width = self.cols * self.cell_size
        self.win_height = self.rows * self.cell_size
        self.radius = int(self.cell_size / 2 - 5)
        self.colors = {
            0: (0, 0, 0),       # Empty
            1: (255, 0, 0),     # Red
            2: (255, 255, 0),   # Yellow
        }

    def step(self, action):
        if not self._is_valid_action(action):
            return (
                self.grid.copy(),
                ILLEGAL_MOVE_PENALTY,
                True,
                False,
                {"is_success": False},
            )
        row = self._drop_piece(action, self.agent_symb)
        reward, done = self._check_game_over(row, action)
        if done:
            info = {}
            if reward != 0:
                info = {"is_success": True}
            else:
                info = {"is_success": False}
            return self.grid.copy(), reward, True, False, info

        # Check if the agent's move blocks a potential win of the opponent
        self.grid[row, action] = self.opp_symb
        x_reward, x_done = self._check_game_over(row, action)
        self.grid[row, action] = self.agent_symb

        # Opponent's turn
        opponent_action = self._get_opp_action()
        row = self._drop_piece(opponent_action, self.opp_symb)
        reward, done = self._check_game_over(row, opponent_action)
        if done:
            info = {"is_success": False}
            return self.grid.copy(), -reward, True, False, info

        if x_done and x_reward != 0:
            return self.grid.copy(), BLOCK_REWARD, False, False, {}
        return self.grid.copy(), 0, False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.agent_symb, self.opp_symb = 1, 2
        if np.random.random() < 0.5:
            self.agent_symb, self.opp_symb = 2, 1
            action = self._get_opp_action()
            self._drop_piece(action, self.opp_symb)
        return self.grid.copy(), {"symbols": (self.agent_symb, self.opp_symb)}

    def render(self, mode="human"):
        time.sleep(1)
        if mode == "stdout":
            print(self.grid)
            print("-" * 20)
        else:
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.win_width, self.win_height))
                pygame.display.set_caption(f"Connect 4 ({self.rows}x{self.cols})")

            self.screen.fill((0, 0, 255))
            for row in range(self.rows):
                for col in range(self.cols):
                    pygame.draw.circle(
                        self.screen,
                        self.colors[self.grid[row, col]],
                        (
                            col * self.cell_size + self.cell_size // 2,
                            row * self.cell_size + self.cell_size // 2,
                        ),
                        self.radius,
                    )
            pygame.display.flip()

    def _drop_piece(self, action, player):
        for row in range(self.rows - 1, -1, -1):
            if self.grid[row, action] == 0:
                self.grid[row, action] = player
                return row
        return None

    def _is_valid_action(self, action):
        if action < 0 or action >= self.cols:
            return False
        return self.grid[0, action] == 0

    def _get_random_action(self):
        valid_cols = [c for c in range(self.cols) if self._is_valid_action(c)]
        return np.random.choice(valid_cols)

    def _get_opp_action(self):
        if not self.opp_agent:
            return self._get_random_action()
        else:
            if np.random.random() < self.opp_epsilon:
                return self._get_random_action()
            else:
                try:
                    action = self.opp_agent.predict(self.grid, deterministic=False)[0]
                    if isinstance(action, (np.ndarray, list, tuple)):
                        action = int(action[0])
                    else:
                        action = int(action)
                    if self._is_valid_action(action):
                        return action
                except Exception:
                    pass
                return self._get_random_action()

    def _check_game_over(self, row, col):
        if row is None or col is None:
            return CONTINUE_REWARD, False

        # Draw check
        if np.all(self.grid != 0):
            return DRAW_REWARD, True

        player_symb = self.grid[row, col]

        # Horizontal
        for i in range(4):
            start_row, start_col = row, col - i
            if start_col >= 0 and start_col + 3 < self.cols:
                if np.all(self.grid[start_row, start_col : start_col + 4] == player_symb):
                    return WIN_REWARD, True

        # Vertical
        for i in range(4):
            start_row, start_col = row - i, col
            if start_row >= 0 and start_row + 3 < self.rows:
                if np.all(self.grid[start_row : start_row + 4, start_col] == player_symb):
                    return WIN_REWARD, True

        # Diagonal \ (top-left to bottom-right)
        for i in range(4):
            start_row, start_col = row - i, col - i
            if (
                start_row >= 0
                and start_col >= 0
                and start_row + 3 < self.rows
                and start_col + 3 < self.cols
            ):
                if np.all(
                    self.grid[start_row : start_row + 4, start_col : start_col + 4].diagonal()
                    == player_symb
                ):
                    return WIN_REWARD, True

        # Diagonal / (bottom-left to top-right)
        for i in range(4):
            start_row, start_col = row + i, col - i
            if (
                start_row < self.rows
                and start_col >= 0
                and start_row - 3 >= 0
                and start_col + 3 < self.cols
            ):
                if np.all(
                    np.fliplr(
                        self.grid[start_row - 3 : start_row + 1, start_col : start_col + 4]
                    ).diagonal()
                    == player_symb
                ):
                    return WIN_REWARD, True

        return CONTINUE_REWARD, False

    def set_opp_agent(self, agent, epsilon=0.1):
        self.opp_agent = agent
        self.opp_epsilon = epsilon

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

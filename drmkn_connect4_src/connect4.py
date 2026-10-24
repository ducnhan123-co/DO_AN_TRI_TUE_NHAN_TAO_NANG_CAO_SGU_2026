import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import pygame
from ensemble import EnsembleAgent
from heuristic import HeuristicAgent
import time
from tqdm import tqdm

WIN_REWARD = 10
BLOCK_REWARD = 3
DRAW_REWARD = 1
CONTINUE_REWARD = 0
ILLEGAL_MOVE_PENALTY = -100


# Define the Connect4 environment with random opponent
class Connect4(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(7)  # 7 columns
        self.observation_space = gym.spaces.Box(
            low=0, high=2, shape=(6, 7), dtype=np.int8
        )
        self.opp_agent = None
        self.grid = np.zeros((6, 7), dtype=np.int8)

        # Pygame variables
        self.screen = None
        self.cell_size = 100  # Size of each cell in pixels
        self.win_width = 7 * self.cell_size
        self.win_height = 6 * self.cell_size
        self.radius = int(self.cell_size / 2 - 5)  # Radius of the tokens
        self.colors = {
            0: (0, 0, 0),  # Black for empty cells
            1: (255, 0, 0),  # Red for Player 1
            2: (255, 255, 0),  # Yellow for Player 2
        }

    def step(self, action):
        if not self._is_valid_action(action):
            return (
                self.grid.copy(),
                ILLEGAL_MOVE_PENALTY,
                True,
                False,
                {"is_success": False},
            )  # Invalid action penalty
        row = self._drop_piece(action, self.agent_symb)
        reward, done = self._check_game_over(row, action)
        if done:
            info = {}
            if reward != 0:
                info = {"is_success": True}
            else:
                info = {"is_success": False}  # No success for draw
            return self.grid.copy(), reward, True, False, info

        # Check if the agent's move blocks a potential win of the opponent
        self.grid[row, action] = (
            self.opp_symb
        )  # see if the oppponents move would have resulted in a win
        x_reward, x_done = self._check_game_over(row, action)  # extra reward
        self.grid[row, action] = self.agent_symb  # restore to actual state

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
        self.grid = np.zeros((6, 7), dtype=np.int8)
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
            """Render the current grid state using pygame."""
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.win_width, self.win_height))
                pygame.display.set_caption("Connect 4")

            self.screen.fill((0, 0, 255))  # Blue background for the board

            for row in range(6):
                for col in range(7):
                    # Draw each cell
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
        for row in range(5, -1, -1):  # Find the lowest empty row
            if self.grid[row, action] == 0:
                self.grid[row, action] = player
                return row

    def _is_valid_action(self, action):
        return self.grid[0, action] == 0  # Column is not full

    def _get_random_action(self):
        valid = False
        while not valid:
            action = self.action_space.sample()
            valid = self._is_valid_action(action)
        return action

    def _get_opp_action(self):
        if not self.opp_agent:
            action = self._get_random_action()
            return action
        else:
            valid = False
            if np.random.random() < self.opp_epsilon:
                while not valid:
                    action = self._get_random_action()
                    valid = self._is_valid_action(action)
            else:
                while not valid:
                    action = self.opp_agent.predict(self.grid, deterministic=False)[0]
                    valid = self._is_valid_action(action)
            return action

    def _check_game_over(self, row, col):
        # Check for a draw
        if np.all(self.grid != 0):
            return DRAW_REWARD, True  # Draw

        player_symb = self.grid[row, col]

        # horizontal
        for i in range(4):
            start_row, start_col = row, col - i
            if start_col >= 0 and start_col + 3 < 7:
                if np.all(
                    self.grid[start_row, start_col : start_col + 4] == player_symb
                ):
                    return WIN_REWARD, True

        # top-left to bottom-right diagonal
        for i in range(4):
            start_row, start_col = row - i, col - i
            if (
                start_row >= 0
                and start_col >= 0
                and start_row + 3 < 6
                and start_col + 3 < 7
            ):
                if np.all(
                    self.grid[
                        start_row : start_row + 4, start_col : start_col + 4
                    ].diagonal()
                    == player_symb
                ):
                    return WIN_REWARD, True

        # bottom-left to top-right diagonal
        for i in range(4):
            start_row, start_col = row + i, col - i
            if (
                start_row < 6
                and start_col >= 0
                and start_row - 3 >= 0
                and start_col + 3 < 7
            ):
                if np.all(
                    np.fliplr(
                        self.grid[
                            start_row - 3 : start_row + 1, start_col : start_col + 4
                        ]
                    ).diagonal()
                    == player_symb
                ):
                    return WIN_REWARD, True

        # vertical
        for i in range(4):
            start_row, start_col = row - i, col
            if start_row >= 0 and start_row + 3 < 6:
                if np.all(
                    self.grid[start_row : start_row + 4, start_col] == player_symb
                ):
                    return WIN_REWARD, True

        return CONTINUE_REWARD, False

    def set_opp_agent(self, agent, epsilon=0.1):
        self.opp_agent = agent
        self.opp_epsilon = epsilon

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    # Wrap the environment with RecordEpisodeStatistics

    # log_path = "./a2c_log"
    # new_logger = configure(log_path, ["stdout", "tensorboard"])

    # agent_self_play = PPO("MlpPolicy", env, verbose=1)
    # agent_self_play.learn(total_timesteps=100_000, progress_bar=True)
    # env = agent_self_play.get_env()
    # env.env_method("set_opp_agent", agent_self_play)
    # agent_self_play.learn(total_timesteps=1_000_000, progress_bar=True)
    # agent_self_play.save("agent_self_play")

    env = Connect4()
    env.reset()

    agent0 = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/curriculum")
    agent0.learn(total_timesteps=100000, progress_bar=True, tb_log_name="PP0_agent0")
    agent0.save("./saved_agents/agent0")
    for i in range(1, 16):
        agent_i = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/curriculum")
        agent_im1 = PPO.load("./saved_agents/" + f"agent{i-1}")
        vec_env = agent_i.get_env()
        vec_env.env_method("set_opp_agent", agent_im1, 0.2)
        agent_i.learn(
            total_timesteps=100000, progress_bar=True, tb_log_name=f"PP0_agent{i}"
        )
        print(f"Finished training agent{i}.")
        agent_i.save("./saved_agents/" + f"agent{i}")

    # heuristic_agent = HeuristicAgent()
    # env = Monitor(Connect4(), info_keywords=("is_success",))
    #
    # agent = PPO("MlpPolicy", env, verbose=1)
    # agent.learn(total_timesteps=100000, progress_bar=True)
    #
    # env = Connect4()
    # env.set_opp_agent(heuristic_agent, 0)
    # vec_env = agent.get_env()
    # vec_env.env_method("set_opp_agent", heuristic_agent, 0)

    # env.render(mode="stdout")
    # while True:
    #     symb1, symb2 = env.reset()[1]["symbols"]
    #     env.render(mode="stdout")
    #     print("-----------------------------", symb1, symb2)
    #     done = False
    #     while not done:
    #         action = agent.predict(env.grid, deterministic=True)[0]
    #         _, reward, done, _, _ = env.step(action)
    #         env.render(mode="stdout")
    #         time.sleep(0.5)
    #     env.render(mode="stdout")

    # agent.learn(total_timesteps=1000000, progress_bar=True)
    # agent.save("new_agent")

    # env = Connect4()
    # env.reset()
    # agent = DQN("MlpPolicy", env, verbose=1,tensorboard_log="./logs/heuristic_agent_logs")
    # agent.learn(total_timesteps=100000, progress_bar=True)
    # vec_env = agent.get_env()
    # heuristic_agent = HeuristicAgent(c=0)
    # vec_env.env_method("set_opp_agent", heuristic_agent)
    # agent.learn(total_timesteps=1000000, progress_bar=True,tb_log_name='DQN_vs_Heuristic')
    # agent.save('./saved_agents/DQN_vs_Heuristic')
    # agent_adv = PPO("MlpPolicy", env, verbose=1)
    # agent_adv.learn(total_timesteps=100000, progress_bar=True)

    # vec_env = agent.get_env()
    # vec_env.env_method("set_opp_agent", agent_adv, 0.1)
    # vec_env_adv = agent_adv.get_env()
    # vec_env_adv.env_method("set_opp_agent", agent, 0.1)

    # for i in tqdm(range(500_000)):
    #     agent_adv.learn(total_timesteps=1000)
    #     agent.learn(total_timesteps=1000)

    # agent.save("adv_agent")
    # agent_adv.save("adv_agent1")

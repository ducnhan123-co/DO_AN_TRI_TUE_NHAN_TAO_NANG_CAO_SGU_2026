from connect4 import Connect4
from stable_baselines3 import DQN, A2C, PPO
from heuristic import HeuristicAgent
from minimax import MinimaxAgent
from ensemble import EnsembleAgent
from tqdm import tqdm
import matplotlib.pyplot as plt

WIN_REWARD = 10
BLOCK_REWARD = 3
DRAW_REWARD = 1
CONTINUE_REWARD = 0
ILLEGAL_MOVE_PENALTY = -100


def test(player, opponent=None, mode=None):
    env = Connect4()
    if opponent:
        env.set_opp_agent(opponent, 0)

    done = False
    env.reset()
    while not done:
        action = player.predict(env.grid)[0]
        _, reward, done, _, _ = env.step(action)

    env.close()
    if reward == WIN_REWARD:
        return True
    elif reward == DRAW_REWARD:
        return False
    else:
        return False


players = [PPO.load(f"./saved_agents/agent{i}") for i in range(15)]
ensemble_player = EnsembleAgent(players)
NUM_GAMES = 10000
wins = 0
for i in tqdm(range(NUM_GAMES)):
    wins += test(ensemble_player)
winrate = wins / NUM_GAMES
print(winrate)

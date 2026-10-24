from connect4 import Connect4
import pygame
from stable_baselines3 import DQN, A2C, PPO
from heuristic import HeuristicAgent
from minimax import MinimaxAgent
import time

WIN_REWARD = 10
BLOCK_REWARD = 3
DRAW_REWARD = 1
CONTINUE_REWARD = 0
ILLEGAL_MOVE_PENALTY = -10


def get_col():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                posx = event.pos[0]
                col = posx // 100  # Cell width = 100
                return col


def play(opponent, player=None):
    if not player:
        mode = "human"
    else:
        mode = "stdout"

    env = Connect4()
    env.set_opp_agent(opponent, 0)
    env.render(mode=mode)
    symb1, symb2 = env.reset()[1]["symbols"]

    done = False
    while not done:
        env.render(mode=mode)
        if not player:
            action = get_col()
        else:
            action = player.predict(env.grid)[0]
        _, reward, done, _, _ = env.step(action)

        if done:
            env.render(mode=mode)
            if reward == WIN_REWARD:
                pygame.display.set_caption("Player wins!")
            elif reward == DRAW_REWARD:
                pygame.display.set_caption("Draw!")
            else:
                pygame.display.set_caption("Opponent wins!")    
            time.sleep(2)

    env.render(mode="stdout")

    print("Player's symbol:", symb1, "\nAI's symbol:", symb2)
    if reward == WIN_REWARD:
        print("You win!")
    elif reward == DRAW_REWARD:
        print("Draw!")
    else:
        print("You lose!")
    env.close()


# player = MinimaxAgent(1, 2)
# agent = PPO.load("./saved_agents/PPO_vs_Heuristic")

# agent = DQN.load("adv_agent1")

# play(agent)

##play against Random agent
# play(opponent=None)

#play against Heurestic agent
play(opponent=HeuristicAgent())
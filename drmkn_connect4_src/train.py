from connect4 import Connect4
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
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


def train(
    agent_algo="PPO",
    total_timesteps=100_000,
    opp_path=None,
    opp_epsilon=0.1,
    opp_algo="PPO",
    tb_dir=None,
    tb_log_name=None,
):
    algos = {"PPO": PPO, "A2C": A2C, "DQN": DQN}

    env = Connect4()
    agent = algos[agent_algo]("MlpPolicy", env, verbose=1, tensorboard_log=tb_dir)
    if opp_path:
        opponent = algos[opp_algo].load("opp_path")
        plug_opp(agent, opponent, opp_epsilon)
    env.reset()
    agent.learn(
        total_timesteps=total_timesteps, progress_bar=True, tb_log_name=tb_log_name
    )
    return agent


def plug_opp(agent, opponent, opp_epsilon=0.1):
    vec_env = agent.get_env()
    vec_env.env_method("set_opp_agent", opponent, opp_epsilon)


if __name__ == "__main__":

    env = Connect4()
    agent = PPO("MlpPolicy", env, verbose=1)
    agent = train("PPO", 100_000)
    env = agent.get_env()

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

import os
from connect4 import Connect4
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from ensemble import EnsembleAgent
from heuristic import HeuristicAgent
from tqdm import tqdm


def train(
    rows=8,
    cols=9,
    agent_algo="PPO",
    total_timesteps=100_000,
    opp_path=None,
    opp_epsilon=0.1,
    opp_algo="PPO",
    tb_dir=None,
    tb_log_name=None,
):
    algos = {"PPO": PPO, "A2C": A2C, "DQN": DQN}

    env = Connect4(rows=rows, cols=cols)
    agent = algos[agent_algo]("MlpPolicy", env, verbose=1, tensorboard_log=tb_dir)
    if opp_path:
        opponent = algos[opp_algo].load(opp_path)
        plug_opp(agent, opponent, opp_epsilon)
    env.reset()
    agent.learn(
        total_timesteps=total_timesteps, progress_bar=True, tb_log_name=tb_log_name
    )
    return agent


def plug_opp(agent, opponent, opp_epsilon=0.1):
    vec_env = agent.get_env()
    vec_env.env_method("set_opp_agent", opponent, opp_epsilon)


def train_curriculum_8x9(total_timesteps_per_agent=100_000, num_agents=16):
    """
    Huấn luyện chuỗi Curriculum Agent cho bàn cờ 8x9:
    Lưu vào thư mục: ./saved_agents_8x9/
    """
    save_dir = "./saved_agents_8x9"
    os.makedirs(save_dir, exist_ok=True)
    tb_log_dir = "./logs/curriculum_8x9"
    os.makedirs(tb_log_dir, exist_ok=True)

    env = Connect4(rows=8, cols=9)
    env.reset()

    print("--- [8x9] Bắt đầu huấn luyện agent0 (đối đầu random) ---")
    agent0 = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tb_log_dir)
    agent0.learn(total_timesteps=total_timesteps_per_agent, progress_bar=True, tb_log_name="PPO_agent0_8x9")
    agent0.save(f"{save_dir}/agent0")
    print("--- Hoàn thành lưu agent0 8x9! ---")

    for i in range(1, num_agents):
        print(f"\n--- [8x9] Bắt đầu huấn luyện agent{i} (đối đầu agent{i-1}) ---")
        agent_i = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tb_log_dir)
        agent_prev = PPO.load(f"{save_dir}/agent{i-1}")
        vec_env = agent_i.get_env()
        vec_env.env_method("set_opp_agent", agent_prev, 0.2)
        agent_i.learn(
            total_timesteps=total_timesteps_per_agent,
            progress_bar=True,
            tb_log_name=f"PPO_agent{i}_8x9",
        )
        agent_i.save(f"{save_dir}/agent{i}")
        print(f"--- Đã hoàn thành lưu agent{i} 8x9! ---")


if __name__ == "__main__":
    # Để bắt đầu huấn luyện AI cho bàn cờ 8x9, bỏ comment dòng dưới:
    # train_curriculum_8x9(total_timesteps_per_agent=100_000, num_agents=16)
    print("Mở file train.py và gọi train_curriculum_8x9() để bắt đầu train cho bàn 8x9.")

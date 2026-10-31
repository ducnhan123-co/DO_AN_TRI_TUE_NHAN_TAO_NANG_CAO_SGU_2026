import os
import numpy as np
from stable_baselines3 import PPO
from connect4 import Connect4
from heuristic import HeuristicAgent


def train_rl_ppo(
    rows=8,
    cols=9,
    total_timesteps=10000,
    save_path="saved_agents_8x9/level6_rl_ppo",
):
    """
    Huấn luyện Cấp 6: Học Tăng Cường (Reinforcement Learning - PPO):
    Huấn luyện qua 10.000 timesteps tương tác trực tiếp với môi trường Connect 4.
    """
    print(f"\n=======================================================")
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN CẤP 6: HỌC TĂNG CƯỜNG (REINFORCEMENT LEARNING - PPO)")
    print(f"Môi trường: Connect4 ({rows}x{cols}) | Số timesteps: {total_timesteps}")
    print(f"=======================================================\n")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tb_dir = "./logs/rl_ppo_8x9"
    os.makedirs(tb_dir, exist_ok=True)

    env = Connect4(rows=rows, cols=cols)
    # Cho đối đầu với Heuristic agent với tỉ lệ khám phá để học đối kháng thực tế
    env.set_opp_agent(HeuristicAgent(c=0), epsilon=0.15)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=tb_dir,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="PPO_Level6")
    model.save(save_path)
    print(f"\n[✓] Đã lưu mô hình Học Tăng Cường PPO tại: {save_path}.zip\n")
    return model


class RLAgent:
    """
    Agent Cấp 6: Reinforcement Learning (PPO Agent).
    """
    def __init__(self, model_path="saved_agents_8x9/level6_rl_ppo", rows=8, cols=9):
        self.rows = rows
        self.cols = cols
        self.model = None

        clean_path = model_path[:-4] if model_path.endswith(".zip") else model_path
        if os.path.exists(clean_path + ".zip") or os.path.exists(clean_path):
            try:
                self.model = PPO.load(clean_path)
            except Exception as e:
                print(f"[Warning] Khong the nap RL model {model_path}: {e}")

        # Fallback Heuristic if model is not trained yet
        self.fallback = HeuristicAgent(c=0)

    def predict(self, obs, deterministic=True):
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            board = obs[0] if obs.shape[0] == 2 else obs
        else:
            board = obs

        valid_cols = [c for c in range(self.cols) if board[0, c] == 0]
        if not valid_cols:
            return (0,)

        if self.model is not None:
            try:
                action, _ = self.model.predict(board, deterministic=deterministic)
                action = int(action) if not isinstance(action, (np.ndarray, list)) else int(action[0])
                if action in valid_cols:
                    return (action,)
            except Exception:
                pass

        return self.fallback.predict(board, deterministic=deterministic)

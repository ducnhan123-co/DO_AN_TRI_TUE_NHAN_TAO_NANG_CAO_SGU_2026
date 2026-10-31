import os
import sys
import copy
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from connect4 import Connect4
from heuristic import HeuristicAgent

# Sửa lỗi hiển thị Unicode trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =====================================================================
# 1. KIẾN TRÚC MẠNG TÍCH CHẬP 2D CHUYÊN DỤNG (CNN FEATURE EXTRACTOR)
# =====================================================================
class Connect4CNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Mạng nơ-ron tích chập (CNN) mô phỏng kiến trúc của AlphaZero:
    - Biến đổi bàn cờ 8x9 thành 3 kênh One-Hot: [Ô trống, Quân ta (1), Quân địch (2)].
    - Sử dụng các bộ lọc tích chập (Conv2D) kích thước 3x3 và 4x4 để tự động học
      các mẫu chuỗi 4 quân theo hàng ngang, dọc và 2 đường chéo.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        rows, cols = observation_space.shape

        # 3 channels: Trống, Quân 1, Quân 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        flattened_size = 128 * rows * cols
        self.linear = nn.Sequential(
            nn.Linear(flattened_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (Batch, Rows=8, Cols=9)
        obs_long = observations.long()
        # Chuyển thành One-Hot 3 kênh: (Batch, 8, 9, 3) -> permute -> (Batch, 3, 8, 9)
        one_hot = th.nn.functional.one_hot(obs_long, num_classes=3).permute(0, 3, 1, 2).float()
        return self.linear(self.cnn(one_hot))


# =====================================================================
# 2. MÔI TRƯỜNG TỰ CHƠI (SELF-PLAY LEAGUE ENVIRONMENT)
# =====================================================================
class SelfPlayConnect4Env(Connect4):
    """
    Môi trường mở rộng cho phép đối thủ được bốc ngẫu nhiên từ kho Agent lịch sử (Opponent Pool).
    Agent đang học sẽ phải đối đầu với:
    - 50%: Các phiên bản quá khứ của chính nó (để chống quên bài).
    - 30%: Phiên bản mới nhất của nó (để cạnh tranh tiến hóa).
    - 20%: Heuristic Agent (để duy trì kỷ luật chặn 3 và bắt 4).
    """
    def __init__(self, rows=8, cols=9):
        super().__init__(rows=rows, cols=cols)
        self.opp_epsilon = 0.05
        self.opponent_pool = []
        self.heuristic_agent = HeuristicAgent(c=0)

    def add_opponent_to_pool(self, agent):
        self.opponent_pool.append(agent)

    def reset(self, *, seed=None, options=None):
        # Chọn đối thủ ngẫu nhiên cho ván đấu này
        if not self.opponent_pool or np.random.random() < 0.20:
            self.opp_agent = self.heuristic_agent
        elif np.random.random() < 0.35 and len(self.opponent_pool) > 1:
            self.opp_agent = self.opponent_pool[-1]  # Phiên bản gần nhất
        else:
            self.opp_agent = np.random.choice(self.opponent_pool)  # Ngẫu nhiên trong quá khứ

        return super().reset(seed=seed, options=options)


# =====================================================================
# 3. HÀM HUẤN LUYỆN SELF-PLAY NÂNG CAO
# =====================================================================
def train_self_play(
    rows=8,
    cols=9,
    num_generations=10,
    timesteps_per_gen=50_000,
    save_dir="./saved_agents_selfplay_8x9",
):
    """
    Quy trình huấn luyện Self-Play tiến hóa qua nhiều thế hệ (Generations):
    - Thế hệ 0: Học với Heuristic & Random Opponent.
    - Sau mỗi thế hệ: Lưu model vào Opponent Pool làm đối thủ cho thế hệ tiếp theo.
    """
    os.makedirs(save_dir, exist_ok=True)
    tb_log_dir = "./logs/selfplay_8x9"
    os.makedirs(tb_log_dir, exist_ok=True)

    env = SelfPlayConnect4Env(rows=rows, cols=cols)
    policy_kwargs = dict(
        features_extractor_class=Connect4CNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128], vf=[128]),
    )

    print(f"\n=======================================================")
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN SELF-PLAY TỰ CHƠI (BÀN {rows}x{cols})")
    print(f"Kiến trúc: Deep Conv2D (AlphaZero Style) + Historical League")
    print(f"Số thế hệ: {num_generations} | Mỗi thế hệ: {timesteps_per_gen} steps")
    print(f"=======================================================\n")

    # Khởi tạo mô hình thế hệ 0
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=tb_log_dir,
    )

    for gen in range(num_generations):
        print(f"\n>>> [Thế hệ {gen + 1}/{num_generations}] Đang huấn luyện đối đầu với kho đối thủ ({len(env.opponent_pool)} agents trong Pool)...")
        model.learn(total_timesteps=timesteps_per_gen, progress_bar=True, tb_log_name=f"SelfPlay_Gen{gen}")
        
        # Lưu checkpoint thế hệ này
        save_path = os.path.join(save_dir, f"gen_{gen}")
        model.save(save_path)
        print(f"[✓] Đã lưu mô hình thế hệ {gen + 1} tại: {save_path}.zip")

        # Nạp lại bản sao đã lưu vào Opponent Pool của môi trường để làm đối thủ tiếp theo
        frozen_agent = PPO.load(save_path)
        env.add_opponent_to_pool(frozen_agent)

    # Lưu bản cuối cùng mạnh nhất
    final_path = os.path.join(save_dir, "best_selfplay_agent")
    model.save(final_path)
    print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN SELF-PLAY! Mô hình tối ưu được lưu tại: {final_path}.zip\n")
    return model


if __name__ == "__main__":
    # Test chạy nhanh thử 2 thế hệ
    train_self_play(rows=8, cols=9, num_generations=2, timesteps_per_gen=10_000)


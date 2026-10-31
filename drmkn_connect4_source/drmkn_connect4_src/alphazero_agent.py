import os
import math
import random
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from connect4 import Connect4
from heuristic import HeuristicAgent


class AlphaZeroCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Trích xuất đặc trưng không gian bàn cờ mô phỏng AlphaZero:
    - Đầu vào: 3 kênh One-Hot [Ô trống, Quân ta, Quân địch].
    - Conv2D trích xuất chuỗi 4 quân theo mọi hướng (ngang, dọc, chéo).
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        rows, cols = observation_space.shape

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
        obs_long = observations.long()
        one_hot = th.nn.functional.one_hot(obs_long, num_classes=3).permute(0, 3, 1, 2).float()
        return self.linear(self.cnn(one_hot))


def train_alphazero_level7(
    rows=8,
    cols=9,
    total_timesteps=10000,
    save_path="saved_agents_8x9/level7_alphazero",
):
    """
    Huấn luyện Cấp 7: AlphaZero Foundation (CNN Policy-Value Network).
    Huấn luyện 10.000 timesteps với mạng tích chập AlphaZero trên bàn 8x9.
    """
    print(f"\n=======================================================")
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN CẤP 7: ALPHAZERO FOUNDATION (DEEP CNN)")
    print(f"Kiến trúc: Conv2D 3 kênh + Policy-Value Network")
    print(f"Số timesteps: {total_timesteps}")
    print(f"=======================================================\n")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tb_dir = "./logs/alphazero_level7"
    os.makedirs(tb_dir, exist_ok=True)

    env = Connect4(rows=rows, cols=cols)
    env.set_opp_agent(HeuristicAgent(c=0), epsilon=0.1)

    policy_kwargs = dict(
        features_extractor_class=AlphaZeroCNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128], vf=[128]),
    )

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
        tensorboard_log=tb_dir,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="AlphaZero_L7")
    model.save(save_path)
    print(f"\n[✓] Đã lưu mô hình AlphaZero Cấp 7 tại: {save_path}.zip\n")
    return model


class AlphaZeroPUCTNode:
    """Nút tìm kiếm PUCT cho AlphaZero MCTS."""
    def __init__(self, prior=1.0):
        self.prior = prior
        self.visits = 0
        self.total_value = 0.0

    @property
    def q_value(self):
        return self.total_value / self.visits if self.visits > 0 else 0.0


class AlphaZeroAgent:
    """
    Agent Cấp 7: AlphaZero Agent kết hợp CNN Policy-Value với PUCT MCTS Search.
    """
    def __init__(self, model_path="saved_agents_8x9/level7_alphazero", rows=8, cols=9, puct_sims=30, c_puct=2.0):
        self.rows = rows
        self.cols = cols
        self.puct_sims = puct_sims
        self.c_puct = c_puct
        self.model = None

        clean_path = model_path[:-4] if model_path.endswith(".zip") else model_path
        if os.path.exists(clean_path + ".zip") or os.path.exists(clean_path):
            try:
                self.model = PPO.load(clean_path)
            except Exception as e:
                print(f"[Warning] Khong the nap AlphaZero model {model_path}: {e}")

        self.fallback = HeuristicAgent(c=0)

    def _determine_turn(self, grid):
        count_1 = np.sum(grid == 1)
        count_2 = np.sum(grid == 2)
        return 1 if count_1 == count_2 else 2

    def _drop_piece(self, board, col, piece):
        for r in range(self.rows - 1, -1, -1):
            if board[r, col] == 0:
                board[r, col] = piece
                return r
        return None

    def _check_winner(self, board, piece):
        rows, cols = board.shape
        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if all(board[r, c : c + 4] == piece):
                    return True
        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if all(board[r : r + 4, c] == piece):
                    return True
        # Diagonal \
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(board[r + i, c + i] == piece for i in range(4)):
                    return True
        # Diagonal /
        for r in range(3, rows):
            for c in range(cols - 3):
                if all(board[r - i, c + i] == piece for i in range(4)):
                    return True
        return False

    def predict(self, obs, deterministic=True):
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            board = obs[0] if obs.shape[0] == 2 else obs
        else:
            board = obs

        valid_cols = [c for c in range(self.cols) if board[0, c] == 0]
        if not valid_cols:
            return (0,)
        if len(valid_cols) == 1:
            return (valid_cols[0],)

        my_symb = self._determine_turn(board)
        opp_symb = 2 if my_symb == 1 else 1

        # 1. Taktik: Nước thắng ngay
        for c in valid_cols:
            b_c = board.copy()
            self._drop_piece(b_c, c, my_symb)
            if self._check_winner(b_c, my_symb):
                return (c,)

        # 2. Taktik: Nước chặn thua ngay
        for c in valid_cols:
            b_c = board.copy()
            self._drop_piece(b_c, c, opp_symb)
            if self._check_winner(b_c, opp_symb):
                return (c,)

        if self.model is None:
            return self.fallback.predict(board, deterministic=deterministic)

        try:
            # Lấy xác suất tiên nghiệm (Priors) từ Policy Network
            obs_tensor = th.as_tensor(board).unsqueeze(0)
            if hasattr(self.model.policy, "get_distribution"):
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]
            else:
                act = self.model.predict(board, deterministic=True)[0]
                return (int(act) if int(act) in valid_cols else valid_cols[0],)

            # Khởi tạo các nút con theo công thức AlphaZero PUCT
            nodes = {c: AlphaZeroPUCTNode(prior=max(1e-4, float(probs[c]))) for c in valid_cols}

            for _ in range(self.puct_sims):
                total_visits = sum(n.visits for n in nodes.values()) + 1
                best_action = None
                best_puct = -float("inf")

                for c, n in nodes.items():
                    u = self.c_puct * n.prior * (math.sqrt(total_visits) / (1 + n.visits))
                    puct_score = n.q_value + u
                    if puct_score > best_puct:
                        best_puct = puct_score
                        best_action = c

                # Giả lập 1 bước đi và đánh giá giá trị bàn cờ bằng Value Network
                b_sim = board.copy()
                self._drop_piece(b_sim, best_action, my_symb)

                if self._check_winner(b_sim, my_symb):
                    val = 1.0
                else:
                    # Lấy giá trị ước lượng value từ mạng nơ-ron
                    with th.no_grad():
                        val_tensor = self.model.policy.predict_values(th.as_tensor(b_sim).unsqueeze(0))
                        val = float(th.tanh(val_tensor / 5.0).item())

                nodes[best_action].visits += 1
                nodes[best_action].total_value += val

            # Chọn nước đi có số lần thăm nhiều nhất (MCTS visit count)
            selected_col = max(nodes.keys(), key=lambda c: nodes[c].visits)
            return (selected_col,)

        except Exception as e:
            # Fallback an toàn
            action, _ = self.model.predict(board, deterministic=deterministic)
            action = int(action) if not isinstance(action, (np.ndarray, list)) else int(action[0])
            if action in valid_cols:
                return (action,)
            return self.fallback.predict(board)

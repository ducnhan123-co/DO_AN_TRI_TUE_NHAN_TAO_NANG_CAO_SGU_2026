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
    num_generations=2,
    timesteps_per_gen=5000,
    save_dir="./saved_agents_selfplay_8x9",
    main_save_path="./saved_agents_8x9/level8_alphazero_master",
):
    """
    Quy trình huấn luyện Self-Play tiến hóa qua nhiều thế hệ (Generations):
    - Thế hệ 0: Học với Heuristic & Random Opponent.
    - Sau mỗi thế hệ: Lưu model vào Opponent Pool làm đối thủ cho thế hệ tiếp theo.
    - Lưu mô hình tốt nhất vào cả thư mục self-play và saved_agents_8x9.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(main_save_path) or ".", exist_ok=True)
    tb_log_dir = "./logs/selfplay_8x9"
    os.makedirs(tb_log_dir, exist_ok=True)

    env = SelfPlayConnect4Env(rows=rows, cols=cols)
    policy_kwargs = dict(
        features_extractor_class=Connect4CNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128], vf=[128]),
    )

    print(f"\n=======================================================")
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN CẤP 8: ALPHAZERO MASTER (SELF-PLAY LEAGUE)")
    print(f"Kiến trúc: Deep Conv2D + Historical Opponent League")
    print(f"Số thế hệ: {num_generations} | Mỗi thế hệ: {timesteps_per_gen} steps | Tổng: {num_generations * timesteps_per_gen} steps")
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
    model.save(main_save_path)
    print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN SELF-PLAY CẤP 8!")
    print(f"[✓] Lưu tại: {final_path}.zip")
    print(f"[✓] Lưu bản chuẩn Level 8 tại: {main_save_path}.zip\n")
    return model


class AlphaZeroMasterAgent:
    """
    Agent Cấp 8: AlphaZero Master Agent
    Kết hợp mô hình Deep CNN Self-Play League với bộ phân tích chiến thuật nhìn trước:
    - Bắt buộc thắng ngay nếu có cơ hội.
    - Chặn ngay nước thắng tức thì của đối thủ.
    - Phát hiện & vô hiệu hóa các thế cờ bẫy 2 đầu (Forks).
    - Lọc bỏ các nước đi tự sát (nước đi giúp đối thủ thả đè lên để thắng).
    - Tích hợp hệ thống ghi nhớ bẫy trực tuyến (Online Adaptive Learning).
    """
    def __init__(self, model_path="saved_agents_8x9/level8_alphazero_master", rows=8, cols=9, learner=None):
        self.rows = rows
        self.cols = cols
        self.learner = learner
        self.model = None

        clean_path = model_path[:-4] if model_path.endswith(".zip") else model_path
        if os.path.exists(clean_path + ".zip") or os.path.exists(clean_path):
            try:
                self.model = PPO.load(clean_path)
            except Exception as e:
                print(f"[Warning] Khong the nap Level 8 model {model_path}: {e}")

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

    def _is_suicide_move(self, board, col, my_symb, opp_symb):
        """Kiểm tra nếu thả vào cột này thì đối thủ thả ngay lên trên có thắng không."""
        b_c = board.copy()
        row = self._drop_piece(b_c, col, my_symb)
        if row is None or row == 0:
            return False
        b_c[row - 1, col] = opp_symb
        return self._check_winner(b_c, opp_symb)

    def _find_fork_moves(self, board, player_symb):
        """Tìm các cột tạo ra từ 2 cửa thắng trở lên."""
        fork_cols = []
        valid_cols = [c for c in range(self.cols) if board[0, c] == 0]
        for col in valid_cols:
            b_c = board.copy()
            self._drop_piece(b_c, col, player_symb)
            winning_threats = 0
            remaining_cols = [c for c in range(self.cols) if b_c[0, c] == 0]
            for next_col in remaining_cols:
                b_next = b_c.copy()
                self._drop_piece(b_next, next_col, player_symb)
                if self._check_winner(b_next, player_symb):
                    winning_threats += 1
            if winning_threats >= 2:
                fork_cols.append(col)
        return fork_cols

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

        # 0. Học thích ứng trực tuyến
        if self.learner:
            adapted_col = self.learner.get_adapted_move(board, valid_cols)
            if adapted_col is not None:
                return (adapted_col,)

        # 1. Thắng ngay
        for c in valid_cols:
            b_c = board.copy()
            self._drop_piece(b_c, c, my_symb)
            if self._check_winner(b_c, my_symb):
                return (c,)

        # 2. Chặn thua ngay
        for c in valid_cols:
            b_c = board.copy()
            self._drop_piece(b_c, c, opp_symb)
            if self._check_winner(b_c, opp_symb):
                return (c,)

        # 3. Chặn bẫy 2 đầu (Fork) của đối thủ
        opp_forks = self._find_fork_moves(board, opp_symb)
        safe_opp_forks = [c for c in opp_forks if not self._is_suicide_move(board, c, my_symb, opp_symb)]
        if safe_opp_forks:
            return (safe_opp_forks[0],)

        # 4. Tạo bẫy 2 đầu cho mình
        my_forks = self._find_fork_moves(board, my_symb)
        safe_my_forks = [c for c in my_forks if not self._is_suicide_move(board, c, my_symb, opp_symb)]
        if safe_my_forks:
            return (safe_my_forks[0],)

        # 5. Lọc bỏ nước tự sát
        safe_moves = [c for c in valid_cols if not self._is_suicide_move(board, c, my_symb, opp_symb)]
        candidate_moves = safe_moves if safe_moves else valid_cols

        # 6. Dự đoán bằng mô hình AlphaZero Master
        if self.model is not None:
            try:
                action, _ = self.model.predict(board, deterministic=deterministic)
                action = int(action) if not isinstance(action, (np.ndarray, list)) else int(action[0])
                if action in candidate_moves:
                    return (action,)
            except Exception:
                pass

        # 7. Fallback Heuristic
        heur_act = self.fallback.predict(board)[0]
        if heur_act in candidate_moves:
            return (heur_act,)

        return (candidate_moves[0],)


if __name__ == "__main__":
    train_self_play(rows=8, cols=9, num_generations=2, timesteps_per_gen=5000)



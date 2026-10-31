import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from heuristic import HeuristicAgent
from minimax import MinimaxAgent


class Connect4SupervisedNet(nn.Module):
    """
    Mạng nơ-ron tích chập (CNN) cho Học Có Giám Sát (Cấp 5):
    - Nhận vào bàn cờ 3 kênh kích thước 8x9: [Ô trống, Quân mình, Quân đối thủ].
    - Xuất ra 9 giá trị logits tương ứng xác suất chọn 9 cột.
    """
    def __init__(self, rows=8, cols=9):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        flattened_dim = 128 * rows * cols
        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, cols),
        )

    def forward(self, x):
        features = self.conv_net(x)
        logits = self.fc(features)
        return logits


def state_to_tensor(board, player_symb):
    """
    Biến đổi bàn cờ 8x9 thành tensor 3D: shape (3, 8, 9).
    Channel 0: Ô trống (0)
    Channel 1: Quân của mình (player_symb)
    Channel 2: Quân của đối phương (opp_symb)
    """
    opp_symb = 2 if player_symb == 1 else 1
    ch_empty = (board == 0).astype(np.float32)
    ch_me = (board == player_symb).astype(np.float32)
    ch_opp = (board == opp_symb).astype(np.float32)
    return np.stack([ch_empty, ch_me, ch_opp], axis=0)


def generate_expert_dataset(target_samples=3000, rows=8, cols=9):
    """
    Sinh tập dữ liệu mẫu từ các chuyên gia (Heuristic và Minimax Agent).
    """
    states = []
    actions = []
    expert_heuristic = HeuristicAgent(c=0)
    expert_minimax = MinimaxAgent(depth=2)

    print(f"[*] Đang sinh tập dữ liệu chuyên gia Connect 4 (Mục tiêu: {target_samples} mẫu)...")
    
    while len(states) < target_samples:
        board = np.zeros((rows, cols), dtype=np.int8)
        current_player = 1
        done = False
        step_count = 0

        while not done and step_count < rows * cols:
            valid_cols = [c for c in range(cols) if board[0, c] == 0]
            if not valid_cols:
                break

            # Chọn chuyên gia ra quyết định nước đi
            if random.random() < 0.6:
                action = expert_heuristic.predict(board)[0]
            else:
                action = expert_minimax.predict(board, depth=2)[0]

            if action not in valid_cols:
                action = random.choice(valid_cols)

            # Lưu mẫu vào dataset
            tensor_obs = state_to_tensor(board, current_player)
            states.append(tensor_obs)
            actions.append(action)

            # Data augmentation: Lật gương ngang bàn cờ (Horizontal Flip)
            flipped_board = np.fliplr(board)
            flipped_action = (cols - 1) - action
            flipped_tensor = state_to_tensor(flipped_board, current_player)
            states.append(flipped_tensor)
            actions.append(flipped_action)

            # Thả quân để tiếp tục ván đấu
            for r in range(rows - 1, -1, -1):
                if board[r, action] == 0:
                    board[r, action] = current_player
                    break

            # Kiểm tra thắng
            if expert_minimax._winning_move(board, current_player):
                done = True
                break

            current_player = 2 if current_player == 1 else 1
            step_count += 1

    states = np.array(states[:target_samples], dtype=np.float32)
    actions = np.array(actions[:target_samples], dtype=np.int64)
    print(f"[✓] Đã tạo thành công {len(states)} mẫu dữ liệu huấn luyện (State-Action pairs)!")
    return states, actions


def train_supervised(
    rows=8,
    cols=9,
    total_steps=10000,
    batch_size=64,
    save_path="saved_agents_8x9/level5_supervised.pth",
):
    """
    Quy trình huấn luyện Học Có Giám Sát (Supervised Learning):
    Chạy đủ total_steps (10.000 bước tối ưu gradient descent).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN CẤP 5: HỌC CÓ GIÁM SÁT (SUPERVISED LEARNING)")
    print(f"Thiết bị: {device} | Tổng số bước tối ưu: {total_steps} steps")
    print(f"=======================================================\n")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    states, actions = generate_expert_dataset(target_samples=4000, rows=rows, cols=cols)

    dataset_size = len(states)
    model = Connect4SupervisedNet(rows=rows, cols=cols).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"[*] Bắt đầu vòng lặp tối ưu hóa {total_steps} bước...")
    running_loss = 0.0
    correct = 0
    total = 0

    for step in range(1, total_steps + 1):
        indices = np.random.randint(0, dataset_size, size=batch_size)
        batch_x = torch.from_numpy(states[indices]).to(device)
        batch_y = torch.from_numpy(actions[indices]).to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

        if step % 1000 == 0 or step == total_steps:
            avg_loss = running_loss / 1000 if step % 1000 == 0 else running_loss / (step % 1000 or 1)
            accuracy = 100.0 * correct / total
            print(f"  -> Bước [{step:5d}/{total_steps}] | Loss: {avg_loss:.4f} | Độ chính xác: {accuracy:.2f}%")
            running_loss = 0.0
            correct = 0
            total = 0

    torch.save(model.state_dict(), save_path)
    print(f"[✓] Đã lưu mô hình Học Có Giám Sát tại: {save_path}\n")
    return model


class SupervisedAgent:
    """
    Agent Học Có Giám Sát (Cấp 5):
    Dự đoán nước đi dựa trên mạng nơ-ron CNN đã học từ các thế cờ chuyên gia.
    """
    def __init__(self, model_path=None, rows=8, cols=9):
        self.rows = rows
        self.cols = cols
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Connect4SupervisedNet(rows=rows, cols=cols).to(self.device)

        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.loaded = True
        else:
            self.model.eval()
            self.loaded = False

    def _determine_turn(self, grid):
        count_1 = np.sum(grid == 1)
        count_2 = np.sum(grid == 2)
        return 1 if count_1 == count_2 else 2

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

        # Quyết định lượt
        player_symb = self._determine_turn(board)
        tensor_obs = state_to_tensor(board, player_symb)
        tensor_obs = torch.from_numpy(tensor_obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_obs).squeeze(0).cpu().numpy()

        # Mask các cột không hợp lệ (đã đầy)
        masked_logits = np.full(self.cols, -1e9)
        masked_logits[valid_cols] = logits[valid_cols]

        if deterministic:
            action = int(np.argmax(masked_logits))
        else:
            # Softmax sampling
            exp_l = np.exp(masked_logits - np.max(masked_logits))
            probs = exp_l / np.sum(exp_l)
            action = int(np.random.choice(range(self.cols), p=probs))

        return (action,)

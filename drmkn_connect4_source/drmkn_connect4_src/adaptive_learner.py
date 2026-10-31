import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO


def _check_winning_action(grid, piece, rows=8, cols=9):
    """Tìm tất cả các cột mà nếu thả quân `piece` vào sẽ tạo chuỗi 4 thắng ngay."""
    winning_cols = []
    for c in range(cols):
        # Tìm ô rơi xuống
        target_r = -1
        for r in range(rows - 1, -1, -1):
            if grid[r, c] == 0:
                target_r = r
                break
        if target_r != -1:
            grid[target_r, c] = piece
            if _is_winner(grid, target_r, c, piece, rows, cols):
                winning_cols.append(c)
            grid[target_r, c] = 0
    return winning_cols


def _is_winner(grid, r, c, piece, rows, cols):
    # Horizontal
    for sc in range(max(0, c - 3), min(cols, c + 4) - 3):
        if (grid[r, sc : sc + 4] == piece).all():
            return True
    # Vertical
    if r <= rows - 4:
        if (grid[r : r + 4, c] == piece).all():
            return True
    # Diagonal \
    for i in range(-3, 1):
        sr, sc = r + i, c + i
        if 0 <= sr <= rows - 4 and 0 <= sc <= cols - 4:
            if all(grid[sr + j, sc + j] == piece for j in range(4)):
                return True
    # Diagonal /
    for i in range(-3, 1):
        sr, sc = r - i, c + i
        if 3 <= sr < rows and 0 <= sc <= cols - 4:
            if all(grid[sr - j, sc + j] == piece for j in range(4)):
                return True
    return False


def _hash_board(grid):
    return str(grid.tolist())


class AdaptiveOnlineLearner:
    """
    Hệ thống Học Tăng Cường Thích Nghi Trực Tuyến (Online Continual / Adaptive RL):
    - Ghi nhận lịch sử nước đi của người chơi trong từng ván.
    - Khi người chơi thắng hoặc tạo bẫy:
      + Phân tích thế cờ dẫn đến thất bại của AI.
      + Áp dụng Policy Gradient Update (Fine-tuning tức thì) lên mạng nơ-ron PPO.
      + Tận dụng tính đối xứng lật ngang (Horizontal Symmetry) để nhân đôi hiệu quả học.
      + Lưu trữ lại thế cờ vào file bộ nhớ `adaptive_memory.json` và file weights.
    """

    def __init__(self, agent_model, model_save_path=None, memory_file=None):
        self.model = agent_model
        self.model_save_path = model_save_path
        self.rows = 8
        self.cols = 9

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if memory_file is None:
            self.memory_file = os.path.join(base_dir, "saved_agents_8x9", "adaptive_memory.json")
        else:
            self.memory_file = memory_file

        self.memory = self._load_memory()
        self.current_match_history = []
        self.total_learned_count = len(self.memory.get("learned_traps", {}))

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"learned_traps": {}, "total_games_played": 0, "player_wins": 0}

    def _save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Warning] Khong the ghi adaptive_memory.json: {e}")

    def reset_match(self):
        """Khởi động lại bộ nhớ ván đấu mới."""
        self.current_match_history = []

    def record_move(self, board_before, piece, action):
        """Ghi lại 1 nước đi vừa xảy ra."""
        self.current_match_history.append({
            "board": board_before.copy(),
            "piece": int(piece),
            "action": int(action)
        })

    def lookup_memory(self, grid):
        """Kiểm tra xem thế cờ này (hoặc thế cờ đối xứng) đã từng được học cách chặn chưa."""
        h1 = _hash_board(grid)
        traps = self.memory.get("learned_traps", {})
        if h1 in traps:
            return traps[h1]

        # Kiểm tra tính đối xứng lật ngang
        grid_flipped = np.fliplr(grid)
        h2 = _hash_board(grid_flipped)
        if h2 in traps:
            flipped_action = traps[h2]
            return self.cols - 1 - flipped_action

        return None

    def learn_from_match(self, winner, player_symb, ai_symb):
        """
        Học thích nghi ngay khi ván đấu kết thúc.
        """
        self.memory["total_games_played"] = self.memory.get("total_games_played", 0) + 1
        if winner == "player":
            self.memory["player_wins"] = self.memory.get("player_wins", 0) + 1

        if not self.current_match_history:
            return {"status": "no_moves"}

        learned_info = []

        # TRƯỜNG HỢP 1: NGƯỜI CHƠI THẮNG (AI THUA) -> AI HỌC CÁCH CHẶN NƯỚC BẪY ĐÓ
        if winner == "player":
            # Lần ngược từ nước cuối cùng mà người chơi đánh thắng
            last_human_move = None
            ai_move_before = None

            for i in range(len(self.current_match_history) - 1, -1, -1):
                m = self.current_match_history[i]
                if m["piece"] == player_symb and last_human_move is None:
                    last_human_move = m
                    # Tìm nước AI đánh ngay trước đó
                    if i > 0 and self.current_match_history[i - 1]["piece"] == ai_symb:
                        ai_move_before = self.current_match_history[i - 1]
                    break

            if last_human_move and ai_move_before:
                # Tại thời điểm AI đánh nước trước đó: lẽ ra AI nên đánh vào cột nào?
                board_ai_turn = ai_move_before["board"]
                winning_human_col = last_human_move["action"]

                # 1. Ghi nhớ thế cờ vào Memory để lần sau chặn ngay lập tức
                h = _hash_board(board_ai_turn)
                self.memory.setdefault("learned_traps", {})[h] = winning_human_col
                # Ghi nhớ cả biến thể đối xứng
                h_flip = _hash_board(np.fliplr(board_ai_turn))
                self.memory["learned_traps"][h_flip] = self.cols - 1 - winning_human_col

                self._save_memory()

                # 2. Cập nhật Policy Gradient tức thì cho Mạng Nơ-ron PPO
                loss_val = self._fine_tune_policy(
                    board=board_ai_turn,
                    correct_action=winning_human_col,
                    bad_action=ai_move_before["action"]
                )

                self.total_learned_count = len(self.memory.get("learned_traps", {}))
                learned_info.append(f"Đã học chặn cột {winning_human_col} (Loss: {loss_val:.3f})")

        # TRƯỜNG HỢP 2: AI THẮNG -> CỦNG CỐ CÁC NƯỚC ĐI TỐT CỦA AI (SELF-IMITATION)
        elif winner == "ai":
            # Củng cố nước đi kết liễu
            if self.current_match_history:
                last_ai_move = self.current_match_history[-1]
                if last_ai_move["piece"] == ai_symb:
                    self._fine_tune_policy(
                        board=last_ai_move["board"],
                        correct_action=last_ai_move["action"]
                    )
                    learned_info.append("Đã củng cố nước cờ kết liễu của AI")

        # Lưu lại checkpoint nếu có chỉ định path
        if self.model and self.model_save_path and winner == "player":
            try:
                self.model.save(self.model_save_path)
            except Exception:
                pass

        return {
            "status": "learned" if learned_info else "none",
            "info": learned_info,
            "total_learned": self.total_learned_count
        }

    def _fine_tune_policy(self, board, correct_action, bad_action=None, steps=3):
        """
        Thực hiện một vài bước lan truyền ngược (Gradient Update) siêu nhanh trên PyTorch:
        - Tăng xác suất chọn `correct_action`.
        - Giảm xác suất chọn `bad_action` (nước đi sai lầm).
        """
        if self.model is None or not hasattr(self.model, "policy"):
            return 0.0

        policy = self.model.policy
        optimizer = policy.optimizer

        # Chuẩn bị Tensor quan sát
        if hasattr(self.model, "observation_space"):
            sp = self.model.observation_space.shape
            if len(sp) == 3 and sp[0] == 2:
                # 2 channels
                c0 = (board == 1).astype(np.float32)
                c1 = (board == 2).astype(np.float32)
                obs_arr = np.stack([c0, c1], axis=0)
            else:
                obs_arr = board.astype(np.float32)
        else:
            obs_arr = board.astype(np.float32)

        obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=policy.device).unsqueeze(0)
        target_tensor = torch.tensor([correct_action], dtype=torch.long, device=policy.device)

        last_loss = 0.0
        for _ in range(steps):
            dist = policy.get_distribution(obs_tensor)
            # Loss chính: Tăng xác suất nước cờ đúng
            loss = -dist.log_prob(target_tensor).mean()

            # Phạt thêm nếu trước đó đã chọn bad_action
            if bad_action is not None and bad_action != correct_action:
                bad_tensor = torch.tensor([bad_action], dtype=torch.long, device=policy.device)
                bad_log_prob = dist.log_prob(bad_tensor).mean()
                loss = loss + 0.5 * bad_log_prob

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

        return last_loss


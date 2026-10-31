import os
import sys

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi UnicodeEncodeError (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from connect4 import Connect4
from heuristic import HeuristicAgent
from minimax import MinimaxAgent
from ensemble import EnsembleAgent
from adaptive_learner import AdaptiveOnlineLearner
from stable_baselines3 import PPO
from gui import Connect4GUI

WIN_REWARD = 10
BLOCK_REWARD = 3
DRAW_REWARD = 1
CONTINUE_REWARD = 0
ILLEGAL_MOVE_PENALTY = -10


def _load_agents_from_dir(dir_path):
    """Tự động tìm và nạp tất cả các agent (agent0, agent1, ...) có trong thư mục."""
    if not os.path.exists(dir_path):
        return []

    agent_files = [f for f in os.listdir(dir_path) if f.startswith("agent") and f.endswith(".zip")]
    # Sắp xếp theo thứ tự số: agent0, agent1, agent2...
    agent_files.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))

    loaded = []
    for f in agent_files:
        name_no_ext = os.path.splitext(f)[0]
        try:
            model = PPO.load(os.path.join(dir_path, name_no_ext))
            loaded.append(model)
        except Exception as e:
            print(f"[Warning] Khong the nap {f}: {e}")
    return loaded


def get_agent_for_board(rows=8, cols=9):
    """
    Nạp Agent cho bàn cờ chuẩn 8x9:
    - Nạp EnsembleAgent gồm các PPO agent đã huấn luyện từ Colab (thư mục saved_agents_8x9).
    - Nếu chưa train xong hoặc chưa có checkpoint, tự động sử dụng HeuristicAgent thông minh cho 8x9.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agents_8x9_dir = os.path.join(base_dir, "saved_agents_8x9")
    players_8x9 = _load_agents_from_dir(agents_8x9_dir)

    # Nếu người dùng giải nén ở thư mục gốc ngoài cùng
    if not players_8x9:
        root_dir = os.path.dirname(base_dir)
        root_agents_dir = os.path.join(root_dir, "saved_agents_8x9")
        players_8x9 = _load_agents_from_dir(root_agents_dir)

    if players_8x9:
        print(f"[8x9] Da nap EnsembleAgent 8x9 gom {len(players_8x9)} agents da train tu Colab!")
        # Kích hoạt hệ thống học thích ứng trực tuyến từ người chơi (Online Adaptive Learning)
        primary_model = players_8x9[-1]  # agent15
        save_path = os.path.join(agents_8x9_dir, "agent15")
        learner = AdaptiveOnlineLearner(agent_model=primary_model, model_save_path=save_path)
        print(f"[⚡] He thong Hoc Thich Nghi (Online Learning) da san sang! (Da nho {learner.total_learned_count} the co)")
        return EnsembleAgent(players_8x9, learner=learner)

    print("[8x9] Chua co checkpoint PPO trong saved_agents_8x9. Su dung HeuristicAgent toi uu cho 8x9!")
    learner = AdaptiveOnlineLearner(agent_model=None)
    heuristic = HeuristicAgent(c=0)
    heuristic.learner = learner
    return heuristic


def play(opponent_factory=None, initial_rows=8, initial_cols=9):
    """
    Khởi chạy giao diện Pygame Connect 4 chuẩn 8 hàng x 9 cột.
    """
    if opponent_factory is None:
        opponent_factory = get_agent_for_board

    gui = Connect4GUI(
        opponent_factory=opponent_factory,
        initial_rows=initial_rows,
        initial_cols=initial_cols,
    )
    gui.run()


if __name__ == "__main__":
    play()

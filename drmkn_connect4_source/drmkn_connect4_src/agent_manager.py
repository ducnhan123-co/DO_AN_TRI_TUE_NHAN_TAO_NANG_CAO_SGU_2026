import os
import sys

from random_agent import RandomAgent
from heuristic import HeuristicAgent
from minimax import MinimaxAgent
from mcts_agent import MCTSAgent
from supervised_agent import SupervisedAgent
from rl_agent import RLAgent
from alphazero_agent import AlphaZeroAgent
from self_play import AlphaZeroMasterAgent


LEVEL_NAMES = {
    1: "Cấp 1: Random (Ngẫu nhiên)",
    2: "Cấp 2: Heuristic (Luật kinh nghiệm)",
    3: "Cấp 3: Minimax (Alpha-Beta)",
    4: "Cấp 4: Monte Carlo (MCTS)",
    5: "Cấp 5: Học Có Giám Sát (CNN)",
    6: "Cấp 6: Học Tăng Cường (PPO)",
    7: "Cấp 7: AlphaZero (CNN + PUCT)",
    8: "Cấp 8: AlphaZero Master (Self-Play)",
}

LEVEL_DESCRIPTIONS = {
    1: "Ra quyết định ngẫu nhiên trong các cột hợp lệ chưa đầy.",
    2: "Đánh giá thế trận bằng hàm điểm cục bộ, bắt 4 và chặn 3 tức thời.",
    3: "Duyệt cây trạng thái Minimax với cắt tỉa Alpha-Beta và hàm định giá nút lá.",
    4: "Mô phỏng ngẫu nhiên Monte Carlo Tree Search (Selection, Expansion, Rollout, Backprop).",
    5: "Mạng nơ-ron tích chập (Deep CNN) được huấn luyện từ các thế cờ chuyên gia.",
    6: "Mô hình Học Tăng Cường PPO tự chơi và học từ phần thưởng trong môi trường Connect-4.",
    7: "Kiến trúc AlphaZero chuẩn kết hợp mạng CNN Policy-Value với tìm kiếm PUCT MCTS.",
    8: "Phiên bản AlphaZero tối thượng: Tự chơi qua nhiều thế hệ (League), chống bẫy 2 đầu và tự học.",
}


def get_agent(level=8, rows=8, cols=9, learner=None, model_dir=None):
    """
    Trả về đối tượng Agent theo đúng Cấp độ từ 1 đến 8.
    """
    if model_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "saved_agents_8x9")

    level = int(level)
    if level == 1:
        return RandomAgent()
    elif level == 2:
        agent = HeuristicAgent(c=0)
        if learner:
            agent.learner = learner
        return agent
    elif level == 3:
        return MinimaxAgent(depth=3)
    elif level == 4:
        return MCTSAgent(simulations=200, c_param=1.414)
    elif level == 5:
        pth_path = os.path.join(model_dir, "level5_supervised.pth")
        return SupervisedAgent(model_path=pth_path, rows=rows, cols=cols)
    elif level == 6:
        zip_path = os.path.join(model_dir, "level6_rl_ppo")
        return RLAgent(model_path=zip_path, rows=rows, cols=cols)
    elif level == 7:
        zip_path = os.path.join(model_dir, "level7_alphazero")
        return AlphaZeroAgent(model_path=zip_path, rows=rows, cols=cols)
    elif level == 8:
        zip_path = os.path.join(model_dir, "level8_alphazero_master")
        return AlphaZeroMasterAgent(model_path=zip_path, rows=rows, cols=cols, learner=learner)
    else:
        raise ValueError(f"Cấp độ không hợp lệ: {level}. Vui lòng chọn từ 1 đến 8.")

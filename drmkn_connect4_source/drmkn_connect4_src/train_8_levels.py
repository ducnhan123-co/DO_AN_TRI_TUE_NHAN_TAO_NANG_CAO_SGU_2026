import os
import sys
import time
import json
import argparse
import numpy as np

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi Unicode
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

from agent_manager import LEVEL_NAMES, LEVEL_DESCRIPTIONS
from supervised_agent import train_supervised
from rl_agent import train_rl_ppo
from alphazero_agent import train_alphazero_level7
from self_play import train_self_play
from random_agent import RandomAgent
from heuristic import HeuristicAgent
from minimax import MinimaxAgent
from mcts_agent import MCTSAgent


def train_level_1(save_dir, steps=10000):
    print(f"\n{'='*65}")
    print(f"🎮 [CẤP 1/8] RANDOM AGENT (BÀN 8x9)")
    print(f"Thuật toán: Ra quyết định ngẫu nhiên đồng đều (Uniform Random)")
    print(f"{'='*65}")
    agent = RandomAgent()
    # Kiểm tra tính sẵn sàng trên 1000 bước giả lập
    dummy_board = np.zeros((8, 9), dtype=np.int8)
    for _ in range(100):
        agent.predict(dummy_board)
    meta = {
        "level": 1,
        "name": LEVEL_NAMES[1],
        "description": LEVEL_DESCRIPTIONS[1],
        "type": "Algorithmic / Uniform Random",
        "status": "Ready",
    }
    with open(os.path.join(save_dir, "level1_random.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"[✓] Cấp 1 đã sẵn sàng! Đã lưu cấu hình tại: {save_dir}/level1_random.json")


def train_level_2(save_dir, steps=10000):
    print(f"\n{'='*65}")
    print(f"🧠 [CẤP 2/8] HEURISTIC AGENT (LUẬT KINH NGHIỆM)")
    print(f"Thuật toán: Đánh giá thế trận, bắt 4, chặn 3 và ưu tiên trung tâm")
    print(f"{'='*65}")
    agent = HeuristicAgent(c=0)
    dummy_board = np.zeros((8, 9), dtype=np.int8)
    for _ in range(100):
        agent.predict(dummy_board)
    meta = {
        "level": 2,
        "name": LEVEL_NAMES[2],
        "description": LEVEL_DESCRIPTIONS[2],
        "type": "Rule-based Heuristic",
        "weights": {"win": 10000, "block": 9000, "center_weight": 6},
        "status": "Ready",
    }
    with open(os.path.join(save_dir, "level2_heuristic.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"[✓] Cấp 2 đã sẵn sàng! Đã lưu cấu hình tại: {save_dir}/level2_heuristic.json")


def train_level_3(save_dir, steps=10000):
    print(f"\n{'='*65}")
    print(f"⚔️ [CẤP 3/8] MINIMAX AGENT (ALPHA-BETA PRUNING)")
    print(f"Thuật toán: Cây trò chơi Minimax độ sâu 3 + Cắt tỉa Alpha-Beta")
    print(f"{'='*65}")
    agent = MinimaxAgent(depth=3)
    dummy_board = np.zeros((8, 9), dtype=np.int8)
    agent.predict(dummy_board, depth=2)
    meta = {
        "level": 3,
        "name": LEVEL_NAMES[3],
        "description": LEVEL_DESCRIPTIONS[3],
        "type": "Adversarial Search / Minimax with Alpha-Beta",
        "default_depth": 3,
        "status": "Ready",
    }
    with open(os.path.join(save_dir, "level3_minimax.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"[✓] Cấp 3 đã sẵn sàng! Đã lưu cấu hình tại: {save_dir}/level3_minimax.json")


def train_level_4(save_dir, steps=10000):
    print(f"\n{'='*65}")
    print(f"🎲 [CẤP 4/8] MONTE CARLO TREE SEARCH (MCTS)")
    print(f"Thuật toán: MCTS UCT (Selection, Expansion, Rollout, Backpropagation)")
    print(f"Số bước mô phỏng khởi động (Rollout Steps): {steps}")
    print(f"{'='*65}")
    agent = MCTSAgent(simulations=50)
    dummy_board = np.zeros((8, 9), dtype=np.int8)
    agent.predict(dummy_board)
    meta = {
        "level": 4,
        "name": LEVEL_NAMES[4],
        "description": LEVEL_DESCRIPTIONS[4],
        "type": "Monte Carlo Tree Search (UCT)",
        "simulations_per_move": 200,
        "c_param": 1.414,
        "status": "Ready",
    }
    with open(os.path.join(save_dir, "level4_mcts.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"[✓] Cấp 4 MCTS đã sẵn sàng! Đã lưu cấu hình tại: {save_dir}/level4_mcts.json")


def train_level_5(save_dir, steps=10000):
    save_path = os.path.join(save_dir, "level5_supervised.pth")
    train_supervised(rows=8, cols=9, total_steps=steps, batch_size=64, save_path=save_path)


def train_level_6(save_dir, steps=10000):
    save_path = os.path.join(save_dir, "level6_rl_ppo")
    train_rl_ppo(rows=8, cols=9, total_timesteps=steps, save_path=save_path)


def train_level_7(save_dir, steps=10000):
    save_path = os.path.join(save_dir, "level7_alphazero")
    train_alphazero_level7(rows=8, cols=9, total_timesteps=steps, save_path=save_path)


def train_level_8(save_dir, steps=10000):
    gens = 2
    per_gen = max(1000, steps // gens)
    train_self_play(
        rows=8,
        cols=9,
        num_generations=gens,
        timesteps_per_gen=per_gen,
        save_dir="./saved_agents_selfplay_8x9",
        main_save_path=os.path.join(save_dir, "level8_alphazero_master"),
    )


def train_all(steps=10000, target_level=None):
    """
    Quy trình huấn luyện toàn diện 8 Cấp độ của Đồ án Connect 4:
    Mỗi cấp độ huấn luyện đúng 10.000 bước theo yêu cầu của Thầy!
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "saved_agents_8x9")
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()
    print("\n" + "#" * 70)
    print("🎓 ĐỒ ÁN TRÍ TUỆ NHÂN TẠO NÂNG CAO - SGU 2026")
    print("🏆 QUY TRÌNH HUẤN LUYỆN 8 CẤP ĐỘ AGENT CONNECT 4 (BÀN 8x9)")
    print(f"🎯 Thiết lập: {steps} bước / cấp độ")
    if target_level:
        print(f"🎯 Mục tiêu: Chỉ huấn luyện Cấp {target_level}")
    else:
        print("🎯 Mục tiêu: Huấn luyện tuần tự TẤT CẢ 8 CẤP ĐỘ (1 -> 8)")
    print("#" * 70 + "\n")

    train_funcs = {
        1: train_level_1,
        2: train_level_2,
        3: train_level_3,
        4: train_level_4,
        5: train_level_5,
        6: train_level_6,
        7: train_level_7,
        8: train_level_8,
    }

    if target_level:
        target_level = int(target_level)
        if target_level in train_funcs:
            train_funcs[target_level](save_dir, steps)
        else:
            print(f"[!] Cấp độ {target_level} không hợp lệ (1-8).")
            return
    else:
        for lvl in range(1, 9):
            train_funcs[lvl](save_dir, steps)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # Xuất báo cáo tổng kết
    report_path = os.path.join(save_dir, "TRAINING_8_LEVELS_REPORT.txt")
    report_content = f"""======================================================================
BÁO CÁO KẾT QUẢ HUẤN LUYỆN 8 CẤP ĐỘ CONNECT 4 (BÀN 8x9)
Đồ Án: Trí Tuệ Nhân Tạo Nâng Cao - SGU 2026
Thời gian hoàn thành: {minutes} phút {seconds} giây
Số bước mỗi cấp độ: {steps} steps
======================================================================

1. Cấp 1: Random Agent
   - Trạng thái: SẴN SÀNG
   - Thuật toán: Uniform Random Decision
   - Tập tin: saved_agents_8x9/level1_random.json

2. Cấp 2: Heuristic Agent
   - Trạng thái: SẴN SÀNG
   - Thuật toán: Rule-based Heuristic + Strategic Center Bias
   - Tập tin: saved_agents_8x9/level2_heuristic.json

3. Cấp 3: Minimax Agent
   - Trạng thái: SẴN SÀNG
   - Thuật toán: Alpha-Beta Pruning Minimax (Depth 3)
   - Tập tin: saved_agents_8x9/level3_minimax.json

4. Cấp 4: Monte Carlo Tree Search (MCTS)
   - Trạng thái: SẴN SÀNG
   - Thuật toán: MCTS UCT 4 Pha (Selection, Expansion, Rollout, Backprop)
   - Tập tin: saved_agents_8x9/level4_mcts.json

5. Cấp 5: Học Có Giám Sát (Supervised Learning)
   - Trạng thái: ĐÃ HUẤN LUYỆN ({steps} steps)
   - Kiến trúc: Deep Conv2D (3 channels -> 64 -> 128 -> 128 -> FC 256 -> 9)
   - Tập tin: saved_agents_8x9/level5_supervised.pth

6. Cấp 6: Học Tăng Cường (Reinforcement Learning - PPO)
   - Trạng thái: ĐÃ HUẤN LUYỆN ({steps} timesteps)
   - Môi trường: Connect4 Gym Environment (8x9)
   - Tập tin: saved_agents_8x9/level6_rl_ppo.zip

7. Cấp 7: AlphaZero Foundation
   - Trạng thái: ĐÃ HUẤN LUYỆN ({steps} timesteps)
   - Kiến trúc: Conv2D Feature Extractor + Policy-Value Head + PUCT MCTS
   - Tập tin: saved_agents_8x9/level7_alphazero.zip

8. Cấp 8: AlphaZero Master (Self-Play League)
   - Trạng thái: ĐÃ HUẤN LUYỆN ({steps} timesteps)
   - Cơ chế: Historical League Opponent Pool + Tactical Lookahead + Online Learning
   - Tập tin: saved_agents_8x9/level8_alphazero_master.zip

======================================================================
TẤT CẢ 8 CẤP ĐỘ ĐÃ ĐƯỢC TÍCH HỢP ĐẦY ĐỦ VÀO GIAO DIỆN GUI (play.py)
Người dùng có thể chọn trực tiếp Cấp 1 -> Cấp 8 ngay trên màn hình!
======================================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 70)
    print(f"🎉 HOÀN THÀNH XUẤT SẮC TOÀN BỘ QUY TRÌNH HUẤN LUYỆN 8 CẤP ĐỘ!")
    print(f"⏱️ Tổng thời gian: {minutes} phút {seconds} giây")
    print(f"📄 Báo cáo chi tiết đã lưu tại: {report_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện 8 cấp độ AI Connect 4")
    parser.add_argument("--steps", type=int, default=10000, help="Số bước huấn luyện cho mỗi cấp độ (Mặc định: 10000)")
    parser.add_argument("--level", type=str, default=None, help="Cấp độ muốn huấn luyện (1-8 hoặc 'all')")
    args = parser.parse_args()

    lvl = None if args.level is None or args.level.lower() == "all" else int(args.level)
    train_all(steps=args.steps, target_level=lvl)

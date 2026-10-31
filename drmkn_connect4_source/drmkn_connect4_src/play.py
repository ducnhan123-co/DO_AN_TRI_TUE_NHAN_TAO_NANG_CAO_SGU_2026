import os
import sys
import argparse

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

from agent_manager import get_agent, LEVEL_NAMES, LEVEL_DESCRIPTIONS
from adaptive_learner import AdaptiveOnlineLearner
from gui import Connect4GUI


# Bộ nhớ học thích ứng trực tuyến toàn cục
_shared_learner = None


def get_shared_learner():
    global _shared_learner
    if _shared_learner is None:
        _shared_learner = AdaptiveOnlineLearner(agent_model=None)
    return _shared_learner


def opponent_factory(level=8, rows=8, cols=9):
    """
    Tạo hoặc nạp Agent tương ứng theo Cấp độ (1 -> 8) cho bàn cờ 8x9.
    """
    learner = get_shared_learner()
    agent = get_agent(level=level, rows=rows, cols=cols, learner=learner)
    return agent


def play(initial_level=8, rows=8, cols=9):
    """
    Khởi chạy giao diện Đồ án Connect 4 chuẩn kích thước 8 hàng x 9 cột,
    tích hợp lựa chọn 8 cấp độ AI theo yêu cầu của Thầy!
    """
    print("\n" + "=" * 65)
    print("🎮 KHỞI CHẠY ARENA CONNECT 4 (BÀN CỜ 8x9) - SGU 2026")
    print(f"⭐ Cấp độ AI ban đầu: {LEVEL_NAMES.get(initial_level, f'Cấp {initial_level}')}")
    print("💡 Mẹo: Nhấn các phím [1] đến [8] để chuyển cấp độ AI tức thì khi đang chơi!")
    print("=" * 65 + "\n")

    gui = Connect4GUI(
        opponent_factory=opponent_factory,
        initial_rows=rows,
        initial_cols=cols,
        initial_level=initial_level,
    )
    gui.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chơi Connect 4 với 8 cấp độ AI")
    parser.add_argument(
        "-l", "--level",
        type=int,
        default=8,
        choices=range(1, 9),
        help="Chọn cấp độ AI từ 1 đến 8 (Mặc định: 8 - AlphaZero Master)",
    )
    parser.add_argument("--rows", type=int, default=8, help="Số hàng bàn cờ (Mặc định: 8)")
    parser.add_argument("--cols", type=int, default=9, help="Số cột bàn cờ (Mặc định: 9)")
    args = parser.parse_args()

    play(initial_level=args.level, rows=args.rows, cols=args.cols)


import numpy as np
import torch
from sb3_contrib import MaskablePPO

from c4core import (
    ROWS, COLS, drop_bit, has_won, legal_cols,
    is_full, bb_to_grid
)
from c4env import TorchPlayer


MODEL_PATH = "models/agent_v2.zip"


def show_board(human, agent):
    grid = bb_to_grid(human, agent, 1)

    print("\n  1 2 3 4 5 6 7")
    print(" +" + "--" * COLS + "-+")

    for row in grid:
        symbols = ["." if v == 0 else ("X" if v == 1 else "O")
                   for v in row]
        print(" | " + " ".join(symbols) + " |")

    print(" +" + "--" * COLS + "-+")
    print("  X = Bạn | O = PPO Agent")


def human_move(human, agent):
    legal = legal_cols(human | agent)

    while True:
        raw = input("Bạn chọn cột (1-7, q để thoát): ").strip()

        if raw.lower() == "q":
            return None

        try:
            col = int(raw) - 1
        except ValueError:
            print("Vui lòng nhập số từ 1 đến 7.")
            continue

        if col not in legal:
            print("Cột này không hợp lệ hoặc đã đầy.")
            continue

        return col


def main():
    torch.set_num_threads(1)

    print("Đang tải PPO Agent...")

    model = MaskablePPO.load(MODEL_PATH)

    # Phải dùng mask đúng như lúc đánh giá model.
    agent = TorchPlayer(
        model.policy,
        deterministic=True,
        flip_avg=True,
        shield=True
    )

    print("Đã tải model thành công!")
    print("Bạn đi trước (X), Agent đi sau (O).")
    print("Nhập q để thoát.")

    while True:
        human = 0
        opponent = 0
        show_board(human, opponent)

        while True:
            # Lượt người chơi
            col = human_move(human, opponent)

            if col is None:
                print("Đã thoát game.")
                return

            move = drop_bit(human | opponent, col)
            human |= move

            show_board(human, opponent)

            if has_won(human):
                print("CHÚC MỪNG! Bạn đã thắng PPO Agent!")
                break

            if is_full(human | opponent):
                print("Ván đấu hòa!")
                break

            # Lượt PPO Agent
            print("PPO Agent đang suy nghĩ...")

            col = agent.act(opponent, human)
            print(f"PPO Agent chọn cột {col + 1}")

            move = drop_bit(human | opponent, col)

            if not move:
                print("Lỗi: Agent chọn cột không hợp lệ.")
                return

            opponent |= move
            show_board(human, opponent)

            if has_won(opponent):
                print("PPO AGENT ĐÃ THẮNG!")
                break

            if is_full(human | opponent):
                print("Ván đấu hòa!")
                break

        again = input("\nChơi ván mới? (y/n): ").strip().lower()

        if again != "y":
            print("Cảm ơn bạn đã chơi!")
            break


if __name__ == "__main__":
    main()

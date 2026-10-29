"""play_v2.py - Menu chơi Connect 4 (giao diện như code cũ) với bàn 8x9 và agent v2.

    python play_v2.py [--model models/agent_8x9.zip] [--cell 100]
"""
import argparse
import os
import time

import numpy as np

from c4core import (COLS, ROWS, MinimaxPlayer, RandomPlayer, TacticalPlayer, drop_bit, has_won,
                    is_full, legal_cols)

MODEL_PATH = "models/agent_8x9.zip"


def load_agent(path):
    from sb3_contrib import MaskablePPO
    from c4env import TorchPlayer
    m = MaskablePPO.load(path)
    return TorchPlayer(m.policy, deterministic=True, flip_avg=True, name=os.path.basename(path))


def play(first, second, name_first, name_second, cell=100, delay=0.5):
    """first/second: Player có act(me, opp), hoặc None = người (chuột). Người/ai đi trước là ĐỎ."""
    from gui import BoardView
    view = BoardView(cell)
    bbs = [0, 0]
    players, names = (first, second), (name_first, name_second)
    title = f"Connect 4 | {name_first} (đỏ, đi trước) vs {name_second} (vàng)"
    view.set_caption(title)
    result = None
    for ply in range(ROWS * COLS):
        cur = ply % 2
        me, opp = bbs[cur], bbs[1 - cur]
        view.draw(bbs[0], bbs[1])
        if not view.pump():
            view.close(); return
        if players[cur] is None:  # người chơi: click chuột, bỏ qua cột đã đầy
            while True:
                a = view.get_col()
                if a is None:
                    view.close(); return
                if drop_bit(me | opp, a):
                    break
        else:
            time.sleep(delay)
            a = int(players[cur].act(me, opp))
            if not drop_bit(me | opp, a):  # phòng hờ: agent trả nước sai -> đi ngẫu nhiên hợp lệ
                a = legal_cols(me | opp)[0]
        bbs[cur] |= drop_bit(me | opp, a)
        if has_won(bbs[cur]):
            result = f"{names[cur]} thắng!"
            break
    if result is None:
        result = "Hoà!"
    view.draw(bbs[0], bbs[1])
    view.set_caption(f"{title} | {result}")
    print("\n========== KẾT QUẢ ==========")
    print(f"Đỏ (đi trước): {name_first} | Vàng: {name_second}")
    print(result)
    t = time.time()
    while time.time() - t < 3 and view.pump():
        time.sleep(0.05)
    view.close()


def human_vs(opponent, opp_name, cell):
    human_first = np.random.random() < 0.5
    if human_first:
        print("Bạn là ĐỎ, đi trước.")
        play(None, opponent, "Human", opp_name, cell)
    else:
        print("Bạn là VÀNG, đối thủ đi trước.")
        play(opponent, None, opp_name, "Human", cell)


def ask_model(default):
    p = input(f"Đường dẫn model [{default}]: ").strip().strip('"') or default
    if not (os.path.exists(p) or os.path.exists(p + ".zip")):
        print("Không tìm thấy model:", p)
        return None, None
    return load_agent(p), "Agent " + os.path.basename(p)


def ask_depth():
    s = input("Độ sâu minimax (1-8) [4]: ").strip()
    return min(max(int(s), 1), 8) if s.isdigit() else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--cell", type=int, default=100, help="kích thước ô (px), giảm nếu màn hình nhỏ")
    a = ap.parse_args()
    while True:
        print("\n" + "=" * 42)
        print(f"     CONNECT 4 ({ROWS}x{COLS}) - AI MENU")
        print("=" * 42)
        print("1. Human vs Random")
        print("2. Human vs Tactical (thắng/chặn nước 1)")
        print("3. Human vs Minimax")
        print("4. Human vs trained agent")
        print("5. Agent vs Agent")
        print("0. Exit")
        print("=" * 42)
        c = input("Choose mode: ").strip()
        if c == "1":
            human_vs(RandomPlayer(), "Random", a.cell)
        elif c == "2":
            human_vs(TacticalPlayer(), "Tactical", a.cell)
        elif c == "3":
            d = ask_depth()
            human_vs(MinimaxPlayer(d), f"Minimax(d={d})", a.cell)
        elif c == "4":
            ag, name = ask_model(a.model)
            if ag:
                human_vs(ag, name, a.cell)
        elif c == "5":
            print("--- Agent A (đỏ, đi trước) ---"); A, na = ask_model(a.model)
            print("--- Agent B (vàng) ---"); B, nb = ask_model(a.model)
            if A and B:
                play(A, B, na + " [A]", nb + " [B]", a.cell, delay=0.6)
        elif c == "0":
            print("Exiting Connect 4."); break
        else:
            print("Invalid choice. Please select 0-5.")


if __name__ == "__main__":
    main()

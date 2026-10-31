import os
import sys
import time
import argparse
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from connect4 import Connect4
from agent_manager import get_agent, LEVEL_NAMES
from random_agent import RandomAgent
from heuristic import HeuristicAgent

WIN_REWARD = 10


def format_input_for(player, grid, symb=1, opp_symb=2):
    if hasattr(player, "observation_space"):
        sp = getattr(player.observation_space, "shape", None)
        if sp and len(sp) == 3 and sp[0] == 2:
            return np.stack(
                [(grid == symb).astype(np.float32), (grid == opp_symb).astype(np.float32)],
                axis=0,
            )
    return grid


def play_eval_game(player, opponent, rows=8, cols=9):
    """
    Chạy 1 ván đối đầu chuẩn giữa player và opponent.
    Trả về: 1 nếu player thắng, -1 nếu opponent thắng, 0 nếu hòa.
    """
    env = Connect4(rows=rows, cols=cols)
    env.set_opp_agent(opponent, epsilon=0.0)
    done = False
    env.reset()

    while not done:
        inp = format_input_for(player, env.grid, env.agent_symb, env.opp_symb)
        pred = player.predict(inp)
        action = pred[0] if isinstance(pred, (tuple, list, np.ndarray)) else pred
        if isinstance(action, (tuple, list, np.ndarray)):
            action = int(action[0])
        else:
            action = int(action)

        _, reward, done, _, _ = env.step(action)

    env.close()
    if reward == WIN_REWARD:
        return 1
    elif reward == -WIN_REWARD:
        return -1
    else:
        return 0


def benchmark_level(level, opponent_agent, opp_name, num_games=10, rows=8, cols=9):
    agent = get_agent(level=level, rows=rows, cols=cols)
    wins = 0
    draws = 0
    losses = 0

    times = []
    for _ in range(num_games):
        t0 = time.time()
        res = play_eval_game(agent, opponent_agent, rows=rows, cols=cols)
        times.append(time.time() - t0)
        if res == 1:
            wins += 1
        elif res == 0:
            draws += 1
        else:
            losses += 1

    win_rate = (wins / num_games) * 100.0
    avg_time = (sum(times) / num_games) * 1000.0 if times else 0.0
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": win_rate,
        "avg_time": avg_time,
    }


def benchmark_all_8_levels(num_games=10):
    print("\n" + "=" * 80, flush=True)
    print("🏆 BẢNG ĐÁNH GIÁ VÀ KIỂM THỬ HIỆU NĂNG 8 CẤP ĐỘ AI CONNECT 4 (BÀN 8x9)", flush=True)
    print(f"📊 Mỗi Cấp độ thi đấu {num_games} ván vs Random và {num_games} ván vs Heuristic", flush=True)
    print("=" * 80 + "\n", flush=True)

    random_opp = RandomAgent()
    heuristic_opp = HeuristicAgent(c=0)

    results = []
    header = f"{'Cấp Độ':<28} | {'Vs Random (Thắng%)':<20} | {'Vs Heuristic (Thắng%)':<22} | {'Tốc độ':<10}"
    print(header, flush=True)
    print("-" * 85, flush=True)

    for lvl in range(1, 9):
        name = LEVEL_NAMES[lvl]
        res_rnd = benchmark_level(lvl, random_opp, "Random", num_games=num_games)
        res_heu = benchmark_level(lvl, heuristic_opp, "Heuristic", num_games=num_games)

        rnd_str = f"{res_rnd['wins']}/{num_games} ({res_rnd['win_rate']:.0f}%)"
        heu_str = f"{res_heu['wins']}/{num_games} ({res_heu['win_rate']:.0f}%)"
        spd_str = f"{res_heu['avg_time']:.0f}ms"

        print(f"{name:<28} | {rnd_str:<20} | {heu_str:<22} | {spd_str:<10}", flush=True)
        results.append((name, rnd_str, heu_str, spd_str))

    print("-" * 85, flush=True)
    print("[✓] Hoàn thành kiểm thử toàn bộ 8 Cấp độ!\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử đánh giá 8 cấp độ AI")
    parser.add_argument("--games", type=int, default=10, help="Số ván mỗi bài test (Mặc định: 10)")
    args = parser.parse_args()

    benchmark_all_8_levels(num_games=args.games)

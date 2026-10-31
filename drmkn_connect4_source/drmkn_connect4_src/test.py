import os
import sys
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tqdm import tqdm
from stable_baselines3 import PPO
from connect4 import Connect4
from ensemble import EnsembleAgent
from heuristic import HeuristicAgent
from minimax import MinimaxAgent

WIN_REWARD = 10
BLOCK_REWARD = 3
DRAW_REWARD = 1
CONTINUE_REWARD = 0
ILLEGAL_MOVE_PENALTY = -100


def format_input_for(player, grid, symb=1, opp_symb=2):
    if hasattr(player, "observation_space"):
        sp = getattr(player.observation_space, "shape", None)
        if sp and len(sp) == 3 and sp[0] == 2:
            return np.stack(
                [(grid == symb).astype(np.float32), (grid == opp_symb).astype(np.float32)],
                axis=0,
            )
    return grid


def test_one_game(player, opponent=None):
    """
    Chạy 1 ván đấu chuẩn xác theo logic gốc của project:
    - Trả về True nếu Player THẮNG (reward == 10).
    - Trả về False nếu HÒA hoặc THUA.
    """
    env = Connect4(rows=8, cols=9)
    if opponent:
        env.set_opp_agent(opponent, 0)

    done = False
    env.reset()
    while not done:
        inp = format_input_for(player, env.grid, env.agent_symb, env.opp_symb)
        action = player.predict(inp)[0]
        if isinstance(action, (np.ndarray, list, tuple)):
            if getattr(action, "ndim", 0) > 0 and len(action) > 0:
                action = int(action[0])
            else:
                action = int(action)
        else:
            action = int(action)
        _, reward, done, _, _ = env.step(action)

    env.close()
    return reward == WIN_REWARD


def test_all_agents(games_per_agent=20):
    """
    Kiểm thử TOÀN BỘ các tác tử trong hệ thống:
    1. Từng Agent PPO (agent0 -> agent15)
    2. Ensemble Agent (kết hợp 16 PPO)
    3. Heuristic Agent (Luật kinh nghiệm)
    4. Minimax Agent (Cây tìm kiếm Alpha-Beta)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.join(base_dir, "saved_agents_8x9")

    agent_files = [f for f in os.listdir(agents_dir) if f.startswith("agent") and f.endswith(".zip")] if os.path.exists(agents_dir) else []
    agent_files.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))

    if not agent_files:
        print("[!] Không tìm thấy mô hình trong saved_agents_8x9!")
        return

    print(f"\n" + "=" * 70)
    print(f"      TIẾN HÀNH KIỂM THỬ TOÀN DIỆN TẤT CẢ CÁC AGENT TRÊN BÀN 8x9")
    print(f"      (Mỗi Agent đối đầu {games_per_agent} ván vs Random & Heuristic)")
    print("=" * 70 + "\n")

    heuristic_opp = HeuristicAgent(c=0)
    results = []

    # 1. Đánh giá từng PPO Agent lẻ (agent0 -> agent15)
    print(f"[*] Đang tải và kiểm thử {len(agent_files)} PPO Agents...")
    loaded_players = []
    for f in agent_files:
        path = os.path.join(agents_dir, os.path.splitext(f)[0])
        model = PPO.load(path)
        loaded_players.append(model)
        name = os.path.splitext(f)[0]

        # Đấu vs Random
        w_rand = sum(1 for _ in range(games_per_agent) if test_one_game(model, opponent=None))
        # Đấu vs Heuristic
        w_heu = sum(1 for _ in range(games_per_agent) if test_one_game(model, opponent=heuristic_opp))

        rate_rand = (w_rand / games_per_agent) * 100
        rate_heu = (w_heu / games_per_agent) * 100
        avg_rate = (rate_rand + rate_heu) / 2
        results.append({
            "name": name,
            "vs_rand": f"{rate_rand:.1f}% ({w_rand}/{games_per_agent})",
            "vs_heu": f"{rate_heu:.1f}% ({w_heu}/{games_per_agent})",
            "avg": f"{avg_rate:.1f}%"
        })
        print(f"   ✓ {name:10s} | vs Random: {rate_rand:5.1f}% | vs Heuristic: {rate_heu:5.1f}%")

    # 2. Đánh giá Ensemble Agent (Tổ hợp 16 PPO)
    print("\n[*] Đang kiểm thử Ensemble Agent (Tổ hợp 16 PPO)...")
    ensemble = EnsembleAgent(loaded_players)
    w_rand = sum(1 for _ in range(games_per_agent) if test_one_game(ensemble, opponent=None))
    w_heu = sum(1 for _ in range(games_per_agent) if test_one_game(ensemble, opponent=heuristic_opp))
    rate_rand = (w_rand / games_per_agent) * 100
    rate_heu = (w_heu / games_per_agent) * 100
    avg_rate = (rate_rand + rate_heu) / 2
    results.append({
        "name": "Ensemble(16 PPO)",
        "vs_rand": f"{rate_rand:.1f}% ({w_rand}/{games_per_agent})",
        "vs_heu": f"{rate_heu:.1f}% ({w_heu}/{games_per_agent})",
        "avg": f"{avg_rate:.1f}%"
    })
    print(f"   ✓ Ensemble   | vs Random: {rate_rand:5.1f}% | vs Heuristic: {rate_heu:5.1f}%")

    # 3. Đánh giá Heuristic Agent vs Random
    print("\n[*] Đang kiểm thử Heuristic Agent vs Random...")
    w_rand = sum(1 for _ in range(games_per_agent) if test_one_game(heuristic_opp, opponent=None))
    rate_rand = (w_rand / games_per_agent) * 100
    results.append({
        "name": "HeuristicAgent",
        "vs_rand": f"{rate_rand:.1f}% ({w_rand}/{games_per_agent})",
        "vs_heu": "  --- (Benchmark)",
        "avg": f"{rate_rand:.1f}%"
    })
    print(f"   ✓ Heuristic  | vs Random: {rate_rand:5.1f}%")

    # 4. Đánh giá Minimax Agent (Depth 2) vs Random & Heuristic
    print("\n[*] Đang kiểm thử Minimax Agent (Depth=2)...")
    minimax_opp = MinimaxAgent(depth=2)
    mini_games = min(games_per_agent, 10) # 10 ván vì minimax duyệt cây
    w_rand = sum(1 for _ in range(mini_games) if test_one_game(minimax_opp, opponent=None))
    w_heu = sum(1 for _ in range(mini_games) if test_one_game(minimax_opp, opponent=heuristic_opp))
    rate_rand = (w_rand / mini_games) * 100
    rate_heu = (w_heu / mini_games) * 100
    avg_rate = (rate_rand + rate_heu) / 2
    results.append({
        "name": "Minimax(d=2)",
        "vs_rand": f"{rate_rand:.1f}% ({w_rand}/{mini_games})",
        "vs_heu": f"{rate_heu:.1f}% ({w_heu}/{mini_games})",
        "avg": f"{avg_rate:.1f}%"
    })
    print(f"   ✓ Minimax    | vs Random: {rate_rand:5.1f}% | vs Heuristic: {rate_heu:5.1f}%")

    # =======================================================
    # BẢNG TỔNG KẾT
    # =======================================================
    print("\n" + "=" * 76)
    print("                 BẢNG XẾP HẠNG TOÀN BỘ CÁC AGENT (CONNECT 4 - 8x9)")
    print("=" * 76)
    print(f" {'Tên Tác Tử':<18} | {'vs Random':<18} | {'vs Heuristic':<18} | {'Trung Bình':<10} ")
    print("-" * 76)
    for r in results:
        sep = ""
        if r["name"] == "Ensemble(16 PPO)":
            print("-" * 76)
        print(f" {r['name']:<18} | {r['vs_rand']:<18} | {r['vs_heu']:<18} | {r['avg']:<10} ")
    print("=" * 76 + "\n")


def run_benchmark(num_games=1000):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.join(base_dir, "saved_agents_8x9")

    # 1. Nạp Ensemble 16 PPO Agents
    agent_files = [f for f in os.listdir(agents_dir) if f.startswith("agent") and f.endswith(".zip")] if os.path.exists(agents_dir) else []
    agent_files.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))

    if not agent_files:
        print("[!] Chưa có model trong saved_agents_8x9!")
        return

    print(f"\n[*] Đang nạp {len(agent_files)} PPO agents từ {agents_dir}...")
    players = [PPO.load(os.path.join(agents_dir, os.path.splitext(f)[0])) for f in agent_files]
    ensemble_player = EnsembleAgent(players)
    print(f"[✓] Đã nạp thành công EnsembleAgent ({len(players)} models)!\n")

    # ==========================================
    # BÀI TEST 1: ĐẤU VỚI RANDOM (TỶ LỆ THẮNG GỐC)
    # ==========================================
    print(f"--- BÀI TEST 1: Ensemble Agent vs RANDOM OPPONENT ({num_games} ván) ---")
    wins_random = 0
    for _ in tqdm(range(num_games), desc="Testing vs Random"):
        if test_one_game(ensemble_player, opponent=None):
            wins_random += 1

    winrate_random = wins_random / num_games
    print(f"\n>> Win rate vs Random: {winrate_random:.4f} ({winrate_random * 100:.2f}%)\n")

    # ==========================================
    # BÀI TEST 2: ĐẤU VỚI HEURISTIC AGENT
    # ==========================================
    heuristic_opp = HeuristicAgent(c=0)
    test_games_heu = min(num_games, 200) # Test 200 ván với Heuristic
    print(f"--- BÀI TEST 2: Ensemble Agent vs HEURISTIC AGENT ({test_games_heu} ván) ---")
    wins_heu = 0
    for _ in tqdm(range(test_games_heu), desc="Testing vs Heuristic"):
        if test_one_game(ensemble_player, opponent=heuristic_opp):
            wins_heu += 1

    winrate_heu = wins_heu / test_games_heu
    print(f"\n>> Win rate vs Heuristic: {winrate_heu:.4f} ({winrate_heu * 100:.2f}%)\n")

    # ==========================================
    # BÀI TEST 3: TIẾN HÓA RL (AGENT 15 VS AGENT 0)
    # ==========================================
    winrate_evo = None
    if len(players) >= 2:
        first_agent = players[0]   # Agent 0 (thế hệ đầu)
        last_agent = players[-1]   # Agent 15 (thế hệ đã học sâu nhất)
        test_games_evo = min(num_games, 100)
        print(f"--- BÀI TEST 3: Agent Cuối ({agent_files[-1]}) vs Agent Đầu ({agent_files[0]}) ({test_games_evo} ván) ---")
        wins_evo = 0
        for _ in tqdm(range(test_games_evo), desc="Testing Evolution"):
            if test_one_game(last_agent, opponent=first_agent):
                wins_evo += 1
        winrate_evo = wins_evo / test_games_evo
        print(f"\n>> Win rate Agent Cuối vs Agent Đầu: {winrate_evo:.4f} ({winrate_evo * 100:.2f}%)\n")

    # ==========================================
    # TỔNG KẾT
    # ==========================================
    print("=" * 60)
    print("           TỔNG KẾT HIỆU NĂNG CÁC ĐỐI THỦ (BENCHMARK)")
    print("=" * 60)
    print(f"1. Đối đầu RANDOM (Đi bừa)         : {winrate_random * 100:.2f}%")
    print(f"2. Đối đầu HEURISTIC (Luật cứng)   : {winrate_heu * 100:.2f}%")
    if winrate_evo is not None:
        print(f"3. Đối đầu EVOLUTION ({agent_files[-1]} vs {agent_files[0]}): {winrate_evo * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["levels", "--levels", "-l"]:
        from test_8_levels import benchmark_all_8_levels
        games = 10
        if len(sys.argv) > 2:
            try:
                games = int(sys.argv[2])
            except ValueError:
                pass
        benchmark_all_8_levels(num_games=games)
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ["all", "--all", "-a"]:
        games = 15
        if len(sys.argv) > 2:
            try:
                games = int(sys.argv[2])
            except ValueError:
                pass
        test_all_agents(games_per_agent=games)
    else:
        n = 100
        if len(sys.argv) > 1:
            try:
                n = int(sys.argv[1])
            except ValueError:
                pass
        run_benchmark(num_games=n)

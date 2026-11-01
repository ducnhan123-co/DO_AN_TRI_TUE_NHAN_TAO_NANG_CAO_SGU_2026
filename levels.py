"""
levels.py - Thang 8 cấp độ khó của AI Connect 4.

| Cấp | Phương pháp                          | Cần train? | Ghi chú |
|-----|--------------------------------------|------------|---------|
| 1   | Random                               | không      | đi ngẫu nhiên |
| 2   | Heuristic                            | không      | chấm điểm bàn cờ sau 1 nước, thắng/chặn ngay |
| 3   | Minimax                              | không      | negamax + alpha-beta, độ sâu 3 |
| 4   | Monte Carlo (MCTS thuần)             | không      | UCT + ván giả lập ngẫu nhiên, 4000 mô phỏng |
| 5   | Học có giám sát                      | có         | mạng bắt chước minimax độ sâu 4 |
| 6   | Học tăng cường                       | có         | MaskablePPO tự chơi |
| 7   | AlphaZero (100 mô phỏng MCTS)        | có         | mạng policy+value + MCTS PUCT |
| 8   | AlphaZero (800 mô phỏng MCTS)        | có         | cùng mạng với cấp 7, tìm kiếm sâu hơn |
"""
import os

from c4core import COLS, ROWS, HeuristicPlayer, MinimaxPlayer, RandomPlayer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LEVEL_NAMES = {
    1: "Random",
    2: "Heuristic",
    3: "Minimax",
    4: "Monte Carlo (MCTS)",
    5: "Học có giám sát",
    6: "Học tăng cường (PPO)",
    7: "AlphaZero (100 mô phỏng)",
    8: "AlphaZero (800 mô phỏng)",
}
NEEDS_TRAINING = {5: True, 6: True, 7: True, 8: True}
MINIMAX_DEPTH = 3
MCTS_SIMS = 4000
AZ_SIMS = {7: 100, 8: 800}

_cache = {}


def level_dir(rows=ROWS, cols=COLS):
    return os.path.join(BASE_DIR, "levels", f"{rows}x{cols}")


def level_file(n, rows=ROWS, cols=COLS):
    name = {5: "level5_supervised.npz", 6: "level6_rl.zip", 7: "level7_8_alphazero.npz", 8: "level7_8_alphazero.npz"}[n]
    return os.path.join(level_dir(rows, cols), name)


def is_ready(n):
    return n not in NEEDS_TRAINING or os.path.exists(level_file(n))


def make_level(n, seed=None):
    """Tạo người chơi (có hàm act(me, opp)) cho cấp n. Ném FileNotFoundError nếu cấp cần train mà chưa có model."""
    if n not in LEVEL_NAMES:
        raise ValueError(f"Cấp độ phải từ 1 đến 8 (nhận {n})")
    if n == 1:
        return RandomPlayer(seed)
    if n == 2:
        return HeuristicPlayer(seed=seed)
    if n == 3:
        return MinimaxPlayer(MINIMAX_DEPTH, seed=seed)
    if n == 4:
        from mcts import MCTSPlayer
        return MCTSPlayer(MCTS_SIMS, seed=seed)
    path = level_file(n)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa train cấp {n} cho bàn {ROWS}x{COLS}: chạy `python train_levels.py` "
                                f"(thiếu {os.path.relpath(path, BASE_DIR)})")
    if n == 6:
        from c4env import TorchPlayer, check_board, load_model
        key = ("rl", path)
        if key not in _cache:
            _cache[key] = check_board(load_model(path))
        return TorchPlayer(_cache[key].policy, deterministic=True, flip_avg=True, name="level6_rl")
    from mcts import AZPlayer, NpPV, SupervisedPlayer
    key = ("np", path)
    if key not in _cache:
        _cache[key] = NpPV.load(path)
    if n == 5:
        return SupervisedPlayer(_cache[key], name="level5_supervised")
    return AZPlayer(_cache[key], sims=AZ_SIMS[n], seed=seed, name=f"level{n}_alphazero{AZ_SIMS[n]}")


def describe(n):
    return f"CẤP {n}: {LEVEL_NAMES[n]}"

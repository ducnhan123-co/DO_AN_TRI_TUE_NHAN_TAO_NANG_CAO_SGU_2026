"""
levels.py - Thang 8 cấp độ khó của AI Connect 4.

| Cấp | Phương pháp                          | Cần train? | Ghi chú |
|-----|--------------------------------------|------------|---------|
| 1   | Random                               | không      | đi ngẫu nhiên |
| 2   | Heuristic                            | không      | chấm điểm bàn cờ sau 1 nước, thắng/chặn ngay |
| 3   | Minimax                              | không      | negamax + alpha-beta, độ sâu 3 |
| 4   | Monte Carlo (MCTS thuần)             | không      | UCT + ván giả lập ngẫu nhiên, 4000 mô phỏng |
| 5   | Học có giám sát                      | có         | mạng bắt chước minimax độ sâu 4 |
| 6   | Học tăng cường                       | có         | MaskablePPO chơi với bot cố định, khởi tạo từ cấp 5 (chưa tự chơi vì ngân sách < mốc chụp 25.000 bước) |
| 7   | AlphaZero (100 mô phỏng MCTS)        | có         | mạng policy+value + MCTS PUCT |
| 8   | AlphaZero (800 mô phỏng MCTS)        | có         | cùng mạng với cấp 7, tìm kiếm sâu hơn |
"""
import os

from c4.core import COLS, ROWS, HeuristicPlayer, MinimaxPlayer, RandomPlayer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục gốc dự án

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


def make_level(n, seed=None, **opts):
    """Tạo người chơi (có hàm act(me, opp)) cho cấp n. Ném FileNotFoundError nếu cấp cần train mà chưa có model.

    opts (tuỳ chọn, dùng cho thí nghiệm / ablation; bỏ trống = cấu hình mặc định của cấp):
      cấp 3 : depth, time_limit(giây/nước)          cấp 4 : sims, time_limit
      cấp 5 : shield, flip_avg                       cấp 6 : shield, flip_avg, search_depth
      cấp 7/8 : sims, time_limit, shield
    """
    if n not in LEVEL_NAMES:
        raise ValueError(f"Cấp độ phải từ 1 đến 8 (nhận {n})")
    opts = dict(opts)
    if n == 1:
        return RandomPlayer(seed)
    if n == 2:
        return HeuristicPlayer(seed=seed)
    if n == 3:
        return MinimaxPlayer(opts.get("depth", MINIMAX_DEPTH), seed=seed, time_limit=opts.get("time_limit"))
    if n == 4:
        from c4.mcts import MCTSPlayer
        return MCTSPlayer(opts.get("sims", MCTS_SIMS), seed=seed, time_limit=opts.get("time_limit"))
    path = level_file(n)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa train cấp {n} cho bàn {ROWS}x{COLS}: xem docs/HUONG_DAN_TRAIN.md "
                                f"(thiếu {os.path.relpath(path, BASE_DIR)})")
    if n == 6:
        from c4.env import TorchPlayer, check_board, load_model
        key = ("rl", path)
        if key not in _cache:
            _cache[key] = check_board(load_model(path))
        return TorchPlayer(_cache[key].policy, deterministic=True, flip_avg=opts.get("flip_avg", True),
                           shield=opts.get("shield", True), search_depth=opts.get("search_depth", 0),
                           seed=seed, name="level6_rl")
    from c4.mcts import AZPlayer, NpPV, SupervisedPlayer
    key = ("np", path)
    if key not in _cache:
        _cache[key] = NpPV.load(path)
    if n == 5:
        return SupervisedPlayer(_cache[key], flip_avg=opts.get("flip_avg", True), shield=opts.get("shield", True),
                                name="level5_supervised")
    sims = opts.get("sims", AZ_SIMS[n])
    return AZPlayer(_cache[key], sims=sims, seed=seed, name=f"level{n}_alphazero{sims}",
                    time_limit=opts.get("time_limit"), shield=opts.get("shield", True))


def describe(n):
    return f"CẤP {n}: {LEVEL_NAMES[n]}"

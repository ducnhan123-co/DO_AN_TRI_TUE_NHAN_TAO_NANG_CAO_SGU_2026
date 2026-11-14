"""
Lab 04 - Bai 6: Cac bien the Hill Climbing cho bai toan n-Queens
Task 1: Steepest-ascent Hill Climbing
Task 2: Inefficient Stochastic Hill Climbing (chon ngau nhien trong MOI nuoc cai thien)
Task 3: Efficient Stochastic Hill Climbing (first-choice: sinh 1 lang gieng ngau nhien)

Chay:  ./.venv/bin/python bai6_nqueens_hc.py
Sinh:  images/bai6_*.png, output/bai6_runs.csv, output/bai6_summary.json
"""
import csv
import json
import os
import time
from dataclasses import asdict, dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "images")
OUT = os.path.join(BASE, "output")
os.makedirs(IMG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SEED = 2024
RUNS = 100
ALGOS = ("Steepest-ascent", "Inefficient Stochastic", "Efficient Stochastic")


def comb2(k: int) -> int:
    return k * (k - 1) // 2


def conflicts(board: np.ndarray) -> int:
    """Ham muc tieu: so cap hau tan cong nhau (cung hang / cheo). O(n)."""
    n = len(board)
    row_cnt = [0] * n
    diag1_cnt = [0] * 2 * n
    diag2_cnt = [0] * 2 * n
    for i in range(n):
        row_cnt[board[i]] += 1
        diag1_cnt[i + board[i]] += 1
        diag2_cnt[i - board[i] + n] += 1
    return sum(map(comb2, row_cnt + diag1_cnt + diag2_cnt))


def random_board(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def num_moves(n: int) -> int:
    return n * (n - 1)


def all_moves(board: np.ndarray):
    """Sinh toan bo N = n(n-1) nuoc di: (cot, hang moi), bo nuoc di dung yen."""
    n = len(board)
    for c in range(n):
        for r in range(n):
            if r != board[c]:
                yield c, r


def apply_move(board: np.ndarray, move: tuple[int, int]) -> np.ndarray:
    c, r = move
    nb = board.copy()
    nb[c] = r
    return nb


def has_plateau_neighbor(board: np.ndarray) -> bool:
    """Chan doan: co lang gieng cung gia tri h khong (khong tinh vao evals)."""
    h = conflicts(board)
    for move in all_moves(board):
        if conflicts(apply_move(board, move)) == h:
            return True
    return False


@dataclass
class SearchResult:
    board: list[int]
    final_h: int
    steps: int
    evals: int
    path: list[int]
    solved: bool
    trials: int
    max_steps_hit: bool


# ---------------------------------------------------------------------------
# Task 1 - Steepest-ascent
# ---------------------------------------------------------------------------
def steepest_ascent(board: np.ndarray, max_steps: int = 1000) -> SearchResult:
    board = board.copy()
    h = conflicts(board)
    evals = 1
    steps = 0
    path = [h]
    while steps < max_steps and h > 0:
        best_h, best_move = h, None
        for move in all_moves(board):
            nh = conflicts(apply_move(board, move))
            evals += 1
            if nh < best_h:
                best_h, best_move = nh, move
        if best_move is None:
            break
        board = apply_move(board, best_move)
        h = best_h
        steps += 1
        path.append(h)
    return SearchResult(board.tolist(), h, steps, evals, path, h == 0,
                        evals, steps >= max_steps)


# ---------------------------------------------------------------------------
# Task 2 - Inefficient Stochastic HC
# ---------------------------------------------------------------------------
def inefficient_stochastic(board: np.ndarray, rng: np.random.Generator,
                           max_steps: int = 1000) -> SearchResult:
    board = board.copy()
    h = conflicts(board)
    evals = 1
    steps = 0
    path = [h]
    while steps < max_steps and h > 0:
        improving = []
        for move in all_moves(board):
            nh = conflicts(apply_move(board, move))
            evals += 1
            if nh < h:
                improving.append((nh, move))
        if not improving:
            break
        best_h, move = improving[int(rng.integers(len(improving)))]
        board = apply_move(board, move)
        h = best_h
        steps += 1
        path.append(h)
    return SearchResult(board.tolist(), h, steps, evals, path, h == 0,
                        evals, steps >= max_steps)


# ---------------------------------------------------------------------------
# Task 3 - Efficient Stochastic HC (first-choice)
# ---------------------------------------------------------------------------
def efficient_stochastic(board: np.ndarray, rng: np.random.Generator,
                         patience: int, max_trials: int = 200000) -> SearchResult:
    """Moi lan thu sinh DUNG MOT lang gieng ngau nhien (O(1)); nhan ngay neu
    tot hon. Dung sau `patience` lan thu lien tiep khong cai thien.
    patience = max(50, 3N), N = n(n-1)."""
    board = board.copy()
    h = conflicts(board)
    evals = 1
    steps = 0
    trials = 0
    fail_streak = 0
    path = [h]
    n = len(board)
    while trials < max_trials and h > 0 and fail_streak < patience:
        c = int(rng.integers(n))
        r = int(rng.integers(n - 1))
        if r >= board[c]:
            r += 1
        nb = board.copy()
        nb[c] = r
        nh = conflicts(nb)
        evals += 1
        trials += 1
        if nh < h:
            board, h = nb, nh
            steps += 1
            fail_streak = 0
            path.append(h)
        else:
            fail_streak += 1
    return SearchResult(board.tolist(), h, steps, evals, path, h == 0,
                        trials, trials >= max_trials)


# ---------------------------------------------------------------------------
# Ve ban co
# ---------------------------------------------------------------------------
def draw_board(ax, board, title: str) -> None:
    n = len(board)
    display = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 != 0:
                display[i, j] = 1
    cmap = mcolors.ListedColormap(["white", "gray"])
    ax.imshow(display, cmap=cmap, norm=mcolors.BoundaryNorm(range(3), cmap.N))
    ax.set_xticks([]); ax.set_yticks([])
    for j in range(n):
        ax.text(j, board[j], "\u265B", fontsize=26, ha="center", va="center")
    ax.set_title(title, fontsize=10)


@dataclass
class Stats:
    runs: int
    success_rate_pct: float
    avg_final_h: float
    min_final_h: int
    max_final_h: int
    median_final_h: float
    avg_steps: float
    avg_evals: float
    avg_time_ms: float
    plateau_share_failed_pct: float
    avg_trials: float


# ---------------------------------------------------------------------------
# 1. Chay 100 lan moi thuat toan, cung 100 ban co dau cho ca ba
# ---------------------------------------------------------------------------
rows = []
stats: dict[tuple[int, str], Stats] = {}
rep_start: np.ndarray | None = None
rep_res: dict[str, SearchResult] = {}
rng_master = np.random.default_rng(SEED)

for n in (4, 8):
    N = num_moves(n)
    patience = max(50, 3 * N)
    starts = [random_board(n, rng_master) for _ in range(RUNS)]
    if n == 8:
        rep_start = next((b for b in starts if conflicts(b) >= 5), starts[0])
        rep_res["Steepest-ascent"] = steepest_ascent(rep_start)
        rep_res["Inefficient Stochastic"] = inefficient_stochastic(
            rep_start, np.random.default_rng([SEED, 8, 999, 1]))
        rep_res["Efficient Stochastic"] = efficient_stochastic(
            rep_start, np.random.default_rng([SEED, 8, 999, 2]), patience)

    for algo_idx, algo in enumerate(ALGOS):
        for i, start in enumerate(starts):
            rng = np.random.default_rng([SEED, n, i, algo_idx])
            t0 = time.perf_counter()
            if algo == "Steepest-ascent":
                res = steepest_ascent(start)
            elif algo == "Inefficient Stochastic":
                res = inefficient_stochastic(start, rng)
            else:
                res = efficient_stochastic(start, rng, patience)
            elapsed = (time.perf_counter() - t0) * 1000
            plateau = (not res.solved) and has_plateau_neighbor(np.array(res.board))
            rows.append({
                "n": n, "algorithm": algo, "run": i,
                "start_h": int(conflicts(start)), "final_h": res.final_h,
                "steps": res.steps, "evals": res.evals,
                "time_ms": round(elapsed, 4), "solved": int(res.solved),
                "plateau": int(plateau),
                "patience": patience if algo == "Efficient Stochastic" else "",
                "trials": res.trials,
            })

    for algo in ALGOS:
        sub = [r for r in rows if r["n"] == n and r["algorithm"] == algo]
        final = np.array([r["final_h"] for r in sub])
        stats[(n, algo)] = Stats(
            runs=len(sub),
            success_rate_pct=round(100 * float(np.mean(final == 0)), 1),
            avg_final_h=round(float(final.mean()), 3),
            min_final_h=int(final.min()),
            max_final_h=int(final.max()),
            median_final_h=float(np.median(final)),
            avg_steps=round(float(np.mean([r["steps"] for r in sub])), 2),
            avg_evals=round(float(np.mean([r["evals"] for r in sub])), 1),
            avg_time_ms=round(float(np.mean([r["time_ms"] for r in sub])), 3),
            plateau_share_failed_pct=round(
                100 * float(np.mean([r["plateau"] for r in sub if r["final_h"] != 0]))
                if any(r["final_h"] != 0 for r in sub) else 0.0, 1),
            avg_trials=round(float(np.mean([r["trials"] for r in sub])), 1),
        )

with open(os.path.join(OUT, "bai6_runs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

summary = {
    "seed": SEED,
    "runs_per_setting": RUNS,
    "patience": {str(n): max(50, 3 * num_moves(n)) for n in (4, 8)},
    "stats": {str(n): {a: asdict(stats[(n, a)]) for a in ALGOS} for n in (4, 8)},
    "representative_n8": {
        "start": rep_start.tolist() if rep_start is not None else [],
        "start_h": int(conflicts(rep_start)) if rep_start is not None else None,
        "results": {a: asdict(r) for a, r in rep_res.items()},
    },
}
with open(os.path.join(OUT, "bai6_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 2. Hinh: hoi tu tren cung ban dau (n = 8)
# ---------------------------------------------------------------------------
assert rep_start is not None
plt.figure(figsize=(7.4, 4.6))
markers = {"Steepest-ascent": "o", "Inefficient Stochastic": "s",
           "Efficient Stochastic": "^"}
for algo in ALGOS:
    p = rep_res[algo].path
    plt.plot(range(len(p)), p, marker=markers[algo], linewidth=1.8,
             markersize=4, label=f"{algo} (h cuoi = {p[-1]})")
plt.xlabel("Số nước đi đã chấp nhận (accepted moves)")
plt.ylabel("Số cặp hậu xung đột h(board)")
plt.title(f"Hội tụ trên cùng bàn đầu n = 8, h0 = {conflicts(rep_start)}")
plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(IMG, "bai6_convergence_n8.png"), dpi=180)
plt.close()

# ---------------------------------------------------------------------------
# 3. Hinh: ban co truoc / sau
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.2))
draw_board(axes[0, 0], rep_start, f"Bàn đầu: h = {conflicts(rep_start)}")
for ax, algo in zip(axes.flat[1:], ALGOS):
    res = rep_res[algo]
    draw_board(ax, np.array(res.board), f"{algo}\nh = {res.final_h} ({res.steps} bước)")
fig.suptitle("n = 8: cùng một bàn cờ xuất phát", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "bai6_boards_n8.png"), dpi=180)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Hinh: ti le thanh cong va so lan danh gia
# ---------------------------------------------------------------------------
colors = ["#4472c4", "#ed7d31", "#70ad47"]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
x = np.arange(len(ALGOS))
width = 0.36
for k, n in enumerate((4, 8)):
    vals = [stats[(n, a)].success_rate_pct for a in ALGOS]
    axes[0].bar(x + (k - 0.5) * width, vals, width, label=f"n = {n}",
                color=colors[k], alpha=0.9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Steepest", "Ineff. Stoch.", "Eff. Stoch."], fontsize=9)
axes[0].set_ylabel("Tỉ lệ đạt h = 0 (%)"); axes[0].set_ylim(0, 105)
axes[0].set_title("Tỉ lệ thành công (100 lần chạy)"); axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

for k, n in enumerate((4, 8)):
    vals = [stats[(n, a)].avg_evals for a in ALGOS]
    axes[1].bar(x + (k - 0.5) * width, vals, width, label=f"n = {n}",
                color=colors[k], alpha=0.9)
axes[1].set_yscale("log")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Steepest", "Ineff. Stoch.", "Eff. Stoch."], fontsize=9)
axes[1].set_ylabel("Số lần gọi conflicts() trung bình (log)")
axes[1].set_title("Chi phí đánh giá hàm mục tiêu"); axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "bai6_success_evals.png"), dpi=180)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Hinh: phan bo h cuoi tren n = 8
# ---------------------------------------------------------------------------
data = [[r["final_h"] for r in rows if r["n"] == 8 and r["algorithm"] == a]
        for a in ALGOS]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bp = ax.boxplot(data, patch_artist=True,
                tick_labels=["Steepest", "Ineff. Stoch.", "Eff. Stoch."])
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col); patch.set_alpha(0.65)
ax.set_ylabel("Số xung đột cuối h")
ax.set_title("Phân bố h cuối trên 100 bàn cờ n = 8")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "bai6_final_h_box.png"), dpi=180)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. In bao cao ngan ra console
# ---------------------------------------------------------------------------
print("=== TONG HOP BAI 6 (100 lan chay moi o) ===")
for n in (4, 8):
    print(f"\n--- n = {n}, N = n(n-1) = {num_moves(n)}, "
          f"patience = {max(50, 3 * num_moves(n))} ---")
    for a in ALGOS:
        s = stats[(n, a)]
        print(f"{a:24s} success={s.success_rate_pct:5.1f}%  "
              f"h_cuoi_tb={s.avg_final_h:.2f}  steps_tb={s.avg_steps:.2f}  "
              f"evals_tb={s.avg_evals:.1f}  time_tb={s.avg_time_ms:.3f} ms  "
              f"plateau/failed={s.plateau_share_failed_pct:.1f}%")
print("\n--- Lan chay dai dien n = 8 ---")
print(f"Ban dau h = {conflicts(rep_start)}: {rep_start.tolist()}")
for a in ALGOS:
    r = rep_res[a]
    print(f"{a:24s} h={r.final_h}  steps={r.steps}  evals={r.evals}  "
          f"solved={r.solved}  trials={r.trials}")

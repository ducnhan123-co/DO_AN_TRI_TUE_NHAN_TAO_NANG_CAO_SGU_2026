"""
Bài 8: Tổng hợp thí nghiệm và phân tích (N-Queens, Task 6 + Advanced)
- Trạng thái: vector q (1 hậu/cột); hàm mục tiêu: số cặp xung đột (minimize, optimum = 0)
- 4 thuật toán: Steepest-ascent HC, Inefficient Stochastic HC, Efficient Stochastic HC (First-choice), SA
- Toán tử chuẩn: chuyển 1 hậu sang hàng khác trong cùng cột
- Advanced: single-step, column-swap, dual-queen, adaptive
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import time, os, random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bai8_output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1234)
random.seed(1234)

# ================= Helpers (giữ nguyên notebook gốc) =================
def comb2(n): return n*(n-1)//2

def conflicts(board):
    n = len(board)
    h = [0]*n; d1 = [0]*2*n; d2 = [0]*2*n
    for i in range(n):
        h[int(board[i])] += 1
        d1[i+int(board[i])] += 1
        d2[i-int(board[i])+n] += 1
    return sum(map(comb2, h+d1+d2))

def random_board(n):
    return rng.integers(0, n, size=n).astype(int)

def save_board(board, title, path, fontsize=None):
    n = len(board)
    if fontsize is None: fontsize = max(10, 48 - 2*n)
    disp = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if (i+j) % 2 != 0: disp[i, j] = 1
    cmap = colors.ListedColormap(['white', 'gray'])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(disp, cmap=cmap, norm=colors.BoundaryNorm(range(3), cmap.N))
    ax.set_xticks([]); ax.set_yticks([])
    for j in range(n):
        ax.text(j, int(board[j]), "♛", fontsize=fontsize, ha='center', va='center')
    ax.set_title(f"{title} ({conflicts(board)} conflicts)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()

# ================= Toán tử lân cận =================
def all_standard_neighbors(board):
    """Toàn bộ lân cận chuẩn: mỗi cột -> n-1 hàng khác. |N| = n(n-1)."""
    n = len(board); out = []
    for c in range(n):
        for r in range(n):
            if r != board[c]:
                nb = board.copy(); nb[c] = r; out.append(nb)
    return out

def sample_standard(board):
    n = len(board); c = rng.integers(0, n)
    r = rng.integers(0, n-1);  r = r if r != board[c] else n-1
    nb = board.copy(); nb[c] = r; return nb

def sample_single_step(board):
    """1 hậu nhích ±1 ô. |N| ≈ 2n."""
    n = len(board)
    if n < 2: return board.copy()
    for _ in range(20):
        c = rng.integers(0, n)
        d = rng.choice([-1, 1])
        if 0 <= board[c]+d < n:
            nb = board.copy(); nb[c] += d; return nb
    return sample_standard(board)

def sample_col_swap(board):
    """Đổi chỗ 2 cột. |N| = C(n,2)."""
    n = len(board); i, j = sorted(rng.choice(n, 2, replace=False))
    nb = board.copy(); nb[i], nb[j] = nb[j], nb[i]; return nb

def sample_dual_queen(board):
    """Di chuyển đồng thời 2 hậu. |N| = C(n,2)(n-1)^2 (rất lớn)."""
    n = len(board); i, j = sorted(rng.choice(n, 2, replace=False))
    nb = board.copy()
    for c in (i, j):
        r = rng.integers(0, n-1); nb[c] = r if r != nb[c] else n-1
    return nb

def queen_conflict_counts(board):
    n = len(board); cnt = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if board[i] == board[j] or abs(int(board[i])-int(board[j])) == j-i:
                cnt[i] += 1; cnt[j] += 1
    return cnt

def sample_adaptive(board):
    """70%: di chuyển 1 hậu đang xung đột nhiều nhất; 30%: move chuẩn ngẫu nhiên."""
    n = len(board)
    if rng.random() < 0.7:
        cnt = queen_conflict_counts(board)
        m = max(cnt)
        if m > 0:
            cand = [i for i, c in enumerate(cnt) if c == m]
            c = rng.choice(cand)
            r = rng.integers(0, n-1); r = r if r != board[c] else n-1
            nb = board.copy(); nb[c] = r; return nb
    return sample_standard(board)

OPERATORS = {
    "standard":   (sample_standard,   lambda n: n*(n-1)),
    "single-step":(sample_single_step,lambda n: 2*n),
    "column-swap":(sample_col_swap,   lambda n: n*(n-1)//2),
    "dual-queen": (sample_dual_queen, lambda n: (n*(n-1)//2)*(n-1)**2),
    "adaptive":   (sample_adaptive,   lambda n: n*(n-1)),
}

# ================= Thuật toán (Tasks 1-5) =================
def steepest_hc(n, init=None, max_iters=500):
    b = random_board(n) if init is None else np.array(init, dtype=int).copy()
    cur = conflicts(b); hist = [cur]; ev = 0
    for _ in range(max_iters):
        best, bestc = None, cur
        for nb in all_standard_neighbors(b):
            c = conflicts(nb); ev += 1
            if c < bestc: bestc, best = c, nb
        if best is None: break
        b, cur = best, bestc; hist.append(cur)
        if cur == 0: break
    return {"board": b, "conflicts": cur, "history": hist, "evals": ev}

def ineff_stochastic_hc(n, init=None, max_iters=500):
    b = random_board(n) if init is None else np.array(init, dtype=int).copy()
    cur = conflicts(b); hist = [cur]; ev = 0
    for _ in range(max_iters):
        up = []
        for nb in all_standard_neighbors(b):
            c = conflicts(nb); ev += 1
            if c < cur: up.append((nb, c))
        if not up: break
        b, cur = random.choice(up); hist.append(cur)
        if cur == 0: break
    return {"board": b, "conflicts": cur, "history": hist, "evals": ev}

def efficient_stochastic_hc(n, init=None, sampler=sample_standard, neigh_size=None,
                            no_improve_factor=5, max_iters=50000):
    """First-choice: 1 lân cận ngẫu nhiên/lần; dừng sau x = factor*|N| lần không cải thiện.
    Lý luận x: coupon-collector — quét gần hết |N| lân cận cần ~|N|·ln|N| mẫu;
    factor=5 bao phủ chắc chắn (|N|·ln|N| < 5|N| với |N| ≤ 380)."""
    b = random_board(n) if init is None else np.array(init, dtype=int).copy()
    N = neigh_size(n) if neigh_size else n*(n-1)
    x = min(no_improve_factor*N, 5000)
    cur = conflicts(b); hist = [cur]; ev = 0; stale = 0
    while ev < max_iters and stale < x:
        nb = sampler(b); c = conflicts(nb); ev += 1
        if c < cur: b, cur = nb, c; hist.append(cur); stale = 0
        else: stale += 1
        if cur == 0: break
    return {"board": b, "conflicts": cur, "history": hist, "evals": ev}

def simulated_annealing(n, init=None, T0=3.0, alpha=0.99, Tmin=1e-4, max_iters=10000):
    b = random_board(n) if init is None else np.array(init, dtype=int).copy()
    cur = conflicts(b); best, bestc = b.copy(), cur
    hist = [cur]; T = T0; it = 0
    while T > Tmin and it < max_iters:
        nb = sample_standard(b); c = conflicts(nb)
        dE = c - cur
        if dE <= 0 or rng.random() < np.exp(-dE/T):
            b, cur = nb, c
            if cur < bestc: best, bestc = b.copy(), cur
        hist.append(cur); T *= alpha; it += 1
        if bestc == 0: break
    return {"board": best, "conflicts": bestc, "history": hist, "evals": it}

def restart_wrapper(algo_fn, n, restarts=100, **kw):
    best, bestc, t = None, 10**9, 0.0
    t0 = time.time()
    for _ in range(restarts):
        r = algo_fn(n, random_board(n), **kw)
        if r["conflicts"] < bestc: best, bestc = r["board"], r["conflicts"]
        if bestc == 0: break
    return {"board": best, "conflicts": bestc, "time": time.time()-t0}

ALGOS = {
    "Steepest asc. HC": steepest_hc,
    "Ineff. Stochastic HC": ineff_stochastic_hc,
    "Efficient Stochastic HC": efficient_stochastic_hc,
    "Simulated Annealing": simulated_annealing,
}

if __name__ == "__main__":
    import pandas as pd
    print("=== Bài 8: N-Queens Task 6 — So sánh (mỗi ô 100 lần chạy) ===")
    N_RUNS = 100
    rows = []
    for n in [4, 8]:
        for name, fn in ALGOS.items():
            ts, cs, opt = [], [], 0
            for _ in range(N_RUNS):
                t0 = time.time()
                r = fn(n, random_board(n))
                ts.append((time.time()-t0)*1000); cs.append(r["conflicts"])
                if r["conflicts"] == 0: opt += 1
            rows.append({"algo": name, "n": n, "avg_time_ms": round(float(np.mean(ts)), 3),
                         "avg_conflicts": round(float(np.mean(cs)), 3),
                         "pct_optimal": opt})
            print(f"n={n} {name:22s} time={np.mean(ts):7.3f}ms  avg_conf={np.mean(cs):.3f}  optimal={opt}%")
    pd.DataFrame(rows).to_csv(f"{OUT}/task6_comparison.csv", index=False)

    # Restart nhanh: steepest + 100 restarts trên n=8
    rr = restart_wrapper(steepest_hc, 8, restarts=100)
    print(f"\nSteepest + 100 restarts (n=8): {rr['conflicts']} conflicts, {rr['time']*1000:.1f}ms")

    # ---- Hội tụ: 1 run đại diện / thuật toán trên n=8 (cùng init) ----
    print("\n--- Convergence (n=8, cùng bàn khởi đầu) ---")
    init8 = random_board(8)
    print("Init:", list(init8), "conflicts:", conflicts(init8))
    save_board(init8, "Bàn đầu n=8", f"{OUT}/board_init_n8.png")
    reps = {}
    for name, fn in ALGOS.items():
        r = fn(8, init8.copy())
        reps[name] = r
        print(f"{name:22s} final={r['conflicts']}  iters={len(r['history'])}  evals={r['evals']}")
    save_board(reps["Steepest asc. HC"]["board"], "Steepest sau tối ưu n=8", f"{OUT}/board_steepest_n8.png")
    save_board(reps["Simulated Annealing"]["board"], "SA sau tối ưu n=8", f"{OUT}/board_sa_n8.png")
    plt.figure(figsize=(8, 5))
    for name, r in reps.items(): plt.plot(r["history"], label=name, marker="o", ms=3)
    plt.xlabel("Vòng lặp"); plt.ylabel("f(board) = số xung đột")
    plt.title("Hội tụ các thuật toán trên 8-Queens (cùng bàn đầu)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/convergence_n8.png", dpi=150); plt.close()

    # ---- Scalability n=4,8,12,16,20 (Steepest, Efficient, SA; 30 runs) ----
    print("\n--- Scalability (30 runs/ô) ---")
    srows = []
    for n in [4, 8, 12, 16, 20]:
        for name in ["Steepest asc. HC", "Efficient Stochastic HC", "Simulated Annealing"]:
            ts = []
            for _ in range(30):
                t0 = time.time(); ALGOS[name](n, random_board(n)); ts.append((time.time()-t0)*1000)
            srows.append({"algo": name, "n": n, "avg_time_ms": float(np.mean(ts))})
            print(f"n={n:3d} {name:22s} {np.mean(ts):9.3f}ms")
    sdf = pd.DataFrame(srows); sdf.to_csv(f"{OUT}/scalability.csv", index=False)
    plt.figure(figsize=(8, 5))
    slopes = {}
    for name in sdf["algo"].unique():
        d = sdf[sdf.algo == name].sort_values("n")
        plt.loglog(d["n"], d["avg_time_ms"], marker="o", label=name)
        slope, _ = np.polyfit(np.log(d["n"]), np.log(d["avg_time_ms"]), 1)
        slopes[name] = slope
    plt.xlabel("n (số hậu)"); plt.ylabel("Thời gian TB (ms)")
    plt.title("Scalability log-log: thời gian theo n")
    plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/scalability_loglog.png", dpi=150); plt.close()
    print("Độ dốc log-log (số mũ thực nghiệm):", {k: round(v, 2) for k, v in slopes.items()})

    # ---- Advanced: Efficient Stochastic HC × 5 toán tử × (n=8, 12) × 100 runs ----
    print("\n--- Advanced move operators (Efficient Stochastic HC, 100 runs) ---")
    arows = []
    curves = {}
    for n in [8, 12]:
        curves[n] = {}
        for op, (sampler, nsize_fn) in OPERATORS.items():
            finals, tts, succ = [], [], 0
            best_curves = []
            for _ in range(100):
                t0 = time.time()
                r = efficient_stochastic_hc(n, random_board(n), sampler=sampler,
                                            neigh_size=nsize_fn)
                dt = (time.time()-t0)*1000
                finals.append(r["conflicts"])
                if r["conflicts"] == 0: succ += 1; tts.append(dt)
                h = r["history"]; best_curves.append(h)
            L = max(map(len, best_curves))
            mat = np.array([h + [h[-1]]*(L-len(h)) for h in best_curves])
            curves[n][op] = mat.mean(axis=0)
            arows.append({"n": n, "operator": op, "neigh_size": nsize_fn(n),
                          "avg_final": round(float(np.mean(finals)), 3),
                          "pct_optimal": succ,
                          "avg_time_to_sol_ms": round(float(np.mean(tts)), 3) if tts else None})
            print(f"n={n:3d} {op:12s} |N|={nsize_fn(n):6d}  avg_final={np.mean(finals):.3f}  optimal={succ}%  t_sol={np.mean(tts) if tts else float('nan'):.3f}ms")
    pd.DataFrame(arows).to_csv(f"{OUT}/advanced_operators.csv", index=False)
    for n in [8, 12]:
        plt.figure(figsize=(8, 5))
        for op, c in curves[n].items(): plt.plot(c, label=op)
        plt.xlabel("Số eval"); plt.ylabel("f(board) TB (best-so-far)")
        plt.title(f"Chất lượng TB theo eval — n={n} (100 runs)")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{OUT}/advanced_curves_n{n}.png", dpi=150); plt.close()
        # histogram phân bố chất lượng cuối
        df_a = pd.DataFrame(arows); sub = df_a[df_a.n == n]
        plt.figure(figsize=(8, 4))
        ops = list(OPERATORS.keys())
        plt.bar([f"{o}\n({int(sub[sub.operator==o]['pct_optimal'].iloc[0])}% opt)" for o in ops],
                [float(sub[sub.operator == o]["avg_final"].iloc[0]) for o in ops])
        plt.ylabel("Số xung đột cuối TB"); plt.title(f"Phân bố chất lượng cuối — n={n}")
        plt.tight_layout(); plt.savefig(f"{OUT}/advanced_final_n{n}.png", dpi=150); plt.close()
    print(f"\nXong. Kết quả tại: {OUT}\nFiles:", os.listdir(OUT))

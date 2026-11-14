"""
Bài 7: Tìm kiếm cục bộ cho TSP
- Trạng thái: hoán vị các thành phố
- Hàm mục tiêu: tổng độ dài chu trình khép kín (minimize)
- Toán tử lân cận THỐNG NHẤT: 2-opt (đảo đoạn), giữ nguyên khi so sánh mọi thuật toán
- Các thuật toán: Steepest-ascent HC, Random Restarts, Stochastic HC, First-choice HC, SA (+ bonus GA)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math, random, time, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bai7_output")
os.makedirs(OUT, exist_ok=True)

# ============ Helper functions (giữ nguyên từ notebook gốc) ============
def random_tour(n):
    """Create a random tour"""
    tour = list(range(n))
    random.shuffle(tour)
    return tour

def random_tsp(n, seed=None):
    """Create random Euclidean TSP in 1x1 square."""
    from scipy.spatial.distance import pdist, squareform
    if seed is not None:
        np.random.seed(seed)
    pos = pd.DataFrame({"x": np.random.uniform(size=n), "y": np.random.uniform(size=n)})
    dist = squareform(pdist(pos))
    return {"pos": pos, "dist": dist}

def tour_length(tsp, tour):
    """Objective function: tổng độ dài chu trình khép kín."""
    if not isinstance(tour, list):
        tour = tour.tolist()
    tl = 0
    dist = tsp["dist"]
    for i in range(len(tour)-1):
        tl += dist[tour[i], tour[i+1]]
    tl += dist[tour[-1], tour[0]]
    return tl

def show_tsp(tsp, tour=None, title=None, savepath=None):
    pos = tsp["pos"]
    plt.figure(figsize=(5,5))
    plt.scatter(pos["x"], pos["y"], c="red", s=40, zorder=3)
    for i, (x, y) in pos.iterrows():
        plt.text(x+0.01, y+0.01, str(i), fontsize=9)
    if tour is not None:
        if not isinstance(tour, list):
            tour = tour.tolist()
        L = tour_length(tsp, tour)
        pos_ = pos.reindex(tour)
        pos_ = pd.concat([pos_, pos_.head(1)])
        plt.plot(pos_["x"], pos_["y"], "b-o", ms=4)
        plt.title(f"{title or ''} (L={L:.3f})")
    else:
        plt.title(title or "TSP cities")
    plt.axis("equal"); plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150); plt.close()
    else:
        plt.close()

# ============ Toán tử lân cận: 2-opt ============
def two_opt_move(tour, i, j):
    """Đảo đoạn tour[i:j+1]. VD [0 1 2 3 4], i=1,j=3 -> [0 3 2 1 4]."""
    assert 0 <= i < j < len(tour)
    return tour[:i] + list(reversed(tour[i:j+1])) + tour[j+1:]

def random_2opt_neighbor(tour):
    n = len(tour)
    i, j = sorted(random.sample(range(n), 2))
    # tránh đảo toàn bộ tour (tương đương tour đảo chiều, cùng độ dài)
    if i == 0 and j == n-1:
        return random_2opt_neighbor(tour)
    return two_opt_move(tour, i, j)

def all_2opt_neighbors(tour):
    """Liệt kê toàn bộ lân cận 2-opt: C(n,2)-1 ~ O(n^2)."""
    n = len(tour)
    neighs = []
    for i in range(n-1):
        for j in range(i+1, n):
            if i == 0 and j == n-1:
                continue
            neighs.append(two_opt_move(tour, i, j))
    return neighs

def swap_move(tour, i, j):
    t = tour.copy(); t[i], t[j] = t[j], t[i]; return t

# ============ 1. Steepest-ascent Hill Climbing ============
def steepest_ascent_hc(tsp, init_tour=None, max_iters=1000):
    tour = (init_tour.copy() if init_tour else random_tour(len(tsp["dist"])))
    cur = tour_length(tsp, tour)
    history = [cur]; n_evals = 0
    for _ in range(max_iters):
        best, best_L = None, cur
        for nb in all_2opt_neighbors(tour):
            L = tour_length(tsp, nb); n_evals += 1
            if L < best_L - 1e-12:
                best_L, best = L, nb
        if best is None:  # cực tiểu địa phương
            break
        tour, cur = best, best_L
        history.append(cur)
    return {"tour": tour, "length": cur, "history": history, "evals": n_evals}

# ============ 2. Random Restarts ============
def random_restart_hc(tsp, n_restarts=20, max_iters=1000):
    best, best_L, histories, total_evals = None, float("inf"), [], 0
    for _ in range(n_restarts):
        r = steepest_ascent_hc(tsp, random_tour(len(tsp["dist"])), max_iters)
        total_evals += r["evals"]
        histories.append(r["history"])
        if r["length"] < best_L:
            best_L, best = r["length"], r["tour"]
    return {"tour": best, "length": best_L, "histories": histories, "evals": total_evals}

# ============ 3. Stochastic Hill Climbing ============
def stochastic_hc(tsp, init_tour=None, max_iters=5000):
    tour = (init_tour.copy() if init_tour else random_tour(len(tsp["dist"])))
    cur = tour_length(tsp, tour)
    history = [cur]; n_evals = 0
    for _ in range(max_iters):
        # liệt kê các move đi lên (uphill = giảm độ dài vì minimize)
        uphill = []
        for nb in all_2opt_neighbors(tour):
            L = tour_length(tsp, nb); n_evals += 1
            if L < cur - 1e-12:
                uphill.append((nb, L))
        if not uphill:
            break
        tour, cur = random.choice(uphill)
        history.append(cur)
    return {"tour": tour, "length": cur, "history": history, "evals": n_evals}

# ============ 4. First-choice Hill Climbing ============
def first_choice_hc(tsp, init_tour=None, max_iters=5000):
    tour = (init_tour.copy() if init_tour else random_tour(len(tsp["dist"])))
    cur = tour_length(tsp, tour)
    history = [cur]; n_evals = 0
    for _ in range(max_iters):
        nb = random_2opt_neighbor(tour)
        L = tour_length(tsp, nb); n_evals += 1
        if L < cur - 1e-12:
            tour, cur = nb, L
            history.append(cur)
    return {"tour": tour, "length": cur, "history": history, "evals": n_evals}

# ============ 5. Simulated Annealing ============
def simulated_annealing(tsp, init_tour=None, T0=1.0, alpha=0.995, Tmin=1e-4, max_iters=20000):
    tour = (init_tour.copy() if init_tour else random_tour(len(tsp["dist"])))
    cur = tour_length(tsp, tour)
    best, best_L = tour.copy(), cur
    history = [cur]
    T = T0
    it = 0
    while T > Tmin and it < max_iters:
        nb = random_2opt_neighbor(tour)
        L = tour_length(tsp, nb)
        dE = L - cur
        if dE < 0 or random.random() < math.exp(-dE / T):
            tour, cur = nb, L
            if cur < best_L:
                best, best_L = tour.copy(), cur
        history.append(cur)
        T *= alpha; it += 1
    return {"tour": best, "length": best_L, "history": history, "evals": it}

# ============ Bonus: Genetic Algorithm ============
def order_crossover(p1, p2):
    n = len(p1)
    i, j = sorted(random.sample(range(n), 2))
    child = [None]*n
    child[i:j+1] = p1[i:j+1]
    fill = [g for g in p2 if g not in child[i:j+1]]
    k = 0
    for idx in list(range(j+1, n)) + list(range(0, i)):
        child[idx] = fill[k]; k += 1
    return child

def genetic_algorithm_tsp(tsp, pop_size=100, generations=300, pc=0.9, pm=0.1):
    n = len(tsp["dist"])
    pop = [random_tour(n) for _ in range(pop_size)]
    best, best_L = None, float("inf")
    history = []
    for g in range(generations):
        fits = [tour_length(tsp, t) for t in pop]
        for t, f in zip(pop, fits):
            if f < best_L: best, best_L = t.copy(), f
        history.append(best_L)
        # tournament selection
        def select():
            a, b = random.sample(list(zip(pop, fits)), 2)
            return a[0] if a[1] < b[1] else b[0]
        newpop = []
        while len(newpop) < pop_size:
            p1, p2 = select(), select()
            c = order_crossover(p1, p2) if random.random() < pc else p1.copy()
            if random.random() < pm:  # mutation = 1 move 2-opt ngẫu nhiên
                c = random_2opt_neighbor(c)
            newpop.append(c)
        pop = newpop
    return {"tour": best, "length": best_L, "history": history, "evals": pop_size*generations}

# ============ Demo + khảo sát ============
if __name__ == "__main__":
    random.seed(1234); np.random.seed(1234)
    print("=== Bài 7: Local Search cho TSP ===\n")

    # --- 1. Kiểm tra helper ---
    tsp10 = random_tsp(10, seed=1234)
    t0 = random_tour(10)
    print(f"random_tsp(10): {len(tsp10['dist'])}x{len(tsp10['dist'])} ma trận khoảng cách Euclid")
    print(f"random_tour(10): {t0}")
    print(f"tour_length: {tour_length(tsp10, t0):.4f}")
    show_tsp(tsp10, t0, "Tour ngẫu nhiên ban đầu n=10", os.path.join(OUT, "tour_init_n10.png"))

    # --- 2. Chạy từng thuật toán trên CÙNG 1 instance n=20, CÙNG toán tử 2-opt ---
    print("\n--- So sánh trên cùng TSP n=20 (seed=42) ---")
    tsp = random_tsp(20, seed=42)
    init = random_tour(20)
    L0 = tour_length(tsp, init)
    print(f"Tour ban đầu: L0 = {L0:.4f}")
    show_tsp(tsp, init, "Trước tối ưu n=20", os.path.join(OUT, "tour_before_n20.png"))

    algos = {
        "Steepest-ascent HC": lambda: steepest_ascent_hc(tsp, init.copy()),
        "Random Restarts (20)": lambda: random_restart_hc(tsp, n_restarts=20),
        "Stochastic HC": lambda: stochastic_hc(tsp, init.copy()),
        "First-choice HC": lambda: first_choice_hc(tsp, init.copy()),
        "Simulated Annealing": lambda: simulated_annealing(tsp, init.copy()),
        "Genetic Algorithm (bonus)": lambda: genetic_algorithm_tsp(tsp),
    }
    results = {}
    for name, fn in algos.items():
        t0_ = time.time()
        r = fn()
        dt = time.time() - t0_
        results[name] = (r["length"], dt, r["evals"], r)
        print(f"{name:28s} L={r['length']:.4f}  time={dt*1000:.1f}ms  evals={r['evals']}  cải thiện={(L0-r['length'])/L0*100:.1f}%")
        show_tsp(tsp, r["tour"], f"{name} n=20", os.path.join(OUT, f"tour_{name.split()[0]}_n20.png"))

    # hội tụ
    plt.figure(figsize=(8,5))
    for name, (_, _, _, r) in results.items():
        h = r.get("history")
        if h: plt.plot(h, label=name)
    plt.xlabel("Iteration"); plt.ylabel("Tour length"); plt.title("Hội tụ các thuật toán (n=20, toán tử 2-opt)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "convergence_n20.png"), dpi=150); plt.close()

    # --- 3. Khảo sát scalability: n = 10, 20, 30, 50 ---
    print("\n--- Scalability (mỗi n: 1 instance, mỗi thuật toán 1 lần chạy) ---")
    rows = []
    for n in [10, 20, 30, 50]:
        tsp_n = random_tsp(n, seed=n)
        init_n = random_tour(n)
        L0n = tour_length(tsp_n, init_n)
        for name, fn in {
            "Steepest": steepest_ascent_hc, "Restart20": None, "Stochastic": stochastic_hc,
            "FirstChoice": first_choice_hc, "SA": simulated_annealing, "GA": None}.items():
            t0_ = time.time()
            if name == "Restart20": r = random_restart_hc(tsp_n, n_restarts=20)
            elif name == "GA": r = genetic_algorithm_tsp(tsp_n, pop_size=80, generations=200)
            elif name == "SA": r = fn(tsp_n, init_n.copy())
            else: r = fn(tsp_n, init_n.copy())
            dt = time.time() - t0_
            rows.append({"n": n, "algo": name, "L0": round(L0n,3), "Lbest": round(r["length"],3),
                         "time_ms": round(dt*1000,1), "evals": r["evals"]})
            print(f"n={n:3d} {name:12s} L0={L0n:.3f} -> Lbest={r['length']:.3f}  {dt*1000:.0f}ms  evals={r['evals']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "bang_khao_sat.csv"), index=False)

    # biểu đồ cột Lbest + đường time
    fig, ax1 = plt.subplots(figsize=(9,5))
    pivot_L = df.pivot(index="n", columns="algo", values="Lbest")
    pivot_L.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Best tour length"); ax1.set_title("Chiều dài tour tốt nhất theo n và thuật toán")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "bar_Lbest_vs_n.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(9,5))
    pivot_T = df.pivot(index="n", columns="algo", values="time_ms")
    for c in pivot_T.columns: ax.plot(pivot_T.index, pivot_T[c], marker="o", label=c)
    ax.set_xlabel("Số thành phố n"); ax.set_ylabel("Thời gian (ms, log)"); ax.set_yscale("log")
    ax.set_title("Thời gian chạy theo n (thang log)"); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "time_vs_n.png"), dpi=150); plt.close()

    print(f"\nXong. Ảnh + CSV lưu tại: {OUT}")
    print("Files:", os.listdir(OUT))

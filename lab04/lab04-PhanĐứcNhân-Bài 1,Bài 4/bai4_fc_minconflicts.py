"""
Lab 04 - Bai 4: Forward Checking va Min-Conflicts cho bai toan to mau do thi.
Sinh do thi phang bang Delaunay, so sanh:
  - Backtracking goc (bien chua gan dau tien)
  - Forward Checking + MRV
  - Min-Conflicts (thuan) va Min-Conflicts + random walk 5%
Kem kiem chung tren ho do thi bipyramid 3 mau chac chan co nghiem.

Chay:  ./.venv/bin/python bai4_fc_minconflicts.py
Sinh:  images/bai4_*.png, output/bai4_runs.csv, output/bai4_summary.json
"""
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "images")
OUT = os.path.join(BASE, "output")
os.makedirs(IMG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SEED = 2024
N_INSTANCES = 30
N_LIST = (10, 20, 30, 50)
COLOR_NAMES = ["red", "blue", "green", "orange"]
NODE_LIMIT = 200000
MC_STEPS = 5000
MC_WALK = 0.05


# ---------------------------------------------------------------------------
# Sinh do thi
# ---------------------------------------------------------------------------
def make_delaunay(n: int, seed: int):
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2)) * 1000.0
    points = points[np.argsort(points[:, 0])]
    tri = Delaunay(points)
    indptr, indices = tri.vertex_neighbor_vertices
    adj = [sorted(int(x) for x in indices[indptr[i]:indptr[i + 1]]) for i in range(n)]
    edges = sorted({(i, j) for i in range(n) for j in adj[i] if i < j})
    return points, adj, edges


def make_bipyramid(m: int):
    """Bipyramid tren vong rim m canh (m chan): n = m + 2 dinh, 3m canh.
    Tat ca dinh co bac chan (rim bac 4, apex bac m) nen do thi 3 mau duoc.
    To mau: rim so le 2 mau, hai apex cung mau thu 3."""
    adj = [[] for _ in range(m + 2)]
    for i in range(m):
        j = (i + 1) % m
        adj[i].append(j)
        adj[j].append(i)
    for a in (m, m + 1):
        for i in range(m):
            adj[a].append(i)
            adj[i].append(a)
    return [sorted(set(x)) for x in adj]


# ---------------------------------------------------------------------------
# Backtracking goc
# ---------------------------------------------------------------------------
def backtrack_baseline(adj, k: int, node_limit: int = NODE_LIMIT):
    n = len(adj)
    assignment = [-1] * n
    nodes = 0
    limit_hit = False

    def rec() -> bool:
        nonlocal nodes, limit_hit
        nodes += 1
        if nodes > node_limit:
            limit_hit = True
            return False
        var = next((v for v in range(n) if assignment[v] == -1), -1)
        if var == -1:
            return True
        for color in range(k):
            if all(assignment[u] != color for u in adj[var]):
                assignment[var] = color
                if rec():
                    return True
                assignment[var] = -1
                if limit_hit:
                    return False
        return False

    ok = rec()
    return ok, assignment, nodes, 0, limit_hit


# ---------------------------------------------------------------------------
# Forward Checking + MRV
# ---------------------------------------------------------------------------
def forward_checking(adj, k: int, node_limit: int = NODE_LIMIT):
    n = len(adj)
    domains = [[c for c in range(k)] for _ in range(n)]
    assignment = [-1] * n
    nodes = 0
    prunes = 0
    limit_hit = False

    def rec() -> bool:
        nonlocal nodes, prunes, limit_hit
        nodes += 1
        if nodes > node_limit:
            limit_hit = True
            return False
        var, best = -1, k + 1
        for v in range(n):
            if assignment[v] == -1 and len(domains[v]) < best:
                best = len(domains[v])
                var = v
        if var == -1:
            return True
        if best == 0:
            return False
        for color in list(domains[var]):
            assignment[var] = color
            removed = []
            ok = True
            for u in adj[var]:
                if assignment[u] == -1 and color in domains[u]:
                    domains[u].remove(color)
                    removed.append(u)
                    prunes += 1
                    if not domains[u]:
                        ok = False
                        break
            if ok and rec():
                return True
            for u in removed:
                domains[u].append(color)
            assignment[var] = -1
            if limit_hit:
                return False
        return False

    ok = rec()
    return ok, assignment, nodes, prunes, limit_hit


# ---------------------------------------------------------------------------
# Min-Conflicts
# ---------------------------------------------------------------------------
def min_conflicts(adj, k: int, rng: np.random.Generator, max_steps: int = MC_STEPS,
                  walk: float = 0.0):
    n = len(adj)
    assignment = [int(rng.integers(k)) for _ in range(n)]
    checks = 0

    def conflicts_of(v: int) -> int:
        return sum(1 for u in adj[v] if assignment[u] == assignment[v])

    def total_conflicts() -> int:
        return sum(1 for v in range(n) for u in adj[v]
                   if u > v and assignment[u] == assignment[v])

    for step in range(max_steps):
        conflicted = [v for v in range(n) if conflicts_of(v) > 0]
        if not conflicted:
            return True, assignment, step, checks, 0
        v = conflicted[int(rng.integers(len(conflicted)))]
        scores = []
        for color in range(k):
            scores.append(sum(1 for u in adj[v] if assignment[u] == color))
            checks += 1
        best_score = min(scores)
        best = [c for c in range(k) if scores[c] == best_score]
        if walk and float(rng.random()) < walk:
            new_color = int(rng.integers(k))
        else:
            if len(best) > 1 and assignment[v] in best:
                best.remove(assignment[v])
            new_color = best[int(rng.integers(len(best)))]
        if new_color != assignment[v]:
            assignment[v] = new_color
    return False, assignment, max_steps, checks, total_conflicts()


@dataclass
class RunRow:
    n: int
    k: int
    instance: int
    edges: int
    solvable: int
    bt_found: int
    bt_nodes: int
    bt_time_ms: float
    bt_limit: int
    fc_found: int
    fc_nodes: int
    fc_prunes: int
    fc_time_ms: float
    fc_limit: int
    mc_found: int
    mc_steps: int
    mc_checks: int
    mc_time_ms: float
    mc_final_conflicts: int
    mcw_found: int
    mcw_steps: int
    mcw_checks: int
    mcw_time_ms: float
    mcw_final_conflicts: int


rows: list[RunRow] = []

for n in N_LIST:
    for i in range(N_INSTANCES):
        points, adj, edges = make_delaunay(n, SEED + n * 100 + i)
        for k in (3, 4):
            t0 = time.perf_counter()
            bt_ok, _, bt_nodes, _, bt_limit = backtrack_baseline(adj, k)
            bt_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            fc_ok, _, fc_nodes, fc_prunes, fc_limit = forward_checking(adj, k)
            fc_time = (time.perf_counter() - t0) * 1000

            # FC la thuat toan complete: neu khong cham limit thi ket qua la ground truth
            solvable = int(fc_ok) if not fc_limit else -1

            t0 = time.perf_counter()
            mc_ok, _, mc_steps, mc_checks, mc_final = min_conflicts(
                adj, k, np.random.default_rng([SEED, n, i, k, 0]))
            mc_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            mcw_ok, _, mcw_steps, mcw_checks, mcw_final = min_conflicts(
                adj, k, np.random.default_rng([SEED, n, i, k, 1]), walk=MC_WALK)
            mcw_time = (time.perf_counter() - t0) * 1000

            rows.append(RunRow(
                n=n, k=k, instance=i, edges=len(edges), solvable=solvable,
                bt_found=int(bt_ok), bt_nodes=bt_nodes, bt_time_ms=round(bt_time, 4),
                bt_limit=int(bt_limit), fc_found=int(fc_ok), fc_nodes=fc_nodes,
                fc_prunes=fc_prunes, fc_time_ms=round(fc_time, 4), fc_limit=int(fc_limit),
                mc_found=int(mc_ok), mc_steps=mc_steps, mc_checks=mc_checks,
                mc_time_ms=round(mc_time, 4), mc_final_conflicts=mc_final,
                mcw_found=int(mcw_ok), mcw_steps=mcw_steps, mcw_checks=mcw_checks,
                mcw_time_ms=round(mcw_time, 4), mcw_final_conflicts=mcw_final,
            ))

with open(os.path.join(OUT, "bai4_runs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
    w.writeheader()
    for r in rows:
        w.writerow(asdict(r))

# ---------------------------------------------------------------------------
# Kiem chung tren ho bipyramid 3 mau (chac chan co nghiem)
# ---------------------------------------------------------------------------
bipyramid_rows: list[dict[str, Any]] = []
for m in (6, 8, 10, 18, 28):
    n = m + 2
    adj = make_bipyramid(m)
    t0 = time.perf_counter()
    bt_ok, _, bt_nodes, _, bt_limit = backtrack_baseline(adj, 3)
    bt_time = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    fc_ok, _, fc_nodes, fc_prunes, fc_limit = forward_checking(adj, 3)
    fc_time = (time.perf_counter() - t0) * 1000
    mc_ok = 0
    mc_steps = []
    for s in range(20):
        ok, _, st, _, _ = min_conflicts(
            adj, 3, np.random.default_rng([SEED, n, s, 99]), walk=MC_WALK)
        mc_ok += int(ok)
        mc_steps.append(st)
    bipyramid_rows.append({
        "n": n, "m": m, "edges": sum(len(x) for x in adj) // 2,
        "bt_found": int(bt_ok), "bt_nodes": bt_nodes, "bt_time_ms": round(bt_time, 4),
        "bt_limit": int(bt_limit),
        "fc_found": int(fc_ok), "fc_nodes": fc_nodes, "fc_prunes": fc_prunes,
        "fc_time_ms": round(fc_time, 4), "fc_limit": int(fc_limit),
        "mc_success_20": mc_ok, "mc_avg_steps": round(float(np.mean(mc_steps)), 2),
    })

# ---------------------------------------------------------------------------
# Hinh minh hoa: mot loi giai 4 mau cho do thi Delaunay n = 20
# ---------------------------------------------------------------------------
pts_ex, adj_ex, edges_ex = make_delaunay(20, SEED + 20 * 100)
ok_ex, ass_ex, _, _, _ = forward_checking(adj_ex, 4)
fig, ax = plt.subplots(figsize=(7.2, 5.6))
for (i, j) in edges_ex:
    ax.plot([pts_ex[i, 0], pts_ex[j, 0]], [pts_ex[i, 1], pts_ex[j, 1]],
            color="gray", linewidth=0.7, zorder=1)
cols = [COLOR_NAMES[c] for c in ass_ex]
ax.scatter(pts_ex[:, 0], pts_ex[:, 1], s=260, c=cols, edgecolors="black",
           linewidths=0.8, zorder=2)
for i, p in enumerate(pts_ex):
    ax.annotate(str(i), p, color="white", fontsize=8, weight="bold",
                ha="center", va="center", zorder=3)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Loi giai 4 mau cho do thi Delaunay n = 20 (Forward Checking)")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "bai4_coloring_example.png"), dpi=180)
plt.close(fig)

# ---------------------------------------------------------------------------
# Tong hop
# ---------------------------------------------------------------------------
def mean(vals):
    return float(np.mean(vals)) if vals else 0.0


summary: dict[str, Any] = {"settings": {"seed": SEED, "instances": N_INSTANCES, "node_limit": NODE_LIMIT,
                        "mc_steps": MC_STEPS, "mc_walk": MC_WALK},
           "random": {}, "bipyramid": bipyramid_rows}

for n in N_LIST:
    summary["random"][str(n)] = {}
    for k in (3, 4):
        sub = [r for r in rows if r.n == n and r.k == k]
        solvable = [r for r in sub if r.solvable == 1]
        unsolvable = [r for r in sub if r.solvable == 0]
        summary["random"][str(n)][str(k)] = {
            "instances": len(sub),
            "avg_edges": round(mean([r.edges for r in sub]), 1),
            "solvable": len(solvable),
            "unsolvable": len(unsolvable),
            "unknown": len([r for r in sub if r.solvable == -1]),
            "bt_found": sum(r.bt_found for r in sub),
            "bt_limit": sum(r.bt_limit for r in sub),
            "bt_avg_nodes_all": round(mean([r.bt_nodes for r in sub]), 1),
            "bt_avg_nodes_unsolvable": round(mean([r.bt_nodes for r in unsolvable]), 1),
            "bt_avg_time_ms": round(mean([r.bt_time_ms for r in sub]), 4),
            "bt_median_time_ms": round(float(np.median([r.bt_time_ms for r in sub])), 4),
            "bt_proved_unsolvable": sum(1 for r in unsolvable if not r.bt_found and not r.bt_limit),
            "fc_found": sum(r.fc_found for r in sub),
            "fc_limit": sum(r.fc_limit for r in sub),
            "fc_avg_nodes_all": round(mean([r.fc_nodes for r in sub]), 1),
            "fc_avg_nodes_unsolvable": round(mean([r.fc_nodes for r in unsolvable]), 1),
            "fc_avg_prunes": round(mean([r.fc_prunes for r in sub]), 1),
            "fc_avg_time_ms": round(mean([r.fc_time_ms for r in sub]), 4),
            "fc_median_time_ms": round(float(np.median([r.fc_time_ms for r in sub])), 4),
            "fc_proved_unsolvable": sum(1 for r in unsolvable if not r.fc_found and not r.fc_limit),
            "mc_found_all": sum(r.mc_found for r in sub),
            "mc_found_solvable": sum(r.mc_found for r in solvable),
            "mc_avg_steps_all": round(mean([r.mc_steps for r in sub]), 1),
            "mc_avg_time_ms": round(mean([r.mc_time_ms for r in sub]), 4),
            "mc_median_time_ms": round(float(np.median([r.mc_time_ms for r in sub])), 4),
            "mc_success_on_solvable_pct": round(100 * sum(r.mc_found for r in solvable) / len(solvable), 1) if solvable else None,
            "mcw_found_all": sum(r.mcw_found for r in sub),
            "mcw_found_solvable": sum(r.mcw_found for r in solvable),
            "mcw_avg_steps_all": round(mean([r.mcw_steps for r in sub]), 1),
            "mcw_avg_time_ms": round(mean([r.mcw_time_ms for r in sub]), 4),
            "mcw_median_time_ms": round(float(np.median([r.mcw_time_ms for r in sub])), 4),
            "mcw_success_on_solvable_pct": round(100 * sum(r.mcw_found for r in solvable) / len(solvable), 1) if solvable else None,
        }

with open(os.path.join(OUT, "bai4_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Hinh 1: 4 mau - COUNT va thoi gian BT vs FC
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
x = np.arange(len(N_LIST))
width = 0.36
bt_nodes = [summary["random"][str(n)]["4"]["bt_avg_nodes_all"] for n in N_LIST]
fc_nodes = [summary["random"][str(n)]["4"]["fc_avg_nodes_all"] for n in N_LIST]
bt_time = [summary["random"][str(n)]["4"]["bt_avg_time_ms"] for n in N_LIST]
fc_time = [summary["random"][str(n)]["4"]["fc_avg_time_ms"] for n in N_LIST]
axes[0].plot(x, bt_nodes, marker="o", label="Backtracking", color="#4472c4")
axes[0].plot(x, fc_nodes, marker="s", label="Forward Checking + MRV", color="#ed7d31")
axes[0].set_xticks(x); axes[0].set_xticklabels([f"n={n}" for n in N_LIST])
axes[0].set_ylabel("COUNT trung bình (nút)")
axes[0].set_title("Số nút tìm kiếm — 4 màu"); axes[0].set_yscale("log")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[1].plot(x, bt_time, marker="o", label="Backtracking", color="#4472c4")
axes[1].plot(x, fc_time, marker="s", label="Forward Checking + MRV", color="#ed7d31")
axes[1].set_xticks(x); axes[1].set_xticklabels([f"n={n}" for n in N_LIST])
axes[1].set_ylabel("Thời gian trung bình (ms)")
axes[1].set_title("Thời gian — 4 màu"); axes[1].set_yscale("log")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "bai4_4color_nodes_time.png"), dpi=180)
plt.close(fig)

# ---------------------------------------------------------------------------
# Hinh 2: 3 mau - COUNT tren cac instance vo nghiem
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 4.4))
x = np.arange(len(N_LIST))
bt_u = [summary["random"][str(n)]["3"]["bt_avg_nodes_unsolvable"] for n in N_LIST]
fc_u = [summary["random"][str(n)]["3"]["fc_avg_nodes_unsolvable"] for n in N_LIST]
cnt_u = [summary["random"][str(n)]["3"]["unsolvable"] for n in N_LIST]
ax.plot(x, bt_u, marker="o", label="Backtracking", color="#4472c4")
ax.plot(x, fc_u, marker="s", label="Forward Checking + MRV", color="#ed7d31")
ax.set_ylim(10, max(bt_u) * 8)
for xi, c, b, f in zip(x, cnt_u, bt_u, fc_u):
    if xi == 0:
        ha, dx = "left", 6
    elif xi == len(N_LIST) - 1:
        ha, dx = "right", -6
    else:
        ha, dx = "center", 0
    ax.annotate(f"{c}/{N_INSTANCES} vô nghiệm", (xi, max(b, f)),
                xytext=(dx, 8), textcoords="offset points", ha=ha, fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in N_LIST])
ax.set_ylabel("COUNT trung bình trên instance vô nghiệm")
ax.set_title("3 màu: chi phí chứng minh vô nghiệm (30 đồ thị mỗi n)")
ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "bai4_3color_unsolvable.png"), dpi=180)
plt.close(fig)

# ---------------------------------------------------------------------------
# Hinh 3: Min-Conflicts tren 4 mau
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
mc_rate = [100 * summary["random"][str(n)]["4"]["mc_found_all"] / N_INSTANCES for n in N_LIST]
mcw_rate = [100 * summary["random"][str(n)]["4"]["mcw_found_all"] / N_INSTANCES for n in N_LIST]
mc_steps = [summary["random"][str(n)]["4"]["mc_avg_steps_all"] for n in N_LIST]
mcw_steps = [summary["random"][str(n)]["4"]["mcw_avg_steps_all"] for n in N_LIST]
axes[0].plot(x, mc_rate, marker="o", label="Min-Conflicts thuần", color="#4472c4")
axes[0].plot(x, mcw_rate, marker="s", label="Min-Conflicts + walk 5%", color="#70ad47")
axes[0].set_xticks(x); axes[0].set_xticklabels([f"n={n}" for n in N_LIST])
axes[0].set_ylim(0, 105); axes[0].set_ylabel("Tỷ lệ tìm được h = 0 (%)")
axes[0].set_title("Min-Conflicts — 4 màu (30 lần/n)")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].plot(x, mc_steps, marker="o", label="Min-Conflicts thuần", color="#4472c4")
axes[1].plot(x, mcw_steps, marker="s", label="Min-Conflicts + walk 5%", color="#70ad47")
axes[1].set_xticks(x); axes[1].set_xticklabels([f"n={n}" for n in N_LIST])
axes[1].set_ylabel("Số bước trung bình")
axes[1].set_title("Số bước — 4 màu")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "bai4_minconflicts_4color.png"), dpi=180)
plt.close(fig)

# ---------------------------------------------------------------------------
# Hinh 4: Bipyramid 3 mau
# ---------------------------------------------------------------------------
bn = [r["n"] for r in bipyramid_rows]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
x = np.arange(len(bn))
axes[0].bar(x - 0.2, [100 * r["fc_found"] for r in bipyramid_rows], 0.4,
            label="Forward Checking + MRV", color="#ed7d31")
axes[0].bar(x + 0.2, [100 * r["mc_success_20"] / 20 for r in bipyramid_rows], 0.4,
            label="Min-Conflicts + walk (20 lần)", color="#70ad47")
axes[0].set_xticks(x); axes[0].set_xticklabels([f"n={n}" for n in bn])
axes[0].set_ylim(0, 130); axes[0].set_ylabel("Tỷ lệ tìm được nghiệm (%)")
axes[0].set_title("Họ bipyramid 3 màu — luôn có nghiệm")
axes[0].legend(fontsize=8, loc="upper center", ncol=2); axes[0].grid(axis="y", alpha=0.3)
axes[1].bar(x - 0.2, [r["bt_nodes"] for r in bipyramid_rows], 0.4,
            label="Backtracking", color="#4472c4")
axes[1].bar(x + 0.2, [r["fc_nodes"] for r in bipyramid_rows], 0.4,
            label="Forward Checking + MRV", color="#ed7d31")
axes[1].set_xticks(x); axes[1].set_xticklabels([f"n={n}" for n in bn])
axes[1].set_ylabel("COUNT")
axes[1].set_title("COUNT trên họ bipyramid")
axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "bai4_bipyramid_3color.png"), dpi=180)
plt.close(fig)

# ---------------------------------------------------------------------------
# In tong hop
# ---------------------------------------------------------------------------
print("=== BAI 4: TONG HOP (30 do thi moi n) ===")
for n in N_LIST:
    for k in (3, 4):
        s = summary["random"][str(n)][str(k)]
        print(f"n={n:2d} k={k}: solvable={s['solvable']:2d}/{s['instances']} "
              f"BT nodes={s['bt_avg_nodes_all']:.1f} (vo nghiem {s['bt_avg_nodes_unsolvable']:.1f}) "
              f"FC nodes={s['fc_avg_nodes_all']:.1f} (vo nghiem {s['fc_avg_nodes_unsolvable']:.1f}) "
              f"MC={s['mc_found_all']}/{s['instances']} MCw={s['mcw_found_all']}/{s['instances']}")
print("\n=== BIPYRAMID 3 MAU ===")
for r in bipyramid_rows:
    print(f"n={r['n']:2d} edges={r['edges']:3d} BT={r['bt_found']} ({r['bt_nodes']} nut) "
          f"FC={r['fc_found']} ({r['fc_nodes']} nut) MC={r['mc_success_20']}/20")

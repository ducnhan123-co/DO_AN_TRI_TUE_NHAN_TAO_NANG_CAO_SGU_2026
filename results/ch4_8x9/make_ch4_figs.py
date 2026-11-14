"""Vẽ hình + tính Elo cho Chương 4 từ kết quả thật (tournament/ablation/log train)."""
import csv, json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = sys.argv[1]            # thư mục connect4 (nhánh Nghia + model main)
OUT = sys.argv[2]
NAVY, RED, YEL, BLUE, GREY = "#1F2A44", "#D7263D", "#F2B705", "#1F5FBF", "#9AA3B2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "savefig.dpi": 200, "savefig.bbox": "tight"})
NAMES = {1: "Random", 2: "Heuristic", 3: "Minimax d3", 4: "MCTS 4000", 5: "Giám sát", 6: "PPO", 7: "AZ 100", 8: "AZ 800"}

# --- 4.1 cấp 5: loss + độ chính xác giữ lại (lần chạy cuối = model đang dùng)
rows = list(csv.DictReader(open(f"{R}/levels/8x9/train_log_level5.csv")))
runs, cur = [], []
for r in rows:
    if cur and int(r["step"]) <= int(cur[-1]["step"]):
        runs.append(cur); cur = []
    cur.append(r)
runs.append(cur); last = runs[-1]
st = [int(r["step"]) for r in last]
fig, ax = plt.subplots(figsize=(7, 3.4)); ax2 = ax.twinx()
ax.plot(st, [float(r["loss"]) for r in last], "-o", color=NAVY, label="Loss (cross-entropy, batch)")
ax2.plot(st, [100 * float(r["holdout_acc"]) for r in last], "-s", color=RED, label="Độ chính xác tập giữ lại (%)")
ax.set_xlabel("Bước cập nhật gradient"); ax.set_ylabel("Loss"); ax2.set_ylabel("Độ chính xác (%)")
ax2.set_ylim(55, 70); ax.grid(alpha=.3)
h = ax.get_legend_handles_labels(); h2 = ax2.get_legend_handles_labels()
ax.legend(h[0] + h2[0], h[1] + h2[1], loc="upper right", fontsize=9)
fig.savefig(f"{OUT}/fig4_1_level5.png"); plt.close(fig)

# --- 4.2 AlphaZero
rows = list(csv.DictReader(open(f"{R}/levels/8x9/train_log_az.csv")))
st = [int(r["step"]) for r in rows]
fig, ax = plt.subplots(figsize=(7, 3.4)); ax2 = ax.twinx()
ax.plot(st, [float(r["loss_pi"]) for r in rows], "-o", color=NAVY, label="Loss policy (cross-entropy)")
ax2.plot(st, [float(r["loss_v"]) for r in rows], "-s", color=RED, label="Loss value (MSE)")
ax.set_xlabel("Bước cập nhật gradient"); ax.set_ylabel("Loss policy"); ax2.set_ylabel("Loss value"); ax.grid(alpha=.3)
h = ax.get_legend_handles_labels(); h2 = ax2.get_legend_handles_labels()
ax.legend(h[0] + h2[0], h[1] + h2[1], loc="upper right", fontsize=9)
fig.savefig(f"{OUT}/fig4_2_az.png"); plt.close(fig)

# --- 4.3 round-robin heatmap + Elo
S = json.load(open(f"{R}/results/tournament_8x9/default/summary.json"))
M = np.full((8, 8), np.nan)
for k, s in S["pairs"].items():
    a, b = [int(x.strip()[1:]) for x in k.split("vs")]
    M[a - 1, b - 1] = s["score"]; M[b - 1, a - 1] = 1 - s["score"]
fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
for i in range(8):
    for j in range(8):
        if i != j:
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
lab = [f"C{n}\n{NAMES[n]}" for n in range(1, 9)]
ax.set_xticks(range(8)); ax.set_xticklabels([f"C{n}" for n in range(1, 9)], fontsize=9); ax.set_yticks(range(8)); ax.set_yticklabels(lab, fontsize=8)
ax.set_xlabel("Đối thủ (cột)"); ax.set_ylabel("Tác tử (hàng)")
fig.colorbar(im, ax=ax, fraction=.046, label="Điểm của tác tử hàng")
fig.savefig(f"{OUT}/fig4_3_heatmap.png"); plt.close(fig)

# Elo (Bradley-Terry, hòa = nửa thắng, prior nhẹ 1 ván hòa ảo mỗi cặp để tránh vô cực), chuẩn hoá C1 = 0
n = np.zeros((8, 8)); w = np.zeros((8, 8))
for k, s in S["pairs"].items():
    a, b = [int(x.strip()[1:]) - 1 for x in k.split("vs")]
    n[a, b] = n[b, a] = s["n"] + 1
    w[a, b] = s["score"] * s["n"] + 0.5; w[b, a] = (1 - s["score"]) * s["n"] + 0.5
g = np.ones(8)
for _ in range(5000):
    for i in range(8):
        den = sum(n[i, j] / (g[i] + g[j]) for j in range(8) if j != i)
        g[i] = w[i].sum() / den
    g /= g[0]
elo = 400 * np.log10(g)
json.dump({f"L{i + 1}": round(float(e)) for i, e in enumerate(elo)}, open(f"{OUT}/elo.json", "w"))

# --- 4.4 ablation: ngân sách tìm kiếm
def abl(aid):
    return json.load(open(f"{R}/results/ablation_8x9/{aid}.json"))["rows"]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
for ax, aid, key, ttl, xl in ((axs[0], "A1", "depth", "A1: Minimax (neo: Heuristic)", "Độ sâu"),
                              (axs[1], "A2", "sims", "A2: MCTS (neo: Heuristic)", "Số mô phỏng"),
                              (axs[2], "B3", "sims", "B3: AlphaZero (neo: MCTS 1000)", "Số mô phỏng")):
    rs = abl(aid); x = [r["opts"][key] for r in rs]; y = [r["score"] for r in rs]
    lo = [r["score"] - r["wilson"][0] for r in rs]; hi = [r["wilson"][1] - r["score"] for r in rs]
    ax.errorbar(x, y, yerr=[lo, hi], fmt="-o", color=NAVY, capsize=3)
    if key == "sims": ax.set_xscale("log")
    ax.axhline(.5, color=GREY, ls="--", lw=1); ax.set_ylim(0, 1); ax.set_title(ttl, fontsize=10); ax.set_xlabel(xl); ax.grid(alpha=.3)
axs[0].set_ylabel("Điểm (Wilson 95%)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_4_budget.png"); plt.close(fig)

# --- 4.5 ablation mask/gương cho PPO chính
rows = list(csv.DictReader(open(f"{R}/log_ablmain.txt"))) if False else None
lines = [l for l in open(f"{R}/log_ablmain.txt") if l.count(",") == 8 and not l.startswith("variant")]
data = {}
for l in lines:
    v, o, *_r = l.strip().split(","); data.setdefault(v, {})[o] = float(_r[4])
opps = ["random", "tactical", "minimax2", "minimax4"]
fig, ax = plt.subplots(figsize=(8, 3.4)); wbar = .2
for i, (v, c) in enumerate(zip(data, [NAVY, BLUE, RED, YEL])):
    ax.bar(np.arange(4) + (i - 1.5) * wbar, [data[v].get(o, 0) for o in opps], wbar, label=v, color=c)
ax.set_xticks(range(4)); ax.set_xticklabels(["Random", "Tactical", "Minimax d2", "Minimax d4"]); ax.set_ylabel("Điểm (100 ván)")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.3)
fig.savefig(f"{OUT}/fig4_5_mask.png"); plt.close(fig)

# --- 4.6 độ trễ
cost = S["cost"]
lv = [l for l in [f"L{i}" for i in range(1, 9)] if l in cost]
fig, ax = plt.subplots(figsize=(8, 3.4)); x = np.arange(len(lv))
ax.bar(x - .2, [max(cost[l]["mean_ms"], 1e-3) for l in lv], .4, label="TB (ms/nước)", color=NAVY)
ax.bar(x + .2, [max(cost[l]["p95_ms"], 1e-3) for l in lv], .4, label="P95 (ms/nước)", color=RED)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"C{l[1:]}\n{NAMES[int(l[1:])]}" for l in lv], fontsize=8)
ax.set_ylabel("ms / nước (thang log)"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3)
fig.savefig(f"{OUT}/fig4_6_latency.png"); plt.close(fig)
print("ok", json.load(open(f"{OUT}/elo.json")))

"""Vẽ các hình của báo cáo từ log/kết quả thật (không cần torch; chỉ cần matplotlib).

    python tools/plot_results.py train5                       # Hình 4.2.1  loss + độ chính xác cấp 5 (levels/8x9/train_log_level5.csv)
    python tools/plot_results.py az                           # Hình 4.2.2  loss policy / value AlphaZero (levels/8x9/train_log_az.csv)
    python tools/plot_results.py ppo runs/main/eval_log.csv   # Hình 3.4.5.1 điểm đánh giá PPO theo số bước
    python tools/plot_results.py matrix  results/runs/<giải đấu>   # Hình 4.4.1 ma trận điểm round-robin
    python tools/plot_results.py elo     results/runs/<giải đấu>   # Elo Bradley–Terry
    python tools/plot_results.py latency results/runs/<giải đấu>   # Hình 4.5.1 độ trễ mỗi nước (TB và P95, thang log)
    python tools/plot_results.py budget  results/runs/<ablation A1,A2,B3>   # Hình 4.4.1 điểm theo ngân sách tìm kiếm
    python tools/plot_results.py ablation-agent results/runs/<ablation-agent>  # Hình 4.4.2 ablation PPO chính (mask / gương)
    python tools/plot_results.py all results/runs/<giải đấu>   # matrix + elo + latency
Tuỳ chọn:  --out THƯ_MỤC (mặc định: <thư mục kết quả>/figures, hoặc results/figures cho train5/az/ppo)  --dark
Mọi hình lưu PNG 200 dpi. Màu: bảng 8 màu phân loại cố định (xanh, cam, ngọc, vàng, ...) theo thứ tự; nền trắng.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # thư mục gốc dự án
import argparse
import csv
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from c4.expio import BASE

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]  # thứ tự cố định
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
LEVEL_NAMES = {"L1": "Random", "L2": "Heuristic", "L3": "Minimax d3", "L4": "MCTS 4000", "L5": "Học giám sát",
               "L6": "PPO", "L7": "AlphaZero 100", "L8": "AlphaZero 800"}


def setup(dark=False):
    global INK, INK2, GRID
    bg, INK, INK2, GRID = ("#191919", "#efefef", "#9c9c9c", "#383838") if dark else ("#ffffff", "#191919", "#666666", "#e6e6e6")
    plt.rcParams.update({"figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2})


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def last_run(rows, key):
    """Log huấn luyện được NỐI THÊM khi chạy lại (bước đếm lại từ đầu) => chỉ giữ lần chạy cuối để không vẽ đường nối thừa."""
    start = 0
    for i in range(1, len(rows)):
        if float(rows[i][key]) <= float(rows[i - 1][key]):
            start = i
    if start:
        print(f"  (log có {start} dòng của các lần chạy trước; chỉ vẽ lần chạy cuối, {len(rows) - start} dòng)")
    return rows[start:]


def num(rows, key):
    return np.array([float(r[key]) for r in rows])


def err(sc, lo, hi):
    """Thanh sai số [dưới, trên] của khoảng Wilson; cắt về >= 0 vì sai số làm tròn float có thể cho -1e-17 (matplotlib báo lỗi)."""
    return [np.clip(sc - lo, 0, None), np.clip(hi - sc, 0, None)]


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Đã vẽ:", os.path.relpath(p, BASE) if p.startswith(BASE) else p)
    return p


def lab(l):
    return f"{l} {LEVEL_NAMES[l]}" if l in LEVEL_NAMES else str(l)


# ----------------------------------------------------------------------------- huấn luyện
def plot_train5(csv_path, out):
    rows = last_run(read_csv(csv_path), "step")
    step, loss, acc = num(rows, "step"), num(rows, "loss"), num(rows, "val_acc")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(step, loss, color=SERIES[0]); ax[0].set_title("Loss (cross-entropy)"); ax[0].set_xlabel("Bước gradient")
    ax[1].plot(step, acc * 100, color=SERIES[1]); ax[1].set_title("Độ chính xác trên tập giữ lại (%)")
    ax[1].set_xlabel("Bước gradient")
    ax[1].annotate(f"{acc[-1] * 100:.1f}%", (step[-1], acc[-1] * 100), textcoords="offset points", xytext=(-6, 6), ha="right",
                   color=INK)
    fig.suptitle("Huấn luyện cấp 5 (học có giám sát bắt chước Minimax)", y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "train_level5.png")


def plot_az(csv_path, out):
    rows = last_run(read_csv(csv_path), "step")
    step, lp, lv = num(rows, "step"), num(rows, "loss_pi"), num(rows, "loss_v")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(step, lp, color=SERIES[0]); ax[0].set_title("Loss policy"); ax[0].set_xlabel("Bước gradient")
    ax[1].plot(step, lv, color=SERIES[1]); ax[1].set_title("Loss value"); ax[1].set_xlabel("Bước gradient")
    fig.suptitle("Huấn luyện AlphaZero thu nhỏ (cấp 7–8)", y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "train_alphazero.png")


def plot_ppo(csv_path, out):
    rows = last_run(read_csv(csv_path), "timesteps")
    t = num(rows, "timesteps")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, (key, name) in enumerate([("random", "Random"), ("tactical", "Tactical"), ("minimax2", "Minimax d2"),
                                     ("minimax4", "Minimax d4")]):
        ax.plot(t, num(rows, key), color=SERIES[k], label=name, marker="o", markersize=4)
    ax.plot(t, num(rows, "hard_score"), color=INK, linestyle="--", label="hard_score (chọn best)")
    ax.set_xlabel("Số bước huấn luyện"); ax.set_ylabel("Điểm"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Điểm đánh giá định kỳ của PPO theo số bước"); ax.legend(ncol=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    return save(fig, out, "ppo_eval.png")


# ----------------------------------------------------------------------------- giải đấu
def _pairs(run):
    return read_csv(os.path.join(run, "metrics_pairs.csv"))


def plot_matrix(run, out):
    rows = _pairs(run)
    labels = sorted({r["A"] for r in rows} | {r["B"] for r in rows}, key=lambda s: (len(s), s))
    k = len(labels)
    M = np.full((k, k), np.nan)
    for r in rows:
        i, j, s = labels.index(r["A"]), labels.index(r["B"]), float(r["score_A"])
        M[i, j], M[j, i] = s, 1 - s
    fig, ax = plt.subplots(figsize=(0.95 * k + 2.2, 0.8 * k + 1.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([lab(l) for l in labels], rotation=35, ha="right"); ax.set_yticklabels([lab(l) for l in labels])
    for i in range(k):
        for j in range(k):
            if i != j and not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > 0.6 else "#191919", fontsize=9)
    ax.set_xlabel("Đối thủ (cột)"); ax.set_ylabel("Tác tử (hàng)")
    ax.set_title("Điểm của tác tử hàng khi gặp tác tử cột")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Điểm")
    fig.tight_layout()
    return save(fig, out, "tournament_matrix.png")


def _overall(run):
    with open(os.path.join(run, "metrics_overall.json"), encoding="utf-8") as f:
        return json.load(f)


def plot_elo(run, out):
    ov = _overall(run)
    elo = ov.get("elo")
    if not elo:  # lần chạy cũ chưa có Elo => tính từ metrics_pairs.csv
        from c4.elo import elo_from_pairs
        rows = _pairs(run)
        pts = {(r["A"], r["B"]): (float(r["score_A"]) * int(r["games"]), int(r["games"])) for r in rows}
        labels = sorted({x for ab in pts for x in ab}, key=lambda s: (len(s), s))
        elo = elo_from_pairs(pts, labels, anchor="L1" if "L1" in labels else None)
    items = sorted(elo.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(items) + 1.4))
    ax.barh([lab(l) for l, _ in items], [v for _, v in items], color=SERIES[0], height=0.62)
    for y, (_, v) in enumerate(items):
        ax.text(v, y, f" {v:.0f}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Elo tương đối (Bradley–Terry, Random = 0)"); ax.set_title("Xếp hạng Elo của các cấp"); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, out, "elo.png")


def plot_latency(run, out):
    cost = _overall(run).get("cost") or {}
    labels = [l for l in sorted(cost, key=lambda s: (len(s), s))]
    if not labels:
        raise SystemExit("metrics_overall.json không có mục 'cost'")
    x = np.arange(len(labels))
    mean = np.array([cost[l]["mean_ms"] for l in labels]); p95 = np.array([cost[l]["p95_ms"] for l in labels])
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    ax.bar(x - 0.2, mean, 0.38, color=SERIES[0], label="Trung bình")
    ax.bar(x + 0.2, p95, 0.38, color=SERIES[1], label="P95")
    ax.set_yscale("log"); ax.set_ylabel("ms / nước (thang log)")
    ax.set_xticks(x); ax.set_xticklabels([lab(l) for l in labels], rotation=25, ha="right")
    ax.set_title("Độ trễ mỗi nước theo cấp"); ax.legend(); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, out, "latency.png")


# ----------------------------------------------------------------------------- ablation
def plot_budget(run, out):
    rows = _pairs(run)
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    titles = {"A1": "Minimax: độ sâu", "A2": "MCTS: số mô phỏng", "B3": "AlphaZero: số mô phỏng"}
    ids = [i for i in ("A1", "A2", "B3") if i in by] or sorted(by)
    fig, axs = plt.subplots(1, len(ids), figsize=(3.8 * len(ids), 3.6), squeeze=False)
    for k, (ax, i) in enumerate(zip(axs[0], ids)):
        rs = by[i]
        names = [re.sub(r"^\S+ ", "", r["variant"]).split("=")[-1] for r in rs]
        sc = num(rs, "score"); lo = num(rs, "wilson_lo"); hi = num(rs, "wilson_hi")
        ax.errorbar(range(len(rs)), sc, yerr=err(sc, lo, hi), color=SERIES[k], marker="o", capsize=3)
        ax.set_xticks(range(len(rs))); ax.set_xticklabels(names); ax.set_ylim(0, 1.02)
        ax.set_title(titles.get(i, i)); ax.set_xlabel("Ngân sách"); ax.set_ylabel("Điểm so với neo" if k == 0 else "")
    fig.suptitle("Điểm theo ngân sách tìm kiếm (Wilson 95%)", y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "search_budget.png")


def plot_ablation_agent(run, out):
    rows = read_csv(os.path.join(run, "ablation_agent.csv"))
    vs = list(dict.fromkeys(r["variant_label"] for r in rows))
    ops = list(dict.fromkeys(r["opponent"] for r in rows))
    w = 0.8 / len(vs)
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    for k, v in enumerate(vs):
        rs = [next(r for r in rows if r["variant_label"] == v and r["opponent"] == o) for o in ops]
        sc, lo, hi = num(rs, "score"), num(rs, "wilson_lo"), num(rs, "wilson_hi")
        xs = np.arange(len(ops)) + (k - (len(vs) - 1) / 2) * w
        ax.bar(xs, sc, w * 0.92, color=SERIES[k], label=v, yerr=err(sc, lo, hi), error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    ax.set_xticks(range(len(ops))); ax.set_xticklabels(ops); ax.set_ylim(0, 1.05); ax.set_ylabel("Điểm")
    ax.set_title("Ablation tác tử PPO chính: mask chiến thuật và lật gương"); ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, out, "ablation_agent.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["train5", "az", "ppo", "matrix", "elo", "latency", "budget", "ablation-agent", "all"])
    ap.add_argument("path", nargs="?", help="thư mục lần chạy hoặc file CSV (xem đầu file)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args()
    setup(a.dark)
    lv = os.path.join(BASE, "levels", "8x9")
    default_out = os.path.join(BASE, "results", "figures")
    if a.what == "train5":
        plot_train5(a.path or os.path.join(lv, "train_log_level5.csv"), a.out or default_out)
    elif a.what == "az":
        plot_az(a.path or os.path.join(lv, "train_log_az.csv"), a.out or default_out)
    elif a.what == "ppo":
        if not a.path:
            ap.error("ppo cần đường dẫn eval_log.csv")
        plot_ppo(a.path, a.out or default_out)
    else:
        if not a.path:
            ap.error(f"{a.what} cần thư mục lần chạy")
        out = a.out or os.path.join(a.path, "figures")
        fns = {"matrix": plot_matrix, "elo": plot_elo, "latency": plot_latency, "budget": plot_budget,
               "ablation-agent": plot_ablation_agent}
        for name in (["matrix", "elo", "latency"] if a.what == "all" else [a.what]):
            fns[name](a.path, out)


if __name__ == "__main__":
    main()
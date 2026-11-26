"""
run_ablation.py - Các thí nghiệm ablation (đề cuối kỳ: ít nhất 2). Mỗi biến thể chơi >= 100 ván với một đối thủ NEO cố định
(cùng khai cuộc ghép cặp, đổi bên, seed như tournament.py), rồi so sánh score (kèm Wilson 95%) và chi phí.

    python run_ablation.py --board 6x7 --only A1,A2          # không cần train
    python run_ablation.py --board 8x9 --only B1,B2,B3       # cần model cấp 5/6/7-8 (xem docs/HUONG_DAN_TRAIN.md)
    python run_ablation.py --list

Không cần train:  A1 độ sâu minimax (cấp 3) | A2 số mô phỏng MCTS (cấp 4)
Cần model đã train: B1 mask chiến thuật + lật gương (cấp 5) | B2 mask/lật gương/tìm kiếm lai (cấp 6)
                    B3 số mô phỏng AlphaZero (cấp 7/8)
"""
from c4 import boardenv  # noqa: F401

import argparse
import os
import sys
import time

from c4 import expio
from c4.core import COLS, ROWS
from c4.levels import LEVEL_NAMES, is_ready, make_level
from tournament import BOARD, agent_cost, pair_costs, run_pair, summarize_pair

# id -> (mô tả, cấp của biến thể, danh sách (nhãn, opts), neo (cấp, opts), cấp cần model)
ABLATIONS = {
    "A1": ("Độ sâu minimax (cấp 3)", 3, [(f"depth={d}", {"depth": d}) for d in (1, 2, 3, 4, 5)], (2, {})),
    "A2": ("Số mô phỏng MCTS (cấp 4)", 4, [(f"sims={s}", {"sims": s}) for s in (50, 200, 1000, 4000)], (2, {})),
    "B1": ("Cấp 5: mask chiến thuật / lật gương", 5,
           [("đầy đủ", {}), ("không shield", {"shield": False}), ("không flip", {"flip_avg": False}),
            ("không cả hai", {"shield": False, "flip_avg": False})], (3, {})),
    "B2": ("Cấp 6: shield / flip / tìm kiếm lai", 6,
           [("đầy đủ", {}), ("không shield", {"shield": False}), ("không flip", {"flip_avg": False}),
            ("+minimax d2", {"search_depth": 2}), ("+minimax d4", {"search_depth": 4})], (3, {})),
    "B3": ("Cấp 7/8: số mô phỏng AlphaZero", 7, [(f"sims={s}", {"sims": s}) for s in (10, 25, 100, 400)], (4, {"sims": 1000})),
}


def run_one(aid, seeds, games_per_seed, opening_plies, run_dir, sink):
    """Chạy một ablation; trả về (records, danh sách hàng kết quả) hoặc None nếu thiếu model."""
    desc, lvl, variants, (anc_lvl, anc_opts) = ABLATIONS[aid]
    need = [lvl, anc_lvl]
    if any(not is_ready(n) for n in need):
        print(f"[{aid}] bỏ qua: cấp {[n for n in need if not is_ready(n)]} chưa có model cho bàn {BOARD}")
        return None
    print(f"\n[{aid}] {desc}  (neo: cấp {anc_lvl} {LEVEL_NAMES[anc_lvl]} {anc_opts or ''})", flush=True)
    anchor = ("anchor", anc_lvl, anc_opts)
    rows, allrecs = [], []
    for k, (label, o) in enumerate(variants):
        name = f"{aid} {label}"
        recs = run_pair((name, lvl, o), anchor, seeds, games_per_seed, opening_plies, pair_idx=k, sink=sink, tag=aid)
        allrecs += recs
        s = summarize_pair(recs)
        cost = agent_cost(pair_costs(recs, name, "anchor")).get(name, {})
        rows.append({"id": aid, "description": desc, "variant": name, "opts": o, **s, "cost": cost})
        print(f"    {label:16s} {s['win']}W/{s['draw']}D/{s['loss']}L  score={s['score']:.3f} "
              f"[{s['wilson'][0]:.3f},{s['wilson'][1]:.3f}]  ms/nước TB={cost.get('mean_ms', 0):.1f} p95={cost.get('p95_ms', 0):.1f}", flush=True)
    return allrecs, rows


def main():
    ap = argparse.ArgumentParser(description="Ablation Connect 4")
    ap.add_argument("--board", default=None)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--only", default="A1,A2", help="vd A1,A2,B1,B2,B3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--games-per-seed", type=int, default=50, help="chẵn; mặc định 2 seed x 50 = 100 ván/biến thể")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--goal", default="", help="MỤC TIÊU (ghi vào nhật ký)")
    ap.add_argument("--hypothesis", default="", help="GIẢ THUYẾT viết trước khi xem kết quả")
    ap.add_argument("--author", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default=None, help="thư mục lần chạy (mặc định results/runs/<mã>)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in ABLATIONS.items():
            print(k, "-", v[0], "| biến thể:", [x[0] for x in v[2]])
        return
    if a.games_per_seed % 2:
        raise SystemExit("--games-per-seed phải chẵn")
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    ids = [x.strip().upper() for x in a.only.split(",") if x.strip()]
    for aid in ids:
        if aid not in ABLATIONS:
            raise SystemExit(f"không có ablation {aid}; dùng --list")
    run_id = expio.new_run_id("ablation", BOARD, "-".join(ids) + (f"_{a.tag}" if a.tag else ""))
    out = a.out or expio.make_run_dir(run_id)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    started, t0 = expio.now(), time.time()
    allrecs, allrows, params = [], [], {}
    with expio.Tee(os.path.join(out, "run.log.txt")):
        print(f"=== {run_id} -> {out}")
        with open(os.path.join(out, "data", "games.jsonl"), "w", encoding="utf-8") as sink:
            for aid in ids:
                r = run_one(aid, seeds, a.games_per_seed, a.opening_plies, out, sink)
                if r:
                    allrecs += r[0]
                    allrows += r[1]
                    desc, lvl, variants, (anc_lvl, anc_opts) = ABLATIONS[aid]
                    for label, o in variants:
                        params[f"{aid} {label}"] = expio.agent_params(make_level(lvl, seed=0, **o), lvl, o)
                    params[f"{aid} anchor"] = expio.agent_params(make_level(anc_lvl, seed=0, **anc_opts), anc_lvl, anc_opts)
        if not allrows:
            raise SystemExit("không chạy được ablation nào (thiếu model?)")
        duration = time.time() - t0
        app_cfg = {"run_id": run_id, "kind": "ablation", "ablations": ids, "board": BOARD, "rows": ROWS, "cols": COLS,
                   "seeds": seeds, "games_per_seed": a.games_per_seed, "games_per_variant": a.games_per_seed * len(seeds),
                   "opening_plies": a.opening_plies, "command": ["python"] + sys.argv, "goal": a.goal,
                   "hypothesis": a.hypothesis, "author": a.author, "notes": a.notes, "started": started,
                   "ended": expio.now(), "duration_s": round(duration, 1), "code_hash": expio.code_hash(),
                   "environment": expio.environment()}
        pair_rows = [{"id": r["id"], "variant": r["variant"], "games": r["n"], "win": r["win"], "draw": r["draw"],
                      "loss": r["loss"], "score": round(r["score"], 4), "wilson_lo": round(r["wilson"][0], 4),
                      "wilson_hi": round(r["wilson"][1], 4), "boot_lo": round(r["bootstrap"][0], 4),
                      "boot_hi": round(r["bootstrap"][1], 4), "mean_ms": round(r["cost"].get("mean_ms", 0), 3),
                      "p95_ms": round(r["cost"].get("p95_ms", 0), 3), "mean_nodes": round(r["cost"].get("mean_nodes", 0), 1),
                      "mean_sims": round(r["cost"].get("mean_sims", 0), 1)} for r in allrows]
        md = [f"# Ablation {', '.join(ids)} - bàn {BOARD}", ""]
        for aid in ids:
            rs = [r for r in allrows if r["id"] == aid]
            if not rs:
                continue
            md += [f"## {aid}: {rs[0]['description']}", "", "| Biến thể | W/D/L | Score | Wilson 95% | ms/nước TB | p95 |", "|---|---|---|---|---|---|"]
            for r in rs:
                md.append(f"| {r['variant']} | {r['win']}/{r['draw']}/{r['loss']} | {r['score']:.3f} | [{r['wilson'][0]:.3f}, "
                          f"{r['wilson'][1]:.3f}] | {r['cost'].get('mean_ms', 0):.1f} | {r['cost'].get('p95_ms', 0):.1f} |")
            md.append("")
        _, overall = expio.finalize(out, run_id, "ablation", app_cfg, params, allrecs, False, pair_rows, {}, "\n".join(md))
        best = max(allrows, key=lambda r: r["score"])
        expio.journal_write(
            {"Mã TN": run_id, "Bắt đầu": started, "Thời lượng (phút)": round(duration / 60, 2), "Loại": "Ablation " + ",".join(ids),
             "Người thực hiện": a.author, "Mục tiêu": a.goal, "Giả thuyết": a.hypothesis, "Bàn cờ": BOARD,
             "Đối tượng": "; ".join(f"{i}: {ABLATIONS[i][0]}" for i in ids), "Ngân sách": "mặc định",
             "Seeds": ",".join(map(str, seeds)), "Ván/cặp": app_cfg["games_per_variant"], "Khai cuộc (nước)": a.opening_plies,
             "Lệnh chạy": " ".join(app_cfg["command"]), "Mã băm code": app_cfg["code_hash"],
             "Phần cứng": f"{app_cfg['environment']['cpu']} x{app_cfg['environment']['cpu_count']}",
             "Thư mục kết quả": os.path.relpath(out, expio.BASE),
             "Kết quả chính (tự động)": f"Biến thể score cao nhất: {best['variant']} = {best['score']:.3f} "
                                        f"[{best['wilson'][0]:.3f}, {best['wilson'][1]:.3f}]",
             "Bất thường / ghi chú": a.notes, "Trạng thái": "Hoàn tất"},
            ablation_rows=[[run_id, r["id"], r["description"], r["variant"], r["win"], r["draw"], r["loss"],
                            round(r["score"], 4), round(r["wilson"][0], 4), round(r["wilson"][1], 4),
                            round(r["cost"].get("mean_ms", 0), 3), round(r["cost"].get("p95_ms", 0), 3)] for r in allrows])
        print(f"Đã ghi nhật ký: {os.path.relpath(expio.JOURNAL)}")


if __name__ == "__main__":
    main()

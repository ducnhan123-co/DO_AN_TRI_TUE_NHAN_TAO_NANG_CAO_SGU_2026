"""
run_ablation.py - Các thí nghiệm ablation (đề cuối kỳ: ít nhất 2). Mỗi biến thể chơi >= 100 ván với một đối thủ NEO cố định
(cùng khai cuộc ghép cặp, đổi bên, seed như tournament.py), rồi so sánh score (kèm Wilson 95%) và chi phí.

    python run_ablation.py --board 6x7 --only A1,A2          # không cần train
    python run_ablation.py --board 8x9 --only B1,B2,B3       # cần model cấp 5/6/7-8 (xem HUONG_DAN_TRAIN.md)
    python run_ablation.py --list

Không cần train:  A1 độ sâu minimax (cấp 3) | A2 số mô phỏng MCTS (cấp 4)
Cần model đã train: B1 mask chiến thuật + lật gương (cấp 5) | B2 mask/lật gương/tìm kiếm lai (cấp 6)
                    B3 số mô phỏng AlphaZero (cấp 7/8)
"""
import boardenv  # noqa: F401

import argparse
import json
import os
import time

from levels import BASE_DIR, LEVEL_NAMES, is_ready
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


def run_one(aid, seeds, games_per_seed, opening_plies, out_root):
    desc, lvl, variants, (anc_lvl, anc_opts) = ABLATIONS[aid]
    need = [lvl, anc_lvl]
    if any(not is_ready(n) for n in need):
        print(f"[{aid}] bỏ qua: cấp {[n for n in need if not is_ready(n)]} chưa có model cho bàn {BOARD}")
        return None
    print(f"\n[{aid}] {desc}  (neo: cấp {anc_lvl} {LEVEL_NAMES[anc_lvl]} {anc_opts or ''})", flush=True)
    anchor = ("anchor", anc_lvl, anc_opts)
    os.makedirs(out_root, exist_ok=True)
    rows = []
    with open(os.path.join(out_root, f"{aid}_games.jsonl"), "w", encoding="utf-8") as sink:
        for k, (label, o) in enumerate(variants):
            recs = run_pair((label, lvl, o), anchor, seeds, games_per_seed, opening_plies, pair_idx=k, sink=sink, tag=aid)
            s = summarize_pair(recs)
            cost = agent_cost(pair_costs(recs, label, "anchor")).get(label, {})
            rows.append({"variant": label, "opts": o, **s, "cost": cost})
            print(f"    {label:16s} {s['win']}W/{s['draw']}D/{s['loss']}L  score={s['score']:.3f} "
                  f"[{s['wilson'][0]:.3f},{s['wilson'][1]:.3f}]  ms/nước TB={cost.get('mean_ms', 0):.1f} p95={cost.get('p95_ms', 0):.1f}", flush=True)
    with open(os.path.join(out_root, f"{aid}.json"), "w", encoding="utf-8") as f:
        json.dump({"id": aid, "description": desc, "board": BOARD, "anchor": {"level": anc_lvl, "opts": anc_opts},
                   "seeds": list(seeds), "games_per_seed": games_per_seed, "opening_plies": opening_plies,
                   "date": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}, f, indent=2, ensure_ascii=False)
    md = [f"## {aid}: {desc} (bàn {BOARD}; neo = cấp {anc_lvl})", "",
          "| Biến thể | W/D/L | Score | Wilson 95% | ms/nước TB | p95 | nút TB | mô phỏng TB |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["cost"]
        md.append(f"| {r['variant']} | {r['win']}/{r['draw']}/{r['loss']} | {r['score']:.3f} | [{r['wilson'][0]:.3f}, {r['wilson'][1]:.3f}] "
                  f"| {c.get('mean_ms', 0):.1f} | {c.get('p95_ms', 0):.1f} | {c.get('mean_nodes', 0):.0f} | {c.get('mean_sims', 0):.0f} |")
    with open(os.path.join(out_root, f"{aid}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Ablation Connect 4")
    ap.add_argument("--board", default=None)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--only", default="A1,A2", help="vd A1,A2,B1,B2,B3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--games-per-seed", type=int, default=50, help="chẵn; mặc định 2 seed x 50 = 100 ván/biến thể")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in ABLATIONS.items():
            print(k, "-", v[0], "| biến thể:", [x[0] for x in v[2]])
        return
    if a.games_per_seed % 2:
        raise SystemExit("--games-per-seed phải chẵn")
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    out = a.out or os.path.join(BASE_DIR, "results", f"ablation_{BOARD}")
    for aid in [x.strip().upper() for x in a.only.split(",") if x.strip()]:
        if aid not in ABLATIONS:
            raise SystemExit(f"không có ablation {aid}; dùng --list")
        run_one(aid, seeds, a.games_per_seed, a.opening_plies, out)


if __name__ == "__main__":
    main()

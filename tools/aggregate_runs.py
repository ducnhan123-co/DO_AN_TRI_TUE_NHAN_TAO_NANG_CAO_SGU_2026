"""Tổng hợp lợi thế ĐI TRƯỚC / ĐI SAU xuyên suốt nhiều lần chạy giải đấu (RQ4 của báo cáo) + Elo Bradley–Terry gộp.

    python tools/aggregate_runs.py results/runs/*tournament*default*            # nhiều thư mục lần chạy (hoặc glob)
    python tools/aggregate_runs.py results/runs/X/data/games.jsonl --include-random
    python tools/aggregate_runs.py results/runs/* --out results/aggregate --elo

Ghi:  <out>/first_mover.json, first_mover_pairs.csv, first_mover_agents.csv, first_mover.md  (+ elo_pooled.csv nếu --elo)
Mặc định bỏ các cặp có L1 (Random) như báo cáo; --include-random để giữ lại.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # thư mục gốc dự án
import argparse
from c4 import expio
from c4.aggregate import first_mover_report, load_records, report_markdown

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("runs", nargs="+", help="thư mục lần chạy (có data/games.jsonl), file games.jsonl, hoặc mẫu glob")
ap.add_argument("--out", default=os.path.join(expio.BASE, "results", "aggregate"))
ap.add_argument("--exclude", default="L1", help="các nhãn tác tử bị loại, cách nhau dấu phẩy (mặc định L1 = Random)")
ap.add_argument("--include-random", action="store_true", help="không loại cặp nào")
ap.add_argument("--elo", action="store_true", help="tính thêm Elo Bradley–Terry gộp mọi ván (bootstrap 95%%)")
ap.add_argument("--boot", type=int, default=200, help="số lần bootstrap cho Elo")
a = ap.parse_args()

recs = load_records(a.runs)
if not recs:
    raise SystemExit("không có ván nào trong các đường dẫn đã cho")
exclude = () if a.include_random else tuple(x.strip() for x in a.exclude.split(",") if x.strip())
rep = first_mover_report(recs, exclude)
os.makedirs(a.out, exist_ok=True)
expio.write_json(os.path.join(a.out, "first_mover.json"), rep)
expio.write_csv(os.path.join(a.out, "first_mover_pairs.csv"), rep["pairs"])
expio.write_csv(os.path.join(a.out, "first_mover_agents.csv"), rep["agents"])
md = report_markdown(rep)
with open(os.path.join(a.out, "first_mover.md"), "w", encoding="utf-8") as f:
    f.write(md)
print(md)
if a.elo:
    from c4.elo import elo_bootstrap
    labels = sorted({r["a"] for r in recs} | {r["b"] for r in recs}, key=str)
    res = elo_bootstrap(recs, labels, n_boot=a.boot, anchor="L1" if "L1" in labels else None)
    rows = [{"agent": l, "elo": round(e, 1), "elo_lo": round(lo, 1), "elo_hi": round(hi, 1)}
            for l, (e, lo, hi) in sorted(res.items(), key=lambda kv: -kv[1][0])]
    expio.write_csv(os.path.join(a.out, "elo_pooled.csv"), rows)
    print("Elo gộp (Bradley–Terry):")
    for r in rows:
        print(f"  {r['agent']:6s} {r['elo']:8.1f}  [{r['elo_lo']:.1f}; {r['elo_hi']:.1f}]")
print("Đã lưu:", a.out)

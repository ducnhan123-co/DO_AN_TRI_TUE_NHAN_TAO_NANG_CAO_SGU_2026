"""Kiểm thử các công cụ phân tích (Elo, tổng hợp đi trước/sau, đánh giá/ablation tác tử, vẽ hình). KHÔNG cần torch.
Chạy: python tests/test_analysis.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # thư mục gốc dự án
import json
import math
import tempfile

from c4.agent_eval import (VARIANTS, ablation_markdown, games_for, match_split, parse_opponents, parse_variants,
                           run_agent_ablation)
from c4.aggregate import first_mover_report
from c4.core import MinimaxPlayer, RandomPlayer, match
from c4.elo import elo_bootstrap, elo_from_pairs, elo_from_records, expected_score

OK = []


def check(name, cond):
    OK.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


# ---- Elo
e = elo_from_pairs({("a", "b"): (75, 100)}, virtual_draws=0)
check("Elo: 75% điểm => chênh 400*log10(3)", abs((e["a"] - e["b"]) - 400 * math.log10(3)) < 1e-6)
check("Elo: tổng bằng 0 khi không anchor", abs(e["a"] + e["b"]) < 1e-6)
e = elo_from_pairs({("a", "b"): (0, 50), ("a", "c"): (0, 50), ("b", "c"): (30, 50)}, anchor="a")
check("Elo: thua toàn bộ vẫn hữu hạn nhờ ván hòa ảo, anchor = 0", e["a"] == 0 and all(math.isfinite(v) for v in e.values()) and e["b"] > 0)
check("Elo: thứ hạng đúng (b > c > a)", e["b"] > e["c"] > e["a"])
check("expected_score(0, 0) = 0.5 và đối xứng", expected_score(0, 0) == 0.5 and abs(expected_score(200, 0) + expected_score(0, 200) - 1) < 1e-12)
recs = [{"a": "x", "b": "y", "a_outcome": 1.0 if i % 4 else 0.0, "a_player": 1 + i % 2, "winner": 1, "plies": 20, "seed": 0}
        for i in range(80)]
er = elo_from_records(recs, ["x", "y"], virtual_draws=0)
check("Elo từ bản ghi: 75% => 190.8", abs(er["x"] - er["y"] - 400 * math.log10(3)) < 1e-6)
bt = elo_bootstrap(recs, ["x", "y"], n_boot=30, seed=1, virtual_draws=0)
check("Elo bootstrap: khoảng chứa điểm ước lượng, lặp lại được", bt["x"][1] <= bt["x"][0] <= bt["x"][2]
      and bt == elo_bootstrap(recs, ["x", "y"], n_boot=30, seed=1, virtual_draws=0))

# ---- tổng hợp đi trước / đi sau
syn = []
for g in range(40):
    ap = 1 + g % 2
    w = 1  # người đi trước luôn thắng
    syn.append({"a": "L2", "b": "L3", "a_player": ap, "winner": w, "a_outcome": 1.0 if w == ap else 0.0, "plies": 20 + g % 3, "seed": 0})
syn += [{"a": "L1", "b": "L3", "a_player": 1, "winner": 2, "a_outcome": 0.0, "plies": 10, "seed": 0}] * 10
rep = first_mover_report(syn)
check("Tổng hợp: loại cặp có Random mặc định", rep["overall"]["games"] == 40)
check("Tổng hợp: người đi trước thắng mọi ván => điểm 1,0", rep["overall"]["first_wins"] == 40 and rep["overall"]["first_score"] == 1.0)
check("Tổng hợp: score A đi trước 1,0 / đi sau 0,0 và A tốt hơn khi đi trước",
      rep["pairs"][0]["score_A_first"] == 1.0 and rep["pairs"][0]["score_A_second"] == 0.0 and rep["overall"]["pairs_better_when_first"] == 1)
check("Tổng hợp: --include-random giữ đủ ván", first_mover_report(syn, exclude=())["overall"]["games"] == 50)
check("Tổng hợp: rỗng => ValueError", raises(lambda: first_mover_report(syn[:0]), ValueError))

# ---- đánh giá tác tử
check("parse_opponents: mặc định 5 đối thủ, minimax6 hợp lệ", parse_opponents("all") == ["random", "tactical", "minimax2", "minimax4", "minimax6"]
      and parse_opponents("random, minimax6") == ["random", "minimax6"])
check("parse_opponents: tên sai => ValueError", raises(lambda: parse_opponents("foo"), ValueError))
check("games_for: minimax6 chậm => games//3 (>= 20), --d6-games ghi đè", games_for("minimax6", 100) == 33 and games_for("minimax6", 30) == 20
      and games_for("minimax6", 100, 7) == 7 and games_for("random", 100) == 100)
a, b = MinimaxPlayer(2, seed=1), RandomPlayer(2)
m1, m2 = match(a, b, 12, 2, seed=7), match_split(MinimaxPlayer(2, seed=1), RandomPlayer(2), 12, 2, seed=7)
check("match_split trùng core.match (cùng seed)", (m1["win"], m1["draw"], m1["loss"]) == (m2["win"], m2["draw"], m2["loss"]))
check("match_split: có score_first / score_second", m2["games_first"] == 6 and m2["games_second"] == 6 and 0 <= m2["score_first"] <= 1)
check("parse_variants: mặc định 4 biến thể, sai => lỗi", parse_variants("default") == ["full", "no_flip", "no_mask", "no_mask_no_flip"]
      and raises(lambda: parse_variants("x"), ValueError))
check("VARIANTS: no_mask => shield False; no_flip => flip_avg False", VARIANTS["no_mask"]["shield"] is False and VARIANTS["no_flip"]["flip_avg"] is False
      and VARIANTS["full"] == {"shield": True, "flip_avg": True})


class FakeAgent:  # thay TorchPlayer: with_options đổi độ sâu minimax theo cờ shield
    def __init__(self, depth=2):
        self.p = MinimaxPlayer(depth, seed=1)
        self.name = "fake"

    def with_options(self, **kw):
        return FakeAgent(2 if kw.get("shield", True) else 1)

    def act(self, me, opp):
        return self.p.act(me, opp)


rows = run_agent_ablation(FakeAgent(), ["full", "no_mask"], ["random", "tactical"], games=8, log=lambda *_: None)
check("run_agent_ablation: 2 biến thể x 2 đối thủ = 4 hàng", len(rows) == 4 and {r["variant"] for r in rows} == {"full", "no_mask"})
md = ablation_markdown(rows)
check("ablation_markdown: có đủ cột đối thủ và nhãn biến thể", "random" in md and "tactical" in md and "Bỏ mask" in md)

# ---- vẽ hình (cần matplotlib)
try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
if HAVE_MPL:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import plot_results as pr
    from c4 import expio
    pr.setup()
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "run")
        os.makedirs(run)
        pair_rows = [{"mode": "default", "A": "L1", "B": "L2", "games": 20, "win": 0, "draw": 0, "loss": 20, "score_A": 0.0},
                     {"mode": "default", "A": "L1", "B": "L3", "games": 20, "win": 0, "draw": 0, "loss": 20, "score_A": 0.0},
                     {"mode": "default", "A": "L2", "B": "L3", "games": 20, "win": 5, "draw": 0, "loss": 15, "score_A": 0.25}]
        expio.write_csv(os.path.join(run, "metrics_pairs.csv"), pair_rows)
        with open(os.path.join(run, "metrics_overall.json"), "w") as f:
            json.dump({"cost": {"L1": {"mean_ms": 0.1, "p95_ms": 0.2}, "L2": {"mean_ms": 0.5, "p95_ms": 1.0},
                                "L3": {"mean_ms": 2.0, "p95_ms": 5.0}}}, f)  # không có 'elo' => tính lại từ metrics_pairs.csv
        expio.write_csv(os.path.join(run, "ablation_agent.csv"), rows)
        expio.write_csv(os.path.join(d, "t.csv"), [{"step": 1000, "loss": 1.0, "val_acc": 0.5}, {"step": 2000, "loss": 0.8, "val_acc": 0.6},
                                                    {"step": 1000, "loss": 1.2, "val_acc": 0.4}, {"step": 2000, "loss": 0.9, "val_acc": 0.55}])
        out = os.path.join(d, "fig")
        files = [pr.plot_matrix(run, out), pr.plot_elo(run, out), pr.plot_latency(run, out), pr.plot_ablation_agent(run, out),
                 pr.plot_train5(os.path.join(d, "t.csv"), out)]
        check("Vẽ hình: tạo đủ 5 file PNG không rỗng", all(os.path.getsize(f) > 2000 for f in files))
        check("Vẽ hình: log huấn luyện nối thêm chỉ lấy lần chạy cuối", len(pr.last_run(pr.read_csv(os.path.join(d, "t.csv")), "step")) == 2)

bad = [n for n, ok in OK if not ok]
print(f"\n{len(OK) - len(bad)}/{len(OK)} đạt")
sys.exit(1 if bad else 0)

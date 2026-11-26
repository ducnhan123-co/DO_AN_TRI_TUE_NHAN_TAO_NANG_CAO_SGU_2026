"""Kiểm thử không cần torch/pygame thật:  python -m unittest tests.test_arena -v   (hoặc python -m tests.test_arena)"""
import os
import json
import tempfile
import time
import unittest

import tempfile as _tf

os.environ["C4_JOURNAL"] = os.path.join(_tf.mkdtemp(), "nhat_ky.xlsx")  # không đụng nhật ký thật khi test

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from tests import fake_pygame as fp

fp.install()  # pygame giả (Pillow) - chỉ để kiểm tra luồng điều khiển của giao diện

from c4 import arena  # noqa: E402
from c4 import levels  # noqa: E402
import tournament as T  # noqa: E402
from c4.core import COLS, ROWS, MinimaxPlayer, drop_bit, has_won  # noqa: E402
from c4.expstats import bootstrap_score_ci, percentile, wilson_interval  # noqa: E402
from c4.mcts import MCTSPlayer  # noqa: E402


class TestLevelsAndBudget(unittest.TestCase):
    def test_levels_1_to_4_play_legal(self):
        for n in (1, 2, 3, 4):
            p = levels.make_level(n, seed=1, **({"sims": 30} if n == 4 else {}))
            a = p.act(0, 0)
            self.assertTrue(0 <= a < COLS)

    def test_no_pretrained_weights_shipped(self):
        for n in (5, 6, 7, 8):
            self.assertFalse(levels.is_ready(n), f"cấp {n} không được có model dựng sẵn")
            with self.assertRaises(FileNotFoundError):
                levels.make_level(n)

    def test_minimax_counts_nodes_and_time_budget(self):
        p = MinimaxPlayer(3, seed=0)
        p.act(0, 0)
        self.assertGreater(p.last_stats["nodes"], 0)
        q = MinimaxPlayer(time_limit=0.05, seed=0)
        t0 = time.perf_counter()
        a = q.act(0, 0)
        self.assertLess(time.perf_counter() - t0, 0.5)
        self.assertTrue(0 <= a < COLS)
        self.assertGreaterEqual(q.last_stats["depth"], 1)
        self.assertIsNone(__import__("c4.core", fromlist=["x"])._DEADLINE[0])

    def test_mcts_time_limit_and_sims(self):
        p = MCTSPlayer(sims=10**9, seed=0, time_limit=0.03)
        t0 = time.perf_counter()
        p.act(0, 0)
        self.assertLess(time.perf_counter() - t0, 0.5)
        self.assertGreater(p.last_stats["sims"], 10)

    def test_budget_params(self):
        self.assertEqual(arena.budget_params(2, 100), {})
        self.assertEqual(arena.budget_params(3, 100), {"time_limit": 0.1})
        self.assertEqual(arena.budget_params(4, 100)["sims"], 10**9)
        self.assertEqual(arena.budget_params(3, 0), {})


class TestStats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson_interval(50, 100)
        self.assertAlmostEqual((lo + hi) / 2, 0.5, delta=0.01)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_bootstrap_and_percentile(self):
        lo, hi = bootstrap_score_ci([1.0] * 50 + [0.0] * 50, n_boot=300)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 50), 3)


class TestSession(unittest.TestCase):
    def test_ai_series_records_every_game(self):
        with tempfile.TemporaryDirectory() as d:
            rec = arena.Recorder(d)
            s = arena.Session(arena.PlayerSpec("ai", 2), arena.PlayerSpec("ai", 3, {"depth": 2}), games=6,
                              opening_plies=2, recorder=rec, seed=3)
            for g in range(6):
                while not s.over:
                    s.run_ai_blocking()
                if g < 5:
                    s.new_game()
            self.assertTrue(s.finished_series)
            recs = rec.read()
            self.assertEqual(len(recs), 6)
            self.assertEqual(sum(s.score.values()), 6)
            firsts = [r["a_first"] for r in recs]
            self.assertEqual(firsts, [True, False] * 3)  # đổi bên mỗi ván
            self.assertEqual(recs[0]["opening"], recs[1]["opening"])  # khai cuộc ghép cặp
            st = rec.stats()
            self.assertEqual(st[0]["games"], 6)
            self.assertEqual(st[0]["win"] + st[0]["draw"] + st[0]["loss"], 6)
            self.assertTrue(os.path.exists(rec.export_csv()))
            # phát lại nước đi phải ra đúng kết quả
            r = recs[0]
            bbs = [0, 0]
            for i, m in enumerate(r["moves"]):
                bbs[i % 2] |= drop_bit(bbs[0] | bbs[1], m)
            if r["result"] != "draw":
                self.assertTrue(has_won(bbs[(len(r["moves"]) - 1) % 2]))

    def test_human_move_validation_and_record(self):
        with tempfile.TemporaryDirectory() as d:
            rec = arena.Recorder(d)
            s = arena.Session(arena.PlayerSpec("human"), arena.PlayerSpec("human"), recorder=rec)
            self.assertFalse(s.human_move(-1))
            self.assertFalse(s.human_move(COLS))
            for _ in range(4):  # A xếp 4 quân dọc cột 0, B cột 1
                self.assertTrue(s.human_move(0))
                if s.over:
                    break
                self.assertTrue(s.human_move(1))
            self.assertTrue(s.over)
            self.assertEqual(rec.read()[0]["result"], "A")


class TestTournament(unittest.TestCase):
    def test_round_robin_small(self):
        from c4 import expio
        expio.JOURNAL = os.path.join(tempfile.mkdtemp(), "nk.xlsx")
        with tempfile.TemporaryDirectory() as d:
            T.run_round_robin([1, 2, 3], "default", [0], 4, 2, 0, d, progress=False)
            s = json.load(open(os.path.join(d, "metrics_overall.json"), encoding="utf-8"))
            self.assertEqual(len(s["per_agent"]), 3)
            self.assertEqual(sum(1 for _ in open(os.path.join(d, "data", "games.jsonl"))), 12)
            self.assertIn("p95_ms", s["cost"]["L3"])
            self.assertGreater(s["cost"]["L3"]["mean_nodes"], 0)
            for f in ("app_cfg.json", "params_cfg.json", "run.log.txt" if False else "summary.md", "games_all.csv",
                      "games_seed0.csv", "metrics_seed.csv", "metrics_seed_mean.json", "metrics_pairs.csv"):
                self.assertTrue(os.path.exists(os.path.join(d, f)), f)
            cfg = json.load(open(os.path.join(d, "app_cfg.json"), encoding="utf-8"))
            self.assertEqual(cfg["games_per_pair"], 4)
            self.assertEqual(len(cfg["code_hash"]), 12)
            # nhật ký Excel
            import openpyxl
            wb = openpyxl.load_workbook(expio.JOURNAL)
            self.assertEqual(wb["Nhật ký"].max_row, 2)
            self.assertEqual(wb["Kết quả cặp"].max_row, 4)
            # tiếp tục (resume) không chơi lại
            T.run_round_robin([1, 2, 3], "default", [0], 4, 2, 0, d, progress=False)
            self.assertEqual(sum(1 for _ in open(os.path.join(d, "data", "games.jsonl"))), 12)

    def test_reproducible(self):
        def run():
            with tempfile.TemporaryDirectory() as d:
                T.run_round_robin([2, 3], "default", [0], 4, 2, 0, d, progress=False)
                return [json.loads(l)["moves"] for l in open(os.path.join(d, "data", "games.jsonl"))]
        self.assertEqual(run(), run())

    def test_side_balance_and_paired_openings(self):
        ops = T.make_openings(0, 5, 4)
        self.assertEqual(len(ops), 5)
        with tempfile.TemporaryDirectory() as d:
            T.run_round_robin([2, 3], "default", [0], 4, 2, 0, d, progress=False)
            recs = [json.loads(l) for l in open(os.path.join(d, "data", "games.jsonl"))]
            self.assertEqual(sorted(r["a_player"] for r in recs), [1, 1, 2, 2])
            self.assertEqual(recs[0]["opening"], recs[1]["opening"])


class TestJournal(unittest.TestCase):
    def test_rerun_replaces_row_and_keeps_manual_conclusion(self):
        from c4 import expio
        import openpyxl
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "j.xlsx")
            row = {"Mã TN": "X1", "Loại": "t", "Kết quả chính (tự động)": "a"}
            expio.journal_write(row, path=p)
            wb = openpyxl.load_workbook(p)
            wb["Nhật ký"].cell(2, 19).value = "kết luận của người"
            wb.save(p)
            expio.journal_write({**row, "Kết quả chính (tự động)": "b"}, path=p)
            expio.journal_write({"Mã TN": "X2"}, path=p)
            ws = openpyxl.load_workbook(p)["Nhật ký"]
            self.assertEqual(ws.max_row, 3)
            vals = {ws.cell(r, 1).value: (ws.cell(r, 18).value, ws.cell(r, 19).value) for r in (2, 3)}
            self.assertEqual(vals["X1"], ("b", "kết luận của người"))
            self.assertEqual(openpyxl.load_workbook(p).sheetnames[-1], "Từ điển")

    def test_metrics_helpers(self):
        from c4 import expio
        recs = [{"seed": s, "game": g, "a": "A", "b": "B", "a_player": 1, "opening": [], "winner": 1, "a_outcome": 1.0,
                 "plies": 7, "moves": [3], "log": [(1, 3, 0.001, 5, 0), (2, 2, 0.002, 0, 9)]} for s in (0, 1) for g in range(2)]
        ms = expio.per_seed_metrics(recs, True)
        self.assertEqual(len(ms), 4)
        self.assertEqual([r["score"] for r in ms if r["agent"] == "A"], [1.0, 1.0])
        sm = expio.seed_mean(ms)
        self.assertEqual(sm["B"]["score_mean"], 0.0)
        self.assertEqual(sm["A"]["games"], 4)


class TestGUI(unittest.TestCase):
    def _click(self, app, bid):
        for r, b in app.buttons:
            if b == bid:
                app.handle(fp.Ev(fp.MOUSEBUTTONDOWN, button=1, pos=r.center))
                app.draw()
                return
        raise KeyError(bid)

    def test_menu_to_game_records_and_results(self):
        from c4 import gui
        with tempfile.TemporaryDirectory() as d:
            app = gui.ArenaApp(headless=True)
            app.rec = arena.Recorder(d)
            app.draw()
            self._click(app, ("a", 5))  # chưa train -> bị khóa, không đổi lựa chọn
            self.assertEqual(app.st["a"], 0)
            self.assertIn("Chưa train", app.msg)
            self._click(app, ("a", 2))
            self._click(app, ("b", 3))
            self._click(app, "games")  # 50 -> chỉ cần bắt đầu; ép 2 ván bên dưới
            self.assertEqual(app._mode_text(), "Máy - Máy")
            self._click(app, "start")
            self.assertEqual(app.screen_name, "game")
            app.session.games_total = 2
            deadline = time.time() + 60
            while not app.session.finished_series and time.time() < deadline:
                app.update_game()
                time.sleep(0.0005)
            self.assertTrue(app.session.finished_series)
            self.assertEqual(len(app.rec.read()), 2)
            app.draw()
            self._click(app, "results")
            self.assertEqual(app.screen_name, "results")
            self._click(app, "csv")
            self.assertIn("summary.csv", app.msg)

    def test_human_click_flow(self):
        from c4 import gui
        with tempfile.TemporaryDirectory() as d:
            app = gui.ArenaApp(headless=True)
            app.rec = arena.Recorder(d)
            app.draw()
            self._click(app, ("b", 1))
            self._click(app, "start")
            self.assertTrue(app.session.human_turn())
            x = app.panel_x + app.BM + app.cell * 2 + app.cell // 2
            app.handle(fp.Ev(fp.MOUSEBUTTONDOWN, button=1, pos=(x, app.panel_y + 50)))
            self.assertEqual(app.session.moves, [2])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Kiểm thử không cần torch/pygame thật:  python -m unittest test_arena -v   (hoặc python test_arena.py)"""
import json
import os
import tempfile
import time
import unittest

import fake_pygame as fp

fp.install()  # pygame giả (Pillow) - chỉ để kiểm tra luồng điều khiển của giao diện

import arena  # noqa: E402
import levels  # noqa: E402
import tournament as T  # noqa: E402
from c4core import COLS, ROWS, MinimaxPlayer, drop_bit, has_won  # noqa: E402
from expstats import bootstrap_score_ci, percentile, wilson_interval  # noqa: E402
from mcts import MCTSPlayer  # noqa: E402


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
        self.assertIsNone(__import__("c4core")._DEADLINE[0])

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
        with tempfile.TemporaryDirectory() as d:
            T.run_round_robin([1, 2, 3], "default", [0], 4, 2, 0, d, progress=False)
            s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
            self.assertEqual(len(s["pairs"]), 3)
            self.assertEqual(sum(1 for _ in open(os.path.join(d, "games.jsonl"))), 12)
            self.assertEqual(s["pairs"]["L1 vs L2"]["n"], 4)
            self.assertIn("p95_ms", s["cost"]["L3"])
            self.assertGreater(s["cost"]["L3"]["mean_nodes"], 0)
            # tiếp tục (resume) không chơi lại
            T.run_round_robin([1, 2, 3], "default", [0], 4, 2, 0, d, progress=False)
            self.assertEqual(sum(1 for _ in open(os.path.join(d, "games.jsonl"))), 12)

    def test_reproducible(self):
        def run():
            with tempfile.TemporaryDirectory() as d:
                T.run_round_robin([2, 3], "default", [0], 4, 2, 0, d, progress=False)
                return [json.loads(l)["moves"] for l in open(os.path.join(d, "games.jsonl"))]
        self.assertEqual(run(), run())

    def test_side_balance_and_paired_openings(self):
        ops = T.make_openings(0, 5, 4)
        self.assertEqual(len(ops), 5)
        with tempfile.TemporaryDirectory() as d:
            T.run_round_robin([2, 3], "default", [0], 4, 2, 0, d, progress=False)
            recs = [json.loads(l) for l in open(os.path.join(d, "games.jsonl"))]
            self.assertEqual(sorted(r["a_player"] for r in recs), [1, 1, 2, 2])
            self.assertEqual(recs[0]["opening"], recs[1]["opening"])


class TestGUI(unittest.TestCase):
    def _click(self, app, bid):
        for r, b in app.buttons:
            if b == bid:
                app.handle(fp.Ev(fp.MOUSEBUTTONDOWN, button=1, pos=r.center))
                app.draw()
                return
        raise KeyError(bid)

    def test_menu_to_game_records_and_results(self):
        import gui
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
        import gui
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

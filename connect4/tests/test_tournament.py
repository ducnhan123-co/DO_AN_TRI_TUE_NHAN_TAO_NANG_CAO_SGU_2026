import unittest

from c4ai.agents import RandomAgent, HeuristicAgent, AlphaBetaAgent
from c4ai.stats import wilson_interval, percentile
from c4ai.tournament import AgentSpec, run_matchup, summarize_pair, make_openings
from c4ai.game import Connect4


class TestStats(unittest.TestCase):
    def test_wilson_known_values(self):
        lo, hi = wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.404, places=2)
        self.assertAlmostEqual(hi, 0.596, places=2)
        lo, hi = wilson_interval(100, 100)
        self.assertLess(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)

    def test_percentile(self):
        self.assertEqual(percentile(list(range(101)), 95), 95)


class TestTournament(unittest.TestCase):
    def specs(self):
        return (AgentSpec("Heu", lambda s: HeuristicAgent(s)),
                AgentSpec("Rnd", lambda s: RandomAgent(s)))

    def test_openings_valid_and_nonterminal(self):
        for o in make_openings(0, 50, 4):
            g = Connect4.from_moves(o)
            self.assertFalse(g.is_terminal())
            self.assertEqual(len(o), 4)

    def test_balanced_sides_and_reproducible(self):
        a, b = self.specs()
        r1 = run_matchup(a, b, seeds=(0,), games_per_seed=20)
        r2 = run_matchup(a, b, seeds=(0,), games_per_seed=20)
        self.assertEqual([r["moves"] for r in r1], [r["moves"] for r in r2])
        self.assertEqual(sum(r["a_player"] == 1 for r in r1), 10)
        self.assertEqual(sum(r["a_player"] == 2 for r in r1), 10)
        s = summarize_pair(r1)
        self.assertEqual(s["win"] + s["draw"] + s["loss"], 20)
        self.assertGreater(s["score"], 0.8)  # heuristic >> random

    def test_deterministic_agents_still_vary_via_openings(self):
        a = AgentSpec("AB", lambda s: AlphaBetaAgent(depth=2, seed=s))
        b = AgentSpec("AB2", lambda s: AlphaBetaAgent(depth=3, seed=s))
        recs = run_matchup(a, b, seeds=(0,), games_per_seed=20)
        self.assertGreater(len({tuple(r["moves"]) for r in recs}), 10)


if __name__ == "__main__":
    unittest.main()

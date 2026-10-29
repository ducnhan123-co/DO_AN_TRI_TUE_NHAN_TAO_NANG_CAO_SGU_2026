import random
import unittest

from c4ai.game import Connect4
from c4ai.agents import RandomAgent, HeuristicAgent, AlphaBetaAgent, MCTSAgent

# X sắp thắng ở cột 3 (ngang: cột 0,1,2 hàng dưới) -> X đến lượt phải đi 3
WIN_IN_1 = [
    ".......",
    ".......",
    ".......",
    ".......",
    "OO.....",
    "XXX...O",
]
# O đến lượt: X có 3 quân ngang ở hàng dưới, phải chặn cột 3
BLOCK_IN_1 = [
    ".......",
    ".......",
    ".......",
    ".......",
    "O.O.O..",
    "XXX.X..",
]
def all_agents(seed=0):
    return [
        RandomAgent(seed),
        HeuristicAgent(seed),
        AlphaBetaAgent(depth=3, seed=seed),
        MCTSAgent(n_sims=300, seed=seed),
    ]


class TestAllAgents(unittest.TestCase):
    def test_always_legal_and_do_not_mutate(self):
        for agent in all_agents(3):
            rng = random.Random(5)
            for _ in range(6):
                g = Connect4()
                while not g.is_terminal():
                    before = (g.key(), g.mask, g.to_move, list(g.moves))
                    a = agent.select_action(g)
                    self.assertEqual(before, (g.key(), g.mask, g.to_move, list(g.moves)),
                                     f"{agent.name} làm thay đổi game")
                    self.assertIn(a, g.legal_actions(), agent.name)
                    g.step(a if rng.random() < 0.7 else rng.choice(g.legal_actions()))

    def test_single_legal_move(self):
        rng = random.Random(0)
        g = None
        for _ in range(5000):
            h = Connect4()
            while not h.is_terminal():
                if len(h.legal_actions()) == 1:
                    g = h
                    break
                h.step(rng.choice(h.legal_actions()))
            if g:
                break
        self.assertIsNotNone(g)
        only = g.legal_actions()[0]
        for agent in all_agents():
            self.assertEqual(agent.select_action(g), only, agent.name)


class TestTactics(unittest.TestCase):
    def test_win_in_one(self):
        g = Connect4.from_grid(WIN_IN_1)
        self.assertEqual(g.to_move, 1)
        for agent in [HeuristicAgent(0), AlphaBetaAgent(depth=2), MCTSAgent(n_sims=500, seed=0)]:
            self.assertEqual(agent.select_action(g), 3, agent.name)

    def test_block_in_one(self):
        g = Connect4.from_grid(BLOCK_IN_1)
        self.assertEqual(g.to_move, 2)
        for agent in [HeuristicAgent(0), AlphaBetaAgent(depth=2), MCTSAgent(n_sims=2000, seed=0)]:
            self.assertEqual(agent.select_action(g), 3, agent.name)

    def test_heuristic_prefers_win_over_block(self):
        # cả hai bên cùng có nước thắng ngay: phải thắng
        g = Connect4.from_grid([
            ".......",
            ".......",
            ".......",
            ".......",
            "OOO....",
            "XXX....",
        ])
        self.assertEqual(g.to_move, 1)
        self.assertEqual(HeuristicAgent(0).select_action(g), 3)

    def test_heuristic_rule_flags(self):
        g = Connect4.from_grid(BLOCK_IN_1)
        # tắt luật chặn: không còn bắt buộc đi cột 3 (chỉ kiểm tra không lỗi)
        a = HeuristicAgent(0, use_block=False).select_action(g)
        self.assertIn(a, g.legal_actions())

    def test_alphabeta_finds_forced_win_in_two(self):
        # X đi cột 3 tạo ".XXX." với hai đầu (cột 0 và 4) đều trống -> đe dọa kép
        g = Connect4.from_grid([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            ".XX..OO",
        ])
        self.assertEqual(g.to_move, 1)
        ab = AlphaBetaAgent(depth=4)
        self.assertEqual(ab.select_action(g), 3)
        self.assertGreaterEqual(ab.last_stats["value"], 900_000)  # thấy thắng cưỡng bức

    def test_alphabeta_avoids_losing_move(self):
        # O đang bị đe dọa ngang, chặn bằng cột 3 là duy nhất
        g = Connect4.from_grid(BLOCK_IN_1)
        self.assertEqual(AlphaBetaAgent(depth=4).select_action(g), 3)


class TestAlphaBeta(unittest.TestCase):
    def test_pruning_gives_same_move_and_value_but_fewer_nodes(self):
        rng = random.Random(11)
        for _ in range(12):
            g = Connect4()
            for _ in range(rng.randint(0, 14)):
                if g.is_terminal():
                    break
                g.step(rng.choice(g.legal_actions()))
            if g.is_terminal():
                continue
            on = AlphaBetaAgent(depth=4, prune=True)
            off = AlphaBetaAgent(depth=4, prune=False)
            self.assertEqual(on.select_action(g), off.select_action(g))
            self.assertEqual(on.last_stats["value"], off.last_stats["value"])
            self.assertLessEqual(on.last_stats["nodes"], off.last_stats["nodes"])

    def test_node_count_unpruned_depth2_from_start(self):
        # từ thế cờ đầu: 1 (gốc) -> ở đây _search_root không tính nút gốc:
        # depth 2 = 7 con + 49 cháu = 56 nút
        ab = AlphaBetaAgent(depth=2, prune=False)
        ab.select_action(Connect4())
        self.assertEqual(ab.last_stats["nodes"], 7 + 49)

    def test_time_limit_iterative_deepening(self):
        ab = AlphaBetaAgent(time_limit=0.05)
        a = ab.select_action(Connect4())
        self.assertIn(a, range(7))
        self.assertGreaterEqual(ab.last_stats["depth"], 1)

    def test_prefers_faster_win(self):
        # có thể thắng ngay -> phải chọn thắng ngay dù độ sâu lớn
        g = Connect4.from_grid(WIN_IN_1)
        ab = AlphaBetaAgent(depth=5)
        self.assertEqual(ab.select_action(g), 3)


class TestMCTS(unittest.TestCase):
    def test_reproducible_with_seed(self):
        g = Connect4.from_moves([3, 3, 2])
        a1 = MCTSAgent(n_sims=200, seed=7).select_action(g)
        a2 = MCTSAgent(n_sims=200, seed=7).select_action(g)
        self.assertEqual(a1, a2)

    def test_sim_budget_respected_and_visits_sum(self):
        m = MCTSAgent(n_sims=137, seed=1)
        m.select_action(Connect4())
        self.assertEqual(m.last_stats["sims"], 137)
        self.assertEqual(sum(m.last_stats["visits"].values()), 137)

    def test_time_budget(self):
        m = MCTSAgent(n_sims=10**9, time_limit=0.05, seed=1)
        m.select_action(Connect4())
        self.assertGreater(m.last_stats["sims"], 10)

    def test_light_rollout_runs(self):
        m = MCTSAgent(n_sims=100, rollout="light", seed=1)
        self.assertIn(m.select_action(Connect4()), range(7))


if __name__ == "__main__":
    unittest.main()

import random
import unittest

from c4ai.game import Connect4, COLS, ROWS, N_CELLS


def play(*cols):
    g = Connect4()
    for c in cols:
        g.step(c)
    return g


class TestRulesAndAPI(unittest.TestCase):
    def test_initial_state(self):
        g = Connect4()
        self.assertEqual(g.legal_actions(), list(range(7)))
        self.assertEqual(g.to_move, 1)
        self.assertFalse(g.is_terminal())
        self.assertIsNone(g.result())

    def test_gravity_and_turns(self):
        g = play(3, 3, 3)
        self.assertEqual([g.cell(r, 3) for r in range(3)], [1, 2, 1])
        self.assertEqual(g.to_move, 2)

    def test_vertical_win(self):
        g = play(0, 1, 0, 1, 0, 1, 0)
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.result(), 1)
        self.assertEqual(g.reward(1), 1)
        self.assertEqual(g.reward(2), -1)

    def test_horizontal_win_player2(self):
        g = play(0, 1, 0, 2, 6, 3, 6, 4)  # X: 0,0,6,6   O: 1,2,3,4
        self.assertEqual(g.result(), 2)

    def test_diagonal_up_win(self):
        # X cần (0,0) (1,1) (2,2) (3,3)
        g = play(0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3)
        self.assertEqual(g.result(), 1)

    def test_diagonal_down_win(self):
        # X cần (3,0) (2,1) (1,2) (0,3)
        g = play(3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0)
        self.assertEqual(g.result(), 1)

    def test_no_false_win_across_column_border(self):
        # 3 quân cuối hàng + 1 quân đầu hàng kế tiếp không được tính là 4 liên tiếp
        g = Connect4.from_grid([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            "XXX...X",
        ])
        self.assertEqual(g.result(), None)
        g2 = Connect4.from_grid([
            ".......",
            ".......",
            ".......",
            ".......",
            "......X",
            "X.....X",
        ])
        self.assertIsNone(g2.result())

    def test_no_wrap_between_top_and_bottom_of_adjacent_columns(self):
        # cột 0 hàng 3,4,5 (3 quân trên cùng) + cột 1 hàng 0 không tạo thành 4
        g = Connect4.from_grid([
            "X......",
            "X......",
            "X......",
            ".......",
            ".......",
            ".X.....",
        ])
        self.assertIsNone(g.result())

    def test_draw(self):
        # ván hòa đã biết: lấp đầy bàn không có 4 liên tiếp
        rows = [
            "XOXOXOX",
            "XOXOXOX",
            "OXOXOXO",
            "OXOXOXO",
            "XOXOXOX",
            "XOXOXOX",
        ]
        g = Connect4.from_grid(rows)
        # (cột XOXO... theo từng cột thay phiên nên không có dọc 4)
        self.assertEqual(g.result(), 0)
        self.assertEqual(g.reward(1), 0)
        self.assertEqual(g.legal_actions(), [])

    def test_win_on_last_cell_is_win_not_draw(self):
        # nước thứ 42 tạo 4 liên tiếp phải tính là THẮNG (lỗi của code cũ)
        rng = random.Random(0)
        found = 0
        for _ in range(20000):
            g = Connect4()
            while not g.is_terminal():
                g.step(rng.choice(g.legal_actions()))
            if len(g.moves) == N_CELLS and g.winner:
                found += 1
                self.assertEqual(g.result(), g.winner)
        self.assertGreater(found, 0, "không gặp được ván thắng ở nước 42, tăng số ván")

    def test_illegal_moves_rejected(self):
        g = play(*([0] * 6))
        with self.assertRaises(ValueError):
            g.step(0)  # cột đầy
        with self.assertRaises(ValueError):
            g.step(7)
        with self.assertRaises(ValueError):
            g.step(-1)
        g2 = play(0, 1, 0, 1, 0, 1, 0)
        with self.assertRaises(ValueError):
            g2.step(3)  # ván đã kết thúc
        with self.assertRaises(ValueError):
            Connect4().undo()

    def test_undo_restores_exactly(self):
        rng = random.Random(1)
        for _ in range(200):
            g = Connect4()
            snapshots = []
            while not g.is_terminal():
                snapshots.append((g.key(), g.mask, g.to_move, g.winner, g.heights[:]))
                g.step(rng.choice(g.legal_actions()))
            while snapshots:
                g.undo()
                s = snapshots.pop()
                self.assertEqual((g.key(), g.mask, g.to_move, g.winner, g.heights), s)

    def test_clone_is_independent(self):
        g = play(3, 3)
        c = g.clone()
        c.step(4)
        self.assertEqual(len(g.moves), 2)
        self.assertEqual(len(c.moves), 3)

    def test_render(self):
        txt = play(3).render()
        self.assertIn("X", txt)
        self.assertEqual(len(txt.splitlines()), ROWS + 1)

    def test_distinct_positions_match_known_counts(self):
        # số thế cờ khác nhau sau n nước (không có thắng trước nước 7):
        # 1, 7, 49, 238, 1120, 4263  (dãy đã biết cho Connect Four 7x6)
        expected = [1, 7, 49, 238, 1120, 4263]
        level = {Connect4().key(): []}
        for n in range(1, len(expected)):
            nxt = {}
            for moves in level.values():
                g = Connect4.from_moves(moves)
                for a in g.legal_actions():
                    g.step(a)
                    nxt.setdefault(g.key(), moves + [a])
                    g.undo()
            level = nxt
            self.assertEqual(len(level), expected[n])

    def test_random_games_agree_with_naive_checker(self):
        """So sánh bitboard với bộ kiểm tra thắng ngây thơ trên ma trận."""

        def naive_winner(grid):
            for r in range(ROWS):
                for c in range(COLS):
                    p = grid[r][c]
                    if not p:
                        continue
                    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                        ok = True
                        for k in range(1, 4):
                            rr, cc = r + dr * k, c + dc * k
                            if not (0 <= rr < ROWS and 0 <= cc < COLS and grid[rr][cc] == p):
                                ok = False
                                break
                        if ok:
                            return p
            return 0

        rng = random.Random(2)
        for _ in range(3000):
            g = Connect4()
            while not g.is_terminal():
                g.step(rng.choice(g.legal_actions()))
                self.assertEqual(g.winner, naive_winner(g.to_grid()))


if __name__ == "__main__":
    unittest.main()

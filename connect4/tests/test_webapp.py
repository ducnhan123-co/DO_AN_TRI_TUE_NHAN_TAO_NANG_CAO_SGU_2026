import json
import threading
import unittest
import urllib.error
import urllib.request

from c4ai.webapp import BadRequest, Handler, ai_move, replay, state_payload, winning_cells
from c4ai.game import Connect4
from http.server import ThreadingHTTPServer


class TestWebLogic(unittest.TestCase):
    def test_state_initial(self):
        s = state_payload(replay([]))
        self.assertEqual(s["to_move"], 1)
        self.assertEqual(s["legal"], list(range(7)))
        self.assertFalse(s["over"])
        self.assertEqual(len(s["grid"]), 6)

    def test_win_cells(self):
        g = replay([0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(g.winner, 1)
        self.assertEqual(winning_cells(g), [[2, 0], [3, 0], [4, 0], [5, 0]])

    def test_rejects_bad_input(self):
        for bad in ([7], [-1], [0] * 7, "abc", [1.5], [True], [0, 1, 0, 1, 0, 1, 0, 3]):
            with self.assertRaises(BadRequest):
                replay(bad)

    def test_ai_moves_for_each_type(self):
        for cfg in ({"type": "random"}, {"type": "heuristic"}, {"type": "alphabeta", "depth": 3},
                    {"type": "mcts", "sims": 50}):
            r = ai_move([3, 3], cfg)
            self.assertIn(r["action"], range(7))
        with self.assertRaises(BadRequest):
            ai_move([], {"type": "nope"})
        with self.assertRaises(BadRequest):
            ai_move([0, 1, 0, 1, 0, 1, 0], {"type": "random"})  # ván đã kết thúc

    def test_limits_are_clamped(self):
        r = ai_move([], {"type": "alphabeta", "depth": 999})
        self.assertLessEqual(r["stats"]["depth"], 9)


class TestHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def post(self, path, obj):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(obj).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_index_served(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/") as r:
            html = r.read().decode("utf-8")
        self.assertIn("Connect Four AI Lab", html)

    def test_endpoints(self):
        code, body = self.post("/api/state", {"moves": [3]})
        self.assertEqual((code, body["to_move"]), (200, 2))
        code, body = self.post("/api/ai", {"moves": [3], "agent": {"type": "heuristic"}})
        self.assertEqual(code, 200)
        self.assertIn(body["action"], range(7))
        self.assertEqual(self.post("/api/state", {"moves": [9]})[0], 400)
        self.assertEqual(self.post("/api/nothing", {})[0], 404)


if __name__ == "__main__":
    unittest.main()

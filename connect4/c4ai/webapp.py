"""Giao diện web: server HTTP nhỏ (thư viện chuẩn) + trang ui/index.html.

Server không giữ trạng thái: trình duyệt gửi danh sách nước đi, server dựng lại ván
bằng engine thật, rồi trả trạng thái hoặc nước đi của agent (Random / Heuristic /
Alpha-Beta / MCTS) -> giao diện luôn dùng đúng code đã được kiểm thử và đánh giá.

API (JSON, POST):
  /api/state {moves}                 -> trạng thái bàn, người thắng, ô thắng
  /api/ai    {moves, agent:{type,..}} -> {action, ms, name, stats}
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agents import AlphaBetaAgent, HeuristicAgent, MCTSAgent, RandomAgent
from .game import COLS, ROWS, Connect4

UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
MAX_BODY = 4096


class BadRequest(Exception):
    pass


def replay(moves) -> Connect4:
    if not isinstance(moves, list) or len(moves) > COLS * ROWS or not all(isinstance(m, int) and not isinstance(m, bool) for m in moves):
        raise BadRequest("moves phải là danh sách số nguyên (cột 0-6)")
    try:
        return Connect4.from_moves(moves)
    except ValueError as e:
        raise BadRequest(str(e))


def winning_cells(g: Connect4) -> list[list[int]]:
    """Các ô (hàng từ trên xuống, cột) thuộc đường 4 quân thắng; [] nếu không có."""
    if not g.winner:
        return []
    grid = g.to_grid()
    p = g.winner
    cells = set()
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                line = [(r + k * dr, c + k * dc) for k in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS and grid[rr][cc] == p for rr, cc in line):
                    cells.update(line)
    return sorted([list(x) for x in cells])


def state_payload(g: Connect4) -> dict:
    return {
        "moves": list(g.moves),
        "to_move": g.to_move,
        "grid": g.to_grid(),
        "legal": g.legal_actions(),
        "winner": g.winner,
        "over": g.is_terminal(),
        "result": g.result(),
        "win_cells": winning_cells(g),
    }


def build_agent(cfg: dict):
    kind = cfg.get("type")
    if kind == "random":
        return RandomAgent()
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "alphabeta":
        depth = int(cfg.get("depth", 5))
        return AlphaBetaAgent(depth=max(1, min(depth, 9)))
    if kind == "mcts":
        sims = int(cfg.get("sims", 1000))
        return MCTSAgent(n_sims=max(10, min(sims, 20000)))
    raise BadRequest(f"agent không hợp lệ: {kind!r}")


def ai_move(moves, cfg) -> dict:
    g = replay(moves)
    if g.is_terminal():
        raise BadRequest("Ván đã kết thúc")
    try:
        agent = build_agent(cfg or {})
    except (TypeError, ValueError) as e:
        raise BadRequest(str(e))
    action, dt, stats = agent.timed_action(g)
    return {"action": action, "ms": round(dt * 1000, 2), "name": agent.name, "stats": stats}


class Handler(BaseHTTPRequestHandler):
    server_version = "Connect4UI/1.0"

    def log_message(self, fmt, *args):  # im lặng
        pass

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(UI_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > MAX_BODY:
                raise BadRequest("body quá lớn")
            data = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(data, dict):
                raise BadRequest("body phải là object JSON")
            if self.path == "/api/state":
                self._json(200, state_payload(replay(data.get("moves", []))))
            elif self.path == "/api/ai":
                self._json(200, ai_move(data.get("moves", []), data.get("agent", {})))
            else:
                self._json(404, {"error": "not found"})
        except BadRequest as e:
            self._json(400, {"error": str(e)})
        except (json.JSONDecodeError, ValueError):
            self._json(400, {"error": "JSON không hợp lệ"})
        except Exception as e:  # pragma: no cover
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def make_server(host="127.0.0.1", port=8765, tries=20) -> ThreadingHTTPServer:
    for p in range(port, port + tries):
        try:
            return ThreadingHTTPServer((host, p), Handler)
        except OSError:
            continue
    raise SystemExit(f"Không tìm được cổng trống từ {port}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Giao diện web Connect Four AI")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="mặc định chỉ cho máy này (127.0.0.1)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    srv = make_server(a.host, a.port)
    url = f"http://{a.host}:{srv.server_address[1]}/"
    print(f"Connect Four AI đang chạy tại {url}  (Ctrl+C để dừng)")
    if not a.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()

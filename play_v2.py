"""
play.py - Chạy chương trình chơi Connect 4 (giao diện nằm trong gui.py).

    python play.py                          # hiện menu 1-5 như bản cũ
    python play.py --mode agent             # vào thẳng: Người vs AI đã train (bàn 8x9)
    python play.py --mode minimax --depth 4 # Người vs Minimax
    python play.py --mode aivai             # xem hai AI đấu nhau
    python play.py --rows 6 --cols 7        # bàn 6x7 cổ điển
    python play.py --cell 44                # thu nhỏ nếu màn hình thấp
Các mode: random | tactical | minimax | agent | aivai
"""
import argparse
import os
import subprocess
import sys


def _set_board_from_argv():
    """--rows/--cols phải được đọc TRƯỚC khi import c4core (kích thước bàn cố định lúc import)."""
    for flag, var in (("--rows", "C4_ROWS"), ("--cols", "C4_COLS")):
        if flag in sys.argv:
            os.environ[var] = sys.argv[sys.argv.index(flag) + 1]


_set_board_from_argv()

from c4core import COLS, ROWS, MinimaxPlayer, RandomPlayer, TacticalPlayer  # noqa: E402

MODES = ("random", "tactical", "minimax", "agent", "aivai")


def model_path(rows=ROWS, cols=COLS):
    return f"models/agent_{rows}x{cols}.zip"


def exists(p):
    return os.path.exists(p) or os.path.exists(p + ".zip")


def load_agent(path):
    from sb3_contrib import MaskablePPO
    from c4env import TorchPlayer
    m = MaskablePPO.load(path)
    return TorchPlayer(m.policy, deterministic=True, flip_avg=True, name=os.path.basename(path))


def ask_model(prompt, default):
    p = input(f"{prompt} [{default}]: ").strip().strip('"') or default
    return p


def ask_depth(default=4):
    s = input(f"Độ sâu minimax (1-8) [{default}]: ").strip()
    return min(max(int(s), 1), 8) if s.isdigit() else default


def build(mode, opts):
    """Trả về dict tham số cho gui.App, hoặc None nếu thiếu model."""
    kw = dict(me=None, labels=("BẠN", "AI"))
    if mode == "random":
        kw.update(opponent=RandomPlayer(), subtitle=f"VS RANDOM ({ROWS}x{COLS})")
    elif mode == "tactical":
        kw.update(opponent=TacticalPlayer(), subtitle=f"VS TACTICAL BOT ({ROWS}x{COLS})")
    elif mode == "minimax":
        d = opts["depth"]
        kw.update(opponent=MinimaxPlayer(d), subtitle=f"VS MINIMAX d={d} ({ROWS}x{COLS})")
    elif mode == "agent":
        p = opts["model"] or model_path()
        if not exists(p):
            print("Không tìm thấy model:", p)
            return None
        kw.update(opponent=load_agent(p), subtitle=f"VS PPO NEURAL AI ({ROWS}x{COLS})")
    elif mode == "aivai":
        pa, pb = opts["model"] or model_path(), opts["model_b"] or opts["model"] or model_path()
        for p in (pa, pb):
            if not exists(p):
                print("Không tìm thấy model:", p)
                return None
        kw.update(me=load_agent(pa), opponent=load_agent(pb), labels=("AI A", "AI B"),
                  subtitle=f"AI vs AI ({ROWS}x{COLS})", ai_delay=0.6)
    return kw


def run_mode(mode, opts):
    """Mở cửa sổ. Nếu người dùng bấm 'đổi bàn' thì chạy lại chương trình với bàn mới và thoát."""
    kw = build(mode, opts)
    if kw is None:
        return
    import pygame
    from gui import App

    def can_switch(nr, nc):
        if mode in ("agent", "aivai") and not exists(model_path(nr, nc)):
            return f"Không tìm thấy {model_path(nr, nc)}"
        return None

    app = App(cell=opts["cell"], can_switch=can_switch, **kw)
    res = app.run()
    pygame.quit()
    if res:
        nr, nc = res
        cmd = [sys.executable, os.path.abspath(__file__), "--mode", mode, "--depth", str(opts["depth"]),
               "--rows", str(nr), "--cols", str(nc), "--cell", str(opts["cell"])]
        sys.exit(subprocess.call(cmd))


def menu(opts):
    while True:
        print("\n" + "=" * 42)
        print(f"     CONNECT 4 ({ROWS}x{COLS}) - AI MENU")
        print("=" * 42)
        print("1. Human vs Random")
        print("2. Human vs Tactical (thắng/chặn nước 1)")
        print("3. Human vs Minimax")
        print("4. Human vs trained agent")
        print("5. Agent vs Agent (xem đấu)")
        print("0. Exit")
        print("=" * 42)
        c = input("Choose mode: ").strip()
        o = dict(opts)
        if c == "1":
            run_mode("random", o)
        elif c == "2":
            run_mode("tactical", o)
        elif c == "3":
            o["depth"] = ask_depth(opts["depth"])
            run_mode("minimax", o)
        elif c == "4":
            o["model"] = ask_model("Đường dẫn model", opts["model"] or model_path())
            run_mode("agent", o)
        elif c == "5":
            o["model"] = ask_model("Model Agent A (đỏ/cam)", opts["model"] or model_path())
            o["model_b"] = ask_model("Model Agent B", o["model"])
            run_mode("aivai", o)
        elif c == "0":
            print("Exiting Connect 4.")
            return
        else:
            print("Invalid choice. Please select 0-5.")


def main():
    ap = argparse.ArgumentParser(description="Connect 4 - AI Strategy Arena")
    ap.add_argument("--mode", choices=MODES, default=None, help="bỏ trống = hiện menu")
    ap.add_argument("--depth", type=int, default=4, help="độ sâu minimax")
    ap.add_argument("--model", default=None, help="mặc định models/agent_<hàng>x<cột>.zip")
    ap.add_argument("--model-b", dest="model_b", default=None, help="model của AI B (mode aivai)")
    ap.add_argument("--cell", type=int, default=0, help="cỡ ô (px); 0 = tự chọn theo màn hình")
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--cols", type=int, default=COLS)
    a = ap.parse_args()
    opts = dict(depth=a.depth, model=a.model, model_b=a.model_b, cell=a.cell)
    if a.mode:
        run_mode(a.mode, opts)
    else:
        menu(opts)


if __name__ == "__main__":
    main()
"""
play.py - Chạy chương trình chơi Connect 4 (giao diện nằm trong gui.py).

    python play.py                          # hiện menu 1-5 như bản cũ
    python play.py --mode agent             # vào thẳng: Người vs AI đã train (bàn 8x9)
    python play.py --mode minimax --depth 4 # Người vs Minimax
    python play.py --mode agent --search 4  # AI + minimax d4 loại nước tệ (mạnh hơn)
    python play.py --level 5                # thang 8 cấp độ (trong game bấm phím 1-8 để đổi cấp)
    python play.py --mode aivai             # xem hai AI đấu nhau
    python play.py --rows 6 --cols 7        # bàn 6x7 cổ điển
    python play.py --cell 44                # thu nhỏ nếu màn hình thấp
Các mode: random | tactical | minimax | agent | aivai | level
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

MODES = ("random", "tactical", "minimax", "agent", "aivai", "level")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def model_path(rows=ROWS, cols=COLS):
    """Đường dẫn TUYỆT ĐỐI tới model mặc định => chạy từ thư mục nào cũng được."""
    return os.path.join(BASE_DIR, "models", f"agent_{rows}x{cols}.zip")


def exists(p):
    return os.path.exists(p) or os.path.exists(p + ".zip")


def load_agent(path, search=0):
    from c4env import TorchPlayer, check_board, load_model
    m = check_board(load_model(path))
    return TorchPlayer(m.policy, deterministic=True, flip_avg=True, search_depth=search, name=os.path.basename(path))


def safe_load_agent(path, search=0):
    """Nạp model; file hỏng / sai kích thước bàn => in lỗi và trả None (quay lại menu, không sập)."""
    try:
        return load_agent(path, search)
    except Exception as e:  # noqa: BLE001
        print(f"Model lỗi: {path}\n  -> {type(e).__name__}: {e}")
        print(f"  (model phải được train cho đúng bàn {ROWS}x{COLS})")
        return None


def ask_model(prompt, default):
    p = input(f"{prompt} [{default}]: ").strip().strip('"') or default
    return p


def ask_search(default=0):
    s = input("Tăng sức mạnh AI bằng tìm kiếm minimax? Độ sâu (0 = tắt, 4 = khuyến nghị) [4]: ").strip()
    return min(max(int(s), 0), 8) if s.isdigit() else 4


def ask_depth(default=4):
    s = input(f"Độ sâu minimax (1-8) [{default}]: ").strip()
    return min(max(int(s), 1), 8) if s.isdigit() else default


def on_level(n):
    """Tạo đối thủ cho cấp n (1-8). Trả về (người chơi, tiêu đề) hoặc chuỗi thông báo lỗi."""
    import levels
    try:
        return levels.make_level(n), f"{levels.describe(n)} ({ROWS}x{COLS})"
    except (FileNotFoundError, ValueError) as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"Lỗi nạp cấp {n}: {type(e).__name__}: {e}"


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
    elif mode == "level":
        res = on_level(opts["level"])
        if isinstance(res, str):
            print(res)
            return None
        kw.update(opponent=res[0], subtitle=res[1], on_level=on_level)
    elif mode == "agent":
        p = opts["model"] or model_path()
        if not exists(p):
            print("Không tìm thấy model:", p)
            return None
        ag = safe_load_agent(p, opts["search"])
        if ag is None:
            return None
        tag = f" + SEARCH d{opts['search']}" if opts["search"] else ""
        kw.update(opponent=ag, subtitle=f"VS PPO NEURAL AI{tag} ({ROWS}x{COLS})")
    elif mode == "aivai":
        pa, pb = opts["model"] or model_path(), opts["model_b"] or opts["model"] or model_path()
        for p in (pa, pb):
            if not exists(p):
                print("Không tìm thấy model:", p)
                return None
        a_ag, b_ag = safe_load_agent(pa, opts["search"]), safe_load_agent(pb, opts["search"])
        if a_ag is None or b_ag is None:
            return None
        kw.update(me=a_ag, opponent=b_ag, labels=("AI A", "AI B"),
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
        cmd = [sys.executable, os.path.abspath(__file__), "--mode", mode, "--depth", str(opts["depth"]), "--search", str(opts["search"]), "--level", str(opts["level"]),
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
        print("6. Human vs Level 1-8 (thang độ khó)")
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
            o["search"] = ask_search(opts["search"])
            run_mode("agent", o)
        elif c == "5":
            o["model"] = ask_model("Model Agent A (đỏ/cam)", opts["model"] or model_path())
            o["model_b"] = ask_model("Model Agent B", o["model"])
            o["search"] = ask_search(opts["search"])
            run_mode("aivai", o)
        elif c == "6":
            s = input("Chọn cấp 1-8 (trong game bấm phím 1-8 để đổi cấp) [1]: ").strip()
            o["level"] = min(max(int(s), 1), 8) if s.isdigit() else 1
            run_mode("level", o)
        elif c == "0":
            print("Exiting Connect 4.")
            return
        else:
            print("Invalid choice. Please select 0-6.")


def main():
    ap = argparse.ArgumentParser(description="Connect 4 - AI Strategy Arena")
    ap.add_argument("--mode", choices=MODES, default=None, help="bỏ trống = hiện menu")
    ap.add_argument("--depth", type=int, default=4, help="độ sâu minimax")
    ap.add_argument("--model", default=None, help="mặc định models/agent_<hàng>x<cột>.zip")
    ap.add_argument("--model-b", dest="model_b", default=None, help="model của AI B (mode aivai)")
    ap.add_argument("--search", type=int, default=0,
                    help="chế độ lai cho AI: minimax độ sâu N loại các nước dẫn tới thế thua (0 = tắt)")
    ap.add_argument("--level", type=int, default=0, help="cấp độ 1-8 (tự vào mode level)")
    ap.add_argument("--cell", type=int, default=0, help="cỡ ô (px); 0 = tự chọn theo màn hình")
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--cols", type=int, default=COLS)
    a = ap.parse_args()
    opts = dict(depth=a.depth, model=a.model, model_b=a.model_b, cell=a.cell, search=a.search,
                level=a.level or 1)
    if a.level and not a.mode:
        a.mode = "level"
    try:
        if a.mode:
            run_mode(a.mode, opts)
        else:
            menu(opts)
    except (EOFError, KeyboardInterrupt):
        print("\nĐã thoát.")


if __name__ == "__main__":
    main()

"""boardenv.py - đặt kích thước bàn từ --rows/--cols (hoặc --board 8x9 / 6x7) TRƯỚC khi import c4core.

c4core đọc C4_ROWS/C4_COLS ngay lúc import nên mọi script có tham số bàn phải `import boardenv` ở dòng đầu tiên.
"""
import os
import sys


def _set():
    argv = sys.argv
    for flag, var in (("--rows", "C4_ROWS"), ("--cols", "C4_COLS")):
        if flag in argv and argv.index(flag) + 1 < len(argv):
            os.environ[var] = argv[argv.index(flag) + 1]
    if "--board" in argv and argv.index("--board") + 1 < len(argv):
        r, c = argv[argv.index("--board") + 1].lower().split("x")
        os.environ["C4_ROWS"], os.environ["C4_COLS"] = r, c  # dạng HÀNGxCỘT, vd 6x7 hoặc 8x9


_set()

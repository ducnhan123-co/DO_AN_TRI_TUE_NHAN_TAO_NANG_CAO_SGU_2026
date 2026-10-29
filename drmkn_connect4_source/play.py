import os
import sys
import importlib.util

# Trỏ vào thư mục drmkn_connect4_src
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, "drmkn_connect4_src")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

os.chdir(src_dir)

target_path = os.path.join(src_dir, "play.py")
spec = importlib.util.spec_from_file_location("drmkn_play", target_path)
_mod = importlib.util.module_from_spec(spec)
sys.modules["drmkn_play"] = _mod
spec.loader.exec_module(_mod)

play = _mod.play
get_agent_for_board = _mod.get_agent_for_board

if __name__ == "__main__":
    play()

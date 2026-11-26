"""Ablation cho tác tử PPO CHÍNH (models/agent_8x9.zip): bỏ mask chiến thuật / bỏ lật gương / thêm tìm kiếm lai,
rồi đo điểm với Random, Tactical, Minimax d2, Minimax d4 (tuỳ chọn d6). Tái tạo Bảng 4.4 và Hình 4.5 của báo cáo.

    python tools/ablation_agent.py models/agent_8x9.zip
    python tools/ablation_agent.py models/agent_8x9.zip --variants full,no_flip,no_mask --games 100
    python tools/ablation_agent.py models/agent_8x9.zip --variants all --opponents random,tactical,minimax2,minimax4,minimax6

Biến thể: full | no_flip | no_mask | no_mask_no_flip | full_search2 | full_search4   (mặc định 4 biến thể đầu)
Kết quả -> results/runs/<mã>_ablation-agent/ : ablation_agent.csv, summary.md, app_cfg.json, params_cfg.json, run.log.txt
Vẽ hình:  python tools/plot_results.py ablation-agent <thư mục kết quả>
Lưu ý: nên chạy tuần tự trên một máy; cùng seed => cùng kết quả (mạng + minimax đều xác định).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # thư mục gốc dự án
import argparse, time, warnings
warnings.filterwarnings("ignore")
import torch
from c4 import expio
from c4.agent_eval import (ABLATION_FIELDS, DEFAULT_OPPONENTS, VARIANTS, ablation_markdown, parse_opponents,
                           parse_variants, run_agent_ablation)
from c4.core import COLS, ROWS
from c4.env import TorchPlayer, check_board, load_model

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model")
ap.add_argument("--variants", default="default", help="danh sách biến thể, 'default' hoặc 'all' (xem đầu file)")
ap.add_argument("--opponents", default="random,tactical,minimax2,minimax4", help="random,tactical,minimaxN (hoặc 'all')")
ap.add_argument("--games", type=int, default=100)
ap.add_argument("--d6-games", type=int, default=None, help="số ván với minimax độ sâu >= 6 (mặc định max(20, games//3))")
ap.add_argument("--opening-plies", type=int, default=2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default=None, help="thư mục kết quả (mặc định results/runs/<mã>)")
ap.add_argument("--goal", default=""); ap.add_argument("--hypothesis", default=""); ap.add_argument("--author", default="")
a = ap.parse_args()
if a.games < 1:
    ap.error("--games phải >= 1")
try:
    variants, opps = parse_variants(a.variants), parse_opponents(a.opponents)
except ValueError as e:
    ap.error(str(e))

torch.set_num_threads(1)
m = check_board(load_model(a.model))
base = TorchPlayer(m.policy, deterministic=True, flip_avg=True, shield=True, name="ppo_main")
board = f"{ROWS}x{COLS}"
run_id = expio.new_run_id("ablation", board, "agent" + (f"_{a.tag}" if a.tag else ""))
out = a.out or expio.make_run_dir(run_id)
os.makedirs(out, exist_ok=True)
t0, started = time.time(), expio.now()
with expio.Tee(os.path.join(out, "run.log.txt")):
    print(f"=== {run_id} -> {out}\nmodel {a.model} | bàn {board} | biến thể {variants} | đối thủ {opps} | {a.games} ván/ô")
    rows = run_agent_ablation(base, variants, opps, a.games, a.opening_plies, a.seed, a.d6_games)
    expio.write_csv(os.path.join(out, "ablation_agent.csv"), rows, ABLATION_FIELDS)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(ablation_markdown(rows))
    expio.write_json(os.path.join(out, "app_cfg.json"), {
        "run_id": run_id, "kind": "ablation-agent", "board": board, "model": a.model,
        "model_sha256": expio.sha256_file(a.model), "variants": {v: VARIANTS[v] for v in variants}, "opponents": opps,
        "games": a.games, "d6_games": a.d6_games, "opening_plies": a.opening_plies, "seed": a.seed,
        "goal": a.goal, "hypothesis": a.hypothesis, "author": a.author, "started": started, "ended": expio.now(),
        "duration_s": round(time.time() - t0, 1), "command": ["python"] + sys.argv, "code_hash": expio.code_hash(),
        "environment": expio.environment()})
    print("\n" + ablation_markdown(rows))
    print("Đã lưu:", out)

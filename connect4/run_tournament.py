#!/usr/bin/env python3
"""Round-robin giữa 4 tác tử cấp 1-4, theo 2 chế độ ngân sách tính toán.

  default  : cấu hình mặc định minh bạch (Alpha-Beta độ sâu 5, MCTS 1000 mô phỏng)
  time     : cùng ngân sách THỜI GIAN mỗi nước (mặc định 50 ms) cho Alpha-Beta
             (iterative deepening) và MCTS; Random/Heuristic không dùng ngân sách.

Ví dụ:
  python run_tournament.py --mode default
  python run_tournament.py --mode time --ms 50
  python run_tournament.py --mode all --games-per-seed 100 --seeds 0 1
"""

import argparse
import os

from c4ai.agents import RandomAgent, HeuristicAgent, AlphaBetaAgent, MCTSAgent
from c4ai.tournament import AgentSpec, round_robin


def specs_default(depth, sims):
    return [
        AgentSpec("Random", lambda s: RandomAgent(s)),
        AgentSpec("Heuristic", lambda s: HeuristicAgent(s)),
        AgentSpec("AlphaBeta", lambda s: AlphaBetaAgent(depth=depth, seed=s)),
        AgentSpec("MCTS", lambda s: MCTSAgent(n_sims=sims, seed=s)),
    ]


def specs_time(ms):
    t = ms / 1000
    return [
        AgentSpec("Random", lambda s: RandomAgent(s)),
        AgentSpec("Heuristic", lambda s: HeuristicAgent(s)),
        AgentSpec("AlphaBeta", lambda s: AlphaBetaAgent(time_limit=t, seed=s)),
        AgentSpec("MCTS", lambda s: MCTSAgent(n_sims=10**9, time_limit=t, seed=s)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["default", "time", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--games-per-seed", type=int, default=100, help="chẵn; mỗi khai cuộc chơi 2 lần")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--depth", type=int, default=5, help="độ sâu Alpha-Beta (chế độ default)")
    ap.add_argument("--sims", type=int, default=1000, help="số mô phỏng MCTS (chế độ default)")
    ap.add_argument("--ms", type=float, default=50, help="ngân sách ms/nước (chế độ time)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    assert a.games_per_seed % 2 == 0

    if a.mode in ("default", "all"):
        round_robin(specs_default(a.depth, a.sims), os.path.join(a.out, "tournament_default"), "default",
                    a.seeds, a.games_per_seed, a.opening_plies,
                    extra_config={"alphabeta_depth": a.depth, "mcts_sims": a.sims})
    if a.mode in ("time", "all"):
        round_robin(specs_time(a.ms), os.path.join(a.out, f"tournament_time_{int(a.ms)}ms"), f"time-{int(a.ms)}ms",
                    a.seeds, a.games_per_seed, a.opening_plies,
                    extra_config={"time_budget_ms": a.ms})


if __name__ == "__main__":
    main()

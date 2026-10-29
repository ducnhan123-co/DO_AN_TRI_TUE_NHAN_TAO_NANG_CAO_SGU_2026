#!/usr/bin/env python3
"""Demo chơi trong terminal: người-máy hoặc máy-máy, chọn cấp độ AI 1-4.

  python play.py --ai 4                    # bạn (X, đi trước) vs MCTS
  python play.py --ai 3 --depth 6 --second # bạn đi sau, Alpha-Beta độ sâu 6
  python play.py --ai 2 --ai2 4            # máy-máy: Heuristic (X) vs MCTS (O)
Cấp: 1 Random, 2 Heuristic, 3 Alpha-Beta, 4 MCTS.
"""

import argparse
import time

from c4ai.agents import RandomAgent, HeuristicAgent, AlphaBetaAgent, MCTSAgent
from c4ai.game import Connect4


def build(level, args, seed):
    if level == 1:
        return RandomAgent(seed)
    if level == 2:
        return HeuristicAgent(seed)
    if level == 3:
        return AlphaBetaAgent(depth=args.depth, seed=seed)
    if level == 4:
        return MCTSAgent(n_sims=args.sims, seed=seed)
    raise SystemExit("Cấp phải là 1..4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", type=int, default=3, help="cấp AI đối thủ (1-4)")
    ap.add_argument("--ai2", type=int, default=None, help="nếu có: chạy máy-máy, cấp AI đi sau (O)")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--second", action="store_true", help="người chơi đi sau")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    g = Connect4()
    if args.ai2 is not None:
        agents = {1: build(args.ai, args, args.seed), 2: build(args.ai2, args, args.seed)}
        human = None
    else:
        ai = build(args.ai, args, args.seed)
        human = 2 if args.second else 1
        agents = {3 - human: ai}
        print(f"Bạn là {'O (đi sau)' if args.second else 'X (đi trước)'} - đối thủ: {ai.name}")

    while not g.is_terminal():
        print()
        g.render()
        p = g.to_move
        if p == human:
            while True:
                try:
                    a = int(input(f"Lượt bạn ({'XO'[p - 1]}), chọn cột 0-6: "))
                except ValueError:
                    continue
                if g.is_legal(a):
                    break
                print("Nước đi không hợp lệ.")
        else:
            a, dt, st = agents[p].timed_action(g)
            extra = f"nodes={st.get('nodes')}" if st.get("sims", 0) == 0 else f"sims={st.get('sims')}"
            print(f"{agents[p].name} ({'XO'[p - 1]}) chọn cột {a}  [{dt * 1000:.0f} ms, {extra}]")
            if args.ai2 is not None:
                time.sleep(0.3)
        g.step(a)
    print()
    g.render()
    r = g.result()
    print("Hòa!" if r == 0 else f"{'XO'[r - 1]} thắng!" + (" (bạn thắng)" if r == human else ""))


if __name__ == "__main__":
    main()

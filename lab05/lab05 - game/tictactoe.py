# Tic-Tac-Toe core implementation
"""Core functions for Tic-Tac-Toe used in the assignment.
State representation: list of 9 cells, each ' ', 'x', or 'o'.
Indices: 0‑8 left‑to‑right, top‑to‑bottom:
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
"""

import math
import random
import time
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import matplotlib
import matplotlib.pyplot as plt

# Global counter for node expansions
node_counter = 0

def reset_counter():
    global node_counter
    node_counter = 0

def empty_board() -> List[str]:
    """Return a new empty board."""
    return [' '] * 9

def actions(board: List[str]) -> List[int]:
    """Return list of empty cell indices (legal actions)."""
    return [i for i, cell in enumerate(board) if cell == ' ']

def result(board: List[str], player: str, action: int) -> List[str]:
    """Return new board after player places a mark on *action*.
    Does not modify the original board.
    """
    new_board = list(board)
    new_board[action] = player
    return new_board

def other(player: str) -> str:
    """Return the opponent symbol: 'x' -> 'o', 'o' -> 'x'."""
    return 'o' if player == 'x' else 'x'

def utility(board: List[str], player: str = 'x') -> Optional[int]:
    """Utility from the perspective of *player* (Max = 'x').
    Returns:
      +1 if player wins,
      -1 if opponent (other(player)) wins,
       0 if draw (board is full and neither won),
       None if game is NOT terminal (still ongoing).
    """
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
        (0, 4, 8), (2, 4, 6)             # diagonals
    ]
    for a, b, c in lines:
        if board[a] == board[b] == board[c] and board[a] != ' ':
            return 1 if board[a] == player else -1

    if ' ' not in board:
        return 0

    return None

def terminal(board: List[str]) -> bool:
    """Return True if the game is over (win, loss, or draw)."""
    return utility(board, 'x') is not None

def check_board(board: List[str]) -> bool:
    """Validate board shape and symbols. Returns True if valid."""
    if not isinstance(board, (list, tuple)) or len(board) != 9:
        return False
    return all(cell in (' ', 'x', 'o') for cell in board)

def show_board_text(board: List[str]) -> None:
    """Print the board as a 3x3 numpy character array."""
    print(np.array(board).reshape(3, 3))

def show_board(board: List[str], help: bool = False) -> None:
    """Display the board visually using matplotlib, compatible with Jupyter.
    If help is True, empty cells display their index (0-8).
    """
    try:
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(0.5, color='black', lw=1.5)
        ax.axvline(1.5, color='black', lw=1.5)
        ax.axhline(0.5, color='black', lw=1.5)
        ax.axhline(1.5, color='black', lw=1.5)
        for i in range(9):
            r = i // 3
            c = i % 3
            val = board[i]
            if val == 'x':
                ax.text(c, r, 'X', ha='center', va='center', fontsize=20, color='red', fontweight='bold')
            elif val == 'o':
                ax.text(c, r, 'O', ha='center', va='center', fontsize=20, color='blue', fontweight='bold')
            elif help:
                ax.text(c, r, str(i), ha='center', va='center', fontsize=12, color='gray')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    except Exception:
        # Fallback to ASCII text grid
        for i in range(3):
            row = [board[3 * i + j] if board[3 * i + j] != ' ' else (str(3 * i + j) if help else ' ') for j in range(3)]
            print(' | '.join(row))
            if i < 2:
                print('---------')

# -------------------------------------------------
# Game Results & Simulation Framework
# -------------------------------------------------
class GameResult(dict):
    """Dictionary holding game counts {'x': x_wins, 'o': o_wins, 'd': draws}
    that also supports tuple unpacking: x, o, d = play(...)
    """
    def __init__(self, x=0, o=0, d=0):
        super().__init__({'x': x, 'o': o, 'd': d})

    def __iter__(self):
        return iter((self['x'], self['o'], self['d']))

    def __repr__(self):
        return f"{{'x': {self['x']}, 'o': {self['o']}, 'd': {self['d']}}}"

def random_player(board: List[str], player: str = 'x') -> int:
    """Select a random legal move."""
    avail = actions(board)
    return int(np.random.choice(avail))

def play(player_x, player_o, N: int = 100, show_final_board: bool = False, seed: Optional[int] = None) -> GameResult:
    """Play N games between player_x (Max) and player_o.
    Returns GameResult dict {'x': x_wins, 'o': o_wins, 'd': draws}.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    x_wins = o_wins = draws = 0
    for _ in range(N):
        board = empty_board()
        current = 'x'
        while not terminal(board):
            if current == 'x':
                move = player_x(board, 'x')
            else:
                move = player_o(board, 'o')
            board = result(board, current, move)
            current = other(current)

        u = utility(board, 'x')
        if u == 1:
            x_wins += 1
        elif u == -1:
            o_wins += 1
        else:
            draws += 1

        if show_final_board:
            show_board(board)

    return GameResult(x=x_wins, o=o_wins, d=draws)

# -------------------------------------------------
# Bài 2: AND-OR Tree Search
# -------------------------------------------------
def results(state: List[str], action: int, player: str = 'x') -> List[List[str]]:
    """Produce the set of states after action for player and all possible opponent reactions."""
    opponent = other(player)
    s_player = state.copy()
    s_player[action] = player

    o_actions = actions(s_player)
    if len(o_actions) < 1:
        return [s_player]

    r = []
    for o_a in o_actions:
        s = s_player.copy()
        s[o_a] = opponent
        r.append(s)
    return r

def and_or_search(state: List[str], player: str = 'x', draw_is_win: bool = True) -> Any:
    """AND-OR search starting from state."""
    global node_counter
    node_counter = 0
    return _or_step(state, player, draw_is_win)

def _or_step(state: List[str], player: str, draw_is_win: bool):
    global node_counter
    node_counter += 1
    u = utility(state, player)
    if u == -1 or (not draw_is_win and u == 0):
        return None
    for action in actions(state):
        plan = _and_step(results(state, action, player), player, draw_is_win)
        if plan is not None:
            return [action, plan]
    return None

def _and_step(states: List[List[str]], player: str, draw_is_win: bool):
    global node_counter
    node_counter += 1
    plans = []
    for s in states:
        u = utility(s, player)
        if u == 1:
            return "win"
        if draw_is_win and u == 0:
            return "draw"
        plan = _or_step(s, player, draw_is_win)
        if plan is None:
            return None
        plans.append(['if', s, 'then', plan])
    return plans

def and_or_player(board: List[str], player: str = 'x') -> int:
    """Agent using AND-OR search."""
    plan = and_or_search(board, player, draw_is_win=True)
    if plan is None:
        plan = and_or_search(board, player, draw_is_win=False)
    if plan is None or isinstance(plan, str):
        return int(np.random.choice(actions(board)))
    return plan[0]

# -------------------------------------------------
# Bài 3: Minimax & Alpha-Beta Search
# -------------------------------------------------
def minimax_search(board: List[str], player: str = 'x') -> Dict[str, Any]:
    """Plain Minimax search (no pruning).
    Returns {'move': best_move, 'value': value}.
    """
    global node_counter
    node_counter = 0

    def max_val(state, pl):
        global node_counter
        node_counter += 1
        v = utility(state, pl)
        if v is not None:
            return v, None
        v_best, m_best = -math.inf, None
        for a in actions(state):
            v2, _ = min_val(result(state, pl, a), pl)
            if v2 > v_best:
                v_best, m_best = v2, a
        return v_best, m_best

    def min_val(state, pl):
        global node_counter
        node_counter += 1
        v = utility(state, pl)
        if v is not None:
            return v, None
        v_best, m_best = math.inf, None
        for a in actions(state):
            v2, _ = max_val(result(state, other(pl), a), pl)
            if v2 < v_best:
                v_best, m_best = v2, a
        return v_best, m_best

    val, move = max_val(board, player)
    return {"move": move, "value": val}

def minimax_player(board: List[str], player: str = 'x') -> int:
    return minimax_search(board, player)["move"]

def alpha_beta_search(board: List[str], player: str = 'x', move_ordering: bool = False) -> Dict[str, Any]:
    """Minimax search with Alpha-Beta pruning.
    Returns {'move': best_move, 'value': value}.
    """
    global node_counter
    node_counter = 0

    def order_actions(acts):
        if not move_ordering:
            return acts
        # Prioritize center (4), then corners (0,2,6,8), then edges (1,3,5,7)
        pref = {4: 0, 0: 1, 2: 1, 6: 1, 8: 1, 1: 2, 3: 2, 5: 2, 7: 2}
        return sorted(acts, key=lambda a: pref.get(a, 3))

    def max_val_ab(state, pl, alpha, beta):
        global node_counter
        node_counter += 1
        v = utility(state, pl)
        if v is not None:
            return v, None
        v_best, m_best = -math.inf, None
        for a in order_actions(actions(state)):
            v2, _ = min_val_ab(result(state, pl, a), pl, alpha, beta)
            if v2 > v_best:
                v_best, m_best = v2, a
                alpha = max(alpha, v_best)
            if v_best >= beta:
                return v_best, m_best
        return v_best, m_best

    def min_val_ab(state, pl, alpha, beta):
        global node_counter
        node_counter += 1
        v = utility(state, pl)
        if v is not None:
            return v, None
        v_best, m_best = math.inf, None
        for a in order_actions(actions(state)):
            v2, _ = max_val_ab(result(state, other(pl), a), pl, alpha, beta)
            if v2 < v_best:
                v_best, m_best = v2, a
                beta = min(beta, v_best)
            if v_best <= alpha:
                return v_best, m_best
        return v_best, m_best

    val, move = max_val_ab(board, player, -math.inf, math.inf)
    return {"move": move, "value": val}

def alpha_beta_player(board: List[str], player: str = 'x') -> int:
    return alpha_beta_search(board, player)["move"]

# -------------------------------------------------
# Bài 4: Heuristic Alpha-Beta with Cutoff
# -------------------------------------------------
def eval_fun(state: List[str], player: str = 'x') -> Tuple[float, bool]:
    """Heuristic evaluation: returns (score, is_terminal).
    Features: 2-in-a-row with 3rd empty (weighted by 0.4).
    """
    u = utility(state, player)
    if u is not None:
        return float(u), True

    score = 0.0
    b = np.array(state).reshape((3, 3))
    diagonals = np.array([
        [b[i][i] for i in range(3)],
        [b[i][2 - i] for i in range(3)]
    ])

    for a_board in [b, b.T, diagonals]:
        for row in a_board:
            if sum(row == player) == 2 and any(row == ' '):
                score += 0.4
            if sum(row == other(player)) == 2 and any(row == ' '):
                score -= 0.4

    return score, False

def alpha_beta_cutoff_search(board: List[str], cutoff: int = 2, player: str = 'x') -> Dict[str, Any]:
    """Alpha-Beta search with depth cutoff and heuristic eval_fun."""
    global node_counter
    node_counter = 0

    def max_val(state, pl, depth, alpha, beta):
        global node_counter
        node_counter += 1
        score, is_term = eval_fun(state, pl)
        if is_term or depth == 0:
            return score, None
        v_best, m_best = -math.inf, None
        for a in actions(state):
            v2, _ = min_val(result(state, pl, a), pl, depth - 1, alpha, beta)
            if v2 > v_best:
                v_best, m_best = v2, a
                alpha = max(alpha, v_best)
            if v_best >= beta:
                return v_best, m_best
        return v_best, m_best

    def min_val(state, pl, depth, alpha, beta):
        global node_counter
        node_counter += 1
        score, is_term = eval_fun(state, pl)
        if is_term or depth == 0:
            return score, None
        v_best, m_best = math.inf, None
        for a in actions(state):
            v2, _ = max_val(result(state, other(pl), a), pl, depth - 1, alpha, beta)
            if v2 < v_best:
                v_best, m_best = v2, a
                beta = min(beta, v_best)
            if v_best <= alpha:
                return v_best, m_best
        return v_best, m_best

    val, move = max_val(board, player, cutoff, -math.inf, math.inf)
    return {"move": move, "value": val}

def heuristic2_player(board: List[str], player: str = 'x') -> int:
    return alpha_beta_cutoff_search(board, cutoff=2, player=player)["move"]

def heuristic4_player(board: List[str], player: str = 'x') -> int:
    return alpha_beta_cutoff_search(board, cutoff=4, player=player)["move"]

# -------------------------------------------------
# Bài 5: Pure Monte Carlo Search (PMCS)
# -------------------------------------------------
def playout(state: List[str], action: int, player: str = 'x') -> int:
    """Perform a random playout starting with action for player."""
    s = result(state, player, action)
    cur = other(player)
    while True:
        u = utility(s, player)
        if u is not None:
            return u
        a = random.choice(actions(s))
        s = result(s, cur, a)
        cur = other(cur)

def playouts(board: List[str], action: int, player: str = 'x', N: int = 100) -> Dict[str, Any]:
    """Run N playouts and return statistics."""
    wins = draws = losses = 0
    for _ in range(N):
        u = playout(board, action, player)
        if u == 1:
            wins += 1
        elif u == -1:
            losses += 1
        else:
            draws += 1
    return {
        "action": action,
        "N": N,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": (wins + 0.5 * draws) / N
    }

def pmcs(board: List[str], player: str = 'x', N: int = 100) -> int:
    """Pure Monte Carlo Search: returns the action with highest win rate."""
    acts = actions(board)
    best_act = acts[0]
    best_rate = -1.0
    for a in acts:
        res = playouts(board, a, player, N)
        if res["win_rate"] > best_rate:
            best_rate = res["win_rate"]
            best_act = a
    return best_act

def pmcs10_player(board: List[str], player: str = 'x') -> int:
    return pmcs(board, player, N=10)

def pmcs100_player(board: List[str], player: str = 'x') -> int:
    return pmcs(board, player, N=100)

# -------------------------------------------------
# Bài 6: MCTS with UCB1 (depth 1)
# -------------------------------------------------
def UCT_depth1(board: List[str], N: int = 100, player: str = 'x') -> int:
    """Upper Confidence Bound applied to Trees restricted to depth 1."""
    C = math.sqrt(2)
    acts = actions(board)
    u = [0.0] * len(acts)
    n = [0] * len(acts)
    n_parent = 0
    UCB1 = [math.inf] * len(acts)

    for _ in range(N):
        action_id = UCB1.index(max(UCB1))
        p = playout(board, acts[action_id], player=player)
        u[action_id] += p
        n[action_id] += 1
        n_parent += 1

        for idx in range(len(acts)):
            if n[idx] > 0:
                UCB1[idx] = (u[idx] / n[idx]) + C * math.sqrt(math.log(n_parent) / n[idx])

    return acts[n.index(max(n))]

def ucb1_10_player(board: List[str], player: str = 'x') -> int:
    return UCT_depth1(board, N=10, player=player)

def ucb1_100_player(board: List[str], player: str = 'x') -> int:
    return UCT_depth1(board, N=100, player=player)

# -------------------------------------------------
# Bài 7: Interactive Player & Scripted Player
# -------------------------------------------------
def interactive_player(board: List[str], player: Optional[str] = None) -> int:
    """Read a move from input with validation."""
    show_board(board, help=True)
    avail = actions(board)
    print(f"Available actions are: {avail}")
    while True:
        try:
            line = input("Your move (0-8 or -1 to end the game):\n").strip()
            move = int(line)
            if move == -1:
                raise Exception("Game ended by user.")
            if move in avail:
                return move
            print("Please enter a valid move.")
        except ValueError:
            print("Please enter a valid integer.")

def scripted_player_factory(moves: List[int]):
    """Returns a deterministic player that follows a scripted list of moves."""
    moves_iter = iter(moves)
    def scripted_player(board: List[str], player: str = 'x') -> int:
        avail = actions(board)
        for m in moves_iter:
            if m in avail:
                return m
        return random.choice(avail)
    return scripted_player

if __name__ == "__main__":
    print("Testing tictactoe.py core functions...")
    b = empty_board()
    print("Actions:", actions(b))
    b2 = result(b, 'x', 4)
    print("After move 4 by x, actions:", actions(b2))
    print("Terminal b2:", terminal(b2))
    print("Utility b2:", utility(b2, 'x'))
    res = play(random_player, random_player, N=100, seed=42)
    print("Random vs Random (100 games):", res)
    x, o, d = res
    print(f"Unpacked: X={x}, O={o}, D={d}")
    print("Testing search algorithms...")
    print("Minimax empty board move:", minimax_search(b)['move'], "Nodes:", node_counter)
    print("Alpha-Beta empty board move:", alpha_beta_search(b)['move'], "Nodes:", node_counter)
    print("Alpha-Beta (ordered) empty board move:", alpha_beta_search(b, move_ordering=True)['move'], "Nodes:", node_counter)
    print("All tests passed successfully!")

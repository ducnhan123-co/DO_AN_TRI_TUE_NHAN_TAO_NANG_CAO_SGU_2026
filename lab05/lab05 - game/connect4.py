# Connect 4 Core Implementation
"""Core functions and search algorithms for Connect 4.
Board representation: 2D numpy array of shape (rows, cols), default (6, 7).
Values:
  0: Empty
  1: Player 1 (Red)
 -1: Player 2 (Yellow)
Rows: 0 (top) to rows-1 (bottom). Pieces fall to the lowest available row.
"""

import math
import random
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import matplotlib.pyplot as plt

# Global counter for node expansions
node_counter = 0

def reset_counter():
    global node_counter
    node_counter = 0

def empty_board(shape: Tuple[int, int] = (6, 7)) -> np.ndarray:
    """Return a new empty board filled with zeros."""
    return np.zeros(shape, dtype=int)

def visualize(board: np.ndarray) -> None:
    """Visualize board using matplotlib (Randolph Rankin style)."""
    fig, ax = plt.subplots(figsize=(len(board[0]) * 0.8, len(board) * 0.8))
    rectangle = plt.Rectangle((-0.5, len(board) * -1 + 0.5), len(board[0]), len(board), fc='blue')
    ax.add_patch(rectangle)
    for i, row in enumerate(board):
        for j, val in enumerate(row):
            color = 'white' if val == 0 else ('red' if val == 1 else 'yellow')
            circle = plt.Circle((j, i * -1), 0.4, fc=color)
            ax.add_patch(circle)
    ax.set_xlim(-0.5, len(board[0]) - 0.5)
    ax.set_ylim(-len(board) + 0.5, 0.5)
    ax.axis('scaled')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def actions(state: np.ndarray) -> List[int]:
    """Return list of valid column indices where a piece can be dropped."""
    return [c for c in range(state.shape[1]) if state[0, c] == 0]

def drop_piece(state: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, int]:
    """Drop piece and return (new_board, landing_row)."""
    new_state = np.copy(state)
    for r in range(state.shape[0] - 1, -1, -1):
        if new_state[r, action] == 0:
            new_state[r, action] = player
            return new_state, r
    raise ValueError(f"Column {action} is full!")

def result(state: np.ndarray, player: int, action: int) -> np.ndarray:
    """Drop a piece of *player* into column *action*. Returns a new board."""
    new_state, _ = drop_piece(state, player, action)
    return new_state

def check_win_at(state: np.ndarray, r: int, c: int, player: int, k: int = 4) -> bool:
    """Fast incremental win check around the piece just placed at (r, c)."""
    R, C = state.shape
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        # Forward
        i = 1
        while 0 <= r + i * dr < R and 0 <= c + i * dc < C and state[r + i * dr, c + i * dc] == player:
            count += 1
            i += 1
        # Backward
        i = 1
        while 0 <= r - i * dr < R and 0 <= c - i * dc < C and state[r - i * dr, c - i * dc] == player:
            count += 1
            i += 1
        if count >= k:
            return True
    return False

def check_win(state: np.ndarray, player: int, k: int = 4) -> bool:
    """Full scan to check if *player* has k pieces in a row."""
    R, C = state.shape
    if R < k and C < k:
        return False
    # Horizontal
    for r in range(R):
        for c in range(C - k + 1):
            if np.all(state[r, c:c+k] == player):
                return True
    # Vertical
    for c in range(C):
        for r in range(R - k + 1):
            if np.all(state[r:r+k, c] == player):
                return True
    # Down-right
    for r in range(R - k + 1):
        for c in range(C - k + 1):
            if all(state[r+i, c+i] == player for i in range(k)):
                return True
    # Up-right
    for r in range(k - 1, R):
        for c in range(C - k + 1):
            if all(state[r-i, c+i] == player for i in range(k)):
                return True
    return False

def utility(state: np.ndarray, player: int = 1) -> Optional[int]:
    """Utility from perspective of *player* (+1 if win, -1 if loss, 0 if draw, None if ongoing)."""
    opp = -player
    if check_win(state, player):
        return 1
    if check_win(state, opp):
        return -1
    if len(actions(state)) == 0:
        return 0
    return None

def terminal(state: np.ndarray) -> bool:
    """Return True if game has finished (win or draw)."""
    return utility(state, 1) is not None

# -------------------------------------------------
# Agents & Play Framework
# -------------------------------------------------
def random_player(board: np.ndarray, player: int = 1) -> int:
    """Select a random legal column."""
    acts = actions(board)
    return int(random.choice(acts))

class Connect4Result(dict):
    """Dictionary holding match outcome counts {'1': p1_wins, '-1': p2_wins, 'draw': draws}."""
    def __init__(self, p1=0, p2=0, draws=0):
        super().__init__({'1': p1, '-1': p2, 'draw': draws})

    def __iter__(self):
        return iter((self['1'], self['-1'], self['draw']))

def play(player1, player2, N: int = 100, shape: Tuple[int, int] = (6, 7), seed: Optional[int] = None) -> Connect4Result:
    """Play N games between player1 (1 / Red) and player2 (-1 / Yellow)."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    p1_wins = p2_wins = draws = 0
    for _ in range(N):
        board = empty_board(shape)
        current = 1
        while not terminal(board):
            if current == 1:
                col = player1(board, 1)
            else:
                col = player2(board, -1)
            board = result(board, current, col)
            current = -current

        u = utility(board, 1)
        if u == 1:
            p1_wins += 1
        elif u == -1:
            p2_wins += 1
        else:
            draws += 1

    return Connect4Result(p1=p1_wins, p2=p2_wins, draws=draws)

# -------------------------------------------------
# Task 3: Minimax with Alpha-Beta Pruning
# -------------------------------------------------
def order_columns(acts: List[int], C: int) -> List[int]:
    """Order columns prioritizing center columns: for 7 cols -> [3, 2, 4, 1, 5, 0, 6]."""
    center = (C - 1) / 2.0
    return sorted(acts, key=lambda c: abs(c - center))

def minimax_alpha_beta(board: np.ndarray, player: int = 1, depth: Optional[int] = None, move_ordering: bool = True) -> Dict[str, Any]:
    """Minimax search with Alpha-Beta pruning."""
    global node_counter
    node_counter = 0
    C = board.shape[1]

    def max_value(state, pl, alpha, beta, d, last_r, last_c):
        global node_counter
        node_counter += 1
        if last_r is not None and check_win_at(state, last_r, last_c, -pl):
            return -1, None
        acts = actions(state)
        if not acts:
            return 0, None
        if depth is not None and d >= depth:
            return 0, None

        v_best = -math.inf
        m_best = acts[0]
        if move_ordering:
            acts = order_columns(acts, C)

        for a in acts:
            nxt_state, r_drop = drop_piece(state, pl, a)
            v2, _ = min_value(nxt_state, pl, alpha, beta, d + 1, r_drop, a)
            if v2 > v_best:
                v_best = v2
                m_best = a
                alpha = max(alpha, v_best)
            if v_best >= beta:
                return v_best, m_best
        return v_best, m_best

    def min_value(state, pl, alpha, beta, d, last_r, last_c):
        global node_counter
        node_counter += 1
        if last_r is not None and check_win_at(state, last_r, last_c, pl):
            return 1, None
        acts = actions(state)
        if not acts:
            return 0, None
        if depth is not None and d >= depth:
            return 0, None

        v_best = math.inf
        m_best = acts[0]
        if move_ordering:
            acts = order_columns(acts, C)

        for a in acts:
            nxt_state, r_drop = drop_piece(state, -pl, a)
            v2, _ = max_value(nxt_state, pl, alpha, beta, d + 1, r_drop, a)
            if v2 < v_best:
                v_best = v2
                m_best = a
                beta = min(beta, v_best)
            if v_best <= alpha:
                return v_best, m_best
        return v_best, m_best

    val, col = max_value(board, player, -math.inf, math.inf, 0, None, None)
    return {"col": col, "value": val, "nodes": node_counter}

def alpha_beta_player(board: np.ndarray, player: int = 1) -> int:
    return minimax_alpha_beta(board, player, move_ordering=True)["col"]

# -------------------------------------------------
# Task 4: Heuristic Alpha-Beta with Cutoff
# -------------------------------------------------
def evaluate_window(window: List[int], player: int) -> int:
    """Evaluate a 4-cell window for the given player."""
    opp = -player
    count_player = window.count(player)
    count_opp = window.count(opp)
    count_empty = window.count(0)

    score = 0
    if count_player == 4:
        score += 1000
    elif count_player == 3 and count_empty == 1:
        score += 50
    elif count_player == 2 and count_empty == 2:
        score += 10

    if count_opp == 4:
        score -= 1000
    elif count_opp == 3 and count_empty == 1:
        score -= 50
    elif count_opp == 2 and count_empty == 2:
        score -= 10

    return score

def eval_fun(state: np.ndarray, player: int = 1) -> float:
    """Heuristic evaluation function for Connect 4."""
    u = utility(state, player)
    if u is not None:
        return float(u * 10000)

    R, C = state.shape
    total_score = 0

    # Center column bonus
    center_col = C // 2
    center_pieces = list(state[:, center_col]).count(player)
    center_opp = list(state[:, center_col]).count(-player)
    total_score += (center_pieces - center_opp) * 6

    # Horizontal windows
    for r in range(R):
        row_list = list(state[r, :])
        for c in range(C - 3):
            window = row_list[c:c+4]
            total_score += evaluate_window(window, player)

    # Vertical windows
    for c in range(C):
        col_list = list(state[:, c])
        for r in range(R - 3):
            window = col_list[r:r+4]
            total_score += evaluate_window(window, player)

    # Positive diagonals
    for r in range(R - 3):
        for c in range(C - 3):
            window = [state[r+i, c+i] for i in range(4)]
            total_score += evaluate_window(window, player)

    # Negative diagonals
    for r in range(3, R):
        for c in range(C - 3):
            window = [state[r-i, c+i] for i in range(4)]
            total_score += evaluate_window(window, player)

    return float(np.tanh(total_score / 200.0))

def heuristic_alpha_beta(board: np.ndarray, player: int = 1, cutoff: int = 4) -> Dict[str, Any]:
    """Alpha-Beta search with depth cutoff and heuristic eval_fun."""
    global node_counter
    node_counter = 0
    C = board.shape[1]

    def max_val(state, pl, alpha, beta, depth, last_r, last_c):
        global node_counter
        node_counter += 1
        if last_r is not None and check_win_at(state, last_r, last_c, -pl):
            return -10000.0, None
        acts = actions(state)
        if not acts:
            return 0.0, None
        if depth >= cutoff:
            return eval_fun(state, pl), None

        v_best = -math.inf
        m_best = acts[0]
        acts = order_columns(acts, C)

        for a in acts:
            nxt_state, r_drop = drop_piece(state, pl, a)
            v2, _ = min_val(nxt_state, pl, alpha, beta, depth + 1, r_drop, a)
            if v2 > v_best:
                v_best = v2
                m_best = a
                alpha = max(alpha, v_best)
            if v_best >= beta:
                return v_best, m_best
        return v_best, m_best

    def min_val(state, pl, alpha, beta, depth, last_r, last_c):
        global node_counter
        node_counter += 1
        if last_r is not None and check_win_at(state, last_r, last_c, pl):
            return 10000.0, None
        acts = actions(state)
        if not acts:
            return 0.0, None
        if depth >= cutoff:
            return eval_fun(state, pl), None

        v_best = math.inf
        m_best = acts[0]
        acts = order_columns(acts, C)

        for a in acts:
            nxt_state, r_drop = drop_piece(state, -pl, a)
            v2, _ = max_val(nxt_state, pl, alpha, beta, depth + 1, r_drop, a)
            if v2 < v_best:
                v_best = v2
                m_best = a
                beta = min(beta, v_best)
            if v_best <= alpha:
                return v_best, m_best
        return v_best, m_best

    val, col = max_val(board, player, -math.inf, math.inf, 0, None, None)
    return {"col": col, "value": val, "nodes": node_counter}

def heuristic_player_factory(cutoff: int = 4):
    def player(board: np.ndarray, pl: int = 1) -> int:
        return heuristic_alpha_beta(board, pl, cutoff=cutoff)["col"]
    return player

# -------------------------------------------------
# Task 5: Pure Monte Carlo Search (PMCS)
# -------------------------------------------------
def playout(state: np.ndarray, action: int, player: int = 1) -> int:
    """Fast random rollout starting with action for player."""
    s, r_drop = drop_piece(state, player, action)
    if check_win_at(s, r_drop, action, player):
        return 1

    cur = -player
    while True:
        acts = actions(s)
        if not acts:
            return 0
        a = random.choice(acts)
        s, r_drop = drop_piece(s, cur, a)
        if check_win_at(s, r_drop, a, cur):
            return 1 if cur == player else -1
        cur = -cur

def playouts(board: np.ndarray, action: int, player: int = 1, N: int = 50) -> Dict[str, Any]:
    """Execute N playouts from action."""
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
        "col": action,
        "N": N,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": (wins + 0.5 * draws) / N
    }

def pmcs(board: np.ndarray, player: int = 1, N: int = 50) -> int:
    """Pure Monte Carlo Search for Connect 4."""
    acts = actions(board)
    best_act = acts[0]
    best_score = -1.0
    for a in acts:
        res = playouts(board, a, player, N)
        if res["win_rate"] > best_score:
            best_score = res["win_rate"]
            best_act = a
    return best_act

def pmcs_player_factory(N: int = 50):
    def player(board: np.ndarray, pl: int = 1) -> int:
        return pmcs(board, pl, N=N)
    return player

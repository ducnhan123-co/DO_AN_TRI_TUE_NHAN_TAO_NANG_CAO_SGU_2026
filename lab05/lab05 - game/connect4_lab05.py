"""Connect 4 Lab05: reproducible baseline and adversarial/Monte Carlo agents.

Board: numpy R x C with 0=empty, +1/-1=players. Bottom row is R-1.
This is a reference solution for the Connect 4 extension, NOT Mean Connect 4.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import math
import numpy as np

WIN_SCORE = 1_000_000


def empty_board(shape=(6, 7)):
    if len(shape) != 2 or min(shape) < 4:
        raise ValueError('Connect 4 needs at least four rows and four columns')
    return np.zeros(shape, dtype=np.int8)


def actions(board):
    return [c for c in range(board.shape[1]) if board[0, c] == 0]


def result(board, player, action):
    if player not in (-1, 1) or not isinstance(action, (int, np.integer)):
        raise ValueError('Invalid player or action')
    if terminal(board):
        raise ValueError('Game already ended')
    if action not in actions(board):
        raise ValueError('Illegal or full column')
    new = board.copy()
    row = int(np.where(new[:, action] == 0)[0][-1])
    new[row, action] = player
    return new


def winners(board):
    R, C = board.shape
    found = set()
    for r in range(R):
        for c in range(C):
            val = int(board[r, c])
            if val == 0:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_r, end_c = r + 3*dr, c + 3*dc
                if 0 <= end_r < R and 0 <= end_c < C:
                    if all(board[r + k*dr, c + k*dc] == val for k in range(1, 4)):
                        found.add(val)
    return found


def utility(board, player=1):
    found = winners(board)
    if len(found) > 1:
        raise ValueError('Invalid board: both players have four-in-a-row')
    if found:
        return 1 if player in found else -1
    if not actions(board):
        return 0
    return None


def terminal(board):
    return bool(winners(board)) or len(actions(board)) == 0


def all_windows(board):
    R, C = board.shape
    for r in range(R):
        for c in range(C-3):
            yield board[r, c:c+4]
    for r in range(R-3):
        for c in range(C):
            yield board[r:r+4, c]
    for r in range(R-3):
        for c in range(C-3):
            yield np.array([board[r+k, c+k] for k in range(4)])
            yield np.array([board[r+3-k, c+k] for k in range(4)])


def eval_fun(board, player=1):
    u = utility(board, player)
    if u is not None:
        return u * WIN_SCORE
    score = 0
    for w in all_windows(board):
        mine, enemy, blanks = int(np.count_nonzero(w == player)), int(np.count_nonzero(w == -player)), int(np.count_nonzero(w == 0))
        if enemy == 0:
            score += {0: 0, 1: 1, 2: 10, 3: 60, 4: WIN_SCORE}[mine]
        if mine == 0:
            score -= {0: 0, 1: 1, 2: 10, 3: 80, 4: WIN_SCORE}[enemy]
    # Central preference is only a heuristic, not a proof of optimality.
    center = (board.shape[1]-1)/2
    for c in range(board.shape[1]):
        score += int(np.count_nonzero(board[:, c] == player)) * (board.shape[1] - abs(c-center))
    return float(score)


def ordered_actions(board, player):
    moves = actions(board)
    winning = []
    blocks = []
    for c in moves:
        if utility(result(board, player, c), player) == 1:
            winning.append(c)
        if utility(result(board, -player, c), -player) == 1:
            blocks.append(c)
    center = (board.shape[1]-1)/2
    return sorted(moves, key=lambda c: (c not in winning, c not in blocks, abs(c-center), c))


@dataclass
class SearchResult:
    move: int | None
    value: float
    nodes: int
    elapsed_ms: float


def alpha_beta_search(board, player=1, cutoff=4, move_ordering=True):
    if player not in (-1, 1):
        raise ValueError('player must be +1/-1')
    if cutoff is not None and cutoff < 1:
        raise ValueError('cutoff must be None or >= 1')
    t0 = perf_counter()
    nodes = 0

    def search(state, to_move, depth, alpha, beta):
        nonlocal nodes
        nodes += 1
        u = utility(state, player)
        if u is not None:
            return float(u * WIN_SCORE), None
        if cutoff is not None and depth >= cutoff:
            return eval_fun(state, player), None
        options = ordered_actions(state, to_move) if move_ordering else actions(state)
        best_move = None
        if to_move == player:
            best = -math.inf
            for c in options:
                val, _ = search(result(state, to_move, c), -to_move, depth+1, alpha, beta)
                if val > best:
                    best, best_move = val, c
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        else:
            best = math.inf
            for c in options:
                val, _ = search(result(state, to_move, c), -to_move, depth+1, alpha, beta)
                if val < best:
                    best, best_move = val, c
                beta = min(beta, best)
                if alpha >= beta:
                    break
        return best, best_move

    value, move = search(board, player, 0, -math.inf, math.inf)
    return SearchResult(move, value, nodes, (perf_counter()-t0)*1000)


def random_player(board, player=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return int(rng.choice(actions(board)))


def playout(board, first_move, player=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    b = result(board, player, first_move)
    turn = -player
    while not terminal(b):
        b = result(b, turn, random_player(b, turn, rng))
        turn = -turn
    return utility(b, player)


def pmcs(board, player=1, N=100, rng=None):
    """N total playouts, nearly equal budgets; differs from per-action N."""
    rng = np.random.default_rng() if rng is None else rng
    moves = actions(board)
    if not moves:
        return None
    if N < len(moves):
        raise ValueError('N must be >= number of available moves')
    wins = {c: 0.0 for c in moves}
    counts = {c: 0 for c in moves}
    for i in range(N):
        c = moves[i % len(moves)]
        wins[c] += playout(board, c, player, rng)
        counts[c] += 1
    return max(moves, key=lambda c: (wins[c]/counts[c], -abs(c-(board.shape[1]-1)/2)))


def play(agent_x, agent_o, games=100, seed=42, shape=(6, 7)):
    rng = np.random.default_rng(seed)
    results = {'x': 0, 'o': 0, 'd': 0}
    move_durations = []
    for _ in range(games):
        board = empty_board(shape)
        current = 1
        while not terminal(board):
            agent = agent_x if current == 1 else agent_o
            t0 = perf_counter()
            action = agent(board.copy(), current, rng)
            dt = (perf_counter()-t0)*1000
            if current == 1:
                move_durations.append(dt)
            board = result(board, current, action)
            current *= -1
        winner = utility(board, 1)
        results['x' if winner == 1 else 'o' if winner == -1 else 'd'] += 1
    return results, (sum(move_durations)/len(move_durations) if move_durations else None)


def minimax_search(board, player=1, cutoff=None):
    """Unpruned minimax for small late-game boards only (same leaf scoring)."""
    t0 = perf_counter()
    nodes = 0
    def visit(state, turn, depth):
        nonlocal nodes
        nodes += 1
        u = utility(state, player)
        if u is not None:
            return float(u * WIN_SCORE), None
        if cutoff is not None and depth >= cutoff:
            return eval_fun(state, player), None
        choices = [(visit(result(state, turn, a), -turn, depth+1)[0], a)
                   for a in actions(state)]
        return max(choices, key=lambda z: z[0]) if turn == player else min(choices, key=lambda z: z[0])
    value, move = visit(board, player, 0)
    return SearchResult(move, value, nodes, (perf_counter()-t0)*1000)


class Connect4Agent:
    """Callable agent with its own RNG state, compatible with play(agent_x,agent_o)."""
    def __init__(self, method='random', *, cutoff=4, N=100, seed=42, ordering=True):
        self.method, self.cutoff, self.N, self.ordering = method, cutoff, N, ordering
        self.rng = np.random.default_rng(seed)
    def __call__(self, board, player, rng=None):
        if self.method == 'random':
            return random_player(board, player, self.rng)
        if self.method == 'alpha_beta':
            return alpha_beta_search(board, player, self.cutoff, self.ordering).move
        if self.method == 'pmcs':
            return pmcs(board, player, self.N, self.rng)
        if self.method == 'minimax':
            return minimax_search(board, player, self.cutoff).move
        raise ValueError('Unknown method')

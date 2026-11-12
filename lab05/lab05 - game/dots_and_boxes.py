import random
import time
from typing import List, Tuple, Dict, Any, Optional

def empty_board(size=(3, 3)) -> dict:
    """Size is (rows, cols) of dots. Boxes count is (rows-1) * (cols-1)."""
    return {
        'size': size,
        'lines': dict(),
        'boxes': dict()
    }

def all_lines(size=(3, 3)) -> list:
    R, C = size
    lines = []
    # Horizontal lines: R rows, C-1 segments each
    for r in range(R):
        for c in range(C - 1):
            lines.append(('h', r, c))
    # Vertical lines: R-1 segments, C columns each
    for r in range(R - 1):
        for c in range(C):
            lines.append(('v', r, c))
    return lines

def actions(board: dict) -> list:
    R, C = board['size']
    possible = []
    # Horizontal
    for r in range(R):
        for c in range(C - 1):
            line = ('h', r, c)
            if line not in board['lines']:
                possible.append(line)
    # Vertical
    for r in range(R - 1):
        for c in range(C):
            line = ('v', r, c)
            if line not in board['lines']:
                possible.append(line)
    return possible

def adjacent_boxes(line: Tuple[str, int, int], size: Tuple[int, int]) -> list:
    """Return boxes (r, c) adjacent to the line."""
    orient, r, c = line
    R, C = size
    adj = []
    if orient == 'h':
        # Box below: (r, c) if r < R - 1 and c < C - 1
        if r < R - 1 and c < C - 1:
            adj.append((r, c))
        # Box above: (r - 1, c) if r > 0 and c < C - 1
        if r > 0 and c < C - 1:
            adj.append((r - 1, c))
    elif orient == 'v':
        # Box to the right: (r, c) if r < R - 1 and c < C - 1
        if r < R - 1 and c < C - 1:
            adj.append((r, c))
        # Box to the left: (r, c - 1) if r < R - 1 and c > 0
        if r < R - 1 and c > 0:
            adj.append((r, c - 1))
    return adj

def is_box_completed(box: Tuple[int, int], lines: dict) -> bool:
    r, c = box
    return (('h', r, c) in lines and
            ('h', r + 1, c) in lines and
            ('v', r, c) in lines and
            ('v', r, c + 1) in lines)

def box_sides_count(box: Tuple[int, int], lines: dict) -> int:
    r, c = box
    cnt = 0
    if ('h', r, c) in lines: cnt += 1
    if ('h', r + 1, c) in lines: cnt += 1
    if ('v', r, c) in lines: cnt += 1
    if ('v', r, c + 1) in lines: cnt += 1
    return cnt

def result(board: dict, player: int, action: Tuple[str, int, int]) -> Tuple[dict, bool]:
    """
    Apply action for player.
    Returns (new_board, got_extra_turn).
    """
    new_board = {
        'size': board['size'],
        'lines': board['lines'].copy(),
        'boxes': board['boxes'].copy()
    }
    new_board['lines'][action] = True
    
    extra_turn = False
    for box in adjacent_boxes(action, board['size']):
        if box not in new_board['boxes'] and is_box_completed(box, new_board['lines']):
            new_board['boxes'][box] = player
            extra_turn = True
            
    return new_board, extra_turn

def terminal(board: dict) -> bool:
    R, C = board['size']
    total_boxes = (R - 1) * (C - 1)
    return len(board['boxes']) == total_boxes

def utility(board: dict, player: int = 1) -> Optional[int]:
    """
    Returns net score or +1 / -1 / 0 if terminal.
    If not terminal, returns None.
    """
    if not terminal(board):
        return None
    p_score = sum(1 for p in board['boxes'].values() if p == player)
    opp_score = sum(1 for p in board['boxes'].values() if p == -player)
    if p_score > opp_score:
        return 1
    elif p_score < opp_score:
        return -1
    else:
        return 0

def random_player(board: dict, player: int = 1) -> Tuple[str, int, int]:
    acts = actions(board)
    if not acts:
        return ('h', 0, 0)
    return random.choice(acts)

def play(player_1, player_2, N: int = 100, size: Tuple[int, int] = (3, 3), seed: Optional[int] = None) -> dict:
    if seed is not None:
        random.seed(seed)
        
    wins = {'1': 0, '-1': 0, 'draw': 0}
    for _ in range(N):
        b = empty_board(size)
        curr = 1
        while not terminal(b):
            p_func = player_1 if curr == 1 else player_2
            act = p_func(b, player=curr)
            b, extra = result(b, curr, act)
            if not extra:
                curr = -curr
        u = utility(b, player=1)
        if u == 1:
            wins['1'] += 1
        elif u == -1:
            wins['-1'] += 1
        else:
            wins['draw'] += 1
    return wins

# --- Minimax with Alpha-Beta Pruning ---
node_counter = 0

def minimax_alpha_beta(board: dict, player: int = 1, depth: Optional[int] = None,
                       alpha: float = -float('inf'), beta: float = float('inf')) -> Tuple[float, Optional[Tuple]]:
    global node_counter
    node_counter += 1
    
    if terminal(board):
        p1_boxes = sum(1 for p in board['boxes'].values() if p == 1)
        pm1_boxes = sum(1 for p in board['boxes'].values() if p == -1)
        return float(p1_boxes - pm1_boxes), None
        
    if depth is not None and depth <= 0:
        p1_boxes = sum(1 for p in board['boxes'].values() if p == 1)
        pm1_boxes = sum(1 for p in board['boxes'].values() if p == -1)
        return float(p1_boxes - pm1_boxes), None
        
    legal_acts = actions(board)
    if not legal_acts:
        return 0.0, None
        
    best_act = legal_acts[0]
    
    if player == 1:
        v = -float('inf')
        for act in legal_acts:
            nxt, extra = result(board, 1, act)
            # If extra turn, player remains 1!
            nxt_player = 1 if extra else -1
            val, _ = minimax_alpha_beta(nxt, nxt_player, depth - 1 if depth is not None else None, alpha, beta)
            if val > v:
                v = val
                best_act = act
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        return v, best_act
    else:
        v = float('inf')
        for act in legal_acts:
            nxt, extra = result(board, -1, act)
            # If extra turn, player remains -1!
            nxt_player = -1 if extra else 1
            val, _ = minimax_alpha_beta(nxt, nxt_player, depth - 1 if depth is not None else None, alpha, beta)
            if val < v:
                v = val
                best_act = act
            beta = min(beta, v)
            if beta <= alpha:
                break
        return v, best_act

def minimax_player_factory(depth: Optional[int] = None):
    def player_func(board, player=1):
        global node_counter
        node_counter = 0
        _, act = minimax_alpha_beta(board, player=player, depth=depth)
        return act if act is not None else actions(board)[0]
    return player_func

# Heuristic Alpha-Beta
def eval_fun(board: dict, player: int = 1) -> float:
    """
    Score = (player boxes - opponent boxes) * 10
    - Penalize creating a 3rd side on any box (which gives opponent a free box)!
    + Reward taking a 4th side (capturing a box).
    """
    p1 = sum(1 for p in board['boxes'].values() if p == 1)
    pm1 = sum(1 for p in board['boxes'].values() if p == -1)
    net_boxes = (p1 - pm1) if player == 1 else (pm1 - p1)
    
    R, C = board['size']
    danger_sides = 0
    for r in range(R - 1):
        for c in range(C - 1):
            if (r, c) not in board['boxes']:
                sides = box_sides_count((r, c), board['lines'])
                if sides == 3:
                    danger_sides += 1
                    
    # Leaving 3-sided boxes gives opponent free boxes
    return float(net_boxes * 10.0 - danger_sides * 4.0)

def heuristic_player_factory(cutoff: int = 4):
    def player_func(board, player=1):
        _, act = minimax_alpha_beta(board, player=player, depth=cutoff)
        return act if act is not None else actions(board)[0]
    return player_func


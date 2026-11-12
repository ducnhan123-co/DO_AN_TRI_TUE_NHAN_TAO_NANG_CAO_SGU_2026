import numpy as np
import random
import time
from typing import List, Tuple, Union, Optional

def empty_board(shape=(6, 7)) -> np.ndarray:
    return np.zeros(shape, dtype=int)

def actions(state: np.ndarray, player: int = 1) -> list:
    """
    Returns list of legal actions for *player*:
    - Standard drop: ('drop', c) for column c where state[0, c] == 0.
    - Mean action: ('mean', c_rem, c_drop) where state[R-1, c_rem] == -player
      and column c_drop has space (or c_drop == c_rem after removal).
    """
    R, C = state.shape
    legal = []
    
    # 1. Standard drops
    for c in range(C):
        if state[0, c] == 0:
            legal.append(('drop', c))
            
    # 2. Mean actions: can remove opponent's piece from bottom row
    opp = -player
    bottom_row = R - 1
    removable = [c for c in range(C) if state[bottom_row, c] == opp]
    
    for c_rem in removable:
        for c_drop in range(C):
            # c_drop has space if top is 0, OR if c_drop == c_rem (since we just freed top space)
            if state[0, c_drop] == 0 or c_drop == c_rem:
                legal.append(('mean', c_rem, c_drop))
                
    return legal

def result(state: np.ndarray, player: int, action: Union[Tuple, int]) -> np.ndarray:
    """Apply action for player. Returns new state."""
    R, C = state.shape
    new_state = state.copy()
    
    if isinstance(action, (int, np.integer)):
        action = ('drop', int(action))
        
    act_type = action[0]
    if act_type == 'drop':
        c = action[1]
        for r in range(R - 1, -1, -1):
            if new_state[r, c] == 0:
                new_state[r, c] = player
                break
    elif act_type == 'mean':
        c_rem, c_drop = action[1], action[2]
        opp = -player
        # Remove bottom disc in c_rem, shift down
        new_state[1:R, c_rem] = new_state[0:R-1, c_rem]
        new_state[0, c_rem] = 0
        
        # Drop opponent disc into c_drop
        for r in range(R - 1, -1, -1):
            if new_state[r, c_drop] == 0:
                new_state[r, c_drop] = opp
                break
    return new_state

def check_win(state: np.ndarray, player: int, win_length: int = 4) -> bool:
    """Check if player has formed win_length in a row."""
    R, C = state.shape
    k = win_length
    
    # Horizontal
    for r in range(R):
        for c in range(C - k + 1):
            if all(state[r, c + i] == player for i in range(k)):
                return True
                
    # Vertical
    for r in range(R - k + 1):
        for c in range(C):
            if all(state[r + i, c] == player for i in range(k)):
                return True
                
    # Positive diagonal (\)
    for r in range(R - k + 1):
        for c in range(C - k + 1):
            if all(state[r + i, c + i] == player for i in range(k)):
                return True
                
    # Negative diagonal (/)
    for r in range(k - 1, R):
        for c in range(C - k + 1):
            if all(state[r - i, c + i] == player for i in range(k)):
                return True
                
    return False

def terminal(state: np.ndarray, win_length: int = 4) -> bool:
    if check_win(state, 1, win_length) or check_win(state, -1, win_length):
        return True
    return bool(np.all(state[0, :] != 0))

def utility(state: np.ndarray, player: int = 1, win_length: int = 4) -> Optional[int]:
    """Returns +1 if player wins, -1 if opponent wins, 0 if draw, None if ongoing."""
    win_p = check_win(state, player, win_length)
    win_opp = check_win(state, -player, win_length)
    
    if win_p and not win_opp:
        return 1
    if win_opp and not win_p:
        return -1
    if win_p and win_opp:
        return -1  # Suicide rule: triggering opponent win is a loss
    if np.all(state[0, :] != 0):
        return 0
    return None

def random_player(board: np.ndarray, player: int = 1) -> Union[Tuple, int]:
    acts = actions(board, player)
    if not acts:
        return ('drop', 0)
    return random.choice(acts)

def play(player_1, player_2, N: int = 10, shape=(6, 7), win_length: int = 4, seed: Optional[int] = None) -> dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    wins = {'1': 0, '-1': 0, 'draw': 0}
    for _ in range(N):
        b = empty_board(shape)
        curr = 1
        while True:
            u = utility(b, player=1, win_length=win_length)
            if u is not None:
                if u == 1:
                    wins['1'] += 1
                elif u == -1:
                    wins['-1'] += 1
                else:
                    wins['draw'] += 1
                break
                
            p_func = player_1 if curr == 1 else player_2
            act = p_func(b, player=curr)
            b = result(b, curr, act)
            curr = -curr
    return wins

# --- Evaluation Function & Search ---
def eval_fun(state: np.ndarray, player: int = 1, win_length: int = 4) -> float:
    u = utility(state, player, win_length)
    if u is not None:
        return u * 10000.0
        
    R, C = state.shape
    k = win_length
    score = 0.0
    opp = -player
    
    # Center column preference
    mid = C // 2
    score += np.sum(state[:, mid] == player) * 4.0
    score -= np.sum(state[:, mid] == opp) * 4.0
    
    # Window evaluations
    def score_window(window):
        p_cnt = np.sum(window == player)
        o_cnt = np.sum(window == opp)
        if p_cnt > 0 and o_cnt > 0:
            return 0.0
        if p_cnt == 3: return 50.0
        if p_cnt == 2: return 10.0
        if o_cnt == 3: return -60.0
        if o_cnt == 2: return -12.0
        return 0.0

    # Horizontal
    for r in range(R):
        for c in range(C - k + 1):
            score += score_window(state[r, c:c+k])
    # Vertical
    for r in range(R - k + 1):
        for c in range(C):
            score += score_window(state[r:r+k, c])
    # Diagonal \
    for r in range(R - k + 1):
        for c in range(C - k + 1):
            score += score_window(np.array([state[r+i, c+i] for i in range(k)]))
    # Diagonal /
    for r in range(k - 1, R):
        for c in range(C - k + 1):
            score += score_window(np.array([state[r-i, c+i] for i in range(k)]))
            
    return score

def order_actions(state: np.ndarray, acts: list, player: int, win_length: int = 4) -> list:
    """Prioritize immediate winning moves, then drops near center, then mean actions."""
    R, C = state.shape
    mid = C // 2
    
    def key_func(act):
        # High priority to drops
        if act[0] == 'drop':
            c = act[1]
            dist = abs(c - mid)
            return (0, dist)
        else:
            # Mean action
            return (1, 0)
            
    return sorted(acts, key=key_func)

node_counter = 0

def minimax_alpha_beta(state: np.ndarray, player: int = 1, depth: Optional[int] = None, 
                       alpha: float = -float('inf'), beta: float = float('inf'), 
                       win_length: int = 4, ordering: bool = True) -> Tuple[float, Optional[Tuple]]:
    global node_counter
    node_counter += 1
    
    u = utility(state, player=1, win_length=win_length)
    if u is not None:
        return float(u), None
        
    if depth is not None and depth <= 0:
        return eval_fun(state, player=1, win_length=win_length) / 10000.0, None
        
    legal_acts = actions(state, player)
    if not legal_acts:
        return 0.0, None
        
    if ordering:
        legal_acts = order_actions(state, legal_acts, player, win_length)
        
    best_act = legal_acts[0]
    
    if player == 1:
        v = -float('inf')
        for act in legal_acts:
            nxt = result(state, 1, act)
            val, _ = minimax_alpha_beta(nxt, -1, depth - 1 if depth is not None else None, 
                                        alpha, beta, win_length, ordering)
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
            nxt = result(state, -1, act)
            val, _ = minimax_alpha_beta(nxt, 1, depth - 1 if depth is not None else None, 
                                        alpha, beta, win_length, ordering)
            if val < v:
                v = val
                best_act = act
            beta = min(beta, v)
            if beta <= alpha:
                break
        return v, best_act

def minimax_player_factory(depth: Optional[int] = None, win_length: int = 4, ordering: bool = True):
    def player_func(board, player=1):
        global node_counter
        node_counter = 0
        _, act = minimax_alpha_beta(board, player=player, depth=depth, win_length=win_length, ordering=ordering)
        return act if act is not None else ('drop', 0)
    return player_func

# Pure Monte Carlo Search for Mean Connect 4
def playout(state: np.ndarray, player: int, win_length: int = 4) -> int:
    curr_b = state.copy()
    curr_p = player
    while True:
        u = utility(curr_b, player=1, win_length=win_length)
        if u is not None:
            return u
        acts = actions(curr_b, curr_p)
        if not acts:
            return 0
        act = random.choice(acts)
        curr_b = result(curr_b, curr_p, act)
        curr_p = -curr_p

def pmcs(state: np.ndarray, player: int = 1, N: int = 20, win_length: int = 4) -> Tuple:
    acts = actions(state, player)
    if not acts:
        return ('drop', 0)
    if len(acts) == 1:
        return acts[0]
        
    best_act = acts[0]
    best_score = -float('inf') if player == 1 else float('inf')
    
    for act in acts:
        nxt = result(state, player, act)
        # Check immediate win
        u = utility(nxt, player=player, win_length=win_length)
        if u == 1:
            return act
        score = sum(playout(nxt, -player, win_length) for _ in range(N)) / N
        if player == 1 and score > best_score:
            best_score = score
            best_act = act
        elif player == -1 and score < best_score:
            best_score = score
            best_act = act
    return best_act


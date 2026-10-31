import math
import random
import numpy as np


class MCTSNode:
    """
    Một nút trong cây Monte Carlo Tree Search (MCTS).
    """
    def __init__(self, state, player_turn, parent=None, action=None):
        self.state = state.copy()          # Trạng thái bàn cờ 8x9
        self.player_turn = player_turn    # 1 hoặc 2 (quân đi lượt này)
        self.parent = parent
        self.action = action              # Cột đã thả để đến trạng thái này
        self.children = {}                # action -> MCTSNode
        self.visits = 0
        self.total_reward = 0.0           # Điểm số tích lũy
        self.untried_actions = [c for c in range(state.shape[1]) if state[0, c] == 0]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return (
            MCTSAgent._check_winner(self.state, 1)
            or MCTSAgent._check_winner(self.state, 2)
            or np.all(self.state[0, :] != 0)
        )

    def best_child(self, c_param=1.414):
        best_score = -float("inf")
        best_nodes = []
        for child in self.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.total_reward / child.visits
                exploration = c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif math.isclose(score, best_score, rel_tol=1e-5):
                best_nodes.append(child)

        return random.choice(best_nodes)


class MCTSAgent:
    """
    Cấp 4: Monte Carlo Tree Search (MCTS) cho Connect 4 (Bàn cờ 8x9)
    Thuật toán áp dụng 4 pha kinh điển:
    1. Selection: Dùng UCB1 đi sâu xuống cây.
    2. Expansion: Mở rộng nhánh con mới.
    3. Simulation (Rollout): Mô phỏng ngẫu nhiên siêu tốc với kiểm tra cục bộ O(1).
    4. Backpropagation: Cập nhật ngược số lần thăm và điểm thắng/thua lên gốc.
    """
    def __init__(self, simulations=100, c_param=1.414, max_rollout_depth=35):
        self.simulations = simulations
        self.c_param = c_param
        self.max_rollout_depth = max_rollout_depth

    @staticmethod
    def _determine_turn(grid):
        count_1 = np.sum(grid == 1)
        count_2 = np.sum(grid == 2)
        return 1 if count_1 == count_2 else 2

    @staticmethod
    def _drop_piece(board, col, piece):
        for r in range(board.shape[0] - 1, -1, -1):
            if board[r, col] == 0:
                board[r, col] = piece
                return r
        return None

    @staticmethod
    def _check_local_win(board, row, col, piece):
        rows, cols = board.shape
        # Horizontal
        c_min = max(0, col - 3)
        c_max = min(cols - 1, col + 3)
        count = 0
        for c in range(c_min, c_max + 1):
            if board[row, c] == piece:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0

        # Vertical
        r_min = max(0, row - 3)
        r_max = min(rows - 1, row + 3)
        count = 0
        for r in range(r_min, r_max + 1):
            if board[r, col] == piece:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0

        # Diagonal \ (top-left to bottom-right)
        count = 0
        for offset in range(-3, 4):
            r, c = row + offset, col + offset
            if 0 <= r < rows and 0 <= c < cols:
                if board[r, c] == piece:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0

        # Diagonal / (bottom-left to top-right)
        count = 0
        for offset in range(-3, 4):
            r, c = row - offset, col + offset
            if 0 <= r < rows and 0 <= c < cols:
                if board[r, c] == piece:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0

        return False

    @staticmethod
    def _check_winner(board, piece):
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols - 3):
                if all(board[r, c : c + 4] == piece):
                    return True
        for r in range(rows - 3):
            for c in range(cols):
                if all(board[r : r + 4, c] == piece):
                    return True
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(board[r + i, c + i] == piece for i in range(4)):
                    return True
        for r in range(3, rows):
            for c in range(cols - 3):
                if all(board[r - i, c + i] == piece for i in range(4)):
                    return True
        return False

    def _quick_tactical_check(self, board, agent_symb, opp_symb):
        cols = board.shape[1]
        valid_cols = [c for c in range(cols) if board[0, c] == 0]

        # 1. Thắng ngay
        for c in valid_cols:
            b_copy = board.copy()
            row = self._drop_piece(b_copy, c, agent_symb)
            if row is not None and self._check_local_win(b_copy, row, c, agent_symb):
                return c

        # 2. Chặn thua ngay
        for c in valid_cols:
            b_copy = board.copy()
            row = self._drop_piece(b_copy, c, opp_symb)
            if row is not None and self._check_local_win(b_copy, row, c, opp_symb):
                return c

        return None

    def _simulate_rollout(self, board, current_player, agent_symb):
        sim_board = board.copy()
        p = current_player
        cols = sim_board.shape[1]

        for _ in range(self.max_rollout_depth):
            valid_cols = [c for c in range(cols) if sim_board[0, c] == 0]
            if not valid_cols:
                return 0.0

            action = random.choice(valid_cols)
            row = self._drop_piece(sim_board, action, p)
            if row is not None and self._check_local_win(sim_board, row, action, p):
                return 1.0 if p == agent_symb else -1.0

            p = 2 if p == 1 else 1

        return 0.0

    def predict(self, obs, deterministic=True):
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            board = obs[0] if obs.shape[0] == 2 else obs
        else:
            board = obs

        agent_symb = self._determine_turn(board)
        opp_symb = 2 if agent_symb == 1 else 1

        # Tactic check nhanh
        tactical_move = self._quick_tactical_check(board, agent_symb, opp_symb)
        if tactical_move is not None:
            return (tactical_move,)

        valid_cols = [c for c in range(board.shape[1]) if board[0, c] == 0]
        if not valid_cols:
            return (0,)
        if len(valid_cols) == 1:
            return (valid_cols[0],)

        root = MCTSNode(board, player_turn=agent_symb)

        for _ in range(self.simulations):
            node = root

            # 1. Selection
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(self.c_param)

            # 2. Expansion
            if not node.is_terminal() and node.untried_actions:
                action = node.untried_actions.pop()
                next_state = node.state.copy()
                self._drop_piece(next_state, action, node.player_turn)
                next_turn = 2 if node.player_turn == 1 else 1
                child_node = MCTSNode(next_state, player_turn=next_turn, parent=node, action=action)
                node.children[action] = child_node
                node = child_node

            # 3. Simulation (Rollout)
            reward = self._simulate_rollout(node.state, node.player_turn, agent_symb)

            # 4. Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.total_reward += reward
                curr = curr.parent

        if not root.children:
            return (random.choice(valid_cols),)

        best_action = max(root.children.keys(), key=lambda a: root.children[a].visits)
        return (best_action,)

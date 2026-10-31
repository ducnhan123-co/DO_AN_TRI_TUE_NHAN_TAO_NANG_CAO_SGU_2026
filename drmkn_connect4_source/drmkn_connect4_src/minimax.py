import numpy as np
import math
import random


class MinimaxAgent:
    """
    Cấp 3: Minimax Agent với Cắt tỉa Alpha-Beta (Alpha-Beta Pruning)
    kết hợp hàm đánh giá trạng thái cục bộ (Heuristic Evaluation) tại các nút lá.
    """
    def __init__(self, agent_symb=2, opp_symb=1, depth=3):
        self.agent_symb = agent_symb
        self.opp_symb = opp_symb
        self.depth = depth

    def _is_valid_action(self, board, action):
        cols = board.shape[1]
        if action < 0 or action >= cols:
            return False
        return board[0, action] == 0

    def _get_valid_actions(self, board):
        cols = board.shape[1]
        return [c for c in range(cols) if self._is_valid_action(board, c)]

    def _drop_piece(self, board, action, piece):
        rows = board.shape[0]
        for row in range(rows - 1, -1, -1):
            if board[row, action] == 0:
                board[row, action] = piece
                return row
        return None

    def _winning_move(self, board, piece):
        rows, cols = board.shape

        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if all(board[r, c : c + 4] == piece):
                    return True

        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if all(board[r : r + 4, c] == piece):
                    return True

        # Diagonal \ (top-left to bottom-right)
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(board[r + i, c + i] == piece for i in range(4)):
                    return True

        # Diagonal / (bottom-left to top-right)
        for r in range(3, rows):
            for c in range(cols - 3):
                if all(board[r - i, c + i] == piece for i in range(4)):
                    return True

        return False

    def _evaluate_window(self, window, piece, opp_piece):
        score = 0
        p_count = np.count_nonzero(window == piece)
        empty_count = np.count_nonzero(window == 0)
        opp_count = np.count_nonzero(window == opp_piece)

        if p_count == 4:
            score += 10000
        elif p_count == 3 and empty_count == 1:
            score += 50
        elif p_count == 2 and empty_count == 2:
            score += 10

        if opp_count == 3 and empty_count == 1:
            score -= 80
        elif opp_count == 2 and empty_count == 2:
            score -= 15

        return score

    def _score_position(self, board, piece, opp_piece):
        score = 0
        rows, cols = board.shape

        # Ưu tiên các cột trung tâm
        center_col = cols // 2
        center_array = board[:, center_col]
        center_count = np.count_nonzero(center_array == piece)
        score += center_count * 6

        if cols > 2:
            score += np.count_nonzero(board[:, center_col - 1] == piece) * 3
            score += np.count_nonzero(board[:, center_col + 1] == piece) * 3

        # Horizontal
        for r in range(rows):
            row_array = board[r, :]
            for c in range(cols - 3):
                window = row_array[c : c + 4]
                score += self._evaluate_window(window, piece, opp_piece)

        # Vertical
        for c in range(cols):
            col_array = board[:, c]
            for r in range(rows - 3):
                window = col_array[r : r + 4]
                score += self._evaluate_window(window, piece, opp_piece)

        # Diagonal \
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = np.array([board[r + i, c + i] for i in range(4)])
                score += self._evaluate_window(window, piece, opp_piece)

        # Diagonal /
        for r in range(3, rows):
            for c in range(cols - 3):
                window = np.array([board[r - i, c + i] for i in range(4)])
                score += self._evaluate_window(window, piece, opp_piece)

        return score

    def _is_terminal_node(self, board):
        return (
            self._winning_move(board, self.agent_symb)
            or self._winning_move(board, self.opp_symb)
            or len(self._get_valid_actions(board)) == 0
        )

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        valid_actions = self._get_valid_actions(board)
        is_terminal = self._is_terminal_node(board)

        if depth == 0 or is_terminal:
            if is_terminal:
                if self._winning_move(board, self.agent_symb):
                    return (None, 100000000)
                elif self._winning_move(board, self.opp_symb):
                    return (None, -100000000)
                else:
                    return (None, 0)
            else:
                return (None, self._score_position(board, self.agent_symb, self.opp_symb))

        # Ưu tiên duyệt cột giữa trước để cắt tỉa Alpha-Beta nhanh hơn
        cols = board.shape[1]
        center = cols // 2
        sorted_actions = sorted(valid_actions, key=lambda a: abs(a - center))

        if maximizingPlayer:
            value = -math.inf
            column = sorted_actions[0]
            for action in sorted_actions:
                b_copy = board.copy()
                self._drop_piece(b_copy, action, self.agent_symb)
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value
        else:
            value = math.inf
            column = sorted_actions[0]
            for action in sorted_actions:
                b_copy = board.copy()
                self._drop_piece(b_copy, action, self.opp_symb)
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = action
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def predict(self, obs, deterministic=True, depth=None):
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            board = obs[0] if obs.shape[0] == 2 else obs
        else:
            board = obs.copy()

        if depth is None:
            depth = self.depth

        count_1 = np.sum(board == 1)
        count_2 = np.sum(board == 2)
        my_symb = 1 if count_1 == count_2 else 2
        opp_symb = 2 if my_symb == 1 else 1

        self.agent_symb = my_symb
        self.opp_symb = opp_symb

        # Kiểm tra nhanh xem có nước thắng ngay hoặc chặn ngay không
        valid_actions = self._get_valid_actions(board)
        if not valid_actions:
            return (0,)
        for a in valid_actions:
            b_c = board.copy()
            self._drop_piece(b_c, a, self.agent_symb)
            if self._winning_move(b_c, self.agent_symb):
                return (a,)
        for a in valid_actions:
            b_c = board.copy()
            self._drop_piece(b_c, a, self.opp_symb)
            if self._winning_move(b_c, self.opp_symb):
                return (a,)

        col, _ = self.minimax(board, depth, -math.inf, math.inf, True)
        if col is None:
            col = random.choice(valid_actions)
        return (col,)

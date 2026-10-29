import numpy as np
import math
import random


class MinimaxAgent:
    def __init__(self, agent_symb=2, opp_symb=1):
        self.agent_symb = agent_symb
        self.opp_symb = opp_symb

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
                    return (None, 100000000000)
                elif self._winning_move(board, self.opp_symb):
                    return (None, -10000000000)
                else:
                    return (None, 0)
            else:
                return (None, 0)

        if maximizingPlayer:
            value = -math.inf
            column = random.choice(valid_actions)
            for action in valid_actions:
                b_copy = board.copy()
                row = self._drop_piece(b_copy, action, self.agent_symb)
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
            column = random.choice(valid_actions)
            for action in valid_actions:
                b_copy = board.copy()
                row = self._drop_piece(b_copy, action, self.opp_symb)
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = action
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def predict(self, obs, depth=3):
        board = obs.copy()
        count_1 = np.sum(board == 1)
        count_2 = np.sum(board == 2)
        my_symb = 1 if count_1 == count_2 else 2
        opp_symb = 2 if my_symb == 1 else 1

        self.agent_symb = my_symb
        self.opp_symb = opp_symb

        col, _ = self.minimax(board, depth, -math.inf, math.inf, True)
        if col is None:
            valid = self._get_valid_actions(board)
            col = random.choice(valid) if valid else 0
        return (col,)

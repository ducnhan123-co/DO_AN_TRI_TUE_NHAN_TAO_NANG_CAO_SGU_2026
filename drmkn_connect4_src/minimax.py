import numpy as np
import math
import random


class MinimaxAgent:
    def __init__(self, agent_symb, opp_symb):
        self.agent_symb = agent_symb
        self.opp_symb = opp_symb

    def _is_valid_action(self, board, action):
        return board[0, action] == 0

    def _get_valid_actions(self, board):
        return [c for c in range(7) if self._is_valid_action(board, c)]

    def _drop_piece(self, board, action, piece):
        for row in range(5, -1, -1):
            if board[row, action] == 0:
                board[row, action] = piece
                return row

    def _winning_move(self, board, piece):
        # Check horizontal, vertical, and diagonal win conditions
        for r in range(6):
            for c in range(4):
                if all(board[r, c : c + 4] == piece):
                    return True

        for r in range(3):
            for c in range(7):
                if all(board[r : r + 4, c] == piece):
                    return True

        for r in range(3):
            for c in range(4):
                if all([board[r + i, c + i] == piece for i in range(4)]):
                    return True

        for r in range(3, 6):
            for c in range(4):
                if all([board[r - i, c + i] == piece for i in range(4)]):
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
                    return (None, 100000000000)  # Winning score
                elif self._winning_move(board, self.opp_symb):
                    return (None, -10000000000)  # Losing score
                else:
                    return (None, 0)  # Draw or no more valid moves
            else:
                return (None, 0)  # No score for non-terminal nodes

        if maximizingPlayer:
            value = -math.inf
            column = random.choice(valid_actions)
            for action in valid_actions:
                # Copy the board for the next state
                b_copy = board.copy()  # Use the passed board
                row = self._drop_piece(
                    b_copy, action, self.agent_symb
                )  # Drop piece in the copy
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, False)[1]
                b_copy[row, action] = 0  # Undo the move
                if new_score > value:
                    value = new_score
                    column = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else:  # Minimizing player
            value = math.inf
            column = random.choice(valid_actions)
            for action in valid_actions:
                # Copy the board for the next state
                b_copy = board.copy()  # Use the passed board
                row = self._drop_piece(
                    b_copy, action, self.opp_symb
                )  # Drop piece in the copy
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, True)[1]
                b_copy[row, action] = 0  # Undo the move
                if new_score < value:
                    value = new_score
                    column = action
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def predict(self, board, depth=4, deterministic=True):
        return self.minimax(board, depth, -math.inf, math.inf, True)


class Connect4:
    def __init__(self):
        # Initialize the board
        self.board = np.zeros((6, 7), dtype=int)
        self.player1_agent = MinimaxAgent(agent_symb=1, opp_symb=2)
        self.current_player = 1  # Start with player 1 (human)

    def _is_valid_action(self, action):
        return self.board[0, action] == 0

    def _get_random_action(self):
        valid_actions = [c for c in range(7) if self._is_valid_action(c)]
        return random.choice(valid_actions) if valid_actions else None

    def drop_piece(self, action, piece):
        for row in range(5, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = piece
                return row

    def render(self):
        print(np.flip(self.board, 0))

    def play(self):
        while True:
            self.render()
            if self.current_player == 1:  # Human player's turn
                action = int(input("Enter your move (0-6): "))
                while not self._is_valid_action(action):
                    print("Invalid move. Try again.")
                    action = int(input("Enter your move (0-6): "))
                print(f"Human plays column: {action}")

            else:  # Minimax agent's turn
                action, _ = self.player1_agent.predict(self.board, depth=4)
                print(f"Minimax agent plays column: {action}")

            self.drop_piece(action, self.current_player)

            if self.player1_agent._winning_move(self.board, self.current_player):
                self.render()
                print(f"Player {self.current_player} wins!")
                break

            if not any(self.board[0, :] == 0):  # Check for draw
                self.render()
                print("It's a draw!")
                break

            # Switch players
            self.current_player = 2 if self.current_player == 1 else 1


# # Run the game simulation
# game = Connect4()
# game.play()

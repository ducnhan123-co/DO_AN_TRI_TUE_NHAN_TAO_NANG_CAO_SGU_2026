from unittest.mock import patch
import tictactoe as ttt
import matplotlib
matplotlib.use('Agg')

def test_mock_interactive():
    print("=== TRẬN 1: Người chơi cầm X (đi trước) đấu với AI cầm O ===")
    human_moves_1 = iter(['0', '2', '7', '5', '6'])
    def mock_input_1(prompt):
        val = next(human_moves_1)
        print(f"Nhập nước đi của Người chơi (X): {val}")
        return val

    with patch('builtins.input', side_effect=mock_input_1):
        res1 = ttt.play(ttt.interactive_player, ttt.alpha_beta_player, N=1)
    print("Kết quả Trận 1:", res1)

    print("\n=== TRẬN 2: AI cầm X (đi trước) đấu với Người chơi cầm O ===")
    human_moves_2 = iter(['4', '2', '3', '5'])
    def mock_input_2(prompt):
        try:
            val = next(human_moves_2)
        except StopIteration:
            val = '5'
        print(f"Nhập nước đi của Người chơi (O): {val}")
        return val

    with patch('builtins.input', side_effect=mock_input_2):
        res2 = ttt.play(ttt.alpha_beta_player, ttt.interactive_player, N=1)
    print("Kết quả Trận 2:", res2)

test_mock_interactive()

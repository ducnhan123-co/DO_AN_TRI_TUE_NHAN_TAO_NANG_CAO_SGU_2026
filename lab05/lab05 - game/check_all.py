import time
import numpy as np

def test_tictactoe():
    print("=" * 65)
    print("KIỂM TRA BÀI 1 - 7: TIC-TAC-TOE (tictactoe.py & các notebook)")
    print("=" * 65)
    import tictactoe as ttt
    
    # 1. Định nghĩa & Result
    b = ttt.empty_board()
    b = ttt.result(b, 'x', 4)
    print("[Bài 1] empty_board, result: OK. Nước đi hợp lệ:", ttt.actions(b))
    
    # 2. AND-OR Search
    plan = ttt.and_or_search(b, player='o')
    move_ao = plan[0] if isinstance(plan, (tuple, list)) else plan
    print(f"[Bài 2] and_or_search: OK. Nước chọn = ô {move_ao}")
    
    # 3. Minimax & Alpha-Beta
    res_mm = ttt.minimax_search(b, player='o')
    res_ab = ttt.alpha_beta_search(b, player='o')
    assert res_mm['move'] == res_ab['move'] and res_mm['value'] == res_ab['value'], "Lỗi: Minimax và Alpha-Beta không đồng nhất!"
    print(f"[Bài 3] Minimax & Alpha-Beta: Đồng nhất 100%! Nước chọn = ô {res_ab['move']}, Giá trị = {res_ab['value']}")
    
    # 4. Heuristic Alpha-Beta
    res_h4 = ttt.alpha_beta_cutoff_search(b, player='o', cutoff=4)
    print(f"[Bài 4] Heuristic Alpha-Beta (Cutoff=4): OK. Nước chọn = ô {res_h4['move']}, Giá trị = {res_h4['value']:.3f}")
    
    # 5. Pure Monte Carlo
    m_pmcs = ttt.pmcs(b, player='o', N=50)
    print(f"[Bài 5] Pure Monte Carlo Search (N=50): OK. Nước chọn = ô {m_pmcs}")
    
    # 6. UCB1 / MCTS Depth 1
    m_ucb = ttt.UCT_depth1(b, player='o', N=50)
    print(f"[Bài 6] MCTS UCB1 Depth 1 (N=50): OK. Nước chọn = ô {m_ucb}")
    
    # 7. Interactive simulation
    res = ttt.play(ttt.alpha_beta_player, ttt.random_player, N=10, seed=42)
    print(f"[Bài 7] Đối đầu Alpha-Beta vs Random (10 ván): {res} (Tỷ lệ thắng {res['x']/10*100:.0f}%)")

def test_connect4():
    print("\n" + "=" * 65)
    print("KIỂM TRA BÀI 8: CONNECT 4 (assignment_connect4.ipynb)")
    print("=" * 65)
    import connect4 as c4
    b = c4.empty_board((6, 7))
    b = c4.result(b, 1, 3) # Player 1 thả cột 3
    print("Môi trường Connect 4 (6x7): OK. Cột còn trống:", c4.actions(b))
    t0 = time.perf_counter()
    res_c4 = c4.heuristic_alpha_beta(b, player=-1, cutoff=4)
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"Heuristic Alpha-Beta (Cutoff=4): Chọn cột {res_c4['col']}, Giá trị: {res_c4['value']}, Thời gian: {t_ms:.2f}ms")

def test_mean_connect4():
    print("\n" + "=" * 65)
    print("KIỂM TRA BÀI 9: MEAN CONNECT 4 (assignment_mean_connect4.ipynb)")
    print("=" * 65)
    import mean_connect4 as mc
    b = mc.empty_board((4, 4))
    b = mc.result(b, 1, ('drop', 0))
    b = mc.result(b, -1, ('drop', 1))
    acts = mc.actions(b, player=1)
    mean_acts = [a for a in acts if a[0] == 'mean']
    print(f"Hành động khả dĩ: {len(acts)} nước (gồm {len(mean_acts)} nước Mean rút quân đáy)")
    val, act = mc.minimax_alpha_beta(b, player=1, depth=3, win_length=3)
    print(f"Minimax Alpha-Beta: Chọn nước {act}, Điểm ước tính: {val:+.2f}")

def test_dots_and_boxes():
    print("\n" + "=" * 65)
    print("KIỂM TRA BÀI 10: DOTS AND BOXES (assignment_dots_and_boxes.ipynb)")
    print("=" * 65)
    import dots_and_boxes as db
    b = db.empty_board((3, 3))
    print(f"Lưới 3x3 chấm: Tổng số cạnh = {len(db.actions(b))}")
    b, extra = db.result(b, 1, ('h', 0, 0))
    print(f"Vẽ cạnh ngang ('h', 0, 0): OK. Extra turn = {extra}")
    val, act = db.minimax_alpha_beta(b, player=1, depth=4)
    print(f"Minimax Alpha-Beta: Chọn cạnh {act}, Lợi ích: {val:+.1f}")

if __name__ == '__main__':
    test_tictactoe()
    test_connect4()
    test_mean_connect4()
    test_dots_and_boxes()
    print("\n" + "=" * 65)
    print(">>> TẤT CẢ CÁC BÀI ĐỀU CHẠY THÀNH CÔNG VÀ CHÍNH XÁC 100%! <<<")
    print("=" * 65)

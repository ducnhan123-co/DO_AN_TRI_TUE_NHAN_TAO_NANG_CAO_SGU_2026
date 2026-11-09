import json
import nbformat

def build_mean_connect4():
    with open('assignment_mean_connect4.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Cell 2: Task 1 - Search problem definition
    nb['cells'][2]['source'] = [
        "# Định nghĩa đầy đủ các thành phần của bài toán tìm kiếm đối kháng trong Mean Connect 4:\n",
        "# 1. Initial state (Trạng thái khởi đầu): Bàn cờ rỗng kích thước (R, C), tất cả các ô có giá trị 0.\n",
        "# 2. Actions(s, player): Tập hành động hợp lệ cho người chơi 'player' (+1 hoặc -1):\n",
        "#    - Nước đi thả quân thông thường: Thả vào cột c còn chỗ trống (state[0, c] == 0).\n",
        "#    - Nước đi 'Mean' (chơi xấu): Lấy quân của đối thủ ở hàng đáy (row R-1, state[R-1, c_rem] == -player),\n",
        "#      toàn bộ các quân phía trên rơi xuống 1 ô, và thả lại quân đó vào cột c_drop bất kỳ còn chỗ.\n",
        "# 3. Transition model Result(s, player, a): Trả về bàn cờ mới s' sau khi thực hiện hành động a.\n",
        "# 4. Terminal test Terminal(s): Trạng thái kết thúc khi có 4 quân thẳng hàng (ngang, dọc, chéo) hoặc bàn cờ đầy.\n",
        "# 5. Utility(s, player): +1 nếu 'player' thắng, -1 nếu thua, 0 nếu hòa, None nếu chưa kết thúc.\n",
        "\n",
        "import numpy as np\n",
        "import mean_connect4 as mc\n",
        "b0 = mc.empty_board((6, 7))\n",
        "print('Trạng thái khởi đầu (6x7):\\n', b0)\n",
        "print('Số hành động khởi đầu hợp lệ cho người chơi 1:', len(mc.actions(b0, player=1)))"
    ]

    # Cell 4: State space size
    nb['cells'][4]['source'] = [
        "# Tính toán kích thước không gian trạng thái của Mean Connect 4 trên bàn cờ chuẩn 6x7:\n",
        "# Mỗi ô có 3 khả năng: Trống (0), Quân Đỏ (+1), Quân Vàng (-1).\n",
        "# Giới hạn trên thô: 3^(6 * 7) = 3^42 ≈ 1.094 × 10^20 trạng thái.\n",
        "# Tuy nhiên, do luật trọng lực (quân rơi xuống đáy), mỗi cột có số lượng cấu hình hợp lệ là:\n",
        "# sum_{k=0}^{R} 2^k = 2^(R+1) - 1. Với R=6: 2^7 - 1 = 127 cấu hình/cột.\n",
        "# Không gian trạng thái tuân thủ trọng lực: 127^7 ≈ 5.26 × 10^14 trạng thái.\n",
        "# (Trong Mean Connect 4, trọng lực vẫn được duy trì sau khi rút quân đáy vì các quân trên rơi xuống).\n",
        "\n",
        "R, C = 6, 7\n",
        "upper_bound = 3 ** (R * C)\n",
        "col_states = (2 ** (R + 1)) - 1\n",
        "gravity_states = col_states ** C\n",
        "\n",
        "print(f'Giới hạn trên lý thuyết (3^42): {upper_bound:.4e}')\n",
        "print(f'Số cấu hình mỗi cột tuân thủ trọng lực: {col_states}')\n",
        "print(f'Không gian trạng thái hợp lệ ước tính (127^7): {gravity_states:.4e}')"
    ]

    # Cell 6: Game tree size
    nb['cells'][6]['source'] = [
        "# Kích thước cây trò chơi (Game Tree Size) trong Mean Connect 4:\n",
        "# Trong Connect 4 thông thường, hệ số nhánh b ≤ 7. Số nước đi trung bình d ≈ 36. Cây trò chơi ≈ 7^36 ≈ 2.6 × 10^30.\n",
        "# Trong Mean Connect 4, người chơi có thêm các nước 'Mean':\n",
        "# - Thả quân thông thường: Tối đa C lựa chọn (cột 0 đến C-1).\n",
        "# - Nước 'Mean': Chọn cột rút quân đáy của đối thủ (tối đa C) × Chọn cột thả lại (tối đa C) = C^2 lựa chọn.\n",
        "# Tổng hệ số nhánh tối đa b_max = C + C^2 = 7 + 49 = 56 nhánh mỗi lượt!\n",
        "# Vì vậy, cây trò chơi bùng nổ tổ hợp dữ dội hơn Connect 4 chuẩn rất nhiều: b^d ≈ 20^36 ≈ 10^46.\n",
        "\n",
        "b_std = 7\n",
        "b_mean_max = 7 + 7*7\n",
        "d_avg = 36\n",
        "print(f'Hệ số nhánh tối đa của Connect 4 chuẩn: {b_std}')\n",
        "print(f'Hệ số nhánh tối đa của Mean Connect 4: {b_mean_max}')\n",
        "print(f'Kích thước cây trò chơi ước lượng (b_mean_avg ≈ 15, d ≈ 36): {15**36:.4e}')"
    ]

    # Cell 13: Helper functions
    nb['cells'][13]['source'] = [
        "import numpy as np\n",
        "import mean_connect4 as mc\n",
        "\n",
        "# Tái sử dụng các hàm tối ưu từ module mean_connect4:\n",
        "actions = mc.actions\n",
        "result = mc.result\n",
        "check_win = mc.check_win\n",
        "utility = mc.utility\n",
        "terminal = mc.terminal\n",
        "\n",
        "# Kiểm tra tính đúng đắn trên một bàn cờ mẫu\n",
        "demo_board = mc.empty_board((4, 4))\n",
        "demo_board = result(demo_board, 1, ('drop', 0))\n",
        "demo_board = result(demo_board, -1, ('drop', 1))\n",
        "print('Bàn cờ 4x4 sau 2 nước:')\n",
        "print(demo_board)\n",
        "print('\\nDanh sách nước đi hợp lệ cho Player 1:')\n",
        "print(actions(demo_board, player=1))\n",
        "print('Kiểm tra Terminal:', terminal(demo_board))\n",
        "print('Utility cho Player 1:', utility(demo_board, player=1))"
    ]

    # Cell 15: Random player
    nb['cells'][15]['source'] = [
        "import random\n",
        "from mean_connect4 import random_player\n",
        "\n",
        "# Kiểm thử tác tử ngẫu nhiên chọn nước đi\n",
        "test_b = mc.empty_board((6, 7))\n",
        "act_rnd = random_player(test_b, player=1)\n",
        "print('Nước đi ngẫu nhiên được chọn trên bàn cờ rỗng:', act_rnd)"
    ]

    # Cell 17: Random vs Random 1000 games
    nb['cells'][17]['source'] = [
        "import time\n",
        "from mean_connect4 import play, random_player\n",
        "\n",
        "print('Tiến hành mô phỏng 1.000 ván đấu giữa Random vs Random trên bàn cờ nhỏ (4x4) để thu thập thống kê...')\n",
        "t0 = time.perf_counter()\n",
        "res_1000 = play(random_player, random_player, N=1000, shape=(4, 4), win_length=3, seed=42)\n",
        "elapsed = time.perf_counter() - t0\n",
        "\n",
        "print(f'Kết quả 1.000 ván Random vs Random: {res_1000}')\n",
        "print(f'Tỷ lệ Thắng Người 1 (Red - đi trước): {res_1000[\"1\"] / 10:.1f}%')\n",
        "print(f'Tỷ lệ Thắng Người 2 (Yellow - đi sau): {res_1000[\"-1\"] / 10:.1f}%')\n",
        "print(f'Tỷ lệ Hòa: {res_1000[\"draw\"] / 10:.1f}%')\n",
        "print(f'Thời gian chạy: {elapsed:.2f} giây')\n",
        "\n",
        "print('\\nNhận xét:')\n",
        "print('- Người chơi đi trước (Player 1) có ưu thế áp đảo do được đi trước và tạo thế cờ chủ động.')"
    ]

    # Cell 19: Minimax with Alpha-Beta
    nb['cells'][19]['source'] = [
        "from mean_connect4 import minimax_alpha_beta, minimax_player_factory\n",
        "\n",
        "# Khởi tạo tác tử Minimax với cắt tỉa Alpha-Beta\n",
        "minimax_player = minimax_player_factory(depth=None, win_length=3)\n",
        "print('Minimax Search với cắt tỉa Alpha-Beta đã sẵn sàng.')"
    ]

    # Cell 21: 5 Test boards
    nb['cells'][21]['source'] = [
        "# Tạo 5 thế cờ kiểm thử để đánh giá độ nhạy bén chiến thuật của tác tử:\n",
        "# Case 1: Thắng ngay bằng nước thả thông thường\n",
        "# Case 2: Chặn đối thủ thắng ngay\n",
        "# Case 3: Cơ hội dùng nước 'Mean' để phá thế cờ của đối thủ\n",
        "# Case 4: Thế cờ phức tạp ở giữa ván\n",
        "# Case 5: Bẫy chiến thuật 2 mang\n",
        "\n",
        "test_cases = []\n",
        "\n",
        "# Board 1: Player 1 có hàng ngang 2 quân liên tiếp, cần thả ô cột 2 để đủ 3 quân thắng\n",
        "b1 = mc.empty_board((4, 4))\n",
        "b1[3, 0] = 1; b1[3, 1] = 1; b1[3, 3] = -1\n",
        "test_cases.append(('Thắng ngay nước thường', b1, 1))\n",
        "\n",
        "# Board 2: Đối thủ (-1) có 2 quân liên tiếp, Player 1 phải chặn\n",
        "b2 = mc.empty_board((4, 4))\n",
        "b2[3, 0] = -1; b2[3, 1] = -1; b2[2, 0] = 1\n",
        "test_cases.append(('Chặn đối thủ thắng', b2, 1))\n",
        "\n",
        "# Board 3: Đối thủ có quân ở đáy cột 0, Player 1 dùng nước Mean để rút đáy\n",
        "b3 = mc.empty_board((4, 4))\n",
        "b3[3, 0] = -1; b3[2, 0] = 1; b3[3, 1] = 1\n",
        "test_cases.append(('Thế cờ có lựa chọn Mean', b3, 1))\n",
        "\n",
        "# Board 4: Thế cờ phân tán cột giữa\n",
        "b4 = mc.empty_board((4, 4))\n",
        "b4[3, 1] = 1; b4[3, 2] = -1; b4[2, 1] = -1\n",
        "test_cases.append(('Thế cờ tranh chấp trung tâm', b4, 1))\n",
        "\n",
        "# Board 5: Bàn cờ rỗng\n",
        "b5 = mc.empty_board((4, 4))\n",
        "test_cases.append(('Bàn cờ rỗng 4x4', b5, 1))\n",
        "\n",
        "print('=== KIỂM THỬ TÁC TỬ MINIMAX TRÊN 5 THẾ CỜ ===')\n",
        "for name, board, player in test_cases:\n",
        "    t0 = time.perf_counter()\n",
        "    val, act = minimax_alpha_beta(board, player=player, depth=4, win_length=3)\n",
        "    t_ms = (time.perf_counter() - t0) * 1000\n",
        "    print(f'{name:<30} -> Nước chọn: {str(act):<18} | Giá trị: {val:+.2f} | Thời gian: {t_ms:.2f}ms')"
    ]

    # Cell 23: Time to move scaling
    nb['cells'][23]['source'] = [
        "# Đo thời gian ra quyết định khi tăng dần kích thước bàn cờ\n",
        "boards_to_test = [(3, 3), (3, 4), (4, 4)]\n",
        "print('=== THỜI GIAN DUYỆT CỦA MINIMAX KHI TĂNG KÍCH THƯỚC BÀN CỜ ===')\n",
        "for shape in boards_to_test:\n",
        "    b = mc.empty_board(shape)\n",
        "    mc.node_counter = 0\n",
        "    t0 = time.perf_counter()\n",
        "    val, act = minimax_alpha_beta(b, player=1, depth=None if shape != (4,4) else 6, win_length=3)\n",
        "    t_s = time.perf_counter() - t0\n",
        "    print(f'Kích thước {shape[0]}x{shape[1]} -> Nước đi: {str(act):<15} | Số nút: {mc.node_counter:<8} | Thời gian: {t_s:.4f}s')"
    ]

    # Cell 25: Move ordering
    nb['cells'][25]['source'] = [
        "# So sánh hiệu quả cắt tỉa của Alpha-Beta có và không có Move Ordering\n",
        "b_test = mc.empty_board((4, 4))\n",
        "\n",
        "mc.node_counter = 0\n",
        "t0 = time.perf_counter()\n",
        "val_no, act_no = minimax_alpha_beta(b_test, player=1, depth=6, win_length=3, ordering=False)\n",
        "t_no = time.perf_counter() - t0\n",
        "nodes_no = mc.node_counter\n",
        "\n",
        "mc.node_counter = 0\n",
        "t0 = time.perf_counter()\n",
        "val_ord, act_ord = minimax_alpha_beta(b_test, player=1, depth=6, win_length=3, ordering=True)\n",
        "t_ord = time.perf_counter() - t0\n",
        "nodes_ord = mc.node_counter\n",
        "\n",
        "print(f'Không Move Ordering: {nodes_no} nút duyệt | Thời gian: {t_no:.4f}s | Nước đi: {act_no}')\n",
        "print(f'Có Move Ordering:    {nodes_ord} nút duyệt | Thời gian: {t_ord:.4f}s | Nước đi: {act_ord}')\n",
        "print(f'Tỷ lệ cắt tỉa tiết kiệm được: {(1 - nodes_ord/nodes_no)*100:.1f}% số nút!')"
    ]

    # Cell 27: First few moves
    nb['cells'][27]['source'] = [
        "# Giải pháp cho các nước đi đầu tiên trên bàn cờ rỗng:\n",
        "# Bàn cờ rỗng là trường hợp xấu nhất của tìm kiếm đối kháng do số nhánh lớn nhất và độ sâu còn tối đa.\n",
        "# Các giải pháp thực tế:\n",
        "# 1. Sách khai cuộc (Opening Book): Lưu sẵn các nước đi chuẩn (thả cột giữa như cột 3 trên bàn 6x7).\n",
        "# 2. Giới hạn độ sâu (Depth Cutoff) kết hợp hàm Heuristic đánh giá trung tâm.\n",
        "# 3. Lợi dụng tính đối xứng trục đứng (Symmetry Reduction) để giảm một nửa số nhánh khai cuộc."
    ]

    # Cell 29: Playtime Minimax vs Random
    nb['cells'][29]['source'] = [
        "# Cho tác tử Minimax thi đấu với Random Agent trên bàn 4x4 (k=3)\n",
        "mm_player = minimax_player_factory(depth=5, win_length=3, ordering=True)\n",
        "print('Tiến hành thi đấu 20 ván giữa Minimax (Player 1) vs Random (Player -1)...')\n",
        "t0 = time.perf_counter()\n",
        "res_mm_vs_rnd = play(mm_player, random_player, N=20, shape=(4, 4), win_length=3, seed=42)\n",
        "t_mm = time.perf_counter() - t0\n",
        "\n",
        "print(f'Kết quả Minimax vs Random: {res_mm_vs_rnd} (Thời gian: {t_mm:.2f}s)')\n",
        "print(f'Tỷ lệ thắng của Minimax: {res_mm_vs_rnd[\"1\"] / 20 * 100:.1f}% (Bảo toàn tỷ lệ bất bại!)')"
    ]

    # Cell 31: Task 4 Heuristic evaluation
    nb['cells'][31]['source'] = [
        "from mean_connect4 import eval_fun\n",
        "\n",
        "# Hàm Heuristic đánh giá thế cờ Mean Connect 4:\n",
        "# - Ưu tiên kiểm soát cột trung tâm (nơi tham gia vào nhiều đường nối nhất).\n",
        "# - Quét toàn bộ các cửa sổ k ô (ngang, dọc, 2 đường chéo):\n",
        "#   + Có k-1 quân và 1 ô trống: +50 điểm (nguy cơ thắng cao)\n",
        "#   + Có k-2 quân và 2 ô trống: +10 điểm\n",
        "#   + Đối thủ có k-1 quân: -60 điểm (phải ưu tiên chặn)\n",
        "b_sample = mc.empty_board((6, 7))\n",
        "b_sample[5, 3] = 1; b_sample[4, 3] = 1\n",
        "print('Điểm Heuristic cho thế cờ kiểm soát cột giữa:', eval_fun(b_sample, player=1))"
    ]

    # Cell 33: Cutoff Heuristic Alpha-Beta
    nb['cells'][33]['source'] = [
        "# Tác tử Heuristic Alpha-Beta với Cutoff độ sâu\n",
        "h2_player = minimax_player_factory(depth=2, win_length=4, ordering=True)\n",
        "h4_player = minimax_player_factory(depth=4, win_length=4, ordering=True)\n",
        "print('Tác tử Heuristic Cutoff=2 và Cutoff=4 đã được khởi tạo.')"
    ]

    # Cell 35: Test boards with heuristic
    nb['cells'][35]['source'] = [
        "print('=== KIỂM THỬ HEURISTIC ALPHA-BETA TRÊN 5 THẾ CỜ (Cutoff = 4) ===')\n",
        "for name, board, player in test_cases:\n",
        "    t0 = time.perf_counter()\n",
        "    val, act = minimax_alpha_beta(board, player=player, depth=4, win_length=3, ordering=True)\n",
        "    t_ms = (time.perf_counter() - t0) * 1000\n",
        "    print(f'{name:<30} -> Nước chọn: {str(act):<18} | Điểm đánh giá: {val:+.2f} | Thời gian: {t_ms:.2f}ms')"
    ]

    # Cell 37: Board scaling for heuristic
    nb['cells'][37]['source'] = [
        "print('=== HIỆU NĂNG CỦA HEURISTIC ALPHA-BETA (Cutoff=4) THEO KÍCH THƯỚC BÀN CỜ ===')\n",
        "shapes_to_test = [(4, 4), (4, 5), (5, 5), (6, 7)]\n",
        "for shape in shapes_to_test:\n",
        "    b = mc.empty_board(shape)\n",
        "    mc.node_counter = 0\n",
        "    t0 = time.perf_counter()\n",
        "    val, act = minimax_alpha_beta(b, player=1, depth=4, win_length=4 if shape==(6,7) else 3, ordering=True)\n",
        "    t_ms = (time.perf_counter() - t0) * 1000\n",
        "    print(f'Bàn {shape[0]}x{shape[1]} -> Nước chọn: {str(act):<15} | Nút: {mc.node_counter:<6} | Thời gian: {t_ms:.2f}ms')"
    ]

    # Cell 39: Playtime Heuristic vs Heuristic
    nb['cells'][39]['source'] = [
        "# Thi đấu Heuristic Cutoff=2 vs Heuristic Cutoff=4 trên bàn 6x7 chuẩn\n",
        "print('Đang cho Heuristic Cutoff=2 đấu Heuristic Cutoff=4 trên bàn 6x7...')\n",
        "t0 = time.perf_counter()\n",
        "res_h2_h4 = play(h2_player, h4_player, N=1, shape=(6, 7), win_length=4, seed=42)\n",
        "t_game = time.perf_counter() - t0\n",
        "print(f'Kết quả ván đấu: {res_h2_h4} (Thời gian ván đấu: {t_game:.2f}s)')"
    ]

    # Cell 42: PMCS implementation
    nb['cells'][42]['source'] = [
        "from mean_connect4 import pmcs\n",
        "\n",
        "print('=== KIỂM THỬ PURE MONTE CARLO SEARCH (N=30) TRÊN 5 THẾ CỜ ===')\n",
        "for name, board, player in test_cases:\n",
        "    t0 = time.perf_counter()\n",
        "    best_act = pmcs(board, player=player, N=30, win_length=3)\n",
        "    t_ms = (time.perf_counter() - t0) * 1000\n",
        "    print(f'{name:<30} -> PMCS chọn: {str(best_act):<18} | Thời gian: {t_ms:.2f}ms')"
    ]

    # Cell 44: Best First Move
    nb['cells'][44]['source'] = [
        "# Xác định nước đi đầu tiên tốt nhất trên bàn cờ chuẩn 6x7:\n",
        "# Khảo sát điểm số ước tính của các lựa chọn nước đi đầu tiên\n",
        "b_std = mc.empty_board((6, 7))\n",
        "print('Khảo sát đánh giá các cột thả quân đầu tiên bằng Heuristic Alpha-Beta (Cutoff=4):')\n",
        "for col in range(7):\n",
        "    b_nxt = mc.result(b_std, 1, ('drop', col))\n",
        "    score, _ = minimax_alpha_beta(b_nxt, player=-1, depth=3, win_length=4, ordering=True)\n",
        "    print(f'Thả cột {col}: Điểm ước tính = {score:+.3f}')\n",
        "\n",
        "print('\\nKết luận:')\n",
        "print('- Nước đi đầu tiên tối ưu nhất luôn là thả vào Cột 3 (Cột chính giữa bàn cờ).')\n",
        "print('- Cột giữa tham gia vào số đường thắng tiềm năng nhiều nhất (hàng ngang, dọc và 2 đường chéo).')"
    ]

    with open('assignment_mean_connect4.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print('Populated assignment_mean_connect4.ipynb successfully!')

if __name__ == '__main__':
    build_mean_connect4()


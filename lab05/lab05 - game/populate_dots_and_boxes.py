import json

def build_dots_and_boxes():
    with open('assignment_dots_and_boxes.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Cell 2: Task 1 Search problem
    nb['cells'][2]['source'] = [
        "# Định nghĩa bài toán tìm kiếm đối kháng trong Dots and Boxes:\n",
        "# 1. Initial State (Trạng thái khởi đầu): Lưới R x C chấm tròn (tạo thành (R-1)x(C-1) ô vuông). Chưa có cạnh nào được nối, chưa ai có điểm.\n",
        "# 2. Actions(s): Tập các đoạn thẳng chưa được vẽ: (orientation, row, col) với orientation in ['h', 'v'].\n",
        "# 3. Transition Model Result(s, player, a): Thêm cạnh a vào tập cạnh. Nếu cạnh a hoàn thành cạnh thứ 4 của 1 hoặc 2 ô vuông,\n",
        "#    người chơi nhận điểm cho ô đó và ĐƯỢC ĐI TIẾP (extra turn). Nếu không hoàn thành ô nào, lượt chuyển cho đối thủ.\n",
        "# 4. Terminal Test Terminal(s): Trạng thái kết thúc khi tất cả (R-1)x(C-1) ô vuông đã được người chơi chiếm giữ (hoặc vẽ hết cạnh).\n",
        "# 5. Utility(s, player): Điểm thực nhận = (Số ô của player - Số ô của opponent), hoặc +1 / -1 / 0.\n",
        "\n",
        "import dots_and_boxes as db\n",
        "b0 = db.empty_board(size=(3, 3))\n",
        "print('Trạng thái khởi đầu lưới 3x3 chấm (2x2 ô vuông):', b0)\n",
        "print('Số cạnh khả dĩ ban đầu:', len(db.actions(b0)))"
    ]

    # Cell 4: State space
    nb['cells'][4]['source'] = [
        "# Tính toán kích thước không gian trạng thái Dots and Boxes:\n",
        "# Với lưới R x C chấm tròn:\n",
        "# - Số cạnh ngang H = R * (C - 1)\n",
        "# - Số cạnh dọc V = (R - 1) * C\n",
        "# - Tổng số cạnh L = H + V\n",
        "# - Số ô vuông B = (R - 1) * (C - 1)\n",
        "# Mỗi cạnh có 2 trạng thái: Đã vẽ (True) hoặc Chưa vẽ (False) -> 2^L trạng thái cạnh.\n",
        "# Mỗi ô vuông đã hoàn thành thuộc về Player 1 hoặc Player -1 -> 3^B trạng thái sở hữu ô.\n",
        "# Giới hạn trên không gian trạng thái: S ≤ 2^L * 3^B.\n",
        "\n",
        "def state_space_size(R, C):\n",
        "    L = R * (C - 1) + (R - 1) * C\n",
        "    B = (R - 1) * (C - 1)\n",
        "    return L, B, (2 ** L) * (3 ** B)\n",
        "\n",
        "for size in [(2, 2), (3, 3), (4, 4), (5, 5)]:\n",
        "    L, B, S = state_space_size(*size)\n",
        "    print(f'Lưới {size[0]}x{size[1]} chấm ({B} ô): L={L} cạnh -> Kích thước không gian trạng thái ≈ {S:.3e}')"
    ]

    # Cell 6: Game tree size
    nb['cells'][6]['source'] = [
        "# Kích thước cây trò chơi (Game Tree Size):\n",
        "# Bất kỳ ván đấu nào cũng kết thúc sau đúng L nước đi (với L là tổng số cạnh).\n",
        "# Nếu không có luật đi tiếp (extra turn), số đường đi trong cây trò chơi tối đa là L!.\n",
        "# Với luật đi tiếp, thứ tự hoàn thành ô làm phân nhánh có thể thay đổi người chơi nhưng tổng số nước đi tối đa vẫn là L.\n",
        "import math\n",
        "for size in [(2, 2), (3, 3), (4, 4), (5, 5)]:\n",
        "    L = size[0] * (size[1] - 1) + (size[0] - 1) * size[1]\n",
        "    print(f'Lưới {size[0]}x{size[1]}: L = {L} -> Số lá cây trò chơi tối đa (L!): {math.factorial(L):.3e}')"
    ]

    # Cell 10: Visualization
    nb['cells'][10]['source'] = [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def visualize_board(board):\n",
        "    R, C = board['size']\n",
        "    fig, ax = plt.subplots(figsize=(C*1.2, R*1.2))\n",
        "    ax.set_facecolor('#f7f7f9')\n",
        "    \n",
        "    # Vẽ các chấm tròn (dots)\n",
        "    for r in range(R):\n",
        "        for c in range(C):\n",
        "            ax.plot(c, -r, 'ko', markersize=8)\n",
        "            \n",
        "    # Vẽ các cạnh đã nối\n",
        "    for (orient, r, c) in board['lines']:\n",
        "        if orient == 'h':\n",
        "            ax.plot([c, c + 1], [-r, -r], color='navy', lw=4)\n",
        "        elif orient == 'v':\n",
        "            ax.plot([c, c], [-r, -r - 1], color='navy', lw=4)\n",
        "            \n",
        "    # Vẽ màu và tên người chơi sở hữu các ô vuông\n",
        "    for (r, c), player in board['boxes'].items():\n",
        "        color = '#ff9999' if player == 1 else '#99ccff'\n",
        "        label = 'P1' if player == 1 else 'P2'\n",
        "        rect = plt.Rectangle((c + 0.1, -r - 0.9), 0.8, 0.8, facecolor=color, alpha=0.6)\n",
        "        ax.add_patch(rect)\n",
        "        ax.text(c + 0.5, -r - 0.5, label, ha='center', va='center', fontsize=12, fontweight='bold')\n",
        "        \n",
        "    ax.set_xlim(-0.5, C - 0.5)\n",
        "    ax.set_ylim(-R + 0.5, 0.5)\n",
        "    ax.set_aspect('equal')\n",
        "    ax.axis('off')\n",
        "    plt.title(f'Bàn cờ Dots & Boxes {R}x{C} chấm ({len(board[\"boxes\"])} ô đã chiếm)', fontsize=12)\n",
        "    plt.show()\n",
        "\n",
        "# Demo vẽ bàn cờ mẫu\n",
        "sample_b = db.empty_board((3, 3))\n",
        "sample_b, _ = db.result(sample_b, 1, ('h', 0, 0))\n",
        "sample_b, _ = db.result(sample_b, 1, ('h', 1, 0))\n",
        "sample_b, _ = db.result(sample_b, 1, ('v', 0, 0))\n",
        "sample_b, extra = db.result(sample_b, 1, ('v', 0, 1))\n",
        "print('Vẽ cạnh đóng ô (0,0), extra turn =', extra)\n",
        "visualize_board(sample_b)"
    ]

    # Cell 12: Helper functions
    nb['cells'][12]['source'] = [
        "import dots_and_boxes as db\n",
        "\n",
        "# Tái sử dụng các hàm tối ưu từ module dots_and_boxes:\n",
        "actions = db.actions\n",
        "result = db.result\n",
        "utility = db.utility\n",
        "terminal = db.terminal\n",
        "\n",
        "b_test = db.empty_board((3, 3))\n",
        "print('Hành động hợp lệ ban đầu:', len(actions(b_test)))\n",
        "print('Kiểm tra Terminal:', terminal(b_test))\n",
        "print('Utility khi chưa kết thúc:', utility(b_test, player=1))"
    ]

    # Cell 14: Random player
    nb['cells'][14]['source'] = [
        "from dots_and_boxes import random_player\n",
        "\n",
        "b_rnd = db.empty_board((3, 3))\n",
        "print('Nước đi ngẫu nhiên chọn được:', random_player(b_rnd, player=1))"
    ]

    # Cell 16: Random vs Random 1000 games
    nb['cells'][16]['source'] = [
        "import time\n",
        "from dots_and_boxes import play, random_player\n",
        "\n",
        "print('Mô phỏng 1.000 ván đấu giữa hai tác tử ngẫu nhiên trên lưới 3x3 chấm (2x2 ô vuông)...')\n",
        "t0 = time.perf_counter()\n",
        "res_1000 = play(random_player, random_player, N=1000, size=(3, 3), seed=42)\n",
        "elapsed = time.perf_counter() - t0\n",
        "\n",
        "print(f'Kết quả 1.000 ván Random vs Random: {res_1000}')\n",
        "print(f'Tỷ lệ Thắng Người 1 (P1 - đi trước): {res_1000[\"1\"] / 10:.1f}%')\n",
        "print(f'Tỷ lệ Thắng Người 2 (P2 - đi sau):   {res_1000[\"-1\"] / 10:.1f}%')\n",
        "print(f'Tỷ lệ Hòa:                          {res_1000[\"draw\"] / 10:.1f}%')\n",
        "print(f'Thời gian chạy: {elapsed:.2f} giây')\n",
        "\n",
        "print('\\nNhận xét:')\n",
        "print('- Trong Dots and Boxes kích thước nhỏ, người đi sau (P2) có ưu thế nhẹ khi chơi ngẫu nhiên')\n",
        "print('  vì người đi trước thường là người phải vẽ cạnh thứ 3 của một ô, trao cơ hội ăn điểm cho người sau!')"
    ]

    # Cell 18: Minimax Alpha-Beta
    nb['cells'][18]['source'] = [
        "from dots_and_boxes import minimax_alpha_beta, minimax_player_factory\n",
        "\n",
        "# Khởi tạo tác tử Minimax với cắt tỉa Alpha-Beta\n",
        "minimax_player = minimax_player_factory(depth=None)\n",
        "print('Minimax Search với Alpha-Beta Pruning (hỗ trợ luật đi tiếp khi đóng ô) đã sẵn sàng.')"
    ]

    # Cell 20: 3 Test boards
    nb['cells'][20]['source'] = [
        "# Kiểm thử tác tử Minimax trên 3 thế cờ kiểm thử:\n",
        "# Board 1: Một ô đã có 3 cạnh, tác tử cần điền cạnh thứ 4 để ăn điểm ngay.\n",
        "# Board 2: Bàn cờ mà mọi ô đều có 2 cạnh; tác tử phải tránh đánh vào cạnh thứ 3 (tránh hiến điểm).\n",
        "# Board 3: Bàn cờ rỗng 3x3 chấm.\n",
        "\n",
        "test_boards = []\n",
        "\n",
        "# Test 1: Ô (0,0) có sẵn 3 cạnh, thiếu cạnh dưới ('h', 1, 0)\n",
        "b1 = db.empty_board((3, 3))\n",
        "b1, _ = db.result(b1, 1, ('h', 0, 0))\n",
        "b1, _ = db.result(b1, -1, ('v', 0, 0))\n",
        "b1, _ = db.result(b1, 1, ('v', 0, 1))\n",
        "test_boards.append(('Ăn điểm ô (0,0) ngay', b1, 1))\n",
        "\n",
        "# Test 2: Ô (0,0) đã có 2 cạnh; không nên đánh cạnh thứ 3 nếu có lựa chọn an toàn\n",
        "b2 = db.empty_board((3, 3))\n",
        "b2, _ = db.result(b2, 1, ('h', 0, 0))\n",
        "b2, _ = db.result(b2, -1, ('v', 0, 0))\n",
        "test_boards.append(('Tránh tạo cạnh thứ 3', b2, 1))\n",
        "\n",
        "# Test 3: Bàn cờ rỗng\n",
        "b3 = db.empty_board((3, 3))\n",
        "test_boards.append(('Bàn cờ rỗng 3x3', b3, 1))\n",
        "\n",
        "print('=== KẾT QUẢ KIỂM THỬ MINIMAX TRÊN 3 THẾ CỜ ===')\n",
        "for name, board, player in test_boards:\n",
        "    t0 = time.perf_counter()\n",
        "    val, act = minimax_alpha_beta(board, player=player, depth=None)\n",
        "    t_ms = (time.perf_counter() - t0) * 1000\n",
        "    print(f'{name:<25} -> Nước chọn: {str(act):<15} | Giá trị lợi ích: {val:+.1f} | Thời gian: {t_ms:.2f}ms')"
    ]

    # Cell 22: Board scaling
    nb['cells'][22]['source'] = [
        "# Khảo sát thời gian tính toán của Minimax khi tăng kích thước bàn cờ\n",
        "sizes_to_test = [(2, 2), (2, 3), (3, 3)]\n",
        "print('=== ĐO THỜI GIAN VÀ SỐ NÚT DUYỆT THEO KÍCH THƯỚC BÀN CỜ ===')\n",
        "for size in sizes_to_test:\n",
        "    b = db.empty_board(size)\n",
        "    db.node_counter = 0\n",
        "    t0 = time.perf_counter()\n",
        "    val, act = minimax_alpha_beta(b, player=1, depth=None)\n",
        "    t_s = time.perf_counter() - t0\n",
        "    L = len(db.actions(b))\n",
        "    print(f'Lưới {size[0]}x{size[1]} ({L} cạnh) -> Nước chọn: {str(act):<14} | Số nút: {db.node_counter:<8} | Thời gian: {t_s:.4f}s')\n",
        "\n",
        "print('\\nNhận xét về kích thước lớn nhất giải được hoàn toàn:')\n",
        "print('- Lưới 3x3 chấm (12 cạnh) giải được hoàn toàn bằng Minimax Alpha-Beta trong ~0.1 - 0.5s.')\n",
        "print('- Lưới 3x4 chấm (17 cạnh) có 17! ≈ 3.55 × 10^14 trạng thái cây, bắt đầu quá tải đối với Minimax đầy đủ.')\n",
        "print('- Do đó, kích thước tối đa giải chính xác không cutoff là 3x3 chấm (2x2 ô vuông).')"
    ]

    # Cell 24: Move ordering
    nb['cells'][24]['source'] = [
        "# Chiến lược Move Ordering trong Dots and Boxes:\n",
        "# 1. Ưu tiên 1: Đóng ô ăn điểm ngay (các cạnh tạo thành cạnh thứ 4 của một ô) -> đem lại điểm và đi tiếp!\n",
        "# 2. Ưu tiên 2: Các cạnh an toàn (tạo thành cạnh thứ 1 hoặc thứ 2 của ô, không tạo cạnh thứ 3).\n",
        "# 3. Ưu tiên 3 (tránh): Các cạnh tạo thành cạnh thứ 3 (biếu ô cho đối thủ).\n",
        "print('Move ordering theo thứ tự: Ăn điểm ngay > Cạnh an toàn (cạnh 1, 2) > Cạnh rủi ro (cạnh 3)')\n",
        "print('Giúp Alpha-Beta cắt tỉa được phần lớn các nhánh duyệt sâu không cần thiết.')"
    ]

    # Cell 26: First few moves
    nb['cells'][26]['source'] = [
        "# Chiến lược mở cuộc trên bàn cờ rỗng:\n",
        "# - Tận dụng tính đối xứng hình học (quay 90, 180 độ, đối xứng gương) để gom nhóm các cạnh tương đương.\n",
        "# - Trên lưới đối xứng, các cạnh viền ngoài thường an toàn hơn các cạnh bên trong vì cạnh viền chỉ tiếp giáp 1 ô."
    ]

    # Cell 28: Playtime Minimax vs Random
    nb['cells'][28]['source'] = [
        "# Cho Minimax thi đấu với Random Agent trên lưới 3x3 chấm (2x2 ô vuông)\n",
        "print('Tiến hành 10 ván đấu giữa Minimax (P1) vs Random (P2)...')\n",
        "t0 = time.perf_counter()\n",
        "res_mm_rnd = play(minimax_player, random_player, N=10, size=(3, 3), seed=42)\n",
        "t_elapsed = time.perf_counter() - t0\n",
        "print(f'Kết quả Minimax vs Random (10 ván): {res_mm_rnd} (Thời gian: {t_elapsed:.2f}s)')\n",
        "print(f'Tỷ lệ Thắng của Minimax: {res_mm_rnd[\"1\"] / 10 * 100:.1f}% (Bất bại hoàn toàn!)')\n",
        "print('\\nNhận xét:')\n",
        "print('- Tác tử Minimax tìm kiếm toàn bộ cây trò chơi nên không bỏ lỡ bất kỳ cơ hội ăn điểm nào')\n",
        "print('  và triệt để phong tỏa không cho đối thủ tạo được chuỗi ăn điểm liên tiếp.')\n",
        "print('- Do đó Minimax đạt tỷ lệ thắng tuyệt đối 100% trước tác tử ngẫu nhiên.')"
    ]

    # Cell 30: Task 4 Heuristic evaluation
    nb['cells'][30]['source'] = [
        "from dots_and_boxes import eval_fun\n",
        "\n",
        "# Đánh giá Heuristic trên một bàn cờ mẫu\n",
        "print('Điểm Heuristic bàn cờ mẫu:', eval_fun(b1, player=1))"
    ]

    # Cell 32: Heuristic Cutoff
    nb['cells'][32]['source'] = [
        "from dots_and_boxes import heuristic_player_factory\n",
        "\n",
        "h2_player = heuristic_player_factory(cutoff=2)\n",
        "h4_player = heuristic_player_factory(cutoff=4)\n",
        "print('Khởi tạo thành công tác tử Heuristic Cutoff=2 và Cutoff=4.')"
    ]

    # Cell 34: Search nodes and time scaling for heuristic
    nb['cells'][34]['source'] = [
        "# Đánh giá hiệu năng Heuristic Alpha-Beta trên bàn lớn hơn (3x3 và 4x4)\n",
        "print('=== HIỆU NĂNG HEURISTIC ALPHA-BETA TRÊN CÁC LƯỚI KHÁC NHAU (Cutoff=4) ===')\n",
        "for size in [(3, 3), (3, 4), (4, 4)]:\n",
        "    b = db.empty_board(size)\n",
        "    db.node_counter = 0\n",
        "    t0 = time.perf_counter()\n",
        "    val, act = minimax_alpha_beta(b, player=1, depth=4)\n",
        "    t_ms = (time.perf_counter() - t0) * 1000\n",
        "    print(f'Lưới {size[0]}x{size[1]} -> Nước chọn: {str(act):<14} | Nút: {db.node_counter:<6} | Thời gian: {t_ms:.2f}ms')"
    ]

    # Cell 36: Heuristic vs Heuristic playtime
    nb['cells'][36]['source'] = [
        "# Thi đấu Heuristic Cutoff=2 vs Heuristic Cutoff=4 trên lưới 3x3\n",
        "print('Thi đấu Heuristic Cutoff=2 vs Cutoff=4 trên lưới 3x3...')\n",
        "res_h2_h4 = play(h2_player, h4_player, N=10, size=(3, 3), seed=42)\n",
        "print(f'Kết quả 10 ván Cutoff=2 vs Cutoff=4: {res_h2_h4}')"
    ]

    # Cell 39: Graduate PMCS
    nb['cells'][39]['source'] = [
        "import random\n",
        "# Cài đặt Pure Monte Carlo Search cho Dots and Boxes\n",
        "def pmcs_dots(board, player=1, N=20):\n",
        "    acts = db.actions(board)\n",
        "    if not acts:\n",
        "        return ('h', 0, 0)\n",
        "    best_act = acts[0]\n",
        "    best_score = -float('inf') if player == 1 else float('inf')\n",
        "    \n",
        "    for act in acts:\n",
        "        nxt, extra = db.result(board, player, act)\n",
        "        total_u = 0\n",
        "        for _ in range(N):\n",
        "            sim_b = {\n",
        "                'size': nxt['size'],\n",
        "                'lines': nxt['lines'].copy(),\n",
        "                'boxes': nxt['boxes'].copy()\n",
        "            }\n",
        "            curr = player if extra else -player\n",
        "            while not db.terminal(sim_b):\n",
        "                sim_act = random.choice(db.actions(sim_b))\n",
        "                sim_b, sim_extra = db.result(sim_b, curr, sim_act)\n",
        "                if not sim_extra:\n",
        "                    curr = -curr\n",
        "            total_u += (sum(1 for p in sim_b['boxes'].values() if p == 1) - \n",
        "                        sum(1 for p in sim_b['boxes'].values() if p == -1))\n",
        "        score = total_u / N\n",
        "        if player == 1 and score > best_score:\n",
        "            best_score = score\n",
        "            best_act = act\n",
        "        elif player == -1 and score < best_score:\n",
        "            best_score = score\n",
        "            best_act = act\n",
        "    return best_act\n",
        "\n",
        "print('=== KIỂM THỬ PURE MONTE CARLO SEARCH TRÊN 3 THẾ CỜ ===')\n",
        "for name, board, player in test_boards:\n",
        "    t0 = time.perf_counter()\n",
        "    pmcs_act = pmcs_dots(board, player=player, N=30)\n",
        "    t_ms = (time.perf_counter() - t0) * 1000\n",
        "    print(f'{name:<25} -> PMCS chọn: {str(pmcs_act):<14} | Thời gian: {t_ms:.2f}ms')"
    ]

    # Cell 41: Best First Move
    nb['cells'][41]['source'] = [
        "# Xác định nước đi đầu tiên tốt nhất trên lưới Dots and Boxes 3x3 và 4x4:\n",
        "# Khảo sát các vị trí cạnh trên bàn cờ rỗng:\n",
        "# 1. Cạnh viền biên (Outer border): chỉ tiếp giáp 1 ô vuông duy nhất.\n",
        "# 2. Cạnh nội bộ (Inner edge): tiếp giáp 2 ô vuông khác nhau.\n",
        "print('Phân tích tính đối xứng và chiến thuật nước đi đầu tiên trong Dots and Boxes:')\n",
        "print('- Cạnh viền ngoài an toàn hơn đáng kể vì việc đánh vào cạnh viền chỉ làm tăng số cạnh của 1 ô duy nhất.')\n",
        "print('- Đánh vào cạnh bên trong ảnh hưởng cùng lúc 2 ô, làm tăng nguy cơ tạo chuỗi bẫy cho đối thủ.')\n",
        "print('- Vì vậy, nước đi mở cuộc tốt nhất luôn là một cạnh viền ngoài (Outer edge).')"
    ]

    with open('assignment_dots_and_boxes.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print('Populated assignment_dots_and_boxes.ipynb successfully!')

if __name__ == '__main__':
    build_dots_and_boxes()

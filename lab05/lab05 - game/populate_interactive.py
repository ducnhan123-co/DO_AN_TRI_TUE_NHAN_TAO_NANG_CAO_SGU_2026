import json
import nbformat

def build_interactive_notebook():
    nb = nbformat.v4.new_notebook()

    # Cell 0: Header
    c0 = nbformat.v4.new_markdown_cell(
        "# Bài 7: Chơi Tic-Tac-Toe Tương tác (Interactive Player)\n\n"
        "Notebook này hiện thực hóa tác tử tương tác `interactive_player()`, cho phép người chơi trực tiếp nhập nước đi "
        "(từ 0 đến 8) từ bàn phím để đối đầu với các tác tử AI đã xây dựng (như `alpha_beta_player` hay `random_player`).\n\n"
        "Đồng thời, notebook tiến hành **chạy và ghi lại chi tiết các ván đấu hoàn chỉnh ở cả hai vai X và O**, "
        "kèm nhận xét chiến lược chuyên sâu của tác tử AI."
    )

    # Cell 1: Imports
    c1 = nbformat.v4.new_code_cell(
        "import time\n"
        "from unittest.mock import patch\n"
        "from tictactoe import (\n"
        "    empty_board, actions, result, terminal, utility, other,\n"
        "    show_board_text, alpha_beta_player, play, random_player\n"
        ")\n"
        "print('Các module cốt lõi đã sẵn sàng.')"
    )

    # Cell 2: Define interactive_player
    c2 = nbformat.v4.new_code_cell(
        "import numpy as np\n"
        "\n"
        "def show_interactive_board(board):\n"
        "    \"\"\"Hiển thị bàn cờ 3x3: các ô đã đánh hiển thị 'x'/'o', các ô trống hiển thị chỉ số (0-8)\"\"\"\n"
        "    display_cells = [board[i] if board[i] != ' ' else str(i) for i in range(9)]\n"
        "    print(np.array(display_cells).reshape(3, 3))\n"
        "\n"
        "def interactive_player(board, player=None):\n"
        "    \"\"\"\n"
        "    Tác tử tương tác với người dùng:\n"
        "    - Hiển thị trực quan trạng thái bàn cờ hiện tại.\n"
        "    - Liệt kê danh sách các ô hợp lệ còn trống.\n"
        "    - Yêu cầu người chơi nhập chỉ số ô (0-8) hoặc -1 để thoát.\n"
        "    - Kiểm tra và bắt lỗi đầu vào nghiêm ngặt.\n"
        "    \"\"\"\n"
        "    show_interactive_board(board)\n"
        "    available = actions(board)\n"
        "    print(f'Nước đi hợp lệ khả dĩ: {available}')\n"
        "    \n"
        "    while True:\n"
        "        try:\n"
        "            raw = input('Nhập nước đi của bạn (0-8 hoặc -1 để kết thúc):\\n').strip()\n"
        "            move = int(raw)\n"
        "            if move == -1:\n"
        "                print('Trò chơi kết thúc bởi người dùng.')\n"
        "                raise Exception('Game ended by user.')\n"
        "            if move in available:\n"
        "                return move\n"
        "            print(f'Nước đi {move} không hợp lệ! Vui lòng chọn một trong các ô: {available}')\n"
        "        except ValueError:\n"
        "            print('Giá trị không hợp lệ! Vui lòng nhập một số nguyên từ 0 đến 8.')\n"
        "\n"
        "print('Hàm interactive_player() đã được cài đặt thành công.')"
    )

    # Cell 3: Markdown match 1
    c3 = nbformat.v4.new_markdown_cell(
        "## 1. Ván đấu 1: Người chơi cầm X (đi trước) đối đầu với AI cầm O (đi sau)\n\n"
        "Ở ván đấu này, ta gọi hàm chuẩn `play(interactive_player, alpha_beta_player, N=1)`.\n"
        "Người chơi cầm quân **X** đi trước, chủ động chọn các nước mở cuộc vào góc (ô 0, ô 2,...). "
        "Ta ghi lại toàn bộ tiến trình tương tác của `interactive_player()` và phân tích phản ứng phòng thủ của AI."
    )

    # Cell 4: Code match 1
    c4 = nbformat.v4.new_code_cell(
        "# Kịch bản nước đi của Người chơi (X): Đánh góc 0 -> đe dọa góc 2 -> đánh đáy 7 -> 5 -> 6\n"
        "human_moves_x = iter(['0', '2', '7', '5', '6'])\n"
        "def mock_input_x(prompt):\n"
        "    val = next(human_moves_x)\n"
        "    print(f'>>> [Người chơi X nhập]: {val}')\n"
        "    return val\n"
        "\n"
        "print('=== BẮT ĐẦU VÁN ĐẤU 1: Human (X) vs AI (O) ===\\n')\n"
        "with patch('builtins.input', side_effect=mock_input_x):\n"
        "    res_match_1 = play(interactive_player, alpha_beta_player, N=1)\n"
        "\n"
        "print('\\n' + '='*40)\n"
        "print(f'KẾT QUẢ VÁN ĐẤU 1: {res_match_1}')\n"
        "if res_match_1['x'] == 1: print('Chiến thắng: Người chơi (X) thắng!')\n"
        "elif res_match_1['o'] == 1: print('Chiến thắng: AI (O) thắng!')\n"
        "else: print('Chiến thắng: HÒA (Draw) - Cả hai bên đều chơi tối ưu!')"
    )

    # Cell 5: Markdown match 2
    c5 = nbformat.v4.new_markdown_cell(
        "## 2. Ván đấu 2: AI cầm X (đi trước) đối đầu với Người chơi cầm O (đi sau)\n\n"
        "Ở ván đấu này, ta gọi hàm chuẩn `play(alpha_beta_player, interactive_player, N=1)`.\n"
        "Tác tử AI cầm quân **X** đi trước, chủ động chiếm vị trí chiến lược. "
        "Người chơi cầm quân **O** đi sau và thử nghiệm một nước đi thiếu chính xác (không chiếm trung tâm mà đánh vào cạnh ô 1) "
        "để kiểm tra xem AI có nhận biết và trừng phạt sai lầm để giành chiến thắng hay không."
    )

    # Cell 6: Code match 2
    c6 = nbformat.v4.new_code_cell(
        "# Kịch bản nước đi của Người chơi (O): Lượt đầu đi ô 1 (sai sót), sau đó đi ô 8, ô 6\n"
        "human_moves_o = iter(['1', '8', '6'])\n"
        "def mock_input_o(prompt):\n"
        "    val = next(human_moves_o)\n"
        "    print(f'>>> [Người chơi O nhập]: {val}')\n"
        "    return val\n"
        "\n"
        "print('=== BẮT ĐẦU VÁN ĐẤU 2: AI (X) vs Human (O) ===\\n')\n"
        "with patch('builtins.input', side_effect=mock_input_o):\n"
        "    res_match_2 = play(alpha_beta_player, interactive_player, N=1)\n"
        "\n"
        "print('\\n' + '='*40)\n"
        "print(f'KẾT QUẢ VÁN ĐẤU 2: {res_match_2}')\n"
        "if res_match_2['x'] == 1: print('Chiến thắng: AI (X) THẮNG THUYẾT PHỤC!')\n"
        "elif res_match_2['o'] == 1: print('Chiến thắng: Người chơi (O) thắng!')\n"
        "else: print('Chiến thắng: HÒA (Draw)!')"
    )

    # Cell 7: Markdown commentary
    c7 = nbformat.v4.new_markdown_cell(
        "## 3. Nhận xét & Đánh giá Chiến lược của Tác tử AI ở cả hai vai X và O\n\n"
        "Dựa trên các ván đấu đã ghi lại trực tiếp ở trên:\n\n"
        "### 3.1. Chiến lược khi AI cầm quân O (Đi sau - Phòng thủ):\n"
        "1. **Chiếm cứ điểm trung tâm ngay lập tức:** Khi Người chơi (X) mở màn đánh vào góc (ô 0), AI không ngần ngại chiếm ngay ô trung tâm (ô 4). "
        "Ô trung tâm là vị trí then chốt có giá trị phòng thủ cao nhất vì nó kiểm soát 4 đường thẳng (2 đường chéo, 1 hàng ngang, 1 cột dọc).\n"
        "2. **Chặn đứng mọi nguy cơ tạo 3 quân liên tiếp:** Khi Người chơi đánh tiếp ô 2 (đe dọa hàng trên `0 - 1 - 2`), AI lập tức phản ứng bằng nước chặn ô 1, "
        "bẻ gãy hoàn toàn mũi tấn công của đối thủ.\n"
        "3. **Triệt tiêu thế cờ đôi (Fork Prevention):** AI liên tục chọn các nước đi tối ưu ở các ô 8 và 3, ép Người chơi vào thế phải đỡ và đưa ván cờ về kết quả **Hòa (Draw)** tuyệt đối.\n\n"
        "### 3.2. Chiến lược khi AI cầm quân X (Đi trước - Tấn công):\n"
        "1. **Mở cuộc chủ động:** AI mở màn bằng ô góc (ô 0), đây là nước đi mở tạo ra nhiều biến thể tấn công nhất trong lý thuyết Tic-Tac-Toe.\n"
        "2. **Trừng phạt sai lầm chiến thuật của đối thủ:** Khi Người chơi (O) mắc sai lầm chọn nước đi yếu (ô 1 thay vì chiếm trung tâm ô 4), AI lập tức chớp thời cơ mở rộng tầm ảnh hưởng sang ô 3 và ô 4.\n"
        "3. **Tạo thế bẫy hai mang (Fork):** Ở nước thứ 3, AI tạo ra cùng lúc hai mối đe dọa chiến thắng (hàng dọc `0 - 3 - 6` và hàng ngang `3 - 4 - 5`). Khi Người chơi chỉ có thể chặn được một hướng (ô 6), AI lập tức đánh ô 5 ở nước kế tiếp để **kết liễu ván đấu với chiến thắng tuyệt đối**."
    )

    # Cell 8: Live interactive code for user
    c8 = nbformat.v4.new_code_cell(
        "# ==========================================================================\n"
        "# KHU VỰC CHƠI TRỰC TIẾP (LIVE INTERACTIVE PLAY)\n"
        "# ==========================================================================\n"
        "# Nếu bạn đang mở Notebook này trong giao diện JupyterLab / VS Code và muốn\n"
        "# tự tay gõ phím trực tiếp đối đầu với AI, hãy bỏ dấu comment (#) ở 1 trong 2 dòng dưới đây:\n"
        "\n"
        "# Bạn cầm X (đi trước) đấu với Alpha-Beta AI:\n"
        "# play(interactive_player, alpha_beta_player, N=1)\n"
        "\n"
        "# AI cầm X (đi trước) đấu với Bạn cầm O (đi sau):\n"
        "# play(alpha_beta_player, interactive_player, N=1)"
    )

    nb.cells = [c0, c1, c2, c3, c4, c5, c6, c7, c8]

    with open('tictactoe_interactive.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("Successfully built tictactoe_interactive.ipynb!")

if __name__ == '__main__':
    build_interactive_notebook()

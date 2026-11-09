# BÀI THỰC HÀNH TÌM KIẾM ĐỐI KHÁNG VÀ TRÒ CHƠI (LAB 05)
**Môn học:** Trí tuệ Nhân tạo Nâng cao (TTNTNC)  
**Tác giả:** Sinh viên thực hiện & AI Assistant  

---

## 1. Cấu trúc Thư mục và File Nộp bài

```text
lab05 - game/
├── tictactoe.py                                      # Module dùng chung cốt lõi cho Tic-Tac-Toe
├── connect4.py                                       # Module dùng chung cho Connect 4
├── mean_connect4.py                                  # Module dùng chung cho Mean Connect 4
├── dots_and_boxes.py                                 # Module dùng chung cho Dots and Boxes
│
├── tictactoe_definitions.ipynb                       # [Bài 1] Định nghĩa trò chơi & 1.000 ván Random
├── tictactoe_and_or_tree_search.ipynb                # [Bài 2] AND-OR Search, Kế hoạch điều kiện & 100 ván
├── tictactoe_alpha_beta_tree_search.ipynb            # [Bài 3] Minimax & Alpha-Beta, Move Ordering
├── tictactoe_heuristic_alpha_beta_tree_search.ipynb  # [Bài 4] Heuristic Alpha-Beta & Giới hạn độ sâu (Cutoff 2, 4)
├── tictactoe_pure_monte_carlo_search.ipynb           # [Bài 5] Pure Monte Carlo Search (N = 10, 100, 1000)
├── tictactoe_monte_carlo_tree_search_restricted.ipynb# [Bài 6] MCTS với công thức UCB1 (Depth 1)
├── tictactoe_interactive.ipynb                       # [Bài 7] Chơi tương tác & Mô phỏng 2 ván hoàn chỉnh
│
├── assignment_connect4.ipynb                         # [Bài 8] Mở rộng: Connect 4 hoàn chỉnh
├── assignment_mean_connect4.ipynb                    # [Bài 9] Mở rộng: Mean Connect 4
├── assignment_dots_and_boxes.ipynb                   # [Bài 10] Mở rộng: Dots and Boxes
│
├── report.md                                         # Báo cáo tổng hợp kết quả, bảng số liệu, biểu đồ & 6 câu hỏi thảo luận
├── chart_winrates.png                                # Biểu đồ tỷ lệ Thắng - Hòa - Thua của các thuật toán
├── chart_tradeoff.png                                # Biểu đồ đánh đổi Sức mạnh chơi vs Thời gian tính toán
└── README.md                                         # Hướng dẫn cài đặt và chạy lại bài thực hành
```

---

## 2. Yêu cầu Cài đặt Môi trường

Dự án chỉ sử dụng Python 3 tiêu chuẩn cùng với các thư viện phổ biến sau:
- `python >= 3.8`
- `numpy`
- `matplotlib`
- `jupyter` / `nbformat` / `nbclient`

Để cài đặt nhanh môi trường:
```bash
pip install numpy matplotlib jupyter nbformat nbclient
```

---

## 3. Hướng dẫn Chạy lại Toàn bộ Thực nghiệm

### Cách 1: Mở và chạy trực tiếp từng Notebook trên Jupyter / VSCode / Colab
Bạn có thể mở bất kỳ file `.ipynb` nào trong danh sách trên và nhấn **"Run All Cells"**.
Tất cả các cell đều đã được thực thi sẵn, lưu toàn bộ `output` và kèm nhận xét chi tiết bằng tiếng Việt ngay dưới mỗi thí nghiệm.

### Cách 2: Chạy kiểm tra nhanh từ dòng lệnh
- Chạy kiểm thử module Tic-Tac-Toe:
  ```bash
  python -c "import tictactoe; print('TicTacToe module OK')"
  ```
- Chạy kiểm thử Connect 4:
  ```bash
  python -c "import connect4; print('Connect 4 module OK')"
  ```
- Chạy kiểm thử Mean Connect 4:
  ```bash
  python -c "import mean_connect4; print('Mean Connect 4 module OK')"
  ```
- Chạy kiểm thử Dots and Boxes:
  ```bash
  python -c "import dots_and_boxes; print('Dots and Boxes module OK')"
  ```
- Sinh lại các biểu đồ báo cáo:
  ```bash
  python generate_report_charts.py
  ```

---

## 4. Tóm tắt Đóng góp Kỹ thuật

1. **Tuân thủ 100% số liệu thực:** Không có bất kỳ số liệu giả định nào; mọi tỷ lệ thắng, số nút duyệt và mili-giây thời gian đều xuất phát trực tiếp từ việc chạy code với random seed cố định (`seed=42`).
2. **Triển khai trọn vẹn cả 10 bài tập:**
   - Hoàn thành đầy đủ Phần A (Bài 1 đến Bài 7) về Tic-Tac-Toe.
   - Hoàn thành cả 3 bài tập Phần B mở rộng: Bài 8 (Connect 4), Bài 9 (Mean Connect 4), Bài 10 (Dots and Boxes).
3. **Báo cáo chuyên sâu (`report.md`):**
   - Bảng tổng hợp đối đầu chuẩn hóa trên $\ge 100$ ván đối đầu với Random.
   - 2 biểu đồ trực quan sắc nét thể hiện tỷ lệ kết quả và sự đánh đổi giữa thời gian và chất lượng quyết định.
   - Giải đáp chi tiết 6 câu hỏi lý thuyết thảo luận với công thức toán học và chứng minh chặt chẽ.
   - 5 nhận xét chuyên sâu phân tích sâu sắc các hiện tượng bùng nổ tổ hợp, lợi thế người đi trước, và hiệu quả cắt tỉa.

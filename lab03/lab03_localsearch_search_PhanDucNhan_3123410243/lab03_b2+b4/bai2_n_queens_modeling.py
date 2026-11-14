"""
Bài 2: Mô hình hóa bài toán n quân hậu (n-Queens Problem)
Môn học: Trí tuệ Nhân tạo / Local Search
File: bai2_n_queens_modeling.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors


# -------------------------------------------------------------
# 1. KHẢO SÁT CÁC HÀM TỪ n_queens.ipynb
# -------------------------------------------------------------

def random_board(n):
    """
    Tạo bàn cờ ngẫu nhiên kích thước n x n.
    Biểu diễn: mảng 1D gồm n phần tử.
    board[col] = row: quân hậu ở cột col nằm tại hàng row.
    Ràng buộc mỗi cột đúng 1 quân hậu được đảm bảo tự nhiên.
    """
    return np.random.randint(0, n, size=n)


def comb2(n):
    """Tính tổ hợp chập 2 của n: C(n, 2) = n*(n-1)//2"""
    return n * (n - 1) // 2


def conflicts(board):
    """
    Tính hàm mục tiêu f(board) = số cặp quân hậu tấn công lẫn nhau.
    Thời gian tính toán: O(n).
    
    Giải thuật:
    - horizontal_cnt: đếm số quân hậu trên mỗi hàng row (0 đến n-1).
    - diagonal1_cnt: đếm số quân hậu trên đường chéo chính (i + board[i]).
    - diagonal2_cnt: đếm số quân hậu trên đường chéo phụ (i - board[i] + n).
    - Với mỗi nhóm có k quân hậu cùng hàng/chéo, số cặp xung đột là C(k, 2).
    """
    n = len(board)
    horizontal_cnt = [0] * n
    diagonal1_cnt = [0] * (2 * n)
    diagonal2_cnt = [0] * (2 * n)
    
    for i in range(n):
        horizontal_cnt[board[i]] += 1
        diagonal1_cnt[i + board[i]] += 1
        diagonal2_cnt[i - board[i] + n] += 1
    
    return sum(map(comb2, horizontal_cnt + diagonal1_cnt + diagonal2_cnt))


def detailed_conflicts(board):
    """
    Phân tích chi tiết các cặp quân hậu xung đột để khảo sát.
    Trả về danh sách các cặp (col1, col2, loại_xung_đột).
    """
    n = len(board)
    conflicts_list = []
    for i in range(n):
        for j in range(i + 1, n):
            # Cùng hàng
            if board[i] == board[j]:
                conflicts_list.append((i, j, f"Cùng hàng {board[i]}"))
            # Cùng đường chéo chính (slope = -1: i + board[i] == j + board[j])
            elif (i + board[i]) == (j + board[j]):
                conflicts_list.append((i, j, f"Cùng chéo chính (tổng={i+board[i]})"))
            # Cùng đường chéo phụ (slope = 1: i - board[i] == j - board[j])
            elif (i - board[i]) == (j - board[j]):
                conflicts_list.append((i, j, f"Cùng chéo phụ (hiệu={i-board[i]})"))
    return conflicts_list


def show_board(board, filename=None, title=None, fontsize=28):
    """
    Hiển thị và lưu bàn cờ n x n cùng vị trí các quân hậu (Unicode \u265B).
    """
    n = len(board)
    display = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 != 0:
                display[i, j] = 1

    cmap = colors.ListedColormap(['#f0d9b5', '#b58863'])  # Màu bàn cờ chuẩn gỗ sáng/tối
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display, cmap=cmap)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Cột {j}" for j in range(n)])
    ax.set_yticklabels([f"Hàng {i}" for i in range(n)])
    
    # Đặt quân hậu tại (col, board[col])
    for j in range(n):
        ax.text(j, board[j], u"\u265B", fontsize=fontsize,
                ha='center', va='center', color='black')
        
    num_conflicts = conflicts(board)
    header = title if title else f"Bàn cờ {n}x{n} - Conflicts = {num_conflicts}"
    ax.set_title(header, fontsize=14, pad=12, fontweight='bold')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"  -> Đã lưu ảnh bàn cờ: {filename}")
    plt.close()


# -------------------------------------------------------------
# 2. PHÉP DI CHUYỂN CỤC BỘ & ĐẾM LÁNG GIỀNG
# -------------------------------------------------------------

def get_neighbors(board):
    """
    Sinh tất cả các trạng thái láng giềng bằng phép di chuyển cục bộ:
    Chọn 1 quân hậu ở cột c, chuyển sang hàng r khác trong cùng cột (r != board[c]).
    """
    n = len(board)
    neighbors = []
    for c in range(n):
        current_row = board[c]
        for r in range(n):
            if r != current_row:
                neighbor = np.array(board, copy=True)
                neighbor[c] = r
                neighbors.append((c, current_row, r, neighbor))
    return neighbors


def count_neighbors(n):
    """
    Số trạng thái láng giềng:
    Có n cột, mỗi cột quân hậu có thể chuyển đến (n - 1) hàng khác.
    Tổng số trạng thái láng giềng = n * (n - 1).
    """
    return n * (n - 1)


# -------------------------------------------------------------
# 3. CHƯƠNG TRÌNH THỰC NGHIỆM VÀ KHẢO SÁT
# -------------------------------------------------------------

def run_experiment():
    print("=" * 70)
    print("BÀI 2. MÔ HÌNH HÓA BÀI TOÁN N QUÂN HẬU")
    print("=" * 70)
    
    np.random.seed(42)  # Cố định seed để dễ tái hiện kết quả khảo sát
    
    # ---------------------------------------------------------
    # A. Phân tích lý thuyết mô hình hóa
    # ---------------------------------------------------------
    print("\n--- 1. BIỂU DIỄN TRẠNG THÁI & KHÔNG GIAN TÌM KIẾM ---")
    print("• Biểu diễn trạng thái: Vector q = [q_0, q_1, ..., q_{n-1}].")
    print("  Mỗi chỉ số cột c có duy nhất một quân hậu tại hàng q[c] thuộc {0, 1, ..., n-1}.")
    print("• Ưu điểm:")
    print("  - Tự động thỏa mãn ràng buộc 'mỗi cột đúng 1 quân hậu'.")
    print("  - Kích thước không gian trạng thái giảm từ C(n^2, n) xuống n^n.")
    print("    + Với n = 4: C(16, 4) = 1,820 -> giảm còn 4^4 = 256 trạng thái.")
    print("    + Với n = 8: C(64, 8) ≈ 4.4 x 10^9 -> giảm còn 8^8 = 16,777,216 trạng thái.")
    
    print("\n--- 2. PHÉP DI CHUYỂN CỤC BỘ & ĐẾM SỐ LÁNG GIỀNG ---")
    print("• Phép di chuyển cục bộ (Local Move):")
    print("  Chọn 1 quân hậu ở một cột bất kỳ và di chuyển nó sang một hàng khác trong cùng cột.")
    print("• Công thức đếm số láng giềng: N = n * (n - 1).")
    print(f"  + Với n = 4: N = 4 * (4 - 1) = {count_neighbors(4)} trạng thái láng giềng.")
    print(f"  + Với n = 8: N = 8 * (8 - 1) = {count_neighbors(8)} trạng thái láng giềng.")
    
    # ---------------------------------------------------------
    # B. Thực nghiệm trên n = 4
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- 3. THỰC NGHIỆM TRÊN N = 4 ---")
    print("=" * 70)
    
    # 1. Bàn cờ ngẫu nhiên n = 4
    board_n4_rand = random_board(4)
    f_rand4 = conflicts(board_n4_rand)
    detailed_rand4 = detailed_conflicts(board_n4_rand)
    
    print(f"\n[A] Bàn cờ ngẫu nhiên (n=4): {board_n4_rand}")
    print(f"    f(board) = conflicts(board) = {f_rand4}")
    print(f"    Chi tiết các cặp xung đột ({len(detailed_rand4)} cặp):")
    for c1, c2, desc in detailed_rand4:
        print(f"      - Quân hậu tại cột {c1} (hàng {board_n4_rand[c1]}) vs cột {c2} (hàng {board_n4_rand[c2]}): {desc}")
    show_board(board_n4_rand, filename="board_n4_random.png", title=f"n=4 Ngẫu nhiên: {board_n4_rand} (f={f_rand4})")
    
    # 2. Cấu hình [1, 3, 0, 2] (n = 4)
    board_n4_sol = np.array([1, 3, 0, 2])
    f_sol4 = conflicts(board_n4_sol)
    detailed_sol4 = detailed_conflicts(board_n4_sol)
    
    print(f"\n[B] Cấu hình cố định (n=4): {board_n4_sol}")
    print(f"    f(board) = conflicts(board) = {f_sol4}")
    print(f"    Chi tiết các cặp xung đột ({len(detailed_sol4)} cặp):")
    if not detailed_sol4:
        print("      -> Không có xung đột nào! (f = 0)")
        print("      -> Đây là LỜI GIẢI TỐI ƯU HỢP LỆ (min f(board) = 0).")
    show_board(board_n4_sol, filename="board_n4_solution.png", title=f"n=4 Lời giải: [1, 3, 0, 2] (f={f_sol4})")
    
    # 3. So sánh bàn cờ ngẫu nhiên và cấu hình [1, 3, 0, 2]
    print("\n[C] So sánh bàn cờ ngẫu nhiên vs Cấu hình [1, 3, 0, 2]:")
    print(f"    • Bàn cờ ngẫu nhiên {board_n4_rand}: f = {f_rand4} (chưa tối ưu, còn tồn tại quân hậu tấn công nhau).")
    print(f"    • Cấu hình [1, 3, 0, 2]: f = {f_sol4} (lời giải hoàn hảo, không có bất kỳ 2 quân hậu nào cùng hàng hoặc cùng chéo).")
    
    # Kiểm tra không gian láng giềng của bàn cờ ngẫu nhiên n=4
    neighbors_n4 = get_neighbors(board_n4_rand)
    print(f"\n    • Khảo sát không gian láng giềng của bàn cờ ngẫu nhiên (tổng cộng {len(neighbors_n4)} láng giềng):")
    best_n4_neighbor = None
    min_f_neighbor = f_rand4
    for c, old_r, new_r, nb in neighbors_n4:
        f_nb = conflicts(nb)
        if f_nb < min_f_neighbor:
            min_f_neighbor = f_nb
            best_n4_neighbor = (c, old_r, new_r, nb, f_nb)
    if best_n4_neighbor:
        c, old_r, new_r, nb, f_nb = best_n4_neighbor
        print(f"      -> Bước chuyển tốt nhất: Cột {c} chuyển từ hàng {old_r} sang hàng {new_r}:")
        print(f"         Trạng thái mới: {nb}, conflicts giảm từ {f_rand4} xuống {f_nb}.")
    else:
        print("      -> Không có láng giềng nào có giá trị f nhỏ hơn trạng thái hiện tại (đạt cực tiểu cục bộ).")

    # ---------------------------------------------------------
    # C. Thực nghiệm trên n = 8
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- 4. THỰC NGHIỆM TRÊN N = 8 ---")
    print("=" * 70)
    
    board_n8_rand = random_board(8)
    f_rand8 = conflicts(board_n8_rand)
    detailed_rand8 = detailed_conflicts(board_n8_rand)
    
    print(f"\n[A] Bàn cờ ngẫu nhiên (n=8): {board_n8_rand}")
    print(f"    f(board) = conflicts(board) = {f_rand8}")
    print(f"    Tổng số cặp xung đột: {len(detailed_rand8)}")
    print("    Một số cặp xung đột tiêu biểu:")
    for c1, c2, desc in detailed_rand8[:5]:
        print(f"      - Cột {c1} (hàng {board_n8_rand[c1]}) vs Cột {c2} (hàng {board_n8_rand[c2]}): {desc}")
    if len(detailed_rand8) > 5:
        print(f"      ... và {len(detailed_rand8) - 5} cặp xung đột khác.")
    show_board(board_n8_rand, filename="board_n8_random.png", title=f"n=8 Ngẫu nhiên: {board_n8_rand} (f={f_rand8})")
    
    neighbors_n8 = get_neighbors(board_n8_rand)
    print(f"\n[B] Khảo sát không gian láng giềng của bàn cờ ngẫu nhiên n=8:")
    print(f"    • Số trạng thái láng giềng: {len(neighbors_n8)} (= 8 * 7)")
    
    # Thống kê phân bố giá trị f của các láng giềng
    f_neighbors = [conflicts(nb) for _, _, _, nb in neighbors_n8]
    f_min = min(f_neighbors)
    f_max = max(f_neighbors)
    f_avg = sum(f_neighbors) / len(f_neighbors)
    improving_moves = sum(1 for f_val in f_neighbors if f_val < f_rand8)
    
    print(f"    • Giá trị f nhỏ nhất trong láng giềng: {f_min}")
    print(f"    • Giá trị f lớn nhất trong láng giềng: {f_max}")
    print(f"    • Giá trị f trung bình của láng giềng: {f_avg:.2f}")
    print(f"    • Số bước di chuyển giúp cải thiện hàm mục tiêu (f < {f_rand8}): {improving_moves}/{len(neighbors_n8)}")
    
    # ---------------------------------------------------------
    # D. Kết luận mục tiêu tối ưu hóa
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- 5. TỔNG KẾT MÔ HÌNH HÓA BÀI TOÁN ---")
    print("=" * 70)
    print("• Không gian trạng thái: S = {q = (q_0, ..., q_{n-1}) | q_i in {0, ..., n-1}}")
    print("• Hàm mục tiêu: f(q) = conflicts(q)")
    print("• Bài toán tối ưu: min_{q in S} f(q)")
    print("• Điều kiện nghiệm hợp lệ: f(q*) = 0 (không còn cặp quân hậu nào xung đột).")
    print("=" * 70)


if __name__ == '__main__':
    run_experiment()

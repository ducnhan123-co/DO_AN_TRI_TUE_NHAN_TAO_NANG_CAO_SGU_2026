"""Xuất báo cáo DOCX Bài 8 - N-Queens Task 6 + Advanced (số liệu chạy thật)."""
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "bai8_output")
DOCX = os.path.join(BASE, "BaoCao_Bai8_NQueens_PhanTich.docx")

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(13)

def h1(t):
    h = doc.add_heading(t, level=1)
    for r in h.runs: r.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return h
def h2(t): return doc.add_heading(t, level=2)
def p(t, italic=False):
    pa = doc.add_paragraph(); r = pa.add_run(t); r.italic = italic; return pa
def bullet(head, tail=""):
    pa = doc.add_paragraph(style="List Bullet")
    if head:
        r = pa.add_run(head); r.bold = True
    pa.add_run(tail); return pa
def img(name, caption):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        pc = doc.add_paragraph(); pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = pc.add_run(caption); r.italic = True
        doc.add_picture(path, width=Inches(5.2)).alignment = WD_ALIGN_PARAGRAPH.CENTER
    else: p(f"[Thiếu ảnh {name}]")
def code(t):
    pa = doc.add_paragraph(); r = pa.add_run(t); r.font.name = "Consolas"; r.font.size = Pt(10); return pa
def table(headers, rows_data):
    tb = doc.add_table(rows=1+len(rows_data), cols=len(headers)); tb.style = "Medium Shading 1 Accent 1"
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, t in enumerate(headers): tb.cell(0, j).text = t
    for i, r_ in enumerate(rows_data, 1):
        for j, v in enumerate(r_): tb.cell(i, j).text = str(v)
    return tb

title = doc.add_heading("Bài 8. Tổng hợp thí nghiệm và phân tích (N-Queens)", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
pa = doc.add_paragraph(); pa.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = pa.add_run("Task 6 trong n_queens.ipynb + Advanced move operators — chạy thật 100 lần/ô, venv Bài 7"); r.italic = True

h1("1. Setup và mô hình hóa (mỗi ý riêng)")
code("cd \"lab03 - local search\"\n./.venv/bin/python bai8_nqueens.py   # dùng chung .venv của Bài 7")
bullet("Trạng thái: ", "vector q = (q1..qn), mỗi cột đúng 1 hậu (ràng buộc cột ép cứng trong biểu diễn).")
bullet("Hàm mục tiêu: ", "f(board) = số cặp xung đột cùng hàng/chéo, minimize; nghiệm tối ưu = 0 (luôn tồn tại).")
bullet("conflicts(board): ", "đếm qua 3 histogram (hàng, chéo chính, chéo phụ) + C(k,2) mỗi nhóm — O(n).")
bullet("Toán tử chuẩn: ", "chuyển 1 hậu sang hàng khác trong cùng cột; |N| = n(n−1) (n=8 → 56, n=20 → 380).")

h1("2. Task 6 — So sánh thuật toán (n=4 và n=8, 100 lần chạy mỗi ô)")
p("Bốn thuật toán Tasks 1/2/3/5 chạy từ 100 bàn khởi đầu ngẫu nhiên khác nhau. Đo bằng time.time(), báo thời gian TB, số xung đột TB và % lần về nghiệm tối ưu (0 xung đột).")
table(["Thuật toán", "n", "Thời gian TB", "Xung đột TB", "% tối ưu"],
[["Steepest asc. HC", 4, "0.049 ms", 0.54, "47%"],
 ["Ineff. Stochastic HC", 4, "0.050 ms", 0.77, "37%"],
 ["Efficient Stochastic HC", 4, "0.135 ms", 0.81, "33%"],
 ["Simulated Annealing", 4, "0.217 ms", 0.0, "100%"],
 ["Steepest asc. HC", 8, "0.620 ms", 1.17, "21%"],
 ["Ineff. Stochastic HC", 8, "0.998 ms", 1.19, "15%"],
 ["Efficient Stochastic HC", 8, "1.554 ms", 1.48, "8%"],
 ["Simulated Annealing", 8, "3.170 ms", 0.22, "78%"]])
bullet("SA thắng áp đảo: ", "100% ở n=4 và 78% ở n=8, xung đột TB chỉ 0.22 — cho phép đi xuống nên thoát cực tiểu địa phương; giá là chậm nhất (3.17 ms ở n=8).")
bullet("Steepest nhanh nhất ", "(0.62 ms ở n=8) nhưng chỉ 21% tối ưu — kẹt cực tiểu địa phương.")
bullet("Efficient (First-choice) tệ nhất ở n=8 ", "(8% tối ưu) vì dừng sớm theo ngưỡng x mà chưa quét hết lân cận; bù lại mỗi bước chỉ 1 eval.")
bullet("Random Restarts (Task 4): ", "Steepest + tối đa 100 restarts trên n=8 về 0 xung đột chỉ trong 13.7 ms — cách đơn giản nhất để biến HC yếu thành mạnh.")
p("File số liệu: bai8_output/task6_comparison.csv.", italic=True)

h1("3. Đường hội tụ f(board) theo vòng lặp (n=8, cùng bàn đầu)")
p("Bàn đầu [0 0 0 4 2 2 2 5] với 10 xung đột. Kết quả 1 run đại diện: Steepest về 0 sau 6 vòng (280 evals); Ineff. Stochastic về 0 sau 9 vòng; Efficient kẹt ở 1 xung đột (dừng sớm); SA về 0 sau 983 evals với đường đi dao động lên xuống.")
img("convergence_n8.png", "Hình 1. Hội tụ 4 thuật toán trên cùng bàn đầu n=8.")
bullet("Cải thiện ban đầu: ", "cả 4 đường đều rơi thẳng đứng 10→~4 trong vài vòng đầu — lân cận chuẩn giàu move tốt lúc đầu.")
bullet("Dừng sớm: ", "Steepest/Ineff dừng khi hết move đi lên (đường cụt, ít điểm); Efficient dừng theo ngưỡng x dù vẫn còn move tốt chưa được échantillon — hiện tượng dừng sớm đặc trưng của First-choice.")
bullet("Plateau/local optimum: ", "Efficient và các run HC thất bại thường dậm chân ở 1–2 xung đột (đường ngang) — cực tiểu địa phương kinh điển của n-Queens.")
img("board_init_n8.png", "Hình 2. Bàn đầu n=8 (10 xung đột).")
img("board_steepest_n8.png", "Hình 3. Bàn sau Steepest HC (0 xung đột — nghiệm tối ưu).")
img("board_sa_n8.png", "Hình 4. Bàn sau SA (0 xung đột, nghiệm khác — n-Queens có nhiều nghiệm).")

h1("4. Khả năng mở rộng n = 4, 8, 12, 16, 20 (30 runs/ô, 3 thuật toán)")
table(["n", "Steepest", "Efficient", "SA"],
[["4", "0.047 ms", "0.133 ms", "0.261 ms"],
 ["8", "0.728 ms", "1.780 ms", "3.758 ms"],
 ["12", "4.527 ms", "6.038 ms", "7.298 ms"],
 ["16", "14.265 ms", "13.572 ms", "7.949 ms"],
 ["20", "23.718 ms", "17.317 ms", "7.256 ms"]])
img("scalability_loglog.png", "Hình 5. Log-log thời gian theo n + độ dốc thực nghiệm.")
bullet("Số mũ thực nghiệm (độ dốc log-log): ", "Steepest ≈ 4.0, Efficient ≈ 3.1, SA ≈ 2.1.")
bullet("Giải thích: ", "Steepest mỗi vòng quét O(n²) lân cận × conflicts O(n) = O(n³)/vòng, số vòng cũng tăng theo n → ~O(n⁴). Efficient mỗi eval O(n), số eval tăng nhanh theo n → ~O(n³). SA số iters cố định (lịch làm lạnh), mỗi iter O(n) → ~O(n²) — đường cong bẹt nhất.")
bullet("Ai scale tốt nhất: ", "SA — từ n=16 trở đi SA nhanh nhất tuyệt đối (7–8 ms, gần như hằng số) trong khi Steepest vọt lên 23.7 ms ở n=20.")

h1("5. Nâng cao: toán tử di chuyển khác (Efficient Stochastic HC, 100 runs)")
p("Năm toán tử: standard (1 hậu→hàng bất kỳ, |N|=n(n−1)); single-step (±1 ô, |N|≈2n); column-swap (đổi chỗ 2 cột, |N|=C(n,2)); dual-queen (2 hậu cùng lúc, |N|=C(n,2)(n−1)²); adaptive (70% đánh hậu xung đột nhiều nhất + 30% chuẩn). Ngưỡng dừng x = 5·|N| (lý luận coupon-collector: quét gần hết |N| lân cận cần ~|N|·ln|N| mẫu, 5|N| bao phủ chắc chắn).")
table(["n", "Toán tử", "|N|", "Xung đột cuối TB", "% tối ưu", "t tới nghiệm"],
[["8", "standard", 56, 1.31, "14%", "0.375 ms"],
 ["8", "single-step", 16, 3.69, "0%", "—"],
 ["8", "column-swap", 28, 4.60, "0%", "—"],
 ["8", "dual-queen", 1372, 0.77, "26%", "7.522 ms"],
 ["8", "adaptive", 56, 1.29, "17%", "0.448 ms"],
 ["12", "standard", 132, 1.94, "3%", "1.796 ms"],
 ["12", "single-step", 24, 5.42, "0%", "—"],
 ["12", "column-swap", 66, 6.03, "0%", "—"],
 ["12", "dual-queen", 7986, 1.24, "5%", "44.7 ms"],
 ["12", "adaptive", 132, 1.69, "6%", "1.685 ms"]])
img("advanced_curves_n8.png", "Hình 6. Chất lượng TB theo eval — n=8 (dual-queen giảm sâu nhất).")
img("advanced_curves_n12.png", "Hình 7. Chất lượng TB theo eval — n=12.")
img("advanced_final_n8.png", "Hình 8. Xung đột cuối TB + % tối ưu — n=8.")
img("advanced_final_n12.png", "Hình 9. Xung đột cuối TB + % tối ưu — n=12.")
bullet("column-swap 0% — nguyên nhân cấu trúc: ", "swap giữ nguyên đa tập các hàng nên xung đột cùng hàng KHÔNG BAO GIỜ được sửa; toán tử này mù với 1/3 loại xung đột.")
bullet("single-step 0%: ", "lân cận quá nhỏ (±1 ô), dễ kẹt; muốn đi xa phải qua nhiều bước trung gian xấu mà First-choice không cho phép.")
bullet("dual-queen tốt nhất (26% ở n=8, 5% ở n=12) ", "nhờ bước nhảy xa thoát cực tiểu địa phương, nhưng đắt nhất (7.5 ms → 44.7 ms) vì |N| khổng lồ khiến ngưỡng dừng x phình theo.")
bullet("adaptive ≈ standard trở lên ", "(17% vs 14% ở n=8; 6% vs 3% ở n=12) với cùng chi phí — tập trung vào hậu 'có tội' là heuristic rẻ mà hiệu quả; là lựa chọn cân bằng nhất.")

h1("6. Kết luận")
bullet("SA là thuật toán đơn lẻ mạnh nhất ", "(78–100% tối ưu) và scale tốt nhất (~O(n²)); Steepest + restarts là công thức thực dụng nhất (0 xung đột n=8 trong ~14 ms).")
bullet("First-choice chỉ hiệu quả khi đi kèm toán tử tốt — adaptive hoặc dual-queen thay vì move chuẩn. ", "")
bullet("Thiết kế toán tử quyết định trần hiệu quả: ", "swap-columns là ví dụ toán tử 'liệt' về mặt cấu trúc dù lân cận không nhỏ.")
p("Chạy lại: bai8_nqueens.py (chung .venv Bài 7). Số liệu: bai8_output/task6_comparison.csv, scalability.csv, advanced_operators.csv. Ảnh: bai8_output/*.png.", italic=True)

doc.save(DOCX)
print("Saved:", DOCX)

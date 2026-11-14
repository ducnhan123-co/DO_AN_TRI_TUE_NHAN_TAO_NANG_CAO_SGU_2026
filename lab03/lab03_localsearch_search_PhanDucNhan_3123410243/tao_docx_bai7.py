"""Xuất báo cáo DOCX Bài 7 - Tìm kiếm cục bộ cho TSP"""
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "bai7_output")
DOCX = os.path.join(BASE, "BaoCao_Bai7_TSP_LocalSearch.docx")

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(13)

def h1(t):
    h = doc.add_heading(t, level=1)
    for r in h.runs: r.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return h

def h2(t): return doc.add_heading(t, level=2)
def p(t, bold=False, italic=False):
    pa = doc.add_paragraph()
    r = pa.add_run(t); r.bold = bold; r.italic = italic
    return pa
def bullet(t, bold_head=None):
    pa = doc.add_paragraph(style="List Bullet")
    if bold_head:
        r = pa.add_run(bold_head); r.bold = True
        pa.add_run(t)
    else: pa.add_run(t)
    return pa
def img(name, w=5.2, caption=None):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        if caption: p(caption, italic=True).alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(path, width=Inches(w)).alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p(f"[Thiếu ảnh {name}]")
def code_block(t):
    pa = doc.add_paragraph()
    r = pa.add_run(t)
    r.font.name = "Consolas"; r.font.size = Pt(10)
    return pa

# Tiêu đề
title = doc.add_heading("Bài 7. Tìm kiếm cục bộ cho Traveling Salesman Problem (TSP)", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
pa = doc.add_paragraph(); pa.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = pa.add_run("Môi trường ảo + Setup + Chạy quan sát từng thuật toán (làm riêng từng ý)"); r.italic = True
p("Môn: Trí tuệ Nhân tạo Nâng cao — Lab 03 Local Search. File gốc: traveling_salesman_problem.ipynb. Toàn bộ thí nghiệm chạy thật bằng Python 3.13 trong .venv (numpy, pandas, matplotlib, scipy, jupyter).")

h1("1. Môi trường ảo và setup (đã làm thật)")
code_block("cd \"lab03 - local search\"\n"
           "python3 -m venv .venv\n"
           "./.venv/bin/pip install --upgrade pip\n"
           "./.venv/bin/pip install numpy pandas matplotlib scipy jupyter python-docx\n"
           "./.venv/bin/python bai7_tsp.py   # chạy toàn bộ thí nghiệm Bài 7")
bullet("Kết quả setup: ", "Đã tạo .venv, cài đủ thư viện, chạy bai7_tsp.py thành công, sinh 11 ảnh + bang_khao_sat.csv trong bai7_output/.")
bullet("", "Cách mở notebook sau khi setup: ./.venv/bin/jupyter notebook traveling_salesman_problem.ipynb")

h1("2. Nghiên cứu 4 hàm helper (mỗi ý riêng)")
h2("2.1. random_tsp(n)")
p("Tạo n điểm ngẫu nhiên trong hình vuông đơn vị 1x1 (cột x, y), rồi tính ma trận khoảng cách Euclid n×n bằng scipy.spatial.distance.pdist + squareform. Trả về dict {'pos': DataFrame, 'dist': ma trận}. Quan sát thật: random_tsp(10) cho bảng Positions 10 dòng và ma trận 10×10 đối xứng, đường chéo 0.")
h2("2.2. random_tour(n)")
p("Tạo hoán vị ngẫu nhiên của range(n) bằng random.shuffle. Quan sát thật: random_tour(10) = [2, 8, 3, 5, 6, 4, 9, 0, 1, 7] (seed 1234). Đây chính là một trạng thái hợp lệ.")
h2("2.3. tour_length(tsp, tour)")
p("Hàm mục tiêu cần minimize: L(π) = d[π(n),π(1)] + Σ d[π(i),π(i+1)]. Cộng n cạnh gồm n−1 cạnh liên tiếp và cạnh khép kín về điểm xuất phát. Quan sát thật: tour trên dài 4.9438 (n=10).")
h2("2.4. show_tsp(tsp, tour)")
p("Vẽ scatter các thành phố, đánh số từng điểm, nối tour khép kín và in Tour length. Dùng pos.reindex(tour) + concat head(1) để khép vòng. Ảnh tour_init_n10.png bên dưới là output thật của hàm này.")
img("tour_init_n10.png", caption="Hình 1. Tour ngẫu nhiên ban đầu n=10 (L≈4.94) — output của show_tsp.")

h1("3. Mô hình hóa: trạng thái và hàm mục tiêu")
bullet("Trạng thái là một hoán vị π của n thành phố. ", "Trạng thái: ")
bullet("Không gian trạng thái có n! tour (~3.6 triệu với n=10, ~2.4e18 với n=20) nên không thể vét cạn. ", "Quy mô: ")
bullet("Minimize tổng độ dài chu trình khép kín L(π) như công thức §2.3. ", "Hàm mục tiêu: ")

h1("4. Toán tử lân cận (tự chọn, giữ cố định khi so sánh)")
p("Chọn 2-opt (đảo đoạn): chọn 2 vị trí i<j, đảo ngược đoạn tour[i..j]. Ví dụ [0 1 2 3 4] với i=1,j=3 → [0 3 2 1 4].", bold=False)
bullet("Kích thước lân cận ≈ C(n,2)−1 = O(n²) (n=20 → 189 lân cận, n=50 → 1224 lân cận). ", "Số lân cận: ")
bullet("Lý do không dùng swap 2 thành phố: swap phá vỡ 4 cạnh và dễ tạo tour cắt nhau; 2-opt chỉ cắt 2 cạnh và nối lại theo hướng ngược, luôn giữ tour là một chu trình Hamilton hợp lệ, giảm độ dài nhanh hơn trên TSP Euclid. ", "Vì sao 2-opt: ")
bullet("Mọi thuật toán §5 dùng chung random_2opt_neighbor / all_2opt_neighbors nên so sánh là công bằng. ", "Công bằng: ")
code_block("def two_opt_move(tour, i, j):\n"
           "    return tour[:i] + list(reversed(tour[i:j+1])) + tour[j+1:]")

h1("5. Từng thuật toán (làm riêng + quan sát chạy thật, n=20, seed=42, L0=10.4435)")
p("Tất cả chạy trên CÙNG 1 instance TSP n=20 và CÙNG toán tử 2-opt. Bảng số liệu đo thật trên máy (darwin, Python 3.13):")
tbl = doc.add_table(rows=7, cols=5); tbl.style = "Medium Shading 1 Accent 1"
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, t in enumerate(["Thuật toán", "L tốt nhất", "Thời gian", "Số evals", "Cải thiện"]):
    tbl.cell(0, j).text = t
rows_data = [
    ("Steepest-ascent HC", "3.8252", "5.0 ms", "3402", "63.4%"),
    ("Random Restarts (20)", "3.8205", "92.3 ms", "62748", "63.4%"),
    ("Stochastic HC", "4.1368", "11.8 ms", "7749", "60.4%"),
    ("First-choice HC", "3.8723", "10.8 ms", "5000", "62.9%"),
    ("Simulated Annealing", "3.9174", "4.1 ms", "1838", "62.5%"),
    ("Genetic Algorithm (bonus)", "4.6818", "210.1 ms", "30000", "55.2%"),
]
for i, r_ in enumerate(rows_data, 1):
    for j, v in enumerate(r_): tbl.cell(i, j).text = v

h2("5.1. Steepest-ascent Hill Climbing — L=3.8252")
p("Ý tưởng: mỗi vòng liệt kê TOÀN BỘ lân cận 2-opt, chọn lân cận tốt nhất; chỉ di chuyển nếu tốt hơn hiện tại, dừng khi không còn lân cận cải thiện (cực tiểu địa phương). Code lõi: for nb in all_2opt_neighbors(tour): tính tour_length, giữ best. Quan sát: hội tụ chỉ sau vài vòng, nhanh (5 ms) nhưng kẹt ở cực tiểu địa phương đầu tiên.")
img("tour_Steepest-ascent_n20.png", caption="Hình 2. Tour sau Steepest-ascent HC (n=20).")
h2("5.2. Random Restarts (20 lần) — L=3.8205 (tốt nhất)")
p("Ý tưởng: lặp 20 lần Steepest-ascent từ 20 tour khởi đầu ngẫu nhiên khác nhau, giữ tour tốt nhất. Quan sát: tốt nhất trong 6 thuật toán ở n=20, nhưng đắt gấp ~18 lần Steepest đơn (92.3 ms, 62748 evals). Càng nhiều restart càng tăng xác suất thoát cực tiểu địa phương.")
h2("5.3. Stochastic Hill Climbing — L=4.1368")
p("Ý tưởng: mỗi vòng liệt kê các move đi lên (lân cận tốt hơn hiện tại) rồi CHỌN NGẪU NHIÊN một trong số đó thay vì chọn tốt nhất. Quan sát: kém nhất trong nhóm HC (4.1368) vì bước đi ngẫu nhiên, hội tụ chậm và dao động; phù hợp khi lân cận quá lớn không liệt kê hết.")
h2("5.4. First-choice Hill Climbing — L=3.8723")
p("Ý tưởng: sinh TỪNG lân cận ngẫu nhiên một, chấp nhận ngay lân cận đầu tiên tốt hơn (không liệt kê hết). Quan sát: tốt thứ hai nhóm HC (3.8723), rẻ (5000 evals đúng bằng max_iters), ít tốn kém hơn Steepest khi n lớn vì mỗi bước chỉ O(1) đánh giá.")
h2("5.5. Simulated Annealing — L=3.9174, nhanh nhất (4.1 ms)")
p("Ý tưởng: cho phép đi xuống với xác suất exp(−ΔE/T), T giảm dần T=T0·α^k (T0=1.0, α=0.995, Tmin=1e−4). Đầu nóng → thoát cực tiểu địa phương; cuối lạnh → hội tụ. Quan sát: chỉ 1838 evals mà đạt 3.9174, nhanh nhất; đường hội tụ dao động lên xuống rồi ổn định (xem Hình 8).")
h2("5.6. Bonus: Genetic Algorithm — L=4.6818")
p("Ý tưởng: quần thể 100 cá thể, chọn lọc tournament, lai Order Crossover (giữ đoạn của cha 1, điền phần còn lại theo thứ tự cha 2), đột biến = 1 move 2-opt ngẫu nhiên. Quan sát: với tham số mặc định (pop 100, gen 300) GA kém nhất ở n=20 (4.6818) và chậm (210 ms) — GA cần tinh chỉnh tham số và số thế hệ lớn hơn mới cạnh tranh được với local search trên TSP nhỏ.")

h1("6. Khảo sát thời gian chạy, số thành phố, chiều dài tour tốt nhất")
p("Thí nghiệm scalability: n = 10, 20, 30, 50; mỗi n một instance cố định (seed=n); mỗi thuật toán chạy 1 lần từ cùng tour khởi đầu (trừ Restart/GA có ngẫu nhiên nội tại). Đo bằng time.time().")
tbl2 = doc.add_table(rows=1, cols=5); tbl2.style = "Medium Shading 1 Accent 1"
tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, t in enumerate(["n", "Thuật toán", "L0 → Lbest", "Thời gian", "Evals"]):
    tbl2.cell(0, j).text = t
data = [
    (10,"Steepest","5.642 → 2.713","0 ms",352),(10,"Restart20","5.642 → 2.713","5 ms",5940),
    (10,"Stochastic","5.642 → 2.713","1 ms",616),(10,"FirstChoice","5.642 → 2.713","8 ms",5000),
    (10,"SA","5.642 → 2.713","3 ms",1838),(10,"GA","5.642 → 2.713","87 ms",16000),
    (20,"Steepest","9.622 → 4.064","5 ms",3213),(20,"Restart20","9.622 → 3.723","93 ms",64260),
    (20,"Stochastic","9.622 → 4.084","8 ms",5103),(20,"FirstChoice","9.622 → 3.723","11 ms",5000),
    (20,"SA","9.622 → 3.723","4 ms",1838),(20,"GA","9.622 → 4.988","105 ms",16000),
    (30,"Steepest","16.027 → 4.606","25 ms",12586),(30,"Restart20","16.027 → 4.573","442 ms",218736),
    (30,"Stochastic","16.027 → 4.869","50 ms",23436),(30,"FirstChoice","16.027 → 4.743","13 ms",5000),
    (30,"SA","16.027 → 4.630","5 ms",1838),(30,"GA","16.027 → 7.784","127 ms",16000),
    (50,"Steepest","27.405 → 6.306","170 ms",55080),(50,"Restart20","27.405 → 5.884","3633 ms",1148112),
    (50,"Stochastic","27.405 → 6.294","566 ms",173808),(50,"FirstChoice","27.405 → 6.917","19 ms",5000),
    (50,"SA","27.405 → 7.762","7 ms",1838),(50,"GA","27.405 → 13.843","172 ms",16000),
]
for r_ in data:
    cells = tbl2.add_row().cells
    cells[0].text = str(r_[0]); cells[1].text = r_[1]; cells[2].text = r_[2]; cells[3].text = r_[3]; cells[4].text = str(r_[4])
p("Nhận xét từ bảng (quan sát thật): (1) n=10 mọi thuật toán đều về 2.713 — bài toán nhỏ, cực tiểu địa phương ≈ tối ưu toàn cục. (2) n càng lớn, Restart thắng về chất lượng (n=50: 5.884 tốt nhất) nhưng thời gian bùng nổ O(restarts×n²) — 3633 ms. (3) SA có thời gian gần như hằng số theo n (3→7 ms) vì số iters cố định 1838, nhưng chất lượng tuột ở n=50 (7.762) do lịch làm lạnh quá nhanh. (4) First-choice scale tốt: 19 ms ở n=50 vì mỗi bước chỉ 1 eval. (5) GA mặc định đuối dần khi n tăng (13.843 ở n=50) — cần tăng pop/generations.")

h1("7. Vẽ tour trước/sau tối ưu + biểu đồ")
img("tour_before_n20.png", caption="Hình 3. Trước tối ưu n=20 (L0=10.4435) — tour rối, nhiều cạnh cắt nhau.")
img("tour_Random_n20.png", caption="Hình 4. Sau Random Restarts (L=3.8205) — tour gọn, gần như không còn cạnh cắt nhau.")
img("tour_First-choice_n20.png", caption="Hình 5. Sau First-choice HC (L=3.8723).")
img("tour_Simulated_n20.png", caption="Hình 6. Sau Simulated Annealing (L=3.9174).")
img("tour_Stochastic_n20.png", caption="Hình 7. Sau Stochastic HC (L=4.1368) — còn 1–2 chỗ绕.")
img("convergence_n20.png", caption="Hình 8. Đường hội tụ các thuật toán (n=20). Steepest/Restart giảm bậc thang; SA dao động rồi ổn định; First-choice giảm dần.")
img("bar_Lbest_vs_n.png", caption="Hình 9. Chiều dài tour tốt nhất theo n và thuật toán.")
img("time_vs_n.png", caption="Hình 10. Thời gian chạy theo n (thang log) — Restart và Stochastic tăng dốc nhất.")

h1("8. Kết luận")
bullet("Steepest đơn nhanh nhưng dễ kẹt; Restart là cách đơn giản và hiệu quả nhất để cải thiện (giá = thời gian × số restart). ", "Thực hành: ")
bullet("First-choice là lựa chọn cân bằng khi n lớn (chất lượng gần Steepest, chi phí O(1) mỗi bước). ", "Mở rộng: ")
bullet("SA cần chỉnh lịch làm lạnh theo n (tăng T0/giảm α chậm hơn) mới giữ chất lượng ở n=50. ", "Lưu ý SA: ")
bullet("GA là bonus, cần tham số mạnh hơn mới cạnh tranh. ", "Bonus GA: ")
p("File chạy lại: bai7_tsp.py; số liệu chi tiết: bai7_output/bang_khao_sat.csv; ảnh: bai7_output/*.png.", italic=True)

doc.save(DOCX)
print("Saved:", DOCX)

"""Tao bao cao DOCX Lab 04 (khung day du + Bai 1 & Bai 6 da chay that).

Chay:  ./.venv/bin/python tao_baocao_lab04.py
Xuat:  lab04_csp_PhanDucNhan_3123410243.docx
"""
import json
import os
from typing import cast

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.styles.style import ParagraphStyle

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "images")
OUT = os.path.join(BASE, "output")
DOCX = os.path.join(BASE, "lab04_csp_PhanDucNhan_3123410243.docx")

with open(os.path.join(OUT, "bai1_ketqua.json"), encoding="utf-8") as f:
    b1 = json.load(f)
with open(os.path.join(OUT, "bai4_summary.json"), encoding="utf-8") as f:
    b4 = json.load(f)

NAVY = RGBColor(0x1F, 0x49, 0x7D)
GRAY = RGBColor(0x59, 0x59, 0x59)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

ALGOS = ["Steepest-ascent", "Inefficient Stochastic", "Efficient Stochastic"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_run(run, size: float = 14, bold=False, italic=False, name="Times New Roman",
            color=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade_paragraph(paragraph, fill):
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.insert_element_before(
        shd, "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap",
        "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN",
        "w:bidi", "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
        "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
        "w:textDirection", "w:textAlignment", "w:textboxTightWrap",
        "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
        "w:pPrChange")


def shade_cell(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.insert_element_before(
        shd, "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText",
        "w:vAlign", "w:hideMark", "w:headers", "w:cellIns", "w:cellDel",
        "w:cellMerge", "w:tcPrChange")


def p(text="", size: float = 14, bold=False, italic=False, align=None, after=6,
      before=0, indent=1.0, line=1.15, color=None, keep=False):
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.space_after = Pt(after)
    pf.space_before = Pt(before)
    pf.line_spacing = line
    if indent is not None:
        pf.first_line_indent = Cm(indent)
    if align is not None:
        par.alignment = align
    if keep:
        pf.keep_with_next = True
    if text:
        run = par.add_run(text)
        set_run(run, size=size, bold=bold, italic=italic, color=color)
    return par


def bullet(text, bold_prefix=None, size: float = 13.5):
    par = doc.add_paragraph(style="List Bullet")
    par.paragraph_format.space_after = Pt(3)
    par.paragraph_format.line_spacing = 1.12
    if bold_prefix:
        r = par.add_run(bold_prefix)
        set_run(r, size=size, bold=True)
    r = par.add_run(text)
    set_run(r, size=size)
    return par


def h1(text, page_break=True):
    par = doc.add_paragraph(style="Heading 1")
    par.paragraph_format.page_break_before = page_break
    par.paragraph_format.space_before = Pt(0)
    par.paragraph_format.space_after = Pt(10)
    par.paragraph_format.keep_with_next = True
    run = par.add_run(text)
    set_run(run, size=16, bold=True, color=NAVY)
    return par


def h2(text):
    par = doc.add_paragraph(style="Heading 2")
    par.paragraph_format.space_before = Pt(10)
    par.paragraph_format.space_after = Pt(4)
    par.paragraph_format.keep_with_next = True
    run = par.add_run(text)
    set_run(run, size=14, bold=True, color=NAVY)
    return par


def code_block(lines, size: float = 9.5):
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.left_indent = Cm(0.45)
    pf.space_before = Pt(3)
    pf.space_after = Pt(7)
    pf.line_spacing = 1.0
    shade_paragraph(par, "F2F2F2")
    run = par.add_run()
    for i, line in enumerate(lines):
        if i:
            run.add_break()
        run.add_text(line)
    set_run(run, size=size, name="Consolas")
    return par


def figure(filename, caption, width_cm=15.5, size: float = 12):
    path = os.path.join(IMG, filename)
    if not os.path.exists(path):
        p(f"[Thiếu ảnh {filename}]", italic=True, color=RGBColor(0xC0, 0, 0))
        return
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    run = cap.add_run(caption)
    set_run(run, size=size, italic=True, color=GRAY)


def table(headers, rows, widths=None, font_size: float = 11, center_cols=(), zebra=True):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    hdr = tbl.rows[0]
    trpr = hdr._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    trpr.append(tbl_header)
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ""
        par = cell.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(2)
        run = par.add_run(h)
        set_run(run, size=font_size, bold=True, color=WHITE)
        shade_cell(cell, "1F497D")
    for ri, row in enumerate(rows):
        cells = tbl.add_row().cells
        for ci, value in enumerate(row):
            cell = cells[ci]
            cell.text = ""
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            run = par.add_run(str(value))
            set_run(run, size=font_size)
            if ci in center_cols:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if zebra and ri % 2 == 1:
                shade_cell(cell, "EAF0F8")
    if widths:
        for row in tbl.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return tbl


def placeholder(text):
    par = doc.add_paragraph()
    par.paragraph_format.left_indent = Cm(0.3)
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(8)
    shade_paragraph(par, "FFF2CC")
    run = par.add_run(text)
    set_run(run, size=12.5, italic=True, color=RGBColor(0x7F, 0x60, 0x00))


def add_toc_static(entries):
    for title, page in entries:
        par = doc.add_paragraph()
        pf = par.paragraph_format
        pf.space_after = Pt(4)
        pf.line_spacing = 1.0
        pf.tab_stops.add_tab_stop(Cm(15.8), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        run = par.add_run(title)
        set_run(run, size=13)
        tab = par.add_run("\t")
        set_run(tab, size=13)
        num = par.add_run(str(page))
        set_run(num, size=13)


def add_page_number(paragraph):
    run = paragraph.add_run("Trang ")
    set_run(run, size=9, color=GRAY)
    run2 = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run2._r.append(fld_begin)
    run2._r.append(instr)
    run2._r.append(fld_end)
    set_run(run2, size=9, color=GRAY)


# ---------------------------------------------------------------------------
# Document setup
# ---------------------------------------------------------------------------
doc = Document()
style = cast(ParagraphStyle, doc.styles["Normal"])
style.font.name = "Times New Roman"
style.font.size = Pt(14)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

section = doc.sections[0]
section.page_width = Cm(21.0)
section.page_height = Cm(29.7)
section.left_margin = Cm(2.54)
section.right_margin = Cm(2.54)
section.top_margin = Cm(2.2)
section.bottom_margin = Cm(2.2)
section.different_first_page_header_footer = True

# Header + footer (trang 2 tro di)
header_p = section.header.paragraphs[0]
header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
hr = header_p.add_run("Lab 04 – Bài toán thỏa mãn ràng buộc (CSP)  |  Trí tuệ Nhân tạo Nâng cao")
set_run(hr, size=9, italic=True, color=GRAY)
footer_p = section.footer.paragraphs[0]
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_page_number(footer_p)



# ---------------------------------------------------------------------------
# Cover
# ---------------------------------------------------------------------------
def cover_line(text, size=14, before=0, after=6):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(before)
    par.paragraph_format.space_after = Pt(after)
    par.paragraph_format.line_spacing = 1.0
    run = par.add_run(text)
    set_run(run, size=size, bold=True)
    return par


for _ in range(2):
    cover_line("", after=0)
cover_line("ỦY BAN NHÂN DÂN", 14)
cover_line("THÀNH PHỐ HỒ CHÍ MINH", 14)
cover_line("TRƯỜNG ĐẠI HỌC SÀI GÒN", 14)
for _ in range(4):
    cover_line("", after=0)
cover_line("BÁO CÁO BÀI TẬP HÀNG TUẦN", 13)
cover_line("TRÍ TUỆ NHÂN TẠO NÂNG CAO", 13)
for _ in range(4):
    cover_line("", after=0)
cover_line("BÀI TOÁN THỎA MÃN RÀNG BUỘC (CSP)", 18)
cover_line("(Lab 04)", 14)
for _ in range(3):
    cover_line("", after=0)
cover_line("Giảng Viên Hướng Dẫn : Đỗ Như Tài", 14)
for _ in range(2):
    cover_line("", after=0)
cover_line("Nhóm sinh viên thực hiện", 14)
cover_line("Phan Đức Nhân : 3123410243", 14)
cover_line("Võ Gia Kiệt : 3123410182", 14)
cover_line("Trần Trọng Nghĩa : 3123410236", 14)
cover_line("Nguyễn Đăng Khoa : 3123410168", 14)
for _ in range(4):
    cover_line("", after=0)
cover_line("Thành phố Hồ Chí Minh, tháng 10 năm 2026", 13)
cover_line("", after=0)
note = doc.add_paragraph()
note.alignment = WD_ALIGN_PARAGRAPH.CENTER
note.paragraph_format.space_before = Pt(10)
nr = note.add_run("Bản tổng hợp khung báo cáo: Bài 1 và Bài 4 đã hoàn thành, chạy thực nghiệm và ghi kết quả; "
                  "các bài còn lại giữ khung để nhóm ghép minh chứng.")
set_run(nr, size=10.5, italic=True, color=GRAY)

doc.add_page_break()


# ---------------------------------------------------------------------------
# TOC
# ---------------------------------------------------------------------------
toc_title = doc.add_paragraph()
toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
toc_title.paragraph_format.space_after = Pt(12)
tr = toc_title.add_run("MỤC LỤC")
set_run(tr, size=16, bold=True, color=NAVY)
add_toc_static([
    ("MỤC TIÊU, MÔI TRƯỜNG VÀ PHƯƠNG PHÁP", 3),
    ("BÀI 1. BIỂU DIỄN CSP VÀ SINH ĐỒ THỊ CẦN TÔ MÀU", 5),
    ("BÀI 2. KIỂM TRA TÍNH ĐẦY ĐỦ VÀ NHẤT QUÁN", 10),
    ("BÀI 3. BACKTRACKING SEARCH VÀ THỨ TỰ CHỌN BIẾN", 11),
    ("BÀI 4. FORWARD CHECKING VÀ MIN-CONFLICTS", 12),
    ("BÀI 5. BIỂU DIỄN BÀI TOÁN N QUÂN HẬU", 18),
    ("BÀI 6. CÁC BIẾN THỂ HILL CLIMBING (TASK 1–3)", 19),
    ("BÀI 7. RANDOM RESTART VÀ SIMULATED ANNEALING (TASK 4–5)", 20),
    ("BÀI 8. ĐÁNH GIÁ TỔNG HỢP VÀ BÀI NÂNG CAO (TASK 6)", 21),
    ("ĐỐI CHIẾU THỰC NGHIỆM VÀ CÂU HỎI THẢO LUẬN", 22),
    ("TÀI LIỆU, MÃ NGUỒN VÀ CHẠY LẠI", 25),
    ("PHỤ LỤC HÌNH MINH HỌA", 26),
])

p("Ghi chú: mục lục tĩnh theo bố cục hiện tại; cập nhật lại số trang sau khi ghép đủ bài của nhóm.",
  italic=True, size=10.5, before=8, color=GRAY, indent=0)

doc.add_page_break()


# ---------------------------------------------------------------------------
# Muc tieu, moi truong, phuong phap
# ---------------------------------------------------------------------------
h1("MỤC TIÊU, MÔI TRƯỜNG VÀ PHƯƠNG PHÁP", page_break=False)
p("Theo đề “TTNTNC – Lab04 – Bài toán thỏa mãn ràng buộc”, nhóm cần biểu diễn CSP bằng tập biến X, "
  "miền giá trị D và tập ràng buộc C; cài đặt Backtracking cho bài toán tô màu đồ thị và khảo sát MRV, "
  "Forward Checking; mô hình hóa n-Queens bằng số cặp quân hậu xung đột; cài đặt các biến thể Hill Climbing, "
  "Random Restart và Simulated Annealing; đánh giá lời giải theo tính hợp lệ, số xung đột, số nút/lần đánh giá, "
  "thời gian chạy, tỷ lệ thành công và khả năng mở rộng.", indent=1.0)
p("Dữ liệu và chương trình sử dụng: CSP_graph_coloring_example.ipynb (sinh đồ thị phẳng bằng Delaunay, CSP 3/4 màu, "
  "complete/consistent, Backtracking), n_queens.ipynb (random_board, conflicts, show_board, Task 1–6), "
  "README.md (giới thiệu hai notebook và quan hệ CSP với tìm kiếm cục bộ).", indent=1.0)
p("Phương pháp: (i) đọc đề PDF và xác định đúng sản phẩm từng bài; (ii) chạy lại dữ liệu gốc của notebook với seed cố định; "
  "(iii) cài đặt các thuật toán của Bài 1 và Bài 4 thành script chạy được từ đầu, lưu log/JSON/CSV và hình; "
  "(iv) đối chiếu kết quả với lý thuyết (định lý 4 màu, K4, MRV, Forward Checking, min-conflicts); "
  "(v) chỉ ghi các số đo thật, không suy đoán kết quả của các bài chưa chạy.", indent=1.0)

p("Bảng 1. Phân công và trạng thái các bài trong Lab 04", bold=True, italic=True, size=12, before=6, after=3, keep=True)
table(
    ["Bài", "Nội dung", "Phụ trách", "Trạng thái"],
    [
        ["1", "CSP, sinh đồ thị, khảo sát 3/4 màu", "Phan Đức Nhân", "Hoàn thành"],
        ["2", "complete/consistent, lưu trữ csp", "Võ Gia Kiệt", "Khung chờ ghép"],
        ["3", "Backtracking, MRV, LCV", "Nguyễn Đăng Khoa", "Khung chờ ghép"],
        ["4", "Forward Checking, Min-Conflicts", "Phan Đức Nhân", "Hoàn thành"],
        ["5", "Mô hình n-Queens, h(board)", "Trần Trọng Nghĩa", "Khung chờ ghép"],
        ["6", "Steepest / Inefficient / Efficient HC", "Trần Trọng Nghĩa", "Khung (đã bàn giao)"],
        ["7", "Random Restart, Simulated Annealing", "Nguyễn Đăng Khoa", "Khung chờ ghép"],
        ["8", "Task 6, hội tụ, log-log, nâng cao", "Võ Gia Kiệt", "Khung chờ ghép"],
    ],
    widths=[1.0, 8.4, 3.2, 2.8],
    font_size=9.5,
    center_cols=(0,),
)

p("Môi trường chạy thực nghiệm: macOS (darwin), Python 3.13.14; NumPy 2.5.4, SciPy 1.18.1, "
  "Matplotlib 3.11.2, python-docx. Bài 1 dùng seed 1111 đúng như notebook gốc; Bài 4 dùng seed 2024, "
  "30 đồ thị cho mỗi n ∈ {10, 20, 30, 50} và mỗi miền 3/4 màu; mỗi đồ thị chạy Backtracking, Forward Checking "
  "và Min-Conflicts. Thời gian đo bằng time.perf_counter(), đơn vị mili-giây; vì thời gian rất nhỏ nên báo cáo "
  "dùng trung vị để giảm ảnh hưởng của outlier và chỉ so sánh tương đối trên cùng một máy.", indent=1.0)
p("Lưu ý về phạm vi: báo cáo này do Phan Đức Nhân tổng hợp. Bài 1 và Bài 4 là phần đã thực hiện, có script, "
  "log, CSV và hình thật. Các Bài 2, 3, 5, 6, 7, 8 được giữ khung theo đúng mẫu Lab 03 để nhóm ghép kết quả, "
  "không ghi số liệu ước lượng hoặc số liệu của thành viên khác khi chưa có minh chứng.", indent=1.0)

p("Tóm tắt kết quả nhanh:", bold=True, before=8)
bullet("đồ thị Delaunay n = 10 có 21 cạnh; 4 màu tìm được nghiệm sau 11 nút; 3 màu vô nghiệm vì chứa K4 "
       "{2, 3, 4, 9}, Backtracking duyệt hết 70 nút để chứng minh.", "Bài 1: ")
bullet("Forward Checking giảm COUNT trên instance 3 màu vô nghiệm (ví dụ 133,0 → 39,8 ở n = 50) nhưng chi phí "
       "quản lý miền có thể làm thời gian tăng trên instance 4 màu dễ; Min-Conflicts nhanh trên 4 màu "
       "(30/30, 30/30, 29/30, 28/30) nhưng không chứng minh được vô nghiệm. Trong 30 đồ thị Delaunay mỗi n, "
       "n = 20/30/50 có 0/30 đồ thị 3 màu được; họ bipyramid 3 màu kiểm chứng trường hợp có nghiệm.", "Bài 4: ")
bullet("bản chạy thử Task 1–3 đã bàn giao cho Trần Trọng Nghĩa trong thư mục bai6_hill_climbing/; "
       "báo cáo này không tính các số đó vào kết quả của Nhân.", "Bài 6: ")


# ---------------------------------------------------------------------------
# BAI 1
# ---------------------------------------------------------------------------
h1("BÀI 1. BIỂU DIỄN CSP VÀ SINH ĐỒ THỊ CẦN TÔ MÀU")
p("Mục tiêu của bài: chạy các ô tạo dữ liệu của CSP_graph_coloring_example.ipynb với n = 10 điểm ngẫu nhiên và "
  "phép tam giác hóa Delaunay; quan sát hình đồ thị và danh sách neighbors; xác định X, D, C; giải thích vì sao "
  "chỉ tạo ràng buộc cho cạnh i < j; khảo sát miền 4 màu và 3 màu; phân biệt phép gán đầy đủ với phép gán hợp lệ.",
  indent=1.0)

h2("1.1. Sinh dữ liệu và quan sát đồ thị")
p("Đoạn mã giữ nguyên seed 1111 và n = 10 như notebook gốc. Điểm được sắp xếp theo hoành độ trước khi gọi "
  "scipy.spatial.Delaunay, nên chỉ số đỉnh trùng với thứ tự quan sát trong notebook.", indent=1.0)
code_block([
    "np.random.seed(1111)",
    "n = 10",
    "points = np.random.randint(100, size=(n, 2))",
    "points = points[np.argsort(points[:, 0]), :]",
    "tri = Delaunay(points)",
    "indptr, indices = tri.vertex_neighbor_vertices",
    "neighbors = [np.sort(indices[indptr[k]:indptr[k+1]]) for k in range(n)]",
])
p("Kết quả chạy thật: 10 đỉnh, 21 cạnh, tổng bậc 42. Đỉnh 4 có bậc lớn nhất (7). Số cạnh khớp công thức "
  "đồ thị phẳng cực đại m = 3n − 3 − h với h = 6 đỉnh trên bao lồi (0, 2, 9, 8, 7, 1): 3·10 − 3 − 6 = 21. "
  "Đây là lý do đồ thị trông dày và có nhiều tam giác.", indent=1.0)
p("Bảng 2. Thông số đồ thị Delaunay thu được", bold=True, italic=True, size=12, before=4, after=3)
table(
    ["Thuộc tính", "Giá trị"],
    [
        ["Số đỉnh n", "10"],
        ["Số cạnh (ràng buộc)", "21"],
        ["Tổng bậc", "42"],
        ["Bậc lớn nhất", "7 (đỉnh 4)"],
        ["Đỉnh trên bao lồi h", "6 (0, 2, 9, 8, 7, 1)"],
        ["Kiểm tra 3n − 3 − h", "3·10 − 3 − 6 = 21"],
    ],
    widths=[7.0, 8.6],
    font_size=11,
    center_cols=(1,),
)
figure("bai1_graph_delaunay.png", "Hình B1.1. Đồ thị Delaunay n = 10 với seed 1111; số trên đỉnh là chỉ số biến.", 14.0)

p("Bảng 3. Danh sách neighbors thật của từng đỉnh", bold=True, italic=True, size=12, before=4, after=3)
rows = [[str(i), ", ".join(str(x) for x in nb)] for i, nb in enumerate(b1["neighbors"])]
table(["Đỉnh", "Neighbors (đã sắp xếp)"], rows, widths=[2.0, 13.6], font_size=11, center_cols=(0,))

h2("1.2. Mô hình CSP")
p("Mô hình hóa bài toán tô màu như sau:", indent=1.0)
bullet("X = {0, 1, …, 9} — mỗi đỉnh là một biến, cần gán một màu.", "Tập biến: ")
bullet("Với 4 màu: Dᵢ = {red, blue, green, orange} cho mọi i. Với 3 màu: Dᵢ = {red, blue, green}.", "Miền giá trị: ")
bullet("Với mỗi cạnh (i, j) của đồ thị, ràng buộc nhị phân Cᵢⱼ: color(i) ≠ color(j). Có 21 ràng buộc.", "Tập ràng buộc: ")
p("Vì sao chỉ tạo ràng buộc cho cạnh i < j? Mỗi cạnh vô hướng xuất hiện hai lần trong danh sách neighbors: "
  "j nằm trong neighbors[i] và i nằm trong neighbors[j]. Nếu duyệt cả hai chiều, ta sẽ lưu 42 mục ràng buộc, "
  "tức mỗi cạnh bị kiểm tra hai lần. Điều kiện i < j chỉ giữ một đại diện cho mỗi cạnh, đúng 21 ràng buộc, "
  "không làm thay đổi ngữ nghĩa nhưng tránh kiểm tra trùng trong consistent() và tránh nhân đôi chi phí.",
  indent=1.0)
code_block([
    "variables = [str(v) for v in range(n)]",
    "domains = {v: ['red', 'blue', 'green', 'orange'] for v in variables}",
    "constraints = {}",
    "for i in range(len(neighbors)):",
    "    for j in neighbors[i]:",
    "        if i < j:",
    "            constraints[(str(i), str(j))] = True",
    "csp = {'variables': variables, 'domains': domains,",
    "       'constraints': constraints.keys()}",
])

h2("1.3. complete() và consistent()")
p("complete(assignment, csp) kiểm tra mọi biến đã có trong assignment chưa. consistent(assignment, csp) kiểm tra "
  "mọi ràng buộc mà cả hai đầu đã được gán: nếu hai đỉnh kề nhau trùng màu thì trả về False. Một phép gán rỗng "
  "được coi là consistent (không có ràng buộc nào bị vi phạm) nhưng chưa complete.", indent=1.0)
code_block([
    "def complete(assignment, csp):",
    "    return all(v in assignment for v in csp['variables'])",
    "",
    "def consistent(assignment, csp):",
    "    for a, b in csp['constraints']:",
    "        if a in assignment and b in assignment:",
    "            if assignment[a] == assignment[b]:",
    "                return False",
    "    return True",
])
p("Bảng 4. Năm ca kiểm thử complete()/consistent() — tất cả đều cho kết quả đúng kỳ vọng", bold=True,
  italic=True, size=12, before=4, after=3)
table(
    ["#", "Phép gán", "complete()", "consistent()", "Kết luận"],
    [
        ["1", "{} (rỗng)", "False", "True", "Chưa đủ biến; chưa vi phạm ràng buộc nào"],
        ["2", "{0: red, 1: blue, 2: green}", "False", "True", "Gán một phần, hợp lệ"],
        ["3", "{2: red, 3: red}", "False", "False", "Cạnh (2,3) bị vi phạm"],
        ["4", "Tất cả 10 đỉnh đều red", "True", "False", "Đầy đủ nhưng vi phạm 21 cạnh"],
        ["5", "Lời giải 4 màu thu được", "True", "True", "Nghiệm CSP hợp lệ"],
    ],
    widths=[0.9, 5.4, 2.3, 2.4, 5.2],
    font_size=10.5,
    center_cols=(0, 2, 3),
)
p("Nhận xét: complete() chỉ nói “đã gán đủ biến”, không nói “gán đúng”. Phép gán tất cả red là complete nhưng "
  "vi phạm mọi cạnh, nên không phải nghiệm. Ngược lại, một phép gán một phần có thể consistent nhưng chưa "
  "complete. Nghiệm CSP chỉ khi đồng thời complete = True và consistent = True. Đây là điểm khác nhau cốt lõi "
  "giữa “đầy đủ” và “hợp lệ”.", indent=1.0)

h2("1.4. Kết quả tô màu với 4 màu và 3 màu")
p("Cùng đồ thị và cùng thứ tự biến, chạy Backtracking gốc (chọn biến chưa gán đầu tiên, thử màu theo thứ tự "
  "red → blue → green → orange) cho hai miền:", indent=1.0)
table(
    ["Miền màu", "Kết quả", "Số màu dùng", "COUNT (nút)", "Thời gian", "Ghi chú"],
    [
        ["4 màu", "Hợp lệ", "4", "11", "0.05 ms", "Tìm thấy nghiệm nhanh"],
        ["3 màu", "Không có nghiệm", "—", "70", "0.25 ms", "Duyệt hết, chứng minh vô nghiệm"],
    ],
    widths=[2.4, 3.2, 2.2, 2.2, 2.4, 3.4],
    font_size=10.5,
    center_cols=(0, 1, 2, 3, 4),
)
p("Lời giải 4 màu thật: 0 = red, 1 = blue, 2 = blue, 3 = red, 4 = green, 5 = orange, 6 = red, 7 = green, "
  "8 = blue, 9 = orange. Kiểm tra lại bằng consistent(): mọi cạnh đều khác màu. Lời giải dùng đủ 4 màu, "
  "phù hợp với định lý Bốn màu cho đồ thị phẳng: mọi đồ thị phẳng đều tô được bằng không quá 4 màu.",
  indent=1.0)
figure("bai1_coloring_4.png", "Hình B1.2. Lời giải tô màu hợp lệ với 4 màu (COUNT = 11 nút, 0.05 ms).", 14.0)

p("Với 3 màu, Backtracking trả về None sau 70 nút. Kết quả này không phải do thuật toán thiếu sót: đồ thị chứa "
  "một clique K4 gồm các đỉnh {2, 3, 4, 9}. Bốn đỉnh này đôi một kề nhau (6 cạnh), nên chúng bắt buộc phải nhận "
  "4 màu khác nhau. Vì vậy không tồn tại phép tô 3 màu nào; Backtracking đã vét cạn toàn bộ cây tìm kiếm và "
  "việc trả về None là một chứng minh vô nghiệm cho instance cụ thể này.", indent=1.0)
figure("bai1_coloring_3_fail.png", "Hình B1.3. Không thể tô 3 màu: các cạnh đỏ làm nổi bật clique K4 = {2, 3, 4, 9}.", 14.0)
p("Phân biệt hai tình huống thường bị nhầm: (i) thuật toán đúng nhưng instance vô nghiệm — như trường hợp 3 màu "
  "ở đây, kết luận vô nghiệm là hợp lệ vì đã duyệt hết; (ii) thuật toán bị giới hạn thời gian nên dừng trước khi "
  "duyệt hết — khi đó chỉ được nói “chưa tìm thấy”, không được nói “vô nghiệm”. Với 4 màu, định lý Bốn màu bảo đảm "
  "luôn có nghiệm cho đồ thị phẳng, nên thất bại (nếu có) chỉ có thể do lỗi cài đặt hoặc do dừng sớm.",
  indent=1.0)

h2("1.5. Kết luận Bài 1")
bullet("Đồ thị Delaunay n = 10 có 21 cạnh, 10 biến, miền 4 màu; mô hình CSP đầy đủ gồm X, D, C.", "Mô hình: ")
bullet("Mỗi cạnh chỉ lưu một lần với i < j: 42 mục directed → 21 ràng buộc, tránh kiểm tra trùng.", "Ràng buộc: ")
bullet("complete (đủ biến) khác consistent (không vi phạm); nghiệm cần cả hai. Phép gán tất cả red là complete nhưng không consistent.", "Kiểm tra: ")
bullet("4 màu: có nghiệm sau 11 nút; 3 màu: vô nghiệm, chứng minh bằng K4 {2,3,4,9} và duyệt hết 70 nút.", "Thực nghiệm: ")


# ---------------------------------------------------------------------------
# BAI 2 - frame
# ---------------------------------------------------------------------------
h1("BÀI 2. KIỂM TRA TÍNH ĐẦY ĐỦ VÀ NHẤT QUÁN")
placeholder("KHUNG CHỜ GHÉP — Bài 2 do Võ Gia Kiệt thực hiện. Khi ghép, thay đoạn này bằng bảng kiểm thử, "
            "mã nguồn và nhận xét thật; không giữ lại số liệu minh họa.")
p("Nội dung cần điền theo đề:", bold=True, before=2)
bullet("Chạy complete() và consistent() với phép gán rỗng, một phần, đầy đủ và phép gán có hai đỉnh kề cùng màu.")
bullet("Viết ít nhất 3 ca kiểm thử, ghi rõ kết quả mong đợi và kết quả thực tế; nhận xét vì sao complete() không đồng nghĩa consistent().")
bullet("Xác định dạng lưu trữ csp gồm variables, domains, constraints; phân biệt phép gán từng phần và nghiệm CSP.")
p("Khung bảng gợi ý:", bold=True, before=4)
table(
    ["Ca", "Phép gán", "complete()", "consistent()", "Kỳ vọng", "Thực tế"],
    [
        ["1", "{}", "", "", "", ""],
        ["2", "Một phần", "", "", "", ""],
        ["3", "Đầy đủ nhưng trùng màu", "", "", "", ""],
        ["4", "Đầy đủ và hợp lệ", "", "", "", ""],
    ],
    widths=[1.2, 4.6, 2.2, 2.3, 2.2, 2.2],
    font_size=10.5,
    center_cols=(0, 2, 3, 4, 5),
)
placeholder("Ghi chú ghép bài: giữ nguyên định nghĩa complete/consistent của Bài 1 để bốn bài dùng chung một mô hình csp.")


# ---------------------------------------------------------------------------
# BAI 3 - frame
# ---------------------------------------------------------------------------
h1("BÀI 3. BACKTRACKING SEARCH VÀ THỨ TỰ CHỌN BIẾN")
placeholder("KHUNG CHỜ GHÉP — Bài 3 do Nguyễn Đăng Khoa thực hiện. Cần có bảng so sánh baseline, MRV và LCV "
            "trên cùng một đồ thị/seed, kèm COUNT và thời gian thật.")
p("Nội dung cần điền theo đề:", bold=True, before=2)
bullet("Chạy backtrack_search(csp), ghi COUNT, thời gian và kết quả tô màu; trực quan hóa bằng đoạn mã trong notebook.")
bullet("Phân tích select_unassigned_var(): bản gốc chọn biến chưa gán đầu tiên; cài đặt MRV theo miền còn hợp lệ.")
bullet("Bổ sung Least Constraining Value (LCV) và so sánh với bản gốc trên cùng đồ thị.")
p("Số liệu dưới đây lấy từ báo cáo Bài 3 của Nguyễn Đăng Khoa: n = 40, miền 4 màu.", bold=True, before=4)
table(
    ["Cấu hình", "Kết quả", "COUNT", "Thời gian (ms)", "Số màu dùng", "Ghi chú"],
    [
        ["Backtracking gốc", "Có nghiệm", "158", "9.483", "4", "Nhiều nhánh cụt, quay lui"],
        ["MRV", "Có nghiệm", "41", "1.594", "4", "Giảm ~74% số nút"],
        ["MRV + LCV", "Có nghiệm", "41", "1.530", "4", "Nút tối thiểu, nhanh nhất"],
    ],
    widths=[4.0, 2.6, 2.0, 2.8, 2.4, 3.0],
    font_size=10.5,
    center_cols=(1, 2, 3, 4),
)
p("Với n = 10, cả ba cấu hình đều dừng ở 11 nút (0.118 / 0.127 / 0.273 ms), nên MRV/LCV không cải thiện rõ; "
  "với n = 40, MRV giảm 158 → 41 nút và LCV giữ nguyên số nút nhưng giảm nhẹ thời gian.", italic=True)
p("Câu hỏi cần trả lời: MRV tác động thế nào đến cây tìm kiếm; khi nào việc kiểm tra miền làm tăng chi phí; "
  "vì sao 3 màu có thể thất bại dù thuật toán đúng.", italic=True)



# ---------------------------------------------------------------------------
# BAI 4
# ---------------------------------------------------------------------------
h1("BÀI 4. FORWARD CHECKING VÀ MIN-CONFLICTS")
p("Bài 4 bổ sung hai hướng tiếp cận cho CSP tô màu: Forward Checking (tìm kiếm đầy đủ có suy luận) và "
  "Min-Conflicts (tìm kiếm cục bộ). Mục tiêu là so sánh tỷ lệ tìm nghiệm, COUNT/số bước và thời gian trên "
  "nhiều đồ thị Delaunay, với miền 3 và 4 màu; lưu ý các instance 3 màu có thể vô nghiệm.", indent=1.0)

h2("4.1. Mô hình và thuật toán")
p("Mô hình CSP giữ nguyên như Bài 1: X là các đỉnh, Dᵢ là 3 hoặc 4 màu, C là các cạnh i < j với ràng buộc "
  "color(i) ≠ color(j). Ba thuật toán dùng chung biểu diễn adjacency list:", indent=1.0)
bullet("chọn biến chưa gán đầu tiên, thử màu theo thứ tự, chỉ kiểm tra nhất quán với các đỉnh đã gán. "
       "Đây là mốc so sánh, không có MRV và không có Forward Checking.", "Backtracking gốc: ")
bullet("chọn biến chưa gán có miền nhỏ nhất (MRV); sau khi gán màu c, loại c khỏi miền các láng giềng chưa gán; "
       "nếu miền rỗng thì quay lui ngay; khi quay lui, phục hồi các màu đã loại. Đếm thêm số lần tỉa miền (prunes).",
       "Forward Checking + MRV: ")
bullet("khởi tạo phép gán đầy đủ ngẫu nhiên; mỗi bước chọn ngẫu nhiên một đỉnh đang vi phạm, thử cả k màu và chọn "
       "màu ít xung đột nhất (hòa thì chọn ngẫu nhiên). Bản có random walk 5% thỉnh thoảng chọn màu ngẫu nhiên để "
       "thoát plateau. Dừng khi hết xung đột hoặc sau 5000 bước.", "Min-Conflicts: ")
p("Khác biệt bản chất: Backtracking và Forward Checking là tìm kiếm đầy đủ (complete), nên nếu duyệt hết mà không "
  "có nghiệm thì đó là chứng minh vô nghiệm. Min-Conflicts là tìm kiếm cục bộ (incomplete): nó chỉ có thể tìm ra "
  "nghiệm, không thể chứng minh vô nghiệm; thất bại sau 5000 bước chỉ có nghĩa là “chưa tìm thấy”.", indent=1.0)
code_block([
    "def forward_checking(adj, k):",
    "    domains = [[c for c in range(k)] for _ in range(n)]",
    "    assignment = [-1] * n",
    "    # chon bien MRV, gan mau, loai mau khoi mien lang gieng chua gan",
    "    # neu mien rong -> quay lui; khi quay lui -> phuc hoi mau da loai",
])
code_block([
    "while steps < max_steps:",
    "    conflicted = [v for v in range(n) if conflicts_of(v) > 0]",
    "    if not conflicted:",
    "        return True",
    "    v = conflicted[rng.integers(len(conflicted))]",
    "    scores = [sum(assignment[u] == c for u in adj[v]) for c in range(k)]",
    "    best = [c for c in range(k) if scores[c] == min(scores)]",
    "    assignment[v] = best[rng.integers(len(best))]",
    "    steps += 1",
])
p("Chi phí: Forward Checking làm thêm công việc quản lý miền ở mỗi nút (duyệt láng giềng, xóa/phục hồi màu), nên "
  "COUNT giảm nhưng thời gian mỗi nút tăng. Min-Conflicts mỗi bước chỉ đánh giá k màu cho một đỉnh, nhưng không có "
  "bảo đảm tiến triển và có thể lặp vô hạn nếu không có max_steps hoặc random walk.", indent=1.0)

h2("4.2. Thiết lập thực nghiệm")
p("Với mỗi n ∈ {10, 20, 30, 50}, sinh 30 đồ thị Delaunay từ các seed 2024 + 100n + i (i = 0..29). Mỗi đồ thị chạy "
  "với miền 3 màu và 4 màu. Giới hạn 200.000 nút cho Backtracking/FC; Min-Conflicts chạy tối đa 5000 bước, bản walk "
  "dùng xác suất 5%. Với mỗi cấu hình, ghi COUNT, số lần tỉa miền, số bước, thời gian và kết quả.", indent=1.0)
p("Vì FC là thuật toán đầy đủ và không chạm giới hạn nút trong toàn bộ thí nghiệm, kết quả FC được dùng làm "
  "“ground truth” cho tính có nghiệm của instance 3 màu: FC tìm được nghiệm → solvable; FC duyệt hết và trả None → "
  "vô nghiệm. Với 4 màu, định lý Bốn màu bảo đảm mọi đồ thị phẳng đều có nghiệm, nên cả 30/30 instance đều solvable.",
  indent=1.0)
p("Thời gian rất nhỏ (cỡ 0,01–0,1 ms cho BT/FC; 0,05–60 ms cho Min-Conflicts), nên báo cáo dùng trung vị. Chỉ một "
  "lần đo bị nhiễu hệ điều hành cũng có thể kéo trung bình lên nhiều lần, trong khi trung vị ổn định hơn; vì vậy các "
  "bảng dùng cột “trung vị” và không dùng trung bình để so sánh.", indent=1.0)

h2("4.3. Kết quả với 4 màu")
p("Bảng 4.1. Backtracking và Forward Checking — 4 màu, 30 đồ thị mỗi n", bold=True, italic=True, size=12, before=4, after=3)
rows41 = []
for n in (10, 20, 30, 50):
    v = b4["random"][str(n)]["4"]
    rows41.append([str(n), f"{v['avg_edges']:.1f}", f"{v['bt_avg_nodes_all']:.1f}",
                   f"{v['fc_avg_nodes_all']:.1f}", f"{v['bt_median_time_ms']:.4f}",
                   f"{v['fc_median_time_ms']:.4f}", f"{v['fc_avg_prunes']:.1f}"])
table(["n", "Cạnh TB", "BT COUNT", "FC COUNT", "BT trung vị (ms)", "FC trung vị (ms)", "FC prunes"],
      rows41, widths=[1.3, 2.0, 2.4, 2.4, 2.8, 2.8, 2.4], font_size=10.5, center_cols=(0, 1, 2, 3, 4, 5, 6))
p("Cả 30/30 đồ thị đều được Backtracking và FC giải. FC giảm COUNT khi n ≥ 20, rõ nhất ở n = 50 (xem Bảng 4.1), nhưng thời gian "
  "trung vị gần như không giảm, thậm chí cao hơn ở n = 10/20/30; chỉ đến n = 50 FC mới nhỉnh hơn. Nguyên nhân: FC giảm "
  "số nút nhưng mỗi nút phải duyệt láng giềng, xóa và phục hồi màu trong list domain, nên chi phí mỗi nút tăng. "
  "Đây là minh họa định lượng cho việc “kiểm tra miền làm tăng chi phí” khi instance vốn đã dễ.", indent=1.0)
figure("bai4_4color_nodes_time.png", "Hình B4.1. 4 màu: COUNT (trái, thang log) và thời gian trung bình (phải) của Backtracking và FC.", 15.0)
figure("bai4_coloring_example.png", "Hình B4.2. Một lời giải 4 màu cho đồ thị Delaunay n = 20 do Forward Checking tìm được.", 12.5)

p("Bảng 4.2. Min-Conflicts — 4 màu, 30 đồ thị mỗi n (bản thuần và bản random walk 5%)", bold=True, italic=True, size=12, before=4, after=3)
rows42 = []
for n in (10, 20, 30, 50):
    v = b4["random"][str(n)]["4"]
    rows42.append([str(n), f"{v['mc_found_all']}/30", f"{v['mc_avg_steps_all']:.1f}",
                   f"{v['mc_median_time_ms']:.4f}", f"{v['mcw_found_all']}/30",
                   f"{v['mcw_avg_steps_all']:.1f}", f"{v['mcw_median_time_ms']:.4f}"])
table(["n", "MC thuần", "Số bước TB", "Trung vị (ms)", "MC + walk", "Số bước TB", "Trung vị (ms)"],
      rows42, widths=[1.3, 2.4, 2.6, 2.6, 2.4, 2.6, 2.6], font_size=10.5, center_cols=(0, 1, 2, 3, 4, 5, 6))
p("Min-Conflicts thuần đạt 30/30 ở n = 10/20, 29/30 ở n = 30 và 28/30 ở n = 50. Bản random walk 5% đạt 30/30 ở cả "
  "bốn n và giảm mạnh số bước ở n = 30/50 (xem Bảng 4.2): khi rơi vào plateau, bước đi ngẫu nhiên giúp thoát thay vì "
  "lặp mãi một màu. Đổi lại, walk có thể nhận bước tệ hơn nên không còn là “chỉ đi xuống” thuần túy; đây là đánh đổi "
  "có kiểm soát, không phải min-conflicts nguyên bản.", indent=1.0)
figure("bai4_minconflicts_4color.png", "Hình B4.3. Min-Conflicts 4 màu: tỷ lệ tìm được nghiệm (trái) và số bước trung bình (phải).", 15.0)

h2("4.4. Kết quả với 3 màu")
p("Bảng 4.3. 3 màu trên đồ thị Delaunay ngẫu nhiên — số instance có nghiệm do FC xác định", bold=True, italic=True, size=12, before=4, after=3)
rows43 = []
for n in (10, 20, 30, 50):
    v = b4["random"][str(n)]["3"]
    rows43.append([str(n), f"{v['solvable']}/30", f"{v['unsolvable']}/30",
                   f"{v['bt_avg_nodes_unsolvable']:.1f}", f"{v['fc_avg_nodes_unsolvable']:.1f}",
                   f"{v['bt_median_time_ms']:.4f}", f"{v['fc_median_time_ms']:.4f}", f"{v['fc_avg_prunes']:.1f}"])
table(["n", "Có nghiệm", "Vô nghiệm", "BT COUNT (vô nghiệm)", "FC COUNT (vô nghiệm)",
       "BT trung vị (ms)", "FC trung vị (ms)", "FC prunes"],
      rows43, widths=[1.1, 1.9, 1.9, 2.6, 2.6, 2.4, 2.4, 2.0], font_size=9.5, center_cols=(0, 1, 2, 3, 4, 5, 6, 7))
p("Kết quả nổi bật: với n = 20/30/50, không có đồ thị Delaunay nào trong 30 mẫu là 3 màu được; n = 10 chỉ có 4/30. "
  "Đây không phải lỗi thuật toán. Delaunay tạo ra triangulation phẳng cực đại; phần lớn có đỉnh bậc lẻ và/hoặc chứa K4, "
  "nên không thể tô 3 màu. FC đã chứng minh vô nghiệm cho 26/30 instance ở n = 10 và 30/30 ở n = 20/30/50, không chạm "
  "giới hạn nút. Với các instance vô nghiệm, FC dùng ít nút hơn Backtracking khoảng 2–3 lần (Bảng 4.3); thời gian trung vị "
  "cũng thấp hơn nhưng không tỷ lệ thuận với mức giảm COUNT, vì chi phí quản lý miền.", indent=1.0)
figure("bai4_3color_unsolvable.png", "Hình B4.4. 3 màu: chi phí chứng minh vô nghiệm của Backtracking và FC (30 đồ thị mỗi n).", 14.0)
p("Min-Conflicts trên 3 màu: với 4 instance solvable ở n = 10, cả bản thuần và bản walk đều đạt 4/4; với 26 instance "
  "vô nghiệm, cả hai chạy hết 5000 bước và không tìm được nghiệm — đúng như bản chất incomplete. Trên toàn bộ 30 instance "
  "n = 10, Min-Conflicts chỉ đạt 4/30, trong khi FC vừa tìm được 4 nghiệm vừa chứng minh 26 instance còn lại vô nghiệm. "
  "Ở n = 20/30/50, vì không có instance solvable, Min-Conflicts đạt 0/30 và tiêu tốn 5000 bước mỗi lần (trung vị "
  "vài chục ms), so với FC chứng minh vô nghiệm trong dưới 0,1 ms. Đây là khác biệt cốt lõi giữa tìm kiếm đầy đủ và "
  "tìm kiếm cục bộ: local search không thể phân biệt “vô nghiệm” với “chưa tìm thấy”.", indent=1.0)

h2("4.5. Kiểm chứng trên họ đồ thị 3 màu chắc chắn có nghiệm")
p("Để kiểm tra trường hợp 3 màu có nghiệm ở n lớn, nhóm dùng họ bipyramid: vòng rim m cạnh (m chẵn), thêm hai đỉnh apex "
  "nối với mọi đỉnh rim; n = m + 2, số cạnh = 3m. Cách tô màu: rim tô so le hai màu (vì m chẵn), hai apex cùng màu thứ ba. "
  "Mọi đỉnh có bậc chẵn (rim bậc 4, apex bậc m), nên đồ thị 3 màu được; đây là kiểm chứng độc lập với kết quả “vô nghiệm” "
  "của Delaunay ngẫu nhiên.", indent=1.0)
p("Bảng 4.4. Họ bipyramid 3 màu — Backtracking, Forward Checking và Min-Conflicts", bold=True, italic=True, size=12, before=4, after=3)
rows44 = []
for r in b4["bipyramid"]:
    rows44.append([str(r["n"]), str(r["edges"]), str(r["bt_nodes"]), str(r["fc_nodes"]),
                   str(r["fc_prunes"]), f"{r['mc_success_20']}/20", f"{r['mc_avg_steps']:.1f}"])
table(["n", "Cạnh", "BT COUNT", "FC COUNT", "FC prunes", "MC + walk", "Số bước TB"],
      rows44, widths=[1.4, 1.8, 2.4, 2.4, 2.4, 2.4, 2.6], font_size=10.5, center_cols=(0, 1, 2, 3, 4, 5, 6))
p("Trên họ này, Backtracking và FC đều tìm được nghiệm với COUNT nhỏ (9–31 nút); FC tỉa 13–57 giá trị nhưng không làm "
  "tăng đáng kể COUNT vì đồ thị đã dễ. Min-Conflicts + walk đạt 20/20 lần ở mọi n, số bước tăng từ 15,7 (n = 8) lên "
  "232,9 (n = 30). Kết quả này xác nhận: khi instance thực sự có nghiệm 3 màu, cả ba thuật toán đều giải được; thất bại "
  "hàng loạt trên Delaunay ngẫu nhiên là do instance vô nghiệm, không phải do cài đặt sai.", indent=1.0)
figure("bai4_bipyramid_3color.png", "Hình B4.5. Họ bipyramid 3 màu: tỷ lệ tìm nghiệm (trái) và COUNT của BT/FC (phải).", 15.0)

h2("4.6. Kết luận Bài 4")
bullet("FC giảm COUNT trên instance khó/vô nghiệm, nhưng chi phí quản lý miền có thể làm thời gian tăng ở instance dễ. "
       "Phải đo cả COUNT và thời gian, không chỉ nhìn số nút.", "Forward Checking: ")
bullet("Rất nhanh trên 4 màu (30/30 ở n = 10/20, 29/30 ở n = 30 và 28/30 ở n = 50), nhưng không thể chứng minh vô nghiệm. "
       "Random walk 5% cải thiện độ ổn định và giảm số bước khi plateau.", "Min-Conflicts: ")
bullet("Đa số đồ thị Delaunay không 3 màu được (0/30 ở n = 20/30/50; 4/30 ở n = 10). FC chứng minh vô nghiệm nhanh; "
       "Min-Conflicts chạy hết 5000 bước mà không kết luận được gì. Họ bipyramid chứng minh trường hợp có nghiệm vẫn giải được.",
       "3 màu: ")
bullet("Dùng Backtracking/FC khi cần kết luận đúng/sai (có nghiệm hoặc vô nghiệm); dùng Min-Conflicts khi đã biết instance "
       "có nghiệm và cần nghiệm nhanh trên đồ thị lớn.", "Chọn thuật toán: ")
# ---------------------------------------------------------------------------
# BAI 5 - frame
# ---------------------------------------------------------------------------
h1("BÀI 5. BIỂU DIỄN BÀI TOÁN N QUÂN HẬU")
placeholder("KHUNG CHỜ GHÉP — Bài 5 do Trần Trọng Nghĩa thực hiện. Cần hình bàn cờ n = 4 và n = 8 kèm h(board), "
            "cùng kiểm tra h(board) = 0 khi và chỉ khi không còn cặp quân hậu tấn công nhau.")
p("Nội dung cần điền theo đề:", bold=True, before=2)
bullet("Khảo sát random_board(n), conflicts(board), show_board(board) và bàn cờ mẫu [1, 3, 0, 2].")
bullet("Mỗi cột có đúng một quân hậu; board[i] là chỉ số hàng của quân ở cột i. Viết rõ hàm mục tiêu h(board).")
bullet("Tạo bàn cờ n = 4 và n = 8; hiển thị bàn cờ cùng h(board); kiểm tra h(board) = 0 khi và chỉ khi không có cặp tấn công.")
code_block([
    "h(board) = sum(C(k, 2) for k in row_count + diag1_count + diag2_count)",
    "# C(k,2) = k(k-1)/2: nhóm k quân trên cùng hàng/chéo tạo đúng C(k,2) cặp xung đột",
])



# ---------------------------------------------------------------------------
# BAI 6
# ---------------------------------------------------------------------------
h1("BÀI 6. CÁC BIẾN THỂ HILL CLIMBING (TASK 1–3)")
placeholder("KHUNG CHỜ GHÉP — Bài 6 do Trần Trọng Nghĩa thực hiện. Bản chạy thử Task 1–3 đã được bàn giao trong "
            "lab04_TrầnTrọngNghĩa-Bài 5, Bài 6/bai6_hill_climbing/; Nghĩa kiểm tra, chạy lại và chốt số liệu chính thức.")
p("Nội dung cần điền theo đề:", bold=True, before=2)
bullet("Task 1 — Steepest-ascent: đánh giá toàn bộ N = n(n−1) nước đi lân cận, chọn nước cải thiện tốt nhất, dừng tại cực trị cục bộ.")
bullet("Task 2 — Inefficient Stochastic: vẫn liệt kê toàn bộ nước cải thiện nhưng chọn ngẫu nhiên một nước, không chọn nước tốt nhất.")
bullet("Task 3 — Efficient Stochastic: sinh một lân cận ngẫu nhiên mỗi lần thử, nhận ngay nếu cải thiện; nêu rõ điều kiện dừng (patience) để tránh lặp vô hạn.")
bullet("Ghi nhận h cuối, số bước, số lần đánh giá hàm mục tiêu và các trường hợp local optimum/plateau; chạy tối thiểu 100 lần cho n = 4 và n = 8.")
p("Bàn giao: thư mục bai6_hill_climbing/ chứa bai6_nqueens_hc.py, 4 hình (hội tụ, bàn cờ, tỷ lệ thành công/chi phí, phân bố h) "
  "và output/bai6_runs.csv + bai6_summary.json từ một lần chạy thử 100 lần/cấu hình. Báo cáo này không trích các số đó "
  "như kết quả của Nhân; Nghĩa chịu trách nhiệm kiểm tra và cập nhật.", italic=True, size=12)
# ---------------------------------------------------------------------------
# BAI 7 - frame
# ---------------------------------------------------------------------------
h1("BÀI 7. RANDOM RESTART VÀ SIMULATED ANNEALING (TASK 4–5)")
placeholder("KHUNG CHỜ GHÉP — Bài 7 do Nguyễn Đăng Khoa thực hiện. Cần bảng 100 lần khởi động lại cho cả ba biến thể HC, "
            "bảng tham số SA (ít nhất hai lịch làm nguội) và hình hội tụ thật.")
p("Nội dung cần điền theo đề:", bold=True, before=2)
bullet("Task 4: chạy lại ba biến thể Hill Climbing từ bàn cờ ngẫu nhiên, tối đa 100 lần khởi động lại; thống kê số lần thành công và chi phí.")
bullet("Task 5: cài đặt Simulated Annealing, thử ít nhất hai lịch làm nguội; nêu công thức p = exp(−Δh/T) và cách giảm T.")
bullet("Giữ cùng n và điều kiện dừng khi so sánh; phân biệt “chạy thành công” (h = 0) với “nghiệm tốt nhất” (h nhỏ nhất).")
p("Khung bảng gợi ý:", bold=True, before=4)
table(
    ["Thuật toán", "n", "Số lần chạy", "Thành công (%)", "Số restart TB", "Thời gian TB (ms)", "h tốt nhất"],
    [["Steepest + Random Restart", "8", "100", "", "", "", ""],
     ["Inefficient + Random Restart", "8", "100", "", "", "", ""],
     ["Efficient + Random Restart", "8", "100", "", "", "", ""],
     ["Simulated Annealing (lịch 1)", "8", "100", "", "", "", ""],
     ["Simulated Annealing (lịch 2)", "8", "100", "", "", "", ""]],
    widths=[5.2, 1.3, 2.2, 2.4, 2.2, 2.4, 1.8],
    font_size=9.5,
    center_cols=(1, 2, 3, 4, 5, 6),
)


# ---------------------------------------------------------------------------
# BAI 8 - frame
# ---------------------------------------------------------------------------
h1("BÀI 8. ĐÁNH GIÁ TỔNG HỢP VÀ BÀI NÂNG CAO (TASK 6)")
placeholder("KHUNG CHỜ GHÉP — Bài 8 do Võ Gia Kiệt thực hiện. Cần tối thiểu 100 lượt cho n = 4 và n = 8, "
            "đồ thị hội tụ, đồ thị log-log thời gian theo n ∈ {4, 8, 12, 16, 20} và phần toán tử nâng cao.")
p("Nội dung cần điền theo đề:", bold=True, before=2)
bullet("Task 6: chạy tối thiểu 100 lượt với bàn cờ khởi tạo khác nhau tại n = 4 và n = 8; so sánh thời gian, h cuối, tỷ lệ thành công; trình bày bằng bảng/biểu đồ.")
bullet("Vẽ đồ thị hội tụ h theo số vòng lặp cho một lượt đại diện của mỗi thuật toán tại n = 8; nhận xét tốc độ cải thiện và plateau.")
bullet("Vẽ log-log thời gian chạy theo n ∈ {4, 8, 12, 16, 20} cho ít nhất hai thuật toán; thảo luận độ phức tạp thực nghiệm.")
bullet("Nâng cao (tùy chọn): khảo sát toán tử di chuyển cục bộ khác và so sánh với toán tử ban đầu.")
p("Khung bảng gợi ý:", bold=True, before=4)
table(
    ["Thuật toán", "n = 4: thời gian / h / %", "n = 8: thời gian / h / %", "n = 12", "n = 16", "n = 20"],
    [["Steepest-ascent", "", "", "", "", ""],
     ["Inefficient Stochastic", "", "", "", "", ""],
     ["Efficient Stochastic", "", "", "", "", ""],
     ["Simulated Annealing", "", "", "", "", ""]],
    widths=[4.6, 3.2, 3.2, 1.8, 1.8, 1.8],
    font_size=9.5,
    center_cols=(1, 2, 3, 4, 5),
)



# ---------------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------------
h1("ĐỐI CHIẾU THỰC NGHIỆM VÀ CÂU HỎI THẢO LUẬN")
p("Bảng 8. Bảng tổng hợp theo mẫu mục 4.2 của đề — các dòng Bài 1 và Bài 4 là số đo thật; các dòng n-Queens giữ khung.",
  bold=True, italic=True, size=12, before=2, after=3)
rows_sum = [
    ["Tô màu", "n = 10", "Backtracking", "4", "100 (1/1)", "COUNT 11", f"{b1['backtracking_4']['time_ms']/1000:.5f}"],
    ["Tô màu", "n = 10", "Backtracking", "3", "0 (0/1)", "COUNT 70", f"{b1['backtracking_3']['time_ms']/1000:.5f}"],
    ["Tô màu", "n = 10/20/30/50", "Forward Checking", "4", "30/30 mỗi n",
     "COUNT " + " / ".join(f"{b4['random'][str(n)]['4']['fc_avg_nodes_all']:.1f}" for n in (10, 20, 30, 50)),
     " / ".join(f"{b4['random'][str(n)]['4']['fc_median_time_ms']/1000:.7f}" for n in (10, 20, 30, 50))],
    ["Tô màu", "n = 10/20/30/50", "Forward Checking", "3",
     " / ".join(f"{b4['random'][str(n)]['3']['solvable']}/30" for n in (10, 20, 30, 50)),
     "COUNT " + " / ".join(f"{b4['random'][str(n)]['3']['fc_avg_nodes_unsolvable']:.1f}" for n in (10, 20, 30, 50)),
     " / ".join(f"{b4['random'][str(n)]['3']['fc_median_time_ms']/1000:.7f}" for n in (10, 20, 30, 50))],
    ["Tô màu", "n = 10", "Min-Conflicts", "3",
     f"{b4['random']['10']['3']['mc_found_solvable']}/{b4['random']['10']['3']['solvable']} (solvable)",
     f"steps {b4['random']['10']['3']['mc_avg_steps_all']:.0f} (toàn bộ)",
     f"{b4['random']['10']['3']['mc_median_time_ms']/1000:.4f}"],
    ["Tô màu", "n = 10/50", "Min-Conflicts", "4",
     f"{b4['random']['10']['4']['mc_found_all']}/30; {b4['random']['50']['4']['mc_found_all']}/30",
     f"steps {b4['random']['10']['4']['mc_avg_steps_all']:.1f} / {b4['random']['50']['4']['mc_avg_steps_all']:.1f}",
     f"{b4['random']['10']['4']['mc_median_time_ms']/1000:.4f} / {b4['random']['50']['4']['mc_median_time_ms']/1000:.4f}"],
    ["Tô màu", "n = 50", "Min-Conflicts + walk", "4",
     f"{b4['random']['50']['4']['mcw_found_all']}/30",
     f"steps {b4['random']['50']['4']['mcw_avg_steps_all']:.1f}",
     f"{b4['random']['50']['4']['mcw_median_time_ms']/1000:.4f}"],
    ["n-Queens", "n = 4", "Steepest HC", "100", "Chờ Bài 6", "", ""],
    ["n-Queens", "n = 8", "Inefficient Stochastic HC", "100", "Chờ Bài 6", "", ""],
    ["n-Queens", "n = 8", "Efficient Stochastic HC", "100", "Chờ Bài 6", "", ""],
    ["n-Queens", "n = 8", "Random Restart", "100", "Chờ Bài 7", "", ""],
    ["n-Queens", "n = 8", "Simulated Annealing", "100", "Chờ Bài 7", "", ""],
]
table(["Bài toán", "Kích thước", "Thuật toán", "Số màu / lượt", "Thành công (%)", "Xung đột / COUNT", "Thời gian (s)"],
      rows_sum, widths=[2.0, 2.4, 3.6, 1.9, 2.3, 2.6, 2.2], font_size=8.8, center_cols=(1, 3, 4, 5, 6))
p("* Với tô màu, mỗi cấu hình chạy 30 đồ thị (Bài 4) hoặc 1 đồ thị (Bài 1). “Thành công” của FC/Min-Conflicts tính trên "
  "30 instance; với 3 màu, một instance có thể vô nghiệm nên phải đọc thêm cột COUNT/thời gian và phần 4.4. "
  "COUNT của tô màu không đồng nhất với số lần đánh giá hàm mục tiêu của n-Queens. Với Min-Conflicts 3 màu, cột steps "
  "là trung bình trên cả 30 instance, gồm cả các lần chạy hết 5000 bước trên instance vô nghiệm; vì vậy nó không phản ánh "
  "chi phí trên instance solvable.", size=11, italic=True)

p("Trả lời sáu câu hỏi thảo luận (mục 4.3 của đề):", bold=True, before=8)
qa = [
    ("1. Khác nhau giữa complete() và consistent()? Một phép gán đầy đủ có luôn là nghiệm CSP không?",
     "complete() chỉ kiểm tra độ phủ: mọi biến đã được gán chưa. consistent() kiểm tra tính đúng: trong các biến đã gán, "
     "có ràng buộc nào bị vi phạm không. Phép gán đầy đủ không luôn là nghiệm: ví dụ tất cả 10 đỉnh cùng red là complete "
     "nhưng vi phạm cả 21 cạnh, nên consistent = False. Nghiệm CSP là phép gán vừa complete vừa consistent. "
     "Ngược lại, phép gán rỗng là consistent (không có gì để vi phạm) nhưng chưa complete."),
    ("2. MRV và Forward Checking tác động thế nào đến cây Backtracking? Khi nào kiểm tra miền làm tăng chi phí?",
     "MRV chọn biến có ít giá trị hợp lệ nhất trước, tức ưu tiên nhánh dễ thất bại (fail-first), làm giảm hệ số phân nhánh "
     "và thường rút ngắn cây. Forward Checking sau mỗi phép gán loại các màu xung đột khỏi miền của láng giềng chưa gán, "
     "phát hiện sớm miền rỗng và cắt nhánh trước khi đi sâu. Bài 4 định lượng: trên 3 màu, FC giảm COUNT từ 74,2 xuống 36,2 "
     "(n = 20), 95,2 xuống 39,0 (n = 30), 133,0 xuống 39,8 (n = 50), nhưng thời gian trung vị giảm ít hơn nhiều so với mức giảm COUNT; trên 4 màu, FC "
     "ít nút hơn nhưng chậm hơn ở n = 10/20/30 vì chi phí xóa/phục hồi miền. Khi đồ thị thưa, miền lớn hoặc instance dễ, "
     "chi phí quản lý miền có thể lớn hơn lợi ích cắt nhánh — phải đo cả COUNT và thời gian."),
    ("3. Vì sao tô màu bằng 3 màu có thể thất bại dù dùng thuật toán đúng? Làm sao phân biệt hết giới hạn tính toán và chứng minh vô nghiệm?",
     "Thất bại có thể do instance vô nghiệm, không phải do thuật toán sai. Ở Bài 1, đồ thị chứa K4 {2, 3, 4, 9}: bốn đỉnh "
     "đôi một kề nhau nên cần 4 màu phân biệt; Backtracking duyệt hết cây (70 nút) rồi trả None — đó là chứng minh vô nghiệm. "
     "Ở Bài 4, 30 đồ thị Delaunay mỗi n cho thấy n = 20/30/50 có 0/30 đồ thị 3 màu được, n = 10 có 4/30; FC chứng minh vô nghiệm "
     "cho 26/30 và 30/30 mà không chạm giới hạn nút. Ngược lại, họ bipyramid 3 màu (n = 8–30) luôn có nghiệm và cả BT/FC/MC đều "
     "giải được, nên thất bại trên Delaunay là do instance. Nếu thuật toán dừng vì hết thời gian/patience, chỉ được nói “chưa tìm "
     "thấy”, không được kết luận vô nghiệm; chỉ tìm kiếm đầy đủ đã duyệt hết cây mới có quyền kết luận vô nghiệm."),
    ("4. Hill Climbing gặp local optimum hoặc plateau ra sao? Random Restart và Simulated Annealing xử lý theo cách nào?",
     "Hill Climbing chỉ nhận nước đi cải thiện thật sự (strict <). Tại cực tiểu cục bộ, mọi lân cận đều tệ hơn; tại plateau, "
     "có lân cận bằng h nhưng không nhỏ hơn, nên thuật toán cũng dừng. Bản chạy thử bàn giao cho Bài 6 cho thấy 96,6–100% lần "
     "thất bại kết thúc ở trạng thái có ít nhất một lân cận cùng h — plateau là điểm kẹt phổ biến hơn cực tiểu nghiêm ngặt "
     "(số chính thức sẽ do Bài 6 chốt). Random Restart không sửa được điểm kẹt bên trong một lần chạy, nhưng chạy lại từ nhiều "
     "bàn đầu khác nhau sẽ tăng xác suất gặp vùng hút tốt; chi phí tăng theo số lần restart. Simulated Annealing chấp nhận nước "
     "đi tệ hơn với xác suất p = exp(−Δh/T): T cao lúc đầu cho phép đi ngang/đi xuống để thoát plateau và cực tiểu, T giảm dần "
     "để cuối cùng chỉ còn khai thác."),
    ("5. So sánh Steepest, Inefficient Stochastic và Efficient Stochastic về số lần đánh giá các láng giềng.",
     "Steepest và Inefficient đều quét toàn bộ N = n(n−1) lân cận ở mỗi vòng: Steepest chọn nước tốt nhất, Inefficient chọn "
     "ngẫu nhiên trong các nước cải thiện. Vì vậy chi phí mỗi bước của hai bản này xấp xỉ nhau và bằng O(N) lần đánh giá. "
     "Efficient chỉ sinh một lân cận ngẫu nhiên mỗi lần thử, tốn 1 lần đánh giá, nhưng phải thử patience lần liên tiếp không "
     "cải thiện mới kết luận kẹt. Bản chạy thử bàn giao cho Bài 6 (n = 8, 100 lần) cho evals trung bình 224,4 / 315,7 / 205,4 "
     "cho Steepest / Inefficient / Efficient; ở n = 4, Efficient lại tốn hơn vì patience 50 lớn hơn N = 12. Số chính thức và "
     "phân tích plateau sẽ nằm ở Bài 6."),
    ("6. Vì sao cần nhiều lần chạy và cố định điều kiện thí nghiệm khi đánh giá thuật toán ngẫu nhiên?",
     "Khởi tạo và nước đi ngẫu nhiên biến kết quả thành một phân bố, không phải một con số cố định. Một lần chạy chỉ là một "
     "mẫu; có thể gặp may hoặc rất xui. Cần nhiều lần để ước lượng tỷ lệ thành công, trung bình và trung vị ổn định hơn. "
     "Đồng thời phải cố định n, seed, cùng tập đồ thị/bàn đầu cho mọi thuật toán, cùng tiêu chí dừng và cùng cách đếm eval; "
     "nếu thay đổi những yếu tố này giữa các thuật toán thì chênh lệch quan sát được có thể chỉ là nhiễu. Bài 4 dùng 30 đồ thị "
     "chung cho BT/FC/MC với seed 2024; Bài 6 (bàn giao) dùng 100 bàn đầu chung cho ba biến thể HC."),
]
for q, a in qa:
    p(q, bold=True, size=13.5, before=6, after=2)
    p(a, size=13.5, indent=1.0)

# ---------------------------------------------------------------------------
# Tai lieu, ma nguon
# ---------------------------------------------------------------------------
h1("TÀI LIỆU, MÃ NGUỒN VÀ CHẠY LẠI")
p("Mã nguồn và minh chứng trong thư mục lab04-PhanĐứcNhân-Bài 1,Bài 4:", bold=True, before=2)
bullet("bai1_csp_coloring.py — sinh đồ thị, dựng CSP, kiểm thử complete/consistent, chạy Backtracking 4/3 màu, xuất JSON/log/hình.")
bullet("bai4_fc_minconflicts.py — cài đặt Forward Checking + MRV, Min-Conflicts (thuần và walk 5%), chạy 30 đồ thị mỗi n, xuất CSV/JSON/hình.")
bullet("output/bai1_ketqua.json, output/bai1_log.txt — kết quả và log Bài 1.")
bullet("output/bai4_runs.csv, output/bai4_summary.json — kết quả thô và tổng hợp Bài 4.")
bullet("images/ — 3 hình Bài 1 (đồ thị, tô 4 màu, K4 khi 3 màu) và 5 hình Bài 4 (4 màu nodes/time, min-conflicts, 3 màu vô nghiệm, bipyramid, ví dụ tô màu n = 20).")
p("Bàn giao Bài 6 cho Trần Trọng Nghĩa: thư mục ../lab04_TrầnTrọngNghĩa-Bài 5, Bài 6/bai6_hill_climbing/ chứa script, "
  "4 hình và output của một lần chạy thử Task 1–3. Đây là bản tham khảo, không phải kết quả chính thức của báo cáo này.",
  italic=True, size=12)
p("Chạy lại từ đầu (không phụ thuộc đường dẫn cá nhân):", bold=True, before=6)
code_block([
    "cd \"lab04-PhanĐứcNhân-Bài 1,Bài 4\"",
    "python3 -m venv .venv",
    "./.venv/bin/pip install numpy scipy matplotlib python-docx",
    "./.venv/bin/python bai1_csp_coloring.py",
    "./.venv/bin/python bai4_fc_minconflicts.py",
    "./.venv/bin/python tao_baocao_lab04.py",
])
p("Thư viện: numpy, scipy (Delaunay), matplotlib, python-docx. Phiên bản đã kiểm: NumPy 2.5.4, SciPy 1.18.1, "
  "Matplotlib 3.11.2, Python 3.13.14 trên macOS. Không dùng thư viện tìm kiếm/agent có sẵn; toàn bộ thuật toán "
  "CSP và tìm kiếm cục bộ được cài đặt trực tiếp.", indent=1.0)

p("Tài liệu tham khảo:", bold=True, before=8)
refs = [
    "[1] Russell, S., Norvig, P. (2021). Artificial Intelligence: A Modern Approach, 4th ed., Chapter 6 – "
    "Constraint Satisfaction Problems. Pearson.",
    "[2] Dechter, R. (2003). Constraint Processing. Morgan Kaufmann.",
    "[3] Kirkpatrick, S., Gelatt, C. D., Vecchi, M. P. (1983). Optimization by Simulated Annealing. "
    "Science, 220(4598), 671–680.",
    "[4] Appel, K., Haken, W. (1977). Every planar map is four colorable. Illinois Journal of Mathematics, 21, 429–490.",
    "[5] SciPy documentation: scipy.spatial.Delaunay; NumPy documentation: numpy.random.Generator.",
]
for r in refs:
    par = doc.add_paragraph()
    par.paragraph_format.left_indent = Cm(0.6)
    par.paragraph_format.first_line_indent = Cm(-0.6)
    par.paragraph_format.space_after = Pt(3)
    run = par.add_run(r)
    set_run(run, size=12)

# ---------------------------------------------------------------------------
# Appendix
# ---------------------------------------------------------------------------
h1("PHỤ LỤC HÌNH MINH HỌA")
p("Một số hình được lặp lại từ phần thân để tiện tra cứu khi ghép báo cáo nhóm.", italic=True, size=12)
figure("bai1_graph_delaunay.png", "Hình PL.1. Đồ thị Delaunay n = 10, seed 1111 — dữ liệu gốc của Bài 1.", 13.0)
figure("bai1_coloring_4.png", "Hình PL.2. Lời giải 4 màu hợp lệ cho đồ thị Bài 1.", 13.0)
figure("bai4_coloring_example.png", "Hình PL.3. Lời giải 4 màu cho đồ thị Delaunay n = 20 do Forward Checking tìm được.", 12.0)
figure("bai4_3color_unsolvable.png", "Hình PL.4. 3 màu: chi phí chứng minh vô nghiệm của Backtracking và FC.", 13.5)
figure("bai4_bipyramid_3color.png", "Hình PL.5. Họ bipyramid 3 màu — kiểm chứng trường hợp chắc chắn có nghiệm.", 15.0)

p("— Hết —", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, before=12, indent=0)

doc.save(DOCX)
print("Da tao:", DOCX)

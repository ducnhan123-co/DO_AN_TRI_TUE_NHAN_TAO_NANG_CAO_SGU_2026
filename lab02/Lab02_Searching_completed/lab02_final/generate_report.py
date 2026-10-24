import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_report():
    doc = Document()
    
    # Page setup: Standard A4 margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        
    # Set default style font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(30, 30, 30)

    # ==========================
    # TRANG BÌA (COVER PAGE)
    # ==========================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_uni = p_inst.add_run("TRƯỜNG ĐẠI HỌC SÀI GÒN\n")
    r_uni.bold = True
    r_uni.font.size = Pt(14)
    r_fac = p_inst.add_run("KHOA CÔNG NGHỆ THÔNG TIN\n")
    r_fac.bold = True
    r_fac.font.size = Pt(13)
    p_inst.add_run("-----------------------***-----------------------\n\n\n")

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_type = p_title.add_run("BÁO CÁO THỰC HÀNH LAB 02\n")
    r_type.bold = True
    r_type.font.size = Pt(18)
    r_type.font.color.rgb = RGBColor(0, 51, 102)
    
    r_main_title = p_title.add_run("GIẢI QUYẾT BÀI TOÁN MÊ CUNG VỚI TÁC TỬ TÌM KIẾM DỰA TRÊN MỤC TIÊU\n(GOAL-BASED SEARCHING AGENT)\n\n")
    r_main_title.bold = True
    r_main_title.font.size = Pt(16)
    r_main_title.font.color.rgb = RGBColor(180, 0, 0)
    
    r_sub = p_title.add_run("Môn học: Trí tuệ Nhân tạo Nâng cao\n\n\n\n")
    r_sub.font.size = Pt(13)
    r_sub.italic = True

    # Info table
    table_info = doc.add_table(rows=4, cols=2)
    table_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_info.autofit = False
    
    info_data = [
        ("Giảng viên hướng dẫn:", "TS./ThS. Đỗ Như Tài"),
        ("Sinh viên thực hiện:", "Nguyễn Đăng Khoa"),
        ("Mã số sinh viên (MSSV):", "3123410168"),
        ("Khóa học / Niên khóa:", "2023 - 2028")
    ]
    
    for i, (k, v) in enumerate(info_data):
        row = table_info.rows[i]
        c0 = row.cells[0]
        c1 = row.cells[1]
        c0.width = Inches(2.5)
        c1.width = Inches(3.5)
        
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.size = Pt(12)
        
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(12)
        
    doc.add_paragraph("\n\n\n\n\n")
    p_loc = doc.add_paragraph()
    p_loc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_loc = p_loc.add_run("TP. HỒ CHÍ MINH, THÁNG 10/2026")
    r_loc.bold = True
    r_loc.font.size = Pt(12)

    doc.add_page_break()

    # ==========================
    # MỤC LỤC
    # ==========================
    h_toc = doc.add_heading("MỤC LỤC", level=1)
    h_toc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_toc.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    toc_items = [
        ("PHẦN 1. ĐẶC TẢ BÀI TOÁN TÌM KIẾM TRONG MÊ CUNG", "3"),
        ("    1.1. Khái quát bài toán và mô hình tác tử", "3"),
        ("    1.2. Thành phần bài toán tìm kiếm (Search Problem Formulation)", "3"),
        ("    1.3. Ước lượng không gian bài toán (Problem-size Estimates)", "3"),
        ("PHẦN 2. BẢNG KẾT QUẢ THỰC NGHIỆM VÀ SO SÁNH CÁC THUẬT TOÁN", "4"),
        ("    2.1. Cấu hình thử nghiệm", "4"),
        ("    2.2. Bảng kết quả tổng hợp trên các mê cung", "4"),
        ("    2.3. Biểu đồ trực quan hóa số lượng nút mở rộng và chi phí đường đi", "5"),
        ("PHẦN 3. MINH HỌA ĐƯỜNG ĐI VÀ VÙNG TRẠNG THÁI TÌM KIẾM", "6"),
        ("    3.1. Minh họa trên Small Maze: BFS so sánh với DFS", "6"),
        ("    3.2. Minh họa trên Medium Maze & Open Maze: A* so sánh với GBFS", "7"),
        ("PHẦN 4. NHẬN XÉT VÀ ĐÁNH GIÁ CHUYÊN SÂU (1-2 TRANG)", "8"),
        ("    4.1. Tính đầy đủ (Completeness)", "8"),
        ("    4.2. Tính tối ưu (Optimality)", "8"),
        ("    4.3. Độ phức tạp thời gian thực thi (Time Complexity)", "8"),
        ("    4.4. Độ phức tạp không gian và sử dụng bộ nhớ (Space Complexity)", "9"),
        ("    4.5. Vai trò và chất lượng của hàm Heuristic (Manhattan vs. Euclidean)", "9"),
        ("    4.6. Kiểm tra chu trình và xử lý trùng lặp trạng thái (Cycle Checking)", "10"),
        ("PHẦN 5. MỞ RỘNG NÂNG CAO (ADVANCED TASKS)", "10"),
        ("    5.1. Thuật toán tìm kiếm sâu lặp (Iterative Deepening Search - IDS)", "10"),
        ("    5.2. Mê cung với nhiều mục tiêu (Multiple Goals)", "10"),
        ("    5.3. Tìm kiếm A* có trọng số (Weighted A*)", "11"),
        ("KẾT LUẬN VÀ BÀI HỌC KINH NGHIỆM", "11")
    ]

    for item, page in toc_items:
        p_t = doc.add_paragraph()
        p_t.paragraph_format.line_spacing = 1.2
        r_item = p_t.add_run(item)
        r_item.font.size = Pt(11)
        if item.startswith("PHẦN") or item.startswith("KẾT"):
            r_item.bold = True
        p_t.add_run(" " + "." * max(2, 70 - len(item)) + " " + page)

    doc.add_page_break()

    # ==========================
    # NỘI DUNG BÁO CÁO
    # ==========================
    
    # PHẦN 1
    h1 = doc.add_heading("PHẦN 1. ĐẶC TẢ BÀI TOÁN TÌM KIẾM TRONG MÊ CUNG", level=1)
    h1.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    doc.add_heading("1.1. Khái quát bài toán và mô hình tác tử", level=2)
    doc.add_paragraph(
        "Trong bài tập Lab 02, chúng ta xây dựng tác tử dựa trên mục tiêu (Goal-based Agent) để giải quyết bài toán tìm đường đi "
        "từ điểm xuất phát 'S' (Start) đến điểm đích 'G' (Goal) trên bản đồ mê cung dạng lưới 2 chiều (Grid Map). "
        "Môi trường là môi trường quan sát toàn phần (fully observable), tiền định (deterministic), tĩnh (static) và rời rạc (discrete)."
    )

    doc.add_heading("1.2. Thành phần bài toán tìm kiếm (Search Problem Formulation)", level=2)
    doc.add_paragraph(
        "Theo chuẩn giáo trình trí tuệ nhân tạo (Russell & Norvig, AIMA), bài toán tìm kiếm bao gồm 5 thành phần cốt lõi:"
    )
    
    components = [
        ("Trạng thái ban đầu (Initial State):", "Tọa độ vị trí xuất phát s0 = (r0, c0) của ký tự 'S' trên mê cung, được xác định qua hàm find_pos(maze, 'S')."),
        ("Không gian hành động (Actions):", "Tập 4 hướng di chuyển cơ bản Actions(s) = {North, South, East, West} tương ứng với độ lệch tọa độ (-1, 0), (1, 0), (0, 1), (0, -1). Hành động là hợp lệ nếu ô đích nằm trong biên bản đồ và không phải là tường chắn ('X')."),
        ("Mô hình chuyển trạng thái (Transition Model):", "Hàm tất định Result(s, a) = s', trả về tọa độ ô kế tiếp sau khi thực hiện hành động a từ trạng thái s."),
        ("Kiểm tra đích (Goal Test):", "Hàm kiểm tra IsGoal(s) trả về True nếu s == (r_goal, c_goal) ứng với vị trí ô 'G'."),
        ("Chi phí đường đi (Path Cost):", "Hàm chi phí bước c(s, a, s') = 1 cho mọi bước di chuyển hợp lệ (unit cost). Tổng chi phí g(n) là tổng số bước đi từ điểm bắt đầu đến trạng thái hiện tại.")
    ]
    for c_title, c_desc in components:
        p_c = doc.add_paragraph(style='List Bullet')
        r_ct = p_c.add_run(c_title + " ")
        r_ct.bold = True
        p_c.add_run(c_desc)

    doc.add_heading("1.3. Ước lượng kích thước bài toán (Problem-size Estimates)", level=2)
    doc.add_paragraph(
        "Kích thước bài toán được xác định thông qua các thông số lý thuyết và thực nghiệm:"
    )
    p_size1 = doc.add_paragraph(style='List Bullet')
    p_size1.add_run("n (Kích thước không gian trạng thái): ").bold = True
    p_size1.add_run("Tổng số ô có thể đi lại được (bao gồm các ô trống ' ', ô 'S' và ô 'G'). Ví dụ: small_maze có n = 94 trạng thái, large_maze có n = 630 trạng thái, open_maze có n = 687 trạng thái.")
    
    p_size2 = doc.add_paragraph(style='List Bullet')
    p_size2.add_run("d (Độ sâu nghiệm tối ưu): ").bold = True
    p_size2.add_run("Số bước đi ngắn nhất từ 'S' đến 'G'. Vì chi phí mỗi bước là 1, d được xác định chính xác thông qua nghiệm của thuật toán BFS (ví dụ: small_maze có d = 19, large_maze có d = 210, open_maze có d = 54).")

    p_size3 = doc.add_paragraph(style='List Bullet')
    p_size3.add_run("b (Hệ số phân nhánh tối đa): ").bold = True
    p_size3.add_run("Tại mỗi ô trên lưới 4 hướng, tối đa tác tử có thể rẽ sang 4 hướng lân cận, do đó b <= 4 (trong các hành lang hẹp b thường là 1 hoặc 2, tại ngã ba/ngã tư b là 3 hoặc 4).")

    p_size4 = doc.add_paragraph(style='List Bullet')
    p_size4.add_run("m (Độ sâu tối đa của cây tìm kiếm): ").bold = True
    p_size4.add_run("Khi áp dụng cơ chế kiểm tra chu trình, độ dài của đường đi đơn tối đa không vượt quá tổng số ô đi lại được, tức là m <= n.")

    # PHẦN 2
    h2 = doc.add_heading("PHẦN 2. BẢNG KẾT QUẢ THỰC NGHIỆM VÀ SO SÁNH CÁC THUẬT TOÁN", level=1)
    h2.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    doc.add_heading("2.1. Cấu hình thử nghiệm", level=2)
    doc.add_paragraph(
        "Thử nghiệm được thực hiện trên 8 tập dữ liệu mê cung với cấu trúc đa dạng (hành lang hẹp, không gian mở, chu trình): "
        "small_maze, medium_maze, large_maze, open_maze, L_maze, loops_maze, empty_maze, empty_maze_2. "
        "Bốn thuật toán tìm kiếm được so sánh trực tiếp gồm: BFS (Breadth-First Search), DFS (Depth-First Search với path-based cycle checking), "
        "GBFS (Greedy Best-First Search với heuristic khoảng cách Manhattan), và A* Search (f = g + h Manhattan)."
    )

    doc.add_heading("2.2. Bảng kết quả tổng hợp trên các mê cung", level=2)

    # Comparison Table
    headers = ["Mê cung", "Thuật toán", "Path Cost", "Số nút mở rộng", "Max Tree Depth", "Max Nodes Memory", "Max Frontier"]
    data = [
        # Small maze
        ["small", "BFS", "19", "93", "20", "96", "9"],
        ["small", "DFS", "37", "38", "37", "44", "7"],
        ["small", "GBFS", "29", "40", "29", "49", "5"],
        ["small", "A*", "19", "53", "19", "66", "8"],
        # Medium maze
        ["medium", "BFS", "68", "269", "68", "274", "9"],
        ["medium", "DFS", "74", "75", "74", "82", "8"],
        ["medium", "GBFS", "74", "79", "74", "86", "4"],
        ["medium", "A*", "68", "222", "68", "235", "8"],
        # Large maze
        ["large", "BFS", "210", "621", "211", "630", "8"],
        ["large", "DFS", "210", "418", "222", "257", "35"],
        ["large", "GBFS", "210", "468", "210", "507", "20"],
        ["large", "A*", "210", "550", "210", "565", "12"],
        # Open maze
        ["open", "BFS", "54", "683", "54", "687", "25"],
        ["open", "DFS", "206", "207", "206", "405", "199"],
        ["open", "GBFS", "68", "90", "68", "221", "66"],
        ["open", "A*", "54", "536", "54", "576", "25"],
        # Loops maze
        ["loops", "BFS", "22", "71", "22", "74", "7"],
        ["loops", "DFS", "32", "33", "32", "39", "12"],
        ["loops", "GBFS", "24", "68", "24", "71", "10"],
        ["loops", "A*", "22", "64", "22", "67", "6"],
        # Empty maze
        ["empty", "BFS", "14", "96", "14", "101", "12"],
        ["empty", "DFS", "68", "69", "68", "169", "81"],
        ["empty", "GBFS", "14", "69", "14", "72", "29"],
        ["empty", "A*", "14", "96", "14", "125", "31"]
    ]

    tbl = doc.add_table(rows=len(data) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True

    # Header row formatting
    hdr_cells = tbl.rows[0].cells
    for j, h_text in enumerate(headers):
        hdr_cells[j].text = h_text
        set_cell_background(hdr_cells[j], "003366")
        set_cell_margins(hdr_cells[j], top=120, bottom=120, left=100, right=100)
        p = hdr_cells[j].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.bold = True
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(255, 255, 255)

    # Data rows formatting
    for i, row_data in enumerate(data):
        row_cells = tbl.rows[i + 1].cells
        bg = "F2F5F8" if (i // 4) % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row_data):
            row_cells[j].text = val
            set_cell_background(row_cells[j], bg)
            set_cell_margins(row_cells[j], top=80, bottom=80, left=100, right=100)
            p = row_cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.size = Pt(9.5)
                if val == "A*" or (j == 2 and val in ["19", "68", "210", "54", "22", "14"]):
                    r.bold = True

    doc.add_paragraph("\nBảng 1: Bảng tổng hợp so sánh các chỉ số hiệu năng của BFS, DFS, GBFS và A* trên các mê cung tiêu biểu.")

    doc.add_page_break()

    # PHẦN 3
    h3 = doc.add_heading("PHẦN 3. MINH HỌA ĐƯỜNG ĐI VÀ VÙNG TRẠNG THÁI TÌM KIẾM", level=1)
    h3.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    doc.add_paragraph(
        "Dưới đây là hình ảnh kết quả chạy thuật toán được trích xuất trực tiếp từ mã nguồn thực nghiệm. "
        "Trong các hình vẽ: ký tự 'S' (màu xanh dương) là điểm xuất phát, 'G' (màu xanh lá) là điểm đích, "
        "các ô màu xám '.' biểu thị vùng không gian trạng thái đã được khám phá (reached/explored states), "
        "và các ô màu đỏ 'P' biểu thị đường đi kết quả (solution path) được tác tử lựa chọn."
    )

    doc.add_heading("3.1. Minh họa trên Small Maze: BFS so sánh với DFS", level=2)
    
    # Insert images for Small Maze
    t_img1 = doc.add_table(rows=1, cols=2)
    t_img1.alignment = WD_TABLE_ALIGNMENT.CENTER
    c0 = t_img1.rows[0].cells[0]
    c1 = t_img1.rows[0].cells[1]
    
    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if os.path.exists('fig_small_bfs.png'):
        p0.add_run().add_picture('fig_small_bfs.png', width=Inches(3.0))
    p0.add_run("\nHình 1(a): BFS trên small_maze\n(Đường đi tối ưu d=19, duyệt 93 ô)").italic = True
    
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if os.path.exists('fig_small_dfs.png'):
        p1.add_run().add_picture('fig_small_dfs.png', width=Inches(3.0))
    p1.add_run("\nHình 1(b): DFS trên small_maze\n(Đường đi dài d=37, duyệt 38 ô)").italic = True

    doc.add_paragraph(
        "Nhận xét hình ảnh 1: BFS quét theo làn sóng đồng tâm (vùng xám phủ gần kín 93 ô) và tìm được đường đi ngắn nhất tuyệt đối (19 bước). "
        "Ngược lại, DFS lao sâu theo nhánh đầu tiên, chỉ mất 38 nút mở rộng nhưng đường đi tìm được dài tới 37 bước, ngoằn ngoèo và không tối ưu."
    )

    doc.add_heading("3.2. Minh họa trên Medium Maze & Open Maze: A* so sánh với GBFS", level=2)
    
    t_img2 = doc.add_table(rows=1, cols=2)
    t_img2.alignment = WD_TABLE_ALIGNMENT.CENTER
    c2_0 = t_img2.rows[0].cells[0]
    c2_1 = t_img2.rows[0].cells[1]
    
    p2_0 = c2_0.paragraphs[0]
    p2_0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if os.path.exists('fig_medium_astar.png'):
        p2_0.add_run().add_picture('fig_medium_astar.png', width=Inches(3.0))
    p2_0.add_run("\nHình 2(a): A* trên medium_maze\n(Đường đi tối ưu d=68, duyệt 222 ô)").italic = True

    p2_1 = c2_1.paragraphs[0]
    p2_1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if os.path.exists('fig_open_astar.png'):
        p2_1.add_run().add_picture('fig_open_astar.png', width=Inches(3.0))
    p2_1.add_run("\nHình 2(b): A* trên open_maze\n(Đường đi tối ưu d=54, có định hướng)").italic = True

    doc.add_paragraph(
        "Nhận xét hình ảnh 2: Thuật toán A* nhờ hàm đánh giá f(n) = g(n) + h(n) đã định hướng tìm kiếm mạnh mẽ về phía Goal, "
        "cắt tỉa đáng kể các nhánh đi ngược hướng so với BFS mù quáng. Trên medium_maze, A* giảm số nút mở rộng từ 269 (BFS) xuống 222 mà vẫn giữ đúng độ dài tối ưu 68."
    )

    doc.add_page_break()

    # PHẦN 4
    h4 = doc.add_heading("PHẦN 4. NHẬN XÉT VÀ ĐÁNH GIÁ CHUYÊN SÂU (1-2 TRANG)", level=1)
    h4.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    doc.add_heading("4.1. Tính đầy đủ (Completeness)", level=2)
    p_comp = doc.add_paragraph()
    p_comp.add_run(
        "Tính đầy đủ thể hiện khả năng thuật toán luôn tìm ra giải pháp nếu giải pháp tồn tại, hoặc thông báo thất bại nếu không có đường đi.\n"
    )
    p_comp.add_run("• BFS (Breadth-First Search): ").bold = True
    p_comp.add_run(
        "Luôn ĐẦY ĐỦ trong không gian trạng thái hữu hạn hoặc khi chi phí mỗi bước c >= epsilon > 0. Do BFS duyệt theo từng mức độ sâu tăng dần, "
        "khi tồn tại nghiệm ở độ sâu d hữu hạn, BFS chắc chắn sẽ chạm tới trước khi xét các mức sâu hơn.\n"
    )
    p_comp.add_run("• DFS (Depth-First Search): ").bold = True
    p_comp.add_run(
        "Chỉ ĐẦY ĐỦ trong không gian hữu hạn khi và chỉ khi có cơ chế kiểm tra chu trình (Cycle Checking). Nếu là Tree Search thuần túy không kiểm tra chu trình, "
        "DFS sẽ lập tức rơi vào lặp vô tận (infinite loop) ngay cả trong mê cung nhỏ có các ô qua lại (A <-> B). Trong bài làm, chúng ta cài đặt cơ chế "
        "kiểm tra chu trình trên đường đi hiện tại (path-based cycle checking), do đó DFS đầy đủ trên mọi mê cung hữu hạn thử nghiệm.\n"
    )
    p_comp.add_run("• GBFS (Greedy Best-First Search): ").bold = True
    p_comp.add_run(
        "Trong bài tập, GBFS được cài đặt dạng Graph Search (với tập reached), do đó đầy đủ trên không gian hữu hạn. Tuy nhiên theo lý thuyết, "
        "nếu chạy Tree Search trong không gian vô hạn hoặc có chu trình, GBFS có thể không đầy đủ vì bị heuristic đánh lừa dẫn vào nhánh cụt vô hạn.\n"
    )
    p_comp.add_run("• A* Search: ").bold = True
    p_comp.add_run(
        "Hoàn toàn ĐẦY ĐỦ trên đồ thị hữu hạn khi hàm heuristic h(n) hữu hạn và chi phí bước c(s, a, s') = 1 > 0."
    )

    doc.add_heading("4.2. Tính tối ưu (Optimality)", level=2)
    p_opt = doc.add_paragraph()
    p_opt.add_run("• BFS: ").bold = True
    p_opt.add_run(
        "ĐẢM BẢO TỐI ƯU TUYỆT ĐỐI vì trong bài toán này chi phí mọi bước di chuyển đều bằng 1 (unit step cost). "
        "Độ dài đường đi BFS tìm được chính là khoảng cách ngắn nhất d.\n"
    )
    p_opt.add_run("• DFS: ").bold = True
    p_opt.add_run(
        "HOÀN TOÀN KHÔNG TỐI ƯU. DFS phụ thuộc hoàn toàn vào thứ tự ưu tiên sinh con (ở đây là N -> E -> S -> W). "
        "Nó sẽ chọn đi sâu hết cỡ vào nhánh đầu tiên tìm thấy. Số liệu thực nghiệm chứng minh: trên small_maze, đường đi tối ưu là 19, "
        "nhưng DFS đi mất 37 bước. Trên open_maze, đường đi tối ưu là 54, nhưng DFS đi tới 206 bước (dài gần gấp 4 lần) do đi men theo chu vi tường.\n"
    )
    p_opt.add_run("• GBFS: ").bold = True
    p_opt.add_run(
        "KHÔNG TỐI ƯU. Do GBFS chỉ nhìn vào h(n) (khoảng cách ước lượng đến đích) mà hoàn toàn 'bỏ quên' chi phí đã đi qua g(n). "
        "Nó có thể chọn một bước đi có vẻ gần đích hơn về mặt hình học nhưng thực chất lại dẫn vào ngõ cụt hoặc đường vòng dài. "
        "Ví dụ trên small_maze: GBFS tốn 29 bước (trong khi tối ưu là 19); trên open_maze: GBFS tốn 68 bước (tối ưu là 54).\n"
    )
    p_opt.add_run("• A* Search: ").bold = True
    p_opt.add_run(
        "ĐẢM BẢO TỐI ƯU TUYỆT ĐỐI. Vì hàm heuristic khoảng cách Manhattan h(n) là hàm chấp nhận được (admissible) và nhất quán (consistent) "
        "trên lưới di chuyển 4 hướng không đi chéo. Kết quả thực nghiệm cho thấy A* luôn luôn cho ra Path Cost bằng đúng BFS trên 100% các mê cung: "
        "small (19), medium (68), large (210), open (54), loops (22), empty (14)."
    )

    doc.add_heading("4.3. Độ phức tạp thời gian thực thi (Time Complexity)", level=2)
    p_time = doc.add_paragraph()
    p_time.add_run(
        "Độ phức tạp thời gian được phản ánh rõ nét qua số lượng nút mở rộng (nodes expanded) và thời gian thực thi (CPU time):\n"
    )
    p_time.add_run("• BFS: ").bold = True
    p_time.add_run(
        "Thời gian tệ nhất là O(b^d). Trên thực tế, BFS phải mở rộng gần như toàn bộ các nút ở các tầng trước d. "
        "Tại open_maze, BFS mở rộng tới 683 nút (gần như toàn bộ 687 ô trống của bản đồ).\n"
    )
    p_time.add_run("• DFS: ").bold = True
    p_time.add_run(
        "Thời gian tệ nhất là O(b^m). Tuy nhiên, nếu may mắn chọn đúng nhánh hướng về đích, DFS có thể chạm đích rất nhanh với số nút mở rộng rất ít "
        "(trên small_maze chỉ mở rộng 38 nút so với 93 của BFS; trên medium_maze chỉ mở rộng 75 nút so với 269 của BFS).\n"
    )
    p_time.add_run("• GBFS: ").bold = True
    p_time.add_run(
        "Thời gian phụ thuộc lớn vào chất lượng hàm heuristic. Khi mê cung ít chướng ngại vật cản đường (như open_maze), "
        "GBFS lao thẳng về đích như một mũi tên và chỉ cần mở rộng 90 nút (so với 683 của BFS và 536 của A*).\n"
    )
    p_time.add_run("• A*: ").bold = True
    p_time.add_run(
        "Nhờ kết hợp f(n) = g(n) + h(n), A* tập trung vùng tìm kiếm vào một 'hình elip' hướng về phía đích. "
        "Số nút mở rộng của A* luôn thấp hơn hoặc bằng BFS trên mọi mê cung (small: 53 vs 93; medium: 222 vs 269; large: 550 vs 621; open: 536 vs 683)."
    )

    doc.add_heading("4.4. Độ phức tạp không gian và sử dụng bộ nhớ (Space Complexity)", level=2)
    p_space = doc.add_paragraph()
    p_space.add_run(
        "Bộ nhớ là thước đo phân định sự khác biệt then chốt giữa các giải thuật tìm kiếm:\n"
    )
    p_space.add_run("• Nút thắt cổ chai của BFS: ").bold = True
    p_space.add_run(
        "BFS có độ phức tạp bộ nhớ là O(b^d). Toàn bộ biên (frontier) và tập reached phải lưu lại trong RAM. "
        "Số nút cực đại trong bộ nhớ của BFS luôn xấp xỉ tổng số trạng thái của mê cung (large: 630 nút, open: 687 nút). "
        "Trong các không gian trạng thái lớn (hàng triệu trạng thái), BFS sẽ tràn bộ nhớ (Out-Of-Memory) trước khi hết thời gian chạy.\n"
    )
    p_space.add_run("• Ưu thế vượt trội của DFS: ").bold = True
    p_space.add_run(
        "DFS phiên bản Tree Search chỉ lưu các nút trên nhánh hiện tại và các nhánh rẽ dự phòng, độ phức tạp bộ nhớ chỉ là O(b * m). "
        "Trên large_maze, trong khi BFS cần lưu tới 630 nút thì DFS chỉ chiếm cực đại 257 nút trong bộ nhớ. "
        "Đây là lý do DFS là nền tảng cho các thuật toán tìm kiếm tiết kiệm bộ nhớ như IDA* hay RBFS."
    )

    doc.add_heading("4.5. Vai trò và chất lượng của hàm Heuristic (Manhattan vs. Euclidean)", level=2)
    p_heu = doc.add_paragraph()
    p_heu.add_run(
        "Hàm heuristic h(n) đóng vai trò kim chỉ nam dẫn đường cho các thuật toán có thông tin:\n"
    )
    p_heu.add_run("• Khoảng cách Manhattan (L1 Norm): ").bold = True
    p_heu.add_run("h_Manhattan((r1, c1), (r2, c2)) = |r1 - r2| + |c1 - c2|.\n")
    p_heu.add_run("• Khoảng cách Euclidean (L2 Norm): ").bold = True
    p_heu.add_run("h_Euclidean((r1, c1), (r2, c2)) = sqrt((r1 - r2)^2 + (c1 - c2)^2).\n")
    p_heu.add_run("• Đánh giá tính chất: ").bold = True
    p_heu.add_run(
        "Vì bài toán di chuyển trên lưới 4 hướng (chỉ đi ngang và dọc, không đi chéo), bước đi ngắn nhất giữa hai ô kế nhau luôn là 1. "
        "Do đó, h_Manhattan(n) biểu diễn chính xác số bước đi tối thiểu nếu không có vật cản, nên h_Manhattan là admissible (không bao giờ đánh giá quá cao chi phí thực tế) "
        "và consistent (thỏa mãn bất đẳng thức tam giác h(n) <= c(n, a, n') + h(n')). "
        "Mặt khác, trong hình học ta luôn có h_Euclidean(n) <= h_Manhattan(n) <= h*(n). "
        "Điều này có nghĩa là Manhattan 'thống trị' (dominates) Euclidean. Sử dụng Manhattan heuristic cho phép A* cắt tỉa không gian tìm kiếm "
        "chặt chẽ hơn, mở rộng ít trạng thái dư thừa hơn so với Euclidean."
    )

    doc.add_heading("4.6. Kiểm tra chu trình và xử lý trùng lặp trạng thái (Cycle Checking)", level=2)
    p_cycle = doc.add_paragraph()
    p_cycle.add_run(
        "Trong bài toán mê cung, đồ thị có vô số chu trình (ví dụ đi vòng quanh 1 bức tường hoặc đi qua lại giữa 2 ô trống). "
        "Nếu không có kiểm tra chu trình, thuật toán sẽ duyệt lặp lại vô tận. Chúng ta phân biệt 2 kỹ thuật kiểm tra chu trình:\n"
    )
    p_cycle.add_run("1. Path-based cycle checking (Kiểm tra trên đường đi): ").bold = True
    p_cycle.add_run(
        "Chỉ kiểm tra xem trạng thái con nxt có nằm trong chuỗi các nút tổ tiên từ gốc đến nút hiện tại hay không (nxt not in path_positions). "
        "Ưu điểm: Không cần tập lưu trữ toàn cục, giữ nguyên độ phức tạp bộ nhớ tuyến tính O(bm) của DFS. "
        "Nhược điểm: Vẫn có thể duyệt lại cùng một ô nếu ô đó được tiếp cận từ một nhánh rẽ khác.\n"
    )
    p_cycle.add_run("2. Global reached set (Tập trạng thái toàn cục - Graph Search): ").bold = True
    p_cycle.add_run(
        "Lưu mọi trạng thái đã từng được phát hiện vào một tập hash set (reached). "
        "Ưu điểm: Ngăn chặn 100% việc duyệt lại các trạng thái trùng lặp, đảm bảo mỗi ô chỉ được đưa vào frontier tối đa 1 lần trong BFS. "
        "Nhược điểm: Tốn bộ nhớ O(|V|) tỷ lệ thuận với số ô trống của mê cung."
    )

    # PHẦN 5
    h5 = doc.add_heading("PHẦN 5. MỞ RỘNG NÂNG CAO (ADVANCED TASKS)", level=1)
    h5.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    doc.add_heading("5.1. Thuật toán tìm kiếm sâu lặp (Iterative Deepening Search - IDS)", level=2)
    doc.add_paragraph(
        "IDS giải quyết hoàn hảo bài toán đánh đổi giữa BFS và DFS bằng cách lặp lại thuật toán Depth-Limited Search với giới hạn độ sâu limit = 0, 1, 2, ... "
        "IDS kết hợp được ưu điểm bộ nhớ O(b * d) của DFS với tính tối ưu O(b^d) của BFS. "
        "Thực nghiệm trên small_maze: IDS tìm ra đúng đường đi tối ưu độ dài 19 với số nút mở rộng tích lũy là 431 nút."
    )

    doc.add_heading("5.2. Mê cung với nhiều mục tiêu (Multiple Goals)", level=2)
    doc.add_paragraph(
        "Khi mê cung có nhiều điểm đích G1, G2, ..., bài toán được giải quyết bằng cách sửa đổi hàm Goal Test: IsGoal(s) = (s in Goals). "
        "Thuật toán BFS sẽ đảm bảo tìm đến điểm đích gần nhất về mặt chi phí bước đi."
    )

    doc.add_heading("5.3. Tìm kiếm A* có trọng số (Weighted A*)", level=2)
    doc.add_paragraph(
        "Weighted A* sử dụng hàm đánh giá f(n) = g(n) + w * h(n) với w > 1.0. "
        "Khi tăng trọng số w, thuật toán nghiêng dần về phía tham lam (GBFS), giúp tăng tốc độ tìm kiếm và giảm mạnh số nút mở rộng, "
        "với cái giá phải trả là nghiệm tìm được có thể dài hơn nghiệm tối ưu một lượng không quá w * C*."
    )

    # KẾT LUẬN
    h_concl = doc.add_heading("KẾT LUẬN VÀ BÀI HỌC KINH NGHIỆM", level=1)
    h_concl.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    conclusions = [
        "1. Không có thuật toán tìm kiếm nào hoàn hảo tuyệt đối cho mọi tình huống: luôn tồn tại sự đánh đổi giữa tính tối ưu, thời gian tính toán và dung lượng bộ nhớ RAM.",
        "2. BFS là chuẩn mực để tìm đường đi ngắn nhất trong đồ thị có chi phí bước đồng nhất, nhưng bị giới hạn nghiêm ngặt bởi bộ nhớ.",
        "3. DFS có ưu thế vượt trội về tiết kiệm bộ nhớ nhưng cần cơ chế kiểm tra chu trình và không đảm bảo tìm ra đường đi tối ưu.",
        "4. Heuristic là cầu nối quan trọng đưa tri thức bài toán vào thuật toán: Một hàm heuristic admissible và consistent như Manhattan giúp A* đạt được cả hai mục tiêu: vừa tối ưu tuyệt đối như BFS, vừa tiết kiệm thời gian mở rộng nút."
    ]
    for c_text in conclusions:
        doc.add_paragraph(c_text)

    # Save document
    report_filename = "Bao_Cao_Lab02_Searching_NguyenDangKhoa_3123410168.docx"
    doc.save(report_filename)
    print(f"Report successfully generated and saved to: {report_filename}")

if __name__ == '__main__':
    create_report()

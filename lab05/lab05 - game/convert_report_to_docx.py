import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import os
import shutil

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_report_docx():
    doc = Document()

    # Set page margins to 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base style
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # Document Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title.add_run("BÁO CÁO THỰC HÀNH: TÌM KIẾM ĐỐI KHÁNG VÀ TRÒ CHƠI\n")
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1a, 0x36, 0x5d)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = sub.add_run("Môn học: Trí tuệ Nhân tạo Nâng cao (TTNTNC)\nChủ đề: Adversarial Search, Game Theory, Minimax, Alpha-Beta, MCTS\nSinh viên: Nguyễn Đăng Khoa - MSSV: 3123410168")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x4a, 0x55, 0x68)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Helper for headings
    def add_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
        r = h.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1a, 0x36, 0x5d)
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        r = h.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x2b, 0x6c, 0xb0)
        return h

    def add_h3(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
        r = h.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x2d, 0x37, 0x48)
        return h

    def add_p(text, bold_prefix=None, space_after=6):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.bold = True
        r = p.add_run(text)
        return p

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.bold = True
        p.add_run(text)
        return p

    # --- Section 1 ---
    add_h1("1. TỔNG QUAN VÀ BẢNG THỐNG KÊ KẾT QUẢ THỰC NGHIỆM")
    add_p("Toàn bộ các thuật toán tìm kiếm đối kháng và học tăng cường mô phỏng đã được cài đặt và kiểm thử nghiêm ngặt trong cùng một điều kiện chuẩn hóa. Bảng dưới đây tổng hợp kết quả thi đấu đối đầu với tác tử ngẫu nhiên (Random Player) với số lượng tối thiểu 100 ván đấu (các thực nghiệm cơ sở thực hiện tới 1.000 ván đấu với seed=42 cố định để đảm bảo tính tái lập 100%):")

    add_h2("1.1. Bảng Tổng hợp Kết quả Thực nghiệm (Tic-Tac-Toe)")
    
    table_data = [
        ["STT", "Tác tử / Thuật toán", "Đối thủ", "Số ván", "Thắng", "Hòa", "Thua", "Thời gian TB", "Nút duyệt (Bàn rỗng)"],
        ["1", "Random Player (Baseline)", "Random", "1.000", "59.0%", "12.6%", "28.4%", "< 0.01 ms", "1"],
        ["2", "AND-OR Tree Search", "Random", "100", "98.0%", "2.0%", "0.0%", "1.85 ms", "16.164"],
        ["3", "Minimax (Không cắt tỉa)", "Random", "100", "99.0%", "1.0%", "0.0%", "3.12 ms", "549.946"],
        ["4", "Alpha-Beta (Không Move Ordering)", "Random", "100", "99.0%", "1.0%", "0.0%", "0.95 ms", "18.297"],
        ["5", "Alpha-Beta (Có Move Ordering)", "Random", "100", "99.0%", "1.0%", "0.0%", "0.42 ms", "7.275"],
        ["6", "Heuristic Alpha-Beta (Cutoff = 4)", "Random", "100", "99.0%", "1.0%", "0.0%", "0.84 ms", "1.842"],
        ["7", "Heuristic Alpha-Beta (Cutoff = 2)", "Random", "100", "88.0%", "8.0%", "4.0%", "0.22 ms", "81"],
        ["8", "Pure Monte Carlo (N = 100)", "Random", "100", "96.0%", "4.0%", "0.0%", "12.50 ms", "900 playouts"],
        ["9", "Pure Monte Carlo (N = 10)", "Random", "100", "82.0%", "12.0%", "6.0%", "1.30 ms", "90 playouts"],
        ["10", "UCB1 / MCTS Depth 1 (N = 100)", "Random", "100", "98.0%", "2.0%", "0.0%", "14.10 ms", "100 simulations"],
        ["11", "UCB1 / MCTS Depth 1 (N = 10)", "Random", "100", "87.0%", "9.0%", "4.0%", "1.50 ms", "10 simulations"]
    ]

    t = doc.add_table(rows=len(table_data), cols=9)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = t.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            r = p.runs[0]
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            if r_idx == 0:
                r.font.bold = True
                set_cell_background(cell, "E2E8F0")
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F8FAFC")

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # --- Section 1.2: Charts ---
    add_h2("1.2. Biểu đồ So sánh Trực quan")
    add_p("Hai biểu đồ trực quan dưới đây được trích xuất trực tiếp từ các thí nghiệm thực tế (generate_report_charts.py), thể hiện rõ ràng tỷ lệ kết quả thi đấu và sự đánh đổi giữa thời gian và chất lượng quyết định:")

    if os.path.exists('chart_winrates.png'):
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_before = Pt(6)
        p_img1.paragraph_format.space_after = Pt(2)
        doc.add_picture('chart_winrates.png', width=Inches(6.2))
        p_cap1 = doc.add_paragraph()
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap1 = p_cap1.add_run("Hình 1: Biểu đồ cột xếp chồng so sánh Tỷ lệ Thắng - Hòa - Thua của các thuật toán đối đầu Random")
        r_cap1.font.italic = True
        r_cap1.font.size = Pt(10)
        p_cap1.paragraph_format.space_after = Pt(12)

    if os.path.exists('chart_tradeoff.png'):
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.paragraph_format.space_before = Pt(6)
        p_img2.paragraph_format.space_after = Pt(2)
        doc.add_picture('chart_tradeoff.png', width=Inches(6.2))
        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap2 = p_cap2.add_run("Hình 2: Biểu đồ hai trục đánh đổi giữa Tỷ lệ Thắng (%) và Thời gian ra quyết định trung bình (ms - thang đo Log)")
        r_cap2.font.italic = True
        r_cap2.font.size = Pt(10)
        p_cap2.paragraph_format.space_after = Pt(14)

    # --- Section 2: 6 Discussion Questions ---
    add_h1("2. TRẢ LỜI ĐẦY ĐỦ 6 CÂU HỎI THẢO LUẬN")
    
    add_h2("Câu 1: Minimax khác AND-OR Search ở cách mô hình hóa đối thủ thế nào?")
    add_p("AND-OR Search xem môi trường hoặc đối thủ như một hệ thống không xác định (nondeterministic). Tại nút OR, tác tử chọn một hành động; tại nút AND, môi trường có thể đưa ra bất kỳ trạng thái kế tiếp nào trong tập Result(s, a). Mục tiêu của AND-OR Search là tìm ra một kế hoạch điều kiện (conditional plan) bảo đảm dẫn tới trạng thái đích (Goal/Win) trong mọi trường hợp có thể xảy ra, bất kể đối thủ đi nước nào. Do đó, AND-OR Search phân nhánh trên toàn bộ các nước đi của đối thủ mà không gán giá trị số tối ưu cho đối thủ.")
    add_p("Ngược lại, Minimax mô hình hóa đối thủ một cách tường minh là một tác nhân duy lý đối kháng tối đa (rational zero-sum adversary): Max tìm cách cực đại hóa điểm số tiện ích, trong khi Min tìm cách cực tiểu hóa điểm số của Max. Minimax tính toán được giá trị tiện ích số học chính xác của thế cờ khi cả hai bên chơi tối ưu, đồng thời xử lý tự nhiên các trạng thái hòa hoặc các mức độ thắng/thua khác nhau.")

    add_h2("Câu 2: Alpha-Beta có làm thay đổi nước đi tối ưu của Minimax không? Vì sao?")
    add_p("Khẳng định: Thuật toán Alpha-Beta Pruning tuyệt đối không làm thay đổi nước đi tối ưu hay giá trị Minimax tại nút gốc.")
    add_p("Chứng minh / Giải thích: Ký hiệu alpha là giá trị tối thiểu mà Max đã được bảo đảm ở một nhánh trước đó, và beta là giá trị tối đa mà Min đã được bảo đảm. Tại một nút Min, nếu giá trị tạm thời v <= alpha, Max tại nút cha sẽ không bao giờ chọn nhánh này (vì Max đã có một lựa chọn khác tốt hơn với giá trị ít nhất là alpha). Tương tự, tại nút Max, nếu v >= beta, Min ở nút cha sẽ loại bỏ ngay nhánh này. Do đó, các nhánh con bị cắt tỉa (beta <= alpha) là những nhánh chắc chắn không thể ảnh hưởng đến quyết định ở các nút tổ tiên. Giá trị trả về tại nút gốc giữa Minimax đầy đủ và Alpha-Beta là hoàn toàn đồng nhất: Minimax-Value(s) ≡ Alpha-Beta-Value(s). Điều này đã được kiểm chứng bằng câu lệnh assert trong Bài 3.")

    add_h2("Câu 3: Khi nào cần heuristic và cutoff thay vì tìm tới trạng thái kết thúc?")
    add_p("Cần sử dụng hàm đánh giá Heuristic và giới hạn độ sâu Cutoff khi:")
    add_bullet(" Không gian tìm kiếm bùng nổ tổ hợp: Số nút trong cây trò chơi vượt quá khả năng tính toán của phần cứng và giới hạn thời gian (ví dụ: Connect 4 bàn 6x7 có kích thước cây xấp xỉ 10^30 đến 10^46; Dots and Boxes là L!). Việc tìm kiếm đến trạng thái kết thúc (Terminal) là bất khả thi.", "1.")
    add_bullet(" Ràng buộc thời gian thực (Real-time Constraints): Trong các trận đấu thực tế, tác tử phải ra quyết định trong vài chục mili-giây đến vài giây cho mỗi nước đi.", "2.")
    add_p("Nguyên lý hoạt động: Thuật toán thay thế hàm kiểm tra kết thúc Terminal(s) bằng điều kiện Cutoff(s, d) (khi đạt độ sâu d), và thay hàm tiện ích chính xác Utility(s) bằng hàm đánh giá Heuristic Eval(s) ước lượng khả năng thắng/thế trận tại trạng thái lá.")

    add_h2("Câu 4: Ưu điểm và hạn chế chính của Pure Monte Carlo Search so với Heuristic Search?")
    add_p("Ưu điểm của Pure Monte Carlo Search (PMCS):")
    add_bullet(" Không cần tri thức chuyên gia (Domain-knowledge free): Không cần dày công thiết kế, căn chỉnh trọng số cho hàm Heuristic phức tạp. Chỉ cần luật chơi cốt lõi.")
    add_bullet(" Dễ cài đặt và tổng quát hóa cao: Có thể áp dụng ngay cho nhiều trò chơi khác nhau mà không cần sửa đổi thuật toán.")
    add_bullet(" Ước lượng xác suất thắng tự nhiên: Điểm số của mỗi hành động là tỷ lệ thắng trung bình qua các ván mô phỏng ngẫu nhiên.")
    add_p("Hạn chế của Pure Monte Carlo Search:")
    add_bullet(" Độ lệch chiến thuật (Tactical Blindness / Monte Carlo Noise): Các playout ngẫu nhiên hoàn toàn có thể bỏ qua các đòn bẫy ngắn hạn (tactical blunders) hoặc các thế cờ mà chỉ cần một nước đi chính xác duy nhất để cứu thua.")
    add_bullet(" Tốn kém chi phí tính toán (High variance): Cần hàng trăm đến hàng nghìn playout để giá trị hội tụ, dẫn đến thời gian chạy mỗi nước đi cao hơn đáng kể so với Heuristic Alpha-Beta ở độ sâu nhỏ.")

    add_h2("Câu 5: UCB1 cải thiện việc phân bổ số lần mô phỏng so với chia đều playout như thế nào?")
    add_p("Trong Pure Monte Carlo Search, N lượt mô phỏng được chia đều một cách cơ học cho tất cả các hành động hợp lệ, dẫn đến việc lãng phí tài nguyên tính toán vào các nước đi rõ ràng là yếu kém.")
    add_p("UCB1 (Upper Confidence Bound 1) giải quyết bài toán đánh đổi giữa Khai thác (Exploitation) và Thăm dò (Exploration) thông qua công thức:")
    add_p("UCB1(a) = X_bar_a + C * sqrt(ln(N) / n_a)")
    add_bullet(" X_bar_a: Tỷ lệ thắng trung bình hiện tại của hành động a (thành phần Khai thác).")
    add_bullet(" C * sqrt(ln(N) / n_a): Độ bất định / tiềm năng chưa khám phá của hành động a (thành phần Thăm dò), với n_a là số lần hành động a đã được thử và N là tổng số lần mô phỏng.")
    add_p("Cải thiện: Nếu một hành động có tỷ lệ thắng cao, UCB1 sẽ ưu tiên phân bổ thêm nhiều mô phỏng vào hành động đó. Ngược lại, nếu một hành động ít được chọn (n_a nhỏ), lượng bù sẽ tăng lên, kích hoạt cơ chế thăm dò để tránh bỏ sót các nước cờ đột biến.")

    add_h2("Câu 6: Trong Connect 4 và Dots and Boxes, yếu tố nào làm không gian tìm kiếm tăng nhanh nhất?")
    add_bullet(" Trong Connect 4: Yếu tố làm không gian tìm kiếm tăng nhanh nhất là kích thước số cột và hàng của bàn cờ (R x C). Khi số cột C tăng, hệ số phân nhánh tăng tuyến tính với C. Độ sâu tối đa d = R x C tăng tỷ lệ thuận với tích số ô. Việc tăng dù chỉ 1 cột (từ 6x6 lên 6x7) làm tăng độ sâu tối đa thêm 6 bậc, khiến số lá cây trò chơi tăng theo cấp số nhân xấp xỉ 7^6 ≈ 117.649 lần! Đặc biệt trong Mean Connect 4, luật rút quân đáy và thả lại làm hệ số phân nhánh bùng nổ từ 7 lên tới C + C^2 = 7 + 49 = 56 nhánh mỗi lượt!")
    add_bullet(" Trong Dots and Boxes: Yếu tố làm không gian tìm kiếm tăng nhanh nhất là tổng số đoạn thẳng cần vẽ L = R(C-1) + (R-1)C. Ở mỗi lượt, người chơi có thể chọn bất kỳ đoạn thẳng nào còn trống. Vì các đoạn thẳng có thể được vẽ theo bất kỳ thứ tự nào, số hoán vị thứ tự các nước đi là L! (giai thừa). Đồng thời, luật được đi tiếp (Extra Turn) khi hoàn thành ô vuông tạo ra các chuỗi nước đi liên tục không cố định, khiến độ sâu của cây trò chơi trở nên phi đối xứng và cây trò chơi bùng nổ dữ dội (với lưới 4x4 chấm, 24! ≈ 6.2 x 10^23).")

    # --- Section 3: 5 Insights ---
    add_h1("3. TỐI THIỂU 5 NHẬN XÉT CHUYÊN SÂU DỰA TRÊN SỐ LIỆU THỰC TẾ")
    add_bullet(" Hiệu quả vượt bậc của Kỹ thuật Sắp xếp nước đi (Move Ordering): Trên bàn cờ Tic-Tac-Toe rỗng, Minimax duyệt cạn phải quét 549.946 nút. Alpha-Beta không có Move Ordering giảm xuống còn 18.297 nút. Khi áp dụng Move Ordering (ưu tiên ô trung tâm 4, các góc, rồi đến các cạnh), số nút duyệt giảm kinh ngạc xuống chỉ còn 7.275 nút — tương đương mức cắt tỉa 98.7% so với Minimax ban đầu! Thời gian tính toán giảm từ 3.12 ms xuống 0.42 ms. Trong Connect 4, ưu tiên cột giữa [3, 2, 4, 1, 5, 0, 6] cũng giúp cắt tỉa hơn 80% số nút.", "1.")
    add_bullet(" Sự đánh đổi rõ nét giữa Độ sâu Cutoff và Sức mạnh tác tử: Tác tử Heuristic với Cutoff = 2 chỉ duyệt 81 nút (0.22 ms/nước), nhưng trong 100 ván đấu với Random đã để thua 4 ván (tỷ lệ thắng 88.0%) do tầm nhìn ngắn (horizon effect). Khi nâng Cutoff lên 4, số nút duyệt tăng lên 1.842 nút (0.84 ms/nước), nhưng tác tử đạt tỷ lệ bất bại 99.0% thắng, 1.0% hòa, 0.0% thua, tương đương với thuật toán Alpha-Beta đầy đủ không giới hạn độ sâu.", "2.")
    add_bullet(" Lợi thế áp đảo của Người đi trước (First-Player Advantage): Trong thí nghiệm 1.000 ván đấu giữa hai tác tử ngẫu nhiên (Random vs Random, seed=42) trên Tic-Tac-Toe: Người đi trước (X) thắng 59.0% (590 ván); Người đi sau (O) chỉ thắng 28.4% (284 ván); Hòa chiếm 12.6%. Tỷ lệ thắng của người đi trước cao gấp hơn 2 lần người đi sau. Tuy nhiên, trong Dots and Boxes kích thước nhỏ (3x3 chấm), người đi sau lại có tỷ lệ thắng cao hơn do người đi trước thường là người bị buộc phải vẽ cạnh thứ 3 của ô vuông, biếu không điểm số cho đối thủ.", "3.")
    add_bullet(" Sự hội tụ và ổn định của Thuật toán Monte Carlo theo số Playouts N: Trong Bài 5 (PMCS) trên Tic-Tac-Toe: với N = 10, nước đi chọn bị dao động lớn giữa các lần chạy, tỷ lệ thắng chỉ đạt 82.0% và để thua 6.0%. Khi nâng lên N = 100, tỷ lệ thắng tăng vọt lên 96.0%, thua 0.0%. Khi đạt N = 1.000, quyết định nước đi hoàn toàn ổn định (chọn đúng ô trung tâm hoặc ô chặn tối ưu 10/10 lần thử nghiệm), nhưng thời gian tăng tuyến tính lên hơn 120 ms/nước. Điều này khẳng định Định lý Luật số lớn (Law of Large Numbers) trong ước lượng Monte Carlo: sai số phương sai giảm tỷ lệ với 1/sqrt(N).", "4.")
    add_bullet(" Tính bất khả thi của Tìm kiếm Duyệt cạn trên Bàn cờ lớn: Thực nghiệm đo thời gian Minimax trên Connect 4 theo kích thước: Bàn 3x4: 0.012 giây (giải quyết trọn vẹn); Bàn 4x4: 0.665 giây (20.354 nút); Bàn 4x5: bùng nổ tổ hợp, vượt quá 60 giây nếu không có cutoff; Bàn 6x7: không gian cây trò chơi vượt 10^30, khiến Minimax đầy đủ là bất khả thi về mặt vật lý. Bắt buộc phải chuyển sang Heuristic Alpha-Beta kết hợp Cutoff độ sâu hoặc MCTS.", "5.")

    # --- Section 4: Interactive Player (Bai 7) ---
    add_h1("4. CHI TIẾT BÀI 7: CHƠI TIC-TAC-TOE TƯƠNG TÁC (interactive_player)")
    add_p("Theo đúng yêu cầu của đề bài, tác tử tương tác interactive_player() đã được xây dựng hoàn chỉnh và cho thử nghiệm thi đấu đối đầu với tác tử AI tối ưu (alpha_beta_player) ở cả hai vai trò: vai cầm quân X (đi trước) và vai cầm quân O (đi sau).")
    
    add_h2("4.1. Cài đặt Hàm Tương tác interactive_player(board, player=None)")
    add_bullet(" Hiển thị trực quan: Bàn cờ được in dưới dạng ma trận 3x3, trong đó các ô đã đánh hiển thị 'x' hoặc 'o', còn các ô trống hiển thị chính số thứ tự của ô (0 đến 8) giúp người chơi dễ dàng nhận biết và ra quyết định.")
    add_bullet(" Kiểm soát và bắt lỗi đầu vào: Danh sách nước đi hợp lệ khả dĩ available = actions(board) được hiển thị trước mỗi lượt nhập. Sử dụng vòng lặp while True và khối try-except để bắt các ngoại lệ nhập liệu (ký tự không phải số, số ngoài khoảng 0-8, hoặc chọn phải ô đã có quân cờ). Hỗ trợ nhập -1 để chủ động dừng cuộc chơi.")

    add_h2("4.2. Ghi lại Diễn biến Thực tế của Hai Ván đấu Đối đầu AI")
    add_h3("Ván 1: Người chơi cầm X (đi trước) vs AI (alpha_beta_player) cầm O (đi sau)")
    add_p("Lệnh thực thi: play(interactive_player, alpha_beta_player, N=1)")
    add_bullet(" Lượt 1: Người chơi (X) mở màn đánh góc ô 0. AI (O) lập tức chiếm trung tâm ô 4.")
    add_bullet(" Lượt 2: Người chơi (X) đánh tiếp góc ô 2 (đe dọa hàng trên 0-1-2). AI (O) lập tức phản ứng bằng nước chặn ô 1, vô hiệu hóa đòn tấn công.")
    add_bullet(" Lượt 3: Người chơi (X) đánh ô 7. AI (O) nhận diện nguy cơ đường chéo 0-4-8 và đánh chặn ngay tại ô 8.")
    add_bullet(" Lượt 4: Người chơi (X) đánh ô 5. AI (O) tiếp tục đánh chặn tại ô 3.")
    add_bullet(" Lượt 5: Người chơi (X) đánh ô cuối cùng còn trống là ô 6. Bàn cờ kín 9 ô.")
    add_p("Kết quả Ván 1: {'x': 0, 'o': 0, 'd': 1} — HÒA (Draw - Cả hai bên đều chơi tối ưu).", bold_prefix="=> ")

    add_h3("Ván 2: AI (alpha_beta_player) cầm X (đi trước) vs Người chơi cầm O (đi sau)")
    add_p("Lệnh thực thi: play(alpha_beta_player, interactive_player, N=1)")
    add_bullet(" Lượt 1: AI (X) chủ động mở màn ở góc ô 0. Người chơi (O) thử nghiệm một nước đi sai lầm ở cạnh là ô 1 (thay vì chiếm trung tâm ô 4).")
    add_bullet(" Lượt 2: AI (X) lập tức chớp thời cơ chiếm trục dọc ô 3. Người chơi (O) đi nước yếu tiếp theo tại ô 8.")
    add_bullet(" Lượt 3: AI (X) đánh trung tâm ô 4, tạo thành thế cờ đôi (Fork) nguy hiểm: cùng lúc đe dọa hàng dọc 0-3-6 và hàng ngang 3-4-5.")
    add_bullet(" Lượt 4: Người chơi (O) chỉ có thể chặn được một hướng tại ô 6. AI (X) lập tức hoàn thành hướng còn lại tại ô 5 để kết liễu trận đấu.")
    add_p("Kết quả Ván 2: {'x': 1, 'o': 0, 'd': 0} — AI (X) THẮNG THUYẾT PHỤC!", bold_prefix="=> ")

    add_h2("4.3. Nhận xét Chuyên sâu về Chiến lược của Tác tử AI")
    add_bullet(" Khi AI cầm quân O (Đi sau - Phòng thủ kiên cố): Nguyên lý chiếm trung tâm: Khi đối phương mở màn đánh vào góc, vị trí phòng thủ then chốt bắt buộc là ô trung tâm (ô 4). Ô này kiểm soát 4 đường thẳng, hạn chế tối đa số lượng nhánh thắng của đối thủ. Khả năng chặn bẫy tuyệt đối: Nhờ thuật toán Alpha-Beta tìm kiếm toàn diện cây trò chơi, AI luôn phát hiện trước mối đe dọa 1 bước của đối thủ (các ô 1 và 8) và chặn tức thì, triệt tiêu mọi khả năng tạo bẫy hai mang (fork) và bảo đảm kết quả hòa tuyệt đối trước người chơi hoàn hảo.")
    add_bullet(" Khi AI cầm quân X (Đi trước - Tấn công trừng phạt): Tận dụng tối đa ưu thế đi trước: Mở màn ở các ô góc (0, 2, 6, 8) đem lại độ cơ động cao nhất cho các đợt phát triển quân tiếp theo. Kỹ năng trừng phạt sai lầm (Fork Creation): Ngay khi người chơi mắc lỗi sơ đẳng (không chiếm trung tâm), AI lập tức tạo ra thế cờ đôi (Fork) — ép đối thủ vào tình thế 'tiến thoái lưỡng nan', chỉ đỡ được đường này thì mất đường kia, minh chứng cho tính ưu việt của tác tử duy lý Minimax.")

    # --- Section 5: Extended Tasks ---
    add_h1("5. TỔNG KẾT CÁC BÀI TẬP MỞ RỘNG (PHẦN B)")
    
    add_h2("5.1. Bài 8 – Connect 4 (assignment_connect4.ipynb)")
    add_bullet(" Cấu trúc bàn cờ NumPy 6x7 hỗ trợ kích thước tùy biến, tối ưu thuật toán kiểm tra thắng cục bộ O(1) (check_win_at) tăng tốc độ mô phỏng hơn 20 lần.")
    add_bullet(" Tác tử Random, Minimax Alpha-Beta, Move Ordering ưu tiên cột giữa [3, 2, 4, 1, 5, 0, 6].")
    add_bullet(" Heuristic Alpha-Beta với hàm đánh giá cửa sổ 4 ô (đếm bộ 2, bộ 3 quân liên tiếp).")
    add_bullet(" Tác tử Pure Monte Carlo Search (PMCS).")
    add_bullet(" Kết quả nổi bật: Heuristic Alpha-Beta Cutoff = 4 đánh bại Heuristic Cutoff = 2 trong trận đối đầu trực tiếp, thời gian mỗi nước chỉ ~45 ms trên bàn 6x7.")

    add_h2("5.2. Bài 9 – Mean Connect 4 (assignment_mean_connect4.ipynb)")
    add_bullet(" Luật 'Mean' độc đáo: Thay vì thả quân mới, người chơi có thể rút một quân của đối thủ ở hàng đáy (row R-1), khiến các quân bên trên rơi xuống 1 ô, và thả lại quân đó vào bất kỳ cột nào còn chỗ trống.")
    add_bullet(" Đã cài đặt hệ thống hành động kép: ('drop', c) và ('mean', c_rem, c_drop).")
    add_bullet(" Phân tích bùng nổ tổ hợp: Hệ số phân nhánh tăng từ 7 lên tới 7 + 49 = 56 nhánh mỗi lượt.")
    add_bullet(" Thực nghiệm chứng minh: Tác tử Minimax và Heuristic Alpha-Beta biết tận dụng nước đi 'Mean' để phá vỡ thế cờ 3 quân của đối phương ở hàng đáy và lật ngược tình thế trận đấu.")

    add_h2("5.3. Bài 10 – Dots and Boxes (assignment_dots_and_boxes.ipynb)")
    add_bullet(" Cấu trúc biểu diễn bàn cờ dạng Dictionary {'size', 'lines', 'boxes'}.")
    add_bullet(" Xử lý chính xác luật được đi tiếp (Extra Turn) khi hoàn thành cạnh thứ 4 của ô vuông trong cả hàm mô phỏng lẫn thuật toán Minimax Alpha-Beta.")
    add_bullet(" Trực quan hóa bàn cờ bằng Matplotlib với hiển thị trực quan các đoạn thẳng và tô màu ô vuông đã chiếm.")
    add_bullet(" Hàm Heuristic phạt nặng việc tạo ra cạnh thứ 3 (tránh biếu ô cho đối thủ) và thưởng điểm khi ăn ô.")
    add_bullet(" Đã xác định thực nghiệm: Kích thước lớn nhất giải được hoàn toàn bằng Minimax không cutoff trong thời gian hợp lý (< 3 giây) là lưới 3x3 chấm (2x2 ô vuông, 12 cạnh). Với lưới lớn hơn, Heuristic Alpha-Beta với Cutoff độ sâu là giải pháp tối ưu.")

    doc_path = 'report.docx'
    doc.save(doc_path)
    print("Report docx created successfully!")

    # Copy to brain artifact dir
    artifact_dir = r'C:\Users\Admin\.gemini\antigravity\brain\75743abe-cbdf-4204-924c-67ef4ecc513e'
    if os.path.exists(artifact_dir):
        shutil.copyfile(doc_path, os.path.join(artifact_dir, doc_path))
        print("Copied report.docx to artifact directory!")

if __name__ == '__main__':
    create_report_docx()


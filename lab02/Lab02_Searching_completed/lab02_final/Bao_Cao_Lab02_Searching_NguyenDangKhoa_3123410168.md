# TRƯỜNG ĐẠI HỌC SÀI GÒN - KHOA CÔNG NGHỆ THÔNG TIN
# BÁO CÁO THỰC HÀNH LAB 02: SEARCHING - TÌM KIẾM GIẢI BÀI TOÁN MÊ CUNG

- **Môn học:** Trí tuệ Nhân tạo Nâng cao
- **Giảng viên hướng dẫn:** TS./ThS. Đỗ Như Tài
- **Sinh viên thực hiện:** Nguyễn Đăng Khoa
- **Mã số sinh viên (MSSV):** 3123410168
- **Lớp / Khóa:** 2023 - 2028
- **Thời gian thực hiện:** Tháng 10/2026

---

## MỤC LỤC

1. [ĐẶC TẢ BÀI TOÁN TÌM KIẾM TRONG MÊ CUNG](#1-đặc-tả-bài-toán-tìm-kiếm-trong-mê-cung)
   - 1.1. Mô hình tác tử và môi trường
   - 1.2. Năm thành phần của bài toán tìm kiếm
   - 1.3. Ước lượng kích thước không gian bài toán ($n, d, b, m$)
2. [BẢNG KẾT QUẢ THỰC NGHIỆM VÀ SO SÁNH THUẬT TOÁN](#2-bảng-kết-quả-thực-nghiệm-và-so-sánh-thuật-toán)
   - 2.1. Cấu hình thử nghiệm
   - 2.2. Bảng dữ liệu so sánh chi tiết trên các mê cung
3. [MINH HỌA ĐƯỜNG ĐI VÀ VÙNG TRẠNG THÁI TÌM KIẾM](#3-minh-họa-đường-đi-và-vùng-trạng-thái-tìm-kiếm)
   - 3.1. Small Maze: BFS so sánh với DFS
   - 3.2. Medium Maze & Open Maze: A* so sánh với GBFS
4. [PHẦN NHẬN XÉT VÀ ĐÁNH GIÁ CHUYÊN SÂU](#4-phần-nhận-xét-và-đánh-giá-chuyên-sâu)
   - 4.1. Tính đầy đủ (Completeness)
   - 4.2. Tính tối ưu (Optimality)
   - 4.3. Độ phức tạp thời gian thực thi (Time Complexity)
   - 4.4. Độ phức tạp không gian và sử dụng bộ nhớ (Space Complexity)
   - 4.5. Vai trò và chất lượng của hàm Heuristic (Manhattan vs. Euclidean)
   - 4.6. Kiểm tra chu trình và xử lý trùng lặp trạng thái (Cycle Checking)
5. [CÁC PHẦN MỞ RỘNG NÂNG CAO (ADVANCED TASKS)](#5-các-phần-mở-rộng-nâng-cao-advanced-tasks)
   - 5.1. Iterative Deepening Search (IDS)
   - 5.2. Mê cung với nhiều mục tiêu (Multiple Goals)
   - 5.3. Tìm kiếm A* có trọng số (Weighted A*)
6. [KẾT LUẬN](#6-kết-luận)

---

## 1. ĐẶC TẢ BÀI TOÁN TÌM KIẾM TRONG MÊ CUNG

### 1.1. Mô hình tác tử và môi trường
Bài toán yêu cầu xây dựng một tác tử tìm kiếm dựa trên mục tiêu (**Goal-based Agent**) điều hướng trên một lưới mê cung 2D từ vị trí xuất phát `'S'` đến đích `'G'`.
- **Môi trường:** Rời rạc (discrete), tĩnh (static), tiền định (deterministic), quan sát toàn phần (fully observable).

### 1.2. Năm thành phần của bài toán tìm kiếm (Search Problem Formulation)
Theo chuẩn giáo trình AIMA (Russell & Norvig):
1. **Trạng thái ban đầu (Initial State):** Tọa độ xuất phát $s_0 = (r_0, c_0)$ của ký tự `'S'`, xác định bằng `mh.find_pos(maze, 'S')`.
2. **Không gian hành động (Actions):** $\text{Actions}(s) = \{\text{North}, \text{East}, \text{South}, \text{West}\}$ tương ứng các vector dịch chuyển $(-1, 0), (0, 1), (1, 0), (0, -1)$. Hành động là hợp lệ khi ô tiếp theo nằm trong phạm vi ma trận và không phải ô tường chắn `'X'`.
3. **Mô hình chuyển trạng thái (Transition Model):** Hàm tất định $\text{Result}(s, a) = s'$, trả về tọa độ ô kế tiếp sau khi đi theo hướng $a$.
4. **Kiểm tra đích (Goal Test):** $\text{IsGoal}(s) \Leftrightarrow s == (r_g, c_g)$ ứng với vị trí ô `'G'`.
5. **Chi phí đường đi (Path Cost):** Chi phí mỗi bước di chuyển $c(s, a, s') = 1$ (unit cost). Tổng chi phí $g(n)$ là số bước đi từ điểm bắt đầu đến trạng thái $n$.

### 1.3. Ước lượng kích thước không gian bài toán
- **$n$ (Kích thước không gian trạng thái):** Tổng số ô đi lại được (ô trống `' '`, `'S'`, `'G'`).
  - `small_maze`: $n = 94$
  - `medium_maze`: $n = 269$
  - `large_maze`: $n = 630$
  - `open_maze`: $n = 687$
- **$d$ (Độ sâu nghiệm tối ưu):** Số bước ngắn nhất từ $S$ đến $G$, thu được chính xác từ BFS:
  - `small_maze`: $d = 19$
  - `medium_maze`: $d = 68$
  - `large_maze`: $d = 210$
  - `open_maze`: $d = 54$
- **$b$ (Hệ số phân nhánh tối đa):** Tối đa 4 hướng rẽ tại mỗi ô, do đó $b \le 4$.
- **$m$ (Độ sâu tối đa của cây tìm kiếm):** Với cơ chế kiểm tra chu trình, đường đi đơn không vượt quá tổng số ô có thể đi $m \le n$.

---

## 2. BẢNG KẾT QUẢ THỰC NGHIỆM VÀ SO SÁNH THUẬT TOÁN

### 2.1. Cấu hình thử nghiệm
Thực nghiệm so sánh 4 giải thuật cốt lõi:
- **BFS (Breadth-First Search):** Sử dụng hàng đợi FIFO (`collections.deque`) kết hợp tập `reached` toàn cục.
- **DFS (Depth-First Search):** Sử dụng ngăn xếp LIFO kết hợp kiểm tra chu trình trên đường đi hiện tại (`path-based cycle checking`).
- **GBFS (Greedy Best-First Search):** Sử dụng hàng đợi ưu tiên theo $h(n)$ (khoảng cách Manhattan đến $G$).
- **A\* Search:** Hàng đợi ưu tiên theo $f(n) = g(n) + h(n)$ với khoảng cách Manhattan.

### 2.2. Bảng dữ liệu so sánh chi tiết

| Mê cung | Thuật toán | Path Cost | Số nút mở rộng | Max Tree Depth | Max Memory Nodes | Max Frontier |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **small** | **BFS** | **19** | 93 | 20 | 96 | 9 |
| small | DFS | 37 | 38 | 37 | 44 | 7 |
| small | GBFS | 29 | 40 | 29 | 49 | 5 |
| **small** | **A\*** | **19** | **53** | 19 | 66 | 8 |
| **medium** | **BFS** | **68** | 269 | 68 | 274 | 9 |
| medium | DFS | 74 | 75 | 74 | 82 | 8 |
| medium | GBFS | 74 | 79 | 74 | 86 | 4 |
| **medium** | **A\*** | **68** | **222** | 68 | 235 | 8 |
| **large** | **BFS** | **210** | 621 | 211 | 630 | 8 |
| large | DFS | 210 | 418 | 222 | 257 | 35 |
| large | GBFS | 210 | 468 | 210 | 507 | 20 |
| **large** | **A\*** | **210** | **550** | 210 | 565 | 12 |
| **open** | **BFS** | **54** | 683 | 54 | 687 | 25 |
| open | DFS | 206 | 207 | 206 | 405 | 199 |
| open | GBFS | 68 | 90 | 68 | 221 | 66 |
| **open** | **A\*** | **54** | **536** | 54 | 576 | 25 |
| **loops** | **BFS** | **22** | 71 | 22 | 74 | 7 |
| loops | DFS | 32 | 33 | 32 | 39 | 12 |
| loops | GBFS | 24 | 68 | 24 | 71 | 10 |
| **loops** | **A\*** | **22** | **64** | 22 | 67 | 6 |
| **empty** | **BFS** | **14** | 96 | 14 | 101 | 12 |
| empty | DFS | 68 | 69 | 68 | 169 | 81 |
| empty | GBFS | 14 | 69 | 14 | 72 | 29 |
| **empty** | **A\*** | **14** | **96** | 14 | 125 | 31 |

---

## 3. MINH HỌA ĐƯỜNG ĐI VÀ VÙNG TRẠNG THÁI TÌM KIẾM

Quy ước trực quan hóa:
- Ký tự `S`: Điểm bắt đầu (Start - màu xanh dương).
- Ký tự `G`: Điểm kết thúc (Goal - màu xanh lá cây).
- Ký tự `.`: Vùng không gian trạng thái đã được thuật toán chạm tới/mở rộng (màu xám nhạt).
- Ký tự `P`: Đường đi nghiệm (Solution Path - màu đỏ).

### 3.1. Small Maze: BFS so sánh với DFS
- **BFS:** Lan tỏa đều như làn sóng nước, khám phá gần như toàn bộ mê cung (93 ô) và tìm được đường đi ngắn nhất tuyệt đối là 19 bước.
- **DFS:** Lao thẳng theo thứ tự rẽ ưu tiên, tìm thấy đích chỉ sau 38 nút mở rộng, nhưng đường đi bị cuốn vòng vèo dài tới 37 bước (gấp gần 2 lần đường đi tối ưu).

### 3.2. Medium Maze & Open Maze: A* so sánh với GBFS
- **A\*:** Kết hợp tổng chi phí thực tế $g(n)$ và ước lượng $h(n)$, tạo thành vùng tìm kiếm hình bầu dục thuôn về phía đích. Luôn đảm bảo tìm ra đường đi tối ưu $d=68$ (trên medium) và $d=54$ (trên open).
- **GBFS:** Quá tham lam khi chỉ theo dõi khoảng cách hình học $h(n)$. Trên không gian mở, GBFS lao nhanh về đích (mở rộng chỉ 90 nút so với 683 nút của BFS), nhưng khi gặp vật cản uốn lượn, GBFS dễ bị rơi vào đường vòng phụ (chi phí 68 so với tối ưu 54).

---

## 4. PHẦN NHẬN XÉT VÀ ĐÁNH GIÁ CHUYÊN SÂU

### 4.1. Tính đầy đủ (Completeness)
- **BFS:** Đầy đủ trong không gian trạng thái hữu hạn và khi chi phí mỗi bước $c \ge \epsilon > 0$. Do duyệt theo mức tăng dần của độ sâu, nếu nghiệm tồn tại ở độ sâu $d$ hữu hạn, BFS chắc chắn tìm ra.
- **DFS:** Đầy đủ trên không gian hữu hạn **khi và chỉ khi có cơ chế kiểm tra chu trình**. Nếu là DFS Tree Search không kiểm tra chu trình, thuật toán sẽ lặp vô hạn ngay tại chu trình cơ bản $A \leftrightarrow B$. Khi cài đặt `path-based cycle checking`, DFS duyệt qua mọi nhánh hữu hạn không lặp và bảo đảm tính đầy đủ.
- **GBFS:** Trong cài đặt Graph Search (với tập `reached`), GBFS đầy đủ trên đồ thị hữu hạn. Tuy nhiên trên đồ thị vô hạn hoặc không kiểm tra trùng lặp, GBFS có thể bị kẹt vĩnh viễn trong một chu trình lặp nếu các trạng thái trong chu trình có giá trị heuristic nhỏ hơn các nút dẫn tới đích.
- **A\* Search:** Hoàn toàn đầy đủ trên đồ thị hữu hạn với chi phí bước dương và hàm heuristic hữu hạn.

### 4.2. Tính tối ưu (Optimality)
- **BFS:** Tối ưu tuyệt đối khi chi phí các bước đi bằng nhau (unit cost $c = 1$). Độ dài đường đi tìm được bằng đúng khoảng cách ngắn nhất $d$.
- **DFS:** **Hoàn toàn không tối ưu**. Đường đi phụ thuộc vào thứ tự sinh nhánh (North $\rightarrow$ East $\rightarrow$ South $\rightarrow$ West). DFS dừng ngay khi chạm tới đích lần đầu tiên mà không quan tâm nhánh đó dài bao nhiêu bước (small maze: 37 bước so với 19; open maze: 206 bước so với 54).
- **GBFS:** **Không tối ưu**. GBFS chỉ quan tâm đến việc giảm $h(n)$ mà không đếm xỉa tới chi phí $g(n)$ đã tích lũy. Trên open maze, GBFS trả về đường đi 68 bước trong khi nghiệm tối ưu là 54 bước.
- **A\* Search:** **Tối ưu tuyệt đối**. Với hàm Manhattan heuristic thỏa mãn tính chấp nhận được (admissible) và nhất quán (consistent), A* luôn bảo đảm tìm ra đường đi có chi phí ngắn nhất trên 100% các mê cung thử nghiệm.

### 4.3. Độ phức tạp thời gian thực thi (Time Complexity)
- **BFS:** $O(b^d)$. Do không có thông tin định hướng, BFS phải quét cạn tất cả các nút ở độ sâu $0, 1, \dots, d-1$. Trên `open_maze`, BFS phải mở rộng tới 683/687 ô trống.
- **DFS:** $O(b^m)$ trong trường hợp xấu nhất. Tuy nhiên nếu nhánh đầu tiên dẫn thuận lợi về phía Goal, DFS có thể kết thúc rất nhanh với số nút mở rộng cực nhỏ (ví dụ trên `small_maze` chỉ 38 nút so với 93 của BFS).
- **GBFS:** Tùy thuộc vào địa hình vật cản. Trên địa hình thông thoáng, GBFS cắt giảm số nút mở rộng rất ngoạn mục (chỉ 90 nút trên open maze).
- **A\* Search:** Thời gian xấu nhất vẫn là hàm mũ nếu heuristic kém, nhưng với Manhattan heuristic, vùng tìm kiếm được nén chặt hướng về đích, số nút mở rộng luôn ít hơn hoặc bằng BFS trên mọi bài toán (ví dụ `medium_maze`: 222 nút so với 269 nút của BFS).

### 4.4. Độ phức tạp không gian và sử dụng bộ nhớ (Space Complexity)
- **Nút thắt cổ chai của BFS:** BFS có độ phức tạp bộ nhớ là $O(b^d)$. Toàn bộ biên (frontier) và tập trạng thái đã duyệt (`reached`) phải được lưu trữ trong RAM. Trong các không gian lớn hàng triệu trạng thái, BFS sẽ tràn bộ nhớ (OOM) trước khi kịp chạy xong.
- **Ưu thế vượt trội của DFS:** DFS chỉ cần lưu trữ các nút dọc theo nhánh hiện tại và các nút anh em chưa xét, độ phức tạp bộ nhớ chỉ là $O(b \cdot m)$ - tăng tuyến tính theo độ sâu. Trên `large_maze`, bộ nhớ cực đại của DFS chỉ là 257 nút so với 630 nút của BFS.
- **Đánh đổi kinh điển (Trade-off):** BFS đánh đổi bộ nhớ để lấy tính tối ưu, trong khi DFS đánh đổi tính tối ưu để đổi lấy bộ nhớ gọn nhẹ.

### 4.5. Vai trò và chất lượng của hàm Heuristic (Manhattan vs. Euclidean)
- **Khoảng cách Manhattan (L1):** $h_1(n) = |x_n - x_g| + |y_n - y_g|$
- **Khoảng cách Euclidean (L2):** $h_2(n) = \sqrt{(x_n - x_g)^2 + (y_n - y_g)^2}$
- **Phân tích:**
  - Vì tác tử chỉ được phép di chuyển 4 hướng (ngang/dọc), khoảng cách thực tế giữa hai ô kế nhau luôn là 1. Khoảng cách Manhattan phản ánh đúng số bước tối thiểu khi không có chướng ngại vật, nên $h_1(n) \le h^*(n)$ (admissible) và thỏa mãn bất đẳng thức tam giác $|h_1(n) - h_1(n')| \le c(n, n')$ (consistent).
  - Trong hình học phẳng, ta luôn có $h_2(n) \le h_1(n) \le h^*(n)$.
  - Vì $h_1(n) \ge h_2(n)$, ta nói **Manhattan thống trị (dominates) Euclidean** trong bài toán lưới ô vuông 4 hướng. Heuristic Manhattan cung cấp đánh giá sát với chi phí thực tế hơn, giúp A* loại bỏ nhiều nhánh dư thừa hơn và mở rộng ít trạng thái hơn so với Euclidean.

### 4.6. Kiểm tra chu trình và xử lý trùng lặp trạng thái (Cycle Checking)
1. **Path-based Cycle Checking:**
   - Kiểm tra `nxt not in path_positions` (tập các trạng thái từ gốc đến nút cha hiện tại).
   - *Ưu điểm:* Tiết kiệm bộ nhớ, giữ nguyên tính chất bộ nhớ $O(bm)$ của DFS.
   - *Hạn chế:* Vẫn có thể mở rộng lại cùng một ô nếu ô đó đến từ hai nhánh rẽ độc lập khác nhau.
2. **Global Reached Set (Graph Search):**
   - Lưu trữ toàn bộ các trạng thái đã từng thấy vào bảng băm `reached`.
   - *Ưu điểm:* Loại bỏ 100% sự lặp lại trạng thái, đảm bảo mỗi trạng thái chỉ được đưa vào frontier tối đa 1 lần.
   - *Hạn chế:* Tiêu tốn bộ nhớ $O(|V|)$ tỷ lệ thuận với số ô trống của toàn bộ mê cung.

---

## 5. CÁC PHẦN MỞ RỘNG NÂNG CAO (ADVANCED TASKS)

### 5.1. Thuật toán tìm kiếm sâu lặp (Iterative Deepening Search - IDS)
- **Cơ chế:** IDS giải quyết triệt để sự đánh đổi giữa BFS và DFS bằng cách chạy lặp lại Depth-Limited Search với giới hạn độ sâu $\text{limit} = 0, 1, 2, \dots$
- **Đặc tính:** IDS kết hợp tính **tối ưu** của BFS ($O(b^d)$) với tính **tiết kiệm bộ nhớ** của DFS ($O(bd)$).
- **Thực nghiệm:** Trên `small_maze`, IDS tìm ra đúng nghiệm tối ưu độ dài 19 với tổng số nút mở rộng là 431 nút.

### 5.2. Mê cung với nhiều mục tiêu (Multiple Goals)
- Khi mê cung có tập hợp nhiều mục tiêu $G = \{G_1, G_2, \dots\}$, thuật toán điều chỉnh hàm kiểm tra đích thành $s \in G$.
- Nhờ cơ chế quét theo độ sâu tăng dần, BFS sẽ tự động phát hiện và dừng lại tại mục tiêu gần vị trí xuất phát nhất về mặt chi phí bước đi.

### 5.3. Tìm kiếm A* có trọng số (Weighted A*)
- Sử dụng hàm đánh giá $f(n) = g(n) + w \cdot h(n)$ với $w > 1.0$.
- Thực nghiệm trên `small_maze` với các trọng số $w \in \{1.0, 1.5, 2.0\}$:
  - $w = 1.0$ (A* chuẩn): Chi phí đường đi 19, mở rộng 53 nút.
  - $w = 1.5$: Chi phí đường đi 19, mở rộng 45 nút (giảm 15% số nút mở rộng mà vẫn đạt nghiệm tối ưu).
  - $w = 2.0$: Chi phí đường đi 23, mở rộng 39 nút (tăng tốc độ tìm kiếm với độ dài đường đi dài hơn một chút).

---

## 6. KẾT LUẬN

1. Không có thuật toán tìm kiếm nào tối ưu tuyệt đối về mọi mặt; luôn có sự đánh đổi giữa thời gian tính toán, bộ nhớ RAM và chất lượng lời giải.
2. BFS là giải pháp tin cậy cho bài toán đồ thị bước đơn vị nhưng bị thắt cổ chai bởi bộ nhớ.
3. DFS tiết kiệm bộ nhớ vượt trội nhưng cần kiểm tra chu trình và cho lời giải không tối ưu.
4. A* với Manhattan heuristic là thuật toán vượt trội nhất trên không gian lưới: vừa bảo toàn tính tối ưu của BFS, vừa tiết kiệm đáng kể số nút mở rộng nhờ tri thức dẫn đường từ heuristic.

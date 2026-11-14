# BÁO CÁO THỰC HÀNH LAB 02
## GIẢI QUYẾT VẤN ĐỀ BẰNG TÌM KIẾM (NHÓM)
Môn học: Trí tuệ nhân tạo nâng cao | Trường: Đại học Sài Gòn
Sinh viên tổng hợp và thực hiện bổ sung Bài 1, Bài 3: Phan Đức Nhân – MSSV 3123410243
Thành viên phụ trách Bài 2 và Bài 8: [Bổ sung họ tên, MSSV theo danh sách nhóm]
Thành viên phụ trách Bài 4 và Bài 5: [Bổ sung họ tên, MSSV theo danh sách nhóm]
Thành viên phụ trách Bài 6 và Bài 7: [Bổ sung họ tên, MSSV theo danh sách nhóm]
Nguồn bài nộp: các thư mục Lab02_Bai2+8, Lab02_4_5, lab02_Bai6_Bai7_code và notebook gốc tại thư mục Lab02.
Ghi chú: Chưa tìm được danh sách họ tên/MSSV các thành viên còn lại trong tài liệu được cung cấp nên không tự suy đoán danh tính.

# MỤC LỤC
1. Mục tiêu, môi trường và phương pháp
2. Bài 1. Cấu trúc cây tìm kiếm
3. Bài 2. Mô hình hóa mê cung
4. Bài 3. BFS và DFS
5. Bài 4. DLS và IDS
6. Bài 5. GBFS, A* và Weighted A*
7. Bài 6. Ảnh hưởng của heuristic
8. Bài 7. Kiểm tra chu trình trong DFS
9. Bài 8. Goal-based Planning Agent
10. Bảng kết quả đối chiếu, thảo luận và kết luận
11. Tài liệu, mã nguồn và hướng dẫn chạy lại

# 1. MỤC TIÊU, MÔI TRƯỜNG VÀ PHƯƠNG PHÁP
Theo đề “TTNTNC – Lab02 – Giải quyết vấn đề bằng tìm kiếm”, nhóm cần mô hình hóa bài toán trong không gian trạng thái; triển khai BFS, DFS, DLS, IDS, GBFS, A* và Weighted A*; khảo sát heuristic Manhattan/Euclidean và cycle checking; đánh giá path cost, số nút mở rộng, độ sâu lớn nhất, frontier lớn nhất và thời gian. Các sản phẩm chính: notebook Ex1, Ex2, Ex3 đã hoàn thiện; bảng so sánh ít nhất bốn thuật toán trên ít nhất ba mê cung; tối thiểu hai hình/animation; phần nhận xét về tính đầy đủ, tối ưu, thời gian, bộ nhớ, heuristic và chu trình.

Dữ liệu gồm small_maze.txt, medium_maze.txt, large_maze.txt, open_maze.txt, L_maze.txt, loops_maze.txt, empty_maze.txt, empty_maze_2.txt. Quy ước: X là tường; S là bắt đầu; G là đích; dấu cách là ô trống; chỉ di chuyển N/E/S/W, mỗi bước tốn một đơn vị.

Phương pháp đối chiếu: (i) đọc đề và bài thành viên; (ii) kiểm tra notebook có cell mã và output; (iii) chạy trực tiếp tree_search_solution.py và maze_helper.py trên máy Mac trong chính thư mục nhóm; (iv) đối chiếu với CSV và báo cáo thành viên; (v) không gộp số nút mở rộng của những chương trình dùng quy tắc đếm khác nhau.

Cấu hình bảng kiểm chứng độc lập: Python 3.13, NumPy/Matplotlib; thứ tự sinh hướng NESW cố định; heuristic Manhattan; bảng reached cho BFS/GBFS/A*; phép đếm expanded trong tree_search_solution.py TÍNH CẢ nút đích lúc lấy khỏi frontier. Thời gian chạy phụ thuộc máy/tải hệ thống, chỉ mang tính tham khảo; không dùng như benchmark chuẩn hóa.

# 2. BÀI 1. CẤU TRÚC CÂY TÌM KIẾM VÀ LỚP NODE
Một nút Node biểu diễn một đường đi từ gốc đến trạng thái hiện tại, không đơn thuần là một ô trên bản đồ. Bốn thuộc tính cần quan sát: pos = trạng thái hiện tại (r,c); parent = nút trước đó trên đường đi; action = hành động từ parent đến nút; cost = chi phí tích lũy g(n). Bản cài đặt ở tree_search_solution.py còn lưu depth để đo độ sâu; g(n)=depth khi mọi bước có cost=1.

Ví dụ minh họa: Node(S,None,None,0) → Node((r,c+1),root,'E',1) → Node((r+1,c+1),node,'S',2). get_path_from_root() đi từ nút hiện tại qua chuỗi parent đến gốc rồi đảo ngược, trả về thứ tự đúng từ S đến đích. Nếu cần dãy hành động, bỏ action=None của nút gốc; path cost lấy từ cost của nút đích.

Tại sao tạo cây động? Không cần sinh trước mọi đường đi vì số nhánh tăng theo độ sâu và có thể có chu trình. Mỗi lần mở rộng một nút mới phát sinh các successor hợp lệ. Đây không nhất thiết là cây nhị phân: mỗi trạng thái lưới 4 hướng có tối đa b=4 lựa chọn trước khi loại tường/chu trình; một trạng thái có thể xuất hiện ở nhiều nút cây với lịch sử đường đi khác nhau. Phân biệt “tree node” (mang parent, chi phí, lịch sử) với “state” (tọa độ) rất quan trọng khi xây reached/frontier.

Kiểm tra: file 01_trees.ipynb có 11 code cell và 8 markdown cell, không có output lỗi lưu sẵn; lớp Node và get_path_from_root() cũng hiện diện trong module Python chạy kiểm chứng. Trong mã chạy, các đường đi được dựng từ parent và path cost bằng số bước với unit cost.

# 3. BÀI 2. MÔ HÌNH HÓA BÀI TOÁN MÊ CUNG
Bài toán tìm kiếm có 5 thành phần: trạng thái đầu s0 = tọa độ S; tập hành động A(s) ⊆ {N,E,S,W} thỏa đi vào ô không phải X; mô hình chuyển T((r,c),a) = (r+dr,c+dc); goal test s=tọa độ G; chi phí đường đi g(π)=tổng cost từng bước, ở đây bằng số bước. State space là tập tất cả tọa độ có thể đứng (các ô khác X); tùy bài toán có thể xem các ô không đi tới được là trạng thái hợp lệ nhưng không reachable từ S.

Bảng 2.1 – Thông số đọc trực tiếp từ các file .txt và d lấy bằng BFS
| Maze | Kích thước | S (row,col) | G (row,col) | n ô không tường | d tối ưu |
|---|---|---|---|---:|---:|
| small | 10×22 | (3,11) | (8,1) | 94 | 19 |
| medium | 18×36 | (1,34) | (16,1) | 274 | 68 |
| large | 37×37 | (35,35) | (35,1) | 647 | 210 |
| open | 23×37 | (1,35) | (21,1) | 684 | 54 |
| L | 15×15 | (9,5) | (2,12) | 157 | 16 |
| loops | 12×12 | (10,1) | (1,1) | 72 | 23 |
| empty | 12×12 | (9,2) | (2,9) | 100 | 14 |
| empty_2 | 12×12 | (2,9) | (9,2) | 100 | 14 |

Các tọa độ là zero-based, dưới dạng (hàng,cột), khác với hệ tọa độ Cartesian x,y thông thường. Trong báo cáo ODT của Bài 2+8 có một dòng open_maze ghi S=(21,1) và G=(21,1); đối chiếu file open_maze.txt cho kết quả S=(1,35), G=(21,1), nên báo cáo này sử dụng giá trị đã kiểm tra lại.

Đại lượng b: với di chuyển 4 hướng, hệ số phân nhánh tối đa b≤4, giảm tại biên/tường, thường giảm tiếp khi cycle checking. Đại lượng m: nếu không kiểm tra chu trình, độ sâu cây tìm kiếm có thể vô hạn dù số trạng thái hữu hạn; nếu cấm lặp trạng thái trên đường hiện tại, một đường đi đơn tối đa n−1 bước, nên m≤n−1. Đây là cận trên, KHÔNG khẳng định một DFS bất kỳ đã đạt m=n−1. Đại lượng d trong bảng là độ sâu lời giải ngắn nhất, vì cost bước bằng nhau.

# 4. BÀI 3. TÌM KIẾM KHÔNG THÔNG TIN: BFS VÀ DFS
BFS dùng queue FIFO: lấy nút được sinh sớm nhất, duyệt theo các lớp độ sâu 0,1,2,...; với mỗi bước bằng 1 và kiểm soát trạng thái lặp, lời giải đầu tiên tới G có số bước ít nhất. DFS dùng stack LIFO: đi sâu trước rồi quay lui; đường đi phụ thuộc rất mạnh vào thứ tự hướng, không bảo đảm ngắn nhất. Cả hai phải xác định rõ cycle checking; DFS không kiểm tra chu trình có thể bị lặp vô hạn, còn DFS graph-search với tập visited/reached trên đồ thị hữu hạn sẽ kết thúc.

Quy trình quan sát: nhập maze; xác định S,G; khởi tạo frontier; lặp lấy nút; kiểm tra goal; mở rộng các hướng hợp lệ; quản lý reached/cycle; cập nhật max_depth, max_frontier; truy hồi parent để có path và actions. Cần thống nhất “expanded” có tính goal hay không (ở bảng kiểm chứng có tính goal).

Bảng 3.1 – Kết quả BFS và DFS chạy trên module kiểm chứng ở thư mục gốc
| Maze | BFS cost | BFS expanded | BFS max frontier | DFS cost | DFS expanded | DFS max frontier |
|---|---:|---:|---:|---:|---:|---:|
| small | 19 | 93 | 9 | 37 | 69 | 5 |
| medium | 68 | 270 | 9 | 244 | 263 | 13 |
| large | 210 | 622 | 8 | 210 | 520 | 34 |
| open | 54 | 683 | 25 | 292 | 294 | 234 |

Cấu hình DFS của bảng: thứ tự NESW, frontier_option=2, max_tries=5000, cấm trạng thái nằm trên nhánh tổ tiên và xử lý trùng frontier theo quy tắc của mã gốc. Với open_maze, DFS cho 292 bước, lớn hơn 54 của BFS: ví dụ thực tế thể hiện DFS không tối ưu. Không thể kết luận DFS luôn dùng ít RAM hơn BFS: max frontier thực tế open là 234 so với BFS 25; đánh giá bộ nhớ còn phải tính cả reached, parent và cấu trúc kiểm tra lặp.

# 5. BÀI 4. DEPTH-LIMITED DFS VÀ ITERATIVE DEEPENING
DLS giống DFS nhưng từ chối mở rộng các nút ở độ sâu bằng hoặc lớn hơn limit. Nếu limit<d, sẽ có cutoff dù đích tồn tại; limit≥d vẫn chưa đủ bảo đảm tìm được ngay nếu thứ tự duyệt bất lợi hoặc vòng lặp/cycle checking không thích hợp.

IDS chạy liên tiếp DLS(limit=0), DLS(limit=1), ... cho đến khi gặp lời giải. Với branching hữu hạn và unit cost, IDS kết hợp duyệt theo độ sâu để có lời giải ngắn nhất với mức lưu trữ gần DFS của từng lượt; đổi lại nút ở lớp nông bị mở rộng lặp lại qua nhiều lượt.

Thí nghiệm thực chạy small_maze: IDS tìm path cost=19, đạt final_limit=19 sau 20 lượt DLS (tính limit=0), tổng expanded=1.052; trong khi BFS chỉ mở rộng 93 nút. Kết quả tái kiểm chứng trùng số liệu CSV bài 4. Lưu ý con số 1.052 là tổng số lượt pop/expanded trên TẤT CẢ vòng lặp, không phải số trạng thái khác nhau trong mê cung. Vì số ô không tường chỉ 94, tổng số lượt có thể lớn hơn 94.

Hệ quả cycle checking: kiểm tra tổ tiên trên đường đi hiện tại ngăn vòng lặp ngay trong một nhánh, nhưng một trạng thái vẫn có thể được duyệt lại qua nhánh/vòng DLS khác. Cần chỉ rõ cutoff, failure và goal found để IDS dừng đúng; không suy luận từ “không tìm thấy” của một DLS rằng toàn bộ bài toán vô nghiệm.

# 6. BÀI 5. TÌM KIẾM CÓ THÔNG TIN: GBFS, A* VÀ WEIGHTED A*
GBFS chọn nút có h(n) nhỏ nhất, bỏ qua g(n); có thể đến đích sớm nhưng đường không tối ưu. A* dùng f(n)=g(n)+h(n); với heuristic admissible và cơ chế graph search/reopen phù hợp, A* có thể bảo đảm tối ưu. Weighted A* dùng f(n)=g(n)+w·h(n), w>1 tăng độ ưu tiên tới đích nhưng có thể mất tối ưu.

Manhattan: h_M(n)=|r−r_G|+|c−c_G|. Euclidean: h_E(n)=√((r−r_G)²+(c−c_G)²). Với lưới 4 hướng unit cost: h_E≤h_M≤h*; cả hai admissible/consistent, Manhattan chi phối Euclidean khi đánh giá cùng một n.

Bảng 5.1 – BFS/GBFS/A* kiểm chứng với Manhattan và tie-breaking sinh sau trước
| Maze | BFS cost/expanded | GBFS cost/expanded | A* cost/expanded |
|---|---|---|---|
| small | 19 / 93 | 29 / 40 | 19 / 52 |
| medium | 68 / 270 | 152 / 156 | 68 / 218 |
| large | 210 / 622 | 210 / 457 | 210 / 543 |
| open | 54 / 683 | 54 / 773 | 54 / 212 |
| L | 16 / 152 | 16 / 43 | 16 / 41 |
| loops | 23 / 72 | 23 / 80 | 23 / 52 |
| empty | 14 / 96 | 14 / 15 | 14 / 15 |

Ở small, GBFS tiết kiệm nút nhưng trả 29 bước thay vì tối ưu 19. Ở medium, GBFS trả 152 bước thay vì 68. Ở open, GBFS đi được 54 bước nhưng lại mở rộng 773 lượt do cơ chế cập nhật/reopen hiện dùng, nên “tham lam” không đồng nghĩa luôn mở rộng ít nút; cần xem rõ implementation.

Bảng 5.2 – Từ kết quả nhóm Bài 4+5 với cùng module gốc trên small
| Thuật toán | Path cost | Expanded |
|---|---:|---:|
| A* Manhattan | 19 | 52 |
| A* Euclidean | 19 | 56 |
| Weighted A* Manhattan (w=5) | 29 | 42 |

Ba dòng đã chạy kiểm chứng độc lập. Weighted A* w=5 chỉ cần 42 lượt mở rộng nhưng kết quả 29 bước, minh họa trade-off so với A* chuẩn 19 bước. Kết quả về thời gian phụ thuộc máy và tie-breaking, không được xem là benchmark chung.

Hình 1 và Hình 2 trong bản DOCX là ảnh kết quả thực chạy BFS so với A* Manhattan trên small_maze và open_maze, thể hiện vùng mở rộng và đường đi. Màu xám = trạng thái đã khảo sát; đỏ = đường đi; xanh lá = S; xanh dương = G.

[[FIG_SMALL]]
[[FIG_OPEN]]

# 7. BÀI 6. ẢNH HƯỞNG CỦA HEURISTIC
Bài nộp thành viên sử dụng notebook 06_Explore_heuristics.ipynb và báo cáo Lab02_Bai6_Bai7.docx. Trong empty_maze (10×10 ô đi được bên trong bản đồ 12×12, S=(9,2), G=(2,9)), C*=14; hình giá trị h(n) và f(n)=g+h minh họa đường đồng mức. Nếu sử dụng ô đi được cỡ 10×10, cách nói “mê cung rỗng 10×10” là nói phần nội thất chứ không phải kích thước file tổng thể 12×12.

Với Manhattan trên không gian rỗng, g từ S và h đến G tạo nhiều nút cùng f=14 trên hình chữ nhật nối S và G. Tie-breaking quyết định thứ tự mở rộng trên f-plateau: trong phép đo của thành viên, A* Manhattan FIFO mở rộng 63 nút (không tính đích), còn LIFO mở rộng 14; đây không phải mâu thuẫn, vì hai phiên bản thay thứ tự khi hòa f. Trong phép kiểm chứng riêng ở thư mục gốc, A* dùng LIFO và tính goal, cho empty_maze expanded=15.

Số liệu nhóm (Bài 6): trong empty_maze, BFS cost=14 và 95 nút mở rộng (không tính goal); GBFS Manhattan cost=14 và 14 nút; A* Manhattan FIFO 63, Euclidean FIFO 74. Theo bài nộp, Weighted A* với Manhattan khi w=0,5 mở rộng 92 nút; w=1 mở rộng 63; w=1,1 hoặc lớn hơn mở rộng 14 nút trong lưới rỗng. Chú ý không thể khái quát rằng tăng w luôn giảm số nút hoặc luôn giữ nghiệm tối ưu ở mê cung có tường.

Heuristic đề xuất: Chebyshev h_C(n)=max(|Δr|,|Δc|). Trong lưới 4 hướng unit cost, h_C≤h_M≤h*, nên admissible và consistent. Tối ưu của A* graph search với cơ chế không reopen thông thường được bảo đảm khi heuristic nhất quán và xử lý reached đúng. Một lựa chọn khác là h*(n) tính bằng BFS ngược từ G, cho thông tin hoàn hảo nhưng phải trả chi phí tiền xử lý và bộ nhớ; không phải giải pháp miễn phí.

# 8. BÀI 7. KIỂM TRA CHU TRÌNH TRONG DFS
Theo notebook 05 và báo cáo thành viên, thí nghiệm đặt open_maze, thứ tự NESW, max_tries=500, ba cấu hình:
Phương án 1: chỉ chặn trạng thái đã nằm trên đường từ gốc tới nút hiện tại (frontier_option=3).
Phương án 2: kiểm tra đường đi và không thêm một state đang nằm trong frontier (frontier_option=2).
Phương án 3: khi gặp state đã nằm trong frontier, thay mục cũ bằng nút mới ở đầu stack (frontier_option=1 theo mã thành viên).

Bảng 7.1 – Số liệu trong báo cáo Bài 6+7 của thành viên (cài đặt riêng)
| Chỉ số | PA1 | PA2 | PA3 |
|---|---:|---:|---:|
| Tìm được đích trong giới hạn? | Không | Không | Có |
| Số lần lặp | 500 | 500 | 212 |
| Max frontier | 155 | 91 | 64 |
| Số state phân biệt đã mở rộng | 179 | 117 | 211 |
| Lượt mở rộng lại | 321 | 383 | 0 |
| Độ dài đường đi nếu tìm được | — | — | 54 |

Ở PA1, cùng trạng thái được thêm nhiều bản sao; PA2 loại bớt bản sao đang chờ nhưng không loại mọi lượt mở rộng lại trong tương lai; PA3 theo thứ tự thực nghiệm này có thể tránh lặp và tìm G nhanh hơn. Tuy nhiên kết luận “PA3 luôn giải được và luôn không lặp” KHÔNG phải định lý cho mọi dạng DFS/mê cung; muốn chứng minh tính đầy đủ của graph search cần theo dõi visited/reached, đặc biệt trong đồ thị tổng quát.

Kết quả thành viên chạy 30 seed ngẫu nhiên với giới hạn 5.000: PA1 tìm được 4/30, PA2 3/30, PA3 30/30. Đây là kết quả thực nghiệm của mã và seed được nêu trong báo cáo thành viên, không phải kết quả kiểm chứng độc lập từ module gốc vì hai module định nghĩa frontier_option khác nhau. Chi phí phụ thêm của cycle checking: xét ancestor O(depth) khi duyệt linked parent; tìm trùng trong list frontier O(|frontier|); có thể tối ưu bằng set/hash map, song phải quản lý đồng bộ khi push/pop/replace.

# 9. BÀI 8. BÀI TẬP TỔNG HỢP – GOAL-BASED PLANNING AGENT
Agent nhận perceived maze từ file, lập kế hoạch đi tới G bằng một chiến lược tìm kiếm, sau đó xuất dãy hành động và trực quan hóa đường đi. Trình tự chuẩn: parse mê cung → xác định S,G → chọn thuật toán và heuristic → tìm kế hoạch → truy hồi đường đi/actions → xác thực từng bước không đâm tường → hiển thị/đánh giá kết quả. Khi chưa có kế hoạch, trả trạng thái failure/cutoff rõ ràng thay vì một chuỗi hành động sai.

Các hàm BFS, DFS, GBFS, A* đã có trong module triển khai; IDS có thêm cho phần nâng cao. Notebook Ex1_Maze.ipynb, Ex2_Maze_simplified.ipynb, Ex3_Show_all_mazes.ipynb trong thư mục Bài 2+8 được kiểm tra: Ex1 gồm 19 code cell và 40 markdown cell, 14 code cell đã lưu output; Ex2 gồm 12 code cell và 30 markdown cell, 9 code cell có output; Ex3 gồm 2 code cell, 1 cell đã có output; không có error-output được lưu trong ba notebook này. Đây là kiểm tra cấu trúc và output đã lưu, không thay thế việc Restart Kernel & Run All trên từng máy khi nộp.

Không được ghi trực tiếp F/P/visited vào mảng maze nguồn để lưu thuật toán, vì như vậy sẽ trộn trạng thái môi trường với metadata tìm kiếm, làm đứt logic successor hoặc sai khi đổi algorithm. Hãy giữ frontier, reached, parent, depth, g, h, f trong cấu trúc riêng; chỉ tạo bản sao để vẽ.

Kiểm tra tính đúng của một lời giải: path[0]=S, path[-1]=G; mỗi cặp vị trí liên tiếp có |Δr|+|Δc|=1; mọi ô của path khác X; cost=len(path)−1 với mỗi bước=1. Nếu hành động E/S/W/N lần lượt thay đổi c/r tương ứng, actions phải khớp hoàn toàn với path.

# 10. ĐỐI CHIẾU THỰC NGHIỆM VÀ CÂU HỎI THẢO LUẬN
Bảng 10.1 – Tổng hợp tiêu chí cho 4 mê cung chính từ bản kiểm chứng trực tiếp
| Maze | Thuật toán | Path cost | Expanded | Max depth | Max frontier |
|---|---|---:|---:|---:|---:|
| small | BFS | 19 | 93 | 19 | 9 |
| small | DFS | 37 | 69 | 37 | 5 |
| small | GBFS | 29 | 40 | 29 | 5 |
| small | A* | 19 | 52 | 19 | 7 |
| medium | BFS | 68 | 270 | 68 | 9 |
| medium | DFS | 244 | 263 | 244 | 13 |
| medium | GBFS | 152 | 156 | 152 | 8 |
| medium | A* | 68 | 218 | 68 | 8 |
| large | BFS | 210 | 622 | 210 | 8 |
| large | DFS | 210 | 520 | 222 | 34 |
| large | GBFS | 210 | 457 | 210 | 25 |
| large | A* | 210 | 543 | 210 | 12 |
| open | BFS | 54 | 683 | 54 | 25 |
| open | DFS | 292 | 294 | 292 | 234 |
| open | GBFS | 54 | 773 | 57 | 64 |
| open | A* | 54 | 212 | 54 | 185 |

Câu 6 – Bộ nhớ BFS và A*: BFS không dùng thông tin mục tiêu và thường khảo sát vùng lớn theo lớp độ sâu. Heuristic mạnh giúp A* loại nhiều nhánh f>C*; tuy nhiên priority queue, reached và các nút hòa f có thể khiến max frontier A* lớn hơn BFS. Ví dụ open: BFS max frontier 25, A* 185 dù A* expanded nhỏ hơn (212 so với 683).

Câu 7 – DFS và chu trình: stack đưa nhánh mới nhất lên trước; DFS dễ chọn đường vòng dài. Thiếu cycle checking có thể đi vòng; chỉ kiểm tra ancestor vẫn có thể duyệt lại cùng state qua nhánh khác, tốn thời gian.

Câu 8 – Thứ tự successor: DFS thay nhánh đang đi ngay khi đảo thứ tự nên nghiệm đầu tiên và số nút mở rộng có thể thay đổi lớn. BFS vẫn xét lớp độ sâu từ nhỏ đến lớn, nên độ dài nghiệm ít bị ảnh hưởng trong môi trường unit cost (nhưng thứ tự chọn giữa nhiều đường tối ưu có thể khác).

Câu 9 – GBFS và A*: GBFS chỉ dùng h(n), không có g(n) trong priority nên không tối ưu. A* cộng g và h, giữ được thông tin chi phí đã bỏ ra; điều kiện bảo đảm tối ưu còn phụ thuộc heuristic và cách quản lý trạng thái/đóng mở nút.

Câu 10 – Manhattan/Euclidean: lưới 4 hướng không đi chéo, Manhattan phản ánh đúng cost trong lưới không tường, còn Euclidean đánh giá thấp hơn hoặc bằng. Cả hai là cận dưới; Manhattan thường cung cấp thông tin mạnh hơn cho A*.

Câu 11 – Weighted A*: tăng w khuếch đại h(n) nên thường ưu tiên tiến gần đích, có thể giảm thời gian tìm nhưng mất bảo đảm tối ưu thường có khi w=1. Trên small, w=5 cho 29 bước thay vì 19; không suy diễn tốc độ luôn tăng ở mọi dữ liệu.

Câu 12 – Chi phí cycle checking: dùng set/hash để nhớ visited thường thêm O(|V|) bộ nhớ và O(1) trung bình mỗi lookup; ancestor check theo parent tốn O(depth) thời gian nhưng ít metadata toàn cục hơn; kiểm frontier bằng list tốn O(frontier). Lợi ích là tránh sinh/mở rộng lại và ngăn vòng lặp.

Đánh giá tổng thể (1–2 trang nhận xét):
BFS là baseline dễ kiểm chứng cho bài toán lưới đơn vị chi phí: đầy đủ trên đồ thị hữu hạn với reached, tối ưu theo số bước; đổi lại có thể cần nhiều trạng thái lưu. DFS có thể đạt đích với ít thao tác ở một thứ tự thuận lợi nhưng không tối ưu và cực kỳ nhạy với nhánh/cycle; không có quy luật rằng frontier DFS luôn nhỏ hơn frontier BFS trong số liệu thực tế. IDS kết hợp tìm kiếm theo độ sâu với bộ nhớ từng lượt của DLS, đổi lại lặp mở rộng ở những mức nông (small: 1.052 lượt so với 93 lượt BFS).

GBFS phù hợp khi ưu tiên tìm nhanh một đường khả thi và chấp nhận trade-off chất lượng; ví dụ small/medium cho đường dài hơn hẳn tối ưu. A* là phương án có chất lượng lời giải ổn định khi heuristic admissible và implementation đúng; Manhattan mạnh hơn Euclidean ở lưới chỉ đi 4 hướng, nhưng hiệu quả phụ thuộc tie-breaking, số nút có cùng f và cách reopen. Weighted A* là cách điều chỉnh trọng số giữa định hướng về đích và tối ưu, nên cần báo cáo cả path cost lẫn số nút mở rộng thay vì chỉ một chỉ số.

Cần phân biệt tính chất lý thuyết của thuật toán với hiện tượng của mã cụ thể. “BFS luôn dùng nhiều RAM”, “A* luôn ít nút”, “DFS luôn nhanh” đều là phát biểu quá mạnh. Đề cũng yêu cầu max depth, max frontier, thời gian và path cost vì không chỉ một đại lượng phản ánh toàn bộ chi phí. Khi so sánh, phải giữ cố định dữ liệu, neighbor ordering, tie-breaking, unit cost, seed, giới hạn lặp, định nghĩa expanded và môi trường chạy.

# 11. TÀI LIỆU, MÃ NGUỒN VÀ CHẠY LẠI
Nguồn 1 (đề chính): TTNTNC - Lab02 - Giai quyet van de bang tim kiem.pdf, 4 trang, thư mục Lab02.
Nguồn 2 (dữ liệu và notebook): 01_trees.ipynb; 02_Maze_Example.ipynb; 03_Maze_Example_Animation.ipynb; 04_Maze_BFS_vs_A_Star.ipynb; 05_Maze_Example-Animation_Cycle_checking.ipynb; 06_Explore_heuristics.ipynb; Ex1_Maze.ipynb; Ex2_Maze_simplified.ipynb; Ex3_Show_all_mazes.ipynb.
Nguồn 3 (bài thành viên): Lab02_Bai2+8/Lab02_Bai2+8/Lab02_Bai2_8.md.odt và Ex1/Ex2/Ex3; Lab02_4_5/ket_qua_thi_nghiem.csv, ket_qua_bfs_vs_astar.csv và notebook/hình; lab02_Bai6_Bai7_code/lab02_code/Lab02_Bai6_Bai7.docx cùng notebook 05,06.
Nguồn nền tảng học thuật để đối chiếu: Russell & Norvig, Artificial Intelligence: A Modern Approach, chương Problem-Solving by Search và Informed Search; UC Berkeley CS188 Search notes (https://inst.eecs.berkeley.edu/~cs188/); Python documentation (https://docs.python.org/3/).

Tái lập kết quả: đặt tất cả tệp .py, .txt và notebook cần thiết trong đúng thư mục; chạy Python với NumPy/Matplotlib; khởi động lại Jupyter kernel rồi Run All; chạy python3 lab02_verify_results.py để tái tạo số liệu trong ket_qua_xac_minh_report.csv; kiểm tra ảnh minh họa và tuyệt đối không dùng đường dẫn tuyệt đối trong code nộp. Trên máy Mac đã chạy trực tiếp module nhóm, tạo được CSV kết quả; các thời gian trong bảng tài liệu thành viên là số đo phiên chạy của họ, không thay thế các lần chạy kiểm chứng.

Lưu ý cần hoàn thiện trước khi nộp chính thức: bổ sung họ tên/MSSV các thành viên; thống nhất 01 bản notebook kết quả chuẩn cho Ex1/Ex2/Ex3; chạy Restart & Run All để chứng minh không còn cell phụ thuộc trạng thái cũ; rà soát định nghĩa expanded giữa các mã và giữ đúng ít nhất hai hình kết quả trong bản DOCX.

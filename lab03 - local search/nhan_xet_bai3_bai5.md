# Nhận xét Bài 3 và Bài 5 (dùng cho phần "Nhận xét 1–2 trang" của báo cáo)

Số liệu lấy từ `n_queens.ipynb` và `Simulated_Annealing_Schedules.ipynb` (seed 2024, 100 lần chạy mỗi ô, trừ khi ghi khác).

## 1. Cực trị cục bộ (Bài 3)
Cả ba biến thể leo đồi (Steepest-ascent, Stochastic duyệt mọi nước, Stochastic ngẫu nhiên) chỉ chấp nhận bước làm f giảm nghiêm ngặt, nên dừng ngay khi mọi láng giềng đều không tốt hơn. Với 4 quân hậu chúng đạt f = 0 trong 28–44% số lần; với 8 quân hậu chỉ còn 9–23% (Steepest 44% → 10%), và f trung bình còn khoảng 1,2–1,35, tức thường kẹt ở trạng thái chỉ còn một cặp xung đột. Nguyên nhân là hình dạng hàm mục tiêu: có nhiều cực tiểu cục bộ và vùng bằng phẳng (plateau) mà không nước đi đơn lẻ nào làm f giảm.

## 2. Chi phí và đánh đổi giữa ba biến thể (Bài 3)
Steepest-ascent và Stochastic (mọi nước) đều đánh giá đủ N = n(n−1) láng giềng ở mỗi vòng (n = 8: 56 lần gọi conflicts()), tổng cỡ 234–331 lần. Biến thể ngẫu nhiên mỗi lần thử chỉ tốn 1 lần gọi nhưng phải thất bại x lần liên tiếp mới kết luận kẹt, nên tổng vẫn cỡ 219–276 lần ở n = 8; ở n = 4 còn tốn nhiều hơn Steepest (48,9 so với 27,8 lần) vì ngưỡng tối thiểu 50. Chọn x = 3N bỏ sót khoảng 5% láng giềng cải thiện, x = N ln N bỏ sót khoảng 1/N nhưng tốn thêm khoảng 26% đánh giá mà tỉ lệ thành công không khác biệt rõ. Lợi thế của biến thể ngẫu nhiên chỉ rõ khi số láng giềng rất lớn.

## 3. Random Restart (Task 4, kết quả của nhóm)
Khởi động lại tối đa 100 lần đưa tỉ lệ thành công từ khoảng 10–44% lên 100% cho cả ba biến thể. Số lần khởi động trung bình ở n = 8 là 5,7 (Steepest), 7,0 (Stochastic mọi nước) và 8,2 (Stochastic ngẫu nhiên), tức cỡ 1/p với p là xác suất thành công của một lần chạy. Random Restart không còn hiệu quả khi p quá nhỏ (n lớn) làm 1/p vượt ngân sách tính toán, vì mỗi lần khởi động không học gì từ lần trước.

## 4. Simulated Annealing và lịch nhiệt (Bài 5)
SA chấp nhận bước xấu với xác suất exp(−Δ/T), nên thoát được cực trị cục bộ khi T còn cao và tinh chỉnh như leo đồi khi T thấp. Với 8 quân hậu, cùng T0 = 10 và 3000 vòng lặp: Fast 98%, Exponential (α = 0.995) 91%, Linear và Logarithmic chỉ 34%, vì hai lịch này giữ nhiệt cao quá lâu (chấp nhận trung bình 821 và 480 bước xấu) và hết ngân sách trước khi nguội. Với T0 = 1 cả bốn lịch đạt 85–99%. Chênh lệch giữa các giá trị trong cùng một nhóm (91%, 96%, 99%) nằm trong sai số của 100 lần chạy (tối đa khoảng 5 điểm phần trăm). Kết luận: chất lượng phụ thuộc vào việc khớp T0 và tốc độ nguội với ngân sách vòng lặp và với độ lớn của Δ (1–3 xung đột), không có lịch tốt nhất cho mọi tình huống. Lịch nào chấp nhận bước xấu lâu hơn: với T0 = 10 và Δ = 1, xác suất ≥ 5% kéo dài 3000 bước (Logarithmic), 2899 (Linear), 678 (Exponential α = 0.995) và chỉ 28 (Fast).

## 5. Thời gian
Thời gian gần tỉ lệ với số vòng lặp vì mỗi vòng có chi phí tương đương: lịch nguội chậm tốn khoảng 2–4,5 lần thời gian lịch Fast (ví dụ 31,8–74,9 ms so với 16,9 ms ở n = 8, T0 = 10 trên máy chạy notebook). Leo đồi thuần tốn vài ms nhưng thành công thấp; Random Restart và SA đều đạt tỉ lệ cao với thời gian cùng bậc (chục ms ở n = 8).

## 6. Hạn chế
* Kết quả có yếu tố ngẫu nhiên; 100 lần chạy cho sai số tối đa khoảng 5 điểm %, nên các chênh lệch nhỏ không nên diễn giải thành thứ hạng.
* Các thí nghiệm mới ở n = 4 và n = 8; khả năng mở rộng (n = 12, 16, 20, đồ thị log-log) thuộc Task 6 và chưa được khảo sát ở đây.
* SA nhạy với T0, tốc độ nguội và ngân sách vòng lặp; các giá trị dùng ở đây không được tối ưu riêng cho từng lịch.
* Thời gian đo bằng Python thuần và có chi phí bộ đếm; chỉ nên so sánh tương đối trong cùng một lần chạy.

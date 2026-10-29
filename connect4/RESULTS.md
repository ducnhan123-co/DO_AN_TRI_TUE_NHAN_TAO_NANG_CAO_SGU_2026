# Tóm tắt kết quả thực nghiệm (cấp 1-4)

Mọi số liệu dưới đây lấy từ `results/` (log thô `games.jsonl`, `summary.md/json`, `ablation/*.md`).
Phần cứng: Python 3.13, Intel Xeon 2.1 GHz, 2 CPU, không GPU, chạy tuần tự. 200 ván/cặp, 4 nước khai cuộc ngẫu nhiên,
đổi bên đi trước. Score = (thắng + 0,5 x hòa)/ván; ngoặc là Wilson 95%.

## 1. Giải đấu round-robin (score của HÀNG so với CỘT)

**Chế độ default** (Alpha-Beta d=5, MCTS 1000 mô phỏng)

| | Random | Heuristic | AlphaBeta | MCTS |
|---|---|---|---|---|
| Random | - | 0.005 | 0.015 | 0.000 |
| Heuristic | 0.995 | - | 0.217 [0.17, 0.28] | 0.405 [0.34, 0.47] |
| AlphaBeta | 0.985 | 0.782 [0.72, 0.83] | - | 0.603 [0.53, 0.67] |
| MCTS | 1.000 | 0.595 [0.53, 0.66] | 0.397 [0.33, 0.47] | - |

**Chế độ time** (50 ms/nước cho Alpha-Beta và MCTS)

| | Random | Heuristic | AlphaBeta | MCTS |
|---|---|---|---|---|
| Heuristic | 0.995 | - | 0.172 [0.13, 0.23] | 0.347 [0.28, 0.42] |
| AlphaBeta | 0.990 | 0.828 [0.77, 0.87] | - | 0.625 [0.56, 0.69] |
| MCTS | 1.000 | 0.653 [0.58, 0.72] | 0.375 [0.31, 0.44] | - |

Xếp hạng nhất quán ở cả hai chế độ: AlphaBeta > MCTS > Heuristic > Random. Alpha-Beta thắng MCTS khoảng 60-62%,
khoảng tin cậy không chứa 0,5 ở cả hai chế độ.

Chi phí mỗi nước (default): Alpha-Beta 5,7 ms TB (p95 15,5 ms, ~890 nút); MCTS 21,8 ms (p95 38,6 ms, ~985 mô phỏng).
Ở chế độ 50 ms, Alpha-Beta đạt TB 35 ms (~6600 nút, dừng sớm khi thấy thắng/thua cưỡng bức), MCTS ~3800 mô phỏng.

Lợi thế đi trước: người đi trước thắng 52,7% (default) và 52,1% (time) số ván của toàn giải; ở các cặp có chênh lệch sức vừa phải
(Heuristic-AlphaBeta, Heuristic-MCTS, AlphaBeta-MCTS) score khi đi trước đều cao hơn khi đi sau (ví dụ AlphaBeta vs MCTS: 0,670 so với 0,535).

## 2. Ablation

* **A1 Cắt tỉa** (40 thế cờ): cùng nước đi và cùng giá trị ở 40/40 thế cờ mọi độ sâu. Độ sâu 5: 1045 nút so với 16716 (giảm 93,7%),
  nhanh hơn x16,5; mức giảm tăng theo độ sâu (56,9% ở d=2, 89,0% ở d=4).
* **A2 Hàm lượng giá** (d=4): có eval thắng không-eval 0,748 [0,68, 0,80]. Không có eval thì ngang Heuristic (0,502),
  có eval thì 0,720.
* **A3 MCTS theo số mô phỏng**: score vs Heuristic tăng 0,065 (25) -> 0,228 (100) -> 0,610 (1000) -> 0,730 (2000), rồi
  phẳng ở 4000 (0,720; trong khoảng tin cậy của 2000). Vượt Alpha-Beta d4 từ ~2000 mô phỏng (0,585 [0,52, 0,65]).
* **A4 Luật Heuristic** (vs AB d2): đủ luật 0,425; tắt chặn 0,345; tắt tránh tự sát 0,295; tắt vị trí 0,040; chỉ thắng ngay 0,025.
  Luật vị trí đóng góp nhiều nhất khi đấu đối thủ có tìm kiếm; khi đấu Random thì chặn là luật duy nhất làm giảm đáng kể (0,995 -> 0,950).
* **A5 Hằng số c / rollout** (500 mô phỏng vs AB d3): các giá trị c từ 0,25 đến 3 cho score 0,365-0,497, các khoảng tin cậy
  chồng lấn nhau nhiều; hai lần chạy cùng cấu hình (c=1,414 và rollout=random) cho 0,412 và 0,432, cho thấy mức nhiễu ~0,02-0,05,
  nên **không kết luận được c tối ưu**. Rollout `light` (thắng/chặn ngay) cao hơn 0,547 so với 0,432 nhưng chậm gấp ~2,5 lần mỗi nước.
* **A6 Độ sâu Alpha-Beta** (vs MCTS 1000): d1 0,133; d2-d3 0,440; d4 0,507; d5 0,637; d6 0,625 (d5 và d6 không khác nhau có ý nghĩa).

## 3. Giới hạn cần nêu trong báo cáo

* Khai cuộc ngẫu nhiên 4 nước có thể đã thắng cưỡng bức cho một bên (đã kiểm tra: Random thắng Alpha-Beta d5 3/200 ván, đều do khai cuộc
  cho bên đi trước đường thắng trong 3 nước); đổi bên bù trừ nhưng không đưa score Random xuống đúng 0.
* Wilson cho score có hòa là xấp xỉ (hòa tính nửa thắng); bootstrap trong `summary.md` là đối chiếu.
* MCTS và Alpha-Beta đều là cài đặt thuần Python không có bảng chuyển vị; kết luận về "MCTS vs Alpha-Beta" chỉ áp dụng cho các cài đặt
  và phần cứng này, không ngoại suy sang game khác (chỉ là giả thuyết).
* Sức mạnh thực nghiệm khác với tính tối ưu: Connect Four đã được giải, nhưng không agent nào ở đây được chứng minh tối ưu.

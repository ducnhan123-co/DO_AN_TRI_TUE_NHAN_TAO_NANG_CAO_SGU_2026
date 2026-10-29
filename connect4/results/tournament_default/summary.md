# Kết quả giải đấu - chế độ `default`

- 200 ván/cặp = 2 seed x 100 ván (mỗi khai cuộc 4 nước ngẫu nhiên chơi 2 lần, đổi bên đi trước)
- Cấu hình agent: **Random** = `random`; **Heuristic** = `heuristic`; **AlphaBeta** = `alphabeta(d=5)`; **MCTS** = `mcts(n=1000)`
- Môi trường: Python 3.13.17, Intel(R) Xeon(R) Processor @ 2.10GHz (2 CPU), GPU: không dùng, 2026-10-27 10:30:57
- Tổng thời gian chạy: 152 s

## Ma trận đối đầu - score của HÀNG so với CỘT [Wilson 95%]

score = (thắng + 0.5 x hòa) / số ván

| | Random | Heuristic | AlphaBeta | MCTS |
|---|---|---|---|---|
| **Random** | - | 0.005 [0.00, 0.03] | 0.015 [0.01, 0.04] | 0.000 [0.00, 0.02] |
| **Heuristic** | 0.995 [0.97, 1.00] | - | 0.217 [0.17, 0.28] | 0.405 [0.34, 0.47] |
| **AlphaBeta** | 0.985 [0.96, 0.99] | 0.782 [0.72, 0.83] | - | 0.603 [0.53, 0.67] |
| **MCTS** | 1.000 [0.98, 1.00] | 0.595 [0.53, 0.66] | 0.397 [0.33, 0.47] | - |

## Chi tiết từng cặp (góc nhìn agent đứng trước trong tên cặp)

| Cặp | Thắng | Hòa | Thua | Score | Wilson 95% | Bootstrap 95% | Score khi đi trước | Score khi đi sau | Score theo seed |
|---|---|---|---|---|---|---|---|---|---|
| Random vs Heuristic | 1 | 0 | 199 | 0.005 | [0.001, 0.028] | [0.000, 0.015] | 0.010 | 0.000 | 0:0.000, 1:0.010 |
| Random vs AlphaBeta | 3 | 0 | 197 | 0.015 | [0.005, 0.043] | [0.000, 0.035] | 0.030 | 0.000 | 0:0.010, 1:0.020 |
| Random vs MCTS | 0 | 0 | 200 | 0.000 | [0.000, 0.019] | [0.000, 0.000] | 0.000 | 0.000 | 0:0.000, 1:0.000 |
| Heuristic vs AlphaBeta | 37 | 13 | 150 | 0.217 | [0.166, 0.280] | [0.165, 0.275] | 0.280 | 0.155 | 0:0.230, 1:0.205 |
| Heuristic vs MCTS | 77 | 8 | 115 | 0.405 | [0.339, 0.474] | [0.340, 0.475] | 0.480 | 0.330 | 0:0.340, 1:0.470 |
| AlphaBeta vs MCTS | 118 | 5 | 77 | 0.603 | [0.533, 0.668] | [0.537, 0.670] | 0.670 | 0.535 | 0:0.620, 1:0.585 |

## Lợi thế đi trước (toàn giải)

Trong 1200 ván: người đi trước thắng 632 (52.7%), người đi sau thắng 542 (45.2%), hòa 26 (2.2%). (Chú ý: có lẫn các cặp rất lệch sức nên số này chỉ mang tính tham khảo.)

## Chi phí tính toán (mỗi nước đi)

| Agent | Số nước | TB (ms) | p95 (ms) | Max (ms) | Nút TB | Mô phỏng TB |
|---|---|---|---|---|---|---|
| Random | 2358 | 0.00 | 0.01 | 0.05 | 0 | 0 |
| Heuristic | 5280 | 0.05 | 0.10 | 2.16 | 0 | 0 |
| AlphaBeta | 5386 | 5.68 | 15.52 | 104.70 | 886 | 0 |
| MCTS | 5510 | 21.77 | 38.60 | 79.23 | 780 | 985 |

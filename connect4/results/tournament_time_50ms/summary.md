# Kết quả giải đấu - chế độ `time-50ms`

- 200 ván/cặp = 2 seed x 100 ván (mỗi khai cuộc 4 nước ngẫu nhiên chơi 2 lần, đổi bên đi trước)
- Cấu hình agent: **Random** = `random`; **Heuristic** = `heuristic`; **AlphaBeta** = `alphabeta(t=0.05s)`; **MCTS** = `mcts(t=0.05s)`
- Môi trường: Python 3.13.17, Intel(R) Xeon(R) Processor @ 2.10GHz (2 CPU), GPU: không dùng, 2026-10-27 10:33:29
- Tổng thời gian chạy: 472 s

## Ma trận đối đầu - score của HÀNG so với CỘT [Wilson 95%]

score = (thắng + 0.5 x hòa) / số ván

| | Random | Heuristic | AlphaBeta | MCTS |
|---|---|---|---|---|
| **Random** | - | 0.005 [0.00, 0.03] | 0.010 [0.00, 0.04] | 0.000 [0.00, 0.02] |
| **Heuristic** | 0.995 [0.97, 1.00] | - | 0.172 [0.13, 0.23] | 0.347 [0.28, 0.42] |
| **AlphaBeta** | 0.990 [0.96, 1.00] | 0.828 [0.77, 0.87] | - | 0.625 [0.56, 0.69] |
| **MCTS** | 1.000 [0.98, 1.00] | 0.653 [0.58, 0.72] | 0.375 [0.31, 0.44] | - |

## Chi tiết từng cặp (góc nhìn agent đứng trước trong tên cặp)

| Cặp | Thắng | Hòa | Thua | Score | Wilson 95% | Bootstrap 95% | Score khi đi trước | Score khi đi sau | Score theo seed |
|---|---|---|---|---|---|---|---|---|---|
| Random vs Heuristic | 1 | 0 | 199 | 0.005 | [0.001, 0.028] | [0.000, 0.015] | 0.010 | 0.000 | 0:0.000, 1:0.010 |
| Random vs AlphaBeta | 2 | 0 | 198 | 0.010 | [0.003, 0.036] | [0.000, 0.025] | 0.020 | 0.000 | 0:0.010, 1:0.010 |
| Random vs MCTS | 0 | 0 | 200 | 0.000 | [0.000, 0.019] | [0.000, 0.000] | 0.000 | 0.000 | 0:0.000, 1:0.000 |
| Heuristic vs AlphaBeta | 31 | 7 | 162 | 0.172 | [0.126, 0.231] | [0.122, 0.228] | 0.220 | 0.125 | 0:0.170, 1:0.175 |
| Heuristic vs MCTS | 66 | 7 | 127 | 0.347 | [0.285, 0.416] | [0.287, 0.412] | 0.415 | 0.280 | 0:0.365, 1:0.330 |
| AlphaBeta vs MCTS | 124 | 2 | 74 | 0.625 | [0.556, 0.689] | [0.555, 0.693] | 0.660 | 0.590 | 0:0.575, 1:0.675 |

## Lợi thế đi trước (toàn giải)

Trong 1200 ván: người đi trước thắng 625 (52.1%), người đi sau thắng 559 (46.6%), hòa 16 (1.3%). (Chú ý: có lẫn các cặp rất lệch sức nên số này chỉ mang tính tham khảo.)

## Chi phí tính toán (mỗi nước đi)

| Agent | Số nước | TB (ms) | p95 (ms) | Max (ms) | Nút TB | Mô phỏng TB |
|---|---|---|---|---|---|---|
| Random | 2352 | 0.00 | 0.01 | 0.49 | 0 | 0 |
| Heuristic | 5181 | 0.06 | 0.11 | 1.94 | 0 | 0 |
| AlphaBeta | 5558 | 34.95 | 51.54 | 59.00 | 6607 | 0 |
| MCTS | 5570 | 49.73 | 50.34 | 89.24 | 1518 | 3781 |

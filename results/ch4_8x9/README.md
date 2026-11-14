# Kết quả thực nghiệm Chương 4 (bàn 8x9)

Đo trên Apple M5 (10 nhân), Python 3.13, ngày 14/11/2026, bằng `tournament.py` / `run_ablation.py` của nhánh Nghia với các mô hình trên main (`levels/8x9`, `models/agent_8x9.zip`).

- `tournament_default/`: vòng tròn 8 cấp, 200 ván/cặp, khai cuộc 4 nước ghép cặp đổi bên.
- `tournament_time50ms/`: cấp 3, 4, 7, 8 cùng 50 ms/nước, 100 ván/cặp.
- `ablation/`: A1 độ sâu minimax, A2 số mô phỏng MCTS, B1/B2 mask + lật gương (cấp 5/6), B3 số mô phỏng AlphaZero.
- `agent_8x9_*.csv`: tác tử PPO chính với đối thủ cố định (thuần và `--search 4`) và ablation mask/lật gương (`abl_main.py`).
- `fig/` + `elo.json`: hình và Elo Bradley–Terry, tạo bằng `make_ch4_figs.py <thư mục connect4> <thư mục ra>`.

Log thô từng ván (`games.jsonl`, ~70 MB) không đưa lên.

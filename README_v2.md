# Connect4 v2 - ghi chú

Cài: `pip install torch stable-baselines3 sb3-contrib gymnasium tensorboard`

Train : `python train_v2.py --steps 3000000 --arch cnn --channels 64 --out runs/main` (nối tiếp: thêm `--resume runs/main/last.zip`)
Đánh giá: `python evaluate_v2.py models/agent_v2.zip --old ../drmkn_connect4_src`
Kiểm thử lõi luật: `python test_core.py`

Lưu ý: model v2 PHẢI chơi cùng action mask (TorchPlayer đã làm sẵn). Bỏ mask thì agent thua gần hết.

## Lỗi trong code cũ (đã sửa trong v2)
1. Quan sát thô (quân 1/2, agent không biết mình là bên nào) -> v2 dùng 2 kênh "của mình / của đối thủ".
2. Nước phạm quy = thua -100 và kết thúc ván; 4-10% ván thua vì lý do này -> v2 dùng action mask.
3. `_check_game_over` kiểm tra hoà trước thắng -> nước thứ 42 thắng bị tính hoà.
4. Hoà là +1 hay -1 tuỳ ai đi cuối; `is_success=True` cả khi hoà; README (-50) khác code (-100).
5. Mỗi agent train lại TỪ ĐẦU chỉ với 100k bước, chỉ đấu 1 đối thủ cố định -> học "xếp 4 dọc" khai thác đối thủ liền trước.
6. `test.py` chạy 10.000 ván ngay khi import, chỉ đo với random, bỏ agent15; Ensemble bỏ phiếu không loại nước phạm quy, hoà thì chọn cột trái.
7. `play.py`: bấm vào cột đầy = thua ngay; `MinimaxAgent(1, 2)` cố định ký hiệu nhưng env đổi bên ngẫu nhiên (minimax có thể chơi cho người).
8. minimax cũ: không có hàm đánh giá ở lá, không ưu tiên thắng nhanh, rất chậm (numpy) -> v2 dùng bitboard (depth 8 ~0.1s).

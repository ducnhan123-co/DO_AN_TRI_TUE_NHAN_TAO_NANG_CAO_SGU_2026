# Changelog

## Thang 8 cấp độ (mới)
- Thêm `levels.py`, `mcts.py`, `train_levels.py`, `evaluate_levels.py`, `HeuristicPlayer` (c4core), `--init-sl` (train_v2).
- Cấp 1 Random, 2 Heuristic, 3 Minimax d3, 4 Monte Carlo (MCTS thuần), 5 học có giám sát, 6 học tăng cường, 7-8 AlphaZero.
- `play.py --level N` / menu mục 6; trong game bấm phím 1-8 để đổi cấp.
- Cấp 5/7/8 lưu `.npz` (chỉ mảng số, nạp bằng `allow_pickle=False`).
- Kết quả (20 ván/ô) trong `results/levels_ladder.csv` và README: thứ tự độ khó đúng ở cấp 1-4 và 6-8; cấp 5-8 chưa vượt cấp 4 với 10.000 bước.

## Bản sửa theo review code

Mỗi mục đều được tái hiện trước khi sửa. `python test_regression.py` kiểm tra lại các mục có thể tự động hoá.

### Nghiêm trọng
1. **Model lưu lambda -> nạp ở Python khác có thể segfault** *(đã xác nhận: `learning_rate`, `lr_schedule`, `clip_range` bị cloudpickle vào zip)*.
   `learning_rate` là số thực; lịch giảm gắn sau khi tạo model; `save_model()` loại `lr_schedule`/`clip_range`;
   `load_model()` ghi đè các khoá này bằng hằng số (nạp được cả model cũ). Lớp `C4Extractor` chuyển sang `c4env.py`
   (trước đây là `__main__` nên model CNN chỉ nạp được từ `train_v2.py`). Hai model phát hành đã được lưu lại sạch.
2. **TorchPlayer cứng tensor CPU** *(xác nhận theo mô tả; không có GPU để chạy thật)*. Bản sao mạng luôn chuyển về CPU
   nên model train bằng CUDA/MPS vẫn dùng được; `--device` cho train.

### Quan trọng
3. **Chọn `best.zip` bằng policy khác policy đem chơi** *(có thật; trên 100 ván mình đo chênh nhỏ hơn bản review: 0.35 vs 0.37)*.
   Đánh giá khi train dùng `flip_avg=True` giống lúc chơi, số ván đánh giá tăng (60/60/30).
4. **`with_options(seed=...)` không seed gì** *(xác nhận: các bản sao dùng chung một generator)*. Mỗi bản có generator riêng.
5. **Agent/model lỗi làm sập menu**. Nạp model lỗi -> báo lỗi và quay lại menu; agent ném lỗi / trả nước sai -> thanh
   trạng thái báo và dùng nước dự phòng; model sai kích thước bàn bị từ chối ngay khi nạp.
6. **Cột không kiểm tra biên** *(xác nhận: `drop_bit(0, -1)` trả bit của cột cuối)*. `drop_bit` trả 0 cho cột ngoài bàn.
7. **Chất lượng model** (không phải lỗi code). Thêm **chế độ lai `--search N`**: minimax độ sâu N loại nước dẫn tới thế
   thua; trên 8x9 điểm vs minimax d2 tăng 0.42 -> 0.71 (d4) / 0.85 (d6), vs d4 tăng 0.20 -> 0.57 (d4). Tắt mặc định.
8. **`--resume` ghi đè `best.zip` bằng model tệ hơn**. `best_score.json` được khôi phục khi resume.

### Nhỏ
9. `.gitignore` đúng tên (có dấu chấm). Cần `git rm -r --cached __pycache__ .serena` cho file đã lỡ commit.
10. `test_core.py` có `main()` + `if __name__ == "__main__"`.
11. `match(n_games=0)` ném `ValueError` rõ ràng; `evaluate_v2.py --games 0` báo lỗi tham số.
12. `pool.pt` chỉ lưu tensor + chuỗi, nạp bằng `torch.load(weights_only=True)`. Pool cũ (định dạng pickle) bị bỏ qua kèm cảnh báo.
13. Đường dẫn model mặc định là tuyệt đối; bàn đầy -> `ValueError` rõ ràng thay cho `IndexError`.

### Ghi chú
- `play_v2.py` trong bản review đã được thay bằng `play.py` + `gui.py`; các điểm 5, 6, 13 được kiểm tra trên code mới.
- Chưa kiểm thử GPU/MPS thật.

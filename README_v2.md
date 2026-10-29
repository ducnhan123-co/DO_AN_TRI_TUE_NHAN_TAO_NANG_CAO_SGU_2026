# Connect4 v2 - bàn 8 hàng x 9 cột

Cài: `pip install torch stable-baselines3 sb3-contrib gymnasium tensorboard pygame`

| Việc | Lệnh |
|---|---|
| Chơi (menu + giao diện cũ) | `python play_v2.py` |
| Đánh giá model | `python evaluate_v2.py models/agent_8x9.zip` |
| Train | `python train_v2.py --steps 3000000 --arch cnn --channels 64 --out runs/main` |
| Train tiếp | thêm `--resume runs/main/last.zip` |
| Kiểm thử luật | `python test_core.py` |

* Kích thước bàn nằm ở `c4core.py` (mặc định 8x9, luật vẫn là 4 quân thẳng hàng). Đổi bằng biến môi trường,
  ví dụ PowerShell: `$env:C4_ROWS=6; $env:C4_COLS=7`. Model chỉ dùng được với đúng kích thước đã train.
* Model PHẢI chơi cùng action mask (TorchPlayer đã bật sẵn).
* `play_v2.py` thay `play.py`: bấm vào cột đầy sẽ bị bỏ qua (code cũ xử thua), đối thủ "Heuristic" cũ
  (chỉ hỗ trợ 6x7) được thay bằng "Tactical".

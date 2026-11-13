# Hướng dẫn tự huấn luyện (cấp 5, 6, 7-8) và chạy thí nghiệm

Bản này **không kèm model huấn luyện sẵn**. Cấp 1-4 chạy ngay; cấp 5-8 hiện ở trạng thái "khóa" trong menu cho tới khi bạn train xong.
Mọi lệnh chạy ở thư mục gốc của dự án.

## 0. Cài đặt

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt      # numpy, torch, gymnasium, stable-baselines3, sb3-contrib, pygame, tensorboard
python -m unittest test_arena -v     # kiểm thử nhanh (không cần train)
python test_core.py
```

Python >= 3.10. Chỉ cần CPU. Chọn bàn bằng `--board 8x9` (hàng x cột, mặc định của nhóm) hoặc `--board 6x7`
(bàn cổ điển, **đề cuối kỳ khuyến nghị**; bàn khác cần giảng viên duyệt). **Model chỉ dùng được đúng cỡ bàn đã train**,
nên train riêng cho từng bàn (model được lưu ở `levels/<hàng>x<cột>/`).

## 1. Train

```bash
# Cả ba cấp học cho bàn 8x9 (10.000 bước mỗi cấp - ngân sách mặc định, khá nhỏ)
python train_levels.py --levels 5-8

# Bàn 6x7 cổ điển (đề khuyến nghị)
python train_levels.py --board 6x7 --levels 5-8
```

Từng cấp, theo thứ tự phụ thuộc (cấp 6 mặc định khởi tạo từ mạng cấp 5):

| Cấp | Lệnh | Phương pháp |
|---|---|---|
| 5 | `python train_levels.py --levels 5` | sinh dữ liệu bằng minimax độ sâu 4 (`--teacher-depth`), học bắt chước; dữ liệu **tách train/val/test 80/10/10 theo ván**; chọn checkpoint theo VAL, TEST đo đúng 1 lần |
| 6 | `python train_levels.py --levels 6` | MaskablePPO tự chơi với bể đối thủ (gọi `train_v2.py`); `--rl-scratch` để khởi tạo ngẫu nhiên thay vì từ cấp 5 |
| 7-8 | `python train_levels.py --levels 7-8` | AlphaZero (tự chơi + PUCT); cấp 7 và 8 **dùng chung một mạng**, chỉ khác số mô phỏng lúc chơi (100/800) |

Tham số hay dùng: `--steps N` (số bước mỗi cấp), `--seed S`, `--sl-positions N` (số thế cờ giám sát, mặc định 60.000),
`--teacher-depth D`, `--az-train-sims`, `--az-games-per-iter`, `--az-steps-per-iter`. Xem `python train_levels.py --help`.

### Chạy lâu / ngắt quãng

`--max-seconds 1800` giới hạn thời gian mỗi lần chạy; **chạy lại đúng lệnh đó để tiếp tục** (dữ liệu giám sát và trạng thái
AlphaZero được lưu lại: `sl_data.npz`, `az_state.pt`, `az_buffer.npz`). Đừng đổi `--seed`/`--steps` giữa các lần tiếp tục.

### Thời gian

Tôi **chưa đo** thời gian train trên máy của bạn nên không đưa số cụ thể. Cách ước lượng: chạy `--steps 1000 --sl-positions 3000`
một lần, xem `total_seconds` trong `meta_*.json`, rồi nhân tỉ lệ. Chi phí chủ yếu nằm ở sinh dữ liệu minimax (cấp 5),
PPO (cấp 6) và tự chơi MCTS (cấp 7-8; tăng theo số mô phỏng). Ngân sách mặc định 10.000 bước là để chạy thử; kết quả của
bản huấn luyện nhỏ có thể **chưa vượt cấp 4**. Muốn mạnh hơn: tăng `--steps` (vd 50.000-200.000) và `--sl-positions`.

### File tạo ra (trong `levels/<hàng>x<cột>/`)

| File | Nội dung |
|---|---|
| `level5_supervised.npz`, `level7_8_alphazero.npz` | trọng số (chỉ mảng số, nạp bằng `allow_pickle=False`) |
| `level6_rl.zip` | model MaskablePPO (stable-baselines3) |
| `train_log_level5.csv`, `train_log_az.csv` | đường cong học (vẽ biểu đồ đưa vào báo cáo) |
| `meta_level5.json`, `meta_level6.json`, `meta_level7_8_alphazero.json` | seed, toàn bộ tham số, phần cứng, tổng thời gian, ngày giờ; cấp 5 ghi thêm kích thước train/val/test và độ chính xác VAL/TEST |
| `sl_data.npz`, `az_buffer.npz`, `az_state.pt`, `rl_run/` | dữ liệu/trạng thái trung gian (đã `.gitignore`) |

Muốn train lại từ đầu: xóa thư mục `levels/<hàng>x<cột>/` (hoặc các file tương ứng) rồi chạy lại. Kết quả thí nghiệm
(`results/`) không bị ảnh hưởng, nhưng nên chạy lại thí nghiệm sau khi train lại.

## 2. Chơi và ghi kết quả trong giao diện

```bash
python play.py                 # bàn 8x9
python play.py --board 6x7
```

Menu trong cửa sổ: chọn **Người / cấp 1-8** cho bên A (cam) và B (đỏ), bàn cờ (bấm để đổi - chương trình tự khởi động lại),
ai đi trước, đổi bên mỗi ván, số ván (loạt máy-máy), khai cuộc ngẫu nhiên, thời gian/nước (cho cấp 3, 4, 7, 8). Cấp chưa train bị khóa.
Mỗi ván xong được ghi vào `results/play_log/games.jsonl`; nút **Kết quả** hiện bảng thắng/hòa/thua, điểm, khoảng Wilson 95%,
ms/nước và xuất `summary.csv`. Phím: `R` ván mới, `Space` tạm dừng, `1-9` thả quân, `Esc` về menu.

> Số liệu **chính thức** cho báo cáo nên lấy từ `tournament.py` (mục 3), không phải từ ván chơi trong giao diện.

## 3. Thí nghiệm theo đề cuối kỳ

```bash
# Round-robin cấp 1-8, mỗi cặp 200 ván (2 seed x 100), hai chế độ ngân sách
python tournament.py --board 6x7 --levels 1-8 --seeds 0,1 --games-per-seed 100 --mode all --budget-ms 100

# Chỉ cấp 1-4 (không cần train)
python tournament.py --board 6x7 --levels 1-4 --mode all

# Khuyến nghị đề: 4 seed
python tournament.py --board 6x7 --levels 1-8 --seeds 0,1,2,3 --games-per-seed 100 --mode default
```

Kết quả: `results/tournament_<bàn>/{default,time}/` gồm `config.json` (cấu hình + phần cứng), `games.jsonl` (log thô từng ván: nước đi,
ms/nước, số nút, số mô phỏng), `pairs/` (từng cặp; **chạy lại lệnh sẽ tự bỏ qua các cặp đã xong**, thêm `--no-resume` để làm lại),
`summary.json`, `summary.md` (xếp hạng, thắng/hòa/thua, score + Wilson 95% + bootstrap 95%, theo bên đi trước/sau và theo seed, ms/nước TB/p95/max, nút, mô phỏng).

Thiết kế: mỗi khai cuộc (4 nước ngẫu nhiên) chơi **hai lần đổi bên** (cân bằng lượt đi); khai cuộc theo seed, dùng chung mọi cặp;
seed từng ván suy ra từ (seed, cặp, ván); chạy tuần tự trong một tiến trình để đo thời gian ít nhiễu.
Chế độ `default` = cấu hình mặc định mỗi cấp; chế độ `time` = cùng `--budget-ms` ms/nước cho cấp 3 (minimax đào sâu dần), 4 (MCTS),
7-8 (AlphaZero); cấp 1, 2, 5, 6 không tìm kiếm nên giữ nguyên. Cấp chưa train bị bỏ qua kèm thông báo.

### Ablation (>= 2 thí nghiệm)

```bash
python run_ablation.py --list
python run_ablation.py --board 6x7 --only A1,A2          # không cần train: độ sâu minimax, số mô phỏng MCTS
python run_ablation.py --board 6x7 --only B1,B2,B3       # sau khi train: shield/flip (cấp 5, 6), tìm kiếm lai, số mô phỏng AlphaZero
```

Mỗi biến thể đấu >= 100 ván (2 seed x 50) với một đối thủ neo cố định; kết quả `results/ablation_<bàn>/<id>.{json,md}`.
**Ablation cần đúng cỡ bàn đã train** (B1-B3).

## 4. Đưa vào báo cáo

* **Tái lập:** ghi seed, lệnh đã chạy, `config.json`/`meta_*.json` (phần cứng, phiên bản Python/torch), commit của mã.
* **Nguồn gốc:** cấp 1-4 và khung tournament/ablation/UI menu do nhóm tự làm; cấp 5-8 là cài đặt của nhóm
  (`train_levels.py`, `mcts.py`, `c4env.py`); AlphaZero ở đây là phiên bản **"lấy cảm hứng"** (MLP nhỏ, tự chơi quy mô nhỏ) - phải ghi đúng như vậy.
* **Hạn chế cần nêu:** model train 10.000 bước rất nhỏ; bàn 8x9 khác bàn khuyến nghị 7x6 (cần giảng viên duyệt);
  số đo thời gian phụ thuộc phần cứng; khai cuộc ngẫu nhiên 4 nước có thể tạo thế thắng cưỡng bức cho một bên (đổi bên bù trừ).

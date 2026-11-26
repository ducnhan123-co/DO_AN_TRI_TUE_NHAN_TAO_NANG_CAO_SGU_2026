# Hướng dẫn tự huấn luyện (cấp 5, 6, 7-8) và chạy thí nghiệm

Bản này **không kèm model huấn luyện sẵn**. Cấp 1-4 chạy ngay; cấp 5-8 hiện ở trạng thái "khóa" trong menu cho tới khi bạn train xong.
Mọi lệnh chạy ở thư mục gốc của dự án.

## 0. Cài đặt

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt      # numpy, torch, gymnasium, stable-baselines3, sb3-contrib, pygame, tensorboard
python -m unittest tests.test_arena -v     # kiểm thử nhanh (không cần train)
python tests/test_core.py
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
| 6 | `python train_levels.py --levels 6` | MaskablePPO tự chơi với bể đối thủ (gọi `train.py`); `--rl-scratch` để khởi tạo ngẫu nhiên thay vì từ cấp 5 |
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

Kết quả: một thư mục `results/runs/<mã-thí-nghiệm>/` cho mỗi chế độ (cấu trúc ở **mục 5**) và một dòng trong nhật ký Excel; xem thêm `summary.md` (xếp hạng, thắng/hòa/thua, score + Wilson 95% + bootstrap 95%, theo bên đi trước/sau và theo seed, ms/nước TB/p95/max, nút, mô phỏng).

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

Mỗi biến thể đấu >= 100 ván (2 seed x 50) với một đối thủ neo cố định; kết quả trong `results/runs/<mã>/` (mục 5) và sheet **Ablation** của nhật ký.
**Ablation cần đúng cỡ bàn đã train** (B1-B3).

## 4. Đưa vào báo cáo

* **Tái lập:** ghi seed, lệnh đã chạy, `config.json`/`meta_*.json` (phần cứng, phiên bản Python/torch), commit của mã.
* **Nguồn gốc:** cấp 1-4 và khung tournament/ablation/UI menu do nhóm tự làm; cấp 5-8 là cài đặt của nhóm
  (`train_levels.py`, `mcts.py`, `env.py`); AlphaZero ở đây là phiên bản **"lấy cảm hứng"** (MLP nhỏ, tự chơi quy mô nhỏ) - phải ghi đúng như vậy.
* **Hạn chế cần nêu:** model train 10.000 bước rất nhỏ; bàn 8x9 khác bàn khuyến nghị 7x6 (cần giảng viên duyệt);
  số đo thời gian phụ thuộc phần cứng; khai cuộc ngẫu nhiên 4 nước có thể tạo thế thắng cưỡng bức cho một bên (đổi bên bù trừ).

## 5. Thư mục kết quả + Nhật ký thí nghiệm (Excel / Google Sheets)

Mỗi lần chạy `tournament.py`, `run_ablation.py` hoặc `train_levels.py` tạo **một thư mục** `results/runs/<mã-thí-nghiệm>/`
(mã = `ngày-giờ_loại_bàn_nhãn`) và **thêm một dòng vào nhật ký** `results/NHAT_KY_THI_NGHIEM.xlsx`.

| File trong thư mục lần chạy | Nội dung | Tương ứng trong thí nghiệm học máy |
|---|---|---|
| `app_cfg.json` | lệnh chạy, mục tiêu, giả thuyết, bàn, seed, số ván, ngân sách, thời gian, **mã băm code**, phần cứng/phiên bản thư viện | `app_cfg.json` |
| `params_cfg.json` | tham số **thực** của từng tác tử (độ sâu, số mô phỏng, shield, ...; sha256 của model); với train: toàn bộ tham số train | `params_cfg.json` |
| `run.log.txt` / `train.log.txt` | toàn bộ log console | `train.log.txt` |
| `games_seed<k>.csv`, `games_all.csv` | log từng ván: cặp, khai cuộc, bên đi trước, kết quả, số nước, ms/nước, chuỗi nước đi | `train_logs_fold<k>.csv`, `train_logs_all.csv` |
| `train_logs_level5.csv`, ... (lần train) | đường cong học | `train_logs_*.csv` |
| `metrics_seed.csv` | số đo từng tác tử theo từng seed (W/D/L, score, Wilson, đi trước/sau, ms, p95, nút, mô phỏng) | `metrics_best.csv` |
| `metrics_seed_mean.json` | trung bình ± độ lệch chuẩn qua các seed, score gộp + Wilson 95% | `metrics_best_mean.json` |
| `metrics_pairs.csv` | từng cặp (gộp seed): W/D/L, score, Wilson, bootstrap | `metrics_avg.csv` |
| `metrics_overall.json` | xếp hạng tổng + chi phí tính toán (lần train: val/test acc, sha256 model) | `metrics_avg_mean.json` |
| `summary.md` | báo cáo đọc nhanh | |
| `models/` | bản sao model đã dùng/được train (nếu < 60 MB) | `best_model_fold*.pt` |
| `data/` | log thô `games.jsonl` (nước đi, ms, nút, mô phỏng), `pairs/` để chạy tiếp | `data/` |

Chạy tiếp một tournament dở: `python tournament.py ... --resume results/runs/<mã>` (bỏ qua các cặp đã xong).

### Điền nhật ký đúng cách (như sổ thí nghiệm khoa học)

```bash
python tournament.py --board 6x7 --levels 1-8 --seeds 0,1,2,3 --games-per-seed 100 --mode all \
  --goal "So sánh 8 cấp ở cấu hình mặc định và cùng 100 ms/nước" \
  --hypothesis "Điểm tăng theo cấp; AlphaZero-800 > AlphaZero-100; cấp 5 chưa vượt cấp 4 với 10.000 bước" \
  --author "Trần Trọng Nghĩa" --tag baseline
```

* **Trước khi chạy:** viết `--goal` và `--hypothesis` (giả thuyết phải có trước khi thấy kết quả).
* **Sau khi chạy:** mở `NHAT_KY_THI_NGHIEM.xlsx`, sheet **Nhật ký**, điền tay cột **Kết luận** (giả thuyết đúng/sai, vì sao, độ tin cậy dựa vào khoảng Wilson),
  **Bất thường / ghi chú** và **Hành động tiếp theo**. Chạy lại cùng mã TN sẽ ghi đè các cột tự động nhưng **giữ nguyên phần bạn đã điền tay**.
* Các sheet khác: **Kết quả cặp**, **Số đo theo tác tử**, **Ablation**, **Huấn luyện** (đều khóa bằng Mã TN), **Từ điển** (giải thích cột).
* Cột **Mã băm code** + seed + `params_cfg.json` = đủ để tái lập; đổi code thì mã băm đổi.

### Dùng Google Trang tính (Google Sheets)

Chương trình ghi file `.xlsx` (chạy offline, không cần tài khoản). Để dùng trên Google Sheets: tải `NHAT_KY_THI_NGHIEM.xlsx` lên Google Drive →
mở bằng Google Trang tính (hoặc Trang tính → Tệp → Nhập). Nếu nhiều người cùng ghi, hãy chạy trên một máy rồi chia sẻ file đó; **chương trình không tự đồng bộ lên Google Sheets**.

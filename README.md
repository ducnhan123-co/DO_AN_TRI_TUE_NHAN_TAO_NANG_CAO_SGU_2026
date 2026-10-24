# DO_AN_TRI_TUE_NHAN_TAO_NANG_CAO_SGU_2026

Đồ án môn Trí tuệ nhân tạo nâng cao (841458) — SGU, HK1 2026-2027.

## Cấu trúc nhánh

| Nhánh | Nội dung |
|---|---|
| `main` | README tổng |
| `lab01` | Bài tập nhóm Agent (Gridhunt, Lunar Lander, Robot vacuum) |
| `connect4` | Đồ án Connect 4 bằng Reinforcement Learning — thư mục `drmkn_connect4_src/` |

## Connect 4 — chạy nhanh

```bash
cd drmkn_connect4_src
./run_ui.sh            # UI web tĩnh: http://127.0.0.1:8777/ui/ (không cần cài gì ngoài python3)

pip install gymnasium stable-baselines3 pygame tqdm matplotlib
python play.py         # chơi với HeuristicAgent bằng pygame
```

Chi tiết kiến trúc, reward và các lỗi đã biết: xem [`drmkn_connect4_src/NOTES.md`](drmkn_connect4_src/NOTES.md).

# 🔴 Connect 4 AI - Hệ Thống 8 Cấp Độ Trí Tuệ Nhân Tạo (Bàn cờ 8x9) 🟡
> **Đồ Án Môn Học: Trí Tuệ Nhân Tạo Nâng Cao (TTNTNC) - SGU 2026**

Dự án phát triển hệ sinh thái tác tử AI toàn diện cho trò chơi **Connect 4 mở rộng (8 hàng × 9 cột)** với đầy đủ **8 Cấp độ AI từ cơ bản đến nâng cao theo yêu cầu của Thầy**:
1. 🎲 **Cấp 1: Random Agent** — Ra quyết định ngẫu nhiên hợp lệ.
2. 🧠 **Cấp 2: Heuristic Agent** — Đánh giá thế trận theo hàm điểm cục bộ, bắt 4 và chặn 3 tức thời.
3. ⚔️ **Cấp 3: Minimax Agent** — Cây trò chơi đối kháng với cắt tỉa Alpha-Beta và hàm định giá nút lá.
4. 🌲 **Cấp 4: Monte Carlo Tree Search (MCTS)** — Tìm kiếm cây Monte Carlo 4 pha (Selection, Expansion, Rollout, Backpropagation).
5. 🎓 **Cấp 5: Học Có Giám Sát (Supervised Learning)** — Mạng nơ-ron tích chập (Deep CNN) học từ bộ dữ liệu thế cờ chuyên gia.
6. 🎯 **Cấp 6: Học Tăng Cường (Reinforcement Learning - PPO)** — Mô hình tự học thông qua phần thưởng trực tiếp từ môi trường Gym.
7. 🚀 **Cấp 7: AlphaZero Foundation** — Kiến trúc chuẩn AlphaZero của DeepMind: Conv2D Policy-Value kết hợp MCTS PUCT.
8. 👑 **Cấp 8: AlphaZero Master (Self-Play League)** — Tự chơi qua nhiều thế hệ đối kháng lịch sử, nhận diện bẫy 2 đầu (Fork) và tự học thích nghi trực tuyến từ người chơi.

---

## 📌 1. Các Tính Năng Nổi Bật
- **Chuẩn kích thước bàn cờ 8x9**: $8 \times 9 = 72$ ô cờ, $9$ hành động (cột 0 - 8).
- **Đầy đủ 8 Cấp độ AI độc lập**: Chuyển đổi mượt mà giữa các cấp độ trực tiếp trong GUI hoặc phím tắt `[1]` đến `[8]`.
- **Huấn luyện tự động 10.000 bước/cấp**: Script `train_8_levels.py --steps 10000` tự động chạy và lưu toàn bộ 8 mô hình vào `saved_agents_8x9/`.
- **Bảng kiểm thử hiệu năng toàn diện**: Script `test_8_levels.py` đối đầu với Random và Heuristic để đo tỷ lệ thắng của từng cấp độ.
- **Học Thích Nghi Trực Tuyến (Online Continual Learning)**: AI ghi nhớ bẫy của người chơi sau mỗi trận để chặn trước trong các ván tiếp theo.
- **Giao diện Pygame hiện đại (Modern Dark Aesthetic)**: Đồ họa chất lượng cao, hiệu ứng thả quân vật lý, đổi lượt đi `[T]`, tự đấu tự học `[A]`.
- **Hỗ trợ Google Colab 100%**: File `Colab_Connect4_Train.ipynb` cho phép huấn luyện GPU T4 trên đám mây và đồng bộ Google Drive.

---

## 📂 2. Cấu Trúc Thư Mục Dự Án
```text
drmkn_connect4_source/
├── README.md                           # Hướng dẫn chi tiết đồ án
├── Colab_Connect4_Train.ipynb           # Notebook huấn luyện GPU T4 trên Google Colab
└── drmkn_connect4_src/                 # Thư mục mã nguồn chính
    ├── play.py                         # Khởi chạy Arena thi đấu (chọn Cấp 1 -> 8)
    ├── train_8_levels.py               # Script huấn luyện tự động 8 Cấp độ (10.000 bước/cấp)
    ├── test_8_levels.py                # Script kiểm thử đo tỷ lệ thắng 8 Cấp độ
    ├── agent_manager.py                # Bộ quản lý và nạp 8 Cấp độ AI
    ├── random_agent.py                 # Cấp 1: Random Agent
    ├── heuristic.py                    # Cấp 2: Heuristic Agent
    ├── minimax.py                      # Cấp 3: Minimax với Alpha-Beta Pruning
    ├── mcts_agent.py                   # Cấp 4: Monte Carlo Tree Search
    ├── supervised_agent.py             # Cấp 5: Học Có Giám Sát (Deep CNN)
    ├── rl_agent.py                     # Cấp 6: Học Tăng Cường (PPO Agent)
    ├── alphazero_agent.py              # Cấp 7: AlphaZero Foundation (CNN + PUCT)
    ├── self_play.py                    # Cấp 8: AlphaZero Master (Self-Play League)
    ├── adaptive_learner.py             # Module Học Thích Nghi Trực Tuyến (Online Learning)
    ├── ensemble.py                     # Ensemble PPO Agents
    ├── gui.py                          # Giao diện đồ họa Pygame bàn cờ 8x9
    ├── connect4.py                     # Môi trường Gym Connect-4 chuẩn bàn 8x9
    ├── requirements.txt                # Thư viện cần thiết
    └── saved_agents_8x9/               # Thư mục lưu trữ weights của các Cấp độ AI
```

---

## 🚀 3. Hướng Dẫn Cài Đặt

### Bước 1: Di chuyển vào thư mục mã nguồn
```bash
cd drmkn_connect4_source/drmkn_connect4_src
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 🎯 4. Hướng Dẫn Huấn Luyện 8 Cấp Độ (`train_8_levels.py`)

Chạy lệnh bên dưới để huấn luyện **toàn bộ 8 cấp độ**, mỗi cấp độ đúng **10.000 bước**:
```bash
python train_8_levels.py --steps 10000
```
> Nếu chỉ muốn huấn luyện một cấp độ cụ thể (ví dụ Cấp 5 hoặc Cấp 8):
```bash
python train_8_levels.py --steps 10000 --level 5
python train_8_levels.py --steps 10000 --level 8
```

---

## 📊 5. Kiểm Thử Đo Tỷ Lệ Thắng 8 Cấp Độ (`test_8_levels.py`)

Chạy kiểm thử đo đạc tỷ lệ thắng và thời gian tính toán của cả 8 Cấp độ:
```bash
python test_8_levels.py --games 10
# Hoặc thông qua test.py
python test.py levels 10
```

---

## 🎮 6. Hướng Dẫn Chơi Game (`play.py`)

Khởi động giao diện bàn cờ 8x9:
```bash
python play.py
```
> Mặc định game sẽ khởi động với **Cấp 8 (AlphaZero Master)**. Để chọn cấp độ khác từ dòng lệnh:
```bash
python play.py --level 4    # Đấu với Cấp 4 (MCTS)
python play.py --level 7    # Đấu với Cấp 7 (AlphaZero)
```

### Phím Tắt và Điều Khiển Trong Game:
* 🔢 **Phím `[1]` đến `[8]`**: **Chuyển đổi trực tiếp Cấp độ AI đối thủ ngay khi đang chơi!** (Hoặc click trực tiếp vào hàng nút Cấp độ ở đầu màn hình).
* 🖱️ **Chuột trái**: Di chuột để ngắm và click để thả quân cờ đỏ (🔴).
* 🔄 **Phím `[R]`**: Khởi động lại ván cờ mới.
* 👥 **Phím `[T]`**: Đổi lượt đi (**Bạn đi trước** hoặc **AI đi trước**).
* ⚡ **Phím `[A]`**: **Bật/Tắt chế độ AI Tự Đấu Tự Học** (Hai AI tự chơi liên tục và tự học thích nghi).
* ❌ **Phím `[ESC]`**: Thoát khỏi game.
> - Máy tính của bạn hoàn toàn mát mẻ, không bị chiếm dụng CPU hay RAM.

> ⚠️ **LƯU Ý QUAN TRỌNG:**  
> Các file `agent0.zip` đến `agent15.zip` là định dạng lưu mặc định của thư viện Stable-Baselines3. **KHÔNG GIẢI NÉN** các file `agent*.zip` này ra thư mục, hãy để nguyên đuôi `.zip`.

---

## 📊 6. Đánh Giá Hiệu Năng & Tỷ Lệ Thắng (`test.py`)

Dự án cung cấp 2 chế độ kiểm thử benchmark:

### Chế độ 1: Kiểm thử toàn bộ tất cả Agent (`test all`)
Kiểm thử và so sánh lần lượt 16 Agents PPO, Ensemble Agent, Heuristic và Minimax:
```bash
python test.py all
# Hoặc chỉ định số ván cho mỗi agent (ví dụ 10 hoặc 20 ván)
python test.py all 15
```

### Chế độ 2: Kiểm thử tập trung Ensemble PPO mạnh nhất
```bash
python test.py 100
```
Script sẽ tự động chạy mô phỏng không cần giao diện đồ họa và xuất tỷ lệ thắng (Win Rate) ra màn hình console.

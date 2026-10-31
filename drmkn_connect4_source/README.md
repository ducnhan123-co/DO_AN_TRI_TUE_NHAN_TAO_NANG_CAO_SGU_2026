# 🔴 Connect 4 AI - Reinforcement Learning (Bàn cờ 8x9) 🟡
> **Đồ Án Môn Học: Trí Tuệ Nhân Tạo Nâng Cao (TTNTNC) - SGU 2026**

Dự án phát triển tác tử AI chơi cờ **Connect 4 mở rộng (8 hàng × 9 cột)** sử dụng phương pháp **Học tăng cường (Curriculum Reinforcement Learning - PPO)** kết hợp **Ensemble Learning**, cùng các tác tử đối chuẩn thông minh (**Heuristic** và **Minimax**).

---

## 📌 1. Các Tính Năng Nổi Bật
- **Kích thước bàn cờ 8x9**: Không gian trạng thái $8 \times 9 = 72$ ô cờ, $9$ hành động (cột 0 - 8).
- **Hệ thống AI đa dạng**:
  - 🧠 **Ensemble PPO Agents (Mạnh nhất)**: Kết hợp biểu quyết từ chuỗi 16 mô hình PPO được huấn luyện qua kỹ thuật Curriculum Learning.
  - ⚡ **Học Thích Nghi Trực Tuyến (Online Continual Learning)**: AI tự động phân tích và học hỏi nước cờ của người chơi sau mỗi ván đấu, tự động nhớ bẫy và cập nhật mạng nơ-ron PPO để chặn trước trong các ván tiếp theo.
  - ⚡ **Heuristic Agent**: Đánh giá thế cờ theo kinh nghiệm, tự động chặn chuỗi 3, bắt chuỗi 4 và kiểm soát khu vực trung tâm.
  - 🔍 **Minimax Agent**: Tìm kiếm vét cạn nước đi tối ưu kết hợp cắt tỉa Alpha-Beta.
- **Giao diện Pygame đồ họa cao cấp**:
  - Hiệu ứng thả quân chân thực, đổ bóng, hạt phát sáng (particle effects).
  - Tùy chọn đi trước / đi sau trực tiếp trên màn hình (`[T]`).
  - Tự động phát hiện 4 quân thắng cuộc và viền sáng hiệu ứng.
- **Huấn luyện đám mây độc lập qua Google Colab**:
  - Không làm nặng máy hay nóng laptop cá nhân.
  - Tự động lưu checkpoint an toàn vào Google Drive.

---

## 📂 2. Cấu Trúc Thư Mục Dự Án
```text
drmkn_connect4_source/
├── Colab_Connect4_Train.ipynb       # Notebook huấn luyện 16 PPO agents trên Google Colab
├── README.md                       # Hướng dẫn chi tiết dự án
└── drmkn_connect4_src/             # Thư mục mã nguồn chính của game
    ├── connect4.py                 # Môi trường Gym Connect 4 (8 hàng x 9 cột)
    ├── gui.py                      # Giao diện đồ họa Pygame bàn cờ 8x9
    ├── play.py                     # File chính để khởi động và chơi game
    ├── heuristic.py                # Thuật toán Heuristic Agent
    ├── minimax.py                  # Thuật toán Minimax (Alpha-Beta Pruning)
    ├── ensemble.py                 # Thuật toán Ensemble Agent (Biểu quyết đa mô hình)
    ├── adaptive_learner.py         # Module Tự Học Thích Nghi Trực Tuyến từ người chơi
    ├── self_play.py                # Huấn luyện AlphaZero Self-Play + CNN 2D
    ├── train.py                    # Script huấn luyện cục bộ
    ├── test.py                     # Script đo đạc tỷ lệ thắng (Benchmark Win-rate)
    ├── requirements.txt            # Danh sách thư viện cần thiết
    └── saved_agents_8x9/           # Thư mục chứa 16 file weights (agent0.zip ... agent15.zip)
```

---

## 🚀 3. Hướng Dẫn Cài Đặt (Cho Bạn Bè / Thành Viên Nhóm)

### Bước 1: Clone mã nguồn về máy tính
```bash
git clone https://github.com/ducnhan123-co/DO_AN_TRI_TUE_NHAN_TAO_NANG_CAO_SGU_2026.git
cd DO_AN_TRI_TUE_NHAN_TAO_NANG_CAO_SGU_2026/drmkn_connect4_source/drmkn_connect4_src
```

### Bước 2: Cài đặt các thư viện Python
Yêu cầu Python 3.9 trở lên:
```bash
pip install -r requirements.txt
```
*(Bao gồm: `gymnasium`, `stable-baselines3`, `pygame`, `torch`, `numpy`, `tqdm`, `tensorboard`)*

---

## 🎮 4. Hướng Dẫn Chơi Game Trên Máy Tính (`play.py`)

Sau khi cài đặt xong, bạn chỉ cần chạy lệnh:
```bash
python play.py
```

### Cách điều khiển trong Game:
* 🖱️ **Chuột trái**: Rê chuột qua các cột để ngắm và click để thả quân cờ đỏ (🔴).
* 🔄 **Phím `[R]`**: Khởi động lại ván cờ mới.
* 👥 **Phím `[T]`**: Đổi lượt (**Bạn đi trước** hoặc **AI đi trước**).
* ⚡ **Phím `[A]`**: **Bật/Tắt chế độ Tự Đấu Tự Học (Auto-Play)** — Hai AI sẽ tự động đấu liên tục trên màn hình và tự học thích nghi sau mỗi ván mà bạn không cần phải click chuột!
* ❌ **Phím `[ESC]`**: Thoát khỏi game.

> **💡 Cơ chế nạp AI tự động:**
> - Nếu thư mục `saved_agents_8x9/` đã có đủ các file `agent0.zip` ... `agent15.zip`, game sẽ nạp **Tổ hợp Ensemble 16 AI PPO** mạnh nhất.
> - Nếu chưa có file weights, game sẽ tự động dùng **Heuristic Agent** thông minh để bạn chơi thử ngay mà không bị báo lỗi.

---

## ☁️ 5. Hướng Dẫn Chạy Toàn Bộ Dự Án Trên Google Colab

Toàn bộ dự án đã được tích hợp liên kết trực tiếp với **Google Colab** qua file `Colab_Connect4_Train.ipynb`:

1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn **Upload** và tải lên file **`Colab_Connect4_Train.ipynb`** (nằm ở thư mục dự án).
3. Vào menu **Runtime** ➜ **Change runtime type** ➜ Chọn **T4 GPU** (hoàn toàn miễn phí).
4. **Các bước chạy trực tiếp trên Colab:**
   - 🔗 **Bước 1 (Kết nối toàn diện)**: Tự động kết nối Google Drive, clone toàn bộ mã nguồn mới nhất từ GitHub và cài đặt thư viện.
   - 📊 **Bước 2 (Kiểm thử Win-rate)**: Chạy `!python test.py all 15` để đo tỷ lệ thắng của toàn bộ 19 biến thể AI ngay trên đám mây.
   - 🧠 **Bước 3 (Huấn luyện Curriculum PPO)**: Huấn luyện chuỗi 16 agents (tự động đồng bộ checkpoint sang Google Drive).
   - 🚀 **Bước 4 (Huấn luyện AlphaZero Self-Play)**: Huấn luyện mạng tích chập Conv2D tự đấu qua các thế hệ.
   - 📦 **Bước 5 (Tải về máy tính)**: Tự động nén và tải file `saved_agents_8x9.zip` về máy tính chỉ với 1 click!

> 💡 **Ưu điểm khi chạy trên Colab:**
> - Sử dụng GPU T4 mạnh mẽ, tốc độ train nhanh gấp 5-10 lần so với laptop.
> - Checkpoints được lưu trực tiếp vào Google Drive (`Connect4_Models/saved_agents_8x9`), không lo mất dữ liệu nếu mất mạng.
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

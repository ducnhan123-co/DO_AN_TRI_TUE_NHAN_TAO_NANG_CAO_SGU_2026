# 🔴 Connect 4 AI - Reinforcement Learning (Bàn cờ 8x9) 🟡
> **Đồ Án Môn Học: Trí Tuệ Nhân Tạo Nâng Cao (TTNTNC) - SGU 2026**

Dự án phát triển tác tử AI chơi cờ **Connect 4 mở rộng (8 hàng × 9 cột)** sử dụng phương pháp **Học tăng cường (Curriculum Reinforcement Learning - PPO)** kết hợp **Ensemble Learning**, cùng các tác tử đối chuẩn thông minh (**Heuristic** và **Minimax**).

---

## 📌 1. Các Tính Năng Nổi Bật
- **Kích thước bàn cờ 8x9**: Không gian trạng thái $8 \times 9 = 72$ ô cờ, $9$ hành động (cột 0 - 8).
- **Hệ thống AI đa dạng**:
  - 🧠 **Ensemble PPO Agents (Mạnh nhất)**: Kết hợp biểu quyết từ chuỗi 16 mô hình PPO được huấn luyện qua kỹ thuật Curriculum Learning.
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
* 👥 **Phím `[T]`** hoặc nút trên góc trái: Đổi lượt (**Bạn đi trước** hoặc **AI đi trước**).
* ❌ **Phím `[ESC]`**: Thoát khỏi game.

> **💡 Cơ chế nạp AI tự động:**
> - Nếu thư mục `saved_agents_8x9/` đã có đủ các file `agent0.zip` ... `agent15.zip`, game sẽ nạp **Tổ hợp Ensemble 16 AI PPO** mạnh nhất.
> - Nếu chưa có file weights, game sẽ tự động dùng **Heuristic Agent** thông minh để bạn chơi thử ngay mà không bị báo lỗi.

---

## ☁️ 5. Hướng Dẫn Tự Huấn Luyện AI Bằng Google Colab

Nếu bạn muốn tự train lại hoặc huấn luyện thêm agent mới mà không làm nóng laptop:

1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn **Upload** và tải lên file **`Colab_Connect4_Train.ipynb`** (nằm ở thư mục dự án).
3. Vào menu **Runtime** ➜ **Change runtime type** ➜ Chọn **T4 GPU** (hoặc CPU High-RAM).
4. Bấm **Chạy tất cả (Run all)**:
   - Colab sẽ tự động cài thư viện và kết nối Google Drive của bạn.
   - Quá trình Curriculum Learning sẽ tự train chuỗi 16 agents (Agent sau học cách khắc chế Agent trước).
   - Checkpoint được lưu thẳng vào Google Drive (`Connect4_Models/saved_agents_8x9`).
5. **Tải về máy tính:**
   - Chạy ô cuối cùng (**Bước 5**) trong notebook, trình duyệt sẽ tự động tải file `saved_agents_8x9.zip` về máy.
   - Giải nén file zip để lấy các file con (`agent0.zip` ... `agent15.zip`).
   - Đặt toàn bộ 16 file con này vào thư mục:
     ```text
     drmkn_connect4_source/drmkn_connect4_src/saved_agents_8x9/
     ```
   - Chạy `python play.py` trên máy tính để trải nghiệm AI vừa train!

> ⚠️ **LƯU Ý QUAN TRỌNG:**  
> Các file `agent0.zip` đến `agent15.zip` là định dạng lưu mặc định của thư viện Stable-Baselines3. **KHÔNG GIẢI NÉN** các file `agent*.zip` này ra thư mục, hãy để nguyên đuôi `.zip`.

---

## 📊 6. Đánh Giá Hiệu Năng & Tỷ Lệ Thắng (`test.py`)
Để đo đạc tỷ lệ thắng của Ensemble Agent qua hàng ngàn ván đấu đối chuẩn:
```bash
python test.py
```
Script sẽ tự động chạy mô phỏng không cần giao diện đồ họa và xuất tỷ lệ thắng (Win Rate) ra màn hình console.

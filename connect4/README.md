# Connect Four - Game AI cấp 1-4


## Cấu trúc

```
c4ai/
  game.py            Engine bitboard + API: reset, legal_actions, step, is_terminal, result,
                     reward, render, clone, undo (+ key, from_moves, from_grid)
  evaluation_fn.py   Hàm lượng giá tĩnh (69 cửa sổ 4 ô + kiểm soát cột giữa)
  agents/
    random_agent.py  Cấp 1  Random
    heuristic.py     Cấp 2  Rule-based (thắng > chặn > tránh tự sát > vị trí; bật/tắt từng luật)
    alphabeta.py     Cấp 3  Negamax + Alpha-Beta (tắt được cắt tỉa / hàm lượng giá; iterative deepening)
    mcts.py          Cấp 4  MCTS-UCT (số mô phỏng hoặc thời gian; rollout random / light)
  tournament.py      Round-robin, khai cuộc ghép cặp, đổi bên, seed, log JSONL, Wilson/bootstrap
  stats.py           Wilson CI, bootstrap, phân vị
  webapp.py          Server web (thư viện chuẩn) + ui/index.html: giao diện chơi
tests/               44 unit test (engine, agent, giải đấu, web)
run_ui.py            Chạy giao diện web (http://127.0.0.1:8765)
run_tournament.py    Giải đấu 2 chế độ ngân sách
run_ablation.py      Ablation A1-A6
play.py              Demo chơi trong terminal (người-máy / máy-máy, chọn cấp AI)
results/             Kết quả đã chạy (log thô, summary.md/json, ablation)
```

## Chạy

```bash
python -m unittest discover -s tests -t .          # kiểm thử tự động (~4 s)
python run_ui.py                                   # GIAO DIỆN WEB: chọn cấp AI cho từng bên
python play.py --ai 4                              # (terminal) chơi với MCTS, bạn là X đi trước
python play.py --ai 3 --depth 6 --second           # chơi với Alpha-Beta độ sâu 6, bạn đi sau
python play.py --ai 2 --ai2 4                      # máy-máy: Heuristic vs MCTS
python run_tournament.py --mode all                # 2 chế độ ngân sách, 200 ván/cặp
python run_ablation.py                             # A1-A6 (vài phút)
python run_ablation.py --only A1 A4
```

Mọi kết quả sinh lại được: seed cố định (`--seeds 0 1`), cấu hình ghi trong `config.json`.

## Giao diện web

`python run_ui.py` mở trang chơi tại http://127.0.0.1:8765 (thêm `--port`, `--no-browser` nếu cần; mặc định chỉ nhận kết nối
từ máy bạn). Không cần cài thêm thư viện.

* Chọn loại người chơi cho **mỗi bên** (Người / Random / Heuristic / Alpha-Beta / MCTS) -> người-máy, máy-máy, người-người;
  có sẵn 3 cấu hình nhanh. Alpha-Beta chỉnh được độ sâu, MCTS chỉnh được số mô phỏng.
* Nhấn cột hoặc phím 1-7 để thả quân (hiển thị cột 1-7; trong code và log là 0-6). Hoàn tác, ván mới, tạm dừng, tốc độ máy-máy.
* Hiển thị phân tích nước đi của máy: thời gian, số nút/mô phỏng, độ sâu, phân bố lượt thăm MCTS, và Alpha-Beta báo khi thấy
  chuỗi thắng/thua cưỡng bức.
* Giao diện chỉ là lớp hiển thị: nước đi của máy do chính các agent Python trong `c4ai/agents/` tính (server dựng lại ván từ
  danh sách nước đi, không giữ trạng thái).

## Mô hình hóa bài toán (Mục 3.1 của đề)

| Mục | Nội dung |
|---|---|
| Luật | Bàn 7 cột x 6 hàng; hai người thay phiên thả quân vào cột, quân rơi xuống ô trống thấp nhất; 4 quân liên tiếp (ngang/dọc/chéo) thắng; bàn đầy mà không ai thắng là hòa. Nước thứ 42 tạo 4 liên tiếp được tính là THẮNG. |
| Người chơi | 1 (X, đi trước) và 2 (O, đi sau) |
| Quan sát | đầy đủ (thông tin hoàn hảo) |
| Tính chất | xác định, tổng bằng không, luân phiên |
| Trạng thái | bàn cờ + người đến lượt; ~4,5 x 10^12 thế cờ hợp lệ (Tromp), cây trò chơi ~10^21 |
| Hành động | cột 0..6, hợp lệ khi cột chưa đầy; nước bất hợp lệ ném `ValueError` |
| Reward (góc nhìn người chơi p) | +1 thắng / 0 hòa / -1 thua, chỉ phát ở cuối ván (`game.reward(p)`) |
| Input/Output của agent | input: `Connect4` (trạng thái + lượt); output: một cột hợp lệ |

Kiểm thử engine (`tests/test_game.py`): thắng ngang/dọc/hai đường chéo, không "vòng" qua biên cột,
hòa, thắng ở nước 42, nước bất hợp lệ, `undo` khôi phục chính xác, `clone` độc lập, so khớp với bộ kiểm tra
ngây thơ trên 3000 ván ngẫu nhiên, và đếm số thế cờ khác nhau sau n nước khớp dãy đã biết
(1, 7, 49, 238, 1120, 4263).

## Các tác tử (Mục 3.2)

**Cấp 1 - Random.** Chọn đều trong các cột hợp lệ.

**Cấp 2 - Heuristic.** Thứ tự ưu tiên: (1) thắng ngay, (2) chặn đối thủ thắng ngay, (3) tránh nước làm mở ô thắng
cho đối thủ ngay phía trên, (4) chọn nước cho thế cờ có điểm tĩnh cao nhất. Xung đột luật: thắng ưu tiên hơn chặn;
nếu đối thủ có đe dọa kép thì chặn một ô (không cứu được); nếu mọi nước đều tự sát thì bỏ luật 3.
Không có trạng thái toàn cục (khác code cũ dùng nhiễu UCB làm kết quả phụ thuộc lịch sử).

**Cấp 3 - Alpha-Beta.** Negamax giới hạn độ sâu, hàm lượng giá cửa sổ 4 ô, điểm thắng giảm theo độ sâu
(`WIN - ply`, ưu tiên thắng nhanh), duyệt cột từ giữa ra biên, chọn nước đầu tiên khi đồng điểm (xác định).
Tùy chọn: `prune=False` (minimax thuần), `use_eval=False` (nút lá = 0), `time_limit` (iterative deepening).
Đếm số nút mỗi nước.

**Cấp 4 - MCTS-UCT.** Selection (UCB1, c = √2) - Expansion - Simulation (random rollout, hoặc `light`: thắng/chặn ngay) -
Backpropagation (W theo góc nhìn người vừa đi). Chọn nước có nhiều lượt thăm nhất. Ngân sách: số mô phỏng
hoặc thời gian.

## Thiết kế thí nghiệm (Mục 3.5)

* **Round-robin 4 agent, 200 ván/cặp** = 2 seed x 100 ván. Mỗi khai cuộc (4 nước ngẫu nhiên hợp lệ, chưa kết thúc) được chơi
  **hai lần** đổi bên đi trước -> cân bằng lượt đi. Tập khai cuộc dùng chung cho mọi cặp (ghép cặp).
  Cần khai cuộc ngẫu nhiên vì Alpha-Beta/Heuristic gần như xác định; nếu không, mọi ván giống hệt nhau.
* **Hai chế độ ngân sách**: `default` (Alpha-Beta d=5, MCTS 1000 mô phỏng) và `time` (50 ms/nước cho cả Alpha-Beta
  iterative deepening và MCTS). Không coi "cùng độ sâu" là công bằng tính toán.
* Chạy tuần tự 1 tiến trình để số đo thời gian không nhiễu. Số liệu: thắng/hòa/thua, score = (thắng + 0,5 x hòa)/ván,
  Wilson 95% và bootstrap 95%, score theo bên đi trước/sau và theo seed, thời gian TB/p95/max mỗi nước, số nút/mô phỏng TB.
* Log thô từng ván: `results/*/games.jsonl`.
* Ablation (A1-A6) trong `run_ablation.py`, kết quả ở `results/ablation/`.

## Hạn chế

* Giải đấu dùng khai cuộc ngẫu nhiên 4 nước: một số khai cuộc đã thắng cưỡng bức cho một bên (Random đôi khi thắng nhờ đó);
  việc đổi bên bù trừ nhưng điểm số giữa agent mạnh và Random vì thế không đúng 1,0 tuyệt đối.
* Alpha-Beta không có bảng chuyển vị/bộ giải hoàn chỉnh, MCTS chưa dùng giá trị học được: sức mạnh thực nghiệm
  không phải tính tối ưu đã chứng minh (Connect Four đã được giải: bên đi trước thắng nếu đi giữa).
* Số đo thời gian phụ thuộc phần cứng (xem `config.json`).

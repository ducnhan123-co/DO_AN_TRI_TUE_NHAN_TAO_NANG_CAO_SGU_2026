# Ghi chú đồ án Connect 4 (RL)

Ghi chú đọc code, bổ sung cho `README.md` (README gốc bằng tiếng Anh). Số dòng tham chiếu theo commit hiện tại.

## 1. Các file

| File | Vai trò |
|---|---|
| `connect4.py` | Môi trường Gymnasium `Connect4`: bàn 6×7 (`0` trống, `1`/`2` quân), action `Discrete(7)`, đối thủ nằm *bên trong* env, render bằng pygame |
| `heuristic.py` | `HeuristicAgent`: thắng ngay → chặn → chấm điểm (gần cột giữa, số quân liền kề) + nhiễu UCB (`c = √2`) |
| `minimax.py` | `MinimaxAgent`: minimax + alpha-beta, mặc định `depth=4` |
| `ensemble.py` | `EnsembleAgent`: bỏ phiếu đa số giữa nhiều agent |
| `train.py` | Hàm `train()` (PPO/A2C/DQN của Stable-Baselines3) + vòng curriculum PPO `agent0..agent15` |
| `test.py` | `test()` chơi 1 ván, trả về thắng/không; script đo winrate ensemble 15 agent PPO |
| `play.py` | Người chơi bằng chuột với pygame (mặc định đấu `HeuristicAgent`) |
| `ui/` + `run_ui.sh` | UI web tĩnh (HTML/CSS/JS thuần): chơi thử + xem giải thích từng đoạn code |

## 2. Luồng một bước `env.step(action)`

```
agent thả quân ─┬─ nước sai?  → ILLEGAL_MOVE_PENALTY, kết thúc ván
                ├─ thắng/hòa? → WIN_REWARD / DRAW_REWARD, kết thúc
                ├─ thử đặt quân đối thủ vào ô vừa đi → nếu đối thủ sẽ thắng ở đó ⇒ đánh dấu "đã chặn"
                └─ đối thủ đi (random / agent gắn qua set_opp_agent, có epsilon random)
                       ├─ đối thủ thắng/hòa → trả về -reward, kết thúc
                       └─ không → BLOCK_REWARD nếu đã chặn, ngược lại 0
```

`reset()` random 50% để đối thủ đi trước (agent khi đó là quân `2`).

## 3. Reward: README vs code

| | README | Code (`connect4.py`, `train.py`, `test.py`) |
|---|---|---|
| Thắng | +10 | +10 |
| Chặn | +3 | +3 |
| Nước sai | −50 | **−100** (`play.py` lại ghi −10, chỉ dùng để so sánh) |
| Hòa | 0 | **+1** |

Khi viết báo cáo nên lấy số trong code làm chuẩn.

## 4. Lỗi / điểm cần lưu ý (chưa sửa)

1. `train.py:32` — `algos[opp_algo].load("opp_path")` đang load **chuỗi** `"opp_path"` chứ không phải biến `opp_path` → train với đối thủ đã lưu sẽ lỗi không tìm thấy file.
2. `connect4.py` `step()` — hòa trả `DRAW_REWARD = 1 ≠ 0` nên `info["is_success"] = True`; số liệu success rate sẽ tính hòa là thắng.
3. Đối thủ hòa → env trả `-reward = -1`, tức cùng một kết cục hòa nhưng reward khác dấu tùy ai đi nước cuối.
4. `test.py`, `play.py` chạy code ở mức module (không có `if __name__ == "__main__"`) → `import test` sẽ chạy luôn 10.000 ván.
5. README nhắc `assets/play_against_heurestic_agent.gif` và `./saved_agents/` — **chưa có trong repo** (model chưa commit).

## 5. Cách chạy

```bash
# UI web (chỉ cần python3)
./run_ui.sh          # hoặc ./run_ui.sh 8788 nếu cổng 8777 bận

# Phần Python
pip install gymnasium stable-baselines3 pygame tqdm matplotlib
python play.py       # chơi với HeuristicAgent
python train.py      # train PPO + curriculum, lưu ./saved_agents, log ./logs
python test.py       # cần ./saved_agents/agent0..14 đã train
```

## 6. Trạng thái kiểm chứng

- Mục 1–4: đọc trực tiếp source trong thư mục này.
- **Chưa chạy** phần Python trên máy viết note (máy chưa cài `gymnasium`), nên chưa đo winrate hay chạy thử train.

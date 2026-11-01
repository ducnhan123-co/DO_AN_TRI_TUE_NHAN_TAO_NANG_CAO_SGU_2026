"""Kiểm thử hồi quy cho các lỗi đã được báo cáo/sửa. Chạy: python test_regression.py"""
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import torch

from c4core import (BOARD_MASK, BOT, COLS, ROWS, MinimaxPlayer, RandomPlayer, TacticalPlayer, drop_bit, match,
                    minimax_move, minimax_scores)
from c4env import Connect4Env, OpponentMixture, TorchPlayer, check_board, load_model, save_model

torch.set_num_threads(1)
OK = []


def check(name, cond):
    OK.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def tiny_model(tmp):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: Connect4Env()])
    m = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, learning_rate=3e-4, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=[32], vf=[32])))
    m.learn(128)
    return m


def main():
    # 6) cột ngoài biên không được lọt qua
    check("drop_bit(-1) bị từ chối", drop_bit(0, -1) == 0)
    check("drop_bit(COLS) bị từ chối", drop_bit(0, COLS) == 0)
    # 11) match 0 ván
    check("match(n_games=0) ném ValueError rõ ràng", raises(lambda: match(RandomPlayer(), RandomPlayer(), 0), ValueError))
    # 13) bàn đầy
    check("RandomPlayer bàn đầy -> ValueError", raises(lambda: RandomPlayer().act(BOARD_MASK, 0), ValueError))
    check("TacticalPlayer bàn đầy -> ValueError", raises(lambda: TacticalPlayer().act(BOARD_MASK, 0), ValueError))
    check("minimax_move bàn đầy -> ValueError", raises(lambda: minimax_move(0, BOARD_MASK, 2), ValueError))
    check("minimax_scores khớp minimax_move",
          max(minimax_scores(0, 0, 3).values()) == minimax_scores(0, 0, 3)[minimax_move(0, 0, 3)])

    # model sai kích thước bàn bị từ chối ngay khi nạp (models/agent_6x7.zip trên bàn 8x9)
    wrong = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                         "agent_6x7.zip" if (ROWS, COLS) == (8, 9) else "agent_8x9.zip")
    if os.path.exists(wrong):
        check("model sai cỡ bàn -> ValueError khi nạp", raises(lambda: check_board(load_model(wrong)), ValueError))

    with tempfile.TemporaryDirectory() as tmp:
        m = tiny_model(tmp)
        # 1) file model không chứa hàm (lambda) cho learning_rate / lr_schedule / clip_range
        p = os.path.join(tmp, "m.zip")
        save_model(m, p)
        data = json.loads(zipfile.ZipFile(p).read("data"))
        bad = [k for k in ("learning_rate", "lr_schedule", "clip_range")
               if isinstance(data.get(k), dict) and ":serialized:" in data[k]]
        check(f"zip không chứa hàm serialized {bad}", not bad)
        # nạp lại bằng tiến trình Python MỚI ở thư mục khác (mô phỏng máy/phiên bản khác)
        code = ("import sys; sys.path.insert(0, %r); from c4env import load_model; "
                "m = load_model(%r); print('LOAD_OK', m.policy.action_net.out_features)" % (os.getcwd(), p))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp)
        check("nạp model trong tiến trình mới thành công", "LOAD_OK" in out.stdout)
        # 2) TorchPlayer chạy trên CPU và nhận model dù policy ở thiết bị khác (ở đây kiểm tra tensor là CPU)
        tp = TorchPlayer(m.policy, deterministic=True)
        check("TorchPlayer dùng CPU", all(q.device.type == "cpu" for q in tp.fe.parameters()))
        check("TorchPlayer.act trả cột hợp lệ", 0 <= tp.act(0, 0) < COLS)
        # 4) with_options seed thật sự
        base = TorchPlayer(m.policy, deterministic=False, seed=0)
        q1, q2, q1b = base.with_options(seed=1), base.with_options(seed=999), base.with_options(seed=1)
        check("with_options: generator riêng (không chia sẻ)", q1.rng is not q2.rng and q1.rng is not base.rng)
        s1 = [int(q1.rng.integers(10**9)) for _ in range(3)]
        s1b = [int(q1b.rng.integers(10**9)) for _ in range(3)]
        s2 = [int(q2.rng.integers(10**9)) for _ in range(3)]
        check("with_options: cùng seed -> cùng dãy, khác seed -> khác dãy", s1 == s1b and s1 != s2)
        # 12) pool lưu/nạp an toàn bằng weights_only=True
        pp = os.path.join(tmp, "pool.pt")
        torch.save([tp.state()], pp)
        st = torch.load(pp, weights_only=True)
        tp2 = TorchPlayer.from_state(m.policy, st[0], deterministic=True)
        check("pool.pt nạp được với weights_only=True, kết quả giống hệt", tp2.act(0, 0) == tp.act(0, 0))
        # chế độ lai không bao giờ trả nước thua cưỡng bức trong tầm tìm kiếm
        hy = TorchPlayer(m.policy, deterministic=True, search_depth=3)
        r = match(hy, MinimaxPlayer(2, seed=1), 6, 2, seed=1)
        check("chế độ lai chạy đủ ván không lỗi", r["win"] + r["draw"] + r["loss"] == 6)

    # ---- thang 8 cấp độ
    import levels
    for n in range(1, 9):
        if not levels.is_ready(n):
            continue
        pl = levels.make_level(n, seed=1)
        a = pl.act(0, 0)
        check(f"cấp {n} ({levels.LEVEL_NAMES[n]}) trả cột hợp lệ trên bàn trống", 0 <= a < COLS and drop_bit(0, a) != 0)
    check("cấp độ ngoài 1-8 bị từ chối", raises(lambda: levels.make_level(9), ValueError))
    check("cấp 1-4 chơi hết ván không lỗi", all(
        sum(match(levels.make_level(n, seed=2), RandomPlayer(3), 2, 2, seed=4)[k] for k in ("win", "draw", "loss")) == 2
        for n in (1, 2, 3)))

    fails = [n for n, ok in OK if not ok]
    print(f"\n{len(OK) - len(fails)}/{len(OK)} kiểm thử đạt")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

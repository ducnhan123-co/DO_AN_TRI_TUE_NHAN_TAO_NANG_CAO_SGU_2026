"""
train_levels.py - Train các cấp độ cần học của thang 8 cấp (mỗi cấp mặc định 10.000 bước).

    python train_levels.py                      # train cấp 5,6,7(+8) với 10.000 bước mỗi cấp
    python train_levels.py --levels 5           # chỉ cấp 5
    python train_levels.py --steps 50000        # tăng ngân sách (mạnh hơn, lâu hơn)
    python train_levels.py --max-seconds 240    # giới hạn thời gian mỗi lần chạy; chạy lại lệnh để TIẾP TỤC

"Bước" được hiểu như sau (mỗi phương pháp một đơn vị riêng, nói rõ để so sánh công bằng):
  Cấp 1-4 : không cần train (luật / tìm kiếm thuần).
  Cấp 5   : 10.000 bước cập nhật gradient, học bắt chước minimax độ sâu 4 (dữ liệu sinh sẵn, tăng cường bằng lật gương).
  Cấp 6   : 10.000 bước môi trường của MaskablePPO (tự chơi với bể đối thủ).
  Cấp 7-8 : 10.000 bước cập nhật gradient của AlphaZero, xen kẽ tự chơi + MCTS; cấp 7 và 8 DÙNG CHUNG một mạng,
            khác nhau ở số mô phỏng MCTS lúc chơi (100 / 800).
Cấp 5/7/8 lưu dạng .npz (chỉ mảng số, không pickle); cấp 6 là file zip của stable-baselines3.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from c4core import COLS, ROWS, drop_bit, encode, has_won, is_full, minimax_scores
from levels import BASE_DIR, LEVEL_NAMES, level_dir, level_file
from mcts import AZSearch, NpPV


# --------------------------------------------------------------------------- mạng
class PVNet(nn.Module):
    """MLP policy (+ value). Đầu vào 2 x ROWS x COLS (quân của mình / của đối thủ)."""

    def __init__(self, hidden=256, value=True):
        super().__init__()
        self.l0 = nn.Linear(2 * ROWS * COLS, hidden)
        self.l1 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, COLS)
        self.v = nn.Linear(hidden, 1) if value else None

    def forward(self, x):
        h = F.relu(self.l1(F.relu(self.l0(x.flatten(1)))))
        return self.pi(h), (torch.tanh(self.v(h)).squeeze(-1) if self.v is not None else None)

    def arrays(self):
        t = lambda m: (m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())  # noqa: E731
        w0, b0 = t(self.l0)
        w1, b1 = t(self.l1)
        wp, bp = t(self.pi)
        d = dict(rows=ROWS, cols=COLS, w0=w0, b0=b0, w1=w1, b1=b1, wp=wp, bp=bp)
        if self.v is not None:
            d["wv"], d["bv"] = t(self.v)
        return d

    def export(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **self.arrays())


def log_row(name, row):
    path = os.path.join(level_dir(), name)
    os.makedirs(level_dir(), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(list(row.keys()))
        w.writerow(list(row.values()))


# --------------------------------------------------------------------------- CẤP 5: học có giám sát
def gen_sl_data(path, n_pos, depth, seed, deadline):
    """Sinh dữ liệu: cho minimax (giáo viên) tự đấu với nhiễu; mỗi thế cờ gán nhãn = phân phối đều trên các nước
    tốt nhất theo minimax độ sâu `depth`. Chạy nối tiếp được (ghi thêm vào file)."""
    X, Y = [], []
    if os.path.exists(path):
        with np.load(path) as z:
            X, Y = list(z["X"]), list(z["Y"])
    rng = np.random.default_rng(seed + len(X))
    t0, start_n = time.time(), len(X)
    while len(X) < n_pos and time.time() < deadline:
        me = opp = 0
        ply = 0
        while True:
            if is_full(me | opp):
                break
            sc = minimax_scores(me, opp, depth)
            best = max(sc.values())
            ties = [c for c, v in sc.items() if v == best]
            y = np.zeros(COLS, dtype=np.float32)
            y[ties] = 1.0 / len(ties)
            X.append(encode(me, opp).astype(np.uint8))
            Y.append(y)
            if ply < 4 or rng.random() < 0.15:  # mở đầu ngẫu nhiên + nhiễu để thế cờ đa dạng
                legal = [c for c in sc]
                a = int(legal[rng.integers(len(legal))])
            else:
                a = int(ties[rng.integers(len(ties))])
            mv = drop_bit(me | opp, a)
            me |= mv
            if has_won(me):
                break
            me, opp = opp, me
            ply += 1
    np.savez(path, X=np.array(X, dtype=np.uint8), Y=np.array(Y, dtype=np.float32))
    print(f"  dữ liệu giám sát: {len(X)}/{n_pos} thế cờ (+{len(X) - start_n} trong {time.time() - t0:.0f}s)", flush=True)
    return len(X) >= n_pos


def train_sl(steps, data_path, out_path, batch=128, lr=2e-3, seed=0):
    with np.load(data_path) as z:
        X = torch.from_numpy(z["X"].astype(np.float32))
        Y = torch.from_numpy(z["Y"])
    n = len(X)
    torch.manual_seed(seed)
    net = PVNet(value=False)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))) * 0.95 + 0.05)
    g = torch.Generator().manual_seed(seed)
    hold = max(1, n // 10)  # 10% giữ lại để đo độ chính xác (chỉ để theo dõi)
    tr_idx = torch.arange(hold, n)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = tr_idx[torch.randint(len(tr_idx), (batch,), generator=g)]
        x, y = X[idx], Y[idx]
        flip = torch.rand(batch, generator=g) < 0.5  # lật gương
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        y = torch.where(flip[:, None], y.flip(-1), y)
        logits, _ = net(x)
        logits = logits.masked_fill(x[:, :, 0, :].sum(1) > 0, -1e9)  # cột đầy (hàng trên cùng có quân)
        loss = -(y * F.log_softmax(logits, dim=1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps:
            with torch.no_grad():
                xh, yh = X[:hold], Y[:hold]
                lh, _ = net(xh)
                lh = lh.masked_fill(xh[:, :, 0, :].sum(1) > 0, -1e9)
                acc = (yh.gather(1, lh.argmax(1, keepdim=True)) > 0).float().mean().item()
            print(f"  [cấp 5] bước {step:6d}/{steps} | loss {loss.item():.3f} | độ chính xác (giữ lại) {acc:.2%}", flush=True)
            log_row("train_log_level5.csv", dict(step=step, loss=round(loss.item(), 4), holdout_acc=round(acc, 4),
                                                 seconds=round(time.time() - t0)))
    net.export(out_path)


def run_level5(a, deadline):
    os.makedirs(level_dir(), exist_ok=True)
    data = os.path.join(level_dir(), "sl_data.npz")
    print(f"[cấp 5] {LEVEL_NAMES[5]}: dữ liệu {a.sl_positions} thế cờ (giáo viên minimax d{a.teacher_depth}), "
          f"huấn luyện {a.steps} bước", flush=True)
    if not gen_sl_data(data, a.sl_positions, a.teacher_depth, a.seed, deadline):
        print("  -> chưa đủ dữ liệu, chạy lại lệnh để tiếp tục.")
        return False
    train_sl(a.steps, data, level_file(5), seed=a.seed)
    print(f"  -> đã lưu {os.path.relpath(level_file(5), BASE_DIR)}", flush=True)
    return True


# --------------------------------------------------------------------------- CẤP 6: học tăng cường
def run_level6(a, deadline):
    out = os.path.join(level_dir(), "rl_run")
    init = [] if (a.rl_scratch or not os.path.exists(level_file(5))) else ["--init-sl", level_file(5), "--lr", "1e-4"]
    print(f"[cấp 6] {LEVEL_NAMES[6]}: MaskablePPO {a.steps} bước môi trường, "
          + ("khởi tạo từ mạng cấp 5 (giám sát -> tăng cường)" if init else "khởi tạo ngẫu nhiên"), flush=True)
    env = dict(os.environ, C4_ROWS=str(ROWS), C4_COLS=str(COLS))
    n_envs = 10
    per = max(1, a.steps // n_envs)  # n_envs * n_steps = a.steps / số lần cập nhật
    n_steps = max(8, per // 10)
    cmd = [sys.executable, os.path.join(BASE_DIR, "train_v2.py"), "--arch", "mlp", "--steps", str(a.steps), "--out", out,
           "--threads", "1", "--n-envs", str(n_envs), "--n-steps", str(n_steps),
           "--batch-size", str(n_envs * n_steps // 2), "--eval-every", str(a.steps), "--seed", str(a.seed)] + init
    subprocess.run(cmd, check=True, env=env, cwd=BASE_DIR)
    shutil.copy(os.path.join(out, "final.zip"), level_file(6))
    print(f"  -> đã lưu {os.path.relpath(level_file(6), BASE_DIR)}", flush=True)
    return True


# --------------------------------------------------------------------------- CẤP 7-8: AlphaZero
def az_self_play(search, rng, temp_moves=12):
    """Một ván tự chơi. Trả về danh sách (thế cờ, phân phối lượt thăm, z theo góc nhìn người đi)."""
    me = opp = 0
    recs, ply, winner = [], 0, None
    while True:
        N, root = search.run(me, opp, noise=True)
        Nn = np.where(root.allowed, N, 0.0)
        pi = (Nn / Nn.sum()) if Nn.sum() > 0 else root.allowed / root.allowed.sum()
        recs.append((encode(me, opp).astype(np.uint8), pi.astype(np.float32), ply % 2))
        a = int(rng.choice(COLS, p=pi)) if ply < temp_moves else int(pi.argmax())
        mv = drop_bit(me | opp, a)
        me |= mv
        if has_won(me):
            winner = ply % 2
            break
        if is_full(me | opp):
            break
        me, opp = opp, me
        ply += 1
    return [(x, p, 0.0 if winner is None else (1.0 if par == winner else -1.0)) for x, p, par in recs]


def run_az(a, deadline):
    d = level_dir()
    os.makedirs(d, exist_ok=True)
    state_p, buf_p = os.path.join(d, "az_state.pt"), os.path.join(d, "az_buffer.npz")
    torch.manual_seed(a.seed)
    net = PVNet(value=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.az_lr, weight_decay=1e-4)
    step = games = 0
    if os.path.exists(state_p):
        st = torch.load(state_p, weights_only=True)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        step, games = int(st["step"]), int(st["games"])
    BX, BP, BZ = [], [], []
    if os.path.exists(buf_p):
        with np.load(buf_p) as z:
            BX, BP, BZ = list(z["X"]), list(z["P"]), list(z["Z"])
    print(f"[cấp 7-8] {LEVEL_NAMES[7]} / {LEVEL_NAMES[8]}: AlphaZero {a.steps} bước gradient "
          f"(đã có {step} bước, {games} ván tự chơi, buffer {len(BX)})", flush=True)
    rng = np.random.default_rng(a.seed + games)
    t0 = time.time()
    while step < a.steps and time.time() < deadline:
        np_net = NpPV(net.arrays())
        search = AZSearch(np_net, sims=a.az_train_sims, rng=rng)
        lens = []
        for _ in range(a.az_games_per_iter):
            recs = az_self_play(search, rng)
            lens.append(len(recs))
            for x, p, z in recs:
                for xx, pp in ((x, p), (x[..., ::-1].copy(), p[::-1].copy())):  # + bản lật gương
                    BX.append(xx), BP.append(pp), BZ.append(z)
            games += 1
        if len(BX) > a.az_buffer:
            cut = len(BX) - a.az_buffer
            BX, BP, BZ = BX[cut:], BP[cut:], BZ[cut:]
        if len(BX) < 1000:
            continue
        Xa = torch.from_numpy(np.array(BX, dtype=np.float32))
        Pa = torch.from_numpy(np.array(BP, dtype=np.float32))
        Za = torch.from_numpy(np.array(BZ, dtype=np.float32))
        net.train()
        for _ in range(min(a.az_steps_per_iter, a.steps - step)):
            idx = torch.randint(len(Xa), (a.az_batch,))
            logits, v = net(Xa[idx])
            full = Xa[idx][:, :, 0, :].sum(1) > 0
            logp = F.log_softmax(logits.masked_fill(full, -1e9), dim=1)
            l_pi = -(Pa[idx] * logp).sum(1).mean()
            l_v = F.mse_loss(v, Za[idx])
            loss = l_pi + l_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        net.eval()
        if (step // a.az_steps_per_iter) % 5 == 0 or step >= a.steps:
            print(f"  [AZ] bước {step:6d}/{a.steps} | ván {games:5d} | buffer {len(BX):6d} | độ dài TB {np.mean(lens):.1f} | "
                  f"loss_pi {l_pi.item():.3f} loss_v {l_v.item():.3f} | {time.time() - t0:.0f}s", flush=True)
            log_row("train_log_az.csv", dict(step=step, games=games, buffer=len(BX), loss_pi=round(l_pi.item(), 4),
                                             loss_v=round(l_v.item(), 4)))
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, games=games), state_p)
    np.savez(buf_p, X=np.array(BX, dtype=np.uint8), P=np.array(BP, dtype=np.float32), Z=np.array(BZ, dtype=np.float32))
    if step >= a.steps:
        net.export(level_file(7))
        print(f"  -> đã lưu {os.path.relpath(level_file(7), BASE_DIR)} (dùng cho cấp 7 và 8)", flush=True)
        return True
    print(f"  -> chưa xong ({step}/{a.steps} bước), chạy lại lệnh để tiếp tục.", flush=True)
    return False


# --------------------------------------------------------------------------- main
def parse_levels(s):
    out = set()
    for part in s.split(","):
        part = part.strip()
        if part == "all":
            out |= set(range(1, 9))
        elif "-" in part:
            lo, hi = part.split("-")
            out |= set(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    bad = [x for x in out if not 1 <= x <= 8]
    if bad:
        raise SystemExit(f"cấp độ không hợp lệ: {bad} (chỉ 1-8)")
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description="Train thang 8 cấp độ Connect 4")
    ap.add_argument("--levels", default="all", help="vd: all | 5,6 | 5-8")
    ap.add_argument("--steps", type=int, default=10_000, help="số bước train cho MỖI cấp cần học")
    ap.add_argument("--max-seconds", type=float, default=0, help="giới hạn thời gian mỗi lần chạy (0 = không giới hạn)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sl-positions", type=int, default=60_000, help="số thế cờ dữ liệu giám sát")
    ap.add_argument("--rl-scratch", action="store_true", help="cấp 6: khởi tạo ngẫu nhiên thay vì từ mạng cấp 5")
    ap.add_argument("--teacher-depth", type=int, default=4, help="độ sâu minimax của giáo viên (cấp 5)")
    ap.add_argument("--az-train-sims", type=int, default=64, help="số mô phỏng MCTS khi tự chơi (AlphaZero)")
    ap.add_argument("--az-games-per-iter", type=int, default=20)
    ap.add_argument("--az-steps-per-iter", type=int, default=100)
    ap.add_argument("--az-batch", type=int, default=128)
    ap.add_argument("--az-buffer", type=int, default=60_000)
    ap.add_argument("--az-lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_num_threads(1)
    deadline = time.time() + a.max_seconds if a.max_seconds else float("inf")
    lv = parse_levels(a.levels)
    print(f"Bàn {ROWS}x{COLS} | cấp: {lv} | {a.steps} bước mỗi cấp cần học\n")
    for n in lv:
        if n <= 4:
            print(f"[cấp {n}] {LEVEL_NAMES[n]}: không cần train.")
    done = True
    if 5 in lv:
        done &= run_level5(a, deadline)
    if 6 in lv:
        done &= run_level6(a, deadline)
    if 7 in lv or 8 in lv:
        done &= run_az(a, deadline)
    print("\nHOÀN TẤT." if done else "\nCHƯA XONG: chạy lại đúng lệnh này để tiếp tục.")


if __name__ == "__main__":
    main()

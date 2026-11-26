"""
expio.py - Ghi kết quả thí nghiệm theo cấu trúc thư mục chuẩn + NHẬT KÝ THÍ NGHIỆM (Excel / Google Sheets).

Mỗi lần chạy (tournament / ablation / train) tạo MỘT thư mục:  results/runs/<mã-thí-nghiệm>/

    app_cfg.json          cấu hình lần chạy: lệnh, mục tiêu, giả thuyết, bàn, seed, số ván, ngân sách, phần cứng, mã băm của code
    params_cfg.json       tham số THỰC của từng tác tử (độ sâu, số mô phỏng, shield, ...; mã băm sha256 của file model)
    run.log.txt           toàn bộ log console (train.log.txt với lần train)
    games_seed<k>.csv     log từng ván của seed k   (tương ứng train_logs_fold<k>.csv)
    games_all.csv         gộp mọi seed              (tương ứng train_logs_all.csv)
    metrics_seed.csv      số đo của từng tác tử theo từng seed   (tương ứng metrics_best.csv)
    metrics_seed_mean.json trung bình ± độ lệch chuẩn qua các seed + Wilson 95% gộp   (tương ứng metrics_best_mean.json)
    metrics_pairs.csv     kết quả từng cặp, gộp mọi seed        (tương ứng metrics_avg.csv)
    metrics_overall.json  xếp hạng tổng + chi phí tính toán     (tương ứng metrics_avg_mean.json)
    summary.md            báo cáo đọc nhanh
    models/               bản sao model đã dùng (nếu có, để tái lập)
    data/                 dữ liệu thô: games.jsonl (nước đi, ms, nút, mô phỏng), pairs/ (bộ nhớ đệm để chạy tiếp)

Nhật ký: results/NHAT_KY_THI_NGHIEM.xlsx (mỗi lần chạy thêm 1 dòng + các bảng chi tiết). Mở bằng Excel, hoặc tải lên
Google Drive -> mở bằng Google Trang tính (Google Sheets). Các cột "Kết luận", "Hành động tiếp theo" do NGƯỜI điền.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time

from c4.expstats import mean, percentile, wilson_interval

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục gốc dự án
RUNS_DIR = os.path.join(BASE, "results", "runs")
JOURNAL = os.environ.get("C4_JOURNAL") or os.path.join(BASE, "results", "NHAT_KY_THI_NGHIEM.xlsx")
CODE_FILES = ("c4/core.py", "c4/mcts.py", "c4/levels.py", "c4/env.py", "c4/arena.py", "c4/expstats.py", "c4/expio.py",
              "c4/elo.py", "c4/agent_eval.py", "c4/aggregate.py", "tournament.py", "run_ablation.py", "train_levels.py",
              "tools/train.py", "tools/evaluate.py", "tools/ablation_agent.py")


# ----------------------------------------------------------------------------- tiện ích
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def new_run_id(kind, board, tag=""):
    return "_".join(x for x in (time.strftime("%Y%m%d-%H%M%S"), kind, board, tag) if x)


def make_run_dir(run_id, root=None):
    d = os.path.join(root or RUNS_DIR, run_id)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    return d


def sha256_file(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def code_hash():
    """Mã băm của mã nguồn (không có git): đổi mã => đổi hash => biết kết quả sinh ra từ phiên bản nào."""
    h = hashlib.sha256()
    for fn in CODE_FILES:
        p = os.path.join(BASE, fn)
        if os.path.exists(p):
            h.update(fn.encode())
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:12]


def environment():
    cpu = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": cpu or platform.processor(),
           "cpu_count": os.cpu_count()}
    for mod in ("numpy", "torch", "stable_baselines3", "pygame"):
        try:
            env[mod] = __import__(mod).__version__ if mod != "pygame" else __import__("pygame").version.ver
        except Exception:  # noqa: BLE001
            pass
    return env


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel đọc đúng tiếng Việt
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class Tee:
    """Chép mọi thứ in ra màn hình vào run.log.txt (dùng: with Tee(path): ...)."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()

    def __enter__(self):
        self.out, sys.stdout = sys.stdout, self
        return self

    def __exit__(self, *a):
        sys.stdout = self.out
        self.f.close()


def agent_params(player, level, opts):
    """Tham số thực của tác tử (chỉ các thuộc tính đơn giản) + mã băm model nếu là cấp học."""
    d = {"level": level, "class": type(player).__name__, "name": getattr(player, "name", ""), "opts": opts}
    for k, v in vars(player).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            d[k] = v
    if level >= 5:
        try:
            from c4 import levels
            p = levels.level_file(level)
            if os.path.exists(p):
                d["model_file"] = os.path.relpath(p, BASE)
                d["model_sha256"] = sha256_file(p)
        except Exception:  # noqa: BLE001
            pass
    return d


def copy_models(run_dir, params, limit_mb=60):
    out = os.path.join(run_dir, "models")
    for lab, pr in params.items():
        mf = pr.get("model_file")
        if mf and os.path.exists(os.path.join(BASE, mf)) and os.path.getsize(os.path.join(BASE, mf)) < limit_mb * 1e6:
            os.makedirs(out, exist_ok=True)
            shutil.copy2(os.path.join(BASE, mf), os.path.join(out, f"{lab}_{os.path.basename(mf)}"))


# ----------------------------------------------------------------------------- số đo từ log ván
def read_games(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _subjects(rec, both):
    ms = lambda pl: [x[2] * 1000 for x in rec["log"] if x[0] == pl]  # noqa: E731
    nd = lambda pl: [x[3] for x in rec["log"] if x[0] == pl]  # noqa: E731
    sm = lambda pl: [x[4] for x in rec["log"] if x[0] == pl]  # noqa: E731
    out = [(rec["a"], rec["a_outcome"], ms(rec["a_player"]), nd(rec["a_player"]), sm(rec["a_player"]), rec["a_player"] == 1)]
    if both:
        bp = 3 - rec["a_player"]
        out.append((rec["b"], 1 - rec["a_outcome"], ms(bp), nd(bp), sm(bp), bp == 1))
    return out


def per_seed_metrics(records, both=True):
    """Hàng (seed, tác tử): số ván, W/D/L, score, ms/nước TB-p95, nút, mô phỏng."""
    agg = {}
    for r in records:
        for subj, outc, ms, nd, sm, first in _subjects(r, both):
            e = agg.setdefault((r["seed"], subj), {"o": [], "ms": [], "nd": [], "sm": [], "first": []})
            e["o"].append(outc)
            e["ms"] += ms
            e["nd"] += nd
            e["sm"] += sm
            e["first"].append((first, outc))
    rows = []
    for (seed, subj), e in sorted(agg.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        n = len(e["o"])
        w, d = sum(o == 1.0 for o in e["o"]), sum(o == 0.5 for o in e["o"])
        lo, hi = wilson_interval(w + 0.5 * d, n)
        rows.append({"seed": seed, "agent": subj, "games": n, "win": w, "draw": d, "loss": n - w - d,
                     "score": round((w + 0.5 * d) / n, 4), "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
                     "score_first": round(mean(o for f, o in e["first"] if f), 4) if any(f for f, _ in e["first"]) else "",
                     "score_second": round(mean(o for f, o in e["first"] if not f), 4) if any(not f for f, _ in e["first"]) else "",
                     "mean_ms": round(mean(e["ms"]), 3), "p95_ms": round(percentile(e["ms"], 95), 3) if e["ms"] else 0,
                     "mean_nodes": round(mean(e["nd"]), 1), "mean_sims": round(mean(e["sm"]), 1)})
    return rows


def seed_mean(rows):
    """Trung bình ± độ lệch chuẩn (mẫu) của score qua các seed, kèm Wilson 95% gộp mọi ván."""
    out = {}
    for agent in sorted({r["agent"] for r in rows}, key=str):
        rs = [r for r in rows if r["agent"] == agent]
        sc = [r["score"] for r in rs]
        n = sum(r["games"] for r in rs)
        pts = sum(r["win"] + 0.5 * r["draw"] for r in rs)
        lo, hi = wilson_interval(pts, n)
        out[str(agent)] = {"n_seeds": len(rs), "games": n, "score_mean": round(mean(sc), 4),
                           "score_std": round(statistics.stdev(sc), 4) if len(sc) > 1 else None,
                           "score_pooled": round(pts / n, 4), "wilson_pooled": [round(lo, 4), round(hi, 4)],
                           "mean_ms": round(mean(r["mean_ms"] for r in rs), 3), "p95_ms": max(r["p95_ms"] for r in rs)}
    return out


GAME_FIELDS = ["seed", "game", "a", "b", "a_player", "opening", "winner", "a_outcome", "plies", "a_mean_ms", "b_mean_ms", "moves"]


def game_rows(records):
    rows = []
    for r in records:
        pa, pb = r["a_player"], 3 - r["a_player"]
        ms = lambda pl: [x[2] * 1000 for x in r["log"] if x[0] == pl]  # noqa: E731
        rows.append({"seed": r["seed"], "game": r["game"], "a": r["a"], "b": r["b"], "a_player": pa,
                     "opening": " ".join(map(str, r["opening"])),
                     "winner": "draw" if r["winner"] == 0 else ("A" if r["winner"] == pa else "B"),
                     "a_outcome": r["a_outcome"], "plies": r["plies"],
                     "a_mean_ms": round(mean(ms(pa)), 3), "b_mean_ms": round(mean(ms(pb)), 3),
                     "moves": " ".join(map(str, r["moves"]))})
    return rows


def write_game_logs(run_dir, records):
    rows = game_rows(records)
    for s in sorted({r["seed"] for r in rows}):
        write_csv(os.path.join(run_dir, f"games_seed{s}.csv"), [r for r in rows if r["seed"] == s], GAME_FIELDS)
    write_csv(os.path.join(run_dir, "games_all.csv"), rows, GAME_FIELDS)


# ----------------------------------------------------------------------------- hoàn tất một lần chạy
def finalize(run_dir, run_id, kind, app_cfg, params_cfg, records, both, pair_rows, overall_extra, summary_md=None):
    """Ghi toàn bộ file chuẩn của thư mục lần chạy. Trả về (metrics_seed rows, overall dict)."""
    write_json(os.path.join(run_dir, "app_cfg.json"), app_cfg)
    write_json(os.path.join(run_dir, "params_cfg.json"), params_cfg)
    write_game_logs(run_dir, records)
    ms = per_seed_metrics(records, both)
    write_csv(os.path.join(run_dir, "metrics_seed.csv"), ms)
    sm = seed_mean(ms)
    write_json(os.path.join(run_dir, "metrics_seed_mean.json"), sm)
    write_csv(os.path.join(run_dir, "metrics_pairs.csv"), pair_rows)
    overall = {"run_id": run_id, "kind": kind, "per_agent": sm, **overall_extra}
    write_json(os.path.join(run_dir, "metrics_overall.json"), overall)
    if summary_md:
        with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(summary_md)
    copy_models(run_dir, params_cfg)
    return ms, overall


# ----------------------------------------------------------------------------- NHẬT KÝ (xlsx)
SHEETS = {
    "Nhật ký": ["Mã TN", "Bắt đầu", "Thời lượng (phút)", "Loại", "Người thực hiện", "Mục tiêu", "Giả thuyết", "Bàn cờ",
                "Đối tượng", "Ngân sách", "Seeds", "Ván/cặp", "Khai cuộc (nước)", "Lệnh chạy", "Mã băm code",
                "Phần cứng", "Thư mục kết quả", "Kết quả chính (tự động)", "Kết luận (điền tay)",
                "Bất thường / ghi chú", "Trạng thái", "Hành động tiếp theo (điền tay)"],
    "Kết quả cặp": ["Mã TN", "Chế độ", "A", "B", "Số ván", "Thắng", "Hòa", "Thua", "Score A", "Wilson thấp", "Wilson cao",
                    "Bootstrap thấp", "Bootstrap cao", "Score khi đi trước", "Score khi đi sau"],
    "Số đo theo tác tử": ["Mã TN", "Tác tử", "Số seed", "Số ván", "Score TB qua seed", "Độ lệch chuẩn", "Score gộp",
                          "Wilson thấp", "Wilson cao", "ms/nước TB", "p95 ms", "Nút TB", "Mô phỏng TB"],
    "Ablation": ["Mã TN", "Mã ablation", "Mô tả", "Biến thể", "Thắng", "Hòa", "Thua", "Score", "Wilson thấp", "Wilson cao",
                 "ms/nước TB", "p95 ms"],
    "Huấn luyện": ["Mã TN", "Cấp", "Seed", "Bàn cờ", "Số bước", "Thời gian (phút)", "VAL acc", "TEST acc", "Hoàn tất",
                   "File model", "sha256 model"],
    "Từ điển": ["Mục", "Giải thích"],
}
DICT = [
    ("Mã TN", "<ngày-giờ>_<loại>_<bàn>_<nhãn>; trùng tên thư mục trong results/runs/. Mỗi lần chạy = 1 dòng ở sheet 'Nhật ký'."),
    ("Mục tiêu / Giả thuyết", "Điền bằng tham số --goal / --hypothesis TRƯỚC khi chạy (ghi giả thuyết trước khi thấy kết quả)."),
    ("Kết quả chính (tự động)", "Do chương trình ghi từ số liệu thật; không sửa tay. Diễn giải đặt ở 'Kết luận'."),
    ("Kết luận (điền tay)", "Nhận xét của người làm: giả thuyết được ủng hộ hay bác bỏ, vì sao, độ tin cậy (xem khoảng Wilson)."),
    ("Score", "(thắng + 0,5 × hòa) / số ván, từ góc nhìn tác tử ở cột trước."),
    ("Wilson / Bootstrap", "Khoảng tin cậy 95% của score; chồng lấn nhau => chưa đủ bằng chứng kết luận khác biệt."),
    ("Ngân sách", "default = cấu hình mặc định mỗi cấp; time = cùng X ms/nước cho cấp 3, 4, 7, 8."),
    ("Mã băm code", "sha256 rút gọn của mã nguồn lúc chạy: cùng mã băm + cùng seed => kết quả tái lập được."),
    ("Trạng thái", "Hoàn tất | Dang dở (có thể chạy tiếp) | Lỗi | Bỏ (kết quả không dùng)."),
    ("Thư mục kết quả", "Chứa app_cfg.json, params_cfg.json, run.log.txt, games_*.csv, metrics_*.csv/json, data/ (log thô)."),
    ("Dùng với Google Sheets", "Tải file .xlsx lên Google Drive -> mở bằng Google Trang tính; hoặc Trang tính > Tệp > Nhập."),
]


def _open_journal(path):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    if os.path.exists(path):
        wb = load_workbook(path)
    else:
        wb = Workbook()
        wb.remove(wb.active)
    for name, cols in SHEETS.items():
        if name in wb.sheetnames:
            continue
        ws = wb.create_sheet(name)
        ws.append(cols)
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="305496")
            c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.freeze_panes = "B2"
        for i, col in enumerate(cols, 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = 40 if col == "Mã TN" else 20 if col == "Bắt đầu" else 46 if col in (
                "Mục tiêu", "Giả thuyết", "Lệnh chạy", "Kết quả chính (tự động)", "Kết luận (điền tay)", "Bất thường / ghi chú",
                "Hành động tiếp theo (điền tay)", "Giải thích", "Thư mục kết quả", "Phần cứng") else 16
        if name == "Nhật ký":
            dv = DataValidation(type="list", formula1='"Hoàn tất,Dang dở,Lỗi,Bỏ"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add("U2:U2000")
        if name == "Từ điển":
            ws.column_dimensions["A"].width = 28
            ws.column_dimensions["B"].width = 110
            for a, b in DICT:
                ws.append([a, b])
    order = list(SHEETS)
    wb._sheets = [wb[n] for n in order if n in wb.sheetnames] + [s for s in wb._sheets if s.title not in order]
    return wb


def _replace_rows(ws, run_id, rows):
    """Xóa các dòng cũ của run_id (khi chạy lại/tiếp tục) rồi thêm dòng mới; KHÔNG đụng dòng của lần chạy khác."""
    keep_manual = {}
    for r in range(ws.max_row, 1, -1):
        if ws.cell(r, 1).value == run_id:
            if ws.title == "Nhật ký":  # giữ phần người đã điền tay
                keep_manual = {"Kết luận (điền tay)": ws.cell(r, 19).value, "Hành động tiếp theo (điền tay)": ws.cell(r, 22).value}
            ws.delete_rows(r)
    for row in rows:
        ws.append(row)
    return keep_manual


def journal_write(row_log, pair_rows=(), agent_rows=(), ablation_rows=(), train_rows=(), path=None):
    """Thêm/ghi đè các dòng của một lần chạy vào nhật ký. row_log: dict theo SHEETS['Nhật ký']."""
    path = path or JOURNAL
    wb = _open_journal(path)
    run_id = row_log["Mã TN"]
    cols = SHEETS["Nhật ký"]
    ws = wb["Nhật ký"]
    manual = {}
    for r in range(ws.max_row, 1, -1):
        if ws.cell(r, 1).value == run_id:
            manual = {19: ws.cell(r, 19).value, 22: ws.cell(r, 22).value}
            ws.delete_rows(r)
    vals = [row_log.get(c, "") for c in cols]
    for idx, v in manual.items():
        if v:
            vals[idx - 1] = v
    ws.append(vals)
    from openpyxl.styles import Alignment
    for c in ws[ws.max_row]:
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for sheet, rows in (("Kết quả cặp", pair_rows), ("Số đo theo tác tử", agent_rows), ("Ablation", ablation_rows),
                        ("Huấn luyện", train_rows)):
        if rows:
            _replace_rows(wb[sheet], run_id, [list(r) for r in rows])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    wb.save(tmp)
    os.replace(tmp, path)  # ghi nguyên tử: lỡ tắt giữa chừng không hỏng nhật ký cũ
    return path


def agent_rows_for_journal(run_id, overall):
    rows = []
    for agent, m in overall["per_agent"].items():
        c = overall.get("cost", {}).get(agent, {})
        rows.append([run_id, agent, m["n_seeds"], m["games"], m["score_mean"], m["score_std"], m["score_pooled"],
                     m["wilson_pooled"][0], m["wilson_pooled"][1], m["mean_ms"], m["p95_ms"],
                     c.get("mean_nodes", ""), c.get("mean_sims", "")])
    return rows

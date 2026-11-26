# Bổ sung: công cụ phân tích khớp với báo cáo

Thêm mới
- `c4/elo.py` - Elo Bradley–Terry (hòa = nửa thắng, 1 ván hòa ảo/cặp, Random = 0), kèm bootstrap; `tournament.py` ghi cột Elo vào `summary.md` và `metrics_overall.json`.
- `c4/agent_eval.py`, `tools/ablation_agent.py` - ablation tác tử PPO chính (đầy đủ / bỏ gương / bỏ mask / cả hai / tìm kiếm lai) x Random, Tactical, Minimax d2, d4 (, d6).
- `tools/evaluate.py` - thêm `--no-mask`, `--no-flip`, `--opponents`, `--d6-games`, `--opening-plies`, `--seed`; in Wilson và điểm đi trước / đi sau.
- `c4/aggregate.py`, `tools/aggregate_runs.py` - lợi thế đi trước / đi sau gộp nhiều lần chạy (+ Elo gộp).
- `tools/plot_results.py` - vẽ hình: loss cấp 5, loss AlphaZero, đường eval PPO, ma trận điểm, Elo, độ trễ, ngân sách tìm kiếm, ablation PPO.
- `tests/test_analysis.py` - 23 kiểm thử (không cần torch).

Sửa
- `tests/test_regression.py`: `from c4env import ...` -> `from c4.env import ...` (tên module cũ).
- `c4/expio.py`: `CODE_FILES` trỏ tới `tools/train.py` (không còn `train_v2.py`) và gồm các file mới.
- README / docs / docstring: bỏ tên cũ `train_v2.py`, `evaluate_v2.py`, `c4env`, `c4core`.

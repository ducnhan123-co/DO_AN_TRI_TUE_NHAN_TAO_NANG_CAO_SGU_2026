# Kết quả thí nghiệm (Connect4)

Cột 'thắng/hoà/thua' tính từ góc nhìn của agent. Số ván nhỏ (20-100) nên có sai số vài chục phần trăm.

## 1. Agent cũ (PPO agent0-15, bàn 6x7)
- Thắng random TB 82% | tactical 5.6% | heuristic 0.3% | minimax d2 0% | minimax d4 0%.
- 3-10% ván thua vì đi nước phạm quy (cột đầy).
- Agent i thắng agent i-1 ~90% nhưng gặp các thế hệ khác chỉ ~50%; Spearman(thế hệ, điểm) = -0.07 -> không tiến bộ.
- Chi tiết: old_agents_vs_baselines.csv, old_agents_cross_play.csv

## 2. Pipeline mới (MaskablePPO + mask 1-ply + bể đối thủ)
- Bàn 6x7: thắng agent15 cũ 100/100, ensemble cũ 98/100.
- Chi tiết: final_models_vs_baselines.csv
- Không dùng mask 1-ply: học chậm (sau 350k bước chỉ thắng tactical ~20%, minimax2 ~0%); với mask: 70% / ~25% sau 400k bước.
- Bỏ mask khi chơi thì agent v2 thua gần hết (tactical 13/100).

## 3. Bàn 8x9
- Train từ đầu ~1,75M bước (MLP): xem final_models_vs_baselines.csv.
- Chuyển giao từ 6x7 (~1,5M bước): khởi đầu tốt hơn nhưng cuối cùng thua model train từ đầu 24-76.

## Lưu log khi tự train
- runs/<tên>/eval_log.csv : điểm theo thời gian (random, tactical, minimax2, minimax4, hard_score)
- runs/<tên>/tb/ : TensorBoard (tensorboard --logdir runs)
- python evaluate_v2.py model.zip --save results/my_eval.csv : ghi kết quả đánh giá

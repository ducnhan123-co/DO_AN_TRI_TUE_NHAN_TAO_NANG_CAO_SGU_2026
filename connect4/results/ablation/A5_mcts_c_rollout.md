## A5 - MCTS: hằng số khám phá c và kiểu rollout (500 mô phỏng, đấu AB d3)

| Cấu hình | T/H/B | Score | Wilson 95% | ms/nước (TB) |
|---|---|---|---|---|
| MCTS c=0.25 vs AB d3 | 77/9/114 | 0.407 | [0.34, 0.48] | 11.1 |
| MCTS c=0.5 vs AB d3 | 96/7/97 | 0.497 | [0.43, 0.57] | 10.5 |
| MCTS c=1.0 vs AB d3 | 92/4/104 | 0.470 | [0.40, 0.54] | 10.5 |
| MCTS c=1.414 vs AB d3 | 81/3/116 | 0.412 | [0.35, 0.48] | 10.6 |
| MCTS c=2.0 vs AB d3 | 69/8/123 | 0.365 | [0.30, 0.43] | 10.9 |
| MCTS c=3.0 vs AB d3 | 85/2/113 | 0.430 | [0.36, 0.50] | 10.8 |
| MCTS rollout=random vs AB d3 | 83/7/110 | 0.432 | [0.37, 0.50] | 10.6 |
| MCTS rollout=light vs AB d3 | 106/7/87 | 0.547 | [0.48, 0.61] | 26.8 |

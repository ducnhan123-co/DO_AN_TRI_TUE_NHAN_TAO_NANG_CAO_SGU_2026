# Ablation tác tử PPO chính agent_8x9.zip: shield (mask 1 nước) / flip_avg. 100 ván mỗi ô, đổi bên, 2 nước mở đầu ngẫu nhiên.
import warnings; warnings.filterwarnings("ignore")
import torch; torch.set_num_threads(1)
from c4core import TacticalPlayer, MinimaxPlayer, RandomPlayer, match
from c4env import TorchPlayer, load_model
from expstats import wilson_interval
m = load_model("models/agent_8x9.zip")
base = TorchPlayer(m.policy, deterministic=True, flip_avg=True)
variants = {"đầy đủ (mask + gương)": {}, "bỏ gương": {"flip_avg": False},
            "bỏ mask": {"shield": False}, "bỏ cả hai": {"shield": False, "flip_avg": False}}
opps = {"random": lambda: RandomPlayer(1), "tactical": lambda: TacticalPlayer(0, 2), "minimax2": lambda: MinimaxPlayer(2, seed=3), "minimax4": lambda: MinimaxPlayer(4, seed=4)}
print("variant,opponent,games,win,draw,loss,score,ci_lo,ci_hi")
for vn, o in variants.items():
    p = base.with_options(seed=0, **o)
    for on, mk in opps.items():
        r = match(p, mk(), 100, 2, seed=11)
        lo, hi = wilson_interval(r["score"]*100, 100)
        print(f"{vn},{on},100,{r['win']},{r['draw']},{r['loss']},{r['score']:.2f},{lo:.2f},{hi:.2f}", flush=True)

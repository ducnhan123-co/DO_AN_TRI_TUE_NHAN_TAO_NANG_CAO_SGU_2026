import numpy as np, random, time
from c4core import *

def brute_win(grid, s):
    R,C=grid.shape
    for r in range(R):
        for c in range(C):
            for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
                if all(0<=r+dr*i<R and 0<=c+dc*i<C and grid[r+dr*i,c+dc*i]==s for i in range(4)): return True
    return False

rng=random.Random(1)
n=0
for g in range(3000):
    grid=np.zeros((6,7),dtype=np.int8); heights=[0]*7; bbs=[0,0]
    for ply in range(42):
        cols=[c for c in range(7) if heights[c]<6]
        if not cols: break
        c=rng.choice(cols); r=5-heights[c]; heights[c]+=1
        p=ply%2; grid[r,c]=p+1; bbs[p]|=drop_bit(bbs[0]|bbs[1],c)
        for pp in (0,1):
            assert has_won(bbs[pp])==brute_win(grid,pp+1),(grid,pp)
        a,b=grid_to_bb(grid,1); assert (a,b)==(bbs[0],bbs[1])
        assert (bb_to_grid(bbs[0],bbs[1],1)==grid).all()
        assert (legal_mask(bbs[0]|bbs[1])==np.array([h<6 for h in heights])).all()
        n+=1
        if has_won(bbs[0]) or has_won(bbs[1]): break
print("bitboard OK trên", n, "vị trí")

# minimax: thắng ngay, chặn ngay, thắng trong 3 nước (2 mối đe doạ)
def pos(rows):  # rows: 6 chuỗi (trên->dưới) ký tự . X O ; X đi
    g=np.zeros((6,7),dtype=np.int8)
    for r,s in enumerate(rows):
        for c,ch in enumerate(s): g[r,c]={'.':0,'X':1,'O':2}[ch]
    return g
g=pos(["......." ,"......." ,"......." ,"......." ,"OO....." ,"XXX...."])
me,opp=grid_to_bb(g,1); assert minimax_move(me,opp,2)==3
g=pos(["......." ,"......." ,"......." ,"......." ,"XX....." ,"OOO...."])
me,opp=grid_to_bb(g,1); assert minimax_move(me,opp,2)==3, minimax_move(me,opp,2)
# bẫy 2 đầu: X . X X . -> đặt giữa tạo 2 đầu mở (cần đáy kín)
g=pos(["......." ,"......." ,"......." ,"......." ,"......." ,".XX...."])
g[5,0]=0
me,opp=grid_to_bb(g,1)
# O có 2 quân, đến lượt X: (số quân X=2, O=0 -> sai chẵn lẻ nhưng chỉ test không crash)
print("minimax cơ bản OK")
t=time.time(); rr=np.random.default_rng(0)
for d in (2,4,6):
    t=time.time(); m=minimax_move(0,0,d,rr); print(f"depth {d}: mở đầu chọn cột {m}  {time.time()-t:.3f}s")
# đếm tốc độ giữa ván
me,opp=0,0
rg=np.random.default_rng(3)
for _ in range(10):
    cols=legal_cols(me|opp); c=int(rg.choice(cols)); mv=drop_bit(me|opp,c); me|=mv; me,opp=opp,me
for d in (2,4,6):
    t=time.time(); m=minimax_move(me,opp,d,rr); print(f"giữa ván depth {d}: {time.time()-t:.3f}s")
# minimax d4 vs random
a=MinimaxPlayer(4,seed=1); b=RandomPlayer(2)
print(match(a,b,20))

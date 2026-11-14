import numpy as np
import pytest
from connect4_lab05 import *

def test_basics():
    b=empty_board((4,4))
    assert actions(b)==[0,1,2,3]
    a=result(b,1,2)
    assert a[3,2]==1 and np.count_nonzero(b)==0
    assert utility(a,1) is None and not terminal(a)
    with pytest.raises(ValueError):result(a,1,9)

def test_four_vertical_horizontal_diagonal():
    a=empty_board((4,4))
    for _ in range(4):
        a=result(a,1,0)
    assert utility(a,1)==1 and utility(a,-1)==-1 and terminal(a)
    with pytest.raises(ValueError):result(a,-1,1)
    b=empty_board((4,4))
    for c in range(4):b=result(b,1,c)
    assert utility(b,1)==1
    c=empty_board((4,4))
    for col,height in enumerate((1,2,3,4)):
        for n in range(height-1):
            c=result(c,-1,col)
        c=result(c,1,col)
    assert utility(c,1)==1

def test_alphabeta_forced_win_and_block():
    w=empty_board((4,4))
    for _ in range(3):w=result(w,1,0)
    assert alpha_beta_search(w,player=1,cutoff=2).move==0
    lose=empty_board((4,4))
    for _ in range(3):lose=result(lose,-1,3)
    assert alpha_beta_search(lose,player=1,cutoff=2).move==3

def test_alpha_beta_equivalence_and_pmcs():
    b=empty_board((4,4))
    for turn,c in ((1,0),(-1,1),(1,0),(-1,1),(1,2),(-1,2)):
        b=result(b,turn,c)
    a=alpha_beta_search(b,player=1,cutoff=None,move_ordering=False)
    o=alpha_beta_search(b,player=1,cutoff=None,move_ordering=True)
    assert a.value==o.value
    assert a.move in actions(b) and o.move in actions(b)
    assert pmcs(b,player=1,N=40,rng=np.random.default_rng(42)) in actions(b)
    with pytest.raises(ValueError):pmcs(b,player=1,N=1)

def test_determinism():
    b=empty_board((4,4))
    x=pmcs(b,N=40,rng=np.random.default_rng(42))
    y=pmcs(b,N=40,rng=np.random.default_rng(42))
    assert x==y

def test_minimax_alpha_beta_same_value():
    b=empty_board((4,4))
    for turn,c in ((1,0),(-1,1),(1,0),(-1,1),(1,2),(-1,2)):
        b=result(b,turn,c)
    slow=minimax_search(b,player=1,cutoff=None)
    fast=alpha_beta_search(b,player=1,cutoff=None,move_ordering=False)
    assert slow.value==fast.value and slow.move in actions(b)
    assert fast.nodes<=slow.nodes

def test_agent_class():
    board=empty_board((4,4))
    a=Connect4Agent('pmcs',N=40,seed=5)
    assert a(board,1) in actions(board)

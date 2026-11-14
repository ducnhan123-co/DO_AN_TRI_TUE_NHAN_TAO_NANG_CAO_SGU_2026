import nbformat
from pathlib import Path
base=Path(__file__).parent
for fn,nums in [("01_trees.ipynb",[2,6,8,10,12]),("02_Maze_Example.ipynb",[1,3,6,8,10,12,14,33,34]),("03_Maze_Example_Animation.ipynb",[1,3,6,8,10,12,15,17])]:
 n=nbformat.read(base/fn,as_version=4)
 print("\n####",fn)
 for i in nums:print("CELL",i,"\n",n.cells[i].source[:2500])

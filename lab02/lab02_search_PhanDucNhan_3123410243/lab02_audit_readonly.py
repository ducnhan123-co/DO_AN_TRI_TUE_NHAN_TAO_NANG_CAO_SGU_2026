import json, pathlib, re
base=pathlib.Path(__file__).parent
paths=["01_trees.ipynb","Lab02_4_5/02_Maze_Example.ipynb","Lab02_4_5/04_Maze_BFS_vs_A_Star.ipynb","lab02_Bai6_Bai7_code/lab02_code/05_Maze_Example-Animation_Cycle_checking.ipynb","lab02_Bai6_Bai7_code/lab02_code/06_Explore_heuristics.ipynb","Lab02_Bai2+8/Lab02_Bai2+8/Ex1_Maze.ipynb","Lab02_Bai2+8/Lab02_Bai2+8/Ex2_Maze_simplified.ipynb","Lab02_Bai2+8/Lab02_Bai2+8/Ex3_Show_all_mazes.ipynb"]
for name in paths:
 try:
  j=json.loads((base/name).read_text())
  code=[c for c in j["cells"] if c["cell_type"]=="code"]
  md=[c for c in j["cells"] if c["cell_type"]=="markdown"]
  errs=[(o.get("ename"),str(o.get("evalue"))[:85]) for c in code for o in c.get("outputs",[]) if o["output_type"]=="error"]
  out=sum(bool(c.get("outputs")) for c in code)
  todo=sum(bool(re.search("TODO|YOUR CODE|YOUR ANSWER|NotImplementedError", "".join(c["source"]),re.I)) for c in j["cells"])
  headings=[" ".join("".join(c["source"]).split())[:110] for c in md if "Task" in "".join(c["source"]) or "Bài" in "".join(c["source"])]
  print(name,"code",len(code),"markdown",len(md),"output",out,"errors",errs[:2],"markers",todo,"tasks",headings[:7])
 except Exception as exc:print(name,"ERROR",str(exc))

from pathlib import Path
import json, math, time, csv
import numpy as np
import maze_helper as mh
import tree_search_solution as ts

p=Path(__file__).parent
ts.set_order("NESW",random=False)
results=[]
for name in ["small","medium","large","open","L","loops","empty_maze","empty_maze_2"]:
 path=p/((name if name.startswith("empty") else name+"_maze")+".txt")
 a=mh.parse_maze(path.read_text())
 start,goal=mh.find_pos(a,"S"),mh.find_pos(a,"G")
 n=int((a!="X").sum())
 print("MAP",name,a.shape,"n",n,"start",start,"goal",goal)
 for kind in ["BFS","GBFS","A*"]:
  ts.heuristic=ts.manhattan
  try:
   r=ts.best_first_search(a,strategy=kind)
   row=[name,kind,r["cost"],r["expanded"],r["max_depth"],r["max_frontier"],round(r["time"]*1000,3)]
   print("RUN",*row)
   results.append(row)
  except Exception as e:print("ERROR",name,kind,repr(e))
 if name in ["small","medium","large","open"]:
  try:
   r=ts.DFS(a,max_tries=5000,frontier_option=2)
   print("DFS",name,r["cost"],r["expanded"],r["max_depth"],r["max_frontier"],r["cutoff"])
   results.append([name,"DFS-limit5000",r["cost"],r["expanded"],r["max_depth"],r["max_frontier"],round(r["time"]*1000,3)])
  except Exception as e:print("ERROR",name,"DFS",repr(e))
if results:
 with (p/"ket_qua_xac_minh_report.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["maze","algorithm","cost","expanded_including_goal","maxdepth","maxfrontier","runtime_ms"]);w.writerows(results)

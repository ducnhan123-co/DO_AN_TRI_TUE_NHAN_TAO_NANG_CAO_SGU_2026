import json,pathlib,sys,importlib.util
p=pathlib.Path(__file__).parent
print("VERSIONS",sys.version.split()[0],{x: bool(importlib.util.find_spec(x)) for x in ["nbclient","nbformat","ipykernel","matplotlib","numpy","docx","PIL"]})
for name in ["01_trees.ipynb","02_Maze_Example.ipynb","03_Maze_Example_Animation.ipynb"]:
 d=json.loads((p/name).read_text())
 print("\n###",name,"kernel",d.get("metadata",{}).get("kernelspec"))
 for i,c in enumerate(d["cells"]):
  src="".join(c.get("source",[]))
  if c["cell_type"]=="code":
   print(f"CODE {i}: lines={len(src.splitlines())} chars={len(src)} run={c.get('execution_count')} output={len(c.get('outputs',[]))}: "+' '.join(src.split())[:320])
  else:
   print(f"MARKDOWN {i}: "+' '.join(src.split())[:160])

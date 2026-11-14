from pathlib import Path
from shutil import copy2
import nbformat, time
from nbclient import NotebookClient
b=Path(__file__).parent
backup=b/"BACKUP_BAI1_BAI3_20261009"
backup.mkdir(exist_ok=True)
for f in ["01_trees.ipynb","02_Maze_Example.ipynb","03_Maze_Example_Animation.ipynb","lab02_search_PhanDucNhan_3123410243.docx"]:
 src=b/f;dst=backup/f
 if not dst.exists():copy2(src,dst);print("BACKUP",f)
n=nbformat.read(b/"01_trees.ipynb",as_version=4)
t=time.perf_counter()
try:
 out=NotebookClient(n,timeout=45,kernel_name="python3",resources={"metadata":{"path":str(b)}}).execute()
 errors=[(i,x.get("ename"),x.get("evalue")) for i,c in enumerate(out.cells) if c.cell_type=="code" for x in c.get("outputs",[]) if x.output_type=="error"]
 nbformat.write(out,b/"01_trees.ipynb")
 print("RUN_ALL_01",round(time.perf_counter()-t,3),"errors",errors,"outputs",[(i,len(c.get("outputs",[]))) for i,c in enumerate(out.cells) if c.cell_type=="code"])
except Exception as e:print("RUN_ALL_01_FAILED",type(e).__name__,str(e)[:1300])

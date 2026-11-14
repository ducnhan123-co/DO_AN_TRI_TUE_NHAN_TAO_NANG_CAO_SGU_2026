import nbformat,sys,time,json,traceback
from pathlib import Path
from nbclient import NotebookClient
b=Path(__file__).parent
name=sys.argv[1]
n=nbformat.read(b/name,as_version=4)
start=time.perf_counter()
try:
 out=NotebookClient(n,timeout=50,kernel_name="python3",resources={"metadata":{"path":str(b)}}).execute()
 errors=[(i,x.get("ename"),x.get("evalue")) for i,c in enumerate(out.cells) if c.cell_type=="code" for x in c.get("outputs",[]) if x.output_type=="error"]
 assert not errors, errors
 nbformat.write(out,b/name)
 report={"file":name,"status":"PASS","seconds":round(time.perf_counter()-start,2),
         "code_cells":sum(c.cell_type=="code" for c in out.cells),
         "output_cells":sum(c.cell_type=="code" and bool(c.get("outputs")) for c in out.cells),"errors":errors}
 print("RUN_RESULT",json.dumps(report,ensure_ascii=False))
except BaseException as e:
 report={"file":name,"status":"FAIL","seconds":round(time.perf_counter()-start,2),
         "error_type":type(e).__name__,"message":str(e)[:1800]}
 print("RUN_RESULT",json.dumps(report,ensure_ascii=False))
 raise

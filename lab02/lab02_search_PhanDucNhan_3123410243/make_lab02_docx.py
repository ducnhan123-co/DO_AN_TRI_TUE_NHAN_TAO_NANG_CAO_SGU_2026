from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import maze_helper as mh
import tree_search_solution as ts

ROOT=Path(__file__).parent
SRC=ROOT/"LAB02_BAO_CAO_TONG_HOP.md"
OUT=ROOT/"lab02_search_PhanDucNhan_3123410243.docx"

def make_fig(name):
    maze=mh.parse_maze((ROOT/(name+"_maze.txt")).read_text())
    ts.set_order("NESW",random=False)
    ts.heuristic=ts.manhattan
    res1=ts.best_first_search(maze,"BFS")
    res2=ts.best_first_search(maze,"A*")
    fig,ax=plt.subplots(1,2,figsize=(11,4.6))
    mh.render(maze,explored=res1["explored"],path=res1["path"],ax=ax[0],title=f"{name}: BFS | cost={res1['cost']} | expanded={res1['expanded']}")
    mh.render(maze,explored=res2["explored"],path=res2["path"],ax=ax[1],title=f"{name}: A* Manhattan | cost={res2['cost']} | expanded={res2['expanded']}")
    fig.tight_layout(pad=1.5)
    img=ROOT/(f"lab02_verified_{name}_bfs_astar.png")
    fig.savefig(img,dpi=170,bbox_inches="tight")
    plt.close(fig)
    return img

fig_small=make_fig("small")
fig_open=make_fig("open")

doc=Document()
sec=doc.sections[0]
sec.page_height=Cm(29.7);sec.page_width=Cm(21)
sec.top_margin=Cm(2.2);sec.bottom_margin=Cm(2);sec.left_margin=Cm(2.3);sec.right_margin=Cm(2.1)
sec.header_distance=Cm(.9);sec.footer_distance=Cm(.9)
styles=doc.styles
styles["Normal"].font.name="Times New Roman";styles["Normal"].font.size=Pt(11)
styles["Normal"].paragraph_format.space_after=Pt(6)
styles["Normal"].paragraph_format.line_spacing=1.18
for key,size in [("Title",19),("Heading 1",14),("Heading 2",12)]:
    st=styles[key];st.font.name="Times New Roman";st.font.size=Pt(size);st.font.bold=True
    st.font.color.rgb=RGBColor(22,54,91)
    st.paragraph_format.space_before=Pt(9);st.paragraph_format.space_after=Pt(6)

header=sec.header.paragraphs[0]
header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
rr=header.add_run("SGU  |  TRÍ TUỆ NHÂN TẠO NÂNG CAO  |  LAB 02")
rr.font.name="Arial";rr.font.size=Pt(8);rr.font.color.rgb=RGBColor(95,109,125)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
run=footer.add_run("LAB 02  •  Trang ")
run.font.size=Pt(9)
fld=OxmlElement("w:fldSimple");fld.set(qn("w:instr"),"PAGE");footer._p.append(fld)

def set_cell(cell, text, bold=False):
    cell.text=text
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for p in cell.paragraphs:
        p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(1)
        for r in p.runs:
            r.font.name="Times New Roman";r.font.size=Pt(8 if bold else 8.3);r.font.bold=bold

def add_table(lines):
    rows=[]
    for line in lines:
        cells=[x.strip() for x in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?",v or "") for v in cells):continue
        rows.append(cells)
    if not rows:return
    cols=max(len(r) for r in rows)
    table=doc.add_table(rows=len(rows),cols=cols)
    table.style="Table Grid";table.alignment=WD_TABLE_ALIGNMENT.CENTER
    table.autofit=True
    for i,row in enumerate(rows):
        for j in range(cols):
            set_cell(table.cell(i,j),row[j] if j<len(row) else "",i==0)
        if i==0:
            trPr=table.rows[i]._tr.get_or_add_trPr();v=OxmlElement("w:tblHeader");v.set(qn("w:val"),"true");trPr.append(v)
            for cell in table.rows[i].cells:
                shd=OxmlElement("w:shd");shd.set(qn("w:fill"),"DFEAF6");cell._tc.get_or_add_tcPr().append(shd)
    doc.add_paragraph()

source=SRC.read_text(encoding="utf-8")
lines=source.splitlines()
tablelines=[]
first=True
for line in lines:
    stripped=line.strip()
    if stripped.startswith("|"):
        tablelines.append(stripped)
        continue
    if tablelines:
        add_table(tablelines);tablelines=[]
    if not stripped:continue
    if stripped in ("[[FIG_SMALL]]","[[FIG_OPEN]]"):
        f=fig_small if stripped=="[[FIG_SMALL]]" else fig_open
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(f),width=Cm(16))
        p2=doc.add_paragraph(("Hình 1. " if stripped=="[[FIG_SMALL]]" else "Hình 2. ")+"Đường đi và vùng tìm kiếm, kết quả chạy trực tiếp trên dữ liệu mê cung.")
        p2.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in p2.runs:r.font.italic=True;r.font.size=Pt(9)
        continue
    m=re.match(r"^(#{1,3})\s+(.*)",line)
    if m:
        level=len(m.group(1));value=m.group(2)
        if first:
            p=doc.add_paragraph(value,style="Title");p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            first=False
        elif level==2 and ("GIẢI QUYẾT VẤN ĐỀ" in value):
            p=doc.add_paragraph(value,style="Heading 1");p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif level==1:
            if value.startswith("1. MỤC TIÊU"):
                doc.add_page_break()
            doc.add_heading(value,1)
        else:doc.add_heading(value,min(level,2))
        continue
    p=doc.add_paragraph()
    if stripped.startswith("Môn học:") or stripped.startswith("Sinh viên") or stripped.startswith("Thành viên"):
        p.paragraph_format.space_after=Pt(3)
    p.add_run(line)
if tablelines:add_table(tablelines)
doc.core_properties.title="Báo cáo Lab 02 - Giải quyết vấn đề bằng tìm kiếm"
doc.core_properties.subject="BFS, DFS, DLS, IDS, GBFS, A*, heuristic, cycle checking"
doc.save(OUT)
reload=Document(OUT)
print("CREATED",OUT,"bytes",OUT.stat().st_size,"tables",len(reload.tables),"paragraphs",len(reload.paragraphs),"images",len(reload.inline_shapes))
print("FIG",fig_small.name,fig_small.stat().st_size,fig_open.name,fig_open.stat().st_size)

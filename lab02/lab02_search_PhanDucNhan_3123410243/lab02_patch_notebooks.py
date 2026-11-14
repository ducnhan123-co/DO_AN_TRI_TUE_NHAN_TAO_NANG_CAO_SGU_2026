from pathlib import Path
import nbformat
P=Path(__file__).parent
M=nbformat.v4.new_markdown_cell
C=nbformat.v4.new_code_cell

def write(name,n):
    for x in n.cells:
        if x.cell_type=="code": x.outputs=[]; x.execution_count=None
    n.nbformat_minor=5
    nbformat.validate(n)
    nbformat.write(n,P/name)
    print("PATCHED",name,"cells",len(n.cells))

a=nbformat.read(P/"BACKUP_BAI1_BAI3_20261009"/"01_trees.ipynb",as_version=4)
a.cells.extend([
 M("## Bài 1 – Chạy kiểm chứng và truy hồi đường đi\nDùng lại lớp Node của notebook gốc, tạo một cây ba nút, kiểm tra pos, parent, action, cost và get_path_from_root(). Số reference-count của Python không được dùng làm kết quả cố định."),
 C("""import matplotlib.pyplot as plt
from IPython.display import display
from pathlib import Path
test_root = Node((0,0),None,None,0)
test_east = Node((0,1),test_root,'E',1)
test_south = Node((1,1),test_east,'S',2)
path_nodes = test_south.get_path_from_root()
positions = [n.pos for n in path_nodes]
actions = [n.action for n in path_nodes[1:]]
costs = [n.cost for n in path_nodes]
assert positions == [(0,0),(0,1),(1,1)] and actions == ['E','S']
assert costs == [0,1,2] and path_nodes[2].parent is path_nodes[1]
print('Positions:',positions)
print('Actions:',actions)
print('Costs:',costs,'| Final path cost:',test_south.cost)
print('PASS: Node, parent, get_path_from_root()')
fig,ax=plt.subplots(figsize=(9,3))
for x,label in [(.12,'Root\\n(0,0)'),(.50,'E\\n(0,1)'),(.88,'S\\n(1,1)')]:
    ax.scatter([x],[.5],s=3300,c='#e0edf8',edgecolors='#386f9d',linewidths=2)
    ax.text(x,.5,label,ha='center',va='center',fontsize=11)
for start,end,lab in [(.23,.39,'E / +1'),(.61,.77,'S / +1')]:
    ax.annotate('',xy=(end,.5),xytext=(start,.5),arrowprops={'arrowstyle':'->','color':'#386f9d','lw':2})
    ax.text((start+end)/2,.73,lab,ha='center',fontsize=10)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.set_title('Bai 1: Cay tim kiem duoc tao dong')
fig.tight_layout()
fig.savefig('lab02_bai1_node_tree.png',dpi=170,bbox_inches='tight')
display(fig);plt.close(fig)"""),
 M("**Nhận xét:** Node chứa trạng thái và thông tin của đường đi (parent, action, cost), trong khi một trạng thái có thể xuất hiện ở nhiều nút tìm kiếm với các parent khác nhau. Cây được sinh động khi mở rộng nút. Trong mê cung có tối đa bốn hướng đi, cây không nhất thiết là cây nhị phân. Kết quả có thể tái lập: vị trí [(0,0),(0,1),(1,1)], hành động [E,S], chi phí đường đi bằng 2.")
])
write("01_trees.ipynb",a)

a=nbformat.read(P/"02_Maze_Example.ipynb",as_version=4)
for i in [8,10]:a.cells[i].source=a.cells[i].source.replace("max_tries = 100000","max_tries = 5000")
a.cells[12].source=a.cells[12].source.replace("max_tries = 100000","max_tries = 1500").replace("for _ in range(5):","import random\nrandom.seed(20261009)\nfor _ in range(5):")
a.cells[14].source=a.cells[14].source.replace("N = 100","import random\nrandom.seed(20261009)\nN = 12").replace("max_tries = 10000","max_tries = 3000")
a.cells[33].source="""from pathlib import Path
maze = parse_maze(Path('small_maze.txt').read_text())
ts.show_maze(maze)
ts.set_order('NESW',random=False)
print('Timing: small_maze, NESW, fixed tie policy')"""
a.cells[34].source="""# Correct: DFS must call DFS(), not best_first_search(... strategy='DFS')
import timeit
reps=3
ts.set_order('NESW',random=False)
ts.heuristic=ts.manhattan
measure = {
 'BFS':lambda:ts.best_first_search(maze,strategy='BFS'),
 'DFS':lambda:ts.DFS(maze,frontier_option=2,max_tries=5000),
 'GBFS':lambda:ts.best_first_search(maze,strategy='GBFS'),
 'A*':lambda:ts.best_first_search(maze,strategy='A*'),
 'IDS':lambda:ts.IDS(maze,frontier_option=2,max_tries=5000),
}
times={k:round(timeit.timeit(f,number=reps)*1e6/reps,1) for k,f in measure.items()}
print('microseconds / run:',times)"""
a.cells.extend([
 M("## Bài 3 – Thực nghiệm BFS và DFS đã kiểm chứng\nSo sánh trên small, medium, open. BFS dùng FIFO, DFS dùng LIFO. Cấu hình: NESW cố định, DFS frontier_option=2, max_tries=5000. Module tree_search_solution.py đếm cả lượt lấy nút đích trong Expanded. Các khung hình được vẽ trực tiếp từ kết quả chạy, không phải hình minh họa giả định."),
 C("""from pathlib import Path
import pandas as pd
import maze_helper as mh
import tree_search_solution as ts
from IPython.display import display
ts.set_order('NESW',random=False)
ts.heuristic=ts.manhattan
records=[];cases={}
for maze_name in ['small','medium','open']:
    arr=mh.parse_maze(Path(maze_name+'_maze.txt').read_text())
    start,goal=mh.find_pos(arr,'S'),mh.find_pos(arr,'G')
    for algorithm in ['BFS','DFS']:
        res=ts.best_first_search(arr,strategy='BFS') if algorithm=='BFS' else ts.DFS(arr,frontier_option=2,max_tries=5000)
        if res['path'] is not None:
            assert res['path'][0]==start and res['path'][-1]==goal
            assert len(res['path'])-1==res['cost']
            assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(res['path'],res['path'][1:]))
            assert all(arr[r,c]!='X' for r,c in res['path'])
        records.append({'Maze':maze_name,'Algorithm':algorithm,'Cost':res['cost'],
          'Expanded':res['expanded'],'Max depth':res['max_depth'],
          'Max frontier':res['max_frontier'],'Time ms':round(res['time']*1000,3)})
        cases[(maze_name,algorithm)]=(arr,res)
bai3_table=pd.DataFrame(records)
display(bai3_table)
bai3_table.to_csv('lab02_bai3_bfs_dfs_verified.csv',index=False)
print('PASS - paths valid, CSV saved')"""),
 C("""import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,7))
for i,maze_name in enumerate(['small','open']):
    for j,algorithm in enumerate(['BFS','DFS']):
        arr,res=cases[(maze_name,algorithm)]
        mh.render(arr,explored=res['explored'],path=res['path'],ax=axes[i,j],
          title=f'{maze_name} | {algorithm} | cost={res["cost"]} | expanded={res["expanded"]}')
fig.tight_layout(pad=1.4)
fig.savefig('lab02_bai3_bfs_dfs_paths.png',dpi=170,bbox_inches='tight')
display(fig);plt.close(fig)
print('Saved lab02_bai3_bfs_dfs_paths.png')"""),
 M("### Tính phụ thuộc vào thứ tự duyệt và ngẫu nhiên\nGiữ dữ liệu small_maze, thay seed theo từng lần chạy DFS. Với giới hạn lượt mở rộng, nếu không tìm được đường trong một lần chạy thì không thể kết luận bài toán vô nghiệm."),
 C("""import random
runs=[]
small=mh.parse_maze(Path('small_maze.txt').read_text())
for seed in range(10,18):
    random.seed(seed)
    ts.set_order('NESW',random=True)
    x=ts.DFS(small,frontier_option=2,max_tries=5000)
    runs.append({'Seed':seed,'Found':x['path'] is not None,'Cost':x['cost'],
                 'Expanded':x['expanded'],'Max frontier':x['max_frontier']})
ts.set_order('NESW',random=False)
bai3_random=pd.DataFrame(runs)
display(bai3_random)
bai3_random.to_csv('lab02_bai3_random_dfs_seeds.csv',index=False)
print('Found:',int(bai3_random.Found.sum()),'/',len(bai3_random))"""),
 M("**Kết luận Bài 3:** BFS tìm đường ngắn nhất với chi phí mỗi bước bằng nhau và reached phù hợp. DFS đi sâu trước, nghiệm đầu tiên có thể dài hơn tối ưu, phụ thuộc thứ tự successor và seed. Max frontier là kích thước hàng đợi/stack quan sát được, không tương đương toàn bộ lượng bộ nhớ cần cho reached và parent. Restart Kernel → Run All để tái lập bảng, CSV và hình.")
])
write("02_Maze_Example.ipynb",a)

a=nbformat.read(P/"03_Maze_Example_Animation.ipynb",as_version=4)
for i in [8,10,12]:
    a.cells[i].source=a.cells[i].source.replace("max_tries = 100000","max_tries = 1200")
for i in [6,8,10,12,15,17]:
    a.cells[i].source=a.cells[i].source.replace("ts.animate_maze(result)","ts.animate_maze(result, frames=24)")
a.cells[12].source="import random\nrandom.seed(20261009)\n"+a.cells[12].source
for i in [15,17]:
    a.cells[i].source=a.cells[i].source.replace("ts.set_order(random=True)","ts.set_order('NESW',random=False)")
a.cells.append(M("## Bài 3 – Nhận xét từ animation\nBFS khảo sát các lớp trạng thái theo độ sâu, còn DFS đi sâu trước; đổi thứ tự sinh successor NESW/ENWS hoặc seed làm thay đổi đường đi. Các animation chỉ lấy 24 khung hình để giảm dung lượng. Các lượt DFS được giới hạn 1200 pop để tránh Run All treo: dừng vì giới hạn **không có nghĩa** mê cung vô nghiệm. Số liệu chính của Bài 3 được kiểm chứng trên các bảng CSV và hình trong 02_Maze_Example.ipynb."))
write("03_Maze_Example_Animation.ipynb",a)

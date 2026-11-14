"""Cài đặt tìm kiếm dùng cho Lab 02 (thay cho tree_search_solution.py không được công bố).
Cung cấp: set_order, DFS (3 kiểu cycle checking), BFS, best_first (GBFS/A*/Weighted A*), animate_maze.
Không dùng đường dẫn tuyệt đối; các file *.txt đọc từ thư mục hiện hành.
"""
import heapq, time, math, random
import numpy as np
from maze_helper import find_pos, animate_maze, parse_maze, show_maze   # noqa: F401 (animate_maze được notebook gọi qua ts.)

_ORDER = {"order": "NESW", "random": False}
_DIRS = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}

def set_order(order="NESW", random=False):
    """Thứ tự thêm các trạng thái con vào frontier (N,E,S,W). random=True: xáo trộn mỗi lần mở rộng."""
    _ORDER["order"], _ORDER["random"] = order, random

def _neighbors(maze, p, order):
    out = []
    for a in order:
        q = (p[0] + _DIRS[a][0], p[1] + _DIRS[a][1])
        if 0 <= q[0] < maze.shape[0] and 0 <= q[1] < maze.shape[1] and maze[q] != 'X':
            out.append((a, q))
    return out

class Node:
    __slots__ = ("pos", "parent", "action", "cost")
    def __init__(self, pos, parent=None, action=None, cost=0):
        self.pos, self.parent, self.action, self.cost = pos, parent, action, cost
    def path_nodes(self):
        n, r = self, []
        while n is not None: r.append(n); n = n.parent
        return r[::-1]

def _states_on_path(n):
    s = set()
    while n is not None: s.add(n.pos); n = n.parent
    return s

def DFS(maze, vis=False, animation=False, max_tries=500, frontier_option=1, rng=None, frame_step=5):
    """DFS tìm kiếm trên cây, chỉ nhớ đường đi hiện tại (không có bảng reached).
    frontier_option (khớp với notebook 05):
      3 = kiểm tra đường đi (bỏ qua con có trạng thái đã nằm trên đường đi hiện tại)
      2 = kiểm tra đường đi + không thêm lại trạng thái đang có trong frontier
      1 = kiểm tra đường đi + đưa trạng thái tìm thấy lại lên đầu stack
    animation=True: lưu 1 khung hình mỗi frame_step lần lặp (và khung cuối) để file notebook không quá nặng.
    Trả về dict: path, reached, actions, maze_anim, stats."""
    start, goal = find_pos(maze, 'S'), find_pos(maze, 'G')
    frontier = [Node(start)]                    # đỉnh stack = cuối danh sách
    it = exp = maxf = 0; maxf = 1; maxdepth = 0; dupmax = 0
    expanded = {}; frames = []; sol = None
    def snapshot(cur_node, force=False):
        if not force and it % frame_step != 1 and frame_step > 1: return
        m = np.copy(maze)
        for f in frontier:
            if m[f.pos] == ' ': m[f.pos] = 'F'
        for n in cur_node.path_nodes():
            if m[n.pos] in (' ', 'F'): m[n.pos] = 'P'
        frames.append(m)
    while frontier and it < max_tries:
        n = frontier.pop(); it += 1
        anc = _states_on_path(n.parent)
        if n.pos in anc:                         # cycle check khi lấy khỏi frontier
            if animation: snapshot(n)
            continue
        if n.pos == goal: sol = n; break
        exp += 1; expanded[n.pos] = expanded.get(n.pos, 0) + 1
        maxdepth = max(maxdepth, len(anc))
        order = ''.join((rng or random).sample("NESW", 4)) if _ORDER["random"] else _ORDER["order"]
        onpath = anc | {n.pos}
        for a, q in _neighbors(maze, n.pos, order):
            if q in onpath: continue
            if frontier_option == 2 and any(x.pos == q for x in frontier): continue
            if frontier_option == 1: frontier = [x for x in frontier if x.pos != q]
            frontier.append(Node(q, n, a, n.cost + 1))
        maxf = max(maxf, len(frontier))
        dupmax = max(dupmax, len(frontier) - len({x.pos for x in frontier}))
        if animation: snapshot(n)
    if animation and 'n' in locals(): snapshot(n, force=True)
    stats = dict(solved=sol is not None, iterations=it, expanded=exp, max_frontier=maxf,
                 distinct_states=len(expanded), reexpansions=exp - len(expanded),
                 max_dup_in_frontier=dupmax, max_depth=maxdepth, path_len=sol.cost if sol else None)
    if sol is None:
        return dict(path=None, reached=set(expanded), actions=None, maze_anim=frames or [maze], stats=stats)
    nodes = sol.path_nodes()
    if animation:
        m = np.copy(maze)
        for x in nodes:
            if m[x.pos] == ' ': m[x.pos] = 'P'
        frames.append(m)
    return dict(path=[x.pos for x in nodes], reached=set(expanded), actions=[x.action for x in nodes[1:]],
                maze_anim=frames or [maze], stats=stats)

# ---------------- BFS, GBFS, A*, Weighted A* (có bảng reached) ----------------
H = {
    'manhattan': lambda p, g: abs(p[0]-g[0]) + abs(p[1]-g[1]),
    'euclid':    lambda p, g: math.hypot(p[0]-g[0], p[1]-g[1]),
    'chebyshev': lambda p, g: max(abs(p[0]-g[0]), abs(p[1]-g[1])),
}

def best_first(maze, kind, h='manhattan', w=1.0, tie='fifo'):
    """kind: 'gbfs' (f=h), 'astar' (f=g+w*h), 'ucs' (f=g). tie: 'fifo' | 'lifo' khi f bằng nhau."""
    start, goal = find_pos(maze, 'S'), find_pos(maze, 'G'); hf = H[h]
    pr = lambda g, p: hf(p, goal) if kind == 'gbfs' else (g if kind == 'ucs' else g + w*hf(p, goal))
    t = time.perf_counter(); cnt = 0
    fr = [(pr(0, start), 0, start, 0)]; reached = {start: 0}; parent = {start: None}; exp = 0; maxf = 1; order = []
    while fr:
        f, _, p, g = heapq.heappop(fr)
        if g > reached[p]: continue
        if p == goal:
            path = []; q = p
            while q is not None: path.append(q); q = parent[q]
            return dict(cost=g, exp=exp, maxfr=maxf, time=time.perf_counter()-t, path=path[::-1], order=order)
        exp += 1; order.append(p)
        for _, q in _neighbors(maze, p, "NESW"):
            if q not in reached or g+1 < reached[q]:
                reached[q] = g+1; parent[q] = p; cnt += 1
                heapq.heappush(fr, (pr(g+1, q), cnt if tie == 'fifo' else -cnt, q, g+1))
        maxf = max(maxf, len(fr))
    return None

def bfs(maze):
    from collections import deque
    start, goal = find_pos(maze, 'S'), find_pos(maze, 'G'); t = time.perf_counter()
    fr = deque([(start, 0)]); reached = {start}; exp = 0; maxf = 1
    while fr:
        p, g = fr.popleft()
        if p == goal: return dict(cost=g, exp=exp, maxfr=maxf, time=time.perf_counter()-t)
        exp += 1
        for _, q in _neighbors(maze, p, "NESW"):
            if q not in reached: reached.add(q); fr.append((q, g+1))
        maxf = max(maxf, len(fr))
    return None

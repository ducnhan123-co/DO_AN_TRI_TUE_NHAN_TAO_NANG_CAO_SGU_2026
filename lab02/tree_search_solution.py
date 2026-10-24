"""Cài đặt các thuật toán tìm kiếm trên mê cung (thay thế cho tree_search_solution.py không công bố).

API:
    set_order, best_first_search (BFS/GBFS/A*/Weighted A*), DFS (có limit = Depth-Limited DFS),
    IDS, show_path, show_maze, animate_maze, manhattan, euclidean, heuristic, min_index.

Kết quả của mỗi thuật toán là một dict gồm:
    path, actions, cost            : đường đi, dãy hành động, chi phí (None nếu không có lời giải)
    expanded, max_depth, max_frontier, time (giây)
    explored                       : tập ô đã được mở rộng (để vẽ vùng xám)
    visit_order                    : thứ tự mở rộng (dùng cho animation)
"""
import heapq
import math
import random
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import maze_helper as mh

DIRS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}

_order = "NESW"
_random = False


def set_order(order="NESW", random=False):
    """Thứ tự hướng đi khi sinh nút con. random=True: xáo trộn ngẫu nhiên mỗi lần mở rộng."""
    global _order, _random
    _order, _random = order, random


class Node:
    """Nút của cây tìm kiếm: pos (trạng thái), parent, action, cost (chi phí tích lũy), depth."""

    def __init__(self, pos, parent, action, cost):
        self.pos, self.parent, self.action, self.cost = pos, parent, action, cost
        self.depth = 0 if parent is None else parent.depth + 1

    def get_path_from_root(self):
        node, out = self, []
        while node is not None:
            out.append(node)
            node = node.parent
        return out[::-1]


def manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def euclidean(pos, goal):
    return math.hypot(pos[0] - goal[0], pos[1] - goal[1])


heuristic = manhattan          # có thể gán lại: ts.heuristic = ts.euclidean


def min_index(values):
    return int(np.argmin(values))


def _children(maze, node):
    dirs = list(_order)
    if _random:
        random.shuffle(dirs)
    out = []
    for a in dirs:
        dr, dc = DIRS[a]
        r, c = node.pos[0] + dr, node.pos[1] + dc
        if maze[r, c] != "X":
            out.append(Node((r, c), node, a, node.cost + 1))
    return out


def _on_path(node, pos):
    while node is not None:
        if node.pos == pos:
            return True
        node = node.parent
    return False


def _result(maze, goal, expanded, max_depth, max_frontier, t0, visit_order, **extra):
    nodes = None if goal is None else goal.get_path_from_root()
    res = dict(maze=maze,
               path=None if nodes is None else [n.pos for n in nodes],
               actions=None if nodes is None else [n.action for n in nodes[1:]],
               cost=None if goal is None else goal.cost,
               expanded=expanded, max_depth=max_depth, max_frontier=max_frontier,
               time=time.perf_counter() - t0,
               visit_order=visit_order, explored=set(visit_order))
    res.update(extra)
    return res


def best_first_search(maze, strategy="A*", W=1.0, debug=False, vis=False, animation=False, **kwargs):
    """BFS, GBFS, A* và Weighted A*. Có bảng reached; hòa điểm: ưu tiên nút thêm vào sau cùng."""
    t0 = time.perf_counter()
    start, goal = mh.find_pos(maze, "S"), mh.find_pos(maze, "G")
    root = Node(start, None, None, 0)

    def f(n):
        if strategy == "GBFS":
            return heuristic(n.pos, goal)
        return n.cost + W * heuristic(n.pos, goal)          # A* (W = 1) hoặc Weighted A*

    reached = {start: root}
    visit, expanded, max_depth, max_frontier = [], 0, 0, 1
    if strategy == "BFS":
        frontier = deque([root])
        while frontier:
            n = frontier.popleft()
            expanded += 1; visit.append(n.pos); max_depth = max(max_depth, n.depth)
            if n.pos == goal:
                return _result(maze, n, expanded, max_depth, max_frontier, t0, visit)
            for k in _children(maze, n):
                if k.pos not in reached:
                    reached[k.pos] = k
                    frontier.append(k)
            max_frontier = max(max_frontier, len(frontier))
    else:
        counter = 0
        frontier = [(f(root), 0, root)]
        while frontier:
            _, _, n = heapq.heappop(frontier)
            if n.cost > reached[n.pos].cost:
                continue                                    
            expanded += 1; visit.append(n.pos); max_depth = max(max_depth, n.depth)
            if n.pos == goal:
                return _result(maze, n, expanded, max_depth, max_frontier, t0, visit)
            for k in _children(maze, n):
                if k.pos not in reached or k.cost < reached[k.pos].cost:
                    reached[k.pos] = k
                    counter += 1
                    heapq.heappush(frontier, (f(k), -counter, k))
            max_frontier = max(max_frontier, len(frontier))
    return _result(maze, None, expanded, max_depth, max_frontier, t0, visit)


def DFS(maze, limit=None, check_cycle=True, frontier_option=1, max_tries=100000, vis=False, **kwargs):
    """DFS dùng stack, không có bảng reached (bộ nhớ O(b·m)).
    limit=None: DFS thường; limit=L: Depth-Limited DFS (nút có depth >= L không được mở rộng).
    check_cycle=True : không thêm nút con nếu trạng thái đã nằm trên đường đi hiện tại.
    frontier_option=2: thêm điều kiện - bỏ nút con nếu frontier đã có cùng trạng thái ở độ sâu <= nút con.
    max_tries        : số lần mở rộng tối đa; vượt quá thì dừng và coi như không tìm thấy."""
    t0 = time.perf_counter()
    start, goal = mh.find_pos(maze, "S"), mh.find_pos(maze, "G")
    frontier = [Node(start, None, None, 0)]
    in_frontier = {start: {0: 1}}
    visit, expanded, max_depth, max_frontier, cutoff = [], 0, 0, 1, False
    while frontier and expanded < max_tries:
        n = frontier.pop()
        in_frontier[n.pos][n.depth] -= 1
        expanded += 1; visit.append(n.pos); max_depth = max(max_depth, n.depth)
        if n.pos == goal:
            return _result(maze, n, expanded, max_depth, max_frontier, t0, visit, cutoff=cutoff)
        if limit is not None and n.depth >= limit:
            cutoff = True
            continue
        for k in reversed(_children(maze, n)):              # đảo để hướng đầu tiên được thăm trước
            if check_cycle and _on_path(n, k.pos):
                continue
            depths = in_frontier.setdefault(k.pos, {})
            if frontier_option == 2 and any(c > 0 and d <= k.depth for d, c in depths.items()):
                continue
            frontier.append(k)
            depths[k.depth] = depths.get(k.depth, 0) + 1
        max_frontier = max(max_frontier, len(frontier))
    return _result(maze, None, expanded, max_depth, max_frontier, t0, visit,
                   cutoff=cutoff or bool(frontier))


def IDS(maze, frontier_option=1, check_cycle=True, max_tries=2_000_000, max_limit=None, vis=False, **kwargs):
    """Iterative Deepening Search: lặp Depth-Limited DFS với limit = 0, 1, 2, ...
    max_tries là giới hạn số lần mở rộng cho MỖI vòng DLS. Kết quả có thêm:
    iterations (số vòng DLS), final_limit, log (danh sách (limit, số nút mở rộng))."""
    t0 = time.perf_counter()
    total, max_front, log = 0, 0, []
    if max_limit is None:                       # đường đi đơn dài nhất không vượt quá số ô đi được
        max_limit = int((maze != "X").sum())
    for L in range(max_limit + 1):
        r = DFS(maze, limit=L, check_cycle=check_cycle, frontier_option=frontier_option, max_tries=max_tries)
        total += r["expanded"]; max_front = max(max_front, r["max_frontier"]); log.append((L, r["expanded"]))
        if r["path"] is not None or not r["cutoff"]:
            r.update(expanded=total, max_frontier=max_front, time=time.perf_counter() - t0,
                     iterations=len(log), final_limit=L, log=log)
            return r
    return dict(r, expanded=total, max_frontier=max_front, time=time.perf_counter() - t0,
                iterations=len(log), final_limit=None, log=log)


def show_path(maze, result, ax=None, title=""):
    """Vẽ mê cung với vùng đã khảo sát (xám) và đường đi (đỏ)."""
    if result is None or result.get("path") is None:
        print("No solution found!")
        mh.render(maze, explored=(result or {}).get("explored"), ax=ax, title=title)
    else:
        mh.render(maze, explored=result["explored"], path=result["path"], ax=ax, title=title)
    if ax is None:
        plt.show()


def show_maze(maze, ax=None):
    mh.show_maze(maze, ax)


def animate_maze(result, frames=60, interval=120):
    """Animation: vùng xám lớn dần theo thứ tự mở rộng, cuối cùng hiện đường đi."""
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    maze, order = result["maze"], result["visit_order"]
    steps = sorted(set(np.linspace(0, len(order), frames).astype(int)))
    fig, ax = plt.subplots(figsize=(8, 4))

    def draw(i):
        ax.clear()
        last = i == len(steps) - 1
        mh.render(maze, explored=order[:steps[i]], path=result["path"] if last else None, ax=ax,
                  title=f"đã mở rộng {steps[i]}/{len(order)} nút")

    anim = FuncAnimation(fig, draw, frames=len(steps), interval=interval)
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html

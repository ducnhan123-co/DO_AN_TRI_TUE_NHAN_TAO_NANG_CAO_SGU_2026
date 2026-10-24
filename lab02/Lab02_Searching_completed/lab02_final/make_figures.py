import matplotlib.pyplot as plt
import numpy as np
import maze_helper as mh
from collections import deque
import heapq

class Node:
    def __init__(self, pos, parent, action, cost):
        self.pos = tuple(pos)
        self.parent = parent
        self.action = action
        self.cost = cost
    def get_path_from_root(self):
        p = []
        n = self
        while n is not None:
            p.append(n)
            n = n.parent
        return list(reversed(p))

ACTIONS = [('E', (0, 1)), ('S', (1, 0)), ('W', (0, -1)), ('N', (-1, 0))]

def legal_successors(maze, pos):
    r, c = pos
    out = []
    for action, (dr, dc) in ACTIONS:
        nxt = (r + dr, c + dc)
        if 0 <= nxt[0] < maze.shape[0] and 0 <= nxt[1] < maze.shape[1] and mh.look(maze, nxt) != 'X':
            out.append((action, nxt))
    return out

def actions_from_node(node):
    return [n.action for n in node.get_path_from_root() if n.action is not None]

def plan_bfs(maze, start, goal):
    root = Node(start, None, None, 0)
    frontier = deque([root])
    reached = {root.pos}
    while frontier:
        node = frontier.popleft()
        if node.pos == goal:
            return actions_from_node(node), reached
        for action, nxt in legal_successors(maze, node.pos):
            if nxt not in reached:
                reached.add(nxt)
                frontier.append(Node(nxt, node, action, node.cost + 1))
    return None, reached

def plan_dfs(maze, start, goal):
    root = Node(start, None, None, 0)
    stack = [root]
    reached = set()
    while stack:
        node = stack.pop()
        reached.add(node.pos)
        if node.pos == goal:
            return actions_from_node(node), reached
        path_positions = {n.pos for n in node.get_path_from_root()}
        children = []
        for action, nxt in legal_successors(maze, node.pos):
            if nxt not in path_positions:
                children.append(Node(nxt, node, action, node.cost + 1))
        stack.extend(reversed(children))
    return None, reached

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def plan_astar(maze, start, goal):
    root = Node(start, None, None, 0)
    counter = 0
    h = lambda p: manhattan(p, goal)
    frontier = [(0, counter, root)]
    reached = {start}
    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node.pos == goal:
            return actions_from_node(node), reached
        for action, nxt in legal_successors(maze, node.pos):
            if nxt in reached:
                continue
            reached.add(nxt)
            counter += 1
            child = Node(nxt, node, action, node.cost + 1)
            heapq.heappush(frontier, (child.cost + h(nxt), counter, child))
    return None, reached

def plan_gbfs(maze, start, goal):
    root = Node(start, None, None, 0)
    counter = 0
    h = lambda p: manhattan(p, goal)
    frontier = [(h(start), counter, root)]
    reached = {start}
    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node.pos == goal:
            return actions_from_node(node), reached
        for action, nxt in legal_successors(maze, node.pos):
            if nxt in reached:
                continue
            reached.add(nxt)
            counter += 1
            child = Node(nxt, node, action, node.cost + 1)
            heapq.heappush(frontier, (h(nxt), counter, child))
    return None, reached

def save_result_image(filename, alg_name, solve_fn, out_img_path):
    with open(filename) as f:
        maze = mh.parse_maze(f.read())
    start = mh.find_pos(maze, 'S')
    goal = mh.find_pos(maze, 'G')
    path, reached = solve_fn(maze, start, goal)

    display_maze = np.copy(maze)
    for pos in reached:
        if display_maze[pos] == ' ':
            display_maze[pos] = '.'
    if path is not None:
        pos = start
        for action in path:
            dr, dc = dict(ACTIONS)[action]
            pos = (pos[0] + dr, pos[1] + dc)
            if display_maze[pos] not in ('S', 'G'):
                display_maze[pos] = 'P'

    cmap = plt.matplotlib.colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'lightgray', 'orange'])
    d_num = np.copy(display_maze)
    d_num[d_num == ' '] = 0
    d_num[d_num == 'X'] = 1
    d_num[d_num == 'S'] = 2
    d_num[d_num == 'G'] = 3
    d_num[d_num == 'P'] = 4
    d_num[d_num == '.'] = 5
    d_num[d_num == 'F'] = 6
    d_num = d_num.astype(int)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(d_num, cmap=cmap, norm=plt.matplotlib.colors.BoundaryNorm(list(range(cmap.N + 1)), cmap.N))
    ax.text(start[1], start[0], 'S', fontsize=12, color='white', ha='center', va='center', weight='bold')
    ax.text(goal[1], goal[0], 'G', fontsize=12, color='white', ha='center', va='center', weight='bold')
    path_len = len(path) if path else 0
    title_str = f"{filename} - {alg_name} (Cost: {path_len}, Reached: {len(reached)})"
    ax.set_title(title_str, fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=200)
    plt.close()
    print('Successfully saved:', out_img_path)

if __name__ == '__main__':
    save_result_image('small_maze.txt', 'BFS', plan_bfs, 'fig_small_bfs.png')
    save_result_image('small_maze.txt', 'DFS', plan_dfs, 'fig_small_dfs.png')
    save_result_image('medium_maze.txt', 'A*', plan_astar, 'fig_medium_astar.png')
    save_result_image('medium_maze.txt', 'GBFS', plan_gbfs, 'fig_medium_gbfs.png')
    save_result_image('open_maze.txt', 'A*', plan_astar, 'fig_open_astar.png')
    save_result_image('open_maze.txt', 'BFS', plan_bfs, 'fig_open_bfs.png')

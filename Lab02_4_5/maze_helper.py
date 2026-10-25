"""Hàm hỗ trợ đọc và hiển thị mê cung (X = tường, S = bắt đầu, G = đích, khoảng trắng = đi được)."""
import numpy as np
import matplotlib.pyplot as plt


def parse_maze(maze_str):
    """Chuyển chuỗi văn bản của file *.txt thành mảng numpy (hàng, cột) các ký tự."""
    lines = maze_str.rstrip("\n").split("\n")
    width = max(len(l) for l in lines)
    return np.array([list(l.ljust(width)) for l in lines])


def find_pos(maze, what):
    """Tìm vị trí (hàng, cột) của ký tự `what` (ví dụ 'S' hoặc 'G')."""
    r, c = np.argwhere(maze == what)[0]
    return int(r), int(c)


def render(maze, explored=None, path=None, ax=None, title=""):
    """Vẽ mê cung: đen = tường, xám = đã khảo sát, đỏ = đường đi, xanh lá = S, xanh dương = G."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    img = np.ones(maze.shape + (3,))
    img[maze == "X"] = (0.1, 0.1, 0.1)
    for (r, c) in (explored or ()):
        img[r, c] = (0.8, 0.8, 0.8)
    for (r, c) in (path or ()):
        img[r, c] = (0.9, 0.2, 0.2)
    img[find_pos(maze, "S")] = (0.1, 0.7, 0.1)
    img[find_pos(maze, "G")] = (0.1, 0.3, 0.9)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    return ax


def show_maze(maze, ax=None):
    render(maze, ax=ax)
    if ax is None:
        plt.show()

"""gui.py - Giao diện pygame giữ nguyên phong cách code cũ:
nền xanh dương, ô trống đen, quân 1 (đi trước) màu ĐỎ, quân 2 màu VÀNG, ô 100px, click chuột chọn cột."""
import time

import pygame

from c4core import COLS, ROWS, bb_to_grid

COLORS = {0: (0, 0, 0), 1: (255, 0, 0), 2: (255, 255, 0)}


class BoardView:
    def __init__(self, cell_size=100):
        self.cell = cell_size
        self.radius = int(cell_size / 2 - 5)
        pygame.init()
        self.screen = pygame.display.set_mode((COLS * cell_size, ROWS * cell_size))
        pygame.display.set_caption("Connect 4")

    def draw(self, first_bb, second_bb):
        """first_bb / second_bb: bitboard của người đi trước (đỏ) / đi sau (vàng)."""
        grid = bb_to_grid(first_bb, second_bb, 1)
        self.screen.fill((0, 0, 255))
        for r in range(ROWS):
            for c in range(COLS):
                pygame.draw.circle(
                    self.screen, COLORS[int(grid[r, c])],
                    (c * self.cell + self.cell // 2, r * self.cell + self.cell // 2), self.radius,
                )
        pygame.display.flip()

    def get_col(self):
        """Chờ người chơi click; trả về cột (hoặc None nếu đóng cửa sổ)."""
        clock = pygame.time.Clock()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return None
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    return min(ev.pos[0] // self.cell, COLS - 1)
            clock.tick(30)

    def pump(self):
        """Giữ cửa sổ phản hồi; trả về False nếu người dùng đóng cửa sổ."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
        return True

    def set_caption(self, text):
        pygame.display.set_caption(text)

    def close(self):
        pygame.quit()

"""
gui.py - Giao diện "Connect 4 - AI Strategy Arena" (pygame).

Nền tối, bảng điểm, chọn ai đi trước, đổi bàn 8x9 <-> 6x7, rê chuột thấy cột sáng + quân ma,
quân rơi có hiệu ứng, viền sáng 4 quân thắng. Dùng cho cả Người vs AI và AI vs AI (xem đấu).
File này chỉ lo giao diện; chạy chương trình bằng play.py.
"""
import os
import sys
import time

import pygame  
import pygame.gfxdraw  

from c4core import COLS, H, ROWS, bb_to_grid, drop_bit, has_won, is_full  

# ---------------------------------------------------------------- màu sắc
BG, PANEL, BORDER = (15, 19, 38), (26, 32, 56), (46, 54, 88)
HOLE, HOLE_EDGE, COL_HL = (11, 14, 30), (36, 43, 72), (38, 46, 78)
WHITE, GRAY, CYAN = (240, 243, 255), (150, 158, 185), (40, 175, 225)
ORANGE, RED = (255, 168, 28), (226, 58, 64)
BTN_ACTIVE = (84, 82, 210)
GREEN, BLUE_ST, RED_ST, GRAY_ST = (28, 125, 62), (52, 64, 138), (150, 45, 48), (84, 90, 116)


def circle(surf, color, center, r):
    x, y, r = int(center[0]), int(center[1]), int(r)
    pygame.gfxdraw.filled_circle(surf, x, y, r, color)
    pygame.gfxdraw.aacircle(surf, x, y, r, color)


def piece(surf, center, r, base):
    dark = tuple(int(v * 0.68) for v in base)
    light = tuple(min(255, int(v * 1.15 + 30)) for v in base)
    circle(surf, dark, center, r)
    circle(surf, base, center, r - 3)
    pygame.draw.circle(surf, light, (int(center[0]), int(center[1])), int(r * 0.55), 2)


def winning_cells(grid, sym):
    out = set()
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + dr * i, c + dc * i) for i in range(4)]
                if all(0 <= a < ROWS and 0 <= b < COLS and grid[a, b] == sym for a, b in cells):
                    out.update(cells)
    return out


class App:
    """Bên A (quân CAM) đi trước hoặc sau tuỳ nút bấm; bên B là quân ĐỎ.
    - Người vs AI : me=None (A là người, click chuột), opponent = AI (B).
    - AI vs AI    : me = AI A, opponent = AI B (chỉ xem)."""

    PAD, BM = 20, 14

    def __init__(self, opponent, subtitle, me=None, labels=("BẠN", "AI"), cell=0, a_first=True,
                 can_switch=None, ai_delay=0.45, headless=False):
        pygame.init()
        pygame.display.set_caption(f"Connect 4 - AI Strategy Arena (Bàn cờ {ROWS}x{COLS})")
        info = pygame.display.Info()
        if not cell:  # tự chọn cỡ ô vừa màn hình
            cell = int(min(64, max(34, (info.current_h * 0.86 - 214 - 2 * self.BM - 2 * self.PAD) / (ROWS + 0.9))))
            cell = int(min(cell, (info.current_w * 0.9 - 2 * self.BM) / COLS))
        self.cell_arg, self.cell = cell, cell
        self.opponent, self.me, self.subtitle = opponent, me, subtitle
        self.lab_a, self.lab_b = labels
        self.can_switch, self.ai_delay = can_switch, ai_delay
        self.a_first, self.headless = a_first, headless
        self.switch_to = None
        self.GH = int(cell * 0.9)
        self.panel_w = COLS * cell + 2 * self.BM
        self.panel_h = self.BM + self.GH + ROWS * cell + self.BM
        self.W = max(self.panel_w + 2 * self.PAD, 520)
        self.panel_x = (self.W - self.panel_w) // 2
        self.panel_y = 14 + 100 + 10 + 34 + 8 + 36 + 10
        self.Hh = self.panel_y + self.panel_h + self.PAD
        self.screen = pygame.display.set_mode((self.W, self.Hh))
        ff = "segoeui,arial,dejavusans,freesans"
        self.f_title = pygame.font.SysFont(ff, 30, bold=True)
        self.f_sub = pygame.font.SysFont(ff, 13, bold=True)
        self.f_lab = pygame.font.SysFont(ff, 12, bold=True)
        self.f_num = pygame.font.SysFont(ff, 26, bold=True)
        self.f_btn = pygame.font.SysFont(ff, 14, bold=True)
        self.f_st = pygame.font.SysFont(ff, 15, bold=True)
        by = 14 + 100 + 10
        inner = self.W - 2 * self.PAD
        w1 = int(inner * 0.28)
        w3 = inner - 2 * w1 - 16
        self.btn_first = pygame.Rect(self.PAD, by, w1, 34)
        self.btn_second = pygame.Rect(self.PAD + w1 + 8, by, w1, 34)
        self.btn_size = pygame.Rect(self.PAD + 2 * w1 + 16, by, w3, 34)
        self.status_rect = pygame.Rect(self.PAD, by + 34 + 8, inner, 36)
        self.score = dict(a=0, draw=0, b=0)
        self.hover = None
        self.message = None
        self.animate_on = not headless
        self.clock = pygame.time.Clock()
        self.new_game()

    # ------------------------------------------------------------ trạng thái ván
    @property
    def a_is_human(self):
        return self.me is None

    def _name(self, idx):
        return self.lab_a if idx == self.a_idx else self.lab_b

    def new_game(self):
        self.bbs = [0, 0]
        self.turn = 0
        self.a_idx = 0 if self.a_first else 1
        self.over, self.result = False, None
        self.last, self.win_cells = None, set()
        self.message = None
        self.ai_ready_at = time.time() + 0.5

    def human_turn(self):
        return not self.over and self.a_is_human and self.turn == self.a_idx

    def _row_of(self, col):
        mask = self.bbs[0] | self.bbs[1]
        height = ((mask >> (col * H)) & ((1 << ROWS) - 1)).bit_count()
        return ROWS - 1 - height

    def _apply(self, col):
        mask = self.bbs[0] | self.bbs[1]
        mv = drop_bit(mask, col)
        row = self._row_of(col)
        who = self.turn
        if self.animate_on:
            self.animate(col, row, ORANGE if who == self.a_idx else RED)
        self.bbs[who] |= mv
        self.last = (row, col)
        if has_won(self.bbs[who]):
            self.over, self.result = True, ("a" if who == self.a_idx else "b")
            self.score[self.result] += 1
            self.win_cells = winning_cells(bb_to_grid(self.bbs[0], self.bbs[1], 1), who + 1)
        elif is_full(self.bbs[0] | self.bbs[1]):
            self.over, self.result = True, "draw"
            self.score["draw"] += 1
        else:
            self.turn = 1 - who
            self.ai_ready_at = time.time() + self.ai_delay

    def human_move(self, col):
        if self.human_turn() and 0 <= col < COLS and drop_bit(self.bbs[0] | self.bbs[1], col):
            self.message = None
            self._apply(col)

    def ai_move(self):
        player = self.me if self.turn == self.a_idx else self.opponent
        me, opp = self.bbs[self.turn], self.bbs[1 - self.turn]
        a = int(player.act(me, opp))
        if not drop_bit(me | opp, a):  # phòng hờ: nước sai -> cột hợp lệ đầu tiên
            a = next(c for c in range(COLS) if drop_bit(me | opp, c))
        self._apply(a)

    def try_switch_board(self):
        nr, nc = (6, 7) if (ROWS, COLS) == (8, 9) else (8, 9)
        err = self.can_switch(nr, nc) if self.can_switch else None
        if err:
            self.message = err
            return False
        self.switch_to = (nr, nc)
        return True

    # ------------------------------------------------------------ vẽ
    def _text(self, s, font, color, pos, anchor="topleft"):
        img = font.render(s, True, color)
        r = img.get_rect(**{anchor: pos})
        self.screen.blit(img, r)
        return r

    def _cx(self, c):
        return self.panel_x + self.BM + c * self.cell + self.cell / 2

    def _cy(self, r):
        return self.panel_y + self.BM + self.GH + r * self.cell + self.cell / 2

    def _status(self):
        if self.message:
            return RED_ST, self.message, None
        if self.over:
            if self.result == "draw":
                return GRAY_ST, "HÒA!  Click để chơi ván mới", None
            win_a = self.result == "a"
            name = self.lab_a if win_a else self.lab_b
            return (GREEN if win_a else RED_ST, f"{name} THẮNG!  Click để chơi ván mới", ORANGE if win_a else RED)
        if self.human_turn():
            return GREEN, "LƯỢT CỦA BẠN - Hãy click vào cột để thả quân", ORANGE
        cur_a = self.turn == self.a_idx
        return BLUE_ST, f"{self._name(self.turn)} đang suy nghĩ...", ORANGE if cur_a else RED

    def draw(self, falling=None):
        s = self.screen
        s.fill(BG)
        hr = pygame.Rect(self.PAD, 14, self.W - 2 * self.PAD, 100)
        pygame.draw.rect(s, PANEL, hr, border_radius=14)
        pygame.draw.rect(s, BORDER, hr, 1, border_radius=14)
        self._text("CONNECT 4", self.f_title, WHITE, (hr.x + 18, hr.y + 14))
        self._text(self.subtitle, self.f_sub, CYAN, (hr.x + 18, hr.y + 52))
        for i, (lab, key, col) in enumerate(((self.lab_b, "b", RED), ("HÒA", "draw", None), (self.lab_a, "a", ORANGE))):
            cx = hr.right - 44 - i * 66
            if col:
                circle(s, col, (cx - 20, hr.y + 24), 5)
                self._text(lab, self.f_lab, GRAY, (cx - 12, hr.y + 24), "midleft")
            else:
                self._text(lab, self.f_lab, GRAY, (cx, hr.y + 24), "center")
            self._text(str(self.score[key]), self.f_num, WHITE, (cx, hr.y + 60), "center")
        la = "Bạn đi trước" if self.a_is_human else f"{self.lab_a} đi trước"
        lb = f"{self.lab_b} đi trước"
        for rect, label, active in ((self.btn_first, la, self.a_first), (self.btn_second, lb, not self.a_first)):
            pygame.draw.rect(s, BTN_ACTIVE if active else PANEL, rect, border_radius=8)
            pygame.draw.rect(s, BTN_ACTIVE if active else BORDER, rect, 1, border_radius=8)
            self._text(label, self.f_btn, WHITE if active else GRAY, rect.center, "center")
        pygame.draw.rect(s, PANEL, self.btn_size, border_radius=8)
        pygame.draw.rect(s, CYAN, self.btn_size, 1, border_radius=8)
        oth = "6x7" if (ROWS, COLS) == (8, 9) else "8x9"
        self._text(f"Bàn Cờ {ROWS}x{COLS}  (đổi sang {oth})", self.f_btn, CYAN, self.btn_size.center, "center")
        color, txt, dot = self._status()
        pygame.draw.rect(s, color, self.status_rect, border_radius=8)
        r = self._text(txt, self.f_st, WHITE, self.status_rect.center, "center")
        if dot:
            circle(s, dot, (r.x - 14, r.centery), 6)
        pr = pygame.Rect(self.panel_x, self.panel_y, self.panel_w, self.panel_h)
        pygame.draw.rect(s, PANEL, pr, border_radius=16)
        pygame.draw.rect(s, BORDER, pr, 1, border_radius=16)
        if self.hover is not None and self.human_turn():
            hl = pygame.Rect(self.panel_x + self.BM + self.hover * self.cell, self.panel_y + self.BM,
                             self.cell, self.GH + ROWS * self.cell)
            pygame.draw.rect(s, COL_HL, hl, border_radius=10)
            if drop_bit(self.bbs[0] | self.bbs[1], self.hover):
                piece(s, (self._cx(self.hover), self.panel_y + self.BM + self.GH / 2), self.cell * 0.4, ORANGE)
        grid = bb_to_grid(self.bbs[0], self.bbs[1], 1)
        rad = self.cell * 0.4
        for r_ in range(ROWS):
            for c_ in range(COLS):
                ctr = (self._cx(c_), self._cy(r_))
                v = int(grid[r_, c_])
                if v == 0:
                    circle(s, HOLE_EDGE, ctr, rad + 1)
                    circle(s, HOLE, ctr, rad)
                else:
                    piece(s, ctr, rad, ORANGE if v - 1 == self.a_idx else RED)
                    if (r_, c_) in self.win_cells:
                        pygame.draw.circle(s, WHITE, (int(ctr[0]), int(ctr[1])), int(rad), 3)
                    elif self.last == (r_, c_) and not self.over:
                        pygame.draw.circle(s, WHITE, (int(ctr[0]), int(ctr[1])), int(rad * 0.22))
        if falling:
            piece(s, (falling[0], falling[1]), rad, falling[2])
        pygame.display.flip()

    def animate(self, col, row, color):
        x = self._cx(col)
        y0, y1 = self.panel_y + self.BM + self.GH / 2, self._cy(row)
        dur = 0.16 + 0.035 * row
        t0 = time.time()
        while True:
            p = min(1.0, (time.time() - t0) / dur)
            if pygame.event.get(pygame.QUIT):
                pygame.quit()
                sys.exit(0)
            self.draw(falling=(x, y0 + (y1 - y0) * p * p, color))
            if p >= 1.0:
                return
            self.clock.tick(120)

    # ------------------------------------------------------------ vòng lặp
    def col_at(self, x):
        c = (x - self.panel_x - self.BM) // self.cell
        return int(c) if 0 <= c < COLS else None

    def run(self):
        """Chạy tới khi đóng cửa sổ. Trả về (hàng, cột) mới nếu người dùng bấm đổi bàn, ngược lại None."""
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return None
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return None
                    if ev.key in (pygame.K_r, pygame.K_SPACE, pygame.K_RETURN):
                        self.new_game()
                elif ev.type == pygame.MOUSEMOTION:
                    in_panel = self.panel_y <= ev.pos[1] <= self.panel_y + self.panel_h
                    self.hover = self.col_at(ev.pos[0]) if in_panel else None
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self.btn_first.collidepoint(ev.pos):
                        self.a_first = True
                        self.new_game()
                    elif self.btn_second.collidepoint(ev.pos):
                        self.a_first = False
                        self.new_game()
                    elif self.btn_size.collidepoint(ev.pos):
                        if self.try_switch_board():
                            return self.switch_to
                    elif self.panel_y <= ev.pos[1] <= self.panel_y + self.panel_h:
                        if self.over:
                            self.new_game()
                        else:
                            c = self.col_at(ev.pos[0])
                            if c is not None:
                                self.human_move(c)
                    elif self.status_rect.collidepoint(ev.pos) and self.over:
                        self.new_game()
            if not self.over and not self.human_turn() and time.time() >= self.ai_ready_at:
                self.draw()
                self.ai_move()
            self.draw()
            self.clock.tick(60)
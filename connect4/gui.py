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

from c4core import COLS, H, ROWS, TacticalPlayer, bb_to_grid, drop_bit, has_won, is_full  

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
                 can_switch=None, ai_delay=0.45, on_level=None, headless=False):
        pygame.init()
        pygame.display.set_caption(f"Connect 4 - AI Strategy Arena (Bàn cờ {ROWS}x{COLS})")
        info = pygame.display.Info()
        if not cell:  # tự chọn cỡ ô vừa màn hình
            cell = int(min(64, max(34, (info.current_h * 0.86 - 214 - 2 * self.BM - 2 * self.PAD) / (ROWS + 0.9))))
            cell = int(min(cell, (info.current_w * 0.9 - 2 * self.BM) / COLS))
        self.cell_arg, self.cell = cell, cell
        self.opponent, self.me, self.subtitle = opponent, me, subtitle
        self.lab_a, self.lab_b = labels
        self.can_switch, self.ai_delay, self.on_level = can_switch, ai_delay, on_level
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
        """AI đi một nước. Agent ném lỗi hoặc trả nước không hợp lệ (cột âm / ngoài bàn / đã đầy)
        => KHÔNG sập chương trình: báo trên thanh trạng thái và dùng nước dự phòng của bot Tactical."""
        player = self.me if self.turn == self.a_idx else self.opponent
        me, opp = self.bbs[self.turn], self.bbs[1 - self.turn]
        try:
            a = int(player.act(me, opp))
            if not drop_bit(me | opp, a):
                raise ValueError(f"nước không hợp lệ (cột {a})")
            self.message = None
        except Exception as e:  # noqa: BLE001 - agent ngoài có thể ném bất kỳ lỗi gì
            self.message = f"AI lỗi ({type(e).__name__}) - dùng nước dự phòng"
            a = TacticalPlayer().act(me, opp)
        self._apply(a)

    def change_level(self, n):
        """Đổi cấp độ đối thủ (phím 1-8). Cấp chưa train / lỗi nạp -> báo trên thanh trạng thái, giữ nguyên đối thủ cũ."""
        res = self.on_level(n)
        if isinstance(res, str):
            self.message = res
            return
        self.opponent, self.subtitle = res
        self.score = dict(a=0, draw=0, b=0)
        self.new_game()

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
                    elif self.on_level and pygame.K_1 <= ev.key <= pygame.K_8:
                        self.change_level(ev.key - pygame.K_0)
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


# =============================================================================================================
# ArenaApp - giao diện có MENU chọn chế độ ngay trong UI + ghi kết quả + màn hình thống kê.
# Cùng phong cách với App ở trên; dùng arena.Session (bitboard của c4core) để chơi và ghi games.jsonl.
# =============================================================================================================
SHORT = {1: "1 Random", 2: "2 Heuristic", 3: "3 Minimax", 4: "4 MCTS", 5: "5 Giám sát", 6: "6 PPO", 7: "7 AZ-100", 8: "8 AZ-800"}
GAMES_OPTS, OPEN_OPTS, BUDGET_OPTS = (1, 10, 50, 100, 200), (0, 2, 4), (0, 50, 100, 300)


def _fit(font, s, width):
    if font.render(s, True, WHITE).get_width() <= width:
        return s
    while len(s) > 1 and font.render(s + "…", True, WHITE).get_width() > width:
        s = s[:-1]
    return s + "…"


class ArenaApp:
    PAD, BM = 20, 14
    W = 760

    def __init__(self, state=None, cell=0, headless=False):
        from arena import BOARD, Recorder
        import levels
        self.levels, self.BOARD, self.rec = levels, BOARD, Recorder()
        pygame.init()
        pygame.display.set_caption(f"Connect 4 - AI Strategy Arena (Bàn cờ {BOARD})")
        info = pygame.display.Info()
        if not cell:
            cell = int(min(64, max(34, (info.current_h * 0.86 - 214 - 2 * self.BM - 2 * self.PAD) / (ROWS + 0.9))))
            cell = int(min(cell, (info.current_w * 0.9 - 2 * self.BM) / COLS))
        self.cell, self.GH = cell, int(cell * 0.9)
        self.panel_w = COLS * cell + 2 * self.BM
        self.panel_h = self.BM + self.GH + ROWS * cell + self.BM
        self.panel_x = (self.W - self.panel_w) // 2
        self.panel_y = 14 + 100 + 10 + 34 + 8 + 36 + 10
        self.Hh = max(self.panel_y + self.panel_h + self.PAD, 640)
        self.screen = pygame.display.set_mode((self.W, self.Hh))
        ff = "segoeui,arial,dejavusans,freesans"
        F = lambda n, b=True: pygame.font.SysFont(ff, n, bold=b)  # noqa: E731
        self.f_title, self.f_sub, self.f_lab, self.f_num = F(30), F(13), F(12), F(26)
        self.f_btn, self.f_st, self.f_small = F(14), F(15), F(12, False)
        st = dict(a=0, b=2, first="A", alternate=True, games=3, opening=2, budget=0)  # a=0: Người; b=2: cấp 2
        st.update(state or {})
        self.st = st
        self.screen_name, self.buttons, self.msg, self.back_to = "menu", [], None, ""
        self.session = None
        self.hover, self.paused, self.anim, self.next_at, self.ai_ready_at = None, False, None, 0.0, 0.0
        self.switch_to, self.headless = None, headless
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------ tiện ích vẽ
    def _text(self, s, font, color, pos, anchor="topleft"):
        img = font.render(s, True, color)
        r = img.get_rect(**{anchor: pos})
        self.screen.blit(img, r)
        return r

    def _button(self, rect, label, bid, active=False, locked=False, font=None, color=None):
        s = self.screen
        bg = BTN_ACTIVE if active else (PANEL if not locked else (21, 25, 44))
        pygame.draw.rect(s, bg, rect, border_radius=8)
        pygame.draw.rect(s, color or (BTN_ACTIVE if active else BORDER), rect, 1, border_radius=8)
        font = font or self.f_btn
        self._text(_fit(font, label, rect.w - 8), font, GRAY if locked else WHITE, rect.center, "center")
        self.buttons.append((rect, bid))

    def _header(self, subtitle, sub_w=None):
        s = self.screen
        hr = pygame.Rect(self.PAD, 14, self.W - 2 * self.PAD, 100)
        pygame.draw.rect(s, PANEL, hr, border_radius=14)
        pygame.draw.rect(s, BORDER, hr, 1, border_radius=14)
        self._text("CONNECT 4", self.f_title, WHITE, (hr.x + 18, hr.y + 14))
        self._text(_fit(self.f_sub, subtitle, sub_w or hr.w - 300), self.f_sub, CYAN, (hr.x + 18, hr.y + 52))
        return hr

    # ------------------------------------------------------------ mô tả lựa chọn
    def _spec(self, which):
        from arena import PlayerSpec, budget_params
        v = self.st[which]
        if v == 0:
            return PlayerSpec("human")
        return PlayerSpec("ai", v, budget_params(v, self.st["budget"]))

    def _mode_text(self):
        h = [self.st["a"] == 0, self.st["b"] == 0]
        return "Người - Người" if all(h) else "Máy - Máy" if not any(h) else "Người - Máy"

    # ------------------------------------------------------------ MENU
    def draw_menu(self):
        s, st = self.screen, self.st
        self._header(f"Chọn người chơi cho từng bên rồi bấm BẮT ĐẦU  |  bàn {self.BOARD}  |  chế độ: {self._mode_text()}", sub_w=self.W - 80)
        colw = (self.W - 2 * self.PAD - 16) // 2
        for k, (key, title, col) in enumerate((("a", "BÊN A (quân CAM)", ORANGE), ("b", "BÊN B (quân ĐỎ)", RED))):
            x0 = self.PAD + k * (colw + 16)
            circle(s, col, (x0 + 8, 134), 6)
            self._text(title, self.f_lab, WHITE, (x0 + 20, 126))
            bw, bh = (colw - 16) // 3, 40
            for i, v in enumerate(range(0, 9)):
                r = pygame.Rect(x0 + (i % 3) * (bw + 8), 148 + (i // 3) * (bh + 6), bw, bh)
                locked = v >= 1 and not self.levels.is_ready(v)
                self._button(r, "Người" if v == 0 else SHORT[v], (key, v), active=(st[key] == v), locked=locked)
        y = 148 + 3 * 46 + 10
        bw = (self.W - 2 * self.PAD - 16) // 2
        first_lab = {"A": "Bên A", "B": "Bên B"}[st["first"]]
        opts = [("board", f"Bàn cờ: {self.BOARD}  (bấm để đổi)"), ("first", f"Đi trước (ván 1): {first_lab}"),
                ("alt", f"Đổi bên mỗi ván: {'Bật' if st['alternate'] else 'Tắt'}"),
                ("games", f"Số ván (máy-máy): {GAMES_OPTS[st['games']]}"),
                ("open", f"Khai cuộc ngẫu nhiên: {OPEN_OPTS[st['opening']]} nước"),
                ("budget", "Thời gian/nước (cấp 3,4,7,8): " + ("mặc định" if not BUDGET_OPTS[st["budget"]] else f"{BUDGET_OPTS[st['budget']]} ms"))]
        for i, (bid, lab) in enumerate(opts):
            self._button(pygame.Rect(self.PAD + (i % 2) * (bw + 16), y + (i // 2) * 44, bw, 38), lab, bid)
        y += 3 * 44 + 6
        n_rec = len(self.rec.read())
        self._button(pygame.Rect(self.PAD, y, bw, 46), "BẮT ĐẦU", "start", active=True, font=self.f_st, color=GREEN)
        self._button(pygame.Rect(self.PAD + bw + 16, y, bw, 46), f"KẾT QUẢ ĐÃ GHI ({n_rec} ván)", "results", font=self.f_st)
        self._text(f"Kết quả mỗi ván được ghi vào {os.path.relpath(self.rec.path)}", self.f_small, GRAY, (self.PAD, y + 54))
        self._text("Cấp 5-8 bị khóa nếu chưa train (xem HUONG_DAN_TRAIN.md). Khai cuộc/số ván chỉ áp dụng khi máy-máy.",
                   self.f_small, GRAY, (self.PAD, y + 72))
        if self.msg:
            r = pygame.Rect(self.PAD, y + 94, self.W - 2 * self.PAD, 34)
            pygame.draw.rect(s, RED_ST, r, border_radius=8)
            self._text(_fit(self.f_st, self.msg, r.w - 16), self.f_st, WHITE, r.center, "center")

    def click_menu(self, bid):
        st = self.st
        self.msg = None
        if isinstance(bid, tuple):
            key, v = bid
            if v >= 1 and not self.levels.is_ready(v):
                self.msg = f"Chưa train cấp {v} cho bàn {self.BOARD}: xem HUONG_DAN_TRAIN.md"
                return
            st[key] = v
        elif bid == "board":
            self.switch_to = (6, 7) if (ROWS, COLS) == (8, 9) else (8, 9)
        elif bid == "first":
            st["first"] = "B" if st["first"] == "A" else "A"
        elif bid == "alt":
            st["alternate"] = not st["alternate"]
        elif bid == "games":
            st["games"] = (st["games"] + 1) % len(GAMES_OPTS)
        elif bid == "open":
            st["opening"] = (st["opening"] + 1) % len(OPEN_OPTS)
        elif bid == "budget":
            st["budget"] = (st["budget"] + 1) % len(BUDGET_OPTS)
        elif bid == "results":
            self.screen_name, self.back_to = "results", ""
        elif bid == "start":
            self.start_session()

    def start_session(self):
        from arena import Session
        a, b = self._spec("a"), self._spec("b")
        games = GAMES_OPTS[self.st["games"]] if (not a.is_human and not b.is_human) else 1
        try:
            self.session = Session(a, b, a_first=self.st["first"] == "A", games=games, alternate=self.st["alternate"],
                                   opening_plies=OPEN_OPTS[self.st["opening"]], recorder=self.rec,
                                   seed=int(time.time()) % 100000)
        except Exception as e:  # noqa: BLE001 - vd thiếu torch cho cấp 6
            self.msg = f"Không tạo được phiên chơi: {type(e).__name__}: {e}"
            self.session = None
            return
        self.screen_name, self.paused, self.anim = "game", False, None
        self.ai_ready_at = time.time() + 0.4
        self.next_at = 0.0

    # ------------------------------------------------------------ GAME
    def _fast(self):
        return self.session.games_total > 1

    def draw_game(self):
        se, s = self.session, self.screen
        sub = f"A: {se.specs['A'].label()}  vs  B: {se.specs['B'].label()}  |  ván {se.game_no}/{se.games_total}  |  {self.BOARD}"
        hr = self._header(sub)
        for i, (lab, key, col) in enumerate(((se.specs["B"].label().split(" ")[0], "B", RED), ("HÒA", "draw", None),
                                              (se.specs["A"].label().split(" ")[0], "A", ORANGE))):
            cx = hr.right - 44 - i * 66
            if col:
                circle(s, col, (cx - 20, hr.y + 24), 5)
                self._text(f"{key}", self.f_lab, GRAY, (cx - 12, hr.y + 24), "midleft")
            else:
                self._text(lab, self.f_lab, GRAY, (cx, hr.y + 24), "center")
            self._text(str(se.score[key]), self.f_num, WHITE, (cx, hr.y + 60), "center")
        by, inner = 14 + 100 + 10, self.W - 2 * self.PAD
        bw = (inner - 3 * 8) // 4
        labs = [("menu", "Menu (Esc)"), ("new", "Ván mới (R)"), ("pause", "Tiếp tục" if self.paused else "Tạm dừng (Space)"),
                ("results", "Kết quả")]
        for i, (bid, lab) in enumerate(labs):
            self._button(pygame.Rect(self.PAD + i * (bw + 8), by, bw, 34), lab, bid, active=(bid == "pause" and self.paused))
        sr = pygame.Rect(self.PAD, by + 34 + 8, inner, 36)
        color, txt = self._status()
        pygame.draw.rect(s, color, sr, border_radius=8)
        self._text(_fit(self.f_st, txt, inner - 16), self.f_st, WHITE, sr.center, "center")
        pr = pygame.Rect(self.panel_x, self.panel_y, self.panel_w, self.panel_h)
        pygame.draw.rect(s, PANEL, pr, border_radius=16)
        pygame.draw.rect(s, BORDER, pr, 1, border_radius=16)
        if self.hover is not None and se.human_turn():
            hl = pygame.Rect(self.panel_x + self.BM + self.hover * self.cell, self.panel_y + self.BM, self.cell,
                             self.GH + ROWS * self.cell)
            pygame.draw.rect(s, COL_HL, hl, border_radius=10)
            if se.legal(self.hover):
                piece(s, (self._cx(self.hover), self.panel_y + self.BM + self.GH / 2), self.cell * 0.4,
                      ORANGE if se.to_move_side == "A" else RED)
        grid = bb_to_grid(se.bbs[0], se.bbs[1], 1)
        rad = self.cell * 0.4
        wins = self._win_cells() if se.over else set()
        hide = (self.anim[1], self.anim[0]) if self.anim else None
        for r_ in range(ROWS):
            for c_ in range(COLS):
                ctr = (self._cx(c_), self._cy(r_))
                v = int(grid[r_, c_])
                if v == 0 or (r_, c_) == hide:
                    circle(s, HOLE_EDGE, ctr, rad + 1)
                    circle(s, HOLE, ctr, rad)
                else:
                    piece(s, ctr, rad, ORANGE if se.side_of_idx(v - 1) == "A" else RED)
                    if (r_, c_) in wins:
                        pygame.draw.circle(s, WHITE, (int(ctr[0]), int(ctr[1])), int(rad), 3)
                    elif se.last_move is not None and not se.over and (r_, c_) == self._last_cell():
                        pygame.draw.circle(s, WHITE, (int(ctr[0]), int(ctr[1])), int(rad * 0.22))
        if self.anim:
            col, row, color, t0, dur = self.anim
            p = min(1.0, (time.time() - t0) / dur)
            y0, y1 = self.panel_y + self.BM + self.GH / 2, self._cy(row)
            piece(s, (self._cx(col), y0 + (y1 - y0) * p * p), rad, color)

    def _last_cell(self):
        se = self.session
        col = se.last_move
        h = ((se.bbs[0] | se.bbs[1]) >> (col * H) & ((1 << ROWS) - 1)).bit_count()
        return (ROWS - h, col)

    def _win_cells(self):
        se = self.session
        if se.winner_idx is None:
            return set()
        return winning_cells(bb_to_grid(se.bbs[0], se.bbs[1], 1), se.winner_idx + 1)

    def _status(self):
        se = self.session
        if se.message:
            return RED_ST, se.message
        if self.paused:
            return GRAY_ST, "TẠM DỪNG"
        if se.over:
            t = se.result_text()
            tail = "  -  loạt đã xong" if se.finished_series else ("  -  ván kế tiếp tự động bắt đầu" if se.series_active() else "  -  click / R để chơi ván mới")
            return (GRAY_ST if t == "HÒA" else GREEN), t + tail
        if se.human_turn():
            return GREEN, "LƯỢT CỦA BẠN - click vào cột để thả quân"
        return BLUE_ST, f"{se.specs[se.to_move_side].label()} đang suy nghĩ..."

    def _cx(self, c):
        return self.panel_x + self.BM + c * self.cell + self.cell / 2

    def _cy(self, r):
        return self.panel_y + self.BM + self.GH + r * self.cell + self.cell / 2

    def _moved(self):
        """Bắt đầu hiệu ứng rơi cho nước vừa đi (nếu không ở chế độ nhanh)."""
        se = self.session
        if self.headless or self._fast():
            self.ai_ready_at = 0.0
            return
        col = se.last_move
        row = self._last_cell()[0]
        side = se.side_of_idx(1 - se.turn) if not se.over else se.side_of_idx(se.winner_idx if se.winner_idx is not None else se.turn)
        dur = 0.16 + 0.035 * row
        self.anim = (col, row, ORANGE if side == "A" else RED, time.time(), dur)
        self.ai_ready_at = time.time() + dur + 0.25

    def update_game(self):
        se, now = self.session, time.time()
        if self.anim and now - self.anim[3] >= self.anim[4]:
            self.anim = None
        if self.paused:
            return
        if se.over:
            if se.series_active() and self.next_at == 0.0:
                self.next_at = now + (0.0 if self._fast() else 1.2)
            if self.next_at and now >= self.next_at:
                self.next_at = 0.0
                se.new_game()
                self.ai_ready_at = now + 0.1
            return
        if se.poll_ai():
            self._moved()
        elif se.ai_turn() and not self.anim and now >= self.ai_ready_at:
            se.start_ai()

    def click_game(self, bid):
        se = self.session
        if bid == "menu":
            self.screen_name, self.msg = "menu", None
        elif bid == "new":
            se.new_game()
            self.next_at, self.anim = 0.0, None
        elif bid == "pause":
            self.paused = not self.paused
        elif bid == "results":
            self.screen_name = "results"
            self.back_to = "game"

    # ------------------------------------------------------------ KẾT QUẢ
    def draw_results(self):
        s = self.screen
        recs = self.rec.read()
        self._header(f"Kết quả các ván đã ghi: {len(recs)} ván  |  file: {os.path.relpath(self.rec.path)}", sub_w=self.W - 80)
        rows = self.rec.stats(recs)
        cols = [("Bàn", 44), ("Bên A", 160), ("Bên B", 160), ("Ván", 40), ("W/D/L", 70), ("Điểm A [Wilson 95%]", 150), ("ms A|B", 76)]
        x, y = self.PAD, 126
        pygame.draw.rect(s, PANEL, pygame.Rect(self.PAD, y - 4, self.W - 2 * self.PAD, 26), border_radius=6)
        for name, w in cols:
            self._text(name, self.f_lab, CYAN, (x + 4, y))
            x += w
        y += 30
        for r in rows[:14]:
            x = self.PAD
            vals = [r["board"], r["A"], r["B"], str(r["games"]), f"{r['win']}/{r['draw']}/{r['loss']}",
                    f"{r['score_A']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", f"{r['mean_ms_A']:.0f}|{r['mean_ms_B']:.0f}"]
            for (name, w), v in zip(cols, vals):
                self._text(_fit(self.f_small, v, w - 6), self.f_small, WHITE, (x + 4, y))
                x += w
            y += 24
        if not rows:
            self._text("Chưa có ván nào được ghi. Hãy chơi vài ván rồi quay lại.", self.f_btn, GRAY, (self.PAD, y))
        if len(rows) > 14:
            self._text(f"... và {len(rows) - 14} cặp khác (xem summary.csv)", self.f_small, GRAY, (self.PAD, y))
        by = self.Hh - 60
        bw = (self.W - 2 * self.PAD - 16) // 3
        self._button(pygame.Rect(self.PAD, by, bw, 40), "Quay lại", "back", active=True)
        self._button(pygame.Rect(self.PAD + bw + 8, by, bw, 40), "Xuất summary.csv", "csv")
        self._button(pygame.Rect(self.PAD + 2 * (bw + 8), by, bw, 40), "Làm mới", "refresh")
        if self.msg:
            self._text(_fit(self.f_small, self.msg, self.W - 2 * self.PAD), self.f_small, GREEN, (self.PAD, by - 22))

    def click_results(self, bid):
        if bid == "back":
            self.screen_name = "game" if (self.session and getattr(self, "back_to", "") == "game") else "menu"
            self.back_to, self.msg = "", None
        elif bid == "csv":
            self.msg = "Đã xuất: " + os.path.relpath(self.rec.export_csv())

    # ------------------------------------------------------------ vòng lặp
    def draw(self):
        self.screen.fill(BG)
        self.buttons = []
        getattr(self, f"draw_{self.screen_name}")()
        pygame.display.flip()

    def handle(self, ev):
        """Trả về True nếu cần thoát."""
        if ev.type == pygame.QUIT:
            return True
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                if self.screen_name == "menu":
                    return True
                self.screen_name, self.msg = "menu", None
            elif self.screen_name == "game":
                if ev.key == pygame.K_r:
                    self.click_game("new")
                elif ev.key == pygame.K_SPACE:
                    self.click_game("pause")
                elif pygame.K_1 <= ev.key <= pygame.K_9 and self.session.human_turn():
                    self._human(ev.key - pygame.K_1)
        elif ev.type == pygame.MOUSEMOTION and self.screen_name == "game":
            inp = self.panel_y <= ev.pos[1] <= self.panel_y + self.panel_h
            c = (ev.pos[0] - self.panel_x - self.BM) // self.cell
            self.hover = int(c) if inp and 0 <= c < COLS else None
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for rect, bid in self.buttons:
                if rect.collidepoint(ev.pos):
                    {"menu": self.click_menu, "game": self.click_game, "results": self.click_results}[self.screen_name](bid)
                    return False
            if self.screen_name == "game" and self.panel_y <= ev.pos[1] <= self.panel_y + self.panel_h:
                se = self.session
                if se.over and not se.series_active():
                    self.click_game("new")
                else:
                    c = (ev.pos[0] - self.panel_x - self.BM) // self.cell
                    if 0 <= c < COLS:
                        self._human(int(c))
        return False

    def _human(self, col):
        if not self.anim and self.session.human_move(col):
            self._moved()

    def run(self):
        """Chạy tới khi đóng cửa sổ. Trả về (state, (hàng, cột) mới) nếu người dùng bấm đổi bàn, ngược lại None."""
        while True:
            for ev in pygame.event.get():
                if self.handle(ev):
                    return None
            if self.switch_to:
                return self.st, self.switch_to
            if self.screen_name == "game":
                self.update_game()
            self.draw()
            self.clock.tick(60)

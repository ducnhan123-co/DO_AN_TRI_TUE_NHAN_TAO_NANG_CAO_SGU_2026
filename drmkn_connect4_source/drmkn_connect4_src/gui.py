import time
import math
import numpy as np
import pygame
from heuristic import HeuristicAgent
from minimax import MinimaxAgent

# Rewards aligned with Connect4 gym environment
WIN_REWARD = 10
BLOCK_REWARD = 3
DRAW_REWARD = 1
CONTINUE_REWARD = 0
ILLEGAL_MOVE_PENALTY = -10

# UI Color Palette (Modern Dark / Glassmorphism Aesthetic)
COLOR_BG_DARK = (15, 23, 42)         # Deep slate background (#0f172a)
COLOR_BG_CARD = (30, 41, 59)         # Card background (#1e293b)
COLOR_BOARD_BG = (24, 34, 53)        # Metallic navy board
COLOR_BOARD_BORDER = (59, 130, 246, 120)  # Luminous blue border
COLOR_CELL_EMPTY = (10, 15, 29)      # Deep recessed empty cell
COLOR_CELL_RIM = (40, 53, 76)        # Cell slot rim

# Player Token Colors (Red / Ruby 3D)
COLOR_RED_MAIN = (239, 68, 68)       # Bright Ruby
COLOR_RED_DARK = (185, 28, 28)       # Dark Shadow
COLOR_RED_LIGHT = (254, 202, 202)    # Gloss Highlight

# AI Token Colors (Amber / Gold 3D)
COLOR_YELLOW_MAIN = (245, 158, 11)   # Bright Amber
COLOR_YELLOW_DARK = (180, 83, 9)     # Dark Gold Shadow
COLOR_YELLOW_LIGHT = (254, 240, 138) # Gloss Highlight

COLOR_TEXT_PRIMARY = (248, 250, 252) # White
COLOR_TEXT_MUTED = (148, 163, 184)   # Slate 400
COLOR_ACCENT_CYAN = (6, 182, 212)    # Cyan glow
COLOR_ACCENT_GOLD = (250, 204, 21)   # Gold glow
COLOR_ACCENT_GREEN = (34, 197, 94)   # Green for Player turn


def get_winning_coords(grid, piece):
    """Finds and returns the 4 (row, col) coordinates that form a win."""
    rows, cols = grid.shape
    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if all(grid[r, c + i] == piece for i in range(4)):
                return [(r, c + i) for i in range(4)]
    # Vertical
    for r in range(rows - 3):
        for c in range(cols):
            if all(grid[r + i, c] == piece for i in range(4)):
                return [(r + i, c) for i in range(4)]
    # Diagonal Down-Right \
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(grid[r + i, c + i] == piece for i in range(4)):
                return [(r + i, c + i) for i in range(4)]
    # Diagonal Up-Right /
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(grid[r - i, c + i] == piece for i in range(4)):
                return [(r - i, c + i) for i in range(4)]
    return None


class Connect4GUI:
    def __init__(self, opponent_factory=None, initial_rows=8, initial_cols=9):
        """
        Giao diện đồ họa Connect 4 chuẩn kích thước 8 hàng x 9 cột.
        opponent_factory: Hàm nhận vào (rows, cols) và trả về đối tượng opponent agent tương ứng
        """
        self.opponent_factory = opponent_factory
        self.board_rows = 8
        self.board_cols = 9
        if self.opponent_factory:
            self.opponent = self.opponent_factory(self.board_rows, self.board_cols)
        else:
            self.opponent = HeuristicAgent(c=0)

        self.scores = {"player": 0, "ai": 0, "draw": 0}
        self.first_turn_player = True

        # Display config gọn gàng, vừa vặn mọi màn hình laptop (không tràn màn hình)
        self.win_width = 800
        self.win_height = 690
        self._calculate_board_geometry()

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.win_width, self.win_height))
        pygame.display.set_caption("Connect 4 - AI Strategy Arena (Bàn cờ 8x9)")
        self.clock = pygame.time.Clock()

        # Fonts cân đối cho màn hình laptop
        self.font_title = pygame.font.SysFont(["segoeui", "calibri", "arial"], 22, bold=True)
        self.font_sub = pygame.font.SysFont(["segoeui", "calibri", "arial"], 13, bold=True)
        self.font_toggle = pygame.font.SysFont(["segoeui", "calibri", "arial"], 12, bold=True)
        self.font_score = pygame.font.SysFont(["segoeui", "calibri", "arial"], 17, bold=True)
        self.font_status = pygame.font.SysFont(["segoeui", "calibri", "arial"], 13, bold=True)
        self.font_btn = pygame.font.SysFont(["segoeui", "calibri", "arial"], 14, bold=True)
        self.font_modal_title = pygame.font.SysFont(["segoeui", "calibri", "arial"], 24, bold=True)

        self.auto_play = False
        self.auto_player = HeuristicAgent(c=0)
        self.auto_reset_timer = 0.0

        self.reset_game()

    def _calculate_board_geometry(self):
        """Tính toán kích thước ô cờ và vị trí bàn cờ gọn gàng cho 8 hàng x 9 cột."""
        self.cell_size = 58
        self.radius = 24
        self.board_w = self.board_cols * self.cell_size + 30
        self.board_h = self.board_rows * self.cell_size + 24
        self.board_x = (self.win_width - self.board_w) // 2
        self.board_y = 160

    def reset_game(self):
        """Khởi động lại ván cờ với kích thước 8x9."""
        self.grid = np.zeros((self.board_rows, self.board_cols), dtype=np.int8)
        self.done = False
        self.winner = None
        self.winning_coords = None
        self.adaptive_msg = ""

        if hasattr(self.opponent, "learner") and self.opponent.learner:
            self.opponent.learner.reset_match()

        if self.first_turn_player:
            self.player_symb = 1
            self.opp_symb = 2
            self.current_turn = "player"
        else:
            self.player_symb = 2
            self.opp_symb = 1
            self.current_turn = "ai"

        self.hover_col = None
        self.animating_token = None

    def get_cell_center(self, row, col):
        x = self.board_x + 15 + col * self.cell_size + self.cell_size // 2
        y = self.board_y + 12 + row * self.cell_size + self.cell_size // 2
        return x, y

    def get_col_from_x(self, mouse_x):
        start_x = self.board_x + 15
        if start_x <= mouse_x <= start_x + self.board_cols * self.cell_size:
            col = int((mouse_x - start_x) // self.cell_size)
            if 0 <= col < self.board_cols:
                return col
        return None

    def draw_token(self, surface, x, y, piece, alpha=255):
        if piece == 1:
            col_main, col_dark, col_light = COLOR_RED_MAIN, COLOR_RED_DARK, COLOR_RED_LIGHT
        else:
            col_main, col_dark, col_light = COLOR_YELLOW_MAIN, COLOR_YELLOW_DARK, COLOR_YELLOW_LIGHT

        token_surf = pygame.Surface((self.radius * 2 + 8, self.radius * 2 + 8), pygame.SRCALPHA)
        cx, cy = self.radius + 4, self.radius + 4

        pygame.draw.circle(token_surf, (0, 0, 0, int(80 * (alpha / 255))), (cx, cy + 2), self.radius)
        pygame.draw.circle(token_surf, (*col_dark, alpha), (cx, cy), self.radius)
        pygame.draw.circle(token_surf, (*col_main, alpha), (cx, cy - 1), self.radius - 2)
        pygame.draw.circle(token_surf, (*col_dark, int(alpha * 0.7)), (cx, cy - 1), max(2, self.radius - 10), width=2)
        pygame.draw.circle(token_surf, (*col_light, int(180 * (alpha / 255))), (cx - 5, cy - 6), max(3, self.radius // 4))

        surface.blit(token_surf, (x - cx, y - cy))

    def draw_header(self):
        mouse_pos = pygame.mouse.get_pos()
        header_h = 142
        header_rect = pygame.Rect(self.board_x, 10, self.board_w, header_h)
        pygame.draw.rect(self.screen, COLOR_BG_CARD, header_rect, border_radius=14)
        pygame.draw.rect(self.screen, (51, 65, 85), header_rect, width=2, border_radius=14)

        # 1. Title & Opponent
        title_txt = self.font_title.render("CONNECT 4", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title_txt, (self.board_x + 16, 14))

        opp_name = self.opponent.__class__.__name__ if self.opponent else "Random AI"
        if opp_name == "HeuristicAgent":
            opp_name = "Heuristic AI"
        elif "PPO" in opp_name or "Ensemble" in opp_name:
            opp_name = "PPO Neural AI"

        badge_txt = self.font_sub.render(f"VS {opp_name.upper()} ({self.board_rows}x{self.board_cols})", True, COLOR_ACCENT_CYAN)
        self.screen.blit(badge_txt, (self.board_x + 18, 40))

        # 2. Scoreboard Widgets
        score_x = self.board_x + self.board_w - 240
        p_col = COLOR_RED_MAIN if self.player_symb == 1 else COLOR_YELLOW_MAIN
        pygame.draw.circle(self.screen, p_col, (score_x + 8, 32), 6)
        p_lbl = self.font_sub.render("BẠN", True, COLOR_TEXT_MUTED)
        self.screen.blit(p_lbl, (score_x + 18, 16))
        p_val = self.font_score.render(str(self.scores["player"]), True, COLOR_TEXT_PRIMARY)
        self.screen.blit(p_val, (score_x + 18, 32))

        d_lbl = self.font_sub.render("HÒA", True, COLOR_TEXT_MUTED)
        self.screen.blit(d_lbl, (score_x + 88, 16))
        d_val = self.font_score.render(str(self.scores["draw"]), True, COLOR_TEXT_PRIMARY)
        self.screen.blit(d_val, (score_x + 88, 32))

        ai_col = COLOR_YELLOW_MAIN if self.player_symb == 1 else COLOR_RED_MAIN
        pygame.draw.circle(self.screen, ai_col, (score_x + 158, 32), 6)
        ai_lbl = self.font_sub.render("AI", True, COLOR_TEXT_MUTED)
        self.screen.blit(ai_lbl, (score_x + 168, 16))
        ai_val = self.font_score.render(str(self.scores["ai"]), True, COLOR_TEXT_PRIMARY)
        self.screen.blit(ai_val, (score_x + 168, 32))

        # 3. SELECTOR ROW: Turn buttons & Board Size Badge
        btn_h = 28
        btn_w_turn = 120
        btn_player_rect = pygame.Rect(self.board_x + 14, 66, btn_w_turn, btn_h)
        btn_ai_rect = pygame.Rect(self.board_x + 14 + btn_w_turn + 8, 66, btn_w_turn, btn_h)

        # Draw Player First
        is_player_first = self.first_turn_player
        bg_pf = (16, 185, 129) if is_player_first else ((51, 65, 85) if btn_player_rect.collidepoint(mouse_pos) else (30, 41, 59))
        pygame.draw.rect(self.screen, bg_pf, btn_player_rect, border_radius=6)
        pygame.draw.rect(self.screen, (255, 255, 255) if is_player_first else (71, 85, 105), btn_player_rect, width=1, border_radius=6)
        t_pf = self.font_toggle.render("👤 Bạn đi trước" if is_player_first else "Bạn đi trước", True, (255, 255, 255) if is_player_first else (148, 163, 184))
        self.screen.blit(t_pf, (btn_player_rect.centerx - t_pf.get_width() // 2, btn_player_rect.centery - t_pf.get_height() // 2))

        # Draw AI First
        is_ai_first = not self.first_turn_player
        bg_af = (99, 102, 241) if is_ai_first else ((51, 65, 85) if btn_ai_rect.collidepoint(mouse_pos) else (30, 41, 59))
        pygame.draw.rect(self.screen, bg_af, btn_ai_rect, border_radius=6)
        pygame.draw.rect(self.screen, (255, 255, 255) if is_ai_first else (71, 85, 105), btn_ai_rect, width=1, border_radius=6)
        t_af = self.font_toggle.render("🤖 AI đi trước" if is_ai_first else "AI đi trước", True, (255, 255, 255) if is_ai_first else (148, 163, 184))
        self.screen.blit(t_af, (btn_ai_rect.centerx - t_af.get_width() // 2, btn_ai_rect.centery - t_af.get_height() // 2))

        # Adaptive Learning Status Badge
        badge_adapt_w = 135
        badge_adapt_x = self.board_x + 14 + (btn_w_turn * 2) + 16
        badge_adapt_rect = pygame.Rect(badge_adapt_x, 66, badge_adapt_w, btn_h)
        pygame.draw.rect(self.screen, (15, 23, 42), badge_adapt_rect, border_radius=6)
        pygame.draw.rect(self.screen, (16, 185, 129), badge_adapt_rect, width=1, border_radius=6)
        learned_count = 0
        if hasattr(self.opponent, "learner") and self.opponent.learner:
            learned_count = getattr(self.opponent.learner, "total_learned_count", 0)
        t_adapt = self.font_toggle.render(f"⚡ Tự học: {learned_count} bẫy", True, (52, 211, 153))
        self.screen.blit(t_adapt, (badge_adapt_rect.centerx - t_adapt.get_width() // 2, badge_adapt_rect.centery - t_adapt.get_height() // 2))

        # Board Size Info Badge (Dedicated 8x9)
        badge_w = 110
        badge_x = self.board_x + self.board_w - badge_w - 14
        badge_rect = pygame.Rect(badge_x, 66, badge_w, btn_h)
        pygame.draw.rect(self.screen, (30, 41, 59), badge_rect, border_radius=6)
        pygame.draw.rect(self.screen, (59, 130, 246), badge_rect, width=1, border_radius=6)
        t_badge = self.font_toggle.render("⊞ Cờ 8x9", True, (147, 197, 253))
        self.screen.blit(t_badge, (badge_rect.centerx - t_badge.get_width() // 2, badge_rect.centery - t_badge.get_height() // 2))

        # 4. Status Banner
        status_bar = pygame.Rect(self.board_x + 14, 102, self.board_w - 28, 30)
        if not self.done:
            if self.current_turn == "player":
                bar_col = (22, 101, 52)
                p_symb_name = "🔴 Đỏ" if self.player_symb == 1 else "🟡 Vàng"
                txt = f"👉 LƯỢT CỦA BẠN ({p_symb_name}) - Hãy click vào cột để thả quân"
                txt_col = (187, 247, 208)
            else:
                bar_col = (133, 77, 14)
                ai_symb_name = "🔴 Đỏ" if self.opp_symb == 1 else "🟡 Vàng"
                txt = f"🤖 LƯỢT CỦA AI ({ai_symb_name}) - AI đang tính nước đi..."
                txt_col = (254, 240, 138)
        else:
            if self.winner == "player":
                bar_col = (22, 101, 52)
                txt = getattr(self, "adaptive_msg", "") or "🎉 XUẤT SẮC! BẠN ĐÃ CHIẾN THẮNG!"
                txt_col = (187, 247, 208)
            elif self.winner == "ai":
                bar_col = (153, 27, 27)
                txt = getattr(self, "adaptive_msg", "") or "🤖 AI ĐÃ CHIẾN THẮNG TRẬN NÀY!"
                txt_col = (254, 202, 202)
            else:
                bar_col = (71, 85, 105)
                txt = "🤝 TRẬN ĐẤU KẾT THÚC HÒA!"
                txt_col = (241, 245, 249)

        pygame.draw.rect(self.screen, bar_col, status_bar, border_radius=6)
        st_surface = self.font_status.render(txt, True, txt_col)
        self.screen.blit(
            st_surface,
            (
                status_bar.centerx - st_surface.get_width() // 2,
                status_bar.centery - st_surface.get_height() // 2,
            ),
        )

        return {
            "btn_player_first": btn_player_rect,
            "btn_ai_first": btn_ai_rect,
        }

    def draw_board(self):
        board_rect = pygame.Rect(self.board_x, self.board_y, self.board_w, self.board_h)
        pygame.draw.rect(self.screen, COLOR_BOARD_BG, board_rect, border_radius=20)
        pygame.draw.rect(self.screen, (51, 65, 85), board_rect, width=3, border_radius=20)

        # Hover column highlight
        if not self.done and self.current_turn == "player" and self.hover_col is not None:
            col_x = self.board_x + 15 + self.hover_col * self.cell_size
            beam_surf = pygame.Surface((self.cell_size, self.board_h - 16), pygame.SRCALPHA)
            beam_surf.fill((255, 255, 255, 18))
            self.screen.blit(beam_surf, (col_x, self.board_y + 8))

        # Draw sockets & tokens
        for r in range(self.board_rows):
            for c in range(self.board_cols):
                cx, cy = self.get_cell_center(r, c)
                piece = self.grid[r, c]

                pygame.draw.circle(self.screen, COLOR_CELL_EMPTY, (cx, cy), self.radius)
                pygame.draw.circle(self.screen, COLOR_CELL_RIM, (cx, cy), self.radius, width=2)

                if piece != 0:
                    self.draw_token(self.screen, cx, cy, piece)

        # Winning line
        if self.winning_coords:
            pulse = (math.sin(pygame.time.get_ticks() * 0.008) + 1) * 0.5
            halo_radius = int(self.radius + 3 + pulse * 5)
            halo_surf = pygame.Surface((halo_radius * 2 + 10, halo_radius * 2 + 10), pygame.SRCALPHA)

            for r, c in self.winning_coords:
                cx, cy = self.get_cell_center(r, c)
                halo_surf.fill((0, 0, 0, 0))
                pygame.draw.circle(
                    halo_surf,
                    (*COLOR_ACCENT_GOLD, int(160 + pulse * 70)),
                    (halo_radius + 5, halo_radius + 5),
                    halo_radius,
                    width=3,
                )
                self.screen.blit(halo_surf, (cx - halo_radius - 5, cy - halo_radius - 5))

            p_start = self.get_cell_center(*self.winning_coords[0])
            p_end = self.get_cell_center(*self.winning_coords[-1])
            pygame.draw.line(self.screen, COLOR_ACCENT_GOLD, p_start, p_end, width=7)
            pygame.draw.line(self.screen, (255, 255, 255), p_start, p_end, width=3)

        # Dropping animation
        if self.animating_token:
            col = self.animating_token["col"]
            cur_y = self.animating_token["current_y"]
            piece = self.animating_token["piece"]
            cx, _ = self.get_cell_center(0, col)
            self.draw_token(self.screen, cx, cur_y, piece)

        # Ghost token hover
        if not self.done and self.current_turn == "player" and self.hover_col is not None and not self.animating_token:
            if self.grid[0, self.hover_col] == 0:
                cx, _ = self.get_cell_center(0, self.hover_col)
                hover_y = self.board_y - 25
                self.draw_token(self.screen, cx, hover_y, self.player_symb, alpha=175)

    def draw_footer(self):
        if getattr(self, "auto_play", False):
            pulse = int((math.sin(pygame.time.get_ticks() * 0.008) + 1) * 0.5 * 100 + 155)
            footer_txt = self.font_sub.render(
                "⚡ TỰ ĐẤU ĐANG BẬT: Hai AI đang tự chơi & tự học...  |  Nhấn [A] để tắt",
                True,
                (52, pulse, 153),
            )
        else:
            footer_txt = self.font_sub.render(
                "[R]: Ván mới  |  [T]: Đổi lượt  |  [A]: Bật AI Tự Đấu Tự Học  |  [ESC]: Thoát",
                True,
                (148, 163, 184),
            )
        self.screen.blit(
            footer_txt,
            (self.win_width // 2 - footer_txt.get_width() // 2, self.win_height - 24),
        )

    def draw_modal(self):
        if not self.done or self.animating_token is not None or getattr(self, "auto_play", False):
            return None

        backdrop = pygame.Surface((self.win_width, self.win_height), pygame.SRCALPHA)
        backdrop.fill((10, 15, 26, 185))
        self.screen.blit(backdrop, (0, 0))

        modal_w, modal_h = 500, 275
        modal_x = (self.win_width - modal_w) // 2
        modal_y = (self.win_height - modal_h) // 2
        modal_rect = pygame.Rect(modal_x, modal_y, modal_w, modal_h)

        pygame.draw.rect(self.screen, COLOR_BG_CARD, modal_rect, border_radius=20)
        pygame.draw.rect(self.screen, (51, 65, 85), modal_rect, width=2, border_radius=20)

        if self.winner == "player":
            title = "CHIẾN THẮNG!"
            title_col = (74, 222, 128)
            sub = "Bạn đã vượt qua AI với chiến thuật xuất sắc!"
        elif self.winner == "ai":
            title = "AI CHIẾN THẮNG!"
            title_col = (248, 113, 113)
            sub = "AI đã kết nối thành công 4 quân cờ liên tiếp."
        else:
            title = "KẾT QUẢ HÒA!"
            title_col = COLOR_ACCENT_GOLD
            sub = "Bàn cờ đã đầy và không bên nào giành chiến thắng."

        t_surf = self.font_modal_title.render(title, True, title_col)
        self.screen.blit(t_surf, (modal_rect.centerx - t_surf.get_width() // 2, modal_y + 24))

        s_surf = self.font_sub.render(sub, True, COLOR_TEXT_MUTED)
        self.screen.blit(s_surf, (modal_rect.centerx - s_surf.get_width() // 2, modal_y + 64))

        lbl_next = self.font_sub.render("CHỌN LƯỢT ĐI CHO VÁN TIẾP THEO:", True, (203, 213, 225))
        self.screen.blit(lbl_next, (modal_rect.centerx - lbl_next.get_width() // 2, modal_y + 102))

        mouse_pos = pygame.mouse.get_pos()
        btn_w, btn_h = 210, 42

        btn_player_start = pygame.Rect(modal_rect.centerx - btn_w - 8, modal_y + 134, btn_w, btn_h)
        btn_ai_start = pygame.Rect(modal_rect.centerx + 8, modal_y + 134, btn_w, btn_h)
        btn_exit = pygame.Rect(modal_rect.centerx - 90, modal_y + 196, 180, 36)

        col_ps = (16, 185, 129) if btn_player_start.collidepoint(mouse_pos) else (5, 150, 105)
        col_as = (99, 102, 241) if btn_ai_start.collidepoint(mouse_pos) else (79, 70, 229)
        col_ex = (100, 116, 139) if btn_exit.collidepoint(mouse_pos) else (51, 65, 85)

        pygame.draw.rect(self.screen, col_ps, btn_player_start, border_radius=10)
        pygame.draw.rect(self.screen, col_as, btn_ai_start, border_radius=10)
        pygame.draw.rect(self.screen, col_ex, btn_exit, border_radius=8)

        t_ps = self.font_btn.render("👤 Bạn đi trước (🔴)", True, (255, 255, 255))
        t_as = self.font_btn.render("🤖 AI đi trước (🔴)", True, (255, 255, 255))
        t_ex = self.font_btn.render("Thoát (ESC)", True, (226, 232, 240))

        self.screen.blit(t_ps, (btn_player_start.centerx - t_ps.get_width() // 2, btn_player_start.centery - t_ps.get_height() // 2))
        self.screen.blit(t_as, (btn_ai_start.centerx - t_as.get_width() // 2, btn_ai_start.centery - t_as.get_height() // 2))
        self.screen.blit(t_ex, (btn_exit.centerx - t_ex.get_width() // 2, btn_exit.centery - t_ex.get_height() // 2))

        return {"modal_player_start": btn_player_start, "modal_ai_start": btn_ai_start, "exit": btn_exit}

    def start_drop_animation(self, col, target_row, piece, on_complete_callback=None):
        start_y = self.board_y - 20
        _, target_y = self.get_cell_center(target_row, col)
        self.animating_token = {
            "col": col,
            "target_row": target_row,
            "current_y": start_y,
            "target_y": target_y,
            "velocity": 120.0,
            "piece": piece,
            "callback": on_complete_callback,
        }

    def update_animation(self, dt):
        if not self.animating_token:
            return

        gravity = 3600.0
        token = self.animating_token
        token["velocity"] += gravity * dt
        token["current_y"] += token["velocity"] * dt

        if token["current_y"] >= token["target_y"]:
            token["current_y"] = token["target_y"]
            if abs(token["velocity"]) > 400:
                token["velocity"] = -token["velocity"] * 0.22
            else:
                self.grid[token["target_row"], token["col"]] = token["piece"]
                cb = token.get("callback")
                self.animating_token = None
                if cb:
                    cb()

    def check_game_over(self, last_row, last_col, last_piece):
        coords = get_winning_coords(self.grid, last_piece)
        if coords:
            self.done = True
            self.winning_coords = coords
            if last_piece == self.player_symb:
                self.winner = "player"
                self.scores["player"] += 1
            else:
                self.winner = "ai"
                self.scores["ai"] += 1

            # Tự động học thích nghi tức thì (Online Continual Learning)
            if hasattr(self.opponent, "learner") and self.opponent.learner:
                res = self.opponent.learner.learn_from_match(self.winner, self.player_symb, self.opp_symb)
                cnt = res.get("total_learned", 0)
                if self.winner == "player":
                    self.adaptive_msg = f"🎉 BẠN THẮNG! 🧠 AI vừa học được bẫy này! (Tổng: {cnt} bẫy)"
                elif self.winner == "ai":
                    self.adaptive_msg = f"🤖 AI CHIẾN THẮNG! (Trí nhớ thích nghi: {cnt} bẫy)"

            return True

        if np.all(self.grid != 0):
            self.done = True
            self.winner = "draw"
            self.scores["draw"] += 1
            if hasattr(self.opponent, "learner") and self.opponent.learner:
                self.opponent.learner.learn_from_match("draw", self.player_symb, self.opp_symb)
            return True

        return False

    def trigger_ai_move(self):
        if self.done:
            return

        valid_cols = [c for c in range(self.board_cols) if self.grid[0, c] == 0]
        if not valid_cols:
            return

        ai_action = None
        if not self.opponent:
            ai_action = int(np.random.choice(valid_cols))
        else:
            try:
                pred = self.opponent.predict(self.grid)
                action = pred[0] if isinstance(pred, (tuple, list, np.ndarray)) else pred
                if int(action) in valid_cols:
                    ai_action = int(action)
            except Exception:
                pass

        if ai_action is None or ai_action not in valid_cols:
            ai_action = int(np.random.choice(valid_cols))

        target_row = self._find_target_row(ai_action)

        # Ghi nhận nước đi của AI vào bộ nhớ học
        if hasattr(self.opponent, "learner") and self.opponent.learner:
            self.opponent.learner.record_move(self.grid, self.opp_symb, ai_action)

        def on_ai_land():
            if not self.check_game_over(target_row, ai_action, self.opp_symb):
                self.current_turn = "player"

        self.start_drop_animation(ai_action, target_row, self.opp_symb, on_ai_land)

    def _find_target_row(self, col):
        for r in range(self.board_rows - 1, -1, -1):
            if self.grid[r, col] == 0:
                return r
        return None

    def handle_player_click(self, col):
        if self.done or self.current_turn != "player" or self.animating_token:
            return

        if self.grid[0, col] != 0:
            return

        target_row = self._find_target_row(col)

        # Ghi nhận nước đi của người chơi vào bộ nhớ học
        if hasattr(self.opponent, "learner") and self.opponent.learner:
            self.opponent.learner.record_move(self.grid, self.player_symb, col)

        def on_player_land():
            if not self.check_game_over(target_row, col, self.player_symb):
                self.current_turn = "ai"
                pygame.time.set_timer(pygame.USEREVENT + 1, 350, loops=1)

        self.start_drop_animation(col, target_row, self.player_symb, on_player_land)

    def set_first_turn(self, player_first):
        self.first_turn_player = player_first
        self.reset_game()
        if self.current_turn == "ai":
            pygame.time.set_timer(pygame.USEREVENT + 1, 400, loops=1)

    def run(self):
        running = True
        modal_btns = None
        header_btns = None

        if self.current_turn == "ai":
            pygame.time.set_timer(pygame.USEREVENT + 1, 450, loops=1)

        while running:
            dt = self.clock.tick(60) / 1000.0
            mouse_x, mouse_y = pygame.mouse.get_pos()

            if not self.done and self.current_turn == "player" and not self.animating_token:
                self.hover_col = self.get_col_from_x(mouse_x)
            else:
                self.hover_col = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.USEREVENT + 1:
                    if not self.done and self.current_turn == "ai" and not self.animating_token:
                        self.trigger_ai_move()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_game()
                        if self.current_turn == "ai":
                            pygame.time.set_timer(pygame.USEREVENT + 1, 400, loops=1)
                    elif event.key == pygame.K_t:
                        self.set_first_turn(not self.first_turn_player)
                    elif event.key == pygame.K_a:
                        self.auto_play = not getattr(self, "auto_play", False)
                        self.auto_reset_timer = 0.0

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.done and modal_btns:
                        if modal_btns["modal_player_start"].collidepoint(event.pos):
                            self.set_first_turn(True)
                        elif modal_btns["modal_ai_start"].collidepoint(event.pos):
                            self.set_first_turn(False)
                        elif modal_btns["exit"].collidepoint(event.pos):
                            running = False
                    else:
                        if header_btns:
                            if header_btns["btn_player_first"].collidepoint(event.pos):
                                self.set_first_turn(True)
                                continue
                            elif header_btns["btn_ai_first"].collidepoint(event.pos):
                                self.set_first_turn(False)
                                continue

                        col = self.get_col_from_x(event.pos[0])
                        if col is not None:
                            self.handle_player_click(col)

            # Chế độ AI Tự Đấu Tự Học (Auto-Play)
            if getattr(self, "auto_play", False) and not self.animating_token:
                if not self.done:
                    if self.current_turn == "player":
                        valid_cols = [c for c in range(self.board_cols) if self.grid[0, c] == 0]
                        if valid_cols:
                            try:
                                pred = self.auto_player.predict(self.grid)
                                col = pred[0] if isinstance(pred, (tuple, list, np.ndarray)) else pred
                                if int(col) in valid_cols:
                                    self.handle_player_click(int(col))
                            except Exception:
                                pass
                else:
                    self.auto_reset_timer += dt
                    if self.auto_reset_timer >= 1.0:
                        self.auto_reset_timer = 0.0
                        self.reset_game()
                        if self.current_turn == "ai":
                            pygame.time.set_timer(pygame.USEREVENT + 1, 350, loops=1)

            self.update_animation(dt)

            self.screen.fill(COLOR_BG_DARK)
            header_btns = self.draw_header()
            self.draw_board()
            self.draw_footer()

            modal_btns = self.draw_modal()

            pygame.display.flip()

        pygame.quit()

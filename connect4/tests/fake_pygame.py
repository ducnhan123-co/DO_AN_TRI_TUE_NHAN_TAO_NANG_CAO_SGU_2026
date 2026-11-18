"""Giả lập tối thiểu pygame (vẽ bằng Pillow) để kiểm thử gui.ArenaApp khi máy không có pygame.

Chỉ dùng cho kiểm thử/chụp bố cục; KHÔNG thay thế việc chạy thật với pygame (phông chữ/khử răng cưa khác).
"""
import sys
import types

from PIL import Image, ImageDraw, ImageFont

QUIT, KEYDOWN, MOUSEMOTION, MOUSEBUTTONDOWN, MOUSEWHEEL = 256, 768, 1024, 1025, 1027
K_ESCAPE, K_r, K_SPACE, K_1, K_9 = 27, ord("r"), 32, ord("1"), ord("9")
SCALED, RESIZABLE = 1, 2
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
EVENTS: list = []
LAST: dict = {}


class Rect:
    def __init__(self, x, y=None, w=None, h=None):
        if y is None:
            x, y, w, h = x
        self.x, self.y, self.w, self.h = x, y, w, h

    right = property(lambda s: s.x + s.w)
    bottom = property(lambda s: s.y + s.h)
    center = property(lambda s: (s.x + s.w / 2, s.y + s.h / 2))

    def collidepoint(self, p):
        return self.x <= p[0] < self.x + self.w and self.y <= p[1] < self.y + self.h

    def box(self):
        return [self.x, self.y, self.x + self.w, self.y + self.h]


class Text:
    def __init__(self, img):
        self.img = img

    def get_width(self):
        return self.img.size[0]

    def get_rect(self, **kw):
        w, h = self.img.size
        (k, v), = kw.items()
        x, y = v
        if k == "center":
            x, y = x - w / 2, y - h / 2
        elif k == "topright":
            x -= w
        elif k == "midleft":
            y -= h / 2
        return Rect(x, y, w, h)


class Font:
    def __init__(self, size):
        self.f = ImageFont.truetype(_FONT, int(size * 0.85))

    def render(self, s, aa, color):
        im = Image.new("RGBA", (max(1, int(self.f.getlength(s)) + 2), int(self.f.size * 1.4)), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), s, font=self.f, fill=tuple(color))
        return Text(im)


class Surface:
    def __init__(self, size):
        self.im = Image.new("RGB", size, (0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def fill(self, c):
        self.d.rectangle([0, 0, *self.im.size], fill=tuple(c))

    def blit(self, t, rect):
        self.im.paste(t.img, (int(rect.x), int(rect.y)), t.img)


def _rect(surf, color, rect, width=0, border_radius=0):
    r = rect.box()
    if width:
        surf.d.rounded_rectangle(r, border_radius, outline=tuple(color), width=width)
    else:
        surf.d.rounded_rectangle(r, border_radius, fill=tuple(color))


def _circle(surf, color, c, r, width=0):
    b = [c[0] - r, c[1] - r, c[0] + r, c[1] + r]
    if width:
        surf.d.ellipse(b, outline=tuple(color), width=width)
    else:
        surf.d.ellipse(b, fill=tuple(color))


def install():
    m = types.ModuleType("pygame")
    m.Rect = Rect
    for k in ("QUIT", "KEYDOWN", "MOUSEMOTION", "MOUSEBUTTONDOWN", "MOUSEWHEEL", "K_ESCAPE", "K_r", "K_SPACE", "K_1", "K_9",
              "SCALED", "RESIZABLE"):
        setattr(m, k, globals()[k])
    m.init = lambda: None
    m.quit = lambda: None
    m.draw = types.SimpleNamespace(rect=_rect, circle=_circle)
    m.K_RETURN = 13
    m.gfxdraw = types.SimpleNamespace(
        filled_circle=lambda surf, x, y, r, c: _circle(surf, c, (x, y), r),
        aacircle=lambda surf, x, y, r, c: None)
    sys.modules["pygame.gfxdraw"] = m.gfxdraw
    m.font = types.SimpleNamespace(SysFont=lambda name, size, bold=False: Font(size))

    def set_mode(size, flags=0):
        LAST["surface"] = Surface(size)
        return LAST["surface"]

    m.display = types.SimpleNamespace(set_caption=lambda s: None, Info=lambda: types.SimpleNamespace(current_w=1600, current_h=1000), set_mode=set_mode, flip=lambda: None)

    def get_events(t=None):
        if t is not None:
            return [e for e in EVENTS if e.type == t] and []
        ev = list(EVENTS)
        EVENTS.clear()
        return ev

    m.event = types.SimpleNamespace(get=get_events)

    class Clock:
        def tick(self, fps=0):
            return 0

    m.time = types.SimpleNamespace(Clock=Clock)
    sys.modules["pygame"] = m
    return m


class Ev:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)

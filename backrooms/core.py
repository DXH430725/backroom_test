"""
Shared engine for the Backrooms series: resolution, colour, camera, text, and the
found-footage treatment (VHS, glitch, grain, HUD, captions, archive cards).

Resolution is chosen once per process with the BR_RES environment variable
("1080" — default — or "4k"). Every pixel-sized constant in the engine and in the
episodes is written in 1080p units and multiplied by S, so the same code renders
an identical-looking film at 1920x1080 or 3840x2160.
"""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_RES = os.environ.get("BR_RES", "1080").lower()
if _RES in ("4k", "2160", "uhd"):
    W, H = 3840, 2160
elif _RES in ("720",):
    W, H = 1280, 720
else:
    W, H = 1920, 1080
S = H / 1080.0
FPS = 24


def px(v):
    """1080p pixel units -> current resolution (int)."""
    return int(round(v * S))


def focal(hfov_deg):
    return (W / 2) / math.tan(math.radians(hfov_deg / 2))


F0 = focal(65)

FONT_CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_ITALIC = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
FONT_SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ----------------------------------------------------------------------------
# maths / colour
# ----------------------------------------------------------------------------
def s2l(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def l2s(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def ramp(t, a, b):
    return float(np.clip((t - a) / (b - a), 0, 1))


def lerp(a, b, s):
    return a + (b - a) * s


def value_noise(n, octaves=5, seed=0):
    """Tileable fractal value noise in [0,1], n x n."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    amp = 1.0
    for o in range(octaves):
        g = 4 * 2 ** o
        base = rng.random((g, g)).astype(np.float32)
        big = cv2.resize(np.tile(base, (3, 3)), (n * 3, n * 3), interpolation=cv2.INTER_CUBIC)[n:2 * n, n:2 * n]
        out += amp * big
        amp *= 0.5
    out -= out.min()
    return out / out.max()


# ----------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------
class Cam:
    """yaw 0 looks along +y, positive pitch looks up, z is up."""

    def __init__(self, pos, yaw, pitch, roll=0.0, f=None):
        self.pos = np.array(pos, np.float64)
        y, p, r = math.radians(yaw), math.radians(pitch), math.radians(roll)
        Fw = np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])
        R = np.array([math.cos(y), -math.sin(y), 0.0])
        U = np.cross(R, Fw)
        self.F = Fw
        self.R = R * math.cos(r) + U * math.sin(r)
        self.U = U * math.cos(r) - R * math.sin(r)
        self.f = F0 if f is None else f

    def project(self, P):
        rel = P - self.pos
        d = rel @ self.F
        ds = np.where(d > 1e-6, d, 1e-6)
        sx = W / 2 + self.f * (rel @ self.R) / ds
        sy = H / 2 - self.f * (rel @ self.U) / ds
        return sx, sy, d

    def rays(self, step=1):
        """Unit ray directions (dx, dy, dz) for every pixel (or every `step` pixels)."""
        w, h = W // step, H // step
        X = (np.arange(w, dtype=np.float32) + 0.5) * step - W / 2
        Y = H / 2 - (np.arange(h, dtype=np.float32) + 0.5) * step
        X, Y = np.meshgrid(X, Y)
        f = self.f
        dx = self.F[0] * f + self.R[0] * X + self.U[0] * Y
        dy = self.F[1] * f + self.R[1] * X + self.U[1] * Y
        dz = self.F[2] * f + self.R[2] * X + self.U[2] * Y
        n = np.sqrt(dx * dx + dy * dy + dz * dz)
        return (dx / n).astype(np.float32), (dy / n).astype(np.float32), (dz / n).astype(np.float32)


def look_at(pos, target):
    d = np.asarray(target, float) - np.asarray(pos, float)
    yaw = math.degrees(math.atan2(d[0], d[1]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return yaw, pitch


def handheld(t, amp=1.0, seed=0.0):
    """(dyaw, dpitch, droll) degrees: layered sines that read like a person holding a camera."""
    s = seed
    dy = (0.55 * math.sin(0.61 * t + s) + 0.3 * math.sin(1.37 * t + 2 * s) + 0.12 * math.sin(3.1 * t + s)) * amp
    dp = (0.4 * math.sin(0.83 * t + 1 + s) + 0.22 * math.sin(1.91 * t + 3 * s) + 0.1 * math.sin(4.3 * t)) * amp
    dr = (0.5 * math.sin(0.47 * t + 2 + s) + 0.2 * math.sin(1.21 * t)) * amp
    return dy, dp, dr


def to_u8(img, dither=True):
    x = img * 255
    if dither:
        x = x + np.random.uniform(-0.5, 0.5, (H, W, 1)).astype(np.float32)
    return np.ascontiguousarray(np.clip(x, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------
# text (sizes and positions in 1080p units)
# ----------------------------------------------------------------------------
_text_cache = {}


def text_alpha(text, font=FONT_CJK, size=40, tracking=0.0, blur=0):
    key = (text, font, size, tracking, blur)
    if key in _text_cache:
        return _text_cache[key]
    fnt = ImageFont.truetype(font, size)
    pad = size
    if tracking:
        widths = [fnt.getlength(ch) for ch in text]
        tw = int(sum(widths) + tracking * size * (len(text) - 1))
    else:
        tw = int(fnt.getlength(text))
    th = int(size * 1.4)
    im = Image.new("L", (tw + 2 * pad, th + 2 * pad), 0)
    dr = ImageDraw.Draw(im)
    if tracking:
        x = pad
        for ch, wd in zip(text, widths):
            dr.text((x, pad), ch, font=fnt, fill=255)
            x += wd + tracking * size
    else:
        dr.text((pad, pad), text, font=fnt, fill=255)
    a = np.asarray(im, np.float32) / 255.0
    if blur:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    _text_cache[key] = a
    return a


def put_text(img, text, cx, cy, size=40, font=FONT_CJK, color=(1, 1, 1), opacity=1.0, tracking=0.0,
             shadow=0.55, glow=0.0, anchor="center"):
    """cx, cy, size are in 1080p units."""
    if opacity <= 0.002 or not text:
        return
    size = max(6, int(round(size * S)))
    cx, cy = cx * S, cy * S
    a = text_alpha(text, font, size, tracking)
    h, w = a.shape
    pad = size
    x0 = int(cx - w / 2) if anchor == "center" else int(cx - pad)
    y0 = int(cy - h / 2)
    layers = []
    if shadow:
        layers.append((text_alpha(text, font, size, tracking, blur=max(2, size // 10)), (0.0, 0.02, 0.08), shadow))
    if glow:
        layers.append((text_alpha(text, font, size, tracking, blur=max(4, size // 4)), color, -glow))
    layers.append((a, color, 1.0))
    for al, col, strength in layers:
        xa, ya = max(x0, 0), max(y0, 0)
        xb, yb = min(x0 + w, W), min(y0 + h, H)
        if xb <= xa or yb <= ya:
            continue
        sub = al[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * opacity
        region = img[ya:yb, xa:xb]
        col = np.array(col, np.float32)
        if strength < 0:
            region += sub * col * (-strength)
        else:
            region[:] = region * (1 - sub * strength) + col * sub * strength


W1, H1 = W / S, H / S  # frame size in 1080p units (1920 x 1080)


def subtitles(img, t, subs, letterbox=0.0):
    """subs: [(start, end, zh, en)]"""
    for a, b, zh, en in subs:
        if a <= t <= b:
            op = min(1, (t - a) / 0.5, (b - t) / 0.5)
            yb = H1 - 60 if letterbox > 0.5 else H1 - 175
            put_text(img, zh, W1 / 2, yb - 50, 44, FONT_CJK, (1, 1, 1), op, shadow=0.75)
            if en:
                put_text(img, en, W1 / 2, yb + 2, 28, FONT_ITALIC, (0.86, 0.9, 1.0), op * 0.9, shadow=0.7)


def draw_card(img, lines, lt, x0=260, y0=300, dy=64, size=38, cps=18, color=(0.86, 0.9, 0.95)):
    """Typewriter archive card. lines: [(text, start_seconds)]."""
    for text, start in lines:
        if not text:
            continue
        n = int(max(0, lt - start) * cps)
        if n <= 0:
            continue
        shown = text[:n]
        cursor = "▌" if n < len(text) and (lt * 3) % 1 < 0.6 else ""
        put_text(img, shown + cursor, x0, y0, size, FONT_CJK, color, 1.0, shadow=0, anchor="left")
        y0 += dy
    return y0


def card_bg():
    return np.full((H, W, 3), (0.012, 0.014, 0.02), np.float32)


def letterbox(img, amount, aspect=2.39):
    if amount <= 0:
        return img
    bar = int(amount * (H - W / aspect) / 2)
    if bar > 0:
        img[:bar] = 0
        img[H - bar:] = 0
    return img


# ----------------------------------------------------------------------------
# found-footage treatment
# ----------------------------------------------------------------------------
def vhs(img, amt, rng, t):
    if amt <= 0.01:
        return img
    s = 1 + 1.6 * amt
    small = cv2.resize(img, (int(W / s), int(H / s)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    sh = px(2 + 5 * amt)
    img[..., 0] = np.roll(img[..., 0], sh, axis=1)
    img[..., 2] = np.roll(img[..., 2], -sh // 2, axis=1)
    rows = np.arange(H)
    wob = (amt * 3.0 * S * np.sin(rows / (37.0 * S) + t * 9) * (rng.random() < 0.5)).astype(np.int64)
    band = int((t * 140 * S) % (H + px(200))) - px(100)
    wob[max(band, 0):max(min(band + px(40), H), 0)] += px(18 * amt)
    if np.any(wob):
        cols = (np.arange(W)[None, :] - wob[:, None]) % W
        img = img[rows[:, None], cols]
    scan = ((rows // max(1, px(1))) % 2 == 0)
    img[scan] *= 1 - 0.07 * amt
    for _ in range(int(rng.poisson(3 * amt))):
        y = rng.integers(0, H)
        hh = px(rng.integers(1, 4)) or 1
        x = rng.integers(0, W)
        ln = px(rng.integers(40, 500))
        img[y:y + hh, x:x + ln] = img[y:y + hh, x:x + ln] * 0.4 + 0.6 * rng.random()
    hs = px(10 * amt)
    if hs:
        img[-hs:] = np.roll(img[-hs:], px(rng.integers(10, 60)), axis=1) * 0.8 + 0.2 * rng.random((hs, W, 1))
    return img


def glitch(img, amt, rng):
    if amt <= 0.01:
        return img
    img = img.copy()
    for _ in range(int(4 + 26 * amt)):
        y = rng.integers(0, H)
        hh = px(rng.integers(2, int(10 + 90 * amt)))
        img[y:y + hh] = np.roll(img[y:y + hh], int(rng.normal(0, 160 * amt) * S), axis=1)
    sh = px(30 * amt)
    if sh:
        img[..., 0] = np.roll(img[..., 0], sh, axis=1)
        img[..., 2] = np.roll(img[..., 2], -sh, axis=1)
    for _ in range(int(12 * amt)):
        bx, by = rng.integers(0, W - px(64)), rng.integers(0, H - px(32))
        bw, bh = px(rng.integers(16, 260)), px(rng.integers(8, 90))
        blk = img[by:by + bh, bx:bx + bw]
        img[by:by + bh, bx:bx + bw] = np.floor(blk * 4) / 4 if rng.random() < 0.5 else blk[:, ::-1]
    return img


def static(rng, level=1.0):
    n = rng.random((540, 960), dtype=np.float32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_NEAREST)
    return np.repeat(n[..., None], 3, 2) * level


def grain(img, amt, rng):
    if amt <= 0:
        return img
    g = rng.standard_normal((540, 960), dtype=np.float32)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    return img + g[..., None] * amt


_VIG = None


def vignette(img, amt):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        _VIG = np.clip(1 - 0.35 * r ** 2.2, 0, 1)[..., None].astype(np.float32)
    return img * (1 - amt + amt * _VIG)


def hud(img, t, alpha, tc, level, center=None, rec=True, cam_label="CAM-02  E.BECK"):
    if alpha <= 0.01:
        return
    c = (0.95, 0.95, 0.95)
    if rec and (t % 1.0) < 0.62:
        x0, y0, sz = px(80), px(70), px(33)
        patch = img[y0:y0 + sz, x0:x0 + sz].copy()
        cv2.circle(patch, (sz // 2, sz // 2), px(11), (0.95, 0.1, 0.1), -1, cv2.LINE_AA)
        img[y0:y0 + sz, x0:x0 + sz] = img[y0:y0 + sz, x0:x0 + sz] * (1 - alpha) + patch * alpha
    put_text(img, "REC", 150, 86, 30, FONT_MONO, c, alpha, shadow=0.4)
    put_text(img, cam_label, W1 - 210, 86, 26, FONT_MONO, c, alpha * 0.9, shadow=0.4)
    put_text(img, "TC " + tc, 210, H1 - 84, 28, FONT_MONO, c, alpha, shadow=0.4)
    put_text(img, level, W1 - 160, H1 - 84, 28, FONT_MONO, c, alpha, shadow=0.4)
    if center:
        put_text(img, center, W1 / 2, H1 - 84, 28, FONT_MONO, c, alpha, shadow=0.4)
    x, y = px(1920 - 330), px(1080 - 98)
    lw = max(1, px(2))
    cv2.rectangle(img, (x, y), (x + px(44), y + px(26)), (alpha,) * 3, lw)
    cv2.rectangle(img, (x + px(44), y + px(8)), (x + px(49), y + px(18)), (alpha,) * 3, -1)
    cv2.rectangle(img, (x + px(5), y + px(5)), (x + px(17), y + px(21)), (alpha,) * 3, -1)


def fmt_tc(sec):
    sec = max(sec, 0)
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = int(sec % 60)
    fr = int((sec % 1) * 24)
    return f"{h:02d}:{m:02d}:{s:02d}:{fr:02d}"


def finish(img, rng, t, vh=0.0, gl=0.0, gr=0.02, vig=0.55, hud_args=None, lb=0.0, subs=None, fade=None):
    """Standard post chain. Returns uint8 frame."""
    img = vhs(img, vh, rng, t)
    img = glitch(img, gl, rng)
    img = grain(img, gr, rng)
    img = vignette(img, vig)
    if hud_args:
        hud(img, t, **hud_args)
    letterbox(img, lb)
    if subs:
        subtitles(img, t, subs, lb)
    if fade is not None and fade[1] > 0:
        img = img * (1 - fade[1]) + np.array(fade[0], np.float32) * fade[1]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------
# walking paths
# ----------------------------------------------------------------------------
class Path:
    """A walkable 2D polyline (world metres) with rounded corners and arc-length lookup."""

    def __init__(self, pts, rounds=3):
        P = np.asarray(pts, np.float64)
        for _ in range(rounds):  # Chaikin corner cutting, keep the end points
            if len(P) < 3:
                break
            Q = [P[0]]
            for a, b in zip(P[:-1], P[1:]):
                Q.append(0.75 * a + 0.25 * b)
                Q.append(0.25 * a + 0.75 * b)
            Q.append(P[-1])
            P = np.array(Q)
        self.P = P
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.length = float(self.s[-1])

    def at(self, s):
        s = float(np.clip(s, 0, self.length))
        x = np.interp(s, self.s, self.P[:, 0])
        y = np.interp(s, self.s, self.P[:, 1])
        return np.array([x, y])

    def yaw(self, s, ahead=1.5):
        a = self.at(s - 0.3)
        b = self.at(s + ahead)
        if np.allclose(a, b):
            a = self.at(s - ahead)
            b = self.at(s)
        d = b - a
        return math.degrees(math.atan2(d[0], d[1]))


def walk_bob(t, speed, amp=0.022):
    """vertical head bob for walking at `speed` m/s"""
    return amp * min(speed / 1.2, 1.5) * math.sin(2 * math.pi * 1.75 * t)


def unwrap_deg(a, ref):
    """bring angle a (deg) within 180 of ref"""
    return ref + ((a - ref + 180) % 360) - 180


def subs_from_lines(lines_path, tail=0.4):
    """captions straight from a voice script: [(start, end, zh, en)]"""
    import json
    spec = json.load(open(lines_path, encoding="utf-8"))
    out = []
    for ln in spec["lines"]:
        if ln.get("subtitle"):
            out.append((ln["start"], ln["start"] + ln.get("max_dur", 3) + tail, ln["subtitle"], ln.get("en", "")))
    return out


def bloom(img, thresh=0.72, strength=0.5, sigma=14.0):
    """soft glow around bright areas (display-referred)"""
    b = np.clip(img - thresh, 0, None)
    if not b.any():
        return img
    small = cv2.resize(b, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4 * S) + 0.5 * cv2.GaussianBlur(small, (0, 0), sigma * S)
    return img + cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) * strength

"""
Ep03 — LEVEL 37 "Sublimity": timeline, cameras, captions.

    python -m backrooms.render ep03_level37 --still 15,40,70,110,160
    python -m backrooms.render ep03_level37 --all [--res 4k]
"""
import math
import os

import cv2
import numpy as np

from backrooms.core import (
    W, H, W1, H1, FPS, FONT_CJK, FONT_SERIF, Cam, Path, handheld, put_text, static, fmt_tc,
    smooth, smoother, ramp, lerp, draw_card, card_bg, finish, walk_bob, subs_from_lines, px,
)
from . import scene as sc
from .scene import cc

TITLE = "LEVEL 37 —「崇高」Sublimity"
DURATION = 200.0
HERE = os.path.dirname(os.path.abspath(__file__))

T_UNDER = 10.0
T_SURF = 20.1
T_WADE = 24.0
T_HALL = 55.0
T_ECHO = 85.0
T_STAIRS = 105.0
T_DARK = 130.0
T_TUNNEL = 141.5
T_FLOAT = 150.0
T_SINK = 166.0
T_END = 181.0

SUBS = subs_from_lines(os.path.join(HERE, "voice", "lines.json"))

CARD_OPEN = [
    ("M.E.G. 视听档案馆", 0.4),
    ("录像编号 ········ 037-S", 1.4),
    ("来源 ············ 独行流浪者 · 防水摄像机", 2.4),
    ("入口 ············ 潜入 Level 58 的水中", 3.8),
    ("", 0),
    ("附注：本录像全程未出现任何实体。", 5.6),
]
CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 37「崇高」Sublimity", 0.6),
    ("生存难度 ········ 0  ·  安全 · 不稳定 · 无实体", 1.8),
    ("危险 ············ 溺水 · 深水区照明不足 · 出口稀少", 2.8),
    ("入口 ············ 潜入 Level 58 / 119 的水中 · Level 233 的瓷砖区", 3.8),
    ("出口 ············ 沉入水中 → Level 130 · 黑暗隧道 → Level 43", 4.8),
    ("基地 ············ 无。没有人能在这里定居", 5.8),
    ("", 0),
]

P_WADE = Path([(cc(16), cc(14)), (cc(16), cc(21) + 0.3), (cc(16), cc(23)), (cc(16), cc(44))])
P_HALL = Path([(cc(16), cc(44)), (cc(16), cc(75))])


def floor_at(x, y):
    i, j = sc.WORLD.cell_of(x, y)
    return sc.WORLD.fz[i, j]


def cam_under(t):
    rise = smoother(ramp(t, 10.5, 19.6))
    z = lerp(-1.75, -0.22, rise)
    z = lerp(z, 0.34, smooth(ramp(t, 19.7, 20.5)))
    pitch = lerp(66, 12, smoother(ramp(t, 12.0, 19.8)))
    pitch = lerp(pitch, -4, smooth(ramp(t, 20.0, 23.5)))
    hy, hp, hr = handheld(t, 0.9, 0.3)
    drift = 0.12 * math.sin(t * 0.7)
    bob = 0.05 * math.sin(t * 2.1) * ramp(t, 20.5, 21)
    return Cam((cc(16) + drift, cc(14) - 0.4 + 0.15 * rise, z + bob), hy * 0.6, pitch + hp, hr * 1.5)


def cam_wade(t):
    speed = 1.35
    s = speed * max(t - T_WADE - 0.3, 0)
    xy = P_WADE.at(s)
    fz = floor_at(*xy)
    climb = smooth(ramp(t, 26.8, 28.4))
    z = lerp(0.34, 1.65 + max(fz, -0.22), climb) if t < 29 else 1.65 + max(fz, -0.25)
    hy, hp, hr = handheld(t, 1.0, 1.0)
    return Cam((xy[0], xy[1], z + walk_bob(t, speed) * climb), P_WADE.yaw(s) + hy, -4 + hp, hr)


def cam_hall(t):
    walk = smooth(ramp(t, T_HALL, 78.0))
    s = walk * P_HALL.length
    xy = P_HALL.at(s)
    turn = smoother(ramp(t, 78.0, 82.0))
    yaw = 90 * turn + 16 * math.sin((t - T_HALL) * 0.28) * (1 - turn)
    pitch = -3 + 14 * smooth(ramp(t, 59, 64)) * (1 - smooth(ramp(t, 69, 74)))
    hy, hp, hr = handheld(t, 0.8, 2.0)
    return Cam((xy[0], xy[1], 1.7 + walk_bob(t, 1.3) * (1 - ramp(t, 77, 78))), yaw + hy, pitch + hp, hr)


def cam_echo(t):
    look = smoother(ramp(t, 87.5, 91.5)) * (1 - smoother(ramp(t, 98.0, 103.5)))
    pitch = lerp(-3, -38, look)
    hy, hp, hr = handheld(t, 0.6, 3.0)
    x = cc(16) + 0.3 * smooth(ramp(t, T_ECHO, 95))
    return Cam((x, cc(75) + 1.6, 1.7), 90 - 42 * look + hy, pitch + hp, hr)


def cam_stairs(t):
    walk = smooth(ramp(t, T_STAIRS, 114.0))
    x = lerp(cc(64) + 0.2, cc(65) + 0.6, walk)
    pitch = lerp(-12, -32, smooth(ramp(t, 109, 116)))
    hy, hp, hr = handheld(t, 0.7 + 0.4 * ramp(t, 116, 122), 4.0)
    push = 0.25 * smooth(ramp(t, 122, T_DARK))
    return Cam((x + push, cc(75) + 0.75, 1.7 + walk_bob(t, 1.0) * (1 - walk)), 90 + hy, pitch + hp, hr)


def cam_dark(t):
    if t < T_TUNNEL:
        walk = smooth(ramp(t, T_DARK, T_TUNNEL - 0.5))
        x = lerp(cc(80), cc(87), walk)
        y = cc(62) + 0.4
        yaw = lerp(90, 60, smooth(ramp(t, 135, 140)))
        hy, hp, hr = handheld(t, 1.0, 5.0)
        return Cam((x, y, 1.6 + walk_bob(t, 1.0)), yaw + hy, -4 + hp, hr)
    walk = smooth(ramp(t, T_TUNNEL, T_FLOAT))
    y = lerp(cc(86), cc(90), walk)
    hy, hp, hr = handheld(t, 0.9, 6.0)
    return Cam((cc(sc.TUNNEL_X) + 0.1, y, 1.4), hy, -2 + hp, hr)


def cam_float(t):
    lt = t - T_FLOAT
    sink = smoother(ramp(t, T_SINK, 176.0))
    z = lerp(0.12, -1.9, sink) + 0.02 * math.sin(t * 1.4) * (1 - sink)
    yaw = 20 + 2.2 * lt
    pitch = lerp(36, 70, sink)
    x = cc(16) + 0.4 * math.sin(lt * 0.12)
    y = cc(16) + 0.3 * math.cos(lt * 0.1)
    return Cam((x, y, z), yaw, pitch, 0.0)


def lens_drops(img, t, t0, rng_seed=37):
    """water drops clinging to the lens after surfacing: blurred refractive blobs, sliding down"""
    a = 1 - ramp(t, t0 + 0.2, t0 + 3.8)
    if a <= 0:
        return img
    rng = np.random.default_rng(rng_seed)
    out = img.copy()
    big = cv2.GaussianBlur(img, (0, 0), px(9))
    for _ in range(26):
        x = rng.uniform(0, W)
        y = rng.uniform(0, H) + (t - t0) * rng.uniform(20, 140) * S_
        r = px(rng.uniform(14, 60))
        y0, y1 = int(max(y - r, 0)), int(min(y + r, H))
        x0, x1 = int(max(x - r, 0)), int(min(x + r, W))
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = np.clip(1 - ((xx - x) ** 2 + (yy - y) ** 2) / (r * r), 0, 1) ** 0.5 * a
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - m[..., None]) + (big[y0:y1, x0:x1] * 1.08) * m[..., None]
    return out


S_ = H / 1080.0


def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 3)
    vh, gl, gr = 0.32, 0.0, 0.02
    hud_a, tc, level, center = 1.0, "", "LV-37", None
    lb, fade = 0.0, None

    if t < T_UNDER:
        img = card_bg()
        draw_card(img, CARD_OPEN, t)
        vh, gr, hud_a = 0.3, 0.03, 0.0
        fade = ((0, 0, 0), 1 - ramp(t, 0, 0.6) + ramp(t, T_UNDER - 0.8, T_UNDER))

    elif t < T_WADE:
        cam = cam_under(t)
        img, R, below = sc.render(cam, t, exposure=1.35)
        if below:
            img = cv2.GaussianBlur(img, (0, 0), px(1.6))
        img = lens_drops(img, t, T_SURF)
        tc = fmt_tc(t - T_UNDER)
        hud_a = smooth(ramp(t, 10.5, 11.5))
        fade = ((0.0, 0.05, 0.06), 1 - smooth(ramp(t, T_UNDER, T_UNDER + 1.5)))
        gl = 0.35 * math.exp(-((t - T_SURF) / 0.12) ** 2)

    elif t < T_HALL:
        cam = cam_wade(t)
        img, R, _ = sc.render(cam, t, exposure=1.35)
        tc = fmt_tc(14 + t - T_WADE)
        a = smooth(ramp(t, 31.0, 32.4)) * (1 - smooth(ramp(t, 37.2, 38.6)))
        if a > 0:
            put_text(img, "LEVEL 37", W1 / 2, H1 / 2 - 40, 118, FONT_SERIF, (0.12, 0.33, 0.42), a, tracking=0.32, shadow=0, glow=0.0)
            put_text(img, "S U B L I M I T Y", W1 / 2, H1 / 2 + 62, 34, FONT_SERIF, (0.12, 0.33, 0.42), a * 0.95, shadow=0)
            put_text(img, "崇 高", W1 / 2, H1 / 2 + 118, 34, FONT_CJK, (0.12, 0.33, 0.42), a * 0.9, shadow=0)

    elif t < T_ECHO:
        cam = cam_hall(t)
        img, R, _ = sc.render(cam, t, exposure=1.35, tmax=170.0)
        tc = fmt_tc(14 * 60 + 2 + t - T_HALL)
        gl = 0.4 * (1 - ramp(t, T_HALL, T_HALL + 0.3))

    elif t < T_STAIRS:
        cam = cam_echo(t)
        img, R, _ = sc.render(cam, t, exposure=1.35, tmax=170.0)
        tc = fmt_tc(14 * 60 + 32 + t - T_ECHO)

    elif t < T_DARK:
        cam = cam_stairs(t)
        img, R, _ = sc.render(cam, t, exposure=1.3, ripple=1.2)
        tc = fmt_tc(2 * 3600 + 8 * 60 + t - T_STAIRS)
        gl = 0.4 * (1 - ramp(t, T_STAIRS, T_STAIRS + 0.3))

    elif t < T_FLOAT:
        cam = cam_dark(t)
        img, R, _ = sc.render(cam, t, exposure=1.3)
        tc = fmt_tc(3 * 3600 + 41 * 60 + t - T_DARK)
        gl = 0.4 * (1 - ramp(t, T_DARK, T_DARK + 0.3)) + 0.4 * (1 - ramp(t, T_TUNNEL, T_TUNNEL + 0.3))
        vh = 0.45

    elif t < T_END:
        cam = cam_float(t)
        img, R, below = sc.render(cam, t, exposure=1.1)
        if below:
            img = cv2.GaussianBlur(img, (0, 0), px(1.6 + 2.5 * ramp(t, 170, 178)))
        tc = fmt_tc(6 * 3600 + 15 * 60 + t - T_FLOAT)
        gl = 0.4 * (1 - ramp(t, T_FLOAT, T_FLOAT + 0.3))
        hud_a = 1 - smooth(ramp(t, 156, 160))
        vh = lerp(0.32, 0.12, ramp(t, 156, 162))
        dk = smooth(ramp(t, 174.0, 179.5))
        fade = ((0.0, 0.03, 0.05), dk + (1 - dk) * 0) if dk > 0 else None
        if t > 179.5:
            fade = ((0.0, 0.0, 0.0), 1.0)

    else:
        lt = t - T_END
        img = card_bg()
        draw_card(img, CARD_END, lt, y0=260, size=36, dy=60)
        a = smooth(ramp(lt, 13.5, 15.0))
        if a > 0:
            put_text(img, "「难以形容地奇怪，又难以形容地完美。」", W1 / 2, 760, 52, FONT_CJK, (0.55, 0.88, 0.92), a, shadow=0, glow=0.45 * a)
        vh, gr, hud_a = 0.3, 0.03, 0.0
        fade = ((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)) + 1 - ramp(t, T_END, T_END + 0.5))

    return finish(img, rng, t, vh=vh, gl=gl, gr=gr, vig=0.5,
                  hud_args=dict(alpha=hud_a, tc=tc, level=level, center=center, cam_label="CAM-37  WATERPROOF"),
                  lb=lb, subs=SUBS, fade=fade)

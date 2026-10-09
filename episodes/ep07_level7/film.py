"""
Ep07 — LEVEL 7 "Thalassophobia": an M.E.G. deep dive, on a breath-hold, to the cave.

    python -m backrooms.render ep07_level7 --still 15,26,35,60,85,110,125,150,166
    python -m backrooms.render ep07_level7 --all [--res 4k]
"""
import math
import os

import cv2
import numpy as np

from backrooms.core import (
    W, H, W1, H1, S, FPS, F0, FONT_CJK, FONT_SERIF, FONT_MONO, Cam, handheld, look_at, put_text,
    static, fmt_tc, smooth, smoother, ramp, lerp, draw_card, card_bg, finish, subs_from_lines, px, bloom,
)
from . import scene as sc

TITLE = "LEVEL 7 —「深海恐惧」Thalassophobia"
DURATION = 200.0
HERE = os.path.dirname(os.path.abspath(__file__))

T_ROOM = 10.0
T_FLIP = 24.6
T_SPLASH = 27.6
T_SURF = 28.0
T_DIVE = 48.0
T_TWI = 72.0
T_MID = 100.0
T_THING = 113.0
T_DARK = 121.0
T_RELIGHT = 138.5
T_ABYSS = 142.0
T_CAVE = 166.0
T_END = 174.0

SUBS = subs_from_lines(os.path.join(HERE, "voice", "lines.json"))

CARD_OPEN = [
    ("M.E.G. 视听档案馆 · 深潜记录 007", 0.4),
    ("下潜员 ·········· 深潜一号 · 屏息下潜（本层空气可支持约 30 分钟）", 1.6),
    ("目标 ············ 午夜层下部 · 海底山脉上的洞穴（传闻通往 Level 8）", 3.0),
    ("", 0),
]
CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 7「深海恐惧」Thalassophobia", 0.6),
    ("生存难度 ········ 4", 1.8),
    ("分层 ············ 白昼层 · 暮光层（约 1 km）· 午夜层（约 4 km）· 深渊（7 km 以下）", 2.8),
    ("实体 ············ 「第七层的东西」（午夜层 / 深渊上层）· 「Tiny」（白昼层 / 暮光层）", 4.0),
    ("出口 ············ 海底山脉洞穴 → Level 8 · 西侧约 150 m 深处的木门 → Level 9", 5.2),
    ("", 0),
]


def depth_at(t):
    """metres below the surface (the dive is time-compressed)"""
    if t < T_DIVE:
        return 0.0
    if t < T_TWI:
        return 900 * smooth(ramp(t, T_DIVE, T_TWI)) ** 1.3
    if t < T_MID:
        return lerp(1000, 3500, ramp(t, T_TWI, T_MID))
    if t < T_ABYSS:
        return lerp(4000, 6600, ramp(t, T_MID, T_ABYSS))
    return lerp(7000, 7800, ramp(t, T_ABYSS, T_END))


def breath_left(t):
    """30-minute breath-hold, time-compressed during the descent"""
    used = 0.0 if t < T_DIVE else (t - T_DIVE) * 7.0
    return max(0.0, 30 * 60 - 40 - used)


def cam_room(t):
    hy, hp, hr = handheld(t, 0.7, 0.5)
    walk = smoother(ramp(t, 20.0, T_FLIP))
    y = lerp(-2.3, 2.2, walk)
    pitch = lerp(-10, -4, walk) + hp
    return Cam((0.25, y, 1.6), 3 + hy, pitch, hr)


def cam_fall(t):
    """after the gravity flip: falling 4.5 m from the doorway, face-down to the sea"""
    k = smooth(ramp(t, T_FLIP, T_SPLASH))
    z = lerp(4.5, 0.3, k * k)
    roll = 90 * (1 - smooth(ramp(t, T_FLIP, T_FLIP + 1.2)))
    return Cam((0.0, -2.0, z), 0, lerp(-89, -60, k), roll)


def cam_surf(t):
    lt = t - T_SURF
    bob = 0.15 * math.sin(lt * 1.1) + 0.06 * math.sin(lt * 2.7)
    yaw = 20 + 160 * smoother(ramp(t, 31.0, 41.0)) + 120 * smoother(ramp(t, 43.0, 47.5))
    pitch = 3 + 38 * smooth(ramp(t, 36.0, 38.5)) * (1 - smooth(ramp(t, 40.0, 42.5)))
    hy, hp, hr = handheld(t, 0.8, 1.5)
    return Cam((2.5, -5.0, 0.45 + bob), yaw + hy, pitch + hp, hr + 4 * math.sin(lt * 1.1))


def cam_dive(t):
    hy, hp, hr = handheld(t, 0.5, 2.5)
    pitch = lerp(-30, -55, smooth(ramp(t, T_DIVE, T_DIVE + 6)))
    pitch = lerp(pitch, -10, smooth(ramp(t, T_TWI - 4, T_TWI + 3)))
    return Cam((0, 0, -max(depth_at(t), 2.0)), 10 * math.sin(t * 0.13) + hy, pitch + hp, hr)


def cam_deep(t):
    hy, hp, hr = handheld(t, 0.5 + 0.4 * (T_THING < t < T_DARK), 3.5)
    yaw = 15 * math.sin(t * 0.21)
    pitch = -12 + 5 * math.sin(t * 0.17)
    if T_THING - 1 < t < T_DARK:
        yaw += 30 * smooth(ramp(t, T_THING, T_THING + 2.5))       # turning towards the movement
    return Cam((0, 0, -depth_at(t)), yaw + hy, pitch + hp, hr)


def cam_abyss(t):
    s = smoother(ramp(t, T_ABYSS, T_CAVE + 6))
    y = lerp(40.0, sc.CAVE[1] - 2.0, s)
    x = sc.CAVE[0] + 2.0 * math.sin(t * 0.2) * (1 - s)
    ground = float(sc.abyss_height(np.array([x]), np.array([y]))[0])
    z = max(ground + 7.0 - 4 * s, lerp(-60, sc.CAVE[2] + 1.0, smooth(ramp(t, T_CAVE - 6, T_CAVE + 3))))
    tgt = (sc.CAVE[0], sc.CAVE[1] + 4, sc.CAVE[2])
    yaw, pitch = look_at((x, y, z), tgt)
    yaw = lerp(0, yaw, smooth(ramp(t, 150, 158)))
    pitch = lerp(-22, pitch, smooth(ramp(t, 150, 158)))
    hy, hp, hr = handheld(t, 0.6, 5.5)
    return Cam((x, y, z), yaw + hy, pitch + hp, hr)


# bones and debris around the descent line, placed relative to the diver
def _field(seed, kinds, n, zr):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        out.append({"k": kinds[k % len(kinds)], "p": (rng.uniform(-9, 9), rng.uniform(4, 22), rng.uniform(*zr)),
                    "yaw": rng.uniform(0, 6.28), "s": rng.uniform(0.9, 1.6)})
    return out


TWI_OBJS = _field(71, ["fish", "human", "metal", "human", "fish"], 26, (-6, 6))
MID_OBJS = _field(72, ["weird", "human", "fish", "weird", "human"], 22, (-6, 6))


def objs_at(t, objs, t0, t1):
    """objects drift upward past the diver as she sinks (positions relative to her)"""
    rise = (t - t0) * 2.2
    d = depth_at(t)
    res = []
    for o in objs:
        p = list(o["p"])
        p[2] = ((p[2] + rise + 6) % 14) - 8 - d
        res.append(dict(o, p=tuple(p)))
    return res


def dive_hud(img, t, depth, lamp_on, alpha=1.0):
    c = (0.85, 0.95, 0.95)
    put_text(img, f"DEPTH {depth:7,.0f} m", 1600, 180, 30, FONT_MONO, c, alpha, shadow=0.5)
    b = breath_left(t)
    warn = (1.0, 0.45, 0.35) if b < 14 * 60 else c
    put_text(img, f"BREATH {int(b // 60):02d}:{int(b % 60):02d}", 1600, 230, 30, FONT_MONO, warn, alpha, shadow=0.5)
    put_text(img, "LAMP " + ("ON " if lamp_on else "OFF"), 1600, 280, 30, FONT_MONO, c if lamp_on else (0.5, 0.5, 0.5), alpha, shadow=0.5)


def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 7)
    vh, gl, gr = 0.3, 0.0, 0.025
    lb, fade = 0.0, None
    hud = dict(alpha=1.0, tc=fmt_tc(max(0, t - T_ROOM)), level="LV-7", center=None, cam_label="DIVER-1  M.E.G.")
    blm = 0.3
    depth = depth_at(t)
    lamp = 0.0

    if t < T_ROOM:
        img = card_bg()
        draw_card(img, CARD_OPEN, t)
        vh, gr, blm = 0.3, 0.03, 0
        hud = None
        fade = ((0, 0, 0), 1 - ramp(t, 0, 0.6) + ramp(t, T_ROOM - 0.8, T_ROOM))

    elif t < T_FLIP:
        cam = cam_room(t)
        img, _ = sc.room(cam, t)
        sc.room_props(img, cam)
        fade = ((0, 0, 0), 1 - ramp(t, T_ROOM, T_ROOM + 0.6))

    elif t < T_SURF:
        if t < T_SPLASH:
            cam = cam_fall(t)
            img = sc.surface(cam, t)
            gl = 0.2 * math.exp(-((t - T_FLIP) / 0.15) ** 2)
        else:
            img = np.ones((H, W, 3), np.float32) * np.array([0.75, 0.82, 0.85], np.float32)
            gl = 0.6
        vh = 0.45

    elif t < T_DIVE:
        cam = cam_surf(t)
        img = sc.surface(cam, t)
        a = smooth(ramp(t, 33.5, 34.8)) * (1 - smooth(ramp(t, 39.2, 40.4)))
        if a > 0:
            put_text(img, "LEVEL 7", W1 / 2, H1 / 2 - 40, 118, FONT_SERIF, (1, 1, 1), a, tracking=0.32, shadow=0.6, glow=0.15)
            put_text(img, "T H A L A S S O P H O B I A", W1 / 2, H1 / 2 + 62, 34, FONT_SERIF, (1, 1, 1), a * 0.95, shadow=0.6)
            put_text(img, "深 海 恐 惧", W1 / 2, H1 / 2 + 118, 34, FONT_CJK, (1, 1, 1), a * 0.9, shadow=0.6)
        # droplets clearing from the lens
        if t < T_SURF + 2.5:
            img = cv2.GaussianBlur(img, (0, 0), px(6 * (1 - ramp(t, T_SURF, T_SURF + 2.5)) + 0.1))

    elif t < T_ABYSS:
        lamp = 1.0 if t > T_TWI + 2.0 else 0.0
        if T_DARK <= t < T_RELIGHT:
            lamp = 0.0
        if t >= T_RELIGHT:
            lamp = 0.6 * smooth(ramp(t, T_RELIGHT, T_RELIGHT + 1.5))
        cam = cam_dive(t) if t < T_MID else cam_deep(t)
        col = sc.underwater(cam, t, depth, lamp)
        img = sc._up(col)
        if t < T_TWI:
            img = sc.snow(img, cam, t, depth, lamp, 0.7)
        elif t < T_MID:
            img = sc.bones(img, cam, objs_at(t, TWI_OBJS, T_TWI, T_MID), lamp, depth)
            img = sc.snow(img, cam, t, depth, lamp, 1.0)
        else:
            img = sc.bones(img, cam, objs_at(t, MID_OBJS, T_MID, T_ABYSS), lamp, depth)
            img = sc.snow(img, cam, t, depth, lamp, 1.0)
            if T_THING < t < T_DARK + 0.6:
                k = ramp(t, T_THING, T_DARK + 0.6)
                p = (lerp(-60, 30, k), 30.0, -depth - 3 + 4 * math.sin(k * 3))
                img = sc.the_thing(img, cam, p, 0.05, 22.0, max(lamp, 0.4), t)
            if T_DARK <= t < T_RELIGHT and 127.5 < t < 131.5:
                # in the dark, something enormous blots out the faint particles
                k = ramp(t, 127.5, 131.5)
                p = (lerp(30, -40, k), 9.0, -depth + 1)
                img = sc.the_thing(img, cam, p, 3.1, 26.0, 0.0, t)
        if lamp == 0.0 and t > T_TWI:
            gr = 0.05
        gl = 0.15 * math.exp(-((t - T_DARK) / 0.08) ** 2)

    elif t < T_END:
        cam = cam_abyss(t)
        lamp = 0.85
        img = sc.abyss(cam, t, lamp)
        img = sc.bubbles(img, cam, t, (sc.CAVE[0] + 14, sc.CAVE[1] - 20, sc.CAVE[2] - 10), 50, 3)
        img = sc.bubbles(img, cam, t, (sc.CAVE[0] - 18, sc.CAVE[1] - 40, -60), 40, 4)
        img = sc.snow(img, cam, t, depth, lamp, 0.8)
        dk = smooth(ramp(t, T_CAVE + 2.0, T_END - 1.0))
        fade = ((0, 0, 0), dk)
        if t > T_CAVE + 5.0:
            gl = 0.5 + 0.5 * (rng.random() < 0.5)
            st = smooth(ramp(t, T_CAVE + 5.0, T_END - 0.5)) * 0.4
            img = img * (1 - st) + static(rng) * st

    else:
        lt = t - T_END
        img = card_bg()
        draw_card(img, CARD_END, lt, y0=270, size=34, dy=58)
        vh, gr, blm = 0.3, 0.03, 0
        hud = None
        fade = ((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)) + 1 - ramp(t, T_END, T_END + 0.4))

    if hud is not None and t >= T_SURF:
        dive_hud(img, t, depth, lamp > 0.3)
    if blm:
        img = bloom(img, 0.75, blm)
    return finish(img, rng, t, vh=vh, gl=gl, gr=gr, vig=0.6, hud_args=hud, lb=lb, subs=SUBS, fade=fade)

"""
Ep02 — LEVEL 0 "Threshold": timeline, cameras, captions.

    python -m backrooms.render ep02_level0 --still 30,80,160
    python -m backrooms.render ep02_level0 --all [--res 4k]
"""
import math
import os

import cv2
import numpy as np

from backrooms.core import (
    W, H, W1, H1, FPS, F0, FONT_CJK, FONT_SERIF, Cam, Path, handheld, look_at, put_text, static,
    fmt_tc, smooth, smoother, ramp, lerp, draw_card, card_bg, finish, walk_bob, subs_from_lines,
    unwrap_deg, px,
)
from . import scene as sc
from .scene import cc

TITLE = "LEVEL 0 —「门槛」Threshold"
DURATION = 205.0
HERE = os.path.dirname(os.path.abspath(__file__))

T_WAKE = 10.0
T_MAZE = 20.0
T_SHIFT = 52.0
T_PILL = 72.0
T_HOLE = 90.0
T_CORR = 102.0
T_BLACK = 118.0
T_RED = 138.0
T_ARCH = 156.0
T_NOCLIP = 179.5
T_L1 = 182.0
T_END = 187.5

SUBS = subs_from_lines(os.path.join(HERE, "voice", "lines.json"))

CARD_OPEN = [
    ("M.E.G. 视听档案馆", 0.4),
    ("录像编号 ········ 000-A", 1.4),
    ("来源 ············ 身份不明的流浪者 · 手持摄像机", 2.4),
    ("回收地点 ········ 基地阿尔法 外围", 3.9),
    ("", 0),
    ("附注：拍摄者在录像开始前「掉出了现实」。", 5.6),
]
CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 0「门槛」Threshold", 0.6),
    ("生存难度 ········ 1", 1.8),
    ("特征 ············ 外围位移 · 隔离效应 · 停电区 · 红房间", 2.8),
    ("入口 ············ 从现实中「掉出来」", 4.0),
    ("出口 ············ 穿过一面闪烁的墙 → Level 1", 5.0),
    ("实体 ············ 未确认（有报告称走廊中有黑色人影）", 6.0),
    ("", 0),
    ("隔离效应：同时进入的人无法找到彼此。", 7.6),
]


def P(cells):
    return Path([(cc(x), cc(y)) for x, y in cells])


PATH_MAZE = P(sc.PATH_MAZE)
PATH_BLACK = P(sc.PATH_BLACK)


def cam_wake(t):
    lt = t - T_WAKE
    up = smoother(ramp(t, 13.5, 18.8))
    z = lerp(0.18, 1.6, up)
    pitch = lerp(72, -8, smoother(ramp(t, 12.5, 16.5)))
    yaw = lerp(-60, 0, smoother(ramp(t, 13.0, 19.0)))
    roll = lerp(-28, 0, smoother(ramp(t, 12.5, 17.5)))
    hy, hp, hr = handheld(t, 1.4, 0.5)
    pos = (cc(5) + 0.4 * (1 - up), cc(5) - 0.8 * (1 - up), z)
    return Cam(pos, yaw + hy, pitch + hp, roll + hr)


def cam_maze(t):
    speed = PATH_MAZE.length / (T_SHIFT - T_MAZE - 0.4)
    s = speed * max(t - T_MAZE - 0.2, 0)
    xy = PATH_MAZE.at(s)
    hy, hp, hr = handheld(t, 1.0, 1.0)
    look = 14 * math.sin((t - T_MAZE) * 0.35)
    return Cam((xy[0], xy[1], 1.6 + walk_bob(t, speed)), PATH_MAZE.yaw(s, 2.2) + look * 0.4 + hy, -3 + hp, hr)


def cam_shift(t):
    y0, y1 = cc(20), cc(25)
    walk = smooth(ramp(t, T_SHIFT - 0.2, 56.0))
    y = lerp(y0, y1, walk)
    x = cc(15)
    turn1 = smoother(ramp(t, 56.3, 58.3))
    turn2 = smoother(ramp(t, 61.5, 63.5))
    turn3 = smoother(ramp(t, 65.6, 67.0))
    yaw = 180 * turn1 + 180 * turn2 + 90 * turn3
    out = smooth(ramp(t, 67.0, T_PILL))
    x = lerp(x, cc(19), out)
    stare = 0.25 * smooth(ramp(t, 58.3, 61.4)) * (1 - turn2)
    y -= stare
    hy, hp, hr = handheld(t, 1.0 + 0.8 * turn1 * (1 - turn2), 2.0)
    bob = walk_bob(t, 1.4) * (1 - smooth(ramp(t, 55.5, 56.2)))
    return Cam((x, y, 1.6 + bob), yaw + hy, -2 + hp, hr)


def shift_state(t):
    if t < 57.3:
        return 0
    if t < 62.5:
        return 1
    return 2


def cam_pill(t):
    lt = t - T_PILL
    y = cc(30) + 0.85 * lt
    hy, hp, hr = handheld(t, 0.7, 3.0)
    yaw = -28 + 50 * smoother(lt / 18) + hy
    return Cam((cc(202) + 0.4, y, 1.6 + walk_bob(t, 0.85)), yaw, -1.5 + 2 * smooth(ramp(t, 80, 88)) + hp, hr)


def cam_hole(t):
    walk = smoother(ramp(t, T_HOLE, 97.0))
    y = lerp(cc(9), cc(16) - 0.5, walk)
    x = lerp(cc(330), cc(330) + 0.25, smooth(ramp(t, 97, 100)))
    look = smoother(ramp(t, 96.5, 99.5))
    yaw = lerp(0, 72, look)
    pitch = lerp(-4, -62, look)
    hy, hp, hr = handheld(t, 0.8 + 0.6 * look, 4.0)
    return Cam((x, y, 1.6 + walk_bob(t, 1.4) * (1 - walk)), yaw + hy, pitch + hp, hr)


def cam_corr(t):
    lt = t - T_CORR
    y = cc(8) + 0.55 * min(lt, 7.0) + 0.1 * max(lt - 7, 0)
    hy, hp, hr = handheld(t, 0.6 + 0.6 * ramp(t, 108, 112), 5.0)
    return Cam((cc(sc.CORR_X), y, 1.6 + walk_bob(t, 0.55) * (1 - ramp(t, 108, 109))), hy, -1 + hp, hr)


FIGURE_Y = cc(34)


def cam_black(t):
    speed = 0.95
    s = speed * max(t - T_BLACK - 0.5, 0)
    xy = PATH_BLACK.at(s)
    hy, hp, hr = handheld(t, 1.2, 6.0)
    return Cam((xy[0], xy[1], 1.55 + walk_bob(t, speed)), PATH_BLACK.yaw(s, 2.0) + hy, -7 + hp, hr)


def cam_red(t):
    x = cc(58)
    fwd = smooth(ramp(t, T_RED, 147.0))
    y = lerp(cc(46), cc(53), fwd)
    whip = smoother(ramp(t, 147.3, 147.9))
    run = smooth(ramp(t, 148.0, 153.5))
    y = lerp(y, cc(43), run)
    shake = 1.0 + 3.5 * whip * (1 - ramp(t, 153.5, 156))
    hy, hp, hr = handheld(t * (1 + 1.5 * whip), shake, 8.0)
    bob = walk_bob(t, 1.0 + 2.5 * run * (1 - ramp(t, 153, 154)), 0.03)
    return Cam((x, y, 1.6 + bob), 180 * whip + hy, -3 + hp, hr)


ARCH_Y0 = cc(sc.ARCH_Y0)
WALL_Y = (sc.ARCH_Y1 + 1) * sc.C


def arch_y(t):
    a = lerp(ARCH_Y0, ARCH_Y0 + 25.0, smoother(ramp(t, T_ARCH, 172.0)) * 0.98 + 0.02 * ramp(t, T_ARCH, 172))
    return lerp(a, WALL_Y - 0.05, smoothing_in(t))


def smoothing_in(t):
    return smooth(ramp(t, 172.0, T_NOCLIP + 0.4))


def cam_arch(t):
    y = arch_y(t)
    hy, hp, hr = handheld(t, 0.5, 9.0)
    look = 10 * math.sin((t - T_ARCH) * 0.3) * (1 - ramp(t, 168, 172))
    return Cam((cc(sc.ARCH_X), y, 1.6 + walk_bob(t, 1.4) * 0.6), look + hy, 0 + hp, hr)


def cam_l1(t):
    lt = t - T_L1
    hy, hp, hr = handheld(t, 0.9, 11.0)
    return Cam((22.0, 20.0 + 0.9 * lt, 1.65 + walk_bob(t, 0.9)), 25 + 6 * lt + hy, 3 + hp, hr)


def corr_flicker(t):
    r = np.random.default_rng(int(t * 9)).random()
    base = 1.0 - 0.75 * (r < 0.12 + 0.3 * ramp(t, 108, 113))
    if 113.0 < t < 113.5:
        return 0.0
    return base


def draw_figure(img_lin_srgb, cam, R, dist_y, alpha=1.0):
    """a tall dark silhouette standing at the end of the hall, depth-tested against the trace"""
    pts = np.array([
        (cc(sc.CORR_X) - 0.12, dist_y, 0.0), (cc(sc.CORR_X) - 0.1, dist_y, 1.0),
        (cc(sc.CORR_X) + 0.12, dist_y, 0.0), (cc(sc.CORR_X) + 0.1, dist_y, 1.0),
        (cc(sc.CORR_X), dist_y, 1.0), (cc(sc.CORR_X), dist_y, 1.65), (cc(sc.CORR_X), dist_y, 1.88),
        (cc(sc.CORR_X) - 0.24, dist_y, 1.62), (cc(sc.CORR_X) + 0.24, dist_y, 1.62),
        (cc(sc.CORR_X) - 0.3, dist_y, 0.82), (cc(sc.CORR_X) + 0.3, dist_y, 0.82),
    ])
    qx, qy, qd = cam.project(pts)
    if (qd < 0.5).any():
        return img_lin_srgb
    m = np.zeros((H, W), np.uint8)
    s = cam.f / qd[0]
    L = cv2.LINE_AA
    p = lambda k: (int(qx[k] * 8), int(qy[k] * 8))
    th = lambda r: max(1, int(r * s))
    cv2.line(m, p(0), p(1), 255, th(0.14), L, 3)
    cv2.line(m, p(2), p(3), 255, th(0.14), L, 3)
    cv2.line(m, p(4), p(5), 255, th(0.44), L, 3)
    cv2.line(m, p(7), p(9), 255, th(0.1), L, 3)
    cv2.line(m, p(8), p(10), 255, th(0.1), L, 3)
    cv2.circle(m, p(6), int(0.12 * s * 8), 255, -1, L, 3)
    depth = cv2.resize(R["t"].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    a = (m.astype(np.float32) / 255.0) * (depth > qd[0] - 0.3) * alpha
    return img_lin_srgb * (1 - a[..., None] * 0.93)


def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 2)
    vh, gl, gr = 0.45, 0.0, 0.025
    hud_a, tc, level, center = 1.0, "", "LV-0", None
    lb, fade = 0.0, None
    tc_base = 0.0

    if t < T_WAKE:
        img = card_bg()
        draw_card(img, CARD_OPEN, t)
        vh, gr, hud_a = 0.3, 0.03, 0.0
        fade = ((0, 0, 0), 1 - ramp(t, 0, 0.6) + ramp(t, T_WAKE - 0.8, T_WAKE))

    elif t < T_MAZE:
        cam = cam_wake(t)
        sc.set_shift(0)
        img, R = sc.render_world(cam, t)
        # eyes opening: blinks, blur
        blink = 1 - smooth(ramp(t, 10.3, 11.0)) + 0.9 * math.exp(-((t - 11.8) / 0.18) ** 2) + 0.8 * math.exp(-((t - 12.9) / 0.15) ** 2)
        blur = 1 + 14 * (1 - smooth(ramp(t, 10.5, 15.5)))
        if blur > 1.3:
            k = px(blur)
            img = cv2.GaussianBlur(img, (0, 0), k)
        fade = ((0, 0, 0), float(np.clip(blink, 0, 1)))
        vh, gr = 0.55, 0.03
        hud_a = smooth(ramp(t, 14, 15))
        tc = fmt_tc(t - T_WAKE)

    elif t < T_SHIFT:
        cam = cam_maze(t)
        sc.set_shift(0)
        img, R = sc.render_world(cam, t, lights=0.94 + 0.06 * (rng.random() > 0.08))
        tc = fmt_tc(14 + t - T_MAZE)
        a = smooth(ramp(t, 27.6, 29.0)) * (1 - smooth(ramp(t, 33.6, 35.0)))
        if a > 0:
            put_text(img, "LEVEL 0", W1 / 2, H1 / 2 - 40, 118, FONT_SERIF, (1, 0.97, 0.85), a, tracking=0.32, shadow=0.6, glow=0.25)
            put_text(img, "T H R E S H O L D", W1 / 2, H1 / 2 + 62, 34, FONT_SERIF, (1, 0.97, 0.85), a * 0.95, shadow=0.6)
            put_text(img, "门 槛", W1 / 2, H1 / 2 + 118, 34, FONT_CJK, (1, 0.97, 0.85), a * 0.9, shadow=0.6)

    elif t < T_PILL:
        cam = cam_shift(t)
        sc.set_shift(shift_state(t))
        img, R = sc.render_world(cam, t)
        tc = fmt_tc(46 + t - T_SHIFT)
        gl = 0.18 * math.exp(-((t - 57.3) / 0.08) ** 2) + 0.18 * math.exp(-((t - 62.5) / 0.08) ** 2)

    elif t < T_HOLE:
        cam = cam_pill(t)
        img, R = sc.render_world(cam, t, tmax=130.0)
        tc = fmt_tc(2 * 3600 + 17 * 60 + t - T_PILL)
        gl = 0.5 * (1 - ramp(t, T_PILL, T_PILL + 0.3))

    elif t < T_CORR:
        cam = cam_hole(t)
        img, R = sc.render_world(cam, t)
        tc = fmt_tc(5 * 3600 + 2 * 60 + t - T_HOLE)
        gl = 0.5 * (1 - ramp(t, T_HOLE, T_HOLE + 0.3))

    elif t < T_BLACK:
        cam = cam_corr(t)
        fl = corr_flicker(t)
        img, R = sc.render_world(cam, t, corr_lights=fl)
        visible = t < 113.0 and fl > 0.5
        if visible:
            img = draw_figure(img, cam, R, FIGURE_Y)
        tc = fmt_tc(9 * 3600 + 40 * 60 + t - T_CORR)
        gl = 0.5 * (1 - ramp(t, T_CORR, T_CORR + 0.3)) + 0.25 * (fl < 0.5)

    elif t < T_RED:
        cam = cam_black(t)
        die = ramp(t, 118.6, 121.8)
        dl = (1 - die) * (0.4 + 0.6 * (rng.random() > 0.3 * die)) if die < 1 else 0.0
        flash = smooth(ramp(t, 122.3, 122.5))
        img, R = sc.render_world(cam, t, dark_lights=dl, flash=flash)
        tc = fmt_tc(11 * 3600 + 3 * 60 + t - T_BLACK)
        gl = 0.5 * (1 - ramp(t, T_BLACK, T_BLACK + 0.3))
        center = "LIGHT ON" if flash > 0.5 and t < 125 else None
        vh = 0.55

    elif t < T_ARCH:
        cam = cam_red(t)
        red = smooth(ramp(t, 140.5, 147.0)) * (1 - smooth(ramp(t, 151.5, 155.5)))
        img, R = sc.render_world(cam, t, red_amt=red)
        tc = fmt_tc(13 * 3600 + 30 * 60 + t - T_RED)
        gl = 0.5 * (1 - ramp(t, T_RED, T_RED + 0.3)) + 0.12 * red * (rng.random() < 0.3)
        vh = 0.5 + 0.3 * red

    elif t < T_L1:
        cam = cam_arch(t)
        fw = 0.0
        if t > 169.5:
            rate = lerp(2.5, 14, ramp(t, 169.5, T_NOCLIP))
            fw = float(math.sin(t * rate * 6.283) > 0.3) * (0.4 + 0.6 * ramp(t, 172, T_NOCLIP))
        img, R = sc.render_world(cam, t, flicker_wall=fw)
        tc = fmt_tc(19 * 3600 + 51 * 60 + t - T_ARCH)
        gl = 0.5 * (1 - ramp(t, T_ARCH, T_ARCH + 0.3)) + 1.2 * smooth(ramp(t, T_NOCLIP - 0.3, T_L1))
        st = smooth(ramp(t, T_NOCLIP, T_L1))
        if st > 0:
            img = img * (1 - st) + static(rng) * st * 0.7
        vh = 0.4 + 0.5 * st
        a = smooth(ramp(t, 160.0, 161.2)) * (1 - smooth(ramp(t, 165.2, 166.4)))
        if a > 0:
            put_text(img, "THE ARCHES", W1 / 2, H1 / 2 - 260, 54, FONT_SERIF, (0.3, 0.26, 0.15), a, tracking=0.3, shadow=0)

    elif t < T_END:
        cam = cam_l1(t)
        img = sc.render_level1(cam, t)
        level = "LV-1"
        tc = fmt_tc(0.5 + t - T_L1)
        st = 0.6 * (1 - ramp(t, T_L1, T_L1 + 0.6)) + 0.9 * ramp(t, T_END - 0.5, T_END)
        img = img * (1 - st) + static(rng) * st
        gl = 0.4 * (1 - ramp(t, T_L1, T_L1 + 0.8)) + ramp(t, T_END - 0.6, T_END)
        vh = 0.55

    else:
        lt = t - T_END
        img = card_bg()
        draw_card(img, CARD_END, lt, y0=250, size=36, dy=60)
        a = smooth(ramp(lt, 11.4, 12.8))
        if a > 0:
            put_text(img, "「不要回应。」", W1 / 2, 770, 64, FONT_CJK, (1.0, 0.86, 0.35), a, shadow=0, glow=0.5 * a)
        vh, gr, hud_a = 0.3, 0.03, 0.0
        gl = 0.8 * (1 - ramp(t, T_END, T_END + 0.4))
        fade = ((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)))

    return finish(img, rng, t, vh=vh, gl=gl, gr=gr, vig=0.6,
                  hud_args=dict(alpha=hud_a, tc=tc, level=level, center=center, cam_label="CAM-??  UNKNOWN"),
                  lb=lb, subs=SUBS, fade=fade)

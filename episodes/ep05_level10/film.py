"""
Ep05 — LEVEL 10 "Bumper Crop": a family's home-video tape.

    python -m backrooms.render ep05_level10 --still 15,40,60,90,115,135,155,175
    python -m backrooms.render ep05_level10 --all [--res 4k]
"""
import math
import os

import cv2
import numpy as np

from backrooms.core import (
    W, H, W1, H1, S, FPS, F0, FONT_CJK, FONT_SERIF, FONT_MONO, FONT_SANS_BOLD, Cam, Path, handheld,
    look_at, put_text, static, smooth, smoother, ramp, lerp, draw_card, card_bg, finish, walk_bob,
    subs_from_lines, px, s2l, l2s, value_noise,
)
from . import scene as sc

TITLE = "LEVEL 10 —「丰收」Bumper Crop"
DURATION = 200.0
HERE = os.path.dirname(os.path.abspath(__file__))

T_PICNIC = 10.0
T_NOCLIP = 20.6
T_MONT = 24.0
T_ARRIVE = 35.0
T_ROAD = 56.0
T_BARN = 80.0
T_INSIDE = 90.0
T_STEP = 102.0
T_RAIN = 124.0
T_NIGHT = 146.0
T_RUN = 160.0
T_LAKE = 166.0
T_END = 186.0

SUBS = subs_from_lines(os.path.join(HERE, "voice", "lines.json"))

CARD_OPEN = [
    ("M.E.G. 档案 · 拾获物品", 0.4),
    ("物品 ············ 家用摄像机 1 台 · 录像带 1 盒", 1.4),
    ("地点 ············ Level 10 · 湖畔", 2.6),
    ("附注 ············ 发现时，一只狗守在旁边", 3.8),
    ("", 0),
]
CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 10「丰收」Bumper Crop", 0.6),
    ("生存难度 ········ 1  ·  出口 2/5 · 环境 1/5 · 实体 0/5", 1.8),
    ("出口 ············ 沿道路长途行走 → Level 11", 3.0),
    ("                 走进麦田中的油菜花丛 → Level 184", 4.0),
    ("                 游进湖中 → 水域层级（研究中）", 5.0),
    ("备注 ············ M.E.G. 已停止对本层小麦的常规收割", 6.2),
    ("", 0),
]

ROAD_X = sc.HEDGE + 2.2


def datestamp(img, t, text, a=1.0):
    put_text(img, text, W1 - 560, H1 - 90, 34, FONT_SANS_BOLD, (1.0, 0.86, 0.35), a * 0.9, shadow=0.6, anchor="left")


def home_hud(img, t, a=1.0, mode="PLAY"):
    put_text(img, ("▶ " if mode == "PLAY" else "■ ") + mode, 150, 86, 30, FONT_SANS_BOLD, (1, 1, 1), a * 0.85, shadow=0.6)
    put_text(img, "SP", W1 - 150, 86, 28, FONT_SANS_BOLD, (1, 1, 1), a * 0.75, shadow=0.6)


def stamp_for(t):
    # the family camcorder never had its clock reset after the picnic
    days = 0 if t < T_NOCLIP else (1 + int((t - T_NOCLIP) * 1.3) if t < T_ARRIVE else 31 + int((t - T_ARRIVE) / 6))
    if t >= T_STEP:
        days = 52 + int(max(0, t - 111) * 2.5) if t < 117 else 66
    if t >= T_RAIN:
        days = 140
    if t >= T_NIGHT:
        days = 213
    d0 = 21 + days
    months = ["AUG", "SEP", "OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR"]
    mlen = [31, 30, 31, 30, 31, 31, 28, 31, 30]
    mi = 0
    while d0 > mlen[mi]:
        d0 -= mlen[mi]
        mi += 1
    hh = 3 + int(t * 0.1) % 9
    return f"{months[mi]}.{d0:2d} {'1999' if mi < 5 else '2000'}  PM {hh}:{int(t * 7) % 60:02d}"


# ---------------------------------------------------------------------------
def cam_picnic(t):
    hy, hp, hr = handheld(t, 1.6, 0.4)
    return Cam((60.0, 40.0, 1.3), 30 + 25 * math.sin(t * 0.3) + hy, -6 + hp, hr)


def cam_arrive(t):
    walk = smoother(ramp(t, 39.0, T_ROAD))
    y = lerp(20.0, 34.0, walk)
    hy, hp, hr = handheld(t, 1.2, 1.2)
    pan = -35 + 70 * smoother(ramp(t, 37.0, 50.0))
    return Cam((ROAD_X, y, 1.25 + walk_bob(t, 0.8) * walk), pan * (1 - smooth(ramp(t, 51, 55.5))) + hy, 2 + hp, hr)


def cam_road(t):
    lt = t - T_ROAD
    y = 34.0 + 1.0 * lt
    hy, hp, hr = handheld(t, 1.1, 2.2)
    yaw = 0
    yaw += 75 * smooth(ramp(t, 63.0, 67.0)) * (1 - smooth(ramp(t, 71.5, 75.0)))   # look out across the fields
    return Cam((ROAD_X, y, 1.25 + walk_bob(t, 1.0)), yaw + hy, 1.5 + hp, hr)


def cam_barn(t):
    walk = smoother(ramp(t, T_BARN, T_INSIDE))
    p = np.array([lerp(12.0, 17.6, walk), lerp(22.0, 41.5, walk)])
    yaw, pitch = look_at((*p, 1.25), (sc.BARN[0], sc.BARN[1] - 7, 2.6))
    hy, hp, hr = handheld(t, 1.0, 3.0)
    return Cam((p[0], p[1], 1.25 + walk_bob(t, 1.1)), yaw + hy, pitch * 0.6 + hp, hr)


def cam_inside(t):
    lt = t - T_INSIDE
    hy, hp, hr = handheld(t, 0.9, 4.0)
    return Cam((-2.0 + 0.08 * lt, -5.0 + 0.1 * lt, 1.2), 30 + 18 * math.sin(lt * 0.25) + hy, -4 + hp, hr)


def cam_rain(t):
    lt = t - T_RAIN
    hy, hp, hr = handheld(t, 1.3, 5.0)
    if t < 135:
        return Cam((70.0, 30.0 + 0.5 * lt, 1.12), 20 + hy, 2 + hp, hr)          # among the stalks
    return Cam((ROAD_X - 0.4, 120 + 0.6 * lt, 1.25), 8 + hy, 0 + hp, hr)


def cam_night(t):
    lt = t - T_NIGHT
    hy, hp, hr = handheld(t, 1.4 + 1.5 * ramp(t, 157, 160), 6.0)
    wake = smoother(ramp(t, T_NIGHT, 150.0))
    pitch = lerp(-60, -8, wake)
    return Cam((1.5, 3.0, lerp(0.5, 1.0, wake)), lerp(-120, -170, smooth(ramp(t, 151, 156))) + hy, pitch + hp, hr + 15 * (1 - wake))


def cam_run(t):
    lt = t - T_RUN
    hy, hp, hr = handheld(t * 2, 3.5, 7.0)
    return Cam((sc.BARN[0] + 1.0, sc.BARN[1] - 8 - 3.2 * lt, 1.15 + walk_bob(t, 3.5, 0.04)), 200 + hy, -4 + hp, hr)


LAKE = sc.LAKES[0]


TH = math.radians(-115)
KID = np.array([LAKE[0] + LAKE[2] * 1.07 * math.cos(TH), LAKE[1] + LAKE[3] * 1.07 * math.sin(TH)])
CAMP = np.array([LAKE[0] + LAKE[2] * 1.2 * math.cos(TH - 0.08), LAKE[1] + LAKE[3] * 1.2 * math.sin(TH - 0.08)])


def cam_lake(t):
    hy, hp, hr = handheld(t, 0.15, 8.0)          # the camera is set down on a rock
    yaw, _ = look_at((*CAMP, 0.7), (LAKE[0] + 10, LAKE[1], 0.0))
    return Cam((CAMP[0], CAMP[1], 0.7), yaw + hy, -3.5 + hp, hr + 1.5)


# ---------------------------------------------------------------------------
# the doorstep close-up: a plank step on soil, a grain of wheat, the worm
# ---------------------------------------------------------------------------
_NS = value_noise(512, 6, 77)


def doorstep(t, day=1.0):
    cam = Cam((0.12, -0.62, 0.62), -6, -38, 0)
    hy, hp, hr = handheld(t, 0.25, 9.0)
    cam = Cam((0.12, -0.62, 0.62), -6 + hy, -38 + hp, hr)
    X = (np.arange(W // 2, dtype=np.float32) + 0.5) * 2 - W / 2
    Y = H / 2 - (np.arange(H // 2, dtype=np.float32) + 0.5) * 2
    X, Y = np.meshgrid(X, Y)
    f = cam.f
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    ox, oy, oz = cam.pos
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(dz < 0, -oz / dz, 1e9)
        tt_ = np.where(dz < 0, (oz - 0.18) / -dz, 1e9)
        tf = np.where(dy > 0, (0.0 - oy) / dy, 1e9)
    hx_t, hy_t = ox + dx * tt_, oy + dy * tt_
    top = (np.abs(hx_t) < 0.6) & (hy_t > 0) & (hy_t < 0.35)
    hz_f = oz + dz * tf
    hx_f = ox + dx * tf
    front = (tf < tg) & (np.abs(hx_f) < 0.6) & (hz_f > 0) & (hz_f < 0.18) & ~top
    gx, gy = ox + dx * tg, oy + dy * tg
    soil = s2l((70, 52, 36)) * (0.6 + 0.7 * _NS[np.floor(gy * 300).astype(np.int64) % 512, np.floor(gx * 300).astype(np.int64) % 512][..., None])
    plank_u = np.where(top, hx_t, hx_f)
    grainv = _NS[np.floor(np.where(top, hy_t, hz_f) * 900).astype(np.int64) % 512, np.floor(plank_u * 30).astype(np.int64) % 512]
    wood = s2l((150, 112, 72)) * (0.7 + 0.45 * grainv[..., None])
    seam = np.abs((np.where(top, hy_t, 0.5) / 0.117) % 1 - 0.5) > 0.46
    wood = wood * np.where(seam & top, 0.5, 1.0)[..., None]
    col = np.where(top[..., None], wood, np.where(front[..., None], wood * 0.6, soil))
    col = col * (0.55 + 0.45 * day) * np.array([0.95, 0.97, 1.0], np.float32)
    img = l2s(np.clip(col, 0, 1))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    out = np.ascontiguousarray((img * 255).astype(np.uint8))
    # the grain of wheat
    g = cam.project(np.array([[0.04, 0.16, 0.185]]))
    eaten = t > 120.6
    if not eaten:
        s = cam.f / g[2][0]
        cv2.ellipse(out, (int(g[0][0] * 8), int(g[1][0] * 8)), (int(0.006 * s * 8), int(0.0035 * s * 8)), 20, 0, 360,
                    tuple(float(v) for v in l2s(s2l((220, 180, 100)) * (0.6 + 0.4 * day)) * 255), -1, cv2.LINE_AA, 3)
    # the worm: rises from the soil, reaches over the step's edge, takes the grain, goes back down
    up = smoother(ramp(t, 117.6, 120.2)) * (1 - smoother(ramp(t, 121.0, 123.4)))
    if up > 0.01:
        n = 26
        s_ = np.linspace(0, 1, n) * up
        base = np.array([0.2, -0.1, 0.0])
        tip = np.array([0.05, 0.15, 0.2])
        ctrl = np.array([0.22, -0.02, 0.32])
        pts = ((1 - s_)[:, None] ** 2 * base + 2 * ((1 - s_) * s_)[:, None] * ctrl + (s_ ** 2)[:, None] * tip)
        pts[:, 0] += 0.01 * np.sin(np.linspace(0, 6, n) + t * 6)
        qx, qy, qd = cam.project(pts)
        for k in range(n - 1):
            s = cam.f / qd[k]
            w = max(3, int(0.022 * s * (1 - 0.3 * k / n)))
            ring = 0.8 + 0.2 * (k % 2)
            c = tuple(float(v) * ring for v in l2s(s2l((150, 92, 88)) * (0.6 + 0.4 * day)) * 255)
            cv2.line(out, (int(qx[k] * 8), int(qy[k] * 8)), (int(qx[k + 1] * 8), int(qy[k + 1] * 8)), c, w, cv2.LINE_AA, 3)
    return out.astype(np.float32) / 255.0


def montage(t, rng):
    lt = t - T_MONT
    labels = [("LEVEL 1", 0.0), ("LEVEL 4", 2.2), ("LEVEL 5", 4.2), ("LEVEL 6", 6.2), ("LEVEL 8", 8.2)]
    cur = max([l for l in labels if lt >= l[1]], key=lambda q: q[1])
    k = labels.index(cur)
    tones = [(0.32, 0.32, 0.3), (0.42, 0.4, 0.34), (0.3, 0.2, 0.14), (0.0, 0.0, 0.0), (0.18, 0.14, 0.12)]
    img = np.ones((H, W, 3), np.float32) * np.array(tones[k], np.float32)
    nz = cv2.resize(rng.random((54, 96)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = img * (0.75 + 0.5 * nz[..., None])
    put_text(img, cur[0], W1 / 2, H1 / 2, 70, FONT_SERIF, (0.9, 0.9, 0.9), 0.85, tracking=0.3, shadow=0.6)
    return img


def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 5)
    vh, gl, gr = 0.55, 0.0, 0.03
    lb, fade = 0.0, None
    play = True
    stamp = True
    mode = "PLAY"

    if t < T_PICNIC:
        img = card_bg()
        draw_card(img, CARD_OPEN, t)
        vh, gr, play, stamp = 0.3, 0.03, False, False
        fade = ((0, 0, 0), 1 - ramp(t, 0, 0.6) + ramp(t, T_PICNIC - 0.8, T_PICNIC))

    elif t < T_MONT:
        cam = cam_picnic(t)
        img, d = sc.render(cam, t, summer=True, fogd=900, light=1.15)
        img = sc.draw_trees(img, d, cam, summer=True, fogd=900, light=1.15)
        img = sc.figure(img, d, cam, 63.0, 46.0, 1.7, (0.5, 0.3, 0.27), 1.15, "adult")        # Mum
        img = sc.figure(img, d, cam, 61.5, 47.5, 1.82, (0.3, 0.33, 0.42), 1.15, "adult")        # Dad
        img = sc.figure(img, d, cam, 62.0 + 1.2 * math.sin(t * 1.4), 44.0 + 0.6 * math.cos(t), 0, (0.5, 0.36, 0.2), 1.15, "dog", t)
        gl = 1.4 * smooth(ramp(t, T_NOCLIP - 0.3, T_NOCLIP + 0.6))
        st = smooth(ramp(t, T_NOCLIP, T_NOCLIP + 1.2))
        img = img * (1 - st) + static(rng) * st * 0.7
        dk = smooth(ramp(t, T_NOCLIP + 1.6, T_MONT))
        fade = ((0, 0, 0), dk + (1 - ramp(t, T_PICNIC, T_PICNIC + 0.4)))

    elif t < T_ARRIVE:
        img = montage(t, rng)
        gl = 0.5 + 0.5 * (rng.random() < 0.3)
        st = 0.35
        img = img * (1 - st) + static(rng) * st
        vh = 0.8
        fade = ((0, 0, 0), smooth(ramp(t, T_ARRIVE - 1.0, T_ARRIVE)))

    elif t < T_ROAD:
        cam = cam_arrive(t)
        img, d = sc.render(cam, t, gust=1.2)
        img = sc.draw_trees(img, d, cam)
        img = sc.stalks(img, d, cam, t, 1.2)
        k = ramp(t, 39.5, 45)
        img = sc.figure(img, d, cam, ROAD_X + 0.6, lerp(24, 33, k), 1.7, (0.5, 0.3, 0.27), 1.0)
        img = sc.figure(img, d, cam, ROAD_X - 0.8, lerp(23, 31, k), 1.82, (0.28, 0.31, 0.4), 1.0)
        img = sc.figure(img, d, cam, ROAD_X + 0.3 + 0.5 * math.sin(t * 2), lerp(25, 40, ramp(t, 40, 47)), 0, (0.45, 0.32, 0.18), 1.0, "dog", t)
        # stepping out of the dark: the last cave (Level 8) gives way to daylight
        out = smoother(ramp(t, T_ARRIVE, 38.5))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((xx - W / 2) / W, (yy - H / 2) / H)
        hole = np.clip((out * 1.4 - r) / 0.12, 0, 1)
        img = img * hole[..., None] + (1 - hole[..., None]) * np.array([0.02, 0.018, 0.015], np.float32)
        img = img * (1 + 0.8 * (1 - smooth(ramp(t, 37.0, 41.0))) * out)          # dazzled by the light
        a = smooth(ramp(t, 46.0, 47.4)) * (1 - smooth(ramp(t, 51.6, 53.0)))
        if a > 0:
            put_text(img, "LEVEL 10", W1 / 2, H1 / 2 - 40, 118, FONT_SERIF, (1, 1, 1), a, tracking=0.32, shadow=0.5, glow=0.15)
            put_text(img, "B U M P E R   C R O P", W1 / 2, H1 / 2 + 62, 34, FONT_SERIF, (1, 1, 1), a * 0.95, shadow=0.5)
            put_text(img, "丰 收", W1 / 2, H1 / 2 + 118, 34, FONT_CJK, (1, 1, 1), a * 0.9, shadow=0.5)

    elif t < T_BARN:
        cam = cam_road(t)
        img, d = sc.render(cam, t, gust=1.0)
        img = sc.draw_trees(img, d, cam)
        img = sc.stalks(img, d, cam, t)
        lt = t - T_ROAD
        img = sc.figure(img, d, cam, ROAD_X + 0.6, 40 + 1.0 * lt, 1.7, (0.5, 0.3, 0.27), 1.0)
        img = sc.figure(img, d, cam, ROAD_X - 0.7, 41 + 1.0 * lt, 1.82, (0.28, 0.31, 0.4), 1.0)
        img = sc.figure(img, d, cam, ROAD_X + 0.2 + 0.6 * math.sin(t * 1.7), 37.5 + 1.0 * lt + 1.2 * math.sin(t * 0.9), 0, (0.45, 0.32, 0.18), 1.0, "dog", t)

    elif t < T_INSIDE:
        cam = cam_barn(t)
        img, d = sc.render(cam, t)
        img = sc.draw_trees(img, d, cam)
        sc.draw_barn(img, d, cam)
        img = sc.figure(img, d, cam, sc.BARN[0] - 1.5, sc.BARN[1] - 8.5, 1.82, (0.28, 0.31, 0.4), 1.0)
        img = sc.figure(img, d, cam, sc.BARN[0] + 0.5 + 0.8 * math.sin(t), sc.BARN[1] - 10 + 0.5 * math.cos(t), 0, (0.45, 0.32, 0.18), 1.0, "dog", t)
        gl = 0.4 * (1 - ramp(t, T_BARN, T_BARN + 0.3))

    elif t < T_STEP:
        cam = cam_inside(t)
        img, d = sc.barn_interior(cam, t)
        img = sc.figure(img, d, cam, 2.5, 2.0, 1.82, (0.27, 0.29, 0.36), 1.0)          # Dad, hammering
        img = sc.figure(img, d, cam, -2.2, 4.5, 1.7, (0.45, 0.28, 0.25), 1.0)         # Mum by the stove
        img = sc.figure(img, d, cam, 0.5 + 0.5 * math.sin(t), 0.0, 0, (0.45, 0.32, 0.18), 1.0, "dog", t)
        gl = 0.4 * (1 - ramp(t, T_INSIDE, T_INSIDE + 0.3))

    elif t < T_RAIN:
        lt = t - T_STEP
        day = 1.0
        if 111.5 < t < 116.8:                                    # time-lapse: days pass, the grain stays
            day = 0.55 + 0.45 * math.cos((t - 111.5) * 2 * math.pi * 1.3)
        img = doorstep(t, day)
        gl = 0.4 * (1 - ramp(t, T_STEP, T_STEP + 0.3)) + 0.12 * (111.5 < t < 116.8)
        if 111.5 < t < 116.8:
            mode = "▶▶"

    elif t < T_NIGHT:
        cam = cam_rain(t)
        img, d = sc.render(cam, t, gust=1.6, fogd=160, light=0.82)
        img = sc.draw_trees(img, d, cam, light=0.82, fogd=160)
        img = sc.stalks(img, d, cam, t, 1.6, light=0.82)
        if t >= 135:
            img = sc.figure(img, d, cam, ROAD_X + 0.3, 120 + 0.6 * (t - T_RAIN) + 6, 0, (0.4, 0.3, 0.18), 0.82, "dog", t)
        # rain
        out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        for _ in range(900):
            x, y = rng.integers(0, W), rng.integers(0, H)
            cv2.line(out, (int(x), int(y)), (int(x) - px(4), int(y) + px(34)), (205, 208, 212), 1, cv2.LINE_AA)
        img = out.astype(np.float32) / 255 * 0.9 + img * 0.1
        gl = 0.4 * (1 - ramp(t, T_RAIN, T_RAIN + 0.3)) + 0.4 * (1 - ramp(t, 135, 135.3)) * (t > 135)

    elif t < T_RUN:
        cam = cam_night(t)
        img, d = sc.barn_interior(cam, t, light=0.5, smoke=0.85, night=0.85)
        blur = 1 + 10 * (1 - smooth(ramp(t, T_NIGHT, 150.5)))
        if t > 151:   # two shapes on the floor, out of focus, barely visible through the smoke
            img = sc.figure(img, d, cam, -2.6, -4.5, 0.4, (0.05, 0.05, 0.05), 1.0, "adult")
            img = sc.figure(img, d, cam, -1.0, -5.6, 0.35, (0.05, 0.05, 0.05), 1.0, "adult")
            blur = max(blur, 4)
        img = cv2.GaussianBlur(img, (0, 0), px(blur))
        img = img * (0.8 + 0.2 * math.sin(t * 17) ** 2)
        gr = 0.06
        gl = 0.5 * (1 - ramp(t, T_NIGHT, T_NIGHT + 0.3))

    elif t < T_LAKE:
        cam = cam_run(t)
        img, d = sc.render(cam, t, night=0.85, fogd=120)
        img = sc.draw_trees(img, d, cam, night=0.85, fogd=120)
        img = sc.figure(img, d, cam, cam.pos[0] + 0.3, cam.pos[1] - 2.0 - 0.6 * math.sin(t * 9), 0, (0.03, 0.03, 0.03), 1.0, "dog", t * 2)
        gr = 0.07
        gl = 0.6 * (1 - ramp(t, T_RUN, T_RUN + 0.3)) + 0.15 * (rng.random() < 0.3)
        vh = 0.8

    elif t < T_END:
        cam = cam_lake(t)
        img, d = sc.render(cam, t, gust=0.5, fogd=260, light=0.78)
        img = sc.draw_trees(img, d, cam, light=0.78, fogd=260)
        kx, ky = KID
        img = sc.figure(img, d, cam, kx, ky, 0.9, (0.12, 0.12, 0.12), 0.78, "adult")           # sitting
        img = sc.figure(img, d, cam, kx + 0.9, ky - 0.1, 0, (0.3, 0.22, 0.14), 0.78, "dog", 0)
        gl = 0.5 * (1 - ramp(t, T_LAKE, T_LAKE + 0.3))
        dk = smooth(ramp(t, T_END - 3.0, T_END - 0.6))
        fade = ((0, 0, 0), dk)
        if t > T_END - 3.0:
            mode = "STOP"

    else:
        lt = t - T_END
        img = card_bg()
        draw_card(img, CARD_END, lt, y0=270, size=34, dy=58)
        vh, gr, play, stamp = 0.3, 0.03, False, False
        fade = ((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)) + 1 - ramp(t, T_END, T_END + 0.4))

    if play:
        home_hud(img, t, 1.0, mode)
    if stamp:
        datestamp(img, t, stamp_for(t))
    return finish(img, rng, t, vh=vh, gl=gl, gr=gr, vig=0.55, hud_args=None, lb=lb, subs=SUBS, fade=fade)

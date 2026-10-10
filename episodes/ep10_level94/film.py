"""
Ep10 — LEVEL 94 "Motion": Mara Quist's field film. A clay town shot on twos, a cheerful
siren-pole tune, the music winding down at dusk — and the Animations.

    python -m backrooms.render ep10_level94 --still 14,21,30,36,44,56,70,80,100,115,122,136,140,150,166
    python -m backrooms.render ep10_level94 --all [--res 4k]
"""
import math
import os

import cv2
import numpy as np

from backrooms.core import (
    W, H, W1, H1, S, FPS, FONT_CJK, FONT_SERIF, FONT_MONO, FONT_SANS_BOLD, Cam, handheld, look_at, put_text,
    fmt_tc, smooth, smoother, ramp, lerp, draw_card, card_bg, finish, walk_bob, subs_from_lines, px, bloom, s2l, l2s,
)
from . import scene as sc

TITLE = "LEVEL 94 —「动画」Motion"
DURATION = 200.0
HERE = os.path.dirname(os.path.abspath(__file__))

T_TITLE = 10.0
T_HOLE = 18.0
T_HILL = 24.0
T_CASTLE = 52.0
T_TOWN = 64.0
T_DUSK = 96.0
T_NIGHT = 110.0
T_RUN = 144.0
T_DARK = 156.0
T_L9 = 158.0
T_END = 176.0

SUBS = subs_from_lines(os.path.join(HERE, "voice", "lines.json"))

CARD_OPEN = [
    ("M.E.G. 档案 · 外勤胶片 94-A", 0.4),
    ("拍摄者 ·········· 玛拉·奎斯特（外勤研究员）", 1.6),
    ("器材 ············ 手摇 16mm 摄影机（得自枢纽）", 2.8),
    ("", 0),
]
CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 94「动画」Motion", 0.6),
    ("生存难度 ········ 3  ·  白天安全，夜晚极危险", 1.8),
    ("入口 ············ 逃离 Level 601 后，枢纽中出现的矩形洞口", 3.0),
    ("实体 ············ 动画 · 机器人守卫 · 动画之王 · 夜间：笑魇 · 猎犬", 4.2),
    ("出口 ············ 通过动画之王的试炼 · 部分房屋 → Level 7 / 9 / 53", 5.4),
    ("", 0),
]

CASTLE = np.array([-150.0, 560.0, 75.0])
WALL_P = np.array([11.0, -96.0])


def gz(x, y):
    return float(sc.ground_h(np.array([x], np.float64), np.array([y], np.float64))[0])


def held(t):
    """stop-motion: everything moves on twos (12 poses a second)"""
    return math.floor(t * 12) / 12


def tod_at(t):
    return smooth(ramp(t, T_DUSK, T_DUSK + 7)) + smooth(ramp(t, T_DUSK + 7, T_NIGHT + 1))


# ---------------------------------------------------------------------------
# the camera (a person holding a hand-cranked camera, animated a pose at a time)
# ---------------------------------------------------------------------------
def cam_world(t):
    hy, hp, hr = handheld(t, 0.8, 0.0)
    if t < T_CASTLE:
        y = lerp(-114.0, -104.0, smooth(ramp(t, T_HILL + 2, 33.0)))
        x = 0.0
        crouch = smoother(ramp(t, 34.0, 35.5)) * (1 - smoother(ramp(t, 39.0, 40.5)))
        pitch = lerp(-9, -55, crouch)
        yaw = 0.0
        # the lone wall, the stray car, the water tower
        y = lerp(y, -101.0, smooth(ramp(t, 41, 46)))
        x = lerp(x, 4.0, smooth(ramp(t, 41, 46)))
        yw_wall, pt_wall = look_at((x, y, gz(x, y) + 1.6), (WALL_P[0], WALL_P[1], gz(*WALL_P) + 1.2))
        k = smooth(ramp(t, 40.5, 42.5)) * (1 - smooth(ramp(t, 47.0, 48.5)))
        yaw = lerp(yaw, yw_wall, k)
        pitch = lerp(pitch, pt_wall, k)
        yaw = yaw + lerp(0, -40, smooth(ramp(t, 47.5, 49.5))) + lerp(0, 105, smooth(ramp(t, 49.5, 51.8)))
        z = gz(x, y) + lerp(1.6, 0.8, crouch)
        return Cam((x, y, z), yaw + hy, pitch + hp, hr)
    if t < T_TOWN:
        x, y = 4.0, -101.0
        z = gz(x, y) + 1.6
        yw, pt = look_at((x, y, z), CASTLE)
        k = smooth(ramp(t, T_CASTLE, T_CASTLE + 3))
        yaw = lerp(65, yw, k)
        pitch = lerp(-6, pt + 3, k)
        # lower the camera back to the town as the archivist talks
        k2 = smooth(ramp(t, 61.5, T_TOWN))
        yaw = lerp(yaw, 0, k2)
        pitch = lerp(pitch, -8, k2)
        return Cam((x, y, z), yaw + hy, pitch + hp, hr)
    if t < T_DUSK:
        y = lerp(-56.0, -24.0, smooth(ramp(t, T_TOWN, 75.5)))
        x = 0.6
        # step up to a house to look at the fingerprints
        k = smooth(ramp(t, 75.5, 78.0)) * (1 - smooth(ramp(t, 83.0, 86.0)))
        hx, hy_ = house_front(-1, -3)
        x = lerp(x, hx + 1.0, k)
        y = lerp(y, hy_, k)
        yaw = lerp(4 * math.sin(t * 0.4), -90, k)
        pitch = lerp(-3, -6, k)
        y = lerp(y, -11.0, smooth(ramp(t, 85.0, T_DUSK)))
        x = lerp(x, -3.4, smooth(ramp(t, 85.0, T_DUSK)) * (1 - k))
        yaw = yaw + 20 * smooth(ramp(t, 88, 92)) * (1 - smooth(ramp(t, 93, 95)))
        z = 1.6 + 0.05 * math.sin(t * 7) * (1 - k)
        return Cam((x, y, z), yaw + hy, pitch + hp, hr)
    if t < T_NIGHT:
        x, y = -3.4, -11.0
        yaw = 40 * math.sin((t - T_DUSK) * 0.35) + 25 * smooth(ramp(t, 99, 101)) * (1 - smooth(ramp(t, 104, 107)))
        pitch = 6 * smooth(ramp(t, 99, 101)) * (1 - smooth(ramp(t, 104, 107)))   # up at the silent siren pole
        return Cam((x, y, 1.6), yaw + hy, pitch + hp, hr)
    if t < T_RUN:
        y = lerp(-11.0, -13.5, smooth(ramp(t, 128, 140)))
        amp = 1.0 + 2.0 * smooth(ramp(t, 133, 135))
        hy, hp, hr = handheld(t, amp, 1.0)
        yaw = 3.0 + 6 * smooth(ramp(t, 131, 133)) - 14 * smooth(ramp(t, 134.5, 136)) * (1 - smooth(ramp(t, 137, 138.5)))
        pitch = -2 + 3 * smooth(ramp(t, 139.0, 139.4))
        return Cam((-3.4 + 0.6 * smooth(ramp(t, 128, 140)), y, 1.6), yaw + hy, pitch + hp, hr)
    # the run: whip round, down the street, to the dark wooden door
    hy, hp, hr = handheld(t * 1.7, 3.0, 2.0)
    door = sc.door_world()
    whip = smooth(ramp(t, T_RUN, T_RUN + 1.2))
    p0 = np.array([-2.8, -13.5])
    p1 = np.array([door[0] - 2.2, door[1]])
    k = smooth(ramp(t, T_RUN + 1.0, 150.5))
    pos = p0 * (1 - k) + p1 * k
    yaw = lerp(3, 160, whip)
    yaw = lerp(yaw, 90, smooth(ramp(t, 148.5, 150.5)))
    # into the doorway
    k2 = smooth(ramp(t, 152.0, 155.0))
    pos = pos * (1 - k2) + np.array([door[0] + 0.6, door[1]]) * k2
    z = 1.6 + 0.12 * abs(math.sin(t * 9)) * (1 - k2)
    return Cam((pos[0], pos[1], z), yaw + hy, -2 + hp, hr)


def house_front(ci, cj):
    w, dd, hw, rh, rot, lean, ox, oy = [float(v) for v in sc.house_params(np.array([ci]), np.array([cj]))]
    return ci * sc.CELL + ox + w / 2, cj * sc.CELL + oy


# ---------------------------------------------------------------------------
# the Animations
# ---------------------------------------------------------------------------
DOORWAYS = [(-6.6, 18.2), (6.8, 26.5), (-6.5, 32.0), (6.6, 12.0), (-6.8, 6.0), (6.9, 40.0)]


def animations(t):
    """list of (pos, height, seed, walk, run, look, grin)"""
    out = []
    if t < 112.0:
        return out
    # the first: at the far end of the street, then walking towards her
    k = smooth(ramp(t, 125.5, 143.0))
    y = lerp(42.0, 2.0, k)
    look = 1.0 if t > 117.0 else -1.0
    out.append((np.array([0.3, y, 0.0]), 1.55, 1, 1.0 if 125.5 < t < 143 else 0.0, 0.0, look, 1.0))
    # more, stepping out of every door
    for i, (dx_, dy_) in enumerate(DOORWAYS):
        t0 = 129.0 + i * 1.3
        if t < t0:
            continue
        kk = smooth(ramp(t, t0, t0 + 2.0))
        x = lerp(dx_, dx_ * 0.45, kk)
        yy = dy_ - 8.0 * smooth(ramp(t, t0 + 2.0, t0 + 14.0))
        out.append((np.array([x, yy, 0.0]), 1.45 + 0.1 * (i % 3), 10 + i, 1.0, 0.0, 1.0, 1.0))
    # the one that leaps right up to the lens
    if 138.6 < t < 144.0:
        kk = smooth(ramp(t, 138.6, 139.4))
        out.append((np.array([lerp(1.5, -2.6, kk), lerp(4.0, -10.6, kk), 0.0]), 1.5, 30, 0.0, 1.0, 1.0, 1.3))
    # during the run, the street fills behind her
    if t >= T_RUN:
        for i in range(5):
            yy = lerp(2.0 - i * 2.5, -9.0 - i * 1.5, smooth(ramp(t, T_RUN + 1.5, 152.0)))
            out.append((np.array([-2.0 + i * 1.1, yy, 0.0]), 1.5, 40 + i, 0.0, 1.0, 1.0, 1.0))
    return out


# ---------------------------------------------------------------------------
# 2D pieces: the cartoon title, the Hub hole, the film look
# ---------------------------------------------------------------------------
def title_card(t):
    lt = t - T_TITLE
    img = np.zeros((H, W, 3), np.float32)
    # sunburst
    cx, cy = W / 2, H * 0.55
    ang = np.arctan2(np.arange(H)[:, None] - cy, np.arange(W)[None, :] - cx) + held(lt) * 0.15
    rays = (np.floor((ang + math.pi) / (2 * math.pi) * 28) % 2).astype(np.float32)
    c1, c2 = s2l((248, 226, 168)), s2l((240, 176, 96))
    img = c1 * rays[..., None] + c2 * (1 - rays[..., None])
    # a banner disc
    cv2.circle(img, (int(cx), int(cy)), px(330), tuple(float(v) for v in s2l((180, 40, 40))), -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), px(330), (0.02, 0.02, 0.02), px(10), cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), px(300), tuple(float(v) for v in s2l((250, 236, 200))), px(4), cv2.LINE_AA)
    bob = 6 * math.sin(held(lt) * 7)
    a = smooth(ramp(lt, 0.3, 0.9))
    put_text(img, "M.E.G. 档案馆 出品", W1 / 2, 330 + bob * 0.5, 40, FONT_CJK, (1.0, 0.95, 0.85), a, shadow=0.8)
    s_ = 1.0 + 0.08 * math.sin(held(lt) * 9) * (lt < 3)
    put_text(img, "LEVEL 94", W1 / 2, 560 + bob, int(150 * s_), FONT_SERIF, (1.0, 0.97, 0.88), smooth(ramp(lt, 0.8, 1.4)), tracking=0.08, shadow=0.95, glow=0.0)
    put_text(img, "~ M O T I O N ~", W1 / 2, 690 - bob, 52, FONT_SERIF, (1.0, 0.92, 0.7), smooth(ramp(lt, 1.6, 2.2)), shadow=0.9)
    put_text(img, "「动 画」", W1 / 2, 770 - bob, 46, FONT_CJK, (1.0, 0.92, 0.7), smooth(ramp(lt, 2.2, 2.8)), shadow=0.9)
    # iris out
    r = lerp(1.2, 0.0, smoother(ramp(lt, 6.4, 7.8))) * math.hypot(W, H) / 2
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (int(cx), int(H / 2)), int(max(0, r)), 1.0, -1, cv2.LINE_AA)
    return img * m[..., None]


def film_look(img, rng, t, fi, strength=1.0):
    """16mm: gate weave, exposure flicker per pose, dust and the odd scratch"""
    k = int(t * 12)
    r2 = np.random.default_rng(k * 31 + 5)
    img = img * (1.0 + 0.035 * strength * r2.normal())
    dx, dy = rng.normal(0, px(1.5) * strength), rng.normal(0, px(1.2) * strength)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    for _ in range(int(rng.integers(0, 6) * strength)):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
        c = (0.9, 0.9, 0.85) if rng.random() < 0.5 else (0.0, 0.0, 0.0)
        cv2.circle(img, (x, y), int(rng.uniform(1, px(4))), c, -1, cv2.LINE_AA)
    if rng.random() < 0.15 * strength:
        x = int(rng.uniform(0, W))
        cv2.line(img, (x, 0), (x + int(rng.normal(0, px(10))), H), (0.8, 0.8, 0.75), 1, cv2.LINE_AA)
    return img


_CACHE = {}
DOOR_ART = None


def world_frame(t):
    """the clay world at one held pose (cached so the twin frame of each pose is free)"""
    th = held(t)
    key = round(th * 12)
    if key in _CACHE:
        return _CACHE[key].copy()
    _CACHE.clear()
    tod = tod_at(th)
    cam = cam_world(th)
    flick = float(np.random.default_rng(key).normal(0, 0.02))
    img, d = sc.render(cam, th, tod, jitter=flick)
    img = sc.clouds(img, d, cam, th, tod)
    img = sc.castle(img, d, cam, th, tod)
    img = sc.moon(img, d, cam, tod)
    for pos, hgt, seed, walk, run, look, grin in animations(th):
        img = sc.animation(img, d, cam, pos, hgt, th, seed, walk=walk, run=run, look=look, grin=grin)
    if th > 149.5:
        img = door_open(img, d, cam, th)
    _CACHE[key] = img
    return img.copy()


def door_open(img, d, cam, t):
    """the dark wooden door swings in on nothing — on the other side it's raining"""
    door = sc.door_world()
    k = smooth(ramp(t, 149.5, 151.5))
    w = 0.9 * (1 - 0.85 * k)
    c = np.array([door[0] - 0.02, door[1], 0.0])
    q = cam.project(np.array([c + [0, -0.45, 1.75], c + [0, 0.45, 1.75], c + [0, 0.45, 0.0], c + [0, -0.45, 0.0]]))
    if (q[2] < 0.2).any():
        return img
    poly = np.stack([q[0], q[1]], 1).astype(np.int32)
    m = np.zeros((H, W), np.float32)
    cv2.fillPoly(m, [poly], 1.0, cv2.LINE_AA)
    inner = np.zeros_like(img) + np.array([0.004, 0.005, 0.008], np.float32)
    return img * (1 - m[..., None] * k) + inner * m[..., None] * k


def level9(t, rng):
    """through the door: the real world again (well — Level 9). No more stop-motion."""
    from episodes.ep09_level9 import scene as l9
    hy, hp, hr = handheld(t, 1.2, 9.0)
    step = smooth(ramp(t, 168.0, 174.0))
    cam = Cam((16.2 - 2.5 * step, 128.0, 1.65), -90 + hy + 10 * math.sin((t - T_L9) * 0.2), -3 + hp, hr)
    lamp_i = int(np.argmin([(L[0] - 4.9) ** 2 + (L[1] - 120.0) ** 2 for L in l9.LAMPS]))
    v = math.sin(t * 6.1) + 0.6 * math.sin(t * 19.3)
    b = 0.85 if v > -0.4 else 0.05
    b *= smooth(ramp(t, T_L9 + 2.0, T_L9 + 2.2))
    lit = [l9.lamp_light(lamp_i, b)] if b > 0.05 else []
    img, d, pm = l9.render(cam, t, flash=0.0, lamps=lit, fog=60.0)
    img = l9.puddle_reflect(img, d, pm, cam, t)
    img = l9.draw_trees(img, d, cam, 0.0, lit)
    img = l9.draw_power(img, d, cam, 0.0, lit)
    img = l9.draw_lamps(img, d, cam, {lamp_i: b}, 0.0, lit)
    # rain
    out = l9.to8(img)
    base = out.copy()
    for _ in range(420):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(-px(60), H))
        ln = int(rng.uniform(px(14), px(40)))
        cv2.line(out, (x, y), (x - px(3), y + ln), (60, 64, 72), 1, cv2.LINE_AA)
    img = l9.from8(img, base, out)
    img = img * 3.0
    return img / (1 + img * 0.5)


# ---------------------------------------------------------------------------
def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 94)
    if t < T_TITLE:
        img = card_bg()
        draw_card(img, CARD_OPEN, t)
        return finish(img, rng, t, vh=0.3, gl=0, gr=0.03, vig=0.6, hud_args=None, lb=0, subs=SUBS,
                      fade=((0, 0, 0), 1 - ramp(t, 0, 0.6) + ramp(t, T_TITLE - 0.8, T_TITLE)))
    if t < T_HOLE:
        img = title_card(held(t))
        img = film_look(img, rng, t, fi, 1.0)
        return finish(img, rng, t, vh=0.0, gl=0, gr=0.05, vig=0.75, hud_args=None, lb=0, subs=SUBS,
                      fade=((0, 0, 0), 1 - ramp(t, T_TITLE, T_TITLE + 0.3)))
    if t >= T_END:
        lt = t - T_END
        img = card_bg()
        draw_card(img, CARD_END, lt, y0=270, size=34, dy=58)
        a = smooth(ramp(lt, 7.5, 9.0)) * (1 - smooth(ramp(lt, 16.0, 17.5)))
        if a > 0:
            put_text(img, "「你看起来……可一点都不像动画哟！」", W1 / 2, 700, 46, FONT_CJK, (0.95, 0.95, 0.9), a, shadow=0, glow=0.25 * a)
            put_text(img, "“You don't look very… animated!”", W1 / 2, 770, 26, FONT_SERIF, (0.8, 0.8, 0.78), a, shadow=0)
        return finish(img, rng, t, vh=0.3, gl=0, gr=0.03, vig=0.6, hud_args=None, lb=0, subs=SUBS,
                      fade=((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)) + 1 - ramp(t, T_END, T_END + 0.4)))
    if t >= T_DARK:
        if t < T_L9:
            img = np.zeros((H, W, 3), np.float32)
        else:
            img = level9(t, rng)
        img = film_look(img, rng, t * 2, fi, 0.6)            # still 16mm, but no longer on twos
        fade = ((0, 0, 0), 1 - smooth(ramp(t, T_L9, T_L9 + 2.5)) * (1 - smooth(ramp(t, T_END - 1.5, T_END))))
        return finish(img, rng, t, vh=0.0, gl=0, gr=0.06, vig=0.8, hud_args=None, lb=0, subs=SUBS, fade=fade)

    # ------------------------------------------------------------------ the clay world
    img = world_frame(t)
    if t < T_HILL:
        # the Hub: darkness, and a rectangular hole into colour, coming closer
        k = smoother(ramp(held(t), T_HOLE + 0.5, T_HILL - 0.3))
        hw_, hh_ = lerp(0.07, 0.75, k) * W, lerp(0.09, 0.75, k) * H
        m = np.zeros((H, W), np.float32)
        cv2.rectangle(m, (int(W / 2 - hw_), int(H / 2 - hh_)), (int(W / 2 + hw_), int(H / 2 + hh_)), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), px(2))
        hub = np.zeros_like(img) + np.array([0.006, 0.006, 0.007], np.float32)
        img = img * m[..., None] + hub * (1 - m[..., None])
    img = np.clip(img, 0, None)
    img = img / (1 + 0.15 * img)
    img = film_look(img, rng, t, fi, 1.0)
    img = bloom(l2s(np.clip(img, 0, 1)) ** 2.2, 0.75, 0.3) if False else img
    fade = ((0, 0, 0), smooth(ramp(t, T_DARK - 1.2, T_DARK)))
    return finish(img, rng, t, vh=0.0, gl=0, gr=0.045, vig=0.7, hud_args=None, lb=0, subs=SUBS, fade=fade)

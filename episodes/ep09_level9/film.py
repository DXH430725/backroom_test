"""
Ep09 — LEVEL 9 "The Suburbs": Bart0nius's last video. A forum post, then a phone held
up in an endless neighbourhood at midnight.

    python -m backrooms.render ep09_level9 --still 16,30,47,65,80,100,110,125,140,160,170
    python -m backrooms.render ep09_level9 --all [--res 4k]
"""
import math
import os

import cv2
import numpy as np

from backrooms.core import (
    W, H, W1, H1, S, FPS, FONT_CJK, FONT_SERIF, FONT_MONO, FONT_SANS_BOLD, Cam, handheld, look_at, put_text, static,
    fmt_tc, smooth, smoother, ramp, lerp, draw_card, card_bg, finish, walk_bob, subs_from_lines, px, bloom, s2l, l2s,
)
from . import scene as sc

TITLE = "LEVEL 9 —「郊区」The Suburbs"
DURATION = 200.0
HERE = os.path.dirname(os.path.abspath(__file__))

T_POST = 10.0
T_WAKE = 24.0
T_WALK = 40.0
T_TWIN = 60.0
T_DUST = 72.0
T_WATCH = 88.0
T_COLD = 118.0
T_PILLAR = 132.0
T_RUN = 148.0
T_SIGN = 158.0
T_CORRUPT = 168.0
T_END = 176.0

SUBS = subs_from_lines(os.path.join(HERE, "voice", "lines.json"))

CARD_OPEN = [
    ("M.E.G. 档案 · 补充材料 9-B", 0.4),
    ("来源 ············ 论坛帖子 / 附带视频（手机）", 1.6),
    ("作者 ············ u/Bart0nius（「探险者」小组，失踪）", 2.8),
    ("", 0),
]
CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 9「郊区」The Suburbs", 0.6),
    ("生存难度 ········ 5  ·  主要入口：在 Level 8 随机掉穿地板", 1.8),
    ("危险 ············ 街道 · 雾 · 一切会通电的东西", 3.0),
    ("实体 ············ 邻里守望 · 残形者 · 笑魇 · 剥皮者 · 猎犬", 4.2),
    ("出口 ············ 沿箭头路牌 → Level 11 · 草地小径 → 9.1 / 10", 5.4),
    ("", 0),
]

DUST_P = np.array([7.0, 161.0, 0.0])
CAR_Y = 186.0
SIGN_P = np.array([5.6, 281.5, 0.0])
PILLAR_XY = (-700.0, 260.0)


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
def phone(pos, yaw, pitch, t, amp=1.0, seed=0.0, roll=0.0, f=None):
    hy, hp, hr = handheld(t * 1.2, amp, seed)
    # phone footage: small fast jitter on top of the slow drift
    jy = 0.25 * amp * math.sin(t * 11.3 + seed) + 0.15 * amp * math.sin(t * 17.1)
    jp = 0.2 * amp * math.sin(t * 13.7 + 2 * seed)
    return Cam(pos, yaw + hy + jy, pitch + hp + jp, roll + hr, f)


def walk_y(t, t0, t1, y0, y1):
    return lerp(y0, y1, smooth(ramp(t, t0, t1)))


def cam_at(t):
    if t < T_WALK:
        # on his back on the asphalt, looking at the sky and a dying lamp; then up
        sit = smoother(ramp(t, 27.0, 30.5))
        stand = smoother(ramp(t, 30.5, 33.0))
        z = lerp(0.25, 0.9, sit) + lerp(0, 0.7, stand)
        pitch = lerp(58, 18, sit) - 22 * stand
        yaw = lerp(-128, -15, sit) + 15 * stand
        return phone((-1.0, 100.0, z), yaw, pitch, t, 1.4, 0.0, lerp(25, 0, sit))
    if t < T_TWIN:
        y = walk_y(t, T_WALK, T_TWIN, 100.0, 146.0)
        yaw = 55 * smooth(ramp(t, 44, 46.5)) * (1 - smooth(ramp(t, 51, 53.5))) - 25 * smooth(ramp(t, 55, 57)) * (1 - smooth(ramp(t, 58.5, 60)))
        return phone((1.0 + 2.5 * smooth(ramp(t, 43, 46)) * (1 - smooth(ramp(t, 52, 55))), y, 1.6 + walk_bob(t, 0.8)), yaw, -4, t, 1.1, 1.0)
    if t < T_DUST:
        y = walk_y(t, T_TWIN, T_DUST, 146.0, 154.0)
        yaw = -40 * smooth(ramp(t, 61.5, 63.5)) * (1 - smooth(ramp(t, 65.0, 66.5))) + 40 * smooth(ramp(t, 65.5, 67.0)) * (1 - smooth(ramp(t, 70, 72)))
        return phone((0.5, y, 1.6 + walk_bob(t, 0.6)), yaw, -2, t, 1.0, 2.0)
    if t < T_WATCH:
        y = walk_y(t, T_DUST, 76.0, 152.0, 158.6)
        crouch = smoother(ramp(t, 75.5, 77.0)) * (1 - smoother(ramp(t, 84.5, 86.5)))
        x = lerp(0.5, 4.6, smooth(ramp(t, T_DUST, 76.0)))
        tgt = DUST_P + np.array([0, 0, 0.1])
        yw, pt = look_at((x, y, 1.6 - 0.85 * crouch), tgt)
        yaw = lerp(10, yw, smooth(ramp(t, 73.5, 75.5)))
        pitch = lerp(-6, pt, smooth(ramp(t, 73.5, 75.5)))
        pitch = lerp(pitch, 0, smooth(ramp(t, 85, 87.5)))
        yaw = lerp(yaw, 0, smooth(ramp(t, 85, 87.5)))
        return phone((x, y, 1.6 - 0.85 * crouch), yaw, pitch, t, 0.9, 3.0)
    if t < T_COLD:
        # walk on, the phone starts to glitch; the eye comes out; run, hide behind the car
        y = walk_y(t, T_WATCH, 96.0, 158.6, 170.0)
        x = 4.6
        z = 1.6
        hide = smoother(ramp(t, 101.5, 104.0))
        y = lerp(y, 182.6, hide)
        x = lerp(x, 3.3, hide)
        z = lerp(z, 1.05, hide)
        w = watcher(t)
        yw, pt = look_at((x, y, z), w["pos"])
        k = smooth(ramp(t, 95.0, 97.5))
        yaw = lerp(0, yw * 0.7, k)
        pitch = lerp(-2, pt * 0.6, k) + 4 * hide
        amp = 1.0 + 2.5 * smooth(ramp(t, 101, 102)) * (1 - smooth(ramp(t, 104, 105)))
        return phone((x, y, z), yaw, pitch, t, amp, 4.0)
    if t < T_PILLAR:
        up = smoother(ramp(t, 119, 121))
        out = smoother(ramp(t, 120.5, 123.0))
        y = walk_y(t, 123, T_PILLAR, 182.6, 198.0)
        x = lerp(3.3, 0.6, out) + lerp(0, 1.4, smooth(ramp(t, 127, T_PILLAR)))
        return phone((x, y, lerp(1.05, 1.6, up)), -5 * math.sin(t * 0.3) - 30 * out * (1 - smooth(ramp(t, 123, 125))), -2, t, 1.3, 5.0)
    if t < T_RUN:
        # at the crossroads: look left down the cross street, then up, and up
        yaw = lerp(-5, -78, smooth(ramp(t, 133, 135.5)))
        pitch = lerp(0, 24, smooth(ramp(t, 136, 140)))
        shake = 1.0 + 4.0 * (math.exp(-((t - 138.5) / 0.5) ** 2) + math.exp(-((t - 143.5) / 0.5) ** 2))
        return phone((2.0, 199.0, 1.6), yaw, pitch, t * (1 + 0.3 * (shake > 2)), shake, 6.0)
    if t < T_CORRUPT:
        run = smooth(ramp(t, T_RUN, T_RUN + 1.5))
        y = 199.0 + 6.0 * max(0.0, t - T_RUN - 0.8) * run
        y = min(y, 273.0) if t < T_SIGN else lerp(273.0, 279.0, smooth(ramp(t, T_SIGN, 162)))
        turn = smooth(ramp(t, 163.5, 166.5))
        yaw = lerp(lerp(-60, 0, smooth(ramp(t, T_RUN, T_RUN + 1.2))), 0, 0)
        yaw = lerp(yaw, 25, smooth(ramp(t, 155.5, 157.5)) * (1 - turn)) + 90 * turn
        x = 2.0 + 3.0 * turn
        y = y + 6.0 * turn
        bob = walk_bob(t * (2.2 if t < T_SIGN else 1.0), 2.5 if t < T_SIGN else 0.6)
        amp = 3.0 if t < T_SIGN - 1 else 1.4
        return phone((x, y, 1.6 + bob), yaw, -2, t * 1.6, amp, 7.0)
    turn = smooth(ramp(t, T_CORRUPT, 171))
    return phone((5.0 + 1.3 * (t - T_CORRUPT), 285.0, 1.6 + walk_bob(t, 0.7)), 90, -1, t, 1.2, 8.0)


# ---------------------------------------------------------------------------
# the Watcher
# ---------------------------------------------------------------------------
def watcher(t):
    x = lerp(34.0, -30.0, (t - 95.0) / 23.0)
    pos = np.array([x, 203.0, 4.6 + 0.3 * math.sin(t * 0.9)])
    # gaze: down the cross street ahead of it, sweeping; at 109–113 it looks down the avenue toward the car
    ahead = np.array([-1.0, 0.0, -0.25])
    toward = np.array([3.0, CAR_Y, 0.8]) - pos
    toward /= np.linalg.norm(toward)
    k = smooth(ramp(t, 108.5, 110.0)) * (1 - smooth(ramp(t, 113.0, 114.8)))
    sweep = np.array([math.sin(t * 0.7) * 0.4, 0.0, -0.1])
    look = ahead + sweep
    look = look / np.linalg.norm(look)
    look = look * (1 - k) + toward * k
    look /= np.linalg.norm(look)
    tgt = pos + look * (pos[2] / max(0.15, -look[2]))
    tgt[2] = 0.0
    if k > 0.5:
        tgt = np.array([3.0, CAR_Y, 0.9])
    beam = smooth(ramp(t, 97.5, 99.0)) * (1 - smooth(ramp(t, 116.5, 118.0)))
    return dict(pos=pos, radius=2.2, look=look, beam=beam, target=tgt, vel=np.array([-2.8, 0, 0]))


# ---------------------------------------------------------------------------
# lamps
# ---------------------------------------------------------------------------
def _nearest_lamp(x, y):
    d = [(L[0] - x) ** 2 + (L[1] - y) ** 2 for L in sc.LAMPS]
    return int(np.argmin(d))


L_WAKE = _nearest_lamp(-4.9, 98.0)
L_STEADY = _nearest_lamp(4.9, 136.0)
L_DUST = _nearest_lamp(-4.9, 174.0)
L_FOG = _nearest_lamp(4.9, 212.0)
L_SIGN = _nearest_lamp(4.9, 288.0)


def flicker(t, seed, rate=1.0, duty=0.6):
    v = math.sin(t * 7.3 * rate + seed) + 0.6 * math.sin(t * 23.1 * rate + 2 * seed) + 0.4 * math.sin(t * 2.1 * rate + seed * 3)
    return 1.0 if v > (1 - 2 * duty) else 0.08


def lamp_states(t):
    st = {}
    if t < T_WATCH + 4:
        st[L_WAKE] = flicker(t, 1.0, 0.7, 0.55) * 0.9
        st[L_STEADY] = 0.55
        st[L_DUST] = flicker(t, 4.0, 1.3, 0.35) * 0.8
    if T_WATCH < t < T_WATCH + 6:            # the Watch kills the power that's left
        for k in list(st):
            st[k] = st[k] * (rng_flick(t) if t < T_WATCH + 4 else 0)
    if T_COLD + 4 < t < T_RUN + 10:
        st[L_FOG] = 0.8 * smooth(ramp(t, T_COLD + 4, T_COLD + 4.5)) * flicker(t, 7.0, 0.4, 0.8)
    if t > T_SIGN - 6:
        st[L_SIGN] = 0.9 * flicker(t, 9.0, 0.5, 0.75)
    return st


def rng_flick(t):
    return 1.0 if math.sin(t * 41.0) > 0.2 else 0.0


# ---------------------------------------------------------------------------
# 2D pieces: the forum post, the phone HUD, the arrow sign, corrupt frames
# ---------------------------------------------------------------------------
POST_TITLE = "fell through the floor in level 8??"
POST_BODY = ["ok so i was with the explorers in the caves and the floor just",
             "went. now i'm in like a neighborhood?? it's night. every",
             "streetlight is off. nobody's here. houses everywhere.",
             "i think i'm dreaming lol. video below"]


def forum(t):
    img = np.zeros((H, W, 3), np.float32) + np.array([0.012, 0.013, 0.016], np.float32)
    lt = t - T_POST
    cw, ch = 1180, 640
    x0, y0 = (W1 - cw) / 2, 200
    ov = img.copy()
    cv2.rectangle(ov, (px(x0), px(y0)), (px(x0 + cw), px(y0 + ch)), (0.035, 0.037, 0.045), -1)
    cv2.rectangle(ov, (px(x0), px(y0)), (px(x0 + cw), px(y0 + ch)), (0.08, 0.085, 0.1), max(1, px(2)))
    img = ov
    put_text(img, "WANDERER NETWORK  ·  /levels/8", x0 + 40, y0 + 46, 24, FONT_MONO, (0.45, 0.5, 0.55), 1.0, anchor="left", shadow=0)
    put_text(img, "u/Bart0nius  ·  [M.E.G. archive copy]", x0 + 40, y0 + 92, 26, FONT_MONO, (0.95, 0.55, 0.35), 1.0, anchor="left", shadow=0)
    put_text(img, POST_TITLE, x0 + 40, y0 + 160, 46, FONT_SANS_BOLD, (0.92, 0.93, 0.95), 1.0, anchor="left", shadow=0)
    chars = int(max(0, lt - 1.0) * 28)
    yy = y0 + 240
    for line in POST_BODY:
        if chars <= 0:
            break
        put_text(img, line[:chars], x0 + 40, yy, 30, FONT_SANS_BOLD, (0.7, 0.72, 0.76), 1.0, anchor="left", shadow=0)
        chars -= len(line)
        yy += 48
    if lt > 7.0:
        # the attached video: a black thumbnail with a play triangle, then it opens
        a = smooth(ramp(lt, 7.0, 7.6))
        bx, by, bw, bh = x0 + 40, y0 + 440, 300, 168
        cv2.rectangle(img, (px(bx), px(by)), (px(bx + bw), px(by + bh)), (0.0, 0.0, 0.0), -1)
        tri = np.array([[bx + bw / 2 - 22, by + bh / 2 - 26], [bx + bw / 2 - 22, by + bh / 2 + 26], [bx + bw / 2 + 28, by + bh / 2]]) * S
        cv2.fillPoly(img, [tri.astype(np.int32)], (0.8 * a, 0.8 * a, 0.8 * a), cv2.LINE_AA)
        put_text(img, "VID_0412.mp4  ·  3:12", bx + bw + 30, by + bh / 2 + 10, 26, FONT_MONO, (0.5, 0.52, 0.56), a, anchor="left", shadow=0)
    if lt > 12.6:
        k = smoother(ramp(lt, 12.6, 14.0))
        img = img * (1 - k)
    return img


def phone_hud(img, t, batt, glitchy=0.0):
    el = max(0.0, t - T_WAKE)
    c = (0.96, 0.96, 0.96)
    if (t % 1.0) < 0.6:
        cv2.circle(img, (px(92), px(80)), px(13), (0.9, 0.12, 0.1), -1, cv2.LINE_AA)
    tc = "%02d:%02d" % (int(el) // 60, int(el) % 60)
    if glitchy > 0.5 and (int(t * 13) % 3 == 0):
        tc = "%02d:%02d" % ((int(el) * 7919) % 60, (int(el) * 104729) % 60)
    put_text(img, tc, 118, 92, 34, FONT_MONO, c, 1.0, anchor="left", shadow=0.6)
    # battery
    bx, by, bw, bh = 1736, 66, 64, 28
    cv2.rectangle(img, (px(bx), px(by)), (px(bx + bw), px(by + bh)), c, max(1, px(2)))
    cv2.rectangle(img, (px(bx + bw), px(by + 8)), (px(bx + bw + 5), px(by + bh - 8)), c, -1)
    fill = max(0.02, batt)
    colb = (0.95, 0.25, 0.2) if batt < 0.2 else c
    cv2.rectangle(img, (px(bx + 4), px(by + 4)), (px(bx + 4 + (bw - 8) * fill), px(by + bh - 4)), colb, -1)
    put_text(img, "%d%%" % int(batt * 100), bx - 16, by + 23, 26, FONT_MONO, c, 1.0, anchor="right", shadow=0.6)
    put_text(img, "1×", W1 / 2, 1010, 26, FONT_MONO, (0.9, 0.9, 0.9), 0.8, shadow=0.6)


def batt_at(t):
    b = 0.31 - 0.0006 * max(0, t - T_WAKE)
    if t > T_WATCH:
        b -= 0.18 * smooth(ramp(t, T_WATCH + 2, T_COLD))
    if t > T_CORRUPT:
        b -= 0.1 * ramp(t, T_CORRUPT, T_END)
    return max(0.01, b)


def sign_art():
    a = np.zeros((300, 300, 3), np.float32)
    # a yellow diamond warning sign with a heavy black arrow
    pts = np.array([[150, 6], [294, 150], [150, 294], [6, 150]], np.int32)
    cv2.fillPoly(a, [pts], (0.95, 0.78, 0.12), cv2.LINE_AA)
    cv2.polylines(a, [pts], True, (0.05, 0.05, 0.05), 8, cv2.LINE_AA)
    arrow = np.array([[70, 132], [168, 132], [168, 92], [236, 150], [168, 208], [168, 168], [70, 168]], np.int32)
    cv2.fillPoly(a, [arrow], (0.04, 0.04, 0.04), cv2.LINE_AA)
    return a


SIGN = None


def eye_frame(seed):
    """an 'unfamiliar image': a close, staring eye, the kind of thing the Watch leaves in a phone's memory"""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32) + 0.03
    cx, cy = W // 2 + int(rng.integers(-px(200), px(200))), H // 2 + int(rng.integers(-px(80), px(80)))
    R = px(rng.integers(260, 420))
    cv2.circle(img, (cx, cy), R, (0.75, 0.72, 0.7), -1, cv2.LINE_AA)
    for _ in range(40):
        a = rng.uniform(0, 2 * math.pi)
        r0 = R * rng.uniform(0.5, 0.95)
        pts = [(cx + math.cos(a) * R, cy + math.sin(a) * R)]
        for k in range(6):
            r = R - (R - r0) * (k + 1) / 6
            aa = a + rng.normal(0, 0.06)
            pts.append((cx + math.cos(aa) * r, cy + math.sin(aa) * r))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0.3, 0.05, 0.06), max(1, px(3)), cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(R * 0.42), (0.22, 0.26, 0.2), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(R * 0.2), (0.0, 0.0, 0.0), -1, cv2.LINE_AA)
    cv2.circle(img, (cx - int(R * 0.1), cy - int(R * 0.1)), int(R * 0.05), (0.9, 0.9, 0.9), -1, cv2.LINE_AA)
    return img


def phone_post(img, rng, t, gain, glitch_amt):
    """auto-exposure in the dark, crushed codec blocks, chroma noise, oversharpening"""
    img = img * gain
    img = img / (1 + img * 0.6)                  # soft highlight roll-off
    img = np.clip(img, 0, 1)
    img = l2s(img)
    # chroma noise at quarter res (night mode) and luma grain
    cn = rng.normal(0, 0.035, (H // 4, W // 4, 3)).astype(np.float32)
    cn = cv2.resize(cn, (W, H), interpolation=cv2.INTER_LINEAR)
    img = img + cn * (1 - img) * 0.9
    # oversharpening
    bl = cv2.GaussianBlur(img, (0, 0), px(2))
    img = img + (img - bl) * 0.6
    # codec: blocky when the picture moves or the Watch is near
    if glitch_amt > 0.05:
        bs = px(16)
        small = cv2.resize(img, (W // bs, H // bs), interpolation=cv2.INTER_AREA)
        blocky = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (rng.random((H // bs + 1, W // bs + 1)) < glitch_amt * 0.5).astype(np.float32)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        img = img * (1 - m) + blocky * m
        # torn rows
        for _ in range(int(glitch_amt * 12)):
            y0 = int(rng.integers(0, H - px(40)))
            hh = int(rng.integers(px(4), px(40)))
            sh = int(rng.integers(-px(120), px(120)) * glitch_amt)
            img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], sh, axis=1)
        # chroma split
        sh = int(px(10) * glitch_amt)
        if sh:
            img[..., 0] = np.roll(img[..., 0], sh, axis=1)
            img[..., 2] = np.roll(img[..., 2], -sh, axis=1)
    return np.clip(img, 0, 1)


def breath(img, t, t0, amount):
    """his breath, visible in the sudden cold, drifting up across the lens bottom"""
    if amount <= 0:
        return img
    ph = ((t - t0) % 3.2) / 3.2
    if ph > 0.55:
        return img
    g = np.zeros((H // 4, W // 4), np.float32)
    k = ph / 0.55
    cx, cy = W // 8 + int(px(40) * math.sin(t)) // 4, H // 4 - int(px(60 + 220 * k)) // 4
    cv2.ellipse(g, (cx, cy), (int(px(120 + 300 * k) // 4), int(px(60 + 160 * k) // 4)), 0, 0, 360, 1.0, -1)
    g = cv2.GaussianBlur(g, (0, 0), px(40) / 4)
    g = cv2.resize(g, (W, H)) * (1 - k) * 0.35 * amount
    return img + g[..., None] * np.array([0.6, 0.62, 0.66], np.float32)


# ---------------------------------------------------------------------------
def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 9)
    vh, gl, gr = 0.0, 0.0, 0.03
    fade = None
    subs = SUBS

    if t < T_POST:
        img = card_bg()
        draw_card(img, CARD_OPEN, t)
        return finish(img, rng, t, vh=0.3, gl=0, gr=0.03, vig=0.6, hud_args=None, lb=0, subs=subs,
                      fade=((0, 0, 0), 1 - ramp(t, 0, 0.6) + ramp(t, T_POST - 0.8, T_POST)))
    if t < T_WAKE:
        img = forum(t)
        return finish(img, rng, t, vh=0.0, gl=0, gr=0.015, vig=0.3, hud_args=None, lb=0, subs=subs,
                      fade=((0, 0, 0), 1 - ramp(t, T_POST, T_POST + 0.6)))
    if t >= T_END:
        lt = t - T_END
        img = card_bg()
        draw_card(img, CARD_END, lt, y0=270, size=34, dy=58)
        a = smooth(ramp(lt, 7.5, 9.0)) * (1 - smooth(ramp(lt, 16.0, 17.5)))
        if a > 0:
            put_text(img, "「如果有人看到这个——」", W1 / 2, 700, 46, FONT_CJK, (0.95, 0.85, 0.55), a, shadow=0, glow=0.3 * a)
            put_text(img, "“If anyone sees this—”", W1 / 2, 770, 26, FONT_SERIF, (0.8, 0.75, 0.6), a, shadow=0)
        return finish(img, rng, t, vh=0.3, gl=0, gr=0.03, vig=0.6, hud_args=None, lb=0, subs=subs,
                      fade=((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)) + 1 - ramp(t, T_END, T_END + 0.4)))

    # ------------------------------------------------------------------ the phone footage
    cam = cam_at(t)
    st = lamp_states(t)
    lit = [sc.lamp_light(i, b) for i, b in st.items() if b > 0.05]
    flash = 1.0 if (t > 28.6 and t < 103.6) or (T_COLD + 2 < t < T_PILLAR + 1) or t > T_RUN + 0.5 else 0.0
    if 28.6 < t < 29.2:
        flash = 1.0 if (int(t * 20) % 3) else 0.0
    w = watcher(t) if 95.0 < t < 118.5 else None
    fogd = lerp(140.0, 24.0, smooth(ramp(t, T_COLD + 1, T_COLD + 12))) if t < T_RUN + 4 else lerp(24.0, 70.0, smooth(ramp(t, T_RUN + 4, T_SIGN + 4)))
    fogc = (0.010, 0.011, 0.014)
    if t > T_COLD:
        fogc = tuple(np.array([0.010, 0.011, 0.014]) * (1 - smooth(ramp(t, T_COLD + 2, T_COLD + 10))) + np.array([0.022, 0.023, 0.026]) * smooth(ramp(t, T_COLD + 2, T_COLD + 10)))
    img, d, pm = sc.render(cam, t, flash=flash, lamps=lit, watcher=w, fog=fogd, fogcol=fogc)
    img = sc.puddle_reflect(img, d, pm, cam, t)
    img = sc.draw_trees(img, d, cam, flash, lit)
    img = sc.draw_power(img, d, cam, flash, lit)
    img = sc.draw_lamps(img, d, cam, st, flash, lit)
    if 60 < t < T_WATCH + 10:
        img = sc.dust_pile(img, d, cam, DUST_P, flash, lit)
    global SIGN
    if t > T_RUN:
        if SIGN is None:
            SIGN = sign_art()
        p = SIGN_P
        pole_top = p + np.array([0, 0, 2.9])
        b8 = sc.to8(img)
        out = b8.copy()
        sc._line3(out, d, cam, p, pole_top, (70, 70, 72), 0.07, 4)
        img = sc.from8(img, b8, out)
        lt_ = sc.light_at(pole_top, cam, flash, lit)
        shade = float(np.clip(lt_.mean() * 1.4, 0.03, 1.2))
        c = pole_top + np.array([-0.02, 0, 0.0])
        img = sc.quad(img, d, cam, [c + [-0.45, 0, 0.45], c + [0.45, 0, 0.45], c + [0.45, 0, -0.45], c + [-0.45, 0, -0.45]], SIGN, shade)
    if w is not None:
        img = sc.beam(img, d, cam, w)
        img = sc.veins(img, d, cam, w, t)
    if t > T_COLD + 6:
        img = sc.mangled(img, d, cam, t, PILLAR_XY, smooth(ramp(t, T_COLD + 6, T_PILLAR + 4)))
    # the cold
    if T_COLD + 1 < t < T_RUN + 8:
        img = breath(img, t, T_COLD + 1, smooth(ramp(t, T_COLD + 1, T_COLD + 3)) * (1 - smooth(ramp(t, T_RUN + 5, T_RUN + 8))))
    img = bloom(img, 0.5, 0.6)

    # glitching: near the Watch the picture corrupts; at the end it comes apart
    g = 0.0
    g += 0.15 * smooth(ramp(t, 90.0, 92.0)) * (1 - smooth(ramp(t, 118, 122)))
    g += 0.45 * smooth(ramp(t, 101, 104)) * (1 - smooth(ramp(t, 116, 119))) * (0.4 + 0.6 * (rng.random() < 0.5))
    g += 0.3 * (rng.random() < 0.05) * (t > 120)
    g += smooth(ramp(t, T_CORRUPT, T_END - 1)) * (0.5 + 0.5 * rng.random())
    if (t > T_CORRUPT and rng.random() < 0.06 + 0.25 * ramp(t, T_CORRUPT, T_END)) or (101 < t < 117 and rng.random() < 0.025):
        img = eye_frame(int(t * 3)) * (0.6 + 0.4 * rng.random()) + img * 0.2
    gain = 2.2 if flash > 0 else 3.4
    img = phone_post(img, rng, t, gain, min(1.0, g))
    phone_hud(img, t, batt_at(t), g)
    if t > T_CORRUPT + 2:
        k = smooth(ramp(t, T_CORRUPT + 2, T_END - 0.3)) * 0.9
        img = img * (1 - k) + static(rng, 0.8) * k
    a = smooth(ramp(t, 34.0, 35.2)) * (1 - smooth(ramp(t, 38.6, 39.8)))
    if a > 0:
        put_text(img, "LEVEL 9", W1 / 2, H1 / 2 - 40, 118, FONT_SERIF, (1, 1, 1), a, tracking=0.32, shadow=0.6, glow=0.15)
        put_text(img, "T H E   S U B U R B S", W1 / 2, H1 / 2 + 62, 34, FONT_SERIF, (1, 1, 1), a * 0.95, shadow=0.6)
        put_text(img, "郊 区", W1 / 2, H1 / 2 + 118, 34, FONT_CJK, (1, 1, 1), a * 0.9, shadow=0.6)
    fade = ((0, 0, 0), 1 - ramp(t, T_WAKE, T_WAKE + 0.4) + ramp(t, T_END - 0.3, T_END))
    gl = 0.05 * g
    return finish(img, rng, t, vh=0.0, gl=gl, gr=0.03, vig=0.35, hud_args=None, lb=0, subs=subs, fade=fade)

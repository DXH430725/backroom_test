"""
Ep06 — LEVEL 11 "The City That Never Sleeps": an M.E.G. Camp Amber orientation tape
(1990s corporate-video style), and what was recorded after it.

    python -m backrooms.render ep06_level11 --still 15,30,60,80,100,120,140,160,180
    python -m backrooms.render ep06_level11 --all [--res 4k]
"""
import math
import os

import cv2
import numpy as np

from backrooms.core import (
    W, H, W1, H1, S, FPS, F0, FONT_CJK, FONT_SERIF, FONT_MONO, FONT_SANS_BOLD, Cam, handheld, look_at,
    put_text, static, fmt_tc, smooth, smoother, ramp, lerp, draw_card, card_bg, finish, walk_bob,
    subs_from_lines, px, bloom, l2s, s2l,
)
from . import scene as sc

TITLE = "LEVEL 11 —「不夜城」The City That Never Sleeps"
DURATION = 200.0
HERE = os.path.dirname(os.path.abspath(__file__))

T_LOGO = 10.0
T_FLY = 19.0
T_STREET = 44.0
T_PARK = 70.0
T_CANAL = 92.0
T_CAPITAL = 106.0
T_ENT = 128.0
T_NIGHT = 148.0
T_OUTRO = 164.0
T_AFTER = 172.0
T_END = 186.0

SUBS = subs_from_lines(os.path.join(HERE, "voice", "lines.json"))

CARD_OPEN = [
    ("M.E.G. 视听档案馆", 0.4),
    ("录像 ············ 营地琥珀 新人导览 第 11 卷", 1.4),
    ("用途 ············ 内部教育", 2.6),
    ("附注 ············ 片尾之后仍有记录", 3.8),
    ("", 0),
]
CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 11「不夜城」The City That Never Sleeps", 0.6),
    ("生存难度 ········ 2  ·  安全 · 不稳定 · 实体稀少", 1.8),
    ("人口 ············ 约 129,500", 2.8),
    ("据点 ············ 首都 · M.E.G. 基地贝塔 / 营地琥珀 · 新时代广场 · 底比斯", 3.8),
    ("出口 ············ 几乎通往所有已知层级 · 多为隐藏、单向", 5.0),
    ("", 0),
]

X0 = sc.street_coord(0)            # a north-south avenue
Y0 = sc.street_coord(1)            # an east-west street
CANAL_X = sc.street_coord(-1)      # canals replace every 5th avenue

AMBER = (1.0, 0.62, 0.12)
TEAL = (0.1, 0.55, 0.62)


def lower_third(img, t, t0, t1, title, sub):
    a = smooth(ramp(t, t0, t0 + 0.5)) * (1 - smooth(ramp(t, t1 - 0.5, t1)))
    if a <= 0:
        return
    slide = (1 - smooth(ramp(t, t0, t0 + 0.6))) * 600
    x0, y0 = px(110 - slide), px(120)
    bw = px(720)
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + bw, y0 + px(70)), AMBER, -1)
    cv2.rectangle(ov, (x0, y0 + px(70)), (x0 + int(bw * 0.8), y0 + px(110)), TEAL, -1)
    img[:] = img * (1 - a) + ov * a
    fnt = FONT_CJK if any(ord(ch) > 127 for ch in title) else FONT_SANS_BOLD
    put_text(img, title, 130 - slide, 155, 40, fnt, (0.08, 0.06, 0.04), a, shadow=0, anchor="left")
    put_text(img, sub, 130 - slide, 220, 24, FONT_CJK, (1, 1, 1), a, shadow=0, anchor="left")


def logo_card(t):
    lt = t - T_LOGO
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = np.ones((H, W, 3), np.float32) * (np.array([0.05, 0.12, 0.3], np.float32) * (1 - yy) + np.array([0.0, 0.03, 0.1], np.float32) * yy)
    # sweeping 90s light bars
    for k in range(5):
        x = ((lt * 0.35 + k * 0.23) % 1.4 - 0.2) * W
        cv2.rectangle(img, (int(x), 0), (int(x + px(60 + 30 * k)), H), (0.1, 0.25, 0.5), -1)
    img = cv2.GaussianBlur(img, (0, 0), px(18))
    a = smooth(ramp(lt, 0.4, 1.6))
    sc_ = 0.6 + 0.4 * smoother(ramp(lt, 0.3, 1.8))
    put_text(img, "M.E.G.", W1 / 2, H1 / 2 - 90, int(150 * sc_), FONT_SANS_BOLD, AMBER, a, shadow=0.6, glow=0.4 * a)
    put_text(img, "CAMP AMBER  ·  ORIENTATION SERIES", W1 / 2, H1 / 2 + 40, 34, FONT_SANS_BOLD, (1, 1, 1), smooth(ramp(lt, 1.6, 2.4)), tracking=0.15, shadow=0.5)
    put_text(img, "营 地 琥 珀 · 新 人 导 览", W1 / 2, H1 / 2 + 100, 32, FONT_CJK, (0.85, 0.9, 1), smooth(ramp(lt, 2.2, 3.0)), shadow=0.5)
    put_text(img, "TAPE 11", W1 / 2, H1 / 2 + 190, 46, FONT_MONO, TEAL, smooth(ramp(lt, 3.0, 3.8)), shadow=0.4)
    return img


def cam_fly(t):
    lt = t - T_FLY
    s = smoother(ramp(t, T_FLY, T_STREET))
    p0 = np.array([-2300.0, -2500.0, 620.0])
    p1 = np.array([-700.0, -900.0, 330.0])
    pos = p0 + (p1 - p0) * s
    yaw, pitch = look_at(pos, (0, 0, 60))
    return Cam(pos, yaw + 6 * math.sin(lt * 0.2), pitch + 2, -3 * math.sin(lt * 0.15))


def cam_street(t):
    lt = t - T_STREET
    hy, hp, hr = handheld(t, 0.4, 2.0)              # a steadicam-ish presenter shot
    y = Y0 + 20 + 1.2 * lt
    yaw = 0 + 42 * smooth(ramp(t, 54.5, 59.0)) * (1 - smooth(ramp(t, 64.5, 68.5)))
    pitch = 4 + 22 * smooth(ramp(t, 55.0, 59.0)) * (1 - smooth(ramp(t, 64.5, 68.5)))
    return Cam((X0 + 1.5, y, 1.7), yaw + hy, pitch + hp, hr)


# the car park demonstration: one block is a parking lot
def _make_lot():
    # turn one city block east of the avenue into an open car park
    i0, j0 = sc.CITY.cell_of(X0 + 30.0, Y0 + 230.0)
    bi, bj = (i0 // sc.B) * sc.B, (j0 // sc.B) * sc.B
    sc.CITY.kind[bi + 1:bi + sc.B, bj + 1:bj + sc.B] = sc.STREET
    sc.CITY.h[bi + 1:bi + sc.B, bj + 1:bj + sc.B] = 0.0
    cx = sc.CITY.origin[0] + (bi + sc.B / 2 + 0.5) * sc.C
    cy = sc.CITY.origin[1] + (bj + sc.B / 2 + 0.5) * sc.C
    return np.array([cx, cy])


LOT = _make_lot()
CARS_A = [(LOT[0] - 12 + 5.0 * k, LOT[1] - 6, math.pi / 2, c) for k, c in enumerate(
    [(0.5, 0.1, 0.1), (0.15, 0.2, 0.5), (0.8, 0.8, 0.78), (0.1, 0.1, 0.1), (0.2, 0.4, 0.25)])]
CARS_B = [(LOT[0] - 14 + 4.6 * k, LOT[1] + 7, -math.pi / 2, c) for k, c in enumerate(
    [(0.8, 0.8, 0.78), (0.6, 0.45, 0.1), (0.5, 0.1, 0.1), (0.12, 0.12, 0.14), (0.15, 0.2, 0.5), (0.45, 0.45, 0.5)])]


def cam_park(t):
    hy, hp, hr = handheld(t, 0.4, 3.0)
    pos = np.array([LOT[0] - 3, LOT[1] - 34, 1.7])
    away = smoother(ramp(t, 76.0, 77.6)) * (1 - smoother(ramp(t, 79.0, 80.6)))
    yaw0, pitch0 = look_at(pos, (*LOT, 0.5))
    yaw = yaw0 - 150 * away
    yaw += 34 * smooth(ramp(t, 85.0, 87.5))                  # then look at the billboard
    return Cam(tuple(pos), yaw + hy, pitch0 + 2 * smooth(ramp(t, 85.0, 87.5)) + hp, hr)


BB = [(LOT[0] + 22, LOT[1] - 11, 16), (LOT[0] + 22, LOT[1] + 5, 16), (LOT[0] + 22, LOT[1] + 5, 8), (LOT[0] + 22, LOT[1] - 11, 8)]


def ad_art(kind):
    a = np.zeros((300, 600, 3), np.uint8)
    D = cv2.FONT_HERSHEY_DUPLEX
    if kind == 0:   # Almond Water
        a[:] = (235, 224, 190)
        cv2.circle(a, (470, 150), 110, (140, 96, 50), -1, cv2.LINE_AA)
        cv2.putText(a, "ALMOND", (40, 140), D, 2.6, (76, 50, 26), 5, cv2.LINE_AA)
        cv2.putText(a, "WATER", (40, 230), D, 2.6, (76, 50, 26), 5, cv2.LINE_AA)
    elif kind == 1:  # B.N.T.G. — New Times Square
        a[:] = (26, 30, 90)
        cv2.putText(a, "B.N.T.G.", (30, 130), D, 3.0, (255, 204, 50), 6, cv2.LINE_AA)
        cv2.putText(a, "PRESSES ACCEPTED", (40, 230), D, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    else:            # something that should not be an advert
        a[:] = (242, 242, 242)
        cv2.putText(a, "YOU ARE", (40, 130), D, 2.6, (12, 12, 12), 5, cv2.LINE_AA)
        cv2.putText(a, "HOME NOW", (40, 230), D, 2.6, (150, 12, 12), 5, cv2.LINE_AA)
    return a.astype(np.float32) / 255.0


ADS = [ad_art(0), ad_art(1), ad_art(2)]


def cam_canal(t):
    lt = t - T_CANAL
    hy, hp, hr = handheld(t, 0.3, 4.0)
    return Cam((CANAL_X - 4.5, Y0 + 60 + 2.2 * lt, 4.0), 0 + hy, -6 + hp, hr)


def cam_capital(t):
    lt = t - T_CAPITAL
    if t < 118.5:
        hy, hp, hr = handheld(t, 0.5, 5.0)
        return Cam((X0 + 2, Y0 - 6 + 0.8 * lt, 1.7), 20 - 30 * smooth(ramp(t, 108, 117)) + hy, 3 + hp, hr)
    s = smoother(ramp(t, 118.5, T_ENT))
    pos = np.array([X0 - 700 + 260 * s, Y0 - 900 + 300 * s, 300 - 60 * s])
    yaw, pitch = look_at(pos, (0, 0, 120))
    return Cam(tuple(pos), yaw, pitch + 4, 1.5)


def cam_ent(t):
    hy, hp, hr = handheld(t, 0.6 + 1.5 * ramp(t, 140.5, 143), 6.0)
    y = Y0 + 300 + 0.5 * (t - T_ENT)
    yaw = 0 - 18 * smooth(ramp(t, 133, 137)) * (1 - smooth(ramp(t, 140, 142)))
    return Cam((X0 + 1.0, y, 1.7), yaw + hy, 1 + hp, hr)


def cam_night(t):
    s = smoother(ramp(t, T_NIGHT, T_OUTRO))
    pos = np.array([-1800 + 900 * s, 1600 - 700 * s, 520 - 120 * s])
    yaw, pitch = look_at(pos, (0, 0, 40))
    return Cam(tuple(pos), yaw, pitch + 3, 2 * math.sin(t * 0.2))


def cam_after(t):
    hy, hp, hr = handheld(t, 0.05, 8.0)           # a camera left running on the kerb
    return Cam((X0 - 4.6, Y0 + 520, 0.5), 8 + hy, 3 + hp, hr)


def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 6)
    vh, gl, gr = 0.38, 0.0, 0.02
    lb, fade = 0.0, None
    hud = None
    blm = 0.35

    if t < T_LOGO:
        img = card_bg()
        draw_card(img, CARD_OPEN, t)
        vh, gr, blm = 0.3, 0.03, 0
        fade = ((0, 0, 0), 1 - ramp(t, 0, 0.6) + ramp(t, T_LOGO - 0.8, T_LOGO))

    elif t < T_FLY:
        img = logo_card(t)
        fade = ((1, 1, 1), 1 - smooth(ramp(t, T_LOGO, T_LOGO + 0.4))) if t < T_LOGO + 0.4 else ((0, 0, 0), smooth(ramp(t, T_FLY - 0.6, T_FLY)))

    elif t < T_STREET:
        cam = cam_fly(t)
        img, d = sc.render(cam, t, haze=5200)
        a = smooth(ramp(t, 24.5, 25.8)) * (1 - smooth(ramp(t, 29.6, 30.8)))
        if a > 0:
            put_text(img, "LEVEL 11", W1 / 2, H1 / 2 - 40, 118, FONT_SERIF, (1, 1, 1), a, tracking=0.32, shadow=0.6, glow=0.2)
            put_text(img, "THE  CITY  THAT  NEVER  SLEEPS", W1 / 2, H1 / 2 + 62, 34, FONT_SERIF, (1, 1, 1), a * 0.95, shadow=0.6)
            put_text(img, "不 夜 城", W1 / 2, H1 / 2 + 118, 34, FONT_CJK, (1, 1, 1), a * 0.9, shadow=0.6)
        fade = ((0, 0, 0), 1 - smooth(ramp(t, T_FLY, T_FLY + 1.0)))

    elif t < T_PARK:
        cam = cam_street(t)
        img, d = sc.render(cam, t)
        lower_third(img, t, 47.5, 53.5, "LEVEL 11", "一座以普通西方城市为蓝本的、空无一人的大都会")
        lower_third(img, t, 57.0, 63.5, "MIRROR BUILDINGS", "约三分之一：无法进入 · 无法破坏")

    elif t < T_CANAL:
        cam = cam_park(t)
        img, d = sc.render(cam, t)
        cars = CARS_A if t < 78.3 else CARS_B
        for x, y, yw, c in sorted(cars, key=lambda q: -math.hypot(q[0] - cam.pos[0], q[1] - cam.pos[1])):
            sc.car(img, d, cam, x, y, yw, c)
        sc.billboard(img, d, cam, BB, ADS[0] if t < 78.3 else ADS[1])
        lower_third(img, t, 83.0, 89.5, "FLUIDITY  流动性", "无人观察时：车辆转移 · 广告更换 · 道路施工出现又消失")
        gl = 0.25 * math.exp(-((t - 78.3) / 0.06) ** 2)

    elif t < T_CAPITAL:
        cam = cam_canal(t)
        img, d = sc.render(cam, t)
        lower_third(img, t, 95.0, 101.0, "CANALS  运河", "永远不会到达海洋，有时转入地下")

    elif t < T_ENT:
        cam = cam_capital(t)
        img, d = sc.render(cam, t, haze=3200)
        if t < 118.5:
            r2 = np.random.default_rng(118)
            crowd = [(X0 + r2.uniform(-5, 5), Y0 + r2.uniform(5, 60), (r2.uniform(0.1, 0.5), r2.uniform(0.1, 0.4), r2.uniform(0.1, 0.4))) for _ in range(40)]
            for k, (x, y, c) in enumerate(sorted(crowd, key=lambda q: -math.hypot(q[0] - cam.pos[0], q[1] - cam.pos[1]))):
                sc.person(img, d, cam, x + 0.6 * math.sin(t * 0.7 + k), y + 0.8 * math.sin(t * 0.5 + k * 1.3), 1.65 + 0.15 * ((k * 7) % 3) / 2, c)
        else:
            pass
        lower_third(img, t, 110.5, 117.5, "POPULATION  人口", "约 129,500 · 首都约 45,000")
        lower_third(img, t, 120.0, 126.5, "BASE BETA → CAMP AMBER", "M.E.G. 唯一不在室内层级的主要基地 · 高架步道相连")

    elif t < T_NIGHT:
        cam = cam_ent(t)
        img, d = sc.render(cam, t)
        hound_x = X0 - 6 + 4.0 * ramp(t, 129.5, 136.5) * 3
        sc.hound(img, d, cam, hound_x, Y0 + 320, 0.0, t)
        sc.person(img, d, cam, X0 + 4.5, Y0 + 312, 1.75, (0.55, 0.35, 0.45), face=(0.86, 0.8, 0.76))
        lower_third(img, t, 132.5, 139.5, "LEVEL 11 EFFECT", "敌对实体不太愿意发动攻击")
        if 141.0 < t < 145.0:
            # the presenter's smile slips; so does the tape
            gl = 0.3 + 0.3 * (rng.random() < 0.4)
            vh = 0.7
            img = img * (1 - 0.35 * smooth(ramp(t, 141, 142)) * (1 - smooth(ramp(t, 144, 145))))
        gl += 0.4 * (1 - ramp(t, T_ENT, T_ENT + 0.3))

    elif t < T_OUTRO:
        cam = cam_night(t)
        img, d = sc.render(cam, t, night=1.0, haze=4500)
        blm = 0.6
        gl = 0.5 * (1 - ramp(t, T_NIGHT, T_NIGHT + 0.3))

    elif t < T_AFTER:
        img = logo_card(t + 2)
        lt = t - T_OUTRO
        put_text(img, "下 一 卷 · TAPE 12", W1 / 2, H1 - 160, 30, FONT_CJK, (1, 1, 1), smooth(ramp(lt, 3.5, 4.5)), shadow=0.5)
        fade = ((0, 0, 0), smooth(ramp(t, T_AFTER - 0.8, T_AFTER - 0.1)))

    elif t < T_END:
        # the tape keeps running after the credits: a camera on the kerb, at night
        cam = cam_after(t)
        img, d = sc.render(cam, t, night=1.0)
        cars = CARS_A if (t < 177.0 or 181.0 < t < 182.5) else CARS_B
        for x, y, yw, c in cars[:3]:
            sc.car(img, d, cam, X0 + 2.0, Y0 + 535 + (x - LOT[0]) * 2.0, math.pi / 2, c, night=1.0)
        if t > 182.5:
            sc.person(img, d, cam, X0 - 1.2, Y0 + 529, 1.75, (0.05, 0.05, 0.06), face=(0.75, 0.7, 0.66))
        gl = 0.35 * math.exp(-((t - 177.0) / 0.05) ** 2) + 0.35 * math.exp(-((t - 182.5) / 0.05) ** 2) + 0.4 * (1 - ramp(t, T_AFTER, T_AFTER + 0.4))
        gl += smooth(ramp(t, T_END - 1.0, T_END))
        vh = 0.6
        gr = 0.05
        blm = 0.6
        hud = dict(alpha=0.9, tc=fmt_tc(t - T_AFTER), level="LV-11", center=None, cam_label="CAM-??")
        fade = ((0, 0, 0), smooth(ramp(t, T_END - 0.4, T_END)) + 1 - ramp(t, T_AFTER, T_AFTER + 0.3))

    else:
        lt = t - T_END
        img = card_bg()
        draw_card(img, CARD_END, lt, y0=270, size=34, dy=58)
        vh, gr, blm = 0.3, 0.03, 0
        fade = ((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)) + 1 - ramp(t, T_END, T_END + 0.4))

    if blm:
        img = bloom(img, 0.72, blm)
    return finish(img, rng, t, vh=vh, gl=gl, gr=gr, vig=0.45, hud_args=hud, lb=lb, subs=SUBS, fade=fade)

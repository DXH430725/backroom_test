"""
Ep08 — LEVEL 8 "Cave Systems": Diver-1, continued. Out of the sea and into the rock.

    python -m backrooms.render ep08_level8 --still 18,32,62,78,92,110,120,140,160
    python -m backrooms.render ep08_level8 --all [--res 4k]
"""
import math
import os

import cv2
import numpy as np

from backrooms.core import (
    W, H, W1, H1, S, FPS, FONT_CJK, FONT_SERIF, FONT_MONO, Cam, handheld, look_at, put_text, static,
    fmt_tc, smooth, smoother, ramp, lerp, draw_card, card_bg, finish, walk_bob, subs_from_lines, px, bloom,
)
from . import scene as sc

TITLE = "LEVEL 8 —「洞穴系统」Cave Systems"
DURATION = 200.0
HERE = os.path.dirname(os.path.abspath(__file__))

T_POOL = 10.0
T_TUNNEL = 26.0
T_CHAMBER = 56.0
T_WRANGLER = 84.0
T_LOOP = 104.0
T_FOLLOW = 126.0
T_ROAD = 146.0
T_END = 176.0

SUBS = subs_from_lines(os.path.join(HERE, "voice", "lines.json"))

CARD_OPEN = [
    ("M.E.G. 视听档案馆 · 深潜记录 007（续）", 0.4),
    ("下潜员 ·········· 深潜一号", 1.6),
    ("位置 ············ 经 Level 7 海底洞穴 → 未知", 2.8),
    ("", 0),
]
CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 8「洞穴系统」Cave Systems", 0.6),
    ("生存难度 ········ 4  ·  出口 4/5 · 环境 4/5 · 实体 4/5", 1.8),
    ("危险 ············ 有毒气体 · 突发洪水 · 失温 · 坠落 · 蛛巢 · 焦油之手", 3.0),
    ("实体 ············ 掘地者 · 节肢类 · 焦油之手 · 腐化者 · 引光者", 4.2),
    ("据点 ············ M.E.G.「空巢」前哨（第九号公路）", 5.4),
    ("", 0),
]


def floor_z(y):
    return sc.floor_of(y)


def walk_cam(y, h, yaw_off=0.0, pitch=-4.0, t=0.0, amp=0.9, seed=0.0, bob=1.0):
    fz, cx = floor_z(y)
    cx2, _ = sc.path(np.array([y + 3.0]))
    yaw = math.degrees(math.atan2(float(cx2[0]) - cx, 3.0))
    pitch += 0.7 * math.degrees(math.atan2(floor_z(y + 2.5)[0] - floor_z(y - 2.5)[0], 5.0))
    hy, hp, hr = handheld(t, amp, seed)
    return Cam((cx, y, fz + h + walk_bob(t, bob)), yaw + yaw_off + hy, pitch + hp, hr)


def cam_pool(t):
    rise = smoother(ramp(t, 14.0, T_TUNNEL))
    return walk_cam(4.0 + 3.0 * rise, lerp(0.35, 1.6, rise), 0, lerp(8, -4, rise), t, 1.4, 0.3, 0.0)


def cam_tunnel(t):
    y = lerp(7.0, 44.0, smooth(ramp(t, T_TUNNEL, T_CHAMBER)))
    look = -38 * smooth(ramp(t, 29, 31)) * (1 - smooth(ramp(t, 34, 35.5))) + 40 * smooth(ramp(t, 36, 37.5)) * (1 - smooth(ramp(t, 40, 41.5)))
    return walk_cam(y, 1.55, look, -3, t, 0.9, 1.0)


def cam_chamber(t):
    if t < 66:
        y = lerp(47.0, 58.0, smooth(ramp(t, T_CHAMBER, 66)))
        return walk_cam(y, 1.7, -14 + 22 * smooth(ramp(t, 57, 65)), -9, t, 0.8, 2.0)
    y = lerp(124.0, 134.0, smooth(ramp(t, 66, T_WRANGLER)))
    pitch = -7 + 25 * smooth(ramp(t, 71, 75)) * (1 - smooth(ramp(t, 79, 83)))
    return walk_cam(y, 2.0, 8 * math.sin(t * 0.2), pitch, t, 0.7, 3.0, 0.6)


def cam_wrangler(t):
    shake = 1.0 + 3.0 * smooth(ramp(t, 85, 87)) * (1 - smooth(ramp(t, 96, 100)))
    c = walk_cam(134.0, 1.4, 0, 18 + 4 * smooth(ramp(t, 87, 90)), t * 2.0, shake, 4.0, 0.0)
    return c


GLOW_Y = 206.0


def cam_loop(t):
    if t < 114.5:
        y = lerp(198.0, 214.0, smooth(ramp(t, T_LOOP, 114.5)))
        pitch = -4 - 30 * math.exp(-((t - 106.6) / 1.2) ** 2)
    else:
        # the same passage again (the cave has quietly put us back)
        y = lerp(196.0, 204.0, smooth(ramp(t, 114.5, T_FOLLOW)))
        pitch = -6 - 9 * smooth(ramp(t, 116.0, 117.5)) * (1 - smooth(ramp(t, 122, 124)))
    return walk_cam(y, 1.55, 0, pitch, t, 0.9, 5.0)


def cam_follow(t):
    y = 232.0 + 0.5 * (t - T_FOLLOW)
    whip = smoother(ramp(t, 131.2, 131.9)) * (1 - smoother(ramp(t, 134.5, 135.3)))
    return walk_cam(y, 1.55, 180 * whip, -3, t * (1 + whip), 1.0 + 2.5 * whip, 6.0, 0.4)


def cam_road(t):
    y = lerp(300.0, 352.0, smooth(ramp(t, T_ROAD, T_END - 2)))
    return walk_cam(y, 1.6, -6 * smooth(ramp(t, 152, 156)) * (1 - smooth(ramp(t, 160, 163))), -2, t, 0.8, 7.0)


def chemlight(img, depth, cam, p, t, on=1.0):
    q = cam.project(np.array([p, p + np.array([0.15, 0.05, 0.02])]))
    if (q[2] < 0.3).any():
        return img
    x, y = int(q[0][0]), int(q[1][0])
    if not (0 <= x < W and 0 <= y < H) or depth[y, x] < q[2][0] - 0.6:
        return img
    g = np.zeros((H, W), np.float32)
    cv2.line(g, (x, y), (int(q[0][1]), int(q[1][1])), 1.0, max(2, int(px(10) / q[2][0])), cv2.LINE_AA)
    halo = cv2.GaussianBlur(g, (0, 0), max(3, px(40) / q[2][0])) * 4
    return img + (g + halo)[..., None] * np.array([0.3, 1.0, 0.25], np.float32) * on


SIGN = None


def _hero(t, kind, look_dist, L):
    """a particular stalagmite placed where the camera is looking when the diver mentions it"""
    cam = cam_tunnel(t)
    p = cam.pos + cam.F * look_dist
    cx, cz = sc.path(np.array([p[1]]))
    ang = math.atan2(p[2] - float(cz[0]), p[0] - float(cx[0]))
    base = sc.surface_point(float(p[1]), ang, -0.1)
    inward = np.array([float(cx[0]) - base[0], 0.0, float(cz[0]) - base[2]])
    inward /= np.linalg.norm(inward)
    sc.SPIKES.append((base, inward, L, 0.16, kind))


_hero(32.0, "hand", 3.0, 1.5)
_hero(38.6, "stapler", 3.0, 0.9)


def sign_art():
    a = np.full((160, 520, 3), (150, 120, 80), np.uint8)
    cv2.rectangle(a, (6, 6), (513, 153), (90, 70, 45), 6)
    cv2.putText(a, "WELCOME TO", (110, 62), cv2.FONT_HERSHEY_DUPLEX, 1.4, (250, 245, 235), 3, cv2.LINE_AA)
    cv2.putText(a, "THE 9TH ROAD", (40, 132), cv2.FONT_HERSHEY_DUPLEX, 2.0, (250, 245, 235), 4, cv2.LINE_AA)
    return a.astype(np.float32) / 255.0


def cave_hud(img, t, lamp_pct, breath_text):
    c = (0.85, 0.95, 0.95)
    put_text(img, breath_text, 1600, 180, 30, FONT_MONO, c, 1.0, shadow=0.5)
    col = (1.0, 0.45, 0.35) if lamp_pct < 20 else c
    put_text(img, f"LAMP {lamp_pct:3d}%", 1600, 230, 30, FONT_MONO, col, 1.0, shadow=0.5)


def lamp_pct_at(t):
    return int(max(3, 100 - max(0, t - T_POOL) * 0.55 - 25 * ramp(t, 128, 145)))


def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 8)
    vh, gl, gr = 0.32, 0.0, 0.03
    lb, fade = 0.0, None
    hud = dict(alpha=1.0, tc=fmt_tc(16 * 60 + 2 + max(0, t - T_POOL)), level="LV-8", center=None, cam_label="DIVER-1  M.E.G.")
    blm = 0.45
    lamp = 1.0
    breath = "BREATH --:--"

    if t < T_POOL:
        img = card_bg()
        draw_card(img, CARD_OPEN, t)
        vh, gr, blm = 0.3, 0.03, 0
        hud = None
        fade = ((0, 0, 0), 1 - ramp(t, 0, 0.6) + ramp(t, T_POOL - 0.8, T_POOL))

    elif t < T_TUNNEL:
        if t < 13.4:
            # still under: black water, bubbles, a faint green glimmer above
            img = np.zeros((H, W, 3), np.float32)
            yy = np.linspace(1, 0, H, dtype=np.float32)[:, None, None]
            img = img + np.array([0.0, 0.05, 0.04], np.float32) * yy ** 3 * smooth(ramp(t, 11, 13.4))
            breath = "BREATH 13:%02d" % int(59 - (t - T_POOL) * 6)
        else:
            cam = cam_pool(t)
            img, d = sc.render(cam, t, lamp=smooth(ramp(t, 15.5, 16.2)))
            img = sc.draw_spikes(img, d, cam, 1.0, t)
            blur = 6 * (1 - ramp(t, 13.4, 16))
            if blur > 0.3:
                img = cv2.GaussianBlur(img, (0, 0), px(blur))
        gl = 0.4 * math.exp(-((t - 13.4) / 0.1) ** 2)
        a = smooth(ramp(t, 20.0, 21.4)) * (1 - smooth(ramp(t, 24.4, 25.6)))
        if a > 0:
            put_text(img, "LEVEL 8", W1 / 2, H1 / 2 - 40, 118, FONT_SERIF, (1, 1, 1), a, tracking=0.32, shadow=0.6, glow=0.2)
            put_text(img, "C A V E   S Y S T E M S", W1 / 2, H1 / 2 + 62, 34, FONT_SERIF, (1, 1, 1), a * 0.95, shadow=0.6)
            put_text(img, "洞 穴 系 统", W1 / 2, H1 / 2 + 118, 34, FONT_CJK, (1, 1, 1), a * 0.9, shadow=0.6)

    elif t < T_CHAMBER:
        cam = cam_tunnel(t)
        img, d = sc.render(cam, t)
        img = sc.draw_spikes(img, d, cam, 1.0, t)

    elif t < T_WRANGLER:
        cam = cam_chamber(t)
        img, d = sc.render(cam, t, tmax=90)
        img = sc.draw_spikes(img, d, cam, 1.0, t)
        img = sc.draw_mushrooms(img, d, cam, t)
        if t >= 66:
            img = sc.draw_floaters(img, d, cam, t, lamp=1.0)
        gl = 0.4 * (1 - ramp(t, 66, 66.3)) * (t >= 66)

    elif t < T_LOOP:
        cam = cam_wrangler(t)
        lamp = 1.0 - 0.75 * smooth(ramp(t, 93.5, 95.0))
        img, d = sc.render(cam, t, lamp=lamp, tmax=90)
        img = sc.draw_mushrooms(img, d, cam, t, 0.8)
        img = sc.draw_floaters(img, d, cam, t, lamp=lamp)
        if 87.0 < t < 99.5:
            img = sc.wrangler(img, cam, t, 87.0, 150.0, lamp)
        # dust falling from the ceiling while it passes
        if 86 < t < 98:
            out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            for _ in range(160):
                x, y = rng.integers(0, W), rng.integers(0, H)
                cv2.line(out, (int(x), int(y)), (int(x), int(y) + px(rng.integers(6, 30))), (70, 66, 60), 1, cv2.LINE_AA)
            img = out.astype(np.float32) / 255 * 0.5 + img * 0.5
        gl = 0.08 * (86 < t < 98) * (rng.random() < 0.4)

    elif t < T_FOLLOW:
        cam = cam_loop(t)
        img, d = sc.render(cam, t)
        img = sc.draw_spikes(img, d, cam, 1.0, t)
        fz, cx = floor_z(GLOW_Y)
        drop = t > 106.8
        if drop:
            img = chemlight(img, d, cam, np.array([cx - 0.95, GLOW_Y, fz + 0.3]), t)
        gl = 0.3 * math.exp(-((t - 114.5) / 0.06) ** 2)

    elif t < T_ROAD:
        cam = cam_follow(t)
        flick = 1.0 if rng.random() > 0.12 * ramp(t, 136, 145) else 0.25
        lamp = (0.55 - 0.3 * ramp(t, 136, 146)) * flick
        img, d = sc.render(cam, t, lamp=lamp)
        img = sc.draw_spikes(img, d, cam, lamp, t)
        gr = 0.05

    elif t < T_END:
        cam = cam_road(t)
        lamp = 0.25
        lights = sc.road_lights(max(296.0, cam.pos[1] - 6), cam.pos[1] + 30)
        fz_end, cx_end = floor_z(372.0)
        lights.append(((cx_end, 372.0, fz_end + 2.0), (3.0, 2.2, 1.2), 14.0))      # Hollow Nest
        img, d = sc.render(cam, t, lamp=lamp, extra_light=lights)
        img = sc.draw_spikes(img, d, cam, 0.5, t)
        img = sc.rope_lights(img, d, cam, t, 296, 372)
        global SIGN
        if SIGN is None:
            SIGN = sign_art()
        img = sc.sign_9th(img, d, cam, sc.SIGN_Y, SIGN * (0.6 + 0.4 * smooth(ramp(t, T_ROAD, 150))))
        blm = 0.7
        dk = smooth(ramp(t, T_END - 2.5, T_END))
        fade = ((0, 0, 0), dk)

    else:
        lt = t - T_END
        img = card_bg()
        draw_card(img, CARD_END, lt, y0=270, size=34, dy=58)
        a = smooth(ramp(lt, 7.5, 9.0)) * (1 - smooth(ramp(lt, 12.2, 13.0)))
        if a > 0:
            put_text(img, "「你仍被囚禁，在大地的心脏里。」", W1 / 2, 700, 46, FONT_CJK, (0.55, 0.95, 0.85), a, shadow=0, glow=0.35 * a)
            put_text(img, "“You are imprisoned still, in the heart of the earth.”", W1 / 2, 770, 26, FONT_SERIF, (0.6, 0.8, 0.75), a, shadow=0)
        vh, gr, blm = 0.3, 0.03, 0
        hud = None
        fade = ((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)) + 1 - ramp(t, T_END, T_END + 0.4))

    if hud is not None:
        cave_hud(img, t, lamp_pct_at(t), breath)
    if blm:
        img = bloom(img, 0.6, blm)
    return finish(img, rng, t, vh=vh, gl=gl, gr=gr, vig=0.6, hud_args=hud, lb=lb, subs=SUBS, fade=fade)

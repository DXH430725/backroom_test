"""
Timeline, cameras, captions and post-processing for LEVEL 254 — "BLUE HEAVEN".

    python film/film.py --still 40,90,150          # write stills to out/stills
    python film/film.py --render                   # full film -> out/level254_video.mp4 (no audio)
"""
import argparse
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from level254 import (  # noqa: E402
    W, H, FPS, DURATION, F0, FONT_CJK, FONT_SERIF, FONT_MONO, FONT_ITALIC, CIRCLE, SEAT,
    Cam, handheld, blue_scene, hall_scene, city_scene, put_text, vhs, glitch, static, grain,
    vignette, hud, fmt_tc, smooth, smoother, ramp, lerp,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

# ----------------------------------------------------------------------------
# shot list (seconds)
# ----------------------------------------------------------------------------
T_HALL = 11.0      # Level 287
T_NOCLIP = 27.4    # walk into the wall
T_VOID = 29.0      # Level 254, empty blue
T_ZOOM = 56.0      # +3h: a line on the horizon
T_RING = 74.0      # the outer ring
T_AISLE = 96.0     # time-lapse down the aisle
T_SEAT = 118.0     # the empty seat
T_RISE = 138.0     # out of body
T_SKY = 176.0      # everything is just blue
T_FALL = 181.0     # no-clip through the floor
T_CITY = 183.6     # Level 11
T_END = 187.5      # archive card

# ----------------------------------------------------------------------------
# captions: (start, end, chinese, english)
# ----------------------------------------------------------------------------
SUBS = [
    (33.2, 37.8, "「那是你这辈子见过的，最美的蓝色。」", "“The most beautiful shade of blue you will ever see.”"),
    (38.4, 42.4, "「一直延伸下去，永无尽头。」", "“Stretching out forever.”"),
    (51.6, 55.4, "—— 艾莉丝·贝克，M.E.G. 访谈记录 #7350", "— Elise Beck, M.E.G. Interview Log 7350"),
    (64.5, 69.5, "[ 抵达后第三个小时，地平线上出现了一条细线 ]", "[ Hour three. A thin line on the horizon. ]"),
    (83.6, 88.8, "受本层精神效应影响的流浪者，终日静坐于地面。", "Wanderers under the level’s psychic influence. They sit on the floor. Forever."),
    (89.4, 94.8, "所有进入 Level 254 的人，最终都会加入他们。", "Everyone who enters Level 254 eventually joins them."),
    (99.5, 104.5, "没有一个人回头。", "Not one of them turns around."),
    (121.0, 125.6, "「这里好安静。」", "“It’s so peaceful.”"),
    (126.4, 131.0, "「所有的烦恼都消失了。」", "“All of your troubles are gone.”"),
    (131.8, 136.8, "「外面的世界，已经不重要了。」", "“The outside world doesn’t matter.”"),
    (166.0, 171.5, "「一切，都只是蓝色。」", "“Everything is just blue.”"),
    (176.4, 180.6, "「我的队员还在那里。……那是值得的。」", "“My team is still there. …It was worth it.”"),
    (T_CITY + 0.2, T_CITY + 2.8, "出口：穿过地板 no-clip → LEVEL 11", "EXIT: no-clip through the floor → Level 11"),
]

CARD_OPEN = [
    ("M.E.G. 视听档案馆", 0.4),
    ("录像编号 ········ 254-EB-01", 1.6),
    ("来源 ············ 「艾莉丝小队」(Track Mappers) 头戴摄像机", 2.8),
    ("状态 ············ 唯一回收的影像  ·  未经核实", 4.4),
    ("", 0),
    ("警告：本录像含 ρ 级精神危害。", 6.0),
    ("观看过程中如感到平静，请立即停止播放。", 7.4),
]

CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 254「蓝色天堂」", 0.6),
    ("生存难度 ········ CLASS ρ (RHO)", 1.8),
    ("                 不安全  ·  不稳定  ·  精神危害", 2.8),
    ("入口 ············ 穿过 Level 287 的墙壁 (no-clip)", 4.0),
    ("出口 ············ 穿过地板 (no-clip) → Level 11", 5.0),
    ("已知生还者 ······ 1", 6.0),
    ("", 0),
    ("访谈记录 7350 结束时，受访者只提出了一个请求：", 7.6),
]


# ----------------------------------------------------------------------------
# helpers for the ring shot
# ----------------------------------------------------------------------------
def _outer_figure():
    FX, FY = CIRCLE[0], CIRCLE[1]
    target = np.array([-2.2, -448.6])
    i = int(np.argmin((FX - target[0]) ** 2 + (FY - target[1]) ** 2))
    return np.array([FX[i], FY[i]]), CIRCLE[2][i]


HERO, HERO_YAW = _outer_figure()


def look_at(pos, target):
    d = np.asarray(target, float) - np.asarray(pos, float)
    yaw = math.degrees(math.atan2(d[0], d[1]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return yaw, pitch


# ----------------------------------------------------------------------------
# per-shot cameras
# ----------------------------------------------------------------------------
def cam_hall(t):
    lt = t - T_HALL
    speed = 13.0 / 10.5
    turn = smoother(ramp(t, 21.5, 24.0))
    y = min(lt, 10.5) * speed + 0.25 * smooth(ramp(t, 21.5, 23.0))
    x = 1.0 * smooth(ramp(t, 23.2, T_NOCLIP + 0.4)) + 0.02
    bob = 0.025 * math.sin(2 * math.pi * 1.7 * lt) * (1 - 0.7 * smooth(ramp(t, 21.5, 23)))
    hy, hp, hr = handheld(t, 1.2, 1.3)
    yaw = 90 * turn + hy
    return Cam((x, y, 1.62 + bob), yaw, hp - 1.5, hr)


def cam_void(t):
    lt = t - T_VOID
    hy, hp, hr = handheld(t, 0.8, 0.2)
    yaw = -18 + 30 * smoother(lt / 27.0) + hy
    pitch = -9 + 13 * smooth(ramp(t, T_VOID + 2, 40)) - 20 * smooth(ramp(t, 40, 44.5)) + 14 * smooth(ramp(t, 50, 56)) + hp
    y = -1600 + 0.6 * lt
    return Cam((0.0, y, 1.64 + 0.012 * math.sin(lt * 9.5)), yaw, pitch, hr)


def cam_zoom(t):
    lt = t - T_ZOOM
    hy, hp, hr = handheld(t, 0.7, 3.0)
    y = -660 + 1.1 * min(lt, 4.0)
    z = 1.3 + 3.5 * (smooth(ramp(t, 60, 64)) - smooth(ramp(t, 69.5, 72.5)))
    zoom = 1 + 4.5 * (smooth(ramp(t, 60.5, 64.5)) - smooth(ramp(t, 69.5, 73)))
    return Cam((0.0, y, 1.64), 1.5 + hy / zoom, -0.35 + hp / zoom, hr, F0 * zoom), zoom


def cam_ring(t):
    lt = t - T_RING
    hy, hp, hr = handheld(t, 0.6, 5.0)
    head = np.array([HERO[0], HERO[1], 0.72])
    start = np.array([-0.4, -466.0, 1.64])
    # approach behind the figure
    behind = np.array([HERO[0] - 0.4, HERO[1] - 2.4, 1.6])
    a = smoother(ramp(t, T_RING, 82.0))
    if t < 82.0:
        pos = start + (behind - start) * a
        pos[2] += 0.02 * math.sin(lt * 10.5) * (1 - a)
        tgt = lerp(np.array([0.0, 0.0, 1.0]) + np.array([0, pos[1] + 60, 0.0]), head, smooth(ramp(t, 76, 81)))
        tgt = np.array([lerp(pos[0], head[0], smooth(ramp(t, 76, 81))), lerp(pos[1] + 60, head[1], smooth(ramp(t, 76, 81))), lerp(1.2, head[2], smooth(ramp(t, 76, 81)))])
        yaw, pitch = look_at(pos, tgt)
    else:
        # orbit round to the side of the figure, then turn to the aisle
        b = smoother(ramp(t, 82.0, 90.0))
        ang0 = math.atan2(behind[1] - HERO[1], behind[0] - HERO[0])
        ang = ang0 + b * math.radians(-95)
        rad = lerp(math.hypot(behind[0] - HERO[0], behind[1] - HERO[1]), 1.45, b)
        pos = np.array([HERO[0] + rad * math.cos(ang), HERO[1] + rad * math.sin(ang), lerp(1.6, 1.25, b)])
        c = smoother(ramp(t, 89.5, T_AISLE))
        pos = pos + (np.array([0.0, -450.8, 1.64]) - pos) * c
        yaw, pitch = look_at(pos, head)
        yaw = lerp(yaw, 0.0, c) if abs(yaw) < 180 else yaw
        pitch = lerp(pitch, -1.0, c)
    return Cam(tuple(pos), yaw + hy, pitch + hp, hr)


AISLE_Y0, AISLE_Y1 = -450.8, -10.0


def aisle_y(t):
    s = smoother(ramp(t, T_AISLE, T_SEAT))
    return lerp(AISLE_Y0, AISLE_Y1, s)


def cam_aisle(t):
    hy, hp, hr = handheld(t, 0.5, 7.0)
    return Cam((0.0, aisle_y(t), 1.64), hy, -1.0 + hp, hr)


def cam_seat(t):
    hy, hp, hr = handheld(t, lerp(0.6, 0.15, ramp(t, 126, 134)), 9.0)
    y = lerp(-10.0, -3.55, smoother(ramp(t, T_SEAT, 127.0)))
    y = lerp(y, SEAT[1] - 0.05, smooth(ramp(t, 128.0, 134.0)))
    z = lerp(1.64, 0.92, smoother(ramp(t, 128.0, 134.5)))
    pitch = lerp(-4.0, 1.0, smooth(ramp(t, 125, 134))) + hp
    return Cam((0.0, y, z), hy, pitch, hr)


def cam_rise(t):
    """Out-of-body ascent: straight up out of the seat, a slow turning top-down view of the
    whole circle, then an orbit outward to an oblique view, and finally up into the sky."""
    c = smooth(ramp(t, 144.0, 158.0))
    cx, cy = SEAT[0] * (1 - c), (SEAT[1] - 0.05) * (1 - c)
    z = 0.92 * math.exp(math.log(820 / 0.92) * smoother(ramp(t, 141.0, 161.0)))
    z = lerp(z, 470.0, smoother(ramp(t, 161.0, 172.0)))
    z += 900 * smoother(ramp(t, 171.5, T_FALL))
    phi = math.radians(180 + 75 * smoother(ramp(t, 146.0, 173.0)))  # orbit angle / heading
    D = 1150 * smoother(ramp(t, 160.0, 172.5))
    x = cx + D * math.sin(phi)
    y = cy + D * math.cos(phi)
    yaw = math.degrees(phi) + 180
    look = -math.degrees(math.atan2(z, D)) if D > 1e-3 else -89.9
    pitch = lerp(1.0, max(look, -89.9), smoother(ramp(t, 141.5, 153.0)))
    pitch = lerp(pitch, 40.0, smoother(ramp(t, 172.0, T_SKY + 1.0)))
    pitch = lerp(pitch, 64.0, smooth(ramp(t, T_SKY + 1, T_FALL)))
    breath = 0.03 * math.sin(t * 1.3) * (1 - smooth(ramp(t, 141, 146)))
    return Cam((x, y, z + breath), yaw, pitch, 0.0)


# ----------------------------------------------------------------------------
# frame
# ----------------------------------------------------------------------------
def draw_card(img, lines, lt, x0=260, y0=300, dy=64, size=38, cps=18):
    for text, start in lines:
        if not text:
            continue
        n = int(max(0, lt - start) * cps)
        if n <= 0:
            continue
        shown = text[:n]
        cursor = "▌" if n < len(text) and (lt * 3) % 1 < 0.6 else ""
        put_text(img, shown + cursor, x0, y0, size, FONT_CJK, (0.86, 0.9, 0.95), 1.0, shadow=0, anchor="left")
        y0 += dy
    return y0


def subtitles(img, t, letterbox):
    for a, b, zh, en in SUBS:
        if a <= t <= b:
            op = min(1, (t - a) / 0.5, (b - t) / 0.5)
            yb = H - (175 if letterbox < 0.5 else 70)
            if letterbox > 0.5:
                yb = H - 60
            put_text(img, zh, W / 2, yb - 50, 44, FONT_CJK, (1, 1, 1), op, shadow=0.75)
            put_text(img, en, W / 2, yb + 2, 28, FONT_ITALIC, (0.86, 0.9, 1.0), op * 0.9, shadow=0.7)


def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 13)
    vh = 0.0      # VHS amount
    gr = 0.02     # grain
    gl = 0.0      # glitch
    hud_a = 0.0
    tc = ""
    level = ""
    zoomtxt = None
    lb = 0.0      # letterbox
    fade = None   # (colour, amount)
    img = None

    if t < T_HALL:
        img = np.full((H, W, 3), (0.012, 0.014, 0.02), np.float32)
        lt = t
        draw_card(img, CARD_OPEN, lt)
        gr = 0.03
        vh = 0.35
        fade = ((0, 0, 0), 1 - ramp(t, 0.0, 0.6) + ramp(t, T_HALL - 0.9, T_HALL))

    elif t < T_VOID:
        lt = t - T_HALL
        cam = cam_hall(t)
        fr = np.random.default_rng(int(t * 11))

        def flicker(idx):
            idx = np.asarray(idx)
            base = np.ones(idx.shape, np.float32)
            bad = (idx % 5 == 3)
            state = 0.25 + 0.75 * (fr.random() > 0.35 + 0.3 * math.sin(t * 3))
            return np.where(bad, state, base)

        gy = 30.0
        ghost = None
        if 13.5 < t < 20.5 and (int(t * 6) % 7) not in (2, 5):
            ghost = (gy, 1.0)
        img = hall_scene(cam, t, flicker, ghost)
        # progressive loss of colour, as documented for deep Level 287
        desat = smooth(ramp(t, 13, 24))
        lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
        img = img * (1 - desat) + lum[..., None] * desat
        vh = 0.55 + 0.35 * ramp(t, 20, T_NOCLIP)
        gl = 0.12 * (fr.random() < 0.08 + 0.3 * ramp(t, 18, 27)) + smooth(ramp(t, T_NOCLIP - 0.6, T_VOID)) * 1.1
        gr = 0.035 + 0.03 * ramp(t, 18, 28)
        hud_a = 1.0
        tc = fmt_tc(41 * 60 + 7 + lt)
        level = "LV-287"
        st = ramp(t, 16, 28) * 0.18 + 0.7 * smooth(ramp(t, T_NOCLIP + 0.5, T_VOID))
        img = img * (1 - st) + static(rng) * st
        fade = ((0, 0, 0), 1 - ramp(t, T_HALL, T_HALL + 0.5))

    elif t < T_ZOOM:
        cam = cam_void(t)
        img = blue_scene(cam, t, figures=False)
        k = 1 - smooth(ramp(t, T_VOID, T_VOID + 2.2))
        gl = 1.2 * k
        if k > 0.02:
            img = img * (1 - 0.8 * k * k) + static(rng) * 0.8 * k * k
        # the level heals the picture: noise and the camera's HUD dissolve
        vh = lerp(0.6, 0.0, smooth(ramp(t, T_VOID + 1, 44)))
        gr = lerp(0.03, 0.008, ramp(t, T_VOID, 44))
        hud_a = 1 - smooth(ramp(t, 40, 47))
        tc = "--:--:--:--" if t < T_VOID + 2.2 else fmt_tc(t - T_VOID)
        level = "LV-???" if t < 36 else "LV-254"
        # title
        a = smooth(ramp(t, 43.0, 45.2)) * (1 - smooth(ramp(t, 49.6, 51.4)))
        if a > 0:
            put_text(img, "LEVEL 254", W / 2, H / 2 - 40, 118, FONT_SERIF, (1, 1, 1), a, tracking=0.32, shadow=0.35, glow=0.35)
            put_text(img, "B L U E    H E A V E N", W / 2, H / 2 + 62, 34, FONT_SERIF, (0.92, 0.95, 1), a * 0.95, shadow=0.35)
            put_text(img, "蓝 色 天 堂", W / 2, H / 2 + 118, 34, FONT_CJK, (0.92, 0.95, 1), a * 0.85, shadow=0.35)

    elif t < T_RING:
        cam, zoom = cam_zoom(t)
        img = blue_scene(cam, t)
        vh = 0.18
        gr = 0.012
        hud_a = 0.75 * smooth(ramp(t, T_ZOOM, T_ZOOM + 0.6))
        tc = fmt_tc(3 * 3600 + 12 * 60 + 40 + (t - T_ZOOM))
        level = "LV-254"
        zoomtxt = f"ZOOM x{zoom:.1f}" if zoom > 1.05 else None
        gl = 0.5 * (1 - ramp(t, T_ZOOM, T_ZOOM + 0.35))
        fade = ((0, 0, 0), 1 - ramp(t, T_ZOOM, T_ZOOM + 0.3))

    elif t < T_AISLE:
        cam = cam_ring(t)
        img = blue_scene(cam, t)
        vh = 0.15
        gr = 0.012
        hud_a = 0.7
        tc = fmt_tc(3 * 3600 + 31 * 60 + 5 + (t - T_RING))
        level = "LV-254"
        gl = 0.5 * (1 - ramp(t, T_RING, T_RING + 0.35))
        fade = ((0, 0, 0), 1 - ramp(t, T_RING, T_RING + 0.3))
        a = smooth(ramp(t, 77.6, 79)) * (1 - smooth(ramp(t, 82.4, 83.4)))
        if a > 0:
            put_text(img, "THE MEDITATION CIRCLE", W / 2, H / 2 - 20, 60, FONT_SERIF, (1, 1, 1), a, tracking=0.25, shadow=0.5, glow=0.25)
            put_text(img, "冥 想 圈", W / 2, H / 2 + 60, 40, FONT_CJK, (1, 1, 1), a, shadow=0.5)

    elif t < T_SEAT:
        # time-lapse: motion-blurred sub-frames
        acc = None
        for k in range(3):
            ts = t + (k / 3 - 1 / 3) / FPS
            im = blue_scene(cam_aisle(ts), t)
            acc = im if acc is None else acc + im
        img = acc / 3
        vh = 0.2
        gr = 0.014
        hud_a = 0.7
        speed = (aisle_y(t + 0.05) - aisle_y(t - 0.05)) / 0.1
        tc = fmt_tc(3 * 3600 + 53 * 60 + (t - T_AISLE) * (1 + speed * 1.6))
        level = "LV-254"
        zoomtxt = f"▶▶ x{max(1, int(1 + speed * 1.6))}"

    elif t < T_RISE:
        cam = cam_seat(t)
        img = blue_scene(cam, t)
        vh = lerp(0.15, 0.0, ramp(t, 124, 132))
        gr = lerp(0.012, 0.006, ramp(t, 124, 132))
        hud_a = 0.7 * (1 - smooth(ramp(t, 124, 131)))
        tc = fmt_tc(5 * 3600 + 58 * 60 + (t - T_SEAT))
        level = "LV-254"
        lb = smooth(ramp(t, 131, 137))

    elif t < T_FALL:
        cam = cam_rise(t)
        img = blue_scene(cam, t, figures=cam.pos[2] < 5000)
        gr = 0.006
        lb = 1.0
        # bloom into light at the very end
        w = smooth(ramp(t, 178.8, T_FALL))
        img = img * (1 - 0.55 * w) + np.array([0.86, 0.93, 1.0], np.float32) * 0.55 * w

    elif t < T_CITY:
        # no-clip through the floor
        lt = t - T_FALL
        if lt < 1.0:
            cz = max(1.4 * (1 - lt / 0.8), 0.004)
            cam = Cam((0, -120, cz), 20 * lt, -89 * smooth(lt / 0.6), 25 * lt)
            img = blue_scene(cam, t, figures=False)
            img = img * 0.4 + static(rng) * 0.6 * smooth(lt)
        else:
            img = static(rng) * 0.45 * (1 - ramp(t, T_FALL + 1.0, T_CITY))
        gl = 1.0
        vh = 1.0
        gr = 0.05
        hud_a = 0.9 * (rng.random() < 0.7)
        tc = "--:--:--:--"
        level = "LV-254 >> ??"

    elif t < T_END:
        img = city_scene(t - T_CITY)
        vh = 0.9
        gr = 0.04
        gl = 0.25 * (rng.random() < 0.25) + 1.0 * ramp(t, T_END - 0.7, T_END)
        hud_a = 1.0
        tc = fmt_tc(0.4 + t - T_CITY)
        level = "LV-11"
        st = 0.5 * (1 - ramp(t, T_CITY, T_CITY + 0.5)) + 0.8 * ramp(t, T_END - 0.4, T_END)
        img = img * (1 - st) + static(rng) * st

    else:
        lt = t - T_END
        img = np.full((H, W, 3), (0.012, 0.014, 0.02), np.float32)
        yend = draw_card(img, CARD_END, lt, y0=250, size=36, dy=60)
        a = smooth(ramp(lt, 10.6, 12.4))
        if a > 0:
            put_text(img, "「让我回去。」", W / 2, 820, 64, FONT_CJK, (0.45, 0.66, 1.0), a, shadow=0, glow=0.6 * a)
        b = smooth(ramp(lt, 13.4, 14.4))
        if b > 0:
            put_text(img, "[ 本档案已依据 Mass Trimming Project 归档 · 信息可能不完整 ]", W / 2, H - 90, 24, FONT_CJK, (0.5, 0.52, 0.58), b, shadow=0)
        gr = 0.03
        vh = 0.3
        gl = 0.8 * (1 - ramp(t, T_END, T_END + 0.4))
        fade = ((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)))

    # ---- post
    img = vhs(img, vh, rng, t)
    img = glitch(img, gl, rng)
    img = grain(img, gr, rng)
    img = vignette(img, 0.55 if lb < 0.5 else 0.35)
    hud(img, t, hud_a, tc, level, zoomtxt)
    if lb > 0:
        bar = int(lb * (H - W / 2.39) / 2)
        if bar > 0:
            img[:bar] = 0
            img[H - bar:] = 0
    subtitles(img, t, lb)
    if fade is not None and fade[1] > 0:
        img = img * (1 - fade[1]) + np.array(fade[0], np.float32) * fade[1]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def _worker(fi):
    return fi, render_frame(fi).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", default="")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=DURATION)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(OUT, "level254_video.mp4"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.still:
        from PIL import Image
        os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
        for s in args.still.split(","):
            t = float(s)
            t0 = time.time()
            fr = render_frame(int(round(t * FPS)))
            p = os.path.join(OUT, "stills", f"t{t:06.1f}.jpg")
            Image.fromarray(fr).save(p, quality=90)
            print(p, f"{time.time() - t0:.2f}s", flush=True)
    if args.render:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        f0, f1 = int(args.start * FPS), int(args.end * FPS)
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
               "-tune", "grain", "-pix_fmt", "yuv420p", args.out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        t0 = time.time()
        with Pool(args.jobs) as pool:
            for n, (fi, buf) in enumerate(pool.imap(_worker, range(f0, f1), chunksize=2)):
                p.stdin.write(buf)
                if n % 48 == 0:
                    el = time.time() - t0
                    print(f"frame {fi}/{f1}  {el:.0f}s elapsed  eta {el / (n + 1) * (f1 - f0 - n - 1):.0f}s", flush=True)
        p.stdin.close()
        p.wait()
        print("done", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""
Ep04 — LEVEL 881 "The Courtyard of Windows": timeline, cameras, captions.
Recovered helmet-camera logs of M.E.G. team Owl-Eyes (January 2017).

    python -m backrooms.render ep04_level881 --still 15,30,60,90,115,140,160
    python -m backrooms.render ep04_level881 --all [--res 4k]
"""
import math
import os

import cv2
import numpy as np

from backrooms.core import (
    W, H, W1, H1, S, FPS, FONT_CJK, FONT_SERIF, FONT_MONO, Cam, handheld, look_at, put_text, static,
    fmt_tc, smooth, smoother, ramp, lerp, draw_card, card_bg, finish, walk_bob, subs_from_lines, px,
    bloom, value_noise,
)
from . import scene as sc

TITLE = "LEVEL 881 —「窗之庭院」The Courtyard of Windows"
DURATION = 198.0
HERE = os.path.dirname(os.path.abspath(__file__))

T_ENTER = 11.0
T_SURVEY = 26.0
T_LOG2 = 52.0
T_LOG3 = 76.0
T_MUSIC = 91.5
T_CORR = 104.0
T_ADD = 124.0
T_FINAL = 150.0
T_BOOK = 157.5
T_BOOK_END = 165.5
T_FALL = 171.0
T_END = 176.0

SUBS = subs_from_lines(os.path.join(HERE, "voice", "lines.json"))

CARD_OPEN = [
    ("M.E.G. 档案 · 远征记录 881", 0.4),
    ("小队 ············ 指南针点团「猫头鹰之眼」· 4 人", 1.4),
    ("进入 ············ 2017 年 1 月", 2.6),
    ("状态 ············ 未归还 · 头盔摄像机记录部分回收", 3.6),
    ("", 0),
]
CARD_END = [
    ("M.E.G. 档案  ·  LEVEL 881「窗之庭院」", 0.6),
    ("生存难度 ········ 5  ·  出口 5/5（死路）· 环境 4/5 · 实体 5/5", 1.8),
    ("入口 ············ 出现在各室内层级的窗户，进入后 2–3 分钟消失", 3.0),
    ("出口 ············ 唯一确认的出口是你进来的那扇窗", 4.2),
    ("禁止接触 ········ 其他窗户 · 玻璃栏杆 · 木制圆盘 · 桌子及桌上一切物品", 5.4),
    ("", 0),
]

N = sc.ENTRY_N
P0 = sc.ENTRY_P
YAW_IN = math.degrees(math.atan2(N[0], N[1]))
TABLE1 = np.array([6.5, 3.0])
TABLE2 = np.array([0.0, 0.4])
DOOR_P, DOOR_N, DOOR_U = sc.door_point(7, 0.42)
DOOR_EDGE = 7
LAMP3 = np.array([-4.3, 6.4])


def at(p, n=None, d=0.0):
    return p + (N if n is None else n) * d


def cam_enter(t):
    lt = t - T_ENTER
    d = lerp(-3.4, 4.5, smoother(ramp(t, 12.8, 19.5)))
    pos = at(P0, d=d)
    z = 1.65 - 0.25 * math.exp(-((t - 15.6) / 0.6) ** 2)          # duck through the frame
    pitch = lerp(0, 42, smoother(ramp(t, 19.0, 24.0))) - 4
    yaw = YAW_IN + 25 * smooth(ramp(t, 20, 25.5))
    hy, hp, hr = handheld(t, 1.0, 0.7)
    return Cam((pos[0], pos[1], z + walk_bob(t, 1.0) * ramp(t, 14, 15) * (1 - ramp(t, 19, 20))), yaw + hy, pitch + hp, hr)


def cam_survey(t):
    lt = t - T_SURVEY
    pos = at(P0, d=4.5 + 0.35 * lt)
    yaw = YAW_IN + 25 + 335 * smoother(ramp(t, T_SURVEY, 49.5))
    pitch = 18 * math.sin(lt * 0.45) * (1 - ramp(t, 46, 50)) + 2
    hy, hp, hr = handheld(t, 0.8, 1.7)
    return Cam((pos[0], pos[1], 1.7 + walk_bob(t, 0.35)), yaw + hy, pitch + hp, hr)


def cam_log2(t):
    base = at(P0, d=4.5 + 0.35 * (T_LOG2 - T_SURVEY))
    turn = smoother(ramp(t, 52.0, 54.5))
    yaw0 = YAW_IN + 360
    yaw = yaw0 + 180 * turn                                          # look back at the entry wall
    walk = smoother(ramp(t, 56.0, 63.0))
    target = np.array([2.2, -1.6])
    pos = base + (target - base) * walk
    yaw = lerp(yaw, yaw0 + 180 + (look_at((*base, 0), (*target, 0))[0] - (YAW_IN + 180)), smooth(ramp(t, 55.5, 57)))
    pitch = lerp(0, -28, smooth(ramp(t, 58.5, 61))) * (1 - smooth(ramp(t, 63, 65)))
    walk2 = smoother(ramp(t, 64.5, 69.5))
    pos = pos + (TABLE1 + np.array([-1.3, -0.9]) - pos) * walk2
    if t > 64.5:
        y2, p2 = look_at((*pos, 1.7), (*TABLE1, 0.8))
        k = smooth(ramp(t, 64.5, 66.5))
        yaw = lerp(yaw, y2 + 360 * round((yaw - y2) / 360), k)
        pitch = lerp(pitch, p2, smooth(ramp(t, 68.5, 71)))
    hy, hp, hr = handheld(t, 0.9, 2.5)
    return Cam((pos[0], pos[1], 1.7 + walk_bob(t, 1.0) * max(ramp(t, 56, 57) * (1 - ramp(t, 62.5, 63)), ramp(t, 64.6, 65) * (1 - ramp(t, 69, 69.5)))), yaw + hy, pitch + hp, hr)


def cam_log3(t):
    pos = np.array([-1.5, -6.0])
    yaw = lerp(-60, 30, smoother(ramp(t, T_LOG3, 89.0)))
    pitch = -6
    push = smooth(ramp(t, 89.0, 95.5))
    pos = pos + (TABLE2 + np.array([-0.6, -2.2]) - pos) * push
    if t > 89:
        y2, p2 = look_at((*pos, 1.7), (*TABLE2, 0.9))
        yaw = lerp(yaw, y2, smooth(ramp(t, 89, 91)))
        pitch = lerp(pitch, p2, smooth(ramp(t, 89, 91)))
    turn = smoother(ramp(t, 96.4, 98.4))
    yd, pd = look_at((*pos, 1.7), (*DOOR_P, 1.2))
    yaw = lerp(yaw, yd, turn)
    pitch = lerp(pitch, pd, turn)
    run = smooth(ramp(t, 98.4, T_CORR))
    pos = pos + (DOOR_P + DOOR_N * 0.8 - pos) * run
    hy, hp, hr = handheld(t * (1 + run), 1.0 + 2.0 * run, 3.3)
    return Cam((pos[0], pos[1], 1.7 + walk_bob(t, 1.0 + 2 * run, 0.03) * max(push * (1 - ramp(t, 95, 95.5)), run)), yaw + hy, pitch + hp, hr)


def cam_corr(t):
    lt = t - T_CORR
    y = 1.0 + 1.1 * min(lt, 10.5) + 0.2 * max(0, lt - 10.5)
    back = smoother(ramp(t, 121.8, 123.0))
    hy, hp, hr = handheld(t, 1.0 + 1.5 * ramp(t, 113, 118), 4.4)
    return Cam((0.15, y, 1.68 + walk_bob(t, 1.1) * (1 - ramp(t, 114, 115))), 180 * back + hy, -2 + hp, hr)


def cam_final(t):
    pos = np.array([-2.0, -3.4])
    lift = smoother(ramp(t, T_FINAL, 153.5))
    pitch = lerp(-40, 14, lift)
    yaw = lerp(10, -40, smoother(ramp(t, 151, T_BOOK)))
    fall = smoother(ramp(t, 166.0, T_FALL + 1.5))
    yb, pb = look_at((*pos, 0.95), (1.5, -1.0, 0))
    yaw = lerp(yaw, yb, smooth(ramp(t, 165.6, 167.5)))
    pitch = lerp(pitch, -60, smooth(ramp(t, 165.6, 167.5)))
    pos = pos + (np.array([1.2, -0.9]) - pos) * fall
    z = lerp(0.95, 0.02, fall)
    hy, hp, hr = handheld(t, 0.7 + 0.8 * fall, 6.0)
    return Cam((pos[0], pos[1], z), yaw + hy, pitch + hp, hr + 25 * fall * fall)


def depth_full(d):
    return cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)


def team(img, d, cam, t, members):
    """members: list of (x, y, yaw, kneel)"""
    for x, y, yw, kn in sorted(members, key=lambda m: -np.hypot(m[0] - cam.pos[0], m[1] - cam.pos[1])):
        sc.person(img, d, cam, x, y, yw, 1.0, lamp=True, kneel=kn)


GATSBY = [
    "In my younger and more vulnerable years my father gave",
    "me some advice that I've been turning over in my mind",
    "ever since. \"Whenever you feel like criticizing any one,\"",
    "he told me, \"just remember that all the people in this",
    "world haven't had the advantages that you've had.\"",
    "He didn't say any more, but we've always been unusually",
    "communicative in a reserved way, and I understood that he",
    "meant a great deal more than that.",
]
WRONG = [
    "In my younger and more vulnerable years my window gave",
    "me some advice that I've been turning over in my floor",
    "ever since. \"Whenever you feel like leaving any one,\"",
    "it told me, \"just remember that all the windows in this",
    "room haven't had the advantages that you've had.\"",
    "It didn't say any more, but we've always been watching",
    "you in a reserved way, and I understood that we",
    "do not enter the courtyard of windows.",
]


def book_page(t):
    lt = t - T_BOOK
    img = np.ones((H, W, 3), np.float32) * np.array([0.82, 0.78, 0.66], np.float32)
    nz = value_noise(256, 5, 5)
    img *= (0.88 + 0.16 * cv2.resize(nz, (W, H)))[..., None]
    rng = np.random.default_rng(int(t * 3))
    for i, (a, b) in enumerate(zip(GATSBY, WRONG)):
        k = ramp(lt, 1.5 + i * 0.6, 3.0 + i * 0.6)
        line = "".join(bc if (rng.random() < k and bc != ac) or (k > 0.98) else ac for ac, bc in zip(a.ljust(len(b)), b))
        put_text(img, line, 300, 300 + i * 66, 40, FONT_SERIF, (0.14, 0.11, 0.09), 1.0, shadow=0, anchor="left")
    put_text(img, "— 1 —", W1 / 2, 880, 30, FONT_SERIF, (0.3, 0.26, 0.2), 1.0, shadow=0)
    # vignette of a head-torch on paper
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx - W * 0.48) / W, (yy - H * 0.45) / H)
    img *= np.clip(1.25 - 1.6 * r, 0.15, 1)[..., None]
    sh = 3.0 * math.sin(lt * 1.3)
    M = np.float32([[1, 0, px(sh)], [0, 1, px(sh * 0.6)]])
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


_FAB = None


def blanket(t, rng):
    """inside the blanket: dark woven fabric, light leaking through, breathing"""
    global _FAB
    if _FAB is None:
        n = value_noise(512, 6, 9)
        yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
        weave = 0.5 + 0.5 * np.sin(xx * 1.9) * np.sin(yy * 1.9)
        _FAB = (0.5 * n + 0.5 * weave).astype(np.float32)
    br = 1.04 + 0.03 * math.sin(t * 2.2)
    fab = cv2.resize(_FAB, (int(W * br), int(H * br)))[:H, :W]
    leak = np.clip(1 - np.hypot(*np.mgrid[0:H, 0:W].astype(np.float32)[::-1] - np.array([W * 0.7, H * 0.2])[:, None, None]) / (W * 0.6), 0, 1) ** 2
    img = (0.025 + 0.05 * fab[..., None] + 0.1 * leak[..., None] * fab[..., None]) * np.array([1.0, 0.85, 0.7], np.float32)
    return img


def render_frame(fi):
    t = fi / FPS
    rng = np.random.default_rng(fi * 7919 + 4)
    vh, gl, gr = 0.35, 0.0, 0.022
    hud_a, tc, level, center = 1.0, "", "LV-881", None
    lb, fade = 0.0, None
    blm = True

    if t < T_ENTER:
        img = card_bg()
        draw_card(img, CARD_OPEN, t)
        a = smooth(ramp(t, 6.0, 6.8))
        if a > 0:
            put_text(img, "不要进入窗之庭院。", W1 / 2, 700, 64, FONT_CJK, (0.9, 0.16, 0.12), a, shadow=0, glow=0.35 * a)
            put_text(img, "DO NOT ENTER THE COURTYARD OF WINDOWS.", W1 / 2, 780, 28, FONT_MONO, (0.75, 0.2, 0.16), a, shadow=0)
        vh, gr, hud_a, blm = 0.3, 0.03, 0.0, False
        fade = ((0, 0, 0), 1 - ramp(t, 0, 0.6) + ramp(t, T_ENTER - 0.8, T_ENTER))

    elif t < T_SURVEY:
        cam = cam_enter(t)
        img, d = sc.render(cam, t, ent=0.1, entry_open=True)
        tc = fmt_tc(t - T_ENTER)
        center = "LOG 1"
        a = smooth(ramp(t, 21.0, 22.4)) * (1 - smooth(ramp(t, 24.6, 25.8)))
        if a > 0:
            put_text(img, "LEVEL 881", W1 / 2, H1 / 2 - 40, 118, FONT_SERIF, (1, 1, 1), a, tracking=0.32, shadow=0.5, glow=0.2)
            put_text(img, "THE  COURTYARD  OF  WINDOWS", W1 / 2, H1 / 2 + 62, 34, FONT_SERIF, (1, 1, 1), a * 0.95, shadow=0.5)
            put_text(img, "窗 之 庭 院", W1 / 2, H1 / 2 + 118, 34, FONT_CJK, (1, 1, 1), a * 0.9, shadow=0.5)

    elif t < T_LOG2:
        cam = cam_survey(t)
        img, d = sc.render(cam, t, ent=0.12 + 0.06 * ramp(t, 30, 40), entry_open=True)
        d = depth_full(d)
        lt = t - T_SURVEY
        team(img, d, cam, t, [(-4 + 0.2 * lt, 6.0, -0.5, False), (3.0 + 0.1 * lt, 1.8 + 0.04 * lt, 0.8, False), (1.0, -9.0 + 0.1 * lt, 1.7, False)])
        tc = fmt_tc(t - T_ENTER)
        center = "LOG 1"

    elif t < T_LOG3:
        cam = cam_log2(t)
        img, d = sc.render(cam, t, ent=0.2, entry_open=False)
        d = depth_full(d)
        sc.table(img, d, cam, *TABLE1, 0.4)
        team(img, d, cam, t, [(1.9, 0.4, 2.4, t > 57), (-3.5, 5.0, -0.8, False), (4.5, -2.0, 1.2, False)])
        tc = fmt_tc(14 * 60 + 5 + t - T_LOG2)
        center = "LOG 2"
        gl = 0.5 * (1 - ramp(t, T_LOG2, T_LOG2 + 0.3))

    elif t < T_CORR:
        cam = cam_log3(t)
        dark = 0.72
        img, d = sc.render(cam, t, ent=0.42 + 0.15 * ramp(t, T_MUSIC, T_CORR), epoch=1, light=dark,
                           door=(DOOR_EDGE, DOOR_U) if t > 95.0 else None)
        d = depth_full(d)
        sc.table(img, d, cam, *TABLE2, 1.3, light=dark)
        sc.lamp_on_floor(img, d, cam, *LAMP3, dark)
        team(img, d, cam, t, [(2.2, -3.4, 2.6, False), (-2.8, -1.5, 0.2, False)])
        tc = fmt_tc(4 * 3600 + 12 * 60 + t - T_LOG3)
        center = "LOG 3"
        gl = 0.5 * (1 - ramp(t, T_LOG3, T_LOG3 + 0.3)) + 0.25 * math.exp(-((t - T_MUSIC) / 0.1) ** 2)

    elif t < T_ADD:
        cam = cam_corr(t)
        lt = t - T_CORR
        fy = 24.0 if 112.5 < t < 121.6 else None
        img = sc.corridor(cam, t, faceling_y=fy)
        d = np.full((H, W), 100.0, np.float32)
        mates = [(-0.45, 3.2 + 1.1 * min(lt, 9.5), -math.pi / 2 + 3.14159, False)]
        if t < 121.0:
            mates.append((0.5, 4.0 + 1.05 * min(lt, 9.6), math.pi / 2, False))
        for x, y, yw, kn in mates:
            sc.person(img, d, cam, x, y, math.pi / 2, 1.0, lamp=True)
        tc = fmt_tc(4 * 3600 + 41 * 60 + lt)
        center = "LOG 3"
        gl = 0.6 * (1 - ramp(t, T_CORR, T_CORR + 0.4)) + 0.9 * math.exp(-((t - 121.3) / 0.25) ** 2) + 0.15 * (fy is not None) * (rng.random() < 0.3)
        st = 0.5 * math.exp(-((t - 121.3) / 0.3) ** 2)
        img = img * (1 - st) + static(rng) * st
        vh = 0.45

    elif t < T_FINAL:
        lt = t - T_ADD
        img = blanket(t, rng)
        if lt < 2.5:
            img = img * ramp(lt, 1.2, 2.5)
            put_text(img, "ADDENDUM · LOG 4–6", W1 / 2, H1 / 2, 40, FONT_MONO, (0.85, 0.85, 0.85), 1 - ramp(lt, 1.6, 2.4), shadow=0)
        for th in (136.7, 142.4, 147.0):
            if abs(t - th) < 0.12:
                cam = Cam((0, -2, 1.0), 40 * (th - 140), 18)
                hal, _ = sc.render(cam, t, all_ent=True, light=0.55)
                img = img * 0.3 + hal * 0.7
                gl = 0.6
        tc = fmt_tc(6 * 3600 + 3 * 60 + lt)
        center = "LOG 6"
        vh = 0.5
        gr = 0.04
        blm = False

    elif t < T_END:
        if T_BOOK <= t < T_BOOK_END:
            img = book_page(t)
            tc = fmt_tc(13 * 3600 + 49 * 60 + t - T_FINAL)
            center = "LOG 6"
        else:
            cam = cam_final(t)
            img, d = sc.render(cam, t, all_ent=True, epoch=2, light=0.55, blinds_glow=1.25)
            d = depth_full(d)
            sc.table(img, d, cam, *TABLE2, 1.3, light=0.55)
            sc.lamp_on_floor(img, d, cam, *LAMP3, 0.55)
            sc.lamp_on_floor(img, d, cam, 3.1, 4.2, 0.55)
            lift = ramp(t, T_FINAL, 153.0)
            if lift < 1:
                bl = blanket(t, rng)
                cut = int(H * (1 - smooth(lift)))
                img[:cut] = bl[H - cut:]
            tc = fmt_tc(13 * 3600 + 49 * 60 + t - T_FINAL)
            center = "LOG 6"
            gl = 0.4 * (1 - ramp(t, T_BOOK_END, T_BOOK_END + 0.3)) + 1.3 * smooth(ramp(t, T_FALL, T_END))
            dk = smooth(ramp(t, T_FALL + 1.0, T_END - 0.5))
            st = smooth(ramp(t, T_FALL, T_FALL + 1.5)) * (1 - dk)
            img = img * (1 - st) + static(rng) * st * 0.6
            fade = ((0, 0, 0), dk)
        vh = 0.45

    else:
        lt = t - T_END
        img = card_bg()
        draw_card(img, CARD_END, lt, y0=260, size=34, dy=58)
        a = smooth(ramp(lt, 13.5, 15.0))
        if a > 0:
            put_text(img, "「不要进入窗之庭院。」", W1 / 2, 760, 64, FONT_CJK, (0.9, 0.16, 0.12), a, shadow=0, glow=0.4 * a)
        vh, gr, hud_a, blm = 0.3, 0.03, 0.0, False
        fade = ((0, 0, 0), smooth(ramp(t, DURATION - 2.0, DURATION)) + 1 - ramp(t, T_END, T_END + 0.4))

    if blm:
        img = bloom(img, 0.7, 0.45)
    return finish(img, rng, t, vh=vh, gl=gl, gr=gr, vig=0.6,
                  hud_args=dict(alpha=hud_a, tc=tc, level=level, center=center, cam_label="OWL-1  LT.FASSBENDER"),
                  lb=lb, subs=SUBS, fade=fade)

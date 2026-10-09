"""
Ep02 — LEVEL 0 "Threshold": world building and shading.

One GridWorld holds every sub-zone the canon page describes, laid side by side:
the segmented rooms (with the corridor where the Peripheral Shift happens, a long
flickering hall, a blackout zone with standing fluid and a red-room loop), the vast
Pillars, the Holes, and the Arches with the flickering exit wall. A second, taller
world is Level 1. Shading is done per pixel from the raycaster's hit buffers.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, px, s2l, l2s, value_noise
from backrooms.gridworld import (GridWorld, sample, OPEN, SOLID, DOOR, ARCH, FLICKER,
                                 FAR, WALL, FLOOR, CEIL, PIT)

C = 1.5
RS = 1.5                      # internal render scale: picture traced at W/RS x H/RS
IW, IH = int(W / RS), int(H / RS)

# ---------------------------------------------------------------------------
# zones (cell coordinates)
# ---------------------------------------------------------------------------
MAZE = (1, 1, 82, 80)
PILL = (100, 1, 300, 200)
HOLE = (320, 1, 400, 80)
ARCHZ = (420, 1, 560, 130)
NX, NY = 600, 210

Z_MAZE, Z_PILL, Z_HOLE, Z_ARCH = 0, 1, 2, 3


def cc(i):
    """cell index -> world coordinate of the cell centre"""
    return i * C + C / 2


def _divide(w, rng, x0, y0, x1, y1, depth=0):
    """recursive division into rooms and halls with doorways"""
    wd, ht = x1 - x0, y1 - y0
    if (wd < 6 and ht < 6) or depth > 9:
        return
    horiz = ht > wd if wd != ht else rng.random() < 0.5
    if horiz and ht >= 6:
        y = int(rng.integers(y0 + 2, y1 - 3))
        w.hwall(x0, x1, y, SOLID)
        for _ in range(int(rng.integers(1, 4))):
            w.wallN[int(rng.integers(x0, x1)), y] = DOOR
        if rng.random() < 0.25:
            g = int(rng.integers(x0, x1 - 1))
            w.wallN[g:g + 2, y] = OPEN
        _divide(w, rng, x0, y0, x1, y + 1, depth + 1)
        _divide(w, rng, x0, y + 1, x1, y1, depth + 1)
    elif wd >= 6:
        x = int(rng.integers(x0 + 2, x1 - 3))
        w.vwall(x, y0, y1, SOLID)
        for _ in range(int(rng.integers(1, 4))):
            w.wallE[x, int(rng.integers(y0, y1))] = DOOR
        if rng.random() < 0.25:
            g = int(rng.integers(y0, y1 - 1))
            w.wallE[x, g:g + 2] = OPEN
        _divide(w, rng, x0, y0, x + 1, y1, depth + 1)
        _divide(w, rng, x + 1, y0, x1, y1, depth + 1)


def carve(w, cells):
    """make sure the camera can walk the cell path: open every edge it crosses, clear solids"""
    for (ax, ay), (bx, by) in zip(cells[:-1], cells[1:]):
        x, y = ax, ay
        while (x, y) != (bx, by):
            sx, sy = np.sign(bx - x), np.sign(by - y)
            if sx:
                e = min(x, x + sx)
                if w.wallE[e, y] not in (OPEN, ARCH):
                    w.wallE[e, y] = DOOR
                x += sx
            else:
                e = min(y, y + sy)
                if w.wallN[x, e] not in (OPEN, ARCH):
                    w.wallN[x, e] = DOOR
                y += sy
            w.solid[x, y] = False
            w.hole[x, y] = False


# camera routes (cell paths)
PATH_MAZE = [(5, 5), (5, 14), (15, 14), (15, 20)]
PATH_SHIFT = [(15, 20), (15, 25)]
CORR_X = 40                     # the long hall
PATH_BLACK = [(52, 4), (52, 16), (64, 16), (64, 24)]
PATH_RED = [(58, 46), (58, 60)]
ARCH_X = 430
ARCH_Y0, ARCH_Y1 = 6, 30


def build_world():
    rng = np.random.default_rng(0)
    w = GridWorld(NX, NY, C, 2.7)
    zone = np.full((NX, NY), -1, np.int8)
    light = np.zeros((NX, NY), np.float32)
    dark = np.zeros((NX, NY), np.float32)     # blackout cells
    red = np.zeros((NX, NY), np.float32)      # red-room cells
    corr = np.zeros((NX, NY), np.float32)     # flickering hall cells

    # ---- the rooms
    x0, y0, x1, y1 = MAZE
    w.box(x0, y0, x1, y1)
    _divide(w, rng, x0, y0, x1, y1)
    zone[x0:x1, y0:y1] = Z_MAZE
    for x in range(x0, x1):
        for y in range(y0, y1):
            if x % 2 == 0 and y % 3 == 1:
                light[x, y] = 1.0
    # wake-up room: clear a 6x6 room around the start
    w.wallE[2:9, 2:10] = OPEN
    w.wallN[2:10, 2:9] = OPEN
    # shift corridor: a straight hall at x=15, y=20..30
    w.wallN[15, 19:31] = OPEN
    w.vwall(14, 20, 31, SOLID)
    w.vwall(15, 20, 31, SOLID)
    for y in (20, 23, 26, 29):
        light[15, y] = 1.0
    w.wallN[15, 21] = DOOR      # the doorway that will disappear
    # the long hall at x = CORR_X
    w.wallN[CORR_X, 5:75] = OPEN
    w.vwall(CORR_X - 1, 6, 75, SOLID)
    w.vwall(CORR_X, 6, 75, SOLID)
    w.hwall(CORR_X, CORR_X + 1, 74, SOLID)
    for y in range(6, 75):
        corr[CORR_X, y] = 1.0
        light[CORR_X, y] = 1.0 if y % 3 == 0 else 0.0
    # blackout region
    dark[48:80, 1:30] = 1.0
    # red-room loop
    red[52:66, 50:66] = 1.0
    for p in (PATH_MAZE, PATH_SHIFT, PATH_BLACK, PATH_RED):
        carve(w, p)
    carve(w, [(CORR_X, 6), (CORR_X, 70)])
    carve(w, [(16, 25), (19, 25)])
    # exit lit door at end of blackout path
    # ---- the Pillars: an endless grid of square columns
    x0, y0, x1, y1 = PILL
    w.box(x0, y0, x1, y1)
    zone[x0:x1, y0:y1] = Z_PILL
    for x in range(x0, x1):
        for y in range(y0, y1):
            if x % 4 == 0 and y % 4 == 0:
                w.solid[x, y] = True
            if x % 2 == 1 and y % 2 == 1:
                light[x, y] = 1.0
    # ---- the Holes: pits in a grid
    x0, y0, x1, y1 = HOLE
    w.box(x0, y0, x1, y1)
    zone[x0:x1, y0:y1] = Z_HOLE
    for x in range(x0, x1):
        for y in range(y0, y1):
            if x % 3 == 1 and y % 3 == 1:
                w.hole[x, y] = True
            if x % 6 == 4 and y % 6 == 4:
                w.solid[x, y] = True
            if x % 3 == 0 and y % 3 == 0:
                light[x, y] = 1.0
    # ---- the Arches: pale rooms opening into each other forever
    x0, y0, x1, y1 = ARCHZ
    w.box(x0, y0, x1, y1)
    zone[x0:x1, y0:y1] = Z_ARCH
    for x in range(x0, x1):
        if x % 3 == 2:
            w.wallE[x, y0:y1] = ARCH
    for y in range(y0, y1):
        if y % 3 == 2:
            w.wallN[x0:x1, y] = ARCH
    for x in range(x0, x1):
        for y in range(y0, y1):
            if x % 3 == 1 and y % 3 == 1:
                light[x, y] = 1.0
    w.wallN[ARCH_X, ARCH_Y1] = FLICKER
    return w, zone, light, dark, red, corr


WORLD, ZONE, LIGHT, DARK, RED, CORR = build_world()

# Peripheral Shift: three versions of the shift corridor's walls
WN_A = WORLD.wallN.copy()
WE_A = WORLD.wallE.copy()
WN_B = WN_A.copy()
WN_B[15, 21] = SOLID            # the doorway you just walked through is gone
WN_C = WN_B.copy()
WN_C[15, 28] = SOLID            # and the hall ahead ends in a wall
WE_C = WE_A.copy()
WE_C[15, 25] = DOOR             # a new door that was never there


def set_shift(state):
    if state == 0:
        WORLD.wallN, WORLD.wallE = WN_A, WE_A
    elif state == 1:
        WORLD.wallN, WORLD.wallE = WN_B, WE_A
    else:
        WORLD.wallN, WORLD.wallE = WN_C, WE_C


# ---------------------------------------------------------------------------
# lighting fields (0.25 m raster)
# ---------------------------------------------------------------------------
RES = 0.25
K = int(C / RES)


def _light_field(mask):
    img = np.zeros((NY * K, NX * K), np.float32)
    ys, xs = np.nonzero(mask.T > 0)
    for y, x in zip(ys, xs):
        cy, cx_ = y * K + K // 2, x * K + K // 2
        img[cy - 2:cy + 3, cx_ - 1:cx_ + 2] = mask[x, y]
    r = 22
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32) * RES
    ker = 1.0 / (1 + (xx ** 2 + yy ** 2 + 2.6 ** 2 - 2.6 ** 2 + 1.2) / 3.0) ** 1.6
    out = cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_CONSTANT)
    return out / 4.2


_on = LIGHT.copy()
L_BASE = _light_field(_on * (1 - DARK) * (1 - CORR))
L_DARK = _light_field(_on * DARK)
L_CORR = _light_field(_on * CORR)
DIST = WORLD.wall_distance(RES)
ZONE_R = np.repeat(np.repeat(ZONE.T, K, 0), K, 1)
RED_R = cv2.GaussianBlur(np.repeat(np.repeat(RED.T, K, 0), K, 1), (0, 0), 6)
DARK_R = cv2.GaussianBlur(np.repeat(np.repeat(DARK.T, K, 0), K, 1), (0, 0), 3)

# textures
N_A = value_noise(512, 6, 11)
N_B = value_noise(512, 6, 12)
N_C = value_noise(256, 4, 13)

WALLPAPER = s2l((184, 168, 98))
WALL_STRIPE = s2l((178, 160, 78))
PEEL = s2l((120, 24, 22))
CARPET = s2l((148, 124, 80))
CEILING = s2l((214, 205, 162))
ARCH_WALL = s2l((228, 220, 190))
ARCH_FLOOR = s2l((176, 158, 112))
HAZE = s2l((120, 106, 52))
REDTINT = np.array([1.35, 0.32, 0.26], np.float32)


def shade(cam, t, R, lights=1.0, dark_lights=1.0, corr_lights=1.0, flash=0.0, red_amt=0.0,
          flicker_wall=0.0, exposure=0.72, wet=1.0):
    """linear RGB image (IH, IW, 3) from hit buffers."""
    k, tt = R["kind"], R["t"]
    hx, hy, hz = R["hx"], R["hy"], R["hz"]
    wall = k == WALL
    floor = k == FLOOR
    ceil = k == CEIL
    pit = k == PIT
    zone = sample(ZONE_R.astype(np.float32), hx, hy, RES, -1)
    archz = np.abs(zone - Z_ARCH) < 0.5
    # --- albedo
    along = np.where(R["axis"], hy, hx)                     # coordinate along the wall
    u512 = (along * 90).astype(np.int64) % 512
    v512 = (hz * 90).astype(np.int64) % 512
    stain = N_A[v512, u512]
    stripe = (np.sin(along * 2 * math.pi / 0.16) > 0.55).astype(np.float32)
    motif = ((np.abs(np.sin(along * math.pi / 0.08)) * np.abs(np.sin(hz * math.pi / 0.12))) > 0.82).astype(np.float32)
    wallcol = WALLPAPER * (1 - 0.06 * stripe[..., None] - 0.05 * motif[..., None])
    wallcol = wallcol * (1 - 0.35 * np.clip(stain - 0.6, 0, 1)[..., None] * 2.5)
    pu = (along * 12).astype(np.int64) % 512
    pv = (hz * 12 + along * 0.7).astype(np.int64) % 512
    peel = (N_B[pv, pu] > 0.9) & (hz > 0.4) & (stain > 0.45)
    wallcol = np.where(peel[..., None], PEEL, wallcol)
    wallcol = np.where(archz[..., None], ARCH_WALL * (1 - 0.08 * stain[..., None]), wallcol)
    wallcol = wallcol * (1 - 0.35 * np.exp(-np.maximum(hz, 0) / 0.22))[..., None]   # damp at the floor
    fu = (hx * 160).astype(np.int64) % 512
    fv = (hy * 160).astype(np.int64) % 512
    fiber = N_B[fv, fu]
    blot = N_C[(hy * 9).astype(np.int64) % 256, (hx * 9).astype(np.int64) % 256]
    carpet = CARPET * (0.86 + 0.28 * fiber[..., None]) * (1 - 0.32 * np.clip(blot - 0.55, 0, 1)[..., None] * 2.2)
    carpet = np.where(archz[..., None], ARCH_FLOOR * (0.9 + 0.2 * fiber[..., None]), carpet)
    tiles = ((np.abs((hx / 0.6) % 1 - 0.5) > 0.485) | (np.abs((hy / 0.6) % 1 - 0.5) > 0.485)).astype(np.float32)
    ceilcol = CEILING * (1 - 0.25 * tiles[..., None]) * (0.95 + 0.1 * N_C[(hy * 20).astype(np.int64) % 256, (hx * 20).astype(np.int64) % 256][..., None])
    alb = np.where(wall[..., None], wallcol, np.where(floor[..., None], carpet, ceilcol))
    # --- light
    Lb = sample(L_BASE, hx, hy, RES) * lights
    Ld = sample(L_DARK, hx, hy, RES) * dark_lights
    Lc = sample(L_CORR, hx, hy, RES) * corr_lights
    dk = sample(DARK_R, hx, hy, RES)
    E = (Lb + Ld) * (1 - dk * (1 - dark_lights)) + Lc
    dw = sample(DIST, hx, hy, RES, 0.0)
    ao_fc = 1 - 0.38 * np.exp(-dw / 0.3)
    ao_w = 1 - 0.28 * np.exp(-np.maximum(hz, 0) / 0.25) - 0.18 * np.exp(-np.maximum(2.7 - hz, 0) / 0.2)
    vert = np.where(wall, 0.7 + 0.35 * np.clip(hz / 2.7, 0, 1), np.where(ceil, 0.55, 1.0))
    E = E * vert * np.where(wall, ao_w, ao_fc) + 0.012
    # camera light (blackout)
    if flash > 0:
        cosang = R["dx"] * cam.F[0] + R["dy"] * cam.F[1] + R["dz"] * cam.F[2]
        spot = np.clip((cosang - 0.80) / 0.2, 0, 1) ** 1.6 + 0.08 * np.clip((cosang - 0.5) / 0.5, 0, 1)
        E = E + flash * 3.2 * spot / (1 + (tt / 2.2) ** 2)
    col = alb * E[..., None]
    # ceiling light panels
    lx = (hx / C) % 1
    ly = (hy / C) % 1
    cxi = np.clip((hx // C).astype(np.int64), 0, NX - 1)
    cyi = np.clip((hy // C).astype(np.int64), 0, NY - 1)
    lv = LIGHT[cxi, cyi] * (lights * (1 - DARK[cxi, cyi]) * (1 - CORR[cxi, cyi])
                           + dark_lights * DARK[cxi, cyi] + corr_lights * CORR[cxi, cyi])
    panel = ceil & (np.abs(lx - 0.5) < 0.2) & (np.abs(ly - 0.5) < 0.4) & (lv > 0)
    col = np.where(panel[..., None], np.array([1.0, 0.97, 0.82], np.float32) * (0.15 + 1.9 * lv)[..., None], col)
    # standing fluid in the blackout zone: dark, with a glint of the camera light
    if wet > 0:
        fl = floor & (dk > 0.5)
        rip = 0.5 + 0.5 * np.sin(hx * 7 + t * 1.3) * np.sin(hy * 6 - t * 1.1)
        col = np.where(fl[..., None], col * 0.35 + (0.05 * rip * flash / (1 + (tt / 3) ** 2))[..., None] * np.array([1, 0.95, 0.8], np.float32), col)
    # pits: the walls of the hole fade into nothing
    pitcol = WALLPAPER * 0.3 * np.exp(np.minimum(hz, 0) / 0.32)[..., None] * (E[..., None] + 0.05)
    col = np.where(pit[..., None], pitcol, col)
    # red rooms
    if red_amt > 0:
        rr = sample(RED_R, hx, hy, RES) * red_amt
        col = col * (1 + (REDTINT - 1) * rr[..., None])
    # the flickering wall
    if flicker_wall > 0:
        fw = wall & (R["edge"] == FLICKER)
        col = np.where(fw[..., None], col * (1 - flicker_wall) + flicker_wall * np.array([0.02, 0.02, 0.03], np.float32), col)
    # haze
    fog = np.exp(-tt / 48.0)[..., None]
    haze = WALLPAPER * 0.42 * lights * (1 - 0.95 * dk[..., None] * (1 - dark_lights))
    col = col * fog + haze * (1 - fog)
    col = np.where((k == FAR)[..., None], haze, col)
    return col * exposure


def render_world(cam, t, **kw):
    """-> display sRGB float image (H, W, 3)"""
    # rays at internal resolution
    X = (np.arange(IW, dtype=np.float32) + 0.5) - IW / 2
    Y = IH / 2 - (np.arange(IH, dtype=np.float32) + 0.5)
    X, Y = np.meshgrid(X, Y)
    f = cam.f / RS
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    R = WORLD.trace(cam.pos, dx / n, dy / n, dz / n, tmax=kw.pop("tmax", 110.0))
    col = shade(cam, t, R, **kw)
    img = l2s(np.clip(col, 0, 1))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img, R


# ---------------------------------------------------------------------------
# Level 1 — "Habitable Zone": a vast concrete warehouse in fog
# ---------------------------------------------------------------------------
def build_level1():
    w = GridWorld(160, 160, 3.0, 6.0)
    w.box(1, 1, 159, 159)
    for x in range(1, 159):
        for y in range(1, 159):
            if x % 4 == 0 and y % 4 == 0:
                w.solid[x, y] = True
    return w


L1 = build_level1()
CONC = s2l((118, 118, 112))


def render_level1(cam, t):
    X = (np.arange(IW, dtype=np.float32) + 0.5) - IW / 2
    Y = IH / 2 - (np.arange(IH, dtype=np.float32) + 0.5)
    X, Y = np.meshgrid(X, Y)
    f = cam.f / RS
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    R = L1.trace(cam.pos, dx / n, dy / n, dz / n, tmax=90.0)
    k, tt, hx, hy, hz = R["kind"], R["t"], R["hx"], R["hy"], R["hz"]
    nz = N_A[(hz * 40 + hx * 13).astype(np.int64) % 512, (hy * 40 + hx * 7).astype(np.int64) % 512]
    alb = CONC * (0.75 + 0.4 * nz[..., None])
    alb = np.where((k == FLOOR)[..., None], CONC * 0.7 * (0.8 + 0.3 * N_B[(hy * 30).astype(np.int64) % 512, (hx * 30).astype(np.int64) % 512][..., None]), alb)
    # sodium lamps every 12 m
    lx, ly = (hx % 12) - 6, (hy % 12) - 6
    E = 0.12 + 1.6 / (1 + (lx ** 2 + ly ** 2 + (6 - hz) ** 2) / 10)
    col = alb * E[..., None] * np.array([1.0, 0.9, 0.75], np.float32)
    lamp = (k == CEIL) & (lx ** 2 + ly ** 2 < 0.3)
    col = np.where(lamp[..., None], np.array([1.4, 1.2, 0.8], np.float32), col)
    puddle = (k == FLOOR) & (N_C[(hy * 3).astype(np.int64) % 256, (hx * 3).astype(np.int64) % 256] > 0.62)
    col = np.where(puddle[..., None], col * 0.4 + 0.05, col)
    fog = np.exp(-tt / 22.0)[..., None]
    col = col * fog + np.array([0.09, 0.09, 0.085], np.float32) * (1 - fog)
    img = l2s(np.clip(col, 0, 1))
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

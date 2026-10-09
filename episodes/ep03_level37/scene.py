"""
Ep03 — LEVEL 37 "Sublimity" (the Poolrooms): world and water rendering.

White tile everywhere, lukewarm blue-green water with a constant ripple, pillars
rising out of deep pools, stairs that descend into the dark. Water is rendered
properly: each water pixel traces a reflected and a refracted ray (Fresnel mix,
Beer–Lambert absorption below the surface), the camera can be underwater (the
surface seen from below, with total internal reflection), and caustics play on
every tiled surface near the water.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, s2l, l2s, value_noise
from backrooms.gridworld import (GridWorld, sample, OPEN, SOLID, DOOR, ARCH, STEP,
                                 FAR, WALL, FLOOR, CEIL, PIT, WATER)

C = 1.5
RS = 1.5
IW, IH = int(W / RS), int(H / RS)
NX, NY = 130, 135
WL = 0.0


def cc(i):
    return i * C + C / 2


# ---------------------------------------------------------------------------
# world
# ---------------------------------------------------------------------------
POOL_A = (10, 10, 23, 23)          # arrival pool room (walls), pool inside
POOL_IN = (12, 12, 21, 21)
CORR = (15, 23, 18, 50)            # wading corridor (3 cells wide)
HALL = (2, 50, 64, 101)            # the great hall
LANE_X, LANE_Y = 16, 75
STAIRS = (64, 70, 84, 81)          # stairs room
STAIR_Y = (73, 78)
DARK = (84, 60, 112, 92)
TUNNEL_X = 97


def build():
    w = GridWorld(NX, NY, C, 4.5)
    w.heights(0.05, 4.5)
    w.enable_water(WL)
    light = np.zeros((NX, NY), np.float32)
    zone = np.zeros((NX, NY), np.int8)
    w.solid[:, :] = True

    def room(x0, y0, x1, y1, floor=0.05, ceil=4.5, water=False):
        w.solid[x0:x1, y0:y1] = False
        w.fz[x0:x1, y0:y1] = floor
        w.cz[x0:x1, y0:y1] = ceil
        w.water[x0:x1, y0:y1] = water

    # arrival pool
    room(*POOL_A, 0.05, 5.0)
    x0, y0, x1, y1 = POOL_IN
    w.fz[x0:x1, y0:y1] = -2.4
    w.water[x0:x1, y0:y1] = True
    for x in range(POOL_A[0], POOL_A[2]):
        for y in range(POOL_A[1], POOL_A[3]):
            if (x + y) % 4 == 0:
                light[x, y] = 0.9
    # corridor (ankle-deep water)
    room(*CORR, -0.22, 3.6, True)
    for y in range(CORR[1], CORR[3], 3):
        light[16, y] = 0.8
    # the hall
    room(*HALL, -1.9, 16.0, True)
    x0, y0, x1, y1 = HALL
    for x in range(x0, x1):
        for y in range(y0, y1):
            if x % 5 == 0 and y % 5 == 0:
                w.solid[x, y] = True                      # pillars out of the water
    w.fz[LANE_X - 1:LANE_X + 2, y0:y1] = 0.05             # walkways
    w.water[LANE_X - 1:LANE_X + 2, y0:y1] = False
    w.fz[x0:x1, LANE_Y - 1:LANE_Y + 2] = 0.05
    w.water[x0:x1, LANE_Y - 1:LANE_Y + 2] = False
    w.solid[LANE_X - 1:LANE_X + 2, y0:y1] = False
    w.solid[x0:x1, LANE_Y - 1:LANE_Y + 2] = False
    # shallow shelves along the walls
    w.fz[x0:x0 + 2, y0:y1] = -0.3
    w.fz[x1 - 2:x1, y0:y1] = -0.3
    rng = np.random.default_rng(37)
    for x in range(x0, x1):
        for y in range(y0, y1):
            if (x % 4 == 2 and y % 4 == 2) and rng.random() < 0.75:
                light[x, y] = 1.3 * rng.uniform(0.4, 1.2)
    # stairs into the deep
    room(*STAIRS, 0.05, 5.0, True)
    x0, y0, x1, y1 = STAIRS
    w.fz[x0:x1, y0:y1] = -0.4
    for x in range(x0, x1):
        k = x - (x0 + 2)
        if k >= 0:
            w.fz[x, STAIR_Y[0]:STAIR_Y[1]] = -0.3 - 0.42 * k
    w.fz[x0:x0 + 2, :] = np.where(w.solid[x0:x0 + 2, :], w.fz[x0:x0 + 2, :], 0.05)
    w.water[x0:x0 + 2, y0:y1] = False
    w.fz[x1 - 5:x1, STAIR_Y[0]:STAIR_Y[1]] = -40.0           # the pit
    light[x0, (y0 + y1) // 2] = 0.9
    light[x0 + 4, (y0 + y1) // 2] = 0.4
    # connection hall -> stairs room
    w.solid[62:66, LANE_Y - 1:LANE_Y + 2] = False
    w.fz[62:66, LANE_Y - 1:LANE_Y + 2] = 0.05
    w.water[62:66, LANE_Y - 1:LANE_Y + 2] = False
    w.cz[62:66, LANE_Y - 1:LANE_Y + 2] = 4.0
    # dark rooms and the tunnel to Level 43
    room(*DARK, -0.2, 3.8, True)
    x0, y0, x1, y1 = DARK
    for x in range(x0 + 4, x1, 6):
        w.solid[x, y0:y1 - 4] = True
        w.solid[x, y0 + (x * 7) % 9: y0 + (x * 7) % 9 + 3] = False
    room(TUNNEL_X - 1, DARK[3], TUNNEL_X + 1, DARK[3] + 25, -0.3, 2.2, True)
    room(78, 60, 84, 66, 0.05, 3.5, False)                       # a lit antechamber
    light[80, 62] = 1.0
    w.solid[82:86, 62:64] = False
    w.fz[82:86, 62:64] = 0.05
    zone[DARK[0]:DARK[2], DARK[1]:DARK[3] + 26] = 2
    return w, light, zone


WORLD, LIGHT, ZONE = build()

# ---------------------------------------------------------------------------
# lighting & helper rasters (0.25 m)
# ---------------------------------------------------------------------------
RES = 0.25
K = int(C / RES)


def _light_field(mask, radius_m=7.0, height=4.0):
    img = np.zeros((NY * K, NX * K), np.float32)
    ys, xs = np.nonzero(mask.T > 0)
    for y, x in zip(ys, xs):
        cy, cx_ = y * K + K // 2, x * K + K // 2
        img[cy - 2:cy + 3, cx_ - 2:cx_ + 3] = mask[x, y]
    r = int(radius_m / RES)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32) * RES
    ker = 1.0 / (1 + (xx ** 2 + yy ** 2) / (height ** 2)) ** 1.5
    return cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_CONSTANT) / 60.0


LF = _light_field(LIGHT)
WATER_R = cv2.GaussianBlur(np.repeat(np.repeat(WORLD.water.T.astype(np.float32), K, 0), K, 1), (0, 0), 6)
DARK_R = cv2.GaussianBlur(np.repeat(np.repeat((ZONE.T == 2).astype(np.float32), K, 0), K, 1), (0, 0), 5)
DIST = WORLD.wall_distance(RES)


def _caustic_tex(n=256, seed=3):
    """photon-splat caustics: bend a grid of rays through a wavy surface, count where they land"""
    rng = np.random.default_rng(seed)
    hgt = value_noise(n, 3, seed) * 1.0
    gy, gx = np.gradient(cv2.GaussianBlur(hgt, (0, 0), 3))
    acc = np.zeros((n, n), np.float32)
    m = 4
    ys, xs = np.mgrid[0:n * m, 0:n * m].astype(np.float32) / m
    sx = (xs + gx[ys.astype(int) % n, xs.astype(int) % n] * 900) % n
    sy = (ys + gy[ys.astype(int) % n, xs.astype(int) % n] * 900) % n
    np.add.at(acc, (sy.astype(int) % n, sx.astype(int) % n), 1.0)
    acc = cv2.GaussianBlur(acc, (0, 0), 0.8)
    acc /= acc.mean()
    return np.clip(acc - 0.6, 0, 6).astype(np.float32)


CAUS_A = _caustic_tex(256, 3)
CAUS_B = _caustic_tex(256, 8)
RIP = value_noise(256, 4, 21)
RGY, RGX = np.gradient(cv2.GaussianBlur(RIP, (0, 0), 2))


def caustics(x, y, t, scale=1.6):
    a = CAUS_A[((y * 128 / scale + t * 9) % 256).astype(np.int64) % 256, ((x * 128 / scale + t * 5) % 256).astype(np.int64) % 256]
    b = CAUS_B[((y * 128 / scale - t * 6) % 256).astype(np.int64) % 256, ((x * 128 / scale + t * 8) % 256).astype(np.int64) % 256]
    return np.minimum(a, b) * 1.4


TILE = s2l((236, 240, 241))
GROUT = s2l((176, 186, 192))
POOLTILE = s2l((196, 228, 236))
ACCENT = s2l((40, 120, 160))
WATER_SCATTER = np.array([0.03, 0.20, 0.22], np.float32)
SIGMA = np.array([0.42, 0.10, 0.085], np.float32)   # absorption per metre (red goes first)


def tile_grout(a, b, fw, ts=0.2, gw=0.006):
    """antialiased grout lines on a surface parameterised by (a, b) metres"""
    out = np.zeros(a.shape, np.float32)
    fade = np.clip(fw / ts * 2.5 - 0.4, 0, 1)
    for c in (a, b):
        fr = (c / ts) % 1.0
        d = np.minimum(fr, 1 - fr) * ts
        cov = np.clip((gw - d) / np.maximum(fw, 1e-5) + 0.5, 0, 1)
        out = np.maximum(out, cov * (1 - fade) + fade * (2 * gw / ts))
    return out


def ripple_normal(x, y, t, amp=1.0):
    gx = (RGX[((y * 40 + t * 7) % 256).astype(np.int64) % 256, ((x * 40 + t * 3) % 256).astype(np.int64) % 256]
          + RGX[((y * 23 - t * 4) % 256).astype(np.int64) % 256, ((x * 23 + t * 6) % 256).astype(np.int64) % 256])
    gy = (RGY[((y * 40 + t * 7) % 256).astype(np.int64) % 256, ((x * 40 + t * 3) % 256).astype(np.int64) % 256]
          + RGY[((y * 23 - t * 4) % 256).astype(np.int64) % 256, ((x * 23 + t * 6) % 256).astype(np.int64) % 256])
    gx = gx * 6 * amp + 0.012 * amp * np.sin(x * 3.1 + t * 1.7)
    gy = gy * 6 * amp + 0.012 * amp * np.cos(y * 2.7 - t * 1.3)
    n = np.sqrt(gx * gx + gy * gy + 1)
    return -gx / n, -gy / n, 1 / n


def shade_surface(R, t, cam_f, lights=1.0, dark=1.0):
    """linear colour of FLOOR/CEIL/WALL/PIT hits (no water handling)."""
    k, tt = R["kind"], R["t"]
    hx, hy, hz = R["hx"], R["hy"], R["hz"]
    wall = k == WALL
    ceil = k == CEIL
    fw = tt / (cam_f / RS) * 1.0
    along = np.where(R["axis"], hy, hx)
    a = np.where(wall, along, hx)
    b = np.where(wall, hz, hy)
    g = tile_grout(a, b, fw)
    under = hz < WL - 0.01
    base = np.where(under[..., None], POOLTILE, TILE)
    # an accent band of darker tiles just above the water line on walls
    band = wall & (hz > WL) & (hz < WL + 0.2)
    base = np.where(band[..., None], ACCENT * 1.6, base)
    alb = base * (1 - g[..., None]) + GROUT * g[..., None]
    # light
    E = sample(LF, hx, hy, RES) * lights
    dk = sample(DARK_R, hx, hy, RES)
    E = E * (1 - 0.92 * dk * dark) + 0.08 * (1 - 0.9 * dk * dark)
    dw = sample(DIST, hx, hy, RES, 0.0)
    fzc = WORLD.fz[np.clip(R["cx"], 0, NX - 1), np.clip(R["cy"], 0, NY - 1)]
    ao = np.where(wall, 1 - 0.22 * np.exp(-np.abs(hz - fzc) / 0.3), 1 - 0.25 * np.exp(-dw / 0.35))
    E = E * np.where(ceil, 0.8, np.where(wall, 0.85, 1.0)) * ao
    # caustics: under water (strong), and reflected onto walls/ceiling near water (soft)
    near_w = sample(WATER_R, hx, hy, RES)
    cu = np.where(wall, along, hx)
    cv_ = np.where(wall, hz * 1.3, hy)
    cs = caustics(cu, cv_, t)
    E = E + np.where(under, 0.55 * cs * (E + 0.1), np.where(ceil, 1.1 * cs * near_w * (E + 0.05), np.where(wall, 0.42 * cs * near_w * (E + 0.05), 0)))
    col = alb * E[..., None]
    # pits / far
    col = np.where((k == PIT)[..., None], 0.0, col)
    fog = np.exp(-tt / 85.0)[..., None]
    haze = np.array([0.5, 0.56, 0.58], np.float32) * 0.35 * lights * (1 - 0.9 * dk[..., None] * dark)
    col = col * fog + haze * (1 - fog)
    col = np.where((k == FAR)[..., None], haze, col)
    # light fixtures on the ceiling: square panels
    lx = (hx / C) % 1
    ly = (hy / C) % 1
    cxi = np.clip((hx // C).astype(np.int64), 0, NX - 1)
    cyi = np.clip((hy // C).astype(np.int64), 0, NY - 1)
    lv = LIGHT[cxi, cyi] * lights
    panel = ceil & (np.abs(lx - 0.5) < 0.22) & (np.abs(ly - 0.5) < 0.22) & (lv > 0)
    col = np.where(panel[..., None], np.array([1.3, 1.32, 1.3], np.float32) * (0.4 + lv[..., None]), col)
    return col


def _refract(dx, dy, dz, nx_, ny_, nz_, eta):
    cosi = -(dx * nx_ + dy * ny_ + dz * nz_)
    k = 1 - eta * eta * (1 - cosi * cosi)
    tir = k < 0
    ks = np.sqrt(np.maximum(k, 0))
    tx = eta * dx + (eta * cosi - ks) * nx_
    ty = eta * dy + (eta * cosi - ks) * ny_
    tz = eta * dz + (eta * cosi - ks) * nz_
    return tx, ty, tz, tir, cosi


def _fresnel(cosi):
    return 0.02 + 0.98 * np.clip(1 - np.abs(cosi), 0, 1) ** 5


def render(cam, t, lights=1.0, dark=1.0, ripple=1.0, tmax=110.0, exposure=1.0):
    """display sRGB (H, W, 3). Handles the camera above or below the water."""
    X = (np.arange(IW, dtype=np.float32) + 0.5) - IW / 2
    Y = IH / 2 - (np.arange(IH, dtype=np.float32) + 0.5)
    X, Y = np.meshgrid(X, Y)
    f = cam.f / RS
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / n, dy / n, dz / n
    ox, oy, oz = cam.pos
    below = oz < WL and WORLD.water[WORLD.cell_of(ox, oy)]
    R = WORLD.trace_rays(ox, oy, oz, dx, dy, dz, tmax)
    col = shade_surface(R, t, cam.f, lights, dark)
    wat = R["kind"] == WATER
    if wat.any():
        j = np.nonzero(wat)
        hx, hy, hz = R["hx"][j], R["hy"][j], R["hz"][j]
        ddx, ddy, ddz = dx[j], dy[j], dz[j]
        nx_, ny_, nz_ = ripple_normal(hx, hy, t, ripple)
        if not below:
            # from the air: reflect up, refract down
            dot = ddx * nx_ + ddy * ny_ + ddz * nz_
            rx, ry, rz = ddx - 2 * dot * nx_, ddy - 2 * dot * ny_, np.abs(ddz - 2 * dot * nz_)
            Rr = WORLD.trace_rays(hx, hy, hz + 1e-3, rx, ry, rz, tmax)
            rc = shade_surface(Rr, t, cam.f * 0.7, lights, dark)
            rc = np.where((Rr["kind"] == WATER)[..., None], WATER_SCATTER * 0.6 * lights, rc)
            tx, ty, tz, tir, cosi = _refract(ddx, ddy, ddz, nx_, ny_, nz_, 1 / 1.33)
            tz = np.minimum(tz, -0.05)
            Rt = WORLD.trace_rays(hx, hy, hz - 1e-3, tx, ty, tz, 30.0)
            tc = shade_surface(Rt, t, cam.f * 0.7, lights, dark)
            L = np.minimum(Rt["t"], 30.0)[..., None]
            Tr = np.exp(-SIGMA * L)
            E0 = sample(LF, hx, hy, RES)[..., None] * lights + 0.05
            under = tc * Tr + WATER_SCATTER * E0 * (1 - Tr)
            F = _fresnel(cosi)[..., None]
            col[j] = F * rc + (1 - F) * under
        else:
            # from below: look up through the surface (or mirror it, past the critical angle)
            tx, ty, tz, tir, cosi = _refract(ddx, ddy, ddz, -nx_, -ny_, -nz_, 1.33)
            Ra = WORLD.trace_rays(hx, hy, hz + 1e-3, tx, ty, np.maximum(tz, 0.02), tmax)
            ac = shade_surface(Ra, t, cam.f * 0.7, lights, dark) * np.array([0.5, 0.82, 0.88], np.float32)
            dot = -(ddx * nx_ + ddy * ny_ + ddz * nz_)
            rx, ry, rz = ddx + 2 * dot * nx_, ddy + 2 * dot * ny_, -np.abs(ddz + 2 * dot * nz_)
            Rb = WORLD.trace_rays(hx, hy, hz - 1e-3, rx, ry, rz, 30.0)
            bc = shade_surface(Rb, t, cam.f * 0.7, lights, dark)
            Lb = np.minimum(Rb["t"], 30.0)[..., None]
            Tb = np.exp(-SIGMA * Lb)
            bc = bc * Tb + WATER_SCATTER * (1 - Tb) * 0.6
            F = np.where(tir, 1.0, _fresnel(cosi))[..., None]
            col[j] = F * bc + (1 - F) * ac
    if below:
        L = np.minimum(R["t"], 40.0)[..., None]
        T = np.exp(-SIGMA * L * 1.2)
        E0 = sample(LF, ox, oy, RES) * lights + 0.05
        col = col * T * np.array([0.7, 0.92, 0.95], np.float32) + WATER_SCATTER * min(E0, 0.6) * 0.8 * (1 - T)
        # light shafts from the surface
        sh = caustics(X / f * 2 + ox, Y / f * 2 + oy, t * 0.6, 3.0) * np.clip(dz, 0, 1) * 0.05
        col = col + sh[..., None] * np.array([0.4, 0.9, 1.0], np.float32)
    col = col * exposure
    col = col * (1 + col / 4.0) / (1 + col)          # gentle filmic shoulder
    img = l2s(np.clip(col, 0, 1))
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR), R, below

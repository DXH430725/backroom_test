"""
Ep06 — LEVEL 11 "The City That Never Sleeps": shading for the endless city.

Uses backrooms.city (2.5D heightfield raycaster). Facades get floors and window bays;
a third of the buildings are the indestructible kind with dark mirror windows that
reflect the sky; canals mirror the sky too. Golden hour by day; by night the windows
and street lamps light up to the horizon. Cars, billboards, the M.E.G. skywalk and
people are projected on top, depth-tested against the trace.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, px, s2l, l2s, value_noise
from backrooms.city import City, FAR, WALL, ROOF, GROUND, WATER, STREET, PARK, CANAL, BUILD

RS = 1.5
IW, IH = int(W / RS), int(H / RS)
CITY = City()
C = CITY.C
B = CITY.block

N_A = value_noise(512, 6, 111)
N_B = value_noise(256, 4, 112)

PALETTE = [s2l(c) for c in ((196, 186, 168), (168, 160, 150), (150, 110, 90), (200, 200, 196), (120, 124, 130))]
ROOFC = s2l((120, 118, 114))
ASPH = s2l((58, 58, 60))
WALK = s2l((140, 136, 128))
GRASS = s2l((80, 104, 56))


def street_coord(k):
    """world coordinate of the centre of street line k (k = 0 at the city centre)"""
    i = CITY.n // 2 // B * B + k * B
    return CITY.origin[0] + i * C + C / 2


def sky(dx, dy, dz, night=0.0):
    e = np.clip(dz, -0.2, 1)
    sun_az = math.radians(245)
    sx, sy, sz = math.sin(sun_az) * math.cos(0.2), math.cos(sun_az) * math.cos(0.2), math.sin(0.2)
    cs = np.clip(dx * sx + dy * sy + dz * sz, -1, 1)
    day = (s2l((250, 196, 140)) * (1 - np.clip(e * 3, 0, 1)[..., None]) + s2l((110, 140, 190)) * np.clip(e * 3, 0, 1)[..., None])
    day = day + s2l((255, 200, 120)) * (np.clip(cs, 0, 1) ** 24 * 2.5 + np.clip(cs, 0, 1) ** 4 * 0.35)[..., None]
    nit = s2l((12, 16, 30)) * (1 - np.clip(e * 4, 0, 1)[..., None]) * 1.6 + s2l((4, 6, 14)) * np.clip(e * 4, 0, 1)[..., None]
    nit = nit + s2l((70, 50, 40)) * np.exp(-np.clip(e, 0, 1) / 0.04)[..., None] * 0.6     # city glow on the haze
    return day * (1 - night) + nit * night


def _hash3(a, b, c):
    v = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791)
    return ((v * 2654435761) & 0xFFFFFFFF) / 4294967296.0


def shade(R, cam, t, night=0.0, haze=1800.0):
    k, tt = R["kind"], R["t"]
    hx, hy, hz = R["hx"], R["hy"], R["hz"]
    dx, dy, dz = R["dx"], R["dy"], R["dz"]
    I, J = R["i"], R["j"]
    wall = k == WALL
    roof = k == ROOF
    gnd = k == GROUND
    wat = k == WATER
    sun_az = math.radians(245)
    sx, sy, sz = math.sin(sun_az) * math.cos(0.2), math.cos(sun_az) * math.cos(0.2), math.sin(0.2)
    # --- facades
    nx = np.where(R["axis"], R["side"], 0).astype(np.float32)
    ny = np.where(R["axis"], 0, R["side"]).astype(np.float32)
    lam = np.clip(nx * sx + ny * sy, 0, 1)
    along = np.where(R["axis"], hy, hx)
    fl = np.floor(hz / 3.6).astype(np.int64)
    bay = np.floor(along / 3.0).astype(np.int64)
    fu = (along / 3.0) % 1
    fv = (hz / 3.6) % 1
    win = (fu > 0.16) & (fu < 0.84) & (fv > 0.28) & (fv < 0.86) & (hz > 4.0)
    tint = CITY.tint[I, J]
    pal = np.stack(PALETTE)[np.clip(tint, 0, 4)]
    n = N_A[np.floor(hz * 4).astype(np.int64) % 512, np.floor(along * 4).astype(np.int64) % 512]
    wcol = pal * (0.85 + 0.25 * n[..., None])
    mirror = CITY.mirror[I, J]
    refl = sky(dx * np.where(R["axis"], -1, 1), dy * np.where(R["axis"], 1, -1), dz, night)
    glass_day = refl * 0.35 + 0.02
    lit = _hash3(I * 100 + bay, J * 100 + fl, 7) < (0.08 + 0.55 * night)
    warm = np.array([1.0, 0.75, 0.42], np.float32) * (0.5 + 0.8 * _hash3(I, J * 3 + fl, bay)[..., None])
    glass = np.where(lit[..., None] & (night > 0.05), warm * night + glass_day * (1 - night), glass_day)
    E_wall = (0.32 + 0.9 * lam * (1 - night))[..., None] * (0.55 + 0.45 * np.clip(hz / 40, 0, 1))[..., None]
    E_wall = E_wall * (1 - 0.82 * night)
    facade = np.where(win[..., None], glass, wcol * E_wall)
    mirror_col = refl * 0.75 + 0.015
    mframe = ((fu < 0.04) | (fv < 0.05))
    mirror_col = np.where(mframe[..., None], s2l((40, 40, 44)) * (1 - 0.8 * night), mirror_col)
    facade = np.where(mirror[..., None], mirror_col, facade)
    # --- roofs
    rn = N_A[np.floor(hy * 2).astype(np.int64) % 512, np.floor(hx * 2).astype(np.int64) % 512]
    rcol = ROOFC * (0.8 + 0.35 * rn[..., None]) * (0.5 + 0.7 * sz * 3 * (1 - night) + 0.05)
    rcol = rcol * (1 - 0.85 * night)
    # --- ground
    kind = CITY.kind[I, J]
    lx = ((hx - CITY.origin[0]) / C) % 1
    ly = ((hy - CITY.origin[1]) / C) % 1
    vert = (I % B == 0) & (J % B != 0)
    horz = (J % B == 0) & (I % B != 0)
    dash = ((np.where(vert, hy, hx) / 6.0) % 1) < 0.5
    line = (vert & (np.abs(lx - 0.5) < 0.012) & dash) | (horz & (np.abs(ly - 0.5) < 0.012) & dash)
    side = (vert & ((lx < 0.17) | (lx > 0.83))) | (horz & ((ly < 0.17) | (ly > 0.83)))
    gn = N_B[np.floor(hy * 3).astype(np.int64) % 256, np.floor(hx * 3).astype(np.int64) % 256]
    gcol = ASPH * (0.85 + 0.3 * gn[..., None])
    gcol = np.where(side[..., None], WALK * (0.85 + 0.2 * gn[..., None]), gcol)
    gcol = np.where(line[..., None], s2l((220, 210, 170)), gcol)
    gcol = np.where((kind == PARK)[..., None], GRASS * (0.75 + 0.5 * gn[..., None]), gcol)
    Eg = (0.35 + 0.6 * sz * (1 - night)) * (1 - 0.88 * night)
    # street lamps at night: pools of light along the kerbs every 24 m
    lampd = np.minimum(np.abs(((np.where(vert, hy, hx)) % 24) - 12), 12)
    pool = np.exp(-(lampd / 4.0) ** 2) * (vert | horz)
    gcol = gcol * (Eg + 0.6 * night * pool)[..., None]
    # --- canals
    ccol = sky(dx, dy, -dz, night) * 0.6 + s2l((20, 40, 40)) * 0.4
    col = np.where(wall[..., None], facade, np.where(roof[..., None], rcol, np.where(gnd[..., None], gcol, ccol)))
    # aerial perspective
    hzc = sky(dx, dy, np.full_like(dz, 0.02), night)
    fog = np.exp(-tt / haze)[..., None]
    col = col * fog + hzc * (1 - fog)
    skyc = sky(dx, dy, dz, night)
    col = np.where((k == FAR)[..., None], skyc, col)
    return col


def render(cam, t, night=0.0, haze=1800.0, tmax=2600.0):
    X = (np.arange(IW, dtype=np.float32) + 0.5) - IW / 2
    Y = IH / 2 - (np.arange(IH, dtype=np.float32) + 0.5)
    X, Y = np.meshgrid(X, Y)
    f = cam.f / RS
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    nn = np.sqrt(dx * dx + dy * dy + dz * dz)
    R = CITY.trace(cam.pos, dx / nn, dy / nn, dz / nn, tmax=tmax)
    col = shade(R, cam, t, night, haze)
    col = col * (1 + col / 6.0) / (1 + col)
    img = l2s(np.clip(col * 1.25, 0, 1))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(np.where(R["kind"] == FAR, 1e6, R["t"]).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return img, depth


# ---------------------------------------------------------------------------
# props
# ---------------------------------------------------------------------------
def _quad(out, depth, cam, pts, col, alpha=1.0):
    qx, qy, qd = cam.project(np.asarray(pts, np.float64))
    if (qd < 0.2).any():
        return
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.stack([qx * 8, qy * 8], 1).astype(np.int32)], 255, cv2.LINE_AA, 3)
    a = m.astype(np.float32) / 255 * (depth > qd.min() - 0.3) * alpha
    out[:] = out * (1 - a[..., None]) + np.array(col, np.float32) * a[..., None]


def car(out, depth, cam, x, y, yaw, color, night=0.0):
    ca, sa = math.cos(yaw), math.sin(yaw)

    def P(a, b, z):
        return (x + a * ca - b * sa, y + a * sa + b * ca, z)
    L, Wd = 2.2, 0.9
    body = [P(-L, -Wd, 0.25), P(L, -Wd, 0.25), P(L, Wd, 0.25), P(-L, Wd, 0.25)]
    top = lambda p: (p[0], p[1], 0.95)
    cab = [P(-1.0, -0.8, 0.95), P(0.9, -0.8, 0.95), P(0.6, -0.75, 1.45), P(-0.9, -0.75, 1.45)]
    cp = np.array(cam.pos)
    faces = [([body[0], body[1], top(body[1]), top(body[0])], 0.7), ([body[1], body[2], top(body[2]), top(body[1])], 0.8),
             ([body[2], body[3], top(body[3]), top(body[2])], 0.6), ([body[3], body[0], top(body[0]), top(body[3])], 0.75),
             ([top(b) for b in body], 1.0)]
    faces.sort(key=lambda f: -np.linalg.norm(np.mean(np.array(f[0]), 0) - cp))
    k = 1 - 0.85 * night
    for pts, sh in faces:
        _quad(out, depth, cam, pts, tuple(c * sh * k for c in color))
    cab2 = [(p[0], p[1], p[2]) for p in cab]
    _quad(out, depth, cam, cab2, (0.08 * k, 0.09 * k, 0.1 * k))
    _quad(out, depth, cam, [P(-1.0, 0.8, 0.95), P(0.9, 0.8, 0.95), P(0.6, 0.75, 1.45), P(-0.9, 0.75, 1.45)], (0.08 * k, 0.09 * k, 0.1 * k))
    _quad(out, depth, cam, [P(-0.9, -0.75, 1.45), P(0.6, -0.75, 1.45), P(0.6, 0.75, 1.45), P(-0.9, 0.75, 1.45)], tuple(c * 0.9 * k for c in color))


def billboard(out, depth, cam, corners, art):
    """perspective-warp a small RGB image onto a world-space quad (tl, tr, br, bl)"""
    qx, qy, qd = cam.project(np.asarray(corners, np.float64))
    if (qd < 0.5).any():
        return
    if qx[0] > qx[1]:                 # seen from behind the quad's orientation: keep the art readable
        art = art[:, ::-1].copy()
    h, w = art.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(np.stack([qx, qy], 1)))
    warped = cv2.warpPerspective(art, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    a = mask * (depth > qd.min() - 1.0)
    out[:] = out * (1 - a[..., None]) + warped * a[..., None]


def walkway(out, depth, cam, p0, p1, z, width=4.0, hgt=3.2, night=0.0):
    """the elevated walkway between Base Beta and Camp Amber: a glazed box bridge"""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d) * width / 2
    a, b, c, e = p0 + nrm, p1 + nrm, p1 - nrm, p0 - nrm
    k = 1 - 0.8 * night
    side = (0.55 * k, 0.6 * k, 0.65 * k)
    glow = (0.9, 0.75, 0.45) if night > 0.5 else (0.35, 0.45, 0.55)
    cp = np.array(cam.pos[:2])
    faces = [([(*a, z), (*b, z), (*b, z + hgt), (*a, z + hgt)], glow), ([(*e, z), (*c, z), (*c, z + hgt), (*e, z + hgt)], glow),
             ([(*a, z + hgt), (*b, z + hgt), (*c, z + hgt), (*e, z + hgt)], side), ([(*a, z), (*b, z), (*c, z), (*e, z)], side)]
    faces.sort(key=lambda f: -np.linalg.norm(np.mean(np.array(f[0])[:, :2], 0) - cp))
    for pts, colr in faces:
        _quad(out, depth, cam, pts, colr)


def person(out, depth, cam, x, y, h=1.72, col=(0.2, 0.2, 0.22), face=None):
    pts = np.array([[x - 0.1, y, 0], [x - 0.1, y, 0.5 * h], [x + 0.1, y, 0], [x + 0.1, y, 0.5 * h],
                    [x, y, 0.5 * h], [x, y, 0.82 * h], [x, y, 0.93 * h],
                    [x - 0.21, y, 0.8 * h], [x + 0.21, y, 0.8 * h], [x - 0.26, y, 0.5 * h], [x + 0.26, y, 0.5 * h]])
    qx, qy, qd = cam.project(pts)
    if (qd < 0.3).any():
        return
    s = cam.f / qd[4]
    m = np.zeros((H, W), np.uint8)
    L = cv2.LINE_AA
    p = lambda k: (int(qx[k] * 8), int(qy[k] * 8))
    th = lambda r: max(1, int(r * s))
    cv2.line(m, p(0), p(1), 255, th(0.14), L, 3)
    cv2.line(m, p(2), p(3), 255, th(0.14), L, 3)
    cv2.line(m, p(4), p(5), 255, th(0.42), L, 3)
    cv2.line(m, p(7), p(9), 255, th(0.1), L, 3)
    cv2.line(m, p(8), p(10), 255, th(0.1), L, 3)
    cv2.circle(m, p(6), int(0.12 * s * 8), 255, -1, L, 3)
    a = m.astype(np.float32) / 255 * (depth > qd[4] - 0.5)
    out[:] = out * (1 - a[..., None]) + np.array(col, np.float32) * a[..., None]
    if face is not None:   # a faceling: a pale, smooth head
        hm = np.zeros((H, W), np.uint8)
        cv2.circle(hm, p(6), int(0.11 * s * 8), 255, -1, L, 3)
        ha = hm.astype(np.float32) / 255 * (depth > qd[4] - 0.5)
        out[:] = out * (1 - ha[..., None]) + np.array(face, np.float32) * ha[..., None]


def hound(out, depth, cam, x, y, yaw, t, col=(0.08, 0.08, 0.08)):
    """Entity 8: a gaunt, crawling human shape"""
    ca, sa = math.cos(yaw), math.sin(yaw)
    g = math.sin(t * 6)

    def P(a, b, z):
        return (x + a * ca - b * sa, y + a * sa + b * ca, z)
    pts = np.array([P(0.6, 0, 0.55), P(-0.5, 0, 0.6), P(0.85, 0, 0.62), P(0.5, 0.2, 0), P(0.5, -0.2, 0), P(-0.45, 0.2, 0), P(-0.45, -0.2, 0)])
    qx, qy, qd = cam.project(pts)
    if (qd < 0.3).any():
        return
    s = cam.f / qd[0]
    m = np.zeros((H, W), np.uint8)
    L = cv2.LINE_AA
    p = lambda k: (int(qx[k] * 8), int(qy[k] * 8))
    cv2.line(m, p(0), p(1), 255, max(1, int(0.26 * s)), L, 3)
    cv2.circle(m, p(2), int(0.13 * s * 8), 255, -1, L, 3)
    for leg, root in ((3, 0), (4, 0), (5, 1), (6, 1)):
        k2 = 0.08 * g * (1 if leg % 2 else -1)
        a = cam.project(np.array([[pts[leg][0] + k2, pts[leg][1], 0]]))
        cv2.line(m, p(root), (int(a[0][0] * 8), int(a[1][0] * 8)), 255, max(1, int(0.08 * s)), L, 3)
    a = m.astype(np.float32) / 255 * (depth > qd[0] - 0.5)
    out[:] = out * (1 - a[..., None]) + np.array(col, np.float32) * a[..., None]

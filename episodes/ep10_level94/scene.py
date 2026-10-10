"""
Ep10 — LEVEL 94 "Motion": a stop-motion clay diorama that goes on forever.

A ray-marched signed-distance world: rolling clay hills, a 1930s toy town on a grid
(rounded houses with crooked roofs, a fountain of almond water, siren poles, milk vans),
stray props in the hills. Shading is a soft two-band "clay under studio lights" model
with one crisp key light and soft shadows, smudged surfaces and fingerprints.
The Animations are flat black-and-white rubber-hose cartoons, drawn as boiling 2D
sprites standing in the 3D world — they are the only thing here that isn't clay.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, px, s2l, l2s, value_noise

RS = 2.5
IW, IH = int(W / RS), int(H / RS)

N_A = value_noise(512, 6, 941)
N_B = value_noise(256, 5, 942)
N_C = value_noise(512, 4, 943)

CELL = 9.0
TOWN_R = 5          # cells
SKY_DAY = (np.array([0.30, 0.55, 0.92]), np.array([0.85, 0.90, 0.95]))


def _tex(T, x, y):
    n = T.shape[0]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ax, ay = x - x0, y - y0
    x0m, y0m, x1m, y1m = x0 % n, y0 % n, (x0 + 1) % n, (y0 + 1) % n
    return T[y0m, x0m] * (1 - ax) * (1 - ay) + T[y0m, x1m] * ax * (1 - ay) + T[y1m, x0m] * (1 - ax) * ay + T[y1m, x1m] * ax * ay


def _hash(a, b, k=0):
    v = np.sin(a * 127.1 + b * 311.7 + k * 74.7) * 43758.5453
    return v - np.floor(v)


# ---------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------
def is_street(ci, cj):
    return (ci % 3 == 0) | (cj % 4 == 0)


def ground_h(x, y):
    r = np.sqrt(x * x + y * y)
    out = 0.15 * (_tex(N_C, x * 6, y * 6) - 0.5)
    far = r > 52.0
    if far.any():
        xf, yf = x[far], y[far]
        town = np.clip((r[far] - 52.0) / 40.0, 0, 1)
        town = town * town * (3 - 2 * town)
        hills = (_tex(N_A, xf * 0.9 + 100, yf * 0.9) - 0.5) * 22.0 + (_tex(N_B, xf * 2.5, yf * 2.5) - 0.5) * 7.0
        hills = hills + 10.0 * np.exp(-(xf * xf + (yf + 118.0) ** 2) / 45.0 ** 2)     # the knoll she arrives on
        out = out.copy()
        out[far] += hills * town
    return out


def _sd_rbox(px_, py_, pz_, bx, by, bz, r):
    qx, qy, qz = np.abs(px_) - bx + r, np.abs(py_) - by + r, np.abs(pz_) - bz + r
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2) + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r


def _sd_cyl(px_, py_, pz_, r, h):
    """vertical cylinder centred at origin, half-height h"""
    d1 = np.sqrt(px_ * px_ + py_ * py_) - r
    d2 = np.abs(pz_) - h
    return np.minimum(np.maximum(d1, d2), 0) + np.sqrt(np.maximum(d1, 0) ** 2 + np.maximum(d2, 0) ** 2)


def house_params(ci, cj):
    h = lambda k: _hash(ci, cj, k)
    w = 3.4 + 1.6 * h(1)
    d = 3.0 + 1.2 * h(2)
    hw = 2.4 + 1.4 * h(3)
    rh = 1.6 + 1.3 * h(4)
    rot = (h(5) - 0.5) * 0.25
    lean = (h(6) - 0.5) * 0.12
    ox = (h(7) - 0.5) * 1.2
    oy = (h(8) - 0.5) * 1.2
    return w, d, hw, rh, rot, lean, ox, oy


def _houses(x, y, z):
    ci = np.round(x / CELL)
    cj = np.round(y / CELL)
    inside = (np.abs(ci) <= TOWN_R) & (np.abs(cj) <= TOWN_R) & ~is_street(ci.astype(int), cj.astype(int))
    w, dd, hw, rh, rot, lean, ox, oy = house_params(ci, cj)
    lx = x - ci * CELL - ox
    ly = y - cj * CELL - oy
    c, s = np.cos(rot), np.sin(rot)
    lx, ly = c * lx - s * ly, s * lx + c * ly
    lz = z
    lx = lx - lean * lz             # crooked: the whole house leans
    body = _sd_rbox(lx, ly, lz - hw / 2, w / 2, dd / 2, hw / 2, 0.35)
    rz = lz - hw                    # roof: a squashed gable prism along x, with eaves
    slope = rh / (dd / 2 + 0.45)
    nn = 1 / np.sqrt(1 + slope * slope)
    roof = np.maximum(np.maximum((np.abs(ly) * slope + rz - rh) * nn, -rz), np.abs(lx) - w / 2 - 0.35) - 0.08
    chim = _sd_rbox(lx - w * 0.25, ly - dd * 0.15, lz - hw - rh * 0.75, 0.32, 0.32, rh * 0.45, 0.1)
    hd = np.where(inside, np.minimum(np.minimum(body, roof), chim), 1e3)
    m = np.where(hd == body, 1, np.where(hd == roof, 2, 3)).astype(np.int8)
    # never step further than the next cell's edge
    bnd = np.minimum(CELL / 2 - np.abs(x - ci * CELL), CELL / 2 - np.abs(y - cj * CELL)) + 0.6
    return hd, np.minimum(hd, bnd), m


def _fountain(x, y, z):
    basin = np.maximum(_sd_cyl(x, y, z - 0.35, 3.0, 0.35), -_sd_cyl(x, y, z - 0.6, 2.65, 0.4)) - 0.08
    col = _sd_cyl(x, y, z - 1.2, 0.32, 1.2) - 0.05
    bowl = np.maximum(_sd_cyl(x, y, z - 2.35, 1.1, 0.16), -_sd_cyl(x, y, z - 2.5, 0.95, 0.12)) - 0.05
    top = np.sqrt(x * x + y * y + (z - 2.8) ** 2) - 0.32
    stone = np.minimum(np.minimum(basin, col), np.minimum(bowl, top))
    water = np.maximum(_sd_cyl(x, y, z - 0.5, 2.7, 0.04), -stone)
    d = np.minimum(stone, water)
    return d, np.where(water < stone, 5, 4).astype(np.int8)


def _pole(x, y, z):
    pj = np.round(y / 18.0)
    pyl = y - pj * 18.0
    on = (np.abs(pj) <= 3) & (pj != 0)
    pxl = x - 3.3
    pole = _sd_cyl(pxl, pyl, z - 3.2, 0.13, 3.2)
    horns = np.minimum(_sd_rbox(pxl + 0.3, pyl, z - 6.1, 0.35, 0.16, 0.16, 0.12), _sd_rbox(pxl, pyl + 0.3, z - 6.1, 0.16, 0.35, 0.16, 0.12))
    horns = np.minimum(horns, np.sqrt(pxl * pxl + pyl * pyl + (z - 6.45) ** 2) - 0.2)
    return np.where(on, np.minimum(pole, horns), 1e3), np.full(x.shape, 6, np.int8)


def _van(vx, vy, vr):
    def f(x, y, z):
        ax_, ay_ = x - vx, y - vy
        cc, ss = math.cos(vr), math.sin(vr)
        ax_, ay_ = cc * ax_ - ss * ay_, ss * ax_ + cc * ay_
        vb = _sd_rbox(ax_, ay_, z - 1.35, 1.0, 1.9, 0.8, 0.4)
        cab = _sd_rbox(ax_, ay_ - 1.9, z - 1.05, 0.95, 0.8, 0.55, 0.35)
        wheels = np.minimum(_sd_rbox(ax_, ay_ - 1.6, z - 0.42, 1.1, 0.4, 0.4, 0.38), _sd_rbox(ax_, ay_ + 1.3, z - 0.42, 1.1, 0.4, 0.4, 0.38))
        vd = np.minimum(vb, cab)
        return np.minimum(vd, wheels), np.where(wheels < vd, 9, 7).astype(np.int8)
    return f


def _prop(kind, qx, qy):
    def f(x, y, z):
        gx, gy = x - qx, y - qy
        gz = z - PROP_Z[kind]
        if kind == "wall":
            pd2 = _sd_rbox(gx, gy, gz - 1.6, 2.6, 0.18, 1.6, 0.06)
            sofa = np.minimum(_sd_rbox(gx, gy + 1.0, gz - 0.35, 1.2, 0.45, 0.32, 0.18), _sd_rbox(gx, gy + 0.6, gz - 0.75, 1.2, 0.15, 0.45, 0.12))
            lamp = np.minimum(_sd_cyl(gx - 1.9, gy + 0.8, gz - 0.8, 0.05, 0.8), _sd_cyl(gx - 1.9, gy + 0.8, gz - 1.75, 0.28, 0.2) - 0.03)
            pd2 = np.minimum(pd2, np.minimum(sofa, lamp))
        elif kind == "car":
            pd2 = np.minimum(_sd_rbox(gx, gy, gz - 0.7, 0.9, 2.0, 0.4, 0.35), _sd_rbox(gx, gy + 0.3, gz - 1.3, 0.75, 0.9, 0.35, 0.3))
            pd2 = np.minimum(pd2, np.minimum(_sd_rbox(gx, gy - 1.3, gz - 0.35, 1.0, 0.33, 0.33, 0.32), _sd_rbox(gx, gy + 1.3, gz - 0.35, 1.0, 0.33, 0.33, 0.32)))
        else:   # water tower
            tank = _sd_cyl(gx, gy, gz - 9.5, 2.6, 1.6) - 0.3
            cap = np.sqrt(gx * gx + gy * gy + (gz - 11.0) ** 2 * 3.0) - 2.4
            legs = 1e3
            for lxo, lyo in ((1.7, 1.7), (-1.7, 1.7), (1.7, -1.7), (-1.7, -1.7)):
                legs = np.minimum(legs, _sd_cyl(gx - lxo, gy - lyo, gz - 4.0, 0.16, 4.0))
            pd2 = np.minimum(np.minimum(tank, cap), legs)
        return pd2, np.full(x.shape, 8, np.int8)
    return f


def _components():
    """(bounding centre, radius, fn) for everything except the ground and houses"""
    comps = [((0.0, 0.0, 1.5), 4.2, _fountain)]
    for (vx, vy, vr) in VANS:
        comps.append(((vx, vy, 1.2), 3.4, _van(vx, vy, vr)))
    for kind, (qx, qy) in PROPS:
        comps.append(((qx, qy, PROP_Z[kind] + (6.5 if kind == "tower" else 1.5)), 7.5 if kind == "tower" else 3.8, _prop(kind, qx, qy)))
    return comps


def sdf(x, y, z, want_mat=False):
    """distance to the clay; material ids: 0 ground, 1 wall, 2 roof, 3 chimney, 4 fountain stone, 5 almond water,
    6 pole, 7 van, 8 prop (hill furniture), 9 van wheel / tyre"""
    d = (z - ground_h(x, y)) * 0.7
    mat = np.zeros(x.shape, np.int8) if want_mat else None
    # houses: only near the town and below the chimneys
    ext = TOWN_R * CELL + CELL
    near = (np.abs(x) < ext) & (np.abs(y) < ext) & (z < 9.5)
    if near.any():
        hd, hstep, hm = _houses(x[near], y[near], z[near])
        if want_mat:
            mm = mat[near]
            mat[near] = np.where(hd < d[near], hm, mm)
        d[near] = np.minimum(d[near], hstep)
    above = (np.abs(x) < ext) & (np.abs(y) < ext) & (z >= 9.5)
    d = np.where(above, np.minimum(d, z - 9.0), d)
    # siren poles (a thin strip along the main street)
    near = (np.abs(x - 3.3) < 1.2) & (np.abs(y) < 60) & (z < 7.2)
    if near.any():
        pd, pm = _pole(x[near], y[near], z[near])
        if want_mat:
            mat[near] = np.where(pd < d[near], pm, mat[near])
        d[near] = np.minimum(d[near], pd)
    side = (np.abs(x - 3.3) >= 1.2) & (np.abs(x - 3.3) < 3.0) & (np.abs(y) < 60) & (z < 7.2)
    d = np.where(side, np.minimum(d, np.abs(x - 3.3) - 1.0), d)
    # everything else, each inside its bounding sphere
    for (cx, cy, cz), R, fn in COMPS:
        bd = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - R
        near = bd < 0.4
        if near.any():
            cd, cm = fn(x[near], y[near], z[near])
            if want_mat:
                mat[near] = np.where(cd < d[near], cm, mat[near])
            d[near] = np.minimum(d[near], cd)
        far = ~near
        d = np.where(far, np.minimum(d, bd + 0.3), d)
    return d, mat


VANS = [(-2.2, -14.0, 0.05), (2.4, 22.0, math.pi + 0.08)]
PROPS = [("wall", (11.0, -96.0)), ("car", (-14.0, -84.0)), ("tower", (52.0, -66.0))]
PROP_Z = {}


def _settle_props():
    for kind, (qx, qy) in PROPS:
        PROP_Z[kind] = float(ground_h(np.array([qx]), np.array([qy]))[0]) - 0.05


_settle_props()
COMPS = _components()

# the door that leads to Level 9: the street-facing door of the house in cell (1, 2)
DOOR_CELL = (1, -2)


def door_world():
    ci, cj = DOOR_CELL
    w, dd, hw, rh, rot, lean, ox, oy = [float(v) for v in house_params(np.array([ci]), np.array([cj]))]
    c, s = math.cos(-rot), math.sin(-rot)
    lx, ly = -w / 2, 0.0             # the -x face of the house (toward the main street)
    wx = c * lx - s * ly + ci * CELL + ox
    wy = s * lx + c * ly + cj * CELL + oy
    return np.array([wx - 0.05, wy, 0.0])


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
PALETTE_WALL = [(240, 222, 170), (246, 196, 160), (176, 222, 196), (190, 210, 236), (244, 176, 170), (236, 236, 214),
                (214, 190, 230)]
PALETTE_ROOF = [(186, 70, 58), (60, 132, 140), (110, 72, 50), (70, 96, 160), (150, 60, 90)]


def sky_color(dx, dy, dz, tod, t):
    """tod: 0 day, 1 dusk, 2 night"""
    up = np.clip(dz * 2.2, 0, 1)
    day = SKY_DAY[1] * (1 - up[..., None]) + SKY_DAY[0] * up[..., None]
    dusk = np.array([1.0, 0.62, 0.42]) * (1 - up[..., None]) + np.array([0.42, 0.30, 0.55]) * up[..., None]
    night = np.array([0.05, 0.07, 0.16]) * (1 - up[..., None]) + np.array([0.012, 0.016, 0.05]) * up[..., None]
    a = np.clip(tod, 0, 1)
    b = np.clip(tod - 1, 0, 1)
    sky = day * (1 - a) + dusk * a
    sky = sky * (1 - b) + night * b
    return (sky ** 2.2).astype(np.float32)


def light_setup(tod):
    """sun direction and colours for day / dusk / night"""
    a, b = np.clip(tod, 0, 1), np.clip(tod - 1, 0, 1)
    sun_day = np.array([0.45, -0.35, 0.82])
    sun_dusk = np.array([-0.85, 0.15, 0.22])
    moon = np.array([0.3, 0.6, 0.74])
    L = sun_day * (1 - a) + sun_dusk * a
    L = L * (1 - b) + moon * b
    L = L / np.linalg.norm(L)
    cday, cdusk, cnight = np.array([1.0, 0.95, 0.86]) * 2.1, np.array([1.0, 0.55, 0.32]) * 2.0, np.array([0.35, 0.45, 0.8]) * 0.35
    C = cday * (1 - a) + cdusk * a
    C = C * (1 - b) + cnight * b
    amb_day, amb_dusk, amb_night = np.array([0.42, 0.52, 0.7]), np.array([0.45, 0.32, 0.45]), np.array([0.05, 0.07, 0.14])
    A = amb_day * (1 - a) + amb_dusk * a
    A = A * (1 - b) + amb_night * b
    return L, C, A


def render(cam, t, tod=0.0, steps=110, tmax=320.0, jitter=0.0, lamps_on=0.0):
    X = (np.arange(IW, dtype=np.float32) + 0.5) * RS - W / 2
    Y = H / 2 - (np.arange(IH, dtype=np.float32) + 0.5) * RS
    X, Y = np.meshgrid(X, Y)
    f = cam.f
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    nn = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = (dx / nn).ravel(), (dy / nn).ravel(), (dz / nn).ravel()
    ox, oy, oz = cam.pos
    n = dx.size
    tt = np.full(n, 0.05)
    idx = np.arange(n)
    hit = np.zeros(n, bool)
    for _ in range(steps):
        if idx.size == 0:
            break
        p0, p1, p2 = ox + dx[idx] * tt[idx], oy + dy[idx] * tt[idx], oz + dz[idx] * tt[idx]
        d, _ = sdf(p0, p1, p2)
        done = d < 0.004 * tt[idx] + 0.01
        hit[idx[done]] = True
        tt[idx] += np.where(done, 0, np.maximum(d * 0.8, 0.01))
        keep = ~done & (tt[idx] < tmax)
        idx = idx[keep]
    sky = ~hit
    tt = np.minimum(tt, tmax)
    px_, py_, pz_ = ox + dx * tt, oy + dy * tt, oz + dz * tt
    d0, mat = sdf(px_, py_, pz_, True)
    e = 0.02 + 0.0015 * tt
    nx = sdf(px_ + e, py_, pz_)[0] - sdf(px_ - e, py_, pz_)[0]
    ny = sdf(px_, py_ + e, pz_)[0] - sdf(px_, py_ - e, pz_)[0]
    nz = sdf(px_, py_, pz_ + e)[0] - sdf(px_, py_, pz_ - e)[0]
    nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    # clay: thumb-smoothed bumps nudge the normal
    bump = 0.18
    bx_ = _tex(N_C, px_ * 30 + pz_ * 13, py_ * 30) - 0.5
    by_ = _tex(N_C, py_ * 30 + 77, pz_ * 30 + px_ * 11) - 0.5
    nx, ny = nx + bump * bx_, ny + bump * by_
    nl = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    # albedo by material
    ci = np.round(px_ / CELL)
    cj = np.round(py_ / CELL)
    hsh = _hash(ci, cj, 11)
    alb = np.zeros((n, 3), np.float32)
    PW = np.array([s2l(c) for c in PALETTE_WALL], np.float32)
    PR = np.array([s2l(c) for c in PALETTE_ROOF], np.float32)
    wall_c = PW[(hsh * len(PALETTE_WALL)).astype(int) % len(PALETTE_WALL)]
    roof_c = PR[(_hash(ci, cj, 12) * len(PALETTE_ROOF)).astype(int) % len(PALETTE_ROOF)]
    # ground: grass outside, cobbled streets and plaza in town
    r = np.sqrt(px_ ** 2 + py_ ** 2)
    in_town = (np.abs(ci) <= TOWN_R) & (np.abs(cj) <= TOWN_R)
    street = in_town & is_street(ci.astype(int), cj.astype(int))
    grass = s2l((84, 156, 66)) * (0.8 + 0.35 * _tex(N_B, px_ * 3, py_ * 3))[:, None]
    tool = 0.95 + 0.05 * np.sin(px_ * 9 + 2 * _tex(N_A, px_ * 4, py_ * 4) * 6)       # sculpting-tool strokes
    grass = grass * tool[:, None]
    cob_u, cob_v = px_ * 2.2 + 0.5 * np.floor(py_ * 2.2) % 2, py_ * 2.2
    cob = (np.abs((cob_u % 1.0) - 0.5) > 0.42) | (np.abs((cob_v % 1.0) - 0.5) > 0.42)
    stonec = s2l((206, 190, 160)) * (0.85 + 0.25 * _tex(N_C, np.floor(cob_u) * 9.1, np.floor(cob_v) * 7.3))[:, None]
    stonec = np.where(cob[:, None], s2l((150, 134, 110)), stonec)
    gcol = np.where(street[:, None] | (in_town & (r < 7.0))[:, None], stonec, grass)
    alb = np.where((mat == 0)[:, None], gcol, alb)
    alb = np.where((mat == 1)[:, None], wall_c, alb)
    alb = np.where((mat == 2)[:, None], roof_c * (0.85 + 0.15 * (((pz_ * 5) % 1.0) > 0.2))[:, None], alb)
    alb = np.where((mat == 3)[:, None], s2l((160, 90, 70)), alb)
    alb = np.where((mat == 4)[:, None], s2l((222, 214, 196)), alb)
    alb = np.where((mat == 5)[:, None], s2l((246, 236, 214)), alb)
    alb = np.where((mat == 6)[:, None], s2l((200, 60, 50)), alb)
    alb = np.where((mat == 7)[:, None], s2l((248, 246, 236)), alb)
    alb = np.where((mat == 9)[:, None], s2l((40, 36, 36)), alb)
    alb = np.where((mat == 8)[:, None], s2l((200, 120, 90)), alb)
    # windows and doors on the houses (local coordinates)
    w, dd, hw, rh, rot, lean, ox_, oy_ = house_params(ci, cj)
    lx = px_ - ci * CELL - ox_
    ly = py_ - cj * CELL - oy_
    c, s = np.cos(rot), np.sin(rot)
    lx, ly = c * lx - s * ly, s * lx + c * ly
    lx = lx - lean * pz_
    onx = (np.abs(np.abs(lx) - w / 2) < 0.08) & (mat == 1)     # faces toward ±x (streets)
    ony = (np.abs(np.abs(ly) - dd / 2) < 0.08) & (mat == 1)
    u = np.where(onx, ly, lx)
    half = np.where(onx, dd / 2, w / 2)
    win = (np.abs(np.abs(u) - half * 0.55) < 0.42) & (np.abs(pz_ - hw * 0.62) < 0.45) & (onx | ony)
    door = onx & (np.abs(u) < 0.45) & (pz_ < 1.75)
    win &= ~door
    frame = (onx | ony) & ((np.abs(np.abs(u) - half * 0.55) < 0.52) & (np.abs(pz_ - hw * 0.62) < 0.55)) & ~win & ~door
    glass_day = s2l((70, 110, 150))
    glass_night = s2l((255, 200, 110))
    lit_win = _hash(ci, cj, 21) < 0.0      # nobody is home
    wc = glass_day * (1 - min(1.0, tod)) + s2l((30, 30, 50)) * min(1.0, tod)
    alb = np.where(win[:, None], wc, alb)
    alb = np.where(frame[:, None], s2l((250, 250, 244)), alb)
    special = (ci == DOOR_CELL[0]) & (cj == DOOR_CELL[1])
    dcol = np.where(special[:, None], s2l((70, 44, 30)), s2l((150, 80, 60)))
    alb = np.where(door[:, None], dcol, alb)
    # fingerprints: little whorls pressed into walls and roofs
    fpc_x, fpc_y = np.floor(px_ * 0.7 + pz_ * 0.3), np.floor(py_ * 0.7 - pz_ * 0.2)
    fx0 = (fpc_x + _hash(fpc_x, fpc_y, 31)) / 0.7
    fy0 = (fpc_y + _hash(fpc_x, fpc_y, 32)) / 0.7
    fr = np.sqrt((px_ - fx0) ** 2 + (py_ - fy0) ** 2)
    whorl = (fr < 0.38) & (_hash(fpc_x, fpc_y, 33) < 0.35) & ((mat == 1) | (mat == 2) | (mat == 4))
    ridge = 0.5 + 0.5 * np.sin(fr * 95 + 3 * _tex(N_B, px_ * 40, py_ * 40))
    alb = alb * np.where(whorl, 0.93 + 0.07 * ridge, 1.0)[:, None]
    # lighting: key light (with soft shadow), sky fill, AO
    L, C, A = light_setup(tod)
    lam = nx * L[0] + ny * L[1] + nz * L[2]
    key = np.clip(lam * 1.6 + 0.15, 0, 1)
    key = 0.25 * key + 0.75 * (key > 0.42)          # two bands, softened below
    # soft shadow
    sh = np.ones(n, np.float32)
    st = np.full(n, 0.08)
    sidx = np.nonzero(hit & (lam > 0))[0]
    for _ in range(28):
        if sidx.size == 0:
            break
        q0 = px_[sidx] + L[0] * st[sidx] + nx[sidx] * 0.03
        q1 = py_[sidx] + L[1] * st[sidx] + ny[sidx] * 0.03
        q2 = pz_[sidx] + L[2] * st[sidx] + nz[sidx] * 0.03
        dd_, _ = sdf(q0, q1, q2)
        sh[sidx] = np.minimum(sh[sidx], np.clip(10 * dd_ / st[sidx], 0, 1))
        st[sidx] += np.clip(dd_, 0.06, 3.0)
        keep = (sh[sidx] > 0.02) & (st[sidx] < 40)
        sidx = sidx[keep]
    sh = sh * sh * (3 - 2 * sh)
    ao = np.clip(sdf(px_ + nx * 0.35, py_ + ny * 0.35, pz_ + nz * 0.35)[0] / 0.35, 0.25, 1.0)
    flick = 1.0 + jitter
    col = alb * ((key * sh)[:, None] * C[None] * flick + (A[None] * (0.6 + 0.4 * nz)[:, None]) * ao[:, None])
    # clay sheen: a broad soft highlight
    hx, hy, hz = L[0] - dx, L[1] - dy, L[2] - dz
    hn = np.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
    spec = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0, 1) ** 24 * 0.12 * sh
    col = col + spec[:, None] * C[None] * flick
    # almond water: a pale, slow shimmer
    wat = mat == 5
    col = np.where(wat[:, None], col * (0.9 + 0.2 * np.sin(px_ * 5 + py_ * 4 + t * 2))[:, None], col)
    # the special door's brass knob catches the light at night
    # sky and aerial haze (a painted backdrop look: haze tints toward the horizon colour)
    skc = sky_color(dx, dy, dz, tod, t)
    hz_ = 1 - np.exp(-tt / (260 if tod < 1.5 else 120))
    hazec = sky_color(dx, dy, np.zeros_like(dz) + 0.02, tod, t)
    col = col * (1 - hz_[:, None]) + hazec * hz_[:, None]
    col = np.where(sky[:, None], skc, col)
    img = col.reshape(IH, IW, 3).astype(np.float32)
    depth = np.where(sky, 1e4, tt).reshape(IH, IW).astype(np.float32)
    # gentle contour darkening where depth jumps (clay edges catch shadow)
    dl = np.log(depth + 1)
    gx = np.abs(cv2.Sobel(dl, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(dl, cv2.CV_32F, 0, 1, ksize=3))
    edge = np.clip((gx + gy) * 1.2 - 0.15, 0, 1)
    img = img * (1 - 0.45 * edge[..., None])
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    return img, depth


# ---------------------------------------------------------------------------
# painted backdrop pieces
# ---------------------------------------------------------------------------
def clouds(img, depth, cam, t_eff, tod):
    """cotton-wool clouds hung on wires in front of the sky"""
    rng = np.random.default_rng(94)
    out = img
    a = np.clip(1 - tod * 0.7, 0.15, 1)
    for k in range(9):
        p = np.array([rng.uniform(-260, 260), rng.uniform(-120, 320), rng.uniform(70, 120)])
        p[0] += 0.4 * math.sin(t_eff * 0.3 + k)
        q = cam.project(p[None])
        if q[2][0] < 1:
            continue
        x0, y0 = q[0][0], q[1][0]
        sc_ = cam.f / q[2][0]
        g = np.zeros((H // 4, W // 4), np.float32)
        for j in range(7):
            ox = rng.uniform(-14, 14) * sc_ / 4
            oy = rng.uniform(-3, 3) * sc_ / 4
            r = rng.uniform(5, 9) * sc_ / 4
            cv2.circle(g, (int(x0 / 4 + ox), int(y0 / 4 + oy)), int(max(1, r)), 1.0, -1, cv2.LINE_AA)
        g = cv2.GaussianBlur(g, (0, 0), 1.5)
        g = cv2.resize(g, (W, H))
        g = g * (depth > 5000)
        # the wire it hangs from
        top = cam.project(np.array([[p[0], p[1], p[2] + 200]]))
        cw = np.zeros((H, W), np.float32)
        cv2.line(cw, (int(x0), int(y0)), (int(top[0][0]), int(top[1][0])), 0.5, 1, cv2.LINE_AA)
        cw *= (depth > 5000)
        lit = np.array([1.0, 0.98, 0.95]) * (1 - min(1, tod)) + np.array([1.0, 0.7, 0.6]) * min(1, tod) * (tod < 1.5) + np.array([0.12, 0.14, 0.24]) * (tod >= 1.5)
        shade = np.clip((np.arange(H)[:, None] - y0) / (8 * sc_ + 1), -1, 1)
        cc = (lit ** 2.2)[None, None] * (0.95 - 0.25 * shade[..., None])
        out = out * (1 - g[..., None] * a) + cc * g[..., None] * a
        out = out * (1 - cw[..., None] * 0.6)
    return out


def castle(img, depth, cam, t_eff, tod):
    """the floating castle on its transparent mountain, far over the hills"""
    anchor = np.array([-150.0, 560.0, 75.0])
    q = cam.project(anchor[None])
    if q[2][0] < 1:
        return img
    x0, y0 = q[0][0], q[1][0]
    sc_ = cam.f / q[2][0]
    lay = np.zeros((H, W, 3), np.float32)
    m = np.zeros((H, W), np.float32)
    # the transparent mountain: only its edges and glints show
    mt = np.zeros((H, W), np.float32)
    tri = np.array([[x0 - 70 * sc_, y0 + 140 * sc_], [x0 + 70 * sc_, y0 + 140 * sc_], [x0, y0 + 8 * sc_]])
    cv2.polylines(mt, [tri.astype(np.int32)], True, 1.0, max(1, int(1.2 * sc_)), cv2.LINE_AA)
    cv2.fillPoly(mt, [tri.astype(np.int32)], 0.12, cv2.LINE_AA)
    # towers
    rng = np.random.default_rng(7)
    cols = [(236, 190, 214), (214, 200, 240), (250, 236, 200), (190, 214, 240)]
    for k in range(9):
        tw = rng.uniform(6, 12) * sc_
        th = rng.uniform(18, 46) * sc_
        tx = x0 + rng.uniform(-36, 36) * sc_
        ty = y0 + 10 * sc_
        c = np.array(s2l(cols[k % len(cols)]))
        cv2.rectangle(lay, (int(tx - tw / 2), int(ty - th)), (int(tx + tw / 2), int(ty)), tuple(float(v) for v in c), -1, cv2.LINE_AA)
        cv2.rectangle(m, (int(tx - tw / 2), int(ty - th)), (int(tx + tw / 2), int(ty)), 1.0, -1, cv2.LINE_AA)
        roof = np.array([[tx - tw * 0.65, ty - th], [tx + tw * 0.65, ty - th], [tx, ty - th - tw * 1.4]]).astype(np.int32)
        rc = tuple(float(v) for v in s2l((120, 80, 170)))
        cv2.fillPoly(lay, [roof], rc, cv2.LINE_AA)
        cv2.fillPoly(m, [roof], 1.0, cv2.LINE_AA)
        # a flag that jumps between poses (stop-motion)
        fl = int(t_eff * 6) % 3
        fx, fy = tx, ty - th - tw * 1.4
        flag = np.array([[fx, fy], [fx + (6 + fl) * sc_, fy + 1.5 * sc_ * (fl - 1)], [fx, fy + 3 * sc_]]).astype(np.int32)
        cv2.fillPoly(lay, [flag], tuple(float(v) for v in s2l((230, 60, 60))), cv2.LINE_AA)
        cv2.fillPoly(m, [flag], 1.0, cv2.LINE_AA)
    lay = lay * (1.0 if tod < 1.5 else 0.25)
    sky = (depth > 5000).astype(np.float32)
    a = 0.9 * m * sky
    out = img * (1 - a[..., None]) + lay * a[..., None]
    gl = np.array([0.8, 0.9, 1.0]) * (0.25 if tod < 1.5 else 0.1)
    out = out + (mt * sky)[..., None] * gl
    return out


def moon(img, depth, cam, tod):
    if tod < 1.3:
        return img
    a = min(1.0, (tod - 1.3) / 0.5)
    p = np.array([60.0, 300.0, 130.0])
    q = cam.project(p[None])
    if q[2][0] < 1:
        return img
    x0, y0 = int(q[0][0]), int(q[1][0])
    r = int(14 * cam.f / q[2][0])
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (x0, y0), r, 1.0, -1, cv2.LINE_AA)
    cv2.circle(m, (x0 + int(r * 0.45), y0 - int(r * 0.2)), int(r * 0.9), 0.0, -1, cv2.LINE_AA)
    top = cam.project(np.array([[p[0], p[1], p[2] + 300]]))
    cv2.line(m, (x0 - int(r * 0.2), y0 - r), (int(top[0][0]), int(top[1][0])), 0.35, max(1, int(px(1.5))), cv2.LINE_AA)
    m *= (depth > 5000)
    # glitter stars
    rng = np.random.default_rng(11)
    st = np.zeros((H, W), np.float32)
    for _ in range(140):
        sx, sy = int(rng.uniform(0, W)), int(rng.uniform(0, H * 0.6))
        cv2.circle(st, (sx, sy), max(1, int(px(1.5))), rng.uniform(0.3, 1.0), -1)
    st *= (depth > 5000)
    return img + (m * a)[..., None] * np.array([0.9, 0.85, 0.6], np.float32) + (st * a * 0.5)[..., None] * np.array([0.9, 0.9, 1.0], np.float32)


# ---------------------------------------------------------------------------
# the Animations: flat rubber-hose cartoons standing in a clay world
# ---------------------------------------------------------------------------
def animation(img, depth, cam, pos, height, t_eff, seed, facing=0.0, walk=0.0, run=0.0, grin=1.0, look=1.0):
    """draw one boiling black-and-white cartoon figure at world position pos (feet), `height` metres tall"""
    q = cam.project(np.array([pos, pos + np.array([0, 0, height])]))
    if (q[2] < 0.5).any():
        return img
    fx, fy = q[0][0], q[1][0]
    hpx = fy - q[1][1]
    if hpx < 6:
        return img
    rng = np.random.default_rng(seed * 1000 + int(t_eff * 12))
    j = lambda: rng.normal(0, 0.012 * hpx)          # line boil: each held frame redraws a little differently
    u = hpx / 10.0                                    # drawing unit
    ph = t_eff * (6 + 6 * run) + seed
    bounce = abs(math.sin(ph)) * u * (0.4 + 0.6 * run) * walk
    lay = np.full((H, W, 3), 0, np.uint8)
    m = np.zeros((H, W), np.uint8)
    black, white, grey = (14, 14, 14), (238, 236, 228), (150, 148, 140)

    def P(x, y):
        return (int((fx + x * u + j()) * 4), int((fy - y * u - bounce + j()) * 4))

    def circ(x, y, r, c, fill=-1):
        cv2.circle(lay, P(x, y), int(max(1, r * u) * 4), c, fill, cv2.LINE_AA, 2)
        cv2.circle(m, P(x, y), int(max(1, r * u) * 4), 255, fill, cv2.LINE_AA, 2)

    def ell(x, y, rx, ry, c, ang=0):
        cv2.ellipse(lay, P(x, y), (int(max(1, rx * u) * 4), int(max(1, ry * u) * 4)), ang, 0, 360, c, -1, cv2.LINE_AA, 2)
        cv2.ellipse(m, P(x, y), (int(max(1, rx * u) * 4), int(max(1, ry * u) * 4)), ang, 0, 360, 255, -1, cv2.LINE_AA, 2)

    def hose(pts, c, wd):
        pp = np.array([P(x, y) for x, y in pts], np.int32)
        cv2.polylines(lay, [pp], False, c, int(max(1, wd * u)), cv2.LINE_AA, 2)
        cv2.polylines(m, [pp], False, 255, int(max(1, wd * u)), cv2.LINE_AA, 2)

    sw = math.sin(ph) * walk
    # legs: noodles with big shoes
    for side in (-1, 1):
        k = sw * side
        hip = (0.9 * side, 4.0)
        knee = (0.9 * side + 0.8 * k, 2.0)
        foot = (0.9 * side + 1.6 * k, 0.4 + max(0, -k) * 0.6)
        hose([hip, knee, foot], black, 0.7)
        ell(foot[0] + 0.7 * side * (1 if facing >= 0 else -1), foot[1], 1.1, 0.55, black)
    # body: a pear, with shorts and two buttons
    ell(0, 5.2, 1.9, 1.9, black)
    ell(0, 4.3, 2.0, 1.2, grey)
    circ(-0.6, 4.4, 0.25, white)
    circ(0.6, 4.4, 0.25, white)
    # arms: noodles, gloves
    for side in (-1, 1):
        k = -sw * side
        sh_ = (1.5 * side, 6.4)
        el = (2.6 * side, 5.6 + 0.8 * k + run * 1.2)
        hand = (3.4 * side + 0.6 * k, 4.6 + 1.4 * k + run * 2.4)
        hose([sh_, el, hand], black, 0.55)
        circ(hand[0], hand[1], 0.75, white)
        circ(hand[0], hand[1], 0.75, black, max(1, int(0.12 * u)))
    # head: black, a light face mask, pie-cut eyes, a grin far too wide; a single curl of hair
    circ(0, 8.3, 2.3, black)
    ell(0, 7.8, 1.9, 1.6, white)
    hose([(0.0, 10.5), (0.5, 11.3), (0.0, 11.8), (-0.4, 11.4)], black, 0.35)
    for side in (-1, 1):
        ell(0.75 * side, 8.6, 0.6, 1.0, white)
        cv2.ellipse(lay, P(0.75 * side, 8.6), (int(0.6 * u * 4), int(1.0 * u * 4)), 0, 0, 360, black, int(max(1, 0.12 * u)), cv2.LINE_AA, 2)
        pxo = 0.15 * side * look + 0.2 * facing
        circ(0.75 * side + pxo, 8.4, 0.38, black)
        # the pie-cut wedge
        cx_, cy_ = P(0.75 * side + pxo, 8.4)
        r4 = int(0.38 * u * 4)
        cv2.ellipse(lay, (cx_, cy_), (r4, r4), 0, -70, -40, white, -1, cv2.LINE_AA, 2)
    # grin
    gw = 1.4 + 0.5 * grin
    cv2.ellipse(lay, P(0, 7.4), (int(gw * u * 4), int(0.75 * u * 4 * grin + 4)), 0, 0, 180, black, -1, cv2.LINE_AA, 2)
    if grin > 0.5:
        cv2.ellipse(lay, P(0, 7.35), (int((gw - 0.25) * u * 4), int(0.45 * u * 4 * grin + 2)), 0, 0, 180, white, -1, cv2.LINE_AA, 2)
        for k in range(-3, 4):
            xk = k * (gw - 0.25) / 3.5
            cv2.line(lay, P(xk, 7.35), P(xk, 7.35 - 0.4 * grin), black, max(1, int(0.08 * u)), cv2.LINE_AA, 2)
    circ(0, 7.95, 0.35, black)            # nose
    # composite with a depth test at the feet
    mx = np.clip(int(fx), 0, W - 1)
    my = np.clip(int(fy) - 2, 0, H - 1)
    vis = (depth > q[2][0] - 1.0).astype(np.float32)
    a = (m.astype(np.float32) / 255) * vis
    colf = s2l(lay.astype(np.float32))
    return img * (1 - a[..., None]) + colf * a[..., None]

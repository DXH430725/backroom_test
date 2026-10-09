"""
Ep04 — LEVEL 881 "The Courtyard of Windows": the atrium and its windows.

An irregular indoor atrium (~45 m across, 30 m tall) of white concrete, its walls
covered in evenly spaced windows — some with blinds drawn, some dark, and some that
are not windows at all. On the floor: living moss, a pentagonal paved path with four
inroads, and a hollow circle of pale birchwood at the centre. Ray-cast analytically
(polygon prism) per pixel; props and people are projected and depth-tested.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, px, s2l, l2s, value_noise, to_u8

RS = 1.5
IW, IH = int(W / RS), int(H / RS)
HT = 30.0

# irregular, convex 9-sided plan
_rng = np.random.default_rng(881)
_ang = np.sort(np.linspace(0, 2 * np.pi, 9, endpoint=False) + _rng.uniform(-0.18, 0.18, 9))
_rad = 22.0 + _rng.uniform(-2.5, 2.5, 9)
POLY = np.stack([np.cos(_ang) * _rad, np.sin(_ang) * _rad], 1)
EDGES = [(POLY[i], POLY[(i + 1) % 9]) for i in range(9)]
ELEN = [float(np.linalg.norm(b - a)) for a, b in EDGES]

WIN_COL, WIN_ROW0, WIN_ROW = 3.2, 3.6, 3.7
WIN_W, WIN_H = 1.45, 2.15
ENTRY_EDGE, ENTRY_U = 4, None        # entry window position, set below


def window_hash(e, c, r, epoch):
    v = (e * 73856093) ^ (c * 19349663) ^ (r * 83492791) ^ (epoch * 2654435761)
    return ((v * 2246822519) & 0xFFFFFFFF) / 4294967296.0


N_A = value_noise(512, 6, 81)
N_B = value_noise(256, 5, 82)

CONC = s2l((214, 214, 208))
MOSS = s2l((58, 82, 40))
PAVER = s2l((150, 148, 140))
BIRCH = s2l((222, 204, 170))
FRAME = s2l((60, 60, 62))
BLIND = np.array([1.0, 0.82, 0.56], np.float32)
COLD = np.array([0.92, 0.96, 1.0], np.float32)


def pentagon_dist(x, y, R=12.5):
    """distance to the outline of a regular pentagon (point-up)"""
    ang = np.arctan2(y, x) - np.pi / 2
    seg = 2 * np.pi / 5
    a = (ang + seg / 2) % seg - seg / 2
    r = np.hypot(x, y)
    apothem = R * math.cos(math.pi / 5)
    return np.abs(r * np.cos(a) - apothem)


def floor_material(x, y):
    r = np.hypot(x, y)
    pd = pentagon_dist(x, y)
    path = pd < 1.0
    for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4):
        ux, uy = math.cos(a), math.sin(a)
        along = x * ux + y * uy
        side = np.abs(-x * uy + y * ux)
        path |= (side < 0.9) & (along > 2.8) & (along < 11.0)
    birch = r < 3.0
    n = N_A[((y * 30) % 512).astype(np.int64) % 512, ((x * 30) % 512).astype(np.int64) % 512]
    moss = MOSS * (0.6 + 0.8 * n[..., None]) * (0.8 + 0.4 * N_B[((y * 3) % 256).astype(np.int64) % 256, ((x * 3) % 256).astype(np.int64) % 256][..., None])
    gx, gy = (x / 0.55) % 1, (y / 0.55) % 1
    gap = (np.minimum(gx, 1 - gx) < 0.04) | (np.minimum(gy, 1 - gy) < 0.04)
    pav = PAVER * (0.85 + 0.3 * n[..., None]) * np.where(gap, 0.55, 1.0)[..., None]
    rings = 0.92 + 0.08 * np.sin(r * 26 + 3 * N_B[((y * 6) % 256).astype(np.int64) % 256, ((x * 6) % 256).astype(np.int64) % 256])
    plank = (np.abs((x / 0.18) % 1 - 0.5) > 0.47)
    bir = BIRCH * rings[..., None] * np.where(plank, 0.8, 1.0)[..., None]
    rim = (r > 2.88) & (r < 3.0)
    bir = np.where(rim[..., None], BIRCH * 0.45, bir)
    col = np.where(birch[..., None], bir, np.where(path[..., None], pav, moss))
    return col, birch


def _ss(e0, e1, x):
    k = np.clip((x - e0) / (e1 - e0), 0, 1)
    return k * k * (3 - 2 * k)


def silhouette(u, v, kind):
    """soft shadow of a person standing behind the blinds; u,v in [0,1] window coords"""
    if kind == 0:   # standing a little back from the glass
        head = 1 - _ss(0.7, 1.3, (u - 0.5) ** 2 / 0.012 + (v - 0.68) ** 2 / 0.02)
        w = 0.21 - 0.08 * np.clip(v - 0.2, 0, 1)
        body = (1 - _ss(-0.03, 0.05, np.abs(u - 0.5) - w)) * (1 - _ss(0.54, 0.62, v))
        return np.maximum(head, body) * 0.85
    # pressed close: huge head and shoulders filling the window
    head = 1 - _ss(0.75, 1.15, (u - 0.5) ** 2 / 0.05 + (v - 0.55) ** 2 / 0.09)
    sh = (1 - _ss(-0.04, 0.04, np.abs(u - 0.5) - 0.47)) * (1 - _ss(0.16, 0.24, v))
    return np.maximum(head, sh)


def render(cam, t, ent=0.25, epoch=0, light=1.0, entry_open=True, blinds_glow=1.0, all_ent=False, door=None):
    """-> display sRGB (H,W,3), depth (IH,IW)"""
    X = (np.arange(IW, dtype=np.float32) + 0.5) - IW / 2
    Y = IH / 2 - (np.arange(IH, dtype=np.float32) + 0.5)
    X, Y = np.meshgrid(X, Y)
    f = cam.f / RS
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    nn = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / nn, dy / nn, dz / nn
    ox, oy, oz = cam.pos
    tw = np.full(dx.shape, np.inf, np.float32)
    eid = np.zeros(dx.shape, np.int32)
    su = np.zeros(dx.shape, np.float32)
    for i, (a, b) in enumerate(EDGES):
        ex, ey = b[0] - a[0], b[1] - a[1]
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = ((a[0] - ox) * ey - (a[1] - oy) * ex) / den
            ss = ((a[0] - ox) * dy - (a[1] - oy) * dx) / den
        ok = (tt > 1e-4) & (ss >= 0) & (ss <= 1) & (tt < tw)
        if entry_open and i == ENTRY_EDGE and ENTRY_U is not None:
            zz = oz + dz * tt
            ok &= ~((np.abs(ss * ELEN[i] - ENTRY_U) < 0.68) & (zz > 0.47) & (zz < 2.53))
        tw = np.where(ok, tt, tw)
        eid = np.where(ok, i, eid)
        su = np.where(ok, ss * ELEN[i], su)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(dz < -1e-6, -oz / dz, np.inf)
        tc = np.where(dz > 1e-6, (HT - oz) / dz, np.inf)
    tmin = np.minimum(np.minimum(tw, tf), tc)
    hx, hy, hz = ox + dx * tmin, oy + dy * tmin, oz + dz * tmin
    wall = tw <= np.minimum(tf, tc)
    floor = ~wall & (tf <= tc)
    ceil = ~wall & ~floor
    # ---- walls
    u, z = su, hz
    n = N_A[((z * 22) % 512).astype(np.int64) % 512, ((u * 22) % 512).astype(np.int64) % 512]
    wcol = CONC * (0.9 + 0.12 * n[..., None])
    seam = (np.abs(((z - WIN_ROW0 + 1.85) / WIN_ROW) % 1 - 0.5) > 0.497) | (np.abs((u / 6.4) % 1 - 0.5) > 0.498)
    wcol = wcol * np.where(seam, 0.85, 1.0)[..., None]
    E = (0.26 + 0.3 * np.clip(z / HT, 0, 1) - 0.1 * np.exp(-z / 1.5)) * light
    wcol = wcol * E[..., None]
    # windows
    off = (epoch * 1.1) % WIN_COL
    cidx = np.floor((u - off) / WIN_COL).astype(np.int64)
    ridx = np.floor((z - WIN_ROW0 + WIN_ROW / 2) / WIN_ROW).astype(np.int64)
    cu = (u - off) - (cidx + 0.5) * WIN_COL
    cz = z - (WIN_ROW0 + ridx * WIN_ROW)
    inwin = wall & (np.abs(cu) < WIN_W / 2) & (np.abs(cz) < WIN_H / 2) & (ridx >= 0) & (ridx < 7)
    inwin &= (u > 1.0) & (u < np.array(ELEN, np.float32)[eid] - 1.0)
    fr = inwin & ((np.abs(cu) > WIN_W / 2 - 0.07) | (np.abs(cz) > WIN_H / 2 - 0.07))
    glass = inwin & ~fr
    wu = (cu + WIN_W / 2) / WIN_W
    wv = (cz + WIN_H / 2) / WIN_H
    # per-pixel hash, computed via unique windows for speed
    key = (eid.astype(np.int64) * 1000 + cidx) * 16 + ridx
    keys, inv = np.unique(np.where(glass, key, -1), return_inverse=True)
    hv = np.array([window_hash(int(k // 16 // 1000), int((k // 16) % 1000), int(k % 16), epoch) if k >= 0 else 0.0
                   for k in keys], np.float32)[inv.reshape(key.shape)]
    hv2 = np.array([window_hash(int(k // 16 // 1000), int((k // 16) % 1000), int(k % 16), 99) if k >= 0 else 0.0
                    for k in keys], np.float32)[inv.reshape(key.shape)]
    is_ent = (hv < ent) | all_ent
    dark = ~is_ent & (hv2 < 0.25)
    slat = 0.75 + 0.25 * np.sin(wv * WIN_H / 0.055 * 2 * np.pi)
    lit = BLIND * (0.55 + 0.45 * slat[..., None]) * blinds_glow * (0.9 + 0.7 * hv2[..., None]) * (0.6 + 0.4 * light)
    sil0 = silhouette(wu, wv, 0)
    sil1 = silhouette(wu, wv, 1)
    sil = np.where(hv2 < 0.3, sil1, sil0)
    entcol = lit * (1 - 0.92 * sil[..., None])
    darkcol = np.array([0.02, 0.025, 0.035], np.float32) + 0.04 * n[..., None]
    gcol = np.where(is_ent[..., None], entcol, np.where(dark[..., None], darkcol, lit))
    wcol = np.where(fr[..., None], FRAME * E[..., None], wcol)
    wcol = np.where(glass[..., None], gcol, wcol)
    # the entry window, ground floor
    if ENTRY_U is not None:
        ent_w = wall & (eid == ENTRY_EDGE) & (np.abs(u - ENTRY_U) < 0.75) & (z > 0.4) & (z < 2.6)
        if entry_open:
            ent_fr = ent_w & ((np.abs(u - ENTRY_U) > 0.68) | (z < 0.47) | (z > 2.53))
            wcol = np.where(ent_w[..., None], np.array([0.08, 0.08, 0.07], np.float32), wcol)
            wcol = np.where(ent_fr[..., None], FRAME * E[..., None], wcol)
    if door is not None:
        de, du = door
        dmask = wall & (eid == de) & (np.abs(u - du) < 0.62) & (z < 2.3)
        dfr = dmask & ((np.abs(u - du) > 0.55) | (z > 2.22))
        grain = 0.8 + 0.25 * N_B[np.floor(z * 40).astype(np.int64) % 256, np.floor(u * 4).astype(np.int64) % 256]
        wcol = np.where(dmask[..., None], np.array([0.2, 0.07, 0.035], np.float32) * grain[..., None] * light, wcol)
        wcol = np.where(dfr[..., None], np.array([0.1, 0.035, 0.02], np.float32) * light, wcol)
    # ---- floor
    fcol, birch = floor_material(hx, hy)
    rr = np.hypot(hx, hy)
    Ef = (0.36 + 0.14 * np.exp(-rr / 10)) * light
    # soft darkening near walls
    fcol = fcol * Ef[..., None]
    # ---- ceiling: a pale grid of skylights
    gxc, gyc = (hx / 4.0) % 1, (hy / 4.0) % 1
    sky = (np.abs(gxc - 0.5) < 0.42) & (np.abs(gyc - 0.5) < 0.42)
    ccol = np.where(sky[..., None], COLD * 0.75 * light, CONC * 0.25 * light)
    col = np.where(wall[..., None], wcol, np.where(floor[..., None], fcol, ccol))
    # camera still outside the atrium (climbing in through the entry window): everything on
    # this side of the wall is the dark room we are leaving
    if ENTRY_U is not None and not inside(ox, oy):
        a, b = EDGES[ENTRY_EDGE]
        ex, ey = b[0] - a[0], b[1] - a[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = ((a[0] - ox) * ey - (a[1] - oy) * ex) / (dx * ey - dy * ex)
        outside = (tmin <= tp + 1e-3) | (tp < 0)
        col = np.where(outside[..., None], np.array([0.012, 0.012, 0.014], np.float32), col)
    # cold haze
    fog = np.exp(-tmin / 70.0)[..., None]
    col = col * fog + COLD * 0.18 * light * (1 - fog)
    img = l2s(np.clip(col, 0, 1))
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR), tmin


def inside(x, y):
    for a, b in EDGES:
        if (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 0:
            return False
    return True


def setup_entry():
    global ENTRY_U
    ENTRY_U = ELEN[ENTRY_EDGE] * 0.5 + 0.0
    a, b = EDGES[ENTRY_EDGE]
    p = a + (b - a) * (ENTRY_U / ELEN[ENTRY_EDGE])
    nrm = np.array([-(b - a)[1], (b - a)[0]]) / ELEN[ENTRY_EDGE]
    if np.dot(nrm, -p) < 0:
        nrm = -nrm
    return p, nrm     # point on the wall, inward normal


ENTRY_P, ENTRY_N = setup_entry()


def door_point(edge=7, frac=0.4):
    a, b = EDGES[edge]
    p = a + (b - a) * frac
    nrm = np.array([-(b - a)[1], (b - a)[0]]) / ELEN[edge]
    if np.dot(nrm, -p) < 0:
        nrm = -nrm
    return p, nrm, ELEN[edge] * frac


# ---------------------------------------------------------------------------
# props and people (projected polygons, depth-tested against the atrium)
# ---------------------------------------------------------------------------
def _poly(out, depth_full, cam, pts3, col, zbias=0.0):
    qx, qy, qd = cam.project(np.asarray(pts3, np.float64))
    if (qd < 0.1).any():
        return
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.stack([qx * 8, qy * 8], 1).astype(np.int32)], 255, cv2.LINE_AA, 3)
    a = m.astype(np.float32) / 255 * (depth_full > qd.mean() - zbias)
    out[:] = out * (1 - a[..., None]) + np.array(col, np.float32) * a[..., None]


def _shade(c, k):
    return tuple(min(1.0, v * k) for v in c)


def box(out, depth, cam, cx, cy, z0, sx, sy, sz, yaw, col, light=1.0):
    ca, sa = math.cos(yaw), math.sin(yaw)

    def P(x, y, z):
        return (cx + x * ca - y * sa, cy + x * sa + y * ca, z)
    hx, hy = sx / 2, sy / 2
    v = [P(-hx, -hy, z0), P(hx, -hy, z0), P(hx, hy, z0), P(-hx, hy, z0),
         P(-hx, -hy, z0 + sz), P(hx, -hy, z0 + sz), P(hx, hy, z0 + sz), P(-hx, hy, z0 + sz)]
    faces = [((4, 5, 6, 7), 1.0), ((0, 1, 5, 4), 0.62), ((1, 2, 6, 5), 0.75), ((2, 3, 7, 6), 0.55), ((3, 0, 4, 7), 0.7)]
    cp = cam.pos
    fl = []
    for idx, sh in faces:
        c = np.mean([v[i] for i in idx], 0)
        fl.append((np.linalg.norm(c - cp), idx, sh))
    for _, idx, sh in sorted(fl, key=lambda q: -q[0]):
        _poly(out, depth, cam, [v[i] for i in idx], _shade(col, sh * light))


def gramophone(out, depth, cam, cx, cy, z0, yaw, light=1.0, t=0.0):
    box(out, depth, cam, cx, cy, z0, 0.42, 0.42, 0.18, yaw, (0.22, 0.1, 0.05), light)
    # turntable disc
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    _poly(out, depth, cam, [(cx + 0.17 * math.cos(a), cy + 0.17 * math.sin(a), z0 + 0.19) for a in th], _shade((0.05, 0.05, 0.05), light))
    # the horn: a cone opening sideways and up
    ca, sa = math.cos(yaw), math.sin(yaw)
    base = np.array([cx - 0.12 * ca, cy - 0.12 * sa, z0 + 0.2])
    axis = np.array([ca * 0.5, sa * 0.5, 0.38])
    mouth = base + axis
    ax = axis / np.linalg.norm(axis)
    up = np.cross(ax, [0, 0, 1.0])
    up /= np.linalg.norm(up)
    v2 = np.cross(ax, up)
    ring = [mouth + 0.27 * (math.cos(a) * up + math.sin(a) * v2) for a in th]
    for k in range(len(th)):
        _poly(out, depth, cam, [base, ring[k], ring[(k + 1) % len(th)]], _shade((0.62, 0.45, 0.18), light * (0.6 + 0.4 * math.cos(th[k]))))
    _poly(out, depth, cam, ring, _shade((0.12, 0.08, 0.03), light))


def table(out, depth, cam, cx, cy, yaw, light=1.0, book=True, gram=True, t=0.0):
    ca, sa = math.cos(yaw), math.sin(yaw)
    for lx, ly in ((-0.7, -0.38), (0.7, -0.38), (0.7, 0.38), (-0.7, 0.38)):
        box(out, depth, cam, cx + lx * ca - ly * sa, cy + lx * sa + ly * ca, 0, 0.07, 0.07, 0.72, yaw, (0.25, 0.11, 0.06), light)
    box(out, depth, cam, cx, cy, 0.72, 1.6, 0.9, 0.05, yaw, (0.32, 0.14, 0.07), light)
    if gram:
        gramophone(out, depth, cam, cx + 0.35 * ca, cy + 0.35 * sa, 0.77, yaw + 0.6, light, t)
    if book:
        box(out, depth, cam, cx - 0.4 * ca + 0.1 * sa, cy - 0.4 * sa - 0.1 * ca, 0.77, 0.32, 0.22, 0.04, yaw + 0.2, (0.15, 0.25, 0.32), light)


def person(out, depth, cam, x, y, yaw, light=1.0, lamp=True, color=(0.12, 0.13, 0.12), kneel=False, dress=False):
    """an expedition member (helmet + headlamp) or, with dress=True, a pink-dressed faceling"""
    ca, sa = math.cos(yaw), math.sin(yaw)
    h = 0.6 if kneel else 1.0

    def P(fx, lz, z):
        return (x + fx * ca - lz * sa, y + fx * sa + lz * ca, z)
    pts = np.array([P(0, -0.11, 0), P(0, -0.12, 0.88 * h), P(0, 0.11, 0), P(0, 0.12, 0.88 * h),
                    P(0, 0, 0.9 * h), P(0, 0, 1.5 * h), P(0.02, 0, 1.72 * h),
                    P(0, -0.24, 1.48 * h), P(0, 0.24, 1.48 * h), P(0.1, -0.3, 0.92 * h), P(0.1, 0.3, 0.92 * h),
                    P(0.13, 0, 1.76 * h)])
    qx, qy, qd = cam.project(pts)
    if (qd < 0.3).any():
        return
    s = cam.f / qd[4]
    m = np.zeros((H, W), np.uint8)
    L = cv2.LINE_AA
    p = lambda k: (int(qx[k] * 8), int(qy[k] * 8))
    th = lambda r: max(1, int(r * s))
    if dress:
        tri = np.array([p(5), (int((qx[0] - 0.42 * s) * 8), int(qy[0] * 8)), (int((qx[2] + 0.42 * s) * 8), int(qy[2] * 8))], np.int32)
        cv2.fillPoly(m, [tri], 255, L, 3)
        cv2.line(m, p(4), p(5), 255, th(0.32), L, 3)
    else:
        cv2.line(m, p(0), p(1), 255, th(0.15), L, 3)
        cv2.line(m, p(2), p(3), 255, th(0.15), L, 3)
        cv2.line(m, p(4), p(5), 255, th(0.48), L, 3)
    cv2.line(m, p(7), p(9), 255, th(0.11), L, 3)
    cv2.line(m, p(8), p(10), 255, th(0.11), L, 3)
    cv2.circle(m, p(6), int(0.13 * s * 8), 255, -1, L, 3)
    a = m.astype(np.float32) / 255 * (depth > qd[4] - 0.4)
    c = np.array([1.0, 0.55, 0.65], np.float32) * 0.75 * light if dress else np.array(color, np.float32) * light
    out[:] = out * (1 - a[..., None]) + c * a[..., None]
    if dress:   # pale, featureless head
        hm = np.zeros((H, W), np.uint8)
        cv2.circle(hm, p(6), int(0.12 * s * 8), 255, -1, L, 3)
        ha = hm.astype(np.float32) / 255 * (depth > qd[4] - 0.4)
        out[:] = out * (1 - ha[..., None]) + np.array([0.86, 0.8, 0.76], np.float32) * light * ha[..., None]
    facing = (ca * (cam.pos[0] - x) + sa * (cam.pos[1] - y)) > 0
    if lamp and facing and qd[11] > 0.3:
        r = max(2, int(0.05 * s))
        cx_, cy_ = int(qx[11]), int(qy[11])
        if 0 <= cx_ < W and 0 <= cy_ < H and depth[cy_, cx_] > qd[11] - 0.4:
            g = np.zeros((H, W), np.float32)
            cv2.circle(g, (cx_, cy_), r, 1.0, -1, L)
            g = cv2.GaussianBlur(g, (0, 0), r * 2.5) * 6 + g
            out[:] = out + np.clip(g, 0, 1.5)[..., None] * np.array([1.0, 0.95, 0.8], np.float32) * 0.6


def lamp_on_floor(out, depth, cam, x, y, light=1.0):
    qx, qy, qd = cam.project(np.array([[x, y, 0.06]]))
    if qd[0] < 0.3:
        return
    cx_, cy_ = int(qx[0]), int(qy[0])
    if not (0 <= cx_ < W and 0 <= cy_ < H):
        return
    s = cam.f / qd[0]
    g = np.zeros((H, W), np.float32)
    cv2.ellipse(g, (cx_, cy_), (max(2, int(0.07 * s)), max(1, int(0.03 * s))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    g = cv2.GaussianBlur(g, (0, 0), max(2, 0.12 * s)) * 4 + g
    out[:] = out + np.clip(g, 0, 1.5)[..., None] * np.array([1.0, 0.95, 0.8], np.float32) * 0.5


# ---------------------------------------------------------------------------
# the white corridor behind the mahogany door
# ---------------------------------------------------------------------------
def corridor(cam, t, faceling_y=None, light=1.0):
    X = (np.arange(IW, dtype=np.float32) + 0.5) - IW / 2
    Y = IH / 2 - (np.arange(IH, dtype=np.float32) + 0.5)
    X, Y = np.meshgrid(X, Y)
    f = cam.f / RS
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    nn = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / nn, dy / nn, dz / nn
    ox, oy, oz = cam.pos
    big = 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        tR = np.where(dx > 1e-6, (1.1 - ox) / dx, big)
        tL = np.where(dx < -1e-6, (-1.1 - ox) / dx, big)
        tF = np.where(dz < -1e-6, -oz / dz, big)
        tC = np.where(dz > 1e-6, (2.6 - oz) / dz, big)
        tE = np.where(dy > 1e-6, (34.0 - oy) / dy, big)
    tt = np.minimum.reduce([tR, tL, tF, tC, tE])
    hy = oy + dy * tt
    hz = oz + dz * tt
    hx = ox + dx * tt
    wall = (tt == tR) | (tt == tL)
    ceil = tt == tC
    floor = tt == tF
    lamp = ceil & (np.abs(hx) < 0.25) & ((hy % 3.0) < 1.6)
    base = np.where(floor, 0.62, np.where(ceil, 0.8, 0.88))
    # doors along the walls (mahogany, closed)
    door = wall & ((hy % 6.0) < 1.0) & (hz < 2.1)
    E = 0.85 + 0.25 * np.cos((hy % 3.0 - 0.8) / 3.0 * 2 * np.pi)
    col = (base * E)[..., None] * np.array([1.0, 1.0, 0.98], np.float32)
    col = np.where(door[..., None], np.array([0.24, 0.09, 0.05], np.float32) * E[..., None], col)
    col = np.where(lamp[..., None], np.array([1.6, 1.6, 1.55], np.float32), col)
    col = col * light * np.exp(-tt / 60.0)[..., None]
    img = l2s(np.clip(col, 0, 1))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(tt.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    if faceling_y is not None:
        person(img, depth, cam, 0.0, faceling_y, -math.pi / 2, light, lamp=False, dress=True)
    return img

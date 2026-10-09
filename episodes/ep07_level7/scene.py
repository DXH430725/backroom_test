"""
Ep07 — LEVEL 7 "Thalassophobia": the sideways room, the ocean under a concrete sky,
and the long way down.

- room(): the entrance room, built sideways into the ceiling. Through its doorway the
  ocean stands up like a wall, because the room is rotated 90 degrees to the world.
- surface(): an endless cold sea under a suspended concrete ceiling; barren islands.
- underwater(): depth-graded fog and colour, drifting marine snow, the diver's lamp cone,
  skeletons and debris (sprites), and The Thing.
- abyss(): heightfield ray-march of the tar-and-rock mountains, with the cave.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, px, s2l, l2s, value_noise, Cam

RS = 1.5
IW, IH = int(W / RS), int(H / RS)
CEIL = 60.0

N_A = value_noise(512, 6, 701)
N_B = value_noise(256, 4, 702)
N_H = value_noise(512, 7, 703)
RGY, RGX = np.gradient(cv2.GaussianBlur(value_noise(256, 4, 704), (0, 0), 2))


def _tex(T, x, y, sc):
    n = T.shape[0]
    return T[np.floor(y * sc).astype(np.int64) % n, np.floor(x * sc).astype(np.int64) % n]


def _texb(T, x, y, sc):
    """bilinear, wrapping lookup (smooth heightfields need it)"""
    n = T.shape[0]
    fx, fy = x * sc, y * sc
    x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    ax, ay = fx - x0, fy - y0
    x0m, y0m, x1m, y1m = x0 % n, y0 % n, (x0 + 1) % n, (y0 + 1) % n
    return (T[y0m, x0m] * (1 - ax) * (1 - ay) + T[y0m, x1m] * ax * (1 - ay) + T[y1m, x0m] * (1 - ax) * ay + T[y1m, x1m] * ax * ay)


def _rays(cam, step=RS):
    iw, ih = int(W / step), int(H / step)
    X = (np.arange(iw, dtype=np.float32) + 0.5) - iw / 2
    Y = ih / 2 - (np.arange(ih, dtype=np.float32) + 0.5)
    X, Y = np.meshgrid(X, Y)
    f = cam.f / step
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    return dx / n, dy / n, dz / n


def _up(img):
    return cv2.resize(l2s(np.clip(img, 0, 1)), (W, H), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# the surface world: sea at z=0, concrete ceiling at z=CEIL
# ---------------------------------------------------------------------------
SEA = s2l((54, 66, 70))
CONC = s2l((120, 118, 112))
ISLANDS = [(-900.0, 1400.0, 120.0, 25.0), (1500.0, 2600.0, 200.0, 40.0), (-2600.0, 3400.0, 260.0, 30.0), (600.0, -1800.0, 150.0, 20.0)]


def ceiling_col(x, y, light=1.0):
    seam = ((np.abs((x / 12.0) % 1 - 0.5) > 0.49) | (np.abs((y / 12.0) % 1 - 0.5) > 0.49))
    n = _tex(N_A, x, y, 0.4)
    return CONC * (0.5 + 0.35 * n[..., None]) * np.where(seam, 0.6, 1.0)[..., None] * light


def sea_world(ox, oy, oz, dx, dy, dz, t, light=1.0):
    """shade rays in the surface world (no rotation): sea, ceiling, haze, islands"""
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(dz < -1e-5, -oz / dz, np.inf)
        tc = np.where(dz > 1e-5, (CEIL - oz) / dz, np.inf)
    hx, hy = ox + dx * np.where(np.isfinite(ts), ts, 0), oy + dy * np.where(np.isfinite(ts), ts, 0)
    gx = _tex(RGX, hx * 1.5 + t * 0.8, hy * 1.5, 1.0) * 8 + _tex(RGX, hx * 0.4 - t * 0.3, hy * 0.4, 1.0) * 5
    gy = _tex(RGY, hx * 1.5 + t * 0.8, hy * 1.5, 1.0) * 8 + _tex(RGY, hx * 0.4 - t * 0.3, hy * 0.4, 1.0) * 5
    rz = np.abs(dz) + 0.25 * gy * 0.1 + 0.02
    rx = dx + gx * 0.08
    ry = dy + gy * 0.08
    with np.errstate(divide="ignore", invalid="ignore"):
        trc = (CEIL - 0) / np.maximum(rz, 1e-3)
    refl = ceiling_col(hx + rx * trc, hy + ry * trc, light)
    fresnel = 0.04 + 0.96 * (1 - np.clip(-dz, 0, 1)) ** 5
    water = SEA * light * 0.35 * (1 - fresnel[..., None]) + refl * fresnel[..., None] * 0.8
    lit = np.exp(-((hx - ox) ** 2 + (hy - oy) ** 2) / (2 * max(oz, 1.0) ** 2 * 1.5)) * (0.4 + 3.0 * (gx ** 2 + gy ** 2)) * np.exp(-oz / 30)
    water = water + lit[..., None] * np.array([0.12, 0.12, 0.1], np.float32)
    cx, cy = ox + dx * np.where(np.isfinite(tc), tc, 0), oy + dy * np.where(np.isfinite(tc), tc, 0)
    ceil = ceiling_col(cx, cy, light)
    tt = np.where(dz < 0, ts, tc)
    col = np.where((dz < 0)[..., None], water, ceil)
    # islands: low dark humps on the sea, drawn analytically as their silhouette above the waterline
    for ix_, iy_, r, h in ISLANDS:
        relx, rely = ix_ - ox, iy_ - oy
        d = math.hypot(relx, rely)
        if d < 50:
            continue
        az = math.atan2(relx, rely)
        raz = np.arctan2(dx, dy)
        da = np.abs((raz - az + math.pi) % (2 * math.pi) - math.pi)
        ang_w = r / d
        el = np.arcsin(np.clip(dz, -1, 1))
        prof = (h / d) * np.sqrt(np.clip(1 - (da / ang_w) ** 2, 0, 1)) * (0.8 + 0.2 * np.sin(da / ang_w * 7))
        on = (el > -oz / d) & (el < prof - oz / d) & (da < ang_w)
        col = np.where(on[..., None], s2l((40, 40, 38)) * light * 0.9, col)
        tt = np.where(on, d, tt)
    fog = np.exp(-tt / 1600.0)[..., None]
    haze = s2l((96, 102, 104)) * light
    return col * fog + haze * (1 - fog)


def surface(cam, t, light=1.0):
    dx, dy, dz = _rays(cam)
    ox, oy, oz = cam.pos
    col = sea_world(ox, oy, oz, dx, dy, dz, t, light)
    # the entrance room hangs from the ceiling on a concrete column; its doorway faces down,
    # 4.5 m above the water, lit from inside
    bx0, bx1, by0, by1, bz0, bz1 = -2.0, 2.0, -1.3, 1.3, 4.5, CEIL
    with np.errstate(divide="ignore", invalid="ignore"):
        tx0, tx1 = (bx0 - ox) / dx, (bx1 - ox) / dx
        ty0, ty1 = (by0 - oy) / dy, (by1 - oy) / dy
        tz0, tz1 = (bz0 - oz) / dz, (bz1 - oz) / dz
    tmin = np.maximum.reduce([np.minimum(tx0, tx1), np.minimum(ty0, ty1), np.minimum(tz0, tz1)])
    tmax = np.minimum.reduce([np.maximum(tx0, tx1), np.maximum(ty0, ty1), np.maximum(tz0, tz1)])
    hitb = (tmax >= tmin) & (tmin > 0)
    hz = oz + dz * tmin
    hxb, hyb = ox + dx * tmin, oy + dy * tmin
    bottom = hitb & (np.abs(hz - bz0) < 1e-3)
    side_x = hitb & (np.abs(np.abs(hxb) - 2.0) < 1e-3)
    seam = (np.abs((hz / 3.0) % 1 - 0.5) > 0.47)
    colb = CONC * light * np.where(side_x, 0.55, 0.42)[..., None] * np.where(seam, 0.7, 1.0)[..., None]
    door = bottom & (np.abs(hxb) < 0.55) & (np.abs(hyb) < 1.05)
    colb = np.where(door[..., None], np.array([1.2, 1.12, 0.9], np.float32), colb)
    fogb = np.exp(-tmin / 1600.0)[..., None]
    colb = colb * fogb + s2l((96, 102, 104)) * light * (1 - fogb)
    col = np.where(hitb[..., None], colb, col)
    return _up(col)


# ---------------------------------------------------------------------------
# the entrance room (room space: floor z=0, 4 x 5 x 2.6 m; the doorway is in the +y wall)
# ---------------------------------------------------------------------------
def room(cam, t, light=1.0):
    dx, dy, dz = _rays(cam)
    ox, oy, oz = cam.pos
    w, l, h = 2.0, 2.5, 2.6
    big = 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 0, (w - ox) / dx, np.where(dx < 0, (-w - ox) / dx, big))
        ty = np.where(dy > 0, (l - oy) / dy, np.where(dy < 0, (-l - oy) / dy, big))
        tf = np.where(dz < 0, -oz / dz, big)
        tc = np.where(dz > 0, (h - oz) / dz, big)
    tt = np.minimum.reduce([tx, ty, tf, tc])
    hx, hy, hz = ox + dx * tt, oy + dy * tt, oz + dz * tt
    wallx = tt == tx
    wally = tt == ty
    floor = tt == tf
    ceil = tt == tc
    along = np.where(wallx, hy, hx)
    paper = s2l((150, 140, 110)) * (0.8 + 0.3 * _tex(N_A, along, hz, 3)[..., None])
    carpet = s2l((90, 80, 66)) * (0.75 + 0.4 * _tex(N_A, hx, hy, 9)[..., None])
    puddle = floor & (_tex(N_B, hx, hy, 0.6) > 0.55)
    ceilc = s2l((180, 176, 160)) * 0.8
    col = np.where(floor[..., None], carpet, np.where(ceil[..., None], ceilc, paper))
    # fluorescent lamp
    lamp = ceil & (np.abs(hx) < 0.3) & (np.abs(hy) < 0.7)
    E = 0.25 + 1.1 / (1 + (hx ** 2 + hy ** 2 + (h - hz) ** 2) / 3)
    col = col * E[..., None] * light
    col = np.where(lamp[..., None], np.array([1.3, 1.3, 1.2], np.float32), col)
    # the puddle reflects the lamp
    col = np.where(puddle[..., None], col * 0.5 + 0.1 * np.exp(-(hx ** 2 + hy ** 2) / 1.5)[..., None], col)
    # the doorway on the +y wall: through it, the sea — turned on its side
    door = wally & (dy > 0) & (np.abs(hx) < 0.55) & (hz < 2.1)
    if door.any():
        j = np.nonzero(door)
        # room -> world: the room is rotated so that its +y axis points down into the sea
        wx, wy, wz = dx[j], dz[j], -dy[j]
        sea = sea_world(0.0, -2.0, 4.5, wx, wy, wz, t, 1.0)
        col[j] = sea
    frame = wally & (dy > 0) & (np.abs(hx) < 0.62) & (hz < 2.18) & ~door
    col = np.where(frame[..., None], s2l((70, 60, 50)) * light, col)
    return _up(col), tt


def room_props(img, cam, light=1.0):
    """bookcase, coffee table, chair: flat-shaded boxes"""
    def boxq(x0, y0, z0, x1, y1, z1, colr):
        V = np.array([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
        faces = [((4, 5, 6, 7), 1.0), ((0, 1, 5, 4), 0.6), ((1, 2, 6, 5), 0.75), ((2, 3, 7, 6), 0.55), ((3, 0, 4, 7), 0.7)]
        cp = np.array(cam.pos)
        fl = sorted(faces, key=lambda f: -np.linalg.norm(V[list(f[0])].mean(0) - cp))
        for idx, sh in fl:
            qx, qy, qd = cam.project(V[list(idx)])
            if (qd < 0.1).any():
                continue
            c = tuple(float(v) for v in l2s(s2l(colr) * sh * light))
            cv2.fillPoly(img, [np.stack([qx * 8, qy * 8], 1).astype(np.int32)], c, cv2.LINE_AA, 3)
    boxq(-1.95, -1.2, 0, -1.6, 0.6, 2.0, (90, 60, 40))           # bookcase
    for k in range(4):                                          # shelves of books
        boxq(-1.6, -1.15, 0.15 + k * 0.48, -1.58, 0.55, 0.45 + k * 0.48, (120 + 20 * (k % 2), 50, 40))
    boxq(0.3, -0.6, 0, 1.1, 0.0, 0.42, (100, 70, 45))            # coffee table
    boxq(1.3, -1.6, 0, 1.75, -1.15, 0.45, (60, 60, 70))          # chair seat
    boxq(1.3, -1.62, 0.45, 1.75, -1.55, 1.0, (60, 60, 70))       # chair back


# ---------------------------------------------------------------------------
# underwater
# ---------------------------------------------------------------------------
def zone_color(depth):
    """ambient water colour by depth (metres)"""
    k = np.clip(depth / 1000.0, 0, 1)
    k2 = np.clip((depth - 1000) / 3000.0, 0, 1)
    c = s2l((60, 96, 108)) * (1 - k) + s2l((10, 26, 44)) * k
    c = c * (1 - k2) + np.array([0.0005, 0.0012, 0.003], np.float32) * k2
    return c


def underwater(cam, t, depth, lamp=0.0, surface_above=True):
    """background water: ambient light from above at shallow depths, lamp cone, marine snow"""
    dx, dy, dz = _rays(cam)
    amb = zone_color(depth)
    up = np.clip(dz, -1, 1)
    sky_light = np.exp(-depth / 220.0)
    col = amb * (0.6 + 0.6 * up[..., None]) + np.array([0.3, 0.5, 0.55], np.float32) * sky_light * np.clip(up, 0, 1)[..., None] ** 3 * 0.8
    # the surface seen from below while shallow: shimmering
    if depth < 30 and surface_above:
        sh = _tex(N_B, dx / np.maximum(dz, 0.05) * 20 + t * 2, dy / np.maximum(dz, 0.05) * 20, 1) * (dz > 0.3)
        col = col + sh[..., None] * np.array([0.4, 0.55, 0.55], np.float32) * np.exp(-depth / 8)
    if lamp > 0:
        cosang = dx * cam.F[0] + dy * cam.F[1] + dz * cam.F[2]
        beam = np.clip((cosang - 0.86) / 0.14, 0, 1) ** 1.5
        col = col + beam[..., None] * np.array([0.05, 0.07, 0.07], np.float32) * lamp
    return col


def snow(img, cam, t, depth, lamp=0.0, density=1.0, seed=0):
    """marine snow: particles in a box around the camera, lit by ambient + lamp"""
    rng = np.random.default_rng(seed)
    n = int(1400 * density)
    P = rng.uniform(-6, 6, (n, 3))
    drift = np.array([0.0, 0.0, -0.08]) * t + np.array([0.05, 0.02, 0]) * t
    rel = (P - (np.array(cam.pos) % 12) + drift + 6) % 12 - 6
    qx, qy, qd = cam.project(np.array(cam.pos) + rel)
    ok = (qd > 0.2) & (qx > 0) & (qx < W) & (qy > 0) & (qy < H)
    out = img
    cosang = (rel @ cam.F) / (np.linalg.norm(rel, axis=1) + 1e-6)
    amb = max(0.0, 1 - depth / 1500.0) * 0.25
    lit = amb + lamp * np.clip((cosang - 0.86) / 0.14, 0, 1) * 1.2 / (1 + qd ** 2 / 9)
    for k in np.nonzero(ok & (lit > 0.02))[0]:
        r = max(1, int(px(2.2) / qd[k]))
        c = float(min(1.0, lit[k]))
        cv2.circle(out, (int(qx[k]), int(qy[k])), r, (c * 0.8, c * 0.9, c * 0.85), -1, cv2.LINE_AA)
    return out


def bones(img, cam, objs, lamp, depth):
    """draw skeletons/debris lit only where the lamp (or the dim ambient) reaches them"""
    amb = max(0.0, 1 - depth / 1500.0) * 0.35
    for o in sorted(objs, key=lambda o: -np.linalg.norm(np.array(o["p"]) - cam.pos)):
        p = np.array(o["p"], float)
        rel = p - cam.pos
        dist = np.linalg.norm(rel)
        if dist > 40:
            continue
        cosang = rel @ cam.F / (dist + 1e-6)
        light = amb * math.exp(-dist / 12) + lamp * max(0.0, (cosang - 0.84) / 0.16) ** 1.2 * 1.6 / (1 + dist ** 2 / 30)
        if light < 0.02:
            continue
        c = tuple(float(min(1, light * v)) for v in o.get("col", (0.85, 0.82, 0.74)))
        s = o.get("s", 1.0)
        yaw = o.get("yaw", 0.0)
        ca, sa = math.cos(yaw), math.sin(yaw)

        def P(a, b, z):
            return p + np.array([a * ca - b * sa, a * sa + b * ca, z]) * s

        segs = []
        if o["k"] == "human":      # humanoid skeleton: oversized jaw, flipper legs
            segs = [(P(0, 0, 0.9), P(0, 0, 0.2), 0.04)]                                  # spine
            for k in range(6):
                z = 0.85 - k * 0.09
                segs += [(P(0, 0, z), P(0.12, -0.18, z - 0.05), 0.02), (P(0, 0, z), P(0.12, 0.18, z - 0.05), 0.02)]
            segs += [(P(0, -0.2, 0.85), P(0.1, -0.45, 0.45), 0.025), (P(0, 0.2, 0.85), P(0.1, 0.45, 0.45), 0.025)]
            segs += [(P(0, -0.1, 0.2), P(-0.1, -0.25, -0.5), 0.05), (P(0, 0.1, 0.2), P(-0.1, 0.25, -0.5), 0.05)]
            segs += [(P(-0.1, -0.25, -0.5), P(-0.05, -0.45, -0.65), 0.09), (P(-0.1, 0.25, -0.5), P(-0.05, 0.45, -0.65), 0.09)]
            skull, jaw = P(0.03, 0, 1.05), P(0.18, 0, 0.95)
            segs += [(skull, skull, 0.12), (P(0.05, -0.09, 0.98), jaw, 0.05), (P(0.05, 0.09, 0.98), jaw, 0.05)]
        elif o["k"] == "fish":     # a big fish skeleton
            L = 2.2
            segs = [(P(-L, 0, 0), P(L, 0, 0), 0.05), (P(L, 0, 0), P(L, 0, 0), 0.25)]
            for k in range(14):
                a = -L + 0.3 + k * (2 * L - 0.8) / 14
                hgt = 0.5 * math.sin(math.pi * (k + 1) / 15)
                segs += [(P(a, 0, 0), P(a - 0.15, 0, hgt), 0.02), (P(a, 0, 0), P(a - 0.15, 0, -hgt), 0.02)]
            segs += [(P(-L, 0, 0), P(-L - 0.5, 0, 0.4), 0.03), (P(-L, 0, 0), P(-L - 0.5, 0, -0.4), 0.03)]
        elif o["k"] == "metal":
            segs = [(P(-0.8, -0.3, 0), P(0.9, 0.2, 0.3), 0.18), (P(0.2, -0.6, -0.2), P(0.4, 0.7, 0.4), 0.1)]
        elif o["k"] == "weird":    # "incomprehensible": too many limbs, wrong angles
            rng = np.random.default_rng(int(abs(p[0] * 31 + p[2])) % 9999)
            core = P(0, 0, 0)
            segs = [(core, core, 0.3)]
            for k in range(11):
                a = P(*rng.uniform(-1.6, 1.6, 3))
                segs += [(core, a, 0.04), (a, a + rng.uniform(-0.6, 0.6, 3) * s, 0.025)]
        for a, b, r in segs:
            q = cam.project(np.array([a, b]))
            if (q[2] < 0.2).any():
                continue
            th = max(1, int(r * 2 * cam.f / q[2].mean()))
            if np.allclose(a, b):
                cv2.circle(img, (int(q[0][0] * 8), int(q[1][0] * 8)), max(2, int(r * cam.f / q[2][0] * 8)), c, -1, cv2.LINE_AA, 3)
            else:
                cv2.line(img, (int(q[0][0] * 8), int(q[1][0] * 8)), (int(q[0][1] * 8), int(q[1][1] * 8)), c, th, cv2.LINE_AA, 3)
    return img


def the_thing(img, cam, p, yaw, size=14.0, lamp=0.0, t=0.0):
    """a vast dark body crossing the edge of the light; its skin only catches the beam at the rim"""
    ca, sa = math.cos(yaw), math.sin(yaw)
    pts = []
    for k in range(40):
        a = 2 * math.pi * k / 40
        bx = math.cos(a) * size
        bz = math.sin(a) * size * 0.28 * (1 - 0.6 * max(0, math.cos(a)) ** 4) + math.sin(a * 3 + t) * 0.4
        pts.append(np.array(p) + np.array([bx * ca, bx * sa, bz]))
    # a dorsal fin and a long pectoral fin
    fin = [np.array(p) + np.array([(0.1 * size) * ca, (0.1 * size) * sa, size * 0.55]), np.array(p) + np.array([(-0.2 * size) * ca, (-0.2 * size) * sa, size * 0.25]),
           np.array(p) + np.array([(0.3 * size) * ca, (0.3 * size) * sa, size * 0.25])]
    q = cam.project(np.array(pts))
    if (q[2] < 0.5).any():
        return img
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.stack([q[0] * 8, q[1] * 8], 1).astype(np.int32)], 255, cv2.LINE_AA, 3)
    qf = cam.project(np.array(fin))
    if (qf[2] > 0.5).all():
        cv2.fillPoly(m, [np.stack([qf[0] * 8, qf[1] * 8], 1).astype(np.int32)], 255, cv2.LINE_AA, 3)
    a = m.astype(np.float32) / 255
    rim = cv2.GaussianBlur(a, (0, 0), px(6)) - cv2.GaussianBlur(a, (0, 0), px(2))
    rim = np.clip(-rim * 3, 0, 1) * lamp
    img = img * (1 - a[..., None] * 0.97)
    img = img + rim[..., None] * np.array([0.12, 0.15, 0.14], np.float32)
    return img


# ---------------------------------------------------------------------------
# the abyss: heightfield of tar and rock, ray-marched; the cave mouth
# ---------------------------------------------------------------------------
CAVE = np.array([0.0, 160.0, 0.0])          # local abyss coordinates; z is set onto the mountainside below


def abyss_height(x, y):
    h = _texb(N_H, x, y, 2.5) * 30 + _texb(N_H, y, x, 12.0) * 9 + _texb(N_A, x, y, 40.0) * 1.6
    peak = 120 * np.exp(-((x - CAVE[0]) ** 2 + (y - CAVE[1] - 30) ** 2) / (2 * 70 ** 2))
    return h + peak - 100


CAVE[2] = float(abyss_height(np.array([CAVE[0]]), np.array([CAVE[1]]))[0]) - 1.0


def abyss(cam, t, lamp=1.0, cave_glow=0.0):
    dx, dy, dz = _rays(cam, RS * 1.25)
    ox, oy, oz = cam.pos
    tt = np.full(dx.shape, 0.5)
    hit = np.zeros(dx.shape, bool)
    for _ in range(90):
        px_, py_, pz_ = ox + dx * tt, oy + dy * tt, oz + dz * tt
        hgt = abyss_height(px_, py_)
        d = pz_ - hgt
        newhit = (d < 0.3) & ~hit
        hit |= newhit
        tt = np.where(hit, tt, tt + np.maximum(d * 0.45, 0.25))
        if (tt > 90).all():
            break
    tt = np.minimum(tt, 90)
    px_, py_, pz_ = ox + dx * tt, oy + dy * tt, oz + dz * tt
    e = 0.6
    nx = abyss_height(px_ - e, py_) - abyss_height(px_ + e, py_)
    ny = abyss_height(px_, py_ - e) - abyss_height(px_, py_ + e)
    nz = np.full_like(nx, 2 * e)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    tar = _texb(N_B, px_, py_, 6.0) > 0.58
    alb = np.where(tar[..., None], np.array([0.02, 0.018, 0.016], np.float32), s2l((96, 88, 80)) * (0.45 + 0.8 * _texb(N_A, px_ + pz_ * 0.7, py_, 25)[..., None]))
    # lamp: a spot from the camera
    lx, ly, lz = -dx, -dy, -dz
    lam = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    cosang = dx * cam.F[0] + dy * cam.F[1] + dz * cam.F[2]
    beam = np.clip((cosang - 0.8) / 0.2, 0, 1) ** 1.3
    E = lamp * beam * (0.06 + lam ** 1.5) * 3.4 / (1 + tt ** 2 / 140)
    col = alb * E[..., None]
    # the cave: a black mouth in the mountainside, faintly outlined
    rc = np.sqrt((px_ - CAVE[0]) ** 2 + (pz_ - CAVE[2]) ** 2 * 1.6 + (py_ - CAVE[1]) ** 2 * 0.3)
    mouth = hit & (rc < 7.5)
    rim = hit & (rc >= 7.5) & (rc < 9.5)
    col = np.where(rim[..., None], col * 0.45, col)
    col = np.where(mouth[..., None], np.array([0.0, 0.0, 0.0], np.float32) + cave_glow * 0.02, col)
    col = np.where(hit[..., None], col, 0.0)
    fog = np.exp(-tt / 38)[..., None]
    col = col * fog + np.array([0.0004, 0.001, 0.0025], np.float32) * (1 - fog)
    img = l2s(np.clip(col, 0, 1))
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)


def bubbles(img, cam, t, src, n=60, seed=7):
    rng = np.random.default_rng(seed)
    for k in range(n):
        ph = (t * rng.uniform(0.3, 0.7) + rng.random()) % 1.0
        p = np.array(src) + np.array([rng.uniform(-3, 3) + 0.4 * math.sin(t * 2 + k), rng.uniform(-3, 3), ph * 30])
        q = cam.project(p[None])
        if q[2][0] < 0.3:
            continue
        r = max(1, int(px(9) / q[2][0]))
        c = 0.25 / (1 + q[2][0] / 8)
        cv2.circle(img, (int(q[0][0]), int(q[1][0])), r, (c, c * 1.1, c * 1.1), max(1, r // 4), cv2.LINE_AA)
    return img

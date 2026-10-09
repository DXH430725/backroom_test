"""
Ep05 — LEVEL 10 "Bumper Crop": endless wheat under a leaden sky.

Analytic outdoor renderer: an infinite tiling of field plots (wheat, barley, a rare
canola patch) separated by hedgerows and dirt roads, with lakes. The wheat canopy is
a plane at 0.95 m whose surface carries travelling wind waves; field edges show the
side of the crop; lakes mirror the overcast. Trees, the barn, people and the dog are
projected sprites/polygons, depth-tested. Near the camera, individual stalks are drawn.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, px, s2l, l2s, value_noise

RS = 1.5
IW, IH = int(W / RS), int(H / RS)
CANOPY = 0.95

PX, PY = 140.0, 90.0          # plot size
HEDGE = 4.0                   # hedgerow strip
ROADW = 4.5                   # road strip (on the x-boundaries of every 3rd plot column)

N_A = value_noise(512, 6, 101)
N_B = value_noise(512, 5, 102)
N_C = value_noise(256, 4, 103)
N_W = value_noise(256, 3, 104)

WHEAT = s2l((196, 168, 96))
BARLEY = s2l((178, 170, 112))
CANOLA = s2l((226, 204, 40))
SOIL = s2l((96, 78, 56))
GRASS = s2l((92, 104, 58))
ROAD = s2l((122, 104, 80))
SKY_TOP = s2l((150, 156, 160))
SKY_HOR = s2l((196, 198, 196))
TREE = s2l((46, 58, 34))

LAKES = [(-60.0, 230.0, 70.0, 40.0), (300.0, -120.0, 120.0, 55.0), (40.0, 620.0, 160.0, 70.0)]
CANOLA_PLOT = (2, 4)


def _tex(T, x, y, scale):
    n = T.shape[0]
    return T[np.floor(y * scale).astype(np.int64) % n, np.floor(x * scale).astype(np.int64) % n]


def regions(x, y):
    """-> (field mask, kind 0 wheat / 1 barley / 2 canola, hedge, road, lake, shore)"""
    i = np.floor(x / PX).astype(np.int64)
    j = np.floor(y / PY).astype(np.int64)
    lx = x - i * PX
    ly = y - j * PY
    hedge = (lx < HEDGE) | (ly < HEDGE)
    road = (i % 3 == 0) & (lx >= HEDGE) & (lx < HEDGE + ROADW)
    lake = np.zeros(x.shape, bool)
    shore = np.zeros(x.shape, bool)
    for cx, cy, rx, ry in LAKES:
        d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
        lake |= d < 1
        shore |= (d >= 1) & (d < 1.25)
    field = ~hedge & ~road & ~lake & ~shore
    h = ((i * 73856093) ^ (j * 19349663)) & 0xFFFF
    kind = np.where(h % 5 == 0, 1, 0)
    kind = np.where((i == CANOLA_PLOT[0]) & (j == CANOLA_PLOT[1]), 2, kind)
    return field, kind, hedge, road, lake, shore, lx, ly


def wind(x, y, t, gust=1.0):
    """travelling bright/dark bands across the crop"""
    a = _tex(N_W, x + 9.0 * t, y + 2.0 * t, 0.06)
    b = _tex(N_W, x * 0.7 - 5.0 * t, y * 0.7 + 7.5 * t, 0.11)
    return ((a - 0.5) * 0.9 + (b - 0.5) * 0.5) * gust


def sky(dx, dy, dz, t, sun=None):
    e = np.clip(dz, 0, 1)
    col = SKY_HOR * (1 - e[..., None] ** 0.6) + SKY_TOP * e[..., None] ** 0.6
    # big soft cloud masses, drifting
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 800.0 / np.maximum(dz, 0.02)
    cx, cy = dx * k + 6 * t, dy * k + 2 * t
    cl = _tex(N_A, cx, cy, 0.0016) * 0.6 + _tex(N_B, cx, cy, 0.004) * 0.4
    col = col * (0.82 + 0.3 * cl[..., None])
    if sun:  # the picnic: a blue summer sky
        col = s2l((120, 170, 230)) * (1 - e[..., None] ** 0.5) * 0.7 + s2l((70, 130, 220)) * e[..., None] ** 0.5
        col = col * (0.9 + 0.12 * cl[..., None]) + (cl > 0.62)[..., None] * 0.35 * cl[..., None]
    return col


def render(cam, t, gust=1.0, light=1.0, fogd=380.0, rain=0.0, night=0.0, summer=False):
    """display sRGB (H,W,3), depth (H,W)"""
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
    down = dz < -1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(down, oz / -dz, np.inf)
        tc = np.where(down & (oz > CANOPY), (oz - CANOPY) / -dz, np.inf)
    # canopy hit?
    cxp, cyp = ox + dx * np.where(np.isfinite(tc), tc, 0), oy + dy * np.where(np.isfinite(tc), tc, 0)
    fC, kC, *_ = regions(cxp, cyp)
    gxp, gyp = ox + dx * np.where(np.isfinite(tg), tg, 0), oy + dy * np.where(np.isfinite(tg), tg, 0)
    fG, kG, hedge, road, lake, shore, lx, ly = regions(gxp, gyp)
    hitC = np.isfinite(tc) & fC & (not summer)
    # shrubbery along the hedgerows: a 1.5 m tall band
    HB = 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        th = np.where(down & (oz > HB), (oz - HB) / -dz, np.inf)
    hxp, hyp = ox + dx * np.where(np.isfinite(th), th, 0), oy + dy * np.where(np.isfinite(th), th, 0)
    hedgeTop = np.isfinite(th) & regions(hxp, hyp)[2] & (th < np.where(hitC, tc, np.inf))
    side = ~hitC & down & fG                         # the wall of crop at a field's edge
    hedgeSide = ~hitC & ~hedgeTop & down & hedge & ~side
    t_hit = np.where(hedgeTop, th, np.where(hitC, tc, np.where(down, tg, 1e5)))
    # in-field camera (child's eye below the canopy): stalks everywhere close by
    inside_field = oz < CANOPY and bool(regions(np.array([ox]), np.array([oy]))[0][0])
    hx = np.where(hitC, cxp, gxp)
    hy = np.where(hitC, cyp, gyp)
    kind = np.where(hitC, kC, kG)
    base = np.where((kind == 2)[..., None], CANOLA, np.where((kind == 1)[..., None], BARLEY, WHEAT))
    tex_near = _tex(N_A, hx, hy, 9.0)
    fw = t_hit / f
    mixf = np.clip(fw * 8, 0, 1)[..., None]
    grain = (0.78 + 0.44 * tex_near[..., None]) * (1 - mixf) + mixf
    wv = wind(hx, hy, t, gust)
    crop = base * grain * (1 + 0.55 * wv[..., None])
    sidecol = base * 0.45 * (0.7 + 0.5 * _tex(N_B, hx + hy, 0.95 + dz * np.minimum(t_hit, 1e4), 6.0)[..., None])
    rd = road
    track = rd & (np.abs((lx - HEDGE) - ROADW * 0.28) < 0.35) | rd & (np.abs((lx - HEDGE) - ROADW * 0.72) < 0.35)
    rcol = np.where(track[..., None], ROAD * 0.8, GRASS * 0.85 * (0.8 + 0.4 * _tex(N_C, hx, hy, 2.0)[..., None]))
    rcol = np.where((rd & ~track)[..., None], rcol, np.where(track[..., None], ROAD * (0.8 + 0.3 * _tex(N_A, hx, hy, 4)[..., None]), rcol))
    gcol = np.where(hedge[..., None], GRASS * 0.6, GRASS * (0.75 + 0.4 * _tex(N_C, hx, hy, 1.5)[..., None]) * (0.8 + 0.4 * _tex(N_A, hx, hy, 7)[..., None] * (1 - mixf[..., 0][..., None] * 0.8)))
    gcol = np.where(shore[..., None], SOIL * (0.8 + 0.4 * _tex(N_A, hx, hy, 3)[..., None]), gcol)
    gcol = np.where(rd[..., None], rcol, gcol)
    # lakes: mirror of the sky, broken by small ripples
    rip = (_tex(N_W, hx * 3 + t * 2, hy * 3, 1.0) - 0.5) * 0.08
    skyr = sky(dx + rip, dy, -dz, t, sun=summer)
    fres = 0.25 + 0.75 * (1 - np.clip(-dz, 0, 1)) ** 4
    lcol = s2l((40, 50, 48)) * (1 - fres[..., None]) + skyr * fres[..., None]
    gcol = np.where(lake[..., None], lcol, gcol)
    if summer:
        side = side & False
        gcol = np.where((fG & ~lake)[..., None], s2l((86, 140, 60)) * (0.75 + 0.45 * _tex(N_C, hx, hy, 1.5)[..., None]) * (0.85 + 0.3 * _tex(N_A, hx, hy, 6)[..., None]), gcol)
    col = np.where(hitC[..., None], crop, np.where(side[..., None], sidecol, gcol))
    shrub = TREE * 1.1 * (0.6 + 0.7 * _tex(N_A, hxp + hyp, hyp - hxp, 3.0)[..., None])
    shrub_s = TREE * 0.8 * (0.5 + 0.7 * _tex(N_A, gxp + gyp, 0.7 + dz * np.where(np.isfinite(tg), tg, 0), 2.5)[..., None])
    col = np.where(hedgeTop[..., None], shrub, np.where(hedgeSide[..., None], shrub_s, col))
    if inside_field:
        near = t_hit < 6.0
        col = np.where(near[..., None], WHEAT * 0.5 * (0.7 + 0.6 * tex_near[..., None]), col)
    col = col * light
    # sky + fog
    skyc = sky(dx, dy, dz, t, sun=summer)
    fogc = SKY_HOR * (0.95 if not summer else 0.0) + (s2l((170, 200, 230)) if summer else 0)
    fogc = fogc * light
    fog = np.exp(-t_hit / fogd)[..., None]
    col = col * fog + fogc * (1 - fog)
    col = np.where(down[..., None], col, skyc * light)
    if night > 0:
        col = col * (1 - night * 0.93) + np.array([0.004, 0.006, 0.012], np.float32) * night
    img = l2s(np.clip(col, 0, 1))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(np.where(down, t_hit, 1e5).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return img, depth


# ---------------------------------------------------------------------------
# near wheat: individual stalks around the camera
# ---------------------------------------------------------------------------
_STALKS = None


def _stalk_field():
    rng = np.random.default_rng(10)
    n = 26000
    return rng.uniform(0, 1, (n, 2)), rng.uniform(0.85, 1.12, n), rng.uniform(0, 6.28, n), rng.uniform(0.7, 1.15, n)


def stalks(img, depth, cam, t, gust=1.0, radius=9.0, light=1.0, night=0.0):
    """draw the stalks of wheat nearest to the camera (only inside fields)"""
    global _STALKS
    if _STALKS is None:
        _STALKS = _stalk_field()
    P, hgt, ph, shade = _STALKS
    ox, oy, oz = cam.pos
    # tile the 2D stalk pattern (density ~ 75 / m^2 in a 18x18 m tile) around the camera
    tile = 2 * radius
    sx = (P[:, 0] * tile + np.floor((ox - radius) / tile) * tile)
    sy = (P[:, 1] * tile + np.floor((oy - radius) / tile) * tile)
    xs = np.concatenate([sx + a * tile for a in (0, 1) for b in (0, 1)])
    ys = np.concatenate([sy + b * tile for a in (0, 1) for b in (0, 1)])
    hh = np.tile(hgt, 4)
    pp = np.tile(ph, 4)
    sh = np.tile(shade, 4)
    d2 = (xs - ox) ** 2 + (ys - oy) ** 2
    m = d2 < radius ** 2
    xs, ys, hh, pp, sh, d2 = xs[m], ys[m], hh[m], pp[m], sh[m], d2[m]
    fm = regions(xs, ys)[0]
    xs, ys, hh, pp, sh, d2 = xs[fm], ys[fm], hh[fm], pp[fm], sh[fm], d2[fm]
    if xs.size == 0:
        return img
    sway = 0.12 * gust * (np.sin(t * 2.1 + pp + xs * 0.3) + 0.6 * wind(xs, ys, t, gust) * 3)
    top = np.stack([xs + sway, ys + 0.3 * sway, hh * CANOPY], 1)
    bot = np.stack([xs, ys, np.zeros_like(xs)], 1)
    qx0, qy0, qd0 = cam.project(bot)
    qx1, qy1, qd1 = cam.project(top)
    ok = (qd0 > 0.15) & (qd1 > 0.15) & (qx1 > -50) & (qx1 < W + 50)
    order = np.argsort(-d2[ok])
    out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    base = l2s(WHEAT * light * (1 - 0.9 * night)) * 255
    idx = np.nonzero(ok)[0][order]
    for k in idx:
        dd = math.sqrt(d2[k])
        fogk = math.exp(-dd / 380.0)
        c = tuple(float(v) * sh[k] * (0.55 + 0.45 * min(1, dd / radius)) for v in base)
        th = max(1, int(px(2.2) / max(qd1[k], 0.3)))
        p0 = (int(qx0[k] * 8), int(qy0[k] * 8))
        p1 = (int(qx1[k] * 8), int(qy1[k] * 8))
        cv2.line(out, p0, p1, c, th, cv2.LINE_AA, 3)
        head = max(2, int(px(9) / max(qd1[k], 0.3)))
        dxh = (qx1[k] - qx0[k])
        dyh = (qy1[k] - qy0[k])
        L = math.hypot(dxh, dyh) + 1e-6
        p2 = (int((qx1[k] + dxh / L * head) * 8), int((qy1[k] + dyh / L * head) * 8))
        cv2.line(out, p1, p2, tuple(v * 1.12 for v in c), th + max(1, head // 5), cv2.LINE_AA, 3)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# hedgerow trees
# ---------------------------------------------------------------------------
def _trees():
    rng = np.random.default_rng(1010)
    pts = []
    for i in range(-8, 9):
        for j in range(-10, 14):
            x0, y0 = i * PX, j * PY
            for s in np.arange(0, PX, 7.0):      # along the y-boundary (bottom edge)
                if rng.random() < 0.8:
                    pts.append((x0 + s + rng.uniform(-2, 2), y0 + HEDGE / 2 + rng.uniform(-1, 1), rng.uniform(5, 10), rng.uniform(2.2, 3.8)))
            for s in np.arange(0, PY, 7.0):
                if rng.random() < 0.8:
                    pts.append((x0 + HEDGE / 2 + rng.uniform(-1, 1), y0 + s + rng.uniform(-2, 2), rng.uniform(5, 10), rng.uniform(2.2, 3.8)))
    return np.array(pts)


TREES = _trees()


def draw_trees(img, depth, cam, light=1.0, night=0.0, fogd=380.0, summer=False):
    ox, oy, oz = cam.pos
    rel = TREES[:, :2] - np.array([ox, oy])
    d = rel[:, 0] * cam.F[0] + rel[:, 1] * cam.F[1] + (TREES[:, 2] * 0.6 - oz) * cam.F[2]
    dist = np.hypot(rel[:, 0], rel[:, 1])
    m = (d > 1.0) & (dist < 900)
    T = TREES[m]
    dist = dist[m]
    centers = np.stack([T[:, 0], T[:, 1], T[:, 2] * 0.62], 1)
    qx, qy, qd = cam.project(centers)
    r = T[:, 3] * cam.f / np.maximum(qd, 0.1)
    vis = (qx > -r) & (qx < W + r) & (qy > -r * 3) & (qy < H + r)
    T, dist, qx, qy, qd, r = T[vis], dist[vis], qx[vis], qy[vis], qd[vis], r[vis]
    order = np.argsort(-dist)
    out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    leaf = TREE * light * (1 - 0.9 * night)
    if summer:
        leaf = s2l((60, 100, 40)) * light
    fogc = SKY_HOR * light * (1 - 0.9 * night)
    for k in order:
        fg = math.exp(-dist[k] / fogd)
        tree_depth = qd[k]
        cxp, cyp = int(qx[k]), int(qy[k])
        if 0 <= cxp < W and 0 <= cyp < H and depth[cyp, cxp] < tree_depth - 4:
            continue
        rr = r[k]
        base = leaf * fg + fogc * (1 - fg)
        c0 = tuple(float(v) for v in l2s(base * 0.7) * 255)
        c1 = tuple(float(v) for v in l2s(base * 1.15) * 255)
        # trunk
        tb = cam.project(np.array([[T[k, 0], T[k, 1], 0.0]]))
        if tb[2][0] > 0.1:
            cv2.line(out, (int(qx[k] * 8), int(qy[k] * 8)), (int(tb[0][0] * 8), int(tb[1][0] * 8)),
                     tuple(float(v) for v in l2s(s2l((50, 40, 30)) * light * fg + fogc * (1 - fg)) * 255), max(1, int(rr * 0.12)), cv2.LINE_AA, 3)
        # crown: an irregular blob, darker underneath, a lighter lobe where the sky lights it
        if rr < 2.5:
            cv2.circle(out, (int(qx[k] * 8), int(qy[k] * 8)), max(4, int(rr * 8)), c0, -1, cv2.LINE_AA, 3)
            continue
        hsh = int(abs(T[k, 0] * 131 + T[k, 1] * 71)) % 997
        rg = np.random.default_rng(hsh)
        lobes = [(rg.uniform(-0.45, 0.45), rg.uniform(-0.35, 0.25), rg.uniform(0.42, 0.62)) for _ in range(7)]
        for lx_, ly_, lr in sorted(lobes, key=lambda q: -q[1]):
            shade = 0.85 + 0.5 * (-ly_)
            cc_ = tuple(float(v) for v in l2s(base * 0.75 * shade) * 255)
            cv2.circle(out, (int((qx[k] + lx_ * rr) * 8), int((qy[k] + ly_ * rr) * 8)), int(lr * rr * 8), cc_, -1, cv2.LINE_AA, 3)
        for _ in range(3):
            lx_, ly_ = rg.uniform(-0.35, 0.15), rg.uniform(-0.45, -0.2)
            cv2.circle(out, (int((qx[k] + lx_ * rr) * 8), int((qy[k] + ly_ * rr) * 8)), int(rr * 0.3 * 8), c1, -1, cv2.LINE_AA, 3)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# the barn
# ---------------------------------------------------------------------------
BARN = np.array([18.0, 52.0])      # centre, on the grass by a road
BARN_W, BARN_L, BARN_H, BARN_R = 9.0, 14.0, 4.5, 3.0


def _fill(out, depth, cam, pts, col, alpha=1.0):
    qx, qy, qd = cam.project(np.asarray(pts, np.float64))
    if (qd < 0.1).any():
        return
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.stack([qx * 8, qy * 8], 1).astype(np.int32)], 255, cv2.LINE_AA, 3)
    a = m.astype(np.float32) / 255 * (depth > qd.min() - 0.5) * alpha
    out[:] = out * (1 - a[..., None]) + np.array(col, np.float32) * a[..., None]
    depth[a > 0.5] = np.minimum(depth[a > 0.5], qd.mean())


def draw_barn(img, depth, cam, light=1.0, fogd=380.0, night=0.0, door_open=False, lit=False):
    bx, by = BARN
    w, l, h, r = BARN_W / 2, BARN_L / 2, BARN_H, BARN_R
    dist = math.hypot(bx - cam.pos[0], by - cam.pos[1])
    fg = math.exp(-dist / fogd)
    fogc = SKY_HOR * light

    def c(rgb, k=1.0):
        v = s2l(rgb) * k * light * (1 - 0.9 * night)
        return tuple(float(x) for x in l2s(v * fg + fogc * (1 - fg) * (1 - 0.9 * night)))
    A = (bx - w, by - l, 0)
    B = (bx + w, by - l, 0)
    Cc = (bx + w, by + l, 0)
    D = (bx - w, by + l, 0)
    up = lambda p: (p[0], p[1], h)
    ridge0, ridge1 = (bx, by - l, h + r), (bx, by + l, h + r)
    faces = [
        ([A, B, up(B), ridge0, up(A)], c((120, 42, 30), 0.95)),            # front gable (south, faces -y)
        ([D, Cc, up(Cc), ridge1, up(D)], c((120, 42, 30), 0.7)),
        ([A, D, up(D), up(A)], c((120, 42, 30), 0.8)),
        ([B, Cc, up(Cc), up(B)], c((120, 42, 30), 0.65)),
        ([up(A), up(D), ridge1, ridge0], c((60, 60, 62), 0.9)),
        ([up(B), up(Cc), ridge1, ridge0], c((60, 60, 62), 0.7)),
    ]
    cp = np.array(cam.pos)
    faces.sort(key=lambda f: -np.linalg.norm(np.mean(np.array(f[0]), 0) - cp))
    for pts, col in faces:
        _fill(img, depth, cam, pts, col)
    # big doors on the south gable
    dw = 1.6
    door = [(bx - dw, by - l - 0.02, 0), (bx + dw, by - l - 0.02, 0), (bx + dw, by - l - 0.02, 3.2), (bx - dw, by - l - 0.02, 3.2)]
    if door_open:
        dc = (0.9, 0.7, 0.4) if lit else (0.02, 0.02, 0.02)
        _fill(img, depth, cam, door, tuple(v * (1 - 0.5 * night) if not lit else v for v in dc))
    else:
        _fill(img, depth, cam, door, c((96, 34, 24), 0.85))
        _fill(img, depth, cam, [(bx - 0.04, by - l - 0.03, 0), (bx + 0.04, by - l - 0.03, 0), (bx + 0.04, by - l - 0.03, 3.2), (bx - 0.04, by - l - 0.03, 3.2)], c((40, 20, 15)))


# ---------------------------------------------------------------------------
# people & the dog (soft silhouettes)
# ---------------------------------------------------------------------------
def figure(img, depth, cam, x, y, height=1.75, col=(0.2, 0.18, 0.16), light=1.0, kind="adult", phase=0.0, fogd=380.0):
    ox, oy, oz = cam.pos
    dist = math.hypot(x - ox, y - oy)
    fg = math.exp(-dist / fogd)
    colv = s2l(np.array(col, np.float32) * 255) * light * fg + SKY_HOR * light * (1 - fg)
    if kind == "dog":
        pts = np.array([[x, y, 0.0], [x, y, 0.45], [x + 0.35, y, 0.48], [x - 0.35, y, 0.45], [x + 0.5, y, 0.6], [x - 0.5, y, 0.6]])
        qx, qy, qd = cam.project(pts)
        if (qd < 0.2).any():
            return img
        s = cam.f / qd[0]
        m = np.zeros((H, W), np.uint8)
        L = cv2.LINE_AA
        p = lambda k: (int(qx[k] * 8), int(qy[k] * 8))
        cv2.line(m, p(2), p(3), 255, max(1, int(0.26 * s)), L, 3)                    # body
        cv2.circle(m, p(4), int(0.15 * s * 8), 255, -1, L, 3)                         # head
        for lx in (-0.25, 0.25):
            a = cam.project(np.array([[x + lx, y, 0.42], [x + lx + 0.05 * math.sin(phase * 7 + lx * 9), y, 0.0]]))
            cv2.line(m, (int(a[0][0] * 8), int(a[1][0] * 8)), (int(a[0][1] * 8), int(a[1][1] * 8)), 255, max(1, int(0.08 * s)), L, 3)
        tail = cam.project(np.array([[x - 0.35, y, 0.45], [x - 0.6, y, 0.62 + 0.08 * math.sin(phase * 11)]]))
        cv2.line(m, (int(tail[0][0] * 8), int(tail[1][0] * 8)), (int(tail[0][1] * 8), int(tail[1][1] * 8)), 255, max(1, int(0.05 * s)), L, 3)
        dref = qd[0]
    else:
        h = height
        pts = np.array([[x - 0.1, y, 0], [x - 0.1, y, 0.5 * h], [x + 0.1, y, 0], [x + 0.1, y, 0.5 * h],
                        [x, y, 0.5 * h], [x, y, 0.82 * h], [x, y, 0.93 * h],
                        [x - 0.2 * h / 1.75, y, 0.8 * h], [x + 0.2 * h / 1.75, y, 0.8 * h],
                        [x - 0.25 * h / 1.75, y, 0.5 * h], [x + 0.25 * h / 1.75, y, 0.5 * h]])
        qx, qy, qd = cam.project(pts)
        if (qd < 0.2).any():
            return img
        s = cam.f / qd[4]
        m = np.zeros((H, W), np.uint8)
        L = cv2.LINE_AA
        p = lambda k: (int(qx[k] * 8), int(qy[k] * 8))
        th = lambda rr: max(1, int(rr * s * h / 1.75))
        cv2.line(m, p(0), p(1), 255, th(0.14), L, 3)
        cv2.line(m, p(2), p(3), 255, th(0.14), L, 3)
        cv2.line(m, p(4), p(5), 255, th(0.42), L, 3)
        cv2.line(m, p(7), p(9), 255, th(0.1), L, 3)
        cv2.line(m, p(8), p(10), 255, th(0.1), L, 3)
        cv2.circle(m, p(6), int(0.12 * s * 8 * h / 1.75), 255, -1, L, 3)
        dref = qd[4]
    a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), px(1.0)) * (depth > dref - 0.5)
    return img * (1 - a[..., None]) + l2s(colv)[None, None, :] * a[..., None]


# ---------------------------------------------------------------------------
# inside the barn
# ---------------------------------------------------------------------------
def barn_interior(cam, t, light=1.0, smoke=0.0, night=0.0):
    """a plank box 9 x 14 x 4.5 m; daylight through gaps between the boards"""
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
    w, l, h = BARN_W / 2, BARN_L / 2, BARN_H
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
    along = np.where(wallx, hy, hx)
    plank = np.abs((along / 0.22) % 1 - 0.5) > 0.47
    gap = (np.abs((along / 0.22) % 1 - 0.5) > 0.493) & (wallx | wally)
    wood = s2l((120, 88, 60)) * (0.75 + 0.4 * _tex(N_A, along * 0.3, hz * 3, 8)[..., None]) * np.where(plank, 0.6, 1.0)[..., None]
    hay = s2l((170, 140, 80)) * (0.6 + 0.6 * _tex(N_A, hx, hy, 14)[..., None])
    col = np.where(floor[..., None], hay, wood * 0.9)
    col = np.where((tt == tc)[..., None], s2l((60, 44, 30)) * (0.7 + 0.4 * _tex(N_B, hx, hy, 3)[..., None]), col)
    E = (0.35 + 0.25 * np.clip(hz / h, 0, 1)) * light * (1 - 0.92 * night)
    col = col * E[..., None]
    daylight = SKY_HOR * 1.4 * (1 - night)
    col = np.where(gap[..., None], daylight, col)
    # door glow on the south wall
    door = wally & (dy < 0) & (np.abs(hx) < 1.6) & (hz < 3.2)
    col = np.where(door[..., None], SKY_HOR * 1.1 * (1 - 0.95 * night), col)
    if smoke > 0:
        s = smoke * (1 - np.exp(-tt / 3.0))[..., None] * (0.7 + 0.5 * _tex(N_W, hx + t * 0.4, hz - t * 0.3, 0.8)[..., None])
        col = col * (1 - s) + np.array([0.09, 0.085, 0.08], np.float32) * s
    img = l2s(np.clip(col, 0, 1))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(tt.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return img, depth

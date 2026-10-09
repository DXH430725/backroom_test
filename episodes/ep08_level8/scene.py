"""
Ep08 — LEVEL 8 "Cave Systems": a ray-marched karst cave.

The cave is a signed-distance tube wound along a path (y is the main axis), with its
radius modulated by noise into crawlspaces and vast chambers. Inside it: a stream of
almond water, bioluminescent moss and mushrooms, stalagspikes growing in every
direction (projected sprites), floating rocks, rope lights along the 9th Road, and,
once, a Wrangler sliding through the rock.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, px, s2l, l2s, value_noise

RS = 2.25
IW, IH = int(W / RS), int(H / RS)

N_A = value_noise(512, 6, 801)
N_B = value_noise(256, 5, 802)
N_C = value_noise(512, 4, 803)


def _texb(T, x, y, sc):
    n = T.shape[0]
    fx, fy = x * sc, y * sc
    x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    ax, ay = fx - x0, fy - y0
    x0m, y0m, x1m, y1m = x0 % n, y0 % n, (x0 + 1) % n, (y0 + 1) % n
    return T[y0m, x0m] * (1 - ax) * (1 - ay) + T[y0m, x1m] * ax * (1 - ay) + T[y1m, x0m] * (1 - ax) * ay + T[y1m, x1m] * ax * ay


# chambers along the path: (centre y, radius, length)
CHAMBERS = [(60.0, 16.0, 30.0), (150.0, 30.0, 60.0), (330.0, 22.0, 40.0)]
POOL_Y = (0.0, 9.0)            # the pool she surfaces in


def MEADOW(y):
    """where the glowing mushrooms grow: the floors of the first two chambers"""
    return np.clip(1 - np.abs(y - 60) / 14, 0, 1) ** 0.5 + np.clip(1 - np.abs(y - 150) / 32, 0, 1) ** 0.5


def path(y):
    x = 6 * np.sin(y * 0.031) + 3 * np.sin(y * 0.083 + 1.0)
    z = 4 * np.sin(y * 0.022 + 2.0)
    return x, z


def base_radius(y):
    r = 2.6 + 0.8 * np.sin(y * 0.11)
    for cy, cr, cl in CHAMBERS:
        r = np.maximum(r, cr * np.exp(-((y - cy) / (cl / 2)) ** 4))
    return r


def tri(T, x, y, z, wx, sc, off=0.0):
    """world-space texture: walls sample the (y, z) plane, floors and ceilings the (x, y) plane"""
    return wx * _texb(T, y * sc + off, z * sc, 1.0) + (1 - wx) * _texb(T, x * sc + off, y * sc, 1.0)


def _wx(dx, dz):
    a, b = dx * dx, dz * dz
    a, b = a * a, b * b
    return a / (a + b + 1e-9)


def floor_depth(r0):
    """how far below the tube's axis the walkable floor lies: chambers mostly open upwards"""
    rt = np.minimum(r0, 3.4)
    return rt * 0.72 + 0.2 * (r0 - rt)


def floor_of(y):
    yy = np.array([y], np.float64)
    cx, cz = path(yy)
    return float(cz[0] - floor_depth(base_radius(yy))[0]), float(cx[0])


def sdf(x, y, z, fine=True):
    """positive inside the cave (distance to the wall, roughly)"""
    cx, cz = path(y)
    dx, dz = x - cx, z - cz
    d = np.sqrt(dx * dx + dz * dz)
    wx = _wx(dx, dz)
    r0 = base_radius(y)
    rough = (tri(N_A, x, y, z, wx, 21) - 0.5) * 2.2 + (tri(N_B, x, y, z, wx, 40, 50) - 0.5) * 0.7
    r = r0 * (1 + 0.16 * rough)
    if fine:
        # knobbly limestone: bumps of a few centimetres to a metre, plus flowstone ridges
        b = tri(N_C, x, y, z, wx, 100)
        rid = 1 - np.abs(2 * tri(N_B, x, y, z, wx, 60, 200) - 1)
        r = r - 0.25 * b - 0.12 * rid ** 3
    s = r - d
    # a flat-ish floor so you can walk: cut the tube at a floor level, with rubble on it
    floor = cz - floor_depth(r0)
    fl = floor
    if fine:
        bd = np.abs(x - cx - 0.4 * np.sin(y * 0.13)) - (0.32 + 0.12 * np.sin(y * 0.2))
        bank = np.clip(bd / 0.8, 0, 1)
        fl = floor + 0.04 + bank * (0.08 + 0.12 * _texb(N_C, x * 90.0 + 100, y * 90.0, 1.0) + np.clip(_texb(N_A, x * 25.0, y * 25.0, 1.0) - 0.62, 0, 1))
    return np.minimum(s, z - fl + 0.02), floor


def render(cam, t, lamp=1.0, glow=1.0, extra_light=None, steps=90, tmax=70.0):
    X = (np.arange(IW, dtype=np.float32) + 0.5) - IW / 2
    Y = IH / 2 - (np.arange(IH, dtype=np.float32) + 0.5)
    X, Y = np.meshgrid(X, Y)
    f = cam.f / RS
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    nn = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = (dx / nn).ravel(), (dy / nn).ravel(), (dz / nn).ravel()
    ox, oy, oz = cam.pos
    n = dx.size
    tt = np.zeros(n)
    idx = np.arange(n)
    hit = np.zeros(n, bool)
    for _ in range(steps):
        if idx.size == 0:
            break
        px_, py_, pz_ = ox + dx[idx] * tt[idx], oy + dy[idx] * tt[idx], oz + dz[idx] * tt[idx]
        d, _fl = sdf(px_, py_, pz_)
        done = d < 0.02
        hit[idx[done]] = True
        tt[idx] += np.where(done, 0, np.maximum(d * 0.6, 0.02))
        keep = ~done & (tt[idx] < tmax)
        idx = idx[keep]
    tt = np.minimum(tt, tmax)
    px_, py_, pz_ = ox + dx * tt, oy + dy * tt, oz + dz * tt
    e = 0.06
    d0, fl = sdf(px_, py_, pz_)
    nx = sdf(px_ + e, py_, pz_)[0] - sdf(px_ - e, py_, pz_)[0]
    ny = sdf(px_, py_ + e, pz_)[0] - sdf(px_, py_ - e, pz_)[0]
    nz = sdf(px_, py_, pz_ + e)[0] - sdf(px_, py_, pz_ - e)[0]
    nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / nl, ny / nl, nz / nl             # gradient of the inside-positive field: points into the cave
    cx, cz = path(py_)
    wn = nx * nx * nx * nx
    wn = wn / (wn + nz ** 4 + 1e-9)
    on_floor = (pz_ - fl < 0.5) & (nz > 0.55)
    band = np.abs(px_ - cx - 0.4 * np.sin(py_ * 0.13)) < 0.32 + 0.12 * np.sin(py_ * 0.2)
    stream = band & (pz_ - fl < 0.1) & (nz > 0.8)
    big = np.clip((base_radius(py_) - 4.0) / 10.0, 0, 1)
    # rock albedo: limestone mottled at three scales, banded with strata, darker and wetter low down
    m1 = tri(N_C, px_, py_, pz_, wn, 260, 31)
    m2 = tri(N_A, px_, py_, pz_, wn, 40, 77)
    strata = 0.82 + 0.18 * np.sin(pz_ * 6.0 + 3.0 * tri(N_B, px_, py_, pz_, wn, 20, 11))
    rock = s2l((128, 118, 102)) * ((0.45 + 0.75 * m1) * (0.6 + 0.6 * m2) * strata)[..., None]
    veins = np.clip(1 - np.abs(tri(N_B, px_, py_, pz_, wn, 45, 300) - 0.5) / 0.008, 0, 1)
    rock = rock * (1 - 0.35 * veins)[..., None]
    mud = s2l((84, 74, 60)) * (0.6 + 0.6 * _texb(N_C, px_ * 30.0, py_ * 30.0, 1.0))[..., None]
    rock = np.where(on_floor[..., None], mud, rock)
    wet = np.clip(1.2 - (pz_ - fl) * 0.8, 0.25, 1.0) * (0.6 + 0.4 * (m2 > 0.5))
    # ambient occlusion from the distance field: crevices go dark
    ao = np.clip(sdf(px_ + nx * 0.5, py_ + ny * 0.5, pz_ + nz * 0.5, False)[0] / 0.5, 0.12, 1.0) ** 1.3
    # lighting: the headlamp (a spot from the camera)
    lx, ly, lz = -dx, -dy, -dz
    lam = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    cosang = dx * cam.F[0] + dy * cam.F[1] + dz * cam.F[2]
    beam = np.clip((cosang - 0.84) / 0.16, 0, 1) ** 1.3 + 0.10 * np.clip((cosang - 0.45) / 0.55, 0, 1)
    fall = lamp * beam * 3.2 / (1 + tt ** 2 / 14)
    spec = lam ** 40 * wet * 0.9
    col = rock * (fall * (0.05 + lam) * ao)[..., None] + (fall * spec)[..., None] * np.array([0.9, 0.92, 1.0], np.float32)
    # the cave's own faint light: moss and mushrooms below fill big chambers with a cold glow from beneath
    amb = (0.004 + 0.06 * big) * (0.55 - 0.35 * nz) * ao * glow
    # the mushroom meadows on chamber floors light the lower walls
    amb = amb + 0.22 * big * np.exp(-np.maximum(pz_ - fl, 0) / 3.5) * (0.6 + 0.4 * nz) * ao * glow * MEADOW(py_)
    col = col + rock * amb[..., None] * np.array([0.35, 1.0, 0.9], np.float32)
    # bioluminescent moss: speckled patches on walls and ceiling, cyan-green
    patch = np.clip(tri(N_B, px_, py_, pz_, wn, 15, 400) - (0.66 - 0.06 * big), 0, 1) * 3
    speck = np.clip((tri(N_A, px_, py_, pz_, wn, 700, 9) - 0.6) * 6, 0, 1)
    moss = np.clip(patch, 0, 1) * (0.25 + 0.75 * speck) * (~on_floor) * ao
    moss = moss * (0.75 + 0.25 * np.sin(t * 0.7 + py_ * 0.3))
    col = col + np.clip(moss, 0, 1)[..., None] * np.array([0.02, 0.16, 0.13], np.float32) * glow
    col = col + rock * (np.clip(patch, 0, 1) * 0.05)[..., None] * np.array([0.2, 1.0, 0.8], np.float32) * glow
    if extra_light is not None:
        for (Lx, Ly, Lz), Lc, Lr in extra_light:
            d2 = (px_ - Lx) ** 2 + (py_ - Ly) ** 2 + (pz_ - Lz) ** 2
            ldir = np.stack([Lx - px_, Ly - py_, Lz - pz_]) / np.sqrt(d2 + 1e-9)
            lam2 = np.clip(nx * ldir[0] + ny * ldir[1] + nz * ldir[2], 0, 1)
            col = col + rock * (np.array(Lc, np.float32) * (lam2 * ao / (1 + d2 / (Lr * Lr)))[..., None])
    # almond water: black and glossy, the lamp and the moss smeared across its ripples
    rip = _texb(N_C, px_ * 140 + t * 9, py_ * 70 - t * 60, 1.0)
    glint = (fall * (0.003 + 4.0 * np.clip(rip - 0.72, 0, 1) ** 2))[..., None] * np.array([0.9, 0.95, 1.0], np.float32)
    wat = col * 0.1 + glint + (amb * 0.5 + 0.004 * glow)[..., None] * np.array([0.2, 0.9, 0.8], np.float32)
    col = np.where(stream[..., None], wat, col)
    col = np.where(hit[..., None], col, 0.0)
    fog = np.exp(-tt / 45)[..., None]
    col = col * fog
    # the beam in the damp air
    haze = lamp * 0.018 * (1 - np.exp(-tt / 6)) * np.clip((cosang - 0.8) / 0.2, 0, 1)
    col = col + haze[..., None] * np.array([0.9, 0.9, 0.85], np.float32)
    img = l2s(np.clip(col, 0, 1)).reshape(IH, IW, 3)
    depth = tt.reshape(IH, IW)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR), cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)


def surface_point(y, ang, inset=0.0):
    """a point on the cave wall at cross-section y, angle ang (marched out from the axis)"""
    cx, cz = path(np.array([y]))
    c = np.array([float(cx[0]), y, float(cz[0])])
    dvec = np.array([math.cos(ang), 0.0, math.sin(ang)])
    tt_ = 0.0
    for _ in range(40):
        p = c + dvec * tt_
        d = float(sdf(np.array([p[0]]), np.array([p[1]]), np.array([p[2]]))[0][0])
        if d < 0.03:
            break
        tt_ += max(d * 0.6, 0.02)
    return c + dvec * max(tt_ - inset, 0.0)


# ---------------------------------------------------------------------------
# sprites: stalagspikes, mushrooms, floating rocks, the 9th Road, the Wrangler
# ---------------------------------------------------------------------------
def _spikes(seed=8, n=420, y0=10, y1=420):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        y = rng.uniform(y0, y1)
        ang = rng.uniform(-math.pi, math.pi)
        base = surface_point(y, ang, -0.1)
        cx, cz = path(np.array([y]))
        inward = np.array([float(cx[0]) - base[0], 0.0, float(cz[0]) - base[2]])
        inward /= np.linalg.norm(inward) + 1e-9
        s_ = math.sin(ang)
        if s_ > 0.5:                     # stalactites hang
            dirv = np.array([0, 0, -1.0]) + rng.normal(0, 0.15, 3)
        elif s_ < -0.5:                  # stalagmites rise
            dirv = np.array([0, 0, 1.0]) + rng.normal(0, 0.15, 3)
        else:                            # and here, they also grow sideways, as if gravity were a suggestion
            dirv = inward + rng.normal(0, 0.6, 3)
        dirv /= np.linalg.norm(dirv)
        L = rng.uniform(0.3, 2.2) * (2.2 if base_radius(np.array([y]))[0] > 10 else 1.0)
        kind = "spike"
        out.append((base, dirv, L, rng.uniform(0.07, 0.14) * L ** 0.6 + 0.04, kind))
    return out


SPIKES = _spikes()


def _mushrooms(seed=9):
    rng = np.random.default_rng(seed)
    out = []
    for cy, half, n in ((60.0, 13.0, 260), (150.0, 30.0, 700)):
        k = 0
        while k < n:
            y = cy + rng.uniform(-half, half)
            yy = np.array([y])
            cx, cz = path(yy)
            r = float(base_radius(yy)[0])
            fd = float(floor_depth(np.array([r]))[0])
            fl = float(cz[0]) - fd
            hw = math.sqrt(max(r * r - fd ** 2, 1.0)) * 0.8
            x = float(cx[0]) + rng.uniform(-hw, hw)
            # clumps: reject some to make clusters
            if _texb(N_B, np.array([x * 6.0]), np.array([y * 6.0]), 1.0)[0] < 0.45 + 0.1 * rng.random():
                continue
            if abs(x - float(cx[0]) - 0.4 * math.sin(y * 0.13)) < 0.9:
                continue
            out.append((np.array([x, y, fl + 0.1]), rng.uniform(0.08, 0.45) * (2.2 if rng.random() < 0.06 else 1.0), rng.uniform(0, 6.28)))
            k += 1
    return out


MUSH = _mushrooms()


def _shade(light, k=1.0):
    return tuple(float(v) for v in l2s(np.clip(s2l((122, 112, 98)) * light * k, 0, 1)) * 255)


def draw_spikes(img, depth, cam, lamp=1.0, t=0.0, glow=1.0):
    cp = np.array(cam.pos)
    cand = [s for s in SPIKES if abs(s[0][1] - cp[1]) < 45]
    cand.sort(key=lambda s: -np.linalg.norm(s[0] - cp))
    out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    for base, dirv, L, r, kind in cand:
        rel = base - cp
        dist = np.linalg.norm(rel)
        if dist < 3.2 and kind == "spike":
            continue
        tip = base + dirv * L
        if np.linalg.norm(tip - cp) < 2.8 or np.linalg.norm(base + dirv * L * 0.5 - cp) < 2.8:
            continue
        q = cam.project(np.array([base, tip, base + dirv * L * 0.5]))
        if (q[2] < 0.3).any():
            continue
        bx, by = int(q[0][0]), int(q[1][0])
        mx, my = int(q[0][2]), int(q[1][2])
        vis = False
        for xx, yy, zz in ((bx, by, q[2][0]), (mx, my, q[2][2])):
            if 0 <= xx < W and 0 <= yy < H and depth[yy, xx] > zz - 0.5:
                vis = True
        if not vis:
            continue
        cosang = rel @ cam.F / (dist + 1e-6)
        light = lamp * (max(0.0, (cosang - 0.84) / 0.16) ** 1.3 + 0.1) * 3.2 / (1 + dist ** 2 / 14) * math.exp(-dist / 45)
        light = min(light, 0.9) + 0.02 + 0.3 * glow * math.exp(-dist / 25) * float(MEADOW(base[1]))
        w = max(1.5, r * cam.f / q[2][0])
        d2 = np.array([q[0][1] - q[0][0], q[1][1] - q[1][0]])
        ln = np.linalg.norm(d2) + 1e-6
        perp = np.array([-d2[1], d2[0]]) / ln * w
        B = np.array([q[0][0], q[1][0]])
        T = np.array([q[0][1], q[1][1]])
        if kind == "spike":
            # a cone: lit flank, shadowed flank, a darker wet base, a pale tip
            for (k0, k1), fl_l, fl_r in (((0.0, 0.35), 0.55, 0.3), ((0.35, 0.75), 0.85, 0.42), ((0.75, 1.0), 1.1, 0.6)):
                P0, P1 = B + (T - B) * k0, B + (T - B) * k1
                w0, w1 = 1 - k0, 1 - k1
                left = np.array([P0 + perp * w0, P1 + perp * w1, P1, P0])
                right = np.array([P0, P1, P1 - perp * w1, P0 - perp * w0])
                cv2.fillPoly(out, [(left * 8).astype(np.int32)], _shade(light, fl_l), cv2.LINE_AA, 3)
                cv2.fillPoly(out, [(right * 8).astype(np.int32)], _shade(light, fl_r), cv2.LINE_AA, 3)
            if light > 0.4:
                cv2.line(out, tuple((B + perp * 0.45).astype(int) * 8), tuple(((B + T) / 2 + perp * 0.2).astype(int) * 8), _shade(light, 1.6), 1, cv2.LINE_AA, 3)
        elif kind == "hand":       # a stalagmite grown into the shape of a hand
            arm = np.array([B + perp * 1.1, T + perp * 0.45, T, B])
            cv2.fillPoly(out, [(arm * 8).astype(np.int32)], _shade(light, 0.85), cv2.LINE_AA, 3)
            arm = np.array([B, T, T - perp * 0.45, B - perp * 1.1])
            cv2.fillPoly(out, [(arm * 8).astype(np.int32)], _shade(light, 0.45), cv2.LINE_AA, 3)
            for k in range(5):
                a_ = (k - 2) * 0.32
                fx = T[0] + (math.cos(a_) * d2[0] - math.sin(a_) * d2[1]) * (0.26 - 0.04 * abs(k - 2))
                fy = T[1] + (math.sin(a_) * d2[0] + math.cos(a_) * d2[1]) * (0.26 - 0.04 * abs(k - 2))
                cv2.line(out, tuple((T * 8).astype(int)), (int(fx * 8), int(fy * 8)), _shade(light, 0.55 + 0.08 * k), int(max(1, w * 0.35)), cv2.LINE_AA, 3)
        else:                      # a stapler-shaped one, because Level 8
            rect = np.array([B + perp * 0.9, T + perp * 0.9, T - perp * 0.1, B - perp * 0.1])
            cv2.fillPoly(out, [(rect * 8).astype(np.int32)], _shade(light, 0.9), cv2.LINE_AA, 3)
            rect2 = np.array([B - perp * 0.25, T - perp * 0.25, T - perp * 0.9, B - perp * 0.9])
            cv2.fillPoly(out, [(rect2 * 8).astype(np.int32)], _shade(light, 0.45), cv2.LINE_AA, 3)
    return out.astype(np.float32) / 255.0


def draw_mushrooms(img, depth, cam, t, glow=1.0):
    cp = np.array(cam.pos)
    g = np.zeros((H, W), np.float32)
    for p, sz, ph in MUSH:
        if abs(p[1] - cp[1]) > 60:
            continue
        q = cam.project(np.array([p, p + np.array([0, 0, sz * 1.1])]))
        if (q[2] < 0.3).any():
            continue
        bx_, by_ = int(q[0][0]), int(q[1][0])
        if not (0 <= bx_ < W and 0 <= by_ < H) or depth[by_, bx_] < q[2][0] - 0.6:
            continue
        x, y = int(q[0][1]), int(q[1][1])
        s = cam.f / q[2][0]
        rcap = max(1, int(sz * 0.7 * s))
        a = (0.75 + 0.25 * math.sin(t * 1.3 + ph)) * math.exp(-q[2][0] / 50)
        cv2.line(g, (bx_, by_), (x, y), 0.35 * a, max(1, rcap // 4), cv2.LINE_AA)
        cv2.ellipse(g, (x, y), (rcap, max(1, int(rcap * 0.55))), 0, 180, 360, a, -1, cv2.LINE_AA)
        cv2.ellipse(g, (x, y), (rcap, max(1, int(rcap * 0.2))), 0, 0, 180, 0.5 * a, -1, cv2.LINE_AA)
    halo = cv2.GaussianBlur(g, (0, 0), px(6)) * 1.6 + cv2.GaussianBlur(g, (0, 0), px(40)) * 3.0
    return img + (g * 0.9 + halo)[..., None] * np.array([0.22, 0.8, 0.95], np.float32) * glow


def draw_floaters(img, depth, cam, t, n=40, seed=11, center=None, lamp=1.0):
    """rocks hanging in mid-air: gravity here has its own ideas"""
    rng = np.random.default_rng(seed)
    cp = np.array(cam.pos)
    c0 = np.array(center if center is not None else (path(np.array([150.0]))[0][0], 150.0, floor_of(150.0)[0] + 7.0))
    items = []
    for k in range(n):
        p = c0 + rng.uniform([-16, -22, -3], [16, 22, 12]) + np.array([0, 0, 0.6 * math.sin(t * 0.3 + k)])
        items.append((p, rng.uniform(0.3, 2.2), rng.uniform(0, 6.28)))
    items.sort(key=lambda it: -np.linalg.norm(it[0] - cp))
    out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    for p, r, ph in items:
        q = cam.project(p[None])
        if q[2][0] < 0.5:
            continue
        x, y = int(q[0][0]), int(q[1][0])
        if not (-200 < x < W + 200 and -200 < y < H + 200):
            continue
        if 0 <= x < W and 0 <= y < H and depth[y, x] < q[2][0]:
            continue
        s = cam.f / q[2][0]
        rel = p - cp
        dist = np.linalg.norm(rel)
        cosang = rel @ cam.F / (dist + 1e-6)
        light = lamp * (max(0.0, (cosang - 0.8) / 0.2) + 0.15) * 1.0 / (1 + dist ** 2 / 25) + 0.04
        c = tuple(float(v) for v in l2s(s2l((110, 100, 90)) * min(light, 1.2)) * 255)
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False) + ph + t * 0.05
        jit = 0.88 + 0.1 * np.sin(ang * 2 + ph) + 0.06 * np.sin(ang * 5 + 2 * ph)
        poly = np.stack([x + np.cos(ang) * r * s * jit, y + np.sin(ang) * r * s * jit * 0.8], 1)
        x0, y0 = int(max(0, poly[:, 0].min() - 2)), int(max(0, poly[:, 1].min() - 2))
        x1, y1 = int(min(W, poly[:, 0].max() + 3)), int(min(H, poly[:, 1].max() + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [((poly - [x0, y0]) * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
        m = m.astype(np.float32)[..., None] / 255
        # dark crown, lit cold from the meadow beneath
        gy = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None, None]
        gy = np.clip((gy * (y1 - y0) + y0 - (y - r * s * 0.8)) / (1.6 * r * s + 1e-6), 0, 1) ** 2
        g = math.exp(-dist / 60)
        top = np.array(c, np.float32)
        bot = top * 0.5 + np.array([18, 62, 58], np.float32) * g
        tex = 0.8 + 0.4 * _texb(N_C, np.arange(x0, x1)[None, :] * 0.5 + ph * 50, np.arange(y0, y1)[:, None] * 0.5, 1.0)[..., None]
        col_ = (top * (1 - gy) + bot * gy) * tex
        roi = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(roi * (1 - m) + col_ * m, 0, 255).astype(np.uint8)
    return out.astype(np.float32) / 255.0


def road_post(y):
    """the base of the 9th Road's cable stake at y (left of the stream)"""
    fl, cx = floor_of(y)
    return np.array([cx - 1.15 + 0.15 * math.sin(y * 0.3), y, fl + 0.15])


def road_bulbs(y0, y1):
    pts = []
    for y in np.arange(y0, y1, 1.0):
        ya, yb = 4.0 * math.floor(y / 4.0), 4.0 * math.floor(y / 4.0) + 4.0
        a, b = road_post(ya) + [0, 0, 1.9], road_post(yb) + [0, 0, 1.9]
        f = (y - ya) / 4.0
        pts.append(a * (1 - f) + b * f - np.array([0, 0, 0.35 * math.sin(math.pi * f)]))
    return np.array(pts)


def rope_lights(img, depth, cam, t, y0, y1, on=1.0):
    """the 9th Road: a cable of small warm bulbs strung between wooden stakes"""
    cp = np.array(cam.pos)
    y0 = max(y0, cp[1] - 4)
    pts = road_bulbs(y0, y1)
    q = cam.project(pts)
    g = np.zeros((H, W), np.float32)
    out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    # stakes: dark silhouettes with a warm edge from the bulbs
    for y in np.arange(4.0 * math.ceil(y0 / 4.0), y1, 4.0):
        b = road_post(y)
        qq = cam.project(np.array([b, b + [0, 0, 1.95]]))
        if (qq[2] < 0.3).any():
            continue
        bx, by = int(qq[0][0]), int(qq[1][0])
        if 0 <= bx < W and 0 <= by < H and depth[by, bx] < qq[2][0] - 0.5:
            continue
        w = max(1, int(0.07 * cam.f / qq[2][0]))
        cv2.line(out, (bx, by), (int(qq[0][1]), int(qq[1][1])), (26, 20, 15), w, cv2.LINE_AA)
        cv2.line(out, (bx + w // 2, by), (int(qq[0][1]) + w // 2, int(qq[1][1])), (90, 62, 38), max(1, w // 3), cv2.LINE_AA)
    img = out.astype(np.float32) / 255.0
    for k in range(len(pts) - 1):
        if q[2][k] > 0.3 and q[2][k + 1] > 0.3:
            cv2.line(g, (int(q[0][k]), int(q[1][k])), (int(q[0][k + 1]), int(q[1][k + 1])), 0.06, 1, cv2.LINE_AA)
    for k in range(len(pts)):
        if q[2][k] > 0.3:
            x, y = int(q[0][k]), int(q[1][k])
            if 0 <= x < W and 0 <= y < H and depth[y, x] > q[2][k] - 0.6:
                fl = 1.0 if (k * 7919 + int(t * 3)) % 97 else 0.3
                cv2.circle(g, (x, y), max(1, int(0.05 * cam.f / q[2][k])), fl, -1, cv2.LINE_AA)
    halo = cv2.GaussianBlur(g, (0, 0), px(8)) * 3 + cv2.GaussianBlur(g, (0, 0), px(30)) * 4
    return img + (g + halo)[..., None] * np.array([1.0, 0.72, 0.38], np.float32) * on


def road_lights(y0, y1, step=5.0):
    """the bulbs as point lights for the ray-marcher"""
    out = []
    for y in np.arange(y0, y1, step):
        p = road_bulbs(y, y + 1.0)[0]
        out.append(((p[0], p[1], p[2]), (0.5, 0.33, 0.16), 3.0))
    return out


SIGN_Y = 318.0


def sign_9th(img, depth, cam, y, art):
    """a hand-painted plywood sign on two stakes beside the road, facing the way she comes"""
    fl, cx = floor_of(y)
    c = np.array([cx - 1.9, y, fl])
    tl, tr = c + [-1.0, 0, 2.3], c + [1.0, 0, 2.3]
    br, bl = c + [1.0, 0, 1.5], c + [-1.0, 0, 1.5]
    q = cam.project(np.array([tl, tr, br, bl, c + [-0.8, 0, 0], c + [0.8, 0, 0], c + [-0.8, 0, 1.5], c + [0.8, 0, 1.5]]))
    if (q[2] < 0.5).any():
        return img
    out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    for i in (4, 5):
        cv2.line(out, (int(q[0][i]), int(q[1][i])), (int(q[0][i + 2]), int(q[1][i + 2])), (30, 22, 16), max(1, int(0.08 * cam.f / q[2][i])), cv2.LINE_AA)
    img = out.astype(np.float32) / 255.0
    h, w = art.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(np.stack([q[0][:4], q[1][:4]], 1)))
    warped = cv2.warpPerspective(art, M, (W, H))
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H)) * (depth > q[2][:4].min() - 0.8)
    return img * (1 - mask[..., None]) + warped * mask[..., None]


def wrangler(img, cam, t, t0, y_mid, lamp=1.0):
    """a section of a Wrangler's body sliding out of one wall and into the other, high in the chamber"""
    k = (t - t0) * 7.0
    cx, cz = path(np.array([y_mid]))
    fl = floor_of(y_mid)[0]
    pts = []
    for s_ in np.linspace(-45, 45, 90):
        x = float(cx[0]) + s_
        y = y_mid + 12 + 5 * math.sin(s_ * 0.05 + k * 0.04)
        z = fl + 21 + 3 * math.sin(s_ * 0.07 - k * 0.08)
        pts.append((x, y, z))
    pts = np.array(pts)
    q = cam.project(pts)
    out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    R = 3.6
    for layer in range(3):
        for i in range(len(pts) - 1):
            if q[2][i] < 1 or q[2][i + 1] < 1:
                continue
            w = R * cam.f / q[2][i]
            seg = int((pts[i][0] - k) // 2.5) % 2
            p0 = np.array([q[0][i], q[1][i]])
            p1 = np.array([q[0][i + 1], q[1][i + 1]])
            if layer == 0:      # the body, barely darker than the dark
                c, off, th = ((10, 9, 8) if seg else (17, 15, 13)), 0.0, 2 * w
            elif layer == 1:    # its belly, lit cold from the mushroom meadow below
                c, off, th = (int(14 + 4 * seg), int(46 + 8 * seg), int(44 + 8 * seg)), 0.78 * w, 0.35 * w
            else:               # a wet glint along the plates
                c, off, th = (34, 40, 40), -0.5 * w, max(1, 0.04 * w)
            d0 = np.array([0, off])
            cv2.line(out, tuple(((p0 + d0) * 8).astype(int)), tuple(((p1 + d0) * 8).astype(int)), c, int(max(1, th)), cv2.LINE_AA, 3)
    return out.astype(np.float32) / 255.0

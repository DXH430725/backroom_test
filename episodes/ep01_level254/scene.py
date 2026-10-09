"""
Ep01 — LEVEL 254 "Blue Heaven": scene renderers.

The blue void with height fog, the Meditation Circle (~300k seated figures in three
levels of detail), Level 287's white hall of black doors, and a glimpse of Level 11.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, px, s2l, l2s, to_u8, value_noise

# "the most beautiful shade of blue you will ever see"
GROUND = s2l((34, 92, 236))
ZENITH = s2l((20, 66, 212))
HORIZON = s2l((158, 198, 255))
GLOW = s2l((215, 232, 255))
FIG = s2l((16, 26, 58))

RHO0 = 1.0 / 170.0   # fog density at the floor (1/m)
HF = 40.0            # fog scale height (m)
RHO_SKY = 0.05       # thin luminous haze that never thins out: keeps heaven soft at altitude


def optical_depth(z0, z1, t):
    z0 = np.asarray(z0, np.float64)
    z1 = np.asarray(z1, np.float64)
    dz = z1 - z0
    a = np.exp(-z0 / HF)
    b = np.exp(-z1 / HF)
    flat = np.abs(dz) < 1e-3
    safe = np.where(flat, 1.0, dz)
    return RHO0 * t * np.where(flat, a, HF * (a - b) / safe)


# ----------------------------------------------------------------------------
# the blue void
# ----------------------------------------------------------------------------
def blue_background(cam):
    bw, bh = W // 4, H // 4
    X = (np.arange(bw) + 0.5) * 4 - W / 2
    Y = H / 2 - (np.arange(bh) + 0.5) * 4
    X, Y = np.meshgrid(X, Y)
    dz = (cam.F[2] * cam.f + cam.R[2] * X + cam.U[2] * Y) / np.sqrt(cam.f ** 2 + X ** 2 + Y ** 2)
    cz = max(cam.pos[2], 0.02)
    down = dz < -1e-5
    tg = np.where(down, cz / np.maximum(-dz, 1e-5), 1e7)
    tau_g = optical_depth(cz, 0.0, tg)
    tau_s = (RHO0 * HF * math.exp(-cz / HF) + RHO_SKY) / np.maximum(dz, 2e-4)
    tau = np.where(down, tau_g, tau_s)
    T = np.exp(-tau)[..., None]
    base = np.where(down[..., None], GROUND, ZENITH)
    col = base * T + HORIZON * (1 - T)
    col = col + GLOW * (0.22 * np.exp(-np.abs(dz) / 0.025))[..., None]
    col = l2s(col.astype(np.float32))
    return cv2.resize(col, (W, H), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------------
# the Meditation Circle
# ----------------------------------------------------------------------------
R0, DR, RMAX = 3.2, 1.7, 450.0
SEAT = np.array([0.0, -3.2])

# local figure skeleton (forward, left, up) — a person sitting cross-legged, head bowed
PTS = np.array([
    (-0.02, 0.00, 0.20),  # 0 pelvis
    (0.00, 0.00, 0.56),   # 1 chest
    (0.02, 0.00, 0.72),   # 2 neck
    (0.07, 0.00, 0.83),   # 3 head (bowed)
    (0.00, 0.19, 0.63),   # 4 shoulder L
    (0.00, -0.19, 0.63),  # 5 shoulder R
    (0.12, 0.27, 0.37),   # 6 elbow L
    (0.12, -0.27, 0.37),  # 7 elbow R
    (0.27, 0.19, 0.21),   # 8 hand L
    (0.27, -0.19, 0.21),  # 9 hand R
    (0.03, 0.12, 0.14),   # 10 hip L
    (0.03, -0.12, 0.14),  # 11 hip R
    (0.30, 0.34, 0.10),   # 12 knee L
    (0.30, -0.34, 0.10),  # 13 knee R
    (0.17, -0.10, 0.06),  # 14 ankle L (crossed)
    (0.17, 0.10, 0.06),   # 15 ankle R
], np.float64)
CAPS = [  # (i, j, radius, shade)
    (10, 12, 0.090, 0.82), (11, 13, 0.090, 0.82),
    (12, 14, 0.068, 0.78), (13, 15, 0.068, 0.78),
    (0, 1, 0.170, 0.95), (1, 2, 0.140, 1.04),
    (4, 5, 0.085, 1.08),
    (4, 6, 0.060, 1.0), (5, 7, 0.060, 1.0),
    (6, 8, 0.050, 0.94), (7, 9, 0.050, 0.94),
    (2, 3, 0.050, 1.0),
    (3, 3, 0.108, 1.22),  # head
]


def build_circle(seed=254):
    rng = np.random.default_rng(seed)
    xs, ys, yaws, scs, shs, app = [], [], [], [], [], []
    k = 0
    r = R0
    while r < RMAX:
        if k > 0 and k % 26 == 0:  # concentric walkways
            k += 1
            r += DR
            continue
        n = int(2 * math.pi * r / 1.25)
        if k == 0:
            ang = -math.pi / 2 + np.arange(n) * 2 * math.pi / n
            keep = np.ones(n, bool)
            keep[0] = False  # the empty seat
        else:
            ang = rng.uniform(0, 2 * math.pi / n) + np.arange(n) * 2 * math.pi / n
            keep = rng.random(n) > 0.01
            naisle = 4 if r < 110 else (8 if r < 260 else 16)
            for i in range(naisle):
                a0 = -math.pi / 2 + i * 2 * math.pi / naisle
                d = np.abs((ang - a0 + math.pi) % (2 * math.pi) - math.pi) * r
                keep &= d > (1.35 if i % (naisle // 4) == 0 else 1.0)
        ang = ang[keep]
        m = len(ang)
        rr = r + (rng.uniform(-0.08, 0.08, m) if k else 0)
        ang = ang + (rng.uniform(-0.08, 0.08, m) / r if k else 0)
        xs.append(rr * np.cos(ang))
        ys.append(rr * np.sin(ang))
        yaws.append(ang + math.pi + (rng.uniform(-0.12, 0.12, m) if k else 0))
        scs.append(rng.uniform(0.9, 1.08, m))
        shs.append(rng.uniform(0.82, 1.18, m))
        app.append(np.full(m, -1e9))
        k += 1
        r += DR
    # the new arrival — appears in the empty seat once you have sat down
    xs.append(np.array([SEAT[0]]))
    ys.append(np.array([SEAT[1]]))
    yaws.append(np.array([math.pi / 2]))
    scs.append(np.array([1.0]))
    shs.append(np.array([1.1]))
    app.append(np.array([141.2]))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(yaws),
            np.concatenate(scs), np.concatenate(shs), np.concatenate(app))


CIRCLE = build_circle()


def fig_colors(T, shade):
    """sRGB 0..255 colour of a figure part given fog transmittance T and shade (arrays)."""
    lin = FIG[None, :] * shade[:, None] * T[:, None] + HORIZON[None, :] * (1 - T[:, None])
    return l2s(lin) * 255.0


def draw_circle_scene(cam, t, img, shadows=True):
    """Draw every member of the Meditation Circle into img (float sRGB, HxWx3)."""
    FX, FY, FYAW, FSC, FSH, FAPP = CIRCLE
    alive = FAPP <= t
    cx, cy, cz = cam.pos
    relx, rely, relz = FX - cx, FY - cy, 0.45 - cz
    d = relx * cam.F[0] + rely * cam.F[1] + relz * cam.F[2]
    vis = alive & (d > 0.25)
    idx = np.nonzero(vis)[0]
    relx, rely, d = relx[idx], rely[idx], d[idx]
    sx = W / 2 + cam.f * (relx * cam.R[0] + rely * cam.R[1] + relz * cam.R[2]) / d
    sy = H / 2 - cam.f * (relx * cam.U[0] + rely * cam.U[1] + relz * cam.U[2]) / d
    hp = 0.95 * cam.f * FSC[idx] / d
    mg = px(40)
    m = (sx > -hp - mg) & (sx < W + hp + mg) & (sy > -hp - mg) & (sy < H + hp + mg)
    idx, sx, sy, d, hp = idx[m], sx[m], sy[m], d[m], hp[m]
    dist = np.sqrt((FX[idx] - cx) ** 2 + (FY[idx] - cy) ** 2 + relz ** 2)
    T = np.exp(-optical_depth(cz, 0.45, dist)).astype(np.float32)

    # --- soft contact shadows: one blob per figure, blurred in buckets of apparent size
    if shadows:
        near = np.nonzero((dist < 110) & (hp > 4 * S))[0]
        if len(near):
            th = np.linspace(0, 2 * math.pi, 20, endpoint=False)
            ii = idx[near]
            ca, sa = np.cos(FYAW[ii]), np.sin(FYAW[ii])
            ox = FX[ii] + 0.12 * ca
            oy = FY[ii] + 0.12 * sa
            rad = 0.5 * FSC[ii]
            P = np.stack([ox[:, None] + rad[:, None] * np.cos(th)[None] * 1.1,
                          oy[:, None] + rad[:, None] * np.sin(th)[None],
                          np.zeros((len(ii), len(th)))], -1)
            qx, qy, pd = cam.project(P)
            okp = (pd > 0.05).all(1)
            buckets = [(0, 10 * S, 1.2 * S), (10 * S, 40 * S, 4.0 * S), (40 * S, 150 * S, 13.0 * S), (150 * S, 1e9, 40.0 * S)]
            total = np.zeros((H // 2, W // 2), np.float32)
            for lo, hi, sig in buckets:
                sel = np.nonzero(okp & (hp[near] > lo) & (hp[near] <= hi))[0]
                if not len(sel):
                    continue
                sm = np.zeros((H // 2, W // 2), np.float32)
                for q in sel:
                    poly = (np.stack([qx[q], qy[q]], 1) * 0.5 * 16).astype(np.int32)
                    cv2.fillPoly(sm, [poly], float(T[near[q]]), cv2.LINE_AA, shift=4)
                sm = cv2.GaussianBlur(sm, (0, 0), sig)
                total = np.maximum(total, sm)
            total = cv2.resize(total, (W, H), interpolation=cv2.INTER_LINEAR)
            img *= (1 - 0.55 * total)[..., None]

    # --- tiny LOD: bilinear splats of coverage
    tiny = hp <= 3.2 * S
    if tiny.any():
        # splat on a 1080p-sized grid at every resolution so the crowd texture looks the same in 4K
        gw, gh = int(round(W / S)), int(round(H / S))
        tx, ty, td, tT = sx[tiny] / S, sy[tiny] / S, d[tiny], T[tiny]
        cov = np.minimum(1.5 * (FSC[idx[tiny]] * cam.f / td) ** 2 / (S * S), 6.0)
        col = fig_colors(tT, FSH[idx[tiny]] * 0.98) / 255.0
        x0 = np.floor(tx - 0.5).astype(np.int64)
        y0 = np.floor(ty - 0.5).astype(np.int64)
        fx = (tx - 0.5 - x0).astype(np.float32)
        fy = (ty - 0.5 - y0).astype(np.float32)
        acc_a = np.zeros(gw * gh, np.float32)
        acc_c = np.zeros((3, gw * gh), np.float32)
        for ox, oy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xx, yy = x0 + ox, y0 + oy
            ok = (xx >= 0) & (xx < gw) & (yy >= 0) & (yy < gh)
            lin = (yy * gw + xx)[ok]
            wv = (wgt * cov)[ok]
            acc_a += np.bincount(lin, wv, gw * gh).astype(np.float32)
            for c in range(3):
                acc_c[c] += np.bincount(lin, wv * col[ok, c], gw * gh).astype(np.float32)
        a = acc_a.reshape(gh, gw)
        alpha = 1 - np.exp(-a)
        cc = (acc_c / np.maximum(acc_a, 1e-6)).reshape(3, gh, gw).transpose(1, 2, 0)
        if S != 1:
            alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
            cc = cv2.resize(np.ascontiguousarray(cc), (W, H), interpolation=cv2.INTER_LINEAR)
        img[:] = img * (1 - alpha[..., None]) + cc * alpha[..., None]

    # --- dither to 8 bit for OpenCV's antialiased primitives
    out = to_u8(img)

    order = np.argsort(-d)
    mid_or_full = order[~tiny[order]]
    SH = 3  # subpixel bits
    SS = 1 << SH
    for j in mid_or_full:
        i = idx[j]
        ca, sa = math.cos(FYAW[i]), math.sin(FYAW[i])
        sc = FSC[i]
        # local -> world
        Lx = FX[i] + (PTS[:, 0] * ca - PTS[:, 1] * sa) * sc
        Ly = FY[i] + (PTS[:, 0] * sa + PTS[:, 1] * ca) * sc
        Lz = PTS[:, 2] * sc
        P = np.stack([Lx, Ly, Lz], 1)
        qx, qy, pd = cam.project(P)
        if (pd < 0.08).any():
            continue
        Tj = float(T[j])
        base = FSH[i]
        if hp[j] <= 16 * S:
            parts = [(12, 13, 0.12, 0.85), (0, 2, 0.16, 1.0), (3, 3, 0.108, 1.2)]
        else:
            parts = CAPS
            dep = np.array([(pd[a] + pd[b]) * 0.5 for a, b, _, _ in parts])
            parts = [parts[q] for q in np.argsort(-dep)]
        shades = np.array([p[3] for p in parts]) * base
        cols = fig_colors(np.full(len(parts), Tj, np.float32), shades)
        hcols = fig_colors(np.full(len(parts), Tj, np.float32), shades * 1.45)
        full = hp[j] > 16 * S
        for q, (a, b, r, _) in enumerate(parts):
            dd = (pd[a] + pd[b]) * 0.5
            rad = r * sc * cam.f / dd
            p1 = (int(qx[a] * SS), int(qy[a] * SS))
            p2 = (int(qx[b] * SS), int(qy[b] * SS))
            col = tuple(float(c) for c in cols[q])
            if a == b:
                cv2.circle(out, p1, max(int(rad * SS), SS // 2), col, -1, cv2.LINE_AA, SH)
                if full and rad > 3 * S:
                    hc = tuple(float(c) for c in hcols[q])
                    off = int(rad * 0.3 * SS)
                    cv2.circle(out, (p1[0], p1[1] - off), int(rad * 0.62 * SS), hc, -1, cv2.LINE_AA, SH)
            else:
                th = max(1, int(round(2 * rad)))
                cv2.line(out, p1, p2, col, th, cv2.LINE_AA, SH)
                if full and th > 6 * S:
                    hc = tuple(float(c) for c in hcols[q])
                    off = int(rad * 0.28 * SS)
                    cv2.line(out, (p1[0], p1[1] - off), (p2[0], p2[1] - off), hc, max(1, int(th * 0.45)), cv2.LINE_AA, SH)
    return out.astype(np.float32) / 255.0


def blue_scene(cam, t, figures=True):
    img = blue_background(cam)
    if figures:
        img = draw_circle_scene(cam, t, img)
    return img


# ----------------------------------------------------------------------------
# Level 287 — "Glitched Halls": white corridor, black doors, flickering tubes
# ----------------------------------------------------------------------------
NOISE_A = value_noise(512, 6, 1)
NOISE_B = value_noise(512, 6, 2)
HALL_W, HALL_H, PERIOD = 1.25, 2.7, 4.0


def hall_scene(cam, t, flicker, ghost):
    # ray directions at full res
    dx, dy, dz = cam.rays()
    cx, cy, cz = cam.pos
    big = np.float32(1e6)
    tR = np.where(dx > 1e-6, (HALL_W - cx) / np.where(dx > 1e-6, dx, 1), big)
    tL = np.where(dx < -1e-6, (-HALL_W - cx) / np.where(dx < -1e-6, dx, 1), big)
    tF = np.where(dz < -1e-6, (0 - cz) / np.where(dz < -1e-6, dz, 1), big)
    tC = np.where(dz > 1e-6, (HALL_H - cz) / np.where(dz > 1e-6, dz, 1), big)
    tt = np.minimum(np.minimum(tR, tL), np.minimum(tF, tC))
    hx, hy, hz = cx + dx * tt, cy + dy * tt, cz + dz * tt
    wall = (tt == tR) | (tt == tL)
    floor = tt == tF
    ceil = tt == tC
    side = np.where(tt == tR, 1, -1)

    alb = np.full(tt.shape, 0.78, np.float32)
    # walls: white paint with stains
    u = ((hy * 60).astype(np.int64)) % 512
    v = ((hz * 60).astype(np.int64)) % 512
    stain = NOISE_A[v, u]
    alb = np.where(wall, 0.80 - 0.18 * np.clip(stain - 0.55, 0, 1) * 2.2, alb)
    # doors: black slabs with a frame, alternating sides
    yoff = np.where(side > 0, 3.0, 1.0)
    dyy = (hy - yoff) % PERIOD
    dyy = np.minimum(dyy, PERIOD - dyy)
    door = wall & (dyy < 0.48) & (hz < 2.15)
    frame = wall & (dyy < 0.56) & (hz < 2.23) & ~door
    alb = np.where(frame, 0.35, alb)
    alb = np.where(door, 0.035 + 0.02 * NOISE_B[v, u], alb)
    knob = door & (np.abs(dyy - 0.36) < 0.035) & (np.abs(hz - 1.0) < 0.035)
    alb = np.where(knob, 0.4, alb)
    # floor: worn grey tiles
    fu = ((hx * 50).astype(np.int64)) % 512
    fv = ((hy * 50).astype(np.int64)) % 512
    tile = ((np.floor(hx / 0.6) + np.floor(hy / 0.6)) % 2).astype(np.float32)
    alb = np.where(floor, 0.36 + 0.05 * tile - 0.1 * NOISE_B[fv, fu], alb)
    grout = floor & ((np.abs((hx / 0.6) % 1 - 0.5) > 0.48) | (np.abs((hy / 0.6) % 1 - 0.5) > 0.48))
    alb = np.where(grout, 0.22, alb)
    # ceiling panels + tubes
    alb = np.where(ceil, 0.7, alb)
    ly = (hy - 2.0) % PERIOD
    ly = np.minimum(ly, PERIOD - ly)
    tube = ceil & (ly < 0.62) & (np.abs(hx) < 0.28)

    # lighting from the three nearest tubes
    li = np.zeros_like(tt)
    base_i = np.floor((hy - 2.0) / PERIOD)
    for k in (-1, 0, 1, 2):
        yl = (base_i + k) * PERIOD + 2.0
        idx = (base_i + k).astype(np.int64)
        fl = flicker(idx)
        d2 = hx ** 2 + (hy - yl) ** 2 + (hz - 2.65) ** 2
        li += fl * 2.3 / (1 + d2 / 2.2) ** 1.25
    lin = alb * (0.05 + li)
    tube_i = flicker(np.floor((hy - 2.0 + PERIOD / 2) / PERIOD).astype(np.int64))
    lin = np.where(tube, 1.6 * tube_i + 0.05, lin)
    fog = np.exp(-tt / 22.0)
    lin = lin * fog
    col = np.stack([lin * 1.02, lin * 1.0, lin * 0.9], -1)
    img = l2s(np.clip(col, 0, 1))

    out = to_u8(img, dither=False)
    if ghost is not None:
        gy, galpha = ghost
        # faceless man in a 1930s suit and hat, standing still at the end of the hall
        P = np.array([
            (0.25, gy, 0.0), (0.25, gy, 0.9), (-0.25, gy, 0.0), (-0.25, gy, 0.9),
            (0, gy, 0.95), (0, gy, 1.5), (0, gy, 1.72), (-0.3, gy, 1.5), (0.3, gy, 1.5),
            (-0.34, gy, 0.85), (0.34, gy, 0.85), (0, gy, 1.86),
        ])
        qx, qy, pd = cam.project(P)
        if (pd > 0.5).all():
            s = cam.f / pd[0]
            c = tuple([int(12 + 20 * (1 - galpha))] * 3)
            L = cv2.LINE_AA
            pt = lambda k: (int(qx[k] * 8), int(qy[k] * 8))
            cv2.line(out, pt(0), pt(1), c, max(1, int(0.2 * s)), L, 3)
            cv2.line(out, pt(2), pt(3), c, max(1, int(0.2 * s)), L, 3)
            cv2.line(out, pt(4), pt(5), c, max(1, int(0.58 * s)), L, 3)
            cv2.line(out, pt(7), pt(9), c, max(1, int(0.13 * s)), L, 3)
            cv2.line(out, pt(8), pt(10), c, max(1, int(0.13 * s)), L, 3)
            cv2.circle(out, pt(6), int(0.13 * s * 8), c, -1, L, 3)
            hb = (int(qx[11] * 8), int(qy[11] * 8))
            cv2.ellipse(out, (hb[0], hb[1] + int(0.09 * s * 8)), (int(0.25 * s * 8), int(0.035 * s * 8)), 0, 0, 360, c, -1, L, 3)
            cv2.ellipse(out, (hb[0], hb[1] + int(0.03 * s * 8)), (int(0.14 * s * 8), int(0.09 * s * 8)), 0, 180, 360, c, -1, L, 3)
    return out.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------
# Level 11 — "The Endless City" (a three-second glimpse)
# ----------------------------------------------------------------------------
def _build_city():
    rng = np.random.default_rng(11)
    w, h = int(W * 1.3), int(H * 1.3)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    sky = np.stack([0.55 + 0.1 * yy, 0.57 + 0.1 * yy, 0.6 + 0.08 * yy], -1) * np.ones((h, w, 1), np.float32)
    img = sky.copy()
    for layer, (col, hmin, hmax, wmin, wmax, lit) in enumerate([
        ((0.44, 0.46, 0.5), 0.25, 0.55, px(60), px(180), 0.05),
        ((0.3, 0.31, 0.35), 0.35, 0.75, px(90), px(260), 0.1),
        ((0.15, 0.16, 0.19), 0.45, 0.95, px(140), px(380), 0.16),
    ]):
        x = -px(40)
        while x < w:
            bw = int(rng.uniform(wmin, wmax))
            bh = int(h * rng.uniform(hmin, hmax))
            top = h - bh
            img[top:, x:x + bw] = col
            # windows
            wy = np.arange(top + px(14), h - px(10), px(18 + layer * 4))
            wx = np.arange(x + px(8), x + bw - px(8), px(14 + layer * 4))
            for yv in wy:
                on = rng.random(len(wx)) < lit
                for xv in wx[on]:
                    img[yv:yv + px(6 + layer * 2), xv:xv + px(5 + layer * 2)] = (0.95, 0.82, 0.55)
            if rng.random() < 0.3:
                ah = px(rng.uniform(30, 90))
                img[max(top - ah, 0):top, x + bw // 2 - px(2):x + bw // 2 + px(2)] = col
            x += bw + px(rng.uniform(-20, 30))
        haze = np.linspace(0.5, 0.0, h, dtype=np.float32)[:, None, None] * (0.5 - 0.15 * layer)
        img = img * (1 - haze) + sky * haze
    return img


_CITY = None


def city_scene(t):
    global _CITY
    if _CITY is None:
        _CITY = _build_city()
    ox = px(90 + 60 * math.sin(t * 7.1) + 30 * t)
    oy = px(160 + 40 * math.sin(t * 5.3) - 20 * t)
    img = _CITY[oy:oy + H, ox:ox + W].copy()
    # rain streaks
    rng = np.random.default_rng(int(t * 1000))
    n = 700
    xs = rng.integers(0, W, n)
    ys = rng.integers(0, H, n)
    out = to_u8(img, dither=False)
    for x, y in zip(xs, ys):
        cv2.line(out, (int(x), int(y)), (int(x) - px(6), int(y) + px(40)), (200, 205, 215), max(1, px(1)), cv2.LINE_AA)
    return out.astype(np.float32) / 255.0


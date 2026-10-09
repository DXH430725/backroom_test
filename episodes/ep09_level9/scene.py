"""
Ep09 — LEVEL 9 "The Suburbs": an endless neighbourhood at midnight.

Houses are convex gable prisms ray-traced analytically (each one a handful of planes),
culled to their screen rectangles. The ground is a plane with road / sidewalk / lawn
materials, wet asphalt with lamp streaks and puddles (screen-space reflections). Lights:
the phone's flashlight, a few dying sodium street lamps, and — once — a Watcher's pupil.
Dead trees, poles, wires and signs are depth-tested sprites drawn on top.
"""
import math

import cv2
import numpy as np

from backrooms.core import W, H, S, px, s2l, l2s, value_noise

RS = 2.0
IW, IH = int(W / RS), int(H / RS)

N_A = value_noise(512, 6, 901)
N_B = value_noise(256, 5, 902)
N_C = value_noise(512, 4, 903)

AVENUES = [-160.0, 0.0, 160.0]          # streets along y
CROSS = [-40.0, 200.0, 290.0, 480.0]    # streets along x
ROAD, WALK = 4.0, 6.2
FRONT = 15.0                            # house fronts sit this far from the street's centre line


def to8(img):
    return (l2s(np.clip(img, 0, 1)) * 255 + 0.5).astype(np.uint8)


def from8(img, base, out):
    """write back only the pixels a cv2 drawing touched (sRGB 8-bit), keeping HDR elsewhere"""
    ch = (out != base).any(-1)
    img = img.copy()
    img[ch] = s2l(out[ch])
    return img


def _tex(T, x, y):
    n = T.shape[0]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ax, ay = x - x0, y - y0
    x0m, y0m, x1m, y1m = x0 % n, y0 % n, (x0 + 1) % n, (y0 + 1) % n
    return T[y0m, x0m] * (1 - ax) * (1 - ay) + T[y0m, x1m] * ax * (1 - ay) + T[y1m, x0m] * (1 - ax) * ay + T[y1m, x1m] * ax * ay


# ---------------------------------------------------------------------------
# the neighbourhood
# ---------------------------------------------------------------------------
SIDING = [(196, 192, 182), (176, 170, 156), (150, 160, 166), (140, 150, 132), (168, 150, 130), (120, 74, 62),
          (200, 200, 196), (110, 116, 120), (180, 168, 120), (150, 128, 140)]
ROOFS = [(46, 44, 44), (60, 52, 46), (38, 42, 48), (70, 58, 50), (52, 52, 56)]


class Poly:
    """a convex solid: n·p <= d for every plane"""

    def __init__(self, planes, bbox, mat, col, info=None):
        self.n = np.array([p[0] for p in planes], np.float64)
        self.d = np.array([p[1] for p in planes], np.float64)
        self.bbox = bbox            # (x0, y0, z0, x1, y1, z1)
        self.mat = mat              # "house" | "garage" | "car" | "hedge" | "glass"
        self.col = col
        self.info = info or {}

    @property
    def centre(self):
        b = self.bbox
        return np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2])


def box_planes(x0, y0, z0, x1, y1, z1):
    return [((1, 0, 0), x1), ((-1, 0, 0), -x0), ((0, 1, 0), y1), ((0, -1, 0), -y0), ((0, 0, 1), z1), ((0, 0, -1), -z0)]


def gable(x0, y0, x1, y1, hw, slope, along_y):
    """a house: box walls of height hw with a gable roof whose ridge runs along y (or x)"""
    if along_y:
        half = (x1 - x0) / 2
        top = hw + slope * half
        pl = box_planes(x0, y0, 0, x1, y1, top)[:5]
        pl = pl[:4] + [((0, 0, -1), 0.0)]
        pl += [((-slope, 0, 1), hw - slope * x0), ((slope, 0, 1), hw + slope * x1)]
    else:
        half = (y1 - y0) / 2
        top = hw + slope * half
        pl = box_planes(x0, y0, 0, x1, y1, top)[:4] + [((0, 0, -1), 0.0)]
        pl += [((0, -slope, 1), hw - slope * y0), ((0, slope, 1), hw + slope * y1)]
    return pl, (x0, y0, 0.0, x1, y1, top)


HOUSES = []
TWIN_Y = 168.0


def _house(rng, cx, cy, frontal_axis, side, w, d, params=None):
    """frontal_axis 'x': house faces a street along y (front plane at x = cx ...)."""
    if params is None:
        params = dict(hw=rng.choice([2.9, 2.9, 5.6]), slope=rng.uniform(0.42, 0.72), ridge_par=rng.random() < 0.7,
                      col=SIDING[rng.integers(len(SIDING))], roof=ROOFS[rng.integers(len(ROOFS))],
                      garage=rng.random() < 0.55, gside=rng.choice([-1, 1]), door=rng.uniform(-0.25, 0.25), w=w, d=d,
                      seed=int(rng.integers(1 << 30)))
    p = params
    w, d = p["w"], p["d"]
    out = []
    if frontal_axis == "x":
        xf = cx
        x0, x1 = (xf, xf + side * d) if side > 0 else (xf + side * d, xf)
        y0, y1 = cy - w / 2, cy + w / 2
        pl, bb = gable(x0, y0, x1, y1, p["hw"], p["slope"], along_y=p["ridge_par"])
        out.append(Poly(pl, bb, "house", p["col"], dict(roof=p["roof"], hw=p["hw"], front=("x", xf, side), door=cy + p["door"] * w, seed=p["seed"])))
        if p["garage"]:
            gy0 = y1 if p["gside"] > 0 else y0 - 6.5
            gx0, gx1 = (xf + side * 1.0, xf + side * 8.0) if side > 0 else (xf + side * 8.0, xf + side * 1.0)
            pl, bb = gable(gx0, gy0, gx1, gy0 + 6.5, 2.7, 0.35, along_y=False)
            out.append(Poly(pl, bb, "garage", p["col"], dict(roof=p["roof"], hw=2.7, front=("x", xf + side * 1.0, side), seed=p["seed"] + 1)))
    else:
        yf = cy
        y0, y1 = (yf, yf + side * d) if side > 0 else (yf + side * d, yf)
        x0, x1 = cx - w / 2, cx + w / 2
        pl, bb = gable(x0, y0, x1, y1, p["hw"], p["slope"], along_y=not p["ridge_par"])
        out.append(Poly(pl, bb, "house", p["col"], dict(roof=p["roof"], hw=p["hw"], front=("y", yf, side), door=cx + p["door"] * w, seed=p["seed"])))
        if p["garage"]:
            gx0 = x1 if p["gside"] > 0 else x0 - 6.5
            gy0, gy1 = (yf + side * 1.0, yf + side * 8.0) if side > 0 else (yf + side * 8.0, yf + side * 1.0)
            pl, bb = gable(gx0, gy0, gx0 + 6.5, gy1, 2.7, 0.35, along_y=True)
            out.append(Poly(pl, bb, "garage", p["col"], dict(roof=p["roof"], hw=2.7, front=("y", yf + side * 1.0, side), seed=p["seed"] + 1)))
    return out, params


TREES = []          # (x, y, height, kind)
LAMPS = []          # (x, y, side_dir_x, side_dir_y)
CARS = []


def _build():
    rng = np.random.default_rng(909)
    twin_params = None
    for X in AVENUES:
        for j in range(len(CROSS) - 1):
            ya, yb = CROSS[j] + 26, CROSS[j + 1] - 26
            for side in (-1, 1):
                y = ya
                while y < yb - 12:
                    w = rng.uniform(11, 15)
                    lot = rng.uniform(17, 22)
                    cy = y + lot / 2
                    params = None
                    if X == 0.0 and abs(cy - TWIN_Y) < lot / 2 + 0.1:
                        if twin_params is None:
                            hs, twin_params = _house(rng, X + side * FRONT, cy, "x", side, w, rng.uniform(9, 12))
                            HOUSES.extend(hs)
                            y += lot
                            continue
                        params = dict(twin_params)
                    hs, _ = _house(rng, X + side * FRONT, cy, "x", side, w, rng.uniform(9, 12), params)
                    HOUSES.extend(hs)
                    if rng.random() < 0.75:
                        TREES.append((X + side * rng.uniform(8.5, 12.5), cy + rng.uniform(-lot / 2 + 1.5, lot / 2 - 1.5), rng.uniform(6.5, 11), int(rng.integers(6))))
                    if rng.random() < 0.12 and not (X == 0.0 and 150 < cy < 215):
                        CARS.append((X + side * 2.9, cy + rng.uniform(-4, 4), 0.0 if rng.random() < 0.5 else math.pi, int(rng.integers(5))))
                    y += lot
    for Y in CROSS:
        for i in range(len(AVENUES) - 1):
            xa, xb = AVENUES[i] + 36, AVENUES[i + 1] - 36
            for side in (-1, 1):
                x = xa
                while x < xb - 12:
                    w = rng.uniform(11, 15)
                    lot = rng.uniform(17, 22)
                    hs, _ = _house(rng, x + lot / 2, Y + side * FRONT, "y", side, w, rng.uniform(9, 12))
                    HOUSES.extend(hs)
                    if rng.random() < 0.75:
                        TREES.append((x + lot / 2 + rng.uniform(-lot / 2 + 1.5, lot / 2 - 1.5), Y + side * rng.uniform(8.5, 12.5), rng.uniform(6.5, 11), int(rng.integers(6))))
                    x += lot
    # street lamps: every ~38 m, alternating sides, along every street
    for X in AVENUES:
        for k, y in enumerate(np.arange(CROSS[0] - 30, CROSS[-1] + 30, 38.0)):
            s = 1 if k % 2 else -1
            LAMPS.append((X + s * 4.9, float(y), -s, 0.0))
    for Y in CROSS:
        for k, x in enumerate(np.arange(AVENUES[0] - 30, AVENUES[-1] + 30, 38.0)):
            if min(abs(x - a) for a in AVENUES) < 8:
                continue
            s = 1 if k % 2 else -1
            LAMPS.append((float(x), Y + s * 4.9, 0.0, -s))
    # the lamp he wakes up under
    LAMPS.append((-4.9, 99.0, 1.0, 0.0))
    # the twins: two identical houses facing each other across the avenue
    keep = [h for h in HOUSES if not (abs(h.bbox[0]) < 40 and h.bbox[4] > TWIN_Y - 9 and h.bbox[1] < TWIN_Y + 9)]
    HOUSES[:] = keep
    tp = None
    for side in (-1, 1):
        hs, tp = _house(np.random.default_rng(1717), side * FRONT, TWIN_Y, "x", side, 12.5, 10.5, tp)
        HOUSES.extend(hs)
    # the car he hides behind (and no others: the street is empty)
    CARS[:] = [(2.9, 186.0, 0.0, 0)]
    for (x, y, yaw, c) in CARS:
        HOUSES.extend(_car(x, y, yaw, c))


CAR_COLS = [(40, 44, 50), (70, 36, 34), (120, 120, 118), (36, 48, 42), (64, 60, 54)]


def _car(x, y, yaw, c):
    col = CAR_COLS[c]
    L, Wd = 4.6, 1.85
    out = [Poly(box_planes(x - Wd / 2, y - L / 2, 0.28, x + Wd / 2, y + L / 2, 1.0), (x - Wd / 2, y - L / 2, 0.28, x + Wd / 2, y + L / 2, 1.0), "car", col)]
    off = -0.25 if yaw == 0.0 else 0.25
    cy0, cy1 = y - 1.2 + off, y + 1.2 + off
    # cabin: a box with sloped windscreen and rear window
    pl = box_planes(x - 0.8, cy0 - 0.3, 1.0, x + 0.8, cy1 + 0.3, 1.55)
    pl += [((0, 1.6, 1), 1.55 + 1.6 * (cy1 - 0.3)), ((0, -1.4, 1), 1.55 - 1.4 * (cy0 + 0.3))]
    out.append(Poly(pl, (x - 0.8, cy0 - 0.3, 1.0, x + 0.8, cy1 + 0.3, 1.55), "glass", col))
    return out


_build()
H_CENT = np.array([h.centre for h in HOUSES])


def _front_n(h):
    fr = h.info.get("front")
    if not fr:
        return (0.0, 0.0)
    ax_, _, side = fr
    return (-side, 0.0) if ax_ == "x" else (0.0, -side)


FRONTN = np.array([_front_n(h) for h in HOUSES], np.float32)
DOORU = np.array([h.info.get("door", 1e9) for h in HOUSES], np.float32)


# ---------------------------------------------------------------------------
# tracing
# ---------------------------------------------------------------------------
def _rect(cam, bb):
    xs = [bb[0], bb[3]]
    ys = [bb[1], bb[4]]
    zs = [bb[2], bb[5]]
    P = np.array([[a, b, c] for a in xs for b in ys for c in zs])
    rel = P - cam.pos
    d = rel @ cam.F
    if (d <= 0.05).all():
        return None
    if (d <= 0.05).any():
        return (0, 0, IW, IH)
    sx = (W / 2 + cam.f * (rel @ cam.R) / d) / RS
    sy = (H / 2 - cam.f * (rel @ cam.U) / d) / RS
    x0, x1 = int(max(0, sx.min() - 2)), int(min(IW, sx.max() + 3))
    y0, y1 = int(max(0, sy.min() - 2)), int(min(IH, sy.max() + 3))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def street_dist(x, y):
    ax = np.min(np.abs(np.asarray(x)[..., None] - np.array(AVENUES)), -1)
    ay = np.min(np.abs(np.asarray(y)[..., None] - np.array(CROSS)), -1)
    return ax, ay


def render(cam, t, flash=1.0, lamps=None, watcher=None, fog=110.0, fogcol=(0.010, 0.011, 0.014), far=260.0):
    """returns (linear img HxWx3 at full res, depth HxW, puddle mask HxW)
    lamps: list of ((x, y, z), (r, g, b) intensity) that are lit right now
    watcher: dict(pos, radius, look (unit vec), beam (0..1), target) or None"""
    X = (np.arange(IW, dtype=np.float32) + 0.5) * RS - W / 2
    Y = H / 2 - (np.arange(IH, dtype=np.float32) + 0.5) * RS
    X, Y = np.meshgrid(X, Y)
    f = cam.f
    dx = cam.F[0] * f + cam.R[0] * X + cam.U[0] * Y
    dy = cam.F[1] * f + cam.R[1] * X + cam.U[1] * Y
    dz = cam.F[2] * f + cam.R[2] * X + cam.U[2] * Y
    nn = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / nn, dy / nn, dz / nn
    ox, oy, oz = cam.pos
    T = np.full((IH, IW), np.inf, np.float32)
    OBJ = np.full((IH, IW), -1, np.int32)
    PL = np.zeros((IH, IW), np.int8)
    # houses, garages, cars
    cand = np.nonzero(np.hypot(H_CENT[:, 0] - ox, H_CENT[:, 1] - oy) < far * 0.6)[0]
    for i in cand:
        h = HOUSES[i]
        r = _rect(cam, h.bbox)
        if r is None:
            continue
        x0, y0, x1, y1 = r
        ddx, ddy, ddz = dx[y0:y1, x0:x1], dy[y0:y1, x0:x1], dz[y0:y1, x0:x1]
        tn = np.full(ddx.shape, -np.inf, np.float32)
        tf = np.full(ddx.shape, np.inf, np.float32)
        pn = np.zeros(ddx.shape, np.int8)
        miss = np.zeros(ddx.shape, bool)
        for k in range(len(h.d)):
            n = h.n[k]
            den = n[0] * ddx + n[1] * ddy + n[2] * ddz
            num = h.d[k] - (n[0] * ox + n[1] * oy + n[2] * oz)
            with np.errstate(divide="ignore", invalid="ignore"):
                tk = num / den
            ent = den < -1e-9
            ext = den > 1e-9
            upd = ent & (tk > tn)
            tn = np.where(upd, tk, tn)
            pn = np.where(upd, k, pn)
            tf = np.where(ext & (tk < tf), tk, tf)
            miss |= (np.abs(den) <= 1e-9) & (num < 0)
        hit = (~miss) & (tn <= tf) & (tn > 0.05)
        sub = T[y0:y1, x0:x1]
        better = hit & (tn < sub)
        sub[better] = tn[better]
        OBJ[y0:y1, x0:x1][better] = i
        PL[y0:y1, x0:x1][better] = pn[better]
    # the Watcher
    WOBJ = -2
    if watcher is not None:
        c = np.array(watcher["pos"], np.float64)
        R = watcher["radius"]
        oc = np.array([ox, oy, oz]) - c
        b = dx * oc[0] + dy * oc[1] + dz * oc[2]
        cc = oc @ oc - R * R
        disc = b * b - cc
        ok = disc > 0
        ts = np.where(ok, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
        better = ok & (ts > 0.1) & (ts < T)
        T = np.where(better, ts, T)
        OBJ = np.where(better, WOBJ, OBJ)
    # ground
    tg = np.where(dz < -1e-4, -oz / np.minimum(dz, -1e-4), np.inf)
    gnd = (tg < T) & (tg < far)
    T = np.where(gnd, tg, T)
    sky = ~np.isfinite(T) | (T > far)
    tt = np.where(sky, far, T)
    px_, py_, pz_ = ox + dx * tt, oy + dy * tt, oz + dz * tt
    # ---- normals and albedo
    NX = np.zeros_like(dx)
    NY = np.zeros_like(dx)
    NZ = np.ones_like(dx)
    alb = np.zeros((IH, IW, 3), np.float32)
    rough = np.ones((IH, IW), np.float32)      # 1 = matte, 0 = mirror-ish (wet asphalt)
    emit = np.zeros((IH, IW, 3), np.float32)
    # ground materials
    ax, ay = street_dist(px_, py_)
    road = gnd & ((ax < ROAD) | (ay < ROAD))
    walk = gnd & ~road & ((ax < WALK) | (ay < WALK))
    lawn = gnd & ~road & ~walk
    n1 = _tex(N_A, px_ * 3.0, py_ * 3.0)
    n2 = _tex(N_C, px_ * 18.0, py_ * 18.0)
    asphalt = s2l((52, 52, 54)) * (0.65 + 0.5 * n2)[..., None] * (0.8 + 0.4 * n1)[..., None]
    alb = np.where(road[..., None], asphalt, alb)
    # sidewalk slabs with joints
    along = np.where(ax < ay, py_, px_)
    joint = (np.abs(((along / 1.6) % 1.0) - 0.5) > 0.47) | (np.abs(np.minimum(ax, ay) - 5.1) > 1.0)
    conc = s2l((128, 126, 120)) * (0.7 + 0.35 * n2)[..., None] * np.where(joint, 0.6, 1.0)[..., None]
    curb = walk & (np.minimum(ax, ay) < ROAD + 0.25)
    conc = np.where(curb[..., None], s2l((150, 148, 142)) * 0.9, conc)
    alb = np.where(walk[..., None], conc, alb)
    grass = s2l((62, 64, 44)) * (0.5 + 0.7 * _tex(N_B, px_ * 6, py_ * 6))[..., None] * (0.7 + 0.5 * n2)[..., None]
    alb = np.where(lawn[..., None], grass, alb)
    # fallen leaves: in the gutters and on lawns
    leaf_d = np.where(road, np.clip(1 - (ROAD - np.minimum(ax, ay)) / 1.2, 0, 1), np.where(lawn, 0.5, 0.15))
    leaves = (_tex(N_A, px_ * 400, py_ * 400) > 0.74 - 0.1 * leaf_d) & gnd
    lc = s2l((84, 66, 46)) * (0.5 + 0.7 * _tex(N_B, px_ * 200, py_ * 200))[..., None]
    alb = np.where(leaves[..., None], lc, alb)
    puddle = road & (_tex(N_A, px_ * 1.3 + 40, py_ * 1.3) > 0.58) & ~leaves
    rough = np.where(road, 0.45, rough)
    rough = np.where(puddle, 0.0, rough)
    # houses
    hobj = OBJ >= 0
    if hobj.any():
        ids = np.where(hobj, OBJ, 0)
        # gather plane normals
        maxp = 8
        NTAB = np.zeros((len(HOUSES), maxp, 3), np.float32)
        for i_, h in enumerate(HOUSES):
            nn_ = h.n / np.linalg.norm(h.n, axis=1, keepdims=True)
            NTAB[i_, :len(nn_)] = nn_
        nv = NTAB[ids, PL.astype(np.int64)]
        NX = np.where(hobj, nv[..., 0], NX)
        NY = np.where(hobj, nv[..., 1], NY)
        NZ = np.where(hobj, nv[..., 2], NZ)
        COL = np.array([s2l(h.col) for h in HOUSES], np.float32)
        ROOFC = np.array([s2l(h.info.get("roof", h.col)) for h in HOUSES], np.float32)
        MAT = np.array([{"house": 0, "garage": 1, "car": 2, "glass": 3, "hedge": 4}[h.mat] for h in HOUSES])
        HW = np.array([h.info.get("hw", 0) for h in HOUSES], np.float32)
        SEED = np.array([h.info.get("seed", 0) for h in HOUSES], np.int64)
        m = MAT[ids]
        is_roof = hobj & (np.abs(NZ) > 0.2) & (NZ > 0)
        wall = hobj & (np.abs(NZ) <= 0.2) & ((m == 0) | (m == 1))
        # along-wall coordinate
        u = np.where(np.abs(NX) > 0.5, py_, px_)
        sd = (SEED[ids] % 1000).astype(np.float32)
        lap = 0.88 + 0.12 * (((pz_ / 0.21) % 1.0) > 0.12)
        wcol = COL[ids] * (lap * (0.8 + 0.3 * _tex(N_C, u * 20 + sd, pz_ * 20)))[..., None]
        # windows: a grid on every wall, by floor
        cell = 3.1
        uu = (u + sd * 0.37) / cell
        fu = uu % 1.0
        floor_i = np.floor(pz_ / 2.8)
        fz = pz_ - floor_i * 2.8
        win = wall & (np.abs(fu - 0.5) < 0.2) & (fz > 0.95) & (fz < 2.25) & (pz_ < HW[ids] - 0.25)
        frame = wall & ~win & (np.abs(fu - 0.5) < 0.24) & (fz > 0.88) & (fz < 2.32) & (pz_ < HW[ids] - 0.2)
        # skip some windows (blank wall), deterministic per cell
        cid = (np.floor(uu).astype(np.int64) * 7919 + floor_i.astype(np.int64) * 104729 + SEED[ids]) % 97
        win &= cid > 20
        frame &= cid > 20
        # front doors, facing the street
        isfront = wall & (FRONTN[ids, 0] * NX + FRONTN[ids, 1] * NY > 0.9)
        door = isfront & (np.abs(u - DOORU[ids]) < 0.55) & (pz_ < 2.15) & (m == 0)
        dframe = isfront & ~door & (np.abs(u - DOORU[ids]) < 0.68) & (pz_ < 2.28) & (m == 0)
        win &= ~(np.abs(u - DOORU[ids]) < 1.2) | ~isfront
        frame &= ~(np.abs(u - DOORU[ids]) < 1.0) | ~isfront
        gdoor = isfront & (m == 1) & (pz_ < 2.2) & (np.abs(fu - 0.5) < 2)
        alb = np.where(wall[..., None], wcol, alb)
        alb = np.where(frame[..., None] | dframe[..., None], s2l((200, 200, 194)) * 0.8, alb)
        alb = np.where(door[..., None], COL[ids] * 0.45 * (0.85 + 0.15 * (np.abs(((pz_ / 0.5) % 1) - 0.5) > 0.4))[..., None], alb)
        gpan = s2l((180, 178, 170)) * (0.8 + 0.2 * (((pz_ / 0.55) % 1.0) > 0.1))[..., None]
        alb = np.where((gdoor & (pz_ < 2.1))[..., None], gpan, alb)
        glass = s2l((14, 16, 20)) * (0.7 + 0.5 * _tex(N_B, u * 4 + sd, pz_ * 4))[..., None]
        alb = np.where(win[..., None], glass, alb)
        rough = np.where(win, 0.05, rough)
        shing = 0.8 + 0.25 * (((pz_ / 0.32) % 1.0) > 0.15) * (0.7 + 0.6 * _tex(N_C, u * 12 + sd, pz_ * 12))
        alb = np.where(is_roof[..., None], ROOFC[ids] * shing[..., None], alb)
        # cars
        car = hobj & ((m == 2) | (m == 3))
        alb = np.where(car[..., None], COL[ids] * 0.9, alb)
        carglass = hobj & (m == 3) & (np.abs(NZ) < 0.95)
        alb = np.where(carglass[..., None], s2l((10, 12, 14)), alb)
        rough = np.where(car, 0.2, rough)
        rough = np.where(carglass, 0.05, rough)
    else:
        win = np.zeros_like(gnd)
    # the Watcher's surface
    weye = OBJ == WOBJ
    if watcher is not None and weye.any():
        c = np.array(watcher["pos"], np.float32)
        nxw, nyw, nzw = (px_ - c[0]) / watcher["radius"], (py_ - c[1]) / watcher["radius"], (pz_ - c[2]) / watcher["radius"]
        NX = np.where(weye, nxw, NX)
        NY = np.where(weye, nyw, NY)
        NZ = np.where(weye, nzw, NZ)
        L = np.array(watcher["look"], np.float32)
        cosl = nxw * L[0] + nyw * L[1] + nzw * L[2]
        # veins: noise ridges on the sclera, thicker towards the back
        vein = np.abs(_tex(N_B, (np.arctan2(nyw, nxw) + 3.2) * 40, (nzw + 1) * 60) - 0.5) < 0.03 * (1.3 - cosl)
        sclera = s2l((206, 196, 186)) * (0.85 + 0.2 * _tex(N_C, nxw * 90 + 7, nzw * 90))[..., None]
        sclera = np.where(vein[..., None], s2l((120, 30, 32)), sclera)
        iris = s2l((70, 80, 70)) * (0.6 + 0.8 * _tex(N_A, np.arctan2(nyw, nxw) * 80, cosl * 40))[..., None]
        ec = np.where((cosl > 0.82)[..., None], iris, sclera)
        ec = np.where((cosl > 0.93)[..., None], 0.0, ec)
        alb = np.where(weye[..., None], ec, alb)
        rough = np.where(weye, 0.1, rough)
        emit = emit + (weye & (cosl > 0.955))[..., None] * np.array([0.6, 0.65, 0.7], np.float32) * watcher.get("beam", 0)
    # ---- lighting
    Vx, Vy, Vz = -dx, -dy, -dz
    col = alb * (0.010 * (0.55 + 0.45 * NZ))[..., None] * np.array([0.6, 0.7, 1.0], np.float32)   # the sky's faint grey
    spec = np.zeros((IH, IW, 3), np.float32)
    # reflection direction
    dn = dx * NX + dy * NY + dz * NZ
    Rx, Ry, Rz = dx - 2 * dn * NX, dy - 2 * dn * NY, dz - 2 * dn * NZ

    def add_point(Lp, Lc, spot=None, sk=1.0):
        nonlocal col, spec
        lx, ly, lz = Lp[0] - px_, Lp[1] - py_, Lp[2] - pz_
        d2 = lx * lx + ly * ly + lz * lz + 0.05
        dl = np.sqrt(d2)
        lx, ly, lz = lx / dl, ly / dl, lz / dl
        lam = np.clip(NX * lx + NY * ly + NZ * lz, 0, 1)
        k = 1.0 / d2
        if spot is not None:
            sd_, c0, c1 = spot
            cs = -(lx * sd_[0] + ly * sd_[1] + lz * sd_[2])
            k = k * np.clip((cs - c0) / (c1 - c0), 0, 1) ** 1.5
        col = col + alb * (lam * k)[..., None] * np.array(Lc, np.float32)
        rl = np.clip(Rx * lx + Ry * ly + Rz * lz, 0, 1)
        sh = np.where(rough < 0.5, 300.0, 40.0)
        sp = rl ** sh * (1.0 - rough) * (8.0 if True else 1.0)
        spec = spec + (sp * k * dl ** 0.8 * 0.25 * sk)[..., None] * np.array(Lc, np.float32)

    if flash > 0:
        fp = cam.pos + cam.R * 0.05 - cam.U * 0.05
        add_point(fp, (3.2 * flash, 3.1 * flash, 2.9 * flash), spot=(cam.F, 0.80, 0.93), sk=0.08)
    for Lp, Lc in (lamps or []):
        if (Lp[0] - ox) ** 2 + (Lp[1] - oy) ** 2 < 120 ** 2:
            add_point(Lp, Lc)
    if watcher is not None and watcher.get("beam", 0) > 0:
        L = np.array(watcher["look"], np.float64)
        pup = np.array(watcher["pos"]) + L * (watcher["radius"] + 0.2)
        add_point(pup, (300 * watcher["beam"], 320 * watcher["beam"], 340 * watcher["beam"]), spot=(L, 0.975, 0.995))
    col = col + spec + emit
    # glass and wet asphalt also reflect the sky's faint glow
    fres = (1 - np.clip(-dn, 0, 1)) ** 4
    col = col + ((1 - rough) * fres * 0.012)[..., None] * np.array([0.5, 0.55, 0.7], np.float32)
    # sky: overcast, black at the top, a little grey at the horizon so roofs and trees read as silhouettes
    hz = np.clip(dz * 4.0, 0, 1)
    skyc = np.array([0.011, 0.012, 0.017], np.float32) * (1 - 0.75 * hz)[..., None]
    skyc = skyc * (0.8 + 0.4 * _tex(N_A, dx / (np.abs(dz) + 0.3) * 160 + t * 0.6, dy / (np.abs(dz) + 0.3) * 160))[..., None]
    col = np.where(sky[..., None], skyc, col)
    # fog
    fa = np.exp(-tt / fog)[..., None]
    col = np.where(sky[..., None], col * (0.3 + 0.7 * fa) + np.array(fogcol, np.float32) * (1 - (0.3 + 0.7 * fa)), col * fa + np.array(fogcol, np.float32) * (1 - fa))
    up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    img = up(col.astype(np.float32))
    depth = cv2.resize(np.where(sky, 1e4, tt).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    pmask = up(puddle.astype(np.float32))
    return img, depth, pmask


def puddle_reflect(img, depth, pmask, cam, t, k=0.75):
    """screen-space mirror for wet patches: sample the frame where the reflected ray ends"""
    ys, xs = np.nonzero(pmask > 0.05)
    if len(xs) == 0:
        return img
    sx = (xs - W / 2) / cam.f
    sy = (H / 2 - ys) / cam.f
    d = cam.F[None] + sx[:, None] * cam.R[None] + sy[:, None] * cam.U[None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    tg = depth[ys, xs][:, None]
    P = cam.pos[None] + d * tg
    Rd = d * np.array([1, 1, -1])
    rip = 0.03 * np.sin(P[:, 0] * 9 + t * 2.0) * np.sin(P[:, 1] * 7 - t * 1.6)
    Rd[:, 0] += rip
    Q = P + Rd * 40.0
    qx, qy, qd = cam.project(Q)
    ok = (qd > 0.5) & (qx >= 0) & (qx < W - 1) & (qy >= 0) & (qy < H - 1)
    refl = np.zeros((len(xs), 3), np.float32)
    refl[ok] = img[qy[ok].astype(int), qx[ok].astype(int)]
    refl[~ok] = np.array([0.012, 0.013, 0.018])
    a = (pmask[ys, xs] * k)[:, None]
    out = img.copy()
    out[ys, xs] = img[ys, xs] * (1 - a) + refl * a
    return out


# ---------------------------------------------------------------------------
# sprites
# ---------------------------------------------------------------------------
def _tree_tex(seed, size=384):
    """a dead deciduous tree as an alpha mask (trunk at the bottom centre)"""
    rng = np.random.default_rng(seed)
    a = np.zeros((size, size), np.float32)

    def branch(x, y, ang, L, w, depth):
        if depth > 7 or L < 3:
            return
        x2 = x + math.cos(ang) * L
        y2 = y - math.sin(ang) * L
        cv2.line(a, (int(x * 4), int(y * 4)), (int(x2 * 4), int(y2 * 4)), 1.0, max(1, int(w)), cv2.LINE_AA, 2)
        nb = 2 if depth > 1 else rng.integers(2, 4)
        for _ in range(nb):
            branch(x2, y2, ang + rng.uniform(-0.75, 0.75), L * rng.uniform(0.62, 0.8), w * 0.68, depth + 1)
    branch(size / 2, size - 2, math.pi / 2 + rng.uniform(-0.08, 0.08), size * 0.3, size * 0.035, 0)
    return a


TREE_TEX = [_tree_tex(s) for s in range(6)]


def _billboard(img, depth, cam, base, height, tex, col):
    q = cam.project(np.array([base, base + np.array([0, 0, height])]))
    if (q[2] < 0.5).any():
        return img
    hpx = q[1][0] - q[1][1]
    if hpx < 4:
        return img
    sz = int(hpx * 1.02)
    x0 = int(q[0][0] - sz / 2)
    y0 = int(q[1][0] - sz)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + sz), min(H, y0 + sz)
    if xb <= xa or yb <= ya:
        return img
    tx = cv2.resize(tex, (sz, sz), interpolation=cv2.INTER_AREA if sz < tex.shape[0] else cv2.INTER_LINEAR)
    tx = tx[ya - y0:yb - y0, xa - x0:xb - x0]
    vis = (depth[ya:yb, xa:xb] > q[2][0] - 0.5).astype(np.float32)
    a = (tx * vis)[..., None]
    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a) + np.array(col, np.float32) * a
    return img


def light_at(p, cam, flash, lamps):
    """rough illumination of a sprite at p (for colouring trees and poles)"""
    rel = p - cam.pos
    d = np.linalg.norm(rel) + 1e-6
    cs = rel @ cam.F / d
    e = flash * 3.2 * np.clip((cs - 0.80) / 0.13, 0, 1) ** 1.5 / (d * d + 0.05)
    out = np.array([e, e * 0.97, e * 0.9])
    for Lp, Lc in (lamps or []):
        d2 = np.sum((np.array(Lp) - p) ** 2) + 0.05
        out = out + np.array(Lc) / d2 * 0.06
    return out + 0.004


def draw_trees(img, depth, cam, flash, lamps, far=150.0):
    cp = cam.pos
    items = [tr for tr in TREES if (tr[0] - cp[0]) ** 2 + (tr[1] - cp[1]) ** 2 < far * far]
    items.sort(key=lambda tr: -((tr[0] - cp[0]) ** 2 + (tr[1] - cp[1]) ** 2))
    for x, y, h, k in items:
        p = np.array([x, y, 0.0])
        L = light_at(p + np.array([0, 0, 2.0]), cam, flash, lamps)
        col = s2l((70, 62, 54)) * L
        img = _billboard(img, depth, cam, p, h, TREE_TEX[k], col)
    return img


def _line3(out, depth, cam, a, b, col, w_m, n=12):
    """a depth-tested 3D line of world width w_m (drawn as short segments)"""
    pts = np.linspace(a, b, n + 1)
    q = cam.project(pts)
    for i in range(n):
        if q[2][i] < 0.3 or q[2][i + 1] < 0.3:
            continue
        xm, ym = int((q[0][i] + q[0][i + 1]) / 2), int((q[1][i] + q[1][i + 1]) / 2)
        if 0 <= xm < W and 0 <= ym < H and depth[ym, xm] < (q[2][i] + q[2][i + 1]) / 2 - 0.3:
            continue
        w = max(1, int(w_m * cam.f / q[2][i]))
        cv2.line(out, (int(q[0][i] * 4), int(q[1][i] * 4)), (int(q[0][i + 1] * 4), int(q[1][i + 1] * 4)), col, w, cv2.LINE_AA, 2)


def lamp_head(L):
    x, y, sx, sy = L
    return np.array([x + sx * 1.6, y + sy * 1.6, 6.6])


def draw_lamps(img, depth, cam, states, flash, lamps_lit):
    """street lamp poles; states: {index: brightness 0..1}"""
    cp = cam.pos
    base8 = to8(img)
    out = base8.copy()
    glow = np.zeros((H, W), np.float32)
    for i, L in enumerate(LAMPS):
        x, y, sx, sy = L
        if (x - cp[0]) ** 2 + (y - cp[1]) ** 2 > 140 ** 2:
            continue
        base = np.array([x, y, 0.0])
        top = np.array([x, y, 6.8])
        head = lamp_head(L)
        lt = light_at(base + [0, 0, 3], cam, flash, lamps_lit)
        c = tuple(float(v) for v in np.clip(l2s(np.clip(s2l((90, 90, 92)) * lt, 0, 1)) * 255, 0, 255))
        _line3(out, depth, cam, base, top, c, 0.14, 6)
        _line3(out, depth, cam, top, head, c, 0.1, 3)
        b = states.get(i, 0.0)
        q = cam.project(head[None])
        if q[2][0] > 0.5 and 0 <= q[0][0] < W and 0 <= q[1][0] < H:
            if depth[int(q[1][0]), int(q[0][0])] > q[2][0] - 0.5:
                r = max(1, int(0.35 * cam.f / q[2][0]))
                cv2.ellipse(out, (int(q[0][0]), int(q[1][0])), (r, max(1, r // 2)), 0, 0, 360, c, -1, cv2.LINE_AA)
                if b > 0:
                    cv2.ellipse(glow, (int(q[0][0]), int(q[1][0]) + r // 3), (r, max(1, r // 2)), 0, 0, 360, b, -1, cv2.LINE_AA)
    img = from8(img, base8, out)
    halo = cv2.GaussianBlur(glow, (0, 0), px(10)) * 2 + cv2.GaussianBlur(glow, (0, 0), px(60)) * 3
    return img + (glow * 3 + halo)[..., None] * np.array([1.0, 0.62, 0.28], np.float32)


def lamp_light(i, b):
    return (tuple(lamp_head(LAMPS[i]) - np.array([0, 0, 0.3])), (60 * b, 36 * b, 15 * b))


# power lines: wooden poles along the east side of the avenue at x = 0
POLES = [(9.0, float(y)) for y in np.arange(-10, 420, 34.0)]


def draw_power(img, depth, cam, flash, lamps_lit):
    cp = cam.pos
    out = to8(img)
    base8 = out.copy()
    near = [p for p in POLES if abs(p[1] - cp[1]) < 160]
    for k, (x, y) in enumerate(near):
        lt = light_at(np.array([x, y, 3.0]), cam, flash, lamps_lit)
        c = tuple(float(v) for v in np.clip(l2s(np.clip(s2l((70, 56, 44)) * lt, 0, 1)) * 255, 0, 255))
        _line3(out, depth, cam, np.array([x, y, 0]), np.array([x, y, 9.5]), c, 0.25, 8)
        _line3(out, depth, cam, np.array([x - 1.3, y, 9.0]), np.array([x + 1.3, y, 9.0]), c, 0.12, 3)
        if k + 1 < len(near):
            x2, y2 = near[k + 1]
            for off in (-1.1, 0.0, 1.1):
                pts = []
                for s in np.linspace(0, 1, 14):
                    pts.append((x + off, y + (y2 - y) * s, 9.0 - 0.9 * math.sin(math.pi * s)))
                pts = np.array(pts)
                q = cam.project(pts)
                for i in range(len(pts) - 1):
                    if q[2][i] > 0.5 and q[2][i + 1] > 0.5:
                        cv2.line(out, (int(q[0][i] * 4), int(q[1][i] * 4)), (int(q[0][i + 1] * 4), int(q[1][i + 1] * 4)), (6, 6, 8), 1, cv2.LINE_AA, 2)
    return from8(img, base8, out)


def quad(img, depth, cam, corners, art, shade=1.0):
    q = cam.project(np.array(corners))
    if (q[2] < 0.3).any():
        return img
    h, w = art.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(np.stack([q[0], q[1]], 1)))
    warped = cv2.warpPerspective(art, M, (W, H))
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H)) * (depth > q[2].min() - 0.4)
    return img * (1 - mask[..., None]) + warped * shade * mask[..., None]


def dust_pile(img, depth, cam, p, flash, lamps_lit):
    """a soft little mound of fine grey dust, like salt or sugar"""
    q = cam.project(np.array([p, p + [0.4, 0, 0], p + [0, 0, 0.16]]))
    if (q[2] < 0.3).any():
        return img
    rx = abs(q[0][1] - q[0][0])
    hgt = q[1][0] - q[1][2]
    x0, y0 = int(q[0][0]), int(q[1][0])
    lt = float(np.minimum(light_at(p, cam, flash, lamps_lit), 0.5).mean())
    g = np.zeros((H, W), np.float32)
    cv2.ellipse(g, (x0, y0), (int(rx), max(1, int(rx * 0.22))), 0, 0, 360, 0.6, -1, cv2.LINE_AA)
    cv2.ellipse(g, (x0, y0 - int(hgt * 0.3)), (int(rx * 0.7), max(1, int(hgt * 0.7))), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
    g = cv2.GaussianBlur(g, (0, 0), max(1.0, rx * 0.08))
    rng = np.random.default_rng(3)
    grains = np.zeros((H, W), np.float32)
    for _ in range(140):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.3, 1.9)
        gx, gy = x0 + math.cos(a) * rx * r, y0 + math.sin(a) * rx * r * 0.22
        cv2.circle(grains, (int(gx), int(gy)), 0, rng.uniform(0.3, 1.0), -1)
    shade = (0.55 + 0.45 * np.clip((y0 - np.arange(H, dtype=np.float32)[:, None]) / max(1.0, hgt), 0, 1))
    col = s2l((118, 118, 116)) * lt
    a_ = np.clip(g + grains * 0.8, 0, 1)[..., None]
    return img * (1 - a_) + (col * shade[..., None]) * a_


def veins(img, depth, cam, w, t):
    """the Watcher's optic nerves trailing behind it, hanging and dragging"""
    base8 = to8(img)
    out = base8.copy()
    c = np.array(w["pos"])
    back = -np.array(w["vel"]) / (np.linalg.norm(w["vel"]) + 1e-9)
    for k in range(7):
        a0 = k * 0.9
        start = c + back * w["radius"] * 0.9 + np.array([0, 0.6 * math.cos(a0), 0.6 * math.sin(a0) - 0.5])
        pts = [start]
        for s in range(1, 16):
            fs = s / 15
            p = start + back * fs * 9 + np.array([0, math.sin(t * 1.3 + k + fs * 4) * 0.6 * fs, -fs * fs * (start[2] - 0.1)])
            pts.append(p)
        pts = np.array(pts)
        q = cam.project(pts)
        for i in range(len(pts) - 1):
            if q[2][i] < 0.5 or q[2][i + 1] < 0.5:
                continue
            xm, ym = int(q[0][i]), int(q[1][i])
            if 0 <= xm < W and 0 <= ym < H and depth[ym, xm] < q[2][i] - 0.3:
                continue
            wd = max(1, int((0.32 - 0.2 * i / 15) * cam.f / q[2][i]))
            cv2.line(out, (int(q[0][i] * 4), int(q[1][i] * 4)), (int(q[0][i + 1] * 4), int(q[1][i + 1] * 4)), (40, 14, 16), wd, cv2.LINE_AA, 2)
    return from8(img, base8, out)


def beam(img, depth, cam, w):
    """the pupil's light as a visible cone in the damp air"""
    if w.get("beam", 0) <= 0:
        return img
    L = np.array(w["look"])
    p0 = np.array(w["pos"]) + L * w["radius"]
    tgt = np.array(w["target"])
    d = np.linalg.norm(tgt - p0)
    side = np.cross(L, [0, 0, 1.0])
    side /= np.linalg.norm(side) + 1e-9
    rr = d * math.tan(math.radians(9))
    q = cam.project(np.array([p0, tgt + side * rr, tgt - side * rr, tgt]))
    if (q[2] < 0.3).any():
        return img
    g = np.zeros((H, W), np.float32)
    tri = np.stack([q[0][:3], q[1][:3]], 1)
    cv2.fillPoly(g, [(tri * 4).astype(np.int32)], 1.0, cv2.LINE_AA, 2)
    g = cv2.GaussianBlur(g, (0, 0), px(18))
    return img + g[..., None] * np.array([0.08, 0.09, 0.1], np.float32) * w["beam"]


def mangled(img, depth, cam, t, base_xy, strength=1.0):
    """an impossibly tall black pillar in the fog, far behind the roofs"""
    bx, by = base_xy
    zs = np.linspace(0, 1400, 40)
    pts = np.array([[bx + 18 * math.sin(z * 0.004 + t * 0.05), by, z] for z in zs])
    q = cam.project(pts)
    if (q[2] < 1).any():
        return img
    m = np.zeros((H, W), np.float32)
    for i in range(len(zs) - 1):
        r = (40 - 14 * zs[i] / 1400) * cam.f / q[2][i]
        cv2.line(m, (int(q[0][i] * 4), int(q[1][i] * 4)), (int(q[0][i + 1] * 4), int(q[1][i + 1] * 4)), 1.0, max(1, int(2 * r)), cv2.LINE_AA, 2)
    # smoke curling off it
    for k in range(9):
        z = 60 + k * 80
        x = bx + 30 * math.sin(k * 1.7 + t * 0.2)
        qq = cam.project(np.array([[x, by, z]]))
        if qq[2][0] > 1:
            cv2.circle(m, (int(qq[0][0]), int(qq[1][0])), int(70 * cam.f / qq[2][0]), 0.6, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), px(6))
    far = (depth > 150).astype(np.float32)
    a = m * far * 0.75 * strength
    return img * (1 - a[..., None]) + np.array([0.003, 0.003, 0.004], np.float32) * a[..., None]

"""
GridWorld: a vectorised per-pixel 3D raycaster for architectural levels.

The world is a 2D grid of square cells (size C metres) with a flat floor at z=0 and
a flat ceiling at z=height. Geometry lives on the grid:

  solid[ix, iy]   filled block (pillar, wall mass)
  wallE[ix, iy]   thin wall on the edge between (ix,iy) and (ix+1,iy)
  wallN[ix, iy]   thin wall on the edge between (ix,iy) and (ix,iy+1)
  hole[ix, iy]    bottomless pit instead of floor

Edge types may contain openings (doorways, arches, windows) — a ray passes through
the edge if it crosses inside the opening's shape. Rays are marched cell by cell
(Amanatides–Woo DDA) for every pixel in numpy, with the set of live rays compacted
each step, so a 1280x720 frame of rooms costs well under a second.

trace() returns per-pixel hit buffers; each episode does its own shading.
"""
import math

import cv2
import numpy as np

OPEN, SOLID, DOOR, ARCH, FLICKER, WINDOW, LOWDOOR, STEP = 0, 1, 2, 3, 4, 5, 6, 9
FAR, WALL, FLOOR, CEIL, PIT, WATER = 0, 1, 2, 3, 4, 5


class GridWorld:
    def __init__(self, nx, ny, cell=1.5, height=2.7):
        self.nx, self.ny, self.C, self.Hc = nx, ny, cell, height
        self.solid = np.zeros((nx, ny), bool)
        self.hole = np.zeros((nx, ny), bool)
        self.wallE = np.zeros((nx, ny), np.uint8)
        self.wallN = np.zeros((nx, ny), np.uint8)
        # opening shapes: half width (m), rectangular height (m), arch radius (m) on top
        self.door_hw, self.door_h = 0.48, 2.1
        self.arch_hw, self.arch_spring = 0.62, 1.55
        self.win_lo, self.win_hi, self.win_hw = 0.95, 2.0, 0.6
        self.fz = self.cz = None
        self.water = None
        self.wl = 0.0

    # ---- building helpers (cell coordinates) -------------------------------
    def box(self, x0, y0, x1, y1, kind=SOLID):
        """walls around the rectangle of cells [x0,x1) x [y0,y1)"""
        self.wallN[x0:x1, y0 - 1] = kind
        self.wallN[x0:x1, y1 - 1] = kind
        self.wallE[x0 - 1, y0:y1] = kind
        self.wallE[x1 - 1, y0:y1] = kind

    def hwall(self, x0, x1, y, kind=SOLID):
        """wall along the north edge of row y, from cell x0 to x1-1"""
        self.wallN[x0:x1, y] = kind

    def vwall(self, x, y0, y1, kind=SOLID):
        """wall along the east edge of column x"""
        self.wallE[x, y0:y1] = kind

    def cell_of(self, x, y):
        return int(math.floor(x / self.C)), int(math.floor(y / self.C))

    # ---- tracing ----------------------------------------------------------
    def _opening(self, e, u, z):
        """True where (u along edge, z) lies inside the opening of edge type e."""
        c = self.C / 2
        du = np.abs(u - c)
        inside = np.zeros(e.shape, bool)
        m = e == DOOR
        inside |= m & (du < self.door_hw) & (z < self.door_h) & (z > 0)
        m = e == LOWDOOR
        inside |= m & (du < self.door_hw) & (z < 1.6) & (z > 0)
        m = e == ARCH
        r = self.arch_hw
        inarch = (du < r) & (z > 0) & ((z < self.arch_spring) | (du ** 2 + (z - self.arch_spring) ** 2 < r * r))
        inside |= m & inarch
        m = e == WINDOW
        inside |= m & (du < self.win_hw) & (z > self.win_lo) & (z < self.win_hi)
        return inside

    def trace(self, cam_pos, dx, dy, dz, tmax=120.0, max_steps=400):
        """Trace rays from one origin (the camera)."""
        return self.trace_rays(cam_pos[0], cam_pos[1], cam_pos[2], dx, dy, dz, tmax, max_steps)

    def trace_rays(self, ox, oy, oz, dx, dy, dz, tmax=120.0, max_steps=400):
        """Trace rays with per-ray (or shared scalar) origins. Returns per-ray hit buffers."""
        shape = np.shape(dx)
        dx, dy, dz = (np.ravel(v).astype(np.float64) for v in (dx, dy, dz))
        n = dx.size
        ox = np.broadcast_to(np.asarray(ox, np.float64).ravel() if np.ndim(ox) else np.float64(ox), (n,)).copy()
        oy = np.broadcast_to(np.asarray(oy, np.float64).ravel() if np.ndim(oy) else np.float64(oy), (n,)).copy()
        oz = np.broadcast_to(np.asarray(oz, np.float64).ravel() if np.ndim(oz) else np.float64(oz), (n,)).copy()
        C, Hc = self.C, self.Hc
        var = self.fz is not None
        wet = self.water is not None
        out_t = np.full(n, tmax, np.float64)
        out_k = np.zeros(n, np.uint8)
        out_axis = np.zeros(n, bool)
        out_side = np.zeros(n, np.int8)
        out_e = np.zeros(n, np.uint8)
        out_cx = np.zeros(n, np.int32)
        out_cy = np.zeros(n, np.int32)
        nx, ny = self.nx, self.ny
        with np.errstate(divide="ignore", invalid="ignore"):
            stepX = np.sign(dx).astype(np.int32)
            stepY = np.sign(dy).astype(np.int32)
            tDX = np.where(stepX != 0, C / np.abs(dx), np.inf)
            tDY = np.where(stepY != 0, C / np.abs(dy), np.inf)
            ix = np.floor(ox / C).astype(np.int32)
            iy = np.floor(oy / C).astype(np.int32)
            tMX = np.where(stepX > 0, ((ix + 1) * C - ox) / dx, np.where(stepX < 0, (ix * C - ox) / dx, np.inf))
            tMY = np.where(stepY > 0, ((iy + 1) * C - oy) / dy, np.where(stepY < 0, (iy * C - oy) / dy, np.inf))
            if not var:
                tplc = np.where(dz < -1e-9, -oz / dz, np.where(dz > 1e-9, (Hc - oz) / dz, np.inf))
        idx = np.arange(n)
        tprev = np.zeros(n)
        Lox, Loy, Loz, Ldz = ox, oy, oz, dz
        for _ in range(max_steps):
            if idx.size == 0:
                break
            ax = tMX <= tMY
            tc = np.where(ax, tMX, tMY)
            ixc, iyc = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
            inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            with np.errstate(divide="ignore", invalid="ignore"):
                if var:
                    fzc = self.fz[ixc, iyc]
                    czc = self.cz[ixc, iyc]
                    tf = np.where(Ldz < -1e-9, (fzc - Loz) / Ldz, np.inf)
                    tcl = np.where(Ldz > 1e-9, (czc - Loz) / Ldz, np.inf)
                else:
                    tf = np.where(Ldz < -1e-9, tplc, np.inf)
                    tcl = np.where(Ldz > 1e-9, tplc, np.inf)
                tw = np.full(idx.size, np.inf)
                if wet:
                    tw0 = (self.wl - Loz) / Ldz
                    okw = self.water[ixc, iyc] & inb & (tw0 > np.maximum(tprev, 1e-5)) & np.isfinite(tw0)
                    tw = np.where(okw, tw0, np.inf)
            tpl = np.minimum(np.minimum(tf, tcl), tw)
            far = (tc > tmax) & (tpl > tmax)
            plane = (tpl <= tc) & ~far
            if plane.any():
                j = np.nonzero(plane)[0]
                gi = idx[j]
                down = Ldz[j] < 0
                isw = tw[j] <= np.minimum(tf[j], tcl[j])
                pit = down & ~isw & inb[j] & self.hole[ixc[j], iyc[j]]
                out_t[gi] = np.where(pit, tc[j], tpl[j])
                out_k[gi] = np.where(isw, WATER, np.where(pit, PIT, np.where(down, FLOOR, CEIL)))
                out_cx[gi], out_cy[gi] = ix[j], iy[j]
                out_axis[gi] = ax[j]
            cross = ~plane & ~far
            nxt_x = ix + np.where(ax, stepX, 0)
            nxt_y = iy + np.where(ax, 0, stepY)
            ex = np.where(ax, np.minimum(ix, nxt_x), ix)
            ey = np.where(ax, iy, np.minimum(iy, nxt_y))
            okE = (ex >= 0) & (ex < nx) & (ey >= 0) & (ey < ny)
            exc, eyc = np.clip(ex, 0, nx - 1), np.clip(ey, 0, ny - 1)
            e = np.where(ax, self.wallE[exc, eyc], self.wallN[exc, eyc])
            e = np.where(okE, e, SOLID)
            hx = Lox + dx[idx] * tc
            hy = Loy + dy[idx] * tc
            hz = Loz + Ldz * tc
            u = np.where(ax, hy - iy * C, hx - ix * C)
            okN = (nxt_x >= 0) & (nxt_x < nx) & (nxt_y >= 0) & (nxt_y < ny)
            nxc, nyc = np.clip(nxt_x, 0, nx - 1), np.clip(nxt_y, 0, ny - 1)
            sol_next = ~okN | self.solid[nxc, nyc]
            passable = (e == OPEN) | self._opening(e, u, hz)
            if var:
                step = (hz < self.fz[nxc, nyc] - 1e-6) | (hz > self.cz[nxc, nyc] + 1e-6)
            else:
                step = np.zeros(idx.size, bool)
            blocked = cross & (~passable | sol_next | step)
            if blocked.any():
                j = np.nonzero(blocked)[0]
                gi = idx[j]
                out_t[gi] = tc[j]
                out_k[gi] = WALL
                out_axis[gi] = ax[j]
                out_side[gi] = np.where(ax[j], stepX[j], stepY[j])
                ee = np.where(sol_next[j] & passable[j], SOLID, e[j])
                out_e[gi] = np.where(step[j] & passable[j] & ~sol_next[j], STEP, ee)
                out_cx[gi], out_cy[gi] = ix[j], iy[j]
            keep = cross & ~blocked
            ix = np.where(ax, nxt_x, ix)
            iy = np.where(ax, iy, nxt_y)
            tprev = tc
            tMX = np.where(ax, tMX + tDX, tMX)
            tMY = np.where(ax, tMY, tMY + tDY)
            idx, ix, iy, tMX, tMY, tprev = idx[keep], ix[keep], iy[keep], tMX[keep], tMY[keep], tprev[keep]
            stepX, stepY, tDX, tDY = stepX[keep], stepY[keep], tDX[keep], tDY[keep]
            Lox, Loy, Loz, Ldz = Lox[keep], Loy[keep], Loz[keep], Ldz[keep]
            if not var:
                tplc = tplc[keep]
        hx = ox + dx * out_t
        hy = oy + dy * out_t
        hz = oz + dz * out_t
        R = dict(t=out_t, kind=out_k, axis=out_axis, side=out_side, edge=out_e, cx=out_cx, cy=out_cy,
                 hx=hx, hy=hy, hz=hz, dx=dx, dy=dy, dz=dz)
        return {k: v.reshape(shape) for k, v in R.items()}

    def heights(self, floor=0.0, ceil=None):
        """enable per-cell floor/ceiling heights"""
        self.fz = np.full((self.nx, self.ny), floor, np.float64)
        self.cz = np.full((self.nx, self.ny), self.Hc if ceil is None else ceil, np.float64)

    def enable_water(self, level=0.0):
        self.water = np.zeros((self.nx, self.ny), bool)
        self.wl = level

    # ---- fields for shading -----------------------------------------------
    def raster(self, res=0.25):
        """Binary raster of walls/solids at `res` metres per pixel (rows = y, cols = x)."""
        k = int(round(self.C / res))
        Wd, Hd = self.nx * k, self.ny * k
        img = np.zeros((Hd, Wd), np.uint8)
        sol = np.repeat(np.repeat(self.solid.T, k, 0), k, 1)
        img[sol] = 1
        ys, xs = np.nonzero(self.wallE.T != OPEN)
        for y, x in zip(ys, xs):
            img[y * k:(y + 1) * k, (x + 1) * k - 1:(x + 1) * k + 1] = 1
        ys, xs = np.nonzero(self.wallN.T != OPEN)
        for y, x in zip(ys, xs):
            img[(y + 1) * k - 1:(y + 1) * k + 1, x * k:(x + 1) * k] = 1
        return img

    def wall_distance(self, res=0.25):
        r = self.raster(res)
        d = cv2.distanceTransform((1 - r).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) * res
        return d


def sample(field, x, y, res, default=0.0):
    """Bilinear sample of a raster field (rows=y, cols=x) at world coords."""
    h, w = field.shape[:2]
    fx = x / res - 0.5
    fy = y / res - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    ax = (fx - x0).astype(np.float32)
    ay = (fy - y0).astype(np.float32)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    v = (field[y0c, x0c] * (1 - ax) * (1 - ay) + field[y0c, x1c] * ax * (1 - ay)
         + field[y1c, x0c] * (1 - ax) * ay + field[y1c, x1c] * ax * ay)
    out = (x0 >= -1) & (x0 < w) & (y0 >= -1) & (y0 < h)
    return np.where(out, v, default)


def tex(t, u, v):
    """Wrap-around nearest lookup into a square texture; u, v in texels."""
    n = t.shape[0]
    return t[(v.astype(np.int64)) % n, (u.astype(np.int64)) % n]

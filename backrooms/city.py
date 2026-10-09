"""
City: a 2.5D heightfield raycaster for endless procedural cities (Level 11 and friends).

The city is a grid of cells (C metres). Each cell is street, park, canal or part of a
building with a height. Rays are marched cell by cell (vectorised DDA); in each cell
they can hit a building's side, its roof, or the ground/water. Rays starting above the
skyline jump straight down to it, so aerial shots are cheap.
"""
import math

import numpy as np

FAR, WALL, ROOF, GROUND, WATER = 0, 1, 2, 3, 4
STREET, PARK, CANAL, BUILD = 0, 1, 2, 3


class City:
    def __init__(self, n=420, cell=12.0, block=8, seed=11):
        rng = np.random.default_rng(seed)
        self.n, self.C, self.block = n, cell, block
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        street = (ii % block == 0) | (jj % block == 0)
        kind = np.where(street, STREET, BUILD).astype(np.int8)
        # canals replace some avenues
        canal = (ii % (block * 5) == 0) & ~(jj % block == 0)
        kind = np.where(canal, CANAL, kind)
        # parks: whole blocks
        bi, bj = ii // block, jj // block
        hb = ((bi * 73856093) ^ (bj * 19349663)) & 0xFFFF
        park = (hb % 17 == 0) & ~street
        kind = np.where(park, PARK, kind)
        # building heights by district: tall downtown, low outskirts, with lots of 2x2 cells
        cx = n / 2
        r = np.hypot(ii - cx, jj - cx) * cell
        lot = ((ii // 2) * 7919 + (jj // 2) * 104729) % 1000 / 1000.0
        lot2 = ((ii // 2) * 15485863 + (jj // 2) * 32452843) % 1000 / 1000.0
        dist = np.clip(1 - r / 2400, 0, 1)
        base = 9 + 30 * lot2
        tall = (lot < 0.12 + 0.5 * dist ** 2) * (40 + 220 * dist ** 1.5 * lot2 + 40 * lot)
        h = np.where(kind == BUILD, np.maximum(base, tall), 0.0)
        self.h = h.astype(np.float64)
        self.kind = kind
        self.mirror = (((ii // 2) * 31 + (jj // 2) * 17) % 3 == 0) & (kind == BUILD)
        self.tint = (((ii // 2) * 13 + (jj // 2) * 7) % 5).astype(np.int8)
        self.hmax = float(h.max()) + 1.0
        self.origin = np.array([-cx * cell, -cx * cell])     # world (0,0) at the city centre

    def cell_of(self, x, y):
        return (np.floor((x - self.origin[0]) / self.C).astype(np.int64),
                np.floor((y - self.origin[1]) / self.C).astype(np.int64))

    def trace(self, cam_pos, dx, dy, dz, tmax=2500.0, max_steps=500):
        shape = dx.shape
        dx, dy, dz = dx.ravel().astype(np.float64), dy.ravel().astype(np.float64), dz.ravel().astype(np.float64)
        n = dx.size
        ox, oy, oz = map(float, cam_pos)
        C = self.C
        out_t = np.full(n, tmax)
        out_k = np.zeros(n, np.uint8)
        out_ax = np.zeros(n, bool)
        out_side = np.zeros(n, np.int8)
        out_i = np.zeros(n, np.int64)
        out_j = np.zeros(n, np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = np.where((oz > self.hmax) & (dz < 0), (oz - self.hmax) / -dz, 0.0)
            up_out = (oz > self.hmax) & (dz >= 0)
            sx = ox + dx * t0 - self.origin[0]
            sy = oy + dy * t0 - self.origin[1]
            ix = np.floor(sx / C).astype(np.int64)
            iy = np.floor(sy / C).astype(np.int64)
            stX = np.sign(dx).astype(np.int64)
            stY = np.sign(dy).astype(np.int64)
            tDX = np.where(stX != 0, C / np.abs(dx), np.inf)
            tDY = np.where(stY != 0, C / np.abs(dy), np.inf)
            tMX = t0 + np.where(stX > 0, ((ix + 1) * C - sx) / dx, np.where(stX < 0, (ix * C - sx) / dx, np.inf))
            tMY = t0 + np.where(stY > 0, ((iy + 1) * C - sy) / dy, np.where(stY < 0, (iy * C - sy) / dy, np.inf))
        tin = t0.copy()
        ax_in = np.zeros(n, bool)
        idx = np.nonzero(~up_out)[0]
        ix, iy, tMX, tMY, tin, ax_in = ix[idx], iy[idx], tMX[idx], tMY[idx], tin[idx], ax_in[idx]
        stX, stY, tDX, tDY = stX[idx], stY[idx], tDX[idx], tDY[idx]
        N = self.n
        for _ in range(max_steps):
            if idx.size == 0:
                break
            ddz = dz[idx]
            ax = tMX <= tMY
            tout = np.minimum(tMX, tMY)
            inb = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
            ixc, iyc = np.clip(ix, 0, N - 1), np.clip(iy, 0, N - 1)
            h = np.where(inb, self.h[ixc, iyc], 0.0)
            z0 = oz + ddz * tin
            z1 = oz + ddz * tout
            wall = (h > 0) & (z0 <= h) & (tin > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                troof = np.where(ddz < 0, (oz - h) / -ddz, np.inf)
            roof = (h > 0) & ~wall & (z1 <= h) & (troof >= tin) & (troof <= tout)
            gnd = (h <= 0) & (ddz < 0) & (z1 <= 0)
            far = (tin > tmax) | ((ddz >= 0) & (z0 > self.hmax))
            done = wall | roof | gnd | far
            if done.any():
                j = np.nonzero(done)[0]
                gi = idx[j]
                kk = np.where(wall[j], WALL, np.where(roof[j], ROOF, np.where(gnd[j], GROUND, FAR)))
                tt = np.where(wall[j], tin[j], np.where(roof[j], troof[j], np.where(gnd[j], oz / np.maximum(-ddz[j], 1e-9), tmax)))
                kind_here = self.kind[ixc[j], iyc[j]]
                kk = np.where((kk == GROUND) & (kind_here == CANAL) & inb[j], WATER, kk)
                out_k[gi] = kk
                out_t[gi] = tt
                out_ax[gi] = ax_in[j]
                out_side[gi] = np.where(ax_in[j], -stX[j], -stY[j])
                out_i[gi], out_j[gi] = ixc[j], iyc[j]
            keep = ~done
            tin = tout
            ax_in = ax
            ix = np.where(ax, ix + stX, ix)
            iy = np.where(ax, iy, iy + stY)
            tMX = np.where(ax, tMX + tDX, tMX)
            tMY = np.where(ax, tMY, tMY + tDY)
            idx, ix, iy, tMX, tMY, tin, ax_in = idx[keep], ix[keep], iy[keep], tMX[keep], tMY[keep], tin[keep], ax_in[keep]
            stX, stY, tDX, tDY = stX[keep], stY[keep], tDX[keep], tDY[keep]
        hx, hy, hz = ox + dx * out_t, oy + dy * out_t, oz + dz * out_t
        R = dict(t=out_t, kind=out_k, axis=out_ax, side=out_side, i=out_i, j=out_j, hx=hx, hy=hy, hz=hz, dx=dx, dy=dy, dz=dz)
        return {k: v.reshape(shape) for k, v in R.items()}

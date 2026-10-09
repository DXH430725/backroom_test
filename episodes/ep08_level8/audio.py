"""
Ep08 — LEVEL 8 "Cave Systems": the sound of being underground.
Surfacing in a black pool; then dripping stone, a thread of running water, wet gravel
underfoot. The meadow chamber hums like glass. A Wrangler passes overhead like weather.
The glow stick cracks; the cave folds back on itself; something walks when she walks.
Then a cable of warm bulbs, a generator, and people.
"""
import math

import numpy as np

from backrooms import sound as snd
from backrooms.sound import (
    SR, Bus, mf, tvec, sos_filter, noise, piano, bell, pad_note, saw_pad, choir_note, reverb_ir, reverb, click,
    thump, whoosh, hum,
)
from . import film as fm
from .film import (
    DURATION, CARD_OPEN, CARD_END, T_POOL, T_TUNNEL, T_CHAMBER, T_WRANGLER, T_LOOP, T_FOLLOW, T_ROAD, T_END,
)

N = int(DURATION * SR)
rng = np.random.default_rng(8)
tt = np.arange(N) / SR


def env(points):
    return snd.env_curve(points, N)


def crunch(amp=0.1, wet=0.5):
    """a boot on wet gravel"""
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    x = sos_filter(noise(n), "bandpass", [400, 3500]) * np.exp(-t * 28)
    grains = np.zeros(n)
    for _ in range(rng.integers(4, 9)):
        i = rng.integers(0, n // 2)
        grains[i:i + 60] += rng.uniform(-1, 1) * np.exp(-np.arange(min(60, n - i)) / 12)
    x = x + sos_filter(grains, "highpass", 1500) * 0.6
    squelch = sos_filter(noise(n), "lowpass", 600) * np.exp(-t * 18) * wet * 0.6
    return (x + squelch) * amp + thump(65, 0.16, amp * 0.6, 22)[:n]


def drip(f=None, amp=0.06):
    f = f or rng.uniform(900, 2200)
    t = tvec(0.25)
    sweep = f * (1 + 2.5 * np.exp(-t * 40))
    return np.sin(2 * math.pi * np.cumsum(sweep) / SR) * np.exp(-t * 30) * amp


def cam_at(t):
    if t < T_POOL + 3.4:
        return None
    for t1, fn in ((T_TUNNEL, fm.cam_pool), (T_CHAMBER, fm.cam_tunnel), (T_WRANGLER, fm.cam_chamber),
                   (T_LOOP, fm.cam_wrangler), (T_FOLLOW, fm.cam_loop), (T_ROAD, fm.cam_follow), (T_END, fm.cam_road)):
        if t < t1:
            return fn(t)
    return None


def step_times():
    """footfalls derived from the camera's actual walk (cuts and jumps ignored)"""
    out = []
    acc, prev = 0.0, None
    for t in np.arange(T_POOL + 3.4, T_END, 1 / 30):
        c = cam_at(t)
        if c is None:
            prev = None
            continue
        p = c.pos[:2].copy()
        if prev is not None:
            d = float(np.linalg.norm(p - prev))
            if d < 1.0:
                acc += d
                if acc > 0.72:
                    acc = 0.0
                    out.append(t)
        prev = p
    return out


def build():
    dry, cave, hall, music = Bus(DURATION), Bus(DURATION), Bus(DURATION), Bus(DURATION)
    hiss = sos_filter(noise(N), "highpass", 2000) * 0.4
    dry.x += np.stack([hiss, np.roll(hiss, 977)]) * env([(0, 0), (0.4, 0.02), (DURATION - 1, 0.02), (DURATION, 0)])
    for lines, t0 in ((CARD_OPEN, 0.0), (CARD_END, T_END)):
        for text, start in lines:
            for k, ch in enumerate(text):
                if ch not in " ·":
                    dry.add(t0 + start + k / 18 + rng.uniform(0, 0.012), click(rng.uniform(0.05, 0.1), 1800), rng.uniform(-0.3, 0.3))

    # ---- under, then up: muffled roar, bubbles, the break of the surface, a gasp
    nu = int((13.4 - T_POOL) * SR)
    roar = sos_filter(noise(nu), "lowpass", 180) * 0.35 * np.linspace(0.6, 1, nu)
    dry.add(T_POOL, roar, 0)
    for _ in range(40):
        b = tvec(0.06)
        f = rng.uniform(250, 700)
        dry.add(T_POOL + rng.uniform(0, 3.3), np.sin(2 * math.pi * f * b * (1 + 6 * b)) * np.exp(-b * 50) * 0.05, rng.uniform(-0.5, 0.5))
    dry.add(13.4, sos_filter(noise(int(1.2 * SR)), "bandpass", [300, 7000]) * np.exp(-np.arange(int(1.2 * SR)) / (0.2 * SR)) * 0.5, 0)
    dry.add(13.55, sos_filter(noise(int(0.7 * SR)), "bandpass", [700, 2600]) * np.hanning(int(0.7 * SR)) * 0.22, 0)
    for k in range(4):        # coughing, then breathing hard
        dry.add(14.6 + k * 0.9, sos_filter(noise(int(0.5 * SR)), "bandpass", [500, 2200]) * np.hanning(int(0.5 * SR)) * (0.12 - k * 0.02), 0)
    slosh = sos_filter(noise(N), "bandpass", [200, 1200]) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.7 * tt) ** 4)
    cave.x += np.stack([slosh, np.roll(slosh, 2000)]) * env([(13.4, 0), (13.6, 0.06), (24, 0.03), (27, 0)])

    # ---- the cave itself: air moving through rock, drips everywhere, a thread of water
    air = sos_filter(noise(N), "lowpass", 110) * (0.7 + 0.3 * np.sin(2 * math.pi * 0.05 * tt))
    air_lvl = env([(13.4, 0), (15, 0.1), (T_CHAMBER, 0.1), (T_CHAMBER + 2, 0.16), (T_LOOP, 0.16), (T_LOOP + 1, 0.1),
                   (T_END - 1, 0.1), (T_END, 0)])
    dry.x += np.stack([air, np.roll(air, 4000)]) * air_lvl
    stream = sos_filter(noise(N), "bandpass", [1200, 5000]) * (0.6 + 0.4 * sos_filter(noise(N), "lowpass", 4) * 8)
    st_lvl = env([(15, 0), (17, 0.018), (T_CHAMBER, 0.018), (T_CHAMBER + 1, 0.008), (T_WRANGLER, 0.008), (T_LOOP, 0.02),
                  (T_ROAD, 0.016), (T_END - 1, 0.01), (T_END, 0)])
    cave.x += np.stack([stream, np.roll(stream, 1500)]) * st_lvl
    dt = 15.0
    while dt < T_END - 1:
        if not (T_WRANGLER + 2 < dt < T_WRANGLER + 15):
            cave.add(dt, drip(amp=rng.uniform(0.02, 0.07)), rng.uniform(-0.9, 0.9))
        dt += rng.exponential(0.9)

    # ---- footsteps: hers, from the camera's own walk
    steps = step_times()
    for k, st in enumerate(steps):
        a = 0.07 if st < T_TUNNEL else 0.09
        if st < T_TUNNEL:
            dry.add(st, sos_filter(noise(int(0.3 * SR)), "bandpass", [300, 2500]) * np.hanning(int(0.3 * SR)) * 0.05, 0)
        dry.add(st + rng.uniform(-0.02, 0.02), crunch(a * rng.uniform(0.8, 1.1), 0.7 if st < T_CHAMBER else 0.4), (-0.12 if k % 2 else 0.12))
        # ...and, in the loop, someone else's, a beat behind
        if T_FOLLOW + 1 < st < T_ROAD - 1 and not (131.0 < st < 138.0):
            hall.add(st + 0.31 + rng.uniform(0, 0.04), crunch(0.05, 0.3) * 0.9, 0.55)

    # ---- the hand, the stapler: a small tick of recognition each
    music.add(30.2, bell(mf(76), 4, 0.025), -0.3)
    music.add(37.6, bell(mf(74), 4, 0.02), 0.3)

    # ---- score: cold and sparse in the tunnels
    for m, a in ((38, 0.12), (45, 0.05), (50, 0.035)):
        music.add(15.5, pad_note(m, T_CHAMBER - 15.5 + 3, a, 0.1, 6, 4), rng.uniform(-0.4, 0.4))
    music.add(40, pad_note(57, 14, 0.02, 0, 5, 5), 0.5)
    # the meadow: glass and breath. a choir under, crystalline arpeggios over
    for m, a in ((41, 0.10), (48, 0.05), (53, 0.04), (57, 0.035), (60, 0.02)):
        music.add(T_CHAMBER - 1, choir_note(m, T_WRANGLER - T_CHAMBER + 2, a * 0.8, 5, 4), rng.uniform(-0.6, 0.6))
    arp = [77, 81, 84, 88, 86, 81, 79, 84]
    at, k = T_CHAMBER + 1.5, 0
    while at < T_WRANGLER - 2:
        music.add(at, bell(mf(arp[k % len(arp)]), 3.5, 0.016), math.sin(k * 1.3) * 0.7)
        at += 0.62 + 0.25 * math.sin(k * 0.7)
        k += 1
    # the floating rocks: a low, wrong interval
    music.add(72.5, pad_note(42, 9, 0.05, 0.2, 2, 4), 0)
    music.add(72.5, pad_note(48, 9, 0.03, 0.2, 2, 4), 0)

    # ---- the Wrangler: weather inside the rock
    nw = int((T_LOOP - T_WRANGLER) * SR)
    tw = np.arange(nw) / SR
    rumble = sos_filter(noise(nw), "lowpass", 50) * 2.0 + sos_filter(noise(nw), "bandpass", [60, 240]) * 0.5
    shape = snd.env_curve([(0, 0), (2.0, 0.15), (4.5, 0.7), (9.0, 1.0), (13, 0.9), (16, 0.25), (19.5, 0.0), (20, 0)], nw)
    dry.add(T_WRANGLER, np.stack([rumble, np.roll(rumble, 3000)]) * shape * 0.6, 0)
    grind = sos_filter(noise(nw), "bandpass", [200, 900]) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.9 * tw) ** 8)
    hall.add(T_WRANGLER + 3, grind[: nw - int(3 * SR)] * shape[int(3 * SR):] * 0.12, 0)
    for _ in range(26):     # grit and pebbles falling from the ceiling
        ts = T_WRANGLER + 2.5 + rng.uniform(0, 12)
        hall.add(ts, click(rng.uniform(0.05, 0.18), rng.uniform(1500, 4000), 0.02), rng.uniform(-0.8, 0.8))
    for ts in (87.5, 90.2, 93.0, 95.6):
        dry.add(ts, thump(30, 1.6, 0.45, 3), 0)
    music.add(84.5, saw_pad(29, 17, 0.12, 180, 4, 5), 0)
    dry.add(93.6, click(0.4, 900, 0.03), 0.2)          # the lamp turned down
    for k in range(10):                                 # her breath, held and let go
        b0 = 94.5 + k * 1.15
        if b0 < 99:
            dry.add(b0, sos_filter(noise(int(0.6 * SR)), "bandpass", [600, 2600]) * np.hanning(int(0.6 * SR)) * 0.03, 0)

    # ---- the loop: the glow stick, and the cave folding back on itself
    dry.add(106.8, click(0.5, 1200, 0.02), 0.1)       # snap
    fz = sos_filter(noise(int(1.6 * SR)), "highpass", 4000) * np.exp(-np.arange(int(1.6 * SR)) / (0.6 * SR)) * 0.04
    dry.add(106.85, fz, 0.1)
    music.add(107.0, bell(mf(81), 6, 0.03), 0.2)
    dry.add(114.5, whoosh(0.5, 2000, 200, 0.12), 0)
    # recognition: the same bell, then a cluster
    music.add(117.2, bell(mf(81), 6, 0.035), 0.2)
    for m in (62, 63, 68, 69):
        music.add(117.4, saw_pad(m, 9, 0.025, 1400, 0.2, 6), rng.uniform(-0.6, 0.6))
    music.add(117.4, thump(36, 2.5, 0.3, 2.5), 0)

    # ---- the follow: heartbeat, the whip-round at nothing, the lamp failing
    hb = T_FOLLOW + 1.0
    while hb < T_ROAD + 2:
        bpm = 70 + 40 * min(1, (hb - T_FOLLOW) / 6)
        dry.add(hb, thump(46, 0.3, 0.13), 0)
        dry.add(hb + 0.22, thump(42, 0.3, 0.08), 0)
        hb += 60 / bpm
    dry.add(131.25, whoosh(0.6, 400, 2500, 0.25), 0)
    music.add(131.3, saw_pad(30, 12, 0.1, 300, 0.05, 6), 0)
    music.add(131.3, pad_note(73, 6, 0.012, 0.6, 0.1, 5), 0.6)
    flick = sos_filter(noise(int((T_ROAD - 136) * SR)), "highpass", 3000) * 0.012
    flick *= (rng.random(len(flick) // 2400 + 1).repeat(2400)[: len(flick)] > 0.7)
    dry.add(136, flick, 0.2)
    music.add(T_FOLLOW, pad_note(35, T_ROAD - T_FOLLOW + 1, 0.1, 0.2, 3, 2), 0)

    # ---- the 9th Road: a generator, mains-hum through the bulbs, and something like warmth
    nr = int((T_END - T_ROAD + 1) * SR)
    gen = hum(nr, 50, 8, 0.6) * 0.04 + sos_filter(noise(nr), "bandpass", [80, 300]) * 0.02
    gen_lvl = snd.env_curve([(0, 0), (4, 0.4), (16, 0.8), (T_END - T_ROAD - 1, 1.0), (T_END - T_ROAD + 1, 0)], nr)
    hall.add(T_ROAD, gen * gen_lvl, -0.3)
    buzz = sos_filter(noise(nr), "bandpass", [3000, 7000]) * 0.003 + hum(nr, 100, 4, 0.5) * 0.006
    dry.add(T_ROAD, buzz * snd.env_curve([(0, 0), (2, 1), (T_END - T_ROAD, 1), (T_END - T_ROAD + 1, 0)], nr), -0.5)
    for chord, t0 in (((45, 52, 57, 61, 64), T_ROAD + 1), ((43, 50, 55, 59, 62), T_ROAD + 9), ((41, 48, 53, 57, 64), T_ROAD + 17),
                      ((45, 52, 57, 61, 66), T_ROAD + 23)):
        for i, m in enumerate(chord):
            music.add(t0, pad_note(m, 10, 0.07 if i == 0 else 0.035, 0.25, 3, 4), (i - 2) * 0.25)
    for k, m in enumerate((69, 73, 76, 81, 76, 73)):
        music.add(T_ROAD + 4 + k * 3.1, piano(m, 4, 0.12), (k % 3 - 1) * 0.3)
    dry.add(T_END - 1.0, noise(int(1.0 * SR)) * np.linspace(0, 0.18, int(1.0 * SR)), 0)

    # ---- end card
    music.add(T_END + 7.4, piano(57, 6, 0.25), 0)
    music.add(T_END + 7.4, piano(64, 6, 0.18), 0)
    music.add(T_END + 7.5, bell(mf(81), 6, 0.03), 0)

    print("reverbs…", flush=True)
    mix = (dry.x + reverb(cave.x, reverb_ir(3.5, 2.6, 5000), 0.6) + reverb(hall.x, reverb_ir(6.0, 4.5, 3000), 0.7)
           + reverb(music.x, reverb_ir(5.0, 3.5, 6000), 0.45))
    # under water: everything low-passed until she breaks the surface
    lpf = sos_filter(mix, "lowpass", 700)
    u = env([(T_POOL, 1), (13.38, 1), (13.42, 0)])
    return mix * (1 - u) + lpf * u * 1.3

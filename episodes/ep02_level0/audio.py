"""
Ep02 — LEVEL 0 "Threshold": score and sound design.
The fluorescent buzz is the main character: it swells, spikes, cuts out in the blackout,
detunes in the red rooms and softens under the arches.
"""
import math

import numpy as np

from backrooms import sound as snd
from backrooms.sound import (
    SR, Bus, mf, tvec, sos_filter, noise, adsr, piano, bell, pad_note, saw_pad, choir_note, vowel,
    reverb_ir, reverb, click, thump, whoosh, hum,
)
from .film import (
    DURATION, CARD_OPEN, CARD_END, T_WAKE, T_MAZE, T_SHIFT, T_PILL, T_HOLE, T_CORR, T_BLACK, T_RED,
    T_ARCH, T_NOCLIP, T_L1, T_END, corr_flicker,
)

N = int(DURATION * SR)
rng = np.random.default_rng(1000)
t_all = np.arange(N) / SR


def env(points):
    return snd.env_curve(points, N)


def footsteps(bus, t0, t1, rate, kind="carpet", amp=1.0, jitter=0.03):
    t = t0
    k = 0
    while t < t1:
        pan = -0.12 if k % 2 else 0.12
        if kind == "carpet":       # wet, muffled
            n = int(0.16 * SR)
            sq = sos_filter(noise(n), "bandpass", [180, 1400]) * np.exp(-np.arange(n) / (0.035 * SR)) * 0.09
            bus.add(t, thump(rng.uniform(60, 80), 0.18, 0.18, 16) + np.pad(sq, (0, max(0, int(0.18 * SR) - n)))[: int(0.18 * SR)], pan, amp)
        elif kind == "splash":
            n = int(0.35 * SR)
            sp = sos_filter(noise(n), "bandpass", [400, 5000]) * np.exp(-np.arange(n) / (0.07 * SR)) * 0.16
            drip = sos_filter(noise(n), "bandpass", [1500, 3500]) * np.exp(-((np.arange(n) / SR - 0.12) / 0.01) ** 2) * 0.1
            bus.add(t, sp + drip + np.pad(thump(70, 0.2, 0.1), (0, n - int(0.2 * SR))), pan, amp)
        elif kind == "concrete":
            bus.add(t, thump(95, 0.12, 0.12, 30), pan, amp)
            bus.add(t, click(0.07, 1200, 0.03), pan, amp)
        t += 1.0 / rate + rng.uniform(-jitter, jitter)
        k += 1


def build():
    dry, room, big = Bus(DURATION), Bus(DURATION), Bus(DURATION)

    # ---- tape hiss
    hiss = sos_filter(noise(N), "highpass", 1800) * 0.5 + sos_filter(noise(N), "bandpass", [200, 1200]) * 0.3
    dry.x += np.stack([hiss, np.roll(hiss, 977)]) * env([(0, 0), (0.4, 0.025), (DURATION - 1, 0.025), (DURATION, 0)])

    # ---- typewriter on the cards
    for lines, t0 in ((CARD_OPEN, 0.0), (CARD_END, T_END)):
        for text, start in lines:
            for k, ch in enumerate(text):
                if ch not in " ·":
                    dry.add(t0 + start + k / 18 + rng.uniform(0, 0.012), click(rng.uniform(0.05, 0.1), 1800), rng.uniform(-0.3, 0.3))

    # ---- THE BUZZ
    flick = np.repeat(rng.random(int(DURATION * 13) + 2), SR // 13)[:N]
    flick = np.pad(flick, (0, max(0, N - len(flick))))
    surge = sos_filter(np.repeat(rng.random(int(DURATION * 0.5) + 2), SR * 2)[:N], "lowpass", 0.4)
    surge = np.pad(surge, (0, max(0, N - len(surge))))
    detune = 1 - 0.06 * env([(T_RED + 2, 0), (T_RED + 9, 1), (T_RED + 13, 1), (T_ARCH - 1, 0)])
    ph = np.cumsum(120 * detune) / SR
    buzz = sum(math.pow(0.6, k) * np.sin(2 * math.pi * (k + 1) * ph) for k in range(7))
    rasp = sos_filter(np.sign(np.sin(2 * math.pi * ph)) * 0.35, "bandpass", [1500, 6000])
    whine = 0.05 * np.sin(2 * math.pi * 9200 * t_all)
    lvl = env([
        (T_WAKE, 0.0), (T_WAKE + 0.6, 0.02), (15.5, 0.05), (T_MAZE, 0.06), (T_SHIFT, 0.065), (56.5, 0.05),
        (58.3, 0.11), (61.4, 0.11), (63.5, 0.06), (T_PILL, 0.07), (T_HOLE, 0.05), (T_CORR, 0.06),
        (T_BLACK, 0.06), (T_BLACK + 0.6, 0.08), (121.8, 0.0), (T_RED - 2, 0.0), (T_RED, 0.05),
        (147, 0.12), (153, 0.08), (T_ARCH, 0.035), (170, 0.035), (T_NOCLIP, 0.06), (T_L1, 0.0),
    ])
    # muffled while waking
    muff = sos_filter(buzz + rasp, "lowpass", 300)
    wake_mix = env([(T_WAKE, 1), (13, 1), (16, 0)])
    corr_f = np.array([corr_flicker(tt) for tt in np.arange(N // 480) / 100.0])
    corr_f = np.repeat(corr_f, 480)[:N]
    corr_f = np.pad(corr_f, (0, N - len(corr_f)), constant_values=1)
    in_corr = env([(T_CORR - 0.01, 0), (T_CORR, 1), (T_BLACK - 0.01, 1), (T_BLACK, 0)])
    blk_flick = env([(T_BLACK, 1), (T_BLACK + 0.6, 1), (121.8, 0)]) * (flick > 0.3 * env([(T_BLACK, 0), (121.8, 1)]))
    gate = np.where(in_corr > 0.5, corr_f, 1.0) * np.where((t_all > T_BLACK) & (t_all < T_RED), blk_flick, 1.0)
    gate = gate * (0.85 + 0.15 * (flick > 0.1))
    buzzsig = 0.6 * ((buzz + rasp * (1 + 2 * surge)) * (1 - wake_mix) + muff * wake_mix + whine) * lvl * (0.7 + 0.6 * surge) * gate
    room.x += np.stack([buzzsig, np.roll(buzzsig, 331)])

    # ---- wake: heartbeat, breath
    hb = 10.6
    while hb < 18.5:
        dry.add(hb, thump(48, 0.3, 0.32), 0)
        dry.add(hb + 0.25, thump(44, 0.3, 0.2), 0)
        hb += 60 / 88
    dry.add(T_WAKE + 0.2, whoosh(2.5, 80, 400, 0.15), 0)

    # ---- footsteps
    footsteps(room, 18.6, T_SHIFT - 0.3, 1.75, "carpet", 0.8)
    footsteps(room, T_SHIFT, 56.0, 2.0, "carpet", 0.8)
    footsteps(room, 67.2, T_PILL - 0.2, 1.8, "carpet", 0.8)
    footsteps(big, T_PILL + 0.2, T_HOLE - 0.2, 1.4, "carpet", 0.7)
    footsteps(room, T_HOLE + 0.3, 96.8, 1.8, "carpet", 0.8)
    footsteps(room, T_CORR + 0.3, 108.8, 1.2, "carpet", 0.7)
    footsteps(room, T_BLACK + 0.6, T_RED - 0.4, 1.5, "splash", 0.9)
    footsteps(room, T_RED + 0.2, 147.0, 1.6, "carpet", 0.8)
    footsteps(room, 148.1, 153.6, 3.4, "carpet", 1.1, 0.02)
    footsteps(big, T_ARCH + 0.2, T_NOCLIP - 0.4, 1.7, "carpet", 0.55)
    footsteps(big, T_L1 + 0.3, T_END - 0.5, 1.5, "concrete", 0.8)

    # ---- the score: a low, sick drone and a detuned piano, sparse
    big.add(T_WAKE + 3, pad_note(33, T_SHIFT - T_WAKE - 2, 0.16, 0.1, 6, 4), 0)           # A1
    big.add(T_MAZE + 2, pad_note(45, T_SHIFT - T_MAZE, 0.05, 0.3, 6, 5), -0.3)
    big.add(T_MAZE + 2, pad_note(52, T_SHIFT - T_MAZE, 0.04, 0.3, 6, 5), 0.3)
    big.add(T_MAZE + 2, pad_note(58, T_SHIFT - T_MAZE, 0.025, 0.3, 6, 5), 0.0)            # tritone
    for k, (tt, m) in enumerate([(23.0, 69), (26.8, 72), (30.2, 68), (35.5, 64), (40.0, 69), (44.8, 63), (49.5, 57)]):
        big.add(tt, piano(m, 4.5, 0.32, rng.uniform(-25, 25)), (-1) ** k * 0.4)
    # title hit
    dry.add(27.6, thump(36, 3.5, 0.45), 0)
    big.add(27.6, bell(mf(57), 7, 0.12), 0)
    # peripheral shift: the world clicks
    for ts in (57.3, 62.5):
        dry.add(ts, thump(30, 1.6, 0.5), 0)
        big.add(ts, whoosh(1.4, 1200, 120, 0.12), 0)
    big.add(58.2, pad_note(46, 6, 0.08, 0.6, 0.3, 3), 0)
    big.add(58.2, pad_note(47, 6, 0.06, 0.6, 0.3, 3), 0)                                   # semitone rub
    # the Pillars: something vast
    for m, a in ((33, 0.16), (45, 0.08), (52, 0.06), (57, 0.05), (64, 0.035), (71, 0.02)):
        big.add(T_PILL - 0.5, saw_pad(m, T_HOLE - T_PILL + 2, a, 900, 5, 4), rng.uniform(-0.5, 0.5))
    for k, m in enumerate((76, 72, 69, 64, 60)):
        big.add(T_PILL + 2 + k * 2.6, piano(m, 6, 0.25), (-1) ** k * 0.5)
    # the Holes: wind rising out of the pit
    nh = int((T_CORR - T_HOLE) * SR)
    pitwind = sos_filter(noise(nh), "lowpass", 220)
    pw = np.linspace(0, 1, nh) ** 2 * 0.5
    big.add(T_HOLE, pitwind * pw, 0)
    big.add(96.5, pad_note(26, 6, 0.3, 0, 2, 2), 0)
    # the hall: scratching inside the walls, whispers
    for k in range(40):
        ts = rng.uniform(T_CORR + 1, T_BLACK - 1)
        n = int(rng.uniform(0.05, 0.25) * SR)
        sc = sos_filter(noise(n), "bandpass", [2000, 6000]) * (rng.random(n) < 0.15) * 0.25
        room.add(ts, sc, rng.choice([-0.9, 0.9]))
    nw = int((T_BLACK - T_CORR) * SR)
    wh = vowel(sos_filter(noise(nw), "highpass", 300), "o") * 0.5
    syl = sos_filter(np.repeat(rng.random(int((T_BLACK - T_CORR) * 6) + 2), SR // 6)[:nw] ** 3, "lowpass", 10)
    wenv = np.interp(np.arange(nw), [0, nw * 0.2, nw * 0.6, nw * 0.7, nw], [0, 0.4, 1, 0.2, 0])
    big.add(T_CORR, np.stack([wh * syl * wenv, np.roll(wh * syl * wenv, 9000)]) * 0.5, 0)
    big.add(T_CORR + 1, pad_note(41, 15, 0.06, 0.2, 4, 3), 0)
    dry.add(113.0, thump(28, 2.0, 0.6), 0)                                                   # the figure is gone
    # blackout: drips and breathing
    for k in range(26):
        room.add(rng.uniform(T_BLACK + 2, T_RED - 1), bell(rng.uniform(1800, 3200), 0.4, 0.04), rng.uniform(-0.8, 0.8))
    dry.add(122.3, click(0.5, 800, 0.02), 0.2)                                               # light switch
    dry.add(122.32, click(0.3, 2000, 0.01), 0.2)
    big.add(T_BLACK + 3, pad_note(28, T_RED - T_BLACK - 2, 0.12, 0, 3, 2), 0)
    # red rooms: cluster rising, heartbeat racing
    for m in (57, 58, 59, 60, 61):
        big.add(T_RED + 2, saw_pad(m, 13.5, 0.025, 1500, 7, 0.3), rng.uniform(-0.6, 0.6))
    hb = 141.0
    while hb < 154:
        a = 0.35 * min(1, (hb - 141) / 4)
        dry.add(hb, thump(50, 0.25, a), 0)
        dry.add(hb + 0.18, thump(46, 0.25, a * 0.6), 0)
        hb += 60 / (90 + 70 * min(1, (hb - 141) / 7))
    dry.add(147.3, whoosh(0.7, 300, 3000, 0.35), 0)
    nb = int(6 * SR)
    breath = sos_filter(noise(nb), "bandpass", [500, 3000]) * (0.5 + 0.5 * np.sin(2 * math.pi * 2.6 * np.arange(nb) / SR)) ** 4 * 0.12
    dry.add(148.0, breath, 0)
    # the Arches: rest
    for m, a in ((41, 0.1), (53, 0.06), (60, 0.05), (64, 0.04), (67, 0.035), (72, 0.025)):     # F add9 colours
        big.add(T_ARCH - 0.5, pad_note(m, T_NOCLIP - T_ARCH + 1, a, 0.25, 4, 1.0), rng.uniform(-0.5, 0.5))
    for k, (tt, m) in enumerate([(158.5, 72), (161.0, 76), (163.5, 79), (166.0, 77), (168.5, 72)]):
        big.add(tt, piano(m, 5, 0.22), (-1) ** k * 0.4)
    # flickering wall: electrical crackle in time with the flicker
    for tt in np.arange(169.5, T_NOCLIP, 1 / 48):
        rate = 2.5 + (14 - 2.5) * min(1, (tt - 169.5) / (T_NOCLIP - 169.5))
        if math.sin(tt * rate * 6.283) > 0.3 and rng.random() < 0.6:
            room.add(tt, sos_filter(noise(int(0.02 * SR)), "bandpass", [800, 6000]) * 0.08 * (0.4 + (tt - 169.5) / 10), 0)
    dry.add(T_NOCLIP - 0.4, whoosh(2.6, 200, 4000, 0.5), 0)
    dry.add(T_NOCLIP, noise(int(2.4 * SR)) * np.linspace(0.3, 0, int(2.4 * SR)), 0)
    # Level 1: concrete, fog, distant machinery
    n1 = int((T_END - T_L1) * SR)
    mach = (hum(n1, 60, 4, 0.5) * 0.04 + sos_filter(noise(n1), "lowpass", 120) * 0.25) * np.minimum(np.arange(n1) / (0.3 * SR), 1)
    big.add(T_L1, mach, 0)
    for k in range(8):
        big.add(rng.uniform(T_L1, T_END - 0.5), bell(rng.uniform(1400, 2400), 0.5, 0.05), rng.uniform(-0.8, 0.8))
    dry.add(T_END - 0.6, noise(int(0.6 * SR)) * np.linspace(0, 0.3, int(0.6 * SR)), 0)
    # end card: one note, then the hum comes back, faintly — you are still there
    big.add(T_END + 11.3, piano(57, 6, 0.4), 0)
    big.add(T_END + 11.3, piano(64, 6, 0.25), 0.2)
    nt = int((DURATION - T_END - 11) * SR)
    tail = hum(nt, 120, 6, 0.6) * np.linspace(0, 0.015, nt)
    dry.add(T_END + 11, tail, 0)

    print("reverbs…", flush=True)
    mix = dry.x + reverb(room.x, reverb_ir(1.1, 0.6, 5000), 0.3) + reverb(big.x, reverb_ir(6.0, 4.5, 4500), 0.55)
    return mix

"""
Ep03 — LEVEL 37 "Sublimity": score and sound design.
Water everywhere; glassy, warm pads; and the level's broken acoustics — sounds that
stop dead, echoes that come back wrong.
"""
import math

import numpy as np

from backrooms import sound as snd
from backrooms.sound import (
    SR, Bus, mf, tvec, sos_filter, noise, piano, bell, pad_note, saw_pad, reverb_ir, reverb, click,
    thump, whoosh, hum,
)
from .film import (
    DURATION, CARD_OPEN, CARD_END, T_UNDER, T_SURF, T_WADE, T_HALL, T_ECHO, T_STAIRS, T_DARK,
    T_TUNNEL, T_FLOAT, T_SINK, T_END,
)

N = int(DURATION * SR)
rng = np.random.default_rng(37)
tt = np.arange(N) / SR


def env(points):
    return snd.env_curve(points, N)


def lapping(n, rate=0.35, bright=1800):
    """gentle water against tile: band noise with slow irregular swells and little clucks"""
    x = sos_filter(noise(n), "bandpass", [200, bright])
    sw = sos_filter(rng.random(n // 4800 + 2).repeat(4800)[:n], "lowpass", rate) if n > 4800 else np.ones(n)
    sw = np.pad(sw, (0, max(0, n - len(sw))))[:n]
    return x * (0.3 + sw)


def slosh(amp=1.0, dur=0.45):
    n = int(dur * SR)
    e = np.exp(-np.arange(n) / (0.12 * SR)) * np.minimum(np.arange(n) / (0.03 * SR), 1)
    return (sos_filter(noise(n), "bandpass", [250, 2500]) * e * 0.35 + thump(60, dur, 0.08)[:n]) * amp


def drip(f=None, amp=0.06):
    f = f or rng.uniform(900, 2600)
    t = tvec(0.25)
    chirp = np.sin(2 * math.pi * (f * t + 3000 * t * t)) * np.exp(-t * 38)
    return chirp * amp


def bubbles(n, density=12):
    out = np.zeros(n)
    for _ in range(int(density * n / SR)):
        i = rng.integers(0, max(1, n - 4000))
        f = rng.uniform(300, 1400)
        t = np.arange(3000) / SR
        out[i:i + 3000] += np.sin(2 * math.pi * f * t * (1 + 6 * t)) * np.exp(-t * 60) * rng.uniform(0.02, 0.07)
    return out


def build():
    dry, room, hall = Bus(DURATION), Bus(DURATION), Bus(DURATION)

    hiss = sos_filter(noise(N), "highpass", 2000) * 0.4
    dry.x += np.stack([hiss, np.roll(hiss, 977)]) * env([(0, 0), (0.4, 0.02), (156, 0.02), (162, 0.006), (T_END, 0.006), (T_END + 0.5, 0.02), (DURATION - 1, 0.02), (DURATION, 0)])

    for lines, t0 in ((CARD_OPEN, 0.0), (CARD_END, T_END)):
        for text, start in lines:
            for k, ch in enumerate(text):
                if ch not in " ·":
                    dry.add(t0 + start + k / 18 + rng.uniform(0, 0.012), click(rng.uniform(0.05, 0.1), 1800), rng.uniform(-0.3, 0.3))

    # ---- underwater opening: muffled pressure, bubbles, a heartbeat; then the surface breaks
    nu = int((T_SURF - T_UNDER) * SR)
    press = sos_filter(noise(nu), "lowpass", 160) * 0.5 + bubbles(nu, 9) * 1.2
    press *= np.minimum(np.arange(nu) / (1.2 * SR), 1)
    dry.add(T_UNDER, press, 0)
    hb = T_UNDER + 1.0
    while hb < T_SURF:
        dry.add(hb, thump(46, 0.3, 0.25), 0)
        dry.add(hb + 0.24, thump(42, 0.3, 0.15), 0)
        hb += 60 / 70
    hall.add(T_UNDER + 2.0, sos_filter(pad_note(62, T_SURF - T_UNDER - 1.8, 0.07, 0.1, 4, 0.5), "lowpass", 700), 0)
    # surface!
    n = int(1.2 * SR)
    splash = sos_filter(noise(n), "bandpass", [400, 9000]) * np.exp(-np.arange(n) / (0.18 * SR)) * 0.6
    dry.add(T_SURF, splash, 0)
    dry.add(T_SURF + 0.05, np.stack([sos_filter(noise(int(0.6 * SR)), "highpass", 1500) * 0.1] * 2), 0)
    gasp = sos_filter(noise(int(0.5 * SR)), "bandpass", [600, 2500]) * np.sin(np.linspace(0, math.pi, int(0.5 * SR))) * 0.12
    dry.add(T_SURF + 0.35, gasp, 0)
    for k in range(20):
        dry.add(T_SURF + 0.3 + rng.exponential(1.4), drip(amp=0.05), rng.uniform(-0.6, 0.6))

    # ---- water ambience everywhere above the surface (with the level's sudden drop-outs)
    lap = lapping(N)
    amb = env([(T_SURF, 0), (T_SURF + 0.05, 0.09), (T_HALL, 0.07), (T_HALL + 3, 0.11), (T_STAIRS, 0.11),
               (T_STAIRS + 0.5, 0.06), (T_DARK, 0.05), (T_DARK + 0.5, 0.03), (T_FLOAT, 0.03), (T_FLOAT + 0.5, 0.09),
               (T_SINK + 2, 0.09), (T_SINK + 4, 0.0)])
    # sound stops dead — twice — and comes back as if nothing happened
    for a, b in ((88.1, 89.7), (101.0, 101.9)):
        amb = amb * ~((tt > a) & (tt < b))
    room.x += np.stack([lap, np.roll(lap, 7919)]) * amb

    # footsteps: wading in the corridor, wet slaps on the walkways
    t = 27.0
    while t < T_HALL - 0.3:
        room.add(t, slosh(0.9), rng.uniform(-0.2, 0.2))
        t += 1 / 1.6
    t = T_HALL + 0.2
    while t < 77.8:
        hall.add(t, thump(90, 0.1, 0.1, 35) + sos_filter(noise(int(0.1 * SR)), "highpass", 2500) * np.exp(-np.arange(int(0.1 * SR)) / 300) * 0.05, (-1) ** int(t * 3) * 0.1)
        t += 1 / 1.6
    t = T_STAIRS + 0.3
    while t < 113.8:
        room.add(t, thump(90, 0.1, 0.09, 35), 0)
        t += 1 / 1.3
    t = T_DARK + 0.3
    while t < T_FLOAT - 0.4:
        room.add(t, slosh(0.7), 0)
        t += 1 / (1.2 if t < T_TUNNEL else 0.9)

    # ---- the score
    # wading: warm glassy chords (Dmaj9 colours)
    for m, a in ((50, 0.06), (57, 0.05), (61, 0.04), (64, 0.04), (66, 0.03), (73, 0.02)):
        hall.add(T_WADE + 2, pad_note(m, T_HALL - T_WADE + 2, a, 0.15, 5, 4), rng.uniform(-0.6, 0.6))
    dry.add(31.0, thump(38, 3.5, 0.35), 0)
    hall.add(31.0, bell(mf(74), 8, 0.12), -0.2)
    hall.add(31.02, bell(mf(81), 8, 0.07), 0.3)
    for k, (tq, m) in enumerate([(35.0, 78), (38.5, 81), (42.0, 76), (46.0, 73), (50.0, 74)]):
        hall.add(tq, bell(mf(m), 5, 0.05), (-1) ** k * 0.5)
    # the hall: vast, slow, Bmaj7 -> Gmaj9, with high glass
    for m, a in ((47, 0.09), (54, 0.06), (59, 0.05), (63, 0.05), (66, 0.04), (70, 0.03), (78, 0.015)):
        hall.add(T_HALL - 0.5, saw_pad(m, 16, a * 0.7, 1400, 6, 4), rng.uniform(-0.7, 0.7))
    for m, a in ((43, 0.09), (50, 0.06), (57, 0.05), (62, 0.05), (66, 0.04), (69, 0.03), (81, 0.015)):
        hall.add(T_HALL + 13, saw_pad(m, T_ECHO - T_HALL - 10, a * 0.7, 1400, 5, 4), rng.uniform(-0.7, 0.7))
    for k in range(12):
        hall.add(T_HALL + 3 + k * 2.3, piano(rng.choice([78, 81, 83, 86, 88]), 5, 0.12, 0, 1.3), rng.uniform(-0.8, 0.8))
    # the echo: the call comes back late, reversed, from the wrong side
    nrev = int(2.2 * SR)
    ghost = sos_filter(noise(nrev), "bandpass", [500, 1800]) * np.linspace(0, 1, nrev) ** 3 * 0.08
    hall.add(91.6, ghost, 0.85)
    hall.add(T_ECHO + 4.5, pad_note(62, 14, 0.05, 0.1, 3, 4), 0)
    hall.add(T_ECHO + 4.5, pad_note(69, 14, 0.04, 0.1, 3, 4), 0.3)
    # stairs: something very deep under the water
    hall.add(T_STAIRS, pad_note(26, T_DARK - T_STAIRS + 1, 0.2, 0, 4, 2), 0)
    hall.add(T_STAIRS + 2, saw_pad(38, 20, 0.05, 300, 6, 3), 0)
    hall.add(T_STAIRS + 6, pad_note(39, 16, 0.05, 0.4, 5, 3), 0.2)      # minor second rub
    dry.add(120.0, thump(30, 2.5, 0.3), 0)
    # dark rooms: no music, just drips that echo too long and a breath of air from the tunnel
    for k in range(18):
        hall.add(rng.uniform(T_DARK + 0.5, T_FLOAT - 1), drip(amp=0.07), rng.uniform(-0.9, 0.9))
    nt = int((T_FLOAT - T_TUNNEL) * SR)
    wind = sos_filter(noise(nt), "lowpass", 260) * np.minimum(np.arange(nt) / (2 * SR), 1) * 0.5
    room.add(T_TUNNEL, wind, 0)
    hall.add(T_DARK + 1, pad_note(33, T_FLOAT - T_DARK - 1, 0.12, 0, 3, 2), 0)
    # floating: the most peaceful thing in the series
    for m, a in ((50, 0.07), (57, 0.06), (61, 0.05), (64, 0.05), (69, 0.04), (73, 0.03), (76, 0.02)):
        hall.add(T_FLOAT - 0.5, pad_note(m, T_END - T_FLOAT - 2, a, 0.1, 6, 6), rng.uniform(-0.6, 0.6))
    for k, (tq, m) in enumerate([(152.0, 81), (155.5, 78), (159.0, 76), (162.5, 73), (165.5, 69)]):
        hall.add(tq, piano(m, 6, 0.18, 0, 1.2), (-1) ** k * 0.4)
    # sinking: everything goes under
    ns = int((T_END - T_SINK - 2) * SR)
    sink = sos_filter(noise(ns), "lowpass", 140) * np.linspace(0, 0.5, ns) + bubbles(ns, 14) * 1.3
    dry.add(T_SINK + 2, sink * np.minimum((ns - np.arange(ns)) / (2 * SR), 1), 0)
    dry.add(T_SINK + 2.0, whoosh(1.0, 2000, 200, 0.2), 0)
    # end card
    hall.add(T_END + 13.5, piano(62, 6, 0.3), 0)
    hall.add(T_END + 13.5, piano(69, 6, 0.2), 0.2)
    hall.add(T_END + 13.5, piano(78, 6, 0.12), -0.2)

    print("reverbs…", flush=True)
    mix = dry.x + reverb(room.x, reverb_ir(1.6, 1.0, 6000), 0.35) + reverb(hall.x, reverb_ir(8.0, 6.5, 5000), 0.6)
    # under the surface: muffle everything that isn't the underwater bed
    under = env([(0, 0), (T_UNDER - 0.01, 0), (T_UNDER, 1), (T_SURF - 0.05, 1), (T_SURF + 0.05, 0), (T_SINK + 1.5, 0), (T_SINK + 3, 1), (DURATION, 1)])
    under = under * env([(0, 1), (T_END - 0.1, 1), (T_END, 0), (DURATION, 0)])
    muff = snd.sos_filter(mix, "lowpass", 500)
    mix = mix * (1 - under) + muff * under
    return mix

"""
Score and sound design for LEVEL 254 — "BLUE HEAVEN", synthesised from scratch.

    python film/audio.py            # -> out/level254_audio.wav
"""
import math
import os
import sys
import wave

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(__file__))
from film import (  # noqa: E402
    OUT, CARD_OPEN, CARD_END, T_HALL, T_NOCLIP, T_VOID, T_ZOOM, T_RING, T_AISLE, T_SEAT, T_RISE,
    T_SKY, T_FALL, T_CITY, T_END, aisle_y,
)
from level254 import DURATION  # noqa: E402

SR = 48000
N = int(DURATION * SR)
rng = np.random.default_rng(254)


def mf(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def env_curve(points, n=N):
    """piecewise-linear envelope over the whole film from [(sec, value), ...]"""
    ts = np.array([p[0] for p in points]) * SR
    vs = np.array([p[1] for p in points])
    return np.interp(np.arange(n), ts, vs)


class Bus:
    def __init__(self):
        self.x = np.zeros((2, N))

    def add(self, start, sig, pan=0.0, gain=1.0):
        i0 = int(start * SR)
        if i0 >= N:
            return
        if sig.ndim == 1:
            l = math.cos((pan + 1) * math.pi / 4)
            r = math.sin((pan + 1) * math.pi / 4)
            sig = np.stack([sig * l, sig * r])
        if i0 < 0:
            sig = sig[:, -i0:]
            i0 = 0
        n = min(sig.shape[1], N - i0)
        self.x[:, i0:i0 + n] += sig[:, :n] * gain


def sos_filter(x, kind, freq, order=2):
    sos = signal.butter(order, freq, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


def noise(n):
    return rng.standard_normal(n)


def adsr(n, a=0.01, r=0.1):
    e = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


# ----------------------------------------------------------------------------
# instruments
# ----------------------------------------------------------------------------
def piano(m, dur=3.0, vel=1.0, detune_cents=0.0):
    f = mf(m) * 2 ** (detune_cents / 1200)
    t = tvec(dur)
    out = np.zeros_like(t)
    B = 0.0004
    for n in range(1, 10):
        fn = f * n * math.sqrt(1 + B * n * n)
        if fn > 12000:
            break
        amp = vel / n ** 1.3
        dec = 0.7 + 0.45 * n
        out += amp * np.sin(2 * math.pi * fn * t + rng.uniform(0, 6.28)) * np.exp(-t * dec)
    att = np.minimum(t / 0.004, 1)
    hammer = noise(len(t)) * np.exp(-t * 90) * 0.08 * vel
    return (out * att + hammer) * adsr(len(t), 0, 0.08)


def bell(f, dur=6.0, amp=1.0):
    t = tvec(dur)
    out = np.zeros_like(t)
    for ratio, a, d in ((1, 1, 0.5), (2.76, 0.5, 0.9), (5.40, 0.25, 1.6), (8.93, 0.12, 2.5), (0.5, 0.35, 0.35)):
        out += a * np.sin(2 * math.pi * f * ratio * t) * np.exp(-t * d)
    return out * np.minimum(t / 0.003, 1) * amp


def saw(freq_t, sr=SR):
    ph = np.cumsum(freq_t / sr)
    return 2 * (ph % 1.0) - 1


def pad_note(m, dur, amp=1.0, bright=0.3, a=4.0, r=5.0):
    t = tvec(dur)
    f = mf(m)
    out = np.zeros_like(t)
    for dc in (-6, 0, 7):
        ff = f * 2 ** (dc / 1200)
        ph = rng.uniform(0, 6.28)
        out += np.sin(2 * math.pi * ff * t + ph) + bright * np.sin(4 * math.pi * ff * t + ph) * 0.5
    lfo = 0.75 + 0.25 * np.sin(2 * math.pi * rng.uniform(0.05, 0.12) * t + rng.uniform(0, 6))
    e = np.minimum(t / a, 1) * np.minimum((dur - t) / r, 1)
    return out * lfo * np.clip(e, 0, 1) ** 2 * amp / 3


def choir_note(m, dur, amp=1.0, a=5.0, r=6.0):
    t = tvec(dur)
    f = mf(m)
    out = np.zeros_like(t)
    for v in range(6):
        cents = rng.uniform(-11, 11)
        vib = 1 + 0.004 * np.sin(2 * math.pi * rng.uniform(4.6, 5.6) * t + rng.uniform(0, 6))
        drift = 1 + 0.002 * np.sin(2 * math.pi * 0.13 * t + v)
        out += saw(f * 2 ** (cents / 1200) * vib * drift)
    e = np.minimum(t / a, 1) * np.minimum((dur - t) / r, 1)
    return out * np.clip(e, 0, 1) ** 1.5 * amp / 6


def vowel_ah(x):
    y = np.zeros_like(x)
    for fc, bw, g in ((750, 130, 1.0), (1150, 150, 0.55), (2600, 250, 0.22), (3300, 300, 0.12)):
        sos = signal.butter(2, [fc - bw, fc + bw], btype="bandpass", fs=SR, output="sos")
        y += g * signal.sosfilt(sos, x, axis=-1)
    return y + 0.15 * sos_filter(x, "lowpass", 400)


def reverb_ir(seconds=4.5, decay=1.6, lp=6000):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = np.zeros((2, n))
    for c in range(2):
        nz = sos_filter(noise(n), "lowpass", lp)
        ir[c] = nz * np.exp(-t * 6.9 / (decay * 2.2))
    ir[:, : int(0.012 * SR)] = 0
    return ir / np.sqrt((ir ** 2).sum(1, keepdims=True))


def reverb(x, ir, wet=0.5):
    y = np.stack([signal.fftconvolve(x[0], ir[0])[:N], signal.fftconvolve(x[1], ir[1])[:N]])
    return x * (1 - wet) + y * wet


def click(amp=1.0, hp=2500, dur=0.012):
    n = int(dur * SR)
    return sos_filter(noise(n), "highpass", hp) * np.exp(-np.arange(n) / (0.002 * SR)) * amp


def thump(f=55, dur=0.35, amp=1.0):
    t = tvec(dur)
    return np.sin(2 * math.pi * f * t * (1 + 0.6 * np.exp(-t * 25))) * np.exp(-t * 11) * amp


def whoosh(dur, f0, f1, amp=1.0):
    n = int(dur * SR)
    x = noise(n)
    out = np.zeros(n)
    seg = 2048
    fs = np.geomspace(f0, f1, n // seg + 1)
    for k in range(0, n, seg):
        sos = signal.butter(2, [fs[k // seg] * 0.7, fs[k // seg] * 1.4], btype="bandpass", fs=SR, output="sos")
        out[k:k + seg] = signal.sosfilt(sos, x[max(0, k - 512):k + seg])[-len(out[k:k + seg]):]
    e = np.sin(np.linspace(0, math.pi, n)) ** 2
    return out * e * amp


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def build():
    dry = Bus()     # direct sound
    hall = Bus()    # small room reverb (Level 287)
    heaven = Bus()  # enormous reverb (Level 254)
    choir = Bus()   # formant-filtered, then heaven

    # --- tape hiss bed
    hiss = sos_filter(noise(N), "highpass", 1800) * 0.5 + sos_filter(noise(N), "bandpass", [200, 1200]) * 0.3
    hiss_env = env_curve([(0, 0.0), (0.4, 0.028), (T_HALL, 0.03), (T_NOCLIP, 0.05), (T_VOID, 0.07), (T_VOID + 2, 0.025),
                          (44, 0.0), (T_ZOOM, 0.0), (T_ZOOM + 0.05, 0.012), (T_AISLE, 0.012), (T_SEAT + 6, 0.012),
                          (131, 0.0), (T_FALL, 0.0), (T_FALL + 0.02, 0.09), (T_CITY, 0.05), (T_END, 0.06),
                          (T_END + 0.5, 0.02), (DURATION - 2, 0.02), (DURATION, 0)])
    dry.x += np.stack([hiss, np.roll(hiss, 977)]) * hiss_env

    # --- typewriter on the archive cards
    for lines, t0, x0 in ((CARD_OPEN, 0.0, 0), (CARD_END, T_END, 0)):
        for text, start in lines:
            for k in range(len(text)):
                if text[k] in " ·":
                    continue
                dry.add(t0 + start + k / 18 + rng.uniform(0, 0.012), click(rng.uniform(0.05, 0.1), 1800), rng.uniform(-0.3, 0.3))

    # --- Level 287: fluorescent hum
    t = np.arange(N) / SR
    hum = sum(math.pow(0.55, k) * np.sin(2 * math.pi * 120 * (k + 1) * t) for k in range(6))
    buzz = sos_filter(np.sign(np.sin(2 * math.pi * 120 * t)) * 0.3, "bandpass", [900, 4000])
    flick = (rng.random(int(DURATION * 11) + 2) > 0.2).astype(float)
    flick = np.repeat(flick, SR // 11)[:N]
    flick = np.pad(flick, (0, max(0, N - len(flick))))
    henv = env_curve([(T_HALL, 0), (T_HALL + 0.3, 0.035), (T_NOCLIP, 0.05), (T_VOID, 0.0)])
    dry.x += (hum * 0.6 + buzz * 0.5) * (0.7 + 0.3 * flick) * henv

    # --- footsteps in the hall
    st = T_HALL + 0.3
    k = 0
    while st < T_NOCLIP:
        if not (21.6 < st < 23.2):
            s = thump(rng.uniform(70, 90), 0.2, 0.22)
            hall.add(st, s, -0.15 if k % 2 else 0.15)
            scuff = sos_filter(noise(int(0.09 * SR)), "bandpass", [300, 2500]) * np.exp(-np.arange(int(0.09 * SR)) / 600) * 0.06
            hall.add(st + 0.01, scuff, -0.15 if k % 2 else 0.15)
        st += 0.59 if st < 21.6 else 0.8
        k += 1

    # --- distant piano: an old waltz that grows louder but never closer
    beat = 60 / 84
    mel = [  # (bar, beat, midi, beats)
        (0, 0, 69, 1), (0, 1, 72, 1), (0, 2, 76, 1),
        (1, 0, 74, 2), (1, 2, 72, 1),
        (2, 0, 71, 1), (2, 1, 68, 1), (2, 2, 64, 1),
        (3, 0, 69, 3),
        (4, 0, 76, 1), (4, 1, 74, 1), (4, 2, 72, 1),
        (5, 0, 71, 2), (5, 2, 69, 1),
        (6, 0, 68, 1), (6, 1, 71, 1), (6, 2, 77, 1),
        (7, 0, 76, 3),
    ]
    harm = [(45, (60, 64)), (50, (65, 69)), (40, (56, 62)), (45, (60, 64)), (45, (60, 64)), (40, (56, 59)), (40, (62, 65)), (45, (60, 64))]
    p_start = T_HALL + 0.8
    pb = Bus()
    for bar, bt, m, ln in mel:
        at = p_start + (bar * 3 + bt) * beat
        pb.add(at + rng.uniform(0, 0.03), piano(m, 3.5, 0.9, rng.uniform(-18, 18)), 0.2)
    for bar, (bass, ch) in enumerate(harm):
        at = p_start + bar * 3 * beat
        pb.add(at, piano(bass, 3.0, 0.8, rng.uniform(-15, 15)), 0.1)
        for b in (1, 2):
            for c in ch:
                pb.add(at + b * beat + rng.uniform(0, 0.02), piano(c, 1.4, 0.4, rng.uniform(-20, 20)), 0.3)
    pbus = sos_filter(pb.x, "lowpass", 1700)
    pbus = sos_filter(pbus, "highpass", 180)
    penv = env_curve([(T_HALL, 0.05), (T_HALL + 3, 0.12), (T_NOCLIP - 1, 0.32), (T_NOCLIP + 0.5, 0.35), (T_VOID, 0)])
    hall.x += pbus * penv

    # --- 1930s voices, murmuring behind the doors
    chat = sos_filter(noise(N), "bandpass", [250, 2800])
    syl = np.repeat(rng.random(int(DURATION * 7) + 2) ** 2, SR // 7)[:N]
    syl = sos_filter(np.pad(syl, (0, max(0, N - len(syl)))), "lowpass", 12)
    cenv = env_curve([(16, 0), (22, 0.05), (T_NOCLIP, 0.09), (T_VOID, 0)])
    hall.x += np.stack([chat * syl, np.roll(chat * syl, 3001)]) * cenv

    # --- no-clip into the wall
    dry.add(T_NOCLIP - 0.8, whoosh(2.4, 200, 3000, 0.5), 0)
    st_n = noise(int(1.8 * SR)) * np.linspace(0, 1, int(1.8 * SR)) ** 2 * 0.25
    dry.add(T_NOCLIP, st_n, 0)
    for k in range(10):
        dry.add(T_VOID - 0.05 + rng.uniform(0, 2.0), sos_filter(noise(int(rng.uniform(0.02, 0.12) * SR)), "bandpass", [500, 6000]) * 0.25, rng.uniform(-0.8, 0.8))
    dry.add(T_VOID, thump(40, 1.5, 0.6), 0)

    # --- Level 254: the pad (D major, add9), always there, always kind
    def pad_chord(t0, t1, notes, amp=0.16, bright=0.25, a=5.0, r=6.0):
        for m in notes:
            heaven.add(t0, pad_note(m, t1 - t0, amp, bright, a, r), rng.uniform(-0.5, 0.5))

    pad_chord(T_VOID + 1.5, T_ZOOM + 3, [50, 57, 62, 66, 69, 76], 0.11, 0.2, 9, 5)
    pad_chord(T_ZOOM - 0.5, T_RING + 2, [50, 57, 64, 69, 71], 0.08, 0.15, 3, 4)       # Dsus2 - unresolved
    pad_chord(T_RING - 0.5, T_AISLE + 2, [47, 54, 62, 66, 69, 74], 0.09, 0.2, 3, 4)   # Bm7
    pad_chord(T_AISLE - 1, T_SEAT + 3, [43, 50, 57, 62, 66, 71], 0.09, 0.3, 4, 3)      # Gmaj7
    pad_chord(T_SEAT, T_RISE + 6, [38, 50, 57, 62, 64], 0.10, 0.15, 6, 6)             # D (sus2), lower
    heaven.add(T_VOID + 1, pad_note(26, T_ZOOM - T_VOID, 0.25, 0, 10, 8), 0)            # sub D

    # title bell + boom
    heaven.add(40.6, whoosh(2.6, 120, 900, 0.1), 0)
    dry.add(43.1, thump(33, 4.0, 0.7), 0)
    heaven.add(43.1, bell(mf(74), 9, 0.2), -0.2)
    heaven.add(43.12, bell(mf(81), 9, 0.12), 0.3)
    heaven.add(77.6, bell(mf(78), 8, 0.12), 0.1)

    # zoom motor
    for a, b in ((60.5, 64.5), (69.5, 73.0)):
        n = int((b - a) * SR)
        tt = np.arange(n) / SR
        motor = sos_filter(noise(n), "bandpass", [1800, 4200]) * 0.02 + 0.006 * np.sin(2 * math.pi * 430 * tt + 3 * np.sin(2 * math.pi * 31 * tt))
        dry.add(a, motor * np.sin(np.linspace(0, math.pi, n)) ** 0.5, 0.3)
    for c in (T_ZOOM, T_RING, T_AISLE):
        dry.add(c - 0.02, click(0.4, 600, 0.03), 0)
        dry.add(c, thump(90, 0.12, 0.2), 0)

    # a high, thin tone once the line appears on the horizon
    heaven.add(T_ZOOM + 5, pad_note(93, 14, 0.03, 0, 4, 4), 0.4)

    # soft footsteps in 254 (no echo: nothing to bounce off)
    st = T_RING + 0.2
    while st < 82:
        dry.add(st, thump(rng.uniform(60, 75), 0.15, 0.08), rng.uniform(-0.1, 0.1))
        st += 0.62

    # the breathing of a hundred thousand people
    br = sos_filter(noise(N), "lowpass", 700) + 0.3 * sos_filter(noise(N), "bandpass", [900, 2400])
    cyc = (0.5 - 0.5 * np.cos(2 * math.pi * t / 6.4)) ** 3
    benv = env_curve([(T_ZOOM + 6, 0), (T_RING, 0.02), (T_RING + 4, 0.07), (T_AISLE, 0.09), (T_SEAT, 0.1), (130, 0.06), (T_RISE + 4, 0.0)])
    heaven.x += np.stack([br, np.roll(br, 12000)]) * cyc * benv

    # time-lapse: one soft tick for every ring you pass, plus the fast-forward whine
    ring_prev = None
    for fi in range(int(T_AISLE * 200), int(T_SEAT * 200)):
        tt = fi / 200
        ring = int((-aisle_y(tt) - 3.2) / 1.7)
        if ring_prev is not None and ring != ring_prev:
            dry.add(tt, click(0.03, 3000, 0.02), 0.6 if ring % 2 else -0.6)
            heaven.add(tt, piano(74 + (ring % 3) * 5 if ring % 26 else 62, 0.8, 0.028), 0.6 if ring % 2 else -0.6)
        ring_prev = ring
    seg = np.arange(int(T_AISLE * SR), int(T_SEAT * SR))
    spd = np.gradient(np.array([aisle_y(s / SR) for s in seg[::480]])) / (480 / SR)
    spd = np.interp(np.arange(len(seg)), np.arange(len(spd)) * 480, spd)
    whine = np.sin(2 * math.pi * np.cumsum(300 + 22 * spd) / SR) * 0.012 * (spd / (spd.max() + 1e-9))
    dry.x[:, seg] += whine

    # heartbeat that slows down, then stops
    hb = 121.0
    bpm = 72
    while hb < 136.5:
        amp = 0.35 * (1 - ramp_py(hb, 131, 136.5))
        dry.add(hb, thump(48, 0.3, amp), 0)
        dry.add(hb + 0.24, thump(44, 0.3, amp * 0.6), 0)
        bpm = 72 - 32 * ramp_py(hb, 121, 136)
        hb += 60 / bpm

    # --- the ascent: choir
    prog = [  # (t0, t1, notes, amp)
        (131.0, 147.0, [50, 57, 62, 66], 0.07),
        (145.0, 153.0, [47, 54, 59, 62, 66, 71], 0.085),
        (151.5, 159.0, [43, 55, 59, 62, 67, 71, 74], 0.095),
        (157.5, 163.5, [45, 57, 61, 64, 69, 73, 76], 0.105),
        (162.0, 178.0, [38, 50, 57, 62, 66, 69, 74, 76, 78, 81], 0.12),
        (172.0, 181.0, [74, 78, 81, 86, 88], 0.05),
    ]
    for t0, t1, notes, amp in prog:
        for m in notes:
            choir.add(t0, choir_note(m, t1 - t0, amp, a=3.5, r=3.5), rng.uniform(-0.7, 0.7))
    for t0, t1, notes, amp in prog[:-1]:
        heaven.add(t0, pad_note(notes[0] - 12, t1 - t0, 0.22, 0, 3, 3), 0)
    for k, (tb, m) in enumerate([(147.0, 83), (152.0, 79), (158.0, 81), (162.2, 86), (166.0, 90), (170.0, 93)]):
        heaven.add(tb, bell(mf(m), 8, 0.08), (-1) ** k * 0.5)
    dry.add(162.2, thump(30, 5.0, 0.6), 0)
    # heaven: pure sines, then the light swallows everything
    for m in (74, 78, 81, 88):
        heaven.add(T_SKY - 1, pad_note(m, T_FALL - T_SKY + 1, 0.05, 0, 2, 0.05), rng.uniform(-0.5, 0.5))
    heaven.add(T_FALL - 2.2, whoosh(2.2, 400, 9000, 0.25), 0)

    # --- fall through the floor
    dry.add(T_FALL, noise(int(0.45 * SR)) * 0.4, 0)
    dry.add(T_FALL + 0.1, whoosh(2.5, 3000, 60, 0.9), 0)
    dry.add(T_FALL + 0.1, thump(28, 2.5, 0.6), 0)
    for k in range(14):
        dry.add(T_FALL + rng.uniform(0, 2.4), sos_filter(noise(int(rng.uniform(0.03, 0.15) * SR)), "bandpass", [300, 7000]) * 0.3, rng.uniform(-1, 1))

    # --- Level 11: rain, traffic, a horn
    rn = T_END - T_CITY
    n = int(rn * SR)
    rain = sos_filter(noise(n), "highpass", 900) * 0.07 + sos_filter(noise(n), "lowpass", 180) * 0.25
    fade = np.minimum(np.arange(n) / (0.05 * SR), 1)
    dry.add(T_CITY, np.stack([rain, np.roll(rain, 5000)]) * fade, 0)
    tt = tvec(0.9)
    horn = (np.sign(np.sin(2 * math.pi * 311 * tt)) + np.sign(np.sin(2 * math.pi * 370 * tt))) * 0.02
    hall.add(T_CITY + 1.2, sos_filter(horn, "lowpass", 1500) * adsr(len(tt), 0.05, 0.2), -0.6)
    dry.add(T_END - 0.6, noise(int(0.6 * SR)) * np.linspace(0, 0.35, int(0.6 * SR)), 0)

    # --- archive card: the lure returns
    heaven.add(T_END + 9.5, pad_note(50, DURATION - T_END - 9.5, 0.09, 0.1, 3, 5), -0.2)
    heaven.add(T_END + 9.5, pad_note(57, DURATION - T_END - 9.5, 0.07, 0.1, 3, 5), 0.2)
    heaven.add(T_END + 9.5, pad_note(66, DURATION - T_END - 9.5, 0.05, 0.1, 3, 5), 0.0)
    heaven.add(T_END + 10.9, piano(69, 6, 0.5), 0)
    heaven.add(T_END + 12.3, piano(74, 6, 0.35), 0.1)

    # --- mix
    print("reverbs…", flush=True)
    small = reverb_ir(1.2, 0.7, 5000)
    big = reverb_ir(7.0, 5.5, 5200)
    ch = vowel_ah(choir.x)
    mix = dry.x + reverb(hall.x, small, 0.45) + reverb(heaven.x + ch * 3.4, big, 0.6)
    # hard cuts: everything stops dead at the no-clips
    for c in (T_FALL,):
        i = int(c * SR)
        ramp_n = int(0.01 * SR)
        g = np.ones(N)
        g[i - ramp_n:i] = np.linspace(1, 0, ramp_n)
        keep = dry.x.copy()
        mix = mix * np.where(np.arange(N) < i, g, 0)
        mix[:, i:] += keep[:, i:]
    # post-fall: re-add Level 11 / card material that lives in hall/heaven buses after the cut
    post = reverb(hall.x * (np.arange(N) >= int(T_FALL * SR)), small, 0.45) + reverb(heaven.x * (np.arange(N) >= int(T_FALL * SR)), big, 0.62)
    mix += post
    mix = sos_filter(mix, "highpass", 22)
    peak = np.abs(mix).max()
    mix = np.tanh(mix / peak * 1.25) / math.tanh(1.25) * 0.93
    fade_out = env_curve([(0, 1), (DURATION - 1.5, 1), (DURATION, 0)])
    return mix * fade_out


def ramp_py(x, a, b):
    return min(max((x - a) / (b - a), 0.0), 1.0)


def main():
    mix = build()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "level254_audio.wav")
    pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(path)


if __name__ == "__main__":
    main()

"""
Ep04 — LEVEL 881 "The Courtyard of Windows": score and sound design.
The canon describes a "deafening, heavy, ghostly silence", broken by gramophone music,
rhythmic banging and creaking. So: near-silence, a climate-control hush, a clock —
and then the gramophone.
"""
import json
import math
import os

import numpy as np

from backrooms import sound as snd
from backrooms.sound import (
    SR, Bus, mf, tvec, sos_filter, noise, piano, bell, pad_note, saw_pad, reverb_ir, reverb, click,
    thump, whoosh, hum, saw,
)
from .film import (
    DURATION, HERE, CARD_OPEN, CARD_END, T_ENTER, T_SURVEY, T_LOG2, T_LOG3, T_MUSIC, T_CORR, T_ADD,
    T_FINAL, T_BOOK, T_BOOK_END, T_FALL, T_END,
)

N = int(DURATION * SR)
rng = np.random.default_rng(881)
tt = np.arange(N) / SR


def env(points):
    return snd.env_curve(points, N)


def foxtrot(dur, bpm=112, seed=1):
    """a little 1920s foxtrot in B-flat: oom-pah bass, chord stabs, a lilting tune"""
    r = np.random.default_rng(seed)
    beat = 60 / bpm
    n = int(dur * SR)
    out = np.zeros(n)
    prog = [(46, (58, 62, 65)), (46, (58, 62, 65)), (51, (58, 63, 67)), (53, (57, 60, 65)),
            (46, (58, 62, 65)), (43, (55, 58, 62)), (48, (60, 63, 67)), (53, (57, 60, 63))]
    tune = [70, 72, 74, 77, 74, 72, 70, 67, 69, 70, 72, 70, 67, 65, 62, 65,
            70, 72, 74, 77, 79, 77, 74, 72, 74, 72, 70, 69, 70, 0, 70, 0]

    def tone(m, d, a, bright=3):
        t = tvec(d)
        f = mf(m)
        x = sum(np.sin(2 * math.pi * f * k * t) / k ** (1.6 - 0.1 * bright) for k in range(1, 7))
        return x * np.exp(-t * 3.0) * np.minimum(t / 0.01, 1) * a

    bar = 0
    t0 = 0.0
    while t0 < dur:
        root, ch = prog[bar % len(prog)]
        for b in range(4):
            tb = t0 + b * beat
            i = int(tb * SR)
            if i >= n:
                break
            if b % 2 == 0:
                s = tone(root - 12 + (7 if b == 2 else 0), beat * 0.9, 0.5)
            else:
                s = sum(tone(c, beat * 0.5, 0.18) for c in ch)
            out[i:i + len(s)] += s[: n - i]
            for half in (0, 1):
                m = tune[(bar % 4) * 8 + b * 2 + half]
                if m:
                    ti = int((tb + half * beat / 2) * SR)
                    s2 = tone(m, beat * 0.6, 0.3, 5) * (1 + 0.3 * r.random())
                    out[ti:ti + len(s2)] += s2[: max(0, n - ti)]
        bar += 1
        t0 += 4 * beat
    return out


def gramophone(x, wow=0.004, warp=None):
    """old shellac record on a horn: narrow band, wow & flutter, crackle, surface hiss"""
    n = len(x)
    t = np.arange(n) / SR
    mod = 1 + wow * np.sin(2 * math.pi * 0.55 * t) + 0.0015 * np.sin(2 * math.pi * 6.5 * t)
    if warp is not None:
        mod = mod * warp
    idx = np.clip(np.cumsum(mod) - 1, 0, n - 1)
    y = np.interp(idx, np.arange(n), x)
    y = sos_filter(y, "bandpass", [350, 3200], 3)
    y = np.tanh(y * 1.8) / 1.8
    crackle = (rng.random(n) < 0.0009) * rng.uniform(-1, 1, n) * 0.6
    crackle = sos_filter(crackle, "highpass", 1500)
    surface = sos_filter(noise(n), "bandpass", [1000, 6000]) * 0.03
    return y + crackle + surface


def build():
    dry, room, big = Bus(DURATION), Bus(DURATION), Bus(DURATION)

    hiss = sos_filter(noise(N), "highpass", 2000) * 0.4
    dry.x += np.stack([hiss, np.roll(hiss, 977)]) * env([(0, 0), (0.4, 0.022), (DURATION - 1, 0.022), (DURATION, 0)])
    for lines, t0 in ((CARD_OPEN, 0.0), (CARD_END, T_END)):
        for text, start in lines:
            for k, ch in enumerate(text):
                if ch not in " ·":
                    dry.add(t0 + start + k / 18 + rng.uniform(0, 0.012), click(rng.uniform(0.05, 0.1), 1800), rng.uniform(-0.3, 0.3))
    dry.add(6.0, thump(32, 3.0, 0.5), 0)                         # the red warning

    # ---- the silence: a cold climate-control hush, very low
    hush = sos_filter(noise(N), "lowpass", 300) * 0.5 + sos_filter(noise(N), "bandpass", [2000, 4000]) * 0.05
    hl = env([(T_ENTER, 0), (T_ENTER + 2, 0.06), (T_CORR, 0.06), (T_CORR + 0.2, 0.0), (T_FINAL, 0), (T_FINAL + 3, 0.05), (T_FALL, 0.05), (T_END, 0)])
    big.x += np.stack([hush, np.roll(hush, 3333)]) * hl
    # the clock: ticking from the table, louder as we approach it
    tick_lvl = env([(T_SURVEY, 0.01), (T_LOG2, 0.02), (64, 0.03), (70, 0.08), (T_LOG3, 0.04), (90, 0.08), (T_CORR, 0)])
    k = 0
    for ts in np.arange(T_SURVEY, T_CORR, 1.0):
        a = tick_lvl[int(ts * SR)]
        room.add(ts, click(a * (1.0 if k % 2 else 0.7), 3000, 0.008) + bell(4200, 0.03, a * 0.3)[: int(0.008 * SR)], 0.4)
        k += 1
    # entering: boots, cloth, the dark room behind
    dry.add(15.2, thump(70, 0.3, 0.2), 0)
    dry.add(15.6, sos_filter(noise(int(0.5 * SR)), "bandpass", [300, 2000]) * np.hanning(int(0.5 * SR)) * 0.1, 0)
    # footsteps on paving
    for a, b, rate in ((16.5, 19.5, 1.6), (T_SURVEY, 50, 0.6), (56, 63, 1.6), (64.6, 69.5, 1.6), (89.2, 95.5, 1.4), (98.4, T_CORR, 3.0)):
        ts = a
        while ts < b:
            room.add(ts, thump(85, 0.12, 0.12, 30) + sos_filter(noise(int(0.12 * SR)), "highpass", 2500) * np.exp(-np.arange(int(0.12 * SR)) / 500) * 0.03, rng.uniform(-0.15, 0.15))
            ts += 1 / rate
    # knocking on the hollow birchwood
    for kt in (58.4, 58.75, 59.05, 61.9, 62.2):
        room.add(kt, thump(140, 0.5, 0.35, 9) + thump(95, 0.5, 0.2, 7), 0.2)
    # creaks in the walls
    for _ in range(9):
        ts = rng.uniform(T_SURVEY, T_CORR)
        n = int(rng.uniform(0.4, 1.2) * SR)
        f0 = rng.uniform(180, 420)
        tq = np.arange(n) / SR
        cr = np.sin(2 * math.pi * np.cumsum(f0 + 40 * np.sin(tq * 7)) / SR) * (rng.random(n) < 0.3) * np.hanning(n) * 0.05
        big.add(ts, sos_filter(cr, "bandpass", [150, 1500]), rng.uniform(-0.9, 0.9))

    # ---- the score: almost nothing, then dread
    big.add(T_ENTER + 6, saw_pad(38, 20, 0.05, 400, 6, 6), 0)
    big.add(20.8, bell(mf(50), 9, 0.15), 0)
    dry.add(20.8, thump(30, 4.0, 0.4), 0)
    for m, a in ((38, 0.06), (45, 0.03), (51, 0.025)):
        big.add(T_SURVEY + 2, saw_pad(m, T_LOG3 - T_SURVEY, a, 500, 8, 4), rng.uniform(-0.5, 0.5))
    big.add(53.0, saw_pad(63, 9, 0.02, 2000, 0.5, 4), 0.4)       # entry window gone: a high, thin sting
    big.add(53.0, saw_pad(64, 9, 0.02, 2000, 0.5, 4), -0.4)
    for m, a in ((37, 0.06), (44, 0.03), (49, 0.02), (50, 0.02)):
        big.add(T_LOG3 - 0.5, saw_pad(m, T_MUSIC - T_LOG3 + 3, a, 600, 4, 3), rng.uniform(-0.5, 0.5))

    # ---- THE GRAMOPHONE
    mlen = DURATION - T_MUSIC
    rec = foxtrot(mlen)
    warp = np.ones(len(rec))
    tq = np.arange(len(rec)) / SR + T_MUSIC
    warp *= 1 - 0.06 * np.clip((tq - T_FINAL) / 15, 0, 1) * (0.5 + 0.5 * np.sin(tq * 0.9))   # slows, sags
    g = gramophone(rec, 0.005, warp)
    glvl = env([(T_MUSIC, 0), (T_MUSIC + 0.3, 0.42), (T_CORR - 0.5, 0.45), (T_CORR, 0.07), (T_ADD, 0.05),
                (T_ADD + 10, 0.08), (T_FINAL, 0.1), (T_FINAL + 3, 0.45), (T_FALL, 0.5), (T_END, 0.0)])
    gb = np.zeros(N)
    gb[int(T_MUSIC * SR):int(T_MUSIC * SR) + len(g)] = g[: N - int(T_MUSIC * SR)]
    gmuff = sos_filter(gb, "lowpass", 900)
    behind = env([(T_CORR - 0.1, 0), (T_CORR, 1), (T_FINAL - 0.1, 1), (T_FINAL + 2.5, 0)])
    gsig = (gb * (1 - behind) + gmuff * behind) * glvl
    big.x += np.stack([gsig, np.roll(gsig, 1500)])
    dry.add(T_MUSIC - 0.05, click(0.4, 600, 0.04), 0)              # the needle drops

    # corridor: harsh light hum, ringing ears, a door slamming
    nc = int((T_ADD - T_CORR) * SR)
    room.add(T_CORR, hum(nc, 100, 5, 0.6) * 0.03 * np.minimum(np.arange(nc) / (0.1 * SR), 1), 0)
    big.add(112.6, pad_note(88, 9.5, 0.02, 0, 3, 0.5), 0)
    big.add(112.6, pad_note(89, 9.5, 0.015, 0, 3, 0.5), 0.3)
    ts = T_CORR + 0.2
    while ts < 114.5:
        room.add(ts, thump(100, 0.1, 0.1, 30), 0)
        ts += 1 / 1.8
    dry.add(121.3, noise(int(0.6 * SR)) * np.linspace(0.5, 0, int(0.6 * SR)), 0)
    dry.add(121.3, thump(45, 1.5, 0.6), 0)
    big.add(122.5, thump(60, 1.2, 0.5, 5), 0.6)                    # a door slams somewhere

    # addendum: under the blanket — breath, heartbeat, banging, the music far off
    na = int((T_FINAL - T_ADD) * SR)
    breath = sos_filter(noise(na), "bandpass", [400, 2500]) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.32 * np.arange(na) / SR)) ** 3 * 0.12
    dry.add(T_ADD, breath, 0)
    hb = T_ADD + 1
    while hb < T_FINAL:
        dry.add(hb, thump(48, 0.3, 0.28), 0)
        dry.add(hb + 0.22, thump(44, 0.3, 0.17), 0)
        hb += 60 / 96
    bang = T_ADD + 6
    while bang < T_FINAL - 1:
        big.add(bang, thump(70, 0.6, 0.3, 6) + thump(140, 0.6, 0.15, 12), rng.uniform(-0.6, 0.6))
        bang += 1.2 if bang < 140 else 0.8
    for th in (136.7, 142.4, 147.0):
        dry.add(th - 0.05, noise(int(0.25 * SR)) * 0.4, 0)
    # the final: windows everywhere, the record sagging, then falling through the floor
    for m, a in ((26, 0.25), (38, 0.08), (39, 0.06)):
        big.add(T_FINAL, pad_note(m, T_FALL - T_FINAL + 2, a, 0.2, 3, 2), 0)
    dry.add(T_BOOK, sos_filter(noise(int(0.4 * SR)), "bandpass", [800, 4000]) * np.hanning(int(0.4 * SR)) * 0.15, 0.2)   # page turn
    dry.add(T_FALL - 0.2, whoosh(4.0, 3000, 50, 0.7), 0)
    dry.add(T_FALL + 0.5, thump(24, 4.5, 0.7), 0)
    # end card: the record runs out — crackle in the run-out groove
    nr = int((DURATION - T_END - 13) * SR)
    runout = sos_filter((rng.random(nr) < 0.002) * rng.uniform(-1, 1, nr), "highpass", 1500) * 0.4
    tr = np.arange(nr) / SR
    runout += np.sin(2 * math.pi * 1.8 * tr) ** 30 * sos_filter(noise(nr), "bandpass", [200, 800]) * 0.15
    dry.add(T_END + 13, runout, 0)

    # radio squelch at each radio line
    spec = json.load(open(os.path.join(HERE, "voice", "lines.json"), encoding="utf-8"))
    for ln in spec["lines"]:
        sp = spec["speakers"].get(ln["speaker"], {})
        if ln.get("fx", sp.get("fx")) in ("radio", "radio_far"):
            sq = sos_filter(noise(int(0.12 * SR)), "bandpass", [800, 4000]) * np.hanning(int(0.12 * SR)) * 0.18
            dry.add(ln["start"] - 0.15, sq, sp.get("pan", 0))
            dry.add(ln["start"] + ln["max_dur"] - 0.1, sq * 0.7, sp.get("pan", 0))

    print("reverbs…", flush=True)
    mix = dry.x + reverb(room.x, reverb_ir(3.5, 3.0, 5000), 0.4) + reverb(big.x, reverb_ir(6.5, 5.5, 4500), 0.5)
    # under the blanket: everything outside is muffled
    under = env([(T_ADD - 0.01, 0), (T_ADD, 1), (T_FINAL, 1), (T_FINAL + 2.5, 0)])
    muff = snd.sos_filter(mix, "lowpass", 700)
    keep_dry = dry.x * under * 0.6
    mix = mix * (1 - under) + muff * under + keep_dry
    return mix

"""
Ep05 — LEVEL 10 "Bumper Crop": sound of a family camcorder, and a quiet score.
Birds sing at the picnic — and never again: Level 10 has no birds. Only wind in the
wheat, the dog, a hammer, and later the smoke.
"""
import math

import numpy as np

from backrooms import sound as snd
from backrooms.sound import (
    SR, Bus, mf, tvec, sos_filter, noise, piano, bell, pad_note, saw_pad, reverb_ir, reverb, click,
    thump, whoosh, vowel, saw,
)
from .film import (
    DURATION, CARD_OPEN, CARD_END, T_PICNIC, T_NOCLIP, T_MONT, T_ARRIVE, T_ROAD, T_BARN, T_INSIDE,
    T_STEP, T_RAIN, T_NIGHT, T_RUN, T_LAKE, T_END,
)

N = int(DURATION * SR)
rng = np.random.default_rng(10)
tt = np.arange(N) / SR


def env(points):
    return snd.env_curve(points, N)


def bark(f0=520, amp=0.3):
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    f = f0 * (1 - 0.35 * t / 0.16)
    x = saw(f) * np.exp(-t * 18) * np.minimum(t / 0.004, 1)
    x = vowel(x, "a") * 2 + sos_filter(noise(n), "bandpass", [600, 3000]) * np.exp(-t * 30) * 0.3
    return x * amp


def chirp(amp=0.05):
    n = int(rng.uniform(0.05, 0.12) * SR)
    t = np.arange(n) / SR
    f = rng.uniform(2800, 5200) * (1 + rng.uniform(-0.3, 0.3) * t / t[-1])
    return np.sin(2 * math.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) * amp


def pant(n, rate=3.2, amp=0.05):
    t = np.arange(n) / SR
    return sos_filter(noise(n), "bandpass", [700, 3000]) * (0.5 + 0.5 * np.sin(2 * math.pi * rate * t)) ** 4 * amp


def build():
    dry, room, big = Bus(DURATION), Bus(DURATION), Bus(DURATION)
    # tape hiss & the hum of a cheap camcorder motor
    hiss = sos_filter(noise(N), "highpass", 1800) * 0.4
    motor = np.sin(2 * math.pi * 59.7 * tt) * 0.4 + np.sin(2 * math.pi * 119.4 * tt) * 0.2
    play = env([(T_PICNIC - 0.01, 0), (T_PICNIC, 1), (T_END - 0.6, 1), (T_END - 0.5, 0)])
    dry.x += np.stack([hiss, np.roll(hiss, 977)]) * (0.02 + 0.012 * play) * env([(0, 0), (0.4, 1), (DURATION - 1, 1), (DURATION, 0)])
    dry.x += np.stack([motor, motor]) * 0.004 * play
    for lines, t0 in ((CARD_OPEN, 0.0), (CARD_END, T_END)):
        for text, start in lines:
            for k, ch in enumerate(text):
                if ch not in " ·":
                    dry.add(t0 + start + k / 18 + rng.uniform(0, 0.012), click(rng.uniform(0.05, 0.1), 1800), rng.uniform(-0.3, 0.3))

    # ---- picnic: birdsong, a summer breeze, the dog
    for _ in range(70):
        ts = rng.uniform(T_PICNIC + 0.2, T_NOCLIP)
        for k in range(rng.integers(2, 6)):
            big.add(ts + k * rng.uniform(0.09, 0.16), chirp(rng.uniform(0.02, 0.06)), rng.uniform(-0.9, 0.9))
    breeze = sos_filter(noise(N), "lowpass", 500) * env([(T_PICNIC, 0), (T_PICNIC + 0.5, 0.12), (T_NOCLIP, 0.12), (T_NOCLIP + 0.1, 0)])
    dry.x += np.stack([breeze, np.roll(breeze, 4000)])
    for bt in (12.0, 12.3, 16.0, 18.9, 19.2):
        room.add(bt, bark(rng.uniform(480, 560), 0.25), 0.5)
    # the no-clip: the world tears, then the montage of bad levels
    dry.add(T_NOCLIP - 0.2, whoosh(2.0, 4000, 80, 0.6), 0)
    dry.add(T_NOCLIP, noise(int(1.5 * SR)) * np.linspace(0.45, 0, int(1.5 * SR)), 0)
    dry.add(T_NOCLIP + 0.1, thump(30, 3.0, 0.6), 0)
    for k, ts in enumerate((T_MONT, T_MONT + 2.2, T_MONT + 4.2, T_MONT + 6.2, T_MONT + 8.2)):
        dry.add(ts, noise(int(0.25 * SR)) * 0.25, 0)
        big.add(ts, pad_note([45, 44, 41, 40, 39][k], 2.4, 0.1, 0.3, 0.1, 0.5), 0)
    big.add(T_MONT + 6.2, sos_filter(noise(int(2 * SR)), "lowpass", 150) * 0.4, 0)            # Level 6: dark

    # ---- Level 10: wind in the wheat (gusty), and the camcorder mic catching it
    gusts = sos_filter(np.repeat(rng.random(int(DURATION * 0.6) + 2), int(SR / 0.6))[:N], "lowpass", 0.3)
    gusts = np.pad(gusts, (0, max(0, N - len(gusts))))[:N]
    rustle = sos_filter(noise(N), "bandpass", [1500, 7000]) * (0.4 + gusts) * 0.6
    windlow = sos_filter(noise(N), "lowpass", 220) * (0.3 + gusts)
    buffet = sos_filter(noise(N), "lowpass", 90) * np.clip(gusts - 0.45, 0, 1) * 2.5          # mic buffeting
    outdoors = env([(T_ARRIVE, 0), (T_ARRIVE + 2.5, 1), (T_INSIDE - 0.3, 1), (T_INSIDE, 0), (T_RAIN - 0.01, 0), (T_RAIN, 1.4),
                    (T_NIGHT - 0.3, 1.4), (T_NIGHT, 0), (T_RUN - 0.01, 0), (T_RUN, 0.9), (T_LAKE, 0.9), (T_LAKE + 0.5, 0.7), (T_END - 0.6, 0.7), (T_END - 0.5, 0)])
    w = (rustle * 0.09 + windlow * 0.07 + buffet * 0.05) * outdoors
    dry.x += np.stack([w, np.roll(w, 2500)])
    # rain
    rn = sos_filter(noise(N), "highpass", 1200) * env([(T_RAIN, 0), (T_RAIN + 1, 0.06), (T_NIGHT - 0.3, 0.06), (T_NIGHT, 0)])
    for _ in range(600):
        i = rng.integers(int(T_RAIN * SR), int((T_NIGHT - 0.5) * SR))
        rn[i:i + 200] += np.exp(-np.arange(200) / 30) * rng.uniform(0.02, 0.08)
    dry.x += np.stack([rn, np.roll(rn, 1111)])
    # footsteps on dirt
    for a, b, rate, amp in ((39.5, T_ROAD, 1.4, 0.5), (T_ROAD, T_BARN, 1.6, 0.6), (T_BARN, T_INSIDE - 0.3, 1.7, 0.6),
                            (T_RUN, T_LAKE - 0.3, 3.6, 0.9)):
        ts = a
        while ts < b:
            dry.add(ts, thump(75, 0.15, 0.1 * amp, 24) + sos_filter(noise(int(0.15 * SR)), "bandpass", [300, 2500]) * np.exp(-np.arange(int(0.15 * SR)) / 900) * 0.04 * amp, rng.uniform(-0.1, 0.1))
            ts += 1 / rate
    # the dog
    for bt in (41.5, 41.8, 42.1, 58.0, 70.3, 70.6, 87.5, 137.0, 137.3):
        room.add(bt, bark(rng.uniform(470, 560), 0.22), rng.uniform(-0.6, 0.6))
    dry.add(T_RUN, pant(int((T_LAKE - T_RUN) * SR), 3.8, 0.05), -0.2)
    dry.add(T_LAKE + 4, pant(int(8 * SR), 2.2, 0.025), 0.3)
    # barn: hammering, the stove, a creaky beam
    for k in range(18):
        room.add(92.0 + k * 0.42 + (0.6 if k > 8 else 0), thump(190, 0.12, 0.3, 30), 0.4)
    stove = sos_filter((rng.random(int(12 * SR)) < 0.004) * rng.uniform(-1, 1, int(12 * SR)), "bandpass", [800, 5000]) * 0.4
    room.add(T_INSIDE, stove, -0.4)

    # ---- the score: a small, kind melody that keeps coming back
    tune = [(0, 67, 1.0), (1, 69, 0.5), (1.5, 71, 0.5), (2, 74, 1.5), (3.5, 72, 0.5), (4, 71, 1.0), (5, 69, 1.0), (6, 67, 2.0),
            (8, 64, 1.0), (9, 67, 1.0), (10, 69, 1.5), (11.5, 71, 0.5), (12, 69, 1.0), (13, 67, 3.0)]

    def play_tune(t0, beat, vel=0.3, octave=0, inst="piano"):
        for b, m, d in tune:
            if inst == "piano":
                big.add(t0 + b * beat, piano(m + octave, d * beat * 2.5 + 1, vel, rng.uniform(-6, 6)), rng.uniform(-0.3, 0.3))
            else:
                big.add(t0 + b * beat, bell(mf(m + octave + 12), 2.5, vel * 0.5), rng.uniform(-0.3, 0.3))

    # arrival: the tune, full and hopeful
    for m, a in ((43, 0.08), (50, 0.06), (55, 0.05), (59, 0.04), (62, 0.03)):
        big.add(T_ARRIVE + 1, pad_note(m, T_ROAD - T_ARRIVE + 6, a, 0.2, 4, 5), rng.uniform(-0.6, 0.6))
    play_tune(T_ARRIVE + 3.0, 0.62, 0.32)
    dry.add(46.0, thump(40, 3, 0.3), 0)
    big.add(46.0, bell(mf(67), 7, 0.1), 0)
    # the road: lighter
    for m, a in ((48, 0.06), (55, 0.05), (60, 0.04), (64, 0.03)):
        big.add(T_ROAD + 4, pad_note(m, 22, a, 0.2, 5, 5), rng.uniform(-0.6, 0.6))
    play_tune(T_ROAD + 8, 0.7, 0.2, 12)
    # the doorstep: music box
    play_tune(T_STEP + 1, 0.5, 0.25, 0, "box")
    big.add(117.5, pad_note(38, 7, 0.12, 0.3, 1, 3), 0)                                     # the worm
    big.add(117.5, pad_note(39, 7, 0.06, 0.3, 1, 3), 0)
    # rain: thinner, minor
    for m, a in ((40, 0.06), (52, 0.04), (55, 0.04), (59, 0.03)):
        big.add(T_RAIN + 1, pad_note(m, T_NIGHT - T_RAIN, a, 0.15, 5, 4), rng.uniform(-0.6, 0.6))
    big.add(134.5, pad_note(31, 12, 0.18, 0, 3, 4), 0)                                      # the harvest note
    # the night: smoke, a cough, the heartbeat; then running
    nn = int((T_RUN - T_NIGHT) * SR)
    crack = sos_filter((rng.random(nn) < 0.006) * rng.uniform(-1, 1, nn), "bandpass", [600, 4000]) * 0.35
    room.add(T_NIGHT, crack, 0.3)
    for ct in (148.2, 148.7, 152.6):
        n = int(0.3 * SR)
        cough = sos_filter(noise(n), "bandpass", [400, 2500]) * np.exp(-np.arange(n) / (0.06 * SR)) * 0.35
        dry.add(ct, cough, 0)
    hb = 150.5
    while hb < T_RUN + 5:
        dry.add(hb, thump(46, 0.3, 0.3), 0)
        dry.add(hb + 0.22, thump(42, 0.3, 0.18), 0)
        hb += 60 / (80 + 40 * min(1, (hb - 150) / 10))
    big.add(153.8, pad_note(50, 8, 0.07, 0.1, 2, 4), 0)
    big.add(153.8, pad_note(57, 8, 0.05, 0.1, 2, 4), 0)
    # the lake: the tune, once more, slowly, alone
    play_tune(T_LAKE + 2.0, 1.0, 0.22, -12)
    nl = int((T_END - T_LAKE) * SR)
    lap = sos_filter(noise(nl), "bandpass", [250, 1200]) * (0.4 + 0.6 * np.sin(2 * math.pi * 0.18 * np.arange(nl) / SR) ** 2) * 0.07
    dry.add(T_LAKE, lap, 0)
    dry.add(T_END - 0.6, click(0.5, 500, 0.05), 0)                                           # STOP

    print("reverbs…", flush=True)
    mix = dry.x + reverb(room.x, reverb_ir(1.2, 0.8, 5000), 0.25) + reverb(big.x, reverb_ir(5.0, 4.0, 5000), 0.45)
    return mix

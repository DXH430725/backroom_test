"""
Ep07 — LEVEL 7 "Thalassophobia": the sound of going down.
A buzzing lamp in a sideways room; the splash; then pressure, the diver's own body,
sonar pings from the surface, whale-scale groans in the dark, and finally the bubbling
of the Abyss.
"""
import json
import math
import os

import numpy as np

from backrooms import sound as snd
from backrooms.sound import (
    SR, Bus, mf, tvec, sos_filter, noise, piano, bell, pad_note, saw_pad, reverb_ir, reverb, click,
    thump, whoosh, hum,
)
from .film import (
    DURATION, HERE, CARD_OPEN, CARD_END, T_ROOM, T_FLIP, T_SPLASH, T_SURF, T_DIVE, T_TWI, T_MID, T_THING,
    T_DARK, T_RELIGHT, T_ABYSS, T_CAVE, T_END,
)

N = int(DURATION * SR)
rng = np.random.default_rng(7)
tt = np.arange(N) / SR


def env(points):
    return snd.env_curve(points, N)


def ping(f=1450, amp=0.15):
    t = tvec(1.2)
    return np.sin(2 * math.pi * f * t) * np.exp(-t * 6) * np.minimum(t / 0.005, 1) * amp


def groan(dur=4.0, f0=38, amp=0.4):
    """something enormous, far too close: a slow, bending infrasonic moan with grit"""
    t = tvec(dur)
    f = f0 * (1 + 0.25 * np.sin(np.pi * t / dur) - 0.15 * t / dur)
    ph = np.cumsum(f) / SR
    x = np.sin(2 * math.pi * ph) + 0.5 * np.sin(4 * math.pi * ph + 0.3) + 0.25 * np.sin(6 * math.pi * ph)
    x = x * (1 + 0.3 * sos_filter(noise(len(t)), "lowpass", 30))
    return x * np.sin(np.pi * t / dur) ** 1.5 * amp


def build():
    dry, room, deep = Bus(DURATION), Bus(DURATION), Bus(DURATION)
    hiss = sos_filter(noise(N), "highpass", 2000) * 0.4
    dry.x += np.stack([hiss, np.roll(hiss, 977)]) * env([(0, 0), (0.4, 0.02), (DURATION - 1, 0.02), (DURATION, 0)])
    for lines, t0 in ((CARD_OPEN, 0.0), (CARD_END, T_END)):
        for text, start in lines:
            for k, ch in enumerate(text):
                if ch not in " ·":
                    dry.add(t0 + start + k / 18 + rng.uniform(0, 0.012), click(rng.uniform(0.05, 0.1), 1800), rng.uniform(-0.3, 0.3))

    # ---- the room: a fluorescent buzz, and the sea breathing outside the door
    nr = int((T_FLIP - T_ROOM) * SR)
    buzz = hum(nr, 120, 6, 0.55) * 0.08 + sos_filter(noise(nr), "bandpass", [2500, 6000]) * 0.004
    room.add(T_ROOM, buzz, 0)
    sea = sos_filter(noise(N), "lowpass", 400) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.12 * tt) ** 2)
    sea_lvl = env([(T_ROOM, 0.03), (T_FLIP - 2, 0.08), (T_FLIP, 0.12), (T_SPLASH, 0.15), (T_SPLASH + 0.01, 0), (T_SURF, 0.0),
                   (T_SURF + 0.4, 0.2), (T_DIVE, 0.2), (T_DIVE + 0.3, 0)])
    room.x += np.stack([sea, np.roll(sea, 3000)]) * sea_lvl
    for k in range(int((T_FLIP - T_ROOM) * 1.6)):
        room.add(T_ROOM + 1 + k * 0.6 + rng.uniform(0, 0.2), sos_filter(noise(int(0.05 * SR)), "bandpass", [1500, 4000]) * 0.03, 0.3)  # drips from the puddle
    # footsteps on wet carpet, then the gravity flip
    for k in range(7):
        dry.add(20.2 + k * 0.62, thump(70, 0.15, 0.12, 20) + sos_filter(noise(int(0.15 * SR)), "bandpass", [300, 1500]) * np.exp(-np.arange(int(0.15 * SR)) / 500) * 0.05, 0)
    dry.add(T_FLIP, whoosh(1.0, 120, 1200, 0.3), 0)
    dry.add(T_FLIP + 0.2, thump(34, 2.5, 0.4), 0)
    dry.add(T_FLIP + 0.6, whoosh(2.4, 300, 3000, 0.4), 0)
    # splash, then muffled underwater roar and back up
    dry.add(T_SPLASH, sos_filter(noise(int(1.6 * SR)), "bandpass", [300, 8000]) * np.exp(-np.arange(int(1.6 * SR)) / (0.25 * SR)) * 0.7, 0)
    dry.add(T_SPLASH + 0.05, sos_filter(noise(int(0.4 * SR)), "lowpass", 250) * 0.6, 0)
    dry.add(T_SURF + 0.1, sos_filter(noise(int(0.6 * SR)), "bandpass", [600, 2500]) * np.hanning(int(0.6 * SR)) * 0.2, 0)  # gasp
    # ---- the surface: vast, cold, quiet; a low wind under the concrete sky
    wind = sos_filter(noise(N), "lowpass", 250) * env([(T_SURF, 0), (T_SURF + 1, 0.12), (T_DIVE, 0.12), (T_DIVE + 0.2, 0)])
    room.x += np.stack([wind, np.roll(wind, 5000)])
    for m, a in ((26, 0.25), (38, 0.08), (45, 0.06), (50, 0.04), (57, 0.03)):
        deep.add(T_SURF + 1, pad_note(m, T_DIVE - T_SURF + 3, a, 0.1, 4, 3), rng.uniform(-0.5, 0.5))
    dry.add(33.5, thump(30, 4, 0.5), 0)
    deep.add(33.5, bell(mf(50), 9, 0.1), 0)

    # ---- under water: pressure bed that deepens with depth; the diver's own heartbeat
    def depth_curve(t):
        from .film import depth_at
        return depth_at(t)
    dpts = np.array([depth_curve(x) for x in np.arange(0, DURATION, 0.5)])
    dcur = np.interp(tt, np.arange(0, DURATION, 0.5), dpts)
    under = env([(T_DIVE - 0.2, 0), (T_DIVE, 1), (T_END - 0.5, 1), (T_END, 0)])
    press = sos_filter(noise(N), "lowpass", 140) * (0.35 + 0.65 * np.clip(dcur / 3000, 0, 1))
    dry.x += np.stack([press, np.roll(press, 2222)]) * 0.12 * under
    hb = T_DIVE + 0.5
    while hb < T_END - 1:
        bpm = 52 + (40 if T_THING < hb < T_RELIGHT + 2 else 0)
        dry.add(hb, thump(44, 0.3, 0.16), 0)
        dry.add(hb + 0.24, thump(40, 0.3, 0.1), 0)
        hb += 60 / bpm
    # sonar pings from the surface team (they get further away)
    pt = T_DIVE + 3
    while pt < T_ABYSS + 10:
        a = 0.12 * math.exp(-(pt - T_DIVE) / 70)
        deep.add(pt, ping(1450, a), 0.3)
        pt += 6.5
    # radio squelch on surface lines
    spec = json.load(open(os.path.join(HERE, "voice", "lines.json"), encoding="utf-8"))
    for ln in spec["lines"]:
        if ln["speaker"] == "surface":
            sq = sos_filter(noise(int(0.12 * SR)), "bandpass", [800, 4000]) * np.hanning(int(0.12 * SR)) * 0.15
            dry.add(ln["start"] - 0.15, sq, 0.25)
    # lamp switch clicks
    for lt_ in (T_TWI + 2.0, T_DARK, T_RELIGHT):
        dry.add(lt_, click(0.3, 600, 0.03), 0.2)

    # ---- the score: deep, slow, getting colder
    for m, a in ((33, 0.2), (45, 0.06), (52, 0.05), (55, 0.04), (60, 0.03)):
        deep.add(T_DIVE, pad_note(m, T_TWI - T_DIVE + 2, a, 0.1, 5, 3), rng.uniform(-0.5, 0.5))
    for m, a in ((31, 0.2), (43, 0.05), (46, 0.05), (50, 0.04), (55, 0.025)):
        deep.add(T_TWI, saw_pad(m, T_MID - T_TWI + 2, a, 500, 5, 3), rng.uniform(-0.5, 0.5))
    deep.add(T_TWI + 8, pad_note(70, 14, 0.02, 0, 4, 4), 0.5)
    deep.add(T_TWI + 8, pad_note(71, 14, 0.015, 0, 4, 4), -0.5)
    for m, a in ((28, 0.22), (40, 0.05), (41, 0.04)):
        deep.add(T_MID, saw_pad(m, T_THING - T_MID + 1, a, 300, 4, 1), 0)
    # The Thing
    deep.add(T_THING - 0.5, groan(5.5, 32, 0.3), -0.6)
    deep.add(T_THING + 3.5, groan(4.0, 27, 0.28), 0.4)
    dry.add(T_DARK - 0.05, thump(28, 1.0, 0.3), 0)
    # darkness: almost nothing. her breath held; a single, enormous pass
    deep.add(126.5, groan(7.0, 24, 0.42), -0.8)
    dry.add(127.5, whoosh(5.0, 60, 400, 0.4), 0)
    deep.add(T_RELIGHT, pad_note(38, 6, 0.08, 0, 2, 3), 0)
    # the abyss: bubbling, tar, the mountain
    na = int((T_END - T_ABYSS) * SR)
    bub = np.zeros(na)
    for _ in range(int(na / SR * 25)):
        i = rng.integers(0, na - 3000)
        f = rng.uniform(150, 600)
        tb = np.arange(3000) / SR
        bub[i:i + 3000] += np.sin(2 * math.pi * f * tb * (1 + 4 * tb)) * np.exp(-tb * 25) * rng.uniform(0.03, 0.09)
    deep.add(T_ABYSS, bub, 0.2)
    for m, a in ((26, 0.25), (38, 0.06), (45, 0.05), (49, 0.04)):
        deep.add(T_ABYSS, pad_note(m, T_CAVE - T_ABYSS + 4, a, 0.2, 4, 4), rng.uniform(-0.5, 0.5))
    deep.add(T_CAVE - 1, bell(mf(57), 6, 0.06), 0)
    # into the cave: everything closes in, then the tape dies
    dry.add(T_CAVE + 1, whoosh(4.0, 800, 60, 0.3), 0)
    dry.add(T_END - 1.2, noise(int(1.2 * SR)) * np.linspace(0, 0.3, int(1.2 * SR)), 0)
    deep.add(T_END + 13.5, piano(45, 6, 0.3), 0)

    print("reverbs…", flush=True)
    mix = dry.x + reverb(room.x, reverb_ir(1.2, 0.7, 5000), 0.3) + reverb(deep.x, reverb_ir(9.0, 7.5, 2500), 0.6)
    # under water, everything is low-passed
    lpf = snd.sos_filter(mix, "lowpass", 900)
    u = env([(T_DIVE - 0.3, 0), (T_DIVE, 0.85), (T_END - 0.5, 0.85), (T_END, 0)])
    return mix * (1 - u) + lpf * u * 1.3

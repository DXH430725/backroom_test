"""
Ep06 — LEVEL 11 "The City That Never Sleeps": a cheerful 90s orientation-video score
(FM electric piano, synth bass, drum machine) that stops dead at the one honest line,
and the near-silence of the footage recorded after the credits.
"""
import math

import numpy as np

from backrooms import sound as snd
from backrooms.sound import (
    SR, Bus, mf, tvec, sos_filter, noise, piano, bell, pad_note, saw_pad, reverb_ir, reverb, click,
    thump, whoosh, hum, saw,
)
from .film import (
    DURATION, CARD_OPEN, CARD_END, T_LOGO, T_FLY, T_STREET, T_PARK, T_CANAL, T_CAPITAL, T_ENT, T_NIGHT,
    T_OUTRO, T_AFTER, T_END,
)

N = int(DURATION * SR)
rng = np.random.default_rng(11)
tt = np.arange(N) / SR


def env(points):
    return snd.env_curve(points, N)


def epiano(m, dur, vel=0.3):
    """two-operator FM electric piano"""
    t = tvec(dur)
    f = mf(m)
    idx = 2.2 * np.exp(-t * 6) + 0.25
    x = np.sin(2 * math.pi * f * t + idx * np.sin(2 * math.pi * f * t))
    x += 0.25 * np.sin(2 * math.pi * f * 14 * t) * np.exp(-t * 40)         # tine "ding"
    return x * np.exp(-t * 1.6) * np.minimum(t / 0.002, 1) * vel


def synbass(m, dur, vel=0.4):
    t = tvec(dur)
    x = sos_filter(saw(np.full(len(t), mf(m))) + 0.5 * np.sin(2 * math.pi * mf(m) * t), "lowpass", 600)
    return x * np.exp(-t * 3) * np.minimum(t / 0.004, 1) * vel


def kick(a=0.5):
    return thump(52, 0.35, a, 9)


def snare(a=0.25):
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    return (sos_filter(noise(n), "bandpass", [1200, 7000]) * np.exp(-t * 22) + 0.4 * np.sin(2 * math.pi * 190 * t) * np.exp(-t * 30)) * a


def hat(a=0.08):
    n = int(0.05 * SR)
    return sos_filter(noise(n), "highpass", 7000) * np.exp(-np.arange(n) / (0.012 * SR)) * a


def groove(bus, t0, t1, bpm=112, prog=None, drums=True, bass=True, keys=True, vel=1.0):
    beat = 60 / bpm
    prog = prog or [(60, (64, 67, 71)), (65, (69, 72, 76)), (62, (65, 69, 72)), (67, (71, 74, 77))]   # Cmaj7 F Dm G
    bar = 0
    t = t0
    while t < t1 - 0.01:
        root, ch = prog[bar % len(prog)]
        for b in range(4):
            tb = t + b * beat
            if tb >= t1:
                break
            if drums:
                if b in (0, 2):
                    bus.add(tb, kick(0.45 * vel), 0)
                if b in (1, 3):
                    bus.add(tb, snare(0.22 * vel), 0.1)
                bus.add(tb, hat(0.06 * vel), 0.4)
                bus.add(tb + beat / 2, hat(0.045 * vel), 0.4)
            if bass:
                bus.add(tb, synbass(root - 24 + (12 if b == 3 else 0), beat * 0.9, 0.35 * vel), 0)
            if keys and b in (0, 2):
                for k, c in enumerate(ch):
                    bus.add(tb + 0.012 * k, epiano(c, beat * 2.2, 0.14 * vel), (-0.4, 0, 0.4)[k])
        # a little topline every other bar
        if keys and bar % 2 == 1:
            for k, m in enumerate((ch[-1] + 12, ch[-2] + 12, ch[-1] + 12, ch[0] + 12)):
                bus.add(t + k * beat * 0.5 + 2 * beat, epiano(m, beat, 0.1 * vel), 0.2)
        bar += 1
        t += 4 * beat


def jingle(bus, t0, vel=1.0):
    """M.E.G. ident: rising FM arpeggio + chord hit + shimmer"""
    for k, m in enumerate((60, 64, 67, 72, 76, 79)):
        bus.add(t0 + k * 0.09, epiano(m, 1.6, 0.18 * vel), (k - 2.5) * 0.25)
    for m in (48, 60, 64, 67, 71, 74):
        bus.add(t0 + 0.6, epiano(m, 3.5, 0.15 * vel), 0)
    bus.add(t0 + 0.6, kick(0.6 * vel), 0)
    bus.add(t0 + 0.6, sos_filter(noise(int(2.5 * SR)), "highpass", 6000) * np.exp(-np.arange(int(2.5 * SR)) / (0.6 * SR)) * 0.08 * vel, 0)


def build():
    dry, music, city, big = Bus(DURATION), Bus(DURATION), Bus(DURATION), Bus(DURATION)
    hiss = sos_filter(noise(N), "highpass", 2000) * 0.4
    dry.x += np.stack([hiss, np.roll(hiss, 977)]) * env([(0, 0), (0.4, 0.02), (DURATION - 1, 0.02), (DURATION, 0)])
    for lines, t0 in ((CARD_OPEN, 0.0), (CARD_END, T_END)):
        for text, start in lines:
            for k, ch in enumerate(text):
                if ch not in " ·":
                    dry.add(t0 + start + k / 18 + rng.uniform(0, 0.012), click(rng.uniform(0.05, 0.1), 1800), rng.uniform(-0.3, 0.3))

    # ---- ident and theme
    jingle(music, T_LOGO + 0.4)
    # aerial: a big pad opening into the groove
    for m, a in ((48, 0.1), (55, 0.07), (60, 0.06), (64, 0.05), (71, 0.035), (76, 0.02)):
        music.add(T_FLY - 0.5, pad_note(m, T_STREET - T_FLY + 2, a, 0.4, 4, 3), rng.uniform(-0.6, 0.6))
    music.add(24.5, kick(0.7), 0)
    music.add(24.5, bell(mf(72), 5, 0.1), 0)
    groove(music, 30.0, T_STREET, keys=True, drums=False, bass=True, vel=0.8)
    groove(music, T_STREET, T_PARK, vel=0.8)
    # the car park trick: a "ta-da" sparkle exactly when we look back
    groove(music, T_PARK, 76.0, vel=0.7)
    music.add(78.3, sos_filter(noise(int(1.2 * SR)), "highpass", 5000) * np.exp(-np.arange(int(1.2 * SR)) / (0.25 * SR)) * 0.12, 0)
    for k, m in enumerate((84, 88, 91, 96)):
        music.add(78.3 + k * 0.06, bell(mf(m), 1.5, 0.05), (k - 1.5) * 0.4)
    groove(music, 80.0, T_CANAL, vel=0.7)
    groove(music, T_CANAL, T_CAPITAL, prog=[(57, (60, 64, 67)), (65, (69, 72, 76)), (60, (64, 67, 71)), (67, (71, 74, 77))], vel=0.6)
    groove(music, T_CAPITAL, T_ENT, vel=0.75)
    groove(music, T_ENT, 141.0, vel=0.65)
    # ---- 141.0: the smile slips. Music stops dead; the tape warbles.
    dry.add(141.0, click(0.3, 400, 0.03), 0)
    big.add(141.2, pad_note(37, 4.0, 0.12, 0.2, 0.05, 2.0), 0)
    big.add(141.2, pad_note(38, 4.0, 0.08, 0.2, 0.05, 2.0), 0)
    groove(music, 145.0, T_NIGHT, vel=0.6)                                       # and back, as if nothing happened
    # night: the city hums, the theme goes slow and dreamy
    for m, a in ((45, 0.1), (52, 0.07), (57, 0.06), (60, 0.05), (64, 0.04), (71, 0.025)):
        music.add(T_NIGHT - 0.5, pad_note(m, T_OUTRO - T_NIGHT + 1.5, a, 0.3, 3, 2), rng.uniform(-0.6, 0.6))
    for k in range(8):
        music.add(T_NIGHT + 1 + k * 1.9, epiano(rng.choice([76, 79, 81, 84]), 3, 0.08), rng.uniform(-0.6, 0.6))
    # outro jingle
    jingle(music, T_OUTRO + 0.4)
    groove(music, T_OUTRO + 2.0, T_AFTER - 0.8, vel=0.6)

    # ---- the city itself: an empty metropolis — wind, a far hum, nothing alive
    wind = sos_filter(noise(N), "lowpass", 300) * 0.5 + sos_filter(noise(N), "bandpass", [600, 2000]) * 0.06
    cl = env([(T_FLY, 0.5), (T_STREET, 0.35), (T_STREET + 0.2, 0.25), (T_CAPITAL, 0.25), (T_CAPITAL + 0.3, 0.15),
              (118.5, 0.15), (118.8, 0.4), (T_ENT, 0.4), (T_ENT + 0.2, 0.25), (T_OUTRO, 0.25), (T_OUTRO + 0.2, 0),
              (T_AFTER, 0), (T_AFTER + 0.2, 0.3), (T_END - 0.3, 0.3), (T_END, 0)])
    city.x += np.stack([wind, np.roll(wind, 7000)]) * cl * 0.25
    far = hum(N, 50, 4, 0.5) * 0.012 * env([(T_STREET, 0), (T_STREET + 1, 1), (T_OUTRO, 1), (T_OUTRO + 0.1, 0), (T_AFTER, 0), (T_AFTER + 1, 1.5), (T_END, 1.5), (T_END + 0.1, 0)])
    city.x += np.stack([far, far])
    # canal water
    nc = int((T_CAPITAL - T_CANAL) * SR)
    water = sos_filter(noise(nc), "bandpass", [200, 1500]) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.3 * np.arange(nc) / SR) ** 2) * 0.06
    city.add(T_CANAL, water, -0.3)
    # the Capital: people! murmur and footsteps
    ncap = int((118.5 - T_CAPITAL) * SR)
    mur = sos_filter(noise(ncap), "bandpass", [250, 2500])
    syl = sos_filter(rng.random(ncap // 2400 + 2).repeat(2400)[:ncap], "lowpass", 8)
    syl = np.pad(syl, (0, max(0, ncap - len(syl))))[:ncap]
    city.add(T_CAPITAL, np.stack([mur * syl, np.roll(mur * syl, 3000)]) * 0.12 * np.minimum(np.arange(ncap) / SR, 1), 0)
    # entities: the hound scuttles
    for k in range(18):
        city.add(129.5 + k * 0.38, click(0.06, 900, 0.03), -0.3 + k * 0.03)
    # after the credits: silence, a far car, the glitches, and something standing on the kerb
    for gt in (177.0, 182.5):
        dry.add(gt, noise(int(0.12 * SR)) * 0.3, 0)
    big.add(182.6, pad_note(34, 3.4, 0.2, 0.1, 0.05, 1.0), 0)
    dry.add(T_END - 0.8, noise(int(0.8 * SR)) * np.linspace(0, 0.35, int(0.8 * SR)), 0)

    print("reverbs…", flush=True)
    m = reverb(music.x, reverb_ir(1.6, 1.0, 7000), 0.18)
    # an old tape: a little bandwidth loss and wow on the music
    m = sos_filter(m, "lowpass", 9000)
    mix = dry.x + m * 0.9 + reverb(city.x, reverb_ir(5.5, 4.0, 4000), 0.5) + reverb(big.x, reverb_ir(6.0, 4.0, 4000), 0.5)
    return mix

"""
Ep09 — LEVEL 9 "The Suburbs": what a phone hears at midnight in a neighbourhood that
never ends. Almost nothing: a dying lamp's ballast, a power line, his own shoes on wet
asphalt. Then the phone starts to buzz the way phones do near interference, and something
huge breathes in the fog with a hundred human voices.
"""
import math

import numpy as np

from backrooms import sound as snd
from backrooms.sound import (
    SR, Bus, mf, tvec, sos_filter, noise, piano, bell, pad_note, saw_pad, choir_note, vowel, reverb_ir, reverb, click,
    thump, whoosh, hum,
)
from . import film as fm
from .film import (
    DURATION, CARD_OPEN, CARD_END, POST_BODY, T_POST, T_WAKE, T_WALK, T_TWIN, T_DUST, T_WATCH, T_COLD, T_PILLAR, T_RUN,
    T_SIGN, T_CORRUPT, T_END,
)

N = int(DURATION * SR)
rng = np.random.default_rng(9)
tt = np.arange(N) / SR


def env(points):
    return snd.env_curve(points, N)


def step(amp=0.1, wet=0.6, hard=0.5):
    """a sneaker on wet asphalt or concrete"""
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    tap = sos_filter(noise(n), "bandpass", [700, 5000]) * np.exp(-t * 45) * hard
    slap = sos_filter(noise(n), "bandpass", [250, 1800]) * np.exp(-t * 22) * wet
    return (tap + slap) * amp + thump(80, 0.18, amp * 0.5, 25)[:n]


def gsm(dur, amp=0.1):
    """the dit-dit-dit of a phone's radio bleeding into its own microphone"""
    t = tvec(dur)
    gate = ((t * 217.0) % 1.0) < 0.12
    burst = (np.sin(2 * math.pi * 0.9 * t) > -0.2) & (((t * 3.7) % 1.0) < 0.75)
    x = np.sign(np.sin(2 * math.pi * 217 * t)) * gate * burst
    return sos_filter(x.astype(np.float64), "bandpass", [180, 4000]) * amp


def roar(dur, f0=55.0, amp=0.3, seed=0):
    """a roar made of human voices: many formant-filtered throats, pitched down, smeared"""
    r = np.random.default_rng(seed)
    t = tvec(dur)
    out = np.zeros(len(t))
    for k in range(14):
        f = f0 * r.uniform(0.85, 2.2) * (1 + 0.04 * np.sin(2 * math.pi * r.uniform(0.2, 0.6) * t + k))
        ph = np.cumsum(f) / SR
        x = 2 * (ph % 1.0) - 1
        out += vowel(x, r.choice(["a", "o", "u"])) * r.uniform(0.5, 1.0)
    e = np.minimum(t / (dur * 0.3), 1) * np.minimum((dur - t) / (dur * 0.5), 1)
    return out / 14 * np.clip(e, 0, 1) ** 1.5 * amp


def step_times():
    out = []
    acc, prev = 0.0, None
    for t in np.arange(T_WAKE + 6.0, T_END, 1 / 30):
        c = fm.cam_at(t)
        p = c.pos[:2].copy()
        if prev is not None:
            d = float(np.linalg.norm(p - prev))
            if d < 1.0:
                acc += d
                if acc > 0.78:
                    acc = 0.0
                    out.append(t)
        prev = p
    return out


def build():
    card, dia, world, music = Bus(DURATION), Bus(DURATION), Bus(DURATION), Bus(DURATION)
    hiss = sos_filter(noise(N), "highpass", 2000) * 0.4
    card.x += np.stack([hiss, np.roll(hiss, 977)]) * env([(0, 0), (0.4, 0.02), (T_WAKE - 0.2, 0.02), (T_WAKE, 0),
                                                          (T_END, 0), (T_END + 0.3, 0.02), (DURATION - 1, 0.02), (DURATION, 0)])
    for lines, t0 in ((CARD_OPEN, 0.0), (CARD_END, T_END)):
        for text, start in lines:
            for k, ch in enumerate(text):
                if ch not in " ·":
                    card.add(t0 + start + k / 18 + rng.uniform(0, 0.012), click(rng.uniform(0.05, 0.1), 1800), rng.uniform(-0.3, 0.3))
    # ---- the forum post: a laptop keyboard in a quiet room
    room = sos_filter(noise(int((T_WAKE - T_POST) * SR)), "lowpass", 300) * 0.01
    card.add(T_POST, room, 0)
    total = sum(len(l) for l in POST_BODY)
    for k in range(total):
        tk = T_POST + 1.0 + k / 28 + rng.uniform(-0.01, 0.01)
        if tk < T_POST + 7.0:
            card.add(tk, click(rng.uniform(0.04, 0.09), rng.uniform(1200, 2600), 0.015), rng.uniform(-0.2, 0.2))
    card.add(T_POST + 7.3, click(0.25, 900, 0.02), 0.0)        # the trackpad click on the video

    # ---- the phone: handling noise, wind on the mic, a midnight with nothing in it
    nph = N
    wind = sos_filter(noise(nph), "lowpass", 160) * (0.4 + 0.6 * np.clip(sos_filter(noise(nph), "lowpass", 0.5) * 30, 0, 1))
    w_lvl = env([(T_WAKE, 0), (T_WAKE + 0.5, 0.25), (T_COLD, 0.25), (T_COLD + 6, 0.6), (T_RUN, 0.8), (T_SIGN, 0.45),
                 (T_END - 0.3, 0.45), (T_END, 0)])
    world.x += np.stack([wind, np.roll(wind, 700)]) * w_lvl
    air = sos_filter(noise(N), "bandpass", [100, 900]) * env([(T_WAKE, 0), (T_WAKE + 0.5, 0.012), (T_END - 0.3, 0.012), (T_END, 0)])
    world.x += np.stack([air, np.roll(air, 3000)])
    # waking: a groan, clothes on wet asphalt, sitting up
    dia.add(T_WAKE + 0.6, sos_filter(noise(int(1.2 * SR)), "bandpass", [200, 900]) * np.hanning(int(1.2 * SR)) * 0.05, 0)
    for k in range(5):
        world.add(27.0 + k * 0.7 + rng.uniform(0, 0.2), sos_filter(noise(int(0.4 * SR)), "bandpass", [800, 5000]) * np.hanning(int(0.4 * SR)) * 0.04, rng.uniform(-0.3, 0.3))
    dia.add(28.6, click(0.35, 2500, 0.01), 0.1)          # flashlight on
    # the lamp: a sodium ballast buzz that cuts with the flicker
    for li, gain in ((fm.L_WAKE, 0.012), (fm.L_DUST, 0.007), (fm.L_FOG, 0.008), (fm.L_SIGN, 0.007)):
        lamp = np.zeros(N)
        ts = np.arange(T_WAKE, T_END, 1 / 60)
        b = np.array([fm.lamp_states(x).get(li, 0.0) for x in ts])
        bb = np.interp(tt, ts, b, left=0, right=0)
        base = hum(N, 120, 7, 0.5) + sos_filter(noise(N), "bandpass", [3000, 6000]) * 0.2
        world.x += np.stack([base, base]) * (bb * gain)
        edges = ts[1:][np.abs(np.diff(b)) > 0.3]
        for e in edges:
            world.add(float(e), click(0.08 * gain / 0.012, 1500, 0.01), 0.2)
    # the power lines overhead: a faint mains hum
    pl = hum(N, 60, 6, 0.6) * 0.006
    world.x += np.stack([pl, pl]) * env([(T_WAKE + 6, 0), (T_WAKE + 10, 1), (T_END, 1)])
    # drips from gutters, a leaf skittering
    dt = T_WAKE + 2
    while dt < T_COLD:
        f = rng.uniform(1000, 2500)
        tdr = tvec(0.15)
        world.add(dt, np.sin(2 * math.pi * f * tdr * (1 + 2 * tdr)) * np.exp(-tdr * 40) * rng.uniform(0.01, 0.03), rng.uniform(-0.8, 0.8))
        dt += rng.exponential(2.2)
    for lt in (49.0, 58.5, 69.5):
        n = int(0.9 * SR)
        world.add(lt, sos_filter(noise(n), "bandpass", [2000, 7000]) * (rng.random(n) < 0.02) * np.hanning(n) * 0.6, rng.uniform(-0.8, 0.8))

    # ---- his feet; his breath
    for k, st in enumerate(step_times()):
        running = T_RUN < st < T_SIGN
        world.add(st + rng.uniform(-0.015, 0.015), step(0.12 if running else 0.07, 0.6, 0.6 if running else 0.4), -0.1 if k % 2 else 0.1)
    bt = T_WAKE + 3.0
    while bt < T_END - 0.5:
        fast = (T_RUN < bt < T_SIGN + 4) or (101 < bt < 104)
        held = 104.5 < bt < 114.0
        per = 0.75 if fast else (3.6 if held else 2.9)
        amp = 0.035 if fast else 0.012
        if not held:
            n = int(per * 0.45 * SR)
            dia.add(bt, sos_filter(noise(n), "bandpass", [500, 2600]) * np.hanning(n) * amp, 0)
        bt += per
    # the dust: rubbing it between his fingers
    n = int(1.6 * SR)
    grit = sos_filter(noise(n), "highpass", 3000) * (rng.random(n) < 0.08) * np.hanning(n) * 0.25
    dia.add(78.4, grit, 0.1)

    # ---- the Watch: the phone's radio buzzing into its own mic, then the eye
    dia.add(90.0, gsm(28.0, 0.035) * np.linspace(0.3, 1, int(28.0 * SR)), 0)
    music.add(95.0, saw_pad(31, 24, 0.09, 220, 3, 4), 0)
    music.add(95.0, pad_note(43, 24, 0.04, 0.5, 4, 4), 0)
    nw = int(23.5 * SR)
    tw = np.arange(nw) / SR
    wet = sos_filter(noise(nw), "lowpass", 90) * (0.7 + 0.3 * np.sin(2 * math.pi * 0.4 * tw)) * 0.5
    world.add(95.0, wet * snd.env_curve([(0, 0), (3, 1), (20, 1), (23.5, 0)], nw), -0.2)
    # the beam: a thin high whine; the sweep over the car: everything rises
    nb = int(19.5 * SR)
    tb = np.arange(nb) / SR
    whine = np.sin(2 * math.pi * (3150 + 40 * np.sin(2 * math.pi * 0.3 * tb)) * tb) * 0.008
    whine *= snd.env_curve([(0, 0), (1.5, 1), (11.0, 1), (12.5, 2.4), (15.5, 2.4), (17, 1), (19.5, 0)], nb)
    world.add(97.5, whine, -0.3)
    dia.add(101.5, whoosh(0.8, 300, 2000, 0.12), 0)          # the dash to the car
    dia.add(103.2, thump(60, 0.3, 0.2), 0.2)                  # knees on asphalt
    dia.add(103.6, click(0.3, 2500, 0.01), 0.1)               # flashlight off
    hb = 104.0
    while hb < 119:
        dia.add(hb, thump(46, 0.3, 0.16), 0)
        dia.add(hb + 0.22, thump(42, 0.3, 0.1), 0)
        hb += 60 / (125 if 108 < hb < 115 else 100)
    music.add(109.0, saw_pad(73, 6, 0.012, 3000, 1.5, 3), 0.4)
    music.add(109.0, saw_pad(74, 6, 0.012, 3000, 1.5, 3), -0.4)
    dia.add(119.0, sos_filter(noise(int(2.0 * SR)), "bandpass", [400, 2000]) * np.hanning(int(2.0 * SR)) * 0.05, 0)  # the exhale

    # ---- the cold and the fog: houses creak as the temperature falls; a drone under everything
    for ct in (121.5, 124.8, 127.3, 130.9):
        n = int(0.6 * SR)
        f = rng.uniform(300, 700)
        tc = np.arange(n) / SR
        world.add(ct, np.sin(2 * math.pi * f * tc * (1 - 0.2 * tc)) * (rng.random(n) < 0.25) * np.hanning(n) * 0.05, rng.uniform(-0.9, 0.9))
    for m, a in ((26, 0.08), (33, 0.03), (38, 0.02)):
        music.add(T_COLD + 2, pad_note(m, T_RUN - T_COLD + 4, a, 0.3, 6, 3), rng.uniform(-0.4, 0.4))
    # ---- the Mangled: tremors, rattling windows, a roar of voices from very far away
    for tr in (138.5, 143.5):
        world.add(tr - 0.1, thump(26, 3.0, 0.55, 2.0), 0)
        n = int(2.5 * SR)
        world.add(tr, sos_filter(noise(n), "bandpass", [1500, 6000]) * (rng.random(n) < 0.05) * np.exp(-np.arange(n) / SR * 1.2) * 0.3, rng.uniform(-0.5, 0.5))
    music.add(139.0, roar(8.0, 48.0, 0.25, 1), -0.5)
    music.add(144.0, roar(6.0, 42.0, 0.3, 2), -0.6)
    # ---- run; the sign; something like hope, then the phone dies
    for m, a in ((50, 0.05), (57, 0.04), (62, 0.03), (66, 0.02)):
        music.add(T_SIGN - 1, pad_note(m, 12, a, 0.2, 2.5, 4), rng.uniform(-0.5, 0.5))
    music.add(T_SIGN + 0.5, bell(mf(74), 5, 0.03), 0.2)
    dia.add(T_CORRUPT, gsm(T_END - T_CORRUPT, 0.06), 0)
    ncr = int((T_END - T_CORRUPT) * SR)
    stut = sos_filter(noise(ncr), "bandpass", [300, 6000]) * (np.random.default_rng(5).random(ncr // 1200 + 1).repeat(1200)[:ncr] < 0.3) * 0.08
    dia.add(T_CORRUPT, stut * np.linspace(0.2, 1, ncr), 0)

    # ---- end card
    music.add(T_END + 7.5, piano(50, 6, 0.08), 0)
    music.add(T_END + 7.5, piano(57, 6, 0.05), 0)
    music.add(T_END + 9.1, piano(62, 6, 0.04), 0)

    print("reverbs…", flush=True)
    # everything the phone hears goes through its little microphone
    ph = dia.x + reverb(world.x, reverb_ir(1.6, 1.2, 4000), 0.35)
    ph = sos_filter(ph, "highpass", 140)
    ph = sos_filter(ph, "lowpass", 9000)
    # the phone dies at the end: a hard cut
    cut = int(T_END * SR)
    ph[:, cut:] = 0
    mus = reverb(music.x, reverb_ir(6.0, 4.5, 4000), 0.6)
    mus[:, cut:int((T_END + 6.0) * SR)] = 0
    mix = card.x + ph + mus
    mix[:, cut - int(0.02 * SR):cut] *= np.linspace(1, 0, int(0.02 * SR))
    mix[:, cut:int((T_END + 0.3) * SR)] = 0
    return mix

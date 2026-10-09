"""
Synthesis toolkit shared by every episode's score: buses, instruments, filters and
reverbs, all built from numpy/scipy. Nothing is sampled.
"""
import math

import numpy as np
from scipy import signal

SR = 48000
rng = np.random.default_rng(254)


def seed(n):
    global rng
    rng = np.random.default_rng(n)


def mf(m):
    """MIDI note -> Hz"""
    return 440.0 * 2 ** ((m - 69) / 12)


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def env_curve(points, n):
    """piecewise-linear envelope of length n from [(sec, value), ...]"""
    ts = np.array([p[0] for p in points]) * SR
    vs = np.array([p[1] for p in points])
    return np.interp(np.arange(n), ts, vs)


class Bus:
    def __init__(self, duration):
        self.n = int(duration * SR)
        self.x = np.zeros((2, self.n))

    def add(self, start, sig, pan=0.0, gain=1.0):
        i0 = int(start * SR)
        if i0 >= self.n:
            return
        if sig.ndim == 1:
            l = math.cos((pan + 1) * math.pi / 4)
            r = math.sin((pan + 1) * math.pi / 4)
            sig = np.stack([sig * l, sig * r])
        if i0 < 0:
            sig = sig[:, -i0:]
            i0 = 0
        n = min(sig.shape[1], self.n - i0)
        if n > 0:
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
        e[:na] = np.linspace(0, 1, min(na, n))[: len(e[:na])]
    if nr and nr < n:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def stereo_noise(n, kind="lowpass", freq=800, decor=4001):
    x = sos_filter(noise(n), kind, freq)
    return np.stack([x, np.roll(x, decor)])


# ----------------------------------------------------------------------------
# instruments
# ----------------------------------------------------------------------------
def piano(m, dur=3.0, vel=1.0, detune_cents=0.0, bright=1.0):
    f = mf(m) * 2 ** (detune_cents / 1200)
    t = tvec(dur)
    out = np.zeros_like(t)
    B = 0.0004
    for n in range(1, 10):
        fn = f * n * math.sqrt(1 + B * n * n)
        if fn > 12000:
            break
        amp = vel / n ** (1.3 / bright)
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


def saw(freq_t):
    ph = np.cumsum(freq_t / SR)
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
    e = np.minimum(t / max(a, 1e-3), 1) * np.minimum((dur - t) / max(r, 1e-3), 1)
    return out * lfo * np.clip(e, 0, 1) ** 2 * amp / 3


def saw_pad(m, dur, amp=1.0, cutoff=1200, a=3.0, r=4.0, voices=4, detune=9):
    t = tvec(dur)
    f = mf(m)
    out = np.zeros_like(t)
    for v in range(voices):
        c = rng.uniform(-detune, detune)
        out += saw(np.full(len(t), f * 2 ** (c / 1200)))
    out = sos_filter(out / voices, "lowpass", cutoff)
    e = np.minimum(t / max(a, 1e-3), 1) * np.minimum((dur - t) / max(r, 1e-3), 1)
    return out * np.clip(e, 0, 1) ** 1.5 * amp


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


VOWELS = {
    "a": ((750, 130, 1.0), (1150, 150, 0.55), (2600, 250, 0.22), (3300, 300, 0.12)),
    "o": ((450, 100, 1.0), (800, 120, 0.5), (2830, 250, 0.12), (3500, 300, 0.06)),
    "u": ((330, 80, 1.0), (700, 100, 0.35), (2400, 250, 0.08), (3300, 300, 0.04)),
}


def vowel(x, v="a"):
    y = np.zeros_like(x)
    for fc, bw, g in VOWELS[v]:
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
    n = x.shape[1]
    y = np.stack([signal.fftconvolve(x[0], ir[0])[:n], signal.fftconvolve(x[1], ir[1])[:n]])
    return x * (1 - wet) + y * wet


def click(amp=1.0, hp=2500, dur=0.012):
    n = int(dur * SR)
    return sos_filter(noise(n), "highpass", hp) * np.exp(-np.arange(n) / (0.002 * SR)) * amp


def thump(f=55, dur=0.35, amp=1.0, decay=11):
    t = tvec(dur)
    return np.sin(2 * math.pi * f * t * (1 + 0.6 * np.exp(-t * 25))) * np.exp(-t * decay) * amp


def whoosh(dur, f0, f1, amp=1.0):
    n = int(dur * SR)
    x = noise(n)
    out = np.zeros(n)
    seg = 2048
    fs = np.geomspace(f0, f1, n // seg + 1)
    for k in range(0, n, seg):
        lo = max(fs[k // seg] * 0.7, 20)
        hi = min(fs[k // seg] * 1.4, SR / 2 - 100)
        sos = signal.butter(2, [lo, hi], btype="bandpass", fs=SR, output="sos")
        out[k:k + seg] = signal.sosfilt(sos, x[max(0, k - 512):k + seg])[-len(out[k:k + seg]):]
    e = np.sin(np.linspace(0, math.pi, n)) ** 2
    return out * e * amp


def hum(n, base=120.0, harmonics=6, decay=0.55):
    t = np.arange(n) / SR
    return sum(math.pow(decay, k) * np.sin(2 * math.pi * base * (k + 1) * t) for k in range(harmonics))


def drone(n, f, amp=1.0, beat=0.15):
    t = np.arange(n) / SR
    return amp * (np.sin(2 * math.pi * f * t) + 0.6 * np.sin(2 * math.pi * (f + beat) * t)
                  + 0.25 * np.sin(2 * math.pi * 2 * f * t + 1))


def master(mix, dur, drive=1.25, ceiling=0.93, fade=1.5):
    mix = sos_filter(mix, "highpass", 22)
    peak = np.abs(mix).max() + 1e-9
    mix = np.tanh(mix / peak * drive) / math.tanh(drive) * ceiling
    n = mix.shape[1]
    return mix * env_curve([(0, 1), (dur - fade, 1), (dur, 0)], n)


def write_wav(path, mix):
    import wave
    pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    """-> stereo float array (2, n) at SR"""
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x / 32768.0
    elif x.dtype == np.int32:
        x = x / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x - 128) / 128.0
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x])
    else:
        x = x.T[:2]
        if x.shape[0] == 1:
            x = np.concatenate([x, x])
    if sr != SR:
        g = math.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g, axis=1)
    return x

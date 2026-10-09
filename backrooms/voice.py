"""
Voice-over: drops the recorded lines from <episode>/voice/wav/<id>.wav onto the
score at the times in voice/lines.json, colours each speaker with its "fx" chain
and ducks the music under the voice. Missing files are simply skipped, so an
episode renders fine before any voice exists.
"""
import json
import math
import os

import numpy as np

from .sound import SR, sos_filter, noise, reverb_ir, reverb, read_wav


def _fx(x, kind):
    """x: stereo (2, n). Effect chains named in lines.json -> speakers -> fx."""
    if kind == "tape":        # interview cassette: narrow band, soft saturation, hiss, wow
        n = x.shape[1]
        wow = 1 + 0.0025 * np.sin(2 * math.pi * 0.7 * np.arange(n) / SR)
        idx = np.clip(np.cumsum(wow) - 1, 0, n - 1)
        x = np.stack([np.interp(idx, np.arange(n), c) for c in x])
        x = sos_filter(x, "bandpass", [280, 3600], 3)
        x = np.tanh(x * 2.2) / 2.2
        x = x + 0.004 * sos_filter(noise(n), "highpass", 2000)
        return x * 1.25
    if kind == "radio":       # walkie-talkie / intercom
        x = sos_filter(x, "bandpass", [450, 2800], 4)
        x = np.tanh(x * 4.0) / 3.0
        return x * 1.4
    if kind == "whisper":     # intimate, close, a little air
        return sos_filter(x, "highpass", 120) * 1.1
    if kind == "distant":     # far away in a big space
        x = sos_filter(x, "lowpass", 2500)
        return reverb(x, reverb_ir(3.0, 2.5, 3000), 0.7) * 0.8
    if kind == "muffled":     # speech swallowed by the room: dulled, then cut off mid-word
        x = sos_filter(x, "lowpass", 900, 2) * 1.6
        cut = int(0.55 * SR)
        if x.shape[1] > cut:
            fade = int(0.02 * SR)
            x[:, cut - fade:cut] *= np.linspace(1, 0, fade)
            x[:, cut:] = 0
        return x
    if kind == "phone":       # a phone's own mic, held at arm's length outdoors
        x = sos_filter(x, "bandpass", [160, 7000], 2)
        x = np.tanh(x * 1.8) / 1.8
        return reverb(x, reverb_ir(0.8, 0.5, 5000), 0.12) * 1.2
    if kind == "corrupt":     # the same, coming apart: crushed, dropping out
        x = _fx(x, "phone")
        q = 2 ** 5
        x = np.round(x * q) / q
        n = x.shape[1]
        blk = int(0.05 * SR)
        keep = (np.random.default_rng(9).random(n // blk + 1) > 0.25).repeat(blk)[:n]
        return x * keep
    if kind == "radio_far":
        x = _fx(x, "radio")
        return reverb(x, reverb_ir(2.0, 1.4, 3000), 0.35)
    # "clean" narrator: gentle low cut and presence
    x = sos_filter(x, "highpass", 80)
    return x + 0.12 * sos_filter(x, "bandpass", [2500, 5000])


def mix_voice(music, ep_dir, duck_depth=0.55, voice_gain=1.0, report=True):
    """music: (2, n). Returns (mixed, n_lines_found)."""
    path = os.path.join(ep_dir, "voice", "lines.json")
    if not os.path.exists(path):
        return music, 0
    spec = json.load(open(path, encoding="utf-8"))
    speakers = spec.get("speakers", {})
    n = music.shape[1]
    vbus = np.zeros((2, n))
    found = 0
    for line in spec["lines"]:
        wav = os.path.join(ep_dir, "voice", "wav", f"{line['id']}.wav")
        if not os.path.exists(wav):
            continue
        x = read_wav(wav)
        peak = np.abs(x).max() + 1e-9
        x = x / peak * 0.5
        sp = speakers.get(line["speaker"], {})
        x = _fx(x, line.get("fx", sp.get("fx", "clean")))
        x *= line.get("gain", 1.0) * sp.get("gain", 1.0)
        pan = line.get("pan", sp.get("pan", 0.0))
        x[0] *= math.cos((pan + 1) * math.pi / 4) * math.sqrt(2)
        x[1] *= math.sin((pan + 1) * math.pi / 4) * math.sqrt(2)
        i0 = int(line["start"] * SR)
        m = min(x.shape[1], n - i0)
        if m <= 0:
            continue
        vbus[:, i0:i0 + m] += x[:, :m]
        found += 1
        dur = x.shape[1] / SR
        if report and dur > line.get("max_dur", 1e9) * 1.05:
            print(f"  [voice] {line['id']} is {dur:.2f}s (max {line['max_dur']}s) — may overlap the next cue")
    if not found:
        return music, 0
    # ducking: envelope follower on the voice bus
    lvl = np.abs(vbus).max(0)
    win = int(0.03 * SR)
    lvl = np.convolve(lvl, np.ones(win) / win, mode="same")
    gate = (lvl > 0.01).astype(float)
    att, rel = int(0.08 * SR), int(0.5 * SR)
    env = np.zeros(n)
    e = 0.0
    step = 64
    for i in range(0, n, step):
        g = gate[i]
        e += (g - e) * (step / att if g > e else step / rel)
        env[i:i + step] = e
    out = music * (1 - duck_depth * env) + vbus * voice_gain
    if report:
        print(f"  [voice] mixed {found} line(s)")
    return out, found

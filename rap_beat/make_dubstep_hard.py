#!/usr/bin/env python3
"""
make_dubstep_hard.py - a heavy 1-minute dubstep track built around the "meh"
in a video clip.

The "meh" is everywhere:
  * The full, uncut "meh" is the lead vocal hook. It sits on top of the mix
    and the bass ducks out of its way every time it plays.
  * Every bass hit TALKS: a 28-band vocoder imprints the shape of your "meh"
    onto distorted saw / FM-growl / wobble carriers, so the bass says "meh".
  * Build-ups stutter it ("m-m-m-m-MEH") and the drop is announced by the
    full "MEH" landing right on the downbeat.

140 BPM half-time, 35 bars = 60 s:
  bars  1-4   intro    dry "meh" calls with ping-pong delay + reverb, sub drone
  bars  5-8   build    meh stutters 1/4 -> 1/8 -> 1/16 -> 1/32, snare roll, riser
  bars  9-20  DROP 1   talking bass call-and-response with "MEH" on every snare
  bars 21-24  break    huge reverb meh, long "meeeeh" bass, second build
  bars 25-34  DROP 2   faster, nastier patterns, pitch dives, meh stutters, demon meh
  bar  35     end      impact + demon "MEEEH" ringing out

Usage
  python make_dubstep_hard.py INPUT_VIDEO [-o dubstep_hard.wav] [--bpm 140]

Requires: ffmpeg on PATH, and `pip install numpy scipy soundfile`.
Lives next to make_beat.py and make_dubstep.py (reuses their helpers).
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_beat import (SR, load_audio, loud_region, fade, filt, pitch_shift,  # noqa: E402
                       env, estimate_f0, make_snare, make_hats)
from make_dubstep import saw, svf_lowpass, noise_riser  # noqa: E402

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ================================================================ the "meh"
def get_meh(src):
    meh = loud_region(src, thresh_ratio=0.08)
    meh = filt(meh, "highpass", 110)
    meh = np.tanh(2.0 * meh / np.abs(meh).max())        # saturate -> thicker, louder
    return fade(meh / np.abs(meh).max(), 0.003, 0.04)


class Vocoder:
    """Channel vocoder: stores the band envelopes of the "meh" and imprints
    them (stretched to any length) onto a carrier."""

    def __init__(self, voice, n_bands=28, lo=100, hi=8000, hop=32):
        edges = np.geomspace(lo, hi, n_bands + 1)
        self.sos = [butter(2, [a, b], "bandpass", fs=SR, output="sos")
                    for a, b in zip(edges[:-1], edges[1:])]
        smooth = butter(2, 60, "lowpass", fs=SR, output="sos")
        E = []
        for s in self.sos:
            y = sosfilt(s, voice)
            E.append(np.sqrt(np.maximum(sosfilt(smooth, y * y), 0))[::hop])
        E = np.array(E) * np.geomspace(1, 3, n_bands)[:, None]  # brighten -> clearer
        tot = E.sum(axis=0)
        on = np.where(tot > 0.08 * tot.max())[0]
        self.E = E[:, on[0]:on[-1] + 1] / E.max()

    def __call__(self, carrier):
        n, F = len(carrier), self.E.shape[1]
        pos = np.linspace(0, F - 1, n)
        out = np.zeros(n)
        for b, s in enumerate(self.sos):
            out += sosfilt(s, carrier) * np.interp(pos, np.arange(F), self.E[b])
        return out


# ================================================================ bass
def lfo_wave(n, rate, beat):
    ph = np.arange(n) * rate / (beat * SR)
    return 0.5 - 0.5 * np.cos(2 * np.pi * ph)


def carrier(kind, freq, n, beat, rate, rng):
    f = np.full(n, float(freq))
    if kind == "dive":
        f = freq * np.geomspace(1, 0.3, n)
    if kind in ("talk", "dive"):
        c = (saw(f) + 0.7 * saw(f * 1.008) + 0.7 * saw(f * 0.992)
             + 0.5 * saw(2 * f * 1.003) + 0.3 * saw(3 * f * 0.998))
    elif kind == "growl":                        # FM growl, index swept by the LFO
        lfo = lfo_wave(n, rate, beat)
        ph = 2 * np.pi * np.cumsum(f) / SR
        c = np.sin(ph + (1.5 + 8 * lfo) * np.sin(2 * ph + 3 * lfo * np.sin(3 * ph)))
        c = np.sign(c) * np.abs(c) ** 0.4
    else:                                        # "wub": classic LFO low-pass wobble
        lfo = lfo_wave(n, rate, beat)
        c = saw(f) + 0.7 * saw(f * 1.01) + 0.7 * saw(f * 0.99) + 0.4 * saw(2 * f)
        c = svf_lowpass(c, 110 * (1 + 35 * lfo ** 1.5), q=0.15)
    return np.tanh(2.5 * c) + 0.03 * rng.standard_normal(n), f


def bass_hit(voc, kind, freq, dur, beat, rate, rng):
    n = int(dur * SR)
    c, f = carrier(kind, freq, n, beat, rate, rng)
    v = voc(c)
    v = np.tanh(4.0 * v / (np.abs(v).max() + 1e-9))      # hard distortion
    v = filt(v, "highpass", 110)
    sub_f = np.where(f > 100, f / 2, f)
    sub = np.sin(2 * np.pi * np.cumsum(sub_f) / SR)
    out = 0.95 * v + 0.75 * sub
    shape = np.ones(n)
    a, r = int(0.002 * SR), min(int(0.012 * SR), n // 2)
    shape[:a] = np.linspace(0, 1, a)
    shape[-r:] = np.linspace(1, 0, r)
    return out * shape


# ================================================================ drums & fx
def hard_kick(rng):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 45 + 170 * np.exp(-t / 0.025)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.0005, 0.22)
    click = filt(rng.standard_normal(n), "highpass", 3000) * env(n, 0, 0.004)
    k = np.tanh(3.5 * (body + 0.45 * click))
    return k / np.abs(k).max()


def hard_snare(meh, rng):
    n = int(0.4 * SR)
    base = np.pad(make_snare(meh, rng), (0, n))[:n]
    noise = filt(rng.standard_normal(n), "bandpass", [900, 6000])
    clap = np.zeros(n)
    for off in (0, 0.009, 0.018):
        i = int(off * SR)
        clap[i:] += noise[: n - i] * env(n - i, 0.0005, 0.007)
    clap += noise * env(n, 0.02, 0.13) * 0.8
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * np.cumsum(180 + 60 * np.exp(-t / 0.01)) / SR) * env(n, 0.0005, 0.07)
    s = np.tanh(2.5 * (0.6 * base + 0.8 * clap + 0.6 * body))
    return s / np.abs(s).max()


def crash(rng):
    n = int(1.8 * SR)
    x = filt(rng.standard_normal(n), "highpass", 4500) * env(n, 0.001, 0.55)
    return x / np.abs(x).max()


def impact(rng):
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(28 + 50 * np.exp(-t / 0.15)) / SR) * env(n, 0.001, 0.6)
    hit = filt(rng.standard_normal(n), "lowpass", 900) * env(n, 0.0005, 0.12)
    x = np.tanh(2.5 * (boom + 0.6 * hit))
    return x / np.abs(x).max()


def reverb_ir(decay, seed):
    rng = np.random.default_rng(seed)
    n = int(decay * SR)
    t = np.arange(n) / SR
    ir = filt(rng.standard_normal(n), "lowpass", 6000) * np.exp(-t * 6.9 / decay)
    ir[: int(0.012 * SR)] = 0                                 # pre-delay
    return ir / np.sqrt((ir ** 2).sum())


# ================================================================ arrangement
def build(src, bpm=140, seed=11):
    rng = np.random.default_rng(seed)
    meh = get_meh(src)
    f0 = estimate_f0(meh)
    midi = round(69 + 12 * np.log2(f0 / 440))
    root = 440 * 2 ** ((midi - 69) / 12)
    while root > 75:
        root /= 2
    print(f"meh: {len(meh)/SR:.2f}s, key {NOTE_NAMES[midi % 12]} minor, bass root {root:.1f} Hz")

    voc = Vocoder(meh)
    beat = 60 / bpm
    step = beat / 4
    bar = 4 * beat
    n_bars = 35
    total = int((n_bars * bar + 3) * SR)
    drums = np.zeros((total, 2))
    fx = np.zeros((total, 2))
    bass = np.zeros(total)
    vox = np.zeros(total)          # dry lead "meh"
    send = np.zeros(total)         # delay/reverb send
    kicks = []

    def put(buf, clip, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i < 0 or i >= len(buf):
            return
        clip = np.asarray(clip)[: len(buf) - i]
        if buf.ndim == 1:
            buf[i:i + len(clip)] += gain * clip
        else:
            th = (pan + 1) * np.pi / 4
            buf[i:i + len(clip), 0] += gain * np.cos(th) * np.sqrt(2) * clip
            buf[i:i + len(clip), 1] += gain * np.sin(th) * np.sqrt(2) * clip

    KICK, SNARE, CRASH, IMPACT = hard_kick(rng), hard_snare(meh, rng), crash(rng), impact(rng)
    HAT_C, HAT_O = make_hats(meh, rng)

    def kick(t, g=1.0):
        put(drums, KICK, t, g); kicks.append(t)

    # ---- vocal "meh" variants
    MEH = meh
    DEMON = fade(pitch_shift(meh, -7), 0.003, 0.08)      # deep, slow "MEEEH"
    DEMON12 = fade(pitch_shift(meh, -12)[: int(8 * step * SR)], 0.003, 0.1)
    CHIP = fade(pitch_shift(meh, 12), 0.002, 0.02)        # high, fast "meh"

    def frag(length_s, semis=0.0):
        """first `length_s` of the meh, i.e. the "m-" for stutters"""
        return fade(pitch_shift(meh, semis)[: int(length_s * SR)], 0.001, 0.006)

    def say(clip, t, g=1.0, wet=0.25):
        put(vox, clip, t, g)
        put(send, clip, t, wet)

    def say_before(clip, t_end, g=1.0, wet=0.3):
        """place a clip so it ENDS exactly on t_end (e.g. the drop downbeat)"""
        say(clip, t_end - len(clip) / SR, g, wet)

    # ---- bass hit cache
    cache = {}

    def bass_at(t, steps, kind, semi, rate=2, g=1.0):
        key = (kind, semi, steps, rate)
        if key not in cache:
            cache[key] = bass_hit(voc, kind, root * 2 ** (semi / 12), steps * step, beat, rate, rng)
        put(bass, cache[key], t, g)

    # ---- patterns (16th steps): (step, length, kind, semitone, wobble rate per beat)
    # "A" bars leave steps 8-15 free so the dry "MEH" lands with the snare.
    D1 = [
        [(0, 3, "talk", 0), (3, 1, "talk", 0), (4, 2, "growl", 0, 4), (6, 2, "talk", 3)],
        [(0, 2, "talk", 0), (2, 2, "talk", 12), (4, 4, "wub", 0, 2), (8, 2, "talk", -4),
         (10, 2, "talk", -4), (12, 4, "growl", -2, 4)],
        [(0, 1, "talk", 0), (1, 1, "talk", 0), (2, 2, "talk", 7), (4, 4, "wub", -2, 4)],
        [(0, 3, "talk", 0), (3, 1, "talk", 0), (4, 2, "growl", 3, 6), (6, 2, "talk", 0),
         (8, 4, "wub", -4, 3), (12, 1, "talk", 12), (13, 1, "talk", 7), (14, 1, "talk", 3),
         (15, 1, "talk", 0)],
    ]
    D2 = [
        [(0, 2, "growl", 0, 8), (2, 1, "talk", 0), (3, 1, "talk", 0), (4, 1, "talk", 12),
         (5, 1, "talk", 12), (6, 2, "talk", 0)],
        [(0, 4, "wub", 0, 6), (4, 2, "talk", 3), (6, 2, "talk", -2), (8, 2, "growl", -4, 8),
         (10, 2, "talk", -4), (12, 4, "dive", -2)],
        [(0, 1, "talk", 0), (1, 1, "talk", 0), (2, 1, "talk", 0), (3, 1, "talk", 0),
         (4, 2, "growl", 7, 8), (6, 2, "dive", 0)],
        [(0, 2, "talk", 0), (2, 2, "talk", 0), (4, 1, "talk", 12), (5, 1, "talk", 7),
         (6, 1, "talk", 3), (7, 1, "talk", 0), (8, 4, "growl", -4, 8), (12, 4, "dive", -2)],
    ]

    def play_pattern(t0, pat, g=1.0):
        for ev in pat:
            s, ln, kind, semi = ev[:4]
            bass_at(t0 + s * step, ln, kind, semi, ev[4] if len(ev) > 4 else 2, g)

    def drop_drums(t0, k, hard):
        kick(t0)
        put(drums, SNARE, t0 + 8 * step, 1.0)
        extra = [10] if k % 2 == 0 else [6, 14]
        if hard:
            extra += [3, 11]
        for s in extra:
            kick(t0 + s * step, 0.85)
        for s in range(0, 16, 1 if hard else 2):
            put(drums, HAT_C, t0 + s * step + rng.normal(0, 0.002),
                (0.35 if s % 4 == 2 else 0.22), pan=0.35 if s % 2 else -0.25)
        if k % 4 == 0:
            put(drums, CRASH, t0, 0.55, pan=-0.3)
        if k % 4 == 3:                                         # triplet snare fill
            for h in range(6):
                put(drums, SNARE, t0 + 12 * step + h * beat / 6, 0.3 + 0.1 * h)
        put(drums, HAT_O, t0 + 14 * step, 0.3, pan=0.4)

    def build_up(t_start, nbars, roll_gain=1.0):
        """stuttering meh + snare roll + riser, ending with "MEH" on the downbeat"""
        for b in range(nbars):
            t0 = t_start + b * bar
            frac0 = b / nbars
            divs = [4, 8, 16, 32] if nbars >= 4 else [16, 32]  # notes per bar
            div = divs[min(len(divs) - 1, int(len(divs) * frac0))]
            sub = 16 / div                                       # spacing in 16ths
            hits = int(16 / sub) if b < nbars - 1 else int(8 / sub)   # last bar: stop halfway
            for h in range(hits):
                t = t0 + h * sub * step
                fr = (b + h * sub / 16) / nbars
                put(drums, filt(SNARE, "highpass", 150 + 2500 * fr), t,
                    roll_gain * (0.25 + 0.6 * fr))
                say(frag(sub * step * 0.95, 7 * fr), t, 0.9, wet=0.1)
            if b < nbars - 1:
                for q in range(4):
                    kick(t0 + q * beat, 0.6 + 0.3 * frac0)
        put(fx, noise_riser(int(nbars * bar * SR), rng), t_start, 0.45)
        say_before(MEH, t_start + nbars * bar, 1.25, wet=0.5)      # MEH -> drop

    # ================================================================ sections
    # intro (bars 0-3)
    n_int = int(4 * bar * SR)
    drone = np.sin(2 * np.pi * root / 2 * np.arange(n_int) / SR) * np.sin(np.linspace(0, np.pi, n_int)) ** 0.5
    put(bass, drone, 0, 0.7)
    for b in range(4):
        t0 = b * bar
        say(MEH, t0, 1.1, wet=0.7)
        if b % 2:
            say(MEH, t0 + 8 * step, 0.9, wet=0.7)
        if b >= 2:
            pad = bass_hit(voc, "talk", root * 2, 2 * bar if b == 2 else bar, beat, 2, rng)
            if b == 2:
                put(bass, filt(pad, "lowpass", 700), t0, 0.55)
            for s in range(0, 16, 2):
                put(drums, HAT_C, t0 + s * step, 0.22, pan=0.3 if s % 4 else -0.3)
            kick(t0, 0.7)
    put(fx, IMPACT, 0, 0.6)

    # build 1 (bars 4-7)
    build_up(4 * bar, 4)

    # drop 1 (bars 8-19)
    for k in range(12):
        t0 = (8 + k) * bar
        if k == 0:
            put(fx, IMPACT, t0, 0.8)
        drop_drums(t0, k, hard=False)
        play_pattern(t0, D1[k % 4])
        if k % 2 == 0:
            say(MEH, t0 + 8 * step, 1.25)                      # MEH on the snare
        elif k % 4 == 1:
            say(CHIP, t0 + 8 * step, 0.9)
        else:
            say(DEMON12, t0 + 8 * step, 1.1)

    # break + build 2 (bars 20-23)
    t0 = 20 * bar
    put(fx, IMPACT, t0, 0.8)
    put(drums, CRASH, t0, 0.6)
    say(MEH, t0, 1.2, wet=1.0)
    say(DEMON, t0 + 2 * beat, 1.0, wet=0.8)
    long_meh = bass_hit(voc, "talk", root, 2 * bar, beat, 2, rng)
    put(bass, filt(long_meh, "lowpass", 1200), t0, 0.7)
    say(MEH, t0 + bar + 8 * step, 1.1, wet=0.9)
    for s in range(0, 32, 4):
        put(drums, HAT_C, t0 + s * step, 0.2)
    build_up(22 * bar, 2, roll_gain=1.1)

    # drop 2 (bars 24-33)
    for k in range(10):
        t0 = (24 + k) * bar
        if k == 0:
            put(fx, IMPACT, t0, 0.9)
        drop_drums(t0, k, hard=True)
        play_pattern(t0, D2[k % 4])
        v = k % 4
        if v == 0:                                             # m-m-MEH
            say(frag(step * 0.9), t0 + 6 * step, 1.0)
            say(frag(step * 0.9), t0 + 7 * step, 1.0)
            say(MEH, t0 + 8 * step, 1.3)
        elif v == 1:
            say(CHIP, t0 + 12 * step, 0.95)
            say(MEH, t0, 0.9, wet=0.1)
        elif v == 2:
            say(DEMON, t0 + 8 * step, 1.15)
        else:
            say(MEH, t0, 1.0)
            say(MEH, t0 + 8 * step, 1.2)
        if k == 9:                                             # last bar: m-m-m-m 32nds
            for h in range(8):
                say(frag(step * 0.45, h), t0 + 12 * step + h * step / 2, 0.9, wet=0.05)

    # ending (bar 34)
    t0 = 34 * bar
    put(fx, IMPACT, t0, 1.0)
    put(drums, CRASH, t0, 0.7)
    kick(t0, 1.0)
    bass_at(t0, 4, "dive", 0, g=0.9)
    say(DEMON, t0, 1.3, wet=1.0)

    # ================================================================ mix
    # bass ducks under kicks...
    duck = np.ones(total)
    d = 1 - 0.75 * env(int(0.2 * SR), 0.0, 0.06)
    for kt in kicks:
        i = int(kt * SR)
        seg = duck[i:i + len(d)]
        duck[i:i + len(d)] = np.minimum(seg, d[: len(seg)])
    # ...and under the vocal, so the "meh" always cuts through
    venv = sosfilt(butter(2, 12, "lowpass", fs=SR, output="sos"), np.abs(vox))
    venv = np.clip(venv / (np.percentile(venv[venv > 1e-4], 95) + 1e-9), 0, 1)
    bass *= duck * (1 - 0.55 * venv)

    # bass: mono lows, Haas-widened mids/highs
    lo = filt(bass, "lowpass", 150)
    hi = bass - lo
    sh = int(0.011 * SR)
    bass_st = np.column_stack([lo + hi, lo + 0.95 * np.concatenate([np.zeros(sh), hi[:-sh]])])

    # vocal fx: ping-pong 3/16 delay into a big stereo reverb
    dl = np.zeros((total, 2))
    dt = int(3 * step * SR)
    for k in range(1, 6):
        dl[k * dt:, k % 2] += 0.5 ** k * send[: total - k * dt]
    wet = np.column_stack([
        fftconvolve(send + dl[:, 0], reverb_ir(2.2, 1))[:total],
        fftconvolve(send + dl[:, 1], reverb_ir(2.2, 2))[:total]])
    vox_fx = dl + 0.9 * wet
    vox_st = np.column_stack([vox, vox])

    mix = 0.95 * drums + 0.8 * bass_st + 1.25 * vox_st + 0.45 * vox_fx + 0.6 * fx
    mix = np.column_stack([filt(mix[:, c], "highpass", 30, 2) for c in range(2)])
    mix /= np.abs(mix).max()
    mix = np.tanh(2.2 * mix) / np.tanh(2.2)                    # slam it
    mix = mix / np.abs(mix).max() * 0.97

    end = int(60.0 * SR)
    mix = mix[:end]
    f = int(0.6 * SR)
    mix[-f:] *= np.linspace(1, 0, f)[:, None]
    stems = {"vox": vox[:end], "bass": bass[:end], "drums": drums[:end, 0]}
    return mix, stems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video or audio file")
    ap.add_argument("-o", "--output", default="dubstep_hard.wav")
    ap.add_argument("--bpm", type=float, default=140)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    mix, _ = build(load_audio(a.input), a.bpm, a.seed)
    sf.write(a.output, mix, SR, subtype="PCM_16")
    print(f"wrote {a.output} ({len(mix)/SR:.1f}s)")


if __name__ == "__main__":
    main()

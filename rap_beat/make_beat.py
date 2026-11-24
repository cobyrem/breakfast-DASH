#!/usr/bin/env python3
"""
make_beat.py - turn the audio from a video clip into a rap (trap) beat.

How it works
  1. Pull the audio out of the video with ffmpeg.
  2. Find the loud part of the clip and use it as the "sample".
  3. Estimate the sample's pitch so the 808 bass is in the same key.
  4. Build the drums partly from the sample itself (its attack becomes
     the snare/kick click, its high frequencies become the hi-hats).
  5. Chop the sample into pieces and replay them at different pitches as
     a melody, plus a reversed swell.
  6. Arrange 16 bars (intro / verse / hook), apply sidechain ducking,
     soft-clip and normalize, then write a WAV.

Usage
  python make_beat.py INPUT_VIDEO [-o beat.wav] [--bpm 140] [--bars 16] [--seed 7]

Requires: ffmpeg on PATH, and `pip install numpy scipy soundfile`.
"""
import argparse
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100


# ---------------------------------------------------------------- loading
def load_audio(path):
    """Decode any audio/video file to mono float32 at SR via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def loud_region(x, thresh_ratio=0.15, win=1024):
    """Return the span of x whose RMS is above a fraction of the max RMS."""
    n = len(x) // win
    rms = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1))
    on = np.where(rms > rms.max() * thresh_ratio)[0]
    start = max(on[0] - 1, 0) * win
    end = min(on[-1] + 2, n) * win
    seg = x[start:end].copy()
    return fade(seg / (np.abs(seg).max() + 1e-9), 0.005, 0.03)


# ---------------------------------------------------------------- DSP helpers
def fade(x, fin=0.002, fout=0.01):
    x = x.copy()
    a, b = int(fin * SR), int(fout * SR)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def filt(x, kind, cutoff, order=4):
    return sosfilt(butter(order, cutoff, btype=kind, fs=SR, output="sos"), x)


def pitch_shift(x, semitones):
    """Old-school sampler pitch shift: resample (changes length too)."""
    rate = 2 ** (semitones / 12)
    idx = np.arange(0, len(x) - 1, rate)
    return np.interp(idx, np.arange(len(x)), x)


def env(n, attack, decay):
    t = np.arange(n) / SR
    e = np.exp(-t / decay)
    a = int(attack * SR)
    if a:
        e[:a] *= np.linspace(0, 1, a)
    return e


def estimate_f0(x, fmin=60, fmax=600):
    """Autocorrelation pitch estimate over the loudest 300 ms."""
    w = int(0.3 * SR)
    energy = np.convolve(x ** 2, np.ones(w), "valid")
    c = int(np.argmax(energy))
    seg = x[c:c + w] * np.hanning(w)
    ac = np.fft.irfft(np.abs(np.fft.rfft(seg, 2 * w)) ** 2)[:w]
    lo, hi = int(SR / fmax), int(SR / fmin)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SR / lag


def onsets(x, hop=512, min_gap=0.08):
    """Simple spectral-flux onset detector -> sample indices for chopping."""
    n = 2048
    frames = [np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n)))
              for i in range(0, len(x) - n, hop)]
    S = np.array(frames)
    flux = np.maximum(np.diff(S, axis=0), 0).sum(axis=1)
    flux /= flux.max() + 1e-9
    thr = np.convolve(flux, np.ones(8) / 8, "same") + 0.1
    out, last = [0], -1e9
    for i in range(1, len(flux) - 1):
        if flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            pos = (i + 1) * hop
            if pos - last > min_gap * SR and pos - out[-1] > min_gap * SR:
                out.append(pos)
                last = pos
    return out


# ---------------------------------------------------------------- instruments
def make_kick(sample):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    freq = 45 + 110 * np.exp(-t / 0.03)               # pitch sweep 155 -> 45 Hz
    body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * env(n, 0.001, 0.18)
    click = filt(sample[: int(0.02 * SR)], "lowpass", 3000)  # sample's attack
    click = np.pad(click, (0, n - len(click))) * env(n, 0, 0.008)
    return np.tanh(1.5 * (body + 0.5 * click))


def make_snare(sample, rng):
    n = int(0.3 * SR)
    noise = filt(rng.standard_normal(n), "bandpass", [1500, 9000]) * env(n, 0.001, 0.07)
    tone = np.sin(2 * np.pi * 190 * np.arange(n) / SR) * env(n, 0.001, 0.05)
    # chunk of the sample, pitched down & band-passed -> gives it "character"
    smp = filt(pitch_shift(sample, -5)[:n], "highpass", 700)
    smp = np.pad(smp, (0, n - len(smp))) * env(n, 0.001, 0.09)
    s = 0.7 * noise + 0.4 * tone + 0.6 * smp
    return s / np.abs(s).max()


def make_hats(sample, rng):
    """Closed + open hats from the sample's top end (noise if it's too dull)."""
    air = filt(sample, "highpass", 7000)
    if np.abs(air).max() < 0.02:
        air = filt(rng.standard_normal(len(sample)), "highpass", 7000)
    air /= np.abs(air).max() + 1e-9
    n_c, n_o = int(0.05 * SR), int(0.25 * SR)
    air = np.tile(air, 1 + n_o // len(air) + 1)
    closed = air[:n_c] * env(n_c, 0.0005, 0.012)
    opened = air[:n_o] * env(n_o, 0.0005, 0.08)
    return closed, opened


def make_808(freq, length, glide_from=None):
    n = int(length * SR)
    f = np.full(n, freq)
    if glide_from:
        g = int(0.06 * SR)
        f[:g] = np.geomspace(glide_from, freq, g)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * np.cumsum(f) / SR)
    e = env(n, 0.003, 0.9) * np.clip((length - t) / 0.02, 0, 1)
    return np.tanh(2.2 * s) * e


# ---------------------------------------------------------------- build
def build_beat(src, bpm=140, bars=16, seed=7):
    rng = np.random.default_rng(seed)
    sample = loud_region(src)
    f0 = estimate_f0(sample)
    midi = 69 + 12 * np.log2(f0 / 440)
    root = 440 * 2 ** ((round(midi) - 69) / 12)      # snap sample pitch to a note
    bass_root = root
    while bass_root > 80:                            # 808 lives around 40-80 Hz
        bass_root /= 2
    print(f"sample: {len(sample)/SR:.2f}s, pitch ~{f0:.0f} Hz "
          f"(note {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][round(midi)%12]}), "
          f"808 root {bass_root:.1f} Hz")

    step = 60 / bpm / 4                              # one 16th note in seconds
    bar_len = 16 * step
    total = int((bars * bar_len + 2) * SR)
    drums = np.zeros(total)
    music = np.zeros(total)
    bass = np.zeros(total)
    kick_times = []

    def put(buf, clip, t, gain=1.0):
        i = int(t * SR)
        if i >= len(buf):
            return
        clip = clip[: len(buf) - i]
        buf[i:i + len(clip)] += gain * clip

    kick = make_kick(sample)
    snare = make_snare(sample, rng)
    hat_c, hat_o = make_hats(sample, rng)

    # chops: slice the sample at its onsets (fallback: even slices)
    cuts = onsets(sample)
    if len(cuts) < 4:
        cuts = list(np.linspace(0, len(sample), 5, dtype=int)[:-1])
    cuts.append(len(sample))
    chops = [fade(sample[a:b], 0.003, 0.02) for a, b in zip(cuts, cuts[1:])
             if b - a > 0.05 * SR]
    print(f"{len(chops)} chops from onsets")

    # natural minor scale degrees (semitones) for the chop melody & 808
    melody = [0, 3, 7, 5, 3, -2, 0, -4]               # one note per half-bar
    bassline = [0, 0, -4, -2]                         # one note per bar (4-bar loop)
    rev = fade(pitch_shift(sample, -12)[::-1], 0.3, 0.01)

    # patterns on a 16-step grid
    kick_pat = [1,0,0,0, 0,0,0,1, 0,0,1,0, 0,0,0,0]
    kick_pat2 = [1,0,0,0, 0,0,0,0, 1,0,1,0, 0,0,0,1]
    snare_pat = [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]  # trap: snare on beat 3

    for bar in range(bars):
        t0 = bar * bar_len
        intro = bar < 4
        hook = 8 <= bar < 16 and bar % 8 >= 4
        last_of_8 = bar % 8 == 7

        # --- sample chops (always on, filtered in the intro)
        for half in range(2):
            deg = melody[(bar * 2 + half) % len(melody)]
            chop = chops[(bar * 2 + half) % len(chops)]
            c = pitch_shift(chop, deg)
            if intro:
                c = filt(c, "lowpass", 900 + 400 * bar)
            put(music, c, t0 + half * 8 * step, 1.6 if intro else 0.55)
            # stutter the chop on 16ths in the hook
            if hook and half == 1:
                stut = fade(c[: int(step * SR)], 0.001, 0.005)
                for k in range(1, 4):
                    put(music, stut, t0 + (8 + 2 * k) * step, 0.35)
        if last_of_8:   # reversed swell leading into the next section
            put(music, rev[-int(bar_len * SR):], t0 + bar_len - min(len(rev), int(bar_len * SR)) / SR, 0.5)

        if intro:
            # intro: only a soft hat so the sample breathes
            for s in range(0, 16, 4):
                put(drums, hat_c, t0 + s * step, 0.5)
            continue

        # --- drums
        kp = kick_pat2 if bar % 2 else kick_pat
        for s in range(16):
            if kp[s]:
                put(drums, kick, t0 + s * step, 0.9)
                kick_times.append(t0 + s * step)
            if snare_pat[s]:
                put(drums, snare, t0 + s * step, 0.75)
        # hats: 8ths with 16th/32nd rolls; denser in the hook
        for s in range(0, 16, 1 if hook else 2):
            g = 0.32 if s % 4 == 0 else 0.22
            put(drums, hat_c, t0 + s * step + rng.normal(0, 0.003), g)
        roll_at = 12 if bar % 2 else 6
        for k in range(4 if not last_of_8 else 8):          # triplet-ish roll
            put(drums, hat_c, t0 + roll_at * step + k * step / 2, 0.18 + 0.03 * k)
        if bar % 4 == 3:
            put(drums, hat_o, t0 + 14 * step, 0.25)

        # --- 808 (follows the kick, in key with the sample)
        deg = bassline[bar % len(bassline)]
        f = bass_root * 2 ** (deg / 12)
        hits = [s for s in range(16) if kp[s]]
        for j, s in enumerate(hits):
            nxt = hits[j + 1] if j + 1 < len(hits) else 16
            glide = f * 2 if (j == len(hits) - 1 and bar % 2) else None
            put(bass, make_808(f, (nxt - s) * step, glide), t0 + s * step, 0.8)

    # --- sidechain: duck the sample under each kick
    duck = np.ones(total)
    d = 1 - 0.6 * env(int(0.25 * SR), 0.0, 0.08)
    for kt in kick_times:
        i = int(kt * SR)
        seg = duck[i:i + len(d)]
        duck[i:i + len(d)] = np.minimum(seg, d[: len(seg)])
    music *= duck

    mix = 0.9 * drums + 0.75 * music + 0.8 * bass
    mix = filt(mix, "highpass", 25, order=2)
    mix = np.tanh(1.3 * mix / (np.abs(mix).max() + 1e-9))   # soft clip / glue
    mix = mix / np.abs(mix).max() * 0.95
    end = int(bars * bar_len * SR + 1.5 * SR)
    return fade(mix[:end], 0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video or audio file")
    ap.add_argument("-o", "--output", default="beat.wav")
    ap.add_argument("--bpm", type=float, default=140)
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    src = load_audio(a.input)
    beat = build_beat(src, a.bpm, a.bars, a.seed)
    sf.write(a.output, np.column_stack([beat, beat]), SR, subtype="PCM_16")
    print(f"wrote {a.output} ({len(beat)/SR:.1f}s @ {a.bpm:g} BPM)")


if __name__ == "__main__":
    main()

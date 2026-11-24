#!/usr/bin/env python3
"""
make_dubstep.py - turn the audio from a video clip into a 1-minute dubstep track.

140 BPM half-time. Structure (35 bars = 60 s):
  bars  1-8   intro     filtered sample chops, sub swell, light hats
  bars  9-12  build     accelerating snare roll, noise riser, pitch-rising chop
  bars 13-20  drop A    half-time drums, wobble bass (1/4 -> 1/8 wobbles)
  bars 21-28  drop B    faster wobbles (1/16, triplets), vocal stutters
  bars 29-35  outro     wobble fades out, reversed sample, chops ring out

The wobble bass uses a single pitch cycle cut from your sample as its waveform
(mixed with detuned saws), run through a resonant low-pass filter whose cutoff
is swept by a tempo-synced LFO. That gives the classic dubstep "wub".

Usage
  python make_dubstep.py INPUT_VIDEO [-o dubstep.wav] [--bpm 140] [--seconds 60]

Requires: ffmpeg on PATH, and `pip install numpy scipy soundfile`.
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_beat import (SR, load_audio, loud_region, fade, filt, pitch_shift,  # noqa: E402
                       env, estimate_f0, onsets, make_kick, make_snare, make_hats)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ---------------------------------------------------------------- synthesis
def wavetable_from_sample(sample, f0):
    """Cut one pitch period out of the loudest part of the sample."""
    period = int(round(SR / f0))
    w = int(0.1 * SR)
    energy = np.convolve(sample ** 2, np.ones(w), "valid")
    c = int(np.argmax(energy)) + w // 2
    # start on a rising zero crossing so the loop is click-free
    z = np.where((sample[c:c + 2 * period][:-1] <= 0) & (sample[c:c + 2 * period][1:] > 0))[0]
    c += int(z[0]) if len(z) else 0
    cyc = sample[c:c + period].copy()
    cyc -= cyc.mean()
    return cyc / (np.abs(cyc).max() + 1e-9)


def osc_table(table, freq):
    """Play a single-cycle wavetable at a per-sample frequency array."""
    phase = np.cumsum(freq / SR) % 1.0
    idx = phase * len(table)
    i0 = idx.astype(int)
    frac = idx - i0
    return table[i0] * (1 - frac) + table[(i0 + 1) % len(table)] * frac


def saw(freq):
    phase = np.cumsum(freq / SR) % 1.0
    return 2 * phase - 1


def svf_lowpass(x, cutoff, q=0.25):
    """State-variable low-pass with a per-sample cutoff array (Chamberlin)."""
    f = 2 * np.sin(np.pi * np.clip(cutoff, 20, SR / 6) / SR)
    low = band = 0.0
    out = np.empty_like(x)
    for i in range(len(x)):
        low += f[i] * band
        high = x[i] - low - q * band
        band += f[i] * high
        out[i] = low
    return out


def wobble_bass(freqs, rates, table, beat_len, depth=1.0):
    """
    freqs: per-sample pitch (Hz); rates: per-sample LFO rate in wobbles per beat.
    """
    lfo_phase = np.cumsum(rates / (beat_len * SR)) % 1.0
    lfo = 0.5 - 0.5 * np.cos(2 * np.pi * lfo_phase)            # 0..1
    cutoff = 90 * (1 + 40 * depth * lfo ** 1.6)                # ~90 Hz -> ~3.6 kHz
    raw = (0.6 * osc_table(table, freqs)
           + 0.35 * saw(freqs * 1.005) + 0.35 * saw(freqs * 0.995))
    wob = svf_lowpass(raw, cutoff, q=0.18)
    wob = np.tanh(3.0 * wob)                                    # grit
    sub = np.sin(2 * np.pi * np.cumsum(freqs / 2) / SR)        # clean sub an octave down
    return 0.8 * wob + 0.55 * sub


def noise_riser(n, rng):
    x = rng.standard_normal(n)
    cutoff = np.geomspace(300, 12000, n)
    y = x - svf_lowpass(x, cutoff, q=0.7)                       # rising high-pass sweep
    return y * np.linspace(0, 1, n) ** 2


# ---------------------------------------------------------------- arrangement
def build_dubstep(src, bpm=140, seconds=60, seed=3):
    rng = np.random.default_rng(seed)
    sample = loud_region(src)
    f0 = estimate_f0(sample)
    midi = round(69 + 12 * np.log2(f0 / 440))
    root = 440 * 2 ** ((midi - 69) / 12)
    bass_root = root
    while bass_root > 75:                       # wobble root around 38-75 Hz
        bass_root /= 2
    print(f"sample {len(sample)/SR:.2f}s, key {NOTE_NAMES[midi % 12]} minor, "
          f"bass root {bass_root:.1f} Hz")

    beat = 60 / bpm
    step = beat / 4
    bar_len = 4 * beat
    bars = int(round(seconds / bar_len))
    total = int((bars * bar_len + 3) * SR)
    drums, music, bass, fx = (np.zeros(total) for _ in range(4))
    kick_times = []

    def put(buf, clip, t, gain=1.0):
        i = int(t * SR)
        if 0 <= i < len(buf):
            clip = clip[: len(buf) - i]
            buf[i:i + len(clip)] += gain * clip

    kick = make_kick(sample)
    snare = make_snare(sample, rng)
    snare = np.tanh(1.8 * snare)                  # punchier for dubstep
    hat_c, hat_o = make_hats(sample, rng)
    table = wavetable_from_sample(sample, f0)

    cuts = onsets(sample)
    if len(cuts) < 4:
        cuts = list(np.linspace(0, len(sample), 5, dtype=int)[:-1])
    cuts.append(len(sample))
    chops = [fade(sample[a:b], 0.003, 0.02) for a, b in zip(cuts, cuts[1:])
             if b - a > 0.05 * SR]
    melody = [0, 3, 7, 3, 10, 7, 5, 3]

    # section boundaries (in bars), scaled if --seconds changes the length
    intro_end = max(2, round(bars * 8 / 35))
    build_end = intro_end + max(2, round(bars * 4 / 35))
    dropA_end = build_end + max(2, round(bars * 8 / 35))
    dropB_end = dropA_end + max(2, round(bars * 8 / 35))

    # per-sample automation for the wobble bass
    bass_freq = np.full(total, bass_root)
    bass_rate = np.zeros(total)
    bass_gain = np.zeros(total)
    # bass note per bar: i, i, bVI, bVII  (in semitones)
    prog = [0, 0, -4, -2]
    # wobble rates (wobbles per beat) per half-bar
    rates_A = [1, 1, 2, 2, 1, 2, 4, 2]
    rates_B = [4, 2, 3, 4, 2, 4, 6, 8]

    for bar in range(bars):
        t0 = bar * bar_len
        i0, i1 = int(t0 * SR), int((t0 + bar_len) * SR)

        if bar < intro_end:
            # ---------- intro
            prog_i = bar % 4
            for half in range(2):
                c = pitch_shift(chops[(bar * 2 + half) % len(chops)],
                                melody[(bar * 2 + half) % len(melody)])
                c = filt(c, "lowpass", 600 + 300 * bar)
                put(music, c, t0 + half * 2 * beat, 1.4)
                put(music, filt(c, "lowpass", 500), t0 + half * 2 * beat + 3 * step, 0.5)  # echo
            n = i1 - i0                                  # sub swell
            sub = np.sin(2 * np.pi * np.cumsum(np.full(n, bass_root * 2 ** (prog[prog_i] / 12) / 2)) / SR)
            put(bass, sub * np.sin(np.linspace(0, np.pi, n)), t0, 0.5)
            if bar >= intro_end // 2:
                for s in range(2, 16, 4):
                    put(drums, hat_c, t0 + s * step, 0.35)
                put(drums, kick, t0, 0.7)

        elif bar < build_end:
            # ---------- build-up
            k = bar - intro_end
            nb = build_end - intro_end
            div = [4, 8, 16, 32][min(3, int(4 * k / nb))]   # 1/4 -> 1/32 rolls
            hits = div if div <= 16 else 32
            for h in range(hits):
                t = t0 + h * bar_len / hits
                prog_frac = (k + h / hits) / nb
                put(drums, filt(snare, "highpass", 200 + 1500 * prog_frac), t,
                    0.25 + 0.5 * prog_frac)
            c = pitch_shift(chops[k % len(chops)], 12 * k / nb)   # rising chop
            for h in range(4):
                put(music, c, t0 + h * beat, 0.8)
            if k == 0:
                put(fx, noise_riser(int(nb * bar_len * SR), rng), t0, 0.5)

        elif bar < dropB_end:
            # ---------- drops (half-time: kick on 1, snare on 3)
            dropB = bar >= dropA_end
            put(drums, kick, t0, 1.0); kick_times.append(t0)
            put(drums, snare, t0 + 2 * beat, 0.95)
            extra_k = [10] if bar % 2 == 0 else [6, 11]
            if dropB:
                extra_k += [3]
            for s in extra_k:
                put(drums, kick, t0 + s * step, 0.8); kick_times.append(t0 + s * step)
            for s in range(0, 16, 2):
                put(drums, hat_c, t0 + s * step + rng.normal(0, 0.002), 0.3 if s % 4 else 0.2)
            if bar % 4 == 3:
                for h in range(6):                        # triplet roll into next phrase
                    put(drums, snare, t0 + 3 * beat + h * beat / 6, 0.25 + 0.08 * h)
            put(drums, hat_o, t0 + 3.5 * beat, 0.3)

            # wobble automation
            rates = rates_B if dropB else rates_A
            k = bar - (dropA_end if dropB else build_end)
            for half in range(2):
                a = i0 + half * (i1 - i0) // 2
                b = i0 + (half + 1) * (i1 - i0) // 2
                bass_rate[a:b] = rates[(k * 2 + half) % len(rates)]
            bass_freq[i0:i1] = bass_root * 2 ** (prog[bar % 4] / 12)
            bass_gain[i0:i1] = 1.0
            # gap on the last 16th of each bar for punch
            bass_gain[i1 - int(step * SR):i1] = 0
            if dropB and bar % 2:                          # pitch dive at the end of odd bars
                g = int(beat * SR)
                bass_freq[i1 - g:i1] *= np.geomspace(1, 0.5, g)

            # sample on top: stabs in drop A, stutters in drop B
            c = chops[bar % len(chops)]
            if dropB:
                st = fade(pitch_shift(c, melody[bar % len(melody)])[: int(step * SR)], 0.001, 0.004)
                for s in (0, 1, 2, 3, 8, 9):
                    put(music, st, t0 + 2 * beat + s * step / 2 if s >= 8 else t0 + s * step, 0.6)
            else:
                put(music, pitch_shift(c, melody[bar % len(melody)] + 12), t0 + beat, 0.45)

        else:
            # ---------- outro
            k = bar - dropB_end
            no = bars - dropB_end
            fadeg = 1 - k / no
            if k == 0:
                rev = fade(pitch_shift(sample, -7)[::-1], 0.4, 0.01)
                put(music, rev, t0 + bar_len - len(rev) / SR, 0.8)
            put(drums, kick, t0, 0.8 * fadeg); kick_times.append(t0)
            if k < no - 1:
                put(drums, snare, t0 + 2 * beat, 0.7 * fadeg)
            for s in range(0, 16, 4):
                put(drums, hat_c, t0 + s * step, 0.25 * fadeg)
            c = filt(pitch_shift(chops[k % len(chops)], melody[k % len(melody)]),
                     "lowpass", max(400, 3000 * fadeg))
            put(music, c, t0, 1.2)
            put(music, c, t0 + 3 * step, 0.5)
            if k < 2:                                       # wobble tails off
                bass_rate[i0:i1] = 1
                bass_freq[i0:i1] = bass_root
                bass_gain[i0:i1] = np.linspace(1 - k / 2, 0.5 - k / 2, i1 - i0)

    # render the wobble bass once with all automation (skip silent regions)
    active = np.where(bass_gain > 0)[0]
    if len(active):
        a, b = active[0], active[-1] + 1
        g = bass_gain[a:b]
        g = np.convolve(g, np.ones(64) / 64, "same")       # de-click gain steps
        bass[a:b] += 0.85 * g * wobble_bass(bass_freq[a:b], bass_rate[a:b], table, beat)

    # sidechain: duck bass + music on kicks
    duck = np.ones(total)
    d = 1 - 0.7 * env(int(0.22 * SR), 0.0, 0.07)
    for kt in kick_times:
        i = int(kt * SR)
        seg = duck[i:i + len(d)]
        duck[i:i + len(d)] = np.minimum(seg, d[: len(seg)])
    bass *= duck
    music *= duck

    mix = 0.9 * drums + 0.7 * music + 0.85 * bass + 0.5 * fx
    mix = filt(mix, "highpass", 28, order=2)
    mix = np.tanh(1.5 * mix / (np.abs(mix).max() + 1e-9))
    mix = mix / np.abs(mix).max() * 0.95
    end = int(min(seconds, bars * bar_len + 1.5) * SR)
    return fade(mix[:end], 0, 1.5), bars


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video or audio file")
    ap.add_argument("-o", "--output", default="dubstep.wav")
    ap.add_argument("--bpm", type=float, default=140)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    track, bars = build_dubstep(load_audio(a.input), a.bpm, a.seconds, a.seed)
    sf.write(a.output, np.column_stack([track, track]), SR, subtype="PCM_16")
    print(f"wrote {a.output} ({len(track)/SR:.1f}s, {bars} bars @ {a.bpm:g} BPM)")


if __name__ == "__main__":
    main()

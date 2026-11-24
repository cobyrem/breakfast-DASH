#!/usr/bin/env python3
"""
make_opera.py - a 1-minute Mozart-style opera piece sung by the "meh"
from a video clip.

  bars  1-4   Orchestra plays the opening of Mozart's "Eine kleine Nachtmusik"
              (G major, strings in octaves + harpsichord)
  bars  5-6   The diva interrupts: "MEH!" - modulation to D minor
  bars  7-22  Aria in D minor (opera-seria style): driving repeated-8th
              cellos, Alberti-bass harpsichord, sustained strings; the
              soprano "meh" sings a classical melody with runs
  bars 23-28  Queen-of-the-Night-style coloratura: staccato arpeggios up
              to a high F6, cadential trill
  bars 29-33  Finale: choir of "MEH"s, timpani roll, final cadence

The soprano is your actual "meh", re-pitched with TD-PSOLA (pitch-synchronous
overlap-add), which changes the pitch while keeping the voice's character,
and stretched by looping the vowel so it can hold long operatic notes with
vibrato. The choir is your "meh" vocoded onto chords. Orchestra is
synthesized (additive harpsichord, detuned-saw string section, timpani)
and everything goes through a concert-hall reverb.

Usage
  python make_opera.py INPUT_VIDEO [-o opera.wav] [--bpm 132]

Requires: ffmpeg on PATH, and `pip install numpy scipy soundfile`.
Lives next to make_beat.py and make_dubstep_hard.py (reuses their helpers).
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_beat import SR, load_audio, loud_region, fade, filt, env  # noqa: E402
from make_dubstep_hard import Vocoder, reverb_ir  # noqa: E402

R = None  # rest


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ================================================================ the score
# soprano: one list per bar of (midi note or R, beats); each bar sums to 4
TRILL = [(77 if i % 2 == 0 else 76, 0.125) for i in range(24)]
SOPRANO = [
    [(R, 4)], [(R, 4)], [(R, 4)], [(R, 4)],                                   # 1-4
    [(R, 1), (83, 3)],                                                        # 5  MEH!
    [(81, 1), (79, 1), (76, 1), (73, 1)],                                     # 6
    [(81, 2), (77, 1), (74, 1)],                                              # 7  Dm
    [(81, 1.5), (82, 0.5), (81, 1), (R, 1)],                                  # 8  Dm
    [(86, 1), (82, 1), (79, 1), (74, 1)],                                     # 9  Gm
    [(81, 1.5), (79, 0.5), (77, 0.5), (76, 0.5), (R, 1)],                     # 10 A7
    [(74, 1), (77, 1), (81, 1), (86, 1)],                                     # 11 Dm
    [(86, 1.5), (84, 0.5), (82, 1), (77, 1)],                                 # 12 Bb
    [(79, .5), (82, .5), (86, 1), (84, .5), (82, .5), (81, .5), (79, .5)],    # 13 Gm
    [(81, 2), (73, 1), (R, 1)],                                               # 14 A7
    [(77, 1), (81, 1), (84, 2)],                                              # 15 F
    [(82, 1), (81, .5), (79, .5), (76, 1), (72, 1)],                          # 16 C7
    [(77, .25), (79, .25), (81, .25), (82, .25), (84, .25), (86, .25),
     (88, .25), (89, .25), (84, 1), (81, 1)],                                 # 17 F  run
    [(86, 1), (81, 1), (77, 1), (74, 1)],                                     # 18 Dm
    [(82, .5), (84, .5), (86, .5), (89, 1.5), (86, 1)],                       # 19 Bb
    [(82, 1), (79, 1), (74, 1), (79, 1)],                                     # 20 Gm
    [(81, 1), (85, 1), (88, 1), (R, 1)],                                      # 21 A7
    [(86, 3), (R, 1)],                                                        # 22 Dm
    [(n, .25) for n in (74, 77, 81, 86, 89, 86, 81, 77, 74, 77, 81, 86)]
    + [(89, 1)],                                                              # 23 coloratura
    [(n, .25) for n in (86, 82, 79, 82, 86, 89, 86, 82, 79, 82, 86, 89)]
    + [(86, 1)],                                                              # 24
    [(n, .25) for n in (76, 81, 85, 88, 85, 81, 76, 81, 85, 88, 85, 81)]
    + [(88, 1)],                                                              # 25
    [(89, .5), (R, .5), (89, .5), (R, .5), (86, 2)],                          # 26
    [(86, 2), (82, 1), (79, 1)],                                              # 27
    TRILL + [(74, .5), (76, .5)],                                             # 28 trill
    [(74, 1), (86, 3)],                                                       # 29 finale
    [(86, 2), (82, 2)],                                                       # 30
    [(81, 2), (85, 2)],                                                       # 31
    [(86, 4)],                                                                # 32
    [(86, 3), (R, 1)],                                                        # 33
]
STACCATO_BARS = {23, 24, 25, 26}       # 1-based

# Eine kleine Nachtmusik, K.525, opening (strings in octaves)
NACHTMUSIK = [
    [(67, 1), (R, .5), (62, .5), (67, 1), (R, .5), (62, .5)],
    [(67, .5), (62, .5), (67, .5), (71, .5), (74, 1), (R, 1)],
    [(72, 1), (R, .5), (69, .5), (72, 1), (R, .5), (69, .5)],
    [(72, .5), (69, .5), (66, .5), (69, .5), (62, 1), (R, 1)],
]

CHORDS = {  # root midi (octave 4), intervals
    "G": (55, [0, 4, 7]), "A7": (57, [0, 4, 7, 10]), "Dm": (62, [0, 3, 7]),
    "Gm": (55, [0, 3, 7]), "Bb": (58, [0, 4, 7]), "F": (53, [0, 4, 7]),
    "C7": (60, [0, 4, 7, 10]),
}
HARMONY = [None, None, None, None, "G", "A7", "Dm", "Dm", "Gm", "A7", "Dm", "Bb",
           "Gm", "A7", "F", "C7", "F", "Dm", "Bb", "Gm", "A7", "Dm", "Dm", "Gm",
           "A7", "Dm", "Gm", "A7", "Dm", "Gm", "A7", "Dm", "Dm"]
# choir "MEH" hits: bar -> list of (beat, beats)
CHOIR = {5: [(0, 1)], 22: [(0, 1.5), (2, 1.5)], 26: [(2, 0.9), (3, 0.9)],
         29: [(0, 2), (2, 2)], 30: [(0, 2), (2, 2)], 31: [(0, 2), (2, 2)],
         32: [(0, 2), (2, 2)], 33: [(0, 3)]}
TIMPANI_BARS = {5, 10, 14, 21, 22, 25, 26, 28}

for i, bar in enumerate(SOPRANO):
    assert abs(sum(d for _, d in bar) - 4) < 1e-9, f"bar {i+1} doesn't add to 4"


# ================================================================ soprano (TD-PSOLA)
class PsolaVoice:
    def __init__(self, x):
        self.x = x
        # --- pitch track (autocorrelation, 10 ms hop)
        hop, w = int(0.01 * SR), int(0.04 * SR)
        lo, hi = int(SR / 500), int(SR / 120)
        periods, rms = [], []
        for i in range(0, len(x) - w, hop):
            seg = x[i:i + w] * np.hanning(w)
            ac = np.correlate(seg, seg, "full")[w - 1:]
            lag = lo + int(np.argmax(ac[lo:hi]))
            periods.append(lag)
            rms.append(np.sqrt((seg ** 2).mean()))
        periods, rms = np.array(periods), np.array(rms)
        voiced = rms > 0.25 * rms.max()
        P0 = int(np.median(periods[voiced]))
        self.f0 = SR / P0

        def period_at(n):
            k = min(len(periods) - 1, max(0, n // hop))
            p = periods[k] if voiced[k] else P0
            return p if abs(p - P0) < 0.3 * P0 else P0

        # --- pitch marks: walk out from the loudest peak, snapping to local maxima
        start = int(np.argmax(rms)) * hop + w // 2
        start = start - P0 // 2 + int(np.argmax(x[start - P0 // 2:start + P0 // 2]))
        marks = [start]
        for direction in (1, -1):
            m = start
            while True:
                p = period_at(m)
                guess = m + direction * p
                a, b = guess - p // 4, guess + p // 4
                if a < p or b > len(x) - p:
                    break
                m = a + int(np.argmax(x[a:b]))
                marks.append(m)
        self.marks = np.array(sorted(marks))
        self.P = np.array([period_at(m) for m in self.marks])
        amp = np.array([np.sqrt((x[m - p:m + p] ** 2).mean()) for m, p in zip(self.marks, self.P)])
        # consonant "m" -> marks before the high-frequency (vowel) energy rises
        hf = filt(x, "bandpass", [700, 3000]) ** 2
        hf_at = np.array([hf[m - p:m + p].mean() for m, p in zip(self.marks, self.P)])
        loud = np.where(amp > 0.3 * amp.max())[0]
        self.first = loud[0]
        vowel_on = loud[np.argmax(hf_at[loud] > 0.4 * hf_at[loud].max())]
        self.vowel_on = max(vowel_on, self.first + 1)
        # steady part of the vowel to loop for sustained notes
        vl = loud[loud >= self.vowel_on]
        self.loop = vl[: max(4, int(len(vl) * 0.7))]
        self.amp = amp

    def grain(self, k):
        m, p = self.marks[k], self.P[k]
        g = self.x[m - p:m + p] * np.hanning(2 * p)
        return g / (self.amp[k] + 1e-9)

    def sing(self, midi, dur, staccato=False, with_consonant=True, rng=None):
        n = int(dur * SR)
        out = np.zeros(n + 4000)
        t = np.arange(n) / SR
        target = mtof(midi)
        # vibrato fades in on held notes; tiny scoop up at the start (operatic)
        vib_depth = 0.35 * np.clip((t - 0.15) / 0.25, 0, 1) if dur > 0.35 else 0 * t
        vib = vib_depth * np.sin(2 * np.pi * 5.6 * t + (rng.uniform(0, 6) if rng is not None else 0))
        scoop = -0.6 * np.exp(-t / 0.03) if dur > 0.2 and not staccato else 0 * t
        f = target * 2 ** ((vib + scoop) / 12)

        # Walk through the source recording at its natural speed (consonant
        # first, then ping-pong through the steady vowel). Each output period
        # reuses the nearest source grain, so high notes repeat the same
        # waveform cleanly instead of switching grains every cycle (= noise).
        lo_t = self.marks[self.loop[0]]
        hi_t = self.marks[self.loop[-1]]
        use_cons = with_consonant and not staccato
        src_t = float(self.marks[self.first] if use_cons else lo_t)
        direction = 1.0
        pos = 0.0
        while pos < n:
            k = int(np.argmin(np.abs(self.marks - src_t)))
            g = self.grain(k)
            fi = f[min(int(pos), n - 1)]
            a = int(pos) - len(g) // 2
            frac = pos - int(pos)
            if a >= 0 and frac > 1e-6:                     # fractional delay
                g = np.interp(np.arange(len(g) + 1) - frac, np.arange(len(g)), g, left=0, right=0)
            if a >= 0:
                out[a:a + len(g)] += g * (self.P[k] * 2 * fi / SR) ** -1
            dt = SR / fi
            pos += dt
            src_t += direction * dt
            if src_t > hi_t and src_t - direction * dt >= lo_t:
                direction = -1.0
            elif src_t < lo_t and direction < 0:
                direction = 1.0
        out = out[:n]
        e = np.ones(n)
        a = int((0.006 if staccato else 0.025) * SR)
        r = int((0.03 if staccato else 0.08) * SR)
        r = min(r, n // 2)
        e[:a] = np.linspace(0, 1, a)
        e[-r:] = np.linspace(1, 0, r)
        if staccato:
            e *= np.exp(-t / 0.12)
        return out * e


# ================================================================ orchestra
_TABLES = {}
TABLE_N = 4096


def bl_saw_table(f_max):
    """Band-limited sawtooth: only harmonics below ~15 kHz at the highest
    frequency it will be played at, so nothing aliases (no digital buzz)."""
    nh = max(1, int(15000 / f_max))
    if nh not in _TABLES:
        t = np.arange(TABLE_N) / TABLE_N
        k = np.arange(1, nh + 1)[:, None]
        tab = (np.sin(2 * np.pi * k * t) / k).sum(axis=0)
        _TABLES[nh] = tab / np.abs(tab).max()
    return _TABLES[nh]


def bl_osc(freq, phase0=0.0):
    """Play a band-limited saw at a per-sample frequency array."""
    tab = bl_saw_table(float(np.max(freq)))
    idx = ((np.cumsum(freq) / SR + phase0) % 1.0) * TABLE_N
    i0 = idx.astype(int)
    fr = idx - i0
    return tab[i0] * (1 - fr) + tab[(i0 + 1) % TABLE_N] * fr


def harpsichord(midi, dur):
    n = int((dur + 0.6) * SR)
    t = np.arange(n) / SR
    f = mtof(midi)
    x = np.zeros(n)
    for k in range(1, 16):
        fk = f * k * (1 + 0.0004 * k * k)
        if fk > SR / 2.2:
            break
        amp = (1 / k ** 0.7) * abs(np.sin(np.pi * k * 0.13))   # pluck position
        x += amp * np.sin(2 * np.pi * fk * t) * np.exp(-t * (1.2 + 0.9 * k) * (f / 400) ** 0.3)
    x *= env(n, 0.001, 10)
    x[int(dur * SR):] *= np.exp(-np.arange(n - int(dur * SR)) / SR / 0.06)  # damper
    return x / 4


def strings(midi, dur, rng, attack=0.08, bright=4000, trem=False):
    n = int((dur + 0.25) * SR)
    t = np.arange(n) / SR
    f0 = mtof(midi)
    x = np.zeros(n)
    for det in (-0.12, 0.0, 0.11):
        f = f0 * 2 ** ((det + 0.06 * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6))) / 12)
        x += bl_osc(f, rng.uniform())
    x = filt(filt(x, "lowpass", bright, 2), "highpass", 120, 2)
    e = np.minimum(1, t / attack) * np.where(t < dur, 1, np.exp(-(t - dur) / 0.08))
    if trem:
        e *= 0.8 + 0.2 * np.sin(2 * np.pi * 7 * t)       # smooth, no chopping
    return x * e / 3


def timpani(midi, dur=1.2, roll=False, rng=None):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = mtof(midi) * (1 + 0.03 * np.exp(-t / 0.05))
    x = (np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.4 * np.sin(2 * np.pi * np.cumsum(f * 1.5) / SR)
         + 0.25 * np.sin(2 * np.pi * np.cumsum(f * 1.98) / SR))
    thump = filt(np.random.default_rng(1).standard_normal(n), "lowpass", 400) * np.exp(-t / 0.02)
    if roll:
        hits = np.zeros(n)
        hits[::int(0.055 * SR)] = 1
        e = fftconvolve(hits, np.exp(-np.arange(int(0.1 * SR)) / SR / 0.04))[:n]
        return (x * 0.6 + thump) * e * np.linspace(0.4, 1, n) * 0.6
    return (x + thump) * np.exp(-t / 0.45)


# ================================================================ build
def build(src, bpm=132, seed=5):
    rng = np.random.default_rng(seed)
    voice_src = filt(loud_region(src, thresh_ratio=0.08), "highpass", 90)
    voice = PsolaVoice(voice_src)
    meh_choir = Vocoder(voice_src / np.abs(voice_src).max(), n_bands=32)
    print(f"meh: {len(voice_src)/SR:.2f}s, f0 {voice.f0:.0f} Hz, {len(voice.marks)} pitch marks, "
          f"vowel loop {len(voice.loop)} periods")

    beat = 60 / bpm
    bar = 4 * beat
    total = int((len(SOPRANO) * bar + 3) * SR)
    sop, orch, choir = np.zeros(total), np.zeros((total, 2)), np.zeros(total)

    def put(buf, clip, t, g=1.0, pan=0.0):
        i = int(round(t * SR))
        clip = clip[: max(0, len(buf) - i)]
        if buf.ndim == 1:
            buf[i:i + len(clip)] += g * clip
        else:
            th = (pan + 1) * np.pi / 4
            buf[i:i + len(clip), 0] += g * np.cos(th) * 1.414 * clip
            buf[i:i + len(clip), 1] += g * np.sin(th) * 1.414 * clip

    # --- soprano
    for b, notes in enumerate(SOPRANO):
        t = b * bar
        stacc = (b + 1) in STACCATO_BARS
        prev_rest = True
        for m, d in notes:
            if m is not None:
                dd = d * beat
                is_trill = (b + 1) == 28 and d == 0.125
                s = voice.sing(m, dd * (0.75 if stacc else 1.0) if not is_trill else dd * 1.05,
                               staccato=stacc or is_trill,
                               with_consonant=prev_rest or d >= 1, rng=rng)
                put(sop, s, t, 0.9 if is_trill else 1.0)
            prev_rest = m is None
            t += d * beat

    # --- Nachtmusik intro (bars 1-4): strings in octaves + harpsichord + cellos
    for b, notes in enumerate(NACHTMUSIK):
        t = b * bar
        for m, d in notes:
            if m is not None:
                dd = d * beat * 0.85
                put(orch, strings(m + 12, dd, rng, attack=0.015, bright=6000), t, 0.9, -0.4)
                put(orch, strings(m, dd, rng, attack=0.015), t, 0.8, -0.1)
                put(orch, strings(m - 12, dd, rng, attack=0.02, bright=2500), t, 0.8, 0.4)
                put(orch, harpsichord(m, dd), t, 0.5, 0.2)
                put(orch, harpsichord(m - 24, dd), t, 0.4, 0.3)
            t += d * beat

    # --- accompaniment from the harmony
    for b, name in enumerate(HARMONY):
        if name is None:
            continue
        bn = b + 1
        t0 = b * bar
        root, iv = CHORDS[name]
        pc = root % 12
        bass_note = 36 + (pc - 36) % 12                    # C2..B2
        alb_root = 48 + (pc - 48) % 12                      # C3..B3
        third, fifth = alb_root + iv[1], alb_root + iv[2]
        if bn in (5,):
            # tutti chord stab, then tremolo under the diva's "MEH"
            for m in [bass_note, bass_note + 12] + [root + i for i in iv] + [root + 12 + iv[1]]:
                put(orch, strings(m, 0.35, rng, attack=0.01, bright=5000), t0, 0.6, rng.uniform(-.5, .5))
            for m in [root + i for i in iv]:
                put(orch, strings(m, 3 * beat, rng, attack=0.3, trem=True), t0 + beat, 0.35, rng.uniform(-.5, .5))
            continue
        final = bn == len(HARMONY)
        # cellos/basses: driving repeated 8ths (opera-seria drama)
        for k in range(8 if not final else 1):
            dd = (0.42 if not final else 3) * beat
            put(orch, strings(bass_note, dd, rng, attack=0.01, bright=1800), t0 + k * beat / 2,
                0.75, 0.35)
            put(orch, strings(bass_note - 12, dd, rng, attack=0.01, bright=900), t0 + k * beat / 2,
                0.5, 0.35)
        # harpsichord Alberti bass: root-5th-3rd-5th in 8ths
        if not final:
            for k, m in enumerate([alb_root, fifth, third, fifth] * 2):
                put(orch, harpsichord(m, beat / 2), t0 + k * beat / 2, 0.55, -0.2)
        # upper strings: held chord (tremolo in the coloratura/finale)
        trem = bn >= 23
        for j, i in enumerate(iv):
            m = root + i + (12 if j == 0 else 0)
            put(orch, strings(m, (3 if final else 4) * beat, rng, attack=0.12 if not trem else 0.02,
                              trem=trem), t0, 0.28 if not trem else 0.34, -0.5 + j * 0.33)
        # timpani on tonic/dominant
        tim = 45 if pc in (9, 0, 4) else 38               # A2 or D2
        if bn in TIMPANI_BARS or final:
            put(orch, timpani(tim), t0, 0.7, 0.1)
        if bn >= 29 and not final:
            put(orch, timpani(tim, dur=bar, roll=True), t0, 0.5, 0.1)
        # choir "MEH"
        for beat_pos, ln in CHOIR.get(bn, []):
            dd = ln * beat
            n = int(dd * SR)
            tt = np.arange(n) / SR
            car = np.zeros(n)
            voices = [bass_note + 12, alb_root + iv[1], alb_root + 12, root + 12 + iv[2] % 12, root + 12 + iv[1]]
            for m in voices:
                for det in (-0.1, 0.1):
                    f = mtof(m) * 2 ** ((det + 0.15 * np.sin(2 * np.pi * 5 * tt + rng.uniform(0, 6))) / 12)
                    car += bl_osc(f, rng.uniform())
            c = meh_choir(car)
            c = c / (np.abs(c).max() + 1e-9) * np.minimum(1, tt / 0.03) * np.minimum(1, (dd - tt) / 0.08)
            put(choir, c, t0 + beat_pos * beat, 1.0)

    # ================================================================ mix
    sop = sop / (np.abs(sop).max() + 1e-9)
    choir = choir / (np.abs(choir).max() + 1e-9)
    orch = orch / (np.abs(orch).max() + 1e-9)
    dry = 0.62 * orch + np.column_stack([sop, sop]) * 0.85 + np.column_stack([choir * 0.9, choir * 1.0]) * 0.5
    send = 0.5 * orch + np.column_stack([sop, sop]) * 0.45 + np.column_stack([choir, choir]) * 0.4
    hall = np.column_stack([fftconvolve(send[:, c], reverb_ir(2.6, 7 + c))[:total] for c in range(2)])
    mix = dry + 0.55 * hall
    mix = np.column_stack([filt(mix[:, c], "highpass", 35, 2) for c in range(2)])
    mix *= 0.89 / np.abs(mix).max()                      # -1 dBFS, no clipping/saturation
    end = int(min(60.0, len(SOPRANO) * bar + 2.0) * SR)
    mix = mix[:end]
    fl = int(0.8 * SR)
    mix[-fl:] *= np.linspace(1, 0, fl)[:, None]
    return mix, {"sop": sop[:end], "orch": orch[:end, 0], "choir": choir[:end]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video or audio file")
    ap.add_argument("-o", "--output", default="opera.wav")
    ap.add_argument("--bpm", type=float, default=132)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    mix, _ = build(load_audio(a.input), a.bpm, a.seed)
    sf.write(a.output, mix, SR, subtype="PCM_16")
    print(f"wrote {a.output} ({len(mix)/SR:.1f}s)")


if __name__ == "__main__":
    main()

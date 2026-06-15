#!/usr/bin/env python3
"""Out-of-sample validation of the SPE-template tail correction.

The tuned template is, by construction, the channels' mean response over the
events used to build it -- so re-measuring on the SAME pulses would show the
undershoot drop almost circularly.  This script tests generalization instead:

  1. SPLIT-HALF: build on TRAIN events, measure the post-pulse undershoot on
     HELD-OUT events, comparing DEFAULT vs a single PER-TYPE FBK correction vs
     13 PER-CHANNEL templates.
  2. OUT-OF-POPULATION: re-deconvolve BRIGHT cosmics (never used to build the
     templates) and check the sub-zero undershoot AND the absolute PE scale
     (peak), which a template change can silently shift (-> flash PE / OpHit
     threshold).

Conclusion (run 27980): per-type beats per-channel out-of-sample (held-out
undershoot 2.3% vs 3.2%, default 4.9%) with no PE-scale shift -> ship per-type.

Usage: oos_validate.py   (uses pdhd/work/027980_fs{8,24,120} train, {16,104,152} test)
"""
import functools, numpy as np
from scipy.signal import find_peaks
import spe_template_tune as T

TRAIN, TEST, ALL = [8, 24, 120], [16, 104, 152], [8, 16, 24, 104, 120, 152]
CAND = [121, 124, 125, 127, 129, 131, 134, 137, 139, 141, 151, 157, 159]
PRE, POSTW, n = T.PRE, T.POSTW, 343808
post = T.build_postfilter(n)
names, tmpl, c2t = T.load_templates()
RUN = 27980


@functools.lru_cache(maxsize=8)
def frames(evt):
    return T.load_frames(f"{T.WORK}/{RUN:06d}_fs{evt}/light-frames-fullstream-wct.tar.bz2", evt)


def Hof(t):
    return np.fft.fft(np.pad(t[:n], (0, max(0, n - len(t))))[:n])


def small_pulses(f, o):
    raw = f["raw"][int(np.where(f["ch"] == o)[0][0])].astype(float)
    wd = f["decon"][int(np.where(f["ch"] == o)[0][0])].astype(float)
    pk, prop = find_peaks(wd, height=0.15, distance=4); h = prop["peak_heights"]
    ps = [p for i, p in enumerate(pk)
          if PRE <= p < len(wd) - POSTW
          and np.searchsorted(pk, p + T.ISO) - np.searchsorted(pk, p - T.ISO) == 1
          and raw[p - 15:p + 25].min() > T.RAIL_LO + 200
          and abs(wd[p - PRE:p - 12].mean()) <= 0.5 and 0.25 <= h[i] <= 1.0]
    return raw, wd, ps


def undershoot(shape):
    s = shape / shape[PRE]; b = s[:PRE - 12].mean()
    return 100 * (s[PRE + 15:PRE + 93].min() - b)


def main():
    # build TRAIN per-channel shapes -> per-channel and one per-type FBK template
    acc = {o: [np.zeros(PRE + POSTW), 0] for o in CAND}
    for evt in TRAIN:
        f = frames(evt)
        for o in CAND:
            _, wd, ps = small_pulses(f, o)
            for p in ps:
                acc[o][0] += wd[p - PRE:p + POSTW]; acc[o][1] += 1
    train = {o: acc[o][0] / acc[o][1] / (acc[o][0][PRE] / acc[o][1]) for o in CAND if acc[o][1]}
    per_chan = {o: T.reconstruct(train[o], tmpl[c2t[o]]) for o in train}
    per_type = T.reconstruct(np.mean([train[o] for o in train], axis=0), tmpl[0])
    Hpc = {o: Hof(per_chan[o]) for o in per_chan}; Hpt = Hof(per_type)

    print("=== OUT-OF-SAMPLE: undershoot on HELD-OUT events (small pulses) ===")
    ta = {k: {o: [np.zeros(PRE + POSTW), 0] for o in CAND} for k in ("def", "pc", "pt")}
    for evt in TEST:
        f = frames(evt)
        for o in CAND:
            raw, wd, ps = small_pulses(f, o)
            wpc = T.deconvolve(raw, Hpc[o], post, n) if o in Hpc else None
            wpt = T.deconvolve(raw, Hpt, post, n)
            for p in ps:
                ta["def"][o][0] += wd[p - PRE:p + POSTW]; ta["def"][o][1] += 1
                if wpc is not None:
                    ta["pc"][o][0] += wpc[p - PRE:p + POSTW]; ta["pc"][o][1] += 1
                ta["pt"][o][0] += wpt[p - PRE:p + POSTW]; ta["pt"][o][1] += 1
    print("ch   default  per-channel  per-type-FBK  (N)")
    agg = {"def": [], "pc": [], "pt": []}
    for o in CAND:
        if ta["def"][o][1] < 5:
            continue
        u = {k: undershoot(ta[k][o][0] / ta[k][o][1]) for k in ta}
        for k in agg:
            agg[k].append(u[k])
        print(f"{o}  {u['def']:+6.1f}%  {u['pc']:+6.1f}%     {u['pt']:+6.1f}%      ({ta['def'][o][1]})")
    for k, lab in (("def", "DEFAULT"), ("pc", "per-channel"), ("pt", "per-type FBK")):
        print(f"  mean |undershoot| {lab:13s}: {np.mean(np.abs(agg[k])):.2f}%")

    print("\n=== OUT-OF-POPULATION: BRIGHT cosmics, sub-zero undershoot + PE scale (per-type) ===")
    print("ch   under def->pt      peak def->pt (PE-scale %)   (N)")
    du, dpt = [], []
    for o in CAND:
        u0l, u1l, pk0, pk1 = [], [], [], []
        for evt in ALL:
            f = frames(evt)
            raw = f["raw"][int(np.where(f["ch"] == o)[0][0])].astype(float)
            wd = f["decon"][int(np.where(f["ch"] == o)[0][0])].astype(float)
            wpt = T.deconvolve(raw, Hpt, post, n)
            pk, prop = find_peaks(wd, height=3, distance=40); h = prop["peak_heights"]
            for i, p in enumerate(pk):
                if not (5 <= h[i] <= 30) or p < PRE or p + POSTW >= len(wd):
                    continue
                if np.searchsorted(pk, p + 200) - np.searchsorted(pk, p - 200) != 1 or raw[p - 15:p + 25].min() <= T.RAIL_LO + 200:
                    continue
                u0l.append(wd[p + 15:p + 190].min() - wd[p - PRE:p - 12].mean())
                u1l.append(wpt[p + 15:p + 190].min() - wpt[p - PRE:p - 12].mean())
                pk0.append(wd[p]); pk1.append(wpt[p])
        if len(u0l) < 3:
            continue
        dscale = 100 * (np.mean(pk1) - np.mean(pk0)) / np.mean(pk0)
        du.append(np.median(u0l)); dpt.append(np.median(u1l))
        print(f"{o}  {np.median(u0l):+5.2f} -> {np.median(u1l):+5.2f}   "
              f"{np.mean(pk0):6.2f} -> {np.mean(pk1):6.2f}  ({dscale:+.1f}%)   ({len(u0l)})")
    print(f"  median bright sub-zero: default {np.median(du):+.2f} -> per-type {np.median(dpt):+.2f}")


if __name__ == "__main__":
    main()

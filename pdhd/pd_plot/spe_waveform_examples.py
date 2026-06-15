#!/usr/bin/env python3
"""Per-channel raw-vs-decon example waveforms (PDHD light), all 160 OpChannels.

For each channel pick THREE representative pulses -- small, medium, large (by
deconvolved peak height), each with a clean flat pre-peak baseline -- and show,
for each, the RAW ADC and the DECON over a wide window (-5 .. +20 us from the
peak).  This is a visual cross-check of the deconvolution per channel: the raw
shows the AC-coupled SiPM pulse (fast dip + slow recovery), the decon collapses it
to a spike whose residual tail/undershoot is the template-match diagnostic studied
in pdhd-spe-template-tuning.md.

  * self-trigger channels (0-119): pulses live in 1024-sample (16 us) snippets, the
    actual readout window.  Both axes are LIMITED to that snippet and the decon is
    taken from the TOOLKIT directly (f["decon"], the software OpDecon over the
    snippet) -- no python re-deconvolution.
  * full-stream channels (120-159): continuous 343k-sample readout.  The decon is
    computed LOCALLY -- only the isolated raw window around the pulse is deconvolved
    (flat-padded with the local pedestal), so the result has no bias from other
    pulses elsewhere in the frame and spans the full -5..+20 us.  It reproduces the
    software full-frame decon to <0.005 PE/tick (baseline-subtracted).

Reads the per-event reco produced by run_light_fullstream_evt.sh:
  work/<RUN>_fs<evt>/   light-frames-fullstream-wct.tar.bz2  (opch 120-159)
  work/<RUN>_snip<evt>/ light-frames-wct.tar.bz2             (opch 0-119)

Output -> pdhd/pics/pd/wf_ch<NNN>.png  (one per channel, 3x2 panels: rows
small/medium/large, columns raw / decon).

Usage: spe_waveform_examples.py <run> [evt evt ...] [-- ch ch ...]
       (default run 27980, events 8 16 24 104 120 152, all channels)
"""
import os, sys
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spe_template_tune as T

PDHD = T.PDHD
WORK = T.WORK
PICS = os.path.join(PDHD, "pics", "pd")
TICK_NS = T.TICK_NS

PRE_US, POST_US = 5.0, 20.0                       # decon window around the peak
PRE_S = int(PRE_US * 1000 / TICK_NS)              # 312 samples
POST_S = int(POST_US * 1000 / TICK_NS)            # 1250 samples
GUARD = 1024                                      # flat-baseline pad each side for
#  the LOCAL deconvolution (absorbs FFT edge wraparound; matches the software
#  full-frame decon to <0.005 PE/tick after pre-peak-baseline subtraction).

# small / medium / large bands by decon peak height (PE/tick) and a target
# amplitude used to pick the most representative pulse in each band.
BANDS = [("small", 0.30, 1.5, 0.6), ("medium", 1.5, 5.0, 2.5), ("large", 5.0, 60.0, 9.0)]


def is_sharp(w, p, h):
    """Genuine fast SPE-like pulse: the decon spike decays to a small fraction of
    its peak within ~0.3 us on both sides.  Rejects the broad ramps/plateaus the
    decon produces around saturating giant cosmics (a flat-topped shoulder is not
    a pulse)."""
    return w[p + 20] < 0.45 * h and w[p - 20] < 0.55 * h


def snippet_bounds(raw, p):
    """Bounds [s, e] of the contiguous nonzero run (self-trigger readout snippet)
    containing the peak p -- the actual readout window for self-trigger channels."""
    s = e = p
    while s > 0 and raw[s - 1] != 0:
        s -= 1
    while e < len(raw) - 1 and raw[e + 1] != 0:
        e += 1
    return s, e


def pulse_quality(raw, p):
    """Local pedestal, pulse dip depth, and pre-peak baseline flatness from the RAW.
    The pre-peak window (up to -5 us, real samples only -> self-trigger gaps masked)
    must be FLAT: pre_ptp is its robust peak-to-peak (3rd..97th pct), so a preceding
    pulse's tail or a slow baseline ramp before the peak shows up as a large pre_ptp
    and is de-selected.  Returns None when too few real pre samples to judge."""
    pre = raw[max(0, p - PRE_S):p - 12]
    pre = pre[pre != 0].astype(float)
    if len(pre) < 20:
        return None
    ped = float(np.median(pre))
    pre_ptp = float(np.percentile(pre - ped, 97) - np.percentile(pre - ped, 3))
    dip = ped - float(raw[p - 3:p + 30].min())
    return ped, dip, pre_ptp


def find_examples(w, raw):
    """Return {band: (idx, height, key, ped, dip)} of the most representative pulse
    per band: a sharp decon spike with a real raw dip AND a clean flat pre-peak
    baseline under it.  Selection key (lower is better) prefers, in order, a flat
    pre-peak baseline, then unsaturated (raw off the ADC rails), then the amplitude
    closest to the band target."""
    pk, prop = find_peaks(w, height=0.20, distance=8)
    h = prop["peak_heights"]
    out = {}
    for name, lo, hi, tgt in BANDS:
        best = None  # (key, idx, height, ped, dip)
        for i, p in enumerate(pk):
            if not (lo <= h[i] < hi):
                continue
            if p - PRE_S < 0 or p + POST_S >= len(w):
                continue
            if not is_sharp(w, p, h[i]):
                continue
            q = pulse_quality(raw, p)
            if q is None:
                continue
            ped, dip, pre_ptp = q
            if dip < max(4.0, 8.0 * h[i]):
                continue
            badness = pre_ptp / max(dip, 1.0)           # pre-peak wander / pulse dip
            bucket = 0 if badness < 0.15 else (1 if badness < 0.30 else 2)
            # post-peak coverage: consecutive real (nonzero) raw samples after the
            # peak.  For self-trigger this is the room left in the 1024-sample
            # snippet -- prefer pulses EARLY in their snippet so the raw covers most
            # of the window and the full AC recovery is captured (full-stream: full).
            tailz = np.flatnonzero(raw[p:p + POST_S + 1] == 0)
            post_room = int(tailz[0]) if len(tailz) else POST_S
            cov = 0 if post_room >= 600 else (1 if post_room >= 300 else 2)
            seg = raw[p - 3:p + 30]
            sat = (seg.min() <= T.RAIL_LO) or (seg.max() >= T.RAIL_HI)
            key = (bucket, cov, 1 if sat else 0, abs(h[i] - tgt))
            if best is None or key < best[0]:
                best = (key, p, float(h[i]), ped, dip)
        if best is not None:
            out[name] = (best[1], best[2], best[0], best[3], best[4])
    return out


def window(arr, p, mask_zero=False):
    """Extract [-PRE_S, +POST_S] around p, NaN-padded outside the frame; optionally
    mask exact zeros (self-trigger snippet gaps) to NaN so they don't draw."""
    seg = np.full(PRE_S + POST_S, np.nan)
    a = max(0, p - PRE_S); b = min(len(arr), p + POST_S)
    seg[a - (p - PRE_S):b - (p - PRE_S)] = arr[a:b]
    if mask_zero:
        seg[seg == 0] = np.nan
    return seg


def decon_cache(c2t):
    """Per-template (H, postfilter) for the local-window length, built once."""
    n = PRE_S + POST_S + 2 * GUARD
    _, tmpl, _ = T.load_templates()
    cache = {}
    for ti in set(c2t.values()):
        t = tmpl[ti]
        cache[ti] = (np.fft.fft(np.pad(t[:n], (0, max(0, n - len(t))))[:n]),
                     T.build_postfilter(n))
    return cache, n


def local_decon(raw, p, ped, ti, cache, n):
    """Deconvolve ONLY the isolated raw window around the pulse (flat-padded with the
    local pedestal, so snippet gaps and the frame edges become a clean baseline),
    using the channel's SPE template -- the validated OpDecon NumPy port.  This
    avoids any bias from other pulses elsewhere in the frame and gives a decon that
    spans the full -5..+20 us even for the 16 us self-trigger snippets.  Reproduces
    the software full-frame decon to <0.005 PE/tick after baseline subtraction.
    Returned cropped to [-PRE_S, +POST_S], with its own pre-peak baseline removed."""
    H, post = cache[ti]
    lo = p - PRE_S - GUARD
    seg = np.full(n, ped)
    a = max(0, lo); b = min(len(raw), lo + n)
    seg[a - lo:b - lo] = raw[a:b]
    seg[seg == 0] = ped                              # flat-fill self-trigger gaps
    dloc = T.deconvolve(seg, H, post, n)
    crop = dloc[GUARD:GUARD + PRE_S + POST_S].copy()
    return crop - np.median(crop[:PRE_S - 30])       # zero the flat pre-peak baseline


def pick_all(run, events, c2t, cache, n):
    """Two-pass over events: keep, per channel and band, the most representative
    pulse (flat pre-peak baseline, unsaturated, closest to the band target).  For
    each kept pulse store the masked raw window and the LOCAL deconvolution of that
    isolated window (not the software full-frame decon)."""
    store = {o: {} for o in range(160)}            # o -> band -> dict(raw,dec,h,evt)
    rank = {o: {} for o in range(160)}             # o -> band -> sort key (lower better)
    for evt in events:
        for dom, pref, sub in ((range(120, 160), "fs", "fullstream-wct"),
                               (range(0, 120), "snip", "wct")):
            fn = f"{WORK}/{run:06d}_{pref}{evt}/light-frames-{sub}.tar.bz2"
            if not os.path.exists(fn):
                continue
            f = T.load_frames(fn, evt)
            snip = (pref == "snip")
            for o in dom:
                idx = np.where(f["ch"] == o)[0]
                if not len(idx):
                    continue
                i = int(idx[0])
                wdec = f["decon"][i].astype(float)
                wraw = f["raw"][i].astype(float)
                ex = find_examples(wdec, wraw)
                for name, lo, hi, tgt in BANDS:
                    if name not in ex:
                        continue
                    p, hgt, key, ped, dip = ex[name]
                    if name in rank[o] and key >= rank[o][name]:
                        continue
                    rank[o][name] = key
                    if snip:
                        # self-trigger: limit to the actual readout snippet and take
                        # the toolkit (software) decon over it -- no python decon.
                        s, e = snippet_bounds(wraw, p)
                        rawseg = wraw[s:e + 1] - ped
                        dec = wdec[s:e + 1] - np.median(wdec[s:max(s + 1, p - 12)])
                        tt = (np.arange(s, e + 1) - p) * TICK_NS / 1000.0
                        h = float(dec[p - s])
                    else:
                        # full-stream: local python decon over the fixed -5..+20 us.
                        dec = local_decon(wraw, p, ped, c2t.get(o, 1), cache, n)
                        rawseg = window(wraw, p) - ped
                        tt = (np.arange(PRE_S + POST_S) - PRE_S) * TICK_NS / 1000.0
                        h = float(dec[PRE_S - 2:PRE_S + 3].max())
                    store[o][name] = dict(raw=rawseg, dec=dec, t=tt, h=h,
                                          evt=evt, snip=snip, dip=dip)
    return store


def fig_for_channel(o, bands, c2t):
    ftype = "FBK" if c2t.get(o, 1) == 0 else "HPK"
    snip = o < 120
    dom = "self-trigger" if snip else "full-stream"
    dlabel = "toolkit decon" if snip else "local decon"
    fig, ax = plt.subplots(3, 2, figsize=(13, 9))
    for row, (name, lo, hi, tgt) in enumerate(BANDS):
        d = bands.get(name)
        ar, ad = ax[row, 0], ax[row, 1]
        if d is None:
            for a in (ar, ad):
                a.text(0.5, 0.5, f"no {name} pulse found", ha="center", va="center",
                       transform=a.transAxes, color="0.5", fontsize=11)
            ar.set_ylabel(name, fontsize=11, fontweight="bold")
            continue
        t = d["t"]
        xlim = (t[0], t[-1]) if snip else (-PRE_US, POST_US)   # self-trigger: snippet
        ar.plot(t, d["raw"], color="0.25", lw=0.8)
        ar.set_ylabel(f"{name}\nraw ADC - ped", fontsize=10, fontweight="bold")
        ar.axhline(0, color="0.7", lw=0.5); ar.axvline(0, color="C3", lw=0.5, ls=":")
        ar.set_title(f"raw (evt {d['evt']}, dip {d['dip']:.0f} ADC)", fontsize=9)
        ar.set_xlim(*xlim)
        ad.plot(t, d["dec"], color="C0", lw=0.9)
        ad.axhline(0, color="0.7", lw=0.5); ad.axvline(0, color="C3", lw=0.5, ls=":")
        ad.axhspan(-0.05 * d["h"], 0, color="red", alpha=0.05)
        ad.set_ylabel("decon [PE/tick]", fontsize=10)
        ad.set_title(f"{dlabel} (peak {d['h']:.2f} PE/tick)", fontsize=9)
        ad.set_xlim(*xlim)
    for a in ax[2, :]:
        a.set_xlabel("time since peak [us]")
    note = "  (self-trigger: actual readout snippet, toolkit decon)" if snip else ""
    fig.suptitle(f"OpChannel {o}  ({ftype}, {dom})   raw vs {dlabel} -- "
                 f"small / medium / large pulses{note}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(PICS, exist_ok=True)
    fig.savefig(f"{PICS}/wf_ch{o:03d}.png", dpi=90)
    plt.close(fig)


def main():
    args = sys.argv[1:]
    chsel = None
    if "--" in args:
        k = args.index("--"); chsel = set(int(a) for a in args[k + 1:]); args = args[:k]
    run = int(args[0]) if args else 27980
    events = [int(a) for a in args[1:]] or [8, 16, 24, 104, 120, 152]
    _, _, c2t = T.load_templates()
    cache, n = decon_cache(c2t)
    store = pick_all(run, events, c2t, cache, n)
    chans = sorted(store) if chsel is None else sorted(chsel)
    nmade = 0
    for o in chans:
        if not store[o]:
            continue
        fig_for_channel(o, store[o], c2t)
        nmade += 1
    print(f"wrote {nmade} per-channel raw-vs-decon figures to {PICS}/wf_ch<NNN>.png")


if __name__ == "__main__":
    main()

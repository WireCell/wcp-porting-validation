#!/usr/bin/env python3
"""Prototype: robust per-channel baseline for full-stream OpHit finding.

The full-stream OpHit finder over-produces flashes because its pedestal is the
larana PedAlgoEdges HEAD method (3 samples) -- meaningless on a 343 808-sample
continuous stream -- feeding an effectively ABSOLUTE low threshold.  Two channels
slip through (see pdhd-fullstream-light-reco.md s6):
  * opch 147 ringing  -- robust RMS ~3.0 vs ~0.02 typical; every lobe fires.
  * opch 121 DC-offset -- flat +0.19 baseline sits above the 0.11 hit threshold.

This script tests the proposed fix at the source (OpHitFinder), in NumPy, before
the C++ change.  It ports AlgoSlidingWindow::RecoPulse faithfully and runs it two
ways on the SAME decon waveforms:
  HEAD   : ped_mean/ped_sigma from the first m_ped_nsamples (current C++).
  ROBUST : ped_mean = per-channel GLOBAL median, ped_sigma = per-channel MAD,
           sliding-window start gate raised to robust_nsigma * MAD.
Because sliding_window already does start_threshold = max(adc, nsigma*ped_sigma),
the only change is feeding it a robust (median, MAD) and raising nsigma -- exactly
the scalar swap planned for the C++ (sliding_window/split_pulse untouched).

The data show the two channels fail DIFFERENTLY and need two robust ingredients:
  * 121 DC-offset is 98.5%-duty -> per-channel GLOBAL MEDIAN ped removes it; its
    residual hits (~152) match a clean channel's (~133), i.e. its REAL hits.
  * 147 ringing has sigma ~3 decon (150x clean) with extreme lobes that survive
    even a large n*sigma start gate; a per-channel n*sigma threshold can't kill it
    without also cutting clean small pulses.  But sigma ITSELF is the veto signal
    (the existing classify_channels calls robust-RMS >= 0.1 decon "ringing"), so
    full-stream VETOES channels whose MAD-sigma exceeds a cap.

Metric (hits, NOT flashes -- flashes are validated only in the C++ end-to-end run):
per-channel emitted hit count vs robust_nsigma, separated into the BAD channels
(must collapse to ~0) and the CLEAN channels (real ~1-PE hits must be RETAINED).
The n that does both, plus the sigma veto, picks the C++ defaults.

Usage: fullstream_baseline_proto.py [run] [evt]   (default 27980 8)
"""
import os, sys, io, tarfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDHD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(PDHD, "work")
PICS = os.path.join(PDHD, "pics")

# OpHitFinder / flash.jsonnet full-stream values.
SCALE = 100.0          # m_scale: decon -> scaled short units before thresholds
HIT_THRESHOLD = 11.0   # m_hit_threshold on pulse peak (scaled units) = 0.11 decon
PED_NSAMPLES = 3       # m_ped_nsamples, head method
VETO_SIGMA = 10.0      # robust MAD-sigma veto (scaled) = 0.1 decon (the "ringing" class cut)
# AlgoSlidingWindow defaults (dune_ophit_finder_deco), scaled units.
ALGO = dict(adc_threshold=3.0, nsigma_threshold=1.0,
            tail_adc_threshold=1.0, tail_nsigma_threshold=1.0,
            end_adc_threshold=1.0, end_nsigma_threshold=1.0,
            min_pulse_width=1, num_presample=2, num_postsample=2)


def load_frames(bz2, evt):
    with tarfile.open(bz2) as t:
        decon = np.load(io.BytesIO(t.extractfile("frame_decon_%d.npy" % evt).read()))
        ch = np.load(io.BytesIO(t.extractfile("channels_decon_%d.npy" % evt).read()))
    return decon, ch


def sliding_window(wf, ped_mean, ped_sigma, nsigma_start):
    """Port of pmtana::AlgoSlidingWindow::RecoPulse (positive polarity, constant
    ped).  Returns list of (t_start, t_end, t_max, peak).  Matches OpHitFinder.cxx
    sliding_window; nsigma_start overrides algo nsigma_threshold for the start gate
    (the one robust lever).  Splitting is NOT applied (it only subdivides found
    pulses; per-channel hit counts -- the over-production metric -- are unchanged)."""
    a = ALGO
    start_threshold = max(a["adc_threshold"], nsigma_start * ped_sigma)
    tail_threshold = max(a["tail_adc_threshold"], a["tail_nsigma_threshold"] * ped_sigma)
    end_threshold = max(a["end_adc_threshold"], a["end_nsigma_threshold"] * ped_sigma)
    num_pre, num_post = a["num_presample"], a["num_postsample"]
    min_width = a["min_pulse_width"]

    pulses = []
    cur = dict(t_start=0, t_end=0, t_max=0, peak=0.0)
    fire = in_tail = in_post = False
    post_integration = 0

    def register(t_end):
        cur["t_end"] = t_end
        if cur["t_end"] - cur["t_start"] >= min_width:
            pulses.append((cur["t_start"], cur["t_end"], cur["t_max"], cur["peak"]))

    for i in range(len(wf)):
        value = wf[i] - ped_mean
        if (not fire or in_tail or in_post) and value > start_threshold:
            if in_tail:
                register(i - 1); cur.update(t_start=0, t_end=0, t_max=0, peak=0.0)
            if not pulses:
                buf = min(num_pre, i)
            else:
                buf = i - pulses[-1][1] - 1
            if buf > num_pre:
                buf = num_pre
            if in_post:
                t_end = i - buf
                if t_end > 0:
                    t_end -= 1
                register(t_end); cur.update(t_start=0, t_end=0, t_max=0, peak=0.0)
            cur["t_start"] = i - buf
            fire = True; in_tail = in_post = False
        if fire and value < tail_threshold:
            fire = False; in_tail = True; in_post = False
        if (fire or in_tail) and value < end_threshold:
            in_post = True; fire = in_tail = False; post_integration = num_post
        if in_post and post_integration < 1:
            register(i - 1); cur.update(t_start=0, t_end=0, t_max=0, peak=0.0)
            fire = in_tail = in_post = False
        if fire or in_tail or in_post:
            if cur["peak"] < value:
                cur["peak"] = value; cur["t_max"] = i
            if in_post:
                post_integration -= 1
    if fire or in_tail or in_post:
        register(len(wf) - 1)
    return pulses


def count_hits(wf_scaled, ped_mean, ped_sigma, nsigma_start):
    """Emitted hits: sliding-window pulses whose peak >= HIT_THRESHOLD (scaled,
    ped-subtracted), as in OpHitFinder operator()."""
    return sum(1 for (_, _, _, pk) in sliding_window(wf_scaled, ped_mean, ped_sigma, nsigma_start)
               if pk >= HIT_THRESHOLD)


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 27980
    evt = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    bz2 = "%s/%06d_fs%d/light-frames-fullstream-wct.tar.bz2" % (WORK, run, evt)
    decon, ch = load_frames(bz2, evt)

    # Per-channel waveform (scaled short units), decon median, robust RMS, class.
    rows = {}
    for opch in range(120, 160):
        idx = np.where(ch == opch)[0]
        if not len(idx):
            continue
        w = decon[int(idx[0])].astype(float)
        wf = (SCALE * w).astype(np.int16).astype(float)  # match C++ static_cast<short>
        med_decon = float(np.median(w))
        mad_decon = float(np.median(np.abs(w - med_decon)) * 1.4826)
        cls = "ringing" if mad_decon >= 0.1 else ("offset" if abs(med_decon) >= 0.05 else "clean")
        # HEAD ped (current C++)
        head_mean = wf[:PED_NSAMPLES].mean()
        head_sigma = wf[:PED_NSAMPLES].std()
        # ROBUST ped: global median + MAD (scaled units)
        rob_mean = float(np.median(wf))
        rob_sigma = float(np.median(np.abs(wf - rob_mean)) * 1.4826)
        rows[opch] = dict(wf=wf, cls=cls, med=med_decon, mad=mad_decon,
                          head_mean=head_mean, head_sigma=head_sigma,
                          rob_mean=rob_mean, rob_sigma=rob_sigma)

    BAD = [c for c in rows if rows[c]["cls"] != "clean"]
    CLEAN = [c for c in rows if rows[c]["cls"] == "clean"]
    print("run %d evt %d: %d full-stream channels (%d clean, %d bad: %s)"
          % (run, evt, len(rows), len(CLEAN), len(BAD),
             ", ".join("ch%d/%s" % (c, rows[c]["cls"]) for c in sorted(BAD))))

    # Baseline: current C++ (HEAD ped, nsigma=1.0)
    head_bad = {c: count_hits(rows[c]["wf"], rows[c]["head_mean"], rows[c]["head_sigma"],
                              ALGO["nsigma_threshold"]) for c in rows}
    head_clean_total = sum(head_bad[c] for c in CLEAN)
    print("\nHEAD (current C++):  bad-channel hits = %s ;  clean-channel hits = %d"
          % ({c: head_bad[c] for c in sorted(BAD)}, head_clean_total))

    # Robust sigma veto (ringing channels): MAD-sigma >= VETO_SIGMA -> dropped.
    vetoed = sorted(c for c in rows if rows[c]["rob_sigma"] >= VETO_SIGMA)
    print("\nrobust sigma veto (>= %.0f scaled = %.2f decon): %s"
          % (VETO_SIGMA, VETO_SIGMA / SCALE, vetoed or "none"))

    # ROBUST n-sweep (global-median ped + MAD sigma, veto applied -> vetoed = 0 hits)
    ns = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    print("\nROBUST (global median + MAD + veto), per-channel start = max(3.0, n*MAD_scaled):")
    print("  per-channel robust sigma (scaled): " +
          ", ".join("ch%d=%.1f(%s)" % (c, rows[c]["rob_sigma"], rows[c]["cls"]) for c in sorted(BAD)))

    def rob_hits(c, n):
        if c in vetoed:
            return 0
        return count_hits(rows[c]["wf"], rows[c]["rob_mean"], rows[c]["rob_sigma"], n)

    print("  %-5s | %-28s | %-18s" % ("n", "bad-channel hits", "clean hits (retain)"))
    sweep = {}
    for n in ns:
        bad = {c: rob_hits(c, n) for c in rows}
        clean_total = sum(bad[c] for c in CLEAN)
        bad_total = sum(bad[c] for c in BAD)
        sweep[n] = dict(bad=bad, clean_total=clean_total, bad_total=bad_total)
        print("  %-5.1f | %-28s | %d" % (n, {c: bad[c] for c in sorted(BAD)}, clean_total))
    print("\nHEAD clean-hit reference (must be ~preserved): %d" % head_clean_total)

    # End-to-end direction at the chosen default (n=3.0): total full-stream hits.
    NCH_DEF = 3.0
    head_total = sum(head_bad.values())
    rob_total = sum(sweep[NCH_DEF]["bad"].values())
    print("\nTOTAL full-stream hits  HEAD=%d  ->  ROBUST(n=%.1f,veto)=%d  (%.0f%% reduction)"
          % (head_total, NCH_DEF, rob_total, 100 * (1 - rob_total / head_total)))
    print("  of which bad channels: HEAD=%d -> ROBUST=%d ; clean: HEAD=%d -> ROBUST=%d"
          % (sum(head_bad[c] for c in BAD), sweep[NCH_DEF]["bad_total"],
             head_clean_total, sweep[NCH_DEF]["clean_total"]))

    # Figure: bad-channel hits and clean retention vs n
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].axhline(head_bad.get(147, 0), ls=":", c="#d62728", lw=1,
                  label="ch147 HEAD (current)=%d" % head_bad.get(147, 0))
    ax[0].axhline(head_bad.get(121, 0), ls=":", c="#ff7f0e", lw=1,
                  label="ch121 HEAD (current)=%d" % head_bad.get(121, 0))
    for c, col in ((147, "#d62728"), (121, "#ff7f0e")):
        if c in rows:
            ax[0].plot(ns, [sweep[n]["bad"][c] for n in ns], "o-", color=col, label="ch%d robust" % c)
    ax[0].set(xlabel="robust n-sigma start gate", ylabel="emitted hits",
              title="A. Bad-channel over-production collapses\n(ch147 ringing, ch121 DC-offset)")
    ax[0].legend(fontsize=8); ax[0].set_yscale("symlog")
    ax[1].axhline(head_clean_total, ls=":", c="k", lw=1, label="HEAD clean total=%d" % head_clean_total)
    ax[1].plot(ns, [sweep[n]["clean_total"] for n in ns], "s-", color="#1f77b4",
               label="robust clean total")
    ax[1].set(xlabel="robust n-sigma start gate", ylabel="emitted hits (all clean channels)",
              title="B. Clean-channel REAL hits retained\n(binding constraint: n*sigma_clean < 0.11)")
    ax[1].legend(fontsize=8)
    fig.suptitle("run %d evt %d: full-stream robust-baseline OpHit prototype" % (run, evt), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(PICS, exist_ok=True)
    out = "%s/pd_fullstream_%d_evt%d_baseline_proto.png" % (PICS, run, evt)
    fig.savefig(out, dpi=95); print("\nwrote", out)


if __name__ == "__main__":
    main()

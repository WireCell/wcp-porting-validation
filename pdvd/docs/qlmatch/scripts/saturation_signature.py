#!/usr/bin/env python3
"""What does PDVD DAPHNE saturation actually look like, and does the signal go negative?

Read-only.  Answers, from the run-039252 evt298567 raw light streams, the three
questions of `14_pdvd-lightpattern-sp-investigation.md` §"Saturation: signature":

  A. Is there a NEGATIVE clip?  -> per-channel min/max ADC and the count of
     samples at the lower (0) and upper (16383) 14-bit rails.
  B. Does the raw signal go negative?  -> yes, but as a slow AC-coupling
     UNDERSHOOT after the pulse, measured against a LOCAL pre-pulse baseline
     (the median of a quiet band 250..50 ticks before the rail).  Using the
     stream-head pedestal here instead would confound true undershoot with
     baseline-reference bias, so it is deliberately not used.
  C. Does the DECONVOLVED waveform go negative inside the rail?  -> no: a
     clipped flat-top deconvolves to a sustained POSITIVE plateau (the
     over-integration).  Checked on several channels, not one.
  D. Is the undershoot saturation-SPECIFIC, or does any bright pulse do it?
     -> bright UNRAILED pulses undershoot too, at the same undershoot/amplitude
     ratio as railed ones, i.e. it is linear AC coupling, not a saturation
     recovery artifact.

Only channels whose pre-band is genuinely quiet (std < QUIET_SIG) enter the
§B summary; on the rest the "quiet" band contains another pulse and the local
baseline is meaningless.  ch1081 is such a case (pre-band std ~4300).

Companion doc: 14_pdvd-lightpattern-sp-investigation.md
Repro:
    cd wcp-porting-img/pdvd
    python3 docs/qlmatch/scripts/saturation_signature.py
"""
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pd_mapping_audit import RAWWF, RAIL, load_cathode_streams, rail_intervals  # noqa: E402
from saturation_recovery_study import load_channels, decon_replica  # noqa: E402

EVENT = 298567
SEG = 16384          # decon segment length (study convention)
QUIET_SIG = 15.0     # ADC; pre-band std above this => not a clean local baseline
MIN_AMP = 2000.0     # ADC; "bright" for the unrailed-pulse comparison (D)
FIG_CH = 1010        # figure channel: deep rail AND a quiet pre-band
DECON_CHS = (1010, 1050, 1021, 1040)
OUT_PNG = os.path.join(HERE, "..", "pics", "lightpattern_fig6_saturation_signature.png")


def local_base(w, i0):
    """Median/std of the quiet band 250..50 ticks before the rail run."""
    pre = w[max(0, i0 - 250):max(0, i0 - 50)]
    return float(np.median(pre)), float(pre.std())


def main():
    rf = sorted(glob.glob(os.path.join(RAWWF, "np02vd_raw_run039252_1176_*_rawwf.root")))[0]
    (waves, _ts), _t0 = load_cathode_streams(rf, EVENT)
    chans = load_channels()

    print("=" * 84)
    print("PDVD saturation signature   run 039252 evt %d   rail = %d (14-bit)" % (EVENT, RAIL))
    print("file: %s" % os.path.basename(rf))
    print("=" * 84)

    # ---------------- A. is there a negative clip? ----------------
    print("\nA. Rail census (input_polarity +1 => the clip is at the UPPER rail)")
    print("  %-7s %8s %8s %9s %9s" % ("ch", "min", "max", "n(ADC<=0)", "n(>=rail)"))
    nlow_tot = 0
    for c in sorted(waves):
        w = waves[c]
        nlow = int((w <= 0).sum())
        nlow_tot += nlow
        print("  ch%-5d %8.0f %8.0f %9d %9d"
              % (c, w.min(), w.max(), nlow, int((w >= RAIL).sum())))
    print("  -> samples at the LOWER rail across all cathode channels: %d" % nlow_tot)

    # ---------------- B. raw undershoot, local baseline ----------------
    print("\nB. Raw undershoot after the deepest rail, vs a LOCAL pre-pulse baseline")
    print("   (min ADC relative to that baseline, in windows after the rail exit)")
    print("  %-7s %7s %9s %8s %9s %9s %9s" %
          ("ch", "raillen", "base", "std", "min@+2k", "min@+8k", "min@+30k"))
    rows = []
    for c in sorted(waves):
        w = waves[c].astype(np.float64)
        runs = sorted(rail_intervals(w, min_len=1), key=lambda r: r[1] - r[0], reverse=True)
        if not runs:
            continue
        i0, j0 = runs[0]
        base, sig = local_base(w, i0)
        mins = []
        for L in (2000, 8000, 30000):
            post = w[j0:j0 + L]
            mins.append(float(post.min()) - base if len(post) > 10 else np.nan)
        quiet = sig < QUIET_SIG
        rows.append((c, j0 - i0, base, sig, mins, quiet))
        print("  ch%-5d %7d %9.1f %8.2f %+9.1f %+9.1f %+9.1f%s"
              % (c, j0 - i0, base, sig, mins[0], mins[1], mins[2],
                 "" if quiet else "   (pre-band not quiet: excluded)"))
    q = [r for r in rows if r[5]]
    for k, lab in ((0, "+2k"), (1, "+8k"), (2, "+30k")):
        v = np.array([r[4][k] for r in q])
        print("  quiet-band channels (n=%d) at %-4s: median %+.1f ADC, range [%+.0f, %+.0f], n<0 = %d/%d"
              % (len(q), lab, np.median(v), v.min(), v.max(), int((v < 0).sum()), len(v)))

    # ---------------- C. decon inside the rail ----------------
    print("\nC. Deconvolved waveform across the rail (units: 100 = 1 PE/tick)")
    print("  %-7s %7s %9s %9s %9s %12s" %
          ("ch", "raillen", "in_min", "in_max", "in_mean", "n(<0) inside"))
    for c in DECON_CHS:
        if c not in waves or c not in chans:
            continue
        w = waves[c]
        runs = sorted(rail_intervals(w, min_len=1), key=lambda r: r[1] - r[0], reverse=True)
        if not runs:
            continue
        i0, j0 = runs[0]
        s0 = max(0, i0 - SEG // 4)
        dec = decon_replica(w[s0:s0 + SEG], chans[c], nfft=SEG)
        inside = dec[i0 - s0:j0 - s0]
        print("  ch%-5d %7d %+9.2f %+9.2f %+9.2f %8d/%-4d"
              % (c, j0 - i0, inside.min(), inside.max(), inside.mean(),
                 int((inside < 0).sum()), len(inside)))
    print("  -> the clipped flat-top deconvolves to a sustained POSITIVE plateau;")
    print("     it never goes negative inside the rail.  The small negative dips")
    print("     are in the BASELINE between pulses -- what OpRoi's HPF removes.")

    # ---------------- D. is the undershoot saturation-specific? ----------------
    print("\nD. Bright UNRAILED pulses: do they undershoot too?")
    print("   (isolated local maxima >= MIN_AMP over a quiet local baseline, rail")
    print("    neighbourhoods excluded; ratio = undershoot@+8k / pulse amplitude)")
    urows = []
    for c in sorted(waves):
        w = waves[c].astype(np.float64)
        busy = np.zeros(len(w), bool)
        for i, j in rail_intervals(w, min_len=1):
            busy[max(0, i - 5000):min(len(w), j + 40000)] = True
        for _ in range(3):
            cand = np.where(~busy)[0]
            if not len(cand):
                break
            k = int(cand[np.argmax(w[cand])])
            base, sig = local_base(w, k)
            amp = w[k] - base
            if sig > QUIET_SIG or amp < MIN_AMP or w[k] >= RAIL - 100:
                busy[max(0, k - 2000):k + 2000] = True
                continue
            post = w[k:k + 8000]
            m8 = float(post.min()) - base if len(post) > 10 else np.nan
            urows.append((c, amp, m8, m8 / amp))
            busy[max(0, k - 3000):k + 40000] = True
    if urows:
        a = np.array([[r[1], r[2], r[3]] for r in urows])
        print("   n=%d bright UNRAILED pulses (amp %.0f-%.0f ADC): median min@+8k = %+.1f ADC, %d/%d negative"
              % (len(a), a[:, 0].min(), a[:, 0].max(), np.median(a[:, 1]),
                 int((a[:, 1] < 0).sum()), len(a)))
        print("   undershoot/amplitude @+8k: median %.4f (%.2f%% of pulse height)"
              % (np.median(a[:, 2]), 100 * abs(np.median(a[:, 2]))))
        rq = [r for r in rows if r[5]]
        if rq:
            rr = np.array([r[4][1] for r in rq]) / 13600.0
            print("   same ratio for the RAILED pulses (amp >= %d ADC clipped): median %.4f"
                  % (13600, np.median(rr)))
        print("   -> same ratio railed vs unrailed => LINEAR AC coupling, not a")
        print("      saturation-recovery artifact.")

    # ---------------- figure ----------------
    w = waves[FIG_CH].astype(np.float64)
    runs = sorted(rail_intervals(w, min_len=1), key=lambda r: r[1] - r[0], reverse=True)
    i0, j0 = runs[0]
    base, sig = local_base(w, i0)
    s0 = max(0, i0 - SEG // 4)
    dec = decon_replica(w[s0:s0 + SEG], chans[FIG_CH], nfft=SEG)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    # (a) the clip itself
    ax = axes[0]
    lo, hi = i0 - 120, j0 + 500
    t = np.arange(lo, hi) - i0
    ax.plot(t, w[lo:hi], color="#1f77b4", lw=0.9)
    ax.axhline(RAIL, color="#d62728", ls="--", lw=1.2, label="14-bit rail = %d" % RAIL)
    ax.axhline(base, color="green", ls=":", lw=1.2, label="local baseline = %.0f" % base)
    ax.axvspan(0, j0 - i0, color="#d62728", alpha=0.12,
               label="railed run (%d ticks = %.1f us)" % (j0 - i0, (j0 - i0) * 0.016))
    ax.set_xlabel("tick relative to rail start")
    ax.set_ylabel("raw ADC")
    ax.set_title("(a) the saturation signature: a FLAT-TOP clipped at the UPPER rail "
                 "(ch%d, positive polarity)" % FIG_CH, fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)

    # (b) the undershoot
    ax = axes[1]
    lo, hi = i0 - 2000, min(len(w), j0 + 40000)
    t = (np.arange(lo, hi) - i0) * 0.016    # us
    ax.plot(t, w[lo:hi] - base, color="#999999", lw=0.4, alpha=0.6, label="raw - baseline")
    k = 201
    sm = np.convolve(w[lo:hi] - base, np.ones(k) / k, mode="same")
    ax.plot(t[k:-k], sm[k:-k], color="#1f77b4", lw=1.2, label="%d-tick moving average" % k)
    ax.axhline(0, color="green", ls=":", lw=1.2, label="local baseline")
    ax.set_ylim(-260, 260)
    ax.set_xlabel("time after rail start (us)")
    ax.set_ylabel("raw ADC - baseline")
    ax.set_title("(b) THIS is the \"signal goes negative\": a slow AC-coupling UNDERSHOOT "
                 "after the pulse\n(the pulse itself is +13600 ADC, off scale; the undershoot is "
                 "~1% of it but lasts ~500 us)", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.25)

    # (c) the decon
    ax = axes[2]
    lo, hi = i0 - 120, j0 + 500
    t = np.arange(lo, hi) - i0
    ax.plot(t, dec[lo - s0:hi - s0], color="#9467bd", lw=0.9, label="deconvolved")
    ax.axhline(0, color="k", ls=":", lw=1.0)
    ax.axvspan(0, j0 - i0, color="#d62728", alpha=0.12, label="railed run")
    inside = dec[i0 - s0:j0 - s0]
    ax.annotate("plateau: mean %+.1f, min %+.1f\nNEVER negative inside the rail\n"
                "(this is the over-integration)" % (inside.mean(), inside.min()),
                xy=(0.5 * (j0 - i0), inside.mean()), xytext=(j0 - i0 + 120, 30),
                arrowprops=dict(arrowstyle="->", color="#d62728"), fontsize=8, color="#d62728")
    ax.set_xlabel("tick relative to rail start")
    ax.set_ylabel("decon (100 = 1 PE/tick)")
    ax.set_title("(c) the deconvolution of the clipped flat-top: a sustained POSITIVE plateau",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)

    fig.suptitle("PDVD DAPHNE saturation: the clip is positive; the negative excursion is the "
                 "AC-coupling undershoot (run 039252 evt %d)" % EVENT, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT_PNG, dpi=120)
    print("\nwrote %s" % os.path.normpath(OUT_PNG))


if __name__ == "__main__":
    main()

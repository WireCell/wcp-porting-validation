#!/usr/bin/env python3
"""Prototype: ROI identification + ROI-based cleaning for the full-stream decon.

The -x full-stream decon (opch 120-159, 343 808 ticks @16ns) is one long record
dominated by baseline between sparse scintillation pulses, plus per-channel DC
offsets (opch 121: +0.19 decon) and a ringing channel (opch 147: MAD 3.04 vs
~0.02 clean).  A threshold OpHit finder over-produces flashes there.

This prototype implements (and validates in NumPy, before the C++ OpRoi
component) a ROI chain that cleans the decon waveform fed to OpHitFinder:

  1. HIGH-PASS the decon with H(f) = 1 - exp(-(f/tau)^2)  (the sigproc induction
     LfFilter form, util/src/Response.cxx).  Scintillation is < 20 us, so a
     corner tau ~ 1/20us = 0.05 MHz passes the pulses and removes slower
     baseline wander -- the "ROI-finding" waveform h.
  2. BASELINE: subtract median(h) (HPF already ~kills DC; belt-and-suspenders).
  3. NOISE + VETO: rms = 1.4826*MAD(h); if rms > veto_cap -> zero whole channel
     (the opch-147 ringing veto, carried over from robust_baseline).
  4. ROI (HYSTERESIS): contiguous runs of h > ext_nsigma*rms (low/extend) that
     reach seed_nsigma*rms somewhere inside (high/seed, ~ the hit threshold so a
     real pulse is required; runs that never reach it are dropped).  The run
     already spans down to the low threshold on both sides, so the ROI hugs the
     pulse and ends where the tail actually returns to noise -- bright pulses get
     ROIs comparable to dim ones instead of ballooning.
  5. PAD + MERGE: small symmetric pad (roi_pad ticks each side), merge overlaps.
     Replaces the old +700-tick blanket post-pad, which glued separate pulses
     (and their HPF ringing wings) into one 60-75 us block on bright channels and
     ramped the linear baseline (step 6) across the whole thing, destroying PE.
  6. APPLY to the ORIGINAL decon d: zero outside all ROIs; per ROI [s,e] subtract
     the line through (s,d[s]),(e,d[e]) so d[s]->0, d[e]->0.

The ROI chain SUPERSEDES the full-stream robust_baseline (user-confirmed): it
owns DC removal (per-ROI linear baseline) and the ringing veto, and OpHitFinder
reverts to the head method on the cleaned (already ~0-baseline) waveform.

Acceptance (printed + plotted):
  * slow/late scintillation light PRESERVED, no artificial undershoot;
  * opch 147 ringing rejected, opch 121 DC removed;
  * clean-channel real ~1-PE hits retained (sliding-window hit count delta).

Usage: fullstream_roi_proto.py [run] [evt]   (default 27980 8)
"""
import os, sys, io, tarfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDHD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(PDHD, "work")
PICS = os.path.join(PDHD, "pics")

# --- detector / OpHitFinder constants (flash.jsonnet full-stream values) ---
TICK_NS = 16.0
SAMPLE_FREQ_MHZ = 62.5          # 1/16ns
SCALE = 100.0                   # OpHitFinder m_scale: decon -> scaled short units
HIT_THRESHOLD = 11.0            # m_hit_threshold on pulse peak (scaled) = 0.11 decon

# --- ROI chain defaults (tuned here, become the C++ OpRoi defaults) ---
HPF_TAU_MHZ = 0.05             # 1 - exp(-(f/tau)^2) corner ~ 1/20us
ROI_SEED_NSIGMA = 5.0          # HIGH/seed: a run must reach this to be a real ROI (~hit thr, 0.1 decon)
ROI_EXT_NSIGMA = 3.0           # LOW/extend: ROI spans the contiguous run down to this (~0.06 decon)
ROI_PAD_PRE = 50               # pad before each run (ticks, 0.8 us) -- no early-side extension
ROI_POST_PEAK = 300            # ROI reaches >= this many ticks past the seed PEAK (4.8 us ~
                               # 3x the 1.6us late-light tau); a brighter tail above ext extends further
VETO_SIGMA = 0.1              # decon-unit MAD cap -> zero ringing channel (opch147)
ROI_HIT_NSIGMA = 3.0          # OpHitFinder start gate (robust_nsigma) on the noise floor

# AlgoSlidingWindow defaults (dune_ophit_finder_deco), scaled units.
ALGO = dict(adc_threshold=3.0, nsigma_threshold=1.0,
            tail_adc_threshold=1.0, tail_nsigma_threshold=1.0,
            end_adc_threshold=1.0, end_nsigma_threshold=1.0,
            min_pulse_width=1, num_presample=2, num_postsample=2)


def load_frames(bz2, evt):
    with tarfile.open(bz2) as t:
        decon = np.load(io.BytesIO(t.extractfile("frame_decon_%d.npy" % evt).read()))
        ch = np.load(io.BytesIO(t.extractfile("channels_decon_%d.npy" % evt).read()))
        ti = np.load(io.BytesIO(t.extractfile("tickinfo_decon_%d.npy" % evt).read()))
    return decon, ch, ti


def hpf_spectrum(n, tau_mhz):
    """H(f) = 1 - exp(-(f/tau)^2) on the rfft bins of an n-sample record."""
    f = np.fft.rfftfreq(n, d=1.0 / SAMPLE_FREQ_MHZ)   # MHz
    return 1.0 - np.exp(-(f / tau_mhz) ** 2)


def find_rois(h, d, ext_thr, seed_thr, pad_pre, post_peak, n):
    """HYSTERESIS ROIs: contiguous runs of h > ext_thr (low) that reach seed_thr
    (high) somewhere inside.  Each ROI is padded pad_pre before the run start and
    extended to at least post_peak ticks past the DECON pulse PEAK (the late-light
    window ~3x the 1.6us tau) -- a brighter tail that stays above ext extends
    further on its own.  Overlapping windows merge adjacent pulses."""
    above = h > ext_thr
    if not above.any():
        return []
    # run starts/ends of the boolean mask
    edges = np.diff(above.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0])
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(n - 1)
    rois = []
    for s, e in zip(starts, ends):
        if h[s:e + 1].max() < seed_thr:   # no real seed inside the run -> drop
            continue
        peak = s + int(np.argmax(d[s:e + 1]))   # the decon pulse peak
        ps = max(0, s - pad_pre)
        pe = min(n - 1, max(e, peak + post_peak))
        if rois and ps <= rois[-1][1]:
            rois[-1] = (rois[-1][0], max(rois[-1][1], pe))
        else:
            rois.append((ps, pe))
    return rois


def roi_clean(d, tau_mhz, seed_nsigma, ext_nsigma, pad_pre, post_peak, veto_cap):
    """Return (cleaned decon, rois, rms, vetoed) for one channel."""
    n = len(d)
    H = hpf_spectrum(n, tau_mhz)
    h = np.fft.irfft(np.fft.rfft(d) * H, n=n)
    h = h - np.median(h)
    rms = 1.4826 * np.median(np.abs(h - np.median(h)))
    if rms > veto_cap:
        return np.zeros_like(d), [], rms, True
    rois = find_rois(h, d, ext_nsigma * rms, seed_nsigma * rms, pad_pre, post_peak, n)
    out = np.zeros_like(d)
    for (s, e) in rois:
        seg = d[s:e + 1].astype(float)
        if e > s:
            base = seg[0] + (seg[-1] - seg[0]) * np.arange(e - s + 1) / (e - s)
        else:
            base = seg[0]
        out[s:e + 1] = seg - base
    return out, rois, rms, False


def sliding_window(wf, ped_mean, ped_sigma, nsigma_start):
    """Port of pmtana::AlgoSlidingWindow::RecoPulse (positive polarity, constant
    ped) -- identical to fullstream_baseline_proto.py.  Returns (s,e,t_max,peak)."""
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


def count_hits(decon_wf, ped_mean, ped_sigma, nsigma_start):
    wf = (SCALE * decon_wf).astype(np.int16).astype(float)  # match C++ short cast
    return sum(1 for (_, _, _, pk) in sliding_window(wf, ped_mean, ped_sigma, nsigma_start)
               if pk >= HIT_THRESHOLD)


def count_hits_roi(cleaned, ped_sigma_scaled, nsigma_start):
    """OpHit count on a ROI-cleaned waveform using the KNOWN clean noise floor.
    With tight (hysteresis) ROIs the in-ROI non-zero samples are signal-dominated,
    so estimating median/MAD from them biases ped_sigma high and the start gate
    closes (clean-channel hits crater to ~45%).  Feed the noise floor carried from
    the ROI step (the HPF rms, scaled) instead; the C++ OpHitFinder gets the same
    via a fixed pedestal-sigma knob.  Pedestal mean is 0 (ROI-cleaned baseline)."""
    wf = (SCALE * cleaned).astype(np.int16).astype(float)
    return sum(1 for (_, _, _, pk) in sliding_window(wf, 0.0, ped_sigma_scaled, nsigma_start)
               if pk >= HIT_THRESHOLD)


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 27980
    evt = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    bz2 = "%s/%06d_fs%d/light-frames-fullstream-wct.tar.bz2" % (WORK, run, evt)
    decon, ch, ti = load_frames(bz2, evt)
    n = decon.shape[1]
    print("run %d evt %d: decon %s, tau=%.3f MHz, seed=%.1f ext=%.1f sigma, pad_pre=%d post_peak=%d, veto=%.2f"
          % (run, evt, decon.shape, HPF_TAU_MHZ, ROI_SEED_NSIGMA, ROI_EXT_NSIGMA, ROI_PAD_PRE, ROI_POST_PEAK, VETO_SIGMA))

    rows = {}
    for opch in range(120, 160):
        idx = np.where(ch == opch)[0]
        if not len(idx):
            continue
        d = decon[int(idx[0])].astype(float)
        med = float(np.median(d)); mad = float(np.median(np.abs(d - med)) * 1.4826)
        cleaned, rois, rms, vetoed = roi_clean(d, HPF_TAU_MHZ, ROI_SEED_NSIGMA,
                                               ROI_EXT_NSIGMA, ROI_PAD_PRE, ROI_POST_PEAK, VETO_SIGMA)
        # OpHit counts: current robust_baseline (median ped + MAD on raw decon) vs
        # ROI-cleaned (ped 0, noise floor = the HPF rms carried from the ROI step).
        rob_mean = SCALE * med
        rob_sigma = SCALE * mad
        n_rob = count_hits(d, rob_mean, rob_sigma, 3.0)         # robust path
        n_roi = count_hits_roi(cleaned, rms * SCALE, ROI_HIT_NSIGMA)   # ROI-cleaned
        retained = float(np.count_nonzero(cleaned)) / len(cleaned)
        cls = ("vetoed" if vetoed else
               "ring" if mad >= VETO_SIGMA else
               "offset" if abs(med) >= 0.05 else "clean")
        rows[opch] = dict(d=d, cleaned=cleaned, rois=rois, rms=rms, vetoed=vetoed,
                          med=med, mad=mad, cls=cls, n_rob=n_rob, n_roi=n_roi,
                          retained=retained)

    BAD = sorted(c for c in rows if rows[c]["cls"] not in ("clean",))
    CLEAN = sorted(c for c in rows if rows[c]["cls"] == "clean")
    print("\n%d channels: %d clean, %d flagged (%s)"
          % (len(rows), len(CLEAN), len(BAD),
             ", ".join("%d/%s" % (c, rows[c]["cls"]) for c in BAD)))

    # --- record retention + ROI geometry (pad tuning) ---
    ret = np.mean([rows[c]["retained"] for c in CLEAN])
    nroi = np.mean([len(rows[c]["rois"]) for c in CLEAN])
    lens = [(e - s) * TICK_NS / 1000.0 for c in CLEAN for (s, e) in rows[c]["rois"]]
    print("\n[GEOMETRY] clean channels: record retained %.1f%%, %.1f ROIs/ch, "
          "ROI len us median=%.1f p90=%.1f max=%.1f"
          % (100 * ret, nroi, np.median(lens) if lens else 0,
             np.percentile(lens, 90) if lens else 0, max(lens) if lens else 0))

    # --- ACCEPTANCE 1: slow-tail PE preserved + no artificial undershoot. ---
    # Pick the brightest NON-over-subtracted clean channel (min decon > -1, so its
    # tail is real scintillation, not a v1-style negative over-subtraction).
    cand = [c for c in CLEAN if rows[c]["d"].min() > -1.0 and len(rows[c]["rois"])]
    bright = max(cand, key=lambda c: rows[c]["d"].max())
    rd = rows[bright]
    pk = int(np.argmax(rd["d"]))
    s, e = next(r for r in rd["rois"] if r[0] <= pk <= r[1])
    local_base = float(np.median(rd["d"][s:max(s + 1, pk - 60)]))  # pre-pulse baseline
    tail0 = min(e, pk + 31)                                        # peak + ~0.5 us
    tail_orig = (rd["d"][tail0:e + 1] - local_base).sum() / SCALE  # vs local baseline
    tail_clean = rd["cleaned"][tail0:e + 1].sum() / SCALE
    # the linear baseline OpRoi subtracted inside the ROI = original(local-sub) - cleaned;
    # it should be a gentle near-flat line, NOT a ramp eating the tail.
    subtracted = (rd["d"][s:e + 1] - local_base) - rd["cleaned"][s:e + 1]
    print("\n[ACCEPT 1] slow-tail preservation on opch %d (peak %.1f, min %.2f):"
          % (bright, rd["d"].max(), rd["d"].min()))
    print("  ROI [%d,%d] = %.1f us, local pre-pulse baseline = %.4f"
          % (s, e, (e - s) * TICK_NS / 1000.0, local_base))
    print("  tail PE [peak+0.5us, end]  original=%.2f  cleaned=%.2f  (abs delta %.3f PE)"
          % (tail_orig, tail_clean, tail_clean - tail_orig))
    print("  undershoot (min in ROI)    original=%.3f  cleaned=%.3f"
          % ((rd["d"][s:e + 1] - local_base).min(), rd["cleaned"][s:e + 1].min()))
    print("  linear baseline subtracted: range [%.4f, %.4f] over ROI (gentle if ~flat)"
          % (subtracted.min(), subtracted.max()))

    # --- ACCEPTANCE 2 + 3: bad channels collapse, clean hits retained. ---
    print("\n[ACCEPT 2/3] OpHit counts  robust_baseline -> ROI-cleaned:")
    for c in BAD:
        print("  opch %d (%s): %d -> %d" % (c, rows[c]["cls"], rows[c]["n_rob"], rows[c]["n_roi"]))
    rob_clean_tot = sum(rows[c]["n_rob"] for c in CLEAN)
    roi_clean_tot = sum(rows[c]["n_roi"] for c in CLEAN)
    print("  clean-channel total hits: %d -> %d (retain ~100%%; %.0f%%)"
          % (rob_clean_tot, roi_clean_tot, 100 * roi_clean_tot / max(1, rob_clean_tot)))
    rob_tot = sum(rows[c]["n_rob"] for c in rows)
    roi_tot = sum(rows[c]["n_roi"] for c in rows)
    print("  ALL-channel total hits:   %d -> %d (%.0f%% reduction)"
          % (rob_tot, roi_tot, 100 * (1 - roi_tot / max(1, rob_tot))))

    # --- Figure: representative channels, raw vs HPF vs cleaned with ROI spans. ---
    show = [bright]
    for cand in (123, 146, 121, 147):     # slow-tail / DC-offset / ringing
        if cand in rows and cand not in show:
            show.append(cand)
    show = show[:5]
    us = (ti[0] + ti[1] * (ti[2] + np.arange(n))) / 1000.0
    fig, axes = plt.subplots(len(show), 1, figsize=(13, 2.4 * len(show)), squeeze=False)
    for ax, c in zip(axes[:, 0], show):
        r = rows[c]
        H = hpf_spectrum(n, HPF_TAU_MHZ)
        h = np.fft.irfft(np.fft.rfft(r["d"]) * H, n=n); h -= np.median(h)
        # zoom to the busiest 60 us around the channel's peak (or full if vetoed)
        pk = int(np.argmax(np.abs(r["d"])))
        lo, hi = max(0, pk - 1500), min(n, pk + 2250)
        ax.plot(us[lo:hi], r["d"][lo:hi], lw=0.5, color="0.6", label="decon")
        ax.plot(us[lo:hi], h[lo:hi], lw=0.5, color="tab:orange", label="HPF (find)")
        ax.plot(us[lo:hi], r["cleaned"][lo:hi], lw=0.7, color="tab:blue", label="cleaned")
        ax.axhline(ROI_SEED_NSIGMA * r["rms"], ls=":", color="tab:green", lw=0.8,
                   label="seed=%.3f" % (ROI_SEED_NSIGMA * r["rms"]))
        ax.axhline(ROI_EXT_NSIGMA * r["rms"], ls=":", color="tab:olive", lw=0.6,
                   label="ext=%.3f" % (ROI_EXT_NSIGMA * r["rms"]))
        for (s, e) in r["rois"]:
            if e >= lo and s <= hi:
                ax.axvspan(us[max(s, lo)], us[min(e, hi - 1)], color="tab:blue", alpha=0.08)
        ax.set_title("opch %d (%s): med=%.3f MAD=%.3f rms_hpf=%.3f  hits %d->%d"
                     % (c, r["cls"], r["med"], r["mad"], r["rms"], r["n_rob"], r["n_roi"]),
                     fontsize=9)
        ax.set_ylabel("decon"); ax.legend(fontsize=6, ncol=4, loc="upper right")
    axes[-1, 0].set_xlabel("time [us]")
    fig.suptitle("run %d evt %d: full-stream ROI cleaning prototype" % (run, evt), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(PICS, exist_ok=True)
    out = "%s/pd_fullstream_%d_evt%d_roi_proto.png" % (PICS, run, evt)
    fig.savefig(out, dpi=95); print("\nwrote", out)


if __name__ == "__main__":
    main()

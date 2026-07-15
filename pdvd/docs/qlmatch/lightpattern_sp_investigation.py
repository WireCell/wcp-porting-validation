#!/usr/bin/env python3
"""Why do some flashes show measured light >> predicted (evt298567 flash 57)?

Repro (all read-only; run from pdvd/docs/qlmatch):
    python3 lightpattern_sp_investigation.py            # tables + figures
    python3 lightpattern_sp_investigation.py --tables   # tables only
    python3 lightpattern_sp_investigation.py --figs     # figures only

Inputs (existing pipeline products, never regenerated here):
  - work/039252_0_arcal/calib-evt298567.json   (arcal QL dump, sat flags)
  - work/039252_*_arcal/calib-evt*.json        (18-event candle round)
  - input_data_light_rawwf/np02vd_raw_run039252_*_rawwf.root
  - toolkit cfg pdvd-spe-templates.json / pdvd-opch-map.json
  - photlib/pdvd-photlib-vis-v5-128nm.json     (unmasked prediction replica)

Findings documented in pdvd-lightpattern-sp-investigation.md:
  1. sat-mask display artifact: dump pred_pe is zeroed on rail-flagged
     channels while dump pe is not -> viewer shows meas 12764 vs pred 0.
  2. self-trigger coverage blindness: membrane/PMT channels are 16.4 us
     snippets (duty ~5-30%); no snippet over the flash => measured
     silently 0 instead of "no data".
  3. broken SPE templates (ch2020 negative area, ch1051 x7, ch3010 x10,
     ch3020 x5, ch2011 x2.6) -> per-OpDet measured-PE scale errors.

Decon here is the NumPy replica of OpDecon's wiener-inspired branch
(validated <1% vs the pipeline dump in saturation_recovery_study.py
--validate; per-population sigma from flash.jsonnet: cathode 1.25 MHz,
membrane 1.0, PMT 3.5 -- pdvd/docs/pdvd-light-filter.md).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "ql_light_calib"))
from pd_mapping_audit import (RAWWF, PDVD, RAIL, rail_intervals)  # noqa: E402

FS = 62.5                    # MHz (16 ns samples), all DAPHNE streams
SNIP = 1024                  # self-trigger snippet length (samples)
EVENT = 298567
FLASH_GID = 57
DUMP = os.path.join(PDVD, "work/039252_0_arcal/calib-evt%d.json" % EVENT)
DUMPS_ALL = os.path.join(PDVD, "work/039252_*_arcal/calib-evt*.json")
TOOLKIT_CFG = os.path.normpath(os.path.join(
    PDVD, "..", "..", "toolkit", "cfg", "pgrapher", "experiment", "protodunevd"))
SPE_FILE = os.path.join(TOOLKIT_CFG, "pdvd-spe-templates.json")
MAP_FILE = os.path.join(TOOLKIT_CFG, "pdvd-opch-map.json")
LIB_FILE = os.path.join(PDVD, "photlib", "pdvd-photlib-vis-v5-128nm.json")

# adopted per-type VUVEfficiency scale factors (toolkit f7c66ab8)
EFF_SCALE = {"cath": 10.116, "memb": 1.655, "pmt": 0.352}
MEMB_ODS = [0, 1, 2, 3, 12, 13, 18, 19]
CATH_ODS = list(range(4, 12))
PMT_ODS = [od for od in range(14, 40) if od not in (18, 19)]
# wiener-inspired band-filter sigma per population (flash.jsonnet:15-16)
WI_SIGMA = {"cath": 1.25, "memb": 1.0, "pmt": 3.5}
WI_POWER = 2.0
WI_EPS_REL = 1e-3
NPED = 20                    # OpDecon pre_trigger(50) - pedestal_buffer(30)


def pop_of_ch(ch):
    return "cath" if ch < 2000 else ("memb" if ch < 3000 else "pmt")


def pop_of_od(od):
    return "cath" if od in CATH_ODS else ("memb" if od in MEMB_ODS else "pmt")


# ------------------------------------------------------------------ raw I/O
def load_all_streams(event=EVENT):
    """All raw records for one event: cathode full streams + every
    membrane/PMT self-trigger snippet.  Timestamps are microseconds
    (LArSoft OpDetWaveform convention -- same clock for both kinds)."""
    import uproot
    rfile = sorted(glob.glob(os.path.join(
        RAWWF, "np02vd_raw_run039252_*_rawwf.root")))[0]
    t = uproot.open(rfile)["rawdump/raw_waveform"]
    arr = t.arrays(["event", "opchannel", "timestamp"], library="np")
    sel = np.where(arr["event"] == event)[0]
    t0 = arr["timestamp"][sel].min()
    recs = {}
    for i in sel:
        c = int(arr["opchannel"][i])
        w = np.asarray(t["adc"].array(entry_start=i, entry_stop=i + 1,
                                      library="np")[0], np.float32)
        recs.setdefault(c, []).append((float(arr["timestamp"][i] - t0), w))
    for c in recs:
        recs[c].sort()
    return recs, t0


def load_spe():
    d = json.load(open(SPE_FILE))
    tmpl = {ch: np.asarray(d["templates"][ti]["values"], np.float64)
            for ch, ti in zip(d["channels"], d["template_index"])}
    return tmpl


def load_map():
    d = json.load(open(MAP_FILE))
    ch2od = {e["opch"]: e["opdet"] for e in d["channels"]}
    od2ch = {}
    for ch, od in ch2od.items():
        od2ch.setdefault(od, []).append(ch)
    return ch2od, od2ch


# ------------------------------------------------------------------ decon
def decon_replica(adc, template, sigma):
    """OpDecon wiener-inspired deconvolution (OpDecon.cxx:216-228, 391-93,
    437-41): G = conj(H) F / (|H|^2 + eps), F(0)=1 band filter, head-mean
    pedestal, post baseline re-zero.  Output units: PE per sample (a 1-PE
    pulse integrates to 1 because the DC gain is 1/H(0) = 1/template-area
    -- which is exactly what a negative or x7 template area corrupts)."""
    n = len(adc)
    ped = float(np.mean(adc[:NPED]))
    xv = np.asarray(adc, np.float64) - ped
    padded = np.zeros(n)
    padded[:min(len(template), n)] = template[:n]
    H = np.fft.fft(padded)
    eps = (WI_EPS_REL * np.abs(H).max()) ** 2
    f = (FS / n) * np.minimum(np.arange(n), n - np.arange(n))
    F = np.exp(-0.5 * (f / sigma) ** WI_POWER)
    dec = np.fft.ifft(np.conj(H) * F * np.fft.fft(xv) /
                      (np.abs(H) ** 2 + eps)).real
    dec -= dec[:NPED].mean()
    return dec


# ------------------------------------------------------------------ dump
def flash57(d=None):
    d = d or json.load(open(DUMP))
    f = [x for x in d["flashes"] if x["gid"] == FLASH_GID][0]
    return d, f


def unmasked_pair_pred(d):
    """Recompute the selected crosser pair's prediction WITHOUT the
    per-flash saturation mask, using the vis-loop replica (validated exact
    vs dump preds in fit_qtol_crossers --relib) x the adopted per-type
    efficiency scale factors."""
    import ablib_gold as ag
    lib = ag.GridLib(LIB_FILE)
    scale = np.ones(40)
    for od in range(40):
        scale[od] = EFF_SCALE[pop_of_od(od)]
    live = np.ones(40, bool)
    pred = np.zeros(40)
    for b in d["bundles"]:
        if b["flash_gid"] == FLASH_GID and b["main_cluster"] in (134, 4000056):
            pred += np.asarray(ag.predict(d, b, lib, live), float) * scale
    return pred


# ------------------------------------------------------------------ tables
def table_flash57(recs, tmpl, ch2od, d, f):
    toff = d["trigger_offsets_us"][0]
    t_light = f["time"] - toff
    print("=" * 100)
    print("T1. flash gid%d (t=%.1f us, total %.0f PE): cathode raw vs dump PE"
          % (FLASH_GID, f["time"], f["total_PE"]))
    print("=" * 100)
    print("%3s %3s %9s %9s %8s   %s" % (
        "od", "sat", "dumpPE", "rawPE", "dump/raw", "per-channel raw area / SPE area"))
    for od in CATH_ODS:
        chs = sorted(c for c, o in ch2od.items() if o == od)
        rawpe, det = 0.0, []
        for c in chs:
            ts, w = recs[c][0]
            s = int((t_light - ts) * FS)
            lo, hi = s - 125, s + int(22 * FS)
            base = np.median(w[max(0, lo - 3000):lo])
            a = float(np.clip(w[lo:hi] - base, 0, None).sum())
            sa = tmpl[c].sum()
            rawpe += a / sa
            rails = rail_intervals(w[lo:hi])
            det.append("%d: %.2fM/%.0f%s" % (
                c, a / 1e6, sa, " RAIL%d" % sum(b - a2 for a2, b in rails)
                if rails else ""))
        r = f["pe"][od] / rawpe if rawpe else float("nan")
        print("%3d %3d %9.1f %9.1f %8.2f   %s" % (
            od, f["sat"][od], f["pe"][od], rawpe, r, "  ".join(det)))
    print("\n=> decon PE is faithful to the raw waveform (0.8-1.4; >1 on"
          " railed ods = saturation repair recovering clipped area).")


def table_pred(d, f):
    pred = unmasked_pair_pred(d)
    masked = np.zeros(40)
    for b in d["bundles"]:
        if b["flash_gid"] == FLASH_GID and b["main_cluster"] in (134, 4000056):
            masked += np.asarray(b["pred_pe"], float)
    print("=" * 100)
    print("T2. measured vs the selected pair's prediction "
          "(cluster 134 + 4000056), X-Arapuca opdets")
    print("=" * 100)
    print("%3s %3s %9s | %13s %13s %9s" % (
        "od", "sat", "meas", "pred(dump)", "pred(unmask)", "meas/unm"))
    for od in MEMB_ODS[:4] + CATH_ODS + [12, 13]:
        r = f["pe"][od] / pred[od] if pred[od] > 1 else float("nan")
        print("%3d %3d %9.1f | %13.1f %13.1f %9.2f" % (
            od, f["sat"][od], f["pe"][od], masked[od], pred[od], r))
    print("\n=> the dump/viewer 'pred' is ZERO exactly on the sat-flagged"
          " ods 4/6/7/8; the unmasked prediction matches measured"
          " everywhere (0.5-1.25).  od7's 0.50 is the ch1051 template bug.")
    return pred


def table_coverage(recs, ch2od, d, f):
    toff = d["trigger_offsets_us"][0]
    t_light = f["time"] - toff
    print("=" * 100)
    print("T3. self-trigger snippet coverage of the flash window "
          "[%.1f, %.1f] us (membrane/PMT)" % (t_light, t_light + 20))
    print("=" * 100)
    print("%5s %4s %6s %6s   %s" % ("ch", "od", "#snip", "cover", "dump meas PE (opdet)"))
    for c in sorted(recs):
        if c < 2000:
            continue
        snips = recs[c]
        cov = sum(1 for ts, _ in snips
                  if ts < t_light + 20 and ts + SNIP / FS > t_light)
        od = ch2od[c]
        print("%5d %4d %6d %6s   %8.1f" % (
            c, od, len(snips), "YES" if cov else "no", f["pe"][od]))
    print("\n=> measured PE is nonzero exactly where a snippet covers the"
          " window; 'no' channels are scored 0 though they carry NO data.")


def table_prevalence():
    print("=" * 100)
    print("T4. P(measured>0) vs flash brightness, membrane/PMT opdets, "
          "18-event arcal round")
    print("=" * 100)
    bins = [(1000, 3000), (3000, 10000), (10000, 1e12)]
    cnt = {(g, b): [0, 0] for g in ("memb", "pmt") for b in range(3)}
    nsat = nfl = nsel = nselsat = 0
    for p in sorted(glob.glob(DUMPS_ALL)):
        d = json.load(open(p))
        satg = set()
        for fl in d["flashes"]:
            nfl += 1
            if sum(fl["sat"]) > 0:
                nsat += 1
                satg.add(fl["gid"])
            for bi, (lo, hi) in enumerate(bins):
                if lo <= fl["total_PE"] < hi:
                    for g, ods in (("memb", MEMB_ODS), ("pmt", PMT_ODS)):
                        for od in ods:
                            cnt[(g, bi)][0] += 1
                            cnt[(g, bi)][1] += fl["pe"][od] > 0
        for b in d["bundles"]:
            if b.get("auto_selected"):
                nsel += 1
                nselsat += b["flash_gid"] in satg
    for g in ("memb", "pmt"):
        for bi, (lo, hi) in enumerate(bins):
            n, k = cnt[(g, bi)]
            print("  %-4s totPE [%6.0f, %8.0f): P(meas>0) = %5d/%5d = %.2f"
                  % (g, lo, hi, k, n, k / n))
    print("  flashes with sat flags: %d/%d; auto-selected bundles on"
          " sat-flagged flashes: %d/%d = %.0f%%"
          % (nsat, nfl, nselsat, nsel, 100 * nselsat / nsel))
    print("\n=> flat ~0.3 in brightness: absence is readout duty cycle,"
          " not dim light.  ~70%% of membrane/PMT pattern entries are"
          " fake zeros.")


def table_anchors():
    print("=" * 100)
    print("T5. per-opdet median meas/pred over the 42 strict-crosser "
          "anchors (dump preds, keep-mask)")
    print("=" * 100)
    import fit_qtol_crossers as fq
    anchors = fq.harvest(sorted(glob.glob(DUMPS_ALL)))
    per = {od: [] for od in range(40)}
    for a in anchors:
        for od in range(40):
            if a["keep"][od] and a["pred"][od] > 5:
                per[od].append(a["meas"][od] / a["pred"][od])
    print("%3s %7s %4s %7s %s" % ("od", "type", "n", "median", "16-84%"))
    for od in range(40):
        v = np.array(per[od])
        if len(v) < 3:
            continue
        print("%3d %7s %4d %7.2f [%5.2f, %5.2f]" % (
            od, pop_of_od(od), len(v), np.median(v),
            np.percentile(v, 16), np.percentile(v, 84)))
    print("\n=> membrane opdets pinned at 0.00 = coverage blindness;"
          " od7 low+tight = ch1051 template; PMT zeros = coverage +"
          " ch3010/3020 templates.")


def table_templates(tmpl):
    print("=" * 100)
    print("T6. SPE template areas (OpDecon PE scale = 1/area); outliers")
    print("=" * 100)
    bands = {"cath": (400, 1500), "memb": (800, 2600), "pmt": (800, 2600)}
    print("%6s %5s %9s %7s" % ("ch", "pop", "area", "amp"))
    for c in sorted(tmpl):
        v = tmpl[c]
        pop = pop_of_ch(c)
        lo, hi = bands[pop]
        bad = not (lo <= v.sum() <= hi)
        print("%6d %5s %9.1f %7.2f%s" % (
            c, pop, v.sum(), v.max(), "   <-- OUTLIER" if bad else ""))
    print("\n=> ch2020 NEGATIVE (decon DC gain < 0: window-dependent,"
          " wrong-sign areas -- PE scale meaningless; its own 24 snippets"
          " are noise triggers); ch1051 x7 (od7 ~x2 low); ch3010 x10 (od14),"
          " ch3020 x5 (od15), ch2011 x2.6 (od0 half low).")


# ------------------------------------------------------------------ figures
def fig1_raw_decon(recs, tmpl, d, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    toff = d["trigger_offsets_us"][0]
    t_light = f["time"] - toff
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    cases = [(1020, "od4 sub-ch 1020 (RAILED; dump od4 = 12764 PE, sat=1)"),
             (1021, "od4 sub-ch 1021 (RAILED)"),
             (1040, "od10 sub-ch 1040 (unrailed; dump od10 = 2683 PE)"),
             (1041, "od10 sub-ch 1041 (unrailed)")]
    for ax, (c, title) in zip(axes.ravel(), cases):
        ts, w = recs[c][0]
        s = int((t_light - ts) * FS)
        lo, hi = s - int(3 * FS), s + int(22 * FS)
        tt = (np.arange(lo, hi) / FS) - (t_light - ts)
        base = np.median(w[max(0, lo - 3000):lo])
        ax.plot(tt, w[lo:hi] - base, lw=0.6, color="C0", label="raw - baseline")
        ax.axhline(RAIL - base, color="r", ls="--", lw=0.8,
                   label="DAPHNE rail (16383)")
        for a, b in rail_intervals(w[lo:hi]):
            ax.axvspan(tt[a], tt[min(b, len(tt) - 1)], color="r", alpha=0.15)
        dec = decon_replica(w, tmpl[c], WI_SIGMA["cath"])
        ax2 = ax.twinx()
        ax2.plot(tt, dec[lo:hi], lw=0.6, color="C1", alpha=0.8)
        ax2.set_ylabel("decon [PE/sample]", color="C1", fontsize=8)
        pe = dec[lo:hi][tt >= -1].sum()
        ax.set_title("%s\nclipped-decon integral %.0f PE" % (title, pe),
                     fontsize=9)
        ax.set_ylabel("ADC - baseline")
        ax.legend(fontsize=7, loc="upper right")
    axes[1][0].set_xlabel("t - flash time [us]")
    axes[1][1].set_xlabel("t - flash time [us]")
    fig.suptitle("evt%d flash gid%d: the big measured PEs are REAL light,\n"
                 "decon follows raw; red spans = rail runs the production "
                 "chain repairs+flags" % (EVENT, FLASH_GID), fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "lightpattern_fig1_raw_decon_cathode.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def fig2_od7_templates(recs, tmpl, d, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    toff = d["trigger_offsets_us"][0]
    t_light = f["time"] - toff
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1.2])
    pes = {}
    for k, c in enumerate((1050, 1051)):
        ax = fig.add_subplot(gs[0, k])
        ts, w = recs[c][0]
        s = int((t_light - ts) * FS)
        lo, hi = s - int(3 * FS), s + int(22 * FS)
        tt = (np.arange(lo, hi) / FS) - (t_light - ts)
        base = np.median(w[max(0, lo - 3000):lo])
        ax.plot(tt, w[lo:hi] - base, lw=0.6, color="C0")
        dec = decon_replica(w, tmpl[c], WI_SIGMA["cath"])
        ax2 = ax.twinx()
        ax2.plot(tt, dec[lo:hi], lw=0.6, color="C1")
        pes[c] = dec[lo:hi][tt >= -1].sum()
        raw_area = float(np.clip(w[lo:hi] - base, 0, None).sum())
        ax.set_title("ch%d  raw area %.2fM ADC-ticks\n"
                     "SPE area %.0f -> decon %.0f PE"
                     % (c, raw_area / 1e6, tmpl[c].sum(), pes[c]), fontsize=9)
        ax.set_xlabel("t - flash time [us]")
        ax.set_ylabel("ADC - baseline")
        ax2.set_ylabel("decon [PE/sample]", color="C1", fontsize=8)
    ax = fig.add_subplot(gs[0, 2])
    tt = np.arange(len(tmpl[1050])) / FS
    ax.plot(tt, tmpl[1050], label="ch1050 tmpl (area %.0f)" % tmpl[1050].sum())
    ax.plot(tt, tmpl[1051], label="ch1051 tmpl (area %.0f)" % tmpl[1051].sum())
    ax.set_xlabel("us")
    ax.set_ylabel("ADC per 1 PE")
    ax.legend(fontsize=7)
    ax.set_title("the x7 template-area bug", fontsize=9)
    fig.suptitle("OpDet 7 = ch1050 + ch1051: same physical light, but ch1051's "
                 "bloated SPE template\ndivides its PE by ~7 -> od7 measured "
                 "~x2 low (meas/pred 0.50)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "lightpattern_fig2_od7_template_bug.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def fig3_ch2020(recs, tmpl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # cleanest bright snippet per channel: no dropped/railed samples, flat
    # head for the pedestal, peak in the interior
    def pick(c):
        best, score = None, -1
        for ts, w in recs[c]:
            if w.min() <= 100 or w.max() >= RAIL:
                continue
            head = w[:100]
            if head.std() > 30:
                continue
            pk = int(np.argmax(w))
            if not 150 < pk < 900:
                continue
            s = w[pk] - np.median(head)
            if s > score:
                best, score = (ts, w), s
        return best or max(recs[c], key=lambda p: p[1].max() - np.median(p[1]))

    # ch2020's own 24 snippets are noise triggers (sick channel), so
    # demonstrate the inversion on the ganged sibling's REAL pulse (same
    # OpDet, same physical light) deconvolved with each template.
    ts, w = pick(2021)
    base = np.median(w[:100])
    tt = np.arange(len(w)) / FS
    pk = int(np.argmax(w))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for k, c in enumerate((2021, 2020)):
        ax = axes[k]
        ax.plot(tt, w - base, lw=0.7, color="C0", label="raw - baseline")
        dec = decon_replica(w, tmpl[c], WI_SIGMA["memb"])
        ax2 = ax.twinx()
        ax2.plot(tt, dec, lw=0.7, color="C1", label="decon")
        ax2.axhline(0, color="k", lw=0.4)
        ax2.set_ylabel("decon [PE/sample]", color="C1", fontsize=8)
        pe = dec[max(0, pk - 50):min(len(dec), pk + 300)].sum()
        ax.set_title("same ch2021 pulse deconvolved with the ch%d template\n"
                     "(SPE area %.0f) -> pulse integral %.1f PE"
                     % (c, tmpl[c].sum(), pe), fontsize=9)
        ax.set_xlabel("us in snippet")
        ax.set_ylabel("ADC - baseline")
    ax = axes[2]
    tt = np.arange(len(tmpl[2020])) / FS
    ax.plot(tt, tmpl[2021], label="ch2021 tmpl (area %.0f)" % tmpl[2021].sum())
    ax.plot(tt, tmpl[2020], label="ch2020 tmpl (area %.0f)" % tmpl[2020].sum())
    ax.fill_between(tt, tmpl[2020], 0, where=tmpl[2020] < 0,
                    color="r", alpha=0.2, label="negative lobe")
    ax.axhline(0, color="k", lw=0.4)
    ax.legend(fontsize=7)
    ax.set_xlabel("us")
    ax.set_title("ch2020 template NET AREA < 0\n(unrepaired AC undershoot)",
                 fontsize=9)
    fig.suptitle("OpDet 2 = ch2020 + ch2021: ch2020's negative-area template "
                 "makes the decon DC gain 1/H(0) negative ->\nthe same real "
                 "pulse loses ~40% in the pulse window and its total area is "
                 "window-dependent / wrong-sign: the PE scale is meaningless",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, "lightpattern_fig3_ch2020_inversion.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def fig4_coverage(recs, ch2od, d, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    toff = d["trigger_offsets_us"][0]
    t_light = f["time"] - toff
    chans = [c for c in sorted(recs) if c >= 2000]
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(13, 9), sharey=True,
        gridspec_kw={"width_ratios": [4, 1]})
    for y, c in enumerate(chans):
        for ts, _ in recs[c]:
            ax.plot([ts / 1000, (ts + SNIP / FS) / 1000], [y, y],
                    color="C0", lw=2.5, solid_capstyle="butt")
    ax.axvspan(t_light / 1000, (t_light + 20) / 1000, color="r", alpha=0.35,
               label="flash gid%d window" % FLASH_GID)
    ax.set_yticks(range(len(chans)))
    ax.set_yticklabels(["%d (od%d)" % (c, ch2od[c]) for c in chans],
                       fontsize=7)
    ax.set_xlabel("time in readout [ms]")
    ax.set_title("membrane/PMT self-trigger snippets, evt%d: each channel is "
                 "LIVE only inside its 16.4-us snippets" % EVENT, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    meas = [f["pe"][ch2od[c]] for c in chans]
    cover = [any(ts < t_light + 20 and ts + SNIP / FS > t_light
                 for ts, _ in recs[c]) for c in chans]
    axr.barh(range(len(chans)),
             [np.log10(max(m, 0.5)) for m in meas],
             color=["C2" if cv else "0.7" for cv in cover])
    axr.set_xlabel("log10 dump PE (opdet)")
    axr.set_title("green = snippet covers flash\ngrey = NO DATA (scored 0!)",
                  fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "lightpattern_fig4_coverage_timeline.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def fig5_pattern(d, f, pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    masked = np.zeros(40)
    for b in d["bundles"]:
        if b["flash_gid"] == FLASH_GID and b["main_cluster"] in (134, 4000056):
            masked += np.asarray(b["pred_pe"], float)
    ods = MEMB_ODS[:4] + CATH_ODS + [12]
    x = np.arange(len(ods))
    meas = [f["pe"][od] for od in ods]
    sat = [f["sat"][od] for od in ods]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x - 0.2, meas, 0.4, color="C0", label="measured")
    for b, s in zip(bars, sat):
        if s:
            b.set_hatch("//")
            b.set_edgecolor("r")
    ax.bar(x + 0.2, [masked[od] for od in ods], 0.4, color="0.7",
           label="pred in dump (sat-MASKED - what the viewer shows)")
    ax.plot(x, [pred[od] for od in ods], "o-", color="C3",
            label="pred UNMASKED (pair 134+4000056, scaled)")
    ax.set_yscale("log")
    ax.set_ylim(1, 3e4)
    ax.set_xticks(x)
    ax.set_xticklabels(["od%d%s" % (od, "*" if f["sat"][od] else "")
                        for od in ods])
    ax.set_ylabel("PE")
    ax.legend(fontsize=9)
    ax.set_title("evt%d flash gid%d: the 'discrepancy' is the dump zeroing pred "
                 "on rail-flagged ods (hatched);\nthe unmasked prediction "
                 "matches measured everywhere (od7 low = ch1051 template)"
                 % (EVENT, FLASH_GID), fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, "lightpattern_fig5_pattern.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--figs", action="store_true")
    args = ap.parse_args()
    do_tables = args.tables or not args.figs
    do_figs = args.figs or not args.tables

    recs, _ = load_all_streams()
    tmpl = load_spe()
    ch2od, _ = load_map()
    d, f = flash57()

    if do_tables:
        table_flash57(recs, tmpl, ch2od, d, f)
        pred = table_pred(d, f)
        table_coverage(recs, ch2od, d, f)
        table_prevalence()
        table_anchors()
        table_templates(tmpl)
    else:
        pred = unmasked_pair_pred(d)
    if do_figs:
        fig1_raw_decon(recs, tmpl, d, f)
        fig2_od7_templates(recs, tmpl, d, f)
        fig3_ch2020(recs, tmpl)
        fig4_coverage(recs, ch2od, d, f)
        fig5_pattern(d, f, pred)


if __name__ == "__main__":
    main()

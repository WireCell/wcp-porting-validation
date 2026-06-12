#!/usr/bin/env python3
"""Tune & validate the Q/L bundle over-prediction prefilter on the 20 hand-scans.

The QLMatching prefilter drops a (flash x cluster) bundle BEFORE the chi2 fit when
its PREDICTED light is much larger than the MEASURED light: a charge cluster that
would emit far more light than the flash actually shows cannot be the right match.
It is one-directional -- under-prediction (pred < meas) is physically fine (a flash
may be lit by several clusters; near-PMT/truncation make measured an underestimate),
so only over-prediction is rejected.

Two metrics, computed over the masked same-TPC PMT channels (the same opdet_mask the
chi2 uses: type==PMT & apa==flash_apa & active):

    R_total = sum(pred_masked) / sum(meas_masked)
    R_max   = pred[ch*] / max(meas[ch*], FLOOR),  ch* = argmax(pred_masked)

Hard constraint: the cut must NOT remove any hand-scanned (ground-truth) bundle.
We tune the ceilings on DATA and validate on MC.  The GT baseline is NOT 1.0 -- true
matches over-predict (ql_pe_error.py: median pred/meas ~1.3) and the baseline shifts
data<->MC (data_qtol=0.86 vs sim=1.0), so the cut is set above the worst-case
non-boundary GT with margin and then re-checked against MC.

Boundary/truncated GT (close_to_PMT | window_truncated | at_x_boundary) are EXEMPT
from the reject (measured is an underestimate there) -- mirrors ql_pe_error.py
dropping exactly those flashes and the prototype's boundary-aware reject.

Inputs (reused from ql_pe_error.py):
  ground truth : work/ql_labels/<mode>/.scan_state-evt<ID>.json  ("selected" pairs)
  NL-on dumps  : work/ql_nl_study/calib-evt<ID>.nl.json          (pmt_nl=true, production)

Usage:  python3 ql_prefilter_tune.py            # data + mc
        python3 ql_prefilter_tune.py data
"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PICS = os.path.join(HERE, "pics")
NL_DIR = os.path.join(HERE, "work", "ql_nl_study")

PMT_TYPE = 1          # opdet type for PMTs (type 0 = X-ARAPUCA, no light)
FLOOR = 1.0           # PE floor in R_max denominator (avoid divide-by-tiny)
MARGIN = 1.5          # ceiling = worst non-boundary GT * MARGIN
BOUNDARY = ("close_to_PMT", "window_truncated", "at_x_boundary")


def calib_path(ev):
    return os.path.join(NL_DIR, "calib-evt%d.nl.json" % ev)


def bundle_metrics(b, flash_pe, flash_apa, ch_keep_by_apa):
    """R_total, R_max and the prototype fired-fraction for one bundle."""
    gid = b["flash_gid"]
    pred = np.asarray(b["pred_pe"], float)
    meas = flash_pe[gid]
    keep = ch_keep_by_apa[flash_apa[gid]]
    p, m = pred[keep], meas[keep]
    tot_p, tot_m = float(p.sum()), float(m.sum())
    r_total = tot_p / tot_m if tot_m > 0 else np.inf
    if p.max() > 0:
        cstar = int(np.argmax(p))
        r_max = p[cstar] / max(m[cstar], FLOOR)
    else:
        r_max = 0.0
    # prototype fired-fraction (normalization-robust fallback): channels predicted
    # to fire (pred>0.33) where measured >= 0.33*pred.
    fire = p > 0.33
    ntot = int(fire.sum())
    nfired = int((m[fire] > 0.33 * p[fire]).sum()) if ntot else 0
    return dict(r_total=r_total, r_max=r_max, ntot=ntot, nfired=nfired)


def load_event(ev, mode):
    """Per-bundle records for one event: metrics + GT flag + boundary flag."""
    state = os.path.join(HERE, "work", "ql_labels", mode, ".scan_state-evt%d.json" % ev)
    want = {tuple(k) for k in json.load(open(state))["selected"]}
    cal = json.load(open(calib_path(ev)))
    ch_apa = np.array([o["apa"] for o in cal["opdets"]])
    ch_type = np.array([o["type"] for o in cal["opdets"]])
    ch_active = np.array([bool(o["active"]) for o in cal["opdets"]])
    flash_pe = {f["gid"]: np.asarray(f["pe"], float) for f in cal["flashes"]}
    flash_apa = {f["gid"]: f["apa"] for f in cal["flashes"]}
    apas = set(flash_apa.values())
    ch_keep_by_apa = {a: (ch_type == PMT_TYPE) & (ch_apa == a) & ch_active for a in apas}

    seen_gt = set()
    recs = []
    for b in cal["bundles"]:
        key = (b["flash_gid"], b["main_cluster"])
        is_gt = key in want
        if is_gt:
            seen_gt.add(key)
        rec = bundle_metrics(b, flash_pe, flash_apa, ch_keep_by_apa)
        rec["is_gt"] = is_gt
        rec["boundary"] = any(b[f] for f in BOUNDARY)
        rec["key"] = key
        recs.append(rec)
    missing_gt = want - seen_gt          # GT pairs absent from the contained dump
    return recs, want, missing_gt


def load_mode(events, mode):
    all_recs, n_missing = [], 0
    print("\n[%s] events: %s" % (mode, events))
    for ev in events:
        recs, want, missing = load_event(ev, mode)
        if missing:
            n_missing += len(missing)
            print("  !! evt%d: %d GT pair(s) NOT in contained dump (dropped by "
                  "geometry containment): %s" % (ev, len(missing), sorted(missing)))
        all_recs.extend(recs)
    n_gt = sum(r["is_gt"] for r in all_recs)
    print("  bundles=%d  GT=%d  GT-missing(geom)=%d" % (len(all_recs), n_gt, n_missing))
    return all_recs, n_missing


def gt_summary(recs, label):
    gt = [r for r in recs if r["is_gt"]]
    nb = [r for r in gt if not r["boundary"]]
    bd = [r for r in gt if r["boundary"]]
    print("\n  -- %s GT (%d total: %d non-boundary, %d boundary-exempt) --"
          % (label, len(gt), len(nb), len(bd)))
    if nb:
        rt = np.array([r["r_total"] for r in nb])
        rm = np.array([r["r_max"] for r in nb])
        print("     non-boundary GT  R_total: max=%.2f  (median=%.2f)" % (rt.max(), np.median(rt)))
        print("     non-boundary GT  R_max  : max=%.2f  (median=%.2f)" % (rm.max(), np.median(rm)))
    if bd:
        rt = np.array([r["r_total"] for r in bd])
        rm = np.array([r["r_max"] for r in bd])
        print("     boundary GT      R_total: max=%.2f   R_max: max=%.2f" % (rt.max(), rm.max()))
    return nb, bd


def apply_cut(recs, ct, cm):
    """Returns (gt_removed, nongt_removed, nongt_total). Boundary bundles exempt."""
    gt_removed = nongt_removed = nongt_total = 0
    for r in recs:
        if not r["is_gt"]:
            nongt_total += 1
        if r["boundary"]:
            continue
        rejected = (r["r_total"] > ct) or (r["r_max"] > cm)
        if rejected:
            if r["is_gt"]:
                gt_removed += 1
            else:
                nongt_removed += 1
    return gt_removed, nongt_removed, nongt_total


def make_plot(data_recs, mc_recs, ct, cm):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for a, (recs, title) in zip(ax, [(data_recs, "DATA"), (mc_recs, "MC")]):
        for sel, c, lab, mk in [
                (lambda r: r["is_gt"] and not r["boundary"], "C2", "GT (non-bnd)", "*"),
                (lambda r: r["is_gt"] and r["boundary"], "C0", "GT (bnd, exempt)", "P"),
                (lambda r: not r["is_gt"], "0.6", "non-GT", ".")]:
            xs = [min(r["r_total"], 50) for r in recs if sel(r)]
            ys = [min(r["r_max"], 50) for r in recs if sel(r)]
            a.scatter(xs, ys, s=(70 if "GT" in lab else 10),
                      c=c, marker=mk, alpha=0.8 if "GT" in lab else 0.3,
                      label="%s (n=%d)" % (lab, len(xs)), zorder=3 if "GT" in lab else 1)
        a.axvline(ct, color="C3", ls="--", lw=1.3, label="R_total cut=%.1f" % ct)
        a.axhline(cm, color="C1", ls="--", lw=1.3, label="R_max cut=%.1f" % cm)
        a.set_xlabel("R_total = sum(pred)/sum(meas)  [masked PMT, clip 50]")
        a.set_ylabel("R_max = pred[argmax]/meas  [clip 50]")
        a.set_title("%s -- reject region = upper/right (non-boundary only)" % title)
        a.set_xscale("log"); a.set_yscale("log")
        a.legend(loc="upper left", fontsize=8)
    fig.suptitle("Q/L over-prediction prefilter: GT must stay below both cuts", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(PICS, exist_ok=True)
    out = os.path.join(PICS, "ql_prefilter_tune.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("\n  wrote", out)


def events_for(mode):
    sd = os.path.join(HERE, "work", "ql_labels", mode)
    return sorted(int(re.search(r"evt(\d+)", os.path.basename(p)).group(1))
                  for p in glob.glob(os.path.join(sd, ".scan_state-evt*.json")))


def main(modes):
    recs = {}
    miss = {}
    for mode in modes:
        recs[mode], miss[mode] = load_mode(events_for(mode), mode)

    if "data" not in recs:
        print("\nNeed data mode to tune.  Done."); return

    nb_data, _ = gt_summary(recs["data"], "DATA")
    if "mc" in recs:
        gt_summary(recs["mc"], "MC")

    # ---- choose ceilings from worst-case non-boundary DATA GT * MARGIN ----
    if not nb_data:
        print("\nNo non-boundary DATA GT -- cannot tune."); return
    rt_worst = max(r["r_total"] for r in nb_data)
    rm_worst = max(r["r_max"] for r in nb_data)
    ct = round(rt_worst * MARGIN, 1)
    cm = round(rm_worst * MARGIN, 1)
    print("\n==== chosen cuts (worst non-boundary DATA GT x %.1f) ====" % MARGIN)
    print("   overpred_total_ratio = %.2f   (worst GT R_total=%.2f)" % (ct, rt_worst))
    print("   overpred_maxch_ratio = %.2f   (worst GT R_max  =%.2f)" % (cm, rm_worst))

    for mode in modes:
        gtr, ngr, ngt = apply_cut(recs[mode], ct, cm)
        status = "OK" if gtr == 0 else "FAIL -- removes GT!"
        print("   [%s] GT-removed=%d (%s)  non-GT removed=%d/%d (%.0f%%)  + geom-missing GT=%d"
              % (mode, gtr, status, ngr, ngt, 100.0 * ngr / max(ngt, 1), miss[mode]))

    if "data" in recs and "mc" in recs:
        make_plot(recs["data"], recs["mc"], ct, cm)


if __name__ == "__main__":
    main(sys.argv[1:] or ["data", "mc"])

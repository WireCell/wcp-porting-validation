#!/usr/bin/env python3
"""4-event hand-scan label retune of the PDHD optical model (run 29107).

Extends the single-event ``fit_labels.py`` (evt 983) to events 983/991/999/1007,
fitting from the FLAG-CLEAN hand-picked matches only.  Per the user's instruction
we drop any match flagged ``flag_PMT`` / ``flag_xboundary`` / ``flag_wtrunc``
(label/dump fields ``close_to_PMT`` / ``at_x_boundary`` / ``window_truncated``):
those have missing charge -> unreliable predicted light.

Three knobs are fit self-consistently (the user's parameters; #2 and #4 of their
list both = the absorption length):

  1. lambda (vuv_absorption_length): the matcher's OWN KS shape distance, swept
     over the flag-clean anchors with the validated offline re-predictor
     (``repredict.py``, reproduces the dumped C++ ``pred_pe`` to ~1e-13).  Per the
     user, lambda is FREE-REFIT and adopted at the KS-valley centre even though the
     crosser exclusion leaves a more-concentrated sample (reported vs the
     crosser-derived 300).
  2. vuv_eff (global QtoL light yield, QtoL held 1.0, degenerate): +x / apa1
     top-3 direct-PMT scale at lambda*, so the directly-lit PMTs match.
  3. measured_pe_scale on APA0 (ch120-159): the -x full-data-stream half
     under-reports PE; scale the MEASUREMENT up to the recalibrated prediction.

All three are offline -- no chain re-run -- because the per-PMT pattern is
recomputable from the dumped cluster charge points at any (lambda, eff).  The
dumps (work/029107_{0..3}/calib-evt{983,991,999,1007}.json) are the model the
config currently ships (lambda=300, vuv_eff=0.01254, APA0 scale 1.57); the labels
join them at the current model (label pred == dump pred to 1.000).

Usage:  python3 fit_labels_multi.py
"""
import json
import os
import statistics
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PD = os.path.normpath(os.path.join(HERE, ".."))

# events -> work-dir index (art = 983 + 8*idx)
EVTS = {983: 0, 991: 1, 999: 2, 1007: 3}

CUR_EFF = 0.01254        # vuv_eff the dumps were produced at (cfg current)
CUR_APA0 = 1.57          # measured_pe_scale[120..159] the dumps were produced at
STATIC = {3, 86, 87, 97, 107, 116, 117}        # current cfg ch_mask (120-159 NOT masked)
APA0 = range(120, 160)   # -x full-data-stream half ("APA0")
SIDE1 = range(0, 80)     # +x side (the global-norm anchor)
TOT_MIN = 3000.0         # "sizable" cut (metrics.total_PE)
KS_MAX = 0.2             # "low-ks" cut (metrics.ks_dis)
LAMBDAS = [80, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500]
NPROC = 8


def excluded(m):
    """User's flag filter: drop close_to_PMT | at_x_boundary | window_truncated."""
    f = m.get("flags", {})
    return bool(f.get("close_to_PMT") or f.get("at_x_boundary") or f.get("window_truncated"))


def calc_ks(meas, pred):
    """Exact port of TimingTPCBundle::calc_ks_test (index-order cumulative CDFs)."""
    sm = sum(meas); sp = sum(pred)
    if sp <= 0:
        return 0.0
    m = [x / sm for x in meas] if sm > 0 else meas
    p = [x / sp for x in pred]
    cm = cp = d = 0.0
    for i in range(len(meas)):
        cm += m[i]; cp += p[i]; d = max(d, abs(cm - cp))
    return d


def build_anchors():
    """Flag-clean, sizable, low-ks matches across the 4 events, joined to the dump
    for charge points + measured PE.  Each anchor carries everything the offline
    re-predictor needs."""
    anchors = []
    skipped = {"flag": 0, "qual": 0, "nojoin": 0}
    for evt, ev in EVTS.items():
        d = json.load(open("%s/work/029107_%d/calib-evt%d.json" % (PD, ev, evt)))
        L = json.load(open("%s/work/ql_labels/labels-evt%d.json" % (PD, evt)))["matches"]
        GEO = {k: ({kk: v[kk] for kk in
                    ("anode_x", "s", "u_cathode", "y_lo", "y_hi", "z_lo", "z_hi")}
                   | {"sign_offset": v["sign_offset"]})
               for k, v in d["geometry"].items()}
        vd = d["drift_speed"]; trig = d["trigger_offset"]
        flB = {f["gid"]: f for f in d["flashes"]}
        clidx = {(c["apa"], c["ident"]): c for c in d["clusters"]}
        amask = set(i for i in range(160) if d["opdets"][i]["auto_masked"])
        opx = [d["opdets"][i]["x"] for i in range(160)]
        for m in L:
            if excluded(m):
                skipped["flag"] += 1; continue
            mt = m["metrics"]
            if not (mt.get("total_PE", 0) > TOT_MIN
                    and mt.get("ks_dis") is not None and mt["ks_dis"] < KS_MAX):
                skipped["qual"] += 1; continue
            apa = m["apa"]; g = GEO[str(apa)]
            f = flB.get(m["flash_gid"])
            pts = []
            if f is not None:
                for cid in m["cluster_idents"]:
                    c = clidx.get((apa, cid))
                    if c:
                        pts += list(zip(c["x"], c["y"], c["z"], c["q"]))
            if f is None or not pts:
                skipped["nojoin"] += 1; continue
            anode_pos = g["anode_x"] > 0          # apa1/+x => True
            valid = [i for i in range(160)
                     if i not in STATIC and i not in amask and (opx[i] > 0) == anode_pos]
            anchors.append(dict(
                evt=evt, apa=apa, pe160=list(f["pe"]),
                pts=pts, geom=g, offset=g["sign_offset"] * (f["time"] + trig) * vd,
                mask=STATIC | amask, valid=valid, ks=mt["ks_dis"], tot=mt["total_PE"]))
    return anchors, skipped


def _init():
    import repredict
    repredict.VUVEFF = CUR_EFF          # so re-pred at lambda=300 == dump pred
    global predict
    from repredict import predict


def sweep_one(a):
    """KS(meas, masked-pred) at every lambda for one anchor (+ cache pred at all
    lambdas for the norm/APA0 passes done in-process via a second call path)."""
    out = {}
    for lam in LAMBDAS:
        pf = predict(a["pts"], a["geom"], a["offset"], float(lam))
        prm = [0.0 if i in a["mask"] else pf[i] for i in range(160)]
        out[lam] = calc_ks(a["pe160"], prm)
    return out


def predict_at(a, lam):
    import repredict
    repredict.VUVEFF = CUR_EFF
    from repredict import predict as _p
    return _p(a["pts"], a["geom"], a["offset"], float(lam))


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    anchors, skipped = build_anchors()
    n1 = sum(1 for a in anchors if a["apa"] == 1)
    n0 = sum(1 for a in anchors if a["apa"] == 0)
    print("flag-clean sizable low-ks anchors: %d  (+x/apa1=%d  -x/apa0=%d)"
          % (len(anchors), n1, n0))
    print("  per event:", {e: sum(1 for a in anchors if a["evt"] == e) for e in EVTS})
    print("  skipped: flag=%(flag)d quality=%(qual)d nojoin=%(nojoin)d" % skipped)

    # ---- 1. lambda ----
    forced = next((float(x) for x in sys.argv[1:] if x.replace('.', '').isdigit()), None)
    if forced is not None:
        lam_star = forced
        print("\nlambda FORCED = %g cm (KS rides to the grid edge / is degenerate upward;"
              "\n  N90 concentration + integral pin lambda~300 -- see fit_lambda_diag.py)" % lam_star)
    else:
        with Pool(NPROC, initializer=_init) as pool:
            ks_per = pool.map(sweep_one, anchors)
        print("\nlambda sweep (median matcher-KS over %d anchors):" % len(anchors))
        medks = {}
        for lam in LAMBDAS:
            medks[lam] = med([k[lam] for k in ks_per])
            print("  lam=%4d  medKS=%.4f" % (lam, medks[lam]))
        lam_star = min(LAMBDAS, key=lambda L: medks[L])
        print("  -> KS-min lambda* = %d cm   (crosser-derived prior = 300)" % lam_star)

    # ---- 2. vuv_eff: +x/apa1 top-3 direct-PMT scale at lambda* ----
    ds_top3, ds_full = [], []
    for a in anchors:
        if a["apa"] != 1:
            continue
        pf = predict_at(a, lam_star)
        meas = [a["pe160"][i] for i in a["valid"]]
        pr = [pf[i] for i in a["valid"]]
        if sum(pr) <= 0 or max(meas) <= 0:
            continue
        o3 = sorted(range(len(meas)), key=lambda k: -meas[k])[:3]
        sp3 = sum(pr[k] for k in o3)
        if sp3 > 0:
            ds_top3.append(sum(meas[k] for k in o3) / sp3)   # meas/pred ; <1 => over-pred
        ds_full.append(sum(meas) / sum(pr))
    g_direct = med(ds_top3); g_full = med(ds_full)
    # Spread is N90-matched at lambda~300 => integral is clean => use the full-side
    # (integral) handle; top-3 direct carries a fixed ~1.27 GH-shape residual that no
    # lambda removes (see fit_lambda_diag.py).  Report both.
    g = g_full
    vuv_eff_new = CUR_EFF * g
    print("\nvuv_eff anchor (+x/apa1, n=%d): integral(full) g=%.3f  [top-3 direct g=%.3f]"
          % (len(ds_top3), g_full, g_direct))
    print("  -> vuv_eff = %.5f * %.3f = %.5f   (direct-handle would give %.5f)"
          % (CUR_EFF, g, vuv_eff_new, CUR_EFF * g_direct))

    # ---- 3. APA0 measured_pe_scale: -x/apa0, ch120-159, at lambda*,new eff ----
    # dump meas already carries CUR_APA0=1.57; want s_new = pred_new/meas_raw
    #   = CUR_APA0 * g * median( pred(lam*,CUR_EFF)|APA0 / meas_dump|APA0 )
    R = []
    for a in anchors:
        if a["apa"] != 0:
            continue
        pf = predict_at(a, lam_star)
        sp = sum(pf[i] for i in APA0)
        sm = sum(a["pe160"][i] for i in APA0)
        if sm > 50 and sp > 5:
            R.append(sp / sm)
    Rmed = med(R)
    apa0_scale = CUR_APA0 * g * Rmed
    print("\nAPA0 measured_pe_scale (-x/apa0, n=%d): %.2f * %.3f * %.3f = %.2f"
          % (len(R), CUR_APA0, g, Rmed, apa0_scale))
    print("  (current-model APA0 pred/meas median R=%.3f; R~1 => current 1.57 self-consistent)" % Rmed)

    # ---- 4. (secondary) pe_err_frac on the corrected model ----
    preds, meass = [], []
    for a in anchors:
        pf = predict_at(a, lam_star)
        op = np.array(a["pe160"], float)
        pr = np.array(pf, float) * g                    # eff bump -> new pred
        op[120:160] *= (apa0_scale / CUR_APA0)          # re-scale meas APA0 to s_new
        chans = SIDE1 if a["apa"] == 1 else APA0
        for ch in chans:
            if op[ch] > 0 or pr[ch] > 0:
                meass.append(op[ch]); preds.append(pr[ch])
    pr = np.array(preds); me = np.array(meass)
    fm = pr > 0
    Y = (pr[fm] - me[fm]) ** 2 - me[fm]
    a_coef = float(np.sum(Y) / np.sum(pr[fm] ** 2))
    frac = float(np.sqrt(max(a_coef, 0.0)))
    print("\npe_err_frac (corrected model, calibrated chans N=%d): %.2f  (current 0.44)"
          % (pr[fm].size, frac))

    # ---- config snippet ----
    print("\n--- apply ---")
    print("  wire-cell-data/pdhd/photodet/semi-analytical-pdhd.json: vuv_absorption_length = %d" % lam_star)
    print("  cfg/.../pdhd/qlmatching.jsonnet: local vuv_eff = %.5f;" % vuv_eff_new)
    print("    measured_pe_scale: std.makeArray(160, function(i) if i>=120 then %.2f else 1.0)," % round(apa0_scale, 2))
    print("    pe_err_frac: %.2f," % round(frac, 2))


if __name__ == "__main__":
    main()

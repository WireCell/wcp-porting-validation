#!/usr/bin/env python3
"""doc pdvd/86 -- doc 78 action item 6, read before the hand check.

Item 6 says the four items have "no fitted charge near the stop; the Michel is
unimaged or in a dead region".  This reads, per item, what the production arm's
prep payload holds near the stop:

  A. the chain's answer (is_stm, michel_found, movers) and the record's reading;
  B. every imaged point within 5 / 10 / 20 cm of the fit end, by Bee cluster and
     bundle flag (image_near is Bee clustering-global, all clusters within 60 cm
     of the stop; doc pdhd/13 sec 4);
  C. the muon cluster's own points within 10 cm of the fit end, by distance to the
     nearest fit point (the chain's muon rows plus every PR segment of the cluster),
     against a control: the same cluster's points around a body stretch rr 30-40 cm
     upstream (the halo width with no Michel in it);
  D. the fit rows from the record's pin to the fit end (dQ/dx, k e/cm);
  E. T_bad_ch bands within 15 channels of the fit end's own U/V/W coordinate, at
     its time slice (pu/pv/pw and T_bad_ch are the same plane-rank scheme,
     PdvdPrMagnifyTrackingVisitor ChanScheme, so no channel map is involved).

Offline reading printed per item: "on-fit" when no own-cluster point within 10 cm
of the fit end lies > 2 cm off the fit (the control's own maximum), else "off-fit".

Usage: STM_SCAN_RECORD=<merged record> d86_offfit.py --prep PREP
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ITEMS = ["039349_43/66", "039349_72/11", "039253_0/44", "039349_30/45"]   # doc 78 sec 5 item 6
ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
a = ap.parse_args()
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a+smx3+smx4 record")
rec = C.load_record()


def band(sel, d, q, lo, hi):
    s = sel & (d >= lo) & (d < hi)
    return "%3d pts %5.0fk" % (s.sum(), q[s].sum() / 1e3)


for k in ITEMS:
    ev, c = k.split("/")
    p = json.load(open(os.path.join(a.prep, "smprep-%s-c%s.json" % (ev, c))))
    v, m, r = p["verdict"], p["muon"], rec[k]
    M = np.c_[m["x"], m["y"], m["z"]]
    rr = np.asarray(m["rr"], float)
    i = int(np.argmin(rr))
    stop = M[i]
    pin = r.get("pin") or {}
    print("=" * 96)
    print("%s  arm %s  muon %.1f cm, %d fit rows" % (k, p["arm"], p["muon_len_cm"], len(M)))
    # A
    print("A  record: %s / michel_kind %s / %s; pin placed %s rr %s moved %s cm" % (
        r["verdict"], r.get("michel_kind"), r.get("source", r.get("confidence")),
        pin.get("placed"), pin.get("rr", r.get("pin_rr")), pin.get("moved_cm")))
    print("   tags: %s" % (json.dumps(r.get("tags", {}), sort_keys=True)))
    print("   production: is_stm %d michel_found %d reject %s | n_split %d n_retreat %d | "
          "n_stop_gammas %d n_dots %d | dead_ahead %d n_dead_pts %d" % (
              v["is_stm"], v["michel_found"], ",".join(v["reject_names"]), v["n_split"], v["n_retreat"],
              v["n_stop_gammas"], v["n_dots"], v["dead_ahead"], v["n_dead_pts"]))
    print("   fit end (%.2f, %.2f, %.2f); PR segments of the cluster (d to the fit end, id, len):" % tuple(stop),
          sorted((round(float(np.min(np.linalg.norm(np.c_[s["x"], s["y"], s["z"]] - stop, axis=1))), 1),
                  s["id"], round(s["len_cm"], 1)) for s in p["pf"]["seg"]))
    # B
    im = p["image_near"]
    X = np.c_[im["x"], im["y"], im["z"]]
    cl = np.asarray(im["c"], int); bb = np.asarray(im["b"], int); q = np.asarray(im["q"], float)
    dstop = np.linalg.norm(X - stop, axis=1)
    for R in (5, 10, 20):
        s = dstop <= R
        cnt = {}
        for cc, b_, qq in zip(cl[s], bb[s], q[s]):
            e = cnt.setdefault((int(cc), int(b_)), [0, 0.0]); e[0] += 1; e[1] += qq
        print("B  imaged within %2d cm of the fit end (cluster, in_bundle): %s" % (
            R, "; ".join("c%d%s %d pts %.0fk" % (kk[0], "" if kk[1] else " OUT-of-bundle", vv[0], vv[1] / 1e3)
                         for kk, vv in sorted(cnt.items(), key=lambda t: (-t[1][0], t[0])))))
    # C
    F = np.vstack([M] + [np.c_[s["x"], s["y"], s["z"]] for s in p["pf"]["seg"]])
    dfit = cKDTree(F).query(X)[0]
    own = (cl == int(c)) & (dstop <= 10.0)
    w = (rr - rr[i]) <= 5.0
    ax = stop - M[w][int(np.argmax(rr[w]))]
    ax /= np.linalg.norm(ax)
    along = (X - stop) @ ax
    print("C  own cluster within 10 cm of the fit end, by distance to the nearest fit point:")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 99)):
        print("     %g-%g cm: %s" % (lo, hi, band(own, dfit, q, lo, hi)))
    past = own & (along > 0.5)
    print("     past the fit end (> 0.5 cm along the last 5 cm's direction): %d pts %.0fk, %d of them > 2 cm off the fit"
          % (past.sum(), q[past].sum() / 1e3, (past & (dfit > 2)).sum()))
    body = M[(rr > rr[i] + 30) & (rr < rr[i] + 40)]
    ctl = (cl == int(c)) & (cKDTree(body).query(X)[0] <= 5.0)
    print("   control, body rr 30-40 cm: %d pts, max off-fit %.2f cm, > 2 cm: %d" % (
        ctl.sum(), dfit[ctl].max(), (ctl & (dfit > 2)).sum()))
    off = int((own & (dfit > 2)).sum())
    print("   offline reading: %s (%d own-cluster points > 2 cm off the fit within 10 cm of the fit end)"
          % ("off-fit" if off else "on-fit", off))
    # D
    if pin.get("placed"):
        ip = int(np.argmin(np.linalg.norm(M - np.array([pin["x"], pin["y"], pin["z"]]), axis=1)))
        tail = rr <= rr[ip]
        print("D  pin at rr %.1f cm; the %d fit rows from the pin to the fit end, dQ/dx k e/cm: %s  (plateau %.0fk)" % (
            rr[ip] - rr[i], tail.sum(), " ".join("%.0f" % (x / 1e3) for x in np.asarray(m["q"])[tail]),
            v["plateau_med"] / 1e3))
    else:
        tail = (rr - rr[i]) <= 5.0
        print("D  no placed pin; the last 5 cm of fit rows, dQ/dx k e/cm: %s  (plateau %.0fk)" % (
            " ".join("%.0f" % (x / 1e3) for x in np.asarray(m["q"])[tail]), v["plateau_med"] / 1e3))
    # E
    for pl, key in (("u", "pu"), ("v", "pv"), ("w", "pw")):
        ch0, t0 = m[key][i], m["pt"][i]
        dd = p["dead"][pl]
        near = sorted({int(x) for x, lo, hi in zip(dd["ch"], dd["t0"], dd["t1"])
                       if abs(x - ch0) <= 15 and lo - 30 <= t0 <= hi + 30})
        print("E  %s: fit end at channel %.1f slice %.1f; dead channels within 15: %s" % (
            pl.upper(), ch0, t0, ("%s (nearest %.1f ch away)" % (near, min(abs(x - ch0) for x in near))) if near else "none"))

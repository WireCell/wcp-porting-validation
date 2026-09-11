#!/usr/bin/env python3
"""doc pdvd/90 -- what production holds at the stop of the owner's smx7 re-judge items.

Fork by duplication (CLAUDE.md M10) of d86_offfit.py (doc 86's four items); the item list,
the record guard and the summary changed.  On the 22 items of doc 89's part (b1), production
rejects the stop on the dQ/dx shape tests alone and builds NO Michel object; the owner (smx7)
called 13 of them STM + MICHEL, 5 STM without a Michel and 4 THRU.  The question is why the
chain sees nothing at the stop on the 13; the owner's 5 STM_ONLY and 4 THRU are the comparison.
The items are read from the smx7 labels and key, not listed here.

Per item, from the production arm's prep payload (the same sections as doc 86):
  A. the record (now the owner's smx7 call and pin) and the chain's answer;
  B. every imaged point within 5 / 10 / 20 cm of the fit end, by Bee cluster and bundle flag;
  C. the muon cluster's own points within 10 cm of the fit end, by distance to the nearest fit
     point, against a body control (rr 30-40 cm upstream); "off-fit" when any lies > 2 cm off
     the fit, else "on-fit";
  D. the fit rows from the owner's pin to the fit end (or the last 5 cm), dQ/dx k e/cm;
  E. T_bad_ch bands within 15 channels of the fit end's own U/V/W coordinate at its time slice.
Then one summary row per item and a per-group table.

Usage: STM_SCAN_RECORD=<smx1a..smx7 record> d90_offfit.py --prep PREP --labels L.json --key K.tsv
"""
import argparse, collections, csv, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--key", required=True)
a = ap.parse_args()
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the smx1a+smx3+smx4+smx5+smx6+smx7 record (doc 90)")
rec = C.load_record()
L = json.load(open(a.labels))["labels"]
K = {r["key"]: r for r in csv.DictReader([l for l in open(a.key) if not l.startswith("#")], delimiter="\t")}
GROUPS = collections.OrderedDict((g, sorted((k for k in K if K[k]["group"] == "b1" and L[k]["choice"] == g),
                                            key=lambda t: int(K[t]["scan_id"])))
                                 for g in ("STM_MICHEL", "STM_ONLY", "THRU"))
print("b1 items by the owner's smx7 call: %s" % {g: len(ks) for g, ks in GROUPS.items()})


def band(sel, d, q, lo, hi):
    s = sel & (d >= lo) & (d < hi)
    return "%3d pts %5.0fk" % (s.sum(), q[s].sum() / 1e3)


SUM = collections.OrderedDict()
for grp, items in GROUPS.items():
    for k in items:
        ev, c = k.split("/")
        p = json.load(open(os.path.join(a.prep, "smprep-%s-c%s.json" % (ev, c))))
        v, m, r = p["verdict"], p["muon"], rec[k]
        M = np.c_[m["x"], m["y"], m["z"]]
        rr = np.asarray(m["rr"], float)
        i = int(np.argmin(rr))
        stop = M[i]
        pin = r.get("pin") or {}
        print("=" * 96)
        print("%s  [owner %s]  arm %s  muon %.1f cm, %d fit rows" % (k, grp, p["arm"], p["muon_len_cm"], len(M)))
        # A
        print("A  record: %s / michel_kind %s / %s; pin placed %s rr %s moved %s cm" % (
            r["verdict"], r.get("michel_kind"), r.get("source", r.get("confidence")),
            pin.get("placed"), pin.get("rr", r.get("pin_rr")), pin.get("moved_cm")))
        print("   production: is_stm %d michel_found %d reject %s | anchor %.1f cm | n_split %d n_retreat %d | "
              "n_stop_arms %d n_stop_other %d n_dots %d | dead_ahead %d n_dead_pts %d" % (
                  v["is_stm"], v["michel_found"], ",".join(v["reject_names"]), v["bragg_anchor_shift_cm"],
                  v["n_split"], v["n_retreat"], v["n_stop_arms"], v["n_stop_other"], v["n_dots"],
                  v["dead_ahead"], v["n_dead_pts"]))
        segs = sorted((round(float(np.min(np.linalg.norm(np.c_[s["x"], s["y"], s["z"]] - stop, axis=1))), 1),
                       s["id"], round(s["len_cm"], 1)) for s in p["pf"]["seg"])
        print("   fit end (%.2f, %.2f, %.2f); PR segments of the cluster (d to the fit end, id, len): %s" % (
            tuple(stop) + (segs,)))
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
        other10 = int(((cl != int(c)) & (dstop <= 10.0)).sum())
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
        if len(body):
            ctl = (cl == int(c)) & (cKDTree(body).query(X)[0] <= 5.0)
            print("   control, body rr 30-40 cm: %d pts, max off-fit %.2f cm, > 2 cm: %d" % (
                ctl.sum(), dfit[ctl].max() if ctl.any() else float("nan"), (ctl & (dfit > 2)).sum()))
        else:
            print("   control, body rr 30-40 cm: the muon is shorter than 30 cm")
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
        ndead = 0
        for pl, key in (("u", "pu"), ("v", "pv"), ("w", "pw")):
            ch0, t0 = m[key][i], m["pt"][i]
            dd = p["dead"][pl]
            near = sorted({int(x) for x, lo, hi in zip(dd["ch"], dd["t0"], dd["t1"])
                           if abs(x - ch0) <= 15 and lo - 30 <= t0 <= hi + 30})
            ndead += bool(near)
            print("E  %s: fit end at channel %.1f slice %.1f; dead channels within 15: %s" % (
                pl.upper(), ch0, t0, ("%s (nearest %.1f ch away)" % (near, min(abs(x - ch0) for x in near))) if near else "none"))
        SUM[k] = dict(grp=grp, off=off, past=int(past.sum()), past_off=int((past & (dfit > 2)).sum()), other10=other10,
                      pin=(pin.get("rr") if pin.get("placed") else None), anchor=v["bragg_anchor_shift_cm"],
                      arms=v["n_stop_arms"], dead=ndead, bits="+".join(v["reject_names"]))

print("\n" + "=" * 96)
print("SUMMARY: one row per item (own = the muon's cluster; off = own points > 2 cm off the fit within 10 cm of the end)")
print("  %-14s %-10s %-22s %4s %5s %8s %8s %6s %5s %5s" % ("item", "owner", "production bits", "off", "past", "past>2cm",
                                                             "other10", "anchor", "pin", "dead"))
for k, s in SUM.items():
    print("  %-14s %-10s %-22s %4d %5d %8d %8d %6.1f %5s %5d" % (k, s["grp"], s["bits"], s["off"], s["past"], s["past_off"],
                                                                 s["other10"], s["anchor"], "-" if s["pin"] is None else "%.1f" % s["pin"], s["dead"]))
print("\nper group: items | off-fit | own points past the end (median) | other-cluster points within 10 cm (median) | owner pin placed | a dead plane at the end")
for grp in GROUPS:
    ss = [s for s in SUM.values() if s["grp"] == grp]
    if not ss: continue
    print("  %-10s %2d | %2d | %5.0f | %5.0f | %2d | %2d" % (
        grp, len(ss), sum(1 for s in ss if s["off"]), np.median([s["past"] for s in ss]), np.median([s["other10"] for s in ss]),
        sum(1 for s in ss if s["pin"] is not None), sum(1 for s in ss if s["dead"])))

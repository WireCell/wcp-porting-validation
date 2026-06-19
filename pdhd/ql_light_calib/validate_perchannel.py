#!/usr/bin/env python3
"""Validate the per-channel measured_pe_scale retune: closure + GT preservation.

OLD = pre-retune dumps (/home/xqian/tmp/ql_perch_before, the current production
model); NEW = work/029107_{0..3}/calib-evt*.json after reprocessing with the
per-channel scales.  Labels (work/ql_labels) are still the pre-reprocess hand scan;
everything is keyed by (apa, round(flash_time,2), cluster_ident) so it survives the
flash-gid renumbering that any measured-PE change causes.

Primary check is GT preservation (out-of-sample: scales fit on clean anchors, the
winner-diff runs over ALL bundles).  Closure is secondary."""
import json
import os
import statistics as st

PD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd"
OLD = "/home/xqian/tmp/ql_perch_before"
EVTS = {983: 0, 991: 1, 999: 2, 1007: 3}
FBK = {4, 14, 24, 34, 40, 42, 45, 46, 47, 49, 50, 52, 55, 56, 57, 59, 60, 62, 65,
       66, 67, 69, 70, 72, 75, 76, 77, 79, 84, 85, 86, 87, 94, 95, 96, 97, 104,
       105, 106, 107, 114, 115, 116, 117, 120, 121, 124, 125, 127, 129, 130, 131,
       134, 135, 137, 139, 140, 141, 144, 145, 147, 149, 150, 151, 154, 155, 157, 159}
STATIC = {3, 86, 87, 97, 107, 116, 117}
OUTLIERS = [20, 23, 25, 30, 35, 48, 88, 98, 108, 118, 139, 149]


def clean(m):
    f = m["flags"]
    return (not f.get("close_to_PMT")) and (not f.get("window_truncated")) \
        and m["metrics"].get("ks_dis", 9) < 0.2


def med(x):
    return st.median(x) if x else float("nan")


def bundle_index(d):
    """(apa, round(time,2), cluster_raw) -> (bundle, flash)."""
    fb = {f["gid"]: f for f in d["flashes"]}
    idx = {}
    for b in d["bundles"]:
        if b["flash_gid"] in fb:
            f = fb[b["flash_gid"]]
            idx[(b["apa"], round(f["time"], 2), b["main_cluster"] % 1000000)] = (b, f)
    return idx


def winners(d):
    fb = {f["gid"]: f for f in d["flashes"]}
    s = set()
    for b in d["bundles"]:
        if b.get("auto_selected") and b["flash_gid"] in fb:
            s.add((b["apa"], round(fb[b["flash_gid"]]["time"], 2), b["main_cluster"] % 1000000))
    return s


def acc(entries):
    s = set()
    for m in entries:
        for ci in m["cluster_idents"]:
            s.add((m["apa"], round(m["flash_time_us"], 2), ci))
    return s


def grp_ratio(pairs):
    sm = sum(o for o, p in pairs); sp = sum(p for o, p in pairs)
    return sp / sm if sm > 0 else float("nan")


def main():
    px_o, px_n = [], []                                  # +x integral pred/meas
    a0f_o, a0f_n, a0h_o, a0h_n = [], [], [], []          # APA0 FBK / HPK pred/meas
    out_o, out_n = [], []                                # outlier channels pred/meas
    tot_g = tot_l = ra_o = ra_n = rj_o = rj_n = 0
    lost = []; newfp = []
    print("per-event winner-diff & GT recall (TIME-keyed):")
    for evt, ev in EVTS.items():
        new = json.load(open("%s/work/029107_%d/calib-evt%d.json" % (PD, ev, evt)))
        old = json.load(open("%s/calib-evt%d.json" % (OLD, evt)))
        lab = json.load(open("%s/work/ql_labels/labels-evt%d.json" % (PD, evt)))
        accept = acc(lab["matches"]); reject = acc(lab.get("rejected_auto", []))
        wo, wn = winners(old), winners(new)
        tot_g += len(wn - wo); tot_l += len(wo - wn)
        ro, rn = len(accept & wo), len(accept & wn)
        fo, fn = len(reject & wo), len(reject & wn)
        ra_o += ro; ra_n += rn; rj_o += fo; rj_n += fn
        lost += [(evt,) + x for x in (accept & wo) - (accept & wn)]
        newfp += [(evt,) + x for x in (reject & wn) - (reject & wo)]
        print("  evt%d: winners %d->%d (+%d/-%d) | GT-accept %d->%d/%d | GT-reject %d->%d/%d"
              % (evt, len(wo), len(wn), len(wn - wo), len(wo - wn), ro, rn, len(accept),
                 fo, fn, len(reject)))
        iN, iO = bundle_index(new), bundle_index(old)
        for m in lab["matches"]:
            if not clean(m):
                continue
            apa, tk = m["apa"], round(m["flash_time_us"], 2)
            N = next((iN.get((apa, tk, c)) for c in m["cluster_idents"] if iN.get((apa, tk, c))), None)
            O = next((iO.get((apa, tk, c)) for c in m["cluster_idents"] if iO.get((apa, tk, c))), None)
            if not N or not O:
                continue
            (bN, fN), (bO, fO) = N, O
            if apa == 1:
                ch = [c for c in range(80) if c not in STATIC]
                sp, sm = sum(bN["pred_pe"][c] for c in ch), sum(fN["pe"][c] for c in ch)
                if sm > 50 and sp > 5:
                    px_n.append(sp / sm)
                sp, sm = sum(bO["pred_pe"][c] for c in ch), sum(fO["pe"][c] for c in ch)
                if sm > 50 and sp > 5:
                    px_o.append(sp / sm)
            else:
                for typ, ro_, rn_ in (("F", a0f_o, a0f_n), ("H", a0h_o, a0h_n)):
                    ch = [c for c in range(120, 160) if c not in STATIC
                          and ((c in FBK) == (typ == "F"))]
                    sp, sm = sum(bN["pred_pe"][c] for c in ch), sum(fN["pe"][c] for c in ch)
                    if sm > 30 and sp > 3:
                        rn_.append(sp / sm)
                    sp, sm = sum(bO["pred_pe"][c] for c in ch), sum(fO["pe"][c] for c in ch)
                    if sm > 30 and sp > 3:
                        ro_.append(sp / sm)
            # outlier channels (whichever side they're on)
            for c in OUTLIERS:
                if c in STATIC:
                    continue
                side = 1 if c < 80 else 0
                if apa != side:
                    continue
                if fN["pe"][c] > 5 and bN["pred_pe"][c] > 3:
                    out_n.append(bN["pred_pe"][c] / fN["pe"][c])
                if fO["pe"][c] > 5 and bO["pred_pe"][c] > 3:
                    out_o.append(bO["pred_pe"][c] / fO["pe"][c])

    print("\nCLOSURE  pred/meas, light-weighted over clean anchors (target ~1.0):")
    print("  +x integral (ch0-79):   old=%.3f -> new=%.3f   n=%d   (should STAY ~old: vuv_eff held)"
          % (med(px_o), med(px_n), len(px_n)))
    print("  APA0 FBK (120-159):     old=%.3f -> new=%.3f   n=%d" % (med(a0f_o), med(a0f_n), len(a0f_n)))
    print("  APA0 HPK (120-159):     old=%.3f -> new=%.3f   n=%d" % (med(a0h_o), med(a0h_n), len(a0h_n)))
    print("  outlier channels:       old=%.3f -> new=%.3f   n=%d" % (med(out_o), med(out_n), len(out_n)))
    print("\nWINNER-DIFF total: +%d / -%d" % (tot_g, tot_l))
    print("GT vs hand-scan: accept recovered %d->%d, reject re-selected %d->%d" % (ra_o, ra_n, rj_o, rj_n))
    print("  GT matches LOST by retune (%d): %s" % (len(lost), lost))
    print("  new human-rejected FPs (%d): %s" % (len(newfp), newfp))


if __name__ == "__main__":
    main()

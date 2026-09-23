#!/usr/bin/env python3
"""doc qlmatch/32 sec 3 -- what differs, at the bundle level, for every (cluster, physical flash) pairing that one arm
auto-selects and the other does not (the movers), between the control arm (twoside light) and the candidate arm.

The calib dumps carry the post-fit bundle fields, not a per-stage trace, so the class is an attribution by the
field that changed, in this order (first that applies):
  no-bundle     -- the other arm has no bundle for this (cluster, flash): flash admission / prefilter / overpred
  lasso-zero    -- strength < strength_cutoff in the losing arm (LASSO did not use the pairing)
  ks            -- losing-arm ks above the loosest selecting rung (hc ks 0.10) while the winning arm's is below
  ratio         -- losing-arm pred/meas (total) outside [0.5, 2.0] (rescue / postcull ratio gates) while the winning
                   arm's is inside
  c2n           -- losing-arm chi2/ndf above 30 (hc miss rung) while the winning arm's is below
  other         -- none of the above (competition, post-cull ordering, cross-TPC logic)
Output: counts by class and direction, and the ks / pred-meas / c2n / strength shifts on the SAME physical flash.

    cd pdvd/docs/qlmatch/scripts && python3 d32_loss_stage.py --work-root /home/xqian/tmp/p31/wroot \
        --ctl q31ctl --cand q32ti --time-map ../d32/time_map_ctl_to_q32ti.json > ../d32/loss_stage.txt
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d32_scan_items as I       # noqa: E402

TOL = 0.5


def bundles_by(calib):
    fl = {f["gid"]: f for f in calib["flashes"]}
    out = {}
    for b in calib["bundles"]:
        f = fl[b["flash_gid"]]
        out.setdefault(b["main_cluster"], []).append((f["time"], b, f))
    return out


def near(lst, t):
    best = None
    for tt, b, f in lst:
        if abs(tt - t) <= TOL and (best is None or abs(tt - t) < abs(best[0] - t)):
            best = (tt, b, f)
    return best


def feats(b, f):
    return dict(ks=b["ks_dis"], c2n=b["chi2"] / b["ndf"] if b["ndf"] else float("nan"), st=b["strength"],
                r=b["total_pred_light"] / max(f["total_PE"], 1e-3), sat=I.railed_cath(f), cons=b["consistent"])


def classify(win, lose, cut):
    if lose is None:
        return "no-bundle"
    if lose["st"] < cut:
        return "lasso-zero"
    if lose["ks"] > 0.10 and win["ks"] <= 0.10:
        return "ks"
    if not (0.5 <= lose["r"] <= 2.0) and (0.5 <= win["r"] <= 2.0):
        return "ratio"
    if lose["c2n"] > 30 and win["c2n"] <= 30:
        return "c2n"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--ctl", default="q31ctl")
    ap.add_argument("--cand", default="q32ti")
    ap.add_argument("--time-map", required=True)
    a = ap.parse_args()
    tmap = I.load(a.time_map)["events"]
    rows = []
    for idx in range(I.NEVT):
        evt = I.EVT0 + I.STEP * idx
        C = I.load(I.calib_path(a.work_root, idx, a.ctl))
        T = I.load(I.calib_path(a.work_root, idx, a.cand))
        cut = C["quality_params"].get("strength_cutoff", 0.05)
        pairs = tmap.get(str(evt), [])
        fwd = lambda t: next((t1 for t0, t1 in pairs if abs(t - t0) <= 1e-3), t)
        bwd = lambda t: next((t0 for t0, t1 in pairs if abs(t - t1) <= 1e-3), t)
        BC, BT = bundles_by(C), bundles_by(T)
        longs = I.long_uids(C) & I.long_uids(T)
        for uid in sorted(longs):
            ac = [(t, b, f) for t, b, f in BC.get(uid, []) if b["auto_selected"]]
            at = [(t, b, f) for t, b, f in BT.get(uid, []) if b["auto_selected"]]
            for t, b, f in ac:                      # auto in ctl; is the same physical flash auto in cand?
                m = near(BT.get(uid, []), fwd(t))
                if m and m[1]["auto_selected"]:
                    continue
                lose = feats(m[1], m[2]) if m else None
                rows.append(dict(evt=evt, uid=uid, dir="lost-in-cand", win=feats(b, f), lose=lose,
                                 cls=classify(feats(b, f), lose, cut)))
            for t, b, f in at:                      # auto in cand; is the same physical flash auto in ctl?
                m = near(BC.get(uid, []), bwd(t))
                if m and m[1]["auto_selected"]:
                    continue
                lose = feats(m[1], m[2]) if m else None
                rows.append(dict(evt=evt, uid=uid, dir="gained-in-cand", win=feats(b, f), lose=lose,
                                 cls=classify(feats(b, f), lose, cut)))
    print(f"# {a.ctl} -> {a.cand}: auto pairings that change (long clusters, 18 run-039252 events); class = first "
          f"bundle field that separates the losing arm from the winning arm (see script docstring)")
    for d in ("lost-in-cand", "gained-in-cand"):
        rr = [r for r in rows if r["dir"] == d]
        c = Counter(r["cls"] for r in rr)
        print(f"{d}: {len(rr)} pairings; railed flash (winning arm) {sum(r['win']['sat'] for r in rr)}; "
              + ", ".join(f"{k} {v}" for k, v in c.most_common()))
        both = [r for r in rr if r["lose"] is not None]
        if both:
            q = lambda k: np.median([r["lose"][k] - r["win"][k] for r in both])
            lr = np.median([math.log10(max(r["lose"]["r"], 1e-4) / max(r["win"]["r"], 1e-4)) for r in both])
            print(f"   same-flash shifts losing - winning (median, n {len(both)}): ks {q('ks'):+.3f}  c2n "
                  f"{q('c2n'):+.1f}  strength {q('st'):+.3f}  log10(pred/meas) {lr:+.3f}")
    print("\n# per pairing: evt uid dir class | win ks c2n st pred/meas sat | lose ks c2n st pred/meas")
    for r in rows:
        w, l = r["win"], r["lose"]
        ls = "no bundle" if l is None else f"{l['ks']:.3f} {l['c2n']:.1f} {l['st']:.3f} {l['r']:.2f}"
        print(f"{r['evt']} {r['uid']} {r['dir']} {r['cls']} | {w['ks']:.3f} {w['c2n']:.1f} {w['st']:.3f} "
              f"{w['r']:.2f} {int(w['sat'])} | {ls}")


if __name__ == "__main__":
    main()

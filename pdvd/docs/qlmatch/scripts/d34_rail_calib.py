#!/usr/bin/env python3
"""doc qlmatch/34 step A -- measure the railed channels against prediction on geometric cathode-crosser anchors.

Doc 12's per-type factors (fit_qtol_crossers.py) exclude railed channels from BOTH sums, and unrailed PE is identical
between production (twoside) and ToT light, so a plain refit on ToT light returns the same numbers.  What ToT newly
allows is to put the railed channels themselves against the prediction.  Per anchor (the fit_qtol_crossers.py
harvest: strict cathode crossers, external geometric truth, no dependence on the LASSO/auto selection):

    s      = Sum(meas) / Sum(pred) over the anchor's kept UNRAILED channels (all groups) -- the same normaliser the
             toolkit knob ks_sat_tol uses (TimingTPCBundle), so f below means the same thing there;
    r_j    = (meas_j / pred_j) / s     for every railed channel j with pred_j > 0.5 PE.

Pre-registered use (d34/prereg.md): f = exp(q84 |ln r|) - 1 on the ToT arm, if >= 30 railed samples (else f = 1.0);
QtoL trigger = rail-inclusive global median Sum(meas)/Sum(pred) vs the rail-excluded one on the same anchors, > 10 %.

    python3 d34_rail_calib.py --work-root /home/xqian/tmp/p31/wroot --arms q32ti,q31ctl > ../d34/rail_calib.txt
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "ql_light_calib"))
from d32_scan_items import calib_path          # noqa: E402
from fit_qtol_crossers import harvest, CATH    # noqa: E402

MIN_RAIL = 30
PRIOR_F = 1.0


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def arm_report(tag, anchors):
    rows, glob_ex, glob_in = [], [], []
    for a in anchors:
        k = a["keep"]
        sm, sp = a["meas"][k].sum(), a["pred"][k].sum()
        if sp <= 0 or sm <= 0:
            continue
        s = sm / sp
        glob_ex.append(s)
        rail = a["rail"]
        ki = k | rail
        si, spi = a["meas"][ki].sum(), a["pred"][ki].sum()
        if spi > 0:
            glob_in.append(si / spi)
        for j in np.flatnonzero(rail):
            if a["pred"][j] > 0.5 and a["meas"][j] > 0:
                rows.append((j, a["pred"][j], a["meas"][j] / a["pred"][j] / s, a["evt"], a["gid"]))
    print(f"\n## arm {tag}: anchors {len(anchors)} (with unrailed light {len(glob_ex)}), railed samples {len(rows)}")
    if not rows:
        return None, None
    lr = np.log(np.array([r[2] for r in rows]))
    print(f"   all railed: median r {np.exp(np.median(lr)):.3f}  [16 % {np.exp(q(lr, 16)):.3f}, "
          f"84 % {np.exp(q(lr, 84)):.3f}]  q84|ln r| {q(np.abs(lr), 84):.3f} -> f = {np.exp(q(np.abs(lr), 84)) - 1:.3f}")
    print("   per channel (railed samples):")
    for ch in sorted({r[0] for r in rows}):
        x = np.log(np.array([r[2] for r in rows if r[0] == ch]))
        print(f"     ch{ch:>2}{' cath' if ch in CATH else '     '}: n {len(x):>3}  median r {np.exp(np.median(x)):.3f}"
              f"  [{np.exp(q(x, 16)):.3f}, {np.exp(q(x, 84)):.3f}]")
    pr = np.array([r[1] for r in rows])
    print("   r vs predicted PE (bright -> dim quintiles would show a visibility-shape trend):")
    edges = np.percentile(pr, [0, 20, 40, 60, 80, 100])
    for a0, a1 in zip(edges[:-1], edges[1:]):
        sel = (pr >= a0) & (pr <= a1)
        x = lr[sel]
        print(f"     pred [{a0:>8.1f}, {a1:>8.1f}]: n {len(x):>3}  median r {np.exp(np.median(x)):.3f}")
    ge, gi = np.median(glob_ex), np.median(glob_in)
    print(f"   global median Sum(meas)/Sum(pred): rail-excluded {ge:.3f}  rail-inclusive {gi:.3f}  "
          f"ratio {gi / ge:.3f}")
    return rows, gi / ge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--arms", default="q32ti,q31ctl")
    a = ap.parse_args()
    res = {}
    for tag in a.arms.split(","):
        paths = [calib_path(a.work_root, idx, tag) for idx in range(18)]
        paths = [p for p in paths if os.path.exists(p)]
        anchors = harvest(paths)
        cache = {}
        for an in anchors:              # railed = the dump's sat flag on this flash, statically kept channel
            if an["path"] not in cache:
                cache[an["path"]] = json.load(open(an["path"]))
            d = cache[an["path"]]
            f = next(f for f in d["flashes"] if f["gid"] == an["gid"])
            keep_static = np.array([od["active"] and not od["auto_masked"] for od in d["opdets"]], bool)
            an["rail"] = np.asarray(f.get("sat", [0] * 40), bool) & keep_static
        print(f"# arm {tag}: {len(paths)} dumps")
        res[tag] = arm_report(tag, anchors)
    t = a.arms.split(",")[0]
    rows, gratio = res[t]
    print(f"\n# PRE-REGISTERED READ-OUT (arm {t})")
    if rows is None or len(rows) < MIN_RAIL:
        print(f"   railed samples {0 if rows is None else len(rows)} < {MIN_RAIL}: f = prior {PRIOR_F}")
    else:
        lr = np.log(np.array([r[2] for r in rows]))
        print(f"   railed samples {len(rows)} >= {MIN_RAIL}: f = exp(q84|ln r|) - 1 = "
              f"{np.exp(q(np.abs(lr), 84)) - 1:.3f}")
    if gratio is not None:
        print(f"   QtoL trigger: rail-inclusive / rail-excluded = {gratio:.3f} -> "
              f"{'QtoL arm (x %.3f)' % gratio if abs(gratio - 1) > 0.10 else 'no QtoL arm'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc qlmatch/33 sec 2 -- what the LASSO sees for a few movers, per arm (from the calib dumps; no replica).

For each chosen mover (a cluster whose auto flash differs between --a and --b), print for every bundle that was
auto-selected in any listed arm, per arm: flash time (own frame), strength, ks, chi2/ndf, pred/meas (all channels),
the round-2 LASSO weight base |pred_tot - meas_tot|/meas_tot (meas_tot = ALL 40 channels, railed included, as the C++
uses), the number of railed active rows on the flash, and the share of the flash's active-channel PE that sits on
railed channels.  Those last two are what fit_round2_shared lets into the solve when saturation_mask_fit=false.

    python3 d33_lasso_trace.py --work-root /home/xqian/tmp/p31/wroot --a q31ctl --b q32ti \
        --arms q31ctl:C,q32ti:T,q33cs:C,q33ts:T --time-map ../d32/time_map_ctl_to_q32ti.json --n 5 --seed 33
"""
import argparse
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d32_scan_items import calib_path, load, auto_times, long_uids, EVT0, STEP, NEVT   # noqa: E402
import d33_scan_items as I                                                            # noqa: E402

KNEE, FLOOR = 0.3, 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--a", default="q31ctl")
    ap.add_argument("--b", default="q32ti")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--time-map", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=33)
    a = ap.parse_args()
    arms = [tuple(x.split(":")) for x in a.arms.split(",")]
    light = dict(arms)
    tmap = load(a.time_map)["events"]
    rng = random.Random(a.seed)
    lost, gained = [], []                  # lost = auto in a not in b (on the mapped flash); gained = the reverse
    for idx in range(NEVT):
        evt = EVT0 + STEP * idx
        fwd, _ = I.mapper(tmap.get(str(evt), []))
        A, B = load(calib_path(a.work_root, idx, a.a)), load(calib_path(a.work_root, idx, a.b))
        fa = (lambda t: fwd(t)) if light[a.b] == "T" and light[a.a] == "C" else (lambda t: t)
        aa, ab = auto_times(A), auto_times(B)
        for uid in sorted(long_uids(A) & long_uids(B)):
            sa = {round(fa(t), 3) for t in aa.get(uid, [])}
            sb = {round(t, 3) for t in ab.get(uid, [])}
            if any(all(abs(x - y) > 0.5 for y in sb) for x in sa):
                lost.append((idx, uid))
            if any(all(abs(x - y) > 0.5 for x in sa) for y in sb):
                gained.append((idx, uid))
    print(f"# {a.a} -> {a.b}: clusters losing an auto flash {len(lost)}, gaining one {len(gained)}; "
          f"tracing {a.n} of each (seed {a.seed})")
    for lab, pool in (("LOST in " + a.b, lost), ("GAINED in " + a.b, gained)):
        for idx, uid in sorted(rng.sample(pool, min(a.n, len(pool)))):
            evt = EVT0 + STEP * idx
            fwd, _ = I.mapper(tmap.get(str(evt), []))
            print(f"\n## {lab}: evt {evt} uid {uid}")
            dumps = {tag: load(calib_path(a.work_root, idx, tag)) for tag, _ in arms}
            # control-frame times of every flash auto-selected for this cluster in any arm
            keys = set()
            for tag, lt in arms:
                for t in auto_times(dumps[tag]).get(uid, []):
                    keys.update(I.inv_all(t, tmap.get(str(evt), [])) if lt == "T" else [t])
            print(f"{'flash(ctl t)':>13} {'arm':>7} {'auto':>4} {'strength':>8} {'ks':>6} {'c2n':>7} {'pred/meas':>9} "
                  f"{'w_base':>6} {'railed':>6} {'railPEshare':>11}")
            for k in sorted(keys):
                for tag, lt in arms:
                    D = dumps[tag]
                    t1 = fwd(k) if lt == "T" else k
                    fl = {f["gid"]: f for f in D["flashes"]}
                    act = np.array([o["active"] and not o.get("auto_masked", False) for o in D["opdets"]])
                    bs = [b for b in D["bundles"] if b["main_cluster"] == uid and abs(fl[b["flash_gid"]]["time"] - t1) <= 0.5]
                    if not bs:
                        print(f"{k:>13.2f} {tag:>7}   -- no bundle")
                        continue
                    b = bs[0]
                    f = fl[b["flash_gid"]]
                    pe = np.asarray(f["pe"], float)
                    sat = np.asarray(f.get("sat", [0] * len(pe)), float) > 0
                    meas = max(f["total_PE"], 1e-9)
                    d = abs(b["total_pred_light"] - meas)
                    w = d / meas if d > KNEE * meas else FLOOR
                    share = float(pe[act & sat].sum() / max(pe[act].sum(), 1e-9))
                    c2n = b["chi2"] / b["ndf"] if b["ndf"] else float("nan")
                    print(f"{k:>13.2f} {tag:>7} {'Y' if b.get('auto_selected') else '.':>4} {b['strength']:>8.3f} "
                          f"{b['ks_dis']:>6.3f} {c2n:>7.1f} {b['total_pred_light'] / meas:>9.2f} {w:>6.2f} "
                          f"{int((act & sat).sum()):>6} {share:>11.2f}")


if __name__ == "__main__":
    main()

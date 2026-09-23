#!/usr/bin/env python3
"""doc qlmatch/33 sec 4 -- light-neutrality proof of the round-2 sheet, BEFORE rendering.

Every round-2 sheet is drawn from the CONTROL dump.  That is equivalent to drawing it from the ToT dump only if
everything the sheet shows (except the hidden railed values) is the same in both.  For every bundle of every mover
cluster in the control dump, find the ToT-arm bundle of the same cluster on the time-mapped flash and compare:
  pred_pe (this cluster's prediction), other_clusters (orange points), unrailed measured pe, the sat (railed) flags.
Also checks that other_clusters is the same set on every flash of a cluster (i.e. it is the cluster's own group, not
a per-flash matcher hypothesis).

    python3 d33_neutrality.py --work-root /home/xqian/tmp/p31/wroot --ctl q31ctl --cand q32ti \
        --time-map ../d32/time_map_ctl_to_q32ti.json --arms q32ti:T,... > ../d33/neutrality.txt
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d32_scan_items import calib_path, load, EVT0, STEP, NEVT   # noqa: E402
import d33_scan_items as I                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--ctl", default="q31ctl")
    ap.add_argument("--cand", default="q32ti")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--time-map", required=True)
    a = ap.parse_args()
    _, movers, _, _ = I.build(a)
    by_evt = defaultdict(set)
    for m in movers:
        by_evt[m["idx"]].add(m["uid"])
    tmap = load(a.time_map)["events"]
    c = Counter()
    worst = dict(pred=0.0, unr=0.0)
    ex = []
    for idx in range(NEVT):
        evt = EVT0 + STEP * idx
        C, T = load(calib_path(a.work_root, idx, a.ctl)), load(calib_path(a.work_root, idx, a.cand))
        fwd, _ = I.mapper(tmap.get(str(evt), []))
        fc = {f["gid"]: f for f in C["flashes"]}
        ft = {f["gid"]: f for f in T["flashes"]}
        tb = defaultdict(list)
        for b in T["bundles"]:
            tb[b["main_cluster"]].append(b)
        groups = defaultdict(set)
        for b in C["bundles"]:
            if b["main_cluster"] not in by_evt[idx]:
                continue
            groups[b["main_cluster"]].add(tuple(sorted(b["other_clusters"])))
            c["bundles"] += 1
            t1 = fwd(fc[b["flash_gid"]]["time"])
            m = [x for x in tb[b["main_cluster"]] if abs(ft[x["flash_gid"]]["time"] - t1) <= 0.5]
            if not m:
                c["no ToT bundle on the mapped flash"] += 1
                continue
            x = m[0]
            fa, fb = fc[b["flash_gid"]], ft[x["flash_gid"]]
            pa, pb = np.asarray(b["pred_pe"], float), np.asarray(x["pred_pe"], float)
            dp = float(np.max(np.abs(pa - pb)) / max(pa.max(), 1e-9))
            worst["pred"] = max(worst["pred"], dp)
            c["pred_pe identical"] += int(dp == 0.0)
            c["pred_pe differs"] += int(dp > 0.0)
            c["other_clusters identical"] += int(sorted(b["other_clusters"]) == sorted(x["other_clusters"]))
            c["other_clusters differ"] += int(sorted(b["other_clusters"]) != sorted(x["other_clusters"]))
            sa, sb = np.asarray(fa.get("sat", []), float) > 0, np.asarray(fb.get("sat", []), float) > 0
            c["sat identical"] += int(np.array_equal(sa, sb))
            c["sat differs"] += int(not np.array_equal(sa, sb))
            u = ~(sa | sb)
            ea, eb = np.asarray(fa["pe"], float)[u], np.asarray(fb["pe"], float)[u]
            du = float(np.max(np.abs(ea - eb) / np.maximum(np.maximum(ea, eb), 1.0))) if u.any() else 0.0
            worst["unr"] = max(worst["unr"], du)
            c["unrailed pe within 1 %"] += int(du <= 0.01)
            c["unrailed pe differs > 1 %"] += int(du > 0.01)
            if (dp > 1e-6 or du > 0.01) and len(ex) < 10:
                ex.append((evt, b["main_cluster"], round(fa["time"], 2), round(dp, 4), round(du, 4)))
        c["clusters"] += len(groups)
        c["clusters whose other_clusters vary by flash"] += sum(1 for g in groups.values() if len(g) > 1)
    print(f"# d33 neutrality: {a.ctl} vs {a.cand} on the bundles of {len(movers)} movers "
          f"(union over {a.arms})")
    for k, v in c.items():
        print(f"{k}: {v}")
    print(f"worst pred_pe rel diff {worst['pred']:.3g}; worst unrailed pe rel diff {worst['unr']:.3g}")
    for e in ex:
        print("  example evt uid t dpred dunrailed:", e)


if __name__ == "__main__":
    main()

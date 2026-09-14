#!/usr/bin/env python3
"""doc 108 sec 4.1: multiple beam-window flashes in the SBND production sample.

Usage:
  python3 d108_flash_census.py [--run /nfs/data/1/xqian/sbnd_data/run] [--cands products/d107/candidates.tsv]
      [--n 3000] [--seed 21] [--lo 0.2] [--hi 2.2]

A. Bee op.json of --n random events: flashes with op_t in [lo, hi) us per event,
   the two-flash events split by TPC ('apa') and time gap, the time-gap
   distribution of different-TPC pairs.
B. Events with two T_tagger rows: are the rows' flashes (T_cluster flash_id ==
   matched_flash_gid; TPC = gid // 1000000) one physical flash seen by both TPCs
   (different TPC, |dt| < 0.1 us) or distinct flashes; vertex / nearest-truth
   status of the rows (doc 107 candidates.tsv).
C. True neutrino time (Bee mc.json T, us) vs the matched flash time for rows whose
   reco vertex is < 5 cm from that true vertex: the offset a truth<->flash time
   match needs.
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import random
import re
import statistics as st
import zipfile

import numpy as np
import uproot


def part_a(run, n, seed, lo, hi):
    bee = sorted(glob.glob(os.path.join(run, "bee", "bee_*.zip")))
    random.seed(seed)
    random.shuffle(bee)
    nwin, nwin_m, cat, dts = collections.Counter(), collections.Counter(), collections.Counter(), []
    for p in bee[:n]:
        z = zipfile.ZipFile(p)
        m = [x for x in z.namelist() if x.endswith("-op.json")]
        if not m:
            nwin["no op.json"] += 1
            continue
        j = json.loads(z.read(m[0]))
        t, apa, cl = j["op_t"], j["apa"], j["op_cluster_ids"]
        w = [i for i, x in enumerate(t) if lo <= x < hi]
        nwin[min(len(w), 3)] += 1
        nwin_m[min(sum(1 for i in w if cl[i]), 3)] += 1
        if len(w) == 2:
            a, b = w
            dt = abs(t[a] - t[b])
            if apa[a] != apa[b]:
                dts.append(dt)
            k = ("diff TPC" if apa[a] != apa[b] else "same TPC") + (", dt<0.1us" if dt < 0.1 else ", dt>=0.1us")
            k += ", clusters on " + ("both" if cl[a] and cl[b] else "one" if (cl[a] or cl[b]) else "neither")
            cat[k] += 1
    print(f"== A: {n} events, beam window [{lo}, {hi}) us")
    print("  in-window flashes per event (3 = 3+):", dict(sorted(nwin.items(), key=str)))
    print("  in-window flashes with matched clusters per event:", dict(sorted(nwin_m.items(), key=str)))
    for k, v in sorted(cat.items()):
        print(f"  two-flash events: {k:45s} {v}")
    edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 2.0]
    h = np.histogram(dts, bins=edges)[0]
    print("  |dt| of different-TPC in-window pairs (us):",
          ", ".join(f"[{edges[i]},{edges[i+1]}): {h[i]}" for i in range(len(h))))


def row_flash(run, key):
    F = uproot.open(os.path.join(run, "tracking-pr", "tracking-pr_r%s_s%s_e%s.root" % key))
    Tc = F["T_cluster"].arrays(["flash_id", "flash_time_us", "flash_pe"], library="np")
    return {int(g): (float(t), float(p)) for g, t, p in zip(Tc["flash_id"], Tc["flash_time_us"], Tc["flash_pe"])}


def part_b(run, cands):
    C = list(csv.DictReader(open(cands), delimiter="\t"))
    ev = collections.defaultdict(list)
    for r in C:
        ev[(r["run"], r["subrun"], r["event"])].append(r)
    two = {k: v for k, v in ev.items() if len(v) == 2}
    s = collections.Counter()
    for k, v in sorted(two.items()):
        ft = row_flash(run, k)
        g = [int(r["gid"]) for r in v]
        t = [ft[x][0] for x in g]
        pair = (g[0] // 1000000 != g[1] // 1000000) and abs(t[0] - t[1]) < 0.1
        key = "one flash, both TPCs (dt<0.1us)" if pair else "distinct flashes"
        vd = [r["vertex_default"] == "1" for r in v]
        s[(key, "events")] += 1
        s[(key, "a row has no vertex")] += int(any(vd))
        if not any(vd):
            dv = math.dist(*[[float(r[c]) for c in ("nu_x", "nu_y", "nu_z")] for r in v])
            near = all(float(r["t_dist_cm"]) < 5 for r in v)
            s[(key, "both vertices, <30 cm apart")] += int(dv < 30)
            s[(key, "both vertices, same nearest truth")] += int(v[0]["t_idx"] == v[1]["t_idx"])
            s[(key, "both <5 cm of DIFFERENT true vertices")] += int(near and v[0]["t_idx"] != v[1]["t_idx"])
    print("== B: events with two T_tagger rows:", len(two))
    for (a, b), n in sorted(s.items()):
        print(f"  {a:34s} {b:42s} {n}")


def part_c(run, cands):
    C = list(csv.DictReader(open(cands), delimiter="\t"))
    rows = [r for r in C if r["vertex_default"] == "0" and 0 <= float(r["t_dist_cm"]) < 5]
    cache, d = {}, []
    for r in rows:
        k = (r["run"], r["subrun"], r["event"])
        if k not in cache:
            cache[k] = row_flash(run, k)
        f = cache[k].get(int(r["gid"]))
        if f:
            d.append(f[0] - float(r["t_T"]))
    q = [round(x, 3) for x in st.quantiles(d, n=20)]
    print("== C: flash time - true nu time (us), rows with reco vertex < 5 cm of it:", len(d))
    print(f"  median {st.median(d):.3f}  5%..95% quantiles {q[0]} .. {q[-1]}  "
          f"|delta - median| < 0.1 us: {sum(abs(x - st.median(d)) < 0.1 for x in d) / len(d):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/nfs/data/1/xqian/sbnd_data/run")
    ap.add_argument("--cands", default="products/d107/candidates.tsv")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=2.2)
    a = ap.parse_args()
    part_a(a.run, a.n, a.seed, a.lo, a.hi)
    part_b(a.run, a.cands)
    part_c(a.run, a.cands)


if __name__ == "__main__":
    main()

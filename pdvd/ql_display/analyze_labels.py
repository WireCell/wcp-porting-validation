#!/usr/bin/env python
"""Empirical study of the human Q/L hand-scan decision rule.

Reads the saved hand-scan labels from PDHD and SBND
(work/ql_labels/.../labels-evt*.json) and contrasts the metric/flag
distributions of

  kept          -- entries in "matches" with flags.auto_selected == True
                   (matcher picks the human confirmed)
  human-added   -- entries in "matches" with auto_selected == False
                   (matches the matcher missed, human added)
  rejected_auto -- entries in "rejected_auto" (matcher picks the human removed)

The output (stdout) is curated into docs/ql-scan-criteria.md and used as the
rubric for the AI scan of the 10 PDVD events.

Usage: python analyze_labels.py
"""

import glob
import json
import math
import os

PDHD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/work/ql_labels"
SBND = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work/ql_labels"

SETS = {
    "pdhd": sorted(glob.glob(os.path.join(PDHD, "labels-evt*.json"))),
    "sbnd-mc": sorted(glob.glob(os.path.join(SBND, "mc", "labels-evt*.json"))),
    "sbnd-data": sorted(glob.glob(os.path.join(SBND, "data", "labels-evt*.json"))),
}

FLAG_KEYS = ["consistent", "potential_bad_match", "close_to_PMT",
             "at_x_boundary", "spec_end", "window_truncated", "contained",
             "at_cathode", "two_boundary"]


def c2n(e):
    m = e["metrics"]
    return m["chi2"] / m["ndf"] if m.get("ndf") else float("nan")


def ratio(e):
    m = e["metrics"]
    return m["total_pred_light"] / m["total_PE"] if m.get("total_PE") else float("nan")


def quantiles(vals, qs=(0.05, 0.25, 0.50, 0.75, 0.95)):
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return ["-"] * len(qs)
    out = []
    for q in qs:
        i = q * (len(vals) - 1)
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        out.append(vals[lo] + (vals[hi] - vals[lo]) * (i - lo))
    return out


def fmtq(vals):
    return "  ".join(f"{v:9.3f}" if v != "-" else f"{v:>9}" for v in quantiles(vals))


def collect():
    cats = {}   # (set, category) -> list of entries
    for name, files in SETS.items():
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            for e in d.get("matches", []):
                cat = "kept" if e["flags"].get("auto_selected") else "added"
                cats.setdefault((name, cat), []).append(e)
            for e in d.get("rejected_auto", []):
                cats.setdefault((name, "rejected"), []).append(e)
    return cats


def main():
    cats = collect()

    print("== sample sizes ==")
    for (name, cat), es in sorted(cats.items()):
        print(f"  {name:10s} {cat:9s} {len(es):5d}")

    metrics = [("ks_dis", lambda e: e["metrics"]["ks_dis"]),
               ("chi2/ndf", c2n),
               ("strength", lambda e: e["metrics"]["strength"]),
               ("pred/meas", ratio),
               ("ndf", lambda e: float(e["metrics"]["ndf"])),
               ("total_PE", lambda e: e["metrics"]["total_PE"])]

    hdr = "  ".join(f"{q:>9}" for q in ["q05", "q25", "q50", "q75", "q95"])
    for mname, fn in metrics:
        print(f"\n== {mname} ==   {hdr}")
        for (name, cat), es in sorted(cats.items()):
            print(f"  {name:10s} {cat:9s} {fmtq([fn(e) for e in es])}")

    print("\n== flag incidence (fraction of entries with flag True) ==")
    for (name, cat), es in sorted(cats.items()):
        row = []
        for k in FLAG_KEYS:
            vals = [e["flags"].get(k) for e in es]
            known = [v for v in vals if v is not None]
            row.append(f"{k}={sum(bool(v) for v in known)/len(known):.2f}"
                       if known else f"{k}=?")
        print(f"  {name:10s} {cat:9s} " + " ".join(row))

    # Threshold sweeps on the set with real rejections (PDHD).
    kept = cats.get(("pdhd", "kept"), [])
    rej = cats.get(("pdhd", "rejected"), [])
    if kept and rej:
        print("\n== PDHD sweeps: fraction of kept / rejected passing cut ==")
        for cut in [0.02, 0.05, 0.06, 0.1, 0.2, 0.4, 0.7]:
            fk = sum(e["metrics"]["ks_dis"] < cut for e in kept) / len(kept)
            fr = sum(e["metrics"]["ks_dis"] < cut for e in rej) / len(rej)
            print(f"  ks_dis < {cut:5.2f}   kept {fk:.2f}   rejected {fr:.2f}")
        for cut in [2, 5, 10, 35, 100, 1000]:
            fk = sum(c2n(e) < cut for e in kept) / len(kept)
            fr = sum(c2n(e) < cut for e in rej) / len(rej)
            print(f"  chi2/ndf < {cut:5d}  kept {fk:.2f}   rejected {fr:.2f}")
        for lo, hi in [(0.25, 4), (0.1, 10), (0.05, 20)]:
            fk = sum(lo < ratio(e) < hi for e in kept) / len(kept)
            fr = sum(lo < ratio(e) < hi for e in rej) / len(rej)
            print(f"  pred/meas in ({lo},{hi})  kept {fk:.2f}   rejected {fr:.2f}")
        for cut in [0.05, 0.2, 0.5, 0.8]:
            fk = sum(e["metrics"]["strength"] > cut for e in kept) / len(kept)
            fr = sum(e["metrics"]["strength"] > cut for e in rej) / len(rej)
            print(f"  strength > {cut:4.2f}   kept {fk:.2f}   rejected {fr:.2f}")

        # 2D occupancy: where do rejected live that kept do not?
        print("\n== PDHD 2D: ks x chi2/ndf occupancy (kept | rejected) ==")
        ks_edges = [0, 0.06, 0.2, 0.5, 1.01]
        c_edges = [0, 5, 35, 1e9]
        for i in range(len(ks_edges) - 1):
            row = []
            for j in range(len(c_edges) - 1):
                nk = sum(ks_edges[i] <= e["metrics"]["ks_dis"] < ks_edges[i+1]
                         and c_edges[j] <= c2n(e) < c_edges[j+1] for e in kept)
                nr = sum(ks_edges[i] <= e["metrics"]["ks_dis"] < ks_edges[i+1]
                         and c_edges[j] <= c2n(e) < c_edges[j+1] for e in rej)
                row.append(f"{nk:3d}|{nr:3d}")
            print(f"  ks[{ks_edges[i]:.2f},{ks_edges[i+1]:.2f})  " + "  ".join(row))
        print("  (columns: chi2/ndf [0,5) [5,35) [35,inf))")


if __name__ == "__main__":
    main()

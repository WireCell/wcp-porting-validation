#!/usr/bin/env python3
"""First-pass STM fit statistics over the 30-event knob-on roots (doc 41).

Per-root and total: fitted clusters/passes, status tally, two-TPC coverage,
reduced_chi2 and dx distributions, x-frame sanity (T0-corrected vs raw), and
the MIP plateau estimate from long tracks (dQ/dx median over points with
rr > 40 cm, per TPC).

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 stmon_stats.py      # reads work-mcp{10,1000,1000b}-stmon/*/tracking-stm.root
"""
import glob
import os

import numpy as np
import uproot

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
ROOTS = ["work-mcp10-stmon", "work-mcp1000-stmon", "work-mcp1000b-stmon"]
STATUS = {0: "accepted-STM", 1: "TGM", 2: "long-leftover", 3: "eval-fail",
          4: "extra-tracks", 5: "proton", 6: "no-decision"}

HALF_DRIFT = 200.0  # cm, SBND per-TPC drift length

tot = dict(files=0, clusters=0, passes=0, pts=0)
x_all = []
status_tally = {}
plateau = {0: [], 1: []}
plateau_acc = {0: [], 1: []}
chi2_all, dx_all = [], []
tpc_cov = dict(tpc0_only=0, tpc1_only=0, both=0)

for root in ROOTS:
    files = sorted(glob.glob(f"{BASE}/{root}/nusel_evt*/tracking-stm.root"))
    print(f"{root}: {len(files)} tracking-stm.root files")
    for fp in files:
        f = uproot.open(fp)
        tot["files"] += 1
        p = f["T_stm_pass"].arrays(library="np")
        tot["passes"] += len(p["pass"])
        tot["clusters"] += len(set(p["cluster_id"].tolist()))
        for s in p["status"]:
            status_tally[int(s)] = status_tally.get(int(s), 0) + 1
        r = f["T_rec_charge"].arrays(["x", "q", "nq", "rr", "cluster_id",
                                      "reduced_chi2", "status"], library="np")
        trun = f["Trun"].arrays(library="np")
        dQ = (r["q"] - trun["dQdx_offset"][0]) / trun["dQdx_scale"][0]
        with np.errstate(divide="ignore", invalid="ignore"):
            dqdx = np.where(r["nq"] > 0, dQ / r["nq"], 0.0)
        tot["pts"] += len(r["x"])
        chi2_all.append(r["reduced_chi2"])
        dx_all.append(r["nq"])
        x_all.append(r["x"])
        apa = (r["x"] > 0).astype(int)
        for cid in set(r["cluster_id"].tolist()):
            m = r["cluster_id"] == cid
            a = set(apa[m].tolist())
            key = ("both" if a == {0, 1}
                   else ("tpc1_only" if a == {1} else "tpc0_only"))
            tpc_cov[key] += 1
        # MIP plateau: points far from the stopping end on long fits, both
        # over every recorded pass and over accepted-STM passes only
        # (status 0) -- the accepted set is the one doc 41 quotes.
        for t in (0, 1):
            m = (apa == t) & (r["rr"] > 40) & (dqdx > 0)
            plateau[t].extend(dqdx[m].tolist())
            plateau_acc[t].extend(dqdx[m & (r["status"] == 0)].tolist())

print("\n=== totals ===")
print(f"events with dump: {tot['files']}  fitted clusters: {tot['clusters']}"
      f"  passes: {tot['passes']}  fit points: {tot['pts']}")
print("pass status tally:")
for s in sorted(status_tally):
    print(f"  {s} {STATUS.get(s, '?'):15s}: {status_tally[s]}")
print(f"per-(cluster,event) TPC coverage: {tpc_cov}")
chi2 = np.concatenate(chi2_all) if chi2_all else np.array([])
dx = np.concatenate(dx_all) if dx_all else np.array([])
if len(chi2):
    print(f"reduced_chi2: median {np.median(chi2):.2f}  "
          f"p90 {np.percentile(chi2, 90):.2f}  max {chi2.max():.1f}")
    print(f"dx (cm): median {np.median(dx):.2f}  p10 {np.percentile(dx,10):.2f}"
          f"  p90 {np.percentile(dx, 90):.2f}")
if x_all:
    # x-frame check (doc 40 §2 CONFIRM): the fitter works in the cluster's
    # default scope, which switch_scope sets to the T0-corrected frame.  A
    # raw-x frame would spread points over the whole ~2.7 m readout window
    # instead of staying inside the 200 cm drift.
    xx = np.concatenate(x_all)
    print(f"x frame: min {xx.min():.1f} max {xx.max():.1f} cm  "
          f"frac |x|>{HALF_DRIFT:.0f}cm {np.mean(np.abs(xx) > HALF_DRIFT):.4f}"
          f"  (T0-corrected if ~0)")
for label, pl in (("all passes", plateau), ("accepted-STM", plateau_acc)):
    for t in (0, 1):
        v = np.array(pl[t])
        if len(v):
            q25, q75 = np.percentile(v, [25, 75]) / 1e3
            print(f"TPC{t} plateau [{label}] (rr>40cm, {len(v)} pts): median "
                  f"dQ/dx {np.median(v)/1e3:.1f} ke/cm (p25-p75 {q25:.1f}-{q75:.1f})"
                  f"  (flat ref 50, muon table ~49)")

#!/usr/bin/env python3
"""doc pr/25 sec 3: is_shower_topology census (SBND evt 321107, a muon PR
called an electron).  Two independent measurements:

`population` -- reads every arm's tracking-pr.root (T_rec_charge), and for
every segment written (not just the nu-candidate main) reports geometric
length, PCA angle to drift, and flag_shower.  Answers two questions:
  (a) does the shower-flag rate on long (>50cm) segments depend on angle to
      drift (the "isochronous tracks get called showers" theory)?
  (b) what is the exact population of long shower-flagged segments -- the
      only segments PRSegmentFunctions.cxx's long-track guard
      (`length>50cm && total_length1/2 < 0.25*L`) can ever touch, and hence
      the exact blast radius of any change to that guard.

`guard` -- parses the env-gated `shower_topo dbg:` lines that
segment_is_shower_topology emits when run with WCT_SHOWER_TOPO_DEBUG=1 (see
PRSegmentFunctions.cxx, added this round -- the function previously had zero
active log lines).  Tabulates, per L>50cm evaluation: which of the 5
disjunction branches fired, the per-bucket dir_3 RMS distribution (median/
p90/count-over-cut), coverage (total_effective_length / geometric length),
and the long-track guard's total_length1/2 fractions -- the exact
measurements the doc uses to reject the "sparse association" and "single
outlier bucket" theories and to show the 0.25 threshold has no natural break
across the affected population, including same-segment verdict flips between
TaggerCheckNeutrino's two clustering_points passes (before/after
improve_vertex).

Usage:
    python3 pr25_shower_topo_census.py population --arm work-vfmcp1k-prodon [--nu-only]
    python3 pr25_shower_topo_census.py guard --arm work-pr25s3-dbg21
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import uproot

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"


def _seg_table(root_path):
    """Return list of (cluster, seg_id, npts, len_cm, angle_to_drift_deg, flag_shower) for
    every sub_cluster_id in T_rec_charge with >=5 points."""
    t = uproot.open(root_path)["T_rec_charge"].arrays(
        ["x", "y", "z", "sub_cluster_id", "flag_shower"], library="np")
    sid = t["sub_cluster_id"]
    out = []
    for s in np.unique(sid):
        if s < 0:
            continue
        m = sid == s
        if m.sum() < 5:
            continue
        P = np.stack([t["x"][m], t["y"][m], t["z"][m]], 1)
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        C = P - P.mean(0)
        _, _, V = np.linalg.svd(C, full_matrices=False)
        ang = float(np.degrees(np.arccos(min(1.0, abs(V[0][0])))))
        out.append((int(s) // 1000, int(s), int(m.sum()), L, ang, float(t["flag_shower"][m].mean())))
    return out


def _nu_main(nusel_tsv):
    if not os.path.exists(nusel_tsv):
        return None
    for ln in open(nusel_tsv):
        p = ln.split()
        if len(p) > 21 and p[-1] == "nu-candidate":
            return int(p[3])
    return None


def _one_event(d):
    ev = int(os.path.basename(d).replace("pr_evt", ""))
    root = os.path.join(d, "tracking-pr.root")
    if not os.path.exists(root):
        return None
    try:
        segs = _seg_table(root)
    except Exception:
        return None
    main = _nu_main(os.path.join(d, f"nusel-evt{ev}.tsv"))
    return ev, main, segs


def cmd_population(arm, nu_only):
    dirs = sorted(glob.glob(os.path.join(BASE, arm, "pr_evt*")))
    rows = [r for r in Pool(8).map(_one_event, dirs) if r]
    print(f"# events with usable tracking-pr.root: {len(rows)}")

    all_long = []
    for ev, main, segs in rows:
        for cl, sid, npts, L, ang, shw in segs:
            if nu_only and cl != main:
                continue
            if L > 50:
                all_long.append((ev, cl, sid, npts, L, ang, shw))

    print(f"# segments >50cm{' (nu-candidate main only)' if nu_only else ' (ALL clusters)'}: {len(all_long)}")

    # (a) angle-dependence of the shower-flag rate
    bins = [(0, 30), (30, 50), (50, 65), (65, 75), (75, 85), (85, 90)]
    print("\nangle-to-drift bin | n | shower-flagged | rate")
    for lo, hi in bins:
        b = [r for r in all_long if lo <= r[5] < hi]
        ns = sum(1 for r in b if r[6] > 0.5)
        rate = 100.0 * ns / max(1, len(b))
        print(f"  {lo:3d}-{hi:3d} deg      | {len(b):4d} | {ns:4d}          | {rate:5.1f}%")

    # (b) the exact blast-radius population
    shw_long = sorted([r for r in all_long if r[6] > 0.5], key=lambda r: -r[4])
    print(f"\n# shower-flagged AND >50cm (blast radius of any guard change): {len(shw_long)}"
          f"  across {len(set(r[0] for r in shw_long))} distinct events")
    print("\nevt      cid   sid    npts   L(cm)  angle(deg)")
    for ev, cl, sid, npts, L, ang, shw in shw_long:
        print(f"{ev:8d} {cl:5d} {sid:6d} {npts:6d} {L:7.1f} {ang:7.1f}")


_GUARD_RE = re.compile(
    r"guard branch (\d+) L ([\d.]+)cm total_length1 ([\d.]+)cm\(([\d.]+)\) "
    r"total_length2 ([\d.]+)cm\(([\d.]+)\) demoted (\w+) final_shower (\w+)")
_ENTRY_RE = re.compile(
    r"seg (\S+) L ([\d.]+)cm assoc_npts (\d+) nbuckets (\d+) n_over0\.4cm (\d+) "
    r"rms_p50 ([\d.]+)cm rms_p90 ([\d.]+)cm max_spread ([\d.]+)cm lsl ([\d.]+)cm "
    r"tel ([\d.]+)cm lsl/tel ([\d.]+) tel/L ([\d.]+) branch (\d+)")


def cmd_guard(arm):
    """Parse WCT_SHOWER_TOPO_DEBUG=1 log output (both the entry-line with the
    5-way disjunction terms and the guard-line with total_length1/2) for
    every pr_evt*/wct_pr_evt*.log under `arm`."""
    dirs = sorted(glob.glob(os.path.join(BASE, arm, "pr_evt*")))
    guard_rows, entry_rows = [], []
    for d in dirs:
        ev = int(os.path.basename(d).replace("pr_evt", ""))
        logf = os.path.join(d, f"wct_pr_evt{ev}.log")
        if not os.path.exists(logf):
            continue
        for ln in open(logf, errors="replace"):
            if "shower_topo dbg" not in ln:
                continue
            m = _GUARD_RE.search(ln)
            if m:
                branch, L, l1, f1, l2, f2, dem, fin = m.groups()
                guard_rows.append(dict(ev=ev, branch=int(branch), L=float(L),
                                        f1=float(f1), f2=float(f2), demoted=dem, final=fin))
                continue
            m = _ENTRY_RE.search(ln)
            if m:
                sid, L, npts, nb, nover, p50, p90, mx, lsl, tel, lsltel, telL, branch = m.groups()
                entry_rows.append(dict(ev=ev, L=float(L), nbuckets=int(nb), n_over=int(nover),
                                        p50=float(p50), p90=float(p90), max_spread=float(mx),
                                        lsl_tel=float(lsltel), tel_L=float(telL), branch=int(branch)))

    long50 = [r for r in guard_rows if r["L"] > 50]
    print(f"# guard-evaluated entries: {len(guard_rows)}   L>50cm: {len(long50)}")
    survived = [r for r in long50 if r["final"] == "true"]
    demoted = [r for r in long50 if r["final"] == "false"]
    print(f"  survived (still shower): {len(survived)}   demoted (existing guard already fixes): {len(demoted)}")

    print("\n=== entry-line decision terms, L>50cm, still-shower (candidates A/B) ===")
    print("evt        L(cm) nbuckets n_over0.4 rms_p50 rms_p90 max_spread tel/L  branch")
    seen = set()
    for r in sorted([e for e in entry_rows if e["L"] > 50], key=lambda r: -r["tel_L"]):
        key = (r["ev"], round(r["L"], 1))
        if key in seen:
            continue
        seen.add(key)
        print(f"{r['ev']:8d} {r['L']:7.1f} {r['nbuckets']:8d} {r['n_over']:9d} "
              f"{r['p50']:7.2f} {r['p90']:7.2f} {r['max_spread']:10.2f} {r['tel_L']:5.3f} {r['branch']:6d}")

    print("\n=== long-track guard fractions, L>50cm, still-shower (candidate C) ===")
    print("evt        L(cm)  f1     f2     max(f1,f2)")
    seen = set()
    for r in sorted(survived, key=lambda r: -max(r["f1"], r["f2"])):
        key = (r["ev"], round(r["L"], 1))
        if key in seen:
            continue
        seen.add(key)
        print(f"{r['ev']:8d} {r['L']:7.1f} {r['f1']:.3f}  {r['f2']:.3f}  {max(r['f1'], r['f2']):.3f}")

    # verdict stability across TaggerCheckNeutrino's two clustering_points passes
    print("\n=== verdict stability across the two clustering_points passes (before/after improve_vertex) ===")
    byev = {}
    for r in long50:
        byev.setdefault(r["ev"], []).append(r)
    for ev, rs in sorted(byev.items()):
        verdicts = set(r["final"] for r in rs)
        if len(verdicts) > 1:
            lens = sorted(set(round(r["L"], 1) for r in rs))
            print(f"  evt {ev}: UNSTABLE -- lengths {lens} verdicts {verdicts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("population")
    p1.add_argument("--arm", required=True)
    p1.add_argument("--nu-only", action="store_true",
                     help="restrict to the nu-candidate main cluster only")
    p2 = sub.add_parser("guard")
    p2.add_argument("--arm", required=True)
    args = ap.parse_args()
    if args.cmd == "population":
        cmd_population(args.arm, args.nu_only)
    elif args.cmd == "guard":
        cmd_guard(args.arm)

#!/usr/bin/env python3
"""doc pr/128 -- characterise the two PF/kine blind spots well enough to write
a predicate for each.

pr127_blindspot_census.py counted them.  This one answers the three questions
that decide the implementation, all from calib dumps + run logs already on
disk (no new arms, no C++):

  Q1  Can the EXISTING orphan predicate reach class A?
      segment_orphan_confident_track requires segment_confident_nonelectron_pid,
      whose last line is `particle_score() < 1.0`.  Pull particle_score for the
      class-A near-and-long objects.  Score 100 is the trajectory branch's
      unconditional sentinel stamp -- if much of the class carries it, the
      existing predicate fires on nothing and a new one is needed.

  Q2  Of the conn-4 energy, how much is the far-away activity the owner says
      must NOT be counted, and how much is the candidate's own?
      conn-4 is assigned in three places (NeutrinoShowerClustering.cxx:3733
      other-clusters 80cm cut, :3858 pr/74 conn3_unreachable, :6435/:6645
      pr/123+pr/124 prune re-seeds).  Split by (a) is the shower's cluster the
      MAIN cluster, (b) distance from that cluster's point cloud to the main
      cluster's point cloud.

  Q3  Which producer made each near conn-4 shower?  Read from the arm's own
      run log (pr74 conn3_unreachable / pr123 pass4_prune / pr124 pass4_prune2
      lines), so the fix goes at the producer and not at the display.

Distances are computed from segment fit points (`points` in the dump).  That
is a proxy for Facade::Cluster::get_closest_dis, which uses the imaging blob
cloud -- good enough to sort 5 cm from 150 cm, not for reproducing the 80 cm
cut exactly.  The exact, C++-side census is the WCT_PFNEAR_DEBUG tape.

Repro:
  ./scripts/pr128_completeness_census.py 'work-pr125r1-flipS98-*' \
      'work-pr125r1-flipS141-*' --tsv docs/pr/pr128-completeness-census.tsv
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)

# producer lines in the per-event run log, in the order they can fire
PRODUCERS = (
    ("pr74_conn3_unreachable", re.compile(r"pr74 conn3_unreachable: promote gidx=(\d+) .*conn=(\d)")),
    ("pr123_pass4_prune",      re.compile(r"pr123 pass4_prune: shower_id=\d+ sheds \d+ detached seg\(s\).*conn=(\d)")),
    ("pr124_pass4_prune2",     re.compile(r"pr124 pass4_prune2: shower_id=\d+ sheds \d+ band seg\(s\).*conn=(\d)")),
)


def producers_for(evdir):
    """Which conn-4-producing passes fired in this event, from its run log."""
    hits = set()
    for lg in glob.glob(os.path.join(evdir, "*.log")):
        try:
            txt = open(lg, errors="replace").read()
        except OSError:
            continue
        for name, rx in PRODUCERS:
            for m in rx.finditer(txt):
                if m.groups()[-1] == "4":
                    hits.add(name)
    return ",".join(sorted(hits)) or "-"


def min_gap(a, b):
    """Min distance between two (N,3) point sets, cm.  -1 if either is empty."""
    if a.size == 0 or b.size == 0:
        return -1.0
    return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--tsv")
    ap.add_argument("--min-cm", type=float, default=10.0)
    ap.add_argument("--near-cm", type=float, default=10.0)
    args = ap.parse_args()

    rows = []
    seen = set()
    for pat in args.roots:
        for root in sorted(glob.glob(os.path.join(SX, pat)) or glob.glob(pat)):
            for dj in sorted(glob.glob(os.path.join(root, "pr_evt*", "calib-pr-evt*.json"))):
                evdir = os.path.dirname(dj)
                ev = int(os.path.basename(evdir)[len("pr_evt"):])
                if ev in seen:
                    continue
                z = os.path.join(evdir, "mabc-pr.zip")
                if not os.path.exists(z):
                    continue
                seen.add(ev)
                j = json.load(open(dj))
                mv = j.get("main_vertex") or {}
                if "x" not in mv:
                    continue
                main_cl = mv.get("cluster_id")

                bycl = {}
                for s in j["segments"]:
                    bycl.setdefault(s["cluster_id"], []).extend(
                        [(p["x"], p["y"], p["z"]) for p in s["points"]])
                bycl = {k: np.array(v) for k, v in bycl.items()}
                mpts = bycl.get(main_cl, np.empty((0, 3)))

                # ---- Q1: class A objects, with the PID score the existing
                # predicate would test.  Same selection as pr127_blindspot_
                # census kind A; gap is to everything the PF tree emits.
                import zipfile
                shown = set()

                def walk(n):
                    shown.add(int(n.get("id", -1)))
                    for c in n.get("children", []):
                        walk(c)

                with zipfile.ZipFile(z) as zf:
                    for r in json.loads(zf.read("data/0/0-mc.json")):
                        walk(r)
                disp = []
                for s in j["segments"]:
                    sid = s.get("shower_id")
                    if s["id"] in shown or (sid not in (None, -1) and sid in shown):
                        disp.extend([(p["x"], p["y"], p["z"]) for p in s["points"]])
                disp = np.array(disp) if disp else np.empty((0, 3))

                prod = None
                for s in j["segments"]:
                    if s.get("flag_shower"):
                        continue
                    if s.get("shower_id") not in (None, -1):
                        continue
                    if s["id"] in shown:
                        continue
                    if s["cluster_id"] == main_cl:
                        kind = "C"          # control: the pools already cover this
                    else:
                        kind = "A"
                    pts = np.array([(p["x"], p["y"], p["z"]) for p in s["points"]])
                    rows.append(dict(
                        kind=kind, event=ev, ident=s["id"], cluster=s["cluster_id"],
                        pdg=s.get("particle_id", 0), score=s.get("particle_score", -1.0),
                        dirsign=s.get("dirsign", 0), length=s["length"], energy=0.0,
                        gap=min_gap(pts, disp), gap_main=min_gap(pts, mpts), producer="-"))

                # ---- Q2/Q3: conn-4 showers
                for sh in j["showers"]:
                    if sh.get("start_connection_type") != 4:
                        continue
                    cl = sh["id"] // 1000
                    pts = bycl.get(cl, np.empty((0, 3)))
                    if prod is None:
                        prod = producers_for(evdir)
                    rows.append(dict(
                        kind="B_main" if cl == main_cl else "B_cross",
                        event=ev, ident=sh["id"], cluster=cl,
                        pdg=sh.get("particle_id", 0), score=-1.0, dirsign=0,
                        length=sh.get("total_length", 0.0),
                        energy=sh.get("kine_best", 0.0),
                        gap=-1.0,
                        gap_main=(0.0 if cl == main_cl else min_gap(pts, mpts)),
                        producer=prod))

    print("%d events scanned\n" % len(seen))

    # ---------------- Q1 ----------------
    A = [r for r in rows if r["kind"] == "A"]
    near = [r for r in A if 0 <= r["gap"] < args.near_cm and r["length"] >= args.min_cm]
    print("Q1  class A -- cross-cluster unclaimed tracks absent from the PF tree")
    print("    %d object(s) in %d event(s); near (<%.0f cm) and >=%.0f cm: %d in %d event(s)" % (
        len(A), len({r["event"] for r in A}), args.near_cm, args.min_cm,
        len(near), len({r["event"] for r in near})))
    ctrl = [r for r in rows if r["kind"] == "C"]
    print("    main-cluster control (the orphan pools DO cover these): %d object(s)" % len(ctrl))
    hi = [r for r in near if r["score"] >= 1.0]
    print("    particle_score >= 1.0 (rejected by segment_confident_nonelectron_pid): "
          "%d of %d  -- of which score==100: %d" % (
              len(hi), len(near), len([r for r in near if r["score"] >= 100.0])))
    print("    %-8s %-8s %-6s %8s %8s %9s" % ("event", "id", "pdg", "len_cm", "gap_cm", "score"))
    for r in sorted(near, key=lambda r: -r["length"]):
        print("    %-8d %-8d %-6d %8.1f %8.2f %9.3f%s" % (
            r["event"], r["ident"], r["pdg"], r["length"], r["gap"], r["score"],
            "   <- score term would reject" if r["score"] >= 1.0 else ""))

    # ---------------- Q2 ----------------
    B = [r for r in rows if r["kind"].startswith("B_")]
    bm = [r for r in B if r["kind"] == "B_main"]
    print("\nQ2  class B -- conn-4 showers (skipped by PF *and* by kine)")
    print("    %d shower(s) in %d event(s), %.0f MeV" % (
        len(B), len({r["event"] for r in B}), sum(r["energy"] for r in B)))
    print("    in the MAIN cluster (the candidate's own material): %d shower(s) in %d event(s), %.0f MeV"
          % (len(bm), len({r["event"] for r in bm}), sum(r["energy"] for r in bm)))
    print("    cross-cluster, by gap to the main cluster:")
    bx = [r for r in B if r["kind"] == "B_cross"]
    for lab, lo, hi_ in (("<5 cm", 0, 5), ("5-20", 5, 20), ("20-50", 20, 50),
                         ("50-80", 50, 80), ("80-150", 80, 150), ("150+", 150, 1e9)):
        sub = [r for r in bx if lo <= r["gap_main"] < hi_]
        big = [r for r in sub if r["energy"] > 20]
        print("      %-8s: %4d shower(s) %7.0f MeV   (>20 MeV: %2d, %6.0f MeV)" % (
            lab, len(sub), sum(r["energy"] for r in sub), len(big),
            sum(r["energy"] for r in big)))
    far = [r for r in bx if r["gap_main"] >= 50]
    print("    => %d of %d conn-4 showers (%.0f of %.0f MeV) are >=50 cm from the candidate:"
          "\n       far-away activity, NOT to be counted (owner 2026-08-29).  Untouched by this round."
          % (len(far), len(B), sum(r["energy"] for r in far), sum(r["energy"] for r in B)))

    # ---------------- Q3 ----------------
    print("\nQ3  the conn-4 showers this round is about (main-cluster, or cross-cluster within 20 cm)")
    print("    %-8s %-8s %-6s %8s %8s %9s  %s" % (
        "event", "id", "pdg", "E_MeV", "len_cm", "gap_main", "conn-4 producers that fired"))
    tgt = bm + [r for r in bx if 0 <= r["gap_main"] < 20]
    for r in sorted(tgt, key=lambda r: -r["energy"]):
        print("    %-8d %-8d %-6d %8.1f %8.1f %9.1f  %s" % (
            r["event"], r["ident"], r["pdg"], r["energy"], r["length"],
            r["gap_main"], r["producer"]))

    if args.tsv:
        out = args.tsv if os.path.isabs(args.tsv) else os.path.join(SX, args.tsv)
        with open(out, "w") as fh:
            fh.write("kind\tevent\tid\tcluster\tpdg\tscore\tdirsign\tlength_cm\t"
                     "energy_mev\tgap_displayed_cm\tgap_main_cluster_cm\tconn4_producers\n")
            for r in sorted(rows, key=lambda r: (r["kind"], -r["length"])):
                fh.write("%s\t%d\t%d\t%d\t%d\t%.3f\t%d\t%.1f\t%.1f\t%.2f\t%.2f\t%s\n" % (
                    r["kind"], r["event"], r["ident"], r["cluster"], r["pdg"], r["score"],
                    r["dirsign"], r["length"], r["energy"], r["gap"], r["gap_main"],
                    r["producer"]))
        print("\nwrote %s (%d rows)" % (out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

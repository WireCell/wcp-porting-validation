#!/usr/bin/env python3
"""doc pr/127 sec 5.2 -- how much reconstructed object never reaches the PF tree?

Motivated by 137238 (doc pr/127): a 79.3 cm muon and its 19.4 cm tail existed
in the reconstruction and appeared in neither the PF tree nor kine, because
every PF orphan pool AND the pr65 orphan audit line are same-cluster gated
(MultiAlgBlobClustering.cxx:1852, :2331, :2398) and conn-4 showers are skipped
outright (conn4_skip_segs).  Before proposing any fix, count the exposure.

Pure offline analysis of arms already on disk -- no new runs, no C++.  For
each event it compares the calib dump (every segment and shower the PR stage
built) against the PF tree actually emitted (mabc-pr.zip data/0/0-mc.json).
Node ids in both are the display id cluster_id*1000 + graph_index, so the
match is exact.

Three blind spots counted separately:

  A. cross-cluster tracks  : track segments (flag_shower=0, not owned by any
                             shower) in a cluster OTHER than the main vertex's,
                             absent from the PF tree.  Reported by length band
                             -- a 5 cm crumb is noise, a 50 cm track is a
                             particle.
                             CRITICAL COLUMN: `gap_cm`, the min distance from
                             the missing track's fit points to the fit points of
                             anything the PF tree DOES show.  A cosmic that the
                             candidate never touched sits metres away and is
                             correctly absent; the 137238 class is the small-gap
                             tail (its muon started 0 cm from a displayed shower
                             member).  Only the small-gap rows are a defect.
  B. conn-4 showers        : showers whose start_connection_type == 4; these are
                             dropped from the PF tree by design.  Summed
                             kine_best = energy the event dropped on the floor.
  C. main-cluster misses   : same as A but INSIDE the main cluster -- the class
                             the orphan machinery already covers, kept as the
                             control.  Should be near zero.

Repro:
  ./scripts/pr127_blindspot_census.py 'work-pr125r1-flipS98-*' 'work-pr125r1-flipS141-*' \
      --tsv docs/pr/pr127-blindspot-census.tsv
"""
import argparse
import glob
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
BANDS = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]


def pf_ids(zip_path):
    ids = set()

    def walk(n):
        ids.add(int(n.get("id", -1)))
        for c in n.get("children", []):
            walk(c)

    with zipfile.ZipFile(zip_path) as zf:
        for root in json.loads(zf.read("data/0/0-mc.json")):
            walk(root)
    return ids


def band(length_cm):
    for lo, hi in BANDS:
        if lo <= length_cm < hi:
            return "%g-%g" % (lo, hi) if hi < 1e9 else "50+"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--tsv")
    ap.add_argument("--min-cm", type=float, default=10.0,
                    help="length above which a missing track is called a particle")
    args = ap.parse_args()

    rows = []
    seen = set()
    for pat in args.roots:
        for root in sorted(glob.glob(os.path.join(SX, pat)) or glob.glob(pat)):
            for dj in sorted(glob.glob(os.path.join(root, "pr_evt*", "calib-pr-evt*.json"))):
                ev = int(os.path.basename(os.path.dirname(dj))[len("pr_evt"):])
                if ev in seen:
                    continue
                z = os.path.join(os.path.dirname(dj), "mabc-pr.zip")
                if not os.path.exists(z):
                    continue
                seen.add(ev)
                j = json.load(open(dj))
                shown = pf_ids(z)
                main_cl = (j.get("main_vertex") or {}).get("cluster_id")
                # point cloud of everything the PF tree DOES show, so a missing
                # object can be scored by its distance to the displayed candidate
                import numpy as np
                disp = []
                for s in j["segments"]:
                    sid = s.get("shower_id")
                    if s["id"] in shown or (sid not in (None, -1) and sid in shown):
                        disp.extend([(p["x"], p["y"], p["z"]) for p in s["points"]])
                disp = np.array(disp) if disp else None

                def gap_to_displayed(seg):
                    if disp is None:
                        return -1.0
                    pts = np.array([(p["x"], p["y"], p["z"]) for p in seg["points"]])
                    # min over pairs; both sides are small (<= a few hundred points)
                    d = np.sqrt(((pts[:, None, :] - disp[None, :, :]) ** 2).sum(-1))
                    return float(d.min())

                for s in j["segments"]:
                    if s.get("flag_shower"):
                        continue
                    if s.get("shower_id") not in (None, -1):
                        continue      # already inside a displayed shower
                    if s["id"] in shown:
                        continue
                    rows.append(dict(kind="A" if s["cluster_id"] != main_cl else "C",
                                     event=ev, arm=os.path.basename(root),
                                     ident=s["id"], cluster=s["cluster_id"],
                                     pdg=s.get("particle_id", 0),
                                     length=s["length"], energy=0.0,
                                     gap=gap_to_displayed(s)))
                for sh in j["showers"]:
                    if sh.get("start_connection_type") != 4:
                        continue
                    rows.append(dict(kind="B", event=ev, arm=os.path.basename(root),
                                     ident=sh["id"], cluster=sh["id"] // 1000,
                                     pdg=sh.get("particle_id", 0),
                                     length=sh.get("total_length", 0.0),
                                     energy=sh.get("kine_best", 0.0), gap=-1.0))

    print("%d events scanned\n" % len(seen))
    for kind, title in (("A", "A. cross-cluster tracks absent from the PF tree"),
                        ("C", "C. main-cluster tracks absent (control -- orphan machinery covers these)"),
                        ("B", "B. conn-4 showers (dropped from PF and kine by design)")):
        sub = [r for r in rows if r["kind"] == kind]
        evts = {r["event"] for r in sub}
        print("%s\n   %d object(s) in %d/%d events" % (title, len(sub), len(evts), len(seen)))
        if kind == "B":
            print("   summed kine_best = %.0f MeV; events with >20 MeV dropped: %d" % (
                sum(r["energy"] for r in sub),
                len({r["event"] for r in sub if r["energy"] > 20})))
        big = [r for r in sub if r["length"] >= args.min_cm]
        print("   >= %.0f cm: %d object(s) in %d event(s), summed length %.0f cm" % (
            args.min_cm, len(big), len({r["event"] for r in big}),
            sum(r["length"] for r in big)))
        counts = {}
        for r in sub:
            counts[band(r["length"])] = counts.get(band(r["length"]), 0) + 1
        print("   by length band: %s\n" % ", ".join(
            "%s cm: %d" % (b, counts[b]) for b in sorted(counts, key=lambda x: float(x.split("-")[0].rstrip("+")))))
        if kind == "A":
            near = [r for r in sub if 0 <= r["gap"] < 10.0]
            nearbig = [r for r in near if r["length"] >= args.min_cm]
            print("   within 10 cm of a DISPLAYED object (the 137238 class): %d object(s) "
                  "in %d event(s); >= %.0f cm: %d in %d event(s)" %
                  (len(near), len({r["event"] for r in near}), args.min_cm,
                   len(nearbig), len({r["event"] for r in nearbig})))
        for r in sorted(big, key=lambda r: -r["length"])[:8]:
            print("     evt %-8d id=%-7d cluster=%-4d pdg=%-5d len=%6.1f cm  E=%7.1f MeV  gap=%s" %
                  (r["event"], r["ident"], r["cluster"], r["pdg"], r["length"], r["energy"],
                   "%.1f cm" % r["gap"] if r["gap"] >= 0 else "-"))
        if kind == "A":
            print("   -- the small-gap rows, sorted by gap:")
            for r in sorted([x for x in sub if 0 <= x["gap"] < 10.0 and x["length"] >= args.min_cm],
                            key=lambda r: r["gap"])[:12]:
                print("     evt %-8d id=%-7d cluster=%-4d pdg=%-5d len=%6.1f cm  gap=%5.2f cm" %
                      (r["event"], r["ident"], r["cluster"], r["pdg"], r["length"], r["gap"]))
        print()

    if args.tsv:
        out = args.tsv if os.path.isabs(args.tsv) else os.path.join(SX, args.tsv)
        with open(out, "w") as fh:
            fh.write("kind\tevent\tarm\tid\tcluster\tpdg\tlength_cm\tenergy_mev\tgap_cm\n")
            for r in sorted(rows, key=lambda r: (r["kind"], -r["length"])):
                fh.write("%s\t%d\t%s\t%d\t%d\t%d\t%.1f\t%.1f\t%.2f\n" %
                         (r["kind"], r["event"], r["arm"], r["ident"], r["cluster"],
                          r["pdg"], r["length"], r["energy"], r["gap"]))
        print("wrote %s (%d rows)" % (out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

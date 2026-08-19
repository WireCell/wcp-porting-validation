#!/usr/bin/env python3
"""doc pr/93 -- per-shower composition census.

For every pdg-11 shower with kine_best > 100 MeV, join segments[].shower_id
== showers[].id and report the quantities the round's rescue designs (K1
"vote widening", K-ACCEPT, K-CASEB) would be evaluated against:

  trk_len      sum of length over non-shower-flagged members with
               |particle_id| in {13,211,2212} (K1's proposed track_length
               bucket)
  trk_frac     trk_len / total_length  -- the vote's own flip point is 0.5
  trk_frac_sl  same restricted to members additionally passing
               segment_is_straight_long_track's own numeric form
               (length>10cm && (direct>=34cm || direct>0.93*length)) --
               approximated here from endpoint distance only, since this
               script has no access to per-point direct-length; flagged
               with a leading '~' as an upper bound, not an exact replay.
  maxUnflag    longest non-shower-flagged member carrying |pdg|==11
               (post-hoc geometry; kept for continuity with the earlier
               census, NOT used as a gate predicate -- see doc caveat)

Never treats this output as ground truth for a fix predicate: every number
here is measured on the FINAL merged object, after every merge and after
the vote has already run (pr/83 round-4 lesson). It is the regression
SCREEN, not the gate.

Usage: pr93_shower_composition.py <arm_dir> [<arm_dir> ...] --out FILE.tsv
"""
import argparse
import glob
import json
import math
import os
import sys

TRACK_PDGS = {13, 211, 2212}


def evtid(path):
    b = os.path.basename(path)
    return b[len("calib-pr-evt"):-len(".json")]


def straight_long_upper_bound(seg):
    """Endpoint-distance-only stand-in for segment_is_straight_long_track.
    True length uses the fitted trajectory's direct arc; here we only have
    start/end 3-D points in the dump, which over-estimates straightness for
    a wiggly track. Upper bound -- may over-count as straight, never under."""
    if not seg.get("points"):
        return seg["length"] > 34.0
    pts = seg["points"]
    p0, p1 = pts[0], pts[-1]
    d = math.dist((p0["x"], p0["y"], p0["z"]), (p1["x"], p1["y"], p1["z"]))
    L = seg["length"]
    if L <= 10.0:
        return False
    return d >= 34.0 or d > 0.93 * L


def rows_for_file(path, sample):
    j = json.load(open(path))
    e = evtid(path)
    segs_by_shower = {}
    for s in j["segments"]:
        segs_by_shower.setdefault(s.get("shower_id"), []).append(s)

    out = []
    for sh in j["showers"]:
        if sh["particle_id"] != 11 or sh["kine_best"] < 100:
            continue
        mem = segs_by_shower.get(sh["id"], [])
        if not mem:
            continue
        total = sh["total_length"] or 1e-9

        trk_len = sum(s["length"] for s in mem
                      if not s["flag_shower"] and abs(s["particle_id"]) in TRACK_PDGS)
        trk_len_sl = sum(s["length"] for s in mem
                          if not s["flag_shower"] and abs(s["particle_id"]) in TRACK_PDGS
                          and straight_long_upper_bound(s))
        unflag_e11 = [s["length"] for s in mem
                      if not s["flag_shower"] and abs(s["particle_id"]) == 11]

        out.append(dict(
            sample=sample, evt=e, shower_id=sh["id"], start_seg=sh.get("shower_id"),
            E=sh["kine_best"], L=sh["total_length"], nmem=len(mem),
            trk_len=round(trk_len, 2), trk_frac=round(trk_len / total, 4),
            trk_frac_sl=round(trk_len_sl / total, 4),
            maxUnflag=round(max(unflag_e11, default=0.0), 2),
            conn=sh["start_connection_type"],
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="arm_dir[:sample_label] pairs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_rows = []
    for spec in args.arms:
        if ":" in spec:
            arm, label = spec.split(":", 1)
        else:
            arm, label = spec, os.path.basename(spec.rstrip("/"))
        for p in sorted(glob.glob(os.path.join(arm, "pr_evt*", "calib-pr-evt*.json"))):
            all_rows.extend(rows_for_file(p, label))

    cols = ["sample", "evt", "shower_id", "start_seg", "E", "L", "nmem",
            "trk_len", "trk_frac", "trk_frac_sl", "maxUnflag", "conn"]
    with open(args.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in all_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    n_over_half = sum(1 for r in all_rows if r["sample"] != "CASES" and r["trk_frac"] > 0.5)
    n_total_ctrl = sum(1 for r in all_rows if r["sample"] != "CASES")
    print(f"wrote {len(all_rows)} rows to {args.out}")
    print(f"control showers with trk_frac>0.5 (K1's own flip point): {n_over_half}/{n_total_ctrl}")


if __name__ == "__main__":
    main()

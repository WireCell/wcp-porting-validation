#!/usr/bin/env python3
"""doc pr/40 round 7: census of long main-cluster pdg==11 segments across an
arm, reading tracking-pr.root:T_rec_charge (no PR_EXTRA_STAGES=pr_display
rerun needed -- validated exact against the calib JSON on work-r2mc-prod0813,
71/71 segments, 0 mismatch).

Motivating cases (SBND evts 54629, 320865): a long, straight, MIP-like track
segment reconstructed and displayed as an electron.  This differs from
pr40_seg_pid.py's existing G4 census in three ways:

  1. reads T_rec_charge, so it needs no pr_display rerun at all;
  2. reports flag_shower as a COLUMN, not a cut -- pr40_seg_pid.py's
     `!flag_shower` filter excludes 320865's own class (Family B:
     particle_id==11, flag_shower==1);
  3. uses MIP_dQdx_median = 43000 e/cm, the scale every C++ knob in this
     pipeline actually binds to (m_mip_dqdx_median).  pr40_seg_pid.py's
     MIP=56000 is a latent scale bug -- see the pr/40 round 7 doc note.

Usage:
    pr40r7_census.py <arm_dir> [<arm_dir> ...] [--min-len CM] [--top N]
                      [--out TSV]

<arm_dir> is a PR out_root (contains pr_evt<ID>/tracking-pr.root).  Multiple
arms may be given (e.g. the mcp1k + nueCC48 + ncpi0 manifest); each event is
attributed to the arm it was found in.

Selection: particle_id==11 AND is_main_cluster AND track length > --min-len
(default 20cm).  For each selected segment reports length, direct/arc ratio
(the segment_is_straight_long_track geometry: True iff L>10cm AND
(direct>=34cm OR direct>0.93*L)), median dQ/dx in xMIP (43000 e/cm), and
flag_shower.  Ranked by "flagrancy" = length x max(0, 1 - |ratio-1|) so a
long, ~1x-MIP, straight segment sorts first; a --top N cap logs how many
were dropped.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import uproot

MIP_DQDX_MEDIAN = 43000.0  # e/cm -- m_mip_dqdx_median scale, NOT the 56000
                            # some older pr/40 scripts use (see docstring).


def straight_long(length_cm, direct_cm, min_length=10.0, min_direct=34.0, straight_ratio=0.93):
    """Mirrors PRSegmentFunctions.cxx segment_is_straight_long_track exactly."""
    if length_cm <= min_length:
        return False
    return direct_cm >= min_direct or direct_cm > straight_ratio * length_cm


def census_arm(arm_dir, min_len_cm):
    rows = []
    for root_path in sorted(glob.glob(os.path.join(arm_dir, "pr_evt*", "tracking-pr.root"))):
        evt_dir = os.path.dirname(root_path)
        evt = int(os.path.basename(evt_dir).replace("pr_evt", ""))
        try:
            f = uproot.open(root_path)
            if "T_rec_charge" not in f:
                continue
            t = f["T_rec_charge"]
            a = t.arrays(
                ["x", "y", "z", "q", "nq", "flag_shower", "real_cluster_id", "cluster_id", "particle_id"],
                library="np",
            )
        except Exception as e:
            print(f"# WARN: {root_path}: {e}", file=sys.stderr)
            continue

        rcid = a["real_cluster_id"]
        cid = a["cluster_id"]
        pdg = a["particle_id"]
        fs = a["flag_shower"]

        # main cluster id = the mode of cluster_id (the nu-candidate main
        # cluster is the only one T_rec_charge carries per event in this
        # pipeline's per-event tree, but guard against multi-cluster dumps).
        if len(cid) == 0:
            continue
        main_cid = np.bincount(cid[cid >= 0]).argmax() if (cid >= 0).any() else None

        for seg in np.unique(rcid):
            m = rcid == seg
            n = int(m.sum())
            if n < 2:
                continue
            is_main = bool((cid[m][0] == main_cid)) if main_cid is not None else False
            if pdg[m][0] != 11 or not is_main:
                continue

            x, y, z = a["x"][m], a["y"][m], a["z"][m]
            d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
            length = float(d.sum())
            if length <= min_len_cm:
                continue
            direct = float(math.dist((x[0], y[0], z[0]), (x[-1], y[-1], z[-1])))

            q, nq = a["q"][m], a["nq"][m]
            valid = nq > 0
            if valid.any():
                dqdx = (q[valid] + 1000.0) * 10.0 / nq[valid]
                med = float(np.median(dqdx))
            else:
                med = 0.0
            xmip = med / MIP_DQDX_MEDIAN if med > 0 else 0.0

            rows.append(
                dict(
                    arm=os.path.basename(arm_dir.rstrip("/")),
                    evt=evt,
                    seg=int(seg),
                    length=length,
                    direct=direct,
                    ratio=direct / length if length > 0 else 0.0,
                    xmip=xmip,
                    flag_shower=int(fs[m][0]),
                    straight_long=straight_long(length, direct),
                )
            )
    return rows


def flagrancy(row):
    # long, and as close to 1x MIP as possible (muon-like), and geometrically
    # straight -- ranks the clearest "this is a track, not a shower" cases
    # first.  Segments with no valid dQ/dx (xmip==0) are ranked by length and
    # straightness alone, after every segment with real charge evidence.
    straight_bonus = 1.0 if row["straight_long"] else 0.3
    if row["xmip"] <= 0:
        return row["length"] * straight_bonus * 0.5
    mip_closeness = max(0.0, 1.0 - abs(row["xmip"] - 1.0))
    return row["length"] * mip_closeness * straight_bonus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="PR out_root dir(s)")
    ap.add_argument("--min-len", type=float, default=20.0, help="minimum segment length, cm (default 20)")
    ap.add_argument("--top", type=int, default=0, help="cap ranked output to top N (0 = no cap)")
    ap.add_argument("--out", default=None, help="write full ranked TSV here")
    args = ap.parse_args()

    all_rows = []
    for arm in args.arms:
        rows = census_arm(arm, args.min_len)
        print(f"# {arm}: {len(rows)} candidate segments", file=sys.stderr)
        all_rows.extend(rows)

    all_rows.sort(key=flagrancy, reverse=True)

    n_ambiguous = sum(1 for r in all_rows if 1.2 <= r["xmip"] < 1.75)
    n_muonlike = sum(1 for r in all_rows if 0 < r["xmip"] < 1.2)
    n_protonlike = sum(1 for r in all_rows if r["xmip"] >= 1.75)
    n_noevidence = sum(1 for r in all_rows if r["xmip"] <= 0)
    print(
        f"# total: {len(all_rows)}  muon-like(<1.2x): {n_muonlike}  "
        f"ambiguous(1.2-1.75x): {n_ambiguous}  proton-like(>=1.75x): {n_protonlike}  "
        f"no-dQdx-evidence: {n_noevidence}",
        file=sys.stderr,
    )

    shown = all_rows if args.top <= 0 else all_rows[: args.top]
    if args.top > 0 and len(all_rows) > args.top:
        print(f"# showing top {args.top} of {len(all_rows)}; {len(all_rows) - args.top} dropped below the cut", file=sys.stderr)

    header = f"{'evt':>8} {'seg':>8} {'arm':>20} {'L(cm)':>8} {'D/L':>6} {'xMIP':>6} {'flag_shower':>11} {'straight_long':>13}"
    print(header)
    for r in shown:
        print(
            f"{r['evt']:>8} {r['seg']:>8} {r['arm']:>20} {r['length']:>8.2f} {r['ratio']:>6.3f} "
            f"{r['xmip']:>6.3f} {r['flag_shower']:>11} {str(r['straight_long']):>13}"
        )

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("evt\tseg\tarm\tlength_cm\tdirect_cm\tratio\txmip\tflag_shower\tstraight_long\n")
            for r in all_rows:
                fh.write(
                    f"{r['evt']}\t{r['seg']}\t{r['arm']}\t{r['length']:.2f}\t{r['direct']:.2f}\t"
                    f"{r['ratio']:.4f}\t{r['xmip']:.4f}\t{r['flag_shower']}\t{r['straight_long']}\n"
                )
        print(f"# wrote {args.out} ({len(all_rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Clean endpoint A/B for the ctoffset reprocess: the SAME physical anode-
touching tracks, measured before (_anodefix, ctoffset=4) and after (_ctoff,
ctoffset=-5.5), with the SAME hand-validated flash, W-plane signal stop.

Why spatial matching: re-clustering the ctoffset-shifted signal assigns new
cluster uids, so the decisions-file uids no longer map to the ctoff dump.  We
take each hand-validated (cluster, flash) from the anodefix decisions, find the
ctoff cluster whose centroid matches (the 1.5 cm drift shift is tiny vs
inter-cluster spacing), and re-measure with the SAME flash gid.  Flash gids are
verified stable across the reprocess (light chain untouched), so gid = same
physical flash.

The frame-level check already proved the SP signal shifted -19 ticks
(-1.49 cm) uniformly on both crates; this proves the reconstructed ENDPOINT
centres, i.e. it survives the nonlinear re-clustering at the threshold blobs
that define the anode end.

Usage (from this directory):
  python3 check_anode_stop_verify.py            # genuine touchers (img-end<=4.5)
"""
import argparse
import glob
import json
import os

import numpy as np

import check_anode_stop_ensemble as m

WORK = m.WORK
DEC = m.DEC


def load_clusters(tag, ev):
    adir, _ = m.find_dirs(ev, tag)
    if adir is None:
        return None
    d = json.load(open(os.path.join(adir, "calib-evt%s.json" % ev)))
    out = {}
    for c in d["clusters"]:
        P = np.column_stack([np.asarray(c["x"], float),
                             np.asarray(c["y"], float),
                             np.asarray(c["z"], float)])
        out[c["uid"]] = (c, P, P.mean(0), len(P))
    return out


def match_ctoff(caf_P, caf_ctr, caf_n, ctc):
    """Nearest ctoff cluster by (y,z) centroid (drift-shift-insensitive) with a
    compatible size; returns uid or None."""
    best, bd = None, 1e9
    for uid, (c, P, ctr, n) in ctc.items():
        dyz = np.hypot(caf_ctr[1] - ctr[1], caf_ctr[2] - ctr[2])
        if dyz > 4.0:
            continue
        if not (0.5 <= n / max(caf_n, 1) <= 2.0):
            continue
        dx = abs(caf_ctr[0] - ctr[0])
        if dx > 6.0:
            continue
        d = dyz + 0.3 * dx
        if d < bd:
            bd, best = d, uid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=4.5,
                    help="genuine-toucher gate on anodefix imaging anode-end u")
    ap.add_argument("--subs", default="boundary,crossers")
    args = ap.parse_args()

    events = sorted({os.path.basename(f)[len("calib-evt"):-len(".json")]
                     for f in glob.glob(os.path.join(
                         WORK, "039252_*_ctoff", "calib-evt*.json"))})

    pairs = {"bot": [], "top": []}   # (u_af, u_ct)
    nmatch = nmiss = 0
    for ev in events:
        ctc = load_clusters("ctoff", ev)
        afc = load_clusters("anodefix", ev)
        if ctc is None or afc is None:
            continue
        seen = set()
        for sub in args.subs.split(","):
            for r in m.read_decisions(sub, ev):
                uid, gid = r["main_cluster_uid"], r["flash_gid"]
                if (ev, uid) in seen or uid not in afc:
                    continue
                seen.add((ev, uid))
                _, Paf, ctr, n = afc[uid]
                # before
                o_af = m.trace(ev, uid, gid, "anodefix")
                if "skip" in o_af or o_af["end_u"] > args.gate \
                        or abs(o_af["dt0"]) >= 18 \
                        or not np.isfinite(o_af["u_gauss"]):
                    continue
                # match + after
                uid_ct = match_ctoff(Paf, ctr, n, ctc)
                if uid_ct is None:
                    nmiss += 1
                    continue
                o_ct = m.trace(ev, uid_ct, gid, "ctoff")
                if "skip" in o_ct or not np.isfinite(o_ct["u_gauss"]) \
                        or abs(o_ct["dt0"]) >= 18:
                    nmiss += 1
                    continue
                nmatch += 1
                pairs[o_af["side"]].append((o_af["u_gauss"], o_ct["u_gauss"]))
                print("  evt%s %s | anodefix %+5.2f -> ctoff %+5.2f  (d %+5.2f)"
                      % (ev, o_af["side"], o_af["u_gauss"], o_ct["u_gauss"],
                         o_ct["u_gauss"] - o_af["u_gauss"]))

    print("\n== matched %d tracks (%d unmatched) ==" % (nmatch, nmiss))
    print("== W-plane anode signal stop, SAME tracks, before -> after ==")
    allshift = []
    for side in ("bot", "top"):
        a = np.array(pairs[side])
        if not len(a):
            print("  %s: n=0" % side)
            continue
        shift = a[:, 1] - a[:, 0]
        allshift += list(shift)
        print("  %s: n=%2d  anodefix median %+5.2f  ->  ctoff median %+5.2f  "
              "| per-track shift median %+5.2f (MAD %.2f)"
              % (side, len(a), np.median(a[:, 0]), np.median(a[:, 1]),
                 np.median(shift), np.median(np.abs(shift - np.median(shift)))))
    if allshift:
        print("  ALL per-track shift median %+.2f cm (target -1.49; frame proof -1.49)"
              % np.median(allshift))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pr/93 round 3 -- per-shower A/B diff between two PR arms.

For every event present in BOTH arms, join showers[] of the two calib dumps
by shower id and report every shower that is pdg-11 on EITHER side and whose
(particle_id, kine_best) moved, appeared, or disappeared.  This is the
shower-level regression screen for the owner's bar: "real EM showers in
nueCC48/NCpi0 must not lose their electron label or energy".

Requires both arms run with PR_EXTRA_STAGES=pr_display (calib-pr-evt*.json).

Usage: pr93_shower_ab_diff.py <armA(before)> <armB(after)> [--sample LABEL]
                              [--tol-mev 0.5] [--out FILE.tsv]
Prints a summary; TSV rows: sample evt shower_id status pdgA pdgB E_A E_B dE
"""
import argparse
import glob
import json
import os


def evtid(path):
    b = os.path.basename(path)
    return b[len("calib-pr-evt"):-len(".json")]


def showers_of(arm):
    out = {}
    for p in sorted(glob.glob(os.path.join(arm, "pr_evt*", "calib-pr-evt*.json"))):
        j = json.load(open(p))
        out[evtid(p)] = {sh["id"]: sh for sh in j.get("showers", [])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armA")
    ap.add_argument("armB")
    ap.add_argument("--sample", default=None)
    ap.add_argument("--tol-mev", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sample = args.sample or os.path.basename(args.armB.rstrip("/"))

    A = showers_of(args.armA)
    B = showers_of(args.armB)
    events = sorted(set(A) & set(B))
    missing = sorted(set(A) ^ set(B))
    if missing:
        print(f"WARNING: {len(missing)} events not in both arms: {missing}")

    rows = []
    for e in events:
        sa, sb = A[e], B[e]
        for sid in sorted(set(sa) | set(sb)):
            a, b = sa.get(sid), sb.get(sid)
            pa = a["particle_id"] if a else None
            pb = b["particle_id"] if b else None
            if 11 not in (abs(pa) if pa is not None else 0,
                          abs(pb) if pb is not None else 0):
                continue
            ea = a["kine_best"] if a else None
            eb = b["kine_best"] if b else None
            if a and b:
                if pa == pb and abs(ea - eb) <= args.tol_mev:
                    continue
                status = "changed"
            else:
                status = "only_A" if a else "only_B"
            rows.append(dict(sample=sample, evt=e, shower_id=sid, status=status,
                             pdgA=pa, pdgB=pb,
                             E_A=round(ea, 1) if ea is not None else "",
                             E_B=round(eb, 1) if eb is not None else "",
                             dE=round(eb - ea, 1) if (a and b) else ""))

    cols = ["sample", "evt", "shower_id", "status", "pdgA", "pdgB", "E_A", "E_B", "dE"]
    lines = ["\t".join(cols)] + ["\t".join(str(r[c]) for c in cols) for r in rows]
    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    n_evt = len({r["evt"] for r in rows})
    print(f"# {len(rows)} moved/appeared/disappeared pdg-11 shower rows "
          f"across {n_evt}/{len(events)} events (tol {args.tol_mev} MeV)")


if __name__ == "__main__":
    main()

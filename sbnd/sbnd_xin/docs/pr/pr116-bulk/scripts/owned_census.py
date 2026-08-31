#!/usr/bin/env python3
"""Per-event recognition census between two arms: how much charge ends up
inside ANY reconstructed shower, and what the leading EM shower is worth.

The scorer only sees the 55 events that carry hand marks.  This sees all 141,
and it measures the thing a shower-clustering change can break without any mark
noticing: segments that stop belonging to any shower at all.

    ./owned_census.py BASE_ARM_GLOB NEW_ARM_GLOB
    ./owned_census.py 'work-em114c-knobsoff-*' 'work-em114c-prodnowdbg-*'
"""
import glob, json, os, sys


def arm(pat):
    out = {}
    for d in glob.glob(pat):
        for p in glob.glob(os.path.join(d, "pr_evt*", "calib-pr-evt*.json")):
            ev = os.path.basename(p).replace("calib-pr-evt", "").replace(".json", "")
            try:
                j = json.load(open(p))
            except Exception:
                continue
            seg = j.get("segments") or []
            sh = j.get("showers") or []
            owned = sum(1 for s in seg if (s.get("shower_id") or -1) != -1)
            lead = max((s.get("kine_charge") or 0.0) for s in sh) if sh else 0.0
            out[ev] = (len(sh), owned, len(seg), lead)
    return out


def main():
    A, B = arm(sys.argv[1]), arm(sys.argv[2])
    common = sorted(set(A) & set(B), key=int)
    print("events compared: %d" % len(common))
    rows = []
    for ev in common:
        (sa, oa, na, la), (sb, ob, nb, lb) = A[ev], B[ev]
        if (sa, oa, round(la, 3)) != (sb, ob, round(lb, 3)):
            rows.append((lb - la, ev, sa, sb, oa, ob, na, la, lb))
    rows.sort()
    print("%-10s %7s %7s %11s %17s" % ("event", "showers", "owned", "leading MeV", "delta"))
    for d, ev, sa, sb, oa, ob, na, la, lb in rows:
        print("%-10s %3d->%-3d %3d->%-3d/%-3d %6.1f->%-6.1f %+8.1f" %
              (ev, sa, sb, oa, ob, na, la, lb, d))
    los = [r for r in rows if r[0] < -0.05]
    gai = [r for r in rows if r[0] > 0.05]
    print("\nevents changed: %d of %d" % (len(rows), len(common)))
    print("  leading-shower energy LOST : %d event(s), Sum %.1f MeV" % (len(los), sum(-r[0] for r in los)))
    print("  leading-shower energy GAINED: %d event(s), Sum %.1f MeV" % (len(gai), sum(r[0] for r in gai)))
    do = sum(B[e][1] - A[e][1] for e in common)
    print("  segments owned by SOME shower: %d -> %d  (net %+d)"
          % (sum(A[e][1] for e in common), sum(B[e][1] for e in common), do))


if __name__ == "__main__":
    main()

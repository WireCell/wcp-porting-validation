#!/usr/bin/env python3
"""doc pr/130 item B -- is dvtx_start's 44->47 cm boundary a real population
gap, or an accident of nine points?

Part 5 measured the vertex-relative features on the EIGHT candidates
stem_backfill_back_guard actually fires on, and found dvtx_start separates the
owner's verdicts 8/8 with a 2.50 cm margin -- but rejected it, because for 7 of
those 8 dvtx_start is numerically identical to dist_cm (already measured
non-separable), so the whole separation rests on one event (292643).

Part 5's own deferred check was: the P120_STEM census tapes EVERY chain
candidate, not just the ones the guard declines.  Run it over all 239 events
and look at the distribution.  If backward candidates are smeared uniformly
across dvtx_start then 44->47 is an accident; if there is a real gap there, the
boundary is worth a knob.

Reads census arms only.  No knob, no C++, nothing shipped.

Repro:
  ./scripts/pr130_arms.sh 98 vtxcen 1 ; ./scripts/pr130_arms.sh 141 vtxcen141 1
  ./scripts/pr130_vtx_census.py
"""
import collections
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
ARMS = ["work-pr130r1-vtxcen-*", "work-pr130r1-vtxcen141-*"]
BACK_ANG = 110.0          # the shipped stem_backfill_back_ang

# The complete labelled set: owner verdicts from doc pr/130 Part 4.
# True  = the owner wants the absorb (the guard is WRONG to decline)
# False = the owner is happy with the decline
VERDICT = {292643: True, 179369: True,
           283515: False, 67394: False, 286655: False,
           347824: False, 281567: False, 47212: False}

FIELDS = ("shower_start_seg", "conn", "seg", "pdg", "len_cm", "ratio", "ok",
          "ang15", "ang60", "dist_cm", "dvtx_start_cm", "dvtx_stem_cm",
          "toward_cm", "vang")
RE_KV = re.compile(r"(\w+)=(-?[\d.]+)")


def rows():
    for pat in ARMS:
        for arm in sorted(glob.glob(os.path.join(SX, pat))):
            for lg in sorted(glob.glob(os.path.join(arm, "pr_evt*", "stdout.log"))):
                ev = int(re.search(r"pr_evt(\d+)", lg).group(1))
                for ln in open(lg, errors="replace"):
                    if "P120_STEM" not in ln:
                        continue
                    d = dict(RE_KV.findall(ln))
                    if "dvtx_start_cm" not in d:
                        continue
                    r = {k: float(d[k]) for k in FIELDS if k in d}
                    r["event"] = ev
                    yield r


def hist(vals, lo=0.0, hi=120.0, w=10.0):
    b = collections.Counter()
    for v in vals:
        b[min(int((v - lo) // w), int((hi - lo) // w))] += 1
    out = []
    for i in range(int((hi - lo) // w) + 1):
        n = b.get(i, 0)
        lab = "%3.0f-%3.0f" % (lo + i * w, lo + (i + 1) * w)
        out.append("    %s  %-3d %s" % (lab, n, "#" * min(n, 60)))
    return "\n".join(out)


def main():
    allr = list(rows())
    if not allr:
        sys.exit("no P120_STEM rows -- run the census arms first (PROBES=1)")
    evs = {r["event"] for r in allr}
    print("P120_STEM census: %d candidate rows over %d events" % (len(allr), len(evs)))

    # The guard's own firing condition: measurable angle beyond the cut.
    fires = [r for r in allr if r.get("ang15", -1) >= 0 and r["ang15"] > BACK_ANG]
    print("\n=== the guard's firing set (ang15 measurable and > %.0f deg) ===" % BACK_ANG)
    print("  %d candidates over %d events" % (len(fires), len({r['event'] for r in fires})))
    print("  %8s %8s %9s %9s %9s %9s  %s"
          % ("event", "seg", "ang15", "dist_cm", "dvtx_st", "dvtx_stem", "owner verdict"))
    for r in sorted(fires, key=lambda r: -r["dvtx_start_cm"]):
        v = VERDICT.get(r["event"])
        lab = {True: "ABSORB WANTED", False: "decline ok", None: "(unlabelled)"}[v]
        print("  %8d %8.0f %9.2f %9.2f %9.2f %9.2f  %s"
              % (r["event"], r["seg"], r["ang15"], r["dist_cm"],
                 r["dvtx_start_cm"], r["dvtx_stem_cm"], lab))

    # The question: is 44->47 a gap in the WHOLE candidate population?
    meas = [r for r in allr if r.get("dvtx_start_cm", -1) >= 0]
    acc = [r for r in meas if r.get("ok", 0) == 1]
    dec = [r for r in meas if r.get("ok", 0) == 0]
    print("\n=== dvtx_start over ALL chain candidates (n=%d) ===" % len(meas))
    print(hist([r["dvtx_start_cm"] for r in meas]))
    print("\n  accepted (ok=1, n=%d):" % len(acc))
    print(hist([r["dvtx_start_cm"] for r in acc]))
    print("\n  declined (ok=0, n=%d):" % len(dec))
    print(hist([r["dvtx_start_cm"] for r in dec]))

    # Is the 44.34 -> 46.84 interval actually empty?
    band = sorted(r["dvtx_start_cm"] for r in meas if 40.0 <= r["dvtx_start_cm"] <= 52.0)
    print("\n=== the decisive check: occupancy of the 40-52 cm band ===")
    print("  %d candidate(s) in [40, 52] cm: %s"
          % (len(band), ", ".join("%.2f" % v for v in band)))
    inside = [v for v in band if 44.34 < v < 46.84]
    print("  strictly inside the claimed gap (44.34, 46.84): %d  %s"
          % (len(inside), ", ".join("%.2f" % v for v in inside) or "-- EMPTY"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

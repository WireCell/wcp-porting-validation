#!/usr/bin/env python3
"""doc sbnd_xin/119 round 3: what the projection-constant hoist costs, paired per event.

Fork by duplication (M10) of scripts/d119/lever_cost.py, which stays as rounds 1-2's record.
ONE thing differs, and it is the whole reason this is a separate script: **the baseline**.

lever_cost.py pairs every lever against the `ctl` arm, which is production as it ran at round 1
-- glibc allocator, no proj_pad.  Round 2 then flipped proj_pad into production, so `ctl` is no
longer what production runs.  Pairing round 3 against `ctl` would fold the pad delta into the
round-3 number and overstate it.  Round 3's baseline is `flip`: the SAME configuration round 3
runs, on the PRE-round-3 binary.  The pair differs in the binary and in nothing else.

  r3  vs flip   the claim.
  r3b vs r3     the NULL PAIR on the new binary -- two runs of one configuration.  A round-3
                delta smaller than this column is not a measurement.  Round 2's floor
                (+0.2 / -1.6 / -2.6 %) was measured on the OLD binary and is NOT inherited
                across a rebuild, which is why r3b exists at all.

WHICH NUMBER TO QUOTE -- unchanged from lever_cost.py, restated because it decides the verdict:
  TICK total   the clustering node's own compute, summed over stages.  Survives batch contention,
               so this is what the claim is made on.
  maxrss       getrusage(RUSAGE_CHILDREN) high-water.  Round 3 is a CPU round; this column is
               here to show memory did not regress, not to claim a memory win.
  wall_s       context ONLY, and only between arms run at the same JOBS on the same box.

Usage:  python3 scripts/d119/r3_cost.py [cv nuecc off]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_rank import scan                                                        # noqa: E402

ARMS = {"cv": "work-r3cv-d119%s", "nuecc": "work-r3nue-d119%s", "off": "work-r3off-d119%s"}
PAIRS = [("flip", "r3", "ROUND 3 CLAIM: same configuration, rebuilt libWireCellClus "
                        "(projection-constant hoist + BFS container)"),
         ("r3", "r3b", "NULL PAIR on the new binary: one configuration run twice -- the floor. "
                       "Read this FIRST.")]


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main():
    samples = sys.argv[1:] or ["nuecc", "cv", "off"]
    print("# doc sbnd_xin/119 round 3 -- cost against the `flip` arm (production as it runs today)")
    print("# claim instrument = TICK total (in-job compute); wall is context only\n")
    for base, lever, label in PAIRS:
        print(f"===== {lever} vs {base}: {label}")
        for s in samples:
            A = scan(ARMS[s] % base)
            B = scan(ARMS[s] % lever)
            common = sorted(set(A) & set(B))
            if not common:
                print(f"  {s:6s} NO PAIRED EVENTS ({base} {len(A)}, {lever} {len(B)})")
                continue
            rows = []
            for idx, name in ((1, "TICK total  s"), (4, "maxrss   GiB"), (3, "wall      s")):
                a = [A[k][idx] for k in common if A[k][idx] is not None]
                b = [B[k][idx] for k in common if B[k][idx] is not None]
                if not a or not b:
                    continue
                ma, mb = sum(a) / len(a), sum(b) / len(b)
                rows.append((name, ma, mb, 100 * (mb - ma) / ma if ma else float("nan"),
                             pct(a, 0.5), pct(b, 0.5), max(a), max(b)))
            ta = sum(A[k][1] for k in common if A[k][1] is not None)
            tb = sum(B[k][1] for k in common if B[k][1] is not None)
            print(f"  {s} ({len(common)} paired events)   arm TICK sum "
                  f"{ta:8.1f} -> {tb:8.1f} s  ({100*(tb-ta)/ta:+.1f} %)")
            for name, ma, mb, d, a50, b50, amx, bmx in rows:
                print(f"     {name:13s} mean {ma:8.3f} -> {mb:8.3f} ({d:+6.1f} %)   "
                      f"p50 {a50:7.3f} -> {b50:7.3f}   max {amx:8.3f} -> {bmx:8.3f}")
        print()


if __name__ == "__main__":
    main()

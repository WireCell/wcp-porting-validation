#!/usr/bin/env python3
"""doc sbnd_xin/119 rounds 1-2: what each lever costs, paired per event against the control arm.

Reuses scripts/d119/perf_rank.py's parsers rather than re-implementing them, so the lever numbers
and the round-0 ranking come from exactly one reading of the TICK/MEM ladder and .time.meta.

WHICH NUMBER TO QUOTE.  Three instruments, and they do not measure the same thing:

  TICK total   the clustering node's own compute, summed over stages.  Survives batch contention,
               so this is what a lever claim is made on.
  maxrss       getrusage(RUSAGE_CHILDREN) high-water over the whole per-event step -- a kernel
               number, not a sampler, so it does not under-report the tail.
  wall_s       wraps untar + the job + tar, at whatever PR_JOBS the arm ran.  Quoted for context
               ONLY, and only between arms run at the same JOBS on the same box.  Doc 116 sec 14
               measured this term at up to 2.5x on a byte-identical configuration.

All three d119 arms (ctl, tcm, pad) were produced back to back at JOBS=8 on an otherwise idle
machine, so wall is comparable here -- but the claim is still on TICK.

Usage:  python3 scripts/d119/lever_cost.py [cv nuecc off]
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_rank import scan                                                        # noqa: E402

ARMS = {"cv": "work-r3cv-d119%s", "nuecc": "work-r3nue-d119%s", "off": "work-r3off-d119%s"}
LEVERS = [("ctl2", "NULL PAIR: the control configuration run a second time -- the noise floor. "
                   "Read this FIRST; a lever delta smaller than this column is not a measurement."),
          ("tcm", "round 1: tcmalloc joined to the PR chain's LD_PRELOAD"),
          ("pad", "round 2: proj_pad_wire 3 / proj_pad_time 3 in the fit JSON")]


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main():
    samples = sys.argv[1:] or ["nuecc", "cv", "off"]
    print("# doc sbnd_xin/119 -- lever cost against the ctl arm (production as it runs today)")
    print("# claim instrument = TICK total (in-job compute); wall is context only\n")
    for lever, label in LEVERS:
        print(f"===== {lever}: {label}")
        for s in samples:
            A = scan(ARMS[s] % "ctl")
            B = scan(ARMS[s] % lever)
            common = sorted(set(A) & set(B))
            if not common:
                print(f"  {s:6s} NO PAIRED EVENTS (ctl {len(A)}, {lever} {len(B)})")
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
            # Sum over the arm is the honest aggregate for a CPU claim: a mean of per-event
            # means hides that the cost lives in a tail.
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

#!/usr/bin/env python3
"""doc sbnd_xin/117 sec 6: what each vertex chain costs, on the same instrument as doc 116 sec 14.

Imports scripts/d116/perf_cost.py (unmodified) for the two instruments -- the job's own TICK / MEM
ladders and .time.meta's getrusage high-water -- and points them at this round's arms.

Two caveats, both stated in the doc:
  * the d117 arms ran under a varying worker count (14 -> 8), so the TICK totals carry the same
    +-10 % environment term doc 116 sec 14 measured with its s0rep control; the STAGE-LEVEL split
    is what carries the signal, because the dual pass lives entirely inside TaggerCheckNeutrino:pr;
  * the exact cost of the second pass does not depend on any of that: the c2 / t2 arms record it
    per event as vertex_scoreboard.dual_chain.off_ms (scripts/d117/dual_chain_census.py).

Usage: python3 scripts/d117/cost.py [cv nuecc off] > docs/117_figs/117_cost.txt
"""
import os
import statistics as st
import sys

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(SX, "scripts", "d116"))
import perf_cost as P  # noqa: E402

ARMS = {  # label -> {sample: arm dir}
    "c2": {"cv": "work-r3cv-d115pr", "nuecc": "work-r3nue-d115pr", "off": "work-r3off-d115pr"},
    "t2": {"cv": "work-r3cv-d116tfull", "nuecc": "work-r3nue-d116tfull", "off": "work-r3off-d116tfull"},
}
for c in ("c1e", "c1x", "t1e", "t1x"):
    ARMS[c] = {"cv": f"work-r3cv-d117{c}", "nuecc": f"work-r3nue-d117{c}", "off": f"work-r3off-d117{c}"}

STAGES = ["TaggerCheckNeutrino:pr", "CreateSteinerGraph:pr", "CreateSteinerGraph:prrefresh",
          "UbooneNueBDTScorer:pr", "UbooneNumuBDTScorer:pr"]


def main(samples):
    print("# doc 117 sec 6 -- cost per vertex chain (doc 116 sec 14's instruments, scripts/d116/perf_cost.py)")
    print("# TICK = the wire-cell job's own ladder; RSS = getrusage(CHILDREN) high-water from .time.meta.")
    for s in samples:
        print(f"\n## {s}")
        print("| arm | n | TICK mean (s) | vs c2 | TICK sum (core-h) | TaggerCheckNeutrino mean (s) | RSS p50 | RSS p99 | RSS max |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        ref = None
        for lab, per in ARMS.items():
            arm = per.get(s)
            if not arm or not os.path.isdir(os.path.join(SX, arm)):
                continue
            A = P.load(arm)
            t = [v["tick"] for v in A.values() if v.get("tick")]
            if not t:
                continue
            r = [v["rss"] for v in A.values() if "rss" in v]
            tcn = [v.get("stg", {}).get("TaggerCheckNeutrino:pr", 0.0) for v in A.values() if v.get("tick")]
            m = st.mean(t)
            if lab == "c2":
                ref = m
            print(f"| {lab} | {len(t)} | {m:.2f} | {m/ref:.3f} | {sum(t)/3600:.2f} | {st.mean(tcn):.2f} | "
                  f"{P.pct(r, .5):.2f} | {P.pct(r, .99):.2f} | {max(r):.2f} |")


if __name__ == "__main__":
    main([a for a in sys.argv[1:]] or ["cv", "nuecc", "off"])

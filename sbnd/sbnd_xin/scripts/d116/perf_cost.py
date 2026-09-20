#!/usr/bin/env python3
"""doc sbnd_xin/116 sec 14: what each study cell costs in CPU and memory, paired per event.

Two instruments, already on disk for every arm of doc 115 and doc 116 -- nothing is re-run:

  .time.meta          written by abtest/timecmd.py around the whole per-event PR step
                      (untar of the stage-A pctree, the wire-cell job, tar of the outputs):
                      wall_s, and maxrss_kb = getrusage(RUSAGE_CHILDREN).ru_maxrss, a kernel
                      high-water mark over the job's descendants -- NOT the 2 s VmHWM sampler
                      of run_pr_evt.sh, so it does not under-report the tail.
  wct_pr_evt*.log     the wire-cell job's own `TICK: <total> ms (this: <dt> ms) <stage>` ladder
                      and its `MEM: ... res=...K <stage>` ladder, i.e. the compute alone.

wall_s is I/O-contention dominated here: the doc-116 arms ran 3-8 lock-sharing workers per cell
while the doc-115 baseline arms ran one driver, and `s0rep` -- byte-identically the baseline
configuration -- is 2.5x the baseline's wall on cv.  So `s0rep` is the CONTROL: every cost ratio
in sec 14 is quoted against it, not against the baseline, and the baseline/s0rep ratio is printed
as the size of the environment term.  The TICK ladder is the instrument that survives this
(s0rep within 10 % of the baseline on all three samples).

Usage:  python3 scripts/d116/perf_cost.py [cv nuecc off] > docs/116_figs/116_cost.txt
"""
import glob
import os
import re
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAMPLES = {"cv": ("work-r3cv-d115pr", "work-r3cv-d116%s"),
           "nuecc": ("work-r3nue-d115pr", "work-r3nue-d116%s"),
           "off": ("work-r3off-d115pr", "work-r3off-d116%s")}
CELLS = ["s0rep", "cs", "p3bw", "csp3bw", "tfull"]
STAGES = ["loaded live", "CreateSteinerGraph:pr", "CreateSteinerGraph:prrefresh",
          "TaggerCheckNeutrino:pr", "TaggerCheckSTM:pr", "TaggerCheckTGM:pr",
          "UbooneNumuBDTScorer:pr", "UbooneNueBDTScorer:pr"]
TICK = re.compile(r"^TICK: (\d+) ms \(this: (\d+) ms\) (.*)$")
MEM = re.compile(r"^MEM: total: size=\S+ res=([0-9.e+]+)K")
GIB = 1048576.0


def one_log(p):
    """(per-stage seconds, TICK total s, MEM-ladder peak res GiB) from the tail of one job log."""
    sz = os.path.getsize(p)
    with open(p, errors="replace") as fh:
        fh.seek(max(0, sz - 120000))
        txt = fh.read()
    stg, tot, mem = {}, None, 0.0
    for line in txt.splitlines():
        m = TICK.match(line)
        if m:
            stg[m.group(3)] = stg.get(m.group(3), 0.0) + int(m.group(2)) / 1000.0
            tot = int(m.group(1)) / 1000.0
            continue
        m = MEM.match(line)
        if m:
            mem = max(mem, float(m.group(1)) / GIB)
    return stg, tot, mem


def load(arm):
    """key (sub-root/pr_evt dir, identical across arms) -> dict of the measurements."""
    root = os.path.join(SX, arm)
    logs = sorted(glob.glob(os.path.join(root, "f*", "pr_evt*", "wct_pr_evt*.log")) +
                  glob.glob(os.path.join(root, "pr_evt*", "wct_pr_evt*.log")))
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one_log, logs))
    out = {}
    for p, (stg, tot, mem) in zip(logs, res):
        out[os.path.relpath(os.path.dirname(p), root)] = {"stg": stg, "tick": tot, "memlad": mem}
    for p in (glob.glob(os.path.join(root, "f*", "pr_evt*", ".time.meta")) +
              glob.glob(os.path.join(root, "pr_evt*", ".time.meta"))):
        d = dict(l.split("=", 1) for l in open(p).read().split())
        e = out.setdefault(os.path.relpath(os.path.dirname(p), root), {})
        e["wall"] = float(d["wall_s"])
        e["rss"] = int(d["maxrss_kb"]) / GIB
    return out


def pct(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))]


def main(samples):
    print("# doc 116 sec 14 -- cost of each cell, from the per-event records already on disk")
    print("# wall = .time.meta (whole PR step, I/O contention included); tick = the wire-cell job's")
    print("# own TICK ladder; rss = getrusage CHILDREN high-water; memlad = the job's MEM ladder max.")
    for s in samples:
        base, tmpl = SAMPLES[s]
        arms = [("baseline", load(base))] + [(c, load(tmpl % c)) for c in CELLS]
        arms = [(n, a) for n, a in arms if a]
        B = dict(arms)["baseline"]
        tagged = {k for k, v in B.items() if v.get("stg", {}).get("TaggerCheckNeutrino:pr", 0) > 0.05}
        print(f"\n## {s}: {len(B)} events, {len(tagged)} of them run the neutrino tagger > 50 ms")
        s0 = dict(arms).get("s0rep")
        ref = st.mean([v["tick"] for v in s0.values() if v.get("tick")]) if s0 else None
        print("| arm | wall mean | tick mean (s) | tick p99 | tick sum (core-h) | tick mean / s0rep "
              "| tick mean, tagger subset | RSS p50 | RSS p99 | RSS max | RSS > 1.5 / > 2.0 GiB | MEM-ladder max |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, A in arms:
            t = [v["tick"] for v in A.values() if v.get("tick")]
            tt = [A[k]["tick"] for k in tagged if k in A and A[k].get("tick")]
            w = [v["wall"] for v in A.values() if "wall" in v]
            r = [v["rss"] for v in A.values() if "rss" in v]
            ml = [v["memlad"] for v in A.values() if v.get("memlad")]
            print(f"| {name} | {st.mean(w):.1f} s | {st.mean(t):.2f} | {pct(t, .99):.1f} | {sum(t)/3600:.2f} | "
                  f"{st.mean(t)/ref:.3f} | {st.mean(tt):.2f} | {pct(r, .5):.2f} | {pct(r, .99):.2f} | {max(r):.2f} | "
                  f"{sum(1 for x in r if x > 1.5)} / {sum(1 for x in r if x > 2.0)} | {max(ml):.2f} |")
        print(f"\n### {s}: where the time goes -- per-stage seconds, mean (median) over all events")
        print("| stage | " + " | ".join(n for n, _ in arms) + " |")
        print("|---" * (len(arms) + 1) + "|")
        for stg in STAGES:
            cells = []
            for _, A in arms:
                v = [x.get("stg", {}).get(stg, 0.0) for x in A.values() if x.get("tick")]
                cells.append(f"{st.mean(v):.2f} ({st.median(v):.2f})")
            print(f"| {stg} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if a in SAMPLES] or list(SAMPLES))

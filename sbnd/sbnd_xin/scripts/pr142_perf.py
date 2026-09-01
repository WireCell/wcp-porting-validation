#!/usr/bin/env python3
"""doc pr/142 -- stage-resolved runtime and memory profile of a PR arm.

The per-event score table (pr_scores_table.py) gives the EVENT totals: wall_s /
core_s from the job's own "Timer: Total" line and maxrss_kb from timecmd.py's
RUSAGE_CHILDREN.  This goes one level down, into what every PR log already
carries and nothing has ever read at population scale:

  "MABC timing: <step> took X ms (cumulative Y ms)"   per clustering step
  "MEM: total: size=..K, res=..K increment: size=..K, res=..K <step>"
  "Timer: N wall-sec, M core-sec: (Component) \"name\""  per graph component
  "Timer: Total N wall-sec, M core-sec"

Peak RSS from the MEM lines is the in-process high-water mark and is therefore
concurrency-insensitive, the same property doc pr/11 sec 405 relies on for
timecmd's number.  Wall is NOT: under PR_JOBS-way concurrency it is contention-
dominated (doc pr/11 sec 90, doc 76 sec 1.1), so CORE is the comparable number
and wall is reported beside it, never instead of it.

Usage:
  pr142_perf.py <arm> [<arm> ...] [--top 20] [--tsv out.tsv] [--jobs 8]
"""
import argparse
import glob
import os
import re
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

RE_MABC = re.compile(r"MABC timing: (.+?) took ([\d.]+) ms \(cumulative ([\d.]+) ms\)")
RE_MEM = re.compile(r"^MEM: total: size=([\d.eE+-]+)K, res=([\d.eE+-]+)K "
                    r"increment: size=([\d.eE+-]+)K, res=([\d.eE+-]+)K (.*)$")
RE_TIMER = re.compile(r"Timer: ([\d.]+) wall-sec, ([\d.]+) core-sec:\s+\((.+?)\) \"(.+?)\"")
RE_TOTAL = re.compile(r"Timer: Total ([\d.]+) wall-sec, ([\d.]+) core-sec")


def scan(path):
    steps, mem, comps, total = {}, {}, {}, None
    peak = 0.0
    with open(path, errors="ignore") as f:
        for ln in f:
            m = RE_MABC.search(ln)
            if m:
                steps[m.group(1)] = steps.get(m.group(1), 0.0) + float(m.group(2))
                continue
            m = RE_MEM.match(ln)
            if m:
                res, inc, name = float(m.group(2)), float(m.group(4)), m.group(5).strip()
                peak = max(peak, res)
                a, b = mem.get(name, (0.0, 0.0))
                mem[name] = (max(a, res), b + inc)
                continue
            m = RE_TIMER.search(ln)
            if m:
                key = f'{m.group(4)} ({m.group(3).split("::")[-1]})'
                w, c = comps.get(key, (0.0, 0.0))
                comps[key] = (w + float(m.group(1)), c + float(m.group(2)))
                continue
            m = RE_TOTAL.search(ln)
            if m:
                total = (float(m.group(1)), float(m.group(2)))
    return steps, mem, comps, total, peak


def profile(arm, jobs):
    logs = sorted(glob.glob(os.path.join(arm, "pr_evt*", "wct_pr_evt*.log")))
    if not logs:
        return None
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(scan, logs))
    agg_steps, agg_mem, agg_comps = {}, {}, {}
    walls, cores, peaks = [], [], []
    for steps, mem, comps, total, peak in res:
        for k, v in steps.items():
            agg_steps.setdefault(k, []).append(v)
        for k, (r, i) in mem.items():
            agg_mem.setdefault(k, []).append(r)
        for k, (w, c) in comps.items():
            agg_comps.setdefault(k, []).append(c)
        if total:
            walls.append(total[0]); cores.append(total[1])
        if peak:
            peaks.append(peak / 1048576.0)
    return dict(arm=arm, n=len(logs), steps=agg_steps, mem=agg_mem,
                comps=agg_comps, walls=walls, cores=cores, peaks=peaks)


def med(v):
    return st.median(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tsv", default=None)
    a = ap.parse_args()

    profs = [p for p in (profile(x, a.jobs) for x in a.arms) if p]
    if not profs:
        print("no logs found", file=sys.stderr); return

    print("=" * 78)
    print("EVENT TOTALS (from the job's own Timer: Total line; peak RSS from MEM lines)")
    print("=" * 78)
    for p in profs:
        w, c, k = p["walls"], p["cores"], p["peaks"]
        print(f"  {os.path.basename(p['arm']):32s} n={p['n']:5d}  "
              f"wall med={med(w):6.1f}s sum={sum(w)/3600:6.2f}h | "
              f"core med={med(c):6.1f}s sum={sum(c)/3600:6.2f}h | "
              f"peakRSS med={med(k):.2f} max={max(k) if k else float('nan'):.2f} GiB")

    print("\n" + "=" * 78)
    print(f"TOP {a.top} CLUSTERING STEPS by median per-event cost (MABC timing)")
    print("=" * 78)
    base = profs[0]
    order = sorted(base["steps"], key=lambda k: -med(base["steps"][k]))[:a.top]
    hdr = f"  {'step':44s}" + "".join(f"{os.path.basename(p['arm'])[-14:]:>16s}" for p in profs)
    print(hdr)
    for k in order:
        line = f"  {k[:44]:44s}"
        for p in profs:
            line += f"{med(p['steps'].get(k, [])):13.1f} ms"
        print(line)

    print("\n" + "=" * 78)
    print("GRAPH COMPONENTS by median per-event core-sec")
    print("=" * 78)
    order = sorted(base["comps"], key=lambda k: -med(base["comps"][k]))
    for k in order:
        line = f"  {k[:44]:44s}"
        for p in profs:
            line += f"{med(p['comps'].get(k, [])):13.2f} s "
        print(line)

    print("\n" + "=" * 78)
    print(f"TOP {a.top} STEPS by median resident set at that step (MEM res=)")
    print("=" * 78)
    order = sorted(base["mem"], key=lambda k: -med(base["mem"][k]))[:a.top]
    for k in order:
        line = f"  {k[:44]:44s}"
        for p in profs:
            v = med(p["mem"].get(k, []))
            line += f"{v/1048576.0:12.3f} GiB"
        print(line)

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("kind\tname\t" + "\t".join(os.path.basename(p["arm"]) for p in profs) + "\n")
            for kind, key, scale, unit in (("step_ms", "steps", 1.0, ""),
                                           ("comp_core_s", "comps", 1.0, ""),
                                           ("mem_gib", "mem", 1 / 1048576.0, "")):
                names = sorted({n for p in profs for n in p[key]})
                for n in names:
                    f.write(f"{kind}\t{n}\t" +
                            "\t".join(f"{med(p[key].get(n, []))*scale:.4f}" for p in profs) + "\n")
            for kind, key in (("event_wall_s", "walls"), ("event_core_s", "cores"),
                              ("event_peak_rss_gib", "peaks")):
                f.write(f"{kind}\tmedian\t" +
                        "\t".join(f"{med(p[key]):.4f}" for p in profs) + "\n")
        print(f"\n# wrote {a.tsv}")


if __name__ == "__main__":
    main()

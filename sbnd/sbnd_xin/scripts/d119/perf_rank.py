#!/usr/bin/env python3
"""doc sbnd_xin/119 round 0: rank EVERY stage of the SBND PR job by post-flip cost, and by how
much the doc-118 flip moved it.

Forked by duplication from scripts/d116/perf_cost.py, which stays untouched (it is doc 116's
record, M10/M13).  Two things differ, both of them the reason round 0 needs a new script:

  * perf_cost.py carries a FIXED 8-entry STAGES list.  A ranking that can only see 8 of the 15
    stages cannot say "nothing else is big" -- it can only confirm what was already suspected.
    This script discovers the stage set from the logs and ranks all of it, so a target that
    nobody named still shows up.
  * perf_cost.py reports per-cell cost ratios.  Round 0 wants the PAIRED per-stage delta between
    exactly two arms (the pre-flip control and what production now runs) plus the tail, because
    the deliverable is a ranked target list, not a cost verdict.

Instruments (already on disk for every arm -- nothing is re-run):
  wct_pr_evt*.log   the job's own `TICK: <total> ms (this: <dt> ms) <stage>` ladder, and the
                    `MEM: ... res=<N>K <stage>` ladder emitted once at destruction.
  .time.meta        abtest/timecmd.py: wall_s and maxrss_kb (getrusage CHILDREN high-water).

WHY THE CONTROL IS s0rep AND NOT THE BASELINE (doc 116 sec 14, repeated here because it decides
how every number below must be read): the doc-116 arms ran 3-8 lock-sharing workers per cell
while the doc-115 baseline arms ran one driver.  `s0rep` is byte-identically the baseline
CONFIGURATION run inside the doc-116 environment, and its wall is up to 2.5x the baseline's.  So
wall_s carries an environment term that has nothing to do with the flip.  The TICK ladder is the
instrument that survives it.  Quote TICK deltas; quote wall only with the arm named.

Usage:
  python3 scripts/d119/perf_rank.py <sample> <arm_before> <arm_after>
  python3 scripts/d119/perf_rank.py nuecc work-r3nue-d116s0rep work-r3nue-d116tfull
"""
import glob
import os
import re
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TICK = re.compile(r"^TICK: (\d+) ms \(this: (\d+) ms\) (.*)$")
MEM = re.compile(r"^MEM: total: size=\S+ res=([0-9.e+]+)K")
GIB = 1048576.0
# The ladder's bookkeeping entries: real work, but not stages of the tagger pipeline.  Kept in
# the table (so the columns still sum to the job) and marked, never silently dropped.
BOOKKEEPING = {"starting MultiAlgBlobClustering", "loaded live", "loaded dead",
               "dump dead regions to bee", "pre clustering", "start clustering",
               "dump live clusters to bee", "done"}


def one_log(path):
    """(dict stage->seconds, TICK total s, MEM peak res GiB) from one job log.

    Only the TAIL matters -- the ladder is emitted once at destruction -- but a job that was
    killed has the inline `MABC timing:` lines and NO ladder, so a missing ladder must return
    None rather than a zero that would silently drag a median down.
    """
    try:
        sz = os.path.getsize(path)
        with open(path, "rb") as fh:
            if sz > 400000:
                fh.seek(sz - 400000)
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    stages, total, peak = {}, None, 0.0
    for line in tail.splitlines():
        m = TICK.match(line)
        if m:
            tot, dt, name = int(m.group(1)), int(m.group(2)), m.group(3).strip()
            # A stage name can repeat if the ladder appears twice in a tail window; last wins.
            stages[name] = dt / 1000.0
            total = tot / 1000.0
            continue
        m = MEM.match(line)
        if m:
            peak = max(peak, float(m.group(1)) / GIB)
    if total is None or not stages:
        return None
    return stages, total, peak


def one_meta(path):
    """wall_s, maxrss_gb from a .time.meta, or (None, None)."""
    try:
        kv = dict(l.split("=", 1) for l in open(path).read().split())
    except (OSError, ValueError):
        return None, None
    try:
        return float(kv.get("wall_s", "nan")), float(kv.get("maxrss_kb", "nan")) / GIB
    except ValueError:
        return None, None


def scan(arm):
    """{event_key: (stages, total_s, ladder_peak_gib, wall_s, maxrss_gib)} for one arm.

    The key carries the sub-root (f000/...), because event ids restart in every stage-A input
    file -- keying on the basename silently pairs one file's event 25 with another's.  That
    exact defect fabricated a whole arm's worth of difference in doc 118's first gate run; see
    the note in scripts/d118/hash_gate.py.
    """
    root = os.path.join(SX, arm)
    dirs = sorted(glob.glob(os.path.join(root, "*", "pr_evt*")) +
                  glob.glob(os.path.join(root, "pr_evt*")))
    out = {}

    def work(d):
        evt = os.path.basename(d)[len("pr_evt"):]
        sub = os.path.basename(os.path.dirname(d))
        sub = "" if os.path.samefile(os.path.dirname(d), root) else sub
        logs = glob.glob(os.path.join(d, "wct_pr_evt*.log"))
        if not logs:
            return None
        got = one_log(logs[0])
        if got is None:
            return None
        wall, rss = one_meta(os.path.join(d, ".time.meta"))
        return (sub, evt), got + (wall, rss)

    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(work, dirs):
            if r:
                out[r[0]] = r[1]
    return out


def pct(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return float("nan")
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def main():
    sample, arm_a, arm_b = sys.argv[1], sys.argv[2], sys.argv[3]
    A, B = scan(arm_a), scan(arm_b)
    common = sorted(set(A) & set(B))
    print(f"# doc sbnd_xin/119 round 0 -- per-stage ranking, sample={sample}")
    print(f"# before = {arm_a}  ({len(A)} events with a ladder)")
    print(f"# after  = {arm_b}  ({len(B)} events with a ladder)")
    print(f"# paired on (sub-root, event): {len(common)} events\n")
    if not common:
        print("NO PAIRED EVENTS -- check the arm names")
        return

    names = []
    for k in common:
        for n in A[k][0]:
            if n not in names:
                names.append(n)
        for n in B[k][0]:
            if n not in names:
                names.append(n)

    rows = []
    for n in names:
        a = [A[k][0].get(n, 0.0) for k in common]
        b = [B[k][0].get(n, 0.0) for k in common]
        rows.append((n, sum(a) / len(a), sum(b) / len(b), st.median(b), pct(b, 0.9), max(b)))
    tot_a = sum(r[1] for r in rows)
    tot_b = sum(r[2] for r in rows)
    rows.sort(key=lambda r: -r[2])

    print(f"{'stage':38s} {'before/evt':>10s} {'after/evt':>10s} {'delta':>9s} "
          f"{'%job':>6s} {'p50':>7s} {'p90':>7s} {'max':>8s}")
    print("-" * 100)
    for n, ma, mb, med, p90, mx in rows:
        tag = "  (bookkeeping)" if n in BOOKKEEPING else ""
        print(f"{n[:38]:38s} {ma:9.3f}s {mb:9.3f}s {mb - ma:+8.3f}s "
              f"{100 * mb / tot_b:5.1f}% {med:6.3f}s {p90:6.3f}s {mx:7.3f}s{tag}")
    print("-" * 100)
    print(f"{'TOTAL (sum of stages)':38s} {tot_a:9.3f}s {tot_b:9.3f}s {tot_b - tot_a:+8.3f}s "
          f"({100 * (tot_b - tot_a) / tot_a:+.1f} %)")

    for label, idx, unit in (("TICK total", 1, "s"), ("ladder peak res", 2, "GiB"),
                             ("wall (env-contaminated)", 3, "s"), ("maxrss", 4, "GiB")):
        a = [A[k][idx] for k in common if A[k][idx] is not None]
        b = [B[k][idx] for k in common if B[k][idx] is not None]
        if not a or not b:
            continue
        print(f"\n{label:26s} before: mean {sum(a)/len(a):7.3f} p50 {pct(a,0.5):7.3f} "
              f"p90 {pct(a,0.9):7.3f} max {max(a):7.3f} {unit}")
        print(f"{'':26s} after : mean {sum(b)/len(b):7.3f} p50 {pct(b,0.5):7.3f} "
              f"p90 {pct(b,0.9):7.3f} max {max(b):7.3f} {unit}")

    # The memory tail is a production constraint (a per-process cap), so name the events that
    # set it rather than only quoting the quantile -- round 0's heap profiling needs the ids.
    tail = sorted(((B[k][4], k) for k in common if B[k][4] is not None), reverse=True)[:8]
    print(f"\ntop maxrss events in {arm_b}:")
    for v, (sub, evt) in tail:
        print(f"  {v:6.3f} GiB   {sub or '.'}/pr_evt{evt}")


if __name__ == "__main__":
    main()

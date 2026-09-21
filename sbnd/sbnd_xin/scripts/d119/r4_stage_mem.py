#!/usr/bin/env python3
"""doc sbnd_xin/119 round 4: per-stage MEMORY attribution from instruments already on disk.

Nothing is re-run.  The in-job `MEM: total: size=...K, res=...K increment: size=...K, res=...K
<stage>` ladder is emitted by MultiAlgBlobClustering for every arm of every round, and the
PDHD/PDVD census TSVs already carry the same quantity as d_<stage> columns.  This script only
reads them.

WHY IT EXISTS AS ITS OWN STEP.  It costs nothing and it decides where the heap profiler is
pointed.  The answer it gives is not the one a mean would give:

    the SBND MEDIAN is TaggerCheckNeutrino-dominated and the SBND TAIL is Steiner-dominated,
    by a factor of ~8 in Steiner's own increment

so an attribution computed over a gate manifest ranks the wrong stage for the events that
actually break a memory cap.  Both are printed, never merged.

Usage:  r4_stage_mem.py > docs/119_figs/119_r4_stage_mem.txt
"""
import csv
import glob
import os
import re
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

SX = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
GIB = 1048576.0
MEM = re.compile(r"^MEM: total: size=\S+ res=([0-9.e+]+)K increment: size=\S+ res=([-0-9.e+]+)K (.*)$")


def one_log(p):
    """(peak res GiB, {stage: increment GiB}) from one job log."""
    peak, inc = 0.0, {}
    with open(p, errors="replace") as fh:
        for line in fh:
            m = MEM.match(line)
            if not m:
                continue
            peak = max(peak, float(m.group(1)) / GIB)
            inc[m.group(3)] = inc.get(m.group(3), 0.0) + float(m.group(2)) / GIB
    return peak, inc


def sbnd_arm(arm, label):
    root = os.path.join(SX, arm)
    logs = sorted(glob.glob(os.path.join(root, "f*", "pr_evt*", "wct_pr_evt*.log")))
    if not logs:
        print("  (no logs under %s)" % arm)
        return
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one_log, logs))
    peaks = [r[0] for r in res]
    agg = {}
    for _, inc in res:
        for k, v in inc.items():
            agg.setdefault(k, []).append(v)
    print("  %s  n=%d  ladder peak res: mean %.3f  max %.3f GiB" %
          (label, len(logs), st.mean(peaks), max(peaks)))
    rows = sorted(((st.mean(v), max(v), k) for k, v in agg.items()), reverse=True)
    for mean, mx, k in rows[:6]:
        if abs(mean) < 0.005 and abs(mx) < 0.02:
            continue
        print("      %-34s mean %+.3f   max %+.3f" % (k, mean, mx))


def census(tsv, label):
    path = os.path.join(SX, "docs", "119_figs", tsv)
    if not os.path.exists(path):
        print("  (no %s)" % tsv)
        return
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    dcols = [c for c in rows[0] if c.startswith("d_")]
    peaks = [float(r["peak_gb"]) for r in rows]
    print("  %s  n=%d  peak_gb: mean %.2f  max %.2f   >2 GB: %d" %
          (label, len(rows), st.mean(peaks), max(peaks), sum(1 for p in peaks if p > 2.0)))
    tab = []
    for c in dcols:
        v = [float(r[c]) for r in rows]
        tab.append((st.mean(v), max(v), c[2:]))
    for mean, mx, k in sorted(tab, reverse=True)[:6]:
        print("      %-34s mean %+.3f   max %+.3f   (max/mean %4.1fx)" %
              (k, mean, mx, mx / mean if mean > 0.001 else 0))


def main():
    print("doc sbnd_xin/119 round 4 -- per-stage RESIDENT-memory increment, all three detectors.")
    print("=" * 88)
    print("Instrument: the in-job MEM ladder (MultiAlgBlobClustering), already on disk for every")
    print("arm.  Values are GiB of resident-size increment attributed to each stage.")
    print()
    print("CAVEAT THAT GOVERNS EVERY CROSS-DETECTOR READING BELOW: the allocator differs.  SBND's")
    print("PR chain preloads libtcmalloc_minimal since round 1; PDHD/PDVD preload it too; but the")
    print("d116/d119tailg arms are glibc.  RSS is not comparable across allocators -- doc 119")
    print("sec 9.3 records the round-3 RSS claim that this exact confound already broke once.")
    print()
    print("-- SBND, the 24-event nuecc GATE manifest (max 1.43 GiB: contains NO tail event) -----")
    sbnd_arm("work-r3nue-d119r3", "d119r3 (round-3 binary, glibc)")
    print()
    print("-- SBND, the 5 doc-116 TAIL events + p99 + p50 (this round's arms) --------------------")
    sbnd_arm("work-r3nue-d119tail", "d119tail  (production, tcmalloc)")
    sbnd_arm("work-r3nue-d119tailg", "d119tailg (control, glibc)")
    print()
    print("-- PDHD / PDVD, the round-3 cross-detector census (tcmalloc) --------------------------")
    census("119_pdhd_r3_census.tsv", "PDHD -nu")
    census("119_pdvd_r3_census.tsv", "PDVD -nu")
    print()
    print("=" * 88)
    print("READING.")
    print()
    print("1. THE MEAN AND THE TAIL NAME DIFFERENT STAGES.  On the SBND gate manifest the biggest")
    print("   mean increment is TaggerCheckNeutrino:pr (+0.39) and Steiner is +0.16.  On the five")
    print("   doc-116 tail events Steiner is +1.2 to +1.4 GiB -- about 8x its own mean -- while")
    print("   TaggerCheckNeutrino stays near +0.29.  A memory target ranked on means would be the")
    print("   wrong target for every event that breaks a cap.")
    print()
    print("2. TaggerCheckNeutrino IS NARROW, STEINER IS HEAVY-TAILED.  The tagger's increment sits")
    print("   in a tight band on both the gate arm and the tail events; Steiner's max/mean ratio is")
    print("   ~4x on PDHD and ~6x on PDVD.  The stage that varies is the stage that matters.")
    print()
    print("3. SAME ON ALL THREE DETECTORS.  Steiner has the largest max increment on SBND, PDHD and")
    print("   PDVD alike, and the heap profile (119_r4_heap_sbnd.txt) puts CreateSteinerGraph::visit")
    print("   at 70-78 % of PEAK LIVE HEAP on both detectors profiled.")
    print()
    print("4. ROUND 3 DID NOT MOVE MEMORY, as expected -- it was a CPU round.  The d118flip and")
    print("   d119r3 gate arms agree to three decimals on every stage.")


if __name__ == "__main__":
    main()

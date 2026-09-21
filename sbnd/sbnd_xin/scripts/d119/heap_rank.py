#!/usr/bin/env python3
"""doc sbnd_xin/119 round 4: reduce a jemalloc dump SERIES to a ranked live-heap attribution.

Why a series and not a dump.  pdhd/stm/perf/je2pprof.py converts ONE jemalloc heap_v2 dump to the
gperftools format google-pprof reads, and doc 30 used it that way because doc 30 profiled with
prof_final alone -- one dump, at exit.  Round 4 asks a question that one dump cannot answer:

    The MEM ladder shows CreateSteinerGraph:pr adding +1.2 to +1.4 GiB on the SBND tail events and
    never releasing it.  Is that memory LIVE at that moment, or was it freed and the allocator kept
    the arena?  An exit-time dump answers only if it is still live at EXIT.  A peak that is
    transient inside the stage is invisible to it -- and would read as "Steiner holds nothing".

So profile_pr119r4.sh runs with lg_prof_interval and jemalloc writes a dump every 2^N bytes of
cumulative allocation.  This script walks the whole series, prints the live-bytes trajectory (which
is the shape the question is about), and ranks call sites at the series' MAX-live dump and at the
FINAL dump separately.  (prof_gdump, the seemingly right trigger, is unusable -- it produced 29 470
dumps and 5.1 GB on a 10-second event; see profile_pr119r4.sh.)

  PEAK live  vs  ladder peak RSS  -> how much of RSS is live heap at all
  PEAK live  vs  FINAL live       -> how much of the peak is released before exit

Usage:  heap_rank.py <outdir> <evt> [--top N] [--binary /path/to/wire-cell]
"""
import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# realpath, not abspath: sbnd_xin is reached through a symlink from the toolkit tree (M9), and
# abspath would keep the symlinked prefix and walk "../.." out of the wrong directory.
HERE = os.path.dirname(os.path.realpath(__file__))
JE2PPROF = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "..", "pdhd", "stm", "perf", "je2pprof.py"))
if not os.path.exists(JE2PPROF):
    sys.exit("je2pprof.py not found at %s" % JE2PPROF)


def dump_order(path):
    """jemalloc names dumps <prefix>.<pid>.<seq>.<kind><n>.heap -- <seq> is the global order."""
    m = re.search(r"\.(\d+)\.(\d+)\.([a-z])(\d*)\.heap$", path)
    return (int(m.group(2)), m.group(3)) if m else (0, "")


def live_bytes(heapfile):
    """Total live bytes from a converted gperftools-format heap profile's header line."""
    with open(heapfile) as fh:
        head = fh.readline()
    m = re.match(r"heap profile:\s*(\d+):\s*(\d+)", head)
    return int(m.group(2)) if m else 0


def convert(src, dst):
    r = subprocess.run([sys.executable, JE2PPROF, src, dst], capture_output=True, text=True)
    if r.returncode != 0:
        print("  je2pprof FAILED on %s: %s" % (os.path.basename(src), r.stderr.strip()),
              file=sys.stderr)
        return None
    return dst


def pprof(binary, heap, top, cum=True):
    cmd = ["google-pprof", "--text", "--inuse_space"]
    if cum:
        cmd.append("--cum")
    cmd += [binary, heap]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return "  google-pprof failed: " + r.stderr.strip()[:400]
    return "\n".join(r.stdout.splitlines()[:top])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("evt")
    ap.add_argument("--top", type=int, default=45)
    ap.add_argument("--binary", default=None)
    a = ap.parse_args()

    binary = a.binary or subprocess.run(["which", "wire-cell"], capture_output=True,
                                        text=True).stdout.strip()
    pfx = os.path.join(a.outdir, "je_evt%s" % a.evt)
    dumps = sorted((p for p in os.listdir(a.outdir)
                    if p.startswith("je_evt%s." % a.evt) and p.endswith(".heap")),
                   key=dump_order)
    if not dumps:
        sys.exit("no dumps at %s.*.heap" % pfx)

    cdir = os.path.join(a.outdir, "conv_evt%s" % a.evt)
    os.makedirs(cdir, exist_ok=True)

    print("=== dump series: %d dumps  (evt %s)" % (len(dumps), a.evt))
    print("    %-34s %14s" % ("dump", "live MiB"))
    # je2pprof is a python process per dump and a long job produces hundreds; serial conversion
    # dominates the round's wall time for no reason.  Conversions are independent.
    def one(d):
        src = os.path.join(a.outdir, d)
        dst = os.path.join(cdir, d + ".pprof")
        return (d, dst) if convert(src, dst) else (d, None)

    with ThreadPoolExecutor(16) as ex:
        conv = list(ex.map(one, dumps))
    series = []
    for d, dst in conv:
        if dst is None:
            continue
        lb = live_bytes(dst)
        series.append((lb, d, dst))
        print("    %-34s %14.1f" % (d, lb / 1048576.0))

    if not series:
        sys.exit("every conversion failed")

    peak = max(series, key=lambda t: t[0])
    final = next((t for t in reversed(series) if re.search(r"\.f\d*\.heap$", t[1])), series[-1])

    print()
    print("=== peak  live %.3f GiB   %s" % (peak[0] / 1073741824.0, peak[1]))
    print("=== final live %.3f GiB   %s" % (final[0] / 1073741824.0, final[1]))
    if peak[0]:
        print("=== released between peak and exit: %.3f GiB (%.1f %% of peak live)"
              % ((peak[0] - final[0]) / 1073741824.0, 100.0 * (peak[0] - final[0]) / peak[0]))
    print("    (compare peak live against the job's MEM-ladder peak res: the GAP is allocator"
          "\n     arena + non-heap mappings, and no call-site knob reaches it)")

    for lbl, t in (("PEAK", peak), ("FINAL", final)):
        print()
        print("=" * 78)
        print("=== %s dump, --inuse_space --cum, top %d   [%s]" % (lbl, a.top, t[1]))
        print("=" * 78)
        print(pprof(binary, t[2], a.top, cum=True))
        # NO FLAT VIEW.  It is worthless here and reading it would mislead: jemalloc's sampler
        # attributes the leaf frame to its own allocation path, so --inuse_space flat reports
        # ~100 % in malloc_usable_size and then a tail of ROOT/cling JIT frames
        # (CGRecordLowering::lower, ItaniumRecordLayoutBuilder::LayoutField) that are real but
        # sub-megabyte.  The cumulative view above is the one that attributes bytes to callers.


if __name__ == "__main__":
    main()

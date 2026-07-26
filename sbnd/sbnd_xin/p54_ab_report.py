#!/usr/bin/env python3
"""Doc 54 A/B report: TGM/STM perf round 1 byte-identicality gate + timing.

GATE.  The optimized binary (wire-indexed dQ_dx_fit) must reproduce the
baseline binary's nusel products member-content-hash exact (never `cmp` on an
archive -- M2).  Both arms were produced by run_perf54_nusel.sh under
`setarch x86_64 -R` from the SAME symlinked ql_evt* inputs (work-*-d55ton),
so any diff is the code change.  Scope per event, following d52_ab_report.py:
mabc-pr.zip + pctree-pr-evt*.tar.gz (hash_archive) + nusel TSV (text) --
tracking-stm.root is compared NUMERICALLY (uproot branch arrays) because ROOT
files embed timestamps and are never raw-byte reproducible.

TIMING.  Per-event wall/RSS from the .time_*.meta files timecmd.py wrote.

Repro:
  cd sbnd_xin
  ./run_perf54_nusel.sh base     # before the code change
  ./run_perf54_nusel.sh opt      # after
  python3 p54_ab_report.py
Full write-up: docs/54_tgm-stm-perf-round1.md
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABTEST = os.path.abspath(os.path.join(HERE, "..", "..", "abtest", "hash_archive.py"))
ROOTS = ["mcp10", "mcp1000", "mcp1000b"]

AP = argparse.ArgumentParser()
AP.add_argument("--base-tag", default="p54base")
AP.add_argument("--opt-tag", default="p54opt")
A = AP.parse_args()


def hash_archive(path):
    if not os.path.isfile(path):
        return None
    try:
        out = subprocess.run([sys.executable, ABTEST, path], capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    f = out[-1].split()
    return (f[0], f[1]) if len(f) >= 2 else None


def root_trees_equal(pa, pb):
    """Numeric comparison of every branch of every tree; None if uproot absent."""
    try:
        import uproot
        import numpy as np
    except ImportError:
        return None
    if not (os.path.isfile(pa) and os.path.isfile(pb)):
        return None
    fa, fb = uproot.open(pa), uproot.open(pb)
    ka = sorted(k for k in fa.keys())
    kb = sorted(k for k in fb.keys())
    if ka != kb:
        return False
    for k in ka:
        ta, tb = fa[k], fb[k]
        if not hasattr(ta, "arrays"):
            continue
        aa, ab = ta.arrays(library="np"), tb.arrays(library="np")
        if sorted(aa) != sorted(ab):
            return False
        def bitwise_eq(x, y):
            # tobytes() is bitwise-exact and NaN-safe; recurse through jagged
            # (object-dtype) branches.
            x, y = np.asarray(x), np.asarray(y)
            if x.dtype == object or y.dtype == object:
                return len(x) == len(y) and all(bitwise_eq(a, b) for a, b in zip(x, y))
            return x.shape == y.shape and x.dtype == y.dtype and \
                x.tobytes() == y.tobytes()
        for br in aa:
            if not bitwise_eq(aa[br], ab[br]):
                return False
    return True


def events(tag, root):
    d = os.path.join(HERE, f"work-{root}-{tag}")
    return sorted(os.path.basename(p)[len("nusel_evt"):]
                  for p in glob.glob(os.path.join(d, "nusel_evt*")))


print("=" * 78)
print(f"GATE  {A.opt_tag} (wire-indexed dQ_dx_fit) vs {A.base_tag} (pre-change)")
print("=" * 78)
ngate = nfail = 0
fails = []
for root in ROOTS:
    based = os.path.join(HERE, f"work-{root}-{A.base_tag}")
    optd = os.path.join(HERE, f"work-{root}-{A.opt_tag}")
    for evt in events(A.base_tag, root):
        for name in ["mabc-pr.zip", f"pctree-pr-evt{evt}.tar.gz"]:
            pa = os.path.join(based, f"nusel_evt{evt}", name)
            pb = os.path.join(optd, f"nusel_evt{evt}", name)
            ha, hb = hash_archive(pa), hash_archive(pb)
            if ha is None or hb is None:
                verdict, extra = "MISSING", f"base={ha} new={hb}"
            elif ha == hb:
                verdict, extra = "identical", f"{ha[0][:16]} ({ha[1]} members)"
            else:
                verdict, extra = "DIFFER", f"base {ha[0][:16]} / new {hb[0][:16]}"
            ngate += 1
            if verdict != "identical":
                nfail += 1
                fails.append((evt, name, extra))
            print(f"{evt:<9} {name:<26} {verdict:<11} {extra}")
        # TSV: plain text
        ta = os.path.join(based, f"nusel_evt{evt}", f"nusel-evt{evt}.tsv")
        tb = os.path.join(optd, f"nusel_evt{evt}", f"nusel-evt{evt}.tsv")
        ngate += 1
        if os.path.isfile(ta) and os.path.isfile(tb) and \
           open(ta).read() == open(tb).read():
            print(f"{evt:<9} {'nusel-*.tsv':<26} {'identical':<11}")
        else:
            nfail += 1
            fails.append((evt, "nusel-*.tsv", "text diff"))
            print(f"{evt:<9} {'nusel-*.tsv':<26} {'DIFFER':<11}")
        # tracking-stm.root: numeric branch compare (byte compare is M2-void)
        ra = os.path.join(based, f"nusel_evt{evt}", "tracking-stm.root")
        rb = os.path.join(optd, f"nusel_evt{evt}", "tracking-stm.root")
        eq = root_trees_equal(ra, rb)
        if eq is None:
            print(f"{evt:<9} {'tracking-stm.root':<26} {'skipped':<11} (no uproot or file)")
        else:
            ngate += 1
            if eq:
                print(f"{evt:<9} {'tracking-stm.root':<26} {'identical':<11} (branch arrays)")
            else:
                nfail += 1
                fails.append((evt, "tracking-stm.root", "branch arrays differ"))
                print(f"{evt:<9} {'tracking-stm.root':<26} {'DIFFER':<11} (branch arrays)")

print(f"\nGATE: {ngate - nfail}/{ngate} comparisons identical -> "
      f"{'PASS' if nfail == 0 else 'FAIL'}")
for f in fails:
    print("   first divergence:", f)
    break

# ---------------------------------------------------------------- timing
print()
print("=" * 78)
print("TIMING  per-event wall/RSS (timecmd.py meta, sequential runs)")
print("=" * 78)
print(f"{'root':<9} {'label':<5} {'wall base':>9} {'wall opt':>9} {'rss base MB':>12} {'rss opt MB':>11}")
tw = {A.base_tag: 0, A.opt_tag: 0}
rmax = {A.base_tag: 0, A.opt_tag: 0}
for root in ROOTS:
    based = os.path.join(HERE, f"work-{root}-{A.base_tag}")
    metas = sorted(glob.glob(os.path.join(based, ".time_*.meta")))
    for ma in metas:
        lab = os.path.basename(ma)[len(".time_"):-len(".meta")]
        mo = os.path.join(HERE, f"work-{root}-{A.opt_tag}", f".time_{lab}.meta")
        def rd(p):
            d = dict(l.strip().split("=") for l in open(p) if "=" in l)
            return int(d["wall_s"]), int(d["maxrss_kb"])
        wa, ra = rd(ma)
        wo, ro = rd(mo) if os.path.isfile(mo) else (-1, -1)
        tw[A.base_tag] += wa; tw[A.opt_tag] += max(wo, 0)
        rmax[A.base_tag] = max(rmax[A.base_tag], ra)
        rmax[A.opt_tag] = max(rmax[A.opt_tag], ro)
        print(f"{root:<9} {lab:<5} {wa:>9} {wo:>9} {ra/1024:>12.0f} {ro/1024:>11.0f}")
print(f"\nTOTAL wall: {A.base_tag} {tw[A.base_tag]} s -> {A.opt_tag} {tw[A.opt_tag]} s "
      f"({100*(tw[A.base_tag]-tw[A.opt_tag])/max(tw[A.base_tag],1):.0f}% less)")
print(f"PEAK  RSS : {A.base_tag} {rmax[A.base_tag]/1024:.0f} MB -> "
      f"{A.opt_tag} {rmax[A.opt_tag]/1024:.0f} MB")

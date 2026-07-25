#!/usr/bin/env python3
"""Doc 52 A/B report: knob-off byte-identicality gate + knob-on effect.

Two independent questions, two tables.

1. GATE.  With the knobs off the new binary must reproduce the pre-change
   products already on disk, member-content-hash exact (never `cmp` on an
   archive -- M2).  Q/L baselines are work-<suff>-mainreal (what
   work-<suff>-d49son's ql_evt* symlinks resolve to); nusel baselines are
   work-<suff>-d49son itself.

2. EFFECT.  With the knobs on, how many clusters did the isolated un-merge
   actually split, and did the tagger verdicts move?  The split count comes from
   the run's own log line
     <ClusteringUnmergeBundle:prassoc> cluster N: A blobs -> main B + K associated
   and the verdicts from the per-event nusel TSV.

Repro:
  cd sbnd_xin
  ./run_d52_campaign.sh both
  python3 d52_ab_report.py
Full write-up: docs/52_isolated-grouping-fix-design.md
"""
import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABTEST = os.path.abspath(os.path.join(HERE, "..", "..", "abtest", "hash_archive.py"))
ROOTS = ["mcp10", "mcp1000", "mcp1000b"]

AP = argparse.ArgumentParser()
AP.add_argument("--off-tag", default="d52off")
AP.add_argument("--on-tag", default="d52on")
AP.add_argument("--nusel-base", default="d49son")
A = AP.parse_args()


def hash_archive(path):
    """Member-content hash (abtest/hash_archive.py), or None if absent."""
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


def events(tag, root):
    d = os.path.join(HERE, f"work-{root}-{tag}")
    return sorted(os.path.basename(p)[len("nusel_evt"):]
                  for p in glob.glob(os.path.join(d, "nusel_evt*")))


# ---------------------------------------------------------------- 1. the gate
print("=" * 78)
print(f"GATE  knob-off ({A.off_tag}) vs pre-change products")
print("=" * 78)
print(f"{'event':<9} {'stage':<7} {'archive':<26} {'verdict':<11} hash")
ngate = nfail = 0
fails = []
for root in ROOTS:
    offd = os.path.join(HERE, f"work-{root}-{A.off_tag}")
    for evt in events(A.off_tag, root):
        pairs = []
        # Q/L stage, baseline = whatever the d49son ql_evt symlink resolves to
        base_ql = os.path.realpath(os.path.join(HERE, f"work-{root}-{A.nusel_base}",
                                                f"ql_evt{evt}"))
        for f in ["mabc-all-apa.zip", "mabc-apa0-face0.zip", "mabc-apa1-face0.zip",
                  f"pctree-evt{evt}.tar.gz"]:
            pairs.append(("ql", f, os.path.join(base_ql, f),
                          os.path.join(offd, f"ql_evt{evt}", f)))
        # nusel stage, baseline = d49son
        base_nu = os.path.join(HERE, f"work-{root}-{A.nusel_base}", f"nusel_evt{evt}")
        for f in ["mabc-pr.zip", f"pctree-pr-evt{evt}.tar.gz"]:
            pairs.append(("nusel", f, os.path.join(base_nu, f),
                          os.path.join(offd, f"nusel_evt{evt}", f)))
        for stage, name, pa, pb in pairs:
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
                fails.append((evt, stage, name, extra))
            print(f"{evt:<9} {stage:<7} {name:<26} {verdict:<11} {extra}")
        # the TSV is plain text: diff it directly
        ta = os.path.join(base_nu, f"nusel-evt{evt}.tsv")
        tb = os.path.join(offd, f"nusel_evt{evt}", f"nusel-evt{evt}.tsv")
        ngate += 1
        if os.path.isfile(ta) and os.path.isfile(tb) and \
           open(ta).read() == open(tb).read():
            print(f"{evt:<9} {'nusel':<7} {'nusel-*.tsv':<26} {'identical':<11}")
        else:
            nfail += 1
            fails.append((evt, "nusel", "nusel-*.tsv", "text diff"))
            print(f"{evt:<9} {'nusel':<7} {'nusel-*.tsv':<26} {'DIFFER':<11}")

print(f"\nGATE: {ngate - nfail}/{ngate} comparisons identical -> "
      f"{'PASS' if nfail == 0 else 'FAIL'}")
for f in fails:
    print("   first divergence:", f)
    break

# ------------------------------------------------------------- 2. the effect
print()
print("=" * 78)
print(f"EFFECT  knob-on ({A.on_tag}): what the isolated un-merge did")
print("=" * 78)
SPLIT = re.compile(r"cluster (\d+): (\d+) blobs -> main (\d+) \+ (\d+) associated")
tot_split = tot_parts = tot_blobs_moved = 0
print(f"{'event':<9} {'clusters split':>14} {'assoc parts':>12} {'blobs moved':>12}")
per_event = {}
for root in ROOTS:
    ond = os.path.join(HERE, f"work-{root}-{A.on_tag}")
    for evt in events(A.on_tag, root):
        log = os.path.join(ond, f"nusel_evt{evt}", f"wct_nusel_evt{evt}.log")
        ns = npart = nmoved = 0
        if os.path.isfile(log):
            for line in open(log, errors="replace"):
                if "prassoc" not in line:
                    continue
                m = SPLIT.search(line)
                if m:
                    ns += 1
                    npart += int(m.group(4))
                    nmoved += int(m.group(2)) - int(m.group(3))
        per_event[evt] = (root, ns, npart, nmoved)
        tot_split += ns
        tot_parts += npart
        tot_blobs_moved += nmoved
        print(f"{evt:<9} {ns:>14} {npart:>12} {nmoved:>12}")
print(f"{'TOTAL':<9} {tot_split:>14} {tot_parts:>12} {tot_blobs_moved:>12}")


# Is the Q/L stage itself identical between the arms?  The knob-on path writes an
# "isolated" fill-in at the per-APA MABC, and "isolated" feeds QLMatching's
# decompose_cluster_groups.  If these hashes match, every verdict delta below is
# attributable to the un-merge alone; if they do not, two effects are mixed and
# the numbers are not interpretable as "what the un-merge did".
print()
print("=" * 78)
print("Q/L STAGE, on vs off  (must be identical for the effect to be attributable)")
print("=" * 78)
nql = nqldiff = 0
for root in ROOTS:
    for evt in events(A.on_tag, root):
        for f in ["mabc-all-apa.zip", "mabc-apa0-face0.zip", "mabc-apa1-face0.zip"]:
            ha = hash_archive(os.path.join(HERE, f"work-{root}-{A.off_tag}", f"ql_evt{evt}", f))
            hb = hash_archive(os.path.join(HERE, f"work-{root}-{A.on_tag}", f"ql_evt{evt}", f))
            nql += 1
            if ha != hb:
                nqldiff += 1
                print(f"  {evt} {f}: off {ha} / on {hb}")
print(f"Q/L: {nql - nqldiff}/{nql} identical -> "
      f"{'attributable' if nqldiff == 0 else 'MIXED EFFECTS'}")


# 5. Degenerate mains: a one-blob "main" is what STM/PR fits next.
print()
print("=" * 78)
print("Degenerate retained mains (main <= 3 blobs) after the isolated un-merge")
print("=" * 78)
ndeg = 0
for root in ROOTS:
    for evt in events(A.on_tag, root):
        log = os.path.join(HERE, f"work-{root}-{A.on_tag}", f"nusel_evt{evt}",
                           f"wct_nusel_evt{evt}.log")
        if not os.path.isfile(log):
            continue
        for line in open(log, errors="replace"):
            if "prassoc" not in line:
                continue
            m = SPLIT.search(line)
            if m and int(m.group(3)) <= 3:
                ndeg += 1
                print(f"  {evt} cluster {m.group(1)}: {m.group(2)} blobs -> "
                      f"main {m.group(3)} + {m.group(4)} assoc")
print(f"total: {ndeg} of {tot_split} splits left a main of <= 3 blobs")


def verdicts(tag):
    """{(evt, cluster): (tgm, stm, fc)} from the per-event nusel TSVs."""
    out = {}
    for root in ROOTS:
        d = os.path.join(HERE, f"work-{root}-{tag}")
        for p in glob.glob(os.path.join(d, "nusel_evt*", "nusel-evt*.tsv")):
            evt = re.search(r"nusel-evt(\d+)\.tsv", p).group(1)
            # Despite the .tsv name these tables are SPACE-aligned, not
            # tab-separated -- csv.DictReader(delimiter="\t") sees one field per
            # line.  Split on whitespace.  The bundle key is main_id; there is no
            # "cluster" column.
            lines = [ln.split() for ln in open(p) if ln.strip()]
            if not lines:
                continue
            hdr = lines[0]
            ix = {c: i for i, c in enumerate(hdr)}
            for row in lines[1:]:
                if len(row) != len(hdr):
                    continue
                key = (evt, row[ix["main_id"]])
                out[key] = tuple(row[ix[k]] for k in ("tgm", "stm", "fc"))
    return out


try:
    voff, von = verdicts(A.off_tag), verdicts(A.on_tag)
    def tally(v, i):
        return sum(1 for t in v.values() if t[i] == "1")
    print()
    print(f"{'verdict':<8} {'off':>6} {'on':>6} {'delta':>7}")
    for i, nm in enumerate(("tgm", "stm", "fc")):
        a, b = tally(voff, i), tally(von, i)
        print(f"{nm:<8} {a:>6} {b:>6} {b - a:>+7}")
    print(f"\nbundles: off {len(voff)}, on {len(von)}")
except Exception as e:                                   # TSV schema drift
    print(f"\n(verdict tally skipped: {e})")

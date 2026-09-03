#!/usr/bin/env python3
"""doc 99 -- A/B of T_cluster's three flash columns across the flash_at() range guard.

pr87_root_tree_diff.py answers "did any tree move".  This answers the two
questions that gate cannot: WHICH rows moved, and are they exactly the rows an
independent measurement said should move.

For every event present in both arms it reads T_cluster's
(cluster_id, flash_id, flash_time_us, flash_pe) and reports each differing row at
full precision, then checks three things:

  1. every differing row reads the invalid default (-1, 0.0, 0.0) in arm B --
     i.e. the fix replaced a garbage read with the documented sentinel, and did
     not merely perturb it;
  2. arm A's value on those rows is NOT the sentinel (otherwise the row would
     not have differed) and is characteristically raw memory -- denormal doubles
     or byte-pattern ints -- which is reported, not asserted, because raw memory
     is allowed to look like anything;
  3. the SET of differing cluster ids equals, per event, the set of clusters
     d99_flash_index_census.py --detail flagged OOB -- computed from the
     archives' cluster scalars without opening a single ROOT file.  Joining on
     the cluster id rather than comparing counts matters: a count match would be
     a weaker claim and a count mismatch an ambiguous one.

     THE DETAIL FILE MUST COME FROM `--stage pr`.  T_cluster is written from the
     PR-STAGE grouping, and the PR chain re-clusters: on mcp1k evt 59685 the Q/L
     archive holds 10 clusters and the PR archive 22, renumbered 1..22.  Feeding
     this check a Q/L-stage census made it fail on 24 events, every one of them
     "moved but not predicted" -- the signature of a predictor that cannot see
     part of the population, not of a bad fix.  Across the 308-event manifest
     the Q/L stage has 94 out-of-range clusters and the PR stage 158.

It also reports the RESIDUAL: clusters the census flagged WRONG (an in-range
index into a flash list that is not the one they matched) are present in these
rows and are NOT expected to move -- the range guard cannot see them.  That
number is the point of doc 99 sec 5.

Repro:
  python3 scripts/analysis/d99_flash_ab.py --a d92gatepr --b d99fixpr \\
      --detail /home/xqian/tmp/d99-flash-detail-308.tsv
Exit 0 iff checks 1 and 3 hold on every event.
"""
import argparse, math, os, re, sys

import numpy as np
import uproot

COLS = ["flash_id", "flash_time_us", "flash_pe"]


def looks_like_raw_memory(fid, t, pe):
    """Report-only characterization of a pre-fix value."""
    tags = []
    for name, v in (("time", t), ("pe", pe)):
        if v != 0.0 and abs(v) < 2.2250738585072014e-308:
            tags.append("%s=denormal(%.3e)" % (name, v))
        elif not math.isfinite(v):
            tags.append("%s=%r" % (name, v))
    b = int(fid) & 0xFFFFFFFF
    lo, hi = b & 0xFFFF, (b >> 16) & 0xFFFF
    if lo in (0x5555, 0xAAAA, 0x7FFF, 0xFFFF) or (lo == hi and lo != 0):
        tags.append("id=bytepattern(0x%08x)" % b)
    return ",".join(tags)


def read_tcluster(path):
    with uproot.open(path) as f:
        if "T_cluster" not in f:
            return None
        t = f["T_cluster"]
        keys = set(t.keys())
        if not {"cluster_id", *COLS} <= keys:
            return None
        a = t.arrays(["cluster_id"] + COLS, library="np")
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline arm suffix (pre-fix), e.g. d92gatepr")
    ap.add_argument("--b", required=True, help="arm suffix under test (post-fix), e.g. d99fixpr")
    ap.add_argument("--samples", default="ncpi0,nuecc48,mcp1k")
    ap.add_argument("--detail", default=None,
                    help="d99_flash_index_census.py --detail TSV; enables check 3")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    ap.add_argument("--show", type=int, default=12, help="differing rows to print per sample")
    args = ap.parse_args()

    oob, wrong = {}, {}
    if args.detail:
        with open(args.detail) as fp:
            hdr = fp.readline().rstrip("\n").split("\t")
            isa, ie, ic, ik = (hdr.index(x) for x in
                               ("sample", "event", "cluster_id", "class"))
            for line in fp:
                p = line.rstrip("\n").split("\t")
                key = (p[isa], int(p[ie]))
                (oob if p[ik] == "OOB" else wrong).setdefault(key, set()).add(int(p[ic]))

    rc = 0
    grand = [0, 0, 0, 0, 0]   # events, rows, differing rows, events with a diff, residual
    for smp in args.samples.split(","):
        da = os.path.join(args.root, "work-%s-%s" % (smp, args.a))
        db = os.path.join(args.root, "work-%s-%s" % (smp, args.b))
        if not (os.path.isdir(da) and os.path.isdir(db)):
            print("skip %s: missing arm" % smp)
            continue
        evts = sorted(int(m.group(1)) for m in
                      (re.match(r"pr_evt(\d+)$", d) for d in os.listdir(db)) if m)
        nev = nrow = ndiff = nevdiff = nresidual = 0
        bad_sentinel = census_mismatch = 0
        shown = 0
        for e in evts:
            pa = os.path.join(da, "pr_evt%d" % e, "tracking-pr.root")
            pb = os.path.join(db, "pr_evt%d" % e, "tracking-pr.root")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            A, B = read_tcluster(pa), read_tcluster(pb)
            if A is None or B is None:
                print("  %s evt%d: no usable T_cluster" % (smp, e)); rc = 1; continue
            if len(A["cluster_id"]) != len(B["cluster_id"]) or \
               not np.array_equal(A["cluster_id"], B["cluster_id"]):
                print("  %s evt%d: T_cluster row set differs -- NOT a flash-only change" % (smp, e))
                rc = 1
                continue
            nev += 1; nrow += len(A["cluster_id"])
            d = np.zeros(len(A["cluster_id"]), dtype=bool)
            for c in COLS:
                if A[c].dtype.kind == "f":
                    d |= ~((A[c] == B[c]) | (np.isnan(A[c]) & np.isnan(B[c])))
                else:
                    d |= (A[c] != B[c])
            n = int(d.sum())
            if n:
                nevdiff += 1
            ndiff += n
            present = set(int(x) for x in A["cluster_id"])
            nresidual += len(wrong.get((smp, e), set()) & present)
            for i in np.flatnonzero(d):
                fa = (A["flash_id"][i], A["flash_time_us"][i], A["flash_pe"][i])
                fb = (B["flash_id"][i], B["flash_time_us"][i], B["flash_pe"][i])
                if not (fb[0] == -1 and fb[1] == 0.0 and fb[2] == 0.0):
                    print("  CHECK1 FAIL %s evt%d cid=%d: B is not the sentinel: %r"
                          % (smp, e, A["cluster_id"][i], fb))
                    bad_sentinel += 1; rc = 1
                if shown < args.show:
                    print("  %-8s evt%-8d cid=%-4d  A: id=%-12d t=%-14.6e pe=%-14.6e  %s"
                          % (smp, e, A["cluster_id"][i], fa[0], fa[1], fa[2],
                             looks_like_raw_memory(*fa) or "(plausible-looking)"))
                    shown += 1
            if args.detail:
                moved = set(int(A["cluster_id"][i]) for i in np.flatnonzero(d))
                want = oob.get((smp, e), set()) & present
                if moved != want:
                    print("  CHECK3 FAIL %s evt%d: moved-but-not-predicted=%s  "
                          "predicted-but-unmoved=%s"
                          % (smp, e, sorted(moved - want), sorted(want - moved)))
                    census_mismatch += 1; rc = 1
        print("%-9s events=%-5d T_cluster rows=%-7d differing rows=%-5d events with a diff=%-4d "
              "sentinel-violations=%d census-mismatches=%d  residual WRONG rows=%d"
              % (smp, nev, nrow, ndiff, nevdiff, bad_sentinel, census_mismatch, nresidual))
        grand[0] += nev; grand[1] += nrow; grand[2] += ndiff; grand[3] += nevdiff
        grand[4] += nresidual

    print("\nTOTAL     events=%d  T_cluster rows=%d  differing rows=%d  events with a diff=%d"
          "  residual WRONG rows=%d" % tuple(grand))
    print("VERDICT: %s" % ("PASS -- every differing row is an out-of-range cluster the census "
                           "predicted, and reads the sentinel after the fix" if rc == 0
                           else "FAIL -- see above"))
    return rc


if __name__ == "__main__":
    sys.exit(main())

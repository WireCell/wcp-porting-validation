#!/usr/bin/env python3
"""valfast gate 1: PR-arm compare with a meaningful exit code.

Fork of pr_arm_compare.py (kept untouched -- prior docs cite its output) with
two deliberate differences:

  1. sys.exit(0) iff everything compared is identical, else 1 -- so
     valfast_compare.sh can gate on it.
  2. VECTOR branches of T_tagger/T_kine are compared as SORTED multisets,
     scalars exactly.  Reason (measured, A/A' vfsmka vs vfsmkb 2026-08-02):
     under setarch -R with one binary and one input, the per-candidate vector
     branches (pio_2_v_*, br3_6_v_*, numu_cc_1_*, ...) come back as
     PERMUTATIONS of identical values -- the M4 residual, pointer-order-
     dependent candidate enumeration.  Archives and every score column are
     bit-stable; only the fill ORDER of these vectors is not.  An exact
     comparison would therefore false-DIFF ~half the events on an A/A' run.
     A knob that changes any VALUE still fails the multiset compare.

usage: vf_tree_compare.py <armA> <armB> <evt> [<evt> ...]
Arms are work-* dirs holding pr_evt<ID>/{mabc-pr.zip,pctree-pr-evt<ID>.tar.gz,tracking-pr.root}.
"""
import sys, subprocess, os
import numpy as np
import uproot

HASH = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest/hash_archive.py"

def archive_hash(path):
    if not os.path.exists(path):
        return "MISSING"
    out = subprocess.run([sys.executable, HASH, path], capture_output=True, text=True)
    return [line.split()[0:2] for line in out.stdout.strip().splitlines()]

def rows(path):
    if not os.path.exists(path):
        return "MISSING"
    f = uproot.open(path)
    res = {}
    for tn in ("T_tagger", "T_kine"):
        if not any(k.split(";")[0] == tn for k in f.keys()):
            res[tn] = "NOTREE"
            continue
        arr = f[tn].arrays(library="np")
        res[tn] = {k: v for k, v in sorted(arr.items())}
    return res

def entries_equal(va, vb):
    if len(va) != len(vb):
        return False
    for x, y in zip(va, vb):
        ax, ay = np.asarray(x), np.asarray(y)
        if ax.shape != ay.shape and ax.size != ay.size:
            return False
        if ax.ndim == 0 or ax.size <= 1:
            if not np.array_equal(ax, ay):
                return False
        else:
            if not np.array_equal(np.sort(ax.ravel()), np.sort(ay.ravel())):
                return False
    return True

def rows_equal(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b, [] if a == b else ["tree-presence"]
    diffs = []
    for tn in ("T_tagger", "T_kine"):
        ta, tb = a[tn], b[tn]
        if isinstance(ta, str) or isinstance(tb, str):
            if ta != tb:
                diffs.append(f"{tn}: {ta} vs {tb}")
            continue
        for k in sorted(set(ta) | set(tb)):
            if k not in ta or k not in tb:
                diffs.append(f"{tn}.{k}: branch missing")
                continue
            try:
                same = entries_equal(ta[k], tb[k])
            except Exception:
                same = str(ta[k]) == str(tb[k])
            if not same:
                diffs.append(f"{tn}.{k}")
    return not diffs, diffs

def main():
    armA, armB = sys.argv[1], sys.argv[2]
    evts = sys.argv[3:]
    n_mabc = n_pct = n_rows = 0
    for evt in evts:
        da = os.path.join(armA, f"pr_evt{evt}")
        db = os.path.join(armB, f"pr_evt{evt}")
        mab = archive_hash(os.path.join(da, "mabc-pr.zip")) == archive_hash(os.path.join(db, "mabc-pr.zip"))
        pct = archive_hash(os.path.join(da, f"pctree-pr-evt{evt}.tar.gz")) == archive_hash(os.path.join(db, f"pctree-pr-evt{evt}.tar.gz"))
        req, diffs = rows_equal(rows(os.path.join(da, "tracking-pr.root")),
                                rows(os.path.join(db, "tracking-pr.root")))
        n_mabc += mab; n_pct += pct; n_rows += req
        tag = "OK " if (mab and pct and req) else "DIFF"
        extra = "" if (mab and pct and req) else \
            f"  mabc={'=' if mab else '≠'} pctree={'=' if pct else '≠'} rows={'=' if req else '≠'} {';'.join(diffs[:6])}"
        print(f"{tag} evt {evt}{extra}")
    n = len(evts)
    print(f"SUMMARY {n} events | mabc identical {n_mabc}/{n} | pctree {n_pct}/{n} | "
          f"tagger+kine rows (vectors as multisets) {n_rows}/{n}")
    return 0 if (n_mabc == n and n_pct == n and n_rows == n) else 1

if __name__ == "__main__":
    sys.exit(main())

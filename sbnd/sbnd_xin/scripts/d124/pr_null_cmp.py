#!/usr/bin/env python3
"""doc sbnd_xin/124 S0 -- compare two PR roots event by event: every branch value of T_tagger, T_kine,
T_bundle, T_cluster, T_flash in tracking-pr.root, plus the nusel-evt table and the 'DL vertex failed' count.

usage: pr_null_cmp.py <pr_root_ref> <pr_root_rerun>
"""
import glob, os, sys
import numpy as np, uproot
ref, new = sys.argv[1], sys.argv[2]
TREES = ("T_tagger", "T_kine", "T_bundle", "T_cluster", "T_flash", "T_rec_charge")
nd = 0; n = 0
for d in sorted(glob.glob(os.path.join(new, "pr_evt*"))):
    e = os.path.basename(d); r = os.path.join(ref, e); n += 1
    diffs = []
    fa = uproot.open(os.path.join(r, "tracking-pr.root")); fb = uproot.open(os.path.join(d, "tracking-pr.root"))
    for t in TREES:
        if (t in fa) != (t in fb):
            diffs.append(t + ":missing"); continue
        if t not in fa:
            continue                     # absent in both (no candidate): identical
        A = fa[t].arrays(library="np"); B = fb[t].arrays(library="np")
        for k in A:
            va, vb = A[k], B.get(k)
            if vb is None: diffs.append("%s.%s:missing" % (t, k)); continue
            try:
                xa = np.concatenate([np.ravel(x) for x in va]) if va.dtype == object else np.ravel(va)
                xb = np.concatenate([np.ravel(x) for x in vb]) if vb.dtype == object else np.ravel(vb)
                same = xa.shape == xb.shape and (np.array_equal(xa, xb) or np.array_equal(xa, xb, equal_nan=True))
            except Exception:
                same = len(va) == len(vb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(va, vb))
            if not same: diffs.append("%s.%s" % (t, k))
    ev = e[6:]
    if open(os.path.join(r, "nusel-evt%s.tsv" % ev)).read() != open(os.path.join(d, "nusel-evt%s.tsv" % ev)).read():
        diffs.append("nusel")
    dlf = sum(l.count("DL vertex failed") for l in open(os.path.join(d, "wct_pr_evt%s.log" % ev), errors="replace"))
    if dlf: diffs.append("DLfail%d" % dlf)
    if diffs:
        nd += 1; print("%s DIFF %d: %s" % (e, len(diffs), " ".join(diffs[:12])))
print("events %d, differing %d" % (n, nd))

#!/usr/bin/env python3
"""Compare two PR-chain arms: archive member hashes + T_tagger/T_kine rows.

usage: nbl_compare.py <armA> <armB> <evt> [<evt> ...]
Arms are sbnd_xin work-* dirs holding pr_evt<ID>/{mabc-pr.zip,pctree-pr-evt<ID>.tar.gz,tracking-pr.root}.
"""
import sys, subprocess, os
import numpy as np
import uproot

HASH = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest/hash_archive.py"

def archive_hash(path):
    if not os.path.exists(path):
        return "MISSING"
    out = subprocess.run([sys.executable, HASH, path], capture_output=True, text=True)
    # output lines are "<content-hash> <nmembers> <abs path>"; drop the path
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
        keys = set(ta) | set(tb)
        for k in sorted(keys):
            va, vb = ta.get(k), tb.get(k)
            if va is None or vb is None:
                diffs.append(f"{tn}.{k}: branch missing")
                continue
            try:
                same = all(np.array_equal(np.asarray(x), np.asarray(y))
                           for x, y in zip(va, vb)) and len(va) == len(vb)
            except Exception:
                same = str(va) == str(vb)
            if not same:
                diffs.append(f"{tn}.{k}")
    return not diffs, diffs

armA, armB = sys.argv[1], sys.argv[2]
evts = sys.argv[3:]
n_mabc = n_pct = n_rows = 0
for evt in evts:
    da = os.path.join(armA, f"pr_evt{evt}")
    db = os.path.join(armB, f"pr_evt{evt}")
    mab = archive_hash(os.path.join(da, "mabc-pr.zip")) == archive_hash(os.path.join(db, "mabc-pr.zip"))
    pct = archive_hash(os.path.join(da, f"pctree-pr-evt{evt}.tar.gz")) == archive_hash(os.path.join(db, f"pctree-pr-evt{evt}.tar.gz"))
    req, diffs = rows_equal(rows(os.path.join(da, "tracking-pr.root")), rows(os.path.join(db, "tracking-pr.root")))
    n_mabc += mab; n_pct += pct; n_rows += req
    tag = "OK " if (mab and pct and req) else "DIFF"
    extra = "" if (mab and pct and req) else f"  mabc={'=' if mab else '≠'} pctree={'=' if pct else '≠'} rows={'=' if req else '≠'} {';'.join(diffs[:6])}"
    print(f"{tag} evt {evt}{extra}")
print(f"SUMMARY {len(evts)} events | mabc identical {n_mabc}/{len(evts)} | pctree {n_pct}/{len(evts)} | tagger+kine rows {n_rows}/{len(evts)}")

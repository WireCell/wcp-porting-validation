#!/usr/bin/env python3
"""doc 108 sec 5: T_tagger branches that never change value over a random
file sample (candidates for "never filled on SBND" -- NOT proof of dead code:
a rarely-firing flag looks identical on a finite sample).

Usage:
  python3 d108_tagger_branch_census.py [--run ...] [--nfiles 1500] [--seed 7] [--procs 24]
"""
import argparse
import collections
import glob
import multiprocessing as mp
import os
import random

import numpy as np
import uproot


def one(p):
    F = uproot.open(p)
    if "T_tagger" not in F:
        return None
    T = F["T_tagger"]
    out = {}
    for k in T.keys():
        if k.startswith("act_"):
            continue
        try:
            a = T[k].array(library="np")
            v = np.concatenate([np.ravel(x) for x in a]) if a.dtype == object else np.ravel(a)
            out[k] = set(np.round(v.astype(float), 4).tolist()[:50])
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/nfs/data/1/xqian/sbnd_data/run")
    ap.add_argument("--nfiles", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.run, "tracking-pr", "tracking-pr_*.root")))
    random.seed(a.seed)
    random.shuffle(files)
    F = uproot.open(files[0])
    print("compression", F.file.compression)
    for t in F.keys():
        tt = F[t]
        if hasattr(tt, "num_entries"):
            print(f"tree {t:18s} entries {tt.num_entries:6d} compressed {tt.compressed_bytes:8d} "
                  f"uncompressed {tt.uncompressed_bytes:8d}")
    with mp.Pool(a.procs) as P:
        res = [r for r in P.imap_unordered(one, files[:a.nfiles], chunksize=20) if r]
    print("sampled files", a.nfiles, "with T_tagger", len(res))
    vals = collections.defaultdict(set)
    for r in res:
        for k, s in r.items():
            if len(vals[k]) < 60:
                vals[k] |= s
    const = sorted(k for k, s in vals.items() if len(s) <= 1)
    print("non-act branches", len(vals), "constant over sample", len(const))
    print("constant by prefix", collections.Counter(k.split('_')[0] for k in const).most_common())
    print("constant", const)


if __name__ == "__main__":
    main()

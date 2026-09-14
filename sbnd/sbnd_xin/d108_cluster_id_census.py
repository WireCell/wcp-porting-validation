#!/usr/bin/env python3
"""doc 108 sec 2.1: what T_tagger.cluster_id points to when it is not the
selected activity (act_cluster_id[argmax(act_is_selected)]).

Usage:
  python3 d108_cluster_id_census.py [--run /nfs/data/1/xqian/sbnd_data/run] [--procs 24]

Per T_tagger row: relation of cluster_id to the selected activity (same /
another act_* entry / not in act_*), whether T_kine row i carries the same
cluster_id + nu_index and vertex, and the T_cluster is_main / is_associated
status and flash_id (== matched_flash_gid under flash_by_gid) of both ids.
Read-only on the production sample.
"""
import argparse
import collections
import glob
import multiprocessing as mp
import os

import numpy as np
import uproot


def one(p):
    out = []
    try:
        F = uproot.open(p)
        if "T_tagger" not in F:
            return out
        T = F["T_tagger"].arrays(["cluster_id", "matched_flash_gid", "nu_index", "act_cluster_id",
                                  "act_is_selected", "nu_x"], library="np")
        K = F["T_kine"].arrays(["cluster_id", "nu_index", "kine_nu_x_corr"], library="np")
        C = F["T_cluster"].arrays(["cluster_id", "is_main", "is_associated", "flash_id"], library="np")
        cl = {int(c): (int(m), int(a), int(f)) for c, m, a, f in
              zip(C["cluster_id"], C["is_main"], C["is_associated"], C["flash_id"])}
        for i in range(len(T["cluster_id"])):
            cid = int(T["cluster_id"][i])
            acts = list(T["act_cluster_id"][i])
            scid = acts[int(np.argmax(list(T["act_is_selected"][i])))]
            rel = "same" if cid == scid else ("in_act" if cid in acts else "not_in_act")
            c, s = cl.get(cid), cl.get(scid)
            out.append(dict(
                rel=rel,
                kine_same=(int(K["cluster_id"][i]) == cid and int(K["nu_index"][i]) == int(T["nu_index"][i])),
                vx_same=abs(float(K["kine_nu_x_corr"][i]) - float(T["nu_x"][i])) < 1e-3,
                cid_in_Tc=c is not None, cid_main=c[0] if c else -1, cid_assoc=c[1] if c else -1,
                cid_flash_eq_gid=(c[2] == int(T["matched_flash_gid"][i])) if c else None,
                sel_main=s[0] if s else -1, sel_assoc=s[1] if s else -1))
    except Exception as e:  # report, never hide, a bad file
        out.append(dict(rel="ERR:" + str(e)[:60]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/nfs/data/1/xqian/sbnd_data/run")
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.run, "tracking-pr", "tracking-pr_*.root")))
    with mp.Pool(a.procs) as P:
        rows = [r for rr in P.imap_unordered(one, files, chunksize=50) for r in rr]
    print("files", len(files), "rows", len(rows))
    print("rel", dict(collections.Counter(r["rel"] for r in rows)))
    for rel in ("same", "in_act", "not_in_act"):
        R = [r for r in rows if r["rel"] == rel]
        print("==", rel, len(R))
        for k in ("kine_same", "vx_same", "cid_in_Tc", "cid_main", "cid_assoc", "cid_flash_eq_gid",
                  "sel_main", "sel_assoc"):
            print("  ", k, dict(collections.Counter(r[k] for r in R)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduction harness for the PDHD clus_all_tpc 111<->110 point-order
non-determinism.

The cluster-array Numpy schema (aux/docs/ClusterArrays.org) stores, per ICluster,
a `cluster_<id>_bnodes.npy` (blob node array; col 0 = global vertex descriptor
`vdesc`) plus edge arrays whose tail/head columns are ROW INDICES into the node
arrays. The reader `ClusterArrays::to_cluster` (aux/src/ClusterArrays.cxx:539)
uses the stored vdesc VALUE directly as the boost-graph vertex descriptor, and
edges resolve endpoints as `node_array[row][vdesc]`. Downstream PointTreeBuilding
iterates a blob's children via a std::set<vertex> SORTED BY vdesc, so the blob
(hence sampled-point) iteration order is determined entirely by the blob vdesc
values -- NOT by row order.

Therefore the faithful, topology-preserving way to reorder points within every
cluster (a proxy for whatever a relink perturbs) is to PERMUTE THE vdesc VALUES
AMONG THE BLOB ROWS:
  - col0 of bnodes is reassigned by a fixed-seed permutation among blob rows;
  - all other columns (geometry, charge) stay with their row (the physical blob);
  - edges index by row -> automatically read the new vdesc -> graph topology and
    geometry are byte-identical, ONLY the blob iteration order changes.

Usage:
  permute_clusters.py SRC_DIR DST_DIR --mode {identity,permute} [--seed N]

`identity` = same-binary control (copy tarballs verbatim, re-tar through the same
code path so the only difference vs `permute` is the vdesc shuffle).
`permute`   = shuffle blob vdesc per cluster with the given seed.

Only the *-ms-active.tar.gz tarballs are rewritten (live clusters feed
cathode_connect); everything else in SRC_DIR is copied verbatim so DST_DIR is a
drop-in input/work dir for `run_clus_evt.sh -s <tag>`.
"""
import sys, os, io, glob, tarfile, argparse, shutil
import numpy as np

def rewrite_active_tarball(src_path, dst_path, mode, seed):
    """Read every cluster_*_bnodes.npy, permute its col0 (vdesc) among rows
    (mode=permute), re-emit the tarball preserving entry names and order."""
    with tarfile.open(src_path, "r:gz") as tin:
        members = tin.getmembers()
        # load all entries into memory
        blobs = {m.name: tin.extractfile(m).read() for m in members}

    n_perm = 0
    for name in list(blobs):
        if name.endswith("_bnodes.npy"):
            arr = np.load(io.BytesIO(blobs[name]))
            if mode == "permute":
                nb = arr.shape[0]
                # deterministic per-cluster permutation: seed varies with the
                # cluster id embedded in the name so different clusters get
                # different shuffles, all reproducible.
                cid = int(name.split("_")[1])
                rng = np.random.default_rng(seed * 1_000_003 + cid)
                perm = rng.permutation(nb)
                col0 = arr[:, 0].copy()
                arr[:, 0] = col0[perm]          # reassign vdesc among rows
                buf = io.BytesIO(); np.save(buf, arr); blobs[name] = buf.getvalue()
                n_perm += 1
    # write out, preserving member order/metadata
    with tarfile.open(dst_path, "w:gz") as tout:
        for m in members:
            data = blobs[m.name]
            ti = tarfile.TarInfo(name=m.name)
            ti.size = len(data); ti.mtime = m.mtime; ti.mode = m.mode
            ti.uid = m.uid; ti.gid = m.gid; ti.type = m.type
            tout.addfile(ti, io.BytesIO(data))
    return n_perm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir"); ap.add_argument("dst_dir")
    ap.add_argument("--mode", choices=["identity", "permute"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    os.makedirs(a.dst_dir, exist_ok=True)
    actives = set(glob.glob(os.path.join(a.src_dir, "clusters-apa-apa*-ms-active.tar.gz")))

    total_perm = 0
    for src in sorted(glob.glob(os.path.join(a.src_dir, "*"))):
        base = os.path.basename(src)
        dst = os.path.join(a.dst_dir, base)
        if src in actives:
            n = rewrite_active_tarball(src, dst, a.mode, a.seed)
            total_perm += n
            print(f"  {base}: {'permuted' if a.mode=='permute' else 'identity'} "
                  f"{n} blob arrays")
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
        # skip subdirs (not needed as clus input)
    print(f"[{a.mode}] wrote {a.dst_dir}  (blob arrays touched: {total_perm})")

if __name__ == "__main__":
    main()

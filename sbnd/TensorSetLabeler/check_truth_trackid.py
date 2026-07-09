#!/usr/bin/env python3
"""Sanity checks for the wclsTensorSetLabeler output.

1. mabc.zip 'truth_trackid' Bee sets: non-empty, labeled fraction, and spatial
   coherence (fraction of nearest-neighbor point pairs sharing the same
   trackid -- visually-close points should mostly carry the same track).
2. trash-all-apa.tar.gz (TensorFileSink dump): presence of the
   truth_per_track tensor, run/subrun/event + nu_* tensor-set metadata, and
   per-blob 'trackid' in the blob scalar PCs.

Usage: python3 check_truth_trackid.py [mabc.zip] [trash-all-apa.tar.gz]
"""
import io
import json
import sys
import tarfile
import zipfile
from collections import Counter

import numpy as np


def check_bee(zpath):
    print(f"=== Bee zip: {zpath}")
    zf = zipfile.ZipFile(zpath)
    names = [n for n in zf.namelist()
             if n.endswith("-truth_trackid.json") or n.endswith("-truth_trackid_labeled.json")]
    if not names:
        print("FAIL: no truth_trackid bee sets found")
        return False
    ok = True
    for n in sorted(names):
        d = json.load(io.BytesIO(zf.read(n)))
        x = np.array(d.get("x", []), float)
        y = np.array(d.get("y", []), float)
        z = np.array(d.get("z", []), float)
        cid = np.array(d.get("cluster_id", []), int)
        npts = len(x)
        rse = (d.get("runNo"), d.get("subRunNo"), d.get("eventNo"))
        if npts == 0:
            print(f"{n}: EMPTY  rse={rse}")
            ok = False
            continue
        labeled = cid >= 0
        frac_lab = labeled.mean()
        uniq = len(set(cid[labeled].tolist()))
        top = Counter(cid[labeled].tolist()).most_common(5)
        # nearest-neighbor coherence on labeled points
        coher = float("nan")
        if labeled.sum() > 10:
            # nearest-neighbor coherence, brute force on a subsample (no scipy)
            pts = np.c_[x, y, z][labeled]
            ids = cid[labeled]
            rng = np.random.default_rng(42)
            m = min(len(pts), 1500)
            sel = rng.choice(len(pts), m, replace=False)
            same = []
            for i in sel:
                d2 = ((pts - pts[i]) ** 2).sum(axis=1)
                d2[i] = np.inf
                same.append(ids[i] == ids[np.argmin(d2)])
            coher = float(np.mean(same))
        print(f"{n}: rse={rse} npts={npts} labeled={frac_lab:.1%} "
              f"ntracks={uniq} NN-coherence={coher:.1%} top5={top}")
        # truth_trackid_labeled carries only labeled points (frac_lab==1 by
        # construction; unlabeled points live in truth_unlabeled).  Coherence
        # is the real quality signal.
        if npts == 0 or (coher == coher and coher < 0.8):
            ok = False
    print("BEE CHECK:", "PASS" if ok else "FAIL")
    return ok


def check_tar(tpath):
    print(f"=== Tensor dump: {tpath}")
    tf = tarfile.open(tpath)
    members = tf.getnames()
    # index tensors by their metadata (member names are positional)
    tmds = {}
    for m in members:
        if m.endswith("_metadata.json") and "tensorset" not in m:
            tmds[m] = json.load(io.BytesIO(tf.extractfile(m).read()))
    truth = [m for m, md in tmds.items() if md.get("datatype") == "truth_per_track"]
    print(f"members={len(members)} truth_per_track tensors={len(truth)}")
    ok = bool(truth)
    for m in truth[:2]:
        md = tmds[m]
        arr = np.load(io.BytesIO(tf.extractfile(m.replace("_metadata.json", "_array.npy")).read()))
        print(f"{md.get('datapath')}: shape={arr.shape}")
        print("  columns:", md.get("columns"))
        if arr.size:
            print("  first row:", dict(zip(md.get("columns", []), arr[0].tolist())))
    # tensor-set metadata (run/subrun/event + nu_*)
    setmd = [m for m in members if "tensorset" in m and m.endswith("_metadata.json")]
    for m in setmd[:2]:
        md = json.load(io.BytesIO(tf.extractfile(m).read()))
        print(m, json.dumps(md)[:300])
        for key in ("runNo", "subRunNo", "eventNo", "nu_flavor"):
            if key not in md:
                print(f"  MISSING metadata key {key}")
                ok = False
    # blob scalar trackid: the live pctree stores the per-blob scalar PC as
    # one concatenated array at .../live/pointclouds/namedpcs/scalar/arrays/trackid
    tid_members = [m for m, md in tmds.items()
                   if "/live/" in md.get("datapath", "")
                   and md.get("datapath", "").endswith("scalar/arrays/trackid")]
    print(f"live scalar trackid arrays: {len(tid_members)}")
    for m in tid_members:
        v = np.load(io.BytesIO(tf.extractfile(m.replace("_metadata.json", "_array.npy")).read()))
        print(f"  {tmds[m]['datapath']}: {len(v)} blobs labeled={np.mean(v >= 0):.1%}, "
              f"unique tracks={len(set(v[v >= 0].tolist()))}")
        ok &= bool(len(v)) and (v >= 0).any()
    print("TAR CHECK:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    zpath = sys.argv[1] if len(sys.argv) > 1 else "mabc.zip"
    tpath = sys.argv[2] if len(sys.argv) > 2 else "trash-all-apa.tar.gz"
    ok = check_bee(zpath)
    try:
        ok &= check_tar(tpath)
    except FileNotFoundError:
        print(f"(no {tpath}, skipping tar check)")
    sys.exit(0 if ok else 1)

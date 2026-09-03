#!/usr/bin/env python3
"""doc pdvd/31: attribute the missing Steiner coverage on 039349/14 to a STAGE.

Doc 26 section 7.5 measured the symptom: cluster 36 is one straight cosmic whose
steiner cloud has 632 points on the half ABOVE the vertex V and 0 along the
111 cm from V to A.  Doc 26 section 8 sent it to a Steiner-terminals campaign on
the assumption that the terminal finder is what starves it.  This script tests
that assumption, entirely offline, from dumps that already exist.

It answers, for a region selected geometrically along a named V->end line:

  1. does the region's charge belong to the cluster at all?  (Bee real_cluster_id)
  2. did BlobSampler place 3D points there?                  (the "3d" point cloud)
  3. do those points pass find_peak_point_indices' candidacy gate?
        charge > cut AND calc_charge_wcp quality, exactly as
        Facade_Cluster.cxx:1031-1112 computes it with
        disable_dead_mix_cell=false (the value production passes,
        CreateSteinerGraph.cxx:283)
  4. how many distinct BLOBS hold at least one candidate?

(4) is the load-bearing number.  find_steiner_terminals runs
find_peak_point_indices once per blob (SteinerGrapher.cxx:605-622) and inserts
every blob's peaks, so a blob with >=1 candidate yields >=1 terminal.  If the
region with no steiner coverage has MORE candidate-bearing blobs than the
control region that has coverage, terminal selection is not what dropped them
and the defect is downstream (the Phase 2/3 filters or the tree build).

It also cross-checks the per-point per-plane charges against the 2-D (wire,
slice) charge map (the ctpc_a<A>f<F>p<P> point clouds), which is what exposed
the separate V-plane defect of doc 31 section 5: charge present in the 2-D map,
zero on the sampled point.

--------------------------------------------------------------------------
FOUR TRAPS, each of which produced a wrong number first (read before editing):

* Use `x_t0cor`, NOT `x`, to compare against PR-log / calib-dump coordinates.
  With `x` the nearest sampled point to V is 16 cm away and every selection
  comes back empty.  But `x_t0cor` is meaningless (order 1e9) for points whose
  cluster has no t0, so never take a min/max over it globally.

* The flat live ".../namedpcs/3d" cloud is a BLOB-ORDERED CONCATENATION.
  np.repeat(arange(n_blobs), scalar/npoints) recovers exact blob attribution;
  sum(npoints) == n_points is the assertion that this still holds.

* The Bee "clustering-global" layer has the same POINT COUNT as the 3d cloud but
  NOT the same order (max |y_bee*10 - y_pc| was 5266 mm).  Select in Bee's own
  (y, z) cm coordinates geometrically; never index one array with the other's
  mask.

* Identify the ctpc dataset by CHARGE-MATCHING, not by decoding `wpid`.  The
  wpid guess gave ctpc_a2f1pV and 0/140 agreement; the true set is ctpc_a4f0pV
  with 140/140 exact.  slice_index = (t/500) floored to a multiple of 4 (tick
  0.5 us, imaging rebin 4).  The script refuses to report cross-referenced
  numbers unless the calibration matches.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/steiner_terminal_attribution.py \
      work/039349_14_d25r13fix
"""
import io
import json
import os
import sys
import tarfile
import zipfile
from collections import Counter

import numpy as np

# doc 26 section 7.5 / stm/gates/r13_duplicate_pair_png.py CASES, in cm.
# V is the vertex where the steiner cloud stops; A is the far end of the
# uncovered 111 cm; U is the far end of the covered half (the control).
V_CM = np.array([273.26, -118.90, 86.61])
A_CM = np.array([196.86, -167.76, 151.48])
U_CM = np.array([392.0, -83.0, 5.0])

TICK_NS = 500.0     # 0.5 us DAQ tick
REBIN = 4           # imaging rebin (protodunevd/img.jsonnet:33)
TOL_MM = 30.0       # 3 cm, the same tolerance doc 26 section 7.5 used


class Dump:
    """Random access to a pctree tar.gz by datapath suffix."""

    def __init__(self, path):
        self.tar = tarfile.open(path)
        self.index = {}
        for name in self.tar.getnames():
            if not name.endswith("_metadata.json"):
                continue
            meta = json.loads(self.tar.extractfile(name).read())
            dp = meta.get("datapath", "")
            if dp:
                self.index[dp] = name.replace("_metadata.json", "_array.npy")

    def find(self, suffix):
        hits = [dp for dp in self.index if dp.endswith(suffix)]
        if len(hits) != 1:
            raise KeyError(f"{suffix!r} matched {len(hits)} datapaths")
        return hits[0]

    def get(self, datapath):
        buf = self.tar.extractfile(self.index[datapath]).read()
        return np.load(io.BytesIO(buf)).ravel()

    def arr(self, suffix):
        return self.get(self.find(suffix))


def seg_mask(P, a, b, tol=TOL_MM):
    """Points within `tol` of the segment a->b, excluding its two endpoints."""
    d = b - a
    t = np.clip(((P - a) @ d) / (d @ d), 0.0, 1.0)
    dist = np.linalg.norm(P - (a + t[:, None] * d), axis=1)
    return (dist < tol) & (t > 0.02) & (t < 0.98)


def calc_charge_wcp(qu, qv, qw, cut):
    """Vectorised Cluster::calc_charge_wcp(..., disable_dead_mix_cell=false).

    Facade_Cluster.cxx:1087-1109.  A plane passes if it clears the cut OR reads
    exactly zero (no signal is not held against the point); the returned charge
    is the RMS over the planes with NONZERO value, and is forced to 0 unless at
    least two planes contributed.  Returns (quality, rms).
    """
    quality = np.ones(len(qu), dtype=bool)
    acc = np.zeros(len(qu))
    n = np.zeros(len(qu), dtype=int)
    for q in (qu, qv, qw):
        quality &= (q > cut) | (q == 0)
        nz = q != 0
        acc += np.where(nz, q * q, 0.0)
        n += nz
    rms = np.where(n > 1, np.sqrt(np.divide(acc, np.maximum(n, 1))), 0.0)
    return quality, rms


def load_ctpc(dump, apa, face, plane):
    """(slice_index, wind) -> charge for one ctpc_a<A>f<F>p<P> point cloud."""
    base = dump.find(f"ctpc_a{apa}f{face}p{plane}/arrays/charge").rsplit("/", 1)[0] + "/"
    wind = dump.get(base + "wind").tolist()
    sind = dump.get(base + "slice_index").tolist()
    chg = dump.get(base + "charge").tolist()
    return {(s, w): c for w, s, c in zip(wind, sind, chg)}


def identify_ctpc(dump, slice_of, wire_index, charge_val, probe):
    """Find the (apa, face) whose ctpc maps reproduce the stored per-point
    charges exactly, by matching values on `probe` points known to be nonzero.

    Returns (apa, face, n_match, n_probe).  Guessing from `wpid` instead of
    doing this is trap 4 in the header.
    """
    best = (0, None, None)
    for dp in [d for d in dump.index if "ctpc_a" in d and d.endswith("pV/arrays/charge")]:
        tag = dp.split("ctpc_a", 1)[1].split("/", 1)[0]      # e.g. "4f0pV"
        apa, rest = tag.split("f", 1)
        face = rest.split("p", 1)[0]
        cmap = load_ctpc(dump, apa, face, "V")
        hit = sum(1 for i in probe
                  if abs(cmap.get((int(slice_of[i]), int(wire_index[i])), -1e30)
                         - charge_val[i]) < 1e-3)
        if hit > best[0]:
            best = (hit, int(apa), int(face))
    return best[1], best[2], best[0], len(probe)


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else "work/039349_14_d25r13fix"
    cut = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0   # PDVD steiner_terminal_charge

    pct = [f for f in os.listdir(workdir) if f.startswith("pctree-") and f.endswith(".tar.gz")]
    if not pct:
        sys.exit(f"no pctree-*.tar.gz in {workdir}")
    dump = Dump(os.path.join(workdir, pct[0]))
    print(f"# {workdir}/{pct[0]}   terminal_charge_threshold = {cut:.0f} e")

    b = dump.find("live/pointclouds/namedpcs/3d/arrays/x").rsplit("/", 1)[0] + "/"
    P = np.stack([dump.get(b + "x_t0cor"), dump.get(b + "y"), dump.get(b + "z")], axis=1)
    t = dump.get(b + "t")
    q = {p: dump.get(b + p + "charge_val").astype(float) for p in "uvw"}
    unc = {p: dump.get(b + p + "charge_unc").astype(float) for p in "uvw"}
    wind = {p: dump.get(b + p + "wire_index") for p in "uvw"}

    # Blob attribution: the flat cloud is a blob-ordered concatenation.
    npoints = dump.arr("live/pointclouds/namedpcs/scalar/arrays/npoints")
    assert npoints.sum() == len(P), (
        f"3d cloud ({len(P)}) is not the concatenation of {len(npoints)} blobs "
        f"({npoints.sum()}) -- the blob-order assumption no longer holds")
    blob_of = np.repeat(np.arange(len(npoints)), npoints)
    slice_of = (t / TICK_NS).astype(int) // REBIN * REBIN

    regions = [("BELOW V (V->A, 0 steiner pts)", A_CM),
               ("ABOVE V (V->U, 632 steiner pts)", U_CM)]

    # ---- 1. cluster ownership, from the Bee layer, in Bee's own coordinates
    zpath = os.path.join(workdir, "mabc-pr.zip")
    if os.path.exists(zpath):
        bee = json.loads(zipfile.ZipFile(zpath).read("data/0/0-clustering-global.json"))
        B = np.stack([np.array(bee["y"]), np.array(bee["z"])], axis=1)   # cm
        rcid = np.array(bee["real_cluster_id"])
        print("\n## 1. Who owns the charge (Bee real_cluster_id, geometric selection)")
        for name, end in regions:
            m = seg_mask(B, V_CM[[1, 2]], end[[1, 2]], tol=TOL_MM / 10.0)
            n = int(m.sum())
            top = Counter(rcid[m].tolist()).most_common(3)
            share = ", ".join(f"{k}: {v} ({100.0*v/n:.0f}%)" for k, v in top) if n else "-"
            print(f"  {name:34s} n={n:5d}  {share}")

    # ---- 2/3/4. sampling, candidacy, blob occupancy
    print("\n## 2-4. Sampling, candidacy and blob occupancy (the load-bearing table)")
    print(f"  {'region':34s} {'pts':>5s} {'blobs':>6s} {'cand':>6s} {'cand%':>6s} {'cand blobs':>11s}")
    sel = {}
    for name, end in regions:
        m = seg_mask(P, V_CM * 10.0, end * 10.0)
        sel[name] = m
        idx = np.where(m)[0]
        quality, rms = calc_charge_wcp(q["u"][m], q["v"][m], q["w"][m], cut)
        cand = quality & (rms > cut)
        nb = len(set(blob_of[idx].tolist()))
        nbc = len(set(blob_of[idx[cand]].tolist()))
        print(f"  {name:34s} {len(idx):5d} {nb:6d} {int(cand.sum()):6d} "
              f"{100.0*cand.mean() if len(idx) else 0:5.1f}% {nbc:11d}")
    print("  find_steiner_terminals emits >=1 terminal per candidate-bearing blob")
    print("  (SteinerGrapher.cxx:605-622), so the last column is a LOWER BOUND on")
    print("  the terminals Phase 1 produced in each region.")

    # ---- 5. per-plane cross-check against the 2-D charge map
    print("\n## 5. Per-point charge vs the 2-D (wire, slice) map")
    ctrl = sel["ABOVE V (V->U, 632 steiner pts)"]
    probe = np.where(ctrl & (q["v"] > 0))[0]
    apa, face, hit, ntot = identify_ctpc(dump, slice_of, wind["v"], q["v"], probe)
    print(f"  ctpc dataset identified by charge-matching: a{apa}f{face}  ({hit}/{ntot} exact)")
    if hit != ntot or ntot == 0:
        print("  !! calibration FAILED -- refusing to report cross-referenced numbers")
        return
    maps = {p: load_ctpc(dump, apa, face, p.upper()) for p in "uvw"}
    for name, _ in regions:
        idx = np.where(sel[name])[0]
        print(f"  {name}: n={len(idx)}")
        for p in "uvw":
            cmap = maps[p]
            exact = missing = zero_but_present = 0
            present = []
            for i in idx:
                c = cmap.get((int(slice_of[i]), int(wind[p][i])))
                if c is None:
                    missing += 1
                elif abs(c - q[p][i]) < 1e-3:
                    exact += 1
                elif q[p][i] == 0:
                    zero_but_present += 1
                    present.append(c)
            med = f"{np.median(present):.0f}" if present else "-"
            print(f"     {p.upper()}: nonzero {np.mean(q[p][idx] != 0):.3f} | "
                  f"exact {exact:4d}/{len(idx)} | absent from map {missing:3d} | "
                  f"stored 0 BUT map has charge {zero_but_present:4d} (median {med} e) | "
                  f"unc==0 {np.mean(unc[p][idx] == 0):.3f} unc>1e10 {np.mean(unc[p][idx] > 1e10):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pdvd/31 section 7: feasibility of an APERTURE-MATCHED per-point charge
for Steiner terminal selection, measured on real PDVD and SBND data.

The question the owner posed: the current terminal criterion reads, per plane,
the charge of the ONE wire nearest the sampled 3D point
(BlobSampler.cxx:315-370 -> ucharge_val/vcharge_val/wcharge_val, read back by
Facade_Cluster.cxx:1031-1112).  Would combining NEARBY WIRES instead -- an
aperture whose half-width is a fixed PHYSICAL distance rather than one wire --
reduce the wire/strip mismatch that doc 28 measured?

Doc 28's mechanism, restated: PDVD's coarser pitch leaves the third view
under-constrained, so BlobSampler's `stepped` strategy emits a median of 4
candidate 3D points per (U,V) wire crossing (84% of crossings ambiguous) vs
SBND's median of 1 (21%).  Only one of them is the true deposit; the rest are
"losing" candidates that see a fraction of the charge and fail the three-plane
AND gate at roughly half the rate of the "peak" candidate.  If a physical
aperture is the right estimator, the peak-vs-losing pass-rate gap should NARROW,
because both members of an ambiguous group sit within one aperture of the same
charge.

What is measured, per detector:

  * the CURRENT estimator: single snapped wire (the stored *charge_val)
  * an APERTURE estimator: charge summed / averaged over every 2-D (wire, slice)
    cell within +-APER_MM of the point, read from the ctpc_a<A>f<F>p<P> maps.
    The window is converted to wires and slices per plane from that detector's
    own pitch and slice width, so "1 cm" means 1 cm on both detectors -- which
    is the whole point: the current 4000 e constant does NOT mean the same thing
    on a 3 mm and a 7.65 mm pitch.

and, under each estimator, the doc 28 quantities: candidacy rate, the fraction
of points that are losing candidates, and the peak-vs-losing pass rates.

NOTE on scope (doc 31 section 4): this feasibility test is motivated by doc 28's
POPULATION measurement, not by 039349/14.  On that event the terminal criterion
is demonstrably not the binding constraint -- see
steiner_terminal_attribution.py.  Nothing here claims to fix it.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/steiner_aperture_feasibility.py \
      work/039252_2_stm1/pctree-evt298595.tar.gz 500
  python3 docs/nf_sp_img_clus/scripts/steiner_aperture_feasibility.py \
      /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-dbg25a-d97off/ql_evt16/pctree-evt16.tar.gz 4000
"""
import io
import json
import sys
import tarfile
from collections import defaultdict

import numpy as np

TICK_NS = 500.0          # 0.5 us DAQ tick, both detectors
REBIN = 4                # imaging rebin, both detectors (doc 28 section 3)
APER_MM = 10.0           # aperture half-width, PHYSICAL, same on both detectors

# per-detector geometry, only used to turn APER_MM into a wire/slice count.
# pitches: doc 28 section 2 (wirecell-util wires-info); slice width = rebin *
# tick * v_drift (PDVD 1.48073 mm/us production, SBND 1.6 mm/us nominal).
GEOM = {
    "pdvd": {"pitch": {"U": 7.65, "V": 7.65, "W": 5.10}, "slice_mm": 2.0 * 1.48073},
    "sbnd": {"pitch": {"U": 3.00, "V": 3.00, "W": 3.00}, "slice_mm": 2.0 * 1.6},
}


class Dump:
    def __init__(self, path):
        self.tar = tarfile.open(path)
        self.index = {}
        for name in self.tar.getnames():
            if not name.endswith("_metadata.json"):
                continue
            dp = json.loads(self.tar.extractfile(name).read()).get("datapath", "")
            if dp:
                self.index[dp] = name.replace("_metadata.json", "_array.npy")

    def find(self, suffix):
        hits = [dp for dp in self.index if dp.endswith(suffix)]
        if len(hits) != 1:
            raise KeyError(f"{suffix!r} matched {len(hits)} datapaths")
        return hits[0]

    def get(self, dp):
        return np.load(io.BytesIO(self.tar.extractfile(self.index[dp]).read())).ravel()

    def arr(self, suffix):
        return self.get(self.find(suffix))


def calc_charge_wcp(planes, cut):
    """Cluster::calc_charge_wcp(..., disable_dead_mix_cell=false), vectorised.
    Facade_Cluster.cxx:1087-1109.  Returns (quality, rms)."""
    n_pt = len(planes[0])
    quality = np.ones(n_pt, dtype=bool)
    acc = np.zeros(n_pt)
    n = np.zeros(n_pt, dtype=int)
    for q in planes:
        quality &= (q > cut) | (q == 0)
        nz = q != 0
        acc += np.where(nz, q * q, 0.0)
        n += nz
    rms = np.where(n > 1, np.sqrt(np.divide(acc, np.maximum(n, 1))), 0.0)
    return quality, rms


def ambiguity_masks(uw, vw, qw_ref):
    """doc 28: group sampled points by their (U,V) wire crossing.  Within a
    group of >1 the max-W-charge point is the 'peak' (true) candidate and the
    rest are 'losing' candidates."""
    n = len(uw)
    groups = defaultdict(list)
    key = uw.astype(np.int64) * 100000 + vw.astype(np.int64)
    for i, k in enumerate(key.tolist()):
        groups[k].append(i)
    peak = np.zeros(n, bool)
    amb = np.zeros(n, bool)
    sizes = []
    for g in groups.values():
        sizes.append(len(g))
        if len(g) > 1:
            amb[g] = True
            peak[g[int(np.argmax(qw_ref[g]))]] = True
        else:
            peak[g[0]] = True
    return peak, amb & ~peak, np.array(sizes)


def main():
    path = sys.argv[1]
    cut = float(sys.argv[2]) if len(sys.argv) > 2 else 4000.0
    det = "sbnd" if "sbnd" in path else "pdvd"
    geom = GEOM[det]
    dump = Dump(path)

    b = dump.find("live/pointclouds/namedpcs/3d/arrays/x").rsplit("/", 1)[0] + "/"
    t = dump.get(b + "t")
    wpid = dump.get(b + "wpid")
    wind = {p: dump.get(b + p + "wire_index") for p in "uvw"}
    qval = {p: dump.get(b + p + "charge_val").astype(float) for p in "uvw"}
    n_pt = len(t)
    slice_of = (t / TICK_NS).astype(int) // REBIN * REBIN

    # wpid packs layer | face<<3 | apa<<4  (iface/src/WirePlaneId.cxx:5-7,35-36)
    apa_of = (wpid >> 4).astype(int)
    face_of = ((wpid >> 3) & 1).astype(int)

    nw = {p: int(np.ceil(APER_MM / geom["pitch"][p.upper()])) for p in "uvw"}
    ns = int(np.ceil(APER_MM / geom["slice_mm"]))
    print(f"# {path}")
    print(f"# detector={det}  n_points={n_pt}  terminal cut={cut:.0f} e")
    print(f"# aperture +-{APER_MM:.0f} mm  ->  wires +-{nw['u']}/{nw['v']}/{nw['w']} (U/V/W), "
          f"slices +-{ns} ({geom['slice_mm']:.2f} mm each)")

    # ---- build the aperture estimate from the 2-D (wire, slice) charge maps
    qsum = {p: np.zeros(n_pt) for p in "uvw"}
    qcnt = {p: np.zeros(n_pt) for p in "uvw"}
    exact = {p: 0 for p in "uvw"}
    for (apa, face) in sorted(set(zip(apa_of.tolist(), face_of.tolist()))):
        sel = np.where((apa_of == apa) & (face_of == face))[0]
        if not len(sel):
            continue
        for p in "uvw":
            try:
                base = dump.find(f"ctpc_a{apa}f{face}p{p.upper()}/arrays/charge").rsplit("/", 1)[0] + "/"
            except KeyError:
                continue
            cw = dump.get(base + "wind").tolist()
            cs = dump.get(base + "slice_index").tolist()
            cc = dump.get(base + "charge").tolist()
            cmap = {}
            for w_, s_, c_ in zip(cw, cs, cc):
                cmap[(s_, w_)] = c_
            R, S = nw[p], ns
            for i in sel:
                s0 = int(slice_of[i]); w0 = int(wind[p][i])
                c0 = cmap.get((s0, w0))
                if c0 is not None and abs(c0 - qval[p][i]) < 1e-3:
                    exact[p] += 1
                tot = 0.0; cnt = 0
                for ds in range(-S, S + 1):
                    for dw in range(-R, R + 1):
                        c = cmap.get((s0 + ds * REBIN, w0 + dw))
                        if c is not None:
                            tot += c; cnt += 1
                qsum[p][i] = tot
                qcnt[p][i] = cnt

    # sanity: the single-wire lookup must reproduce the stored value, else the
    # (apa,face) attribution or the slice mapping is wrong (trap 4 of
    # steiner_terminal_attribution.py).
    print("# single-wire cross-check (stored *charge_val == ctpc at own cell): "
          + "  ".join(f"{p.upper()} {exact[p]}/{int((qval[p] != 0).sum())} nonzero" for p in "uvw"))

    qmean = {p: np.where(qcnt[p] > 0, qsum[p] / np.maximum(qcnt[p], 1), 0.0) for p in "uvw"}

    peak, losing, sizes = ambiguity_masks(wind["u"], wind["v"], qval["w"])
    print(f"# (U,V) crossings: {len(sizes)}  median candidates/crossing {np.median(sizes):.0f}  "
          f"ambiguous {np.mean(sizes > 1):.3f}  losing fraction of all points {losing.mean():.3f}")

    print(f"\n  {'estimator':28s} {'floor':>10s} {'cand':>7s} {'peak':>7s} {'losing':>7s} {'ratio':>6s}")
    def report(name, planes, floor):
        quality, rms = calc_charge_wcp([planes[p] for p in "uvw"], floor)
        cand = quality & (rms > floor)
        pk = cand[peak].mean() if peak.any() else float("nan")
        ls = cand[losing].mean() if losing.any() else float("nan")
        print(f"  {name:28s} {floor:10.0f} {cand.mean():7.3f} {pk:7.3f} {ls:7.3f} "
              f"{(ls/pk if pk else float('nan')):6.2f}")

    report("single wire (production)", qval, cut)
    # relative floors: a fraction of the event's own median nonzero estimate, so
    # the number means the same thing on both detectors (doc 31 section 6b).
    for frac in (0.2, 0.5):
        med = np.median(np.concatenate([qsum[p][qsum[p] > 0] for p in "uvw"]))
        report(f"aperture SUM (rel {frac:.1f})", qsum, frac * med)
    for frac in (0.2, 0.5):
        med = np.median(np.concatenate([qmean[p][qmean[p] > 0] for p in "uvw"]))
        report(f"aperture MEAN (rel {frac:.1f})", qmean, frac * med)
    print("  ratio = losing/peak pass rate; -> 1.00 means the aperture has removed")
    print("  the wire-crossing ambiguity's effect on candidacy (doc 28 section 4.3).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pdvd/31 section 5.1: the per-point induction charge is silently ZERO on
the second segment of every wrapped strip, in half the (apa, face, plane) groups.

PDVD's induction strips are long enough to cross the CRU boundary at
y = +-1685 mm, so 1568 of 12288 channels carry TWO wires ("segments"), one in
each face of the same anode; the two lengths sum to exactly 1720.04 mm.  W is
never wrapped.

BlobSampler stamps each sampled 3D point with the charge of the single nearest
wire per plane (BlobSampler.cxx:343-357):

    IWire::pointer iwire = iwires[wire_index[ipt]];
    channel_ident[ipt]   = iwire->channel();
    channel_attach[ipt]  = p_chi2i[channel_ident[ipt]];   // <-- operator[]
    auto ich             = channels[channel_attach[ipt]];
    auto ait             = activity.find(ich);
    if (ait != activity.end()) { charge_val = ...; charge_unc = ...; }

`p_chi2i` is an unordered_map<int,int> built from THIS plane's channel list, and
`operator[]` on a missing key INSERTS 0 and returns 0 -- so a channel the plane
does not list silently resolves to `channels[0]`, whose activity is normally
absent, leaving BOTH charge_val and charge_unc at exactly 0.  That is
indistinguishable downstream from "this plane saw no signal":
Cluster::calc_charge_wcp (Facade_Cluster.cxx:1087-1091) treats a zero plane as
passing, so the point keeps a healthy two-plane RMS and no warning is emitted.

This script measures the correlation without assuming the mechanism:

  * `charge_val == 0` vs. whether the point's wire is segment 0 or segment 1,
    per plane and per (apa, face) -- the discriminating test;
  * the same split for the two halves of the 039349/14 track, where the effect
    was first seen.

Result on 039349/14 (see the doc): the split is DETERMINISTIC per
(apa, face, plane) -- P(charge==0 | segment==1) is 1.000 for anodes 4-7 and
~0 for anodes 0-3 -- so it is a geometry/bookkeeping bug, not a charge effect.

NOTE on the wires file: its `faces`, `planes` and `wires` fields reference each
other by ARRAY INDEX.  The `ident` fields are LOCAL and repeat (only 292 unique
wire idents among 13856 wires, face idents are just 0 and 1), so building
dicts keyed on `ident` silently collapses the geometry -- which is exactly the
wrong answer this script was first given.

Usage:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/steiner_wrapped_channel_census.py \
      work/039349_14_d27fresh/pctree-evt19689.tar.gz
"""
import bz2
import io
import json
import os
import sys
import tarfile
from collections import Counter

import numpy as np

WIRES = os.environ.get(
    "PDVD_WIRES",
    "/home/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v7-uvwfit.json.bz2")

# doc 26 section 7.5, cm.  V = where the steiner cloud stops; A / U = the two ends.
V_CM = np.array([273.26, -118.90, 86.61])
A_CM = np.array([196.86, -167.76, 151.48])
U_CM = np.array([392.0, -83.0, 5.0])


def load_geometry(path):
    """-> {(apa, face, plane): (segment[], channel[])} indexed by LOCAL wire index."""
    store = json.loads(bz2.open(path).read())["Store"]
    wires = [w["Wire"] for w in store["wires"]]
    planes = [p["Plane"] for p in store["planes"]]
    faces = [f["Face"] for f in store["faces"]]
    anodes = [a["Anode"] for a in store["anodes"]]
    nseg = Counter(w["channel"] for w in wires)
    out = {}
    for a in anodes:
        for fi, fidx in enumerate(a["faces"]):
            if fidx is None or fidx < 0:
                continue
            for pi, pidx in enumerate(faces[fidx]["planes"]):
                wl = planes[pidx]["wires"]
                out[(a["ident"], fi, pi)] = (
                    np.array([wires[i]["segment"] for i in wl]),
                    np.array([wires[i]["channel"] for i in wl]),
                )
    return out, nseg


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
            raise KeyError(f"{suffix!r} matched {len(hits)}")
        return hits[0]

    def get(self, dp):
        return np.load(io.BytesIO(self.tar.extractfile(self.index[dp]).read())).ravel()


def seg_of(points_wire_index, apa, face, plane, geom):
    """Per-point segment number (-1 where unknown)."""
    n = len(points_wire_index)
    out = np.full(n, -1)
    for (a, f) in sorted(set(zip(apa.tolist(), face.tolist()))):
        key = (a, f, plane)
        if key not in geom:
            continue
        m = (apa == a) & (face == f)
        segs = geom[key][0]
        idx = points_wire_index[m]
        good = (idx >= 0) & (idx < len(segs))
        sub = np.full(int(m.sum()), -1)
        sub[good] = segs[idx[good]]
        out[m] = sub
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "work/039349_14_d27fresh/pctree-evt19689.tar.gz"
    geom, nseg = load_geometry(WIRES)
    nwrapped = sum(1 for v in nseg.values() if v == 2)
    print(f"# wires file: {os.path.basename(WIRES)}")
    print(f"# channels {len(nseg)}, wrapped (2 segments) {nwrapped}")

    d = Dump(path)
    b = d.find("live/pointclouds/namedpcs/3d/arrays/x").rsplit("/", 1)[0] + "/"
    P = np.stack([d.get(b + "x_t0cor"), d.get(b + "y"), d.get(b + "z")], axis=1)
    wpid = d.get(b + "wpid")
    apa = (wpid >> 4).astype(int)              # iface/src/WirePlaneId.cxx:5-7,35-36
    face = ((wpid >> 3) & 1).astype(int)
    print(f"# {path}: {len(P)} points")

    seg = {}
    zero = {}
    for pl, letter in ((0, "u"), (1, "v"), (2, "w")):
        wi = d.get(b + letter + "wire_index")
        q = d.get(b + letter + "charge_val").astype(float)
        seg[letter] = seg_of(wi, apa, face, pl, geom)
        zero[letter] = (q == 0)

    print("\n## Event-wide: P(charge_val == 0) by wire segment")
    print(f"  {'plane':6s} {'segment':>8s} {'n points':>9s} {'P(val==0)':>10s}")
    for letter in "uvw":
        for s in (0, 1):
            m = seg[letter] == s
            if not m.any():
                continue
            print(f"  {letter.upper():6s} {s:8d} {int(m.sum()):9d} {zero[letter][m].mean():10.3f}")

    print("\n## Per (apa, face, plane), for segment-1 wires (>=20 points)")
    print(f"  {'apa':>3s} {'face':>4s} {'plane':>5s} {'n seg=1':>8s} {'P(val==0)':>10s}")
    rows = []
    for pl, letter in ((0, "u"), (1, "v")):
        for a in sorted(set(apa.tolist())):
            for f in (0, 1):
                m = (apa == a) & (face == f) & (seg[letter] == 1)
                if int(m.sum()) < 20:
                    continue
                rows.append((a, f, letter.upper(), int(m.sum()), float(zero[letter][m].mean())))
    for r in sorted(rows, key=lambda r: (-r[4], -r[3])):
        print(f"  {r[0]:3d} {r[1]:4d} {r[2]:>5s} {r[3]:8d} {r[4]:10.3f}")
    det = sum(1 for r in rows if r[4] > 0.999)
    print(f"  -> {det} of {len(rows)} groups are DETERMINISTIC (P == 1.000): a bookkeeping")
    print("     property of the (apa, face, plane), not a charge fluctuation.")

    print("\n## The two halves of the 039349/14 track (V-plane)")
    for name, end in (("BELOW V (starved)", A_CM), ("ABOVE V (control)", U_CM)):
        dvec = (end - V_CM) * 10.0
        a0 = V_CM * 10.0
        t = np.clip(((P - a0) @ dvec) / (dvec @ dvec), 0, 1)
        m = (np.linalg.norm(P - (a0 + t[:, None] * dvec), axis=1) < 30.0) & (t > 0.02) & (t < 0.98)
        s1 = (seg["v"] == 1) & m
        z = zero["v"] & m
        print(f"  {name:20s} n={int(m.sum()):5d}  segment==1: {int(s1.sum()):5d} "
              f"({s1.sum()/max(int(m.sum()),1):.3f})  charge==0: {int(z.sum()):5d} "
              f"({z.sum()/max(int(m.sum()),1):.3f})  agreement: {float((s1 == z)[m].mean()):.3f}")


if __name__ == "__main__":
    main()

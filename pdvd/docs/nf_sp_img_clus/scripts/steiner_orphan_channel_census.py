#!/usr/bin/env python3
"""doc pdvd/31 section 5.2: which (apa, face, plane) groups have channels that
`IWirePlane::channels()` cannot resolve -- the exact blast radius of the
BlobSampler wrapped-segment charge bug.

MECHANISM (read off the source, not inferred).  Gen::AnodePlane::configure
builds each plane's channel vector by walking that plane's wires and SKIPPING
every wrapped continuation (AnodePlane.cxx:244-247):

    for (auto w : wires) {
        if (w->segment() > 0) { continue; }        // <-- here
        ...
        plane_channels.push_back(ich);
    }

That is correct as a *channel list* -- each channel appears once per anode,
attached to the plane holding its segment-0 wire.  It is NOT a wire->channel
lookup table.  BlobSampler builds `p_chi2i` from it (BlobSampler.cxx:302-312)
and then does `p_chi2i[channel_ident]` -- `unordered_map::operator[]`, which
INSERTS 0 on a miss -- so a wire whose channel this plane does not list
silently resolves to `channels[0]`.

So the defect fires exactly on wires that are ORPHANS in their own plane:

    segment > 0  AND  channel not carried by any segment-0 wire of the SAME plane.

A wrapped strip whose two segments both live in one plane is NOT an orphan --
its channel is listed via the segment-0 wire and the lookup succeeds.  That is
why the census in section 5.1 found P(charge==0 | segment==1) to be
deterministic per (apa, face, plane) rather than uniform: the groups differ in
whether the wrap stays inside the plane or crosses to the sibling face.

This script computes the orphan set per plane straight from a wires file, for
every detector we build, so the blast radius is measured and not assumed.

Usage:
  python3 docs/nf_sp_img_clus/scripts/steiner_orphan_channel_census.py [wires.json.bz2 ...]
  (no arguments: the production wires file of every detector this tree builds)
"""
import bz2
import json
import os
import sys
from collections import Counter

DATA = "/home/xqian/toolkit-dev/wire-cell-data"

# The production wires file each detector's anode config actually names.
DEFAULT = [
    ("pdvd", "protodunevd-wires-larsoft-v7-uvwfit.json.bz2"),
    ("pdhd", "protodunehd-wires-larsoft-v1.json.bz2"),
    ("sbnd", "sbnd-wires-geometry-v0206.json.bz2"),
    ("uboone", "microboone-celltree-wires-v2.1.json.bz2"),
]

PLANE = "UVW"


def load(path):
    """-> list of (apa_ident, face_index, plane_index, [(segment, channel), ...])

    NOTE: the wires JSON references faces/planes/wires by ARRAY INDEX.  The
    `ident` fields are LOCAL and repeat (PDVD: 292 unique wire idents among
    13856; face idents are only 0 and 1), so a dict keyed on `ident` silently
    collapses the geometry.  Index by position, read `ident` only for labels.
    """
    store = json.loads(bz2.open(path).read())["Store"]
    wires = [w["Wire"] for w in store["wires"]]
    planes = [p["Plane"] for p in store["planes"]]
    faces = [f["Face"] for f in store["faces"]]
    anodes = [a["Anode"] for a in store["anodes"]]
    out = []
    for a in anodes:
        for fi, fidx in enumerate(a["faces"]):
            if fidx is None or fidx < 0:
                continue
            for pi, pidx in enumerate(faces[fidx]["planes"]):
                wl = planes[pidx]["wires"]
                out.append((a["ident"], fi, pi,
                            [(wires[i]["segment"], wires[i]["channel"]) for i in wl]))
    return out


def report(tag, path):
    if not os.path.exists(path):
        print(f"## {tag}: {os.path.basename(path)} -- NOT PRESENT, skipped")
        return
    groups = load(path)
    nwires = sum(len(w) for _, _, _, w in groups)
    allchan = Counter(c for _, _, _, w in groups for _, c in w)
    wrapped = sum(1 for v in allchan.values() if v > 1)
    print(f"## {tag}: {os.path.basename(path)}")
    print(f"   {len(groups)} (apa,face,plane) groups, {nwires} wires, "
          f"{len(allchan)} channels, {wrapped} multi-segment")

    rows = []
    for apa, face, plane, w in groups:
        listed = {c for s, c in w if s == 0}          # what plane_channels holds
        orphan = [c for s, c in w if s > 0 and c not in listed]
        nseg1 = sum(1 for s, _ in w if s > 0)
        if nseg1 or orphan:
            rows.append((apa, face, plane, len(w), nseg1, len(orphan)))
    if not rows:
        print("   no wrapped wires anywhere: this detector CANNOT be affected.\n")
        return
    print(f"   {'apa':>4s} {'face':>4s} {'pl':>3s} {'wires':>6s} {'seg>0':>6s} "
          f"{'ORPHAN':>7s}  (orphan = seg>0 and channel not listed by this plane)")
    tot_o = 0
    for r in sorted(rows, key=lambda r: (-r[5], r[0], r[1], r[2])):
        tot_o += r[5]
        print(f"   {r[0]:4d} {r[1]:4d} {PLANE[r[2]]:>3s} {r[3]:6d} {r[4]:6d} {r[5]:7d}")
    naff = sum(1 for r in rows if r[5])
    print(f"   -> {naff} of {len(groups)} groups carry orphans; "
          f"{tot_o} orphan wires ({tot_o/nwires:.1%} of all wires).")
    print("      Every sampled point landing on one of these gets channels[0]'s "
          "charge\n      (usually absent -> 0.0, but a WRONG NON-ZERO value when "
          "channels[0]\n      happens to be live in that slice).\n")


def main():
    args = sys.argv[1:]
    if args:
        for p in args:
            report(os.path.basename(p), p)
    else:
        for tag, name in DEFAULT:
            report(tag, os.path.join(DATA, name))


if __name__ == "__main__":
    main()

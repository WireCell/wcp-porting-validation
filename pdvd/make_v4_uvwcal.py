#!/usr/bin/env python
"""RECIPE: produce a U/V-vs-W-corrected PDVD wire file from v4.

Keeps U and V untouched; rigidly shifts each CRP's W (collection) plane along z
by the per-CRP offset measured in pdvd-uvw-wire-offset-calibration.md:
    bottom CRP (anodes 0-3): dZ_W = -13.2 mm
    top    CRP (anodes 4-7): dZ_W =  +9.8 mm
Writes a new wire file; does NOT touch the shipped v4.
"""
import sys
from wirecell.util.wires import persist

SRC = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v4.json.bz2"
DST = sys.argv[1] if len(sys.argv) > 1 else "/home/xqian/tmp/protodunevd-wires-larsoft-v4-uvwcal.json.bz2"
DZ = {0: -13.2, 1: -13.2, 2: -13.2, 3: -13.2,     # bottom CRP
      4: +9.8,  5: +9.8,  6: +9.8,  7: +9.8}        # top CRP

store = persist.load(SRC)

# collect z-shift for every point referenced by a W (collection, plane.ident==2) wire
pt_dz = {}
for anode in store.anodes:
    dz = DZ[anode.ident]
    for fi in anode.faces:
        for pi in store.faces[fi].planes:
            pl = store.planes[pi]
            if pl.ident != 2:            # only the W collection plane
                continue
            for wi in pl.wires:
                w = store.wires[wi]
                pt_dz[w.tail] = dz
                pt_dz[w.head] = dz

pts = list(store.points)
for idx, dz in pt_dz.items():
    pts[idx] = pts[idx]._replace(z=pts[idx].z + dz)

persist.dump(DST, persist.todict(store._replace(points=pts)))
print(f"wrote {DST}  ({len(pt_dz)} W points shifted)")

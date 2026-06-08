#!/usr/bin/env python
"""RECIPE: produce the U/V-vs-W-corrected PDVD v5 wire file from v4.

Keeps the W (collection) plane UNTOUCHED; shifts the U and V induction wire
endpoints along the W pitch direction (pure +/-z) by the per-CRP offset measured
in pdvd-uvw-wire-offset-calibration.md so that the U/\\cap V crossing lands on the
hit W wire:

    crossing today sits at  z_cross = z_W + dZ_W   (dZ_W = median mismatch)
    a rigid z-translation t of BOTH U and V lines moves the crossing by exactly t,
    so shifting U,V by  dz = -dZ_W  nulls the mismatch.

    bottom CRP (anodes 0-3): dZ_W = -13.2 mm  ->  shift U,V by  +13.2 mm in z
    top    CRP (anodes 4-7): dZ_W =  +9.8 mm  ->  shift U,V by   -9.8 mm in z

W is collection (vertical wires along y) so its pitch direction is exactly z;
"along the W pitch direction" therefore means a pure z shift.  Writes v5; does
NOT touch the shipped v3/v4.
"""
import sys
from wirecell.util.wires import persist

SRC = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v4.json.bz2"
DST = sys.argv[1] if len(sys.argv) > 1 else \
    "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2"

# dz applied to U and V wire endpoints (pure z = W pitch direction), per CRP.
DZ_UV = {0: +13.2, 1: +13.2, 2: +13.2, 3: +13.2,     # bottom CRP
         4: -9.8,  5: -9.8,  6: -9.8,  7: -9.8}        # top CRP

store = persist.load(SRC)

# collect z-shift for every point referenced by a U or V (induction) wire.
# plane.ident: 0=U, 1=V, 2=W -> shift only ident 0,1 (keep W=2 fixed).
pt_dz = {}
for anode in store.anodes:
    dz = DZ_UV[anode.ident]
    for fi in anode.faces:
        for pi in store.faces[fi].planes:
            pl = store.planes[pi]
            if pl.ident not in (0, 1):       # only U and V induction planes
                continue
            for wi in pl.wires:
                w = store.wires[wi]
                pt_dz[w.tail] = dz
                pt_dz[w.head] = dz

pts = list(store.points)
for idx, dz in pt_dz.items():
    pts[idx] = pts[idx]._replace(z=pts[idx].z + dz)

persist.dump(DST, persist.todict(store._replace(points=pts)))
print(f"wrote {DST}  ({len(pt_dz)} U/V points shifted)")

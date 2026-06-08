#!/usr/bin/env python
"""RECIPE: produce the U/V-vs-W-corrected PDVD v5 wire file from v4.

Keeps the W (collection) plane UNTOUCHED; shifts the U and V induction wire
endpoints along the W pitch direction (pure +/-z) so the U/\\cap V charge crossing
lands on the hit W wire:

    crossing today sits at  z_cross = z_W + dZ_W   (dZ_W = median mismatch)
    a rigid z-translation t of BOTH U and V lines moves the crossing by exactly t,
    so shifting U,V by  dz = -dZ_W  nulls the mismatch.

The shift is assigned BY REGISTRATION TYPE, not by CRP, to RESPECT the cathode
mirror symmetry of the first-wire-offset table (pdvd-wire-geometry-v3-v4.md S1):
the table groups the 8 anodes into two types, each present once per CRP, with
top = bottom + U<->V swap.  Because the cathode mirror is x->-x (z preserved) and
the correction is a common-mode (equal U,V) z-shift, mirror partners take the
SAME signed z-shift.  Measured on one track per type:

    type A  (anodes 0,2,5,7): dZ_W = -13.2 mm  ->  shift U,V by  +13.2 mm   (from anode 0)
    type B  (anodes 1,3,4,6): dZ_W =  +9.8 mm  ->  shift U,V by   -9.8 mm   (from anode 4)

anodes {0,2} and {5,7} are type-A mirror partners; {1,3} and {4,6} are type-B
mirror partners.  W is collection (vertical wires along y) so its pitch direction
is exactly z; "along the W pitch direction" means a pure z shift.  Writes v5; does
NOT touch the shipped v3/v4.
"""
import sys
from wirecell.util.wires import persist

SRC = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v4.json.bz2"
DST = sys.argv[1] if len(sys.argv) > 1 else \
    "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2"

# dz applied to U and V wire endpoints (pure z = W pitch direction), BY TYPE.
# type A {0,2,5,7} = +13.2 (measured anode 0);  type B {1,3,4,6} = -9.8 (measured anode 4)
DZ_UV = {0: +13.2, 2: +13.2, 5: +13.2, 7: +13.2,     # type A (mirror partners)
         1: -9.8,  3: -9.8,  4: -9.8,  6: -9.8}        # type B (mirror partners)

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

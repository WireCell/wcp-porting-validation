#!/usr/bin/env python3
"""doc 96 sec 9 -- feasibility probe 2: the PDHD/PDVD separation operating point on SBND.

clus/docs/clustering-separate-fv.md diagnoses exactly the sec 5.2 blindness on
PDHD/PDVD and ships a fix set: 15 cm y/z FV insets (config-only) plus the
knobs far_point_x_cut / far_point_mid_dis / dec1_guard_main_angle /
track_recarve.  Its own text says "SBND is untouched (its FV was already
inset)" -- which sec 6.1 measures to be false.  This probe applies that
operating point to SBND events by patching the event's own PRODUCTION compiled
config, so the difference from production is exactly the listed keys.

NOTE this is a feasibility probe, not a proposal: the FV in DetectorVolumes is
shared, via select_scope_fv, with clustering_neutrino and the containment
taggers, so an inset here moves more than separation.  That coupling is
precisely why sec 8 records it as owner-decision territory.

Values are WCT internal units (mm).  Writes <out>/pdhdlike/evt<E>.json.
"""
import argparse
import json
import os

AP = argparse.ArgumentParser()
AP.add_argument("--src", default="work-dbg25a-ql")
AP.add_argument("--out", default="/home/xqian/tmp/d96/recarve")
AP.add_argument("--inset", type=float, default=150.0, help="y/z FV inset, mm")
AP.add_argument("events", nargs="+")
A = AP.parse_args()

ARM = "pdhdlike"
for evt in A.events:
    nodes = json.load(open(os.path.join(A.src, f"ql_evt{evt}", f".wct-cfg-evt{evt}.json")))
    d = json.loads(json.dumps(nodes))
    outdir = os.path.join(A.out, ARM, f"ql_evt{evt}")
    os.makedirs(outdir, exist_ok=True)
    nfv = nsep = nout = 0
    for n in d:
        t, data = n.get("type", ""), n.get("data")
        if not isinstance(data, dict):
            continue
        if t == "DetectorVolumes":
            for _blk, md in data.get("metadata", {}).items():
                if not isinstance(md, dict):
                    continue
                for k, sign in (("FV_ymin", +1), ("FV_ymax", -1),
                                ("FV_zmin", +1), ("FV_zmax", -1)):
                    if k in md and md[k] is not None:
                        md[k] = md[k] + sign * A.inset
                        nfv += 1
        if t == "ClusteringSeparate":
            data["track_recarve"] = True
            data["far_point_x_cut"] = 140.0        # 14 cm, the evidently intended cut
            data["far_point_mid_dis"] = 600.0      # 60 cm, as PDHD/PDVD
            data["dec1_guard_main_angle"] = 45.0
            nsep += 1
        for k in ("bee_zip", "outname"):
            v = data.get(k)
            if isinstance(v, str) and f"ql_evt{evt}" in v:
                data[k] = os.path.join(outdir, os.path.basename(v))
                nout += 1
    p = os.path.join(A.out, ARM, f"evt{evt}.json")
    json.dump(d, open(p, "w"))
    print(f"evt{evt}: {nfv} FV field(s) inset by {A.inset/10:.0f} cm, "
          f"{nsep} ClusteringSeparate block(s) reconfigured, {nout} output path(s) -> {p}")

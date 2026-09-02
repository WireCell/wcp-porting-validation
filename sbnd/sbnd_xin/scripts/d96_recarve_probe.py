#!/usr/bin/env python3
"""doc 96 sec 9 -- feasibility probe: would `track_recarve` split the owner's two events?

`track_recarve` (clus/src/clustering_separate.cxx:2187, default OFF, threaded
in cfg/pgrapher/common/clus.jsonnet:1080) is a k=2 3D-line self-split whose
standalone trigger is documented in clus/docs/clustering-separate-fv.md as
targeting exactly the topology sec 5.1 describes: "Two crossing tracks whose
arms end INSIDE the volume expose at most one surface contact, so neither
Dec_2 nor the angle ladders can ever fire on them."  PDHD/PDVD enable it;
SBND does not.

Rather than argue from the gate constants, run it.  This patches the event's
own PRODUCTION compiled config -- so the ONLY difference from production is
the one added key -- redirects every output path into a fresh directory, and
emits an OFF arm (paths only) plus an ON arm (paths + track_recarve).  The OFF
arm doubles as the proof that the path redirect alone changes nothing.

Writes <out>/evt<E>/{off,on}.json ready for `wire-cell -c`.
"""
import argparse
import json
import os

AP = argparse.ArgumentParser()
AP.add_argument("--src", default="work-dbg25a-ql", help="production Q/L work root")
AP.add_argument("--out", default="/home/xqian/tmp/d96/recarve", help="scratch root")
AP.add_argument("events", nargs="+")
A = AP.parse_args()

for evt in A.events:
    cfg = os.path.join(A.src, f"ql_evt{evt}", f".wct-cfg-evt{evt}.json")
    nodes = json.load(open(cfg))
    for arm in ("off", "on"):
        d = json.loads(json.dumps(nodes))
        outdir = os.path.join(A.out, arm, f"ql_evt{evt}")
        os.makedirs(outdir, exist_ok=True)
        nsep = nout = 0
        for n in d:
            t, data = n.get("type", ""), n.get("data")
            if not isinstance(data, dict):
                continue
            if t == "ClusteringSeparate" and arm == "on":
                data["track_recarve"] = True
                nsep += 1
            for k in ("bee_zip", "outname"):
                v = data.get(k)
                if isinstance(v, str) and f"ql_evt{evt}" in v:
                    data[k] = os.path.join(outdir, os.path.basename(v))
                    nout += 1
        p = os.path.join(A.out, arm, f"evt{evt}.json")
        json.dump(d, open(p, "w"))
        print(f"evt{evt} {arm}: {nout} output path(s) redirected, "
              f"{nsep} ClusteringSeparate block(s) given track_recarve -> {p}")

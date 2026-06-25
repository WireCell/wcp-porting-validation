#!/usr/bin/env python3
"""Summarize TaggerCheckTGM debug output from a Bee zip into a .npz.

The TGM debug Bee set is "tgm" (algorithm "tgm-global"); MultiAlgBlobClustering
writes one JSON per event at  data/<idx>/<idx>-tgm-global.json  with arrays
x,y,z (cm), q, cluster_id, real_cluster_id, plus runNo/subRunNo/eventNo.

Only TGM-tagged clusters carry points here.  Endpoints were written with
q == endpoint charge (default 10000), the track body with q == body charge
(default 100).  We cumulate, across ALL events:
  - endpoints   (q >= endpoint_threshold)         -> the box-shaped FV boundary
  - body points (everything else)                 -> the tagged tracks
labelled by a global (event, cluster_id) index so the viewer can colour by track.

Usage:
  summarize_tgm.py <mabc.zip> [-o tgm_points.npz] [--algo tgm-global]
                   [--endpoint-threshold 5000]
"""
import argparse
import json
import os
import re
import zipfile

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", help="Bee zip (e.g. mabc.zip)")
    ap.add_argument("-o", "--out", default="tgm_points.npz")
    ap.add_argument("--algo", default="tgm-global",
                    help="Bee algorithm/type name of the TGM point set")
    ap.add_argument("--endpoint-threshold", type=float, default=5000.0,
                    help="q at/above this is an endpoint (debug_endpoint_charge=10000)")
    args = ap.parse_args()

    # endpoints
    ex, ey, ez, e_evt, e_clid = [], [], [], [], []
    # body
    bx, by, bz, b_evt, b_clid = [], [], [], [], []

    n_events_with_tgm = 0
    n_tracks = 0

    pat = re.compile(r"data/(\d+)/\d+-" + re.escape(args.algo) + r"\.json$")
    with zipfile.ZipFile(args.zip) as zf:
        names = sorted(n for n in zf.namelist() if pat.search(n))
        if not names:
            print("WARNING: no %s JSON files found in %s" % (args.algo, args.zip))
            print("  available (sample):",
                  [n for n in zf.namelist() if n.endswith(".json")][:10])
        for name in names:
            with zf.open(name) as fh:
                d = json.load(fh)
            x = np.asarray(d.get("x", []), dtype=float)
            if x.size == 0:
                continue
            y = np.asarray(d["y"], dtype=float)
            z = np.asarray(d["z"], dtype=float)
            q = np.asarray(d["q"], dtype=float)
            clid = np.asarray(d.get("cluster_id", np.zeros_like(x)), dtype=int)
            # event number: prefer eventNo, fall back to the zip index
            try:
                evt = int(d.get("eventNo"))
            except (TypeError, ValueError):
                evt = int(pat.search(name).group(1))

            n_events_with_tgm += 1
            n_tracks += len(np.unique(clid))

            is_end = q >= args.endpoint_threshold
            ex.extend(x[is_end]); ey.extend(y[is_end]); ez.extend(z[is_end])
            e_evt.extend([evt] * int(is_end.sum())); e_clid.extend(clid[is_end])

            nb = int((~is_end).sum())
            bx.extend(x[~is_end]); by.extend(y[~is_end]); bz.extend(z[~is_end])
            b_evt.extend([evt] * nb); b_clid.extend(clid[~is_end])

    out = dict(
        end_x=np.asarray(ex), end_y=np.asarray(ey), end_z=np.asarray(ez),
        end_evt=np.asarray(e_evt, dtype=int), end_clid=np.asarray(e_clid, dtype=int),
        body_x=np.asarray(bx), body_y=np.asarray(by), body_z=np.asarray(bz),
        body_evt=np.asarray(b_evt, dtype=int), body_clid=np.asarray(b_clid, dtype=int),
    )
    np.savez_compressed(args.out, **out)
    print("Wrote %s" % args.out)
    print("  events with TGM tracks : %d" % n_events_with_tgm)
    print("  total tagged tracks    : %d" % n_tracks)
    print("  endpoint points        : %d" % out["end_x"].size)
    print("  body points            : %d" % out["body_x"].size)
    if out["end_x"].size:
        for ax, a in (("x", out["end_x"]), ("y", out["end_y"]), ("z", out["end_z"])):
            print("  endpoint %s range cm   : [%.1f, %.1f]" % (ax, a.min(), a.max()))


if __name__ == "__main__":
    main()

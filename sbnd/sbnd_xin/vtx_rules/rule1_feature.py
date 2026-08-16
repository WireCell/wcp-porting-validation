#!/usr/bin/env python3
"""doc pr/89 Arm C -- the rule-1 aggregate, as a number instead of prose.

pr/80 sec 4 measured the strongest single vertex discriminator in the whole
campaign: "86.5% of truth vertices have every attached decisive track
pointing away, vs 31.9% of the other main-cluster vertices" -- and no
pr/77-81 formulation ever used it (they ranked on the 23-column scoreboard
row, which has no dQ/dx, no Bragg direction, no topology).  The number lived
only as prose; NO script computed it.  This one does, on the primitives that
already enforce the pr/80 sec F5 contamination discipline (`outgoing_purity`
reads neither `rr` nor `dirsign`; arclen from the polyline).

Two modes:

  --c0     doc pr/89 Arm C step C0: recover the published 86.5% / 31.9% on
           pr/80's own frozen dev half (runs/split-20260815.tsv), against
           the archived prod0813 dumps (extracted to scratch; --remap points
           the labels' recorded `source` paths there).  Several definitional
           variants are printed because the prose does not pin the
           denominator; C0 passes when one matches, and THAT definition is
           then frozen for C1.

  --dumps  emit the per-(evt, vertex) feature TSV for rerank_replay.py
           --topo-file: evt, vertex_id, frac, n_votes, collinear.
           frac = outgoing_purity (=-1 when no attached track is decisive:
           pr/88 P6 -- a missing vote must never read as 0.0), n_votes = the
           decisive count, collinear = 1 when any attached prong pair leaves
           back-to-back at >= 150 deg (the pr/80 sec 10.8 confound; report
           split on it, never fold it in silently).
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtx_geom as G                                             # noqa: E402
import vtx_io                                                    # noqa: E402
from vtx_rules import CFG, outgoing_purity, _cand_vertices       # noqa: E402

COLLINEAR_DEG = 150.0


def vertex_features(seg_of, vid, cfg=CFG):
    """(frac, n_votes, collinear) for one vertex."""
    pur, dec = outgoing_purity(seg_of, vid, cfg)
    att = seg_of.get(vid, [])
    coll = 0
    for i in range(len(att)):
        for j in range(i + 1, len(att)):
            a = G.prong_angle(att[i], att[j], vid)
            if a is not None and a >= COLLINEAR_DEG:
                coll = 1
                break
        if coll:
            break
    return (-1.0 if pur is None else pur), dec, coll


def run_c0(args):
    rows = []
    with open(args.split) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["half"] == args.half:
                rows.append(r)
    labels = {l["key"]: l for l in vtx_io.load_labels()}
    print("C0 on the %s half: %d split rows, %d labels loaded"
          % (args.half, len(rows), len(labels)))

    remap = [tuple(m.split("=", 1)) for m in (args.remap or [])]
    # counters per variant: (truth-hits, truth-n, other-hits, other-n)
    variants = {
        "purity==1 & n>=1, all cand vertices in denom": [0, 0, 0, 0],
        "purity==1 & n>=1, voting vertices only":        [0, 0, 0, 0],
        "purity>0  & n>=1, all cand vertices in denom": [0, 0, 0, 0],
    }
    n_used = n_skip = 0
    for r in rows:
        key = (r["tag"], int(r["runNo"]), int(r["subRunNo"]), int(r["eventNo"]))
        lab = labels.get(key)
        if lab is None or lab.get("truth_vid") is None:
            n_skip += 1
            continue
        src = lab["source"]
        for old, new in remap:
            src = src.replace(old, new)
        if not os.path.exists(src):
            n_skip += 1
            continue
        with open(src) as fh:
            dump = json.load(fh)
        cid = vtx_io.main_cluster_id(dump)
        if cid is None:
            n_skip += 1
            continue
        seg_of = vtx_io.segments_of_vertex(dump)
        tvid = lab["truth_vid"]
        n_used += 1
        for v in _cand_vertices(dump, cid):
            pur, dec = outgoing_purity(seg_of, v["id"])
            is_truth = v["id"] == tvid
            for name, c in variants.items():
                if "voting" in name and dec == 0:
                    continue
                if "purity>0" in name:
                    hit = pur is not None and pur > 0.0 and dec >= 1
                else:
                    hit = pur == 1.0 and dec >= 1
                if is_truth:
                    c[0] += int(hit); c[1] += 1
                else:
                    c[2] += int(hit); c[3] += 1

    print("events used %d, skipped %d (manual pick / missing dump / no "
          "main cluster)" % (n_used, n_skip))
    print("published (pr/80 sec 4): truth 86.5%%  others 31.9%%")
    for name, (th, tn, oh, on) in variants.items():
        print("  %-48s truth %5.1f%% (%d/%d)   others %5.1f%% (%d/%d)"
              % (name, 100.0 * th / tn if tn else float("nan"), th, tn,
                 100.0 * oh / on if on else float("nan"), oh, on))
    return 0


def run_features(args):
    paths = []
    for spec in args.dumps:
        if os.path.isfile(spec) and not spec.endswith(".json"):
            with open(spec) as fh:
                paths += [l.strip() for l in fh if l.strip()]
        else:
            paths += sorted(glob.glob(spec))
    seen_evt = {}
    n_rows = 0
    with open(args.tsv, "w") as out:
        out.write("evt\tvertex_id\tfrac\tn_votes\tcollinear\n")
        for p in paths:
            with open(p) as fh:
                dump = json.load(fh)
            evt = (dump.get("meta") or {}).get("eventNo")
            sb = dump.get("vertex_scoreboard") or {}
            rows = sb.get("rows") or []
            if evt is None or not rows:
                continue
            if evt in seen_evt and seen_evt[evt] != p:
                raise SystemExit(
                    "EVENT ID COLLISION: evt%d in both %s and %s -- a merged "
                    "topo file joined on (evt, vertex_id) would silently mix "
                    "arms (same guard as build_dataset.py)"
                    % (evt, seen_evt[evt], p))
            seen_evt[evt] = p
            seg_of = vtx_io.segments_of_vertex(dump)
            for r in sorted({row["vertex_id"] for row in rows}):
                frac, dec, coll = vertex_features(seg_of, r)
                out.write("%d\t%d\t%.4f\t%d\t%d\n" % (evt, r, frac, dec, coll))
                n_rows += 1
    print("wrote %s: %d (evt, vertex) rows over %d dumps"
          % (args.tsv, n_rows, len(seen_evt)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", action="store_true")
    ap.add_argument("--split", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs", "split-20260815.tsv"))
    ap.add_argument("--half", default="dev")
    ap.add_argument("--remap", nargs="*", default=[],
                    help="OLD=NEW substring substitutions applied to each "
                         "label's recorded source path (archived arms "
                         "extracted to scratch)")
    ap.add_argument("--dumps", nargs="*", default=[],
                    help="calib dump globs, paths, or list files -> feature "
                         "TSV mode")
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()
    if args.c0:
        return run_c0(args)
    if args.dumps:
        if not args.tsv:
            raise SystemExit("--dumps mode needs --tsv")
        return run_features(args)
    raise SystemExit("pick a mode: --c0 or --dumps")


if __name__ == "__main__":
    sys.exit(main())

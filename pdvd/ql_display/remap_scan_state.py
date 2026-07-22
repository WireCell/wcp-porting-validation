#!/usr/bin/env python
"""Carry confirmed hand-scan picks across a reprocessing into a fresh ql_scan tag.

The ql_scan viewer restores a saved selection from
work/ql_labels/<tag>/.scan_state-evt<ID>.json, keyed by (flash_gid,
main_cluster_uid).  Cluster uids (apa*1e6 + ident) are stable across a
matching-only reprocess (same cluster tarballs, same clustering config), but
flash_gid is the flash's INDEX in QLMatching's admitted flash list -- any change
to flash admission (e.g. the cathode-scoped flash_sel_* knobs) renumbers every
gid after the first cut flash.  This script re-keys confirmed picks from an OLD
scan onto a NEW calib dump by matching the flash TIME (physical, bit-stable:
same opflash archive + same trigger offsets) and keeping the uid.

Sources (either/both):
  --labels    a viewer labels-evt<ID>.json: every entry of "matches" is a pick
  --decisions a decisions-evt<ID>.jsonl (make_labels.py schema): verdicts
              keep/add are picks, everything else ignored

A pick can be lost two ways, both REPORTED (never silently dropped):
  flash-cut:      no flash within --tol us of the old time (admission cut it)
  bundle-missing: flash survives but (gid, uid) is not a candidate bundle in
                  the new dump (containment / prefilter / clustering change)

Writes work/ql_labels/<tag>/.scan_state-evt<ID>.json only (the viewer's Save
then produces fresh labels-evt<ID>.json from the new dump).  Refuses to
overwrite an existing scan_state unless --force (M13: scan records are
append-only; a new scan gets a new tag).

Usage:
  python remap_scan_state.py --calib work/039252_0_cathxa/calib-evt298567.json \
      --tag cathxa --labels work/ql_labels/wfresc/labels-evt298567.json
  python remap_scan_state.py --calib work/039252_1_cathxa/calib-evt298581.json \
      --tag cathxa --decisions ql_display/decisions-crossers-keep/decisions-evt298581.jsonl \
                   --decisions ql_display/decisions-boundary-keep/decisions-evt298581.jsonl
"""

import argparse
import json
import os

GID_STRIDE = 1000000   # QLMatching kFlashGidStride (uid = apa*stride + ident)


def picks_from_labels(path):
    with open(path) as fh:
        lab = json.load(fh)
    out = []
    for e in lab["matches"]:
        uid = e["apa"] * GID_STRIDE + e["cluster_idents"][0]
        out.append((e["flash_time_us"], uid, "labels"))
    return out


def picks_from_decisions(path):
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r["verdict"] in ("keep", "add"):
                out.append((r["flash_time_us"], r["main_cluster_uid"],
                            os.path.basename(os.path.dirname(path))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="NEW calib dump (target)")
    ap.add_argument("--tag", required=True, help="output ql_labels tag")
    ap.add_argument("--labels", action="append", default=[])
    ap.add_argument("--decisions", action="append", default=[])
    ap.add_argument("--tol", type=float, default=0.5, help="flash-time match tolerance (us)")
    ap.add_argument("--time-map", default=None,
                    help="tmerge_time_map.py JSON: translate pick times "
                         "recorded at merged-away flash members to the "
                         "surviving seed's time (doc 26)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(args.calib) as fh:
        d = json.load(fh)
    base = os.path.basename(args.calib)
    event = base[len("calib-"):-len(".json")]

    flashes = [(f["time"], f["gid"]) for f in d["flashes"]]
    bundle_keys = {(b["flash_gid"], b["main_cluster"]) for b in d["bundles"]}

    picks = []
    for p in args.labels:
        picks += picks_from_labels(p)
    for p in args.decisions:
        picks += picks_from_decisions(p)

    if args.time_map:
        with open(args.time_map) as fh:
            m = json.load(fh)
        pairs = m["events"].get(event.replace("evt", ""), [])
        moved = 0
        for i, (t, uid, src) in enumerate(picks):
            for t_old, t_new in pairs:
                if abs(t - t_old) <= args.tol:
                    picks[i] = (t_new, uid, src)
                    moved += 1
                    break
        print("[time-map] %s: %d pick times translated" % (event, moved))

    selected, seen = [], set()
    lost_flash, lost_bundle = [], []
    for t, uid, src in picks:
        best = min(flashes, key=lambda fg: abs(fg[0] - t)) if flashes else None
        if best is None or abs(best[0] - t) > args.tol:
            lost_flash.append((t, uid, src))
            continue
        key = (best[1], uid)
        if key not in bundle_keys:
            lost_bundle.append((t, uid, src))
            continue
        if key in seen:
            continue
        seen.add(key)
        selected.append(list(key))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(args.calib))), "ql_labels", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, ".scan_state-%s.json" % event)
    if os.path.exists(dest) and not args.force:
        raise SystemExit("%s exists; new scan => new tag (or --force)" % dest)
    with open(dest, "w") as fh:
        json.dump({"selected": selected}, fh)

    print("%s: %d picks -> %d selected, %d lost to flash cut, %d lost to "
          "missing bundle -> %s"
          % (event, len(picks), len(selected), len(lost_flash),
             len(lost_bundle), dest))
    for t, uid, src in lost_flash:
        print("  [flash-cut]      t=%.2f us uid=%d (%s)" % (t, uid, src))
    for t, uid, src in lost_bundle:
        print("  [bundle-missing] t=%.2f us uid=%d (%s)" % (t, uid, src))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pr/88 §8.6 — materialise the gated auto-accept picks as label files.

The §7 gate (39/40 = 97.5%, bar 90%) admitted the 341-event auto-accept tier
to the training pool.  But nothing consumed it: `build_dataset.py` reads
`vertex_labels/<tag>/labels-evt*.json` via `iter_labels`, and an auto-accept
pick has no such file — it exists only as a row in a wave's `review.json`.
So the round's headline pool of 993 was, until this script, 217 events a
trainer could actually load.  This closes that gap.

THREE THINGS THIS SCRIPT IS CAREFUL ABOUT.

1.  **A separate tag, and provenance in every record.**  These are AI-scanner
    picks admitted by a statistical gate, not human labels.  They go under
    `vtxscan-mcp2k-auto`, never into `vtxscan-mcp2k`, and each carries
    `label_source: "ai-scanner"` plus the gate that admitted it.  A later
    reader pooling tags must be able to tell which labels a human looked at;
    tag name alone is too easy to lose in a `--tags` line.

2.  **The owner's label always wins.**  The 40 calibration events are in both
    sets and the scanner was WRONG on one of them.  Writing an auto label for
    a calibration event would silently reintroduce that error and give the
    event two label files — and `vtx_io.TAGS_MCP2K`'s "these are disjoint
    events so pooling cannot duplicate a key" reasoning stops holding the
    moment a second tag exists.  Every event with an existing label under
    ANY `vtxscan-mcp2k*` tag is skipped, by event id, at write time.

3.  **Never overwrites** (M13).  An existing destination file is an error,
    not an update.  Re-running after a partial run is safe; re-running to
    "refresh" a label is not a thing this script will do.

Usage:
  python3 vtx_rules/materialize_auto_labels.py --runs /home/xqian/tmp/scan-mcp2k \
      --tag vtxscan-mcp2k-auto --drop-unscannable [--dry-run]
"""
import argparse
import datetime
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scannability                                              # noqa: E402
import vtx_io                                                    # noqa: E402

AUTO = "auto-accept"
GATE = ("doc pr/88 sec 7: blind 40-event calibration of this tier on this "
        "sample, 39/40 = 97.5% correct at 1.0 cm (bar 90%, 95% CI "
        "[86.8%, 99.9%])")


def existing_labelled_events(root, prefix="vtxscan-mcp2k"):
    """Every eventNo that already has a label under any tag starting with
    `prefix` -- the owner's picks, which outrank a scanner's everywhere."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, prefix + "*"))):
        tag = os.path.basename(d)
        for f in glob.glob(os.path.join(d, "labels-evt*.json")):
            try:
                ev = json.load(open(f))["eventNo"]
            except (ValueError, KeyError):
                continue
            out.setdefault(ev, []).append(tag)
    return out


def build_label(row, tag):
    """One auto-accept review row -> a label document, or (None, reason)."""
    dump = json.load(open(row["dump"]))
    vid = row["vertex_id"]
    v = next((v for v in dump.get("vertices") or [] if v.get("id") == vid),
             None)
    if v is None:
        return None, "picked vertex %s not in the dump" % vid
    pos = vtx_io.vertex_xyz(v)
    if pos is None:
        return None, "picked vertex %s has no fit position" % vid
    meta = dump.get("meta") or {}
    evt = meta.get("eventNo")
    if evt is None:
        return None, "dump has no meta.eventNo"
    mv = dump.get("main_vertex")
    dis = vtx_io.dist(pos, vtx_io.xyz(mv))
    return dict(
        event="evt%d" % evt, eventNo=evt,
        runNo=meta.get("runNo"), subRunNo=meta.get("subRunNo"),
        arm=os.path.basename(os.path.dirname(os.path.dirname(
            os.path.abspath(row["dump"])))),
        scan_tag=tag,
        source=os.path.abspath(row["dump"]),
        saved_utc=datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        confidence=row.get("conf"),
        not_a_candidate=False,
        main_vertex=mv,
        # --- provenance: this is NOT a human label -----------------------
        label_source="ai-scanner",
        label_gate=GATE,
        scanner_why=row.get("why"),
        agrees_with_reco=bool(row.get("agrees")),
        reco_sep_cm=row.get("reco_sep_cm"),
        picks=[dict(rank=1, kind="candidate", vertex_id=vid,
                    cluster_id=v.get("cluster_id"), degree=v.get("degree"),
                    is_main=bool(v.get("is_main")),
                    main_candidate=bool(v.get("main_candidate")),
                    x=pos[0], y=pos[1], z=pos[2],
                    dis_to_main=dis)],
    ), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tag", default="vtxscan-mcp2k-auto")
    ap.add_argument("--sbnd-root", default=vtx_io.BASE)
    ap.add_argument("--drop-unscannable", action="store_true",
                    help="skip 'only dots' events -- an auto-accepted pick on "
                         "an event with no readable vertex is exactly the "
                         "noise the owner asked to filter out")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = []
    for w in sorted(glob.glob(os.path.join(a.runs, "wave*", "review.json"))):
        rows += [r for r in json.load(open(w))["rows"] if r["bucket"] == AUTO]
    print("auto-accept rows: %d" % len(rows))

    root = os.path.join(a.sbnd_root, "vertex_labels")
    owned = existing_labelled_events(root)
    out_dir = os.path.join(root, a.tag)

    skip_owned = skip_dots = skip_exists = 0
    made, errs = [], []
    for r in rows:
        if a.drop_unscannable and scannability.unscannable(r["dump"]):
            skip_dots += 1
            continue
        doc, why = build_label(r, a.tag)
        if doc is None:
            errs.append((r["event"], why))
            continue
        if doc["eventNo"] in owned:
            skip_owned += 1
            continue
        path = os.path.join(out_dir, "labels-evt%d.json" % doc["eventNo"])
        if os.path.exists(path):
            skip_exists += 1
            continue
        made.append((path, doc))

    print("  skipped %d already labelled by the owner (owner's pick wins)"
          % skip_owned)
    print("  skipped %d 'only dots'" % skip_dots)
    print("  skipped %d already written under %s" % (skip_exists, a.tag))
    if errs:
        print("  UNRESOLVABLE %d: %s" % (len(errs), errs[:5]))
    print("  to write: %d -> %s" % (len(made), out_dir))
    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 1 if errs else 0

    os.makedirs(out_dir, exist_ok=True)
    for path, doc in made:
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
    print("\nwrote %d labels" % len(made))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())

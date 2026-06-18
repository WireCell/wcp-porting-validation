#!/usr/bin/env python3
"""Migrate pre-joint-node PDHD Q/L hand-scan results to the new calib id scheme.

Before the joint QLMatching node, PDHD wrote one calib JSON per drift side and the
viewer merged the two, adding a per-side offset SIDE_OFF = 1e9 to every side-1
gid/uid so the two files could not collide. The joint node now writes ONE combined
calib file whose ids are already globally unique via the apa*1e6 stride, so the
viewer no longer adds SIDE_OFF. Scan results saved by the old viewer therefore key
their selections on the offset ids and will not restore against the new dump.

This converts the saved files in a ql_labels dir back to the bare (apa*1e6+ident)
ids by stripping SIDE_OFF (v -> v % SIDE_OFF; side-0 ids < 1e9 are unchanged):
  - .scan_state-*.json : "selected" = [[flash_gid, main_cluster], ...]  (both stripped)
  - labels-*.json      : matches[].flash_gid  (stripped; cluster_idents/apa are raw
                         idents + side, never offset, so they stay as-is)

The original file is preserved as <name>.prejoint (never overwritten); the live file
is rewritten FROM that backup each run, so re-running is idempotent.

Usage:  ./convert_scan_prejoint.py <ql_labels_dir> [more dirs ...]
        (default: ../work/ql_labels and its --tag subdirs)
"""
import glob
import json
import os
import sys

SIDE_OFF = 1_000_000_000


def _strip(v):
    return v % SIDE_OFF


def _load_source(path):
    """Read from the .prejoint backup if it exists (so re-runs are idempotent),
    else create the backup from the current file and read that."""
    bak = path + ".prejoint"
    if os.path.exists(bak):
        with open(bak) as fh:
            return json.load(fh), bak
    with open(path) as fh:
        d = json.load(fh)
    with open(bak, "w") as fh:
        json.dump(d, fh, indent=1)
    return d, bak


def convert_scan_state(path):
    d, bak = _load_source(path)
    sel = d.get("selected", [])
    d["selected"] = [[_strip(fg), _strip(mc)] for fg, mc in sel]
    with open(path, "w") as fh:
        json.dump(d, fh)
    return len(sel), bak


def convert_labels(path):
    d, bak = _load_source(path)
    n = 0
    for m in d.get("matches", []):
        if "flash_gid" in m and m["flash_gid"] >= SIDE_OFF:
            m["flash_gid"] = _strip(m["flash_gid"])
            n += 1
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1)
    return n, bak


def convert_dir(d):
    states = sorted(glob.glob(os.path.join(d, "**", ".scan_state-*.json"), recursive=True))
    labels = sorted(glob.glob(os.path.join(d, "**", "labels-*.json"), recursive=True))
    states = [p for p in states if not p.endswith(".prejoint")]
    labels = [p for p in labels if not p.endswith(".prejoint")]
    for p in states:
        n, bak = convert_scan_state(p)
        print("scan_state: %s  (%d selections; backup %s)" % (p, n, os.path.basename(bak)))
    for p in labels:
        n, bak = convert_labels(p)
        print("labels:     %s  (%d offset flash_gids stripped; backup %s)"
              % (p, n, os.path.basename(bak)))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = sys.argv[1:] or [os.path.join(here, "..", "work", "ql_labels")]
    for d in dirs:
        if os.path.isdir(d):
            convert_dir(d)
        else:
            print("skip (not a dir):", d)

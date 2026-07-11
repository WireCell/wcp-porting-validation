#!/usr/bin/env python
"""Turn AI scan decisions into viewer-compatible hand-scan label files.

Reads a calib dump + a decisions JSONL (one line per judged bundle:
{"event","group","flash_gid","apa","main_cluster_uid","verdict",...} with
verdict keep|reject|add) and writes, byte-compatible with the ql_scan
viewer's Save (`on_save`/`_match_entry`) and autosave (`save_state`):

  work/ql_labels/<tag>/labels-evt<ID>.json
  work/ql_labels/<tag>/.scan_state-evt<ID>.json

so the scan can be reviewed interactively with
  ./ql_scan/serve_ql_scan.sh <port> --tag <tag> <calib globs>

Selection = {auto bundles} - rejects + adds. Mirrors the human workflow of
"Load auto-match, then edit what the viewer shows": every auto bundle whose
main cluster passes the viewer's 5 cm length filter MUST carry a keep/reject
decision (error otherwise); auto bundles hidden by the length filter default
to keep (the human never saw them either).

Usage:
  python make_labels.py <calib-evt*.json> <decisions.jsonl> [--tag claude] [--check]
"""

import argparse
import json
import math
import os

MIN_CLUS_LEN = 5.0   # viewer length filter


def load_calib(path):
    with open(path) as fh:
        d = json.load(fh)
    for g, f in enumerate(sorted(d["flashes"], key=lambda f: f["time"]), start=1):
        f["group"] = g
    d["flash_by_gid"] = {f["gid"]: f for f in d["flashes"]}
    d["cluster_by_uid"] = {c["uid"]: c for c in d["clusters"]}
    return d


def cluster_length(d, uid):
    c = d["cluster_by_uid"][uid]
    if not c["x"]:
        return 0.0
    return math.sqrt(sum((max(c[a]) - min(c[a])) ** 2 for a in "xyz"))


def match_entry(d, j):
    """Verbatim ql_scan_viewer._match_entry."""
    b = d["bundles"][j]
    f = d["flash_by_gid"][b["flash_gid"]]
    return {
        "flash_gid": b["flash_gid"], "flash_id": b["flash_id"],
        "flash_time_us": f["time"], "apa": b["apa"], "group": f["group"],
        "cluster_idents": [d["cluster_by_uid"][u]["ident"]
                           for u in [b["main_cluster"]] + b["other_clusters"]],
        "op_pes": f["pe"], "op_pe_err": f["pe_err"], "pred_pes": b["pred_pe"],
        "metrics": {"ks_dis": b["ks_dis"], "chi2": b["chi2"], "ndf": b["ndf"],
                    "strength": b["strength"],
                    "total_PE": b["total_PE"],
                    "total_pred_light": b["total_pred_light"]},
        "flags": {k: b.get(k) for k in ("consistent", "potential_bad_match",
                                        "close_to_PMT", "at_cathode",
                                        "at_x_boundary", "spec_end",
                                        "window_truncated", "two_boundary",
                                        "contained", "auto_selected")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("decisions")
    ap.add_argument("--tag", default="claude")
    ap.add_argument("--check", action="store_true",
                    help="round-trip check the written files")
    args = ap.parse_args()

    d = load_calib(args.calib)
    base = os.path.basename(args.calib)
    event = base[len("calib-"):-len(".json")]          # evt<ID>

    by_key = {}
    for j, b in enumerate(d["bundles"]):
        by_key.setdefault((b["flash_gid"], b["main_cluster"]), []).append(j)

    decisions = {}
    with open(args.decisions) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r["event"] != event:
                continue
            key = (r["flash_gid"], r["main_cluster_uid"])
            if key not in by_key:
                raise SystemExit("decision key %s not in calib bundles" % (key,))
            if key in decisions:
                raise SystemExit("duplicate decision for %s" % (key,))
            decisions[key] = r["verdict"]

    selected, implicit_keep = set(), 0
    for j, b in enumerate(d["bundles"]):
        key = (b["flash_gid"], b["main_cluster"])
        verdict = decisions.get(key)
        if b["auto_selected"]:
            if verdict == "keep":
                selected.add(j)
            elif verdict == "reject":
                pass
            elif verdict == "add":
                raise SystemExit("verdict 'add' on auto bundle %s" % (key,))
            elif cluster_length(d, b["main_cluster"]) <= MIN_CLUS_LEN:
                selected.add(j)          # hidden by the viewer length filter
                implicit_keep += 1
            else:
                raise SystemExit("auto bundle %s (len %.1f cm) has no decision"
                                 % (key, cluster_length(d, b["main_cluster"])))
        elif verdict == "add":
            selected.add(j)
        elif verdict == "keep":
            raise SystemExit("verdict 'keep' on non-auto bundle %s (use add)"
                             % (key,))

    # one flash per cluster (viewer match-filter invariant), main clusters
    # only. The auto matcher itself can violate this (e.g. one cluster kept
    # on two nearby flashes), so a violation is surfaced, not fatal — the
    # scan decisions are where it gets resolved.
    flash_of = {}
    for j in sorted(selected):
        b = d["bundles"][j]
        uid = b["main_cluster"]
        if uid in flash_of and flash_of[uid] != b["flash_gid"]:
            print("WARNING: cluster uid %d matched to flashes %d and %d"
                  % (uid, flash_of[uid], b["flash_gid"]))
        flash_of[uid] = b["flash_gid"]

    rejected = sorted(j for j, b in enumerate(d["bundles"])
                      if b["auto_selected"] and j not in selected)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(args.calib))), "ql_labels", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    out = {"event": event, "source": base,
           "matches": [match_entry(d, j) for j in sorted(selected)],
           "rejected_auto": [match_entry(d, j) for j in rejected]}
    dest = os.path.join(out_dir, "labels-%s.json" % event)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)

    keys = [[d["bundles"][j]["flash_gid"], d["bundles"][j]["main_cluster"]]
            for j in sorted(selected)]
    state = os.path.join(out_dir, ".scan_state-%s.json" % event)
    with open(state, "w") as fh:
        json.dump({"selected": keys}, fh)

    n_add = sum(1 for j in selected if not d["bundles"][j]["auto_selected"])
    print("%s: %d matches (%d kept auto, %d implicit-keep short, %d added), "
          "%d rejected -> %s" % (event, len(selected),
                                 len(selected) - n_add - implicit_keep,
                                 implicit_keep, n_add, len(rejected), dest))

    if args.check:
        with open(dest) as fh:
            lab = json.load(fh)
        assert len(lab["matches"]) == len(selected)
        assert len(lab["rejected_auto"]) == len(rejected)
        with open(state) as fh:
            want = {tuple(k) for k in json.load(fh)["selected"]}
        sel2 = {j for j, b in enumerate(d["bundles"])
                if (b["flash_gid"], b["main_cluster"]) in want}
        assert sel2 == selected, "scan_state does not round-trip"
        print("check OK: labels + scan_state round-trip")


if __name__ == "__main__":
    main()

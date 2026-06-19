#!/usr/bin/env python3
"""Regenerate a ql_scan labels-evt<N>.json from a calib dump + a scan_state, faithfully
replicating ql_scan_viewer.py's Event load + _match_entry + on_save. Used to refresh the
labels export after a reprocess (companion to remap_scan_after_reprocess.py, which only
re-keys the scan_state).  build_labels(dump_path, scan_state_path, event_label) -> dict."""
import json, os, sys


def load_event(dump_path):
    """Replicate Event.__init__ for a single combined dump: clusters/bundles/flashes +
    phantom-flash collapse + flash_by_gid(canonical) + cluster_by_uid."""
    d = json.load(open(dump_path))
    flashes, clusters, bundles = d["flashes"], d["clusters"], d["bundles"]
    cluster_by_uid = {c["uid"]: c for c in clusters}
    _lit = lambda f: 0 if sum(f["pe"][80:]) >= sum(f["pe"][:80]) else 1
    _sig = lambda f: (_lit(f), round(f["time"], 4), round(f["total_PE"], 3))
    canon_gid = {_sig(f): f["gid"] for f in flashes if f["apa"] == _lit(f)}
    remap = {f["gid"]: canon_gid[_sig(f)] for f in flashes
             if f["apa"] != _lit(f) and _sig(f) in canon_gid and f["gid"] != canon_gid[_sig(f)]}
    for b in bundles:
        b["flash_gid"] = remap.get(b["flash_gid"], b["flash_gid"])
    flashes = [f for f in flashes if f["gid"] not in remap]
    flash_by_gid = {}
    for f in flashes:
        if f["gid"] not in flash_by_gid or f["apa"] == _lit(f):
            flash_by_gid[f["gid"]] = f
    return bundles, flash_by_gid, cluster_by_uid


def match_entry(b, flash_by_gid, cluster_by_uid):
    """Replicate ql_scan_viewer._match_entry exactly."""
    f = flash_by_gid[b["flash_gid"]]
    return {
        "flash_gid": b["flash_gid"], "flash_id": b["flash_id"],
        "flash_time_us": f["time"], "apa": b["apa"], "group": f["group"],
        "cluster_idents": [cluster_by_uid[u]["ident"]
                           for u in [b["main_cluster"]] + b["other_clusters"]],
        "op_pes": f["pe"], "op_pe_err": f["pe_err"], "pred_pes": b["pred_pe"],
        "metrics": {"ks_dis": b["ks_dis"], "chi2": b["chi2"], "ndf": b["ndf"],
                    "strength": b["strength"],
                    "total_PE": b["total_PE"], "total_pred_light": b["total_pred_light"]},
        "flags": {k: b[k] for k in ("consistent", "potential_bad_match",
                                    "close_to_PMT", "at_x_boundary", "spec_end",
                                    "window_truncated", "contained", "auto_selected")},
    }


def build_labels(dump_path, scan_state_path, event_label):
    """matches = scan_state picks; rejected_auto = auto_selected not picked (on_save)."""
    bundles, fbg, cbu = load_event(dump_path)
    want = {tuple(k) for k in json.load(open(scan_state_path)).get("selected", [])}
    selected = sorted(j for j, b in enumerate(bundles)
                      if (b["flash_gid"], b["main_cluster"]) in want)
    rejected = sorted(j for j, b in enumerate(bundles)
                      if b["auto_selected"] and j not in selected)
    return {"event": event_label, "source": os.path.basename(dump_path),
            "matches": [match_entry(bundles[j], fbg, cbu) for j in selected],
            "rejected_auto": [match_entry(bundles[j], fbg, cbu) for j in rejected]}


if __name__ == "__main__":
    dump, scan, evt, out = sys.argv[1:5]
    lab = build_labels(dump, scan, evt)
    with open(out, "w") as fh:
        json.dump(lab, fh, indent=1)
    print("wrote %s: %d matches, %d rejected_auto"
          % (out, len(lab["matches"]), len(lab["rejected_auto"])))

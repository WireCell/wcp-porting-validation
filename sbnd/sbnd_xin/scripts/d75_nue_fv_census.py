#!/usr/bin/env python3
"""doc 75 -- nue_sp_consistent_fv knob census: T_tagger FV-block branches +
numu_score/nue_score, arm A (OFF) vs arm B (ON).

Forked from d74_cosmict_census.py's shape (same doc-round lineage: an
off-vs-on per-event value census, not a byte gate), reading T_tagger via
uproot instead of calib-pr JSON because the FV-fix branches
(anc_flag_main_outside, br3_2_other_fid/flag, shw_sp_br3_2_other_fid/flag,
stw_*_flag) are not surfaced in the calib-pr summary -- only the *_score
BDT features are.  numu_score/nue_score are still read from calib-pr (same
denominator doc 74's census used).

Usage: d75_nue_fv_census.py <armOFF> <armON> [--label name]
Exit 0 always (census, not a gate).
"""
import argparse
import glob
import json
import os
import re
import sys

import uproot

# Per-event scalar branches only.  stw_2_v_flag/stw_3_v_flag/stw_4_v_flag
# are jagged (one entry per companion shower at the vertex) and are NOT
# scalars per T_tagger row; they are excluded here and would need an
# awkward-array read to census properly -- the stw_*_score summaries in
# calib-pr JSON (not read by this script) already fold them into a single
# number per candidate.
TAGGER_FLAGS = [
    "anc_flag_main_outside",
    "br3_2_other_fid", "br3_2_flag",
    "shw_sp_br3_2_other_fid", "shw_sp_br3_2_flag",
    "stw_1_flag", "stw_1_flag_single_shower",
]
SCORES = ["numu_score", "nue_score"]


def events_of(arm):
    out = {}
    for p in glob.glob(os.path.join(arm, "pr_evt*")):
        m = re.match(r".*pr_evt(\d+)$", p)
        if m and os.path.exists(os.path.join(p, "tracking-pr.root")):
            out[int(m.group(1))] = p
    return out


def flags_of(evt_dir):
    """Read the T_tagger FV-block branches for every row (nu_index) of the
    event's tracking-pr.root.  Returns list-of-dict, one per row; [] if the
    tree/branches are absent (e.g. an event with no PR output)."""
    path = os.path.join(evt_dir, "tracking-pr.root")
    try:
        with uproot.open(path) as f:
            if "T_tagger" not in [k.split(";")[0] for k in f.keys()]:
                return []
            tt = f["T_tagger"]
            have = [b for b in TAGGER_FLAGS if b in tt.keys()]
            if not have:
                return []
            arrs = tt.arrays(have, library="np")
            n = len(arrs[have[0]])
            return [{b: arrs[b][i].item() if hasattr(arrs[b][i], "item") else arrs[b][i]
                     for b in have} for i in range(n)]
    except Exception as err:  # noqa: BLE001
        print(f"# WARN: {path}: {err}", file=sys.stderr)
        return []


def scores_of(evt_dir, evt_id):
    p = os.path.join(evt_dir, f"calib-pr-evt{evt_id}.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f).get("tagger", {}) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armA", help="OFF arm")
    ap.add_argument("armB", help="ON arm")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ea, eb = events_of(args.armA), events_of(args.armB)
    common = sorted(set(ea) & set(eb))
    print(f"# {args.label} events: A={len(ea)} B={len(eb)} common={len(common)}")

    n_flag_evt = 0
    n_score_evt = 0
    flag_flip_counts = {f: 0 for f in TAGGER_FLAGS}
    for evt in common:
        fa, fb = flags_of(ea[evt]), flags_of(eb[evt])
        diffs = []
        # Positional row compare (row 0 = primary candidate in both arms;
        # row-count mismatch itself is reported, not assumed away).
        if len(fa) != len(fb):
            diffs.append(f"row_count:{len(fa)}->{len(fb)}")
        for i in range(min(len(fa), len(fb))):
            for f in TAGGER_FLAGS:
                va, vb = fa[i].get(f), fb[i].get(f)
                if va != vb:
                    diffs.append(f"row{i}.{f}:{va}->{vb}")
                    flag_flip_counts[f] += 1
        sa, sb = scores_of(ea[evt], evt), scores_of(eb[evt], evt)
        sc = []
        for s in SCORES:
            va, vb = float(sa.get(s, 0) or 0), float(sb.get(s, 0) or 0)
            if abs(va - vb) > 1e-9:
                sc.append(f"{s}:{va:.4f}->{vb:.4f}")
        if diffs or sc:
            print(f"evt {evt}: " + " ".join(diffs + sc))
            if diffs:
                n_flag_evt += 1
            if sc:
                n_score_evt += 1
    print(f"# summary {args.label}: flag-diff events {n_flag_evt}/{len(common)}, "
          f"score-diff events {n_score_evt}/{len(common)}")
    for f, c in flag_flip_counts.items():
        if c:
            print(f"#   {f}: {c} row(s) flipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

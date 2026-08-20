#!/usr/bin/env python3
"""doc pr/94 Phase 5 -- the knob-ON primary row must equal the knob-OFF row,
branch for branch, on every event of a population arm.

This is the load-bearing evidence for Phase 6 (flipping `nu_per_bundle` ON in
SBND production).  The claim being tested is precise:

    turning the knob on is PURELY ADDITIVE -- it adds rows for the other
    in-beam bundles and does not perturb the candidate the legacy chain
    would have reported.

`pr94_root_gate.py` cannot test that: with the knob on `T_tagger` has N rows
and with it off exactly 1, so a whole-tree comparison differs trivially on
every multi-bundle event and tells you nothing.  Here the ON arm's PRIMARY row
is selected with `pr94_rows.primary_index()` -- the longest selected main
activity, which reproduces the legacy single-winner rule -- and compared
against the OFF arm's only row.

Every shared branch of `T_tagger` and `T_kine` is compared, not a hand-picked
subset: an earlier round checked five fields (vertex, numu_score, nue_score,
cosmict_flag, Enu) and would not have noticed a drift in any of the other
~1200 `T_tagger` branches.  Branches that exist only in the ON file (the pr/94
identity fields and the whole `act_*` block) are the point of the change and
are skipped, and which ones were skipped is reported so the exclusion cannot
quietly widen.

NaN is normalised to the string "nan" -- `T_rec_charge.reduced_chi2` carries
legitimate NaNs and `nan != nan` would report every such event as a diff (the
same trap `pr94_root_gate.py` documents).

Usage: pr94_primary_gate.py <off_arm> <on_arm> [--verbose]
Exit 0 = every event's primary row is identical.
"""
import argparse
import math
import os
import re
import sys

import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr94_rows import primary_index, n_rows            # noqa: E402

# Present only when the knob is on; comparing them is meaningless.
ON_ONLY = ("cluster_id", "matched_flash_gid", "nu_index")


def norm(v):
    """Scalar -> comparable value, with NaN collapsed to a token."""
    try:
        if isinstance(v, float) and math.isnan(v):
            return "nan"
    except TypeError:
        pass
    try:
        return v.tolist()
    except AttributeError:
        return v


def row_of(tree, i, skip):
    out = {}
    for br in tree.keys():
        if br in skip or br.startswith("act_"):
            continue
        try:
            out[br] = norm(tree[br].array(library="np")[i])
        except Exception:                                 # noqa: BLE001
            out[br] = "<unreadable>"
    return out


def check(off_path, on_path):
    """-> (n_on_rows, [diff strings], set(skipped), status)

    `status` distinguishes the four OFF/ON tagger-output combinations.  An
    event where TaggerCheckNeutrino selected no candidate has no TrackFitting,
    so UbooneTaggerOutputVisitor books nothing and the file carries only
    Trun/T_proj/T_bad_ch (rc still 0).  Whether that happens on both sides is
    the point: "OFF empty, ON populated" is per-bundle mode reporting an event
    the legacy chain reported nothing for -- a headline Phase 5 number, not an
    error -- while "OFF populated, ON empty" would be a real regression, since
    the knob is supposed to be purely additive.
    """
    with uproot.open(off_path) as fa, uproot.open(on_path) as fb:
        diffs, skipped = [], set()
        nrow = 0
        has_a = "T_tagger" in [k.split(";")[0] for k in fa.keys()]
        has_b = "T_tagger" in [k.split(";")[0] for k in fb.keys()]
        if not has_a and not has_b:
            return 0, [], skipped, "both-empty"
        if not has_a and has_b:
            with uproot.open(on_path) as f2:
                nrow = n_rows(f2["T_tagger"])
            return nrow, [], skipped, "gained"
        if has_a and not has_b:
            return 0, ["OFF has tagger output, ON has none"], skipped, "lost"
        for tname in ("T_tagger", "T_kine"):
            if tname not in [k.split(";")[0] for k in fa.keys()]:
                diffs.append("%s missing from the OFF file" % tname)
                continue
            ta, tb = fa[tname], fb[tname]
            if ta.num_entries != 1:
                diffs.append("%s: OFF arm has %d rows, expected 1"
                             % (tname, ta.num_entries))
                continue
            i = primary_index(tb)
            if tname == "T_tagger":
                nrow = n_rows(tb)
            only_on = set(tb.keys()) - set(ta.keys())
            skipped |= only_on
            skip = only_on | set(ON_ONLY)
            ra, rb = row_of(ta, 0, skip), row_of(tb, i, skip)
            for br in sorted(set(ra) & set(rb)):
                if ra[br] != rb[br]:
                    diffs.append("%s.%s: OFF %r != ON[%d] %r"
                                 % (tname, br, ra[br], i, rb[br]))
    return nrow, diffs, skipped, "compared"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off_arm")
    ap.add_argument("on_arm")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    evts = sorted(int(m.group(1)) for d in os.listdir(args.off_arm)
                  if (m := re.match(r"pr_evt(\d+)$", d)))
    nok = nbad = nskip = 0
    rowhist, prim_not0, all_skipped, status = {}, [], set(), {}
    gained_evts = []
    for e in evts:
        pa = os.path.join(args.off_arm, "pr_evt%d" % e, "tracking-pr.root")
        pb = os.path.join(args.on_arm, "pr_evt%d" % e, "tracking-pr.root")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            nskip += 1
            continue
        nrow, diffs, skipped, st = check(pa, pb)
        all_skipped |= skipped
        status[st] = status.get(st, 0) + 1
        if st == "gained":
            gained_evts.append(e)
        if st in ("compared", "gained"):
            rowhist[nrow] = rowhist.get(nrow, 0) + 1
            with uproot.open(pb) as f:
                if primary_index(f["T_tagger"]) != 0:
                    prim_not0.append(e)
        if diffs:
            nbad += 1
            print("DIFF evt %d: %s" % (e, "; ".join(diffs[:4])))
        else:
            nok += 1
            if args.verbose:
                print("evt %d OK (%s, %d row(s))" % (e, st, nrow))

    print("# events identical: %d  differing: %d  skipped(no file): %d"
          % (nok, nbad, nskip))
    print("# OFF/ON tagger-output status: %s"
          % ", ".join("%s=%d" % kv for kv in sorted(status.items())))
    print("#   'gained' = ON produced tagger output where OFF produced none: %d %s"
          % (len(gained_evts), gained_evts[:10]))
    print("#   'lost'   = OFF produced output and ON did not (MUST be 0): %d"
          % status.get("lost", 0))
    print("# rows per event (events with output): %s"
          % ", ".join("%d->%d" % kv for kv in sorted(rowhist.items())))
    print("# events whose primary row is NOT row 0: %d %s"
          % (len(prim_not0), prim_not0[:10]))
    print("# branches skipped (ON-only, i.e. the pr/94 additions): %d"
          % len(all_skipped))
    print("PASS" if nbad == 0 else "FAIL")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

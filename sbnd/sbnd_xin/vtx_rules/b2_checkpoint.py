#!/usr/bin/env python3
"""doc pr/88 pilot checkpoint -- is the B2-merged candidate list safe at scale?

doc pr/80 sec 13.5 left one thing open: B2 (scankit.MERGE_R = 0.8 cm) shrinks
the candidate lists a scanner is shown by 16-33% on the affected events, and
**no scanner had ever scanned from a merged picture** -- the 47->47 rescoring
of sec 13.4 replayed old picks through the new lookup, it did not re-scan.
doc pr/88 is the first at-scale use, so the pilot wave answers that question
on 60 events instead of 845.

There is no truth on a new sample, so accuracy cannot be measured here.  What
CAN be measured is whether the merge is doing what it claims:

  1. off-list picks -- a pick naming a vertex that is in no candidate group.
     `selfscan.review` resolves those to pos=None -> agrees=False and files
     them as REVIEW with no error at all (selfscan.py:432-438), so they are
     invisible unless counted.  If B2 hid a vertex the scanner could see in a
     panel, this is where it shows up.  BAR: zero.
  2. alias resolution -- picks naming an absorbed id rather than the group
     representative.  These MUST resolve through the merge (they are the same
     answer); a nonzero count is healthy, it proves the merge is load-bearing
     and that `review`/`score` honour it.
  3. shrink -- candidate-list size before vs after merge, against the 16-33%
     range sec 13.4 measured on the affected events.

Usage:  python3 vtx_rules/b2_checkpoint.py --dir /home/xqian/tmp/scan-mcp2k/wave0
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scankit                                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()

    manifest = {m["event"]: m
                for m in json.load(open(os.path.join(a.dir, "manifest.json")))}
    picks = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "picks-*.json"))):
        for p in json.load(open(f)):
            picks[p["event"]] = p
    if not picks:
        print("no picks-*.json in %s" % a.dir)
        return 1

    off, alias, rep, abstain = [], [], 0, 0
    raw_tot = mer_tot = 0
    shrunk = []
    for ev, m in sorted(manifest.items()):
        # raw (unmerged) vs merged candidate counts, straight from scankit so
        # this cannot drift from what the panels actually drew.
        d = scankit.sanitize(json.load(open(m["dump"])))
        n_raw = len(scankit.candidates(d, merge_r=0.0))
        n_mer = len(m["candidates"])
        raw_tot += n_raw
        mer_tot += n_mer
        if n_raw != n_mer:
            shrunk.append((ev, n_raw, n_mer))

        p = picks.get(ev)
        if p is None:
            continue
        vid = p.get("vertex_id")
        if vid is None:
            abstain += 1
            continue
        hit = None
        for c in m["candidates"]:
            if vid == c["vertex_id"]:
                hit = "rep"
            elif vid in (c.get("aliases") or []):
                hit = "alias"
            if hit:
                break
        if hit == "alias":
            alias.append((ev, vid))
        elif hit == "rep":
            rep += 1
        else:
            off.append((ev, vid))

    n = len(picks)
    print("=== doc pr/88 B2 pilot checkpoint: %s ===" % a.dir)
    print("events with picks: %d (abstentions %d)" % (n, abstain))
    print()
    print("[1] off-list picks (BAR: zero) : %d" % len(off))
    for ev, vid in off[:20]:
        print("      %s  named vertex %s -- in NO candidate group" % (ev, vid))
    print("[2] picks resolving via an ALIAS: %d   via representative: %d"
          % (len(alias), rep))
    for ev, vid in alias[:10]:
        print("      %s  named absorbed id %s (resolved through the merge)"
              % (ev, vid))
    print("[3] candidate-list shrink      : %d -> %d over %d events (%.1f%%)"
          % (raw_tot, mer_tot, len(manifest),
             100.0 * (raw_tot - mer_tot) / max(1, raw_tot)))
    print("      events where the merge changed the list: %d of %d"
          % (len(shrunk), len(manifest)))
    if shrunk:
        aff = 100.0 * sum(r - m for _, r, m in shrunk) / \
            max(1, sum(r for _, r, m in shrunk))
        print("      shrink ON THOSE events: %.1f%%  "
              "(doc pr/80 sec 13.4 measured 16-33%%)" % aff)
    print()
    ok = not off
    print("VERDICT: %s" % ("PASS -- no pick fell through the merge" if ok else
                           "FAIL -- off-list picks above; B2 is first suspect, "
                           "STOP and report (plan Phase 2 checkpoint)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

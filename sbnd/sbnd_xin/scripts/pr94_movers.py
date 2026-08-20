#!/usr/bin/env python3
"""doc pr/94 Phase 5 -- what the knob actually changes on a population arm.

The headline numbers come from the ROOT files themselves, NOT from any
`event_label`, and that choice is load-bearing.  Per event, from `T_tagger`:

  * does the event have tagger output at all (an event where
    TaggerCheckNeutrino selected nothing has no TrackFitting, so
    UbooneTaggerOutputVisitor books neither tree);
  * how many rows it has; and
  * how many of those rows carry a reconstructed vertex (nu_x/y/z != 0).

Comparing OFF against ON on those three is definition-free: it measures what
the chain produced, so it cannot be confounded by two labellers disagreeing
about what a word means.  `events gaining a FIRST vertex` is the number that
answers "did per-bundle mode find neutrinos the legacy chain missed", and
`events LOSING a vertex` must be 0 -- that is the additivity check.

**Why the label comparison is reported second, and hedged.**  The legacy label
in `nusel-events.tsv` (`nusel_extract.py`'s `merge()`) and the corrected label
in `nusel-events-pr94.tsv` (`pr94_mains_sidecar.py`) DO NOT MEAN THE SAME
THING, so the crosstab is not a like-for-like recovery rate:

  - legacy `nu-candidate` is bundle-level and does not require the chain to
    have reconstructed anything.  Measured on mcp1k: 82 events labelled
    `nu-candidate` produced no tagger output in EITHER arm (e.g. 50919, 55583,
    55925 -- `tracking-pr.root` holds only Trun/T_proj/T_bad_ch).
  - the corrected `nu-candidate` means at least one row reconstructed a vertex.
  - legacy has a `no-beam-flash` category the sidecar does not, so all 117 of
    those move to `no-bundle` purely as vocabulary.

So read the crosstab as "how the two labellers differ", not as "how many
events pr/94 rescued" -- the rescue number is `events gaining a FIRST vertex`
above.  Fixing this properly means patching `nusel_extract.py` itself, which
is an open owner decision (doc sec 9, item ii).

Usage: pr94_movers.py <off_arm> <on_arm> [--sample NAME]
"""
import argparse
import os
import re
import sys

import uproot


def scan_arm(arm):
    """{event: (n_rows, n_rows_with_vertex)} straight from T_tagger."""
    out = {}
    for d in sorted(os.listdir(arm)):
        m = re.match(r"pr_evt(\d+)$", d)
        if not m:
            continue
        p = os.path.join(arm, d, "tracking-pr.root")
        if not os.path.exists(p):
            continue
        e = int(m.group(1))
        with uproot.open(p) as f:
            if "T_tagger" not in [k.split(";")[0] for k in f.keys()]:
                out[e] = (0, 0)          # no candidate selected anywhere
                continue
            a = f["T_tagger"].arrays(["nu_x", "nu_y", "nu_z"], library="np")
            n = len(a["nu_x"])
            v = sum(1 for i in range(n)
                    if (float(a["nu_x"][i]), float(a["nu_y"][i]),
                        float(a["nu_z"][i])) != (0.0, 0.0, 0.0))
            out[e] = (n, v)
    return out


def read_tsv(path, key_cols=("run", "subrun", "event")):
    """Whitespace-aligned TSV -> {(run,subrun,event): {col: val}}."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        header = fh.readline().split()
        for line in fh:
            f = line.split()
            if len(f) != len(header):
                continue
            row = dict(zip(header, f))
            try:
                out[tuple(int(row[c]) for c in key_cols)] = row
            except (KeyError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off_arm")
    ap.add_argument("on_arm")
    ap.add_argument("--sample", default=None)
    args = ap.parse_args()
    name = args.sample or os.path.basename(args.on_arm)

    # ---- part 1: straight from the ROOT files, no labels involved ----------
    ao, an = scan_arm(args.off_arm), scan_arm(args.on_arm)
    ks = sorted(set(ao) & set(an))
    print("== %s -- measured from T_tagger (definition-free) ==" % name)
    print("events compared: %d" % len(ks))
    print("%-28s OFF %5d   ON %5d   %+d"
          % ("events with tagger output", sum(1 for k in ks if ao[k][0]),
             sum(1 for k in ks if an[k][0]),
             sum(1 for k in ks if an[k][0]) - sum(1 for k in ks if ao[k][0])))
    ov = sum(1 for k in ks if ao[k][1])
    nv = sum(1 for k in ks if an[k][1])
    print("%-28s OFF %5d   ON %5d   %+d  (%+.1f %%)"
          % ("events with >=1 vertex", ov, nv, nv - ov,
             100.0 * (nv - ov) / max(1, ov)))
    print("%-28s OFF %5d   ON %5d" % ("total rows",
                                      sum(ao[k][0] for k in ks),
                                      sum(an[k][0] for k in ks)))
    print("%-28s OFF %5d   ON %5d" % ("total reconstructed vertices",
                                      sum(ao[k][1] for k in ks),
                                      sum(an[k][1] for k in ks)))
    gain = [k for k in ks if ao[k][1] == 0 and an[k][1] > 0]
    lost = [k for k in ks if an[k][1] < ao[k][1]]
    print("events gaining a FIRST vertex: %d  e.g. %s" % (len(gain), gain[:8]))
    print("events LOSING a vertex (additivity check, MUST be 0): %d %s"
          % (len(lost), lost[:8]))
    print()

    # ---- part 2: the two labellers, which are NOT commensurable ------------
    off = read_tsv(os.path.join(args.off_arm, "nusel-events.tsv"))
    new = read_tsv(os.path.join(args.on_arm, "nusel-events-pr94.tsv"))
    if off is None:
        sys.exit("no nusel-events.tsv in %s" % args.off_arm)
    if new is None:
        sys.exit("no nusel-events-pr94.tsv in %s -- run pr94_mains_sidecar.py first"
                 % args.on_arm)

    both = sorted(set(off) & set(new))
    print("== %s ==" % name)
    print("events: OFF %d, ON %d, joined %d" % (len(off), len(new), len(both)))

    # 1. row cardinality
    hist = {}
    for k in both:
        n = int(new[k]["n_inbeam_bundle"])
        hist[n] = hist.get(n, 0) + 1
    gained = sum(v for n, v in hist.items() if n > 1)
    print("rows per event: %s"
          % ", ".join("%d->%d" % kv for kv in sorted(hist.items())))
    print("events with MORE THAN ONE row: %d (%.1f %%)"
          % (gained, 100.0 * gained / max(1, len(both))))

    # 2. label movement
    moves, recovered, adverse = {}, [], []
    for k in both:
        a, b = off[k]["event_label"], new[k]["event_label"]
        if a == b:
            continue
        moves[(a, b)] = moves.get((a, b), 0) + 1
        if a == "cosmic-tagged" and b == "nu-candidate":
            recovered.append(k)
        elif b == "cosmic-tagged" and a == "nu-candidate":
            adverse.append(k)
    n_cosmic = sum(1 for k in both if off[k]["event_label"] == "cosmic-tagged")
    print("legacy label counts: %s"
          % ", ".join("%s=%d" % (lbl, sum(1 for k in both if off[k]["event_label"] == lbl))
                      for lbl in sorted({off[k]["event_label"] for k in both})))
    print("label moves (legacy -> corrected):")
    for (a, b), n in sorted(moves.items(), key=lambda kv: -kv[1]):
        print("   %-18s -> %-18s %d" % (a, b, n))
    print("cosmic-tagged -> nu-candidate: %d of %d legacy cosmic-tagged (%.1f %%)"
          % (len(recovered), n_cosmic, 100.0 * len(recovered) / max(1, n_cosmic)))
    if recovered[:10]:
        print("   e.g. events %s" % [k[2] for k in recovered[:10]])
    print("nu-candidate -> cosmic-tagged: %d %s"
          % (len(adverse), [k[2] for k in adverse[:10]]))
    print("NOTE: the two labellers are not commensurable -- see the module "
          "docstring.  The rescue number is 'events gaining a FIRST vertex' "
          "in part 1, not the crosstab above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

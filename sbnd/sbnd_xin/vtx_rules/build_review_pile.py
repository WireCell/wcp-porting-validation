#!/usr/bin/env python3
"""doc pr/88 Phase 3 -- assemble the owner's port-5017 review pile.

The owner capped the pile at ~120 events (~1 h at their measured 30 s/event),
so this is a SELECTION, and the thing that matters is that the selection is
recorded rather than asserted.  Four tiers, in priority order:

  1. REVIEW FIRST  -- scanner `certain` AND disagreeing with the
     reconstruction.  doc pr/80 sec 13.2: five of these caught two genuine
     reconstruction errors, enrichment x2.86.  Highest yield per minute.
  2. abstentions   -- the scanner could not read a vertex at all.
  3. CALIBRATION   -- N auto-accept events drawn at RANDOM and left
     UNMARKED in the served list.  This is doc pr/82 sec 9 gate 5: the
     95.5%-at-36.7% auto-accept prior comes from 60 held-out events of a
     DIFFERENT reconstruction epoch, and until it is re-measured here the
     ~300 auto-accept labels are ungated for training.  The draw is written
     out BEFORE the owner scans, because a subset identified afterwards is
     not a random sample of anything.
  4. fill          -- the ranker-hottest `likely`/`unclear` events that
     disagree with the reconstruction, up to the cap.

Tier 3 is spread evenly through the emitted order rather than blocked, so the
owner cannot tell a calibration event from a review event; tiers 1/2/4 keep
their priority order so the value stays front-loaded.

Usage:
  python3 vtx_rules/build_review_pile.py --runs /home/xqian/tmp/scan-mcp2k \
      --rank /home/xqian/tmp/pr88/rank-mcp2k.tsv --cap 120 --calib 40 \
      --out /home/xqian/tmp/pr88/pile
"""
import argparse
import csv
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtx_io                                                    # noqa: E402
import scannability                                              # noqa: E402

REVIEW_FIRST = "REVIEW FIRST (confident disagreement)"
ABSTAIN = "REVIEW (scanner abstained)"
REVIEW = "REVIEW"
AUTO = "auto-accept"


def load_rows(runs, allow_partial=False):
    """Every wave's review.json, concatenated.  Refuses a partial set unless
    `allow_partial`, which exists so the owner can start scanning before all
    845 events are done.  A partial pile is an INSTALMENT: the calibration
    draw is then random within the waves that ARE in, i.e. stratified by
    wave rather than simple-random over the whole auto-accept tier.  That is
    still a valid sample and Phase 4 pools the strata -- but it has to be
    recorded, because "we drew 40 at random" and "we drew 14 at random from
    the first third and 26 from the rest" are different claims.
    """
    rows, waves = [], sorted(glob.glob(os.path.join(runs, "wave*")))
    if not waves:
        raise SystemExit("no wave dirs under %s" % runs)
    missing = []
    for w in waves:
        rj = os.path.join(w, "review.json")
        if not os.path.exists(rj):
            missing.append(os.path.basename(w))
            continue
        for r in json.load(open(rj))["rows"]:
            r["wave"] = os.path.basename(w)
            rows.append(r)
    if missing and not allow_partial:
        raise SystemExit(
            "no review.json in: %s -- every wave must be reviewed before the "
            "final pile is built, or the tier census is over a subset and the "
            "calibration draw is not random over the auto-accept tier.  Pass "
            "--allow-partial to build an instalment instead."
            % ", ".join(missing))
    if missing:
        print("INSTALMENT: %d of %d waves reviewed (%s still scanning).  The "
              "calibration draw below is random within these waves only.\n"
              % (len(waves) - len(missing), len(waves), ", ".join(missing)))
    return rows


def load_rank(path):
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                out["evt" + str(int(r["evt"]))] = float(
                    r.get("p_corrective") or r["score"])
            except (ValueError, KeyError):
                continue
    return out


def interleave(priority, sprinkle):
    """Spread `sprinkle` evenly through `priority`, preserving both orders."""
    if not sprinkle:
        return list(priority)
    if not priority:
        return list(sprinkle)
    out, step = [], (len(priority) + len(sprinkle)) / float(len(sprinkle))
    pi = si = 0
    nxt = step / 2.0
    for i in range(len(priority) + len(sprinkle)):
        if si < len(sprinkle) and i >= nxt:
            out.append(sprinkle[si]); si += 1; nxt += step
        elif pi < len(priority):
            out.append(priority[pi]); pi += 1
        elif si < len(sprinkle):
            out.append(sprinkle[si]); si += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rank")
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--calib", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--allow-partial", action="store_true",
                    help="build an instalment from the waves that are "
                         "reviewed so far (see load_rows docstring)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="previous pile.json files; their events are "
                         "never re-served (M13: an instalment must not "
                         "hand the owner the same event twice, and a "
                         "re-served calibration event is no longer blind)")
    ap.add_argument("--drop-unscannable", action="store_true",
                    help="remove 'only dots' events (scannability.py) "
                         "from the served pile AND from the calibration "
                         "pool.  The owner declined 13 of these by hand "
                         "in instalment 1; serving them again spends "
                         "their time on events nobody can answer, and "
                         "auto-accepting them writes training labels "
                         "onto events with no readable vertex.")
    ap.add_argument("--tag", default="instalment")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = load_rows(a.runs, a.allow_partial)
    served = set()
    for f in a.exclude:
        served |= {r["event"] for r in json.load(open(f))["order"]}
    if served:
        n0 = len(rows)
        rows = [r for r in rows if r["event"] not in served]
        print("excluded %d already-served events from %d prior "
              "instalment(s)\n" % (n0 - len(rows), len(a.exclude)))
    if a.drop_unscannable:
        n0 = len(rows)
        dots = [r for r in rows
                if scannability.unscannable(r["dump"])]
        dotset = {r["event"] for r in dots}
        rows = [r for r in rows if r["event"] not in dotset]
        import collections as _c
        print("dropped %d 'only dots' events (longest fitted segment "
              "< %.1f cm) of %d; by bucket: %s\n"
              % (len(dots), scannability.DEFAULT_LONGEST_CM, n0,
                 dict(_c.Counter(r["bucket"] for r in dots))))
    rank = load_rank(a.rank)
    by = {}
    for r in rows:
        by.setdefault(r["bucket"], []).append(r)

    print("=== bucket census over %d scanned events ===" % len(rows))
    for b in (REVIEW_FIRST, ABSTAIN, REVIEW, AUTO,
              "no candidates (empty PR graph)"):
        v = by.get(b, [])
        print("  %-40s %4d  (%.1f%%)" % (b, len(v), 100.0*len(v)/len(rows)))
    # A pick with no reco separation is NOT necessarily an off-list pick --
    # it is also what you get when the dump has `main_vertex: null` (the
    # reconstruction found vertices but selected no main one).  Distinguish
    # them, because `review` buckets the second case as REVIEW FIRST via a
    # "disagreement" with an answer that does not exist.  Off-list picks
    # proper are counted by vtx_rules/b2_checkpoint.py against the candidate
    # list; this is the residual class.
    nosep = [r for r in rows if r.get("vertex_id") is not None
             and r.get("reco_sep_cm") is None]
    print("  picks with no reco separation (main_vertex null) %4d %s"
          % (len(nosep), ", ".join(r["event"] for r in nosep[:6])))

    def hot(r):
        return -rank.get(r["event"], 0.0)

    t1 = sorted(by.get(REVIEW_FIRST, []), key=hot)
    t2 = sorted(by.get(ABSTAIN, []), key=hot)
    auto = sorted(by.get(AUTO, []), key=lambda r: r["event"])

    rng = random.Random(a.seed)
    ncal = min(a.calib, len(auto))
    t3 = sorted(rng.sample(auto, ncal), key=lambda r: r["event"]) if auto else []

    room = max(0, a.cap - len(t1) - len(t2) - len(t3))
    t4 = [r for r in sorted(by.get(REVIEW, []), key=hot) if not r["agrees"]][:room]

    priority = t1 + t2 + t4
    order = interleave(priority, t3)

    os.makedirs(a.out, exist_ok=True)
    tier = {}
    for r in t1: tier[r["event"]] = "1-review-first"
    for r in t2: tier[r["event"]] = "2-abstained"
    for r in t3: tier[r["event"]] = "3-CALIBRATION"
    for r in t4: tier[r["event"]] = "4-fill-ranker-hot"

    with open(os.path.join(a.out, "pile-dumps.txt"), "w") as fh:
        for r in order:
            fh.write(os.path.abspath(r["dump"]) + "\n")
    # The calibration draw, written BEFORE the owner scans.  This file is the
    # whole audit trail for Phase 4's fork; without it the 40 are just "the
    # ones we happened to check".
    json.dump(dict(seed=a.seed, drawn_from_n=len(auto), n=ncal,
                   instalment=a.tag, partial=bool(a.allow_partial),
                   waves=sorted({r["wave"] for r in rows}),
                   events=[r["event"] for r in t3],
                   purpose="doc pr/88 Phase 4: auto-accept precision on mcp2k "
                           "(pr/82 sec 9 gate 5)"),
              open(os.path.join(a.out, "calibration-draw.json"), "w"), indent=1)
    json.dump(dict(cap=a.cap, served=len(order),
                   tiers={k: sum(1 for v in tier.values() if v == k)
                          for k in sorted(set(tier.values()))},
                   order=[dict(i=i, event=r["event"], tier=tier[r["event"]],
                               bucket=r["bucket"], conf=r["conf"],
                               vertex_id=r["vertex_id"],
                               reco_sep_cm=r["reco_sep_cm"],
                               rank=rank.get(r["event"]), wave=r["wave"],
                               why=r["why"], dump=r["dump"])
                          for i, r in enumerate(order)]),
              open(os.path.join(a.out, "pile.json"), "w"), indent=1)

    print("\n=== pile: %d events (cap %d) ===" % (len(order), a.cap))
    for k in sorted(set(tier.values())):
        print("  %-20s %3d" % (k, sum(1 for v in tier.values() if v == k)))
    print("  NOT served (stay unlabelled): %d REVIEW + %d auto-accept"
          % (len(by.get(REVIEW, [])) - len(t4), len(auto) - ncal))
    print("\nwrote %s/{pile-dumps.txt,pile.json,calibration-draw.json}" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

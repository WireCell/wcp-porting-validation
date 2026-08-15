"""Freeze the dev/test split for the vertex hand-scan rule round (doc pr/80).

Run ONCE, before any rule exists, and commit the output.  A holdout drawn after
the rules are written is not a holdout.

  python3 vtx_rules/make_split.py            # writes runs/split-20260815.tsv
  python3 vtx_rules/make_split.py --check     # re-derives and compares, no write

Stratification is on (scan tag, is-the-scanned-arm-already-correct).  The second
key uses the label's OWN recorded main_vertex (the min_accept=4.0 arm the scan
was taken on), not the currently deployed arm: it is carried inside every label
file, so the split is reproducible from the labels alone and cannot shift if an
arm directory is rebuilt or retired.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtx_io                                                    # noqa: E402

SEED = 20260815
N_DEV = 150
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "runs", "split-20260815.tsv")
COLS = ["half", "tag", "runNo", "subRunNo", "eventNo", "event",
        "stratum", "b1_cm"]


def build():
    labels = vtx_io.load_labels()
    # Sort by the full join key first: python's dict/glob order is stable here,
    # but an explicit total order makes the split independent of filesystem
    # enumeration, which is the kind of thing that silently changes on a copy.
    labels.sort(key=lambda d: (d["tag"], d.get("runNo") or -1,
                               d.get("subRunNo") or -1, d.get("eventNo") or -1))

    strata = {}
    for d in labels:
        # b1 is None only when the label has no main_vertex recorded
        # (labels-evt287431.json).  That is its own stratum, not a silent
        # merge into "wrong".
        if d["b1"] is None:
            ok = "none"
        else:
            ok = "ok" if vtx_io.correct(d["b1"]) else "miss"
        strata.setdefault((d["tag"], ok), []).append(d)

    rng = random.Random(SEED)
    frac = float(N_DEV) / len(labels)
    dev = []
    # Deterministic order over strata, then a seeded shuffle inside each.
    for key in sorted(strata):
        grp = strata[key]
        rng.shuffle(grp)
        # Proportional allocation, rounded so the total lands on N_DEV: take
        # the floor per stratum and hand the remainder to the largest ones.
        n = int(len(grp) * frac)
        dev.append((key, grp, n))

    short = N_DEV - sum(n for _, _, n in dev)
    for i in sorted(range(len(dev)), key=lambda i: -len(dev[i][1]))[:max(short, 0)]:
        key, grp, n = dev[i]
        dev[i] = (key, grp, n + 1)

    rows = []
    for key, grp, n in dev:
        for i, d in enumerate(grp):
            rows.append(dict(
                half="dev" if i < n else "test",
                tag=d["tag"], runNo=d.get("runNo"), subRunNo=d.get("subRunNo"),
                eventNo=d.get("eventNo"), event=d["event"],
                stratum="%s/%s" % key,
                b1_cm=("%.4f" % d["b1"]) if d["b1"] is not None else ""))
    rows.sort(key=lambda r: (r["tag"], r["runNo"] or -1, r["subRunNo"] or -1,
                             r["eventNo"] or -1))
    return rows


def render(rows):
    out = ["\t".join(COLS)]
    for r in rows:
        out.append("\t".join("" if r[c] is None else str(r[c]) for c in COLS))
    return "\n".join(out) + "\n"


def load_split(path=OUT):
    """half -> set of (tag, runNo, subRunNo, eventNo) keys."""
    halves = {}
    with open(path) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            v = dict(zip(cols, line.rstrip("\n").split("\t")))
            key = (v["tag"], int(v["runNo"]), int(v["subRunNo"]),
                   int(v["eventNo"]))
            halves.setdefault(v["half"], set()).add(key)
    return halves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="re-derive and compare against the committed file")
    args = ap.parse_args()

    rows = build()
    text = render(rows)
    n_dev = sum(1 for r in rows if r["half"] == "dev")

    if args.check:
        with open(OUT) as fh:
            have = fh.read()
        same = have == text
        print("split reproduces byte-identically: %s" % same)
        return 0 if same else 1

    if os.path.exists(OUT):
        # The split is a scientific record (CLAUDE.md M13): once it is written
        # and rules have been fitted against it, silently redrawing it would
        # invalidate every number in doc pr/80 with no trace.
        print("refusing to overwrite %s -- it is already frozen.\n"
              "Use --check to verify it still reproduces." % OUT)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print("wrote %s: %d events, %d dev / %d test" %
          (OUT, len(rows), n_dev, len(rows) - n_dev))
    from collections import Counter
    c = Counter((r["stratum"], r["half"]) for r in rows)
    for k in sorted(c):
        print("   %-32s %-5s %4d" % (k[0], k[1], c[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

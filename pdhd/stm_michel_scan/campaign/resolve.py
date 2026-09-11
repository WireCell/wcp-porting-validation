#!/usr/bin/env python3
"""Collapse ROUND_DIR/v_parts/<scanner>/ into ONE record per item, deterministically.

    resolve.py ROUND_DIR ITEMS_FILE [--double-prefix dbl]

doc pdhd/18.  Ported from the doc pdvd/55 tranche-2 scratch copy.  There a
duplicate was always an accident (a wave-list race), and the tie-break was "the
record matching the label the app wrote" -- which needs the apply to have run
first.  Here the order is resolve -> apply, and duplicates come in two kinds:

  * DELIBERATE: waves whose name starts with --double-prefix are the seeded
    re-scan (reproducibility).  They never enter the resolved record; they are
    written to ROUND_DIR/double_scan.tsv beside the primary record.
  * ACCIDENTAL: any other key written twice.  Identical verdict/kind/tags ->
    the EARLIEST written record is kept (sorted by the record's own `written`
    stamp, then path -- never filesystem order).  Disagreeing -> the earliest is
    kept too, and the pair is listed in ROUND_DIR/duplicates.tsv; a disagreement
    is data (doc pdvd/55 sec 0), not corruption, and it must be visible.

Refuses if any item of ITEMS_FILE has no primary record, or any record names a
key not in ITEMS_FILE.
"""
import argparse, collections, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("round"); ap.add_argument("items")
ap.add_argument("--double-prefix", default="dbl")
ap.add_argument("--supersede", default=None,
                help="file of '<key>\\t<reason>' lines: for these keys the LATEST record wins "
                     "(a targeted re-scan after a factual rubric correction, doc pdhd/18 sec 6); "
                     "the superseded records are listed in duplicates.tsv, never deleted")
ap.add_argument("--partial", action="store_true",
                help="resolve the items that HAVE a record and skip the rest (pipelined "
                     "apply between waves); the final run must be without it")
a = ap.parse_args()

PARTS = os.path.join(a.round, "v_parts")
OUT = os.path.join(a.round, "v_resolved")
want = [l.strip() for l in open(a.items) if l.strip() and not l.startswith("#")]

prim, dbl = collections.defaultdict(list), collections.defaultdict(list)
for dd in sorted(os.listdir(PARTS)):
    p = os.path.join(PARTS, dd)
    if not os.path.isdir(p):
        continue
    for fn in sorted(os.listdir(p)):
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(p, fn)))
            (dbl if dd.startswith(a.double_prefix) else prim)[r["key"]].append((dd, r))

extra = sorted(set(prim) - set(want))
missing = [k for k in want if k not in prim]
if extra:
    sys.exit("records for keys not in %s: %s" % (a.items, extra[:10]))
if missing and not a.partial:
    sys.exit("%d item(s) have no primary record: %s" % (len(missing), missing[:10]))
if missing:
    print("--partial: %d item(s) not yet scanned are skipped" % len(missing))
    want = [k for k in want if k in prim]


def sig(r):
    return json.dumps([r["verdict"], r["michel_kind"], r["tags"], r.get("pin_rr")],
                      sort_keys=True)


SUP = {}
if a.supersede:
    for l in open(a.supersede):
        if l.strip() and not l.startswith("#"):
            k, _, why = l.rstrip("\n").partition("\t")
            SUP[k] = why or "superseded"

os.makedirs(OUT, exist_ok=True)
# v_resolved is a pure function of v_parts: rebuild it from clean every run, so a
# record resolved by an earlier --partial run can never survive into a later one
for fn in os.listdir(OUT):
    if fn.endswith(".json"):
        os.remove(os.path.join(OUT, fn))
dup_rows, n_dup, n_dis, n_sup = [], 0, 0, 0
PICK = {}
for k in want:
    c = sorted(prim[k], key=lambda t: (t[1].get("written", ""), t[0]))
    pick = c[-1] if k in SUP else c[0]
    if k in SUP and len(c) < 2:
        if a.partial:        # its re-scan has not landed yet: not final, skip it
            continue
        sys.exit("%s is listed to be superseded but has only %d record" % (k, len(c)))
    if len(c) > 1:
        n_dup += 1
        n_sup += k in SUP
        if len({sig(r) for _, r in c}) > 1:
            n_dis += 1
        for dd, r in c:
            res = "KEPT" if (dd, r) == pick else "dropped"
            if k in SUP:
                res = ("KEPT (re-scan: %s)" % SUP[k]) if (dd, r) == pick else "superseded"
            dup_rows.append("\t".join([k, dd, r["verdict"], r["michel_kind"], r["confidence"],
                                       str(r.get("pin_rr", "")), res]))
    PICK[k] = pick
    with open(os.path.join(OUT, k.replace("/", "_") + ".json"), "w") as fh:
        json.dump(dict(pick[1], wave=pick[0]), fh, indent=1, ensure_ascii=False)

with open(os.path.join(a.round, "duplicates.tsv"), "w") as fh:
    fh.write("# items with more than one primary record (not the deliberate double scan): the earliest\n"
             "# `written` is kept, except keys in --supersede, where the re-scan is kept\n")
    fh.write("key\twriter\tverdict\tmichel_kind\tconfidence\tpin_rr\tresolution\n")
    fh.write("".join(r + "\n" for r in dup_rows))
with open(os.path.join(a.round, "double_scan.tsv"), "w") as fh:
    fh.write("# deliberate seeded re-scan (waves %s*): primary vs re-scan\n" % a.double_prefix)
    fh.write("key\tprimary_writer\tprimary_verdict\tprimary_kind\tprimary_conf\tprimary_pin_rr"
             "\trescan_writer\trescan_verdict\trescan_kind\trescan_conf\trescan_pin_rr\n")
    for k in sorted(dbl):
        if k not in PICK:           # --partial: its primary is not resolved yet
            continue
        pw = PICK[k]
        for dd, r in dbl[k]:
            fh.write("\t".join([k, pw[0], pw[1]["verdict"], pw[1]["michel_kind"], pw[1]["confidence"],
                                str(pw[1].get("pin_rr", "")), dd, r["verdict"], r["michel_kind"],
                                r["confidence"], str(r.get("pin_rr", ""))]) + "\n")
print("resolved %d items into %s (%d awaiting their re-scan)" % (len(PICK), OUT, len(want) - len(PICK)))
print("  multiply-scanned items: %d (disagreeing %d, superseded by a re-scan %d) -> duplicates.tsv"
      % (n_dup, n_dis, n_sup))
print("  deliberate re-scans: %d items -> double_scan.tsv" % len(dbl))
print("  verdicts: %s" % dict(collections.Counter(p[1]["verdict"] for p in PICK.values())))

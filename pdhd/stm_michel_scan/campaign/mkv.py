#!/usr/bin/env python3
"""Write one verdict record, filling in every object row the display draws.

    mkv.py KEY VERDICT KIND CONF --shots-dir D --out-dir D \
           --tags 'muon:a,b michel:c gamma:d' --default 'delta / other' \
           --evidence '...' [--notes '...'] [--pin-rr 6.0]

doc pdhd/18.  Ported from the doc pdvd/55 tranche-2 scratch copy (never
committed).  Unchanged: the alphabets, the every-row rule, the kind-from-tags
rule, the 12-word evidence floor.  Added, each from a defect doc pdvd/55
sec 17.4 recorded:
  * the verdict must agree with the tags (3 PDVD records shipped STM_MICHEL with
    no michel tag): STM_MICHEL needs a michel tag, STM_ONLY may not carry one,
    THRU / FRAG_THRU / MESSY carry kind `none`;
  * `straddles the stop` is refused (used 0 of 569 times; contradicts the
    no-split rule);
  * the record is stamped with the rubric sha (ROUND_DIR/RUBRIC.sha, found from
    --out-dir's parent chain or $H18_ROUND) and the writing scanner (the out dir
    name), so a record always names the rubric text it was judged under.

The point of --default is the owner's instruction to tag EVERY drawn object:
the object list comes from the item's own context.json, so a row cannot be
missed by forgetting to type it.  A record with an untagged row is refused.
"""
import argparse, datetime, json, os, sys

VERDICTS = {"STM_MICHEL", "STM_ONLY", "THRU", "MESSY", "UNCLEAR",
            "FRAG_STM_MICHEL", "FRAG_STM_ONLY", "FRAG_THRU"}
KINDS = {"none", "attached", "detached dots", "both"}
TAGS = {"muon", "michel", "gamma", "delta / other"}
CONFS = {"high", "medium", "low"}


def rubric_sha(out_dir):
    d = os.path.abspath(out_dir)
    cands = [os.environ.get("H18_ROUND", "")]
    while d and d != "/":
        cands.append(d)
        d = os.path.dirname(d)
    for c in cands:
        if c and os.path.exists(os.path.join(c, "RUBRIC.sha")):
            return open(os.path.join(c, "RUBRIC.sha")).read().split()[0]
    return None


ap = argparse.ArgumentParser()
ap.add_argument("key"); ap.add_argument("verdict")
ap.add_argument("kind"); ap.add_argument("conf")
ap.add_argument("--shots-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--tags", default="")
ap.add_argument("--default", default=None)
ap.add_argument("--evidence", required=True)
ap.add_argument("--notes", default=None)
ap.add_argument("--pin-rr", type=float, default=None)
a = ap.parse_args()

if a.verdict not in VERDICTS:
    sys.exit("bad verdict %r; one of %s" % (a.verdict, sorted(VERDICTS)))
if a.kind not in KINDS:
    sys.exit("bad michel_kind %r; one of %s" % (a.kind, sorted(KINDS)))
if a.conf not in CONFS:
    sys.exit("bad confidence %r; one of %s" % (a.conf, sorted(CONFS)))
if len(a.evidence.split()) < 12:
    sys.exit("evidence is %d words; write the real paragraph"
             % len(a.evidence.split()))

cpath = os.path.join(a.shots_dir, a.key.replace("/", "_"), "context.json")
if not os.path.exists(cpath):
    sys.exit("no context.json for %s at %s -- check the key (event/cluster)" % (a.key, cpath))
ctx = json.load(open(cpath))
rows = [o["key"] for o in ctx["objects"]]

tags = {}
for grp in a.tags.split():
    if not grp.strip():
        continue
    g, _, ks = grp.partition(":")
    g = {"other": "delta / other"}.get(g, g)
    if g in ("straddle", "straddles the stop"):
        sys.exit("`straddles the stop` is not used in this round (rubric, Attribution): "
                 "tag the row for what most of it is and say so in the evidence")
    if g not in TAGS:
        sys.exit("bad tag group %r; one of muon, michel, gamma, other" % g)
    for k in ks.split(","):
        if k.strip():
            tags[k.strip()] = g
unknown = [k for k in tags if k not in rows]
if unknown:
    sys.exit("%s: tagged rows that are not in the table: %s\n  table: %s"
             % (a.key, unknown, rows))
if a.default:
    if a.default not in TAGS:
        sys.exit("bad --default %r" % a.default)
    for k in rows:
        tags.setdefault(k, a.default)
miss = [k for k in rows if k not in tags]
if miss:
    sys.exit("%s: %d of %d rows untagged: %s" % (a.key, len(miss), len(rows), miss))

# michel_kind is mechanical from the tags (rubric, michel_kind).
mi = "michel" in tags.values()
ga = "gamma" in tags.values()
want_kind = ("both" if (mi and ga) else "attached" if mi
             else "detached dots" if ga else "none")
if a.kind != want_kind:
    sys.exit("%s: michel_kind %r contradicts the tags (michel=%s gamma=%s "
             "=> %r).  Either the kind or a tag is wrong; fix one and re-run."
             % (a.key, a.kind, mi, ga, want_kind))

# the verdict against the tags (doc pdvd/55 sec 17.4 item 1)
if a.verdict in ("STM_MICHEL", "FRAG_STM_MICHEL") and not mi:
    sys.exit("%s: %s with no row tagged michel -- tag the electron, or the verdict "
             "is STM_ONLY" % (a.key, a.verdict))
if a.verdict in ("STM_ONLY", "FRAG_STM_ONLY") and mi:
    sys.exit("%s: %s with a row tagged michel -- the verdict is STM_MICHEL, or the "
             "tag is wrong" % (a.key, a.verdict))
if a.verdict in ("THRU", "FRAG_THRU", "MESSY") and want_kind != "none":
    sys.exit("%s: %s carries kind none -- michel/gamma are defined relative to a stop "
             "(rubric, Attribution); tag those rows `other`, or revisit the verdict"
             % (a.key, a.verdict))
if a.verdict == "MESSY" and any(t != "delta / other" for t in tags.values()):
    sys.exit("%s: MESSY tags every row `delta / other` (rubric, Attribution)" % a.key)

rec = dict(key=a.key, verdict=a.verdict, michel_kind=a.kind,
           confidence=a.conf, evidence=a.evidence, tags=tags,
           scanner=os.path.basename(os.path.abspath(a.out_dir)),
           rubric_sha=rubric_sha(a.out_dir),
           written=datetime.datetime.now().isoformat(timespec="seconds"))
if a.notes:
    rec["notes"] = a.notes
if a.pin_rr is not None:
    rec["pin_rr"] = a.pin_rr
os.makedirs(a.out_dir, exist_ok=True)
p = os.path.join(a.out_dir, a.key.replace("/", "_") + ".json")
with open(p + ".tmp", "w") as fh:
    json.dump(rec, fh, indent=1, ensure_ascii=False)
os.replace(p + ".tmp", p)       # a killed writer never leaves a short record
print("wrote %s  %s / %s  %d tags over %d rows"
      % (a.key, a.verdict, a.kind, len(tags), len(rows)))

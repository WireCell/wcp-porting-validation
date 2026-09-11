#!/usr/bin/env python3
"""doc pdvd/90 -- build the `smx7` blind re-judge: doc 89's part (b1), the 22 missed stoppers that
production rejects on the dQ/dx shape tests alone with no Michel object, on a single scanner's
medium- or low-confidence verdict (doc 55 tranche 2, an agent scan), plus a control group.

Fork by duplication (CLAUDE.md M10) of d70_build_smx4.py (the multi-item shuffled blind) with
d88_build_smx6.py's refusals.

  * BLIND: every item reads the same in production (is_stm 0, michel_found 0, shape bits only) --
    asserted, because the display shows production's answer.  A set of record stoppers alone would
    still give the answer away, so CONTROLS are mixed in: record THRU items at medium / low
    confidence from the same agent scan with the same production reading, drawn with a fixed seed.
    The owner's verdict on a control folds into the record like any other.
  * The order is shuffled (fixed seed); every sheet row is tranche 1; no question carries the record
    verdict, its evidence or its pin (the scanner's pin_rr would show the answer); the group, the
    record verdict and the offline reading live only in the --key file, read at scoring time.
  * The payload is production's (p88vprod), taken before doc 90's flip.  Doc 90's twin says the flip
    moves none of these items except, possibly, 039349_26/34 through doc 75's geometric re-read.

Usage (STM_SCAN_RECORD must name the merged smx1a+smx3+smx4+smx5+smx6 record):
  d90_build_smx7.py --census /home/xqian/tmp/p89/census.json --prep PREP --outprep DIR --sheet TSV \\
                    --questions JSON --key TSV [--n-controls 8] [--seed 90] [--force-questions]
--force-questions rewrites ONLY the questions JSON of an existing set (wording);
the prep dir, the sheet and the key are never rebuilt in place (M13).
"""
import argparse, json, os, random, shutil, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
for k in ("census", "prep", "outprep", "sheet", "questions", "key"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--n-controls", type=int, default=8)
ap.add_argument("--seed", type=int, default=90)
ap.add_argument("--force-questions", action="store_true")
a = ap.parse_args()

if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a+smx3+smx4+smx5+smx6 record (doc 88 sec 9.2)")
if a.force_questions:
    for p in (a.outprep, a.sheet, a.key):
        if not os.path.exists(p):
            sys.exit("--force-questions needs the existing set; %s is missing" % p)
else:
    if os.path.exists(a.outprep) and os.listdir(a.outprep):
        sys.exit("REFUSING: %s exists and is not empty (a scan set is never rebuilt in place)" % a.outprep)
    for p in (a.sheet, a.questions, a.key):
        if os.path.exists(p):
            sys.exit("REFUSING: %s exists" % p)

rec = C.load_record()
SHAPE = {"no_bragg", "shape_flat", "plateau_off_mip", "profile_sparse"}


def payfile(k):
    ev, c = k.split("/")
    return os.path.join(a.prep, "smprep-%s-c%s.json" % (ev, c))


def reads_alike(k):
    """production's reading that every item must share: no stopper, no Michel, shape bits only."""
    if not os.path.exists(payfile(k)): return False
    v = json.load(open(payfile(k)))["verdict"]
    rn = set(C.reject_names(v))
    return int(v["is_stm"]) == 0 and int(v["michel_found"]) == 0 and rn and rn <= SHAPE


b1 = json.load(open(a.census))["partition"]["b1"]
assert len(b1) == 22, len(b1)
for k in b1:
    r = rec[k]
    assert C.judged(r) and C.is_stopper(r) and r.get("confidence") in ("medium", "low") and r.get("source") == "smx1a", k
    assert reads_alike(k), "%s: production does not read is_stm 0 / michel_found 0 / shape bits only" % k
pool = sorted(k for k, r in rec.items() if C.judged(r) and C.bare(r["verdict"]) == "THRU" and r.get("source") == "smx1a"
              and r.get("confidence") in ("medium", "low") and reads_alike(k))
controls = sorted(random.Random(a.seed).sample(pool, a.n_controls))
print("control pool: %d record-THRU items (smx1a, medium/low) reading like the 22; drew %d with seed %d"
      % (len(pool), len(controls), a.seed))

GLOSS = {
    "no_bragg": "the dQ/dx at the end does not rise enough above the muon's plateau",
    "shape_flat": "the dQ/dx shape near the end fits a flat MIP profile better than a Bragg peak",
    "profile_sparse": "too few live dQ/dx points near the end to judge",
    "plateau_off_mip": "the muon's plateau dQ/dx is outside 0.6-1.6 MIP",
}

INSTR = (
    "<b>Please judge</b> from the picture (3-D view, measurement tab, dQ/dx panel):<br>"
    "<b>1. The verdict</b> (one button; the click saves the row)"
    "<ul style='margin:2px 0 2px 0'>"
    "<li><b>STM + MICHEL</b> &mdash; the muon stops inside the detector and an electron starts at its "
    "end. Then set the Michel radio to <i>attached</i>, <i>detached dots</i> or <i>both</i>.</li>"
    "<li><b>STM, no Michel</b> &mdash; the muon stops, but you see no Michel.</li>"
    "<li><b>THRU</b> &mdash; the muon does not stop here: it goes on, or leaves through a wall, the "
    "cathode or a dead region.</li>"
    "<li><b>FRAG &rarr; &hellip;</b> if this cluster is only part of the muon (give the whole muon's "
    "verdict); <b>MESSY</b> if it is not one track; <b>UNCLEAR</b> if you cannot tell.</li></ul>"
    "<b>2. The pin</b>: if the muon stops but its true end is not where the fit ends (the fit runs on "
    "past the Bragg peak, or stops short of it), move the pin to the true end and press "
    "<b>SAVE this item</b>.<br>"
    "<b>3. Notes</b> (optional): anything that decided it, e.g. &ldquo;Bragg peak 4 cm before the fit "
    "end&rdquo;, &ldquo;exits the cathode&rdquo;. Then <b>next unlabelled &gt;&gt;</b>.")


def html(v):
    why = "; ".join("<b>%s</b> (%s)" % (r, GLOSS.get(r, "")) for r in C.reject_names(v))
    return ("<b>smx7 &mdash; does this muon stop inside the detector?</b> (doc pdvd/89 &sect;9, a blind "
            "re-judge)<br>Today's production <b>does not call it a stopper</b> (is_stm 0) and "
            "<b>finds no Michel</b>; it rejects the end on %s. Judge the muon, not the chain.<br>%s" % (why, INSTR))


items = {}
for grp, ks in (("b1", b1), ("control", controls)):
    for k in ks:
        p = json.load(open(payfile(k)))
        items[k] = dict(group=grp, pay=p, v=p["verdict"], rec=rec[k])

order = sorted(items)
random.Random(a.seed).shuffle(order)
with open(a.questions, "w") as fh:
    json.dump({"scan": "pdvd smx7, the blind re-judge of doc 89's (b1) missed stoppers plus controls (doc pdvd/90)",
               "prep": a.prep, "items": {k: dict(html=html(items[k]["v"]), anchor_rr_cm=None) for k in order}}, fh, indent=1)
if a.force_questions:
    print("rewrote %s only (%d items)" % (a.questions, len(order)))
    sys.exit(0)

os.makedirs(a.outprep, exist_ok=True)
shutil.copy2(os.path.join(a.prep, "dqdx_ref_pdvd.json"), a.outprep)
rows, krows = [], []
for sid, k in enumerate(order, 1):
    it = items[k]
    shutil.copy2(payfile(k), a.outprep)
    ev, c = k.split("/")
    rows.append("%d\t1\t%s\t%s\t%d\t%.1f" % (sid, ev, c, int(it["pay"]["npts"]), float(it["pay"]["muon_len_cm"])))
    v, r = it["v"], it["rec"]
    krows.append("%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
        sid, k, it["group"], r["verdict"], r.get("michel_kind"), r.get("confidence"),
        "" if r.get("pin_rr") is None else "%.1f" % r["pin_rr"], "+".join(C.reject_names(v)),
        "%.2f" % ((v["plateau_med"] or 0) / 55000.0)))
with open(a.sheet, "w") as fh:
    fh.write("# doc pdvd/90 -- smx7, PDVD: doc 89's (b1) re-judge + %d controls. Production (p88vprod) reads "
             "is_stm 0 / michel_found 0 / shape bits only on every item; order shuffled (seed %d); the group key "
             "is kept in a separate file.\n" % (len(controls), a.seed))
    fh.write("# built by d90_build_smx7.py from prep=%s\n" % a.prep)
    fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
    fh.write("\n".join(rows) + "\n")
with open(a.key, "w") as fh:
    fh.write("# doc pdvd/90 -- smx7 KEY (do not show the scanner). group: b1 = record stopper doc 89 could not "
             "grade (medium/low agent verdict), control = record THRU with the same production reading. "
             "record_pin_rr = the agent's overshoot reading (not shown).\n")
    fh.write("scan_id\tkey\tgroup\trecord_verdict\trecord_michel_kind\trecord_confidence\trecord_pin_rr\t"
             "reject_names\tplateau_mip\n")
    fh.write("\n".join(krows) + "\n")
print("items %d: b1 %d + controls %d; order seed %d" % (len(order), len(b1), len(controls), a.seed))

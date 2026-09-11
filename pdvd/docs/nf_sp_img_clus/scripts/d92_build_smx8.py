#!/usr/bin/env python3
"""doc pdvd/92 -- build the `smx8` blind re-judge: the DECISION SET of the wide Bragg-peak read
(doc 91 sec 10 item 1), plus controls.

WHAT THE TRANCHE IS.  d92_twin.py section 4 names every candidate that any of the four candidate
operating points (W/R/D = 8/1.3/0.8, 8/1.5/0.8, 8/1.3/1.0, 10/1.3/0.8) would turn from "not a
stopper" into "a stopper".  Those are the only items whose true verdict can still change the owner's
pick between them, so they are the tranche.

WHAT IT DOES NOT DO.  The tranche is chosen BY the quantity being measured -- distance to the R / D
boundary -- so folding its verdicts back in and re-running the grid would hand back a new "best
point" that is pure circularity.  The operating point stays doc 91's pre-registered W 8 / R 1.3 /
D 0.8.  The scan answers two factual questions (pred.txt): is 039349_81/25 through-going (R 1.3 vs
R 1.5), and are the four d 0.93-0.97 items through-going (whether D 1.0 could ever be admissible).

BLIND.  Every item must read the SAME in production -- is_stm 0, michel_found 0, shape bits only --
because the display shows production's answer; an item that reads differently would mark itself out.
An item of the decision set that fails that test is DROPPED and named (039253_12/93 has
michel_found 1).  Record-verdict CONTROLS that read alike and that no grid point moves are mixed in
with a fixed seed, so the set is not "the items the rule would move".  The order is shuffled; no
question carries the record verdict, its evidence or its pin; the group, the record verdict and the
offline reading live only in the --key file, read at scoring time.

The payload is production's own (doc 90's p90vprod), so the pictures are what production reconstructs
today -- the rule changes no geometry.

Fork by duplication (CLAUDE.md M10) of d90_build_smx7.py; d90's script stays as doc 90's record.

Usage (STM_SCAN_RECORD must name the merged smx1a..smx7 record):
  d92_build_smx8.py --twin /home/xqian/tmp/p92/twin.json --prep PREP --outprep DIR --sheet TSV \\
                    --questions JSON --key TSV [--n-controls 8] [--seed 92] [--force-questions]
--force-questions rewrites ONLY the questions JSON of an existing set (wording);
the prep dir, the sheet and the key are never rebuilt in place (M13).
"""
import argparse, json, os, random, shutil, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
for k in ("twin", "prep", "outprep", "sheet", "questions", "key"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--n-controls", type=int, default=8)
ap.add_argument("--seed", type=int, default=92)
ap.add_argument("--force-questions", action="store_true")
a = ap.parse_args()

if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a+smx3+smx4+smx5+smx6+smx7 record (doc 90)")
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
    """production's reading every item must share: no stopper, no Michel, shape bits only."""
    if not os.path.exists(payfile(k)):
        return False
    v = json.load(open(payfile(k)))["verdict"]
    rn = set(C.reject_names(v))
    return int(v["is_stm"]) == 0 and int(v["michel_found"]) == 0 and rn and rn <= SHAPE


TW = json.load(open(a.twin))
if TW.get("bad"):
    sys.exit("the twin disagrees with production on %d item(s); fix that before building a scan on it" % len(TW["bad"]))
dec_all = sorted(TW["decision"])
decision = [k for k in dec_all if reads_alike(k)]
for k in dec_all:
    if k not in decision:
        v = json.load(open(payfile(k)))["verdict"]
        print("DROPPED from the tranche (does not read like the rest, so it would not be blind): "
              "%s -- is_stm %d michel_found %d %s" % (k, v["is_stm"], v["michel_found"], "+".join(C.reject_names(v))))
if not decision:
    sys.exit("no decision item survives the blindness test")

# controls: judged, read alike, and NOT in the decision set (no candidate point moves them).
pool = sorted(k for k, r in rec.items()
              if C.judged(r) and k not in dec_all and reads_alike(k))
controls = sorted(random.Random(a.seed).sample(pool, a.n_controls))
print("decision set %d of %d; control pool %d items reading alike, drew %d with seed %d"
      % (len(decision), len(dec_all), len(pool), len(controls), a.seed))

GLOSS = {
    "no_bragg": "the dQ/dx at the end does not rise enough above the muon's plateau",
    "shape_flat": "the dQ/dx shape near the end fits a flat MIP profile better than a Bragg peak",
    "profile_sparse": "too few live dQ/dx points near the end to judge",
    "plateau_off_mip": "the muon's plateau dQ/dx is outside 0.6-2.0 MIP",
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
    "<b>3. Notes</b> (optional): anything that decided it, e.g. &ldquo;Bragg peak 6 cm before the fit "
    "end&rdquo;, &ldquo;exits the cathode&rdquo;, &ldquo;a second track carries on&rdquo;. Then "
    "<b>next unlabelled &gt;&gt;</b>.")


def html(v):
    why = "; ".join("<b>%s</b> (%s)" % (r, GLOSS.get(r, "")) for r in C.reject_names(v))
    return ("<b>smx8 &mdash; does this muon stop inside the detector?</b> (doc pdvd/92, a blind "
            "re-judge)<br>Today's production <b>does not call it a stopper</b> (is_stm 0) and "
            "<b>finds no Michel</b>; it rejects the end on %s. Judge the muon, not the chain.<br>"
            "Several of these tracks have charge well before the fit's last point &mdash; the fit may "
            "run past the true stop. <b>The pin is what says where the muon really ends.</b><br>%s"
            % (why, INSTR))


items = {}
for grp, ks in (("decision", decision), ("control", controls)):
    for k in ks:
        p = json.load(open(payfile(k)))
        items[k] = dict(group=grp, pay=p, v=p["verdict"], rec=rec[k])

order = sorted(items)
random.Random(a.seed).shuffle(order)
with open(a.questions, "w") as fh:
    json.dump({"scan": "pdvd smx8, the blind re-judge of the wide-anchor decision set plus controls (doc pdvd/92)",
               "prep": a.prep,
               "items": {k: dict(html=html(items[k]["v"]), anchor_rr_cm=None) for k in order}}, fh, indent=1)
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
    pts = " ".join("%g/%g/%g" % tuple(t) for t in TW["decision"].get(k, []))
    krows.append("%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
        sid, k, it["group"], r["verdict"], r.get("michel_kind"), r.get("confidence"),
        "" if r.get("pin_rr") is None else "%.1f" % r["pin_rr"], "+".join(C.reject_names(v)),
        "%.2f" % ((v["plateau_med"] or 0) / 55000.0), pts))
with open(a.sheet, "w") as fh:
    fh.write("# doc pdvd/92 -- smx8, PDVD: the wide-anchor decision set (%d) + %d controls. Production "
             "(p90vprod) reads is_stm 0 / michel_found 0 / shape bits only on every item; order shuffled "
             "(seed %d); the group key is kept in a separate file.\n" % (len(decision), len(controls), a.seed))
    fh.write("# built by d92_build_smx8.py from prep=%s twin=%s\n" % (a.prep, a.twin))
    fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
    fh.write("\n".join(rows) + "\n")
with open(a.key, "w") as fh:
    fh.write("# doc pdvd/92 -- smx8 KEY (do not show the scanner). group: decision = an operating point "
             "would call it a stopper, control = no grid point moves it. moved_at = the W/R/D points that "
             "move it. record_pin_rr = the previous scanner's overshoot reading (not shown).\n")
    fh.write("scan_id\tkey\tgroup\trecord_verdict\trecord_michel_kind\trecord_confidence\trecord_pin_rr\t"
             "reject_names\tplateau_mip\tmoved_at\n")
    fh.write("\n".join(krows) + "\n")
print("items %d: decision %d + controls %d; order seed %d" % (len(order), len(decision), len(controls), a.seed))
print("  decision items: %s" % " ".join(decision))
print("  controls      : %s" % " ".join(controls))

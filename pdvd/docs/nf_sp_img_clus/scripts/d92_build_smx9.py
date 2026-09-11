#!/usr/bin/env python3
"""doc pdvd/92 -- build the `smx9` blind tie-break: the item whose verdict the owner has given twice,
differently, and which is the entire cost of the flip.

WHY A THIRD LOOK.  039349_71/37 was judged STM_MICHEL in smx7 (pin 6.5 cm) and THRU in smx8
(pin 1.8 cm), both blind, both by the owner, 80 minutes apart.  It is a gain of the wide Bragg-peak
read at BOTH operating points, so it alone decides whether the rule is "+5 stoppers, 0 false
positives" or "+4 stoppers, 1 false positive".

WHY BLIND AGAIN RATHER THAN ADJUDICATED.  Showing the two prior calls would anchor the third, and a
third anchored call settles nothing.  Judged blind, the three are independent and a 2-1 majority is
a real tie-break; if the third call splits from both -- or simply feels as uncertain -- that IS the
answer: the item is inside the record's own resolution, and no threshold should be chosen on it.
The prior verdicts are written to the --key file (never shown) so the reveal afterwards is exact.

BLIND.  Every item must read the same in production (is_stm 0, michel_found 0, shape bits only),
because the display shows production's answer.  Controls are drawn with a fixed seed from items that
read alike and that no candidate operating point moves, EXCLUDING everything already in --exclude-key
so nothing judged minutes earlier is served again.

Fork by duplication (CLAUDE.md M10) of d92_build_smx8.py; the population is an explicit list rather
than the twin's decision set.

Usage (STM_SCAN_RECORD must name the merged smx1a..smx8 record):
  d92_build_smx9.py --items 039349_71/37 --prior-record SMX7.json --exclude-key SMX8_KEY.tsv \\
      --prep PREP --outprep DIR --sheet TSV --questions JSON --key TSV [--n-controls 6] [--seed 93]
"""
import argparse, csv, json, os, random, shutil, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
for k in ("prep", "outprep", "sheet", "questions", "key"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--items", required=True, help="comma-separated event/cluster keys")
ap.add_argument("--prior-record", default=None, help="an earlier record, to store the previous verdict in the key")
ap.add_argument("--exclude-key", default=None, help="a key TSV whose items may not be drawn as controls")
ap.add_argument("--n-controls", type=int, default=6)
ap.add_argument("--seed", type=int, default=93)
a = ap.parse_args()

if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_smx7_smx8_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a..smx8 record (doc 92)")
if os.path.exists(a.outprep) and os.listdir(a.outprep):
    sys.exit("REFUSING: %s exists and is not empty (a scan set is never rebuilt in place)" % a.outprep)
for p in (a.sheet, a.questions, a.key):
    if os.path.exists(p):
        sys.exit("REFUSING: %s exists" % p)

rec = C.load_record()
prior = {}
if a.prior_record:
    prior = {v["key"]: v for v in json.load(open(a.prior_record))}
SHAPE = {"no_bragg", "shape_flat", "plateau_off_mip", "profile_sparse"}


def payfile(k):
    ev, c = k.split("/")
    return os.path.join(a.prep, "smprep-%s-c%s.json" % (ev, c))


def reads_alike(k):
    if not os.path.exists(payfile(k)):
        return False
    v = json.load(open(payfile(k)))["verdict"]
    rn = set(C.reject_names(v))
    return int(v["is_stm"]) == 0 and int(v["michel_found"]) == 0 and rn and rn <= SHAPE


items = [k.strip() for k in a.items.split(",") if k.strip()]
for k in items:
    if not reads_alike(k):
        sys.exit("%s does not read like a blind item (is_stm 0 / michel_found 0 / shape bits only)" % k)

excl = set(items)
if a.exclude_key:
    excl |= {r["key"] for r in csv.DictReader([l for l in open(a.exclude_key) if not l.startswith("#")], delimiter="\t")}
pool = sorted(k for k, r in rec.items() if C.judged(r) and k not in excl and reads_alike(k))
controls = sorted(random.Random(a.seed).sample(pool, a.n_controls))
print("tie-break items %d: %s" % (len(items), " ".join(items)))
print("control pool %d (excluding %d already-served items); drew %d with seed %d"
      % (len(pool), len(excl), len(controls), a.seed))

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
    "<b>3. Notes</b> (optional): anything that decided it &mdash; and if you are genuinely unsure, "
    "say so in the notes rather than forcing a call. Then <b>next unlabelled &gt;&gt;</b>.")


def html(v):
    why = "; ".join("<b>%s</b> (%s)" % (r, GLOSS.get(r, "")) for r in C.reject_names(v))
    return ("<b>smx9 &mdash; does this muon stop inside the detector?</b> (doc pdvd/92, a blind "
            "re-judge)<br>Today's production <b>does not call it a stopper</b> (is_stm 0) and "
            "<b>finds no Michel</b>; it rejects the end on %s. Judge the muon, not the chain.<br>"
            "Some of these tracks have charge well before the fit's last point &mdash; the fit may "
            "run past the true stop. <b>The pin is what says where the muon really ends.</b><br>%s"
            % (why, INSTR))


allk = {}
for grp, ks in (("tiebreak", items), ("control", controls)):
    for k in ks:
        p = json.load(open(payfile(k)))
        allk[k] = dict(group=grp, pay=p, v=p["verdict"], rec=rec[k])
order = sorted(allk)
random.Random(a.seed).shuffle(order)

with open(a.questions, "w") as fh:
    json.dump({"scan": "pdvd smx9, the blind tie-break on the wide-anchor cost item plus controls (doc pdvd/92)",
               "prep": a.prep,
               "items": {k: dict(html=html(allk[k]["v"]), anchor_rr_cm=None) for k in order}}, fh, indent=1)

os.makedirs(a.outprep, exist_ok=True)
shutil.copy2(os.path.join(a.prep, "dqdx_ref_pdvd.json"), a.outprep)
rows, krows = [], []
for sid, k in enumerate(order, 1):
    it = allk[k]
    shutil.copy2(payfile(k), a.outprep)
    ev, c = k.split("/")
    rows.append("%d\t1\t%s\t%s\t%d\t%.1f" % (sid, ev, c, int(it["pay"]["npts"]), float(it["pay"]["muon_len_cm"])))
    r = it["rec"]; pr = prior.get(k) or {}
    krows.append("%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
        sid, k, it["group"], r["verdict"], r.get("confidence"), r.get("source", ""),
        pr.get("verdict", ""), "+".join(C.reject_names(it["v"]))))
with open(a.sheet, "w") as fh:
    fh.write("# doc pdvd/92 -- smx9, PDVD: the blind tie-break on %s (judged twice, differently) + %d "
             "controls. Production (p90vprod) reads is_stm 0 / michel_found 0 / shape bits only on every "
             "item; order shuffled (seed %d).\n" % (", ".join(items), len(controls), a.seed))
    fh.write("# built by d92_build_smx9.py from prep=%s\n" % a.prep)
    fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
    fh.write("\n".join(rows) + "\n")
with open(a.key, "w") as fh:
    fh.write("# doc pdvd/92 -- smx9 KEY (do not show the scanner). group: tiebreak = the contested item, "
             "control = settled and moved by no operating point. record_verdict is the CURRENT (smx8) call; "
             "prior_verdict is the earlier record's (smx7) call, so the reveal can show all three.\n")
    fh.write("scan_id\tkey\tgroup\trecord_verdict\trecord_confidence\trecord_source\tprior_verdict\treject_names\n")
    fh.write("\n".join(krows) + "\n")
print("items %d: tie-break %d + controls %d; order seed %d" % (len(order), len(items), len(controls), a.seed))
print("  controls: %s" % " ".join(controls))

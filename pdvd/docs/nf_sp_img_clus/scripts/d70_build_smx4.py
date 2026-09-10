#!/usr/bin/env python3
"""doc pdvd/70 -- build the `smx4` hand-scan set that grades proposal P1
(`topology_stop_evidence`: a good Michel at the stop clears no_bragg + shape_flat).

The set is doc 70 sec 3's "group A": every judged record item on which today's
production (anchor arm d68a3) reads is_stm 0 AND michel_found 1 -- 40 the record
calls stoppers (P1 would recover 26) and 14 the record calls THRU (P1 would admit
4).  The owner re-judges all 54 without being told which is which:

  * production reads the same (is_stm 0, michel_found 1) on every item -- asserted
    here, because the un-blinded display shows production's answer and would
    otherwise leak the group;
  * the order is shuffled with a fixed seed and every sheet row is tranche 1;
  * the question text never carries the record verdict or what P1 would read
    (feedback_blind_the_scan_sheet); the group, the record verdict and P1's
    reading live only in the separate --key file, read at scoring time.

Residual leak, stated: the display's chain-answer box prints the Michel's KE and
length, which are P1's gate variables.  The panel asks the scanner to judge from
the picture.

Four of the 40 were already judged by the owner in smx3 (doc 68); they stay in the
shuffle as a consistency check and are marked rejudge=1 in the key.

Usage (STM_SCAN_RECORD must name the merged smx1a+smx3 record):
  d70_build_smx4.py --cells d70_cells.json --prep PREP --outprep DIR --sheet TSV \\
                    --questions JSON --key TSV [--seed 70] [--force-questions]
--force-questions rewrites ONLY the questions JSON of an existing set (wording);
the prep dir, the sheet and the key are never rebuilt in place (M13).
"""
import argparse, json, os, random, shutil, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
for k in ("cells", "prep", "outprep", "sheet", "questions", "key"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--owner-labels", default="/home/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/scan/"
                "pdvd_stm_michel_smx3_labels.json")
ap.add_argument("--seed", type=int, default=70)
ap.add_argument("--force-questions", action="store_true")
a = ap.parse_args()

if not os.environ.get("STM_SCAN_RECORD", "").endswith("pdvd_stm_michel_smx1a_smx3_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a+smx3 record (doc 68)")
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
cells = json.load(open(a.cells))
owner = set(json.load(open(a.owner_labels))["labels"])


def payfile(k):
    ev, c = k.split("/")
    return os.path.join(a.prep, "smprep-%s-c%s.json" % (ev, c))


def p1_admits(v):
    # doc 70 sec 3, the argued point; same rule as d70_sizing.py section B
    return (v["michel_conn_type"] in (1, 2) and (v["michel_ke_best"] or 0) >= 10.0
            and (v["michel_len"] or 0) >= 3.0 and "stop_near_boundary" not in C.reject_names(v))


items = {}
for grp, src in (("rec_stopper", "FN_michel"), ("rec_thru", "TN_michel")):
    for k in cells[src]:
        p = json.load(open(payfile(k)))
        v = p["verdict"]
        assert int(v["is_stm"]) == 0 and int(v["michel_found"]) == 1, \
            "%s: production does not read is_stm 0 / michel_found 1 -- the blind would leak" % k
        r = rec[k]
        assert C.judged(r) and C.is_stopper(r) == (grp == "rec_stopper"), k
        items[k] = dict(group=grp, pay=p, v=v, rec=r, rejudge=int(k in owner),
                        p1=int(p1_admits(v)), rn=[x for x in C.reject_names(v) if x != "STM"])

n = {g: sum(1 for i in items.values() if i["group"] == g) for g in ("rec_stopper", "rec_thru")}
n1 = {g: sum(i["p1"] for i in items.values() if i["group"] == g) for g in n}
assert (n["rec_stopper"], n["rec_thru"], n1["rec_stopper"], n1["rec_thru"]) == (40, 14, 26, 4), (n, n1)

GLOSS = {
    "no_bragg": "the dQ/dx at the end does not rise enough above the muon's plateau",
    "shape_flat": "the dQ/dx shape near the end fits a flat MIP profile better than a Bragg peak",
    "profile_sparse": "too few live dQ/dx points near the end to judge",
    "stop_near_boundary": "the stop lies inside the fiducial margin of a detector boundary",
    "continuation": "a collinear MIP-like track leaves the stop",
    "vertex_hadron": "a long heavily-ionizing prong leaves the muon body",
}

INSTR = (
    "<b>Please select</b> &mdash; judge from the picture (3-D and wire views), <i>not</i> from "
    "the Michel's energy and length in the chain's-answer box:<br>"
    "<b>1. The verdict</b> (one button)"
    "<ul style='margin:2px 0 2px 0'>"
    "<li><b>STM + MICHEL</b> &mdash; the muon stops inside the detector and an electron starts "
    "at its end. Then set the Michel radio to <i>attached</i>, <i>detached dots</i> or "
    "<i>both</i> (the button is refused while it reads &ldquo;not set&rdquo;).</li>"
    "<li><b>STM, no Michel</b> &mdash; the muon stops, but you see no Michel.</li>"
    "<li><b>THRU</b> &mdash; the muon does not stop here: it goes on past the &ldquo;Michel&rdquo;, "
    "or leaves through a wall or a dead region. The chain's &ldquo;Michel&rdquo; is then a delta "
    "ray, a crossing track or a piece of the muon.</li>"
    "<li><b>FRAG &rarr; &hellip;</b> if this cluster is only part of the muon (give the whole "
    "muon's verdict); <b>MESSY</b> if it is not one track; <b>UNCLEAR</b> if you cannot tell.</li>"
    "</ul>"
    "<b>2. If the muon stops but the chain's Michel is the wrong object</b> (it took a delta ray "
    "or a gamma, and the real Michel is elsewhere or missed), write that in <b>notes</b> "
    "<i>before</i> clicking the verdict.<br>"
    "<b>3. The pin</b>: move it only if the muon's true end is not where the fit ends. A verdict "
    "click saves the row; after moving the pin or editing the notes, press <b>SAVE this item</b>. "
    "Then <b>next unlabelled &gt;&gt;</b>.")


def html(it):
    why = "; ".join("<b>%s</b> (%s)" % (r, GLOSS.get(r, "see the chain's-answer box")) for r in it["rn"])
    return ("<b>smx4 &mdash; can the Michel alone prove the stop?</b> (doc pdvd/70, proposal P1)<br>"
            "Today's production <b>found a Michel</b> at the end of this muon (michel_found 1) but "
            "<b>did not call the muon a stopper</b>: is_stm 0, rejected by %s.<br>%s" % (why, INSTR))


order = sorted(items)
random.Random(a.seed).shuffle(order)

with open(a.questions, "w") as fh:
    json.dump({"scan": "pdvd smx4, P1 topology-first stop evidence (doc pdvd/70)", "prep": a.prep,
               "items": {k: dict(html=html(items[k])) for k in order}}, fh, indent=1)
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
    krows.append("%d\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%.1f\t%.1f\t%s" % (
        sid, k, it["group"], r["verdict"], r.get("confidence"), "owner-smx3" if it["rejudge"] else "smx1a",
        it["rejudge"], it["p1"], v["michel_conn_type"], v["michel_ke_best"], v["michel_len"], ",".join(it["rn"])))
with open(a.sheet, "w") as fh:
    fh.write("# doc pdvd/70 P1 -- smx4, PDVD. Production (d68a3) reads is_stm 0 / michel_found 1 on every "
             "item; order shuffled (seed %d); the group key is kept in a separate file.\n" % a.seed)
    fh.write("# built by d70_build_smx4.py from prep=%s\n" % a.prep)
    fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
    fh.write("\n".join(rows) + "\n")
with open(a.key, "w") as fh:
    fh.write("# doc pdvd/70 P1 -- smx4 KEY (do not show the scanner). group: rec_stopper = record stopper "
             "(an is_stm FN), rec_thru = record THRU (a TN). p1 = P1 at KE>=10 MeV, len>=3 cm clears the item.\n")
    fh.write("scan_id\tkey\tgroup\trecord_verdict\trecord_confidence\trecord_source\trejudge\tp1\t"
             "michel_conn\tmichel_ke\tmichel_len\treject_names\n")
    fh.write("\n".join(krows) + "\n")

print("items %d: record stoppers %d (P1 clears %d), record THRU %d (P1 clears %d); re-judges of owner smx3 %d"
      % (len(order), n["rec_stopper"], n1["rec_stopper"], n["rec_thru"], n1["rec_thru"],
         sum(i["rejudge"] for i in items.values())))

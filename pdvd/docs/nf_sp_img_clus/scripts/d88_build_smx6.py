#!/usr/bin/env python3
"""doc pdvd/88 -- build the `smx6` hand check: the stop-local keep's one record-FP.

Fork by duplication (CLAUDE.md M10) of d86_build_smx5.py.

The flip candidate of doc 88 (stop_local_residual_cm 5 with a 5-terminal / 5 cm
floor, doc 87, and the reachable stop snap, doc 88) turns PR's dropped residual at the stop into a Michel on 039253_15/36,
which the record calls STM_ONLY (the smx1a scanner, high confidence, on a display
that "draws exactly ONE object, nothing past the stop" -- which is what a
dropped residual looks like there, and what made 0/44's old call wrong).  The
owner chose to see it before the flip.

  * BLIND: the payload is PRODUCTION's (p85vwh == p85vprod), so the chain's kept-
    residual Michel is not drawn; production must read michel_found 0 -- asserted;
    the record's call and the arm's reading go only in the --key file;
  * the question is smx5's vocabulary (MECH / DEAD), so the answer lands in the
    same terms as the owner's 0/44 and 30/45.

Usage (STM_SCAN_RECORD must name the merged smx1a+smx3+smx4+smx5 record):
  d88_build_smx6.py --prep PREP --outprep DIR --sheet TSV --questions JSON --key TSV
                    [--force-questions]
--force-questions rewrites ONLY the questions JSON of an existing set (wording);
the prep dir, the sheet and the key are never rebuilt in place (M13).
"""
import argparse, json, os, shutil, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
for k in ("prep", "outprep", "sheet", "questions", "key"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--force-questions", action="store_true")
a = ap.parse_args()

if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a+smx3+smx4+smx5 record (doc 86)")
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

# pred = the answer the arm's reading predicts (doc 88), written before the
# scan; never shown to the scanner.
ITEMS = [
    ("039253_15/36", "unfitted", "none",
     "PR fits a 12-terminal 8.78 cm residual at the stop and drops it as isolated (pr54, production); "
     "p88v5fr keeps it (floor 5/5, radius 5 cm) and T3b makes it the Michel. d86 sec 8.1 U2: 77 own "
     "points in excess of the body rate within 10 cm of the fit end, the largest on an STM_ONLY item. "
     "Record: STM_ONLY (smx1a, high)"),
]

VOCAB = (
    "<ul style='margin:2px 0 2px 0'>"
    "<li><code>fit-through</code> &mdash; the Michel is the muon fit's own last few cm: the fit runs past "
    "the true stop and on through the Michel (put the pin at the junction).</li>"
    "<li><code>unfitted</code> &mdash; the Michel is imaged (3-D points beside or beyond the fit end, in "
    "this cluster) but no fitted segment follows it.</li>"
    "<li><code>other-cluster</code> &mdash; it is imaged in a different cluster (say which, e.g. c198).</li>"
    "<li><code>dead-region</code> &mdash; it runs into a dead band (grey in the measurement tab) or across "
    "a CRP/CRU seam (the stopping-point badge prints the nearest seam), and its 3-D image is missing or "
    "broken there.</li>"
    "<li><code>not-imaged</code> &mdash; you believe there is Michel charge the 3-D image does not show at "
    "all. This display cannot show charge outside this cluster's own cells; say so, and I will build a "
    "wire-level (decon) view of this item.</li>"
    "<li><code>no-michel</code> &mdash; there is no Michel at this stop.</li>"
    "<li><code>unclear</code>.</li></ul>")

INSTR = (
    "<b>Please judge</b> from the 3-D view and the measurement tab:<br>"
    "<b>1. Where is the Michel?</b> Type the first line of <b>notes</b> as "
    "<code>MECH: &lt;word&gt;; DEAD: &lt;U/V/W/seam or none&gt;;</code> then anything else, <i>before</i> "
    "clicking the verdict. MECH is one of:" + VOCAB +
    "DEAD names the planes where a dead band crosses the Michel (or the muon's last few cm), "
    "<code>seam</code> if a CRP/CRU seam does, or none.<br>"
    "<b>2. The pin</b>: put it at the muon's true end, where the Bragg rise ends and the Michel begins; "
    "leave it on the fit end if that is right.<br>"
    "<b>3. The verdict</b> (the click saves the row): <b>STM + MICHEL</b> with the Michel radio set "
    "(<i>attached</i> / <i>detached dots</i> / <i>both</i>), or <b>STM, no Michel</b>; FRAG / MESSY / "
    "UNCLEAR as usual. After moving the pin or editing the notes, press <b>SAVE this item</b>. "
    "If a PR segment in the particle-flow table <i>is</i> the Michel, tag it michel there.")

HEAD = ("<b>smx6 &mdash; is there a Michel at this stopper's end, and where?</b> (doc pdvd/88 = doc 78 "
        "action item 7)<br>"
        "Production calls this muon a <b>stopper</b> (is_stm 1) and finds <b>no Michel object</b> "
        "(michel_found 0). The stopper call is not what is asked here &mdash; the Michel is. "
        "Answer <code>no-michel</code> if there is none.<br>")


def context(k, r):
    pin = r.get("pin") or {}
    if k == "039253_15/36":
        return "You have not judged this item before.<br>", None
    s = "In <b>smx3</b> you called it <b>%s</b> (Michel: %s)" % (
        r["verdict"].replace("_", " + ").replace("STM + MICHEL", "STM + MICHEL"), r.get("michel_kind"))
    if r.get("evidence"):
        s += ", noting &ldquo;%s&rdquo;" % r["evidence"].replace("<", "&lt;")
    if pin.get("placed"):
        s += ", and moved the pin %.1f cm back from the fit end (rr %.1f &mdash; the <b style='color:#2ca02c'>" \
             "green dot-dash line</b> in the dQ/dx panel)" % (pin["moved_cm"], pin["rr"])
        anchor = float(pin["rr"])
    else:
        s += ", and left the pin on the fit end"
        anchor = None
    tags = r.get("tags") or {}
    gam = sorted(t for t, v in tags.items() if v == "gamma")
    if gam:
        s += "; you tagged %s as gamma" % ", ".join(gam)
    return s + ".<br>", anchor


items = {}
for k, mech, dead, why in ITEMS:
    ev, c = k.split("/")
    pf = os.path.join(a.prep, "smprep-%s-c%s.json" % (ev, c))
    p = json.load(open(pf))
    v = p["verdict"]
    assert int(v["is_stm"]) == 1 and int(v["michel_found"]) == 0, \
        "%s: production does not read is_stm 1 / michel_found 0 -- the panel would be wrong" % k
    r = rec[k]
    ctx, anchor = context(k, r)
    items[k] = dict(pf=pf, pay=p, rec=r, mech=mech, dead=dead, why=why,
                    q=dict(html=HEAD + ctx + INSTR, anchor_rr_cm=anchor))

order = [k for k, *_ in ITEMS]
with open(a.questions, "w") as fh:
    json.dump({"scan": "pdvd smx6, the stop-local keep's record-FP (doc pdvd/88 = doc 78 item 7)",
               "prep": a.prep, "items": {k: items[k]["q"] for k in order}}, fh, indent=1)
if a.force_questions:
    print("rewrote %s only (%d items)" % (a.questions, len(order)))
    sys.exit(0)

os.makedirs(a.outprep, exist_ok=True)
shutil.copy2(os.path.join(a.prep, "dqdx_ref_pdvd.json"), a.outprep)
rows, krows = [], []
for sid, k in enumerate(order, 1):
    it = items[k]
    shutil.copy2(it["pf"], a.outprep)
    ev, c = k.split("/")
    rows.append("%d\t1\t%s\t%s\t%d\t%.1f" % (sid, ev, c, int(it["pay"]["npts"]), float(it["pay"]["muon_len_cm"])))
    r = it["rec"]
    krows.append("%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
        sid, k, r["verdict"], r.get("michel_kind"), r.get("source", r.get("confidence")),
        it["mech"], it["dead"], it["why"]))
with open(a.sheet, "w") as fh:
    fh.write("# doc pdvd/88 -- smx6, PDVD, doc 78 action item 7. Production (p85vwh == p85vprod) reads "
             "is_stm 1 / michel_found 0; the payload is production's (blind to the flip candidate).\n")
    fh.write("# built by d88_build_smx6.py from prep=%s\n" % a.prep)
    fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
    fh.write("\n".join(rows) + "\n")
with open(a.key, "w") as fh:
    fh.write("# doc pdvd/88 -- smx6 KEY (do not show the scanner): the record's reading and the answer "
             "the p88v5fr arm's reading predicts, written before the scan.\n")
    fh.write("scan_id\tkey\trecord_verdict\trecord_michel_kind\trecord_source\tpred_mech\tpred_dead\toffline_reading\n")
    fh.write("\n".join(krows) + "\n")
print("items %d: %s" % (len(order), ", ".join("%s pred %s/%s" % (k, items[k]["mech"], items[k]["dead"]) for k in order)))

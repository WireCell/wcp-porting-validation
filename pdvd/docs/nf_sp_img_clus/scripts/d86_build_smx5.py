#!/usr/bin/env python3
"""doc pdvd/86 -- build the `smx5` hand check of doc 78 action item 6.

Four items production calls stoppers (is_stm 1) with no Michel object
(michel_found 0).  The owner is asked WHERE the Michel is, in a controlled
vocabulary typed as the first line of the notes (MECH / DEAD), plus the pin and
the verdict (the verdict click is what saves a row).

  * production must read is_stm 1 / michel_found 0 on every item -- asserted,
    because the panel states it;
  * three items carry the owner's own smx3 verdict and pin, and the panel repeats
    them (a diagnostic re-look, not a blind re-judge); `039349_30/45` was judged
    only by the smx1a scanner, at medium confidence, on the fit's collapse plus an
    unfitted fan -- nothing of that reading goes on its panel;
  * the offline reading and the predicted answer (d86_offfit.py, written before the
    scan) live only in the --key file.

Usage (STM_SCAN_RECORD must name the merged smx1a+smx3+smx4 record):
  d86_build_smx5.py --prep PREP --outprep DIR --sheet TSV --questions JSON --key TSV
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

if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a+smx3+smx4 record (doc 70)")
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

# doc 78 sec 5 item 6, in its order.  pred = the answer d86_offfit.py's reading
# predicts (doc 86 sec 3), written before the scan; never shown to the scanner.
ITEMS = [
    ("039349_43/66", "fit-through", "V?",
     "on-fit: 0 of 55 own points within 10 cm of the fit end > 2 cm off the fit; the last 8 rows from "
     "the owner's pin read 98 65 91 63 19 33 54 67 k (plateau 54k): a dip to 0.36 plateau 2 cm "
     "before the end, then a second deposit the fit ends on. V dead 4978-4979, 1 channel off the end"),
    ("039349_72/11", "fit-through", "none",
     "on-fit: 0 of 73 own points > 2 cm off the fit; from the owner's pin 77 69 40 41 38 31 k (plateau "
     "60k): a drop to 0.5-0.7 plateau over the last 2 cm with no recovery. Weakest of the four: the "
     "Michel may also be unimaged. Dead channels 7 (U) and 11 (W) away"),
    ("039253_0/44", "unfitted", "V",
     "off-fit: 82 own points > 2 cm off the fit within 10 cm of the end (1.09 M e), only 2 past the end; "
     "one PR segment in the cluster; rise to 131k (1.8 plateau) at the end. V dead 4360-4367, 1.8 "
     "channels off the end -- the off-axis points may be two-plane imaging there"),
    ("039349_30/45", "unfitted", "none",
     "off-fit: 50 own points > 2 cm off the fit, 19 past the end (15 of them > 2 cm off); last 5 cm "
     "85 68 35 9 7 21 39 k (plateau 49k): collapse, then a second rise the fit ends on. No dead "
     "channel within 15. Verdict open (smx1a medium)"),
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

HEAD = ("<b>smx5 &mdash; where is this stopper's Michel?</b> (doc pdvd/86 = doc 78 action item 6)<br>"
        "Production calls this muon a <b>stopper</b> (is_stm 1) and finds <b>no Michel object</b> "
        "(michel_found 0); no other PR segment of the cluster lies within 20 cm of the fit end. "
        "The stopper call is not what is asked here &mdash; the Michel is.<br>")


def context(k, r):
    pin = r.get("pin") or {}
    if k == "039349_30/45":
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
    json.dump({"scan": "pdvd smx5, where is the stopper's Michel (doc pdvd/86 = doc 78 item 6)",
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
    fh.write("# doc pdvd/86 -- smx5, PDVD, doc 78 action item 6. Production (p85vwh == p85vprod) reads "
             "is_stm 1 / michel_found 0 on every item; doc 78's order.\n")
    fh.write("# built by d86_build_smx5.py from prep=%s\n" % a.prep)
    fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
    fh.write("\n".join(rows) + "\n")
with open(a.key, "w") as fh:
    fh.write("# doc pdvd/86 -- smx5 KEY (do not show the scanner): the record's reading and the answer "
             "d86_offfit.py's reading predicts, written before the scan.\n")
    fh.write("scan_id\tkey\trecord_verdict\trecord_michel_kind\trecord_source\tpred_mech\tpred_dead\toffline_reading\n")
    fh.write("\n".join(krows) + "\n")
print("items %d: %s" % (len(order), ", ".join("%s pred %s/%s" % (k, items[k]["mech"], items[k]["dead"]) for k in order)))

#!/usr/bin/env python3
"""doc pdvd/68 -- build the hand-scan set that grades two OPTIONS the chain
does not run in production, against today's production (doc pdvd/67).

Groups, in the order a partial scan is still useful:
  1 (A)  anchor-contested: record items where `bragg_peak_anchor` 3 cm, run on
         today's production bag, flips `is_stm` AND disagrees with the smx1a
         record (a new FP or a lost TP).  Payload = production (the anchor is
         verdict-only, so the geometry is identical -- asserted); the panel says
         what the option reads and the dQ/dx panel draws where it puts the peak.
  2 (B)  dx-new: candidates that exist only with `dx_norm_length` 4 mm (keys the
         record never saw).  A candidate whose muon points sit on a record item
         that the option LOST (same event, median nearest distance < 1.5 cm over
         >= 50 % of its points) is the same track under a new cluster id: its
         verdict is already in the record, so it is reported, not scanned.
         Payload = the option's own reconstruction.
  3 (C)  dx-contested: record items where dx 4 mm flips `is_stm` against
         production AND disagrees with the record.  Payload = the option's.
  4 (D)  the doc 65/67 list of 8 that no longer flip against today's
         production, kept because the owner asked for them.  Payload = production.

The question text carries what PRODUCTION and the OPTION say -- never the smx1a
verdict, which is what this scan checks (feedback_blind_the_scan_sheet).

Usage: d68_build_scan_set.py --prod P --anchor P --dx P --outprep DIR --sheet TSV --questions JSON
"""
import argparse, glob, json, os, shutil, sys
import numpy as np
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

OLD8 = ["039252_2/79", "039349_14/22", "039349_24/23", "039349_43/66", "039349_72/11",
        "039253_3/66", "039253_6/85", "039349_76/75"]

ap = argparse.ArgumentParser()
for k in ("prod", "anchor", "dx", "outprep", "sheet", "questions"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
if os.path.exists(a.outprep) and os.listdir(a.outprep):
    sys.exit("REFUSING: %s exists and is not empty (a scan set is never rebuilt in place)" % a.outprep)

rec = C.load_record()


def load(prep):
    out = {}
    for f in glob.glob(os.path.join(prep, "smprep-*.json")):
        ev, c = os.path.basename(f)[7:-5].rsplit("-c", 1)
        out["%s/%s" % (ev, c)] = f
    return out


def pay(f):
    with open(f) as fh:
        return json.load(fh)


FP, FA, FD = load(a.prod), load(a.anchor), load(a.dx)
V = {}
def verdict(src, k):
    kk = (src, k)
    if kk not in V:
        V[kk] = pay({"prod": FP, "anchor": FA, "dx": FD}[src][k])["verdict"]
    return V[kk]


def reads(v):
    rn = [r for r in (v.get("reject_names") or []) if r != "STM"]
    s = "<b>is_stm %d</b>" % int(v["is_stm"]) + ("" if not rn else " (rejected by %s)" % ", ".join(rn))
    return s + ", michel_found %d" % int(v["michel_found"])


judged = {k for k in rec if C.judged(rec[k])}
stop = {k: C.is_stopper(rec[k]) for k in rec}
items = {}            # key -> dict(group, src, html, anchor_rr_cm)

# ---- group 1: anchor-contested -------------------------------------------
for k in sorted(set(FP) & set(FA) & judged):
    p, o = verdict("prod", k), verdict("anchor", k)
    if int(p["is_stm"]) != int(o["is_stm"]) and bool(o["is_stm"]) != stop[k]:
        items[k] = dict(group=1, src="prod")

# ---- group 3: dx-contested ------------------------------------------------
for k in sorted(set(FP) & set(FD) & judged):
    p, o = verdict("prod", k), verdict("dx", k)
    if int(p["is_stm"]) != int(o["is_stm"]) and bool(o["is_stm"]) != stop[k] and k not in items:
        items[k] = dict(group=3, src="dx")

# ---- group 2: dx-new, minus the same track under a new id ----------------
def muon_xyz(f):
    m = pay(f)["muon"]
    return np.column_stack([m["x"], m["y"], m["z"]])

lost = sorted(k for k in rec if k not in FD)
paired = {}
for k in sorted(k for k in FD if k not in rec):
    ev = k.split("/")[0]
    P0 = muon_xyz(FD[k])
    for l in lost:
        if l.split("/")[0] != ev or l not in FP:
            continue
        Q0 = muon_xyz(FP[l])
        d = np.array([np.linalg.norm(Q0 - p, axis=1).min() for p in P0])
        if (d < 1.5).mean() >= 0.5:
            paired[k] = (l, float(np.median(d)), float((d < 1.5).mean()))
            break
    if k not in paired:
        items[k] = dict(group=2, src="dx")

# ---- group 4: the old list, where it no longer flips ---------------------
for k in OLD8:
    if k not in items:
        items[k] = dict(group=4, src="prod")

# ---- question text ---------------------------------------------------------
ANCHOR = ("<b>Option under test: the Bragg-peak anchor</b> (<code>bragg_peak_anchor</code>, "
          "3 cm search). It re-reads the Bragg-peak tests from the dQ/dx maximum it finds near "
          "the end of the fit instead of from the fit's last point.")
DX = ("<b>Option under test: <code>dx_norm_length</code> 4 mm</b> (the track fit's charge-"
      "smoothing length; production 6 mm). It changes the fit, so the display below is the "
      "reconstruction <b>with the option</b>.")
ASK = ("<br><b>Your verdict decides which reading is right</b>: does the muon stop inside, is "
       "there a Michel, and where does it stop (move the pin if the fit end is wrong)?")
for k, it in items.items():
    g = it["group"]
    ar = None
    if g in (1, 4):
        p, o = verdict("prod", k), verdict("anchor", k)
        # the anchor is verdict-only: the chain geometry must be the production one
        rp, ro = pay(FP[k])["muon"]["rr"], pay(FA[k])["muon"]["rr"]
        assert rp == ro, "anchor arm moved the geometry of %s" % k
        ar = float(o.get("bragg_anchor_shift_cm") or 0.0)
        mk = ("<br>The <span style='color:#2ca02c'><b>green dot-dash line</b></span> on the "
              "dQ/dx panel is where the option puts the Bragg peak: <b>%.1f cm</b> before the "
              "fit end." % ar)
        if g == 1:
            html = "%s<br>Production reads %s. The option reads %s.%s%s" % (ANCHOR, reads(p), reads(o), mk, ASK)
        else:
            html = ("%s<br><i>On today's production the option no longer changes this item</i> "
                    "(both read %s); it was contested against the earlier production (doc 65 "
                    "&sect;5.1) and stays on the list at your request.%s%s" % (ANCHOR, reads(o), mk, ASK))
    elif g == 2 and k in FP:
        # a candidate today's production ALSO flags: it postdates the 569-item
        # draw (the candidate set moved since), so it is unscanned for both
        p, o = verdict("prod", k), verdict("dx", k)
        html = ("%s<br>This cluster is a stopping-muon candidate <b>both</b> with the option and "
                "in today's production, but it is not in the 569-item scan (the candidate set "
                "moved after that sample was drawn), so it has never been scanned. Production "
                "reads %s. The option reads %s.%s" % (DX, reads(p), reads(o), ASK))
    elif g == 2:
        o = verdict("dx", k)
        html = ("%s<br>With the option the STM tagger flags this cluster as a stopping-muon "
                "candidate; <b>production does not</b>, so it has never been scanned. The option "
                "reads %s.%s" % (DX, reads(o), ASK))
    else:
        p, o = verdict("prod", k), verdict("dx", k)
        html = "%s<br>Production reads %s. The option reads %s.%s" % (DX, reads(p), reads(o), ASK)
    it["html"] = "<b>[group %s]</b> " % {1: "1: anchor", 2: "2: dx 4 mm, new candidate",
                                         3: "3: dx 4 mm", 4: "4: anchor, earlier list"}[g] + html
    it["anchor_rr_cm"] = ar

# ---- write ------------------------------------------------------------------
os.makedirs(a.outprep, exist_ok=True)
shutil.copy2(os.path.join(a.prod, "dqdx_ref_pdvd.json"), a.outprep)
rows = []
order = sorted(items, key=lambda k: (items[k]["group"], k))
for sid, k in enumerate(order, 1):
    it = items[k]
    f = {"prod": FP, "dx": FD}[it["src"]][k]
    shutil.copy2(f, a.outprep)
    p = pay(f)
    ev, c = k.split("/")
    rows.append("%d\t%d\t%s\t%s\t%d\t%.1f" % (sid, it["group"], ev, c, int(p["npts"]), float(p["muon_len_cm"])))
with open(a.sheet, "w") as fh:
    fh.write("# doc pdvd/68 -- option scan, PDVD. tranche = group: 1 anchor-contested, 2 dx-4mm new "
             "candidates, 3 dx-4mm contested, 4 the earlier anchor list.\n")
    fh.write("# built by d68_build_scan_set.py from prod=%s anchor=%s dx=%s\n" % (a.prod, a.anchor, a.dx))
    fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
    fh.write("\n".join(rows) + "\n")
with open(a.questions, "w") as fh:
    json.dump({"scan": "pdvd option scan (doc pdvd/68)", "prod": a.prod, "anchor": a.anchor, "dx": a.dx,
               "items": {k: dict(group=items[k]["group"], src=items[k]["src"], html=items[k]["html"],
                                 anchor_rr_cm=items[k]["anchor_rr_cm"]) for k in order}}, fh, indent=1)

n = {g: sum(1 for k in items if items[k]["group"] == g) for g in (1, 2, 3, 4)}
print("items %d: group 1 anchor-contested %d, group 2 dx-new %d, group 3 dx-contested %d, group 4 earlier list %d"
      % (len(items), n[1], n[2], n[3], n[4]))
print("dx candidates not in the record: %d; paired to a record item the option lost (not scanned): %d"
      % (sum(1 for k in FD if k not in rec), len(paired)))
for k, (l, md, fr) in sorted(paired.items()):
    print("   %s  == record %s  (median %.2f cm, %.0f %% of points within 1.5 cm)" % (k, l, md, 100 * fr))
print("record items the option lost: %d; of them paired: %d" % (len(lost), len({v[0] for v in paired.values()})))
for g in (1, 3, 4):
    print("group %d: %s" % (g, " ".join(k for k in order if items[k]["group"] == g)))

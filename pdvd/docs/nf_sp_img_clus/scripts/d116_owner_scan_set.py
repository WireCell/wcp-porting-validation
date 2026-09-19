#!/usr/bin/env python3
"""doc pdvd/116 sec 6.3 -- the owner's look at the false positives of the carried operating point R2 on the NEW
trajectory.  Fork by duplication of d103_owner_scan_set.py (untouched).

    python3 d116_owner_scan_set.py --det pdvd --a0 d115voff --r2 d116vr2 --prep /home/xqian/tmp/d116/own/prep_d116vr2 \
        --out /home/xqian/tmp/d116/own/set_pdvd

Tier 1: every item that is a false positive in R2 and not in A0 on is_stm or michel_found under the doc-116 truth
(d116_grade.truth with the smx116 blind record folded at the lowest precedence) -- exactly the `fp_new` class of
figs/116_movers_r_<det>_R2.tsv, owner-labelled items included (those labels were given on the OLD trajectory).
Controls: N_CONTROL judged, non-owner-labelled items tagged identically in A0 and R2 with is_stm 1, drawn with
random.Random(116).  Every item is shown on R2's payload (the new trajectory + the R2 tagger), shuffled, behind one
question panel that is the same for every item (no chain answer, no prior label, no role).  The roles live only in
<out>/items.tsv.

Writes <out>/{prep/, manifest.tsv, questions.json, items.tsv}; refuses an existing <out> or an existing label dir.
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d116_grade as G16                                       # noqa: E402  (the doc-116 truth and grade)
import d103_union_grade as U                                   # noqa: E402

TAG = {"pdhd": "own116h", "pdvd": "own116v"}
EXTRA = {"pdhd": f"{U.IMG}/pdhd/docs/scan/pdhd_stm_michel_smx116_verdicts.json",
         "pdvd": f"{U.IMG}/pdvd/docs/scan/pdvd_stm_michel_smx116_verdicts.json"}
N_CONTROL = {"pdhd": 4, "pdvd": 8}
SEED = 116
OWNER_SRC = ("owner_review", "owner", "own103h", "own103h2", "own103v", "own103v2")

QUESTION = (
    "<b>own116v &mdash; owner look at the new trajectory</b> (doc pdvd/116 sec 6.3). The same panel is shown for every "
    "item: it names no chain answer and no earlier label. Every item is drawn on the <b>new</b> trajectory "
    "(prefer3 + tree+path &alpha; 0.5) with the R2 tagger.<br>"
    "<b>Please judge on this display.</b><ul style='margin:2px 0 2px 0'>"
    "<li>Stopper: STM + MICHEL (Michel radio) if an electron leaves the stop, STM if it stops with no Michel, THRU if it "
    "does not stop. UNCLEAR / MESSY as usual.</li>"
    "<li>The Michel kind matters here: <b>attached</b>, <b>detached dots</b>, or none.</li></ul>"
    "<b>Three rubric gaps the scanners asked about.</b> A short note is enough.<ol style='margin:2px 0 2px 0'>"
    "<li>A detached piece <b>5&ndash;10 cm</b> from the stop: is it a Michel, or a gamma? Please give its distance.</li>"
    "<li>An unfitted C-row cluster by the stop with no dQ/dx: can it be a capture gamma?</li>"
    "<li>A track that ends at the <b>readout-window edge</b>: is it a stop or THRU?</li></ol>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--a0", required=True, help="production arm (A0)")
    ap.add_argument("--r2", required=True, help="the carried operating point on the new trajectory (R2)")
    ap.add_argument("--prep", required=True, help="prep dir built on the R2 arm")
    ap.add_argument("--fallback-prep", default=None, help="prep dir of the A0 arm: used only for a tier-1 item the R2 prep "
                    "cannot draw (below the prep's filters on the new trajectory); recorded in items.tsv shown_on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-movers", default=None, help="figs/116_movers_r_<det>_R2.tsv: tier 1 must equal its fp_new keys")
    a = ap.parse_args()
    labels = f"{U.IMG}/{a.det}/work/stm_michel_labels/{TAG[a.det]}"
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists (a scan set is never rebuilt in place)")
    if os.path.exists(labels):
        sys.exit(f"REFUSING: {labels} exists (M13: new scan => new tag)")

    T, src = G16.truth(a.det, EXTRA[a.det])
    R = {"A0": U.cell_rows(a.det, a.a0), "R2": U.cell_rows(a.det, a.r2)}
    G, _ = G16.D.grade(a.det, T, R)
    owner = lambda k: T[k][2] in OWNER_SRC
    fp_new = {}
    for metric, pop, TR, idx in (("is_stm", G["pop"], G["stm_truth"], 0), ("michel", G["mpop"], G["m_truth"], 1)):
        for k in pop:
            if not TR[k] and R["R2"].get(k, (0, 0))[idx] and not R["A0"].get(k, (0, 0))[idx]:
                fp_new.setdefault(k, []).append(f"R2:{metric}")
    tier1 = sorted(fp_new)
    if a.check_movers:
        want = sorted({l.split("\t")[2] for l in open(a.check_movers) if "\tfp_new\t" in l})
        if want != tier1:
            sys.exit(f"tier 1 {tier1} != fp_new keys of {a.check_movers} {want}")

    def payload(k, fallback=False):
        ev, cid = k.split("/")
        for d in (a.prep,) + ((a.fallback_prep,) if fallback and a.fallback_prep else ()):
            p = f"{d}/smprep-{ev}-c{cid}.json"
            if os.path.exists(p):
                return p
        return None

    miss = [k for k in tier1 if not payload(k, True)]
    if miss:
        sys.exit(f"tier-1 items with no payload on {a.prep} (or the fallback): {miss}")
    fell = [k for k in tier1 if not payload(k)]
    pool = [k for k in sorted(set(R["A0"]) & set(R["R2"])) if k in G["judged"] and not owner(k) and k not in fp_new
            and R["A0"][k][:2] == R["R2"][k][:2] and R["R2"][k][0] and payload(k)]
    controls = sorted(random.Random(SEED).sample(pool, N_CONTROL[a.det]))
    items = [(k, "tier1") for k in tier1] + [(k, "control") for k in controls]
    random.Random(SEED).shuffle(items)

    os.makedirs(a.out + "/prep")
    os.symlink(f"{a.prep}/dqdx_ref_{a.det}.json", f"{a.out}/prep/dqdx_ref_{a.det}.json")
    ans = lambda lab, k: (f"is_stm {int(R[lab][k][0])} michel_found {int(R[lab][k][1])}" if k in R[lab] else "not a candidate")
    man, q, rows = [], {}, []
    for i, (k, role) in enumerate(items, 1):
        ev, cid = k.split("/")
        s = payload(k, True)
        os.symlink(s, f"{a.out}/prep/smprep-{ev}-c{cid}.json")
        pay = json.load(open(s))
        man.append(f"{i}\t1\t{ev}\t{cid}\t{pay['npts']}\t{pay['muon_len_cm']:.1f}")
        q[k] = dict(html=QUESTION.replace("own116v", TAG[a.det]))
        rows.append("\t".join(map(str, [i, k, role, ",".join(fp_new.get(k, [])) or "-", ans("A0", k), ans("R2", k),
                                        T[k][0], T[k][1], T[k][2], "owner" if owner(k) else "",
                                        a.r2 if k not in fell else f"{a.a0} (FALLBACK: below the prep filters on {a.r2})"])))

    with open(a.out + "/manifest.tsv", "w") as fh:
        fh.write(f"# doc pdvd/116 -- {TAG[a.det]}, owner look at R2's false positives, shown on {a.r2} (roles only in items.tsv)\n")
        fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n" + "\n".join(man) + "\n")
    with open(a.out + "/questions.json", "w") as fh:
        json.dump(dict(scan=f"{a.det} {TAG[a.det]}: owner look at the new trajectory (doc pdvd/116 sec 6.3)",
                       prep=a.out + "/prep", items=q), fh, indent=1)
    with open(a.out + "/items.tsv", "w") as fh:
        fh.write("scan_id\tkey\trole\tfp_in\tA0\tR2\tlabel_verdict\tlabel_michel_kind\tlabel_source\tlabel_confidence\tshown_on\n")
        fh.write("\n".join(rows) + "\n")
    print(f"{a.det} {TAG[a.det]}: truth sources {src}; tier 1 {len(tier1)} (is_stm {sum('R2:is_stm' in v for v in fp_new.values())}, "
          f"michel {sum('R2:michel' in v for v in fp_new.values())}; owner-labelled on the old trajectory {sum(owner(k) for k in tier1)}), "
          f"controls {len(controls)} (pool {len(pool)}), shown on {a.r2}; on the fallback payload {fell}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()

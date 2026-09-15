#!/usr/bin/env python3
"""doc pdvd/103 -- the owner's adjudication set own103h (figs/103_pred_amend2.txt sec 2).  Fork of d100_michel_scan_set.py.

    python3 d103_owner_scan_set.py --det pdhd --new-record F --out /home/xqian/tmp/d103/own/set_pdhd

Tier 1: every judged, non-owner-labelled item that is a false positive in exactly one of A0 (production) / A1 (both
levers on), on is_stm or michel_found.  Controls: 8 judged, non-owner-labelled items tagged identically in A0 and A1 with
is_stm 1, drawn with random.Random(103).  Shown on A1's payload, shuffled, behind a question panel that is the same for
every item (no chain answer, no prior label, no role).  The roles live only in <out>/items.tsv.

Writes <out>/{prep/, manifest.tsv, questions.json, items.tsv}; refuses an existing <out> or an existing label dir.
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d103_union_grade as U

PREP = "/home/xqian/tmp/d103/round/prep_{arm}"
TAG = {"pdhd": "own103h"}
BASE = {"pdhd": U.PDHD_RECORD}          # figs/103_pred_amend3.txt: run with D103_PDHD_RECORD=<smx27>
N_CONTROL = 8

QUESTION = (
    "<b>own103h &mdash; owner adjudication</b> (doc pdvd/103). The same panel is shown for every item: it names no chain "
    "answer and no earlier label.<br>"
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
    ap.add_argument("--det", required=True, choices=["pdhd"])
    ap.add_argument("--new-record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels = f"{U.IMG}/{a.det}/work/stm_michel_labels/{TAG[a.det]}"
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists (a scan set is never rebuilt in place)")
    if os.path.exists(labels):
        sys.exit(f"REFUSING: {labels} exists (M13: new scan => new tag)")

    T, _ = U.load_truth(a.det, a.new_record)
    R = {lab: U.cell_rows(a.det, arm) for lab, arm in U.CELLS[a.det]}
    arm = dict(U.CELLS[a.det])
    conf = {r["key"]: r.get("confidence", "") for r in json.load(open(BASE[a.det]))}
    for r in json.load(open(a.new_record)):
        conf.setdefault(r["key"], r.get("confidence", ""))
    owner = lambda k: T[k][2] in ("owner_review", "owner") or conf.get(k) == "owner"
    judged = lambda k: k in T and T[k][0] not in ("MESSY", "UNCLEAR")
    stopper = lambda k: T[k][0] in ("STM_MICHEL", "STM_ONLY")

    def fp(k, lab, i):                                          # exactly d103_fp_classes.py's definition
        if not judged(k) or not R[lab].get(k, (0, 0))[i]:
            return False
        if i == 0:
            return not stopper(k)
        return stopper(k) and T[k][1] not in ("attached", "both") and not (T[k][1] is None and T[k][2] == "owner_review")

    def payload(k):
        ev, cid = k.split("/")
        p = f"{PREP.format(arm=arm['A1'])}/smprep-{ev}-c{cid}.json"
        return p if os.path.exists(p) else None

    keys = sorted(set(R["A0"]) | set(R["A1"]))
    tier1 = [k for k in keys if judged(k) and not owner(k) and any(fp(k, "A0", i) != fp(k, "A1", i) for i in (0, 1))]
    miss = [k for k in tier1 if not payload(k)]
    if miss:
        sys.exit(f"tier-1 items with no A1 payload: {miss}")
    pool = [k for k in keys if judged(k) and not owner(k) and k not in tier1 and k in R["A0"] and k in R["A1"]
            and R["A0"][k][:2] == R["A1"][k][:2] and R["A1"][k][0] and payload(k)]
    controls = sorted(random.Random(103).sample(pool, N_CONTROL))
    items = [(k, "tier1") for k in tier1] + [(k, "control") for k in controls]
    random.Random(103).shuffle(items)

    os.makedirs(a.out + "/prep")
    ref = f"{PREP.format(arm=arm['A1'])}/dqdx_ref_{a.det}.json"
    os.symlink(ref, f"{a.out}/prep/dqdx_ref_{a.det}.json")
    ans = lambda lab, k: (f"is_stm {int(R[lab][k][0])} michel_found {int(R[lab][k][1])}" if k in R[lab] else "not a candidate")
    man, q, rows = [], {}, []
    for i, (k, role) in enumerate(items, 1):
        ev, cid = k.split("/")
        src = payload(k)
        os.symlink(src, f"{a.out}/prep/smprep-{ev}-c{cid}.json")
        pay = json.load(open(src))
        man.append(f"{i}\t1\t{ev}\t{cid}\t{pay['npts']}\t{pay['muon_len_cm']:.1f}")
        q[k] = dict(html=QUESTION)
        fps = ",".join(f"{lab}:{('is_stm', 'michel')[j]}" for lab in ("A0", "A1") for j in (0, 1) if fp(k, lab, j))
        rows.append("\t".join(map(str, [i, k, role, fps or "-", ans("A0", k), ans("A1", k), T[k][0], T[k][1], T[k][2],
                                        conf.get(k, "")])))

    with open(a.out + "/manifest.tsv", "w") as fh:
        fh.write(f"# doc pdvd/103 -- {TAG[a.det]}, owner adjudication, shown on {arm['A1']} (roles only in items.tsv)\n")
        fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n" + "\n".join(man) + "\n")
    with open(a.out + "/questions.json", "w") as fh:
        json.dump(dict(scan=f"{a.det} {TAG[a.det]}: owner adjudication (doc pdvd/103, figs/103_pred_amend2.txt)",
                       prep=a.out + "/prep", items=q), fh, indent=1)
    with open(a.out + "/items.tsv", "w") as fh:
        fh.write("scan_id\tkey\trole\tfp_in\tA0\tA1\tlabel_verdict\tlabel_michel_kind\tlabel_source\tlabel_confidence\n")
        fh.write("\n".join(rows) + "\n")
    print(f"{a.det} {TAG[a.det]}: tier 1 {len(tier1)}, controls {len(controls)} (pool {len(pool)}), shown on {arm['A1']}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()

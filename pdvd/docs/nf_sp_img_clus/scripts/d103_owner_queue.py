#!/usr/bin/env python3
"""doc pdvd/103 sec 6 -- the owner review queue for one detector (figs/103_pred.txt RESTATEMENT).  Read-only.

Sections, each a table of keys with the facts the owner needs to judge it (no chain reasoning beyond the tags):
  A  false positives of the both-on cell A1 on the union record (is_stm, michel_found), with label source and
     confidence -- the purity cost; which of them are also FPs in production A0
  B  new blind labels on items tagged is_stm or michel_found in any cell (the purity-deciding new labels)
  C  new blind labels with medium or low confidence (not already in B)
  D  calibration items where the blind label disagrees with the existing record
  E  new blind labels whose notes name the readout-window edge / DEAD at the window (a class the rubric does not name)
Truth / population / chain exactly as d103_union_grade.py (imported).

Usage: d103_owner_queue.py --det pdhd --new-record F --items /home/xqian/tmp/d103/items/pdhd_items.tsv > figs/103_owner_queue_pdhd.md
"""
import argparse, csv, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d103_union_grade as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--new-record", required=True)
    ap.add_argument("--items", required=True)
    a = ap.parse_args()
    T, _ = U.load_truth(a.det, a.new_record)
    R = {lab: U.cell_rows(a.det, arm) for lab, arm in U.CELLS[a.det]}
    new = {r["key"]: r for r in json.load(open(a.new_record))}
    items = {r["key"]: r for r in csv.DictReader((l for l in open(a.items) if not l.startswith("#")), delimiter="\t")}
    old = U.load_truth(a.det, None)[0]
    conf_old = {}
    if a.det == "pdhd":
        for r in json.load(open(f"{U.IMG}/pdhd/docs/scan/pdhd_stm_michel_smx22_verdicts.json")):
            conf_old[r["key"]] = r.get("confidence", "")
    else:
        for r in json.load(open(f"{U.IMG}/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json")):
            conf_old[r["key"]] = r.get("confidence", "")

    def conf(k):
        return new[k]["confidence"] if T[k][2] == "new_agent" else conf_old.get(k, "")

    def cells(k, idx):
        return "".join(lab if R[lab].get(k, (0, 0))[idx] else "." for lab, _ in U.CELLS[a.det])

    judged = lambda k: k in T and T[k][0] not in ("MESSY", "UNCLEAR")
    stopper = lambda k: T[k][0] in ("STM_MICHEL", "STM_ONLY")
    print(f"# doc pdvd/103 owner review queue ({a.det})")
    print("Cells column: letters of the cells that tag the item (A0 production, K fit knobs, S sampler, A1 both on); "
          "'.' = not tagged.  Label source: new_agent = this round's verdict-blind scan; smx22/record/owner* = existing.\n")

    print("## A. False positives of A1 (both on) on the union record\n")
    for title, idx, truth in (("is_stm", 0, lambda k: stopper(k)),
                              ("michel_found", 1, lambda k: T[k][0] in ("STM_MICHEL",) if a.det == "pdvd"
                               else (stopper(k) and T[k][1] in ("attached", "both")))):
        rows = []
        for k in sorted(R["A1"]):
            if not judged(k) or not R["A1"][k][idx]:
                continue
            if a.det == "pdhd" and title == "michel_found" and not stopper(k):
                continue          # PDHD Michel population = hand stoppers
            if truth(k):
                continue
            rows.append(k)
        print(f"### {title}: {len(rows)} (also an FP in A0: {sum(1 for k in rows if R['A0'].get(k, (0, 0))[idx])})\n")
        print("| key | hand verdict | michel_kind | label source | confidence | tagged in | notes (new labels) |")
        print("|---|---|---|---|---|---|---|")
        for k in rows:
            print(f"| {k} | {T[k][0]} | {T[k][1]} | {T[k][2]} | {conf(k)} | {cells(k, idx)} | "
                  f"{(new[k].get('notes', '') if k in new else '')[:80]} |")
        print()

    B = sorted(k for k in new if not new[k].get("calibration") and any(R[l].get(k, (0, 0))[0] or R[l].get(k, (0, 0))[1] for l in R))
    print(f"## B. New blind labels on items tagged is_stm or michel_found in some cell: {len(B)}\n")
    print("| key | verdict | michel_kind | confidence | is_stm in | michel_found in | display arm | notes |")
    print("|---|---|---|---|---|---|---|---|")
    for k in B:
        r = new[k]
        print(f"| {k} | {r['verdict']} | {r['michel_kind']} | {r['confidence']} | {cells(k, 0)} | {cells(k, 1)} | "
              f"{r.get('display_arm', '')} | {r.get('notes', '')[:80]} |")
    C = sorted(k for k in new if not new[k].get("calibration") and k not in B and new[k]["confidence"] in ("medium", "low"))
    print(f"\n## C. Other new blind labels at medium / low confidence: {len(C)}\n")
    print("| key | verdict | michel_kind | confidence | notes |")
    print("|---|---|---|---|---|")
    for k in C:
        r = new[k]
        print(f"| {k} | {r['verdict']} | {r['michel_kind']} | {r['confidence']} | {r.get('notes', '')[:90]} |")
    STOP = ("STM_MICHEL", "STM_ONLY")
    D = [k for k in new if new[k].get("calibration") and k in old
         and (U.strip(new[k]["verdict"]) in STOP) != (old[k][0] in STOP)]
    print(f"\n## D. Calibration disagreements (blind vs existing record): {len(D)}\n")
    for k in sorted(D):
        print(f"- {k}: existing {old[k][0]} ({old[k][2]}, {conf_old.get(k, '')}) vs blind {new[k]['verdict']} ({new[k]['confidence']})")
    edge = re.compile(r"(readout|recorded|time)[- ](time[- ])?(window|frame)|frame[- ](edge|end|start)|window[- ](edge|end|start)")
    E = sorted(k for k in new if edge.search((new[k].get("notes", "") + " " + new[k].get("evidence", "")).lower()))
    print(f"\n## E. New labels that name the readout-window edge: {len(E)}\n")
    for k in E:
        print(f"- {k}: {new[k]['verdict']} ({new[k]['confidence']}); is_stm in {cells(k, 0)}; notes: {new[k].get('notes', '')[:90]}")


if __name__ == "__main__":
    main()

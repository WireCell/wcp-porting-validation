#!/usr/bin/env python3
"""doc pdvd/100 round 2 -- the owner's look (own100m, not blind) at every object whose chain answer moved between production
(p99wflip) and the gain-flip candidate (p100c) on the latest carried record.

    python3 d100_michel_scan_set.py --out /home/xqian/tmp/p100/mscan

Round 1 (doc sec 4) graded both arms on the latest carried record: Michel purity 0.938 -> 0.824, is_stm 0.978 -> 0.925.  The
records were drawn on production's display, so a Michel the gain makes visible reads as a false positive.  This set is
every michel_found mover between the two arms, in BOTH directions (21 new against the record, 16 lost, 6 found, 3 dropped;
tranche 1), then every is_stm mover not already in it (tranche 2).  The movers are re-derived here exactly as d99_grade.py
lists them (same items, same record join by original key); d99_grade prints the first 12 names only.

Each object is shown on p100c (the configuration going to production) with its p100c payload (prep_stm_michel_scan.py,
doc sec 3 twin prep).  The question names both arms' chain answers and the record verdict.  The owner's verdicts are
folded by d100_michel_scan_score.py into a new record and both arms re-graded on the corrected record (d100/prereg_round2.md);
no purity is computed on this selected set.

Writes <out>/{prep/, manifest.tsv, questions.json, items.tsv}; refuses an existing <out> or an existing own100m label dir.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d99_grade as G

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
LABELS = IMG + "/pdvd/work/stm_michel_labels/own100m"
P100 = "/home/xqian/tmp/p100"
ARMS = {"p99wflip": P100 + "/carry/latest_on_p99wflip.json", "p100c": P100 + "/carry/latest_on_p99rwon.json"}
DESC = {"p99wflip": "production: SP gain OFF, v5 wire order, C 0.7941",
        "p100c": "candidate: SP top gain 0.889, v7 wire order, C 0.8630"}


def movers():
    A, R = {}, {}
    for arm, rec in ARMS.items():
        its, _, orig, _ = G.items_on(arm, rec)
        A[arm] = {orig[k]: (k, v, d) for k, v, kind, src, d in its}
        R[arm] = {r["key"]: r for r in json.load(open(rec))}
    out = []
    for o in sorted(set(A["p99wflip"]) & set(A["p100c"])):
        kw, v, dw = A["p99wflip"][o]
        kc, _, dc = A["p100c"][o]
        mm = int(dw["michel_found"]) != int(dc["michel_found"])
        sm = int(dw["is_stm"]) != int(dc["is_stm"])
        if mm or sm:
            out.append(dict(orig=o, kw=kw, kc=kc, verdict=v, dw=dw, dc=dc, michel_mover=mm, stm_mover=sm,
                            vol="top" if float(dc["stop_x"]) > 0 else "bottom",
                            rec_w=R["p99wflip"][kw], rec_c=R["p100c"][kc]))
    return out


def sheet_rows(path):
    out = {}
    for line in open(path):
        if line.startswith("#") or line.startswith("scan_id"):
            continue
        f = line.rstrip("\n").split("\t")
        out[f"{f[2]}/{f[3]}"] = dict(npts=f[4], muon_len_cm=f[5])
    return out


def answer(d):
    s = f"is_stm {int(d['is_stm'])}, michel_found {int(d['michel_found'])}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists (a scan set is never rebuilt in place)")
    if os.path.exists(LABELS):
        sys.exit(f"REFUSING: {LABELS} exists (M13: new scan => new tag)")

    mv = movers()
    keys = [m["kc"] for m in mv]
    if len(set(keys)) != len(keys):
        sys.exit("two record objects map to one p100c cluster; not a one-to-one set")
    for m in mv:
        m["tranche"] = 1 if m["michel_mover"] else 2
        if m["michel_mover"]:
            want = "michel" if m["verdict"] == "STM_MICHEL" else "no Michel"
            m["kind"] = (f"record {want}, michel_found {int(m['dw']['michel_found'])} -> {int(m['dc']['michel_found'])}")
        else:
            m["kind"] = f"record {m['verdict']}, is_stm {int(m['dw']['is_stm'])} -> {int(m['dc']['is_stm'])}"
    mv.sort(key=lambda m: (m["tranche"], int(m["kc"].split("_")[0]), int(m["kc"].split("_")[1].split("/")[0]),
                           int(m["kc"].split("/")[1])))

    sheet = sheet_rows(P100 + "/sheet_p100c/pdvd_stm_michel_scan_sheet.tsv")
    os.makedirs(a.out + "/prep")
    os.symlink(P100 + "/prep_p100c/dqdx_ref_pdvd.json", a.out + "/prep/dqdx_ref_pdvd.json")
    man, q_items, rows = [], {}, []
    for i, m in enumerate(mv, 1):
        evt, cid = m["kc"].split("/")
        src = f"{P100}/prep_p100c/smprep-{evt}-c{cid}.json"
        if not os.path.exists(src):
            sys.exit(f"missing payload {src}")
        os.symlink(src, f"{a.out}/prep/smprep-{evt}-c{cid}.json")
        sr = sheet[m["kc"]]
        man.append(f"{i}\t{m['tranche']}\t{evt}\t{cid}\t{sr['npts']}\t{sr['muon_len_cm']}")
        dc, dw, rc = m["dc"], m["dw"], m["rec_c"]
        pay = json.load(open(src))["verdict"]
        mich = (f"its Michel: {pay.get('michel_ke_best') or 0:.1f} MeV, {pay.get('michel_len') or 0:.1f} cm, "
                f"conn type {pay.get('michel_conn_type')}" if int(pay.get("michel_found") or 0) else "no Michel object")
        conf = rc.get("confidence")
        src_rec = rc.get("latest_source") or rc.get("source")
        html = (
            f"<b>own100m &mdash; what the gain flip moves</b> (doc pdvd/100 round 2; not blind; tranche {m['tranche']} of 2)<br>"
            f"Shown on <b>p100c</b> ({DESC['p100c']}).<ul style='margin:2px 0 2px 0'>"
            f"<li>p100c chain: <b>{answer(dc)}</b> ({mich})</li>"
            f"<li>p99wflip chain ({DESC['p99wflip']}), cluster {m['kw']}: <b>{answer(dw)}</b></li>"
            f"<li>record (carried by geometry, drawn on production's display): <b>{m['verdict']}</b> ({conf}; {src_rec})</li>"
            f"</ul><b>Please judge on this display:</b> STM + MICHEL (Michel radio) if the muon stops and an electron leaves the "
            f"stop, STM if it stops with no Michel, THRU if it does not stop. UNCLEAR / MESSY as usual. A faint or short Michel "
            f"counts if you would call it a Michel on any display; say in the notes if it is visible only here.")
        q_items[m["kc"]] = dict(html=html)
        rows.append("\t".join(map(str, [i, m["tranche"], m["kc"], m["kw"], m["orig"], m["vol"], m["verdict"], conf,
                                        answer(dw), answer(dc), int(m["michel_mover"]), int(m["stm_mover"]), m["kind"]])))

    with open(a.out + "/manifest.tsv", "w") as fh:
        fh.write("# doc pdvd/100 round 2 -- own100m, p99wflip -> p100c chain movers, owner look (not blind)\n")
        fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
        fh.write("\n".join(man) + "\n")
    with open(a.out + "/questions.json", "w") as fh:
        json.dump(dict(scan="pdvd own100m, what the gain flip moves: p99wflip -> p100c chain movers (doc pdvd/100 round 2)",
                       prep=a.out + "/prep", items=q_items), fh, indent=1)
    with open(a.out + "/items.tsv", "w") as fh:
        fh.write("scan_id\ttranche\tkey_p100c\tkey_p99wflip\trecord_key\tvol\trecord_verdict\trecord_confidence\t"
                 "p99wflip\tp100c\tmichel_mover\tstm_mover\tkind\n")
        fh.write("\n".join(rows) + "\n")
    t1 = [m for m in mv if m["tranche"] == 1]
    print(f"items {len(mv)}: tranche 1 (michel_found movers) {len(t1)}, tranche 2 (is_stm-only movers) {len(mv) - len(t1)}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()

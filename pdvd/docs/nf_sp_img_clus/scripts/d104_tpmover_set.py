#!/usr/bin/env python3
"""doc pdvd/104 sec 5 -- the symmetric check of figs/104_pred_amend7.txt: the owner set own103h2 (PDHD, Michel only).

    D103_PDHD_RECORD=<smx27> python3 d104_tpmover_set.py --new-record <smx28> --owner-record <own103h> \
        --out /home/xqian/tmp/d104/own/set_pdhd_tpmover

Fork by duplication of d103_tpmover_set.py, which hard-codes PDVD cells, TAG own103v2 and SEED 106.  Forking rather
than adding --det keeps doc 103's committed record and figures reproducible byte-for-byte from the script that made
them.  Three things change, all from amendment 7:

  * Michel only (sec 2).  Both is_stm metrics already pass, so no is_stm mover is drawn and METRICS carries the
    michel_found index alone -- the index matters, michel_found is element 1 of each cell row, is_stm is element 0.
  * amendment 5's convention (sec 2): population() is called with stm_only_unset_negative=True, so an owner
    STM_ONLY with no michel_kind counts Michel-negative.  This is what the PDHD headline grade uses; calling it
    without the flag would draw from a different Michel truth than the grade the check feeds.
  * TAG own103h2, SEED 107 (sec 4).

Prints COUNTS ONLY.  The drawn keys, strata and roles go only into <out>/items.tsv (amendment 7 sec 4).  Refuses an
existing <out> or label dir (M13).  movers() is imported by d104_tpmover_score.py so both use one definition.
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d103_union_grade as U
from d103_owner_scan_set import QUESTION, PREP

DET = "pdhd"
TAG = "own103h2"
SEED = 107
N_STRATUM = 10
N_CONTROL = 4
METRICS = {"michel": 1}                 # name -> index into a cell row (is_stm, michel_found)


def context(new_record, owner_record):
    """truth (own103h > smx27 > smx28), rows of every cell, the grader's population, the owner test"""
    T, _ = U.load_truth(DET, new_record, owner_record)
    R = {lab: U.cell_rows(DET, arm) for lab, arm in U.CELLS[DET]}
    G = U.population(DET, T, R, stm_only_unset_negative=True)      # amendment 5 / amendment 7 sec 2
    conf = {r["key"]: r.get("confidence", "") for r in json.load(open(U.PDHD_RECORD))}
    for r in json.load(open(new_record)):
        conf.setdefault(r["key"], r.get("confidence", ""))
    owner = lambda k: T[k][2] in ("owner_review", "owner") or conf.get(k) == "owner"
    return T, R, G, conf, owner


def movers(T, R, G, owner):
    """{key: {'A1:michel'} or {'A0:michel'}} for every Michel TP-mover (amendment 7 sec 3)"""
    truth = {"michel": G["m_truth"]}
    out = {}
    for k in G["mpop"]:
        if owner(k) or not truth["michel"].get(k):
            continue
        for m, i in METRICS.items():
            g0, g1 = R["A0"].get(k, (0, 0))[i], R["A1"].get(k, (0, 0))[i]
            if g0 != g1:
                out.setdefault(k, set()).add(f"{'A1' if g1 else 'A0'}:{m}")
    return out


def strata(M):
    s1 = sorted(k for k, v in M.items() if any(x.startswith("A1:") for x in v))
    s0 = sorted(k for k, v in M.items() if k not in s1 and any(x.startswith("A0:") for x in v))
    return {"A1": s1, "A0": s0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-record", required=True)
    ap.add_argument("--owner-record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if [arm for _, arm in U.CELLS[DET]][::3] != ["d101hnew", "d102hcs"]:
        sys.exit(f"cells {U.CELLS[DET]}: A0/A1 must be d101hnew/d102hcs (amendment 7 sec 3)")
    if "smx27" not in U.PDHD_RECORD:
        sys.exit(f"D103_PDHD_RECORD={U.PDHD_RECORD}: amendment 3 requires smx27")
    labels = f"{U.IMG}/{DET}/work/stm_michel_labels/{TAG}"
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists (a scan set is never rebuilt in place)")
    if os.path.exists(labels):
        sys.exit(f"REFUSING: {labels} exists (M13: new scan => new tag)")

    T, R, G, conf, owner = context(a.new_record, a.owner_record)
    own_keys = {r["key"] for r in json.load(open(a.owner_record))}
    M = movers(T, R, G, owner)
    S = strata(M)
    drawn = {s: sorted(random.Random(SEED).sample(S[s], min(N_STRATUM, len(S[s])))) for s in ("A1", "A0")}
    pool = [k for k in sorted(set(R["A0"]) & set(R["A1"])) if k in G["judged"] and not owner(k) and k not in own_keys
            and R["A0"][k][:2] == R["A1"][k][:2] and R["A1"][k][0]]
    controls = sorted(random.Random(SEED).sample(pool, N_CONTROL))
    items = [(k, "A1") for k in drawn["A1"]] + [(k, "A0") for k in drawn["A0"]] + [(k, "control") for k in controls]
    random.Random(SEED).shuffle(items)

    arm = dict(U.CELLS[DET])
    def payload(k):
        ev, cid = k.split("/")
        for lab in ("A1", "A0"):
            p = f"{PREP.format(arm=arm[lab])}/smprep-{ev}-c{cid}.json"
            if k in R[lab] and os.path.exists(p):
                return p
        return None
    os.makedirs(a.out + "/prep")
    miss = [k for k, _ in items if not payload(k)]
    if miss:
        with open(a.out + "/MISSING_PAYLOADS.txt", "w") as fh:
            fh.write("\n".join(miss) + "\n")
        sys.exit(f"{len(miss)} drawn items have no payload (listed in {a.out}/MISSING_PAYLOADS.txt); set not written")

    os.symlink(f"{PREP.format(arm=arm['A1'])}/dqdx_ref_{DET}.json", f"{a.out}/prep/dqdx_ref_{DET}.json")
    ans = lambda lab, k: (f"is_stm {int(R[lab][k][0])} michel_found {int(R[lab][k][1])}" if k in R[lab] else "not a candidate")
    man, q, rows, shown = [], {}, [], {"A1": 0, "A0": 0}
    for i, (k, role) in enumerate(items, 1):
        ev, cid = k.split("/")
        src = payload(k)
        shown["A1" if f"/prep_{arm['A1']}/" in src else "A0"] += 1
        os.symlink(src, f"{a.out}/prep/smprep-{ev}-c{cid}.json")
        pay = json.load(open(src))
        man.append(f"{i}\t1\t{ev}\t{cid}\t{pay['npts']}\t{pay['muon_len_cm']:.1f}")
        q[k] = dict(html=QUESTION.replace("own103h", TAG))
        rows.append("\t".join(map(str, [i, k, role, ",".join(sorted(M.get(k, ()))) or "-", ans("A0", k), ans("A1", k),
                                        T[k][0], T[k][1], T[k][2], conf.get(k, "")])))
    with open(a.out + "/manifest.tsv", "w") as fh:
        fh.write(f"# doc pdvd/104 -- {TAG}, owner symmetric check (figs/104_pred_amend7.txt); roles only in items.tsv\n")
        fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n" + "\n".join(man) + "\n")
    with open(a.out + "/questions.json", "w") as fh:
        json.dump(dict(scan=f"{DET} {TAG}: owner symmetric check (doc pdvd/104, figs/104_pred_amend7.txt)",
                       prep=a.out + "/prep", items=q), fh, indent=1)
    with open(a.out + "/items.tsv", "w") as fh:
        fh.write("scan_id\tkey\trole\tmovers\tA0\tA1\tlabel_verdict\tlabel_michel_kind\tlabel_source\tlabel_confidence\n")
        fh.write("\n".join(rows) + "\n")
    n_m = {f"{s}:michel": sum(1 for v in M.values() if f"{s}:michel" in v) for s in ("A1", "A0")}
    print(f"{DET} {TAG}: Michel TP-movers {len(M)} {n_m}; strata A1 {len(S['A1'])}, A0 {len(S['A0'])}; "
          f"drawn A1 {len(drawn['A1'])}, A0 {len(drawn['A0'])}; controls {len(controls)} (pool {len(pool)}); "
          f"items {len(items)}; shown on A1 payload {shown['A1']}, A0 payload {shown['A0']}.  "
          f"Keys only in {a.out}/items.tsv")


if __name__ == "__main__":
    main()

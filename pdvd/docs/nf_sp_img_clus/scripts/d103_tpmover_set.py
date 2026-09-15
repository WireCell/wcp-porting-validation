#!/usr/bin/env python3
"""doc pdvd/103 sec 12 -- the symmetric check of figs/103_pred_amend6.txt: the owner set own103v2 (PDVD).

    D103_PDVD_CELLS=d103v0,d103v1 D103_PDVD_RECORD=<carried> python3 d103_tpmover_set.py --new-record <smx11> \
        --owner-record <own103v> --out /home/xqian/tmp/d103/own/set_pdvd_tpmover

TP-movers (amendment 6 sec 2): judged, non-owner-labelled items that are hand-positive on a metric and tagged on that
metric in exactly one of A0 d103v0 / A1 d103v1.  Stratum A1 = tagged in A1 only on either metric (wins a tie), stratum
A0 = tagged in A0 only.  Draw (sec 3): 10 per stratum with a fresh random.Random(106) each, 4 controls, shuffled with
random.Random(106), shown on A1's payload where the item is an A1 candidate, else A0.

Prints COUNTS ONLY.  The drawn keys and their roles are written only into <out>/items.tsv (amendment 6 sec 3).
Refuses an existing <out> or label dir.  movers() is imported by d103_tpmover_score.py so both use one definition.
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d103_union_grade as U
from d103_owner_scan_set import QUESTION, PREP

TAG = "own103v2"
SEED = 106
N_STRATUM = 10
N_CONTROL = 4
METRICS = ("is_stm", "michel")


def context(new_record, owner_record):
    """truth (own103v > carried > smx11), rows of both cells, the grader's population, owner test"""
    T, _ = U.load_truth("pdvd", new_record, owner_record)
    R = {lab: U.cell_rows("pdvd", arm) for lab, arm in U.CELLS["pdvd"]}
    G = U.population("pdvd", T, R)
    conf = {r["key"]: r.get("confidence", "") for r in json.load(open(U.PDVD_RECORD))}
    for r in json.load(open(new_record)):
        conf.setdefault(r["key"], r.get("confidence", ""))
    owner = lambda k: T[k][2] in ("owner_review", "owner") or conf.get(k) == "owner"
    return T, R, G, conf, owner


def movers(T, R, G, owner):
    """{key: set of 'A1:is_stm' / 'A0:michel' ...} for every TP-mover (amendment 6 sec 2)"""
    truth = {"is_stm": G["stm_truth"], "michel": G["m_truth"]}
    out = {}
    for k in G["pop"]:
        if owner(k):
            continue
        for i, m in enumerate(METRICS):
            if not truth[m].get(k):
                continue
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
    if [arm for _, arm in U.CELLS["pdvd"]] != ["d103v0", "d103v1"]:
        sys.exit(f"cells {U.CELLS['pdvd']}: set D103_PDVD_CELLS=d103v0,d103v1 (amendment 4)")
    labels = f"{U.IMG}/pdvd/work/stm_michel_labels/{TAG}"
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

    arm = dict(U.CELLS["pdvd"])
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

    os.symlink(f"{PREP.format(arm=arm['A1'])}/dqdx_ref_pdvd.json", f"{a.out}/prep/dqdx_ref_pdvd.json")
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
        fh.write(f"# doc pdvd/103 -- {TAG}, owner symmetric check (figs/103_pred_amend6.txt); roles only in items.tsv\n")
        fh.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n" + "\n".join(man) + "\n")
    with open(a.out + "/questions.json", "w") as fh:
        json.dump(dict(scan=f"pdvd {TAG}: owner symmetric check (doc pdvd/103, figs/103_pred_amend6.txt)",
                       prep=a.out + "/prep", items=q), fh, indent=1)
    with open(a.out + "/items.tsv", "w") as fh:
        fh.write("scan_id\tkey\trole\tmovers\tA0\tA1\tlabel_verdict\tlabel_michel_kind\tlabel_source\tlabel_confidence\n")
        fh.write("\n".join(rows) + "\n")
    n_m = {f"{s}:{m}": sum(1 for v in M.values() if f"{s}:{m}" in v) for s in ("A1", "A0") for m in METRICS}
    print(f"pdvd {TAG}: TP-movers {len(M)} {n_m}; strata A1 {len(S['A1'])}, A0 {len(S['A0'])}; drawn A1 {len(drawn['A1'])}, "
          f"A0 {len(drawn['A0'])}; controls {len(controls)} (pool {len(pool)}); items {len(items)}; shown on A1 payload "
          f"{shown['A1']}, A0 payload {shown['A0']}.  Keys only in {a.out}/items.tsv")


if __name__ == "__main__":
    main()

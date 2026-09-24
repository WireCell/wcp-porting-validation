#!/usr/bin/env python3
"""doc qlmatch/35 check C4 -- the STM / Michel grade of the ToT candidate against production, with the record carried.

d116_grade.py (the doc 113/116 rule, imported untouched) reads every cell's T_stm_michel keyed "run_idx/cluster_id"
and one truth dict keyed the same way.  That holds while the arms share a clustering; the ToT light changes the Q/L
flash association and the PR cluster ids are renumbered (first reading: 228 unlabelled candidates on the candidate arm,
251 control-only candidates -- d35/c4_grade_uncarried.txt).  Doc 29 met the same thing and carried the record by
geometry (d100_carry_verdicts.py).  This does the same inside the grade:

  every truth key and every control candidate (keyed on the control arm A0, where the records apply as-is: A0 grades
  identically to d116vflip) is carried to the candidate arm by d100_carry_verdicts.one() -- the same pctree keeps the
  key, else d99_match.match_cluster / status_of on the Bee clustering-global points with doc 99's pre-registered
  thresholds (R 1 cm, F_OK 0.7, split / merged / ambiguous refused, t0 x-offset retried), carried only on status ok;
  the candidate arm's rows are re-keyed into A0's key space through the inverse of that map (a target claimed by two
  keys is a collision and dropped); rows with no preimage keep a private key "<evt>/T<cid>" (unlabelled -> NEG / POS);
  the grade is then d116_grade.reading() unchanged.  Keys whose carry failed are listed; a second reading drops them
  from both cells (sensitivity only -- the first reading is the rule's).

    python3 d35_stm_grade.py --a0 q35ctl --t q35tk > ../d35/c4_grade.txt
"""
import argparse
import collections
import os
import sys
from concurrent.futures import ProcessPoolExecutor

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
NS = f"{IMG}/pdvd/docs/nf_sp_img_clus/scripts"
sys.path.insert(0, NS)
import d116_grade as G16            # noqa: E402  (imports d113_grade / d103_union_grade, sets the record env)
import d100_carry_verdicts as C     # noqa: E402

SCAN = f"{IMG}/pdvd/docs/scan"
EXTRA = f"{SCAN}/pdvd_stm_michel_smx116_verdicts.json"
OWNER = f"{SCAN}/pdvd_stm_michel_own116v_verdicts.json"


def carry_map(a0, t, keys, jobs):
    by_evt = collections.defaultdict(set)
    for k in keys:
        e, c = k.split("/")
        by_evt[e].add(int(c))
    todo = [(e, a0, t, sorted(cs)) for e, cs in sorted(by_evt.items())]
    fwd, status = {}, collections.Counter()
    with ProcessPoolExecutor(jobs) as ex:
        for (e, _, _, _), res in zip(todo, ex.map(C.one, todo)):
            for c, (tc, st) in res.items():
                status[st] += 1
                fwd[f"{e}/{c}"] = (f"{e}/{tc}" if tc is not None else None, st)
    return fwd, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--t", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--unlabelled-out", default=None,
                    help="tsv of every candidate with no truth in either cell (the items a scan would have to label)")
    a = ap.parse_args()
    det = "pdvd"
    T, src = G16.truth(det, EXTRA, OWNER)
    RA = G16.U.cell_rows(det, a.a0)
    RT_raw = G16.U.cell_rows(det, a.t)
    keys = set(T) | set(RA)
    fwd, status = carry_map(a.a0, a.t, keys, a.jobs)
    claims = collections.defaultdict(list)
    for k, (tk, st) in fwd.items():
        if tk is not None:
            claims[tk].append(k)
    inv = {tk: ks[0] for tk, ks in claims.items() if len(ks) == 1}
    collisions = {tk: ks for tk, ks in claims.items() if len(ks) > 1}
    RT, private, native = {}, 0, {}
    for k, v in RT_raw.items():
        if k in inv:
            RT[inv[k]] = v
            native[inv[k]] = k
        else:
            e, c = k.split("/")
            RT[f"{e}/T{c}"] = v
            native[f"{e}/T{c}"] = k
            private += 1
    if a.unlabelled_out:
        with open(a.unlabelled_out, "w") as fh:
            fh.write(f"# doc qlmatch/35 C4 -- candidates with no truth, per arm; key in {a.a0}'s id space, the arm's own key, "
                     "is_stm, michel_found\n# arm\tkey\tnative_key\tis_stm\tmichel_found\n")
            for lab, arm, R_, nat in (("A0", a.a0, RA, {}), ("T", a.t, RT, native)):
                for k in sorted(R_):
                    if k not in T:
                        fh.write(f"{arm}\t{k}\t{nat.get(k, k)}\t{R_[k][0]}\t{R_[k][1]}\n")
    failed = sorted(k for k, (tk, st) in fwd.items() if tk is None or tk in collisions)
    failed_judged = [k for k in failed if k in T]
    print(f"# doc qlmatch/35 C4 -- STM / Michel grade, A0={a.a0}, T={a.t} (T re-keyed into A0's cluster ids by geometry)")
    print(f"# truth sources (precedence order, items taken): {src}")
    print(f"# carry: {len(fwd)} keys (truth U A0 candidates); status {dict(status)}; collisions {len(collisions)} "
          f"targets ({sum(len(v) for v in collisions.values())} keys)")
    print(f"# T rows {len(RT_raw)}: re-keyed {len(RT_raw) - private}, private (no preimage) {private}")
    print(f"# keys not carried: {len(failed)} (of them judged in the record: {len(failed_judged)})")
    for k in failed_judged:
        tk, st = fwd[k]
        print(f"#   {k}: {st}{' (collision)' if tk in collisions else ''}  truth {T[k][0]}  A0 row {RA.get(k)}")
    cells = [("A0", a.a0), ("T", a.t)]

    def report(title, T_, R_):
        Gd, out, scen = G16.reading(det, T_, R_, cells, "A0")
        print(f"\n## {title}: population {len(Gd['pop'])} (fixed {len(Gd['fixed'])}); unlabelled candidates "
              f"A0 {sum(1 for k in Gd['unlabelled'] if k in R_['A0'])}, T {sum(1 for k in Gd['unlabelled'] if k in R_['T'])}")
        for m in ("is_stm", "michel_found"):
            for lab, arm in cells:
                c, (p, e), P, TR, chain = out[(m, lab)]
                (plo, phi), (elo, ehi) = G16.U.boot(P, TR, chain)
                print(f"  {m:12s} {lab:3s} {arm:8s} TP {c['tp']:3d} FP {c['fp']:3d} FN {c['fn']:3d} TN {c['tn']:4d} | "
                      f"purity {p:.3f} [{plo:.3f}, {phi:.3f}] efficiency {e:.3f} [{elo:.3f}, {ehi:.3f}]")
        verdicts = []
        for m in ("is_stm", "michel_found"):
            for i, name in ((0, "purity"), (1, "efficiency")):
                d = out[(m, "T")][1][i] - out[(m, "A0")][1][i]
                dn = scen["NEG"][(m, "T")][1][i] - scen["NEG"][(m, "A0")][1][i]
                dp = scen["POS"][(m, "T")][1][i] - scen["POS"][(m, "A0")][1][i]
                v113 = "PASS" if min(dn, dp) >= G16.LIMIT else "FAIL" if max(dn, dp) < G16.LIMIT else "UNDECIDED"
                verdicts.append(v113)
                print(f"  T {m:12s} {name:10s}: {d:+.3f}  (NEG {dn:+.3f}, POS {dp:+.3f})  doc-113 {v113}")
        print(f"  -> {'PASS' if all(v == 'PASS' for v in verdicts) else 'STOP'} (every metric PASS in both NEG and POS)")
        events = sorted({k.split("/")[0] for lab in R_ for k in R_[lab]})
        for hname, half in (("odd", set(events[1::2])), ("even", set(events[0::2]))):
            Rh = {lab: {k: v for k, v in R_[lab].items() if k.split("/")[0] in half} for lab in R_}
            _, outh = G16.D.grade(det, T_, Rh)
            ds = [outh[(m, "T")][1][i] - outh[(m, "A0")][1][i] for m in ("is_stm", "michel_found") for i in (0, 1)]
            print(f"  split-half {hname}: is_stm pur {ds[0]:+.3f} eff {ds[1]:+.3f}, michel pur {ds[2]:+.3f} "
                  f"eff {ds[3]:+.3f} (reported; doc 116 guard -0.040)")
        return verdicts

    v = report("reading 1 (the rule's): carry failures count as untagged in T", T, {"A0": RA, "T": RT})
    drop = set(failed)
    T2 = {k: x for k, x in T.items() if k not in drop}
    report("reading 2 (sensitivity): keys whose carry failed dropped from both cells", T2,
           {"A0": {k: x for k, x in RA.items() if k not in drop}, "T": RT})
    print(f"\nVERDICT C4 {a.t} vs {a.a0}: {'PASS' if all(x == 'PASS' for x in v) else 'STOP'} (reading 1)")


if __name__ == "__main__":
    main()

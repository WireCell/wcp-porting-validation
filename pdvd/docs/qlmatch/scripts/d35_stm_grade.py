#!/usr/bin/env python3
"""doc qlmatch/35 check C4 -- the STM / Michel grade of the ToT candidate against production, with the record carried.

d116_grade.py (the doc 113/116 rule, imported untouched) reads every cell's T_stm_michel keyed "run_idx/cluster_id"
and one truth dict keyed the same way.  That holds while the arms share a clustering; the ToT light changes the Q/L
flash association and the PR cluster ids are renumbered (first reading: 228 unlabelled candidates on the candidate arm,
251 control-only candidates -- d35/c4_grade_uncarried.txt).  Doc 29 met the same thing and carried the record by
geometry (d100_carry_verdicts.py).  This does the same inside the grade:

  the records are keyed on the doc 103-116 lineage (--key-arm, default d116vflip: the d103vflip pctrees).  Every truth
  key and every key-arm candidate is carried key-arm -> A0 and key-arm -> T by d100_carry_verdicts.one() (same pctree
  keeps the key, else d99_match.match_cluster / status_of on the Bee clustering-global points with doc 99's
  pre-registered thresholds, R 1 cm, F_OK 0.7, split / merged / ambiguous refused, t0 x-offset retried; carried only
  on status ok), and each cell's rows are re-keyed through the inverse of its map (a target claimed twice is a
  collision and dropped).  Rows with no preimage keep a private key "<evt>/A<cid>" / "<evt>/T<cid>", shared by the two
  cells where the A0 -> T carry puts one object in both (unlabelled -> NEG / POS).  The grade is then
  d116_grade.reading() unchanged.  Keys whose carry failed are listed; a second reading drops them from both cells
  (sensitivity only -- the first reading is the rule's).  The first doc-35 version keyed A0 on the records directly,
  which is wrong where A0 re-clustered (039349_39, 039349_46: ids shifted by 2) -- doc 35 sec 8.

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


def rekey(key_arm, arm, R_raw, K, jobs):
    """rows of `arm` re-keyed into key_arm's cluster ids: every key of K (key_arm space) is carried key_arm -> arm, and
    an arm row takes the key whose carry lands on it (a target claimed twice is a collision and dropped).
    -> (rows by key, native key by key, leftover native keys, forward map, status counter, collisions)"""
    if arm == key_arm:
        return dict(R_raw), {k: k for k in R_raw}, [], {k: (k, "identity") for k in K}, collections.Counter(identity=len(K)), {}
    fwd, status = carry_map(key_arm, arm, K, jobs)
    claims = collections.defaultdict(list)
    for k, (tk, st) in fwd.items():
        if tk is not None:
            claims[tk].append(k)
    inv = {tk: ks[0] for tk, ks in claims.items() if len(ks) == 1}
    coll = {tk: ks for tk, ks in claims.items() if len(ks) > 1}
    R, native, left = {}, {}, []
    for k, v in R_raw.items():
        if k in inv:
            R[inv[k]] = v
            native[inv[k]] = k
        else:
            left.append(k)
    return R, native, left, fwd, status, coll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--t", required=True)
    ap.add_argument("--key-arm", default="d116vflip",
                    help="the arm whose cluster ids the records are keyed on (the d103vflip pctree lineage of docs "
                         "103-116); both cells are re-keyed into it.  The first doc-35 reading keyed on A0 directly, "
                         "which is wrong in 039349_39 and 039349_46 (ids shifted by 2) -- doc 35 sec 8")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--scan-record", default=None,
                    help="doc 35 amendment 1: a blind-scan record (rows carry display_arm) folded at the LOWEST "
                         "precedence; a row's native key is mapped through its display arm's re-keying")
    ap.add_argument("--unlabelled-out", default=None,
                    help="tsv of every candidate with no truth in either cell (the items a scan would have to label)")
    a = ap.parse_args()
    det = "pdvd"
    T, src = G16.truth(det, EXTRA, OWNER)
    RK = G16.U.cell_rows(det, a.key_arm)
    K = set(T) | set(RK)
    RA_raw = G16.U.cell_rows(det, a.a0)
    RT_raw = G16.U.cell_rows(det, a.t)
    RA, natA, leftA, fwdA, stA, collA = rekey(a.key_arm, a.a0, RA_raw, K, a.jobs)
    RT, natT, leftT, fwdT, stT, collT = rekey(a.key_arm, a.t, RT_raw, K, a.jobs)
    # leftovers (no preimage in the key arm): one private key per object, shared by the two cells where the A0 -> T
    # carry of an A0 leftover lands on a T leftover
    shared = {}
    if leftA and leftT:
        fAT, _ = carry_map(a.a0, a.t, set(leftA), a.jobs)
        lt = set(leftT)
        cl = collections.Counter(tk for tk, _ in fAT.values() if tk in lt)
        shared = {tk: k for k, (tk, _) in fAT.items() if tk in lt and cl[tk] == 1}
    for k in leftA:
        e, c = k.split("/")
        RA[f"{e}/A{c}"] = RA_raw[k]
        natA[f"{e}/A{c}"] = k
    for k in leftT:
        if k in shared:
            e, c = shared[k].split("/")
            pk = f"{e}/A{c}"
        else:
            e, c = k.split("/")
            pk = f"{e}/T{c}"
        RT[pk] = RT_raw[k]
        natT[pk] = k
    folded = []
    if a.scan_record:
        import json
        n2k = {a.a0: {v: k for k, v in natA.items()}, a.t: {v: k for k, v in natT.items()}}
        for r in json.load(open(a.scan_record)):
            if r.get("calibration"):
                continue
            k = n2k.get(r["display_arm"], {}).get(r["key"])
            if k is None:
                folded.append((r["key"], r["display_arm"], "UNMAPPED", r["verdict"]))
            elif k in T:
                folded.append((r["key"], r["display_arm"], "already labelled " + k, r["verdict"]))
            else:
                T[k] = G16.U.row_truth(r, "smx35")
                folded.append((r["key"], r["display_arm"], k, r["verdict"]))
        src = src + [(os.path.basename(a.scan_record), sum(1 for f in folded if not f[2].startswith(("UNMAPPED", "already"))))]
    if a.unlabelled_out:
        with open(a.unlabelled_out, "w") as fh:
            fh.write(f"# doc qlmatch/35 C4 -- candidates with no truth, per arm; key in {a.key_arm}'s id space (A<cid> / "
                     f"T<cid> = no preimage there), the arm's own key, is_stm, michel_found\n"
                     "# arm\tkey\tnative_key\tis_stm\tmichel_found\n")
            for arm, R_, nat in ((a.a0, RA, natA), (a.t, RT, natT)):
                for k in sorted(R_):
                    if k not in T:
                        fh.write(f"{arm}\t{k}\t{nat[k]}\t{R_[k][0]}\t{R_[k][1]}\n")
    print(f"# doc qlmatch/35 C4 -- STM / Michel grade, A0={a.a0}, T={a.t}; both cells re-keyed by geometry into the "
          f"record lineage's cluster ids ({a.key_arm})")
    print(f"# truth sources (precedence order, items taken): {src}")
    for f in folded:
        print(f"#   fold {f[0]} ({f[1]}) -> {f[2]}: {f[3]}")
    for lab, arm, R_raw, fwd, st, coll, left in (("A0", a.a0, RA_raw, fwdA, stA, collA, leftA),
                                                 ("T", a.t, RT_raw, fwdT, stT, collT, leftT)):
        moved = sum(1 for k, (tk, _) in fwd.items() if tk is not None and tk != k)
        failed = sorted(k for k, (tk, _) in fwd.items() if (tk is None or tk in coll) and k in T)
        print(f"# {lab} {arm}: carry of {len(fwd)} keys {dict(st)}, collisions {len(coll)}; carried to a DIFFERENT id "
              f"{moved}; rows {len(R_raw)}, no preimage {len(left)}"
              + (f" (shared with an A0 leftover {len(shared)})" if lab == "T" else ""))
        print(f"#   judged keys not carried: {len(failed)}")
        for k in failed:
            print(f"#     {k}: {fwd[k][1]}  truth {T[k][0]}")
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

    v = report("reading 1 (the rule's): carry failures count as untagged", T, {"A0": RA, "T": RT})
    drop = {k for fwd, coll in ((fwdA, collA), (fwdT, collT)) for k, (tk, _) in fwd.items()
            if (tk is None or tk in coll) and k in T}
    T2 = {k: x for k, x in T.items() if k not in drop}
    report("reading 2 (sensitivity): keys whose carry failed dropped from both cells", T2,
           {"A0": {k: x for k, x in RA.items() if k not in drop}, "T": RT})
    print(f"\nVERDICT C4 {a.t} vs {a.a0}: {'PASS' if all(x == 'PASS' for x in v) else 'STOP'} (reading 1)")


if __name__ == "__main__":
    main()

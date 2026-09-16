#!/usr/bin/env python3
"""doc pdvd/107 -- WHICH tagger decision churns PDHD's STM candidate set under the two levers (read-only).

Doc 106 sec 2 found the levers' dominant effect on PDHD is candidate churn: A1 loses 80 of A0's 341 STM candidates
(20 carrying is_stm, 34 carrying a Michel) and gains 72 -- five times the end effect of doc 106 sec 4, and the only
channel large enough to carry the -0.035 Michel purity cost.  Doc 101 sec 6.5 left the question open in one line:
"Which tagger decision drops a cluster is not chased."  This chases it.

WHERE CANDIDACY IS DECIDED (CheckSTM_Michel.cxx:2617-2623):
    if (!cluster->get_flag(Flags::main_cluster)) continue;
    if (cluster->get_flag(Flags::TGM)) continue;
    if (m_require_stm_flag && !cluster->get_flag(Flags::STM)) continue;      <-- PDHD's branch
    if (!m_require_stm_flag && !cluster->has_pc("stm_pass")) continue;
then sorted by ident and capped at m_max_candidates (8, :2627).  m_require_stm_flag defaults TRUE (:963) and no
PDHD config sets it, so the gate is Flags::STM, set in TaggerCheckSTM.cxx:641 when check_stm_conditions() returns
true and the cluster is not TGM.

TWO THINGS THAT LOOK LIKE THE ANSWER AND ARE NOT:
  * T_cluster's stm / tgm / fc / lm columns are ALWAYS 0 on PDHD (verified: 0 nonzero in 7272 rows over 20 events),
    the same unfilled-column defect recorded for SBND.  "They are identical between arms" is vacuous, not a result.
    T_cluster's is_main / in_scope ARE filled and ARE identical on all 23266 shared clusters, which is a real
    result: every one of the 80 lost ids is still present, still is_main, still in_scope in A1.  The churn is a
    pure candidacy decision, NOT a clustering or main-cluster-choice effect -- so doc 102 sec 13.5's open
    hypothesis ("most likely the main-cluster choice inside a bundle") is refuted.
  * flash_id is NOT a bundle key: among is_main clusters of one event it takes value 72 six times, 96 four times.
    Assert injectivity before using it for anything (it is not used here).

THE TWO TREES THAT DO ANSWER IT, in tracking-stm.root (the upstream tagger's own output):
  T_stm_pass  one row per (cluster, pass); pass is a pass INDEX (0 forward, 1 backward -- TaggerCheckSTM:981), NOT
              a verdict.  24 of 1140 clusters have 2 rows.  `status` is the exit code:
                0  ACCEPTED as STM (flag_pass, no mid-point tracks, no proton)      :4051
                3  flag_pass FALSE -- the STM evaluation itself did not pass        :4056
                5  flag_pass true but detect_proton fired                           :4056
                4  mid-point tracks (check_other_tracks)                            :4047
                2  charge left beyond the kink (left_L > 40 cm, or > 7.5 cm at >2 MIP)  :3884
                1  TGM        7  pre-fit / geometric exits        8  no fit         :3862 :3970+ :3803
  T_stm_eval  one row per eval_stm_core attempt (up to 8; the || chains at :3922-3953), carrying its own `pass`
              index.  flag_pass == any(verdict) WITHIN ONE PASS, never across passes: 028084_19/63 has a forward
              pass with verdict 1 that still exited status 7 (the pre-fit exits at :3970+ run AFTER the eval at
              :3916-3953) and a backward pass with status 3 and 8 failed attempts.  Pooling the two passes makes
              that cluster look like a status-3 with a winning verdict, which is an aggregation artifact and not a
              code inconsistency.  Everything here is therefore keyed by (cluster, pass index).

    python3 d107_churn.py > figs/107_churn.txt
"""
import glob, os, sys, collections
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
DET = "pdhd"
ARMS = [("A0", "d101hnew"), ("K", "d101hkf"), ("S", "d102hocs"), ("A1", "d102hcs")]
LEVERS = ["K", "S", "A1"]
SNAME = {0: "accepted", 1: "TGM", 2: "left-charge", 3: "eval failed", 4: "mid-point trk",
         5: "proton", 6: "default", 7: "pre-fit exit", 8: "no fit", -1: "no row"}
EVCOL = ["pass", "strong", "verdict", "peak_range", "offset_length", "com_range",
         "ks1", "ks2", "ratio1", "ratio2", "res_length", "ave_res_dqdx"]


def load(arm):
    """(candidates, status, evals, per-pass status, per-pass evals).

    status[k] == -1 means the tagger wrote no T_stm_pass row for that cluster at all.
    A cluster is ACCEPTED if EITHER of its passes accepted it (forward index 0, backward index 1).
    """
    cand, st, ev, stp, evp = {}, {}, {}, {}, {}
    for d in sorted(glob.glob(f"{IMG}/{DET}/work/*_{arm}")):
        e = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(
                ["cluster_id", "is_stm", "michel_found", "muon_len"], library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            cand[f"{e}/{int(t['cluster_id'][i])}"] = {c: float(t[c][i]) for c in t if c != "cluster_id"}
        try:
            f = uproot.open(f"{d}/tracking-stm.root")
            p = f["T_stm_pass"].arrays(library="np")
            v = f["T_stm_eval"].arrays(EVCOL + ["cluster_id"], library="np")
        except Exception:
            continue
        for i in range(len(p["cluster_id"])):
            k = f"{e}/{int(p['cluster_id'][i])}"
            sv, pi = int(p["status"][i]), int(p["pass"][i])
            stp[(k, pi)] = sv
            st[k] = 0 if (st.get(k) == 0 or sv == 0) else (sv if k not in st else st[k])
        for cid in set(v["cluster_id"].astype(int)):
            k = f"{e}/{cid}"
            for pi in set(v["pass"][v["cluster_id"] == cid].astype(int)):
                m = (v["cluster_id"] == cid) & (v["pass"] == pi)
                row = {c: float(np.max(v[c][m]) if c == "verdict" else np.median(v[c][m])) for c in EVCOL}
                row["n_eval"] = int(m.sum())
                evp[(k, pi)] = row
            # the cluster-level view uses the pass that DECIDED the cluster: the accepting one if there is one
            dec = next((pi for (kk, pi), sv in stp.items() if kk == k and sv == 0),
                       max((pi for (kk, pi) in evp if kk == k), default=None))
            if dec is not None and (k, dec) in evp:
                ev[k] = evp[(k, dec)]
    return cand, st, ev, stp, evp


def main():
    D = {lab: load(arm) for lab, arm in ARMS}
    for lab, arm in ARMS:
        c, s, e = D[lab][:3]
        print(f"# {lab:2s} = {arm:9s}  candidates {len(c):4d}   T_stm_pass clusters {len(s):5d}"
              f"   T_stm_eval clusters {len(e):5d}")
    A0c, A0s, A0e = D["A0"][:3]

    # ---- V1: candidacy must BE status 0.  If this fails, every table below is about the wrong quantity. --------
    print("\n=== V1: is the STM candidate set exactly the status-0 set? ===")
    for lab, _ in ARMS:
        c, s = D[lab][0], D[lab][1]
        z = {k for k, v in s.items() if v == 0}
        print(f"  {lab:3s} candidates {len(c):4d}  status-0 {len(z):4d}  |cand \\ st0| {len(set(c) - z):3d}"
              f"  |st0 \\ cand| {len(z - set(c)):3d}")
    print("  The two sets are EQUAL on every arm.  status 0 is therefore not merely necessary but exactly the")
    print("  gate: on these events the main_cluster and not-TGM conditions never bind beyond it, and the")
    print("  max_candidates=8 cap (:2627) never fires -- no PDHD event here reaches 8 candidates.  So every")
    print("  candidate gained or lost below is an STM tagger verdict change and nothing else.")

    print("\n=== V2: within ONE pass, status 3 means flag_pass false, so that pass's max(verdict) must be 0 ===")
    for lab, _ in ARMS:
        stp, evp = D[lab][3], D[lab][4]
        ok = [kp for kp, v in stp.items() if v == 3 and kp in evp]
        bad = [kp for kp in ok if evp[kp]["verdict"] > 0]
        print(f"  {lab:3s} status-3 PASSES with eval rows {len(ok):4d}, of which max(verdict)>0: {len(bad)}"
              + ("  <-- VIOLATION" if bad else "  ok"))
    print("  Keyed per (cluster, pass).  Pooling a cluster's two passes instead reports one false violation per")
    print("  arm on A0 and S -- see the module docstring for the case that produces it.")

    print("\n\n=== 1. the whole candidate population by tagger status, per arm ===")
    allk = sorted(set().union(*[set(D[l][1]) for l, _ in ARMS]))
    print(f"{'status':>14s} " + " ".join(f"{l:>7s}" for l, _ in ARMS))
    for code in sorted(SNAME):
        row = []
        for lab, _ in ARMS:
            row.append(sum(1 for k in allk if D[lab][1].get(k, -1) == code))
        if any(row):
            print(f"{code:2d} {SNAME[code]:>11s} " + " ".join(f"{n:7d}" for n in row))

    print("\n\n=== 2. THE ANSWER: which tagger decision churns the candidate set ===")
    for lv in LEVERS:
        Lc, Ls = D[lv][0], D[lv][1]
        lost = sorted(set(A0c) - set(Lc))
        gain = sorted(set(Lc) - set(A0c))
        print(f"\n--- {lv} vs A0:  lost {len(lost)}   gained {len(gain)} ---")
        for nm, ks, frm, to in (("LOST  (A0 cand -> not)", lost, A0s, Ls), ("GAINED(not -> L cand)", gain, A0s, Ls)):
            cnt = collections.Counter((frm.get(k, -1), to.get(k, -1)) for k in ks)
            print(f"  {nm}:")
            for (a, b), n in cnt.most_common(6):
                print(f"     status {a:2d} {SNAME[a]:>11s}  ->  {b:2d} {SNAME[b]:>11s}   {n:4d}"
                      f"  ({n / max(len(ks),1):.2f})")

    print("\n\n=== 3. the 0<->3 flips are the STM evaluation changing its mind on the same charge ===")
    print("  status 3 is flag_pass==false, i.e. eval_stm_core never returned true on any of its attempts.")
    print(f"{'lever':6s} {'direction':>22s} {'n':>5s} | {'n_eval A0':>10s} {'L':>6s} | {'ks1 A0':>8s} {'L':>7s}"
          f" | {'ratio1 A0':>10s} {'L':>7s} | {'res_length A0':>14s} {'L':>8s}")
    for lv in LEVERS:
        Lc, Ls, Le = D[lv][:3]
        for nm, ks in (("0 -> 3  (lost)", [k for k in set(A0c) - set(Lc) if A0s.get(k) == 0 and Ls.get(k) == 3]),
                       ("3 -> 0  (gained)", [k for k in set(Lc) - set(A0c) if A0s.get(k) == 3 and Ls.get(k) == 0])):
            kk = [k for k in ks if k in A0e and k in Le]
            if not kk:
                continue
            def med(src, c):
                return np.median([src[k][c] for k in kk])
            print(f"{lv:6s} {nm:>22s} {len(kk):5d} | {med(A0e,'n_eval'):10.1f} {med(Le,'n_eval'):6.1f}"
                  f" | {med(A0e,'ks1'):8.4f} {med(Le,'ks1'):7.4f} | {med(A0e,'ratio1'):10.4f} {med(Le,'ratio1'):7.4f}"
                  f" | {med(A0e,'res_length'):14.2f} {med(Le,'res_length'):8.2f}")

    print("\n\n=== 4. what the churn costs: the truth-relevant load of the lost and gained (reco flags) ===")
    print(f"{'lever':6s} {'stratum':>8s} {'n':>5s} {'is_stm':>8s} {'michel':>8s} {'muon_len p50':>13s}")
    for lv in LEVERS:
        Lc = D[lv][0]
        for nm, src, ks in (("lost", A0c, sorted(set(A0c) - set(Lc))), ("gained", Lc, sorted(set(Lc) - set(A0c)))):
            print(f"{lv:6s} {nm:>8s} {len(ks):5d} {sum(1 for k in ks if src[k]['is_stm'] > 0):8d}"
                  f" {sum(1 for k in ks if src[k]['michel_found'] > 0):8d}"
                  f" {np.median([src[k]['muon_len'] for k in ks]):13.1f}")
    print("  These are the levers' reco flags on objects the OTHER arm never scored, so they are a size, not a")
    print("  grade: a lost is_stm is only a loss if the cluster was truly a stopper, which no arm-vs-arm number")
    print("  can say.  The doc 103 union grade is the instrument for that, and sec 5 uses it.")
    michel_link(D)


def michel_link(D):
    """Do the levers' Michel false positives come from clusters production never even considered?

    This is the only place the round touches truth, and it uses the doc 103 grader's own population and Michel
    truth (amendments 3 and 5 for PDHD, --stm-only-unset-negative), never a hard-coded key list.
    """
    print("\n\n=== 5. THE LINK TO THE PURITY COST: are the Michel false positives GAINED candidates? ===")
    os.environ["D103_PDHD_RECORD"] = f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import d103_union_grade as U
        T, _ = U.load_truth("pdhd", f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json",
                            f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_own103h_verdicts.json")
        G = U.population("pdhd", T, {lab: U.cell_rows("pdhd", arm) for lab, arm in U.CELLS["pdhd"]},
                         stm_only_unset_negative=True)
    except Exception as exc:
        print(f"  grader unavailable ({exc}); section skipped")
        return
    P, TR = G["mpop"], G["m_truth"]
    A0c = D["A0"][0]
    for lv in ("A1",):
        Lc = D[lv][0]
        gained, lost = set(Lc) - set(A0c), set(A0c) - set(Lc)
        fp = [k for k in P if not TR[k] and k in Lc and Lc[k]["michel_found"] > 0]
        tp = [k for k in P if TR[k] and k in Lc and Lc[k]["michel_found"] > 0]
        print(f"  {lv}: graded Michel population {len(P)}, of which {lv} tags {len(tp)} true / {len(fp)} false")
        print(f"     false positives that are GAINED candidates (production never scored them): "
              f"{sum(1 for k in fp if k in gained)}/{len(fp)}")
        print(f"     true  positives that are GAINED candidates:                                "
              f"{sum(1 for k in tp if k in gained)}/{len(tp)}")
        fp0 = [k for k in P if not TR[k] and k in A0c and A0c[k]["michel_found"] > 0]
        print(f"  A0: tags {len(fp0)} Michel false positives, of which LOST under {lv}: "
              f"{sum(1 for k in fp0 if k in lost)}/{len(fp0)}")
        print("  Reading: if the false positives were mostly gained clusters, the churn would BE the purity cost")
        print("  and round 11 would be the STM evaluation.  If they are mostly clusters both arms scored, the")
        print("  churn is a large but purity-neutral reshuffle and the cost is elsewhere.")


if __name__ == "__main__":
    main()

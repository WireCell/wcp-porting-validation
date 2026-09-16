#!/usr/bin/env python3
"""doc pdvd/104 sec 2 -- the Michel admission feature table, per candidate, both detectors (read-only, no chain run).

Doc 103 sec 14.4b left round 7 as "build an admission rule on sec 10.4's class".  This script is the first half of
testing that premise: it lays out every quantity CheckSTM_Michel's admission actually turns on, for the true Michels
and the false positives side by side, so a cut can be sized -- or refuted -- on the real distributions.

The admission gate is stm_michel_michel_gate (StmMichelFunctions.cxx:668-688) acting on a StmMichelArm measured at
the stop vertex (measure_arm, :585-605); michel_found is derived, rec.michel_found = (michel_conn_type > 0)
(CheckSTM_Michel.cxx:4482).  The gate's own inputs -- len, mip, kink_deg, far_len -- are persisted per cluster as
michel_len / michel_mip / michel_kink_deg / michel_far_len, so the table below is the gate's view of each candidate.

WHAT conn_type MEANS, and why the table is split on it (this is the trap the length floor of sec 2.2 falls into):
  1 attached     -- a kMichel arm at the stop vertex; michel_len is that arm's fitted length
  2 bridged      -- no attached arm; the nearest admitted companion piece seeds the object
  3 charge only  -- no fitted segment at all, only an unfitted companion cluster
For conn 2 and 3 there is no fitted seed arm, so michel_len is ~0 BY CONSTRUCTION, not because the object is small.
A floor on michel_len pooled over all three therefore removes conn 2/3 objects wholesale for a reason that has
nothing to do with their size.

Truth is each detector's headline, set here so the script is self-contained (amendments 3, 5 for PDHD; 4 for PDVD):
  PDHD  own103h > smx27 > smx28, --stm-only-unset-negative; cells d101hnew (A0, production) / d102hcs (A1)
  PDVD  own103v > carried > smx11;                          cells d103v0 (A0, production) / d103v1 (A1)
Population and Michel truth come from d103_union_grade.population -- the grader's own definition, all cells of the
lineage -- so the TP/FP counts here are the same objects the grade of doc 103 sec 10.3 / sec 12 counts.

    python3 d104_michel_features.py > figs/104_michel_features.txt
"""
import glob, os, sys

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
os.environ["D103_PDHD_RECORD"] = f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json"
os.environ["D103_PDVD_CELLS"] = "d103v0,d103v1"
os.environ["D103_PDVD_RECORD"] = f"{IMG}/pdvd/docs/scan/pdvd_stm_michel_p99rwon_carried_corrected_verdicts.json"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uproot
import d103_union_grade as U

SETUP = {
    "pdhd": dict(new=f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json",
                 own=f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_own103h_verdicts.json", cells=("d101hnew", "d102hcs")),
    "pdvd": dict(new=f"{IMG}/pdvd/docs/scan/pdvd_stm_michel_smx11_verdicts.json",
                 own=f"{IMG}/pdvd/docs/scan/pdvd_stm_michel_own103v_verdicts.json", cells=("d103v0", "d103v1")),
}

# every branch the Michel admission turns on, plus the stop/muon bookkeeping sec 3 needs
COLS = ["cluster_id", "is_stm", "michel_found", "michel_conn_type", "michel_len", "michel_mip", "michel_kink_deg",
        "michel_far_len", "michel_dis_cm", "michel_ke_best", "michel_n_pieces", "michel_n_clusters",
        "n_dots", "n_dot_clusters_unfit", "n_stop_arms", "contrast", "contrast_expected", "bragg_valid",
        "short_track", "muon_len", "n_retreat", "n_split", "stop_dis", "stop_x", "stop_y", "stop_z"]


def rows(det, arm, cols=COLS):
    """One dict per (event, cluster) from an arm's T_stm_michel.  Key format is the grader's: '<run>_<evt>/<cid>'."""
    out = {}
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(cols, library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            out[f"{ev}/{int(t['cluster_id'][i])}"] = {c: float(t[c][i]) for c in cols if c != "cluster_id"}
    return out


def load(det):
    """The grader's Michel population and truth, plus both cells' rows and production A0's purity/efficiency.

    Returns dict(P, TR, A0, A1, nP, p0, e0, mexcl).  p0/e0 are production's numbers: every lever in sec 2 is scored
    against them unchanged, because a knob ships WITH the flip and production has no knob (doc 103 sec 12.4 reports
    both framings; only the A1-only one describes a shippable unit).
    """
    s = SETUP[det]
    T, _ = U.load_truth(det, s["new"], s["own"])
    G = U.population(det, T, {lab: U.cell_rows(det, arm) for lab, arm in U.CELLS[det]}, stm_only_unset_negative=True)
    assert (U.CELLS[det][0][1], U.CELLS[det][-1][1]) == s["cells"], U.CELLS[det]
    A0, A1 = rows(det, s["cells"][0]), rows(det, s["cells"][1])
    P, TR = G["mpop"], G["m_truth"]
    nP = sum(1 for k in P if TR[k])
    tp0 = sum(1 for k in P if TR[k] and A0.get(k, {}).get("michel_found", 0))
    fp0 = sum(1 for k in P if not TR[k] and A0.get(k, {}).get("michel_found", 0))
    return dict(P=P, TR=TR, A0=A0, A1=A1, nP=nP, mexcl=G["mexcl"], cells=s["cells"],
                p0=tp0 / max(1, tp0 + fp0), e0=tp0 / max(1, nP), tp0=tp0, fp0=fp0)


def split(D, arm="A1"):
    """(true Michels tagged by that arm, false positives of that arm) as (key, featuredict) lists."""
    R = D[arm]
    tp = [(k, R[k]) for k in D["P"] if D["TR"][k] and R.get(k, {}).get("michel_found", 0)]
    fp = [(k, R[k]) for k in D["P"] if not D["TR"][k] and R.get(k, {}).get("michel_found", 0)]
    return tp, fp


def quant(vals, qs=(0, .25, .5, .75, 1)):
    v = sorted(vals)
    return [v[int(q * (len(v) - 1))] for q in qs] if v else []


def main():
    print("# doc pdvd/104 sec 2: the Michel admission feature table (A1 = both trajectory levers on), by conn_type.")
    print("# Gate inputs are michel_len/mip/kink_deg/far_len (stm_michel_michel_gate, StmMichelFunctions.cxx:668-688).")
    print("# conn 1 attached / 2 bridged / 3 charge-only; michel_len is ~0 BY CONSTRUCTION for 2 and 3.")
    for det in SETUP:
        D = load(det)
        tp, fp = split(D)
        print(f"\n===== {det}: A0 {D['cells'][0]} (production) purity {D['p0']:.3f} eff {D['e0']:.3f} "
              f"| A1 {D['cells'][1]} TP {len(tp)} FP {len(fp)} | Michel population {len(D['P'])} "
              f"(hand positives {D['nP']}, excluded {D['mexcl']}) =====")
        for conn in (1, 2, 3):
            t = [v for _, v in tp if v["michel_conn_type"] == conn]
            f = [(k, v) for k, v in fp if v["michel_conn_type"] == conn]
            if not t and not f:
                continue
            print(f"\n  -- conn {conn}: TP {len(t)}  FP {len(f)}")
            if t:
                for nm in ("michel_len", "michel_mip", "michel_kink_deg", "michel_ke_best"):
                    q = quant([x[nm] for x in t])
                    print(f"     TP {nm:16s} min/q1/med/q3/max: " + " ".join(f"{x:7.2f}" for x in q))
            for k, v in f:
                print(f"     FP {k:16s} len {v['michel_len']:5.1f} mip {v['michel_mip']:5.2f} "
                      f"kink {v['michel_kink_deg']:6.1f} far {v['michel_far_len']:5.1f} dis {v['michel_dis_cm']:5.1f} "
                      f"ke {v['michel_ke_best']:6.1f} pieces {int(v['michel_n_pieces'])} "
                      f"ndots {int(v['n_dots'])} is_stm {int(v['is_stm'])}")
        # the class doc 103 sec 10.4 named: tagged by A1 and not by A0
        A0 = D["A0"]
        only = [(k, v) for k, v in fp if not A0.get(k, {}).get("michel_found", 0)]
        shared = [(k, v) for k, v in fp if A0.get(k, {}).get("michel_found", 0)]
        print(f"\n  -- A1-only Michel false positives: {len(only)} (shared with A0: {len(shared)}); "
              f"these are the class round 7 was asked to cut on")
        for k, v in only:
            print(f"     {k:16s} conn {int(v['michel_conn_type'])} len {v['michel_len']:5.1f} "
                  f"mip {v['michel_mip']:5.2f} kink {v['michel_kink_deg']:6.1f} ke {v['michel_ke_best']:6.1f} "
                  f"| A0 candidate: {'yes' if k in A0 else 'NO'}")


if __name__ == "__main__":
    main()

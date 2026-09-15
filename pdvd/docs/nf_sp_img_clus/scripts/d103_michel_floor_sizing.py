#!/usr/bin/env python3
"""doc pdvd/103 sec 12.4 -- offline sizing of a Michel energy floor as the admission lever (read-only, no chain run).

Question: does one simple admission rule, "michel_found only when the chain's Michel carries >= E MeV"
(T_stm_michel.michel_ke_best), clear the PDHD Michel-purity failure of sec 10.3 without costing PDVD its D1 of sec 12?

This is an OFFLINE PREDICATE: michel_found is zeroed after the fact when michel_ke_best < E; is_stm is left as the arm
wrote it.  A knob at CheckSTM_Michel's Michel admission can differ (a rejected Michel may change what else the tagger
decides), so a promising floor must be re-derived from a real arm before it is believed.

Truth is each detector's current headline, set here so the script is self-contained:
  PDHD  own103h > smx27 > smx28, --stm-only-unset-negative (amendments 3, 5); cells d101hnew / d102hcs
  PDVD  own103v > carried > smx11 (amendment 4);                            cells d103v0 / d103v1
Population and Michel truth come from d103_union_grade.population (the grader's own definition, all cells of the
lineage).  Floor 0 must reproduce figs/103_union_grade_pdhd_own103h_stmonlyneg.txt and
figs/103_union_grade_pdvd_own103v.txt, including the PDHD Michel exclusion count.

    python3 d103_michel_floor_sizing.py > figs/103_michel_floor_sizing.txt
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
FLOORS = (0, 1, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15)
LIMIT = -0.020


def rows(det, arm):
    out = {}
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(
                ["cluster_id", "is_stm", "michel_found", "michel_ke_best"], library="np")
        except Exception:
            continue
        for c, s, m, e in zip(t["cluster_id"], t["is_stm"], t["michel_found"], t["michel_ke_best"]):
            out[f"{ev}/{int(c)}"] = (int(s), int(m), float(e))
    return out


def grade(P, TR, chain):
    tp = sum(1 for k in P if TR[k] and chain.get(k, 0))
    fp = sum(1 for k in P if not TR[k] and chain.get(k, 0))
    fn = sum(1 for k in P if TR[k] and not chain.get(k, 0))
    return tp, fp, fn, tp / max(1, tp + fp), tp / max(1, tp + fn)


def main():
    print("# doc pdvd/103 sec 12.4: offline Michel energy floor (michel_found := michel_found and michel_ke_best >= E); "
          "is_stm untouched; OFFLINE PREDICATE, not an arm")
    for det, s in SETUP.items():
        assert (U.CELLS[det][0][1], U.CELLS[det][-1][1]) == s["cells"], U.CELLS[det]
        T, _ = U.load_truth(det, s["new"], s["own"])
        G = U.population(det, T, {lab: U.cell_rows(det, arm) for lab, arm in U.CELLS[det]}, stm_only_unset_negative=True)
        P, TR = G["mpop"], G["m_truth"]
        R = {lab: rows(det, arm) for lab, arm in zip(("A0", "A1"), s["cells"])}
        assert all(set(R[lab]) == set(U.cell_rows(det, arm)) for lab, arm in zip(("A0", "A1"), s["cells"]))
        print(f"\n== {det} ({s['cells'][0]} -> {s['cells'][1]}; Michel population {len(P)}, excluded {G['mexcl']}, "
              f"hand positives {sum(TR.values())})")
        print("   E MeV |  A0 TP  FP  FN purity  eff |  A1 TP  FP  FN purity  eff | A1-A0 purity   eff | reading")
        for E in FLOORS:
            g = {lab: grade(P, TR, {k: int(v[1] and v[2] >= E) for k, v in R[lab].items()}) for lab in ("A0", "A1")}
            dp, de = g["A1"][3] - g["A0"][3], g["A1"][4] - g["A0"][4]
            print(f"   {E:5.1f} | {g['A0'][0]:5d} {g['A0'][1]:3d} {g['A0'][2]:3d}  {g['A0'][3]:.3f} {g['A0'][4]:.3f} | "
                  f"{g['A1'][0]:5d} {g['A1'][1]:3d} {g['A1'][2]:3d}  {g['A1'][3]:.3f} {g['A1'][4]:.3f} | "
                  f"      {dp:+.3f} {de:+.3f} | {'both within 0.02' if min(dp, de) >= LIMIT else 'fails'}")
        # the reference that matters for a flip: the floor applied to A1 only, against production A0 as it runs today
        print("   floor on A1 only, against production A0 unchanged:")
        g0 = grade(P, TR, {k: v[1] for k, v in R["A0"].items()})
        for E in FLOORS[1:]:
            g1 = grade(P, TR, {k: int(v[1] and v[2] >= E) for k, v in R["A1"].items()})
            dp, de = g1[3] - g0[3], g1[4] - g0[4]
            print(f"   {E:5.1f} | A1 TP {g1[0]:3d} FP {g1[1]:3d} FN {g1[2]:3d} purity {g1[3]:.3f} eff {g1[4]:.3f} | "
                  f"vs A0 {g0[3]:.3f}/{g0[4]:.3f}: {dp:+.3f} {de:+.3f} | {'both within 0.02' if min(dp, de) >= LIMIT else 'fails'}")
        lost = sorted((v[2], k) for k, v in R["A1"].items() if k in TR and TR[k] and v[1] and v[2] < 10)
        print(f"   A1 true Michels below 10 MeV (a floor there removes them): {len(lost)}: "
              + ", ".join(f"{k} {e:.1f}" for e, k in lost))
        fps = sorted((v[2], k) for k, v in R["A1"].items() if k in TR and not TR[k] and v[1])
        print(f"   A1 Michel false positives by energy: {len(fps)}: " + ", ".join(f"{k} {e:.1f}" for e, k in fps))


if __name__ == "__main__":
    main()

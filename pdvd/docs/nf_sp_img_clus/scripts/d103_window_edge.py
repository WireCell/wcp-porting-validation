#!/usr/bin/env python3
"""doc pdvd/103 sec 11 -- the readout-window-edge class in the PDVD production-lineage blind record smx11.  Read-only.

The rubric (d760e223) has no rule for a fit end at the edge of the recorded time window.  Run 039349's frame closes at
slice ~1595; tracks that reach it lose whatever lies past it.  Scanners resolved it differently: most read the edge as a
dead region (THRU, DEAD: / NO_MEASUREMENT:), w1_a5 applied rule 3 as written (FLAT_STOP STM_ONLY).

EDGE below is a hand-curated list, not a regex.  A phrase search over-matches (isochronous tracks at slice ~1600 on the
longer 039252 frames, channel-window edges, notes that say "not at a frame edge") and under-matches ("imaged slices
end").  An item is in the list when its scanner names the start or end of the recorded time window at or near the
fit end; the quote is the scanner's own words.  Tier "near": the edge is named but lies >= 40 slices past the fit end
and the call rests on another rule.  Excluded on purpose (the scanner says the edge is NOT at the end):
039349_56/48, 039349_55/75, 039349_64/56, 039252_14/80, 039253_17/21, 039252_6/97, 039252_8/43.

This script prints the class by scanner and verdict, each item's tags in A0 / A1, and the A0 / A1 grade
(d103_union_grade definitions, amendment 4 cells and truth) three ways: as labelled, with the class removed, and with
the class's stopper calls read as THRU (the "edge = dead region" reading).

    D103_PDVD_CELLS=d103v0,d103v1 D103_PDVD_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_p99rwon_carried_corrected_verdicts.json \
        python3 d103_window_edge.py > figs/103_window_edge_pdvd_prod.txt
"""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d103_union_grade as U

N11 = f"{U.IMG}/pdvd/docs/scan/pdvd_stm_michel_smx11_verdicts.json"
# key: (edge, tier, the scanner's words)
EDGE = {
    "039349_1/72":   ("end", "at",   "the measured time range ends ~14 slices past the fit end"),
    "039349_11/27":  ("end", "at",   "the end coincides with the end of the readout frame"),
    "039349_14/48":  ("end", "at",   "up to what looks like the readout-window edge (slice ~1595)"),
    "039349_23/52":  ("end", "near", "imaged slices end at ~1598, about 40 slices past the fit end (call rests on the z seam)"),
    "039349_23/55":  ("end", "at",   "fit end coincides with the end of the readout window (slice ~1597)"),
    "039349_3/68":   ("end", "at",   "the end coincides with the end of the readout frame"),
    "039349_30/28":  ("end", "at",   "the end is ~20 time slices before the edge of the recorded window"),
    "039349_36/50":  ("end", "at",   "fit end coincides with the end of the recorded time window"),
    "039349_36/59":  ("end", "at",   "W carries the track line from the fit end (~1575) to the readout-window edge (~1592)"),
    "039349_38/25":  ("end", "at",   "the end coincides with the end of the readout window (slice ~1597)"),
    "039349_38/50":  ("end", "at",   "the imaged window appears to end ~15 slices past the stop (profile rises)"),
    "039349_41/65":  ("end", "at",   "the event imaged slices end ~3 cm of drift past the fit end; rubric has no rule for this"),
    "039349_44/59":  ("end", "at",   "the measured time range appears to end at the fit end ... competing reading THRU"),
    "039349_47/51":  ("end", "at",   "~10 cm of track before the event imaged slices end; rubric has no rule for this"),
    "039349_49/73":  ("end", "at",   "the event imaged slices end ~2-3 cm of drift past the fit end; rubric has no rule for this"),
    "039349_5/25":   ("end", "at",   "grey above the stop in W up to slice ~1590 where the imaged window appears to end"),
    "039349_52/31":  ("end", "at",   "the event imaged slices end ~3 cm of drift past the fit end"),
    "039349_53/28":  ("end", "at",   "the fit end sits at the end of the measured time range (slice ~1595)"),
    "039349_56/20":  ("end", "near", "grey continues the line past the end to the edge of the time window (call rests on CONTINUES)"),
    "039349_56/56":  ("end", "at",   "fit end coincides with the end of the recorded time window (slice ~1570)"),
    "039349_57/21":  ("end", "at",   "the end is at the edge of the recorded time window (t~1582 vs window end ~1590)"),
    "039349_60/61":  ("end", "at",   "fit end coincides with the end of the recorded time window (slice ~1585)"),
    "039349_68/24":  ("end", "at",   "the track cells reach slice ~1585 where the imaged window appears to end"),
    "039349_69/20":  ("end", "at",   "the end profile is cut by the recorded time window (unruled rule-7 case)"),
    "039349_72/47":  ("end", "at",   "fit end ~10 slices before the end of the recorded frame"),
    "039349_73/70":  ("end", "at",   "the end coincides with the end of the readout frame"),
    "039349_76/32":  ("end", "at",   "the fit end coincides with the end of the recorded time window (slice ~1580)"),
    "039349_76/72":  ("end", "at",   "fit end coincides with the end of the readout window (slice ~1597)"),
    "039349_76/77":  ("end", "at",   "fit end ~10 slices before the end of the readout window"),
    "039349_77/34":  ("end", "at",   "the end coincides with the end of the readout frame"),
    "039349_79/36":  ("end", "at",   "the imaged window appears to end at the fit end"),
    "039253_16/104": ("end", "at",   "fit end ~18 slices before the end of the readout frame (blank past slice ~2500)"),
    "039349_29/40":  ("start", "at", "stop at time slice ~22, near the start of the readout frame"),
    "039349_4/76":   ("start", "at", "grey carries the line on to the readout-window edge (slice 0)"),
    "039349_52/59":  ("start", "at", "the imaged window appears to start at slice ~8"),
    "039349_65/45":  ("start", "at", "the stop maps to about slice 0 ... the event imaged slices begin there"),
    "039349_71/64":  ("start", "at", "stop at time slice ~20 at the start of the readout window"),
}
STOPPER = ("STM_MICHEL", "STM_ONLY")


def grade(pop, T, R):
    out = []
    for title, idx, truth in (("is_stm", 0, lambda k: T[k][0] in STOPPER),
                              ("michel_found", 1, lambda k: T[k][0] == "STM_MICHEL")):
        TR = {k: truth(k) for k in pop}
        v = {}
        for lab in ("A0", "A1"):
            c = U.counts(pop, TR, {k: x[idx] for k, x in R[lab].items()})
            v[lab] = U.pe(c) + (c["fp"],)
        out.append(f"{title:12s} purity {v['A0'][0]:.3f} -> {v['A1'][0]:.3f} ({v['A1'][0] - v['A0'][0]:+.3f}; FP {v['A0'][2]} -> {v['A1'][2]}), "
                   f"efficiency {v['A0'][1]:.3f} -> {v['A1'][1]:.3f} ({v['A1'][1] - v['A0'][1]:+.3f})")
    return out


def main():
    if not os.environ.get("D103_PDVD_CELLS"):
        sys.exit("set D103_PDVD_CELLS and D103_PDVD_RECORD (figs/103_pred_amend4.txt)")
    recs = {r["key"]: r for r in json.load(open(N11))}
    missing = [k for k in EDGE if k not in recs]
    assert not missing, f"curated keys not in smx11: {missing}"
    T, _ = U.load_truth("pdvd", N11)
    R = {lab: U.cell_rows("pdvd", arm) for lab, arm in U.CELLS["pdvd"]}
    srcs = collections.Counter(T[k][2] for k in EDGE)

    print(f"# doc pdvd/103 sec 11: smx11 items whose scanner names the recorded-time-window edge at the fit end: "
          f"{len(EDGE)} of {len(recs)} (truth source of these keys on the union record: {dict(srcs)})")
    print("by edge/tier: " + str(dict(sorted(collections.Counter(f"{e}/{t}" for e, t, _ in EDGE.values()).items()))))
    print("by verdict:   " + str(dict(collections.Counter(recs[k]["verdict"] for k in EDGE).most_common())))
    print("\nscanner: verdicts on edge items")
    tab = collections.defaultdict(collections.Counter)
    for k in EDGE:
        tab[recs[k]["scanner"]][recs[k]["verdict"]] += 1
    for s in sorted(tab, key=lambda x: int(x.split("_a")[-1])):
        print(f"  {s:7s} {dict(sorted(tab[s].items()))}")

    print("\nitems: key | edge/tier | scanner | verdict (conf) | truth used | A0 is_stm,michel | A1 is_stm,michel | words")
    for k in sorted(EDGE, key=lambda k: (EDGE[k][0], recs[k]["verdict"], k)):
        e, t, q = EDGE[k]
        r = recs[k]
        a0 = R["A0"].get(k); a1 = R["A1"].get(k)
        f = lambda x: "-  " if x is None else f"{x[0]},{x[1]}"
        print(f"  {k:14s} {e}/{t:4s} {r['scanner']:7s} {r['verdict']:10s} ({r['confidence'][:3]}) {T[k][0]:10s} "
              f"A0 {f(a0)}  A1 {f(a1)} | {q}")

    judged = {k for k, (v, mk, s) in T.items() if v not in ("MESSY", "UNCLEAR")}
    rec_keys = {r["key"] for r in json.load(open(U.PDVD_RECORD))}
    pop = sorted({k for k in judged if k in rec_keys and k in R["A0"]} | ((set(R["A0"]) | set(R["A1"])) & judged))
    print("\n== A0 -> A1 grade three ways (amendment 4 definitions; provisional agent labels; D2 if any metric < -0.020)")
    print(f"  as labelled (population {len(pop)}; reproduces figs/103_union_grade_pdvd_prod.txt):")
    print("    " + "\n    ".join(grade(pop, T, R)))
    pop_x = [k for k in pop if k not in EDGE]
    print(f"  edge class removed (population {len(pop_x)}; {len(pop) - len(pop_x)} removed):")
    print("    " + "\n    ".join(grade(pop_x, T, R)))
    T_dead = dict(T)
    flipped = [k for k in EDGE if k in pop and T[k][0] in STOPPER and T[k][2] == "new_agent"]
    for k in flipped:
        T_dead[k] = ("THRU", "none", T[k][2])
    print(f"  edge = dead region: the class's new-agent stopper calls read as THRU ({len(flipped)}: {', '.join(sorted(flipped))}):")
    print("    " + "\n    ".join(grade(pop, T_dead, R)))


if __name__ == "__main__":
    main()

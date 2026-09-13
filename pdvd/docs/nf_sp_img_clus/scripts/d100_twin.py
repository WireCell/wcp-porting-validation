#!/usr/bin/env python3
"""doc pdvd/100 -- doc 89's 0-FP rule for the two data-sized Michel thresholds, re-run on the gain-flipped scale.

    python3 d100_twin.py --prep DIR --record R --arm p100c > d100/twin_p100c.txt

A fork of d90_twin.py (that script stays untouched; it demands the smx6 record and only widens).  Over every candidate
payload in DIR it re-evaluates, on recorded inputs, the pre-registered grid (d100/prereg.md sec 2)

    topology_michel_ke_min in {2.0, 2.5, 3.0, 3.5, 4.0, 5.0} MeV  x  plateau_mip_hi in {1.6, 1.8, 2.0, 2.25, 2.5}

and grades is_stm on the record's judged items.  Both directions:
  * P1 (CheckSTM_Michel.cxx, stm_michel_topology_clear, runs last): the bits before P1 are reject_bits |
    topology_cleared_bits; a found Michel (conn 1/2) with michel_ke_best >= ke and michel_len >= 3 cm clears
    no_bragg / shape_flat / plateau_short.  Exact in both directions.
  * the plateau window: PM (plateau_off_mip) is cleared when bragg_valid and LO <= plateau_med/MIP <= hi, and SET when
    bragg_valid and plateau_med/MIP > hi.  The recorded plateau_med is the reading that stood (anchored or doc 75's
    geometric re-read), and the re-read itself depends on the window, so plateau predictions are approximate (doc 90:
    a lower bound when widening); a changed value is measured by an arm before adoption (prereg).
The twin must reproduce the arm itself at the arm's own values first (0 mismatches), or it predicts nothing.

Rule (prereg): keep the current values unless another grid point accepts MORE judged stoppers with NO MORE judged
non-stoppers; among those the fewest grid steps from the current point; ties keep the current values.
"""
import argparse, json, os, sys

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--record", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--cur-ke", type=float, default=3.0)
ap.add_argument("--cur-hi", type=float, default=2.0)
ap.add_argument("--json", default=None)
a = ap.parse_args()
os.environ["STM_SCAN_RECORD"] = a.record
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

KE = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
HI = [1.6, 1.8, 2.0, 2.25, 2.5]
NB, SF, PS, PM = 1 << 2, 1 << 3, 1 << 9, 1 << 10
MIP, LO = 55000.0, 0.6

R = {r["key"]: r for r in json.load(open(a.record))}
V = {}
for f in sorted(os.listdir(a.prep)):
    if f.startswith("smprep-") and f.endswith(".json"):
        V[f[7:-5].replace("-c", "/")] = json.load(open(os.path.join(a.prep, f)))["verdict"]


def bits(x, ke, hi):
    b = int(x["reject_bits"] or 0) | int(x.get("topology_cleared_bits") or 0)
    if x.get("bragg_valid") and x.get("plateau_med") is not None:
        r = (x["plateau_med"] or 0) / MIP
        if LO <= r <= hi:
            b &= ~PM
        elif r > hi:
            b |= PM
    clr = 0
    if x["michel_found"] and x["michel_conn_type"] in (1, 2) and x["michel_ke_best"] >= ke and x["michel_len"] >= 3.0:
        clr = b & (NB | SF | PS)
    return b & ~clr


def vol(x):
    for k in ("stop_x", "fit_stop_x", "tagger_stop_x"):
        if x.get(k) is not None:
            return "top" if x[k] > 0 else "bottom"
    return "?"


bad = [k for k, x in V.items() if bits(x, a.cur_ke, a.cur_hi) != int(x["reject_bits"] or 0)]
print(f"# doc pdvd/100 -- d100_twin.py on {a.arm}: prep {a.prep}; record {a.record} ({len(R)} items)")
print(f"twin vs the arm itself at ke {a.cur_ke} / hi {a.cur_hi} on {len(V)} candidates: {len(bad)} mismatches {bad[:8]}")
if bad:
    sys.exit("the twin does not reproduce the arm: no prediction")
J = [k for k in R if C.judged(R[k])]
print(f"judged record items {len(J)} (with a candidate on the arm {sum(k in V for k in J)})")


def census(ke, hi):
    stm = lambda k: k in V and bits(V[k], ke, hi) == 0
    out = {}
    for v in ("all", "top", "bottom"):
        keys = [k for k in J if v == "all" or (k in V and vol(V[k]) == v)]
        tp = sum(stm(k) and C.is_stopper(R[k]) for k in keys)
        fp = sum(stm(k) and not C.is_stopper(R[k]) for k in keys)
        fn = sum((not stm(k)) and C.is_stopper(R[k]) for k in keys)
        out[v] = (tp, fp, fn)
    movers = [k for k in sorted(V) if (bits(V[k], ke, hi) == 0) != (bits(V[k], a.cur_ke, a.cur_hi) == 0)]
    return out, movers, sum(bits(V[k], ke, hi) == 0 for k in V)


cur, _, n_cur = census(a.cur_ke, a.cur_hi)
print(f"\ncurrent ke {a.cur_ke} hi {a.cur_hi}: is_stm {n_cur}; judged TP/FP/FN all {cur['all']} top {cur['top']} bottom {cur['bottom']}")
print("\n  ke   hi    is_stm   all TP/FP/FN     top TP/FP/FN     bottom TP/FP/FN   movers vs current [record class]")
grid, cands = {}, []
for ke in KE:
    for hi in HI:
        c, mv, n = census(ke, hi)
        grid[f"{ke}/{hi}"] = dict(census=c, is_stm=n, movers=mv)
        cls = lambda k: ("unjudged" if k not in R else ("MESSY/UNCLEAR" if not C.judged(R[k]) else
                         ("stopper" if C.is_stopper(R[k]) else "non-stopper")))
        print(f"  {ke:<4} {hi:<5} {n:>6}   {str(c['all']):<16} {str(c['top']):<16} {str(c['bottom']):<16}  "
              + " ".join(f"{k}[{cls(k)}:{'+' if bits(V[k], ke, hi) == 0 else '-'}]" for k in mv))
        if c["all"][0] > cur["all"][0] and c["all"][1] <= cur["all"][1]:
            steps = abs(KE.index(ke) - KE.index(a.cur_ke)) + abs(HI.index(hi) - HI.index(a.cur_hi))
            cands.append((steps, -c["all"][0], ke, hi))
if cands:
    cands.sort()
    best = [x for x in cands if x[:2] == cands[0][:2]]
    if len(best) > 1:
        print(f"\nRULE: tie among {[(x[2], x[3]) for x in best]} -> keep ke {a.cur_ke} hi {a.cur_hi}")
    else:
        print(f"\nRULE: change to ke {best[0][2]} hi {best[0][3]} (more stoppers, no more non-stoppers; "
              f"{best[0][0]} grid steps) -- to be measured by an arm before adoption")
else:
    print(f"\nRULE: no grid point accepts more judged stoppers with no more non-stoppers -> keep ke {a.cur_ke} hi {a.cur_hi}")
if a.json:
    json.dump(dict(current=[a.cur_ke, a.cur_hi], grid=grid, candidates=cands), open(a.json, "w"), indent=1)

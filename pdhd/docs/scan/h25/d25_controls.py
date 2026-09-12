#!/usr/bin/env python3
"""doc pdhd/25 sec 6 -- FREEZE the blind re-judge's controls, on production's payload, BEFORE any arm
of sec 5 has a result.

    python3 d25_controls.py [--arm h23conf] [--seed 25]

Why two strata.  scan_harness.py --blind drops is_stm / reject_names from context.json, but a Michel the
chain found is still visible in the drawn object grouping.  doc pdvd/92's smx8 dropped an item for
exactly that mismatch.  Lever 1's predicted movers split on it (michel_found 1 on the P1-floor items,
0 on the ks_margin items), so each stratum gets controls that read like its own decision items:
  stratum M = production michel_found 1,  stratum N = production michel_found 0.

Population: the APA0-excluded MAJORITY population (a superset of strict), production is_stm 0, reject
bits a subset of the shape bits {no_bragg, shape_flat, plateau_off_mip, profile_sparse} -- the reading
every decision item shares.

THRU nearest the cut, 4 per stratum, ranked by (the rule is fixed here, before ranking):
  1. the number of reject bits left once shape_flat can never fire and the P1 floors are 5 MeV / 1.5 cm
     (d25_misses.reverdict with ks_margin -10, ke_min 5, len_min 1.5), fewest first;
  2. then the shortfall of a P1-eligible Michel (michel_found, conn_type 1 or 2) to 5 MeV / 1.5 cm,
     max(0, 1 - ke/5) + max(0, 1 - len/1.5), and 2.0 when there is no eligible Michel;
  3. then |contrast/expected - 0.60|.
  If a stratum has fewer than 4 read-alike THRU, it is filled from any-bits THRU on the same ranking,
  and the relaxation is printed.

Stopper controls, 2 per stratum: hand stoppers production misses, read-alike, that lever 1's offline
model (d25_misses.reverdict at ks_margin -0.10, P1 5 MeV / 1.5 cm, on THIS population -- it must
reproduce TWIN_K exactly on the strict subset) and h25r's pre-registered movers do NOT move, in a
seed-25 shuffle.  The FULL ordered lists are
printed: if an arm later moves a control, it joins the decision set and the next item in its list
takes its place, so the rule for replacements is also fixed now.
"""
import argparse, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d25_bragg_michel as Q
import d25_misses as M

SHAPE = M.B["no_bragg"] | M.B["shape_flat"] | M.B["plateau_off_mip"] | M.B["profile_sparse"]
# lever 1's offline twin (q3_misses.txt sec 4, all four APAs are a superset) and h25r's pre-registered movers
TWIN_K = {"028084_10/46", "028084_2/116", "028084_2/49", "028084_20/116", "028084_5/115", "029107_1/85",
          "029107_12/118", "029107_12/95", "029107_21/65", "029107_24/33", "029107_28/109", "029107_3/99",
          "029107_7/85"}
TWIN_R = {"028084_20/116", "028084_5/115", "029107_24/33"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="h23conf")
    ap.add_argument("--seed", type=int, default=25)
    a = ap.parse_args()
    P0 = M.prod_params("pdhd")
    its, _ = Q.items("pdhd", a.arm, "majority")
    got = Q.census(its)[0]
    if got != (79, 1, 30, 70):
        sys.exit(f"GATE FAILED: {a.arm} majority reads {got}, doc 25 says (79, 1, 30, 70)")
    print(f"GATE {a.arm} APA0-majority is_stm {got} == (79, 1, 30, 70) PASS")
    # the offline twin must name only items of this population's misses, or the exclusion list is stale
    allits, _ = Q.items("pdhd", a.arm, "all")
    allkeys = {x[0] for x in allits}
    stale = sorted((TWIN_K | TWIN_R) - allkeys)
    if stale:
        sys.exit(f"twin names keys with no candidate: {stale}")
    PX = dict(P0, ks_margin=-10.0, ke_min=5.0, len_min=1.5)

    def p1_short(d):
        if not int(d["michel_found"]) or int(d["michel_conn_type"]) not in (1, 2):
            return 2.0
        ke, ln = float(d["michel_ke_best"]), float(d["michel_len"])
        if not (math.isfinite(ke) and math.isfinite(ln)):
            return 2.0
        return max(0.0, 1 - ke / 5.0) + max(0.0, 1 - ln / 1.5)

    def ce(d):
        e = float(d["contrast_expected"])
        return float(d["contrast"]) / e if e > 0 else float("nan")

    def row(x):
        k, v, kind, src, d = x
        left = bin(M.reverdict(d, PX)[1]).count("1")
        c = ce(d)
        return (left, p1_short(d), abs(c - 0.60) if math.isfinite(c) else 9.0, k,
                M.names(int(d["reject_bits"])), src, "strict" if not d["apa_any0"] else "majority-only",
                int(d["michel_found"]), float(d["michel_ke_best"]), float(d["michel_len"]), c,
                float(d["ks_flat"]) - float(d["ks_mu"]))

    def show(r):
        return (f"    {r[3]:15s} bits_left {r[0]} p1_short {r[1]:.2f} |c/e-0.6| {r[2]:.2f}  "
                f"reject {r[4]:24s} truth-src {r[5]:12s} {r[6]:13s} mf {r[7]} ke {r[8]:.1f} len {r[9]:.1f} "
                f"c/e {r[10]:.2f} ks_gap {r[11]:+.3f}")

    # lever 1's offline movers on THIS population, from the model itself (TWIN_K is the strict list only;
    # the first freeze used it alone and let the majority-only mover 028084_29/53 into STOP_N's order --
    # corrected before any arm launched)
    PK = dict(P0, ks_margin=-0.10, ke_min=5.0, len_min=1.5)
    moverK = {x[0] for x in its if int(x[4]["is_stm"]) == 0 and M.reverdict(x[4], PK)[1] == 0}
    strict_keys = {x[0] for x in Q.items("pdhd", a.arm, "strict")[0]}
    if moverK & strict_keys != TWIN_K:
        sys.exit(f"offline lever-1 movers on strict {sorted(moverK & strict_keys)} != TWIN_K")
    print(f"offline lever-1 movers, majority population: {len(moverK)} ({len(moverK & strict_keys)} strict == TWIN_K)")
    EXCL = moverK | TWIN_K | TWIN_R
    out_thru, out_stop = {}, {}
    for s, mf in (("M", 1), ("N", 0)):
        base_pool = [x for x in its if int(x[4]["is_stm"]) == 0 and int(x[4]["michel_found"]) == mf]
        alike = [x for x in base_pool if int(x[4]["reject_bits"]) & ~SHAPE == 0 and int(x[4]["reject_bits"])]
        thru = sorted(row(x) for x in alike if x[1] == "THRU")
        relaxed = []
        if len(thru) < 4:
            relaxed = sorted(row(x) for x in base_pool if x[1] == "THRU" and x not in alike)
        print(f"\n== stratum {s} (production michel_found {mf}): THRU nearest the cut, ranked ==")
        print(f"  read-alike THRU {len(thru)}" + (f"; RELAXED to any-bits THRU for the rest ({len(relaxed)} more)" if relaxed else ""))
        ranked = thru + relaxed
        for i, r in enumerate(ranked):
            print(("  * " if i < 4 else "    ") + show(r)[4:] + ("   [relaxed: not read-alike]" if r in relaxed else ""))
        out_thru[s] = [r[3] for r in ranked]
        stop = sorted(x[0] for x in alike if x[1] in Q.STOP and x[0] not in EXCL)
        random.Random(a.seed * 10 + mf).shuffle(stop)
        byk = {x[0]: x for x in alike}
        print(f"\n== stratum {s}: stopper controls, seed-{a.seed} order (first 2 taken; a moved one is replaced by the next) ==")
        for i, k in enumerate(stop):
            print(("  * " if i < 2 else "    ") + show(row(byk[k]))[4:])
        out_stop[s] = stop
    print("\nFROZEN LISTS (in order; the first 4 THRU and first 2 stoppers per stratum are the controls):")
    for s in ("M", "N"):
        print(f"  THRU_{s}  {' '.join(out_thru[s])}")
        print(f"  STOP_{s}  {' '.join(out_stop[s])}")


if __name__ == "__main__":
    main()

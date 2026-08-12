#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/72 round 2 -- offline analysis of the ES3 merge
census (es3_census.py output): V5 (deg2==1 confirmation for 196649), V5b
(ang3>ang10 independence check), population report, grid scan over the
stub-guard predicate, and the residual ("misses") list.

Usage:
    python3 es3_analysis.py --census es3_census_census.tsv \\
                             --merges es3_census_merges.tsv \\
                             --pb es3_census_pb.tsv
"""
import argparse
import itertools
import sys

import pandas as pd


def load(census_path, merges_path, pb_path):
    census = pd.read_csv(census_path, sep="\t")
    merges = pd.read_csv(merges_path, sep="\t")
    pb = pd.read_csv(pb_path, sep="\t")
    return census, merges, pb


def add_short_long(df):
    """Per row, determine which arm (1 or 2) is shorter, and add
    len_short/len_long/deg_short/deg_long/nfit_short/nfit_long columns.
    Ties (len1==len2) default to arm 1 as short -- rare, and both arms are
    then symmetric anyway."""
    is1_short = df["len1"] <= df["len2"]
    df["len_short"] = df["len1"].where(is1_short, df["len2"])
    df["len_long"] = df["len2"].where(is1_short, df["len1"])
    df["deg_short"] = df["deg1"].where(is1_short, df["deg2"])
    df["deg_long"] = df["deg2"].where(is1_short, df["deg1"])
    df["nfit_short"] = df["nfit1"].where(is1_short, df["nfit2"])
    df["nfit_long"] = df["nfit2"].where(is1_short, df["nfit1"])
    df["ratio"] = df["len_long"] / df["len_short"].replace(0, float("nan"))
    return df


def find_196649(census, merges):
    """The event has multiple degree-2 junctions in its census (most never
    merge); the one that matters is the one production actually merged --
    cross-reference merges.tsv for its coordinates rather than taking an
    arbitrary row for the event."""
    mrow = merges[merges["event"] == 196649]
    if mrow.empty:
        return census.iloc[0:0]
    m = mrow.iloc[0]
    match = census[
        (census["event"] == 196649)
        & (census["vtx_x"].round(2) == round(m["vtx_x"], 2))
        & (census["vtx_y"].round(2) == round(m["vtx_y"], 2))
        & (census["vtx_z"].round(2) == round(m["vtx_z"], 2))
    ]
    return match


def v5_check(census, merges):
    print("\n=== V5: deg(far end of short arm) for evt 196649 ===")
    row = find_196649(census, merges)
    if row.empty:
        print("V5 FAILED: no census row found for evt196649's actual merge "
              "(check merges.tsv / corruption)!", file=sys.stderr)
        return None
    row = add_short_long(row.copy())
    r = row.iloc[0]
    print(f"evt196649 clus={r['clus']} vtx=({r['vtx_x']},{r['vtx_y']},{r['vtx_z']}) "
          f"len_short={r['len_short']:.2f} len_long={r['len_long']:.2f} "
          f"deg_short={r['deg_short']} deg_long={r['deg_long']} "
          f"ang10={r['ang10']:.3f} ang3={r['ang3']:.3f} predmerge={r['predmerge']}")
    if r["deg_short"] == 1:
        print("V5 PASSED: deg_short == 1 (short arm's far end is a free terminus, "
              "confirms round 1's inference)")
    else:
        print(f"V5 FAILED: deg_short == {r['deg_short']} (!= 1) -- the topological "
              "narrowing condition is VOID for this event. Stop and re-plan the "
              "predicate around the angle-ratio family alone.", file=sys.stderr)
    return r


def v5b_independence(census):
    print("\n=== V5b: is ang3 > ang10 independent, or mechanically implied by "
          "len_short<8 && ratio>3? ===")
    df = add_short_long(census.copy())
    regime = df[(df["len_short"] < 8) & (df["ratio"] > 3) & (df["ang10"] >= 0) & (df["ang3"] >= 0)]
    if regime.empty:
        print("No junctions in the (len_short<8 && ratio>3) regime -- cannot assess.")
        return
    n = len(regime)
    n_gt = (regime["ang3"] > regime["ang10"]).sum()
    frac = n_gt / n
    print(f"regime population: {n} junctions with len_short<8cm && len_long/len_short>3")
    print(f"  of those, ang3 > ang10 in {n_gt}/{n} = {frac:.1%}")
    if frac > 0.85:
        print("  -> HIGH: the ang3>ang10 ratio term looks largely mechanical "
              "(lever-arm asymmetry), not primarily a signal of two distinct "
              "particles meeting. Treat it as a WEAK tie-breaker, not a strong cut.")
    else:
        print("  -> The ratio term carries real separating power beyond the "
              "lever-arm-asymmetry artifact; keep it as a meaningful predicate term.")


def population_report(census, merges, pb):
    print("\n=== Population report ===")
    for sample in sorted(set(census["sample"]) | set(merges["sample"])):
        c = census[census["sample"] == sample]
        m = merges[merges["sample"] == sample]
        p = pb[pb["sample"] == sample]
        n_events = c["event"].nunique() if not c.empty else 0
        terminal = c  # NOTE: without a "final sweep" flag, approximate the
        # terminal-sweep population as max(sweep) per (event,clus); see below.
        print(f"{sample}: events_with_census_rows={n_events} "
              f"junctions(rows)={len(c)} merges={len(m)} pb_skips={len(p)}")

    df = add_short_long(census.copy())
    n_degenerate = ((df["nfit1"] < 2) | (df["nfit2"] < 2)).sum()
    print(f"\ndegeneracy population (nfit<2 on either arm, would break an "
          f"unguarded length-ratio predicate): {n_degenerate}/{len(df)}")

    # Terminal-sweep-only population: for each (sample,event,clus), the
    # terminal sweep is the max sweep value reached for that cluster (the
    # sweep with no further merge). Junctions logged only in earlier,
    # truncated sweeps are re-scans of ones already accounted for.
    df["grp"] = list(zip(df["sample"], df["event"], df["clus"]))
    max_sweep = df.groupby("grp")["sweep"].transform("max")
    terminal_df = df[df["sweep"] == max_sweep]
    print(f"terminal-sweep-only junctions: {len(terminal_df)} "
          f"(vs {len(df)} counting every re-scan sweep)")

    merges_per_event = merges.groupby(["sample", "event"]).size()
    print(f"\nmerges-per-event distribution (events with >=1 merge):")
    print(merges_per_event.describe())
    return terminal_df


def grid_scan(census, merges):
    print("\n=== Grid scan: (stub_max, len_ratio, ang3_min, ang_ratio, require_terminal) ===")
    df = add_short_long(census.copy())
    # Only evaluate the predicate against junctions ES3 actually merged
    # (predmerge==1) -- suppressing a declined junction is a no-op, the
    # blast radius is entirely about what the predicate would have blocked
    # among ACTUAL merges.
    merged = df[df["predmerge"] == 1].copy()
    print(f"total actual merges with an intact census row: {len(merged)}")

    target = merged[merged["event"] == 196649]
    if target.empty:
        print("WARNING: evt196649's merge not found in the predmerge==1 set "
              "(census corruption?) -- grid scan cannot verify the MUST-fix "
              "criterion.", file=sys.stderr)

    stub_max_grid = [6, 7, 8, 9, 10]
    len_ratio_grid = [2, 3, 4, 5]
    ang3_min_grid = [10, 15, 18, 20]
    ang_ratio_grid = [1.0, 1.1, 1.2]
    require_terminal_grid = [True, False]

    results = []
    for stub_max, len_ratio, ang3_min, ang_ratio, req_term in itertools.product(
        stub_max_grid, len_ratio_grid, ang3_min_grid, ang_ratio_grid, require_terminal_grid
    ):
        deg_guard = (merged["deg_short"] == 1) if req_term else True
        nfit_guard = (merged["nfit_short"] >= 2) & (merged["nfit_long"] >= 2)
        len_floor = merged["len_short"] > 0.5
        suppress = (
            nfit_guard & len_floor
            & (merged["len_short"] < stub_max)
            & (merged["ratio"] > len_ratio)
            & (merged["ang3"] > ang3_min)
            & (merged["ang3"] > ang_ratio * merged["ang10"])
            & deg_guard
        )
        n_suppress = suppress.sum()
        n_events_touched = merged.loc[suppress, "event"].nunique()
        keeps_196649 = bool(suppress[merged["event"] == 196649].any()) if not target.empty else None
        results.append(dict(
            stub_max=stub_max, len_ratio=len_ratio, ang3_min=ang3_min,
            ang_ratio=ang_ratio, require_terminal=req_term,
            n_suppress=n_suppress, n_events_touched=n_events_touched,
            keeps_196649=keeps_196649,
        ))

    rdf = pd.DataFrame(results)
    candidates = rdf[rdf["keeps_196649"] == True].sort_values(  # noqa: E712
        ["n_events_touched", "n_suppress"]
    )
    print(f"\ngrid points evaluated: {len(rdf)}; keep-196649 candidates: {len(candidates)}")
    print(candidates.head(15).to_string(index=False))
    return rdf, merged


def residual_list(merged, chosen, out_path):
    """Merges the chosen predicate does NOT suppress but that look
    suspicious (a real local kink, terminal short arm) -- the "misses"
    Bee population."""
    deg_guard = (merged["deg_short"] == 1) if chosen["require_terminal"] else True
    nfit_guard = (merged["nfit_short"] >= 2) & (merged["nfit_long"] >= 2)
    len_floor = merged["len_short"] > 0.5
    suppress = (
        nfit_guard & len_floor
        & (merged["len_short"] < chosen["stub_max"])
        & (merged["ratio"] > chosen["len_ratio"])
        & (merged["ang3"] > chosen["ang3_min"])
        & (merged["ang3"] > chosen["ang_ratio"] * merged["ang10"])
        & deg_guard
    )
    suspicious = (merged["ang3"] > 15) & (merged["deg_short"] == 1) & (~suppress)
    residual = merged[suspicious].sort_values(["ang3", "len_short"], ascending=[False, True])
    residual.to_csv(out_path, sep="\t", index=False)
    print(f"\nresidual (misses) list: {len(residual)} junctions -> {out_path}")
    print(residual[["sample", "event", "clus", "len_short", "len_long", "ratio",
                     "ang10", "ang3", "deg_short"]].head(20).to_string(index=False))
    return residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--merges", required=True)
    ap.add_argument("--pb", required=True)
    ap.add_argument("--residual-out", default="es3_residual.tsv")
    args = ap.parse_args()

    census, merges, pb = load(args.census, args.merges, args.pb)
    v5_check(census, merges)
    v5b_independence(census)
    population_report(census, merges, pb)
    rdf, merged = grid_scan(census, merges)

    candidates = rdf[rdf["keeps_196649"] == True].sort_values(  # noqa: E712
        ["n_events_touched", "n_suppress"]
    )
    if not candidates.empty:
        chosen = candidates.iloc[0]
        print(f"\n=== Chosen operating point ===\n{chosen}")
        residual_list(merged, chosen, args.residual_out)
    else:
        print("\nNo grid point keeps 196649 -- widen the grid.", file=sys.stderr)


if __name__ == "__main__":
    main()

# `flag_high_consistent` flag-aware ladder (SBND)

`flag_high_consistent` is the pre-LASSO quality gate: `cull_inconsistent` drops a cluster's
*non-consistent* bundles whenever it has a consistent one, so it must be **pure** (a wrong
consistent bundle culls the true match before the fit). The old single branch
(`ks<0.06 && ndf≥3`) recovered only 18/99 hand-scan true matches. The new ladder (toolkit
`highconsist_ladder`, SBND-on; see `match/docs/chisquare_flags_comparison.md` §4.1b) ports the
MicroBooNE prototype's multi-branch *structure* and re-derives every number from the **10 hand-scan
data events** — the prototype's χ²/ndf multipliers (36/45/55) cannot transfer because the SBND
post-recipe χ²/ndf for a good match is ~1–2, not tens.

## What the hand-scans showed (drives the design)

- **KS is the purity lever; χ²/ndf is not.** In every flag category true matches sit at ks
  ~0.09–0.12 while background sits at ks ~0.44–0.68, but χ²/ndf is ~15 for *both*. So the ladder is
  KS-led; χ²/ndf ceilings only fence the tail.
- The user's physical grouping is confirmed: *clean* → tight ks + low χ²; *two_boundary* → both
  moderate, rare in background; *at_x_boundary / close_to_PMT / window_truncated* → good ks but high
  χ² (missing charge).
- `flag_spec_end` is **never set in SBND** (0/99, 0/762) → not used.

## The TIGHT ladder (chosen: pure, lower efficiency OK)

`c2n = chi2/ndf`, OR of branches:

| # | branch | condition |
|---|---|---|
| B1 | clean very-good | `ndf≥3 && ks<0.06 && c2n<6` |
| B2 | general good | `ndf≥3 && ks<0.09 && c2n<4` |
| B3 | two_boundary | `flag_two_boundary && ndf≥3 && ks<0.10 && c2n<8` |
| B4 | x-bnd / close-PMT / trunc | `(at_x_boundary‖close_to_PMT‖window_truncated) && ndf≥5 && ks<0.08 && c2n<60` |

## Measured impact

| metric | single-branch | ladder (TIGHT) |
|---|---|---|
| flag recall (data hand-scans) | 18% (18/99) | **44%** (44/99) |
| flag purity (data) | 82% | **88%** |
| true matches culled (data) | 2 | **1** (one double-flash ambiguity) |
| end-to-end true-match agreement (data) | 89/100 | **93/100** (+4) |
| MC — purity / recall (validation only) | — | **94% / 53%** |
| end-to-end agreement (MC) | 87/113 | **92/113** (+5) |

The ~12% data "impurity" is mostly legitimate excellent-light matches (ks 0.03–0.09) the hand-scan
did not tag, so real purity is higher. MC is tighter (cleaner sim); the data-tuned ceilings are
conservative for it — MC validates "true matches survive / nothing breaks", it is **not** retuned to.

## Reproduce

```
PMT_NL=true ./run_ql_evt.sh data -calib all      # ladder is the SBND default; dumps -> work/ql_evt*/
PMT_NL=true ./run_ql_evt.sh mc   -calib all
```

Then split the regenerated `consistent` / `auto_selected` flags against the scan states in
`work/ql_labels/{data,mc}/.scan_state-evt*.json` (selected = true matches). The ladder ceilings are
jsonnet-exposed (`hc_clean_ks` … `hc_miss_min_ndf` in `cfg/.../sbnd/qlmatching.jsonnet`) for
retuning; set `highconsist_ladder: false` to recover the single-branch baseline.

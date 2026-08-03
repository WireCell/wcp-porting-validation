# Per-PMT light error and the data/MC normalization (hand-scan PE-error study)

## Purpose

The SBND charge–light matcher (`QLMatching`) assigns each PMT a light error of **30%**
(`m_pe_err_frac = 0.3`, `match/inc/WireCellMatch/QLMatching.h`), applied as
`PE_err = (meas<1) ? 0.3 : 0.3·meas` (`match/src/Opflash.cxx`) and entering the chi² through
`σ² = meas + PE_err²`. This study **measures** that error from the trusted (flash ↔ cluster)
matches that were hand-scanned in the `ql_scan` viewer (10 data + 10 MC events), rather than
assuming it.

`sbnd_xin/scripts/analysis/ql/ql_pe_error.py` produces the plots; this note records the method and findings.

## Model and estimator

Per PMT, model the squared residual's expectation as

```
E[(pred − meas)²] = meas + a · pred²            a = (fractional error)²    (30% ⇒ a = 0.09)
```

- `pred` = the matcher's predicted PE for that PMT, **summed over the clusters matched to the
  flash** (`bundle.pred_pe`, group-summed; raw `q·QtoL·vis·eff`, with the per-PMT non-linearity
  applied above the 700 PE knee — see `match/docs/sbnd-opdetsim-chain.md`).
- `meas` = the flash's measured PE for that PMT (`flashes[gid].pe`).
- Headline estimator (method of moments, mean-based): `a_hat = Σ(R−meas) / Σ pred²`,
  `R = (pred−meas)²`. Report `√a_hat` as the fractional error.
- Cross-check: the pull `(pred−meas)/√(meas + a·pred²)` has robust width (1.4826·MAD) ≈ 1 when
  `a` is right; we report the width at `a_hat` and at `a = 0.09` (30%).

> Use the **mean** per bin, not the median: `R` is `σ²·χ²₁` (median ≈ 0.455 σ²), so a median
> profile would bias `a` low by ~2×.

## Selection (the hand-scan cuts)

1. Reconstruct the selected bundles per event by matching `(flash_gid, main_cluster)` from
   `work/ql_labels/<mode>/.scan_state-evt*.json` against the calib dump's bundles.
2. **Aggregate per flash**: `pred` summed over all selected bundles sharing a `flash_gid`,
   `meas` counted once (some flashes carry two hand-picked clusters, e.g. data evt1720/2028).
3. Drop a flash if any contributing bundle is `window_truncated` or `close_to_PMT`.
4. Keep PMT channels only (`opdet type == 1`; type 0 = X-ARAPUCA, always 0) in the flash's TPC,
   with `meas > 0` or `pred > 0`. Fit on `pred > 0`.
5. *High-consistency variant*: additionally require every contributing bundle to have
   `chi²/ndf < 5` and `ks_dis < 0.1`.

Predicted light is read NL-on from `work/ql_nl_study/calib-evt<ID>.nl.json` and NL-off from the
linear `work/ql_evt<ID>/calib-evt<ID>.json`. The two are verified identical in clusters/flashes
(only `pred_pe` differs), and every hand-scan key resolves in the regenerated dumps.

## Results

| variant | data `a` (√a) | data meas/pred | MC `a` (√a) | MC meas/pred |
|---|---|---|---|---|
| NL-on, all matches      | 0.154 (39%) | 0.86 | 0.058 (24%) | 1.14 |
| NL-off, all matches     | 0.139 (37%) | 0.85 | 0.051 (23%) | 1.13 |
| high-consistency subset | 0.108 (33%) | 0.90 | 0.022 (15%) | 1.08 |

Figures: `pics/ql_pe_error_{data,mc}{,_nloff,_consistent}.png` (panel 1 = pred-vs-meas with the
`meas = norm·pred` line; panel 2 = local `a` vs pred; panel 3 = `Y=(pred−meas)²−meas` vs pred²;
panel 4 = pull).

### Findings

1. **The non-linearity barely matters here.** NL only rescales per-PMT predictions above the
   700 PE knee, which are rare (~1.5% of summed pred), so `a` moves only ~1 pt (data 39→37%, MC
   24→23%) and the plots are near-identical. NL on/off is not what drives the result.

2. **The per-PMT error is ~30% for data, tighter for MC.** On clean matches data sits at ~33%
   (pull width at 0.09 ≈ 0.94 — 30% is right, slightly generous); MC is ~15% (pull width at 0.09
   ≈ 0.64 — 30% badly over-covers). MC light patterns track the prediction far better, as
   expected for truth-consistent simulation. **Conclusion: keeping `m_pe_err_frac = 0.3` is
   reasonable — conservative for MC, about right for data.** The naïve full-sample 39% (data) is
   an upper bound inflated by a coherent normalization offset (below) plus a tail of weak matches,
   neither of which is the per-PMT statistical error.

3. **The data/MC normalization flips sign — the headline.** The same photon model
   **over**-predicts data light (median meas/pred ≈ 0.86) but **under**-predicts MC
   (≈ 1.14): per unit predicted light MC is ~33% brighter than data (1.14/0.86). The matcher's
   per-bundle `strength` (LASSO coefficient) absorbs this internally — `pred_pe` in the dump is
   the **raw** prediction, not scaled by `strength` — so matching is unaffected, but it flags a
   **light-yield / `QtoL` (or charge-scale) data-vs-MC tuning question**, separate from the
   per-PMT error knob. (This is the aggregate normalization; for the bright-end-only trend see
   `11_ql-light-compare.md` and the non-linearity section of `sbnd-opdetsim-chain.md`.)

## Reproduce

```
cd sbnd_xin
python3 scripts/analysis/ql/ql_pe_error.py            # data + mc -> pics/ql_pe_error_*.png
```

Requires the NL dumps in `work/ql_nl_study/` (regenerate with
`PMT_NL=true CALIB_SUFFIX=.nl ./run_ql_evt.sh <mode> -calib all`, then copy the
`calib-evt*.nl.json` out before any linear rerun, since `run_ql_evt.sh` wipes `work/ql_evt<ID>/`).

## The data recipe (implemented)

From the above the data needs two corrections (sim is left alone — it under-predicts and is
internally clean). Both live in `cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet`:

1. **DATA prediction scale `QtoL = 0.86`** (sim `QtoL = 1.0`). Data over-predicts ~16%; since
   `pred = q·QtoL·vis·eff`, lowering the data `QtoL` is exactly a prediction scale. Toggle by
   `data_qtol = 1.0`.
2. **PE-dependent error `σ² = meas + max(5 PE, 0.25·pred)²`** (config `pe_err_floor=5`,
   `pe_err_frac=0.25`, `pe_err_knee=20`) for **both** data and sim — a constant 5 PE floor below
   a 20 PE knee, 25% fractional above, so the fractional error is large at low PE and ~25% at
   high PE. Conservative for sim, kept consistent so cuts derived on data carry over.
   `pe_err_on_pred=true` makes the **bundle χ²** compute this error from the *predicted* PE
   (`TimingTPCBundle::examine_bundle`); the **LASSO** keeps the per-flash measured-based weight
   (its measured vector is shared across candidate bundles, so the weight there must be per-flash).

Derivation/verification: `ql_derive*.py` (the σ vs PE fit) and the recipe figure
`pics/ql_pe_error_data_recipe.png` (pull flattens to ≈1; median χ²/ndf in the python fit ≈ 1).

## Data vs MC after the recipe

`scripts/analysis/ql/ql_recipe_compare.py` regenerates the calib dumps with the recipe (now the SBND default) and
compares the updated χ²/ks for the hand-scans (`pics/ql_recipe_data_vs_mc.png`):

| metric (hand-scan kept matches) | DATA | MC |
|---|---|---|
| median χ²/ndf (was ~5 with flat 30%) | **1.72** | **1.04** |
| median ks (shape; ~unchanged by recipe) | 0.077 | 0.026 |
| median meas/pred per flash | **1.01** | 1.17 |

- **χ²/ndf collapses from ~5 to ~1–2** for both modes — the PE-error model calibrates the χ².
- **Data normalization is fixed** (meas/pred 0.86 → 1.01); MC stays 1.17 (unscaled, under-predicts).
- **ks barely moves** — it is shape-normalized, so scaling `pred` leaves it invariant (only small
  group-composition shifts touch it). So "updated ks" looks ~unchanged by design, not a bug.
- MC matches are tighter (lower χ²/ndf and ks) — sim is internally cleaner; the data-tuned error
  is conservative for it.

These updated χ²/ks (+ the hand-scan labels) are the inputs for further QLMatching cut tuning.

## Reproduce

```
cd sbnd_xin
python3 scripts/analysis/ql/ql_pe_error.py            # per-PMT error study (NL on/off, consistent) -> pics/ql_pe_error_*.png
python3 scripts/analysis/ql/ql_recipe_compare.py      # data vs MC after the recipe -> pics/ql_recipe_data_vs_mc.png
```

Regenerate recipe calib dumps (recipe is the SBND default): `PMT_NL=true ./run_ql_evt.sh <mode>
-calib all`, then copy `work/ql_evt*/calib-evt*.json` into `work/ql_recipe/`. (`run_ql_evt.sh`
wipes `work/ql_evt<ID>/` each run, so copy out before switching.) The earlier NL on/off study
dumps live in `work/ql_nl_study/`.

Note: changing the error/scale changes the matcher's merge/selection, so a hand-scan
`(flash_gid, main_cluster)` key can be relabeled within its flash group (one data case, evt1720:
the main/associate flipped between two co-grouped clusters — same physical match). Clusters and
flashes themselves are upstream and unchanged, so the hand-scan-keyed analysis stays the anchor.

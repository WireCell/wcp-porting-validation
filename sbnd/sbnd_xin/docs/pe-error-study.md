# Per-PMT light error and the data/MC normalization (hand-scan PE-error study)

## Purpose

The SBND charge–light matcher (`QLMatching`) assigns each PMT a light error of **30%**
(`m_pe_err_frac = 0.3`, `match/inc/WireCellMatch/QLMatching.h`), applied as
`PE_err = (meas<1) ? 0.3 : 0.3·meas` (`match/src/Opflash.cxx`) and entering the chi² through
`σ² = meas + PE_err²`. This study **measures** that error from the trusted (flash ↔ cluster)
matches that were hand-scanned in the `ql_scan` viewer (10 data + 10 MC events), rather than
assuming it.

`sbnd_xin/ql_pe_error.py` produces the plots; this note records the method and findings.

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
   `ql-light-compare.md` and the non-linearity section of `sbnd-opdetsim-chain.md`.)

## Reproduce

```
cd sbnd_xin
python3 ql_pe_error.py            # data + mc -> pics/ql_pe_error_*.png
```

Requires the NL dumps in `work/ql_nl_study/` (regenerate with
`PMT_NL=true CALIB_SUFFIX=.nl ./run_ql_evt.sh <mode> -calib all`, then copy the
`calib-evt*.nl.json` out before any linear rerun, since `run_ql_evt.sh` wipes `work/ql_evt<ID>/`).

## Possible follow-up

Normalize `pred` to `meas` per flash (mimicking the matcher's `strength`) to remove the
light-yield offset and isolate the pure per-PMT *shape* scatter — the cleanest read on the 30%,
and the natural way to quantify the data/MC normalization mismatch on its own.

# PDHD Q/L light-model calibration (λ + normalization)

Offline tools that tune the PDHD semi-analytical optical model from two-boundary
(anode↔cathode) crosser anchors. Current calibration uses **run 29107** (the good
data run); see `../docs/ql-light-normalization-study.md` for the writeup.

The prediction depends on the VUV absorption length λ only through `exp(−d/λ)`, and
`vuv_eff`/`QtoL` are linear scales, so the full per-PMT pattern is recomputed
offline from the `-calib` dumps' charge points + the model JSON — no chain re-run
needed for the λ/eff sweep.

- **`repredict.py`** — faithful Python port of `SemiAnalyticalModel::VUVVisibility`
  (rectangle solid angle + Gaisser–Hillas + border corrections + `exp(−d/λ)`).
  Validated to machine precision against the C++ `pred_pe` (per-channel ratio
  1.0000 to ~2e-12). `predict(points, geom, flash_x_offset, λ)`.
- **`fit.py`** — selects clean two-boundary anchors from `work/029107_*/calib-*.json`,
  parallel-sweeps λ, and reports the **matcher's own KS** (an exact port of
  `TimingTPCBundle::calc_ks_test`, validated to the dumped `ks_dis`), N90,
  dark-fraction, integral- and direct-PMT-scale per λ. Writes `sweep_rows_29107.json`.
- **`after_metrics.py`** — real-C++ aggregate metrics from the reprocessed dumps
  (after applying the tuned λ + vuv_eff and re-running `run_clus_evt.sh -calib`).
  Pass a dump dir to compare BEFORE vs AFTER.
- **`plot_norm.py`** — before/after figures (`../pics/ql_norm_*.png`) from the
  λ=100 (backed-up) and λ=300 (reprocessed) dumps.
- **`fit_labels_multi.py`** — 4-event hand-scan label retune (evts 983/991/999/1007):
  `vuv_eff`, APA0 `measured_pe_scale`, `pe_err_frac` from the flag-clean labels.
- **`fit_perchannel_scale.py`** — per-channel (per-PD) gain calibration: builds the
  **block × type** (FBK/HPK) `measured_pe_scale` array + tight-outlier overrides from the
  54 cleanest labels, holds the +x integral (⇒ `vuv_eff` fixed), and refits `pe_err_frac`.
  Writes `perchannel_scale.json` + the jsonnet snippet. `--drop-xb` for the robustness check.
- **`validate_perchannel.py`** — real-C++ validation of the per-channel retune: closure
  (+x integral held, APA0 FBK/HPK + outlier channels → ~1.0) and the GT-preservation
  winner-diff (accept preserved, no new false positives) against the pre-retune dumps.
- **`validate_chain.py`** — round-by-round chain-tuning validator (enabling the remaining
  SBND features: `highconsist_ladder`, `chi2_pmt_excess` re-scale, `empty_rescue`, the
  `lasso_boundary_weight` sweep). Winner-diff between a BASELINE and a NEW dump set,
  **bucketed by the user's 3 hand-scan rules**: xTPC accepts are a HARD gate (must not
  regress), non-xTPC accepts are soft, hand-`rejected_auto` re-selection should fall, and
  auto-matches absent from the hand scan are neutral (flagged only when they steal a GT
  flash). `validate_chain.py BASELINE_DIR NEW_DIR`; the frac=0.4 reference lives in
  `baseline_frac04/` (gitignored — durable on disk, regenerate by reverting the round's
  jsonnet + reprocessing).

Tuned values applied (run 29107, 37 clean two-boundary crossers): `vuv_absorption_length`
100→300 cm (`wire-cell-data/pdhd/photodet/semi-analytical-pdhd.json`), `vuv_eff`
0.023→0.0145 (`cfg/pgrapher/experiment/pdhd/qlmatching.jsonnet`). Supersedes the
earlier 27305 tuning (λ=100, eff=0.023), which had only 5–7 anchors and
over-concentrated the model.

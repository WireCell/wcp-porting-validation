# PDHD Q/L light-model calibration (λ + normalization)

Offline tools that tuned the PDHD semi-analytical optical model from run-27305
crosser anchors. See `../docs/ql-light-normalization-study.md` for the writeup.

The prediction depends on the VUV absorption length λ only through `exp(−d/λ)`, so
the full per-PMT pattern is recomputed offline from the `-calib` dumps' charge
points + the model JSON — no chain re-run needed for the sweep.

- **`repredict.py`** — faithful Python port of `SemiAnalyticalModel::VUVVisibility`
  (rectangle solid angle + Gaisser–Hillas + border corrections + `exp(−d/λ)`).
  Validated to machine precision against the C++ `pred_pe` dumped at λ=2000
  (per-channel ratio 1.0000). `predict(points, geom, flash_x_offset, λ)`.
- **`fit.py`** — selects clean crosser anchors from `work/027305_*/calib-*.json`
  and sweeps λ, reporting KS / N90 / direct-PMT scale per anchor.
- **`after_metrics.py`** — aggregate study metrics on the reprocessed dumps
  (run after applying the tuned λ + vuv_eff and re-running `run_clus_evt.sh -calib`).

Tuned values applied: `vuv_absorption_length` 2000→100 cm
(`wire-cell-data/pdhd/photodet/semi-analytical-pdhd.json`), `vuv_eff` 0.03→0.023
(`cfg/pgrapher/experiment/pdhd/qlmatching.jsonnet`). Provisional — pinned by ~1–2
clean crossers.

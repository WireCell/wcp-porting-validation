# `dqdx_rr_sample/` — the curated stopping-track dQ/dx-vs-residual-range sample

13 stopping tracks (12 muon-like, 1 proton-like) selected out of every STM-fit
block in the 30-event MCP2025C reco1 `d55ton` set, kept only where the
dQ/dx-vs-residual-range **shape** is highly consistent with an SBND
stopping-particle expectation curve.

Written up in [`../docs/55_dqdx-vs-rr-three-bundles.md`](../docs/55_dqdx-vs-rr-three-bundles.md)
§§6–8. The three bundles of §§0–5 of that doc are all in here
(289343 blk 90, 285999 blk 220, 286065 blk 30).

**Doc 55 §11 adds a proton *population*** — the 12 usable protons the owner
hand-identified in the doc-62 :5012 scan, read out of `work-mcp1kall-d59k`
(`proton_*.tsv`, `sample_points_p12.tsv`). They are selected by the owner's eye,
not by the cuts below, and their point files are separate: `sample_points.tsv`
and the `sample_*.png` / `recomb_fit.png` / `muon_proton_vs_models.png` figures
stay the 12-muon/1-proton record that §§6–10 of the doc are computed from.

## Files

| file | what |
|---|---|
| `collect_dqdx_rr_sample.py` | the selector: sweeps the ROOT files, applies the cuts, writes the two `.tsv` and the overlay PNG |
| `sample_index.tsv` | one row per selected track: work root, event, block, cuts' values, free scale against each hypothesis |
| `sample_points.tsv` | one row per fitted point: `rr`, `dqdx` (e/cm), `dx`, `x/y/z`, drift distance and drift time |
| `sample_overlay.png` | all 13 tracks' binned medians on the SBND muon/proton curves, and their raw ratio to their own curve |
| `fit_recombination.py` | the recombination-model study: measured dQ/dx vs dE/dx, a 12-family model zoo (Modified Box and Birks variants) fitted jointly and per particle, the A–B degeneracy, the electron-lifetime check.  `--plane rr` fits in the residual-range plane instead, `--zoo` runs the whole zoo |
| `recomb_fit.png` | the recombination curve and the residuals of the candidate models |
| `plot_muon_proton_models.py` | the sample-average muon and the proton in the residual-range plane against the current expectation, the free-B fit and the free-power fit, with every normalisation printed on the figure |
| `muon_proton_vs_models.png` | that figure (doc 55 §7f, §7g) |
| `collect_proton_sample.py` | **doc 55 §11**: the 13 protons the OWNER hand-identified in the doc-62 scan, read out of `work-mcp1kall-d59k`.  No cut is applied — the identification is the owner's — and the doc-55 selector quantities are written out per track for the record.  Checks that d59k and d55ton agree bit-for-bit on the track they share before it will merge the two |
| `proton_index.tsv` | one row per owner-identified proton, with `use` and the named reason when it is `no` |
| `proton_points.tsv` | per-point, same columns as `sample_points.tsv` |
| `sample_points_p12.tsv` | the merged sample the §11 fits run on: the 12 `d55ton` muons + the 12 usable `d59k` protons, each track counted once |
| `proton_sample_p12.png` | the 12 protons on the shipped Box and free-power proton curves (doc 55 §11.2) |
| `proton_model_check.py` | **fits nothing**: evaluates the *committed* free-power parameters on the enlarged sample, plus proton/muon at matched dE/dx and the per-track drift trend (doc 55 §11.3, §11.5, §11.6) |
| `proton_vs_frozen_model_p12.png` | that figure — the population against the model fitted without it |
| `muon_proton_vs_models_p12.png` | the §7f/§7g figure regenerated on the enlarged sample (doc 55 §11.4) |
| `make_ref_tables.py` | builds all FIVE reference tables (muon/electron/pion/kaon/proton) under the free-power model on `convert_field.C`'s grid, regression-checks the Box versions against the shipped ROOT tables, and writes both sets into `../nusel_display/stm_ref_dqdx.json` (doc 55 §10) |
| `ref_tables_free_power.png` | the five tables, current vs free power, with a ratio row (doc 55 §10.2) |

`sample_points.tsv` is the reusable product — it needs no ROOT and no toolkit,
just the two `dE/dx`-vs-residual-range graphs in
`energy_loss/pion_travel/stopping.root` if you want to work in the dE/dx plane.

## Regenerate

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 dqdx_rr_sample/collect_dqdx_rr_sample.py --plot dqdx_rr_sample/sample_overlay.png
python3 dqdx_rr_sample/collect_dqdx_rr_sample.py --verbose   # why each block was dropped
python3 dqdx_rr_sample/fit_recombination.py -o dqdx_rr_sample/recomb_fit.png
python3 dqdx_rr_sample/fit_recombination.py --rr-max 60       # robustness variant
python3 dqdx_rr_sample/fit_recombination.py --zoo --min-in-bin 3   # model zoo, dE/dx plane
python3 dqdx_rr_sample/fit_recombination.py --plane rr --zoo       # model zoo, rr plane
python3 dqdx_rr_sample/plot_muon_proton_models.py \
    -o dqdx_rr_sample/muon_proton_vs_models.png
python3 dqdx_rr_sample/make_ref_tables.py --dry-run             # print, write nothing
python3 dqdx_rr_sample/make_ref_tables.py \
    --json nusel_display/stm_ref_dqdx.json \
    -o dqdx_rr_sample/ref_tables_free_power.png
```

`plot_muon_proton_models.py` verifies its "current expectation" curve against
`stopping_ave_dQ_dx_sbnd.root` (max relative deviation 8e-4) before it draws
anything, and imports the best-fit parameters from `fit_recombination.py` rather
than hard-coding them, so the two scripts cannot drift apart.
`make_ref_tables.py` does the same for all five particles and *refuses to write*
if any of the five misses, so a table set can never be published against a recipe
that no longer reproduces the shipped one.

Nothing is re-run from the reconstruction; every input `tracking-stm.root`
already existed in the `d55ton` arms.

## The one thing to read before using these numbers

This is **uncalibrated data** — no gain calibration and no electron-lifetime
correction is applied anywhere upstream (doc 42 §0, doc 48 §8 item 3). Every
absolute dQ/dx here carries one unknown common factor. What the sample is good
for is *relative* statements: track against track, and particle against particle
at matched dE/dx. Doc 55 §8 uses it that way and measures the lifetime
(τ of order 10 ms) out of the sample itself.

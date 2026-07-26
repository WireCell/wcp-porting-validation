# `dqdx_rr_sample/` — the curated stopping-track dQ/dx-vs-residual-range sample

13 stopping tracks (12 muon-like, 1 proton-like) selected out of every STM-fit
block in the 30-event MCP2025C reco1 `d55ton` set, kept only where the
dQ/dx-vs-residual-range **shape** is highly consistent with an SBND
stopping-particle expectation curve.

Written up in [`../docs/55_dqdx-vs-rr-three-bundles.md`](../docs/55_dqdx-vs-rr-three-bundles.md)
§§6–8. The three bundles of §§0–5 of that doc are all in here
(289343 blk 90, 285999 blk 220, 286065 blk 30).

## Files

| file | what |
|---|---|
| `collect_dqdx_rr_sample.py` | the selector: sweeps the ROOT files, applies the cuts, writes the two `.tsv` and the overlay PNG |
| `sample_index.tsv` | one row per selected track: work root, event, block, cuts' values, free scale against each hypothesis |
| `sample_points.tsv` | one row per fitted point: `rr`, `dqdx` (e/cm), `dx`, `x/y/z`, drift distance and drift time |
| `sample_overlay.png` | all 13 tracks' binned medians on the SBND muon/proton curves, and their raw ratio to their own curve |
| `fit_recombination.py` | the recombination-model study: measured dQ/dx vs dE/dx, Modified Box and Birks fits, the A–B degeneracy, the electron-lifetime check |
| `recomb_fit.png` | the recombination curve and the residuals of the candidate models |
| `plot_muon_proton_models.py` | the sample-average muon and the proton in the residual-range plane against the current expectation and the best fit, both normalisations printed on the figure |
| `muon_proton_vs_models.png` | that figure (doc 55 §7f) |

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
python3 dqdx_rr_sample/plot_muon_proton_models.py \
    -o dqdx_rr_sample/muon_proton_vs_models.png
```

`plot_muon_proton_models.py` verifies its "current expectation" curve against
`stopping_ave_dQ_dx_sbnd.root` (max relative deviation 8e-4) before it draws
anything, and imports the best-fit parameters from `fit_recombination.py` rather
than hard-coding them, so the two scripts cannot drift apart.

Nothing is re-run from the reconstruction; every input `tracking-stm.root`
already existed in the `d55ton` arms.

## The one thing to read before using these numbers

This is **uncalibrated data** — no gain calibration and no electron-lifetime
correction is applied anywhere upstream (doc 42 §0, doc 48 §8 item 3). Every
absolute dQ/dx here carries one unknown common factor. What the sample is good
for is *relative* statements: track against track, and particle against particle
at matched dE/dx. Doc 55 §8 uses it that way and measures the lifetime
(τ of order 10 ms) out of the sample itself.

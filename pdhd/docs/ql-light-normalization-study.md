# PDHD Q/L light: predicted-vs-measured pattern & normalization study (run 29107)

**Run 29107, all 30 real events.** Goal: use well-constrained cosmic tracks that
cross **two** TPC boundaries (anode→cathode) to (1) confirm the predicted optical
pattern has the same **shape** as the measured flash, and (2) deduce the
**normalization** of the predicted light yield.

> **Supersedes the earlier run-27305 study.** That run yielded only 5–7 clean
> anchors and was not a clean data run, so its tuning (`vuv_absorption_length`
> λ = 100 cm, `vuv_eff` = 0.023) was biased — it **over-concentrated** the model.
> 29107 yields a properly statistical sample (37 clean two-boundary crossers) and
> moves the tuning to **λ = 300 cm, `vuv_eff` = 0.0145**. The 27305 result is
> retired; see "What changed from 27305" below.

**Headline:** on this good run the measured two-boundary crossers are
**more diffuse** than the 27305 anchors implied. At the old λ = 100 the model is
**too concentrated** — its light fills 90 % of the pattern in ~0.6× as many PMTs
as the data (`N90` ratio pred/meas = 0.61), with the matcher KS shape distance
median 0.099. Widening the effective absorption length to **λ ≈ 300 cm** matches
the data concentration (`N90` ratio → 0.95) and improves the matcher KS to 0.077
(−23 %); the direct-PMT normalization then calibrates with `vuv_eff` 0.023 → 0.0145.
Unlike 27305, **light-on-dark-PMTs is a non-issue here** (≤ 2 % at every λ),
because these extended crossers light most of the wall.

## Inputs & method

- Calib dumps from the clustering chain with `-calib`:
  `run_clus_evt.sh -calib 29107 <idx>` →
  `work/029107_<idx>/calib-evt<art>-group{02,13}.json`
  (group02 = APAs 0+2, drift −x; group13 = APAs 1+3, drift +x;
  art = 983 + 8·idx). Each bundle carries `pred_pe[160]` and `ks_dis`; the matched
  flash carries `pe[160]`.
- **Crosser selection — two-boundary primary.** Bundles flagged `two_boundary`
  (track crosses anode **and** cathode), `at_x_boundary`, `ndf ≥ 30`, brightest
  valid PMT > 300 PE, `measTot > 3000` PE, `ks_dis < 0.40`, **deduplicated per
  flash** (keep the lowest-`ks` bundle, so one bright flash's cluster fragments
  count once). This gives **37** clean two-boundary anchors (28 with `ks < 0.20`,
  the primary shape set). For context the broader `at_x_boundary` population is
  441 raw → 190 clean, but it is contaminated (direct-PMT scale tail to 5×), so
  the two-boundary set is used throughout.
- **Valid PMTs:** `active & ¬auto_masked & ch ∉ static_mask`
  (static = {3,86,87,97,107,116,117} ∪ {120…159}; see `qlmatching.jsonnet`).
- **Offline re-predictor, validated to machine precision.** λ enters the model
  only through `exp(−d/λ)` (`SemiAnalyticalModel.cxx`) and `vuv_eff`/`QtoL` are
  linear scales, so the whole per-PMT pattern is recomputable at any (λ, eff)
  from the dumped charge points + model JSON with **no chain re-run**
  (`ql_light_calib/repredict.py`). At λ = 100 it reproduces the dumped C++
  `pred_pe` to **2 × 10⁻¹²** per channel.
- **The shape objective is the matcher's own KS.** `ql_light_calib/fit.py` ports
  `TimingTPCBundle::calc_ks_test` exactly — an **index-order** cumulative CDF over
  **all 160 channels**, with the prediction zeroed on the static|auto masked
  channels (the C++ `opdet_mask`). This Python KS reproduces the dumped C++
  `ks_dis` to **machine precision** (max |Δ| = 0, 28/28 anchors) at λ = 100, so
  the λ sweep below is the matcher's metric, not a proxy. (A naive sorted-by-PE
  KS over only the valid subset — the obvious wrong port — correlates just 0.63
  with the real `ks_dis` and would have mis-located the optimum toward λ ≈ 250.)

## λ + normalization sweep (offline, two-boundary `ks < 0.20`, 28 anchors)

| λ (cm) | matcher KS | `N90` ratio (pred/meas) | dark % | integral scale | → `vuv_eff` |
|---:|---:|---:|---:|---:|---:|
| 150 | 0.089 | 0.74 | 1 | 0.73 | 0.0173 |
| 200 | 0.081 | 0.84 | 1 | 0.64 | 0.0157 |
| 250 | 0.077 | 0.90 | 1 | 0.59 | 0.0149 |
| **300** | **0.076** | **0.96** | 2 | 0.56 | **0.0145** |
| 350 | 0.076 | 1.00 | 2 | 0.53 | 0.0142 |
| 400 | 0.077 | 1.04 | 2 | 0.51 | 0.0139 |

- **λ = 300 cm**, chosen as the **centre of the flat KS valley** (250–400 all
  within 0.076–0.077). N90-match and KS-min agree here; 300 vs 350 is genuinely
  under-determined (KS identical, `N90` 0.96 vs 1.00) — quoted bracket **[250, 350]**.
- **`vuv_eff` = 0.0145**, from the median direct-PMT (top-3) scale **0.63** at
  λ = 300: `vuv_eff = 0.023 × 0.63 ≈ 0.0145`. The light yield enters linearly
  (`QtoL × vuv_eff`); this drives the direct-PMT scale to 1.0.

## Result — real-C++ confirmation

The configs were set to (λ = 300, `vuv_eff` = 0.0145) and the **18 events holding
the clean two-boundary anchors were reprocessed** end-to-end (`run_clus_evt.sh
-calib`). The real-C++ after-table reproduces the offline prediction to within
anchor-selection noise — and `direct-scale = 1.00` exactly confirms the `vuv_eff`
edit took effect (offline at the old eff was 0.63; 0.63 × 1.586 = 1.00), while
`KS 0.099 → 0.077` confirms the λ edit took effect.

| metric (median, two-boundary `ks<0.20`) | before λ=100, eff=0.023 | after λ=300, eff=0.0145 (offline) | after (real C++) |
|---|---:|---:|---:|
| matcher KS | 0.099 | 0.076 | **0.077** |
| `N90` ratio (pred/meas) | 0.61 | 0.96 | **0.95** |
| direct-PMT scale | 0.90 | 1.00 | **1.00** |
| integral scale | 0.96 | 0.88 | **0.91** |
| light on dark PMTs | 0 % | 2 % | **1 %** |

![concentration](../pics/ql_norm_concentration.png)

*Cumulative light vs PMT rank, 30 matched two-boundary crossers. Before (red,
λ=100) rises too steeply — over-concentrated; after (green, λ=300) tracks the
measured (black) diffuseness. The dotted line is the 90 % (`N90`) level.*

![per-PMT PE](../pics/ql_norm_perpmt.png)

*Per-PMT PE (channels sorted by measured, symlog), measured / before λ=100 /
after λ=300, for the brightest crossers. After (green) widens onto the
secondary PMTs the data actually lit and lowers the over-bright peaks.*

![crosser patterns](../pics/ql_norm_crosser_patterns.png)

*2-D PMT maps (y vs z), colour = normalized PE. Right column (after) spreads the
predicted light toward the measured pattern (left).*

## What changed from 27305, and why a 3× λ shift is expected

The move λ 100 → 300 and `vuv_eff` 0.023 → 0.0145 is large but coherent. λ here is
an **effective spread parameter**, not the physical 128 nm LAr absorption length
(~85 cm) — by convention the geometric fall-off is folded into the Gaisser–Hillas
fit, and λ absorbs residual model error. Four reasons the better run lands far
from the old one:

1. **Sample richness + selection.** 27305 had 5–7 anchors, mostly *concentrated*
   short clusters; 29107 has 37 genuine **two-boundary anode-cathode crossers** —
   extended, full-drift line sources whose measured patterns are intrinsically
   diffuse (`N90` up to 54). An effective λ tuned on extended sources is naturally
   larger than one tuned on concentrated ones.
2. **Flash reconstruction changed between the two studies** (ADC-saturation veto,
   per-PD `min_fired_pe`, `min_fired_pds`/`min_total_pe` quality cuts), which
   alters the measured per-PMT PE pattern and can shift λ on its own.
3. **The dark-PMT failure mode that justified 27305's λ=100 does not exist here.**
   On 27305's concentrated anchors a diffuse model spilled ~47 % of light onto
   measured-zero PMTs; on 29107's extended crossers dark-fraction is ≤ 2 % at
   *every* λ, so it cannot pull λ down.
4. **The optimum is not an artefact of selecting on the old λ.** Anchors were cut
   on the λ=100 `ks_dis`; nonetheless the optimum lands at 300, and the looser
   `at_x` set prefers an even *larger* λ (KS-min ~500). If anything the selection
   biases the result *conservatively* toward smaller λ.

## Caveats / next steps

- **Spread fixed for the population, residual on long multi-track blobs.** The
  per-anchor N90-match λ is bimodal: most anchors want 150–500 cm but ~¼ (the
  longest multi-track blobs) never concentrate enough and pin at the 2000 ceiling.
  A single global λ cannot capture this; the next-order correction is the angular
  **Gaisser–Hillas** terms (or a voxel photon library).
- **Integral residual, reported not re-optimised.** After the direct-PMT
  normalization the integrated prediction is ~1.1× the measured (integral scale
  ~0.9). Top-3 direct-PMT is the right normalization handle (the integral is not
  clean until the spread is perfect); the residual is recorded, not fitted away.
- **Scope: matching effect not yet validated on the full population.** `vuv_eff`
  0.023 → 0.0145 is a ~37 % normalization change affecting *every* PDHD event's
  matching, but only the 18 anchor events were reprocessed (the prediction math is
  confirmed; the full-population matching outcome — assignments, purity — is
  **unverified pending a full 30-event reprocess**).
- **More clean crossers** (other good runs, both drift sides) would tighten λ and
  let it be checked per side.

## Reproduce

```bash
# 1. offline lambda+eff sweep on the existing calib dumps (no chain re-run):
cd pdhd/ql_light_calib && python3 fit.py          # -> sweep_rows_29107.json + the table above
# 2. set the tuning: wire-cell-data .../semi-analytical-pdhd.json vuv_absorption_length=300,
#    cfg/.../pdhd/qlmatching.jsonnet vuv_eff=0.0145
# 3. reprocess the anchor events and read the real-C++ after-table:
./run_clus_evt.sh -calib 29107 <idx>              # for each anchor event idx
python3 after_metrics.py                          # real-C++ medians from the dumps
# 4. before/after figures (BEFORE = backed-up lambda=100 dumps, AFTER = reprocessed):
python3 plot_norm.py                              # -> ../pics/ql_norm_*.png
```

Per-bundle fields used: `two_boundary`, `at_x_boundary`, `main_cluster`,
`flash_gid`, `ks_dis`, `ndf`, `pred_pe[160]`; flash `pe[160]`; opdet
`{x,y,z,active,auto_masked}`. Scripts: `ql_light_calib/` (`repredict.py`
validated re-predictor, `fit.py` matcher-KS λ sweep, `after_metrics.py`
real-C++ metrics, `plot_norm.py` before/after figures). See
`qlmatching-chain.md` for the matching chain and tunables.

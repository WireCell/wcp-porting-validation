# 47 — The STM tagger's Bragg-peak reference, and what SBND has to retune

**Status: inventory + analysis only. No code, config, or data was changed.**
Nothing here is a tuning decision; the numbers that would change a physics
verdict are presented as readings with the measurement that settles them
(escalation rule 7), exactly as doc 41 did for the MIP plateau.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
# every number in sections 3-5 (closed-form recombination arithmetic, no fitting):
python3 sbnd_xin/stm_dqdx_reference.py
# the reference curve's provenance -- the 5 TGraphs the jsonnet tables came from:
cd prototype_base/input_data_files
root -l -b -q '/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/dump_stopping_dqdx.C("stopping_ave_dQ_dx.root")'
root -l -b -q '/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/dump_stopping_dqdx.C("stopping_ave_dQ_dx_v2.root")'
# the code paths quoted below:
sed -n '1500,1600p' clus/src/TaggerCheckSTM.cxx          # eval_stm_core, the KS comparison
sed -n '216,240p'   util/src/KSTest.cxx                  # kslike_compare
grep -n '50000\|50e3' clus/src/TaggerCheckSTM.cxx        # 35 matching lines
```

Empirical numbers quoted from earlier docs: 41 §MIP-plateau (data), 42 §7 /
44 / 46 (MC evt18 block 150). The dQ/dx reference tables live in
`wcp-porting-img/sbnd/particle_dataset.jsonnet`; the smearing constants in
`toolkit/sbnd_xin/sbnd_track_fitting.json`.

## 0. Short answers to the three questions

1. **Compared to what?** To a tabulated *stopping-muon* dQ/dx-vs-residual-range
   curve, `MuonDeDx` — 60 samples at 0.5, 1.5, … 59.5 cm, linearly
   interpolated, clamped flat outside — and, as the null hypothesis, to a
   **flat 50 000 e/cm** MIP line. The discriminant is a shape-only
   Kolmogorov-like distance (`kslike_compare`) plus a separate normalization
   `ratio`. `clus/src/TaggerCheckSTM.cxx:1541-1556`.

2. **Configurable?** *Half.* The five dQ/dx curves and five range curves are
   jsonnet (`LinterpFunction` components aggregated by a `ParticleDataSet`),
   so the reference curve **is** replaceable without touching C++. Everything
   else that sets the physics scale is a C++ literal: the flat `50e3` (35
   lines), every KS/ratio threshold, every window length, every absolute
   charge cut. `TaggerCheckSTM` exposes only seven knobs
   (`grouping`, `require_in_scope`, `shorted_y_w_range`, `save_stm_fit`,
   `trackfitting_config_file`, `particle_dataset`, `recombination_model`) —
   none of them a dQ/dx scale.

3. **Diffusion / smearing.** `DL = 6.2` and `DT = 9.8` cm²/s, entering as
   σ = √(2·D·t_drift) with `t_drift` floored at `min_drift_time = 50 µs`,
   added in quadrature to the SP software-filter widths, then divided by tick
   width / wire pitch. The filter widths **were** re-derived for SBND from
   `sbnd/sp-filters.jsonnet` (§6 below); `DT` was **not** — it is still
   uBooNE's value.

The headline finding is §3: the reference curve is anchored to uBooNE's
**0.273 kV/cm** drift field, SBND runs at **0.5 kV/cm**, and the required
correction is not a single factor — it is residual-range dependent, so it
distorts the Bragg *shape* the test keys on, not just its normalization.

## 1. What the Bragg test actually does

`check_stm_conditions` fits the track, finds the candidate stopping end and
the first kink, then calls `eval_stm_core` (up to eight times per track with
different windows). Inside (`TaggerCheckSTM.cxx:1436-1596`):

1. build `dQ_dx[i] = fit.dQ / (fit.dx/cm)` — electrons per cm — along the
   fitted path, and `L[i]` its arc length;
2. locate the peak with a 5-point moving average inside `peak_range` of the
   end; call it `end_L`;
3. collect the points with `0 < end_L − L[i] < com_range` into `vec_y`, with
   `vec_x` = residual range from the peak;
4. compare against two references:

```cpp
ref_muon[i] = particle_data()->get_dEdx_function("muon")
                  ->scalar_function((vec_x[i] + offset_length) / units::cm);
ref_flat[i] = 50e3;
double ks1 = WireCell::kslike_compare(test_data, ref_muon);   // stopping hypothesis
double ks2 = WireCell::kslike_compare(test_data, ref_flat);   // through-going hypothesis
double ratio1 = sum(ref_muon) / sum(test_data);
double ratio2 = sum(ref_flat) / sum(test_data);
```

5. accept "stopping" when the muon hypothesis beats the flat one, subject to a
   ladder of cuts on `ks1, ks2, ratio1, ratio2` and on the *residual* past the
   peak (`res_length`, `ave_res_dQ_dx`, `res_dis1`) that vetoes Michel
   electrons and protons.

**`kslike_compare` is area-normalized** (`util/src/KSTest.cxx:216-238`): it
divides both vectors by their own sum and returns the max CDF difference. So

- a wrong **absolute scale** leaves `ks1`/`ks2` untouched and moves only
  `ratio1`/`ratio2`;
- a wrong **shape** (peak-to-plateau contrast) moves `ks1` itself.

That distinction drives §4: SBND needs both, which is why no single
multiplicative knob fixes it.

Windows actually used (`TaggerCheckSTM.cxx:2341-2382`), all C++ literals:
`peak_range` ∈ {5, 20−left_L, 40−left_L, 40} cm, `offset_length` ∈ {0, 3} cm,
`com_range` ∈ {15, 35} cm. Maximum residual range ever queried is
35 + 3 = 38 cm, comfortably inside the table's 0.5–59.5 cm support, so table
coverage is **not** an issue — that question is closed.

## 2. Where the reference lives, and where it came from

| layer | file | content |
|---|---|---|
| component | `clus/src/ParticleDataSet.cxx` | maps particle name → `IScalarFunction` by type/name lookup |
| instance | `wcp-porting-img/sbnd/particle_dataset.jsonnet:697-711` | `dedx_functions: {muon: MuonDeDx, proton: …, pion, kaon, electron}` + five `range_functions` |
| the numbers | same file, lines 14-93 | five `LinterpFunction`s, `start: 0.5, step: 1.0`, 60 `values` each |
| wiring | `cfg/pgrapher/experiment/sbnd/clus.jsonnet:538-541` | `particle_dataset=wc.tn(particle_dataset)`, `recombination_model=wc.tn(sbnd_box_recomb)` |
| caller | `sbnd_xin/wct-pr-perevt.jsonnet:174-180` | `local pds = (import '../particle_dataset.jsonnet')()` |

The SBND jsonnet header calls these tables "detector-agnostic NIST/PDG". **They
are not.** They are the point-by-point contents of the prototype's
`prototype_base/input_data_files/stopping_ave_dQ_dx.root`, verified by dumping
it: the `muon` TGraph has exactly 60 points, first `(0.5, 123417)`, second
`(1.5, 92909.8)`, last `(59.5, 48879.4)` — identical to the jsonnet. The
prototype loads the same file at `pid/src/ToyFiducial.cxx:53-58`. The values
are **charge** per cm, so they carry a recombination model and a drift field.

Two notes found while checking provenance, both reported and **not** acted on
per the "mention, don't fix" rule:

- **`stopping_ave_dQ_dx_v2.root` exists and is unused.** Its muon curve has a
  visibly softer Bragg peak (`0.5 cm`: 115 657 vs 123 417 e/cm) and the same
  plateau (48 876 vs 48 879). Every prototype reference to it is commented out
  (`pid/apps/wire-cell-prod-nue.cxx:195` and four siblings); the live path is
  v1. The toolkit therefore matches the prototype. Whether v2 was a
  resolution-smeared or re-fit iteration is not recorded.
- **The jsonnet `electron` table is not a faithful copy of v1.** v1's electron
  graph has 61 points starting at `(0, 180000)`; the jsonnet has
  `start: -0.5` with `260000, 100000, 46999.4, …`. Irrelevant to STM (which
  only reads `"muon"`), but it means the "copied verbatim" claim in the
  jsonnet header does not hold for every curve. Not touched.

No generating script survives in `prototype_base/` (only `export_dQ_dx.C`,
which *reads* the file), so the curve's construction is not documented. §3
recovers what matters anyway.

## 3. Finding: the curve is anchored to uBooNE's 0.273 kV/cm

The modified box model (`gen/src/RecombinationModels.cxx:83-100`) is

    ξ = B·(dE/dx)/(E·ρ),   R = ln(A+ξ)/ξ,   dQ/dx = R·(dE/dx)/W_i

and has the closed form **dQ/dx = ln(A+ξ)·(E·ρ)/(B·W_i)** — at fixed field the
charge depends on dE/dx only inside the logarithm.

**Test.** Take the *independent* muon **range** table in the same jsonnet
(range → kinetic energy), differentiate it to get the CSDA dE/dx at each
residual range, and push that through the box model at uBooNE's field with the
configured `A=1.0, B=0.255, ρ=1.38, W_i=23.6 eV`:

| R [cm] | table [e/cm] | dE/dx CSDA [MeV/cm] | Q(CSDA, 0.273 kV/cm) | Q/table |
|---|---|---|---|---|
| 0.5 | 123 417 | 9.245 | 124 083 | **1.005** |
| 2.5 | 82 815 | 4.823 | 90 789 | 1.096 |
| 9.5 | 62 368 | 3.050 | 70 108 | 1.124 |
| 29.5 | 51 881 | 2.346 | 59 531 | 1.147 |
| 59.5 | 48 879 | 2.162 | 56 435 | 1.155 |

At the Bragg peak the table reproduces CSDA-dE/dx-through-uBooNE-box to
**0.5 %**. The growing excess toward the plateau is the expected
*restricted*-vs-*unrestricted* difference: at high kinetic energy part of the
loss goes into δ-rays energetic enough to leave the local `dx`, so a locally
measured dQ/dx sits below CSDA; at the Bragg peak there are no such δ-rays and
the two coincide. (This is the same physics doc 46 measured directly: evt18
block 150's truth is 49.83 ke/cm restricted vs 60.59 ke/cm with secondaries,
+22 %.)

So the table is **restricted energy loss × uBooNE recombination**. A different
field would have scaled the whole curve; it did not.

**Consequence.** Invert each table point for its effective dE/dx and re-emit it
at SBND's 0.5 kV/cm:

| R [cm] | table (uB) | dE/dx eff | SBND, A=1.0/B=0.255 | ×    | SBND, A=0.93/B=0.212 | ×    |
|---|---|---|---|---|---|---|
| 0.5 | 123 417 | 9.13 | 169 218 | **1.371** | 181 761 | **1.473** |
| 2.5 | 82 815 | 4.07 | 105 233 | 1.271 | 107 492 | 1.298 |
| 9.5 | 62 368 | 2.52 | 75 545 | 1.211 | 73 615 | 1.180 |
| 29.5 | 51 881 | 1.91 | 61 148 | 1.179 | 57 360 | 1.106 |
| 59.5 | 48 879 | 1.75 | 57 141 | **1.169** | 52 859 | **1.081** |

and the flat MIP line: 50 000 e/cm at 0.273 kV/cm corresponds to
dE/dx = 1.806 MeV/cm, which at 0.5 kV/cm gives **58 631 e/cm (×1.173)** with
the configured parameters or **54 531 e/cm (×1.091)** with LArSoft's.

Two things follow, and the second is the one that matters:

- **The scale is 8–17 % low** for SBND on the plateau. Which end depends on the
  recombination parameters — see §5, this is *not* settled here.
- **The two parameter sets agree to 0.02 % at uBooNE's field** (48 871 vs
  48 879 e/cm) and diverge by 8 % at SBND's. Whichever set generated the table,
  it could not have been distinguished at 0.273 kV/cm. At 0.5 kV/cm the choice
  becomes first-order. This is precisely why the port cannot inherit it.

## 4. Why one scale factor is not enough

The rescale is monotonic in residual range: ×1.17 on the plateau, ×1.37 at the
Bragg peak (×1.08 → ×1.47 with LArSoft's parameters). Recombination saturates
less at higher field, so a higher field *sharpens* the Bragg peak in charge
space. Peak(0.5 cm)/plateau(29.5 cm) contrast:

| curve | contrast |
|---|---|
| table as shipped (uBooNE) | 2.379 |
| SBND, A=1.0 / B=0.255 | 2.767 |
| SBND, A=0.93 / B=0.212 | 3.169 |

Since `kslike_compare` is area-normalized, the **contrast is what moves `ks1`**.
The shipped curve understates SBND's true Bragg contrast by 16–33 %, so `ks1`
(distance to the stopping hypothesis) is inflated for a genuine SBND stopper
while `ks2` (distance to flat) is unaffected — the test is biased *against*
tagging stoppers, on top of the `ratio1`/`ratio2` normalization error. A single
`MIP_dQdx`-style scale knob would fix `ratio*` and leave `ks*` wrong. **The
reference curve itself has to be replaced**, which the jsonnet already allows.

## 5. What we cannot decide from here (do not pick a scale)

Three effects of comparable size are entangled, and the readings in hand do not
separate them:

| reading | value | source |
|---|---|---|
| shipped plateau reference | 48.9 (table) / 50.0 (flat) ke/cm | `particle_dataset.jsonnet`, `TaggerCheckSTM.cxx:1547` |
| recombination-scaled prediction, A=0.93/B=0.212 | **52.9** ke/cm | §3 |
| recombination-scaled prediction, A=1.0/B=0.255 | **57.1** ke/cm | §3 |
| SBND MC evt18 blk150, truth restricted (primary only) | 49.83 ke/cm | doc 46 |
| SBND MC evt18 blk150, truth incl. δ-rays | 60.59 ke/cm | doc 46 |
| SBND MC evt18 blk150, fitted | 51.63 ke/cm | doc 46 |
| SBND data, accepted-STM tracks, rr > 40 cm median | TPC0 **59.5** / TPC1 **55.8** ke/cm (p25 ≈ 50, high tail) | doc 41 |

Confounds:

1. **Recombination parameters.** The toolkit's SBND `BoxRecombination` is
   `A=1.0, B=0.255` (`clus.jsonnet:462-466`) — the uBooNE values with only the
   field changed. LArSoft's modified-box defaults are `A=0.930, B=0.212`. If
   SBND MC was simulated with the latter, the toolkit's dQ→dE inversion and any
   recombination-rescaled reference disagree with the simulation by 8 %.
   **Verify against the SBND LArSoft configuration** (`LArG4Parameters` /
   `ISCalcSeparate` `ModBoxA`/`ModBoxB`, `larproperties`); it is not in this
   tree.
2. **Restricted vs total energy loss.** The table is a restricted quantity
   (§3), but doc 46 showed the fit recovers ~17 % of δ-ray charge, so the
   measured dQ/dx sits *between* restricted (49.8) and total (60.6). The
   table's effective restriction energy is undocumented. This alone spans
   ~20 % — the same size as the field effect.
3. **No electron-lifetime correction is applied anywhere.**
   `clus/src/PRSegmentFunctions.cxx:1194-1197` states plainly that for
   multi-APA detectors "callers must pre-apply lifetime corrections to dQ
   before this function is called" — and nothing does. Doc 41's high-side tail
   and ~6 % TPC0>TPC1 asymmetry are exactly what an uncorrected attenuation
   would look like. This is a **prerequisite** to any absolute-scale claim.

**The measurement that discriminates:** the fitted-vs-true dQ/dx plateau of
long muons, binned in drift x, on SBND MC where the truth charge is known per
deposit (the doc 44/46 machinery already produces both, with δ-rays as a
separate channel). Field/recombination shifts the plateau uniformly in x;
missing lifetime tilts it with x; the restricted/total ambiguity moves
restricted and total truth apart while leaving the fit alone. One scan over
the doc 40 phase-4 sample separates all three. Until then, quoting a single
SBND MIP scale would be tuning to make a plot look right.

## 6. Smearing and diffusion — what is already SBND, what is not

`TrackFitting` predicts each fitted point's charge footprint by smearing the
trajectory with a Gaussian that must **match the software filters signal
processing already imprinted on the data**. Per plane
(`clus/src/TrackFitting.cxx:6120-6131`):

```
t_drift    = max(min_drift_time, |x − x_anode| / v_drift)
σ_L        = hypot( sqrt(2·DL·t_drift), add_sigma_L ) / tick_width
σ_T,{u,v,w}= hypot( sqrt(2·DT·t_drift), {ind,ind,col}_sigma_*_T ) / pitch_{u,v,w}
```

Geometry (pitch, tick, wire angles, `v_drift`, time offset) comes from
DetectorVolumes/grouping at runtime, so it is already SBND's
(`v_drift = 1.563 mm/µs`, consistent with 0.5 kV/cm). The constants come from
`sbnd_xin/sbnd_track_fitting.json` (47 keys; C++ presets in
`TrackFitting.h:35-101` are uBooNE and must never be relied on).

| key | uBooNE | SBND (in use) | derivation | status |
|---|---|---|---|---|
| `DL` | 6.4e-7 | **6.2e-7** (6.2 cm²/s) | longitudinal diffusion | changed for SBND; source = SBND Q/L chain value |
| `DT` | 9.8e-7 | 9.8e-7 (9.8 cm²/s) | transverse diffusion | **inherited, unverified** — the correct SBND value lives in SBND's LArSoft `larproperties`, not in this tree |
| `add_sigma_L` | 1.5699937 | **2.4876** mm | 1/(2π·σ_`Gaus_wide`) × v_drift = 1/(2π×0.10 MHz) × 1.563 mm/µs | derived from SBND SP filters |
| `ind_sigma_u_T` | 0.3626937 | **0.48359** mm | [(1/√π)/`Wire_ind` 1.05] × 3 mm × 0.3 | derived |
| `ind_sigma_v_T` | 0.6044895 | **0.80599** mm | same × 0.5 | derived |
| `col_sigma_w_T` | 0.112836 | **0.09403** mm | [(1/√π)/`Wire_col` 3.60] × 3 mm × 0.2 | derived |
| `min_drift_time` | 50 µs | 50 µs | floor on t_drift ⇒ σ never below its value at 7.8 cm drift | inherited, uBooNE readout-geometry choice |
| `div_sigma` | 6.0 mm | 6.0 mm | Gaussian charge-division width | inherited placeholder |

SBND's filters are `Gaus_wide` σ = 0.10 MHz, `Wire_ind` = (1/√π)×1.05,
`Wire_col` = (1/√π)×3.60 (`cfg/pgrapher/experiment/sbnd/sp-filters.jsonnet:81,109-110`)
— confirmed still matching the JSON's stated derivation. **If the SBND SP
filters are ever retuned, this JSON must be re-derived**; that dependency is
easy to miss because nothing enforces it.

Two residual uBooNE fingerprints inside the "derived" rows: the trailing
per-plane factors **0.2 / 0.3 / 0.5** and the `min_drift_time` floor are
empirical uBooNE tunings kept on purpose, to be revisited once SBND fit
residuals exist (`reduced_chi2` is already dumped by `save_stm_fit`; doc 41
measured p90 = 2.7, max 18.8, which is the handle).

## 7. Retune inventory

Columns: current value · where it lives · does changing it need C++ · what
measurement sets it. This extends §6.6 of
`sbnd/docs/sbnd-pattern-recognition.md` (the PID-constant inventory) rather
than duplicating it — §6.6 covers the `43e3` MIP constant across ~10 PR files
(97 sites) and the `0.8866 + 0.9533·(18/L)^0.4234` length correction; the
tables below are the STM-specific layer.

### 7a. The Bragg reference itself

| item | current | where | C++? | set by |
|---|---|---|---|---|
| stopping-muon dQ/dx curve | 60 pts, 123 417 → 48 879 e/cm, uBooNE field | `particle_dataset.jsonnet:14-29` | **no** | measured SBND stopping-muon dQ/dx vs residual range (MC truth + calibrated data) |
| proton / pion / kaon / electron curves | same origin | same, 30-93 | **no** | same; only `muon` + `proton` + `electron` are read (`detect_proton`, PR PID) |
| flat MIP line `50e3` |  hardcoded, 35 lines | `TaggerCheckSTM.cxx:1547` + 34 more lines | **yes** | SBND MIP plateau (§5) |
| `offset_length` {0, 3} cm | hardcoded | `:2343-2382` | **yes** | how far past the last fitted point the muon actually stops; SBND `dx` median 0.60 cm ≈ uBooNE's, so likely portable |
| `com_range` {15, 35} cm, `peak_range` {5, 20, 40} cm | hardcoded | `:2341-2382` | **yes** | track-length scale; SBND half-drift 200 cm vs uBooNE 256 cm |

### 7b. MIP-*relative* cuts — these would follow one scale knob

| cut | value | where |
|---|---|---|
| proton peak height `dQ_dx[max_bin]/50e3` | > 2.3, 2.5, 3.0, 3.5, 4.3 | `:1377-1394` |
| track median `segment_median_dQ_dx·cm/50000` | < 1.0, > 0.5, > 0.8 | `:1207, 1247, 1388, 2020` |
| residual `ave_res_dQ_dx/50000` (Michel veto) | 0.9, 1.2, 1.4, 2.3, 4.5 | `:1573-1589` |
| `ave_res_dQ_dx` absolute twins of the same ladder | 72 500, 85 000, 92 500 | `:1578-1586` |
| leftover-charge `left_Q/left_L/50e3` | > 2.0, < 1.5/1.7/1.8/1.9 | `:2304-2327` |
| kink fwd/bwd charge ratio normalization | `/50e3` | `:1025-1026, 1126-1127` |

Note rows 3 and 4 are the *same* physical cut expressed both ways
(`ave_res_dQ_dx/50000. > 1.2` next to `ave_res_dQ_dx > 72500`, i.e. 1.45×50e3)
— a single scale knob would silently desynchronize them. Any retune must
rewrite both forms together.

**Two more MIP-scale constants hide in internal units** (`units::cm = 10`, so a
dQ/dx expressed per internal length unit is e/cm ÷ 10). Easy to misfile as
absolute charge cuts; they are not:

| constant | literal | in e/cm | = MIP × |
|---|---|---|---|
| `default_dQ_dx` (`TrackFitting.cxx:5691`, `dQ = default_dQ_dx · dx`) | 5000 | 50 000 | **1.00** |
| `min_dQ_dx` low-charge gate (`TaggerCheckSTM.cxx:607`, `adjust_rough_path`) | 1000 | 10 000 | **0.20** |

`default_dQ_dx` is *exactly* the uBooNE MIP; it fills in the charge of points
the fit could not measure, so leaving it at 5000 while SBND's MIP is higher
biases those points low by the same 8–17 %.

### 7c. Absolute charge constants that do **not** follow a MIP scale

If SBND's true MIP is 8–17 % higher, each of these becomes effectively 8–17 %
*tighter* than uBooNE intended:

| constant | value | where | role |
|---|---|---|---|
| `charge_cut` | 2000 e | `sbnd_track_fitting.json:32` | per-channel charge for point association |
| `default_charge_th` / `default_charge_err` | 100 / 1000 e | `:15-16` | dead/low-signal handling |
| `add_charge_uncer` | 600 e | `:14` | additive charge error |
| `share_charge_err` | 8000 e | `:36` | charge-sharing penalty |
| `add_uncer_col` | 300 | `:11` | additive collection-plane uncertainty |
| `rel_uncer_ind` / `rel_uncer_col` | 0.075 / 0.05 | `:8-9` | per-plane relative uncertainty — SBND noise differs from uBooNE's |

`rel_uncer_*`, `add_uncer_col`, `add_charge_uncer` and `share_charge_err` are
set by SBND's *noise* level, not its MIP scale: they are what makes
`reduced_chi2` come out ≈ 1. Doc 41's p90 = 2.7 says they are currently too
tight (or the model is missing a term). That is the cheapest self-consistent
retune available and needs no new data.

### 7d. Geometry / topology thresholds inherited from uBooNE

Not charge-related, listed so they are not mistaken for calibrated values:
`TaggerCheckSTM.cxx` carries ~30 hardcoded lengths and angles — anode-proximity
2 cm and 6 cm (`:2264-2267`), perpendicularity ±12.5° with 15 cm arm
(`:2274-2275`), `exit_L`/`left_L` gates 3/7.5/8/20/40 cm (`:2280-2341`),
`check_other_clusters` scale 35 cm and its `number_clusters/3` formula
(`:2000`), `vhough_transform` 30 cm (`:2291`), `search_other_tracks` 1.5 cm /
0.8 (`:1664`). SBND's shorter drift (200 vs 256 cm) and 3 mm-vs-3 mm identical
pitch mean most transfer, but the drift-length-scaled ones (40 cm windows on a
200 cm drift) deserve a look once statistics exist.

### 7e. Correctly inert — do not chase these

- **`shorted_y_w_range`**: uBooNE's shorted-Y-wire guard. Default
  `m_shorted_y_w_min = -1` disables it and the SBND config never sets it, so
  both guard sites (`:993-996, 1072-1073`) are dead. Correct for SBND.
- **Reference-table coverage**: 0.5–59.5 cm vs a 38 cm maximum query (§1).
- **`examine_x_boundary`'s 257 cm**: already overridden by DetectorVolumes
  metadata in the SBND configs (`sbnd-pattern-recognition.md` §7).

## 8. Proposals (stated, not implemented)

1. **Replace the curve, don't scale it** (§4). Measure SBND's stopping-muon
   dQ/dx vs residual range and write a new `MuonDeDx` `values` array. This is
   jsonnet-only, needs no C++, and is the only fix that corrects `ks1` as well
   as `ratio1`. A recombination-rescaled curve (multiply the table by the §3
   per-R factors) is a defensible *interim* step **if and only if** confound 1
   is resolved first.
2. **Expose the MIP scale as one configurable** (`mip_dqdx`, C++ default
   `50e3` ⇒ key omitted ⇒ byte-identical) replacing all 35 literals, with
   §7b's dual-form cuts unified. Cheap, but note it does **not** fix §4.
3. **Apply the electron-lifetime correction** before any absolute-scale claim
   (confound 3). This is a real behavior change needing its own knob + gate.
4. **Retune `rel_uncer_*` / `add_*_uncer` to `reduced_chi2` ≈ 1** (§7c) — the
   one item that needs no external calibration input.

Ordering matters: 3 → 1 → 2, with 4 in parallel. Doing 2 first would bake the
current biased scale into a knob that then looks authoritative.

## 9. Already SBND-correct

- `BoxRecombination` field `Efield: 0.5` and `v_drift = 1.563 mm/µs` are
  mutually consistent and right for SBND (the `A`/`B` *parameters* are the open
  question, not the field).
- The SP-filter-derived smearing sigmas (§6), still matching
  `sbnd/sp-filters.jsonnet` as of this doc.
- The muon/proton/… **range** tables (range ↔ kinetic energy): pure
  material-property CSDA, genuinely detector-agnostic, no field dependence.
  These are fine as inherited — and §3 uses them as the independent cross-check
  precisely because they are.
- `shorted_y_w_range` disabled (§7e).

## 10. Open questions to route

1. **SBND MC recombination parameters** (`ModBoxA`/`ModBoxB` in the SBND
   LArSoft configuration): 0.93/0.212 or 1.0/0.255? Decides whether the SBND
   plateau prediction is 52.9 or 57.1 ke/cm, and whether the toolkit's
   `sbnd_box_recomb` matches the simulation it is applied to. **Owner/SBND
   calibration question — not resolvable in this tree.**
2. **SBND transverse diffusion `DT`** — same, lives in SBND LArSoft.
3. **The table's restriction energy** — undocumented; determines whether the
   comparison should be against restricted truth (49.8) or δ-ray-inclusive
   truth (60.6) on MC. Doc 46's two-channel truth (`true_dQ` + `true_dQ_sec`)
   is already the right instrument.
4. **`stopping_ave_dQ_dx_v2.root`** — what it was and why it was abandoned;
   its softer peak is what a resolution-smeared curve would look like, which
   would matter for a *fitted* comparison. Prototype-history question.

Nothing in 1–4 blocks the doc-40 phase-4 hand scan; all four block quoting an
SBND STM efficiency.

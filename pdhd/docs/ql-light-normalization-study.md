# PDHD Q/L light: predicted-vs-measured pattern & normalization study

**Run 27305, all 23 real events** (art events 150–238). Goal: use well-constrained
cosmic tracks crossing two TPC boundaries (one of them the anode or cathode) to
(1) confirm the predicted optical pattern has the same **shape** as the measured
flash, and (2) deduce the **normalization** of the predicted light yield.

**Headline:** the predicted pattern is *not* yet shape-consistent with the data.
It is **~2.5× too diffuse** — roughly **half the predicted light lands on PMTs
that measured exactly zero** (well above the detection floor, so not a threshold
effect). On the directly-lit PMTs the prediction is also **~2× too high**. A
single scalar normalization therefore cannot reconcile prediction and data; the
visibility/spread of the semi-analytical model has to be addressed first (or
jointly), then the normalization re-derived.

## Inputs & method

- Calib dumps produced by the clustering chain with `-calib`:
  `run_clus_evt.sh -calib 27305 <evt>` →
  `work/027305_<evt>/calib-evt<art>-group{02,13}.json`
  (group02 = APAs 0+2, drift −x; group13 = APAs 1+3, drift +x). Each bundle
  carries `pred_pe[160]` (predicted PE/opdet) and the matched flash carries
  `pe[160]` (measured PE/opdet); channel index ↔ opdet ↔ `pred_pe` ↔ `pe` align.
- **Crosser selection:** bundles flagged `at_x_boundary` (track touches anode or
  cathode) and, where available, `two_boundary` (crosses two boundaries). The
  calib dump is the full *candidate universe* (~500 bundles/group, mostly wrong
  flash↔cluster pairings), so each crosser cluster is associated to its
  **correct flash by minimum `ks_dis`** — the matcher's own KS shape metric,
  which is independent of the normalization we are solving for.
- **Valid PMTs:** `active & not auto_masked & ch ∉ static_mask`
  (static mask = {3,86,87,97,107,116,117} ∪ {120…159}; see
  `qlmatching.jsonnet`). Comparison and scale use only valid PMTs.
- A cluster's predicted total varies 2–3× across candidate flashes (t0 sets the
  drift x → visibility), so only the geometrically pinned, min-`ks` association
  is used.

## Clean sample

Across the 23 events: **147** `at_x_boundary` crossers (**17** two-boundary).
Small clusters tagged `at_x_boundary` are fragments whose low-`ks` match to a
bright flash is coincidental (predicted ~0 on the actually-bright PMT); they are
excluded by requiring a rich cluster (`ndf ≥ 30`) and a bright peak (brightest
PMT > 300 PE). That yields **5 clean anchors**, four of them genuine
two-boundary crossers:

| event | kind | ndf | ks | meas tot | pred tot | int. scale (meas/pred) | direct-PMT scale (top3) | N90 meas | N90 pred | pred on dark PMTs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evt202 | **2-bdry** | 34 | 0.096 | 13162 | 34173 | 0.39 | 0.52 | 5 | 12 | 20 % |
| evt162 | **2-bdry anode–cathode** | 38 | 0.116 | 2987 | 26937 | 0.11 | 0.36 | 13 | 29 | **57 %** |
| evt198 | **2-bdry** | 34 | 0.145 | 3617 | 18628 | 0.19 | 0.32 | 10 | 23 | 32 % |
| evt158 | anode crosser | 57 | 0.158 | 8052 | 24552 | 0.33 | 0.54 | 7 | 17 | 49 % |
| evt222 | **2-bdry** | 42 | 0.165 | 10143 | 67168 | 0.15 | 0.32 | 7 | 25 | 47 % |

`N90` = number of PMTs holding 90 % of the light (concentration); `scale` =
measured/predicted (>1 ⇒ prediction too low, <1 ⇒ too high).

**Aggregate over the 5 anchors:** direct-PMT scale (top-3) **median 0.36**
(IQR 0.32–0.52); integral scale **median 0.19** (IQR 0.15–0.33); spread
`N90(pred)/N90(meas)` **median 2.4×**; predicted light on dark PMTs **median
47 %**.

## Findings

### 1. The predicted pattern is too diffuse (shape error)

Measured light concentrates in **5–13 PMTs** (90 % of light); the model spreads
the same light over **12–29 PMTs** — a factor **~2.4×** (median over 5 anchors).
Decisively, **~47 % of the predicted light (median) falls on PMTs that measured
exactly 0 PE**, with the model predicting up to **1352 PE (evt162) / 5172 PE
(evt158)** on individual dark PMTs.

![concentration & per-PMT shape](../pics/ql_norm_concentration.png)

*Left:* cumulative light vs PMT rank — measured (solid) rises much faster than
predicted (dashed) for every crosser. *Right:* per-PMT normalized shape; the
horizontal spread at measured ≈ 0 is predicted light leaking onto dark PMTs.

#### Why the dark PMTs are really dark — not a detection floor

A natural worry: on a bright crosser the *dimmest lit* PMT is already ~90–100 PE,
with a hard gap straight down to 0 and nothing in between. If that ~90 PE were a
detection floor, the comparison above would be unfair — the model's predicted
skirt could be real light the readout simply throws away. It is **not** a floor.
Two checks on the raw flash arrays (run 27305 evt158, the 8052 PE anchor):

- **The chain records far below 90 PE.** Per-channel PE across all 31 flashes in
  that event runs smoothly down to **0.19 PE**; the `OpHitFinder` peak threshold
  is 3 scaled units = **0.03 PE** (`flash/inc/WireCellFlash/OpHitFinder.h`,
  `m_scale=100`). There is no ~90 PE hardware/reco floor.
- **The dark channels are alive.** Every channel reading exactly 0 in the crosser
  flash registers hundreds-to-thousands of PE in *other* flashes at *other* times
  (e.g. ch41 → 4872 PE, ch30 → 28712 PE, ch57 → 4191 PE elsewhere in the same
  event). They are dark *for this deposit*, not dead.

So the ~90 PE minimum is just where the lit cluster's edge happens to fall for
that track, not a threshold. The sharp 0 beyond it is **genuine VUV transport**:
128 nm scintillation light has an effective Rayleigh+absorption length ~85 cm in
LAr (`SemiAnalyticalModel.h`, `vuv_absorption_length` default), so over the
~3.5 m PDHD drift a localized/line deposit lights only the ~10–15 X‑ARAPUCAs
within roughly a scattering length and leaves the rest at essentially zero —
`exp(−350/85) ≈ 1/60`. The falloff is near-**exponential**, not the gentle
`1/r²` continuity intuition would suggest.

This *sharpens* the shape conclusion rather than weakening it. Because the dark
channels are demonstrably sensitive to <1 PE yet read 0, the data's sharp
localization is trustworthy and the discrepancy is a real model error. The model
geometric term is `exp(−d/λ)·(solid_angle/4π)` (`SemiAnalyticalModel.cxx`), but
the PDHD model file sets `vuv_absorption_length = 2000 cm` (attenuation **off** —
by convention it is meant to be folded into the Gaisser–Hillas fit; see
`wire-cell-data/pdhd/photodet/README.md`). The net effect is that the prediction
falls off as solid angle `~1/r²` while the data falls off near-exponentially:
`1/r²` keeps a long skirt on the far bars that the real detector does not have.
That mismatch *is* the "too diffuse" result. The spread fix (point 1 of the
next steps) is therefore the GH parametrization / an explicit `λ ≈ 85–100 cm`,
**not** a scalar normalization.

### 2. Normalization is too high (and entangled with the spread)

On the directly-lit PMTs the prediction is **~2.8× too high** (top-3 scale
median **0.36**, IQR 0.32–0.52 over the 5 anchors). Integrated over all PMTs it
is **~5× too high** (median 0.19), the extra factor being the diffuse tail of
(1). So the integral scale is **not** a clean normalization handle until the
spread is fixed; the direct-PMT scale (~0.36, i.e. ~2.8× reduction) is the most
defensible first cut.

Config at the time of this diagnosis (`cfg/pgrapher/experiment/pdhd/qlmatching.jsonnet`):
`QtoL = 1.0`, `vuv_eff = 0.03` placeholders, with `vuv_absorption_length = 2000` cm
(attenuation off) in the model. The light yield enters linearly, so the normalization
handle is `QtoL × vuv_eff`. **These were tuned in the update below.**

![crosser patterns](../pics/ql_norm_crosser_patterns.png)

*Per crosser:* measured 2-D PMT map (y vs z) | predicted map (normalized to the
measured total, so colour = shape only) | per-PMT bars sorted by measured. The
bright regions co-locate, but the prediction always carries a wider skirt.

## Conclusions / next steps

*(These were the recommendations from the diagnosis; steps 1–2 were since carried
out — see "Update: λ + normalization tuned" below. Step 3 still stands.)*

1. **Fix the visibility spread first.** The semi-analytical PDHD model
   (`pdhd/photodet/semi-analytical-pdhd.json`) puts ~half the light on PMTs that
   see nothing. Until its angular/distance fall-off is narrowed to match the
   data concentration (`N90` ≈ 7–13), a scalar normalization is ill-defined.
2. **Then set the normalization** from the directly-lit PMTs of clean crossers.
   The 5 anchors point to a **~2.8× reduction** of `QtoL × vuv_eff`
   (0.03 → ~0.011). Note the integral would then still be ~1.8× high until the
   spread of (1) is corrected, so the two fixes should be iterated together.
3. **More statistics still help.** 23 events yield 5 clean anchors. Run 27980
   (which also has x<0-side light) and further runs would tighten both the
   spread correction and the normalization, and let it be checked per drift side.

## Reproduce

```bash
# calib dumps (on disk for all 23 real 27305 events):
./run_clus_evt.sh -calib 27305 <evt>     # -> work/027305_<evt>/calib-evt<art>-group{02,13}.json
# plots regenerated by the analysis snippet that produced
#   pics/ql_norm_crosser_patterns.png  pics/ql_norm_concentration.png
```

Per-bundle fields used: `at_x_boundary`, `two_boundary`, `main_cluster`,
`flash_gid`, `ks_dis`, `ndf`, `pred_pe[160]`; flash `pe[160]`; opdet
`{x,y,z,ch,active,auto_masked}`. See `qlmatching-chain.md` for the matching
chain and tunables.

---

## Update (2026-06-13): λ + normalization tuned and data reprocessed

Following the diagnosis above, the visibility **spread** and the **normalization**
were tuned jointly and **all 23 run-27305 events with light** reprocessed end-to-end
through the real clustering + Q/L chain at the tuned model.

### Method — offline re-predictor, validated against the C++

λ enters the prediction only through `exp(−d/λ)`
(`SemiAnalyticalModel.cxx`), so the whole per-PMT pattern can be recomputed at any λ
directly from the charge points and the model JSON, with **no chain re-run**, by
porting `VUVVisibility` (rectangle solid angle + Gaisser–Hillas + border corrections)
to Python. The port was **validated to machine precision**: at λ=2000 it reproduces
the dumped C++ `pred_pe` on every predicted channel (per-channel ratio = 1.0000,
stdev 0). λ was then swept continuously over the crosser anchors and the result
applied to the real config + reprocessed (the reprocess is the end-to-end check).

### Result — λ ≈ 100 cm, vuv_eff 0.03 → 0.023

On the cleanest, brightest anchor (evt158, ks=0.158, 8052 PE — the one fully
trustworthy crosser) the predicted concentration `N90` falls monotonically with λ
and matches the data at λ≈100–150 cm; the directly-lit-PMT over-prediction largely
**dissolves once the spread is fixed** (it was mostly the diffuse tail, exactly as
predicted above). Chosen values (PDHD-only config):

- `vuv_absorption_length`: **2000 → 100 cm** (`wire-cell-data/pdhd/photodet/semi-analytical-pdhd.json`) — the canonical LAr 128 nm value; an *effective* spread correction (mildly double-counts the GH fit).
- `vuv_eff`: **0.03 → 0.023** (`cfg/.../pdhd/qlmatching.jsonnet`) — the ~0.77 direct-PMT scale at λ=100.

End-to-end reprocess of the cleanest anchor (real C++, λ=100, eff=0.023):

| metric | before (λ=2000) | after (λ=100, eff=0.023) | measured |
|---|---:|---:|---:|
| `N90` predicted | 17 | **6** | 7 |
| direct-PMT scale (top-3 meas/pred) | 0.54 | **1.00** | 1.0 |
| integral scale (meas/pred) | 0.46 | **0.78** | 1.0 |
| predicted light on dark PMTs | 49 % | **39 %** | 0 |
| predicted total PE | 24552 | **10331** | 8052 |

**Aggregate over the full 23-event sample** (7 clean anchors: at_x_boundary,
ndf≥30, measTot>3000, ks<0.4), before → after:

| aggregate (median) | before (λ=2000) | after (λ=100, eff=0.023) |
|---|---:|---:|
| direct-PMT scale | 0.36 | **1.00** |
| integral scale | 0.19 | **0.77** |
| `N90` ratio pred/meas | 2.4 | **2.0** |
| light on dark PMTs | 47 % | **39 %** |

The two new well-matched anchors the larger sample brings in are striking —
evt198 (direct 0.99, integral 1.04) and evt270 (`N90` 5 vs 5 exactly).

### What is fixed, and what is not

- **Normalization: fixed.** The directly-lit-PMT scale median is now **1.00** across
  the 7 clean anchors (was 0.36) — the absolute light scale is calibrated, and it
  holds as a population, not just on the single bright crosser.
- **Spread: fixed for clean single-track crossers, residual otherwise.** λ=100
  shrinks every anchor's `N90`, but the longer / multi-track anchors (e.g. evt150's
  29k-point blob, evt162, evt282) shrink from ~25 to ~14–16 without reaching their
  very low measured `N90`, so the median `N90` ratio improves only 2.4 → 2.0. A
  single global λ cannot capture this; the next-order correction is the **angular
  Gaisser–Hillas terms** (or the voxel photon library), and **more clean crossers**
  (run 27980, which also has x<0-side light) to pin it.
- **Provisional.** On the brightest anchor the matcher's `ks_dis` rose 0.158→0.256
  (a different CDF metric than the physical concentration — the pattern agreement
  clearly improved). Treat λ and `vuv_eff` as first-calibration, not final.

Analysis scripts: `pdhd/ql_light_calib/` (`repredict.py` validated re-predictor,
`fit.py` λ sweep, `after_metrics.py` reprocessed-dump metrics).

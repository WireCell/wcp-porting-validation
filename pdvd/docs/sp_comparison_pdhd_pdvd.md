# SP Chain Comparison — ProtoDUNE-HD vs ProtoDUNE-VD

For per-detector detail see [`pdhd/docs/sp.md`](../../pdhd/docs/sp.md) and
[`pdvd/docs/sp.md`](../../pdvd/docs/sp.md).  For the NF chain comparison see
[`pdvd/docs/nf_comparison_pdhd_pdvd.md`](nf_comparison_pdhd_pdvd.md).

All claims below are validated against
`cfg/pgrapher/experiment/{pdhd,protodunevd}/sp{,-filters}.jsonnet` and
`cfg/pgrapher/experiment/protodunevd/params.jsonnet`.

---

## 1. Pipeline shape

| Aspect | PDHD | PDVD |
|--------|------|------|
| Anodes per pipeline | 4 (APAs) | 8 (4 bottom CRPs + 4 top CRPs) |
| Per-anode SP factory | `make_sigproc(anode)` (`pdhd/sp.jsonnet:29`) | same function name (`protodunevd/sp.jsonnet:25`) |
| Sub-graph composition | `OmnibusSigProc` → optional L1SP merger | same |
| Per-anode branching axis | APA0 vs APA1–3 (geometry / wire orientation) | bottom (`ident<4`) vs top (`ident≥4`) (electronics + ADC fullscale) |
| `use_multi_plane_protection` default | `false` | `true` |
| L1SP default mode | `'process'` (live) | `'process'` (live; bottom-tuned trigger-gate overrides applied to all anodes) |

---

## 2. Per-anode branching — what splits and on what axis

### PDVD: top vs bottom (electronics, not geometry)

PDVD splits on electronics, not wire-orientation anomalies.  The filter
name suffixes `_b` (ident 0–3) and `_t` (ident 4–7) exist throughout
`sp-filters.jsonnet` and `sp.jsonnet`, but **the numerical values in every
`_b` and `_t` filter are currently byte-identical**.  The top/bottom split
is a structural hook for future independent tuning, not an active
parameter difference.  The only knobs that truly differ between top and
bottom today are electronics-derived:

| Knob | Bottom (ident 0–3) | Top (ident 4–7) | Why |
|------|--------------------|-----------------|-----|
| `elecresponse` | `tools.elec_resps[0]` — `ColdElecResponse`, 7.8 mV/fC, 2.2 µs shaping (`params.jsonnet:124–126`) | `tools.elec_resps[1]` — `JsonElecResponse` from `dunevd-coldbox-elecresp-top-psnorm_400.json.bz2`, postgain 1.36 (`params.jsonnet:129–132`) | Physically distinct front-end electronics on the two drift faces |
| `fullscale` (→ `ADC_mV`) | `params.adc.fullscale[1] − [0]` = 1.4 V → `ADC_mV` ≈ 11.71 / mV | hard-overridden to 2.0 V → `ADC_mV` ≈ 8.19 / mV (`sp.jsonnet:66–69`) | Top ADC spans 0–2 V; bottom 0.2–1.6 V |
| L1SP `kernels_file` | `pdvd_bottom_l1sp_kernels.json.bz2` | `pdvd_top_l1sp_kernels.json.bz2` | Per-region response kernels generated offline |
| L1SP `gain_scale` | `elec.gain / 7.8 mV/fC` | `1.0` (top gain is JSON-fixed; no runtime knob) | Reference electronics differ by region |
| L1SP `gauss_filter` | `'HfFilter:Gaus_wide_b'` | `'HfFilter:Gaus_wide_t'` | Picks the same-region Gaus_wide instance |

Everything else in the SP stage — all ROI threshold knobs, filter parameters,
multi-plane protection — is **the same for top and bottom**.

### PDHD vs PDVD per-anode branching compared

| Knob | PDHD APA0 | PDHD APA1–3 | PDVD (bottom = top for these) |
|------|-----------|-------------|-------------------------------|
| `field_response` | `np04hd-garfield-6paths-mcmc-bestfit.json.bz2` | `dune-garfield-1d565.json.bz2` | `protodunevd_FR_imbalance3p_260501.json.bz2` (uniform; top = bottom) |
| `filter_responses_tn` | 3 `FilterResponse` objects (U↔V order swap) | `[]` | not used |
| `plane2layer` | `[0, 2, 1]` (U↔V swap) | `[0, 1, 2]` | not configured (default `[0,1,2]`) |
| `r_th_factor` | `2.5` (looser) | `3.0` | `3.0` (uniform) |
| Wiener tight filter set | `_APA1` triplet (narrower σ) | default triplet | single triplet, `_b` == `_t` (identical values) |
| L1SP `process_planes` | `[0]` (APA0 V anomalous) | `[0, 1]` | `[0, 1]` for all anodes |

PDHD branches to handle a known APA0 wire-orientation anomaly; PDVD branches
to handle physically distinct electronics.  PDVD has no equivalent of PDHD's
APA0 U↔V correction or its loosened refinement threshold.

---

## 3. `OmnibusSigProc` knobs that are the same in both detectors

These knobs are set to the same literal value in `pdhd/sp.jsonnet` and
`protodunevd/sp.jsonnet`.  Where both override a WCT built-in default that
differs from the configured value, it is strong evidence the PDVD numbers
were copied from PDHD without independent tuning.

| Knob | Common value | WCT built-in default | Notes |
|------|-------------|----------------------|-------|
| `ftoffset` | `0.0` µs | `0.0` | field-response time offset |
| `fft_flag` | `0` | `0` | low-memory FFT path |
| `postgain` | `1.0` | `1.2` | **both override the default** — strong copy signal |
| `isWrapped` | `false` | — | wires/strips do not wrap |
| `troi_col_th_factor` | `5.0` | `5.0` | tight-ROI collection threshold (× noise RMS) |
| `troi_ind_th_factor` | `3.0` | `3.0` | tight-ROI induction threshold |
| `lroi_rebin` | `6` | `6` | rebin factor for loose-ROI search |
| `lroi_th_factor` | `3.5` | `3.5` | loose-ROI primary threshold |
| `lroi_th_factor1` | `0.7` | `0.7` | loose-ROI secondary (lower wing) |
| `lroi_jump_one_bin` | `1` | `0` | **both override the default** — allows ROI to bridge 1 empty bin |
| `r_fake_signal_low_th` | `375` | `500` | **both override the default** fake-signal rejection lower bound (e⁻) |
| `r_fake_signal_high_th` | `750` | `1000` | **both override the default** fake-signal upper bound (e⁻) |
| `r_fake_signal_low_th_ind_factor` | `1.0` | `1.0` | induction scale factor (lower) |
| `r_fake_signal_high_th_ind_factor` | `1.0` | `1.0` | induction scale factor (upper) |
| `r_th_peak` | `3.0` | `3.0` | peak detection threshold within refined ROI |
| `r_sep_peak` | `6.0` | `6.0` | minimum peak separation (ticks) |
| `r_low_peak_sep_threshold_pre` | `1200` | `1200` | pre-split charge threshold (e⁻) |

The three overrides of WCT defaults (`postgain`, `lroi_jump_one_bin`,
`r_fake_signal_*`) appear verbatim in both configs with identical numeric
values — the PDVD numbers were almost certainly carried forward from PDHD,
not independently retuned for the VD geometry.

---

## 4. `OmnibusSigProc` knobs that differ

| Knob | PDHD | PDVD | Why / status |
|------|------|------|--------------|
| `ctoffset` | `1.0 µs` (`pdhd/sp.jsonnet:61`) | `4.0 µs` (`protodunevd/sp.jsonnet:97`) | Must align the deconvolved output with the field-response reference time; determined by the FR file used. PDVD comment: "consistent with FR: `protodunevd_FR_imbalance3p_260501.json.bz2`" |
| `field_response` | per-APA (APA0 vs APA1–3; two files) | uniform — one file for all 8 anodes | PDVD uses a single simulated response for both drift faces |
| `filter_responses_tn` | APA0 only, 3 entries | not used at all | PDHD-only per-plane frequency correction; no PDVD equivalent |
| `r_th_factor` | `2.5` (APA0) / `3.0` (APA1–3) | `3.0` uniform | PDHD loosens refinement on APA0 to compensate for the V-plane anomaly; no PDVD analogue |
| `use_multi_plane_protection` | `false` | `true` | Real algorithmic difference: PDVD enables MP3/MP2 coincidence vetoes to suppress single-plane fake ROIs; PDHD leaves it off. Origin not documented; likely enabled during PDVD commissioning to reduce isolated induction-plane noise artefacts. |
| `plane2layer` | `[0,2,1]` (APA0) / `[0,1,2]` (APA1–3) | not set (uses default `[0,1,2]`) | U↔V swap is an APA0-specific geometry detail; not applicable to CRP strip geometry |
| `wiener_threshold_tag` | commented out in PDHD (deprecated) | still set (`'threshold%d'`) in PDVD | Minor: PDVD still emits the per-channel threshold summary trace tag under that name |
| `Wiener_tight_filters` list | APA0: `_APA1` set; APA1–3: default set | bottom/top: `_b` / `_t` sets respectively (values identical) | Structural split only on PDVD |

---

## 5. Filter catalogue — `sp-filters.jsonnet`

### Key observation: PDVD `_b` == PDVD `_t`

Every filter in `protodunevd/sp-filters.jsonnet` is registered twice under
the `_b` (bottom) and `_t` (top) suffixes.  **All numerical values are
currently byte-identical between the two**.  The table below therefore lists
a single PDVD column; where it says "PDVD `_b`" read it as equally applying
to `_t`.

### Low-frequency (LF) filters

| Name | PDHD τ (MHz) | PDVD `_b`/`_t` τ (MHz) | Same? | Notes |
|------|-------------|------------------------|-------|-------|
| `ROI_loose_lf` | `0.002` | `0.002` (b) / `0.003` (t) | ❌ | top CRP tightened for baseline suppression |
| `ROI_tight_lf` | `0.016` | `0.014` | ❌ slight | PDVD ~13% lower — marginally broader time support for tight ROIs |
| `ROI_tighter_lf` | `0.08` | `0.06` | ❌ | PDVD ~25% lower — broader LF envelope in the refinement path |

Higher τ → stronger low-frequency rejection → tighter ROI boundary.

### Gaussian (HF) filters

| Name | PDHD σ (MHz) | PDVD σ (MHz) | Same? |
|------|-------------|--------------|-------|
| `Gaus_tight` | `0.0` | `0.0` | ✅ |
| `Gaus_wide` | `0.12` | `0.12` | ✅ — identical; also seeds the L1SP smearing kernel |

### Wiener tight filters (primary output path)

| Plane | PDHD APA1–3 σ / power | PDHD APA0 (`_APA1`) σ / power | PDVD `_b`/`_t` σ / power | Same as PDHD APA1–3? | Notes |
|-------|----------------------|-------------------------------|---------------------------|----------------------|-------|
| U | `0.221933` / `6.55413` | `0.203451` / `5.78093` | `0.148788` / `3.76194` | ❌ | PDVD σ is narrower than both PDHD sets |
| V | `0.222723` / `8.75998` | `0.160191` / `3.54835` | `0.1596568` / `4.36125` | ❌ | PDVD σ close to PDHD APA0 V but not identical |
| W | `0.225567` / `3.47846` | `0.125448` / `5.27080` | `0.13623` / `3.35324` | ❌ | PDVD σ close to PDHD APA0 W range |

The PDVD Wiener-tight values match neither PDHD's current APA1–3 set
nor its APA0 `_APA1` set.  They are consistent with the May-2019
commented-out WCT default block that appears at the top of both
`sp-filters.jsonnet` files — suggesting PDVD inherited an older
pre-PDHD-tuning snapshot rather than copying PDHD's calibrated values.
This is worth verifying with the author: it may have been an intentional
conservative starting point, or an accidental inheritance from a stale
copy.

### Wiener wide filters (alternative path; not selected by default)

| Plane | PDHD σ / power | PDVD `_b`/`_t` σ / power | Same? |
|-------|---------------|---------------------------|-------|
| U | `0.186765` / `5.05429` | `0.186765` / `5.05429` | ✅ byte-exact |
| V | `0.1936` / `5.77422` | `0.1936` / `5.77422` | ✅ byte-exact |
| W | `0.175722` / `4.37928` | `0.175722` / `4.37928` | ✅ byte-exact |

The wide Wiener set is copied exactly from PDHD (neither is selected by
default in either detector's `wct-nf-sp.jsonnet`).

### Wire-domain (spatial) filters

| Name | PDHD σ (wire units) | PDVD `_b`/`_t` σ | Same? | Notes |
|------|---------------------|------------------|-------|-------|
| `Wire_ind` | `0.75/√π` ≈ `0.423` | `5.0/√π` ≈ `2.82` | ❌ **6.7× wider** | CRP induction strips subtend more wires per track width than PDHD APA wires; narrow smoothing would underweight adjacent-strip signal. This appears to be an intentional PDVD-specific choice. |
| `Wire_col` | `10.0/√π` ≈ `5.64` | `10.0/√π` ≈ `5.64` | ✅ | identical |

---

## 6. L1SP — unipolar-induction correction

`L1SPFilterPD` runs downstream of `OmnibusSigProc` inside `make_sigproc`
when `l1sp_pd_mode != ''`.

| Knob | PDHD | PDVD bottom | PDVD top |
|------|------|-------------|---------|
| `l1sp_pd_mode` default | `'process'` (live; replaces gauss/wiener) | `'process'` | `'process'` |
| `kernels_file` | `pdhd_l1sp_kernels.json.bz2` | `pdvd_bottom_l1sp_kernels.json.bz2` | `pdvd_top_l1sp_kernels.json.bz2` |
| `gain_scale` | `elec.gain / 14 mV/fC` | `elec.gain / 7.8 mV/fC` | `1.0` |
| `process_planes` default | APA0: `[0]` (V anomalous); APA1–3: `[0,1]` | `[0,1]` | `[0,1]` |
| `l1_len_very_long` / `l1_asym_very_long` | `140` / `0.35` (5th arm enabled) | C++ default (OFF) | C++ default (OFF) |
| `gauss_filter` | `'HfFilter:Gaus_wide'` | `'HfFilter:Gaus_wide_b'` | `'HfFilter:Gaus_wide_t'` |
| `l1_adj_enable` / `l1_adj_max_hops` | `true` / `3` | `true` / `3` | `true` / `3` |
| Raw-ADC thresholds at reference gain | `l1_raw_asym_eps=20`, `raw_ROI_th_adclimit=10`, `adc_sum_threshold=160` | same × gain_scale | same |

PDVD copies PDHD's raw-threshold numerical defaults and scales them to the
per-region reference electronics via `gain_scale`.  PDVD does not apply
PDHD's "very-long" 5th arm (`l1_len_very_long`, calibrated for run 027409
in 2026) or PDHD's APA0-specific V-plane suppression — those are
PDHD-specific calibration outcomes.

PDVD is now in `process` mode for all anodes (0–7).  The bottom-tuned
trigger-gate overrides (`l1_len_long_mod=180`, `l1_len_fill_shape=90`,
`l1_fill_shape_fill_thr=0.30`, `l1_fill_shape_fwhm_thr=0.25`,
`l1_pdvd_track_veto_enable=true`) are applied uniformly.  Top-CRP
threshold validation against a hand-scan is pending; the bottom-tuned
set is used as the starting point.

---

## 7. Output frame and downstream consumption

Identical between detectors:

- Output tags: `gauss{N}` (Gaussian charge), `wiener{N}` (Wiener-optimal)
- `FrameFileSink` with `digitize: false` and `masks: true`
- Per-channel threshold summary attached to `wiener{N}`
- Both `gauss{N}` and `wiener{N}` carry the L1SP result when L1SP is in
  `'process'` mode (the `FrameMerger` at the end of `make_sigproc` replaces
  both tags with the L1SP-modified gauss)

---

## 8. Quick-reference summary

| Question | PDHD | PDVD |
|----------|------|------|
| Multi-plane protection on by default? | No | Yes |
| Per-anode field-response file? | Yes (2 files) | No (1 file, shared) |
| Per-anode `r_th_factor`? | Yes (APA0=2.5) | No (uniform 3.0) |
| `filter_responses_tn` used? | APA0 only | No |
| `plane2layer` U↔V swap? | APA0 only | No |
| Top vs bottom electronics branching? | n/a | Yes (`elecresponse`, `fullscale`, L1SP) |
| Top vs bottom *filter parameters* differ? | n/a | **No** — `_b` == `_t` numerically |
| L1SP enabled by default? | Yes (`'process'`) | Yes (`'process'`, bottom-tuned overrides on all anodes) |
| L1SP per-region kernel files? | Single file | Yes (bottom + top) |
| Wiener-tight tuning origin? | PDHD calibration (run 027409) | Appears to be a pre-2019 WCT baseline |
| Wiener-wide tuning? | PDHD calibration | Byte-exact copy from PDHD |
| Wire-domain induction smoothing | Narrow (`0.75/√π`) | Wide (`5.0/√π`, ~6.7× PDHD) |
| `lroi_jump_one_bin`, `postgain`, `r_fake_signal_*` | Override WCT defaults | Same overrides — likely copied from PDHD |

---

## 9. Implications

- **Most OmnibusSigProc knobs are shared.** The long list of matching values
  in section 3 means PDVD's deconvolution, ROI finding, and charge-extraction
  thresholds are essentially the PDHD numbers.  Any PDHD SP tuning study is a
  useful starting point for PDVD, modulo differences in noise level and
  detector geometry.

- **Multi-plane protection is a real algorithmic difference.** With
  `use_multi_plane_protection: true`, PDVD vetoes ROIs that appear in only
  one plane without matching activity elsewhere.  PDHD keeps those ROIs.
  This will produce systematically fewer but cleaner ROIs on PDVD, and
  any direct gauss-output comparison between the two detectors must account
  for this.

- **Wiener-tight filters on PDVD are likely stale.** They do not match
  PDHD's calibrated APA1–3 set (which is significantly wider), and they
  appear to match the pre-2019 WCT baseline that was commented out.  The
  Wiener-tight path affects the `wiener{N}` output (and through it, track
  finding); a re-tuning pass against PDVD data should be planned.

- **Wire-domain induction smoothing was independently tuned for CRP
  geometry.**  The `Wire_ind` change from `0.75/√π` to `5.0/√π` is one
  clear deliberate VD-specific choice, acknowledging that CRP induction
  strips span more channels per track than PDHD APA wires.

- **Top/bottom split is a structural hook, not a tuning.**  When
  VD-specific per-region SP calibration becomes available, the `_b`/`_t`
  suffix infrastructure is already in place.  No code changes needed — only
  new numerical values in `sp-filters.jsonnet`.

- **ctoffset encodes the FR reference time.**  The 3 µs difference between
  PDHD and PDVD is not a physics difference but a field-response file
  convention; if the FR file is replaced, `ctoffset` must be re-verified.

---

## 10. Source cross-reference

| File | PDHD | PDVD |
|------|------|------|
| SP pnode factory | `cfg/pgrapher/experiment/pdhd/sp.jsonnet` | `cfg/pgrapher/experiment/protodunevd/sp.jsonnet` |
| Filter catalogue | `cfg/pgrapher/experiment/pdhd/sp-filters.jsonnet` | `cfg/pgrapher/experiment/protodunevd/sp-filters.jsonnet` |
| Detector parameters | `cfg/pgrapher/experiment/pdhd/params.jsonnet` | `cfg/pgrapher/experiment/protodunevd/params.jsonnet` |
| Top-level NF+SP driver | `pdhd/wct-nf-sp.jsonnet` | `pdvd/wct-nf-sp.jsonnet` |
| C++ SP engine | `sigproc/src/OmnibusSigProc.cxx` (shared) | same |
| L1SP C++ | `sigproc/src/L1SPFilterPD.cxx` (shared) | same |
| L1SP docs | `sigproc/docs/l1sp/L1SPFilterPD.md` | same |
| Field-response file | `np04hd-garfield-6paths-mcmc-bestfit.json.bz2` (APA0) / `dune-garfield-1d565.json.bz2` (APA1–3) | `protodunevd_FR_imbalance3p_260501.json.bz2` (all anodes) |

---

## 11. SP-knob deep dive — five questions answered

Sections 3–5 list these variables as table rows.  This section explains
the mechanism and cross-detector implications for each.

---

### 11.1 `use_multi_plane_protection` does not change traditional ROI output ✅

**Claim (from a colleague): enabling this option does not affect the traditional ROI determination.**  The code confirms this.

When `use_multi_plane_protection: true` (the PDVD default;
`protodunevd/sp.jsonnet:134`; C++ member default is `false`,
`OmnibusSigProc.h:246`), the code appends two extra steps **after**
`CleanUpROIs` and `generate_merge_ROIs`:

```
roi_refine.MP3ROI(iplane, ...)   // OmnibusSigProc.cxx:1754
roi_refine.MP2ROI(iplane, ...)   // OmnibusSigProc.cxx:1756
save_mproi(..., mp3_roi_traces, ...)  // :1758
save_mproi(..., mp2_roi_traces, ...)  // :1759
```

`MP3ROI` / `MP2ROI` populate two dedicated members (`proteced_rois`,
`mp_rois`) inside `ROI_refinement` and those members are saved to
`mp3_roi_tag` / `mp2_roi_tag` output traces
(`OmnibusSigProc.cxx:1911-1913`).

The key: **the traditional refinement loop that immediately follows**
(`BreakROIs` → `CheckROIs` → `CleanUpROIs` → `ShrinkROIs` →
`CleanUpCollectionROIs` / `CleanUpInductionROIs` → `ExtendROIs`,
`OmnibusSigProc.cxx:1768-1812`) **does not read `proteced_rois` or
`mp_rois`**.  Those members are only accessed through
`get_mp3_rois()` / `get_mp2_rois()`, which are called exactly once (in
the `save_mproi` calls above) to write the extra tagged traces.
`ROI_formation`'s loose/tight determination runs even earlier and is
wholly independent.

The companion flag `do_not_mp_protect_traditional`
(`OmnibusSigProc.cxx:1760-1763`) clears the MP ROI maps after tagging;
this flag confirms the design intent — the traditional path was never
going to read them anyway.

**Practical meaning for PDVD.**  Enabling MP generates additional
`mp3_roi_tag` / `mp2_roi_tag` trace streams consumed by downstream
imaging and DNN-SP stages.  The `gauss%d` and `wiener%d` output
frames, and the ROI positions and boundaries that define them, are
bit-identical whether MP is on or off.

---

### 11.2 `r_th_factor` — ROI-refinement RMS threshold scale

`r_th_factor` is a multiplier applied to the per-channel deconvolved
RMS during ROI **refinement**, not during the initial tight/loose ROI
**formation**.

**Stage distinction.**  Two separate threshold-factor knobs feed into
`OmnibusSigProc.cxx:1637`:

```
ROI_formation: receives th_factor_ind / th_factor_col (troi_ind/col_th_factor in jsonnet)
ROI_refinement: receives r_th_factor  ←  this knob
```

Inside `ROI_refinement`, the factor builds a per-row threshold
`plane_rms.at(irow) * th_factor` that gates `get_above_threshold(...)`
across `CleanUpROIs`, `BreakROIs`, `ShrinkROIs`, and extension-boundary
checks (`ROI_refinement.cxx:323,497,1193,1203,1223,1246,1253,1272,
1625,1628,1671,1674,1704,1707`).  A sample that falls below this bar
contributes to shrinking or splitting an existing ROI or causes a weak
ROI to be discarded.

**Values.**  PDVD uniform `3.0` (`protodunevd/sp.jsonnet:109`);
PDHD `2.5` on APA0 / `3.0` on APA1–3 (`pdhd/sp.jsonnet:73`).
PDHD loosens APA0 to compensate for the V-plane anomaly that creates
noisier induction; PDVD has no equivalent anomaly so uses a single
uniform value.  The C++ default is also `3.0` (`OmnibusSigProc.h:147`).

**Effect.**  Raising the factor trims more low-amplitude samples at ROI
edges and rejects more weak ROIs; lowering it keeps smaller pulses at
the cost of more noise-originated ROI fragments.

---

### 11.3 `r_fake_signal_*` — induction-plane charge threshold for ROI cleanup

Four variables gate **absolute charge** (in deconvolved electrons)
during the ROI cleanup step.

| Variable | C++ default | PDVD / PDHD value |
|----------|-------------|-------------------|
| `r_fake_signal_low_th` | `500 e⁻` | **375 e⁻** |
| `r_fake_signal_high_th` | `1000 e⁻` | **750 e⁻** |
| `r_fake_signal_low_th_ind_factor` | `1.0` | `1.0` |
| `r_fake_signal_high_th_ind_factor` | `1.0` | `1.0` |

Defaults in `OmnibusSigProc.h:148-151`; PDVD override at
`protodunevd/sp.jsonnet:110-113`.

**What they do.**  After the main ROI refinement loop, two cleanup
routines run:

- `CleanUpCollectionROIs()` (`ROI_refinement.cxx:1289`): keeps a
  collection ROI iff **any sample ≥ `high_th`** OR **mean ≥ `low_th`**
  (line `1300`).
- `CleanUpInductionROIs()` (`ROI_refinement.cxx:1371`): same logic
  with thresholds multiplied by the `_ind_factor` knobs
  (lines `1377-1378`).

A ROI that fails both conditions is discarded as a "fake signal" (the
name reflects the assumption that it is a noise fluctuation below MIP
scale — the comment at line `1294` reads "electrons, about 1/2 of MIP
per tick").

**Effect of the PDVD/PDHD override (375/750 vs 500/1000).**  The
lower thresholds are **more permissive**: weaker induction ROIs that
would be discarded at 500/1000 survive at 375/750.  This matches the
observation in §3 that both detectors override the WCT defaults
identically — the numbers were carried forward from PDHD and have not
been independently retuned for VD geometry or noise levels.

**No top/bottom split in PDVD.**  The same thresholds apply to
anodes 0–7 (`protodunevd/sp.jsonnet:110-113` does not branch on
`ident`).  The `_ind_factor` knobs provide a relative induction/collection
scaling axis for future per-region tuning without changing the
collection-plane bar.

---

### 11.4 Wiener wide is not stale — it sets the shape of the wiener output

The two Wiener filter sets serve **separate roles** in the
deconvolution pipeline.  Wide is live and required.

**Tight** (σ ≈ 0.149 / 0.160 / 0.136 MHz for U/V/W;
`protodunevd/sp-filters.jsonnet:97-102`) drives all the internal
deconvolutions used to **find** ROIs:

```
decon_2D_tightROI     → OmnibusSigProc.cxx:1274,1284,1294
decon_2D_tighterROI   → :1320,1330,1340
decon_2D_looseROI     → :1371,1391,1411
```

**Wide** (σ ≈ 0.187 / 0.194 / 0.176 MHz for U/V/W;
`protodunevd/sp-filters.jsonnet:104-109`) is the sole filter inside
`decon_2D_hits()` (`OmnibusSigProc.cxx:1525,1530,1535`), which
re-deconvolves the raw data with a wider HF envelope and writes the
result into `m_r_data[plane]`.  Immediately after, the ROI mask from
the tight pipeline is applied (`roi_refine.apply_roi`, line `1815`)
and the result is saved as the `wiener%d` output frame (line `1822`).

**Implication.**  ROI *positions and boundaries* are determined by the
tight path; the *signal amplitude and shape* inside those ROIs in the
published `wiener%d` frame comes from the wide path.  Removing the
wide filter list from the jsonnet config would cause `decon_2D_hits()`
to silently use a null or default filter, corrupting the wiener output.

**The §5 note "not selected by default in either detector's
`wct-nf-sp.jsonnet`"** refers to the top-level driver not having an
explicit `Wiener_wide_filters` override.  `sp.jsonnet:85-90` does
pass both lists to `OmnibusSigProc` explicitly; the C++ then calls
`decon_2D_hits()` unconditionally (line `1813`) within the normal
processing loop.  Both PDHD and PDVD always run the wide path.

---

### 11.5 Wire-domain induction filter — PDVD has ~6.7× more wire-direction smearing than PDHD

The `Wire_ind` and `Wire_col` filters are Gaussian kernels applied
along the wire/strip axis (before `inv_c2r`) to smooth the
2-D deconvolved data across adjacent readout elements.  A wider σ
means more cross-channel mixing.

| Detector | `Wire_ind` σ | `Wire_col` σ | Source |
|----------|-------------|-------------|--------|
| PDHD | `0.75/√π` ≈ 0.423 wire units | `10.0/√π` ≈ 5.64 | `pdhd/sp-filters.jsonnet:83-84` |
| PDVD bottom | `5.0/√π` ≈ 2.82 | `10.0/√π` ≈ 5.64 | `protodunevd/sp-filters.jsonnet:111,113` |
| PDVD top | `5.0/√π` ≈ 2.82 | `10.0/√π` ≈ 5.64 | `protodunevd/sp-filters.jsonnet:112,114` |

**PDVD induction smearing is ~6.7× wider than PDHD's.**  The
`Wire_ind_b` and `Wire_ind_t` values are numerically identical today
(the `_b`/`_t` split is a structural hook — see §2).

**Physical motivation.**  PDHD APAs use tensioned wires with ≈5 mm
pitch; induced charge from a charged-particle track lands primarily on
one to two wires.  PDVD CRP induction *strips* are wider and at an
angle such that a typical track projects across more strips; the SP
filter is sized accordingly so that the cross-strip smoothing mirrors
the intrinsic charge sharing rather than fighting it.  The collection
plane uses the same σ (10.0/√π) on both detectors because W-plane
collection geometry is more similar across the two designs.

**Context among other detectors.**  uBooNE `Wire_ind` ≈ 1.4/√π,
SBND ≈ 1.05/√π, pDSP ≈ 0.75/√π (matching PDHD).  PDVD's 5.0/√π
is the largest among production WCT configs and reflects the
strip-geometry readout rather than a tuning choice.

---

## 12. Wiener and LF filter selection — when each is used and how to tune

---

### 12.1 Wiener tight vs Wiener wide — different pipeline stages, not alternatives

The tight and wide Wiener sets are **not** two optional choices for the
same step.  They are active in different stages and serve opposite design
goals: tight is for reliable *detection*, wide is for accurate *amplitude*.

**Tight** (σ ≈ 0.149 / 0.160 / 0.136 MHz U/V/W on PDVD;
`protodunevd/sp-filters.jsonnet:97-102`) is used in every
ROI-*finding* deconvolution:

| `decon_2D_*` call | Tight filter applied at | What the result feeds |
|---|---|---|
| `decon_2D_tighterROI` | `OmnibusSigProc.cxx:1320, 1330, 1340` | `r_data_tight` → `find_ROI_by_decon_itself` cross-check |
| `decon_2D_tightROI` | `:1274, 1284, 1294` | `find_ROI_by_decon_itself` primary ROI seed |
| `decon_2D_looseROI` | `:1371, 1391, 1411` | `find_ROI_loose` wide-sweep ROI seed |
| `decon_2D_ROI_refine` | `:1250` | `roi_refine.load_data` → all refinement steps (`BreakROIs`, `ShrinkROIs`, `MP3ROI`, `MP2ROI`, `ExtendROIs`) |

**Wide** (σ ≈ 0.187 / 0.194 / 0.176 MHz U/V/W; identical across PDHD
and PDVD; `protodunevd/sp-filters.jsonnet:104-109`) appears in
exactly **one** call:

```
decon_2D_hits()   OmnibusSigProc.cxx:1525, 1530, 1535
```

`decon_2D_hits` re-deconvolves the raw data with the wide HF envelope,
writes the result into `m_r_data[plane]`, and that array is immediately
masked by `roi_refine.apply_roi` (line 1815) and saved as the
`wiener%d` output frame (line 1822).

**Why two σ values?**
A narrower HF cutoff (tight) suppresses more high-frequency noise before
the threshold test.  The signal peak is also squashed, but for a
boolean "is this above threshold?" decision that distortion is harmless
— the ROI boundary comes out cleaner.  The wide σ is then used to
re-deconvolve the same raw data with a softer HF cutoff, which preserves
the true pulse shape and amplitude inside the already-identified ROIs.
Using wide for detection would let more noise through and generate noisy
ROI candidates; using tight for the published output would produce
systematically clipped peaks.

`decon_2D_charge` (`OmnibusSigProc.cxx:1554`) plays the analogous
"wide" role for the `gauss%d` output using `m_Gaus_wide_filter`
(lines 1560, 1564, 1568), with the same ROI mask applied at line 1838.
So both `gauss%d` and `wiener%d` share the same ROI *positions* from
the tight pipeline; they differ only in which HF filter shaped their
amplitudes.

---

### 12.2 The three LF filters — one per ROI-finding decon path

The LF (low-frequency) filters are induction-plane-only high-pass
filters that kill 1/f-like noise below their rolloff frequency τ.  Each
of the three variants is bound to a specific ROI-finding decon:

| LF filter | τ PDVD / PDHD (MHz) | Bound to `decon_2D_*` | Purpose |
|---|---|---|---|
| `ROI_loose_lf` | 0.002 (b) / 0.003 (t) / 0.002 | `decon_2D_looseROI` (default) | Mildest rejection — retains the slow bipolar induction lobes for a wide, inclusive ROI sweep |
| `ROI_tight_lf` | 0.014 / 0.016 | `decon_2D_tightROI` + per-channel fallback in `decon_2D_looseROI` | Mid rejection — primary threshold-based ROI detection |
| `ROI_tighter_lf` | 0.06 / 0.08 | `decon_2D_tighterROI` | Strongest rejection — produces `r_data_tight`, a cross-check used inside `find_ROI_by_decon_itself` |

Collection plane (W) is never LF-filtered.  `decon_2D_tightROI` skips
the LF step when `plane == 2` (`:1294` only applies the HF filter),
and `decon_2D_looseROI` returns immediately for plane 2 (`:1360-1362`).

**Per-channel fallback in `decon_2D_looseROI`.**
The loose path pre-computes two filter arrays: `roi_hf_filter_wf`
(Wiener_tight × ROI_loose_lf) and `roi_hf_filter_wf1` (Wiener_tight ×
ROI_tight_lf).  For each channel it then picks between them
(`:1422-1425`):

```cpp
roi_hf_filter_wf2 = roi_hf_filter_wf;             // loose_lf by default
if (masked_neighbors("bad", ...) or
    masked_neighbors("lf_noisy", ...))
    roi_hf_filter_wf2 = roi_hf_filter_wf1;         // switch to tight_lf
```

Channels surrounded by "bad" or "lf_noisy" masked neighbors get the
more aggressively LF-rejected filter, which limits loose-ROI
contamination from their elevated LF noise.

**Why three paths instead of one?**
`find_ROI_by_decon_itself` (`ROI_formation.cxx:391`) takes both
`r_data` (tight-LF result) and `r_data_tight` (tighter-LF result) as
arguments: it uses the primary pass to find ROI candidates and the
secondary pass as a cross-check so that only candidates that survive
both levels of LF rejection are promoted.  The loose path is
independent — it runs at a lower threshold to catch activity that would
fall below the tight cut, broadening acceptance at the cost of a higher
false-positive rate that the subsequent refinement (`BreakROIs`,
`ShrinkROIs`, `CleanUpROIs`, …) must manage.

There is no `tighter_lf_tag` frame-output knob — only `tight_lf_tag`
and `loose_lf_tag` are emitted as tagged output traces.  The
tighter-LF decon is entirely internal to the ROI-finding stage.

---

### 12.3 How to tune

All numerical values live in
`cfg/pgrapher/experiment/protodunevd/sp-filters.jsonnet`
(PDVD) or `cfg/pgrapher/experiment/pdhd/sp-filters.jsonnet` (PDHD).
No C++ rebuild is required — only a jsonnet change.  For PDVD, apply
changes to **both** `_b` and `_t` instances (currently identical values;
see §5).

**Wiener tight** (controls ROI finding and refinement quality)

- Decrease σ or increase `power` → sharper HF cutoff → cleaner
  threshold discrimination → fewer noise-driven false ROIs.
- Increase σ or decrease `power` → softer cutoff → more signal-like
  pulses pass the threshold → fewer missed ROIs but more noise
  contamination.
- Symptom: many tiny spurious ROIs on noisy channels → narrow σ or
  raise power.
- Symptom: signal ROIs broken into fragments or disappearing near
  threshold → widen σ.
- PDVD's current U-plane σ ≈ 0.149 MHz is already the narrowest of any
  WCT production config and is likely an untuned inherited value (§5,
  §9).

**Wiener wide** (controls `wiener%d` pulse shape and noise inside ROIs)

- Increase σ → softer HF rolloff → more faithful peak shape in
  `wiener%d` → also more HF noise retained inside ROIs.
- Decrease σ → smoother `wiener%d` output → squished peaks.
- Symptom: wiener peaks look clipped or too narrow in time → widen σ.
- Symptom: wiener output is noisy inside ROIs → narrow σ.
- The PDHD and PDVD wide sets are currently byte-exact (§5); any PDVD
  re-tuning of the wide set should happen in `protodunevd/sp-filters.jsonnet`
  only.

**LF filters** (control induction-plane LF noise rejection in ROI finding)

- Increase τ on `ROI_tight_lf` → more aggressive LF cut → primary ROI
  finder sees cleaner data on noisy, LF-heavy channels → may miss ROIs
  from particles with slow drift signals.
- Decrease τ on `ROI_loose_lf` → even gentler → wider sweep ROIs grow
  further into LF-noise territory → refinement has more to clean up.
- Tune by inspecting the `loose_lf_tag` debug trace, which carries the
  output of `decon_2D_looseROI_debug_mode` (no per-channel tight-LF
  fallback; `OmnibusSigProc.cxx:1885`) — that trace shows the pure
  loose-LF decon without the noisy-neighbor override, useful for
  diagnosing how much LF content survives.
- The per-channel fallback threshold (neighbor count `n_bad_nn`,
  `n_lfn_nn` at `:1415-1416`) is hard-coded; only τ is configurable
  from jsonnet.

**Tag-name knobs do not control filter selection.**
`tight_lf_tag` and `loose_lf_tag` in `sp.jsonnet` only rename the
output trace frame tags.  The actual filter used is set by the string
knobs `ROI_tight_lf_filter`, `ROI_tighter_lf_filter`,
`ROI_loose_lf_filter`, `Wiener_tight_filters`, and
`Wiener_wide_filters`.  Changing a tag name has no effect on which
filter object is loaded.

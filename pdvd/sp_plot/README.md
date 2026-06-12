# pdvd/sp_plot — PDVD signal-processing inspection scripts

Eight families of scripts live here; each is documented below.

| Script | Purpose |
|---|---|
| `find_long_decon_artifacts_pdvd.py` | Offline reference detector for L1SP induction-plane artifacts (clustered output) |
| `eval_l1sp_trigger_pdvd.py` | Compare the L1SP tagger output (Python CSV or C++ NPZ) against the hand-scan ground truth |
| `extract_l1sp_clusters.py` | Extract L1SP-tagged ROI clusters from a calibration-dump NPZ and print a per-cluster summary table (any anode, top or bottom) |
| `cmd_plot_frames.py` | U/V/W frame views from a `FrameFileSink` archive |
| `track_response_l1sp_pdvd.py` | Validator for the PDVD L1SPFilterPD kernel JSONs (top + bottom) |
| `illustrate_pdvd_w_sentinel_path_bug.py` | Diagnostic plot for the all-zero sentinel-path bug in the PDVD W FR |
| `compare_sp_filters.py` | Compare PDVD and PDHD high-frequency-cutoff filters (Wiener wide/tight, Gaus, Wire) in frequency and time/wire-index domains |
| `compare_lf_filters.py` | Compare PDVD and PDHD low-frequency-cutoff filters (LfFilter: loose/tight/tighter) in frequency domain, impulse response, and synthetic-waveform demo |
| `noise_spectrum_compare.py` | Cross-detector post-NF noise frequency spectrum comparison: PDVD-top, PDVD-bottom, PDHD |
| `wiener_filter_construct.py` | Data-driven Wiener filter W(f) = \|S\|²/(\|S\|²+\|N\|²) for PDHD, PDVD-bottom, PDVD-top (U and V planes); plots frequency- and time-domain kernels |
| `filter_tune_viewer.py` + `serve_filter_tune_viewer.sh` | Interactive Bokeh viewer for tuning Wire / HF (Wiener / Gaus_wide) / LF (ROI_*_lf) software filters live on top of the **bare** pre-Wire-filter, pre-ROI deconvolved waveform (`hu/v/w_rawdecon<ident>`); production-gauss overlay for comparison |

Reference data:

| File | Purpose |
|---|---|
| `handscan_039324_anode0.csv` | Hand-scan ground truth for run 39324 events 0-5, PDVD bottom anode 0.  Schema mirrors `pdhd/nf_plot/handscan_27409.csv` plus a `real ∈ {Yes, No, Missing}` column (Yes = real artifact must fire; No = real prolonged track that must NOT fire; Missing = real artifact the gate currently misses).  Consumed by `--validate` mode of `find_long_decon_artifacts_pdvd.py` and the default ground truth of `eval_l1sp_trigger_pdvd.py`. |
| `pdvd_l1sp_rois_039324_evt{0..5}_anode0.csv` | Per-event clustered-ROI tables emitted by `find_long_decon_artifacts_pdvd.py --csv`.  Refresh after any defaults / algorithm change. |

---

## `noise_spectrum_compare.py` — post-NF noise spectrum comparison

Reads the post-NF `_raw` Magnify TH2 for one anode/APA per detector
(PDVD-bot anode 0, PDVD-top anode 4, PDHD APA 1, all from event 0),
masks signal-like samples per-channel using the `Microboone::SignalFilter`
algorithm (4 × percentile-RMS threshold, ±8-tick pad), then computes the
mean `|FFT|` across all channels in each wire plane.

```bash
/nfs/data/1/xqian/toolkit-dev/local/bin/python3 noise_spectrum_compare.py
```

Output PNGs (written beside the script):

| File | Content |
|---|---|
| `noise_spectrum_compare_U.png` | U-plane spectra (left = absolute ADC, right = area-normalized) |
| `noise_spectrum_compare_V.png` | V-plane spectra |
| `noise_spectrum_compare_W.png` | W-plane spectra |

Each PNG is a **1 × 2** panel:
- **Left** — mean `|FFT|` in ADC vs frequency (0–1 MHz, linear scale).
  Absolute amplitude reflects each detector's gain × ADC/mV product.
- **Right** — same spectra divided by their trapezoidal area (units 1/MHz),
  so all three curves integrate to 1 over 0–1 MHz. Lets you compare
  spectral *shape* independent of overall amplitude.

Signal masking mirrors `Microboone::SignalFilter`
(`sigproc/src/Microboone.cxx:573`): percentile-based RMS from unsaturated
samples (< 4096 ADC), threshold at 4 × rms, padded by ±8 ticks. Masked
samples are zeroed after anchoring the baseline to the non-signal mean, so
zeroing signal peaks does not introduce a spurious DC offset.

---

## `wiener_filter_construct.py` — data-driven Wiener filter

Constructs the textbook Wiener filter

```
W(f) = |S(f)|² / (|S(f)|² + |N(f)|²)
```

for PDHD APA1, PDVD bottom, and PDVD top, then iFFT's it into the
time domain.

**Signal `|S(f)|²`**: analytic FR ⊗ ER perpendicular-line MIP-track
response (one wire pitch) — the same "standard candle" as
`nf_plot/track_response_compare.py`.  Absolute ADC scale per detector
(own gain, shaping, postgain, ADC/mV), so S/N differences between
detectors feed directly into the filter shape.

**Noise `|N(f)|²`**: mean *power* per frequency bin — `<|FFT(ch)|²>`
averaged over all channels in the plane after signal masking (Microboone
SignalFilter), from the same post-NF `_raw` ROOT files as
`noise_spectrum_compare.py`.  This is `<|X|²>`, not `<|X|>²` (the
existing noise_spectrum_compare.py returns the latter; the Jensen bias
matters for the Wiener denominator).

**Window and scaling**: the filter is built on a **150 µs / 300-tick**
grid.  The noise was measured on a 3000 µs frame, so the mean power per
bin is scaled by `N_short / N_long` before interpolating onto the 150 µs
frequency grid.  Power (not amplitude) scales linearly with window length.

```bash
python3 wiener_filter_construct.py
```

Output PNGs (written beside the script):

| File | Content |
|---|---|
| `wiener_filter_U.png` | U-plane: W(f) (top) and w(t) = iFFT[W] (bottom) for all three detectors |
| `wiener_filter_V.png` | V-plane: same layout |

Each PNG is a **2 × 1** panel:
- **Top** — W(f) vs frequency (0–0.5 MHz).  W → 1 in the signal-dominated
  band, W → 0 where noise dominates (typically above 0.3 MHz).
- **Bottom** — w(t) vs time (±50 µs).  The time-domain filter kernel:
  tighter detectors (lower S/N) produce broader kernels.

---

## `filter_tune_viewer.py` — interactive Wire/HF/LF filter tuning

Bokeh-served web tool that loads the **bare** deconvolved waveform from
a magnify ROOT file and lets you apply Wire / HF (Wiener / Gaus_wide) /
LF (ROI_*_lf) software filters live on top, per channel.

### Input frame: `h{u,v,w}_rawdecon<ident>`

The viewer reads the special-mode TH2 family `h{u,v,w}_rawdecon<ident>`
emitted by `OmnibusSigProc::decon_2D_init` immediately after FR/ER
division but BEFORE any software filter (Wire_ind, Wire_col,
Wiener_*, Gaus_wide, ROI_*_lf) and BEFORE any ROI mask.  Production
runs do NOT emit this tag — it is opt-in via the `-R` flag on
`run_nf_sp_evt.sh` and `run_sp_to_magnify_evt.sh` (see those scripts'
`-h` help).

If a magnify file lacks `rawdecon` (i.e. it was produced before the
tap or without `-R`), the viewer falls back to `h{u,v,w}_wiener<ident>`
— note that this is the post-Wiener-tight, post-LF, post-ROI-mask
production frame, NOT a bare decon.  A red warning appears in the
info row when the fallback is in effect.

The same magnify file also yields `h{u,v,w}_wiener<ident>` —
the production post-Wiener+LF+ROI output, shown always-on as a
dotted-green overlay on the time-domain and spectrum panels for
visual comparison "your tuned filter vs. production".

### Filter forms (`util/src/Response.cxx:435-444`)

```
Wire / HF : H(f) = exp(-0.5 * (f/sigma)**power),  H[0]=0 if flag=true
LF        : L(f) = 1 - exp(-(f/tau)**2)
```

Wire is applied across the WIRE axis (frequency = cycles/wire),
HF and LF across the time axis (frequency = MHz, Nyquist = 1 MHz).

Production presets per detector (loaded into the dropdowns).
Values taken verbatim from
`cfg/pgrapher/experiment/{pdhd,protodunevd}/sp-filters.jsonnet`
and cross-checked against `wiener_filter_construct.py:99-117`.

**Wire filters** (shared across all HF groups):

| Filter | PDHD | PDVD bottom + top |
|---|---|---|
| Wire_ind (U, V planes) | σ=0.75/√π ≈ 0.4231, p=2 | σ=5.0/√π ≈ 2.821, p=2 |
| Wire_col (W plane) | σ=10.0/√π ≈ 5.642, p=2 | σ=10.0/√π ≈ 5.642, p=2 |

**HF filters** — three separate preset groups (selector updates on file change):

| Filter | PDHD (APA0/2/3) | PDHD APA1 | PDVD bottom (`_b`) | PDVD top (`_t`) |
|---|---|---|---|---|
| Gaus_wide | σ=0.12, p=2 | (same) | σ=0.12, p=2 | σ=0.12, p=2 |
| Wiener_tight U | σ=0.221933, p=6.554 | σ=0.203451, p=5.781 | σ=0.148788, p=3.762 | (= bottom) |
| Wiener_tight V | σ=0.222723, p=8.760 | σ=0.160191, p=3.548 | σ=0.159657, p=4.361 | (= bottom) |
| Wiener_tight W | σ=0.225567, p=3.478 | σ=0.125448, p=5.271 | σ=0.136230, p=3.353 | (= bottom) |
| Wiener_wide U | σ=0.186765, p=5.054 | (same) | σ=0.186765, p=5.054 | (= bottom) |
| Wiener_wide V | σ=0.193600, p=5.774 | (same) | σ=0.193600, p=5.774 | (= bottom) |
| Wiener_wide W | σ=0.175722, p=4.379 | (same) | σ=0.175722, p=4.379 | (= bottom) |

PDHD file selector automatically picks the PDHD group; PDVD anode
0–3 → bottom group; anode 4–7 → top group.

**LF filters**:

| Filter | PDHD | PDVD bottom + top |
|---|---|---|
| ROI_loose_lf | τ=0.003 | τ=0.003 (bottom + top) |
| ROI_tight_lf | τ=0.016 | τ=0.014 |
| ROI_tighter_lf | τ=0.08 | τ=0.06 |

### Launch

```bash
cd pdvd/sp_plot
./serve_filter_tune_viewer.sh 5007    # bundles all PDHD APA0-3 + PDVD anode0-7
```

View from a remote laptop:

```bash
ssh -L 5007:localhost:5007 user@workstation
# then open http://localhost:5007/filter_tune_viewer
```

The launcher bakes in **all** PDHD APAs (0-3, run 27409 evt 0) and
**all** PDVD anodes (0-7, run 39324 evt 0; anodes 0-3 = bottom CRP,
4-7 = top CRP).  Edit the `SPECS` array in the launcher to point at
different events / files; each spec is `label|path|ident|detector`
with `detector ∈ {pdhd, pdvd}` (selects which preset list appears in
the dropdowns).

### UI

- **Top row**: file selector, U/V/W plane radio, channel TextInput
  (typed global channel id), prev/next buttons, "Largest p-p"
  button (jumps to the channel with the largest peak-to-peak in the
  current plane), **Update plots** button (manual redraw fallback),
  **status div** (`idle` / `computing...` / `done · compute = X.X ms`).
- **Filter A row** (blue) and **Filter B row** (orange dashed) — two
  independent filter sets that are computed and overlaid simultaneously,
  so you can compare e.g. a production preset against a custom tune
  without switching back and forth.  Each row has three columns:
  - **Wire** column: preset Select (Wire_ind / Wire_col / Custom /
    `(none)`), σ Spinner (cycles/wire), power Spinner.  U/V
    auto-switch to Wire_ind, W to Wire_col, on plane change.
  - **HF** column: preset Select (detector-specific list; see tables
    above), σ Spinner (MHz), power Spinner, zero-DC checkbox.
  - **LF** column: preset Select (PDHD or PDVD values; see tables
    above), τ Spinner (MHz).

  Picking a named preset writes its values into the Spinners.
  Editing a Spinner value while a named preset is active switches
  the preset to `Custom` — the filter remains applied with your
  typed values.  Selecting `(none)` bypasses the filter entirely.
  Spinners use Bokeh's `value_throttled` event so a redraw fires
  only on **Enter / focus-loss**, not on every keystroke.

- **Range row**: t-min / t-max tick TextInputs + Apply / Reset
  buttons for explicit time-range zoom (wheel/box-zoom also work).

- **Plots**:
  1. *Time domain* — raw decon (grey dashed), Filter A (blue solid),
     Filter B (orange dashed), production wiener post-ROI (green dotted).
  2. *Filter shapes* — Wire (purple / pink), HF (blue / orange),
     LF (red / brown), HF×LF product (green / yellow-green dashed).
     Wire-axis x is rescaled to share the panel with the time-axis
     filters; absolute σ values are in the filter-row title bars.
  3. *Channel spectrum* — `|X(f)|` of the raw decon (linear-y);
     Filter A (blue), Filter B (orange), and production wiener (green)
     are overlaid.

### How the Wire filter is applied

The offline Wire filter exactly reproduces the production order in
`OmnibusSigProc::decon_2D_init`.  `rawdecon` arrives already wire-shifted
and stripped to `m_nwires` rows; the viewer reverses that:

1. Zero-pad `rawdecon` (time-rfft) to `m_fft_nwires = m_nwires + 2·WIRE_PAD`
   rows (WIRE_PAD = 10, hardcoded to match both PDVD and PDHD field-response
   files which use 21 wire paths).
2. Undo the production wire-shift (`np.roll(..., -WIRE_PAD, axis=0)`).
3. FFT along axis=0, multiply by `H_wire` and the time-axis filter `HL`.
4. IFFT along axis=0 (keeping complex — no `.real`), redo wire-shift,
   strip the pad rows to recover `m_nwires` rows.
5. `irfft` along the time axis for the selected channel.

The plane's time-rfft is cached lazily on first selection of a (file, plane),
so subsequent renders only redo steps 2-5 (~10 ms for a PDVD 476×6400 plane).
Choosing `(none)` for Wire skips steps 1-4 and goes straight to `irfft`
per channel (sub-millisecond).

### Environment

bokeh 3.9 lives in
`/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/`; uproot lives
in `/nfs/data/1/xqian/toolkit-dev/local/`.  The launcher prepends the
`local/` site-packages onto `PYTHONPATH` so the bokeh-env interpreter
can find uproot+awkward without modifying either env.

---

## `compare_sp_filters.py` — HF-cutoff filter comparison

Analytic reproduction of the high-frequency-cutoff (`HfFilter`) filters from
`protodunevd/sp-filters.jsonnet` and `pdhd/sp-filters.jsonnet`.  Formula
verified against `toolkit/util/src/Response.cxx:435-444` and
`toolkit/sigproc/src/HfFilter.cxx:38-52`.

```bash
python compare_sp_filters.py           # all five PNGs
python compare_sp_filters.py --only 5  # wire-filter comparison only
```

Output PNGs (written beside the script):

| File | Content |
|---|---|
| `compare_wiener_wide_freq.png` | Wiener wide |H(f)| vs frequency — shows PDVD and PDHD wide are identical |
| `compare_wiener_wide_time.png` | Wiener wide time-domain kernel (500 ns/tick, FWHM annotated) |
| `compare_wiener_wide_vs_tight.png` | Wide vs tight Wiener + Gaus_wide per plane: frequency + time domain overlaid |
| `compare_gauss.png` | Gaus_wide (σ=0.12 MHz, power=2): same for all detectors |
| `compare_wire_filter.png` | Wire_ind / Wire_col in wire-frequency and wire-index spatial domains |

**Wire-filter note.**  The σ values in jsonnet (e.g. `5.0/√π` for PDVD induction)
are *frequency-domain* parameters.  A *larger* σ_code means the filter stays
near 1 across all bins → near-delta in the spatial (wire-index) domain → **less**
strip-to-strip smearing.  Approximate spatial width: σ_spatial ≈ 1/(π σ_code).

| Detector | `Wire_ind` σ_code | σ_spatial (wires) |
|---|---|---|
| PDVD top/bottom | 5.0/√π ≈ 2.82 | ≈ 0.11 wire |
| PDHD            | 0.75/√π ≈ 0.42 | ≈ 0.75 wire |

PDHD induction smears ~6.7× wider across strips.  The collection wire filter
(σ_code = 10.0/√π) is identical on both detectors.

---

## `compare_lf_filters.py` — LF-cutoff filter comparison

Analytic reproduction of the low-frequency-cutoff (`LfFilter`) filters from
`protodunevd/sp-filters.jsonnet` and `pdhd/sp-filters.jsonnet`.  Formula
verified against `toolkit/util/src/Response.cxx:444`.

```bash
python compare_lf_filters.py
```

Output PNGs (written beside the script):

| File | Content |
|---|---|
| `compare_lf_filters_freq.png` | `\|H(f)\|` = 1 − exp(−(f/τ)²) for all three variants and both detectors |
| `compare_lf_filters_impulse.png` | Actual impulse response l(t) = iFFT[L(f)] = δ(t) − g(t); left=spike at t=0, right=negative wings (y clipped) |
| `compare_lf_filters_demo.png` | Filter applied to a synthetic waveform (narrow signal + slow + medium sinusoidal baselines); shows what each variant removes |

The LfFilter is a high-pass filter wired into the ROI-finding deconvolutions
(`decon_2D_looseROI` / `_tightROI` / `_tighterROI` in `OmnibusSigProc`).
Its actual impulse response is l(t) = δ(t) − g(t), where g(t) =
iFFT[exp(−(f/τ)²)] is a Gaussian.  Convolving a waveform with l(t) passes it
as-is (the δ spike) while subtracting a Gaussian-smeared copy (the −g wings),
giving a high-pass effect.  The Gaussian wing FWHM = 2√(2 ln 2)/(π·τ)
reveals the time-scale over which the filter affects neighboring samples:

| Variant | PDVD τ (MHz) | PDHD τ (MHz) | Wing FWHM g(t) (µs) |
|---------|-------------|-------------|----------------------|
| loose   | 0.003 (b+t) | 0.003 | ≈ 350 µs — DC-like drift only; l(t) ≈ δ(t) |
| tight   | 0.014 | 0.016        | ≈ 53 (PDVD) / 47 (PDHD) |
| tighter | 0.060 | 0.080        | ≈ 12 (PDVD) / 9 (PDHD) |

PDVD tight and tighter τ values are byte-identical top/bottom.  PDVD loose is now also byte-identical: bottom = top = PDHD = 0.003.

---

## `extract_l1sp_clusters.py` — L1SP calibration-dump cluster extractor

Reads the per-ROI NPZ files produced by `run_nf_sp_evt.sh -c <calib_dir>`
(`L1SPFilterPD` dump mode) and prints a per-cluster summary table in the
format:

```
Run    Event/Anode  plane  ch_lo  ch_hi  t_lo  t_hi  nch  len_max  gmax  fill  fwhm_f  asym  efrac  triggered
```

Each cluster groups adjacent-channel, time-overlapping tagged ROIs (`flag_l1_adj ≠ 0`).
Feature columns (`gmax`, `fill`, `fwhm_f`, `asym`, `efrac`) come from the
max-gmax seed ROI in the cluster.  The `triggered` column lists the union of
trigger arms that fired across all seed ROIs, ordered as in `decide_trigger()`:
`asym_strong`, `L_long`, `L_loose`, `fill_shape`.  BFS-adjacency-promoted
ROIs (promoted by a neighbour, not self-triggered) contribute `BFS_adj`.

Thresholds are selected per anode:

| Anode | Source |
|-------|--------|
| 0–3 (bottom) | `sp.jsonnet` PDVD overrides: `l1_len_long_mod=180`, `l1_len_fill_shape=90`, `fill=0.30`, `fwhm=0.25` |
| 4–7 (top) | C++ header defaults (no sp.jsonnet override yet pending top-CRP hand-scan validation) |

```bash
# Single event, anodes 6 and 7
python extract_l1sp_clusters.py \
    --calib-dir /home/xqian/tmp/pdvd_l1sp_calib_039324_0/calib \
    --run 39324 --event 0 --anode 6 7

# All anodes found in the calib dir
python extract_l1sp_clusters.py \
    --calib-dir /home/xqian/tmp/pdvd_l1sp_calib_039324_0/calib \
    --run 39324 --event 0
```

The calib dir is produced by:

```bash
cd pdvd
./run_nf_sp_evt.sh -c <calib_dir> [-a <anode>] 039324 0
# or directly:
wire-cell ... --tla-str l1sp_pd_mode=dump \
              --tla-str l1sp_pd_dump_path=<calib_dir> ...
```

`--wire-schema` is optional when `WIRECELL_PATH` includes the directory
containing `protodunevd-wires-larsoft-v3.json.bz2`.

---

## `find_long_decon_artifacts_pdvd.py` — offline reference detector

Reads the per-event Magnify ROOT file (`magnify-runRRRRRR-evtN-anodeA.root`)
and applies the same multi-arm gate as the C++ `L1SPFilterPD`, then
clusters per-channel sub-window candidates into per-cluster ROIs.

Three differences vs the C++ tagger (Python is the offline detector,
C++ is the production filter; both are tuned against the same hand-scan):

* operates at the **clustered** level (max-feature aggregation across a
  cluster's per-channel ROIs), while the C++ gate is per-sub-window;
* implements an additional **multi-channel-track veto** at the cluster
  level (`--multi-ch-min` / `--multi-ch-asym-esc`) that the C++ side
  realises differently via `l1_pdvd_track_veto_enable` (per-sub-window);
* defaults are tuned for PDVD bottom anode 0 against
  `handscan_039324_anode0.csv` (l_combo=90, ff_thr=0.30, fwhm_thr=0.25,
  len_long=180, asym_mod=0.50; multi-ch-min=4, multi-ch-asym-esc=0.85).

```bash
# Single event, print clusters and validate against the hand-scan
python find_long_decon_artifacts_pdvd.py --run 39324 --evt 0 --anode 0 --validate

# Save clusters to CSV (used by eval_l1sp_trigger_pdvd.py --source csv)
python find_long_decon_artifacts_pdvd.py --run 39324 --evt 0 --anode 0 \
    --csv pdvd_l1sp_rois_039324_evt0_anode0.csv
```

---

## `eval_l1sp_trigger_pdvd.py` — hand-scan evaluator

Compares the L1SP tagger output to `handscan_039324_anode0.csv` with
channel ∩ time overlap matching.  Two input sources:

* `--source csv` (default): reads
  `pdvd_l1sp_rois_039324_evt*_anode0.csv` (Python script's clustered
  output).
* `--source npz`: reads the C++ tagger's per-event NPZ dumps under
  `pdvd/work/<RUN>_<EVT>/l1sp_calib/apa<APA>_*.npz` and uses
  `flag_l1_adj` (the post-adjacency polarity that actually drives the
  LASSO; pass `--trigger-only` for the un-promoted `flag_l1`).  This
  mode also lets you re-apply the gate offline with overridden
  thresholds via CLI flags so you can probe what each threshold movement
  costs without rebuilding C++.

```bash
# Eval the Python script's current output
python eval_l1sp_trigger_pdvd.py --source csv

# Eval the C++ tagger's live output
python eval_l1sp_trigger_pdvd.py --source npz --use-cpp-flag

# Sweep one threshold offline against the C++ NPZ data
python eval_l1sp_trigger_pdvd.py --source npz --asym-mod 0.55
```

Mirrors the PDHD pattern at `pdhd/nf_plot/eval_l1sp_trigger.py`.

---

## `track_response_l1sp_pdvd.py` — kernel validator

Loads `pdvd_top_l1sp_kernels.json.bz2` and `pdvd_bottom_l1sp_kernels.json.bz2`
(via `WIRECELL_PATH`) and produces five inspection PNGs in this directory:

```
track_response_l1sp_pdvd_top_U.png
track_response_l1sp_pdvd_top_V.png
track_response_l1sp_pdvd_bottom_U.png
track_response_l1sp_pdvd_bottom_V.png
track_response_l1sp_pdvd_compare.png    # top vs bottom overlay
```

Each per-plane PNG has two stacked panels: positive ROI (bipolar +
W shifted to land at the bipolar zero crossing) on top, negative ROI
(bipolar + neg-half(bipolar), no shift) on bottom.  The compare PNG
overlays top and bottom on a shared time axis (relative to each
detector's V-plane zero crossing) so the relative W shift between
the two CRPs is visible at a glance.

By default the script also rebuilds the kernels in-process from the
field-response file and electronics preset (`load_detector_config` +
`build_l1sp_kernels`) and overlays them as thick translucent bands
behind the from-JSON curves.  Any drift between the on-disk JSON and
the FR source-of-truth is then immediately visible (the bands won't
hug the thin lines).  Pass `--no-rebuild` to skip the overlay if you
just want to inspect the JSON.

```bash
python track_response_l1sp_pdvd.py
# --top-file / --bottom-file override the defaults
# --no-rebuild skips the FR-rebuild overlay (faster)
```

Mirrors the PDHD validator at
`pdhd/nf_plot/track_response_l1sp_kernels.py`; uses the PDVD U/V wire
pitch (7.65 mm) for the `×N_MIP` ADC scaling.

---

## `illustrate_pdvd_w_sentinel_path_bug.py` — sentinel-path diagnostic

Documents an all-zero "sentinel" path at `pp=0` on the W plane of
`protodunevd_FR_imbalance3p_260501.json.bz2`.  Before the fix in
`wire-cell-python` commit `b1249b8`, `wirecell.sigproc.{l1sp,
track_response}.line_source_response` treated this entry as
legitimate data and pinned the trapezoidal integrator's central
weight to zero, under-normalising the W collection peak by ~12%
(integral −0.823 e → −0.920 e per electron, closer to the canonical
−1).  PDHD/uBooNE/SBND/PDVD-U/PDVD-V are unaffected.

The fix is one line: skip identically-zero paths in the input
loop.  No interpolation, no per-detector special case — the
trapezoidal weights at the surviving samples (pp=±0.51 mm here)
naturally widen to fill the gap.

```bash
python illustrate_pdvd_w_sentinel_path_bug.py
# writes pdvd_w_sentinel_path_bug.png — 3×3 grid:
#   row 0: PDVD U/V/W central-wire path currents (W column flagged ← SENTINEL)
#   row 1: PDHD U/V/W central-wire path currents (control: pp=0 always real)
#   row 2: line_source_response buggy vs fixed; U/V identical, W shows Δ
```

### Resolution: postgain values updated after FR fix

The all-zero sentinel path was **also in the upstream Garfield FR file**
(`protodunevd_FR_imbalance3p_260501.json.bz2`).  The FR has since been
regenerated (`FR_xn_boost_3.json.bz2`, copied over the same filename in
`wire-cell-data/`); re-running this script confirms the buggy/fixed
integrators agree (`peak ×1.0000`, `∫ ×1.0000`), so the W-plane
under-normalisation is gone.  The detector-calibration `postgain` values
that absorbed the deficit have been de-compensated accordingly:

- **PDVD-bottom: `postgain` 1.1365 → 1.0.**  PDVD-bottom shares cold
  electronics with PDHD (gain = 7.8 mV/fC vs PDHD's 14 mV/fC; everything
  else is the same chip).  PDHD has `postgain = 1.0`; the 1.1365 / 1.0 ≈
  1.137 excess closely tracked the W-plane line-source-integrator deficit
  (peak ×1.124, integral ×1.117).  After the FR fix the bottom postgain
  drops to PDHD-equivalent 1.0.
- **PDVD-top: `postgain` 1.52 → 1.36** (= 1.52 / 1.117).  Same calibration
  path (collection plane), same W under-normalisation, but with the
  top-CRP `JsonElecResponse` layered on top.

Updates landed in:

1. `wirecell/sigproc/track_response_defaults.jsonnet` —
   `pdvd-bottom.postgain = 1.0`, `pdvd-top.postgain = 1.36`.
2. `pdvd/nf_plot/track_response_pdvd_{bottom,top}.py` —
   `POSTGAIN` module constants updated.
3. `cfg/pgrapher/experiment/protodunevd/params.jsonnet` —
   `elecs[0].postgain = 1.0`, `elecs[1].postgain = 1.36`.
4. L1SP kernel JSONs regenerated in `wire-cell-data/`:

   ```
   wirecell-sigproc gen-l1sp-kernels -d pdvd-bottom  pdvd_bottom_l1sp_kernels.json.bz2
   wirecell-sigproc gen-l1sp-kernels -d pdvd-top     pdvd_top_l1sp_kernels.json.bz2
   ```

The coherent-noise removal kernels (`chndb-resp-bot.jsonnet`,
`chndb-resp-top.jsonnet`) were **not** regenerated — the NF thresholds
were tuned against those response shapes and re-tuning is deferred to a
later NF re-calibration pass.  See the headers of those files for the
generation-time postgain (1.1365 / 1.52).

---

## `cmd_plot_frames.py` — frame viewer

Draws U, V, W wire-plane views from a WireCell `FrameFileSink` archive (`.tar.bz2`).
Each output is a single PNG with three stacked panels — one per plane.

## Requirements

```
pip install numpy matplotlib
```

## Usage

Run the script directly — no woodpecker installation needed:

```bash
python cmd_plot_frames.py data/protodune-sp-frames-anode2.tar.bz2
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `frame_file` | yes | Path to a `*-anode<N>.tar.bz2` archive |
| `--tag TAG` | no | Frame tag to load (`raw`, `gauss`, `wiener`, …). Defaults to auto-detect. |
| `--out PATH` | no | Output PNG path. Defaults to `<frame_file>.png` beside the input. |
| `--tick-range T0 T1` | no | Restrict displayed ticks to `[T0, T1)` (relative, 0-based). |
| `--zrange ZMIN ZMAX` | no | Fix color-scale range. Otherwise auto-scaled per plane. |
| `--dpi N` | no | Output image resolution (default 150). |

## Examples

```bash
# Basic — auto-detect tag, output next to input file
python cmd_plot_frames.py data/protodune-sp-frames-anode2.tar.bz2

# Explicit tag
python cmd_plot_frames.py data.tar.bz2 --tag raw2

# Custom output path
python cmd_plot_frames.py data.tar.bz2 --out my_frames.png

# Zoom into ticks 1000–3000
python cmd_plot_frames.py data.tar.bz2 --tick-range 1000 3000

# Fix color scale to ±50 ADC
python cmd_plot_frames.py data.tar.bz2 --zrange -50 50

# High-res export
python cmd_plot_frames.py data.tar.bz2 --dpi 300
```

## Input archive format

The archive must contain `.npy` files produced by WireCell's `FrameFileSink`:

| Key pattern | Content |
|---|---|
| `frame_<tag>_<N>.npy` | 2-D array `(nchannels, nticks)` of ADC values |
| `channels_<tag>_<N>.npy` | 1-D array of channel IDs |
| `tickinfo_<tag>_<N>.npy` | `[start_tick, nticks, tick_period]` |
| `chanmask_bad_<N>.npy` | Optional bad-channel mask `(M, 3)` |

The anode index `N` is inferred from the filename (`anode<N>`).

## Color scale logic

| Plane / tag | Color map | Range |
|---|---|---|
| Any `gauss` tag | `hot_r` (white→black) | Fixed `0–1000` |
| W (collection), default | `hot_r` | `0 … 10×plane RMS` |
| U, V (induction), default | `RdBu_r` (blue–white–red) | `±10×plane RMS` |
| Any plane, `--zrange` | `RdBu_r` | User-supplied |

Bad channels are drawn as thin blue vertical lines on each panel.

## Tick axis

The y-axis shows **relative** ticks (0-based index into the stored frame), not the
absolute simulation clock tick. The absolute start tick is printed to stdout but not
shown on the plot, since it is typically a large simulation offset with no visual value.

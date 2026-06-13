# Per-channel PDHD light deconvolution — raw vs decon, old vs v1

Per-channel study of the WCT light deconvolution (`Flash::OpDecon`) on the
reference event **run 27305, event 150** (best single-event channel coverage:
59 self-triggered channels, OpChannel 0–82). Answers two questions:

1. **Is the deconvolution reasonable on every channel?** — `decon_ch<NNN>.png`,
   top panel = raw (baseline-subtracted, polarity-corrected ADC), bottom panel
   = deconvolved waveform (≈ PE/tick).
2. **Did the switch to the v1 per-channel SPE templates make any channel
   worse than the previous 2024-average templates?** — same figures overlay
   both deconvolutions; `decon_summary.png` ranks all channels.

## Files

| file | content |
|---|---|
| `decon_ch<NNN>.png` | one per channel: raw (top) + old-2024-avg (blue) vs v1/LArSoft (red) decon (bottom) |
| `decon_summary.png` | all channels: fast-peak amplitude (top) and late-tail level as % of peak (bottom) |
| `make_perchannel_plots.py` | regenerates everything (faithful NumPy port of `flash/src/OpDecon.cxx`) |
| `pdhd-spe-templates-2024.json` | the previous 2024 FBK/HPK average templates, for the comparison |

Regenerate: `python3 make_perchannel_plots.py` (needs the converted
`pdhd/work/027305_150/light-frames*.tar.bz2`, produced by
`run_light_evt.sh -m both 27305 150`).

## What the two template sets are

- **old (blue)** — the 2 average templates `SPE_NP04_{FBK,HPK}_2024_without_pretrigger.dat`
  with a flat Wiener noise term `N² = LineNoiseRMS²·Samples`. This is what
  `flash/` used before 2026-06-12.
- **v1 (red)** — the 113 per-channel run28368 v1 templates
  (`protodunehd_template_list_v1`) plus the run27950 per-channel noise power
  spectra. This is the actual DUNE production calibration
  (`protodunehd_pds_channels_data_v1`).

The v1 curve **reproduces the in-file LArSoft reference deconvolution exactly**
— the per-channel max|Δ| over all 57 checked channels is **3.5e-6** (printed
by the script, and annotated on each per-channel figure). So the red curve is
both "our v1 result" and "LArSoft", and the old-template (blue) result comes
from the same validated code path.

## Findings

### 1. The deconvolution is reasonable everywhere (fast peak)

On every channel the **fast prompt peak** is cleanly recovered — the raw
negative DAPHNE dip with its AC undershoot becomes a single positive ≈ PE/tick
peak. Old and v1 peak amplitudes agree to within ~10–20% channel by channel
(`decon_summary.png` top); v1 peaks are often slightly higher. No channel
fails to deconvolve. Channels 60 and 82 are essentially flat (no real pulse in
this event); channels 3/86/87/97/107/116/117 and 120–159 have no v1 template
(LArSoft `IgnoreChannels`: dead, noisy, or full-stream) and are skipped, as in
production.

### 2. v1 over-subtracts the slow tail on (almost) every channel

This is the real difference, and the user's instinct was right: the **old
2024-average templates gave a flatter, near-zero, physical late tail**, while
the **v1 (= LArSoft production) deconvolution pushes the slow scintillation
tail below zero** on essentially every channel (`decon_summary.png` bottom):

- late-tail mean over 5–15 µs, as a fraction of the peak: **median −3.5%**,
  **40 / 57 channels below −2%**, **18 / 57 below −5%**;
- worst channels (relative tail): **ch 1 (−34%), ch 11 (−28%), ch 41 (−23%),
  ch 20 (−18%), ch 10 (−14%), ch 30 (−13%), ch 2/0 (−12%), ch 51 (−11%)**;
- on the few brightest channels the negative excursion reaches −15 to −20
  PE/tick in absolute terms.

A negative deconvolved amplitude is non-physical (light rate ≥ 0), so this is
an **over-subtraction artifact of the v1 templates**, not present (or much
milder) with the 2024 average templates.

### 3. Root cause: the v1 SPE template *shape*, not the noise

Ablation (4 template/noise combinations, late-tail mean over 5–15 µs):

| ch | old + flat N² | old + run27950 N² | v1 + flat N² | v1 + run27950 N² (=LArSoft) |
|---|---|---|---|---|
| 0  | +0.51 | +0.41 | **−3.32** | −3.35 |
| 1  | +0.79 | −1.35 | **−18.68** | −19.69 |
| 11 | +1.37 | −0.04 | **−15.47** | −16.35 |
| 41 | +1.39 | +0.13 | **−18.69** | −18.49 |
| 61 | +1.95 | +0.15 | **−5.96** | −5.89 |

Swapping the **template** (old→v1) flips the tail strongly negative on its own
(`v1 + flat N²` column); swapping only the **noise** (old + run27950) barely
moves it. So the over-subtraction is driven by the per-channel v1 template
shape (its undershoot/return), with the run27950 noise spectrum a small
secondary effect.

## So — did the v1 update make things worse?

Two honest answers, depending on the goal:

- **For matching DUNE production: no — it is now correct.** v1 reproduces the
  official LArSoft deconvolution bit-for-bit (3.5e-6). The old 2024-average
  templates were never the production calibration; they just happened to give
  a visually cleaner tail.
- **For a physically clean deconvolution / robust slow-light PE: yes,
  somewhat.** The production v1 deconvolution over-subtracts the slow tail into
  negative values, which the cruder average templates did not. The **prompt
  PE is essentially unaffected** (the fast peak is preserved and the OpHit
  sliding window closes before integrating most of the negative tail), so the
  flash-level impact is bounded; but **late-light / long-pulse integrated PE is
  reduced**, most on the bright channels above.

Both are a one-line config choice in `flash.jsonnet` — `OpDecon` reads
`spe_file` / `noise_file`, so reverting to the average templates (or trying a
template with a corrected baseline) needs no code change. Note ch 41, the
worst over-subtractor, is dropped at hit level anyway by the LArSoft geometry
check (`IsValidOpChannel` → "unrecognized channel number 41"), so its bad
decon never reaches a flash.

## Decision (2026-06-12): default switched to the 2024 averages

Based on this, `flash/`'s **default `pdhd-spe-templates.json` is now the 2024
FBK/HPK average templates** with flat Wiener N² (`noise_file: ''` in
`flash.jsonnet`). The v1 per-channel set is preserved in
`pdhd-spe-templates-v1.json` and regenerable with
`extract_pdhd_spe_templates_v1.py`. This deliberately **diverges from the
LArSoft production deconvolution** (which uses v1 and over-subtracts) in favour
of a physically flat, DC-balanced result.

## Can we calibrate the SPE from the data and do better? (`spe_calibration_study.png`)

Tested directly (`make_spe_calibration_study.py`): isolate single-PE pulses
from the raw snippets, average them (cross-correlation aligned), and use the
result as the kernel. Conclusion — **not with this dataset**:

- **Single PEs sit right at the noise/trigger floor.** The HPK SPE peak is
  ~12–14 ADC while the line-noise RMS is ~4.5 ADC, i.e. only ~2.5–3σ, and the
  self-trigger threshold cuts in at the same level. The amplitude spectrum of
  isolated pulses peaks at ~11–12 ADC with no clean valley separating it from
  noise.
- **The low-amplitude average is noise-biased narrow.** Averaging 1-PE-band
  pulses gives an SPE FWHM ~96 ns, far narrower than the 2024 template's
  256 ns. As the amplitude band is raised (better SNR) the measured width
  climbs through ~190 → ~225 ns, **converging on the 2024 template** — i.e. the
  narrowing is a noise/threshold artefact and the 2024 width is consistent with
  the data (panels 1–2).
- **The long DC-balancing tail can't be measured.** What makes the 2024
  template deconvolve flat is its long, smooth, DC-balanced (total area ≈ 0)
  undershoot tail. In flash/cosmic data the late tail of any SPE is
  contaminated by later light, so a data-extracted template is necessarily
  short/truncated; such templates deconvolve far worse and erratically
  (late-tail anywhere from +3 % to below −50 % of peak depending only on
  extraction choices), versus the 2024 average's robust ≈ 0 % (panel 3).

Doing better would need **dedicated low-light / LED calibration runs** with
clean single-PE separation and a fully sampled response — exactly how the
official 2024 and v1 templates were produced — not beam/cosmic self-triggered
data. Reproduce: `python3 make_spe_calibration_study.py`.

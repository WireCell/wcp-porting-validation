# PDVD light deconvolution — software-filter determination

Milestone 2 of the light-reconstruction plan (`light-reco-plan.md` §3): fix the software
filter of the `OpDecon` stage for PDVD. The chain uses the PDHD-style **fixed Wiener filter**
`G = conj(H)·R/(|H|²R + 1)` (R = `fixed_snr`; length- and amplitude-independent, the
prerequisite for deconvolving the 1024-tick snippets and the 468k-tick cathode stream with
the same filter) followed by a **Gauss post-filter** (`postfilter_cutoff`, MHz). PDHD
production values: R = 0.005, cutoff = 1.5 MHz.

> Script: `pdvd/pd_plot/filter_scan.py` (uses the milestone-1 templates and real quiet-noise
> records). Figures: `docs/pds/filter_scan.png`, `docs/pds/filter_noise_psd.png`.
> Companion: `pdvd-spe-template.md` (templates, normalization).

## Method

Scan `fixed_snr` ∈ {0.001…0.02} × cutoff ∈ {0.5…6 MHz} per channel; aggregate per
population (median). Everything is measured on **real data**, so the non-white noise is
automatically accounted for:

- **decon noise floor** — real quiet stretches (snippet pre-trigger heads; pulse-masked,
  HPF'd cathode stream segments) pushed through the filter, in autoscaled PE/tick — this is
  what sets the OpHit threshold;
- **1-PE peak height** — the held-out average 1-PE pulse deconvolved with the channel
  template; **significance = peak/floor** is the quality lever;
- **matched-kernel FWHM** (two-pulse separation) and **negative sidelobe** (ringing).

## Results (medians; full tables from the script)

| population | 1-PE significance @1.5 MHz | best-region behaviour |
|---|---|---|
| cathode XA (4 quiet ch) | 6.9 | flat plateau 0.5–1.5 MHz (max ~7.0 @1.0), falls above |
| membrane XA bottom (8 ch) | 7.6 | 12.8 @0.5, 8.7 @1.0 — monotone drop, but kernel widens to 38/18 ticks |
| membrane XA top (6 ch) | 1.9 | dominated by a broad 0.5–10 MHz pickup structure (see PSD); 5.6 @0.5 |
| PMT (20 ch) | 31.7 | **flat 29–34 across the whole scan** — the fast PMT pulse is wide-band |

`fixed_snr` is a weak lever everywhere on real noise (curves nearly coincide); larger R only
tames the high-cutoff sidelobe. Note a pleasant algebraic fact: under `fixed_snr` the filter
depends only on R and the template spectrum — `line_noise_rms` cancels and needs no
per-population tuning.

## Decisions

| knob | XA branches (cathode full-stream + membrane snippets) | PMT branch |
|---|---|---|
| `fixed_snr` | **0.005** (PDHD parity; insensitive) | **0.005** |
| `postfilter_cutoff` | **1.5 MHz** (significance within ~2 % of its plateau, kernel FWHM 12–14 ticks vs 38 at 0.5 MHz, sidelobe ≤0.2 %) | **3.0 MHz** (significance unchanged, kernel FWHM 6 ticks — 2× better two-pulse separation, sidelobe ~0.01 %) |

**Chain implication:** the PMT cutoff differs from the XA one, so the self-trigger chain
splits into two parallel `OpDecon`+`OpHitFinder` branches (membrane XA 20xx vs PMT 30xx)
before `OpHitMerge` — `postfilter_cutoff` is per-instance, not per-channel. This split is
free and also lets the two branches carry different hit thresholds.

**Starting OpHit thresholds** (5× the measured decon floor, in the OpHitFinder ×100 scaled
units): membrane-bottom ≈ 4, cathode ≈ 5–7, PMT(3 MHz) ≈ 2.5, membrane-top ≈ 12. To be
finalized at chain assembly (the PDHD analogue: snippet 3.0, full-stream 11.0).

## Caveats

- **Membrane-top pickup**: the six top-wall XA channels carry a broad coherent structure
  (0.5–10 MHz, ~30× the bottom-wall power — `filter_noise_psd.png`). The flat-noise Wiener is
  suboptimal there; their 1-PE significance stays ~2–3 at practical cutoffs. Options, in
  order: per-channel noise spectra via the existing `OpDecon` `noise_file` mechanism, a
  lower dedicated cutoff, or simply higher PE thresholds on those channels. Deferred.
- The cathode floor is measured on HPF'd quiet stretches — valid because the production
  full-stream path runs `OpRoi` (same HPF) before hit finding; only 4 of 8 cathode OpDet
  streams yielded clean quiet segments (the others are busier/noisier), enough for the median.
- Noise is non-white everywhere (1/f-ish toward ~1 MHz + white floor above ~15 MHz); the scan
  results absorb this since they use real noise, but a `noise_file` Wiener is the principled
  upgrade if more 1-PE sensitivity is ever needed.

## Reproduce

```bash
cd pdvd
python3 pd_plot/spe_build.py 0          # if the harvest cache is absent
python3 pd_plot/filter_scan.py 39252
```

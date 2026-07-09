# PDVD light deconvolution — software-filter determination

Milestone 2 of the light-reconstruction plan (`light-reco-plan.md` §3): fix the software
filter of the `OpDecon` stage for PDVD. The initial scan (below) used the PDHD-style
**fixed Wiener filter** `G = conj(H)·R/(|H|²R + 1)` (R = `fixed_snr`) followed by a
**Gauss post-filter** (`postfilter_cutoff`, MHz), PDHD production values R = 0.005,
cutoff = 1.5 MHz. The **adopted filter is the Wiener-INSPIRED one** (§"Wiener-inspired
filter" below): same band shape, but pinned to exactly 1 at zero frequency so the filter
knobs cannot move the PE normalization.

> Scripts: `pdvd/pd_plot/filter_scan.py` (Wiener scan), `pdvd/pd_plot/filter_wi.py`
> (Wiener-inspired fit + decision; uses the milestone-1 templates and real quiet-noise
> records). Figures: `docs/pds/filter_scan.png`, `docs/pds/filter_noise_psd.png`,
> `docs/pds/filter_wi.png`. Companion: `pdvd-spe-template.md` (templates, normalization).

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

These Wiener-scan results (fixed_snr = 0.005 insensitive; cutoffs 1.5 MHz XA / 3.0 MHz PMT)
define the *reference band shape* that the adopted Wiener-inspired filter is fitted to.

## Wiener-inspired filter (adopted)

The Wiener form has two blemishes for a calibration chain:

1. its DC gain `|H(0)|²R/(|H(0)|²R+1)` depends on `fixed_snr` and the template spectrum, so
   the deconvolved area (= the PE scale) moves with the filter knobs and has to be patched
   up by the AutoScale positive-lobe renormalization;
2. the effective high-frequency cutoff is an opaque mix of `fixed_snr`, the template
   spectrum and the post-filter — there is no single physical knob to trim on a noisy
   channel group.

Following the WCT signal-processing convention, deconvolve instead with the **pure inverse
`M/H` times a Wiener-inspired band filter**

```
F(f) = exp(−0.5 (f/σ)^p),   F(0) = 1 exactly
```

with (σ, p) **fitted to the actual Wiener × post-filter band shape** per population
(`filter_wi.py`, `filter_wi.png`). Because F(0) = 1, a 1-PE pulse deconvolves to unit area
*independent of the filter parameters* — no AutoScale needed — and σ is a single physical
cutoff (MHz) that can be lowered on noisy populations without touching the normalization.
It stays length- and amplitude-independent (F is a function of physical frequency only),
so full-stream and snippets share it, like `fixed_snr`.

**Fits** (per-channel spread tiny): cathode σ = 1.76 / p = 2.05, membrane top σ = 1.79 /
p = 2.02, membrane bottom σ = 1.80 / p = 2.03, PMT σ = 3.59 / p = 2.00 — i.e. the
reference band is almost exactly Gaussian (p ≈ 2). At the fitted (σ, p) the WI filter
reproduces the Wiener baseline in every metric (significance, FWHM, sidelobe) — it is the
same filter, minus the normalization dependence.

**σ trim scan** (real quiet-noise floors; `filter_wi.png` middle row): cutting the high
side pays where the noise is worst — membrane-top significance 2.0 → 4.4 at σ×0.5 and
membrane-bottom 7.0 → 9.5, while cathode is flat (6.3–6.5) and PMT flat (31–33). The cost
is impulse FWHM 12 → 22–26 ticks (0.35–0.42 µs), harmless against the ~1 µs XA pulse and
the 1 µs flash bin.

### Decisions (final)

| branch | σ (MHz) | p | 1-PE significance | impulse FWHM | vs Wiener baseline |
|---|---|---|---|---|---|
| cathode full-stream | **1.25** | **2** | 6.4 | 18 ticks | 6.3 |
| membrane XA (top+bottom, one branch) | **1.0** | **2** | bottom 9.0 / top 3.9 | 22 ticks | 7.0 / 2.1 |
| PMT | **3.5** | **2** | 32.2 | 6 ticks | 32.1 |

Normalization stability, quantified: sweeping `fixed_snr` 0.001→0.02 moves the raw
(pre-AutoScale) 1-PE decon area by ≲0.2 % on these templates (the Wiener DC-gain error is
small here), but the WI area is *exactly* filter-independent by construction and the
AutoScale stage becomes a no-op — one fewer moving part in the PE calibration.

**Chain implication:** `OpDecon` needs a Wiener-inspired mode (jsonnet-togglable, default
OFF): `G = conj(H)/(|H|² + ε) · F(f)` with knobs `wi_sigma_mhz`, `wi_power`; AutoScale
skipped in this mode. The PMT σ still differs from the XA σ, so the self-trigger chain
splits into membrane-XA and PMT `OpDecon`+`OpHitFinder` branches before `OpHitMerge` as
before; membrane top and bottom now share one branch (σ = 1.0 doubles top significance
*and* improves bottom — the earlier membrane-top caveat is resolved this way).

**Starting OpHit thresholds** (5× the measured decon floor, OpHitFinder ×100 scaled units,
WI floors): membrane-bottom ≈ 2, membrane-top ≈ 4, cathode ≈ 3.7, PMT ≈ 2.2. To be
finalized at chain assembly (the PDHD analogue: snippet 3.0, full-stream 11.0).

## Caveats

- **Membrane-top pickup**: the six top-wall XA channels carry a broad coherent structure
  (0.5–10 MHz, ~30× the bottom-wall power — `filter_noise_psd.png`). The σ = 1.0 MHz
  membrane trim (above) recovers their 1-PE significance to ~3.9; a per-channel
  `noise_file` Wiener remains the principled upgrade if more sensitivity is ever needed.
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
python3 pd_plot/filter_wi.py 39252      # Wiener-inspired fit + decision
```

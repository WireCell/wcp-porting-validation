# Why some flashes show measured light ≫ predicted (evt 298567 flash gid 57)

**Repro:**
```
cd pdvd/docs/qlmatch && python3 lightpattern_sp_investigation.py   # tables + figs
# inputs: work/039252_0_arcal/calib-evt298567.json (+ the 17 sibling arcal dumps),
# input_data_light_rawwf/np02vd_raw_run039252_*_rawwf.root,
# toolkit cfg pdvd-spe-templates.json / pdvd-opch-map.json,
# photlib/pdvd-photlib-vis-v5-128nm.json
```

## Symptom

On the `arcal` candle round (port 5019), flash gid 57 (t = −270.6 µs, grp 58,
matched to the cathode-crosser pair top:56 / bot:134) shows several cathode
X-Arapucas with measured PE far above the displayed prediction (od4: 12764
measured vs 0 predicted) while other X-Arapucas agree well.  Similar
mismatches appear on other cathode crossers.  Since Q-L matching *is* the
comparison of these patterns, this had to be understood.

## What the waveforms say: the light is real and the deconvolution is faithful

Raw-waveform examination (fig 1) shows one genuine, simultaneous, very bright
flash on **all** cathode channels: a single prompt peak at the flash time with
a ~8 µs tail, 30k PE total, four opdets railed at the DAPHNE 14-bit ceiling.
Decon PE tracks the raw area on every cathode channel:

| od | sat | dump PE | raw-area PE | dump/raw |
|---|---|---|---|---|
| 4 | 1 | 12764 | 9061 | 1.41 (repair recovers clipped area) |
| 5 | 0 | 940 | 1160 | 0.81 |
| 6 | 1 | 3773 | 3955 | 0.95 |
| 7 | 1 | 2462 | 2189 | 1.12 |
| 8 | 1 | 4738 | 5571 | 0.85 |
| 9 | 0 | 1139 | 1183 | 0.96 |
| 10 | 0 | 2683 | 2880 | 0.93 |
| 11 | 0 | 1052 | 1124 | 0.94 |

And the **unmasked** prediction of the selected pair (vis-loop replica ×
adopted per-type eff scale factors) matches measured on every cathode
channel, *including the railed ones* (fig 5):

| od | meas | pred shown in viewer | pred unmasked | meas/unmasked |
|---|---|---|---|---|
| 4* | 12764 | **0** | 15570 | 0.82 |
| 6* | 3773 | **0** | 4825 | 0.78 |
| 7* | 2462 | **0** | 4887 | 0.50 (ch1051 template, below) |
| 8* | 4738 | **0** | 6728 | 0.70 |
| 5/9/10/11/12 | — | = unmasked | — | 0.77–1.25 |

So there is **no per-flash signal-processing failure here** — the mismatch
decomposes into three real, independent defects:

## Root cause 1 — saturation mask zeroes the *displayed prediction*

With `use_saturation_flag` on, QLMatching removes rail-flagged channels from
the per-flash opdet mask (QLMatching.cxx:1316-17); the bundle prediction is
accumulated under that mask (:1445), and the calib dump writes this *masked*
`pred_pe` (:2878, :2892-94) while the measured `pe` is written unmasked
(:2803-05).  The ql_scan viewer never reads `flash["sat"]`, so it displays
meas 12764 vs pred 0.  **37% of auto-selected bundles** in the candle round
sit on sat-flagged flashes — this artifact is everywhere.

Per owner direction: the full predicted light must always be shown; the
saturation flag should only exclude channels from chi2/KS/LASSO.

## Root cause 2 — self-trigger coverage blindness (membrane XA + PMTs)

Cathode 10xx channels are full 468800-sample (7.5 ms) streams.  Membrane
20xx / PMT 30xx channels are **self-triggered 1024-sample (16.4 µs)
snippets** — each channel is live only ~5–30% of the readout (fig 4).  When
no snippet overlaps a flash window, the chain silently scores measured = 0.

Flash-57 correspondence is exact (T3): every membrane/PMT opdet with a
covering snippet has nonzero dump PE; every one without reads 0.0 — e.g. od1
(ch2030) reads 0 against a 28-PE prediction, od3 (ch2040/41) 0 against 37,
while covered od12 (ch2050) reads 438 vs 370 predicted.

Brightness scan over the 18-event round (T4): P(meas>0) for membrane opdets
is 0.27/0.32/0.32 for flashes of 1–3k/3–10k/>10k PE — **flat in brightness**
⇒ the zeros are readout duty cycle, not dim light.  ~70% of membrane/PMT
entries entering chi2/KS — and the 2026-07-14 per-type efficiency fits — are
fake zeros.  The adopted membrane ×1.655 and PMT ×0.352 scale factors are
therefore contaminated and must be refit once coverage is handled.

Across the 42 strict-crosser anchors (T5), membrane opdets 0/1/2/12/19 have
median meas/pred = 0.00 — entirely explained by coverage.

## Root cause 3 — broken SPE templates distort per-channel PE scales

In OpDecon's wiener-inspired branch the net PE normalization is
1/template-area (DC gain 1/H(0), OpDecon.cxx:391-93); OpHitFinder's
scale/spe_area (100/100) cancel.  A wrong template *area* is a wrong
measured-PE scale.  Outlier scan of `pdvd-spe-templates.json` (T6):

| channel | area (ADC·ticks) | peers | opdet | effect |
|---|---|---|---|---|
| 1051 | 8557 | cathode ~500–1300 | 7 | od7 PE ÷~2 (fig 2: same pulse, 1066 PE via ch1050's template vs 75 via ch1051's) |
| 2020 | **−1595** | membrane ~1200–2400 | 2 | decon DC gain < 0: pulse-window PE −40%, total area window-dependent/wrong-sign — PE meaningless (fig 3); its own 24 self-triggers are noise (sick channel) |
| 2011 | 3132 | sibling 2010 = 1221 | 0 | od0 half low ×2.6 |
| 3010 | 15935 | PMT ~900–2500 | 14 | PE ÷~10 |
| 3020 | 8479 | PMT ~900–2500 | 15 | PE ÷~5 |

These match the anchor table: od7 low with a *tight* band (0.77 [0.63,
1.03]), od14 pinned at ~0, od15 at 0.64.

## Why it hid

- The saturation **veto** used to zero the measured PE on exactly these
  bright channels — meas 0 vs pred 0 looked "consistent".  The keep-and-mark
  chain (2026-07-14) surfaced the real measured values, and the masked
  prediction became visible as a huge mismatch.
- The per-type factor fits aggregate Σmeas/Σpred at flash level, where
  covered channels dominate the numerator — the fake zeros depressed the
  ratio smoothly instead of breaking it.
- Template outliers hit single sub-channels of ganged pairs; the healthy
  sibling kept the opdet nonzero, hiding the scale error inside the pair sum.

## Fix (phased; each phase gated and committed separately)

1. **Dump/viewer**: QLMatching dumps the *unmasked* prediction (`pred_pe`)
   while chi2/KS/LASSO keep the masked one; ql_scan viewer marks sat-flagged
   channels and excludes them from the ratio panel (toolkit + wcp commits).
   **DONE — toolkit af0a0284** (knob-off dumps proven identical field-by-field
   old-vs-new code, 198 flashes / 5171 bundles, setarch -R, reference tags
   `work/039252_0_p1{on,off}{old,new}`; knob-on: fit metrics identical,
   pred_pe gains the full prediction on exactly the 2303 sat-flagged
   entries — flash 57 ods 4/6/7/8 now dump 15570/4825/4887/6728) **+ wcp
   viewer commit** (orange rings/triangles for sat channels, ratio panel
   excludes them; session-tested on port 5021 scratch).  NOTE for A/B
   bookkeeping: owner commit 3c30cf58 (static ch_mask += {2,16,17,33})
   landed between the `arcal` round and these gates — dumps produced before
   it carry small nonzero pred on ch 2/17/33; not a code effect.
2. **Coverage chain**: OpHitFinder `emit_coverage` → coverage tensor →
   OpHitMerge passthrough → OpFlashFinder `flash_cov` companion tensor →
   optical PCs → `Opflash::get_cov` → QLMatching `use_coverage_flag` /
   `coverage_min` masking + dump `cov` array; viewer "no data" marker.  All
   default-OFF, byte-identical off; PDVD runners turn them on.
   **DONE — toolkit commit (flash/aux/clus/match) + wcp runner/viewer
   commit.**  Gates: compiled-config knob-off byte-identical for
   `wct-light-reco` and `wct-clustering` (keys present on);  PDVD light
   knob-off hash PASS vs the pre-change `_satrep` archive
   (`work/039252_light298567_p2off`); PDVD QL knob-off dump byte-identical
   vs the Phase-1 binary (`039252_0_p2qloff` == `039252_0_p1onnew`); PDHD
   all-PD light hash PASS (`029107_allpd1015_p2covgate` vs reference);
   wcdoctest flash/match/aux/clus.  Knob-on smoke (`_p2cov` /
   `039252_0_p2qlcov`): `flash_cov` for gid57 reproduces the raw snippet
   coverage channel-by-channel (od1/od3/od12 = 0, min-over-subchannel
   rule); the true crosser bundle (57, top:56) chi2 drops **459.6 → 91.8**
   (ndf 18→13, KS 0.120→0.088) once the fake zeros leave the fit; every
   flash carries some uncovered channel; 54/108 auto-selections move
   (intended knob-on physics change → Phase-4 refit).  Coverage semantic:
   an OpDet counts covered only when ALL its ganged DAPHNE sub-channels
   cover the window (od12's half-covered pair is masked); dead-RO OpDets
   24/27/28/34 have no mapped raw channel ⇒ cov 0 (already static-masked).
3. **SPE templates v2**: validated re-harvest (positivity, per-type area
   band, sibling-transfer for channels without a clean 1-PE peak);
   `pdvd-spe-templates-v2.json` behind a `spe_file` selector defaulting to
   v1; runner selects v2.
4. **Reprocess + refit**: 120-event reprocess at the new operating point,
   coverage-aware refit of the per-type efficiency factors (owner gate before
   adoption), fresh candle round on port 5019.

## Verification

- This doc's script regenerates every table/figure from the existing dumps
  (read-only).  Figures:
  - `lightpattern_fig1_raw_decon_cathode.png` — raw vs decon, railed +
    unrailed cathode channels.
  - `lightpattern_fig2_od7_template_bug.png` — ch1050/ch1051 same light,
    ×7 template-area bug.
  - `lightpattern_fig3_ch2020_inversion.png` — negative-area template
    corrupting the decon PE normalization.
  - `lightpattern_fig4_coverage_timeline.png` — snippet livetime vs the
    flash window; covered ⇔ measured>0.
  - `lightpattern_fig5_pattern.png` — measured vs masked vs unmasked
    prediction for flash 57.
- Smoke case for all fixes: evt 298567 flash gid 57 (od4/6/7/8 sat-marked
  with full pred shown; od1/od3 coverage-masked; od7 ×~1.9 after template
  v2).
- Status: knobs-off paths must stay **byte-identical** (hash_archive gates
  per detector); the knob-on operating point is **NOT bit-identical** by
  design and triggers the Phase-4 revalidation + factor refit.

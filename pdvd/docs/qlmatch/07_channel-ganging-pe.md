# PDVD X-Arapuca two-channel ganging: one predicted light, two different measured PE

**Repro**

```bash
cd wcp-porting-img/pdvd
python3 docs/qlmatch/scripts/channel_ganging_pe.py \
    input_data_light/np02vd_raw_run039252_1176_df-s03-d3_dw_0_20250830T054542_rawwf.root
```

## Question

Each PDVD X-Arapuca is read out by **two DAPHNE OpChannels** (a SiPM pair),
e.g. `1010`/`1011`, `1020`/`1021`. Two things were asked:

1. Is the **predicted light** (photon library — what we sim/predict) always the
   same for the two channels of a pair?
2. In the **data**, is the **measured PE** of the two channels the same?

Short answers: **(1) yes, identical by construction; (2) no, they differ — and
both channels are combined at the OpDet level before QLMatching.**

## The ganging: two channels → one OpDet

`OpFlashFinder` gangs the 51 live DAPHNE OpChannels into the **40 OpDet** PE
columns via
`toolkit/cfg/pgrapher/experiment/protodunevd/pdvd-opch-map.json`. The X-Arapucas
(cathode `10xx`, membrane `20xx`) map two channels to one OpDet; PMTs (`30xx`)
are one channel per OpDet:

| opch pair | OpDet | opch pair | OpDet |
|---|---|---|---|
| 1010, 1011 | 6 | 1050, 1051 | 7 |
| 1020, 1021 | 4 | 1060, 1061 | 5 |
| 1030, 1031 | 8 | 1070, 1071 | 9 |
| 1040, 1041 | 10 | 1080, 1081 | 11 |

So **QLMatching never sees the DAPHNE channels** — it sees a 40-column
opflash and a 40-row library, both at **OpDet granularity**.

## (1) Predicted light — one value per OpDet (identical for the pair)

The photon library
`wire-cell-data/pdvd/photodet/pdvd-photlib-vis-v5-128nm.json` carries exactly
**one visibility row per OpDet** (`"nchan": 40`), and `PhotonLibraryModel`
(`match/inc/WireCellMatch/PhotonLibraryModel.h`) interpolates **one number per
OpDet** at each 3-D point. There is no per-DAPHNE-channel entry.

Therefore the two channels of a pair are not two predictions that happen to
agree — they are **a single OpDet entry**. Attributed back to the channels, the
predicted light for `1010` and `1011` is *identical by construction* (same
OpDet, same library row, same detection efficiency), and likewise for every XA
pair. This holds in every mode QLMatching runs (`light_model: 'library'`).

## (2) Measured PE — the two channels differ, then are summed

In the data the two DAPHNE channels are physically distinct SiPM readouts and
their measured charge is **not** equal. `OpFlashFinder` **sums** them into the
OpDet PE column (positive polarity, per the opch-map).

Measured directly from the raw full-stream waveforms of run 039252 (18 events),
integrating each channel's pulse charge above a per-channel median baseline
(noise-thresholded), the within-pair ratio A/B is:

| pair → OpDet | A/B ratio (18 evts) | both saturate? |
|---|---|---|
| 1010/1011 → 6 | **1.231 ± 0.023** (1.19–1.28) | yes |
| 1020/1021 → 4 | **0.760 ± 0.013** (0.74–0.78) | yes |
| 1030/1031 → 8 | 1.029 ± 0.056 | yes |
| 1040/1041 → 10 | 0.975 ± 0.026 | yes |
| 1050/1051 → 7 | 1.004 ± 0.047 | yes |
| 1060/1061 → 5 | **0.755 ± 0.017** | yes |
| 1070/1071 → 9 | **0.873 ± 0.011** | yes |
| 1080/1081 → 11 | 1.025 ± 0.014 | yes |

The two SiPM channels of one cathode X-Arapuca differ by up to **~24%**, and the
ratio is **stable to ~2%** across 18 events (each a different cosmic pattern) —
i.e. a **fixed per-channel property** (different baselines, gain/SPE, and/or
light-collection asymmetry between the two SiPM sub-cells), not random
fluctuation.

**Membrane XAs (`20xx`) are far more asymmetric still** — they are 1024-tick
**self-trigger snippets**, so on a given event often only one of the two
channels crosses its self-trigger threshold. Their within-pair ratios swing over
orders of magnitude (0 → hundreds) event to event. The clean, quantitative
statement above comes from the cathode full-stream pairs, where both channels
are always present.

**Saturation caveat.** On bright cosmics both cathode channels rail at the
14-bit ADC ceiling (16383) — e.g. evt 298651, `1020` saturates 323 samples,
`1021` 523 — so the integrals are lower bounds. This does not change the
conclusion (the brighter channel is even brighter than the ratio shows); see
`project_run29107_evt1015_adc_saturation` for the DAPHNE-rail context.

## Summary: both sides meet at the OpDet

| | granularity into QLMatching | two channels of a pair |
|---|---|---|
| **Predicted light** (library) | per OpDet (40 rows) | a *single* entry → identical |
| **Measured PE** (data opflash) | per OpDet (40 cols) | two different channels **summed** |

So yes — the per-channel PE is combined into the OpDet **before QLMatching** on
both sides: the prediction is inherently one value per OpDet, and the data sums
the two genuinely-different channel measurements into that same OpDet column.
QLMatching then compares 40 predicted vs 40 measured. This is why the two
channels are *ganged* rather than treated as independent measurements of the
same quantity — in the data they are not the same quantity.

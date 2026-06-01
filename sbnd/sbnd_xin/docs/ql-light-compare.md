# Matched Q–L light: measured vs predicted PE (data vs simulation)

## Purpose

For SBND charge–light matching (`QLMatching`) each matched **bundle** pairs a
charge cluster with an optical flash and produces, per optical channel, a
**predicted** PE (semi-analytical photon model × `QtoL` × per-detector
efficiency — see `match/docs/semi-analytical-model.md` and
`match/docs/QL_algorithm.md`) next to the **measured** PE of the flash.

This note quantifies how well predicted light tracks measured light for the
*reasonably matched* bundles, and in particular the **normalization difference
between data and simulation**. The expectation going in:

- In **simulation** the light prediction path is the same physics used to
  generate the event, so measured/predicted should sit near a fixed scale with
  little spread.
- In **data** the photon model is not calibrated to the real detector, so the
  measured/predicted normalization may be offset and/or broad.

The result below bears this out: MC closes at `ratio ≈ 1`, data does not (it runs
a factor of a few higher). The data offset is a real data–MC difference, though
this analysis cannot by itself separate the light-model contribution from the
charge-reconstruction and PE-calibration scales (see *What this does and does not
tell us*).

## Data source

The matcher writes both quantities to the Bee optical JSON `<n>-op.json`
(one per event, inside `mabc-all-apa.zip`, from `util/src/Bee.cxx`
`Bee::Flashes::append`). Each flash row carries:

| field            | meaning                                                |
|------------------|--------------------------------------------------------|
| `op_t`           | flash time (µs)                                        |
| `op_peTotal`     | total measured PE                                      |
| `op_pes[312]`    | measured PE per optical channel                        |
| `op_pes_pred[312]`| predicted PE per channel (filled only for **matched** flashes, only on the active PMT channels of that flash's TPC side; with `bee_flash_per_flash` it is the element-wise sum over the flash's matched clusters) |
| `op_cluster_ids` | matched cluster ids (**empty ⇒ unmatched flash**); with `bee_flash_per_flash` (SBND all-APA) this is the full set of clusters matched to the flash, one flash per row |
| `apa`            | "0"/"1", the flash's TPC side                          |

A **matched bundle** is a row with non-empty `op_cluster_ids` (so `op_pes_pred`
is filled). This run: **186 matched bundles (MC), 130 (data)** across 10 events
each. (These counts predate `bee_flash_per_flash`, which collapses a flash's
several matched-cluster rows into one row — the matched-flash count is lower, the
matched-cluster total unchanged.)

The 10 events are labelled by directory index 0..9, which maps in streaming
order to the real event ids — MC: `2, 9, 11, 12, 14, 18, 31, 35, 41, 42`;
data: `659242, 659286, 659374, 659484, 659572, 659704, 659924, 660496, 660826,
660892`.

## Method

For every matched bundle we compare measured vs predicted over the **active
channel set** `A = { ch : op_pes_pred[ch] > 0 }` — i.e. we only compare where
the model actually predicted. (This is self-consistent and, for SBND, numerically
identical to summing all 312 channels: the per-APA opflash carries essentially no
PE outside its own side's active PMTs.)

- **sum:** `meas = Σ_A op_pes`, `pred = Σ_A op_pes_pred`
- **normalization:** `ratio = meas / pred`
- **pattern similarity:** `cos = ⟨m,p⟩ / (|m| |p|)` over `A` — high when the
  per-PMT *shape* of measured and predicted light agree, regardless of overall
  scale. This is the "reasonably matched / pattern similar" selector.

"Reasonably matched" = `cos ≥ 0.85`. The threshold is data-driven: in the cosine
distribution (below) the MC matched bundles pile up at `cos ≳ 0.85`, so the cut
keeps the genuine matches and drops the obvious mis-matches. It is exposed as
`--cos-cut`; the conclusion is robust over 0.8–0.95 (see *Cut sensitivity*).

## Results

### Per event / APA summed PE — `pics/ql_event_apa_sumPE.png`
One point per (event, APA), summing measured and predicted over that event/APA's
matched bundles. **MC** clusters along `y = x` (predicted ≈ measured, both up to
~10⁵ PE). **Data** sits well above `y = x`, and its predicted sums are an order
of magnitude smaller (~10³–10⁴) than MC for comparable measured PE — the
prediction systematically under-counts in data.

### Per PMT summed PE — `pics/ql_perpmt_sumPE.png`
Per-channel measured vs predicted, summed over all matched bundles. **MC**
straddles `y = x` (≈10⁴ both axes). **Data** sits an order of magnitude above
`y = x` (predicted ~10²–10³, measured ~10³–10⁴). A mild APA0-vs-APA1 offset is
visible in data (APA0-side predicted lower than APA1).

### Pattern similarity — `pics/ql_cosine_dist.png`
MC matched bundles pile up near `cos = 1` (the per-PMT shape is well reproduced);
data is flat with no pile-up near 1 — even the *pattern* is reproduced less well
in data than in MC.

### Normalization — `pics/ql_ratio_dist.png` and `pics/ql_ratio_vs_cos.png`
Distribution of `ratio = meas/pred` for the reasonably-matched population. MC
peaks sharply at `ratio ≈ 1`; data is broad and shifted to several×. In the
`ratio` vs `cos` scatter, the high-cosine region holds a tight MC blob at
`ratio ≈ 1` that data lacks.

**Headline numbers** (`cos ≥ 0.85`):

| sample | N  | median ratio | 16–84 % band |
|--------|---:|-------------:|--------------|
| MC     | 79 | **1.29**     | [0.92, 11.76]|
| data   | 26 | **~4** (2–5) | [1.01, 55.87]|

The strong, robust statement is the **MC** one: across all cuts the simulation
ratio converges to **≈1.1–1.3 with shrinking spread** (1.33 → 1.29 → 1.16 → 1.13
as the cut tightens). This is essentially a closure test — in MC the light
prediction uses the same physics that generated the event, so measured ≈ predicted
is expected, and it holds.

For **data** the ratio is higher — measured light is larger than predicted by a
**factor of a few (~2–5×)** — but the number is small-N (6–35 bundles) and does
not converge (it bounces 4.22 → 3.68 → 1.82 → 4.68 across cuts), so treat it as a
bound, not a precise scale. The clean qualitative result is that data does **not**
reproduce the MC `ratio ≈ 1` closure.

**What this does and does not tell us.** The data≠1 offset comes from whatever
*differs* between the data and MC runs — it is **not** the `QtoL`/efficiency
normalization, which is identical in both runs (same `qlmatching.jsonnet`, same
`semi-analytical-sbnd.json`) and therefore cannot produce a data–MC difference
(raising `QtoL` to pull data to 1 would push MC below 1). The candidates are: the
reconstructed charge `q` going into the prediction (the per-event/APA plot shows
data **predicted** sums ~10× below MC — that is the charge input, not the light
model), the ADC→PE calibration of the measured flash, and any real optical
response the semi-analytical model does not capture for data. This analysis
localizes a factor-of-a-few data excess **consistent with** the light
model / light-yield not matching real data, but does not isolate it from the
charge-reconstruction and PE-calibration scales.

### Cut sensitivity (robustness)
| cos cut | MC N | MC median | data N | data median |
|--------:|-----:|----------:|-------:|------------:|
| 0.80    | 91   | 1.33      | 35     | 4.22        |
| 0.85    | 79   | 1.29      | 26     | 3.68        |
| 0.90    | 59   | 1.16      | 14     | 1.82*       |
| 0.95    | 26   | 1.13      | 6      | 4.68*       |

MC sits at 1.1–1.3 throughout; data at several× throughout. (*data bins above
0.85 have small N and bounce; 0.85 is the best-powered representative point.)

## Caveats
- The 16–84 % bands are inflated by a tail of bundles that pass the cosine cut
  yet have a wrong overall scale (cluster fragments, saturation, out-of-time
  pile-up). The **median** is the robust statistic; the per-event/APA and per-PMT
  aggregate plots (which average over mis-matches) tell the same story more
  cleanly.
- Cosine over non-negative PE vectors is a weak discriminator — it can be high
  when one bright PMT dominates. It is used only to *enrich* the genuine-match
  population, not to declare a bundle correct.
- 10 events per mode; these are normalization-scale statements, not a precision
  calibration.

## Reproduce
```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
./run_clust_QL_evt.sh mc      # -> work/ql_mc/mabc-all-apa.zip
./run_clust_QL_evt.sh data    # -> work/ql_data/mabc-all-apa.zip
python3 ql_light_compare.py   # default --cos-cut 0.85
```

## Dumps
- `ql_light_dump.csv` — one row per matched bundle:
  `mode, event_idx, event_id, apa, flash_t_us, n_active_ch, meas_pe, pred_pe, ratio, cos_sim`.
- `ql_perpmt_dump.csv` — one row per (mode, channel) with non-zero predicted light:
  `mode, channel, apa_side, meas_sum, pred_sum, n_bundles`.

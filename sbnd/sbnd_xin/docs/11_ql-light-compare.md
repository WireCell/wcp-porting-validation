# Matched Q–L light: measured vs predicted PE (data vs simulation)

## Purpose

For SBND charge–light matching (`QLMatching`) each matched **bundle** pairs a
charge cluster with an optical flash and produces, per optical channel, a
**predicted** PE (semi-analytical photon model × `QtoL` × per-detector
efficiency — see `match/docs/semi-analytical-model.md` and
`match/docs/QL_algorithm.md`) next to the **measured** PE of the flash.

> Note: the 24 uncoated central PMTs now carry `VUVEfficiency = 0` (only
> `VISEfficiency = 0.0357`), so their predicted PE comes from the reflected
> visible component alone — matching the sbndcode simulation, which never
> digitizes direct VUV on uncoated channels. See
> `match/docs/sbnd-pmt-efficiencies.md`.

This note quantifies how well predicted light tracks measured light for the
*reasonably matched* bundles, and in particular the **normalization difference
between data and simulation**. The expectation going in:

- In **simulation** the light prediction path is the same physics used to
  generate the event, so measured/predicted should sit near a fixed scale with
  little spread.
- In **data** the photon model is not calibrated to the real detector, so the
  measured/predicted normalization could be offset and/or broad.

On this data sample (`input-1file-data-v10_14_02_02`) the result is that **both
close near 1**: MC sits at `ratio ≈ 1.0` and data at `≈ 0.86` (measured ~14 %
below predicted), each with a tight spread and a strong pattern (cosine) pile-up
near 1. The data photon prediction now tracks the measured light well — a marked
change from an earlier data sample (run-659xxx) on which data ran a factor of a
few *high*; the data–MC normalization is therefore production-dependent, not a
fixed property of the matcher. Any residual data–MC difference this analysis sees
cannot by itself be separated into light-model, charge-reconstruction, and
PE-calibration contributions (see *What this does and does not tell us*).

## Data source

The matcher writes both quantities to the Bee optical JSON `<n>-op.json`
(one per event, inside `mabc.zip`, from `util/src/Bee.cxx`
`Bee::Flashes::append`). Each flash row carries:

| field            | meaning                                                |
|------------------|--------------------------------------------------------|
| `op_t`           | flash time (µs)                                        |
| `op_peTotal`     | total measured PE                                      |
| `op_pes[312]`    | measured PE per optical channel                        |
| `op_pes_pred[312]`| predicted PE per channel (filled only for **matched** flashes, only on the active PMT channels of that flash's TPC side; with `bee_flash_per_flash` it is the element-wise sum over the flash's matched clusters) |
| `op_cluster_ids` | matched cluster ids (**empty ⇒ unmatched flash**); with `bee_flash_per_flash` (SBND all-APA) this is the full set of clusters matched to the flash, one flash per row |
| `apa`            | "0"/"1", the flash's TPC side                          |
| `op_flash_group` | (SBND all-APA, `flash_group_window>0`) ±80 ns flash-flash coincidence id shared by TPC0/TPC1 flashes; absent ⇒ no grouping. Display-only — does not affect the analysis below. |

A **matched bundle** is a row with non-empty `op_cluster_ids` (so `op_pes_pred`
is filled). This run: **103 matched bundles (MC), 101 (data)** across 10 events
each (96 active PMT channels each). With `bee_flash_per_flash` (SBND all-APA) a
flash's several matched-cluster rows are collapsed into one, so a matched bundle
here is one matched flash.

The 10 events are labelled by directory index 0..9, which maps in streaming
order to the real event ids — MC: `2, 9, 11, 12, 14, 18, 31, 35, 41, 42`;
data: `686, 1258, 1302, 1346, 1698, 1720, 1808, 1852, 2028, 2050`.

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
matched bundles. **Both** MC and data cluster along `y = x` (predicted ≈ measured,
both ~10⁴–2×10⁵ PE). Data sits right on the diagonal — its predicted sums are now
comparable to MC for comparable measured PE, with no systematic under-count.

### Per PMT summed PE — `pics/ql_perpmt_sumPE.png`
Per-channel measured vs predicted, summed over all matched bundles. Both **MC**
and **data** straddle `y = x` (≈10⁴ both axes). The same APA0-vs-APA1 split shows
in both: the APA0-side channels form a lower cloud (~6×10³) than the APA1-side
(~10⁴–2×10⁴), but both lie along the diagonal.

### Pattern similarity — `pics/ql_cosine_dist.png`
Both MC and data matched bundles pile up near `cos = 1` (the per-PMT shape is well
reproduced), with a small low-cosine tail in each. Data now tracks MC closely —
the *pattern* of the light is reproduced about as well in data as in MC.

### Normalization — `pics/ql_ratio_dist.png` and `pics/ql_ratio_vs_cos.png`
Distribution of `ratio = meas/pred` for the reasonably-matched population. Both MC
and data peak near `ratio ≈ 1`; MC sits right at 1, data slightly below (~0.86),
both with comparable, modest spread. In the `ratio` vs `cos` scatter the
high-cosine region holds a tight blob near `ratio ≈ 1` in **both** samples.

**Headline numbers** (`cos ≥ 0.85`):

| sample | N  | median ratio | 16–84 % band |
|--------|---:|-------------:|--------------|
| MC     | 65 | **1.01**     | [0.78, 1.90] |
| data   | 63 | **0.86**     | [0.69, 1.38] |

Both samples now pass a closure-like test. **MC** converges to `ratio ≈ 1.0–1.07`
across all cuts (1.07 → 1.01 → 1.01 → 1.07) with shrinking spread — the expected
result, since in MC the light prediction uses the same physics that generated the
event. **Data** converges to `ratio ≈ 0.86–0.99` (0.87 → 0.86 → 0.87 → 0.99),
i.e. measured light ~14 % below predicted and stable across cuts. The clean
qualitative result is that on this sample data **does** reproduce the MC closure
to within ~15 %, with comparable N (data 63 vs MC 65 at `cos ≥ 0.85`).

**What this does and does not tell us.** On this sample the data–MC normalization
difference is small (data ≈ 0.86 vs MC ≈ 1.0, ~15 %), so there is no longer a
large offset to attribute. The `QtoL`/efficiency normalization is identical in
both runs (same `qlmatching.jsonnet`, same `semi-analytical-sbnd.json`) and so by
construction cannot create a data–MC difference; the residual ~15 % comes from
whatever *differs* between the two runs — the reconstructed charge `q` going into
the prediction, the ADC→PE calibration of the measured flash, and any real optical
response the semi-analytical model does not capture for data. This analysis shows
that, for this data production, those effects largely cancel into a near-unity
ratio; it does not isolate the individual contributions, and the result is
sample-dependent (an earlier data sample ran several× high).

### Cut sensitivity (robustness)
| cos cut | MC N | MC median | data N | data median |
|--------:|-----:|----------:|-------:|------------:|
| 0.80    | 73   | 1.07      | 73     | 0.87        |
| 0.85    | 65   | 1.01      | 63     | 0.86        |
| 0.90    | 54   | 1.01      | 45     | 0.87        |
| 0.95    | 26   | 1.07      | 22     | 0.99        |

MC sits at ≈1.0–1.07 throughout; data at ≈0.86–0.99 throughout, rising toward 1 as
the cut tightens. Both are stable and well-powered (N stays comparable between
samples), so 0.85 is a representative point.

## Caveats
- The 16–84 % bands carry a tail of bundles that pass the cosine cut yet have a
  wrong overall scale (cluster fragments, saturation, out-of-time pile-up). The
  **median** is the robust statistic; the per-event/APA and per-PMT
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
./run_clust_QL_evt.sh mc      # -> work/ql_mc/mabc.zip
./run_clust_QL_evt.sh data    # -> work/ql_data/mabc.zip
python3 scripts/analysis/ql/ql_light_compare.py   # default --cos-cut 0.85
```

## Dumps
- `ql_light_dump.csv` — one row per matched bundle:
  `mode, event_idx, event_id, apa, flash_t_us, n_active_ch, meas_pe, pred_pe, ratio, cos_sim`.
- `ql_perpmt_dump.csv` — one row per (mode, channel) with non-zero predicted light:
  `mode, channel, apa_side, meas_sum, pred_sum, n_bundles`.

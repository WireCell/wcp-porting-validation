# PDVD flash time-coincidence (Δt) across the four PD groups

Milestone 3 of the light-reconstruction plan (`02_light-reco-plan.md` §5/§6): measure how
tightly the four PD populations — top-wall membrane XA, bottom-wall membrane XA, cathode XA,
bottom PMTs — see the same physical activity in time, to set the OpFlashFinder coincidence
bin (`bin_width`; PDHD uses 1000 ns) and check for per-population time offsets.

> Script: `pdvd/pd_plot/flash_dt.py`. Figure: `docs/pics/flash_dt.png`.
> Run 039252, all 18 events; ~10–16 k nearest-neighbour pairs per population pair.

## The common clock (verified, load-bearing for the whole chain)

The `timestamp` branch of the rawwf tree is in **microseconds** on one common clock for both
readout modes, quantized at the DAPHNE tick (16 ns, with 8 ns substructure):

- the cathode full-stream record **starts at its timestamp** (all 16 DAPHNE channels of an
  event share it);
- snippet timestamps are **trigger times on the same clock**: clusters of self-trigger
  timestamps line up with the bright cathode-stream pulses at the same microsecond values;
- self-triggers span the full 7.5 ms stream window (−16 µs … +7570 µs relative to stream
  start), i.e. the two readout modes cover the same window.

Hit times used here: snippets → `timestamp + (decon spike − 64) × 16 ns` (64 = nominal
trigger sample; raw peak sits at ~76); stream → `timestamp + sample × 16 ns`.

## Result

Nearest-neighbour Δt between population hits (reference: cathode hits ≥ 3 PE — the
double-sided cathode XAs see both volumes and anchor everything; other populations ≥ 0.7 PE):

| pair | n | median (µs) | 68 % half-width (µs) | 95 % half-width (µs) |
|---|---|---|---|---|
| membrane_top − cathode | 10 392 | +0.000 | 0.080 | 3.19 |
| membrane_bot − cathode | 15 646 | +0.000 | 0.496 | 3.77 |
| pmt − cathode | 12 405 | +0.000 | **0.032** | 0.19 |
| membrane_bot − pmt | 9 675 | +0.064 | 0.120 | 2.38 |
| membrane_bot − membrane_top | 11 062 | +0.032 | 0.592 | 4.13 |

All pairs show a sharp coincidence spike at zero (`flash_dt.png`):

- **Per-population systematic offsets are ≤ 64 ns** (4 ticks) — no per-type time-offset
  correction is needed at flash-binning scale. (The small +64 ns membrane−pmt shift is the
  slow-XA-rise vs fast-PMT decon alignment.)
- **68 % cores are within ±0.6 µs** for every pair; the tightest (pmt−cathode, ±32 ns) shows
  the intrinsic clock quality.
- The few-µs **tails are physics, not clock**: self-triggers fired by LAr late light
  (triplet ~1.6 µs) trailing the prompt hit, plus flat random dark-count coincidences.

## Recommendation

**`bin_width = 1000 ns`** — identical to PDHD. The double-offset accumulator with 1 µs bins
comfortably contains every measured 68 % core; the late-light tails should NOT be absorbed
into a wider bin (that would merge unrelated activity in these busy 7.5 ms windows —
~2 000 self-triggers per event) but are handled downstream by the existing OpFlashFinder
machinery (`refine_hits_in_flash` width extension, `remove_late_light`, `flash_refine`).
The single all-PD flash (plan §5, option 1) is consistent with these measurements: the
cathode XAs are genuinely coincident with **both** volumes at tick level.

Follow-up at chain-assembly time: `TRIG_SAMPLE = 64` (snippet trigger sample) should be
confirmed against the decoder/DAPHNE configuration; it shifts all snippet hits by a common
constant and only matters for the absolute light↔charge alignment (`offset_us`), not for
flash forming.

## Reproduce

```bash
cd pdvd
python3 pd_plot/spe_build.py 0     # harvest cache, if absent
python3 pd_plot/flash_dt.py 0
```

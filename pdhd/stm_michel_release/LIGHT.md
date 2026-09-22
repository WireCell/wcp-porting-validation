# The light: flashes, OpHits and waveforms

The photon-detector side of each event is in three trees of the per-event ROOT file.  All
of it comes from the Wire-Cell optical reconstruction that the charge-light matching used:
raw waveforms → Wiener-type deconvolution per channel (`decon`, in PE per sample) →
[ROI cleaning on the continuous-stream channels (`decon_roi`)] → OpHit finding per channel →
OpFlash finding over all photon detectors in 1 µs bins.

## 1. Time axes — read this first

There are two clocks.

* **The light axis** (`T_flash.time_us`, `T_ophit.*_time_us`, `T_opwf.t0_us`, `T_stm.t0_us`,
  `T_stm.flash_time_us`): the optical reconstruction's own tick origin, trigger-relative,
  in µs.  Everything optical in this file is on this axis, so flashes, hits and waveform
  samples compare directly.
* **The charge axis** (the drift time of the TPC readout, tick = 0.5 µs, `T_stm_2d.tick`,
  `T_stm_fit.pt * nticks_per_slice`): add the trigger offset of `T_event` to a light-axis time
  to land on it.  PDHD: `trigger_offset_us` (one value per event, ≈ 250 µs).  PDVD: per
  drift volume, `trigger_offset_bot_us` / `trigger_offset_top_us` (≈ −2500 µs, the TDE/BDE
  crates open their windows up to ~30 µs apart).  `T_flash.time_charge_bot_us` /
  `time_charge_top_us` already carry the sum (on PDHD both columns are the same).
  The drift coordinate x of the reconstructed points was placed with the matched flash's
  time, so the charge-side geometry is already consistent with the flash.

The Michel electron's light arrives 0.1–10 µs after the muon's, inside the same 1 µs-binned
flash or in the next one: the waveform window covers it (§3).

## 2. `T_flash` and `T_ophit`

`T_flash` has **every** flash of the event (not only the matched ones), one row each,
`flash_id` = the row index used by `T_stm.flash_id`, `T_ophit.flash_id`, `T_opwf.flash_id`.

| branch | meaning |
|---|---|
| `time_us` | flash time on the light axis (the seed bin of the 1 µs accumulator; the first OpHit of a bright flash can peak up to ~1 µs earlier) |
| `time_charge_bot_us`, `time_charge_top_us` | the same on the charge axis (§1) |
| `total_pe` | summed PE over the photon detectors |
| `pe` | vector, PE per **OpDet** (index = OpDet number: PDHD 0..159, PDVD 0..39) |
| `y_center_mm`, `z_center_mm`, `y_width_mm`, `z_width_mm` | PE-weighted flash position and spread (mm, as in the archive) |
| `nhits` | OpHits in the flash |
| `n_matched_clusters`, `matched_cluster_ids` | the charge clusters the charge-light matching assigned to this flash (their t0 is this flash) |
| `is_stm_flash`, `stm_cluster_ids` | 1 when one of them is an STM candidate (a `T_stm` row); which |
| `sat`, `cov` | PDVD only, per OpDet: the channel railed (saturated) during the flash; the fraction of the flash window a self-triggered channel actually recorded (0 = no snippet, so PE = 0 means "not measured") |

Which OpDet is where: PDHD's 160 OpDets are the X-ARAPUCA bars of the four APAs, 40 per
APA; OpDets 0–119 are read out as self-triggered snippets and 120–159 as a continuous
stream.  Their positions are in the toolkit's `cfg/pgrapher/experiment/pdhd/pdhd-opdet-geom.json`.
PDVD's 40 OpDets are the cathode X-ARAPUCAs (OpDet 0–15, DAPHNE channels 10xx, two
channels per OpDet), the membrane X-ARAPUCAs (20xx) and the PMTs (30xx, one channel each);
the channel → OpDet map is `pdvd-opch-map.json` in `cfg/pgrapher/experiment/protodunevd/`.

`T_ophit`: the OpHits of the flashes matched to STM candidates only (`is_stm_flash == 1`),
one row per pulse: `channel` (readout channel: PDHD = OpDet; PDVD = DAPHNE channel),
`opdet`, `peak_time_us`, `start_time_us`, `width_us`, `area`, `amplitude`, `pe`,
`fast_to_total` (the prompt fraction of the pulse).  A hit may be one sub-pulse of a split
pulse (PDHD splits merged pulses).

## 3. `T_opwf` — the waveform windows

For every flash matched to an STM candidate and **every readout channel of the detector**,
one row with the raw and the deconvolved samples in the window
`[flash time − wf_pre_us, flash time + wf_post_us]` (2 µs before, 10 µs after by default;
the values used are in `T_event`).

| branch | meaning |
|---|---|
| `flash_id`, `cluster_id` | the flash and (the first) STM candidate matched to it |
| `channel`, `opdet`, `branch` | the readout channel, its OpDet, and which reconstruction branch it came through (PDHD: `allpd-snip` = self-triggered snippets, OpDets 0–119; `allpd-fs` = continuous stream, OpDets 120–159. PDVD: `cath` continuous stream, `mem`, `pmt` self-triggered snippets) |
| `t0_us`, `tick_ns`, `n` | the window start on the light axis (= flash time − `wf_pre_us` exactly), the sample spacing (16 ns), the number of samples (750 for the default 12 µs); sample i is at `t0_us + i * tick_ns / 1000`, taken from the nearest recorded tick (≤ 8 ns off); samples the recording does not reach are 0 |
| `raw` | the raw ADC samples (pedestal **not** subtracted; 14-bit DAPHNE, PDHD baseline ≈ 8200) |
| `decon` | the deconvolved samples, **PE per sample**: summing `decon` over a pulse gives its PE, the same scale as `T_ophit.pe` and `T_flash.pe` |
| `decon_roi` | continuous-stream branches only (PDHD `allpd-fs`, PDVD `cath`): the deconvolved trace after the ROI cleaning the OpHit finder ran on; empty otherwise |
| `n_raw_nonzero` | how many of the `n` raw samples are real: self-triggered channels only record ~16 µs snippets around a pulse, and outside a snippet the dense frame is exactly 0 (raw) — treat those samples as "not recorded", not as ADC 0 |
| `pe` | the flash's PE on this OpDet (copied from `T_flash.pe`) |

Saturation: a railed pulse (ADC 16383 on PDHD) is flagged and, on PDHD, its hit vetoed; on
PDVD the railed run is bridged before deconvolution and the flash's `sat` vector marks the
OpDet.  The raw samples show the rail as it was recorded.

## 4. Examples

`scripts/plot_flash_waveforms.py --event <run6>_<evtid> --cluster <id>` draws the PE
pattern of the matched flash and the raw / deconvolved waveforms of the brightest photon
detectors with the OpHit peak times overlaid.  In code:

```python
import stm_release as sr, numpy as np
ev = sr.open_event("events/029107_1135/stm_michel_pdhd_029107_1135.root")
c = sr.candidates(ev, ["cluster_id", "is_stm", "michel_found", "flash_id"])
fid = int(c["flash_id"][0])
fl = sr.flash(ev, fid)                         # fl["pe"] : PE per OpDet
hits = sr.ophits(ev, flash_id=fid)             # every pulse of that flash
w = sr.waveform(ev, fid, channel=int(np.argmax(fl["pe"])))
t = w["t_us"] - fl["time_us"]                  # time relative to the flash
michel_pe = w["decon"][(t > 0.5) & (t < 8)].sum()   # e.g. the late light on this PD
```

## 5. How the light was reconstructed (for reference)

PDHD: the 120 self-triggered OpDets (1024-sample snippets) and the 40 continuous-stream
OpDets (343 808 samples) are deconvolved with the same fixed Wiener filter and the 2024
average SPE templates; the continuous-stream branch gets an ROI cleaning (high-pass +
threshold seeds); OpHits with peak splitting; one OpFlashFinder over all 160 OpDets,
grouped per cathode side, 1 µs bins, minimum 5 fired PDs and 20 PE.  PDVD: three branches
(cathode continuous stream 468 864 samples, membrane and PMT snippets) with the
Wiener-inspired filter (sigma 1.25 / 1.0 / 3.5 MHz), the v2 SPE templates, saturation
repair and flagging, coverage rows, one OpFlashFinder over the 40 OpDets (minimum 2 PDs,
10 PE).  The production jobs are `cfg/pgrapher/experiment/pdhd/wct-light-allpd-reco.jsonnet`
and `cfg/pgrapher/experiment/protodunevd/wct-light-reco.jsonnet` in the toolkit; the
waveform dump used for this release is their `frames_dir` option.

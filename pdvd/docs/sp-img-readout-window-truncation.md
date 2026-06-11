# PDVD run 39252 evt 298679: SP/imaging disagreement = imaging tick-window truncation

**Symptom.** For run 39252, event 298679 (work dir `work/039252_8`, file index 8),
anode 0, the Magnify SP display (`magnify-run039252-evt8-anode0-dnnroi.root`)
shows many clean deconvoluted tracks, but the imaging display shows far fewer
tracks.

**Root cause: `max_tbin: 8000` in the imaging slicing config, not channel
mapping and not a corrupted readout.**  Run 39252 was recorded with a
10000-tick readout; the PDVD imaging config hard-caps slicing at tick 8000, so
the last 2000 ticks (~16% of the SP charge, several whole tracks) never enter
tiling.

## Evidence

Per-plane overlay of SP activity (gauss, `protodune-sp-frames-anode0.tar.bz2`)
vs the channel/tick footprint of every blob in
`clusters-apa-anode0-ms-active.tar.gz` (blob `bounds` wire ranges mapped to
channels with the v5 wires file, slice `start/span` mapped to ticks):

![overlay](sp-img-tick-window-39252-anode0.png)

green = SP activity covered by a blob, red = SP activity with **no** blob,
dashed line = tick 8000.

* **Below tick 8000 imaging is healthy**: every SP track is traced by blobs in
  all three planes, on both faces (face ids 0 and 8).  Only 5–7% of SP-active
  samples below 8000 lack blob coverage (scattered single-sample edges, normal
  tiling granularity — not missing tracks).
* **Above tick 8000 coverage is exactly zero.**  The blob slice starts span
  ticks 176 → 7996 and stop dead at the cap.  Tracks crossing tick 8000 (e.g.
  the rising U track at plane-channel 60→260, ticks 7400→8800) are cut
  mid-flight at precisely the dashed line — the signature of a time-window
  cut, not of any per-channel effect.
* Activity above tick 8000: U 5826 / V 4634 / W 5804 nonzero samples
  (15–19% of each plane's total; ~16% of the charge per plane).  These are
  the "missing" tracks in the imaging display.

## Hypotheses ruled out

* **Channel mapping**: blob wire→channel projection lands exactly on the SP
  tracks, channel-by-channel, in U, V and W simultaneously (a mapping error
  would destroy the 3-view coincidence and misalign the overlay).
* **Corrupted readout / per-channel time offsets**: SP and imaging agree
  perfectly over ticks 0–8000; tracks are continuous and straight across the
  full channel range.  Also the "good" and "missing" regions are separated by
  a global tick boundary, identical for all channels.
* **DNN-ROI vs traditional input mismatch**: the gauss frames in
  `protodune-sp-frames-anode0.tar.bz2` (what imaging reads) and
  `protodune-sp-dnnroi-frames-anode0.tar.bz2` are bit-identical for this
  event (same 101081 nonzero samples, same total charge).

## Why only this run

The PDVD readout length is not constant across the runs we process:

| run | frame length (ticks) | truncated by `max_tbin: 8000`? |
|---|---|---|
| 039324 | 6400 | no (cap above frame end) |
| 040475 | 8000 | no (cap exactly at frame end) |
| 039252 | 10000 | **yes — last 2000 ticks dropped** |

All earlier PDVD imaging validation used 039324, where the cap is invisible.
SBND hit the same class of bug and documented it in its own config
(`cfg/pgrapher/experiment/sbnd/img.jsonnet`: `max_tbin: 3427 // ... was 3400,
dropped 27 ticks of real data`).

## Where the cap lives

`cfg/pgrapher/experiment/protodunevd/img.jsonnet`, `slicing()`:

```jsonnet
min_tbin: 0,
max_tbin: 8000,
```

Both the active (3-view, `tick_span=4`) and masked (2-view, `tick_span=500`)
slicings go through this function, so dead-region blobs are truncated the same
way.  All 8 anodes are affected equally.

## Recommended fix

`MaskSlice.cxx` auto-derives the window from the input frame when **both**
knobs are 0 (`min_tbin == 0 and max_tbin == 0` → min/max over the traces'
`tbin`/`tbin+size`).  Setting

```jsonnet
min_tbin: 0,
max_tbin: 0,
```

adapts to the per-run readout length automatically: 6400- and 8000-tick runs
produce the same slices as today (auto extent ≤ current cap), and 10000-tick
runs recover the lost 2000 ticks.  All four parallel slicing branches of one
anode see the same fanned-out frame, so the BlobSetMerge slice-sync invariant
is preserved.  Alternatively a hard `max_tbin: 10000` works but just moves the
trip-wire.

Note: with no T0 in PDVD, slices at late ticks map to apparent drift x beyond
the cathode; downstream clustering containment filters already run with the
relaxed scope filter (see clustering-boundary-merge.md), so the recovered
late-window tracks survive to the display.

### Secondary tick-domain knobs (audit when readout length changes)

Same config file; not the cause here, but tuned for shorter readouts:

* `CMMModifier` `org_hlimit: [8500]` — dead-channel range organization capped
  at tick 8500.
* `FrameQualityTagging` `min_time: 3180, max_time: 7870` — frame-quality
  evaluation window (MicroBooNE-era values).
* `ChargeErrorFrameEstimator` `time_limits: [12, 800]`.

## Reproduction

Analysis scripts (throwaway): `/home/xqian/tmp/img_sp_invest/{cmp_frames,planes,blobmap,overlay,quant}.py`.
Inputs: `pdvd/work/039252_8/` — `protodune-sp-frames-anode0.tar.bz2`,
`clusters-apa-anode0-ms-active.tar.gz`; wires
`protodunevd-wires-larsoft-v5.json.bz2`; blob→channel mapping via
`pdvd/img_plot/geom.py` (`PlaneGeom.channel(wip)`).

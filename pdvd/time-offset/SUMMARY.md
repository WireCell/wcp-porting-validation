# PDVD / PDHD per-event trigger time offset — findings

*2026-06-12. Question: how to get the trigger-to-waveform-start time offset event by
event for ProtoDUNE-VD (cosmics ~2500 us, beam ~411 us, possibly mixed within one
raw file).*

## TL;DR

- The larsoft fcl parameters (`DefaultTrigTime`, `TriggerOffsetTPC`) are **static
  per job** and for PDVD their defaults (250 / -250 us, copied from PDHD) **do not
  match the real VD readout windows** (2500 us or 400 us depending on run).
- The true offset is in every raw fragment header:
  `offset = trigger_timestamp - window_begin`. It is a **per-run DAQ readout
  configuration**, identical for all events and all trigger types within the runs
  checked — but each event carries its own copy, so always read it per event.
- The trigger **type** (beam / cosmic / laser) *does* vary event by event and is in
  the TriggerCandidate fragment.
- `pdvd_event_trigger_offset.py` (this directory) extracts all of this directly
  from the raw HDF5 with plain h5py — no DAQ or larsoft software needed.
  Results: `pdvd_trigger_offsets.csv`, `pdhd_trigger_offsets.csv`.

## Measured values (raw HDF5, fragment headers)

### PDVD (np02vd)

| run   | trigger types in file                          | offset_us | window_us |
|-------|-----------------------------------------------|-----------|-----------|
| 39252 | kCTBBeamChkvHLx/HxLx + kCTBOffSpillSnapshot    | 2500      | 5000      |
| 39253 | kCTBBeamChkvHLx/HxLx + kCTBOffSpillSnapshot    | 2500      | 5000      |
| 39349 | kCTBBeamChkvHL/HLx/HxLx                        | 400       | 3200      |

- Offset and window are constant within each run **regardless of trigger type**
  (the off-spill snapshots in 39252/53 also get 2500/5000). "2500 for cosmics,
  411 for beam" is a per-run *configuration* statement, not per-event physics.
- The first-TDE-frame offset (`frame_offset_us`, what ends up in `RDTimeStamp`)
  jitters 0–33 us above nominal because frames are ~33 us long and the window
  opens mid-frame: 2500–2531 us for runs 39252/53, 400–432 us for 39349. This is
  where the colleague's "411 us" comes from (nominal 400 + alignment).

### PDHD (np04hd)

| run   | trigger types                                  | offset_us | window_us |
|-------|-----------------------------------------------|-----------|-----------|
| 27305 | kCTBBeam                                       | 250       | 3000 (*)  |
| 27980 | kCTBOffSpillSnapshot                           | 250       | 3000      |
| 28084 | kCTBOffSpillSnapshot                           | 250       | 3000      |
| 29107 | kCTBBeam + kCTBOffSpill(Snapshot/CRTCosmic)    | 250       | 3000      |

- PDHD TPC is uniform: 250 us / 3000 us (6000 ticks) for beam AND cosmics —
  exactly what the larsoft fcl assumes. The per-event offset problem is
  PDVD-specific.
- (*) two events in 27305 have longer windows (5218 / 5496 us) on the first WIBEth
  stream — rare DAQ anomaly, another reason to read the window per event.
- WIBEth frame jitter: frame_offset_us = 250–282 us (frame = 32.768 us).
- Caveat: the readout window is **per subsystem**. In run 27980 the DAPHNE (PDS)
  fragments have 2750 us / 5500 us while the TPC has 250 / 3000. Always take the
  window from a WIBEth/TDEEth (TPC) fragment, not the first Detector_Readout
  dataset (the script does this).

## LArSoft side (dunesw v10_20_08d00)

- `services.DetectorClocksService.DefaultTrigTime` / `.TriggerOffsetTPC` defaults
  live in dunecore `fcl/detectorclocks_dune.fcl`. PDVD uses
  `protodunevd_detectorclocks: @local::protodunehd_detectorclocks` →
  `TriggerOffsetTPC: -250`, `DefaultTrigTime: 250`. Wrong for VD data
  (should be -2500 or -400 depending on run).
- Existing practice for run-dependent offsets is hardcoded per-run fcl overrides
  (see Iceberg `signal2noise_dataprep_run4/5.fcl`: -250 / -335).
- A per-event mechanism exists: `DetectorClocksServiceStandard::DataFor(evt)`
  reads a `raw::Trigger` product (label = `TrigModuleName`) and overrides the
  trigger time per event — but **no module in duneprototypes produces
  `raw::Trigger` for PDVD**, so the static default always wins. Writing a small
  producer (offset from RDTimeStamp vs TriggerCandidate, below) + setting
  `TrigModuleName` would make all downstream `TriggerOffsetTPC()` consumers
  correct automatically.

## Three ways to get the per-event offset

1. **Raw HDF5 (this directory's script)** — no framework needed:
   `python3 pdvd_event_trigger_offset.py <file-or-dir> [--csv out.csv]`
   Parses per TriggerRecord:
   - TriggerRecordHeader (marker `0x33334444`): `trigger_timestamp` u64 @24
     (DTS ticks, 16 ns)
   - first WIBEth/TDEEth fragment header (marker `0x11112222`, 72 B):
     `trigger_timestamp` @24, `window_begin` @32, `window_end` @40
   - Trigger_Candidate fragment payload (`trgdataformats2::TriggerCandidateData`
     v3): `time_candidate` u64 @24, `type` i32 @36 (beam/cosmic/laser enum)
   - first data frame's DAQEthHeader timestamp (u64 @ payload+8) = actual
     waveform start.
2. **Stage1 art files** (colleague's recipe): offset =
   `raw::RDTimeStamps_tpcrawdecoder_daq.obj.fTimeStamp[0]` minus
   `TriggerCandidateDatas_triggerrawdecoder_daq.obj.time_candidate`
   (both DTS ticks; x16 ns). Verified equivalent: `time_candidate` ==
   TriggerRecordHeader `trigger_timestamp` exactly in every event checked, and
   `RDTimeStamp` is the frame-aligned waveform start. Both products are available
   to WireCell since `wclsdatavd` runs after the decoders.
3. **Proper larsoft fix**: a producer filling `raw::Trigger` from (2) + set
   `services.DetectorClocksService.TrigModuleName` — makes the per-event time
   flow through `DetectorClocksData` everywhere.

## Files here

- `pdvd_event_trigger_offset.py` — extractor (works for PDVD and PDHD; PDS/CRT
  fragments are skipped deliberately).
- `pdvd_trigger_offsets.csv`, `pdhd_trigger_offsets.csv` — per-event results.
  Columns: file, run, trig_num, seq, tc_type, trig_ts, tc_time_candidate,
  win_begin, win_end (DTS ticks), offset_us (trigger - window start, nominal),
  frame_offset_us (trigger - first frame, what RDTimeStamp sees), window_us.

## Open questions

- Runs 39252/39253 carry the "cosmics" window config (2500/5000) but their
  trigger types are mostly CTB beam-Cherenkov — per-event `tc_type` is the
  trustworthy label of what each event is, not the run's window config.
- No laser-trigger (kCIBLaserTriggerP1-3) events in the files at hand; their
  window config is unverified.

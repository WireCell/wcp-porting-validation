# Gen2 real data: run frameshift first (REMINDER)

**For all Gen2 real data events, run `run_frameshift.fcl` to obtain the
frameshift (timing) information before using the data downstream.**

This is a reminder for future-me / future-Claude: raw Gen2 SBND data needs the
per-event frame-shift timing correction, produced by the sbndcode FrameShift
producer.  Do this once per data file to get a `..._frameshift.root` that
carries the frameshift product (everything else is copied through).

## How

```bash
# SL7 + sbndcode env (setup-local-opt.sh has sbndcode v10_14_02_03):
lar -c run_frameshift.fcl -s <decoded_reco1_data>.root
# -> <decoded_reco1_data>_frameshift.root   (output name = %ifb_frameshift.root)
```

- fcl: `run_frameshift.fcl` (sbndcode; ref
  https://github.com/SBNSoftware/sbndcode/blob/v10_14_02_04/sbndcode/Timing/FrameShift/run_frameshift.fcl
  -- v10_14_02_03 in our stack is equivalent for this).  It runs the
  `frameshift_data` producer (`frameshift_sbnd_data.fcl`) and RootOutputs
  `%ifb_frameshift.root`, dataTier "reco".
- Env: any standard sbndcode setup; here `sbnd/setup-local-opt.sh` works.

## Done so far (2026-07-21)

- Input: `sbnd/samples/filtered-reco1/data_filtered_decoded_reco1-fe6033f3-07a0-4971-cea5-16ce59269fba_eventidfiltered.root`
- Output: `sbnd/samples/filtered-reco1/data_filtered_decoded_reco1-fe6033f3-07a0-4971-cea5-16ce59269fba_eventidfiltered_frameshift.root` (48 events).

## Downstream

Run the WCT imaging/clustering/matching data chain on the `_frameshift.root`:

```bash
lar -c wcls-img-clus-matching-xin-data.fcl -s <..._frameshift>.root
# reality=data -> grouped reco config: pos_offset_on=true, use_sce=false.
```

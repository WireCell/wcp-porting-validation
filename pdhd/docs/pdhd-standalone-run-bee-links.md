# PDHD standalone DNN-ROI processing — run inventory & Bee links

Standalone (no art/LArSoft) processing of all ProtoDUNE-HD runs under the two
`input_data_<gain>_<old|new>_coh_grouping` roots, taken end-to-end through
NF → SP → DNN-ROI → L1SP → imaging → clustering → combined Bee upload.

Processed 2026-06-08.

## Pipeline / parameters

Per event, per run:

```sh
./run_nf_sp_dnnroi_evt.sh <run> <evt>     # all anodes (default), DNN-ROI + L1SP
./run_img_evt.sh -d on <run> all          # imaging on the DNN frames
./run_clus_evt.sh <run> all               # per-APA + all-APA clustering
./run_bee_combined_evt.sh <run>           # one combined Bee link / run
```

- **All four APAs** (the `run_nf_sp_dnnroi_evt.sh` default; the model is shared
  across the geometrically identical APAs).
- **L1SP on, DNN tagger** (`use_l1sp_dnn=true`, `l1sp_pd_mode=dnn`,
  `l1sp/pdhd/l1sp_dnn_pdhd_v1.ts`) — final frame carries L1SP-corrected
  `gauss`/`wiener`.
- **APA0 W-plane ROI tune on** (toolkit default).
- **FE gain + coherent-noise grouping auto-derived from the input-root dir
  name** (see [pdhd-coh-groups-preflip.md](pdhd-coh-groups-preflip.md)):
  `*_14_old_*` → 14 mV/fC + pre-flip grouping; `*_7p8_new_*` → 7.8 mV/fC +
  post-flip grouping.  Confirmed in each run's log
  (`elecGain:` + `[coh] … -> PRE/POST-FLIP`).
- DNN model: `dnnroi/pdhd/pipe_distill_transformer_6ch.ts` (fp32, 6-ch).

Each Bee link carries, per event, the full per-stage instance set grouped by
drift side: `imaging-group02/13`, `clustering-group02/13`, `clustering-global`,
`channel-deadarea-group02/13`.

## Runs

| run | dir → gain / grouping | events | DNN frames | Bee link |
|---|---|---|---|---|
| 027380 | 14 / pre-flip  | 0–7 (8) | 32 | https://www.phy.bnl.gov/twister/bee/set/3944325f-2f93-43fe-9708-ef1ae7a280bb/event/list/ |
| 027409 | 14 / pre-flip  | 0–7,12 (9) | 36 | https://www.phy.bnl.gov/twister/bee/set/ac85822b-c1dc-44ac-84d9-3dab9985b59a/event/list/ |
| 027425 | 14 / pre-flip  | 5,6,9,12,20,21,27,32,33 (9) | 36 | https://www.phy.bnl.gov/twister/bee/set/3ae58236-8b50-4b92-bca4-1dcd0dbe5c89/event/list/ |
| 027980 | 7.8 / post-flip | 0–4 (5) | 20 | https://www.phy.bnl.gov/twister/bee/set/46e2f162-5a8f-418b-afdc-8f3d1ad71e41/event/list/ |
| 029107 | 7.8 / post-flip | 0–4 (5) | 20 | https://www.phy.bnl.gov/twister/bee/set/85c752d9-71b4-41c1-a637-9c35ad936115/event/list/ |

(DNN frames = events × 4 anodes; every event clustered.)

## Note on shared-host job limiting

A full all-anode DNN `wire-cell` process sizes its TorchScript intra-op thread
pool to the core count (~64 threads) regardless of `OMP_NUM_THREADS`, so running
many events concurrently oversubscribes the box.  These runs were therefore
driven at **2 concurrent DNN jobs, each `taskset`-pinned to a disjoint 16-core
block**, with imaging/clustering capped via `PDHD_MAX_JOBS=4`.  Pinning (not a
thread-count env var) is the reliable lever for bounding concurrent DNN
footprint here.

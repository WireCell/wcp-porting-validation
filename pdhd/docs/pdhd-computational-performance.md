# PD-HD computational performance per event (reviewer backup slide)

Per-event wall time, CPU time, and peak memory for the full ProtoDUNE-HD
Wire-Cell reconstruction chain — the backup slide for the standard reviewer
question "what does this cost per event?".  Companion of the chain diagrams
(`nf-chain-diagram.md`, `sp-chain-diagram.md`, `imaging-chain-diagram.md`,
`qlmatching-chain.md`) and successor of the pre-optimization profile in
[pdhd-pipeline-resource-profile.md](pdhd-pipeline-resource-profile.md).

## The backup slide

**ProtoDUNE-HD data, run 029107, full WCT chain (NF+SP+DNN-ROI → imaging →
light reco → clustering + Q/L matching), one event end-to-end:**

| stage | wall [s] | CPU [core-s] | peak RSS [GB] |
|---|---|---|---|
| NF + SP + DNN-ROI (4 APAs, one process) | 79–95 | 307–418 | 2.1–2.6 |
| 3-D imaging (4 APAs, sequential processes) | 34–56 | 37–60 | 0.39–0.88 |
| light reco (all 160 PDs) | 4–5 (1.6–1.9 wire-cell) | ~17 | 0.42 |
| clustering + Q/L matching | 30–150 | 35–156 | 0.54–1.08 |
| **full chain** | **~2.5–4.8 min** | **~400–570** (≈0.11–0.16 core-h) | **≤2.6** (NF/SP peak) |

- Three example events measured solo (uncontended) at toolkit HEAD
  `39893440` (2026-07-19): **evt 983** (idx 0, typical), **evt 1127**
  (idx 18, busiest cosmic activity of the 30-event set), **evt 1015**
  (idx 4, the known bright-flash outlier: 443 flashes, 1.3 M PE).
- The ranges above span typical → worst: a median event is the low edge
  (~2.5 min, ~0.11 core-h); even the pathological evt 1015 stays under
  5 min / 1.1 GB downstream of NF/SP.
- Peak memory over the whole chain is set by NF/SP/DNN-ROI (~2.1–2.6 GB,
  dominated by the one-shot mkldnn transposed-convolution transient of the
  DNN-ROI forward); everything downstream stays ≤1.1 GB.
- CPU > wall only in NF/SP (~4 cores: per-APA pipeline threads + the libtorch
  intra-op pool) and light reco; imaging and clustering are effectively
  single-threaded per event.
- Population context (all 30 events of run 029107, production batch):
  clustering+Q/L wall median ~30 s, range 21–151 s, peak RSS 0.39–1.07 GB;
  light reco ~1.6 s/event flat.  NF/SP and imaging are population-flat.
- Per-event archive footprint: raw input 4 × 19 MB ≈ 76 MB → SP/DNN-ROI
  frames 232 MB, imaging clusters 23 MB, Bee zips 4.2 MB, opflash 0.8 MB.

### The evt-1015 story (why the tail is now benign)

Before the Q/L optimization campaign the bright outlier evt 1015 cost
**15–22 min and 12–15 GB** in Q/L matching alone
(`pdhd-pipeline-resource-profile.md`).  The result-preserving track
(sparse LASSO assembly `c98ca863`, Gram const-ref `4c85ff34`, sparse solve
`99d8cb17`, cross-side visibility hoist `b9c198b4`, xtpc chunk-box pruning
`2a2612d3` — all gated byte-identical, doc
`match/docs/qlmatching-perf-evt1015-pdhd.md`) brought the same event to
**150 s / 1.08 GB** — ~8× wall and ~13× memory — so the per-event resource
envelope is now flat enough for routine farm scheduling.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
# fresh tag so no existing work/ dir is touched; stages run solo, in order
./run_nf_sp_dnnroi_evt.sh -O _perfslide 029107 0        # defaults: -P fp32 -D cpu -L on
# imaging into a tagged workdir: run_img_evt.sh's -s + -d on input resolution
# only searches the untagged dir, so replicate its per-anode production
# invocation (run_img_evt.sh lines 244-272) with
#   input_prefix=work/029107_0_perfslide/protodunehd-sp-dnnroi-frames
#   output_dir=work/029107_0_perfslide, nticks probed from frame_gauss0_*,
#   per-anode sequential, tcmalloc preload, GOGC=off
./run_light_allpd_evt.sh -s _perfslide 29107 983        # run UNPADDED (a leading
                                                        # zero trips printf %06d)
ln -s ../029107_allpd983_perfslide/opflash_pdhd-allpd-wct.tar.gz \
      work/029107_0_perfslide/opflash_pdhd-wct.tar.gz
./run_clus_evt.sh -s perfslide 029107 0
# idx 18 / idx 4 identically (DAQ event number = 983 + 8*idx)
```

Wall/CPU/maxrss per stage from wrapping each runner in a `wait4`-rusage
timer (GNU `time` is not installed on this host); peak RSS cross-checked
against the runners' own `/proc` VmHWM samplers (`PDHD_RESMON`, files
`clus_resource_*.txt` / `light_resource_*.txt` / `time_*.txt` in the
`work/029107_{0,18,4}_perfslide` dirs).

## Measurement conditions

- Host: AMD Ryzen Threadripper 7970X (32 cores / 64 threads), 252 GB RAM,
  local NVMe-backed NFS; load < 5 throughout (solo, sequential stages).
- Toolkit `apply-pointcloud` @ `39893440`, libs installed in `local/lib`
  (waf up-to-date check = freshness proof); tcmalloc preloaded, `GOGC=off`
  (the production runner environment).
- DNN-ROI: TorchScript fp32 on **CPU** + L1SP (`-P fp32 -D cpu -L on`, the
  runner defaults); no GPU used in these numbers.
- Runner defaults = the PDHD production operating point (`sparse_lasso`,
  `crossside_skip_vis`, containment/over-prediction prefilters ON in the
  PDHD jsonnet; C++ defaults byte-identical OFF).
- Light-reco wall: the runner spends ~3 s converting the DAQ ROOT file
  (`fullstream_to_decoana.py`); the wire-cell reco step itself is 1.6–1.9 s
  (`light_resource_*.txt`).

## Detailed per-event numbers

| stage | evt 983 (idx 0, typical) | evt 1127 (idx 18, busy) | evt 1015 (idx 4, outlier) |
|---|---|---|---|
| NF+SP+DNN-ROI wall / CPU / RSS | 79.1 s / 307 s / 2.06 GB | 94.8 s / 418 s / 2.50 GB | 80.6 s / 321 s / 2.30 GB |
| imaging wall / CPU / RSS | 33.5 s / 37 s / 0.39 GB | 50.7 s / 55 s / 0.60 GB | 55.5 s / 60 s / 0.88 GB |
| — per-APA imaging wall | 6–11 s | 8–19 s | 7–22 s |
| light reco wall (wire-cell) / RSS | 4.1 s (1.6 s) / 0.42 GB | 4.6 s (1.6 s) / 0.42 GB | 4.9 s (1.9 s) / 0.42 GB |
| clustering only (no Q/L) wall / RSS | 14 s / 0.49 GB | 32 s / 0.87 GB | 35 s / 0.82 GB |
| clustering + Q/L wall / CPU / RSS | 29.5 s / 35 s / 0.54 GB | 77.0 s / 83 s / 0.87 GB | 149.7 s / 156 s / 1.08 GB |
| **full chain wall / CPU** | **146 s / 396 s** | **227 s / 574 s** | **291 s / 554 s** |

The Q/L-matching increment over base clustering is ~13 s on a typical event,
~45 s on the busiest, ~115 s on the bright-flash outlier — the residual
scaling is the per-point visibility loop over 443 flashes (see
`match/docs/qlmatching-perf-evt1015-pdhd.md` §9–11 for the profile and the
further `vis_sample_stride` lever, measured byte-identical at stride 8 but
left OFF as a physics decision).

## Scaling remarks (for the DUNE question)

- Events are fully independent: the chain parallelizes trivially event-wise.
  Batch production on this box runs `PDHD_MAX_JOBS≈6` concurrent events;
  quote solo numbers and size farms in CPU core-seconds.
- Budget numbers: **~0.11–0.16 CPU core-hour and ≤2.6 GB peak RSS per
  event**; a standard 2 GB/core grid slot fits the whole chain with a 2-core
  request (or 1 core once NF/SP is given a GPU / smaller DNN tile).
- The only stage above 1.1 GB is NF/SP/DNN-ROI; its peak is a known,
  optimizable transient (`sigproc/docs/nfsp-dnnroi-perf-round1.md` round-2
  leads: DNN tiling knob, torch OMP wait policy).
- Deeper profiles: `pdhd-pipeline-resource-profile.md` (30-event
  pre-optimization baseline), `match/docs/qlmatching-perf-evt1015-pdhd.md`
  (Q/L), `sigproc/docs/nfsp-dnnroi-perf-round{0,1}.md` (NF/SP/DNN-ROI),
  `clus/docs/imgclus-optimization-log.md` (imaging+clustering rounds 1–9).

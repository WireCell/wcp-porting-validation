# PD-VD computational performance per event (reviewer backup slide)

Per-event wall time, CPU time, and peak memory for the full ProtoDUNE-VD
Wire-Cell reconstruction chain — the backup slide for the standard reviewer
question "what does this cost per event?".  Companion of the chain diagrams
([16_pdvd-sim-chain.md](16_pdvd-sim-chain.md),
[17_pdvd-clustering-qlmatching-chain.md](17_pdvd-clustering-qlmatching-chain.md))
and of the optimization history in
[15_pdvd-light-ql-perf.md](15_pdvd-light-ql-perf.md).

## The backup slide

**ProtoDUNE-VD data, run 039252, full WCT chain (NF+SP+DNN-ROI → imaging →
light reco → clustering + Q/L matching), one event end-to-end:**

| stage | wall [s] | CPU [core-s] | peak RSS [GB] |
|---|---|---|---|
| NF + SP + DNN-ROI (8 CRPs, one process) | 122 | 250–254 | 2.2–2.3 |
| 3-D imaging (8 CRPs, sequential processes) | 51–60 | 51–61 | 0.35–0.39 |
| light reco (all 40 PDs) | ~2 | ~6 | 0.42 |
| clustering + Q/L matching | 18–27 | 26–34 | 0.72–0.83 |
| **full chain** | **~3.2–3.5 min** | **~340–350** (≈0.1 core-h) | **≤2.3** (NF/SP peak) |

- Two example events measured solo (uncontended) at toolkit HEAD `39893440`
  (2026-07-19): **evt 298567** (idx 0, the hand-scan reference event) and
  **evt 298693** (idx 9).
- Peak memory over the whole chain is set by NF/SP/DNN-ROI (~2.3 GB, the
  one-shot mkldnn transposed-convolution transient of the DNN-ROI forward);
  everything downstream stays under 0.9 GB.
- CPU > wall only in NF/SP (~2 cores: per-CRP pipeline threads + the libtorch
  intra-op pool) and light reco; imaging and clustering are effectively
  single-threaded per event.
- Population context (18-event batch of run 039252, production operating
  point): clustering+Q/L wall 15–68 s, peak RSS 0.68–1.22 GB; light reco
  1.7–2.0 s solo.  The 120-event light+Q/L campaign (doc 15) gives QLMatching
  alone: median 2.2 s, p90 3.4 s, max 4.2 s after the round-1 optimization.
- Per-event archive footprint: raw input 8 × 19 MB ≈ 152 MB → SP/DNN-ROI
  frames 445 MB, imaging clusters 34 MB, Bee zips 3.2 MB, opflash 0.6 MB.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
# fresh tag so no existing work/ dir is touched; stages run solo, in order
./run_nf_sp_dnnroi_evt.sh -O _perfslide 039252 0        # defaults: -P fp32 -D cpu
./run_img_evt.sh        -O _perfslide 039252 0
./run_light_evt.sh      -s _perfslide 39252 298567      # run UNPADDED (a leading
                                                        # zero trips printf %06d)
PDVD_LIGHT_SUFFIX=_perfslide ./run_clus_evt.sh -s perfslide 039252 0
# idx 9 identically with event number 298693 (event no = 298567 + 14*idx)
```

Wall/CPU/maxrss per stage from wrapping each runner in a `wait4`-rusage
timer (GNU `time` is not installed on this host); peak RSS cross-checked
against the runners' own `/proc` VmHWM samplers (`PDVD_RESMON`, files
`clus_resource_*.txt` / `light_resource_*.txt` / `time_*.txt` in the
`work/039252_{0,9}_perfslide` dirs).

## Measurement conditions

- Host: AMD Ryzen Threadripper 7970X (32 cores / 64 threads), 252 GB RAM,
  local NVMe-backed NFS; load < 3 throughout (solo, sequential stages).
- Toolkit `apply-pointcloud` @ `39893440`, libs installed in
  `local/lib` (waf up-to-date check = freshness proof); tcmalloc preloaded,
  `GOGC=off` (the production runner environment).
- DNN-ROI: TorchScript fp32 on **CPU** (`-D cpu`, the runner default); no GPU
  used anywhere in these numbers.
- Runner defaults = the PDVD production operating point (doc 17: the
  operating-point knobs live in the runners; jsonnet defaults stay
  byte-identical OFF).
- Light-reco walls are the wire-cell step (`light_resource_*.txt`: 1.6–1.8 s);
  the runner adds ~0.3 s of trigger-offset probing.

## Detailed per-event numbers

| stage | evt 298567 (idx 0) | evt 298693 (idx 9) |
|---|---|---|
| NF+SP+DNN-ROI wall / CPU / RSS | 121.6 s / 254 s / 2.22 GB | 121.3 s / 250 s / 2.25 GB |
| imaging wall / CPU / RSS | 50.6 s / 51 s / 0.35 GB | 60.2 s / 61 s / 0.39 GB |
| — per-CRP imaging wall | 5–8 s each | 5–11 s each |
| light reco wall / CPU / RSS | 1.9 s / 6.2 s / 0.42 GB | 2.1 s / 6.3 s / 0.42 GB |
| clustering only (no Q/L) wall / RSS | 14 s / 0.53 GB | 22 s / 0.66 GB |
| clustering + Q/L wall / CPU / RSS | 18.1 s / 26 s / 0.72 GB | 26.7 s / 34 s / 0.83 GB |
| **full chain wall / CPU** | **192 s / 338 s** | **210 s / 351 s** |

The Q/L-matching increment over base clustering is 3–5 s wall per event at
the current HEAD — the post-optimization cost of the joint two-volume
LASSO match (doc 15: it was ~7 s median before round 1).

## Scaling remarks (for the DUNE question)

- Events are fully independent: the chain parallelizes trivially event-wise.
  Batch production on this box runs `PDVD_MAX_JOBS≈6` concurrent events
  (batch walls are then contention-inflated ~2.5×; doc 15 gotcha — quote
  solo numbers, size farms in CPU core-seconds).
- Budget numbers: **~0.1 CPU core-hour and ≤2.3 GB peak RSS per event**;
  a standard 2 GB/core grid slot fits the whole chain with a 2-core request
  (or 1 core once NF/SP is given a GPU / smaller DNN tile).
- The only stage above 1 GB is NF/SP/DNN-ROI; its peak is a known,
  optimizable transient (`sigproc/docs/nfsp-dnnroi-perf-round1.md` round-2
  leads: DNN tiling knob, torch OMP wait policy).
- Deeper profiles: `15_pdvd-light-ql-perf.md` (light+QL, 120 events),
  `sigproc/docs/nfsp-dnnroi-perf-round{0,1}.md` (NF/SP/DNN-ROI),
  `clus/docs/imgclus-optimization-log.md` (imaging+clustering rounds 1–9).

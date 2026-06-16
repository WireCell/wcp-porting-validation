# PDHD light+charge pipeline resource profile (run 29107)

Per-stage **wall-time and peak-memory** breakdown of the full ProtoDUNE-HD
reconstruction of run 29107 (30 events), from raw to Q/L-matched flashes:

```
NF/SP/DNNROI  ─▶  imaging  ─▶  light reco (all-PD)  ─▶  clustering  ─▶  QLMatching
run_nf_sp_dnnroi   run_img      run_light_allpd          run_clus        (inside run_clus -q)
```

The question this answers: **which stage eats the time, and which eats the
memory?** (The 11.4 GB job the operator noticed.)

> **One-line answer.** The base imaging+clustering chain is already
> well-optimized (see [`clus/docs/imgclus-resource-profile.md`](../../clus/docs/imgclus-resource-profile.md)
> and [`imgclus-optimization-log.md`](../../clus/docs/imgclus-optimization-log.md))
> — clustering stays **≤0.9 GB / ≤44 s** even on the busiest 29107 event.
> The cost the operator saw is **QLMatching** (run inside the clustering step
> with `-q`/`-calib`/`-op`): it is modest on normal events (≈3× the base
> clustering time, ≈2× its memory) but **explodes on the single bright outlier
> event 1015** to **≈12–15 GB / ≈15–22 min**. Nothing else in the pipeline exceeds
> ~2 GB or ~95 s. So: **time and memory are dominated by QLMatching, and almost
> entirely by one event.**

---

## 1. Per-stage summary (the answer)

Single-threaded `wire-cell` per stage; wall = stage log first→last timestamp;
peak RSS = per-process `VmHWM` (`/proc/<pid>/status`). "typical" = the 29
ordinary events (min–max, median in **bold**); "evt 1015" = the bright outlier
(DAQ event 1015, bee/charge index 4).

| stage | tool | wall typical (s) | wall evt 1015 (s) | peak RSS typical | peak RSS evt 1015 | other |
|---|---|---|---|---|---|---|
| NF / SP / DNNROI | `run_nf_sp_dnnroi_evt.sh` | 89–103 (**92**) | 94 | 1.7–2.5 GB (**1.9**) | 2.0 GB | GPU VRAM ≈18 MiB |
| Imaging (4 anodes, sequential) | `run_img_evt.sh` | 30–56 (**37**) | 61 | 0.40–0.90 GB (**0.44**)¹ | ≈0.9 GB | per-anode peak (not summed) |
| Light reco (all-PD) | `run_light_allpd_evt.sh` | 3.4–4.2 (**3.5**) | 4.2 | ≈0.43 GB | 0.43 GB | snippet + full-stream OpRoi |
| Clustering — **base, no Q/L** | `run_clus_evt.sh` | 13–44 (**19**)¹ | 40 | 0.4–0.9 GB (**0.53**)¹ | 0.82 GB | already optimized |
| Clustering — **+ QLMatching + calib + op** | `run_clus_evt.sh -calib -op` | 38–157 (**66**) | **877** | 0.7–2.4 GB (**1.3**) | **12.4 GB** | the operator's 11.4 GB |

¹ Imaging RSS and base-clustering time/RSS are the locked numbers from the
post-round-8 full sweep in
[`clus/docs/imgclus-resource-profile.md`](../../clus/docs/imgclus-resource-profile.md)
(run 029107 row), measured **without** Q/L. All other cells are measured here.

**Reading the table:**

- **Time.** For an ordinary event the longest stage is actually **NF/SP/DNNROI
  (~92 s)**, then clustering+Q/L (~66 s), imaging (~37 s), light (~4 s). For the
  bright event, **clustering+Q/L (877 s) dwarfs everything** — 9× the entire
  rest of the pipeline combined.
- **Memory.** For ordinary events the floor is **NF/SP (~1.9 GB)**; everything
  else is ≤1.3 GB. For the bright event, **clustering+Q/L (12.4 GB)** is ~6× the
  next-largest stage.
- **Event dependence.** NF/SP, imaging and light are **flat** across all 30
  events (cosmic-rate-independent readout/SP). Only **clustering+Q/L is
  event-dependent**, and its variance is dominated by one event.

---

## 2. Method & instrumentation

- **Clustering+Q/L** (`run_clus_evt.sh`): a `/proc/<pid>/status` sampler added to
  the run script (`PDHD_RESMON=on`, default) backgrounds `wire-cell` and polls
  `VmRSS`/`VmHWM` every 2 s, writing per event:
  - `clus_resource_<run>_<evt>.txt` — one line: `wall_s`, `peak_rss_gb`, flags;
  - `clus_rss_<run>_<evt>.csv` — the full RSS-vs-wallclock trace (timestamps
    align with the debug-log stage markers, so the peak can be attributed to a
    stage).
- **NF/SP/DNNROI**: `run_nf_sp_dnnroi_evt.sh` already records `time_<…>.txt`
  (`VmHWM_kB`) + `gpu_mem_<…>.csv` (nvidia-smi VRAM trace); wall from the log.
- **Imaging / light**: wall from the stage logs; imaging peak RSS cited from the
  existing img+clus sweep; light peak RSS measured with a process-tree sampler.
- **Q/L attribution controls** (event 1015, scratch work dirs, inputs symlinked):
  - `run_clus_evt.sh` (no flags) — **base clustering**;
  - `run_clus_evt.sh -q` — **+ QLMatching**, no JSON dumps;
  - `run_clus_evt.sh -calib -op` — **+ calib hand-scan + optical-bee dumps**.

All stages run one single-threaded `wire-cell` (Pgrapher), so the per-event
numbers are not inflated by intra-event threading.

---

## 3. Per-event clustering (+ QLMatching + calib + op), all 30 events

Peak RSS and wall from `clus_rss_*.csv`. This is the only event-dependent stage.

| idx | DAQ evt | wall (s) | peak RSS (GB) |
|---|---|---|---|
| 0 | 983 | 64 | 1.41 |
| 1 | 991 | 72 | 1.23 |
| 2 | 999 | 54 | 1.07 |
| 3 | 1007 | 52 | 1.00 |
| **4** | **1015** | **877** | **12.38** |
| 5 | 1023 | 56 | 1.11 |
| 6 | 1031 | 44 | 1.02 |
| 7 | 1039 | 40 | 0.73 |
| 8 | 1047 | 78 | 1.75 |
| 9 | 1055 | 66 | 1.23 |
| 10 | 1063 | 80 | 1.39 |
| 11 | 1071 | 48 | 1.00 |
| 12 | 1079 | 94 | 1.30 |
| 13 | 1087 | 66 | 1.46 |
| 14 | 1095 | 58 | 1.33 |
| 15 | 1103 | 157 | 2.39 |
| 16 | 1111 | 143 | 1.99 |
| 17 | 1119 | 55 | 1.48 |
| 18 | 1127 | 149 | 1.90 |
| 19 | 1135 | 52 | 1.20 |
| 20 | 1143 | 86 | 1.84 |
| 21 | 1151 | 84 | 1.39 |
| 22 | 1159 | 38 | 0.96 |
| 23 | 1167 | 52 | 1.31 |
| 24 | 1175 | 66 | 1.15 |
| 25 | 1183 | 69 | 1.23 |
| 26 | 1191 | 82 | 1.41 |
| 27 | 1199 | 50 | 0.96 |
| 28 | 1207 | 52 | 0.92 |
| 29 | 1215 | 80 | 1.48 |

Excluding evt 1015: wall **38–157 s** (median 66 s), peak RSS **0.73–2.39 GB**
(median 1.3 GB). Event 1015 is a **5.6× / 9.5×** outlier in RSS / wall over the
next-worst event.

---

## 4. What costs the time and memory — Q/L attribution (event 1015)

Control runs on the bright event isolate the cost:

| configuration | wall (s) | peak RSS (GB) | Δ vs base |
|---|---|---|---|
| base clustering (no Q/L) | 40 | 0.82 | — |
| + QLMatching, **no dumps** (`-q`) | 1295 | **14.92** | +1255 s / +14.1 GB |
| + QLMatching + calib + op (`-calib -op`) | 877 | 12.38 | +837 s / +11.6 GB |

**The dumps are not the cost — QLMatching is.** The `-q`-only run (no JSON dumps)
is *heavier and slower* (14.9 GB / 22 min) than the full `-calib -op` run
(12.4 GB / 15 min): QLMatching on this event swings **~12–15 GB / ~15–22 min
run-to-run**, and the calib/op dumps' contribution (a ~1.4 GB step at finalize,
seen in the `-calib -op` RSS-vs-time trace, plus a few seconds of I/O for the
29 MB calib JSON) is **smaller than that matching variance** — i.e. negligible.
The RSS trace of the `-calib -op` run climbs **monotonically through the matching
window** (6 GB at t+225 s → 10 GB at t+450 s → 11 GB at t+680 s) and only steps
up ~1.4 GB at the very end. So **both the time and the memory are QLMatching**;
base clustering (0.82 GB / 40 s) and the dumps are rounding error next to it.

The large run-to-run swing (±~2–3 GB, ±~few min) is itself characteristic of the
matching on this event: the bundle/LASSO solver's working set for ~454 flashes ×
~100 clusters × a huge point cloud is allocation-order-sensitive.

---

## 5. Why event 1015 is pathological

Event 1015 is the **bright outlier** documented in
[`pdhd-pd-activity-per-event.md`](pdhd-pd-activity-per-event.md) §6: a
spatially-extended very-high-light physics event. From the Q/L calib dump it has
**454 flashes** (vs ~100–160 for ordinary events) and a **1.33 M-PE** maximum
flash (vs ~20–48 k). QLMatching evaluates candidate bundles ≈ *flashes × clusters*
and, per bundle, loops the predicted-light sum over **every charge point of the
group** — so the cost scales like *flashes × clusters × points*. With ~3× the
flashes on top of the busiest charge image, this is the ~20× time / ~15× memory
blow-up. **It is one event, not the run**: the other 29 stay ≤2.4 GB / ≤157 s.

---

## 6. Takeaways

- **The clustering optimization holds.** Base imaging+clustering is ≤0.9 GB /
  ≤44 s even on the busiest 29107 event — not the bottleneck. (Provenance:
  [`imgclus-optimization-log.md`](../../clus/docs/imgclus-optimization-log.md).)
- **QLMatching is the new dominant cost**, in both time and memory, and is
  concentrated on the single bright event. Budget **~15 GB / ~15–22 min for the
  worst event** (run-to-run swing ±~2–3 GB); the rest of the run is ~1.3 GB /
  ~1 min per event.
- **The calib/op dumps are diagnostic, not the hog** (~1.4 GB at finalize).
  Production Q/L (`-q` without `-calib`/`-op`) avoids that last step but **not**
  the dominant matching cost.
- **NF/SP is the per-event floor** (~2 GB / ~90 s) and is flat — the natural
  target if ordinary-event throughput (not the outlier) becomes the concern.
- **Optimization lead for the outlier:** QLMatching's *flashes × clusters ×
  points* bundle loop. Capping flashes per group, pre-filtering clusters by the
  flash time window before the point loop, or down-sampling the point cloud used
  for the light prediction would each cut the event-1015 cost without touching
  ordinary events.

## Reproduce

```
# per-event clustering+Q/L resource (writes clus_resource_*.txt + clus_rss_*.csv):
PDHD_RESMON=on ./run_clus_evt.sh -calib -op 29107 all     # default PDHD_MAX_JOBS=nproc

# Q/L attribution controls on the bright event (scratch dirs, inputs symlinked):
./run_clus_evt.sh -s base   29107 4   # base clustering (no Q/L)
./run_clus_evt.sh -q -s qonly 29107 4 # + QLMatching, no dumps
./run_clus_evt.sh -calib -op 29107 4  # + calib + op dumps

# other stages already record their own time/mem (time_*.txt, gpu_mem_*.csv);
# imaging RSS + base-clustering numbers: clus/docs/imgclus-resource-profile.md
```

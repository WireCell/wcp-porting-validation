# 08 — Imaging cost and memory of the FD chain: diagnosis, what PDHD/PDVD/SBND teach, and the fixes

Doc 07 §3 recorded that the wcfm imaging step on the `_sub4f` tier costs up to 1188 s and 15.7 GB per
anode (event 122 anode 2, a nue shower APA with 628 k sub-blobs) and listed four options without
taking any. This doc answers the owner's question — why so slow and so much memory, and what the
PDHD/PDVD/SBND imaging campaigns teach for the FD chain — with profiles rather than inference, and
then takes the byte-identical fixes the profiles justify. Nothing here changes any production output:
every toolkit change is gated by archive content hashes (§5), and the one configuration change is a
default-OFF knob.

## 0. Repro

```
# profiles (scratch, nothing written into work/): event 122 anode 2, uncut and _sub4f
cd wcfm; export WIRECELL_PATH=$PWD:$WCT_BASE/toolkit/cfg:$WCT_BASE/wire-cell-data
IN=work/000001_122; OUT=/home/xqian/tmp/wcfm-prof/<run>
wcsonnet -A input_prefix=$IN/sim-frames -S 'anode_indices=[2]' -A output_dir=$OUT \
         -A depos=$IN/sim-depos.tar.bz2 [-S blob_cutting=true -S cut_length=4] -o $OUT/cfg.json wct-img-all.jsonnet
# CPU: tcmalloc + gperftools sampler (production preloads tcmalloc; SIGPROF needs the precompiled JSON, M17)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4 GOGC=off CPUPROFILE=$OUT/cpu.prof CPUPROFILE_FREQUENCY=250 \
  wire-cell -l stderr -l $OUT/wct.log:debug -L debug -c $OUT/cfg.json      # + abtest/rss_sample.sh <pid> $OUT/rss.tsv
google-pprof --text [--cum] [--focus=<frame>] $(which wire-cell) $OUT/cpu.prof | head -40
# heap: jemalloc sampled profile (tcmalloc HEAPPROFILE records every allocation: ~10x slowdown, abandoned)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 GOGC=off \
  MALLOC_CONF="prof:true,prof_prefix:$OUT/je,lg_prof_sample:19,lg_prof_interval:31,prof_final:true" wire-cell ... 
/home/xqian/tmp/wcfm-prof/jeprof --text --inuse_space $(which wire-cell) $OUT/je.<pid>.<n>.i<n>.heap   # jeprof 5.3.0 (bin/jeprof.in)
# the truth-only tier (sec 5.1)
./run_img_evt.sh -U -C -L 4 -a 2 -O _img08to 1 122
# gates (sec 5): pre/post batches of the manifest in /home/xqian/tmp/wcfm-img08/manifest.txt into work/<evt>_img08{pre,post}/
/home/xqian/tmp/wcfm-img08/run_snap.sh _img08pre; <build>; /home/xqian/tmp/wcfm-img08/run_snap.sh _img08post
/home/xqian/tmp/wcfm-img08/gate.sh _img08pre _img08post wcfm-img08-b1      # -> abtest/snap/wcfm-img08-b1-{pre,post}/
```

Tables: `docs/08_tables/{timers,cpu_profiles,rss_ladder,heap_dumps}.md` are the raw profile outputs;
the numbers quoted below come from them and from the logs named in each section. Profile runs, dumps
and batch logs live under `/home/xqian/tmp/wcfm-prof/` and `/home/xqian/tmp/wcfm-img08/` (scratch).

## 1. The uncut FD chain is not slow: FD vs PDHD, PDVD, SBND

**Same chain, same settings.** `wcfm/img.jsonnet` is a verbatim fork of the PDHD module: FrameFileSource →
Reframer → CMMModifier → FrameMasking → ChargeErrorFrameEstimator → an active fork of four MaskSlices+GridTiling
passes (planes [012], [01]+masked 2, [12]+masked 0, [02]+masked 1, tick span 4) → BlobSetMerge → BlobClustering →
ProjectionDeghosting → S1 → InSliceDeghosting(1) → ProjectionDeghosting → S2 → InSliceDeghosting(2) → S3 →
InSliceDeghosting(3) → GlobalGeomClustering → ClusterFileSink(numpy), with S = BlobGrouping → ChargeSolving(uniform)
→ LocalGeomClustering → ChargeSolving(uboone), plus a masked fork (span 1500, clustering only). Every per-node
parameter equals PDHD's and PDVD's standalone runner: `nthreshold` 1e-6 (charge > 0 slicing), tiling `nudge` 1e-2,
`good_blob_charge_th` 300, `solve_config` uboone, `whiten` true, ChargeErrorFrameEstimator rebin 4 / fudge
[2.31, 2.31, 1.1], CMM `org_hlimit` 8500. No detector sets any LASSO or ProjectionDeghosting knob
(`lasso_tolerance`/`lasso_minnorm` are parsed and never used; `nchan`/`nslice` stay at the MicroBooNE 8256 × 9592).
What wcfm adds is optional: BlobCutting (`-C`), the two BlobDepoFill truth catchers (`tru0` after BlobClustering,
`tru` after GlobalGeomClustering) and a dead-channel sidecar. SBND is the detector that differs: `nthreshold` 3.6,
`max_tbin` 3427, one GridTiling per anode, a ChannelSelector, masked span 500, its own charge-error map. Doc 07's
sentence "PDHD, PDVD and SBND run the same node chain and the same solver settings" is therefore right for PDHD/PDVD
and wrong for SBND.

**Same input size.** An FD-HD anode and a PDHD APA are both 2560 channels × 6000 ticks at 0.5 µs = 1500 slices;
PDVD is 1536 channels × 6400–10048 ticks; SBND 5638 × 3427.

**Measured cost per anode (all single-threaded, one wire-cell process per anode):**

| | anode-events | wall s min / median / p90 / max | peak RSS | blobs per anode min / median / p90 / max |
|---|---|---|---|---|
| FD uncut (events 1–10, `work/000001_N/`) | 23 | 4 / 5 / 10 / 11 | 0.45 GB (max 0.51) | 76 … 15 830 |
| FD uncut, nue shower 122/2 (this doc's profile) | 1 | 66 (of which truth catchers 30, sinks 6) | 1.5 GB | 90 291 |
| PDHD per APA (136 logs, 38 events) | 136 | 5.4 / 8.3 / 12.8 / 81.5 | 0.4–3.1 GB per event | 3 379 / 11 648 / 25 376 / 234 576 |
| PDVD per anode index (56 logs) | 56 | 2.7 / 6.0 / 10.5 / 23.6 | 0.23–0.69 GB per event | 1 526 / 11 102 / 29 427 / 80 114 |
| SBND per event (d92 stageA TSV, 16-event groups) | – | 4.7 / 6.2 / – / 8.2 | ~1 GB per 16-event process | 2 604 median per APA-event |
| FD `_sub4f` (all 180 events) | 570 | 4 / 11 / 115 / 1188 | 0.45 / 0.46 / 5.1 / 18.5 GB | 21 957 median, 628 350 max |

The uncut blob load of every FD anode-event we have is below what PDHD handles routinely: the BlobCutting *input*
count of all 570 `_sub4f` anode-events (which equals the uncut blob count, checked on the 23 pairs) is median 1 785,
p90 8 381, max 93 151 — below the PDHD per-APA median and far below its maximum. Per unit of work the FD chain is
PDHD's: ChargeSolving + ProjectionDeghosting seconds per 1 000 blobs on anodes with ≥ 5 000 blobs are PDHD 0.10 /
0.11 / 0.19 / 0.21 (min / median / p90 / max), PDVD 0.06 / 0.10 / 0.12 / 0.15, SBND 0.19, FD uncut 0.08 / 0.21 / – /
0.25 (5 anodes), FD `_sub4f` 0.08 / 0.36 / 1.28 / 4.39. The factor ≈ 2 of FD uncut over PDHD at matched blob count
(e.g. 9 934 blobs: 2.4 s vs 1.0 s) goes with far fewer measures and larger components (115 blobs per solver
subcluster vs 6 on PDHD) — both faces live, wrapped U/V — and is not what the owner asked about.

**Conclusion 1.** There is nothing to fix in the FD chain's uncut configuration, and nothing the production
detectors do that FD does not (their perf campaigns shipped as code, not knobs: tcmalloc, `GOGC=off`, precompiled
JSON, per-anode processes, the numpy sink, masked span 1500 — wcfm uses all of them).

## 2. The cost is the `_sub4f` research tier

BlobCutting at 4 wires ahead of BlobClustering (doc 03; the GNN dataset's sub-blob generator) multiplies the graph
the legacy chain then has to deghost and solve. On the 14 uncut/cut anode-event pairs with ≥ 1 000 uncut blobs:
blobs × 3–13, blob-blob edges × 9–42 (edges per blob 2.7 → 9–12), ChargeSolving + ProjectionDeghosting time × 2–43.
Two structural reasons, both visible in the logs and the code:

- BlobCutting halves the widest strip only and re-adds the other two unchanged (`img/src/BlobCutting.cxx:119-145`),
  so sibling pieces keep identical wire ranges in the two uncut planes. They therefore join the same BlobGrouping
  measures and land in the same ChargeSolving component: on 122/2 the 570 k blobs form 1 200 components with
  14 391 measures — about 475 blobs × 12 measures each, up to 13.5 k blobs in one slice
  (`in=944 cut=698 -> sub=13296`). The LASSO Gram of such a component is nearly dense (Σ over measures of k² ≈ N²).
- The first two InSliceDeghosting rounds remove almost nothing on cut pieces (570 k → 521 k blobs); only round 3,
  after the last solve, removes 87 % (→ 66.7 k). So all six ChargeSolving passes and both ProjectionDeghosting passes
  run on the full sub-blob graph.

Per-node wall on 122/2, `_sub4f` (`docs/08_tables/timers.md`, the CPU-profiled run, 1184 s total):

| node | wall s |
|---|---|
| BlobDepoFill tru0 (628 k blobs, 274 k depos on face 1) | 303 |
| ChargeSolving × 6 (83–92 each) | 522 |
| ProjectionDeghosting 1st / 2nd | 90 / 82 |
| InSliceDeghosting 1st / 2nd / 3rd | 27 / 7 / 4 |
| BlobClustering | 26 |
| LocalGeomClustering × 3 | 20 / 16 / 13 |
| BlobDepoFill tru (66.7 k blobs) + tru0/tru sinks | 19 + 19 + 2 |
| BlobGrouping × 3 | 8 / 7 / 5 |
| BlobCutting | 2 |

What the production campaigns already did and what they left (`toolkit/img/docs/examinations/efficiency-concerns.md`,
`toolkit/img/docs/imaging-timing-profile.md` (SBND), `toolkit/clus/docs/imgclus-optimization-log.md`): the imaging
quadratics that dominate an uncut APA are gone — BlobGrouping per-wire cache (c18e06b9, once 78 % of the job),
ProjectionDeghosting projection eviction (448a192d, −51 % RSS), the BlobShadow flat edge list (44887604), Gram
symmetric half + zero-beta skip + support-pair enumeration in LassoModel (22579bf9, 8726f292), `move_graph` between
stages (1b0e1b60), SBND R1/R2/R5. Left as "NOT FIXED" and exactly what the cut tier hits: concern 1
(InSliceDeghosting 2-view × 3-view loops; round 1 lacks the `break` round 2 has), concern 4 (BlobDepoFill depo ×
blob), the BlobShadow pair enumeration Σ k_w², and the LASSO Gram assembly (round 9 dropped a LASSO restructure
because Fit was ≤ 15 % of the worst production anode — on this tier it is 40 %). Concern 13 (BlobGrouping) is
listed as unfixed but was fixed by c18e06b9.

## 3. Profiles: where the CPU and the memory go

### 3.1 CPU (gperftools sampler under tcmalloc, `docs/08_tables/cpu_profiles.md`)

Uncut 122/2 (66 s, 17 k samples): BlobDepoFill 44 % of samples, of which 70 % are shared_ptr refcount traffic
(`_Sp_counted_base::_M_release` + `__atomic_add`) — the inner loop copies the blob pointer out of the graph variant
for every (depo, wire, blob) triple and recomputes four ray crossings and two heap vectors per blob per depo
(`img/src/BlobDepoFill.cxx:325-365` before this doc); ProjectionDeghosting 13 %; the six ChargeSolving 10 %; the
sinks 8 % (`ClusterArrays::to_arrays`).

`_sub4f` 122/2 (1184 s, 295 k samples):

| frame (cumulative) | share | what it is |
|---|---|---|
| `LassoModel::Fit` | 39.8 % | of which Gram assembly ≈ 27 % of the job: `setFromTriplets` 13.4 %, `SparseMatrix::operator=` 6.8 %, triplet `emplace_back` 7.2 %; the coordinate-descent sweep itself (Fit self) 12.4 % |
| `BlobDepoFill::operator()` | 27.2 % | 78 % of it refcount traffic, as uncut |
| `ProjectionDeghosting::operator()` | 14.4 % | 80 % of it `BlobShadow::shadow_list`: `try_emplace` / rehash / node destruction of the `layer_edges` unordered_map |
| `RayGrid::associate` (BlobClustering + LocalGeomClustering) | 6.0 % | per-blob `std::set` per layer, 11.7 bb edges per blob |
| `LocalGeomClustering` / `InSliceDeghosting` | 3.9 % / 3.0 % | |

### 3.2 Memory (RSS ladder every 0.5 s + jemalloc in-use dumps, `docs/08_tables/{rss_ladder,heap_dumps}.md`)

| phase (122/2 `_sub4f`) | RSS after | in use (jeprof) | held by |
|---|---|---|---|
| BlobClustering done | 2.9 GB | 2.5 GB | the IndexedGraph built by `geom_clustering` (vertex index 1.2 GB, edges 0.4 GB), the blobs 0.07 GB |
| tru0 BlobDepoFill (300 s) | 3.1 GB | 2.5 GB | flat |
| tru0 `make_cluster` + ClusterFileSink | 8.2 GB | 6.6 GB | `copy_graph` 1.95 GB + `ClusterArrays::to_arrays` 2.15 GB, freed after the write |
| **ProjectionDeghosting 1st, mid-pass** | **15.0 GB** | **13.9 GB** | **`BlobShadow::shadow_list` 11.5 GB**: the (blob-pair, layer) → edge map and edge vector, accumulated over all 551 slices, freed at the end of the pass |
| after ProjectionDeghosting | 15.7 GB (tcmalloc high-water mark) | 2.8 GB | |
| the six solves | flat | ≤ 6.0 GB | `LassoModel::Fit` 3.3 GB (triplets + Gram + `setFromTriplets` temporary), a 1.8 GB `copy_graph` |

So the peak is the BlobShadow pair map of the first ProjectionDeghosting pass on the pre-deghost sub-blob graph,
and tcmalloc never returns that high-water mark (under jemalloc the same run ends at 3 GB RSS after a 13.4 GB
high-water mark). Doc 07's hypothesis that the flat 15–19 GB was a LASSO Gram transient is refuted: the Gram peaks at
3.3 GB, after the RSS plateau is already set. The plateau is flat across blob counts because the shadow map scales
with Σ over (slice, wire) of k_w² of the busiest slices, not with the blob total.

## 4. Options, ranked by what the profiles say

| option | what it buys on 122/2 | byte-identical? | taken |
|---|---|---|---|
| truth-only tier knob: end the active fork after the `tru0` catcher | −93 % wall (no PD / solve / ISD / apa / tru), −60 % peak; everything the GNN dataset and the E1/E2 probes read is the tru0 archive + frames | config only, default OFF, JSON unchanged when off | **yes, §5.1** |
| BlobDepoFill blob-by-wire index + hoisted per-(blob, wire) geometry | −300 s (27 % of the job; 30 s of every uncut truth run) | yes: same doubles, same per-blob accumulation order | **yes, §5.2** |
| BlobShadow: slice-local dedup map | the 11.5 GB peak → the largest slice's share; smaller hash tables (−12 % CPU) | yes: keys are slice-local, first-encounter order kept | **yes, §5.3** |
| LassoModel: assemble the Gram directly in compressed-column form | −27 % of the job, −3 GB transient; production LASSO share ≈ 10–15 % smaller | yes: same values, same per-column (row, value) sequence | §5.4 |
| InSliceDeghosting round-1 `break` at `count == 2` | −3 % | yes (idempotent insert) | §5.5 |
| ProjectionDeghosting `nchan`/`nslice` at the anode's real sizes | untested; `judge_coverage` walks `nslice` columns per call | config only | no: not measured to matter here |
| cut at 8 wires, nue off the cut tier, `full_deghost=false`, higher `nthreshold`, `hash_setS` | large | **no** (result-changing; `hash_setS` rejected in the SBND campaign) | no |
| IndexedGraph vertex index (1.2 GB), sink `to_arrays` (2.1 GB) and `make_cluster` copies | transients below the new peak | – | no |

## 5. Changes taken, with gates

Manifest (`/home/xqian/tmp/wcfm-img08/manifest.txt`, 16 anode-events): uncut events 1, 2, 7, 8 on every simulated
anode (9 anode-events) and 122/2; `_sub4f` (cut 4) anode-events 8/11, 2/2, 7/11, 215/0, 216/4, 126/2. Pre and post
arms are imaged into new directories `work/000001_<evt>_img08pre/` and `_img08post/` (`-O`, existing dirs untouched;
the uncut 122/2 pre arm is the profile run's output, same build). Every archive (`apa` active + masked, `tru0`,
`tru`) is compared by `abtest/hash_archive.py` member content; hashes, timing and the verdict are under
`abtest/snap/wcfm-img08-b<N>-{pre,post}/`. Production code additionally runs the standard PDHD/PDVD imaging gate
(`abtest/run_events.sh <label> img` on `abtest/events.txt`, `ab_compare.sh`).

### 5.1 The truth-only tier (`truth_only`, default OFF) — wcp only

`wcfm/img.jsonnet` config key `truth_only` (false) and TLA `truth_only` of `wct-img-all.jsonnet`; runner flag
`run_img_evt.sh -U` (refused together with `-T`; `img-provenance.txt` gains `truth_only=`). With the knob on and
`depos` given, the active fork is `[slicing/tiling] → BlobClustering → [BlobCutting] → BlobDepoFill → tru0 sink`
(the catcher's fan is not needed: the cluster goes straight into BlobDepoFill port 0) and no apa dump is attached;
the masked fork is unchanged. Compiled-config proof: `wcsonnet` output with the knob off is byte-identical to the
pre-change output for the plain, depos and depos+cut TLA sets (`cmp` on `/home/xqian/tmp/wcfm-img08/{pre,post}_*.json`);
with the knob on the JSON holds 0 ProjectionDeghosting / ChargeSolving / InSliceDeghosting nodes, 1 BlobDepoFill and
the `trusink-tru0` + masked sinks only. The tru0 archive it writes is hash-identical to the full chain's
(`0beeaa2c…` for 122/2: truth-only run, profile run and the existing `_sub4f` archive agree).

`scripts/gnn_dataset.py --legacy-sub <suffix>` reads the legacy `apa` archive from another tier (the uncut one,
`''`) when the sub-blob tier has none: a sub-blob is "present"/"kept" when an uncut blob of the same face and slice
whose three wire ranges contain it exists / survived the solver (containment join instead of the exact geometric
key). Default = `--sub`, unchanged behaviour.

Cost on 122/2 `_sub4f`: 1184 s / 15.7 GB → 355 s / 6.6 GB before §5.2, **59 s / 6.6 GB** after it (the 6.6 GB is
the sink's `to_arrays` + `copy_graph` transient on the 628 k-blob graph).

### 5.2 BlobDepoFill: blob-by-wire index (toolkit `img`, truth-only component)

`img/src/BlobDepoFill.cxx`: per slice, each blob is visited once to take a raw pointer, its primary-plane wire range
and, per wire it covers, its along-wire extent [wlo, whi] from the four ray crossings (a function of the blob and
the wire only); a `wip → [blob …]` index in the slice's blob order replaces the scan over every blob for every
(depo, wire). Each blob's charge is still accumulated in (slice, depo, wire) order from the same doubles, so the
output is bit-identical. Doctest added (`img/test/doctest_blobdepofill_bounds.cxx`, "fills each blob independently"):
a blob filled alone must carry bit-for-bit the value it gets filled with the whole slice, for several depos — the
invariant the index relies on. `./build/img/wcdoctest-img`: 229 assertions pass.

Gate `wcfm-img08-b1`: `wcfm-img08-b1: 88 archives compared, 0 differ -> PASS` (`abtest/snap/wcfm-img08-b1-{pre,post}/hashes.txt`; the uncut arms of 2/2, 7/11 and 8/11 were overwritten by their cut runs in both arms — same output directory — so the uncut coverage is 1/8, 1/10, 2/4, 7/8, 7/9, 8/9 and 122/2; the truth-only 122/2 tru0 archive is hash-identical pre/post as well). Wall (`time_img_a<N>.txt`, tru0 catcher timer):

| anode-event | tier | wall s pre → post | peak RSS GB pre → post | tru0 catcher s pre → post |
|---|---|---|---|---|
| 122/2 | uncut | 68 → 40 | 1.4 → 1.4 | 21.8 (profiled) → ? |
| 126/2 | cut 4 | 690 → 568 | 15.5 → 15.5 | 85.2 → 3.6 |
| 216/4 | cut 4 | 116 → 112 | 4.7 → 4.5 |  →  |
| 8/11 | cut 4 | 105 → 84 | 10.0 → 7.2 | 8.4 → 0.5 |
| 2/2 | cut 4 | 110 → 85 | 4.5 → 4.5 | 12.5 → 0.2 |
| 7/11 | cut 4 | 96 → 74 | 2.7 → 2.4 | 18.6 → 0.5 |
| 215/0 | cut 4 | 43 → 38 | 2.3 → 3.0 | 1.1 → 0.1 |
| 1/10 | uncut | 6 → 5 | 0.5 → 0.5 |  →  |
| 7/9 | uncut | 9 → 7 | 0.5 → 0.5 |  →  |
| 122/2 | cut 4, truth-only (`-U`) | 355 → 59 | 6.6 → 6.6 | 303 → 1.0 |

(122/2 uncut pre = the CPU-profiled run, 68 s under the sampler; the post arm's tru0 catcher took 1.0 s and tru 0.8 s against 21.8 s and 7.8 s in the profiled pre run.)


### 5.3 BlobShadow: slice-local pair map (toolkit `aux`, production code)

`aux/src/BlobShadow.cxx::shadow_list`: the `layer_edges` map that merges the wires a blob pair shares is cleared at
the start of each slice's loop. Both blobs of a pair are out-neighbours of the same slice vertex and a blob belongs
to one slice, so no key can recur across slices: the edges, their first-encounter order and their beg/end are
unchanged. `./build/aux/wcdoctest-aux`: 111 331 assertions pass.

Gates: `wcfm-img08-b2` `wcfm-img08-b2: 88 archives compared, 0 differ -> PASS` (pre = the B1 post arm, `abtest/snap/wcfm-img08-b2-{pre,post}/`); standard PDHD/PDVD `img08-b2-{pre,post}` `=== OVERALL: PASS ===` (`abtest/snap/img08-b2-{pre,post}/`, 4 PDHD + 2 PDVD events, every `clusters-apa-*` archive; `ab_compare.sh` log in `/home/xqian/tmp/wcfm-img08/logs/prod_b2_compare.txt`).

| anode-event | wall s pre → post | peak RSS GB pre → post |
|---|---|---|
| 126/2 cut 4 | 568 → 518 | 15.5 → 12.4 |
| 8/11 cut 4 | 84 → 77 | 7.2 → 6.4 |
| 7/11 cut 4 | 74 → 67 | 2.4 → 2.3 |
| 216/4 cut 4 | 112 → 100 | 4.5 → 5.8 |
| 2/2 cut 4 | 85 → 82 | 4.5 → 4.6 |
| uncut 1/8, 1/10, 2/4, 7/8, 7/9, 8/9, 122/2 | unchanged (4–40 s) | unchanged (0.45–1.4) |
| PDHD 027305/0 (production gate, 4 APAs) | 138 → 136 | 1.98 → 1.67 |
| PDHD 027409/0, 027980/3, PDVD 039349/0, 039252/5 | unchanged | unchanged |

On 122/2 the first ProjectionDeghosting pass went from 90 s to 59 s. The peak did not fall as far as the 11.5 GB
in-use figure suggested: the busiest slices of a shower anode hold most of the pairs (13.5 k sub-blobs in one
slice give up to 9 × 10⁷ pairs × 3 layers on their own), so the largest slice's map is still several GB, and
tcmalloc's high-water mark then sits at whichever transient is largest (on 216/4 the peak even rose by 1.3 GB —
a different allocation sequence fragmenting differently, not more live memory). The remaining pair-map memory
is inherent to enumerating every blob pair per shared wire; halving it further needs a different shadow
algorithm (result-order-sensitive) and is not attempted here. (On 122/2 the full chain on the B2 build: 763 s and 14.7 GB against 1184 s and 15.7 GB.)


### 5.4 LassoModel: direct compressed-column Gram assembly (toolkit `util`, production code)

`util/src/LassoModel.cxx::Fit`, sparse-X branch only (the imaging blob-measure matrix and the Q/L normal
equations; the dense-X branch is untouched): the support-overlap enumeration already yields, for each column in
ascending order, the diagonal and then the pairs (i, j > i) in ascending j. Scattered as (row i, col j) and (row j,
col i) that is every column's rows in ascending order, so the Gram is filled straight into the compressed-column
matrix (`reserve` with the exact per-column counts, `insertBackUncompressed`, `makeCompressed`) instead of
`tripletList` + `setFromTriplets` + `operator=`: the same doubles in the same per-column (row, value) sequence, so
the coordinate-descent sweep is unchanged. New doctest `util/test/doctest_lassomodel_gram.cxx` (the dense-X,
sparse-X and Eigen-sparse-product paths agree to 1e-8 on an imaging-like and on a dense response);
`./build/util/wcdoctest-util` passes in full.

Gates (pre = the B2 build, post = the B3 build; libraries unchanged during the arms, checked by mtime):

| gate | verdict |
|---|---|
| wcfm manifest `wcfm-img08-b3` (`abtest/snap/wcfm-img08-b3-{pre,post}/`) | 88 archives compared, 0 differ → PASS |
| PDHD/PDVD standard imaging manifest `img08-b3-{pre,post}` (pre = `img08-b2-post`) | `=== OVERALL: PASS ===` (65 archive PASS lines) |
| SBND standalone imaging, 2 mcp1k events (frames from `work-mcp1k-d123lgflip/evt{168526,168614}/frames-dnn.tar.bz2`, outputs in scratch; the mergegate `input-10evt-mc` set no longer exists) | 8 `icluster-apa*.npz` compared by array content, 0 differ → PASS |
| uBooNE Q/L sweep `qlport/scripts/sweep/img08b3{pre,post}` (35 events, `ab_check.sh`) | Bee zips 35/35 content-identical; tagger logs 34 identical, 1 DIFF = event 6805, the known bistable event (`kine_pio_angle` 14.81 vs 109.51 alternates run to run on one binary, heap-address tie-break in `shower_less`; memory `feedback_clustering_global_is_t0_corrected`): null pair on the B3 build, `repeat_check.sh 22 3 img08b3rep 3`: Bee zips 3/3 identical, tagger hashes 2 distinct of 3 runs — the same flip without any code change, so the sweep is PASS modulo the known bistable event |

Wall and peak RSS (`time_img_a<N>.txt`, cut-4 anode-events, B2 build → B3 build):

| anode-event | wall s | peak RSS GB |
|---|---|---|
| 122/2 (full chain, separate runs) | 763 → 652 | 14.7 → 14.4 |
| 126/2 | 518 → 464 | 12.4 → 12.9 |
| 216/4 | 100 → 87 | 5.8 → 3.3 |
| 8/11 | 77 → 68 | 6.4 → 4.1 |
| 2/2 | 82 → 75 | 4.6 → 2.5 |
| 7/11 | 67 → 60 | 2.3 → 1.5 |
| 215/0 | 35 → 34 | 3.0 → 1.4 |
| uncut anode-events, PDHD/PDVD/SBND/uBooNE | within noise | within noise |

On 122/2 the six ChargeSolving passes went from 522 s (profile run) to 82–92 s each on B2 to 59–67 s each on B3;
the triplet transient is gone, which is what halves the peak on the mid-size cut anodes. On the two shower
anodes the peak is now set by the end of the first ProjectionDeghosting pass plus BlobGrouping and the first
solve (RSS ladder of the B3 run: 8.2 GB after the tru0 sink, 13.5 GB by the first solve, 14.4 GB peak).

### 5.5 InSliceDeghosting round-1 `break` — not taken

3 % of the cut-tier job (`InSliceDeghosting.cxx:603-611`, the `count == 2` early exit that round 2 already has);
production code, so it needs the PDHD/PDVD arm again. Left for a later round; the change is one line and its
byte-identity argument (idempotent set insert) is in §4.

**Overall on the worst anode (122/2, cut 4, full chain): 1184 s / 15.7 GB → 652 s / 14.4 GB; the same anode as a
truth-only tier: 59 s / 6.6 GB.** The uncut FD chain, which is what production would run, keeps its 4–11 s per
anode with the truth catchers now costing ~1 s instead of 8–30 s.

## 6. What this means for the campaign

- The `_sub4f` tier is now affordable as a truth-only tier for the GNN/E2 work (`-U`): the shower anodes cost a
  minute instead of twenty, and the tier that E1/E2 read is byte-identical to before.
- The full legacy chain on cut sub-blobs remains a research configuration: after §5.2–5.3 its cost is the LASSO on
  near-dense components (§2) and that is inherent to the 4-wire cut, not to the FD chain.
- For FD production (uncut) the only FD-specific cost is the truth catchers, now negligible; the chain itself is
  PDHD's, and the PDHD/PDVD/SBND lessons are already in it.

## 7. Files

- `wcfm/img.jsonnet`, `wcfm/wct-img-all.jsonnet` (`truth_only`), `wcfm/run_img_evt.sh` (`-U`),
  `wcfm/scripts/gnn_dataset.py` (`--legacy-sub`).
- toolkit (uncommitted at the time of writing, one commit each intended): `img/src/BlobDepoFill.cxx` +
  `img/test/doctest_blobdepofill_bounds.cxx` (B1), `aux/src/BlobShadow.cxx` (B2), `util/src/LassoModel.cxx` +
  `util/test/doctest_lassomodel_gram.cxx` (B3).
- `docs/08_tables/`: `timers.md`, `cpu_profiles.md`, `rss_ladder.md`, `heap_dumps.md` (profiles), `gates.md` (every gate's
  verdict and timing lines verbatim).
- Scratch (not committed): `/home/xqian/tmp/wcfm-prof/` (profiles, heap dumps, `jeprof`), `/home/xqian/tmp/wcfm-img08/`
  (manifest, batch/gate/build scripts and logs), `work/000001_<evt>_img08{pre,post}/`, `work/000001_122_img08to{,_post}/`.

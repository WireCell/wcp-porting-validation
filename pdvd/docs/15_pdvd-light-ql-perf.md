# 15 — PDVD light + QLMatching performance/memory campaign (120 events)

Scope: running time and memory of the PDVD **light signal processing** chain
(`run_light_evt.sh` → `wct-light-reco.jsonnet`: OpDecon → OpRoi → OpHitFinder
→ OpHitMerge → OpFlashFinder) and of **QLMatching** (inside
`run_clus_evt.sh` → `wct-clustering.jsonnet`), over the 120-event validation
set: runs 039252 (18 events), 039253 (18), 039349 (84, 3 stream files).
Base clustering (MultiAlgBlobClustering stages) is **out of scope** — it is
profiled here only to place QLMatching's share in context.

Ground rule: every optimization in this campaign is **bit-identical**
(member-content hash via `abtest/hash_archive.py`; gate label below) or is
not shipped. Anything that can only be approximated is written up as a
default-OFF-knob proposal, not enabled.

Prior art (not redone here): QLMatching sparse LASSO (`sparse_lasso` ON in
PDVD cfg), cull_cross_tpc AABB pruning, vis-loop hoists + buffer reuse
(`match/docs/qlmatching-perf-event60933.md`,
`match/docs/qlmatching-perf-evt1015-pdhd.md`); light round 1: lazy template
FFTs, cached AutoScale, `use_real_dft` ON, folded postfilter
(`flash/docs/light-perf-round1.md`).

## Repro

```bash
cd wcp-porting-img/pdvd
# instrumentation (round-0 commit): PDVD_RESMON sampler in run_clus_evt.sh,
# scripts/stage_ql_tag.sh, scripts/collect_perf.py, profile_ql.sh

# light population (fresh suffix _perf0), 120 events
./run_light_all.sh -s _perf0 39252
./run_light_all.sh -s _perf0 39253
for f in input_data_light/np02vd_raw_run039349_*_rawwf.root; do
    ./run_light_all.sh -f "$f" -s _perf0 39349
done

# QL population (fresh tag perfql0; matching-only rerun on _keep clusters)
while IFS=$'\t' read -r run idx evt; do [ "$run" = run ] && continue
    ./scripts/stage_ql_tag.sh "$run" "$idx" perfql0
done < scripts/perf_manifest.tsv
for r in 39252 39253 39349; do
    PDVD_LIGHT_SUFFIX=_keep ./run_clus_evt.sh "$r" all -s perfql0
done

# collect
./scripts/collect_perf.py light _perf0 > light_perf0.tsv
./scripts/collect_perf.py clus  perfql0 > ql_perfql0.tsv

# profiles (wcsonnet precompile inside; gperftools; M17-safe)
./profile_ql.sh 39253 8        # worst QL event (evt 49846)
./profile_ql.sh 39253 4        # median QL event (evt 49766)
./profile_light.sh 39349 19509 # worst light event
./profile_light.sh 39252 298595 # median light event
```

Baseline hash snapshot (gate reference for every change in this campaign):
`abtest/snap/pdvd-lightql-perf-r0/{light,ql}-hashes.txt` — 120 light
`opflash_pdvd-wct.tar.gz` rollups + 3360 QL archives (28/event: 27
`mabc-*.zip` + `trash-all-apa.tar.gz`).

Determinism pre-check (prerequisite for single-run gates): light evt 298567
run twice (`_det0a`/`_det0b`) → opflash hash identical; QL 039252 idx 0 run
twice (`resmonchk_on`/`resmonchk_off`, also the RESMON on/off validation) →
all 28 archives hash-identical.

## Round 0 — baseline (toolkit b9cb1561, wcp a1fef6f)

Conditions: 6-way batch (`PDVD_MAX_JOBS=6`, load ~48 on 64 cores), production
runner defaults (doc-19 operating point, `light_model=library`,
`sparse_lasso` on). Batch walls are contention-inflated (solo light run:
1.7–2.0 s vs 5.1 s batch median); every A/B in this campaign compares
equal-condition batches, and headline profiles are solo runs.

### Populations (n=120)

| metric | min | med | mean | p90 | max | worst event |
|---|---|---|---|---|---|---|
| light wall (s, batch) | 3.05 | 5.07 | 5.18 | 6.46 | 7.72 | 039349/19509 |
| light peak RSS (GB) | 0.31 | 0.40 | 0.39 | 0.41 | 0.42 | — |
| clus+QL wall (s, batch) | 17 | 32 | 34.2 | 51 | 78 | — |
| clus+QL peak RSS (GB) | 0.3 | 0.6 | 0.6 | 1.0 | 1.2 | — |
| QLMatching total (ms) | 3133 | 6792 | 6795 | 9099 | 10939 | 039253 idx 8 / evt 49846 |
| — build_bundles (ms) | 2502 | 5698 | 5643 | 7583 | 9203 | (83% of QL total) |
| — vis_loop (ms) | 661 | 1549 | 1576 | 2470 | 3386 | (28% of build_bundles) |
| — fit (ms) | 18 | 125 | 128 | 221 | 275 | (sparse LASSO: solved) |
| — xtpc_cull (ms) | 29 | 151 | 194 | 358 | 963 | |
| — output (ms) | 194 | 399 | 405 | 541 | 679 | |

QLMatching is ~20% of the clus-job wall (the rest is base clustering, out of
scope). TSVs: `pdvd/docs/perf/light_perf0.tsv`, `pdvd/docs/perf/ql_perfql0.tsv`
(4/120 rows miss the `QLtiming operator` sub-split — interleaved log lines;
`ql_took_ms` itself is present for all 120).

### Profiles — where the time actually is

**QLMatching (worst event 039253 idx 8, gperftools 1 kHz, 11617 samples):**

```
2447  21.1%  QLMatching::operator()
1966  16.9%    build_bundles
1300  11.2%      compute_endpoint_flags     <- 66% of build_bundles
1073   9.2%        Blob::points (inside endpoint_flags)
1417  12.2%    Blob::points (total)
 264   2.3%    PhotonLibraryModel::visibilities
```

The pre-round-0 assumption ("~90% of QL is the vis loop") is **wrong for
PDVD**: with the PhotonLibraryModel (trilinear grid gather, cheap) the
dominant cost is `compute_endpoint_flags`, called once per (flash × group)
bundle (~190 × ~30 × 2 ApaRuns ≈ 10k calls/event). Each call re-walks the
cluster's nested `time_blob_map`, calls `Blob::points("3d")` **per slice**
(string-keyed dataset lookup + selection + full point-array copy, just to
read the first point's x), and re-sorts — although everything except the
constant `flash_x_offset` shift is flash-independent. The flat profile is
correspondingly string-compare-bound (`basic_string::compare` +
`__memcmp` ≈ 22% of job CPU). Second target: the vis loop's own
`blob->points()` refetch per (flash × group × blob) (~12% of
build_bundles). Median event (039253 idx 4) has the same shape:
build_bundles 13.3% of job CPU, endpoint_flags 8.3%, Blob::points 9.1%,
PhotonLibraryModel 2.2%.

**Light (worst 039349/19509 + median 039252/298595, ~380 samples each):**
only ~0.4 s of the ~2 s solo wall is CPU; the rest is ROOT input I/O. Of the
CPU: ~30–40% is FFTW **planning** (mkplan/search0/md5), `OpDecon::deconvolve`
~33–38% cumulative (spectral loop + FFT execution), OpHitFinder ~3%. Heap
(tcmalloc, worst event): 92 MB sampled live, 62% transient waveform vectors —
memory is not a light-chain problem (peak RSS 0.31–0.42 GB). Conclusion:
light-chain headroom is small in absolute terms; the shippable bit-identical
levers are the dead upper-spectrum work and allocation churn in
OpDecon/OpRoi (round 1, L-items), worth ~a fraction of the 0.4 s CPU.

### Round-1 plan (evidence-ranked)

1. **QL-1a** — cache the flash-independent slice scan of
   `compute_endpoint_flags` per cluster (raw x of each slice's first blob
   point, per-slice blob/point/charge tallies, min/max raw ticks), rebuild
   `u` per flash with the identical FP ops, keep the per-call sort.
   Expected: removes ~2/3 of build_bundles CPU.
2. **QL-1b** — hoist the vis-loop `blob->points()` fetch into a per-cluster
   contiguous-array cache built once per ApaRun; add a conservative per-blob
   AABB pre-gate (exact FP bounds; skipped blobs replay only the
   `total_charge_point` accumulation, which is FP-order-identical).
3. **L-1/L-2** — OpDecon/OpRoi: skip the upper-Nyquist half of the spectral
   loops under `use_real_dft` (both `IDFT::inv_c2r_1d` implementations
   provably ignore those bins — `iface/src/IDFT.cxx` enforces Hermitian
   symmetry, FftwDFT uses native c2r), drop the `xG` full buffer on the
   wiener-inspired path, reuse member scratch for the per-trace spectra.
4. **L-3** — `overflow_to_rail` const-first scan (skip the per-trace copy
   when no overflow run exists).
5. **QL-2** (profile-contingent) — `examine_bundle` 5×vector(m_nchan)
   scratch; decide on the round-1 re-profile.

Confirm-and-reject from the plan's candidate list (profile evidence):
per-flash opdet-mask copies and the `prefit_snapshot` map copy do not appear
in the top-100 lines of either QL profile — closed as negligible.
`SemiAnalyticalModel` micro-opts: not a PDVD lever (library model in
production); deferred.

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

## Round 1 — implemented (toolkit d3c6975d + c56a44af)

Both changes are **bit-identical with no knob**: same values, same FP
operation order, verified by full-manifest hash gates against the round-0
snapshot. Round-1 timing tags double as the gate runs (equal 6-way batch
conditions vs round 0).

### QL-1 — endpoint-flags slice cache + vis-loop point cache with AABB
pre-gate (`match` d3c6975d)

- `compute_endpoint_flags`: the `time_blob_map` walk (window-truncation tick
  bounds; per-slice first-point x, blob/point/charge tallies) is scanned once
  per cluster into `m_endpoint_slice_cache` (cleared per event alongside the
  sibling caches) instead of once per (flash × group). The per-flash drift
  coordinate is rebuilt with the identical FP ops (`u = s*((x0 + off) −
  anode_x)`) and the per-call `std::sort` is kept, so rounding-tie ordering
  matches the legacy per-call behavior exactly.
- `build_bundles`: each cluster's blob coordinates are fetched once per
  ApaRun into contiguous arrays (same values, same blob/point order),
  killing the per-(flash × group × blob) `Blob::points()` string-keyed
  fetch. A conservative per-blob AABB pre-gate skips whole blobs outside the
  PE-inclusion box at the flash's offset: every per-coordinate op (add
  constant, subtract constant, multiply by ±s) is monotone in FP, so the
  cached bounds bracket every point's `u` exactly and a box strictly outside
  on one axis fails the identical per-point compare for all points. Skipped
  blobs replay only the `total_charge_point` accumulation (it never touched
  coordinates), keeping even the debug-log totals FP-identical.

**Gate**: 120-event manifest, tag `ql1gate` vs
`abtest/snap/pdvd-lightql-perf-r0/ql-hashes.txt` — **3360/3360 archives
PASS** (plus a 5-event smoke at 140/140). `wcdoctest-match` 23/23;
freshness proof done.

**Effect** (n=120, equal batch conditions; TSV `perf/ql_ql1gate.tsv`):

| metric (ms) | r0 med | r1 med | r0 p90 | r1 p90 | r0 max | r1 max |
|---|---|---|---|---|---|---|
| QLMatching total | 6792 | 2178 | 9099 | 3355 | 10939 | 4216 |
| build_bundles | 5698 | 1064 | 7583 | 1814 | 9202 | 2615 |
| vis_loop | 1549 | 788 | 2470 | 1398 | 3386 | 2206 |

**QLMatching 3.1× at the median, 3.0× on the population sum**; clus-job
wall median 32 → 27 s; peak RSS unchanged (the point cache costs a few MB
per ApaRun, freed at end of build_bundles).

### L-1/L-2/L-3 — flash dead-work hygiene (`flash` c56a44af)

Under `use_real_dft` the inverse transform ignores bins above Nyquist by
contract (`iface/inc/WireCellIface/IDFT.h`; both the widening default in
`iface/src/IDFT.cxx` and FftwDFT's native c2r enforce Hermitian symmetry).
OpDecon's spectral k-loop + `auto_scale` fill and OpRoi's HPF multiply are
halved when no full-spectrum consumer (the folded-postfilter pedestal
accumulator) is active; the `xG` buffer now exists only for the not-cached
AutoScale path; `overflow_to_rail` scans const-first and copies the trace
only when a floor-pinned sample exists.

**Gate**: 120-event light rerun, suffix `_l1gate` vs
`.../light-hashes.txt` — **120/120 opflash archives PASS**;
`wcdoctest-flash` 31/31. **Effect**: small, as round 0 predicted (light CPU
is ~0.4 s of a ~2 s solo wall, FFTW-planning/I/O bound): solo evt 298567
wall 1.72–1.97 → 1.67–1.72 s; batch populations statistically unchanged
(median 5.07 → 5.43 s is 6-way contention noise, TSV `perf/light_l1gate.tsv`).
Shipped as bit-identical hygiene; the round-1 headline is QL-1.

### Profile-contingent items — closed with evidence

- **QL-2** (`examine_bundle` per-call vector scratch): 3 samples / 11617
  (0.03%) in the round-0 worst-event profile — far below the 1% bar.
  **Rejected**, not implemented.
- **QL-3** (opdet-mask copies, `prefit_snapshot` map copy): absent from the
  profile top-100 (`set_opdet_mask` = 1 sample). **Rejected.**
- OpDecon/OpRoi member-scratch DFT buffers: allocation is not hot in the
  light profile (FFTW planning + execution dominate; tc_new absent from the
  light top lines). **Rejected** — would add mutable-state churn for no
  measurable win.
- QL memory: worst-event heap profile shows 226 MB live dominated by the
  point-cloud tree itself (strings/vectors from `as_pctree`), peak RSS
  0.3–1.2 GB — no QLMatching-specific memory lever; memory is a non-issue
  for this chain.

## Round-2 decision — declined, campaign closed

Re-profile of the worst event (039253 idx 8, round-1 SHA, 10165 samples):

```
round 0  ->  round 1
2447 21.1%   949  9.3%   QLMatching::operator()
1966 16.9%   477  4.7%     build_bundles
1300 11.2%    62  0.6%     compute_endpoint_flags   (the round-1 kill)
1417 12.2%   134  1.3%     Blob::points (total)
 264  2.3%   247  2.4%     PhotonLibraryModel::visibilities
```

The largest remaining single QLMatching item is
`PhotonLibraryModel::visibilities` at 2.4% of job CPU — the trilinear
photon-library gather, i.e. the physics kernel itself; no bit-identical
speedup is on offer (the only lever is `vis_sample_stride` coarsening,
which changes results and stays a default-OFF owner decision). Everything
else in QLMatching is now sub-2%. On the light side the CPU is ~0.4 s/job
dominated by FFTW planning; importing FFTW wisdom would switch plan
algorithms and change round-off — not bit-identical, rejected. Per the
round-2 criterion (a remaining ≥10%-of-chain hotspot with a plausible
bit-identical lever) there is nothing left in scope: **campaign closed
after round 1.**

Base clustering (out of scope here) is now 68% of the clus-job CPU on the
worst event — the natural target of any future round, under the existing
imgclus optimization campaign rather than this doc.

## Final state

| chain | metric | before | after |
|---|---|---|---|
| QLMatching | median / p90 / max (s) | 6.79 / 9.10 / 10.94 | 2.18 / 3.36 / 4.22 |
| clus job (incl. clustering) | median wall (s, batch) | 32 | 27 |
| light | solo wall evt 298567 (s) | 1.72–1.97 | 1.67–1.72 |
| both | peak RSS | unchanged | unchanged |

Commits: toolkit `d3c6975d` (match QL-1), `c56a44af` (flash L-1/2/3);
wcp-porting-img `a1fef6f` (instrumentation), `b027021` (round-0 doc),
this commit (round-1/close). Gate labels: `abtest/snap/pdvd-lightql-perf-r0`
(reference), tags `ql1smoke`/`ql1gate`/`_l1gate` (comparisons, all PASS).
All changes are knob-free bit-identical; no revalidation of physics results
is required.

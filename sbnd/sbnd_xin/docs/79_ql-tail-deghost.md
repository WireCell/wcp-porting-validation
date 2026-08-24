# Doc 79 — Q/L heavy tail, the last straggler: ClusteringDeghost / connect_graph_ctpc (perf round 3)

Owner ask (2026-08-24): *"can you perform a similar optimization for
ClusterDeghost?  The requirement is similar to [doc 78]"* — i.e. exact
byte-identical restructurings first, any approximation busy-gated behind a
default-OFF knob, judged by the final PR results, then SBND production ON.
Doc 78 sec 9 had parked exactly this: "Deghost (`connect_graph_ctpc`,
16.5 s on 280884) — parked, different mechanism (shortest-path), the
natural next round."

**Status.**  Rounds 0–2 DONE, all gates PASS, SBND PRODUCTION ON.
Deghost on the worst event (mcp1k 280884) **16.5 → 2.9 s** (event wall
35 → 23 s at -j6); all six ≥5 s Deghost-tail events cut 3–5×.  Knob-ON
outputs **byte-identical to legacy on the whole 186-event manifest**
(the lazy path engaged on 41 events), so the "final PR results"
validation is trivially clean — plus an explicit 308-event PR gate.
Toolkit commits e84de9da (round 0) + 5a5418ec (round 1) + c8e0b9f5
(round 2) + the production flip.

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
S=<scratch>/dgtail   # this round: /home/xqian/tmp/claude-25225/.../a3a37371-*/scratchpad/dgtail

# 0a. Tail distribution from the existing ql0819 production logs (no rerun)
for d in work-mcp1k-ql0819/ql_evt*/; do e=${d#*ql_evt}; e=${e%/}
  grep -h "MABC timing: ClusteringDeghost" $d/wct_ql_evt*.log \
   | sed 's/.*took \([0-9.]*\) ms.*/\1/' | awk -v e=$e '{s+=$1} END{print e, s/1000}'
done   # -> $S/dg_times_mcp1k.txt (same for mcp2k, nuecc48, ncpi0)

# 0b. Census of 2 tail + 2 regular events (env probe added in tk e84de9da)
WCT_CTPC_EDGE_CENSUS=1 ROOT=$S/census-r0 ./run_ql_batch.sh -j 2 280884 286617 62019 58321

# 0c. CPU profile (M17: precompiled cfg, tcmalloc, 250 Hz)
cd $S/census-r0/ql_evt280884
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4 \
  CPUPROFILE=$S/dg280884.prof CPUPROFILE_FREQUENCY=250 GOGC=off \
  wire-cell -l stderr -L info -c .wct-cfg-evt280884.json
google-pprof --text --cum $(which wire-cell) $S/dg280884.prof | head -45

# 1+2. A/B on the doc-78 186-event manifest (base = doc-78 base-ql/base-ql-fix composite)
QL_EXTRA=-save-pctree ROOT=$S/dg79off-ql ./run_ql_batch.sh -j 6 -f <qltail>/ql_manifest.txt
SBND_DG_FAST=1 QL_EXTRA=-save-pctree ROOT=$S/dg79on-ql ./run_ql_batch.sh -j 6 -f <qltail>/ql_manifest.txt
python3 <qltail>/ql_gate_r1.py <qltail> ../dgtail/dg79off-ql   # and dg79on-ql
# PR gate (base arms = doc 78's r3pr-{mcp1k,nuecc48,ncpi0}, all reality=data):
PR_JOBS=8 ./run_pr_chain_batch.sh work-mcp1k-ql0819   $S/dg79pr-mcp1k   data $(cat <rm10>/mcp1k241.txt)
PR_JOBS=8 ./run_pr_chain_batch.sh work-nuecc48-ql0819 $S/dg79pr-nuecc48 data
PR_JOBS=8 ./run_pr_chain_batch.sh work-ncpi0-ql0819   $S/dg79pr-ncpi0   data
python3 scripts/pr85_hash_gate.py --jobs 8 <qltail>/r3pr-<s> $S/dg79pr-<s>
python3 scripts/pr94_root_gate.py          <qltail>/r3pr-<s> $S/dg79pr-<s>
diff <(sort <qltail>/r3pr-<s>/nusel-table.tsv) <(sort $S/dg79pr-<s>/nusel-table.tsv)
# PDHD+PDVD: abtest run_events.sh post_doc79 clus + ab_compare.sh post_doc78r3 post_doc79
# uBooNE:    qlport sweep_5384.sh doc79 4 + ab_check.sh doc79 doc78r3
```

## 1. The tail, measured (production ql0819 logs, summed ClusteringDeghost ms)

| sample | n | events ≥5 s | their values (s) |
|---|---:|---:|---|
| mcp1k | 1000 | 3 | 280884 **18.5**, 286617 9.5, 407170 5.1 (next: 2.8) |
| mcp2k | 2000 | 3 | 178410 9.6, 77885 6.8, 97172 6.6 (next: 4.8) |
| nueCC48 | 48 | 0 | max 1.8 (174637) |
| NCpi0 | 19 | 0 | max 1.3 (285567) |

Unlike the ExamineBundles tail (30 mcp1k events ≥5 s), this is a
**six-event tail** — but on those events Deghost was the dominant
remaining cost after doc 78 (280884: 16.5 s of a 37 s wall, ~45%).

## 2. Round 0 — where the time actually goes

New env-gated census `WCT_CTPC_EDGE_CENSUS` (tk e84de9da; unset ⇒ no
code path, the `WCT_RELAXED_EDGE_CENSUS` pattern) at the top of
`connect_graph_ctpc`, plus a Deghost-side ctpc cache line.

### 2.1 Census (280884 + 286617 + 62019 moderate + 58321 median)

| evt | monster ncomp | pairs | main-walk steps | kills | savable-by-early-break |
|---|---:|---:|---:|---:|---:|
| 280884 | 985 (41 760 pts) | 484 620 | 49.5 M | 3 037 | 0.30 M (**0.6%**) |
| 286617 | 702 (36 615 pts) | 246 051 | 23.4 M | 12 804 | 1.59 M (6.8%) |
| 62019 | 279+278 | 2×38.6 k | 2×4.5 M | 2×5 080 | 2×0.62 M (14%) |
| 58321 | 9 | 36 | 4 401 | 16 | 3 213 |

Three structural facts that shaped the rounds:
* The busy signal separates cleanly at the Deghost stage too: monsters
  985/702/279 vs ≤42 for everything else on these events — the doc-78
  threshold **200** carries over unchanged.
* **The doc-78 early-break win does NOT carry over**: on the 280884
  monster only 0.6% of steps are post-verdict (kills are rare — 3 037 of
  484 620 pairs), so round 1 is small here; the cost is the sheer
  O(num²) full pair walk itself.  The lever must be walking fewer pairs,
  not shorter walks.
* Every Deghost `graph_algorithms("ctpc")` request is a **cache miss**
  (`cached=false` on all 20 builds of both tail events): ClusteringSeparate
  runs earlier with the same flavor but emits *new* cluster objects, so
  nothing carries.  A separate `ctpc_fast` cache key therefore costs
  nothing.
* dir1/dir2 walks and Houghs are negligible on busy clusters (the size
  gate's first disjunct self-disables at num ≥ 100; hough_pairs=1 on the
  monster) — and the ctpc Houghs feed *geometry* (dir candidates), so
  the doc-78 site-B lazy-skip does not apply.  Deliberately not ported.

### 2.2 CPU profile (250 Hz, 7 651 samples, whole Q/L job on 280884)

| cumulative | function |
|---:|---|
| **50.6%** | `clustering_deghost` (≈100% of it `make_graph_ctpc`) |
| 47.8% | `connect_graph_ctpc` |
| 41.2% | `Grouping::is_good_point` (the 1 cm walk, 49.5 M steps) |
| 19.3% | `Simple3DPointCloud::get_closest_points` (pair kd, shared with relaxed) |
| 7.5% | `make_graph_closely` |
| — | Dijkstra: not in the top 45 |

So ~86% of `connect_graph_ctpc` is the per-pair `is_good_point` walk.

## 3. Round 1 — exact restructurings (byte-identical, no knob, all consumers)

tk 5a5418ec, in `connect_graph_ctpc.cxx` (the `"ctpc"` flavor —
also exercised by ClusteringSeparate, ClusteringCTPointCloud, and
TaggerCheckTGM, hence the all-detector gates below):

1. **Walk early break** (all three walks): the kill predicate
   `num_bad>7 || (num_bad>2 && num_bad>=0.75*num_steps)` is monotone in
   the walk (num_bad nondecreasing, num_steps fixed), so once it holds
   the verdict is decided — break.  Census mode still walks fully so the
   savable counters stay complete.
2. **Direct sentinel construction** of the five num² tuple matrices
   (drop the redundant zero-fill + overwrite pass — the fix the relaxed
   flavor already carried).

Effect (byte-identical, pure waste removal): 280884 Deghost 16.5→15.8 s,
286617 8.5→7.4 s — the census-predicted small win.  Smoke: 4 events × 5
archives all byte-identical.

## 4. Round 2 — busy-gated lazy Kruskal (`ctpc_fast`), C++ default OFF

tk c8e0b9f5, the doc-78 `relaxed_fast` scheme transplanted:

* New flavor **`ctpc_fast`** = `make_graph_ctpc` with
  `CtpcFastCfg{busy_num_threshold=200}` (connect_graphs.h; optional
  trailing `fast` pointer on `connect_graph_ctpc`, 4 registrations in
  Facade_Cluster.cxx).  `num <= 200` ⇒ legacy path bit-for-bit.
* Above the gate the per-pair main walk is deferred into an
  `eval_main_verdict(j,k)` lambda (pure code motion) and evaluated only
  when a pair would **bridge two union-find components**, in ascending
  (distance, j, k) order — single-pass Kruskal.  Eager exceptions, both
  learned in doc 78: pairs `< 3 cm` (the direct-edge rule consumes their
  verdict unconditionally) and pairs carrying a directional candidate
  (MST #2's `process_mst_deterministically` copies the MAIN entry, and a
  killed main blocks the dir edges downstream).
* `ClusteringDeghost` gains a `graph_name` config key (default `"ctpc"`
  — the previous hardcoded literal) and a populated
  `default_configuration()`; doctest pins all five defaults
  (`doctest_clus_knob_defaults.cxx`, new ClusteringDeghost case).
* Engagement on the monster: **4 of 484 620 pairs walked** (286617: 2 of
  246 051) — the 49.5 M-step cost simply vanishes; what remains is the
  pair `get_closest_points` matrix.

Wiring: `deghost()` builder gains `graph_name=null` (key-suppression);
SBND `clus_per_face`/`per_apa` thread `dg_fast` (both per-face
instances; SBND has no all-APA deghost); TLA `dg_fast` in
`wct-clus-matching-perevt.jsonnet`; runner escape `SBND_DG_FAST` in
`run_ql_evt.sh` (tri-state `knob_bool`).

## 5. Verification (all PASS)

* Freshness proof each build; `wcdoctest-clus` 2 390 assertions PASS
  (+11 new).
* Compiled-config proof: OFF compile **byte-identical** to the
  pre-change compile (`cmp` clean); ON carries `graph_name: ctpc_fast`
  ×2 (the two per-face Deghost instances); C++ default `"ctpc"` so every
  other detector (PDHD/PDVD instantiate `cm.deghost(...)` with no
  graph_name) compiles and runs unchanged.
* **SBND Q/L, knob OFF** (`$S/dg79off-ql`, 186 events × 6 archives vs
  the doc-78 base composite): **185/186 byte-identical**; the single
  FAIL is mcp2k 283833 `pctree` only — the doc-78-documented bimodal LM
  nondeterminism, and the arm hash is *exactly* the doc-78 r1-rep1
  state (`9e187c62…` vs the base's `cc578c25…`; mabc zips identical).
  Pre-existing, adjudicated, not introduced here.
* **SBND Q/L, knob ON** (`$S/dg79on-ql`): **185/186 byte-identical to
  legacy** with the same single 283833 bimodal (again the r1-rep1
  state); lazy path engaged on **41 events**.  Kruskal == per-component
  Prim empirically, as in doc 78; ties are not provably absent, hence
  the knob stays a knob.
* **SBND PR, 308 events** (base = doc-78 `r3pr-*` arms, same binary as
  the doc-79 baseline, all `reality=data`): pr85 archive hashes PASS
  482/482 + 96/96 + 38/38, pr94 ROOT gate PASS 241 + 48 + 19, sorted
  `nusel-table.tsv` diff 0 lines for all three samples — **zero
  selection-flag flips, zero score drift** (inputs byte-identical and PR
  outputs byte-identical, so the owner's "judge by final PR results" bar
  is met with zero movers).
* **PDHD+PDVD** (`ab_compare.sh post_doc78r3 post_doc79`, events.txt,
  clus): **OVERALL PASS** — this exercises the round-1 exact changes via
  Separate/Deghost on both detectors.
* **uBooNE** (`sweep_5384.sh doc79 4` + `ab_check.sh doc79 doc78r3`):
  zips **35/35 content-identical**; tagger 34/35 with the single diff at
  ev **6805** — the pr/97-class binary-LAYOUT-dependent event proven in
  doc 78 with an inert-padding control build (zips identical there too).
  Pre-existing, adjudicated.

### 5.1 Perf summary (per-event `.status` walls at -j6, off vs on arms)

| evt | wall off→on (s) | Deghost off→on (s) |
|---|---|---|
| 280884 | 35 → 23 | 16.0 → 2.9 |
| 286617 | 23 → 18 | 7.9 → 2.0 |
| 178410 | 33 → 27 | 9.2 → 2.1 |
| 77885 | 18 → 14 | 6.0 → 1.7 |
| 97172 | 18 → 14 | 5.4 → 1.2 |
| 407170 | 11 → 8 | 4.5 → 0.9 |

(The "off" Deghost values already include round 1; the pre-campaign
production values were 18.5/9.5/9.6/6.8/6.6/5.1 s.)  186-event manifest
summed walls 1 708 → 1 695 s — regular events are untouched by
construction (gate + engagement log).

## 6. SBND production ON + escapes

`wct-clus-matching-perevt.jsonnet` default flipped: `dg_fast = true`
(the doc-76/78 precedent).  Compiled-config proof: the default compile
carries `graph_name: ctpc_fast` ×2; the escape `SBND_DG_FAST=0`
compiles with zero occurrences of the key and was proven byte-identical
to the pre-change compile.  C++ default is OFF (`graph_name="ctpc"`), so
uBooNE, PDHD, PDVD and every other consumer compile and run unchanged.

## 7. What was deliberately not done

* `connect_graph_ctpc_with_reference` (`ctpc_pid`/`ctpc_ref_pid`,
  Steiner/PID consumers) — untouched; it is a separate near-copy.
* No hough lazy-skip (doc-78 site B): the ctpc houghs produce dir-candidate
  geometry, not just predicate selection — skipping is not exact.
* Deghost's own per-point 3-plane uniqueness loop — secondary in the
  profile (memoized `DynamicPointCloud::kd2d` + the 24%/25-point early
  break) and unchanged.
* `connect_graph` (the second, cheaper connector in `make_graph_ctpc`)
  — not in the profile top-45; unchanged.
* The 283833 LM bimodal and the uBooNE 6805 layout dependence remain
  reported pre-existing issues (doc 78 secs 4.1/7).

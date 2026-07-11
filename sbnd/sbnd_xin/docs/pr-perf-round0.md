# SBND pattern-recognition chain — perf round 0 (measurement + verdict)

Campaign Tier-5 (2026-07-10): first resource profile of the per-event PR tail
(M0–8 of `PR_integration.md`: pctree → switch_scope → steiner → fiducialutils
→ TGM → STM → neutrino), now that the full chain runs on SBND.  Methodology:
`run_pr_evt.sh <mode> -nu <idx>` solo per event, wall + wire-cell `VmHWM`
sampled by a wrapper (`/home/xqian/tmp/img_t3_prof/sbnd_pr_r0.sh`, tables
`pr_r0_{data,mc}_nu2.tsv`), per-stage attribution from the `MABC timing:`
debug markers, gperftools CPU profile of the heaviest event.

## 1. Population (10 data + 10 MC events, full `-nu` chain, solo)

| sample | wall (s) min/med/max | peak RSS (MB) min/med/max |
|---|---|---|
| data (686…2050) | 5.0 / 6.7 / 8.9 | 155 / 245 / 404 |
| MC (2…42) | 5.0 / 5.9 / 7.8 | 137 / 223 / 297 |

Of the 5–9 s process wall, **~4 s is fixed per-job overhead** (gojsonnet TLA
compile of `wct-pr-perevt.jsonnet` + plugin/config startup + pctree load) —
a per-event-harness artifact, absent when the PR visitors run inside the
production clustering job.  The MABC (chain) time itself is **1.0–3.5 s**.

## 2. Per-stage attribution (MABC timing markers)

- **`CreateSteinerGraph` is ~85 % of the chain**: 0.3–3.0 s/event (retile +
  ctpc_ref_pid graph + steiner tree, per beam-candidate cluster group).
- `TaggerCheckTGM` / `TaggerCheckSTM`: 1–100 ms.
- `TaggerCheckNeutrino`: ~0 ms on most events; **0.2–2.5 s when an in-window
  candidate reaches the full pattern build** (track fitting; data evt 1258 =
  2.5 s worst).
- `ClusteringSwitchScope` 2–43 ms; load/save/bee ≤ 0.2 s.

CPU inside the heaviest data event (1720, ~4.3 s process CPU, gperftools):
`connect_graph_closely_pid` 15.7 % cum (~0.7 s), nanoflann kd searches ~10 %,
allocator/Rb-tree the rest — the same shape as the uBooNE qlport profile,
scaled down by SBND's smaller per-cluster point counts.

## 3. Beam-window gating: a selection, not a perf lever

A/B on the 10 data events, default window `[0.5,2.5] µs` vs wide-open
`[-1000,1000] µs`: wall moves ≤ ±0.6 s/event (±5 %, within noise).  Opening
the window shifts work between taggers (TGM tags nothing, STM sees all,
neutrino sees more candidates: evt 2050 neutrino 0 → 292 ms) but the chain
stays steiner-dominated either way.

## 4. Verdict

- **No Tier-5 optimization round is warranted.** The whole PR tail costs
  1.0–3.5 s/event at ≤ 0.4 GB — 1–2 orders below the upstream stages
  (imaging median 44 s, NF/SP/DNNROI 61–78 s/event).
- **The shared-with-uBooNE reopen question is answered by measurement**:
  `connect_graph_closely_pid` (the qlport round-4 "not-worth-it" item) is
  ~0.7 s on the worst SBND event — the ×10 cluster multiplicity did NOT flip
  its ROI.  It stays closed.
- Revisit only if the PR tail is later run per-bundle over many more
  events/candidates than today's one-main-per-event pattern.

## 5. Crash fixed en route: empty steiner point cloud (SBND MC evt 11)

MC evt 11 (idx 3) segfaulted the `-nu` chain: its main cluster yields **zero
steiner points**, and five downstream consumers assumed a non-empty steiner
PC.  Fixed in toolkit commit (clus):

1. `SteinerGrapher::establish_same_blob_steiner_edges_steiner_graph` — null
   coordinate arrays → nothing to connect (was: segfault).
2. `Facade_Cluster::get_two_boundary_steiner_graph_idx` — null arrays now
   raise the already-designed "Empty Steiner point cloud" error instead of
   dereferencing null.
3. `CreateSteinerGraph` — the main-cluster boundary probe is non-fatal
   (warn + continue).
4. `CreateSteinerGraph` — an empty steiner PC is **not transferred** onto the
   real cluster (schema-deviant local PC broke the pctree serializer:
   `Dataset::append: missing keys wpid x_t0cor y z`); the cluster is left
   without steiner products, which every consumer already guards on.
5. `Clustering_Util::cluster_fc_check` + `NeutrinoPatternBase::
   init_first_segment` — guard on `size_major()` (points), not `size()`
   (arrays), and before the throwing call.

Gates: evt 11 now completes the full `-nu` chain (one "empty steiner_pc …
skipping transfer" warning); the other **19/19 events' `mabc-pr.zip` are
byte-identical** pre/post fix (guards are no-ops on healthy clusters); the
empty-pipeline round-trip identity was already clean.

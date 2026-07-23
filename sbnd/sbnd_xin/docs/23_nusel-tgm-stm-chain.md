# 23 — Neutrino-selection chain: Q/L bundles → TGM/STM taggers → per-bundle label table

**Goal.** Establish the downstream chain that consumes the Q/L matching result
and labels every beam-coincident bundle, as the first stage of a neutrino
selection: run the cosmic taggers (TGM = through-going muon, then STM = stopped
muon) on every matched bundle and extract a per-bundle table — flash time/PE,
main-cluster id, bundle size, tagger flags, label.  The beam light flash sits in
**(0.2, 2.2) µs**; several flashes can fall in that window, so labels are per
*bundle*, not per event.

Scripts: `run_nusel_evt.sh` (driver) + `nusel_extract.py` (table).  Chain
machinery: PR milestones M1–M8 of `sbnd/docs/sbnd-pattern-recognition.md`.
Sample: MCP2025C reco1 data, `work-mcp10`, 10 events, run 18255 (doc 21/22).

> **Round 1 (2026-07-23) exposed two reconstruction defects, both now fixed**
> (§4).  All results below are post-fix; the round-1 numbers are kept only as
> the before column, because they were dominated by the defects.

## 1. The chain

```
work/evt<ID>/icluster-*.npz ──run_ql_evt.sh -save-pctree──▶ work/ql_evt<ID>/
    (imaging, prerequisite)                                 ├─ mabc-all-apa.zip
                                                            └─ pctree-evt<ID>.tar.gz
pctree ──wct-pr-perevt.jsonnet──▶ switch_scope → steiner → fiducialutils
                                  → tagger_check_tgm → tagger_check_stm
                                  ──▶ work/nusel_evt<ID>/{log, mabc-pr.zip}
{pctree, mabc-pr.zip, mabc-all-apa.zip, PR log} ──nusel_extract.py──▶ nusel-evt<ID>.tsv
    'all' merge ──▶ work/nusel-table.tsv (per bundle) + work/nusel-events.tsv (per event)
```

Pipeline order is load-bearing: `fiducialutils` must precede the taggers (they
silently no-op without it) and TGM must precede STM (`TaggerCheckSTM` skips
TGM-flagged mains).  `tagger_check_neutrino` is deliberately NOT in this
pipeline — this is the tagger/label stage.

The Q/L step is skipped when the pctree exists.  **After any toolkit change
affecting the QL job, delete `ql_evt*/pctree-*.tar.gz` first** or the rerun
silently reuses stale trees.

### Usage

```
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all
#   -bw l,h        beam window in µs (default 0.2,2.2)
#   -save-pr-tree  also persist the post-PR tree
```

## 2. Table definition

A row is one **qualifying bundle**: a cluster that is

1. flagged `flag_main_cluster` — QLMatching's matched main, **and**
2. **in scope** — passes `switch_scope`'s active-volume filter,

i.e. exactly the population the taggers evaluate.  Main-flagged clusters
failing (2) are non-physical shards (§4.2); they are counted on stderr, not
tabulated.  Plus one `no-bundle` row per in-window physical flash with no
qualifying bundle.

**Flashes are deduplicated across APAs first.**  SBND reconstructs light per
TPC, so one physical flash yields one flash object per APA (APA1 gids offset
+1000000).  Flashes within 80 ns on opposite sides are one physical flash
(`flash_grp`) and count once — without this a bright beam flash double-counts,
which is what made `n_inbeam` read 2 for evt284349 in round 1.

| column | meaning |
|---|---|
| `run subrun event` | RSE (from the reco1 opflash tensor metadata) |
| `main_id` | main-cluster ident = the **Bee clustering-layer `cluster_id`** (verified point-by-point); −1 for no-bundle rows |
| `flash_gid`, `flash_apa` | matched flash and its TPC |
| `flash_grp` | deduplicated physical-flash id (both TPCs) |
| `flash_time_us` | `cluster_t0` in µs = matched flash time, trigger-referenced |
| `flash_pe`, `flash_pe_grp` | this flash's PE; summed over the physical flash |
| `in_beam` | 1 if in [bw_low, bw_high) |
| `n_bundle` | clusters sharing this `matched_flash_gid` |
| `npts_main`, `npts_bundle` | 3d points of the main / whole bundle |
| `len_main_cm` | end-to-end length of the **dominant merge component** (see below) |
| `n_frag` | how many pre-merge clusters were flash-merged into this one |
| `tgm`, `stm`, `label` | verdicts; `TGM`/`STM`/`nu-candidate`/`not-tagged`/`no-bundle` |

`len_main_cm` must be measured **within one merge component**.
`examine_bundles` can graft a distant fragment onto a cluster, and then any
whole-cluster extent reports the gap rather than the track: evt284349 cluster 11
reads 447 cm (bbox) or 426 cm (farthest pair) across a 3-point speck in the
other TPC, while the actual track is **202 cm**.  The per-point
`real_cluster_id` in the Bee clustering layer gives the pre-merge component, so
the extractor measures the dominant one — hence `--qlbee`.  Note Bee coordinates
are in cm while pctree arrays are in WCT internal units (mm).

Verdicts come from the PR log; the in-scope set comes from `mabc-pr.zip` (its
clustering layer *is* the in-scope set), because `scope_filter` is runtime state
and is not persisted.  The zip is also more robust than the log, whose lines can
be **torn by interleaved writes** (observed on evt286329, where a TGM verdict
line was split across 98 lines).

## 3. Results — MCP2025C reco1, 10 events (run 18255)

**74 qualifying bundles**; 45 main-flagged out-of-volume shards dropped.
Bee: `https://www.phy.bnl.gov/twister/bee/set/83cba0f2-3c3d-47a2-992e-69476a749911/event/list/`
(bee idx 0–9 = events 284349, 284657, 285185, 285999, 286021, 286065, 286197,
286241, 286329, 286527).

| label | bundles | in-window | round 1 (buggy) |
|---|---|---|---|
| TGM | 33 | **0** | 67 / 6 in-window |
| not-tagged (out-of-window) | 33 | — | 35 |
| nu-candidate | 7 | 7 | 6 |
| **STM** | **1** | **1** | 0 |
| no-bundle | 2 | 2 | 8 |

Per event: **7 nu-candidate**, **1 cosmic-tagged** (284349), **2 no-bundle**
(286021, 286197).  `n_inbeam_flash` is now exactly 1 for every event.

**Every in-window bundle now gets a physics-meaningful verdict.**  Round 1's six
in-window TGM tags were *all* out-of-volume shards; with those excluded the
in-window TGM count is zero, and the surviving in-window objects are substantial
clusters (97–304 cm, 0.3–4.4 k points).

**First genuine SBND STM tag: evt284349 cluster 11** — the 202 cm, 2176-point
beam-window track in TPC0 (t0 = 1.555 µs).  This is the cluster that round 1
mislabeled: its `flag_main_cluster` had been lost to a 3-point speck, so the
taggers never saw it, and the event's label came from a 0.8 cm shard tagged TGM.
It closes open item 6 of `sbnd-pattern-recognition.md` §6.7 ("find a genuine SBND
stopped muon").  It should be hand-scanned before being trusted.

35 of 74 bundles have `n_frag > 1`, i.e. flash-merging is the norm, not the
exception — which is why the flag-provenance fix (§4.1) matters at this scale.

## 4. The two defects found and fixed

Both C++ changes are **opt-in, default false**, so every non-SBND config is
byte-identical by construction and the uBooNE qlport gate is unaffected.
Verified: the SBND production `mabc-all-apa.zip` content hash is **unchanged**
on 3 events (`c0bdbdec…`, `97cd3968…`, `25364ce6…`) — the flags live in the
pctree `cluster_scalar`, not in any dumped Bee layer, so production output is
untouched.

### 4.1 Flag provenance lost on the flash merge

`merge_clusters` (`clus/src/ClusteringFuncs.cxx`) calls `Cluster::from()` once
per member, and `flags_from()` copies the donor's flag *values including zeros*
— so the merged cluster's flags are whichever member was visited last.  A
matched main that absorbs a tiny co-merged fragment therefore **loses
`flag_main_cluster` to it**.  The code already recognised this hazard for the
flash annotation ("override from()'s arbitrary first-wins flash with the longest
flash-bearing member") but never applied it to the flags.

Fix: `flags_from_longest` (new arg on `merge_clusters`, config key on
`ClusteringExamineBundles`) re-applies the flags of the *same* representative
member that donates the flash.  Enabled in SBND's `clus_all_apa` — note this is
the **QL** job, so the pctree must be regenerated for it to take effect.

Effect on the 10 events: main-flagged clusters **108 → 121**, matched-but-
unflagged **71 → 58**.  (Some events lose a main — the rule is deterministic, so
a flag spuriously inherited by the wrong member is now correctly *not* set.)

### 4.2 Taggers ignored the active-volume filter

`switch_scope` **separates** rather than deletes: blobs whose T0-corrected
points fall outside the active volume become their own cluster that stays in the
grouping, carrying `scope_filter = false` and an inherited `flag_main_cluster`.
The Bee writer (`filter:1`) and `clustering_examine_bundles` both honor that
flag; `TaggerCheckTGM`/`TaggerCheckSTM` did not — they selected on
`flag_main_cluster` alone.

This is not benign: an out-of-volume cluster is outside the FV *by
construction*, so the TGM CASE-A test is satisfied almost automatically.  Round-1
measurement: **46 of 107 evaluated mains (43%) were such shards, tagged TGM at
85% vs 44% for real clusters.**  This — not prototype fidelity — is the real
source of the "tiny anode fragments get tagged" behavior recorded at M7.

Fix: `require_in_scope` (new config on both taggers), on in SBND's `clus_pr`.
The taggers now log `skipped N out-of-scope main cluster(s)`.

## 5. Gotchas

- **Invoke via the real path.** `sbnd_xin` is also reachable through the
  `toolkit/sbnd_xin` symlink; the PR jsonnet's relative
  `import '../particle_dataset.jsonnet'` only resolves from the real location.
  `run_nusel_evt.sh` canonicalizes with `pwd -P` (`run_pr_evt.sh` still has this
  latent bug).
- **Regenerate pctrees after any QL-side toolkit change** (§1).
- The beam window feeds both the TGM beam protection and the table's
  `in_beam`/label columns — the driver passes one `-bw` to both; keep it so.
- Do not read tagger flags from a `-save-pr-tree` tarball: flags set on only
  some clusters do not survive serialization.

## 6. Bee display: op-layer ids are not clustering-layer ids

Two independent traps when hand-scanning:

1. **The op layer's cluster ids are the *pre-pipeline* numbering** (the code
   comment says it "runs at the same pre-pipeline point" as the img dump), while
   the clustering layer shows post-pipeline ids, and `real_cluster_id` is a
   *third* generation (pre-*merge*).  In evt284349 the beam flash shows
   `cluster 7`, the object is clustering-layer cluster **11**, and the tree
   *does* contain a different cluster 7 (a 10-point TPC1 shard) — so a naive
   op→clustering lookup lands on the wrong object.  Ids 12/13/14 appear in the
   op layer and exist in no other layer at all.
2. **The op layer hides dim matches.**  `fill_bee_flashes` drops any match whose
   predicted light is < 100 PE, emitting an empty `cluster_id` list — visually
   identical to an unmatched flash.  In evt284349, 11 flashes matched but Bee
   displays only 8; the APA1 beam flash (7.0 PE predicted vs 317 measured) is
   one of the hidden three.  The same threshold gates `store_flash_groups`, so
   those flashes also never pair across the cathode.  Both consumers are
   display-facing — the reconstruction is unaffected — but **the Bee op layer is
   not a reliable answer to "did this flash match?"**

**Workaround now:** cross-reference by flash **time**, not cluster id — and use
`nusel-table.tsv`, which carries `(flash_gid, flash_time_us, main_id)` with
`main_id` being the clustering-layer id, as the explicit join.

**Proposed proper fix (deferred, not implemented).**  Stamp a per-blob
`orig_cluster_id` at the pre-pipeline dump point and have the Bee clustering
writer emit it, so the op layer's ids become joinable regardless of how many
times a cluster is later split or merged.  `merge_clusters` already has the
per-blob `orig_id_aname` machinery (it writes `real_cluster_id`), but it
overwrites rather than preserves, and the array would also have to survive
`separate()`.  This changes `real_cluster_id`'s current semantics and is visible
in uBooNE Bee output, so it needs the qlport gate — hence deferred rather than
folded into this round.  Separately, the 100 PE display cut should either be
lowered or the hidden matches marked, so a dim match is distinguishable from no
match.

## 7. Follow-ups

1. Hand-scan the evt284349 STM tag (§3) — first of its kind on SBND.
2. Port `check_neutrino_candidate` (2dtoy `ToyFiducial.cxx:1284`): still the
   blocker for real TGM decisions on in-window bundles.
3. Run the chain over `work-mcp1000`: beam-window calibration with statistics,
   STM rate, and whether the 7/10 nu-candidate rate survives.
4. Implement §6 (op↔clustering id join) behind the uBooNE gate.
5. Check whether `cathode_connect` (also `use_flash_t0`, 800 ns window) drops
   flags the same way §4.1 did — it calls `merge_clusters` but writes no
   `real_cluster_id`, so its merges leave no trace in the Bee output.
6. Add LM (light-mismatch) and fully-contained flags to the label set when
   their ports land.

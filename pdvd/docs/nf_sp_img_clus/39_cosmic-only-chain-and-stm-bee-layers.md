# 39 — A cosmic-taggers-only PDVD chain, and an STM-scoped Bee display

**Status.** Shipped: the reduced chain is the PDVD runner default, and the Bee
set now carries three STM-scoped layers. **Not shipped, gate FAILED:** building
the Steiner graph only for STM candidates. The knob for it exists and is
verdict-neutral; its *ordering prerequisite* is not affordable (§5).

## 0. Repro block

```
# Pinned library used for every arm below (a peer rebuilds local/lib mid-campaign):
#   /home/xqian/tmp/d39/lib_d39/libWireCellClus.so   2026-09-04 17:47  md5 742f9b2df5293e83
# toolkit HEAD at the time: 20773e0b   wcp-porting-img: 128ec5dc
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd

# The deliverable: event 298595 = run 39252, event index 2, on the reduced chain.
# M13 -- fresh tag, pctree symlinked read-only from the source of truth.
mkdir -p work/039252_2_d39stm2
for f in pctree-evt298595.tar.gz pctree-evt298595.tlas img-provenance.txt; do
  ln -sf "$PWD/work/039252_2_d27fresh/$f" "work/039252_2_d39stm2/$f"; done
LD_LIBRARY_PATH=/home/xqian/tmp/d39/lib_d39 ./run_pr_evt.sh -s d39stm2 -stm-fit 39252 2

# The full-PR control, same binary, same event (shows exactly what -stm removes):
LD_LIBRARY_PATH=/home/xqian/tmp/d39/lib_d39 ./run_pr_evt.sh -s d39nu -nu -stm-fit 39252 2

# The sec.5 gate, 20 events x 3 arms (16-way parallel):
#   A  -stm                                            (production order)
#   B  -stmlean  -S steiner_skip_flags=['TGM']         (reorder + skip)
#   C  -stmlean                                        (reorder only)
docs/nf_sp_img_clus/scripts/d39_verdict_census.py work/<ev>_d39gA work/<ev>_d39gB
# full record: stm/gates/d39_tgmskip_gate.txt

# Compiled-config gate (knobs absent => byte-identical):
CFGROOT=<cfg tree at HEAD> abtest/compile_all_cfg.sh <outdir>   # 16 job configs
# plus pdvd_pr (both pipelines) and uboone_mabc compiled by hand, base vs new.
```

## 1. What prompted this

PR validation was blocked by output volume, not by a bug: every event's Bee set
carried `track_fit`, `shower_track`, `vertices` and the `mc` particle flow on
top of `clustering` and `stm_fit`. The ask was to stop the chain at the cosmic
taggers, scope the display to the STM result, and build Steiner only for
clusters that can reach the STM tagger.

## 2. Two things that were already true

Both were asked for; neither needed any work. Recording them because in each
case the **source comment says the opposite**.

- **STM already skips TGM-flagged mains.** `TaggerCheckSTM.cxx:566`,
  `if (main_cluster->get_flag(Flags::TGM)) { ...; continue; }`. The comment at
  `:564-565` — *"No existing pipeline pre-sets the flag, so this is inert unless
  tagger_check_tgm runs earlier"* — is **stale**: both the PDVD and the SBND
  production chains run `tagger_check_tgm` first, so the skip is live. Event
  298595 exercises it 31 times (`"cluster N already TGM; skipping"`).
- **STM already skips fully-contained clusters.** `TaggerCheckSTM.cxx:3421`
  returns on `fc_result.is_fc`, computed by the same `Facade::cluster_fc_check`
  (`Clustering_Util.cxx:75`) that `TaggerCheckFC.cxx:207` uses, and under PDVD
  defaults (`stm_consistent_fv=true`) with the same fiducial. STM computes it
  itself because `tagger_check_fc` runs *after* STM.

## 3. PDVD has no effective beam-coincident bundle

Worth stating because it sizes everything else. `wct-pr-perevt.jsonnet:1030`
sets `beam_window_us = [-10000, 10000]` — readout-wide, "every matched bundle is
in window" (doc 25 §2.1). The `beam_window_only` gate is *on* and selects
*everything*; the log says so directly:

```
visit: TaggerCheckTGM: beam_window_only [-10000.000, 10000.000) us: 92 main(s) evaluated, 0 out of window
```

Combined with `flag_mains` making every flash-matched cluster a main (PDVD's Q/L
matching flags none), that is 92 mains evaluated on one event, 121 clusters
built. This is the source of the volume, and it is why "build Steiner only for
STM candidates" looked worth real time.

## 4. What shipped

### 4.1 The reduced chain is the runner default

`run_pr_evt.sh:76`, `MODE=nu` → `MODE=stm`. `-nu` restores the full PR tail in
one token. No config edit was needed for the layer removal: the four PR layers
are *self-gating* on `visitor: 'TaggerCheckNeutrino:pr'`
(`protodunevd/pr.jsonnet:1826,1854,1887`), that visitor never fires without the
stage, and an unfilled Bee set is never written
(`MultiAlgBlobClustering.cxx:637`). The `mc` set bails the same way (`:3741`).

Measured on event 298595, same binary, `-nu` vs `-stm`:

| layer | `-nu` | `-stm` |
|---|---:|---:|
| `clustering` | 7 532 107 B | 7 532 107 B |
| `stm_fit` | 352 188 | 352 188 |
| `stm` | 219 532 | 219 532 |
| `steiner_graph` | 209 891 | 209 891 |
| `steiner_terminals` | 48 481 | 48 481 |
| `track_fit` | 39 574 | **absent** |
| `shower_track` | 208 392 | **absent** |
| `vertices` | 2 779 | **absent** |
| `mc` | 6 407 | **absent** |

Exactly the four named layers disappear; every other layer is byte-identical
(`hash_archive.py --members`), so the PR tail does not feed anything the cosmic
stage produces. Wall 60 s → 52 s, peak RSS 2.84 → 1.75 GB.

### 4.2 Three STM-scoped Bee layers

Two default-OFF fields on `BeePointsConfig`:

- **`require_flag`** (string, default `""`) — restricts a set to clusters
  carrying a tagger flag, tested in `fill_bee_points`'s three cluster loops.
- **`steiner_terminals_only`** (bool, default `false`) — inside the *existing*
  `pcname == "steiner_pc"` branch, keeps only `flag_steiner_terminal` points.

and three entries in `protodunevd/pr.jsonnet`'s `bee_points_sets`, all bound to
`visitor: 'TaggerCheckSTM:pr'` so they capture the grouping as the tagger saw it
(before `protect_bundle` can split a cluster out from under its flag), and all
wrapped in `if std.member(pipeline_names, 'tagger_check_stm')`.

Event 298595:

| layer | points | clusters | cluster ids |
|---|---:|---:|---|
| `clustering` | 196 745 | 121 | all |
| `stm` | 5 660 | 9 | 39, 40, 55, 86, 87, 97, 100, 109, 111 |
| `steiner_graph` | 5 569 | 9 | same 9 |
| `steiner_terminals` | 1 282 | 9 | same 9 |
| `stm_fit` | 8 869 | **25** | every cluster STM *evaluated* |

Invariants checked: the 9 ids are exactly the 9 `STM=1` log lines;
`steiner_terminals` (1 282) equals the count of `real_cluster_id == 1` inside
`steiner_graph` and every one of its points carries the terminal flag.

The `stm_fit`/`stm` contrast is the useful one and is deliberate (owner
decision): `persist_stm_fit` is called for **every** evaluated main
(`TaggerCheckSTM.cxx:614`, unconditional on the verdict), so `stm_fit` shows 25
candidates and `stm` tells you which 9 were actually tagged.

**Bee set for event 298595 (cosmic-only chain), uploaded at the owner's request:**
https://www.phy.bnl.gov/twister/bee/set/6d8cb2c4-abbc-4f3d-97fc-e430583344e3/event/list/

### 4.3 A crash landmine, removed

The first arm died with SIGSEGV at `MultiAlgBlobClustering.cxx:2923`.
`Dataset::get()` returns a null pointer for an array the cloud does not carry
and the existing code dereferenced it immediately. The steiner cloud uses the
**default-scope array names** (`x_t0cor,y,z`), not plain `x,y,z` — the uBooNE
steiner set spells it correctly (`clus/test/uboone-mabc.jsonnet:387`), my first
config did not. Fixed on both sides: the config uses `t0cor_coords`, and the C++
now names the missing array in a WARN and skips the cluster instead of crashing.

## 5. The gate that FAILED: Steiner only for STM candidates

### 5.1 The half that is impossible

**Fully-contained cannot be excluded from the Steiner build.**
`cluster_fc_check` requires a non-empty `steiner_pc` and returns the
conservative `is_fc=false` without one (`Clustering_Util.cxx:85-90`). FC-ness is
*computed from* the Steiner boundary, so "skip FC clusters" is circular. Only
the TGM half is even expressible.

### 5.2 The knob, and the ordering it needs

`CreateSteinerGraph` gained `skip_flags` (list of strings, default empty),
applied in the cluster filter with an INFO line reporting the saving. Because
`steiner` runs *third* in production — before every tagger — nothing is flagged
yet at that point, so using it requires moving `tagger_check_tgm` ahead of
`steiner`. That is runner mode `-stmlean`. `steiner_refresh` carries the same
list: it runs `replace=false`, i.e. it builds exactly the clusters with no graph
yet, so without it the refresh rebuilds everything the first pass skipped.

### 5.3 Three arms factorize the change exactly

20 PDVD events, same pinned binary, verdict **sets** compared per tagger — not
counts, which hide swaps.

| comparison | isolates | TGM | STM | FC |
|---|---|---|---|---|
| B vs A | reorder **+** skip | 18/20 events differ, 48 clusters | 14/20, 25 | 2/20, 2 |
| C vs A | **reorder alone** | 18/20, **48** | 14/20, **25** | 0/20, 0 |
| B vs C | **skip alone** | **0/20, 0** | **0/20, 0** | 2/20, 2 |

The reorder and the skip separate cleanly:

- **`skip_flags` is verdict-neutral.** Holding the order fixed, TGM and STM are
  identical on all 20 events. Its only effect is FC on 2 events — and both moves
  are on TGM-tagged clusters, which is the predicted and physically right answer
  (a through-going muon is not fully contained).
- **The reorder is the whole problem.** Moving `tagger_check_tgm` ahead of
  `steiner` changes the TGM verdict on 18 of 20 events and pulls clusters out of
  STM into TGM on 14 of 20.

Mechanism, on event 298595 clusters 97 and 118: in the production order TGM
*rejects* them on its charge-support test —

```
check_tgm: cluster 97 CASE-A pair (0,1) rejected: no 30.0 cm-step charge path between the ends (302.8 cm chord)
check_tgm: cluster 118 CASE-B pair (0,5) rejected: rescued end, straight chord 173.0 cm has an unsupported run > 30.0 cm
```

— and in the reordered chain it accepts both (`TGM=true`). So TGM's
chord-support test depends on grouping-level state that `CreateSteinerGraph`
touches first in production. **Which** state, and how, is not established (§7):
`CreateSteinerGraph` calls `destroy_child` on every path, so its retiled
clusters do not survive, and the attribution here is to the ordering, not yet to
a named cache. **Not a relabeling artefact:**
`clustering-global` is byte-identical across all three arms
(`d13aede9…`), so the cluster ids being compared are the same objects.

### 5.4 Verdict

**Do not flip.** The reorder buys a 23 % wall saving (mean 39 s → 30 s over the
20 events; 33 of 121 clusters skipped on event 298595, both Steiner passes) and
costs a cosmic-tagger verdict change on nearly every event. For a chain whose
entire purpose right now is to validate those taggers, that is the wrong trade.

Both pieces ship **available and OFF**: `skip_flags` defaults to `[]`
(`steiner_skip_flags=[]` in the driver), and `-stmlean` is a runner mode nobody
gets by accident. Re-running the arm is one flag plus one TLA.

## 6. Gates

- **Binary, knobs off ⇒ byte-identical.** The pre-change library was rebuilt
  from `HEAD~1` of the four touched C++ files and snapshotted
  (`/home/xqian/tmp/d39/lib_base`, no `has no array` string), then 3 manifest
  events (039252_10, 039349_14, 039349_67) were re-run on it with `-stm`.
  `clustering-global` and `stm_fit-global` member hashes are identical to the
  new-binary arm on all 3, and `clustering-global` is identical across all four
  arms on event 298595 — 4 events total. The new fields are absent from those
  layers' configs, so this is the knob-off path.
- **Compiled config, knobs absent ⇒ byte-identical.** All 16 configs in
  `abtest/compile_all_cfg.sh` identical, including `sbnd_pr`; `uboone_mabc`
  identical (0 diff lines). `pdvd_pr` differs on **exactly** the three new
  `bee_points_sets` entries and nothing else — no `skip_flags` key appears, so
  the Steiner stage is unchanged. Checked on both the `-nu` and `-stm` pipelines.
- **`./build/clus/wcdoctest-clus`**: 295 cases / 22 628 assertions pass,
  including a new case pinning `skip_flags` empty.
- **Freshness (M1)**: `libWireCellClus.so` 2026-09-04 17:47, newer than every
  source edit; the new code verified present in the pinned copy before the arms.

## 7. Not established

- The gate ran on **20 events, one detector, data only**. The 18/20 TGM rate is
  a rate on that sample, not a bound. The knob-off binary equivalence (§6) rests
  on 4 events, not the full manifest.
- The *reason* TGM's chord-support test sees different charge before vs after
  `CreateSteinerGraph` is characterized empirically (which clusters, which
  rejection lines) but not root-caused to a specific cache or population step.
  That is the open question if anyone wants the Steiner saving back.
- `stm_fit` was deliberately **left unrestricted**, so it still shows rejected
  candidates. SUPERSEDED by §11-12.1: `stm_fit` still shows them, and the other
  three STM layers were widened to match it instead.
- Pre-existing, reported not fixed: the `steiner_pc` Bee branch hard-codes a
  4000 e threshold in `calc_charge_wcp` (`MultiAlgBlobClustering.cxx:2961`) — a
  uBooNE value applied to a PDVD dump. It affects only the `q` shading of the
  two new Steiner layers.

## 8. Next

1. **Hand-scan the 298595 set** (§4.2 link): are the 9 STM clusters the right 9,
   and do their Steiner terminals sit on the track skeleton?
2. Turn the PR tail back on with `-nu` when the cosmic stage is validated.
3. If the Steiner build cost matters later, §5.3's factorization says the knob
   is sound — what needs solving is giving TGM its production view of the charge
   without running `CreateSteinerGraph` first.

## 9. Round 2 — the owner's two questions on the Bee set

The owner reviewed the §4.2 set and raised two things. Both are real. One is a
scope mistake in §4.2 that I made; the other is a genuine reconstruction defect
that SBND already fixed and PDVD never did.

> 1. The Steiner Graph image seems to jump many gaps that are not in the
>    original 3D images. This seems to imply that the cluster are not separated
>    properly, it may contain the main cluster as well as the isolated clusters
>    done in the initial clustering step. In SBND, we have a step to separate to
>    individual clusters.
> 2. I do not quite understand why the stm-global's images are much smaller than
>    the stm-fit-global's result.

### 9.1 Repro block for everything below

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
# Pinned library for every round-2 arm:
#   /home/xqian/tmp/d39r2_libpin/libWireCellClus.so   2026-09-04 18:54
# toolkit HEAD at the time: 28cd60d8   wcp-porting-img: 6e7f6350

# The clus stage now records what clustering_isolated merged (M13: fresh tags).
PDVD_LIGHT_SUFFIX=_keep ./run_clus_evt.sh -save-pctree -save-assoc -s d39r2prov 39252 2
# ... and the CONTROL, same binary, same day, knob off (sec 13.1):
PDVD_LIGHT_SUFFIX=_keep ./run_clus_evt.sh -save-pctree             -s d39r2ctl  39252 2

# Two PR passes over the SAME pctree; only unmerge_assoc differs.
./run_pr_evt.sh            -s d39r2base 39252 2
./run_pr_evt.sh -unmerge   -s d39r2unm  39252 2

# The 21-event gate (10-way parallel), then the census:
#   scratch driver: d39r2_arms.sh -- one clus pass + two PR passes per event
docs/nf_sp_img_clus/scripts/d39_unmerge_census.py \
    work/039252_2_d39r2base work/039252_2_d39r2unm
# full record: stm/gates/d39r2_unmerge_gate.txt
```

## 10. "The Steiner graph jumps gaps" — confirmed, and it is `clustering_isolated`

### 10.1 The gaps are real and the points in them are fabricated

For each Steiner point, the distance to the nearest **live 3D point anywhere in
the 121-cluster event**. A point with none within 3 cm was not reconstructed from
charge — it was manufactured.

| STM cluster (evt 298595) | 3D pts | Steiner pts | Steiner pts with no 3D within 3 cm |
|---|---:|---:|---:|
| 100 | 18 | 51 | **39 (76.5 %)** |
| 87 | 152 | 175 | 52 (29.7 %) |
| 109 | 1049 | 1279 | 268 (21.0 %) |
| 111 | 61 | 49 | 10 (20.4 %) |
| 97 | 1927 | 2016 | 7 (0.3 %) |
| 40 | 1201 | 1034 | 4 (0.4 %) |
| 39 / 55 / 86 | 311 / 343 / 598 | 283 / 254 / 428 | 0 |

Cluster 100's 39 fabricated points interpolate a straight line from
`(91.6, −334.1, 234.6)` to `(72.1, −333.8, 261.1)` — a 41 cm bridge between its
only two real pieces.

They are **not borrowed from a neighbouring cluster**: every one of them is >3 cm
from every point of every one of the 121 clusters, and no Steiner cloud extends
more than **1.8 cm** beyond its own cluster's 3D bounding box on any axis (the
boxes shrink in places — cluster 87's Steiner x-extent is 31.4 cm against the
cluster's 59.2 — but they never reach outside). The fabricated points sit
*inside* the object, in its gaps.

### 10.2 Why: the retiler synthesises wire activity along a whole-cluster path

`CreateSteinerGraph` merges nothing — one cluster in, one out, and the scratch
child is destroyed on every path (`CreateSteinerGraph.cxx:271-275, :354-356,
:374`). The fabrication is in the **retiler**, which on PDVD is
`ImproveCluster_2` (`pr.jsonnet:1246`, passed at `:1316`/`:1336`):

- `improvecluster_2.cxx:124-128` takes a **whole-cluster** shortest path between
  the two boundary points;
- `:209, :214` call `hack_activity_improved` (`improvecluster_1.cxx:451`), which
  interpolates that path at 0.3 cm and writes `hit` into all three planes over
  ±3 time slices wherever real activity is missing. Tiling then produces blobs in
  charge-free space, and those become Steiner nodes and terminals.
- The path crosses the gaps because the graph flavors that carry it
  (`basic_pid`, `ctpc_ref_pid`) bridge disconnected components with an
  **uncapped** MST — `connect_graph.cxx:20-26, :91`; a >5 cm bridge only changes
  the weight branch (`:161-166`), never whether the edge is added.

### 10.3 And the reason a path has gaps to cross: `clustering_isolated`

The clusters really do contain a body plus detached clumps. Minimum distance from
each satellite to the main body, evt 298595:

| cluster | main | satellites (points @ gap) |
|---|---:|---|
| 109 | 938 | 19@76 cm, 11@72, 14@62, 19@61, 10@54, 12@46, 12@38, 5@38, 9@16 |
| 87 | 102 | 27@81 cm, 3@66, 15@55, 5@19 |
| 100 | 13 | 5@41 cm |
| 97 | 1809 | 7@69 cm, 3@13, 108@6 |

48 of the 54 clusters with ≥50 points are multi-component at a 6 cm link.

The membership is set in the **clustering** stage, not in PR: the same 1049
points are one cluster already in `mabc-group4567.zip` (id 387) and in
`mabc-all-apa.zip` (id 124).

The site is `ClusteringIsolated` — `clustering_isolated.cxx:601`,
`merge_clusters(g, live_grouping, "isolated")`. It **physically merges** a main
with the small clusters near but *not connected to* it; the prototype only
*groups* them. PDVD called it bare (`protodunevd/clus.jsonnet:552`,
`cm.isolated(),`). SBND calls it with provenance and undoes it in PR, and says
why in its own config (`sbnd/clus.jsonnet:2011-2018`):

> *"the toolkit physically merges them, so the STM/PR endpoint finder walks into
> a detached clump across empty space (docs 50, 51)"*

**So the owner's recollection was right, and so was the inference.** PDVD does run
`cm.separate(...)` (`clus.jsonnet:534`, with all the doc 25/26 refinements), but
that is a different question; the stage SBND has and PDVD lacked is
`unmerge_assoc`.

### 10.4 Why nothing downstream rescues it

PDVD's only PR-stage splitter is `protect_bundle`
(`ClusteringProtectBundle` = uBooNE `Protect_Over_Clustering`). It runs *after*
the taggers, and `skip_convicted` makes it refuse exactly the clusters that
carry a verdict:

```
OC53SKIP member ident=109 nblobs=496 main=1 convicted TGM=0 STM=1 lm=-1 -- never split
split 0 bundle cluster(s) into 0 extra cluster(s) (0 cathode re-join(s),
   14 convicted main(s) skipped, 78 main(s) refused for holding no STM tag, ...)
```

It would not have helped anyway: it splits a *bundle* into its member clusters,
and these fragments live inside one cluster ident.

### 10.5 The disclosure that matters (reported, not tuned — CLAUDE.md §5.7)

**Cluster 100 was tagged STM=1 on 18 real 3D points in two fragments 41 cm apart,
with 39 of its 51 Steiner points fabricated.** Cluster 87 (152 points in five
fragments over 90 cm, 30 % fabricated) is the same shape. This is the defect the
companion doc `39_stm-fit-residual-and-the-second-stage.md` named as the sole
residual — *"cluster splitting is the only lead that moves both axes"* — found
independently by the owner in the display.

## 11. "stm is much smaller than stm_fit" — a scope mismatch in §4.2

Not a bug in the reconstruction; a wrong gate on the layer I added.

`TaggerCheckSTM::persist_stm_fit` writes the `stm_fit` local PC for **every**
evaluated main that recorded a fit pass — the call at `TaggerCheckSTM.cxx:614`
is unconditional on `is_stm`. `stm` carried `require_flag: 'STM'`, i.e. the
verdict. The funnel on evt 298595 (§4.2 arm):

| stage | count | source |
|---|---:|---|
| clusters in the live grouping | 121 | `clustering-global` |
| flash-matched mains | 92 | `TaggerCheckTGM` verdict lines |
| skipped as already TGM | 31 | `already TGM; skipping` |
| evaluated by STM | 61 | `STM=0/1` lines |
| **wrote an `stm_fit` PC** | **25** | `save_stm_fit stored 29 segment(s)` |
| **tagged STM** | **9** | `set_flag(Flags::STM)` |

Two corrections worth stating plainly:

- **`stm` is not sparser.** On the nine ids the two layers share, `stm` carries
  5660 points and `stm_fit` 2687 — `stm` is 2.1× *denser*. The whole visual
  difference is the 16 fitted-but-untagged clusters (6182 `stm_fit` points).
- The layers draw **different geometry**: `stm_fit` is a 1-D fitted polyline
  sampled along `rec.segment->fits()`; `stm` is the per-blob 3D cloud. Even on
  identical cluster sets they would not be the same picture.

## 12. What shipped in round 2

### 12.1 `require_pc` — the layers now cover one object set

New `BeePointsConfig` field `require_pc` (default `""`), ANDed with
`require_flag` in the existing per-cluster gate
(`MultiAlgBlobClustering.cxx:857-872`): admit a cluster only if it carries a
non-empty local PC of that name. No new flag was needed in `TaggerCheckSTM` —
the `stm_fit` PC **is** the "this cluster was fitted" marker.

PDVD (`protodunevd/pr.jsonnet`, PDVD only) now declares four STM layers:

| layer | gate | evt 298595 |
|---|---|---|
| `stm` (3D image) | `require_pc: 'stm_fit'` | 23 objects |
| `steiner_graph` | `require_pc: 'stm_fit'` | 23 |
| `steiner_terminals` | `require_pc: 'stm_fit'` | 23 |
| `stm_fit` (unchanged) | — (culled by the missing PC) | 23 |
| **`stm_tagged`** (the verdict) | `require_flag: 'STM'` | 9 |

The census asserts the first four sets are identical in every arm; they were, on
all 21 gate events, in both the base and the un-merged arm.

**Read `stm_tagged` first.** `stm` is now the candidate population — 58 893 points
on evt 298595, about 30 % of the whole `clustering` layer — so it is heavy enough
to want hiding while scanning.

### 12.2 The PDVD un-merge chain (config only — no new C++)

Every piece already existed; PDVD simply never wired them.

| where | change | default |
|---|---|---|
| `protodunevd/clus.jsonnet:552` | `cm.isolated(save_assoc_id=save_assoc_id)` | off |
| same, both group MABCs + `clus_all_tpc` | `[if save_assoc_id then 'save_assoc_cluster_id']: true` | off |
| `protodunevd/pr.jsonnet` `cm_by_name` | `unmerge_assoc: cm.unmerge_bundle(name='assoc', mode='real', id_aname='assoc_cluster_id', main_aname='assoc_cluster_main')` | not in `pipeline_names` |
| `pdvd/wct-clustering.jsonnet` | TLA `clus_save_assoc_id` | false |
| `pdvd/run_clus_evt.sh` | `-save-assoc` | off |
| `pdvd/run_pr_evt.sh` | `-unmerge` → `PIPE_STM_UNMERGE`, + a provenance guard | mode stays `-stm` |

**The guard matters more than it looks.** `ClusteringUnmergeBundle`'s stance is
"no usable provenance ⇒ skip, never guess" (it must not fall back to splitting on
graph connectivity — that would break cathode crossers), so `-unmerge` on a
pctree written without `-save-assoc` finishes **rc=0 with a normal-looking Bee
set and nothing split**: a vacuous arm that reads like a real one. Its
`require_provenance` knob does *not* cover this — it guards only the `wasmain`
array under `restore_demoted_mains` (`ClusteringUnmergeBundle.cxx:426`). So
`run_pr_evt.sh` now probes the pctree's metadata for a `perblob` datapath
(~0.1 s on a 20 MB tree) and refuses with rc=4; `PDVD_ALLOW_NO_ASSOC=1`
overrides. Verified with a causal negative control — the guard refuses the
`d39r2ctl` tree, passes the `d39r2prov` tree, and leaves `-stm` untouched on
both. Its first form was **broken**: `grep -qm1` exits on the first match,
SIGPIPEs the `tar`, and under the runner's `set -o pipefail` the pipeline then
reports failure *on a match* — so the guard rejected every tree, provenance or
not. Only the positive control caught it.

Pipeline position, per the owner's decision:

```
switch_scope, flag_mains, unmerge_assoc, steiner, fiducialutils,
tagger_check_tgm, tagger_check_stm, tagger_check_fc,
protect_bundle, steiner_refresh, pr_display
```

*Before* `steiner` because `separate()` does not carry node-local PCs, so the
split must precede `steiner_pc` creation. *After* `flag_mains` so the split-off
fragments are removed from the main's Steiner build and fit **without** being
promoted to mains and given cosmic verdicts of their own. That worked exactly as
intended: **1277 mains evaluated in both arms across the manifest, identical.**

The attribution is confirmed, not assumed — the visitor's own accounting on
evt 298595 reproduces the satellites measured in §10.3 before any of this was
built:

```
cluster 109: 496 blobs -> main 434 + 11 associated cluster(s) holding 62 (real mode)
cluster  87:  77 blobs -> main  69 +  2 associated cluster(s) holding  8
cluster 100:  14 blobs -> main   9 +  1 associated cluster(s) holding  5
```

## 13. The round-2 gates

### 13.1 The provenance knobs change nothing but provenance

Two clus runs, same binary, same day, same event, fresh tags `d39r2ctl`
(off) and `d39r2prov` (on):

- `mabc-all-apa.zip`, `mabc-group0123.zip`, `mabc-group4567.zip` — **every member
  content hash identical** (`abtest/hash_archive.py --members`, never `cmp`, M2).
- the pctree gains **exactly** 1465 → 1474 members, and the added datapaths are
  exactly `live/pointclouds/namedpcs/perblob` + its three arrays
  `isolated`, `assoc_cluster_id`, `assoc_cluster_main`, plus the `lpcmaps` entry.

### 13.2 Compiled-config, knobs absent

`abtest/compile_all_cfg.sh` before vs after, pre-change tree reconstructed from
`git show HEAD:` — **16/16 PASS**, 0 normalized diff lines, same element counts,
same component order, same Pgrapher edges.

`pdvd_pr` compiled by hand (compile-only, both arms): the whole diff is the four
Bee layer entries — three `require_flag: 'STM'` → `require_pc: 'stm_fit'` and the
new `stm_tagged` block. **No reconstruction component changed.** Compiled-config
proof with the knobs ON: `save_assoc_cluster_id` appears on all three MABC nodes,
`save_assoc_id=true` on both `ClusteringIsolated`, and
`ClusteringUnmergeBundle:prassoc mode=real id_aname=assoc_cluster_id` in the
`-unmerge` config.

`./build/clus/wcdoctest-clus`: **296/296 pass** (the new case pins
`require_flag` / `require_pc` / `steiner_terminals_only` all admitting
everything).

### 13.3 The A/B: 21 events, one clus pass each, two PR passes over it

Full record: `stm/gates/d39r2_unmerge_gate.txt`.

Cluster ids are **not** comparable across these arms — the un-merge splits one
cluster into several — so the census matches objects **geometrically** through
the `clustering` layer and scores each base cluster against the arm cluster that
inherited the largest share of its points. An id-keyed diff here would be
meaningless.

| metric | base (`-stm`) | arm (`-unmerge`) |
|---|---:|---:|
| clusters in the live grouping | 61–161 per event | 180–832 |
| mains evaluated | **1277** | **1277** (identical) |
| TGM tagged | 421 | **463** |
| STM tagged | 136 | **113** |
| FC tagged | 407 | **434** |
| **mains carrying any cosmic tag** | 964 (75.5 %) | **1010 (79.1 %)** |
| STM candidates that are multi-component | 351/392 = **90 %** | 89/329 = **27 %** |
| Steiner points with no 3D support | 24 691/488 262 = **5.06 %** | 7337/456 497 = **1.61 %** |
| worst cluster's fabricated fraction | 19–85 % per event | 0–47 % |
| mean wall | 25.1 s | **20.0 s (−20 %)** |

Where the STM tags went: of the **54** base STM tags whose heir is not STM,
**28 are TGM or FC in the arm** — still cosmic, reclassified — and **26 carry no
cosmic tag at all**. There are also **28** `cand -> STM` gains. On evt 298595
specifically, clusters 39, 40 and 100 became TGM and 111 became FC; only 113 lost
its cosmic verdict.

### 13.4 Verdict

The knob **ships OFF**. It is not a byte-identical change and it is not meant to
be — it moves cosmic verdicts by construction, which is the point. What the gate
establishes is that the movement is in the right direction on every aggregate
that can be measured without a hand scan: the same mains evaluated, more of them
tagged, far fewer fabricated points, 20 % less wall. What it cannot establish is
the 26 objects that lost their cosmic tag. **That is the owner's call and it
wants eyes, not another aggregate.**

Bee, event 298595, same pctree, same binary, only `unmerge_assoc` differing:

- **before** — https://www.phy.bnl.gov/twister/bee/set/ddc5da45-b1a5-4cb9-9876-de32dc48c332/event/list/
- **after**  — https://www.phy.bnl.gov/twister/bee/set/01f28648-c628-4f70-89bd-031fcccd865c/event/list/

Both verified live (HTTP 200, event 0, all six layers present).

## 14. Round 2 — not established

- **The 26 objects that lost their cosmic tag are not explained.** They are the
  reason the knob is off.
- The gate is **21 events, PDVD data only**. Every rate above is a rate on that
  sample.
- **Un-merging does not cap the bridge length.** `ImproveCluster_2` will still
  fabricate blobs across a gap *inside* one cluster, because the component
  bridges in `basic_pid` / `ctpc_ref_pid` are uncapped
  (`connect_graph.cxx:20-26`). The residual shows in the arm: 27 % of STM
  candidates are still multi-component, and evt 298595 cluster 84 has 17
  components. Those come from the long-range merge family (docs 25/26), not from
  `clustering_isolated`. A distance cap on the MST bridge is the obvious next
  knob and was not attempted here.
- `ClusteringUnmergeBundle` warns once per cluster that has **no** `perblob` PC
  at all (33 of 92 mains on evt 298595) — clusters `clustering_isolated` never
  touched. Cosmetic; the homogenization loop at
  `MultiAlgBlobClustering.cxx:3898` only reaches clusters that already have the
  PC. Not fixed here.
- The round-2 arms use a **fresh clus pass**, so they are not directly comparable
  to the §4.2 arm, which read the `d27fresh` pctree from an older cfg epoch
  (23 fitted / 9 tagged here vs 25 / 9 there). Base-vs-arm comparisons are all
  within one epoch.
- Still unfixed from round 1: the `steiner_pc` Bee branch hard-codes a 4000 e
  uBooNE threshold in `calc_charge_wcp`. It affects only the `q` shading of the
  two Steiner layers.

## 15. Related

- **doc 40 — the follow-up.** The owner scanned the round-2 Bee set and named
  two Steiner points with no image behind them. One is `ctpc_aniso_metric`
  (the owner's 2026-09-04 flip, not this round); the other is the uncapped
  component bridge §14 item 1 leaves open, now measured on 21 events.
- doc 25 §2.1 — PDVD's readout-wide beam window
- doc 30 — the same event 298595, `stm_fit` vs `track_fit` (why they disagree)
- doc 37 — Steiner terminals, the 0.5 cm thinning now in production
- doc 38 — the gap-aware end trim, also in this binary

## 16. `unmerge_assoc` is PDVD PRODUCTION (owner decision, 2026-09-04)

§12 shipped `unmerge_assoc` wired but **held out of the default** pending the
§13 A/B adjudication. The owner adjudicated it the same day and flipped it on,
together with the retirement of doc 38's gap-aware end trim (doc 38 §10).

**Repro**

```bash
cd pdvd
./scripts/stage_pr_tag.sh 39252 2 d38qnewprod d39r2prov
./run_pr_evt.sh -s d38qnewprod 39252 2          # no TLAs at all
python3 ../../wcp-porting-img/abtest/hash_archive.py work/039252_2_d38qnewprod/mabc-pr.zip
# -> e728abfe697d4cb309c9d7778dca3ea2a8155a26104fdb8ab63422adcab086a2
```

### 16.1 What changed

| file | change |
|---|---|
| `pdvd/run_pr_evt.sh` | `unmerge_assoc` added to **both** `PIPE_STM` and `PIPE_NU`, so `-nu` does not silently lose the new default. `PIPE_STM_MERGED` / `PIPE_NU_MERGED` keep the pre-flip chains verbatim, reachable as `-nounmerge` / `-nounmerge-nu`. `PIPE_STM_UNMERGE` is now an alias of `PIPE_STM`, so existing `-unmerge` commands keep working. |
| `pdvd/run_pr_evt.sh` | the `-save-assoc` provenance guard is re-keyed on **the selected pipeline containing `unmerge_assoc`**, not on `MODE = unmerge`. Left on `MODE` it would have stopped guarding the moment the default changed, and §12.2's silently-inert arm would be back. |
| `pdvd/run_clus_evt.sh` | `SAVE_ASSOC` default **0 → 1**, with `-no-save-assoc` to restore. Required: the default PR chain now hard-errors rc=4 on a pctree with no `perblob` provenance, so a default-off clustering stage would make the default PR stage refuse its own default input. |
| `cfg/…/protodunevd/pdvd_track_fitting.json` | `end_trim_gap_len` 200 → 0 (doc 38 §10). |

The clus-stage pctree is therefore **no longer byte-identical** to a pre-flip
one — it gains the three `perblob` provenance arrays. Cluster membership and
every physics quantity are unchanged; the arrays are additive.

### 16.2 The gates

Both on 039252/2 (evt 298595), same binary (`local/lib/libWireCellClus.so`
2026-09-04 18:54, newest `clus` source 18:47), same pctree (`d39r2prov`).

| gate | arms | result |
|---|---|---|
| the flip reproduces the measured configuration | `d38qnewprod` (**no TLAs**) vs `d38qunoff` (unmerge + trim 0 via TLA) | **PASS**, `e728abfe…` both |
| the escape hatch reproduces the pre-flip chain | `d38qesc` (`-nounmerge`) vs `d38qoff` (merged + trim 0 via TLA) | **PASS**, `2079fd78…` both |
| noise floor | `d38qunrep` vs `d38qunon`; `d38qrep` vs `d38qon` | **0** — identical, no `setarch -R` |

The first gate is the load-bearing one: a run with **no TLAs at all** landing on
the hash that the explicitly-configured arm produced proves both edits landed,
that the runner picks up `unmerge_assoc` by default, and that the cfg picks up
`end_trim_gap_len: 0` — in one check.

### 16.3 What `unmerge_assoc` does, on this event

`d38qon` → `d38qunon` (trim 20 cm on both sides, so this isolates the unmerge):

| | merged | unmerged |
|---|---|---|
| clusters | 121 | **522** |
| total 3-D points | 196745 | 196745 |
| STM-evaluated | 23 | **15** |
| STM-tagged | 9 | **5** |
| `stm_fit` points | 5807 | 4051 |

Tag set `[39,40,55,86,87,100,109,111,113]` → `[55,83,86,87,109]`.

**The sign of that is unknown and this doc does not claim one.** Fewer tags may
be correct — §12's stated purpose is to let the taggers see individual objects
instead of a main body plus detached clumps, and a clump that was dragging a
main into an STM verdict *should* stop doing so. It may equally be lost
efficiency. It needs a hand scan, on more than one event.

The 33 `no flash-merge provenance … not split` warnings on this event are the
**known cosmetic condition** §14 already records (clusters `clustering_isolated`
never touched, so there is nothing to undo) — not a provenance gap and not a
lost split. Count matches §14 exactly.

### 16.4 Not established by this flip

- **n = 1.** Both the flip's gates and the §16.3 census are one event. The
  120-event manifest is the general claim and was not run.
- **The `-nu` chain: GRADED, and the trim retirement is neutral there too.**
  Arms on 039252/2, chain held at the new default (`unmerge_assoc` ON), varying
  only `end_trim_gap_len`:

  | arm | `end_trim_gap_len` | `hash_archive.py` |
  |---|---|---|
  | `d38qnusmoke` | 0 (production) | `17147f16…` |
  | `d38qnurep` | 0 (repeat) | `17147f16…` |
  | `d38qnutrim` | 200 (pre-flip) | `17147f16…` |

  Noise floor **0** — and that repeat arm is mandatory here, not optional: the
  `-nu` chain runs `tagger_check_neutrino` with the DL/SCN vertex, which is not
  bit-stable in general (M4). It was on this event.

  So the trim is a no-op on the **neutrino** chain as well, `track_fit` layer
  included. This **retires the §16.4 concern as written and doc 38 §9's**: that
  section measured `end_trim_gap_len` moving `track_fit` by +174 clusters /
  +13.0 % points, but that was on the **merged** chain. Under `unmerge_assoc`
  the effect is exactly zero, on both consumers of `pdvd_track_fitting.json`.
  The compiled `-nu` config binds that file twice — `TaggerCheckSTM:pr` and
  `TaggerCheckNeutrino:pr` — so both were exercised.

  Liveness, for the record: rc=0, wall **41 s**, peak RSS **2.67 GB**, zero
  error/critical log lines, full layer set (`track_fit`, `shower_track`,
  `vertices` alongside the STM layers). It does not hit doc 25 §13.11's
  `ProtectBundle` cost even at 522 clusters, because the per-bundle PR and
  `ProtectBundle` are both gated on STM-tagged bundles and that set went 9 → 5.

- **The §13 A/B that §12 was waiting on** was not what adjudicated this; the
  owner's decision was. The measurement here is narrower.

### 16.5 Bee

- old production (merged + trim 20 cm) vs trim OFF:
  https://www.phy.bnl.gov/twister/bee/set/49f303b0-ddb5-4787-b897-928da86cf355/event/list/
  (idx 0 = trim 20 cm, idx 1 = trim 0)
- **new production vs old production**:
  https://www.phy.bnl.gov/twister/bee/set/52e201f4-68b0-47ab-a9a5-6df0b394ea01/event/list/
  (**idx 0 = new production** — `unmerge_assoc` ON, trim OFF; **idx 1 = old
  production** — merged, trim 20 cm)

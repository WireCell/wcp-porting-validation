# PDHD doc 06 — `unmerge_assoc` on PDHD: it was never wired, and the reason on file was wrong

**Owner, 2026-09-06:** *"For the PDHD chain, I wonder if you have the unmerge implemented already?
… if not, it should be added, like what we did for PDVD, and then regenerate the bee link for me."*

Short answer: **no, it was not**, and the comment in `run_pr_evt.sh` explaining why was **factually
wrong**. It is now added, default-OFF and gated, with runner flags on both jobs. What it does on
PDHD is substantial — 40–46 main clusters split per event — but it does **not** recover the two
stopping muons of [doc 04](04_stm-tagger-scan.md) §12 item 0.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd

# clustering must record the provenance; PR consumes it
./run_clus_evt.sh -q -save-pctree -save-assoc -s <tag> 029107 <evt>
./run_pr_evt.sh   -stm-fit -unmerge          -s <tag> 029107 <evt>

# the arms this doc is measured on (binary pinned, libWireCellClus md5 495ed07e…,
# toolkit ef995685 + this change; events 1 and 12 of run 029107)
#   d06um    -save-assoc clustering + -unmerge PR
#   d06base  the SAME pctree, PR without -unmerge   (the control)

python3 docs/scripts/d05_arms_bee.py --dirs work/029107_{1,12}_d06um \
        --out bee-pr-run029107-d06um --highlight 1:113,12:108
```

## 1. The answer, in three parts

**The C++ exists and is detector-agnostic.** `ClusteringUnmergeBundle` (`clus/src/`) is the same
component SBND and PDVD use.

**The PDHD config already carried the visitor definition** — `cfg/pgrapher/experiment/pdhd/pr.jsonnet`
defines `unmerge_assoc` — because that block was forked from PDVD together with its comments. It
was in no PDHD pipeline, so it never ran.

**And it could not have worked if it had been named**, because the prerequisite was missing: the
PDHD clustering job wrote no `assoc_cluster_id` / `assoc_cluster_main` perblob arrays, so the
visitor had nothing to undo and would have been **silently inert**.

## 2. The reason on file was wrong, and the wrong reason hid a real defect

`run_pr_evt.sh` said:

> *"unmerge_assoc is DELIBERATELY absent: PDHD's clustering runs no `cm.isolated()` merge (the
> defect doc pdvd/39 round 2 undoes on PDVD), so there is no isolated grouping to split and the
> stage would be inert."*

The first clause is false. From the **compiled** PDHD clustering config, not the jsonnet source:

| | |
|---|---|
| `ClusteringIsolated` instances | **2** — `group02`, `group13` |
| where | the per-drift-group MABC pipelines (`clus_per_group-group02/13`), 12 visitors each |
| `save_assoc_cluster_id` keys | **0** |

And `cm.isolated()` **merges**. `clustering_isolated.cxx` is explicit that `save_assoc_id` gates
only the *recording*: *"Default false => `merge_clusters()` is called exactly as before and no
[provenance]"*. So PDHD has been physically absorbing detached clumps into main clusters all along
— the same defect doc pdvd/39 round 2 found on PDVD, where it made the STM endpoint finder and the
Steiner build walk across empty space into a clump that is not part of the track.

The conclusion "the stage would be inert" was right; the reason was not. The distinction is not
pedantic: the wrong reason says *there is nothing to undo*, and there is quite a lot to undo (§5).

## 3. What was added

Mirrors PDVD exactly (`protodunevd/clus.jsonnet` `save_assoc_id`, doc pdvd/39 r2).

| file | change |
|---|---|
| `cfg/pgrapher/experiment/pdhd/clus.jsonnet` | `save_assoc_id` arg on `clus_per_group` and `clus_all_tpc` (+ both public wrappers); threaded into `cm.isolated(save_assoc_id=…)` and into the `save_assoc_cluster_id` key on all three MABCs |
| `pdhd/wct-clustering.jsonnet` | `clus_save_assoc_id = false` TLA, passed to `per_group` and both `all_tpc` branches |
| `pdhd/run_clus_evt.sh` | `-save-assoc` (env `PDHD_SAVE_ASSOC`); records `save_assoc_id=` in the pctree `.tlas` sidecar, read out of the **compiled** config |
| `pdhd/run_pr_evt.sh` | `-unmerge` / `-nounmerge`; inserts `unmerge_assoc` after `flag_mains`, before `steiner`; **plus the guard below**; and the wrong comment replaced by what is true |

**Ordering** is PDVD's (doc pdvd/39 §11): after `flag_mains` so the split-off fragments are removed
from the main's Steiner build and STM fit *without* being promoted to mains and given cosmic
verdicts of their own; before `steiner` because `separate()` does not carry node-local PCs, so the
split must precede `steiner_pc` creation.

**The guard.** On a pctree written without `-save-assoc`, `unmerge_assoc` finds no provenance and
is silently inert — the run looks normal and is not. That silence is the exact failure this whole
doc is about, so `run_pr_evt.sh` now refuses:

```
ERROR: … the pipeline contains unmerge_assoc but this pctree carries no isolated-merge
       provenance (save_assoc_id=absent).
       Re-run clustering with: ./run_clus_evt.sh -q -save-pctree -save-assoc -s <tag> 029107 1
```

Verified: rc **4**, nothing written. It is keyed on the **selected pipeline**, not on the
`-unmerge` flag — a guard keyed on the flag that used to select a behaviour goes blind the moment
the behaviour becomes a default (`project_doc38_39_unmerge_flip`).

## 4. Gates

| gate | result |
|---|---|
| compiled clustering config, knob **off**, vs before the change | **byte-identical** (md5 `0806645a169b2768f9c0d4130c2abb11`) |
| compiled clustering config, knob **on** | 3 × `save_assoc_cluster_id`, 2 × `save_assoc_id` — the compiled-config proof |
| clustering Bee `mabc-all-apa.zip`, `-save-assoc` vs `d05mON` | **identical member content** both events (`1d762209…`, `f5f864cb…`) |
| **control**: same pctree, PR *without* `-unmerge`, vs `d05mON` | **identical member content** both events (`ce10012d…`, `504efac5…`) |
| guard: `-unmerge` on a provenance-less pctree | rc 4, refused |

The control is the one that matters: writing the three provenance arrays changes **nothing** in the
PR output, so everything in §5 is `unmerge_assoc`'s doing and not a side effect of recording. The
pctree file itself is *not* byte-identical — it gains the arrays — which is why `-save-assoc` is a
flag and not a default.

## 5. What it does on PDHD

Events 1 and 12 of run 029107, one binary, the same pctree in both arms:

```
ClusteringUnmergeBundle:prassoc  unmerged 40 main cluster(s) into 323 associated cluster(s)
                                 (40 exact/provenance, 0 proxy/component)     [evt 1]
                                 46 -> 243   (46 exact, 0 proxy)              [evt 12]
```

Every split came from **exact provenance**, none from the component-mode fallback. The 38 / 42
`no flash-merge provenance … not split` warnings are the cosmetic ones (clusters
`clustering_isolated` never touched, so there is nothing to undo) — doc pdvd/39 §14.

| | evt 1 base → um | evt 12 base → um |
|---|---|---|
| clusters in the PR Bee | 86 → **418** | 126 → **360** |
| 3-D points (unchanged: membership is redistributed, nothing lost) | 153 706 | 161 854 |
| TGM evaluated | 82 → 82 | 92 → 92 |
| TGM = true | 35 → **36** | 23 → **24** |
| STM evaluated | 47 → 46 | 69 → 68 |
| **STM tagged** | 4 → **4** | 8 → **8** |
| STM fitted | 20 → 16 | 26 → 22 |

TGM-evaluated is unchanged because the split-off fragments are **not** promoted to mains — that is
what the `flag_mains`-before-`unmerge_assoc` ordering buys. So the defect is real and large in
cluster count (40–46 mains per event were carrying detached clumps) while the tagger verdicts move
little on these two events: +1 TGM each, STM tags unchanged, 8 fewer STM fits.

**Sign unknown — this is not graded.** Two events is not a measurement, and doc pdvd/39 made the
same point about PDVD: fewer fits may be correct (a clump dragging a main into a fit should stop
doing so) or may be a loss. Do not flip the default on this.

## 6. It does **not** recover the two lost stoppers

That was the hypothesis worth testing — both are multi-component clusters, which is the signature
of an isolated-merge. Unmerge cleans them up and the verdict does not move.

| | evt 1 cluster 113 | evt 12 cluster 108 |
|---|---|---|
| components, base → um | **8 → 1** | **6 → 2** |
| points in the main | 1 955 → 1 866 | 6 714 → 6 670 |
| STM | 0 → 0 | 0 → 0 |
| fit | status 7, kink 453 → status 7, kink 452 | status 2 → **no STM fit at all** |
| FC | false → false | false → **true** |

**Cluster 113 is now a single component and is still rejected by the same accept guard** (status 7,
`accept_guards_reject`), with a fit that did not move (kink 453 → 452). So the merge was not what
was wrong with it: the desert/spike guard is, and doc 04 §12 item 0 stands unchanged for this one.

**Cluster 108** loses its STM fit entirely and is now called fully contained. Its kink relocation
(doc 05 §8.1) is not explained by the merge either.

Reading these two out is what the Bee links below are for.

## 7. Bee links

| | link |
|---|---|
| **without** unmerge (`d06base`; identical to `d05mON` by the §4 control) | https://www.phy.bnl.gov/twister/bee/set/e2793916-8704-4a70-b41e-27c17509990d/event/list/ |
| **with** unmerge (`d06um`) | https://www.phy.bnl.gov/twister/bee/set/93edc720-18d1-4cd2-9dfb-11fb985c2dc1/event/list/ |

Slot 0 = event 1 (cluster 113), slot 1 = event 12 (cluster 108), both sets. All layers —
`clustering`, `stm`, `stm_fit`, `stm_tagged`, `steiner_graph`, `steiner_terminals`,
`channel-deadarea-*` — plus a `scan` layer holding just the highlighted cluster. Verified from the
server: 7 layers × 2 events listed in each set.

The earlier pair from doc 05 §8 (pre-fix vs fixed) stays valid and is a different comparison.

## 8. Owner 2026-09-06: event 991, (119.2, 96.2, 379.6), cluster 290 "strange for the Steiner-Graph"

Two answers: cluster 290 has **no Steiner graph at all** and that is the display scope, not a bug —
but checking it turned up something that does matter for this feature.

### 8.1 Which set, and what cluster 290 is

That coordinate is cluster 290 in **exactly one** arm on disk. Every other event-991 arm (27 of
them, `stm0` / `d02*` / `d03*` / `d04bee` / `d05m*` / `d06base` …) calls it **cluster 83**:

| | |
|---|---|
| `d06um` (the with-unmerge link, §7) | cluster **290**, 11 points |
| everything else | cluster **83**, 12 289 points |

So cluster 290 is an 11-point clump that `unmerge_assoc` split off cluster 83 — and cluster 83 is
one of the `TGM_gained` movers of doc 04 §10.3, hand-labelled **THRU**. The clump has **nothing
within 50 cm**; the muon's body passes 50–80 cm away (1 032 points of cluster 83 within 80 cm, none
within 50). That is the isolated-merge signature doc pdvd/39 r2 describes almost exactly ("a
938-point body plus nine 5–19 point fragments at 16–76 cm"), so splitting it off is the intended
behaviour, not the anomaly.

### 8.2 Why it has no Steiner graph — and neither does its 12 289-point parent

`pr.jsonnet` scopes the three layers with **`require_pc: 'stm_fit'`**, so `stm`, `steiner_graph`
and `steiner_terminals` cover exactly the clusters `TaggerCheckSTM` recorded a fit for. Proof, not
inference — the layer's cluster set equals the `persist_stm_fit` set exactly, both arms:

| arm | clusters | `steiner_graph` clusters | `persist_stm_fit` clusters |
|---|---|---|---|
| `d06base` | 86 | 20: `29 34 36 37 40 45 46 50 57 73 82 85 88 92 95 99 100 101 104 113` | the same 20 |
| `d06um` | 418 | 15: `29 32 36 37 40 42 50 73 82 85 88 95 101 104 113` | the same 15 |

Cluster 290 is a fragment, never promoted to a main, never evaluated — no fit, no layer. **And
cluster 83 has no Steiner graph either**, in any arm: it is `TGM=true`, so `TaggerCheckSTM: cluster
83 already TGM; skipping`. Measured: **zero** Steiner, terminal, `stm` or `stm_fit` points within
80 cm of that coordinate, in both arms. Nothing is drawn there because nothing is meant to be.

### 8.3 The thing that does matter: the retile undoes the split

Looking at the Steiner layer for the rest of the event, one cluster stands out. Cluster **36**:

| | `d06base` | `d06um` |
|---|---|---|
| charge points | 2 650 | 1 581 (unmerge split off 442 / 439 / 444 / 446 …) |
| Steiner nodes | 2 848 | 2 834 |
| nodes on **its own** charge | 2 848 / 2 848 | 1 759 / 2 834 |
| nodes on **another cluster's** charge | **0** | **1 075** → 521 on cl 442, 173 on cl 439, 112 on cl 444, 80 on cl 446 |

Every one of those four clusters was part of cluster 36 before the split (verified point-for-point,
max match distance 0.000 cm). **So the Steiner build puts 38 % of cluster 36's nodes back on the
charge `unmerge_assoc` had just removed from it.**

The mechanism is in the code, not a guess. `steiner: cm.steiner(retiler=improve2, …)`, and
`ImproveCluster_1::get_activity_improved` (`improvecluster_1.cxx:256`) takes the cluster's
**(u, v, w, t) bounding box** — `get_uvwt_min/max` — and asks the *Grouping* for every good channel's
charge inside it:

```cpp
map_u_tcc = grouping->get_overlap_good_ch_charge(min_time, max_time, min_uch, max_uch, apa, face, 0);
```

There is **no cluster-membership filter**. Any charge inside the box is retiled back in, whoever
owns it. Unmerge removes the fragments in 3-D; the main's bounding box still contains them; they
return in 2-D.

**This is pre-existing and not caused by unmerge** — that had to be checked before blaming the new
feature. Over event 12 the fraction of Steiner nodes sitting on another cluster's charge is
**7.9 % in both arms** (4 242 / 53 585 vs 4 024 / 50 728); the mechanism is simply how the retiler
has always worked. What unmerge changes is *whose* charge it is: a main's own split-off fragments
now count as "another cluster", so on event 1 the fraction goes 0.1 % → 4.8 %, essentially all of
it cluster 36.

**The consequence for this feature is the one worth stating: the unmerge split does not reach the
Steiner build.** Whatever it removes in 3-D comes back through the 2-D bounding-box retile, so the
STM fit still walks over the fragments' charge. PDVD carries the identical wiring
(`protodunevd/pr.jsonnet:1410`, `cm.steiner(retiler=improve2, …)`), so the same limitation presumably
applies to the shipped PDVD unmerge — **measured here on PDHD only**, and worth checking there
before anyone leans on either.

## 9. Not done / next

1. **Grading.** Two events, no hand scan, sign unknown. Before anyone proposes making this a PDHD
   default it needs the 30-event arm and a scan of what the 40-odd splits per event did to the
   objects — the PDVD flip (doc pdvd/38+39) came *after* that work, not before.
2. **The accept guard on cluster 113** is now the sole remaining explanation for that lost stopper
   and is unread: `accept_guards_reject`'s desert + spike tests (`TaggerCheckSTM.cxx:3798-3857`)
   read the charge profile, which is exactly what the wrapped-channel fix changed.
3. **Cluster 108's kink relocation** is unexplained by either the fix or the merge.
4. `-save-assoc` changes the pctree, so a run that wants `-unmerge` cannot reuse an existing
   pctree. Every historical PDHD arm needs re-clustering to be usable with it.

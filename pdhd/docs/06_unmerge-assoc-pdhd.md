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

## 8. Not done / next

1. **Grading.** Two events, no hand scan, sign unknown. Before anyone proposes making this a PDHD
   default it needs the 30-event arm and a scan of what the 40-odd splits per event did to the
   objects — the PDVD flip (doc pdvd/38+39) came *after* that work, not before.
2. **The accept guard on cluster 113** is now the sole remaining explanation for that lost stopper
   and is unread: `accept_guards_reject`'s desert + spike tests (`TaggerCheckSTM.cxx:3798-3857`)
   read the charge profile, which is exactly what the wrapped-channel fix changed.
3. **Cluster 108's kink relocation** is unexplained by either the fix or the merge.
4. `-save-assoc` changes the pctree, so a run that wants `-unmerge` cannot reuse an existing
   pctree. Every historical PDHD arm needs re-clustering to be usable with it.

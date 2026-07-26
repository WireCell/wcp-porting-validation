# SBND `work-*` tag index

Repro:

```bash
cd sbnd_xin
ls -d work* | wc -l              # 46 at the top level after the 2026-07-25 consolidation
ls archive/*/ -d                 # 3 campaign archives, 79 dirs
find . -xtype l | wc -l          # 0 -- MUST stay 0, see "the symlink hazard" below
python3 relink_tags.py           # dry-run repair after any move
```

Every tagged output dir is one arm of one A/B or one hand scan, produced by
`run_ql_evt.sh` / `run_pr_evt.sh` / `run_nusel_evt.sh -t <tag>` over one of four
event manifests:

| manifest prefix | what it is |
|---|---|
| `work-mcp10-<tag>` | the 10/30-event hand-scan set (imaging from `work-mcp10`) |
| `work-mcp1000-<tag>` | 30 events drawn from the 1000-event MC set (imaging from `work-mcp1000`) |
| `work-mcp1000b-<tag>` | a second 30-event draw from the same 1000-event set |
| `work-mcp1kall-<tag>` | **all 1000** events of that set, not a draw (doc 59; imaging from `work-mcp1000`) |
| `work-mcsim-<tag>` | the MC-truth sim set (truth dQ/dx, delta rays) |

**These arms are not reproducible.** A tag names a *config*, not a build. The
binary has moved many commits since most of them were written, and the SBND PR
chain is ASLR-non-deterministic on top of that. Re-running `-t <tag>` today
produces a *different* arm, so retiring a directory permanently retires the
ability to re-check the doc table it backs — the doc's stated PASS becomes the
only surviving record. That is why the classification below is by **who names
the directory**, not by age or size.

## The symlink hazard — read before moving anything

A tag dir does **not** hold a private copy of every event. Any stage the run did
not redo is an **absolute symlink** into an earlier tag's dir, often chained two
deep:

```
work-mcp10-dq48v3/ql_evt284349
    -> /nfs/.../sbnd_xin/work-mcp10-fvxy/ql_evt284349
    -> /nfs/.../sbnd_xin/work-mcp10-mainreal/ql_evt284349
```

So the tags form a dependency graph and **moving a tag breaks every dependent**.
Archiving the doc-29..49 tags below broke 1536 links. It failed silently in the
worst way: `stm_fv_census.py` reported `0 "contained" clusters` instead of 147,
because a missing pctree is a `continue`, not an error. `relink_tags.py`
rewrites broken links to wherever the tag now lives; run it after ANY move and
confirm `find . -xtype l | wc -l` is 0.

Verification that the move was faithful: `python3 stm_fv_census.py` after the
repair reproduces doc 49 §4 line for line (147 contained / 96 outside / 65 %,
median 2.88, p90 3.54, max 3.77, walls 23/61/4/8, agree 96/96), and
`stmon_stats.py` reproduces 30 events / 36 fitted clusters / 18561 fit points.

## BASE — 3 dirs, 10026 MB (top level)

**base input** — the imaging / PR products every tagged arm links into. Never delete: regenerating `work-mcp1000` alone is a 2000-event imaging run.

| dir | entries | size | referenced by |
|---|---|---|---|
| `work` | 715 | 2795M | — |
| `work-mcp10` | 63 | 96M | — |
| `work-mcp1000` | 2000 | 7136M | — |

## LIVE — 7 dirs, 8.5 GB (top level)

**wired into a running viewer** — the port-5011 `nusel_scan_viewer.py` command line names these as its current tag or as a `--prev` baseline. Deleting or moving one blanks the live scan.

The doc-56 `work-*-d56bw` arms are also live as `--prev` baselines of the doc-59
scan; they are listed under CURRENT below.

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp1kall-d59k` | 2999 | 8.3G | `59_full1k-production-scan.md`, `nusel_scan_filter.py`, `run_full1k_nusel.sh`, `make_scan_bee.sh` — the port-5011 `s59k` scan (648 of its 999 tables) |
| `work-mcp10-d49son` | 43 | 29M | `50_stm-fit-scope-and-unmerge.md`, `51_clustering-merge-attribution.md`, `52_isolated-grouping-fix-design.md`, `d52_ab_report.py`, `stm_main_connectivity.py`, `stm_merge_attribution.py` |
| `work-mcp10-d52ron` | 53 | 60M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md`, `unmerge_crosser_audit.py` |
| `work-mcp1000-d49son` | 32 | 23M | `50_stm-fit-scope-and-unmerge.md`, `51_clustering-merge-attribution.md`, `d52_ab_report.py`, `stm_main_connectivity.py` |
| `work-mcp1000-d52ron` | 30 | 46M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md`, `unmerge_crosser_audit.py` |
| `work-mcp1000b-d49son` | 32 | 23M | `50_stm-fit-scope-and-unmerge.md`, `51_clustering-merge-attribution.md`, `d52_ab_report.py` |
| `work-mcp1000b-d52ron` | 30 | 44M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md`, `unmerge_crosser_audit.py` |

## CURRENT — 52 dirs, 2109 MB (top level)

the campaigns still in flight: docs 52 (isolated grouping) and 53 (`real_cluster_id`), plus the `d55b`/`d55t` arms of doc 52 §13, the doc-54 perf A/B arms and the doc-56 beam-window-gate arms (`p56off` knob-off gate, `d56bw` = the new production default, served on :5011).

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp10-d52chk` | 12 | 4M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d52off` | 52 | 60M | `d52_ab_report.py` |
| `work-mcp10-d52on` | 53 | 60M | `52_isolated-grouping-fix-design.md`, `d52_ab_report.py` |
| `work-mcp10-d52roff` | 52 | 60M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d52rpoff` | 52 | 60M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d52trace` | 2 | 2M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d53beeoff` | 52 | 60M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53beeon` | 52 | 60M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53dflt` | 52 | 60M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53leg` | 30 | 30M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53off` | 52 | 60M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53on` | 52 | 60M | `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp10-d53r` | 52 | 60M | `52_isolated-grouping-fix-design.md` |
| `work-mcp10-d55boff` | 52 | 60M | — |
| `work-mcp10-d55bon` | 52 | 60M | — |
| `work-mcp10-d55toff` | 52 | 60M | — |
| `work-mcp10-d55ton` | 53 | 60M | — |
| `work-mcp10-p54base` | 30 | 30M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp10-p54opt` | 30 | 30M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp10-p55opt` | 30 | 30M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp10-p56off` | 30 | 30M | `56_beam-window-tagger-gate.md`, `p54_ab_report.py` |
| `work-mcp10-d56bw` | 30 | 25M | `56_beam-window-tagger-gate.md`, `bwgate_report.py`, `mabc_step_totals.py`, `nusel_display/serve_nusel_scan.sh` |
| `work-mcp10-trace51` | 6 | 36M | `51_clustering-merge-attribution.md`, `stm_merge_attribution.py` |
| `work-mcp1000-d52off` | 30 | 47M | `d52_ab_report.py` |
| `work-mcp1000-d52on` | 30 | 47M | `52_isolated-grouping-fix-design.md`, `d52_ab_report.py` |
| `work-mcp1000-d52roff` | 30 | 47M | `52_isolated-grouping-fix-design.md` |
| `work-mcp1000-d52rpoff` | 30 | 47M | `52_isolated-grouping-fix-design.md` |
| `work-mcp1000-d53off` | 30 | 47M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp1000-d55boff` | 30 | 47M | — |
| `work-mcp1000-d55bon` | 30 | 46M | — |
| `work-mcp1000-d55toff` | 30 | 47M | — |
| `work-mcp1000-d55ton` | 30 | 46M | — |
| `work-mcp1000-p54base` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000-p54opt` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000-p55opt` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000-p56off` | 30 | 24M | `56_beam-window-tagger-gate.md`, `p54_ab_report.py` |
| `work-mcp1000-d56bw` | 30 | 20M | `56_beam-window-tagger-gate.md`, `bwgate_report.py`, `mabc_step_totals.py`, `nusel_display/serve_nusel_scan.sh` |
| `work-mcp1000b-d52off` | 30 | 44M | `d52_ab_report.py` |
| `work-mcp1000b-d52on` | 30 | 44M | `52_isolated-grouping-fix-design.md`, `d52_ab_report.py` |
| `work-mcp1000b-d52roff` | 30 | 44M | `52_isolated-grouping-fix-design.md` |
| `work-mcp1000b-d52rpoff` | 30 | 44M | `52_isolated-grouping-fix-design.md` |
| `work-mcp1000b-d53off` | 30 | 44M | `52_isolated-grouping-fix-design.md`, `53_unmerge-vs-cathode-crossers.md` |
| `work-mcp1000b-d55boff` | 30 | 44M | — |
| `work-mcp1000b-d55bon` | 30 | 44M | — |
| `work-mcp1000b-d55toff` | 30 | 44M | — |
| `work-mcp1000b-d55ton` | 30 | 44M | — |
| `work-mcp1000b-p54base` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000b-p54opt` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000b-p55opt` | 30 | 24M | `54_tgm-stm-perf-round1.md`, `p54_ab_report.py` |
| `work-mcp1000b-p56off` | 30 | 23M | `56_beam-window-tagger-gate.md`, `p54_ab_report.py` |
| `work-mcp1000b-d56bw` | 30 | 19M | `56_beam-window-tagger-gate.md`, `bwgate_report.py`, `mabc_step_totals.py`, `nusel_display/serve_nusel_scan.sh` |
| `work-smoke-d55pv` | 12 | 5M | — |

## `archive/tgm-docs29-39/` — 30 dirs, 257 MB

**TGM / LM / FV campaign.** every arm whose newest citation is doc ≤39 — the merge-aware TGM chain (docs 29-33), the LM tagger (34), interior FV and main-component-pairs (35-36), pctree provenance (38) and the FC/TGM x-y margins (39).

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp10-chord` | 43 | 3M | `29_tgm-chord-charge.md`, `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md`, `32_tgm-component-rescue-fvz.md`, `35_tgm-interior-fv.md`, `nusel_display/README.md`, `run_nusel_evt.sh`, `wct-pr-perevt.jsonnet` |
| `work-mcp10-ctpcfix` | 43 | 33M | `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md`, `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp10-fvxy` | 43 | 3M | `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp10-lm` | 53 | 37M | `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `lm_tune.py`, `nusel_display/nusel_scan_viewer.py`, `nusel_extract.py`, `qlmatching.jsonnet`, `run_ql_evt.sh`, `wct-clus-matching-perevt.jsonnet` |
| `work-mcp10-lm-offgate` | 2 | 2M | `34_lm-tagger.md` |
| `work-mcp10-lm2-offgate` | 2 | 2M | `34_lm-tagger.md` |
| `work-mcp10-mainflag` | 43 | 30M | `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md`, `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md` |
| `work-mcp10-merge` | 42 | 3M | `15_overclustering-evt11-gamma.md`, `23_nusel-tgm-stm-chain.md`, `29_tgm-chord-charge.md`, `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md`, `51_clustering-merge-attribution.md`, `52_isolated-grouping-fix-design.md`, `nusel_display/README.md`, `nusel_display/nusel_scan_viewer.py`, `nusel_display/serve_nusel_scan.sh`, `stm_main_connectivity.py` |
| `work-mcp10-merge2` | 43 | 3M | `29_tgm-chord-charge.md`, `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md`, `nusel_display/README.md`, `nusel_display/serve_nusel_scan.sh` |
| `work-mcp10-offgate` | 42 | 3M | `29_tgm-chord-charge.md` |
| `work-mcp10-offgate2` | 42 | 3M | `29_tgm-chord-charge.md` |
| `work-mcp10-pathchord` | 33 | 3M | `31_tgm-chord-path-mode.md` |
| `work-mcp10-rcidoff` | 30 | 34M | `38_pctree-provenance-tgm-main-real.md` |
| `work-mcp10-rcoff` | 42 | 3M | `33_tgm-rescue-chord.md` |
| `work-mcp10-reschord` | 43 | 3M | `33_tgm-rescue-chord.md`, `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md` |
| `work-mcp10-tgmfv` | 43 | 3M | `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md`, `34_lm-tagger.md` |
| `work-mcp10-tgmfv-offgate` | 2 | 233K | `32_tgm-component-rescue-fvz.md` |
| `work-mcp1000-ctpcfix` | 33 | 25M | `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md`, `34_lm-tagger.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp1000-fvxy` | 33 | 2M | `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp1000-lm` | 33 | 28M | `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `38_pctree-provenance-tgm-main-real.md`, `lm_tune.py`, `nusel_display/nusel_scan_viewer.py`, `nusel_extract.py`, `qlmatching.jsonnet`, `run_ql_evt.sh`, `wct-clus-matching-perevt.jsonnet` |
| `work-mcp1000-mainflag` | 32 | 22M | `30_matched-mains-main-flag.md`, `31_tgm-chord-path-mode.md` |
| `work-mcp1000-pathchord` | 22 | 2M | `31_tgm-chord-path-mode.md` |
| `work-mcp1000-rcoff` | 30 | 2M | `33_tgm-rescue-chord.md` |
| `work-mcp1000-reschord` | 32 | 2M | `33_tgm-rescue-chord.md`, `34_lm-tagger.md`, `36_tgm-main-component-pairs.md` |
| `work-mcp1000-tgmfv` | 33 | 2M | `32_tgm-component-rescue-fvz.md`, `33_tgm-rescue-chord.md`, `34_lm-tagger.md` |
| `work-mcp1000b-fvxy` | 33 | 2M | `39_tgm-fc-fv-xy-margins.md` |
| `work-mcp1000b-fvzi-offgate` | 3 | 131K | `35_tgm-interior-fv.md` |
| `work-mcp1000b-nucdbg` | 2 | 109K | `36_tgm-main-component-pairs.md` |
| `work-mcp1000b-smoke38` | 3 | 1M | `38_pctree-provenance-tgm-main-real.md` |
| `work-mcp1000b-smoke38c` | 3 | 1M | `38_pctree-provenance-tgm-main-real.md` |

## `archive/stm-docs40-49/` — 43 dirs, 622 MB

**STM track-fit campaign.** arms whose newest citation is doc 40-49 — the STM fit dump and showcase (41-43), truth dQ/dx and delta rays (44,46), the un-merge into main+associated (45), the Bragg reference retune (47-48) and the STM containment FV fix (49). Also the three `stmon` arms, named only by `stmon_stats.py`.

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp10-dq48` | 42 | 3M | `48_sbnd-dqdx-tables-and-mip.md`, `49_stm-containment-fv-inconsistency.md`, `stm_fv_census.py` |
| `work-mcp10-dq48base` | 42 | 3M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp10-dq48tab` | 42 | 3M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp10-dq48v3` | 45 | 28M | `45_unmerge-bundle-main-associated.md`, `49_stm-containment-fv-inconsistency.md`, `stm_fv_census.py` |
| `work-mcp10-dq49off` | 42 | 28M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp10-dq49off2` | 42 | 27M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp10-fvzi` | 43 | 3M | `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp10-lm2` | 43 | 37M | `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp10-mainpair` | 42 | 3M | `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp10-mainreal` | 42 | 37M | `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp10-stmon` | 43 | 4M | `41_stm-fit-dump.md`, `42_stm-fit-showcase-evt286241.md`, `43_magnify-tracking-sbnd-bugs.md`, `47_stm-bragg-reference-sbnd-retune.md`, `make_stmfit_bee.py`, `stmfit_showcase.py`, `stmon_stats.py` |
| `work-mcp10-unm45` | 43 | 27M | `45_unmerge-bundle-main-associated.md` |
| `work-mcp1000-dq48` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md`, `stm_fv_census.py` |
| `work-mcp1000-dq48base` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000-dq48tab` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000-dq48v3` | 35 | 22M | `45_unmerge-bundle-main-associated.md`, `49_stm-containment-fv-inconsistency.md`, `stm_fv_census.py` |
| `work-mcp1000-dq49off` | 32 | 22M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp1000-dq49off2` | 32 | 21M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp1000-fvzi` | 32 | 2M | `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000-lm2` | 33 | 28M | `34_lm-tagger.md`, `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000-mainpair` | 32 | 2M | `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000-mainreal` | 32 | 28M | `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000-stmon` | 32 | 4M | `stmon_stats.py` |
| `work-mcp1000-unm45` | 30 | 21M | `45_unmerge-bundle-main-associated.md` |
| `work-mcp1000b-dq48` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md`, `stm_fv_census.py` |
| `work-mcp1000b-dq48base` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000b-dq48tab` | 32 | 2M | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000b-dq48v3` | 34 | 20M | `45_unmerge-bundle-main-associated.md`, `49_stm-containment-fv-inconsistency.md`, `stm_fv_census.py` |
| `work-mcp1000b-dq49off` | 32 | 20M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp1000b-dq49off2` | 32 | 20M | `49_stm-containment-fv-inconsistency.md` |
| `work-mcp1000b-evnew` | 6 | 394K | `48_sbnd-dqdx-tables-and-mip.md` |
| `work-mcp1000b-fvzi` | 32 | 2M | `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000b-lm2` | 33 | 27M | `35_tgm-interior-fv.md`, `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000b-mainpair` | 32 | 2M | `36_tgm-main-component-pairs.md`, `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000b-mainreal` | 32 | 27M | `38_pctree-provenance-tgm-main-real.md`, `39_tgm-fc-fv-xy-margins.md`, `41_stm-fit-dump.md` |
| `work-mcp1000b-stmon` | 32 | 3M | `stmon_stats.py` |
| `work-mcp1000b-unm45` | 30 | 20M | `45_unmerge-bundle-main-associated.md` |
| `work-mcsim-diffusion` | 1 | 5K | `run_ql_evt.sh` |
| `work-mcsim-stmon` | 52 | 109M | `42_stm-fit-showcase-evt286241.md`, `44_stm-fit-truth-dqdx.md`, `45_unmerge-bundle-main-associated.md`, `46_stm-fit-deltarays-and-gui.md` |
| `work-mcsim-unmcomp` | 21 | 397K | `45_unmerge-bundle-main-associated.md` |
| `work-mcsim-unmoff` | 42 | 3M | `45_unmerge-bundle-main-associated.md` |
| `work-mcsim-unmon` | 42 | 3M | `45_unmerge-bundle-main-associated.md` |
| `work-stmbadch` | 2 | 344K | `42_stm-fit-showcase-evt286241.md`, `43_magnify-tracking-sbnd-bugs.md` |

## `archive/aborted-d54/` — 6 dirs, 145 MB

**SBND `d54off`/`d54on` pair — off-arm complete, on-arm empty.** What is
observable: each `d54off` holds a full run (20 `.batch_nusel_*.log`, `ql_evt*`,
`nusel_evt*`), while each `d54on` holds ten event dirs that are empty and **no
batch log at all** — so the on-arm never wrote anything. Whether it was killed,
never launched, or is queued for relaunch is not recorded here, so treat
"aborted" as a description of the output, not of intent. No doc cites either
arm; doc 52's `d54base`/`d54opt2` are `abtest/snap/` labels, a different thing.
Kept rather than deleted because the off-arm is a complete 30-event run.

**If that campaign is relaunched** with `-t d54on`, the runner will create a
fresh dir at the *top level* while its sibling sits here — move the pair back
out (and re-run `relink_tags.py`) rather than leaving the arms split.

| dir | entries | size | referenced by |
|---|---|---|---|
| `work-mcp10-d54off` | 52 | 60M | — |
| `work-mcp10-d54on` | 10 | 5K | — |
| `work-mcp1000-d54off` | 29 | 41M | — |
| `work-mcp1000-d54on` | 10 | 5K | — |
| `work-mcp1000b-d54off` | 30 | 44M | — |
| `work-mcp1000b-d54on` | 10 | 5K | — |

## RETIRED 2026-07-25 — 44 dirs, 307 MB, deleted

Every entry here was verified to be named by no committed doc, no analysis
script, no running viewer, and no `abtest/`/`sweep/` gate snapshot, using a
**bare-tag** search (`dq48tab`, `d53leg`, …) across `docs/`, `nusel_display/`,
`ql_scan/`, `*.py`, `*.sh`, `*.jsonnet`. A full-directory-name search is NOT
sufficient — the docs cite tags without the manifest prefix, and that mistake
initially mislabelled `d53dflt`/`d53leg` (evidence for a gate committed the same
day) as unreferenced.

Group A is the ad-hoc probe class: 2-5 event runs made while chasing one number.
Group B is full-manifest arms that nothing names — mostly gate off-arms and
superseded variants whose partner arm survives. No link anywhere pointed into
any of them, so the deletion broke nothing. Nothing inside `work/` was touched.

| dir | group | entries | size | what it was |
|---|---|---|---|---|
| `work-d49diag-nofit` | A | 3 | 866K | doc 49 STM-containment FV diagnostic probe |
| `work-d49diag-off` | A | 3 | 928K | doc 49 STM-containment FV diagnostic probe |
| `work-d49diag-rep1` | A | 3 | 1M | doc 49 STM-containment FV diagnostic probe |
| `work-d49diag-rep2` | A | 3 | 1M | doc 49 STM-containment FV diagnostic probe |
| `work-det287517-fit1` | A | 3 | 762K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-fit2` | A | 3 | 762K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-flagtest2` | A | 3 | 4M | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-nofit3` | A | 3 | 329K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-nofit4` | A | 3 | 329K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-oldfit` | A | 3 | 762K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-det287517-oldnofit` | A | 3 | 329K | doc 41/42 STM track-fit probe on data evt 287517 |
| `work-mcp10-d49soff` | B | 42 | 27M | doc 49/50 STM-scope gate off-arm |
| `work-mcp10-d53chk` | A | 2 | 2M | doc 53 Bee-layer spot check, evt 284349 |
| `work-mcp10-dq48fit` | B | 43 | 4M | doc 48 dQ/dx fit intermediate |
| `work-mcp10-dq48v2` | B | 42 | 28M | doc 48 dQ/dx table v2 (superseded by v3) |
| `work-mcp10-dq49` | B | 42 | 29M | doc 49 first pass (superseded by dq49off2) |
| `work-mcp10-mp36off` | B | 42 | 3M | doc 36 main-component-pairs off-arm |
| `work-mcp10-mpoff` | B | 42 | 3M | doc 36 main-component-pairs off-arm |
| `work-mcp10-prodoff` | B | 42 | 3M | production-default off-arm |
| `work-mcp10-stmoff` | B | 42 | 3M | STM-tagger off-arm |
| `work-mcp1000-d49soff` | B | 32 | 21M | doc 49/50 STM-scope gate off-arm |
| `work-mcp1000-dq48fit` | B | 33 | 4M | doc 48 dQ/dx fit intermediate |
| `work-mcp1000-dq48v2` | B | 32 | 22M | doc 48 dQ/dx table v2 (superseded by v3) |
| `work-mcp1000-dq49` | B | 32 | 23M | doc 49 first pass (superseded by dq49off2) |
| `work-mcp1000-mp36off` | B | 30 | 2M | doc 36 main-component-pairs off-arm |
| `work-mcp1000-mpoff` | B | 32 | 2M | doc 36 main-component-pairs off-arm |
| `work-mcp1000-prodoff` | B | 30 | 2M | production-default off-arm |
| `work-mcp1000-stmoff` | B | 30 | 2M | STM-tagger off-arm |
| `work-mcp1000-tgmfv-dbg3` | A | 2 | 325K | doc 32 TGM FV debug |
| `work-mcp1000-tgmfv-dbg5` | A | 2 | 330K | doc 32 TGM FV debug |
| `work-mcp1000b-d49soff` | B | 32 | 20M | doc 49/50 STM-scope gate off-arm |
| `work-mcp1000b-dq48fit` | B | 32 | 3M | doc 48 dQ/dx fit intermediate |
| `work-mcp1000b-dq48v2` | B | 32 | 20M | doc 48 dQ/dx table v2 (superseded by v3) |
| `work-mcp1000b-dq49` | B | 32 | 23M | doc 49 first pass (superseded by dq49off2) |
| `work-mcp1000b-evbase` | A | 6 | 394K | doc 48 dQ/dx table pre-check |
| `work-mcp1000b-mp36off` | B | 30 | 2M | doc 36 main-component-pairs off-arm |
| `work-mcp1000b-mpoff` | B | 32 | 2M | doc 36 main-component-pairs off-arm |
| `work-mcp1000b-nucoffdbg` | A | 2 | 109K | doc 36 nu-candidate off-arm debug |
| `work-mcp1000b-prodoff` | B | 30 | 2M | production-default off-arm |
| `work-mcp1000b-sa1` | B | 32 | 20M | doc 49 scope-alignment trial |
| `work-mcp1000b-sa2` | B | 32 | 20M | doc 49 scope-alignment trial |
| `work-mcp1000b-stmoff` | B | 30 | 2M | STM-tagger off-arm |
| `work-mcsim-unmerge` | A | 20 | 6K | doc 45 empty stub (run produced no output) |
| `work-why285185` | A | 15 | 740K | one-off "why is evt 285185 like this" probe |

# doc 109 — a self-describing tracking-pr.root: what the neutrino selection did, corrected cluster flags, multiple beam flashes, provenance

**Owner ask (2026-09-14):** implement items 1, 2 and 4 of doc 108
(`108_root-output-improvement-plan.md`) "to improve the track_pr.root saved
information … test it on the sbnd_xin data events, and make a md file to
summarize the changes". Truth (item 3) is skipped for now.

**Owner decisions for this round:** once the gates pass, turn the new content ON for the
SBND production job, keeping the C++ and jsonnet defaults OFF; push the toolkit branch.

**Status:**
- **Implemented and gated; SBND production ON.**
- **Toolkit:** `apply-pointcloud` `b207b6d8` (code, tests, knobs default OFF) and `c203b400` (SBND production flip).
- **Knobs-off gate:** byte-identical.
- **Knob-on:** reconstruction outputs byte-identical; only `tracking-pr.root` gains content.
- **Rev 2 (same day, sec 7):**
  - group mode gated: `tracking-pr.root` is byte-identical to per-event;
  - new production reference `ref/prod-2026-09-14` (PASS 21/21, every drift attributed);
  - `ref/prod-2026-09-08` removed;
  - no toolkit code change.

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin
# 0. pins: base = toolkit d3b398fc unmodified; new = this round's code
#    ~/tmp/d109-libsnap/{base,new,new2}; new2 = the final build every sec 4 gate ran on
#    (md5 of all three in docs/109_logs/libsnap.md5; "new" = the round-1 build)
#    cfg trees: ~/tmp/d109-cfg/pristine/cfg (git archive d3b398fc), ~/tmp/d109-cfg/new/cfg
# 1. compiled-config gate, knobs off (21 consumers: SBND prod PR job, lar 1-step, PDHD, PDVD, uBooNE)
scripts/cfg/compile_consumers.sh ~/tmp/d109-cfg/pristine/cfg ~/tmp/d109-cfg/A
scripts/cfg/compile_consumers.sh <toolkit>/cfg                ~/tmp/d109-cfg/B
scripts/cfg/cmp_consumers.sh ~/tmp/d109-cfg/A ~/tmp/d109-cfg/B          # docs/109_logs/cfg_gate.txt
# 2. unit tests
<toolkit>/build/clus/wcdoctest-clus ; <toolkit>/build/root/wcdoctest-root
# 3. arms: stage B on the doc 102 stage-A pctrees; nuecc48 (48) + ncpi0 (19) + first 200 mcp1k
#    (docs/109_logs/events_mcp1k200.txt); setarch -R, pr_display, geometric vertex for the byte gates
scripts/d109_arms.sh d109base ~/tmp/d109-libsnap/base SBND_NO_DL=1 PR_CFG_TREE=~/tmp/d109-cfg/pristine/cfg
#    (round 1 arms d109off/d109on/d109dlon ran the first build; the final build is new2, arms *2)
scripts/d109_arms.sh d109off2  ~/tmp/d109-libsnap/new2 SBND_NO_DL=1 SBND_ROOT_OUTPUT=0 PR_CFG_TREE=~/tmp/d109-cfg/new/cfg
scripts/d109_arms.sh d109on2   ~/tmp/d109-libsnap/new2 SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 PR_CFG_TREE=~/tmp/d109-cfg/new/cfg
scripts/d109_arms.sh d109dlon2 ~/tmp/d109-libsnap/new2 SBND_ROOT_OUTPUT=1 PR_CFG_TREE=~/tmp/d109-cfg/new/cfg   # DL vertex ON
# 4. gates
python3 scripts/d109_gate.py d109base d109off2               > docs/109_logs/gate_base_vs_off.txt
python3 scripts/d109_gate.py d109off2 d109on2 --allow T_cluster.tgm T_cluster.stm T_cluster.fc T_cluster.lm T_cluster.beam_flash \
    --prefix T_tagger.act_cluster_id T_tagger.act_length_cm T_tagger.act_is_selected T_tagger.act_is_demoted \
             T_tagger.act_tgm T_tagger.act_stm T_tagger.act_fc T_tagger.act_lm T_tagger.act_evaluated \
                                                              > docs/109_logs/gate_off_vs_on.txt
python3 scripts/d109_root_checks.py d109on2                  > docs/109_logs/checks_d109on.txt
python3 scripts/d109_root_checks.py d109dlon2                > docs/109_logs/checks_d109dlon.txt
```

```bash
# 5. production flip (toolkit c203b400) -- compiled configs and the production smoke
scripts/cfg/compile_consumers.sh <toolkit>/cfg ~/tmp/d109-cfg/C
scripts/cfg/cmp_consumers.sh ~/tmp/d109-cfg/A ~/tmp/d109-cfg/C          # docs/109_logs/cfg_gate_production_flip.txt
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-08 --cfg ~/tmp/d109-cfg/pristine/cfg   # prod_cfg_gate_pristine_head_vs_0908.txt
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-08                                     # prod_cfg_gate_flip_vs_0908.txt
#    (rev 2 removed ref/prod-2026-09-08: restore it with `git -C .. checkout 70a49e8c -- sbnd_xin/ref/prod-2026-09-08`
#     to rerun these two; the current reference is ref/prod-2026-09-14, sec 7.2)
PR_EXTRA_STAGES=pr_display setarch x86_64 -R ./run_pr_chain_batch.sh work-nuecc48-d102m work-nuecc48-d109prod data 137238 269774 10550
PR_EXTRA_STAGES=pr_display setarch x86_64 -R ./run_pr_chain_batch.sh work-mcp1k-d102m  work-mcp1k-d109prod  data <3 no-row events>
python3 scripts/d109_root_checks.py d109prod --samples nuecc48 mcp1k  > docs/109_logs/checks_d109prod_smoke.txt
# 6. rev 2 -- group mode, the base-binary control, the new production reference: sec 7.4
```

---

## 0. Summary

**What an SBND `tracking-pr.root` now answers**, with SBND production writing it by default:

| Doc 108 item | Now in the file |
|---|---|
| 1 — both cluster ids | `T_tagger.sel_cluster_id` (what the selection chose) next to `cluster_id` (what the PR result belongs to), plus `vertex_moved_cluster`. With the DL vertex, 17 of 163 rows on the test events are moved rows. |
| 1 — full activity roster | the bundle's companions appended to `act_*` (`act_role` 2/3), `act_in_pr`, `act_is_final`; the old entries unchanged in place |
| 1 — `has_vertex` | on T_tagger and T_kine |
| 1 — why an event has no row | `T_bundle`, one row per in-window bundle with a reason code, plus Trun counters including the formerly silent "in-window cluster with no matched flash". All 106 no-row events of the 200 mcp1k events are explained. |
| 2 — T_cluster flags | `tgm/stm/fc/lm` read what SBND's taggers set; `beam_flash` derived with the selection's own window test; `matched_flash_gid`, `flash_tpc` added |
| 2 — multiple beam flashes | `T_flash` (every flash, physical TPC, in-window flag) and `flash_group` on T_tagger/T_bundle/T_flash (different-TPC flashes within 0.05 µs = one flash) |
| 2 — run/subrun/event per row | on T_tagger, T_kine, T_bundle, T_flash |
| 4 — self-describing | Trun `wct_version`, the BDT weight files, DL weights, TrackFitting file, and from the runner the operating-point sha256 and git revisions |

**Proof that nothing else moved** (267 sbnd_xin data events):
- **Knobs off vs the pre-round code:** every output byte-identical, including 214 565 ROOT
  branches.
- **Knobs on vs off:** `mabc-pr.zip`, the pctree, the nusel tables and the calib dump are
  identical, and 13 content checks have 0 failures.
- **Compiled configs:** all 21 consumers are unchanged with the knobs off. The production
  flip changes only the SBND PR job, and only by the new keys.

**Reader note:** `act_*` now also lists companions; use `act_role <= 1` for the old meaning
(sec 2.1).

**Production reference (rev 2):** `ref/prod-2026-09-14` at `c203b400`, PASS 21/21.
- **Rev 1's stale-reference drift is now named:** four inherited keys from the 2026-09-10
  master merge.
- **`ref/prod-2026-09-08` removed:** on the owner's word (sec 7.2).
- **Group mode:** `tracking-pr.root` is byte-identical to per-event with both vertex modes
  (sec 7.1).

---

## 1. The knobs

All three are observation-only: they add or correct ROOT content, and the selection,
vertex, PR and scores do not move (sec 4).

| jsonnet (`sbnd/clus.jsonnet` `pr()`, TLA in `sbnd/wct-pr-perevt.jsonnet`) | C++ keys it sets | Component | Effect |
|---|---|---|---|
| `root_nu_record` (+ `flash_pair_dt_us`, null = 0.05) | `nu_provenance` (+ `flash_pair_dt_us`) | `TaggerCheckNeutrino` | fills the new `TaggerInfo`/`KineInfo` fields; publishes a `NuBundleCensus` on the grouping **before** the no-candidate return |
| | `nu_provenance` | `UbooneTaggerOutputVisitor` (via `common/clus.jsonnet` `tagger_output`) | books the new `T_tagger`/`T_kine` branches |
| | `nu_provenance` | `SbndPrMagnifyTrackingVisitor` | writes `T_bundle`, `T_flash`, the Trun census branches |
| `root_cluster_flags` | `fix_cluster_flags` | `SbndPrMagnifyTrackingVisitor` | `T_cluster` tgm/stm/fc/lm/beam_flash corrected; `matched_flash_gid`, `flash_tpc` added |
| `root_provenance` (+ `provenance_extra`) | `provenance` (object) | `SbndPrMagnifyTrackingVisitor` | Trun string branches |

- **One switch, several keys.** `root_nu_record` sets the computing key and both writers'
  booking keys, so filled-but-unbooked or booked-but-empty branches cannot happen
  (same rule as `mcs_enable`, doc 80).
- **Defaults.** Every C++ key defaults off/empty (pinned in
  `clus/test/doctest_clus_knob_defaults.cxx` and
  `root/test/doctest_sbnd_pr_tracking_defaults.cxx`). Keys are omitted from the
  compiled config when off.
- **Runner.** `run_pr_chain_batch.sh` gains `SBND_ROOT_OUTPUT=<0|1>` (tri-state over the
  three TLAs). When the job writes provenance it compiles the job once per batch with
  placeholder per-event TLAs and passes `provenance_extra = {op_config_sha256, toolkit_git,
  wcp_git, runner, cfg_tree}` (`SBND_ROOT_PROVENANCE_EXTRA=0` skips it).

## 2. What the file now carries

### 2.1 `T_tagger` / `T_kine` (one row per neutrino candidate, as before)

| Branch | Tree | Meaning |
|---|---|---|
| `run`, `subrun`, `event` | both | the event (MABC's resolved triplet, = `Trun`) |
| `sel_cluster_id` | T_tagger | the activity the selection chose |
| `vertex_moved_cluster` | T_tagger | 1 when `cluster_id` (the cluster the row was written with) ≠ `sel_cluster_id`: the overall neutrino-vertex search moved the main cluster onto a companion of the same bundle |
| `has_vertex` | both | 1 = a neutrino vertex was found; 0 = none, and `nu_x/y/z` = (0,0,0) is a placeholder, not a point on the cathode |
| `flash_time_us`, `flash_pe`, `flash_tpc`, `flash_group` | T_tagger | the row's matched flash; `flash_tpc` = the flash's physical drift side (opflash `apa`); `flash_group` see 2.3 |
| `act_role` | T_tagger | 0 main, 1 demoted main (both evaluated by the cosmic taggers, same entries and order as before), **2 companion, 3 companion dropped as cosmic** (`skip_cosmic_companions`); each cluster listed once |
| `act_in_pr` | T_tagger | 1 = the cluster took part in this candidate's PR pass: the selected activity and every kept companion. A role-1 demoted main can also be a companion. |
| `act_is_final` | T_tagger | 1 on the entry whose cluster the row was written with |

- **Reader note — `act_*` now also lists companions.** The role-0/1 entries come first and
  are unchanged (gate sec 4.3). To keep the old meaning, filter `act_role <= 1`.
  - Companions carry `act_evaluated = 0`: the taggers' admission gate did not cover them.
    A TGM/STM flag on one is still that tagger's positive verdict.
  - Each cluster appears once. A demoted main that is also a companion keeps its role-1 entry
    with `act_in_pr = 1`.
    - The first build listed such a cluster twice (role 1 and role 2). On the DL-vertex arm,
      6 rows then carried `act_is_final` on two entries, caught by check C3
      (`109_logs/checks_d109dlon_round1_dup_roster.txt`). Fixed before the gates below.
- **The selected activity** is `sel_cluster_id` (or `act_cluster_id[act_is_selected & act_role<=1]`).
  The PR result (vertex, energy, scores) belongs to `cluster_id` (= `act_cluster_id[act_is_final]`).

### 2.2 `T_bundle` (new; one row per in-beam-window flash bundle, every event)

| Branch | Meaning |
|---|---|
| `run`, `subrun`, `event`, `gid` | event and bundle (matched flash gid) |
| `flash_tpc`, `flash_time_us`, `flash_pe`, `flash_group` | the bundle's flash |
| `n_main`, `n_demoted` | in-window mains / demoted mains in the bundle |
| `n_companion`, `n_companion_dropped` | companions kept / dropped as cosmic (selected bundles) |
| `n_rej_cosmic`, `n_rej_stm_only`, `n_rej_floor` | activities the selection rejected, by rule |
| `reason` | 0 selected main · 1 selected demoted-main fallback · 2 every examined activity cosmic-vetoed (TGM/STM/`lm_flag>0`) · 3 at least one activity failed only the `nu_per_bundle_min_length` floor · 4 rejected by `nu_per_bundle_stm_only` · 5 no activity examined |
| `nu_index` | the T_tagger/T_kine row; -1 = no candidate |
| `sel_cluster_id`, `final_cluster_id`, `sel_length_cm` | selected / written cluster |

**Trun census branches:**
- `nu_census_filled` (0 = TaggerCheckNeutrino published nothing);
- `nu_beam_gate`, `nu_per_bundle`;
- `beam_window_low_us`, `beam_window_high_us`, `flash_pair_dt_us` — the selection's own values;
- `nu_n_main`, `nu_n_in_window_main`, `nu_n_in_window_demoted`;
- **`nu_n_in_window_nogid`** — in-window clusters with no matched flash, which the selection drops with no other record;
- `nu_n_bundles`, `nu_n_candidates`, `nu_n_flashes`, `nu_n_flashes_in_window`.

**Scope:**
- Bundle rows exist in the per-bundle selection (SBND production, `nu_per_bundle=true`).
- The legacy single-winner path fills the counters only.
- The census is built inside `TaggerCheckNeutrino` from the same `NuCandidate` records the
  selection uses, so it cannot drift from it; it is not a re-derivation in the writer.

### 2.3 `T_flash` (new; one row per optical flash, every event) and `flash_group`

Branches: `run`, `subrun`, `event`, `gid`, `tpc`, `time_us`, `pe`, `in_window`,
`flash_group`, `n_matched_clusters`, `n_matched_main`, `nu_index`.

- **Source.** Built from the merge-safe `opflash` point cloud, which holds every flash of
  every input; `tpc` is its `apa` column = the flash's physical side
  (`QLMatching::write_opflash_pc`).
  - This closes both "verify" items of doc 108 sec 4.1: the TPC does not depend on the gid
    encoding, and the list is complete.
- **`flash_group`.** Flashes on **different TPCs** closer than `flash_pair_dt_us` (default
  0.05 µs, the end of the one-flash peak in doc 108 sec 4.1) are one physical flash seen by
  both TPCs.
  - Transitive; id = smallest gid in the group (`PR::group_flashes`, unit-tested in
    `clus/test/doctest_nu_bundle_census.cxx`).
  - Two T_tagger rows with the same `flash_group` come from one flash, often with an empty
    second row. Rows in different groups are candidate distinct interactions.

### 2.4 `T_cluster` corrections (`fix_cluster_flags`)

| Column | Before | Now |
|---|---|---|
| `tgm`, `stm`, `fc` | lowercase `Flags::tgm/short_track_muon/fully_contained`: always 0 on SBND | `Flags::TGM/STM/FC`, what TaggerCheckTGM/STM/FC set |
| `lm` | `Flags::light_mismatch`: always 0 | the `lm_flag` scalar Q/L matching sets (-1 = never set) |
| `beam_flash` | `Flags::beam_flash`: never set on SBND | derived: `matched_flash_gid >= 0` and `cluster_t0` in the selection's window, compared in internal units (the selection's own test); -1 if no census |
| `matched_flash_gid`, `flash_tpc` | — | new; `flash_id` is unchanged (it already holds the gid, doc 108 sec 4.2) |

### 2.5 Provenance strings in `Trun` (`root_provenance`)

- **Always:** `wct_version` (the library's build string).
- **From the job config:** `bdt_weights_dir`, `numu_xgboost_xml`, `nue_xgboost_xml`,
  `dl_weights`, `trackfitting_config`.
- **From the runner:** `op_config_sha256`, `toolkit_git`, `wcp_git`, `runner`, `cfg_tree`.
- **The hash** covers the job compiled with placeholder paths/RSE, so it identifies the
  operating point and is the same for every event of a batch.
- **Beam window and `flash_pair_dt_us`** are in the census branches (2.2).

## 3. Reading it

```python
import uproot, numpy as np
f = uproot.open("pr_evt<ID>/tracking-pr.root")
run = f["Trun"].arrays(library="np")                  # counters, window, provenance strings
bun = f["T_bundle"].arrays(library="np")              # one row per in-window bundle, every event
fl  = f["T_flash"].arrays(library="np")
if "T_tagger" in f:                                   # absent when no bundle gave a candidate
    t = f["T_tagger"].arrays(["run", "subrun", "event", "nu_index", "cluster_id", "sel_cluster_id",
                              "vertex_moved_cluster", "has_vertex", "nu_x", "nu_y", "nu_z",
                              "flash_time_us", "flash_tpc", "flash_group",
                              "act_cluster_id", "act_role", "act_in_pr", "act_is_final",
                              "act_is_selected", "act_tgm", "act_stm"], library="np")
```

| Question | Read |
|---|---|
| Which activity was selected / which cluster carries the PR result? | `sel_cluster_id` / `cluster_id`; `vertex_moved_cluster` flags the difference |
| Is the vertex real? | `has_vertex == 1` (a row with 0 has `nu_x/y/z = 0` as a placeholder) |
| The selected activity's cosmic verdicts | roster entry with `act_is_selected == 1 and act_role <= 1` |
| Everything in the candidate's PR pass | `act_in_pr == 1` |
| Why an event has no T_tagger | `T_bundle.reason` for each in-window bundle; if T_bundle is empty, `Trun.nu_n_in_window_main/demoted` (0 = nothing in the beam window) and `nu_n_in_window_nogid` |
| Are two rows one flash seen by both TPCs? | same `flash_group` (T_tagger or T_bundle); different groups = candidate distinct interactions |
| In-window flashes that made no candidate | `T_flash` rows with `in_window == 1 and nu_index == -1` |
| Truth time match (doc 108 sec 3.5) | the row's `flash_time_us` − 0.136 µs ≈ the true ν time |
| Cluster flags / beam flash per cluster | `T_cluster` `tgm/stm/fc/lm/beam_flash`, now filled |
| What produced this file | `Trun` `wct_version`, `op_config_sha256`, `toolkit_git`, `wcp_git`, weight files |

- **Joining trees:** `nu_index` joins T_tagger ↔ T_kine ↔ T_bundle ↔ T_flash, and `gid` joins
  T_bundle ↔ T_flash ↔ `T_cluster.matched_flash_gid`. Every new tree carries
  `run/subrun/event`.
- **`flash_tpc` and the gid:** `flash_tpc` equals `gid // 1000000` on every SBND row checked
  (sec 4.4). The branch is still the physical side and does not rely on that encoding.

## 4. Gates

**Pins** (`109_logs/libsnap.md5`; each arm's md5 was checked at start and end):

| Label | Libraries | Clus / Root md5 | cfg tree |
|---|---|---|---|
| `work-<s>-d109base` | toolkit `d3b398fc` unmodified | `ef0822ec` / `46bf5105` | `git archive d3b398fc` |
| `work-<s>-d109off2`, `d109on2`, `d109dlon2` | this round, final build `new2` = installed `local/lib` | `0b6fbb57` / `118bc640` | working tree before the flip |

**Arms:**
- samples `<s>` ∈ nuecc48 (48), ncpi0 (19), mcp1k (first 200);
- stage-A input `work-<s>-d102m`;
- all 267 × 4 arms rc=0;
- the three byte-gate arms use the geometric vertex (`SBND_NO_DL=1`, CLAUDE.md M4) under `setarch -R`;
- `d109dlon2` runs the production DL vertex (0 "DL vertex failed").

### 4.1 Compiled configuration

- **Knobs off:** `cmp_consumers` of pristine `d3b398fc` vs this round is **21/21 byte-identical** (SBND production PR job, bare PR job, lar 1-step imaging+clustering, standalone Q/L, PDHD, PDVD, sim checks, uBooNE) — `109_logs/cfg_gate_knobs_off.txt`.
- **Knobs on:** the arm's per-event compiled JSON (`pr_evt<ID>/.wct-cfg-evt<ID>.json`) carries `nu_provenance` on the three components, `fix_cluster_flags` and the `provenance` object.
- **Production flip:** sec 5.

### 4.2 Unit tests

- **Results:** `wcdoctest-clus` 406/406 (1 skipped, pre-existing), `wcdoctest-root` 8/8 (`109_logs/doctests.txt`).
- **New:** `doctest_nu_bundle_census.cxx` covers flash grouping (pairs, same TPC, `dt` edge, NaN, transitivity, order independence); the knob-default cases pin `nu_provenance`, `flash_pair_dt_us`, `fix_cluster_flags`, `provenance`, and the tagger writer's `nu_provenance`.

### 4.3 Byte gates (`scripts/d109_gate.py`)

| Output | base vs off2 (`gate_base_vs_off.txt`) | off2 vs on2 (`gate_off_vs_on.txt`) |
|---|---|---|
| `mabc-pr.zip` (member content) | 267/267 same | 267/267 same |
| `pctree-pr-evt<ID>.tar.gz` (member content) | 267/267 same | 267/267 same |
| `nusel-evt<ID>.tsv` | 267/267 same | 267/267 same |
| `calib-pr-evt<ID>.json` (161 events write one) | same | same |
| `tracking-pr.root` | 267 files, 8 trees, **214 565 branches identical** | every shared branch identical except the allowed T_cluster columns (files changed: `lm` 244, `beam_flash` 228, `fc` 140, `tgm` 71, `stm` 26); the 9 legacy `act_*` arrays keep their old entries as an unchanged prefix (153 of 161 files gain companions); added `T_bundle`, `T_flash` (267) and 44 branches |

So the knob-off path is byte-identical to the pre-round code. With the knobs on, nothing
the reconstruction produces moves: only the ROOT file gains content, and T_cluster's
always-0 flag columns take their real values.

### 4.4 Content checks (`scripts/d109_root_checks.py`, C1–C13): **0 failures** on `d109on2` and `d109dlon2`

| | nuecc48 | ncpi0 | mcp1k |
|---|---|---|---|
| T_tagger rows | 48 | 20 | 95 |
| rows with `vertex_moved_cluster = 1`, geometric / **DL** vertex | 0 / **3** | 0 / **10** | 0 / **4** |
| rows with `has_vertex = 0` (geometric / DL) | 0 / 0 | 0 / 0 | 4 / 3 |
| T_bundle reasons 0 / 1 / 2 / 3 | 47 / 1 / 2 / 4 | 20 / 0 / 0 / 3 | 91 / 4 / 40 / 37 |
| events with 0 / 1 / 2 in-window flash groups | 0 / 43 / 5 | 0 / 18 / 1 | 26 / 165 / 9 |
| two-row events: same flash group / different | 0 / 0 | 0 / 1 | 1 (second row has no vertex) / 0 |
| T_cluster rows; `tgm` / `stm` / `fc` / `beam_flash` = 1 | 4 392; 14 / 0 / 432 / 2 021 | 1 872; 7 / 0 / 172 / 765 | 11 676; 56 / 26 / 228 / 1 268 |
| role-1 roster entries also in the PR pass | 183 / 184 | 127 / 127 | 227 / 228 |

**Checks that pass by construction (0 failures):**
- the selected roster entry equals `sel_cluster_id`;
- `vertex_moved_cluster` equals `cluster_id ≠ sel_cluster_id`;
- exactly one `act_is_final`, on `cluster_id`;
- RSE equals Trun on all four trees;
- T_bundle rows with a candidate equal the T_tagger rows (gid, selected and final ids);
- T_cluster flags equal `act_tgm/stm/fc/lm` for every evaluated activity;
- `beam_flash` equals the window test;
- T_tagger flash fields equal T_flash;
- flash groups are different-TPC and within `dt`;
- each cluster appears once in the roster, and `act_in_pr` counts equal 1 + `n_companion`.

**Why the 106 mcp1k events have no T_tagger row** — all explained, none left unexplained:

| Explanation | events |
|---|---|
| no in-window main or demoted cluster | 39 |
| every in-window bundle's activities cosmic-vetoed (reason 2) | 36 |
| bundles rejected by the length floor (reason 3) | 31 |
| in-window clusters with no matched flash | 0 |

80 of the 106 still have an in-window flash (`T_flash.in_window`).

**`flash_tpc`** (opflash `apa`) equals `gid // 1000000` on 17 923 / 17 923 matched T_cluster rows.

### 4.5 Round 1

- **First build:** its arms (`d109off`/`d109on`/`d109dlon`) passed the same byte gates
  (`*_round1.txt`).
- **Check C3 failed on 6 DL-vertex rows** (`checks_d109dlon_round1_dup_roster.txt`): the moved-to
  cluster was a demoted main that was also a companion, listed twice.
- **Fix:** the final build lists each cluster once and adds `act_in_pr`; every gate above was re-run on it.

## 5. Production

**The flip (owner's word, 2026-09-14):** `sbnd/wct-pr-perevt.jsonnet` sets
`root_nu_record = root_cluster_flags = root_provenance = true`. The C++ and `pr()` defaults
stay off, so the lar 1-step chain, PDHD, PDVD and uBooNE do not move. The pre-flip arm is
`SBND_ROOT_OUTPUT=0`.

**Compiled-config proof** (`109_logs/cfg_gate_production_flip.txt`,
`cfg_production_flip_prod_prjob.diff`):
- 20 of 21 consumers are byte-identical to pristine `d3b398fc`.
- `prod_prjob.json` gains exactly these keys, nothing else:
  - `nu_provenance: true` on TaggerCheckNeutrino, UbooneTaggerOutputVisitor and SbndPrMagnifyTrackingVisitor;
  - `fix_cluster_flags: true`;
  - the five-key `provenance` object.

**Production reference (`prod_cfg_gate.py`): `ref/prod-2026-09-14`** (rev 2, sec 7.2).
- In rev 1, `ref/prod-2026-09-08` already drifted at **unmodified HEAD** in four artifacts
  (`109_logs/prod_cfg_gate_pristine_head_vs_0908.txt`). With the flip, the only added drift was
  `prod_prjob.json`, and the gate named exactly the nine keys above (`prod_cfg_gate_flip_vs_0908.txt`).
- Rev 2 names those four drifts key by key and attributes each to a commit. It then cuts
  `ref/prod-2026-09-14` at `c203b400` (PASS 21/21) and removes `prod-2026-09-08`, on the
  owner's word.

**Installed library:** the production job loads `local/lib`. Its
`libWireCellClus.so`/`libWireCellRoot.so` md5 equals the graded pin `new2`
(`109_logs/libsnap.md5`). The smoke below is the stronger proof: it ran from `local/lib`
with no pin, and its files carry the new trees and a clean `toolkit_git`. By contrast,
`strings -a | grep -c '^nu_provenance$'` gave a false 0 on the root library, because the
literal is merged into a longer string there.

**Production smoke** (toolkit `c203b400` committed and installed; default runner, no
`SBND_ROOT_OUTPUT`, no pin, DL vertex ON; `109_logs/checks_d109prod_smoke.txt`,
`smoke_d109prod.txt`):
- **Run:** 6 events rc=0, 0 DL fallbacks, content checks C1–C13 0 failures.
- **Provenance:** every file carries `toolkit_git = c203b4002c27…` (clean) and the same
  `op_config_sha256 = 7809d5ec…`.

| Event | What the file records |
|---|---|
| nuecc48 137238 | `sel_cluster_id` 7, `cluster_id` 144, `vertex_moved_cluster` 1 — one of doc 108's moved rows, now explicit |
| nuecc48 269774 | selected 13, written 87, moved |
| nuecc48 10550 | row on gid 1000002 (TPC 1); a second in-window bundle, gid 7 (TPC 0), rejected by the length floor (reason 3), and both share `flash_group` 7: one flash seen by both TPCs |
| mcp1k 48301, 49445 | no T_tagger; one bundle each, reason 3 (length floor) |
| mcp1k 48565 | no T_tagger and no bundle: nothing in the beam window (`Trun.nu_n_in_window_main = 0`) |

## 6. Not done, and why

- **Truth file** (doc 108 group 3): deferred by the owner.
- **Selection-time vertex:** not recorded. The selection runs before any vertex exists; the
  per-cluster vertex `map_cluster_main_vertices[main_cluster]` (TaggerCheckNeutrino.cxx:3147)
  is an intermediate, pre-overall vertex, and a branch named "selection vertex" would mislead.
- **Per-blob G4 track ids through the PR splits** (doc 108 sec 2.5 c): not needed without
  per-candidate truth matching.
- **Group / `multi_event` path:** gated in rev 2 (sec 7.1).
- **The three switches move together** (owner, rev 2).
  - **Production:** `wct-pr-perevt.jsonnet` defaults all three to true, and the runner's
    `SBND_ROOT_OUTPUT` sets all three.
  - **Only by hand:** setting one alone takes an explicit TLA, and that is not a supported
    configuration. `root_nu_record` on with `root_cluster_flags` off gives real `act_*`
    verdicts beside T_cluster flag columns that still read 0, and `beam_flash` needs the
    census from `root_nu_record` (-1 without it).
  - **No code coupling added:** the compiled production job already carries all three.
- **Unchanged here:** the one-file-name-per-process limitation of the two writers (doc 108
  sec 1.2), and the 80 ns flash-t0 merge bookkeeping issue (doc pr/94 sec 9.8).

## 7. Revision 2 (2026-09-14): group mode, the three switches, a new production reference

**Owner ask (2026-09-14):** "give a try on the group mode, and turn the three settings on
together, create a new production configure reference would be nice. You can then remove the
stale produciton config reference."

**Rev 2 changes no toolkit code.** It adds arms, gates, the reference generation and this section.

### 7.1 Group mode: `tracking-pr.root` is byte-identical to per-event

**Arms.** Each group arm uses the per-event arm's pin (`new2`), cfg tree
(`~/tmp/d109-cfg/new/cfg`) and env, plus `PR_GROUP_SIZE=16` (the runner's `multi_event` +
`rse_map` path):
- `d109grp` pairs with `d109on2` (geometric vertex);
- `d109dlgrp` pairs with `d109dlon2` (DL vertex).

**Allowed difference.** The gate allows `Trun.toolkit_git` and `Trun.wcp_git`, and only
`toolkit_git` differs: the toolkit moved from `d3b398fc` plus uncommitted edits to the
committed `c203b400` between the arms, with the same compiled job (same `op_config_sha256`).

| sample | events | `mabc-pr.zip` / `pctree-pr` / nusel | `tracking-pr.root`: files / branches identical | `calib-pr-evt*.json` |
|---|---|---|---|---|
| nuecc48 | 48 | 48 / 48 / 48 identical | 48 / 66,768 | 45 differ, 3 identical |
| ncpi0 | 19 | 19 / 19 / 19 identical | 19 / 26,429 | 17 differ, 2 identical |
| mcp1k (first 200) | 200 | 200 / 200 / 200 identical | 200 / 139,858 | 79 differ, 121 identical, 106 not written (no candidate) |

- **Every tree is identical**, including the new `T_bundle`, `T_flash`, the `T_tagger`/`T_kine`
  run/subrun/event branches and the `Trun` census counters.
- **This is the test that could have failed.** nuecc48 spans 12 runs, so a group-leader-run
  bug in the new RSE branches would have shown on 45 of 48 files. The per-file content
  check C5 cannot catch it, because it compares each tree with the same file's `Trun`.
- **Content checks C1–C13 on `d109grp`** (267 events): **0 failures**
  (`109_logs/r2/checks_d109grp.txt`).
- **DL vertex** (`d109dlgrp` vs `d109dlon2`, the production vertex mode):
  - **Byte gates:** the same result. `mabc-pr.zip`, `pctree-pr` and nusel are identical, and
    so are all 66,768 / 26,429 / 139,858 ROOT branches (`gate_dlon2_vs_dlgrp_<s>.txt`).
  - **Content checks:** C1–C13 have 0 failures (`checks_d109dlgrp.txt`). The DL-moved rows
    are 3 / 10 / 4, as per-event, and there are 0 `DL vertex failed` lines.
  - **So the new `vertex_moved_cluster` and `sel_cluster_id` survive group mode unchanged.**

**The `calib-pr-evt*.json` differences predate doc 109 and are outside `tracking-pr.root`.**
That file is `PrDisplayDump`, the `pr_display` stage. It has two group-mode defects:
1. **Run number.** `meta.runNo`/`meta.subRunNo` come from the configure-time `m_runNo`/`m_subRunNo`
   (`clus/src/PrDisplayDump.cxx:317-318`), which in group mode is the group leader's run.
   Only `eventNo` is corrected per event (`:254`). Example, nuecc48 30504: `runNo` 18342 → 18255.
2. **Shower ids.** `showers[].shower_id` comes from a process-wide
   `static std::atomic<int> s_shower_id_counter` (`clus/src/PRShower.cxx:13, :185`). The
   ids keep counting across the events of one process: 30504 has 0, 1, 2, 4, 6 per-event and
   34, 35, 36, 38, 40 in group mode.

A third calib difference appears **only with the DL vertex**, and it is not a defect.
- **What it is:** `vertex_scoreboard.dual_chain.off_ms` is a wall-clock duration,
  `MS(Clock::now() - t_total)` (`clus/src/TaggerCheckNeutrino.cxx:4339, :4404`). It differs
  between any two runs; nuecc48 30504 took 6530 ms per-event and 2778 ms in the group.
- **Everything else identical:** every other `dual_chain` field (vertex, distance, mode,
  agreement) is identical.

**Every differing file falls into these classes, with no other key**
(`109_logs/r2/calib_classes_{on2_vs_grp,dlon2_vs_dlgrp}.txt`; `calib_classes.py` names any
other key as OTHER, and there are 0 in both). Geometric vertex:

| sample | run + shower ids | shower ids only | run only | identical |
|---|---|---|---|---|
| nuecc48 | 28 | 17 | 0 | 3 |
| ncpi0 | 13 | 4 | 0 | 2 |
| mcp1k | 0 | 78 | 1 | 15 (106 not written) |

**Control arm: the defects predate doc 109.** `d109basegrp` is the **base** binary `d3b398fc`
with the pristine cfg, in group mode on nuecc48, gated against `d109base`:
- **Same classes, same counts:** 28 / 17 / 0 / 3, and 0 other keys
  (`calib_classes_base_vs_basegrp.txt`).
- **Everything else identical there too:** `mabc-pr.zip`, `pctree-pr`, nusel and all 63,120
  ROOT branches (`gate_base_vs_basegrp_nuecc48.txt`).

So both defects predate doc 109, and group mode already matched per-event on everything
else before it. Neither defect is fixed here: each changes an output, so a fix would be its
own default-OFF round. **Fixed in doc 110** (`110_group-mode-calib-dump-fixes.md`, toolkit
`67937f45`): knobs `rse_from_ensemble` / `reset_shower_ids_per_event`, ON for SBND group mode.

### 7.2 A new production reference: `ref/prod-2026-09-14`

**The new generation.** `scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-14` → **PASS
21/21** at toolkit `c203b400` (`109_logs/r2/gate_head_vs_0914.txt`). The generation record,
with every key named, is `ref/prod-2026-09-14/README.md`.

**How rev 1's unexplained drift was named.** `prod-2026-09-08` keeps only hashes for four of
its artifacts, so they were recompiled instead:
- **The old point reproduces exactly.** `git archive eacacafe cfg`, the commit 09-08 was cut
  at, passes 09-08 **21/21** (`gate_eacacafe_vs_0908.txt`).
- **Diff and attribution:** that compile was diffed key by key against `c203b400`
  (`drift_eacacafe_to_c203b400.txt`, `drift_keys.py`), and each key was traced with
  `git log -S/-G`.

| artifact(s) | keys | origin |
|---|---|---|
| `prod_prjob.json` | the nine doc-109 keys (sec 5) | this doc, on the owner's word |
| `prod.standalone`, `sbnd_clus.json`, `sbnd_ql.json` | `bee_points_sets[1].opflash_time = true` on `MultiAlgBlobClustering` | `b31a0db0` (2026-08-06), via the master merge `98140fee` (2026-09-10); a Bee-output column |
| `sbnd_simcheck.json` | `roi_mad_rms`, `r_break_roi_loop_planes`, `troi_col_th_factor` 5 → 3, `troi_ind_th_factor` 3 → 1.8 on both `OmnibusSigProc` | `b8086bd6` + `06a02ccb` (2026-08-13), via `98140fee`; SP from raw ADC only (`sbnd_img.json` did not move) |

The last two rows are inherited: no sbnd_xin round validated them. The README says so rather
than presenting them as part of this flip.

**The removed generation.** `ref/prod-2026-09-08` is removed on the owner's word. Its
citations are updated so none dangles: the d102m/d102mpr lines in
`scripts/retire/PROTECTED.txt`, doc 102's repro block, and `work-tags.md`. The record survives
in git at wcp `70a49e8c`. **`work-*-d102m`/`d102mpr` stay at `eacacafe`, not at
`prod-2026-09-14`.**

### 7.3 The three switches

- **Already coupled.** They move together in production and in the runner (sec 6), and the
  compiled production job carries all three.
- **No code added.** A coupling assertion in `pr()` would guard a configuration nothing sets.

### 7.4 Repro (rev 2)

```bash
cd wcp-porting-img/sbnd/sbnd_xin
scripts/d109_arms.sh d109grp   ~/tmp/d109-libsnap/new2 SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 PR_JOBS=14 PR_GROUP_SIZE=16 PR_CFG_TREE=~/tmp/d109-cfg/new/cfg
scripts/d109_arms.sh d109dlgrp ~/tmp/d109-libsnap/new2 SBND_ROOT_OUTPUT=1 PR_JOBS=16 PR_GROUP_SIZE=16 PR_CFG_TREE=~/tmp/d109-cfg/new/cfg
LD_LIBRARY_PATH=~/tmp/d109-libsnap/base SBND_NO_DL=1 PR_CFG_TREE=~/tmp/d109-cfg/pristine/cfg PR_GROUP_SIZE=16 PR_JOBS=3 \
  PR_EXTRA_STAGES=pr_display setarch x86_64 -R ./run_pr_chain_batch.sh work-nuecc48-d102m work-nuecc48-d109basegrp data
for s in nuecc48 ncpi0 mcp1k; do
  python3 scripts/d109_gate.py d109on2   d109grp   --samples $s --allow Trun.toolkit_git Trun.wcp_git > docs/109_logs/r2/gate_on2_vs_grp_$s.txt
  python3 scripts/d109_gate.py d109dlon2 d109dlgrp --samples $s --allow Trun.toolkit_git Trun.wcp_git > docs/109_logs/r2/gate_dlon2_vs_dlgrp_$s.txt
done
python3 scripts/d109_gate.py d109base d109basegrp --samples nuecc48          > docs/109_logs/r2/gate_base_vs_basegrp_nuecc48.txt
python3 docs/109_logs/r2/calib_classes.py d109on2   d109grp                  > docs/109_logs/r2/calib_classes_on2_vs_grp.txt
python3 docs/109_logs/r2/calib_classes.py d109dlon2 d109dlgrp                > docs/109_logs/r2/calib_classes_dlon2_vs_dlgrp.txt
python3 docs/109_logs/r2/calib_classes.py d109base  d109basegrp nuecc48      > docs/109_logs/r2/calib_classes_base_vs_basegrp.txt
python3 scripts/d109_root_checks.py d109grp   > docs/109_logs/r2/checks_d109grp.txt
python3 scripts/d109_root_checks.py d109dlgrp > docs/109_logs/r2/checks_d109dlgrp.txt
# the reference: see ref/prod-2026-09-14/README.md "Reproduce"
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-14                       # PASS 21/21
```

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
- **Rev 3 (2026-09-17, sec 8): the five defects a colleague's first look reported.**
  - **Toolkit:** `apply-pointcloud` `3ecb110d` (code, tests, knobs default OFF) and
    `12798c4f` (SBND production flip). Reference `ref/prod-2026-09-17`, PASS 21/21.
  - The scanned files **predate doc 109**; two of the five cases are already answerable at
    HEAD, two were not.
  - **`T_rec_charge` now joins to the candidate that owns its points** — the old
    `cluster_id` was a `Flags::main_cluster` scan that returned `-1` on a demoted-main
    candidate and the pre-swap cluster on a vertex-moved row. `nu_index` and
    `point_cluster_id` added; `T_rec_charge`/`T_proj_data` no longer vanish.
    Knob `rec_charge_provenance` / jsonnet `root_point_ids`, default OFF, **SBND production ON**.
  - **`nu_dedup_flash_group` built default OFF and measured, NOT flipped** (sec 8.7).
  - Vertex-less rows are **marked, not suppressed** (owner's call, sec 8.4).

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

```bash
# ---- rev 3 (sec 8) ----------------------------------------------------------
cd wcp-porting-img/sbnd/sbnd_xin
# 0. pins (docs/109_logs/r3/libsnap.md5).  base = toolkit d2777286 unmodified;
#    new = rev 3 first build; new2 = new + the Group B std::move fix (sec 8.7)
#    and == the installed local/lib.  ~/tmp/d109r3-libsnap/{base,new,new2}
#    cfg trees: ~/tmp/d109r3-cfg/pristine/cfg (git archive d2777286), .../new/cfg
# 1. compiled-config gate, knobs off (21 consumers)
scripts/cfg/compile_consumers.sh ~/tmp/d109r3-cfg/pristine/cfg ~/tmp/d109r3-cfg/A
scripts/cfg/compile_consumers.sh <toolkit>/cfg                 ~/tmp/d109r3-cfg/B
scripts/cfg/cmp_consumers.sh ~/tmp/d109r3-cfg/A ~/tmp/d109r3-cfg/B   # 21/21 identical
# 2. unit tests
<toolkit>/build/clus/wcdoctest-clus ; <toolkit>/build/root/wcdoctest-root
# 3. Group A arms on the doc 109 manifest (nuecc48 48 + ncpi0 19 + first 200 mcp1k)
scripts/d109_arms.sh d109r3h   ~/tmp/d109r3-libsnap/base SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 PR_JOBS=14 PR_CFG_TREE=~/tmp/d109r3-cfg/pristine/cfg
scripts/d109_arms.sh d109r3off ~/tmp/d109r3-libsnap/new  SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 SBND_ROOT_POINT_IDS=0 SBND_NU_DEDUP_FLASH_GROUP=0 PR_JOBS=14 PR_CFG_TREE=~/tmp/d109r3-cfg/new/cfg
scripts/d109_arms.sh d109r3on  ~/tmp/d109r3-libsnap/new  SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 SBND_ROOT_POINT_IDS=1 SBND_NU_DEDUP_FLASH_GROUP=0 PR_JOBS=14 PR_CFG_TREE=~/tmp/d109r3-cfg/new/cfg
# 4. Group A gates.  Trun.cfg_tree: the two arms read DIFFERENT pinned cfg paths
#    by construction (sec 8.8.3).  Trun.op_config_sha256: the knob changes the
#    compiled job, so the operating-point hash SHOULD move (sec 8.8.4).
python3 scripts/d109_gate.py d109r3h   d109r3off --allow Trun.cfg_tree                                 > docs/109_logs/r3/gate_head_vs_off.txt
python3 scripts/d109_gate.py d109r3off d109r3on  --allow T_rec_charge.cluster_id Trun.op_config_sha256 > docs/109_logs/r3/gate_off_vs_on.txt
python3 scripts/d109_root_checks.py d109r3on                                  > docs/109_logs/r3/checks_d109r3on.txt
# 4b. case 1 (the vertex-moved row) needs the PRODUCTION vertex: the geometric
#     arms above have zero moved rows, so they cannot grade it.  Content-only.
scripts/d109r3_dl_arm.sh d109r3dlon ~/tmp/d109r3-libsnap/new2
python3 scripts/d109_root_checks.py d109r3dlon --samples nuecc48              > docs/109_logs/r3/checks_d109r3dlon.txt
# 4c. the pin split: new2 must be new on the Group A path
scripts/d109_arms.sh d109r3on2 ... (nuecc48 only, same env as d109r3on, pin new2)
python3 scripts/d109_gate.py d109r3on d109r3on2 --samples nuecc48             > docs/109_logs/r3/gate_on_vs_on2_nuecc48.txt
# 5. the nu_dedup_flash_group census: the 25 multi-candidate events of the 3067
scripts/d109r3_dedup_arm.sh d109r3ddoff2 ~/tmp/d109r3-libsnap/new2 SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 SBND_ROOT_POINT_IDS=1 SBND_NU_DEDUP_FLASH_GROUP=0 PR_JOBS=6
scripts/d109r3_dedup_arm.sh d109r3ddon2  ~/tmp/d109r3-libsnap/new2 SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 SBND_ROOT_POINT_IDS=1 SBND_NU_DEDUP_FLASH_GROUP=1 PR_JOBS=6
python3 scripts/d109r3_dedup_census.py d109r3ddoff2 d109r3ddon2               > docs/109_logs/r3/dedup_census.txt
python3 scripts/d109_gate.py d109r3ddoff d109r3ddoff2 --samples ncpi0 mcp1k mcp2k > docs/109_logs/r3/gate_new_vs_new2_dedupoff.txt
python3 scripts/d109_root_checks.py d109r3ddon2 --samples ncpi0 mcp1k mcp2k   > docs/109_logs/r3/checks_d109r3ddon2.txt
# 6. the production flip: compiled configs and the new reference
scripts/cfg/compile_consumers.sh <toolkit>/cfg ~/tmp/d109r3-cfg/C
scripts/cfg/cmp_consumers.sh ~/tmp/d109r3-cfg/A ~/tmp/d109r3-cfg/C   # 20/21; prod_prjob.json only
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17               # PASS 21/21
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

---

## 8. Revision 3 (2026-09-17): the five defects a first look at `pr_tracking*.root` found

**Owner ask (2026-09-17):** a colleague scanned the SBND `pr_tracking*.root` files and
reported five failure classes in `presentations/20260917_MV_firstlook.pdf` (slides 19–23).
"Carefully analyse the issues, find the examples in my 3000 data events, make improvements
on the toolkit processing chain to fix them, update the md file, commit and push. For cases
with no good example, do your best and note them."

**The first finding is about the files, not the code.** The scanned sample
(`sbnd_xin/sbnd_mc_data` → `/nfs/data/1/xqian/sbnd_data/run/tracking-pr`, 13 216 files
written 2026-09-09/10) **predates doc 109**: its `Trun` has five branches, and there is no
`T_bundle`, `T_flash`, `sel_cluster_id`, `has_vertex` or `flash_group`. Two of the five cases
are already answerable at HEAD and only needed to be named. Two are **not**, and a doc 109
production file still on disk proves it. One is a physics question this round measures but
does not settle.

**Status:**
- **Group A (output only) implemented, gated, SBND production ON.** `rec_charge_provenance`
  / jsonnet `root_point_ids`.
- **Group B (`nu_dedup_flash_group`) built default OFF and measured. NOT flipped** — it
  removes `T_tagger` rows, so the flip is a separate owner decision (sec 8.7).
- **Vertex-less rows are marked, never suppressed** (owner's call, sec 8.4).

### 8.1 The five cases, measured

Every case was counted twice: on the colleague's 13 216 files, and on the local 3067-event
sample (`work-{nuecc48,ncpi0,mcp1k,mcp2k}-d102mpr`, whose stage-A inputs `…-d102m` make
stage B re-runnable). Rates are per T_tagger row unless stated.

| Case (slide) | colleague's 13 216 | local 3 067 | verdict at HEAD |
|---|---|---|---|
| 1 — main clusterID changes | 488 of 6 236 rows | 122 of 1 460 rows | **labelled** by doc 109 (`sel_cluster_id`, `vertex_moved_cluster`), but `T_rec_charge` still does not join — **fixed here** |
| 2 — empty/fake candidate | 160 rows | 62 rows | **labelled** by doc 109 `has_vertex == 0`; markers sharpened here, rows not suppressed |
| 3 — `T_rec_charge` cluster_id −1 | 344 files all −1, 12 mixed | 89 files all −1, 0 mixed | **broken** — **fixed here** |
| 4 — two candidates, one fake | 98 two-row events | 25 two-row events | cases 2 + 3 together |
| 5 — same ν on both sides | 22 events | 12 events | **labelled** by `flash_group`; dedup built OFF and measured (8.7) |

Supporting counts, local sample: `T_proj_data` absent in 1 675 files; a candidate with an
empty `T_rec_charge` in 43; `nue_score` at its −15 default on 1 260 of 1 460 rows; the
selected activity is a *demoted* main on 57 rows.

### 8.2 Case 1 and 3 are one bug, and doc 109's own output shows it

`T_tagger.cluster_id` and `T_kine.cluster_id` come from the **main-cluster pointer**
(`TaggerCheckNeutrino.cxx`, `tagger_info.cluster_id = main_cluster->get_cluster_id()`).
`T_rec_charge.cluster_id` did not: it was `reco_mother_cluster_id`, one event-level id
stamped on every row and found by **scanning the candidate's PR graph for
`Flags::main_cluster`** (`SbndPrMagnifyTrackingVisitor.cxx`, `mother_cluster_id = -1` then
two scans). The scan cannot reproduce the pointer in three measured situations, and the
local 97 `-1` rows (in 89 files) split cleanly between them:

| route | local rows | what happened |
|---|---|---|
| A — the selected activity is a **demoted main** | 57 | `ClusteringUnmergeBundle` clears `Flags::main_cluster` on every split-off part and deliberately never restores it; `TaggerCheckNeutrino`'s own restore guards have destructed by the time the writer runs. No cluster in the graph carries the flag ⇒ `-1`. `T_bundle.reason == 1` names exactly this case, and the log line `is a demoted main` appears in exactly 57 of the 3067 event logs. |
| B — the flagged main contributes **no segment and no vertex** to that candidate's graph | 16 | the rows that exist came from companions only ⇒ `-1`. |
| swap + selected not flagged | 24 | both at once. |

And when the scan *did* find a flag, on a **vertex-moved** row it found the **pre-swap**
cluster — so the points carried an id that is not the row's `cluster_id`. Across the
colleague's sample the 488 swap rows carry their points under the *selected* id (341) or
under `-1` (147): **never** under the id `T_tagger` reports.

**The measurement that settles it** — doc 109's own production smoke files, knobs ON, still
on disk:

```
work-nuecc48-d109prod/pr_evt137238   T_tagger.cluster_id=144  sel_cluster_id=7
                                     vertex_moved_cluster=1   has_vertex=1
                                     T_rec_charge.cluster_id = [7]     <-- 144 absent
work-nuecc48-d109prod/pr_evt269774   cluster_id=87  sel=13    T_rec_charge = [13]
work-mcp1k-d146sv25/pr_evt313847     cluster_id=19            T_rec_charge = [-1]
```

So on exactly the "moved rows" doc 109 added `vertex_moved_cluster` for, the 3-D points were
not joinable to the candidate. Doc 109's checks C1–C13 never looked at `T_rec_charge`.

### 8.3 What rev 3 changes in the file (`root_point_ids`)

One switch, `rec_charge_provenance` (C++) / `root_point_ids` (jsonnet `pr()`), on
`SbndPrMagnifyTrackingVisitor`. Default OFF; key omitted from the compiled config when off.

| change | why |
|---|---|
| `T_rec_charge.cluster_id` now comes from the candidate's own `TrackFitting` (`TaggerInfo::cluster_id`) | the same pointer-derived value `T_tagger`/`T_kine` carry, so the join holds on every row. Subsumes routes A and B **and** the swap. The flag scan is kept as the knob-off path and as the fallback when the info is unset. |
| **`nu_index`** added to `T_rec_charge` | the colleague's explicit "no nu_index — one tree for the whole event". Points are now attributable per candidate rather than inferred. |
| **`point_cluster_id`** added | the row's own cluster, under an honest name. It is the value `ndf` has been carrying; `ndf` is left exactly as it is because `wire-cell-sbnd-magnify-tracking-convert` blocks tracks on `std::round(ndf)`. |
| `T_rec_charge` and `T_proj_data` are booked **even when empty** | the tree set stopped varying for a non-semantic reason (see 8.5). |

**Deliberately not changed.** `real_cluster_id` and `sub_cluster_id` are bound to the *same
address* and are therefore always identical — a legacy-format compatibility decision with a
comment saying so (M15). A correctly named branch was added instead of redefining them.

### 8.4 Cases 2 and 4: the vertex-less row is marked, not suppressed

The whole placeholder signature — `kine_reco_Enu` 0, vertex (0,0,0), `neutrino_type` 0,
`nue_score` −15.000, `numu_score` −1.942 — has **one** cause: `final_main_vertex == nullptr`.
Every tagger, the kinematics fill and the `neutrino_type` bitmask sit inside
`if (final_main_vertex)`, and the row is published unconditionally afterwards. `−15` is a
literal default; `−1.942` is the numu forest's output on an all-defaults feature vector, i.e.
a real evaluation carrying no information.

**The owner's call for this round is markers only.** What the file already answers, and how:

| question | read | note |
|---|---|---|
| is this row a vertex-less placeholder? | `T_tagger.has_vertex == 0` | added by doc 109; `(0,0,0)` is a real point on SBND's cathode, so never test the vertex |
| did the nue BDT actually run? | `T_tagger.br_filled == 1` | **already in the file** — no code change was needed, only this note. `br_filled` is set deep inside `nue_tagger`, after a shower is found; a vertex with no shower returns earlier. That is why 1 260 of 1 460 local rows read −15 and it is **not** a defect |
| how many 3-D points did this candidate get? | `sum(T_rec_charge.nu_index == i)` | new in rev 3; a placeholder reads 0 |

No redundant branch was added for the last one: `nu_index` makes it a one-line derivation.

*Why the tiny cluster was selected at all* is a separate, selection-level question — the
length floor (`nu_per_bundle_min_length = 15 cm`) exempts the legacy event-wide winner, and
that exemption fires in 462 of the 1000 mcp1k event logs. It is **not** touched here.

### 8.5 The varying tree count ("7 trees" vs "8 trees")

The colleague's checker reported 7 or 8 trees depending on the event. Local census, before:

| trees | files | meaning |
|---|---|---|
| `T_bad_ch T_cluster T_proj Trun` (4) | 1 632 | no neutrino candidate at all |
| all 8 | 1 392 | a candidate with fit points |
| 7 (no `T_proj_data`) | 43 | a candidate whose `fitted_charge_2d` was empty |

(That census is of the pre-doc-109 local arm, which is why `T_bundle`/`T_flash` do not
appear in it.)

`T_proj_data` and `T_rec_charge` were skipped by early returns rather than written empty.
With `root_point_ids` on — and doc 109's `root_nu_record`, which SBND production also runs —
both are always booked, so the file has **10 trees when a candidate was written and 8 when
none was**:

```
always   T_bad_ch  Trun  T_cluster  T_proj  T_proj_data  T_rec_charge  T_bundle  T_flash
+ when a candidate exists            T_tagger  T_kine
```

The one remaining difference is exactly "`T_tagger`/`T_kine` exist ⟺ a candidate exists",
which `T_bundle.reason` already explains bundle by bundle. `T_proj` stays the deliberately
empty compatibility tree it has always been.

**Reader-side consequence.** A script that used `"T_rec_charge" in f` or `"T_proj_data" in f`
as a proxy for "this event has a neutrino candidate" now sees the tree and must read
`T_tagger` instead (or test `num_entries`). Two consumers were checked and are unaffected
because they also require `T_kine` (`scripts/mcs80_pull.py`,
`mcs_upstream/dumper/harvest_sbnd_clouds.py`); one past one-off census,
`scripts/analysis/pr40/pr40r7_census.py`, uses `T_rec_charge` presence alone and would now
reach an empty array on a no-candidate event. It is not re-run by anything and is left
untouched here. `wire-cell-sbnd-magnify-tracking-convert` reads the tree through a `TChain`
and loops on `GetEntries()`, so an empty tree is the same no-op as an absent one.

### 8.6 Reading it after rev 3

```python
import uproot, numpy as np
f = uproot.open("pr_evt<ID>/tracking-pr.root")
rc = f["T_rec_charge"].arrays(["nu_index", "cluster_id", "point_cluster_id", "x", "y", "z", "q"],
                              library="np")
t  = f["T_tagger"].arrays(["nu_index", "cluster_id", "sel_cluster_id", "vertex_moved_cluster",
                           "has_vertex", "br_filled", "flash_group"], library="np")
for i in t["nu_index"]:                      # the points of candidate i
    pts = rc["nu_index"] == i
    assert set(rc["cluster_id"][pts]) <= {t["cluster_id"][i]}
    print(i, "points:", pts.sum(), "from clusters", sorted(set(rc["point_cluster_id"][pts])))
```

| Question | Read |
|---|---|
| which candidate owns these 3-D points? | `T_rec_charge.nu_index` |
| which cluster is the row's PR result on? | `T_rec_charge.cluster_id` = `T_tagger.cluster_id`, now always |
| which cluster did *this point* come from? | `T_rec_charge.point_cluster_id` |
| did the overall vertex move the main? | `T_tagger.vertex_moved_cluster`, with `sel_cluster_id` |
| is the row a vertex-less placeholder? | `T_tagger.has_vertex == 0` |
| did the nue BDT run, or is −15 its default? | `T_tagger.br_filled` |
| are two rows one physical flash? | same `T_tagger.flash_group` |

### 8.7 Case 5: `nu_dedup_flash_group`, built and measured, not flipped

One physical beam flash is seen by **both** SBND drift volumes and arrives as two `opflash`
gids a few ns apart — e.g. r472 s36 e40, gid 5 on TPC 0 at 1.577 µs and gid 1000006 on TPC 1
at 1.583 µs. Bundles are keyed on the **raw gid**, so each side builds its own `NuCandidate`,
each gets a full PR pass, and the event gets two neutrino rows for one flash. In that event
the truth has a single interaction and only one of the two rows sits on it.

`group_flashes()` has known which gids are one flash since doc 109 — the answer was written
to `T_bundle`/`T_flash`/`T_tagger.flash_group` and **never read back into a decision**
(`TaggerCheckNeutrino.cxx`: "Observation only: nothing below reads the census back into a
decision"). `nu_dedup_flash_group` reads it: after the longest-first ordering, candidates
sharing a `flash_group` collapse to the first, which is the longest — the same candidate the
row ordering already puts in slot 0. The dropped bundle keeps its `T_bundle` row with the new
reason code **6 `kDedupFlashGroup`**, so the event still explains itself, and `nu_index` is
dense over the survivors.

The rule itself is `PR::dedup_flash_groups()` in `NuBundleCensus.cxx`, unit-tested in
`clus/test/doctest_nu_bundle_census.cxx` (the r472 pair, disjoint groups, a partially known
map, a transitive three-way group, and the empty/one-candidate no-ops).

**This knob moves the selection** — it removes `T_tagger` rows — so it ships **default OFF**
and is **not flipped**. The doc 109 manifest cannot measure it (it holds almost no
multi-candidate events), so the census runs on the **25 events of the 3067-event sample that
have two or more candidates** — the only events the knob can touch
(`scripts/d109r3_dedup_arm.sh`, `scripts/d109r3_dedup_census.py`).

**A bug the census caught, worth recording.** The first build of this knob moved every
candidate into a `kept` vector but reassigned `candidates = std::move(kept)` only
`if (!dropped.empty())`. On an event where the knob found nothing to drop, `candidates` was
therefore left holding **moved-from shells**: each candidate kept its `main` pointer but lost
its `others` (the companions) and `acts`. The knob-off path and the byte gates were untouched
— the whole block is inside `if (m_nu_dedup_flash_group …)` — so nothing in Group A could
have caught it. What caught it was `d109r3_dedup_census.py`'s **"changed surviving rows"**
check on ncpi0 18625, an event whose two rows are in *different* flash groups and so should
have been bit-identical: its reco Enu moved 1448.6 → 693.5 MeV and 174.1 → 105.0 MeV. The
assignment is now unconditional, and the census re-ran on the fixed build.

This is why the census compares *surviving* rows as well as counting dropped ones: a dedup
knob that silently perturbs the events it does **not** dedup would otherwise look like a
clean result.

#### 8.7.1 What the knob does, measured (`d109r3ddoff2` vs `d109r3ddon2`, 25 events)

| | |
|---|---|
| events compared | 25 (every multi-candidate event of the 3 067) |
| T_tagger rows, knob off → on | **50 → 38** |
| events with a dropped row | **12** |
| rows dropped | 12 |
| **dropped rows with NO vertex (placeholder)** | **12** |
| **dropped rows with a vertex** | **0** |
| rows added, or surviving rows changed | **0 / 0** |

**Every row the knob removes is a vertex-less placeholder** — `has_vertex 0`, reco Enu 0,
vertex (0,0,0), `numu_score` −1.942 — i.e. the *same* rows case 2/4 is about. On this sample
"longest wins" and "has the vertex" never diverge: the real candidate is always the longer
one. Two examples:

```
mcp1k 174422  dropped gid 1000003 (tpc 1, group 5, t0 0.6829 us) sel cluster 8   has_vertex 0
              kept    gid 5       (tpc 0, group 5, t0 0.6787 us) sel cluster 18, 92.8 cm
mcp2k 90751   dropped gid 1       (tpc 0, group 1, t0 0.2404 us) sel cluster 5   has_vertex 0
```

The 13 events that keep both rows are the ones whose two candidates are in **different**
flash groups — genuinely distinct interactions — and they are untouched (ncpi0 18625 among
them). Every dropped bundle keeps its `T_bundle` row with `reason 6` and `nu_index -1`: 12
such rows, one per drop. Content checks C1–C17 on the dedup-on arm: **0 failures**.

**Caveat for the flip decision.** 12 events is a small sample and all of them happen to be
the easy case. The knob is a *ranking* rule ("keep the longest"), not a *quality* rule ("keep
the one with a vertex"); nothing measured here says what it would do on an event where the
shorter candidate is the real neutrino. That is the question a flip has to answer, and it is
left open deliberately.

### 8.8 Gates

**Pins.** Two builds were graded, because a bug in the *Group B* knob (8.7) was found after
the Group A arms had already run:

| pin | libraries | `libWireCellClus.so` / `libWireCellRoot.so` md5 |
|---|---|---|
| `base` | toolkit `d2777286` unmodified (HEAD before this round) | `6d97984e` / `88f0822e` |
| `new` | rev 3 code, first build | `21bcd462` / `d5d378a8` |
| `new2` | `new` + the Group B `std::move` fix | `e5ed8064` / `d5d378a8` (root unchanged) |

`new` → `new2` touches only lines inside `if (m_nu_dedup_flash_group …)`, and that is
**measured, not argued**: with the dedup knob off the two pins are byte-identical (8.8.4).

Arms, all under `setarch x86_64 -R` with `SBND_NO_DL=1` (M4) and `PR_CFG_TREE` pinned:

| label | pin | env | events |
|---|---|---|---|
| `d109r3h` | `base` | `SBND_ROOT_OUTPUT=1`, pristine cfg | 267 |
| `d109r3off` | `new` | + `SBND_ROOT_POINT_IDS=0 SBND_NU_DEDUP_FLASH_GROUP=0` | 267 |
| `d109r3on` | `new` | + `SBND_ROOT_POINT_IDS=1 SBND_NU_DEDUP_FLASH_GROUP=0` | 267 |
| `d109r3on2` | `new2` | as `d109r3on` | 48 (nuecc48) |
| `d109r3ddoff2` / `d109r3ddon2` | `new2` | Group A on, dedup off / on | 25 each |
| `d109r3dlon` | `new2` | Group A on, **DL vertex ON** (no `SBND_NO_DL`) | 48 (nuecc48) |

All arms rc=0 on every event, and `d109r3dlon` has **0** `DL vertex failed` lines — a silent
fallback to the geometric vertex would make 8.8.7 meaningless, so it is counted, not assumed.
Every log quoted below is under `docs/109_logs/r3/`.

#### 8.8.1 Compiled configuration

- **Knobs off:** `cmp_consumers.sh` of pristine `d2777286` against this round is
  **21/21 byte-identical** (SBND production PR job, bare PR job, lar 1-step, standalone Q/L,
  PDHD, PDVD, sim checks, uBooNE).
- **Compiled-config proof, knobs on** (M6): the production PR job compiled with
  `--tla-code root_point_ids=true --tla-code nu_dedup_flash_group=true` differs from the
  knobs-off compile by **exactly two keys and nothing else**:
  ```
  605a606 >   "nu_dedup_flash_group": true,
  809a811 >   "rec_charge_provenance": true,
  ```

#### 8.8.2 Unit tests

`wcdoctest-clus` 434/434 (1 skipped, pre-existing), `wcdoctest-root` 8/8. New: the four
`dedup_flash_groups` cases in `doctest_nu_bundle_census.cxx`, and the `rec_charge_provenance`
/ `nu_dedup_flash_group` default-off assertions.

#### 8.8.3 Byte gate, knobs off vs HEAD — **PASS on 267/267**

| output | result |
|---|---|
| `mabc-pr.zip` (member content) | 267/267 same |
| `pctree-pr-evt<ID>.tar.gz` (member content) | 267/267 same |
| `nusel-evt<ID>.tsv` | 267/267 same |
| `calib-pr-evt<ID>.json` (161 written) | same |
| `tracking-pr.root` | **233 055 branches identical**, 0 failing events |

The one allowed difference is `Trun.cfg_tree` on all 267 files: it is the doc 109 provenance
string recording **which pinned cfg tree path the arm read**, and the two arms point at
different paths by construction. Their *contents* are byte-identical where it matters — that
is what 8.8.1's 21/21 proves. (Doc 109 rev 1 did not hit this, because its base binary
predated `root_provenance` and wrote no such branch at all.)

#### 8.8.4 Byte gate, knobs off vs on — **PASS**

Allowed, and why each is right rather than tolerated:

| allowance | files | why |
|---|---|---|
| `T_rec_charge.cluster_id` | see below | the column this round fixes |
| `Trun.op_config_sha256` | all | the two arms compile the job with different TLAs, so the operating point genuinely differs — the provenance hash is doing its job |

Added: `T_rec_charge.nu_index` and `T_rec_charge.point_cluster_id`. `mabc-pr.zip`, the
pctree, nusel and the calib dump are **identical**, so nothing the reconstruction produces
moves.

**On nuecc48, `cluster_id` changes in only 1 of 48 files — and that is the gate telling the
truth about its own blind spot.** These arms run the geometric vertex (`SBND_NO_DL=1`, M4),
and doc 109 sec 4.4 already measured that the geometric vertex produces **zero**
`vertex_moved_cluster = 1` rows on nuecc48 (the 3 moved rows there need the DL vertex). So
the byte-gate arms exercise the demoted-main route (case 3) and **cannot** exercise the
vertex-move route (case 1). The one changed file is `116962`, whose selected activity is a
demoted main:

```
nuecc48 116962   T_bundle.reason = 1 (selected demoted), T_tagger.cluster_id = 21
  knob off   T_rec_charge  466 rows, cluster_id = [-1]
  knob on    T_rec_charge  466 rows, cluster_id = [21],
                           point_cluster_id = [21, 22, 52, 53, 54, 55]
```

Same row count, same points; only the id they carry changed, and `point_cluster_id` now says
the candidate is cluster 21 plus five companions. Case 1 is covered separately in 8.8.7 with
a DL-vertex arm, which is not bit-stable and is therefore graded on content, not bytes.

**Over all 267 events:**

| | |
|---|---|
| `T_rec_charge.cluster_id` changed | **7** files |
| `T_rec_charge` ADDED (the empty-tree path) | **106** files — exactly the no-candidate events |
| `T_proj_data` ADDED | **108** files — the 106, plus 2 whose `fitted_charge_2d` was empty |
| `nu_index`, `point_cluster_id` ADDED | every file with the tree |
| `mabc-pr.zip` / pctree / nusel / calib | **identical**, 267/267 |
| branches compared | 233 055, **0 failing events** |

#### 8.8.5 `new` vs `new2` with the dedup knob off

The Group B fix must not touch the Group A path, and it does not: gating `d109r3ddoff`
(`new`) against `d109r3ddoff2` (`new2`) with the same env over the 25 multi-candidate events
— **PASS**, all **34 825** branches identical, and `mabc-pr.zip`, `pctree-pr`, nusel and the
calib dump the same on all 25. The nuecc48 arm pair `d109r3on` (`new`) vs `d109r3on2`
(`new2`) repeats this on the full Group A configuration (8.8.8).

#### 8.8.6 Content checks C1–C17 on `d109r3on` — **0 failures, 267 events**

C14–C17 are new in rev 3: the tree set is constant, no `cluster_id` is `-1`, `nu_index` names
a real candidate, every candidate's rows carry exactly its `T_tagger.cluster_id`, and
`point_cluster_id == round(ndf)` with every value a cluster `T_cluster` knows.

| measurement | nuecc48 | ncpi0 | mcp1k |
|---|---|---|---|
| files with `T_rec_charge` **and** `T_proj_data` | 48/48 | 19/19 | **200/200** |
| events with no `T_tagger` | 0 | 0 | 106 |
| … `T_rec_charge` rows on those events | — | — | **0** |
| candidates with rows / with 0 rows | 48 / 0 | 20 / 0 | 92 / **3** |
| candidates whose points are **all** from companions | 0 | 0 | **2** |

The 106 no-candidate events now carry an empty `T_rec_charge` and an empty-row
`T_proj_data` — that is the 8.5 claim **measured**, not read off the source. The 3 candidates
with 0 rows are the vertex-less placeholders of case 2/4, and the 2 "all from companions"
candidates are route B of 8.2, which used to be indistinguishable from route A because both
read `-1`.

#### 8.8.7 Case 1, the vertex-moved rows: the DL-vertex arm

The byte-gate arms run the geometric vertex and produce **no** moved rows (8.8.4), so case 1
needs the production vertex. `d109r3dlon` is nuecc48 on pin `new2` with the DL vertex ON and
the Group A knob on. The DL vertex is not bit-stable (M4), so this arm is graded on
**content**, and the "before" is doc 109's own production smoke, which ran the same vertex
mode at the pre-rev-3 code:

| event | before (doc 109 prod smoke, DL) | after (`d109r3dlon`, DL) |
|---|---|---|
| nuecc48 **137238** | `cluster_id` 144, `sel_cluster_id` 7, moved — `T_rec_charge = [7]`, 566 rows | **`T_rec_charge = [144]`**, 566 rows, `point_cluster_id = 7, 38, 39, 40, 41, 42` |
| nuecc48 **269774** | `cluster_id` 87, `sel` 13, moved — `T_rec_charge = [13]`, 1 248 rows | **`[87]`**, 1 248 rows, `point_cluster_id = 13, 19, 29, 30, 31, 32` |
| nuecc48 **52672** | — | `cluster_id` 82, `sel` 9, moved — **`[82]`**, 551 rows, `point_cluster_id = 9, 13, …` |

Same row counts, same points. The rows now join to the candidate that owns them, and nothing
was lost: the cluster the points *came from* — the pre-swap selected cluster 7 / 13 / 9 — is
still in the file, in `point_cluster_id`, where it belongs.

#### 8.8.8 `new` vs `new2` on the full Group A configuration

`d109r3on` (`new`) vs `d109r3on2` (`new2`), identical env, nuecc48: **PASS**, **66 864
branches identical**, every archive the same, and **no allowance of any kind** — not even
`op_config_sha256` or `cfg_tree`, because both arms compile the same job from the same tree.
Together with 8.8.5 this makes the pin split a measured fact: **every Group A number in this
section is reproducible on the committed code.**

### 8.9 Production

**The flip (owner's word, 2026-09-17):** `sbnd/wct-pr-perevt.jsonnet` sets
`root_point_ids = true`. The C++ default (`rec_charge_provenance`) and the `pr()` default
stay **false**, so the lar 1-step chain, PDHD, PDVD and uBooNE do not move.
`nu_dedup_flash_group` stays **false everywhere** (8.7).

**Compiled-config proof.**
- `cmp_consumers.sh` pristine `d2777286` vs the flipped tree: **20 of 21 artifacts
  byte-identical**; only `prod_prjob.json` differs.
- `prod_prjob.json` gains **exactly one key and nothing else**:
  ```
  809a810 >   "rec_charge_provenance": true,
  ```

**Production reference: `ref/prod-2026-09-17`, cut at `12798c4f`, PASS 21/21.**
- `prod_cfg_gate.py --ref ref/prod-2026-09-14` at the **unmodified** HEAD `d2777286` is
  **PASS 21/21** — unlike rev 1 there is **no inherited drift to attribute** here, so the
  flip moves exactly one artifact by exactly one key. The gate names it:
  ```
  DRIFT     : prod_prjob.json
  SBND PR job, key by key (reference -> current tree):
    ADDED   [24].data.rec_charge_provenance = True
  ```
  Component `[24]` is `SbndPrMagnifyTrackingVisitor`. Generation record with every key named:
  `ref/prod-2026-09-17/README.md`.
- **`ref/prod-2026-09-14` is kept.** Nothing in this round makes it stale as a record, and
  removing a generation is its own decision.

### 8.10 Not fixed in rev 3, and why

- **Case 3's *mixed* shape has no local example.** The colleague's slide 21 shows
  `T_rec_charge cluster_ids [-1, 9]` — one candidate's points correct, the other's `-1`. That
  needs a two-candidate event in which **both** candidates produce points and only one is a
  demoted main. It occurs in 12 of the colleague's 13 216 files and in **0 of the local
  3 067**, because the local sample has only 25 two-row events against their 98. It is the
  same code path and the same single `mother_cluster_id` assignment as the all-`-1` shape, of
  which there are 89 local examples. What *is* verified locally is each half of it:
  the all-`-1` shape becoming correct (174422 `[-1] -> [18]`, 280466 `[-1] -> [15]`), and a
  two-candidate event in which **both** candidates carry points getting one id each
  (**286681**, `[3, 10]` with 877 and 65 rows, unchanged by the knob because it was already
  healthy). The mixed shape is those two facts in one event. Stated here rather than claimed
  as verified on a local example.
- **Why a 0.8 cm cluster becomes a neutrino candidate** (the upstream half of cases 2/4).
  `nu_per_bundle_min_length = 15 cm` has an explicit exemption for the legacy event-wide
  winner, and that exemption fires in 462 of the 1 000 mcp1k event logs. Narrowing it is a
  selection change with its own gate and its own owner decision; rev 3 only makes the
  resulting row unmistakable. **Open.**
- **`real_cluster_id` / `sub_cluster_id` are still the same value.** They are bound to one
  address behind a legacy-format comment. Redefining either would silently change a column
  readers already consume (M15); a correctly named branch was added instead.
- **`ndf` still carries the per-point cluster id.** `wire-cell-sbnd-magnify-tracking-convert`
  blocks tracks on `std::round(ndf)`. `point_cluster_id` now carries the same value under an
  honest name; `ndf` is left alone so the convert app and every recorded arm keep working.
- **`T_tagger`/`T_kine` still appear only when a candidate exists.** Booking them empty means
  touching `UbooneTaggerOutputVisitor`'s ~1 000-branch schema, which PDHD and PDVD also use.
  The presence test is meaningful ("a candidate was written") and `T_bundle.reason` explains
  each bundle, so it is left as is.
- **`nue_score == -15` is not a defect** (8.4). `br_filled` distinguishes "never scored" from
  "scored as background" and is already in the file.
- **`nu_dedup_flash_group` is not flipped** (8.7).
- **Unchanged from rev 1:** the one-file-name-per-process limitation of the two writers, and
  the 80 ns flash-t0 merge bookkeeping issue (doc pr/94 sec 9.8).

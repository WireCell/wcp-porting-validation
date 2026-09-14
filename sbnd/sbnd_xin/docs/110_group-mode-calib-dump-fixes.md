# doc 110 — group mode: the calib dump's run number and shower ids

**Owner ask (2026-09-14):** "can you fix the two calib-dump bugs?". These are the two defects
doc 109 rev 2 (sec 7.1) found when it gated group mode.

**Owner decision (same day):** ship both as C++ knobs that default OFF, so PDHD, PDVD and
uBooNE do not move. Turn them ON for SBND group mode only: `sbnd/clus.jsonnet` emits the keys
only when `event_from_ident` is set, so the per-event production job compiles byte-identically.

**Status:**
- **Fixed and gated:** toolkit `apply-pointcloud` `67937f45`.
- **Knobs off:** byte-identical, per-event and in group mode.
- **SBND group mode, knobs on:** every calib dump equals the one-event-per-process job's.
  With the DL vertex, only the wall-clock `off_ms` differs.
- **Production config:** the per-event job's compiled config is unchanged
  (`ref/prod-2026-09-14` PASS 21/21).

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin
# pins: base = ~/tmp/d110-libsnap/base (toolkit c203b400 = doc 109's graded new2 build);
#       new  = ~/tmp/d110-libsnap/new (this round).  md5 in docs/110_logs/libsnap.md5
# cfg:  ~/tmp/d110-cfg/pristine/cfg (git archive c203b400), ~/tmp/d110-cfg/new/cfg (this round)
# 1. compiled config: per-event production job unchanged; group job gains exactly two keys
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-14 --cfg <toolkit>/cfg          # 110_logs/gate_newcfg_vs_0914.txt
#    group-mode PR job (compile_prjob_cfg.sh TLAs + pr_display + multi_event/evt_subdir/rse_map),
#    pristine vs new, key diff                                                      # 110_logs/grp_prjob_keydiff.txt
# 2. unit tests
<toolkit>/build/clus/wcdoctest-clus ; <toolkit>/build/root/wcdoctest-root           # 110_logs/doctests.txt
# 3. arms (same event set as doc 109: nuecc48 48 + ncpi0 19 + first 200 mcp1k) and gates
~/tmp/d110/run_arms_and_gates.sh    # copied as docs/110_logs/run_arms_and_gates.sh
```

## 1. Symptom

In group mode (`PR_GROUP_SIZE>0`, several events in one `wire-cell` process), each event's
`calib-pr-evt<ID>.json` differed from the one-event-per-process job. Doc 109 rev 2 measured
this on nuecc48 45/48, ncpi0 17/19 and mcp1k 79/200. Every other product was byte-identical
(Bee zip, pctree, nusel table, every `tracking-pr.root` branch). The differences fell into
exactly two classes:
- **`meta.runNo`** is the group leader's run (nuecc48 30504: 18342 → 18255).
- **`showers[].shower_id`** continues from the previous event (30504: 0, 1, 2, 4, 6 → 34,
  35, 36, 38, 40).

**Pre-existing:** the base binary before doc 109 shows the same classes with the same counts
(doc 109 sec 7.1, `d109basegrp`).

## 2. Root cause

1. **Run number.**
   - `PrDisplayDump::dump_meta` writes the configure-time `m_runNo`/`m_subRunNo`/`m_eventNo`
     (`clus/src/PrDisplayDump.cxx`, `dump_meta`).
   - In a group job these are the job's TLAs, i.e. the first event's.
   - `visit()` already corrected `eventNo` from `ensemble.ident()` when the file name is
     templated, but never the run.
   - `MultiAlgBlobClustering` already resolves each event's run from `rse_map` and publishes
     it on the ensemble (`Ensemble::set_rse`, only in its multi-event modes).
     `SbndPrMagnifyTrackingVisitor` reads it, which is why `tracking-pr.root` was right.
     `PrDisplayDump` did not read it.
2. **Shower ids.**
   - `PR::Shower`'s constructor takes its id from a process-wide
     `static std::atomic<int> s_shower_id_counter` (`clus/src/PRShower.cxx`).
   - Nothing ever resets it, so event N's showers are numbered after event N-1's.

## 3. Why it hid

- **Before doc 109 rev 2,** every group-vs-per-event gate (doc 76, doc 81) compared the Bee
  zip, the pctree, the nusel tables and `tracking-pr.root`. None of them compared the calib
  dump.
- **The physics is unaffected.** Shower ids are only compared with each other inside one
  event: sort keys at `PRShower.h` and `NeutrinoShowerClustering.cxx`, and the per-event
  `m_hadronic_retyped_shower_ids`. A constant offset preserves every comparison, and every
  physics product stayed identical.
- **Stored ids cannot collide after a reset.** The only two stored id sets are per-event:
  - `PatternAlgorithms` is a local of `TaggerCheckNeutrino::visit`
    (`TaggerCheckNeutrino.cxx:2711`), and clears its set at `shower_clustering_with_nv` entry;
  - `TrackFitting::m_dropped_satellite_shower_ids` is cleared by `reset_for_new_event()`.

## 4. Fix (toolkit `67937f45`)

| knob | component | C++ default | SBND |
|---|---|---|---|
| `rse_from_ensemble` | `PrDisplayDump` | false | `true` when `event_from_ident` |
| `reset_shower_ids_per_event` | `MultiAlgBlobClustering` | false | `true` when `event_from_ident` (PR MABC `clus_pr` only) |

- **`rse_from_ensemble`:** when on and `ensemble.rse_valid()`, `meta.runNo/subRunNo/eventNo`
  come from the ensemble. Without a published RSE (every one-event job) it is inert.
  - The older `%`-template line that sets `meta.eventNo = ensemble.ident()` is left in place.
    In group mode it writes the same value.
- **`reset_shower_ids_per_event`:** when on, `PR::reset_shower_id_counter()` (new, `PRShower.h`)
  runs at each event start in `MultiAlgBlobClustering::operator()`.
  - **Placement:** right after the RSE is published and before the pipeline visitors, so no
    `Shower` of the event exists yet.
  - **Why not `TrackFitting::reset_for_new_event()`:** it runs once per visitor
    (TaggerCheckSTM, TaggerCheckNeutrino, CheckSTM_Michel). `CheckSTM_Michel` builds
    showers, so a reset there could renumber mid-event.
- **`sbnd/clus.jsonnet`:**
  - both keys are key-suppressed on `event_from_ident`, which `wct-pr-perevt.jsonnet` sets
    from `multi_event`, i.e. only in the runner's group path;
  - the stage-A group job does not build `pr()` and does not move;
  - PDHD/PDVD build `PrDisplayDump` in their own `pr.jsonnet`, and neither sets
    `event_from_ident`.
- **Tests:**
  - `doctest_clus_knob_defaults.cxx` pins both defaults to false;
  - `doctest_shower_id_counter_reset.cxx` checks that a reset restarts at 0 and that without
    one the next event keeps counting.

## 5. Verification

All arms use doc 109's event set (nuecc48 48, ncpi0 19, first 200 mcp1k), `setarch -R`,
`PR_EXTRA_STAGES=pr_display` and the `new` pin. The per-event references are doc 109's
`d109on2` (geometric vertex) and `d109dlon2` (DL vertex), on the `base` build. Logs are in
`docs/110_logs/`.

### 5.1 Compiled configuration

- **Per-event production job unchanged:** `prod_cfg_gate.py --ref ref/prod-2026-09-14`
  → **PASS 21/21** with this round's cfg tree (`gate_newcfg_vs_0914.txt`). No reference
  refresh is needed.
- **Group-mode PR job gains exactly two keys** (`grp_prjob_keydiff.txt`):
  `PrDisplayDump:pr.rse_from_ensemble = true` and
  `MultiAlgBlobClustering:clus_pr.reset_shower_ids_per_event = true`.

### 5.2 Unit tests (`doctests.txt`)

- **clus:** 409/409 cases pass, including the 3 new ones (11 assertions).
- **root:** 8/8 cases pass.
- **Freshness:** `libWireCellClus.so` 13:30 is newer than every edited source (13:27).

### 5.3 Byte gates (`scripts/d109_gate.py`)

**Allowed differences, and only these:**
- `Trun.toolkit_git` and `Trun.wcp_git`, the git revisions.
- `Trun.cfg_tree`, which records the pinned cfg-tree path (`~/tmp/d110-cfg/new/cfg`
  here, `~/tmp/d109-cfg/new/cfg` for the references).
  - **What really tells the arms apart:** they pin different cfg trees, and the two trees
    differ by exactly the two key-suppressed lines (`grp_prjob_keydiff.txt`).
  - **`op_config_sha256` cannot show it.** It is identical, but the runner computes it
    from a per-event compile that never passes `multi_event`/`evt_subdir`/`rse_map`
    (sec 6).
- The calib dump's wall-clock `off_ms` (DL vertex only, doc 109 sec 7.1).

| arm | what it proves | vs | result |
|---|---|---|---|
| `d110off` (per-event, knobs absent) | the knob-off path is unchanged | `d109on2` | Bee zip, pctree, nusel and 233,055 ROOT branches identical on 267 events; every calib dump identical (106 not written); the only other difference is `Trun.cfg_tree` |
| `d110grpoff` (group, old cfg tree, knobs absent) | the knob-off group path is unchanged | `d109grp` | **PASS**, calib dump included (both still carry the two defects) |
| `d110grp` (group, knobs on) | **the fix** | `d109on2` (per-event) | **every calib dump identical**: nuecc48 48/48, ncpi0 19/19, mcp1k 94/94 (`calib_classes_on2_vs_d110grp.txt`); everything else identical; the only other difference is `Trun.cfg_tree` |
| `d110dlgrp` (group DL, knobs on) | the fix with the production vertex | `d109dlon2` (per-event) | calib differs only by wall-clock `off_ms`: 48 / 19 / 92 files, 2 identical, **0 run-number or shower-id differences**, 0 other keys (`calib_classes_dlon2_vs_d110dlgrp.txt`); everything else identical |

**Before and after.** Before the fix, group mode differed from per-event in 141 calib dumps
(45 / 17 / 79, doc 109 rev 2). After it, 0 differ with the geometric vertex, and 0 differ
beyond `off_ms` with the DL vertex.

### 5.4 Content checks (`scripts/d109_root_checks.py`, C1–C13)

- **Both group arms, knobs on:** **0 failures** on `d110grp` and `d110dlgrp`
  (`gates/checks_d110grp.txt`, `gates/checks_d110dlgrp.txt`).
- **Row counts:** 48 / 20 / 95, as doc 109.
- **DL vertex:** the moved-cluster rows are 3 / 10 / 4, the same as per-event, with 0
  `DL vertex failed` lines.

### 5.5 The gates with `Trun.cfg_tree` allowed

**First pass.** Every event failed on the three arms that pin this round's cfg tree
(`d110off`, `d110grp`, `d110dlgrp`). The gate prints details for the first 30 failing
events, and each of those names only `value differs Trun.cfg_tree`, plus the calib file on
the DL arm, which sec 5.3 classifies. That covers 30 events, not all 267.

**The verdict** is therefore the rerun of all three gates on all 267 events, with
`Trun.cfg_tree` added to `--allow` (`gates/gate_*_allowcfg.txt`):

| gate | result |
|---|---|
| `d109on2` vs `d110off` (per-event, knobs absent) | **PASS**, 0 failing events of 267 |
| `d109on2` vs `d110grp` (group, knobs on, vs per-event) | **PASS**, 0 failing events of 267, calib dump included |
| `d109dlon2` vs `d110dlgrp` (group DL, knobs on) | 159 events fail, and only on `calib-pr-evt*.json`: everything else passes on all 267. Those 159 are the wall-clock-only files of sec 5.3 (48 + 19 + 92) |

## 6. Not done

- **PDHD/PDVD group mode:** not enabled. Their jobs do not set `event_from_ident`; turning the
  knobs on there is a one-key change per detector when they run group mode.
- **The two stored shower-id sets:** left as they are. Both are per-event already (sec 3).
- **`op_config_sha256` is blind to group mode.** It is reported, not fixed.
  - **Why:** `run_pr_chain_batch.sh` hashes one compile made with placeholder per-event TLAs
    (doc 109 group 4). It never passes `multi_event`/`evt_subdir`/`rse_map`, so the hash
    cannot see `event_from_ident` or the two keys it switches on.
  - **Consequence:** a group-mode file (knobs on) and a per-event file carry the same hash.
    Only the cfg tree and the runner mode tell them apart.
  - **Where the fix belongs:** the runner's provenance compile, which is doc 109's code.

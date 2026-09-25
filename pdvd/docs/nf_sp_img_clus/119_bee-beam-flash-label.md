# doc pdvd/119: why run 39305 has no bottom drift, and a beam-flash label for the Bee "/" key

**Status (2026-09-25, round 1).** Follow-up to [doc 118](118_kaon-run39305-beam-chain.md). The owner asked two
things:

1. **Why do the run-39305 kaon candidates have no bottom-drift TPC data?**
   - The DAQ never read the bottom drift out in this run, so **no reprocessing can recover it** (section 1).
   - Bottom-drift kaon candidates need jjo's beamline selection on a run whose DAQ included the bottom drift.
     039349 is one: a beam run with both drifts read out (its triggers are CTB beam types 15/22).
2. **Bee's `/` key finds "the in-beam flash" with a fixed op_t window. That works for uBooNE/SBND; what should PDVD
   do?**
   - PDVD's beam flash is always at trigger −0.90 ± 0.05 µs, but the trigger moves on the op_t axis from event to
     event (412-438 µs in these samples). No fixed window works (section 2).
   - Implemented: the chain labels the flash itself with a per-flash `op_beam` 0/1 array in the Bee `op` file, and a
     patched Bee prefers that label over its window (sections 3-4).
   - Everything is **default OFF**. Knob-off outputs are **byte-identical** on PDHD, PDVD, SBND and uBooNE
     (section 5.2).

**Flipped 2026-09-25 (owner): `PDVD_BEAM_LABEL` is ON by default in the PDVD runners, with no PE floor**
(section 7). Bee's patched `/` needs a bee3 deploy before a labelled upload behaves differently on the BNL server.

**Round 2 (2026-09-25, section 8): the Bee side panel drew top-drift clusters in the bottom volume.** There were two
bee3 defects: the volume was guessed from uncorrected x, and the already-corrected clustering layer was corrected a
second time. Fixed with a producer-side per-cluster anode (`op_cluster_anodes`, toolkit knob, ON for PDVD) plus bee3.
The corrected placement reproduces the toolkit's `x_t0cor` on every checked cluster: 268/268 on 39305, 297/297 on
the two-sided runs.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
# 0. private A/B build.  The shared tree held peers' uncommitted edits (img/BlobDepoFill.cxx, match/QLMatching.*)
#    and live SBND jobs, so neither wcbuild nor local/lib was touched.
git -C ../toolkit worktree add --detach /home/xqian/tmp/d119wt e2ae041d
#    /home/xqian/tmp/d119_buildAB.sh then d119_buildB2.sh (ROOTSYS set; arm A2 = clean e2ae041d,
#    arm B2 = e2ae041d + util/Bee + clus/MultiAlgBlobClustering patch)
# 1. knob-OFF gates (A2 vs B2)
abtest/libab_clus_run.sh d119gA /home/xqian/tmp/d119inst/A2      # PDHD+PDVD abtest/events.txt, fresh _d119gA dirs
abtest/libab_clus_run.sh d119gB /home/xqian/tmp/d119inst/B2
abtest/libab_clus_compare.sh d119gA d119gB /home/xqian/tmp/d119_gates/pd
#    SBND: run_chain_group.sh --group 0 --from ql into work-nuecc48-d119g{A,B} (imaging from work-nuecc48-d123flip)
#    uBooNE: qlport/scripts/sweep_5384.sh d119ub{A,B}; ab_check.sh d119ubB d119ubA
#    (all in /home/xqian/tmp/d119_gates2.sh)
# 2. knob-ON, two-sided production chain (039349), label ON vs OFF on arm B2
PREFIX=/home/xqian/tmp/d119inst/B2 pdvd/kaon/run_d119_beam_arms.sh 0 1 3 5 9 15 22
python3 pdvd/kaon/d119_onoff_check.py 039349 d119beam d119beamoff 0 1 3 5 9 15 22
# 3. knob-ON, top-only 39305 (doc 118 scratch settings), arm B2
PREFIX=/home/xqian/tmp/d119inst/B2 pdvd/kaon/run_d119_kaon_beam_scratch.sh
# 4. the doc-118 Bee set with the label added offline (phase C)
python3 pdvd/kaon/tag_bee_beam.py /home/xqian/tmp/d118_bee/d118-kaon5.zip /home/xqian/tmp/d119_bee/d119-kaon5-beam.zip
```

Production (since the flip): nothing to set. `PDVD_BEAM_LABEL=0` on both runners recovers the pre-flip outputs.

## 1. Q1: why run 39305 has no bottom drift

Read with h5py from jjo's raw files (`/nfs/data/1/jjo/data/pdvd_kaon_candidates_full/raw/`, 10 files, 478 trigger
records) and from the two-drift reference runs in xning's `hdf5/PDVD/`:

| run (date) | TDEEth (top) | WIBEth (bottom) | bottom source IDs in the file's source-ID map |
|---|---|---|---|
| 39252 (30 Aug) | 96 | 96 | 400-547 |
| 39253 (30 Aug) | 96 | 96 | 400-547 |
| **39305 (4-5 Sep)** | 96 | **0** | **none** |
| 39349 (9 Sep) | 96 | 96 | 400-547 |
| 39663 (28 Sep) | 0 | 96 | 400-547 (bottom only) |

- **Every 39305 record holds the same fragments:** 96 `TDEEth`, 2 `DAPHNE`, 4 `DAPHNEStream`, 2 `Hardware_Signal`, 1
  `Trigger_Candidate` and 1 `TriggerRecordHeader`.
  - The `source_id_geo_id_map` attribute lists only det_id 11 (`kVD_TopTPC`), 9 and 8 (PDS). There is no det_id 10
    (`kVD_BottomTPC`).
- **The bottom was never requested.** The decoded TriggerRecordHeader requests 105 components (96 TDE + 6 PDS + 2 HSI
  + 1 TC) and no bottom source ID. Its error bits are 0 in all 478 records.
- **It holds for the whole run:** file indices 0002-0430, 4 Sep 14:17 to 5 Sep 08:37, five dataflow writers.
- **jjo's processing is not the cause.**
  - `vd_dryrun_cfg.txt` runs `PDVDTPCReader` / `PDVDDataInterfaceWIBEth` with `CrateList: [-1]` and both
    `SubDetectorString{Bottom,Top}`.
  - The art output is `keep *`.
- `charge_bde_us = 0` in the trigoff tree (doc 118) has the same cause.
- **Not established:** why the bottom was left out of the readout (HV, BDE service, or a deliberate top-only beam
  configuration). The DAQ config name is not in these files; it needs the NP02 e-log / run DB or the raw files'
  metacat metadata.

**Consequence.** Reprocessing cannot help. For a bottom-drift or two-drift kaon sample, jjo's Cherenkov/TOF selection
has to run on a beam run that read out both drifts. 039349 qualifies (section 2.1); the 39252/39253 beam status was
not checked here.

## 2. The beam flash on PDVD, and why a fixed window cannot find it

### 2.1 Trigger types

jjo's analyzer stores `static_cast<int>(tc.type)`. The enum is `TriggerCandidateData::Type`
(`dunedetdataformats` v4_4_4-v4_4_6, `include/detdataformats/trigger/TriggerCandidateData2.hpp`; the values are the
same in every version that has them):

| value | type | seen in |
|---|---|---|
| 14 | kCTBBeam | — |
| 15 | kCTBBeamChkvHL | 039349 (237 of 252 records in 9 files) |
| 20 | kCTBBeamChkvHLx | 039252, 039349 |
| 21 | kCTBBeamChkvHxL | — |
| 22 | kCTBBeamChkvHxLx | all 10 kaon events (39305); 039252, 039349 |
| 29 | kCTBOffSpillSnapshot | 039252 (1 event, not beam) |

- **039349 is a beam run.** The "trigger-coincident flash" statistics in doc 118 and
  `ql_light_calib/check_trigger_flash.py` are therefore beam-flash statistics.
- The label is made only for the CTB beam types {14, 15, 20, 21, 22} (jsonnet `beam_tc_types`).

### 2.2 Where the beam flash sits

Flash nearest the trigger, on the raw flash axis (`trigger = tc_us − chain light t0`), from
`kaon/out/light_events.tsv` (full-precision residuals):

| run / tc_type | events | at −0.98..−0.82 µs | outliers (nearest flash dt, PE) |
|---|---|---|---|
| 39252 / 20 | 4 | 4 | — |
| 39252 / 22 | 13 | 12 | +1.87 µs, 13 PE |
| 39305 / 22 | 10 | 9 | 408552: −2.28 µs, 19 PE |
| 39349 / 15 | 26 | 23 | +0.48 µs (30 k PE), −17.9 µs (10 PE), +6.1 µs (29 k PE) |
| 39349 / 22 | 2 | 1 | +3.07 µs, 77 PE |

- **The beam flash sits at −0.90 ± 0.05 µs** from the trigger, on all three runs.
- **On the Bee op_t axis it moves from event to event.** op_t = raw flash time + the per-event crate offset of
  QLMatching input 0. It falls at 412.5 µs (039349) and 413-438 µs (39305).
  - Bee's inherited 2-6 µs window (`experiment.js` base class; PDVD has no override) never contains it.
  - Any fixed window wide enough to contain it holds ~2 unrelated flashes.
- **Correction to doc 118's `kaon/out/beam_flash.tsv` `beam_flash_dt`** (−0.41..−1.33 µs). That column was built
  from `light_events.tsv`'s `trig_us`, which is written to 4 significant figures (e.g. 2794 for 2793.584), so it
  carries a ±0.5 µs rounding error.
  - The flash identities in that table are unaffected, and doc 118 never quoted the column.
  - The new chain passes `trigger_us` at full precision (`%.6f`).

### 2.3 Window

- **The label window is [−1.5, −0.3] µs around the trigger** (−0.9 ± 0.6 µs, jsonnet `beam_window_rel_us`).
  - It contains every flash of the −0.9 µs population.
  - It excludes 408552's 19 PE flash at −2.28 µs (doc 118: no beam flash), which the first-draft window
    [−2.5, +0.5] µs would have labelled.
- **Accidentals:** 1.2 µs × 72 flashes/ms (median all-flash rate, all PE) ≈ 0.09 per event.
- If more than one flash falls in the window, the brightest is labelled.
- **There is no PE floor.** 039349 idx 0 (event 19409) labels a 30 PE flash at −0.96 µs. That is an in-time flash
  that is dim, and it is labelled as such. A floor is an owner choice (section 6).

## 3. Design: the label is made where the trigger is known and read where the flash is drawn

```
rawwf trigoff (tc_us, tc_type)
  -> run_light_evt.sh   trigger_us = tc_us - chain t0 (full precision), tc_type        [PDVD_BEAM_LABEL=1]
  -> flash.jsonnet      OpFlashFinder metadata_extra {trigger_us, tc_type}  -> opflash _metadata.json
  -> run_clus_evt.sh    reads them back with the offsets; passes beam_trigger_us / beam_tc_type TLAs
  -> wct-clustering.jsonnet
        window = beam_trigger_us + side_trigger_offset[group_sides[0]] + beam_window_rel_us
        (the SAME expression that feeds QLMatching trigger_offsets[0], which QLMatching::write_opflash_pc
         adds to the displayed flash time -> one copy of the offset, right for top-only and two-sided)
        emitted only if tc_type is a beam type
  -> clus.jsonnet       MultiAlgBlobClustering clus_all_tpc  bee_beam_window_us: [lo, hi]
  -> MultiAlgBlobClustering::fill_bee_flashes   brightest flash with op_t in [lo, hi] -> op_beam row = 1
  -> Bee::Flashes::set_beam                     "op_beam": [0|1 per row]   (absent unless called)
  -> bee3 op.js nextMatchingBeam                op_beam present -> step through labelled flashes
```

**Why here.**
- The per-event trigger already reached the light runner, which is where the crate offsets are measured.
- The label is display-only: nothing in Q/L selection reads it.
- `match/src/QLMatching.cxx` is left alone. It carries another session's uncommitted edits, and the `Opflash`
  `type` field it would need is reset to 0 with no caller of `set_flash_type`.
- A later option is to feed the same window to QLMatching's existing default-OFF `beam_pref_*` knobs, so the beam
  flash is preferred in matching. That is a physics change and owner-gated.

**Toolkit changes** (all default OFF; the key is absent means byte-identical):

| file | change |
|---|---|
| `util/inc/WireCellUtil/Bee.h`, `util/src/Bee.cxx` | `Bee::Flashes::set_beam(std::vector<int>)` writes `op_beam`, same idiom as `set_t1` / `set_groups` |
| `clus/inc/.../MultiAlgBlobClustering.h`, `clus/src/MultiAlgBlobClustering.cxx` | optional `bee_beam_window_us: [lo, hi]` (µs, op_t axis); validated `lo <= hi`; `fill_bee_flashes` labels the brightest in-window flash (all its rows) and emits an all-zero array when none is in the window |
| `util/test/doctest_bee_flashes_beam.cxx` | new: absent unless set; row-aligned; all-zero case |
| `cfg/.../protodunevd/flash.jsonnet` | `opflash_finder(trigger_us=null, tc_type=null)` → `metadata_extra` keys, suppressed when null |
| `cfg/.../protodunevd/wct-light-reco.jsonnet` | TLAs `trigger_us`, `tc_type` (default null) threaded to the finder |
| `cfg/.../protodunevd/wct-clustering.jsonnet` | TLAs `beam_trigger_us`, `beam_tc_type` (null), `beam_tc_types` [14,15,20,21,22], `beam_window_rel_us` [−1.5, −0.3]; window computed from `side_trigger_offset[group_sides[0]]` |
| `cfg/.../protodunevd/clus.jsonnet` | `all_tpc(..., bee_beam_window_us=null)` → key suppressed when null |

**wcp changes:**
- `pdvd/run_light_evt.sh`: computes `trigger_us` and `tc_type` always (log line only; `none` if the rawwf lacks the
  branches, so the knob-off path cannot fail on it). It passes them only under `PDVD_BEAM_LABEL=1`. All 15 rawwf
  inputs on disk carry both branches.
- `pdvd/run_clus_evt.sh`: reads them from the metadata (the event number moves to the last field, since it can be
  empty). It passes the TLAs only under `PDVD_BEAM_LABEL=1`, and warns if the archive lacks them.

**Bee (wire-cell-bee3, `events/static/js/bee/physics/op.js`, `docs/overview.md`):**
- `hasBeamLabel()` is true when `op_beam` is present and row-aligned with `op_t`.
- `/` then steps (cyclically) through the labelled flashes, whatever their time and whether or not they are matched.
  - An unmatched beam flash is shown with "BEAM flash, no matched cluster (untick 'Matching Cluster' to see all
    charge)".
  - An all-zero array reports "No in-beam flash in this event" and leaves `currentFlash` untouched.
  - A repeated `/` on the only beam flash re-shows it.
- The status line tags `[BEAM]` on labelled flashes whichever key reached them.
- **Without `op_beam` the legacy window code runs unchanged** (uBooNE/SBND and every existing zip). Its two quirks
  are left as they were: a failed search leaves `currentFlash` one ahead, and the displayed group is skipped.
- Checked with node 24 (`--check`, plus 6 behavioural cases on stubs: label jump, unmatched status, repeat, all-zero,
  legacy path, misaligned array).
- **Not yet checked in a browser.** No dev-server session or key presses have been run; do that on
  `d119-kaon5-beam.zip` before deploying.

## 4. Offline demonstration on the doc-118 Bee set (phase C)

`kaon/tag_bee_beam.py` adds `op_beam` to the 5-event doc-118 zip, matching `beam_flash.tsv`'s flash to its op row by
op_t (±0.05 µs), and writes `/home/xqian/tmp/d119_bee/d119-kaon5-beam.zip`. It is not uploaded.

| Bee idx | event | labelled op row | op_t (µs) | PE | Bee matched ids |
|---|---|---|---|---|---|
| 0 | 157312 | 96 | 430.553 | 41 416 | [30] |
| 1 | 317673 | 102 | 414.897 | 36 331 | [33] |
| 2 | 245576 | 74 | 441.362 | 30 492 | [12, 28, 31] |
| 3 | 2001 | 101 | 437.216 | 64 251 | [] |
| 4 | 408552 | — (all zero) | — | — | — |

The Bee op view numbers clusters in its own id epoch (`fill_bee_flashes` runs pre-pipeline). The calib-dump uids
quoted with the doc-118 Bee link (90/172, 101, 151/115/82) are not what Bee shows. The ids above are.

## 5. Verification

### 5.1 Compiled-config proofs (knob OFF byte-identical; knob ON adds only the key)

The runner prepends the shared `toolkit/cfg` to `WIRECELL_PATH`, so a path overlay does not reach it. Each proof
therefore replays the runner's exact `wcsonnet` command (captured with `bash -x`) against the HEAD cfg
(`/home/xqian/tmp/d119wt/cfg`) and the new cfg, deleting the output before each compile. The runner's `-s` check only
tests that the file exists, so a stale file from a previous compile would pass it.

| job (039349 idx 0 / art 19409) | HEAD vs new, knob off | knob on |
|---|---|---|
| `wct-clustering` (two-sided, production TLAs) | **identical** (md5 `b2e0c8a5…`) | only `bee_beam_window_us` added to `clus_all_tpc`: [16.667, 17.867] for trigger 2500.5, offset −2482.333 |
| `wct-clustering`, `beam_tc_type=4` (non-beam) | identical to off | — |
| `wct-light-reco` (production runner TLAs) | **identical** | only `metadata_extra.trigger_us/tc_type` added |
| the runners themselves, knob unset | `.wct-clus.json` and `.wct-light.json` identical to HEAD | light runner under `PDVD_BEAM_LABEL=1`: `trigger_us 2895.824001`, `tc_type 15` |

The HEAD cfg rejects the new TLAs ("function has no parameter beam_trigger_us"). That is the proof the overlay really
compiled the old files.

### 5.2 Knob-OFF gates (arm A2 = clean e2ae041d vs arm B2 = patch, both private installs)

- **Freshness.**
  - B2 carries `Bee::Flashes::set_beam` and the `bee_beam_window_us` string; A2 carries neither.
  - Libraries differing A2↔B2: Util and Clus (the change), plus Aux, Root and Sio, which include `Bee.h` (debug-info
    churn).
  - Live jobs were checked in `/proc/<pid>/maps` to load their own arm's `libWireCellUtil.so`.
- **Doctests (arm B2).**
  - `wcdoctest-util` all pass, including the 3 new `bee flashes` cases (14 assertions).
  - `wcdoctest-clus` 446/446.
  - `wcdoctest-util` had to be linked by hand with `-L build/util` first, because the private build's `-L` order puts
    the production `local/lib` ahead of the in-tree util.

| detector / stage | manifest | labels | result |
|---|---|---|---|
| PDHD + PDVD clustering (`run_clus_evt.sh`, plain harness call) | `abtest/events.txt` (4 PDHD + 2 PDVD events), imaging symlinked from the production dirs | `<det>/work/<run>_<evt>_d119g{A,B}`; hashes `/home/xqian/tmp/d119_gates/pd/hashes_d119g{A,B}.txt` | **PASS, 120/120 archives** member-content identical |
| SBND clustering + Q/L + Bee (`run_chain_group.sh --from ql`, group 0 = 16 nueCC data events) | nuecc48 g0, imaging from `work-nuecc48-d123flip/g0` | `sbnd_xin/work-nuecc48-d119g{A,B}`; `/home/xqian/tmp/d119_gates/sbnd_hashes_{A,B}.txt` | **PASS**, 416/416 hash lines identical (160 files) |
| uBooNE mabc (`qlport/scripts/sweep_5384.sh` + `ab_check.sh`) | 35-event filelist | `qlport/scripts/sweep/d119ub{A,B}` | **PASS**: ZIPS 35/35 content-identical, TAGGER identical=35 diff=0 |

| **PDVD clustering + Q/L + op** (the path this change touches; the harness call above runs with `flags=q0`, and its Bee zips carry no `op` member) | 039349 idx 0,1,3,5,9,15,22 with `-calib`, light `_d119beam`, imaging from `q35flip` | `pdvd/work/039349_<idx>_d119beamA2off` (A2) vs `_d119beamoff` (B2); `kaon/d119_onoff_check.py` | **PASS, 7/7 events, 196/196 archives identical** (`flags=q1,calib1,op1`) |

All arms ran under `setarch -R`, pinned by `PATH`/`LD_LIBRARY_PATH` to their own prefix, and every job had rc=0. The
standard `abtest/run_events.sh` / `sbndgate_run.sh` were not used: the former writes into the production
`work/<run>_<evt>` dirs (M13), and the latter's input set is gone. Their replacements are `abtest/libab_clus_run.sh` /
`libab_clus_compare.sh` (new, fresh tagged dirs, imaging symlinked read-only).

### 5.3 Knob ON, two-sided production chain (039349; same B2 libs, label ON vs OFF twin)

`kaon/d119_onoff_check.py 039349 d119beam d119beamoff 0 1 3 5 9 15 22` gives **PASS 7/7**.
- In every event all 28 archives are member-content identical except `mabc-all-apa.zip`.
- Its only differing member is `data/0/0-op.json`, and that differs only by the added `op_beam`.

| idx / art evt | tc_type | nearest flash to trigger (light axis) | label | op_t (µs) | PE | Bee matched |
|---|---|---|---|---|---|---|
| 0 / 19409 | 15 | −0.96 µs, 30 PE | **yes** | 412.526 | 30 | — |
| 1 / 19429 | 22 | +3.07 µs | none | — | — | — |
| 3 / 19469 | 15 | +0.48 µs, 30 k PE | none | — | — | — |
| 5 / 19509 | 15 | −0.94 µs, 5.4 k PE | **yes** | 412.659 | 5 438 | — |
| 9 / 19589 | 15 | −17.9 µs | none | — | — | — |
| 15 / 19709 | 22 | −0.84 µs, 260 PE | **yes** | 412.520 | 260 | — |
| 22 / 19849 | 15 | +6.1 µs, 29 k PE | none | — | — | — |

Every label agrees with the light-axis expectation. None of the three 039349 beam flashes is paired with a cluster by
Q/L.

### 5.4 Knob ON, top-only 39305 (doc-118 scratch settings; NOT production-equivalent)

`kaon/run_d119_kaon_beam_scratch.sh` ran all 10 events to rc=0. It used post-cull OFF and `trigger_offsets` padded
to [top, top], because doc 118's two single-side QLMatching defects are still open. Light went to
`work/039305_light<evt>_d119beam`, clustering to `work/039305_<evt>_d119beamscratch`; the canonical doc-118 dirs were
only read.

The window is built from the top offset, because group 0 is the top group on this top-only run. Example: event 2001,
trigger 2807.472 µs on the flash axis, window [436.599, 437.799] µs on op_t. **The chain's label agrees with doc 118's
beam-flash identification on 10/10 events:**

| event | expected (doc 118) op_t (µs) | labelled op_t (µs) | PE | Bee matched ids |
|---|---|---|---|---|
| 2001 | 437.216 | 437.216 | 64 251 | — |
| 36591 | 437.375 | 437.375 | 14 807 | — |
| 69596 | 431.973 | 431.973 | 1 241 | — |
| 157312 | 430.553 | 430.553 | 41 416 | [30] |
| 191916 | 413.346 | 413.346 | 83 592 | [37] |
| 245576 | 441.362 | 441.362 | 30 492 | [12, 28, 31] |
| 317673 | 414.897 | 414.897 | 36 331 | [33] |
| 326459 | 444.090 | 444.090 | 477 | — |
| 351293 | 435.349 | 435.349 | 3 686 | — |
| 408552 | none | none (all-zero `op_beam`) | — | — |

The chain-labelled 10-event Bee set (img, clustering, op and dead area; no PR layers) is
`/home/xqian/tmp/d119_bee/d119-kaon10-chain.zip`, in the doc-118 order followed by the other 5 events. It is not
uploaded.

## 6. Owner decisions / open items (as of round 1; decisions in section 7)

- [ ] **Commit, push and deploy bee3.** The change is uncommitted in `wire-cell-bee3`; push and the BNL deploy are
      owner-gated. Until then an `op_beam` zip looks and
      behaves as before on the server.
- [ ] **Flip `PDVD_BEAM_LABEL=1`** as the PDVD runner default. It is display-only; the gates in section 5 are the
      evidence. It needs the light stage rerun (or the metadata) for existing events.
      - **First the C++ must be committed and installed.** Otherwise the flip is a silent no-op: the jsonnet emits
        `bee_beam_window_us`, and an old `libWireCellClus` ignores the unknown key.
      - Check before flipping: `strings -a local/lib/libWireCellClus.so | grep -c '^bee_beam_window_us$'` must be ≥ 1,
        and `nm -DC local/lib/libWireCellUtil.so | grep -c Flashes::set_beam` must be 1.
- [ ] **PE floor?** A dim in-time flash (30 PE, 039349/19409) is labelled today. The accidental rate at the 1.2 µs
      window is ~0.09/event at any PE.
- [ ] **Beam-flash preference in Q/L** (`beam_pref_*` with this window) is a physics change and needs its own study.
      On these samples the labelled beam flash is unmatched in 039349 (3/3) and in kaon evt 2001.
- [ ] **Bottom-drift kaons:** ask jjo to run the beamline selection on 039349-type (both-drift) beam runs. Ask why
      39305 ran top-only (NP02 e-log).
- [ ] Doc 118's QLMatching single-side defects are still open. The 39305 labels here are on the scratch arm.
- [ ] `abtest/sbndgate_run.sh` is stale: its `input-10evt-mc` set was removed in a disk cleanup. The SBND gate here
      used `run_chain_group.sh --from ql` on nuecc48 group 0 instead.

## 7. Owner decisions and the flip (2026-09-25)

The owner said: commit and push, make the label the PDVD default, and use no minimum-PE cut.

- **Runner defaults.** `run_light_evt.sh` and `run_clus_evt.sh` now use `${PDVD_BEAM_LABEL:-1}`.
- **What changes in production output:**
  - light archive: the `_metadata.json` gains `trigger_us` and `tc_type`. The flashes are identical: label-on vs
    label-off on arm B2, 039349 art 19409 and 19709, 11 members each, the only difference is those two added keys.
  - clustering: the Bee `op` json gains `op_beam`. Every other archive and member is identical (section 5.3, 7/7
    events).
- **Compiled-config checks after the flip** (039349 idx 0):
  - default → `bee_beam_window_us` [411.991, 413.191] (trigger 2895.824 µs, tc_type 15);
  - `PDVD_BEAM_LABEL=0` → configs identical to HEAD (light and clus);
  - a pre-flip light archive with no trigger metadata → a warning, and a clus config identical to HEAD.
- **No PE floor.** This is the brightest-in-window rule of section 3; accidentals are ~0.09/event at any PE.
- **Install dependency.** The flip is only effective once the Util/Clus change is installed and in the shared
  `build/`. Without that, the jsonnet key is ignored by an old `libWireCellClus`. The runtime `LD_LIBRARY_PATH` lists
  `build/<pkg>` ahead of `local/lib`, so both must carry it:

  ```bash
  strings -a local/lib/libWireCellClus.so   | grep -c '^bee_beam_window_us$'   # >= 1
  strings -a build/clus/libWireCellClus.so  | grep -c '^bee_beam_window_us$'   # >= 1
  nm -DC local/lib/libWireCellUtil.so | grep -c 'Flashes::set_beam'             # 1
  ```

  The install record (time, which targets, the checks above) is in section 7.1.

### 7.1 Install record

- **Commits:** toolkit `773c8d15` (util, clus, cfg/protodunevd); wire-cell-bee3 `4d47e3c` (pushed to `main`). The
  runner flip and this doc are in the wcp commit that carries this line.
- **Toolkit pushed:** `773c8d15` on `apply-pointcloud`. The push also carried a peer session's already-committed
  `8822b2a1` (SBND `xtpc_sc1_overpred_max`, default OFF), as the owner chose.
- **Installed 2026-09-25 08:22-08:24.** `/home/xqian/tmp/d119_install.sh` waited for the shared tree to be idle
  (120 s with no `wire-cell` job or driver; a peer SBND stage-B campaign ran until then). It then built and installed
  only `WireCellUtil,WireCellAux,WireCellClus`, with build rc=0 and install rc=0 (log
  `/home/xqian/tmp/d119_install.log`). Img was not rebuilt: the tree holds a peer's uncommitted `BlobDepoFill.cxx`,
  and `libWireCellImg.so` is still the 09-24 build.

  | library | md5 (12) | `bee_beam_window_us` / `set_beam` |
  |---|---|---|
  | `local/lib/libWireCellUtil.so` = `build/util/…` | `b5ce94050774` | `Flashes::set_beam` 1 |
  | `local/lib/libWireCellClus.so` = `build/clus/…` | `300683646def` | key 1 |
  | `local/lib/libWireCellAux.so` | `4d10de1118bf` | (rebuilt: includes `Bee.h`) |

- **End-to-end production smoke test.** The runners were run bare: installed libs, no knob, no pin, 039349 idx 5 /
  art 19509, fresh `_d119prod` light + `d119prod` clus. The output is identical on **28/28 archives** to arm B2's
  label-on output `d119beam` for the same event, `op_beam` included (`kaon/d119_onoff_check.py 039349 d119prod
  d119beam 5`). The log reads "Beam label: trigger 2772.144 us on the flash axis, tc_type 15".
- **Still open:** the bee3 deploy on the BNL Bee server (owner), and a browser check of `/` on a labelled set.

## 8. Bee side panel put top-drift clusters in the bottom volume (2026-09-25, round 2)

The owner, on the 10-event set (`bee/set/a6a48e04-…`, event 1 = 317673): "in the side panel the matched charge
cluster is placed at the bottom detector; these data have only the top detector."

### Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
# 0. private A/B build: arm A3 = clean toolkit 1e6b2905, arm B3 = + this change (kaon/d119s_build.sh, run in the
#    /home/xqian/tmp/d119wt worktree with the change as a patch; the change is toolkit 9de7fcae)
# 1. the defect, replayed offline on the uploaded zip (old rule) and on the fixed zip
python3 kaon/d119s_side_replay.py /home/xqian/tmp/d119_bee/d119-kaon10-chain.zip   # uploaded: 25 of 268 in the bottom
python3 kaon/d119s_side_replay.py /home/xqian/tmp/d119_bee/d119-kaon10-side.zip    # fixed: 0 of 268, dx <= 0.002 cm
# 2. gates + knob-ON arms (kaon/d119s_gates.sh runs all of these plus the harness/SBND/uBooNE gates)
PREFIX=/home/xqian/tmp/d119inst/A3 TAG=d119sA3    CLUS_TLA="-S bee_cluster_anodes=false" kaon/run_d119s_side_arms.sh 0 1 3 5 9 15 22
PREFIX=/home/xqian/tmp/d119inst/B3 TAG=d119sB3off CLUS_TLA="-S bee_cluster_anodes=false" kaon/run_d119s_side_arms.sh 0 1 3 5 9 15 22
PREFIX=/home/xqian/tmp/d119inst/B3 TAG=d119sB3    kaon/run_d119s_side_arms.sh 0 1 3 5 9 15 22
PREFIX=/home/xqian/tmp/d119inst/B3 TAG=d119sB3 RUN=039252 LIGHT_SUFFIX=_tot BEAM=0 kaon/run_d119s_side_arms.sh 0
PREFIX=/home/xqian/tmp/d119inst/B3 kaon/run_d119s_kaon_side_scratch.sh              # 39305 x10 -> *_d119sidescratch
python3 kaon/d119s_arm_check.py 039349 d119sA3 d119sB3off 0 1 3 5 9 15 22                          # knob off
python3 kaon/d119s_arm_check.py --key op_cluster_anodes 039349 d119sB3 d119sB3off 0 1 3 5 9 15 22  # knob on
python3 kaon/make_d119_bee_zip.py d119sidescratch /home/xqian/tmp/d119_bee/d119-kaon10-side.zip
# 3. browser: the live Bee page with the local bee3 methods swapped in and the fixed zip served locally
#    (Playwright + Xvfb; screenshots go to /home/xqian/tmp/d119_pw/)
PATCH=1 ZIP=/home/xqian/tmp/d119_bee/d119-kaon10-side.zip \
    xvfb-run -a -s "-screen 0 1600x1000x24" python3 kaon/bee_side_check.py 0 1 2 7
```

### 8.1 Symptom

Reproduced in the browser on event 1 after `/` (beam flash #102, 414.897 µs, matched cluster 33):

| layer | side-panel x of cluster 33 (cm) | where it belongs (the toolkit's `x_t0cor`) |
|---|---|---|
| `img-global` (raw) | −113.5 … 8.5: **bottom volume** | 9.4 … 131.4, top |
| `clustering-global` (Bee's default layer) | 70.8 … 192.9: top, but **61.4 cm too far from the cathode** | 9.4 … 131.4 |

### 8.2 Root cause: two defects, both in bee3's detector-frame (side-panel) code

1. **The volume of a raw cluster was guessed from its x, and near the cathode x cannot tell.**
   - bee3 `934d031` (July 17) tries both drift directions and keeps the one whose T0-corrected charge lands in a
     box. A cluster within v·t of the cathode lands in a box either way: a tie.
   - A tie was broken by the sign of the cluster's mean raw x. That is not a side signal: a top cluster within v·t of
     the cathode has negative raw x (raw = true − v·t), so it went to the bottom. Mirror-image for bottom clusters.
   - The per-flash `apa` cannot replace it: one PDVD flash matches clusters in both volumes, and even on top-only
     39305 one matched flash has `apa=0`.
   - On the uploaded 10 events, 61 of 268 matched clusters are ties and **25 went to the bottom**, among them the
     beam-matched clusters of 317673 (33), 245576 (28) and 191916 (37).
2. **The side panel T0-corrected a layer that is already corrected.**
   - Since toolkit `9ded936c2` (2026-07-09, the Q/L wiring, which also added the `op` dump), PDVD writes
     `clustering-global` in `x_t0cor`.
   - The side panel treated every layer as raw and shifted it again by v·t. On this layer the geometric test then
     sent 78 of 268 matched clusters to the bottom: 77 by a clear majority of points, 1 by the tie-break.

**Why it hid.** `934d031` was validated on two-sided 039252 evt 298567 against `x_t0cor` and left its 9 ties
unresolved. Most tied clusters on a two-sided run are drawn in *a* volume, just the wrong one. A top-only run,
where anything drawn in the bottom is visibly wrong, is what exposed it.

### 8.3 Fix

- **Toolkit (producer): each matched cluster's anode.**
  - New knob `MultiAlgBlobClustering.bee_flash_cluster_anodes` (C++ default false). When on, `fill_bee_flashes`
    writes `op_cluster_anodes`: one list per op row, parallel to `op_cluster_ids`, holding each cluster's anode ident
    (the anode holding most of its blobs).
  - At that pre-pipeline point a cluster lives on one drift side, so this is exact, not a guess.
  - `util` `Bee::Flashes::set_cluster_anodes`; PDVD `clus.jsonnet` `bee_flash_cluster_anodes` (key-suppressed).
  - `wct-clustering.jsonnet` TLA `bee_cluster_anodes`, **default true for PDVD Q/L jobs** (owner's ask to fix it);
    `-S bee_cluster_anodes=false` restores the previous compiled config.
- **bee3.**
  - `ProtoDUNEVD.detectorFrameCorrection` takes a cluster's volume from `op_cluster_anodes` when present (top for
    anodes 4-7, with that volume's own clock). The geometric test stays only as the fallback for older dumps.
  - New `Experiment.layerInDetectorFrame(sst)`, base false. PDVD returns true for `clustering-global`, and the side
    panel then draws that layer unshifted.
  - `docs/overview.md` documents `op_cluster_anodes`.
  - The clustering-layer half needs only the bee3 deploy and works on the already-uploaded set. The img-layer half
    also needs a zip made with the new toolkit.

### 8.4 Verification

- **Doctests (arm B3).**
  - `wcdoctest-util` passes everything, including the new `bee flashes op_cluster_anodes` case (4 bee cases, 23
    assertions). It was linked by hand with the in-tree util first (section 5.2).
  - `wcdoctest-clus` 446/446.
- **Freshness.** B3 carries `Flashes::set_cluster_anodes` and the `bee_flash_cluster_anodes` string; A3 carries
  neither.
- **Compiled config (039349 idx 0, the runner's captured `wcsonnet` line, HEAD cfg `git archive 1e6b2905` vs new).**
  - HEAD gives md5 `b2e0c8a5…`, the same as section 5.1.
  - The new cfg adds exactly one line, `"bee_flash_cluster_anodes": true` in `clus_all_tpc`.
  - The new cfg with `bee_cluster_anodes=false` is identical to HEAD, and so is `do_qlmatch=false`.
  - The HEAD cfg rejects the TLA ("no parameter bee_cluster_anodes").

| gate | labels | result |
|---|---|---|
| PDVD Q/L + op path, knob off (A3 vs B3, both `bee_cluster_anodes=false`) | `039349_{0,1,3,5,9,15,22}_d119s{A3,B3off}` | **PASS 196/196 archives** identical |
| PDVD knob on vs off (B3) | `_d119sB3` vs `_d119sB3off` | **PASS 7/7**: only `op.json` differs, only by `op_cluster_anodes`; row-aligned, parallel to the ids; anodes 0-7 all present |
| PDHD + PDVD harness (`flags=q0`), A3 vs B3 | `<det>/work/<run>_<evt>_d119sg{A,B}`; `/home/xqian/tmp/d119s_gates/pd/` | **PASS, 120/120 archives** |
| SBND Q/L + Bee, nuecc48 g0, A3 vs B3 | `work-nuecc48-d119sg{A,B}`; `/home/xqian/tmp/d119s_gates/sbnd_hashes_{A,B}.txt` | **PASS**, 416/416 hash lines identical |
| uBooNE mabc, 35 events, A3 vs B3 | `qlport/scripts/sweep/d119sub{A,B}` | Bee zips **35/35** content-identical; tagger identical 34, **differs 1 (idx 22, ev 6805)**. The difference is run-to-run instability that exists without this change, shown by a null pair below |

**uBooNE ev 6805 is unstable run to run, on both arms.** The only differing quantities are
`kine_pio_theta_2/phi_2/dis_2/angle` (the second π⁰ photon).
- Rerunning `run_one.sh 22` under `setarch -R` with `dl_weights` empty gives two tagger outcomes on each arm (the
  tagger-compare log md5 is `fd7837ac` or `ce1b2205`). This includes two consecutive runs on the same B3 binary.
  - clean arm A3, 7 runs: 4 × `fd78`, 3 × `ce1b`;
  - B3, 5 runs: 2 × `fd78`, 3 × `ce1b`.
- The gate simply drew `fd78` on A3 and `ce1b` on B3.
- This is the M4 class: pattern recognition that depends on pointer order. Nothing in this change runs on uBooNE
  (`bee_flash_cluster_anodes` is off there).
- It is **pre-existing on 1e6b2905** and was not seen in round 1: `d119ub{A,B}` on e2ae041d were both `fd78`, from
  one draw each. Reported, not fixed. Labels: `qlport/scripts/sweep/d119sub{A,B}_n{1..6}`.

**Is the placement right? Checked against the toolkit's own `x_t0cor`.**
- **Top-only 39305.** On the 10 events (`*_d119sidescratch`) the img and clustering layers are point-aligned, so the
  img-layer side-panel x of every matched cluster can be compared point by point with `clustering-global`.

  | zip | matched clusters | drawn in the bottom | disagree with `x_t0cor` |
  |---|---|---|---|
  | uploaded (old rule) | 268 | 25 | 25 (up to 466 cm) |
  | `d119-kaon10-side.zip` (new rule) | 268 | **0** | **0** (max 0.0017 cm, float rounding) |

  - The new zip differs from the uploaded one only by `op_cluster_anodes`: 10 of 40 members, anodes {4,5,6,7} only.
- **Two-sided 039349 x7 + 039252 evt 298567.** The layers are not point-aligned there (the clustering layer drops
  and re-enumerates). So img points were matched to clustering points by (y, z, q), and each cluster was graded by
  whether its side-panel x reproduces `x_t0cor`. **New rule 297/297 right; old rule 267/297.**
  - The 30 old errors go both ways (top→bottom and bottom→top). Each is fixed by its anode.
  - 5 of the 30 are on 298567, the event `934d031` was validated on.
- **Browser.** The live Bee page was checked with the local bee3 methods swapped in and the fixed zip served locally
  (`bee_side_check.py`). After `/`, events 0, 1, 2, 7 draw the beam-matched clusters at the same place on both
  layers, all in the top, and for all matched clusters the img layer equals `x_t0cor` to ≤ 0.0015 cm. Event 1,
  cluster 33: 9.4 … 131.4 cm.
  - Without the fix the same page gives −113.5 … 8.5 (img) and 70.8 … 192.9 (clustering).

### 8.5 Not fixed, reported

- **Main (reco-frame) panel on `clustering-global`.** Found from the code, not checked in a browser.
  - `op.js buildGroup` shifts the boxes by v·t·driftDir, which is correct for raw charge. The clustering layer is
    already corrected, so there a matched cluster sits v·t away from its shifted box: 61.4 cm for cluster 33.
  - The fix would be the same `layerInDetectorFrame` test (no box shift for that layer). It is left for the owner's
    call because it changes what the main panel shows.
- **PDHD and SBND also write `clustering-global` in `x_t0cor`** (pdhd/sbnd `clus.jsonnet`). Their side panels may
  double-correct the same way. `layerInDetectorFrame` is PDVD-only here, and the other two are unchecked.
- `run_d119s_kaon_side_scratch.sh` inherits doc 118's scratch settings: the two QLMatching single-side defects are
  still open.

### 8.6 Commits, deploy and install status

- **Pushed:** toolkit `9de7fcae` on `apply-pointcloud`, a fast-forward on top of a peer's `7312f2b3`, which was
  already on the remote; bee3 `9cdccfe` on `main`.
- **Bee deploy (owner / colleague).** The BNL server loads the parcel bundle `static/js/bee/dist/bee.js`, so `main`
  must be rebuilt and redeployed before the fix is visible.
  - After that, `clustering-global` is placed right on the already-uploaded set.
  - `img-global` also needs a zip carrying `op_cluster_anodes`: `/home/xqian/tmp/d119_bee/d119-kaon10-side.zip`,
    **not uploaded**.
  - Post-deploy check: `kaon/bee_side_check.py 1` without `PATCH`. On event 1, `clustering-global`, cluster 33 must
    draw at 9.4 … 131.4 cm.
- **Shared-tree install: done 2026-09-25, by the owner.**
  - `/home/xqian/tmp/d119s_install.sh` never ran. Its idle check matched the command line of a peer wcfm imaging
    launcher that was paused for this install. Running the install from this session was then refused by its
    permission check. The owner ran `./wcb build --notests -p --targets=WireCellUtil,WireCellAux,WireCellClus` and
    then `./wcb install` with the same targets.
  - The build relinked at 11:24-11:25, while the peer's imaging batch had just resumed. None of its jobs failed: all
    `rc=0`, and no library-load errors in its logs.

  | library | `local/lib` = `build/` md5 (12) | installed | check |
  |---|---|---|---|
  | `libWireCellUtil.so` | `32e47a073c8d` | 11:24:43 | `Flashes::set_cluster_anodes` 1 |
  | `libWireCellClus.so` | `3ec8aa213aaa` | 11:25:28 | `bee_flash_cluster_anodes` 1 |
  | `libWireCellAux.so` | `bcae36f8d6fa` | 11:24:47 | (includes `Bee.h`) |
  | `libWireCellImg.so` | `a1406cf527e9` | 08:30:20, untouched | — |

- **Production smoke test.** The production runner on the installed libs, with no TLA or override: 039349 idx 0 /
  art 19409 into the fresh tag `d119sprod` (`flags=q1,calib1,op1`). It is identical to the gated arm B3 `d119sB3` on
  **28/28 archives**, `op_cluster_anodes` included (`kaon/d119s_arm_check.py 039349 d119sprod d119sB3 0`).
  PDVD production now writes `op_cluster_anodes`.
- **Bee redeployed and the fixed set uploaded (2026-09-25, on the owner's ask).**
  - The served `dist/bee.js` contains `op_cluster_anodes` and `layerInDetectorFrame`.
  - `d119-kaon10-side.zip` is uploaded as `https://www.phy.bnl.gov/twister/bee/set/0f1fb8b8-3688-4c94-aaf0-80e8d4e7fdbc/event/list/`.
  - Checked on the live page, no local override (`SET=… kaon/bee_side_check.py 0 1 2 3 7`): after `/` both layers draw
    the beam-matched clusters at the same place, in the top volume. For all matched clusters the img layer equals
    `x_t0cor` to ≤ 0.0015 cm, with 0 in the bottom volume (event 1, cluster 33: 9.4 … 131.4 cm). Event 3's beam
    flash has no matched cluster, as before.
  - On the old set (`a6a48e04`), the redeployed Bee now places `clustering-global` right (9.4 … 131.4). Its
    `img-global` is still in the bottom, because that zip has no `op_cluster_anodes`: use the new set.

## Status flags

- Toolkit C++ (`util` Bee, `clus` MultiAlgBlobClustering) and PDVD jsonnet: **byte-identical when off**. Gates are in
  section 5.2, compiled-config proofs in 5.1.
- wcp runners: `PDVD_BEAM_LABEL` unset gives compiled configs identical to HEAD.
- **PDVD production behaviour changed on purpose** (the flip, section 7): the Bee `op` json gains `op_beam`, and the
  light metadata gains 2 keys. Nothing else changes. The gated binaries are `/home/xqian/tmp/d119inst/{A2,B2}`.
- Bee: display-only change, legacy path untouched when `op_beam` is absent.
- **Round 2 (section 8), `op_cluster_anodes`:** the C++ is **byte-identical when off** (196/196 on the PDVD op path,
  120/120 PDHD+PDVD harness, SBND 416/416, uBooNE zips 35/35; section 8.4, where uBooNE ev 6805's tagger instability
  is shown to be pre-existing). PDVD Q/L jobs turn it on by default, and their only output change is
  the added `op_cluster_anodes` in the Bee `op` json. bee3's side panel changes for PDVD only. Gated binaries:
  `/home/xqian/tmp/d119inst/{A3,B3}`.

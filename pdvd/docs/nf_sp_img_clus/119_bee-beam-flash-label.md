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
- **Install: pending at the time of that commit.** A peer's SBND stage-B campaign
  (`work-r3off-d123lgop` → `…lgoppr`) was launching jobs every few seconds, and a relink of the shared `build/` under a
  job start kills it ("file too short").
  - `/home/xqian/tmp/d119_install.sh` waits for 120 s with no `wire-cell` job and no driver.
  - It then builds and installs only `WireCellUtil,WireCellAux,WireCellClus` (not Img: the tree holds a peer's
    uncommitted `img/src/BlobDepoFill.cxx`), and logs the checks above to `/home/xqian/tmp/d119_install.log`.
  - A follow-up commit records the result here.
- **Until the install lands, the flipped runners already stamp the light metadata.** The clustering jsonnet emits
  `bee_beam_window_us`, which the old `libWireCellClus` ignores, so there is no `op_beam` yet and nothing else
  changes.

## Status flags

- Toolkit C++ (`util` Bee, `clus` MultiAlgBlobClustering) and PDVD jsonnet: **byte-identical when off**. Gates are in
  section 5.2, compiled-config proofs in 5.1.
- wcp runners: `PDVD_BEAM_LABEL` unset gives compiled configs identical to HEAD.
- **PDVD production behaviour changed on purpose** (the flip, section 7): the Bee `op` json gains `op_beam`, and the
  light metadata gains 2 keys. Nothing else changes. The gated binaries are `/home/xqian/tmp/d119inst/{A2,B2}`.
- Bee: display-only change, legacy path untouched when `op_beam` is absent.

# 95 — the colleague's 25-event MC debug sample through our production chain

> **Bee set (25 events, art-file order):**
> **https://www.phy.bnl.gov/twister/bee/set/e3d1d867-d34e-42a2-9709-1954fdfde54b/event/list/**
>
> Processed at the pinned SBND operating point `ref/prod-2026-09-03`
> (`stm_entry_rise_guard` ON), `reality=sim`. Verdicts: **13 TGM, 9
> nu-candidate, 6 STM** over 29 in-beam bundles in 25 events.
>
> This sample is a **superset of doc 93's 8 events**. On those 8, every
> kinematic number reproduces doc 93 exactly and the only differences are the
> **4 STM → nu-candidate releases** the doc-94 guards were shipped to make.
> Among the **17 events doc 93 did not contain, no guard fired at all.**

## Repro block

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# A0 -- stage each art entry into its OWN single-event sample dir.
#       NOT run_reco1_dump.sh over the whole file: see sec 1.
./scripts/dbg25_stage.sh 25 6

# A1 -- rebuild into two collision-free sample dirs (20 + 5 events)
./scripts/dbg25_groups.sh

# A2/B -- imaging, Q/L, then the 15-stage PR chain, for both groups
SBND_MAX_JOBS=6 PR_JOBS=6 ./scripts/dbg25_run.sh all "a b"

# the tables and the Bee set
python3 scripts/dbg25_manifest.py -o bee/dbg25/dbg25.manifest.tsv
python3 scripts/dbg25_table.py -m bee/dbg25/dbg25.manifest.tsv \
        -o bee/dbg25/dbg25-tagger-summary.tsv
python3 scripts/bee/make_dbg25_bee.py -m bee/dbg25/dbg25.manifest.tsv \
        -o bee/dbg25/dbg25.zip
(cd bee/dbg25 && BROWSER=echo bash ../../upload-to-bee.sh dbg25.zip)
```

No code was changed, no default moved, no production runner was edited. New
files are the five scripts above plus the Bee sidecars.

## 1. The sample — and the collision that would have eaten five events

`input_files_reco1/stm_tagger_feedback/debug-25evt-reco1.root`, 25 entries.
Branch scan settles the lineage exactly as in doc 93 §1:

| product | data branch (C++ default) | this file |
|---|---|---|
| DNN-SP wires | `recob::Wires_sptpc2d_dnnsp_Reco1.` | `recob::Wires_simtpc2d_dnnsp_DetSim.` |
| bad-channel masks | `ints_sptpc2d_badmasks_Reco1.` | `ints_simtpc2d_badmasks_DetSim.` |
| Wiener summary | `doubles_sptpc2d_wienersummary_Reco1.` | `doubles_simtpc2d_wienersummary_DetSim.` |
| FrameShiftInfo | present | **absent** |

plus `GenieGen`/`corsika`/`MCTruth`/`MCShower`. So **MC**: `-mc -caf none`,
`reality=sim` throughout (which gates the `switch_scope` `pos_offset`
transverse correction — running these as `data` would move every point ~6.8 cm
in y–z, doc pr/38 round 3).

**The trap.** A whole-file `run_reco1_dump.sh -t dbg25` produces 25 frame
members but only **20 distinct event ids**:

```
$ tar tjf .../frames-dnn.tar.bz2 | grep -c '^frame_dnnsp_'          # 25
$ tar tjf .../frames-dnn.tar.bz2 | grep '^frame_dnnsp_' | sort -u | wc -l  # 20
```

Every downstream name — `frame_dnnsp_<ID>.npy`, `work/evt<ID>`, `ql_evt<ID>`,
`pr_evt<ID>` — is keyed on the **bare event id**, which is unique only within a
`(run, subrun)`. These 25 debug events are drawn from 100 files and span **20
different runs**, so five ids collide. In one sample dir the second copy
silently overwrites the first: the chain would have run 20 events, reported
success, and no error would appear anywhere.

Authoritative map, from each staged entry's `opflash_tensorset_*_metadata.json`
(`input_files_reco1/staged-dbg25/entry_event_map.tsv`):

```
A1 distinct (run,subrun,event): 25 / 25  OK
A2 distinct bare event ids    : 20 / 25  FAIL  dups=[12, 14, 22, 31, 34]
A3 distinct runs              : 20
```

| bare id | entry → RSE | entry → RSE |
|---:|---|---|
| 12 | 3 → 707-18-12 | 18 → 651-84-12 |
| 14 | 20 → 993-78-14 | 21 → 44-40-14 |
| 22 | 7 → 541-46-22 | 13 → 966-2-22 |
| 31 | 4 → 146-60-31 | 5 → 60-38-31 |
| 34 | 6 → 445-18-34 | 19 → 651-84-34 |

**Fix**: stage per entry (`--tla-str entry=<i>`, the mechanism
`staged-mcp2025c-1000evt/stage_all.sh` exists for), then rebuild into two
sample dirs whose ids are internally unique — group **a** = first occurrence of
each id (20 events), group **b** = the five second occurrences. Membership is
derived from the map, not hand-typed. Each group gets its own work-root pair,
so `work/evt<ID>` is collision-free by construction.

**This is why `RSE`, not the event id, is the identity in every table and
sidecar of this round.**

## 2. Verdicts (the deliverable)

One row per **in-beam** bundle; 4 of the 25 events have two, so 29 rows.
Full table: `bee/dbg25/dbg25-tagger-summary.tsv`.

| bee_idx | RSE | verdict | t0 (µs) | flash PE | main len (cm) | main pts | doc 93 |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | 492-51-23 | TGM | 1.239 | 15102 | 152.4 | 4903 | |
| 1 | 827-27-4 | **nu-candidate** | 0.366 | 6384 | 114.4 | 1293 | STM → **released** |
| 2 | 304-6-28 | **nu-candidate** | 1.430 | 27834 | 109.0 | 1884 | STM → **released** |
| 3 | 707-18-12 | **STM** | 0.738 | 15168 | 123.6 | 3523 | STM — unchanged |
| 4 | 146-60-31 | **nu-candidate** | 1.513 | 7393 | 100.9 | 1212 | STM → **released** |
| 5 | 60-38-31 | **STM** | 1.469 | 45579 | 186.4 | 2906 | |
| 6 | 445-18-34 | **STM** | 1.099 | 10503 | 151.1 | 2422 | |
| 7 | 541-46-22 | TGM | 0.798 | 6672 | 160.7 | 3057 | |
| 8 | 272-2-30 | TGM | 0.790 | 29330 | 422.6 | 20516 | |
| 9 | 105-23-2 | TGM | 1.648 | 7485 | 230.3 | 3407 | |
| 10 | 105-23-5 | **STM** | 1.505 | 10352 | 117.0 | 1405 | |
| 11 | 105-23-21 | TGM | 0.904 | 20353 | 439.1 | 4760 | |
| 12 | 36-77-17 | nu-candidate | 1.216 | 5490 | 69.8 | 1360 | nu-candidate — unchanged |
| 13 | 966-2-22 | **nu-candidate** | 0.292 | 6947 | 89.7 | 1824 | STM → **released** |
| 14 | 921-29-10 | TGM | 1.483 | 2320 | 67.9 | 170 | TGM — unchanged |
| 15 | 921-29-41 | TGM | 1.540 | 7883 | 160.5 | 3327 | |
| 16 | 890-21-16 | TGM | 0.703 | 32814 | 340.5 | 11153 | |
| 17 | 890-21-39 | TGM | 0.968 | 9992 | 295.6 | 10938 | |
| 18 | 651-84-12 | nu-candidate | 1.260 | 21648 | 141.7 | 3197 | |
| 19 | 651-84-34 | **STM** | 1.357 | 11881 | 256.6 | 5806 | |
| 20 | 993-78-14 | TGM | 1.067 | 7001 | 207.1 | 5316 | |
| 21 | 44-40-14 | TGM | 0.410 | 11136 | 370.6 | 10628 | |
| 22 | 411-27-8 | TGM | 0.413 | 18146 | 53.9 | 33 | |
| 23 | 710-80-44 | **STM** | 0.674 | 29371 | 234.7 | 4208 | |
| 24 | 658-38-25 | TGM | 0.449 | 27868 | 128.1 | 2140 | TGM — unchanged |

Second in-beam bundles: idx 5 (nu-candidate, 0.7 cm), idx 7 (nu-candidate,
47.5 cm, fully contained), idx 19 (`no-bundle`), idx 21 (nu-candidate, 60.2 cm).

## 3. The doc-93 overlap is an unplanned regression test

All 8 doc-93 events are entries 1, 2, 3, 4, 12, 13, 14, 24 of this file.
`scripts/dbg25_vs_doc93.py` parses doc 93's §3 table **out of the markdown**
(not retyped) and compares it field by field with this round:

```
8 events x 6 fields = 48 comparisons: 44 same, 4 differ
verdict changes: ['146-60-31', '304-6-28', '827-27-4', '966-2-22']
expected changes (the 4 doc-94 guard releases): [same four]
VERDICT: OK
```

So t0, flash PE, main length, main points and bundle count reproduce on **all
8 of 8** through a completely fresh extraction, staging, imaging, Q/L and PR
run, and the *only* four differences are the intended verdict flips. Each is
attributable to a named guard fire in the log:

| bee_idx | RSE | guard | evidence from the run log |
|---:|---|---|---|
| 1 | 827-27-4 | `entry_rise_guard` | 8.4 cm of 2.43 MIP anchored at the boundary decaying to a 0.99 MIP body (10.3 cm extra MIP track); path turns **32°** at L=73.9 cm |
| 2 | 304-6-28 | `entry_rise_guard` | 19.4 cm of 3.25 MIP → 1.52 MIP body (13.9 cm extra); turns **32°** at L=70.2 cm |
| 4 | 146-60-31 | `vertex_hadron_guard` | 14.8 cm prong at 1.65 MIP, straightness 0.861, 144° to drift |
| 13 | 966-2-22 | `vertex_hadron_guard` | 14.9 cm prong at 2.47 MIP, straightness 0.971, 90° to drift |

`707-18-12` (idx 3) stays STM, which is the documented outcome: doc 94 §14
proves it is unreachable by this mechanism — its entry is 1.04 MIP on a 0.84
MIP body with a first-window median exactly at the 1.00 MIP floor, so firing
would need a bar at or below MIP. Its real signature is a 32° turn 44 cm
before the stop at MIP charge (a mid-track two-prong vertex), which
`vertex_kink_guard` misses on all three of its conditions. That is round 4,
not a defect of this run.

## 4. Guard behaviour on the 17 new events

Across all 25 events the log holds exactly **four** `_guard: cluster … rejected:`
lines — the four in §3. **No guard fired on any of the 17 events doc 93 did not
contain.** The 5 new STMs (60-38-31, 445-18-34, 105-23-5, 651-84-34,
710-80-44) were all evaluated by the STM tagger (`stmfit=eval`) and none was
released.

**This is a FIRE-RATE check, not a contamination check.** 0 fires in 17 unseen
MC events says the guards are quiet on new data, consistent with doc 94
§13.12's 2 releases in 3067 data events. It says **nothing** about false
releases, because there are no truth labels on these 17 — a guard that never
fires cannot be caught releasing a cosmic here. Nor does it say the guards are
right to stay quiet: if any of the 5 new STMs is in truth a neutrino, that is a
*miss*, and this run would look exactly the same. The hand scan is what
resolves either direction.

## 5. What to look at in the Bee set

- **The 4 released events (idx 1, 2, 4, 13)** — do they look like neutrino
  interactions? These are the shipped guards' output on MC.
- **The 6 STM events (idx 3, 5, 6, 10, 19, 23)** — the tagger's remaining STM
  calls. idx 3 is the known round-4 target; the other five are new.
- **idx 18 (651-84-12)** — a new nu-candidate at 141.7 cm and 21648 PE.
- **idx 22 (411-27-8) — sparse, worth an eye.** Its in-beam bundle is TGM on
  **33 points over 53.9 cm** (~0.6 pts/cm) while the same event holds bundles
  of 14005, 5732 and 5128 points; it also carries the table's only in-beam
  `lm=2`. Sparse in-beam mains that go TGM do occur here (idx 14, 921-29-10, is
  170 pts over 67.9 cm and doc 93 recorded the same), so this is flagged as a
  scan target, not diagnosed as a defect.
- **The 6 events carrying Q/L + PR clustering only** (idx 3, 8, 11, 14, 22,
  24): `TaggerCheckNeutrino` selected no candidate, so `track_fit-global`,
  `shower_track-global`, `vertices-global` and `mc` are absent by design, not
  missing by fault (doc 93 §6). The other 19 carry all ten layers.

## 6. Provenance and pins

- Binary pinned to `~/tmp/doc94r3b-libsnap`; all 19 `libWireCell*.so`
  md5-identical to `local/lib` at launch **and** at completion.
- PR cfg tree pinned to `~/tmp/dbg25-cfgsnap`. Stages A1/A2 read
  `toolkit/cfg` directly (`run_img_evt.sh` / `run_ql_evt.sh` hardcode it ahead
  of `$WIRECELL_PATH`), so the live tree was hash-fenced instead: **454 cfg
  files, byte-identical before and after the run.**
- `prod_cfg_gate.py`: PASS 21/21 vs `ref/prod-2026-09-03` before launch.
- Operating point actually used, read back from the compiled per-event config
  `work-dbg25a-pr/pr_evt4/.wct-cfg-evt4.json`: `entry_rise_guard = True`,
  `guard_entry_frac = 1.3`, `guard_entry_min_cm = 5`, `guard_entry_max_cm = 60`,
  `guard_entry_min_len_cm = 70`, `guard_entry_kink_deg = 22`,
  `vertex_hadron_guard = True`. No `pos_offset` key ⇒ `sim`.
- **Group b's inputs proved, not assumed** (`scripts/dbg25_verify_frames.py`):
  group b was built by a path nothing had run before (extract 5 single-event
  archives, re-tar, run a 5-event sample dir), and an event whose imaging read
  the *wrong frame* would still show the right RSE — the RSE comes from the
  opflash tar, not the frame. So every member of both group archives was
  compared byte-for-byte against the staged single-event extraction it came
  from: **125 frame members across all 25 entries, 0 mismatches.** This matters
  because two of the six STM calls being scanned (idx 5, 19) and three
  collision partners live in group b.
- Bee set content-verified after upload: the set page lists event indices
  0..24, and inside the zip **all 25 `img-global` layers hash differently** —
  the check that the five colliding bare ids really carry ten distinct events.
- `scripts/dbg25_table.py` cross-checks each `nusel-evt<ID>.tsv`'s own RSE
  against the manifest row and aborts on mismatch, so "wrong work root" cannot
  pass silently.

## 7. Reported, not fixed here

- `run_reco1_dump.sh` will happily write a combined archive with duplicate
  members when a file spans runs (§1). It reports 25 events in its closing
  "Events:" list — the duplicates are visible only if you sort them. A
  uniqueness assertion belongs in that script, next to its existing
  "already holds an extraction" refusal. Not changed here: it is a live
  production script and this round did not need it changed.

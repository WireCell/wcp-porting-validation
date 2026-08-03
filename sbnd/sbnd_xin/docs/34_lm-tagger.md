# LM (light-mismatch) tagger at the end of QLMatching (doc 34)

New per-bundle verdict closing the doc-24 porting gap: the toolkit had no
counterpart of the prototype's `check_LM` (`event_type` bit 1, the third
cosmic-rejection label next to TGM/STM).  `QLMatching` now judges every FINAL
matched bundle by **per-drift-side KS shape distance + pred/meas
normalization** and stamps the verdict as cluster scalar `lm_flag`
(0 = pass, 1 = low-energy, 2 = light mismatch), read by `nusel_extract.py`
into a new `lm` column with label priority **TGM > STM > LM** (an in-beam
bundle failing LM is demoted nu-candidate → `LM`; out-of-beam labels are
unchanged, the viewer badges them `lm`).

Everything is behind the default-OFF `lm_tagger` knob (C++/jsonnet defaults
false; `-lm` runner flag opts in) — knob-off is byte-identical.

## Symptom

**evt286021 bundle grp 7 (t = +1.158 µs, IN BEAM) main 8**: a 2565.6 PE
beam-window flash matched to an 8.3 cm / 141-pt cluster predicting only
437 PE.  The charge cannot explain the flash — the flash is a cosmic
coincidence and the cluster an accidental — yet the chain labeled it
**nu-candidate** (hand-scan label: `LM`, tag mcp10-ctpcfix onward).  Doc 24
established that nothing in the toolkit computes LM and that the inputs
(`ks_dis`, predicted/measured PE per channel, `flag_close_to_PMT`,
`flag_at_x_boundary`) exist only inside `QLMatching` — so the tagger belongs
at the end of QLMatch, not in a pctree-based PR visitor (owner-confirmed).

## Design (prototype → SBND)

Prototype `check_LM` (`prototype_base/2dtoy/src/ToyFiducial.cxx:598-660`,
uBooNE: 32 PMTs on ONE side) cuts on the whole-flash KS distance,
`log10(total_pred/total_meas)`, `flag_close_to_PMT` / `flag_at_x_boundary`
relaxations, and a low-energy exemption (`total_pred < 25 PE` or
`cluster_length < 10 cm` ⇒ "low energy", never LM).  SBND differences:

1. **Two drift volumes, PDs behind both anodes** → the KS and the
   normalization are evaluated **per drift side** (OpDet `x` vs the cathode
   plane, the `flash_phys_side` convention).  Each judged side gets its own
   KS over that side's channels only (within-side renormalized CDFs, fixed
   channel-index order, masked channels excluded from BOTH distributions).
2. **Cathode leakage light**: the photon library / semi-analytical model does
   not model the small light leaking to the far side, so a side is judged
   ONLY when its predicted PE reaches `lm_side_pred_min` (25 PE) — the far
   side's measured PE alone is never read as a mismatch.
3. **Bright flash overrides the low-energy exemption** (owner decision,
   diverges from the prototype): a tiny-prediction cluster matched to a
   ≥ `lm_flash_pe_bright` (1000 PE) flash is exactly the mismatch this tagger
   exists for (the evt286021 case is 8.3 cm — the prototype guard would have
   called it "low energy").  When no side reaches `lm_side_pred_min`, the
   totals are judged instead.
4. **Flash-group resolution** (`apply_lm_verdicts`, a cross-run pass): the
   downstream MABC `examine_bundles` (use_flash_t0) merges EVERY cluster
   matched inside one ±80 ns flash group — across BOTH drift sides — into one
   output cluster, so per-bundle `lm_flag` stamps collide on the composite
   and an arbitrary writer wins.  The stamped verdict is the group's
   **largest-total-predicted-light** bundle's verdict; grouping is the same
   single-linkage `flash_group_window` rule as `store_flash_groups`.  Three
   observed collision classes drove this: a 14-PE-pred grafted speck
   relabeled the 22434-PE through-goer of evt286021 output cluster 16 to
   lm=2; the genuine in-beam nu-candidates of evt287825 (t = 1.41 µs) /
   evt288639 (t = 1.17 µs) — whose flashes are dominated by healthy
   119 / 118 cm bundles — were demoted by subdominant fragments; and
   evt284349's 424 cm TGM read lm=1 from the OTHER side's dim coincident
   flash (gid 9, 306 PE, 27 ns away — same group, different APA run).
5. **Two cut regimes** (tuned on the 20 events).  The healthy long-track
   population under-predicts systematically — judged-side
   `log10(pred/meas)`: median −0.16, 1st percentile −0.99, minimum −1.11;
   per-side KS up to 0.507 on hand-scanned-good crossers — so uBooNE-tight
   cuts would tag a large fraction of good cosmics.  The LONG regime is
   therefore loose (only egregious deficits tag), while the SMALL regime
   (cluster below the prototype low-E thresholds, flash bright) — where the
   fake in-beam nu-candidates live — keeps prototype-tight cuts: a sub-10 cm
   cluster physically cannot explain a kilo-PE flash unless its prediction
   says so.

Verdict per bundle (`QLMatching::check_light_mismatch`, prototype citations
inline):

```
small  = total_pred < lm_pred_pe_min (25 PE)  OR  length < lm_length_min (10 cm)
if small AND total_meas < lm_flash_pe_bright (1000 PE):   verdict 1 (low-E)
relax  = close_to_PMT OR at_x_boundary
ks_max      = small ? lm_small_ks_max (0.45)
                    : relax ? lm_ks_max_relax (0.70) : lm_ks_max (0.55)
lograt_min  = small ? lm_small_lograt_min (-0.55)
                    : relax ? lm_lograt_min_relax (-1.6) : lm_lograt_min (-1.3)
for each side s with pred_s >= lm_side_pred_min (25 PE):
    fail if ks_s > ks_max
    fail if log10(pred_s/meas_s) < lograt_min  or  > lm_lograt_max (2.0)
(no judged side: apply the lograt tests to the totals)
verdict = fail ? 2 (LM) : 0 (pass)
```

`length` is `Facade::Cluster::get_length()` — the same uvwt-range formula as
the prototype's `cluster_length`.

## Fix (files)

- `match/inc/WireCellMatch/QLMatching.h`, `match/src/QLMatching.cxx`:
  `lm_tagger` knob + cut params (all config keys: `lm_pred_pe_min`,
  `lm_length_min`, `lm_flash_pe_bright`, `lm_side_pred_min`, `lm_ks_max`,
  `lm_ks_max_relax`, `lm_lograt_min`, `lm_lograt_min_relax`, `lm_lograt_max`,
  `lm_small_ks_max`, `lm_small_lograt_min`),
  `check_light_mismatch()`, flash-resolved stamping in `apply_matched_t0s`
  (`lm_flag` on main + associated clusters, "LM verdict" debug lines), per-
  bundle `lm`/`lm_ks`/`lm_pred`/`lm_meas`/`lm_length_cm` (each bundle's OWN
  verdict, pre-resolution) + `quality_params.lm` in `dump_calib` (all keys
  emitted only when the knob is on).
- `cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet`: `lm=false` + `lm_params`
  args on `matching()`/`matching_joint()`, key-suppressed.
- `sbnd_xin/qlmatching.jsonnet` (shim), `wct-clus-matching-perevt.jsonnet`:
  `lm` TLA threaded through.
- `sbnd_xin/run_ql_evt.sh`: `-lm` / `SBND_QL_LM=1`.
- `sbnd_xin/run_nusel_evt.sh`: `-lm` (passes `-lm -calib` to the Q/L step it
  launches; an existing pctree is reused untouched — use a fresh work root).
- `sbnd_xin/nusel_extract.py`: `lm` column from the `lm_flag` scalar (−1 when
  absent ⇒ old trees unaffected), label demotion in-beam nu-candidate → `LM`,
  event label counts LM as cosmic-tagged.
- `sbnd_xin/nusel_display/nusel_scan_viewer.py`: verdict column
  `tgm/stm/fc[/lm]`, `lm` badge on out-of-beam mismatches, LM row in the
  details pane, lm in the exported scan JSON; prev-baseline compare now also
  tints on an auto-label change (so the LM demotion is amber) and tolerates
  baselines without the column.
- `sbnd_xin/scripts/analysis/ql/lm_tune.py`: offline cut tuning from the calib dumps (replicates
  the C++ verdict; `--scan key=val` re-evaluates without a rerun).

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild        # + M1 freshness proof
cd sbnd_xin
# fresh roots; per-event imaging symlinked from the ctpcfix roots
mkdir -p work-mcp10-lm work-mcp1000-lm
for d in work-mcp10-ctpcfix/evt*/;   do ln -sfn $PWD/${d%/} work-mcp10-lm/$(basename $d); done
for d in work-mcp1000-ctpcfix/evt*/; do ln -sfn $PWD/${d%/} work-mcp1000-lm/$(basename $d); done
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10-lm \
  ./run_nusel_evt.sh data all -chord -rescue -rescue-chord -fvz 5 -lm
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for e in 10 11 12 13 14 15 16 17 18 19; do
  SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-lm \
    ./run_nusel_evt.sh data 1 -chord -rescue -rescue-chord -fvz 5 -lm
done
python3 nusel_extract.py --merge work-mcp1000-lm/nusel_evt*/nusel-evt*.tsv \
  --out work-mcp1000-lm/nusel-table.tsv --events-out work-mcp1000-lm/nusel-events.tsv
# tuning study:
python3 scripts/analysis/ql/lm_tune.py work-mcp10-lm work-mcp1000-lm --out /home/xqian/tmp/lm_tune
# viewer:
nusel_display/serve_nusel_scan.sh 5010 --tag mcp10-lm \
  --prev ../work-mcp10-reschord:mcp10-reschord \
  --prev ../work-mcp10-tgmfv:mcp10-tgmfv --prev ../work-mcp10-ctpcfix:mcp10-ctpcfix \
  ../work-mcp10-lm ../work-mcp1000-lm
```

## Verification

**Byte-identical, knob off.**
- Compiled config: HEAD cfg + HEAD sbnd_xin jsonnets vs new trees at the
  production TLAs (joint, pmt_nl, main_flag, no lm) — `cmp` PASS (scratch
  `lmcfg/{old,new}-off.json`).  Knob-on compiled config gains exactly
  `lm_tagger: true` on `QLMatching:matching_joint`.
- End-to-end: evt286021 rerun with the new `libWireCellMatch.so`, no `-lm`,
  into `work-mcp10-lm-offgate` — `mabc-all-apa.zip` member hash `d4eef908…`
  and `pctree-evt286021.tar.gz` member hash `7f334bad…` identical to
  `work-mcp10-ctpcfix`.
- `./build/match/wcdoctest-match`: 4/4 cases, 36/36 assertions.

**Knobs on** (`-chord -rescue -rescue-chord -fvz 5 -lm`, tags
`work-mcp10-lm` / `work-mcp1000-lm`, QL re-run per event with `-lm -calib
-save-pctree`; imaging symlinked from the ctpcfix roots):

- **No collateral change**: `tgm/stm/fc` columns identical row-for-row to the
  reschord baseline on all 20 events; the ONLY `label` change on the whole
  sample is the target demotion
  `evt286021 main 8 / gid 1000007: nu-candidate → LM`.
- Compiled config gains `lm_tagger: true` on `QLMatching:matching_joint`.

## Cut tuning (lm_tune.py, 397 auto-selected bundles / 2048 candidates)

Judged-side (`pred_side ≥ 25 PE`) population of auto-selected bundles:

| class | n | log10(pred/meas) p1 / p50 / p99 | KS p50 / p95 / max |
|---|---|---|---|
| clean, len ≥ 50 cm | 93 | −0.99 / −0.16 / +0.31 | 0.097 / 0.360 / 0.507 |
| relax, len ≥ 50 cm | 80 | −0.67 / −0.21 / +0.09 | 0.131 / 0.343 / 0.424 |
| clean, len < 10 cm | 91 | −2.92 / −0.92 / +0.03 | 0.245 / 0.720 / 0.835 |

The healthy long population under-predicts by a factor ~1.5–10 (SBND Q→L
calibration + PE-error model, cf. docs 13/15), so uBooNE's −0.55 floor and
0.25 KS ceiling would tag dozens of hand-scanned-good cosmics: the LONG
cuts sit outside that population (ks 0.55 / 0.70, lograt −1.3 / −1.6).  The
mismatch class — 48 per-bundle lm=2 verdicts — is entirely sub-≈30 cm
clusters (mostly 1–10 cm specks) matched to 1.2k–72k PE flashes at lograt
−0.7 … −3.1, cleanly separated.  Scatter:
`lm_tune_scatter.png` (repro command above; the target sits at
ks₁ = 0.482 / lograt −0.77, failing both SMALL cuts).

## Final table (flash-group-resolved lm column)

In-beam bundles, all 20 events: 17 nu-candidates and 2 TGMs keep their
labels with lm = 0; **1 LM = evt286021 main 8** (8.3 cm / 141 pts, flash
2565.6 PE, pred 437 PE — ks₁ 0.482, lograt −0.77).  The two big in-beam
bundles a naive per-bundle stamp had demoted (evt287825 main 5, 175.6 cm,
t = 1.410 µs; evt288639 main 8, 79.9 cm, t = 1.174 µs) stay nu-candidates —
their flash groups are dominated by healthy bundles (ks 0.101 / 0.094).

Out-of-beam composites with lm = 2 (label untouched; viewer badges `lm`):

| bundle | len (cm) | flash PE | dominant pred PE | reading |
|---|---|---|---|---|
| 284657 m3 gid 5 | 0.5 | 17239 | 10 | unexplained bright flash |
| 285999 m17 gid 1000005 | 13.2 | 2070 | 250 | short stub, −0.92 |
| 286021 m7 gid 1000003 | 3.2 | 8766 | 326 | speck, −1.42 |
| 286065 m6 gid 1000008 | 0.9 | 11171 | 32 | degenerate speck (also TGM) |
| 286241 m7 gid 1000012 | 1.6 | 1212 | 18 | speck (also TGM) |
| 286681 m9 gid 1000000 | 1.2 | 30580 | 25 | unexplained bright flash |
| 288397 m2 gid 11 | 71.0* | 1209 | 251 | sparse stub, −0.68 (small by uvwt pred) |
| 288639 m9 gid 5 | 46.4* | 7964 | 1465 | fragment group, −0.69 |
| 288727 m7 gid 1000000 | 30.7 | 25310 | 871 | −1.46 |
| 288727 m9 gid 1000002 | 14.4 | 23893 | 851 | −1.39 |

(*table len is the Bee dominant-component farthest-pair; the C++ regime uses
the main cluster's uvwt `get_length()`.)

`lm = 1` rows are 1–3 cm specks (or sparse low-pred clusters) on flashes the
group's dominant bundle also cannot judge — the prototype "low energy"
population, deliberately NOT called LM.

Viewer: :5010 tag `mcp10-lm`, `--prev` reschord → tgmfv → ctpcfix (the LM
demotion tints amber via the auto-label compare).

## Round 2: good-shape guard (`lm_shape_ks_max` / `lm_shape_lograt_min`)

**Symptom** (owner hand-scan of the round-1 badges): two out-of-beam LM
badges are NOT mismatches — `evt285999 m17` (t = 524.8 µs, ks₁ 0.200,
lograt −0.92) and `evt286021 m7` (t = 1485.5 µs, `close_to_PMT`, ks₁ 0.082,
lograt −1.42).  Both have activity very close to the PMTs: the light
*pattern* agrees (excellent judged-side KS) but the *strength* is
under-predicted — the library's missing near-field/leakage response — and
the SMALL regime applies no `close_to_PMT` relaxation, so the −0.55
normalization floor alone condemned them.

**Fix** (owner-chosen cuts): a normalization-ONLY failure is rescued when
every norm-failing judged side has `ks < lm_shape_ks_max` (0.25) AND
`lograt ≥ lm_shape_lograt_min` (−1.8).  A KS failure anywhere disables the
guard; over-prediction (`lograt > lograt_max`) and the totals-fallback
(no judged side ⇒ no shape to trust) are never rescued.  Both regimes.
The genuine-mismatch class keeps bad shape (the evt286021 main 8 target:
ks₁ 0.482) or an absurd deficit (specks at lograt −2 … −3.4), so the guard
is orthogonal to it.

**Verification** (tags `work-mcp10-lm2` / `work-mcp1000-lm2`, imaging
symlinked from ctpcfix, same `-chord -rescue -rescue-chord -fvz 5 -lm`
chain; fresh off-gate `work-mcp10-lm2-offgate`):

- Knob-off end-to-end: evt286021 no `-lm` with the round-2
  `libWireCellMatch.so` — `mabc-all-apa.zip` `d4eef908…` and
  `pctree-evt286021.tar.gz` `7f334bad…` identical to `work-mcp10-ctpcfix`
  (same digests as the round-1 gate).  `wcdoctest-match` 4/4 (36
  assertions).  The shape knobs are C++ defaults under the OFF `lm_tagger`
  — no jsonnet change.
- All 214 table rows: `tgm/stm/fc` identical to round 1; **0 label flips**;
  the target `evt286021 main 8` stays `lm=2 / LM`.
- Per-bundle verdicts 46 → 34 lm=2 (12 rescued, low-E count unchanged).
  Flash-group badges flipped 2→0 on exactly 5 rows, all out-of-beam: the
  two owner-flagged ones plus `288397 m2` (ks 0.051, −0.68), `288639 m9`
  (ks 0.229, −0.69), `288727 m7` (ks 0.141, −1.46) — the same
  good-shape/weak-strength signature.  Badges retained: dominant-pred
  specks (`284657 m3`, `286065 m6`, `286681 m9`), `286241 m7` (lograt
  −1.82, just past the floor) and `288727 m9` (ks 0.472 — bad shape).

Repro delta:

```bash
mkdir -p work-mcp10-lm2 work-mcp1000-lm2
for d in work-mcp10-ctpcfix/evt*/;   do ln -sfn $PWD/${d%/} work-mcp10-lm2/$(basename $d); done
for d in work-mcp1000-ctpcfix/evt*/; do ln -sfn $PWD/${d%/} work-mcp1000-lm2/$(basename $d); done
# then the same two run_nusel_evt.sh invocations as above with the lm2 roots
nusel_display/serve_nusel_scan.sh 5010 --tag mcp10-lm2 \
  --prev ../work-mcp10-lm:mcp10-lm --prev ../work-mcp10-reschord:mcp10-reschord \
  --prev ../work-mcp10-tgmfv:mcp10-tgmfv --prev ../work-mcp10-ctpcfix:mcp10-ctpcfix \
  ../work-mcp10-lm2 ../work-mcp1000-lm2
```

## Caveats

- The LONG-regime floors (−1.3 / −1.6) sit just past the observed healthy
  minimum (−1.11); a genuinely dim but real long match beyond that would tag
  — unless its shape qualifies for the round-2 good-shape guard (ks < 0.25,
  lograt ≥ −1.8).  All cuts are config keys (`lm_*`), so retuning is
  config-only.
- The leakage-light protection is one-sided: a side is only judged when its
  own prediction ≥ 25 PE; no explicit leakage fraction is modeled.
- The per-APA (non-joint) node resolves flash groups per side only; SBND
  production uses the joint node, where the resolution spans both sides.
- Only in-beam LM changes `label`; if hand-scan confirms the out-of-beam
  lm = 2 class, a future round could let LM demote `not-tagged` rows too.

# PDVD time-over-threshold (ToT) saturation fill in OpDecon: default-OFF knob, gates, and the doc 11 / 12 / hand-scan reruns

**Status 2026-09-23: implemented (toolkit `d63f2aef`), default OFF, NOT flipped.** The knob is `OpDecon saturation_repair_mode: "tot"`, set
through the runner with `PDVD_SAT_REPAIR_MODE=tot`. Its checks:

- **Knob off.** Byte-identical: compiled config on 3 argument sets; light archives 120/120 against the pre-change
  library and against production `_keep`.
- **Knob on.** It does what doc 30 designed: C++ matches the Python replica to about 2 %. It also recovers the PE that
  production loses at hit level on deep rails (§5).
- **Owner's hand scan.** ToT FAILS the pre-registered doc 23 rule on the existing record (§7). Every other gate passes.
- **Recommendation.** Keep twoside in production, and do not flip ToT on this record. Two things would change this (§8):
  - adjudicating the flash/cluster pairs ToT changes;
  - fixing the hit-level loss that production twoside hides.

Doc 30 (`30_pdvd-saturation-tot-vs-repair.md`) is the study this implements. This doc changes no production default.

**Repro:**
```
# 0. toolkit (apply-pointcloud): flash/ C++ + doctest, cfg/.../protodunevd/{flash,wct-light-reco}.jsonnet + pdvd-tot-shapes-v2.json
wcbuild && ./wcb build --target=wcdoctest-flash && ./build/flash/wcdoctest-flash      # 16 cases incl. 4 new
# 1. shape refit on the production (v2) templates -> d31/{shape,sim,develop,bench,data}.txt, pics/*_v2.png
cd pdvd/docs/qlmatch && python3 scripts/saturation_tot_study.py --spe v2 --all
python3 scripts/export_tot_shapes.py                       # -> toolkit cfg pdvd-tot-shapes-v2.json
(cd scripts && python3 d31_tot_doctest_ref.py)             # the doctest's reference numbers
(cd scripts && python3 d31_refit_starts.py 1051 1071 1011) # d31/refit_starts.txt (why the fit is multi-start)
# 2. light arms, 120 events of pdvd/stm/events.txt, production-light arguments (= _keep)
cd scripts
ARM=_g31old PIN=/home/xqian/tmp/p31/libpin_flash_old ./d31_light_arms.sh   # pre-change libWireCellFlash
ARM=_g31off ./d31_light_arms.sh ; ARM=_g31offb ./d31_light_arms.sh        # knob off, twice (null pair)
ARM=_q31tot2 ENVS="PDVD_SAT_REPAIR_MODE=tot" ./d31_light_arms.sh           # knob on (multi-start shapes)
ARM=_f31ts18  ONLY_RUN=39252 ENVS="PDVD_LIGHT_FRAMES=1" ./d31_light_arms.sh
ARM=_f31tot18 ONLY_RUN=39252 ENVS="PDVD_SAT_REPAIR_MODE=tot PDVD_LIGHT_FRAMES=1" ./d31_light_arms.sh
#    gate hashes: abtest/hash_archive.py over every opflash_pdvd-wct.tar.gz -> d31/gate/hashes_<arm>.txt
# 3. Q/L arms on the q29flip clustering pin
ARM=q31ctl LIGHT=_g31off ./d31_ql_arms.sh ; ARM=q31tot2 LIGHT=_q31tot2 ./d31_ql_arms.sh
# 4. analyses (W = scratch work root: symlinks to the arm dirs + zstd -d copies of the 18 _keep calib dumps)
cd pdvd; W=/home/xqian/tmp/p31/wroot
python3 ql_display/ql_agree_score.py --tag q31ctl  --work-root $W --truth-uid-map-tag keep     # + q31tot2
python3 docs/qlmatch/scripts/d31_time_map.py --b _q31tot2 --calib-tag-b q31tot2 --work-root $W --out docs/qlmatch/d31/time_map_ctl_to_tot2.json
python3 ql_display/ql_agree_score.py --tag q31tot2m --work-root $W --truth-uid-map-tag keep --truth-time-map docs/qlmatch/d31/time_map_ctl_to_tot2.json
python3 docs/qlmatch/scripts/d31_scan_pairs.py --base q31ctl --arm q31tot2 --arm q31tot2m --work-root $W   # d31/scan_pairs.txt
python3 docs/qlmatch/scripts/d31_scan_subset.py --work-root $W --tag q31ctl --tag q31tot2                  # d31/scan_subset.txt
python3 docs/qlmatch/scripts/d31_sat_terms.py <tag> $W/039252_*_<tag>/calib-evt*.json                     # d31/sat_terms_18evt.txt
(cd ql_light_calib && python3 fit_qtol_crossers.py --tag q31ctl)                                           # + q31tot2
python3 docs/qlmatch/scripts/d31_flash_pe.py --b _q31tot2                                                  # d31/flash_pe.txt
python3 docs/qlmatch/scripts/d31_hit_split18.py [3000 6000]                                                # d31/hit_split18*.txt
(cd docs/qlmatch/scripts && python3 d31_frames_closure.py && python3 d31_roi_closure.py)                  # d31/*_closure.txt
(cd docs/qlmatch && python3 scripts/d31_run_census.py)                                                     # d31/run_census.txt
```

## 1. What was asked, and what was pre-registered

The owner asked for three things:
- implement ToT as a default-OFF option of the waveform-repair step, with a gate showing the output is unchanged
  when off;
- redo the doc 11 §6 and doc 12 matching numbers with it on, since those were tuned on twoside;
- check the Q/L matching against the owner's hand scans.

The Q/L rules were written into `d31/prereg.md` before any Q/L arm ran:
- preconditions;
- the eligible population;
- the doc 23 scan rule, on the unmapped join;
- the chi2-inflate trigger;
- the doc 12 "report, don't retune" rule.

## 2. The shapes shipped: refit on the production v2 templates, multi-start

Doc 30 fit the shape model on the **v1** SPE templates. Production light (`_keep`, and every `PDVD_SPE_V2=1` run) uses
**v2**. Of the 16 cathode channels, only **ch 1051** differs between the two files; doc 30's fit was degenerate there
because the v1 template was anomalous. `saturation_tot_study.py --spe v2` therefore refits everything:
- cache: `/home/xqian/tmp/sat_tot_v2`;
- records: `d31/`;
- figures: `pics/*_v2.png`.

`--spe v1` still reproduces doc 30 unchanged.

**The fit had to be made multi-start.** With the fit's defaults, several channels stopped at the start point
`[0.3, 0.3, 8, 80, 1]`: v2 ch 1051 at 0.301 / 0.299 / 8.00 / 80.0 / 1.00, and 1071 and 1011 likewise.
- **What it looks like.** Restarts from 6 points land on different parameters with similar rms, τ_s anywhere from 80
  to 122 ticks (`d31/refit_starts.txt`, `least_squares` status 3 = x-step tolerance).
- **The fix.** `--spe v2` now takes the lowest-cost fit of 6 starts, with `xtol = ftol = 1e-12` and an `x_scale`. The
  per-pulse prompt-fraction spread uses the same tolerances. Its sim σ_ff becomes 0.017; the single-start value 0.034
  was also an early stop.
- **Superseded records.** The single-start v2 records are kept in `d31/singlestart/`. The single-start light / Q/L arm
  (`_q31tot` / `q31tot`) is superseded and quoted only in §7.

The multi-start v2 shapes pass doc 30's gates (`d31/sim.txt`, `bench.txt`, `data.txt`):

| check | doc 30 (v1) | doc 31 (v2, multi-start) |
|---|---|---|
| sim vs held-out data widths, d 1.11–6.67 | within 3.1 % | within 3.1 % (the sim's [16,84] band is narrower than data's) |
| sim bench `tot_fill`, d 1.1–20 | 0.99–1.00 | 0.990–0.999 |
| data synthetic clips `tot_fill`, held out | 0.951 at d 6.7 | 0.946 [0.886, 1.023] at d 6.7; ≤ 5.4 % bias to d 6.7 |
| real both-railed pair consistency q | within 5 % | 0.938–1.006, all bins pass |

`scripts/export_tot_shapes.py` writes `toolkit/cfg/pgrapher/experiment/protodunevd/pdvd-tot-shapes-v2.json`:
- 16 channels, as `[ff, bi, tau_i, tau_s, sigma]`, rounded to 1e-6;
- degenerate fits (ff ≥ 0.99) would be left out, but none is.

## 3. The C++ (toolkit `flash/`) and config

**`OpDecon`.** Three new keys, each round-tripped in `default_configuration()`:
- `saturation_repair_mode`: `"twoside"` by default, otherwise `"tot"`; anything else throws;
- `tot_shape_file`: empty by default; required in tot mode;
- `tot_merge_gap`: default 2.

`repair_runs` (twoside) is **byte-for-byte untouched**. The new `repair_runs_tot` does the following:
- it merges rail runs that are at most 2 samples apart;
- it ToT-fills each merged run;
- it falls back to twoside on that run's sub-runs in three cases:
  - the channel has no shape;
  - the run touches a trace edge;
  - ToT is at least the table's widest level (534 ticks on the test shape).
- The last case deviates from the Python, which clamps λ at 1e-3. The fallback is there because clamping would fill
  a 600-tick run at 1000 × R.

Detection, the `cmm["saturation"]` masks, the flags and `overflow_to_rail` are identical in both modes.

**Static helpers** mirror the Python functions:

| C++ | Python (`saturation_tot_study.py`) |
|---|---|
| `tot_model` | `kernel` + `model` (template extended by the τ_fall exponential, as `Channel.tshape`) |
| `tot_shape` | `Shape` (2999 log levels, fractional crossings) |
| `tot_level_for` | `level_for` (`np.interp` on the reversed table) |
| `tot_fill_run` | the `methods_pe` fill: A = R/λ, R = rail + 0.5 − pedestal, start at the up-crossing index |

**Doctest** (`flash/test/doctest_opdecon_tot.cxx`). Four cases, with reference numbers from
`scripts/d31_tot_doctest_ref.py`:
1. model, table and level values equal the Python to 1e-6;
2. the width at the solved level equals ToT to 1e-3;
3. a clipped noise-free pulse is filled back:
   - fill area matches the Python to 1e-5;
   - area ratio 0.985–0.988, the integer-ToT discretisation;
   - no sample is lowered;
   - samples outside the run are untouched;
4. a run wider than the table is refused and the waveform is untouched.

`wcdoctest-flash`: 16 cases / 6129 assertions pass. Freshness: `local/lib/libWireCellFlash.so` 12:23 is newer than
`OpDecon.cxx` 12:22.

**jsonnet** (PDVD only):
- `flash.jsonnet opdecon(...)` gains `saturation_repair_mode='twoside'` and `tot_shape_file=''`, key-suppressed
  unless repair is on and the mode is not twoside.
- `wct-light-reco.jsonnet` gains the top-level argument `saturation_repair_mode='twoside'`:
  - it applies on the **cathode branch only**; membrane and PMT snippets were untested in doc 30 and keep twoside;
  - it requires `spe_v2` and errors otherwise;
  - an unknown value errors.

**Runner.** `run_light_evt.sh` gets `PDVD_SAT_REPAIR_MODE`. It passes `-A saturation_repair_mode=…` only when the
value is not empty or twoside.

## 4. Gates (labels and hash files in `d31/gate/`)

| gate | result |
|---|---|
| compiled config, knob off: `_keep` args, runner-default args (tail merge on), minimal args, explicit `saturation_repair_mode=twoside` | **byte-identical** to HEAD (`cmp`) |
| compiled config, knob on | `saturation_repair_mode: "tot"` + `tot_shape_file` on `OpDecon:cath` only; mem/pmt unchanged; bogus value → jsonnet error |
| `_keep` args = today's runner defaults + `PDVD_FLASH_TAIL_MERGE=0` | compiled JSON equal to `work/039252_light298567_keep/.wct-light.json` apart from `_pnode` metadata |
| null pair `_g31off` vs `_g31offb` | 120/120 identical (`hash_archive.py`) |
| `_g31old` (pre-change lib pinned) vs `_g31off` (new lib, knob off) | **120/120 identical** |
| pin loaded? `_g31pintot`: old pin + `PDVD_SAT_REPAIR_MODE=tot`, evt298567 | = `_g31off` (the old lib ignores the key), no ToT log line; `_q31tot` differs (`gate/pin_loaded_check.txt`) |
| **precondition** `_g31off` vs production `_keep` | **120/120 identical**, so the Q/L control is production light |
| `_g31off` vs `_q31tot2` | 0/120 identical (the knob acts) |
| smoke, evt298567 | `OpDecon:cath … tot: 111 merged runs ToT-filled, 0 twoside fallback` (doc 14 counts 111 rail runs there) |
| 120 events | 13 304 merged runs ToT-filled, 12 fall back to twoside (`d31/tot_fill_counts.txt`) |
| Q/L closure: `q31ctl` calib dumps vs `q29flip` | **120/120 byte-identical**; hand scan 683 / 97 / 158 = doc 29's production |

**Report on the runner, not a fix.** The runner default is `PDVD_FLASH_TAIL_MERGE=1`, but production Q/L reads the
pre-tail-merge `_keep` light: `stm/run_campaign.sh` and `d29_stm_arms.sh` hard-code `PDVD_LIGHT_SUFFIX=_keep`. A fresh
default light run is therefore not production light.

## 5. The C++ acts as designed, and exposes a hit-level loss in production twoside

**Decon level.** Evt298567, 111 rail runs, per run over [i − 50, j + 300) (`d31/frames_closure.txt`,
`roi_closure.txt`):
- C++ agrees with the Python replica to 1–2 %, in both modes.
- ToT/twoside decon PE falls with depth, as doc 30 predicts, because twoside over-fills:

  | ToT (ticks) | ToT/twoside decon PE |
  |---|---|
  | 1–20 | 1.00 |
  | 20–71 | 0.97 |
  | 71–141 | 0.87 |
  | 141–260 | 0.72 |

- The Python census over all 2777 run-039252 rail runs agrees (`d31/run_census.txt`): replica window PE
  tot_fill/twoside is 0.74 at ToT 141–194, 0.58 at 194–260 and 0.31 above 260.

**Flash level, which goes the other way** (`d31/flash_pe.txt`, 120 events):
- On flashes with a cathode rail, total PE under ToT is at or above twoside's, with an upper tail: p95 1.17, p99 2.04.
- Summed PE rises from 1.68e7 to 1.95e7 (+16 %).
- Flash count falls from 48 600 to 48 377 (−223).

**The reason is OpHitFinder** (`d31/hit_split18.txt`, 18 events, 2786 runs, frames dumped with `PDVD_LIGHT_FRAMES=1`):

| ToT (ticks) | n | hits / decon_roi, twoside | hits / decon_roi, tot | median hits per run ts / tot |
|---|---|---|---|---|
| 71–141 | 553 | 1.003 | 1.000 | 3 / 1 |
| 141–260 | 401 | **0.450** [0.163, 1.002] | 1.000 [0.759, 1.001] | 4 / 1 |
| > 260 | 152 | **0.075** [0.013, 0.163] | **0.469** [0.332, 0.571] | 16 / 3 |

- **What happens under twoside.** The bridge makes a tall, narrow spike. After decon, the hit finder cuts it into
  several hits and drops the slow tail between them. On runs of ToT ≥ 141, hits carry 0.7 % of the (grossly
  over-filled) ROI PE.
- **Worked example.** Ch 1021, ToT 205: twoside ROI 15 568 PE → 6 hits, 5 231 PE. ToT: ROI 9 587 → 1 hit, 9 600 PE.
- **Window check.** Widening the window to [i − 3000, j + 6000) changes nothing (`hit_split18_wide.txt`), so the PE is
  lost, not booked elsewhere.
- **Net effect.** On deep rails, production's flash PE is an under-count: twoside's over-fill of the waveform is more
  than cancelled at the hit stage. Doc 30's accuracy numbers are decon-window PE, not flash PE. Neither doc has
  flash-PE truth on deep real rails.
- **Limits of ToT here.** ToT removes the fragmentation up to ToT ≈ 260 (d ≈ 8). Above that the hit finder still keeps
  only about half of the ROI PE. That is a second, mode-independent hit-finder limit on very long pulses.

**Knock-on: flash splits and retiming.**
- Twoside's fragments form extra flashes 1.2–1.5 µs after bright railed flashes. That is the "one physical flash
  split" defect doc 26's tail merge addresses, and production light predates tail merge.
- Under ToT, 47 such flashes (18 events) are absorbed into their seed.
- Flash time is the PE-weighted mean of hit peak times. Bright railed flashes therefore move by up to about 1.3 µs
  when the railed channel's PE comes back into one hit at the pulse peak (`d31/time_map.txt`).

Per escalation rule 7, all of this is **reported, not fixed**. OpHitFinder is not touched.

## 6. Doc 11 §6 and doc 12 numbers, rerun (`q31ctl` = production twoside, `q31tot2` = ToT)

**Doc 11 §6.2/6.3** (railed chi2 terms of **selected** matches, pooled over the 18 run-039252 events;
`d31/sat_terms_18evt.txt`; doc 11 used one event):

| | q31ctl | q31tot2 |
|---|---|---|
| selected bundles on railed flashes / railed terms | 706 / 2017 | 715 / 2132 |
| meas/pred median; meas > pred | 1.48; 66.5 % | 1.74; 73.5 % |
| chi2 term median / p90, inflate 0 | 1.13 / 146 | 1.63 / 239 |
| inflate 0.5 (production): median / p90 / max | 0.58 / 3.22 / 4.00 | 0.64 / 3.42 / 4.00 |
| inflate 0.25 / 0.35: p90 | 11.9 / 6.4 | 12.9 / 6.8 |

- **Pre-registered inflate rule.** An extra inflate arm runs only if the p90 at 0.5 falls below 2.0. It is 3.42, so
  the cap still binds. **`chi2_sat_inflate` 0.5 stays; no `q31toti` arm.**
- **Reading meas/pred.** It moves away from 1 under ToT. That is not evidence against ToT: the prediction's scale is
  fit on unrailed channels, and the unrailed high-prediction bin already reads 0.67–0.68 (doc 12 table below).
- **Doc 11 §6.4 does not transfer.** `analyze_sat_maskfit.py` joins bundles by flash id, and ids renumber between these
  arms (its "moved 66" is that artefact). Only 2477 bundles join, so its aggregates cannot tell the arms apart
  (`d31/maskfit_evt298567.txt`).

**Doc 12** (`fit_qtol_crossers.py`, strict crosser anchors, 120 events; `d31/qtol_crossers_*.txt`):

| | q31ctl | q31tot2 |
|---|---|---|
| anchors | 192 | 212 |
| global Σmeas/Σpred → QtoL | 0.833 → 0.0783 | 0.814 → 0.0765 |
| per run 039252 / 039253 / 039349 | 0.757 / 0.857 / 0.860 | 0.756 / 0.839 / 0.837 |
| cathode XA | 0.822 | 0.802 |
| PMTs | 2.130 [0.043, 8.19] | 1.709 [0.025, 8.01] |

- **Control closure.** `q31ctl` reproduces doc 12's 2026-09-13 status line exactly (0.833 → 0.0783, the `p100flip`
  value; production keeps 0.094).
- **ToT result.** Global −2.3 % and cathode −2.4 %, just over the pre-registered 2 % "report" line. Nothing is
  re-tuned.
- **Unpaired.** The two anchor sets are not the same: 20 more crossers qualify under ToT, where flashes no longer
  split. The anchors exclude sat-flagged channels, as expected.

## 7. The owner's hand scan: FAIL on the existing record

**Truth:**
- the gold owner scan `work/ql_labels/wfresc/labels-evt298567.json`;
- the AI+owner `decisions-cathxa` verdicts;
- all 18 run-039252 events, `--truth-uid-map-tag keep`, objective tiers, long tracks.

**Eligible population** (pre-registered): 513 of 775 judged autos, and 576 of 841 scan positives, sit on flashes with a
cathode rail. The scan has power for this lever.

| arm | join | agree | phantom | missed | unknown | flash-cut misses |
|---|---|---|---|---|---|---|
| `q31ctl` (production) | pre-registered | 683 | 97 | 158 | 376 | 0 |
| **`q31tot2`** | **pre-registered** | **654** | **92** | **187** | 409 | 20 |
| `q31tot2m` | diagnostic: truth times translated by `d31/time_map_ctl_to_tot2.json` | 668 | 92 | 173 | 395 | 1 |
| `q31tot` (single-start shapes, superseded) | pre-registered / mapped | 656 / 670 | 93 / 93 | 185 / 171 | 411 / 397 | 19 / 0 |

**Verdict under the doc 23 rule: FAIL.** Missed worsens under both joins.
- **Pre-registered join:** newly missed 35 vs recovered 8, sign test p = 4e-5.
- **Time-mapped join:** 25 vs 12, p = 0.047.
- **Where the movers sit** (`d31/scan_pairs.txt`): mostly on cathode-railed flashes (20 of 25 newly missed, 11 of 12
  recovered, mapped), so this is the lever acting, not noise.
- **Phantom −5 is not a gain under either join.**
  - Raw: retimed flashes leave the judged set and turn up as unknown (+33).
  - Mapped: the map's positive-wins collision rule deleted 5 negative verdicts.
  - Phantom movers 8 / 13 give p = 0.38.

**What the record cannot see:**
- The scan was taken on twoside light, and some verdicts sit on flashes that twoside split off and ToT removes.
- ToT's new autos carry no verdict: unknown +19 (mapped) to +33 (raw).

So the precise statement is: **ToT fails on the existing record, and the record cannot judge ToT's gains.** The time
map is a post-hoc diagnostic in the doc 26 style, and it fails too.

## 8. Recommendation and open items

**Recommendation.** Do not flip. The knob stays default OFF, and production stays on twoside, `chi2_sat_inflate`
0.5 and QtoL 0.094.

**What would change the call, for the owner to scope:**
1. **Adjudicate what ToT changes.** Run a blind scan of the (event, cluster) pairs that move between `q31ctl` and
   `q31tot2m`: 25 newly missed, 12 recovered, 8 + 13 phantom movers and the new unknowns, about 80 items. Without it,
   the scan can only say that ToT disagrees with verdicts taken on twoside flashes.
2. **Fix the hit-level loss (§5) first; it is mode-independent.**
   - Twoside keeps 0.7 % of the ROI PE on runs of ToT ≥ 141, and ToT keeps about 50 % above ToT 260.
   - The hit finder also creates the split flashes that doc 26's tail merge exists to repair.
   - A default-OFF OpHitFinder knob (for example, one hit per flagged rail run) is the candidate. It needs the same
     gates as this doc.
3. **Re-evaluate on tail-merged light.** Production Q/L still reads the pre-merge `_keep` light while the runner
   default is tail merge on (§4). Should ToT be judged on the light production will actually run next?
4. **Doc 30 open items still stand:**
   - shape drift across runs;
   - membrane/PMT ToT;
   - two-pulse runs;
   - an absolute truth for deep real rails.

## 9. Files

**toolkit (`apply-pointcloud`):**
- `flash/inc/WireCellFlash/OpDecon.h`, `flash/src/OpDecon.cxx`;
- `flash/test/doctest_opdecon_tot.cxx`;
- `cfg/pgrapher/experiment/protodunevd/{flash.jsonnet, wct-light-reco.jsonnet, pdvd-tot-shapes-v2.json}`.

**wcp (`pdvd/`):**
- `run_light_evt.sh` (`PDVD_SAT_REPAIR_MODE`);
- this doc;
- `docs/qlmatch/scripts/`:
  - `saturation_tot_study.py` (`--spe v2`, multi-start under v2);
  - `export_tot_shapes.py`, `d31_tot_doctest_ref.py`, `d31_refit_starts.py`;
  - `d31_light_arms.sh`, `d31_ql_arms.sh`, `d31_compile_light.sh`;
  - `d31_flash_pe.py`, `d31_hit_split18.py`, `d31_frames_closure.py`, `d31_roi_closure.py`, `d31_run_census.py`;
  - `d31_time_map.py`, `d31_scan_pairs.py`, `d31_scan_subset.py`, `d31_sat_terms.py`;
- `docs/qlmatch/d31/`:
  - the records, including `prereg.md` and `gate/`;
  - `singlestart/`;
- `docs/qlmatch/pics/saturation_tot_*_v2.png`.

**Fresh tags (M13):**
- light: `_g31old`, `_g31off`, `_g31offb`, `_q31tot`, `_q31tot2`, `_g31pintot`, `_f31ts`, `_f31tot`, `_f31ts18`,
  `_f31tot18`;
- Q/L: `q31ctl`, `q31tot`, `q31tot2`;
- scores: `work/ql_scores/{q31ctl, q31tot, q31totm, q31tot2, q31tot2m}`.

Scratch is `/home/xqian/tmp/p31/`.

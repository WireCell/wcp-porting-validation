# PDVD ToT flip campaign: hit-finder fix, a blind same-scanner record, and Q/L tuning (round 1)

**Status 2026-09-23, round 1 of a multi-session campaign. Nothing is flipped; every new knob is default OFF.**

**Goal (owner).** Make the time-over-threshold (ToT) saturation fill of doc 31 viable, and flip it. The owner/AI
hand-scan record was taken on twoside light, so it is not a neutral judge of a light change. This campaign builds a
blind, same-scanner record on the clusters ToT changes, and re-tunes Q/L matching against it.

**Round 1 in one paragraph.**
- **A real production bug, found and fixed behind a knob.** OpHitFinder stores decon samples as `short`, and the cast
  wraps above 327.67 PE/tick (undefined behaviour).
  - This, not "fragmentation", is the hit-level loss doc 31 saw.
  - The fix is `OpHitFinder int_samples`. With ToT it keeps 100 % of the ROI PE in one hit in every ToT bin; above
    ToT 260 that was 47 % in 3 hits.
  - It cannot ship with twoside: it exposes twoside's ~40× over-fill.
- **A second default-OFF knob, and a negative result.** `QLMatching sat_flag_ignore_channels` lets ToT-repaired
  cathode rails enter chi2/KS/LASSO. It made matching slightly worse.
- **The blind scan is INCONCLUSIVE by its pre-registered bars.** Five blind scanners judged 259 sheets. Only 28 % of
  the 100 changed clusters reached consensus (the bar was 60 %). Calibration against the owner record was 83 % (the
  bar was 85 %).
- **Informational only, n = 28.** Production picks the scanners' flash more often than ToT: 12 vs 6 when exactly one
  of them is right, p = 0.24. This points the same way as the owner record.
- **Where things stand.** The ToT case now rests on light-level PE accuracy. Every Q/L-level measurement so far leans
  against it, but none is separable from noise on the blind record.
- **Round 2 plan (§8).** A consensus rule that does not square abstention, a trace of the LASSO inputs, and the
  owner's own verdicts on the 72 unresolved clusters.

**Repro:**
```
# toolkit (apply-pointcloud): 9d0c2723 flash int_samples, d8fef1f1 match sat_flag_ignore_channels
wcbuild && ./wcb build --target=wcdoctest-flash --target=wcdoctest-match
./build/flash/wcdoctest-flash && ./build/match/wcdoctest-match          # 19/19 and 6/6
cd pdvd/docs/qlmatch/scripts
# 1. hit finder: replica closure + which rule recovers the PE (18 frame-dumped events of d31)
python3 d32_hit_rules.py > ../d32/hit_rules.txt
# 2. light arms (production-light args = _keep), gate hashes
ARM=_g32off ./d32_light_arms.sh ; ARM=_g32offb ./d32_light_arms.sh
ARM=_q32ti ENVS="PDVD_SAT_REPAIR_MODE=tot PDVD_HIT_INT_SAMPLES=1" ./d32_light_arms.sh
ARM=_f32ti18 ONLY_RUN=39252 ENVS="PDVD_SAT_REPAIR_MODE=tot PDVD_HIT_INT_SAMPLES=1 PDVD_LIGHT_FRAMES=1" ./d32_light_arms.sh
ARM=_f32si18 ONLY_RUN=39252 ENVS="PDVD_HIT_INT_SAMPLES=1 PDVD_LIGHT_FRAMES=1" ./d32_light_arms.sh
for a in g32off g32offb q32ti; do ./d32_gate_hash.sh $a > ../d32/gate/hashes_$a.txt; done
python3 d32_hit_closure.py > ../d32/hit_closure.txt
# 3. Q/L arms (d31_ql_arms.sh); q32ctlp/q32tis on the pin with the new libWireCellMatch
ARM=q32ti LIGHT=_q32ti ./d31_ql_arms.sh
ARM=q32ctlp LIGHT=_g31off PIN=/home/xqian/tmp/p32/libpin_q32 ./d31_ql_arms.sh              # gate g32pin
ARM=q32tis LIGHT=_q32ti PIN=/home/xqian/tmp/p32/libpin_q32 ENVS="PDVD_QL_SAT_IGNORE_CATHODE=1" ./d31_ql_arms.sh
cd ../../..; W=/home/xqian/tmp/p31/wroot     # symlinks to the arm dirs + the _keep calib dumps (doc 31)
python3 docs/qlmatch/scripts/d31_time_map.py --a _g31off --b _q32ti --calib-tag q31ctl --calib-tag-b q32ti --work-root $W --out docs/qlmatch/d32/time_map_ctl_to_q32ti.json
python3 ql_display/ql_agree_score.py --tag q32ti --work-root $W --truth-uid-map-tag keep     # + q32tis; +m = --truth-time-map
python3 docs/qlmatch/scripts/d31_flash_pe.py --a _g31off --b _q32ti > docs/qlmatch/d32/flash_pe_q32ti.txt
cd docs/qlmatch/scripts
python3 d32_loss_stage.py --work-root $W --ctl q31ctl --cand q32ti --time-map ../d32/time_map_ctl_to_q32ti.json > ../d32/loss_stage.txt   # + q32tis
# 4. blind scan: items + sheets (seed 32), 5 scanner agents (d32/RUBRIC.md), audit, merge, score
python3 d32_scan_items.py --work-root $W --ctl q31ctl --cand q32ti --time-map ../d32/time_map_ctl_to_q32ti.json \
    --sheet-dir /home/xqian/tmp/p32/scan/r1q --key /home/xqian/tmp/p32/scan_key_r1q/key.json --jobs 16
python3 d32_audit.py <scanner transcript> <wave>                                      # -> d32/scan_r1/audit.txt
python3 d32_merge_record.py --key /home/xqian/tmp/p32/scan_key_r1q/key.json --scan-root /home/xqian/tmp/p32/scan/r1q \
    --time-map ../d32/time_map_ctl_to_q32ti.json --out ../d32/scan_r1
python3 ../../../ql_display/ql_agree_score.py --tag <arm> --work-root $W --truth-uid-map-tag keep \
    --override-truth ../d32/scan_r1/override_consensus_<ctl|cand>.jsonl --out-root /home/xqian/tmp/p32/scores_r1
```

## 1. What was asked, and what was pre-registered
The owner asked for four things:
1. fix the production hit-level problem doc 31 reported but did not fix;
2. blind-scan the results with up to 5 subagents;
3. tune the PDVD Q/L options for ToT;
4. work towards flipping ToT, over several sessions if needed.

`d32/prereg.md` was written before any sheet existed, and extended after a design review but still before rendering.
It fixes:
- the arms;
- the item set: movers, calibration and duplicates;
- blinding;
- the rubric, with railed channels as a sanity check only, identically for both lights;
- the consensus-truth rule;
- the decision bars: calibration ≥ 85 %, consensus on ≥ 60 % of movers, the doc 23 rule and split-half.

## 2. The hit-finder bug and its fix (toolkit `9d0c2723`)
**Symptom.** Doc 31 §5: on deep cathode rails, twoside's hits kept a median 45 % of the ROI PE at ToT 141–260 and
7.5 % above 260. Even ToT kept only 47 % above ToT 260, in 3 hits per run.

**Root cause.** `OpHitFinder::operator()` scales decon by 100 and stores it as `short`:
`wf[i] = static_cast<short>(m_scale * charge[i])`.
- A sample above 327.67 PE/tick does not fit. The conversion is undefined behaviour, and on gcc/x86-64 it wraps to a
  large negative or unrelated value.
- The sliding window then ends the pulse, restarts on the wrapped samples, and cuts the pulse into fragments.
- A Python replica of the hit finder (`scripts/d32_hit_emul.py`) that emulates the wrap reproduces the C++ hits
  exactly, on all 16 channels of evt298567 in both arms. Without the wrap it does not reproduce them.

**Why it hid.**
- Twoside's bridge over-fills deep rails by orders of magnitude: ROI sum 4.95e8 PE on ToT ≥ 71 runs against 1.2e7
  under ToT.
- The wrap threw most of that away, and the flash PE came out "only" about 30 % low instead of 40× high.
- Samples wrapped in the 18 frame-dumped events (`d32/hit_rules.txt`):

  | | wrapped samples |
  |---|---|
  | cathode, twoside | 16 333 |
  | cathode, ToT | 5 362 |
  | membrane XAs, production light | 550 (16 of 270 traces) |
  | PMTs | 0 |

- Note: production twoside flash PE on deep rails currently **depends on this undefined behaviour**.

**Fix.** `OpHitFinder int_samples` (default false). The pulse algorithms (`sliding_window`, `split_pulse`,
`slice_pulse`) are templated on the sample type. `short` is legacy, bit-identical; `int` is used when the knob is on,
clamped to the int range so the cast is always defined.
- The public static `short` functions stay, and gain `int` overloads.
- PDVD wiring: `wct-light-reco.jsonnet hit_int_samples` on the **cathode branch only** (like ToT), and runner
  `PDVD_HIT_INT_SAMPLES=1`. The key is suppressed when off.
- The membrane wrap is reported, not changed.

**Verification.**
- **Doctests.** `wcdoctest-flash` passes 19/19, including 3 new cases:
  - a bright pulse keeps its PE in one hit;
  - the legacy path loses PE on it;
  - `int` overloads equal `short` below the short range.
- **Freshness.** The library (13:43:48) is newer than the source (13:43:38).
- **Compiled config.** Identical to HEAD on 3 argument sets (production `_keep`, + ToT, + tail merge). The key appears
  only on the cathode OpHitFinder when the knob is on.
- **Gate `g32off`** (`d32/gate/`). Light archives made with the new library and the knob off are hash-identical:
  - to the d31 library's `_g31off`, 120/120;
  - to production `_keep`, 120/120;
  - null pair `_g32off` / `_g32offb`: 120/120.

  The library md5 is in `pin_loaded_check.txt`.
- **Closure.** The C++ `int_samples` hits equal the replica's int rule on 576/576 channel-arms
  (`d32/hit_closure.txt`).

**Status:** knob-off byte-identical, gate label `g32off`. Knob on: not bit-identical, needs revalidation (this doc).

## 3. What `int_samples` does to the light (`d32/hit_closure.txt`, `flash_pe_q32ti.txt`)
C++ hit PE over the decon ROI sum per cathode rail run (18 events, window [i−50, j+300)); median, with the number of
hits per run after the slash:

| ToT (ticks) | n | twoside, short (production) | ToT, short (doc 31) | twoside, int | **ToT, int** |
|---|---|---|---|---|---|
| 1–20 | 1019 | 1.001 / 2 | 1.001 / 2 | 1.001 / 2 | 1.001 / 2 |
| 71–141 | 553 | 1.004 / 3 | 1.001 / 1 | 1.007 / 3 | 1.001 / 1 |
| 141–260 | 401 | 0.450 / 4 | 1.000 / 1 | 1.010 / 3 | **1.000 / 1** |
| > 260 | 152 | 0.075 / 15 | 0.469 / 3 | 1.013 / 2 | **1.000 / 1** |
| Σ hits on ToT ≥ 141 | | 3.6e6 | 5.0e6 | **4.98e8** | 9.9e6 |

- **ToT + int.** It is the only combination that is both complete and sane.
- **Twoside + int.** Complete, but it carries twoside's over-fill: 100× the ToT PE. `int_samples` must never be used
  with twoside.
- **Flash level** (`_q32ti` vs `_g31off`, 120 events):
  - flash count 48 600 → 48 342;
  - summed PE on flashes with a cathode rail 1.53e7 → 1.86e7;
  - railed cathode cells: median ratio 1.000, p95 1.32, p99 3.0;
  - unrailed cells are unchanged: p99 1.003.
- The merge rule planned before the replica was dropped: with the wrap gone it adds nothing (`hit_rules.txt`
  int+merge column).

## 4. Q/L arms, and the owner record
Scored against the owner record: agree / phantom / missed, and missed after doc 31's time map.

| arm | light | Q/L | owner record | time-mapped |
|---|---|---|---|---|
| `q31ctl` (production) | `_g31off` = `_keep` | production | 683 / 97 / 158 | – |
| `q32ti` | ToT + int | production | 652 / 88 / 189 | 667 / 88 / 174 |
| `q32tis` | ToT + int | + `sat_flag_ignore_channels` 4–11 | 647 / 89 / 194 | 662 / 89 / 179 |

**Where the changed pairings are decided** (`d32/loss_stage.txt`):
- 89 pairings are lost and 95 gained between `q31ctl` and `q32ti`.
- In 82 and 88 of them respectively, the losing arm's bundle has LASSO `strength` below the 0.05 cutoff.
- ks, chi2/ndf and pred/meas on the same physical flash barely move: median Δks +0.01 to +0.04, Δlog pred/meas
  within ±0.1.
- So the LASSO decides the movers, not the KS/chi2/ratio gates.

**`sat_flag_ignore_channels`** (toolkit `d8fef1f1`, default OFF).
- **What it does.** Railed cathode rows are zeroed in the LASSO in production, so the natural first lever was to let
  ToT's repaired cathode PE into the fit. The knob clears the per-flash rail flag on the listed OpDets when the flash
  is read (`Opflash::clear_sat`).
- **Wiring.** PDVD threads it as `wct-clustering.jsonnet ql_sat_flag_ignore_cathode` (OpDets 4–11), runner
  `PDVD_QL_SAT_IGNORE_CATHODE=1`.
- **Gate `g32pin`** (`d32/gate/g32pin.txt`).
  - The Q/L arms run on a frozen library pin. The pin `libpin_p100b` with only `libWireCellMatch` replaced reproduces
    the `q31ctl` calib dumps byte-for-byte, 120/120, with the knob off.
  - The compiled clustering config is identical to HEAD. The HEAD jsonnet rejects the new argument, which proves the
    overlay was really HEAD.
- **Result: negative.** It is slightly worse on the owner record (above) and on the blind record (§6), so it is not
  recommended. The likely reason is to be tested in round 2, not concluded here: the photon model over-predicts bright
  cathode channels (doc 31: unrailed meas/pred ≈ 0.67 in the high-prediction bin). Feeding a truer PE into the fit
  against a biased prediction does not help.

## 5. The blind scan: design and conduct (`d32/RUBRIC.md`, `d32/scan_r1/`)
**Items.**
- **Movers.** 100 long clusters whose auto flash set differs between `q31ctl` and `q32ti` after the time map. Only
  these can change the comparison. Each was rendered in both lights: 200 sheets.
- **Calibration.** 40 owner-judged non-movers, rendered in the control light:
  - 20 on cathode-railed flashes and 20 unrailed;
  - 6 per class from the owner's own gold event 298567, the rest from the cathxa AI + owner scan.
- **Duplicates.** 20 duplicate sheets.
- **Total.** 260 sheets; 1 was not rendered, because its cluster has no bundle in one light.

**Sheet** (example `pics/32_blind_sheet_example.png`). Per candidate flash:
- the event projections at that flash's T0, with the cluster in red;
- the cluster's x-extent and its distances to the anode, the cathode and the drift box;
- measured bars vs this cluster's predicted line, with the y scale set by the **unrailed** channels. Railed channels
  are marked R and their value is printed when clipped;
- measured (R-clipped) and predicted maps;
- unrailed and all-channel PE sums.

Hidden: auto marks, ks, chi2, strength, flags, ids and the light name.
- **Scale fix before launch.** A first render let a huge ToT R bar set the scale and squash the unrailed pattern. That
  would have made ToT sheets harder to read than twoside ones, so it was fixed and re-rendered (`r1q`); no scanner saw
  the first render.

**Conduct.**
- Five general-purpose agents, one wave of 51–53 sheets each.
- The two lights of a mover always went to different scanners, and duplicates to a third.
- Every verdict went through `d32_record.py`.
- **Audit** (`scan_r1/audit.txt`). Every scanner made one Read per sheet and one recorder call per sheet (259 of each),
  and 0 reads outside its wave directory, the rubric and the recorder. The only flagged calls are the agents' final
  hand-back messages.

**Verdicts.** Of the 259:
- 121 letter picks at med or high confidence;
- 55 letter picks at low confidence;
- 7 `none`;
- 76 `unsure`.

Low confidence counts as unsure, per the rubric. So about half the sheets are abstentions. The scanners' own accounts
agree on where the difficulty lies:
- tiny clusters (10–300 points) predicting tens of PE;
- pairs of flashes a few µs apart with the same geometry;
- the best-lit candidate putting the cluster 15–55 cm outside the drift box, while the geometrically clean candidate is
  dark;
- showers where nearly every channel is railed.

## 6. The blind scan: results (`scan_r1/merge_report.txt`, `informational.txt`)
| pre-registered bar | required | round 1 |
|---|---|---|
| calibration vs owner record (committed verdicts) | ≥ 85 % | **83.3 %** (15/18): gold 6/8, cathxa 9/10; 20 unsure, 2 not comparable |
| movers resolved to a consensus | ≥ 60 % | **28 %**: 28 positive, 0 none, 67 unresolved-unsure, 5 unresolved-disagree |
| duplicate consistency | reported | same 6, one side unsure 14, different 0 |

**Verdict: inconclusive, "scan too noisy to judge". No flip reading is taken from this record.**

**Why consensus failed. This is the rule, not scanner noise.**
- When scanners commit, they agree: duplicates 6/6, calibration 15/18.
- The failure is abstention. Consensus needs a committed verdict in **both** lights, from two independent scanners. At
  a commit rate of about 0.5 that gives about 25–30 % resolved, and 28 % is what came out.
- The 60 % bar was not reachable with this rule. Round 2 pre-registers a rule that does not square abstention (§8).

**Informational only (post hoc, no flip weight).** The 28 consensus-positive movers: does each arm auto-select the
scanners' flash?

| comparison | only ctl right | only candidate right | both | neither | sign test |
|---|---|---|---|---|---|
| `q31ctl` vs `q32ti` | 12 | 6 | 5 | 5 | p = 0.24 |
| `q31ctl` vs `q32tis` | 14 | 6 | 3 | 5 | p = 0.12 |

- Not separable from churn at this size.
- It points the same way as the owner record, and this time from a neutral scanner. It should not be softened: on the
  pairings it could judge, the blind record does not show ToT picking better flashes.
- Consensus-truth scores (owner record with the 100 movers replaced by consensus truth; unresolved movers dropped in
  both arms):
  - `q31ctl` 669 / 94 / 150;
  - `q32ti` 652 / 95 / 167;
  - `q32tis` 646 / 96 / 173.

**Where the movers sit** (`informational.txt`). The time map shows what happened to the control arm's auto flash for
each mover:
- 38: no move;
- 58: retimed ≤ 0.5 µs;
- 1: retimed more than 0.5 µs;
- 3: absorbed.

So the lever is **not** the flash set: twoside's split fragments disappearing under ToT account for only 3 movers.
Something inside the fit changes. The railed rows are zeroed in the LASSO in both arms, but the railed PE still enters
chi2/KS (production `saturation_mask_fit=false`). That is a candidate route by which bundle eligibility, and so the
LASSO's column set, changes. To be traced in round 2.

## 7. Doc 11 / doc 12
Not rerun this round: no Q/L candidate reached a flip reading. Doc 31's numbers for ToT without `int_samples` stand;
the round that proposes a candidate reruns them.

## 8. Round 2 (next session), pre-registration items to write first
1. **Consensus rule that does not square abstention.** A mover is resolved when at least one light has a committed
   (med/high) pick and the other light does not contradict it (`unsure` is not a contradiction). A committed
   contradiction stays unresolved. The round-1 verdicts may be re-merged under it, labelled post-hoc and given no flip
   weight.
2. **Trace the LASSO inputs.** For 3–5 movers, dump the bundles entering the LASSO in each arm: the flash columns, the
   unrailed rows, `pe_err`, and which bundles were pre-cut by chi2/KS. Diff them. Test whether railed PE in chi2/KS
   changes LASSO eligibility, and whether prediction bias on bright cathode channels (doc 31 meas/pred ≈ 0.67) is what
   makes truer PE hurt.
3. **Owner verdicts on the 72 unresolved movers.** Offer them as a viewer tag. The scanners say these need judgment the
   rubric does not give, and the owner's verdicts are the only truth source with authority.
4. **8 `q32tis` movers not yet on any sheet.**
5. **Q/L ladder** only after (2), one lever per arm, tuned on odd events and confirmed on even ones:
   - candidate levers come from the trace (e.g. `saturation_mask_fit` / `chi2_sat_inflate` on the cathode, or a
     prediction-scale fix);
   - `sat_flag_ignore_channels` is dropped from the ladder (§4).
6. **Membrane wrap** (550 wrapped samples): a separate default-OFF item for the membrane branch, if the owner wants it.

## 9. Files
- **toolkit** (`apply-pointcloud`):
  - `9d0c2723`: `flash/{inc,src}/OpHitFinder.*`, `flash/test/doctest_ophitfinder.cxx`,
    `cfg/.../protodunevd/{flash,wct-light-reco}.jsonnet`;
  - `d8fef1f1`: `match/{inc,src}/{Opflash,QLMatching}.*`, `match/test/doctest_qlmatching_config.cxx`,
    `cfg/.../protodunevd/{qlmatching,wct-clustering}.jsonnet`.
- **wcp:**
  - runners `run_light_evt.sh` (`PDVD_HIT_INT_SAMPLES`) and `run_clus_evt.sh` (`PDVD_QL_SAT_IGNORE_CATHODE`);
  - `ql_display/ql_agree_score.py --override-truth` (default off; totals identical without it);
  - scripts `d32_*`;
  - records `d32/`: `prereg.md`, `RUBRIC.md`, `gate/`, `hit_rules.txt`, `hit_closure.txt`, `flash_pe_q32ti.txt`,
    `loss_stage*.txt`, `time_map_ctl_to_q32ti.json`, and `scan_r1/` (key, verdicts, INDEX per wave, audit, merge
    report, override truth files);
  - `pics/32_blind_sheet_example.png`.
- **Scratch** (not committed): sheets `/home/xqian/tmp/p32/scan/r1q/`, pins `/home/xqian/tmp/p32/libpin_*`.

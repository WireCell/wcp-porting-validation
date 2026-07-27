# 63 — STM tagger improvement campaign (against the doc-62 owner baseline)

**Status.** Rounds 1–4 SHIPPED (records in §4): **16 → 5 errors on the
72-bundle owner baseline** — 10 false STMs fixed, the 1 missed STM recovered,
zero regressions on the 56 correct verdicts, knob-off byte-identical, and
the full-population check (all 883 events / 923 in-beam bundles) shows zero
unjustified flips.  **All five knobs are SBND production DEFAULT ON as of
2026-07-26 (owner instruction)** — at the SBND jsonnet/TLA/runner level
only; the C++ defaults stay false so every other detector's chain is
byte-identical.  Opt out (the pre-campaign legacy A/B arm) with
`-no-stm-guards -no-stm-proton-guard -no-stm-cathode-guard -no-stm-anode-fix
-no-stm-track-guard` — note this means post-flip knob-OFF gate arms must
pass those flags explicitly.  Round 5+ is analysis-gated (§2).  Each round is a default-OFF knob in
`TaggerCheckSTM.cxx`, evaluated on the full baseline AND the full population
before it is committed.  Only the STM tagger is touched — TGM, LM and FC are
out of scope by the owner's instruction.

**The target.** `scan-d59k/stm-baseline.tsv` (owner verdicts = truth):

| | n | goal |
|---|---|---|
| code-FALSE-STM (tagger STM, owner not-STM) | 15 | fix as many as possible |
| code-MISSED-STM (62613:17) | 1 | recover |
| code-STM-correct | 36 | **keep every one** |
| code-not-STM-correct | 20 | **keep every one** |

A round is an improvement only if it fixes ≥ 1 error and regresses **0** of the
56 correct verdicts.  The owner's acceptance standard applies throughout: a
Michel electron or a proton at the stopping end does NOT disqualify an STM tag
(doc 62 §4), so no round may reject on "the residual is not empty" alone.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# diagnostics extracted from the STORED d59k production (read-only, M13):
python3 stm_campaign/extract_stm_diag.py --out /home/xqian/tmp/stm-campaign/diag
# one evaluation round (see stm_campaign/run_round.sh -h):
./stm_campaign/run_round.sh r0                  # reference arm, knobs off
./stm_campaign/run_round.sh r1 -stm-guards     # round-1 arm
python3 stm_campaign/score_round.py --round work-stmcamp-r1 --ref work-stmcamp-r0
# full-population regression check (protocol step 5; owner authorized 24 CPUs):
STM_EVENTS="$(tr '\n' ' ' < /home/xqian/tmp/stm-campaign/full-events.txt)" \
  NJOBS=24 ./stm_campaign/run_round.sh r2fullb -stm-guards -stm-proton-guard
python3 stm_campaign/score_full.py --round work-stmcamp-r2fullb
```

The production flag set is `run_full1k_nusel.sh`'s `NUF` verbatim
(`-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5
-fvy 3 -stm-fit -mip 56000 -unmerge-assoc`); rounds append their knob flag.
Work roots `work-stmcamp-<round>/` symlink imaging (`evt<ID>`) and the Q/L
pctree (`ql_evt<ID>`) from `work-mcp1kall-d59k` — the Q/L step is never re-run,
so a round re-executes exactly the PR tagger tail (seconds per event).  Nothing
under `work-mcp1kall-d59k/` is written.

## 1. Where every error comes from (diagnosis, from stored d59k fits)

Sources: the `stm_pass`/`stm_eval`/`stm_fit` PCs that the `-stm-fit` production
already persisted in every `pctree-pr-evt<ID>.tar.gz`, plus the tagger logs.
Extractor: `stm_campaign/extract_stm_diag.py`.  All 72 baseline bundles reached
the round-2 fit (no pre-fit exits).  Statuses: the 15 false STMs were all
**accepted (status 0)**; the miss 62613:17 was killed by **detect_proton
(status 5)**; the 20 correct non-tags exited with status 2 (long leftover),
3 (dQ/dx eval), 4 (extra tracks) or 5 (proton).

Key measurements (all offline, on the stored per-pass fits; MIP = 56000 e/cm):

1. **Charge desert under the fit** — longest contiguous stretch of the fitted
   muon segment with per-point dQ/dx < 0.02 MIP:
   278794:7 = **23.4 cm**, 285443:7 = **6.7 cm**, every other accepted bundle
   (false or correct) = **0.0 cm**.  The fit bridged detached objects through
   empty space — the owner's one-objectness ruling, measurable.  A ≥ 3 cm veto
   separates with infinite margin on this set.

2. **Normalization hole in the eval acceptance** — the first acceptance branch
   (`ks1-ks2 < -0.02 && ...`) never checks that the measured charge resembles
   a muon in NORMALIZATION.  285443:7 was accepted with ratio2 = **3.96**
   (measured charge 4× BELOW the flat-MIP reference — a ghost-mess, not a
   muon).  The highest CORRECT accepting call is ratio2 = **1.48** (283463:14,
   a sub-MIP track the owner accepts — a first-try cap at 1.1 killed it; see
   the round-1 record).  The shipped cap is **2.0**: no under-collected muon
   loses half its charge, a ghost does.

3. **Spike-not-ramp at the stop** — a real Bragg is a RAMP: the muon table
   gives ~1.7 MIP at rr = 3 cm rising to ~3 at rr = 0.5 cm, so the 1.5–4 cm
   "shoulder" before the stop is elevated.  A ν vertex is a SPIKE: MIP-flat
   approach, then one or two very high points.  peak(< 1.5 cm)/median(2–6 cm)
   and the shoulder median separate:

   | bundle (owner) | spike | shoulder (MIP) |
   |---|---|---|
   | 290294:12 (vertex) | 7.0 | 0.66 |
   | 409546:8 (vertex) | 7.4 | 0.81 |
   | 285366:13 (vertex) | 5.3 | 1.69 |
   | 289832:10 (vertex) | 3.6 | 0.84 |
   | worst correct STM: 401824:4 | 3.1 | 1.87 |
   | next: 409458:8 | 2.7 | 1.38 |

   Veto `spike > 3.2 && shoulder < 1.8` catches the four vertex cases and
   protects every correct STM on BOTH conditions (401824 fails both).
   278662:1 ("could be a vertex activity", spike 2.7/shoulder 0.97) is inside
   the correct population's spike range — left for a later round.

4. **The miss** — detect_proton killed 62613:17 through branch
   `dQ_dx[max_bin] > 4.3 MIP && ks3 > 0.03` by margins of 0.0007 (4.3007) and
   0.003 (ks3 = 0.033), while the muon hypothesis fits the end region almost
   perfectly: **ks1 = 0.030, ratio1 = 0.94, ratio3 = 1.06**.  The two proton
   vetoes the owner CONFIRMS (288859:9, 319809:20): 288859 fired through the
   delta-ray path (untouched), 319809 through branch B1 with ks1 = 0.063 and
   ratio3 = 1.22.  A muon-consistency guard `ks1 < 0.045 && ratio3 < 1.1 ⇒
   not a proton` recovers the miss and keeps both correct rejections, with
   ~2× margins on both axes.

5. **Cathode-side flat truncation** (round-3 signal) — a track whose fitted
   stop sits within a few cm of the cathode plane of its own drift volume,
   with NO Bragg rise in the last 5 cm, is overwhelmingly a through-going
   track truncated at the cathode (wrong-side charge lost), not a stopping
   muon.  Survey of ALL 153 d59k STM tags (stop-to-cathode distance along the
   drift axis, from the volume's `inner_bounds`, vs the end-5 cm dQ/dx peak
   in MIP):

   | bundle (verdict) | stop→cathode (cm) | end-5 cm peak (MIP) |
   |---|---|---|
   | 72586:17 (owner: not-STM, flat) | 2.8 | 1.95 |
   | 392200:27 (owner: not-STM, flat) | 1.1 | 1.08 |
   | 56463:12 (AI scan: truncated crosser) | 3.9 | 1.25 |
   | 289343:9 (real cathode-stopping STM) | 6.3 | 5.74 |
   | 281148:13 (real cathode-stopping STM) | 2.2 | 3.31 |

   Veto `distance < 5 cm && peak < 2.5 MIP`: the flat truncations die on the
   peak axis, the genuine cathode stoppers survive on their Bragg rise, and
   every mid-volume STM is out of reach of the distance cut.

6. **A second MIP track hiding in the accepted fit** (round-4b/4c signal,
   from the full-population leftover/segment surveys —
   `/home/xqian/tmp/stm-campaign/{leftover_survey,offtrack_survey}.py`):

   - *Leftover past the kink* (all 22 accepted d59k passes with left_L >
     3 cm): the V-topology false STM **353223:15 has left_L = 28.6 cm with
     straightness 0.972** — the longest leftover on any correct STM is
     22.3 cm (285795:11, straightness 0.892), and the longest STRAIGHT
     (> 0.9) one is 21.1 cm (280092:6, 0.951).  The tagger's own leftover
     rejection only fires above 40 cm or above 2 MIP, so a second MIP leg
     sails through.  Veto `left_L > 25 cm && straightness > 0.9`: kills
     exactly 353223:15 in the whole population.
   - *search_other_tracks segments* (28 instrumented segments across the 883
     events): **349241:15's 20.4 cm segment at 1.00 MIP** escaped through
     the prototype's `len < 25 && medQ < 1` Michel clause (medQ within
     0.005 of the boundary).  The longest MIP-like (0.4–1.5) segment on any
     correct STM is 12.4 cm (174422:13, 1.30 MIP); longer segments on
     correct STMs are all > 1.5 MIP (proton stubs, e.g. 64475:23 19.9 cm at
     2.41).  Veto `len > 15 cm && medQ ∈ (0.4, 1.5)`: kills exactly
     349241:15.

7. **What does NOT separate** (measured, so nobody retries it):
   - Off-track near-stop steiner counting for the vertex-fan case
     **402330:1** ("multiple tracks, neutrino", single-cluster bundle): it
     shows only **8** off-fit points within 8 cm of the stop while correct
     STMs (Michels, deltas) run to **59** (172788:26).  Its prongs barely
     register in the steiner cloud (121 steiner points for a 414-point
     cluster).
   - The flat-track false STMs (48895, 392200, 321107, 321371, 72586) are
     indistinguishable in dQ/dx from owner-ACCEPTED flat STMs (68956, 282715,
     315409, 175428, 172596…): both run ~1 MIP into the stop, same KS values,
     same residual lengths (the accepting comb values interleave: false
     −0.012..−0.053 vs correct −0.001..−0.105).  The owner separated them on
     the DISPLAY (topology/context), not the dQ/dx panel.
   - Off-track steiner charge within 8 cm of the stop: vertex cases often show
     **0** (the prongs live in other clusters of the bundle), while correct
     STMs show up to 33 off-track points (Michels, deltas).  Prong-counting on
     the main cluster's own steiner cloud does not work.

## 2. The round plan

- **Round 1 — `stm_accept_guards` (C++ `accept_guards`, default false).**
  Three pass-level acceptance guards from §1.1–1.3: charge-desert ≥ 3 cm,
  eval-acceptance ratio2 > 2.0, spike > 3.2 with shoulder < 1.8 MIP.
  Expected: fixes 6 of 15 false STMs (278794, 285443, 289832, 285366, 290294,
  409546) — 4 of the 5 vertex-as-Bragg, 2 of the 6 multi-object — with zero
  expected regressions.  The desert and spike guards are computed once per
  pass from the final fit (call-independent); the ratio2 cap is inside
  eval_stm_core (the offline kill-list is conservative for it, since a killed
  call lets the || chain try later calls the stored records never executed —
  the re-run decides).

- **Round 2 — `stm_proton_muon_guard` (C++ `proton_muon_guard`, default
  false).**  §1.4: in detect_proton's end-proton branches, a candidate whose
  end region matches the muon hypothesis (ks1 < 0.045 && ratio3 < 1.1) is not
  a proton.  Expected: recovers 62613:17; 288859/319809 unchanged.

- **Round 3 — `stm_cathode_guard` (C++ `cathode_guard`, default false).**
  §1.5: an accepted pass whose stop is < 5 cm from its drift volume's cathode
  with an end-5 cm peak < 2.5 MIP is rejected (status 7).  Expected: fixes
  72586:17 and 392200:27, plus the justified flip of the non-adjudicated
  56463:12 (the doc-61 AI scan itself called it a truncated cathode crosser);
  289343:9 and 281148:13 protected by their Bragg rise.

- **Round 4a — `stm_anode_dist_fix` (C++ `anode_dist_fix`, default false).**
  The pre-existing face-selection inversion found in round 3, now fixed on
  the owner's instruction: the shipped `dist_to_anode` helper in
  `check_stm_conditions` (`anode_x = (fdx < 0) ? first : second`) selects
  the CATHODE on SBND (verified semantics: `dirx()` points from anode into
  the drift volume; sensitive BB is normalized, so the anode is at min x
  when fdx > 0).  The knob flips the selection to the true anode face.
  Consumer: only the anode-clipped-TGM check (prototype
  `ToyFiducial.cxx:762`, "at Anode").  Expected: no verdict change on this
  sample (the check's other conditions rarely co-fire); validated by direct
  arm-vs-arm comparison.

- **Round 4b/4c — `stm_second_track_guard` (C++ `second_track_guard`,
  default false).**  §1.6: two measured separations for "a second MIP track
  hiding in the accepted fit".  (4b) leftover-track straightness: an
  accepted pass whose leftover past the kink is ≥ 25 cm with chord/path
  straightness > 0.9 is a second track, not a Michel (population max for
  correct STMs: 22.3 cm, straight-correct max 21.1 cm at 0.951) — kills
  353223:15 (28.6 cm at 0.972).  (4c) in check_other_tracks, a segment
  > 15 cm at medium dQ/dx in (0.4, 1.5) MIP is a second track (correct
  MIP-like max 12.4 cm; proton stubs > 1.5 MIP protected) — kills 349241:15
  (20.4 cm at exactly 1.00 MIP, which escaped the `len<25 && medQ<1` Michel
  clause).  Tunables: `guard_left_track_cm/straight`,
  `guard_seg_track_cm/mip_lo/mip_hi`.

- **Round 5+ — exploratory (design against round-4 re-runs).**  Remaining
  false STMs: 278662 (vertex, spike 2.7 inside the correct range), 402330
  (vertex fan; the off-track near-stop steiner count was measured and does
  NOT separate — §1.7), 48895, 321107, 321371 (flat tracks mid-volume —
  §1.7 says dQ/dx cannot do it; candidate signals: dead-region proximity at
  the stop, readout-window truncation geometry).  Each will only ship with
  a §1-style measured separation.

Rounds are cumulative: round N's arm runs with rounds 1..N's knobs ON.
Per-round record in §4; every round commits (both repos) and pushes.

## 3. Evaluation protocol (per round)

1. `wcbuild` then freshness proof (`local/lib/libWireCellClus.so` mtime).
2. `./build/clus/wcdoctest-clus` passes.
3. **Knob-off gate**: 10-event subset re-run with the new binary, knobs off;
   `nusel-evt*.tsv` diff-identical and `hash_archive.py` member-identical
   (mabc-pr.zip, pctree-pr tarball) vs the round-0 reference.  Compiled-config
   proof: knob key absent when off, present when on (`wcsonnet | grep`).
4. **Knob-on arm**: all 72 baseline events, `run_round.sh r<N> <flags>`;
   `score_round.py` prints the confusion vs the owner baseline, the per-bundle
   fixes/regressions vs round 0, and any verdict flip on non-adjudicated
   bundles that share those events (reported as collateral, judged case by
   case — they have no truth label).
5. **Full-population regression check** (owner instruction 2026-07-26: the 72
   were FILTERED from the full set and no regression is acceptable there
   either): re-run every d59k event with an in-beam bundle — **883 events,
   925 in-beam bundles, 153 STM tags** — with the round's knobs ON
   (`score_full.py` vs the stored d59k verdicts).  On the non-adjudicated
   bundles the tagger and the doc-61 AI scan agreed and the owner confirmed
   by silence, so EVERY flip there is a regression candidate and is reviewed
   individually (fit dQ/dx + AI-scan reasoning) before the round may ship.
6. The round ships only on: fixes ≥ 1, regressions = 0, gate PASS, and the
   full-population flips each individually justified.

## 4. Round records

### Round 0 — reference (work root `work-stmcamp-r0`)

All 72 baseline events re-run at the pre-campaign HEAD (2a821fd2) with the
production flag set.  Score: **56 correct / 16 wrong**, exactly the d59k
stored picture (the 15 false STMs + the miss).  This root is the A-arm for
every later byte-identical gate and the `--ref` for scoring.

### Round 1 — `accept_guards` (work roots `r1off`, `r1`, `r1b`)

- Knob-off gate: 10 mixed events with the round-1 binary, knobs off, vs r0:
  **GATE PASS** (nusel TSVs diff-identical; mabc-pr.zip and pctree-pr member
  content hashes identical).  Compiled-config proof: `accept_guards` key
  absent off / present on.  `wcdoctest-clus` 565/565.
- First knob-on arm (`r1`, cap ratio2 > 1.1): 6 fixed, **1 regression** —
  283463:14 (owner "STM is fine"), a sub-MIP track whose accepting call has
  ratio2 = 1.48.  The diagnosis table had been mis-read (its 1.48 overlooked
  when claiming "correct max 0.93").  The evaluation loop caught it.
- Cap raised to 2.0 (285443 stays dead through its 6.7 cm desert, 289832
  through its spike — both call-independent).  Re-run (`r1b`):
  **6 FIXED / 0 REGRESSED / no collateral flips** —
  fixes 278794:7, 285366:13, 285443:7, 289832:10, 290294:12, 409546:8.
  Score 62/72 correct (was 56).  SHIPPED.

### Round 2 — `proton_muon_guard` (work roots `r2off`, `r2`)

- Knob-off gate with the final (round-1 + round-2) binary: 10 mixed events vs
  r0, **GATE PASS** (member-content hashes identical).  Compiled-config
  proof: `proton_muon_guard` key absent off / present on.  `wcdoctest-clus`
  565/565.
- Cumulative arm (`-stm-guards -stm-proton-guard`), all 72 events:
  **7 FIXED / 0 REGRESSED / no collateral flips** — round 1's six plus the
  recovered miss **62613:17**.  The owner-confirmed proton rejections
  288859:9 (delta-ray path) and 319809:20 (ks1 = 0.063, ratio3 = 1.22) are
  unchanged, as designed.  Score **63/72 correct** (was 56).  SHIPPED.

### Full-population check of rounds 1+2 — two amendments (work roots `r2full`, `r2fullb`)

Protocol step 5 (owner instruction) applied retroactively to the shipped
rounds: all **883 events / 923 in-beam bundles** re-run with `-stm-guards
-stm-proton-guard` at 24-way (owner authorized 24 CPUs for this campaign)
and compared bundle-by-bundle against the stored d59k verdicts.

The FIRST full arm (`r2full`) surfaced two flips outside the 72-bundle
baseline that the baseline could not see, each fixed by an amendment:

- **290844:3 — REGRESSION, fixed by the desert cathode-join exemption.**
  A genuine cathode-crossing STM (textbook Bragg, 208 ke/cm at the stop)
  was killed by an 8.3 cm *instrumental* charge desert where the fit crosses
  the CPA join between drift volumes.  Amendment: a below-threshold run whose
  endpoints lie in different (or unknown) TPC volumes is instrumental and
  does not count toward the desert veto, capped at 4× `guard_desert_cm` so a
  genuine ≥ 12 cm gap still kills even across the join.
- **405740:14 — marginal proton-guard reversal, fixed by tightening
  `guard_proton_ks1` 0.045 → 0.040.**  Its proton veto was reversed at
  ks1 = 0.044 (0.001 inside the threshold) though the record shows no Bragg
  trend and no boundary entry.  At 0.040 the reversal no longer fires;
  the round-2 target 62613:17 is recovered at ks1 = 0.030 with 25% margin.

The re-run (`r2fullb`, after both amendments): **8 flips total = the 7
baseline fixes + exactly one non-adjudicated flip**, zero baseline
regressions, zero bundles missing:

- **63163:6 (stm 0→1)** — the proton-guard reversal now tags it STM.  End
  region is muon-consistent (ks1 = 0.032, ratio3 = 0.597), so the guard is
  behaving as designed; the doc-61 AI scan dissents on *topology* ("horizontal
  beam-aligned track… exiting downstream wall"), which dQ/dx cannot see.
  Flagged for owner review; not a baseline regression.

### Round 3 — `cathode_guard` (work roots `r3off`, `r3full`, `r3offb`, `r3fullb`, `dbg1`, `dbg2`)

- Knob-off gate with the final binary: 10 mixed events vs r0, **GATE PASS**
  (member-content hashes identical).  Compiled-config proof: `cathode_guard`
  key absent off / present on.  `wcdoctest-clus` passes.
- **The full arm caught a sign bug the offline survey could not.**  The first
  round-3 arm (`r3full`) produced ZERO new flips: the guard never fired.
  A single-event debug run (`dbg1`, evt 72586, with the guard's inputs
  logged) showed `fdx=+1, bb.x=[-201.45,-0.45] cm, cathode_x=-201.45` — the
  code had selected the ANODE face.  `IAnodeFace::dirx()` is the face-normal
  sign, pointing from the anode INTO the drift volume (toward the cathode),
  and the sensitive-volume BoundingBox is component-wise normalized, so the
  cathode is at max x when fdx > 0 — the initial selection was inverted.
  After the one-character fix (`fdx < 0` → `fdx > 0`), `dbg2` rejects
  72586:17 with "stop 2.4 cm from the cathode, end peak 1.95 MIP".
  *Found in passing, NOT touched: the shipped `dist_to_anode` helper in
  `check_stm_conditions` (`anode_x = (fdx < 0) ? first : second`) appears to
  have the SAME inversion — on SBND apa0 it returns the distance to the
  cathode.  It is in the always-on path, so any correction is a behavior
  change needing its own knob and validation round; flagged for the owner.*
- Corrected cumulative arm (`r3fullb`, all 883 events / 923 in-beam bundles):
  **9 baseline FIXED / 0 REGRESSED** — rounds 1+2's seven plus **72586:17**
  and **392200:27** (the two cathode-truncation flats).  The two protected
  near-cathode genuine STMs 289343:9 and 281148:13 are untouched, as
  designed.  Two non-adjudicated flips, both justified:
  - **56463:12 (stm 1→0)** — exactly the §1.5 third case (stop 3.9 cm from
    the cathode, end peak 1.25 MIP); the doc-61 AI scan had itself dissented
    from the tagger here ("truncated cathode crosser" ⇒ verdict nu), so this
    flip moves the tagger INTO agreement with the scan.
  - **63163:6 (stm 0→1)** — the round-2 proton-guard flip, unchanged from
    the r2fullb review (muon-consistent end, AI dissent is topological;
    flagged for owner review).
  Score **65/72 correct** (was 56).  SHIPPED.

### Production default ON (2026-07-26, work roots `dbg3`, `dbg4`)

Owner instruction: the campaign knobs are production defaults going forward.
Implemented at the SBND level only — `cfg/pgrapher/experiment/sbnd/
clus.jsonnet` (clus_pr + pr arg defaults), `wct-pr-perevt.jsonnet` (TLA
defaults) and `run_nusel_evt.sh` (env defaults) all flip to ON; the **C++
defaults in `TaggerCheckSTM.cxx` stay false**, so no other detector's
compiled config or output changes.  Verified both ways on evt 72586:

- plain run (no flags): all three `... ON` config lines present, 72586:17
  rejected by the cathode guard, `nusel` TSV identical to the `r3fullb` arm;
- opt-out run (`-no-stm-guards -no-stm-proton-guard -no-stm-cathode-guard`):
  byte-identical to the pre-campaign `r0` reference (GATE PASS).

Round 4's `stm_anode_dist_fix` and `stm_second_track_guard` joined the SBND
defaults after their validation (records below); the full opt-out set is now
the five `-no-*` flags in the status header, and the post-round-4 knob-off
gate (`r4boff`) passed with all five explicit.

### Round 4a — `anode_dist_fix` (work roots `r4aoff`, `r4afull`)

The dist_to_anode inversion fix, on the owner's instruction ("Please fix …
as well", then "After fixing it, default on, please after validation").

- Knob-off gate (`r4aoff`, 10 mixed events, all `-no-*` flags): **GATE
  PASS** vs r0.  Compiled-config proof both ways.  `wcdoctest-clus` passes.
- Cumulative full arm (`r4afull`, 883 events, knob ON on top of rounds
  1–3): STM verdict flips **identical to `r3fullb`** — the fix changes no
  STM verdict in this sample.  A direct arm-vs-arm diff of the
  stm/tgm/fc/lm/label columns over all 923 in-beam bundles found exactly
  two differences, both benign evaluation-status changes −1 → 0 with the
  final label unchanged ('nu-candidate' both arms): 67394:18 (fc column)
  and 283009:23 (stm column).  These are bundles whose evaluation now
  completes (the corrected anode distance lets the anode-clipped-TGM check
  run to a decision) without a verdict change.  VALIDATED ⇒ default ON.

### Round 4b/4c — `second_track_guard` (work roots `r4boff`, `dbg6`, `r4bfull`)

- Knob-off gate (`r4boff`, 10 mixed events, all five `-no-*` flags): **GATE
  PASS: 10 event(s) byte-identical (work-stmcamp-r0 vs work-stmcamp-r4boff)**.
  `wcdoctest-clus` passes; freshness proof done (lib 20:35:49 > last edit).
- Smoke (`dbg6`): both designed rejections fire with the designed logs —
  353223 "second_track_guard: cluster 15 rejected: 28.6 cm leftover past
  the kink with straightness 0.972 (a second track, not a Michel)" and
  349241 "second_track_guard: cluster 15 rejected: other-track segment
  20.4 cm at 1.00 MIP (a second track)".
- Cumulative full arm (`r4bfull`, 883/883 events rc=0, everything ON):
  **11 baseline FIXED / 0 REGRESSED** — rounds 1–3's nine plus **349241:15**
  and **353223:15**, exactly the two designed targets and nothing else.
  Non-adjudicated flips: only the same two justified ones (56463:12,
  63163:6) — the guard causes ZERO collateral flips in the whole
  population.  Score **67/72 correct** (was 56).  SHIPPED, default ON.

### Scoreboard after round 4

| | round 0 | round 2 | round 3 | round 4 |
|---|---|---|---|---|
| false STMs (of 15) | 15 | 9 | 7 | **5** |
| missed STMs (of 1) | 1 | 0 | 0 | **0** |
| correct STM kept (36) | 36 | 36 | 36 | 36 |
| correct non-tag kept (20) | 20 | 20 | 20 | 20 |

Remaining 5 false STMs: 278662:1 (vertex, spike inside the correct range),
402330:1 (vertex fan, off-track steiner count does not separate — §1.7),
48895:17, 321107:13, 321371:18 (flat tracks mid-volume — §1.7).  Round 5+
material.

## 5. Files

| path | what |
|---|---|
| `stm_campaign/extract_stm_diag.py` | offline diagnostics from stored pctree-pr tarballs (read-only) |
| `stm_campaign/run_round.sh` | one evaluation arm: fresh work root, 72 baseline events, production flags + round knobs |
| `stm_campaign/score_round.py` | confusion vs owner baseline + fixes/regressions vs reference arm |
| `stm_campaign/score_full.py` | full-population flip classifier vs stored d59k (protocol step 5), joins the doc-61 AI-scan verdicts |
| `stm_campaign/resume_round.sh` | resume an interrupted arm: re-runs only events without rc=0 in `<root>/.status` |
| `scan-d59k/stm-baseline.tsv` | the doc-62 truth set |
| `/home/xqian/tmp/stm-campaign/` | scratch: extracted diagnostics TSVs |

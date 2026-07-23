# 28 — Offset-0 LASSO-economy retune campaign (run 039252)

Status: **COMPLETE** (2026-07-22). Verdict: the doc-19 economy transfers; no dial closes the residual class (phase 4). Owner-commissioned follow-up to doc 27
(decision point 3): re-derive the doc-19 LASSO/ladder economy at the physical
(offset-0) frame, on top of the doc-27 rc14 op point, to close the residual
~24 net scan-agreed losses that the flag-window recalibration could not reach.

## Repro

```
cd pdvd
# phase-0 forensics of the residual losses (classes + lever cross-tab)
python3 docs/qlmatch/scripts/rl_forensics.py rc14
# per-sweep-point recovery counts (after scoring a tag)
python3 docs/qlmatch/scripts/rl_recovery.py rl1 rl2 ...
# scoring any sweep tag (identical machinery to docs 26/27)
python3 ql_display/ql_agree_score.py --tag <tag> \
  --truth-time-map work/ql_scores/tm0/time_map.json \
  --truth-time-shift -13.507 --truth-uid-map-tag tm0k
```

Frame under retune (= doc 27 rc14; all sweeps below inherit it):
`PDVD_QL_EXTRA_OFFSET_US=0`, `PDVD_DRIFT_SPEED_{BOT,TOP}_MMUS=1.4794`,
`PDVD_QL_ANODE_MARGIN_CM=1.0`, `PDVD_QL_CATHODE_EXT1_CM=2.0`,
`PDVD_LIGHT_SUFFIX=_tmerge`, `PDVD_QL_PIN_CONFIRMS_RESCUE=1`,
`PDVD_QL_CATHODE_SOLO_KS=0.30`, `PDVD_QL_CATHODE_SOLO_C2N=40`,
`PDVD_QL_SC1_EVICT_KS_MARGIN=0.05`.

Baselines (identical scoring machinery):

| point | agree | agree% | phantom | missed | missed% |
|---|---|---|---|---|---|
| tm0 (offset 13.507, production) | 752 | 86.6% | 116 | 91 | 10.8% |
| rc14 (offset 0, doc 27 op point) | 728 | 88.6% | 94 | 115 | 13.6% |

Target: close the missed gap (115 → ~91) without giving back the phantom win
(94 vs 116); adoption per the doc-23 rule (a metric improves, none regresses)
vs the tm0 row.

## Phase 0 — forensics: the 50 residual pairs

`rl_forensics.py rc14` — the 50 pairs missed at rc14 but not at tm0
(scorer-mapped uid space; the scorer drops truth entries whose cluster fails
the geometric uid map *before* scoring, so missed-list uids are directly the
rc14 cluster uids):

| class | n | note |
|---|---|---|
| mis-pick (cluster auto-selected a different flash) | 44 | truth bundle present at the truth flash |
| no-truth-bundle | 5 | no candidate at the truth flash at all |
| unmatched (truth bundle present, cluster picked nothing) | 1 | ks 0.41, c2n 512 — legitimately weak |

Truth-bundle strength is **exactly 0.00 in 44/45** present cases (none in the
0–0.05 sub-cutoff band): every loss is the LASSO zeroing an intact candidate,
never the strength cutoff shaving a small coefficient. Metrics on the truth
bundles are healthy (ks median ≈ 0.10; 17/45 carry the ladder `consistent`
flag) — confirming doc 27's "single-pair re-equilibration" diagnosis.

Winner cross-tab (44 mis-picks; T<W = truth ks better than winner ks):

| winner class | T<W | W<T | notes |
|---|---|---|---|
| plain LASSO win (strength ~0.8–0.98, no special flags) | 10 | 12 | 8 of the 10 T<W truths are `consistent`-flagged |
| rescue adoption (winner strength 0.00) | 5 | 8 | cluster_rescue (production nm3 gates ks .15 / c2n 3) adopting the wrong flash |
| wrong pin (winner `xtpc_pin`) | 3 | 3 | pin binds the cluster elsewhere, rivals culled |
| sc1 non-pin | 3 | 0 | all three truths `consistent` — the doc-27 margin (0.05) did not fire |

Reading: ~21 pairs are "truth strictly better by light, LASSO still zeroed
it" — the economy class proper. The 23 W<T pairs are scan-vs-metric
disagreements (many truths carry boundary/truncation flags `C/B/2/W`, where
ks is least reliable) — reachable only through flag-aware weighting
(boundary L1 down-weight, chi2_relax scale), not through raw thresholds.

Economy lever inventory (production values at rc14):

| lever | production | source | env |
|---|---|---|---|
| `lasso_lambda` | 0.2 (doc 19 phase 4; pre-tune 0.1) | runner default | `PDVD_QL_LASSO_LAMBDA` |
| `strength_cutoff` | 0.05 | C++ | `PDVD_QL_STRENGTH_CUTOFF` |
| `lasso_flag_weight` / `lasso_boundary_weight` | true / 0.2 | qlmatching.jsonnet literals (:445) | `PDVD_QL_LASSO_BWEIGHT` (scale only) |
| `chi2_relax` | true | qlmatching.jsonnet literal (:501) | — |
| hc ladder c2n | 12/12/12/30 | runner defaults (doc 19) | `PDVD_QL_HC_*_C2N` |
| hc ladder ks | 0.06/0.09/0.10/0.08 | C++ | `PDVD_QL_HC_*_KS` |
| `delta_charge/light/shape` | 0.01/0.025/0.01 | C++ | `PDVD_QL_DELTA_*` |
| `bkg_weight` | 0.5 | C++ | `PDVD_QL_BKG_WEIGHT` |
| cluster_rescue | ON, nm3 gates (ks .15/c2n 3/ratio .5–2, additive precull) | runner default (doc 20) | `PDVD_QL_CLUSTER_RESCUE` |

## Phase 1 — round 1: single-lever diagnostics

Each point = rc14 frame + one lever moved (18 evts, tags rl1..rl8):

| tag | change vs rc14 | probes |
|---|---|---|
| rl1 | `LASSO_LAMBDA=0.1` | pre-doc-19 regularization (denser solutions) |
| rl2 | `LASSO_LAMBDA=0.15` | half-step |
| rl3 | `LASSO_LAMBDA=0.3` | opposite direction (sparser) |
| rl4 | `STRENGTH_CUTOFF=0.02` | cutoff share (expect null — 44/45 zeros) |
| rl5 | `LASSO_BWEIGHT=0.1` | boundary bundles favored harder |
| rl6 | `LASSO_BWEIGHT=0.4` | boundary favor halved |
| rl7 | `CLUSTER_RESCUE=0` | rescue mis-adoption share (diagnostic only — rescue is a doc-20 adoption) |
| rl8 | `BKG_WEIGHT=0.3` | background column weight |

Results (`rl_recovery.py`, target = the 50 residual pairs; headline from
`scores.json`):

| tag | change | agree | agree% | phantom | missed | recov | new-miss | new-phm |
|---|---|---|---|---|---|---|---|---|
| rc14 | (base) | 728 | 88.6% | 94 | 115 | — | — | — |
| rl1 | λ 0.1 | 727 | 87.7% | 102 | 116 | 2 | 3 | 6 |
| rl2 | λ 0.15 | 727 | 87.9% | 100 | 116 | 0 | 1 | 4 |
| rl3 | λ 0.3 | 726 | 89.1% | 89 | 117 | 0 | 2 | 1 |
| rl4 | cutoff 0.02 | 727 | 88.3% | 96 | 116 | 0 | 1 | 3 |
| rl5 | bweight 0.1 | 724 | 88.1% | 98 | 119 | 1 | 6 | 4 |
| rl6 | bweight 0.4 | 719 | 89.5% | 84 | 124 | 0 | 9 | 3 |
| rl7 | cluster_rescue off | 703 | 88.3% | 93 | 140 | 0 | 24 | 0 |
| rl8 | bkg 0.3 | 727 | 88.6% | 94 | 116 | 0 | 1 | 0 |

**All eight levers are nulls or net-negative; rc14 dominates every point on
agree.** rl4 confirms the phase-0 prediction (strengths are exact zeros, not
sub-cutoff shavings). rl7 is the informative one: with cluster_rescue OFF the
13 rescue-winner pairs are *still not recovered* (0/50) while 24 of the
rescue's legitimate adoptions die — the rescue fills a genuine vacuum; it
does not steal wins the truth bundle would otherwise get.

Sweep-infrastructure fix found on the way: `PDVD_QL_LASSO_BWEIGHT` had been
dead since the doc-19 adoption — the adopted literal
`lasso_boundary_weight: 0.2` in `qlmatching.jsonnet` (:446) collided with the
surviving doc-19 sweep conditional (:574), a jsonnet duplicate-field compile
crash whenever the arg was set. Fixed by folding the null-default arg into
the literal (toolkit cfg commit; knob-off compiled JSON verified
byte-identical, knob-on key lands numerically).

## Phase 2 — closing the chi2-scale axis + the case study

| tag | change | recov | new-miss | new-phm |
|---|---|---|---|---|
| rl9 | λ 0.05 | 2 | 7 | 8 |
| rl10 | delta_light 0.05 (×2) | 0 | 0 | 1 |
| rl11 | delta_shape 0.02 (×2) | 0 | 1 | 0 |

λ keeps trading worse with depth; the chi2 error floors are pure nulls. The
global regularization/scale axis is **closed: 11 points, none recover more
than 2/50, all at higher collateral.**

**Case study — the loss is flag acquisition, not the solver.** evt298735
cluster 4000136 (plain mis-pick, both candidates `consistent`, truth ks
0.056 vs winner ks 0.093, truth strength 0.00):

| frame | truth-flash bundle | winner |
|---|---|---|
| tm0 | ks 0.060, c2n 2.0, flags **[B2c]**, strength **0.975**, selected | (same bundle) |
| rc14 | ks 0.056, c2n 2.5, flags **[c]** — boundary flags GONE — strength 0.00 | rival flash, ks 0.093, [c], strength 0.919 |

The candidate list is identical in both frames (66 bundles, same rivals,
near-identical metrics). What changed is the truth bundle's
`at_x_boundary`/`two_boundary` flags: at physical charge placement its
endpoints sit outside the boundary-flag windows, so it loses the
`lasso_flag_weight` L1 privilege (×0.2 down-weight) it enjoyed at the pulled
frame — and with equal L1 weights the near-degenerate rival wins the
coefficient competition. This also explains why rl5/rl6 (bweight scale) are
nulls: scaling a privilege the truth bundle *no longer has* cannot help it.

Note the rc14 frame runs `PDVD_QL_ANODE_MARGIN_CM=1.0` — *halved* from the
production 2.0 by the doc-26 crosser tuning — which directly narrows the
`at_x_boundary` acquisition window. The doc-26 margin sweep predates the
doc-27 knobs, so the margin × flag-economy interaction was never tested at
the current op point.

## Phase 3 — round 3: flag-window re-widening at the rc14 frame

| tag | change vs rc14 | probes |
|---|---|---|
| rl12 | `ANODE_MARGIN=2.0` | restore production boundary-flag width |
| rl13 | `ANODE_MARGIN=1.5` | half-step |
| rl14 | `CATHODE_EXT1=2.5` | cathode window (differs from doc-27 rc2: doc-27 knobs now on) |
| rl15 | `ANODE_MARGIN=2.0` + `CATHODE_EXT1=2.5` | combined |

| tag | recov | new-miss | new-phm |
|---|---|---|---|
| rl12 | 0 | 1 | 0 |
| rl13 | 0 | 1 | 0 |
| rl14 | 0 | 1 | 3 |
| rl15 | 0 | 2 | 3 |

**All null.** Restoring the production anode margin re-acquires nothing: the
13.507 µs shift moves endpoints ~2 cm, past any tested window extension, and
doc 27 already showed the deeper cathode extensions (rc2 ext1 2.5, rc10
ceiling 3.5, rc1 xtpc tol 14) null or trade ~1:1.

## Phase 4 — flag-transition census and campaign verdict

Flag transitions on the 45 residual truth bundles present in both frames
(tm0 → rc14):

| transition | n | reading |
|---|---|---|
| −xtpc_pin | 10 | pins that never FORM at offset 0 (admission geometry, distinct from the doc-27 purge fix, which needs a formed pin) |
| −at_x_boundary | 7 | boundary L1 privilege lost (phase-2 case study) |
| −consistent | 6 | hc-ladder edge flips (doc 27 rc3 showed tier loosening is net-destructive) |
| −xtpc_consistent / −xtpc_scenario1 | 3 / 3 | crosser candidacy lost with the boundary flag (`:4123`) |
| +at_x_boundary / +at_cathode / +xtpc_cathode_rescued | 3 / 2 / 3 | gains on the wrong side of the ledger |

**Verdict: the doc-19 LASSO economy transfers to the physical frame as-is.**
Fifteen sweep points (rl1–rl15) covering regularization (λ 0.05–0.3), the
strength cutoff, the boundary L1 privilege scale in both directions, the
background column weight, both chi2 error floors, the rescue system, and the
boundary-flag windows themselves recover at most 2 of the 50 residual pairs,
always at larger collateral; rc14 dominates every point on agree. The
residual class is not a solver or regularization artifact: it is
boundary-flag acquisition (pin formation, `at_x_boundary`, hc edges) at
physical charge placement, where the flag windows sit ~2 cm short of the
populations the pulled frame used to catch — plus ~23 scan-vs-metric
disagreements the light metrics alone cannot arbitrate.

What WOULD reach the residual class (out of scope, each a campaign of its
own):
1. **Flag geometry rework** — acquire boundary flags from detachment-aware
   endpoints (doc-23 `robust_endpoint_*` machinery) rather than raw extremal
   points, so physical placement stops moving bundles out of the windows.
2. **Pin-formation admission** at offset 0 (the −xtpc_pin ×10): re-derive
   the xtpc candidate admission (`:4123`) so a crosser pair whose boundary
   flags are lost can still pair; the doc-27 pin knob then protects it.
3. **A rescan at the physical frame** — the 23 winner-ks-better pairs are
   scan verdicts recorded at the pulled frame; some may simply be correct
   matches whose truth changed with the geometry.

## Adoption status

Unchanged from doc 27, now with the economy axis exhausted: **rc14 stands as
the best offset-0 operating point** at 728 agree (88.6%) / 94 phantom / 115
missed vs production tm0 752 (86.6%) / 116 / 91. Against the no-regression
rule it still fails on agree/missed (−24/+24) while winning phantoms (−22)
and rate (+2.0). The owner's options are now sharper: (a) stay at the pulled
frame (adopting the doc-26 tail merge alone); (b) adopt rc14 accepting the
coverage trade — this campaign establishes no dial closes it; (c) commission
one of the three out-of-scope reworks above.

**DECISION (owner, 2026-07-22): option (a) adopted** — tail merge becomes
the production default (`run_light_evt.sh` `PDVD_FLASH_TAIL_MERGE:-1`),
pulled frame/velocity/cuts unchanged. rc14 and the offset-0 frame remain
opt-in via the runner envs; decision (c) reworks stay open. Doc-23 track
scoreboard at (a): A correct, C fixed (both halves on the 21.2k-PE merged
flash, strength 0.96/0.97), B still mis-picked (wrong flash 207 µs early in
every configuration — an independent single-cluster LASSO mis-pick, its
bot half also carries the doc-23 +8.6 µs charge anomaly). Bee links: see
doc 26 adoption note.

## Files

- toolkit `cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet`: dead
  `PDVD_QL_LASSO_BWEIGHT` fix (duplicate-field crash; knob-off byte-identical).
- `scripts/rl_forensics.py` (phase-0/case-study classifier),
  `scripts/rl_recovery.py` (residual-pair recovery counter).
- Sweep tags `work/039252_<idx>_{rl1..rl15}` + `work/ql_scores/rl*`
  (records, keep).

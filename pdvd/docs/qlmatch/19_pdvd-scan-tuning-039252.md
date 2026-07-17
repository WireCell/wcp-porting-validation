# 19 — Scan-driven QLMatch tuning campaign (PDVD run 039252)

Status: IN PROGRESS (Phase 0 complete).

Goal: tune automatic QLMatch (PE uncertainties, priority logic, ladder, flash
selection, LASSO regularization, rescues) so its output matches the frozen
hand/AI scan reference on **long tracks** (>= 25 cm or >= 100 points; short
tracks explicitly out of scope).

## Repro

```bash
# score any tag against the frozen truth (baseline shown):
cd pdvd && python ql_display/ql_agree_score.py --tag cathxa --self-check

# reprocess the 18 events under a new tag (matching-only, reuses _keep clustering):
PDVD_MAX_JOBS=6 PDVD_LIGHT_SUFFIX=_keep PDVD_QL_ROBUST_TRIM=1 \
  PDVD_QL_ROBUST_WALK_FLOOR=1 PDVD_QL_XTPC_CATHODE_TOL_CM=10 \
  ./run_clus_evt.sh -calib 039252 all -s <tag>
```

## Reference truth (frozen for the whole campaign)

- Gold hand scan: `work/ql_labels/wfresc/labels-evt298567.json`
  (44 matches incl. 15 human-added / 84 rejected_auto; tier "gold").
- AI scan: `ql_display/decisions-cathxa/decisions-evt*.jsonl`, 17 events
  298581..298805 (keep 860 / reject 419 / add 122; confidence
  high 691 / med 525 / low 185).
- Objective tiers: gold + high + med. Low-confidence verdicts excluded from
  the objective (owner decision 2026-07-17), reported informationally.
- Join key: (event, flash time +-0.5 us, main-cluster uid) — never flash_gid
  (renumbers with flash admission; remap_scan_state.py precedent).

Caveats (accepted): the AI scan judged the `cathxa` auto output (reference
circularity on rejects/adds), and all 18 events of one run are used for tuning
with **no holdout** — the next data run is the real validation.

## Scorer

`ql_display/ql_agree_score.py` — metrics over auto-selected long-track
bundles: **agree** (scanner kept), **phantom** (scanner rejected), **unknown**
(no verdict), and coverage of scan positives: **missed** (no auto counterpart;
flash-cut subclassified). Flag splits sized over judged autos. Outputs
`work/ql_scores/<tag>/scores.{json,md}` (append-only, M13).

Self-check on `cathxa` reproduces the documented unfiltered headline exactly:
joined 1279, keep 860, agree 67.2%, unjoined rejects 0.

## Phase 0 — Baseline (tag `cathxa`, toolkit b9cb1561)

Objective tiers, long tracks, tol 0.5 us:

| evt | agree | phantom | agree% | unknown | missed | npos | missed% |
|---|---|---|---|---|---|---|---|
| 298567 | 33 | 10 | 76.7% | 8 | 10 | 43 | 23.3% |
| 298581 | 45 | 17 | 72.6% | 0 | 4 | 49 | 8.2% |
| 298595 | 47 | 15 | 75.8% | 0 | 1 | 48 | 2.1% |
| 298609 | 38 | 22 | 63.3% | 0 | 4 | 42 | 9.5% |
| 298623 | 40 | 24 | 62.5% | 0 | 2 | 42 | 4.8% |
| 298637 | 42 | 11 | 79.2% | 0 | 3 | 45 | 6.7% |
| 298651 | 39 | 24 | 61.9% | 0 | 9 | 48 | 18.8% |
| 298665 | 39 | 23 | 62.9% | 0 | 3 | 42 | 7.1% |
| 298679 | 33 | 21 | 61.1% | 0 | 6 | 39 | 15.4% |
| 298693 | 32 | 28 | 53.3% | 0 | 7 | 39 | 17.9% |
| 298707 | 45 | 11 | 80.4% | 0 | 3 | 48 | 6.2% |
| 298721 | 38 | 23 | 62.3% | 0 | 5 | 43 | 11.6% |
| 298735 | 42 | 24 | 63.6% | 0 | 9 | 51 | 17.6% |
| 298749 | 34 | 10 | 77.3% | 0 | 3 | 37 | 8.1% |
| 298763 | 45 | 13 | 77.6% | 0 | 2 | 47 | 4.3% |
| 298777 | 61 | 14 | 81.3% | 0 | 7 | 68 | 10.3% |
| 298791 | 45 | 38 | 54.2% | 0 | 9 | 54 | 16.7% |
| 298805 | 53 | 19 | 73.6% | 0 | 4 | 57 | 7.0% |
| **all** | **751** | **347** | **68.4%** | 8 | **91** | 842 | **10.8%** |

Flag splits over judged autos (phantom% = phantoms / judged carrying flag):

| flag | judged | phantom | phantom% |
|---|---|---|---|
| xtpc_pin | 410 | 62 | 15.1% |
| xtpc_scenario1 | 492 | 125 | 25.4% |
| **xtpc_cathode_rescued** | **45** | **31** | **68.9%** |
| xtpc_consistent | 512 | 136 | 26.6% |
| consistent | 566 | 121 | 21.4% |
| two_boundary | 263 | 44 | 16.7% |
| at_cathode | 436 | 136 | 31.2% |
| at_x_boundary | 635 | 169 | 26.6% |
| close_to_PMT | 95 | 21 | 22.1% |
| window_truncated | 243 | 77 | 31.7% |
| sat_flash | 645 | 128 | 19.8% |

First read: the cathode-rescue path (`xtpc_cathode_rescued`) is 69% phantom —
the single worst admission path, confirming the xtpc pin/rescue phantom mode
as the top target (Phase 3). Missed rate is dominated by non-flash-cut misses
(0 of 91 are flash-cut), i.e. the flash survives but no auto bundle picks it —
LASSO/ladder/rescue territory, not flash admission.

## Phase 1 — Pull analysis + disagreement taxonomy (`ql_display/ql_pull_diag.py`)

Repro: `python ql_display/ql_pull_diag.py --tag cathxa --out work/ql_scores/cathxa/pull_diag.json`

Pulls (pe−pred)/σ, σ rebuilt exactly as `TimingTPCBundle::examine_bundle`
(verified by 2%-level chi2 reproduction on 700/751 confirmed matches; the 51
excluded are merged/xtpc-joint dumps). Note: `quality_params` in the calib
dump omits `pe_err_lowpe_frac/knee` (active 2.0/10.0 verified by the
reproduction guard) — dump gap to close in Phase 2.

| family | subset | n | mean | rms | median | mad |
|---|---|---|---|---|---|---|
| cath_xa | nom | 4129 | 2.08 | 4.79 | 0.62 | 1.50 |
| cath_xa | sat | 1458 | 0.53 | 1.03 | 0.55 | 1.08 |
| pmt | nom | 3930 | 0.53 | 1.75 | 0.15 | 0.85 |

Readings:
- **Saturated cathode-XA channels are already calibrated** (RMS 1.03):
  `chi2_sat_inflate 0.5` is NOT too big — keep it.
- **Nominal cathode-XA errors are far too small** in the 10–500 pred-PE range
  (per-bin mean pulls +2.8…+4.0, RMS 5–7.7, MAD 2.1–2.9): systematic
  underprediction (measured ≫ predicted) that the current frac 0.60 does not
  cover. Target: family-scoped larger frac / lowpe params (Phase 2).
- PMTs mildly under-inflated (RMS 1.75), slight overprediction at high pred
  (mean −1.1 at 20–50 PE).

Exclusive admission-path taxonomy of objective long-track judged autos:

| path | agree | phantom | phantom% |
|---|---|---|---|
| strength_only | 44 | 103 | 70.1% |
| ladder_B2_good | 182 | 85 | 31.8% |
| xtpc_pin | 335 | 53 | 13.7% |
| xtpc_scenario1 | 18 | 41 | 69.5% |
| xtpc_cathode_rescued | 14 | 31 | 68.9% |
| ladder_B1_clean | 155 | 30 | 16.2% |
| ladder_B4_miss | 3 | 4 | 57.1% |

Missed-positive diagnosis: 73/91 = cluster matched to the WRONG flash
(candidate for the right flash exists but lost), 18/91 = cluster left
unmatched with a candidate present, 0 = no candidate. Misses are an
assignment problem, not an admission problem — each wrong-flash fix removes
a phantom and a miss together.

Phase order implications: the top phantom buckets are `strength_only` (103)
and the loose xtpc paths `scenario1`+`cathode_rescued` (72 combined, ~69%
phantom each); `xtpc_pin` itself is comparatively healthy (13.7%).

## Phase 2 — Per-family PE-error knob + calibration runs

Toolkit knob (commit 787a5da8, byte-identical gate PASS 44 archives x
idx{0,12} vs cathxa, non-vacuous via the new `pe_err_lowpe_*` dump keys):
`pe_err_family_channels/floor/frac/lowpe_frac/lowpe_knee` — family-scoped
override of the global PE-error model in BOTH paths (Opflash LASSO
floor/frac; bundle-chi2 `per_opdet_perr` incl. lowpe branch).  PDVD jsonnet
params `pe_err_{cath,pmt}_*`, runner envs `PDVD_QL_PEERR_{CATH,PMT}_*`.
Families: cathode XAs (4-11), PMTs (14-17, 20-39).

Calibration runs (fresh tags, matching-only reprocess; baseline row repeated):

| tag | cath frac/lowpe_frac@knee | pmt frac | agree% | phantom | missed | unknown |
|---|---|---|---|---|---|---|
| cathxa (base) | 0.60 / 2.0@10 | 0.60 | 68.4% | 347 | 91 | 8 |
| tune_pe1 | 0.55 / 2.6@80 | 0.75 | **69.9%** | 319 | 102 | 51 |
| tune_pe2 | 0.60 / 3.0@120 | 0.90 | 70.0% | 316 | 104 | 57 |
| tune_pe3 | 0.55 / 2.2@50 | 0.65 | 69.2% | 331 | 98 | 36 |

Reading: the rescale trades ~30 phantoms for ~10 misses; pe1/pe2 equivalent
within noise — **pe1 chosen** (milder).  The full benefit is gated on
re-tightening the ladder chi2 ceilings to the new chi2 scale (Phase 4; the
current hc ceilings 35/60 were set for the old inflated scale).
chi2_sat_inflate stays 0.5 (Phase 1 showed the saturated subset already
calibrated).  Cross-check `tune_sat035` (chi2_sat_inflate 0.35, all else
baseline): 1693/6525 bundle chi2 values move on evt298637 but NO selection,
flag, or score changes anywhere — every headline number identical to
baseline.  Verdict: the current saturated-PD widening is neither too big nor
too small to matter in [0.35, 0.5]; keep 0.5.

## Phase 3 — xtpc / selection quality gates

Audit findings (code): the joint-pin gate is purely geometric (d < dmax +
axis collinearity; light only picks WHICH coincident flash, never whether
the pair deserves a pin), pinned bundles are exempt from the strength
cutoff and monopolize `cull_inconsistent`; scenario-1 flags are granted at
EVERY coincident flash whose T0 offset makes the halves touch.
`two_boundary` audited healthy (16.7% phantom, no B3 excess) — no knob.

Toolkit knobs (commit 4004c546, all default OFF; gate 44 archives x
idx{0,12} byte-identical; runner envs in 428ae41):

| knob | cut (from scan separations) | attacks |
|---|---|---|
| `xtpc_pin_min_strength` 0.02 | pinned bundle loses strength-cutoff exemption at/below floor | phantom pins (strength p50 0.00 vs agree p10 0.88) |
| `xtpc_sc1_light_gate` (ks 0.3 / c2n 50) | sc1/xtpc-consistent flags need own light | sc1 phantoms (ks p50 0.40, c2n p50 76) |
| `xtpc_cathode_ks_max` 0.32 | rescue survivors need ks too | rescue phantoms (69%) |
| `postcull_unflagged` (ks 0.30 / c2n 20) | post-fit cull of strength-only picks | largest bucket (103 @ 70%) |

`tune_qg1` (gates ON, baseline error model): **agree 81.5%** (from 68.4%),
phantom **167** (from 347), missed 108 (from 91), unknown 89. Flag splits:
xtpc_pin 15.1→6.3%, scenario1 25.4→4.5%, cathode_rescued 68.9→10.0%.
The +17 missed are phantom-kills leaving their cluster unmatched —
Phase 5 rescue territory.

## Phase 4 — ladder ceilings + LASSO regularization sweep

Combined base `tune_c1` = pe1 error model + all four Phase 3 gates:
**agree 81.8%**, phantom 161, missed 119, unknown 127 — slightly better
than gates-alone on agree/phantom, worse on missed (the tighter errors
kill a few more borderline true matches; rescue territory).

Config plumbing (commit e869109c toolkit, 1167907 runner envs): PDVD
jsonnet null-default params for the ladder ceilings
(`hc_{clean,good,tb,miss}_{ks,c2n}`, `hc_miss_min_ndf`) and the LASSO
regularization terms (`lasso_lambda`, `delta_{charge,light,shape}`,
`bkg_weight`, `strength_cutoff`, `lasso_boundary_weight`), all
key-suppressed — compiled JSON byte-identical when unset.

One-at-a-time sweep on top of the `tune_c1` env (9 tags, matching-only
reprocess, all 18 events each):

| tag | knob change | agree% | phantom | missed | unknown |
|---|---|---|---|---|---|
| tune_c1 (base) | — | 81.8% | 161 | 119 | 127 |
| tune_c1_hc12 | hc c2n ceilings 12 (miss 30) | **82.7%** | **152** | **113** | 98 |
| tune_c1_hc20 | hc c2n ceilings 20 (miss 45) | 82.1% | 158 | 115 | 109 |
| tune_c1_sc08 | strength_cutoff 0.08 | 82.0% | 158 | 121 | 129 |
| tune_c1_lam02 | lasso_lambda 0.2 | **83.2%** | **145** | 126 | 127 |
| tune_c1_lam005 | lasso_lambda 0.05 | 82.5% | 153 | 119 | 162 |
| tune_c1_bkg03 | bkg_weight 0.3 | 81.9% | 160 | 119 | 127 |
| tune_c1_bkg08 | bkg_weight 0.8 | 81.8% | 161 | 120 | 127 |
| tune_c1_fm15 | flash_minPE 15 | 82.0% | 157 | 127 | 130 |
| tune_c1_fs8 | flash_sel_minPE 8 | 81.7% | 162 | 119 | 128 |

Reading: two winners. `hc12` — tightening the high-consistent chi2/ndf
ceilings from 35/60 to 12/30 (matching the recalibrated chi2 scale from
Phase 2, where chi2/ndf of true matches sits at ~1-3) improves all three
metrics at once. `lam02` — doubling the LASSO L1 strength kills the most
phantoms (161→145, sparser solutions drop weak double-assignments) at the
cost of +7 missed. `bkg_weight`, `flash_sel_minPE` are null;
`strength_cutoff` 0.08 and `flash_minPE` 15 slightly negative; `lam005`
inflates the unknown (low-confidence) pool. Next: combine hc12+lam02, and
let Phase 5 rescues attack the missed pool.

## Phase 5 — shared-flash-aware rescues

PDVD runs `shared_flash=true`, so the per-run `empty_rescue`/`cluster_rescue`
were skipped by construction (configure() warning).  New toolkit knobs
(commit 6bb43db5, default OFF; runner plumbing wcp 555836f):

- `empty_rescue_shared`: joint emptiness — a physical flash (by flash id,
  identical across ports) is empty only when NO drift side holds a
  surviving bundle; the best pre-LASSO snapshot candidate ACROSS sides is
  adopted under the `rescue_metric_max` bar (pin-locked,
  reassign-only-if-strictly-better, deterministic tie-breaks).
- `cluster_rescue_shared`: the existing per-run cluster-centric ADD-only
  adoption (ks/c2n/ratio gates), run from the shared rounds per side.

Byte-identity gate: 88 archives x idx{0,12} knobs-off vs cathxa, label
`p5gatef`, wcdoctest-match 23/23.  GOTCHA (cost a 5-run false-FAIL hunt):
gates vs cathxa must run WITH the cathxa generation env
(`PDVD_QL_ROBUST_TRIM=1 PDVD_QL_ROBUST_WALK_FLOOR=1
PDVD_QL_XTPC_CATHODE_TOL_CM=10`) — a bare knobs-off run legitimately
differs (containment admits ~11% fewer vis candidate-points).

Results on top of `tune_c2` (= tune_c1 + hc12 + lam02, the phase-4 winner
combination):

| tag | rescues | agree% | phantom | missed | unknown |
|---|---|---|---|---|---|
| tune_c2 | none | 84.1% | 136 | 120 | 97 |
| tune_c2_cr | cluster (0.25/15/0.3–3.0) | **84.3%** | 137 | **109** | 98 |
| tune_c2_er | empty @0.5 | 85.5% | 115 | 162 | 257 |
| tune_c2_er01 | empty @0.1 | 85.2% | 120 | 151 | 201 |
| tune_c2_ercr | both | 85.7% | 115 | 154 | 259 |

Verdict: **cluster rescue adopted** (missed 120→109 at +1 phantom; ADD-only,
sentinel `QLclusrescue: rescued N` fires on 11/36 sides).  **Empty rescue
rejected at both thresholds**: with 184 flashes and ~105 clusters most
flashes are LEGITIMATELY empty, so joint-emptiness adoption force-fills
them — it floods the output with never-scanned pairings (unknown 97→257)
and steals correct matches via reassignment (missed 120→162; 34-47
rescues/event at 0.5).  The knob stays available (default OFF) for
detectors with higher cluster/flash ratios.

## Phase 6 — combined operating point

Final = `tune_c2_cr`.  Before/after on the 18-event objective (long
tracks, gold + high/med AI verdicts):

| | cathxa (baseline) | tune_c2_cr (final) |
|---|---|---|
| agree% (kept-auto) | 68.4% | **84.3%** |
| phantoms | 347 | **137** |
| missed | 91 (10.8%) | 109 (12.9%) |
| per-event agree% range | 53.9–78.2% | 72.5–97.9% |

Flag-split phantom rates, baseline → final: xtpc_pin 15.1→5.6%,
scenario1 25.4→5.1%, cathode_rescued 68.9→9.1%, consistent 21.4→17.2%,
two_boundary 16.7→4.4%, at_cathode 31.2→7.8%, window_truncated
31.7→17.0%, sat_flash 19.8→7.2%.

The +18 missed vs baseline is the price of killing 210 phantoms: the
quality gates remove some true matches whose light evidence is genuinely
poor (window-truncated / saturated candles), and cluster rescue claws
back 11.  Precision-recall trade accepted — the scan showed the baseline's
extra "matches" were 70%+ phantoms in every flagged family.

### Operating-point knob inventory (runner env => active value; toolkit default OFF)

BAKED as `run_clus_evt.sh` defaults 2026-07-17 (each env individually
revertible — export empty or =0 per the in-script comments; the robust-trim
/ xtpc-cathode-tol baseline is baked ON too since the tuned values assume
it).  Verified: bare-env runs (only `PDVD_LIGHT_SUFFIX=_keep`) on idx{0,12}
reproduce `tune_c2_cr` byte-identically (label `p6bake`, 88 archives).

| env (run_clus_evt.sh) | value | knob | phase |
|---|---|---|---|
| PDVD_QL_PEERR_CATH_FRAC | 0.55 | pe_err_family_frac[cath] | 2 |
| PDVD_QL_PEERR_CATH_LOWPE_FRAC | 2.6 | pe_err_family_lowpe_frac[cath] | 2 |
| PDVD_QL_PEERR_CATH_LOWPE_KNEE | 80 | pe_err_family_lowpe_knee[cath] | 2 |
| PDVD_QL_PEERR_PMT_FRAC | 0.75 | pe_err_family_frac[pmt] | 2 |
| PDVD_QL_PIN_MIN_STRENGTH | 0.02 | xtpc_pin_min_strength | 3 |
| PDVD_QL_SC1_LIGHT_GATE | 1 (ks 0.3 / c2n 50) | xtpc_sc1_light_gate | 3 |
| PDVD_QL_CATHODE_KS_MAX | 0.32 | xtpc_cathode_ks_max | 3 |
| PDVD_QL_POSTCULL | 1 (ks 0.30 / c2n 20) | postcull_unflagged | 3 |
| PDVD_QL_HC_{CLEAN,GOOD,TB}_C2N | 12 | hc_*_c2 ladder ceilings | 4 |
| PDVD_QL_HC_MISS_C2N | 30 | hc_miss_c2 | 4 |
| PDVD_QL_LASSO_LAMBDA | 0.2 | lasso_lambda | 4 |
| PDVD_QL_CLUSTER_RESCUE | 1 (0.25/15/0.3–3.0) | cluster_rescue_shared | 5 |
| (pre-existing baseline) | ROBUST_TRIM=1 WALK_FLOOR=1 XTPC_CATHODE_TOL_CM=10 | robust trim + xtpc cathode rescue | docs 15/16 |

### Caveats

- **No holdout**: all 18 events of run 039252 were used for tuning (user
  decision); these numbers are in-sample and WILL be optimistic on new
  runs.  The next processed run is the real validation.
- Reference circularity: the AI scan judged the cathxa auto output;
  low-confidence AI verdicts are excluded from the objective (reported as
  "unknown").
- Every toolkit knob defaults OFF; gates p2gate/p3gate/p5gatef (88
  archives x idx{0,12} each) prove byte-identity vs cathxa knobs-off at
  787a5da8 / 4004c546 / 6bb43db5.

## Campaign plan

- Phase 0: this scorer + baseline. DONE.
- Phase 1: pull analysis (PE-error calibration data) + phantom taxonomy. DONE.
- Phase 2: per-family PE error knob (cathode XA vs PMT) + chi2_sat_inflate scan.
- Phase 3: xtpc pin / cull_inconsistent / two_boundary correctness knobs.
- Phase 4: ladder + flash_minPE + LASSO regularization sweep.
- Phase 5: shared-flash-aware rescues (only if a missed-rate gap remains).
- Phase 6: combined operating point `tune_final`, runner defaults, final table.

All toolkit changes are default-OFF knobs, byte-identical when off (abtest
gate per change); PDVD activation via `run_clus_evt.sh` env defaults
(cathode-XA operating-point precedent).

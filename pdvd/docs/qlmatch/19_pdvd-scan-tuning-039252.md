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

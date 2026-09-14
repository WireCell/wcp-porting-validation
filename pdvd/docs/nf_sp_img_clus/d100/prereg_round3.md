# doc pdvd/100 round 3 — QtoL after the top gain: pre-registration (written 2026-09-13, before the knob was built and before any round-3 arm ran)

Owner, 2026-09-13, after the gain bundle went to production (wcp `cabbdd0b`): *"Let's proceed with the QtoL tuning then
after the scaling of the gain for top. Please update this parameters. I believe we have the old hand scan results that may
be used to help to validate the matching results."*

## 0. The value, measured before this file (read-only on production's own dumps)

`pdvd/ql_light_calib/fit_qtol_crossers.py --tag p100flip` (today's production, gain 0.889 + C 0.8630; its calib-dump
bundles equal `p99rwon`'s on the three events checked, and the fit reproduces round 1's `p99rwon` numbers exactly):
global median Σmeas/Σpred **0.833** [0.662, 1.464], n = 192 strict crosser anchors → **QtoL = 0.094 × 0.833 = 0.0783**.
- per run: 039252 0.757 (n 47) → 0.0712; 039253 0.857 (32) → 0.0806; 039349 0.860 (113) → 0.0808. One global value is
  adopted; no per-run value. The hand-scanned run (039252) is the low one.
- per PD group: cathode XA 0.822 (n 192), PMTs 2.130 (n 158).

**The degeneracy, stated now.** Prediction = QtoL × VUVEfficiency. The per-type factors in toolkit
`protodunevd/qlmatching.jsonnet` (cathode ×10.116, membrane ×1.655, PMT ×0.352; docs/qlmatch/12 §4, owner 2026-07-14)
were fitted so each type's median meas/pred is 1 at QtoL 0.094. A QtoL of 0.0783 is arithmetically the same as scaling
all three factors by 0.833. It is a global renormalisation, driven by the cathode XAs that dominate the sums: expected
cathode 0.822 → ≈0.99 and PMT 2.130 → ≈2.56. The PMT group was already ×2 off on pre-gain production (`p99wflip` 2.274);
no per-type factor is refitted in this round.

## 1. The knob (default = today, compiled config byte-identical)

- toolkit `cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet`: function arg `qtol=0.094`, `QtoL: qtol`. PDHD / SBND
  import their own qlmatching.jsonnet and are untouched.
- wcp `pdvd/wct-clustering.jsonnet`: TLA `ql_qtol = 0.094`, forwarded.
- wcp `pdvd/run_clus_evt.sh`: `PDVD_QTOL` (unset = the driver default) → `-S ql_qtol=`. d100_arms.sh already unsets
  that name.
- G1, on proof tag `p101cfg` (one event per run: 039252_0, 039253_15, 039349_7, `PDVD_CLUS_COMPILE_ONLY=1`): compiled JSON
  after the edits at default == before, md5; with `PDVD_QTOL=0.0783` the only difference is the QtoL leaf.

## 2. The arm

`p101q`: production as shipped (stm/run_campaign.sh staging via scripts/stage_ql_tag.sh, default imaging tag `pvdimg`;
run_clus_evt.sh -calib -save-pctree with `PDVD_LIGHT_SUFFIX=_keep`; the window from readout_window_ticks.txt; then
run_pr_evt.sh -nu -stm-fit) with **`PDVD_QTOL=0.0783`** and nothing else set; all 120 events of stm/events.txt;
`setarch -R`; pin `/home/xqian/tmp/p100/libpin_p100b` (Clus ef0822ec, the binary `p100flip` ran). Base: `p100flip`.
Fork `scripts/d100r3_arms.sh` of d100_arms.sh (that script stays as committed).

## 3. Criteria

**Q1 closure.** `fit_qtol_crossers.py --tag p101q`: global median Σmeas/Σpred within **[0.95, 1.05]**. Per run and per
group reported beside it. FAIL → stop and report; no second iteration of the value.

**Q2 hand-scan agreement (the matching validation).** `pdvd/ql_display/ql_agree_score.py` against its frozen 039252 truth
(gold `work/ql_labels/wfresc/labels-evt298567.json` + AI scan `ql_display/decisions-cathxa/`, objective tiers gold/high/med,
long tracks, tol 0.5 µs), on the 18 events of run 039252:
- truth cluster uids mapped by geometry with `--truth-uid-map-tag keep` (the `keep` dumps carry all 1529 truth uids; the
  July `tm0k`/`cathxa` dumps were dropped in the 2026-09-04 cleanup, so the doc 26 reference 764/118/78 cannot be replayed);
- no `--truth-time-shift` (production is still at the truth's 13.507 µs pull);
- `--truth-time-map work/ql_scores/tm0/time_map.json` decided on `p100flip` BEFORE `p101q` is scored: whichever of with /
  without joins more truth (judged + covered positives) is used for every arm, and recorded;
- scored into new `work/ql_scores/` tags; `p99wflip` (pre-gain production) scored the same way as the reference for the
  gain's own matching cost (trial, uid map only: 701 / 100 / 140 missed vs `p100flip` 678 / 97 / 163).

Rule (doc 23, the house adoption rule of docs 23/26/27/28): against `p100flip`, **a metric improves (agree, phantom or
missed) and none regresses.** A move of one or two pairs against a metric is reported as churn (doc 26 adopted the tail
merge at −1 agree / +1 missed as "agreement-neutral"); a larger regression is a FAIL.

**Q3 the STM chain downstream (all 120 events), against `p100flip`.**
- census `d99rw_census.py --arms p100flip p101q`; record-free transitions `d99_population.py` with pairs
  p100flip:p101q, p101q:p100flip and the null p100flip:p100flip;
- grade on the owner-corrected record: `p100flip` on `/home/xqian/tmp/p100/carry_r2/latest_on_p99rwon_corrected.json`
  (its pctree equals p100c = p99rwon's), `p101q` on the same record carried by `d100_carry_verdicts.py --base-arm p99rwon
  --arm p101q` (same pctree keeps the key; otherwise geometry, status ok only);
- criterion, round 2's bar: `p101q` Michel purity AND is_stm purity each within **1.5σ** (binomial, errors in quadrature)
  of `p100flip`'s. FAIL → no flip; movers by volume and record verdict reported, nothing retuned. Unjudged movers are
  counted and reported (they are candidates for an owner look, not a gate).

## 4. The flip (only if Q1, Q2 and Q3 pass)

- driver `pdvd/wct-clustering.jsonnet` default `ql_qtol` 0.094 → 0.0783; toolkit default stays 0.094.
- compiled-config proof on `p101cfg`: after-flip default == before-flip + `PDVD_QTOL=0.0783`, md5, 3 events.
- F1: `p101flip` = production, no overrides, all 120 events == `p101q` (`d99rw_identity.py --nt all`: pctree, tlas, PR
  trees, mabc-pr) and calib-evt*.json byte-identical.
- docs: doc 100 §8 + status; doc 09 §4 QtoL line; docs/qlmatch/12 status (the "per type = 1 at 0.094" bookkeeping becomes
  a global renormalisation, PMT group pays); the toolkit comment at the literal.

Not in this round: a per-type VUVEfficiency refit, per-run QtoL, the PMT ×2 residual, the 128 nm distance slope.

# doc qlmatch/29 — pre-registration of the arms (written 2026-09-13, after the Step A forensics and before any arm ran)

Owner, 2026-09-13: investigate the history of the PDVD Q/L matching degradation, check drift velocity and LASSO, "investigate
and improve the situation". Scope answers the same day: **flip if all gates pass**; step-0 split with **config-level arms
only** (no old-binary rebuilds).

## 0. What Step A established (records in this directory)
- Reproduction: `d29_attribution.py` recomputes p99wflip 701/100/140, p98voffq 681/90/157, p100flip 678/97/163,
  p101q 672/95/169 exactly from the scorer's own functions (`attribution.txt` §1).
- Step 2 (SP v7 wire order, p99wflip → p98voffq), 35 newly missed (`forensics_step2.txt`): 21 cluster-identity changes
  (17 top), 10 mis-picks (truth bundle ks 0.098 and c2n 2.8 unchanged, LASSO strength 0.964 → 0), 3 no truth bundle,
  1 unmatched; truth bundles lose xtpc_pin ×3.
- Step 3 (top gain, p98voffq → p100flip), 20 newly missed (`forensics_step3.txt`): 11 mis-picks (8 plain-LASSO winners;
  truth c2n 1.14 → 0.86, strength 0.92 → 0), 8 cluster-identity changes (all top), 1 no truth bundle.
- So the recoverable-by-QL class is the LASSO mis-pick (the owner's hint); the identity changes are upstream clustering.
- Doc 28 phase 1 found every LASSO/economy lever null or net-negative at the July rc14 frame (rl1–rl8). This scan asks the
  same question on today's clustering, where the mis-pick population is new.

## 1. Step 0 split (config-level, run 039252 idx 0..17, clustering only, `scripts/d29_arms.sh`)
| arm | imaging source | wires compiled | change vs production config | answers |
|---|---|---|---|---|
| `q29base` | `pvdimg` | v7 | none | control: calib dumps must equal `p100flip` byte for byte on 18/18 |
| `q29v6` | `keep` (July v6 imaging, July frames) | v6 (`PDVD_CLUS_WIRES`) | none | vs tm0k 764/118/78 (July code, same imaging and light): code drift 07-22 → today + wrapped charge ON + SAVE_ASSOC |
| `q29v6nw` | `keep` | v6 | `wrapped_channel_charge=false` | the wrapped-charge flip on July imaging |
| `q29v7nw` | `pvdimg` | v7 | `wrapped_channel_charge=false` | the wrapped-charge flip on today's production (also a lever, §2) |
- `q29v6` → `p99wflip` (July frames, v7 imaging; identical dumps to d27fresh) = wires v6 → v7 in imaging + clustering.
- tm0k's QL parameters cannot be read back (dumps deleted 2026-09-04); git shows no QL default change after 07-22, which is
  the assumption behind "q29v6 vs tm0k = code + wrapped". q29v6 is scored both with the keep map and without a map (tm0k's
  mode; keep's clustering is the truth's), and a residual beyond ±2 pairs is reported as the no-knob commits as a group
  (315f9653f, 95c10cd16, 11bec7f4), not split.
- The imaging-provenance guard stays armed: `keep`'s archives say v6 and the arm compiles v6.

## 2. Lever scan on production (`pvdimg`, v7, production config + ONE lever; clustering only; 18 events)
| arm | lever (runner env / TLA) | production | family |
|---|---|---|---|
| `q29l1` | `PDVD_QL_LASSO_LAMBDA=0.1` | 0.2 | LASSO (mis-pick class) |
| `q29l2` | `PDVD_QL_LASSO_LAMBDA=0.15` | 0.2 | LASSO |
| `q29l3` | `PDVD_QL_LASSO_LAMBDA=0.3` | 0.2 | LASSO |
| `q29l4` | `PDVD_QL_STRENGTH_CUTOFF=0.02` | 0.05 | LASSO |
| `q29l5` | `PDVD_QL_BKG_WEIGHT=0.3` | 0.5 | LASSO |
| `q29l6` | `PDVD_QL_LASSO_BWEIGHT=0.1` | 0.2 | LASSO |
| `q29l7` | `PDVD_QL_LASSO_BWEIGHT=0.4` | 0.2 | LASSO |
| `q29l8` | `PDVD_QL_HC_{CLEAN,GOOD,TB}_C2N=35`, `PDVD_QL_HC_MISS_C2N=60` | 12/12/12/30 | ladder chi2 scale (moved with the gain) |
| `q29l9` | `PDVD_QL_PIN_MIN_STRENGTH=` (empty: pin keeps the strength-cutoff exemption) | 0.02 | pin (xtpc_pin lost ×5) |
| `q29v7nw` | `wrapped_channel_charge=false` | true | pre-QL charge |
| `p101q` (exists) | `PDVD_QTOL=0.0783` | 0.094 | amplitude |
- Each lever arm's compiled config must differ from `q29base`'s only in its lever's leaves (checked on idx 0).
- Then at most ONE combination: the two best single levers that each pass the rule below, together (`q29c1`).
- *Note added 2026-09-13 19:15, after the single-lever scores and before `q29c1` ran:* `q29l6` and `q29l7` set the same
  parameter (`lasso_boundary_weight`) to opposite values, so they cannot be combined. The combination pairs the best
  passing arm with the best passing arm of a DIFFERENT parameter. On the preview (l1–l6) the passing arms are `q29l6`
  (merit +10) and `q29l3` (−1); `q29l8` and `q29l9` fail on full truth. So `q29c1` = `PDVD_QL_LASSO_BWEIGHT=0.1` +
  `PDVD_QL_LASSO_LAMBDA=0.3`, whatever `q29l7`'s halves say.

## 3. Adoption rule (fixed now)
Against `p100flip` (= `q29base`), scored with `ql_agree_score.py --truth-uid-map-tag keep`, no time map / shift:
1. **Doc 23 rule:** at least one of agree / phantom / missed improves, and none regresses by more than 2 pairs.
2. **Stability:** the same holds on the even-idx and odd-idx halves separately (9 events each), with the regression
   allowance 2 per half; and on common truth (entries mapping in every scored arm).
3. **Candidate = the best passing arm** by (agree − phantom − missed) change on full truth; ties → the smaller change.
4. **STM gate (120 events, clustering + PR, production staging + the lever):** `d100r3_grade.py` on the owner-corrected
   record carried to the candidate — is_stm purity and Michel purity each within 1.5σ of `p100flip` (or higher); crosser
   closure (`fit_qtol_crossers.py`) reported.
5. **Flip (owner: flip if all gates pass):** the lever's production default changes in its owning layer
   (`run_clus_evt.sh` env default, or the driver TLA default), then F1: production staging with no overrides, 120 events ==
   the STM-gate arm (`d99rw_identity.py --nt all` + calib dumps byte-identical), plus the compiled-config proof.
If nothing passes 1–3, production is unchanged and doc 29 says so; the losses stay attributed.

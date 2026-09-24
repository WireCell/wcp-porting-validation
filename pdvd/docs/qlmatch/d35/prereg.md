# doc qlmatch/35 -- pre-registration of the pre-flip checks and the flip (written 2026-09-23, before any doc-35 arm ran)

Owner, 2026-09-23: "Can you proceed as you recommended? After confirmation, please update the production default for
PDVD, please update the relevant md file, commit and push."  The recommendation (doc 34 sec 7): doc 11 sec 6 and doc 12
rerun on the `q34tk1` operating point; the other 102 events for crashes and match-rate tails; a flip-equivalence gate;
then the default change.  Added here, following the doc 29 sec 7 precedent for a PDVD Q/L flip: the 120-event STM gate
(clustering + PR) on the production binary.  **The go is conditional on every STOP bar below; any STOP halts the flip
and is reported to the owner.**

## 0. What is flipped (the candidate, unchanged from doc 34)
- Light: `run_light_evt.sh` `PDVD_SAT_REPAIR_MODE` default `tot`; `PDVD_HIT_INT_SAMPLES` default follows the mode (1
  under `tot`, 0 otherwise; twoside + int is the doc 32 over-fill and only ever explicit).
- Q/L: `run_clus_evt.sh` `PDVD_QL_LASSO_W_UNRAILED` default 1, `PDVD_QL_KS_SAT_TOL` default 0.3075.
- Production light record: `stm/run_campaign.sh` `PDVD_LIGHT_SUFFIX` default `_keep` -> `_tot`, a new light tag made
  by the flipped runner with `_keep`'s argument set (`PDVD_FLASH_TAIL_MERGE=0`, the d32 light-arm recipe).
- Toolkit C++/jsonnet defaults stay OFF (doc 12 / doc 29 precedent); no toolkit commit.
- `run_clus_evt.sh` warns (stderr, output unchanged) when the ToT-tuned Q/L defaults meet light whose `.wct-light.json`
  does not say `saturation_repair_mode: tot` -- that combination is the control twin (`q34ck1`), not the candidate.

## 1. Arms (new tags only, M13)
- Pin `/home/xqian/tmp/p35/libpin_prod` = a copy of `local/lib` at 2026-09-23 (md5 list `libpin_prod.md5`; Clus
  `129461a1991e`, Match `f3cfd9b19072` = the ks_sat_tol lib of `libpin_q34`).
- `q35ctl` = production today: staging `pvdimg`, `_keep` light, no Q/L env, `run_clus_evt.sh -calib -save-pctree` +
  `run_pr_evt.sh -nu -stm-fit`, 120 events, setarch -R (`scripts/d35_arms.sh`, fork of `d29_stm_arms.sh`).
- `q35tk` = the same with `_q32ti` light and `PDVD_QL_LASSO_W_UNRAILED=1 PDVD_QL_KS_SAT_TOL=0.3075`.
- `q35flip` (after the edit) = `setarch -R stm/run_campaign.sh q35flip all` with the pin and NO PDVD_* env: the
  production entry, so the light-suffix default is under test too.
- Light `_tot` (after the edit) = `d32_light_arms.sh` with `ARM=_tot`, no ENVS (flipped runner, `_keep` argument set).

## 2. Checks and bars
**C1 -- doc 11 sec 6 rerun** (`d31_sat_terms.py`, pooled over the 120 dumps of `q34tk1` and `q31ctl`): railed-channel
terms on selected matches.  STOP if the candidate's selected-match railed median meas/pred lies outside [0.5, 2.0]
(doc 11 read 3.66 on twoside with the old calibration).  The rest of the table is reported.

**C2 -- doc 12 rerun** (`fit_qtol_crossers.py` harvest, strict crosser anchors, all three runs; `scripts/d35_crossers.py`):
- rail-excluded global median Sum(meas)/Sum(pred): STOP if `q34tk1` / `q31ctl` differs from 1 by more than 0.05
  (compared on the control, not on 1.0: production already reads ~0.83);
- rail-inclusive / rail-excluded on `q34tk1`: STOP if > 1.10 (doc 34's QtoL trigger; a QtoL arm would be needed).

**C3 -- the other 102 events** (runs 039253 + 039349; `scripts/d35_match_tails.py`), on the pinned pair `q31ctl` ->
`q34tk1` and on the production-binary pair `q35ctl` -> `q35tk`.  Long cluster = the scorer's filter; matched = >= 1
auto-selected bundle; mover = a long cluster in both dumps whose auto-selected flash-time set differs beyond 1.5 us
(calibrated on the spent run 039252: 162 vs the time-mapped 161; per-event fraction 0.04-0.24, pooled 0.137).
- crash: STOP if any event with complete control outputs lacks candidate outputs (calib dump + mabc zip; for the q35
  arms also pctree + PR `tracking-pr.root` with the CheckSTM line, 039349_30 excepted if the control also has no STM
  candidate there);
- STOP if any event (control >= 10 matched long clusters) keeps < 75 % of the control's matched long clusters;
- STOP if the pooled 102-event mover fraction exceeds 0.20 (1.5x the scanned run: the scan would not represent them);
- FLAG (reported with the event list, not a stop): per-event matched change < -10 % or mover fraction > 0.30.

**C4 -- STM gate** (`d116_grade.py --det pdvd --cells A0=q35ctl,T=q35tk --extra-record smx116 --owner-record own116v
--split-half`): the doc 113 rule -- each of `is_stm` purity / efficiency and Michel purity / efficiency passes if
T - A0 >= -0.020 in both the NEG and POS scenarios.  STOP on any FAIL or UNDECIDED.  Split-half reported.  Before the
grade is read, `d103_flip_gate.py` of `q35ctl` vs `d116vflip` (T_stm_michel, T_rec_charge) is reported, so a key
mismatch is known before the NEG/POS bounds are.

**C5 -- binary transfer** (report only): `q35ctl` vs `q31ctl` calib dumps (cmp, 120); movers `q35ctl` -> `q35tk` vs
`q31ctl` -> `q34tk1` on run 039252 (161 time-mapped).  If the dumps differ, the doc says the margin was scored on
`libpin_q34`.

**F1 -- flip equivalence** (after the edit; STOP on any failure):
- compiled config (3 events: 039252_0, 039253_15, 039349_7): clus post-flip bare == pre-flip + the two Q/L env;
  post-flip escape (`PDVD_QL_LASSO_W_UNRAILED=0 PDVD_QL_KS_SAT_TOL=`) == pre-flip bare;
- light: `_tot` == `_q32ti` on 120 / 120 by `abtest/hash_archive.py`; `.wct-light.json` equal after the output-dir
  path is normalised, and it shows `saturation_repair_mode: tot`, `int_samples: true`; light escape
  (`PDVD_SAT_REPAIR_MODE=twoside`) on evt298567 compiles to `_keep`'s config (no repair-mode key, no int_samples);
- runtime: `q35flip` == `q35tk` on 120 / 120 calib dumps (cmp) and `d99rw_identity.py` (pctree, tlas, PR trees,
  mabc-pr); `q35flip` differs from `q35ctl` somewhere (negative control); `qp.ks_sat_tol` present in `q35flip` dumps,
  absent in `q35ctl`.

## 3. What this does not re-open
The hand-scan margin verdict of doc 34 (q34tk1 passes non-inferiority on the pinned binary) is taken as given; C3-C5
check that it transfers and breaks nothing downstream.  Known open items stay open and are restated: the top-up
calibration shortfall, a lower tolerance rung, the 26-cluster owner-record deficit, and the runner's
`PDVD_FLASH_TAIL_MERGE=1` default that the production light record (`_keep`, and now `_tot`) never used.

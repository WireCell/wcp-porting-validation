# 21 — Relaxed second-chance rescue for long unmatched clusters (039252, round 2)

Status: sweep COMPLETE, defaults OFF — **awaiting owner operating-point decision**.
Follows doc 20 (nm3 adoption). Toolkit commits `3c90eae8` (C++ knob family) +
`717e12f5` (PDVD cfg threading); wcp `994accc` (runner envs + census extension).

## Repro

```bash
# census what-if grid (offline, on the nm3 dumps):
python ql_display/unmatched_census.py --tag nm3 --relaxed-whatif
# sweep tags (18 evts each, matching-only from _keep):
for t in nm4a nm4b nm4c; do for i in $(seq 0 17); do ./scripts/stage_ql_tag.sh 39252 $i $t; done; done
env PDVD_LIGHT_SUFFIX=_keep PDVD_MAX_JOBS=6 PDVD_QL_CRESCUE_RELAX=1 ./run_clus_evt.sh -calib 39252 all -s nm4a
env PDVD_LIGHT_SUFFIX=_keep PDVD_MAX_JOBS=6 PDVD_QL_CRESCUE_RELAX=1 \
    PDVD_QL_CRESCUE_RELAX_KS=0.25 PDVD_QL_CRESCUE_RELAX_C2N=15 \
    PDVD_QL_CRESCUE_RELAX_RLO=0.3 PDVD_QL_CRESCUE_RELAX_RHI=3.0 ./run_clus_evt.sh -calib 39252 all -s nm4b
env PDVD_LIGHT_SUFFIX=_keep PDVD_MAX_JOBS=6 PDVD_QL_CRESCUE_RELAX=1 \
    PDVD_QL_CRESCUE_RELAX_KS=0.18 PDVD_QL_CRESCUE_RELAX_C2N=4 \
    PDVD_QL_CRESCUE_RELAX_RLO=0.5 PDVD_QL_CRESCUE_RELAX_RHI=2.0 ./run_clus_evt.sh -calib 39252 all -s nm4c
python ql_display/ql_agree_score.py --tag nm4a   # (b, c likewise)
# overpred side-study:
env PDVD_LIGHT_SUFFIX=_keep PDVD_MAX_JOBS=6 PDVD_QL_OVERPRED_TOTAL=8  PDVD_QL_OVERPRED_MAXCH=25 ./run_clus_evt.sh -calib 39252 all -s op4a
env PDVD_LIGHT_SUFFIX=_keep PDVD_MAX_JOBS=6 PDVD_QL_OVERPRED_TOTAL=3  PDVD_QL_OVERPRED_MAXCH=10 ./run_clus_evt.sh -calib 39252 all -s op4b
```

## 1. Problem and owner direction

After nm3 (doc 20), 95 long clusters remain non-matched; 91 are class-C —
a contained candidate bundle exists AT the true flash time but fails the tight
rescue gates, and the light score cannot rank the true flash among ~190
rivals. Owner direction (2026-07-17): (a) a RELAXED second-chance rescue tier
that cannot touch existing matches, restricted to long clusters, sweep gates
then owner picks; (b) survey PDHD/SBND for other ideas; opted IN to the
overpred-prefilter side-study, knowing it changes existing matches.

## 2. PDHD/SBND survey verdicts

- `empty_rescue_shared` (analogue of the empty-flash rescue PDHD+SBND run
  hard-ON): **already tested in the doc-19 round and REJECTED** — floods
  legitimately-empty flashes and steals correct matches at metric 0.5 and 0.1.
  Stays OFF.
- PDHD production `cluster_rescue` gates (ks 0.20 / c2ndf 8 / ratio 0.4–2.5):
  adopted as sweep point nm4a of the relaxed tier.
- `flash_minPE` tightening: doc-19 NULL (minPE 15, flash_sel_minpe 8). Not
  repeated.
- `pmt_nonlinearity` (SBND): SBND-PMT-specific, not applicable.
- Overpred ceilings: PDVD runs 15/50 (cathode-scoped) vs PDHD 3/10 and SBND
  2.9/4.3 (all-channel) — the one genuinely looser PDVD prefilter; §6.

## 3. The knob (toolkit `3c90eae8`/`717e12f5`, all default OFF)

`QLMatching` `cluster_rescue_relaxed{,_ks_max,_chi2ndf_max,_ratio_lo,
_ratio_hi,_min_length}`: after the tight `rescue_unmatched_clusters` pass, a
second pass over clusters STILL unmatched, only when
`get_length() >= min_length`; same pools (snapshot primary / precull
fallback), same score() and deterministic tie-break; adoption stamps
`flag_cluster_rescue_relaxed` on the bundle and the calib dump emits a
knob-gated per-bundle `cluster_rescue_relaxed` bool (precedent:
`xtpc_cathode_rescued`). Additive-only by construction — the pass filters to
unmatched clusters and only push-backs onto `flash_bundles_map`, never
reassigns.

Runner: `PDVD_QL_CRESCUE_RELAX=1` +
`PDVD_QL_CRESCUE_RELAX_{KS,C2N,RLO,RHI,MINLEN_CM}` (defaults when enabled:
PDHD point 0.20/8/0.4–2.5 @ 50 cm). Default **OFF**.

**Gates (knob off = byte-identical):** `relaxoff` tags idx 0/5/15 —
calib dumps byte-identical (`diff -q`) to nm3, mabc-all-apa/group0123/group4567
member-content hashes (hash_archive.py, field 1) == nm3, compiled config
identical to nm3def modulo tag paths with zero `cluster_rescue_relaxed` keys;
`wcdoctest-match` 23/23. Knob-on smoke `relaxon` idx 0: sentinel
`QLclusrescue-relaxed on: ks<0.2 c2ndf<8 ratio 0.4-2.5 min_len 50 cm`, 1
flagged auto-selected adoption. PDHD is covered by the same shared function
being exercised in these gates (relaxed block is a single `if` behind the
knob) + no PDHD config touched.

## 4. Census what-if (offline; `work/ql_scores/nm3/relaxed_whatif.md`)

min_len 50 cm is the sweet spot (trims wrongflash without losing recoveries;
100 cm costs a recovery). Sim at @50: pdhd +5 rec/9 wf, base +8/11 (+1 ph),
t18 +4/7. Known caveat (doc 20 §8): sim is direction-faithful, absolute
counts are not — the real tags below are the truth.

## 5. Real sweep (18 evts, tags `nm4a/nm4b/nm4c`, all min_len 50 cm)

Scorer vs frozen doc-19 truth (`work/ql_scores/<tag>/scores.md`), plus
per-adoption classification of the FLAGGED bundles
(`ql_display/relaxed_adoption_report.py`; wrongflash = known positive adopted
at a wrong flash, invisible to the scorer):

| tag | gates | agree | phantom | missed | unknown | adoptions | recovered | phantom(adopt) | wrongflash | unlabeled | precision | additivity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nm3 (base) | — | 747 | 137 | 95 | 117 | — | — | — | — | — | — | — |
| nm4a | .20/8/.4-2.5 | 748 | 137 | 94 | 130 | 16 | 1 | 0 | 5 | 10 | 0.17 | OK |
| **nm4b** | .25/15/.3-3 | **751** | 138 | **91** | 137 | 26 | **4** | 1 | 6 | 15 | 0.36 | OK |
| nm4c | .18/4/.5-2 | 747 | 137 | 95 | 122 | 5 | 0 | 0 | 3 | 2 | 0.00 | OK |

- **Additivity verified empirically in all three tags**: every nm3
  (cluster, flash-time) auto pair survives unchanged.
- nm4b strictly dominates: −4 missed (95→91) at +1 phantom; its 26 new
  matches = 4 provably right, 7 provably wrong (1 phantom + 6 wrongflash), 15
  rescan-only.
- Non-monotone gate interaction (real example): evt298567 uid 18 is
  RECOVERED at its true flash by nm4a's gates but the wider nm4b band admits
  a lower-score candidate at a truth-negative flash (phantom @2705.5 µs).
- nm4c is strictly bad (nothing recovered, 3 wrongflash).
- nm4b residual census: 91 missed = 87 C + 4 D (same 4 photon-model
  clusters); the well is nearly dry — even base gates on the residual would
  add only +4 with 7 more wrongflash.

## 6. Overpred prefilter side-study (tags `op4a` 8/25, `op4b` 3/10)

(NOT additive — changes existing matches by design; report-only.)

| tag | ceilings (tot/maxch) | agree | phantom | missed | agreed-match regressions | new agrees | pair churn |
|---|---|---|---|---|---|---|---|
| nm3 | 15/50 | 747 | 137 | 95 | — | — | — |
| op4a | 8/25 | 749 | **134** | 93 | 3 | 5 | 13/1493 (0.9%) |
| op4b | 3/10 (PDHD) | 749 | **126** | 93 | 8 | 10 | 49/1493 (3.3%) |

Tightening the cathode-scoped overpred ceilings is a mild NET WIN on this
sample — phantoms drop (−3 / −11), missed drops (−2 both) — but it is not
free: 3 (op4a) / 8 (op4b) currently-agreed matches are lost (uids listed in
the analysis output; e.g. evt298791 uid 104 @ −623.5 µs lost at both points).
The mechanism is the intended one: fewer wildly-overpredicting candidate
bundles reach the LASSO, so rival pressure drops. Unlike the relaxed tier
this reshuffles existing decisions, so adoption would need a rescan of the
churned pairs — flagged here as a candidate for a future round, not part of
the relaxed-tier decision.

## 7. Decision

Owner picks one of:
1. **Adopt nm4b** (base gates @ 50 cm): 4 fewer non-match long tracks per 18
   evts that are provably right, at the cost of 7 provably-wrong + 15
   unverified new matches — all flagged `cluster_rescue_relaxed` in the dumps
   so scans/analyses can segregate them.
2. **Adopt nm4a** (PDHD gates): smaller effect (−1 missed), similar wrong
   rate; not recommended on these numbers.
3. **Keep OFF**: the per-bundle light score has no more clean recall here;
   wait for the upstream levers (flash multiplicity / photon model).

Defaults stay OFF until the owner rules; adoption would follow the nm3
pattern (runner env defaults, toolkit stays OFF byte-identical).

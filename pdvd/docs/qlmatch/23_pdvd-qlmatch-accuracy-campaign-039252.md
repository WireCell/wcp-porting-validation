# 23 — PDVD Q/L matching accuracy campaign (run 039252)

Running doc for the correct-match-rate improvement campaign that follows the
doc-22 non-match explanation round. Owner-approved plan; one section per
phase, each phase ends with a commit (toolkit + wcp-porting-img) and push.

## Repro block

```sh
# Baseline = nm4b runner defaults (wcp 98950c0, toolkit knobs OFF).
# Scorecard (both owner metrics) on all 18 events of 039252:
cd pdvd
python ql_display/ql_agree_score.py --tag <TAG>        # metric 1: agreement vs frozen doc-19 truth
python ql_display/unmatched_census.py --tag <TAG>      # metric 2: missed long clusters
# Byte-identical-off gate (knobs OFF must reproduce nm4b):
python ../abtest/hash_archive.py work/039252_<idx>_<TAG>/calib-evt*.json \
       work/039252_<idx>_nm4b/calib-evt*.json          # idx 0/5/15 + mabc zips
```

## Owner metrics and baseline (fixed)

| metric | baseline (nm4b, 2026-07-17) |
|---|---|
| agreement with AI + owner hand scans (`ql_agree_score.py`, objective tiers) | **84.3%** (phantoms 137) |
| non-matched long tracks (missed) | **91** (66 wrong-flash, 25 unmatched — doc 22) |

Adoption rule per phase: flip runner defaults only if a metric improves and
neither worsens. All toolkit changes are default-OFF knobs, byte-identical
when off. Tags are fresh per phase (`ac1`, `ac2`, ...); nothing under an
existing tag is rewritten.

## Phase index

| phase | lever | status |
|---|---|---|
| 0 | bookkeeping: doc 22 artifacts + this skeleton committed | done |
| 1a | rescue blind-spot fix (4 PASSES_UNADOPTED clusters) | **done — ADOPTED** (agree +4, missed −4, phantom flat) |
| 1b | saturation-aware rescue ratio-high (clean-channel ratio REJECTED) | **done — ADOPTED** (agree +6, missed −6, phantom flat) |
| 2 | phantom-side overpred culls (wtrunc + pin; twins DEAD) | **done — ADOPTED** (phantom −20, agree +3, missed −3) |
| 3 | amplitude-model residual study | **done — NEGATIVE** (model unbiased on clean flashes; no correction knob) |
| 4 | joint-fit levers (cull keep-quality, cross-flash exclusivity) — contingent | pending |
| 5 | final validation + Bee sets for rescan | pending |

Background evidence: doc 20 (census + precull), doc 21 (relaxed tier sweep),
doc 22 (scan comparison; wrong-flash reframe; rescue blind spot), and the
re-ranking sims recorded in doc 22's follow-up (score-argmin breaks 43-50% of
the 751 agreed matches — per-bundle re-ranking is excluded as a lever).

---

## Phase 0 — bookkeeping

Committed the doc-22 deliverables (analysis only, no code):
`docs/qlmatch/22_pdvd-nonmatch-scan-comparison-039252.md`,
`ql_display/nonmatch_explain.py`, and the render context records
`ql_display/png-nonmatch-nm4b/evt*/context.jsonl`, plus this skeleton.
The PNGs themselves (18 renders, 4 events, ~25 MB) follow repo convention
(`*.png` gitignored) and stay untracked — regenerable from the nm4b dumps
with `ql_display/render_groups.py` per the doc-22 Repro block.

---

## Phase 1a — rescue blind-spot fix (`postcull_before_rescue`)

### Symptom

4 of the 91 nm4b misses are `PASSES_UNADOPTED` (doc 22 §4): a gate-passing
candidate exists at the truth flash, both rescue tiers are enabled, yet the
cluster was never rescued and ends the event unmatched.

### Root cause (traced, corrected from doc 22's reading)

Ordering, not flag bookkeeping. In `fit_round2_shared` the pipeline is
`rescue_unmatched_clusters` (QLMatching.cxx:2784) →
`cull_unflagged_lowquality` (:2788). PDVD runs the postcull ON in production
(`PDVD_QL_POSTCULL=1`, gates ks 0.30 / c2n 20). A bundle the LASSO selected
(strength ≈ 0.9) but that carries no quality flag and FAILS the postcull
gates is therefore still live in `flash_bundles_map` while the rescues run —
its main cluster lands in the off-limits `matched` set (:2927) — and is then
removed minutes later by the postcull, leaving the cluster with nothing. The
dump shows the blocker as `auto_selected=false` because `auto_selected` is
derived from the post-cull map (:3542-3605), which is why doc 22 initially
read it as a flag mismatch.

All 4 cases verified against the nm4b dumps (blocker = unflagged,
`POSTCULL victim` = fails ks>0.30 or c2n>20; truth candidate gates shown):

| evt | uid | blocker (fgid, t, str, ks, c2n) | victim | truth cand (fgid, t, PE, ks, c2n, ratio) | passes |
|---|---|---|---|---|---|
| 298693 | 4000076 | 148, 2551.4µs, 0.900, 0.321, 1.4 | yes | 117, 1630.9µs, 39266, 0.105, 1.3, 0.90 | tight |
| 298735 | 163 | 47, −1199.3µs, 0.913, 0.505, 4.5 | yes | 48, −1172.2µs, —, 0.163, 0.8, 1.56 | relaxed |
| 298735 | 94 | 166, 2847.7µs, 0.924, 0.370, 34.3 | yes | 192, 3833.1µs, —, 0.110, 0.3, 1.25 | tight |
| 298777 | 4000245 | 102, 1195.8µs, 0.926, 0.377, 27.3 | yes | 89, 523.1µs, —, 0.118, 2.3, 0.86 | tight |

### Fix

New knob `postcull_before_rescue` (C++ default false — byte-identical off):
when ON (and `postcull_unflagged` is on), run `cull_unflagged_lowquality`
once more BEFORE the §I/§J rescues, in both the per-run (`fit_round2`) and
shared (`fit_round2_shared`) tails. The legacy post-rescue call is untouched,
so rescue adoptions stay subject to the same quality bar as before. The cull
decision is per-bundle (no cross-bundle interaction), so the early pass
removes exactly the bundles the late pass would have.

Threading: `qlmatching.jsonnet` key-suppressed arg → `wct-clustering.jsonnet`
`ql_postcull_before_rescue` → runner env `PDVD_QL_POSTCULL_EARLY` (default 0).
Config-warn if set without `postcull_unflagged` (inert). Doctest round-trip
added (`doctest_qlmatching_config.cxx`).

### Verification

- Build + install rc=0; freshness proof: `local/lib/libWireCellMatch.so`
  mtime 2026-07-17 20:43 > last source edit. `wcdoctest-match`: 25/25 pass.
- Compiled-config proof: knob ON emits `"postcull_before_rescue": true`
  (once); OFF compiled JSON byte-identical to the pre-change config
  (`cmp` PASS, stash-baseline method).
- Knob-off A/B: tag `ac1off` (idx 0/5/15, default env, new binary) vs `nm4b`
  — see scorecard below.
- Knob-on run: tag `ac1` (18 evts, `PDVD_QL_POSTCULL_EARLY=1`) — scorecard
  below.

### Scorecard (tags `ac0`/`ac1off`/`ac1`/`ac1def`)

Off-gate: `ac1off` (new lib, knob off, idx 0/5/15) vs `ac0` (pre-change lib
rebuilt from stash, same runner state): calib `cmp` identical + every mabc
zip member-hash identical ⇒ **byte-identical off, PASS**. (First attempt
compared against the `nm4b` dirs and flagged `mabc-all-apa.zip` — that diff
is the cc3a stage-4 runner-default adoption (wcp `c255930`, post-nm4b), not
this change; matching-level outputs vs nm4b were already identical.)

Knob-on (`ac1`, 18 evts, `PDVD_QL_POSTCULL_EARLY=1`) vs `nm4b`:

| metric | nm4b | ac1 | Δ |
|---|---|---|---|
| agree (scan consistency) | 751 (84.3%) | **755 (84.5%)** | +4 |
| missed long tracks | 91 | **87** | −4 |
| phantom | 138 | 138 | 0 |
| unknown | 137 | 173 | +36 |

Pair-level regression diff: **0 moved, 0 removed, +74 added** (purely
additive). The 4 agree gains are exactly the 4 blind-spot clusters, each
adopted at its truth flash (uid163 via the relaxed tier, the others tight).
The +36 long unknowns are new low-confidence adoptions with no scan verdict
— same rescan queue category as the 15 nm4b relaxed adoptions. No new
phantom: none of the 74 landed on a scan-rejected time.

**ADOPTED as runner default** (`PDVD_QL_POSTCULL_EARLY:-1`); flip-verified:
default-env idx-0 calib `cmp`-identical to `ac1` (tag `ac1def`). Toolkit
knob stays default OFF. Revert: `PDVD_QL_POSTCULL_EARLY=0`.

---

## Phase 1b — saturation-aware rescue ratio-high extension

### Negative result first: the clean-channel ratio is NOT viable

The doc-22 recommendation "compute the rescue ratio over clean (unsaturated,
covered) channels" fails on contact with data: on the uid33 truth flash
(evt298651, 7382 PE) the 3 railed channels carry 7364 PE of the measurement
and 24k of the 25.9k predicted; the "clean" remainder is 18 PE measured vs
1891 predicted → clean ratio ≈ 106 (whole-flash 3.51). Excluding partially-
covered channels (cov < 1) is even worse (13 channels, ratio 106 with
14 PE). **On bright PDVD flashes the light lives on the railed channels** —
exclusion throws away the measurement instead of the bias.

### The viable form: railed PE is a lower bound ⇒ extend the HIGH gate

If a fraction `satfrac` of a flash's measured PE is on saturation-flagged
channels, the true PE is ≥ measured, so pred/meas is an OVERESTIMATE — the
ratio-high gate rejects honest candidates, while ratio-low and ks/chi2 stay
meaningful. 8 of the 87 missed clusters' truth candidates fail ONLY
ratio-high; 7 of those sit on flashes with satfrac ≥ 0.77.

Offline rescue-replay sim on the nm4b dumps (directional only): variants
grid over (satfrac_min, ks cap, ratio-high multiplier). Full skip of the
high gate trades 1:1 (+2 right / +2 wrong); the capped form
**satfrac > 0.5, ratio < ratio_hi × 2.0: +3 right / +0 wrong / +0 unknown**
(recovers uid33 golden crosser, uid32, uid4000006 — the wrong rivals all
have extreme ratios that the ×2 cap excludes).

### Knob

`cluster_rescue_sat_ratio_relax` (+ `cluster_rescue_sat_frac_min`=0.5,
`cluster_rescue_sat_ratio_mult`=2.0), C++ default OFF ⇒ byte-identical.
Applied in both rescue tiers' accept via a shared `sat_ratio_hi_ok` helper
(ratio-low, ks, chi2 untouched). Threaded: `qlmatching.jsonnet`
(key-suppressed, needs `cluster_rescue_shared`) → `wct-clustering.jsonnet`
`ql_cluster_rescue_sat_relax/_sat_frac_min/_sat_ratio_mult` → runner env
`PDVD_QL_CRESCUE_SATRELAX` (+`_SAT_FRAC`, `_SAT_MULT`), default 0 pending
scorecard. Doctest round-trips (29/29 pass).

### Scorecard (tags `ac2off`/`ac2`/`ac2def`)

Off-gate: `ac2off` (new lib, knob off, idx 0/5/15) vs `ac1`: calib `cmp`
identical + all mabc member-hashes identical ⇒ **byte-identical off, PASS**.

Knob-on (`ac2`, 18 evts, `PDVD_QL_CRESCUE_SATRELAX=1`, frac 0.5 / mult 2.0)
vs `ac1`:

| metric | ac1 | ac2 | Δ |
|---|---|---|---|
| agree | 755 (84.5%) | **761 (85.1%)** | +6 |
| missed long tracks | 87 | **81** | −6 |
| phantom | 138 | 138 | 0 |
| unknown | 173 | 170 | −3 |

Pair diff: 3 added + 3 moved, 0 removed. All 6 changed pairs verified at the
scan truth times (uid32, uid4000006, uid33 added; uid4000016, uid77,
uid4000033 moved). The "moves" are legitimate: all 3 were ac1 RESCUE
adoptions (strength 0.000) sitting at a wrong flash; once the sat-relax
admits the truth candidate it wins the rescue score argmin. Better than the
sim's +3/0 because the sim scored candidates against a fixed adoption set.

**ADOPTED as runner default** (`PDVD_QL_CRESCUE_SATRELAX:-1`); flip-verified
(`ac2def` idx-0 calib == `ac2`). Toolkit knobs stay OFF. Revert:
`PDVD_QL_CRESCUE_SATRELAX=0`.

Cumulative since nm4b: agree 751→761 (84.3→85.1%), missed 91→81, phantom
flat 138.

---

## Phase 2 — phantom-side overprediction culls (wtrunc + xtpc-pin)

### Sim survey on the ac2 dumps (directional)

- **Flash-twin channel is DEAD** at the current op point: 1/144 phantoms has
  a truth-positive for the same cluster within 0.5–8 µs. The cathxa-era twin
  problem no longer exists post-tune_c2_cr; twin handling NOT implemented.
- **Ratio-low cull is a dead end**: the under-predicted phantom bulk
  (non-wtrunc phantoms ratio p50 0.31) cannot be separated — every useful
  threshold kills tens of agreed matches (best safe point only −5 phantoms).
- **wtrunc overpred is the clean signature**: wtrunc phantoms ratio
  p50 1.44 / p90 11.3 vs agreed wtrunc matches 0.68 / 1.51. Gate
  `wtrunc && !pin && !sc1 && ratio>2.0` (sat-dominated flashes exempt):
  **−14 phantoms / −0 agreed / −4 unknown** in replay. Per-channel max
  overpred adds agree cost — ratio-only chosen.
- **pin overpred (ratio-only)**: phantom pins ratio p50 1.98 vs agreed 0.69.
  `pin && ratio>2.0` (sat-exempt): **−6 phantoms / −0 agreed**. A ks gate on
  pins is catastrophic (−22..−65 agreed): geometric pins legitimately carry
  bad ks — that is why they were pinned.

### Knobs (both default OFF ⇒ byte-identical)

Second and third branches in `cull_unflagged_lowquality` (now
`postcull_*`-family): `postcull_wtrunc_overpred` + `postcull_wtrunc_ratio_hi`
(2.0) + `postcull_wtrunc_sat_frac` (0.5, shared sat exemption), and
`postcull_pin_overpred` + `postcull_pin_ratio_hi` (2.0). Both apply
regardless of the high-consistent flag; wtrunc branch protects
pin/scenario-1. With 1a's early-cull default, culled clusters immediately
become rescue-eligible in the same pass. Interaction check: none of the 6
phase-1b gains is wtrunc-flagged, and the sat exemption protects railed-
flash adoptions by construction. Runner envs `PDVD_QL_POSTCULL_WTRUNC`,
`PDVD_QL_POSTCULL_PIN` (+ `_RATIO`s), defaults 0 pending scorecard.
Doctests 36/36.

### Scorecard (tags `ac3off`/`ac3off2`/`ac3`/`ac3def`)

Off-gates: `ac3off` (wtrunc-only build) and `ac3off2` (final build with both
culls), knobs off, idx 0/5/15 — calib + all mabc member-hashes identical to
`ac2` ⇒ **byte-identical off, PASS** (both builds).

Knob-on (`ac3`, 18 evts, `PDVD_QL_POSTCULL_WTRUNC=1 PDVD_QL_POSTCULL_PIN=1`)
vs `ac2`:

| metric | ac2 | ac3 | Δ |
|---|---|---|---|
| agree | 761 (85.1%) | **764 (85.4%)** | +3 |
| missed long tracks | 81 | **78** | −3 |
| phantom | 138 | **118** | −20 |
| unknown | 170 | 180 | +10 |

Phantom −20 lands exactly on the sim (−14 wtrunc, −6 pin). Pair diff:
**zero agreed pairs removed or moved**; 16 phantoms culled outright (3
removed, 13 re-adopted by the rescue at unscanned times), **3 phantoms
converted to agree** (cull frees the cluster, the 1a early-cull + rescue
then re-match it at the truth flash — the +3 agree / −3 missed), 6 unknown
cleanups.

**ADOPTED as runner defaults** (`PDVD_QL_POSTCULL_WTRUNC:-1`,
`PDVD_QL_POSTCULL_PIN:-1`); flip-verified (`ac3def` idx-0 calib == `ac3`).
Toolkit knobs stay OFF. Revert: `PDVD_QL_POSTCULL_WTRUNC=0
PDVD_QL_POSTCULL_PIN=0`.

Cumulative since nm4b: **agree 751→764 (84.3→85.4%), missed 91→78,
phantom 138→118**.

---

## Phase 3 — amplitude-model residual study (NEGATIVE: no correction knob)

Script `ql_display/amp_residual_fit.py` (outputs
`work/ql_scores/ac2_amp/amp_residual.{md,json}`); residual = log(meas/pred)
on scan-agreed matches — the GT sample the doc-22 follow-up proposed
fitting on.

- **Clean flashes (satfrac<0.2), flash level:** residual p50 within ±0.11 of
  zero in every brightness decile (one noisy 30-entry bin at +0.35). The
  photon model is **unbiased on clean flashes at every brightness** — there
  is no brightness-dependent under-prediction to fit. The doc-22
  "bright-flash under-prediction" is the saturation censoring plus tail
  variance, not a model bias curve.
- **Saturated flashes (satfrac≥0.2):** p50 ≈ −0.25 at all brightnesses —
  the DAPHNE rail reports a lower bound. This is a measurement property,
  already handled gate-side by the phase-1b/2 sat exemptions; "correcting"
  pred for it would double-count.
- **Topology (clean single-bundle):** boundary +0.34, two_boundary +0.52,
  close_to_PMT +0.66, wtrunc +0.40 — real charge-truncation biases, but
  within-class scatter stays sd ≈ 0.5 (a factor 1.6) everywhere, while the
  doc-22 wrong-flash truth candidates sit at residual +0.8..+3.0. A
  multiplicative correction shifts the bulk without separating the tail —
  and re-scaling predictions by ~e^0.3 would perturb the gate/LASSO balance
  of all 761 agreed matches for no ranking gain.

**Phase 3b (correction knob) NOT built** per the plan's contingency: no
clean functional form exists. The productive amplitude-side levers were the
gate-side saturation handling (phases 1b/2), both adopted.

<!-- phase sections appended as the campaign proceeds -->

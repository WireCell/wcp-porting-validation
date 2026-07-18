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
| 2 | phantom-side precision gates (xtpc pin, wtrunc overpred, flash twins) | pending |
| 3 | amplitude-model residual fit on 751 agreed GT pairs (+ optional knob) | pending |
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

<!-- phase sections appended as the campaign proceeds -->

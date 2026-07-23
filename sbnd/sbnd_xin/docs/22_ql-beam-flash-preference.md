# 22 — Q/L beam-window flash preference (`beam_pref`)

Investigation of two hand-reported QLMatching failures on the reco1 48-event
neutrino-candidate sample ([21_reco1-input.md](21_reco1-input.md), Bee set
`0fbcecd1-23ff-4103-8a58-cd9a23551d80`), and the resulting default-OFF
`beam_pref` knob family: prefer a flash inside the corrected BNB beam window
(0.2–2.2 µs) when it competes with a cosmic flash for a cluster.

All 48 events in this sample carry a neutrino, so the beam flash *should*
almost always win substantial charge; before this knob it frequently lost it
to light-degenerate cosmic flashes.

## Repro block

```sh
cd sbnd_xin
export SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod
export SBND_WORK_ROOT=$PWD/work-fsprod-bpw05          # fresh tree, evt<ID> symlinked to work-fsprod
./run_ql_evt.sh data 18 -beam-pref                    # evt 246579 (case 1)
./run_ql_evt.sh data 38 -beam-pref                    # evt 116962 (case 2)
SBND_MAX_JOBS=6 ./run_ql_evt.sh data all -beam-pref   # population scan
# knob-off A/B (fresh SBND_WORK_ROOT=work-fsprod-bpAB2, no -beam-pref):
# member-hash every mabc-all-apa.zip against work-fsprod-rse with abtest/hash_archive.py

# Validation round 2 (hand-scan + nueCC tuning; see §Validation round 2):
# hand-scan roots (evt<ID> symlinked from work/, both modes, per operating point):
SBND_WORK_ROOT=$PWD/work-bpval2-w050 SBND_MAX_JOBS=6 ./run_ql_evt.sh data all -calib -beam-pref
SBND_WORK_ROOT=$PWD/work-bpval2-w050 SBND_MAX_JOBS=6 ./run_ql_evt.sh mc   all -calib -beam-pref
# reco1 roots (evt<ID> symlinked from work-fsprod), BEAMPREF_WEIGHT scans the L1 weight:
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-2025fall-48evt-fsprod \
SBND_WORK_ROOT=$PWD/work-fsprod-bpv2-w050 BEAMPREF_WEIGHT=0.5 \
SBND_MAX_JOBS=6 ./run_ql_evt.sh data all -calib -beam-pref
# scoring:
./ql_beam_pref_score.py work-bpval-off work-bpval2-w050    # vs work/ql_labels/{data,mc}
./ql_beam_pref_tune.py  work-fsprod-bpv-off work-fsprod-bpv2-w050
```

Reading the numbers below: Bee flash indices/times are from the baseline
`0-op.json` (times in µs, **corrected** by `frame_apply_at_caf`); "ident" is
the cluster id shown by the Bee **img-global** layer (clicking a point);
QL-log `gidx` is the per-side group index. The QL log prints flash times in
units of 100 ns (e.g. `17.34` = 1.734 µs) and the final `flash_bundles_map`
in ns.

## Case 1 — 18255-1-246579: neutrino cluster lost to a −326.8 µs cosmic

**Symptom (user report).** Cluster 7 at (−110.0, −43.4, 59.6) — the neutrino
candidate — is not matched to the beam flash (Bee flash 16, 1.75889 µs,
464.3 PE = the APA1-side twin; the APA0-side beam flash is Bee flash 15,
1.73465 µs, **45636 PE**). It is matched to a non-beam flash instead.

**Baseline outcome.** The −326.818 µs cosmic flash (22823 PE, APA0 side)
takes BOTH big TPC0 clusters — ident 7 (gidx 5, 10728 pts, pred 24799) and
ident 8 (gidx 0, 15899 pts, pred 19519) — a combined prediction of 44317 PE,
**1.94× its measured light**, while the 45.6-kPE beam flash is left with a
43-point cluster (pred 36 PE).

**Root cause.** Per-bundle light metrics are time-degenerate and slightly
favor the cosmic T0:

| bundle | ks | chi2/ndf | pred/meas |
|---|---|---|---|
| beam (1.7347 µs) × ident 7 | 0.133 | 4.04 | 27948 / 45636 |
| beam × ident 8 | 0.267 | 32.9 | 20740 / 45636 |
| −326.8 µs × ident 7 | 0.067 | 1.48 | 24799 / 22823 |
| −326.8 µs × ident 8 | 0.069 | 1.63 | 19519 / 22823 |

Both cosmic bundles pass the high-consistent ladder (B2: ks < 0.09,
chi2/ndf < 4); the beam bundles fail it on ks. `cull_inconsistent` then
**drops both beam-flash bundles before the LASSO ever runs** ("cluster kept
high/xtpc-consistent bundle"), so the fit never sees the only assignment that
explains the beam flash's 45.6 kPE. The joint PE budget strongly favors the
beam pairing (idents 7+8 predict 48.7 kPE ≈ the beam flash's 45.6 kPE; on the
cosmic they over-predict 1.94×), but nothing in the live path tests a flash's
*combined* over-prediction: the `outbeam_pe_frac` QA that would catch 1.94×
runs inside `organize_bundles()`, whose output is discarded (known vestigial
path, see [8_ql-chain.md](8_ql-chain.md)).

**Why it hid.** At the cosmic T0 the clusters shift by ~51 cm in x but remain
contained (`require_containment` passes), and each individual bundle's
pred/meas ratio (0.86×/1.09×) looks fine; only the per-flash sum is absurd.

## Case 2 — 18259-1-116962: beam flash swapped with a 101.33 µs flash

**Symptom (user report).** The beam flash (1.6452 µs, 15287 PE, Bee flash 12)
should match clusters 13 at (−90.3, −72.0, 129.8) and 9 at (−71.5, −68.8,
165.8); instead the 101.328 µs flash (12894 PE, Bee flash 14) takes them.

**Baseline outcome.** Flash 101.328 µs ← idents 6, 9, 13 (pred 12396 ≈ its
12894 PE measured); beam flash ← ident 4 only (pred 6320 of 15287 PE).

**Root cause.** Here no cull is involved — the LASSO itself prefers the swap,
even though the per-bundle light metrics favor the beam flash for *both*
disputed clusters:

| bundle | ks | chi2/ndf | pred |
|---|---|---|---|
| beam (1.6452 µs) × ident 9 | 0.066 | 4.85 | 9021 |
| 101.33 µs × ident 9 | 0.171 | 19.9 | 9434 |
| beam × ident 13 | 0.071 | 105.5 | 2696 |
| 101.33 µs × ident 13 | 0.222 | 112.3 | 2819 |

The fit picks the swap because it zeroes the 101.33 µs flash's PE residual
(12396/12894) while the beam-centric assignment leaves it 96% unexplained.
But that flash's light is almost certainly not TPC0 charge at all: it
coincides within 105 ns with the **101.433 µs APA1-side flash (39048 PE)**
that is matched high-consistent (at_x_boundary + close_to_PMT) to the big
TPC1 cosmic — the APA0-side 12.9 kPE is the same physical cosmic seen from
the TPC0 side (its cathode-clipping sliver, ident 6, predicts only 143 PE).
The per-side light model cannot attribute cross-side light, so the LASSO
"balances" it with stolen beam clusters.

## Fix — the `beam_pref` knob family (QLMatching, default OFF)

Per the owner's suggestion (extra weight for 0.2–2.2 µs flashes via the
chi2-like penalty), three mechanisms, all gated on one master switch and all
byte-identical when off:

| knob | C++ default | runner-on value | acts where |
|---|---|---|---|
| `beam_pref` | false | true | master switch |
| `beam_pref_tlow` / `beam_pref_thigh` | 0.2 / 2.2 µs | (default) | window on **corrected** flash time |
| `beam_pref_lasso_weight` | 1.0 (inert) | 0.5 | per-column L1 weight multiplier for beam-window bundles in both LASSO rounds (`lasso_flag_factor`, composes with `lasso_boundary_weight`) — the beam bundle is shrunk less, tilting marginal competitions (fixes case 2, and case 1's reassignment) |
| `beam_pref_rescue_scale` | 1.0 (inert) | 0.2 | empty-flash rescue steal guard: a NON-beam flash re-stealing a cluster FROM a beam-window flash must beat it by `mF < mX·scale`, not just strictly |
| `beam_pref_max_ks` | 1e9 (inert) | 0.3 | bundle-quality gate (round 2): only a bundle with `ks_dis ≤ max_ks` receives the preference |
| `beam_pref_min_pred_frac` | 0.0 (inert) | 0.02 | bundle-quality gate (round 2): only a bundle predicting ≥ this fraction of the flash's measured PE receives the preference |
| (no knob) | — | — | `cull_inconsistent` exemption: a beam-window bundle survives the "rival kept a high/xtpc-consistent bundle" drop and competes in the LASSO (fixes case 1's pre-fit kill); xtpc scenario-1/joint-pin culls are NOT relaxed |

Design choice: the user's "adjust the chisquare" is implemented as the LASSO
L1 down-weight + cull exemption rather than relaxing the high-consistent
ladder for beam flashes — the `high_consistent` flag gates postculls, rescues
and the xtpc machinery, so mislabelling beam bundles there would leak far
beyond the two competition points, and the chi2/ks numbers in the calib dumps
stay honest.

Threading: `QLMatching.{h,cxx}` (toolkit) → `sbnd_xin/qlmatching.jsonnet`
shim overlay `beampref_on` (sets weight 0.5 / scale 0.2 / gate 0.3 / 0.02) →
`wct-clus-matching-perevt.jsonnet` TLAs `beam_pref` /
`beam_pref_weight` / `beam_pref_rescue` → `run_ql_evt.sh -beam-pref`
(+ `BEAMPREF_WEIGHT` / `BEAMPREF_RESCUE` env for scans). The production
config `cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet` is untouched.

The rescue guard matters: with the aggressive weight 0.2 the LASSO gave the
beam flash both big clusters of case 1, the emptied cosmic flash then
re-stole ident 7 through the empty-flash rescue on raw light metric (0.093 vs
0.406 — "better" light, wrong physics); scale 0.2 blocks exactly that.

## Verification

Knob-off gates (all with the final library, `local/lib` freshness checked):

- 48/48 reco1 events: `mabc-all-apa.zip` member-hash identical to the
  `work-fsprod-rse` campaign (`abtest/hash_archive.py`; e.g. evt 246579
  `8abb8ccc…`, evt 116962 `a0fecd71…`), work tree `work-fsprod-bpAB2`.
- Standard SBND joint-QL gate: data evt 686 under `setarch x86_64 -R`
  reproduces the recorded baseline `4b006453…` (5 members), work tree
  `work-bpAB686b`.
- Compiled config with the knob off is byte-identical to the pre-change
  jsonnet (wcsonnet diff vs git HEAD); knob-on JSON carries
  `beam_pref/beam_pref_lasso_weight/beam_pref_rescue_scale` in the
  QLMatching node. `wcdoctest-match` 36/36.
- Round-2 library (quality gate + weight/rescue TLAs), knob-off regression
  gates re-run: all 48 reco1 `mabc-all-apa.zip` member-hash identical to
  `work-fsprod-rse` (tree `work-fsprod-bpv2-off`), and all 20 hand-scan
  events identical to the round-1-library off run (`work-bpval2-off` vs
  `work-bpval-off`). Compiled off-config still byte-identical to git HEAD;
  `wcdoctest-match` 36/36 again.

Knob-on, the two reported cases (weight 0.5, work tree `work-fsprod-bpw05`):

- **246579**: beam flash (1.7347 µs) ← ident 7 (pred 27948; the user's
  cluster), cosmic −326.8 µs ← ident 8 (pred 19519, 0.86× its meas) — exactly
  the reported expectation, and the per-flash PE budgets both become sane.
- **116962**: beam flash (1.6452 µs) ← idents 4 + 9 + 13 (pred 18096 vs
  15287 meas, 1.18×); the 101.328 µs flash keeps only the cathode sliver
  ident 6 — its remaining ~12.7 kPE is cross-side light of the TPC1 cosmic,
  correctly matched on the APA1 side (101.433 µs ← big TPC1 cluster,
  unchanged).

Round-1 population scan over all 48 events (final `flash_bundles_map` diff,
knob-off vs ungated knob-on): weight 0.2 changed 46/48 events (173 moves onto
beam flashes), weight 0.5 changed 44/48 (149 moves). Both case events fixed
at either weight; this breadth motivated validation round 2 below, which
added the bundle-quality gate and cut the churn by ~5× at the same fix rate.

## Validation round 2 — hand-scan truth + the 48-evt nueCC criterion

Two independent ground-truth sets (owner's suggestion), both scored per
operating point:

1. **Hand-scan labels** — the 10 data + 10 MC events scanned with the
   `ql_scan` viewer ([12_ql-scan-display.md](12_ql-scan-display.md)); truth =
   the `selected` `(flash_gid, main_cluster)` pairs in
   `work/ql_labels/{data,mc}/.scan_state-evt*.json`, the same convention as
   the ladder tuning in [16_highconsist-ladder.md](16_highconsist-ladder.md)
   (93/100 data, 92/113 MC at ladder adoption). Scorer:
   `ql_beam_pref_score.py`. The data sample is off-beam cosmics, so any
   0.2–2.2 µs flash there is a random cosmic — this set measures *harm*.
2. **The 48-evt reco1 nueCC criterion** — every event carries a neutrino, so
   the dominant in-window flash should hold a light-consistent charge budget.
   Grade per event (scorer `ql_beam_pref_tune.py`): GOOD = matched with
   0.33 ≤ Σpred/meas ≤ 3, WEAK = matched outside that band, ORPHAN = no
   cluster.

**Round-2a finding (ungated weight scan 0.7/0.5/0.35/0.2).** The 48-evt side
liked every weight ≤ 0.5 (48/48 GOOD), but the hand scans exposed a real
failure mode: the down-weighted beam flash **sweeps up junk bundles** — tiny
predictions (8–300 PE) at ks 0.4–0.8 and chi2/ndf 250–470 become near-free
LASSO columns that mop up the beam flash's PE residual, and occasionally
steal a hand-tagged true match (MC evt 18 cluster 1 at ks 0.679; MC evt 41 a
pred-8-PE bundle on a 3431-PE flash; data evt 1258 cluster 9 at ks 0.433).
Agreement dropped 101→99 (MC) / 95→94 (data, w0.5). Meanwhile every genuine
beam match across all cases sits at **ks ≤ 0.27 and pred ≥ 17% of the flash
PE** — a clean separation.

**Round-2b fix: bundle-quality gate.** `beam_pref_max_ks` = 0.3 and
`beam_pref_min_pred_frac` = 0.02: a bundle only receives the preference
(cull exemption, L1 down-weight, rescue guard) if it could plausibly *be*
the beam match; junk bundles keep the un-preferred behavior.

Hand-scan agreement (higher = better; "extra" = matcher selections the scan
did not tag, beam-window subset in parentheses):

| config | data agree /100 | data extra (beam) | MC agree /112 | MC extra (beam) |
|---|---|---|---|---|
| knob OFF | 95 | 80 (10) | 101 | 67 (5) |
| ungated w0.5 | 94 | 81 (26) | 99 | 69 (20) |
| **gated w0.5 (adopted)** | **95** | 80 (14) | **101** | 67 (7) |
| gated w0.7 | 95 | 81 (14) | 101 | 67 (7) |
| gated w0.35 | 95 | 80 (14) | 100 | 68 (8) |

48-evt nueCC criterion + churn (moved pairs = symmetric diff of all selected
`(flash, cluster)` pairs vs knob-off):

| config | GOOD | WEAK | events changed | pairs → beam | ← beam | other churn |
|---|---|---|---|---|---|---|
| knob OFF | 47 | 1 (evt 246579) | — | — | — | — |
| ungated w0.5 | 48 | 0 | 42/48 | 155 | 0 | 188 |
| **gated w0.5 (adopted)** | **48** | **0** | **21/48** | **30** | 1 | 42 |
| gated w0.7 | 47 | 1 (evt 246579) | 15/48 | 22 | 1 | 37 |
| gated w0.35 | 48 | 0 | 22/48 | 32 | 1 | 44 |

Operating-point conclusions:

- **Weight 0.5 + gate (0.3 / 0.02) is the adopted point** (now the shim/runner
  default): zero hand-scan regression on either sample, both reported cases
  fixed, 48/48 GOOD beam budgets, and 5× less churn than ungated.
- 0.7 is too weak (case 1's ident-7 recovery needs ≤ 0.5); 0.35 loses one MC
  hand-scan match; 0.2 (round 1) over-collects (case 1's beam flash also
  grabs ident 8, orphaning the cosmic).
- Both cases at gated w0.5: evt 246579 beam ← ident 7 only (pred 27948);
  evt 116962 beam ← idents 4+9+13 (pred 18037/15287) — identical to the
  round-1 fixes, without the junk riders (`(1,3)` at ks 0.57–0.66 on case 1,
  `(2,3)` at ks 0.31–0.40 on case 2 are gone).
- The 7 knob-added hand-scan pairs that survive the gate are small
  (5–143 pred PE), light-consistent (chi2/ndf 0.5–4.1) pickups by small
  (120–391 PE) in-window flashes — plausible small activity the scan never
  tagged, not steals. 6 of 7 sit at ks ≤ 0.27; one (MC evt 31, ks 0.53)
  arrives second-order via the standard empty-flash rescue, not via the
  preference itself.
- Two MC beam-window true pairs stay missed at every config *including
  knob-off* (evt 9 pair (1,8): ks 0.607, chi2/ndf 257; evt 18 pair (2,7):
  ks 0.283, chi2/ndf 433, pred 3.4% of the flash): the light model flatly
  disagrees with the hand tag there, and the gate correctly refuses to force
  the first. Pre-existing, not a knob effect.

## Status & caveats

- **ADOPTED as the SBND production operating point** (owner decision,
  2026-07-21, on the round-2 evidence): the canonical config
  `cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet` now sets `beam_pref:
  true` with weight 0.5, rescue 0.2, gate ks ≤ 0.3 / pred ≥ 2%, and the
  window **explicitly** as `beam_pref_tlow/thigh = 0.2/2.2 µs` — the window
  is the *experiment's* beam gate on corrected flash time; other experiments
  must set their own in their config rather than inherit the C++ default
  (which merely happens to equal SBND's BNB window). This adoption is **NOT
  byte-identical** to the pre-adoption output — the delta is exactly the
  validated one: every `mabc-all-apa.zip` of the flag-free rerun
  (`work-fsprod-bpprod`, `work-bpprod`) is member-hash identical to the
  gated-w0.5 validation roots (`work-fsprod-bpv2-w050`,
  `work-bpval2-w050`), 48+20/68. C++ defaults remain OFF/inert (other
  detectors byte-identical).
- `run_ql_evt.sh -beam-pref` is now an override overlay: it only matters
  together with `BEAMPREF_WEIGHT`/`BEAMPREF_RESCUE` to scan a different
  operating point on top of production.
- Optional follow-up: a spot hand-scan of the 21 changed reco1 events
  (diff `work-fsprod-bpv2-off` vs `work-fsprod-bpv2-w050` — same round, same
  `-calib` basis).
- The knob presumes a *corrected* flash time base (`frame_apply_at_caf`,
  [21_reco1-input.md](21_reco1-input.md)); on uncorrected samples the window
  must be widened or the times fixed first.
- Case 1 also exposed two latent issues, not fixed here: (a) no live check
  of a flash's combined over-prediction (the organize()-side `outbeam_pe_frac`
  QA is vestigial); (b) cross-side light from a cathode-clipping cosmic is
  unattributable in the per-side model (case 2's 101.33 µs flash) — a
  cross-side coincidence discount would address it more physically than a
  time prior.

## Work trees retained on disk (2026-07-23 cleanup)

The campaign's one-off A/B trees and the superseded weight scans were deleted
(~2.0 GB); their results are the tables above. Trees named in this doc but no
longer on disk: `work-fsprod-bpAB`/`-bpAB2`, `work-fsprod-bpon`,
`work-fsprod-bpw05`, `work-fsprod-bptw`, `work-bpAB686`, `work-abfs1`, and the
round-1 scan roots `work-bpval-{off,w020,w035,w050,w070}` /
`work-fsprod-bpv-{off,w020,w035,w050,w070}` plus the non-adopted round-2 points
`{work-bpval2,work-fsprod-bpv2}-{w035,w070}`. Rebuild any of them with the
repro block above.

| tree | role |
|---|---|
| `work/`, `work-fsprod/` | imaging/label **bases** — every tree below symlinks `evt<ID>` into these; do not delete |
| `work-bpprod/`, `work-fsprod-bpprod/` | **production** flag-free rerun (adopted op point) |
| `work-bpval2-w050/`, `work-fsprod-bpv2-w050/` | gated-w0.5 validation roots, member-hash identical to production (68/68) |
| `work-bpval2-off/`, `work-fsprod-bpv2-off/` | round-2 knob-off regression + churn-diff baseline |
| `work-fsprod-rse/` | member-hash baseline of record for the 48 reco1 events |
| `work-bpAB686b/` | evt-686 joint-QL reproducibility gate (`4b006453…`) |

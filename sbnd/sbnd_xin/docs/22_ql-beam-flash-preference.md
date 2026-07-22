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
| (no knob) | — | — | `cull_inconsistent` exemption: a beam-window bundle survives the "rival kept a high/xtpc-consistent bundle" drop and competes in the LASSO (fixes case 1's pre-fit kill); xtpc scenario-1/joint-pin culls are NOT relaxed |

Design choice: the user's "adjust the chisquare" is implemented as the LASSO
L1 down-weight + cull exemption rather than relaxing the high-consistent
ladder for beam flashes — the `high_consistent` flag gates postculls, rescues
and the xtpc machinery, so mislabelling beam bundles there would leak far
beyond the two competition points, and the chi2/ks numbers in the calib dumps
stay honest.

Threading: `QLMatching.{h,cxx}` (toolkit) → `sbnd_xin/qlmatching.jsonnet`
shim overlay `beampref_on` (sets weight 0.5 / scale 0.2) →
`wct-clus-matching-perevt.jsonnet` TLA `beam_pref` → `run_ql_evt.sh
-beam-pref`. The production config `cfg/pgrapher/experiment/sbnd/
qlmatching.jsonnet` is untouched.

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

Knob-on, the two reported cases (weight 0.5, work tree `work-fsprod-bpw05`):

- **246579**: beam flash (1.7347 µs) ← ident 7 (pred 27948; the user's
  cluster), cosmic −326.8 µs ← ident 8 (pred 19519, 0.86× its meas) — exactly
  the reported expectation, and the per-flash PE budgets both become sane.
- **116962**: beam flash (1.6452 µs) ← idents 4 + 9 + 13 (pred 18096 vs
  15287 meas, 1.18×); the 101.328 µs flash keeps only the cathode sliver
  ident 6 — its remaining ~12.7 kPE is cross-side light of the TPC1 cosmic,
  correctly matched on the APA1 side (101.433 µs ← big TPC1 cluster,
  unchanged).

Population scan over all 48 events (final `flash_bundles_map` diff, knob-off
vs knob-on):

| lasso weight | events changed | cluster moves → beam | ← beam | other churn |
|---|---|---|---|---|
| 0.2 | 46/48 | 173 | 2 | 47 |
| 0.5 (adopted for the runner) | 44/48 | 149 | 2 | 42 |

Both case events are fixed at either weight; 0.5 is preferred because case 1
resolves inside the LASSO (the cosmic keeps its own cluster instead of going
charge-orphan) and the collateral churn is smaller.

## Status & caveats

- Default OFF everywhere; enabled only per-run via `-beam-pref`. **Not**
  adopted in the SBND production config: the direction of the moves is right
  (essentially one-way traffic onto beam flashes, ~3 clusters/event) but only
  the two reported cases have ground truth. Before adopting as default, a
  hand-scan of a changed-event sample (44 of 48 events change; diff the
  `flash_bundles_map` blocks of `work-fsprod-bpAB2` vs `work-fsprod-bpw05`)
  should confirm the moved clusters really are in-time — the aggressive tail risk
  is dragging a genuine cosmic to the beam T0 (~±25 cm x-shift for the
  ±150 µs flashes, worse for ms-scale ones).
- The knob presumes a *corrected* flash time base (`frame_apply_at_caf`,
  [21_reco1-input.md](21_reco1-input.md)); on uncorrected samples the window
  must be widened or the times fixed first.
- Case 1 also exposed two latent issues, not fixed here: (a) no live check
  of a flash's combined over-prediction (the organize()-side `outbeam_pe_frac`
  QA is vestigial); (b) cross-side light from a cathode-clipping cosmic is
  unattributable in the per-side model (case 2's 101.33 µs flash) — a
  cross-side coincidence discount would address it more physically than a
  time prior.

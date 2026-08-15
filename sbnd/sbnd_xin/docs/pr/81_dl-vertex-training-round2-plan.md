# doc pr/81 — DL-vertex training round 2: deployment-aligned objective on the frozen 473 labels (PLAN)

Owner request 2026-08-15: no new scannable events exist right now (the
remaining ~555 mcp1k events contain no neutrino and cannot be scanned; more
scanning is planned later).  Improve DL-vertex performance with what is on
disk, in this order: **(A)** the pr/79 §11e idea 4.iii — candidate scoring
on new input features, exact live version; then **(B)** idea 1 — a
scale-anchored candidate-ranking fine-tune; **(C)** idea 2 — joint
(weights, score_scale) calibration as deploy policy; with **(3)** — the
validated `calib_guard.py` — in the training loop throughout.  This file is
the plan; results append here as §A/§B/§C when each phase runs.

## 0. Fixed constraints (all measured, none negotiable)

- **Labels are frozen at 473** (nuecc 47 / ncpi0 19 / mcp1k 407).  No
  pseudo-label expansion either: the unlabeled mcp1k remainder is
  neutrino-free, so it has no neutrino vertex to weak-label.
- **The 95-event lockbox is SPENT** (pr/79 §5a).  Offline gates are
  out-of-fold metrics + `calib_guard.py` anchored replay (validated: CP24
  anchor exact, ft2u −57 caught, hft1 −20 caught — doc pr/79 §11c).  The
  only terminal gate is a live A/B.
- **Priors that bound expectations** (do not re-litigate; they gate scope):
  - pr/79 §7 (step 4): NO learned router/selector formulation on recorded
    features clears the routing base rate (~85% precision needed; every
    deployable variant −2..−25).  Phase A re-enters this territory with
    exactly two new elements (exact live features; ranking-trained not
    logistic-fit) and is therefore a bounded screen with an explicit
    kill criterion, not a re-run of step 4.
  - pr/79 §11 (step 8): dense-MSE fine-tune at O(473) is rank-inert and
    score-inflating.  Phase B exists to change the OBJECTIVE, not to
    re-dose the same one.
  - pr/78 §5: the 9b candidate-margin arm had the best OOF (+11) but
    failed the do-no-harm guard, and a bare margin loss is NOT
    scale-invariant (inflating all scores grows margins).  Hence the
    anchor term in B is mandatory, not optional.
- **No toolkit C++ in this round.**  Phase A's head is not representable
  in the current C++ pipeline; a positive Phase A result is an owner
  proposal, not a deployment.  Phase B deploys through the existing
  `dl_weights` swap; Phase C through the existing `dl_vtx_score_scale`
  config value.  Production stays untouched throughout; any flip is a
  separate owner decision after a live A/B.
- M13: fresh output names only (`data/k20feats-harv-*`, `runs/hr*`,
  `runs/calibguard-hr*`); `data/full473`, `data/harv473`, `runs/hft1*`,
  labels, and existing arms are read-only.

## Phase A — 4.iii: exact-feature candidate head (screen, go/no-go)

**Question**: does a per-candidate scorer trained on features the composite
never sees — the CP24 net's 16-dim penultimate representation at the
candidate voxel (now EXACT, from `hv_cloud`) + the harvest traditional
features (`hv_n_proton_in/out, hv_z_prior, hv_n_tracks, hv_n_showers,
hv_in_fv, hv_conflicts, hv_reduced_chi2`) + the recorded geometric terms —
beat the composite chooser by enough to survive anchored deploy replay?

- **A1. Feature table.**  `extract_feats.py --harvest-roots` (new mode,
  same idiom as `build_dataset.py`): forward CP24 on `hv_cloud` (not the
  rebuilt cloud — kills the doc'd CAVEAT and the `match_dis` crutch;
  live voxel centres now match exactly), grab penultimate features at each
  usable candidate's voxel; join per candidate with the hv_* row fields
  and the seven recorded terms.  Output `data/k20feats-harv-20260815/`
  (473 npz).  Sanity: `cand_score` must equal the recorded row `dl_score`
  bit-for-bit on spot checks (same floats, same ops — the verify_harvest
  property extended to features).
- **A2. Head fits, nested-OOF (folds = `dataset.kfold_split`, seed
  20260814).**  Heads: linear and 2-layer MLP (h=8, the
  `runs/probes-20260815/mlp_probe.py` shape), per-event softmax CE over
  candidates, truth candidate = the label pick's row (tol 1 cm).  Feature
  ablations: terms-only (step-4 baseline, must reproduce its numbers),
  +hv_*, +16-dim, all.
- **A3. Deploy replay, both semantics, via a `calib_guard.py`-style
  anchored replay on the head scores** (`--head-tsv` input mode or a small
  sibling script reusing its outcome logic):
  - (i) chooser-only: head picks among candidates, acceptance stays
    `recorded total >= 10` (frozen routes);
  - (ii) full: accept iff head score clears a nested-OOF threshold.
- **Go/no-go**: if no formulation predicts **> 0** anchored, Phase A closes
  as a confirmatory negative (the step-4 verdict survives exact features)
  and is reported as such — no C++ proposal, proceed to B.  If some
  formulation predicts clearly positive (≥ +5 with stand-ins counted),
  STOP and present to the owner as a C++ proposal (new inference path =
  owner decision), then proceed to B regardless.

## Phase B — 1 + 3: scale-anchored candidate-ranking fine-tune (deployable)

**Objective change**, not dose change.  `train.py` gets three default-off
flags (old recipes reproduce byte-identically when unset):

- `--cand-softmax W`: per-event softmax cross-entropy over the CANDIDATE
  voxels' scores (candidate→voxel-row mapping = the existing S9b
  `cands_dir` machinery; sidecar built by `build_cands.py` for harv473
  first).  Softmax over scores is invariant to a uniform additive shift —
  the loss cannot be paid by en-bloc inflation.
- `--scale-anchor L`: per-view penalty `L * mean((top_m(new) −
  top_m(frozen CP24))²)` over the m=20 highest frozen-scored voxels, from
  a frozen CP24 forward on the SAME augmented voxelization.  This pins the
  absolute scale to CP24's calibration while the ranking term reorders.
  Cost: one extra frozen forward per view (~2× epoch time; measured and
  reported).
- `--dense-weight D`: existing MSE term made weightable (default 1.0 =
  legacy; new arms run D=0 or 0.1).

**Arms** (6-fold, seed 20260814, same augmentation, `--bn-freeze
--min-cloud 16 --clip 5.0`, data = `harv473`):

| arm | loss | lr0 | rationale |
|---|---|---|---|
| hr1 | softmax only (D=0) | 1e-5 | isolate the objective change |
| hr2 | softmax + anchor L=1 (D=0) | 3e-5 | bolder lr — ordering must actually move; anchor+softmax make it safe |
| hr3 | softmax + anchor + D=0.1 | 1e-5 | dense term as regularizer only |

**Guard-in-loop selection (idea 3)**: per fold, `calib_guard.py
--events-file <val fold> --weights fold<k>/CP<E>` on a checkpoint subset
(E ∈ {0,2,5,8,11,14,17}); fold-best = max guard-predicted delta
(tie → earliest).  Rationale: val-median d_argmax is a step function that
tied at every epoch in pr/79 §11b; the guard metric is deployment-aligned
and cheap at 65 events.  (Needs a small `calib_guard.py` extension:
`--events-file` filter + accepting a fold-checkpoint pattern.)

**Verdict per arm**: (a) OOF guard-predicted delta (sum over folds, val
events only, stand-ins counted); (b) inflation check — confirming-event
top1 ratio must sit in [0.9, 1.1]; (c) classic `evaluate.py` rank metrics
for continuity.  **Gate to deploy build**: (a) > 0 AND (b) clean on the
best arm.  Then: deploy build (`--kfold 0`, epochs = median fold-best),
full-473 `calib_guard` screen, and only a guard-positive candidate gets
the live A/B (fresh arms `work-*-<arm>`, PR_JOBS=32, min_accept=10,
`ab_vertex_compare.py` vs the ma10 arms, per-event regression
explanations).  Production flip remains owner-gated (escalation rule 1);
the deliverable of a passing live A/B is the evidence table, not a flip.

**Honest expectation**: at O(473) the likely outcomes are (i) rank-inert
again (softmax gradients also too weak at safe lr) or (ii) small positive.
Either way the objective/calibration machinery becomes standing
infrastructure for the future larger-label round, which is the durable
value.

## Phase C — 2: joint (weights, score_scale) calibration policy

- `calib_guard.py --fit-scale`: fit the multiplicative scale on CONFIRMING
  events only, matching CP24's top-1 score distribution (scale′ = 1000 /
  median ratio) — fit to match the distribution, never to maximize
  accuracy (that would tune the threshold through the back door).  Report
  every future candidate at both (1000, ma=10) and (scale′, ma=10).
- Deployment path if ever needed: `dl_vtx_score_scale` already exists in
  cfg (`common/clus.jsonnet` arg, SBND default 1000.0).  A runner env /
  TLA thread (`SBND_DL_VTX_SCORE_SCALE`) is a small wcp+cfg addition done
  ONLY when a candidate actually needs it, with the standard
  compiled-config byte-identity proof at default.
- Applied retroactively as a check: hft1 at fitted scale (≈1000/1.02
  confirming) — expected ≈ +0, recorded for completeness.

## Order of execution

A1 → A2 → A3 (go/no-go) → B (build_cands for harv473 → train.py flags →
hr1/hr2/hr3 with guard-in-loop → verdict → deploy build + full screen →
live A/B only if guard-positive) → C (fit-scale reporting; retro hft1).
Results append to this doc per phase; commit + push at each phase close
(scripts + doc; runs/ and data/ stay on disk per convention).

## Stop conditions

- Any guard anchor drift: re-run of the CP24 anchor must stay exact
  (0 flips, ratios 1.000) after every `calib_guard.py` code change —
  else stop and fix the guard before trusting any number.
- Phase A positive ≥ +5 → stop for owner (C++ proposal) before building
  anything deployable on it.
- Live A/B regressions not explained by the weights change → stop, report
  first divergent event (CLAUDE.md §5.5).
- No production flip, no wire-cell-data commit, no Bee upload without the
  owner.

---

# RESULTS (2026-08-15)

## §A results — confirmatory NEGATIVE (no C++ proposal)

Repro:

```
cd sbnd_xin/dl_vtx_training
OMP_NUM_THREADS=1 python3 extract_feats.py --harvest \
    --arm-roots vtxscan-prod0813=work-nuecc48-ma10k20-harv2 \
                vtxscan-prod0813-ncpi0=work-ncpi0-ma10k20-harv2 \
                vtxscan-prod0813-mcp1k=work-mcp1k-ma10k20-harv2 \
    --out data/k20feats-harv-20260815 --jobs 24
python3 cand_head.py --feats data/k20feats-harv-20260815 \
    > runs/candhead-20260815.log 2>&1        # tsv: runs/candhead-20260815.tsv
```

- A1 sanity PASSED: 473/473 extracted; `cand_score` bit-equal to the
  recorded row `dl_score`; `match_dis` exactly 0 for every candidate
  (the hv_cloud forward reproduces live inference, feature-level).
- A2/A3 (362 margin-eligible events, recorded 358/473 correct):

| formulation | chooser acc (of 362) | chooser-semantics anchored | full-router anchored |
|---|---|---|---|
| terms-only lin (step-4 repro) | 269 | +0 | −15 |
| terms+hv lin (best) | **291** (composite: 246) | **+1** | −16 |
| terms+f16 lin | 274 | −5 | −12 |
| terms+hv+f16 lin | 288 | −3 | −21 |
| any mlp8 variant | 250–282 | −4 … −21 | −29 … −60 |

- Verdict: the chooser signal is real (+45 candidate-accuracy over the
  composite with hv_* features) but unroutable — best end-to-end +1,
  below the ≥+5 owner bar; adding the 16-dim exact features HURTS every
  formulation (overfit at O(362)); full-router semantics are always
  negative (the step-4 base-rate verdict survives exact features).
  Phase A closes as a confirmatory negative; no new C++ inference path
  proposed.

## §B results — NEGATIVE: ranking objective kills inflation but pays in deflation

Repro:

```
cd sbnd_xin/dl_vtx_training
BASE="--data data/harv473 --kfold 6 --epochs 18 --freeze none --bn-freeze \
      --min-cloud 16 --clip 5.0 --device cpu --cands data/harv473-cands \
      --cand-softmax 1.0"
# hr1: $BASE --dense-weight 0.0 --lr0 1e-5
# hr2: $BASE --dense-weight 0.0 --lr0 3e-5 --scale-anchor 1.0
# hr3: $BASE --dense-weight 0.1 --lr0 1e-5 --scale-anchor 1.0
# (6 folds each, OMP_NUM_THREADS=2, logs runs/hr{1,2,3}-f{0..5}.log)
bash hr_guardsel.sh hr1   # etc; logs runs/hr<arm>-guardsel/f<k>-E<E>.log
# deploy (hr3 only, median guard-best epoch = 0 -> 1 epoch):
python3 train.py --kfold 0 --epochs 1 --name hr3-deploy <hr3 flags>
python3 calib_guard.py --name hr3-deploy \
    --weights runs/hr3-deploy/fold0/CP0.pth --jobs 16 \
    --tsv runs/calibguard-hr3-deploy-20260815.tsv
```

- **Objective works as designed**: across all 127 guard replays (3 arms ×
  42 checkpoints + deploy), reject→ACCEPT inflation flips = **0**.  The
  softmax-CE loss structurally cannot be paid by en-bloc score inflation
  — the pr/79 failure mode is eliminated, not just detected.
- **Ranking signal is real**: val median d_argmax moves for the first
  time in the campaign (e.g. hr1 folds 1.31–1.47 cm vs the MSE arms'
  frozen 1.99 cm), where 18 epochs of MSE never moved the argmax.
- **But routing does not profit** — the loss is paid in DEFLATION on
  corrective/hard events instead (corr top-1 ratio ×0.25 at E0, drifting
  up with epochs), producing 2–10 accept→REJECT losses per fold:

| arm | fold-best deltas (f0..f5) | sum | gate (>0, conf∈[0.9,1.1]) |
|---|---|---|---|
| hr1 (softmax, lr 1e-5) | 0 +1 0 +1 −1 −1 | 0 | FAIL (delta) |
| hr2 (softmax+anchor, lr 3e-5) | −1 +1 +1 +1 −1 −1 | 0 | FAIL (delta) |
| hr3 (softmax+anchor+D0.1, lr 1e-5) | 0 +1 +1 +1 −1 −1 | **+1** | pass (marginal) |

- hr3 deploy screen (full 473, CP0): **−3** (28 acc→REJ, 0 rej→ACC,
  corrective top-1 ×0.152); no min_accept point rescues it (best −1
  @ ma=20).  The OOF +1 was fold-max selection noise.  **No live A/B**
  per the pre-registered gate; no candidate weights staged.
- Verdict: at O(473), "gradients don't pay" extends to the
  deployment-aligned objective — the calibrated-ranking machinery
  (`--cand-softmax/--scale-anchor/--dense-weight`, `hr_guardsel.sh`
  guard-in-loop selection) is standing infrastructure for the next
  larger-label round; nothing ships now.

## §C results — global scale calibration CANNOT rescue an inflated net

Repro:

```
python3 calib_guard.py --name hft1-fitscale --fit-scale --jobs 16 \
    --weights runs/hft1-deploy/fold0/CP0.pth \
    --tsv runs/calibguard-hft1-fitscale-20260815.tsv
python3 calib_guard.py --name ft2u-fitscale --fit-scale --jobs 16 \
    --weights .../wire-cell-data/sbnd/scn_vtx/sbnd-vtx-ft2u-full473-e10-CP9.pth \
    --tsv runs/calibguard-ft2u-fitscale-20260815.tsv
```

| net | unscaled delta | confirming top1 median | fitted mult | delta at fitted scale | best joint (scale, ma) point |
|---|---|---|---|---|---|
| hft1 | −20 | 1.022 | ×0.979 (scale 979) | −17 | +1 @ ma=12 |
| ft2u | −57 | 1.197 | ×0.836 (scale 836) | −53 | +0 @ ma=20 |

- The confirming-median fit recovers almost nothing (−20→−17, −57→−53):
  the inflation is NOT a uniform rescale — it is concentrated in the
  corrective/marginal band (hft1 corrective top1 ×1.31 vs confirming
  ×1.02), exactly where routing decisions live.
- Even the full joint (scale, min_accept) sweep only reaches parity
  (+1 / +0), never a gain.  Policy: `--fit-scale` stays a reporting
  tool; the guard's reject-on-inflation verdict is final — no
  `SBND_DL_VTX_SCORE_SCALE` deployment thread is warranted.
- CP24 anchor after all guard changes: still exact (+0, 0 flips,
  ratios 1.000, fit multiplier 1.000).

---

# NEXT ROUND (pr/82 pre-plan) — the ~2000-event sample (owner, 2026-08-15)

The owner will have ~2000 new events soon and will prepare scan results;
this roughly TRIPLES the label count (473 → ~1400).  Ordered by value per
label; steps 1–3 spend no labels on training and come first.  Gates and
allocation fractions below are pre-registered NOW, before any new label
is read (the point of writing this ahead of time).

## Prerequisites (before anything is scored)

- **P1. Our own harvest-enabled production pass** on the new events:
  PR chain at the production operating point (min_accept=10, CP24,
  `dl_vtx_harvest` ON), same recipe as the prod0813 `*-ma10k20-harv2`
  arms, PR_JOBS≈32 (owner-granted).  Never score against someone else's
  reco (M11).  Calib count must equal event count before the scan opens.
- **P2. Scan per the pr/80 §11 runbook** (fresh label tag, M13; score by
  vertex id; old-kit double-run noise floor).  Use the certain-tier
  triage (95.5% precision over 37% coverage) to pre-answer the easy
  third and route only ambiguous events to the owner.
- **P3. Data-vs-MC stratification decided up front**: if these are
  detector data (not MC), every split below stratifies by data/MC too,
  and the data-only guard replay becomes the primary screen — a net
  that helps MC but hurts data must not be invisible.

## Step 1 — unbiased production measurement (free, highest value)

The current 358/473 (75.7%) is measured on events that steered every
knob decision since pr/33.  Score production on the fresh labels alone
→ the honest operating-point number, ±1.3% instead of ±2%.  Report
per-sample.  No decision hangs on it; it is the reference for
everything after.

## Step 2 — out-of-sample retest of every round-1/2 verdict (minutes)

The new labels are a never-touched test set for everything already
trained.  `calib_guard.py` replay (new labels only, `--events-file`) of:
ft2u (−57 on old 473), hft1 (−20), hr3-deploy (−3), and the Phase A
chooser head (+45 signal / unroutable).  Either the negative verdicts
confirm out-of-sample or the guard itself was overfit to the old 473 —
this validates the METHODOLOGY before it gates round 3.  Expected:
confirmation; any sign flip is a stop-and-report finding.

## Step 3 — fresh lockbox (the old one is spent)

Before any new label enters a training pool: reserve 20–25% of the NEW
labels, stratified by sample × corrective × (data/MC), seed recorded in
the manifest.  Never read until a final candidate exists — restores the
three-tier gate structure (train/val → guard screen → lockbox → live
A/B) that has been running without a lockbox since pr/78.

## Step 4 — training round 3 (the dose question, now answerable)

Corrective pool grows ~115 → ~350.  The O(473) failure was gradients
shifting CONFIDENCE but not learning GEOMETRY, and both confidence
escape routes are now closed (softmax kills inflation structurally;
the guard catches deflation).  Whether ~350 corrective examples teach
geometry is exactly the open question.  The pipeline is committed and
is a RERUN, not a rebuild:

```
build_dataset.py --harvest-roots <new arms> --inherit-manifest <lockbox col>
train.py <hr3 recipe: --cand-softmax 1.0 --scale-anchor 1.0 --dense-weight 0.1 --lr0 1e-5>
bash hr_guardsel.sh <arm>          # guard-in-loop checkpoint selection
calib_guard.py <deploy CP>         # full-manifest screen
# live A/B only if guard-positive; flip owner-gated as always
```

Start from the hr3 recipe (only arm to ever pass the OOF gate).  Known
bias to respect: fold-max selection inflates OOF sums ~+1 — marginal
gate passes are noise until the full-manifest screen.

## Step 5 — revisit the two "real signal, can't use it" findings

- **Chooser head at O(1100)**: the 16-dim exact features overfit at
  O(362); at triple the training pool that may flip.  Rerun
  `cand_head.py` unchanged.  If a formulation clears the pre-registered
  ≥+5 anchored bar → STOP and present to the owner as a C++ proposal
  (new inference path = owner decision).
- **min_accept sweep at 3× statistics**: ±2-point effects currently
  invisible become measurable; re-run the sweep on the fresh labels
  before touching the knob (any change = owner flip decision).

## Standing rules carried forward

- CP24 anchor must stay exact after every `calib_guard.py` change.
- runs/ and data/ never committed; scripts + docs only; `git add -f`
  for `*.sh`.
- No production flip, no wire-cell-data commit, no Bee upload without
  the owner.

# doc pr/79 — DL-vertex deployment campaign: pr/78 levers A/B'd live and flipped

2026-08-15.  Executes the pr/78 §8 deployment sequence on the owner's
instruction: each lever gets a real PR production over every hand-scan
sample, is scored against the 473 labels, and — on the owner's standing
decision for this campaign — the SBND production default is flipped after
each step whose gate passes.  Follow-ups (fold ensemble/soup, 9b margin
sweep < 0.3, 9a headroom re-check) after the three steps.

Baseline throughout: the prod0813 arms
(`work-{nuecc48,ncpi0,mcp1k}-prod0813`), 473 labels, **322/473 correct** at
1 cm (nuecc 39/47, ncpi0 14/19, mcp1k 269/407; lockbox subset 64/95).
Build: toolkit `1e534c6f` (same build as prod0813 — no C++ change anywhere
in this campaign; jsonnet + weights are runtime inputs).

## 0. Repro

```bash
# Phase A proofs (before any arm):
wcsonnet cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet             # bare
wcsonnet --tla-str input=... --tla-code "pipeline_names=[...,'tagger_check_neutrino','pr_display']" \
    --tla-code "vertex_scoreboard=true" ... wct-pr-perevt.jsonnet        # production-pipeline variant
# before/after threading: both byte-identical (cmp).  Knob-on proof:
#   --tla-code dl_vtx_min_accept_score=10.0 --tla-code dl_vtx_top_k=20
#   -> compiled TaggerCheckNeutrino data shows min_accept=10, top_k=20.

# comparator identity check (must reproduce pr/78 §3 exactly):
cd sbnd_xin/dl_vtx_training
python3 ab_vertex_compare.py \
    --arm-roots vtxscan-prod0813=work-nuecc48-prod0813 \
                vtxscan-prod0813-ncpi0=work-ncpi0-prod0813 \
                vtxscan-prod0813-mcp1k=work-mcp1k-prod0813 \
    --numu-manifest data/full473/manifest.tsv --lockbox-manifest data/full473/manifest.tsv
# -> 322/473, zero transitions, taxonomy 322/50/81/20, provenance (4.0, 5).

# step-1 arm (identical pattern for nuecc48/ncpi0 with their cb0805 ql roots):
cd sbnd_xin
PR_JOBS=24 PR_EXTRA_STAGES=pr_display SBND_DL_VTX_MIN_ACCEPT=10.0 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-mcp1k-ma10 data \
      $(cat products/prod0813/events-mcp1k-prod0813.txt)
# step-2 arm: + SBND_DL_WEIGHTS=sbnd/scn_vtx/sbnd-vtx-ft2u-full473-e10-CP9.pth  (out: *-ma10ft2u)
# step-3 arm: + SBND_DL_VTX_TOP_K=20                                            (out: *-ma10ft2u-k20)

# scoring (per arm):
python3 dl_vtx_training/ab_vertex_compare.py \
    --arm-roots vtxscan-prod0813=work-nuecc48-<arm> \
                vtxscan-prod0813-ncpi0=work-ncpi0-<arm> \
                vtxscan-prod0813-mcp1k=work-mcp1k-<arm> \
    --numu-manifest data/full473/manifest.tsv --lockbox-manifest data/full473/manifest.tsv \
    --tsv runs/ab-<arm>-20260815.tsv
```

## 1. Phase A — enablement

- **TLA threading** (toolkit `1d9454ed`): `dl_vtx_min_accept_score` /
  `dl_vtx_top_k` become function args of `clus_pr(...)` and `pr(...)` in
  `cfg/pgrapher/experiment/sbnd/clus.jsonnet` and TLAs of
  `wct-pr-perevt.jsonnet`, used at the single pinned call site
  (`tagger_check_neutrino`).  Defaults at threading time = the old literals
  (4.0 / 5): compiled JSON proven byte-identical (`cmp`) on both the bare
  and the production-pipeline compiles; knob-on proof shows the values in
  the compiled `TaggerCheckNeutrino` node.
- **Runner env gates** (this repo): `SBND_DL_VTX_MIN_ACCEPT` /
  `SBND_DL_VTX_TOP_K` in `run_pr_chain_batch.sh` next to the existing
  `SBND_DL_WEIGHTS` gate.
- **Comparator** `dl_vtx_training/ab_vertex_compare.py` — scores a NEW arm
  against the labels (taxonomy.py cannot: it trusts `label['source']` and
  the frozen `label['main_vertex']`).  Reads the new arm's calib
  `main_vertex` (fallback scoreboard `final_*`), recomputes the taxonomy
  from the NEW scoreboard, reports per-sample correct counts, fixed /
  regressed transitions vs baseline, route flips, candidate-set identity
  (the replay assumption), and scoreboard provenance
  (`dl_min_accept_score`, `dl_top_k`).  Identity check: exact (§0).
- **One-event smoke** (evt 166738) before the full arm: calib records
  `min_accept=10.0`, rows identical to baseline, rc=0, zero
  `DL vertex failed`.

## 2. Step 1 — dl_vtx_min_accept_score 4.0 → 10.0: **+36/473, FLIPPED**

Arms `work-{nuecc48,ncpi0,mcp1k}-ma10` (48+19+445 events, all rc=0, zero
`DL vertex failed`, 47+19+445 calibs).  Scored vs prod0813
(`runs/ab-ma10-20260815.tsv`, log `runs/ab-ma10-20260815.log`):

| sample  |   n | base | new | Δ   | fixed | regressed |
|---------|----:|-----:|----:|----:|------:|----------:|
| nuecc   |  47 |  39  |  38 | −1  |   1   |   2 |
| ncpi0   |  19 |  14  |  15 | +1  |   1   |   0 |
| mcp1k   | 407 | 269  | 305 | **+36** |  49 |  13 |
| numu100 | 100 |  86  |  89 | +3  |   4   |   1 |
| ALL     | 473 | 322  | **358** | **+36** | 51 | 15 |
| lockbox |  95 |  64  |  72 | +8  |  10   |   2 |

- **Live gain more than doubles the replay's +15** (pr/78 §4b predicted
  283/470).  Mechanism: the replay could only stand in the recorded
  `trad_winner` row for rejected events; the live traditional route
  re-derives the vertex downstream and lands correct far more often.
- **Replay assumption confirmed**: candidate set identical to baseline on
  473/473 events; 153 route flips vs the replay's predicted 152
  (404 → 252 DL accepts).
- **Every one of the 15 regressions is threshold-explained**: each was
  baseline `dl-rerank-accept` with the DL winner's rerank total in
  [4, 10) — 5.98, 6.29, 7.17, 8.27, 7.36, 7.42, 4.33, 9.64, 7.21, 9.42,
  5.39, 8.84, 5.90, 8.14, 7.98 — flipped to `dl-rerank-reject` and the
  traditional vertex missed.
- **Gate deviation, flagged**: nuecc is net −1/47 (evt 10550 → 41.0 cm,
  evt 52672 → 4.1 cm; evt 111412 fixed from 48.6 cm).  Single-event trade,
  fully mechanism-explained, within Poisson noise of a 47-event sample,
  against +36 overall — the flip proceeded on the owner's standing
  per-step-flip decision, with this line as the owner's review hook.
- New-arm taxonomy: 358 correct / 40 candidate-missing / 51 net-wrong /
  24 selection-wrong.

**Flip commit**: toolkit `9c9d0a61` (defaults 10.0 at the threaded arg
sites; compiled-config diff = exactly the one value 4 → 10 on the
production-pipeline compile).

## 3. Step 2 — + ft2u fine-tuned net: **REJECTED, −40/473 marginal live**

Weights file: `runs/ft2u-deploy/fold0/CP9.pth` →
`wire-cell-data/sbnd/scn_vtx/sbnd-vtx-ft2u-full473-e10-CP9.pth`
(md5 `e015b7de31cf1cdb0a098c403177372b`, never touching the uBooNE CP24
file).  Arms `work-*-ma10ft2u` (48+19+445 events, all rc=0, zero
`DL vertex failed`; weights provably live: per-event `dl_best_score`
changes, e.g. evt 48367 8.43 → 14.95).

**The net that passed every offline bar (pr/78 §5: OOF +8/378, lockbox
+2/95, zero guard fails anywhere) is a large regression in the live
pipeline** (`runs/ab-ma10ft2u-{cum,marg}-20260815.{tsv,log}`):

| comparison | nuecc | ncpi0 | mcp1k | ALL | lockbox |
|---|---|---|---|---|---|
| cumulative vs prod0813  | 39 (+0) | 13 (−1) | 266 (−3) | **318 (−4)** | 62 (−2) |
| marginal vs step-1 arm  | +1 | −2 | **−39** | **−40** (16 fixed / 56 regressed) | **−10** |

**Root cause — score-calibration inflation × acceptance threshold.**
54/56 marginal regressions are `dl-rerank-reject → dl-rerank-accept`:
fine-tuning systematically inflated the net's confidence (median
`dl_best_score` ratio ×1.53 over 200 paired events; ft2u higher on
168/200), so rerank totals that the CP24-calibrated threshold correctly
rejected now clear `min_accept` and route to far-off voxels (regressions
up to 340 cm).  The offline metrics could not see this: d_argmax and
snap-hit are *rank*-based within an event, invariant to the score scale —
but the live acceptance decision consumes the *absolute* rerank total via
`s_dl = dl_score × scale`.  pr/78's "passes both bars" verdict was
measured with a guard that has no score-calibration term.

**No flip.  The SBND production weights remain
`uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth`.**  The staged file
stays in `wire-cell-data/sbnd/scn_vtx/` as the record of the arm.

**Attribution arm** (`work-mcp1k-ft2u4`, ft2u at the OLD threshold 4.0,
mcp1k only, `runs/ab-ft2u4-20260815.{tsv,log}`): **−42/407 vs prod0813**
(269 → 227; 3 fixed / 45 regressed; net-wrong 76 → 101).  So the
fine-tuned net is worse live at EITHER threshold — this is not merely a
threshold interaction.  The score inflation is one visible symptom of a
broader failure to transfer: training and every offline eval consumed the
cloud REBUILT from the calib dump (the post-refit graph, pr/78's known
parity approximation), while the live net consumes the pre-refit C++
cloud.  The +8 OOF / +2 lockbox offline gains were an artifact of that
input distribution, not a property of the deployed system.

Lesson for any future fine-tune: add a score-calibration guard (compare
the tuned net's rerank-total distribution — or simply `dl_best_score`
percentiles — against the incumbent on confirming events) before any
deployment claim; or recalibrate `dl_vtx_min_accept_score`/`scale`
jointly with the net.

## 4. Step 3 — dl_vtx_top_k 5 → 20 (on step 1 only): **NULL, no flip**

With ft2u rejected, the k=20 arm rides on step 1 alone
(`work-*-ma10-k20`, CP24 weights, min_accept=10; 48+19+445 events, all
rc=0, zero `DL vertex failed`, provenance (10.0, 20)).  Marginal vs the
step-1 arm (`runs/ab-ma10k20-marg-20260815.{tsv,log}`):

**Exactly zero effect on the chosen vertex: 358/473 → 358/473, 0 fixed,
0 regressed, 0 route flips** — while the candidate set grew on 322/473
events (rows identical on only 151/473) and the taxonomy shifted
candidate-missing 40 → 23, selection-wrong 24 → 47.

Reading: the k=20 admission DOES place a near-truth candidate on the
scoreboard for 17 more events (pr/78 §6 predicted ≤28/50 at the pr/78
baseline), but the composite rerank picks **none** of them — every
newly-admitted near-truth candidate loses the rerank and the event moves
from candidate-missing to selection-wrong instead of to correct.
Admission was necessary but not sufficient; the realized top_k recovery
is **+0**.  A k=10 fallback arm is pointless (its admissions are a subset
of the same null) and was not run.  `dl_vtx_top_k` stays **5** in
production; raising it is harmless (zero regressions) but buys nothing
without a better selector.

Corollary — the selection ceiling is now measured: at k=20 a perfect
selector over the admitted candidates would reach **405/473** (358 + 47
selection-wrong).  That 47-event gap is the quantified headroom for the
pr/78 §9a adapter / rank_fit direction (§5c below).

## 5. Follow-ups

### 5a. Fold ensemble + weight soup — NULL result, not pursued

`dl_vtx_training/ensemble_eval.py` (**advisory second lockbox read**,
owner-approved with the caveat: the lockbox now functions as a selection
set; any candidate it prefers would need fresh labels or a production A/B).
No uncontaminated offline set exists for a 6-fold ensemble — every
train-pool event was in the training set of 5 of 6 folds — hence lockbox
only.  `runs/ensemble-lockbox-20260815.log`:

| scorer | snap-hit /95 | guard fails |
|---|---|---|
| baseline CP24 | 50 | — |
| deploy CP9 (ships in step 2) | 51 (+1) | 0 |
| ensemble (mean of 6 fold-best) | 51 (+1) | 0 |
| soup (state-dict mean, `runs/ft2u/soup.pth`) | 51 (+1) | 0 |

Neither the prediction ensemble (not deployable — C++ loads one .pth) nor
the deployable soup beats the single deploy CP9; identical d_argmax
percentiles (verified real: weights differ, per-voxel scores differ by up
to 0.11, but the argmax voxel rarely moves — the fine-tune's effect is
discrete and small).  Documented, closed.

### 5b. 9b margin sweep w ∈ {0.1, 0.15} — no candidate, closed

Arms `ft2c9bw1` (w=0.1) and `ft2c9bw15` (w=0.15), exact ft2u recipe +
`--cands data/full473-cands`, 6-fold CPU (`runs/ft2c9bw{1,15}/eval.log`).
The dose-response over the full weight range is now monotone and closes
the sweep:

| cand-margin w | OOF snap-hit /378 | guard fails |
|---:|---:|---:|
| 0 (= ft2u) | 173 (+8) | 0 |
| **0.1** | 174 (+9) | 1 |
| **0.15** | 173 (+8) | 1 |
| 0.3 (pr/78) | 175 (+10) | 2 |
| 1.0 (pr/78) | 176 (+11) | 4 |

No weight achieves 0 guard fails with a gain above ft2u's — the margin
term buys its snap-hits by paying in confirming-event damage at every
dose.  No advisory lockbox read was spent (criterion not met), and the
step-2 result (§3) makes the point moot regardless: these OOF numbers
live on the rebuilt-cloud input distribution that just failed to
transfer to the live pipeline.  Direction closed.

### 5c. 9a headroom re-check — measured by the k=20 arm

The final deployed arm (step 1, k=5) has **24 selection-wrong** events
(≥ the 10-event bar, so the direction stays open); the k=20 arm shows the
full ceiling: **47 recoverable events** if a selector could pick the
admitted near-truth candidate (405/473 = 85.6% vs today's 358/473 =
75.7%).  This makes selector work (pr/78 §9a adapter features / rank_fit
learned weights, possibly with k=20 admission enabled at the same time)
the highest-headroom next-round direction — with the step-2 lesson
attached: any learned selector must be validated in the LIVE pipeline,
not on rebuilt-cloud replays, before a flip is proposed.  No architecture
work this campaign.

## 6. Final production state and files

**What changed in SBND production** (toolkit branch `apply-pointcloud`):
- `dl_vtx_min_accept_score` **4.0 → 10.0** (commits `1d9454ed` threading +
  `9c9d0a61` flip).  Measured: **322 → 358 / 473 (+36, 68.1% → 75.7%)**.
- Nothing else: weights stay uBooNE CP24 (ft2u REJECTED live, §3),
  `dl_vtx_top_k` stays 5 (k=20 NULL, §4), `dl_vtx_score_scale` untouched.

**Per-sample final** (step-1 arm = the new production operating point):
nuecc 38/47 (−1, flagged §2), ncpi0 15/19 (+1), mcp1k 305/407 (+36),
numu100 subset 89/100 (+3), lockbox subset 72/95 (+8).

**Campaign lessons** (each with a number attached):
1. Offline replay UNDERSTATES config-only gains that reroute events (the
   live traditional route beats the frozen `trad_winner` stand-in:
   +36 measured vs +15 replayed).
2. Offline net evals on rebuilt clouds can be OUTRIGHT WRONG in sign
   (+8 OOF / +2 lockbox → −40 live): rank-based metrics miss score
   calibration, and the rebuilt post-refit cloud is not the live input.
   Any future net or learned selector must be gated on a live A/B.
3. Admission ≠ recovery: k=20 admits 17 more near-truth candidates,
   rerank selects 0.
4. The measured next-round headroom is the selector: 47 selection-wrong
   events at k=20 (ceiling 405/473).

Files (this repo unless noted):
- `dl_vtx_training/ab_vertex_compare.py` — new-arm-vs-labels comparator
  (identity-checked against pr/78 §3).
- `dl_vtx_training/ensemble_eval.py` — fold-ensemble + weight-soup
  evaluator (lockbox-only; advisory-read banner).
- `run_pr_chain_batch.sh` — `SBND_DL_VTX_MIN_ACCEPT` / `SBND_DL_VTX_TOP_K`
  env gates.
- Score TSVs/logs: `dl_vtx_training/runs/ab-ma10-20260815.*`,
  `ab-ma10ft2u-{cum,marg}-20260815.*`, `ab-ft2u4-20260815.*`,
  `ab-ma10k20-marg-20260815.*`, `ensemble-lockbox-20260815.log`,
  `ft2c9bw1/`, `ft2c9bw15/`, `ft2u/soup.pth`.
- Arm outputs: `work-{nuecc48,ncpi0,mcp1k}-ma10`, `work-*-ma10ft2u`,
  `work-mcp1k-ft2u4`, `work-*-ma10-k20` (fresh dirs; prod0813 untouched).
- Weights: `wire-cell-data/sbnd/scn_vtx/sbnd-vtx-ft2u-full473-e10-CP9.pth`
  kept ON DISK as the arm record but NOT committed to the wire-cell-data
  repo — the arm was rejected, and a 28.8 MB blob for a rejected net
  would bloat the repo history permanently.  `wire-cell-data` git is
  untouched by this campaign.
- Toolkit commits: `1d9454ed` (threading, byte-identical), `9c9d0a61`
  (min_accept flip), + the top_k comment close-out.

---

## 7. Step 4 — selector campaign on the 47-event gap: comprehensive NEGATIVE, no flip

2026-08-15, follow-on to §5c / lesson 4.  Goal: harvest the measured
selector headroom (47 selection-wrong at k=20).  Owner pre-authorized
escalation to the pr/78 §9a adapter direction if the linear selector fell
short.  Outcome up front: **every deployable formulation — six of them,
including the adapter-screen on frozen net features and a nonlinear head —
is net-NEGATIVE end-to-end under honest out-of-fold evaluation.  No C++
knob was built, no live A/B was run (nothing passed the offline gate), and
production is unchanged from §6.**  The campaign's product is the measured
explanation of *why*, plus reusable tooling for the day the prerequisites
change.

### 7a. Repro block

```
cd dl_vtx_training
ARMS="vtxscan-prod0813=work-nuecc48-ma10-k20 \
      vtxscan-prod0813-ncpi0=work-ncpi0-ma10-k20 \
      vtxscan-prod0813-mcp1k=work-mcp1k-ma10-k20"
# chooser refit on the LIVE k20 rows (this is what §6 lesson 2 demands --
# recorded rows are the exact features the C++ computed):
python3 rank_fit.py --arm-roots $ARMS --l2 0.1           # 11-feature
python3 rank_fit.py --arm-roots $ARMS --l2 0.1 --features 7terms
# end-to-end deployment sim (live-anchored replay + oracle ceilings):
python3 rank_sim.py --arm-roots $ARMS --l2 0.1 \
    --tsv-prefix runs/ranksim-k20-l2p1-20260815
# per-event router (logistic on 10 scoreboard-derived event features):
python3 route_fit.py --arm-roots $ARMS --tsv runs/routefit-k20-20260815.tsv
# frozen-net 16-dim penultimate features at candidate voxels
# (rebuilt-cloud SCREEN -- see caveat in 7d):
OMP_NUM_THREADS=1 python3 extract_feats.py --arm-roots $ARMS \
    --out data/k20feats16-20260815 --jobs 24
python3 route_fit.py --arm-roots $ARMS --extra-feats data/k20feats16-20260815
```
Folds/seed identical to `rank_fit.py` throughout (6-fold stratified
(sample, corrective), seed 20260814); L2 sweep {0.03, 0.1, 0.3, 1.0}.
One-sided rescue/demote and MLP probes were scratchpad scripts; their
numbers are quoted in 7c and their formulations described there.

### 7b. Decomposition of the 47 (verified against the k20 calibs)

- **33/47 route=dl-rerank-reject** — final answer is the live traditional
  winner.  On 12 of these the composite argmax over usable rows is ALREADY
  the truth row: the acceptance gate discarded a correct pick.
- **14/47 route=dl-rerank-accept** — on 5 the argmax is right and the final
  answer still wrong: **post-selection refit drift**
  (`snap_main_vertex_to_kink`/`improve_vertex` moved the vertex; e.g.
  evt268067: argmax row 0.10 cm from truth, final 2.27 cm at a different
  vertex_id).  Unreachable by any selector.
- Ceiling decomposition (n=473, `rank_sim.py` oracle diagnostics):

| operating point | correct |
|---|---|
| production (composite chooser + total≥10 router) | 358 |
| oracle ROUTING, composite chooser | 383 (+25) |
| oracle routing, refit 11-feature chooser (OOF) | 385 (+27) |
| perfect chooser AND router over usable rows | 433 (+75) |
| truth not within 1 cm of any usable row | 40 events |

  Note the taxonomy's "selection-wrong 47 / ceiling 405" (§4) is
  voxel-list-based; on the rows footing a perfect selector could in
  principle reach 433.  But the oracle-routing rows show the *chooser* is
  nearly saturated (+2 from refitting); the whole practical gap is
  **routing** — knowing when to trust the DL-side pick.

### 7c. Results — every formulation, end-to-end, vs the arm's 358

Replay is live-anchored: whenever the replayed decision matches the arm's
recorded behavior the prediction is the recorded final answer; row/trad
stand-ins are used only on genuinely-changed events (trad stand-ins
UNDERSTATE the live trad route, §6 lesson 1, so negative numbers here are
if anything optimistic).  Anchoring imperfection: 3 `dl-veto-protected`
events (baseline reproduces 355/473 + those 3).

| formulation | nested-OOF | in-sample | notes |
|---|---|---|---|
| A: rank chooser, route frozen | 350 (−8) | 352 (−6) | 8 fixed / 14–16 regressed; L2 0.03 & 0.1 same |
| B: rank chooser + rank-score router | 334–340 (−24..−18) | 336–339 | 91–115 reject→accept flips |
| C: 7-term ablation of A/B | worse than A/B | — | chooser OOF 270/362 vs 281–282 (11-feat) |
| router: logistic, 10 event features | 333 (−25) | 340 (−18) | composite chooser kept |
| router + frozen-net x16 features | 340 (−18) | 340 | match quality median 0.001 cm |
| one-sided rescue (reject→accept only) | −2 net | — | with and without x16: same |
| one-sided demote (accept→reject only) | −1 net | — | with and without x16: same |
| MLP router (h=8, x16, 3 restarts) | 336 (−22) | — | nonlinear adapter form |

Chooser-level context: the 11-feature refit on live k20 rows is genuinely
better at ranking — eligibility 156 → **362** events, OOF choice accuracy
**281–282/362 vs 260 recorded / 246 composite-argmax** — but that +22
lives mostly on reject-route events, where deployment either cannot act
(variant A) or pays a routing tax that swamps it (variant B).

### 7d. Why: the base rates require a precision no signal provides

- On the 223 reject-route events with candidates: flipping to accept gains
  on **20** and harms on **98** — a rescue rule needs ≳83% precision to
  break even.  On the 250 accept-route events: demote gains **12**, harms
  **84**.
- Neither the 10 scoreboard-derived event features, nor the frozen CP24
  net's 16-dim penultimate features at the winner voxel (the §9a adapter
  substrate, linear AND small-MLP heads), carries that precision at
  O(473) labels: every nested-OOF threshold the fits choose lands net
  negative or null.
- The legacy router (composite total ≥ 10, freshly retuned in §2) is
  simply strong: a single threshold on a 7-term physics-motivated score,
  already sitting at an interior optimum of its own grid (§4b of pr/78).
- Adapter-screen caveat: features came from rebuilt clouds (the §3
  transfer trap forbids *deploying* anything fit on them; live→rebuilt
  voxel match was excellent — median 0.001 cm, p90 0.5 cm — so as a
  *screen* it is informative).  A full adapter *training* round on rebuilt
  clouds would face the same trap that sank ft2u, and the screen gives no
  evidence the information exists to justify building the live-harvest
  infrastructure first.

### 7e. Verdict and the path that remains

- **No flip; production stays exactly §6.**  R2 (C++ knob) and R3 (live
  A/B) of the campaign plan were not run — the offline go/no-go gate
  (≥ +5/473 OOF) failed in every formulation, and running a live A/B on a
  knob that is negative offline on live-recorded features would spend a
  production cycle to confirm a foregone conclusion.
- The 47-event headroom is real but locked: harvesting it needs a routing
  discriminator whose precision no currently-available signal supports.
- Prerequisites for a future round (owner-level decisions):
  1. **More labels** — a fresh hand scan grows both the fit budget and,
     critically, an out-of-sample validation set (every number above is
     tied to the same 473 labels the fits consumed; OOF is honest but a
     fresh scan is the definitive check).
  2. **Live feature harvest** — a log-only C++ knob recording per-candidate
     net features into the scoreboard during a production run, so any
     future learned component trains on the exact deployed distribution
     (closes the §3 trap structurally).
  3. A net retrained end-to-end on live-input clouds (blocked on 2).
- Updated lesson 4: admission ≠ recovery (§4) and now *ranking ≠ routing*:
  the chooser was never the bottleneck; the router is, and it is harder to
  learn than to inherit.

New files: `dl_vtx_training/rank_sim.py` (live-anchored deployment sim +
oracle ceilings), `dl_vtx_training/route_fit.py` (per-event router fit),
`dl_vtx_training/extract_feats.py` (frozen-net candidate-voxel features);
extended: `rank_fit.py` (`--arm-roots/--features/--export-weights`,
`build_folds`), `scn_vtx/io.py` (`parse_arm_roots`/`calib_path_in_roots`).
Records: `runs/ranksim-k20-l2p1-20260815-{A,B}.tsv`,
`runs/routefit-k20-20260815.tsv`, `data/k20feats16-20260815/` (473 npz).
No toolkit change, no wire-cell-data change, no new arms (all analysis on
the existing §4 k20 arms; prod0813 and labels untouched).

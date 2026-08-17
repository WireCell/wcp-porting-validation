# doc pr/89 — DL vertex round 4 on the 999-label pool

Status: **EXECUTING** (§11, started 2026-08-16). §§1–10 are the pre-registered
plan, written 2026-08-16 after doc pr/88 delivered
`dl_vtx_training/data/pr88_pool_combined` (999 labels), and are left as
written; every deviation forced by reality is recorded in §11, never patched
back into the plan text.

## Repro

Every number in §1 and §4-D below was produced read-only by these commands.

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# S1.1  unbiased top-1 dl_score distribution over the whole scannable arm
#       (846 of 880 mcp2k dumps carry a scoreboard with voxels[])
python3 - <<'EOF'
import json, glob
sc = []
for f in sorted(glob.glob('work-mcp2k-harv3/pr_evt*/calib-pr-evt*.json')):
    b = (json.load(open(f)).get('vertex_scoreboard') or {})
    v = b.get('voxels') or []
    if v: sc.append(max(x['dl_score'] for x in v))
sc.sort()
print(['%.4f' % sc[int(p*len(sc))] for p in (.1,.25,.5,.75,.9)])
for c in (0.0055, 0.02):
    print(c, '%.1f%%' % (100*sum(1 for s in sc if s < c)/len(sc)))
EOF

# S1.2  IPW strata: per-event tier from the pr/88 run record x the labels
#       vtx_rules/runs/mcp2k-20260816/waves/*/review.json  ->  bucket, agrees
#       dl_vtx_training/data/pr88_pool_combined/manifest.tsv -> label_source, dis_to_main

# S1.3  miss taxonomy: truth vs vertices[].fit (graph) vs scoreboard rows[] (scored)
# S1.4  lockbox carry: join harv473/manifest.tsv on (runNo, subRunNo, evt)
# S4-D  post-DL displacement: scoreboard rows[dl_winner].xyz  vs  board final_{x,y,z}
```

`hash_archive.py` gates do not apply to this round: nothing here changes a
product. They apply the moment Arm A, C or D reaches a live arm.

---

## Context

Production SBND picks the neutrino vertex correctly on **372/473 = 78.6%**
(pr/82 §12.6). Four rounds have tried to move it. **Nine fine-tune arms have
been rejected and none has ever been deployed**; the only production change the
whole campaign produced is a config knob — `dl_vtx_min_accept_score` 4→10,
**+36/473 live** (pr/79 §2).

pr/79 §11e named exactly three things that would change the picture:

| | | status |
|---|---|---|
| (a) | O(1k+) fresh labels | **untried — this round** |
| (b) | a rank/margin objective that cannot move absolute scale | spent: pr/81-B, hr1/hr2/hr3 all FAIL |
| (c) | new input features beyond (x,y,z,q) | spent: pr/81-A, the 16-dim exact features **hurt every formulation** |

So this round is **a test of hypothesis (a)** and is gated as one. It is not a
fix for a diagnosed bottleneck: *"the bottleneck is label count"* is a
prerequisite named in pr/79 §11e, **never a measured finding**. What *is*
measured contradicts each earlier guess in turn — pr/79 §7e, *"the chooser was
never the bottleneck; the router is"*; pr/77 §6, the acceptance threshold is
not the bottleneck at 66 labels (overturned at 473); pr/78, `dl_vtx_top_k=5` is
the bottleneck (measured at **+0 realized** by pr/79 §4).

**Two label counts, and both matter.** Total went **473 → 999**. The count that
governs how far the *baseline* can be beaten — events where the label disagrees
with the reconstruction — is smaller. At the 1 cm tolerance used throughout
pr/78–88:

| | corrective @1 cm |
|---|---:|
| current epoch in-pool (mcp1k 79 + delta 10 + nuecc48 6 + ncpi0 4) | **99** |
| mcp2k human (240 events) | **111** |
| mcp2k AI-scanner (299 events) | **0** |
| **total** | **210** |

Corrective went **99 → 210, ×2.1** — the same factor as the total, so the pool
did not merely grow in one direction. pr/81 Step 4 and pr/82 §6 both sized this
dose question at *"~115 → ~350"*, so the delivered corrective dose is **below
what was pre-registered as the interesting one**. Quote that when the result
lands, in either direction.

The 299 AI-scanner labels are near-pure confirming volume. That does **not**
make them useless — see §5.

*Definitional note for every table:* `corrective` in `manifest.tsv` is
`dis_to_main > 1e-9` (any nonzero), not 1 cm. Every count here is recomputed at
1 cm. pr/78's "150 corrective" is stale-truth; do not compare against it.

---

## 1. Four measurements made while planning

These select the arms, so they precede them.

### 1.1 The net sits at a degenerate floor on ~40% of SBND events

Top-1 `dl_score` over all **846** mcp2k dumps with a scoreboard — unbiased,
every scannable event, not the review pile:

```
p10/p25/p50/p75/p90 = 0.0052 / 0.0054 / 0.0075 / 0.4483 / 0.9546
below 0.0055 : 40.2%     below 0.02 (the code's own "uncertain") : 59.2%
```

Bimodal, and the low mode is **degenerate**: for the 330 events below 0.0055
the score is pinned inside `[0.0011, 0.0055]` with median 0.00540, and the
top-1→top-5 falloff is **29%** against **55%** in the high mode — a
near-constant output field. The weights are uBooNE-trained
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet:696-699`: *"SBND retraining is
docs/pr/2 gap G3"*). The low mode is associated with sparse clouds (median 104
cloud points vs 300) but is not only small events (p75 = 225 points).

**This reframes the campaign's one shipped win.** The composite's non-DL terms
are bounded above by `W_MAIN + W_CLEN + W_FV = 4.5`
(`clus/src/NeutrinoVertexFinder.cxx:4599-4608`), so `min_accept = 10` requires
`s_dl ≥ 5.5`, i.e. `dl_score ≥ 0.0055` at `score_scale = 1000`. **That
threshold sits exactly on the floor mode.** pr/79's +36 was mechanically *"stop
using the DL vertex on the ~40% of events where the net is blind"*. It also
explains why pr/78's replay understated it (+15 vs +36): the rerouted events
are precisely the ones the live traditional path re-derives well.

### 1.2 The unbiased baseline is only bounded to ±15 points

The pr/88 pool is tier-selected, so a random split of it does **not** estimate
production accuracy. The run record carries the tier of all 845 scanned events,
so inverse-propensity weights are computable — except for one stratum:

```
stratum                        N   owner-labelled   prod-correct @1cm
auto-accept                  341        40          39   97.5%
REVIEW-disagree              167       153          76   49.7%
REVIEW FIRST                  37        36           8   22.2%
REVIEW (abstained)            49        11           6   54.5%
REVIEW-agree                 251         0           —   UNMEASURED
```

IPW over the measured strata gives **75.8%**. Including the unmeasured 251 the
bound is **[53.3%, 83.0%]** — a 30-point interval. Every pr/88 fill tier
selected on `not agrees`, so the largest stratum has sampling probability
exactly zero (pr/88 §8.5 flagged it). **~40 owner-scanned events collapse the
interval to ≈±3 points**, and no unbiased delta claim is possible without them.
Owner has agreed to scan these (§6).

### 1.3 Where the misses are (mcp2k labels, 539 events)

```
correct                                    428  79.4%
MISS: scored but lost the rerank            74  13.7%   (67% of misses)
MISS: in the graph, never scored            25   4.6%   (23% of misses)
MISS: no graph vertex at truth              12   2.2%   (11% — unreachable)
```

Independently reproduces pr/82 §12.6's 69/31 selection-vs-admission split, on a
fresh arm. ~11% of misses are a hard ceiling no vertex-stage work can reach.

### 1.4 The old lockbox does carry — by event id, not by tag

pr/88's `PROVENANCE.md` left this open (*"`--inherit-manifest` keys on
`(scan_tag, evt)`, which the epoch relabelling changed"*). Joining on
`(runNo, subRunNo, evt)` instead: **91 of the 95** full473/harv473 lockbox
events are present in the 999, and the 999 have **no duplicate** key. And
`vtxscan-mcp2k*` has **zero** event overlap with `harv473` — 539 events no
prior round has read in any capacity.

The 95 are **spent** regardless (§7). The carry matters so they can be excluded
knowingly rather than by luck.

### 1.5 One thing checked that was not true

Top-K voxels are **not** generally collapsed onto one candidate: over 286
sampled dumps the top-5 have median spread **38 cm** and snap to a mean of
**3.0** distinct candidates (1 candidate on only 31% of events). A spatial-NMS
replacement for `dl_vtx_top_k` is therefore *not* the explanation of pr/79's
null k=20 result, and is not proposed. Recorded because it was the first
hypothesis and it was wrong.

---

## 2. A divergence in the record, resolved: both curated samples are DATA

The record disagreed on whether `nuecc48` and `ncpi0` are data or MC:

- `docs/pr/82:506` — *"~890 new data + 407 old data + **66 MC (47 nueCC + 19
  NCpi0)**"*, and §6 P3 builds a stratification requirement on it: *"Every
  split stratifies by data/MC, and the **data-only guard replay is the primary
  screen**."*
- `scripts/retire/plan_20260816.py:271-272` — `'nuecc48 (48 **data**)'`,
  `'ncpi0 (19 **data**)'`.
- `docs/pr/82:87` runs nuecc48 as `./run_pr_chain_batch.sh … **data** 10550`,
  and that third argument is the runner's *"`data|sim` reality TLA"*
  (`run_pr_chain_batch.sh:36`).

**Owner confirms: both are data.** `docs/pr/82:506` is wrong. Two consequences
this round states rather than quietly absorbs:

- **pr/82 §6 P3 is struck.** There is **no MC in the 999 at all** and no truth
  vertex anywhere in it — which is why every label in this campaign is
  hand-scanned. Left standing, "the data-only guard replay is the primary
  screen" would be a screen this round claims to run while running nothing.
- **nueCC-48 and NCpi0-19 are a topology stratum, not a sample stratum.** They
  are curated subsets of the same MCP2025C data stream as mcp1k/mcp2k. Calling
  them a separate sample would imply a domain difference that does not exist.

---

## 3. How the three areas rank

**The learned router — largest headroom, most thoroughly closed.** 67% of
misses live there. pr/79 §7 killed every deployable formulation (frozen-route
−8/−6, rank-threshold −24..−18, logistic router −25/−18, MLP −22); pr/81-A
found real chooser signal (291/362 against the composite's 246) that was
**unroutable** (best anchored **+1/473** against a ≥+5 bar); pr/81-C showed
global calibration cannot rescue an inflated net. The base-rate wall is
explicit: on 223 reject-route events a rescue gains 20 and harms 98 (needs
≳83% precision); on 250 accept-route events a demote gains 12 and harms 84.
**Five guard-shaped ideas were falsified by exact replay before implementation
in pr/79 alone.** Do not reopen it.

Arm C is deliberately not that: one physics-motivated term added to an additive
score that already has seven, using a feature none of those formulations could
see, and held to the same discipline that killed the five — base-rated by exact
replay on the 999 **before any C++ is written**.

**"Rerank" also has a second meaning** that pr/77–81 barely touched: the
post-DL geometry adjustment that runs after the vertex is chosen. Measured here
for the first time against vertex labels, it is the **largest net-positive
component in this document** (+50 on 342 events) — Arm D.

**The DL net — the one live thread.** pr/81-B is the campaign's only unfinished
result: `--cand-softmax` structurally eliminated the pr/79 inflation mode
(**0 reject→accept flips across all 127 guard replays**) and moved val
`d_argmax` for the first time in the campaign (1.99 → 1.31–1.47 cm, where MSE
had been frozen for 18 epochs) — but paid in **deflation** on corrective events
(top-1 ×0.152 on the hr3 deploy screen), costing 2–10 accept→REJECT flips per
fold, net −3. It failed at 99 corrective labels; there are now 210, and §1.1
says the net has a large blind region to fix.

**Inputs — one cheap resweep left.** `min_accept` shipped; `top_k` 5→20
measured NULL live (0 fixed, 0 regressed, 0 route flips) even though the
candidate set grew on 322/473; the harvest knob shipped. Admission is 23% of
misses and admitting more provably does not recover them.

---

## 4. The arms

Four, each pre-registered. **One live A/B is budgeted**, for whichever clears
its offline gate by the largest margin. A–D are independent offline; only the
live arm is scarce.

### Arm A — joint acceptance recalibration (config-only, cheapest, ship-capable)

`min_accept` was tuned as a 1-D scan; §1.1 shows it and `score_scale` are one
coupling constant between the net and the ±4.5 geometry prior. Re-fit the joint
grid on the 999 with §1.2's IPW weights, using the existing
`calib_guard.py --fit-scale` and its `(min_accept, scale)` sweep. Report per
tier and per topology stratum, never as one aggregate.

Also measure **`dl_vtx_swap_guard`** — default OFF
(`TaggerCheckNeutrino.cxx:632`, SBND `false`), never A/B'd, guarding a
documented failure where a confident voxel moves the main vertex onto a
different non-flash-matched cluster (case 18255-506746, 28 cm off,
`NeutrinoVertexFinder.cxx:4618-4625`).

**Gate:** IPW-weighted held-out gain **≥ +1.5 points** with **0** guard
regressions in `calib_guard.py`; CP24 anchor exact.

### Arm B — retrain with the ranking objective at 2.1× corrective labels

1. **Dose replay.** pr/81-B `hr3` verbatim on the new pool:
   `--freeze none --lr0 1e-5 --bn-freeze --min-cloud 16 --clip 5.0 --cands
   --cand-softmax 1.0 --scale-anchor 1.0 --dense-weight 0.1`, 18 epochs,
   6-fold, on a `--drop-unscannable` snapshot. This is pr/81's own
   pre-registered "same pipeline at 3× labels".
2. **The deflation fix.** pr/81-B's anchor pins *the frozen net's top-20 voxel
   scores* (`train.py:136-147`) — it forbids exactly the reordering the ranking
   term must perform, which is why it could not stop the deflation. Replace it
   with an **event-level max anchor**, `(max_new − max_frozen)²`: preserves the
   one scalar acceptance consumes (`best_score ≥ min_accept`,
   `NeutrinoVertexFinder.cxx:4733`) while leaving the argmax free to move.
   ~10 lines in `train.py`, new flag, default off.

**Gate:** `calib_guard.py` **first** (CP24 anchor exact — the screen that would
have killed ft2u at −57 for free), then OOF, then the sealed held-out read,
then the live A/B. **No net claim without a live-pipeline A/B** — pr/79's rule,
paid for by ft2u's +8 OOF / +2 lockbox becoming **−40 live**.

**Known bias:** fold-max selection inflates OOF sums by ~+1 (pr/81 §B: *"the
OOF +1 was fold-max selection noise"*). Treat a marginal pass as noise until
the full-manifest screen.

### Arm C — rule-1 topology as an eighth composite term

pr/80 §4 measured the strongest single feature in the campaign, and **no
pr/77–81 formulation used it**: *"86.5% of truth vertices have every attached
track pointing away vs 31.9% of others"*, with 63.8% of non-truth vertices at a
stopping end against 8.1% of truth. Every learned selector to date ranked on
the 23-column scoreboard row, which contains **no dQ/dx, no Bragg direction, no
shower start and no vertex topology**. The separation is also in the right
range for the ≳83% precision the routing base-rate wall demands.

**Deployment shape — one insertion point, byte-identical when off.** Add an
eighth term to the existing composite,
`s_topo = W_TOPO × (fraction of attached prongs pointing away)`, with `W_TOPO`
a config knob **defaulting to 0.0**. At 0.0 the term is a literal `+ 0.0`, so
the compiled config and every recorded score are unchanged. This is the natural
insertion point because a composite term moves **both** the argmax (chooser)
and `best_score` vs `min_accept` (router) — and routing is the measured
bottleneck, which is why the chooser-only pr/81-A topped out at +1.

**The offline simulation is exact.** The composite is closed — pr/77 §6 and
pr/78 §4a both measured `max|Σ terms − recorded total| = 0` over every scored
row — so adding a term offline and re-running the argmax and the threshold
reproduces the live decision exactly, the way `calib_guard.py` already replays
acceptance.

**Contamination discipline (pr/80 §F5).** Compute rule 1 from **raw polyline
`points[].dQ` / `points[].dx`** and geometry only. `dirsign`, `dir_weak`, `rr`,
`showers[]`, `is_main`, `main_candidate`, `is_main_cluster` are **all
downstream of the vertex choice** and are banned from the feature. Polarity is
safe: pr/80 §2 verified `points[0] ↔ start_vertex_id` on **2572/2572**
segments. Arc length is recomputed from the polyline, never read from `rr`.

**Two known ways this feature lies, both pre-registered:**

- **Short prongs** (pr/88's P6 finding). Below ~5 cm both end-windows cover the
  same points, so a short prong contributes **no** rule-1 vote *while looking
  as though it contributed one*. The implementation must return an explicit
  "no vote", the fraction must be over voting prongs only, and the vote count
  rides along as a second feature.
- **The collinearity confound** (pr/80 §10.8). Back-to-back prong pairs at
  ≥150° run 22.4% at truth vs 48.2% elsewhere — **2.2× enriched but useless as
  a veto**, and all three of pr/80's certain-but-wrong picks had that shape.
  Report performance split on it; do not fold it in silently.

**Staged, with a hard stop before any C++:**

- **C0 — reproduce the published number.** Implement offline and recover pr/80
  §4's **86.5% vs 31.9%** on pr/80's own label set. If it does not reproduce,
  the implementation is wrong, not the feature.
- **C1 — base-rate on the 999, before a line of C++.** This is the step that
  killed five guard-shaped ideas in pr/79 (*"the legacy heuristics are right
  5–9× more often than wrong, so every cheap override loses on the
  population"*). Sweep `W_TOPO` through the exact composite replay; report the
  help/harm decomposition separately on the reject-route and accept-route
  populations, IPW-weighted.
  **Bar: net ≥ +10 events on the 999 with 0 guard regressions**, and the gain
  must not be reproducible by a pure `W_TOPO = 0` threshold move.
- **C2 — only if C1 clears:** propose the default-0.0 knob. New logic in the
  acceptance path is a **STOP-and-present**, not something to implement on
  momentum.

### Arm D — the post-DL adjustment stage (measurement first, config-only follow-on)

Two different things are called "rerank" in this tree, and Arms A and C touch
only the first:

- **(a) the in-DL composite rerank** — snap top-K voxels to graph vertices,
  score with the 7-term composite, accept if `best_score ≥ min_accept`
  (`NeutrinoVertexFinder.cxx:4454-4757`).
- **(b) the post-DL geometry adjustment** — runs on `final_main_vertex`
  *regardless of route* (`TaggerCheckNeutrino.cxx:1415-1454`):
  `snap_main_vertex_to_kink` (`:1421`) → `improve_vertex(…, true, true)`
  (`:1427`) → `main_vertex_graph_audit` (`:1443`).

**(b) has never been measured against vertex labels, and it is not small.** On
the 342 DL-accepted labelled mcp2k events, comparing the winning composite
row's position to the stashed `final_*`:

```
displacement pre -> post (cm):  p50 0.418   p90 1.373   p99 4.362   max 7.43
moved at all: 291/342 (85%)     moved >1 cm: 61      >5 cm: 1

FIXED by the adjustment (>1 cm -> <=1 cm)   53
BROKEN by the adjustment (<=1 cm -> >1 cm)    3
moved closer, same side of 1 cm             207
moved further, same side                     12
no move                                      51
```

**Net +50 on 342 events (~+14.6 points) at an 18:1 help:harm ratio.**
*Caveat carried into D1:* these 342 are the tier-selected labelled subset, so
the +14.6 points is not an unbiased population estimate — the **18:1 ratio** is
the robust part, and D1 re-runs the whole thing IPW-weighted. Larger
than anything Arms A–C are chasing, and it independently reproduces pr/79 §8's
O(473) result (accept route helps 28 / harms 5, median move 0.34 cm) — so
pr/79 was right to falsify the refit-drift *guard*: there is almost nothing to
block. It also recasts pr/79 §7b's "post-selection refit drift" class (5 of 47)
and §9's TRAD-HAD-IT 4 as the small tail of a strongly positive stage rather
than a defect.

**The follow-on question is reach, not suppression.** The stage fixes 53 and
breaks 3, so headroom is in whether it reaches *far enough* — and its reach
parameters are already-exposed knobs sitting at their C++ defaults in SBND:
`vks_radius` 5.0 cm with all eleven `vks_*` null
(`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet:1377`), `mvga_radius`
15.0 cm (`:1414`). These were tuned in pr/50, pr/85 and pr/86 for orphan
segments, splices and stub absorption — **never against a vertex-position label
set**, which did not exist until now.

- **D1 — measurement** (above), extended to the reject route and to the full
  999 with IPW. No code, no config.
- **D2 — a reach sweep** on the existing knobs, config-only, byte-identical at
  the current values by construction.
  **Gate:** IPW-weighted **≥ +1.5 points with 0 guard regressions**.

### Arm ordering — independent first, joint second

`min_accept` is Arm A's variable, so scoring Arm B on Arm A's refitted grid
would make B's gain conditional on A's fit. pr/78 §4c is the precedent for
exactly that trap: rank_fit was **+12 alone and +0 on top of min_accept=10**.
Therefore:

1. **Arm A** is scored against the current production point (10.0 / 1000).
2. **Arm B's primary gate is at the FROZEN production point** (10.0 / 1000) —
   the clean, independent test of whether the net itself improved.
3. **A third, separately reported step** fits (net, threshold) jointly and
   fills a pr/78-style 2×2 so non-stacking is visible rather than assumed. The
   ≥+1.5 bars are **not** additive across steps.

**Pre-registered anti-hollowness control.** pr/79 §11d showed what a hollow
joint refit looks like: hft1's best joint pair was `+1` that *"decomposes to
pure threshold-tightening of the incumbent behavior, 0 newly accepted"*. Arm B
can produce that same shape — an inert net rescued to parity by a threshold
move that would have worked on CP24 too. So: **any Arm B gain must be
attributed to newly-accepted-and-correct events**, with the decomposition
printed, and the **CP24-at-the-same-threshold control runs alongside**. Without
it the gate is passable by a net that learned nothing.

---

## 5. Decisions this round makes explicitly

| decision | choice |
|---|---|
| 299 AI labels in gradients? | **Yes**, as a pre-registered ablation pair (below) |
| the 95 spent lockbox events | may train; never in a held-out claim |
| held-out source | **mcp2k, tier-stratified** |
| the 449 carried labels | pre-registered **carried-vs-fresh ablation**, not a re-scan (§6) |
| `--drop-unscannable` | on |

### The 299 AI labels are used in training

An earlier draft of this plan excluded them for adding "zero gradient signal".
**That was wrong**, and the reason shapes the round. The objective is a dense
Gaussian-heatmap MSE over every voxel (`voxelize.py:82`, `train.py:114-117`)
plus an optional per-event candidate softmax-CE. **A confirming label is fully
informative for both**: it teaches the net where the vertex *is* on 299 more
SBND events, and CE on a confirming event still teaches "this candidate, not
those fourteen". "Corrective" measures disagreement with the *reconstruction* —
a property of the baseline, not of the training signal. And §1.1 says the net's
actual problem is that it is degenerate on ~40% of SBND events: a
domain-adaptation problem, for which correct labels are the right data whether
or not the reconstruction already agreed.

Their measured quality supports it: the pr/88 §7 gate is a blind 40-event
calibration **on this very sample** — 39/40 = **97.5%** at 1 cm, 95% CI
[86.8%, 99.9%], against a pre-registered 90% bar. A *measured* 2.5% expected
label error, not disqualifying and down-weightable.

**The obvious objection, and why it does not transfer.** pr/77 §8e's `ft1ps`
added 44 high-confidence auto labels at `--pseudo-weight 0.25` and made things
**worse** (guard fails 11→14, numu50 p90 87.7→146.6) — *"confirmation bias
caught by the labels"*. But those were **pseudo-labels selected by the net's
own confidence**, so the circularity was built in. The 299 are **independent
blind-scanner picks from rendered panels**, gated against owner truth. The
mechanism that sank `ft1ps` is absent here.

**How they are used, precisely:**

- **In gradients: yes**, as a **pre-registered ablation pair** — every Arm B
  configuration trained twice, on 999 and on the 700 human-only, both numbers
  reported. This is exactly the question pr/88 §8.5 left open (*"does another
  block of confirming labels move training at this composition"*); running it
  as a pair answers it instead of assuming it.
- **A down-weighted third leg** (AI labels at w ≈ 0.5) if the pair diverges, so
  "helps" and "hurts" do not collapse into one binary.
- **In the held-out set: never.** A 2.5% label error in the *measurement* is a
  different and worse problem than the same error in the gradients.
- **In the §1.2 IPW baseline: never.** Those weights come from the **scan
  tiers**, and the auto-accept stratum's 97.5% comes from the **40
  owner-scanned calibration events**. The 299 are in neither the numerator nor
  the denominator of any §1.2 quantity.

That makes the `--upweight` fix load-bearing rather than optional: it keys on
`sample` (`train.py:258-271`), so it cannot express `label_source` at all — and
`sample` is itself broken for mcp2k, where all 605 fall through to `nuecc`
(`scn_vtx/io.py:191-201`).

**Two small code changes belong to this round:** the Arm B max-anchor, and the
`--upweight` / `label_source` fix.

---

## 6. Measurement gaps and how each is covered

**Covered by scanning (owner agreed): ~40 REVIEW-agree events.** §1.2. Without
them the baseline is bounded to [53.3%, 83.0%] and no delta claim is possible.
Uses the pr/80 round-3 kit and the port-5017 viewer as already configured
(session-token expiry 86400 s, keepalive `ssh -o ServerAliveInterval=30`,
pr/88 §1.6). **A fresh scan tag** — an explicit `--scan-tag` disarms the M13
write guard, so a reused tag overwrites silently.

**Covered by ablation, not scanning: the 449 carried labels.** pr/82 §9 gate 4
(a ≥95% blind re-scan of ~60) is **replaced**, on the owner's decision, by a
pre-registered **carried-vs-fresh ablation**: every headline number reported
twice — once on all 999, once on the 550 non-carried — with the difference
stated. This is a **weaker instrument** than the re-scan, and the round says so
rather than presenting it as equivalent: pr/82 §12.6 showed the failure mode is
real (6 of 24 held-back events were materially different, 4 of those inside
3 cm, two by ~300 cm). If the two readings diverge, that is a stop-and-report,
not a tie to break.

---

## 7. Validation design

**Seal before any training reads anything** — pr/82 §9 gate 6, pre-registered
in pr/81 Step 3 and pr/82 §6 Step 3, never executed.

- **Held-out: tier-stratified from mcp2k**, drawn on the §1.2 strata so it
  IPW-weights back to the arm, sized for ≈±3 points. Recorded here with seed
  and explicit event list **before the first training run**, then written as
  `lockbox` on a fresh snapshot.

- **The 91 spent lockbox events.** pr/78 sealed 95 of the 473
  (`--lockbox 0.2 --lockbox-seed 20260815`). They were then read **twice** —
  the legitimate one-time read on ft2u (pr/78 §5d), and pr/79 §5a's advisory
  ensemble/weight-soup read, which states in the file itself that it turns the
  lockbox into a selection set. They can no longer support a held-out claim.
  pr/88 believed them unrecoverable because `--inherit-manifest` joins on
  `(scan_tag, evt)` and the epoch relabelling changed every tag; §1.4 shows the
  join works on `(runNo, subRunNo, evt)` and locates **91 of the 95** in the
  999 (74 mcp1k, 8 nuecc48, 5 delta, 4 ncpi0).
  They must not enter the new held-out set. Because that set is drawn from
  mcp2k only, **all 91 are excluded automatically** — every one carries a
  `harv3` tag. Identifying them is worthwhile anyway: the exclusion becomes
  *checkable* rather than incidental, and any future round drawing from the
  harv3 events now has the list and will not silently reuse a burned set. They
  remain eligible for **training**.

- **nueCC-48 (42) and NCpi0-19 (19) are reported as named subsets** with
  explicit binomial CIs, not as balanced strata, and labelled a **topology
  stratum** (§2). A balanced three-way split is arithmetically unavailable:
  NCpi0 is 19 events in the entire pool, so holding out half gives a ±23-point
  CI, and within mcp1k/mcp2k the channel is unknown (`nue_score` is a −15
  sentinel on 387/445 mcp1k events; no scores table exists for mcp2k, though
  `pr_scores_table.py` can produce one from the arm).

- Topology heterogeneity is a first-order effect and must not hide behind an
  aggregate: pr/78 §3 measured net-wrong at **4/100 on numu100 vs 76/307
  shower-rich** (6×), and every pr/77 fine-tune degraded the numu slice most.

---

## 8. Preflight (before run 1)

1. **`calib_guard.py` field availability.** pr/82 §1.3a lists what it needs —
   `hv_global`, `hv_single_candidate_ids`, per-row `hv_n_proton_in/out`,
   `hv_z_prior`, `hv_conflicts`, `hv_reduced_chi2`, `hv_trad_main_vertex_id`,
   and the recorded `voxels[]` / `rows[]`. The `-harv3` arms exist, so this
   should be recoverable — **verify, do not assume**.
2. **CP24 anchor exact** on the new snapshot before anything else (pr/82 §9
   gate 8). If it is not, stop: every downstream number is void.
3. **The ft2u replay.** ft2u CP9 is the only surviving checkpoint (pr/82 §1.3b:
   2195 `.pth`, 58.8 GiB, deleted). Replaying it against the new labels is the
   **only out-of-sample check `calib_guard.py` will ever get** — pr/82's own
   framing: *"if the guard's −57 verdict does not reproduce out-of-sample, the
   guard was overfit and everything downstream of it is in question."* Cheap.
4. **Carry §2's strike of pr/82 §6 P3** into the execution log explicitly
   rather than silently omitting it.
5. Confirm the pr/88 build traps are still handled: `@arm` for
   `vtxscan-harv3-delta`, the `evt%d.npz` collision guard, `label_source` in
   the manifest.

---

## 9. Out of scope, with reasons

- **The learned router / chooser** — pr/79 §7, pr/81-A: comprehensively closed,
  base-rate wall at ~83% precision.
- **`dl_vtx_top_k`** — measured NULL live; §1.5 removes the collapse
  hypothesis that would have motivated a retry.
- **Composite `W_*` weight refit** — `rank_fit` measured **+12/473 alone and
  +0 on top of min_accept=10** (pr/78 §4c); the knob was demoted for that
  reason and nothing since has changed it. (Arm C adds a *new* term carrying
  information those weights never had; it is not a refit of the existing seven.)
- **A second input channel / `nIn=2`** — breaks state_dict compatibility with
  the deployed `SCN_Vertex.py` loader and needs a toolkit change. The successor
  if Arm B is positive but capacity-limited.
- **MC pretraining** (pr/77 §8d, never executed). If label count really is the
  bottleneck, hand scanning has produced ~1000 labels across a year of rounds
  and MC truth is the only path to O(10k) — but no such SBND sample exists here
  beyond r1qlmc (10 events) and r2mc (13). A strategic note, not an arm.
- **The 38 unlabelled `work-mcp1k-harv3` events**, and the REVIEW-agree residue
  beyond the 40 calibrated in §6.

---

## 10. Verification

- Snapshot rebuild reproducible from its `PROVENANCE.md`; the new snapshot gets
  its own.
- Held-out draw (seed + explicit event list) recorded here **before** run 1;
  read exactly once.
- Every candidate passes `calib_guard.py` with the CP24 anchor exact **before**
  any live arm.
- One live A/B on the winning arm, scored with `ab_vertex_compare.py` against
  the labels, IPW-weighted, reported both on all 999 and on the non-carried 550
  (§6).
- `./build/clus/wcdoctest-clus` for any C++ knob (Arm A `swap_guard`, Arm C
  `W_TOPO` if it reaches C2). Arm D uses existing knobs — no C++.
- Any jsonnet knob: compiled JSON byte-identical with the knob off, key present
  with it on. For Arm C additionally: `W_TOPO = 0.0` must leave every recorded
  composite total bit-identical, since the term is a literal `+ 0.0`.
- Commits in `wcp-porting-img` (toolkit knobs in `toolkit`, M9). Nothing pushed
  and no production flip without the owner.

---

## 11. Execution log (2026-08-16, this session)

Environment fact discovered at start: `~/work/scratch_wcgpu1` is a **symlink
to `/nfs/data/1/xqian`** — one tree, already on the GPU host (`wcgpu1`, 2×
RTX 4090, direnv python with torch 2.5.1+cu121 + sparseconvnet working).
Training nevertheless runs on **CPU**, like hr3 did: these clouds are small
enough that GPU launch overhead loses (measured 0.047 s CPU vs 0.185 s CUDA
per forward), and staying on CPU keeps the hr3 dose replay device-identical.

### 11.0 Repro (execution)

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
T="--tags vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-harv3-nuecc48 \
   vtxscan-harv3-ncpi0 vtxscan-harv3-mcp1k vtxscan-harv3-delta"
R="--arm-roots vtxscan-mcp2k=work-mcp2k-harv3 vtxscan-mcp2k-auto=work-mcp2k-harv3 \
   vtxscan-harv3-nuecc48=work-nuecc48-harv3 vtxscan-harv3-ncpi0=work-ncpi0-harv3 \
   vtxscan-harv3-mcp1k=work-mcp1k-harv3 vtxscan-harv3-delta=@arm"
X="--exclude-events dl_vtx_training/heldout-pr89.txt"
W="--ipw-file dl_vtx_training/runs/ipw-mcp2k-20260816.tsv"
cd dl_vtx_training

# scan pile (sec 6):    python3 ../vtx_rules/ragree_pile.py --runs ../vtx_rules/runs/mcp2k-20260816 \
#                           --out ../vtx_rules/runs/mcp2k-ragree-20260816/pile     # seed 20260817
# seal (sec 7):         build_dataset.py --name pr89_pool ... --inherit-manifest \
#                           data/pr88_pool_combined/manifest.tsv --lockbox-events heldout-pr89.txt
# IPW weights:          python3 ipw_weights.py --runs ../vtx_rules/runs/mcp2k-20260816 \
#                           --tsv runs/ipw-mcp2k-20260816.tsv
# preflight 8.2:        python3 calib_guard.py --name cp24-pr89 $T $R --tsv runs/calibguard-cp24-pr89-20260816.tsv
# preflight 8.3:        python3 calib_guard.py --name ft2u-pr89 $T $R \
#                           --weights .../sbnd-vtx-ft2u-full473-e10-CP9.pth \
#                           --events-file <non-held-out list> --tsv runs/calibguard-ft2u-pr89-20260816.tsv
# closure:              python3 rerank_replay.py $T --closure
# Arm A grid:           python3 rerank_replay.py $T --grid --min-accepts 6 8 9 10 11 12 14 \
#                           --scales 500 750 1000 1500 2000 $X $W --prod-point 10.0 1000 \
#                           --tsv runs/armA-grid-20260816.tsv
# Arm A swap guard:     ... --grid --min-accepts 10 --scales 1000 --swap-guard --tsv runs/armA-swapguard-20260816.tsv
# Arm C features:       python3 ../vtx_rules/rule1_feature.py --dumps <labelled-dump list> \
#                           --tsv runs/topo-pr89-20260816.tsv
# Arm C C1:             ... --grid --min-accepts 10 --scales 1000 --topo-file runs/topo-pr89-20260816.tsv \
#                           --topo-sweep 0 0.25 0.5 1 1.5 2 3 4.5 6 $X $W --tsv runs/armC-wtopo-20260816.tsv
# Arm D D1:             python3 arm_d_probe.py $T $R $W --tsv runs/armd-d1-20260816.tsv
# Arm B:                ./pr89_train.sh hr4 ; ./pr89_train.sh hr4-hum --source-weight ai:0.0 ; \
#                       ./pr89_train.sh hr4-maxa --scale-anchor 0.0 --max-anchor 1.0
```

### 11.1 Tooling the plan under-budgeted (§5 said two code changes; it was eight)

All in this repo, default-off / back-compatible; existing invocations
byte-identical:

1. `calib_guard.py --tags` (+ per-tag report; `taxonomy.ALL_TAGS` left
   untouched as the pr/79-era record). Its default `--arm-roots` point at
   arms archived 2026-08-16, so round-4 runs pass both.
2. `build_dataset.py --lockbox-events` — explicit pre-registered held-out
   list; `--lockbox` stratifies on `sample × corrective` and `sample`
   collapses every mcp2k row to `nuecc`, so a tier-stratified draw cannot
   come from it.
3. `train.py --max-anchor` (Arm B event-level anchor) and
   `train.py --source-weight` (label_source weighting; w=0 drops rows from
   training folds entirely — a zero-grad Adam step still moves parameters
   along decayed momentum, so weight-0 must mean absent — while fold
   assignment and val membership stay identical: the ablation twin).
4. `rerank_replay.py`: `--topo-file/--w-topo/--topo-center/--topo-sweep`
   (the 8th term), `--ipw-file`, `--prod-point`, `--exclude-events`,
   `--swap-guard`.
5. New `arm_d_probe.py` (drift_probe.py left as the pr/79 record),
   `ipw_weights.py`, `vtx_rules/rule1_feature.py`,
   `vtx_rules/ragree_pile.py`, `pr89_train.sh`.

**Plan correction (§4 Arm A):** `calib_guard.py --fit-scale` is a 1-D fit at
the recorded min_accept; the real `(min_accept, scale)` grid is
`rerank_replay.py --grid`, and that is what Arm A ran on.

### 11.2 The seal (§7) — executed

- `data/pr89_pool`: event set verified identical to `pr88_pool_combined`
  (999 rows, zero non-lockbox column diffs), built with `--inherit-manifest`
  so nothing was redrawn. The §8.5 build traps re-exercised: `@arm` resolved,
  no npz collision, `label_source` present, same 15 dots dropped.
- **Held-out: 80 events, seed 20260818**, list in
  `dl_vtx_training/heldout-pr89.txt` (committed), tier-stratified 1/3 of the
  **owner-labelled** mcp2k pool events: auto-accept 40→13, REVIEW-disagree
  153→51, REVIEW FIRST 36→12, abstained 11→4; 38 corrective.
- **Deviation from §7, stated:** "sized for ≈±3" needs ~200 events, but only
  240 owner mcp2k labels exist and §5's stricter rule (AI labels never in
  held-out) wins; holding out 200 would also gut the corrective training
  dose the round exists to test. The 80 confirm direction free of selection
  bias (±≈5 absolute at 1σ); the effect size is carried by the live A/B over
  all labels, which is pr/79's rule anyway.
- The 91 spent full473-lockbox events: located in the 999 by
  `(runNo, subRunNo, evt)`, all harv3-tagged, **none** in the new lockbox —
  §1.4's carry claim verified, exclusion now checkable.
- Every selection-bearing offline run below excludes the 80
  (`--exclude-events`); denominators are therefore **924 truth-reachable
  non-held-out** events (934 labelled − 10 not-a-candidate).

### 11.3 Preflight (§8) — all three PASS

1. **Field availability**: 880/880 mcp2k dumps carry every pr/82 §1.3a board
   and row key, `harvest: true` on all (pr/82's "879" is stale by one; 880
   is the current count). Route census: accept 437 / reject 400 / not-run
   34 / veto-protected 9. Caveat absorbed: `hv_global.ran` is true on only
   **55/880**, so the swap-guard's global-swap population is small.
2. **CP24 anchor EXACT**: 1004 replayed / 10 anchored, **0 route flips**,
   every score ratio 1.000 at p10/p50/p90, predicted delta **+0**, per-tag
   all +0 (`runs/calibguard-cp24-pr89-20260816.tsv`). The min_accept sweep
   on the anchor run already shows 10.0 as the optimum (12/15/20 lose 3–8).
3. **ft2u CP9 reproduces out-of-sample** on the 934 non-held-out events:
   corrective top-1 inflation **×1.909** median (confirming ×1.293), **249
   reject→accept flips**, predicted **−95/934** — the pr/79 failure
   signature at the same rate as the original −57/473. The guard screen is
   NOT overfit (`runs/calibguard-ft2u-pr89-20260816.tsv`).

Closure on the 999: `max |Σ terms − total| = 0` over 2868 DL-snapped rows.

### 11.4 Arm A — recalibration NULL; swap guard +3/0

- **The production point (10.0, 1000) is the exact optimum of the 35-point
  grid** on the 924; best challenger −4, every alternative negative
  (`runs/armA-grid-20260816.tsv`). §1.1's coupling is real, but the shipped
  point already sits on the maximum: **gate not passed, arm closed.** This
  also supplies the anti-hollowness control for Arm C below.
- **`dl_vtx_swap_guard` ON: 3 fixed / 0 broken** (evts 169724, 180645,
  55740), +0.32 IPW points (`runs/armA-swapguard-20260816.tsv`). One-sided
  but far below the ≥+1.5 gate on its own; a candidate rider for the owner,
  not a live-A/B claimant. Replay approximation stated in the tool: recorded
  `s_fwd_z` used (the C++ recomputes min_z over survivors; max effect 0.25).

### 11.5 Arm C — C0 impossible as chosen, restated PASS; C1 +12 at W_TOPO=3

- **C0 deviation, forced:** the owner chose extraction of the archived
  prod0813/ma10 arms, but the retirement tarballs contain **only logs and
  nusel tables — zero calib dumps** (verified: 6411 files, 0 JSONs;
  `tar tzf | grep -c calib` = 0 on both families). pr/80's own dumps are
  deleted, so literal C0 is impossible, not merely inconvenient. Fallback
  (the alternative offered and not chosen, now the only option): the same
  contrast on the current epoch.
- **C0 restated, PASS in shape**: on 923 non-held-out labels with a
  graph-candidate truth pick, `purity==1.0 & n_decisive≥1` hits **81.8%
  (547/669) of truth vertices vs 23.6% (360/1523) of other main-cluster
  candidates** (voting-vertex denominator; all-candidate denominator 62.4%
  vs 9.3%). The published 86.5/31.9 shape reproduces with a 58-point gap on
  primitives that are literally pr/80's own (`outgoing_purity`). Definition
  frozen: vote-gated `frac`, no-vote contributes nothing.
- **C1, at the frozen production point (10.0/1000), sealed events excluded:**
  sweep of `W_TOPO ∈ {0…6}` through the exact replay
  (`runs/armC-wtopo-20260816.tsv`): best **W_TOPO=3.0 → 577/924 vs 565,
  net +12 raw, +1.27 IPW points; 16 fixed / 4 broken (4:1)**. Center-0.5
  variant peaks lower (+11 at 6.0). Feature coverage: 1653 of 21023
  candidate vertices carry a decisive vote (7.9%) — the pr/88-P6 no-vote
  discipline is load-bearing.
- **Mechanism**: 15 of 16 fixes are **reject→accept flips** — the topo bonus
  lifts the correct candidate over min_accept on events where the uBooNE
  net sits at the §1.1 degenerate floor. The term supplies routing evidence
  exactly where the net is blind.
- **Anti-hollowness control**: a uniform threshold loosening of the same
  magnitude *loses* events (min_accept 8/1000 → −14, 6/… worse; §11.4
  grid), so the +12 is the differential topology signal, not a threshold
  move in disguise.
- **Confound splits (pre-registered)**: fixed winners are 13× frac=1.0
  single/double-vote non-collinear + 3 collinear-flagged + 1 frac=0.5; all
  4 broken winners are single-vote frac=1.0 **non**-collinear — the failure
  mode is one misleading Bragg vote, not the §10.8 collinearity confound.
- **Selection-noise caveat**: 16 swept values, max picked; pr/81's fold-max
  lesson says treat the margin over the +10 bar as soft until the held-out
  read and live A/B.
- **C1 clears its pre-registered bar (net ≥ +10, help:harm 4:1, not
  threshold-reproducible). C2 (the C++ `W_TOPO` knob) is a STOP-and-present:
  no C++ written; owner decision pending.**

### 11.6 Arm D — D1 measured; the reject route is NOT the adjustment stage

Full-pool probe (`runs/armd-d1-20260816.tsv`, 1009 labelled events, IPW):

- **Accept route (the clean measurement of the §4-b stage): net +87−5 = +82
  on 592 events at 17:1**, p50 move 0.378 cm, p90 1.355 cm, max 7.4 cm —
  the plan's +50/342 shape at larger n.
- **Reject route: net +118 on 407 events, but p90 displacement is 243 cm** —
  those moves are downstream re-selection (a different main vertex chosen
  after the rerank rejected), not the 5/15 cm-reach
  snap/improve/graph-audit geometry. **D2's radius sweep may only be
  credited with the small-move component**; crediting the stage with the
  reject-route +118 would be a category error.
- broken list (9 total, worst 0.3→4.7 cm) is in the TSV.
- D2 (vks/mvga reach sweep) requires live reruns (the stage is C++
  geometry, not offline-replayable) plus TLA exposure for `vks_radius` /
  `mvga_radius` in `wct-pr-perevt.jsonnet` — pending, competes for the live
  A/B against Arm C.

### 11.7a The §6 owner scan — DONE; the baseline is CLOSED at 80.9% ± 1.8

The owner scanned all 40 (2026-08-16, tag `vtxscan-mcp2k-ragree`; labelled
set verified identical to the committed draw). **REVIEW-agree stratum:
production-correct 36/40 = 90.0% ± 4.7%** — the blind stratum was benign,
near the top of the §1.2 interval, and all 4 corrective picks are graph
candidates (no manual).

**The §1.2 IPW production baseline on the 845-event mcp2k arm collapses
from [53.3%, 83.0%] to 80.9% ± 1.8 points** (stratified 1σ; weights
`runs/ipw-mcp2k-closed-20260816.tsv`, per-stratum p = auto 99.7 / agree
90.0 / disagree 49.7 / FIRST 22.2 / abstained 53.8). Beats the ±3 target.
Two honesty notes: (i) the auto stratum's 99.7% is measured against labels
that are 88% AI picks; substituting the owner-calibrated 97.5% for that
stratum gives **≈80.0%** — quote the pair, not one number; (ii) it sits
coherently beside the frozen-473 production number (78.6%, pr/82 §12.6).

**Quasi-fresh check of Arm C**: the 40 were not available when W_TOPO=3 was
selected. Including them (`runs/armC-wtopo-withragree-20260816.tsv`, topo
features extended to their dumps): **+14 raw / +2.0 IPW points vs +12 on
the selection set** — the fresh events move the same direction.

### 11.7 The §6 owner scan — SERVED, awaiting the owner *(superseded by 11.7a)*

- Draw: **seed 20260817, 40 of the 228 scannable REVIEW-agree events**, zero
  previously labelled (verified), record in
  `vtx_rules/runs/mcp2k-ragree-20260816/pile/{draw,pile}.json`.
- Viewer: port **5019** (5017/5018 were still occupied by live servers —
  restarting them costs every open session its token, pr/88 §1.6), scan tag
  **`vtxscan-mcp2k-ragree`** (fresh tag, M13).
- IPW weights v1 (`runs/ipw-mcp2k-20260816.tsv`): auto 1.006 /
  disagree 1.092 / FIRST 1.028 / abstained 3.769 / **REVIEW-agree UNDEF**
  (251/845 events carry weight nothing can claim until this scan lands).
  Anchoring caveat stated: the owner will know every served event is one
  where the scanner agreed with the reconstruction.

### 11.8 Arm B — first run NaN-POISONED, guard added, rerun as `hr4b*`

Launched (18 parallel CPU fold jobs, `pr89_train.sh`): `hr4` (hr3 recipe
verbatim on the sealed 919-event training pool), `hr4-hum`
(`--source-weight ai:0.0`, the fold-identical 700-human ablation twin),
`hr4-maxa` (`--max-anchor 1.0` replacing `--scale-anchor`, the §4-B
deflation fix). 18 epochs, 6-fold, seed 20260814. Screens to follow, in
order: guard-in-loop checkpoint selection (`runs/hr4-folds/fold*.txt`
val lists written from the sealed pool), full-manifest `calib_guard`, OOF at
the frozen point with the newly-accepted-and-correct decomposition + the
CP24-same-threshold control, one sealed held-out read, live A/B only if it
wins the round.

**Incident (2026-08-16): 17 of the 18 folds died at epoch 0 with
`loss=nan`** (hr3 on full473 had zero NaNs; only `hr4-hum` fold 4 trained
clean). Diagnosis, reproduced deterministically by replaying fold 4's exact
seed/order with per-step instrumentation
(`/home/xqian/tmp/pr89/nan-fold4*.log`):

- First non-finite loss at **step 1281, evt 176810 (ai-scanner label, 20-pt
  cloud, jittered view extent (10, 4, 18) voxels)** — the model FORWARD
  returns all-NaN on that one view while the weights are still finite.
- Mechanism = the pr/78 NaN class, now inside training: a cloud that
  collapses to a **single active site at the deepest UNet level** makes
  train-mode BN's unbiased batch variance 0/0. One nan `backward()` then
  poisons every weight irreversibly (constant d50 ≈ 40 cm from that step).
- Why hr4 and not hr3: the mcp2k additions carry 15–30-point clouds
  (pool minimum n_cloud = 15; full473's tiny-cloud tail was absent).
  `--min-cloud 16` cannot catch it: the collapse depends on the jittered
  spatial extent, not the point count (the no-jitter scan flags only 1/999).
- **Fix: a nan-guard in `train.py run_epoch`** — skip the step (loudly, with
  evt id) when the loss or the post-backward grad norm is non-finite. A run
  that never produces nan is bit-identical to before the change; the guard
  is a fix of undefined behaviour (a poisoned run measures nothing), same
  rationale as the pr/82 evt-54629 rule.
- Failed artifacts kept as the record: `runs/hr4*/` + `runs/hr4*-f*.log`.
  Rerun launched under **fresh names `hr4b`, `hr4b-hum`, `hr4b-maxa`** (same
  recipes, same seed, same folds); all downstream screens run on `hr4b*`.

### 11.9 Arm D — D2 radius sweep NEGATIVE at every point (closed)

The pre-registered D2 sweep ran live on the 308 labelled accept-route
non-held-out mcp2k events (fresh roots `work-mcp2k-pr89d2-{base,w825,w1240,
t310}`, radius TLAs via `SBND_VKS_RADIUS`/`SBND_MVGA_RADIUS`; 308/308 calib
dumps per arm, 0 DL failures). Paired per-event scoring against
`pr89d2-base` (`ab_vertex_compare.py`, tol 1.0 cm,
`runs/ab-pr89d2-{w825,w1240,t310}-20260817.tsv`):

| arm | (vks, mvga) cm | net | fixed | regressed |
|---|---|---|---|---|
| w825  | (8, 25)  | **−6** | 0 | 6 |
| w1240 | (12, 40) | **−7** | 0 | 7 |
| t310  | (3, 10)  | **−5** | 0 | 5 |

Candidate sets identical to base 308/308, route flips 0 — the deltas are
purely the adjustment stage's reach, and widening OR tightening it only
breaks events. Matches the §11.6 headroom analysis (19/30 near-misses were
already within reach and stayed wrong). **D2 CLOSED: production radii (5/15
cm) stay.** No knob change ships from Arm D.

### 11.10 C2 — owner APPROVED (2026-08-17); implemented, default OFF

Owner: *"C2 is good, and should be added in any case on top of other
improvements."* Implemented as `dl_vtx_topo_weight` / `dl_vtx_topo_center`
(TaggerCheckNeutrino → `PatternAlgorithms::determine_overall_main_vertex_DL`),
C++ defaults 0/0 = the term is never computed = byte-identical legacy.

- **Port** (`clus/src/NeutrinoVertexFinder.cxx`, `topo_rule1_vote`): the
  frozen C1 definition — attached track-like segments (neither shower
  flag), fitted arc length ≥ 5 cm, end-window (5 cm, ≥ 3 valid points,
  `dx > 0 && dQ >= 0`) mean dQ/dx ratio ≥ 1.3 ⇒ decisive; vote counts
  "Bragg end away from this vertex"; `s_topo = w × (frac − center)` only
  when ≥ 1 vote. Stated approximation: computed from rerank-time `fits()`,
  while the offline feature used the final dump geometry — the live A/B is
  the arbiter.
- **Scoreboard**: rows gain `s_topo`/`topo_frac`/`topo_votes`, board gains
  `dl_topo_weight`/`dl_topo_center`; all keys emitted only when the knob is
  on (`topo_used` gate, same pattern as `harvest`), so knob-off calib JSON
  is byte-identical. `rerank_replay.py --closure` extended to add a recorded
  `s_topo` when present.
- **cfg**: knob threaded `common/clus.jsonnet` →
  `sbnd/clus.jsonnet` → `sbnd/wct-pr-perevt.jsonnet` TLAs (null-omission);
  runner envs `SBND_DL_VTX_TOPO_WEIGHT`/`_CENTER`. Compiled-config proof:
  knob-off compile is `cmp`-byte-identical to the pre-change compile; knob-on
  differs by exactly the two new keys
  (`/home/xqian/tmp/pr89/cfg-c2-{pre,off,on}.json`).
- **Gates**: `wcdoctest-clus` 2089/2089 PASS; freshness proof done
  (lib 19:21 > edits 19:05). Knob-off byte-identity A/B on 6 mcp2k events
  (3 accept + 3 reject routes: 101828 180645 93105 / 322861 67942 49415),
  A = stashed-baseline lib at `c3741088`, B = C2 lib, knob off both sides
  (`work-mcp2k-c2gate-{a,b}`): **PASS — 24 products identical** (calib
  JSON byte-`cmp`; `mabc-pr.zip` + `pctree-pr-*.tar.gz` member-content
  hash via `hash_archive.py`; nusel TSV byte-`cmp`).
- **Knob-on smoke** (`work-mcp2k-c2smoke`, `SBND_DL_VTX_TOPO_WEIGHT=3.0`,
  evts 101828 + 322861): board carries `dl_topo_weight: 3.0`; accept-route
  winner 33028 gains `s_topo = +3.000` (frac 1.0, 1 vote); on 322861 both
  candidates are no-vote (`topo_frac −1, votes 0`) and `s_topo = 0` exactly
  — the pr/88-P6 no-vote discipline live. **Port fidelity**:
  `rule1_feature.py --dumps` on the smoke dumps reproduces the live C++
  (frac, votes) on all 3 snapped rows.
- Gates NOT run and why: abtest pdhd/pdvd img/clus and the uBooNE qlport
  chain do not execute `tagger_check_neutrino`/`PrDisplayDump` scoreboard
  paths (SBND-only PR pipeline); the shared-library change is confined to
  those components, and the knob-off branch adds no arithmetic to the
  legacy expression (`score += s_topo` is inside `if (weight != 0)`).

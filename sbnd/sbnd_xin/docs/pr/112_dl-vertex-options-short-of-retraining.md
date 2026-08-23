# doc pr/112 — what we can do about the DL vertex, short of retraining

**Status 2026-08-23 — INVESTIGATION + DESIGN, NO CODE.** No knob is built, no default
moves, no production cfg is touched, no toolkit C++ is edited. Every option below is
*sized* on data and left for the owner to choose. `./build/clus/wcdoctest-clus` is
quoted unchanged as proof the round changed no code. **§5.7 is the implementation
design** for the option this doc recommends (owner, 2026-08-23: *"design this approach
… we will implement it later today"*) — a feasibility audit against the source with
file+line citations, a three-knob surface, a probe-first build order, and a measured
cost. It is a design, not a patch.

**The owner's question.** pr/111 concluded that `fit_exclusion` should stay ON even
though the DL vertex is better with it off. The follow-up:

> *"Other than re-training the network, what else can we do in the meantime to help?
> 1. Can we use the O(1000) to fine tune the model so that they can work with the
> result in exclusion fit? For each event we can get both images with and without
> exclusion fit. 2. Can we update the chain so that we can access both no exclusion fit
> and with exclusion fit for the entire pattern recognition chain? Then we can use the
> no exclusion fit to determine the neutrino vertex on the exclusion fit. This should be
> a knob, since we may not need this once we have retrained the model. Other novel
> ideas?"*

and, on scoping idea 2:

> *"I assume that we can use the same underlying graph, and we also do not need anything
> after the nu vtx determination in the no-exclusion fit case. We only need it to provide
> us a position of the vertex, so that we can use it to improve the final nu vertex
> identification."*

## 0. Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin
# arms regenerated this round (see sec 1); ~22 s/event, PR_JOBS=16
./scripts/pr112_arms.sh harv   nuecc48 16      # exclusion ON + dl_vtx_harvest
./scripts/pr112_arms.sh trad   nuecc48 16      # exclusion ON + DL disabled
./scripts/pr112_arms.sh nofitx mcp1k   16      # fit_exclusion=false + harvest
python3 scripts/pr112_repro_gate.py --sample nuecc48          # sec 1 gate

python3 scripts/pr112_pair.py    --tol 2.0 --tsv runs/pr112-pair-nuecc48.tsv     # sec 4.1 (superseded ruler)
python3 scripts/pr112_offvtx_sim.py --sample nuecc48 \
        --on-arm work-vtx106-harv-base-%s --off-arm work-vtx106-cne-on-%s \
        --tsv runs/pr112-offvtx-nuecc48.tsv                                      # sec 4.2, 5
python3 scripts/pr112_abstain.py --samples nuecc48 --n 8 \
        --tsv runs/pr112-abstain-nuecc48.tsv                                     # sec 6
python3 scripts/pr112_dualchain_sim.py --sample nuecc48 \
        --tsv runs/pr112-dual-nuecc48.tsv                                        # sec 5.2, 5.5
python3 scripts/pr112_pool.py    --n 8     --tsv runs/pr112-pool-nuecc48.tsv     # sec 7
python3 scripts/pr112_design_sizing.py --sample nuecc48                          # sec 5.7.6, 5.7.7b, 5.7.8
```

**Traps carried forward (pr/111 §12).** `SCN_Vertex` returns **packed float32 bytes**,
not a sequence. `top_k=1` takes the legacy branch and returns 3 floats with **no
score** — ask for ≥2. A **rigid translation of the whole cloud is an exact no-op**
(`x - x.min(axis=0)`), so a null perturbation must be per-point. `QL_FIT_EXCLUSION=1`
silently means *false*. New this round: **`PR_EXTRA_STAGES=pr_display` is what writes
`calib-pr-evt<ID>.json`** — without it an arm completes rc=0, reports `ok: 48`, and is
silently useless (cost one full arm pair here before it was caught).

## 1. The arms had to be rebuilt first

The 2026-08-23 retire campaign (`203G→57G, 380 arms`) deleted the arms pr/106 §9/§10
and pr/111 were measured on, **while this round was running**: `work-vtx105-*` went from
8 families to 1 and `work-vtx106-*` from 7 to 4 between two consecutive commands. Every
DL-off (`trad`) arm is gone, and the mcp1k/mcp2k harvest arms with them — so the
statement "all 999 labeled events have both an exclusion-ON and an exclusion-OFF
harvested cloud", true at the start of this session, was false an hour later. The owner
authorised regenerating what the round needs.

New arms (`M13` — never written into a retired name): `work-pr112-{harv,trad,nofitx}-<sample>`,
built by `scripts/pr112_arms.sh` from the surviving Q/L roots `work-<sample>-ql0819`.

**Reproduction gate — `scripts/pr112_repro_gate.py`, PASS 46/46.** The regenerated
`work-pr112-harv-nuecc48` is identical to the surviving `work-vtx106-harv-base-nuecc48`
on every quantity these scripts read: the harvested SCN input cloud (`hv_cloud` x/y/z/q,
**exact float equality**), the candidate set (`n_vertex_rows`, `vertex_ids` in order),
the live route, the DL winner, and the final `main_vertex`. So the regenerated arms are
interchangeable with the retired ones, and the installed binary has not drifted — the
empirical check on `libWireCellClus.so` predating HEAD `b5c9f43a` by 8 minutes
(`b5c9f43a` is the env-gated `WCT_EXCL_DUMP` dump, no code path when unset).

## 2. What pr/111 settled, and what it leaves open

The DL vertex is not starved by exclusion and the net is not more confident without it.
`fit_exclusion` moves the fitted cloud **0.277 cm median**, and the SCN's global argmax
is unstable at a fifth of that: an iid **σ = 0.05 cm** jitter relocates it >2 cm in
**30 %** of draws. The prototype is equally sensitive (WCP 10/34 vs WCT 11/34), uBooNE
as much as SBND (38 % vs 37 %), and it reaches the shipped output. So switching
exclusion **re-rolls a winner-take-all competition**; it does not systematically improve
the input.

That diagnosis constrains every option below. It separates them into two families:

- options that target the **measured** mechanism (instability of the readout), and
- options that target the **unproven** one (a training-distribution mismatch — pr/111 §4
  found *no* systematic confidence gain from removing exclusion, which is evidence
  against it).

## 3. Summary — four options, sized

nueCC48, n = 42 labelled events, original labels (pr/106 convention). **All four
numbers are on the epoch-immune target metric** — target = the pre-DL candidate
nearest the click, hit = the code *picks* that candidate. See §4 for why that
distinction decides this round.

| option | what it is | measured effect | verdict |
|---|---|---|---|
| **C — exclusion distillation** (idea 1) | fine-tune so the ON cloud gives the OFF cloud's answer | **+2 / 42** (4 fixed, **2 broken**) | least-bad, but inside the churn band |
| **B — true dual chain** (idea 2) | **whole PR run exclusion-free**, its vertex transferred in | **+4 / 42** (5 fixed, 1 broken) (§5.2); **+5 guarded** (§5.6) | **the best result in this round**; **1.65× PR wall** (§5.7.6); design in **§5.7** |
| B′ — last-step refit only | `dl_vtx_cloud_no_exclusion`, same graph | +2 / 42 (4 fixed, 2 broken) (§5.1) | already built, default OFF; not idea 2 |
| **A — instability abstention** | don't let the DL override when its answer is unstable | **+0 / 42** best case (§6) | null as a lever, **11× as a flag** |
| **D — pooled readout** | replace the hard argmax with a neighbourhood integral | **+1 / 42**, and **no stability gain** (§7) | refuted on its own prediction |

**One option clears the noise floor and it is the owner's idea 2, run properly.**
The readout-side ideas (A, D) and the last-step refit (B′) all land at +0..+2,
inside the churn band (§5.1's table shows 11 of 47 events churning for a net +1
under a different lever). The **true dual chain** — the whole pattern recognition
run a second time with `fit_exclusion=false`, used only to name the vertex — is
**+4 with a 5:1 fix/break ratio** (**+5 and break-free** once the transfer distance
guards it, §5.6), and it is the only design here that reaches the events whose cure
needs the exclusion-free *topology* rather than just its charge. Its price, measured
rather than guessed, is **1.65× PR wall** (§5.7.6) — not 2×, because the
exclusion-free chain is itself 0.85× production per visit. **§5.7 designs it**, and
finds the chain is already re-entrant: `TaggerCheckNeutrino::visit()` runs its PR
sequence inside a per-candidate loop that already documents what a second pass needs.

⚠ These are **injection-point** numbers — what the transfer *selects*. The production
refinement block then re-points the vertex (0.476 cm median, 13.60 cm max), so the
shipped number is what §5.7.7's live arm has to measure, alongside the `nue_score`
question.

## 4. Option C — idea 1, and the ruler that decides it

`dl_vtx_training/train.py` already carries a **`consistency`** term (label-free view
agreement), today fed the ×4 reflection and jitter views. **The ON/OFF pair is a
drop-in second view**, so the build is small. The question is what it could buy.

A consistency objective makes the two views *agree*; it does not make them agree on
the *right* answer. So the honest ceiling is the signed quantity

```
gain = #(ON wrong AND OFF right) − #(ON right AND OFF wrong)
```

### 4.1 The ruler matters more than the option

Measured two ways, this quantity gives two different answers, and only one of them
is admissible.

**Click ruler (INVALID here) — `scripts/pr112_pair.py`.** Compares the ON arm
against the *global* `fit_exclusion=false` arm, scoring "right" as the net's argmax
landing within 2.0 cm of the hand-scan click. Those are different graphs, so ids do
not correspond and a position ruler is the only option:

```
net argmax correct:  exclusion ON 28/42      exclusion OFF 35/42
pair disagreement (argmax moves > 2.0 cm): 10/42 = 24 %
  ON wrong & OFF right : 7      ON right & OFF wrong : 0      net +7 / 42
```

**That +7/−0 does not survive contact with the right ruler, and it was predictable
that it would not.** `vtx_target_eval.py`'s own docstring records why: *"the
hand-scan labels were taken on fit_exclusion-OFF reconstructions, so a 1 cm match
of the CURRENT fitted vertex to the click is biased against the current sample."*
pr/106 §9 measured what that bias is worth on this exact arm — pr/105 judged nofitx
on the click ruler and got **+135, "mostly epoch."** A click-distance comparison of
an exclusion-ON arm against an exclusion-OFF arm rewards the OFF arm for being the
epoch the labels were clicked in. The 24 % disagreement rate stands (it is a
same-quantity comparison), but the +7/−0 is not a measurement of accuracy.

**Target metric (admissible) — `scripts/pr112_offvtx_sim.py`, `B − A`.** The
same-graph pair (`harv` ON cloud vs `cne` OFF cloud) keeps candidate ids
corresponding, so the epoch-immune metric applies, and the production selector is
used rather than a bare argmax:

```
B - A decomposition (the exclusion-free CLOUD, epoch-immune ruler):
  ON wrong -> OFF right (fix)  : 4  [46363, 235435, 360535, 389538]
  ON right -> OFF wrong (break): 2  [271851, 433451]
  net +2 / 42
```

Those six events are **exactly** pr/106 §10's named fixes and breaks for the same
knob, which is the cross-check that this is the real quantity.

### 4.2 What that does to the recommendation

The case for a **one-directional distillation** objective (OFF = teacher, ON =
student) rested entirely on the **0 breaks** in the click-ruler number. On the
admissible ruler there are **2 breaks against 4 fixes**, so the exclusion-free view
is *not* uniformly the better teacher — it is better on balance, by two events, on
a sample whose churn is larger than that.

**The direction argument is withdrawn.** A distillation objective is still the form
that matches the mechanism (a symmetric consistency loss cannot tell which view is
right, and would average a 4–2 split toward nothing), but "the teacher is never
wrong" is no longer supported, and any campaign would have to carry `271851` and
`433451` as known regressions from the outset.

**Confidence, re-measured on all 42:** median `dl_score` **ON 0.8478 vs OFF
0.8318**. Still no systematic confidence gain from removing exclusion, consistent
with pr/111 §4. So this was never the training-distribution story: a plain
fine-tune on exclusion clouds does not follow from these numbers, and a net
fine-tuned on the new distribution can be exactly as chaotic on it.

### 4.3 Constraints any tuned net must clear

- **`calib_guard.py` before anything live.** The deployed composite consumes
  `1000·dl_score` against `min_accept`, so a score-scale shift is fatal even when
  rank-based metrics look fine — exactly how ft2u passed offline and lost −40/473
  live (pr/79 §3). `train.py`'s `scale_anchor` / `max_anchor` terms exist for this.
- **Labels are hand-scan clicks only.** No MC truth exists in this tree. 999 unique
  labelled events, 700 human / 299 AI-scanner, of which 449 in the current epoch are
  an unvalidated bulk carry. Lockboxes are spent (`vtx106` 353 events, `vtx105`,
  `ft2u` 95) — a new held-out split has to be cut before training, not after.
- **And the labels carry the same epoch bias** that invalidated §4.1's first number.
  Any training target defined by click distance inherits it; the target-anchored
  definition does not.
- **Prior fine-tuning rounds never shipped** (pr/78 ft2u, pr/89 hr3/hr4), and no
  tuned checkpoint survives on disk. This would be a fresh campaign.
- **Report, don't build on it yet:** pr/111 F5 — the training tree
  `T_rec_charge_blob` omits the vertex rows (`flag_skip_vertex=true`) while the
  inference cloud includes a vertex block. A structural train/inference mismatch,
  independent of exclusion.

## 5. Option B — idea 2, and the distinction that decides it

**Correction, owner 2026-08-23:** *"this `dl_vtx_cloud_no_exclusion` is not what I
want, right? I thought that it only has the last step of fitting? Note, the entire
PR is different if I keep the no exclusion fit off, multiple steps, right?"*

Correct on both points, and the distinction is worth more than +2 events.

`m_fit_exclusion` reaches **34 `do_multi_tracking` call sites across five stages** —
`NeutrinoStructureExaminer` (15), `NeutrinoVertexFinder` (7), `NeutrinoGraphAudit`
(5), `NeutrinoPatternBase`/`break_segments` (4), `NeutrinoOtherSegments` (3). Those
stages **edit the graph**: they break segments, merge them, and add and drop
vertices. An exclusion-free chain therefore diverges from production at every one of
them and **ends with a different graph**, not merely a differently-fitted one.

`dl_vtx_cloud_no_exclusion` does something much smaller: **one** refit per cluster,
at DL-vertex time, on the graph the exclusion chain has already finished building
(`NeutrinoVertexFinder.cxx:4796-4809`), then restores it. **Last step only.**

Measured, the two are not close.

### 5.1 The last-step refit — `dl_vtx_cloud_no_exclusion`, already built

Toolkit `14dd031d`, **DEFAULT OFF**; candidate id lists identical to production on
47/47; cost **+8 % PR wall**. `scripts/pr112_offvtx_sim.py` runs all three
same-graph selections on one net:

```
TARGET-hit, n=42
  A production          net(ON ) snap ON   32/42
  B cne knob today      net(OFF) snap ON   34/42
  C snap on OFF positions too                33/42

B - A decomposition (epoch-immune ruler):
  fix 4  [46363, 235435, 360535, 389538]      break 2  [271851, 433451]
```

Those six are exactly pr/106 §10's named movers for this knob — the cross-check that
this is the real quantity. Resolving pr/111's **F4** ordering asymmetry (the cloud's
vertex rows are read at `:4818` while the refit is live, the snap targets at
`:4933`/`:5054`/`:5109` after the restore at `:4840-4848`) makes it *worse* by one,
so that asymmetry is not worth fixing for this purpose.

### 5.2 The true dual chain — what idea 2 actually is

`scripts/pr112_dualchain_sim.py`. The `harv-nofitx` arm **is** the exclusion-free
chain; take its final neutrino vertex and transfer that position into production by
snapping to the nearest production candidate — *"use the no exclusion fit to
determine the neutrino vertex on the exclusion fit"*. Scored on the target metric
defined on the **production** candidate set, since that is the chain that ships.

```
HOW DIFFERENT ARE THE TWO GRAPHS (candidate sets)?
  candidates: production median 104, exclusion-free chain median 106
  same candidate COUNT on 7/42 events
  vertex-id overlap: median 30 % of the production set
  events with an IDENTICAL candidate id set: 0/42

TRANSFER COST (exclusion-free chain vertex -> nearest production candidate):
  median 0.441 cm   p90 1.86   max 3.7 cm ; beyond 2 cm on 4/42 events

TARGET-hit on the PRODUCTION candidate set, n=42
  production (exclusion ON everywhere)      31/42
  TRUE dual chain (OFF chain names the vtx) 35/42   (+4)
  fixed  : 5  [46363, 111412, 235435, 389538, 469665]
  broken : 1  [271851]
```

**Three things this establishes.**

1. **The graphs genuinely differ** — 30 % median id overlap, and *no* event has an
   identical candidate set. §5.1's "same graph" premise, which the owner's first
   scoping assumed and this doc's first version adopted, does not describe a real
   exclusion-free chain.
2. **+4 with a 5:1 fix/break ratio**, against the last-step refit's +2 at 4:2. It
   reaches `111412` and `469665`, which no same-graph design does — consistent with
   pr/106 §10's finding that the cures it could not reproduce "came with the OFF
   *topology*, not only the OFF charge".
3. **The transfer is not where the loss is.** The exclusion-free chain's vertex sits
   0.441 cm (median) from a production candidate; only 4/42 events transfer further
   than 2 cm. Snapping across the two graphs is cheap.

⚠ Baseline reads 31/42 here vs 32/42 in §5.1 because this script derives
production's pick by snapping its shipped `main_vertex` to the nearest candidate,
while §5.1 uses the rerank replay's winner. Both arms are treated identically within
each script, so the deltas are sound; the two baselines are not interchangeable.

### 5.3 What it would cost, and the question that is still open

- **~2× PR wall time.** Bounded by pr/98 (exclusion ON is 1.08× median, 1.7× worst)
  and pr/106 (+8 % for one extra full pass at one site). The owner's scoping —
  *"we do not need anything after the nu vtx determination in the no-exclusion
  case"* — caps it: the OFF chain can stop once it has named a vertex, so the
  taggers, kinematics, BDT scorers and output stages run once, not twice.
- **A knob, as the owner said**, and retired when a retrained net lands.
- **The open question is selection, not vertex count.** §5.1's table shows the
  hybrid penalty is real: global exclusion-OFF costs nothing on nue selection
  (35→36) while the *last-step hybrid* costs −3 (35→32). A dual chain is also a
  hybrid — an exclusion-free vertex handed to an exclusion-fit downstream — so it
  may inherit that penalty. **This round cannot answer it offline**: it needs a live
  arm running the transfer, because the taggers have to re-run on the moved vertex.
  That is the single measurement standing between this result and a build decision.

  | arm | vertex (target metric) | nue-selected |
  |---|---|---|
  | base, exclusion ON | 35/47 | **35** |
  | global `fit_exclusion=false` | 41/47 | **36** (−5/+6) |
  | `dl_vtx_cloud_no_exclusion` ON | 38/47 | **32** (−4/+1) |

  Two cautions on that table: the churn is 11/47 for a net +1, and four of the seven
  vertex "fixes" (`46363, 122660, 268067, 389538`) *lose* nue selection under global
  OFF. **Vertex-right and nue-selected are close to anti-correlated here** — the
  standing reason vertex count is not the objective.

### 5.5 The best result in this round: chain AGREEMENT is a 14× error flag

Running both chains does not only give a better vertex. It gives **two
reconstructions of the same event that are decorrelated in exactly the way that
matters** — different graph, different fit, different net input. pr/111 showed the
SCN argmax is a near-coin-flip on a large minority of events; two physically
distinct chains are two draws from that, and whether they land together is
information.

Measured on the same 42 events (`runs/pr112-dual-nuecc48.tsv`), agreement = the two
chains name the same production candidate:

```
  agree    : 32/42 (76%)   production right 30/32 = 94%
  DISagree : 10/42 (24%)   production right  1/10 = 10%   dual right 5/10 = 50%

  => production wrong 90% when they disagree vs 6% when they agree  =  14.4x

ORACLE (always pick whichever chain is right): 36/42
  headroom above dual-chain: +1     above production: +5
```

**Three consequences, and they are the most useful things this round found.**

1. **Essentially all of production's vertex error is inside the 24 % disagreement
   bucket** — 10 of its 11 misses. Agreement is a sharper flag than the jitter
   ensemble of §6 (14.4× vs 11×), it is physically motivated rather than synthetic,
   and unlike §6's flag it comes with a *better alternative to route to*: on those
   events the OFF chain is right 50 % of the time against production's 10 %.
2. **Simply taking the OFF chain's answer captures 4 of the 5 available points.**
   The oracle that always picks the better chain scores 36/42; the dual chain
   already scores 35/42. So there is **no meaningful gain left in arbitrating
   between the two** — build the transfer, not a chooser.
3. **The remaining error is a 6-event population that neither chain gets** (36/42
   oracle). That is where a retrain, a hand-scan campaign, or a targeted algorithm
   should be aimed — and the flag identifies it prospectively, without labels. It
   shrinks the population needing attention by 4×.

The agreement flag is **free** once both chains run, ships as a per-event quality
label, and is available to every downstream consumer of the vertex. No amount of
net tuning produces it.

### 5.6 The transfer distance is itself the guard — and it removes the only break

Owner: *"We can also use the closest position approach once we have the recommended
nu vtx candidate, right?"* Yes — that is the transfer in §5.2 (nearest production
candidate to the OFF chain's vertex). But the *distance* of that snap is also a
quality signal, and using it as a gate is strictly better than transferring blindly.

Take the OFF chain's vertex only when its nearest production candidate is within
D cm; otherwise keep production's own answer.

```
  D (cm)  hit       xferred  fixed    broken
  0.5     34/42     25       3        0
  1.0     35/42     35       4        0
  2.0     35/42     38       4        0
  2.5     36/42     41       5        0     <- ORACLE
  3.0     36/42     41       5        0
  3.5     36/42     41       5        0
  4.0     35/42     42       5        1
  99      35/42     42       5        1     (unguarded, sec 5.2)
```

Every disagreement event, by transfer distance:

```
   0.00 cm  evt 235435   prod wrong dual RIGHT   <- helps
   0.00 cm  evt 46363    prod wrong dual RIGHT   <- helps
   0.00 cm  evt 52672    prod wrong dual wrong
   0.22 cm  evt 163543   prod wrong dual wrong
   0.30 cm  evt 389538   prod wrong dual RIGHT   <- helps
   0.76 cm  evt 469665   prod wrong dual RIGHT   <- helps
   2.05 cm  evt 38856    prod wrong dual wrong
   2.07 cm  evt 111412   prod wrong dual RIGHT   <- helps
   2.17 cm  evt 30504    prod wrong dual wrong
   3.69 cm  evt 271851   prod RIGHT dual wrong   <- HURTS
```

**The break and the largest transfer are the same event.** `271851` — the one
regression in §5.2, and the one pr/106 §9 also recorded as global-OFF's only nueCC48
break — sits at 3.69 cm, cleanly above the furthest *helpful* transfer (`111412`,
2.07 cm). A guard rejects exactly the bad transfer.

**What is robust and what is not.**

- **Robust: zero breaks at every threshold from 0.5 to 3.5 cm.** That is the property
  that matters for adoption here, where an ADVERSE mover is the stop-the-line class.
  Even a conservative D = 1.0 gives 35/42 at **4 fixed / 0 broken**, strictly better
  than the unguarded 35/42 at 5/1.
- **Not robust: the exact 2.5–3.5 plateau**, which is fixed by a single event's
  separation (3.69 vs 2.17) on n = 42. Treat D as a knob to be set on another sample,
  not as a measured constant. A physically-motivated prior is available: the prototype
  already gates its DL vertex the same way, `dl_vtx_cut = 2.0 cm`
  (`wire-cell-prod-nue-port.cxx:40`), which lands on the conservative 4/0 plateau
  rather than the fitted one.

This also composes with §5.5 rather than replacing it: the guard decides *whether to
transfer*, the agreement flag reports *how much to trust the result either way*.

### 5.7 The design — how the dual chain would actually be built

Owner, 2026-08-23: *"Can you design this approach and add the information into
the md file? We will implement it later today."* This section is that design.
It is a **feasibility audit against the source**, not a patch: every claim below
names the file and line it was read from, and the four numbers that size it were
measured this round from the two arms already on disk.

**Headline: the chain is already re-entrant.** `TaggerCheckNeutrino::visit()`
runs its whole PR sequence inside a loop over neutrino candidates
(`TaggerCheckNeutrino.cxx:1627`, `for nu_index < candidates.size()`), and the
code already documents what a second pass needs — a fresh `TrackFitting` seeded
from the member fitter's parameters (`:1652-1659`), a fresh `PR::Graph`
(`:1710`), and a main-cluster flag guard (`:1590-1624`). A dual chain is that
same second pass with one knob flipped. Nothing structural has to be invented.

**Counter-headline, and the reason the build order in §5.7.5 matters:** that
recipe is **dormant**. Production never takes it. Measured on the arms:
`nu_index > 0` fires on **0 of 47** nueCC48 events and **0 of 437** mcp1k events
(437 events, 192 PR passes total, no event with two). So the dual chain would be
the *first real user* of a code path that has never run. The audit below is
necessary; it is not sufficient. §5.7.5's probe gate is what makes it safe.

#### 5.7.1 Where the split goes

Line numbers are as of `b5c9f43a` and go stale the moment `run_off_chain()` is
inserted; every row names its symbol, which does not.

| line | stage | OFF pass |
|---|---|---|
| `:1627` | `for nu_index` — the candidate loop | shares the loop body |
| `:1652-1659` | fitter selection (member, or fresh for `nu_index>0`) | **needs its own fresh fitter** |
| `:1692` | `preload_clusters` | own preload onto its own fitter |
| `:1710` | `make_shared<PR::Graph>` + `add_graph` | **own graph** |
| `:1713-2140` | `PatternAlgorithms pattern_algos` + ~430 config assignments | own instance, identical except `m_fit_exclusion=false` |
| `:2145-2183` | main-cluster PR (`find_proto_vertex` … `determine_main_vertex`) | runs |
| `:2196-2228` | other-cluster PR | runs |
| `:2241` | `deghosting` | runs |
| `:2250-2304` | `determine_overall_main_vertex[_DL]` | runs — **the OFF pass calls the SCN net too** |
| `:2310-2373` | `snap_to_kink` → `snap_to_junction` → `improve_vertex` → `main_vertex_graph_audit` → `stitch_disconnected` | **runs** — see the note below |
| `:2380` onward | `rough_path_probe`, second `clustering_points`, showers, taggers, kine, trees | **stops here** |
| `:2776-2778` | `grouping.set_track_fitting(...)` | **never reached** ⇒ the OFF fitter is never published |

⚠ **The refinement block is part of vertex determination.** The owner's scoping
was *"we do not need anything after the nu vtx determination in the
no-exclusion case"* — and `:2310-2373` sits after `determine_overall_main_vertex`
but is still vertex determination: it is where `improve_vertex` and the graph
audit settle the position. `pr112_dualchain_sim.py` reads the OFF arm's
**shipped** `main_vertex`, i.e. post-`:2373`, so **the measured +4/+5 requires
this block**. Stopping at `:2304` would price and build a design that was never
measured. The stop point is `:2373`, not `:2304`.

Injection into production: after `:2304` (production's own
`final_main_vertex` is known and its candidate set exists) and **before** `:2310`
(so the snaps and `improve_vertex` polish the transferred vertex, exactly as they
polish today's). The snap target set is every vertex in the production graph —
the same set `determine_overall_main_vertex_DL` builds at
`NeutrinoVertexFinder.cxx:4813-4825`, which is what the offline `hv_cloud`
vertex rows record, so the simulation and the implementation address the same
set by construction.

#### 5.7.2 Re-entrancy audit — what a second pass touches

The question that decides feasibility: does a PR pass leave residue outside the
graph and fitter it was handed? Read out of the source:

- **`PR::Graph`** — created per pass at `:1710`. Segment identity is
  `edge_bundle.index`, the edge index *within its own graph*
  (`NeutrinoPatternBase.cxx:3244`), so there is **no global id counter** for the
  OFF pass to consume and no id shift in production output.
- **`TrackFitting`** — the OFF pass takes its own, per the `:1652` recipe. Its
  `m_cluster_fitted_charge_2d` (the `T_proj_data` source, doc pr/109 §8) is
  therefore isolated, and the writers reach fitters only through
  `collect_nu_fitters` (`root/src/SbndPrMagnifyTrackingVisitor.cxx:38`), which
  walks the `"nu0"`, `"nu1"`, … names registered at `:2776-2778` — a line the
  OFF pass never reaches.
- **`PatternAlgorithms`** — own instance. That isolates the vertex scoreboard by
  construction: `m_vtx_board` is read at exactly one place (`:2746`) and off the
  pass's own object, so `dl_vtx_harvest` and `vertex_scoreboard` cannot pick up
  OFF-pass rows.
- **Facade layer (clusters, blobs, grouping)** — the only write the whole PR
  family makes is `Flags::main_cluster`
  (`NeutrinoPatternBase.cxx:3506/3512` via `swap_main_cluster`, and
  `TaggerCheckNeutrino.cxx:1566/1570/1620`), and it is already guarded and
  restored. Grepped tree-wide, no PR stage writes `cluster.local_pcs()`, a blob
  flag, or grouping state.
- **The one channel that could have broken this, and why it does not.**
  `PatternAlgorithms::transfer_info_from_segment_to_cluster`
  (`NeutrinoPatternBase.cxx:3225`) writes `point_segment_id` and
  `point_flag_shower` into the cluster's `"3d"` cloud and calls
  `cluster.invalidate_segment_data()` — a genuine Facade write that an OFF pass
  would leak. It is **uncalled**: grepped across the whole repo (`clus/`,
  `root/`, `img/`, `aux/`, tests) the only hits are its declaration
  (`NeutrinoPatternBase.h:3090`) and its definition. Pre-existing dead code,
  reported here and **not** touched in this round (CLAUDE.md §5 tie-breaker).
  If a future round revives it, the OFF pass needs a save/restore guard around
  those two arrays.
- **Process-wide PR globals** — `PR::g_shower_traj_refresh_flag`,
  `PR::g_graph_endpoint_policy`, `PR::set_traj_cover_probe`
  (`TaggerCheckNeutrino.cxx:1896-1901`, `:1919`) are written once from config
  and are not exclusion-related, so both passes want the same values.
  `g_port_audit` (`PRGraph.cxx:103`) is a diagnostic atomic counter and would
  simply double-count.

**What the audit cannot see, and the honest limit of it.** A grep finds channels
it is asked about. Two it does not cover: (a) the embedded-Python SCN call — the
OFF pass makes a **second** `WCPPyUtil::SCN_Vertex` inference per event, where
`dl_vtx_cloud_no_exclusion` still makes only one, so per-interpreter state across
two inferences in one event is unaudited (cost is nil, ~10 ms; statefulness is
the question); and (b) anything reached through a pointer the grep did not
follow. That is the whole reason the first thing built is a probe that must pass
a byte-identical gate, not a transfer.

#### 5.7.3 Duplicate the sequence, do not extract it — and test the duplicate

The tempting implementation is to hoist `:2145-2373` into a lambda or helper both
passes call. **That is the refactor CLAUDE.md M10 / §2 forbids**: a production
file with live consumers stays byte-for-byte untouched, and rewriting the
production stage sequence into a shared callable is not byte-identical-when-off
however careful the lambda is.

So: a new private `run_off_chain()` carrying a **duplicated** stage sequence, with
the production block unmodified. Cost, stated so the owner can accept it
knowingly: **241 lines (144 non-comment) duplicated**, and a standing drift risk —
a future round that inserts a stage into the production sequence must insert it
here too or the two chains silently stop being comparable.

**The duplicate has a cheap correctness test, and it should be run before any
physics is read off the OFF chain:** run the OFF pass with
`fit_exclusion=true` — i.e. both passes identical — and assert its vertex equals
production's, event by event. A faithful duplicate must agree exactly. One arm,
and it converts "I copied it carefully" into a measurement. A drift check to
re-run whenever the production sequence changes.

#### 5.7.4 Knob surface

Four keys, all defaulting to today's behaviour, threaded the standard way
(C++ member default → `get(config, …)` → jsonnet key-suppression → `SBND_*` env
in `run_pr_chain_batch.sh`):

| key | default | meaning |
|---|---|---|
| `dl_vtx_dual_chain` | `false` | run the exclusion-free pass at all. **False ⇒ not one instruction executes** — no second graph, no second fitter, no second inference. |
| `dual_chain_transfer` | `false` | may the OFF vertex replace production's pick. **False = probe: the pass runs, its vertex is recorded, nothing moves.** |
| `dual_chain_transfer_max` | `2.0` cm | the §5.6 guard `D`. Read **only** when `dual_chain_transfer` is true; transfer iff the snap distance `<= D`. |
| `dual_chain_allow_cluster_swap` | `true` | may the transfer target live on a different cluster (§5.7.8). |

⚠ **The probe must be its own boolean — do not encode it as `transfer_max = 0`.**
A snap distance is ≥ 0 and **`0.00 cm` is a reachable, and load-bearing, transfer
distance**: §5.6's listing has three events at exactly 0.00 cm (`235435`, `46363`,
`52672`), and two of them are *fixes* — the OFF chain's vertex coincides with a
production candidate that is **not** the one production picked. The offline
instrument already uses `d <= D`. So a `transfer_max = 0` "probe" would move the
vertex on ~3/42 events, `pr85_hash_gate.py` would FAIL, and the failure would look
exactly like the leak the gate exists to detect. Same class of trap as
`QL_FIT_EXCLUSION=1` silently meaning *false* (§0).

`dual_chain_transfer_max` defaults to the prototype's own DL gate,
`dl_vtx_cut = 2.0 cm` (`wire-cell-prod-nue-port.cxx:40`) — physically motivated,
and on the conservative 4-fixed/0-broken plateau rather than the
single-event-determined 2.5–3.5 one. It is inert until `dual_chain_transfer` is on.

**Null-vertex fallback.** If the OFF pass names no vertex, **keep production's own
answer, log it, and count it** — never fall through to an unset pointer. Low risk
but it should not be discovered at implementation time: 0/42 missing on nueCC48,
and pr/106 §9 has the exclusion-free chain succeeding on *more* events than
production (1024 vs 1012 denominators).

#### 5.7.5 Build order — the probe ships first, and it ships §5.5

**Stage 1: probe (`transfer_max = 0`).** The OFF pass runs; its vertex is
recorded next to production's; nothing moves. This is two things at once:

1. **The leakage gate.** A probe arm must be byte-identical to production under
   `scripts/pr85_hash_gate.py`. If it is not, the second pass leaked — and we
   learn that with zero physics at stake, before any vertex has moved. This is
   the check that covers what §5.7.2's grep could not, including the double SCN
   inference. It is also the acceptance test for §5.7.3's duplicated sequence
   at the reachability level (the equality test in §5.7.3 covers fidelity).

   **If the probe gate FAILs, separate the two causes before debugging.** A
   failure conflates a real Facade/fitter leak with interpreter state carried
   across two SCN inferences. Re-run the probe with `SBND_DL_WEIGHTS=''` on both
   passes: nothing transfers in probe mode anyway, so a DL-off probe is still a
   valid leak test and it removes the inference entirely. DL-off probe PASSes and
   DL-on probe FAILs ⇒ the second inference is the culprit, not the pass.
2. **A shippable deliverable on its own.** The probe *is* §5.5's chain-agreement
   flag — the strongest result this round found, a **14.4×** error concentrator
   holding 10 of production's 11 vertex misses in a 24 % bucket — and it produces
   it **without moving the vertex**. So it needs no `nue_score` gate, no ADVERSE
   census, no owner flip on a physics default. It is a new per-event quality
   label available to every downstream consumer, and no amount of net tuning
   produces it.

**Stage 2: transfer (`transfer_max = D`).** Only after the live arm in §5.7.7
reports. This is the part that moves the vertex and therefore carries the whole
validation burden.

Two consequences to write down. The OFF pass must run **first** within the
candidate iteration — it cannot inform a decision already made. And because the
pass sits inside the `nu_index` loop, an N-candidate event costs 2N passes;
today N = 1 on 47/47 nueCC48 and 437/437 mcp1k, so the §5.7.6 projection holds,
but it is an assumption a busier sample could break.

#### 5.7.6 Cost — measured, not bounded

From the per-stage timers already in every arm's log
(`TaggerCheckNeutrino timing: … took … ms`), on the same 47 nueCC48 events:

```
  production (work-vtx106-harv-base-nuecc48)
    TaggerCheckNeutrino visit   median 7436 ms   (80.7 % of the MABC PR job)
    PR job (MABC cumulative)    median 9.32 s    arm total 651 s
  exclusion-free (work-vtx106-harv-nofitx-nuecc48)
    TaggerCheckNeutrino visit   median 6004 ms   = 0.85x production, per event
    up-to-vertex share of it    86.9 % median

  PROJECTED dual chain (OFF pass runs :2145-2373, production runs everything)
    PR-job wall     median 1.65x   mean 1.68x   max 2.08x
    TCN stage       median 1.82x   mean 1.86x   max 2.33x
    arm total       651 s -> 1080 s   (1.66x)
    upper bound, OFF pass running the whole visit: 1.67x / 1.70x / 2.10x
```

Two things worth noting. **The exclusion-free chain is cheaper than production**
(0.85× per visit, and its worst-case tail is far shorter: 42 s vs 77 s on the
slowest main-cluster PR), so the dual chain costs *less* than 2×. And the
projection's spread is narrow — the upper bound differs from the estimate by 1 %,
because the block the OFF pass skips is a small share of the visit.

Not measured: **peak RSS**. The OFF pass adds one graph plus one preloaded
fitter's charge data. The `MEM:` lines in the arm logs are a flat snapshot with
zero increments and do not resolve the PR stage, so this must be read off the
probe arm with `timecmd.py`, not inferred.

Instrument trap, self-inflicted: `/home/xqian/tmp/pr112_time.py` accumulates by
log pattern (`acc[k] += …`). On a real dual-chain arm every timer line appears
twice and it will silently double-count. Any re-use of it must first key on the
pass, or the OFF pass must prefix its timing lines.

#### 5.7.7 What the offline work cannot close — one live arm, two unknowns

**(a) The `nue_score` question (§5.3).** Global exclusion-OFF costs nothing on
selection (35→36); the *last-step hybrid* costs **−3** (35→32). A dual chain is
also a hybrid — an exclusion-free vertex handed to an exclusion-fit downstream —
so it may inherit that penalty. The taggers must re-run on the moved vertex, so
no offline replay can answer it.

**(b) The refinement re-point, which bounds how literally 35/42 and 36/42 can be
read.** Those are **injection-point** numbers: the simulation scores the candidate
the transfer selects. In the real chain that candidate then goes through
`:2310-2373`, and `snap_main_vertex_to_kink` / `snap_main_vertex_to_junction` /
`improve_vertex` *re-point* `final_main_vertex`, they do not merely nudge its fit.
Measured, that block displaces production's vertex by **0.476 cm median, p90
1.13 cm, max 13.60 cm, and beyond 1 cm on 6/42 events** (the OFF chain's own:
0.329 / 1.00 / 1.94). On those events production's refinement of a transferred
vertex may land somewhere the OFF chain's refinement did not — in either
direction. So **35/42 and 36/42 are what the transfer selects, not a predicted
shipped number**, and the same live arm must report the shipped one.

The target metric itself is refinement-immune — it asks which *candidate* the
code picks, and refinement moves the position, not the identity — which is why
the offline number is meaningful at all. It is the shipped position, and
everything downstream reading it, that the arm has to measure.

**The arm:** nueCC48, `dl_vtx_dual_chain=true`, `dual_chain_transfer=true`,
`dual_chain_transfer_max=2.0`,
`PR_EXTRA_STAGES=pr_display`, against `work-vtx106-harv-base-nuecc48`. Read
`nue_score` / nue-selected from `nusel-table.tsv`, the shipped vertex from
`calib-pr-evt*.json`, and run `scripts/pr90_movers.py --tags vtxscan-harv3-nuecc48`
for the ADVERSE census. **ADVERSE movers are the stop-the-line class** — report,
do not tune `D` until the number looks right (CLAUDE.md §5).

#### 5.7.8 One decision the owner should make explicitly

**May the transfer cross clusters?** The snap target set is every vertex in the
production graph, and on **2/42** events the nearest one lives on a different
cluster than production's own pick (`vertex_id // 1000` is the cluster id,
verified 0 mismatches over 39 scoreboard rows):

```
  evt 389538   prod cluster 19 -> 11   at 0.30 cm   production WRONG, dual RIGHT
  evt 52672    prod cluster 76 -> 82   at 0.00 cm   both wrong
```

Restricting the transfer to the production main cluster's own vertices costs
**exactly one event**: 35/42 → 34/42 unguarded, and the guarded sweep tops out at
35/42 (4 fixed / 0 broken) instead of 36/42 (5 fixed / 0 broken). Both are still
break-free at every `D` from 0.5 to 3.5 cm.

Recommendation: **allow it** (`dual_chain_allow_cluster_swap = true`) — it is
worth +1 and 389538 is a genuine 0.30 cm cluster-boundary disagreement, not a
distant jump. But allowing it means the transfer must also carry the main-cluster
swap, and that must **reuse the existing path** — `swap_main_cluster`,
`m_main_vertex_swap_apply` (`:2280-2289`), and `dl_vtx_swap_guard` — rather than
inventing a second one. The knob exists so the restricted variant can be gated
against the permissive one on the same arm.

#### 5.7.9 Acceptance bar (CLAUDE.md §4, instantiated)

- [ ] `dl_vtx_dual_chain=false` byte-identical: `pr85_hash_gate.py` PASS on the
      nueCC48 and mcp1k manifests, labels reported.
- [ ] **Probe gate**: `dl_vtx_dual_chain=true, dual_chain_transfer=false`
      byte-identical to production on the same manifests. This is the leakage
      proof (§5.7.5) and it is not optional. **Not** `transfer_max=0` — that is a
      live guard at zero distance, and it transfers (§5.7.4).
- [ ] On a probe-gate FAIL: the `SBND_DL_WEIGHTS=''` discriminator run (§5.7.5)
      before any other debugging.
- [ ] **Duplicate-fidelity gate**: OFF pass run with `fit_exclusion=true` names
      the same vertex as production, event by event (§5.7.3).
- [ ] Knob-on smoke: the transfer fires and is visible in a quoted log line.
- [ ] `./build/clus/wcdoctest-clus` passes; a new doctest covers the transfer's
      snap-and-guard arithmetic.
- [ ] No iterated pointer-keyed containers introduced (the OFF pass's stage
      sequence inherits `ordered_nodes` / `ordered_edges` throughout).
- [ ] Freshness proof (M1) before the A/B; compiled-config proof for all three
      new jsonnet keys.
- [ ] Wall + peak RSS from `timecmd.py` on the probe arm, against the 1.65×
      projection.
- [ ] `nue_score` and ADVERSE census from the §5.7.7 arm reported **before** any
      flip is proposed.

### 5.4 If a general "both fits everywhere" design is ever wanted

Recorded from the plumbing audit. Note the dual chain above needs **none** of this —
it runs the chain twice and keeps only a position, so no data structure has to hold
two fits at once.

- `Segment::m_fits_noexcl` / `Vertex::m_fit_noexcl` plus a `"fit_noexcl"` named cloud
  are additive and namespace-safe — every existing reader defaults to `"fit"`.
- **Prerequisite:** `dqdx_fit_keep_all_points`, or the two fits are not index-aligned
  (exclusion's third `form_map_graph` pass drops 442 interior points over 47 nueCC48
  events vs 86 with it off).
- **The blocker:** `TrackFitting::m_cluster_fitted_charge_2d` merges last-writer-wins
  per cell and already has a documented failure mode (uBooNE 5384-6528,
  `T_proj_data` Σpred = 0). Two fits per cluster double-write it.
- **dQ/dx consumers need nothing** — pr/108 Test A proved dQ/dx is exactly
  association-independent (382 fits, 45 552 points, max|ΔdQ| = max|Δdx| = 0). This
  also kills the pr/106 §9 shortcut of reading non-excluded association charge
  without refitting.

## 6. Option A — instability abstention: a null as a lever, an 11× flag

The natural response to a brittleness diagnosis: if the net's own answer is visibly
unstable on an event, don't let it override the traditional vertex. Stability is scored
by re-running inference under the pr/111 §7 P1 null (iid σ = 0.05 cm, charge untouched,
8 draws, fixed seed) and measuring how far the argmax scatters. The deployed answer stays
the single unjittered inference — this is a **route** override, not a score change, and
is **not** TTA averaging, which was measured and rejected twice (pr/77 §8a a wash,
82/116 → 82/116 with tails to 357 cm; pr/111 §9 ensemble +2/47 with 2 lost).

Cost is not the obstacle: warm SCN inference is **~10 ms/event on CPU** (349–638 voxels),
against a PR tail of ~22 s/event.

`scripts/pr112_abstain.py`, closure 42/42, fallback = the **real** DL-off arm
`work-pr112-trad-nuecc48` (route `dl-not-run`):

```
primary (n=42)   baseline (production selection) = 32/42
  thresh    hit            abstained delta
  1.0       29/42          14        -3
  2.0       29/42          11        -3
  3.0       32/42          8         +0
  5.0       32/42          8         +0
  12.0      32/42          7         +0
  20.0      31/42          4         -1
```

**Best case +0.** But the signal itself is excellent:

```
top-quartile spread (>=2.57 cm): DL wrong 8/11 = 73%  vs rest 2/31 = 6%   => 11x

spread band    n      DL right     TRAD right
0-1            26     24 (92%)     18 (69%)
1-3            6      6 (100%)     3 (50%)
8-20           4      0 (0%)       1 (25%)
20-1e+09       6      2 (33%)      1 (17%)
```

**The flag works; the fallback does not.** An 11× error concentrator (pr/77 §8a measured
3× with reflections) — but in the bands it flags, the traditional vertex is *also* wrong:
0 % vs 25 % right in the 8–20 cm band, 33 % vs 17 % above 20 cm. Only two events
(`111412`, `389538`) have DL wrong and trad right, against eleven where the DL is right
and trad is not. Abstaining routes to a worse answer.

⚠ **This verdict depends on having the real DL-off arm.** Run against
`hv_trad_main_vertex_id` — the pre-fallback state, pr/106 F16 — the same script reports
a spurious **+1**, because that field understates the live traditional route. The
regenerated `trad` arm is what turns the +1 into +0.

**What the flag is still worth** (not sized this round): it is a per-event confidence
estimate available for 10 ms. It has obvious uses that are *not* vertex routing — ranking
events for hand-scan, weighting an analysis selection, gating the events on which a
downstream tagger trusts the vertex. Those need their own metric and their own round.

## 7. Option D — pooled readout: refuted on its own prediction

pr/111 §10 pointed at "the voxelization and the argmax readout — a 0.5 cm lattice pinned
to a floating origin, read out by a single hard argmax". The obvious fix is to stop
reading a single voxel: score each candidate by the field integrated over a neighbourhood,
which *should* be stable by construction. `scripts/pr112_pool.py`, same P1 null, 8 draws:

```
readout          target-hit   P(chosen candidate flips under the null)
argmax+snap      32/42        mean 0.173   events ever flipping 13/42
pooled R=0.5 cm  29/42        mean 0.229   events ever flipping 19/42
pooled R=1.0 cm  33/42        mean 0.188   events ever flipping 13/42
pooled R=1.5 cm  33/42        mean 0.185   events ever flipping 13/42
pooled R=2.0 cm  33/42        mean 0.196   events ever flipping 14/42
pooled R=3.0 cm  29/42        mean 0.176   events ever flipping 11/42
```

**Pooling does not stabilise the choice** — 0.188 vs 0.173 at the best radius, i.e.
slightly *worse* — and buys at most +1 on accuracy. The prediction that motivated it
fails, and the failure is informative: it confirms pr/111's mechanism. The instability is
**not** the argmax jittering within one peak; it is the peak itself relocating between
clusters (pr/111 §5: `389538`'s rival sits **213 cm** away). No amount of local pooling
around a candidate repairs a field whose global structure has changed.

A pooled score is also not what the net was trained to emit, so deployment would face the
same absolute-scale screen that killed ft2u. Scored on rank only here for that reason.

**Lattice de-phasing**, the other §10 readout idea, is *not* sized this round. The
lattice is pinned to the cloud's own min corner, so an extent change anywhere re-phases
every voxel, and pr/111 measured 11/47 nueCC48 events re-phased by exclusion — but it
also found the large score movers are **not** among them. Given §7's result, a
readout-geometry fix now looks unpromising for the same reason pooling failed; recorded
as unsized rather than dismissed.

## 8. What we are not proposing, and why

- **Turning `fit_exclusion` off globally.** +36 net vertices, but pr/106 §9's own verdict
  is "49 numu breaks … **not recommended**", and pr/98 shipped exclusion ON on fit
  quality (11/12 top movers equal-or-better near the vertex). pr/111 found nothing
  against it. It stays ON.
- **TTA averaging.** Measured twice, a wash both times (§6).
- **Relaxing the exclusion arbitration rule** (a tie margin, or raising the 0.3 cm keep
  floor). pr/109 §9.9 refuted both on their own evidence: exclusion *reassigns* cells
  rather than deleting them — only 41 cells, 0.2 % of charge, lose all support on SBND
  46363 and **0** on uBooNE 6505 — so a margin of 0.1 cm recovers 66 % of 0.2 %.
- **Any further cloud-side intervention.** pr/111 §10: a *fully* exclusion-free cloud
  buys +3/47 on the production target metric, and that is the hard ceiling. §5 confirms
  it from the other direction — the owner's variant lands inside it.
- **Changing a production-ON default on these numbers.** n = 42 labelled nueCC48 events
  is one sample; §9 states what widening costs.

## 9. Scope

n = **42** labelled nueCC48 events for every number in §4–§7. **Owner (2026-08-23):
"for the actual test run nueCC is more than enough."** The mcp1k widening launched
earlier in the round was stopped on that instruction; partial
`work-pr112-{harv,cne,nofitx,trad}-mcp1k` arms remain on disk, incomplete
(~100–120 of 406 events each) and should not be read as arms.

**The all-sample no-exclusion performance already exists** — pr/106 §9, on the
target metric, original labels, 1054 events:

| | ALL | nueCC48 | NCpi0 | mcp1k | mcp2k |
|---|---|---|---|---|---|
| production (exclusion ON) | 767/1012 (75.8 %) | 35/47 | 14/19 | 300/394 | 418/552 |
| **exclusion OFF** | **810/1024 (79.1 %)** | **41/47** | 13/19 | 309/399 | 447/559 |
| DL alone, ON → OFF | 578 → 602 | 34 → **42** | 12 → 11 | 203 → 207 | 329 → 342 |
| no DL, ON → OFF | 708 → 739 | 27 → 34 | 8 → 6 | 288 → 295 | 385 → 404 |

⚠ **Denominators differ** (1012 vs 1024; mcp1k 394→399, mcp2k 552→559) because the
OFF arm has a different candidate cloud — pr/106 flags this, and the percentages
must be quoted with it. NCpi0 goes the other way (14→13, DL-alone 12→11).

That table is the **reference ceiling for §5.2's dual chain across all samples**:
+43/1012 ≈ **+3.3 pp** overall. §5.2 measures nueCC48 at +4/42 ≈ +9.5 pp, i.e. this
sample is on the favourable side of the average — nueCC is where the exclusion-free
chain gains most, and NCpi0 is where it loses. A production decision should carry
the all-sample number, not nueCC48's.

Two gaps, recorded rather than closed:

- §6's jitter flag is superseded by §5.5's agreement flag wherever both chains run,
  but remains the only option if only one chain does.
- The 35 owner adjudications in `qlport/dl_vtx_optimization/dl_master.log` remain
  unscored (pr/111 F6) — keyed to DL voxel rank, and the toolkit's DL decision lines
  still never reach any log at any level tried (pr/111 F3).

## 10. Recommendation — the best way to gain the most on the neutrino vertex

**Build the true dual chain (§5.2), and ship its agreement flag (§5.5).** That is
the largest measured gain in this round and the only design that clears the sample's
noise floor.

Why this and not the alternatives:

- **It is the biggest measured gain.** +4/42 on nueCC48 at a 5:1 fix/break ratio,
  against +2 (4:2) for the last-step refit, +1 for the pooled readout, and +0 for
  jitter abstention. The all-sample ceiling from pr/106 §9 is **75.8 % → 79.1 %**,
  ≈ +43/1012.
- **It reaches what nothing else can.** `111412` and `469665` need the exclusion-free
  *topology*, not just its charge; no same-graph design gets them (§5.2).
- **The transfer is cheap and safe.** The OFF chain's vertex sits 0.441 cm (median)
  from a production candidate; only 4/42 transfer beyond 2 cm.
- **A distance guard reaches the oracle with no regressions.** Transferring only
  when the OFF vertex lands within D cm of a production candidate gives 36/42 at
  D = 2.5–3.5 and **0 breaks at every D from 0.5 to 3.5** (§5.6) — the single break
  is also the single largest transfer. Set D conservatively (the prototype's
  `dl_vtx_cut = 2.0 cm` sits on the 4-fixed/0-broken plateau); the exact optimum is
  fit on one event and must not be treated as measured.
- **Arbitration beyond that is not worth building.** The oracle is 36/42 and the
  guarded transfer already reaches it — take the OFF answer when the guard passes,
  don't write a chooser (§5.5).
- **The flag is the compounding benefit.** 76 % of events ship a vertex that is right
  94 % of the time; the other 24 % carry essentially all the error. That is a
  per-event quality label available to every downstream consumer, free once both
  chains run, and it shrinks the population needing a retrain or a hand-scan by 4×.

**Cost and shape — measured, not guessed (§5.7.6).** **1.65× PR wall** (median; mean
1.68×, worst 2.08×; arm total 651 s → 1080 s), *not* 2× — the exclusion-free chain is
itself 0.85× production per visit and has a much shorter tail. The owner's scoping
caps it: the OFF branch stops once the vertex is settled, so taggers, kinematics, BDT
scorers and output run once. One clarification the cost turns on: the vertex is not
settled at `determine_overall_main_vertex` — the refinement block after it
(`snap_to_kink` → `improve_vertex` → graph audit → stitch, `:2310-2373`) *is* part of
vertex determination, and the measured +4/+5 requires the OFF pass to run it. A knob,
default OFF, retired when a retrained net lands.

**Build it in two stages, and the first stage carries no vertex risk (§5.7.5).** Ship
the **probe** first — the OFF pass runs, its vertex is recorded beside production's,
`dual_chain_transfer = false` so nothing moves (**not** `transfer_max = 0`, which
is a live guard at zero distance and does transfer — §5.7.4). That single arm is both the leakage
gate (a probe arm must be byte-identical under `pr85_hash_gate.py`, which is the only
check that covers what a source grep cannot — including the second SCN inference) and
a shippable deliverable on its own, because **the probe *is* §5.5's agreement flag**.
The transfer follows only after the live arm below reports.

**The one measurement standing between this and a build decision — one arm, two
unknowns (§5.7.7).** (a) A dual chain is a hybrid — an exclusion-free vertex handed to
an exclusion-fit downstream — and the *last-step* hybrid cost **−3 nue-selected**
while global exclusion-OFF cost nothing (§5.3); the taggers must re-run on the moved
vertex, so no offline replay settles it. (b) **35/42 and 36/42 are injection-point
numbers, not predicted shipped numbers**: the transferred candidate still goes through
the refinement block, which *re-points* the vertex (0.476 cm median but 13.60 cm max,
beyond 1 cm on 6/42), so on those events production's refinement may land somewhere
the OFF chain's did not. The target metric is refinement-immune — it scores candidate
identity — which is why the offline number means anything; the shipped position is
not. **Run one live nueCC48 arm** performing the transfer with the downstream stages,
and read both `nue_score` and the shipped vertex off it. If selection holds, build it;
if it repeats the −3, the vertex gain is not worth the selection loss and the answer
reverts to waiting for the retrain.

**The design is §5.7** — split point and code map (§5.7.1), the re-entrancy audit and
its honest limits (§5.7.2), why the stage sequence must be *duplicated* rather than
extracted and the cheap test that validates the duplicate (§5.7.3), the three-knob
surface (§5.7.4), build order (§5.7.5), cost (§5.7.6), what only a live arm can close
(§5.7.7), the one decision the owner should make explicitly — may the transfer cross
clusters, worth +1 on 2/42 events (§5.7.8), and the acceptance bar (§5.7.9).

**Do not build** Options A, D, or the last-step refit as vertex levers (§4.2, §6,
§7). **Option C (fine-tuning)** stays a live but unproven direction: its honest
number is +2 with 2 breaks (§4.1), and its earlier +7/−0 headline came from a ruler
pr/105–106 rejected.

**`fit_exclusion` stays ON** in the production chain. Nothing here is evidence
against it — the dual chain does not turn it off, it runs a second chain beside it.

**What changed during review.** §3, §4, §9 and §10 were rewritten after the Option C
headline was found to rest on the click ruler; §5 was rewritten after the owner
pointed out that `dl_vtx_cloud_no_exclusion` is the *last fitting step only* while a
real exclusion-free chain differs at 34 call sites across five graph-editing stages.
The first two commits (`3a9a25b`, `bc9e050`) carry the superseded framings; they are
corrected here rather than rewritten in history.

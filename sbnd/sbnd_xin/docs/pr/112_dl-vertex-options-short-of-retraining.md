# doc pr/112 — what we can do about the DL vertex, short of retraining

**Status 2026-08-23 — INVESTIGATION ONLY.** No knob is built, no default moves, no
production cfg is touched, no toolkit C++ is edited. Every option below is *sized* on
data and left for the owner to choose. `./build/clus/wcdoctest-clus` is quoted
unchanged as proof the round changed no code.

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
| **B — true dual chain** (idea 2) | **whole PR run exclusion-free**, its vertex transferred in | **+4 / 42** (5 fixed, 1 broken) (§5.2) | **the best result in this round**; costs ~2× PR wall |
| B′ — last-step refit only | `dl_vtx_cloud_no_exclusion`, same graph | +2 / 42 (4 fixed, 2 broken) (§5.1) | already built, default OFF; not idea 2 |
| **A — instability abstention** | don't let the DL override when its answer is unstable | **+0 / 42** best case (§6) | null as a lever, **11× as a flag** |
| **D — pooled readout** | replace the hard argmax with a neighbourhood integral | **+1 / 42**, and **no stability gain** (§7) | refuted on its own prediction |

**One option clears the noise floor and it is the owner's idea 2, run properly.**
The readout-side ideas (A, D) and the last-step refit (B′) all land at +0..+2,
inside the churn band (§5.1's table shows 11 of 47 events churning for a net +1
under a different lever). The **true dual chain** — the whole pattern recognition
run a second time with `fit_exclusion=false`, used only to name the vertex — is
**+4 with a 5:1 fix/break ratio**, and it is the only design here that reaches the
events whose cure needs the exclusion-free *topology* rather than just its charge.
Its price is roughly **2× PR wall time**.

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
- **Arbitration is not worth building.** The oracle is 36/42 and the plain transfer
  already gives 35/42 — take the OFF answer, don't write a chooser (§5.5).
- **The flag is the compounding benefit.** 76 % of events ship a vertex that is right
  94 % of the time; the other 24 % carry essentially all the error. That is a
  per-event quality label available to every downstream consumer, free once both
  chains run, and it shrinks the population needing a retrain or a hand-scan by 4×.

**Cost and shape.** ~2× PR wall, capped by the owner's own scoping: the OFF branch
stops once it has named a vertex, so taggers, kinematics, BDT scorers and output run
once. A knob, default OFF, retired when a retrained net lands.

**The one measurement standing between this and a build decision.** A dual chain is
a hybrid — an exclusion-free vertex handed to an exclusion-fit downstream — and the
*last-step* hybrid cost **−3 nue-selected** while global exclusion-OFF cost nothing
(§5.3). This cannot be settled offline, because the taggers must re-run on the moved
vertex. **Run one live nueCC48 arm** that performs the transfer and re-runs the
downstream stages, and read `nue_score` off it. If selection holds, build it; if it
repeats the −3, the vertex gain is not worth the selection loss and the answer
reverts to waiting for the retrain.

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

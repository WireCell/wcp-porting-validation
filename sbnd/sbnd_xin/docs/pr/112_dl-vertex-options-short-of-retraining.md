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
| **B — OFF branch supplies the vertex** (idea 2) | same graph, exclusion-free fit nominates the vertex | **+1 / 42**, and **−1 vs the knob that already exists** (§5) | already built; the extra step does not pay |
| **A — instability abstention** | don't let the DL override when its answer is unstable | **+0 / 42** best case (§6) | null as a lever, **11× as a flag** |
| **D — pooled readout** | replace the hard argmax with a neighbourhood integral | **+1 / 42**, and **no stability gain** (§7) | refuted on its own prediction |

**Every option lands between +0 and +2 on 42 events.** §5's table shows 11 of 47
events churning for a net +1 under a different lever, so that is the noise floor of
this sample: **nothing measured here clears it.** The pre-registered reading of that
outcome (§9, written before the check that produced it) is that nothing in this
round beats waiting for a retrain — see §10.

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

## 5. Option B — idea 2, and it is already built

The owner's constraint — same graph, OFF branch supplies only a position — collapses the
dual chain onto machinery that already exists. `dl_vtx_cloud_no_exclusion` (toolkit
`14dd031d`, **DEFAULT OFF**) already snapshots every fit, refits each cluster
exclusion-free through a child `TrackFitting` sharing the parent graph, builds the net
input from that fit, and restores bit-for-bit. Candidate id lists identical to
production on 47/47; cost **+8 % PR wall**.

It differs from the owner's design in exactly one place, which pr/111 §11 recorded as
found-not-fixed (**F4**): the cloud's *vertex rows* are read at
`NeutrinoVertexFinder.cxx:4818` **while the refit is live**, but every downstream snap
target (`:4933`, `:5054`, `:5109`) is read **after** the restore at `:4840-4848`. So
today the net sees OFF-fit candidate positions and then snaps to ON-fit ones. **The
owner's design is that asymmetry resolved the other way.** `scripts/pr112_offvtx_sim.py`
simulates all three selections on one graph, one net:

```
CLOSURE  A (production replay == live winner): 42/42
CLOSURE  B (cne replay == live cne winner)   : 40/42   [confirms F4 empirically]

candidate position shift ON vs OFF fit, all candidates pooled (n=4559):
  median 0.0000 cm   p90 0.5094   p99 1.796   max 4.684 cm

TARGET-hit, n=42
  A production          net(ON ) snap ON   32/42
  B cne knob today      net(OFF) snap ON   34/42
  C owner's design      net(OFF) snap OFF  33/42

  B - A = +2      C - A = +1      C - B = -1
```

**The extra step costs a vertex.** The gain comes from the exclusion-free *cloud*, which
the existing knob already delivers; giving the OFF branch its own candidate positions
adds nothing and loses one event. `271851` and `433451` break under C while A gets them
right. Most candidates do not move at all between the two fits (median shift exactly
0.0000 cm) — only a tail does, so there is little for the extra machinery to work with.

**So idea 2's answer is: the thing you want already exists, and running the OFF branch's
own vertex determination on top of it does not improve on it.** What blocks the existing
knob is not the vertex count but the selection cost, and that cost is a *hybrid* effect
— newly measured this round on nueCC48 (`nue_score ≥ 4.3`, the pr/106 §10 threshold; the
`cne-on` column reproduces that doc's number exactly, which is the control):

| arm | vertex (target metric) | nue-selected |
|---|---|---|
| base, exclusion ON | 35/47 | **35** |
| global `fit_exclusion=false` | 41/47 | **36** (−5/+6) |
| `dl_vtx_cloud_no_exclusion` ON | 38/47 | **32** (−4/+1) |

Global OFF costs nothing on selection; **the hybrid costs −3.** The penalty belongs to
the mismatch between an exclusion-free vertex and an exclusion-fit downstream — which is
precisely what idea 2 institutionalises, and what Option C avoids by changing the net
instead of the geometry. Two cautions on that table: the churn is 11/47 for a net +1, and
four of the seven vertex "fixes" (`46363, 122660, 268067, 389538`) *lose* nue selection
under global OFF. **Vertex-right and nue-selected are close to anti-correlated here** —
which is the standing reason vertex count is not the objective.

### If a general "both fits everywhere" design is ever wanted

Recorded from the plumbing audit so the narrow version's advantage is visible:

- `Segment::m_fits_noexcl` / `Vertex::m_fit_noexcl` plus a `"fit_noexcl"` named cloud are
  additive and namespace-safe — every existing reader defaults to `"fit"`.
- **Prerequisite:** `dqdx_fit_keep_all_points`, or the two fits are not index-aligned
  (exclusion's third `form_map_graph` pass drops 442 interior points over 47 nueCC48
  events vs 86 with it off), and no consumer can index across them.
- **The blocker:** `TrackFitting::m_cluster_fitted_charge_2d` merges last-writer-wins per
  cell and already has a documented failure mode (uBooNE 5384-6528, `T_proj_data`
  Σpred = 0). Two fits per cluster double-write it.
- **dQ/dx consumers need nothing** — pr/108 Test A proved dQ/dx is exactly
  association-independent (382 fits, 45 552 points, max|ΔdQ| = max|Δdx| = 0). The
  difference is entirely the trajectory point set and positions. This also kills the
  pr/106 §9 shortcut of reading non-excluded association charge without refitting.
- Cost bound ≈ 2× the *fitting* budget (pr/98: exclusion ON is 1.08× median, 1.7× worst;
  pr/106: +8 % for one extra full pass).

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

## 9. Scope, and what would firm this up

n = **42** labelled nueCC48 events for every number in §4–§7 — one sample, and
small. **All four options move by +0 to +2**, and §5's table shows 11 of 47 events
churning for a net +1 under a different lever. At this n nothing here is
distinguishable from noise. That is the round's actual result.

The widening is cheap and in flight: `work-pr112-{harv,cne,nofitx,trad}-mcp1k`
(406 events) is regenerating, which takes §4's `B − A` and §6's flag to n ≈ 440;
mcp2k (579) would reach the full ~1000. The Q/L roots (`work-<sample>-ql0819`)
survived the retire, so nothing else is needed.

⚠ **The widening must use the same-graph pair.** `pr112_offvtx_sim.py`
(`harv` vs `cne`, ids aligned, target metric) is the instrument; `pr112_pair.py`
(`harv` vs `nofitx`, click ruler) would only measure §4.1's biased quantity more
precisely, because the `harv3-mcp1k` labels carry the same anchoring. The `cne`
mcp1k arm was added for exactly this reason.

Two further gaps, recorded rather than closed:

- §6's flag is worth its own round with a metric suited to a confidence estimate
  rather than a routing decision — ranking events for hand-scan, weighting a
  selection, gating a downstream tagger's trust in the vertex.
- The 35 owner adjudications in `qlport/dl_vtx_optimization/dl_master.log` remain
  unscored (pr/111 F6) — keyed to DL voxel rank, and the toolkit's DL decision lines
  still never reach any log at any level tried (pr/111 F3).

## 10. Recommendation

**On this sample, nothing here beats waiting for a retrain.** That is §9's
pre-registered reading of a ±1–2 outcome, written before the check that produced
it, and it is the honest conclusion at n = 42.

Concretely:

- **Do not start a training campaign on these numbers.** Option C is the least-bad
  at +2/42, but with **2 known breaks** (`271851`, `433451`) and a churn floor above
  its own effect. Its earlier +7/−0 headline came from the click ruler and does not
  survive §4.1. **Finish the mcp1k widening first** (§9): if `B − A` holds a
  positive fix/break ratio at n ≈ 440, that is the green light; if it lands at ±1
  like the rest, the answer is to wait for the retrain and spend the effort there.
- **Do not build Options A, B or D as vertex levers.** B's target is already built
  and default-OFF, and the owner's extension of it measures *worse* than the
  existing knob. A and D are flat, and D is refuted on its own prediction.
- **Keep Option A's stability number as a flag.** It costs 10 ms and is an 11×
  error concentrator. It is the one genuinely new instrument this round produced,
  and its value is not vertex routing (§6, §9).
- **`fit_exclusion` stays ON.** Nothing here is evidence against it.

**What changed during review.** §3, §4, §9 and §10 were rewritten after the
headline number was found to rest on the click ruler; the first commit of this doc
(`3a9a25b`) and its message carry the superseded +7/−0 framing and the
"build Option C" recommendation. Both are corrected here rather than rewritten in
history.

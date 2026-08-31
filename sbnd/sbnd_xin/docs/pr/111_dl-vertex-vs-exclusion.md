# doc pr/111 — Exclusion fit vs. no exclusion: why the DL vertex is better in the latter

**Status (2026-08-22).** Diagnosis round, **no code changed, no knob built, no default moved**.
Every number below comes from arms that already existed on disk plus offline re-runs of the frozen
SCN weights. `./build/clus/wcdoctest-clus` is unchanged (228/228, 2381 assertions) — quoted as
proof the round touched no source.

**Answer in one paragraph.** Exclusion does **not** starve the neutrino vertex of charge, and the
DL net is **not** systematically more confident without it. What `fit_exclusion` does is move
every fitted trajectory point by ~0.28 cm — and the SCN net's *global argmax* is not stable at
that scale. An iid position jitter of **0.05 cm** (one tenth of a voxel, six times below the SBND
slice pitch, far below the fitter's own precision) relocates the net's argmax by more than 2 cm in
**30 %** of draws, and the same test on **uBooNE** — the net's native, in-distribution detector —
gives **38 %**. Exclusion perturbs the cloud by **5.5×** that. So switching `fit_exclusion` re-rolls
a winner-take-all competition over the whole event rather than improving the vertex region. The
nueCC48 **35 → 41 / 47** is that re-roll landing favourably 7 times and unfavourably once — and of
the 7, only **4 are genuine DL improvements** (§2). The two directly comparable numbers are the
**30 %** relocation rate at 0.05 cm and the **0.277 cm** displacement exclusion actually applies;
the pre-registered band test agrees (**H2**, 6/8) but compares two different clouds against a
same-cloud spread, so it is corroboration rather than the primary evidence (§7).

**It reaches the physics output.** In uBooNE, ~**a quarter** of events ship a different neutrino
vertex (`T_tagger.nu_*` moves > 2 cm) depending on `fit_exclusion` — **9/34 prototype, 10/35
toolkit**, 8 of them by more than 20 cm. The prototype's `dl_vtx_cut = 2.0 cm` gate does not absorb
it (§13).

**And the owner's own suggested control settles the remaining alternative.** Running the uBooNE
2×2 — toolkit and prototype, exclusion ON and OFF, DL available in all four — the **prototype is
affected just as much as the toolkit**: turning exclusion off relocates the SCN argmax > 2 cm on
**10/34** events for WCP against **11/34** for WCT, and the prototype's own live `NeutrinoID_DL`
decision flips on **5 of 35** events. So this is not a toolkit port defect (pr/110 hypothesis B) and
§8 shows it is not an SBND out-of-distribution effect either — it is the net (§12).

---

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
python3 scripts/pr111_dl_decomp.py > /tmp/decomp.tsv     # sec 2  ledger
python3 scripts/pr111_scn_validate.py 10                 # sec 3  GATE (must pass first)
python3 scripts/pr111_rival.py                           # sec 5  response at target vs global peak
python3 scripts/pr111_rival_where.py                     # sec 6  where the rival is
python3 scripts/pr111_brittle.py 20                      # sec 7  brittleness + pre-registered verdict
python3 scripts/pr111_ub_brittle.py 20                   # sec 8  uBooNE vs SBND
python3 scripts/pr111_ensemble.py 8                      # sec 9  does averaging help?
```

Arms (all pre-existing, nothing re-run):

| arm | meaning |
|---|---|
| `work-vtx106-harv-base-nuecc48` | `fit_exclusion` **ON** (production) + `dl_vtx_harvest`, 47/47 events |
| `work-vtx106-harv-nofitx-nuecc48` | `fit_exclusion` **OFF** + harvest, 47/47 |
| `work-vtx106-cne-{on,off}-nuecc48` | the pr/106 §10 `dl_vtx_cloud_no_exclusion` pair |
| `qlport/scripts/sweep/pr109e_wct_on` | uBooNE toolkit, exclusion ON, 35 events |

Weights: `wire-cell-data/uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth` (uBooNE-trained; SBND
runs the same file — pr/110).

**Traps.**
- `QL_FIT_EXCLUSION=1` silently means **false** — `uboone-mabc.jsonnet:1513` compares `== "true"`.
  `dl_weights` is **not** wrapped that way; it is a path, so `-A dl_weights=true` is a bogus path.
- `-L clus:debug` does **not** enable `clus.NeutrinoPattern`, where every DL decision is logged.
- `SCN_Vertex(..., top_k=1)` takes the **legacy** branch and returns 3 floats (x,y,z) with **no
  score**. Ask for `top_k>=2` to get the `[x,y,z,score]*K` payload. Rank 0 is the same argmax.
- The shipped `SCN_Vertex` returns **packed float32 bytes**, not a Python sequence.
- A **rigid translation of the whole cloud is an exact no-op**: `SCN_Vertex.py` normalises
  `x - x.min(axis=0)`. It is *not* a usable null perturbation (see §7's note).

---

## 1. What the net actually eats, and how exclusion reaches it

`determine_overall_main_vertex_DL` (`NeutrinoVertexFinder.cxx:4813-4839`) builds the input from two
blocks and nothing else:

- the **vertex block** — one point per graph vertex, `vtx->fit().point`, charge `vtx->fit().dQ`;
- the **segment interior fit points** — `fits[i]` for `1 ≤ i < n−1`, endpoints skipped.

Charge is `q = dQ·0.1 − 1000`. No blob points, no steiner points. **The DL input cloud is exactly
the fitted trajectory**, so `fit_exclusion` reaches the net through both the geometry and the
charge of every point.

`SCN_Vertex.py`: voxel **0.5 cm**; the only feature is the per-voxel **mean q**; the lattice is
sparse, unbounded, and anchored at the cloud's **own min corner** (`x = x - x.min(axis=0)`);
`pred = prediction[:,1] − prediction[:,0]`.

Downstream the net's per-voxel response `dl_score` enters a composite
`total = 1000·dl_score + s_snap + s_clen(≤2) + s_main(2) + s_fv(0.5)`, argmax'd and admitted at
`min_accept = 4.0`. **Two quantities that must never be conflated** — and that previous docs did:

| quantity | meaning |
|---|---|
| `voxels[].dl_score` | what the **net** said |
| `dl_best_score` | what the **selector** chose (composite; its voxel need not be rank 0) |

Because `s_clen + s_main + s_fv ≈ 4.5` on its own clears `min_accept = 4.0`, **a candidate on the
main cluster inside the FV is admitted even when the net is silent.** Measured: the DL vertex is
"accepted" on **45/47** nueCC48 events, and on **7** of those the net's top response is below 0.05,
i.e. it never localised anything. "The DL vertex was accepted" is not "the net found the vertex".

---

## 2. The +6 is four different things, and only four events are genuine DL wins

`scripts/pr111_dl_decomp.py` reproduces pr/106 §9 exactly — nueCC48 **M3 ON 35, OFF 41**, 7 fixes,
1 break — and then classifies each flip. Vertex **IDs drift between arms** (`11002` in ON and
`11004` in OFF are the same physical vertex, 0.18 cm apart), so everything below is measured on
**positions**, never ids.

| evt | class | net@peak ON → OFF | winner moves | target moves | what actually happened |
|---|---|---|---|---|---|
| `46363` | **M-a** | 0.0098 → 0.0894 | 44.0 cm | 0.00 | net response rises 9×, argmax lands on the target |
| `235435` | **M-a** | 0.0109 → 0.0319 | 30.7 cm | 0.00 | same, 3× |
| `389538` | **M-b** | 0.9821 → **0.4276** | **212.9 cm** | 0.18 | cross-cluster swap; net confidence **falls** |
| `122660` | **M-b** | 0.9998 → 0.9780 | 5.3 cm | 0.24 | re-pick at flat confidence |
| `360535` | **M-b** | 0.9930 → 1.0000 | 1.25 cm | 0.18 | ON answer was already **1.09 cm** from target — inside the metric's resolution |
| `268067` | **M-d** | 0.9089 → 0.9282 | **0.48 cm** | **4.51 cm** | **the pick never changed.** Exclusion created an ON-only candidate `15007` that captured the click; with it gone the target reverts to `15004`, which the DL had picked all along |
| `111412` | **M-c** | 0.0094 → 0.0066 | — (rejected) | 1.37 | **the DL declines** (`dl-rerank-reject`) and the traditional vertex `18001`, which was right, takes over — **not a DL improvement at all** |
| `271851` | BREAK | 0.0970 → 0.0142 | 34.4 cm | 2.75 | ON was **exactly** right (0.00 cm); OFF moves 31.7 cm away |

**So the honest ledger of "+6" is: 2 genuine net-response gains, 2 genuine relocations, 1 win
inside the metric's own resolution, 1 target-capture artefact, 1 case of the DL getting out of the
way — minus 1 genuine break.** No previous doc separated these.

`268067` is worth naming separately: it is the one place where exclusion demonstrably *hurts the
candidate set* rather than the net — it fragments an extra vertex into existence 4.5 cm from the
true one, and the target-anchored metric then adopts it. That is a real effect, but it is about
pattern recognition, not about the DL cloud.

---

## 3. GATE — the offline net reproduces the live net bit-exactly

Everything downstream depends on this. `dl_vtx_harvest` (pr/79 §10) stores `vec_xyzq` verbatim;
`scripts/pr111_scn_lib.py` re-runs the **shipped** `pyutil/python/SCN_Vertex.py` on those floats.

```
GATE: 20 reproduced, 0 mismatched;  worst |dscore|=0.000e+00  worst |dpos|=0.000e+00 cm
```

10 events × 2 arms, top-5 voxels each, **exact to the last bit**. `dl_vtx_harvest`'s claim that
offline voxelization reproduces the live input is confirmed.

---

## 4. pr/110 §7.2's discriminating test, now RUN: **no confidence signature**

pr/110 asked whether exclusion-free clouds make the net *more confident* (which would indicate a
training-distribution mismatch, its hypothesis **A**) or merely more often right (hypothesis **B**).
Measured on the correct quantity — `voxels[0].dl_score`, not the composite — over 47 events:

| | median | mean | min | max |
|---|---|---|---|---|
| exclusion **ON** | **0.8923** | 0.6646 | 0.0059 | 1.0000 |
| exclusion **OFF** | **0.8744** | 0.6613 | 0.0055 | 1.0000 |

Net-blind events (`dl_score < 0.05`): **ON 9/47, OFF 7/47.** Per-event the ratio is roughly
symmetric — ~13 events up >5 %, ~14 down >5 %.

**There is no systematic confidence gain from removing exclusion.** pr/110 hypothesis (A) — a
partial-rollout training distribution — predicts one and does not get it. This is evidence
*against* (A), consistent with pr/110's own dating argument, and it means retraining is not
indicated by this observation.

---

## 5. The vertex is not starved — it loses to a rival maximum

The "exclusion punches a charge hole at the junction, so the vertex voxel is starved" story
(`NeutrinoPatternBase.h:963-976`) predicts the net's response **at the true vertex** collapses with
exclusion ON. `scripts/pr111_rival.py` measures exactly that: best response within 2 cm of the
target, against the global peak.

| evt | arm | S@target | S_peak | \|peak−target\| | S@tgt / S_peak |
|---|---|---|---|---|---|
| `389538` | ON | **0.5257** | 0.9821 | **213.00** | 0.535 |
| `389538` | OFF | **0.4276** | 0.4276 | 0.38 | 1.000 |
| `122660` | ON | 0.8098 | 0.9998 | 4.73 | 0.810 |
| `122660` | OFF | 0.9780 | 0.9780 | 0.50 | 1.000 |
| `268067` | ON | 0.3086 | 0.9089 | 3.12 | 0.339 |
| `268067` | OFF | 0.9282 | 0.9282 | 0.77 | 1.000 |
| `46363` | ON | 0.0073 | 0.0098 | 40.83 | 0.748 |
| `46363` | OFF | 0.0894 | 0.0894 | 0.31 | 1.000 |
| `271851` | ON | 0.0970 | 0.0970 | 0.21 | 1.000 |
| `271851` | OFF | 0.0048 | 0.0142 | 26.93 | 0.337 |

**`389538` is decisive: with exclusion ON the net's response at the true vertex is 0.5257 — HIGHER
than the 0.4276 it gives in the arm that gets the vertex right.** The vertex is lost purely because
a 0.9821 rival sits 213 cm away in another cluster. Exclusion OFF wins by **removing the rival**,
not by brightening the vertex. `S@tgt / S_peak = 1.000` in the OFF arm for every fix, and the break
`271851` is the exact mirror image.

The starvation story is refuted on its own prediction.

---

## 6. What the cloud looks like at the rival — a 3 % charge change flips it

`scripts/pr111_rival_where.py`, 3 cm ball around the ON arm's global argmax:

| evt | \|rival−tgt\| | pts ON→OFF | Δpts | ΣdQ ON→OFF | ΔdQ | S at rival ON → OFF |
|---|---|---|---|---|---|---|
| `389538` | 213.00 | 11 → 12 | +9.1 % | 4.21e5 → 5.39e5 | **+28.0 %** | 0.9821 → **0.0235** (−98 %) |
| `122660` | 4.73 | 16 → 17 | +6.2 % | 7.64e5 → 7.90e5 | **+3.4 %** | 0.9998 → **0.0020** (−100 %) |
| `271851` | 0.21 | 6 → 1 | −83.3 % | 1.80e5 → 3.21e4 | −82.1 % | 0.0970 → 0.0000 |
| `268067` | 3.12 | 13 → 13 | 0.0 % | 1.08e6 → 1.07e6 | −1.0 % | 0.9089 → 0.9282 |
| `46363` | 40.83 | 29 → 40 | +37.9 % | 1.75e6 → 1.89e6 | +7.7 % | 0.0098 → 0.0315 |

**At `122660`'s rival, a +3.4 % charge change and a one-point difference take the net's response
from 0.9998 to 0.0020.** That is not a physically meaningful sensitivity. It is what a saturated
classifier does near an arbitrary decision surface.

(`271851`, the break, is the one case with a genuinely large cloud change — exclusion OFF removes
5 of 6 points and 82 % of the charge at the correct vertex. That break is real and mechanistic.)

---

## 7. The measurement that settles it: the net is brittle at a scale far below physics

`scripts/pr111_brittle.py`, N = 20 draws/event, seed 20260822, on the production (exclusion-ON) arm.

- **P1** — iid per-point Gaussian **position jitter, σ = 0.05 cm**. One tenth of a voxel; ~6× below
  the SBND slice pitch (3.13 mm); far below the trajectory fitter's own precision.
- **P2** — **coherent dQ rescale by a single factor 1 ± 0.03**. Changes no geometry at all.

> **Instrument note.** The first null family tried was a *rigid translation* of the whole cloud.
> That is an **exact no-op** — `SCN_Vertex.py` normalises `x − x.min(axis=0)`, so translating every
> point leaves the voxel coordinates bit-identical — and it produced a zero-width band on all 8
> events. The verdict below is from the corrected instrument. The flawed script was deleted, not
> shipped.

| evt | \|Δ net ON−OFF\| | P1 argmax >2 cm | P1 band width | verdict | P2 argmax >2 cm | P2 band width |
|---|---|---|---|---|---|---|
| `46363` | 0.0796 | 9/20 | 0.0282 | OUTSIDE | 0/20 | 0.0024 |
| `235435` | 0.0209 | 13/20 | 0.0750 | inside | 0/20 | 0.0007 |
| `389538` | 0.5545 | 9/20 | 0.3401 | OUTSIDE | 0/20 | 0.0003 |
| `122660` | 0.0217 | 5/20 | 0.5255 | inside | 0/20 | 0.0000 |
| `268067` | 0.0194 | 0/20 | 0.3984 | inside | 0/20 | 0.0263 |
| `271851` | 0.0829 | 2/20 | 0.1571 | inside | 0/20 | 0.0243 |
| `360535` | 0.0070 | 1/20 | 0.1979 | inside | 0/20 | 0.0025 |
| `111412` | 0.0028 | 9/20 | 0.0098 | inside | 0/20 | 0.0037 |

```
POOLED: a negligible perturbation moves the global argmax > 2 cm in
        P1 48/160 (30 %),  P2 0/160 (0 %)
PRE-REGISTERED VERDICT: inside the P1 band on 6/8, outside on 2/8  =>  H2
```

**Pre-registered rule, fixed before the numbers were seen:** inside on ≥5 of 8 ⇒ **H2** (the
exclusion effect is within the net's own input sensitivity); outside on ≥6 ⇒ H1/H3. **Result: H2.**

> **The band comparison is not symmetric, and the headline does not rest on it.** `|Δnet ON−OFF|`
> compares two *different* clouds; the P1 band is the spread of the *same* cloud jittered. A wide
> band says "this cloud's argmax is unstable", not "a 0.28 cm displacement produces effects of this
> size". The load-bearing evidence is the directly comparable pair below — **30 % relocation at
> 0.05 cm vs 0.277 cm of actual displacement** — together with §6's 3.4 %-charge flip and §8's and
> §12's independent replications. Those stand without the band.

Two further facts pin the scale:

- **How far does exclusion actually move the cloud?** Nearest-neighbour ON→OFF displacement over
  the 8 exhibits: **median 0.277 cm**, p90 0.34–1.62 cm, and **55–85 % of points move more than
  0.05 cm**. Exclusion perturbs the cloud by **5.5× the jitter that already relocates the argmax in
  30 % of draws.**
- **The instability is geometric, not calorimetric.** P2 (a coherent 3 % charge rescale) never
  moves the argmax and gives a near-zero band. It is re-binning points across the 0.5 cm lattice
  that destabilises the net — which is also why the lattice's floating anchor matters: because it
  is pinned to the cloud's own min corner, an extent change anywhere re-phases every voxel.
  Measured: **11 of 47** events have their lattice re-phased by exclusion (e.g. `489330`,
  Δy = −0.591 cm). The largest score movers are *not* among them, so re-phasing is a real
  instability but not the movers' mechanism — reported, not sold as the answer.

---

## 8. The owner's uBooNE control: the brittleness is **not** an SBND out-of-distribution effect

The natural next hypothesis is that SBND is simply out of distribution for a uBooNE-trained net.
`scripts/pr111_ub_brittle.py` runs the identical P1 test on uBooNE clouds.

> **Scope caveat, load-bearing.** The uBooNE arms carry no `dl_vtx_harvest`, so their cloud is
> rebuilt from the persisted `T_rec_charge` tree. That is the **end-of-job** fitted trajectory, not
> the DL-time cloud: validated on SBND `46363`, the tree has **887** points against the harvest's
> **731**, and only **23 %** of harvested points have a <0.01 cm match (median nearest distance
> **0.18 cm**). This section therefore does **not** reproduce any uBooNE DL decision. It compares
> *stability*, and to keep that honest **the SBND side is rebuilt the same way**, from its own
> `T_rec_charge`, never from the harvest.

```
uBooNE (toolkit, exclusion ON) POOLED: argmax moves > 2 cm in 53/140 (38 %)
SBND   (toolkit, exclusion ON) POOLED: argmax moves > 2 cm in 59/160 (37 %)
```

**38 % vs 37 %.** The net is exactly as unstable on its own native detector. So:

- The brittleness is a property of **the SCN net and its 0.5 cm voxelization**, not of SBND's
  geometry, segment multiplicity, or the uBooNE-trained weights being used off-detector.
- It follows that **uBooNE production carries the same instability**, and that the prototype's
  `Change to DL vertex!` decisions (31/35 events — the DL relocates the vertex in almost every
  event) rest on the same unstable argmax.
- It also explains pr/109's finding without contradicting it: SBND's 26–37-segment arbitration
  regime is why exclusion *moves the cloud more* on SBND, but the amplifier that turns cloud motion
  into vertex loss is detector-independent.

**Found, not fixed** (report only, §5 tie-breaker): the uBooNE harness's prototype-fidelity gate
compares a **DL-off toolkit** against a **DL-ON prototype reference** — `run_one.sh:61` defaults
`dl_weights` empty, while `run_5384.pl` passes no `-l0` so `flag_dl_vtx` stays `true`
(`wire-cell-prod-nue-port.cxx:35`) — on events where the DL relocates the vertex 31/35 times. Any
residual `n_diff_branches` from `fidelity_compare.py` is partly configuration mismatch, not a
porting defect. Also: toolkit DL-on uses `dl_vtx_rerank=true` (top-5 composite, `min_accept=4`)
while the prototype uses top-1 argmax + `dl_vtx_cut`, and `uboone-mabc.jsonnet` exposes no rerank
TLA — so a like-for-like WCP/WCT DL comparison is not currently configurable.

---

## 9. Does stabilising the readout help? A little, and not reliably

If the argmax is the unstable object, the lever is the **readout**, not the cloud: average the
score field over an ensemble of jittered evaluations, with `fit_exclusion` left ON and the cloud
untouched. `scripts/pr111_ensemble.py`, M = 8 draws at σ = 0.05 cm, all 47 nueCC48 events,
accumulated on the unjittered lattice. Ruler here is the **net's raw argmax within 2 cm of the
target** — deliberately *not* the production selection metric, so it isolates the net.

```
argmax within 2.0 cm of target:  raw 33/47   ensemble-mean 35/47
  RECOVERED: 137238, 196649, 235435, 489330        LOST: 423981, 469665
```

**+2 net, with 2 lost.** Same shuffling character, same order as everything else in this family.
This is a measured ceiling on a proposal, not a proposal to ship.

⚠ **Different rulers, do not add these up.** §9's +2 is on the *net's raw argmax within 2 cm*
(33/47 baseline); §10's "+3/47 ceiling" is on the *production target metric* (35/47 baseline). They
measure the same family of effect at the same order of magnitude, but they are not the same
denominator and the two gains are not additive.

---

## 10. What this means for the levers already on the table

- **`dl_vtx_cloud_no_exclusion` (pr/106 §10, built, DEFAULT OFF)** recovers 5 of the 7 §9 fixes
  (missing `122660`, `268067`), breaks `433451`, and costs **nue-selected 35 → 32 with 2 ADVERSE**.
  Read against §7, that is exactly what re-rolling a chaotic argmax looks like: it wins some, loses
  others, and the physics selection pays. **It should stay off, and the reason is now mechanistic
  rather than empirical.** (Footnote, since the recommendation is "stay off": that knob has an
  ordering asymmetry — the cloud's **vertex rows** are read at `NeutrinoVertexFinder.cxx:4818`
  while the no-exclusion refit is live, but every downstream snap target (`:4933`, `:5054`,
  `:5109`) is read *after* the restore at `:4840-4848`. The in-code comment at `:4775-4778` claims
  the whole remainder of the function sees the exclusion fit; that is true of the snap targets but
  not of the cloud's own vertex block. Sub-voxel in most events — but §7 is precisely about
  sub-voxel effects.)
- **A *fully* exclusion-free cloud buys +3/47 on the target metric — that is the hard ceiling** on
  any cloud-side intervention, and §9 shows readout-side stabilisation buys about the same. There
  is no large systematic gain hiding in this direction.
- **`fit_exclusion` should stay ON.** Nothing here is evidence against it. pr/98 shipped it on fit
  quality (11/12 movers equal-or-better near the vertex) and pr/109 confirmed uBooNE gains on the
  2-D charge residual in both implementations. Its apparent DL-vertex cost is not a cost of the
  feature; it is the net's instability sampled at two arbitrary points.
- **Vertex count is not the objective.** Of the +6, one event (`111412`) is the DL *declining*, one
  (`268067`) is the metric's target moving onto an unchanged pick, and one (`360535`) is a 1.09 cm
  distinction. The defensible statement is **+4 genuine, −1 break**.

**The direction with actual headroom is none of the above.** §7 says the net's response is
saturated and its argmax is decided at a scale ~0.05 cm; §8 says that is true on uBooNE too; §4
says confidence is not the problem. Together those point at the *voxelization and the argmax
readout* — a 0.5 cm lattice pinned to a floating origin, read out by a single hard argmax — rather
than at anything upstream in pattern recognition. Any real fix is a change to how the net is
evaluated (or a retrain with jitter augmentation), and both are outside a knob round.

---

## 11. Scope, limitations, and one observation left for later

- n = 47 nueCC48 events (SBND) and 8 of 35 (uBooNE, stability only). The M-a/M-b/M-c/M-d ledger is
  8 exhibit events — every flip in the sample, but a single sample.
- §8 compares **stability only**; the owner's actual 2×2 is **§12**, and it was run (4 arms × 35
  events, all complete). §12 is still label-free: it compares each code against *itself* across the
  exclusion switch. It does **not** score either code against the 35 owner adjudications in
  `qlport/dl_vtx_optimization/dl_master.log` — that ledger is keyed to DL *voxel rank*, so scoring
  it would require the live top-5 from each arm, and the toolkit's DL decision lines are not emitted
  at the log level these runs used (see §12's note). That is the natural next step and is cheap.
- SCN inference is not bit-stable (M4); every number here is from a **deterministic offline re-run**
  of stored clouds with a fixed seed, which sidesteps that entirely. Any *live* uBooNE DL arm would
  need repeat runs first.
- **Observed, no hypothesis built on it:** the training tree `T_rec_charge_blob` **omits the vertex
  rows** — `fill_skeleton_info` is called with `flag_skip_vertex=true`
  (`wire-cell-prod-nue-port.cxx:3241`) — while the inference cloud **includes** a vertex block
  (`NeutrinoVertexFinder.cxx:4814-4825`). That is a structural train/inference difference in the
  net's own input format, independent of exclusion, and it is the kind of thing that would decide a
  future retraining question.


---

## 12. The owner's uBooNE 2×2: **the prototype is affected just as much**

> *"I wonder if the MicroBooNE toolkit code (turn on and off this feature) vs. prototype running
> may help."* — owner, 2026-08-22

Four arms, **35 events**, all complete, DL available in all four
(`scripts/pr111_ub_arms.sh`, forked from `pr108_wct.sh` + `run_wcp.sh` so the pr/108 arms and the
stored references `prototype_base/nue_5384_*.root` are untouched, M13):

| arm | code | lever | location |
|---|---|---|---|
| `pr111_wct_on_dl` / `_off_dl` | toolkit | `QL_FIT_EXCLUSION=true` / `false` | `qlport/scripts/sweep/` |
| `pr111_wcp/{on,off}` | prototype | default ON / `WCP_FIT_EXCLUSION=0` | `/home/xqian/tmp/pr111_wcp/` |

`fit_exclusion` read back from each toolkit job's own log (`fit_exclusion=true` / `false`) — never
from the env, because the uBooNE TLA compares `== "true"`.

### 12.1 Like-for-like, label-free (`scripts/pr111_ub_2x2.py`)

Both codes' clouds rebuilt the same way from their own `T_rec_charge`, then the **same frozen CP24
weights with the same top-1 rule** — which removes the live asymmetry (toolkit `dl_vtx_rerank=true`,
top-5 composite, `min_accept=4`; prototype top-1 argmax + `dl_vtx_cut`; `uboone-mabc.jsonnet`
exposes no rerank TLA). Same end-of-job-cloud caveat as §8, and it applies identically to both
sides, which is what a WCT-vs-WCP comparison needs.

| code | cloud ON→OFF displacement (median) | SCN argmax move (median) | > 2 cm | > 20 cm |
|---|---|---|---|---|
| **toolkit** | 0.071 cm | 0.71 cm | **11/34 (32 %)** | 7/34 |
| **prototype** | **0.143 cm** | 0.50 cm | **10/34 (29 %)** | 8/34 |

**32 % vs 29 %** — statistically indistinguishable, and the *prototype's* cloud actually moves
**twice as far** when exclusion is switched.

⚠ **How good is that reconstruction? Measured, and it is a rate proxy only.** §13(A) joins these
offline numbers against the prototype's own *live* logged DNN point, event by event: they agree
within 2 cm on **23/34** and fall in the same class (both > 2 cm or both ≤ 2 cm) on **26/34**, but
disagree badly on individual events (`6888` live 292.8 cm vs offline 0.00; `6852` live 0.00 vs
offline 332.8). The **aggregate rates** track well (live 12/34 vs offline 10/34 above 2 cm), which
is what the WCT-vs-WCP comparison above uses — but **no per-event claim should be read off §12.1**,
and the toolkit column inherits the same uncertainty since it is built the same way.

### 12.2 The prototype's own live decision, from its own logs

Independent of the offline reconstruction — counting `Change to DL vertex!` /
`Stay with the traditional vertex!` and the logged `NeutrinoID_DL: DNN:` regressed point:

| prototype arm | Change to DL vertex | Stay with traditional |
|---|---|---|
| exclusion **ON** | 34/35 | 0 |
| exclusion **OFF** | 29/35 | **5** |

**The prototype's live DL decision flips on 5 of 35 events**, and its DNN regressed point moves
> 2 cm on **12/34** events (median 0.61 cm). Events where the decision flips: **6520, 6650, 6786, 6806, 6888** — with DNN
displacements of 3.4, 86.4, 284.3, 102.8 and 292.8 cm respectively.

### 12.3 What this settles

- **pr/110 hypothesis (B) — "the toolkit's exclusion is not WCP's exclusion, and that is why the DL
  vertex suffers" — is refuted as an explanation for the DL-vertex sensitivity.** Both codes are
  perturbed at the same rate. (pr/108's finding that the two implementations differ in *measured
  impact* stands; it is simply not what drives the DL vertex.)
- Combined with §8 (uBooNE 38 % vs SBND 37 %), **neither the implementation nor the detector is the
  variable.** The sensitivity is the net's own argmax instability, and `fit_exclusion` is one of
  many things that perturbs the cloud enough to re-roll it.
- **uBooNE production carries this too — and §13(B) shows it is not absorbed by the gate.** The
  prototype rejects a DL pick further than `dl_vtx_cut = 2.0 cm` from the nearest candidate
  (`wire-cell-prod-nue-port.cxx:40`), so one could hope the instability never reaches the output. It
  does: the **shipped** neutrino vertex `T_tagger.nu_{x,y,z}` moves > 2 cm between the exclusion-ON
  and exclusion-OFF arms on **9/34 prototype** and **10/35 toolkit** events (> 20 cm on 8 each).

**Note on logging, found not fixed.** The toolkit's DL decision lines never appear in these logs.
`-L clus:debug` does not reach the sub-logger; `scripts/pr111_run_one.sh` (a one-line fork of
`run_one.sh` adding `-L clus.NeutrinoPattern:debug`, committed here) does make `[clus.NeutrinoPattern]`
lines appear — but still no DL lines, and **the SBND production logs show the same zero count** even
on events whose scoreboard proves the DL ran. So this is a general toolkit logging gap, not a uBooNE
harness problem. It did not affect any result here (everything is read from the scoreboard or
computed offline), but it is why §12 scores the toolkit offline rather than from its own log.


---

## 13. Two checks §12 does not settle on its own

`scripts/pr111_ub_propagate.py`, on the same four arms.

### (A) Is the end-of-job cloud a good stand-in for the DL-time cloud?

The prototype is the one arm where ground truth exists — it logs its live
`NeutrinoID_DL: DNN: x y z` point — so §12.1's offline argmax can be joined against it per event.

```
n=34: |live - offline| < 2 cm on 23/34;  same class (both >2 or both <=2) on 26/34
      aggregate rate above 2 cm: live 12/34  vs  offline 10/34
```

**Verdict: a good *rate* proxy, a poor *per-event* one.** Some events disagree completely — `6888`
moves 292.8 cm live and 0.00 cm offline; `6852` is the reverse (0.00 live, 332.8 offline). So §8's
and §12.1's caveat is **strengthened, not weakened**: the aggregate WCT-vs-WCP and
SBND-vs-uBooNE *rates* are sound, and no per-event statement from those sections is.

This does not touch §2–§7 or §9, which are all computed on the **harvested** SBND cloud that
reproduces the live network bit-exactly (§3).

### (B) Does the instability reach the shipped vertex, or does `dl_vtx_cut` absorb it?

The prototype rejects a DL pick further than **2.0 cm** from the nearest candidate
(`wire-cell-prod-nue-port.cxx:40`) — and §12.2's 5 decision flips are exactly that gate firing. So
the fair question is what happens to `T_tagger.nu_{x,y,z}`, the vertex that actually ships.

| code | final ν vertex move, exclusion ON vs OFF | > 2 cm | > 20 cm |
|---|---|---|---|
| prototype | median 0.14 cm | **9/34 (26 %)** | 8/34 |
| toolkit | median 0.26 cm | **10/35 (29 %)** | 8/35 |

**The gate does not absorb it.** Roughly **a quarter of uBooNE events ship a different neutrino
vertex** depending on `fit_exclusion` — the same rate in both codes, and 8 of them by more than
20 cm. The instability characterised in §7 is not an internal wobble of an intermediate quantity;
it propagates to the physics output, on the net's native detector, in the reference implementation.

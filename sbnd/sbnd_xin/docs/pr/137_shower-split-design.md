# doc pr/137 — the shower SPLITTER: design, and an honest feasibility answer

**Status: DESIGN + FEASIBILITY PROBE, 2026-08-31. No code, no arms, no knobs.
The probe result is uncomfortable and is stated first: the split is the easy
half, the trigger is the hard half, and every physics-motivated trigger I could
measure tonight gives ≤ 27 % purity against the available proxy. §5 argues that
number is a lower bound and names the one experiment that settles it.
Successor to doc pr/136 §11.9. toolkit `76249b4b`, wcp `9839c808`.**

**The owner's architectural case for merge-then-split is in §1.1 and it is
backed by three independent measurements — most directly that 31 of 90
hand-marked showers are over- AND under-clustered at the same time, which no
single admission predicate can fix.** The feasibility risk is confined to the
trigger (§4–§5), not to the architecture.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
scripts/pr137_split_feasibility.py   # §3  can a blind split recover a known partition?
scripts/pr137_split_separation.py    # §4a geometric trigger: merged vs single
scripts/pr137_split_pi0gate.py       # §4b pi0-mass trigger
scripts/pr137_split_convgap.py       # §4c two-conversion-gap trigger
scripts/pr137_split_refpoint.py      # §1.2a which reference point the split should use
```

All four are READ-ONLY and run off the round-3 arms
(`work-pr136-{off2,onV1c90}-*`) and their `emprep-136*` sidecars.

## 1. The charter

Owner, 2026-08-31, after scanning the round-3 Bee package:

> *"I like the onV1c90 result, since I feel it did a better job in getting the
> shower clustered. The main issue is the overclustering, which we have at the
> start of the point. But now, I feel what we should do is to invent a new
> algorithm to do a split given the shower."*

So the shape is **cluster generously, then split**: keep the pass-4 `angle_v1`
escape's completeness gain, and repair its cost with a new pass that takes an
over-complete shower and divides it.

### 1.1 The architectural argument — the owner's, and it is measured

Owner, same session:

> *"the reason that I propose this new idea is also due to the overclustering
> that we saw, the overclustering and underclustering are mixed, it is rather
> difficult to do them right at the same time. A clean approach would be merge
> them together, and then separate cleanly. This should be a better design than
> the current approach, which is very tricky to tune."*

**This is not a hunch; it is the shape of every measurement this campaign has
produced.** Three independent lines:

1. **The two errors coexist inside the same object.** doc pr/136 §10.1, on the 90
   hand-marked showers at the production point: **pure UNDER 16 | pure OVER 29 |
   BOTH 31 | clean 14**. *Thirty-one of ninety showers are simultaneously
   missing charge they should hold and holding charge they should not.* A single
   admission predicate has one knob-direction and cannot move those two the
   opposite ways at once.

2. **Every round paid in the other direction.** The pr/136 arms are the cleanest
   demonstration, all from one predicate with one threshold moved:

   | arm | `q_miss` | `q_extra` |
   |---|---|---|
   | OFF | 14.0 % | 7.0 % |
   | `onV1` | **10.4 %** | **12.1 %** |
   | `onV1c90` | 11.3 % | 12.2 % |
   | `onV1c90d25` | 12.5 % | **7.0 %** |

   Loosening buys completeness and sells purity; tightening does the reverse.
   The knob traces a curve; it never leaves it. Eleven rounds of pr/123 → pr/136
   are eleven points on that curve.

3. **The seat census says the curve is short anyway.** doc pr/136 §10.2: only
   **35.5–48 %** of the missing charge is reachable by *any* predicate at *any*
   existing seat, and **0 %** of the conn-1 (π⁰ γ) missing charge was ever
   refused by one.

**Merge-then-split breaks the coupling by making them two decisions instead of
one.** The merge decision optimises a single objective — *lose no charge* — and
is allowed to be greedy, which is what `onV1c90` already does well by the
owner's own scan. The split decision then optimises a different single
objective — *put the boundary in the right place* — with the whole object in
hand, information the admission seat structurally does not have (pass 4 decides
one segment at a time, before the shower it is building exists). §3 measures
what that extra information is worth: **0.920 median purity** for a fifteen-line
spherical 2-means, against eleven rounds of threshold work that never moved both
metrics the right way at once.

**Where the argument is not yet proven, stated plainly:** it converts a hard
*tuning* problem into a hard *triggering* problem. §4 shows the trigger is not
free, and §5 says the honest purity is unknown until the owner scans. The
architecture is sound; the open question is whether the second decision is
easier to make than the first, and that is what stage 2 measures.

### 1.2 The owner's design refinement, and the measurement that agrees with it

Owner, same session:

> *"the split can be a direction based from the targeted vertex, the majority
> would be a 2-gamma split. Of course, we want a gate to separate the so-called
> 1-gamma (no split) vs. 2-gamma. There may be small cases that there are
> multiple gammas."*

Three specifications, and tonight's probe independently confirms all three:

**(a) Direction from the targeted vertex.** This was the winning criterion of §3
before the specification arrived, and the reference point matters:

| reference point for the ray clustering | median purity | ≥ 0.90 | ≥ 0.99 |
|---|---|---|---|
| **the ν main vertex** | **0.920** | **25/44 (57 %)** | **15** |
| the object's charge centroid | 0.878 | 18/44 (41 %) | 12 |
| the shower's own start point | 0.864 | 20/44 (45 %) | 13 |

The ν vertex wins, and it should: two γs from a π⁰ share a decay point that sits
at the ν vertex, so they separate in *direction from it*. The shower's own start
is a poor reference precisely because it lies inside one of the two parts.
(`scripts/pr137_split_refpoint.py`.) **Design consequence:** the reference point
is a *parameter*, not a constant — the NC chain already re-seats a π⁰ decay
point away from the ν vertex (K24, `id_pi0_backproject_vertex`), so the pass
should take "the targeted vertex" from the shower's own π⁰ context when one
exists and fall back to the main vertex otherwise. Stage 1's tape prints the
statistic under both so the choice is measured, not assumed.

**(b) k = 2 is the default, and the data says so.** Of the 49 merges §2 found,
**44 are two-way and 5 are three-way — 90 % / 10 %**. So the pass runs k = 2
first and only tries k = 3 when a residual test fires; it never searches k
freely, which would find structure in every real shower.

**(c) A 1-γ vs 2-γ gate is required.** The owner names it as a requirement, and
§4 is the measurement of how hard it is. That agreement matters: the gate is not
an implementation detail bolted on at the end, it is the deliverable. §6's
staging exists so the gate is *fitted to owner verdicts* rather than guessed.

**Why this is well-posed and the previous eleven rounds were not.** pr/123 →
pr/136 all asked *"should this shower admit that segment?"* — an admission
question, answered one segment at a time with no view of the object that
results. doc pr/136 §10.2 measured the ceiling on that whole family: only
35.5–48 % of the missing charge is reachable by any predicate at any existing
seat, and **0 % of the conn-1 (ν-vertex-attached, i.e. π⁰ γ) missing charge was
ever refused by one**. A splitter asks a different question — *"where does one
shower end and the next begin?"* — and it is the question doc pr/136 §11.9's two
verdicts both point at.

**It also owns a class nothing else can reach.** 314838, per the owner: the
second half of one γ was clustered onto the other γ *before any knob ran*. That
is doc pr/130 §6's **EM-vs-EM mis-partition**, disjoint from every shipped
guard, and unreachable by admission predicates by construction — the charge is
already inside an EM object. Only a boundary-redrawing pass can touch it.

## 2. The population — measured, and small

Comparing membership between `emprep-136off2` and `emprep-136onV1c90` over all
239 events, counting for each `onV1c90` shower how many distinct OFF showers
contributed > 5 % of its charge and > 1e5:

| | count |
|---|---|
| ON showers absorbing exactly 1 OFF shower | 1006 |
| **2-way merges** | **44** |
| **3-way merges** | **5** |
| (ON showers with no OFF ancestor above threshold) | 1446 |

**49 merges over 40 events**, ~2 % of showers. Above 1e6 charge and ≥ 3
segments — the population any splitter would actually run on — it is **33
merged against 354 single**.

The largest, with the OFF ancestors that got fused:

| event | ON shower | charge | fused from |
|---|---|---|---|
| 396222 | 9059 | 4.60e7 | OFF9059 51 %, OFF9084 32 %, OFF128276 15 % |
| 176502 | 109119 | 3.89e7 | OFF109119 86 %, OFF109141 13 % |
| 415278 | 23037 | 2.16e7 | OFF23037 94 %, OFF23047 6 % |
| 269774 | 13237 | 1.71e7 | OFF13237 81 %, OFF97197 19 % |
| 142421 | 108104 | 1.59e7 | OFF108104 63 %, OFF7010 37 % |
| 314838 | 110088 | 1.39e7 | OFF110088 82 %, OFF13010 15 % |

Note 142421 and 314838 sit in this table: the escape's best recovery and its one
π⁰ loss are *the same operation*, seen from the two sides. That is exactly why
a splitter is the right instrument — it can keep the charge and restore the
boundary.

## 3. The split itself is the easy half

Take each 2-way merge, hide the answer, and ask a blind criterion to recover the
OFF partition. Score = charge-weighted purity of the best 2↔2 assignment.

| blind criterion | n | median purity | ≥ 0.90 | ≥ 0.99 |
|---|---|---|---|---|
| **2-means on unit rays from the ν vertex** | 44 | **0.920** | **25 (57 %)** | **15** |
| connected components, 2 cm gap | 44 | 0.893 | 21 (48 %) | 12 |
| connected components, 4 cm | 44 | 0.882 | 20 (45 %) | 11 |
| connected components, 8 cm | 44 | 0.863 | 19 (43 %) | 12 |
| connected components, 20 cm | 44 | 0.791 | 11 (25 %) | 5 |
| 2-means on positions | 44 | 0.769 | 16 (36 %) | 12 |

**The angular criterion wins, and it is the physically motivated one**: two γs
from a π⁰ leave a common decay point along two rays, so their members separate
in *direction from the vertex*, not in position. A naive 15-iteration spherical
2-means already reaches 0.920 median purity with 15 of 44 essentially perfect.

The geometry backs it: over the 44 true splits the two parts are separated by a
median **9.4 cm gap (86 % > 2 cm)** and a median **14.0° opening angle from the
vertex (59 % > 10°)**.

**Nothing in this campaign needs a better splitting kernel.** If a shower is
known to be two objects, we can divide it.

## 4. The trigger is the hard half — three candidates, all measured weak

The splitter must decide *which* showers to split. Run the same ray-split on
every `onV1c90` shower (q > 1e6, ≥ 3 segments) and ask whether any statistic
separates the 33 MERGED from the 354 SINGLE.

### 4a. Geometric (opening angle, charge balance, inter-part gap)

The two classes overlap badly — MERGED angle p10/med/p90 = 7.8 / 30.7 / 68.6°
against SINGLE 5.2 / 25.8 / 118.7°; gap 5.0 / 15.4 / 39.5 cm against
0.0 / 12.8 / 43.6 cm.

| accept test | merged fired | single fired | enrichment | **purity** |
|---|---|---|---|---|
| angle > 10, bal > 0.10 | 11/33 (33 %) | 35/354 (10 %) | 3.4× | 24 % |
| angle > 15, bal > 0.15, gap > 2 cm | 9/33 (27 %) | 19/354 (5 %) | 5.1× | 32 % |
| angle > 25, bal > 0.20, gap > 4 cm | 5/33 (15 %) | 10/354 (3 %) | 5.4× | 33 % |
| angle > 30, bal > 0.25, gap > 4 cm | 4/33 (12 %) | 7/354 (2 %) | 6.1× | 36 % |

**Singles outnumber merges 11:1, so a 5× enrichment still means two of every
three splits are wrong.**

### 4b. The π⁰-mass gate — DEAD

The attractive idea: don't split blindly, split only when the two parts make a
π⁰. Form `m = √(4E₁E₂)·sin(θ/2)` from the two parts' ray directions and their
charge (energy from the parent shower's own `kine_charge / Σ dQ`):

| window | merged | single | enrichment |
|---|---|---|---|
| (100, 160) MeV | 1/33 (3 %) | 24/354 (7 %) | **0.4×** |
| (100, 160) + min E > 15 MeV | 1/33 (3 %) | 10/354 (3 %) | 1.1× |
| (110, 160) + min E > 20 MeV | 1/33 (3 %) | 4/354 (1 %) | 2.7× |

**One merged case passes (269774, m = 156 MeV).** The gate has essentially no
population, and no enrichment. The reason is now obvious in hindsight and worth
writing down: **most merges are not two γs.** They are a shower plus a
fragment, a satellite, or a track prong. A π⁰ gate selects the rare sub-case and
throws away the rest of the problem.

### 4c. The two-conversion-gap signature — also weak

The physics: two γs each convert *away* from the ν vertex, so **both** parts of
a true pair start at a distance; an artificial split of one real shower puts the
shower's own start in one part, whose vertex gap is then ≈ 0. Statistic =
min over the two parts of (vertex → nearest point of part).

MERGED p10/med/p90 = 0.0 / 20.0 / 60.2 cm; SINGLE = 0.0 / 7.5 / 81.3 cm — the
medians do separate (20 vs 7.5 cm), but the tails swamp it:

| accept test | merged | single | enrichment | purity |
|---|---|---|---|---|
| min-gap > 8 cm, angle > 15, bal > 0.15 | 7/33 (21 %) | 19/354 (5 %) | 4.0× | **27 %** |
| min-gap > 10 cm, angle > 20, bal > 0.15 | 5/33 (15 %) | 14/354 (4 %) | 3.8× | 26 % |

**Best purity across all three families: 27–36 %.** As stated, that is not
shippable: three of four splits would break a shower that was fine.

## 5. Why 27 % is a LOWER bound, and the one experiment that settles it

**The "SINGLE" class is not truth.** A `onV1c90` shower containing one OFF
shower can still be over-clustered — OFF's own `q_extra` is 7.0 % of target
charge with **29 pure-OVER showers** in the hand scan (doc pr/136 §10.1), and
the owner has just confirmed over-clustering exists at the OFF operating point
too (*"which we have at the start of the point"*). So an unknown fraction of the
14–24 "false" fires are **genuine over-clusters that OFF also got wrong** — in
which case the splitter is right and the proxy is wrong.

This is the same trap doc pr/136 §11.3 hit from the other side, and the same
answer applies: **the proxy cannot adjudicate; only the owner can.**

> **The decisive experiment is not more algorithm work. It is a hand scan of the
> ~19–26 candidates that fire the `angle > 15, bal > 0.15, gap > 2 cm` test** —
> 9 merged + 19 single, one Bee package, one afternoon. If most of the 19
> "singles" are real over-clusters, purity jumps from 32 % toward 90 % and the
> design ships. If they are healthy showers, the geometric trigger is dead and
> §7's fallback is the only route.

## 6. The design that follows from this

**Probe first, knob second.** Given §4 and §5, writing an accept test now would
be fitting a threshold to a proxy the campaign already knows is contaminated —
precisely what CLAUDE.md §5.7 and pr/130's "stop threshold work on admission
features" warn against. So the first deliverable is **a splitter that splits
nothing**.

### Stage 1 — `WCT_SHOWER_SPLIT_DEBUG`, a stderr probe (no knob, byte-neutral)

Placed after `examine_showers` (the last pass that grows a shower) and before
the second kinematics recompute. For every shower above a charge floor with ≥ 3
segments it runs the ray 2-means and prints one line:

```
SHOWER_SPLIT cand shower=<id> nseg=<n> q=<Q> angle=<deg> balance=<f>
    gap_cm=<g> vgap0_cm=<d0> vgap1_cm=<d1> m_pi0=<MeV> e0=<MeV> e1=<MeV>
    part0=<seg,seg,...> part1=<seg,seg,...>
```

Everything §4 measured, per candidate, on the real arm. Changes no bytes; gated
on `getenv` exactly like `WCT_SHOWER_XCLUS_DEBUG` (toolkit `deca3467`) and
proven with the same 478/478 hash gate.

### Stage 2 — the owner scan

`prep_em_scan.py --parse-probes` gains a `splits` section; a Bee package is
built from the fired candidates in charge order, OFF vs a *hypothetical* ON
rendering that colours `part0` / `part1` differently (the `shower_track-global`
layer already colours by `real_cluster_id = cluster*1000 + segment`, so the two
parts can be shown as two colours with no reconstruction change). The owner
labels each **SPLIT / KEEP**.

### Stage 3 — the accept test, fitted to owner verdicts

Only now is a predicate written, and it is fitted to SPLIT/KEEP labels rather
than to the OFF partition. Knobs, all DEFAULT OFF:

| knob | type | default | meaning |
|---|---|---|---|
| `shower_split_rays` | bool | `false` | run the pass at all |
| `shower_split_min_mev` | double | `0` | charge floor on the parent shower |
| `shower_split_min_angle` | double (deg) | `0` | minimum ray opening between the two parts |
| `shower_split_min_balance` | double | `0` | minimum minor-part charge fraction |
| `shower_split_min_gap` | double (cm) | `0` | minimum inter-part gap |
| `shower_split_min_vgap` | double (cm) | `0` | minimum of the two parts' vertex gaps |
| `shower_split_max_parts` | int | `2` | k; 3 only reachable when the k = 2 residual test fires (10 % of the population, §1.2b) |
| `shower_split_ref` | string | `"pi0_then_main"` | reference point: the shower's π⁰ decay vertex when it has one, else the ν main vertex (§1.2a) |

`false` ⇒ the pass is never called ⇒ byte-identical, gated on the standard
239-event manifest.

### Stage 4 — composition with pr/136

The splitter is only worth having if it lets the escape ship. The arm to run is
**`onV1c90` + splitter**, judged against the doc pr/136 §11.2 instruments with
the owner's new verdict folded in:

- `q_extra` must return to ≈ 7.0 % (the OFF value) — this is the whole point;
- `q_miss` must stay near `onV1c90`'s 11.3 %, i.e. the completeness gain survives;
- π⁰ census exact must be **≥ 33** (`onV1c90`'s value), not merely ≥ 32;
- and the owner must not find new merged γs in a rescan of the §2 population.

## 7. Fallback if the geometric trigger dies in stage 2

**Split every candidate, then let a downstream chooser undo it.** Instead of
deciding *whether* to split, always produce both hypotheses and let the π⁰
finder and the PF/kine consumers pick — i.e. move the decision from clustering
to selection, where a mass constraint and a vertex constraint exist. This is
strictly more machinery (two hypotheses per shower through the whole tail) and
should not be attempted unless stage 2 kills the trigger; it is recorded so the
next session does not rediscover it as a new idea.

A second, narrower fallback: **restrict the splitter to the escape's own
admissions.** The knob knows exactly which segments it added (32 for `d25`, 302
for `c90`); a splitter that may only cut along those boundaries has a background
of ~300 rather than 354 whole showers, and a natural semantics — *a deferred
decision on the escape's admissions, re-evaluated with the full shower context
that pass 4 did not have*.

## 8. What this design does NOT claim

- **It is not a promise that splitting works.** §4 measured three triggers and
  none is shippable on the available proxy. §5 says why that may be the proxy's
  fault, and stage 2 is how we find out — not by argument.
- **It does not touch the ν vertex.** The vertex is used as a *reference point*
  for the ray clustering, which is the same use K24 and the NC back-projection
  already make of it; no vertex is moved or refit.
- **It does not assume `onV1c90` ships.** If the splitter fails, the surviving
  pr/136 candidate remains `onV1c90d25`, which already holds `q_extra` flat.

## 9. First actions next session

1. Write stage 1 (`WCT_SHOWER_SPLIT_DEBUG`) — one probe, one arm, hash-gate it.
2. Build the stage-2 Bee package from the fired candidates, ordered by charge,
   with `part0`/`part1` coloured.
3. **Hold for the owner scan.** Do not write an accept test before those
   verdicts exist.
4. Independently of the splitter: **adjudicate 181050**, the one row where
   `onV1c90d25` makes `q_miss` worse (doc pr/136 §11.9), since it is the last
   open item on the surviving pr/136 candidate.

---

# Round 2 — the TRIGGER: literature borrow, offline bake-off, curated label set

**Status: MEASUREMENT COMPLETE, 2026-08-31. No code, no arm, no knob, no flip —
everything below is a read against arms already on disk.
Owner: *"For this round, let's explore the algorithm further and design the work
plan."* §10 is the reframe the literature forces, §11 turns the owner's three
factors into features, §12 the in-situ null model, §13 the bake-off, §14 the
curated set and the agent scan. Two corrections to §4 are carried, and the
headline is that the trigger roughly doubled: 27–36 % → 58 % against the proxy,
and 80 % against fresh hand labels.**

## Repro (round 2)

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
scripts/pr137_null_model.py                    # §12 -> docs/pr/pr137-null-model.tsv
scripts/pr137_seed_split.py                    # §13a -> docs/pr/pr137-seed-split.tsv
scripts/pr137_trigger_bakeoff.py               # §13b -> docs/pr/pr137-trigger-bakeoff.tsv
scripts/pr137_curate.py --sheets               # §14 -> docs/pr/pr137-curated-set.tsv
                                               #      + work/pr137_sheets/*.png (BLIND)
```

All four are READ-ONLY and share `scripts/pr137_lib.py` / `scripts/pr137_features.py`.
They run off `work-pr136-{off2,onV1c90}-*` and the `emprep-136*` sidecars.
Agent scan labels: `em_labels/splitscan-0901-agent/`.

## 10. What the literature does, and why §4 was the wrong shape

The owner asked whether this problem has a published solution. It does, and every
production shower splitter has the **same shape — which is not the shape of §3–§4**:

- **ATLAS topological cluster splitting** (arXiv:1603.02934): inside a topo-cluster,
  cells are searched for local energy maxima above 500 MeV; those maxima **seed** a
  re-clustering that splits the parent. A cell neighbouring two maxima is **shared**.
  Splitting on the finely-segmented EMB1/EME1 strips is explicitly what improves
  π⁰→γγ photon separation.
- **CMS particle flow** (arXiv:1706.04965, arXiv:1401.8155): a seed is a cell above
  threshold *and above all its neighbours*; clusters grow topologically from seeds;
  overlapping deposits are shared **fractionally** through a Gaussian shower profile
  in an expectation-maximisation Gaussian-mixture step.
- **GARLIC** (arXiv:1203.0774, arXiv:0902.3042): seeds from hits in the *early*
  layers → "cores" → clusters grown outward. Two-particle separation is optimised at
  the **seeding** step.
- **Hough-transform photon reconstruction in imaging calorimeters**
  (arXiv:2508.20728, 2025): a photon is a "Hough axis" of ≥ 3 consecutive local
  maxima that **points back at the interaction point**; overlaps are resolved by
  sharing maxima and splitting energy with a two-exponential lateral profile.
- **Arbor** (arXiv:1403.4784): oriented connectors between hits until the ensemble
  is a *tree*; nearby showers separate because their branches separate.
- **CALICE two-shower separation** (arXiv:1802.00672) benchmarks exactly these three
  families (Pandora, GARLIC, Arbor) against each other.

> **The seed count IS the multiplicity decision. There is no separate trigger.**

§4 did the opposite: it ran a global 2-means that **always fires** and then hunted
for an external veto. That is why its null distribution was broad — a forced
2-means on a healthy shower still returns *some* angle, *some* balance, *some* gap.
Seeding makes the null sharp, because a healthy shower has one core.

Two further things the survey forces into the design:

- **Fractional assignment is the literature's answer to owner factor 4** (two γs
  connected). You do not draw a line through the overlap; you share it by profile.
  **We cannot do this** — see §10.1.
- **Deep learning is out.** Sparse-conv instance segmentation is the modern LArTPC
  answer and toolkit `CLAUDE.md` forbids new external dependencies outright
  (`clus` links only `WireCellAux`, `WCPQuickhull`, `WireCellMcs`). Recorded once so
  it is not rediscovered.

### 10.1 The hard constraint: the action space is SEGMENTS

**Geometry and charge are per point; membership is per segment. There is no
per-point shower assignment anywhere in the chain.**

- dump `segments[].points[]` → `x, y, z, dQ, dx, rr` at ~0.6 cm spacing
  (`PrDisplayDump.cxx:403-415,534`; `TrackFitting.cxx:8962` `low_dis_limit = 0.6 cm`)
- sidecar `showers{}.members[]` → `{seg, dQ, …}` (`prep_em_scan.py:403-407`)
- C++ agrees: `Shower::detach_member_set` takes a set of **segments**
  (`PRShower.cxx:640-700`), and it **refuses a set containing the start segment**,
  so the daughter that keeps the start segment is structurally the "kept" one and a
  3-way split is two peel calls.

Three consequences:

1. **Seed at point level, assign at segment level.** §13a is built this way.
2. **The literature's fractional assignment is unavailable.** Sharing a segment
   means splitting a fitted trajectory, which is far outside this scope. The borrow
   degrades to a hard assignment at segment boundaries. Mitigation: segments are
   already cut at kinks and vertices.
3. **The action space and the label space match exactly** — the hand marks
   (`em.marks_by_shower` = `{shower: {segment: in|out}}`) are also per-segment.
   **Verified, not assumed**: of the hand-marked positives, **0 require a
   sub-segment cut** (`pr137_trigger_bakeoff.py`, first block). A hand scan can
   never demand a cut the splitter cannot make.

### 10.2 The prototype has no ancestor — but three reusable primitives

`prototype_base` was searched. **There is no shower-splitting code anywhere**:
shower-level operations are merge-only (`examine_merge_showers` fuses two showers
whose 100 cm directions agree within 10°), and the `Separate_*` family is
cluster-level cosmic/blob separation on PCA and connectivity, not shower-aware. So
**M15 does not bite — we are inventing, not porting.** Three primitives are
directly relevant:

- **`ProtoSegment::is_shower_topology()`** (`ProtoSegment.cxx:319-450`) — per fit
  point, a local frame (tangent, drift×tangent) and the **RMS of associated points
  transverse to the trajectory**, with tuned LArTPC scales: "large spread" at
  **0.4 cm**, decisions at **0.7 / 0.8 / 1.0 cm**. An in-tree cross-check on §12's
  in-situ width fit.
- **`find_peak_point_indices()`** (`PR3DCluster_steiner.h:733`) — a **graph-local
  charge-maximum finder** (charge > 4000, sort descending, accept a peak only if no
  accepted peak is within its `nlevel` neighbourhood). That is the ATLAS/CMS seeding
  primitive, already written in this codebase's idiom — it just feeds Steiner
  terminals instead of splitting. §13a's acceptance follows this rule.
- **`WCP::WCShower::SC_proj_Hough()`** (`WCShower.cxx:84-142`) — dead legacy code
  computing a **charge-weighted angular RMS about a Hough axis**. Off the NeutrinoID
  path, so a reference not a reuse, but the prototype authors reached for the same
  statistic the 2025 Hough paper uses.

`clus` itself has **no** k-means, DBSCAN, density estimator, local-maximum finder,
or Molière/profile/RMS code. It has Eigen (`NeutrinoShowerClustering.cxx:6`),
boost::graph, and a hand-rolled PCA power iteration (`points_pca`, `:5894`). There
is **no nonlinear minimiser** (`MyFCN` was ported off Minuit to Eigen solvers), so a
two-shower Grindhammer–Peters profile fit would need a hand-rolled Gauss–Newton and
is ranked last on implementation cost. **Every feature below is scored on
implementation cost as well as power**: the offline bake-off may use numpy/scipy,
the shipped C++ may not.

### 10.3 Two defects in §4's own measurement, corrected

**(a) The MERGED/SINGLE population was built on a lossy join.**
`pr137_split_separation.py:69` builds the OFF owner map as
`{s: n for n, ms in mo.items() for s in ms}` — **last-writer-wins**, so a segment
held by two OFF showers loses one ancestor. **1.0 % of OFF segments (91 of 9454) are
shared**, and the effect is not proportional:

| join | MERGED | SINGLE |
|---|---|---|
| §4's, last-writer-wins | 34 | 356 |
| faithful, accumulate all | **44** | **346** |

**Ten real merges were labelled SINGLE**, which inflates §4's measured false-positive
rate and deflates its purity. Every number in §12–§14 uses the faithful join.

**(b) Gaps were approximations, not minima.** `[:200]` / `[:400]` truncation inside
the O(N²) min-distance loops of `pr137_split_feasibility.py:73-74,101` and
`pr137_split_separation.py:50-51`. The new library uses a cKDTree and reports exact
minima. (`pr137_split_convgap.py` did not truncate.)

*Not* a defect, checked and cleared: `main_vertex` carries `x/y/z` directly, so
§4's `mv.get('x',0.)` did not silently fall back to the origin.

## 11. The owner's three factors, made into features

> *"1. direction metric in theta-phi space 2. distance matters, compared to the
> nearby large EM shower, 3. size matter, compared to the nearby large EM shower"*

| family | owner factor | features |
|---|---|---|
| **D** direction | 1 | `n_seed`, `d2_over_d1`, **`valley`**, `seed_frac`, `seed_angle`, `bimodal_coef`, `valley_1d`, `dBIC`, `angle` |
| **S** size/distance | 2 + 3 | `w_ratio`, `r_ratio`, `q_ratio`, `w_at_r_ratio` (the owner's ratio form) and `w_pull`, `w_over_expected`, `sep_scaled`, `dr_parts` (the in-situ-null form) |
| **C** conversion | 3, and the answer to 4 | `vgap_min/max`, `void_min`, `dedx0/1`, `dedx15_min/max`, `n_2mip`, `dedx_ratio` |
| **T** topology | 4 | `gap_cm`, `gap_scaled`, `balance` |
| **X** diagnostic only | — | `m_pi0` |

Three design points:

**`valley` is the ATLAS ingredient §4 did not have.** Two angular maxima are not
enough — a bright patch inside *one* shower also makes a maximum. ATLAS's rule is
local-maxima-**with-a-valley**: the charge density must *dip* between them.
`valley` = minimum density along the great-circle arc between the two seeds,
divided by the weaker peak. §13 shows this is the discriminator.

**The owner's normalisation and mine are both built, and reported separately.**
The owner said "compared to the nearby large EM shower" — a per-event ratio against
the dominant object. That is implemented literally (`w_ratio`, `r_ratio`,
`q_ratio`, `w_at_r_ratio`). But **the fallback fires on 37 % of the population**
(§12): the candidate *is* the largest EM object and there is no distinct reference.
For those rows the in-situ null (`w_pull`) is the only available normalisation, and
the row is flagged. Without that flag the owner's stated normalisation would have
silently become the agent's in exactly the events that matter most.

**The π⁰ mass is demoted from trigger to diagnostic**, per the broad-scope decision
(the gate fires on *any* over-clustering, not only 2-γ). §13 confirms the demotion
with an independent number: `m_pi0` AUC = **0.500**, exactly chance.

Also recorded: **energy does not conserve across a split.** `kine_charge` credits a
2D charge cell to any shower within 0.6 cm with no cross-shower de-duplication
(`NeutrinoEnergyReco.cxx:48-145`), so E(A)+E(B) ≥ E(parent) in the overlap. The
dedup exists (`kine_charge_owned_scan`, `:397`) but is knob-gated and runs at
`:9413`. Any π⁰-mass or `q_extra` claim about a split must name its regime.

## 12. The in-situ null model — and owner factor 2 is confirmed

`scripts/pr137_null_model.py`, on the 346 SINGLE objects (q > 1e6, ≥ 3 segments).
Charge-weighted transverse RMS in depth bins along each object's own axis:

| depth (cm) | n | p10 | median | p90 |
|---|---|---|---|---|
| 0–10 | 179 | 0.39 | **1.53** | 7.36 |
| 20–30 | 182 | 1.00 | **2.84** | 9.42 |
| 40–50 | 156 | 1.04 | **3.87** | 10.34 |
| 60–70 | 123 | 1.82 | **5.98** | 17.96 |
| 80–90 | 85 | 2.24 | **6.99** | 17.04 |
| 100–110 | 49 | 3.27 | **8.11** | 18.29 |
| 120–130 | 34 | 2.86 | **9.95** | 16.61 |

**Linear fit: `w_single(r) = 3.575 + 0.0283·r` cm — the slope is positive, so the
owner's factor 2 is confirmed quantitatively**, not merely assumed: a single EM
shower's transverse RMS grows from ~1.5 cm at 5 cm to ~10 cm at 125 cm.

That fit is what makes every other number scale-free. The seeding bandwidth is
`σ_ang(r) = w_single(r)/r`, which **shrinks with depth** — so a compact,
late-converting second γ (owner factor 3) is not smoothed away by the kernel that
fits the near, wide one. **No PDG constant is used as a threshold**; LAr X₀ ≈ 14.0 cm,
R_M ≈ 10.0 cm and the 18 cm conversion length are quoted for scale only.

Whole-object SINGLE medians: transverse RMS **5.76 cm**, angular RMS **8.8°**, start
dQ/dx **1.09 MIP** (normalised by the event's own `dqdx_ref` electron plateau).
Owner-ratio medians: `w_ratio` 0.936, `r_ratio` 0.940, `q_ratio` 0.680 —
all near 1, so deviation is the signal. **Fallback fires 127/346 = 37 %.**

## 13. The bake-off

### 13a. Seeding, and the valley — `scripts/pr137_seed_split.py`

**Recovery** (does the kernel find the OFF partition on a known 2-way merge, with
k forced to 2 so the kernel is not scored on the trigger's failures):

| kernel | n | median purity | ≥ 0.90 | ≥ 0.99 |
|---|---|---|---|---|
| point-level seeded, profile σ, sep 1.6 | 29 | 0.734 | 8 | 3 |
| point-level seeded, flat σ = 4 cm, sep 1.0 | 33 | 0.682 | 10 | 7 |
| **§3's segment-level ray 2-means, recomputed on this population** | 33 | **0.825** | **15** | 7 |

**Point-level seeding is a worse KERNEL than §3's segment-level 2-means.** Recorded
plainly: the reframe improves the *trigger*, not the *split*. **Design consequence:
keep §3's 2-means as the kernel and use the seeded density only to decide whether
and into how many parts to cut.**

**Multiplicity as the trigger** — acceptance = (2nd peak ≥ `dratio`·1st) ∧
(valley ≤ `vmax`) ∧ (minor charge share ≥ `fmin`):

| accept rule | MERGED | SINGLE | enrichment | purity |
|---|---|---|---|---|
| d2/d1 ≥ 0.20, no valley cut | 26/44 (59 %) | 139/346 (40 %) | 1.5× | 16 % |
| d2/d1 ≥ 0.50, no valley cut | 15/44 (34 %) | 61/346 (18 %) | 1.9× | 20 % |
| **d2/d1 ≥ 0.35, valley ≤ 0.90, frac ≥ 0.05** | **12/44 (27 %)** | **8/346 (2 %)** | **11.8×** | **60 %** |
| d2/d1 ≥ 0.50, valley ≤ 0.80, frac ≥ 0.10 | 8/44 (18 %) | 8/346 (2 %) | 7.9× | 50 % |

**The valley is the whole effect.** Without it the best purity is 20 %; with it,
60 % at the same 27 % efficiency §4's best rule had at 36 %. §4's three families
were measuring angle, balance and gap — none of them asks whether the charge
actually *dips* between the two lobes.

### 13b. All features, two positive classes — `scripts/pr137_trigger_bakeoff.py`

- **class A (proxy)**: 44 MERGED vs 345 SINGLE, faithful join. Large, contaminated.
- **class B (labels)**: the pr/136 §10.1 hand marks, **strict node-id join**:
  **10 POS vs 49 NEG**. Real, at segment granularity — and small.

**Class B cannot decide anything today, and that is a measurement.** With
POS = 10, NEG = 49 the AUC standard error is 0.101, so the 2σ band is
**0.5 ± 0.20**. *Every* class-B AUC lies inside it. The strict join census is worth
publishing because it is much smaller than pr/136 §10.1's "29 pure-OVER + 31 BOTH"
suggests: of **112 OUT marks only 29 are current members**, giving **11 showers with
an actionable OUT and 10 splittable**. pr/136 §10.1's larger number comes from
`em117_score`'s charge-overlap matching rather than a node-id join — the same
strict-vs-expanded bracket pr/136's xclus census had to report. Part of the
shortfall is a real and welcome effect: the labels were taken on earlier arms and
the shipped pr/133+134 chain has since removed some marked-out charge
(`q_extra` 8.9 % → 7.0 %).

**No single feature is a trigger.** Best |AUC − 0.5| on class A is 0.154
(`seed_frac` 0.654; `valley` 0.364, i.e. inverted). Single-feature purity at 50 %
efficiency runs 11–22 %, all below §4's low-efficiency 27–36 %.

**Two-feature combinations carry it, and `valley` is in six of the top eight:**

| rule (≥ 6 merged fires) | merged | single | purity |
|---|---|---|---|
| `q_ratio ≥ 0.93 & valley ≤ 0.863` | 7 | 2 | **78 %** |
| `gap_scaled ≤ 2.51 & valley ≤ 0.863` | 7 | 3 | 70 % |
| **`d2_over_d1 ≥ 0.414 & valley ≤ 0.863`** | **11** | **8** | **58 %** |
| `balance ≥ 0.153 & valley ≤ 0.863` | 9 | 7 | 56 % |
| `seed_frac ≥ 0.193 & valley ≤ 0.863` | 11 | 9 | 55 % |
| `gap_scaled ≤ 1.51 & w_pull ≥ 3.06` | 6 | 6 | 50 % |

**The 3-feature scan does not beat the 2-feature one** (best 60 % on 6 fires). With
44 positives that is the resolution limit: **stop adding features, get labels.**

**Two negative results, stated because they were predictions:**

- **dE/dx died on the proxy.** `n_2mip` AUC 0.493, `dedx15_min` 0.456 — the
  "two 2-MIP conversion stubs" signature I expected to be the LArTPC edge is not
  there. §14's scan explains why, and **the sign was backwards** — see §14.2.
- **`m_pi0` AUC = 0.500 exactly**, an independent confirmation of §4b's death of the
  π⁰-mass gate on a completely different code path.

## 14. The curated set, and the agent scan

### 14.1 The set — `scripts/pr137_curate.py`

**172 objects**, stratified, seed 20260901, re-derivable:

| stratum | n | selection |
|---|---|---|
| S1 random control | 100 | uniform over the q > 1e6, ≥ 3-seg population — **drawn before any feature was consulted** |
| S2 known merges | 44 | every object with ≥ 2 OFF ancestors (faithful join) |
| S3 enriched | 40 | top by `valley` + `d2_over_d1`, drawn last |

(S1 ∩ S2 = 12.) An **owner calibration subset of 50** is marked `owner_scan=1`,
spread 25 / 15 / 10 across the strata so agreement is measured across the whole
range and not only on easy objects.

Contact sheets: `work/pr137_sheets/*.png`, four panels each — θ-φ ray map with the
angular maxima marked, width vs depth against the `w_single(r)` null, dE/dx vs
depth in MIP units, and the proposed 2-way split in side view.

**They are BLIND, and that is not decoration.** The proxy class is the very thing
these labels exist to validate; printing it on the sheet would let it steer the
judgement and the resulting agreement number would be circular. The blind sheet
carries event, node, charge and segment count and nothing else. The θ-φ panel is
drawn *raw*; only the side view shows the proposed partition, so the reader sees the
charge before seeing a hypothesis about how to cut it.

### 14.2 The agent scan — and §5's prediction is confirmed

15 objects scanned blind (`em_labels/splitscan-0901-agent/`), chosen as the
decisive set: the 8 objects that fire `d2_over_d1 ≥ 0.414 & valley ≤ 0.863` while
the proxy calls them SINGLE, 3 that fire and the proxy calls MERGED, and 4
non-fired random controls.

| object | proxy | agent verdict | evidence |
|---|---|---|---|
| 91917/17005 | SINGLE | **SPLIT2** | two clumps ~200 cm apart; w 90–122 cm vs null 3–11 |
| 318769/31026 | SINGLE | **SPLIT2** | two disjoint clumps, starts at 28–40 and 42–53 cm, both detached from the vertex |
| 278420/61027 | SINGLE | **SPLIT2** | a compact line at 25–50 cm plus scattered charge at 70–140 cm; w 11–19 vs null 4.5–7 |
| 415278/23012 | SINGLE | **SPLIT2** | two lobes; blue 25–90 cm, red 60–125 cm offset transversely |
| 294174/71067 | SINGLE | **SPLIT2** | 74 pts over 110–210 cm in several disjoint clumps |
| 389538/19021 | SINGLE | **KEEP** | false positive — see below |
| 170761/8026 | SINGLE | **KEEP** | coherent vertex-attached shower; the minor part is 2 isolated points |
| 396037/69026 | SINGLE | UNSURE | 48 pts, too sparse to call |
| 396222/9059 | MERGED | **SPLIT3** | 123 seg, w 14–40 vs null 4.5–7. Trigger right, **kernel fails** (balance 0.003) |
| 176502/109119 | MERGED | **SPLIT3** | three lobes in θ-φ. Trigger right, **kernel fails** |
| 21073/63100 | MERGED | **SPLIT2** | two clean lobes. Trigger right, **kernel right** |
| 256587, 292524, 76346, 392901 | SINGLE (not fired) | **KEEP** ×4 | one tight core each; width at or below the null |

**Five of the eight "false" fires are real over-clusters.** doc pr/137 §5 argued the
proxy purity is a *lower* bound because "SINGLE" is not truth; that is now measured
rather than argued:

| purity of `d2_over_d1 ≥ 0.414 & valley ≤ 0.863` | value |
|---|---|
| against the arm-difference proxy | 58 % (11/19) |
| **against agent hand labels, on the 11 scanned** | **80 % (8 SPLIT / 2 KEEP, 1 UNSURE excluded)** |
| extrapolated to all 19, if the 8 unscanned proxy-MERGED are genuine | 89 % |
| §4's best, for comparison | 27–36 % |

And **4/4 non-fired controls are correctly KEEP.**

**Three findings that only the scan could produce:**

1. **A new false-positive class, and it renames the dE/dx handle.** 389538/19021 is
   a V of two arms meeting at a common point ~215 cm out, with dE/dx **3–4 MIP at
   that shared origin** falling to ~1.5 — **one photon whose e⁺e⁻ pair is resolved**,
   not two photons. §11's Family C looked for *two* 2-MIP stubs as evidence of two
   γs; the more discriminating signature is **one 2-MIP stub at a shared origin as
   evidence of ONE γ**, i.e. a *veto*, not a trigger. That is why `n_2mip` scored at
   chance in §13b — the sign was backwards.
2. **The kernel fails exactly where the trigger is most confident.** On the two
   largest fired objects (396222, 176502) the 2-means returns a degenerate partition
   (balance 0.003) while the θ-φ map plainly shows three lobes. **k must come from
   the seed count, not be fixed at 2** — §1.2b's "k = 3 only on a residual test" is
   confirmed as necessary, and it is needed on the biggest objects, not the rare ones.
3. **Width-vs-depth is the most legible panel for a human.** Every SPLIT case sits
   2–10× above `w_single(r)`; every KEEP control sits at or below it. It did not win
   the AUC ranking, but it is what makes a verdict fast — worth keeping in the
   owner's Bee package as an annotation.

### 14.3 What this round did NOT establish

- **The 80 % is 11 objects labelled by one scanner.** It is not a purity measurement;
  it is a demonstration that the proxy understates. The owner's 50 are what turn it
  into one, and the **agent-vs-owner agreement on the overlap is the noise floor**
  no trigger may be claimed to beat.
- **Efficiency is unmeasured.** The rule fires on 27 % of proxy-MERGED. How much
  real over-clustering it misses needs the S1 control stratum labelled, not just
  spot-checked — the 15 known merges that do *not* fire are the first thing to scan.
- **Nothing is implemented.** No C++, no knob, no arm, no gate. The insert point
  (`NeutrinoShowerClustering.cxx:8213`, after `examine_showers`), the write recipe
  (fork `pass4_prune_detached`, `:8591-8726`) and the kinematics-refresh obligation
  (there is **no** free recompute downstream — `calc_kine_2` runs at `:8202`, before)
  are recorded in §10.1 and §11 so the implementation round does not re-derive them.

### 14.4 Revised plan

1. **Owner scans the 50** marked `owner_scan=1` in `docs/pr/pr137-curated-set.tsv`
   (Bee package `bee/pr137r2/`, built and held). Verdict **SPLIT / KEEP plus the
   boundary** — which segments go to which part.
2. **Agent scans the remaining ~120** blind sheets, same schema.
3. Report **agent-vs-owner agreement** on the 50 overlap *before* any trigger claim.
4. **Refit the trigger on labels, not the proxy**, with the feature set
   pre-registered (`valley` + one of `d2_over_d1` / `q_ratio` / `gap_scaled`) and a
   50/50 event-hash holdout opened once. With ~170 labels the holdout half holds
   only ~15–25 positives, so a pre-registered 2-feature rule is the honest ceiling —
   not a fitted classifier.
5. **Then** stage 1 of §6: `WCT_SHOWER_SPLIT_DEBUG`, byte-neutral, hash-gated —
   now emitting `valley` and the seed list, which §4's design did not know to.
6. Add the **one-γ veto** of §14.2-1 to the stage-1 tape: shared-origin dE/dx and
   the common-point test.
7. Independently: **adjudicate 181050** (doc pr/136 §11.9), still the last open item
   on `onV1c90d25`.

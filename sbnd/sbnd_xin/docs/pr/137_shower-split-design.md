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

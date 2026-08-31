# doc pr/139 — π⁰ reconstruction after the splitter: the plan

**Status: PLAN. No code, no arm, no knob yet.** The SBND production baseline moved
on 2026-08-31 to **`onV1c90` + `shower_split`** on the owner's word (*"lets use
'onV1c90 + splitter' as the new baseline for SBND production"*). This file is the
next round, and it is deliberately short: the owner asked to **"only worry about
the parts that may have a good chance to have improvements"**, so every front
below carries the specific events it would recover and the measurement that says
it can. Everything measured dead is in §3, named, so nobody re-opens it.

Prior art this rests on: `138_shower-split-master-plan.md` §3–§5 (the flip
decision, the clean-stratum result, and the two features measured dead),
`136_em-clustering-charge-attribution-charter.md` §11.2 (the instruments),
`132`–`135` (the π⁰ campaign).

## 0. The brief, in one paragraph

The splitter now ships. It lifted the π⁰ census from 32 to **35 of 66** — the
best of the campaign — by cutting apart over-clustered γ pairs so the finder can
pair them. It also **broke four π⁰s that used to work**. Those four are
adjudicated in §1 and they split cleanly into two causes with two different
fixes, both cheap. Recovering all four would put the census at **39 of 66**, and
that is the number this round is for. The owner has ruled the upstream ν vertex
**out of scope** — *"we do not want to change the upstream nu vertex; the only
thing that we can touch is the neutral pi0, which move the vertex"* — so §2.4 is
the only vertex work here.

## 1. What the flip bought, and the four it broke

Census exact **33 → 35 of 66** at the new baseline, with eight events changing
class. Each one traced to the fire that caused it (`work-pr138r2-c90{off,on}-*`):

| event | π⁰ census | the fire | owner's split label | |
|---|---|---|---|---|
| 280972 | no-group → **exact** | node 79136 | SPLIT2 | ✅ correct cut |
| 56243 | no-group → **exact** | node 69032 | *not in the scan set* | ✅ |
| 314838 | partial → **exact** | node 110088 | SPLIT2 | ✅ correct cut |
| 269774 | no-group → partial | node 13237 | SPLIT2 | ✅ correct cut |
| **281485** | partial → **none** | node 89095 | **SPLIT2** | ❌ **a correct cut broke the pair** |
| **396222** | partial → **none** | node 9059 | **SPLIT4+** | ❌ correct cut, in the busy event the owner discounted |
| **165157** | partial → no-group | node 9000 | **KEEP** | ❌ **false fire** |
| **54332** | exact → partial | node 122091 | **KEEP** | ❌ **false fire** |

**Two causes, and they need different fixes.**

- **Two are false fires** — the trigger cut a shower the owner says is one object,
  and halving a γ destroyed the pair (165157: γ₂ 187.9 → 94.4 MeV, mass 152 →
  108; 54332: γ₁ 185.9 → 129.8, mass 109 → 91). Fixed by not firing there (§2.3).
- **Two are correct cuts** whose π⁰ pair broke anyway — the split was right by the
  hand labels, and the finder then paired on the reduced γ energy. Fixed only by
  making the pairing aware that the parts were once one object (§2.1).

**Also true and not to be forgotten:** three of the four gains are correct cuts.
The splitter is doing its job; this round is about the tail.

## 2. The fronts, ranked by measured headroom

### 2.1 Split-aware π⁰ pairing  ← the highest expected yield, and it costs no efficiency

**What it recovers:** 281485 and 396222 directly (+2 census), and it is the only
fix for that class. **Why it is likely to work:** the mechanism is already
measured — of the eight hand π⁰ pairs the splitter moved, **six went further from
135 MeV**, and the movement is always the same direction: a split γ loses charge
to its sibling and the finder pairs on the reduced value.

**The change.** The splitter runs immediately before `id_pi0_backproject_vertex`
/ `id_pi0_with_vertex` / `id_pi0_without_vertex`. Let a peeled daughter and the
parent it came from be offered to the finder **as a single candidate γ as well as
separately**, and keep whichever pairing scores better on the existing mass
window. Two implementations, cheapest first:

1. **Record the provenance** — a `split_parent` id on each daughter — and let the
   pair loop consider `(parent-as-one, other γ)` alongside `(daughter, other γ)`.
   No re-run of the finder, no new geometry.
2. If that is not enough, run the pairing over the **pre-split** shower set as
   well and take the better-scoring result. Expensive but decisive.

**Success criterion, pre-registered:** census exact **≥ 36** of 66 on the standard
239-event manifest, with **281485 the event that must move** — 396222 is the other
correct-cut loss and the owner has said he is not sure it is a useful event, so a
criterion that needs it to cooperate is one we would only argue about later, with `q_extra` not rising above the current 6.7 % and 0
ADVERSE movers. Knob `pi0_split_aware_pairing`, DEFAULT OFF.

**Risk, stated:** re-admitting the un-split parent may recreate the very
over-clustered γ the splitter removed, in which case the census gain and the
`q_extra` floor fight each other. That is exactly what the criterion above
measures, and it may come back dead.

### 2.2 The daughter's particle type — a defect the sizing found, and it is LIVE

**This one was not in any plan and it is the strongest item here, because it is a
defect in what shipped today rather than a missing feature.**

Sizing the re-home front (`scripts/pr139_daughter_fate.py`) turned up that **11
of the 50 peeled daughters come out typed μ (pdg 13)** and one as a proton — 24 %
of everything the splitter produces is not EM. A μ-typed shower is invisible to
the π⁰ finder except through K20. So the question is whether the split is
*isolating* genuine muon contamination (good — the splitter is doing more than
the census shows) or *manufacturing* a mis-type (bad, and live in production).

**Answered with a dump read, and it is the second.** The **segment-level** PDGs
are byte-identical between the knob-off and knob-on arms — the split retypes
nothing. What changes is the *shower-level* `particle_id`, and SBND runs
`shower_pdg_from_start_segment = true`, so a daughter's type is read off **its
start segment alone**:

| event | daughter | kine | its segments' PDGs | shower PDG |
|---|---|---|---|---|
| 84229 | 9058 | **476.4 MeV** | {13: 16, **11: 17**, 2212: 1} | **13** |
| 269774 | 31043 | **181.6 MeV** | {**11: 10**, 13: 3, 2212: 3} | **13** |
| 281781 | 14007 | 87.7 MeV | {**11: 4**, 13: 1} | **13** |
| 292643 | 112035 | 31.7 MeV | {**11: 3**, 13: 1} | **13** |
| 396037 / 318769 / 281165 | | 24–30 MeV | EM/μ **tied** | 13 |
| 415278 / 284200 / 71642 / 281165 | | 1.5–62 MeV | μ-majority | 13 — correct |

**Four daughters carrying an EM-majority segment composition, one of them
476 MeV and another 182 MeV, are typed as muons and drop out of π⁰
consideration.** Three more are ties. Only four of the eleven are honestly
muonic.

**The cause is mine.** doc pr/138 §B3 chose the daughter's start segment as *the
member nearest the reference vertex* — a choice made to keep `init_dir` pointing
forward for the π⁰ finders, and it does that (116 peels, zero backwards). It
takes **no account of PDG**, so when the member nearest the vertex happens to be
a μ-typed segment the whole daughter is typed μ.

**The fix is small and there are two forms; measure both.**

1. **Prefer an EM-typed member as the daughter's start**, falling back to the
   nearest-the-vertex rule when the daughter has no EM member. Keeps the forward
   check, changes only the tie-break.
2. **Type the daughter from its composition** rather than its start segment — a
   charge-weighted plurality over its members. Closer to what
   `shower_pdg_from_shower_type` already does for other showers, but it touches a
   production typing path and so is the more invasive of the two.

**Success criterion, pre-registered:** the μ-typed fraction of daughters falls
from 11/50 to ≤ 4/50 (the honestly-muonic ones), census exact does not fall below
35, and `q_extra` does not rise above 6.7 %. Knob `shower_split_em_start`,
DEFAULT OFF.

**Why this ranks above the re-home:** it is a defect, not an enhancement; it is
~40 lines; and it puts **658 MeV of EM charge in two objects** back in front of
the π⁰ finder without changing a single membership decision.

### 2.2b Re-home the daughter (doc pr/138 §B4) — narrowed by the sizing

The owner's requirement stands (five scan comments: *"should be part of the
earlier EM shower cluster"*). What was never measured is the **size**, so it is
measured here first:

| | n | EM-typed | **paired into a π⁰** | median kine |
|---|---|---|---|---|
| the peeled **daughters** | 50 | 38 | **9 (18 %)** | 28.5 MeV |
| their **parents**, post-cut | 50 | 49 | 19 (38 %) | 111.7 MeV |
| every other EM shower, same events | 428 | 428 | 26 (**6 %**) | 2.2 MeV |

**A correction worth recording, because the first answer was the opposite.** An
initial join by the daughter's new start-segment id reported *"0 of 51 daughters
are used by the π⁰ finder"* and would have made this the headline. Joining the
other way — iterate the dump's `showers[]` and ask which are daughters — says
they pair at **three times the base rate**. The finder is **not** blind to them.
The first number was a lookup bug, retracted here rather than quietly dropped.

So the front is narrower: **10 EM-typed daughters above 20 MeV that did not
pair** (13 above 10 MeV, 9 above 30). Two are large — evt176502 node 148258 at
**570.7 MeV** and evt415278 node 56091 at **505.3 MeV**.

**And it is DOWNSTREAM of §2.2 and §2.3.** Fixing the μ typing changes which
daughters are even candidates; §2.3's veto, if it goes on, removes two of these
ten (165157 at 67.8 MeV and 54332 at 57.2 MeV are the false fires' daughters).
**Re-size it after those two land** — building it against today's ten would size
it against a population that will not exist.

Change, when its turn comes: after a peel, offer the daughter to the nearest EM
shower under the existing absorb predicates (`samevtx_absorb` `:9270`,
`satellite_absorb` `:9386` — **fork, do not extract**, M10). Knob
`shower_split_rehome`, DEFAULT OFF. Criterion: `q_extra` ≤ 6.7 % and census ≥ 35.

### 2.3 The false fires — a priced knob, not a research front

**What it recovers:** 165157 and 54332 (+2 census). **Why it is a knob and not
research:** doc pr/138 §4.2b measured the obvious separator dead — `void_frac`
came back at **AUC 0.146, backwards**, because *seen from a wrong origin one
shower genuinely looks like two well-separated ones*. This round adds the second
candidate, pr/129's pointing test (does the object's own axis aim back at the ν
vertex — impact parameter `b`), and it is **also not a separator**:

| feature | median, correct cuts | median, false fires | AUC (false > correct) |
|---|---|---|---|
| `vgap` (distance along the ray) | 14.4 cm | 38.9 cm | 0.735 |
| **`b`** (perpendicular miss) | 13.1 cm | 39.7 cm | **0.777** |
| miss angle | 20.9° | 52.2° | 0.742 |

Better than `vgap`, but only just — both are really measuring *"this object is
far from the vertex"*, which is **also true of 17 of the 33 correct cuts** (a γ
converts a mean ≈18 cm out). So it is a dial with a price:

| bound | fires | right | wrong | efficiency | purity |
|---|---|---|---|---|---|
| `b ≤ 10 cm` | 13 | 13 | 0 | 0.302 | 1.000 |
| `b ≤ 15 cm` | 20 | 19 | 1 | 0.442 | 0.950 |
| `b ≤ 30 cm` | 28 | 25 | 3 | 0.581 | 0.893 |
| **no bound (today)** | 41 | 33 | 8 | **0.767** | **0.805** |

**The proposal is a measurement, not a threshold.** `b` is the better of the two
(AUC 0.777 vs 0.735) and is the one to ship, so the arm prices the feature the
table above prices — doc pr/138 §4.3's `vgap` dial is kept only for comparison.
Ship `shower_split_max_impact` (DEFAULT 0 = off) and run **one arm at
`b ≤ 15 cm`**. The
question it answers is arithmetic, not judgement: the 33 correct cuts buy 3
census gains and the 8 false fires cost 2, so a bound that keeps 19 of 33 correct
cuts and removes 7 of 8 false fires is **probably net positive on the census** —
and the arm settles it in one run. If it is not, the knob stays off and we have
lost one arm.

### 2.4 Should the π⁰ vertex re-seat run BEFORE the splitter?  ← the only vertex lever in scope

The owner has ruled the upstream ν vertex out of scope and named the exception:
*"the only thing that we can touch is the neutral pi0, which move the vertex."*
That is exactly `id_pi0_backproject_vertex` (`:6241`, K21, production ON) and
`id_pi0_without_vertex`'s path-2 hack (`:7886`), both of which re-seat
`main_vertex` — and **both run AFTER the splitter**, so the splitter measures
every feature from the *uncorrected* point.

Measured in doc pr/138 §B1: the vertex moves on **5 of 172** scanned objects,
by **60.16 cm** on evt76346 and **14.50 cm** on evt396222 — and evt396222 is one
of the four π⁰s §1 says broke.

**The change:** move the back-projection ahead of the splitter, or give the
splitter a second pass on the objects whose vertex later moved. One ordering
change, one full gate.

**Expected yield, stated honestly: small — 5 objects of 172, of which one is in
the broken-π⁰ list.** It is in the plan because it is cheap, because it is the
only vertex work the owner has allowed, and because the ordering is arguably
wrong on its own terms regardless of yield. **It is not a reason to run a
campaign.**

### 2.5 The joint label set — the enabler, and the owner has offered to scan

Doc pr/138 §3.3 is the binding constraint on measuring **any** of §2.1–§2.4:
`pr136_completeness.py`'s target comes from the 2026-08-27/28 attribution scan,
which called several of these objects one shower, while the 2026-09-01 split scan
says three to five. **93–94 % of the measured `q_miss` rise is that conflict, not
a real cost**, and it runs the other way too (28 % of the `q_extra` gain sits on
objects the split scan calls KEEP).

**The ask is small and specific** (§4): re-mark, in the split display, the
**~15 hand-marked showers the splitter actually touches**, so `target` is defined
per *part*. Not a re-scan of the 90 — fifteen objects.

## 3. What we are NOT doing, and the measurement that closed it

| | why it is closed |
|---|---|
| another feature measured **from** the ν vertex | `void_frac` AUC **0.146** (backwards), `b` 0.777, `vgap` 0.735 — the information is not there (§2.3, doc pr/138 §4.2b) |
| retuning `valley_best ≤ 0.95` | holdout spent (doc pr/138 §A5.4); fires are 80 % right overall and **100 % right on a vertex-attached object** |
| **k ≥ 3 splitting** | missed its pre-registered gate (0.635 → 0.756 vs ≥ 0.85) and is capped by `max_seeds = 4`, which cannot move without moving the trigger. Deferred per the owner's "only good-chance parts" |
| the **no-valley** class | 7 of 9 missed splits have `valley_best = 1.000` — no charge dip exists because the γs overlap. Needs a different observable; not this round |
| chasing the raw `+4.3 pt` of `q_miss` | 93 % of it is §2.5's label-epoch artefact |
| touching the upstream ν vertex finder | **owner, explicitly out of scope** |

## 4. The scan asks — small, specific, and the owner has offered

1. **The four broken π⁰s** (281485, 396222, 165157, 54332) in the split display,
   one question each: *is the cut right, and should the two parts still be one γ
   for the π⁰?* Those two answers are different and §2.1 needs both. **4 objects.**
2. **The ~15 hand-marked showers the splitter touches** (§2.5), re-marked per
   part. **15 objects.**
3. **evt314838** — the adjudication item from doc pr/138 §3.5, where three of the
   owner's own instruments give three answers (split scan SPLIT2 high-confidence;
   attribution scan purity 0.715 → **1.000**; hand π⁰ needs the charge). One call
   settles which instrument leads when they conflict. **1 object.**

Twenty objects total. The tool is `split_display/serve_split_display.sh`, and a
fresh tag (M13) — `splitscan-0902-pi0`.

## 5. Sequencing, and the bar

The order is fixed by dependency, not by expected size:

1. **§2.5's 20-object scan** — without it §2.1 and §2.2b cannot be *measured*,
   only asserted. The owner has offered; it is the cheapest thing here.
2. **§2.2 the μ-typing fix** — a defect, ~40 lines, live in production, and it
   changes which daughters are even π⁰ candidates. Everything downstream is sized
   against a population it moves, so it goes first among the code changes.
3. **§2.3's one arm** at the chosen bound — pure arithmetic and independent of
   the rest, but it must land **before** §2.2b, because if the veto goes on then
   two of §2.2b's ten unpaired daughters (165157, 54332) never exist.
4. **§2.1 split-aware pairing** — the largest single idea, and the one most
   likely to come back dead; run it once the population above has settled.
5. **§2.2b re-home**, re-sized against what §2.2 and §2.3 leave behind.
6. **§2.4** last: 5 objects of 172, one of them the event the owner discounted.

Every stage holds the doc pr/138 bar: knob-off byte-identical on the standard
239-event manifest (478 archives), freshness proof (M1), `wcdoctest-clus` green
with the new defaults pinned in `doctest_clus_knob_defaults.cxx`, compiled-config
proof from the arm's own `.wct-cfg-evt*.json`, and 0 ADVERSE movers. Each knob is
measured **alone** before being combined with another.

**The baseline everything is measured against is the new one** —
`work-pr138r2-c90on-*`, which the post-flip config reproduces byte-for-byte
(flip-equivalence gate PASS, 132 + 212 + 38 + 96 = **478 / 478**,
`work-pr138r3-flipchk-<s>`): `q_miss` 16.7 %, `q_extra` 6.7 %, census **35 / 66**,
19 / 49 impossible pairs, median `q_f1` 0.918.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# section 1 -- the eight moved census events, attributed to their fires
python3 scripts/pr138_flip_analyze.py --png     # -> pr138-flip-decision.{tsv,png}

# section 2.2 -- what happens to every peeled daughter (the re-home sizing)
python3 scripts/pr139_daughter_fate.py          # -> docs/pr/pr139-daughter-fate.tsv
# section 2.3 -- the pointing test (impact parameter b) on the 41 fires
python3 scripts/pr139_pointing.py               # -> docs/pr/pr139-pointing.tsv

# the baseline these are measured against
python3 scripts/pr132_pi0_census.py --manifest98 em117-138c90on98-manifest.tsv \
    --manifest141 em114c-138c90on141-manifest.tsv --fudge 0.86 \
    --overlay-tag pi0scan-0829-agent
```

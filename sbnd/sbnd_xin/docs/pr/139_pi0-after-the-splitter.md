# doc pr/139 — π⁰ reconstruction after the splitter: the round plan **and tracker**

**Status: LIVING.** **`shower_split_em_start` is SBND PRODUCTION ON as of
2026-08-31** (owner flip; flip-equivalence gate `work-pr139r3-flipchk-*` vs
`work-pr139r1-onemst-*` **478 / 478 byte-identical**). The owner's scan is **DONE** (2026-09-01,
39 objects, tag `splitscan-0902-pi0`) and **it overturned this round's own
recommendation** — see **§6**. The four items §3ter ordered are all **now
measured**: item 1 **passes its pre-registered prediction and awaits the owner's
flip** (§8, §10), item 2 **shipped** (§11), items 3 and 4 are **closed as
measured-dead, each with a mechanism** (§12, §13). **§14 is what to read next.**
This file is the tracker for a multi-session round. §1 is the
driver: it carries one row per item, its knob, its state and — once it exists —
its gate label and result. Prose below explains rows; **the table is what the
next session reads first.**

The SBND production baseline moved on 2026-08-31 to **`onV1c90` + `shower_split`**
on the owner's word (*"lets use 'onV1c90 + splitter' as the new baseline for SBND
production"*), toolkit `1c29a2a1`, flip-equivalence gate **478 / 478**.

Owner's scope ruling for the whole round, verbatim:

> *"for nu vertex, we do not want to change the upstream nu vertex. The only thing
> that we can touch is the neutral pi0, which move the vertex."*

Owner's ordering, verbatim (2026-08-31):

> *"I think we should first fix the four events … After that I would be happy to do
> the scan, probably in the port 5022 … After that I think that we should evaluate
> the following. 1. The ν vertex … 2. A joint label set … 3. B4 re-home … 4.
> Split-aware π⁰ pairing."*
> *"I do not like the orphan things to be identified as muon, which will be biased
> the energy a lot due to muon mass. We should re-sit them into the nearby showers."*

Prior art: `138_shower-split-master-plan.md` §3–§5, `136_…charter.md` §11.2,
`132`–`135` (the π⁰ campaign).

---

## 1. Status table — the driver

| # | item | knob (all DEFAULT OFF) | state | gate / arm label | result |
|---|---|---|---|---|---|
| **P1.0** | **the shared knob-off gate** | all four at their shipped defaults | **PASS ×2** | `work-pr139r2-off-*` vs `work-pr138r3-flipchk-*`, **and** vs `work-pr139r1-off-*` | **478 / 478 byte-identical** on both, `missing/unpaired 0`, rc=0 ×4 ×2 |
| **P1.1** | shared-membership peel guard | `shower_split_skip_shared` | **SUPERSEDED BY P1.6** (ships inside it) | `work-pr139r1-onshared-*` | 3 peels refused, the **`kine=0` daughter gone**, `none` 3→2, exact **35**, q_extra 6.7→6.9 %, 0 ADVERSE |
| **P1.2** | impact-parameter veto @ 12 cm | `shower_split_max_impact` (cm, 0 = off) | **MEASURED WRONG AT 12 — do not flip** (§6) | `work-pr139r1-onb12-*` | exact 35 → 36 and all four π⁰s recovered, **but the owner's scan says it suppresses 9 of the 19 cuts he confirms**, including 281485 whose cut he calls correct |
| **P1.6** | **`skip_shared` + `max_impact` = 30** — the operating point the scan supports | `shower_split_skip_shared` + `shower_split_max_impact` | **PASSES all four pre-registered bars; OWNER'S CALL, and it is a TRADE** (§8, §10, §14.2) | `work-pr140r1-on-*` vs `work-pr139r1-onemst-*` | **For:** exact **36**, q_extra **6.9 %**, **0 ADVERSE**, tape reproduces the prediction on **39 of 39**; **5** confirmed cuts suppressed (2 deferred + 3 rejected) vs **11** for the withdrawn `b ≤ 12`. **Against:** the §8.4 bars were fixed *before* item 2's instrument existed, and that instrument (§11.4) prices the arm at **−1 confirmed cut** (294174/71067) and **−0.052** median part `q_f1`. `skip_shared` **alone** is the defensible alternative: 2 suppressions instead of 5, at one census `exact` |
| **P1.3** | daughter EM start segment (μ-typing) | `shower_split_em_start` | **SBND PRODUCTION ON** (owner flip 2026-08-31) | `work-pr139r1-onemst-*`; flip-equivalence `work-pr139r3-flipchk-*` **478 / 478** | μ-typed daughters **11 → 2**, **461 MeV** of EM energy restored (×1.657 confirmed), 51 peels / **0 backwards**, every instrument unchanged, 0 ADVERSE |
| **P1.4** | re-home the orphan daughter | `shower_split_rehome`, `…_rehome_gap` | **CLOSED — do not flip** (§12) | `work-pr140r1-onrh15-*` (15 cm, flipped config) | 11/51 re-homed; census **35**, 0 ADVERSE; the merged target moves on **exactly the same 3 rows** as the single one — it buys sensitivity to *cuts*, not *merges*, so P1.4 was never blocked on the instrument |
| **P1.5** | the combination P1.1 + P1.2 + P1.3 | three knobs | **DONE** | `work-pr139r2-oncomb-*` | exact **36**, q_miss 14.5 %, q_extra 7.6 %, q_f1 **0.932**, μ-typed daughters **1**, 0 ADVERSE |
| **P2** | owner scan, split display port 5022 | — (tag `splitscan-0902-pi0`) | **DONE 2026-09-01 — 39 objects** | `em_labels/splitscan-0902-pi0/`, `docs/pr/pr139-scan-verdicts.tsv` | 20 KEEP / 19 SPLIT; trigger **eff 1.000** / pur 0.792; boundary SPLIT2 median **1.000**; **it overturned P1.2** (§6) |
| **P3.1** | π⁰ re-seat BEFORE the splitter | `pi0_reseat_before_split` | NOT STARTED | — | — |
| **P3.1b** | scope dial `max_vgap` (comparison arm only) | `shower_split_max_vgap` | NOT STARTED | — | superseded by P1.2 unless P1.2 fails |
| **P3.1c** | a pointing test **not** measured from the vertex | — | NOT STARTED | — | the one unexplored feature family |
| **P3.2** | joint label set + the per-part completeness target | — | **DONE** (§11) | `em_display/em140_score.py`, `docs/pr/pr140-perpart-*.tsv` | injective matching, denominator preserved to 4 figures; metric change ALONE on one arm: `q_miss` 16.7→**11.2 %**, `q_extra` 6.7→**8.6 %**; new number: **hand parts with no distinct reco object = 6** on the baseline |
| **P3.3** | re-home, **re-sized** after P1.3/P1.4 land | `shower_split_rehome` (same knob) | **CLOSED with P1.4** (§12.3) | — | widening past 15 cm is a search, and §12.2 says the metric would not reward it |
| **P3.4** | split-aware π⁰ pairing | `pi0_split_aware_pairing` | NOT STARTED | — | **no longer the fix for 281485** — see §2 |
| **C1** | k ≥ 3 splitting | `shower_split_max_parts` | **MEASURED DEAD at the cap** (§13) — reopened as a *kernel* question | `work-pr140r1-onk3-*` | lifting the cap to 3 moves 3 objects: 1 up, 2 down (one of them a SPLIT2); k≥3 mean 0.800 → **0.771**, census 35 → **34**. The cap was hiding that the kernel cannot place a third boundary, not causing it |
| **C2** | any feature measured **from** the ν vertex | — | **MEASURED DEAD** | doc 138 §4.2b, §5 below | `void_frac` AUC 0.146 (backwards) |
| **C3** | the no-valley / overlapping-γ class | — | **SCOPED OUT** | doc 138 §B7 | 7 of 9 misses have `valley_best = 1.000` |
| **C4** | the upstream ν vertex finder | — | **OWNER, OUT OF SCOPE** | — | — |

**Standing bar for every P-row**: knob-off byte-identical on the standard
239-event manifest (478 archives, `missing/unpaired events: 0` quoted), freshness
proof (M1), `wcdoctest-clus` green with the new defaults pinned, compiled-config
proof from the arm's own `.wct-cfg-evt*.json`, 0 ADVERSE movers. **Each knob is
measured alone before any combination** — doc pr/138 §B4 left re-home out of the
splitter's own gate for exactly this reason, and the owner grouping P1.1–P1.4 into
one ask is a *priority* grouping, not permission to collapse the measurement.

---

## 2. Correction — what the four broken π⁰s actually are

**This section supersedes the 2026-08-31 §1 of this file.** That version said
281485 and 396222 were "correct cuts whose pair broke because the finder paired on
the reduced γ energy", and that **split-aware pairing was the only fix for them**.
Reading the arm dumps event by event says otherwise, and the corrected picture is
both different and cheaper. The superseded claim is left visible here rather than
quietly overwritten.

Arms: `work-pr138r2-c90{off,on}-*`. Note the join hazard the first pass fell into:
the dump's `showers[].id` is the **start-segment display id**, and a peeled
daughter can be seeded on a segment that already roots another shower — in 165157
it is, so an `id`-keyed comparison silently merges two distinct objects into one
row. **Every table below is keyed on `shower_id`.**

| event | census | what actually happened | the fix |
|---|---|---|---|
| **281485** | partial → **none** | the peel produced a **`kine_charge = 0.00` daughter** (shower_id 19, 4 segments). Those 4 segments were **also members of shower 91112**, which gained 25.83 → 63.94 — exactly what the parent lost. A π⁰ group still forms in ON (91112 + 87078, mass 122.1); the census scores "none" because the hand π⁰'s charge is in the parent's original members. Not a pairing failure — a **shared-membership peel**. | **P1.1** |
| **165157** | partial → no-group | a **false fire** (owner: KEEP) **and** a shared-membership peel: the daughter's start segment `58027` already roots shower_id 4, whose `kine_charge` moves 0.00 → 23.87. Also drags shower_id 18 from pdg 11 to **2212** — a knock-on retype on a shower the splitter never touched. | **P1.1** and/or **P1.2** |
| **54332** | **exact** → partial | a **false fire** (owner: KEEP). The π⁰ changed partner: OFF 122091 + 27025 → 117.5 MeV; ON 122091 (187.06 → 129.84) + 128111 → 110.6 MeV. **The only one of the four that costs an *exact*.** | **P1.2** |
| **396222** | partial → **none** | **written off, and the reason is stated.** The OFF "exact" came from a **2879 MeV** 123-segment blob (node 9059) landing at 135.7 MeV. That is a coincidence, not a reconstruction worth defending; the owner already discounted the event (*"I am not sure if this event is really useful for our purpose"*, the only low-confidence label in 172). ON is arguably the more honest answer. **Build nothing for it.** | — |

**The arithmetic this forces, and it is smaller than the 2026-08-31 version
promised.** Census counts **exact**. Of the four losses only **54332** was exact
in OFF, so recovering all four returns **+1 exact (35 → 36)**; the other three
return to *partial*. Any claim of "+4" would be counting the wrong metric. The
other three still matter — they are three fewer events where the splitter
degrades a π⁰ — but they move the partial/none classes, not the headline.

### 2.1 The impact parameter separates all eight movers — and that is a warning as much as a result

`b` = perpendicular miss of the object's own charge-weighted principal axis from
the reference vertex (`scripts/pr139_pointing.py`, extended to the movers):

| event | node | class | **b (cm)** | vgap (cm) |
|---|---|---|---|---|
| 269774 | 13237 | gain (partial) | **1.67** | 27.39 |
| 314838 | 110088 | **gain, exact** | **8.32** | 12.36 |
| 280972 | 79136 | **gain, exact** | **9.03** | 18.05 |
| 56243 | 69032 | **gain, exact** | **10.92** | 10.16 |
| 165157 | 9000 | loss (false fire) | 13.05 | 13.38 |
| 281485 | 89095 | loss (shared-membership) | 23.67 | 20.03 |
| 396222 | 9059 | loss (written off) | 29.23 | 14.50 |
| 54332 | 122091 | loss (false fire) | 39.72 | 39.57 |

Every gain is below 11 cm, every loss above 13. **A bound at `b ≤ 12` separates
8 of 8 — which is exactly why it must not be reported as a discovery.** Eight
points, and the bound was chosen after seeing them. What makes it worth one arm
is that an *independent* 43-object label set already put purity at **1.000 at
`b ≤ 10`** and **0.950 at `b ≤ 15`** (doc pr/138 §2.3 table), so `12` sits inside
a window that population supports; it is not a spike found in noise. What that
same table also says is the price: **efficiency falls 0.767 → ~0.40**, so more
than half the fires stop happening, and the splitter's `q_extra` gain is bought by
*all* the fires, not by the four movers. **That trade is what the arm measures.**

---

## 3. Phase 1 — the four events and the orphan  *(the owner's item 1)*

Four knobs, four separate census reads, one shared OFF gate.

### 3.1 P1.1 — the shared-membership peel guard  `shower_split_skip_shared`

**The defect.** WCT showers may share member segments. When the splitter peels a
component containing shared segments, the daughter duplicates them, and
`kine_charge` — which is computed from the shower's own point clouds against the
2D charge maps with **no cross-shower dedup** — lands arbitrarily on one of the two
objects. Both observed failure modes are visible in the arms:

- **281485**: daughter gets `0.00`, the co-owner gains the whole 38.9 MeV.
- **165157**: daughter gets 67.77, the co-owner drops to 23.87, and a third shower
  retypes 11 → 2212.

**Size.** **7 of the 50 fired parents** hold segments that another shower also
owns (`num_segments` exceeds the count of dump segments mapping to them: 76350,
165157, 176502, 281485, 350354, 396222, 415278 — note 3 of the 4 broken π⁰s are in
this list). Of the 50 daughters, **2 are demonstrably pathological**. The dump's
single-valued segment→shower field cannot resolve the rest; **the C++ can, and the
arm reports the exact number.**

**The change.** Build a segment→shower-multiplicity count over all showers at the
top of `shower_split` (keyed on the segment's graph index — never on the pointer,
determinism), and **refuse a component whose segments are shared with a shower
other than the parent**. A refusal, not a repair: making shared-membership peels
work means dividing charge between duplicate objects, which is the `kine_charge`
dedup problem (`kine_charge_owned_scan`, knob-gated, runs later) and not this
round's.

**Criterion, pre-registered:** the two pathological daughters disappear; census
exact **≥ 35** (no loss); `q_extra` **≤ 6.7 %**; 0 ADVERSE.

### 3.2 P1.2 — the impact-parameter veto  `shower_split_max_impact`

Fire only when `b ≤ max_impact` (cm; `0` = no bound = today). §2.1 states the
evidence and the honest caveat. **Arm at `b = 12` cm.**

`b` is added to the `WCT_SHOWER_SPLIT_DEBUG` tape so the C++ value can be checked
against the offline one on the same arm — the doc pr/138 §B1 discipline (172/172
agreement on the accept decision) applied to a new quantity.

**Criterion, pre-registered:** census exact **≥ 36** *with 54332 the event that
must move* (it is the only exact loss); `q_extra` **≤ 7.0 %** — a small rise is
expected and allowed, because more than half the fires stop; if `q_extra` exceeds
that, the veto has given back the splitter's own gain and the knob stays off.

**`shower_split_max_vgap` (P3.1b) is the same dial family** measured along the ray
instead of perpendicular to it (AUC 0.735 vs 0.777). It is **not dropped** — it is
the comparison arm if P1.2 fails, and doc pr/138 §4.3 already priced it.

### 3.3 P1.3 — the daughter's start segment  `shower_split_em_start`

**11 of the 50 daughters come out typed μ and one proton — 24 % of everything the
splitter produces is not EM.** The split retypes no *segment* (segment PDGs are
byte-identical off vs on); SBND runs `shower_pdg_from_start_segment = true`, and
doc pr/138 §B3 chose the daughter's start segment as **the member nearest the
reference vertex, with no regard to PDG**. That choice is mine and it is the cause.

| event | daughter | kine (MeV) | segment PDGs | shower PDG |
|---|---|---|---|---|
| 84229 | 9058 | **476.4** | {13: 16, **11: 17**, 2212: 1} | **13** |
| 269774 | 31043 | **181.6** | {**11: 10**, 13: 3, 2212: 3} | **13** |
| 281781 | 14007 | 87.7 | {**11: 4**, 13: 1} | **13** |
| 415278 | 107184 | 62.4 | μ-majority | 13 — correct |
| 292643 | 112035 | 31.7 | {**11: 3**, 13: 1} | **13** |
| 396037 / 318769 / 281165 | | 24–30 | EM/μ **tied** | 13 |
| 284200 / 71642 / 281165 | | 1.5–46 | μ-majority | 13 — correct |

**The owner's stated harm, verified and with the mechanism corrected.** The owner
wrote *"biased the energy a lot due to muon mass"*. The bias is real and large, but
it is **not the rest mass** — it is the recombination/fudge pair. `kine_charge`
divides the collected charge by `recom × fudge`, and the pair is chosen on
`get_flag_shower()`, which a μ-typed shower does not have
(`NeutrinoEnergyReco.cxx:337-342`; `calculate_kinematics_long_muon` clears the
shower flag). At the SBND production values —
`kine_shower_recom_factor 0.58 × kine_shower_fudge_factor 0.86 = 0.4988` versus
`kine_recom_factor 0.87 × kine_fudge_factor 0.95 (C++ default) = 0.8265` —

> **a μ-typed EM daughter's energy is low by a factor 1.657, i.e. ≈ 40 %.**
> evt84229's 476.4 MeV daughter would read **789 MeV** as an EM shower.

On top of that a μ-typed shower is invisible to the π⁰ finder except through K20.
Both harms, one cause.

**The change:** pick the daughter's start segment as the member nearest the
reference vertex **among EM-typed members**, falling back to nearest-overall when
the daughter has no EM member. Minimal — it changes only the tie-break, not the
membership.

**The property that must survive**, and it is checked rather than asserted: the
nearest-the-vertex rule exists so `init_dir` points forward (doc pr/138 §B3: 116
peels, **zero** backwards). The tape's `fwd=` field is re-read on the ON arm; **any
backwards peel fails this knob.**

**Criterion, pre-registered:** μ-typed daughters fall from 11/50 to ≤ 4/50 (the
honestly-muonic ones); `fwd < 0` on **0** peels; census exact **≥ 35**;
`q_extra` **≤ 6.7 %**.

### 3.4 P1.4 — re-home the orphan  `shower_split_rehome`

The owner's requirement, stated five times in the scan (*"should be part of the
earlier EM shower cluster"*) and again now: *"We should re-sit them into the nearby
showers."* A cut that leaves an orphan has not finished the job.

**The change:** after the peel and the kinematics refresh, offer each daughter to
the nearest EM-typed shower under a forked absorb (M10 — fork
`satellite_absorb` `:9864`, do not extract), then `add_shower` + `erase` +
recompute exactly as that pass does.

**Three constraints the existing absorb predicates do not know about, all
mandatory:**

1. **Never re-home into the parent** — that silently undoes the split. The owner's
   comments all say the *earlier* shower, i.e. a third object.
2. **Host must out-charge the daughter** and be EM-typed (`|pdg| == 11`), matching
   `satellite_absorb`'s host rule.
3. **The sibling-merge failure mode**: if the nearest EM shower is the π⁰ partner,
   re-homing merges the two γs and destroys the pair the split just enabled. **The
   census is the referee** — if census falls below 35 on this arm, check the moved
   events for a daughter merged into its own sibling *first*.

Gap: `shower_split_rehome_gap`, default **4 cm** — the same single-linkage scale
the splitter already uses to build bundles, so the pass introduces no new distance
constant.

**Criterion, pre-registered:** census exact **≥ 35**; `q_extra` **≤ 6.7 %**; the
count of unpaired EM daughters above 20 MeV falls from 10; 0 ADVERSE.

**Sizing note, and a retraction that stands.** An early join by the daughter's new
start-segment id reported *"0 of 51 daughters are used by the π⁰ finder"* and would
have made this the headline front. Joining the other way — iterate the dump's
`showers[]` and ask which are daughters — says daughters pair at **18 %** against a
**6 %** base rate, i.e. three times better. The first number was a lookup bug. The
real front is **10 EM-typed daughters above 20 MeV that did not pair**, two of them
large (evt176502 node 148258 at 570.7 MeV, evt415278 node 56091 at 505.3 MeV).

---

## 3bis. Phase 1 — RESULTS  *(2026-08-31, all four arms measured alone)*

**Gate first.** All four knobs at their shipped defaults reproduce production
byte for byte: `work-pr139r1-off-*` vs `work-pr138r3-flipchk-*`, **rc=0 on all
four samples, 132 + 212 + 38 + 96 = 478 / 478 archives byte-identical,
`missing/unpaired events: 0` on every one.** Compiled-config proof from each
arm's own `.wct-cfg-evt*.json`: the OFF arm carries `{'shower_split': True}` and
**no pr/139 key**; each ON arm carries exactly its own one.
`./build/clus/wcdoctest-clus` 235 cases / **2627 assertions**, 0 failed, with the
five new defaults pinned. Freshness proof done before every arm; the binary was
**pinned** (`/home/xqian/tmp/pin-pr139b`) so a peer's `wcbuild` could not swap
`local/lib` mid-campaign.

**The baseline is re-measured on the same pipeline, not quoted from doc pr/138.**
`work-pr139r1-off-*` scored through `pr139_score.sh` gives q_miss **16.7 %**,
q_extra **6.7 %**, census exact **35 / 66**, median q_f1 **0.922** — identical to
the doc pr/138 baseline, so the prepdir/label epoch matches and every Δ below is
the knob and nothing else.

(*daughters* = objects in the arm's dump whose `shower_id` the pre-split world
`work-pr138r2-c90off-*` does not have — so a re-homed daughter, being merged
away, correctly stops counting. The re-home **tape** counts *decisions* (51),
which is a different question from *survivors*.)

| arm | knob | census **exact** | partial | none | no-group | q_miss | q_extra | med q_f1 | ADVERSE | daughters | μ-typed | kine=0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `off` | — (baseline) | 35 | 16 | 3 | 12 | 16.7 % | 6.7 % | 0.922 | — | 50 | 11 | 1 |
| `onshared` | **P1.1** | 35 | 18 | **2** | **11** | 16.7 % | 6.9 % | 0.922 | **0** | 47 | 11 | **0** |
| `onb12` | **P1.2** @ 12 cm | **36** | 18 | **1** | **11** | **14.5 %** | 7.4 % | **0.932** | **0** | 22 | 4 | **0** |
| `onemst` | **P1.3** | 35 | 16 | 3 | 12 | 16.7 % | 6.7 % | 0.922 | **0** | 50 | **2** | 1 |
| `onrehome` | **P1.4** @ 4 cm | 35 | 16 | 3 | 12 | 16.7 % | 6.8 % | 0.922 | **0** | 44 | 9 | **0** |
| `onrh15` | **P1.4** @ 15 cm | 35 | 16 | 3 | 12 | 16.4 % | 6.9 % | 0.922 | **0** | 38 | 9 | **0** |
| `oncomb` | **P1.1+P1.2+P1.3** | **36** | 18 | **1** | **11** | **14.5 %** | 7.6 % | **0.932** | **0** | 21 | **1** | **0** |

### P1.2 — the veto is the result of this round

**Census exact 35 → 36, and it moved exactly four events — the four of §2 and
nothing else:**

| event | OFF | `onb12` | |
|---|---|---|---|
| **54332** | partial | **exact** | the one exact loss, recovered — the pre-registered must-move event |
| 165157 | no-group | partial | |
| 281485 | none | partial | |
| 396222 | none | partial | |
| 280972 / 56243 / 314838 | exact | **exact** | **every census gain the splitter bought is kept** |
| 269774 | partial | partial | kept |

**4 of 66 events changed class. No other event moved.** `none` 3 → 1,
`no-group` 12 → 11, and `q_miss` **falls 2.2 points** while `q_f1` rises.

**The C++ impact parameter is the one the offline table priced — checked, not
asserted** (`pr139_tape_check.py`, the doc pr/138 §B1 discipline applied to a new
quantity). Over **390** taped candidates joined to an offline object, **383 agree
to within 0.5 cm**, median difference **0.000 cm**. The **7** that do not are exactly the objects doc pr/138 §B1
already named: 76346 (the 60 cm vertex move), 169626 and 396222 — events where
the π⁰ finders re-seat `main_vertex` *after* the splitter, so the dump vertex and
the splitter-time vertex are genuinely different points. That is the known
effect, and it is why P3.1 exists. On the veto's own decisions it changes
nothing: 396222's `b` is 22.15 cm at the splitter-time vertex rather than 29.23,
still far outside the bound. **The veto's decision on all eight movers is
correct: 4 gains kept, 4 losses vetoed, 8 / 8.**

**Criteria, scored honestly.** Census ≥ 36 with 54332 moving — **MET**.
`q_extra ≤ 7.0 %` — **MISSED at 7.4 %.** The criterion the instrument itself
defines (doc pr/136 §11.2: *fails if q_extra rises by more than q_miss falls*)
is met by a wide margin: **q_extra +0.7 pt against q_miss −2.2 pt.** Both
readings are stated because the first was pre-registered and I am not moving the
goalposts after the fact; the second is the rule this campaign has used since
pr/136 and it is the one the owner's own steer describes (*"a bit q_miss is OK,
we want to balance the q_miss vs q_extra"*). **The price is real and must be
quoted with the gain: the veto stops 29 of 51 fires.**

### P1.1 — the guard works, and the C++ saw more sharing than the dump could

Daughters **50 → 47**: the guard refuses **3** peels, one more than the two the
dump's single-valued segment→shower field could resolve — exactly why the count
was built in C++. **The `kine_charge = 0.00` daughter is gone**, `none` 3 → 2 and
`no-group` 12 → 11. Census exact unchanged at 35 (as §2's arithmetic predicted:
none of the three was an exact). `q_extra` 6.7 → 6.9 % — refusing a peel leaves
the shared charge in the parent, and the scanner calls some of it extra. **0
ADVERSE.**

### P1.3 — the defect is fixed, and the census does not notice

**μ-typed daughters 11 → 2**, EM 38 → **47**. The two survivors are the honestly
muonic 3 MeV and 2 MeV fragments. Criterion (≤ 4/50) **MET**.

**The forward property is measured, not asserted.** The nearest-the-vertex rule
existed to keep `init_dir` pointing downstream for the π⁰ finders, and this knob
changes exactly that tie-break, so the tape's `fwd=` field is re-read on the arm:

| arm | peels | **backwards (`fwd < 0`)** | min `fwd` | mean `fwd` |
|---|---|---|---|---|
| shipped rule (`onrehome`, seed unchanged) | 51 | **0** | 0.219 | 0.940 |
| `onemst` (EM-preferring seed) | 51 | **0** | 0.219 | **0.945** |

Zero backwards peels, and the mean alignment is marginally *better* than the
rule it replaces. Criterion **MET**. The census not moving would not have been
evidence of this — a backwards `init_dir` on a daughter that never pairs costs
nothing on the census and still poisons the finder on some later event.

**The 1.657 energy bias is confirmed empirically, not just predicted:**

| event | daughter | kine as μ | kine as EM | ratio |
|---|---|---|---|---|
| 84229 | 9058 | 476.38 | **639.33** | 1.342 |
| 269774 | 31043 | 181.58 | **296.91** | 1.635 |
| 281781 | 14007 | 87.74 | 146.52 | 1.670 |
| 284200 | 32039 | 46.11 | 76.40 | **1.657** |
| 396037 | 57009 | 29.94 | 49.61 | **1.657** |
| 281165 | 56025 | 25.10 | 41.58 | **1.657** |

Three land on the predicted 1.657 exactly; the others differ because the start
segment itself moved, which slightly changes the associate cloud. **461 MeV of EM
energy is restored across nine objects, and 1426 MeV of EM charge becomes
visible to the π⁰ finder at all.**

**And the census does not move — 35, with every class identical to the baseline
to the digit.** That is the honest result: the knob fixes a typing and an energy
defect, and on this 66-event hand-π⁰ set it buys no new pair. It is worth
shipping because a 476 MeV shower called a muon is wrong, not because the census
rewards it.

### P1.4 — measured at two gaps; correct, and not gradeable yet

#### at 4 cm — too tight to meet the owner's ask

**6 of 51 daughters re-homed; 45 stay orphans.** Census, q_miss and median q_f1
all unchanged; `q_extra` 6.7 → 6.8 %; **0 ADVERSE**. The successful re-homes are
at 3.0 and 3.6 cm gaps — i.e. only daughters already touching another shower move.

**The four large orphans it does not reach** are the ones the owner's requirement
is really about: evt396222 node 129310 (1149 MeV), evt176502 node 148258
(603 MeV), evt314838 node 13010 (587 MeV), evt415278 node 24073 (506 MeV).
So the knob is *correct and nearly inert* at the bundle scale, and the question
it raises is the gap.

#### at 15 cm — alive, and it reaches the objects the owner meant

`work-pr139r2-onrh15-*`: **12 of 51 re-homed**, 39 orphans. census exact **35**
(unchanged), q_miss 16.7 → **16.4 %**, q_extra 6.7 → **6.9 %**, median q_f1
0.922, **0 ADVERSE**. The re-homes it adds are the right ones:

| daughter | parent | kine | host | gap |
|---|---|---|---|---|
| evt176502 148258 | 109119 | **602.9 MeV** | 23030 | 5.36 cm |
| evt281485 **91111** | 89095 | 38.8 MeV | **91112** | **0.00 cm** |
| evt76350 76079 | 103093 | 44.6 MeV | 12027 | 10.54 cm |
| evt122660 38040 | 9111 | 20.8 MeV | 9110 | 12.47 cm |

**evt281485's zero-energy daughter goes back into 91112 — the very shower it was
duplicating — at a gap of exactly 0.00 cm.** That is the same defect P1.1
prevents, reached from the other end, and it is the strongest single piece of
evidence that the re-home predicate is picking the right host.

**But it still cannot be graded.** `q_miss`/`q_extra` **cannot judge a re-home**
until the joint label set exists (§5.2), because the target for a re-homed part
is exactly what the two scans disagree about. A 0.3-point `q_miss` fall against a
0.2-point `q_extra` rise is inside that disagreement, not outside it.

### P1.5 — the combination, and what I recommend

`work-pr139r2-oncomb-*` (P1.1 + P1.2 + P1.3 together, the shape the owner would
actually flip):

| | off | oncomb |
|---|---|---|
| census **exact** | 35 | **36** |
| partial / none / no-group | 16 / 3 / 12 | 18 / **1** / **11** |
| q_miss | 16.7 % | 14.5 % |
| q_extra | 6.7 % | 7.6 % |
| median q_f1 | 0.922 | **0.932** |
| daughters / μ-typed / `kine=0` | 50 / 11 / 1 | 21 / **1** / **0** |
| ADVERSE movers | — | **0** |

**Again, exactly four events changed class, and they are the four of §2.**

**One caveat that has to be said out loud, because it cuts against my own
headline.** The `q_miss` fall of 2.2 points is *not* mostly a physics gain. Doc
pr/138 §3.3 measured that **93–94 % of the `q_miss` rise the splitter caused was
a label-epoch artefact** — the attribution scan calls these objects one shower
while the split scan calls them three to five. A veto that stops 29 of 51 fires
reverses that artefact along with everything else. **So P1.2 should be judged on
the census — +1 exact and four recovered π⁰s — and not on `q_miss`.** The same
caveat is why P1.4 cannot be graded at all before §5.2.

### Recommendation  *(SUPERSEDED IN PART by §6 — the row for P1.2 is withdrawn)*

**Read §6 first.** The owner's 2026-09-01 scan tested `b ≤ 12` on fresh labels
and it fails: it suppresses **9 of the 19 cuts he confirms**. The table below is
left as written so the reversal is visible rather than tidied away; §6.5 says
line by line what survived.

| | recommendation (2026-08-31) | why |
|---|---|---|
| **P1.3** `shower_split_em_start` | **FLIPPED 2026-08-31** — *"flip shower_split_em_start now as you said."* | a defect in shipped code, with a measured 40 % energy error behind it; every instrument unchanged to the digit, so there is nothing to trade. Flip-equivalence gate **478 / 478**, compiled-config proof from the arm's own `.wct-cfg`. |
| **P1.2** `shower_split_max_impact = 12` | ~~take to the scan first~~ → **WITHDRAWN by §6** | it did what was asked on the census, but the bound was chosen after seeing 8 movers and the scan it was sent to came back against it: **9 of 19 confirmed cuts suppressed**, including 281485's, which the owner calls a correct cut. The successor is `skip_shared` + `b ≤ 30` (§6.3). |
| **P1.1** `shower_split_skip_shared` | ~~flip with P1.2~~ → **PROMOTED by §6** | not "partly redundant": it is **the** fix for two of the four broken π⁰s (165157 and 281485), it refuses a peel in exactly the three shared-membership events, and it costs no correct cut outside them. It now leads the next arm. |
| **P1.4** `shower_split_rehome` | **hold for §5.2** | it satisfies the owner's requirement at 15 cm (12 of 51, including evt176502's 603 MeV orphan and evt281485's 0 MeV one, the latter at gap 0.00 cm — straight back to the shower it was duplicating). But `q_miss`/`q_extra` **cannot grade a re-home** until the ~15 touched showers are re-marked per part. Flipping it now would be a guess wearing a number. |

## 3ter. Where the round stands, and the next session's order

**Shipped and production ON:** `shower_split_em_start` (P1.3), toolkit
`f5e17798`, flip-equivalence gate 478 / 478.

**Nothing else is flipped**, and after §6 nothing else should be flipped without
the arm below.

| # | next session, in order | why it is next |
|---|---|---|
| **1** | **one arm: `skip_shared` + `max_impact = 30`** (`work-pr140r1-on-*`) | §6.3: the only operating point the fresh labels support. **Pre-registered:** census exact **≥ 36**, `q_extra` **≤ 7.2 %**, 0 ADVERSE, and — the criterion §3bis lacked — **no more than 3 of the 19 owner-confirmed cuts suppressed**, checkable offline from `pr139-scan-verdicts.tsv` before the arm is even scored. |
| **2** | **merge the 2026-09-01 per-part boundaries into the completeness target** | §6 delivered 19 objects with a per-part segment assignment and 20 KEEP. Until `em117_score` reads them, `q_miss`/`q_extra` still cannot grade a splitter *or* a re-home — which is the whole reason P1.4 is parked. |
| **3** | **P1.4 re-home, re-priced at 15 cm against the merged target** | It already reaches the right hosts (evt281485's 0 MeV daughter → its co-owner at gap 0.00 cm). It has never had a metric that could see it. Item 2 gives it one. |
| **4** | **the k ≥ 3 cap** | §6.1: the k ≥ 3 boundary mean of 0.800 is **`max_parts = 2` refusing to make the third cut**, not the kernel getting it wrong — the owner calls 396222 k=7 and 415278/23037 k=5. Raising the cap is now a *measurable* question rather than the blind one doc pr/138 §B3 parked. |
| **5** | **the remaining false fire** 278420/61027 (`b` 26.78) | one object; only worth a feature hunt if items 1–3 leave it isolated. |

**Not next, and now with a reason:** a wider `b` sweep. §6.3 prices the whole
dial; the answer is 30 and searching it again would be fitting to 39 labels.

## 4. Phase 2 — the owner's scan  *(port 5022, tag `splitscan-0902-pi0`)*

```
./split_display/serve_split_display.sh 5022 --scan-tag splitscan-0902-pi0
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -L 5022:localhost:5022 <user>@wcgpu1.phy.bnl.gov
#   then http://localhost:5022/split_viewer
```

5022 is the script's own default and the port the owner named. The tag is fresh
(M13); the viewer refuses to write into a dir holding labels it did not create.

**Twenty objects, three asks:**

1. **The four π⁰s of §2** — one question each, and it is **two** questions, not one:
   *is the cut right*, and *should the two parts still be one γ for the π⁰?*
   P3.4 needs both answers and they can differ. **4 objects.**
2. **The ~15 hand-marked showers the splitter touches**, re-marked **per part** —
   this is what unblocks P3.2 and therefore the measurement of P3.3 and P3.4.
   **15 objects.**
3. **evt314838** — the standing adjudication item (doc pr/138 §3.5): three of the
   owner's own instruments give three answers (split scan SPLIT2 high-confidence;
   attribution scan purity 0.715 → **1.000**; hand π⁰ needs the charge). One call
   settles which instrument leads when they conflict. **1 object.**

---

## 5. Phase 3 — the owner's evaluation list

### 5.1 The ν vertex — P3.1, and it is the only vertex lever in scope

The owner ruled the upstream ν vertex out and named the exception: *"the only thing
that we can touch is the neutral pi0, which move the vertex."* That is
`id_pi0_backproject_vertex` (`:6272`, K21, production ON) and
`id_pi0_without_vertex`'s path-2 hack (`:7896`) — both re-seat `main_vertex`, and
**both run AFTER the splitter**, so the splitter measures every feature (including
P1.2's `b`) from the *uncorrected* point.

Measured (doc pr/138 §B1): the vertex moves on **5 of 172** scanned objects, by
**60.16 cm** on evt76346 and **14.50 cm** on evt396222. **Expected yield: small**,
and one of the five is the event §2 writes off. It is in the plan because it is
cheap, because the ordering is arguably wrong on its own terms, and because it is
the only vertex work allowed — **not** because it is promising.

**P3.1c is the genuinely unexplored one.** The owner asked for *"a pr/129-style
pointing test that doesn't read from the vertex"*. Note that P1.2's `b` **is**
measured from the vertex and therefore belongs to the family §1's C2 row closed.
The complement — a discriminator built from the object's **own internal
structure**, e.g. start dE/dx (a γ conversion deposits ≈ 2 × MIP; MicroBooNE
separates e from γ on exactly this) or axis coherence along the object — has never
been measured here. It is the one feature family with no negative result against
it.

### 5.2 P3.2 — the joint label set  ← the binding constraint

Doc pr/138 §3.3: `pr136_completeness.py`'s target comes from the 2026-08-27/28
attribution scan, which called several of these objects **one** shower, while the
2026-09-01 split scan says three to five. **93–94 % of the measured `q_miss` rise
is that conflict, not a real cost** — and it runs the other way too: **28 % of the
`q_extra` gain sits on objects the split scan calls KEEP.** Until the ~15 touched
showers are re-marked per part, `q_miss`/`q_extra` cannot *grade* a splitter; they
can only be quoted. Phase 2 ask 2 is this.

### 5.3 P3.3 — re-home, re-sized

The same knob as P1.4, re-measured after P1.3 and P1.2 have moved the population:
fixing the μ typing changes which daughters are candidates, and the veto — if it
goes on — removes two of the ten unpaired daughters (165157's at 67.8 MeV and
54332's at 57.2 MeV are the false fires' daughters). **Sizing it against today's
ten would size it against a population that will not exist.**

### 5.4 P3.4 — split-aware π⁰ pairing

**Demoted by §2, not cancelled.** It is no longer the fix for 281485 (that is
P1.1) and there is nothing to build for 396222. What remains is the measured
mechanism: of the eight hand π⁰ pairs the splitter moved, **six went further from
135 MeV**, always the same way — a split γ loses charge to its sibling and the
finder pairs on the reduced value. 54332 is the clean example (partner changed,
117.5 → 110.6 MeV).

Change, when its turn comes: record a `split_parent` id on each daughter and let
the pair loop consider `(parent-as-one, other γ)` alongside `(daughter, other γ)`,
keeping whichever scores better. **Risk, stated:** re-admitting the un-split parent
recreates the over-clustered γ the splitter removed, so the census gain and the
`q_extra` floor fight each other. Knob `pi0_split_aware_pairing`, DEFAULT OFF.
**This front cannot be honestly graded before P3.2.**

---

## 6. The owner's scan — and what it overturns  *(2026-09-01, tag `splitscan-0902-pi0`, 39 objects)*

**The headline is that this round's own recommendation was wrong, and the scan is
what showed it.** §2.1 said, before the scan, that `b ≤ 12` "was chosen after
seeing 8 movers" and "must not be reported as a discovery". It now has an
independent test and **it fails it.**

Verdicts: **20 KEEP, 15 SPLIT2, 2 SPLIT3, 2 SPLIT4+** — 19 confirmed cuts.

### 6.1 The trigger does not miss. The kernel's boundary is right.

| | on the 2026-09-01 labels | doc pr/138's older set |
|---|---|---|
| trigger **efficiency** | **1.000** (19 / 19) | 0.767 |
| trigger **purity** | 0.792 (5 false of 24) | 0.805 |
| boundary, **SPLIT2** | median **1.000**, mean **0.981**, **12 of 15 exact** | median 1.000, mean 0.974 |
| boundary, k ≥ 3 | mean 0.800 — **capped by `max_parts = 2`**, not a kernel failure | 0.756 |

**On these 39 objects the trigger never misses a cut the owner wants.** Every
error is a false fire, and there are five:

| event | node | `b` (cm) |
|---|---|---|
| 174771 | 87065 | 102.10 |
| 393505 | 63025 | 62.82 |
| **54332** | **122091** | **39.72** |
| 278420 | 61027 | 26.78 |
| **165157** | **9000** | **13.05** |

### 6.2 Group A — and 281485 is the one that reverses the plan

| event | node | **owner** | `b` | the 12 cm veto | verdict on the veto |
|---|---|---|---|---|---|
| 54332 | 122091 | **KEEP** | 39.72 | vetoes | **right** |
| 165157 | 9000 | **KEEP** | 13.05 | vetoes | **right** |
| **281485** | **89095** | **SPLIT2** | 23.67 | vetoes | **WRONG — it suppresses a cut the owner confirms** |
| **396222** | **9059** | **SPLIT4+ (k=7)** | 22.15 | vetoes | **WRONG** (though the owner discounts the event: *"very busy events hard to get it right"*) |
| 314838 | 110088 | **SPLIT2** | 8.32 | keeps | right — and this settles §4's ask C: **the split scan leads** |

**So the veto recovered 281485's π⁰ for the wrong reason.** The cut is correct;
what was broken was the *peel*, and **P1.1 is the principled fix.** Measured:
`skip_shared` refuses a peel in exactly three events — **165157, 281485, 350354**
— and on its own it takes 165157 `no-group → partial` and 281485 `none →
partial`, i.e. **it recovers both of the events P1.2 was credited with, without
touching a single correct cut elsewhere.**

### 6.3 The price of the bound, priced properly at last

| `b ≤` | fires | right | wrong | eff | pur | **confirmed cuts suppressed** |
|---|---|---|---|---|---|---|
| 10 | 9 | 9 | 0 | 0.474 | 1.000 | **10 of 19** |
| **12** (what §3bis proposed) | 10 | 10 | 0 | 0.526 | 1.000 | **9 of 19** |
| 20 | 14 | 13 | 1 | 0.684 | 0.929 | 6 of 19 |
| 25 | 16 | 15 | 1 | 0.789 | 0.938 | 4 of 19 |
| **30** | 18 | 16 | 2 | 0.842 | 0.889 | **3 of 19** |
| none (today) | 24 | 19 | 5 | **1.000** | 0.792 | 0 |

**Only 54332 actually needs a bound** — 165157 is handled by P1.1, and 174771 /
393505 sit at 102 and 63 cm where any sane bound catches them. With **P1.1 on**:

| `b ≤` | fires | right | wrong | eff | pur | false fires left |
|---|---|---|---|---|---|---|
| 12 | 9 | 9 | 0 | 0.474 | 1.000 | — |
| 25 | 13 | 13 | 0 | 0.684 | 1.000 | — |
| **30** | 15 | 14 | 1 | **0.737** | **0.933** | 278420/61027 |
| none | 21 | 17 | 4 | 0.895 | 0.810 | 4 incl. 54332 |

**`skip_shared` + `b ≤ 30` is the operating point the labels support**: it
suppresses **3** confirmed cuts (294174 ×2 and 415278/83139, all at `b` 48–209 cm)
instead of nine, and leaves one false fire standing. That is the next arm.

### 6.4 The scan is stable against the previous one

36 of the 39 were also labelled in a `splitscan-0901-*` tag. **Four** changed at
the KEEP/SPLIT level, and **three of those four are the retired `TRIM` class
collapsing to `KEEP`** (179611/10001, 292524/9018, 499577/13009) — a vocabulary
change, not a reversal. **One real reversal**: 396037/69026 `KEEP → SPLIT2`. One
`k` change: 174771/91075 `SPLIT3 → SPLIT2`. The owner is self-consistent, so the
new labels can be trusted as the arbiter above.

### 6.5 What §3bis got right, and what it got wrong

| claim | status |
|---|---|
| the trigger's kernel and boundary are sound | **confirmed on an independent 39** — eff 1.000, SPLIT2 median 1.000 |
| P1.1 fixes a real pathology | **confirmed and promoted** — it is the fix for two of the four, not a modest extra |
| P1.3's μ-typing fix | **unaffected** — already flipped, and nothing here touches it |
| **`b ≤ 12` is the result of the round** | **WRONG.** It buys +1 exact by suppressing 9 of 19 confirmed cuts. Withdrawn. |
| "judge P1.2 on the census, not on q_miss" | right, and **not sufficient** — the census is 66 events and could not see the 9 suppressed cuts, which live outside it. The hand scan could. |

## 7. What is closed, and the measurement that closed it

See §1 rows C1–C4. In words:

| | why it is closed |
|---|---|
| another feature measured **from** the ν vertex | `void_frac` AUC **0.146** (backwards), `b` 0.777, `vgap` 0.735 — the information is not there. `b` is shipped as a *dial* in P1.2 precisely because it is not a separator: it is priced, not believed. |
| retuning `valley_best ≤ 0.95` | holdout spent (doc pr/138 §A5.4); fires are 80 % right overall and 100 % right on a vertex-attached object |
| **k ≥ 3 splitting** | missed its pre-registered gate (0.635 → 0.756 vs ≥ 0.85), capped by `max_seeds = 4` which cannot move without moving the trigger |
| the **no-valley** class | 7 of 9 missed splits have `valley_best = 1.000` — no charge dip exists because the γs overlap |
| chasing the raw `+4.3 pt` of `q_miss` | 93 % of it is §5.2's label-epoch artefact |
| the upstream ν vertex finder | **owner, explicitly out of scope** |

---

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# sec 2 -- the four broken pi0s, keyed on shower_id (NOT on the start-segment id)
python3 scripts/pr139_broken_pi0.py            # -> docs/pr/pr139-broken-pi0.tsv
# sec 2.1 -- the impact parameter on the eight census movers + the 172 scanned
python3 scripts/pr139_pointing.py --movers     # -> docs/pr/pr139-pointing.tsv
# sec 3.3 / 3.4 -- every peeled daughter and its parent
python3 scripts/pr139_daughter_fate.py         # -> docs/pr/pr139-daughter-fate.tsv

# phase 1 -- the arms (each knob ALONE), the gate, and the scores
./scripts/pr139_arms.sh off                    # work-pr139r1-off-*  (knobs at defaults)
./scripts/pr139_gate.sh  off                   # -> 478/478 vs work-pr138r3-flipchk
./scripts/pr139_on_arms.sh                     # onb12 / onshared / onemst / onrehome
./scripts/pr139_r2_arms.sh                     # off / oncomb / onrh15, on the fixed binary
for t in off onb12 onshared onemst onrehome; do ./scripts/pr139_score.sh $t; done
PR139_ARM=work-pr139r2 PR139_BASE=work-pr139r2-off ./scripts/pr139_score.sh oncomb
python3 scripts/pr139_tape_check.py work-pr139r2-oncomb  # C++ b vs offline b: 383/390 within 0.5 cm
# P1.3's forward check -- the property the EM-preferring seed could have broken
grep -h "SHOWER_SPLIT peel" work-pr139r1-onemst-*/pr_evt*/stdout.log \
  | grep -oP 'fwd=\K-?[\d.]+' | awk '{n++; if($1<0) b++} END{print n" peels, "b+0" backwards"}'
python3 scripts/pr139_arm_effect.py onb12 onshared onemst onrehome

# sec 6 -- the owner's 2026-09-01 scan, scored (the measurement that overturned P1.2)
python3 scripts/pr139_scan_analysis.py         # -> docs/pr/pr139-scan-verdicts.tsv

# the baseline everything is measured against: work-pr138r2-c90on-*
#   q_miss 16.7 %, q_extra 6.7 %, census 35/66, 19/49 impossible, median q_f1 0.918
python3 scripts/pr132_pi0_census.py --manifest98 em117-138c90on98-manifest.tsv \
    --manifest141 em114c-138c90on141-manifest.tsv --fudge 0.86 \
    --overlay-tag pi0scan-0829-agent

# phase 2 -- the owner's scan
./split_display/serve_split_display.sh 5022 --scan-tag splitscan-0902-pi0
```

---

## 8. Item 1 — the PRE-REGISTERED prediction  *(committed 2026-08-31, before the arm existed)*

**Naming note:** the arms of this section are `work-pr140r1-*` because §3ter
pre-named them that way. **There is no doc pr/140** — this file is the living
tracker and items 1–4 are its own P-rows. A future session grepping for a
doc 140 will not find one; look here.

§6 recorded a round whose recommendation was chosen *after* seeing eight movers
and was then killed by an independent scan. The structural fix is ordering:
this section, and `docs/pr/pr140-prereg.tsv`, are committed **before** the arm's
census exists, derived from the owner's 2026-09-01 labels alone.

```
python3 scripts/pr140_prereg.py     # -> docs/pr/pr140-prereg.tsv
```

### 8.1 The correction §3ter's criterion needed

§3ter pre-registered "**no more than 3** of the 19 owner-confirmed cuts
suppressed". Deriving it mechanically says that number is **5**, and §6.3's own
efficiency column already said so — 0.737 × 19 = 14 cuts firing, i.e. **5 not
firing.** The "3" was the *bound's marginal* price, counted with `skip_shared`
already applied; it silently omitted the two cuts `skip_shared` itself refuses.
Both numbers are real and they answer different questions, so both are
pre-registered here rather than the flattering one being kept:

| confirmed cuts that stop firing, vs today's production config | n | which |
|---|---|---|
| refused by **`skip_shared`** — the peel is *deferred*, not judged wrong | **2** | 281485/89095, 350354/18092 |
| rejected by the **`b ≤ 30`** bound — the cut is *refused* | **3** | 294174/16004 (b 209), 294174/71067 (b 75), 415278/83139 (b 48) |
| **total** | **5 of 19** | efficiency **0.737** |

The two classes are not equivalent and the doc should not average them. A
`skip_shared` refusal happens because the peel would duplicate segments two
showers both own — the owner's cut is right, but *this* peel cannot make it
cleanly. That is a deferral with a named successor (§8.4). A `b` rejection is
the trigger being told the cut is wrong, and on 294174 ×2 and 415278/83139 the
owner says it is not.

### 8.2 The prediction, in full

| | predicted |
|---|---|
| objects labelled | 39 (19 confirmed cuts, 20 KEEP) |
| splitter fires | **15** — 14 right, 1 wrong |
| trigger **efficiency** | **0.737** (14 / 19) |
| trigger **purity** | **0.933** (14 / 15) |
| confirmed cuts suppressed | **5** total = 2 deferred + 3 rejected |
| false fires left standing | **278420/61027** only (`b` 26.78) |

### 8.3 The margin check — is the offline `b` safe to predict the C++ `b`?

`pr139_tape_check.py` measured offline-`b` vs C++-`b` agreeing to 0.5 cm on
**383 of 390** objects. So a prediction is only safe where no decision sits
within ~1 cm of the bound. Closest three:

| object | `b` (cm) | margin from 30 |
|---|---|---|
| 278420/61027 | 26.78 | **3.22** |
| 406125/38021 | 26.35 | 3.65 |
| 281485/89095 | 23.67 | 6.33 |

**No decision is inside 3 cm of the bound**, i.e. every one of them is 6× the
measured offline-vs-C++ spread. The prediction is safe to hold the arm to.

### 8.4 Pass / fail, fixed now

The arm **passes** only if all four hold:

1. π⁰ census **exact ≥ 36** (baseline `work-pr139r3-flipchk` = the production config);
2. `q_extra` **≤ 7.2 %**;
3. **0 ADVERSE** vertex movers;
4. the tape reproduces §8.2 — **15 fires, exactly 5 confirmed cuts suppressed,
   and the five named objects are the named ones.**

If (4) holds but (1) fails, the operating point is refused and the reason is a
π⁰ effect the labels cannot see — which is the same failure mode §6 caught, in
the other direction, and it must be reported, not tuned around.

**Named successor for the 2 deferred cuts:** a *co-ownership* peel — assign each
shared segment to one part instead of refusing the whole peel. That would
recover 281485's and 350354's cuts without re-creating the duplicate-charge
pathology `skip_shared` exists to stop. Not in this round; it is a new item.

---

## 9. Items 3 and 4 — PRE-REGISTERED, before their arms were scored

Same discipline as §8: the arms `work-pr140r1-onrh15-*` and `work-pr140r1-onk3-*`
were launched in the same batch as item 1's, and these criteria are committed
before either is scored.

### 9.1 Item 4 — the k ≥ 3 cap  (`shower_split_max_parts` 2 → 3)

No C++ is needed: `max_parts` has been a knob since doc pr/138 (`C++ default 2`),
and `max_seeds` is hardcoded 4 upstream of it, so 3 is the one cheap arm.

§6.1 measured the k ≥ 3 boundary mean at **0.800** and read it as `max_parts = 2`
refusing the third cut rather than the kernel erring. Raising the cap changes
**every** object the splitter fires on, not only the four the owner called k ≥ 3,
so the arm needs a second criterion or a trade reads as a win. Both are fixed now:

| # | criterion | bar |
|---|---|---|
| **i** | k ≥ 3 boundary agreement, mean | **≥ 0.85** — doc pr/138 §B3's own bar, reused deliberately so the number compares to the one that parked this question |
| **ii** | **SPLIT2 must not degrade** | median stays **1.000** and **≥ 12 of 15** stay exact (§6.1's measured values) |
| **iii** | merged target (§8/item 2): parts with no reco match | must **fall** below the baseline's **6** |
| **iv** | no regression elsewhere | census exact **≥ 35**, **0 ADVERSE** |

If (i) passes and (ii) fails it is a trade, not a win, and it is reported as one.

### 9.2 Item 3 — the re-home, re-priced at 15 cm against the merged target

`work-pr139r2-onrh15-*` already measured the 15 cm gap **pre-flip**, with
`em_start` OFF; `em_start` changes which segment roots the daughter the re-home
is hunting a host for, so the arm is re-run on the flipped config.

The honest framing: **item 3's first question is about the instrument, not the
knob.** P1.4 has been parked since §3bis because no metric could see it —
re-homing an orphan daughter into a host moves charge *between* reco objects,
and the single-target metric matches one reco shower per hand shower, so the
move is invisible by construction. So:

| # | criterion | bar |
|---|---|---|
| **i** | *does the instrument see it at all* | the merged target must move on ≥ 1 row where the single-target metric does not |
| **ii** | worth flipping | parts with no reco match **falls**, or median part `q_f1` **rises**, with **no** row going backwards |
| **iii** | no regression | census exact **≥ 35**, **0 ADVERSE** |

**(i) failing is a real and publishable result**: it would mean the per-part
target is *still* not the instrument that grades a re-home, and item 3 stays
parked with a second measured reason rather than a hunch.

---

## 10. Item 1 — RESULT: the pre-registered operating point holds, at a price §8.4 could not see  *(arm `work-pr140r1-on-*`)*

`shower_split_skip_shared=1` + `shower_split_max_impact=30`, single arm on the
flipped production config, 239 events × 4 samples. Baseline
`work-pr139r1-onemst-*` (= the production config; proven byte-identical to the
post-flip arm `work-pr139r3-flipchk-*` by a 478/478 gate).

### 10.1 All four §8.4 criteria pass — but note what they could not measure

**§8.4's four bars were fixed before item 2's instrument existed.** They are
honoured below exactly as written, and they are *not the whole verdict*: §11.4
later prices this same arm at **one lost confirmed cut** (294174/71067) and
**−0.052** median part `q_f1`. Read §10 and §11.4 together, and §14.2 for the
trade the owner is actually being asked to make. This is the same audit §13.3
applies to item 4, turned on item 1.

| # | criterion | bar | measured | |
|---|---|---|---|---|
| 1 | π⁰ census **exact** | ≥ 36 | **36** (baseline 35) | **PASS** |
| 2 | `q_extra` | ≤ 7.2 % | **6.9 %** (baseline 6.7 %) | **PASS** |
| 3 | ADVERSE vertex movers | 0 | **0 / 0 / 0 / 0** on mcp1k, mcp2k, ncpi0, nuecc48 | **PASS** |
| 4 | the tape reproduces §8.2 | exactly | **39 of 39 objects did precisely what was pre-registered** | **PASS** |

Criterion 4 in full, from `scripts/pr140_tape_verify.py` against the arm's own
`WCT_SHOWER_SPLIT_DEBUG` tape (393 candidates, 51 fired, 16 vetoed, 3 shared
refusals):

| | predicted (committed `2e996db6`) | measured |
|---|---|---|
| fires | 15 | **15** |
| trigger efficiency | 0.737 | **0.737** |
| trigger purity | 0.933 | **0.933** |
| confirmed cuts suppressed, total | 5 | **5** |
| — deferred by `skip_shared` | 281485/89095, 350354/18092 | **exactly those two** |
| — rejected by the `b` bound | 294174/16004, 294174/71067, 415278/83139 | **exactly those three** |

The offline `b` predicted the shipped C++ on every one of the 39. §8.3's margin
argument (nothing within 3 cm of the bound, versus a 0.5 cm offline-vs-C++
spread) is confirmed rather than merely asserted.

### 10.2 What it buys against the withdrawn `b ≤ 12`

| | baseline (production) | **`skip_shared` + `b ≤ 30`** | the withdrawn `b ≤ 12` combo |
|---|---|---|---|
| π⁰ census **exact** | 35 | **36** | 36 |
| `partial` / `none` / `no-group` | 16 / 3 / 12 | **17 / 2 / 11** | 18 / 1 / 11 |
| `q_miss` | 16.7 % | **16.4 %** | 14.5 % |
| `q_extra` | 6.7 % | **6.9 %** | 7.6 % |
| **confirmed cuts suppressed** | 0 | **5** | **11** (9 by the bound + 2 by `skip_shared`) |

**Same census headline as the withdrawn point, at less than half the cost in
cuts the owner confirms.** That is the whole result of item 1.

---

## 11. Item 2 — the merged (per-part) completeness target, and what it caught immediately

`em_display/em140_score.py` (a fork of `em117_score.py`, which stays byte-
untouched and keeps producing the number every doc from pr/117 quotes) takes
`--split-tag` and merges the owner's per-part boundaries into the target.

### 11.1 The two design decisions that carry the result

1. **Matching is injective.** Each reconstructed shower may be claimed by at
   most one part. Without this, one un-split reco object wins part 0 *and*
   part 1 and **a failure to split scores high on both** — the metric would be
   inverted relative to the bug it exists to fix. An unmatched part scores
   `q_comp = 0`.
2. **The denominator is preserved.** Target segments the split label never
   mentions — measured: **50 of 318** over the 12 overlapping SPLIT showers,
   and they are mostly `in` marks, because the split display shows *reco*
   membership and so never showed a segment the reco does not hold — become a
   residual part `*` that competes on the same injective rule. **Verified:
   `sum q_target` is 4.837e+08 under both metrics, to four figures.**

A third correction was forced by the first run: a part whose intersection with
the completeness target is **empty** is *not* a failed cut — it means the
completeness scan had already marked that part out of this shower. Counting
those (6 of 29 part rows) as misses doubled the apparent failure count. They are
now reported as their own class.

**Fork-fidelity gate**: with no `--split-tag`, `em140_score.py` reproduces
`em117_score.py` byte-for-byte on both label sets against a real arm
(`emprep-139onemst`, 41 and 69 lines of output, `diff` rc=0). The first attempt
at this gate ran on the *default* paths and compared **zero events** — the em114
scan-time arm was retired in the 2026-08-31 cleanup — so it was re-run on an arm
that has data. A gate over an empty set is not a gate.

### 11.2 The metric change alone, measured on the baseline

Same arm, two metrics. **This is why the baseline is scored both ways**: a
metric change and a reco change must never land in the same number.

| on `work-pr139r1-onemst` | single target (em117) | **merged target (em140)** |
|---|---|---|
| `sum q_target` | 4.837e+08 | 4.837e+08 |
| `q_miss` | 16.7 % | **11.2 %** |
| `q_extra` | 6.7 % | **8.6 %** |
| median row `q_f1` | 0.922 | 0.921 |
| **parts with no reco match** | *not expressible* | **6** |

`q_miss` falls 5.5 pt and `q_extra` rises 1.9 pt **with no reconstruction
change at all** — that is the single-target metric mis-booking a correct split
as under-clustering, quantified for the first time.

### 11.3 The number only this metric can produce

**Parts with no reco match = a cut the owner confirmed that the reco did not
make.** On the production baseline there are **6**, carrying 8.3e6 of charge:

| event | shower | part | q_target | why |
|---|---|---|---|---|
| 269774 | 13237 | 1 | 1.06e+06 | |
| 415278 | 23012 | 2 | 1.94e+06 | owner k=3, `max_parts` = 2 |
| 415278 | 23037 | 2, 3, 4 | 1.00e+06, 2.03e+06, 3.51e+05 | owner k=5, `max_parts` = 2 |
| 463565 | 13001 | 2 | 1.93e+06 | owner k=3, `max_parts` = 2 |

**Five of the six are the `max_parts = 2` cap** — so item 2 does not merely
unblock item 3, it hands item 4 a direct measurement of the thing §6.1 could
only infer.

### 11.4 It priced item 1 in a way the census could not

Item 1's arm passes all four §8.4 criteria. Under the merged target it also
shows a cost that neither the census nor `q_extra` could name:

```
delta (work-pr140r1-on  -  baseline)
  q_miss +0.05 pt   q_extra +0.43 pt   median part q_f1 -0.052
  parts with no reco match: 6 -> 7  (+1)
    CUTS NOW MISSED: 294174/71067 part 1
```

**294174/71067** is one of the three cuts the `b ≤ 30` bound rejects (`b` 75.20),
and it is the one of the three that also carries completeness labels. §8 said
three cuts would be rejected; the merged target now shows what one of them costs
in charge attribution. That is the instrument working on its first use.

---

## 12. Item 3 — RESULT: the re-home's blocker was never the metric  *(arm `work-pr140r1-onrh15-*`)*

`shower_split_rehome=1`, `shower_split_rehome_gap=15`, re-run on the flipped
production config because `em_start` changes which segment roots the daughter
the re-home is hunting a host for. Tape: **51 peels, 11 re-homed, 40 left orphan**
(pre-flip it was 12 of 51).

### 12.1 Against the §9.2 bars

| # | criterion | bar | measured | |
|---|---|---|---|---|
| i | the merged target sees something the single target does not | ≥ 1 row | **0 rows** | **FAIL** |
| ii | worth flipping | parts-with-no-match falls **or** median part `q_f1` rises | **6 → 6**, median part `q_f1` **+0.000** | **FAIL** |
| iii | no regression | census exact ≥ 35, 0 ADVERSE | **35**, **0 / 0 / 0 / 0** | PASS |

### 12.2 Criterion (i) is the result, and it has a mechanism

Row by row, over both label sets:

```
single-target rows that moved : 3
merged-target rows that moved : 3
  merged movers the single metric did not see : none
  single movers the merged metric did not see : none
```

**Exactly the same three rows. Not one more.** §3ter's premise for this item —
*"It has never had a metric that could see it. Item 2 gives it one."* — is
**wrong, and now measurably so.**

The mechanism is structural, and it is worth stating because it generalises:

> The per-part target exists to make a **cut** visible. Both metrics match one
> reconstructed object per target row, so **splitting** one hand shower into two
> was invisible to the single target and is visible to the merged one. But a
> re-home **merges** an orphan daughter into a host — it changes that host's
> membership, and *both* metrics see a membership change on the host's row
> identically. **The merged target buys sensitivity to cuts, not to merges.**

So P1.4 was never blocked on the instrument. It is blocked on there being
nothing much to gain: at 15 cm on the flipped config it moves `q_miss`
**−0.28 pt** and `q_extra` **+0.21 pt** on three rows, leaves the census at 35
and every part exactly where it was.

### 12.3 The verdict

**Do not flip the re-home.** It is not harmful (0 ADVERSE, census flat) and the
μ-typing problem the owner raised it against — *"I do not like the orphan things
to be identified as muon"* — was already solved by **P1.3**, which is production
ON and took μ-typed daughters from **11 to 2**. The re-home was the second
answer to a question the first answer had already closed.

40 of 51 daughters still find no host at 15 cm. Widening further is a search,
not a measurement, and §12.2 says the metric would not reward it. **This item is
closed, not parked.**

---

## 13. Item 4 — RESULT: raising the k cap is measured dead  *(arm `work-pr140r1-onk3-*`)*

`shower_split_max_parts` 2 → 3. No C++: the knob has existed since doc pr/138,
and `max_seeds` is hardcoded 4 upstream, so 3 is the one cheap arm.

### 13.1 Against the §9.1 bars — three of four fail

| # | criterion | bar | measured | |
|---|---|---|---|---|
| i | k ≥ 3 boundary agreement, mean | ≥ 0.85 | **0.800 → 0.771** — it went **down** | **FAIL** |
| ii | SPLIT2 must not degrade | median 1.000, ≥ 12 of 15 exact | median 1.000 held, exact **12 → 11** | **FAIL** |
| iii | merged-target parts with no distinct reco object | must fall below 6 | **6 → 5** | **PASS** |
| iv | no regression | census exact ≥ 35, 0 ADVERSE | census **34**, 0 ADVERSE | **FAIL** on the census |

**The `max_parts = 2` reference is validated, not assumed.** `work-pr140r1-onrh15`
is used as the baseline tape on the argument that the re-home acts on an orphan
daughter *after* the peel and so cannot perturb the `SHOWER_SPLIT part` lines.
That argument is checked rather than trusted: it reproduces §6.1's `oncomb`
numbers to three decimals — SPLIT2 median **1.000**, mean **0.981**, **12 of 15**
exact, k ≥ 3 mean **0.800**.

### 13.2 The blast radius is three objects, and two of them get worse

`max_parts = 3` changes only **3** of the 19 confirmed cuts. That is the whole
effect:

| event | node | owner k | k base → arm | agreement base → arm | |
|---|---|---|---|---|---|
| 415278 | 23037 | 5 | 2 → 3 | 0.801 → **0.872** | **UP** — the one real gain |
| 396222 | 9059 | 7 | 2 → 3 | 0.572 → **0.385** | **DOWN** — the third cut lands in the wrong place |
| 406125 | 38021 | **2** | 2 → **3** | 1.000 → **0.948** | **DOWN** — it cuts an object the owner wants left at two |

**The cap was not what was holding k ≥ 3 back.** §6.1 read the k ≥ 3 mean of
0.800 as `max_parts = 2` refusing the third cut. Lifting the cap lets the third
cut be made and it is made in the wrong place on two objects of three, including
one the owner labelled SPLIT2. The kernel does not know *where* a third boundary
goes; the cap was hiding that, not causing it.

### 13.3 Why criterion (ii) earned its keep — a genuine instrument conflict

Three instruments, three answers, on the same arm:

| instrument | says |
|---|---|
| **merged per-part target** (item 2) | **better**: parts with no distinct reco object 6 → 5, `q_extra` −0.98 pt, median part `q_f1` +0.030 |
| **π⁰ census** | **worse**: exact 35 → **34** |
| **boundary agreement** vs the owner's own per-part labels | **worse**: 2 of 3 changed objects moved down, and one is a SPLIT2 |

**Without §9.1's criterion (ii), (iii)'s −1 would have read as a win.** The
merged target is honest about what it measures — one more hand part received a
distinct object — but "a distinct object exists" is not "the boundary is right",
and only the owner's per-part labels can say the second thing. **A metric that
counts objects cannot grade boundaries.** That is the general lesson, and it is
the same shape as §6.5's: the census is 66 events and could not see the 9
suppressed cuts; the per-part count is 29 parts and cannot see where a boundary
went.

### 13.4 The verdict

**Do not raise `max_parts` to 3.** It stays 2. **4 was not tested** — the 3-arm's
direction (one gain, two regressions, one of them on a SPLIT2) argues against
spending an arm on it, but that is an argument, not a measurement. The k ≥ 3
question is **not** answered
by the cap and is now correctly posed for the first time: *the kernel cannot
place a third boundary*, which is a seeding/valley question (doc pr/138 §B3's
`max_seeds`, and §B7's no-valley class), not a cap question. It goes back on the
list as a kernel problem, priced at one clean gain and two regressions per three
objects touched.

---

## 14. Where this leaves the round, and what to do next

### 14.1 State

| | |
|---|---|
| **Production ON** | `onV1c90` + `shower_split` + `shower_split_em_start` (P1.3), toolkit `f5e17798` |
| **Ready to flip, owner's call** | **P1.6** — `shower_split_skip_shared` + `shower_split_max_impact = 30` |
| **Closed this session** | P1.4 / P3.3 the re-home (§12), item 4 the k cap (§13), P1.2 `b ≤ 12` (§6, withdrawn) |
| **Delivered this session** | P3.2 — the per-part completeness target (§11) |
| **No C++ or jsonnet changed** | all four items used existing default-OFF knobs; binary md5-identical to the pinned `pin-pr139b` throughout, so P1.0's 478/478 knob-off gate still covers this work |

### 14.2 The one decision waiting on the owner

**Flip `shower_split_skip_shared` + `shower_split_max_impact = 30`?**

*For*: census exact 35 → **36**; every one of the 39 scanned objects behaved
exactly as pre-registered; it reaches the same census as the withdrawn `b ≤ 12`
while suppressing **5** confirmed cuts instead of **11**; 0 ADVERSE; `q_extra`
6.7 → 6.9 %.

*Against, stated plainly*: it still suppresses **5 of the 19 cuts the owner
confirms**. Two of those are `skip_shared` deferring a peel it cannot make
cleanly (281485, 350354) and have a named successor (§8.4, co-ownership peel).
Three are the `b` bound rejecting cuts the owner says are right — 294174 ×2 and
415278/83139, all at `b` 48–209 cm — and the merged target prices one of them
(§11.4).

**A defensible alternative is `skip_shared` alone**: it takes 165157 and 281485
from `no-group`/`none` to `partial` on its own, suppresses **2** confirmed cuts
instead of 5, and leaves 54332's false fire standing. That trades one census
`exact` for three confirmed cuts. It is the owner's call which side of that trade
he wants, and both arms exist.

### 14.3 Recommended next steps, in order

1. **The co-ownership peel** — the named successor from §8.4, and now the single
   highest-value item. Instead of `skip_shared` refusing a peel whose segments
   two showers both own, **assign each shared segment to one part**. That would
   recover 281485's and 350354's cuts — 2 of the 5 suppressions — *without*
   re-creating the duplicate-charge pathology, and it is the only item on this
   list that removes a cost rather than trading one.
2. **The third-boundary problem, posed correctly** (§13.4). Not the cap: the
   kernel's seeding. 396222 (owner k=7) and 406125 (owner k=2, wrongly cut into
   3) are the two counterexamples any candidate must satisfy, and both now have
   hand boundaries to score against.
3. **Widen the per-part label set.** Every conclusion above about boundaries
   rests on **19** confirmed cuts and **29** hand parts, and §13.3 shows that is
   already the binding constraint — the same shape of limit §6.5 found for the
   census. The splitter fires 51 times per pass on 239 events; scanning ~30 more
   of those objects per part would roughly double the resolving power of every
   instrument in §13.3.
4. **The remaining false fire** 278420/61027 (`b` 26.78) — one object, worth a
   feature hunt only if items 1–3 leave it isolated.

**Not next**: any further sweep of `b`. §6.3 priced the whole dial and §10
confirmed the prediction to the object; searching it again is fitting to 39
labels.

---

## Repro — §8 to §14

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# sec 8 -- the prediction, from the owner's labels ALONE, committed 2e996db6
#          BEFORE any arm below was run
python3 scripts/pr140_prereg.py                  # -> docs/pr/pr140-prereg.tsv

# the three arms, each knob ALONE on the flipped production config
./scripts/pr140_on_arms.sh                       # on / onrh15 / onk3
for t in on onrh15 onk3; do ./scripts/pr140_score.sh $t; done

# sec 10 -- criterion 4: did the shipped C++ do what was pre-registered?
python3 scripts/pr140_tape_verify.py --arm work-pr140r1-on     # 39/39, rc=0

# sec 11 -- the merged per-part target.  Fork-fidelity FIRST, on a real arm:
( cd em_display
  ./em117_score.py --tag emscan-0827 --manifest em117-139onemst98-manifest.tsv \
      --prepdir emprep-139onemst > /tmp/a.txt
  ./em140_score.py --tag emscan-0827 --manifest em117-139onemst98-manifest.tsv \
      --prepdir emprep-139onemst > /tmp/b.txt
  diff /tmp/a.txt /tmp/b.txt; echo "fork-fidelity rc=$?" )   # rc=0 over 41 lines
# the baseline under BOTH metrics (sec 11.2), then each arm's delta
( cd em_display
  ./em140_score.py --split-tag splitscan-0902-pi0 --tag emscan-0827 \
      --manifest em117-139onemst98-manifest.tsv --prepdir emprep-139onemst \
      --tsv ../docs/pr/pr140-perpart-base-98.tsv
  ./em140_score.py --split-tag splitscan-0902-pi0 --tag emscan-0828-agent5 \
      --manifest em114c-139onemst141-manifest.tsv --prepdir emprep-139onemst \
      --tsv ../docs/pr/pr140-perpart-base-141.tsv )
for t in on onrh15 onk3; do
  python3 scripts/pr140_perpart.py --arm $t --base base \
      --tsv docs/pr/pr140-perpart-delta-$t.tsv
done

# sec 12 -- the re-home tape, and the row-level proof that the merged target
#           sees exactly what the single target sees
grep -h "SHOWER_SPLIT rehome" work-pr140r1-onrh15-*/pr_evt*/stdout.log \
  | grep -oP 'verdict=\K\w+' | sort | uniq -c        # 40 orphan, 11 REHOME

# sec 13 -- boundary agreement vs the owner's per-part labels, two arms
python3 scripts/pr140_k3.py --arm work-pr140r1-onk3 --base work-pr140r1-onrh15
```

**Binary**: pinned `/home/xqian/tmp/pin-pr139b`, md5 `fbff08ec…` verified equal
to `local/lib/libWireCellClus.so` at the start *and* the end of the session. **No
C++ or jsonnet was changed in §8–§14** — all four items ride existing default-OFF
knobs, so P1.0's 478/478 knob-off gate still covers them.

---

# Session 3 — the four §14.3 next steps

## 15. Item 1 — the co-ownership peel: the design died, a better one replaced it

### 15.1 A defect in this doc's own instrument, found first

Before any of §14.3 could be measured: **`WCT_SHOWER_CONTENT_DEBUG` was never set
on any doc pr/138 or pr/139 arm.** `prep_em_scan.py --parse-probes` therefore
wrote **zero** sidecars (`em_display/emprep-138*`, `emprep-139*`, `emprep-140*`
are all empty directories), and `em117_score`/`em140_score` silently fell back to
the **dump join** — `segments[].shower_id`, which is single-valued and so credits
a segment held by two showers to exactly one of them.

The scripts even print a lossiness line, and it read `loses 0 member(s)` — which
is **vacuously** true when there is no sidecar to compare against. Another gate
over an empty set, the same shape as §11.1's fork-fidelity slip.

Priced on `work-pr136-onV1c90-*`, the last arms that **do** carry sidecars
(239 events each), scoring the identical arm both ways:

| 141-set, same arm | with sidecar | with the dump join |
|---|---|---|
| median `q_f1` | **0.904** | 0.918 |
| `q_miss` | 2.007e+07 | 2.019e+07 |
| `q_extra` | 2.673e+07 | 2.609e+07 |

**On that arm the join is systematically optimistic** — about 1.4 pt of median
`q_f1` and 2.5 % of `q_extra`. Shared membership is precisely the phenomenon
items 1 and 2 are about, so §16 re-runs every arm with the probe on and
restates. **§16.1 is the result of that restatement, and it is that nothing
moved** — read it before treating anything in §10–§13 as provisional.
**Unaffected**: the π⁰ census and the boundary-agreement numbers of §13 read the
dumps and the debug tape, not the sidecar, so nothing in §6, §10.2's census row
or §13 moves.

### 15.2 The design in §8.4 is dead, and the pr136 sidecars killed it before any C++

§8.4 proposed: *"assign each shared segment to one part"* — i.e. drop the
co-owned members and peel the remainder. The three components `skip_shared`
refuses are the entire population, and they are:

| event | node | owner | component | **co-owned** | exclusive charge left |
|---|---|---|---|---|---|
| **281485** | 89095 | **SPLIT2** | 4 seg | **4 of 4** (all held by 91112) | **0** |
| 165157 | 9000 | **KEEP** | 7 seg | 2 of 7 | 0.749 |
| 350354 | 18092 | **SPLIT2** | 12 seg | 1 of 12 | 0.590 |

So the proposed design is **a no-op on 281485** — the one event it was written
for, with nothing left to peel — and on 165157 it would peel a 0.749-charge
remainder, **making a cut on an object the owner labels KEEP**. It buys one
confirmed cut and pays a false fire. Withdrawn before implementation.

### 15.3 What replaces it: the **shed**

281485's structure is the tell. Its four members are held by shower **91112** as
well, and P1.4's re-home tape independently found the same host at
**gap 0.00 cm** (`dau=91111 parent=89095 host=91112`). The charge does not need
a new home — **it already has one.**

> **The shed**: when `skip_shared` refuses a component and **every** member of it
> is also owned by another shower, detach the component from this parent and
> build **no daughter**. The owner's cut is honoured; no duplicate object is
> created; no charge moves anywhere it was not already.

Knob `shower_split_shed_shared`, **DEFAULT OFF**, inert unless
`shower_split_skip_shared` is also on. The partial-sharing case stays refused,
for the measured reason in §15.2.

### 15.4 Pre-registered, before the arm  *(committed with this section)*

The `on` arm's tape carries **exactly three** shared-member refusals over all
239 events, so the prediction is per-object and complete, not statistical:

| object | owner | prediction |
|---|---|---|
| 281485/89095 part 1 | SPLIT2 | **SHED** |
| 165157/9000 part 0 | **KEEP** | still refused |
| 350354/18092 part 1 | SPLIT2 | still refused |
| **`n_shed` over the manifest** | | **1** |

**Failure conditions, fixed now:** 165157 sheds (it is a KEEP — shedding removes
members the owner says belong); any other of the 39 scanned objects changes
state; `n_shed` > 1; any shower created with `kine_charge` 0; census exact < 36;
`q_extra` > 7.2 %; any ADVERSE mover.

The sharing counts in §15.2 come from the **pr136** sidecars — a different arm,
membership has moved. That is why the new binary **prints `nshared` on every
refusal**: `scripts/pr140_shed_verify.py` holds the arm to the prediction using
the arm's own report, not the pr136 estimate.

---

## 16. The knob-off gate, and the round-2 arms

Two C++ knobs were added this session — `shower_split_shed_shared` (§15) and
`shower_split_max_seeds` (§17). Both **DEFAULT OFF / DEFAULT 4**, so the shipped
path must be untouched, and it is:

```
gate mcp1k   rc=0 :: events in A: 66   compared archives: 132  missing/unpaired: 0  PASS
gate mcp2k   rc=0 :: events in A:106   compared archives: 212  missing/unpaired: 0  PASS
gate ncpi0   rc=0 :: events in A: 19   compared archives:  38  missing/unpaired: 0  PASS
gate nuecc48 rc=0 :: events in A: 48   compared archives:  96  missing/unpaired: 0  PASS
```

**478 / 478 byte-identical**, `work-pr140r2-off-*` (new binary, all knobs at
their defaults) against `work-pr139r3-flipchk-*` (the shipped production config
on the previous binary). doctest **2631** assertions with both new defaults
pinned. Binary pinned at `/home/xqian/tmp/pin-pr140r2`, md5 `5d176a30…`.

**The sidecar defect of §15.1 is fixed at source**: every round-2 arm runs with
`WCT_SHOWER_CONTENT_DEBUG=1` and `prep_em_scan.py` now writes **239 sidecars,
0 warnings** (it wrote 0 for every pr/138 and pr/139 arm). Output names also now
carry the arm round — a round-2 re-run of tag `on` was about to overwrite
round 1's tracked TSVs and manifests and make §10–§13 unreproducible.

---

## 17. Item 2 — RESULT: the third boundary is a SEEDING problem, and the cap is hardcoded

§13.4 reposed the k ≥ 3 question as the kernel's seeding rather than
`shower_split_max_parts`. That is now measured rather than asserted.

### 17.1 The seed cap binds on three quarters of everything the splitter does

`pr138_angular_maxima()` takes `max_seeds = 4` as a **hardcoded default
argument** (`NeutrinoShowerClustering.cxx:5494`), never a knob. On the tape:

| population | n | `n_seed == 4` | distribution |
|---|---|---|---|
| all candidates | 393 | 178 (**45 %**) | 1:66, 2:81, 3:68, **4:178** |
| **fired** candidates | 51 | 39 (**76 %**) | 2:4, 3:8, **4:39** |

**And on all four objects the owner cut into k ≥ 3 — whose owner `k` values are
3, 3, 5 and 7 — `n_seed` sits at exactly 4.**

A four-seed finder **cannot express k = 7**, or k = 5. So the k ≥ 3 population is
capped *upstream* of `shower_split_max_parts`, and §13's arm — which moved only
that knob — could only ever redistribute seeds the kernel already had. **§13's
result is correct and its interpretation was incomplete**: raising the cap made
the third cut in the wrong place *because the third seed was the best of four,
not because a third boundary is unfindable.*

### 17.2 `valley_best` does not separate, again

The same join kills the obvious alternative. Over the owner's confirmed cuts the
C++ boundary-matches **exactly** (agreement 1.000), `valley_best` spans
**0.012 → 0.940**; over the five false fires it spans **0.000 → 0.638**. The
distributions are on top of each other. This is the third independent time
(§7, doc pr/138 §B7) that the valley has failed as a discriminator, and it is
now settled rather than suspected.

### 17.3 What ships: `shower_split_max_seeds`, DEFAULT 4

The cap becomes a knob so the question can be asked at all — default 4, key
suppressed at 4, so the shipped kernel is byte-identical (§16's gate covers it).

**Not answered this session, deliberately.** Doc pr/138 §B3 warned that
`max_seeds` cannot move without moving the *trigger*, and it is right: more seeds
mean more candidate pairs, hence more acceptances, hence a different fire set.
Grading that needs labels on the seed-capped population, and **there are four**.
That is §19, and it is why §19 exists.

---

## 18. Item 4 — RESULT: the last false fire is not separable

At the P1.6 operating point 15 objects fire: **14 owner-confirmed cuts and one
false fire, 278420/61027.** The only question a single object can support is
whether it is an *outlier* — strictly outside the range of all 14 — on any
quantity the tape carries. Nothing is fitted; doc pr/138 §A5.4 spent a holdout
on exactly that mistake.

| feature | 278420 | confirmed-cut range | verdict |
|---|---|---|---|
| `vchi2` | **9.339** | 1.586 – 8.105 | outside, margin **1.234** |
| `b` | 26.78 | 0.31 – **26.35** | outside, margin **0.43 cm** |
| `q_bal` | 0.9345 | 0.035 – **0.9301** | outside, margin **0.0044** |
| `q_small_frac` | 0.4831 | 0.034 – **0.4819** | outside, margin **0.0012** |
| `nseg`, `npts`, `Q`, `n_seed`, `valley`, `angle`, `nacc`, `vgap`, `q_small`, `fwd_min`, `rvtx_max`, `conn_max` | | | **inside** |

**Three of the four margins are under 1 %** — that is not a separator, that is
the definition of "the most extreme one". And `b`'s margin of **0.43 cm is
smaller than the 0.5 cm offline-vs-C++ agreement spread** (§8.3): tightening the
bound to catch it is not even resolvable by the instrument that would set it.

The fourth, `vchi2` = the **main vertex fit's** reduced χ², is the only real gap
— and it is a property of the **event**, not the object, so a cut on it would
veto every candidate in that event. Recorded as an observation, not a lead.

**Verdict: 278420/61027 is not distinguishable by more of the same information.**
The honest next move is more labels, not another feature — which is §19.

### 16.1 The correction pass: the §15.1 defect changed **nothing**

The probe is now on and the sidecars exist (239, 0 warnings), so §15.1's defect
can be priced on **this** population instead of the pr136 one. Same arm
(`work-pr140r2-off-*`), scored with the sidecars that were missing and without:

| baseline, both metrics | with sidecars | with the dump join (what §10–§13 used) |
|---|---|---|
| `q_miss` single-target | 16.7 % | 16.7 % |
| `q_extra` single-target | 6.7 % | 6.7 % |
| median shower `q_f1` | 0.922 | 0.922 |
| `q_miss` merged per-part | **11.1 %** | 11.2 % |
| `q_extra` merged per-part | **8.8 %** | 8.6 % |
| hand parts with no distinct reco object | **6** | 6 |
| lossiness (members the join drops) | **0, measured** | 0, vacuous |

**Nothing in §10–§13 is provisional.** The reported lossiness is now genuinely
zero rather than vacuously zero: on these 239 events at the production config
the two membership sources agree exactly, and the only residual is a 0.4 % shift
in `sum q_target` because the sidecar carries its own per-member `dQ` instead of
re-summing `points[].dQ`. That is well inside the precision anything was quoted
to.

So the honest statement is narrower than §15.1's first draft: **the probe was
genuinely missing and should always have been on** — the pr136 arms show the two
sources *can* differ by 1.4 pt of median `q_f1` — **but on the population doc
pr/139 actually measures, it makes no difference.** The defect is fixed, the
numbers stand, and this is now known rather than assumed.

---

## 19. Item 3 — the wider per-part label set, delivered **scan-ready**

The labels are the owner's; they cannot be produced by this session. What can be
produced, and is: the set, the loadability proof, and the brief
(`docs/pr/pr140-scan-brief.md`). **This item is scan-ready, not done**, and it is
stated that way rather than left looking finished.

### 19.1 Three measurements, one wall

| section | what it could not resolve | the limiting sample |
|---|---|---|
| §13.3 | three instruments, three answers on the `max_parts = 3` arm | the tie-breaker is boundary agreement: **19 confirmed cuts, 29 hand parts** |
| §17 | whether raising `max_seeds` helps | **4** labelled objects are seed-capped at k ≥ 3 |
| §18 | whether the last false fire is separable | **1** negative against 14 positives |

Not a physics wall. Roughly thirty more per-part verdicts about double the
resolving power of all three at once.

### 19.2 The set — `docs/pr/pr140-scan-set.tsv`, 32 objects

Stratified over the pr/137 curated set (known loadable by
`split_model.load_object`) **minus** the 39 already labelled in
`splitscan-0902-pi0`, leaving a pool of 136:

| stratum | n | available | decides |
|---|---|---|---|
| **S3-unjudged-fire** | 8 | 13 | trigger purity — the splitter peels these today and nobody has judged them |
| **S1-seed-capped** | 8 | 57 | §17's `max_seeds` arm; these are the objects it will change |
| **S2-bound-region** | 8 | 30 | the `b ≤ 30` bound, on a sample **not chosen by** the census |
| **S4-control** | 8 | 112 | seeded-random (20260901), chosen **independently of every feature the other strata select on** |

S4 is not decoration. Without it the whole set is selected by the hypotheses it
is meant to test, which is the trap `feedback_blind_the_scan_sheet` records.

**All 32 were verified to load** through `split_model.load_object` before the
brief was written. The pr/139 scan served the owner **181** objects instead of 23
because that check was skipped; it is now a step, not a hope.

```
python3 scripts/pr140_scanset.py           # -> docs/pr/pr140-scan-set.tsv
./split_display/serve_split_display.sh 5022 \
    --scan-tag splitscan-0903-wide --set docs/pr/pr140-scan-set.tsv
```

The tag is fresh (M13). Nothing will be flipped on these labels without showing
the owner the arm first.

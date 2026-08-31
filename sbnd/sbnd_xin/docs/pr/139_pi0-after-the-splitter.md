# doc pr/139 — π⁰ reconstruction after the splitter: the round plan **and tracker**

**Status: LIVING.** **`shower_split_em_start` is SBND PRODUCTION ON as of
2026-08-31** (owner flip; flip-equivalence gate `work-pr139r3-flipchk-*` vs
`work-pr139r1-onemst-*` **478 / 478 byte-identical**). The owner's scan is **DONE** (2026-09-01,
39 objects, tag `splitscan-0902-pi0`) and **it overturned this round's own
recommendation** — see **§6**, and **§3ter** for the next session's order. This file is the tracker for a multi-session round. §1 is the
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
| **P1.1** | shared-membership peel guard | `shower_split_skip_shared` | **DONE — works, modest** | `work-pr139r1-onshared-*` | 3 peels refused, the **`kine=0` daughter gone**, `none` 3→2, exact **35**, q_extra 6.7→6.9 %, 0 ADVERSE |
| **P1.2** | impact-parameter veto @ 12 cm | `shower_split_max_impact` (cm, 0 = off) | **MEASURED WRONG AT 12 — do not flip** (§6) | `work-pr139r1-onb12-*` | exact 35 → 36 and all four π⁰s recovered, **but the owner's scan says it suppresses 9 of the 19 cuts he confirms**, including 281485 whose cut he calls correct |
| **P1.6** | **`skip_shared` + `max_impact` = 30** — the operating point the scan supports | `shower_split_skip_shared` + `shower_split_max_impact` | **PROPOSED, NEXT SESSION'S FIRST ARM** | — | predicted exact **36**; on the fresh labels eff **0.737** / pur **0.933**, **3** confirmed cuts suppressed instead of 9 |
| **P1.3** | daughter EM start segment (μ-typing) | `shower_split_em_start` | **SBND PRODUCTION ON** (owner flip 2026-08-31) | `work-pr139r1-onemst-*`; flip-equivalence `work-pr139r3-flipchk-*` **478 / 478** | μ-typed daughters **11 → 2**, **461 MeV** of EM energy restored (×1.657 confirmed), 51 peels / **0 backwards**, every instrument unchanged, 0 ADVERSE |
| **P1.4** | re-home the orphan daughter | `shower_split_rehome`, `…_rehome_gap` | **DONE — inert at 4 cm, alive at 15** | `work-pr139r1-onrehome-*` (4 cm), `work-pr139r2-onrh15-*` (15 cm) | 6/51 → **12/51** re-homed; census **35** either way; **cannot be graded until P3.2** |
| **P1.5** | the combination P1.1 + P1.2 + P1.3 | three knobs | **DONE** | `work-pr139r2-oncomb-*` | exact **36**, q_miss 14.5 %, q_extra 7.6 %, q_f1 **0.932**, μ-typed daughters **1**, 0 ADVERSE |
| **P2** | owner scan, split display port 5022 | — (tag `splitscan-0902-pi0`) | **DONE 2026-09-01 — 39 objects** | `em_labels/splitscan-0902-pi0/`, `docs/pr/pr139-scan-verdicts.tsv` | 20 KEEP / 19 SPLIT; trigger **eff 1.000** / pur 0.792; boundary SPLIT2 median **1.000**; **it overturned P1.2** (§6) |
| **P3.1** | π⁰ re-seat BEFORE the splitter | `pi0_reseat_before_split` | NOT STARTED | — | — |
| **P3.1b** | scope dial `max_vgap` (comparison arm only) | `shower_split_max_vgap` | NOT STARTED | — | superseded by P1.2 unless P1.2 fails |
| **P3.1c** | a pointing test **not** measured from the vertex | — | NOT STARTED | — | the one unexplored feature family |
| **P3.2** | joint label set (re-mark the touched showers) | — | **PART DELIVERED** — 19 objects now carry a per-part segment assignment, 20 are KEEP | `em_labels/splitscan-0902-pi0/` | still to do: **merge** these into the completeness target so `em117_score` reads per-part |
| **P3.3** | re-home, **re-sized** after P1.3/P1.4 land | `shower_split_rehome` (same knob) | BLOCKED ON P1 | — | — |
| **P3.4** | split-aware π⁰ pairing | `pi0_split_aware_pairing` | NOT STARTED | — | **no longer the fix for 281485** — see §2 |
| **C1** | k ≥ 3 splitting | — | **MEASURED SHORT** | doc 138 §B3 | 0.635 → 0.756 vs target 0.85 |
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

# doc pr/141 — the final round: what is left after the splitter campaign

**Status: CLOSED (sessions 1-2, 2026-08-31).** Session 2 (§16-§21) worked
§15.3's five hand-off items and closes the campaign; the census reads **36 of 66
exact** and production is unchanged.

**Session 1** follows. doc pr/139 §26 closed the splitter
sub-campaign and ranked what remains (§26.5). The owner asked for one last round
against that list:

> *"I would like to act one last round '1. The k=3 kernel recursion … 2. The 8
> missing γs … 3. A hand scan of the 7 badly-wrong masses … 4. Cheap leftovers:
> the pdg=211 EM-only restriction and 278420's separability'. I assume you want
> me to do a scan, can you serve them in port 5022 first, so that I can provide
> you the results while you act on the others. Note, the goal is to have this one
> last round of campaign tonight for improvements."*

**Why 141 and not 140.** The `pr140_*` scripts belong to doc pr/139's sessions
2–4 (§8–§26); doc number 140 is left unused so a script prefix always names the
doc that owns it.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# item 3 -- the MEASUREMENT: run the finder's own pair tape on the nine events
#   (stderr-only, byte-neutral; binary pinned /home/xqian/tmp/pin-pr140r4)
LD_LIBRARY_PATH=/home/xqian/tmp/pin-pr140r4 WCT_PI0_PAIR_DEBUG=1 PR_JOBS=4 \
  ./run_pr_chain_batch.sh work-mcp1k-grp0825 work-pr141dbg-pair-mcp1k data 168432 286655
#   ... and pair-mcp2k (280159 71872), pair2-mcp1k (348691 409634 283713),
#       pair2-mcp2k (397630), pair2-ncpi0 (21073)
python3 scripts/pr141_pairtape.py --tsv docs/pr/pr141-pairtape.tsv

# item 3 -- energies, alternative partners (geometry model SUPERSEDED by the tape)
python3 scripts/pr141_massfail.py --tsv docs/pr/pr141-massfail.tsv

# item 3 -- the owner's scan (port 5022, tag pi0mass-0904-owner)
./em_display/serve_em_display.sh 5022 --scan-tag pi0mass-0904-owner \
    --manifest em_display/em141-massfail9-manifest.tsv \
    --prepdir em_display/emprep-140r2off
```

Arm under scan: `work-pr140r2-off-*` — the **shipped production** configuration
(`onV1c90` + splitter, every doc pr/139 knob off), the arm whose census is
35/66 exact.

---

## 1. Item 3 — the nine mass failures, measured with the finder's own tape

### 1.1 A correction I made and then had to retract

The first pass through this item modelled the finder's geometry offline.
`id_pi0_with_vertex` builds a per-γ direction (`local_dirs`,
`NeutrinoShowerClustering.cxx:7553-7625`): `get_init_dir()` when the shower's
start vertex *is* the candidate vertex (`:7572`, `:7586`), the candidate-vertex →
start ray otherwise (`:7620`). I read the dump's `start_vertex_id`, matched it
against the main vertex id, concluded that six of the nine events take the
*axis* branch, and reported that two events (168432, 280159) were therefore
inside the (100, 160) window and had never been mass failures at all.

**That was wrong.** The dump's `start_vertex_id` is not the quantity `get_svc()`
compares against `cand_vtx`. Running the finder with `WCT_PI0_PAIR_DEBUG=1` on
168432 prints, for the owner's pair:

```
PI0_PAIR P1 pair sh1=22006 sh2=49028 ct1=2 ct2=2 vtx=19001 E1=241.0 E2=115.8 m=53.5
```

— the **ray** branch, m = 53.5, which is exactly the `mass_vertex_convention` the
census used. Across the nine events the finder's mass at the main vertex equals
the label's `mass_vertex_convention` in **7 of 9** (the two exceptions are the
events whose label used a non-main π⁰ vertex). **doc pr/139 §26.3's
classification was right and my re-read of it was not.**

The lesson is the one this campaign keeps re-learning: *a rule read off the
source is a hypothesis.* The probe already existed
(`WCT_PI0_PAIR_DEBUG`, `NeutrinoShowerClustering.cxx:106`), costs one event, and
emits no bytes. `scripts/pr141_massfail.py` is kept for its energy and
alternative-partner columns, with the geometry model marked superseded;
`scripts/pr141_pairtape.py` is the measurement.

### 1.2 What the finder actually did, per event

Nine events, `WCT_PI0_PAIR_DEBUG=1`, production binary (pin
`/home/xqian/tmp/pin-pr140r4`, md5 `28e2b85a…` = `local/lib` at run time), arms
`work-pr141dbg-pair*-{mcp1k,mcp2k,ncpi0}`. Energies are the finder's own
`get_kq` = `kine_charge`.

| event | the owner's pair, as the finder scored it | why it is not the reco's π⁰ |
|---|---|---|
| 21073 | m = 190.5 at main | out of window (high) |
| 168432 | m = 53.5 | out (low); θ from the vertex is only 18.4° |
| 280159 | m = 73.6 | out (low) |
| 286655 | m = 205.4 (at 79033) / 226.1 (at main) | out (high); `80055` also fails the 30° assoc test at main (48.6°) |
| 348691 | m = 276.9, θ = 100.8° | out (high) |
| 409634 | m = 77.4 | out (low) |
| **397630** | **m = 114.4 — IN WINDOW**, at candidate vertex 15000 | **outranked**: K24's main-first tiering took `19010`+`15047` at m = 111.5 **at the main vertex** |
| **71872** | **no pair row at all** | **admission**: `64044` failed the 30° association test at all three candidate vertices (73.3°, 70.4°, 38.7°) |
| **283713** | **m = 99.9** at candidate vertex 17004 | **the window edge**: the tape's own line reads `pi0 window reject: best mass=99.9 MeV delta=-25.1` against a bound of −25.0 |

So the answer to §26.5 item 3, before any scan: **six of the nine are genuine
mass failures; three are not, and each of the three has a different named
mechanism.**

### 1.3 The three mechanisms, and what each is worth

**283713 — a π⁰ lost by 0.1 MeV.** The acceptance test is
`-25 < m - 135 + offset < 35` with `m_pi0_mass_offset = 10` (C++ default and
production), i.e. (100, 160) MeV. The best pair in the event is the owner's, at
**99.9**. `delta = -25.1`. Nothing else was accepted.

This is a knife edge on a *production-wide* constant: an offset of 10.2 instead
of 10.0 admits it, and would move every other event in the sample too. **That is
CLAUDE.md §5.1 territory — changing an existing knob's default — so it is
reported, not tuned.** What makes it worth reporting rather than filing away is
that it is a *one-sided* miss: the lower bound is −25 and the upper is +35, so
the window is already asymmetric about the peak, and the campaign's own
measurement (doc pr/135: the in-window peak sits at 134.7 at fudge 0.86, i.e.
self-consistent) gives no reason for the asymmetry to sit exactly where it does.
The scan settles whether `47021` — 25 MeV, 8.2 cm, **one segment** — is a real γ
at all; if it is not, there is nothing here.

**397630 — the main-vertex preference costs a π⁰.** The owner's pair is in
window at 114.4, at candidate vertex 15000. It loses to `19010`+`15047` at
111.5 at the **main** vertex, because `pi0_prefer_main_vertex` (K24, SBND
production ON 2026-08-31, on the owner's own instruction *"if there is a
direction ambiguity … the preference should give to neutrino vertex"*) ranks any
in-window pair at main above any non-main pair **outright**, before the |Δ|
comparison runs. On the legacy comparator the owner's pair would have won: its
key is |114.4 − 125| − 6 = 4.6 (the 6 MeV bonus for two detached members,
`:7770`) against 13.5 for the pair that was taken. **That last sentence is
INFERRED from reading the comparator, not measured** — the same move §1.1
retracts. It is checkable in one event: `pi0_prefer_main_vertex` is a top-level
jsonnet arg, so a single-event TLA override plus the same tape settles it.

This is the rule working as specified, not a defect — but it is **the first
measured case of K24 costing a π⁰**, and doc pr/134 shipped it on a census that
went 32 → 33. Whether the trade is still right is an owner question, and the
scan answers the input to it: is `15047` (73 MeV, attached at the ν vertex) a γ,
or is `33038` the true partner?

**71872 — an admission failure, with a knob already written for it.** `64044`
(23.0 MeV, 12.4 cm, typed 211) never entered the pool: the disconnected-shower
association test needs the shower's own direction within
`pi0_assoc_angle_deg = 30°` of the vertex→start ray, and `64044` measured
73.3° / 70.4° / 38.7° at the three candidate vertices. doc pr/132 K9 named this
exact shower as the specimen for `pi0_crumb_assoc_max` — "a low-energy crumb's
15-cm PCA direction is noise" — and that knob is **default off and has never
been flipped**. If the scan says `64044` is the γ, K9 is a live, already-built
candidate; if it says `13008` (86 MeV, which the finder paired at 129.4) is the
γ, K9 stays dead and the reco is simply right.

### 1.4 What the six real mass failures still need

Two structural facts hold for the six, from the tape and the dumps:

1. **Five of the nine events carry a reco π⁰ that shares a γ with the owner's
   pair** — 21073 shares *both*, each re-paired with a different partner
   (`60081`+`34027` → 106.1 and `63100`+`11008` → 102.0). The finder is not
   failing to find a π⁰; it is choosing a different partner.
2. **The three events with no reco π⁰ at all** (286655, 409634, 283713) are the
   ones whose partner γ is smallest: `69032` is 36 MeV / 8.7 cm / 2 seg and
   `47021` is 25 MeV / 8.2 cm / **1 seg**.

Direction of the error splits them cleanly: **too high** — 21073 (190.5),
286655 (205.4), 348691 (276.9), all needing the charge product × 0.19–0.50, which
is where a grossly over-clustered γ would show; **too low** — 168432 (53.5),
280159 (73.6), 409634 (77.4), needing × 1.8–12.6, which at these energies is an
angle, i.e. a start-point, problem. `60081` (634.5 MeV over **188.7 cm and 39
segments**) and `12119` (452.9 MeV over 118.5 cm and 33 segments) are the two
over-clustering candidates the scan can settle by eye.

### 1.5 The scan

Served on port 5022, tag `pi0mass-0904-owner`, brief in
`docs/pr/pr141-scan-brief.md`, one question per event. **`em_display`, not
`split_display`** — the question is the π⁰ *pair*, which is `em_display`'s PI0
mode; `split_display` answers "cut this object in two", the question doc pr/139
closed. The tag is fresh and pre-seeded with a copy of each event's existing
label (base wins over overlay, the same precedence `pr132_pi0_census.py` uses);
originals untouched — M13.

---

## 2. Item 3, continued — the owner's scan, and the two mechanisms it names

The owner scanned all nine on port 5022 (tag `pi0mass-0904-owner`, 2026-08-31
18:48–19:06 local) and reported:

> *"The pi0 generally are OK. There are cases where the gamma of pi0 is close to
> the detector boundary, so we miss some energies. There are also other cases,
> where there are likely overclustering leading to overestimation of energy thus
> overestimating the pi0 mass."*

Four events came back **re-paired**. Scored by `scripts/pr141_ownerscan.py`:

| event | the owner's pairing now | m (vertex) | |
|---|---|---|---|
| 283713 | **`23011` + `47021`** (was `67051`+`47021`) | **123.5** | **IN WINDOW** |
| 348691 | **`50073` + `52089`** (was `49046`+`50073`) | **141.8** | **IN WINDOW** |
| 71872 | `79074` + `64044`, unchanged | **101.1** | **IN WINDOW** |
| 409634 | `21002` + **`27015`** (was `69032`) | 96.5 | 3.5 MeV below the edge |
| 21073 | `60081` + **`16019`** (was `63100`) | 182.6 | out (high) |
| 168432, 280159, 286655 | unchanged | 61.6 / 73.6 / 226.1 | out |
| 397630 | — | — | **lost to a seeding defect of mine, see §2.3** |

### 2.1 The boundary claim, made quantitative — and it holds

`pr141_ownerscan.py` measures, per γ, **how much room the shower had**: the
distance from its start point *along its own axis* to the SBND active-volume
wall (x, y ∈ ±200 cm, z ∈ 0…500 cm, taken from the point clouds themselves).
It then corrects each γ's energy by the contained fraction of the PDG
longitudinal profile (`dE/dt ∝ (bt)^{a−1}e^{−bt}`, b = 0.5,
t_max = ln(E/E_c) − 0.5, E_c = 32.8 MeV, X₀ = 14.0 cm) and re-masses the pair.
This is a textbook average profile, not a fit to SBND — it is used only to ask
whether the **order** of the missing energy matches the room available.

Masses below are on **one scale throughout**: the pair mass the *arm* gives,
`√(4·kine_charge₁·kine_charge₂·sin²(θ/2))` with the label's θ (a pure geometry
number). The label's own mass is not usable for this table — it carries whatever
energy scale and hypothesis the scan used, which differs between rows the owner
re-saved and rows they did not.

| event | γ short of room | room | contained f | ARM mass → corrected |
|---|---|---|---|---|
| **168432** | `22006` **1.4 X₀** (its furthest member point is *outside* the volume, wall −0.2 cm) and `49028` 3.5 X₀ | 19.0 / 49.2 cm | 0.21 / 0.72 | **53.5 → 138.4  IN WINDOW** |
| **71872** | `79074` 3.1 X₀ | 43.5 cm | 0.58 / 0.93 | **107.3 → 146.8  IN WINDOW** |
| 280159 | `95114` 3.6 X₀ (furthest point 2.6 cm from the wall) | 50.9 cm | 1.00 / 0.72 | 73.6 → 87.0, still out but moving the right way |
| the other six | none under 5 X₀ | 135–492 cm | ≈ 1.00 | unmoved (≤ 6 MeV) |

**The correction is selective, which is what makes it evidence rather than a
fudge**: it fires only on the three events whose geometry says it should, moves
two of them into the window, and leaves the other six within 6 MeV of where they
were. The owner's boundary mechanism is confirmed, and it is *not* a clustering
defect — it is charge that never existed in the detector.

### 2.2 The over-clustering claim — the same three objects the scan notes name

The events whose mass is too **high** carry no containment excuse (135–492 cm of
room, f ≈ 1.00) and every one of them has an oversized γ:

| event | the object | |
|---|---|---|
| 21073 | `60081` — **634.5 MeV over 188.7 cm, 39 segments** | the largest object in the set |
| 286655 | `79023` — the scan note: *"over-clustered, and its axis is pointing backwards … ten of its eleven members sit at 128–150° to its own axis"* | brought in by `walk_add` and `pass4_angle`, not the cone |
| 280159 | `12119` — the scan note: *"OVERCLUSTERED / merged … it sits ON the vertex … reads as a vertex blob seeded on the muon stem, not a photon"* | 452.9 MeV, 118.5 cm, 33 seg |

So the two mechanisms the owner named are both real, they act in **opposite
directions on the mass**, and they partition the nine cleanly.

### 2.3 A defect of mine, stated

The fresh tag was seeded from each event's **base** label. For 286655, 348691
and 397630 the base label has `pio: null` and the pairing lived in the overlay
tag `pi0scan-0829-agent` — the same base-wins precedence
`pr132_pi0_census.py` uses, applied to the wrong field. So those three opened
with **empty γ slots**, contrary to what the brief promised. The owner
re-created the pairing for two of them; **397630 was left unpaired and its scan
is lost.** Its pre-scan reading stands from the tape (§1.2): in window at 114.4
at candidate vertex 15000, outranked by K24. *Rule for the next seeded tag: seed
from the record that holds the field being scanned, not from the record that
wins in general.*

---

## 3. Item 2 — the eight "missing γ" events, and they are not missing

doc pr/139 §26.3 named this the largest single block — *"a γ is simply NOT
RECONSTRUCTED, 8 events, upstream γ-finding efficiency — nothing to do with
splitting"* — and §26.5 item 2 asked for a scoping pass before more clustering
work.

**"Absent-on-arm" in `pr132_pi0_census.py` means only that the label's γ shower
id is not a key of the arm's `showers[]`.** That is three situations wearing one
name. `scripts/pr141_missing_gamma.py` separates them by **segment id** — the
label stores the γ's member segments, the arm's `segments[]` carries the same
ids, and the probe sidecar says which shower owns each one:

**Proximity alone is not evidence**, and an early version of this script treated
it as if it were. A host shower only accounts for the γ if it can *hold the
charge*: the classifier requires the host's `kine_charge` to be at least half the
γ's label energy, and separates a host of **comparable** energy and segment count
(the same object under a new id — label staleness, no defect) from a
substantially larger one (a genuine over-merge).

| verdict | n of 9 γ slots | meaning |
|---|---|---|
| **MERGED** | **3** | every member segment on the arm, inside a substantially larger shower — a genuine over-merge |
| **MERGED-BY-PROXIMITY** | 1 | 142421: no member list, but a 1197.6 MeV shower starts 8.5 cm from a 706.5 MeV γ |
| **RENAMED** | **2** | 347824 (528.0 → 416.4 MeV, 23 segments both) and 506114 (44.8 → 53.2 MeV) — the same object, new id |
| **UNACCOUNTED** | **2** | 259542 ×2: a **993.0 MeV** γ whose nearest start is a **50.8 MeV proton stub**, and a 138.7 MeV γ whose nearest is 42.9 MeV |
| NO-CANDIDATE | 1 | 396222 g2 — nothing within 20 cm |

**So "8 events where a γ is simply NOT RECONSTRUCTED" is wrong, but the
replacement is not "8 over-merges" either.** Exactly **one** slot has no
candidate at all; two are id renames with no defect behind them; **three** are
genuine over-merges (and one of those, 76346, is a 5.0 MeV crumb absorbed by a
238 MeV shower — real, but worth nothing); two are unaccounted for by any test
this script can make offline. The defensible statement is narrower and still
useful: **the census was reading id mismatches as missing particles, and the
class it named "γ-finding efficiency" is at most one event.**

| event | the γ | where its charge is now |
|---|---|---|
| 281485 | `84070`, 73.4 MeV, 5 seg | **all 5 segments inside `87078`** — 149.4 MeV, 53.0 cm, 12 seg. Absorbed into an object twice its size. |
| 347824 | `24112`, 528.0 MeV, 23 seg | 22 of 23 inside `107060` — 416.4 MeV, 103.3 cm, 23 seg. Comparable size: a **rename**, not an absorption. |
| 142421 | `7010`, 706.5 MeV | nearest start `108104` at 8.5 cm — **1197.6 MeV over 202.8 cm, 47 segments**. Absorbed into something 1.7× bigger. |
| 506114 | `89100`, 44.8 MeV, 2 seg | both inside `82080` — 53.2 MeV, 4 seg. Rename-scale. |
| 76346 | `40030`, 5.0 MeV, 1 seg | inside `14059` — 238.1 MeV, 67.2 cm. A crumb absorbed by a large shower. |
| 71178 | `80039`, 63.1 MeV, 4 seg | 1 seg in a 0.2 MeV shower, 1 unowned, 2 not on the arm. **Fragmented.** |
| 259542 | `17004` 993.0 MeV / `131129` 138.7 MeV | nearest starts 11.6 cm (a 50.8 MeV proton stub) and 5.4 cm (`131131`, 42.9 MeV). Charge present, ownership scattered. |
| 396222 | `128276`, 162.7 MeV | nearest start 20.5 cm and 3.3 MeV. **The only slot with no candidate** — the one real "not reconstructed". |

### 3.1 What this does to doc pr/139's closing conclusion

doc pr/139 §26.4 closed the campaign on the sentence *"the dominant remaining
blockers are missing γs and γ energy, not γ clustering"* and recommended
stopping. **Tonight's two measurements reverse the first half of that.**

| doc pr/139 §26.3 said | what it actually is |
|---|---|
| 8 — a γ is NOT RECONSTRUCTED | **1** (396222) has no candidate; 2 are id **renames** with no defect; **3** are genuine over-merges (one a 5 MeV crumb); 2 unaccounted. |
| 9 — mass outside the window | 3 are **containment** (calorimetry, irreducible); 3 are **over-clustering** (clustering); 2 are **pairing** the owner has now revised; 1 is a window edge. |

The residual π⁰ population is **not** dominated by γ-finding efficiency — that
class is at most one event. What replaces it is mixed: some over-merging, a
calorimetric containment loss no clustering change can recover, and a
non-trivial share of **label staleness in the campaign's own instrument**, which
had been reading id mismatches as missing particles. That last part is a finding
about the metric, not about the reconstruction, and it is worth as much.

**This does not re-open the splitter.** Every object named above is *bigger*
than the γ the owner marked — `87078` swallowing `84070`, `108104` at 1197.6
MeV, `14059` at 238.1 MeV. That is the over-merge direction, which is the
splitter's and the absorber-guard's territory, not the seed-count kernel's; and
doc pr/139 §22 measured the splitter's own configuration space exhausted **on
the objects the owner labelled**, which are a different set from these eight.

---

## 4. The one knob-level lead the scan produced: K20's 40 cm bound

§1.3 read 283713 as *"a π⁰ lost by 0.1 MeV of window"* — the best pair in the
event was `67051`+`47021` at m = 99.9 against a bound of 100.0. **The owner's
scan supersedes that reading.** They re-paired it to `23011`+`47021`, m = 123.5,
comfortably in window. So 283713 is not a window-edge case at all; the finder
never had the owner's pair to consider.

Why not is on the tape, in one line:

```
PI0_PAIR K20 mu-reject sh=23011 why=trackish len=57.3
```

`23011` is **typed pdg = 13** (168.4 MeV, 57.3 cm, 11 segments). A μ-typed
object reaches the disconnected π⁰ pool only through K20
(`pi0_admit_muon_showers`, **SBND production ON** since 2026-08-30,
`NeutrinoShowerClustering.cxx`), whose rule is

```cpp
shower_ish = get_flag_shower() || (total_length < 40*units::cm && seg_dir_weak(ss))
```

`23011` carries no shower flag — that is *why* it was typed 13 — and it is
**57.3 cm long**, so it fails the 40 cm arm and is refused as "trackish". Raising
that bound to ~60 cm would admit it and give the finder a pair at 123.5.

**This is a lead, not a flip.** It is *n* = 1, and this campaign has twice
refused to move a threshold on one example (doc pr/139 §18 on one negative,
§22.5 on two non-EM fires). The 40 cm number came from the file's own
"shower-ish muon" idiom, not from a scan, so there is no measured basis for
either 40 or 60. What makes it worth recording above the other leads is that it
is **the only one tonight where a shipped, production-ON knob has a named
threshold with a measured counterexample on the far side of it** — everything
else needs new code or a new instrument.

**Recommended next step for it**, in the discipline this campaign settled on:
build a targeted sample of μ-typed objects in the 40–80 cm band that sit near a
π⁰-plausible geometry, scan ~15 of them, and only then choose a bound. That is
the same shape as §22.5's `pdg = 211` lead (item 4), and the two should be one
sample: **both are questions about which particle *types* may enter the π⁰
pool**, and both currently rest on n ≤ 2.

---

## 5. Item 1 — the k = 3 recursion is measured dead, and it kills a family

§26.5 item 1 proposed: *"cut a part only if that part itself has an accepted
seed pair, rather than asking for three seeds at once."* The reasoning was that
every k ≥ 3 variant doc pr/138 tried picks more seeds out of **one** angular
density map built around the whole object, so a third core that is invisible in
that map stays invisible however cleverly the seeds are chosen. Re-seeding each
part on its own points is the one thing none of those variants did.

`scripts/pr141_kernel_recurse.py` (fork of `pr138_kernel_k.py`, M10) tests it
offline against every owner split label the campaign holds — the three owner
tags pooled, later scans winning on conflict: **45 SPLIT objects, 10 of them
k ≥ 3**. **No new parameter**: the recursion reuses the shipped trigger's own
two numbers (valley ≤ 0.95, charge fraction ≥ 0.03) and the frozen seed-finding
constants, and recursion depth is capped at 1. So a win would not have been a
fit — and neither is the loss.

| class | n | shipped kernel | recursion |
|---|---|---|---|
| SPLIT2 | 35 | med 1.000 mean 0.943 **exact 28/35** | med 1.000 mean 0.934 **exact 21/35** |
| SPLIT3 | 8 | med 0.601 mean 0.618 exact 0/8 | med 0.601 mean 0.618 **exact 0/8 — identical** |
| SPLIT4+ | 2 | med 0.465 | med 0.746 |
| ALL | 45 | mean 0.864 exact 28/45 | mean 0.870 **exact 21/45** |

**It costs seven exact SPLIT2 boundaries and buys nothing on SPLIT3.** Eight of
the 35 SPLIT2 objects get over-cut (k 2 → 3), five of them dropping from a
perfect 1.000. The single gain is one SPLIT4+ object — 396222/9059, k 2 → 4,
agreement 0.079 → 0.641, still not exact, n = 1.

### 5.1 The mechanism, and why it generalises

The SPLIT3 column is not merely worse — it is **byte-identical to the shipped
kernel**. On all eight SPLIT3 objects `k_rec == k_ship`: **the recursion never
fired.** Re-seeded on its own points, with its own centroid and its own
bandwidth, not one part of one of those objects produced an accepted seed pair.

That is a stronger and more useful statement than doc pr/139 §13's *"raising
`max_parts` places the third cut badly"*:

> **The third core is not visible to this seed finding at any scope** — not in
> the whole object's angular density, and not in the part's own. Changing which
> seeds are picked, or where the map is built, cannot reach it.

So the family is closed: `max_parts = 3` (§13), `max_seeds` (§22.1), the greedy
and pair-then-grow variants (doc pr/138 K1–K7) and now recursion all fail for
the same reason. A k ≥ 3 fix needs a **different observable**, not a different
search over the same one.

And four of the eight SPLIT3 objects have `k_ship = 1` — **the shipped trigger
does not fire on them at all**. So a third of the k ≥ 3 shortfall is not a
boundary problem in the first place; those objects are not being split even
twice.

### 5.2 Verdict

**Item 1 is closed negative.** No C++ was written, no arm was spent, and the
result is stronger than a null: it names the observable as the limit rather
than the search. Cost: one script.

---

## 6. Item 4 — the two "cheap leftovers", and the one that turned into a mechanism

§26.5 item 4 held two leads that were both under-powered: the EM-only
(`pdg = 211`) splitter restriction at n = 2, and 278420/61027's separability at
n = 1. §4 above added a third of exactly the same shape — K20's 40 cm bound at
n = 1. `scripts/pr141_typeset.py` builds the targeted sample all three need, over
the whole production population (`work-pr140r2-off-*`, **239 events**), and
prices each stratum *before* anyone scans it.

### 6.1 S-MU — and the 40 cm bound is not the binding constraint

μ-typed (|pdg| = 13), `conn_type` 2 or 3 (so: the disconnected pool), ≥ 30 MeV,
length in the 40–80 cm band that K20 refuses. For each, pair it with every EM
shower in its event under the finder's own vertex-ray geometry and ask whether
**any** pairing lands in (100, 160).

| | |
|---|---|
| candidates in 239 events | **6** |
| in-window under the finder's own energy (`kine_charge`) | **0 of 6** |
| in-window under the **shower-hypothesis** energy | **3–4 of 6** — m = 133.6, 126.3, 123.5 at ×1.906, and 102.6 which falls out at ×1.66 |

**The energy hypothesis, not the length bound, is what blocks these.** A μ-typed
object's stored `kine_charge` is the **track**-hypothesis energy — a different
recombination and a different fudge — and the π⁰ finder prices pair masses with
`get_kq()` = `get_kine_charge()` at `:7707`. On the one object where both
numbers exist, 283713's `23011`, the dump says **168.4 MeV** and the scan
label's shower hypothesis says **320.9 MeV**, a ratio of 1.906 (the viewer's own
note quotes 1.66 for the same effect, so it is per-object; 1.906 is used here
only to *screen*). **The ratio is not a global constant and the
other-hypothesis energy is not on disk for the unscanned five**, so the count is
quoted as a range: 4 of 6 at ×1.906, 3 of 6 at ×1.66.

That is why §4's reading needs amending. Raising K20's 40 cm bound to 80 cm
would admit all six of these — and the finder would then compute their pair
masses at the **track** energy and put **none** of them in the window. The knob
would fire and change nothing. K20 already re-stamps accepted members EM
(`PI0_PAIR restamp sh=… -> pdg 11` on the tape) — but that happens **after**
acceptance, so it cannot rescue a pair that the window rejected on a
track-priced mass.

**The real question this raises**, stated for the owner rather than answered:
should a μ-typed object admitted to the π⁰ pool be **priced under the shower
hypothesis at pairing time**, the same way it is re-typed after? The measured
prize is bounded and concrete: **4 candidate π⁰ in 239 events** that reach the
window under the shower hypothesis and none under the track one. That is not a
knob to flip — it changes how an existing quantity is computed, i.e. CLAUDE.md
§5.1 — but it is the first lead of the round with a population, a mechanism and
a number.

The six objects are `docs/pr/pr141-typeset.tsv` (S-MU), scan-ready.

### 6.2 S-PI — the EM-only restriction stays under-powered

π-typed (|pdg| = 211), ≥ 3 segments, ≥ 100 MeV: the population an EM-only
splitter restriction would exclude.

| | |
|---|---|
| candidates in 239 events | **8** |
| of which `conn_type == 1` (attached at the ν vertex) | 7 |
| includes §18's standing false fire | yes — 278420 `18002`, 268.3 MeV, 47.3 cm, 5 seg |

Eight objects across the whole population, seven of them attached at the vertex.
§22.5's n = 2 becomes n = 8 as a *sample* — enough to scan, and it is the right
sample — but it does not become evidence on its own. **The recommendation is
unchanged: scan these 8 before restricting the splitter by type.** They are in
`pr141-typeset.tsv` (S-PI), which also resolves item 4's second half: 278420's
object is in the set, so its separability gets scanned alongside rather than as
a special case.

---

## 7. Where this round leaves the campaign

### 7.1 What was measured tonight

| item | verdict |
|---|---|
| **1 — k = 3 recursion** | **dead, with a mechanism that closes the family**: on all 8 SPLIT3 objects the recursion never fires; the third core is invisible to this seed finding at *any* scope (§5) |
| **2 — the 8 "missing γs"** | **not missing**: at most 1 of 9 absent γ slots has no candidate. 3 are genuine over-merges, 2 are id renames (label staleness), 3 unaccounted. Not γ-finding efficiency (§3) |
| **3 — the 7 badly-wrong masses** | **partitioned by mechanism**: 3 containment (measured, correcting them moves 2 into the window), 3 over-clustering, 2 re-paired by the owner into the window, 1 window-edge (§1, §2) |
| **4 — the type leftovers** | the μ-typed lead **re-diagnosed**: the blocker is the track-hypothesis energy, not the 40 cm bound, with a 4-π⁰ prize; the `pdg = 211` lead is now an 8-object sample, still unscanned (§6) |

### 7.2 The correction this round owes doc pr/139

doc pr/139 §26.4 closed the campaign on *"the dominant remaining blockers are
missing γs and γ energy, not γ clustering"*. **The first half is wrong and the
second half is half right.**

- "missing γs" — at most **1**, not 8. The rest splits between genuine
  over-merges (3), id renames that are artifacts of stale labels (2) and slots
  no offline test can account for (2+1) (§3).
- "γ energy" — real, and now split into two very different things: a
  **containment** loss that no clustering change can recover (§2.1), and an
  **over-clustering** inflation that is a clustering defect (§2.2).

So the residual π⁰ population is a mix — **γ identity** (over-merges plus label
staleness), **detector containment** (irreducible), and **pairing/admission**
(small, but carrying the round's only bounded prize). No single class dominates,
and in particular the campaign is **not** blocked on upstream γ-finding the way
§26.4 concluded.

### 7.3 What I recommend next, in order

1. **Price μ-typed π⁰ candidates under the shower hypothesis at pairing time**
   (§6.1). It is the only lead tonight with a mechanism, a population and a
   number: 3–4 candidate π⁰ in 239 events, **0** of which the current pricing can
   reach. It changes an existing computation, so it needs the owner's word
   before any code — but it is a bounded, default-OFF-able change.
2. **Scan the 14-object type set** (`pr141-typeset.tsv`, 6 S-MU + 8 S-PI). It
   settles item 4's `pdg = 211` restriction, 278420's separability and the
   μ-typed question in one pass, and every one of those has been stuck at
   n ≤ 2 for three rounds.
3. **The three over-merges from §3** — `87078` swallowing a whole γ, `108104` at
   1197.6 MeV over 202.8 cm. A small set, and *not* the objects the splitter
   campaign was tuned on, so doc pr/139 §22's "configuration space exhausted"
   does not settle them — but three objects is a lead, not a programme. Worth
   pairing with a **label-refresh pass**: two of the nine slots were renames, so
   some of what the census reports as failure is the label store aging out.
4. **A containment correction to γ energy** (§2.1) is the only route to the
   three boundary events, and it is a calorimetry change, not a clustering one.
   Worth scoping, but it is a different subsystem and a different validation.

**Not recommended**: any further work on the split kernel's seed search (§5.1
closes it), and any threshold moved on tonight's n = 1 evidence.

---

# Session 2 — acting on §7.3's three recommendations

Owner, 2026-08-31: *"can you continue Recommended next, in order: (1) price
μ-typed π⁰ candidates under the shower hypothesis at pairing time … (2) scan the
ready-made 14-object type set … (3) the three over-merges, paired with a
label-refresh pass. for tonight"* — and, on the C++: *"Please proceed this one,
if you need further rounds, please feel free to proceed."*

## 8. Recommendation 1 — three knobs, all DEFAULT OFF, and what they measure

| knob | what it does | default | state |
|---|---|---|---|
| `pi0_mu_shower_hypothesis` (**M1**) | inside `id_pi0_with_vertex`, price a μ-typed, no-shower-flag object under the **shower** recombination + fudge | `false` | **regression, do not flip** (§9.1) |
| `pi0_mu_shower_max_len` (**M2**) | K20's "shower-ish muon" length bound, exposed as a knob | `40 cm` = the shipped literal | **inert, do not flip** (§9.2) |
| `pi0_mu_shower_hyp_min_len` (**M3**) | length floor below which M1 leaves the track price alone | `0` = no floor | **moot** (§9.3) |

**M1 is exact arithmetic, not an estimate.** `kine_charge_from_maps` ends in
`overall / recom_factor / fudge_factor * w_value` (`NeutrinoEnergyReco.cxx:188`),
and the collected charge is hypothesis-independent, so the same charge under the
other hypothesis is a **global constant**:
(0.87 × 0.95) / (0.58 × 0.86) = **1.657** at the SBND production factors.

> **A correction to §6.1.** That section quoted the ratio as 1.906, "measured on
> 283713's `23011`". It was not a ratio at all: `23011`'s label carries
> `energy_marks_delta = 152.48` — **charge the owner had hand-added as `in`
> marks**. The label's `energy_other_hypothesis` field, which *is* the
> hypothesis change, reads 280.0 against `energy_as_reconstructed` 168.4, i.e.
> **1.663** — the config value. §6.1's "3–4 of 6" is therefore **3 of 6**.

**Verification, all four bars met:**
- knob-off gate `work-pr140r4-off-*` vs `work-pr141r1-off-*`: **PASS 478/478
  archives byte-identical, `missing/unpaired events: 0`** on all four samples;
  repeated on the final M1+M2 binary (`work-pr141r1-off2-*`): **PASS 478/478**.
- `./build/clus/wcdoctest-clus`: **2639 assertions, 0 failed** (2635 → 2637 →
  2639 as M1, M2, M3 landed; each pins its default).
- freshness proof before every arm: `local/lib/libWireCellClus.so` newer than the
  last source edit; binaries pinned `/home/xqian/tmp/pin-pr141{,b,c}`
  (md5 `0c2e53f1…`, `1f92dc70…`, `430ffa3e…`), arms run under
  `LD_LIBRARY_PATH` on the pin.
- **compiled-config proof, the real one**: a bare `wcsonnet` of the job emits no
  `TaggerCheckNeutrino` knobs at all (the component's config node only appears
  once the runner's TLA set selects the PR stages) — so grepping it would have
  been vacuous, exactly as doc pr/129 warned. The proof used instead is the
  runner's **own** compiled config, `.wct-cfg-evt283713.json`:
  `"pi0_mu_shower_hypothesis" : true` and `"pi0_mu_shower_max_len" : 80`.

## 9. Recommendation 1 — RESULTS. All three are negative, and the last one is the interesting one

### 9.1 M1 alone is a regression, and it breaks the π⁰ K20 was shipped to rescue

Arm `work-pr141r1-on-*` (M1 alone, flipped production config):

| | exact | partial | no-group | none |
|---|---|---|---|---|
| off | **35** | 16 | 12 | 3 |
| M1 on | **34** | 17 | 12 | 3 |

Six archives differ, and **not one is in the six-object scan set**: 166870,
348691, 393212 (mcp1k), 174771, 282271, 493659 (mcp2k). The census moves on
exactly one: **166870, `exact` → `partial`**.

The tape gives the mechanism in four lines. Off:

```
PI0_PAIR K20 mu-admit sh=85045 E=38.6
PI0_PAIR P1 accept sh1=85045 sh2=87058 vtx=10060 m=116.9
```

On:

```
PI0_PAIR M1 mu-hyp sh=85045 E=38.6 -> 63.9 (x1.657)
PI0_PAIR K20 mu-admit sh=85045 E=38.6
PI0_PAIR P1 accept sh1=85045 sh2=10074 vtx=10060 m=138.8
PI0_PAIR P1 accept sh1=10013 sh2=87058 vtx=10008 m=124.9
```

**Re-pricing moved the greedy partner choice.** With `85045` at 63.9 MeV a
*different* partner (`10074`) lands nearer 135 than the true one, so the correct
pair `85045`+`87058` is displaced and `87058` is consumed by a second pairing.
The event this happens on is **166870 — the specimen doc pr/133 cites as K20's
justification** (*"166870 true pair m=109.1 accepted"*).

So M1's live population is not the one §6.1 reasoned about at all: it is the
**short (< 40 cm) objects K20 already admits**, whose track pricing is part of
the operating point that was validated when K20 shipped.

### 9.2 M2 is inert — and the length bound was never the blocker either

Arm `work-pr141r1-onboth-*` (M1 + M2 at 80 cm): census **34 / 17 / 12 / 3** —
identical to M1 alone. **Neither 283713 nor 350354 gained a π⁰**, and the
pre-registered "exactly three new acceptances" did not happen: there were
**zero**.

The tape says why, and the compiled-config proof is what makes it conclusive:

```
"pi0_mu_shower_max_len" : 80          <- the value DID arrive
PI0_PAIR M1 mu-hyp sh=23011 E=161.7 -> 267.9 (x1.657)
PI0_PAIR K20 mu-reject sh=23011 why=trackish len=57.3
```

`23011` is 57.3 cm, the bound is 80 cm, and it is **still refused**. K20's test is

```cpp
shower_ish = get_flag_shower() || (total_length < m_pi0_mu_shower_max_len && ss && seg_dir_weak(ss))
```

so with the length arm satisfied and no shower flag, the only remaining term is
**`seg_dir_weak(ss)`** — and it is false. The object has a well-defined
direction.

> **The binding constraint is the reco's own direction evidence, not the length
> bound and not the energy hypothesis.** §6.1's headline — *"the energy
> hypothesis, not the length bound, is what blocks these"* — is **wrong**, and
> it was wrong because it was derived from an offline screen rather than from
> the admission code. It is the third time this round that reading the source
> and reading the tape gave different answers, and the tape won every time.

### 9.3 M3 is moot, and stays in as the record of a design that was not needed

`pi0_mu_shower_hyp_min_len` was built to scope M1's re-pricing to what M2 newly
admits, so the legacy < 40 cm population keeps the price it was validated at.
The design is sound and it is the right fix for §9.1's regression — but M2
admits **nothing**, so there is nothing for it to scope. It ships DEFAULT OFF,
gate-covered, unmeasured.

### 9.4 The pre-registration failed, and the failure is the useful part

§`pr141-prereg-m1m2.md`, committed before the arm: *"exactly three new π⁰ …
in 283713, 350354 and 392901 … no event outside these six should move."*

**Measured: zero new π⁰, and six events outside the set moved.** Both halves
wrong. The prediction was built on an offline model of *which objects K20
admits* (length only) when the code requires length **and** direction weakness,
and on the assumption that M1's population was the 40–80 cm band when it is the
< 40 cm one.

The bar the pre-registration set — *"recommend only if the arm produces exactly
the three predicted acceptances"* — is not met, so **M1, M2 and M3 are all
recommended OFF**, which is where they ship.

### 9.5 What the owner's scan settles regardless

Class purity is **3 of 6** (283713 `23011`, 350354 `18009`, 122660 `54071` are
γs; 392901 `23017`, 280159 `90098`, 294174 `25030` are tracks). So even if the
direction test were relaxed enough to admit this class, **half the admissions
would be tracks** — and one of the three in-window candidates
(392901 `23017`, m = 124.6) is one of them, i.e. it would have manufactured a
**false π⁰**. A relaxation worth flipping would have to separate the 3 from the
3, and nothing measured tonight does that.

`122660/54071` is worth recording separately: an **owner-called electron shower
typed `pdg = 13`**, 170 MeV under the shower hypothesis, with no in-window
partner. It changes no π⁰, but its energy enters `kine_reco_Enu` at the **track**
price. That is a real defect in a different subsystem, found and not fixed here.

## 10. Recommendation 3 — the label-refresh pass, and it is worth one π⁰

`scripts/pr141_label_refresh.py` asks, for each of the nine events the owner
rescanned: does the reconstruction's own accepted π⁰ match the **old** hand pair
or the **new** one?

| event | old hand pair | owner's new pair | what the reco accepts | |
|---|---|---|---|---|
| **348691** | 49046+50073 | **50073+52089** | **50073+52089** | **FLIPS TO EXACT** |
| 21073 | 60081+63100 | 16019+60081 | 11008+63100; 34027+60081 | no match |
| 283713 | 47021+67051 | 23011+47021 | none | no match |
| 409634 | 21002+69032 | 21002+27015 | none | no match |
| 71872, 168432, 280159, 286655 | unchanged | unchanged | ≠ | no match |
| 397630 | 19010+33038 | — (lost, §2.3) | 15047+19010 | — |

**348691's reconstruction has been right all along.** The census called it
`partial` against a hand pair the owner has now superseded; under the refreshed
label the reco's `50073`+`52089` at 141.8 MeV **is** the answer.

So the shipped census is **35/66, and 36/66 once one label is refreshed** — a
free π⁰, with no code change. `pr132_pi0_census.py` resolves labels
**base-wins**, which is right for an overlay that only *extends* the
denominator and wrong for a rescan that *corrects* an earlier call. Changing
that precedence edits the scientific record, so it is the owner's call
(CLAUDE.md M13), not something this round does.

Together with §3's two RENAMED γ slots, that is now **three** independent places
where the instrument, not the reconstruction, is what fails.

## 11. RECOMMENDED PRODUCTION CONFIGURATION

> **Keep production exactly as it is. Flip nothing.**
>
> `onV1c90` + `shower_split` + `shower_split_em_start` + the shipped π⁰ chain
> (`pi0_admit_muon_showers`, `pi0_prefer_main_vertex`, `pi0_mass_offset = 10`,
> `kine_shower_fudge_factor = 0.86`) — unchanged from doc pr/139 §23.

Every knob this round added is measured and stays OFF:

| knob | verdict | evidence |
|---|---|---|
| `pi0_mu_shower_hypothesis` (M1) | **do not flip** | census 35 → 34; breaks 166870, K20's own justifying event, by moving the greedy partner choice (§9.1) |
| `pi0_mu_shower_max_len` (M2) | **do not flip** | inert — compiled-config proof shows 80 arrived and the object is still refused; `seg_dir_weak`, not length, is the gate (§9.2) |
| `pi0_mu_shower_hyp_min_len` (M3) | **do not flip** | moot while M2 admits nothing (§9.3) |

**The one change that pays, and it is not code**: refresh 348691's π⁰ label to
the owner's 2026-08-31 pairing. Census **35 → 36 of 66**. It is a records
decision (M13), so it needs the owner's word, not a knob.

## 12. Where the campaign stands after tonight

### 12.1 The two sessions, in one table

| question | answer |
|---|---|
| k = 3 recursion (§5) | dead — the recursion never fires; the third core is invisible to this seed finding at any scope |
| the 8 "missing γs" (§3) | at most **1** is missing; 3 over-merges, 2 renames, 3 unaccounted |
| the 7 bad masses (§1, §2) | 3 containment (irreducible), 3 over-clustering, 2 re-paired by the owner, 1 window edge |
| μ-typed admission (§9) | all three knobs negative; the gate is the **direction** test, and class purity is 3/6 |
| the type set `pdg = 211` (§6.2, §14) | **closed negative**: 4 of 7 are confirmed cuts, so the EM-only restriction would forfeit them; and the splitter fires on only 1 of the 4 |
| label staleness (§10) | worth **+1 exact** on its own, and it explains 2 of the 9 "missing" γs |

### 12.2 What I would do next, in order

1. **Refresh the π⁰ label store** (§10). It is the only measured gain left that
   costs nothing: +1 exact today, and it removes a class of false failure that
   has been inflating every census this campaign has run. Needs an owner
   decision on precedence — *newest scan wins* — not code.
2. ~~Scan the 8 `pdg = 211` objects~~ — **done, §14.** The EM-only restriction
   is refuted; the finding that replaced it is that the splitter under-fires on
   π-typed objects (efficiency 0.250 vs 0.938 on EM).
3. **`122660/54071`** (§9.5) — an owner-called electron shower typed `pdg = 13`
   whose energy enters `kine_reco_Enu` at the track price. Not a π⁰ question; a
   **PID-and-energy** question, and the first concrete specimen of one.
4. **Stop adding knobs to the π⁰ admission path.** Tonight spent three on it and
   all three are off. The measured blockers are now, in order: γ **identity**
   (over-merging), detector **containment**, and **label staleness** — none of
   which is reached by another admission threshold.

### 12.3 The method lesson, stated once

Three times tonight a conclusion read off the source was overturned by the
finder's own tape: the `local_dirs` branch (§1.1), the "energy hypothesis is the
binding constraint" claim (§9.2), and the ×1.906 ratio (§8). `WCT_PI0_PAIR_DEBUG`
costs one event, emits no bytes, and would have caught all three before they
were written down. **Run the tape before modelling the code.**

---

## 13. Review corrections to §8–§12

Three things a review pass caught after §8–§12 were pushed. None reverses a
verdict; all three are recorded because two of them were stated more strongly
than the evidence supported.

### 13.1 The "35 → 36" label refresh, now measured by the census rather than by hand

§10 derived the flip by comparing the owner's new pair against the dump's
`pio_id` groups. That is not the census, and the census is what produces the 35.
Re-run properly — `pr132_pi0_census.py … --overlay-tag pi0mass-0904-owner` on
`work-pr141r1-off-*`:

```
348691   labelsrc=overlay   partial (m 297.6)  ->  exact (m 141.8)
```

**Confirmed, and by the census itself.** But the run also shows the denominator
moving 66 → 52, because the script has a *single* `--overlay-tag` slot: naming
the owner's rescan **displaces** `pi0scan-0829-agent`, and 14 events keep their
pairing only there. So the two totals are not comparable head-to-head.

On the **52 rows both censuses cover**, exactly one row changes and nothing
regresses:

| | exact (of the 52 shared rows) |
|---|---|
| baseline labels | 32 |
| owner's rescan | **33** |

So **35 → 36 of 66 stands**, as 35 + 1 with a measured +1 and zero collateral —
not as a single census run. A genuinely refreshed census needs a *two-overlay,
newest-wins* precedence the script does not have, which is one more reason the
refresh is an owner decision about the record rather than a flag anyone flips.

### 13.2 M3's off-path now has its own gate arm

§8's table presented all three knobs under one gate result, but both gate arms
ran on `pin-pr141b`, which carries M1 and M2 only. M3 landed afterwards in
`pin-pr141c` and was covered by an *argument* (`mu_hyp` returns early when
`!m_pi0_mu_shower_hypothesis`, so M3's line is unreachable with M1 off), not by
a measurement.

Arm `work-pr141r1-off3-*` on `pin-pr141c` (md5 `430ffa3e…`, all three knobs
present and off) vs `work-pr140r4-off-*`:

| sample | archives | missing/unpaired | |
|---|---|---|---|
| mcp1k | 132 | 0 | PASS byte-identical |
| mcp2k | 212 | 0 | PASS |
| ncpi0 | 38 | 0 | PASS |
| nuecc48 | 96 | 0 | PASS |
| **total** | **478** | **0** | **PASS** |

**All three knobs are now gate-covered on the binary that carries all three**,
and the round has three independent 478/478 knob-off gates (`pin-pr141`,
`pin-pr141b`, `pin-pr141c`).

A note on §13.3 while it is fresh: this arm exited **`rc=0` on all four
samples**, where the earlier ones did not. So the runner bug is *intermittent* —
a race, not a deterministic path — which is exactly what makes it dangerous to
read the exit code as a verdict.

### 13.3 A pre-existing runner bug that silently disables the failure check

Three arms this round exited `rc=1` on one sample while the batch summary read
`ok: N / failed: 0` and every output was present. The cause is one line:

```
./run_pr_chain_batch.sh: line 2151: _r: unbound variable
```

Under `set -u` the runner dies in its **final** loop — the doc pr/97 safety net
that re-derives per-event failures from `rc.txt` and "fails loudly", added
precisely because `batch_summary()` returns 0 as long as *any* event succeeded.
It aborts after the merge, so outputs are complete (verified: 239/239 events,
and the gates PASS 478/478), but **the check itself never runs**, and a real
failure would surface only as a bare `rc=1` with no event list.

**This is pre-existing, not from this round** — the same message appears in doc
pr/139's round-2 logs (`pr140-arm-work-pr140r2-off-mcp2k.log`). Per CLAUDE.md it
is reported, not fixed here. It is the doc pr/127 shape exactly: a safety net
that has been dead for at least two rounds, and the only symptom was an exit
code nobody read.

### 13.4 Movers

`pr90_movers.py` on both knob-on arms, all four samples: **ADVERSE 0**, movers
> 0.05 cm: 0. No vertex moved.

---

## 14. Recommendation 2 — the EM-only restriction is REFUTED, and the real defect is the opposite one

Owner scan, 2026-08-31, tag `pisplit-0905-owner`, **all seven at confidence
`high`**:

| event | node | E | length | nseg | **owner** | shipped kernel |
|---|---|---|---|---|---|---|
| 396222 | `9084` | 175.3 MeV | 318.0 cm | 25 | **SPLIT3** | **FIRES**, then collapses to k=1 |
| 388 | `23028` | 722.6 MeV | 153.5 cm | 29 | **SPLIT2** | no fire |
| 278420 | `18002` | 268.3 MeV | 47.3 cm | 5 | **SPLIT2** | no fire |
| 181050 | `15005` | 113.9 MeV | 49.6 cm | 11 | **SPLIT2** | no fire |
| 163543 | `14123` | 402.6 MeV | 84.7 cm | 10 | KEEP | no fire |
| 406125 | `8059` | 144.7 MeV | 33.4 cm | 12 | KEEP | no fire |
| 499577 | `13030` | 106.0 MeV | 23.9 cm | 3 | KEEP | no fire |

### 14.1 The restriction dies on its own premise

doc pr/139 §22.5 built the EM-only proposal on this row:

> | not EM-typed | 8 | **1 confirmed cut** | 1 fire | purity **0.000** |
>
> "An EM-only restriction would remove that false fire and **cost no cut that
> fires today**."

Scanned properly, the π-typed population gives:

| | §22.5 assumed | measured |
|---|---|---|
| confirmed cuts | **1** of 8 | **4** of 7 |
| false fires (a KEEP that fires) | 1 | **0** |
| purity on the population | 0.000 | **1.000** — the single fire is on a confirmed cut |

**Every number that motivated the restriction is wrong, and in the direction
that kills it.** Restricting the splitter to `|pdg| = 11` would forfeit **four
owner-confirmed cuts** and remove **zero** false fires, because on this
population there are none. §22.5's "purity 0.000" came from a mixed 8-object
non-EM set in which the one fire and the one confirmed cut happened not to
coincide; it does not survive contact with the π-typed population itself.

> **Recommendation: do NOT restrict the splitter by particle type.** §7.3 item 2
> is closed, negative, on n = 7 at high confidence.

### 14.2 The defect that is actually there: the splitter *under*-fires on π-typed objects

The same table read the other way:

| | |
|---|---|
| efficiency on π-typed confirmed cuts | **1 of 4 = 0.250** |
| efficiency on the EM population (doc pr/139 §22) | **0.938** |
| purity on π-typed | **1.000** (0 false fires on 3 KEEPs) |

**The splitter is not firing wrongly on π-typed objects — it is barely firing at
all.** Three owner-confirmed cuts pass through untouched, including
**388 `23028`, a 722.6 MeV object 153.5 cm long with 29 segments**, and
**181050 `15005`**, whose owner boundary puts 9 of its 11 segments in the second
part. That is a *missed-cut* population — the exact opposite of the risk §22.5
was guarding against.

And the one object that does fire, **396222 `9084`**, fires and then collapses:
the accept passes but the bundle/centroid assignment puts every bundle in one
part (`k_kernel = 1`), which is doc pr/138 §B3's known failure mode. Its owner
verdict is **SPLIT3**, so it is also one of the k ≥ 3 cases §5 showed the seed
search cannot reach.

*(The agreement numbers on the three non-firing cuts — 0.202, 0.680, 0.187 —
are not boundary quality. With `k = 1` they are just the charge fraction of the
owner's larger part, i.e. the no-split baseline. Reported for completeness, not
as a score.)*

### 14.3 Why this population is invisible to the current trigger

Three of the four confirmed cuts produce **no accepted seed pair at all**.
Combined with §5 — where re-seeding each part also never fired on a SPLIT3 —
the pattern across both sessions is consistent enough to state plainly:

> **The seed-density trigger finds the cuts it was tuned on (EM, efficiency
> 0.938) and is close to blind outside them (π-typed, 0.250).** Neither more
> seeds (§5, doc pr/139 §22.1), nor a deeper search (§5), nor a type restriction
> (§14.1) changes that. A gain here needs a *different observable* — and that is
> now the third independent measurement pointing at the same conclusion.

### 14.4 What this does NOT license

It does not license widening the trigger. Purity on the EM population is 0.857
and the campaign has twice rejected knobs that bought efficiency at its expense
(doc pr/139 §22.3: 10 of 32 confirmed cuts for +1 π⁰). A wider trigger that
reached these four would have to be shown not to cost EM purity, and the 71-object
EM label set is the instrument for that — this 7-object set cannot answer it.

---

## 15. Final state, and what I recommend next

### 15.1 Everything doc pr/141 measured

| item | verdict |
|---|---|
| k = 3 recursion (§5) | **dead** — never fires on a SPLIT3; the third core is invisible to this seed search at any scope |
| the 8 "missing γs" (§3, §13.1) | **at most 1 is missing** — 3 over-merges, 2 id renames, 3 unaccounted |
| the 7 bad masses (§1, §2) | 3 **containment** (irreducible), 3 **over-clustering**, 2 re-paired by the owner, 1 window edge |
| μ-typed π⁰ admission — M1/M2/M3 (§8, §9) | **all three negative**; the gate is `seg_dir_weak`, and class purity is 3/6 |
| the `pdg = 211` restriction (§14) | **refuted** — it would forfeit 4 confirmed cuts and remove 0 false fires |
| label staleness (§10, §13.1) | worth **+1 exact**, and it explains 2 of the 9 "missing" γs |

**Nine knobs were priced across doc pr/139 §22 and doc pr/141. None is
recommended ON.** The production configuration is unchanged (§11).

### 15.2 The three things that are actually true now

1. **The splitter's trigger is population-specific.** Efficiency 0.938 on the EM
   objects it was tuned on, **0.250** on π-typed (§14.2), and 0/8 on the SPLIT3
   class (§5). Three independent measurements, one conclusion: a gain needs a
   different observable, not a different search over the same one.
2. **The residual π⁰ population is not one problem.** It is γ **identity**
   (over-merging), detector **containment** (irreducible), **label staleness**
   in the campaign's own instrument, and a small **admission** share — in that
   order. No single fix touches more than a few events.
3. **The instrument is now a first-order term.** Two of nine "missing γs" were id
   renames; one census row was scored against a superseded pairing; §22.5's
   whole premise came from a mis-scored 8-object set. That is three separate
   places where the measurement, not the reconstruction, was the failure.

### 15.3 Recommended next steps, in order

1. **Refresh the π⁰ label store** — the only measured gain left that costs
   nothing: **+1 exact today** (§13.1), and it removes a class of false failure
   that has been inflating every census this campaign ran. It needs an owner
   decision on precedence (*newest scan wins*) and a two-overlay census, not
   code. **This is the highest value-per-effort item on the list.**
2. **Audit doc pr/139 §22.5's non-EM table** — its "1 confirmed cut / 1 fire /
   purity 0.000" is now known wrong on the π-typed half, and the object it names
   as the non-EM false fire (`318769/31026`) reads `pdg = 11` on both arms
   (§`pr141-scan-brief-pi.md`). Anything else resting on that table should be
   rechecked before it is used.
3. **If the splitter is revisited at all, it is a TRIGGER problem on a new
   observable** — not more seeds, not recursion, not a type gate. §14.4 sets the
   bar: whatever reaches the four missed π-typed cuts must be shown not to cost
   EM purity (0.857), and only the 71-object EM label set can show that.
4. **`122660/54071`** (§9.5) — an owner-called electron shower typed `pdg = 13`,
   170 MeV, whose energy enters `kine_reco_Enu` at the **track** price. Not a π⁰
   question; a **PID-and-energy** one, and the first concrete specimen.
5. **Fix the runner's dead failure check** (§13.3) — `_r: unbound variable`
   kills the doc pr/97 safety net intermittently, so `rc=1` with `failed: 0` has
   been meaningless for at least two rounds. Small, standalone, and it restores
   a guard the campaign thinks it has.

**Not recommended**: any further knob on the π⁰ admission path (three tonight,
all off), and any threshold moved on n ≤ 2 evidence.

---

# Session 2 — the five hand-off items (2026-08-31)

The owner asked for §15.3's list to be worked end to end, closing the campaign:

> *"can you proceed to 1. Refresh the π⁰ label store … 2. Audit §22.5's non-EM
> table … 3. If the splitter is revisited, it's a trigger problem on a new
> observable … 4. 122660/54071 … 5. Fix the runner's dead failure check. … After
> this round, we are done for this campaign."*

and, on the one decision §15.3 said was theirs: **"YEs, newst scan wins."**

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# item 1 -- the refreshed census, denominator held at 66
python3 scripts/pr141_pi0_census2.py \
    --manifest98 em117-140r2off98-manifest.tsv \
    --manifest141 em114c-140r2off141-manifest.tsv --fudge 0.86 \
    --chain pi0mass-0904-owner,pi0scan-0829-agent \
    --tsv docs/pr/pr141-census-refresh.tsv

# item 2 -- the sec 22.5 audit
python3 scripts/pr141_em_audit.py --tsv docs/pr/pr141-em-audit.tsv

# item 4 -- the mu-typed PID set (served to the owner on port 5022)
python3 scripts/pr141_pidset.py --tsv docs/pr/pr141-pidset.tsv \
    --manifest em_display/em141-pidmu18-manifest.tsv
./em_display/serve_em_display.sh 5022 --scan-tag pidmu-0906-owner \
    --manifest em_display/em141-pidmu18-manifest.tsv \
    --prepdir em_display/emprep-140r2off
```

---

## 16. Item 1 — the label store is refreshed, and the census says 36 of 66

### 16.1 Why the naive rule would have been wrong

"Newest scan wins" cannot be applied per *file*. `pi0mass-0904-owner` holds nine
files, but `labels-evt397630.json` has **`pio: null`** — the seeding defect of
§2.3, where three events kept their pairing only in the overlay and opened with
empty γ slots. A file-level newest-wins would let that empty record beat
`pi0scan-0829-agent`'s `19010+33038` and **delete a census row**, moving the
denominator without anyone seeing it.

`scripts/pr141_pi0_census2.py` (a fork of `pr132_pi0_census.py`, M10 — the
production scorer stays byte-identical) therefore resolves **per field**: a tag
wins an event only where it actually carries a populated `pio.gammas`.

Two tags are **excluded from the chain although newer by mtime**:
`pi0mu-0905-owner` (19:40) and `pisplit-0905-owner` (20:24) are object-*type* and
SPLIT/KEEP scans; their `pio` blocks are agent seeds the viewer carries along,
not pairing verdicts. Mtime alone would have promoted them over the owner's own
rescan at 19:06. Provenance decides precedence, not the clock.

### 16.2 The result — the denominator holds and exactly one row moves

Chain `pi0mass-0904-owner → pi0scan-0829-agent → <base per set>`, on the shipped
production arm `work-pr140r2-off-*`:

| | shipped precedence | **refreshed** |
|---|---|---|
| hand π⁰ (denominator) | 66 | **66** |
| **exact** | 35 (53.0 %) | **36 (54.5 %)** |
| partial | 16 | 15 |
| none | 3 | 3 |
| no-group | 12 | 12 |

Row-by-row diff of the two TSVs: **one class change, no regressions.**

```
set141  evt348691   partial -> exact    labelsrc overlay -> pi0mass-0904-owner
```

and `397630` keeps its pairing, sourced from `pi0scan-0829-agent` — the per-field
rule doing exactly the job it was written for. The other 20 rows that change
`labelsrc` are the overlay being named explicitly plus the eight events the owner
rescanned whose pairing was unchanged; every one keeps its class.

**So §13.1's "35 + 1, measured separately" is now a single census run: 36 of 66,
denominator intact.** This is the only gain the campaign closes with, and it
required no code in `clus` and no arm.

---

## 17. Item 2 — the §22.5 audit, and the correction is mine, not doc pr/139's

### 17.1 The table reproduces exactly — from the tape

`scripts/pr141_em_audit.py` rebuilds the table from the `SHOWER_SPLIT cand` tape
on three arms. On all three, and joining **71 of 71** labelled objects with zero
join misses:

| candidate class | n | confirmed cuts | fires | purity | efficiency |
|---|---|---|---|---|---|
| EM-typed (\|pdg\| = 11) | 63 | 31 | 35 | **0.857** | 0.968 |
| not EM-typed | 8 | 1 | 1 | **0.000** | 0.000 |

**Identical to doc pr/139 §22.5.** The table was not mis-scored, and the 0.938 /
0.857 numbers §15.3 flagged as "unverified" are verified.

*But nothing in `scripts/` produces it and no committed TSV carries a `pdg`
column* — it was computed ad hoc. That is the reportable process defect, and it
is why re-deriving it took a script rather than a `grep`.

### 17.2 The pdg SOURCE is the whole question, and I read the wrong one

The tape prints `shower->get_particle_type()` **at the split site**
(`NeutrinoShowerClustering.cxx:5919`). The dump's `particle_id` is written after
the full chain. They are two different reads, and on the production arm they
**disagree on 12 of 390 taped candidates**, always the same way — tape 13 / 211 /
2212, dump 11:

```
166870/85045   tape 13    dump 11        282271/51038   tape 13    dump 11
169626/22034   tape 211   dump 11        318769/31026   tape 211   dump 11
174771/15018   tape 2212  dump 11        406125/8059    tape 211   dump 11
174771/22024   tape 13    dump 11        ... 12 in total
```

Recomputing the same table with the dump's `particle_id` gives 64/31/36/0.833 and
7/1/**0** — a different table, from the same labels and the same arm.

**§14 said "the object §22.5 names as the non-EM false fire (`318769/31026`)
reads `pdg = 11` on both arms."** That is true of the *dump* and false of the
*tape*, and an EM-only restriction would be implemented inside the splitter,
where the tape's value is the one in scope. **So `318769/31026` is a genuine
non-EM false fire and the restriction would remove it.** That half of §14 is
withdrawn.

### 17.3 What survives is the finding, and it is stronger on 14 objects

§14's *premise* still falls: "1 confirmed cut in 8" is an artefact of a label set
built around EM showers. The π-typed sweep and §22.5's eight overlap in **exactly
one object** (396222/9084), so the union is 14 labelled non-EM objects:

| | §22.5's 8 | **union of 14** |
|---|---|---|
| confirmed cuts | 1 | **4** |
| fires | 1 | 1 |
| true positives | 0 | 0 |
| purity | 0.000 | **0.000** |
| efficiency | 0.000 | **0.000** |

**The EM-only restriction is therefore priced, not refuted**: it removes **1**
false fire (purity 0.833 → 0.857) and permanently forecloses **4** confirmed
cuts that the splitter does not reach today. Four times the demand for one unit
of noise. **Still not recommended — but for a different reason than §14 gave.**

### 17.4 A second correction: §14.2's 0.250 was an offline-kernel number

§14.2 reported splitter efficiency 0.250 on π-typed confirmed cuts, counting
396222/9084 as a fire. The production tape says otherwise:

```
SHOWER_SPLIT cand shower=9084 pdg=211 nseg=25 ... nacc=2 nparts=1 fired=0
```

An accepted seed pair whose segment assignment **collapsed to one part** — so it
does not fire. Measured against the shipped binary rather than the offline
kernel, efficiency on non-EM confirmed cuts is **0 of 4 = 0.000**. The
under-firing finding is not weakened by the correction; it is absolute.

---

## 18. Item 3 — the bar, and the misses are NOT where I said they were

### 18.1 The acceptance bar, stated numerically

Any replacement trigger must be measured on **all 78 labelled objects** (71 + the
7 π-typed) and must:

1. reach **≥ 1 of the 4** confirmed non-EM cuts (today: 0);
2. hold EM efficiency at **≥ 0.968** (30 of 31 confirmed EM cuts fire today);
3. hold EM purity at **≥ 0.857** (30 true of 35 fires);
4. keep the seeded-random control stratum at **0 fires** (doc pr/139 §22.2).

The 7-object π set cannot test 2–4 at all — only the 71-object EM set can — which
is why §15.3 said the EM set is the instrument and this one is not.

### 18.2 Where the trigger actually fails, on every labelled object

Across all 78, the shipped trigger misses **5** confirmed cuts, and the tape says
exactly where:

| class | event | object | verdict | n_seed | valley_best | nacc | mechanism |
|---|---|---|---|---|---|---|---|
| nonEM | 388 | 23028 | SPLIT2 | 4 | 1.000 | 0 | no accepted seed **pair** |
| nonEM | 181050 | 15005 | SPLIT2 | 3 | 1.000 | 0 | no accepted seed pair |
| EM | 71372 | 19049 | SPLIT3 | 4 | 1.000 | 0 | no accepted seed pair |
| nonEM | 278420 | 18002 | SPLIT2 | **1** | 1.000 | 0 | one core — genuinely invisible |
| nonEM | 396222 | 9084 | SPLIT3 | 4 | 0.251 | 2 | **assignment** collapsed to 1 part |

`valley_best = 1.000` with `angle_best = -1.00` is the sentinel for *no seed pair
passed the charge-share floor* (`NeutrinoShowerClustering.cxx:5854-5861`: `vbest`
is only updated for pairs with `min(frac_i, frac_j) >= m_shower_split_min_frac`,
default **0.03**). So on **three of the five**, the seed search **already found
3–4 maxima** and the pair was refused by the 3 % charge-share floor, not by the
angular test and not by any absence of structure.

**This corrects §14.3 and §15.3's framing.** "A gain needs a different
observable, not a different search" is right only for 278420/18002. The other
four split into **acceptance** (3 objects, refused at `min_frac`) and
**assignment** (1 object, `nacc = 2 → nparts = 1`) — two named, located
sub-problems inside the existing machinery.

**This is not a recommendation to move `min_frac`.** n = 3, the per-seed
fractions are not on the tape, and doc pr/139 §18's discipline (no threshold on a
handful of objects) applies with full force. It is the measurement that tells the
next person where to look, and what it would cost to look there: any loosening is
priced against §18.1's bar on the 71-object set.

---

## 19. Item 5 — the doc pr/97 failure check was never broken; the *runner* was

### 19.1 §13.3's diagnosis is wrong, and the evidence is in the timestamps

§13.3 blamed the doc pr/97 loop for the intermittent

```
./run_pr_chain_batch.sh: line 2151: _r: unbound variable
```

**The loop is correct.** `_r=$(sed …); _r=${_r:-missing}` cannot raise an unbound
reference under `set -u`; there is no `unset`, no trap, and bash here is 5.2.26.

The real cause is that **bash re-reads a running script from a saved byte
offset**, and this campaign edits `run_pr_chain_batch.sh` (the per-round env→TLA
blocks) *while arms are in flight*. Three pieces of evidence, all consistent:

- the script's mtime, **2026-08-31 19:54:22**, falls **18 s before** the last
  write of the arm log that failed (`pr141-arm-onboth-nuecc48.log`, 19:54:40);
- the reported line number **tracks the edits** — 2145 → 2149 → **2151** across
  three arms, a two-line shift matching an insertion, not a fixed defect;
- the failure lands on the *last* statement in the file, the one with the longest
  exposure window.

Reproduced directly (`/home/xqian/tmp/edittest.sh`, a 230 KB script edited one
second into a three-second run):

```
./edittest.sh: line 3004: xxxxxxxxxxxxxxxxxxxx…: command not found
```

— bash resuming mid-token and executing a fragment of a comment line. With a
different byte shift that fragment is a variable reference, and the message is
`_r: unbound variable`.

### 19.2 The fix, and it is verified by triggering it

The whole body (everything after `set -u`) is wrapped in one compound command, so
bash must **parse it all before executing anything** and a later edit cannot
reach the run. The closing brace is unreachable — the body always exits — but is
required to parse.

| check | result |
|---|---|
| `bash -n run_pr_chain_batch.sh` | rc = 0 |
| no-arg usage path, before vs after | **byte-identical output**, rc = 1 both |
| brace-wrap under a mid-run edit (`edittest3.sh`) | tail ran correctly, **rc = 0** |
| bare brace **without** a trailing `exit` | rc = 2 — bash resumes past the brace; the `exit` is load-bearing |

And the doc pr/97 check itself, extracted and fed a synthetic `rc.txt` set (one
`rc=0`, one `rc=250`, one file missing):

```
# evt=102 rc=250
# evt=103 rc=missing
SNIPPET rc=1
```

**It fires, it names the events, it exits 1.** It was always able to; it was
simply never reached on the runs that tripped the offset.

Because the wrap changes only *when bash parses*, and the guarded loop runs after
the merge, the change can alter nothing but exit code and stderr — no arm output
is affected and no gate is spent.

---

## 20. Item 4 — the μ-typed PID set, served with a pre-registered predictor

`scripts/pr141_pidset.py` takes §9.5's single specimen to its whole population:
**35 μ-typed (`|pdg| = 13`) objects above 50 MeV across the 239 events.** Six are
already typed by hand (§9.5), leaving 29.

A μ-typed object's `kine_charge` is priced under the **track** hypothesis, so a
mis-typed EM object is low in `kine_reco_Enu` by the exact global constant
`(0.87 × 0.95)/(0.58 × 0.86) = 1.657` — i.e. **0.657 × the stored energy is
missing**. That is the ranking metric.

### 20.1 The predictor, pre-registered before the set was served

Two screens were tried on §9.5's six hand-typed objects:

| screen | GAMMA (3) | TRACK (3) | |
|---|---|---|---|
| `kine_charge / kine_range` | 0.85, 1.55, 0.53 | 0.60, 0.29, 0.66 | **overlaps — rejected** |
| **mean segment length `L/nseg`** | 10.7, 5.2, 35.1 cm | 40.3, 78.2, 42.4 cm | separates at ≈ 38 cm/seg |

so the committed prediction is **`L/nseg < 40 cm ⇒ EM`**. A muon is one long
segment; a shower branches. Plain length does **not** work — the highest-energy
μ-typed objects are 300–500 cm cosmics carrying 3–6 kink segments — which is why
the screen is the density and not the count.

**The basis is n = 6 and it is weak in a specific way**: all three TRACKs are
`nseg == 1`, so `L/nseg` reduces to plain length on that side and the 40 cm cut is
effectively set by one object (280159/90098 at 40.3 cm). The density
formulation's real claim — *a kinked muon still has long segments* — is untested,
and this scan cannot test it either: only one served object (499577/13031,
56.9 cm/seg) sits anywhere near the cut. That bounds what the result can conclude
however the verdicts land.

### 20.2 What was served

**18 objects, blind**, ranked by energy at stake, tag `pidmu-0906-owner`, brief in
`docs/pr/pr141-scan-brief-pid.md`. 14 are predicted-EM (1684 MeV of Enu at stake
between them); **4 are predicted-TRACK controls and carry no information** —
503/473/341/331 cm cosmics that nobody would call showers. The predictor must
therefore be scored as **false-positive rate on the 14**, and the controls
reported as confirming nothing.

*Results pending the owner's scan; §21's totals do not include them.*

---

## 21. Campaign close

### 21.1 The five items

| item | verdict |
|---|---|
| 1 — refresh the π⁰ label store | **DONE, +1**. Census **36 of 66** on the shipped arm, denominator held at 66, one row moves, nothing regresses (§16) |
| 2 — audit doc pr/139 §22.5 | **DONE**. The table reproduces exactly *from the tape*; the correction owed is **mine** (§17.2), and the restriction is now priced at 1 false fire removed against 4 confirmed cuts foreclosed (§17.3) |
| 3 — the splitter trigger bar | **DONE**. Bar stated on 78 objects (§18.1); the misses are **acceptance (3) + assignment (1) + genuinely single-cored (1)**, not an absent observable (§18.2) |
| 4 — `122660/54071` | **SCOPED and SERVED**. 35-object population, 1684 MeV at stake, blind 18-object set with a pre-registered predictor (§20). Verdicts pending |
| 5 — the runner's dead failure check | **DONE**. The check was never broken; **mid-run edits** to the runner were. Fixed and verified by triggering both halves (§19) |

### 21.2 What changed in production

**Nothing.** No `clus` source changed this session; no knob moved; no arm was
spent. The only executable change is `run_pr_chain_batch.sh`'s parse-before-run
wrap, which cannot affect arm output (§19.2). The production configuration is
still §11's.

The one number that moves is the **instrument's**: the π⁰ census reads **36 of 66
exact**, not 35, and it always did — the earlier figure was scored against a
pairing the owner had superseded.

### 21.3 The campaign's own error rate, since it is now measurable

Across doc pr/139 §22–26 and doc pr/141, six published numbers turned out to be
about the measurement rather than the reconstruction:

1. two of nine "missing γs" were **id renames** (§3);
2. one census row was scored against a **superseded pairing** (§16);
3. §22.5's premise came from an **EM-enriched** label set (§17.3);
4. §14's non-EM false-fire claim read the **wrong pdg source** (§17.2);
5. §14.2's efficiency counted an **offline-kernel** accept as a fire (§17.4);
6. §13.3 blamed a **correct loop** for a shell-level failure (§19.1).

Four of the six are mine and were caught by re-deriving a published number from
its primary source. **The method lesson of this campaign is that one: the tape,
the arm and the committed script beat every reading of the code — including
mine — and a table with no script behind it is a table nobody can check.**

### 21.4 If the work resumes

In order, and none of it is π⁰ clustering:

1. **The μ-typed PID front** (§20) — the only open measurement, already served.
   If mis-typing is common it is a `kine_reco_Enu` defect with a population and
   an exact price, in a different subsystem from anything this campaign touched.
2. **The splitter's `min_frac` refusal** (§18.2) — three confirmed cuts where the
   seeds are already found. A measurement, against §18.1's bar, on the 71-object
   set. Not a threshold to move on n = 3.
3. **A committed producer for every published table.** §17.1 found one with no
   script; there may be others.

**Everything else this campaign priced is closed negative and should stay
closed**: eleven knobs across doc pr/139 §22 and doc pr/141, the k = 3 recursion,
the EM-only restriction, the seed-count trigger family, and the three μ-typed
admission knobs.

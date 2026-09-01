# doc pr/141 — the final round: what is left after the splitter campaign

**Status: OPEN (session 1, 2026-08-31).** doc pr/139 §26 closed the splitter
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

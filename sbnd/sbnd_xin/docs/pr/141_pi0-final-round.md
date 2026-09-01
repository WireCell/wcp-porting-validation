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
`:7770`) against 13.5 for the pair that was taken.

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

| event | γ short of room | room | contained f | mass → corrected |
|---|---|---|---|---|
| **168432** | `22006` **1.4 X₀** (its furthest member point is *outside* the volume, wall −0.2 cm) and `49028` 3.5 X₀ | 19.0 / 49.2 cm | 0.21 / 0.72 | **61.6 → 159.4  IN WINDOW** |
| **71872** | `79074` 3.1 X₀ | 43.5 cm | 0.58 / 0.93 | **101.1 → 138.4  IN WINDOW** |
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

| verdict | n of 9 γ slots | meaning |
|---|---|---|
| **MERGED** | **5** | every member segment is on the arm, owned by a *different* shower |
| **MERGED-BY-PROXIMITY** | **3** | the label stored no member list; a shower starts within 15 cm of the γ's recorded start |
| NO-LABEL-MEMBERS | 1 | 396222 g2 — no member list and nothing within 20 cm |

**Eight of the nine absent γs have their charge on the arm.** Not one of them is
a γ that failed to be reconstructed. The block is an **identity / merging**
problem:

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
| 8 — a γ is NOT RECONSTRUCTED | **1** (396222). The other 8 slots are **merged or renamed** — clustering. |
| 9 — mass outside the window | 3 are **containment** (calorimetry, irreducible); 3 are **over-clustering** (clustering); 2 are **pairing** the owner has now revised; 1 is a window edge. |

The residual π⁰ population is not dominated by γ-finding efficiency. It is
dominated by **γ identity — over-merging and mis-assignment — plus a
calorimetric containment loss that no clustering change can recover.** The
campaign's own instrument had been reading an id mismatch as a missing particle.

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

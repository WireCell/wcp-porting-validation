# Steiner terminals on PDVD, SBND and MicroBooNE — and does the round-8 alternative make sense?

**Status: two rounds, and they have different statuses. Read §11 before quoting
§0-§10.**

- **Round 1 (§0-§10) — measurement and design review.** No toolkit change, no
  production-config change, no arm of PDVD or SBND re-run, so no A/B gate was
  owed and none was claimed. The only new reconstruction was two 35-event
  MicroBooNE arms running a *copy* of the uBooNE chain whose compiled config at
  default TLAs is byte-identical to the production one (§0.2). It ended in a
  recommendation (§6), not a knob.
- **Round 3 (§15-§16, 2026-09-04) — the flip graded in the configuration
  production actually runs** (DL/SCN vertex ON), with a repeat arm as the noise
  floor. §13's finding survives: the STM-tagged set moves on 79 % of events
  against a **zero** floor. §15.3 **retracts a determinism claim an earlier
  version of this round made** and replaces it with what the measurement really
  showed — doc 36's metric flip, not the DL vertex. §13-vs-§15 comparisons span
  that flip and are not attributed here.
- **Round 2 (§11-§14, 2026-09-04) — the §6 recommendation EXECUTED, at the
  owner's instruction.** `terminal_min_separation` is now a real knob in `clus/`
  (C++ default 0 ⇒ byte-identical), and **PDVD production is flipped to 0.5 cm**
  in `pdvd/wct-pr-perevt.jsonnet`. SBND is deliberately **not** changed. That is
  an intentional, non-bit-identical change to PDVD output; §12 carries the
  byte-identity gates for the OFF path on every job that binds the component,
  and §13 the measured before/after. **Round 2 IS a behaviour change for PDVD
  and needs revalidation of anything built on pre-2026-09-04 PDVD PR output.**

The owner's brief, in their words: the Steiner terminals should **follow the
skeleton of the interaction**; **a little inefficiency is OK**; they must not be
**too smeared**, which defeats the purpose; and they must **not miss terminals
near the vertex**. This doc measures all four on three detectors and returns a
verdict on doc 31 §12.5's ranked alternatives.

---

## 0. Repro block

### 0.1 What was run and what was read

```bash
# ---- the only new reconstruction: two MicroBooNE arms, 35 events each --------
# qlport/uboone-mabc-steinerdump.jsonnet is a COPY of uboone-mabc.jsonnet (the
# frozen reference chain is never edited in place, CLAUDE.md sec 5.3).  It adds
# a PrDisplayDump node -- the only producer of steiner[].flag_terminal -- and
# exposes the two doc pr/29 D1/D12 terminal-filter knobs.  Both additions are
# suppressed at their defaults, so a bare compile equals production (T0 below).
cd wcp-porting-img/qlport/scripts
seq 0 34 | ST_WIRE_TOL=0 ST_ADJ_SLICE=false xargs -P 8 -I{} ./run_one_steinerdump.sh {} d37ub
seq 0 34 | ST_WIRE_TOL=1 ST_ADJ_SLICE=true  xargs -P 8 -I{} ./run_one_steinerdump.sh {} d37ubtol
#   d37ub    35/35 rc=0, 515 s total (14.7 s/event), 560 steiner clusters,
#            135909 points, 17463 terminals
#   d37ubtol 35/35 rc=0, 568 s total (16.2 s/event), 565 clusters,
#            172192 points, 22368 terminals

# ---- read-only: PDVD and SBND come from arms already on disk ----------------
#   PDVD  pdvd/work/*_d34base/calib-pr-evt*.json                  118 events
#   SBND  sbnd/sbnd_xin/work-mcp2k-d97fvpr2/pr_evt*/calib-pr-*    906 events
# Nothing under work/ or sweep/ was written, regenerated or deleted (M13).

# ---- sections 3 (density) ---------------------------------------------------
cd wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
python3 steiner_density_xdet.py     PDVD:... SBND:... UB:... UBtol:...   # strict
python3 steiner_density_xdet.py     ... --minpts 50 --minlen 20          # relaxed
python3 steiner_terminal_geometry.py ... --minpts 50 --minlen 20 \
        --nms 0.3 0.5 0.8 1.0 1.5

# ---- sections 4-5 (the two instruments this round builds) -------------------
python3 steiner_terminal_skeleton.py PDVD:... SBND:... UB:... UBtol:... \
        --nms 0.3 0.5 0.8 1.0 1.5 --core 1.0 --rv 3.0 --exempt 3.0
# and the exemption sweep, --exempt 1.0 / 1.5 / 2.0

# ---- section 0.2, the config gates -----------------------------------------
WIRECELL_PATH=$TOOLKIT/cfg:$WCDATA wcsonnet <production TLAs> \
  qlport/uboone-mabc.jsonnet              > ub_head.json      # 258920 B
WIRECELL_PATH=... wcsonnet <same TLAs> \
  qlport/uboone-mabc-steinerdump.jsonnet  > ub_copy.json      # 258920 B
cmp ub_head.json ub_copy.json                                 # byte-identical
```

### 0.2 Gates

- **T0, compiled config.** The measurement copy at default TLAs compiles to
  **258920 B, byte-identical** to `uboone-mabc.jsonnet`. With
  `-A prdump=... -A steiner_terminal_wire_tol=1 -A steiner_terminal_adjacent_slice=true`
  the compiled JSON gains exactly `terminal_wire_tol: 1`,
  `terminal_adjacent_slice: true` and a `PrDisplayDump:pr` node **last** in the
  MABC pipeline (after `TaggerCheckNeutrino`, as PDVD runs it).
- **The output gate — this round's one real gate.** `d37ub`'s Steiner cloud is
  compared against the **April 2026 production** Bee `steiner-global` layers in
  `qlport/mabc_<idx>.zip`. On **35 of 35 events**: identical point count,
  identical cluster-id sets, and a nearest-neighbour match in (y, z) of at most
  **16 µm** — which is the Bee layer's own printed precision (3 decimals in cm).
  Maximum |Δx| is 4 µm, so the dump and the Bee layer are in the same frame.
  *The new arm therefore describes uBooNE production, not a variant of it.*
- **Library pin, and it was load-bearing.** The tree is shared and a concurrent
  session was building in it. `local/lib` was snapshotted to
  `/home/xqian/tmp/doc37/lib_pin` and `LD_LIBRARY_PATH` pinned to it. The pin
  held `libWireCellClus.so` mtime **08:27, md5 0729c1e2…**; by the time the arms
  finished, the live `local/lib` copy was mtime **08:42, md5 3f4a7e32…** — a
  different binary. All 70 runs used the pinned one, and the output gate above
  proves that binary reproduces April production for this stage.
- Toolkit `28cd60d8`, wcp-porting-img `61b66c4e` at the start of this round.

### 0.3 Caveats that bound every number below

1. **No charge in the dump.** `PrDisplayDump.cxx:1113` stores x/y/z and the
   terminal flag only, so the true `calc_charge_wcp` greedy order is not
   reproducible offline. Every simulated thinning is therefore reported under
   **four** orders — principal axis, seeded shuffle, an adversarial order
   (farthest-from-vertex first), and an **order-free bound** that needs no charge
   at all (§5.2). Routes to a real charge order were checked and are closed:
   PDVD/SBND Bee zips carry no `steiner` layer, and the PDVD pctree dump is
   written at the *clustering* stage, before `steiner_pc` exists.
2. **§4 runs on a selected subset.** Only clusters PR actually fitted have a
   skeleton: **135 of PDVD's, 1317 of SBND's, 175 of uBooNE's** clusters. That
   is a small and non-random fraction (PDVD's 118 events hold ~5300 Steiner
   clusters). §4 grades localization on the clusters that got a PR skeleton.
3. **§5 freezes the vertices.** The PR vertices are the ones the *production*
   terminal set produced. Thinning would change the PR graph, so §5 answers
   "would the terminals that support today's vertices survive?" — not "where
   would the reconstructed vertex end up?". That second question needs the knob
   and an arm, and is named in §8.
4. **uBooNE's selection is thin.** 560 Steiner clusters over 35 events, median 27
   points and 1.8 cm long. The strict doc-31 selection leaves n=8; the relaxed
   one n=25 (n=12 non-main). Every uBooNE median below carries its n.

---

## 1. What the algorithm does, and why the density is what it is

Summarized, not re-derived — the full walk is doc 31 §1.1–1.3 and §12.4.

`Steiner::Grapher::create_steiner_tree` runs four phases: **P1**
`find_steiner_terminals` (`SteinerGrapher.cxx:685`), **P2**
`filter_by_reference_cluster` (`:209`), **P3** `filter_by_path_constraints`
(`:290`), **P4** `get_extreme_points_for_reference` (`:396`), which *inserts* the
cluster's extreme points unconditionally.

P1 is where the density is set, and the mechanism is one line:

```cpp
// SteinerGrapher.cxx:707  -- inside a loop over BLOBS
peaks = find_peak_point_indices(point_indices, graph_name, ddmc, nlevel=1);
```

`find_peak_point_indices` gates candidacy on `charge > threshold && quality`
(`:513`) and then suppresses non-maxima over a 1-hop neighbourhood — but its
charge map is built from **one blob's** points, so out-of-blob neighbours hit a
`continue` (`:576`, `:593`). The suppression therefore **can never remove a
blob's last candidate**. Round 6 measured the consequence directly with an
env-gated counter: **1.02 terminals per candidate-bearing blob**. Terminal
density is candidate-bearing blob density, and no threshold and no `nlevel` can
change that.

**What that predicts, and what this doc tests on a third detector:** for a
drift-parallel track a blob is one time slice, so the spacing law should be *one
terminal per occupied time slice* — i.e. terminal spacing inherits the **slice
pitch**, a detector-configuration number, and the track's drift angle. Neither
is physics.

---

## 2. The three detectors: geometry and operating point

| | wire pitch U/V/W (mm) | ctpc drift step (mm) | pitch / drift step |
|---|---|---|---|
| SBND | 3.000 / 3.000 / 3.000 | 3.126 | 0.96 |
| **MicroBooNE** | 3.003 / 3.003 / 3.000 | **2.202** | 1.36 |
| PDVD | 7.650 / 7.650 / 5.100 | 2.9615 | 2.58 / 2.58 / 1.72 |

Drift step is each detector's own `nticks_live_slice × tick × drift_speed`
(4 × 0.5 µs × 1.48073 / 1.563 / 1.101 mm µs⁻¹). Source: doc 34 §3 — including
its correction that uBooNE's live chain runs **1.101 mm/µs**, not the WCP
prototype's 1.6, so the often-quoted 3.2 mm slice is wrong and the real one is
**2.20 mm, the finest of the three**.

**The operating points, corrected.** Reading `pr.jsonnet`'s defaults gives the
wrong answer for PDVD and SBND: both drivers override them.

| | terminal charge floor | `wire_tol` | `adjacent_slice` | `edge_charge_forward_dead_mix` |
|---|---|---|---|---|
| PDVD (`pdvd/wct-pr-perevt.jsonnet:704,406,407,422`) | **500 e** | 1 | true | true |
| SBND (`sbnd/wct-pr-perevt.jsonnet:381,382,397`) | 4000 e | 1 | true | true |
| MicroBooNE (`qlport/uboone-mabc.jsonnet`, no keys) | 4000 e | **0** | **false** | **false** |

So PDVD and SBND are **identical on all three filter knobs** and differ only in
the charge floor — which round 7 already showed is the same operating point
(18.0 % vs 16.6 % of points pass). MicroBooNE is the odd one out: it still runs
the P2 filter that doc pr/29 D1 measured as discarding **47.7 %** of terminals on
SBND evt 388. A raw three-way comparison would therefore be reading the
**filter**, not the geometry — which is why both settings were run.

*(This is an observation about the frozen reference chain, not a proposal. The
prototype has the ±1 wire slack; the toolkit's uBooNE binding does not. It is
gated byte-identical against a reference produced that way, and nothing here
suggests changing it.)*

---

## 3. Terminal density on three detectors

Relaxed selection (≥ 50 Steiner points, PCA linearity > 0.95, 5th–95th percentile
extent > 20 cm), so uBooNE has a usable n. Medians [IQR] over clusters.

### 3.1 The headline

| arm | n | points/cm | **terminals/cm** | **1 terminal per** |
|---|---|---|---|---|
| PDVD | 3420 | 7.33 | 2.03 | **0.49 cm** |
| SBND | 770 | 6.89 | 1.38 | **0.72 cm** |
| MicroBooNE (production) | 25 | 22.6 | 3.62 | **0.28 cm** |
| MicroBooNE (`wire_tol=1`, `adjacent_slice=true`) | 25 | 25.8 | 4.54 | **0.22 cm** |

and on matched topology — non-main clusters, i.e. cosmics in all three:

| arm | n | terminals/cm | 1 terminal per |
|---|---|---|---|
| PDVD | 26 | 2.23 | 0.45 cm |
| SBND | 135 | 1.55 | 0.65 cm |
| MicroBooNE (production) | 12 | 1.90 | 0.61 cm |
| MicroBooNE (filters on) | 13 | 2.74 | 0.36 cm |

**MicroBooNE is not sparser than PDVD.** As measured it is the densest of the
three, and PDVD sits between SBND and uBooNE rather than being an outlier — which
is the reframing the round-6 complaint needs. **But most of the SBND↔uBooNE gap
is population, not detector**, and §3.4 decomposes it before any of it is used as
a detector statement.

### 3.2 The spacing law, confirmed on three geometries

The exact decomposition (doc 31 §12.1), in the **along-drift** bin
(|cos(axis, x)| > 0.7) where a blob is one time slice, so the three detectors are
matched on the confound that dominates it:

| arm | n | slice pitch | slices/cm | slices lit | **terminals per OCCUPIED slice** | terminals/cm |
|---|---|---|---|---|---|---|
| PDVD | 2226 | 0.296 cm | 3.08 | 0.64 | **1.05** | 2.10 |
| SBND | 143 | 0.313 | 2.60 | 0.56 | **1.04** | 1.56 |
| MicroBooNE | 8 | 0.220 | 4.18 | 0.80 | **1.15** | 3.41 |
| MicroBooNE filters on | 8 | 0.220 | 4.47 | 0.86 | **1.18** | 4.34 |

**One terminal per occupied time slice, on all three detectors, to within 14 %.**
`terminals/cm ≈ (slices/cm) × (fraction lit) × 1.05` reproduces every row. And
`slices/cm` is just `|cos θ| / slice pitch`.

So terminal density is, to first order,

> **terminals per cm ≈ (fraction of crossed slices lit) × |cos(track, drift)| /
> (slice pitch)**

with no physics in any of the three factors. The occupancy factor is not
decorative — it runs 0.56 to 0.86 across the three detectors and is what the
charge threshold moves — but the *ordering* of the three detectors is set by
`1 / slice pitch`: uBooNE 4.5 > PDVD 3.4 > SBND 3.2 cm⁻¹.

**This closes doc 31 §12.7's argument.** That section said SBND is not an
independent reference because it is the same algorithm under a different
geometry. With uBooNE measured, that is now a three-detector statement: *no*
detector is a reference for "correct density", because all three inherit the same
configuration-driven spacing. **Do not tune a radius to reproduce any of these
numbers.** The operating point has to come from what consumes the terminals.

### 3.3 Where the detectors genuinely differ

| arm | nn3d (terminal spacing) | largest terminal-free run | kept at R = 0.3 cm |
|---|---|---|---|
| PDVD | 0.66 cm | 4.46 cm | **0.85** |
| SBND | 0.71 | 3.44 | **1.00** |
| MicroBooNE | 0.41 | 1.86 | **0.82** |
| MicroBooNE filters on | 0.41 | 1.06 | **0.84** |

Doc 31 §12.6 found on 2 events that SBND's production terminal set is already
0.3 cm-separated while PDVD's is not, and read that as a PDVD close-pair tail.
At 118 / 906 / 35 events it holds — and **uBooNE has the same tail as PDVD
(0.82 kept), so SBND is the exception, not PDVD the anomaly.**

### 3.4 Why MicroBooNE reads 2.6x denser than SBND, when the geometries are twins

The two detectors have the same wire pitch (3.000 vs 3.003 mm) and differ mainly
in the slice width, 0.220 vs 0.313 cm — a factor **1.42**. That cannot by itself
produce the 2.66x in §3.1, so the excess has to be named rather than absorbed.
Medians over the relaxed selection:

| factor | SBND | MicroBooNE | ratio |
|---|---|---|---|
| slice pitch (hence slices/cm: 1.29 vs 2.82) | 0.313 cm | 0.220 cm | **1.42** |
| \|cos(axis, drift)\| of the *selected* clusters | 0.35 | 0.52 | **1.49** |
| fraction of crossed slices lit | 0.93 | 0.93 | 1.00 |
| terminals per **occupied** slice | 1.16 | 1.42 | **1.22** |
| **terminals / cm** | **1.36** | **3.62** | **2.66** |

`1.42 x 1.49 x 1.22 = 2.67`, i.e. the whole of it. **Only the first factor is a
property of the detector.** The second is the drift alignment of whatever tracks
the selection kept — a track more parallel to the drift crosses more slices per
cm of its own length — and the third is topology: how many blobs share a slice.

Splitting by cluster role shows the third factor is topology and not algorithm:

| | n | terminals per occupied slice | pts/slice | 1 terminal per |
|---|---|---|---|---|
| SBND main | 289 | 1.16 | 5.2 | 0.74 cm |
| SBND non-main (cosmics) | 47 | **1.21** | 5.3 | 0.69 cm |
| MicroBooNE main | 13 | 1.81 | 11.9 | 0.21 cm |
| MicroBooNE non-main (cosmics) | 12 | **1.30** | 7.7 | 0.53 cm |

**On matched topology — cosmics in both — the criterion column agrees to 7 %
(1.21 vs 1.30), and the spacing gap narrows to 0.69 vs 0.53 cm, i.e. 1.3x,
close to the 1.42x the slice width alone predicts.** So the algorithm is doing
the same thing on both detectors, exactly as §3.2's per-occupied-slice law says;
what differs is the slice it is doing it in, and what the two populations contain.

**And the populations differ by construction, not by chance.** `cm.steiner(...)`
takes `require_beam_flash`, default **true**. MicroBooNE uses the default
(`qlport/uboone-mabc.jsonnet:1263`), so it builds a Steiner graph **only for
beam-coincident clusters** — the neutrino candidate and its companions. PDVD
(`pr.jsonnet:1281`) and SBND (`sbnd/clus.jsonnet:2071`) both pass **false** and
build one for every scope-passing cluster. MicroBooNE's Steiner population is
therefore neutrino-interaction-heavy by design (560 clusters over 35 events,
16/event, against PDVD's ~45/event), and its main clusters are dense multi-blob
objects: 11.9 points per slice against SBND's 5.2.

**What this does and does not change.** It does not touch §3.2 — one terminal per
occupied slice still holds on all three geometries, and that is the finding R1
rests on. It does not touch §5, which is measured per vertex and never divides
one detector by another. What it does change is how §3.1's headline should be
read: *"MicroBooNE is the densest"* is a statement about MicroBooNE's
**beam-flash-selected** population, and the detector-level difference from SBND is
the 1.42x slice width and little else. **n = 25 for MicroBooNE here (13 main,
12 non-main), so the split above is an indication, not a measurement.**

---

## 4. Does the skeleton get followed? (§6.1's third row, built)

Doc 31 §12.8 lists "on the track / vertex — transverse distance to the **fitted**
skeleton" as *still owed, needs a PR product the phase dump does not carry*. The
calib dump does carry it: `segments[].points[]` is the fitted trajectory, in the
same frame as `steiner[]`. `steiner_terminal_skeleton.py` measures the distance
from each terminal to the nearest point **of a line piece** of its own cluster's
fitted polylines (not to the polyline's vertices — most segments carry 2 points
spanning tens of cm, so a vertex-only distance would measure segment sampling).

The only honest form is **terminal minus cloud**, the whole Steiner cloud being
the matched control:

| arm | n clusters | terminal → skeleton | cloud → skeleton | **excess** |
|---|---|---|---|---|
| PDVD | 135 | 0.87 cm | 1.38 | **−0.48** |
| SBND | 1317 | 0.39 | 0.68 | **−0.28** |
| MicroBooNE | 175 | 0.36 | 0.57 | **−0.19** |
| MicroBooNE filters on | 191 | 0.39 | 0.57 | **−0.17** |

**The excess is negative on every detector: terminals sit closer to the fitted
skeleton than the cloud they are drawn from.** The first criterion — "follow the
skeleton of the interaction" — is satisfied today, and this is the first time it
has been measured on any detector.

And thinning does not break it (median terminal → skeleton, cm):

| arm | production | R=0.5 | R=1.0 | R=1.5 |
|---|---|---|---|---|
| PDVD | 0.87 | 0.89 | 0.93 | 0.98 |
| SBND | 0.39 | 0.40 | 0.41 | 0.40 |
| MicroBooNE | 0.36 | 0.45 | 0.47 | 0.44 |

Removing 40–55 % of the terminals moves the survivors **≤ 0.11 cm** further from
the skeleton on every detector. R1 does not smear the skeleton; it thins it in
place. That is the quantitative form of "a little inefficiency is OK".

---

## 5. The vertex region — the owner's fourth criterion

This is the measurement that decides the question, and doc 31 §12.6 could not
make it: it measured the largest terminal-free run along a cluster's *principal
axis*, which is blind to a vertex by construction, because a vertex region is not
linear.

**Method.** For every PR vertex of degree ≥ 2 (PDVD 1026, SBND 1706, uBooNE
216/260):
- terminals within **`core` = 1 cm** of the vertex fit point are **vertex-core
  terminals**, counted separately and *not* branch-assigned — incident polylines
  converge there, so nearest-polyline assignment is a coin flip exactly where the
  measurement matters and "a branch dropped to zero" would be a tie-breaking
  artefact;
- every other terminal within **`R_v` = 3 cm** is assigned to its nearest
  **incident segment** = its branch.

### 5.1 Branches emptied

Total over all vertices, out of all occupied branches:

| arm | R | order-free **bound** | axis | shuffled | adversarial | of branches |
|---|---|---|---|---|---|---|
| PDVD | 0.5 | 14 | 13 | 16 | 14 | 2213 |
| PDVD | 1.0 | 90 | 61 | 78 | 69 | 2213 |
| PDVD | 1.5 | 227 | 181 | 240 | 153 | 2213 |
| SBND | 0.5 | 10 | 15 | 11 | 14 | 3768 |
| SBND | 1.0 | 84 | 98 | 136 | 98 | 3768 |
| SBND | 1.5 | 264 | 278 | 435 | 283 | 3768 |
| MicroBooNE | 0.5 | 6 | 4 | 5 | 2 | 553 |
| MicroBooNE | 1.0 | 16 | 22 | 23 | 19 | 553 |

The **bound** column needs no charge: *a branch can be emptied under some greedy
order only if every one of its near-vertex terminals has a terminal of a
different branch within R.* It is geometric, so it bounds the failure exactly
rather than sampling it — and it tracks the realized numbers closely, which says
the bracketing orders are not hiding a tail.

At **R = 0.5 cm the loss is ≈ 0.4 % of branches**; at R = 1.0 it is 2.6–3.4 %;
at R = 1.5 it is 7–10 %. So plain R1 is safe at 0.5 cm and starts to bite at 1.0.

### 5.2 The sharper signal: vertex-core terminals

Vertices that lose **every** core terminal, of those that had at least one.
All three orders are shown, as in §5.1 — the adversarial column is deliberately
the worst case and is the one §8 flags as an upper bound:

| arm | had ≥1 core | R=0.5 ax / rnd / **adv** | R=0.8 ax / rnd / **adv** | R=1.0 ax / rnd / **adv** | R=1.5 adv |
|---|---|---|---|---|---|
| PDVD | 802 | 12 / 14 / **25** | 24 / 26 / **44** | 59 / 82 / **128** | 367 |
| SBND | 1547 | 8 / 12 / **17** | 21 / 29 / **49** | 54 / 66 / **117** | 551 |
| MicroBooNE | 195 | 0 / 2 / **2** | 1 / 4 / **4** | 3 / 7 / **11** | 72 |
| MicroBooNE filters on | 237 | 0 / 2 / **4** | 2 / 7 / **10** | 4 / 11 / **19** | 107 |

As a rate on PDVD: **1.5–3.1 % at R = 0.5, 7.4–16 % at R = 1.0**, the range being
the ordering spread. Total core terminals surviving on PDVD (adversarial):
**1779 → 1341 (R=0.5) → 820 (R=1.0) → 446 (R=1.5)**.

**This is where plain R1 fails the owner's fourth criterion.** At R = 1.0 cm —
the radius doc 31 §12.6 highlighted, on the grounds that it removes 44 % of
terminals while moving the largest terminal-free run by < 0.3 cm — **7.4 % of
PDVD's multi-branch vertices lose every terminal within 1 cm of the vertex under
the mildest ordering, and 16 % under the harshest.**
The along-axis coverage metric is blind to this, exactly as expected: the
terminals are still there 1.2 cm away, so no *run* opens; what is lost is the
support right at the junction, which is what
`NeutrinoOtherSegments::find_other_segments` (`:159`) proposes branches from and
what `NeutrinoStructureExaminer` (`:820`, `:836`) searches for within 6 cm of a
vertex.

### 5.3 The vertex exemption, and what it costs

Exempting terminals within `X` cm of any PR vertex from the thinning:

| arm | R=1.0, branches emptied: plain | X=1.0 | X=1.5 | X=2.0 | X=3.0 |
|---|---|---|---|---|---|
| PDVD | 69 | 62 | 40 | **26** | 0 |
| SBND | 98 | 72 | 54 | **46** | 0 |
| MicroBooNE | 19 | 16 | 14 | **12** | 0 |

**Read `X = 3.0` as vacuous, not as a result**: with `R_v` also 3 cm, every
near-vertex terminal is protected by construction, so the branch metric cannot
fire. The meaningful radii are `X < R_v`, and at **X = 2.0 the branch loss falls
by ~60 % on PDVD** (69 → 26).

For **vertex-core** terminals the protection is exact by construction for any
X ≥ core = 1.0 cm: 1779 → 1779 on PDVD at every R. That is the design working as
intended rather than a measurement — but the *cost* is a measurement:

| arm | fraction of terminals kept at R=1.0: plain | X=1.0 | X=1.5 | X=2.0 |
|---|---|---|---|---|
| PDVD | 0.58 | 0.60 | 0.62 | **0.64** |
| SBND | 0.58 | 0.69 | 0.73 | **0.76** |
| MicroBooNE | 0.46 | 0.56 | 0.60 | **0.64** |

**The exemption is nearly free on PDVD and expensive on SBND and uBooNE**, and
the reason is topology, not a defect: PDVD's population is cosmics, most of whose
clusters carry no PR graph at all, so there is almost nothing to exempt (on the
298595 scan event the exemption changes 21939 kept terminals to 21988 — 0.2 %).
SBND and uBooNE events are neutrino interactions dense in vertices, so a 2 cm
ball around every vertex covers a large share of the event and undoes much of the
thinning. **Any exemption radius is therefore a per-detector cost decision, and
it is the SBND/uBooNE cost that constrains it, not PDVD's.**

---

## 6. The verdict on doc 31 §12.5

Against the owner's four criteria, with §§3–5 as the evidence:

| criterion | verdict | evidence |
|---|---|---|
| follows the skeleton | **already true, and R1 preserves it** | §4: excess −0.17…−0.48 cm; thinning moves it ≤ 0.11 cm |
| a little inefficiency is OK | **R1 is exactly this** | §5.1: ≈0.4 % of branches at R=0.5 |
| not too smeared | **R1 is the right lever** | §3.2: spacing is 1/slice-pitch, not physics |
| **not miss terminals near the vertex** | **plain R1 FAILS at R ≥ 0.8** | §5.2: 7.4–16 % of PDVD vertices lose all core terminals at R=1.0 |

**(R1) cross-blob thinning at a physical radius — endorsed.**
It removes the actual defect (a spacing set by slice pitch and drift angle,
replaced by a configured physical distance), it has a coverage bound independent
of the charge ordering, and it is a one-function, one-call-site, one-double diff
that is byte-identical at its default. §4 adds a result doc 31 did not have: it
does not degrade localization. The insertion point in doc 31 §12.5 — between P3
(`:126`) and P4 (`:136`) — is right and the reasoning for it stands.

**If protection is wanted, it should be junction-local, not doc 31's R2 global
backstop.** Doc 31 pairs R1 with R2, "at least one terminal per X cm"
everywhere. The owner's constraint is narrower and sharper than that, and §5.2
shows the failure is narrower too: it is **not** a coverage failure along a track
(no run opens — §3.3's runs barely move), it is the loss of support at
*junctions*. A global floor spends terminals along every track to buy protection
at vertices; a local exemption buys it where the risk actually is, and §5.3
prices both halves of that trade. But see the next paragraph before treating it
as available.

**A feasibility constraint that shapes the recommendation, and was checked
rather than assumed.** A *vertex* exemption cannot be written literally where
doc 31 §12.5 puts the thinning. `create_steiner_tree` runs inside
`CreateSteinerGraph`, which is **`steiner` in the pipeline — before
`tagger_check_neutrino`** (`pdvd/wct-pr-perevt.jsonnet:139-143`), and neither
`SteinerGrapher.cxx` nor `CreateSteinerGraph.cxx` contains a single reference to
`PRGraph` / `PRVertex` / `pr_graph`. **The PR vertices measured in §5 do not
exist yet when the terminals are chosen.** §5 is still the right measurement —
it says where the damage would land — but the remedy has to be built from what
*is* in scope at that point.

**Recommended operating point, stated as a recommendation and not a result:**

> **`terminal_min_separation = 0.5 cm`, no exemption.**

At R = 0.5 the exemption is barely needed and the constraint above therefore does
not bind: branch loss is ≈ 0.4 % (§5.1), and **1.5–3.1 % of PDVD's vertices
(12–25 of 802), 0.5–1.1 % of SBND's (8–17 of 1547) and 0–1 % of uBooNE's (0–2 of
195) lose all their core terminals** across the three orderings (§5.2), while
PDVD keeps 0.81 of its terminals and SBND 0.85. A 15–19 % thinning is modest, but it is the
part of the close-pair tail that §3.3 shows is real on PDVD and uBooNE and absent
on SBND — i.e. it removes near-duplicates and nothing else. That is exactly "a
little inefficiency is OK".

**R = 0.8 or 1.0 should not be adopted without a junction-protection mechanism**,
because §5.2 is where they fail. Building one is round 9's first design question,
and the candidates in scope at Steiner time are the P4 extreme points and
branch-like structure of the Steiner graph itself — not PR vertices. That is
named in §9, unsolved.

**(R2) global coverage backstop — demoted, not dismissed.** Its motivation
survives (PDVD's largest terminal-free run 4.46 cm vs SBND 3.44, uBooNE 1.86 —
PDVD really is both denser *and* gappier), but it does not address the criterion
the owner actually raised, and §5 shows a cheaper instrument does. Keep it as a
separate question about PDVD's gaps.

**(R3) coarser partition, (R4) global peak pass — unchanged.** Both remain
designs. R4 stays the principled endpoint and still needs a bounded replacement
for the connected-components merge (doc 31 §12.4 iii). Neither should be built
before R1 has established what density the downstream physics wants.

**(R5) `BlobSampler::stepped` in physical units — closed by doc 31 §12.3.**
**(R6) the charge threshold — closed by doc 31 §11.3.** This round adds one
reason not to reopen R6: §3.2 shows the density is set by the slice pitch, and
the threshold moves only the *fraction of slices lit*, which is the smaller of
the two factors.

---

## 7. A scan set, built and not uploaded

`/home/xqian/tmp/doc37/d37_thinning_scan.zip` (1.2 MB, builder
`/home/xqian/tmp/doc37/build_bee.py`), three Bee events, no reconstruction run:

| # | event | Steiner pts | terminals | R=1.0 | R=1.0 + 2 cm exemption |
|---|---|---|---|---|---|
| 0 | PDVD 039252/2 (298595) | 156671 | 34167 | 21939 (0.64) | 21988 (0.64) |
| 1 | SBND 18255/100222 | 9171 | 1618 | 843 (0.52) | 1230 (0.76) |
| 2 | MicroBooNE 5384/6786 | 11621 | 858 | 414 (0.48) | 536 (0.62) |

PDVD's exemption column barely moves (21939 → 21988, 0.2 %) — **that is the
exemption being nearly free, not failing**: this is a cosmic event and most of
its clusters carry no PR graph at all, so there is almost nothing to exempt.
§5.3 has the same effect at population scale, and the contrast with SBND
(0.52 → 0.76) and uBooNE (0.48 → 0.62) is the point of showing all three.

Layers per event: `steiner` (the cloud), `term-prod`, `term-R1.0`,
`term-R1.0exm`, `skeleton` (the fitted PR trajectory) and `vertices` (main vertex
at q = 15000). **Not uploaded** — that is the owner's call.

---

## 8. What is not established

- **The fourth grading row, "does not cost physics"** — segments and vertices
  proposed, STM tags, track-fit point counts — **cannot be measured this round.**
  It needs the knob and an arm. Everything above is a property of the terminal
  *set*, and terminals are an *input* to physics.
- **§5 freezes the vertices** (§0.3 caveat 3). It measures what thinning removes
  from the neighbourhood of today's vertices, not where the reconstructed vertex
  would end up. The failure mode it identifies is real; its rate under a live
  knob could be larger or smaller.
- **The greedy order is bracketed, not known** (§0.3 caveat 1). The order-free
  bound in §5.1 is what makes the branch conclusion safe; §5.2 now shows all three
  orderings, and the adversarial column is an upper bound on the damage by
  construction. The true order is charge-descending, which is neither of the
  bracketing orders — it is bracketed by them, not equal to one.
- **uBooNE's n is small** — 25 clusters in the relaxed selection, 8 in the
  along-drift bin of §3.2. The per-occupied-slice number (1.15) is consistent
  with PDVD's and SBND's, but on 8 clusters it is corroboration, not proof.
- **Nothing here re-opens the terminal charge floor**, and §3.2 is a further
  argument not to.

## 9. Next (as written at the end of round 1; see §11–§14 for what round 2 did)

1. ~~Build **R1 as a single knob** `terminal_min_separation`~~ — **DONE, §11.**
   C++ default 0 ⇒ byte-identical, inserted between P3 and P4 exactly as doc 31
   §12.5 specifies, arms at **0.5 cm**, no exemption.
2. **Separately**, settle junction protection, which is what would unlock 0.8–1.0.
   The PR vertex list is out of scope at Steiner time (§6), so the candidates are
   the P4 extreme points and branch-like structure of the Steiner graph itself.
   §5's instrument grades any of them without a new arm: re-run
   `steiner_terminal_skeleton.py` with the proxy substituted for the PR vertex
   list and compare the "loses ALL core terminals" column against §5.2.
3. ~~Arms at R = 0.5 on PDVD **and SBND**~~ — **PARTLY DONE, §12.** The
   "SBND untouched" claim is carried by the compiled-config gate *and* a binary
   gate (§12.2, §12.3), exactly as this item demanded — SBND's operating point
   stays at 0 by the owner's instruction, so no SBND *knob-ON* arm was run and
   the question "what would 0.5 cm do to SBND" remains open.
4. ~~Grade on the fourth row, and re-run §4 and §5 on the knob-ON arms~~ —
   **DONE, §13.2 and §13.4.** The refit caveat is closed: both criteria survive
   against the live skeleton. The fourth row's verdict is *not* a clean pass —
   see §13.3, the STM tag set moves on 109/120 events.
5. PDVD's terminal-free runs (4.46 cm vs SBND 3.44, uBooNE 1.86) are a separate,
   still-open question and the surviving motivation for doc 31 R2.

## 10. Related

- `31_steiner-terminal-redesign.md` — rounds 1–8; §12.5 is the proposal this doc
  reviews, §12.4 the code facts it rests on, §6.1/§12.8 the grading table.
- `28_steiner-terminal-charge-pdvd-vs-sbnd.md` — the 500 e / 4000 e decomposition
  and the wire-crossing population.
- `34_ctpc-anisotropic-distance-metric.md` — the slice-pitch and drift-speed
  table §2 uses, including the uBooNE 1.101 mm/µs correction.
- `sbnd_xin/docs/pr/29_steiner-graph-build-port-audit.md` — D1/D12, the two
  terminal-filter knobs §2 tabulates and the uBooNE arms toggle.

---

# Round 2 — R1 built, gated, and flipped ON for PDVD (2026-09-04)

### Repro block (round 2)

```bash
# ---- build.  A new symbol + a doctest that uses it needs two passes: waf links
# the test against the PREVIOUS library in the same run (feedback: new-symbol
# test link/install).  build -k, install, then wcbuild.
cd toolkit && ./wcb build --notests -p -k; ./wcb install --notests -p; wcbuild
./build/clus/wcdoctest-clus                       # 292 passed, 0 failed
nm -DC local/lib/libWireCellClus.so | grep thin_by_min_separation   # freshness

# ---- the two libraries every arm below is pinned to (the tree is shared and a
# peer was rebuilding local/lib throughout; neither side used the live install)
#   /home/xqian/tmp/doc37/lib_base  10:45  symbol ABSENT  (= toolkit 3d057557)
#   /home/xqian/tmp/doc37/lib_new   10:48  symbol PRESENT (= 3d057557 + this change)

# ---- compiled-config gate, 18 jobs, OFF must diff to zero (sec 12.2)
abtest/compile_all_cfg.sh /home/xqian/tmp/doc37/cfg_{base,new}   # before / after
abtest/cmp_cfg.sh /home/xqian/tmp/doc37/cfg_base /home/xqian/tmp/doc37/cfg_new
# plus the two jobs compile_all_cfg.sh does not cover, by hand:
PDVD_KEEP_CFG=1 PDVD_PR_COMPILE_ONLY=1 PDVD_PR_TLA="-S dl_weights=''" \
    pdvd/run_pr_evt.sh -s <tag> 39252 0 0        # -> .wct-pr_<tag>.json
wcsonnet ... qlport/uboone-mabc.jsonnet -o uboone_mabc.json
# NOTE the work-dir tag appears in 6 path strings; normalize it before diffing
# or the 24 resulting lines read as a failure.

# ---- binary gates, OFF path, on the three jobs that bind CreateSteinerGraph
qlport/scripts/sweep_5384.sh d37offbase 12   # LD_LIBRARY_PATH=.../lib_base
qlport/scripts/sweep_5384.sh d37offnew  12   # LD_LIBRARY_PATH=.../lib_new
qlport/scripts/ab_check.sh d37offnew d37offbase
sbnd/sbnd_xin/run_pr_chain_batch.sh work-d97prodchk-mcp2k work-d37sbnd{base,new} sim

# ---- the PDVD arms.  Fresh tags staged from d34base so all three share
# byte-identical input point trees (M13: never write into an existing label).
for run_idx in $(cat manifest); do pdvd/scripts/stage_pr_tag.sh $run_idx <tag> d34base; done
docs/nf_sp_img_clus/scripts/doc37_run_thinning_arms.sh d37off0 .../lib_base 0   14
docs/nf_sp_img_clus/scripts/doc37_run_thinning_arms.sh d37off1 .../lib_new 0   12
docs/nf_sp_img_clus/scripts/doc37_run_thinning_arms.sh d37on05 .../lib_new 0.5  8
docs/nf_sp_img_clus/scripts/doc37_run_thinning_arms.sh d37rep  .../lib_new 0    3

# ---- the numbers
docs/nf_sp_img_clus/scripts/doc37_cmp_arms.py  d37off0 d37off1     # sec 12.3, the gate
docs/nf_sp_img_clus/scripts/doc37_cmp_arms.py  d37rep  d37off1     # sec 12.4, the control
docs/nf_sp_img_clus/scripts/doc37_arm_census.py d37off1 d37on05    # sec 13.1, 13.3
docs/nf_sp_img_clus/scripts/steiner_terminal_skeleton.py --nms 0.5 \
    "OFF:pdvd/work/*_d37off1/calib-pr-evt*.json" \
    "ON:pdvd/work/*_d37on05/calib-pr-evt*.json"                    # sec 13.2
```

**All four PDVD arms ran with `-S dl_weights=''`.** PDVD production has had the
DL/SCN vertex ON since 2026-09-04, so these arms are **not** the production
vertex configuration — but DL inference is not bit-stable (M4), and this is the
only configuration in which an OFF/OFF pair can be a byte-identity gate and an
OFF/ON pair can attribute a difference to the knob. Every §13 number is the
knob's effect in isolation, not a production forecast.

Baseline toolkit `3d057557`, wcp-porting-img `2e950881` at the start of round 2.

## 11. What shipped

The owner's instruction: *"Let's implement R1 with 0.5 cm for PDVD for now. We do
not need to change it for SBND production. Please then turn it on for PDVD
production."* §6's recommendation, executed as written.

### 11.1 The knob

`Grapher::Config::terminal_min_separation`, a length in the WCT system of units,
**C++ default 0 = no thinning**. When positive, `create_steiner_tree` runs a new
**phase 3b** between the existing phase 3 and phase 4:

> sort the surviving terminals by decreasing `calc_charge_wcp` — the same charge,
> the same threshold and the same `disable_dead_mix_cell` that selected them in
> phase 1 — and admit one only when no already-admitted terminal lies **strictly**
> within `terminal_min_separation` of it.

| file | what |
|---|---|
| `clus/inc/WireCellClus/SteinerThinning.h` + `clus/src/SteinerThinning.cxx` | the pure-geometry core, `thin_by_min_separation(ordered, min_separation)`. A uniform grid of cell edge `min_separation`, so 27 cells per candidate and O(N) overall; `std::floor`, not a cast, because half of PDVD sits at negative x and y. |
| `clus/src/SteinerGrapher.{h,cxx}` | the Config field, `thin_terminals_by_separation()` (builds the charge order), the phase-3b call site, and one greppable `steiner_thin: nterm_in=… nterm_out=… min_sep_cm=…` line per cluster at **DEBUG** (TRACE is compiled out of this build). |
| `clus/src/CreateSteinerGraph.cxx` | `get(cfg, "terminal_min_separation", …)`, negative clamped to 0 with a warning, round-tripped in `default_configuration()`. |
| `cfg/pgrapher/common/clus.jsonnet` | `cm.steiner(… terminal_min_separation=0)` with the key-suppression idiom. **Shared function — SBND, uBooNE and ICARUS bind it and stay at 0.** |
| `cfg/pgrapher/experiment/protodunevd/pr.jsonnet` | `steiner_terminal_min_separation=0` threaded into **both** `cm.steiner` call sites (`steiner` and `steiner_refresh`). Default 0: a bare `pr.jsonnet` run is *not* PDVD production. |
| `pdvd/wct-pr-perevt.jsonnet` (wcp-porting-img) | **`steiner_terminal_min_sep_cm = 0.5`** — the PDVD operating point, in the same place and for the same reason as `steiner_terminal_wire_tol` / `_adjacent_slice` / `_charge`. |
| `clus/test/doctest_steiner_terminal_thinning.cxx` | 8 cases. Pins the OFF pass-through as identity of the *sequence*; pins the strict `<` at exactly R; pins that **the first terminal offered can never be suppressed**, which is the invariant the phase-3b-before-phase-4 placement rests on; and brute-forces the pairwise guarantee over a 2197-point cloud. |
| `clus/test/doctest_clus_knob_defaults.cxx` | one case pinning the C++ default at 0. |

### 11.2 Two decisions worth recording

**Why phase 3b sits before phase 4.** Phase 4's
`steiner_terminals.insert(extreme_points…)` is unconditional: the extreme points
pin the ends of the tree. Thinning after it would let an extreme point arriving
mid-order be suppressed. Thinning before it means they cannot be. The doctest
`"the first terminal offered can never be suppressed"` is the mechanical statement
of that, and it fails if anyone moves the pass.

**Scope: exactly one call site.** `create_steiner_tree` is called from
`CreateSteinerGraph.cxx:276` and nowhere else. `improvecluster_2.cxx` — the
retiler — builds its **own** `Grapher::Config` from scratch, sets only
`terminal_charge_threshold`, and calls `find_steiner_terminals` directly rather
than `create_steiner_tree`. Its terminals are therefore untouched by this knob,
by construction and not by configuration. Checked, not assumed.

## 12. Gates

### 12.1 Which jobs actually bind the component

The change is in a **shared** component, so the first question is which jobs
reach it. Compiling all 18 live job configs and counting `CreateSteinerGraph`
instances answers it exactly:

| job | `CreateSteinerGraph` instances |
|---|---|
| `pdvd_pr` | **2** (`pr`, `prrefresh`) |
| `sbnd_pr` | **1** (`pr`) |
| `uboone_mabc` | **1** |
| every img / clus / nfsp / sim job on PDHD, PDVD, SBND | **0** |

So the img+clus `abtest` gate is **vacuous for this change** and was not run —
it would have PASSed without executing a line of the new code. The three jobs
above are gated instead, each on its own chain.

### 12.2 Compiled config — OFF diff-to-zero, ON exactly two keys

`abtest/compile_all_cfg.sh` (16 jobs) plus the PDVD PR driver and the uBooNE
MABC job compiled by hand: **18 of 18 byte-identical** with the knob off
(`cmp_cfg.sh`, normalized `_pnode`; `OVERALL: PASS`). With the driver at 0.5 cm,
the only difference anywhere is:

```
> "terminal_min_separation": 5,      (CreateSteinerGraph:pr)
> "terminal_min_separation": 5,      (CreateSteinerGraph:prrefresh)
```

5 = 0.5 cm in WCT units (`wc.cm` = 10). Nothing else moves, on any detector.

### 12.3 Binary gates — the OFF path on all three chains

| gate | arms | result |
|---|---|---|
| **uBooNE** `qlport/scripts/ab_check.sh` | `d37offnew` vs `d37offbase`, 35 events | **PASS** — 35/35 Bee zips content-identical, 35/35 tagger-compare logs identical |
| **SBND** manual hash, `run_pr_chain_batch.sh` | `work-d37sbndnew` vs `work-d37sbndbase`, 28 events | **PASS** — 28/28 `mabc-pr.zip`, 28/28 `pctree-pr-evt*.tar.gz`, `nusel-table.tsv` and `nusel-events.tsv` diff to zero |
| **PDVD** `doc37_cmp_arms.py` | `d37off1` vs `d37off0`, 120 events | **PASS** — 120/120 `mabc-pr.zip`, 116/116 `calib-pr-evt*.json`, 4 absent on **both** sides |

The 4 PDVD events with no dump on either side (039349 2/41/78 and 65) are the
known STM-only-PR mode: zero STM tags ⇒ no per-bundle PR ⇒ no dump at all (doc
25 §13.10). Absent-on-both is agreement; the comparator counts a **one-sided**
absence separately and fails on it.

### 12.4 The control that makes the PDVD gate mean something

`pdvd/run_pr_evt.sh` has **no `setarch -R`**, so the PDVD arms ran with ASLR on.
A pass on 120 events could in principle be luck, and a *failure* would have been
uninterpretable — "the change leaked" and "PDVD PR is not bit-stable" look the
same. So a fourth arm, `d37rep`, re-ran 3 events with the **same** library and
**same** config as `d37off1`:

```
d37rep vs d37off1:  mabc-pr.zip identical=3  calib-pr.json identical=3
```

PDVD PR is bit-reproducible under ASLR at this code state, so §12.3's identity is
attributable to the knob being off rather than to the axis being blunt.

### 12.5 Unit tests and freshness

```
# waf (this tree's build), all of clus:
./build/clus/wcdoctest-clus       292 passed | 0 failed  (22566 assertions)
  of which the 8 new thinning cases + the default pin: 13844 assertions

# CANONICAL cmake build -- the one upstream CI runs with -Werror.  A new public
# HEADER plus a new SOURCE file is exactly the shape that passes under waf and
# fails there (feedback: verify in the canonical build), so it was run:
cmake -S toolkit -B .../cmbuild -DCMAKE_PREFIX_PATH=.../local -DWCT_WITH_TESTS=ON
cmake --build .../cmbuild -j16                     rc=0, 100 %
cmake --build .../cmbuild -j16 --target wcdoctest  rc=0
.../cmbuild/wcdoctest -tc="steiner thinning*"      8 passed | 0 failed
  ZERO warnings from SteinerThinning.{h,cxx}, SteinerGrapher.* or
  CreateSteinerGraph.cxx.  The build's only warnings are pre-existing
  D3Vector.h ones raised from improvecluster_1.cxx and clustering_*.cxx.

lib_base  10:45  nm -DC | grep thin_by_min_separation -> 0 symbols
lib_new   10:48  nm -DC | grep thin_by_min_separation -> 1 symbol
sources   10:36-10:39
```

Both PDVD/SBND/uBooNE A-sides ran against `lib_base`, both B-sides against
`lib_new`, pinned via `LD_LIBRARY_PATH` — a peer was rebuilding `local/lib` in
this shared tree throughout, so neither side used the live install.

## 13. What the flip actually does — the fourth grading row, finally measurable

Arms `d37off1` (0 cm) and `d37on05` (0.5 cm), same binary, same 120 events,
**both with `-S dl_weights=''`**. That is *not* PDVD's production vertex
configuration — the DL/SCN vertex has been PDVD production since 2026-09-04 —
but it is the only configuration in which a difference can be attributed to this
knob rather than to DL's own run-to-run instability (M4). Everything below is
therefore the knob's effect in isolation, not a production forecast.

### 13.1 The knob does what §6 said it would

Two different denominators, and they should not be conflated:

| quantity | measured at | OFF | ON | change |
|---|---|---|---|---|
| terminals entering / leaving phase 3b | before phase 4 | 2 004 215 | 1 609 200 | **kept 0.8029** |
| `nterm` in the dump | after phase 4 | 1 970 130 | 1 591 413 | **−19.2 %** |
| Steiner cloud points | dump | 7 852 287 | 7 225 019 | −8.0 % |

Per event the kept fraction is **0.802 [p10 0.781, p90 0.818]** — §6 predicted
0.81 from the offline simulation, and the live knob reproduces it. The bound is
tight because it is geometric: every removed terminal was within 0.5 cm of a
higher-charge one that survived.

*Do not subtract row 1 from row 2.* `thin_out` (1 609 200) exceeds the dump's
`nterm` (1 591 413) even though phase 4 only ever **adds**, for two reasons that
are both bookkeeping: the `steiner_thin` lines are summed over **both**
`CreateSteinerGraph` instances — `pr` contributes 6925 and `prrefresh` 1068, and
the doc pr/23 refresh pass rebuilds clusters the first pass already counted — and
a cluster whose graph comes out empty (§13.3) never reaches the dump at all. The
two columns measure different populations on purpose; they are not a balance
sheet.

Wall time **improves**: median 36 s → 33 s per event, 5557 s → 5010 s over the
arm (−9.8 %); peak RSS median 1.21 → 1.15 GB. A smaller terminal set makes a
smaller Steiner tree.

### 13.2 What does NOT move

- **TGM is untouched.** 2116 tags on both arms, `nclus_tgm_eval` 6956 on both,
  and **0 of 116 events** differ. The reading — offered as an inference, not a
  measurement — is that `TaggerCheckTGM` does not consume the Steiner terminals,
  while STM does.
- **Total reconstructed track length is flat**: 81 840 → 81 681 cm, **−0.19 %**.
- **FC barely moves**: 2242 → 2233 (−0.4 %), unchanged on 103 of 116 events.
- Every event that had a main vertex still has one (116/116).

**And — the caveat §8 said could not be closed offline is now closed.** §4 and §5
were computed against the skeleton and vertices of the *un-thinned* arm; with a
live knob both refit, so the honest worry was that the survivors sit on a
skeleton that has moved out from under them. Re-running
`steiner_terminal_skeleton.py` on the two live arms answers it:

| | OFF (0 cm) | ON (0.5 cm) |
|---|---|---|
| clusters with a fitted skeleton | 147 | 179 |
| **terminal → fitted skeleton**, median cm | 0.90 [0.73, 1.23] | **0.91** [0.73, 1.27] |
| same for the whole Steiner cloud (matched control) | 1.38 | 1.31 |
| **excess** (terminal − cloud) | −0.48 | **−0.43** |
| terminal→skeleton p90 (the tail) | 2.62 | **2.28** |
| vertices of degree ≥ 2 with terminals | 1044 | 1081 |
| of those, with ≥ 1 terminal inside 1 cm of the vertex | 814 (**78.0 %**) | 848 (**78.4 %**) |
| vertex-core terminals, total | 1798 | 1477 (−17.9 %) |
| vertex-core terminals per vertex, median | 1.5 | 1.0 |

Both owner criteria that §4 and §5 exist to grade **survive the refit**:

- *follows the skeleton* — the terminals are still **closer to the fitted
  trajectory than the Steiner cloud is** (excess −0.43 cm), the median distance
  moves by 0.01 cm, and the p90 tail gets *shorter*, not longer.
- *does not miss terminals near the vertex* — the fraction of multi-branch
  vertices holding at least one terminal within 1 cm is **78.0 % → 78.4 %**,
  i.e. flat. The count per vertex falls 1.5 → 1.0 at the median, which is the
  price and is above zero. (The two vertex populations are not matched — the ON
  arm has 37 more vertices — so read this as two populations, not a paired test.)

One incidental check falls out and is worth keeping: applying a *further* 0.5 cm
thinning to the ON arm's terminals removes **nothing** (C3 "plain" = 1.00 [1.00,
1.00], and 0/1 vertices lose their core versus 12–25 on the OFF arm). The shipped
C++ is idempotent at its own radius, which is what it should be if it implements
the rule the offline simulation modelled — an end-to-end check that the knob and
§5's instrument agree.

### 13.3 What DOES move — and this is the part that needs a hand scan

**⚠ The STM-tagged cluster-id SET differs on 109 of 120 events.** The *count*
barely moves (685 → 699, +2.0 %; 43 events up, 40 down, 33 unchanged), which is
why a count-only census would have called this flat. The identity of the tagged
clusters is what changed.

That propagates, because PDVD runs the per-bundle PR **only on STM-tagged
bundles** (doc 25 §13.10): a different tagged set means a different set of
bundles is reconstructed at all.

| | OFF | ON | total | per-event median ratio |
|---|---|---|---|---|
| segments | 2004 | 2121 | +5.8 % | **0.923** [0.462, 2.500] |
| vertices | 2417 | 2658 | +10.0 % | **0.976** [0.529, 2.333] |
| showers | 458 | 452 | −1.3 % | 0.750 |

The totals and the medians disagree in sign, and that is the finding: the change
is **not a small systematic shift, it is large per-event churn that mostly
cancels**. **56 of 116 events change their segment count by ≥ 5**, in both
directions — 039252/5 goes 36 → 203, 039252/8 goes 42 → 144, 039349/7 goes
41 → 7, 039253/4 goes 43 → 10. Δnseg and Δnvtx correlate at **0.986** (a split
adds a segment and a vertex together), and Δnseg correlates with Δn(STM tags) at
only **0.033** — so it is *which* bundle is tagged, not *how many*, that drives it.
On the 33 events where even the STM count is unchanged, 16 still move by ≥ 5.

Three events (039349 2, 41, 78) went from **zero** STM tags to non-zero and now
produce a `calib-pr` dump where before they produced none.

**A second, smaller cost, bounded and worth stating plainly: 47 more clusters
lose their Steiner graph entirely.** The `< 2 terminals` check in
`create_steiner_tree` runs *after* phase 4, so the extreme points normally rescue
a cluster the thinning took down to one terminal — but not always, and the
doctest invariant ("the first terminal offered can never be suppressed") does not
claim otherwise. Counted from the logs, which is the ungated axis — unlike
`nsteiner_clusters`, these warnings are upstream of the STM gating that §13.3
shows moves:

| | OFF | ON |
|---|---|---|
| clusters reaching phase 3b | — | 7993 (6925 `pr` + 1068 `prrefresh`) |
| thinned to < 2 terminals at phase 3b | — | 928 |
| `... terminal(s) remain after filtering` (i.e. **still** < 2 after phase 4) | 4 | **51** |
| `produced no steiner_graph` | 552 | 596 |
| `< 2 steiner terminal(s) found` at phase 1 (untouched, as expected) | 548 | 545 |

So phase 4's extreme points rescue 877 of the 928, and **47 clusters do not
recover**, out of 7993 — **0.59 %**. Every one of the 928 entered phase 3b with
**at most 3 terminals** (the maximum `nterm_in` among them is exactly 3): only
clusters that were already at the edge of the ≥ 2 requirement can be pushed over
it. That is a real cost, it is small, and it is bounded by construction rather
than by luck.

### 13.4 Verdict on the fourth grading row

Doc 31 §6.1's fourth row — *does not cost physics* — has been owed for nine
rounds because it needed the knob. It now has an answer, and the answer is
**"not established, and bigger than a cleanup"**:

- **Three of the owner's four criteria hold, and hold against the LIVE refitted
  skeleton, not just the offline simulation** (§13.2): terminals stay closer to
  the fitted trajectory than the cloud is, the inefficiency is 19 % of
  near-duplicates, and the density that motivated the round is what got removed.
  The fourth — *do not miss terminals near the vertex* — also holds: 78.0 % →
  78.4 % of multi-branch vertices keep a terminal within 1 cm.
- But the downstream PR output is **not** approximately unchanged. Aggregates
  are flat and TGM is untouched, yet the STM tag set moves on 109/120 events and
  the per-event segmentation swings by factors of 2–6 on the worst events.
  0.59 % of clusters (47 of 7993) also lose their Steiner graph outright, all of
  them clusters that reached the thinning with ≤ 3 terminals.
- **Nothing here says the ON arm is worse.** It may well be better — 19 % fewer
  terminals is a less over-constrained tree, and the wall time supports that.
  It says the two arms reconstruct many events *differently*, and only a hand
  scan can say which is right. That scan is now the blocking item, and it is
  cheap: both arms' Bee zips already exist, same event, same paths.

**Recommended scan pairs** (`work/<dir>/mabc-pr.zip`, not uploaded — owner
uploads):

| event | why | OFF | ON |
|---|---|---|---|
| 039252/5 | biggest gain, 36 → 203 segments | `039252_5_d37off1` | `039252_5_d37on05` |
| 039349/7 | biggest loss, 41 → 7 segments | `039349_7_d37off1` | `039349_7_d37on05` |
| 039349/2 | zero STM tags → tagged | `039349_2_d37off1` | `039349_2_d37on05` |

**Reverting is one TLA**: `PDVD_PR_TLA="-S steiner_terminal_min_sep_cm=0"`, and
the compiled config is then byte-identical to the pre-flip one (§12.2).

## 14. Arms and records of round 2

| label | what | where |
|---|---|---|
| `d37off0` | PDVD PR, 120 evts, **baseline binary**, knob absent | `pdvd/work/*_d37off0/` |
| `d37off1` | PDVD PR, 120 evts, new binary, knob 0 | `pdvd/work/*_d37off1/` |
| `d37on05` | PDVD PR, 120 evts, new binary, **0.5 cm** | `pdvd/work/*_d37on05/` |
| `d37rep` | PDVD PR, 3 evts, repeat of `d37off1` (§12.4) | `pdvd/work/*_d37rep/` |
| `d37offbase` / `d37offnew` | uBooNE MABC, 35 evts each | `qlport/scripts/sweep/` |
| `work-d37sbndbase` / `work-d37sbndnew` | SBND PR chain, 28 evts each | `sbnd/sbnd_xin/` |

Every tag is fresh; nothing under an existing `work/`, `sweep/` or `snap/` label
was written, regenerated or deleted (M13). All three PDVD arms were staged from
`d34base` with `scripts/stage_pr_tag.sh`, so they share byte-identical input
point trees.

Scripts, all committed under `pdvd/docs/nf_sp_img_clus/scripts/`:
`doc37_run_thinning_arms.sh` (runs one arm against a pinned library),
`doc37_arm_census.py` (the per-event PR-level census; its header carries the
two-denominator and STM-conditioned-on-TGM warnings),
`doc37_cmp_arms.py` (the two-axis comparator).

---

# Round 3 — the flip graded where production actually lives (2026-09-04)

## 15. The DL-vertex arms

Round 2's arms all ran `-S dl_weights=''`. That is the only configuration in
which an OFF/OFF pair can be a byte-identity gate, but it is **not** what PDVD
production runs: the uBooNE SCN vertex has been the PDVD main-vertex result
since 2026-09-04 (doc 28 §27). §13 therefore graded the knob somewhere
production does not live. The owner's call was to close that by running the
missing arm rather than by moving production to the validated configuration.

Three arms, all on `lib_new`, all at the driver's production `dl_weights`:

| tag | `sep_cm` | events | what it is for |
|---|---|---|---|
| `d37dloff` | 0 | 120 | production as it was before the flip |
| `d37dlon` | 0.5 | 120 | production as it is after the flip |
| `d37dlrep` | 0 | 20 | **a repeat of `d37dloff`** — the DL noise floor |

The repeat arm is not optional. DL inference is not bit-stable, so with it on
the OFF/ON difference is knob **plus** DL and means nothing until it is compared
against a run of one arm against itself. Round 2 did not need this (its repeat
was bit-identical, 3/3); round 3 does.

**The DL vertex is genuinely active in these arms**, checked positively rather
than assumed: the compiled config carries the SCN weights, neither failure
signature appears (`DL vertex failed`, `dl_weights path not found` — 0 of each,
and a silent fall-back to geometric is what those warn about), and on
039252/1 the main vertex sits 19 cm from the geometric arm's.

### 15.1 The knob clears the noise floor by a wide margin

`doc37_stm_set_diff.py`, comparing the **set** of STM-tagged cluster ids:

| comparison | n | STM set differs | mean sym-diff | TGM set differs |
|---|---|---|---|---|
| `d37dloff` vs `d37dlrep` — **same config, the floor** | 20 | **0 (0.0 %)** | 0.00 | **0 (0.0 %)** |
| `d37dloff` vs `d37dlon` — the knob, all 120 | 120 | **95 (79.2 %)** | 1.78 | 0 (0.0 %) |
| (round 2, geometric vertex + pre-flip cfg, for reference only — §15.3) | 120 | 109 (90.8 %) | 2.39 | 0 (0.0 %) |

**§13.3's finding survives in the production configuration.** 79 % against a
**zero** floor is not DL instability; the thinning really does move which
clusters the STM tagger claims, on the large majority of events. PDVD PR's
tag sets turn out to be exactly reproducible run-to-run even with DL on.

*(An earlier version of this table reported the floor as 1/20. That was a
measurement error, not a result: the wait loop keyed on log-file **existence**,
and a log exists from the moment a job starts, so one event was read
mid-write. Completion is `pr_resource_*.txt`, which is written at the end.)*

### 15.2 The aggregates, and why they must not be compared to §13's

Same census, `d37dloff` → `d37dlon`, 115 events with a dump on both sides:

| | OFF | ON | change | per-event median ratio |
|---|---|---|---|---|
| terminals kept at phase 3b | 1 979 252 | 1 588 761 | **0.8027** | — |
| `nterm` (dump) | 1 956 358 | 1 579 851 | −19.3 % | — |
| **segments** | 1971 | 1912 | **−3.0 %** | **1.000** [0.500, 2.167] |
| **vertices** | 2385 | 2305 | **−3.4 %** | **1.000** [0.476, 2.222] |
| **total track length** | 75 838 cm | 78 689 cm | **+3.8 %** | 0.991 |
| showers | 421 | 367 | −12.8 % | — |
| **TGM** | 2112 | 2112 | **0.0 %, 0 of 115 events** | — |
| STM | 650 | 670 | +3.1 % | — |
| FC | 2218 | 2210 | −0.4 % | — |
| main vertex present | 115 | 115 | — | — |

The knob's own effect is identical to §13 (0.8027 vs 0.8029 kept — as it must
be, phase 3b is upstream of everything else here).

**The downstream read differs from §13's, and this doc will NOT attribute that
difference to the DL vertex.** §13's arms ran before the doc-36 flip
(`38245d18` / `605833eb`, 2026-09-04 11:31–11:32: PDVD PR moved to the
anisotropic ctpc metric and `good_point_pitch_frac` was retired to 0); §15's ran
after it. **Three things changed between the two rounds — metric, floor and
vertex — so no §13-vs-§15 comparison isolates any one of them.** §15.3 measures
which one it actually was. What can be said, within each round's own epoch:

| | §13 (pre-flip cfg, geometric) | §15 (post-flip cfg, DL — production) |
|---|---|---|
| segments | +5.8 % | **−3.0 %** (per-event median 1.000) |
| vertices | +10.0 % | **−3.4 %** (per-event median 1.000) |
| track length | −0.19 % | **+3.8 %** |

Both are valid statements about the knob **in their own epoch**, and the one that
describes production today is the right-hand column.

- **TGM is untouched for the third independent time** (0 of 115 events, two arm
  pairs, two vertex configurations, two cfg epochs). Whatever `TaggerCheckTGM`
  reads, it is not the Steiner terminals — and §15.3 shows it *does* move for
  other reasons, so this is a real invariance and not a blunt instrument.

The per-event churn does **not** go away: 53 of 115 events still move segment
count by ≥ 5, 47 up and 55 down. So §13.3's headline stands — the change is
large per-event churn that mostly cancels — but in the production configuration
the residual after cancellation points the right way instead of the wrong one.

### 15.3 A claim this round made, then RETRACTED — and what it really was

**An earlier version of this section reported a pointer-order determinism defect.
That was wrong. It is retracted here, and the measurement that replaces it
belongs to doc 36's flip, not to any determinism problem.**

What happened: I compared `d37off1` (round 2) against `d37dloff` (round 3) at the
same knob setting, described the pair as "only `dl_weights` differs", and read the
result — TGM set differing on 33/120 events — as an allocation-order effect. The
pair does **not** differ only in `dl_weights`. Round 2's arms ran before the
doc-36 flip and round 3's after it, so the comparison also spanned the
anisotropic ctpc metric and the retirement of `good_point_pitch_frac`. I had
warned a concurrent round about exactly this class of cross-epoch comparison an
hour earlier, and then published one.

The fix was one 20-event arm, `d37geomoff` — knob 0, **geometric** vertex, run in
the **post-flip** epoch — which makes every factor separable:

All four rows on the **same 20 events**, so the fractions are comparable — an
earlier version mixed a 20-event row with a 120-event one and drew a magnitude
conclusion from what was partly an `n` artefact:

| comparison | what varies | STM set differs | TGM set differs |
|---|---|---|---|
| `d37dloff` vs `d37dlrep` | **nothing** (a repeat) | 0 / 20, sym 0.00 | 0 / 20, sym 0.00 |
| `d37geomoff` vs `d37dloff` | **the vertex only** | **0 / 20**, sym 0.00 | **0 / 20**, sym 0.00 |
| `d37off1` vs `d37geomoff` | **the cfg epoch only** | 20 / 20, sym **4.50** | **7 / 20 (35 %)**, sym 0.40 |
| `d37dloff` vs `d37dlon` | **the thinning knob** | 19 / 20, sym **2.80** | **0 / 20**, sym 0.00 |

(the knob row over all 120 events is 95 / 120 and 0 / 120, §15.1)

**The DL vertex is inert for both cosmic taggers — 0 of 20, on both.** Which is
what the pipeline said all along and what I should have trusted over a
confounded measurement: the SCN weights reach exactly one component
(`TaggerCheckNeutrino:pr`), and `tagger_check_tgm` / `tagger_check_stm` both run
before it. The supporting facts in the retracted section were all correct; the
comparison they were attached to was not.

**The 33/120 belongs to doc 36's flip, and that is a real result worth having.**
With the vertex now proven inert, the epoch-only row isolates it: moving PDVD PR
to the anisotropic ctpc metric and retiring the 0.35 floor **moves the STM-tagged
set on 20 of 20 events and the TGM set on 7 of 20 (35 %)**. Handed to doc 36/38
as theirs.

**How that compares to this doc's flip, stated carefully.** On STM the two are
**comparable in event fraction** — 20/20 against 19/20 — and differ in *degree*:
the epoch flip moves 4.50 clusters per event against the thinning's 2.80, i.e.
about 1.6×. It is **not** the order-of-magnitude difference an earlier version of
this section claimed; that came from setting a 20-event row against a 120-event
one. The genuinely sharp separation is on **TGM: 35 % against 0 %**, and that one
is a clean discriminator because the two were measured on the same events with
the same instrument.

Two consequences for anyone reading this doc:

- **The cross-epoch caveat in §15.2 is not boilerplate.** Two arm sets taken an
  hour apart on this tree can differ by a production flip; check the cfg mtimes
  against the arm timestamps, not the commit message.
- **The TGM invariance under thinning is strengthened, not weakened.** TGM moves
  35 % of the time for the metric flip and 0 % of the time for this one, on the
  same 20 events and the same instrument. So "TGM is untouched" is a
  discriminating measurement rather than a blunt one — the instrument demonstrably
  responds to a different change on the same events.

### 15.4 An independent grade, from doc 38's instrument

Doc 38 (the gap-aware end trim round, a concurrent session) measures PDVD
trajectory **coverage** and **unsupported trajectory** — two axes this round does
not have. Grading its own arms it had to separate this flip out of a
three-change comparison, and the number it reports for the thinning alone,
holding the anisotropic metric and the `good_point_pitch_frac` floor fixed
(`38_gap-aware-end-trim.md`, commit `7ece5327`):

> coverage **71.1 → 71.7 %**, unsupported trajectory **12.6 → 12.7 %**

**Not my measurement and not re-derived here** — quoted from that doc, and read
from its committed text rather than from a message about it. It is worth
recording because it is the only grade of this flip on an axis §13/§15 cannot
see, and it points the same way as §15.2: a small gain on coverage, neutral on
support. It also comes with its own caveat from that doc — the two arms hold
different cluster sets (2176 vs 2159), because the PR stage the metric reads is
downstream of the STM gating §15.1 shows moving.

It does **not** replace the hand scan. Coverage and support are both aggregate
statistics, and §13.3/§15.2's whole point is that this change's aggregates are
nearly flat while individual events move a lot.

### 15.5 Verdict

The flip is graded where production lives, and in that epoch the answer is good:
the three set-level criteria hold, TGM is untouched (0 of 115, while doc 36's
flip moves it on 35 % of events — §15.3), aggregate track length rises 3.8 %, and
segments and vertices sit at a per-event median ratio of exactly 1.000. This is
a statement about the post-flip epoch, not a comparison with §13's. The one thing
§13.4 asked for is still owed and is unchanged by this round — **a hand scan**,
because 53 of 115 events reconstruct differently and no offline metric says which
version is right. Scan pairs, now in the production configuration:

| event | why | OFF | ON |
|---|---|---|---|
| 039252/5 | large mover | `039252_5_d37dloff` | `039252_5_d37dlon` |
| 039349/7 | large mover, other direction | `039349_7_d37dloff` | `039349_7_d37dlon` |
| 039349/2 | zero STM tags → tagged | `039349_2_d37dloff` | `039349_2_d37dlon` |

## 16. Still open

1. **The hand scan** above. Everything else in §13.4, §15.2 and §15.4 is
   measured; this is the only thing that can say whether the flip is an
   improvement.
2. ~~The pointer-order dependence §15.3 found.~~ **RETRACTED — there is no such
   defect.** See §15.3: the comparison spanned doc 36's production flip. The
   DL vertex is inert for both cosmic taggers (0/20) and PDVD PR's tag sets are
   exactly reproducible run-to-run (0/20). Doc 38 §5 recorded the retracted
   claim on my report and has been told.
3. **SBND at 0.5 cm.** Still unrun and still unasked-for: SBND stays at 0 by the
   owner's instruction, but §3.3 and §5 both say the same radius would bite it
   as hard as PDVD, so the number is not predictable from PDVD's.
4. Junction protection, which is what would unlock 0.8–1.0 cm (§6, §9 item 2).

## 17. Arms and records of round 3

| label | `sep_cm` | vertex | cfg epoch | events | role |
|---|---|---|---|---|---|
| `d37dloff` | 0 | DL/SCN | post doc-36 flip | 120 | production before the thinning flip |
| `d37dlon` | 0.5 | DL/SCN | post doc-36 flip | 120 | production after it |
| `d37dlrep` | 0 | DL/SCN | post doc-36 flip | 20 | repeat of `d37dloff` — the noise floor (0/20) |
| `d37geomoff` | 0 | geometric | post doc-36 flip | 20 | isolates the vertex from the cfg epoch (§15.3) |

All staged fresh from `d34base` with `stage_pr_tag.sh`, all pinned to
`/home/xqian/tmp/doc37/lib_new`. Round 2's arms (`d37off0/off1/on05/rep`) are in
the **pre**-doc-36-flip epoch; §14 lists them.

**Completion is `pr_resource_*.txt`, not the log file.** A log exists from the
moment a job starts, so a census keyed on log existence can read an event
mid-write — that is what produced the retracted 1/20 floor in §15.1. Two
`d37dlon` events were also re-run at the end because a `PDVD_PR_COMPILE_ONLY=1`
probe of mine had removed one of their logs, leaving the arm non-uniform; the
numbers in §15.2 are from the repaired, uniform arm and are unchanged by the
repair.

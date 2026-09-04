# Steiner terminals on PDVD, SBND and MicroBooNE — and does the round-8 alternative make sense?

**Status: measurement and design review. No toolkit change, no production-config
change, no arm of PDVD or SBND was re-run, so no A/B gate is owed and none is
claimed.** The only new reconstruction is two 35-event MicroBooNE arms, and they
run a *copy* of the uBooNE chain whose compiled config at default TLAs is
byte-identical to the production one (§0.2).

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

**MicroBooNE is not sparser than PDVD. On the whole population it is the densest
of the three, and it has been for a decade.** That single fact reframes the
round-6 complaint: PDVD is not an outlier, it sits between SBND and uBooNE.

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

## 9. Next

1. Build **R1 as a single knob** `terminal_min_separation` (C++ default 0 ⇒
   byte-identical), inserted between P3 and P4 exactly as doc 31 §12.5
   specifies, and take it to arms at **0.5 cm**. No exemption is needed at that
   radius, which is what makes it the shippable step.
2. **Separately**, settle junction protection, which is what would unlock 0.8–1.0.
   The PR vertex list is out of scope at Steiner time (§6), so the candidates are
   the P4 extreme points and branch-like structure of the Steiner graph itself.
   §5's instrument grades any of them without a new arm: re-run
   `steiner_terminal_skeleton.py` with the proxy substituted for the PR vertex
   list and compare the "loses ALL core terminals" column against §5.2.
3. Arms at R = 0.5 on PDVD **and SBND** — §3.3 and §5 both show the same
   radius bites SBND as hard as PDVD, so "PDVD-only in effect" is false for this
   lever, and the compiled-config gate, not a geometry argument, must carry the
   "SBND untouched" claim.
4. Grade on the fourth row, and re-run §4 and §5 on the knob-ON arms — with a
   live knob the vertices move, which is the caveat §8 cannot close offline.
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

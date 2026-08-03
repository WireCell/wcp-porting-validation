# 49 — Why the STM tagger never fits evt285185 cluster 21: two different "contained"

Chased from the STM hand scan on the 30-event `dq48v3` sample: bundle grp 14
(t = 1.546 µs, main 21, +1 merge fragment, companions 14) of evt285185 looks like
a textbook stopping muon — it enters at the TPC1 anode and stops 142 cm inside —
but the `stm fit` column read `skip` and no dQ/dx panel was drawn.

**Two separate things were wrong. Neither is a tagger bug.**

1. `skip` was a **display artifact**: the tagger *did* log its reason, the log
   line was torn mid-word, and `nusel_extract` could not identify the truncated
   text. Fixed here; the column now reads `contained`.
2. `contained` is the **real answer**, and it exposes a genuine inconsistency:
   `TaggerCheckSTM` is the only tagger never handed a `fiducial` /
   `fv_tolerance`, so it tests containment against the **un-inset per-face
   sensitive volume** while `tagger_check_tgm` and `tagger_check_fc` use the
   **inset `sbnd_pr_fv` box**. Cluster 21's endpoint lands in the 2.9 cm gap
   between them: contained to STM, an exiter to FC, same event, same function.

**Status: FIXED and SHIPPED ON for SBND** (§6). `TaggerCheckSTM` now accepts
`fiducial` + `fv_tolerance` and SBND passes the same objects TGM and FC get.
**NOT bit-identical**: STM 14 → 44 tags on the 30-event sample, TGM and FC
unchanged. The knob-off path is byte-identical (compiled config *and* output).

§4a records a separate, pre-existing finding that came out of this A/B and is
more consequential than the fix: **the SBND PR chain is not run-to-run
reproducible unless run under `setarch x86_64 -R`**, at ±7 STM tags out of ~44.
Any A/B on this chain smaller than that is meaningless, which includes doc 48 §4.

---

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# the two containment boxes, straight out of the run's own geometry dump
grep -n "sensvol\|X planes:" work-mcp10-dq48v3/nusel_evt285185/wct_nusel_evt285185.log

# what the tagger actually said about cluster 21 (line 181 is the torn one)
grep -n "check_stm_conditions" work-mcp10-dq48v3/nusel_evt285185/wct_nusel_evt285185.log

# this cluster, per merge component
python3 scripts/analysis/stm/stm_fv_census.py --detail 285185:21

# the census over all 30 scan events (prediction, before the fix)
python3 scripts/analysis/stm/stm_fv_census.py            # -> §4

# the A/B.  setarch x86_64 -R IS MANDATORY -- see 4a.
F="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000"
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
for arm in off on; do
  [ $arm = off ] && EX="-no-stm-fv" || EX=""
  tag=d49s$arm
  for suff in mcp10 mcp1000 mcp1000b; do        # symlink farm from the dq48v3 inputs
    mkdir -p work-$suff-$tag
    for d in work-$suff-dq48v3/ql_evt* work-$suff-dq48v3/evt*; do
      ln -sfn "$(readlink -f "$d")" "work-$suff-$tag/$(basename "$d")"; done
  done
  SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
    SBND_WORK_ROOT=$PWD/work-mcp10-$tag setarch x86_64 -R ./run_nusel_evt.sh data all $F $EX
  for e in $(seq 10 19); do SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-$tag  setarch x86_64 -R ./run_nusel_evt.sh data 1 $F $EX; done
  for e in $(seq 20 29); do SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000b-$tag setarch x86_64 -R ./run_nusel_evt.sh data 1 $F $EX; done
done
```

`scripts/analysis/stm/stm_fv_census.py` reads only committed scan products (`work-*-dq48v3/`); it runs
no wire-cell and takes ~40 s.  Scan labels: **`work-{mcp10,mcp1000,mcp1000b}-d49soff`**
(knob off) and **`-d49son`** (knob on), both under `setarch -R`.

---

## 1. The two boxes

`TaggerCheckSTM.cxx:2137` calls

```cpp
auto fc_result = Facade::cluster_fc_check(cluster, m_dv);   // no fiducial, no tolerance
```

With `fiducial == nullptr`, `cluster_fc_check`'s `inside_fv` lambda falls back to
`FiducialUtils::inside_fiducial_volume(p)` → `m_sd.fiducial->contained(p)`. In
`sbnd/clus.jsonnet` `MakeFiducialUtils` gets `fiducial=dv`, and
`DetectorVolumes::contained()` (`aux/src/DetectorVolumes.cxx:112`) is
`contained_by(point).valid()`, which tests each face's
`IAnodeFace::sensitive()` bounding box. `AnodePlane.cxx:280-300` builds that box
as `x ∈ [anode_x, cathode_x]` (intersected over the three planes, extended half a
pitch in y/z) — i.e. it runs to the **W plane**, and knows nothing about the
`FV_*` metadata or the `FV_*_margin` values.

The run log prints it, so this is measured, not inferred:

```
<AnodePlane:apa1> X planes: "cathode"@0.0045m, "response"@1.9205m, "anode"@2.0145m, dirx=-1
<AnodePlane:apa1> face:1 with 3 planes and sensvol: [(4.5 -1999.65 0) --> (2014.5 1999.65 5010)]
<AnodePlane:apa0> face:0 with 3 planes and sensvol: [(-2014.5 -1999.65 0) --> (-4.5 1999.65 5010)]
```

| | anode \|x\| | \|y\| | z | CPA hole |
|---|---|---|---|---|
| **STM** (union of `sensitive()`, no margin) | ≤ **201.45** | ≤ **199.965** | 0 … 501.0 | \|x\| < 0.45 |
| **TGM / FC** (`sbnd_pr_fv` + `sbnd_pr_fv_margins`) | ≤ **198.55** | ≤ **196.312** | 3.85 … 497.15 | none (one box) |

The gap is **2.90 cm** at the anode, **3.65 cm** in y, **3.85 cm** at each z
face. Every cluster ending inside that shell is "fully contained" to STM and an
exiter to FC.

Each gap has **two** components, and it matters for §4 that they be kept apart:

| wall | sensitive box exceeds `sbnd_pr_fv` by | + margin | = gap |
|---|---|---|---|
| anode x | 201.45 − 201.05 = **0.40** | 2.5 | 2.90 |
| y | 199.965 − 199.312 = **0.65** | 3.0 | 3.65 |
| z (each face) | 0.85 (0 vs 0.85; 501.0 vs 500.15) | 3.0 | 3.85 |

So even at **zero margin** the STM box is larger than the FV box everywhere. The
margins widen an inconsistency that already exists.

This is the same defect [27_fc-tgm-consistent-fv.md](27_fc-tgm-consistent-fv.md)
fixed for FC — that doc's own words, "Without them FC fell back to FiducialUtils
(per-face sensitive volumes, no margin), which is both more permissive at every
wall than TGM's inset box and holed at the CPA slab" — describe exactly where STM
still is. FC and TGM were converted; STM was not, and neither was
`TaggerCheckNeutrino.cxx:518`, whose identical bare call feeds
`tagger_info.match_isFC` into the neutrino tagger's feature vector.

> Docs bug found on the way: `sbnd/clus.jsonnet:460` says "DetectorVolumes
> implements IFiducial (**box FV from its metadata**)". It does not — the code
> path above never reads the metadata. The comment 14 lines later ("union of
> per-face sensitive volumes, which excludes the CPA slab (|x| < 0.45 cm)") is
> the correct one, and the 0.45 cm it names is confirmed by the log dump.
> Corrected (comment-only; compiled config proven byte-identical).

### 1a. The prototype has one FV, and it is inset

The decisive argument for the fix is that this split does not exist upstream.
`check_stm`, `check_tgm` and `check_fully_contained` are all **member functions
of the same `ToyFiducial` object**, and all call the same
`inside_fiducial_volume(p, offset_x)` with `tolerance_vec = NULL` — i.e. the
identical boundary polygons:

| prototype symbol | file:line |
|---|---|
| `WCPPID::ToyFiducial::check_stm` → `inside_fiducial_volume(p1, offset_x)` | `pid/src/ToyFiducial.cxx:405` → `:456` |
| `WCPPID::ToyFiducial::check_tgm` | `pid/src/Cosmic_tagger.h:1331` |
| `WCP2dToy::ToyFiducial::check_fully_contained` / `check_tgm` | `2dtoy/src/ToyFiducial.cxx:816` / `:905` (→ `:851`) |

uBooNE builds that one object with **`boundary_dis_cut = 3*units::cm`**
(`uboone_nusel_app/apps/prod-wire-cell-matching-nusel.cxx:348-351`, args
`3*cm, top 116, bottom −116, upstream 0, downstream 1037, anode 0, cathode 256`),
and the constructor bakes the inset into the polygons —
`boundary_xy_x.push_back(m_anode + boundary_dis_cut)`,
`boundary_xy_y.push_back(m_bottom + boundary_dis_cut)`, … plus an extra 1 cm on
the z faces (`ToyFiducial.cxx:117-136`). There is **no configuration path by
which uBooNE's STM and TGM could see different volumes.**

So the toolkit's split is an **undocumented porting divergence**, not an
inherited convention — `clus/docs/porting/porting_dictionary.md` has no entry on
fiducial or sensitive volumes (M15: surfaced here rather than silently picked).
And SBND's 2.5 / 3 / 3 cm margins are the same order as uBooNE's 3 cm, so passing
`sbnd_pr_fv_margins` **restores prototype parity** rather than inventing a
number.

One asymmetry survives, inherited from `cluster_fc_check` and shared with FC: the
`check_signal_processing` / `check_dead_volume` fallback branches still use
`FiducialUtils`' own volume internally (`FiducialUtils::check_dead_volume` opens
with `if (!inside_fiducial_volume(p)) return false;`). Only the **direct**
containment tests take the configured box. STM and FC are now identical in this
respect, which is what "consistent" was asked to mean.

## 2. Cluster 21

`scripts/analysis/stm/stm_fv_census.py --detail 285185:21`, on the Bee clustering layer (the only place
per-point `real_cluster_id` survives):

```
cluster 21: 1211 points, merge components [10, 22]

  component real_cluster_id=10: 40 pts, farthest-pair 50.4 cm
    x [  -38.96,  -18.95]  y [   24.31,   70.64]  z [  301.12,  308.17]
    endA (  -38.65,   24.66,  306.97)  inside_STM=True  outside_FC= 0.00 cm (-)
    endB (  -18.95,   70.64,  301.12)  inside_STM=True  outside_FC= 0.00 cm (-)

  component real_cluster_id=22: 1171 pts, farthest-pair 141.9 cm
    x [  129.93,  201.20]  y [   -2.05,   42.98]  z [   20.93,  135.38]
    endA (  129.93,   42.89,  135.23)  inside_STM=True  outside_FC= 0.00 cm (-)
    endB (  201.20,   -1.62,   20.93)  inside_STM=True  outside_FC= 2.65 cm (anode-x)
```

The 1171-point component is the candidate STM: it runs 141.9 cm from a deep
interior end at x = 129.93 to **x = 201.20 cm** at the TPC1 anode. That endpoint
is

- **0.25 cm inside** the STM box (201.45) → no exit candidate → `is_fc = true` →
  `check_stm_conditions` returns at Mid Point A, before `do_rough_path` /
  `do_single_tracking` are ever called, so there is no fit to display;
- **2.65 cm outside** the FC/TGM box (198.55) → which is why the same event's
  `tagger_check_fc` records **FC = 0** (not contained) for this very cluster.

The verdict does not hinge on the T0 correction. Raw x-max is 200.956 and
corrected x-max 201.198 (t₀ = 1.546 µs ⇒ +0.242 cm); **both** are inside 201.45,
so the known `steiner x_t0cor` raw-coordinate issue is not what is happening
here.

**The merge fragment is not the cause.** Component 10 sits 240 cm away in TPC0
and is fully interior under either box, so it contributes no exit candidate. It
does own 3 of the 6 axis extremes (x-min, y-max, z-max), which means the real
track's deep end at x = 129.93 is never tested — harmless here, but it is the
mechanism by which a grafted fragment can hide a genuine exit from
`get_extreme_wcps()`, and STM has no analogue of the `main_component_pairs` guard
TGM got in [36](36_tgm-main-component-pairs.md).

## 3. The `skip` display artifact

Line 181 of the run log:

```
check_stm_conditions: cluster 21 no STM fit: fully co ms) ../clus/src/MultiAlgBlobClustering.cxx:2082
```

WCT writes long multi-line spdlog messages non-atomically, so `fully contained
(Mid Point A)` was cut after 8 characters and another message's tail spliced on.
`stmfit_code()` substring-matched `'fully contained'`, missed, and — because
`parse_stm_skips` keeps the *first* reason per cluster — did not fall through to
the generic catch-all either. The bundle was reported as `skip`.

This is the third instance of the same class of bug (doc 48 §5: the torn
`STM=1` verdict; the `FC=fal` fragment that crashed evt285999). Fix, in
`nusel_extract.py`:

- `STMFIT_CODES` entries now key on each reason's **leading** text, so a
  truncated reason is still a prefix of one of them;
- `stmfit_code()` tries substring match first (unchanged for intact lines), then
  a longest-common-prefix test with a 6-character minimum — the seven reasons all
  diverge within 6 characters, so it is unambiguous;
- an unidentifiable reason now reads **`torn`**, never `skip`: `skip` reads as a
  distinct tagger pathway and there is no such thing.

Effect on the 30-event tables: **exactly one row changed**, `skip` → `contained`
(evt285185 cluster 21). Every other column of all 328 bundles is byte-identical
(`diff` with field 20 blanked, all three tags).

> **Note — the `dq48v3` tables were regenerated in place.** The derived tables
> (`nusel-table.tsv`, `nusel-events.tsv`, `nusel_evt*/nusel-evt*.tsv`) under
> `work-{mcp10,mcp1000,mcp1000b}-dq48v3/` were rewritten by re-running
> `nusel_extract.py` step 3 only, from the existing on-disk artifacts — no
> wire-cell was re-run and no `evt*/`, `ql_evt*/` or `nusel_evt*/` product was
> touched. Every pre-fix table is preserved alongside as `*.tsv.pre49`. This is
> still a write into an existing scan label, which CLAUDE.md M13 reserves for the
> owner's decision; if a fresh `dq48v4` tag is preferred, the symlink-farm recipe
> in doc 48's repro block makes it a two-minute rebuild.

```
        before                after
   146  contained        147  contained
   118  tgm              118  tgm
    37  eval              37  eval
    24  nosteiner         24  nosteiner
     1  skip               -
     1  nexits             1  nexits
     1  midkink            1  midkink
```

## 4. How big is this? 96 of 147, and it is not a corner case

`python3 scripts/analysis/stm/stm_fv_census.py` over the three `dq48v3` tags (30 events, 328 bundles).
For every bundle whose `stmfit` is `contained`, it measures the 6 axis-extreme
points `get_extreme_wcps()` collects against both boxes:

```
147 "contained" clusters over 3 tag(s); 96 have an extreme point OUTSIDE the FC/TGM box (65%)
   outsideness cm: median 2.88  p90 3.54  max 3.77
   by wall: anode-x=23  y=61  z-up=4  z-down=8
   FC=0 (not contained) on these: 96;  geometry says outside: 96;  agree: 96, FC-only: 0, geometry-only: 0
   extreme point outside the STM box too: 0 (must be 0 -- otherwise the tagger would have found an exit)
```

Reading these four lines:

- **65 % of every "fully contained" STM skip is called an exiter by FC** on the
  same event. `contained` is the single largest `stmfit` category (147 of 328
  bundles, 45 %), so this governs a large fraction of all STM coverage.
- **The two independent measurements agree 96/96.** The geometric test (does an
  extreme point lie outside the inset box?) and the FC tagger's own flag pick out
  the *same* 96 clusters, with zero disagreement either way. That is what proves
  the box is the **only** discriminator: if any of the 96 owed its FC = 0 to the
  `check_signal_processing` or `check_dead_volume` branches instead, the two sets
  would differ.
- **Zero of the 147 has an extreme point outside the STM box** — the tagger is
  behaving exactly as its box specifies, which validates the sensitive-volume
  numbers of §1 against 147 real verdicts.
- **The effect lives entirely in the gap.** Max outsideness 3.77 cm, against a
  gap of 2.90 / 3.65 / 3.85 cm. Nothing is grossly outside — every one of the 96
  is a wall case, which is the population the margins exist to catch.
- **But it is not merely a margin argument.** Re-measuring the same 96 against
  the **un-inset** `sbnd_pr_fv` box (zero margin, i.e. the raw FV bounds
  201.05 / 199.312 / 0.85…500.15): **67 of 96 are still outside it**, including
  **all 23** anode-x cases. Split by the inset-box wall: anode-x 23/23, y 37/61,
  z-down 5/8, z-up 2/4. So for two thirds of them the endpoint leaves the FV
  *itself* and STM still calls the cluster contained, purely because the
  sensitive volume runs past the FV bounds by the 0.40 / 0.65 / 0.85 cm of the
  §1 table. The remaining 29 sit inside the FV box and are genuine margin cases.
  Cluster 21 is in the first group: x = 201.20 > 201.05.
- **y dominates** (61 of 96) because its gap is the widest at 3.65 cm; the anode
  wall, cluster 21's case, is second at 23.

## 4a. FINDING: this chain is not run-to-run reproducible without `setarch -R`

This came out of the A/B and matters more than the fix. The first A/B pass showed
"TGM +3, FC 2 lost / 2 gained" alongside the STM change, which the FV change
cannot cause (TGM and FC configs did not move). Chasing it:

**A second run of the *identical* configuration reproduces the whole apparent
effect.** `dq49off` vs `dq49off2` — same binary, same flags, same input tarballs,
two invocations:

| | TGM | STM | FC | stmfit changed | labels flipped |
|---|---|---|---|---|---|
| noise floor (off vs off, 2 runs) | 119 → **122** | 7 → **14** | 51 → 51 (2 lost, 2 gained) | 13 | 10 |
| signal (off vs on, no setarch) | 119 → 122 | 7 → 44 | 51 → 51 (2/2) | 106 | 40 |

So `TGM +3` and `FC ±2` were **entirely noise**, and STM itself is noisy at
**±7 out of ~44**.

Root cause is upstream of every tagger. On byte-identical input
(`md5 fdd61a40dd13` both runs) the **steiner stage itself** produced different
graphs:

```
run A:  CreateSteinerGraph: create_steiner_tree produced no steiner_graph for main 1  (only)
run B:  ... no steiner_graph for main 1, main 11, assoc 15, assoc 16, assoc 19
        create_steiner_tree: only 1 steiner terminal(s) found (need >=2)
```

That is why clusters flipped `contained → nosteiner`: they had no steiner_pc at
all in one run. It is **ASLR-dependent**, i.e. M4 (pointer-order-dependent code,
address-layout-keyed):

```
two runs under setarch x86_64 -R          -> IDENTICAL tables
the same two runs without it              -> differ (13 stmfit, 10 labels)
```

Entry point for a fix: `clus/src/SteinerGrapher.cxx:38,75` (the terminal-count
early returns) and `clus/src/CreateSteinerGraph.cxx:166`. **Not fixed here** — it
is unrelated to this change and wants its own investigation.

**Consequence for doc 48.** Its §4 headline, "4 of 11 STM tags lost to the new
dQ/dx tables", was measured without `setarch -R` against a **±7** noise floor.
That finding is **not established** and must be re-measured before it is acted
on. This doc's own numbers below are all `setarch -R`.

## 4b. The A/B result

30 events, 328 bundles, both arms under `setarch x86_64 -R`, labels
`work-*-d49soff` / `work-*-d49son`:

| verdict | off | on | gained | lost |
|---|---:|---:|---:|---:|
| **TGM** | 122 | 122 | 0 | 0 |
| **STM** | 14 | **44** | 31 | 1 |
| **FC** | 51 | 51 | 0 | 0 |

TGM and FC are **untouched**, which is the check that the change is confined to
where it was aimed. `stmfit`:

| | off | on |
|---|---:|---:|
| contained | 148 | **50** |
| eval / fitted | 31 | **129** |
| already TGM | 122 | 121 |
| no steiner_pc | 25 | 25 |
| midkink | 0 | 2 |
| nexits / postfit | 1 / 1 | 0 / 1 |

98 clusters left `contained` (95 straight to `eval`) against the census's
prediction of **96** — the census used only the 6 axis extremes and was explicitly
a lower bound, so 98 ≥ 96 is the expected agreement. **All 31 newly-STM clusters
came from `contained`**, exactly the predicted mechanism, with main-component
lengths spanning 14.6 – 440.7 cm.

Label flips: 30 `not-tagged → STM`, 1 `nu-candidate → STM`, 1 `STM → not-tagged`.

**The `nu-candidate → STM` flip is evt285185 main 21** — the bundle this doc
started from. It now fits and is accepted: `persist_stm_fit: cluster 21 pass=0
status=0 kink=216 exit_L=135.4 left_L=10.9 npts=234`. Note this is an in-beam
bundle leaving the neutrino-candidate sample, so the change has direct selection
consequences, not only diagnostic ones.

**The one lost tag is explained by the change, not by noise.** evt284657 main 27,
out-of-beam, 30.8 cm, 2 fragments — `stmfit` stays `eval` in both arms, but the
tighter box makes the *other* end the exit point, so the fit runs in the opposite
direction:

```
off:  pass=0 status=0 kink=49 exit_L=30.3 left_L= 3.8 npts=55   -> STM=1
on:   pass=0 status=3 kink= 6 exit_L= 3.5 left_L=29.4 npts=54   -> STM=0
```

A different exit endpoint is a direct consequence of a different containment
boundary (the same mechanism produced the 2 `contained → midkink` and the
`nexits → eval`). Which direction is *right* for a 30 cm stub is a hand-scan
question, flagged in §8, not tuned here.

## 5. What this does *not* say

- It does **not** say all 96 are missed STMs. `is_fc = false` only lets a cluster
  *reach* the fit; the Bragg/KS tests then decide. The honest claim is that 96
  clusters are denied a fit by a containment definition the rest of the chain
  disagrees with.
- It does **not** say the 31 new tags are all correct. `is_fc = false` only lets a
  cluster *reach* the fit; the Bragg/KS tests then decide, and they decided in
  favour 31 times. Whether those 31 are real stopping muons is the hand scan.
- It does **not** settle doc 48 §4's "4 STM tags lost to the dQ/dx tables" — that
  number is now known to sit at the noise floor (§4a) and must be re-measured
  under `setarch -R` before it means anything.
- Cluster 21's own physics is still only *plausible*: it now fits and the fit is
  accepted (status 0, kink at 216 of 234 points), but a hand scan of the dQ/dx
  panel is what confirms a Bragg rise.

## 6. The fix, as shipped

Mirrors what doc 27 did for FC. C++ default unset ⇒ absent keys ⇒ byte-identical;
SBND turns it **on**.

- **C++** `clus/src/TaggerCheckSTM.cxx`: adds the `NeedFiducial` mixin,
  `m_use_fiducial` / `m_fv_tolerance`, and forwards them —
  `cluster_fc_check(cluster, m_dv, m_use_fiducial ? m_fiducial : nullptr, m_fv_tolerance)`.
  The `m_use_fiducial = !config["fiducial"].isNull()` guard is copied verbatim
  from `TaggerCheckFC.cxx:82`: `NeedFiducial::configure` must NOT run when the key
  is absent, or it looks up the type-only name `DetectorVolumes`, which is not an
  instantiated component in the SBND PR config (it is `dv-apa0-1`) and throws.
  Both keys round-trip in `default_configuration()`.
- **jsonnet** `cfg/pgrapher/common/clus.jsonnet`: `tagger_check_stm` gains
  `fiducial=null, fv_tolerance=[]` under the key-suppression idiom.
- **jsonnet** `cfg/pgrapher/experiment/sbnd/clus.jsonnet`: `stm_consistent_fv`
  threaded through `clus_pr()` and `pr()`, **default `true`**, passing the *same*
  `wc.tn(sbnd_pr_fv)` and `sbnd_pr_fv_margins` objects TGM and FC get — the same
  objects, not copies, so the three verdicts stay coupled if a margin knob moves
  later. `sbnd_pr_fv` is added to `tagger_uses` when STM is in the pipeline.
- **runner** `sbnd_xin/run_nusel_evt.sh` `-stm-fv` / `-no-stm-fv`
  (`SBND_STM_FV`, default 1) and the `stm_consistent_fv` TLA in
  `wct-pr-perevt.jsonnet`.

Why default-on rather than a default-off knob: the owner asked for the behavior,
and every neighbouring SBND knob is a threaded function arg — which also gives the
A/B an off switch instead of a `git stash`.

`interior_fv_tolerance` has no counterpart: `cluster_fc_check` performs only
ENDPOINT containment tests, so TGM's separate interior vector (doc 35) does not
apply.

**Deliberately out of scope.** `TaggerCheckNeutrino.cxx:518` still makes the bare
`cluster_fc_check(*main_cluster, m_dv)` call that fills
`tagger_info.match_isFC`. Left alone: it is a neutrino-tagger BDT feature, so
flipping it is not a pure-STM change and deserves its own decision. Note that
until it moves, "contained" means two things in this pipeline rather than three —
STM/TGM/FC agree, `match_isFC` does not.

## 7. Verification

| claim | evidence |
|---|---|
| STM box was the sensitive-volume union, \|x\| ≤ 201.45 | run log `sensvol: [(4.5 …) --> (2014.5 …)]` mm; code path `contained` → `contained_by` → `sensitive()`; `AnodePlane.cxx:296-299` |
| … and the tagger really used it | 0 of 147 `contained` clusters had an extreme point outside that box |
| FC/TGM box is \|x\| ≤ 198.55 | `sbnd_pr_fv` 201.05 − `tgm_fv_x_margin` 2.5 (`sbnd/clus.jsonnet:483,499`) |
| cluster 21 ends 0.25 cm inside STM / 2.65 cm outside FC | `scripts/analysis/stm/stm_fv_census.py --detail 285185:21` |
| the box is the only discriminator | geometry and FC flag agree 96/96, 0 either-only |
| prototype shares one FV, inset 3 cm | §1a file:line table; `prod-wire-cell-matching-nusel.cxx:348-351` |
| the torn-line fix moves nothing else | field-20-blanked `diff` of all three `dq48v3` tables: 1 row (`skip`→`contained`), no other change |
| **M1 freshness** | `libWireCellClus.so` 11:04:45 > `TaggerCheckSTM.cxx` 10:59:23 |
| **`wcdoctest-clus`** | 41/41 cases, 518/518 assertions, `rc=0` |
| **compiled config, knob OFF** | `cmp` vs pre-change: **byte-identical** (248180 B) |
| **compiled config, knob ON** | adds exactly `fiducial: "BoxFiducial:sbnd_pr_fv"` + `fv_tolerance: [-25,-25,-30,-30,-50,-30]` to the `TaggerCheckSTM` inode — **identical** to what `TaggerCheckTGM` and `TaggerCheckFC` already carry |
| **output, knob OFF** | `work-*-dq49off` reproduces `work-*-dq48v3` exactly, all 3 tags, all 328 bundles |
| **A/B, knob ON** | §4b, labels `work-*-d49soff` / `work-*-d49son`, both `setarch -R` |
| **noise floor measured** | §4a — mandatory context for every number above |

## 8. Open

## 8. Open

1. **The steiner-stage ASLR non-determinism (§4a) is the top item.**
   `SteinerGrapher.cxx:38,75` / `CreateSteinerGraph.cxx:166`. Until it is fixed,
   every A/B on this chain must run under `setarch x86_64 -R`, and no result
   below ~10 STM tags is meaningful.
2. **Re-measure doc 48 §4** under `setarch -R`. Its "4 of 11 STM tags lost to the
   dQ/dx tables" is at the noise floor and is currently unestablished.
3. **Hand-scan the 31 new STM tags** (lengths 14.6 – 440.7 cm) plus the 1 loss
   (evt284657 m27) and the in-beam flip (evt285185 m21, `nu-candidate → STM`).
   The last one leaves the neutrino-candidate sample, so it is the highest-stakes
   single verdict in this change.
4. `TaggerCheckNeutrino`'s `match_isFC` uses the same bare call — separate
   decision, it is a BDT feature (§6).
5. No `main_component_pairs` analogue for STM: a merged fragment owning the axis
   extremes can hide a real exit (§2).
6. The underlying non-atomic multi-line log write in WCT is still unfixed. Three
   consumers have now been hardened against it instead; a `set_pattern`/single-
   write change in `Aux::Logger` would remove the class of bug.
7. uBooNE's own `tagger_check_stm` is not configured anywhere in `cfg/` — no
   detector but SBND instantiates these taggers — so nothing else could be
   brought into line even if we wanted to. Worth remembering if uBooNE is ever
   wired up: it should get `sbnd_pr_fv`'s equivalent from the start.

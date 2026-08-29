# doc pr/126 — π⁰ reconstruction audit (toolkit vs WCP prototype) and the EM charge-scale calibration from the 98+141 hand scans

**Status: round 1 complete (2026-08-29). Audit + measurement only.**

**SCOPE DECLARATION: NO C++ AND NO JSONNET IS CHANGED BY THIS ROUND.**
There is therefore no build, no freshness proof, no A/B gate and no arm re-run
to report — nothing can have moved. Every number below is read off calib dumps
and hand-scan labels that already existed on disk. The EM-scale update in §4 is
a **recommendation for the owner, not a flip**: `kine_shower_fudge_factor` is a
production knob value, moving it is not byte-identical, and CLAUDE.md §5.1 makes
that stop-and-ask.

Owner brief, verbatim: *"audit the pi0 reconstruction … 1. pi0 reconstruction
with vertex 2. pi0 reconstruction without vertex 3. pi0 reconstruction in the
T_tagger … compare with Prototype implementation … from the recent hand scan
results (141 + 98) I have tagged many pi0 events, those are useful for us to
tune the pi0 reconstruction itself … the factor scaling the charge should be
calibrated by the pi0 mass (135 MeV, 134.9768 MeV to be accurate) … this factor
would also apply for the general EM shower energy factor."*

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# sec 3 -- the 50 hand-paired pi0 and their re-runnable manifest
python3 scripts/pr126_pi0_select.py --selftest
python3 scripts/pr126_pi0_select.py --tsv docs/pr/pr126-pi0-events.tsv \
                                    --manifest em_display/pr126-pi0-manifest.tsv

# sec 4 -- the mass distribution and the scale fit (E1), then E2
python3 scripts/pr126_pi0_mass.py --selftest
python3 scripts/pr126_pi0_mass.py --tsv docs/pr/pr126-pi0-mass.tsv
python3 scripts/pr126_pi0_mass.py --e2 --e2-tsv docs/pr/pr126-pi0-pairs.tsv

# sec 4f / sec 5 -- identification census
python3 scripts/pr126_pi0_census.py --tsv docs/pr/pr126-pi0-census.tsv
```

Arms read (all pre-existing, none written): the **probe-armed knob-on** arms
`work-pr124r1-onA141v2-{mcp1k,mcp2k}` and `work-pr124r1-onA98-{mcp1k,mcp2k,ncpi0,nuecc48}`
via `em_display/em114c-124onA141-manifest.tsv` and `em_display/em117-124onA98-manifest.tsv`
(239 events, all with a calib dump), and the scan-time `work-*-prod0825` arms via
`em114c-manifest.tsv` / `em114-manifest.tsv`. Probe-armed rather than the
`flipA*` production arms because pr/120 and pr/124 both measured that scoring
against probes-off sidecars makes cross-run numbers wobble; content on the A
events is equivalent.

Labels read: `em_labels/emscan-0827/` (98-set, **owner**) and
`em_labels/emscan-0828-agent5/` (141-set). Read-only throughout (M13).

---

# 1. The three π⁰ paths, and what they actually are

The brief names three items. Two of them are reconstruction; the third is not.

| # | brief | what it is in code |
|---|---|---|
| 1 | π⁰ **with** vertex | `PatternAlgorithms::id_pi0_with_vertex` — `clus/src/NeutrinoShowerClustering.cxx:4929-5294` |
| 2 | π⁰ **without** vertex | `PatternAlgorithms::id_pi0_without_vertex` — `clus/src/NeutrinoShowerClustering.cxx:5296-5709` |
| 3 | π⁰ in the **T_tagger** | **no reconstruction at all** — the taggers only re-test the pair produced by 1 and 2 |

Item 3 needs stating plainly because it is easy to look for a π⁰ tagger and not
find one. `clus/inc/WireCellClus/NeutrinoTaggerPi0.h` is a **0-byte file** and
`clus/src/NeutrinoTaggerPi0.cxx` does not exist. That is correct: the prototype's
`prototype_base/wire-cell/pid/src/NeutrinoID_pio_tagger.h` is itself a **2-byte
stub**. `clus/docs/porting/neutrino_id_function_map.md:368` left this as an open
question ("may be subsumed by shower clustering") — **it is now closed: there is
nothing to port.** The tagger-side π⁰ code lives in the nue and single-photon
taggers and consumes `map_shower_pio_id` / `map_pio_id_mass` / `pio_kine`:

| tagger-side π⁰ | toolkit | prototype |
|---|---|---|
| `pi0_identification` (fills `pio_*`, the `pio_1`/`pio_2` BDT inputs) | `NeutrinoTaggerNuE.cxx:647-779` | `NeutrinoID_nue_tagger.h:2643-2754` |
| `pi0_identification_sp` (fills `shw_sp_pio_*`) | `NeutrinoTaggerSinglePhoton.cxx:2056-2181` | `NeutrinoID_singlephoton_tagger.h:2955-3070` |
| `single_shower_pio_tagger` (`sig_*`) | `NeutrinoTaggerNuE.cxx` (`stw_1_n_pi0` at `:2564`) | `NeutrinoID_nue_tagger.h:2756+` |
| `mip_quality` π⁰ counters | `NeutrinoTaggerNuE.cxx:1817-1905` | `NeutrinoID_nue_tagger.h:1754-1804` |
| SSM π⁰ momentum + KDAR veto | `NeutrinoTaggerSSM.cxx:1134-1141`, `:1302` | `NeutrinoID_ssm_tagger.h:1707-1715`, `:1872` |
| T_tagger / T_kine branches | `root/src/UbooneTaggerOutputVisitor.cxx:642-653, 735-750, 858-864, 1172-1183` | `wire-cell-prod-nue.cxx:2088-2099, 3067-3081` |

`numu_tagger` and `cosmic_tagger` contain **zero** π⁰ code in both trees.

## 1.1 The one formula everything rides on

Every mass site in both trees is the massless-photon invariant mass with the
**charge-derived** energy:

```
m = sqrt( 4 · E1 · E2 · sin²(θ/2) ),   E = get_kine_charge()   [never kine_best]
```

toolkit `NeutrinoShowerClustering.cxx:5113` (with vertex), `:5535` and `:5581`
(without vertex); prototype `NeutrinoID_shower_clustering.h:802`, `:562`, `:604`.
`kine_charge` for a shower-flagged object is

```
E = Q_weighted / recom / fudge · w_value · 1e-6 MeV
```

(`NeutrinoEnergyReco.cxx:188`, and its byte-for-byte duplicate at `:508` on the
dedup path; prototype `NeutrinoID_energy_reco.h:248` and `:450`). **So m is
exactly linear in the EM charge scale** — the property §4 rests on.

---

# 2. Prototype ↔ toolkit: what is a port bug, what is inherited

Written as a **delta** on the existing reviews, not a re-derivation:
`clus/docs/patternrecognition/shower_clustering_review.md` §11/§12 already
covered parts of items 1 and 2. Its three flagged divergences were re-checked
against **current** line numbers:

| ref | claim in the old review | status today |
|---|---|---|
| B.2 | `id_pi0_without_vertex` uses `continue` where the prototype uses `break` when both showers are short | **FIXED** — `NeutrinoShowerClustering.cxx:5592` is now `break;` with the prototype line cited |
| B.3 | missing `|pdg|==11` in the `get_flag_shower()` equivalent | **FIXED** — `:5464-5470` adds the PID-electron acceptance, mirroring `ProtoSegment::get_flag_shower() = trajectory \|\| topology \|\| get_flag_shower_dQdx()` with `get_flag_shower_dQdx() = (\|pdg\|==11)` (`ProtoSegment.cxx:1305-1312`) |
| B.5 | direction taken from the candidate π⁰ vertex instead of the shower's own start vertex | **FIXED** — `:5145-5153` uses `sv1_pt`/`sv2_pt` with the comment naming the prototype's convention |

The old review's line numbers are stale by ~2500 lines; that file should point
here rather than be re-cut (see §6).

New findings, each classified:

### 2.1 The mass and the stored angle use DIFFERENT directions — **inherited, not a port bug**

pr/114 §6.1 measured that `2√(E₁E₂)·sin(angle/2)` reproduces the stored
`map_pio_id_mass` within 1 % on only 171 of 282 pairs, and misses by >5 % on 78.
That doc located two mechanisms and deliberately picked neither. **The prototype
question is now answered**: WCP has exactly the same split.

* the **mass** uses `local_dirs` — `get_init_dir()` for a shower at the candidate
  vertex, or the vertex→start chord for a 30°-associated disconnected shower
  (toolkit `:5065`/`:5081`, prototype `:776`/`:795`);
* the stored **`pio_kine.angle`** is recomputed from
  `cal_dir_3vector(own start vertex, 15 cm)` when the start is within 3 cm, else
  the vertex→start chord (toolkit `:5145-5153`, prototype `:868-879`).

Two different recipes in the same function, in both trees. This is a **WCP design
defect the toolkit ported faithfully**, not a translation error. It matters here
because it is the mass the calibration is measured on *and* the angle the BDTs
see. Per M15/§5.4 it is surfaced, not silently unified.

### 2.2 `kine_pio_vtx_dis` can be left inconsistent — **prototype bug, silently fixed in the toolkit**

Prototype `:839`: `kine_pio_vtx_dis` is a **class member**, reset to `1000*units::cm`
at the top of every higher-energy pair's block, and only rewritten inside the
candidate-vertex loop. If that loop finds no vertex belonging to either shower
(the `continue` at `:848`), `max_energy` is *not* updated — so the rest of the
`kine_pio_*` block still describes an **earlier** pair while `kine_pio_vtx_dis`
has been clobbered to 1000 cm. The toolkit uses a function-local `best_vtx_dis`
(`:5127`), so `pio_kine.vtx_dis` always belongs to the pair the rest of the block
describes. **The toolkit is right; the prototype can emit a 10 m vertex distance
next to a filled mass.** Reported, not "fixed back".

### 2.3 `cal_corr_factor` is a stub in the toolkit — **port gap, and it bites this calibration**

The prototype's per-hit charge correction (`NeutrinoID_energy_reco.h:255-272`)
divides by 0.7 over hard-coded uBooNE U-plane dead-channel ranges. The toolkit's
`NeutrinoEnergyReco.cxx:14-34` returns **1.0** and says so in its own comment
("*So far this is an empty class that needs to be filled with actual logic*").
The uBooNE channel list is rightly not ported — but the consequence is that
**SBND applies no dead-region charge correction at all**, an event-dependent
*low* bias on exactly the quantity being calibrated. It is not a constant, so it
does not simply renormalize into the fudge factor: it widens the mass
distribution and drags its low tail.

### 2.4 The mass-window ladder is inconsistent between finder and tagger — **inherited**

| stage | test | effective window |
|---|---|---|
| `id_pi0_with_vertex` (`:5190`) | `-25 < m − 135 + 10 < 35` | **(100, 160) MeV**, centred on **125** |
| `id_pi0_without_vertex` (`:5646`, `:5660`) | `\|m − 135 + 10\| < 60` | **(65, 185) MeV**, centred on **125** |
| `pi0_identification` (`NeutrinoTaggerNuE.cxx:688-691`) | `\|m − 135\| < 35` (type 1) / `< 60` (type 2) | (100, 170) / (75, 195), centred on **135** |
| `pi0_identification_sp` (`NeutrinoTaggerSinglePhoton.cxx:2098-2101`) | same | same |
| SSM KDAR veto (`NeutrinoTaggerSSM.cxx:1302`) | `70 < m < 200` | — |

Identical in the prototype (`shower_clustering.h:919/930`, `:672/:686`;
`nue_tagger.h:2683`).

**The two near-clone taggers also return different things — inherited.**
`pi0_identification` returns the predicate `flag_pi0_1 || flag_pi0_2`
(`NeutrinoTaggerNuE.cxx:778`), while `pi0_identification_sp` returns
`pi0_flag_pio` — mere membership of `map_shower_pio_id`, i.e. "this shower is in
*some* π⁰", ignoring every energy, symmetry and mass test the function just
computed (`NeutrinoTaggerSinglePhoton.cxx:2181`). The prototype does exactly the
same (`NeutrinoID_nue_tagger.h:2747` vs `NeutrinoID_singlephoton_tagger.h:3068`),
so the toolkit is a faithful port and the divergence is WCP's. Both still write
`shw_sp_pio_flag = !(flag_pi0_1 || flag_pi0_2)`, so the *branch* records the
predicate while the *return value* does not — a caller reading the return gets a
looser answer than the one stored.

The `+10 MeV` carries the prototype's own comment
*"hack pi0 mass to a slightly lower value"* — i.e. **an ad-hoc, uBooNE-era stand-in
for the energy-scale calibration this doc performs**. A pair can pass a finder
and fail its own tagger's re-test. §4e shows the offset and the scale are coupled
and must move together.

### 2.5 `kine_pio_*` and `pio_id` name different pairs — **inherited, and quantified here for the first time**

Both trees run **two independent selections** per path: the `kine_pio_*` block
takes the pair of maximum `E1+E2` with **no mass window**, while
`map_shower_pio_id` records only what the mass-window winner loop accepted.
Measured on the 239 events of both manifests (§4f):

* `kine_pio_flag` is non-zero on **166 of 239** events (69 %),
* but only **69 of 239** (29 %) have any accepted π⁰ group;
* of the 69, `kine_pio_*` names **a pair no group accepted** on **26** (38 %);
* and it is filled with **no accepted π⁰ at all** on **97** events.

So the SSM π⁰ momentum (`NeutrinoTaggerSSM.cxx:1134-1141`) and the KDAR veto
(`:1302`) are fed a non-π⁰ pair most of the time. This is the quantitative form
of pr/113 §6.4's "`kine_pio_flag` is not a π⁰ selector".

### 2.6 Path 2 mutates the neutrino vertex, and Path 1 suppresses Path 2 — **inherited**

`id_pi0_without_vertex` writes `main_vertex->fit().point = vtx_point; …dQ = 0`
(toolkit `:5675-5676`, comment "hack"; prototype `:696-698`
`main_vertex->set_fit_pt(vtx_point); set_dQ(0)`). Everything downstream sees the
moved vertex. And `:5361` (prototype `:447`) returns outright if any main-vertex
shower is already in `pi0_showers`, so Path 2 is a **fallback, not a parallel
search**. Measured from the finders' own audit counters
(`g_pr33_audit.f3_pi0_with_vertex` / `f3_pi0_without_vertex`, printed by
`TaggerCheckNeutrino` as `pi0=with:N,without:M`, the only place the accepted
group's *type* is visible — `PrDisplayDump` writes `map_pio_id_mass[..].first`
and drops `.second`): over all 239 events **`id_pi0_with_vertex` accepted all 76
groups and `id_pi0_without_vertex` accepted ZERO.** Path 2 is completely dormant
on SBND, so its vertex-mutation risk is currently theoretical — and would become
real the moment its pre-gates were loosened.

(Distinct and not to be confused: `kine_pio_flag == 2` on 2 of 239 events. That
is the *BDT-feature* selection of §2.5, which runs whether or not any pair was
accepted, not an acceptance by Path 2.)

### 2.7 Tie-break and ordering — parity confirmed

Prototype uses strict `>` for the with-vertex max-energy scan (`:838`) and `>=`
for the without-vertex one (`:635`), so ties go to `flag = 2`. The toolkit
matches: `:5121` `if (energy_1+energy_2 <= max_energy) continue;` (strict `>`)
and `:5602` `if (energy_1+energy_2 < max_energy) continue;` (`>=`). The
prototype's `std::map<WCShower*,…>` iteration is **pointer-address ordered**; the
toolkit replaces it with `shower_pair_cmp` / `shower_less` on graph indices
(CLAUDE.md §2 determinism). Combined with 2.4's `break`, the prototype's pairing
is allocation-order dependent and the toolkit's is not — a deliberate,
documented divergence.

### 2.8 Dead or unported, mentioned not fixed (§5 tie-breaker)

* **`map_pio_id_saved_pair`** is threaded through all three toolkit signatures and
  never written or read. Not sloppiness: its only prototype writer is
  `fill_pi0_reco_tree` (`NeutrinoID.cxx:1326`/`:1361`), which the toolkit does not
  port — the role is served by `pf_pi0_node_per_id`
  (`MultiAlgBlobClustering.cxx:2089-2118`, whose comment says exactly this).
* **`ssm_kine_pio_*`** — 12 fields declared at `NeutrinoTaggerInfo.h:600-611` and
  booked as T_tagger branches at `UbooneTaggerOutputVisitor.cxx:642-653` — are
  **never written by any code**. They are always `-999`.
* **The prototype's π⁰ PF-node energy is wrong and the toolkit does not copy it.**
  `fill_pi0_reco_tree` writes `mc_startMomentum[3] = (m + 135 MeV)/GeV`, i.e. it
  treats the reconstructed **invariant mass** as a **kinetic energy** — a perfectly
  reconstructed π⁰ is recorded at 270 MeV. The toolkit's π⁰ node
  (`MultiAlgBlobClustering.cxx:2172-2182`) uses `map_pio_id_mass[...].first`
  directly, with no `+135`.
  **But it labels it `"pi0  <x> MeV"` and calls the local `pi0_ke`** — so the
  number a hand scanner reads on a π⁰ node in Bee is the **invariant mass**, not
  E(π⁰) = E₁+E₂. Harmless arithmetically, actively misleading on the display,
  and directly relevant because that label is what the scans in §3 were read
  against.
* **The prototype has a π⁰-mass calibration app**,
  `wire-cell/pid/apps/cal_pi0_mass.cxx`, and both of its branches (`:28`, `:54`)
  multiply the mass by an extra **`*0.95`** that the reconstruction itself never
  applies. Independent corroboration of §4's direction and size. (Its `:40`
  divides `dir2` by `mom1`; a bug in that standalone tool only.)

---

# 3. The hand-scanned π⁰ sample: what exists, and what was prepared

`scripts/pr126_pi0_select.py` → `docs/pr/pr126-pi0-events.tsv` (50 rows) and
`em_display/pr126-pi0-manifest.tsv` (a `sample run subrun event dump` manifest so
the π⁰ subset can be re-run or re-scanned as its own arm).

|  | 98-set | 141-set |
|---|---|---|
| scan tag | `emscan-0827` (**owner**, interactive) | `emscan-0828-agent5` (model, doc pr/116) |
| labels on disk | 97 | 141 |
| samples | nuecc48 48, ncpi0 19, mcp2k 17, mcp1k 14 | mcp2k 89, mcp1k 52 |
| truth `origin == ncpi0` | 46 | **0** |
| **hand-stored π⁰ pairing** | **26** | **24** |

**50 hand-paired π⁰ in total.** Counted from the labels, never from
`docs/pr/pr116-bulk/buckets-141.tsv`, which was written before the scan finished
and reports `has_pio True` on 18 where 24 are live — the selftest asserts this
discrepancy so a future reader does not silently trust the TSV.

A row is a hand π⁰ when `label["pio"]["gammas"]["1"|"2"]` both carry a positive
energy. That block is far richer than a bucket keyword: it names the two γ
showers, their **marks- and orphan-corrected** energies
(`energy`, vs `energy_as_reconstructed`, `energy_without_marks`,
`energy_marks_delta`, `energy_orphan_delta`), the decay vertex and how it was
obtained (`main_vertex` 41 / `manual` 6 / `backproject` 3), each γ's axis and
start, and **both mass conventions**.

The marks correction is not decorative: **19 of 50** rows carry a nonzero
`energy_marks_delta` and **2 of 50** a nonzero `energy_orphan_delta`.

Four different "π⁰ numbers" exist and the scripts keep them apart (pr/114 §6.2):
the **hand pair** (the scanner's, not mass-windowed), the **reco groups**
(showers sharing `pio_id ≥ 0`, mass-windowed), **`kine_pio_*`** (a separate
max-energy scan, §2.5), and the bucket TSV's advisory `pi0` column.

Cross-run stability: **99 of 100** γ shower ids survive from the scan-time
arm to the current arm unchanged; one (evt 347824 γ₁) is resolved by largest
charge-weighted member overlap with the scanner's target set.

The owner's own π⁰ remarks in the 98-set manifest are carried into the TSV's
`scan_note` column and are the tuning targets of §5:
`37112` *"no pi0???"*, `84229` *"pi0 two gamma merged …"*,
`506114` *"incorrect pi0 pair"*, `269774` *"multiple pi0"*, `235435` *"pi0??"*,
`142421` *"a major shower not tagged … as shower"*.
Note that 37112 and 84229 have **no** hand π⁰ block — the scanner could not pair
them, which is itself the efficiency signal of §4f.

---

# 4. The π⁰ mass distribution and the EM charge scale

`scripts/pr126_pi0_mass.py`. Two estimators, **both fixed before any number was
read** (CLAUDE.md §5.7). Geometry (axis, start, decay vertex) is the scanner's
and is held **identical** across energy hypotheses, so a change of column is a
change of **energy alone**.

Fidelity gate on the `em_geom.py` reuse (`--selftest`): recomputing both mass
conventions from the label's own geometry and energies reproduces the stored
`mass_axis_convention` / `mass_vertex_convention` with **worst relative error
0.00e+00** on 50/46 rows, and `m(k) ≡ k·m(1)` holds exactly.

## 4a. E1 — hand-paired, unbiased, low N

Pre-registered: **median** of the gated hand-π⁰ sample on the **vertex-chord**
convention, bootstrap 68 % CI (4000 resamples, fixed seed). Primary gate =
`origin == ncpi0` (the only guaranteed truth-π⁰ subsample) **and**
`min(E₁,E₂) > 15 MeV` — the code's own tagger threshold
(`NeutrinoTaggerNuE.cxx:695`), not one of ours; it removes exactly 1 of 50
(evt 76346, γ₂ = 5.0 MeV, where m ∝ √(E₁E₂) makes the mass meaningless).

Energy hypothesis `now` = the **current** arm's `kine_charge` for the same shower:

| sample | n | median m(π⁰) | CI68 | k = 134.9768/m̂ | implied `kine_shower_fudge_factor` |
|---|---|---|---|---|---|
| **PRIMARY** ncπ⁰ + min(E)>15 | 19 | **137.7** | [132.8, 146.5] | 0.980 | **0.816** [0.787, 0.868] |
| cross-check: all origins + min(E)>15 | 45 | 137.3 | [132.8, 141.9] | 0.983 | 0.814 |
| cross-check: **not** ncπ⁰ | 26 | 135.7 | [119.5, 145.4] | 0.995 | 0.804 |
| ungated (all hand pairs) | 46 | 137.1 | [131.9, 141.8] | 0.985 | 0.812 |

The truth-composition worry does **not** bite: 137.7 (ncπ⁰) vs 135.7 (not ncπ⁰)
are consistent well inside their CIs, so pooling would have been defensible —
but the ncπ⁰-only number is still quoted as primary, because only it is
guaranteed to contain a real π⁰.

Across the three energy hypotheses (primary gate, vertex convention):

| energy | median | fudge |
|---|---|---|
| `scanhand` — scan-time reco **plus the scanner's marks/orphans** | 139.2 | 0.825 |
| `scanreco` — scan-time reco as it stood | 138.1 | 0.818 |
| `now` — the current arm | **137.7** | **0.816** |

pr/123 + pr/124 moved the answer by only ~1 %: the EM-clustering rounds that cut
Σq_extra by 45 % did **not** change the mass scale materially, which is a useful
independent statement about those rounds.

**Direction quality.** The two conventions disagree by ~2× on a subpopulation
(pr/114 §6.1's unreliable-direction class: 103798 137.8/74.5, 168432 117.0/61.6,
280159 133.9/75.6, 71178 149.8/25.6). An agreement gate was scanned rather than
adopted, and on the current arms the primary median is **stable** against it:

| `|m_axis−m_vtx|/mean <` | 0.10 | 0.15 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|
| n | 9 | 13 | 15 | 16 | 16 |
| median | 137.7 | 137.7 | 138.1 | 137.9 | 137.9 |

The **convention choice itself** is the dominant systematic, and it is not small:

| convention | n | median | fudge |
|---|---|---|---|
| vertex chord (primary) | 19 | 137.7 | 0.816 |
| shower axis | 21 | **142.6** | **0.845** |

## 4b. E2 — untruncated offline enumeration: **measured-dead**

Pre-registered: every EM-shower pair in all 239 events, the code's own
`pio_kine` direction convention, no mass window, peak over a linear-sideband
background. Result: **830 candidate pairs over 168 events, and no π⁰ peak.**
The spectrum rises monotonically toward low mass — pure combinatorics — and the
sideband fit returns a meaningless 112 MeV. Two variants, both **post-hoc and
labelled as such**, fail the same way: events with exactly one candidate pair
give median 94.6 MeV (n = 76), and restricting to the ncπ⁰ sample gives median
57.9 MeV (n = 120 pairs over 15 events).

This is a real finding, not a null: **the scanner's pairing is not replaceable by
a cut.** In a sample that is mostly not-π⁰, an inclusive enumeration measures
background. E2 therefore neither corroborates nor contradicts E1, and the
recommendation rests on E1 alone — with E1's stated uncertainty, not with E2's
absent one.

For the same reason the **reco-accepted** spectrum cannot be fitted either: it is
truncated to (100,160)/(65,185) by construction. Its median over the 239 events
is **129.2 MeV** — which says nothing about the π⁰ mass and everything about the
window's 125 MeV centre. The 26 groups the hand scan **confirms** as real π⁰ have
median **140.0 MeV** in the same sample. That 129.2-vs-140.0 gap is §5's leading
purity signal.

## 4c. The recommendation, and its significance

Recombination stays at the pr/10 §6 physics-derived `kine_shower_recom_factor = 0.58`;
the term that carries the doc-55 fit's **deliberately excluded, degenerate
normalization `C`** is the fudge factor, which has never left its uBooNE 0.80.
That is precisely what a π⁰-mass calibration determines.

> **RECOMMENDATION: do NOT flip this round.**
> The measurement is
> `kine_shower_fudge_factor = 0.816`, CI68 **[0.787, 0.868]**,
> against the 0.80 in force (the value is `0.80 × m̂ / 134.9768`, m̂ = 137.7 MeV,
> k = 0.980).
> **That interval contains 0.80.** Its lower half corresponds to m̂ < 134.9768,
> i.e. to no correction at all or a correction of the opposite sign. If the
> owner chooses to move anyway, **0.82** is the point estimate; but the data as
> they stand do not demand a move.
> Track and proton scales are not touched either way: the π⁰ constrains EM only.

**Significance, stated plainly: the correction is NOT resolved at 68 %.** The
primary CI is [132.8, 146.5] MeV and it **contains 134.9768**; mapped through the
formula that is fudge ∈ [0.787, 0.868] straddling 0.800. On the shower-axis
convention the CI is [138.1, 147.9] and does exclude 135 — but that convention is
the less physical of the two for a converted photon (whose direction is the
decay-vertex→conversion-point chord), and its median differs from the primary by
more than the primary's own error, so it is a systematic, not a second
measurement.

What *is* established is the **direction and a bound**: five independent readings
(scanhand / scanreco / now × two conventions, plus the ungated pool) all land
between 135.7 and 142.6 MeV. The EM scale is **high by 0–6 %, best estimate
~2 %**, and never low. That agrees with the prototype's own `cal_pi0_mass.cxx`
extra `*0.95` (⇒ fudge 0.842), which is an independent and much older reading of
the same sign.

**Verdict: measured, direction confirmed, magnitude not resolved — no flip
recommended.** What would resolve it is more hand-paired π⁰: the truth-ncπ⁰
statistics on disk is only 19 events, and of the 46 truth-ncπ⁰ events in the
98-set only 22 were paired, so **scanning the remaining ncπ⁰ events for π⁰ pairs
is the single highest-value next step** (§5, item 0). A doubling of n would
halve the interval and settle it.

## 4d. Why one arm is enough to scan every trial scale

At fixed pairing `m(k) ≡ k·m(1)` exactly (asserted in the selftest). So no
re-run is needed to evaluate a candidate factor — which is why this round needed
no arms at all.

## 4e. …but in production a scale change **re-selects**, and the +10 MeV offset fights it

The naive scaling holds only at fixed pairing. In production the mass windows
re-select. Measured over the 76 accepted groups on the 239 events, and over the
10 hand π⁰ currently blocked by the with-vertex window:

| k (fudge) | offset | window | true π⁰ kept (of the 26 hand-confirmed) | all groups kept (of 76) | hand π⁰ rescued (of 10) |
|---|---|---|---|---|---|
| 1.000 (0.80) | 10 | (100,160) | 26 | 76 | 0 |
| **0.980 (0.82)** | 10 | (100,160) | **25** | **71** | **1** (285443) |
| 0.946 (0.845) | 10 | (100,160) | 25 | 66 | 2 (56243, 285443) |
| 1.000 (0.80) | 5 | (105,165) | 25 | 66 | 1 |
| 0.980 (0.82) | 5 | (105,165) | 25 | 64 | 2 |
| 1.000 (0.80) | 0 | (110,170) | 23 | 60 | 2 |
| 0.980 (0.82) | 0 | (110,170) | 23 | 59 | 2 |

**Read this as coupling, not as an efficiency table.** Lowering the scale moves
every mass *away* from a window whose centre is 125 MeV, so at k = 0.98 five
currently-accepted groups fall out and only one blocked hand π⁰ comes in — a net
loss of four *pairings*, on a sample where "accepted but unscanned" is not the
same as "correct". The prototype's own comment calls the `+10 MeV` a
*"hack pi0 mass to a slightly lower value"*: it is an ad-hoc compensation for a
reconstruction whose masses sat high, i.e. **a substitute for exactly the
calibration performed here**. Scale and offset are one degree of freedom split
across two constants and should be decided together, with a Bee adjudication of
the groups that change — not by this table alone.

## 4f. π⁰ identification against the hand scan

`scripts/pr126_pi0_census.py`, current arms, the 50 hand π⁰:

| the reconstruction… | n | % |
|---|---|---|
| reproduces the scanner's pair **exactly** | 26 | 52 % |
| pairs one of the two γ with something else (**partial**) | 10 | 20 % |
| has a π⁰ group sharing **neither** γ | 2 | 4 % |
| produces **no π⁰ group at all** | 12 | 24 % |

**52 % exact, 72 % sharing at least one γ.** What stopped the other 24:

| blocker | n |
|---|---|
| a γ is **PID-typed as a track** — pdg 211 ×5, 2212 ×2, 13 ×1 | **8** |
| the hand mass falls **outside the (100,160) with-vertex window** | 10 |
| a γ shorter than the 3 cm cut | 1 |
| the γ shower renamed between arms | 1 |

Which finder accepted the groups that *were* found (audit counters, §2.6):
**76 by `id_pi0_with_vertex`, 0 by `id_pi0_without_vertex`.**

The single largest cause is **not** a π⁰ cut. It is upstream PID: a photon typed
211/2212 is a *track*, never enters `map_vertex_to_shower` as a shower, and can
therefore never be paired. That places the dominant π⁰ inefficiency squarely in
the same "root-is-wrong / score-100 sentinel" territory pr/122 and pr/124 §B
measured dead from the seeder side.

---

# 5. Improvements and tunes — proposals only, ranked

Nothing below is implemented. Each item names the measurement behind it, the
shape of the knob, and what it would cost. **No knob is added or flipped in this
round** (CLAUDE.md §5.1).

**0. Scan more π⁰ before flipping anything.** §4c: the calibration's CI contains
135. 46 of the 98-set are truth-ncπ⁰ and only 22 were paired; the 141-set has
zero ncπ⁰. A targeted π⁰-pairing scan over the unpaired ncπ⁰ events — the
`em_display` viewer already stores exactly the block §3 consumes, and
`em_display/pr126-pi0-manifest.tsv` is the seed — would roughly double n and is
the cheapest path to a resolved number. *Cost: scan time only.*

**1. Fix the γ-typed-as-track PID, not the π⁰ cuts.** §4f: 8 of the 24 misses.
Concretely: 169626, 285567, 506746, 54341, 52044 have a γ at pdg 211; 47212 has
**both** γ at 2212. The π⁰ finder cannot see them at all. This is the highest-
efficiency item and it belongs to the recognition thread, not to π⁰ code.
*Shape: none here — a pointer for the next recognition round.*

**2. Decide the scale and the `+10 MeV` offset together.** §4e. The offset is a
uBooNE-era stand-in for the calibration this doc performs; keeping both
double-counts. A `pi0_mass_offset` knob (C++ default `10*units::MeV`, so absent
⇒ byte-identical) would let the pair (`kine_shower_fudge_factor`,
`pi0_mass_offset`) be scanned jointly, with a Bee adjudication of the groups
that change. *Cost: one default-OFF knob + one A/B gate + one Bee round.*

**3. The window is admitting low-mass pairs.** §4b: all 76 accepted groups have
median mass **129.2 MeV**; the 26 the hand scan confirms have median **140.0**.
The 50 unconfirmed groups sit at the window's 125 MeV centre. They are not
*proven* fakes — most are simply unscanned — but the gap is exactly the shape a
mis-centred window produces. Adjudicating a Bee sample of the low-mass
unconfirmed groups is the measurement that would turn item 2 from plausible into
decided. *Cost: one Bee scan, no code.*

**4. Give SBND a `cal_corr_factor`.** §2.3: the toolkit's is a stub returning
1.0 while the prototype corrects dead regions. SBND currently has **no**
dead-region charge correction, an event-dependent low bias on every EM energy —
π⁰ mass, `kine_reco_Enu` and every kine-derived BDT feature alike. It should be
built from SBND's own bad-channel map, not ported. *Cost: real work; a knob with
the identity map as default is byte-identical when off.*

**5. Stop feeding the SSM tagger a non-π⁰ pair.** §2.5: `kine_pio_*` names a pair
no group accepted on 26 of the 69 π⁰ events and is filled on 97 events with no
π⁰ at all, yet `NeutrinoTaggerSSM.cxx:1134-1141` builds the π⁰ momentum from it
and `:1302` vetoes KDAR on `70 < mass < 200`. A knob choosing the **accepted**
pair (when one exists) over the max-energy scan is a small, well-scoped change
with an obvious default-OFF. *Cost: one knob; changes SSM BDT inputs, so a full
gate + score comparison.*

**6. Unify the mass and angle directions — or document the split.** §2.1: 78 of
282 pairs disagree by >5 %, in both trees. Since it is inherited, the choice is
the owner's (M15): either a knob that computes both from `local_dirs`, or an
explicit line in the porting dictionary saying the split is intentional.

**7. Relabel the Bee π⁰ node.** §2.8: `"pi0 <x> MeV"` displays the **invariant
mass** while the code calls the variable `pi0_ke`. Every hand scan to date has
read that number. Renaming the label to `"pi0 m=<x> MeV"` — or showing
E₁+E₂ alongside — costs nothing and removes a standing misreading.

**8. Fill or delete `ssm_kine_pio_*`.** §2.8: 12 T_tagger branches permanently
at −999. A dumped field no code assigns reads as a physics answer of −999 (the
doc pr/26 rule). Either wire them from `pio_kine` or drop the booking.

**9. Path 2 is dormant on SBND** — it accepted **0 of the 76** groups over 239
events (§2.6, audit counters). Before anyone invests
in `id_pi0_without_vertex`, note that its `map_vertex_segments[main_vertex] > 2`
pre-gate plus Path 1's blanket suppression make it nearly unreachable here. Its
neutrino-vertex mutation (§2.6) is therefore a theoretical risk today, and would
become a real one if the pre-gates were ever loosened.

---

# 6. What is NOT claimed

* No code or config changed, so **no byte-identity claim is made or needed**.
* The 50 hand π⁰ are one scanner's pairings on events selected for EM-shower
  interest, not an unbiased π⁰ sample. A gain measured against them is evidence
  of agreement with the scan, not of a physics improvement (the pr/115 caveat,
  restated).
* §4's k is measured on the vertex-chord convention with the scanner's geometry.
  It does **not** validate the reconstruction's own direction estimates.
* The 76 accepted groups are **not** claimed to be 50 fakes plus 26 real. 24 of
  the 50 hand π⁰ are not among them, and most of the unconfirmed 50 are simply
  unscanned.
* E2 failing is a statement about an inclusive enumeration on this sample, not a
  claim that no offline π⁰ selection can work.

# 7. Files

| file | what |
|---|---|
| `scripts/pr126_pi0_select.py` | the 50-event selection; `--selftest` asserts 26/24 and the label round-trip |
| `scripts/pr126_pi0_mass.py` | E1 + E2, the fidelity gate and the linearity assertion |
| `scripts/pr126_pi0_census.py` | §4f identification census, the §2.5 global counts, and the §2.6 finder-type audit-counter pass |
| `docs/pr/pr126-pi0-events.tsv` | 50 rows: hand pair, energies, both conventions, reco groups, completeness |
| `docs/pr/pr126-pi0-mass.tsv` | per-event masses on all three energy hypotheses |
| `docs/pr/pr126-pi0-census.tsv` | per-event match class and blocker |
| `docs/pr/pr126-pi0-pairs.tsv` | the 830 E2 candidate pairs |
| `em_display/pr126-pi0-manifest.tsv` | re-runnable manifest of the π⁰ subset |

Related: pr/114 (the display and the two mass conventions), pr/115 (the 98-set
categorisation), pr/116 (the 141-set), pr/10 §6 (the recombination transfer that
left the fudge factor uBooNE), doc 55 (the SBND recombination fit whose `C` is
the degenerate constant §4c determines), and
`clus/docs/patternrecognition/shower_clustering_review.md` §11/§12 in the toolkit.

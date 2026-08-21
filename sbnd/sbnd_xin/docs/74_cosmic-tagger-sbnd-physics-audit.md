# Doc 74 — Cosmic tagger (NeutrinoID level) SBND physics audit vs the MicroBooNE prototype

**Date:** 2026-08-20
**Scope:** the *selection-level* cosmic tagger — the port of
`WCPPID::NeutrinoID::cosmic_tagger()` (`prototype_base/pid/src/NeutrinoID_cosmic_tagger.h`,
865 lines) into `clus/src/NeutrinoTaggerCosmic.cxx` (`PatternAlgorithms::cosmic_tagger`,
`:471-1356`), invoked from `TaggerCheckNeutrino.cxx:2379`, filling the ten
`cosmict_*` flag/feature blocks of `TaggerInfo`.
**NOT in scope:** the bundle-level TGM/STM/FC/LM taggers (audited in docs 27, 36→pr/36,
39, 49) — they appear only as the FV infrastructure this tagger reuses.
**Method:** principle-level only. No cosmic sample exists for a rate measurement, so
every claim below is a structural/code-level statement, not a measured efficiency.
**Status: no code changed, no config changed. All verdicts in production are unchanged
by this document.**

## Repro

```
# toolkit @ apply-pointcloud 6bf0aafb, wcp-porting-img @ main 540b13c
# side-by-side sources
less prototype_base/pid/src/NeutrinoID_cosmic_tagger.h            # all 866 lines
less clus/src/NeutrinoTaggerCosmic.cxx                            # all 1357 lines
# FV plumbing
sed -n '79,120p'   clus/src/FiducialUtils.cxx                     # inside_fiducial_volume + tolerance
sed -n '115,140p'  prototype_base/pid/src/ToyFiducial.cxx         # scalar SCB polygon, boundary_dis_cut inset
grep -n 'fiducial=' cfg/pgrapher/experiment/sbnd/clus.jsonnet     # :1780 fiducial=dv ; :2022/:2072/:2122/:2234 sbnd_pr_fv
grep -n 'cosmic_y_top\|sbnd_y_top' cfg/pgrapher/experiment/sbnd/clus.jsonnet   # :34, :1186-1189
grep -n 'neutrino_consistent_fv\|muon_dqdx_curve\|mip_dqdx_median' \
        cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet        # :1086, :714, (:704-709)
# history of the two behavioral divergences found
git -C toolkit log -L 868,872:clus/src/NeutrinoTaggerCosmic.cxx   # 02c23735 25*cm -> 25*MeV
git -C toolkit log -L 1178,1182:clus/src/NeutrinoTaggerCosmic.cxx # ce14d372 top_main since original port
```

Prior art cited, not restated: `clus/docs/tagger/cosmic_tagger_review.md` (2026-04-08
logic-fidelity review), `docs/pr/2_uboone-chain-gap-analysis-and-validation-plan.md`
§2e(iv) (the y-cut gap), `docs/27_fc-tgm-consistent-fv.md` /
`docs/49_stm-containment-fv-inconsistency.md` / `docs/pr/36_taggers-port-audit.md` §F1
(the consistent-FV lineage this audit extends), `clus/docs/tagger/tagger_validation_plan.md`
(why only distributional validation is possible).

## 1. What this tagger is and where its output goes

Ten independent checks on the selected neutrino candidate, OR-ed into
`cosmict_flag` (`NeutrinoTaggerCosmic.cxx:1346-1351`; prototype `:854-857`):

| flag | test (one line) | prototype | toolkit |
|---|---|---|---|
| 1 | main vertex outside FV shrunk 1.5 cm | `:26-39` | `:528-539` |
| 2 | single vertex muon backward/exiting, no other activity | `:43-162` | `:548-668` |
| 3 | same, long-muon (multi-segment) variant | `:164-217` | `:670-722` |
| 4 | muon exits FV pointing >100° from beam, no connected showers | `:222-241` | `:725-737` |
| 5 | same, long-muon variant | `:242-264` | `:739-752` |
| 6 | stopped muon + weak-dir second track exiting FV back-to-back | `:493-513` | `:967-983` |
| 7 | stopped muon + Michel (or back-to-back stub), backward/exiting | `:518-547` | `:986-1004` |
| 8 | long muon + one anti-aligned exiting track, little else | `:549-588` | `:1006-1039` |
| 9 | event dominated by vertical (PCA) clusters reaching detector top | `:594-796` | `:1051-1273` |
| 10 | front-face vertex outside FV with beam-parallel weak-dir track | `:799-835` | `:1284-1332` |

Consumption: `cosmict_flag` and the per-check `cosmict_*` features go into
`TaggerInfo` → the numu XGBoost + `cos_tagger_10` sub-BDT
(`sbnd/clus.jsonnet:2473-2478`, **uBooNE-trained weight files**), the PR display dump
(`PrDisplayDump.cxx:871-882`), and — behind `neutrino_type_bitmask=true`
(pr/36 F7, ON: `wct-pr-perevt.jsonnet:1108`) — `neutrino_type` bit 1
(`NeutrinoTaggerCosmic.cxx:1354`). So each check is simultaneously a *hard verdict*
available downstream and a *BDT input distribution* that must stay comparable to the
uBooNE training sample.

## 2. Detector context: what actually differs

| | MicroBooNE (prototype) | SBND (toolkit port) |
|---|---|---|
| active volume | x 0→256, y −116→117, z 0→1037 cm, single drift (+x) | x −201→+201 (two drifts, CPA at x≈0), y ±200, z 0→501 cm |
| E field | 0.273 kV/cm | 0.5 kV/cm |
| MIP dQ/dx scale (median convention) | 43 000 e/cm | 48 000 e/cm (`clus.jsonnet:993-1000`) |
| beam / vertical / drift axes | +z / +y / +x | +z / +y / ±x |
| FV used by *this* tagger | `WCPPID::ToyFiducial` SCB polygons, `boundary_dis_cut = 3 cm` inset (+1 cm z), space-charge corner cuts up to ~120–150 cm in x (`pid/src/ToyFiducial.cxx:61-137`, ctor args `wire-cell-prod-stm.cxx:417-420`) | `FiducialUtils` → `DetectorVolumes::contained()` = union of per-(apa,face) sensitive boxes, **zero margin**, CPA slab \|x\| < 0.45 cm excluded (`clus.jsonnet:1780` `fiducial=dv`; `TaggerCheckFC.cxx:40-49` describes the same volume) |
| drift/T0 handling | `offset_x` from flash time passed to every FV call | points already T0-corrected by `switch_scope`; no offset (correct by construction, same convention as TGM/FC) |
| SCE handling | boundary distorted instead of points (the SCB *is* the space-charge boundary) | `use_sce=false` in both production realities (`wct-pr-perevt.jsonnet:2727`): points uncorrected, boundary a true box |

The good news first: **the algorithm itself has no live drift-direction or absolute-x
dependence.** `dir_drift` is declared but unused on both sides (prototype `:3`,
toolkit `:485`); every directional cut is against beam (+z) or vertical (+y), and the
one azimuth cut `||φ|−90°| ≤ 50°` is symmetric under x→−x. The two-TPC geometry
therefore enters *only* through the fiducial-volume tests — exactly the owner's
framing, and it is where the findings are.

## 3. Finding G1 (main): this tagger tests a zero-margin FV; the prototype never did

Every `flag_inside`-type call in the prototype is
`fid->inside_fiducial_volume(p, offset_x)` with NULL tolerance — the scalar SCB
polygon **inset by 3 cm on every face (4 cm on z)** (`pid/src/ToyFiducial.cxx:117-137`,
`boundary_dis_cut = 3 cm` at `wire-cell-prod-stm.cxx:419`), with the space-charge
corner cuts on top. An exiting track's last reconstructed point — which sits *at* the
instrumented boundary, not beyond it — was therefore reliably "outside FV".

The toolkit's equivalent is `inside_fv(p)` (`NeutrinoTaggerCosmic.cxx:506-509`) →
`FiducialUtils::inside_fiducial_volume(p)` with empty tolerance →
`m_sd.fiducial->contained(p)` (`FiducialUtils.cxx:82-84`) → **`DetectorVolumes`**
(`sbnd/clus.jsonnet:1780` passes `fiducial=dv` into `clustering_methods`, inherited by
`MakeFiducialUtils` at `:1958` via `fiducial_cfg`, `common/clus.jsonnet:60,2202-2209`).
That volume has **no inset at all**. On SBND a reconstructed endpoint is outside it
only by reconstruction scatter across the exact wall (y/z are wire-bounded; a
correctly-T0'd through-anode exit lands *at* x = ±201.45, not past it).

Note this is *not* fixed by the production knob `neutrino_consistent_fv=true`
(`wct-pr-perevt.jsonnet:1086`): that knob feeds `sbnd_pr_fv` + margins only to the
`match_isFC` recompute (`TaggerCheckNeutrino.cxx:455-481`, used at `:2497`), not to
`cosmic_tagger`, which reads the grouping's FiducialUtils directly.

Structural consequence, per check:

| check | FV dependence | status on SBND |
|---|---|---|
| 1 | own −1.5 cm shifted-point tolerance | alive (margin 1.5 cm vs prototype's 1.5 cm on the un-inset SCB arrays — comparable, minus the SCB shape) |
| 2, 3, 7 | `(!flag_inside && angle_beam>40°) OR (flag_inside && weak-dir/no-Bragg OR angle>60°)` | exit branch structurally weakened; the `flag_inside` branch keeps them alive |
| 4, 5 | **require** `!flag_inside` (`:736`, `:750`) | near-dead: fires only on reconstruction scatter past the exact wall |
| 6 | requires `!inside_fv(other_vtx2)` (`:979`) | near-dead |
| 8 | requires `flag_out` = an anti-aligned segment ending outside FV (`:1026`) | near-dead |
| 9 | no FV test (PCA/topology only) | fully alive |
| 10 | requires `!inside_fv(vtx)` at the front face (`:1309`) | near-dead |

So of the ten checks, **five are structurally near-dead and three lose their
exit-detection branch** on SBND, purely because the FV handed to this tagger has no
margin. This is the same defect class as docs 27 (FC vs TGM), 49 (STM), and pr/36 §F1
(`match_isFC`) — each of which was fixed by handing the component
`BoxFiducial:sbnd_pr_fv` + `fv_tolerance` — but the fix was never extended to
`cosmic_tagger`. Equally important for the BDT path: `cosmict_{2,3,4,5,6,7}_flag_inside`
and `cosmict_10_flag_inside` are XGBoost/sub-BDT inputs evaluated with uBooNE-trained
weights; their SBND meaning ("inside the zero-margin instrumented volume") differs from
the training-sample meaning ("inside the 3 cm-inset SCB"), biasing the features toward
`inside=true` relative to training.

**Recommendation (knob candidate, default OFF):** give `cosmic_tagger` the same
containment definition as the rest of the stage — either (a) a
`cosmic_consistent_fv` knob on `TaggerCheckNeutrino` that routes the already-configured
`m_fiducial`/`m_fv_tolerance` (`neutrino_consistent_fv` values: `sbnd_pr_fv` +
`[-2.5,-2.5,-3,-3,-5,-3]` cm) into the `inside_fv` lambda and the flag-1 test, or
(b) reconfigure `MakeFiducialUtils` itself with `sbnd_pr_fv` (heavier: FiducialUtils
also serves dead-region/SP walks elsewhere — (a) is the surgical option and mirrors
how STM/FC/match_isFC were fixed). NOT bit-identical when ON; needs its own census
round on the standard manifest.

## 4. Finding G2: flag 1's shifted-point test straddles the CPA hole

`flag_cosmic_1` tests the six points `p ± 1.5 cm` along each axis for containment
(`FiducialUtils.cxx:96-119`; toolkit `stm_tol_vec(6, −1.5cm)` at
`NeutrinoTaggerCosmic.cxx:536`). Against `DetectorVolumes::contained()`, whose SBND
volume excludes the CPA slab \|x\| < 0.45 cm (`TaggerCheckFC.cxx:46-48`), this
produces two artifacts the prototype geometry could not have:

- a genuine vertex at \|x\| ∈ (1.05, 1.95) cm — perfectly good physics volume ~1–2 cm
  off the cathode — has one x-shifted copy land inside the hole → `flag_cosmic_1 =
  true`, i.e. two ~0.9 cm-wide **false-cosmic bands, one per TPC**;
- conversely a (mis-fitted) vertex *inside* the hole passes, because the tolerance
  path tests only the six shifted copies, never `p` itself (`FiducialUtils.cxx:96-119`)
  — non-monotonic behavior in \|x\|.

Both artifacts vanish under the G1 recommendation (the single cross-cathode
`sbnd_pr_fv` box has no hole). Until then this is a small but real hard-tag hazard:
`cosmict_flag_1` is a direct component of `cosmict_flag`.

## 5. Finding G3: no SCB analogue, no margin, no SCE correction — assess jointly

uBooNE's FV shape was doing double duty: a 3 cm reconstruction margin *plus* the
space-charge boundary (corner cuts up to ~120–150 cm in x,
`pid/src/ToyFiducial.cxx:61-113`, data/MC-dependent). Dropping the SCB *shape* for
SBND is physically defensible — 0.5 kV/cm field and 2 m drift give distortions of
O(cm), not O(m) — but on SBND the production PR chain also runs with `use_sce=false`
(points uncorrected), and this tagger's FV carries **zero** margin to absorb the
residual few-cm distortions near the cathode and walls. The L1 taggers absorb this
with their 2.5–5 cm `fv_tolerance` margins; `cosmic_tagger` has 1.5 cm on check 1 and
nothing anywhere else. Verdict: the box-instead-of-SCB choice is sound *provided* the
G1 margins are adopted; without them the two omissions compound.

## 6. Quantities correctly re-anchored for SBND (verified sound)

- **The four "reaches the detector top" y-cuts** — uBooNE literals 100/102/80/50 cm
  (C++ defaults, `NeutrinoPatternBase.h:2438-2447`) are 17/15/37/67 cm below uBooNE's
  y=+117 top; SBND production sets `sbnd_y_top − {17,15,37,67}` = 183/185/163/133 cm
  (`clus.jsonnet:34,1186-1189`, knobs ON since 2026-07-30, commit `cbd78820`, doc pr/2
  §2e(iv)). The reasoning — these are *entry-point tolerances* for a downward cosmic
  (reconstruction truncation at the top face), which do **not** scale with detector
  height — is physically correct: scaling them proportionally (e.g. 100·200/117 = 171
  → offset 29 cm) would have no physical basis, while keeping the absolute offsets
  preserves the calibrated "how far below the top a clipped cosmic's highest point
  sits". At the uBooNE values these cuts were near-meaningless on SBND (100 cm is
  mid-detector); this was the single biggest geometry gap and it is closed.
- **`mv_pt.y() > 0`** in flag 9 (`:1196`) — the detector *mid-plane*, scale-free on
  both detectors (uBooNE −116→117, SBND ±200). Correctly left un-knobbed (comment at
  `:1198-1199`).
- **dQ/dx normalization** — every prototype `43e3/units::cm` (features *and* cuts:
  `0.75`, `1.4`, `1.2·front` ratios) became `m_mip_dqdx_median`
  (default 43 000 = byte-identical; SBND 48 000, `clus.jsonnet:993-1000`). The 48/43 =
  1.116 rescale is consistent with the STM-side 56/50 = 1.12 (both anchored to the
  0.5 kV/cm recombination tables, doc 57 method). Keeping the *ratios* dimensionless
  and rescaling only the MIP anchor is the right transfer. It also keeps the
  `cosmict_*_dQ_dx_*` BDT features in "MIP units", which is what makes the
  uBooNE-trained weights transferable in principle.
- **Muon median-dQ/dx envelope** — prototype `0.8866+0.9533·(18cm/L)^0.4234`
  (`NeutrinoID_cosmic_tagger.h:52`) is knob-fed (`m_muon_dqdx_curve`,
  `NeutrinoPatternBase.h:2505-2530`) and SBND production uses the re-derived
  0.5 kV/cm fit `[0.8826, 1.0587, 18, 0.4745]` (`wct-pr-perevt.jsonnet:714`,
  docs/pr/10 §4). Sound.
- **Flag 10 front face** — prototype hard-codes `vtx z < 15 cm`; toolkit computes
  `z_front` = min z over all sensitive volumes and tests `z < z_front + 15 cm`
  (`:1284-1309`). On SBND z_front ≈ 0 so the numbers coincide; the 15 cm is an
  absolute beam-entry tolerance and correctly does not scale with detector length.
  Sound multi-APA generalization.
- **T0/offset_x** — prototype threads `offset_x` (flash drift offset) into every FV
  call; the toolkit runs after `switch_scope` so coordinates are already
  T0-corrected and no offset is needed. Equivalent by construction; same convention
  as TGM/FC (`TaggerCheckFC.cxx:27-29`).
- **Everything angular** (beam 40/60/100/25°, θ ≥ 100°, ||φ|−90°| ≤ 50°,
  back-to-back 155–175°, verticality 20/25/30/35/40°) and **every energy threshold**
  (25/60/70/80/150/200 MeV) and **track-length scale** (0.9–120 cm, the 3 cm
  small-cluster split, 40 cm shower-length caps, 100 cm single-cosmic exemption):
  direction-, physics-, and topology-based — detector-size independent. Correctly
  transferred verbatim. `highest_y` init at −100 cm (`:1124`) is inert given all live
  y-cuts sit ≥ 133 cm.

## 7. Fidelity spot-checks (geometry-adjacent; two records to set straight)

- **F1 — the Michel 25 MeV threshold.** Prototype `:398` compares a shower *energy*
  to `25*units::cm` (= 250 MeV in WCP units) — a units bug. The 2026-04-08 review
  (`cosmic_tagger_review.md` §"NOTE", line 132) recorded the toolkit as *faithfully
  reproducing* it, but commit `02c23735` ("cosmic_tagger: fix units bug…") later
  changed it to `25 * units::MeV` (`NeutrinoTaggerCosmic.cxx:870`). The fix is
  physically right (Michel endpoint 52.8 MeV; 250 MeV made the exemption fire on any
  sub-250 MeV shower) but it is an **unconditional, un-knobbed divergence** from the
  prototype and the review doc is now stale on this point. Record here; suggest a
  one-line update to `cosmic_tagger_review.md` §NOTE when next touched.
- **F2 — flag 9's `top_main` uses the main cluster's own top, not the running max.**
  Prototype `:698` tests `highest_y` — the running maximum over all clusters processed
  *so far* (updated at `:685`, `std::map<int,…>` order) — while the toolkit (since the
  original port `ce14d372`, `:1180`) tests `highest_y_cl`, the main cluster's own high
  point. The toolkit is stricter (running max ≥ own top) and deterministic, and it is
  what the knob comment documents ("main cluster's own top", `clus.jsonnet:1186`), but
  the 2026-04-08 review signed flag 9 off as "all match the prototype" — this
  divergence is undocumented. Per M15 discipline: surfacing it here rather than
  changing either side; recommend recording it in `cosmic_tagger_review.md` as a
  deliberate determinism improvement (the prototype's value depends on cluster-id
  ordering) rather than reverting.
- Already-documented divergences reconfirmed, no action: `pca.center.y()` vs
  `pts.front().y` for the <3 cm-debris test (review line 233, "deliberate
  improvement"); `seg_dir_weak()` vs raw `is_dir_weak()` reads
  (porting_dictionary, `dir_weak_use_score` ON); fail-open `inside_fv → true` when
  FiducialUtils is absent (`:507`) is safe-by-ordering (`fiducialutils` precedes the
  tagger stage in the production pipeline) and fails *conservative* (no cosmic tag).

## 8. Findings summary, ranked

| # | finding | class | physics impact | action |
|---|---|---|---|---|
| G1 | tagger tests zero-margin `DetectorVolumes` union instead of an inset FV; prototype used 3–4 cm-inset SCB | FV mechanics | 5 of 10 checks structurally near-dead, 3 lose exit branch; `*_flag_inside` BDT features biased vs uBooNE training | knob candidate `cosmic_consistent_fv` (route `sbnd_pr_fv`+margins into `inside_fv`), default OFF, own census round |
| G2 | flag-1 ±1.5 cm shifted-point test vs CPA hole → two 0.9 cm false-cosmic bands at \|x\|∈(1.05,1.95) cm | FV mechanics | rare hard mis-tag of near-cathode vertices | fixed for free by G1 |
| G3 | no SCB analogue + `use_sce=false` + zero margin compound near boundaries | physics judgment | acceptable *only* with G1 margins | fold into G1 |
| F1 | 25·cm→25·MeV unit fix (02c23735) un-knobbed; review doc stale | fidelity record | deliberate, physically correct | update review doc note |
| F2 | flag 9 `top_main` = own top vs prototype running max; review signed off as identical | fidelity record | stricter + deterministic; small | document in review doc |

Everything else audited — the two-TPC/drift question, T0 handling, all four y-cut
re-anchors, the dQ/dx anchor and envelope, the front-face generalization, all angular
and energy cuts — is **physically sound as ported and correctly parameterized for
SBND**. The owner's premise holds: the algorithm transfers; the geometry/FV plumbing
is the only place it currently falls short, and it falls short in exactly the pattern
(missing consistent-FV wiring) that docs 27/49/pr36-F1 already fixed for the sibling
taggers.

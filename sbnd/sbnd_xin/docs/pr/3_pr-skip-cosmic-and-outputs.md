# 3 — Neutrino PR: skip-cosmic gate + Bee prototype parity + Magnify-tracking / T_tagger outputs

Status: IMPLEMENTED; validation on evt 18253/1/172230 (nu candidate) + 444187
(TGM-tagged in-beam, skip demo) recorded in §6.  Off-path proven byte-identical
(§5).  Follow-up items in §7.

Companion docs: `pr/1_beam-window-cosmic-vs-nu-division.md` (nueCC48 sample),
`pr/2_uboone-chain-gap-analysis-and-validation-plan.md` (gap analysis + V0-V10
plan; this doc executes the V0 instrumentation prerequisite and part of gap G2).

## 0. Repro block

```bash
# toolkit build (all C++ below):
cd /nfs/data/1/xqian/toolkit-dev/toolkit && ./wcb build --notests -p && ./wcb install --notests -p

# uBooNE off-gate (35 nue events, ASLR off, DL off), label pr3out vs gate3:
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/scripts
./sweep_5384.sh pr3out 6 && ./ab_check.sh pr3out gate3

# compiled-config proofs (HEAD worktree vs working tree, wcsonnet, default TLAs):
#   default job: byte-identical;  PR-on job: new keys present (see §5)

# single-event validation (fresh root, M13):
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-nuecc48-prsmoke2
./run_pr3_evt.sh 172230     # nu candidate: full PR + BDT + tracking + T_tagger
./run_pr3_evt.sh 444187     # TGM-tagged in-beam bundle: nu_skip_cosmic demo
```

## 1. What this round adds

Three pieces, all default-OFF at the C++ layer (uBooNE byte-identical), all ON
for the SBND PR job:

1. **`nu_skip_cosmic`** — `TaggerCheckNeutrino` (beam-gate branch only) skips
   in-window mains already tagged cosmic upstream, so the neutrino PR runs only
   on untagged nu candidates ("only run it if it is not cosmics").
2. **Bee prototype parity** — the MABC Bee dumper gains per-set/per-pf knobs
   that reproduce the prototype's five PR visualizations (track/shower,
   per-particle sub-clustering, 3-D dQ/dx with vertex rows, particle-flow
   mc.json).  wire-cell-bee3 untouched (owner directive) — everything is in
   the dumper.
3. **Magnify-tracking + T_tagger/T_kine for SBND** — a 2-APA fork of the
   uBooNE PR tracking writer (`SbndPrMagnifyTrackingVisitor`), plus SBND
   wiring for the (reused, geometry-free) `UbooneTaggerOutputVisitor` and the
   two uBooNE BDT scorers.

## 2. nu_skip_cosmic (toolkit `clus/`)

`clus/src/TaggerCheckNeutrino.cxx` beam-gate selection loop: after the
in-window test, an in-window main is skipped when

- `flag_TGM` set (TaggerCheckTGM verdict, incl. STM's in-algorithm TGM
  promotions), or
- `flag_STM` set (TaggerCheckSTM verdict), or
- cluster scalar `lm_flag > 0` (Q/L LM tagger: 1 = low-energy, 2 =
  light-mismatch; owner chose to skip both).  Absent scalar (-1) = "not
  evaluated" and never skips.

Design points:

- **Per-main, not per-event**: the filter is a `continue` inside the selection
  loop (mirroring STM's skip of TGM-flagged mains), so a cosmic-tagged longest
  bundle does not veto a clean runner-up bundle.  If every in-window main is
  skipped, the existing "no main cluster selected" exit fires and all PR
  consumers (Bee PR layers, BDTs, tracking/tagger visitors) correctly go dark.
- **FC is not checked** — containment is orthogonal, not a cosmic verdict.
- Knob: `nu_skip_cosmic`, C++ default false; `cfg/pgrapher/common/clus.jsonnet`
  `tagger_check_neutrino(...)` emits the key only when true (key-suppression),
  so the uBooNE qlport job compiles byte-identically.  SBND
  (`cfg/pgrapher/experiment/sbnd/clus.jsonnet` `clus_pr`/`pr` arg
  `nu_skip_cosmic=true`) has it ON by default — zero production impact while
  `tagger_check_neutrino` is not in production `pipeline_names`.
- Ordering requirement: TGM/STM (and Q/L LM upstream) must run before
  `tagger_check_neutrino` for the flags to exist — true for the production
  tail + appended PR stage.

## 3. Bee prototype parity (toolkit `clus/` MABC dumper)

Target = the five prototype panels (screenshot
`docs/pics/Screenshot 2026-07-30 at 8.32.35 AM.png`): (a) selected neutrino
activity, (b) track/shower, (c) particle-level sub-clustering, (d) 3-D dQ/dx
with PID capability, (e) particle flow from the neutrino vertex (Bee "MC"
panel).

Gap analysis vs the prototype (all verified against
`prototype_base/pid/src/NeutrinoID.cxx` + `prototype_base/bee/WCReader.cc` and
the checked-in `prototype_base/bee/data/6/` JSONs):

| panel | prototype source | toolkit before | fix (knob, C++ default off) |
|---|---|---|---|
| (a) selected activity | `fill_*` loop only the selected NeutrinoID | PR graph already covers exactly main + matched-flash companions on SBND (beam bundle) | none needed |
| (b) track/shower | `T_rec` q ∈ {0, 15000} on associated points | `shower_track` layer identical convention | already OK |
| (c) per-particle sub-clustering | `fill_point_info` (NeutrinoID.cxx:2179-2218): every segment `cluster*1000+seg_id`, shower segs collapsed to shower start-seg | non-shower segments collapsed to plain `cluster_id` (all tracks in a cluster = one color) | per-set `particle_ids` on `shower_track`: prototype ids for non-shower segments → color-by-real_cluster_id gives panel (c), q gives panel (b), one layer serves both |
| (d) 3-D dQ/dx | `fill_skeleton_info_magnify` (:1852-2002): fitted points q = dQ·0.1−1000 incl. vertex rows with `real_cluster_id=-1` | `track_fit` layer identical q convention but no vertex rows | per-set `include_vertex_points` on `track_fit`: append PR-graph vertex fits, `real_cluster_id=-1` |
| (e) particle flow | `TMC` tree hack → `WCReader::MCJSON`: TDatabasePDG names, int MeV, `KeepMC` floors 5 MeV (γ/e±) / 10 MeV (n/p/nuclei) | structurally done (`fill_bee_pf_tree`), but "p" not "proton", "%.2f" MeV, no KE floors, forced stdout dump | per-pf `prototype_names` (TDatabasePDG-style names + integer MeV) and `em_ke_min`/`np_ke_min` (KeepMC floors); the forced `[fill_bee_pf_tree]` stdout dump is now opt-in via env `WCT_BEE_PF_PRINT` (log-only change) |

SBND wiring (`sbnd/clus.jsonnet` bee sets): the new keys are emitted with the
key-suppression idiom, present **only when `tagger_check_neutrino` is in
`pipeline_names`** — the default production job's compiled JSON is
byte-identical (§5), and any PR job gets prototype parity automatically.
Values: `particle_ids: true`, `include_vertex_points: true`,
`prototype_names: true`, `em_ke_min: 5 MeV`, `np_ke_min: 10 MeV`.

The uBooNE qlport job dumps the same four layers and is protected by the
gate3 byte-identity gate — its config is untouched and all knobs default off,
so its output cannot move (§5).

## 4. Magnify-tracking + T_tagger/T_kine (toolkit `root/`)

Reuse-vs-fork verdicts (from a full read of the three uBooNE components):

- **`UbooneTaggerOutputVisitor` — reused as-is.**  It is a pure
  TaggerInfo/KineInfo dump (T_tagger ~1000 branches + T_kine) from the unnamed
  TrackFitting slot; zero geometry, zero uBooNE literals.  The only
  "uBooNE-ness" is the branch schema itself, which is exactly what we want for
  prototype-comparable ntuples.  It opens the output ROOT file in UPDATE mode,
  so it must run after the tracking writer.
- **`UbooneNumuBDTScorer` / `UbooneNueBDTScorer` — reused as-is.**
  Geometry-free TaggerInfo consumers; they write `numu_score`/`nue_score` back
  into TaggerInfo (serialized into T_tagger).  Weights = the 35 uBooNE-trained
  XMLs from `wire-cell-data/uboone/weights/` (same list as the qlport job).
  On SBND the scores are **uncalibrated** — availability + relative ranking
  only; retraining stays doc pr/2 gap G1.
- **`UbooneMagnifyTrackingVisitor` — forked** (fork-by-duplication, M10) as
  `root/src/SbndPrMagnifyTrackingVisitor.{h,cxx}`.  The uBooNE writer
  hardcodes single-TPC channel offsets `{0, 2400, 4800}` and a single-APA
  ticks-per-slice pick, so on SBND every APA-1 track would be overlaid on
  APA-0 channels.  The fork keeps the uBooNE tree schema (T_bad_ch, Trun,
  T_proj_data, T_rec_charge with `particle_id`, empty T_proj) and adds:
  - the runtime-derived two-TPC `ChanScheme` (copied from the STM fork
    `SbndMagnifyTrackingVisitor`, doc 40): `global = base[p] + apa*nch[p] + w`;
  - per-point APA/face from `PR::Fit::paf` for T_rec_charge pu/pv/pw/pt
    (per-point ticks-per-slice lookup instead of the single-APA pick);
  - the STM fork's T_bad_ch time clamp (`nticks`, default 3427) for unbounded
    dead regions.
  The existing STM visitor was NOT extended: it reads the `stm_fit` PCs and the
  named "stm" TrackFitting slot (no PR graph) and has live STM-validation
  consumers.

SBND wiring (`sbnd/clus.jsonnet` `cm_by_name`, entries compiled in only when
named in `pipeline_names`): `numu_bdt_scorer`, `nue_bdt_scorer`,
`tracking_visitor` (SbndPrMagnifyTrackingVisitor → `<output_dir>/tracking-pr.root`),
`tagger_output` (UbooneTaggerOutputVisitor → same file, UPDATE).  Pipeline
order: `... tagger_check_neutrino, numu_bdt_scorer, nue_bdt_scorer,
tracking_visitor, tagger_output` (BDTs after the PR stage, nue after numu,
tagger_output last).  `wct-pr-perevt.jsonnet` loads `WireCellRoot` when any of
those names (or `save_stm_fit`) is requested.

## 5. Off-path identity evidence

- **uBooNE runtime gate**: sweep label `qlport/scripts/sweep/pr3out` vs base
  `gate3`, 35 nue events, all rc=0 →
  **`ZIPS: 35/35 content-identical`** (hash_archive member hashes,
  `sweep/pr3out/hashes.txt` vs `sweep/gate3/hashes.txt`).  This is the
  probative channel, and it covers the exact changed code paths: the qlport
  zips contain the same `track_fit`/`shower_track`/`vertices`/`mc` layers the
  MABC edits touch, produced with every new knob off.
  The second ab_check channel (`wire-cell-uboone-tagger-compare` verbose
  logs) reported 3 identical / 32 DIFF vs `m66post2` — **non-probative**:
  (i) the accepted historical arms show the same rate (m66post2 vs m66post
  2/33, m66post vs m66pre 2/33, post_stmfit vs pre_stmfit 3/32, cached
  `ab_check_*.log` files in those label dirs), (ii) the differing lines are
  vector-order permutations plus the known-unstable `shw_sp_lol_1_v_*`
  family whose values already moved between the accepted m66pre/m66post2
  arms, and (iii) an A/A control — the same new binary rerunning event 6570
  (label `pr3aa`) — reproduces a 746-line tagger-log diff against its own
  pr3out run while its Bee zip content hash is bit-identical
  (`834151b5…`).  I.e. the T_tagger vector dump order is run-nondeterministic
  (pre-existing; M4 family), the physics content gate is the zip hashes.
- **uBooNE compiled config**: `wcsonnet qlport/uboone-mabc.jsonnet` with the
  pre-change toolkit cfg (HEAD worktree c0501d7e) vs the working tree:
  byte-identical.
- **SBND compiled config, default job**: `wcsonnet wct-pr-perevt.jsonnet`
  (default `pipeline_names`, standard TLAs) pre vs post: byte-identical.
- **SBND compiled config, PR-on job**: with the 13-stage `pipeline_names`
  the compiled JSON contains `nu_skip_cosmic`, `particle_ids`,
  `include_vertex_points`, `prototype_names`, `em_ke_min`, `np_ke_min`,
  `SbndPrMagnifyTrackingVisitor`, `UbooneTaggerOutputVisitor`,
  `UbooneNumuBDTScorer`, `UbooneNueBDTScorer`, `WireCellRoot`,
  `tracking-pr.root` (each grep-verified).
- Unit tests: `./build/clus/wcdoctest-clus` 565/565 assertions pass (no
  doctest runner exists for `root/`).

## 6. Single-event validation (evt 18253/1/172230 first, per owner)

Work root: `work-nuecc48-prsmoke2/` (fresh tag; inputs read from
`work-nuecc48-nuf/ql_evt<EVT>/pctree-evt<EVT>.tar.gz`, M13-clean).  Runner:
`work-nuecc48-prsmoke2/run_pr3_evt.sh` — the 13-stage pipeline (production
tail + `tagger_check_neutrino, numu_bdt_scorer, nue_bdt_scorer,
tracking_visitor, tagger_output`), `setarch -R`, `dl_weights=''`,
production TLAs.  Both events rc=0.

**nu_skip_cosmic (evt 444187, TGM-tagged in-beam bundle):**

```
TaggerCheckNeutrino: in-window cluster 6 (t0 1.096 us, L 210.6 cm) cosmic-tagged
    (TGM=true STM=false lm_flag=0); skipping (nu_skip_cosmic)
TaggerCheckNeutrino: selected main cluster 19 (t0 1.573 us, L 170.5 cm, 6 associated)
```

Cluster 6 is exactly the main the pre-knob smoke (doc pr/2 §5.3) had selected;
with the knob it is skipped and the *untagged* in-window runner-up (cluster
19) gets the PR — demonstrating the per-main (not per-event) semantics: a
cosmic verdict on one bundle does not veto another candidate bundle.

**Bee layers (mabc-pr.zip):**

| check | evt 172230 (nu cand.) | evt 444187 (cluster 19) |
|---|---|---|
| `shower_track` per-particle ids | all-shower event (0 track pts) — 18 distinct shower ids | 1339 track pts now carry ids {19001, 19041} (old code: all collapsed to plain `19`); 3 shower ids |
| `track_fit` vertex rows | 87 rows `real_cluster_id=-1`, q ∈ [0, 20143] | present |
| `vertices` | 87, exactly 1 with q=15000 (main) | present |
| `mc.json` prototype format | `e-  1686 MeV` → {`gamma 5 MeV`→`e- 5 MeV`, `gamma 7 MeV`→`e- 7 MeV`}, int-MeV text, KeepMC floors active | roots: `e-  740 MeV`, `proton  579 MeV` (TDatabasePDG "proton", not "p"), `pi+  1 MeV` (π not floored — correct KeepMC scope) |

**Bee sets (uploaded 2026-07-30, owner-approved):**

- evt 172230: <https://www.phy.bnl.gov/twister/bee/set/0bfe4744-30bf-44cd-945d-b8e13470aaa7/event/list/>
- evt 444187: <https://www.phy.bnl.gov/twister/bee/set/f781b49a-e3bc-4f7e-99bf-1a444e6c1feb/event/list/>
  (cluster 6 visible in `clustering` only — no PR layers = the skip; cluster
  19 carries the PR layers and the MC-panel particle flow)

**tracking-pr.root (SbndPrMagnifyTrackingVisitor + UbooneTaggerOutputVisitor):**

```
trees: T_bad_ch Trun T_proj_data T_rec_charge T_proj T_tagger T_kine
evt172230: T_rec_charge 582 rows, pu [853,1205]  pw [7999,8497]   (APA-0 U/W ranges)
           particle_id ∈ {-1, 11, 13, 2212}; 57 distinct real_cluster_id
           T_tagger: numu_score=-1.992 nue_score=4.301 nu=(-46.1,-84.2,22.9) cm
           T_kine:  kine_reco_Enu=1765.9 MeV
evt444187: T_rec_charge 515 rows, pu [2798,2970] pw [9938,10537]  (APA-1 U/W ranges)
           particle_id ∈ {-1, 11, 13, 211, 2212}
           T_tagger: numu_score=0.277 nue_score=4.301 nu=(39.6,-12.7,166.4) cm
           T_kine:  kine_reco_Enu=1470.1 MeV
```

The two-TPC channel scheme is doing its job: SBND per-plane bases are
U=[0), V=[3968), W=[7936) with per-APA stride (U/V 1984, W 1664), so evt
172230's points sit in the APA-0 bands and evt 444187's in the APA-1 bands
(2798 ∈ [1984, 3968) U-APA1; 9938 ∈ [9600, 11264) W-APA1) — with the uBooNE
writer both events would have landed on [0,1984)/[4800,6464)-style APA-0
channels.  `nu_x` sign agrees (−46.1 cm ↔ APA-0 / +39.6 cm ↔ APA-1).
Both events' `nue_score` saturating at 4.301 is the XGB output clipping in
log-odds — a reminder the scores are uBooNE-trained and uncalibrated here
(record, don't cut; §7).

## 6b. Magnify round-trip (CLOSED 2026-07-30)

`wire-cell-sbnd-magnify-tracking-convert` works on the PR file **unmodified**
(every STM extra is `GetBranch`-guarded, `T_stm_pass`/`T_stm_eval` cloned only
when present):

```bash
cd work-nuecc48-prsmoke2/nupr_evt172230
wire-cell-sbnd-magnify-tracking-convert -btracking-pr.root -otrack_com_pr_evt172230.root -f2
#  -> 32 track block(s), 582 fitted point(s)      (evt 444187: 7 blocks, 515 points)

# view (needs X / ssh -X):
/nfs/data/1/xqian/toolkit-dev/Magnify-tracking-SBND/magnify.sh \
    $PWD/track_com_pr_evt172230.root
```

`Magnify-tracking-SBND` itself needs **no modifications**: its channel axes
(U [0,3968), V [3968,7936), W [7936,11276)) are exactly the ChanScheme the PR
writer emits, its `T_rec` reader takes `reduced_chi2` / `sub_cluster_id` /
`true_*` / stm branches conditionally (`Data.cc:216-242`), and the block-id
convention (new track when `ndf` changes; PR writer: ndf = cluster_id) is the
same one the uBooNE chain used.  Converted files:
`work-nuecc48-prsmoke2/nupr_evt{172230,444187}/track_com_pr_evt*.root`.

## 7. Open items
- **SUPERSEDED for the vertex**: every §6 number here comes from the GEOMETRIC
  vertex (`dl_weights=''`).  On 2026-07-30 the owner adopted the SCN (DL) vertex
  as the SBND default — for evt 172230 it moves the vertex 9.73 cm off the end
  of a proton track onto the true interaction point.  See `pr/4_dl-vertex-default.md`.
  `run_pr3_evt.sh` still pins `dl_weights=''` so this doc's numbers stay
  reproducible as the pre-adoption arm.
- Expand from evt 172230 to the 45 nu-candidate nueCC events + Track B
  (doc pr/2 §3), now that the outputs exist — **both vertex arms** (pr/4 §5).
- BDT scores are uBooNE-trained (uncalibrated on SBND): record distributions,
  do not cut on them (doc pr/2 gap G1).
- `real_cluster_id` in `track_fit` still uses `get_graph_index()` while
  `shower_track`/pf-tree use `seg->id()` — harmless display inconsistency,
  noted for a future parity pass (kept to limit this round's surface).

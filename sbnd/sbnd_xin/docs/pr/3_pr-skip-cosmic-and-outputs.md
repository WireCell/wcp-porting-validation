# 3 — Neutrino PR: skip-cosmic gate + Bee prototype parity + Magnify-tracking / T_tagger outputs

Status: IMPLEMENTED; validation on evt 18253/1/172230 (nu candidate) + 444187
(TGM-tagged in-beam, skip demo) recorded in §6.  Off-path proven byte-identical
(§5).  Follow-up items in §7.  **§8 (2026-08-01) extends the gate from per-main
to per-bundle (`nu_skip_cosmic_bundle`, SBND default ON, NOT bit-identical) —
read §8.5 before treating it as settled.**  §9 (same day) stops the Bee PR layers from dumping the whole clustering when no PR runs.

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

## 8. nu_skip_cosmic_bundle — the cosmic verdict vetoes the whole bundle (2026-08-01)

Status: IMPLEMENTED, **DEFAULT ON for SBND**, C++ default false.  Knob-off path
gated byte-identical; knob-on is **NOT bit-identical** (that is the point) —
4/31 mcp30 events and 3/48 nueCC48 events change, all listed below.  Two of
those, evts 271851 and 10550, are the cases the owner should look at before
this becomes the permanent operating point (§8.5).

### 8.0 Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
./wcb build --notests -p && ./wcb install --notests -p   # -> libWireCellClus.so md5 d00bb418
./build/clus/wcdoctest-clus                               # 49 cases / 565 assertions

# compiled-config proofs
qlport/scripts/compile_ub_cfg.sh    $PWD/cfg /home/xqian/tmp/ub_new.json   # uBooNE: knob off
sbnd_xin/scripts/cfg/compile_prjob_cfg.sh       $PWD/cfg /home/xqian/tmp/pr_new.json   # SBND:  knob on
grep -c nu_skip_cosmic_bundle /home/xqian/tmp/pr_new.json                  # 1

# knob-OFF arm: same driver against a cfg tree at HEAD (the knob absent)
cp -a cfg /home/xqian/tmp/cfg_head
git show HEAD:cfg/pgrapher/common/clus.jsonnet            > /home/xqian/tmp/cfg_head/pgrapher/common/clus.jsonnet
git show HEAD:cfg/pgrapher/experiment/sbnd/clus.jsonnet   > /home/xqian/tmp/cfg_head/pgrapher/experiment/sbnd/clus.jsonnet
cd sbnd_xin
sed 's|^export WIRECELL_PATH=$TK/cfg:|export WIRECELL_PATH=/home/xqian/tmp/cfg_head:|' \
    run_pr_chain_batch.sh > tmp_run_pr_chain_cfghead.sh && chmod +x tmp_run_pr_chain_cfghead.sh

AB31="166870 279256 281808 282292 282541 283040 285366 285443 286177 288468 292450 292643 \
293127 347085 350121 351507 389588 391172 393410 395610 399382 399910 400054 400856 404684 \
409282 493485 58169 65788 72828 52195"
PR_JOBS=6 ./scripts/runners/tmp_run_pr_chain_cfghead.sh work-mcp1kall-d59k work-nscbase-ab31 data $AB31  # old binary
PR_JOBS=6 ./scripts/runners/tmp_run_pr_chain_cfghead.sh work-mcp1kall-d59k work-nscoff-ab31  data $AB31  # new binary, knob off
PR_JOBS=6 ./run_pr_chain_batch.sh       work-mcp1kall-d59k work-nscon-ab31   data $AB31  # new binary, knob on
PR_JOBS=6 ./scripts/runners/tmp_run_pr_chain_cfghead.sh work-nuecc48-nuf   work-nscoff-nuecc48 data
PR_JOBS=6 ./run_pr_chain_batch.sh       work-nuecc48-nuf   work-nscon-nuecc48  data
# comparison = hash_archive.py over mabc-pr.zip + pctree tar.gz, plus T_tagger/T_kine rows
```

### 8.1 Why

`nu_skip_cosmic` (§2) is **per-main**, and a flash bundle can hold more than one
main.  SBND run 18255 evt 52195, bundle gid 6 (t0 +0.566 µs) — read from
`work-mcp1kall-cbron1k/nusel_evt52195/pctree-pr-evt52195.tar.gz`:

| ident | flags | npts | length | what it is |
|---|---|---|---|---|
| 13 | `main_cluster` | 4592 | 513.3 cm | cathode-crossing cosmic → **TGM=true** |
| 23 | `associated_cluster` | 8123 | 399.7 cm | vertical through-going muon, **never evaluated** |
| 5 | `main_cluster` | 5 | 1.3 cm | speck, **out of scope** so never evaluated |
| 44–50 | `associated_cluster` | 9–36 | 1–4 cm | specks |

The taggers iterate main-flagged, in-scope clusters only
(`TaggerCheckTGM.cxx:243`, `require_in_scope`), so cluster 23 (associated) and
cluster 5 (out of scope) carry no verdict at all.  `nu_skip_cosmic` skipped 13,
then selected **5** — a 5-point shard — as the ν candidate, and its companion
set (`matched_flash_gid == 6`) pulled in cluster 23.  Result: a full PR + track
fit (2.58 s) whose Bee `track_fit-global` layer is 821 points tracing the 400 cm
cosmic, on a bundle whose main was already TGM.

Two structural asymmetries make this reachable and both survive this round:
`TaggerCheckNeutrino`'s selection loop has **no scope filter** while the taggers
do, and associated clusters are **never** tagged.  Also note
`normalize_cluster_flags` stamps `flag_TGM = 0` on unevaluated clusters, so
downstream (and Bee) cannot distinguish "evaluated, clean" from "never looked
at".

### 8.2 What the knob does

`clus/src/TaggerCheckNeutrino.cxx`, beam-gate selection block: before selecting,
collect the `matched_flash_gid` of every in-window main carrying a cosmic
verdict (`flag_TGM || flag_STM || lm_flag > 0`) into a `std::set<int>`; any
in-window main whose gid is in that set is skipped with an INFO line:

```
TaggerCheckNeutrino: in-window cluster 5 (t0 0.566 us, L 1.7 cm) shares flash
bundle gid 6 with a cosmic-tagged main; skipping (nu_skip_cosmic_bundle)
TaggerCheckNeutrino: no main cluster selected (16 mains, 2 in-window); skipping.
```

- Bundle identity is `matched_flash_gid` — the same definition the companion
  loop 20 lines below already uses.  `gid < 0` never joins a veto set.
- The set is empty unless `nu_skip_cosmic && nu_skip_cosmic_bundle`, so the
  knob-off path cannot reach the new branch.
- `int`-keyed, so no pointer-order iteration (CLAUDE.md determinism rule).
- Knob `nu_skip_cosmic_bundle`, C++ default false, key-suppressed in
  `cfg/pgrapher/common/clus.jsonnet`; SBND `clus_pr`/`pr` set it true.

### 8.3 Gates

Binaries: baseline `e72494c4` (HEAD 6a5450a0), fix `d00bb418`.  Both rebuilt
from a clean stash/pop cycle and reproduced their md5 exactly.  `wcdoctest-clus`
49/49 cases, 565/565 assertions on the fix binary.  Freshness proof: lib 13:26 >
last source edit 13:24.

**Compiled config.** uBooNE PR job (`compile_ub_cfg.sh`) byte-identical to HEAD
(`cmp` PASS).  SBND PR job: exactly one key added, `nu_skip_cosmic_bundle: true`,
nothing else changed (JSON diff over the whole job).

**Knob-off runtime identity** — `work-nscbase-ab31` (old binary, HEAD cfg) vs
`work-nscoff-ab31` (fix binary, HEAD cfg), 31 events:

```
mabc-pr.zip identical 31/31 | pctree identical 31/31 | T_tagger rows identical 31/31
```

**Knob-on effect** — `work-nscoff-ab31` vs `work-nscon-ab31` (mcp30 + 52195):

| | events | changed |
|---|---|---|
| mabc-pr.zip | 31 | 4 |
| pctree | 31 | 2 |
| T_tagger rows | 31 | 4 |

Changed: **52195** (PR now skipped entirely — the target case), **72828**,
**288468** (PR skipped), **409282** (candidate lost, sentinel row).  PR ran on
18/31 events before, 15/31 after.  All four had `cosmic_flag = 1` before.

**nueCC48** — `work-nscoff-nuecc48` vs `work-nscon-nuecc48`, 48 events: mabc
46/48, pctree 46/48, T_tagger 45/48.  Changed: **271851** (PR skipped),
**10550** (bundle vetoed, a different in-window bundle's 1.7 cm main selected
instead), and **422851**, which is *not* an effect of this knob: its
`mabc-pr.zip` and pctree are byte-identical across arms and only
`kine_reco_Enu` moves 1455.9769 → 1455.9768, the documented ±0.0001 MeV FP
flicker on this event (`clus/docs/audits/pr11-latent-pattern-audit.md`).

### 8.4 Cost

31-event mcp30 arm: 266 s → 256 s total wall (8.6 → 8.3 s/event).  The saving is
small because the vetoed bundles were mostly cheap; evt 52195 alone drops 2.58 s
of `TaggerCheckNeutrino`.

### 8.5 What the owner should look at

The rule does what it says, and on evt 52195 it is unambiguously right.  On two
nueCC48 events it removes a candidate that was **not** obviously junk:

- **evt 271851**, bundle gid 2 (t0 0.764 µs): main 24 = 253.9 cm, TGM-tagged;
  main 7 = 13.0 cm **with 26 associated clusters** — that is the shape of a
  contained interaction sharing a flash with a through-going cosmic.  Before:
  PR ran on cluster 7.  After: no PR at all.
- **evt 10550**, bundle gid 1000002 (t0 1.193 µs): main 11 = 380.0 cm TGM;
  main 7 = 18.5 cm with 28 associated.  After the veto the selection falls
  through to a *different* in-beam bundle's 1.7 cm main (t0 1.209 µs, 0
  associated) — PR runs, but on a shard.

Both are already on record: `pr/1_beam-window-cosmic-vs-nu-division.md` §2.2
lists exactly these two (plus 18259/116962) as the nueCC48 events whose **sole
in-beam bundle is TGM-tagged**, and quotes the resulting loss band as
**6–13 % (3/48 floor)**.  So this knob does not introduce a new loss so much as
make the code agree with pr/1's own accounting: that census counts **bundles**,
while the per-main rule was still selecting a *main inside* a bundle pr/1 had
already written off.  (18259/116962 does not move here — its bundle holds a
single main, so per-main and per-bundle coincide.)

Both are the price of the rule: a beam-coincident cosmic and a real neutrino can
share one flash bundle.  If that trade is not wanted, the natural refinements
(none implemented) are to veto only when the surviving main is below a
size/length threshold or has no associated clusters, or to fix the two
asymmetries in §8.1 instead (give `TaggerCheckNeutrino` the taggers' scope
filter, and/or evaluate associated clusters).

**Resolved (2026-08-01, doc pr/16)**: the size-threshold refinement shipped as
`nu_skip_cosmic_bundle_min_length` (owner-chosen 15 cm, SBND ON) after the
pr/16 interaction study.  10550's 18.5 cm main is restored (row identical to
the pre-veto run); 271851's 13.0 cm main stays vetoed (below the owner's
threshold); 52195 unchanged.  Evaluating associated clusters was considered
and dropped (taggers are designed for mains).

### 8.6 Pre-existing, not fixed here

When `TaggerCheckNeutrino` takes the "no main cluster selected" exit, the Bee PR
layers are **not** empty: `track_fit-global`, `shower_track-global` and
`vertices-global` are each written with ~the full clustering point set (evt
52195 knob-on: 30127 points each vs 30102 in `clustering-global`).  This is not
new — 12/31 events in the *baseline* arm already skip PR and show
`track_fit ≈ clustering` file sizes — but this knob makes the skip path more
common, so the artefact will be more visible in Bee.  Reported, not fixed
(CLAUDE.md: no unrelated fixes in the same change).

### 8.7 Labels

`work-nscbase-ab31`, `work-nscoff-ab31`, `work-nscon-ab31` (mcp30 + 52195, data,
ql root `work-mcp1kall-d59k`); `work-nscoff-nuecc48`, `work-nscon-nuecc48` (ql
root `work-nuecc48-nuf`, data); `work-nscrep-422851` (single-event repeat).

## 9. The Bee PR layers go dark when no PR runs (2026-08-01)

Status: FIXED (`require_pr_graph`, SBND PR layers ON).  Display-only: pctree and
`T_tagger` are untouched, 31/31 in the gate below.  Follow-up to §8.6, which
reported this artefact but did not fix it.

### 9.0 Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
./wcb build --notests -p && ./wcb install --notests -p      # -> libWireCellClus.so md5 55e5e621
./build/clus/wcdoctest-clus                                  # 49 / 565
qlport/scripts/compile_ub_cfg.sh $PWD/cfg /home/xqian/tmp/ub_new2.json   # byte-identical to HEAD
sbnd_xin/scripts/cfg/compile_prjob_cfg.sh    $PWD/cfg /home/xqian/tmp/pr_new2.json   # 3 sets gain the key
cd sbnd_xin && PR_JOBS=6 ./run_pr_chain_batch.sh work-mcp1kall-d59k work-rpg-ab31 data $AB31
cd ../qlport/scripts && ./run_one.sh 0 rpgcheck               # uBooNE fallback still wanted
```

### 9.1 The artefact

`clus/src/MultiAlgBlobClustering.cxx:2466`, the per-visitor Bee dump:

```cpp
auto pr_graph = gs[0]->get_pr_graph();
if (pr_graph) { ... fill from the PR graph ... }
else          { fill_bee_points(config.name, *gs[0]); }   // generic cluster dump
```

`TaggerCheckNeutrino` returns at "no main cluster selected" **before** building a
graph, so `get_pr_graph()` is null and every set bound to that visitor fell into
the generic dump.  Consequences on SBND evt 18255/52195 with §8's bundle veto on:

| set | points | x range |
|---|---:|---|
| `clustering-global` | 30102 | −201.4 … 201.2 cm |
| `track_fit-global` | 30127 | −234.1 … 226.9 cm |
| `shower_track-global` | 30127 | byte-identical to `track_fit` |
| `vertices-global` | 30127 | byte-identical to `track_fit` |

Three failures at once: the sets hold cluster charge instead of fit output; they
hold it in **raw coordinates** (those sets declare `coords: ['x','y','z']`,
correct for PR fit points which are already T0-corrected, while `clustering`
declares `common_corr_coords`) so x runs past the ±200 cm drift volume; and they
ignore the scope filter, adding the 4 out-of-scope clusters (ids 2,3,5,8 = 25
points).  Pre-existing: baseline evt 281808 (PR already skipped before §8) shows
the same signature — `clustering` 32142 pts / 31 clusters / x ∈ [−200.6, 201.4]
vs `track_fit` 32419 pts / 36 clusters / x ∈ [−234.1, 233.8].

### 9.2 The fix

New `BeePointsConfig` field `require_pr_graph` (C++ default false).  When set and
the bound visitor produced no PR graph, the set is left empty and a DEBUG line is
logged instead of falling back.  **The default must stay false**: uBooNE's
`regular` and `steiner` sets (`qlport/uboone-mabc.jsonnet:1311,1321`) are bound
to `CreateSteinerGraph`, which never makes a PR graph, and *rely* on that
fallback to dump their cluster point clouds.

SBND sets the key on `track_fit`, `shower_track` and `vertices` only, guarded by
`std.member(pipeline_names, 'tagger_check_neutrino')` like the neighbouring
`include_vertex_points` / `particle_ids` keys, so a non-PR SBND job compiles
byte-identically.

Empty means **the member is absent from the zip**, not present-and-empty: evt
52195 now ships `clustering-global` + the two dead-area sets and nothing else.

### 9.3 Gates

Binaries `d00bb418` (§8) → `55e5e621`.  `wcdoctest-clus` 49 cases / 565
assertions.  Compiled config: uBooNE PR job `cmp`-identical to HEAD; SBND PR job
gains `require_pr_graph: true` on exactly the three PR sets, `clustering`
untouched.

`work-nscon-ab31` vs `work-rpg-ab31`, same 31 events, same cfg apart from the new
key:

```
mabc-pr.zip identical 15/31 | pctree identical 31/31 | T_tagger rows identical 31/31
```

The 15 are exactly the events where PR ran; the 16 that differ are exactly the
events logging "no main cluster selected", and the zip diff on each is exactly
three members removed (`0-track_fit-global.json`, `0-shower_track-global.json`,
`0-vertices-global.json`) with nothing added or altered.

uBooNE regression check (`qlport/scripts/sweep/rpgcheck`, evt 5384/22/6501,
rc=0): `regular-global` (2.02 MB) and `steiner-global` (7.7 kB) are still
present and populated — the fallback survives where it is wanted.

### 9.4 Labels

`work-rpg-ab31` (SBND, mcp30 + 52195, ql root `work-mcp1kall-d59k`, data);
`qlport/scripts/sweep/rpgcheck` (uBooNE single event).

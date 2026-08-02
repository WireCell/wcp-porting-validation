# doc pr/23 — PR-stage overclustering protection (`protect_bundle`): uboone's second graph examination, ported and made cathode-gap safe

**Status: IN PROGRESS.** Implementation landed (toolkit + runners, default OFF
everywhere); validation campaign V1-V5 and the production flip V6 pending.
**NOT bit-identical when enabled — that is the point of the stage** (doc
pr/22 §8 diagnosed its absence as the residual gap-jumping cause; the owner
requested the port 2026-08-02, accepting the result change).

## 0. Repro

```bash
# toolkit @ <commit set below>, wcp-porting-img @ <commit set below>
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild   # build+install

# compiled-config proofs (both PASS, see §2.3):
cd sbnd_xin
./compile_prjob_cfg.sh /nfs/data/1/xqian/toolkit-dev/toolkit/cfg /home/xqian/tmp/new.json
# vs the same at HEAD^ cfg => byte-identical (stage absent);
# add 'protect_bundle' after unmerge_assoc in the pipeline_names TLA and the
# compiled JSON gains one ClusteringProtectBundle:pr node (knob-on proof).

# enable in any runner (validation phase; production default still OFF):
SBND_PROTECT_BUNDLE=1 ./run_pr_chain_batch.sh <ql_root> <out_root> data <evt...>
./run_pr_evt.sh data -nu -protect <idx>
./run_nusel_evt.sh data -protect <idx>
# knob overrides: SBND_PROTECT_GRAPH=relaxed|relaxed_pid,
# SBND_PROTECT_REJOIN_XCUT/_DYZ/_DIS in cm (0 disables the cathode re-join).
```

## 1. What and why

Doc pr/22 §8: after the runner fix, 29.7 of 33.3 cm of evt 386948's residual
in-void `track_fit` trail is MST bridges between DISTINCT charge fragments
sharing cluster id 16 — the photons `Clustering_neutrino` merged into the nu
cluster. uboone never fit those bridges because a **second graph-examination
round** ran between Q/L matching and NeutrinoID:

- `WCPPID::Protect_Over_Clustering` (`pid/src/ProtectOverClustering.cxx:6-160`,
  called at `wire-cell-prod-nue.cxx:1322` on every beam-window bundle; the
  same `map_parentid_clusters` is what `wire-cell-prod-stm.cxx:815-830` reads).
- Per bundle member: `PR3DCluster::Examine_graph(ct_point_cloud)`
  (`data/src/PR3DCluster.cxx:2311`) rebuilds the graph keeping only
  inter-fragment bridges that pass `check_connectivity` against the 2D
  charge/dead-channel clouds, then splits at the surviving components.
- The main's largest component keeps the main cluster id
  (`ProtectOverClustering.cxx:57-121`); every other component becomes a new
  cluster in the same parent bundle (`:104-137`), fit separately by
  NeutrinoID (`wire-cell-prod-nue.cxx:1345,1360`), re-associated at shower
  level.

The SBND PR chain (`switch_scope, unmerge_bundle, unmerge_assoc, steiner, …`)
had no counterpart: the un-merges restore *pre-merge cluster boundaries* but
cannot split **within** a cluster id.

**The SBND-specific risk (owner's point 3): the cathode gap.** uboone had no
cathode plane inside the active volume; SBND's sits at x=0 (±0.45 cm physical
gap, ~4-5 cm apparent charge gap, ~1.1 cm transverse offset — doc pr/20).
The toolkit already states the consequence outright
(`ClusteringUnmergeBundle.cxx:292-297`): *"splitting on graph connectivity is
a clustering decision (it breaks cathode crossers, whose two halves the
relaxed graph does not join)"*. A verbatim port would undo the doc pr/20
cathode work (A1/A2 `cathode_connect` joins + B0 kink veto). Hence the one
deliberate divergence: a **cathode re-join pass**, knob-gated, default OFF in
C++ (prototype-faithful) and ON in the SBND config.

## 2. Implementation (landed, default OFF)

### 2.1 C++ — `clus/src/ClusteringProtectBundle.cxx` (new file)

`ClusteringProtectBundle` (`IConfigurable` + `Clus::IEnsembleVisitor`,
`NeedDV`+`NeedPCTS`), modeled on `ClusteringUnmergeBundle` (no production file
touched — fork-by-duplication stance). Per visit on the "live" grouping:

1. **Beam gate** (`beam_window_only` + `beam_window_low/high`, same keys and
   idiom as `CreateSteinerGraph.cxx:127-160`): gids of `Flags::main_cluster`
   clusters with `cluster_t0 ∈ [low, high)`; a member is any cluster sharing
   such a gid. Prototype scope: `to_be_checked` is built from beam-window
   flashes only (`wire-cell-prod-nue.cxx:1313-1320`), and **every member** of
   the bundle is examined, main and companions alike
   (`ProtectOverClustering.cxx:57-121` main, `:123-136` others).
2. **Component split**: `cluster->connected_blobs(m_dv, m_pcts, graph_name)`
   (the toolkit `Examine_graph`, `Facade_Cluster.cxx:3061-3078`).
3. **Cathode re-join pass** (the divergence; `cathode_rejoin_xcut <= 0`
   disables): per component pair, closest points via per-component
   `Simple3DPointCloud` (construction as `connect_graph_relaxed.cxx:71-82`);
   union the pair when both endpoints are within `cathode_rejoin_xcut` of
   `cathode_x`, 3D gap < `cathode_rejoin_dis`, transverse offset
   < `cathode_rejoin_dyz`. Union-find, lowest id kept.
4. Longest surviving component by `get_length()` keeps the retained cluster
   (prototype keeps largest by mcell count, `ProtectOverClustering.cxx:60-70`
   — recorded divergence #2, the established toolkit idiom from
   `ClusteringUnmergeBundle::groups_from_components`); fragments:
   `grouping.separate()`, `main_cluster` flag cleared, `associated_cluster`
   set, ident `alloc_ident(taken, main_ident*100 + sub_id)`, per-part
   `perblob` carve, blob-conservation error check — all the
   `ClusteringUnmergeBundle` idiom verbatim.

Determinism: no pointer-keyed iteration anywhere; component ids, `separate()`
maps, and the union-find are int-keyed; ties break to lowest id; MABC re-runs
`enumerate_idents('tree')` after every visitor.

Recorded prototype divergences (M15, decided by measurement in V1/V2):
- **graph flavor**: config `graph_name`, default `"relaxed"` — the documented
  `Examine_graph` mapping (`clus/docs/patternrecognition/examine_graph_review.md:47`)
  and what unmerge/recovering/examine_bundles use. `"relaxed_pid"` is the
  *structurally* closer port of the pid-stage graph (only it calls the ported
  `check_connectivity`, `connect_graph_relaxed.cxx:585-756`, with the ≤0.9 cm
  retry and `examine_middle_path`, no MST). V1 measures both.
- **main selection**: `get_length()` vs prototype mcell count (above).

### 2.2 Config

- `cfg/pgrapher/common/clus.jsonnet`: `protect_bundle(...)` builder next to
  `unmerge_bundle`, key-suppression on every null knob.
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet`: `cm_by_name.protect_bundle`
  (beam window threaded from the existing `beam_window` arg), new
  `protect_*` args on `clus_pr` and `pr()`; SBND `pr()` defaults =
  the operating point `protect_cathode_x=0`, `rejoin_xcut=5*wc.cm`,
  `rejoin_dyz=4*wc.cm`, `rejoin_dis=8*wc.cm` (**INTERNAL units** — unlike
  the cm-taking `cathode_kink_xcut` one block up; the doc pr/20 trap).
- `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`: matching TLAs +
  passthrough (both entry points set, the doc pr/20 "explicit null overrides
  the other file back to OFF" gotcha).
- **`pipeline_names` defaults unchanged everywhere** until V6: the stage acts
  only when named, so every existing pipeline is byte-identical.

### 2.3 Proofs at landing

- Build+install rc=0; `local/lib/libWireCellClus.so` mtime 2026-08-02 14:33 >
  source edit; factory symbol present (M1 freshness proof).
- `./build/clus/wcdoctest-clus`: **565/565 assertions, SUCCESS**.
- Compiled-config, stage absent: `compile_prjob_cfg.sh` (production 13-name
  pipeline) on HEAD-cfg vs edited cfg → **BYTE-IDENTICAL**.
- Compiled-config, stage named: pipeline gains exactly
  `ClusteringProtectBundle:pr` between `ClusteringUnmergeBundle:prassoc` and
  `CreateSteinerGraph:pr`; its `data` block carries
  `beam_window [200,2200)` + `graph_name relaxed` + rejoin 50/40/80 (internal
  = 5/4/8 cm).

### 2.4 Runner plumbing (this repo)

- `run_pr_chain_batch.sh`: `SBND_PROTECT_BUNDLE=1` inserts the stage after
  `unmerge_assoc`; `SBND_PROTECT_GRAPH`, `SBND_PROTECT_REJOIN_XCUT/_DYZ/_DIS`
  (cm, converted via `wirecell.jsonnet` to internal units) override knobs.
- `run_pr_evt.sh`: `-protect` flag (works with `-stm/-tgm/-nu/-dnn` or an
  explicit `-p` containing `unmerge_assoc`), same env overrides.
- `run_nusel_evt.sh`: `-protect` flag; refuses without the un-merges.
- Bare runs of all three remain the pre-pr/23 production chain until V6.

## 3. V1 — pilot on evt 386948: the pathology is gone

Fresh roots `work-pr23-v1a` (graph `relaxed`) / `work-pr23-v1b`
(`relaxed_pid`), ql_evt386948 symlinked from the doc pr/22 exhibit
`work-pr22gap-c` (untouched), runner:

```bash
SBND_INPUT_DIR=$PWD/work-pr22gap-input SBND_WORK_ROOT=$PWD/work-pr23-v1a \
  ./run_pr_evt.sh data -nu -protect 1          # v1b adds SBND_PROTECT_GRAPH=relaxed_pid
python3 gapjump_probe.py work-pr23-v1a/pr_evt386948/mabc-pr.zip
```

| arm | fit pts | uncovered | stretches | DIFF-fragment bridges | stage cost |
|---|---|---|---|---|---|
| OFF (doc pr/22 §6, `work-pr22gap-c`) | 634 | 50 (7.9%), 33.3 cm | 7 | 4 (29.7 cm) | — |
| ON `relaxed` | 559 | 6 (1.1%), **1.8 cm** | 1 | 1 (1.8 cm, 1.5 cm off charge) | 83 ms |
| ON `relaxed_pid` | 512 | **0 (0.0%), 0.0 cm** | 0 | 0 | 59 ms |

The stage log shows exactly the diagnosed split: `cluster 16 (main): 626
blobs -> retained 313 + 7 fragment(s)` (`relaxed`; 9 fragments with
`relaxed_pid`, incl. one same-side cathode re-join at x=-3.5, 1.2 cm gap).
`nu_evaluated` stays 1 in both arms; zero dead-area overlap; the 29.7 cm of
photon-bridging trails are eliminated by either flavor.

## 4. V2 — cathode robustness and re-join tuning

Arms from the pinned mcp1k Q/L root `work-mcp1kall-d59k` (NOTE: pre-pr/20
trees, so the class-A crossers 315497/406796 enter the PR job still split —
their straddle pairs below are pre-existing input structure, not stage
collateral), events 169824, 406796, 315497, 286400, 409634:

| arm | config | new cathode-straddle splits vs OFF |
|---|---|---|
| `work-pr23-cathO` | stage OFF | (baseline) |
| `work-pr23-cathA` | ON, `relaxed`, re-join OFF | **1 — evt 286400 main broken at the cathode** (fragment 58, gap 4.8 cm, x +2.2/−1.6, dyz 2.9 cm; selected length 338.3 → 306.0 cm) |
| `work-pr23-cathB` | ON, `relaxed`, re-join 5/4/8 cm | **0** — the 286400 pair is re-joined (log quotes the exact geometry), selected length restored to 338.3 cm; straddle-pair counts equal arm O on all 5 events; 409634's crossing cluster keeps its full x-extent |
| `work-pr23-cathC` | ON, `relaxed_pid`, re-join OFF | 286400 broken (as A) + more aggressive debris splits (409634 selected length 43.8 → 16.0 cm, vertex moves 95 cm) |
| `work-pr23-cathD` | ON, `relaxed_pid`, re-join 5/4/8 cm | 286400 protected; 2 straddling re-joins fire on 409634; but one NEW band pair remains (21↔62, gap 8.2 cm, **dyz 6.6 cm** — outside crosser phenomenology, doc pr/12's ~1.1 cm median offset), the crossing cluster is trimmed (x max 7.9 → 5.8 cm), and 315497's vertex moves 55 cm |

Notes:
- **169824 (the class-B crosser) is never split by either flavor** — the
  relaxed close-graph joins its 4.7 cm tip-to-tip gap, so the doc pr/20
  fear ("the relaxed graph does not join the halves") applies to the
  *bridge-starved* component pairs, not to this crosser's geometry.
- The re-join criteria (both endpoints within 5 cm of x=0, 3D gap < 8 cm,
  transverse offset < 4 cm) fire exactly on real crossing boundaries
  (286400: 4.8/2.9 cm; 409634 pid: 1.8/0.2 cm) and stay silent on the
  dyz 6.6 cm debris pair — the 4 cm dyz cut is doing the intended
  physics (crosser transverse offset ~1.1 cm median, ≲3.7 cm observed).

### 4.1 Flavor decision: SBND ships `graph_name = "relaxed"`

`relaxed_pid` is the structurally closer port of the uboone pid-stage graph
and scores marginally better on the pilot (0.0% vs 1.1%), but on the cathode
set it is measurably more aggressive: it trims a genuine crossing cluster,
leaves a new cathode-band pair, and relocates two selected vertices by
55-95 cm on the stub events. `relaxed` at the arm-B operating point removes
94% of the pilot pathology with **zero** cathode collateral and is the
flavor every other PR-stage consumer (unmerge, recovering, examine_bundles,
TGM components) already uses. `relaxed_pid` stays one knob away
(`SBND_PROTECT_GRAPH=relaxed_pid`) for a future tightening round.
(M15: both readings measured and recorded, decision by data.)

### 4.2 Operating-point audit + production-tree re-validation (owner request)

**Audit.** The owner asked whether the pr/20 cathode flags were actually set.
Finding: the cfg flags (B0 `cathode_kink_xcut=5`, Part I P1-P4, A1/A2
`tip_touch_cut`/`crosser_pca_angle`) were ON in every pr/23 arm — all arms ran
14:42-15:36 on 2026-08-02, after the last flip (`c8f19b92`, 11:33) — **but the
V1/V2 input trees are legacy** (`work-pr22gap-c`, `work-mcp1kall-d59k`):
they predate A1/A2 and carry no `was_main` array, so P2 fails CLOSED
(per-cluster WARN "restore_demoted_mains is on but 'real_cluster_was_main' is
absent … not flagged" — quoted from both arms' logs). §3-§4.1 are therefore
tuning evidence on frozen shared inputs, not a production-consistent cathode
measurement. V3 (§5) is unaffected: its hub was rebuilt at HEAD
(compiled-config dump: B0/P2/P3/P4 all true; hub log shows
ClusteringCathodeConnect + demoted_main restores firing).

**Re-validation on production trees.** Fresh hub `work-mcp1kall-pr23cath`
(13 cathode events = cathode-5 + cath13-8, `run_full1k_nusel.sh`
TAG=pr23cath, imaging from our `work-mcp1000`; `was_main` + cathode_connect
verified). Three PR arms, 13/13 rc=0 each: `work-pr23c-off` (no stage),
`work-pr23c-noRJ` (stage ON, re-join disabled), `work-pr23c-on` (stage ON,
re-join 5/4/8 = shipped operating point).

Straddle census (band 6 cm, gap < 10 cm):

| evt | OFF | noRJ (prototype-faithful) | ON (re-join 5/4/8) |
|-----|-----|---------------------------|--------------------|
| 169824 | crosser intact | **broken** (gap 3.18) | **restored** |
| 286400 | crosser intact | **broken** (gap 4.82) | **restored** |
| 406796 | crosser intact | **broken** (gap 3.65) | **restored** |
| 56463 | crosser intact | **broken** (gap 2.63) | **restored** |
| 287654 | crosser intact | **broken** | **NOT restored** (residual, below) |
| 315497, 409634 | pre-existing cathode pairs | identical | identical |
| 348691, 59003 | no change | no change | no change |
| 288952, 392200, 398690 | no nu candidate in any arm | — | — |
| 52195 | no nu candidate (TGM) | candidate appears | candidate appears (flag, below) |

So on production trees the prototype-faithful stage breaks **5/13** cathode
crossers and the re-join restores **4 of 5** — confirming both the risk and
the fix at the production operating point.

**Residual 1 — evt 287654 (re-join out of reach).** The OFF crosser
(cluster 12, x −9.9 → +43.3) is split by the graph into a negative-side main
(ends x −0.5) and a positive-side fragment starting at **x +21.4**: the
apparent cathode gap along this steep track is ~22 cm in x, far beyond
`cathode_rejoin_dis=8 cm` (and the fragment endpoint is outside the 5 cm
band). A1/A2 merged it at clustering with PCA-crosser logic, which proximity
re-join cannot imitate. Candidate improvement (not implemented): an
A2-style direction/PCA-agreement re-join term for exactly this topology.

**Flag 2 — evt 52195 (TGM defeat by splitting; rule 7, reported not tuned).**
OFF: in-window cluster 13 (L 469.8 cm) is TGM=true → skipped as cosmic.
ON/noRJ: the stage splits it (L 417.9 cm), TGM=false, and the cosmic is
**promoted to the selected neutrino candidate**. Mechanism: TGM needs both
track ends at boundaries; splitting off an end-adjacent fragment defeats it.
Note the uboone prototype ran Protect_Over_Clustering in the *nue/stm*
executables — the toolkit pipeline order (`protect_bundle` before
`tagger_check_tgm`) is a placement choice that lets splits change cosmic
verdicts. Whether to reorder (taggers first, then protect + steiner rebuild)
is an owner decision; V4's valfast census quantifies how often this happens
at population scale.

**Bee sets (owner: "old vs new for the cathode-5 off cases")** —

| set | trees | URL |
|-----|-------|-----|
| cathode-5 OFF, OLD (d59k legacy trees, arm `cathO`) | pre-pr/20 | <https://www.phy.bnl.gov/twister/bee/set/20d4c217-34be-48e1-9b74-3e8c420198f7/event/list/> |
| cathode-5 OFF, NEW (production trees, arm `pr23c-off`) | at HEAD | <https://www.phy.bnl.gov/twister/bee/set/c5c94f9e-9743-4a6d-aedc-9ff92cf0427e/event/list/> |
| cathode-5 ON, NEW (production trees, re-join 5/4/8) | at HEAD | <https://www.phy.bnl.gov/twister/bee/set/23c92803-47ac-4f05-bb44-86011fa2ca17/event/list/> |

Event order in all three: 169824, 286400, 315497, 406796, 409634.

## 5. V3 — nueCC48 track_fit vs shower_track census

Arms (fresh, M13):
- Hub `work-nuecc48-poc0` — 48-event nusel Q/L rerun at current HEAD
  (post-pr/20-flip trees; `was_main` verified present in the pctree
  tensorsets), imaging seeded by symlink from `work-nuecc48-nuf`.
- `work-poc48-off` / `work-poc48-on` — full 13(+1)-stage PR chain via
  `run_pr_chain_batch.sh` on that hub; ON arm = `SBND_PROTECT_BUNDLE=1`
  at the shipped operating point (`relaxed`, re-join 5/4/8 cm). 48/48
  `rc=0` on both arms; stage present in all 48 ON logs, absent from all
  48 OFF logs.

Stage activity (ON): **40/48 events split; 67 bundle clusters → 223 extra
associated fragments; 1 cathode re-join** —

    <ClusteringProtectBundle:pr> cluster 19: cathode re-join comp 0+1
      (gap 3.30 cm, dyz 2.70 cm, x -0.48/1.42 cm)      [evt 360535]

i.e. one real-data cathode crosser would have been broken by the prototype
behavior and was preserved by the re-join pass.

### 5.1 Coverage census (`pr23_fitcover_census.py`, cover = 1 cm)

Full table: `docs/pr/23_fitcover-nuecc48.tsv` (96 rows). Per-arm totals:

| arm | events | uncovered fit pts | dead | cathode | assoc | bridge | stitch |
|-----|-------|--------------------|------|---------|-------|--------|--------|
| OFF | 47 | 3211/34627 (**9.3%**) | 0.0 | 30.0 cm | 7.6 cm | **2045.7 cm** | 311.7 cm |
| ON  | 47 | 1020/32725 (**3.1%**) | 0.0 | 0.0 cm | 1.1 cm | **208.5 cm** | 286.9 cm |

(47 not 48: evt 116962 has no track_fit/shower_track layers in EITHER arm —
its only in-beam bundle is tagged TGM, a pre-existing property.)

- **Bridge class (the doc pr/22 §8 pathology): 2045.7 → 208.5 cm (−90%).**
  The OFF worst offenders collapse: 400474 205.7→18.6 cm, 389538 161.1→0,
  269774 147.0→10.2, 234638 125.9→0.
- **Residual "bridge" is mostly reclassification noise, not void jumping**:
  per-stretch inspection (42280, 239794) shows charge at median 1.4-2.4 cm
  (max ≤2.9 cm) from the fit points — micro-hops over near-continuous charge
  that only count as "different components" because of the census's 3 cm
  linkage. One genuine residual hop remains: 400474 seg 22032, 18.6 cm with
  charge at median 3.5 / max 6.4 cm — a ct-consistent gap the `relaxed`
  graph keeps (would need `relaxed_pid` or a tighter graph to cut; recorded,
  not chased).
- **Cathode class 30.0 → 0.0 cm** (267597 10.2 cm + 400474 19.8 cm both
  resolved by the re-fit of the split pieces).
- **Dead class is genuinely empty**, not a census defect: only 2/34627 fit
  points lie inside any channel-deadarea polygon in this sample (containment
  self-test passes; the SBND dead regions are small and the fitter does not
  route through them here).
- The remaining uncovered length is dominated by **stitch** (same-charge
  component, the designed WCP shower stitching) — the "makes sense" residual.

### 5.2 Cathode gate on real data

Straddle-pair scan (`pr23_cathprobe.py` logic batched over all 48×2 zips,
band 6 cm, gap < 10 cm): **zero new cathode-straddle splits in the ON arm.**
The only event with such pairs at all, 267597, has the identical two pairs
in both arms (gaps 1.94 / 7.11 cm; only cluster ids renumber). Combined with
the 360535 re-join above: the SBND-specific risk is controlled on data.

### 5.3 Score movement (`pr_scores_table.py`, both arms)

38/48 events move in scores; **no event changes `event_label` or
`nu_evaluated`** (no nu/cosmic verdict flips). The systematic pattern is the
intended one: `nu_sel_len_cm` down, `nu_sel_n_assoc` up (bridged fragments
become separate associated clusters that are fit on their own).

nue-side net effect is **positive**: `nue_score > 0` on 36 → 38 of 48
events; nue-tagger-not-filled (`br_filled=0` ⇒ score −15) 3 → 2 events:

| evt | nue_score OFF → ON | what happened |
|-----|--------------------|----------------|
| 10550 | −15 → −4.3 | vertex moved 77 cm (x −14.8→+60.8!), tagger now fills |
| 433451 | −15 → **+4.3** | vertex moved ~4 cm, tagger now fills, saturated nue-like |
| 271851 | **+4.3 → −15** | vertex moved 25 cm, main 121→71 cm; no ≥80 MeV electron shower attaches at the new vertex (`NeutrinoTaggerNuE.cxx:4299-4309` gate) |

**Flag for owner (rule 7 — reported, not tuned): evt 271851** (the doc
pr/18 iso-band recovery event) loses its nue evaluation ON; 10550's vertex
relocation is also large enough to warrant a look. Both are in the V5 Bee
OFF/ON exemplar list together with 400474 (biggest bridge cleanup + the one
real residual hop), 360535 (re-join fired), 267597 and 269774
(len 174→75 cm).

## 6. V4 — broad impact (valfast 629) (pending)

## 7. Bee sets (for owner examination)

Built with `make_pr_bee.py` (Q/L layers incl. deadarea merged from the
matching ql roots), artifacts + build logs + `urls.txt` under `bee-pr23/`.
OFF/ON pairs share bee event indices, so the same index shows the same event
in both sets.

| set | events | URL |
|-----|--------|-----|
| nueCC48 **OFF** | 47 (116962 has no nu candidate — TGM) | <https://www.phy.bnl.gov/twister/bee/set/cb297161-73f5-4a14-8909-b64ffbc0a9fe/event/list/> |
| nueCC48 **ON** | 47 | <https://www.phy.bnl.gov/twister/bee/set/4c44a311-c16e-4fcd-a327-63773ff38a61/event/list/> |
| 386948 OFF (pilot) | 1 | <https://www.phy.bnl.gov/twister/bee/set/84d691fe-d423-4586-b906-812420bc1407/event/list/> |
| 386948 ON | 1 | <https://www.phy.bnl.gov/twister/bee/set/307fd253-3a41-4133-867a-01565ac04aff/event/list/> |
| cathode-5 OFF (`cathO`) | 169824 286400 315497 406796 409634 | <https://www.phy.bnl.gov/twister/bee/set/20d4c217-34be-48e1-9b74-3e8c420198f7/event/list/> |
| cathode-5 ON (`cathB`, re-join 5/4/8) | same 5 | <https://www.phy.bnl.gov/twister/bee/set/8e342b4b-79fe-4b07-aab1-6d4a24263d1a/event/list/> |

Exemplar bee indices in the nueCC48 pair (§5.3):

| bee idx | evt | why look |
|---------|-----|----------|
| 21 | 271851 | **the flagged regression**: nue tagger stops filling ON (vertex moved 25 cm, main 121→71 cm) |
| 3 | 10550 | vertex relocates 77 cm ON; nue tagger starts filling |
| 27 | 433451 | nue tagger starts filling ON, saturated nue-like |
| 45 | 400474 | biggest bridge cleanup (206→19 cm) + the one real residual 18.6 cm hop |
| 23 | 360535 | the cathode re-join fired here — crosser preserved |
| 40 | 267597 | pre-existing cathode-band pairs (unchanged) + numu_score sign flip |
| 20 | 269774 | largest main-length change (174→75 cm) |

## 8. V6 — production flip (pending)

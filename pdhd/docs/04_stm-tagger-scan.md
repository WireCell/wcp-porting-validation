# doc pdhd/04 -- the STM tagger's own population on PDHD: a Bee scan set, and where the candidates die

**Status (2026-09-06, owner request).** Owner observation: *"the STM tagger does not really pick
too many real STM with Michel electrons, unlike the PDVD; the chain may still have some issues."*
This doc delivers (a) a Bee set of six PDHD events showing the STM tagger's **fitted** population
with its Steiner graph, Steiner terminals, cluster image and fit trajectory, and (b) the
tagger-level census that says where the candidates are lost, PDHD against PDVD.
**No code or config is changed by this doc.**

## 0. Repro

```
cd /home/xqian/toolkit-dev/toolkit/pdhd
# the six arms (chain UP TO the STM tagger; fresh tag, pinned libs -- M1/shared-tree rule)
ARM=d04bee PIN=<libpin dir> MODE=-stm EVENTS="0 1 12 16 20 22" ./docs/scripts/run_d03_arms.sh
# the combined Bee set (order = Bee index order, fixed forever once uploaded)
./docs/scripts/d04_bee_pr_combine.sh d04bee 029107 -e 12,1,16,20,22,0
BROWSER=echo ./upload-to-bee.sh bee-pr-run029107-d04bee.zip
# per-event census of the tagger itself (reads the PR log + the Bee zip, never a log-line count)
python3 docs/scripts/d04_stm_tagger_census.py work/029107_{12,1,16,20,22,0}_d04bee
# the population TSV of sec 3 (PDHD arm d03nu9 x30, PDVD arm d48nu3 x120 -- both pre-existing)
#   -> docs/figs/d04_stm_tagger_census.tsv
# sec 6, the cluster-128 knob arms (evt 12 only; -A for STRING TLAs, -S evaluates as jsonnet)
ARM=d04tgmA  PIN=<pin> MODE=-stm EVENTS="12" EXTRA="-S tgm_chord_charge=false"   ./docs/scripts/run_d03_arms.sh
ARM=d04tgmB  ... EXTRA="-S tgm_main_pair=false"      ARM=d04tgmC ... EXTRA="-S tgm_chord_charge=false -S tgm_main_pair=false"
ARM=d04tgmD2 ... EXTRA="-A tgm_main_pair_mode=path"  ARM=d04tgmE2 ... EXTRA="-A tgm_chord_mode=chord"
# the causal control: production knobs, steiner stage removed
LD_LIBRARY_PATH=<pin> ./run_pr_evt.sh -s d04tgmF \
    -pipe switch_scope,flag_mains,fiducialutils,tagger_check_tgm 029107 12
grep 'TaggerCheckTGM: cluster 128 ' work/029107_12_*/wct_pr_029107_12.log
# sec 7, the point-level probe (env-gated in clus/src/TaggerCheckTGM.cxx; inert when unset)
./wcb build --notests -p            # build only -- do NOT install, keep local/lib as the gate pin
PIN=<dir with local/lib's libWireCell*.so + build/clus/libWireCellClus.so>
LD_LIBRARY_PATH=$PIN WCT_TGM_PATH_DUMP=128 WCT_TGM_PATH_DUMP_DIR=<out> \
    ./run_pr_evt.sh -s d04probe -stm -stm-fit 029107 12      # one cluster
LD_LIBRARY_PATH=$PIN WCT_TGM_PATH_DUMP=-1  WCT_TGM_PATH_DUMP_DIR=<out> \
    ./run_pr_evt.sh -s d04prb   -stm        029107 <evt>     # every evaluated cluster
python3 docs/scripts/d04_cluster128_plot.py docs/figs/d04_tgm_path_cluster128.csv.gz \
    docs/figs/d04_exclusion_by_apa.tsv -o docs/figs/d04_cluster128_path_veto.png
```

Binary: `libWireCellClus b46179b2` = toolkit `5d0b4e77` plus the uncommitted doc sbnd_xin/pr/143
`break_segment` edits of a concurrent session.  `break_segment` has no caller in
`TaggerCheckSTM.cxx`, and the claim is **gated, not argued**: event 0's `mabc-pr.zip` member
content hashes (`abtest/hash_archive.py`) are IDENTICAL to arm `d03stmnew11`, which ran the same
event on the shipped pin `new11` (`a4ff5439`).  The `-stm` chain is unmoved.

## 1. The Bee set

<https://www.phy.bnl.gov/twister/bee/set/1450e84b-009b-4779-a14a-88c14e3d4fae/event/list/>

| bee idx | evt | art event | mains | **fitted** (= the 4 STM layers) | **tagged** (`stm_tagged`) |
|---|---|---|---|---|---|
| 0 | 12 | 1079 | 96 | 29 | 8 |
| 1 | 1 | 991 | 99 | 25 | 7 |
| 2 | 16 | 1111 | 103 | 32 | 7 |
| 3 | 20 | 1143 | 150 | 29 | 9 |
| 4 | 22 | 1159 | 92 | 22 | 5 |
| 5 | 0 | 983 | 98 | 27 | 2 |

Layers per event: `clustering` (every main), then `stm` (3-D image), `steiner_graph` (Steiner
node cloud, `real_cluster_id` 1 on terminals), `steiner_terminals` (that cloud thinned to
terminals) and `stm_fit` (the fit trajectory, q = dQ*0.1 - 1000) -- all four scoped to the
clusters the tagger **fitted** (`require_pc:'stm_fit'`), so they agree object-for-object; plus
`stm_tagged`, the `flag_STM` verdict subset, and the dead-area layers.

**Scope note.** PDHD carries the *fitted* scope (doc pdvd/39 round 2).  PDVD's nominal since the
owner's 2026-09-05 decision (doc pdvd/39 round 3) is `require_flag:'STM'` -- the *tagged* set, and
its `stm_tagged` layer is gone.  On evt 0 that is 27 objects vs 2.  This set is the fitted scope,
per the request's wording; `stm_tagged` is in the same view for the verdict subset.

Frame check: every cluster's `stm_fit` trajectory lies inside its own `stm` cloud bounding box
(0/27 escapes on evt 0), so the fit and the image are in the same drift frame -- the trap of
`feedback_frame_mismatch_hides_at_small_t0` does not apply here.

## 2. Where the mains are lost before a fit exists

Per-main, over the 30-event arm `d03nu9` (PDHD 029107) and the 120-event arm `d48nu3` (PDVD):

| | PDHD /evt | PDVD /evt | PDHD % of mains | PDVD % of mains |
|---|---|---|---|---|
| mains evaluated by `TaggerCheckSTM` | 74.9 | 38.2 | -- | -- |
| **fitted** (the 4 Bee layers) | 26.0 | 14.8 | 34.8 % | 38.7 % |
| **tagged** STM | 5.8 | 5.0 | 7.7 % | 13.0 % |

No-fit reasons (distinct clusters):

| reason | PDHD | PDVD |
|---|---|---|
| evaluated, no pass recorded (exited after the round-1 fit) | 1465 (65 %) | 2810 (61 %) |
| fully contained ("Mid Point A" pre-fit exit test) | 1073 (48 %) | 2544 (55 %) |
| **no `steiner_pc`** | **319 (14.2 %)** | **182 (4.0 %)** |
| round-1 forward fit gave <=3 points | 58 | 66 |

The `no steiner_pc` excess is `create_steiner_tree` returning an empty graph because it found
**0 or 1 terminals** (`SteinerGrapher.cxx:165`, `need >=2`).  Traced, not inferred: all
**319/319** `no steiner_pc` mains carry a matching
`CreateSteinerGraph: create_steiner_tree produced no steiner_graph for main N` line in the same
event (450 mains get an empty tree in total; the other 131 never reach `TaggerCheckSTM` as
evaluated mains -- already TGM, or out of scope).  Count mains per event as a SET, not by log
line: `steiner` and `steiner_refresh` are two `CreateSteinerGraph` instances and each logs the
same main.  This is the one input-side number where PDHD is 3.5x PDVD, and it is exactly what the
`steiner_terminals` layer of sec 1 displays.

## 3. Where the FITTED candidates are rejected

Per-cluster STM status code (`TaggerCheckSTM.cxx:849` legend; 0 wins if any pass accepted):

| status | meaning | PDHD (781 fitted) | PDVD (1777 fitted) |
|---|---|---|---|
| 0 | **accepted (STM)** | 174 (22.3 %) | 594 (**33.4 %**) |
| 2 | long leftover past kink (Mid Point A) | **216 (27.7 %)** | 109 (6.1 %) |
| 3 | dQ/dx eval / KS tests | 176 (22.5 %) | 692 (38.9 %) |
| 4 | extra-tracks veto | **93 (11.9 %)** | 21 (1.2 %) |
| 5 | proton endpoint | 4 (0.5 %) | 56 (3.2 %) |
| 7 | accept_guards / cathode / leftover / deficit / vertex-kink | 117 (15.0 %) | 305 (17.2 %) |
| 8 | round-2 fit returned no points | 1 | 0 |

Named guard lines account for only 124 of PDHD's 607 fitted-but-rejected (65 `readout_edge_guard`,
59 `cathode_guard`) and 306 of PDVD's 1183 -- the rest is carried by the status code alone.

(Status 2's parenthetical "Mid Point A" is the *post-fit* leftover test and is a different code
path from sec 2's pre-fit "fully contained (Mid Point A)" no-fit reason.  Do not join the tables
on that label.)

**The signature is status 2 and status 4, not charge quality.**  PDHD rejects 4.5x more often on
"long leftover past the kink" and 10x more often on the extra-tracks veto, while rejecting *less*
often than PDVD on the dQ/dx shape tests (22.5 % vs 38.9 %).  Both codes mean the same thing
physically: **there is more track-like material attached to the candidate than the tagger will
accept around a stopping muon.**  That points at the cluster the tagger is handed (partition /
over-clustering / delta rays), not at the Bragg test -- which is the opposite of what doc pdhd/03
found *downstream* of the tagger, where PDHD's problem was dead charge stretches.

**The knob confound is excluded, not assumed.**  `pdhd/pr.jsonnet` is a fork by duplication of
`protodunevd/pr.jsonnet` and does diverge deliberately elsewhere (no curved FV, different TGM
margins), so the thresholds were compared before the sentence above was written:

* Compiled `TaggerCheckSTM` `data` blocks (`.wct-pr_d03nu9.json` vs `.wct-pr_d48nu3.json`):
  25 keys each, **17 identical**, and the 5 that differ are all detector-inherent --
  `fiducial`, `fv_tolerance`, `mip_dqdx` (56000 vs 55000 e/cm, 1.8 %), `readout_nticks`,
  `trackfitting_config_file`.  There is **no knob for either test**.
* The two tests are hard-coded prototype thresholds in identical code:
  status 2 is `left_L > 40 cm || (left_L > 7.5 cm && leftover dQ/dx > 2.0 MIP)`
  (`TaggerCheckSTM.cxx:3710`), status 4 is `check_other_tracks()` (`:3864`).  `m_mip_dqdx`
  is their only detector-dependent input and it differs by 1.8 %.
* The two `*_track_fitting.json` files differ in 18 of 57 keys, all of them diffusion /
  transverse-smearing constants (doc pdhd/02, doc pdvd/44) plus three PDVD-only keys that are
  0/inert (`end_trim_gap_len` 0, `good_point_pitch_frac` 0, `proj_skip_unmapped_face` -- doc
  pdvd/46 measured 0 fires in 231 events).  Nothing there sets a leftover length or a
  second-track threshold.

A residual path remains -- the smearing constants change the fit, and the fit is what the two
tests read -- but no *threshold* differs between the detectors.

## 4. The Michel end of it

From the doc pdhd/03 census TSVs (`d03_stm_michel_d03nu9.tsv`, `48_stm_michel_d48nu3.tsv`),
i.e. the `CheckSTM_Michel` stage acting on the tagged population:

| | PDHD (30 evt, 166 candidates) | PDVD (120 evt, 574 candidates) |
|---|---|---|
| `is_stm` (passes every stage check) | 8 (4.8 %) | 99 (17.2 %) |
| `michel_found` | 19 (11.4 %) | 137 (23.9 %) |
| **`is_stm` AND `michel_found`** | **0** | **39** |

**166, not 174.**  The tagger set `flag_STM` on 174 mains over the 30 events but the census has
166 rows: 8 tagged mains (029107 evt 3 cl 123; evt 5 cl 94/116/121; evt 20 cl 150; evt 21 cl 122;
evt 26 cl 86/115) get **no `CheckSTM_Michel` line at all** -- the stage never evaluated them,
4.6 % of the tagged population lost silently between the tagger and the stage.  The likely
mechanism is the one the Bee layers are bound to the STM visitor to avoid (`pr.jsonnet`: "before
protect_bundle can split a cluster out from under its flag"), but that is **not verified here**.

Zero PDHD candidates are simultaneously a confirmed stopping muon and carry a Michel -- that is
the owner's observation, quantified.  doc pdhd/03 sec 7 already established the downstream half
(71/166 unjudgeable from dead charge stretches, 31 with off-MIP plateaus, 67 stopping inside the
taggers' margins); this doc adds the tagger-level half: the tagger tags 22.3 % of what it fits
against PDVD's 33.4 %, and loses the difference to attached material (sec 3).

## 5. Owner question 2026-09-06: does the PDHD chain gate STM on TGM and FC?

Asked: *"it does not put in TGM tagger?  For PDVD we only run STM tagger if they are not TGM.
Can you update the PDHD chain to match?  Same for FC."*  **Both gates are already in place and
PDHD already matches PDVD; no change was made.**  Evidence, from the same six runs as sec 1:

**Stage order is identical on the two detectors.**  From the compiled configs (not the runner
strings): the `MultiAlgBlobClustering` `pipeline` array is

```
PDHD  ... MakeFiducialUtils:pr, TaggerCheckTGM:pr, TaggerCheckSTM:pr, TaggerCheckFC:pr, ...
PDVD  ... MakeFiducialUtils:pr, TaggerCheckTGM:pr, TaggerCheckSTM:pr, TaggerCheckFC:pr, ...
```
(`work/029107_0_d04bee/.wct-pr_d04bee.json`, `pdvd/work/039252_0_d48nu3/.wct-pr_d48nu3.json`).
TGM runs **before** STM on both; FC runs **after** STM on both.

**The TGM gate fires.**  `TaggerCheckSTM.cxx:566` skips any main carrying `Flags::TGM`.  Over the
six events: **128 of 128** `TGM=true` mains were skipped by `TaggerCheckSTM`, matched by cluster
id, not by count.  Over the 30-event arm `d03nu9`: **645 of 645**.  (That is also why the two
denominators differ: 2891 mains reach TGM, 2246 reach STM -- exactly the 645.)

**The FC gate is inside the STM tagger, which is why the stage order does not need to change.**
`TaggerCheckSTM::check_stm_conditions` (`:3444`) calls
`Facade::cluster_fc_check(cluster, m_dv, m_fiducial, m_fv_tolerance)` and returns immediately on
`is_fc` -- the identical call `TaggerCheckFC.cxx:209` makes, and the compiled config gives both
components the same `fiducial` (`BoxFiducial:pdhd_pr_fv`) and the same
`fv_tolerance` (`[-20,-20,-175,-175,-180,-180]` mm).  Over the six events, `TaggerCheckFC` marks
**202** mains `FC=true` and STM's own early exit fires on **202** -- the *same 202 cluster ids*.
Moving `tagger_check_fc` before `tagger_check_stm` would therefore change no verdict; it would
only evaluate the same fiducial test twice, and it would diverge from PDVD.

Full accounting, 30-event arm `d03nu9` (each row is the input to the next):

| | PDHD | PDVD (d48nu3) |
|---|---|---|
| mains reaching `TaggerCheckTGM` | 2891 (96.4/evt) | 7136 (59.5/evt) |
| `TGM=true` -> STM skips them | 645 (21.5/evt, 22.3 % of mains) | 2549 (21.2/evt, 35.7 %) |
| mains reaching `TaggerCheckSTM` | 2246 | 4587 |
| of those, STM's own fully-contained exit | 1073 | 2544 |
| `TaggerCheckFC` `FC=true` (whole event) | 1077 (37.3 % of mains) | 2550 (35.7 %) |
| fitted | 781 | 1777 |
| tagged STM | 174 | 594 |

Per event the TGM yield is the same on both (21.5 vs 21.2 mains); the *fraction* differs (22.3 %
vs 35.7 %) only because PDHD carries 96.4 mains/event against PDVD's 59.5, and the extra ones are
small fragments that are neither TGM nor FC.  **The fractions are not comparable without matching
the main populations** -- no claim about TGM sensitivity is made here.

What the display does NOT show: there is no `tgm` or `fc` Bee layer in `mabc-pr.zip` on either
detector, so a scan cannot see which clusters were removed before the STM tagger ran.  That is
the likely origin of the question, and it is a display gap, not a chain gap.

## 6. Owner question 2026-09-06: cluster 128 of 29107-1079, why not TGM?

Asked about the point `(-83.2, 382.6, 278.5)`, cluster 128, event 29107-1079 (= evt 12, Bee idx 0
of the sec-1 set): *"why is it not tagged as TGM?  I wonder if the fiducial volume is not properly
set for the taggers."*

**It is not the fiducial volume.**  `BoxFiducial:pdhd_pr_fv` compiles to
x[-358.0, 358.0] y[7.61, 606.0] z[0.234, 462.3] cm with `fv_tolerance`
[-20,-20,-175,-175,-180,-180] mm, and the same object + tolerance reach all three taggers
(sec 5).  Cluster 128's two ends are **outside** the inset volume on the z faces:
the STM tagger's own fit runs `(-305.9, 121.1, 1.1)` -> `(51.0, 545.8, 444.5)` cm, 710 cm
end-to-end / 791 cm of arc, and z = 1.1 vs the zlo inset at 18.2 cm, z = 444.5 vs the zhi inset at
444.3 cm.  Geometrically this is a textbook through-going muon and the FV says so.

**It is not a real charge gap either.**  Along the whole 791 cm fit the trajectory is never more
than **2.13 cm** from a cluster charge point (median 0.48, p90 0.69); there are **zero** stretches
longer than 5 cm without charge.  Re-running the tagger's own 30 cm voxel walk offline on the
dumped cloud puts both ends in one component (547 of 550 voxels; the other 3 are single stray
voxels 37/60/94 cm out, carrying 4/10/9 points).

**It is the chord-charge guard, in `path` mode, fed a thinned cloud.**  Six one-event arms on the
pinned binary, `-stm`, evt 12 only:

| arm | change | cluster 128 |
|---|---|---|
| `d04bee` | PDHD production | TGM=**false** |
| `d04tgmA` | `-S tgm_chord_charge=false` | TGM=**true** |
| `d04tgmB` | `-S tgm_main_pair=false` | TGM=false |
| `d04tgmC` | both of the above off | TGM=**true** |
| `d04tgmD2` | `-A tgm_main_pair_mode=path` (not `real`) | TGM=false |
| `d04tgmE2` | `-A tgm_chord_mode=chord` (not `path`) | TGM=**true** |
| `d04tgmF` | production knobs, but `steiner` REMOVED from the pipeline | TGM=**true**, and *no* `check_tgm ... rejected` line at all |

So the veto is `require_chord_charge` + `chord_charge_mode="path"`, i.e.
`TaggerCheckTGM::path_connected()` / `path_components()` (`TaggerCheckTGM.cxx:426,481`).  The
main-component guard and its `real` mode are innocent (arms B, D2).

**Why the path walk fragments a contiguous track.**  `path_components()` skips every point for
which `cluster.is_point_excluded(i)` is true.  That set is written by the *Steiner graph* build:
`connect_graph_ctpc.cxx:543` excludes any point whose nearest reference-cluster point is
`>= 1.0 cm` away ("Key filtering criterion from prototype"), and stores it on the Cluster
(`:561`).  `CreateSteinerGraph:pr` runs **before** `TaggerCheckTGM:pr` in the pipeline (sec 5), so
TGM's charge-connectivity walk sees the Steiner-thinned cloud, not the cluster.  Arm `d04tgmF`
is the causal control: drop the `steiner` stage, nothing else, and the same cluster with the same
knobs is tagged on its first extreme pair.  `set_excluded_points` has exactly two writers in the
tree (`connect_graph.cxx:284`, `connect_graph_ctpc.cxx:561`), both on that path, and
`path_components`'s only other inputs are `npoints()` / `point3d()` / `kd_knn`, which the Steiner
stage does not touch -- so this is the channel.

Two independent log readings say it is the *path connectivity itself* that the Steiner stage
degrades, not merely which pairs get tested:

| | production (steiner) | arm `d04tgmF` (no steiner) |
|---|---|---|
| `component_rescue` (a 30 cm-step path test) | rescued **1 of 7** short components | rescued **8 of 11** |
| `component_extreme_wcps` | 10 comps, 2 above 10 cm -> 6 extreme groups | 14 comps, 3 above 10 cm -> 21 groups |
| pairs tested | 12, **all** path-rejected | first pair passes, **no** rejection line |

**Which exclusion rule.**  Phase 1 of the filter is `is_point_good(cluster, i, 2)`
(`connect_graph.cxx:477`): a 3-D point survives only if **at least 2 of its 3 planes carry
charge > 10**.  Phase 2 is the `>= 1.0 cm` reference-distance test -- and it is **inactive on this
path**: the Steiner build calls `connect_graph_ctpc(cluster, dv, pcts, graph)` with no reference
cluster (`make_graphs.cxx:56-57`), so `use_reference_filtering` is false.  The exclusion that
reaches TGM is therefore purely the **2-of-3-planes charge test**, which is exactly where
per-plane signal quality enters.

**Not measured**: the per-point excluded fraction (that needs a probe in the job).  A first attempt
to read it off the `steiner_graph` Bee layer is INVALID -- that layer is the Steiner tree's node
cloud, not the non-excluded set -- and is not evidence either way.

**Population.**  Over the 30-event arm `d03nu9`, of 2246 mains the tagger left TGM=false:

| | PDHD | PDVD |
|---|---|---|
| >= 1 `no 30.0 cm-step charge path` rejection | **460 (20.5 %)** | 120 (2.6 %) |
| the path guard is the ONLY rejection reason | **257 (11.4 %)** | 54 (1.2 %) |

An 8x excess against PDVD.  Every one of those is a candidate through-going muon that instead
goes on to the STM tagger -- which is the natural source of the sec-3 signature (status 2 "long
leftover past kink" 216, status 4 "extra-tracks veto" 93; a TGM evaluated as an STM candidate
looks exactly like that).  **The magnitudes are consistent but the link is not proven** -- that
needs the flipped arm and a per-cluster join, not this census.

### 6.0 What "chord charge" means

`tgm_chord_charge` (C++ `require_chord_charge`) is a *veto*, not a tagger.  `TaggerCheckTGM`
finds candidate "ends" of a cluster (8 global extremes, or per-component extremes with
`tgm_component_extremes`), groups them, and for each PAIR of end-groups asks the FV whether the
straight line between them crosses out of the detector on both sides -- that is the TGM
geometry test, and by itself it looks only at where the two points ARE.  Nothing in it asks
whether the cluster actually has charge joining them, so a detached speck sitting near a wall
could pair with a genuine track end and fake a through-going muon (SBND evt285185 cluster 20,
doc sbnd 29).

`require_chord_charge=true` adds "...and the two ends must be joined by charge".  `chord_max_gap`
(30 cm) is how big a hole in that charge is tolerated, and `chord_charge_mode` picks HOW it is
measured:

* `"chord"` (the C++ default) samples the **straight segment** between the two ends and refuses if
  any unsupported run exceeds `chord_max_gap`.  Cheap, but a curved track bows off its own chord
  and is falsely refused (SBND evt285185 cluster 16: a real 480 cm crosser bows 10 cm).
* `"path"` (what all three detectors run) refuses unless the two ends are in the same **connected
  component of the cluster's own points**, where connectivity is a union-find over 5 cm voxels
  joined when their representative points are within `chord_max_gap`
  (`TaggerCheckTGM.cxx:426-491`).  Curvature no longer matters -- but the walk is over the
  cluster's points *minus the excluded set*, which is where cluster 128 dies (above).

`tgm_main_pair` / `main_component_mode` is a second, separate veto on the same pair loop (a pair
with NEITHER end in the cluster's main charge component is a fragment-only pair).  It is not
involved here -- arms B and D2.

### 6.1 Provenance of the guard: SBND, inherited unchanged through PDVD

Asked where `tgm_chord_charge` is set and whether it is PDHD-specific.  **It is not new for
PDHD: it is SBND's, copied verbatim into PDVD and then into PDHD.**

| when | commit | what |
|---|---|---|
| 2026-07-23 | `96d4600b4` | C++ knobs `require_chord_charge` + per-component extremes born, for **SBND** evt285185 cluster 20 ("merge-grafted speck faked a TGM end"); doc `sbnd_xin/docs/29_tgm-chord-charge.md` |
| 2026-07-23 | `742c58982` | `chord_charge_mode="path"` added, for **SBND** evt285185 cluster 16 (a curved 480 cm top->anode crosser bows 10 cm off its chord and lost a real TGM) |
| 2026-07-27 | `b2dab726` | **SBND** config turns the whole bag on (doc sbnd 64): the module defaults become `run_full1k_nusel.sh`'s flag set `-chord -rescue -rescue-chord -main-pair-real ...` |
| 2026-09-02 | `784dc8374` | **PDVD** `protodunevd/pr.jsonnet` copies the bag (doc pdvd/25 sec 13) |
| 2026-09-05 | `32a34ac6` | **PDHD** `pdhd/pr.jsonnet` forks PDVD's (doc pdhd/stm-tagger-chain) |

The C++ defaults never moved (`require_chord_charge=false`, `chord_charge_mode="chord"`,
`chord_max_gap=30 cm`, `TaggerCheckTGM.cxx:340-343`); all three detectors turn it on in cfg.
`cfg/pgrapher/experiment/pdhd/pr.jsonnet:185` says so in as many words: *"Every default below is
the SBND PRODUCTION operating point ... adopted as the module defaults 2026-07-27"*.

**And the artifact it defends against does not exist on PDHD.**  Doc sbnd 29's root cause is
`clustering_examine_bundles(use_flash_t0=true)`, which collapses every cluster in a flash-time
group into ONE `Cluster` so the taggers run on a composite and a detached fragment can donate a
fake TGM end.  SBND runs that stage (`sbnd/clus.jsonnet:635`, `use_flash_t0=true`).  **PDHD does
not** -- `cm.examine_bundles()` is commented out (`pdhd/clus.jsonnet:529`, "DISABLED at the
per-drift-group stage (2026-06-14)"), and PDHD's only `use_flash_t0=true` is on
`cm.cathode_connect` (`:661`), a different stage that deliberately joins the two halves of one
cathode crosser.  PDVD is the same.  Consistent with that, every point of evt-12 cluster 128
carries `real_cluster_id == cluster_id`, and so does every point in the event (161 854 of
161 854) -- no flash-merge composite anywhere.

So on PDHD the guard is paying its full cost (257 mains, sec 6) against a failure mode this
chain does not produce.

### 6.2 Is it APA0's different field response?  A 1.46x excess in rejections -- but see sec 7.5, which settles it at the point level: no

PDHD does run one APA on a different field response: `params.jsonnet:189-193` gives
anode 0 `np04hd-garfield-6paths-mcmc-bestfit.json.bz2` and anodes 1/2/3
`dune-garfield-1d565.json.bz2` (wired per anode at `pdhd/sp.jsonnet:100`).  Since the exclusion
that feeds the path walk is a **2-of-3-planes charge test** (sec 6), a per-APA signal-quality
difference is a physically sensible suspect.

APA map, read off the compiled `DetectorVolumes` metadata and the dead-area layers of these very
events: face 0 = the x < 0 drift volume (APA 0, 2), face 1 = x > 0 (APA 1, 3); z < 231 cm =
APA 0, 1 and z > 231 cm = APA 2, 3.  Bucketing every TGM-evaluated main by the APA holding most of
its points, arm `d03nu9`, 30 events:

| APA | field response | mains | path-guard rejected | rate |
|---|---|---|---|---|
| **0** | np04hd-garfield-6paths-mcmc | 569 | 122 | **21.4 %** |
| 1 | dune-garfield-1d565 | 727 | 103 | 14.2 % |
| 2 | dune-garfield-1d565 | 816 | 130 | 15.9 % |
| 3 | dune-garfield-1d565 | 779 | 108 | 13.9 % |

APA0 21.4 % vs 14.7 % for the other three: a **1.46x excess at +4.6 sigma** (binomial).  Real, and
in the direction the field-response hypothesis predicts.  **But it is not the cause**: 341 of the
463 rejections (74 %) are in the three normal-FR APAs, so an APA0 that behaved like its neighbours
would remove about a quarter of the problem.  Cluster 128 itself spans three of them -- its fit is
in APA0 for the first ~400 cm of arc, APA2 to ~680 cm, APA3 to the end.

**Superseded in part by sec 7.5.**  The probe measures the exclusion at the point level over
776 529 points and finds all four APAs in a 20-26 % band, APA0 (26.2 %) barely above APA2
(25.1 %).  The 1.46x above is a *rejection* excess, not an exclusion excess, and survives only as
a mild secondary effect.

**Still open, and NOT measured**: why the same guard costs PDHD 20.5 % of TGM=false mains and
PDVD 2.6 %.  The suspect is that the 2-of-3-plane threshold (`charge > 10`, `ncut = 2`) is an
absolute, pitch-blind constant of the family `feedback_pitch_blind_constants_pdvd` warns about, and
PDHD's 4.669 mm pitch puts less charge per point than PDVD's.  Running the sec 7.4 probe on a PDVD
arm is the one-command test; not done.

**No change was made.**  The three ways out are all knob-level and all change production output:
`tgm_chord_mode='chord'`, `tgm_chord_charge=false`, or making `path_components()` ignore
`excluded_points` (a C++ change, default-OFF knob).  The last is the one that fixes the cause
rather than removing the guard, and PDVD would want it too.  Owner's call.

## 7. Owner question 2026-09-06: the Bee display looks fine — so what did the guard see?

*"From the Bee display this track looks quite good and consistent with the 3-D image, so I do not
understand why the chord check vetoed it."*  The display is right; the guard is not walking what
the display draws.  Measured with a new env-gated probe (sec 7.4), not inferred.

![cluster 128 path veto](figs/d04_cluster128_path_veto.png)

### 7.1 The guard walks 69 % of the points, and the missing 31 % are one contiguous run

`TaggerCheckTGM::path_components` skips every point with `is_point_excluded(i)`.  On cluster 128
that is **3542 of 11 517 points (30.8 %)**.  Bee draws all 11 517 with their summed charge, so the
excluded ones are invisible on the display — they look exactly like their neighbours (panel A).

They are not scattered.  In 10 cm z-slices the excluded fraction is ~0 % from z = 0 to 60,
jumps to **100 % from z = 80 to 200**, and falls back to ~0 % beyond z = 210 (panel C).  That is a
single **130 cm run** in the middle of the track.

### 7.2 Why those points are excluded: only the collection plane has charge there

`is_point_good(cluster, i, 2)` (`connect_graph.cxx:477`) keeps a point only if **at least 2 of its
3 planes carry charge > 10**.  Per-plane charges from the probe:

| stretch | U | V | W | mean planes > 10 |
|---|---|---|---|---|
| z < 60 cm, x < 0 (healthy) | **exactly 0 on 100 %** | ok (median 1545) | ok (median 9820) | 2.00 |
| z 80-200 cm, x < 0 (the run) | **exactly 0 on 98.2 %** | **exactly 0 on 98.2 %** | ok (median 12 137) | **1.00** |

So the U plane contributes nothing to this track anywhere, and over those 130 cm **V drops to zero
as well**, leaving collection alone.  Of the 3542 excluded points, 3470 have their one surviving
plane = W.  The charge is real and the 3-D points are real -- what is missing is the *induction
attribution*, which is what `is_point_good` counts.

### 7.3 That splits the walk, and the two ends land on opposite sides

With those points gone, the 30 cm-step union-find over the survivors returns **5 components**
(panel B):

| component | points | extent |
|---|---|---|
| 0 | 6093 | x[-140.3, 61.0] y[294.2, 564.3] **z[202.7, 461.3]** |
| 2 | 1773 | x[-306.7, -250.0] y[109.9, 194.7] **z[0.7, 71.1]** |
| 10 / 73 / 291 | 50 / 50 / 9 | specks at z = 124, 182, 436 |

The track's two ends -- `(-306.1, 120.3, 0.7)` and `(60.3, 561.8, 460.7)` -- fall in components
**2** and **0**.  The probe prints the verdict directly:

```
TGMPROBE: cluster 128 path_connected a=(60.3,561.8,460.7) i=9439 comp=0
                                   | b=(-306.1,120.3,0.7) i=2180 comp=2 -> SPLIT
```
Every one of the five pairs involving the far end is `SPLIT`, so `path_connected` returns false,
the chord-charge guard vetoes each pair, and no pair survives to tag TGM.  Nothing about the
geometry, the FV, or the real charge is wrong -- **the guard's connectivity is computed on a cloud
that has had its induction-poor points removed, and on this track that removal is 130 cm long.**

### 7.4 The probe, and its byte-identity gate

`clus/src/TaggerCheckTGM.cxx`: `WCT_TGM_PATH_DUMP=<cluster ident | -1>` (with
`WCT_TGM_PATH_DUMP_DIR`) writes one CSV row per 3-D point -- `x,y,z,excluded,comp,qu,qv,qw` -- and
logs the component census plus every `path_connected` verdict.  No verdict reads it.
**Gate**: the probe build with the env var UNSET reproduces `mabc-pr.zip` member content hash
`91ef5b5185d9bd8a66df3af70f882bc5d9cf7326f75c9add74ee1ff7bb795070` on 029107/12, identical to arm
`d04bee` on the probe-free pin.

### 7.5 It is a detector-wide induction deficit, not APA0 (sec 6.2 refined)

The 130 cm run happens to sit in APA0, which invited the field-response explanation.  Running the
probe over **every cluster the guard evaluated in all six events (776 529 points)** settles it:

| APA | field response | points | excluded | 1-plane points |
|---|---|---|---|---|
| **0** | np04hd-garfield-6paths-mcmc-bestfit | 267 772 | **26.2 %** | 26.1 % |
| 1 | dune-garfield-1d565 | 166 205 | 19.8 % | 20.5 % |
| 2 | dune-garfield-1d565 | 132 323 | 25.1 % | 24.9 % |
| 3 | dune-garfield-1d565 | 210 229 | 21.0 % | 20.7 % |

All four APAs sit in a 20-26 % band and APA0 (26.2 %) is barely above APA2 (25.1 %), which runs the
standard field response.  The plane multiplicity is the same shape everywhere (0/1/2/3 planes =
~0/21-26/51-57/19-24 %), and on 2-plane points the missing plane is U or V roughly 50:50 in every
APA while W is missing <1 % of the time.  **A quarter of every 3-D point on PDHD carries charge on
one plane only, and that plane is the collection plane.**  APA0's different field response is not
the driver -- it survives only as the mild 1.46x excess in *rejections* of sec 6.2, which is a
different quantity and is not contradicted.

This is the same family as doc pdhd/01 (Steiner terminals on wrapped induction planes): both
`wrapped_channel_charge` and `retile_wrapped_channel_activity` are ON in this job (PDHD PR
production since 2026-09-05), so what sec 7.2 measures is the **residual after that fix**, the
`ncharge = 3` shortfall doc pdhd/01 sec 6 left open (0.00 -> 0.53, so ~47 % still short).
Whether the residual is the same wrap-topology lookup or a distinct SP/attribution effect is
**not established here**.

### 7.6 What this changes about the fix

Sec 6's three options stand, but the diagnosis reorders them:

1. `path_components` ignoring `excluded_points` (default-OFF C++ knob) is now clearly the fix that
   matches the cause: the exclusion set is a *Steiner graph-building* quality filter and there is
   no reason a charge-connectivity walk should honour it.  On cluster 128 that alone restores the
   tag (arm `d04tgmF` is the equivalent control).
2. `tgm_chord_mode='chord'` also restores it, but by replacing the test rather than fixing it.
3. Turning `tgm_chord_charge` off removes a guard PDHD does not need (sec 6.1) but also removes
   the protection SBND relies on, and the knob is shared.

Still owner's call; nothing was changed.

## 8. Not done / next

1. The Bee set is the scan sheet; no hand scan has been made from it.  Nothing here is a
   recommendation to change a threshold.
2. The status-2/status-4 excess (sec 3) is the lead.  It needs one hand pass over the `stm` +
   `steiner_graph` layers of a few status-2 clusters to say whether the leftover is a real second
   particle, a delta ray, or a clustering merge -- `feedback_owner_kink_discriminator` says charge
   shape will not separate the first two.
3. The `no steiner_pc` 3.5x (sec 2) is a second, independent lead, and it is on the input side of
   the same layers.  doc pdhd/01 changed the terminal finding for PDHD's wrapped planes; whether
   this residual is the same family is unmeasured.
4. The 8 tagged mains the `CheckSTM_Michel` stage never evaluated (sec 4) are a separate, small,
   fully-specified bug hunt: reproduce one (`029107/5` cl 94 is in the standard manifest) and read
   why the stage's candidate loop skips it.
5. A PDVD Bee set at the same scope (`require_pc:'stm_fit'` rather than PDVD's nominal
   `require_flag:'STM'`) would make the two directly comparable by eye.  Not built -- it is a
   config fork of the PDVD production display and needs the owner's word.

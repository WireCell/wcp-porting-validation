# Applying Wire-Cell pattern recognition to ProtoDUNE-VD: stopping muons + Michel electrons

**Scope declaration: this is a design round. No code and no config is changed
by this document.** Every number below is either read out of the tree at the
commit named in the Repro block, or derived here with the substitution shown.
Nothing is measured on PDVD data yet; the milestones in §10 say which run
produces which number.

**Goal.** PDVD today ends at charge–light (Q/L) matching. We want it to
*select stopping muons that decay to a Michel electron*, and to use those
Michels for calibration: the absolute energy scale / recombination response,
and a second idea — using the **diffusion width** of a low-energy deposit to
localise it in drift. There is no beam neutrino in PDVD, so the taggers must
run on **all** matched cluster↔flash pairs, and much of the neutrino-topology
machinery SBND runs is surplus here.

**Companions.** [17_pdvd-clustering-qlmatching-chain.md](17_pdvd-clustering-qlmatching-chain.md)
(the chain as it stands), [09_pdvd-qlmatching.md](09_pdvd-qlmatching.md) and
[10_pdvd-ql-pending.md](10_pdvd-ql-pending.md) (the Q/L operating point),
[07_pdvd-tpc-geometry-fiducial.md](07_pdvd-tpc-geometry-fiducial.md)
(geometry and fiducial), and the SBND counterpart
`wcp-porting-img/sbnd/docs/sbnd-pattern-recognition.md`, which this document
deliberately mirrors section for section.

---

## Repro block

State this document was written against:

```bash
# toolkit @ 4be3f6df (branch apply-pointcloud), wcp-porting-img @ main
git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse --short HEAD
git -C /nfs/data/1/xqian/toolkit-dev/wcp-porting-img rev-parse --short HEAD
```

Every `file:line` citation below is against that toolkit commit. Note that the
working tree at time of writing carries an **uncommitted** doc-94 round-2/3
STM change; where a value differs between HEAD and the working tree this
document quotes **HEAD** and says so.

The commands that will produce this document's numbers once the design is
implemented (none has been run yet — see §10):

```bash
cd pdvd
./run_img_evt.sh   data <idx>                     # existing: per-event imaging
./run_clus_evt.sh  data -save-pctree <idx>        # M1: Q/L + pctree tarball
./run_pr_evt.sh    data -stm <idx>                # M3: PR job, STM on all pairs
./run_pr_evt.sh    data -nu  <idx>                # M5: + full PR -> Michel
# verdicts: grep "TaggerCheckSTM: cluster"  work/<RUN6>_<EVT>/wct_pr_*.log
#           grep "TaggerCheckTGM: cluster"  work/<RUN6>_<EVT>/wct_pr_*.log
# display:  work/<RUN6>_<EVT>/mabc-pr.zip  (clustering + dead + track_fit layers)
# dump:     work/<RUN6>_<EVT>/calib-pr-evt<ID>.json
```

---

## 1. Where PDVD is today

PDVD's clustering + Q/L graph is assembled in
`pdvd/wct-clustering.jsonnet:518-540`. It ends at

```
outnodes=[clus_all_tpc]
```

and stage 4's *entire* pipeline is two entries
(`cfg/pgrapher/experiment/protodunevd/clus.jsonnet:682-714`):

```
cm_old.switch_scope()      # apply the matched cluster T0 -> x_t0cor
cm.cathode_connect(...)    # stitch cathode crossers across x ~ 0
```

**Confirmed: nothing runs after Q/L matching.** There is no `steiner`, no
`fiducialutils`, no `tagger_check_*`, no `TrackFitting`, no recombination
model, no `ParticleDataSet`, no BDT scorer, no tracking visitor. A tree-wide
grep for `TrackFitting|track_fitting|Tracking` over
`cfg/pgrapher/experiment/protodunevd/` returns nothing.

One nuance worth stating, because the names mislead: PDVD *does* run
`cm.neutrino(protect_iso_band=true)` and `cm.isolated()`, but at **stage-3
drift-group scope** (`protodunevd/clus.jsonnet:527-528`), i.e. *before* the
matcher. Those are clustering-tail grouping passes, not pattern recognition.

PDVD is **not** a trimmed fork of SBND's `clus.jsonnet`. Both are built on the
shared `cfg/pgrapher/common/clus.jsonnet` vocabulary; PDVD adds a
per-drift-group stage SBND has no counterpart for, and lacks SBND's entire
`pr()` builder (`sbnd/clus.jsonnet:847-2647`, ~1800 lines), which is almost
the whole 846-vs-2676-line size gap.

One structural asymmetry to respect when adding the PR job: **PDVD's job entry
points live in the working repo, not in `cfg/`.** `pdvd/wct-clustering.jsonnet`
is resolved because `run_clus_evt.sh:274` does `cd "$PDVD_DIR"` first, while
`WIRECELL_PATH` supplies `toolkit/cfg` for the imports. PDHD is the same shape;
SBND is the odd one out. So the new PR driver belongs at
`pdvd/wct-pr-perevt.jsonnet`, not
`cfg/pgrapher/experiment/protodunevd/wct-pr-perevt.jsonnet`.

---

## 2. The three findings that shape the design

These were established by reading the code, and each one removes work that a
naive plan would have budgeted for.

### 2.1 "Run on all matched pairs" is a config setting, not a C++ change

- **STM already loops over every main cluster.** `TaggerCheckSTM::visit`
  admits any cluster carrying `Flags::main_cluster` and associates the
  companions sharing its `matched_flash_gid`
  (`clus/src/TaggerCheckSTM.cxx:499-527`). It is *not* restricted to a
  neutrino candidate — no such flag exists at that point in the chain.
- SBND narrows it to one bundle purely by configuration:
  `beam_window_only=true` with `[0.2, 2.2) us`
  (`sbnd/clus.jsonnet:2098-2100`). The **C++ defaults are `false / 0 / 0`**.
- The same gate,
  `beam_gate = beam_window_only && beam_window[1] > beam_window[0]`
  (`sbnd/clus.jsonnet:1810`), controls `steiner`, `tagger_check_tgm`,
  `tagger_check_stm`, `tagger_check_fc`, `protect_bundle` and
  `steiner_refresh`.
- **The full neutrino PR already runs per flash bundle.** `nu_per_bundle=true`
  is SBND production (`sbnd/wct-pr-perevt.jsonnet:1306`, with
  `nu_per_bundle_min_length=15`); it runs the PR chain **once per
  in-beam-window flash bundle** instead of once per event
  (`clus/src/TaggerCheckNeutrino.cxx:1874-1975`; rationale at
  `common/clus.jsonnet:546-558`).

**The trap.** It is tempting to write `beam_window_only=false`. That is
*wrong* for `TaggerCheckNeutrino`: with `beam_gate` false it takes the legacy
branch (`TaggerCheckNeutrino.cxx:1678-1691`), which picks **one arbitrary
main** (the loop overwrites `main_cluster` each iteration) and gathers
companions by `Flags::beam_flash` — a flag only `ClusteringTaggerFlagTransfer`
ever sets, and no SBND or PDVD config instantiates it. That is uBooNE's
single-bundle assumption, and on PDVD it would silently reconstruct one
cosmic per event.

**The PDVD recipe is therefore a *wide* window, not no window:**

```
beam_window_only = true
beam_window      = [<-T>, <+T>]   # spanning the whole readout, on cluster_t0
nu_per_bundle    = true
```

Every matched bundle is then "in window" and gets its own evaluation. **Zero
C++ change.** The one thing to verify on real data (M3) is that PDVD
`cluster_t0` values are populated and finite for every matched main and fall
inside the chosen window — `qlmatching.jsonnet:406-407` allows flash times of
±1 s, which is the flash *selection* range, not proof about `cluster_t0`.

The prerequisite is already met: `QLMatching`
(`match/src/QLMatching.cxx:1308-1342`) sets `main_cluster`,
`associated_cluster` and `matched_flash_gid` detector-agnostically, and PDVD
runs the same component.

### 2.2 The Michel finder already exists in the ported code

`clus/src/NeutrinoTaggerCosmic.cxx:788-1018` is, verbatim in its own comment,
a *"stopped-muon-with-Michel-electron test (flags 6-8)"*, ported from
`prototype_base/wire-cell/pid/src/NeutrinoID_cosmic_tagger.h:268-588`. It

- walks the segments at the main vertex (`segs_at_vtx(main_vertex, ...)`),
- picks the highest-energy shower starting there as `michel_ele` and records
  `michel_energy = shower->get_kine_best()`,
- separates the muon (pdg 13) and any long-muon chain,
- and fills `cosmict_7_*` / `cosmict_8_*` in `TaggerInfo`
  (`clus/inc/WireCellClus/NeutrinoTaggerInfo.h:1353-1369`).

**PDVD does not need a new Michel finder. It needs this one promoted from a
neutrino-rejection feature to the primary output.** Three checks scope that
work precisely:

1. **A readout path exists, but it is lossy.** `PrDisplayDump` writes
   `calib-pr-evt<ID>.json`, a read-only dump that is independent of the
   uBooNE-trained BDT scorers we are dropping. It emits `cosmict_flag_7`,
   `cosmict_flag_8`, `cosmict_7_filled` and `cosmict_8_filled`
   (`clus/src/PrDisplayDump.cxx:917-935`) — so the *verdict* survives the
   trim. It stops there: the sub-features (`cosmict_7_total_shower_length`,
   `cosmict_7_dQ_dx_end`, `cosmict_8_muon_length`, …) are not dumped, and
   **`michel_energy` is a local variable in the tagger that is never stored in
   `TaggerInfo` at all**. The accurate statement is: *the Michel finder exists
   and fires; its energy is not persisted.* M5 is a small, well-scoped
   persistence change (add `michel_energy` and the Michel shower's identity to
   `TaggerInfo`, then dump them), not a new algorithm.
2. **A single-track topology does get a main vertex.**
   `determine_main_vertex` handles `main_vertex_candidates.size() == 1`
   directly (`clus/src/NeutrinoVertexFinder.cxx:3964-3972`), and
   `init_first_segment` always seeds the two extremal-point vertices, so
   `segs_at_vtx(main_vertex, ...)` is not structurally empty for a stopping
   muon. Whether the mu->e kink actually yields a vertex candidate on real
   PDVD events is an **M5 acceptance test**, not something asserted here.
3. **Both `unmerge` stages drop out of the PDVD chain.** `unmerge_assoc` keys
   off `assoc_cluster_id` / `assoc_cluster_main`, which `isolated` writes only
   under `save_assoc_id` (C++ default false,
   `clus/src/clustering_isolated.cxx:584-598`) and which survive the pctree
   round trip only with MABC `save_assoc_cluster_id=true`
   (`common/clus.jsonnet:1208-1216`). SBND sets it
   (`sbnd/clus.jsonnet:254,330`); **PDVD does not**. `unmerge_bundle` undoes
   `examine_bundles`, which PDVD has deliberately commented out
   (`protodunevd/clus.jsonnet:529-538`). See §4.

Related machinery already in the tree, all C++-default OFF but **ON in SBND
production** (`clus/docs/knobs/sbnd-operating-point.md`):
`michel_stem_michel_check`, `michel_stem_muon_rescue`,
`shower_traj_michel_stem`; plus the `SegmentFlags` Michel-stem bit
(`clus/inc/WireCellClus/PRSegment.h:42-46`) and `track_owns_via_michel_stem`
in the particle-flow output (`clus/src/MultiAlgBlobClustering.cxx:1619-1651`).

### 2.3 The sign inversion — STM is a veto in SBND and the signal in PDVD

In SBND a false-positive STM throws away a neutrino, so the tuning effort has
gone into **suppressing** STM firing. In PDVD, STM-tagged **is the signal**:
efficiency and purity swap roles, and the operating point must be re-derived
rather than inherited. Three concrete consequences:

| SBND production (HEAD) | PDVD | why |
|---|---|---|
| `stm_accept_guards`, `stm_proton_muon_guard`, `stm_cathode_guard`, `stm_second_track_guard`, `stm_deficit_guard`, `stm_vertex_kink_guard`, `stm_vertex_hadron_guard` all **true** (`sbnd/clus.jsonnet:1073,1078,1083,1096,1103,1104,1137`) | start **OFF**, re-derive individually | each exists to abstain from an STM call so a neutrino survives; here each one rejects a good stopping muon |
| `nu_skip_cosmic=true`, `nu_skip_cosmic_bundle=true` | **false** | these skip STM/TGM-tagged mains from the neutrino PR — precisely the objects PDVD wants pushed *into* PR to reach the Michel |
| `stm_michel_res_cm=6.5` via `stm_d66_cuts` (`sbnd/clus.jsonnet:1169`) | re-derive | see below |

(`stm_descent_guard` is already false at HEAD, `sbnd/clus.jsonnet:1114`;
`stm_entry_rise_guard` is false at HEAD, `:1152`, and is under active revision
in the working tree — do not treat its working-tree parameters as shipped.)

**The STM Michel window is PDVD's signal-efficiency knob.** `eval_stm_core`
(`clus/src/TaggerCheckSTM.cxx:2738-2765`) rejects an STM call when the
*residual* past the muon's Bragg fit is both long and dense. It is a
seven-clause disjunction over `res_length` in {2, 4, `michel_res_cut`, 10, 16,
20 cm} and `ave_res_dQ_dx / mip_dqdx` in {1.2, 1.4, 1.45, 1.7, 1.85, 4.5}.
Read carefully, it does **not** reject Michels: the thresholds are placed so
that a Michel-sized residual survives while a second track or a hadronic mess
does not. Those seven clauses therefore *are* the mu+Michel acceptance window.
They were tuned on uBooNE and retuned on SBND (the `stm_d66_cuts` family), and
they must be re-derived for PDVD against hand-scanned stopping muons.

---

## 3. The intermediate file (Milestone 1)

The PR job needs the post-Q/L point-cloud tree persisted. PDVD already has the
save point, inert: a `TensorFileSink` in `dump_mode` at the tail of the
all-TPC stage (`protodunevd/clus.jsonnet:811-819`, writing
`trash-all-apa.tar.gz`) — exactly the node SBND turned into its real save
(SBND doc §3).

Design, knob default-OFF so today's outputs stay byte-identical:

1. `protodunevd/clus.jsonnet`, `clus_all_tpc(...)`: new argument
   `tensor_outname=''`. Empty (default) keeps `outname: 'trash-all-apa.tar.gz',
   dump_mode: true` — a no-op. Non-empty sets
   `outname: tensor_outname, prefix: 'clustering_', dump_mode: false`.
2. `pdvd/wct-clustering.jsonnet` threads a TLA; `run_clus_evt.sh` gains
   `-save-pctree`, writing `work/<RUN6>_<EVT>/pctree-evt<ID>.tar.gz`.

Container: the Wire-Cell TensorDM tar stream (`TensorFileSink` /
`TensorFileSource`, package `sio`), the same representation already flowing in
memory from QLMatching into the all-TPC MABC — so persisting it adds no new
conversion code.

**Round-trip inventory** (what must survive; all of it is on the tree at the
save point already):

| item | where it lives | consumer |
|---|---|---|
| live cluster/blob tree with `3d` sampled points | `/live` tree, per-blob local PCs | everything |
| dead (2-view) tree | `/dead` tree | `FiducialUtils::inside_dead_region`, STM |
| `x_t0cor` corrected coordinates | per-point arrays from `switch_scope` | PR runs in corrected scope |
| `cluster_t0` (ns), `matched_flash_gid` | `cluster_scalar` PC, written by QLMatching | the wide beam window, `nu_per_bundle` |
| `flag_main_cluster`, `flag_associated_cluster` | `cluster_scalar` | STM / neutrino main-cluster selection |
| `opflash` PC (flash x channel: gid, time, ch, pe) | root-node PC | Bee `op` layer, any light-aware tagger |
| run / subrun / event | tensor-set `ident` + job TLAs | bookkeeping, Bee labels |

Runtime-only state that does **not** persist and must be re-established by the
PR job: the active scope (which coordinate arrays are "default") and attached
utility objects (`FiducialUtils`, Steiner graphs). That is why `switch_scope`
and `fiducialutils` both reappear at the head of the PR pipeline.

Gate: the Q/L outputs (`mabc-*.zip`) must be byte-identical with the knob off,
compared by archive **member content hash** (`abtest/hash_archive.py`), never
by `md5sum` of the tarball.

---

## 4. The PDVD PR job — the trimmed chain (Milestones 2–3)

A new `pdvd/wct-pr-perevt.jsonnet`, forked by duplication from
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` (3636 lines). The SBND
file stays byte-for-byte untouched: it has live consumers.

SBND's production PR pipeline is 15 stages
(`sbnd_xin/run_pr_chain_batch.sh:154`):

```
switch_scope, unmerge_bundle, unmerge_assoc, steiner, fiducialutils,
tagger_check_tgm, tagger_check_stm, tagger_check_fc, protect_bundle,
steiner_refresh, tagger_check_neutrino, numu_bdt_scorer, nue_bdt_scorer,
tracking_visitor, tagger_output
```

### Keep / drop, with a dependency reason per row

| stage | PDVD | reason |
|---|---|---|
| `switch_scope` | **keep** | re-applies the flash T0; scope is not persisted in the tarball |
| `unmerge_bundle` | **drop** | undoes `examine_bundles`, which PDVD does not run (`protodunevd/clus.jsonnet:529-538`, commented out) |
| `unmerge_assoc` | **drop** | needs `assoc_cluster_id` / `assoc_cluster_main`, which PDVD's stage-3 `isolated` does not write (`save_assoc_id` unset) |
| `steiner` (retiler `improve_cluster_2`) | **keep** | STM's endpoint and path finding need the Steiner skeleton. Needs `require_beam_flash=false` (SBND M3 change 2) and PDVD per-(APA,face) samplers |
| `fiducialutils` | **keep** | every tagger silently returns false without it (`TaggerCheckSTM.cxx:3367-3369`) |
| `tagger_check_tgm` | **keep** | in an all-cosmic detector the through-going muon is the dominant background to STM; STM also skips TGM-tagged mains (`TaggerCheckSTM.cxx:550-553`) |
| `tagger_check_stm` | **keep — this is the signal** | guards off (§2.3), wide window (§2.1) |
| `tagger_check_fc` | **keep** | cheap, and the natural containment test for the Michel shower |
| `protect_bundle` + `steiner_refresh` | **keep** | ordering is load-bearing: refresh must follow with `replace=false`, else the STM-stage `GraphAlgorithms` dangle (`sbnd/clus.jsonnet:1989-1994`) |
| `tagger_check_neutrino` | **keep** | this is where trajectory + dQ/dx fitting, track/shower separation, PID **and the Michel finder** (§2.2) live |
| DL / SCN vertex (`dl_weights`) | **drop** (`dl_weights=''`) | there is no neutrino vertex to find; the only vertex in this topology is a track end the trajectory fit already gives. The weights are uBooNE-trained, and the DL path is excluded from our A/B gates because it is not bit-stable. Note it **is** ON in SBND production (`sbnd/clus.jsonnet:880`) despite the stale claim at `wct-pr-perevt.jsonnet:9-10` that it stays off |
| `numu_bdt_scorer`, `nue_bdt_scorer` | **drop** | uBooNE-trained BDTs over neutrino topologies; nothing in a mu->e decay reads their scores |
| single-photon / pi0 / NC machinery inside `tagger_check_neutrino` | **measure, then drop** | neutrino-topology only, but they are sub-steps of one component — retire on a firing census, not by assertion |
| `tracking_visitor`, `tagger_output` | **keep (fork)** | §9 |
| `pr_display` (`PrDisplayDump`) | **keep** | the Michel readout path (§2.2) |

The trim is expressed as configuration — `pipeline_names`, `dl_weights=''`,
knobs off — so every "drop" row costs one measured firing census on real PDVD
events to confirm, not a code deletion. Zero fires is not by itself proof a
stage is dead; distinguish "never applicable" from "pre-empted by an earlier
stage" before retiring anything.

### Prerequisites the PR job must assemble

Each of these is a component PDVD has never instantiated:

- **`ParticleDataSet`** — the dQ/dx and range reference tables (§7c).
- **A recombination model** — PDVD instantiates none today (§7c).
- **A track-fitting parameter file** — `pdvd_track_fitting.json` (§7b).
- **A fiducial** — `BoxFiducial` spanning both drift volumes plus margins,
  from [07_pdvd-tpc-geometry-fiducial.md](07_pdvd-tpc-geometry-fiducial.md)
  (§6). Do not invent one.
- **Steiner retiler samplers** per (APA, face) — 16 of them for PDVD.
- `WireCellRoot` must be loaded by the job even though PDVD has no SCE map, if
  the fork keeps SBND's `DetectorVolumes.uses` shape.

---

## 5. Michel identification — what the chain must emit

Operational definition of the object this chain produces, one row per matched
bundle that passes:

- the STM-tagged muon: cluster id, matched flash gid, `cluster_t0`, length,
  entry point, **stop point**;
- its fitted trajectory and dQ/dx profile vs residual range;
- the attached EM object: the shower whose start segment sits at the main
  vertex, its energy (`michel_energy`), its direction relative to the muon,
  and its distance from the muon stop point;
- containment verdicts (`Flags::FC`, and the fiducial distance of the stop);
- the muon–electron time separation where the topology resolves it;
- any additional isolated low-energy deposits near the stop (the "dots").

The route is §2.2: reuse `NeutrinoTaggerCosmic`'s flags-6-8 block; the scoped
work is **persisting** `michel_energy` and the Michel shower's identity into
`TaggerInfo` and emitting them from `PrDisplayDump`, plus a PDVD
`tagger_output` fork if a ROOT tree is wanted alongside the calib JSON.

Acceptance test for M5, stated before the run so it cannot be moved
afterwards: on a hand-scanned set of PDVD stopping-muon candidates, the
flags-6-8 block must *fire* (`cosmict_7_filled != 0`) on a stated fraction,
and the recovered `michel_energy` distribution must have support up to the
Michel endpoint. If the block does not fire, the fallback is a dedicated
extractor reading STM's stop point plus the nearest shower — but that fallback
should not be built before the measurement says it is needed.

---

## 6. Geometry — VD breaks SBND's inherited assumptions

Numbers from [07_pdvd-tpc-geometry-fiducial.md](07_pdvd-tpc-geometry-fiducial.md)
and `protodunevd/params.jsonnet`:

| | PDVD | SBND |
|---|---|---|
| anodes | **8**, two-sided -> **16** (anode, face) volumes | 2, one face each |
| drift volumes | **2** opposed: anodes 0–3 bottom (drift +x), 4–7 top (drift −x) | 2, cathode at x = 0 |
| W (collection) plane | \|x\| = **341.55 cm** | \|x\| = 202.05 cm |
| cathode surface | \|x\| = **3.0 cm** (`cpa_thick` 60 mm) | \|x\| = 0.45 cm |
| **drift distance** | **338.55 cm** (`cpa_plane`) | ~201 cm |
| pitch | U/V **7.65 mm**, W **5.10 mm** (`qlmatch/04:88`) | 3 mm all planes |
| response plane | 18.1 cm from collection | 10 cm |
| tick / nticks | 0.5 us / 6000 (data), 6400 (sim), 10000 (production window) | 0.5 us / 3427 |

Three consequences for the taggers:

1. **Cosmics enter from the top**, so the entry/exit asymmetry SBND's fiducial
   logic encodes does not transfer. The STM entry-face reasoning must be
   re-derived on the VD geometry.
2. **The drift is 1.7x longer than SBND's**, which is what makes §8's
   diffusion measurement plausible at all, and also what makes attenuation
   (electron lifetime) a first-order correction rather than a nuisance.
3. **The cathode is a central slab, not an edge**, so a "cathode crosser" is a
   single physical track split across two drift volumes — PDVD already handles
   this with `cathode_connect` before the PR job sees it.

Known 2-TPC risk points inside `TaggerCheckSTM`, recorded during the SBND port
and equally applicable here (audit these if PDVD results look wrong):

- `dist_to_anode` falls back to `|x|` for points outside all volumes
  ("preserves UBooNE behaviour") — wrong side for a −x-drift volume's corners.
- Several kink-detection helpers hard-code `drift_dir_abs(1,0,0)`; fine while
  drift is parallel to x, but sign-blind near the cathode.
- `shorted_y_w_range` is a uBooNE shorted-wire hack — leave unset.

---

## 7. Numbers to calibrate and set

Every row carries a provenance: **inherited** (a default nobody chose for
PDVD), **PDVD-measured**, or **unmeasured**.

### 7a. LAr transport and field — the blocking item

**Four different drift speeds are live in PDVD today:**

| value | where | provenance |
|---|---|---|
| **1.568 mm/us** | `protodunevd/params.jsonnet:131`, `protodunevd/clus.jsonnet:44` | PDVD-measured (A↔C crossers, rescaled when the cathode moved 2.54 -> 3.0 cm) |
| **1.473 mm/us** | `protodunevd/simparams.jsonnet:13` — **used by imaging** via `img.jsonnet:4` | sim value |
| **1.48073 mm/us** | `run_clus_evt.sh:793-794` (`PDVD_DRIFT_SPEED_BOT/TOP_MMUS` defaults) | PDVD-measured, later round (W-decon A↔C crosser, run 039252 evt 298651) |
| 1.6 mm/us | `cfg/pgrapher/common/params.jsonnet:29` | inherited base default |

A dQ/dx calibration cannot start until one is chosen, because `dx` depends on
it directly and every diffusion sigma in §8 scales with it. **This is a
stop-and-decide item, not something to pick silently.**

The four are not equally arbitrary — they have different consumers, and one of
them is load-bearing for §7b:

- **The grouping value is the one that propagates into the fit's geometry.**
  `TrackFitting::BuildGeometry` reads `m_grouping->get_drift_speed()`
  (`clus/src/TrackFitting.cxx:618`) and builds `slope_t` and the time-tick
  width from it (`:653`). PDVD sets it **per crate** —
  `drift_speed_bot` / `drift_speed_top` (`protodunevd/clus.jsonnet:115,135`),
  defaulting to the 1.568 literal at `:44` but overridden by the runner to
  1.48073 for both crates (`run_clus_evt.sh:793-794`). This is the value §7b
  must match.
- **1.473 is the sim value** and reaches imaging through
  `img.jsonnet:4` -> `simparams.jsonnet:13`. It governs how the frames the fit
  reads were *produced*, which is a separate question from how the fit
  converts ticks to length.
- **1.6 is the inherited base default** and nothing chooses it.

Transport coefficients are equally split:

| quantity | PDVD reco | PDVD sim | provenance |
|---|---|---|---|
| `DL` | 7.2 cm²/s (`common/params.jsonnet:23`) | **4.0** (`simparams.jsonnet:10`) | reco = inherited base default; sim = SBND's pair, copied |
| `DT` | 12.0 cm²/s (`common/params.jsonnet:25`) | **8.8** (`simparams.jsonnet:12`) | same |
| lifetime | 8 ms (`common/params.jsonnet:27`) | 1000 ms (`simparams.jsonnet:8`) | reco inherited; sim effectively infinite |

Neither pair was measured for PDVD. Note that SBND's own reco `DL`/`DT` do not
come from `params.jsonnet` at all — they come from the runtime-loaded
`sbnd_track_fitting.json` (§7b), which is a separate knob that must be kept in
step with the sim values by hand.

**No E-field and no temperature value is configured anywhere in PDVD.**
`protodunevd/funcs.jsonnet:136-190` carries a complete Walkowiak
drift-velocity parameterisation that nothing calls; drift speed is a
data-calibrated literal instead. The nominal field is 500 V/cm (GDML
`volTPCActive` aux), which matters because the dQ/dx templates (§7c) are
field-specific.

### 7b. Reco-side track-fitting constants

`TrackFitting` is a library class, not a component
(`clus/inc/WireCellClus/TrackFitting.h`, `clus/src/TrackFitting.cxx`,
~9600 lines), instantiated by `TaggerCheckSTM` and `TaggerCheckNeutrino`. Its
~50 numeric parameters live in one struct
(`TrackFitting::Parameters`, `TrackFitting.h:36-193`) and are overridden
one-by-one from a JSON file named by the `trackfitting_config_file` knob.

Two properties of that file that determine how it must be validated:

- **It is read at runtime and never enters the compiled jsonnet.** A
  byte-identical compiled config therefore does **not** mean the fit is
  unchanged. Any A/B on these numbers needs an output-level gate, not a
  config diff.
- **Both taggers resolve the path through `WIRECELL_PATH`**
  (`Persist::resolve` at `TaggerCheckSTM.cxx:984` and
  `TaggerCheckNeutrino.cxx:3901`), so a relative name works. The
  `_comment_canonical` in `sbnd_track_fitting.json` claiming callers "must
  pass an ABSOLUTE path" is stale — worth fixing there when someone next
  touches that file.

**Consistency requirement (the key constraint): the fit's smearing function
must match the software filter that signal processing imprinted on the data.**
The uBooNE constants decode exactly to uBooNE's SP filters; SBND's were
re-derived from SBND's. PDVD's must be re-derived from PDVD's
(`protodunevd/sp-filters.jsonnet:94,111-114`):

| filter | uBooNE | SBND | **PDVD** |
|---|---|---|---|
| `Gaus_wide` sigma | 0.111408 MHz | 0.10 MHz | **0.12 MHz** (`_b` and `_t`) |
| `Wire_ind` | (1/sqrt(pi))·1.4 | (1/sqrt(pi))·1.05 | **(1/sqrt(pi))·5.0** |
| `Wire_col` | (1/sqrt(pi))·3.0 | (1/sqrt(pi))·3.60 | **(1/sqrt(pi))·10.0** |

Applying the SBND derivation (SBND doc §6.2) with PDVD's filters, PDVD's
pitches (U/V 7.65 mm, W 5.10 mm) and v = 1.568 mm/us:

| key | formula | SBND | **PDVD candidate** |
|---|---|---|---|
| `add_sigma_L` | [1/(2*pi*sigma_Gaus_wide)] * v_drift | 2.4876 | 1/(2*pi*0.12 MHz) = 1.32629 us; x 1.568 = **2.0796** |
| `ind_sigma_u_T` | [(1/sqrt(pi))/Wire_ind] * pitch_U * 0.3 | 0.48359 | (0.564190/5.0) x 7.65 x 0.3 = **0.2590** |
| `ind_sigma_v_T` | ... * pitch_V * 0.5 | 0.80599 | (0.564190/5.0) x 7.65 x 0.5 = **0.4316** |
| `col_sigma_w_T` | [(1/sqrt(pi))/Wire_col] * pitch_W * 0.2 | 0.09403 | (0.564190/10.0) x 5.10 x 0.2 = **0.0575** |
| `DL` | physical, not filter-derived | 4.0e-07 | **unmeasured** |
| `DT` | physical, not filter-derived | 8.8e-07 | **unmeasured** |

**The formula is validated, not assumed.** Substituting SBND's own filters
(`Gaus_wide` 0.10 MHz, `Wire_ind` 1.05, `Wire_col` 3.60), pitch 3 mm and
v = 1.563 mm/us into the same four expressions reproduces the shipped
`sbnd_track_fitting.json` values exactly — 2.4876 / 0.48359 / 0.80599 /
0.09403 to five decimals. So the PDVD column is a substitution into a
checked formula, not a guess; what remains uncertain for PDVD is the
*inputs* (which drift speed, and the empirical trailing factors), not the
derivation.

Caveats to carry with that table: the trailing 0.2 / 0.3 / 0.5 factors are
empirical uBooNE tunings on top of the filter width and must be re-checked
against PDVD fit residuals; and if PDVD's SP filters are ever retuned this
file must be re-derived — note the dependency in any SP retune.

**Which drift speed goes into `add_sigma_L` is determined, not free.** The
filter width is a *time* quantity, independent of drift speed; the `v` in the
formula is purely the unit conversion into the length units `diff_sigma_L`
uses, and the result is then divided by a time-tick width built from
`m_grouping->get_drift_speed()` (`TrackFitting.cxx:618,653`). So `v` must be
**the drift speed the PR job configures on the grouping**, or the filter term
is mis-scaled against the tick axis it is divided by. SBND is the worked
example: its grouping is configured at 1.563 (`sbnd/clus.jsonnet:22,143`) and
its shipped `add_sigma_L` = 2.4876 is exactly 1/(2*pi*0.10)*1.563. (That this
coincides with SBND's *sim* value is incidental; the operative fact is that it
matches the grouping config.) For PDVD that gives **2.0796** if the PR job
takes the `protodunevd/clus.jsonnet:44` default of 1.568, or **1.964** if it
inherits the runner's 1.48073. Pick whichever the PR job actually configures.

One structural hazard specific to VD: PDVD sets drift speed **per crate**
(`drift_speed_bot` / `drift_speed_top`), while `pdvd_track_fitting.json` is a
single global file. Today both runner defaults are 1.48073 so one
`add_sigma_L` suffices — but if the two drift stacks are ever calibrated
separately, `add_sigma_L` cannot follow the split and the file becomes wrong
for one crate. Flag it if that divergence is ever proposed.

Other reco-side rows PDVD must set rather than inherit:

| knob | C++ default | SBND | note |
|---|---|---|---|
| `mip_dqdx` | 50000 e/cm (MicroBooNE) | 56000 | drives the STM flat template *and* the PR chain's `cal_4mom` amplitude |
| `mip_dqdx_median` | 43000 e/cm | 48000 (`sbnd/clus.jsonnet:1244`) | the shower-topology and PID median threshold |
| muon median-dQ/dx-vs-length envelope `c0 + c1*(pivot/L)^power` | {0.8866, 0.9533, 18 cm, 0.4234} (`NeutrinoPatternBase.h:2949`) | [0.8826, 1.0587, 18, 0.4745] (`sbnd/wct-pr-perevt.jsonnet:804`) | used at nine tagger sites |
| `div_sigma` | 0.6 cm | 6.0 (internal units) | charge-division Gaussian width |
| `dis_end_point_ext` | 0.45 cm | — | dQ/dx endpoint charge-collection radius |
| `lambda` | 0.0005 | 0.0005 | dQ/dx fit regularisation; x8/5 in multi-track, x0.01 when unregularised |

### 7c. dQ/dx templates, recombination, and the charge scale

**Templates.** The prototype read `g_muon` / `g_proton` / `g_electron` from
`input_data_files/stopping_ave_dQ_dx_v2.root`. The toolkit reads **no data
file**: the curves are inlined `LinterpFunction` tables aggregated by a
`ParticleDataSet` component. SBND's are in
`cfg/pgrapher/experiment/sbnd/particle_dataset.jsonnet` — five `*DeDx` tables
(dQ/dx in e/cm vs residual range, `start: 0.5 cm`, `step: 1.0 cm`, 60 points)
and five `*Range` tables (range cm -> KE MeV).

The critical provenance note, from that file's own header
(`particle_dataset.jsonnet:12-36`): **the five `*DeDx` tables are NOT
detector-agnostic.** They are dQ/dx after Modified-Box recombination *at a
specific drift field*; only the `*Range` tables are field-independent. SBND's
were regenerated at E = 0.5 kV/cm (they previously came verbatim from
MicroBooNE's 0.273 kV/cm), via

```
energy_loss/pion_travel/convert_field.C   # root -l -b -q 'convert_field.C(0.5, ...)'
energy_loss/docs/emit_jsonnet_dedx.py
```

with `dQ/dx = ln(alpha + beta'*dE/dx) / (beta' * W_ion) * 0.85`, `alpha = 0.93`,
`beta = 0.212`, `rho = 1.38 g/cm^3`, `W_ion = 23.6 eV`, `beta' = beta/(rho*E)`.
PDVD's nominal field is also 500 V/cm, so **SBND's tables are a defensible
starting point for PDVD** — that is an inheritance with a stated reason, not
an accident. The undocumented `0.85` scale is deliberately retained on the
SBND side pending a real charge calibration; PDVD should not silently inherit
it once a PDVD charge calibration exists.

**Recombination.** PDVD instantiates **no** recombination model today. The
available classes (`gen/inc/WireCellGen/RecombinationModels.h`,
`PracticalRecombinationModels.h`):

| class | constants |
|---|---|
| `BirksRecombination` | E 500 V/cm, A3t 0.8, k3t 0.0486, rho 1.396, Wi 23.6 eV |
| `BoxRecombination` (Modified Box) | E 500 V/cm, A 0.930, B 0.212, rho 1.396, Wi 23.6 eV — **WCT units** |
| `PracticalBoxRecombination` | same arithmetic in **practical units** (kV/cm, g/cm^2/MeV) |
| `PowerBoxRecombination` | A 0.93, k 0.282371, p 1.362179, **C 0.855175**, pivot 2.1 MeV/cm, Wi 23.6e-6, dedx_max 77.0; R = ln(A+u)/u, u = k*(dEdx/pivot)^p |

`RecombinationModels.h:49-64` warns explicitly that the WCT-unit and
practical-unit variants are **not** interchangeable — the mismatch is wrong by
`units::cm/units::MeV = 10` in the quenching term and moved a MIP from 2.10 to
1.37 MeV/cm on 23 of 35 uBooNE events. Pick the Practical variant when the
parameters are quoted in practical units.

SBND ships both `PracticalBoxRecombination` (A 1.0, B 0.255, E 0.5, rho 1.38,
Wi 23.6e-6) and a `PowerBoxRecombination` **fitted to SBND stopping-track
dQ/dx vs residual range** (`sbnd/clus.jsonnet:1835-1848`; canonical parameters
in `sbnd_xin/nusel_display/stm_ref_dqdx.json`), selected by
`use_power_recomb=true`.

**Name the circularity, because it is the plan:** the STM sample this chain is
built to select is *exactly* the sample that fits PDVD's own power-box
recombination. M7 closes that loop — the first pass runs on SBND's
recombination and templates, and the stopping muons it finds supply the fit
for PDVD's own.

**Charge-scale constants.** The charge->energy conversion is
`overall / recom_factor / fudge_factor * w_value` (`NeutrinoEnergyReco.cxx:188`):

| knob | C++ default | SBND |
|---|---|---|
| `kine_recom_factor` | 0.7 | **0.87** |
| `kine_proton_recom_factor` | 0.35 | **0.51** |
| `kine_shower_recom_factor` | 0.5 | **0.58** |
| `kine_shower_fudge_factor` (the EM charge scale) | 0.8 | **0.86** |
| `fudge_factor` (track) | 0.95 | 0.95 |
| `w_value` | 23.6 eV | 23.6 eV |

Naming discipline, because this is easy to get wrong: there is **no** constant
called a "flip fudge factor", and four *distinct* 0.85–0.86 numbers exist that
must not be conflated —

1. `kine_shower_fudge_factor` = 0.86, the EM charge scale
   (`NeutrinoPatternBase.h:46`, `sbnd/wct-pr-perevt.jsonnet:777`);
2. SBND's Q/L `data_qtol` = 0.86 (`sbnd/qlmatching.jsonnet:71`), a
   light-prediction scale applied only on data;
3. the dQ/dx table generator's undocumented 0.85
   (`particle_dataset.jsonnet:27`);
4. `PowerBoxRecombination`'s normalisation `C = 0.855175`.

**The headline for the whole of §7:** `clus/docs/knobs/README.md` records that
**192 of the knobs present in SBND's production config differ from their C++
default.** So a PDVD chain that simply instantiates the components and
inherits the C++ defaults inherits **uBooNE**, not SBND. PDVD needs its own
operating-point delta file, generated the same way:

```bash
python3 clus/docs/knobs/knob_delta.py \
    clus/test/doctest_clus_knob_defaults.cxx  <a compiled PDVD PR job>
```

(the second argument must be a *compiled job*, not an imported module — a
`clus.jsonnet` compiles to `{}` because its fields are hidden).

### 7d. The light side

State the shape correctly, because the usual vocabulary does not apply:
**there is no PE-per-MeV, QE, scintillation yield or prompt/late fraction
anywhere in `match/`.** The entire light-yield chain is one product
(`match/src/QLMatching.cxx:1911-1914`):

```
pred_PE[det] = q * QtoL * direct_visibility * VUVEfficiency[det]
             + q * QtoL * reflected_visibility * VISEfficiency[det]
```

PDVD's values are already calibrated
(`cfg/pgrapher/experiment/protodunevd/qlmatching.jsonnet`):

| quantity | PDVD | note |
|---|---|---|
| `QtoL` | **0.094** (`:316`) | PDVD-measured from 80 beam-flash gold pairs |
| `doReflectedLight` | **false** (`:317`), `VISEfficiency` all zero (`:255`) | the library visibility is total arrival |
| per-type efficiency scales | cathode XA x10.116, membrane XA x1.655, PMT x0.352 (`:234-236`) | PDVD-measured from crosser anchors |
| light model | library, Ar/128 nm, `pdvd-photlib-vis-v5-128nm.json` (`:330`) | |

One recorded caveat matters directly here: the library + official-efficiency
model **over-predicts by ~10x as a single global normalisation**
(`qlmatching.jsonnet:307-308`), attributed to a units/recombination/SPE-scale
product. That global normalisation is degenerate with the charge scale a
Michel energy calibration is trying to measure — so the two must not be tuned
against each other without an external anchor.

Also open on the light side: the pending items in
[10_pdvd-ql-pending.md](10_pdvd-ql-pending.md) §3, and the **+0.75 us
self-trigger offset** found in [23_pdvd-light-timing-check.md](23_pdvd-light-timing-check.md),
whose correction is owner-gated because it would move reconstructed light
times unconditionally.

### 7e. Michel-specific numbers

Three numbers this analysis needs that are not in the toolkit at all:

- the **Michel spectrum endpoint** (~52.8 MeV), the standard candle for the
  absolute energy scale;
- the **mu- capture-vs-decay fraction in argon**, which sets how many stopping
  mu- produce a Michel at all and therefore the efficiency denominator;
- the **effective muon lifetime in argon** (mu+ free lifetime vs the shortened
  mu- lifetime under capture), for the decay-time exponential fit.

**Cite a published source in this document for each before using it.** They
are deliberately left blank here rather than filled from memory.

---

## 8. Diffusion-based drift localisation for low-energy deposits

The owner's second use for this sample: for a low-energy deposit with no
independent t0, measure its **diffusion width** and infer how far it drifted.

The fitter already carries exactly this model, which is convenient — the study
measures the same quantity the fit predicts, not a new formalism
(`clus/src/TrackFitting.cxx:6859-6867`):

```
drift_time   = max(min_drift_time (50 us), drift_distance / drift_speed)
diff_sigma_L = sqrt(2 * DL * drift_time)
diff_sigma_T = sqrt(2 * DT * drift_time)
sigma_L      = hypot(diff_sigma_L, add_sigma_L)      / time_tick_width
sigma_T_{u,v,w} = hypot(diff_sigma_T, {ind,ind,col}_sigma_T) / pitch_{u,v,w}
```

So the observable is always diffusion added **in quadrature** with an
instrumental term set by the SP filter. The design question is which view to
use, and at PDVD the two behave oppositely.

Substituting the PDVD geometry: drift distance 338.55 cm at v = 1.568 mm/us
gives t = 338.55 / 0.1568 = **2159 us**.

| | diffusion sigma at full drift | instrumental sigma (†) | sampling |
|---|---|---|---|
| **longitudinal** | sqrt(2 x 4.0 x 2.159e-3) = **1.31 mm** (DL = 4.0) ; **1.76 mm** (DL = 7.2) | `add_sigma_L` ~ **2.08 mm** | 0.784 mm per 0.5 us tick |
| **transverse (W)** | sqrt(2 x 8.8 x 2.159e-3) = **1.95 mm** (DT = 8.8) ; **2.28 mm** (DT = 12) | `col_sigma_w_T` ~ **0.058 mm** | 5.10 mm pitch |

(†) Both instrumental values are the §7b **derived candidates**, not measured
PDVD numbers. The 34x transverse-dominance claim below inherits that caveat and
must be re-checked once PDVD fit residuals exist.

- **Longitudinal: well sampled, but instrument-dominated.** The observed width
  runs from 2.08 mm at the anode to hypot(2.08, 1.31) = 2.46 mm at the cathode
  (DL = 4.0), i.e. 2.65 -> 3.14 ticks — a swing of about **half a tick across
  the full drift**. It is measurable on an ensemble, marginal per-event, and
  it requires the filter width to be known to roughly 1%.
- **Transverse: diffusion-dominated, but under-sampled.** The instrumental
  term is 34x smaller than the diffusion term, so the *relative* signal is
  far larger — the observed width changes by a factor of ~34 across the drift
  rather than 18%. But 1.95 mm is only **0.38 of the W pitch** (0.25 of the
  7.65 mm U/V pitch), so the charge lands on one or two wires and the
  observable is adjacent-wire charge *sharing*, not a fitted per-hit width.
  VD's coarse CRP pitch is what costs here; SBND's 3 mm pitch would be kinder.

**Sensitivity to §7a.** All of the above uses v = 1.568 mm/us. At the
production runner value 1.48073 mm/us the drift time becomes 2286 us, every
diffusion sigma rises ~3%, and `add_sigma_L` falls ~6% (it scales with v).
Whichever drift speed §7a settles on must be stated wherever these numbers are
quoted.

**The calibration route.** Either way the natural calibrator is **Michels
whose t0 is already fixed by the flash match**. They supply (drift distance,
observed width) pairs across the whole drift, which determines the
instrumental intercept and D simultaneously — the intercept at zero drift is
the filter term, the slope in sqrt(t) is D. Recommend cross-checking on the
*tight* filter or a dedicated wide-band deconvolution, since `Gaus_wide` is
what sets `add_sigma_L`; and note the `min_drift_time = 50 us` floor in the
model above, harmless here (it corresponds to sigma_diff ~ 0.2 mm) but worth
knowing when interpreting near-anode points.

This is what makes `DL` and `DT` **first-class PDVD deliverables** rather than
fit parameters — and recall from §7a that PDVD reco and PDVD sim currently
disagree on both.

---

## 9. Auxiliary tools and the validation chain

### 9.1 Magnify-tracking for PDVD

Today PDVD has only the *waveform* Magnify: `protodunevd/magnify-sinks.jsonnet`
(8 pipe flavours, all `MagnifySink`) driven by `pdvd/wct-sp-to-magnify.jsonnet`
and `pdvd/run_sp_to_magnify_evt.sh`, viewed with `Magnify-PDVD`.

Track overlay is **SBND-only**:

| piece | where |
|---|---|
| STM-stage writer | `root/src/SbndMagnifyTrackingVisitor.cxx` -> `tracking-stm.root`, from the `save_stm_fit` point clouds |
| PR-stage writer | `root/src/SbndPrMagnifyTrackingVisitor.cxx` -> `tracking-pr.root`, from the PR `TrackFitting` slot + PRGraph |
| converter | `root/apps/wire-cell-sbnd-magnify-tracking-convert.cxx` |
| viewer | `Magnify-tracking-SBND` (ROOT GUI; `T_rec_charge`, `T_proj_data`, `T_bad_ch`, `Trun`) |
| format reference | `clus/docs/magnify_tracking_output.md` |

**The causal point:** a tracking visitor writes *fitted track segments*, and
only the PR stage produces those. So "Magnify-tracking for PDVD" is downstream
of §4 — it is not a viewer or config task that can be done first.

**Fork only if needed.** Both SBND visitors already derive the global channel
scheme from the configured anodes at visit time
(`global = plane_base[p] + apa * nch[p] + wire`, with `nch[p]` taken from the
anodes), and the PR-stage one uses the per-point `(apa, face)` recorded in
`PR::Fit::paf`. That is more general than SBND needs. So **first test whether
a PDVD configuration alone suffices**; fork a `PdvdMagnifyTrackingVisitor`
only if PDVD's 16 (anode, face) volumes break it. Duplicating is the right
move if a fork is needed — do not extract a shared helper out of the SBND
file, which has live consumers.

Also relevant: in magnify ROOT files `h*_orig0` is **pre**-NF raw ADC and
`h*_raw0` is **post**-NF. Never read "raw" as unfiltered.

### 9.2 The rest of the validation chain

- **Bee.** `mabc-pr.zip` gains `track_fit`, `shower_track` and `vertices`
  layers on top of `clustering` and `dead`; `save_stm_fit` additionally
  produces a Bee `stm_fit` layer showing the per-pass STM trajectory fits
  (including rejected passes) at no extra cost. Existing `run_bee_*` scripts
  package these.
- **Existing PDVD viewers to reuse**, not rebuild: `ql_display` and `ql_scan`
  (label/decision infrastructure with committed decision sets), `img_plot`
  (Bokeh imaging display), `pd_plot`, `nf_plot`, `sp_plot`, `drift_calib`.
- **A PDVD `pr_scan` hand-scan sheet** for STM / Michel labels. One rule from
  hard experience: keep it **blind** — never print the algorithm's verdict on
  the image being judged, or the agreement number is circular.

### 9.3 Gates

- **M1 byte-identity.** PDVD Q/L outputs must be unchanged with
  `tensor_outname` empty, compared by archive member content hash
  (`abtest/hash_archive.py`). Never `md5sum`/`cmp` on a tarball or zip — they
  embed timestamps and will report a regression that does not exist.
- **Shared components gate every detector that binds them.** Anything touched
  in `clus/`, `match/` or `cfg/pgrapher/common/` must keep SBND *and* the
  uBooNE `qlport` reference byte-identical, not just PDVD.
- **Freshness.** Runtime loads plugins from `local/lib`, not `build/`. After
  `wcbuild`, confirm the library mtime is newer than the last source edit
  before believing any A/B — a stale library makes a comparison vacuously
  pass.
- **Data self-consistency stands in for truth** (there is no MC in the first
  pass): a Bragg peak present at the stop; the mu decay-time distribution
  exponential with the argon lifetime; the Michel energy spectrum ending at
  the endpoint; dQ/dx vs residual range tracking the muon template; and the
  calibrated dQ/dx being independent of drift distance once lifetime and
  recombination are applied.

### 9.4 The PDVD operating-point idiom

PDVD keeps its jsonnet defaults OFF and byte-identical, and turns the
production operating point on through **`run_clus_evt.sh` environment
defaults** ([17_pdvd-clustering-qlmatching-chain.md](17_pdvd-clustering-qlmatching-chain.md)
"Operating point is ON via the runner"). The PR runner must follow the same
convention, or the two halves of the chain will drift apart in how they are
configured and neither will be reproducible from the config alone.

---

## 10. Milestones

| M | deliverable | done when |
|---|---|---|
| **M1** | pctree save | `tensor_outname` knob added; PDVD Q/L outputs byte-identical with it empty (hash gate, label reported); tarball written and readable |
| **M2** | PR job round-trip gate | `pdvd/wct-pr-perevt.jsonnet` loads the tarball with an empty pipeline and reproduces the Q/L `clustering` Bee layer content-identically |
| **M3** | STM on all matched pairs | wide `beam_window` + `nu_per_bundle=true`; `cluster_t0` verified in range for every matched main; STM/TGM verdicts appear for every bundle; **firing census** of all guards and of the drop-row stages in §4 |
| **M4** | trajectory + dQ/dx | PDVD `particle_dataset.jsonnet`, `pdvd_track_fitting.json` (§7b), a recombination model, a PDVD fiducial; fits complete on the 16-volume geometry; dQ/dx-vs-residual-range plots for hand-scanned stopping muons |
| **M5** | Michel identification | does the flags-6-8 block fire on PDVD stopping muons (acceptance test in §5)? then persist `michel_energy` + shower identity into `TaggerInfo` and `PrDisplayDump` |
| **M6** | Magnify-tracking PDVD | config-only test first; fork the visitor only if the 16-volume geometry breaks it |
| **M7** | calibration extraction | PDVD `PowerBoxRecombination` fitted on the STM sample; `DL`/`DT` from the flash-t0-anchored width-vs-drift fit (§8); Michel endpoint as the absolute-scale anchor |

Sequencing note: M4 depends on §7a being settled, and M6 and M7 both depend on
M4. M5 can proceed in parallel with M6.

---

## 11. Attention points (gotchas)

1. **`beam_window_only=false` is the wrong way to run on everything.** Use a
   wide window plus `nu_per_bundle=true` (§2.1). The false path silently
   reconstructs one arbitrary cosmic per event.
2. **`nu_skip_cosmic` must be false for PDVD.** With SBND's setting the
   neutrino PR skips exactly the STM-tagged mains whose Michels we want.
3. **A byte-identical compiled config does not mean the fit is unchanged.**
   `pdvd_track_fitting.json` is read at runtime (§7b).
4. **The STM guards are backwards for this analysis.** Every one of them
   exists to abstain from an STM call (§2.3).
5. **Four live drift speeds** (§7a). State which one every number used.
6. **`fiducialutils` must precede every tagger**, or they silently return
   false with no error.
7. **`tagger_check_tgm` must precede `tagger_check_stm`**; STM skips
   TGM-tagged mains.
8. **`steiner_refresh` must use `replace=false`** and follow `protect_bundle`
   immediately.
9. **`CreateSteinerGraph` needs `require_beam_flash=false`.** Its default is
   uBooNE-only behaviour, and with the default the steiner stage silently
   processes *nothing* on a chain whose flags come from `QLMatching`.
10. **Do not edit another experiment's configs** as a side effect, and do not
    modify the SBND PR job — fork it.
11. **Docs and outputs live in two repos.** Anything under `pdvd/` is
    committed in `wcp-porting-img`, not in the toolkit; `*.sh` there is
    gitignored and needs `git add -f`.
12. **Never regenerate over an existing label or snapshot directory.** A new
    run gets a new tag.

---

## 12. Open items

- **Which drift speed?** (§7a) Blocking for M4 and M7. Four values are live;
  imaging and clustering currently disagree.
- **PDVD sim vs reco transport mismatch** — `DL`, `DT` and lifetime all differ
  between `params.jsonnet` and `simparams.jsonnet`, and neither pair was
  measured for PDVD. Reported, not fixed.
- **No E-field or temperature is configured** anywhere in PDVD, while a full
  Walkowiak parameterisation sits unused in `funcs.jsonnet`.
- **The +0.75 us light self-trigger offset** from doc 23 remains owner-gated;
  it would shift `cluster_t0` for every matched pair.
- **The three Michel-physics constants** in §7e need published sources before
  use.
- **The §4 drop rows** are provisional until the M3 firing census confirms
  them. Zero fires alone does not retire a stage.
- **Whether the flags-6-8 Michel block fires** on PDVD topologies (M5
  acceptance test). If it does not, a dedicated extractor is the fallback —
  but only after the measurement says so.

---

## Milestone log

- **2026-09-02** — design round. No code changed. Owner decisions on record:
  data first (MC later); Michels for absolute energy scale / recombination
  *and* diffusion-based drift localisation of low-energy events; reuse the
  full SBND PR tail trimmed by config, retiring stages only on measurement.

# How to run the PDVD beam-particle pattern recognition (`CheckBeamParticle`)

A user guide: how to configure the PDVD beam-particle PR chain, how to run it, and how to look at its output.
The design, the gates and the per-event results are in the companion
[doc 120](120_beam-particle-pr.md). This guide only covers usage.

**What the chain does.** For each event, it runs the Wire-Cell neutrino PR chain on the **beam-flash-matched
bundle only**. That chain covers proto-segments, EM-shower clustering, particle ID and the particle flow. The
chain is rooted at the **beam particle's entry point** instead of a neutrino vertex. It runs no cosmic taggers
and no DL vertex. The outputs are a Bee event display (clusters + fitted tracks + showers + vertices + the
particle-flow tree), a ROOT file and a JSON summary.

**Code versions.** Everything used here is on `master` of both repositories:
- wire-cell-toolkit `129362fe` or later, which includes the round-2 particle-flow fixes.
- wcp-porting-validation `main`, which provides the `pdvd/` runners.

**Reference result.** The six run-39305 kaon-candidate events, produced with exactly the commands below:
<https://www.phy.bnl.gov/twister/bee/set/6314c336-a71d-48e5-8255-d260fe11441e/event/list/>

| Bee index | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| event | 157312 | 317673 | 245576 | 191916 | 2001 | 36591 |
| outcome | beam PR | beam PR | beam PR | no PR: entry cut | no PR: no beam bundle | no PR: no beam bundle |

## Quick start (the reference events on the BNL machine)

```bash
cd /nfs/data/1/xqian/toolkit-dev && direnv allow          # environment: local/ install, WIRECELL_PATH, wire-cell on PATH
cd wcp-porting-img/pdvd

TAG=mytag                                                  # ALWAYS use a new tag of your own (see "Tags" below)
for e in 157312 317673 245576 191916 2001 36591; do
  d=work/039305_${e}_$TAG; mkdir -p $d                     # the PR job reads the clustering pctree from the tag dir
  ln -sfn ../039305_$e/pctree-evt$e.tar.gz $d/
  ln -sfn ../039305_$e/pctree-evt$e.tlas   $d/
  PDVD_BEAM_LIGHT_SUFFIX=_d119beam PDVD_PR_TLA="-S dl_weights=''" \
    ./run_pr_evt.sh -beam -s $TAG 39305 $e
done

python3 kaon/make_d120_bee_zip.py $TAG /home/$USER/tmp/beam6-$TAG.zip   # six events -> one Bee set
./upload-to-bee.sh /home/$USER/tmp/beam6-$TAG.zip                     # prints the .../set/<uuid>/event/list/ URL
```

Each event takes about 3 s of wall time and 0.3 GB of RSS. With the shared install of 2026-09-26, the Bee members
of 157312 are identical, member by member (`abtest/hash_archive.py`), to the reference set above.

`PDVD_BEAM_LIGHT_SUFFIX=_d119beam` is needed only for these six events. Their original light directories predate
the trigger metadata, so the runner must be pointed at the relabelled `_light<EVT>_d119beam` copies (details
under "Configuration"). Light archives produced today carry the metadata, and new events need no suffix.

## 1. Environment

- **On the BNL machine:** `direnv allow` in `/nfs/data/1/xqian/toolkit-dev`. That puts `wire-cell`/`wcsonnet` on
  `PATH` and sets `WIRECELL_PATH` to `toolkit/cfg:wire-cell-data`. The runner itself also prepends that
  `WIRECELL_PATH`.
- **In your own checkout:**
  1. Build and install wire-cell-toolkit `master` (the usual `./wcb configure`/`build`/`install`, or cmake), then
     put the install's `bin/` on `PATH` and its `lib/` on `LD_LIBRARY_PATH`.
  2. Point `WIRECELL_PATH` at `<toolkit>/cfg` and a `wire-cell-data` checkout.
  3. Clone wcp-porting-validation and run from its `pdvd/` directory.
  4. `run_pr_evt.sh` hard-codes `WCT_BASE=/nfs/data/1/xqian/toolkit-dev` near its top. Edit that line to your
     base directory.
- **Python** needs `numpy`, and `uproot` if you want to read the ROOT output.

## 2. Inputs: what must exist before the PR step

The PR step runs on the output of the PDVD clustering + Q/L-matching job. For event `<EVT>` of run `<RUN>`, it
needs the following, where `<RUN6>` is the run number zero-padded to six digits:

| file | written by | content |
|---|---|---|
| `work/<RUN6>_<EVT>[_<TAG>]/pctree-evt<EVT>.tar.gz` | `run_clus_evt.sh -save-pctree` | clusters with their matched-flash bundle id (`matched_flash_gid`) and `cluster_t0` |
| `work/<RUN6>_<EVT>[_<TAG>]/pctree-evt<EVT>.tlas` | same | drift speeds, trigger offsets, readout window and `opflash_input=` (the light archive the Q/L job used) |
| the light archive named by `opflash_input=` | `run_light_evt.sh` | flashes + `trigger_us`/`tc_type` (the CTB beam trigger) |

For a new event, run the standard PDVD chain first. The run-39305 events use the wrapper
`kaon/run_kaon_chain.sh <stage> <evt…>`, which loops over the stages in the same order:

```bash
for a in 4 5 6 7; do ./run_nf_sp_dnnroi_evt.sh -a $a <RUN> <EVT>; done   # NF + SP (top drift = anodes 4-7)
for a in 4 5 6 7; do ./run_img_evt.sh          -a $a <RUN> <EVT>; done   # imaging
./run_light_evt.sh <RUN> <EVT>                                           # light; PDVD_BEAM_LABEL=1 is the default
./run_clus_evt.sh -a 4,5,6,7 -calib -save-pctree <RUN> <EVT>             # clustering + Q/L matching + pctree
```

- **Anodes:** `-a 4,5,6,7` is the top drift, the only volume read out in run 39305. Use all anodes (the default)
  for a run with both drifts.
- **Trigger metadata:** the light job stamps `trigger_us`/`tc_type` into `opflash_pdvd-wct.tar.gz`
  (`PDVD_BEAM_LABEL=1`, the default since doc pdvd/119). Without it, `-beam` skips the event with a clear message.

## 3. Running

```bash
./run_pr_evt.sh -beam [-s <TAG>] <RUN> <EVT>          # one event
./run_pr_evt.sh -beam [-s <TAG>] <RUN> all            # every work/<RUN6>_<EVT>[_<TAG>] dir of the run (PDVD_MAX_JOBS, default 6)
```

`-beam` selects the pipeline
`switch_scope, flag_mains, unmerge_assoc, steiner, fiducialutils, check_beam_particle, tracking_visitor, tagger_output, pr_display`.

The runner then does the following:
1. Reads the beam trigger time and type from the light archive.
2. Skips the event if the trigger type is not a CTB beam type (14, 15, 20, 21, 22) or if the trigger is missing.
3. Prints the beam window, then compiles and runs the job.

### Tags

- **Output location:** `-s <TAG>` writes into `work/<RUN6>_<EVT>_<TAG>/`. Without `-s`, the job writes into
  `work/<RUN6>_<EVT>/` and **overwrites** the `mabc-pr.zip` / `tracking-pr.root` already there.
- **Pick a new tag per run:** other people's results live in these directories.
- **Tag directory setup:** a tag directory needs the pctree. Symlink it from the untagged directory as in the
  quick start.

### Log lines to check

The verdicts are printed in `work/<RUN6>_<EVT>_<TAG>/wct_pr_<RUN6>_<EVT>.log`:

```bash
grep 'CheckBeamParticle:' work/039305_157312_$TAG/wct_pr_039305_157312.log
```

```
CheckBeamParticle: 43 main(s), 2 in window [2792.084, 2793.284) us over 1 bundle(s); picked gid 96 (brightest)
CheckBeamParticle: gid 96 member cluster 30 L 94.4 cm closest point (110.6, 161.5, 8.6) d_entry 8.4 cm      <- one line per bundle member
CheckBeamParticle: cluster 30 entry (110.6, 168.6, 8.6) cm d_nominal 12.5 cm cos_beam 0.99 -> vertex 30001 (111.0, 168.2, 9.4) snap 1.8 cm split 0 | neutrino-chain vertex 1 (99.2, 110.8, 58.5) d 77.2 cm
CheckBeamParticle: cluster 30 gid 96 entry_ok 1 | 11 shower(s) Enu 492.4 MeV vertex (111.0, 168.2, 9.4) cm isFC 0 | 1151 ms
```

- `entry_ok 1` means the particle flow was built.
- `entry_ok 0` means the main cluster's nearer end lies more than `beam_entry_max_dist_cm` from the nominal entry.
  Nothing is published (event 191916).
- `no beam bundle` means no flash-matched cluster falls in the beam window (events 2001 and 36591).
- "neutrino-chain vertex" is where the ordinary neutrino chain would have put the main vertex. It is printed for
  comparison only.

## 4. Configuration

All knobs are jsonnet top-level arguments of `toolkit/cfg/pgrapher/experiment/protodunevd/wct-pr-perevt.jsonnet`.
Override them through `PDVD_PR_TLA` with `wcsonnet`'s `-S name=<jsonnet code>` form. The value is split on
whitespace, so write objects and arrays without spaces.

```bash
PDVD_PR_TLA="-S dl_weights='' -S beam_entry_max_dist_cm=30 -S beam_entry_point_cm=[105,160,0.6] \
             -S beam_pr_knobs={min_main_length_cm:10,improve_entry_vertex:true}" \
  ./run_pr_evt.sh -beam -s $TAG 39305 157312
```

| TLA | default | meaning |
|---|---|---|
| `beam_trigger_us`, `beam_tc_type` | from the light archive | the CTB trigger time (µs, raw flash-time axis) and type; the runner fills them; `PDVD_BEAM_TRIGGER_US` / `PDVD_BEAM_TC_TYPE` override |
| `beam_tc_types` | `[14,15,20,21,22]` | trigger types accepted as beam |
| `beam_window_rel_us` | `[-1.5,-0.3]` | beam window relative to the trigger: a bundle is "beam" when its `cluster_t0` falls in `[trigger−1.5, trigger−0.3)` µs; the brightest flash in the window wins |
| `beam_entry_point_cm` | `[110,159,0.6]` (C++) | nominal beam entry point (x, y, z), derived from the data (owner ruling 2026-09-25; the GDML beam-plug axis agrees in y, z and angle but sits ~100 cm higher in x, doc 120 sec 1) |
| `beam_dir` | `[-0.095,-0.704,0.704]` (C++) | nominal beam direction (GDML beam-plug axis); used to break ties between the two ends of the main cluster |
| `beam_entry_max_dist_cm` | `50` (C++) | an entry farther than this from the nominal point fails (`entry_ok 0`) |
| `beam_pr_knobs` | `{min_main_length_cm:10}` | extra stage keys, see below. **Overriding replaces the whole object**, so repeat `min_main_length_cm:10` if you want to keep the floor |
| `dl_weights` | production DL weights | pass `''`: the beam chain does not use the DL vertex, and `''` saves loading it |

Useful keys inside `beam_pr_knobs` (C++ defaults in `CheckBeamParticle::default_configuration()`):

| key | default | meaning |
|---|---|---|
| `min_main_length_cm` | 0 (C++) / 10 (PDVD TLA) | a bundle cluster shorter than this cannot be the main (beam) cluster |
| `entry_snap_tol_cm` | 10 | snap the entry vertex onto an existing graph vertex within this distance, else split the segment there |
| `entry_tie_tol_cm` | 5 | when both ends of the main cluster are this close in distance, choose by direction (`beam_dir`) |
| `beam_dir_max_angle_deg` | −1 (off) | optional angular cut between the main cluster and `beam_dir` |
| `improve_entry_vertex` | false | re-fit the entry vertex position from its segments |
| `entry_fail_fallback_geo` | false | on `entry_ok 0`, continue with the neutrino-chain vertex instead of publishing nothing |
| `entry_long_muon_absorb` | false | true restores the round-1 behaviour, where the entry-rooted chain is absorbed into one long "muon" node |
| `kine_charge_t0_frame` | true | compute shower charge energies in the t0-corrected frame; false reproduces the round-1 zero shower energies |

The stage also receives the neutrino chain's full PR knob set (about 200 keys), so any neutrino-chain PR knob
passed through `PDVD_PR_TLA` reaches it as well.

To check what a run will actually use, compile without running:

```bash
PDVD_PR_COMPILE_ONLY=1 PDVD_BEAM_LIGHT_SUFFIX=_d119beam PDVD_PR_TLA="..." ./run_pr_evt.sh -beam -s $TAG 39305 157312
# -> work/039305_157312_$TAG/.wct-pr_$TAG.json ; look at the node with "type": "CheckBeamParticle"
```

The compiled `beam_window_low/high` are in WCT internal time units (ns). For example, 157312 gives
2792084 / 2793284, i.e. [2792.084, 2793.284) µs.

## 5. Outputs

Each output lands in `work/<RUN6>_<EVT>_<TAG>/`:

| file | content |
|---|---|
| `mabc-pr.zip` | Bee event (one event, index 0). Layers: `clustering-global`, `track_fit-global` (fitted track points, dQ/dx), `shower_track-global` (points coloured by track/shower), `vertices-global` (the entry vertex is the q=15000 point), `mc` (the **particle-flow tree**, rooted at the entry), dead-channel areas per anode face |
| `tracking-pr.root` | `T_beam_particle` (one row per event: selection record: `gid`, `t0_us`, `flash_pe`, main `cluster_id`/`main_len_cm`, `entry_*`/`exit_*`, `entry_ok`, `entry_vtx_*`, `geo_vtx_*` = the neutrino-chain vertex, `n_showers`, `kine_reco_Enu`, …); `T_kine` (energies per particle); `T_rec_charge` / `T_proj_data` / `T_cluster` (fitted charge and projections); `T_tagger` (only `match_isFC` is meaningful, the other fields are defaults) |
| `calib-pr-evt<EVT>.json` | JSON dump: segments, showers, vertices, `main_vertex`, `kine`, … |
| `wct_pr_<RUN6>_<EVT>.log` | the job log (the `CheckBeamParticle:` verdict lines) |

Example ROOT read:

```python
import uproot
t = uproot.open("work/039305_157312_mytag/tracking-pr.root")["T_beam_particle"]
print(t.arrays(["cluster_id", "entry_ok", "entry_x", "entry_y", "entry_z", "kine_reco_Enu"], library="np"))
```

## 6. Display (Bee)

- **One event:** `./upload-to-bee.sh work/<RUN6>_<EVT>_<TAG>/mabc-pr.zip`. `mabc-pr.zip` is already a Bee
  upload.
- **Several events:** combine them into one set.
  - For run 39305, use `python3 kaon/make_d120_bee_zip.py <TAG> <out.zip> [evt …]`. It takes the doc-118 six
    events by default, and the set index follows the order given.
  - For another run, change the hard-coded `039305` in that script (one line).
  - Then `./upload-to-bee.sh <out.zip>`.
- **The upload is public.** Anyone with the URL can view it, so upload only data you may share.

In Bee (<https://www.phy.bnl.gov/twister/bee>, the URL the upload prints):
- Select the `track_fit`, `shower_track` and `vertices` layers alongside `clustering` to see the fitted
  trajectories, the track/shower separation and the vertices.
- The **particle flow** is the `mc` tree in the side panel. Its top node is the reconstructed energy, and each
  node is a particle (PDG type, kinetic energy) with its daughters.
- Events without PR (entry cut failed, or no beam bundle) show the clustering only.

## 7. Known limitations

These are open items; details are in doc 120 sec 7:
- The beam track is shown as its chain of track segments, not as one particle. The short entry segment can be
  typed `e-` once shower charge energies are non-zero (157312). A PID lock on the beam chain is the natural next
  knob.
- The neutrino chain's main-vertex-specific passes act at the **entry** point. A secondary interaction vertex
  inside the detector is an ordinary graph vertex to them.
- The nominal entry point is the data-derived one. The GDML beam-plug axis differs by ~100 cm in the drift
  coordinate (x), which is still an open geometry question.
- Only one bundle per event is used (the brightest flash in the window). Other clusters in that bundle are
  companions in the flow, not separate candidates.

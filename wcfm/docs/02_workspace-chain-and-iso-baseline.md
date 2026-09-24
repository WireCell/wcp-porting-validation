# 02 — The workspace chain (W1–W5) and the isochronous baseline

**Date:** 2026-09-24. **Follows:** [01_fm-integration-campaign.md](01_fm-integration-campaign.md) §6.
**Scope:** the DUNE FD-HD `dune10kt-1x2x6` ("workspace") reconstruction chain the FM campaign
runs on — iso-track gun simulation → NF → SP → imaging (+ `BlobDepoFill` truth) → clustering →
pctree — its bugs, its gates, and the first measurement of what the projective-readout ambiguity
costs on isochronous tracks. No FM code yet; this is the substrate F1–F6 build on.

## 0. Repro / provenance

Trees: toolkit `/home/xqian/toolkit-dev/toolkit` (`apply-pointcloud`, base `b5a897f6`, this doc's
commit `69515f37`), wcp `/home/xqian/toolkit-dev/wcp-porting-img` (`main`, base
`853d55e1`, this doc's commit = the one that adds it, `git log -1 -- wcfm/docs/02_workspace-chain-and-iso-baseline.md`). Working area `wcp-porting-img/wcfm/` (toolkit symlink
`wcfm`). Scratch `/home/xqian/tmp/wcfm-w1/`.

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/wcfm
# W1 gate: compiled JSON of every dune10kt-1x2x6 entry config before/after the params fix
/home/xqian/tmp/wcfm-w1/compile_all.sh /home/xqian/tmp/wcfm-w1/post   # (pre = same script on the stashed tree)
# W2 events + simulation (10 events, 5 parallel; ~20-35 s, 0.6-0.8 GB each)
python3 gen_iso_tracks.py --force && WCFM_MAX_JOBS=5 ./run_sim_evt.sh 1 all
# W4 time_offset calibration on event 1 (writes work/000001_1_toff*/)
scripts/scan_time_offset.sh 1 1
# W3+W4 imaging with truth tiers, clustering; then twice more under setarch for the determinism gate
WCFM_MAX_JOBS=5 ./run_img_evt.sh 1 all && WCFM_MAX_JOBS=5 ./run_clus_evt.sh 1 all
cd ../abtest && WCFM_SETARCH=1 ./run_events.sh wcfm-w3-a both ../wcfm/abtest_events.txt \
             && WCFM_SETARCH=1 ./run_events.sh wcfm-w3-b both ../wcfm/abtest_events.txt \
             && ./ab_compare.sh wcfm-w3-a wcfm-w3-b
# W5 baseline tables and plots
cd ../wcfm && python3 scripts/iso_baseline.py --out docs/02_tables work/000001_{1,2,3,4,5,6,7,8,9,10}
# toolkit unit test of the reader fix
cd /home/xqian/toolkit-dev/toolkit && TMPDIR=/home/xqian/tmp ./build/sio/wcdoctest-sio -tc='framefile*'
```

## 1. Bugs found and fixed (toolkit)

### 1.1 `cfg/pgrapher/experiment/dune10kt-1x2x6/params.jsonnet`: APAs placed at their cathodes

**Symptom.** `params.jsonnet:60` set every APA's `centerline = sign*apa_cpa` (±3.63 m) while the
wires file `dune10kt-1x2x6-wires-larsoft-v1.json.bz2` has all 12 APAs at x = 0 (doc 01 §0).
**Root cause.** A copy of the 3-APA-row (APA-CPA-APA) layout comment ("for LArSoft wires the
centerline is −/+/−/+") that never matched this 1-row module; `simparams.jsonnet:73` had already
been corrected to 0 without the base file following. **Why it hid.** Nothing ran this geometry
past SP, and SP does not read the face x positions. **Fix.** `centerline = 0`, comment, `STATUS.md`
§1/§3c/§4 rewritten (layout: 2 rows in y × 6 columns in z, both faces live, channel map).
**Verification (gate).** Compiled JSON of all 12 entry configs before/after (`compile_all.sh`,
`wcls-*` with their LArSoft ext-vars): 10 byte-identical; `wcls-sp` and `wcls-nf-sp` at
`reality=data` change in exactly the 72 `AnodePlane.faces[].{anode,response,cathode}` leaves
(e.g. face 0 anode −3591.2 → 39.5 mm, cathode −1.6 → 3629.2 mm) and are then byte-identical to
their `reality=sim` compile. `wct-sim-check.jsonnet` also lost its dead `fileio.jsonnet` import
and got the 4th `chndb-perfect` argument (latent): compiled JSON identical.
**Status: NOT bit-identical for the two data-reality LArSoft configs (by design — they were
wrong); no WCT production job consumes them.**

### 1.2 `sio/FrameFileSource`: a frame file with an EMPTY channel-mask map is unreadable

**Symptom.** Imaging the clean-sim frames threw
`FrameFileSource: cmm load failed tag="bad"` → `IOError numpy parse error of chanmask`, after
the sim's `FrameFileSink{masks:true}` had written `chanmask_bad_100.npy` of shape (0,3).
**Root cause.** OmnibusNoiseFilter emits the `bad` map even when no channel is masked;
`FrameFileSink::masks` writes it as a (0,3) array; `pigenc::File::as_type<int>()` returns null
for an empty array, so `pigenc::eigen::load` returns false and the source throws.
**Why it hid.** Every data frame file has bad channels; PDHD/PDVD sim files were written with
`masks:false`. **Fix.** `FrameFileSource` treats a chanmask member with `shape[0]==0` as the
tag present with no entries (`sio/src/FrameFileSource.cxx`). Reader-side only: no output file
changes; a previously-throwing path now succeeds. **Verification.** New
`sio/test/doctest_framefile_empty_mask.cxx` (sink→source round trip of an empty and of a
non-empty `bad` map; the empty case failed before the fix), `./build/sio/wcdoctest-sio` passes
(5 cases, 40 assertions, all pass, 2026-09-24 07:48). **Status: byte-identical for every file that was readable before.**

### 1.3 `img/BlobDepoFill`: every depo of a −x face skipped (all its blobs "ghosts")

**Symptom.** With the truth tiers on, every event whose track sat at x < 0 (face 1) came out
with captured true charge exactly 0 and ghost fraction 1.000, every x > 0 event at 95–98 %
(§0 quick table before the fix: events 1, 7, 9, 10 vs 2–6, 8). **Root cause.**
`BlobDepoFill.cxx:275` skipped a depo as "behind the face" when `pimpos->relative(pos)[0] < 0`,
i.e. the raw x offset from the response plane, sign-blind to the face's drift direction: on a
−x face every depo in the volume has x below the response plane. It only ever ran on PDSP anode
0 face 0. A second edge sharpened it: the depo file is float32, so a depo drifted exactly onto
the response plane reads back 2.9 µm off (−130.0155029 vs −130.0155 mm) — on face 1 "behind",
on face 0 "in front". **Fix.** `dirx * rpos[0] < −10 µm` with `dirx = IAnodeFace::dirx()`
(+1 for a +x volume, −1 for −x). Reader-side geometry only; a +x face gets the same depos as
before (the tolerance admits depos up to 10 µm behind the plane, physically nothing).
**Verification.** New `img/test/doctest_blobdepofill_faces.cxx` (one FD-HD APA, both faces,
one depo 3 µm behind each response plane at a blob centre: both faces must receive the charge,
a depo 50 mm behind must not; face 1 received 0 before the fix); event-level: the four −x events
go from 0 to 0.90–0.99 captured. **Status: NOT bit-identical for BlobDepoFill on a −x face (it
was wrong); byte-identical on +x faces (verified: the +x events' numbers are unchanged).**

### 1.4 `aux/ClusterArrays` loader: a cluster file with no blobs throws

**Symptom.** Clustering event 1 aborted in `ClusterFileSource` with `std::out_of_range
unordered_map::at` while loading `clusters-apa-anode10-ms-active.tar.gz`. **Root cause.**
`ProjectionDeghosting` tagged all 15 830 blobs of anode 10 (§5: the isochronous failure the
campaign is about), so the sink wrote s/w/a arrays and a-s/a-w edges but no `b` array, and
`to_cluster` did `nas.at('b')` unconditionally (`aux/src/ClusterArrays.cxx:763`; the `m` array
was already guarded). **Fix.** Guard the blobs block like the measure block. Reader-side; files
with blobs load exactly as before. **Verification.** New `sio/test/doctest_clusterfile_noblobs.cxx`
(sink→source numpy round trip of a slice + channels + wires cluster with no blobs; threw before),
`./build/sio/wcdoctest-sio` passes (5 cases, 40 assertions), `./build/img/wcdoctest-img`
(the new case: 36 assertions) and `./build/aux/wcdoctest-aux` (29 cases, 111 331 assertions) pass. **Status: byte-identical for every file that loaded before.**

### 1.5 Production gate for the three reader fixes (§1.2–1.4)

All three fixes sit on paths that previously threw, so no successfully produced file can change;
the gate confirms it. Pre-fix vs post-fix binaries (the three `.cxx` stashed, `wcbuild`,
`abtest/run_events.sh … both` on the standard manifest `abtest/events.txt` — 4 PDHD + 2 PDVD
events, imaging from the surviving traditional SP frames (`IMG_ARGS="-d off"`; the DNN-ROI frames
were removed by the disk-cleanup rounds, which is also why the 09-19 `d113post` snapshot and the
first attempt `wcfm-fixgate` hold only skipped imaging, rc=2 — both void), then unstash, rebuild,
rerun: labels **`abtest/snap/wcfm-fixgate-pre`** and **`wcfm-fixgate-post`**,
`ab_compare.sh wcfm-fixgate-pre wcfm-fixgate-post` → 178 archives with identical member hashes,
all 24 `img_meta`/`clus_meta` rc=0, `OVERALL: PASS` (2026-09-24 08:0x; libraries rebuilt and
installed between the arms, mtimes recorded in `/home/xqian/tmp/wcfm-w1/fixgate_prepost.sh` output).
Unit tests: `wcdoctest-sio` 5/5, `wcdoctest-img` (new case 36 assertions), `wcdoctest-aux` 29/29.

### 1.6 Not fixed, recorded

- `lar.drift_speed` 1.6 mm/µs (common base) vs 1.565 in the field-response file: kept at 1.6
  everywhere in wcfm so the Drifter, BlobDepoFill and the clustering x agree (`STATUS.md` §4).
- The workspace NF is MicroBooNE's `mbOneChannelNoise` on `chndb-base` with MicroBooNE
  numbers (baseline 2048/400 ADC, U/V harmonic frequency masks, hand-made response). The wcfm sim
  job overrides per anode (last mention wins): nominal baselines from this detector's Digitizer
  (induction 9401.5, collection 3600.7 ADC), no frequency masks/response, RMS cuts off, no bad
  channels. `STATUS.md` §4 already flags the stale NF.
- The three `wcls-sim-drift-*` entries need `--ext-code` (numeric) ext-vars, not `-V`; the doc-01
  "all compile" claim used `-V` and they abort — a harness detail, not a config bug.

## 2. W2 — the iso-track gun simulation (`wcfm/`)

| File | Role |
|---|---|
| `wcfm_params.jsonnet` | one place for the campaign constants: `params` = dune10kt `simparams`, drift speed, tick/nticks/`tick_span` 4, `depofill_time_offset`, per-face FV x extents, overall FV box, `face_groups`, Bee tag |
| `gen_iso_tracks.py` → `events/000001_<evt>.json`, `abtest_events.txt` | the event list (§2.1) |
| `wct-sim-iso-track-nf-sp.jsonnet` | TrackDepos → Drifter → DepoBagger → DepoSetFanout → per anode [DepoTransform → Reframer(tbin 125) → AddNoise → Digitizer → NF → SP → `FrameFileSink{gauss,wiener,masks}`] + `DepoFileSink` (drifted depos + priors); `Random` seeds from the event |
| `run_sim_evt.sh [-n] [-r] <run> <evt|all>` | `work/<run6>_<evt>/sim-frames-anode<N>.tar.bz2`, `sim-depos.tar.bz2`, log, `time_sim.txt` (abtest `timecmd.py`), provenance |
| `_runlib.sh` | pdvd batching, `WCFM_MAX_JOBS` default 6 |

Only the anodes a track crosses are simulated and imaged (the event JSON's `anodes`, from the
wires-file y-row / z-column table); FD-HD APAs are independent sensitive boxes (`DepoTransform`
keeps only depos inside a face's box), so nothing is lost.

### 2.1 The event list (run 1, seeds 1000+evt)

| evt | kind | tracks (angle to the anode plane / length) | anodes | x0 (cm) |
|---|---|---|---|---|
| 1 | iso | 0°/300 cm | 8, 10 | −242.6 |
| 2 | iso | 0°/500 cm | 2, 4 | 275.8 |
| 3 | iso | 2°/300 cm | 4, 6 | 188.9 |
| 4 | iso | 5°/300 cm | 5, 7 | 98.6 |
| 5 | iso | 10°/300 cm | 2, 4, 6 | 223.4 |
| 6 | iso ×2 | 0°/300 + 1°/250 cm | 0, 2, 3, 5 | 73.4 |
| 7 | iso ×3 | 0°/300 + 1°/250 + 2°/400 cm | 8, 9, 11 | −252.1 |
| 8 | iso ×4 | 0°/300 + 0°/200 + 3°/350 + 5°/250 cm | 9, 11 | 141.7 |
| 9 | cosmic | random 3-D, 400 cm | 9 | −332.7 |
| 10 | cosmic ×2 | random 3-D, 300 + 350 cm | 4, 6 | −285.0 |

Overlay tracks share the drift distance (|Δx| < 3 cm) and the APA region (|Δy|,|Δz| < 1 m) so
their projections share slices and wires — that is where projective ghosts come from. Charge
−500 e per 0.1 mm step (5000 e/mm, the pdhd_sim MIP convention), `time 0`, `fixed` sim mode.

Sim cost: 13–35 s wall, 0.61–0.82 GB peak RSS per event (1–4 anodes, 5 in parallel).
Event 1 check: the 0° track lands in all three planes within ticks 2983–3016 (frame-relative
1.49–1.51 ms; drifted-depo time 1.4347 ms at the response plane, |x| = 130 mm), 169/102/161
active U/V/W channels on anode 8, 447/263/424 on anode 10; 30 000 drifted depos + 30 000 priors.

## 3. W3 — imaging and clustering (`wcfm/`)

Forks BY DUPLICATION of the in-tree PDHD production jobs (M10), production cuts untouched:

| File | ← PDHD file | What differs |
|---|---|---|
| `img.jsonnet` | `cfg/pgrapher/experiment/pdhd/img.jsonnet` | params = `wcfm_params`; output names `anode<N>`; the "full" solving chain can carry the W4 truth catchers (§4); nothing else |
| `wct-img-all.jsonnet` | `pdhd/wct-img-all.jsonnet` | no null-face restore (both faces are real, opposite-drift); TLAs `depos`, `time_offset` |
| `clus.jsonnet` | `pdhd/clus.jsonnet` | `dvm` generated for 12 anodes × 2 faces (FV_x face 0 = [30.0, 3629.2] mm, face 1 mirrored; overall box of the module); drift speed 1.6; **topology** below; no `cathode_connect` (the cathodes are the module's outer walls), no Q/L, no flash keys |
| `wct-clustering.jsonnet` | `pdhd/wct-clustering.jsonnet` | the two-face-volume graph; no Q/L branch; pctree always written |
| `run_img_evt.sh`, `run_clus_evt.sh` | `pdvd/run_*_evt.sh` | `[flags] <run> <evt|all>`, one imaging process per anode, `timecmd.py` timing, provenance files, wires-file cross-check |

**Why the PDHD clustering topology cannot be copied.** PDHD merges the two faces of an APA
(stage 2) and then groups x-aligned APAs by ident parity. Here both faces of every APA are live
*and opposite drift volumes*: `validate_drift_group` (`clus/src/ClusteringFuncs.cxx:108`) refuses
a mixed-face scope (`allow_mixed_faces` is for PDVD's shared volume), and ident parity is the
y row. FD-HD is the first geometry in the repo imaging both faces with opposite drift. The wcfm
graph is therefore:

```
per anode:  ClusterFileSource(active) → ClusterFanout(2) ─┬→ per_face(anode,0): ClusterScopeFilter(face) → PointTreeBuilding → MABC stage 1
            ClusterFileSource(masked) → ClusterFanout(2) ─┴→ per_face(anode,1): (same)
per face:   PointTreeMerging(all run anodes, face F) → MABC "groupf<F>": [deghost, protect_overclustering] (PDHD stage 2)
                                                        + [extend, regular×2, parallel_prolong, close, extend_loop, separate, connect1,
                                                           deghost(empty_view_unique), examine_x_boundary, neutrino, isolated] (PDHD stage 3)
all-TPC:    PointTreeMerging(groupf0, groupf1) → MABC [switch_scope] → TensorFileSink pctree-evt<N>.tar.gz (+ mabc-*.zip)
```

Compiled-config proof (event 1, anodes 8+10): 4 `PointTreeBuilding`, 7 `MultiAlgBlobClustering`
(4 per-face, 2 groups, 1 all-TPC), 3 `PointTreeMerging`, DetectorVolumes metadata `a8f0pA`
FV_x [30.0155, 3629.1625] mm / `a8f1pA` [−3629.1625, −30.0155] mm, `dead_apa_groups` =
`groupf0/groupf1` over the run anodes, `TensorFileSink` `dump_mode false` → the pctree.

### 3.1 Runs, cost, determinism gate

All 10 events run through sim → img (+truth) → clus. Cost per event (`abtest/timecmd.py`, peak RSS
of the busiest process; 5 events in parallel, load ≈ 20 from other users):

| stage | wall (s) | peak RSS | notes |
|---|---|---|---|
| sim (1–4 anodes, NF+SP) | 13–35 | 0.61–0.82 GB | 30–120 k depos at 0.1 mm steps |
| imaging + 2 truth tiers, per anode | 4–11 | 0.46 GB | 6–25 s per event sequential over its anodes |
| clustering (both face groups + all-TPC + pctree) | 2–15 | 0.41 GB | evt 8 (4-track overlay, 9 379 blobs) is the 15 s |

**Determinism gate: PASS.** Labels `abtest/snap/wcfm-w3-c` and `wcfm-w3-d` (`scripts/w3_gate.sh`,
two complete img+clus runs of `wcfm/abtest_events.txt` under `setarch x86_64 -R`, snapshots
extended with the `tru0`/`tru` tiers and the pctree): `ab_compare.sh wcfm-w3-c wcfm-w3-d` →
178 archives (per event: 2 cluster files + 2 truth tiers per anode, 7 Bee zips, 1 pctree) with
identical `hash_archive.py` member hashes, `OVERALL: PASS`; every `img_meta`/`clus_meta` rc=0.
Labels `wcfm-w3-a`/`wcfm-w3-b` exist but are **void**: the runners' `env … setarch x86_64 -R
GOGC=off wire-cell` handed `GOGC=off` to `setarch` as the program, both runs failed in 0 s and
`ab_compare` "passed" on the files left in `work/` by the earlier runs. Fixed (env order) and the
gate script now aborts on any non-zero `run_events`/meta rc (M13: the void snapshots are left in
place, not deleted).

The pctree (`pctree-evt<N>.tar.gz`, 274 members on event 1) carries the 90 per-(anode, face,
plane) pixel clouds `pointtrees/<ident>/live/lpcmaps/arrays/ctpc_a<N>f<F>p{U,V,W}` — the join
target of doc 01 §4.4 exists on this geometry.

Two group-stage facts worth recording: (i) every PDHD stage-2/3 visitor ran on the 12-anode
single-face scope without a raise (`validate_drift_group` accepts any number of anodes as long as
the face and the `FV_x` metadata agree — they do, by construction of `dvm`); (ii) event 1's anode
10 enters clustering with **zero blobs** (§5), which is what exposed the loader bug of §1.4.

## 4. W4 — truth (`BlobDepoFill`) inside the imaging job

`wcfm/img.jsonnet` `solving(..., "full")` with `depos != ''` inserts, per anode, a
`ClusterFanout(2) + BlobDepoFill + ClusterFileSink` catcher at two points of the PDHD chain:

- tier **`tru0`** on the `BlobClustering → ProjectionDeghosting` edge: every tiled blob, before any
  deghosting (occupancy only — no charge is solved yet);
- tier **`tru`** on the `GlobalGeomClustering → ClusterFileSink` edge: the survivors, in the same
  vertex order as `clusters-apa-anode<N>-ms-active` (`BlobDepoFill` uses `copy_graph` and keeps
  blob idents), so the two files are `desc`-aligned (asserted by `iso_baseline.py`, together with
  identical blob geometry per desc).

One `DepoFileSource` (the sim's drifted depos, gen 0) → `DepoSetFanout(2)` feeds both fills.
`speed` = the drift speed, `nsigma` 3, `pindex` 2 (the C++ defaults), `time_offset` calibrated
below. Compiled-config proof (anode 8): the direct `GlobalGeomClustering → clustersink` edge is
gone, the 10 catcher edges are present, 0 duplicate edges. Output: `clusters-{tru0,tru}-anode<N>-ms-active.tar.gz`
next to the reconstructed file; the dead-channel sidecar `dead-channels-anode<N>.json` (the frame's
`chanmask_*` arrays, empty here) is what the GNN doc §7.3 asks for.

### 4.1 `time_offset` calibration (measured)

`BlobDepoFill` matches `depo.time() + time_offset` to the frame-relative slice start
(`MaskSlice` pre-creates slices from tick 0 of the frame). Expected: the response-plane transit,
`response_plane / drift_speed` = 100 mm / 1.6 mm/µs = 62.5 µs, plus the field-response peak delay.
Event 1 imaged at a ladder of offsets (`scripts/scan_time_offset.sh 1 1`):

| `time_offset` (µs) | 0…50 | 56.25 | 58 | 60 | 61 | 62 | 62.5 | 63 | 64 | 65 | 67 | 68.75 | 75…125 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| captured true charge (`tru` tier / depo file) | 0.000 | 0.034 | 0.225 | 0.573 | 0.753 | 0.905 | 0.952 | 0.976 | **0.988** | **0.987** | 0.884 | 0.382 | 0.000 |
| same, `tru0` tier | 0.000 | 0.053 | 0.317 | 0.605 | 0.758 | 0.906 | 0.952 | 0.976 | **0.989** | **0.988** | 0.886 | 0.412 | 0.000 |

(event 2: 0° track, 500 cm, anodes 2+4, face 0; scan ladder `scan_time_offset.sh 1 2` then
`… 58 60 61 62 63 64 65 67`; the 12.5 µs ladder points 75/87.5/100 were re-run after a build
collision, see §7.) The plateau is 64–65 µs: the 62.5 µs transit plus ≈2 µs of field-response
peak delay (the same +2 µs the `Reframer`/`tbin` comments in `simparams.jsonnet` describe).
**Adopted: `depofill_time_offset = 64.5 µs`** (`wcfm_params.jsonnet`); the slice is 2 µs wide, so
the residual is ≤ ±1 µs. The 1–2 % not captured at the plateau is the depo Gaussian tails
(`nsigma` 3) falling on wires outside the tiled blobs and the ~2 % of depos outside any blob.

A caution for the ghost label: the *exact-zero* ghost fraction moves with the offset even on
the plateau (0.78 at 62.5–64 µs, 0.84 at 65 µs) because the time tails of the depo Gaussian
leak a tiny fill into blobs of the neighbouring slice. `iso_baseline.py` therefore reports
both `q_true == 0` ("ghost") and `q_true < 0.05·q_reco` ("soft"); the GNN labels (doc 01 §4.6,
GNN doc §7.3) should use a relative threshold, not exact zero.

## 5. W5 — the isochronous baseline

`scripts/iso_baseline.py` on the gate's run-D outputs → `docs/02_tables/` (`iso_baseline_events.tsv`,
`iso_baseline_slices.tsv` — one row per (event, anode, face, slice) — `iso_baseline_summary.md`,
two PNGs). Definitions: **ghost** = blob whose `BlobDepoFill` charge is exactly 0; **soft** =
true charge < 5 % of the solved charge; **captured** = Σ true charge in the tier's blobs / |Σ
drifted depo charge|; strip width = `bounds.end − beg` (wires, half-open) per plane.

### 5.1 Per event

| evt | kind | θ (°) | tracks | slices with blobs | blobs `tru0` → final | ghost frac `tru0` / final / soft | max strip U/V/W (wires) | median blobs per slice | captured `tru0` / final | Σq_reco/Σq_true − 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | iso | 0 | 1 | 7 | 17 928 → 1 341 | 1.000 / 0.998 / 0.998 | 169/100/161 | 114 | 0.986 / **0.270** | −0.009 |
| 2 | iso | 0 | 1 | 14 | 7 194 → 4 550 | 0.840 / 0.766 / 0.871 | 98/628/433 | 364 | 0.989 / 0.989 | −0.008 |
| 3 | iso | 2 | 1 | 44 | 3 859 → 2 439 | 0.711 / 0.780 / 0.809 | 69/92/78 | 50 | 0.997 / 0.993 | +0.016 |
| 4 | iso | 5 | 1 | 91 | 482 → 426 | 0.726 / 0.697 / 0.758 | 10/54/27 | 4 | 0.980 / 0.980 | +0.001 |
| 5 | iso | 10 | 1 | 177 | 1 554 → 1 329 | 0.409 / 0.426 / 0.530 | 32/19/22 | 7 | 0.979 / 0.979 | −0.002 |
| 6 | iso ×2 | 0+1 | 2 | 41 | 6 628 → 4 872 | 0.948 / 0.967 / 0.975 | 233/517/448 | 27 | 0.972 / 0.970 | −0.006 |
| 7 | iso ×3 | 0+1+2 | 3 | 73 | 11 403 → 6 090 | 0.627 / 0.650 / 0.711 | 241/203/141 | 78 | **0.809** / 0.805 | −0.021 |
| 8 | iso ×4 | 0+0+3+5 | 4 | 116 | 16 247 → 9 379 | 0.837 / 0.845 / 0.874 | 334/391/379 | 29 | 0.994 / 0.991 | −0.004 |
| 9 | cosmic | — | 1 | 953 | 1 210 → 1 155 | 0.027 / 0.023 / 0.064 | 6/9/6 | 1 | 0.995 / 0.995 | +0.022 |
| 10 | cosmic ×2 | — | 2 | 687 | 1 984 → 1 806 | 0.143 / 0.122 / 0.276 | 11/16/9 | 2 | 0.984 / 0.984 | +0.014 |

Per anode (tiled → surviving blobs, `ProjectionDeghosting` first-pass tags): the only anode wiped
out is **event 1 anode 10: 15 830 → 0, all 15 830 tagged by the first `ProjectionDeghosting`**
(the 0° slab is one 2-D component per view, `c2b.size()=1`, and the pass tags the whole 3-D
cluster); its 8.5 × 10⁶ e of true charge (73 % of the event) are gone before charge solving.
Every other iso anode loses 0–10 % of its blobs to that pass (e.g. evt 2 a2 367/4970, evt 7 a9
444/6185) and the ghosts that remain — 43–97 % of the survivors — go through the three
`InSliceDeghosting` rounds untouched. Several anodes carry blobs on the *opposite* face
(evt 4 a5, 6 a0, 7 a11, 8 a9): pure projective ghosts built from the wrapped U/V channels plus a
chance W coincidence.

### 5.2 Per slice (the trigger calibration, doc 01 §4.6)

Over the 563 iso slices and 1 640 cosmic slices holding ≥ 1 surviving blob:

| | iso slices | cosmic slices |
|---|---|---|
| max strip width in slice, median / p90 / p99 / max (wires) | 31 / 90 / 459 / 628 | 7 / 11 / 14 / 16 |
| blobs per slice, median / p90 / p99 / max | 10 / 101 / 818 / 2 925 | 1 / 4 / 6 / 8 |
| slices with max width > 30 wires | 52.6 % | 0 |
| slices with max width > 50 wires | 27.9 % | 0 |
| slices with > 100 blobs | 10.1 % | 0 |

Fraction of an event's true charge sitting in "wide" slices (max strip > 30 wires): 100 % at
0–1° (events 1, 2, 6), 99.5–99.7 % at 2° (3, 7), 75–97 % at 5° (4, 8), 0.6 % at 10° (5), 0 on the
cosmics. So **`T_wires = 30` separates the isochronous regime from ordinary tracks with no false
positive on this sample (cosmic maximum 16)**, and `T_cells = 100` blobs/slice is a strict
sub-trigger (10 % of iso slices, cosmic maximum 8); the 10° track is not isochronous by either
measure, i.e. the regime ends between 5° and 10° for 3 m tracks.

Plots: `02_tables/iso_baseline_vs_angle.png` (ghost fractions and blob counts vs θ, cosmics as
dashed lines) and `02_tables/iso_baseline_strip_width.png` (max strip width per slice, iso vs
cosmic, per plane).

## 6. What this means for the FM stage (doc 01 §4.6, §7)

1. **The deficit is real and large.** On isochronous slices 43–100 % of the blobs the current chain
   keeps are ghosts (cosmics: 2–12 %), and the chain can delete a whole true track (event 1: 73 %
   of the charge, `ProjectionDeghosting` on a 15 830-blob slab). Charge solving itself is unbiased
   on what survives (Σq_reco/Σq_true within ±2 % on every event) — the problem is *which* cells
   exist, not how much charge they get. This is the number the sub-blob + GNN stage must beat, and
   the metric set must include the **captured-charge fraction**, not only ghost purity, or a stage
   that deletes tracks would score well.
2. **The blobs are giant.** Median max-strip 31 wires and up to 628 wires per slice; a cell-level
   classifier is meaningless at this granularity. Sub-blob generation (doc 01 §4.6, GNN doc §4:
   re-tiling the slab at coarse wire groups) is a prerequisite, not an optimisation.
3. **Triggers.** `T_wires 30` / `T_cells 100` are validated on this sample with zero cosmic false
   positives; the FM stage can be gated on them so ordinary events cost nothing.
4. **Truth labels** must use a relative threshold (§4.1): the exact-zero ghost fraction moves by 6
   points across the 1 µs plateau of the time offset.
5. **The join target exists** (`ctpc_a<N>f<F>p<P>` in the pctree, 90 clouds on event 1), so F4 can
   start on this chain as is.
6. Two loss channels to keep in view: event 7 has 19 % of its true charge outside every tiled
   blob (`tru0` captured 0.809 — not deghosting; likely the tracks crossing the anode y/z gaps or
   below the slicing threshold; not investigated here), and the opposite-face ghosts (§5.1) are a
   FD-HD-specific ghost class PDHD never showed (its wall faces do not image).

## 7. Open items

- Drift speed 1.6 vs 1.565 (field file): a 2 % x scale, irrelevant for the baseline, to be settled
  before any x-resolution study (owner call; would move the Reframer `tbin`).
- The NF is still MicroBooNE's algorithm with corrected numbers; a clean sim never exercises its
  chirp/noisy logic, data would.
- `bee_detector` is `protodunehd` (there is no FD-HD Bee geometry); the Bee zips draw on the
  PDHD frame.
- Group-stage visitors dropped: none (§3.1).
- Why `ProjectionDeghosting` tags the entire event-1 anode-10 slab (and 0 of event 2's 0° slab):
  not analysed; it is the single biggest item in §5 and belongs in the sub-blob design.
- Event 7's 19 % of true charge outside every tiled blob (§6.6).
- Operational: the `wire-cell` on PATH is `toolkit/build/apps/wire-cell` and loads plugins from
  `build/` (rpath), so a `./wcb build` while jobs run makes them fail with "invalid ELF header"
  (three ladder points of §4.1 were re-run for that reason). Never build during a run.
- `wcfm-w3-a`/`wcfm-w3-b` snapshots are void (§3.1); the valid gate is `wcfm-w3-c`/`wcfm-w3-d`.

## 8. Files

Toolkit (`69515f37`): `cfg/pgrapher/experiment/dune10kt-1x2x6/{params.jsonnet,STATUS.md,wct-sim-check.jsonnet}`,
`sio/src/FrameFileSource.cxx`, `sio/test/doctest_framefile_empty_mask.cxx`, `img/src/BlobDepoFill.cxx`,
`img/test/doctest_blobdepofill_faces.cxx`, `aux/src/ClusterArrays.cxx`, `sio/test/doctest_clusterfile_noblobs.cxx`.
wcp (the commit adding this doc): `wcfm/{wcfm_params,wct-sim-iso-track-nf-sp,img,wct-img-all,clus,wct-clustering}.jsonnet`,
`wcfm/{_runlib,run_sim_evt,run_img_evt,run_clus_evt}.sh`, `wcfm/gen_iso_tracks.py`,
`wcfm/scripts/{iso_baseline.py,scan_time_offset.sh}`, `wcfm/events/*.json`, `wcfm/abtest_events.txt`,
`wcfm/docs/02_*.md`, `wcfm/docs/02_tables/`. Outputs stay in `wcfm/work/` (not committed) and
`abtest/snap/wcfm-w3-{a,b}/`.

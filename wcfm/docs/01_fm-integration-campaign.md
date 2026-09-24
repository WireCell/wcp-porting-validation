# wcfm doc 01 — integrating the Wire-Cell foundation model into the toolkit: campaign design

Written 2026-09-24. Design only: no code, no config and no production default changes in this doc.
It records what exists on the model side and on the toolkit side, the owner's four decisions, the
architecture they imply, the memory plan, and the ordered work for the next sessions. Everything the
design rests on was checked against the files cited; the checks are in §0 so they can be re-run.

## 0. Repro / provenance

Trees and revisions read:

| Tree | Path | Revision |
|---|---|---|
| toolkit (`apply-pointcloud`) | `/home/xqian/toolkit-dev/toolkit` | `b5a897f6` |
| wcp-porting-img (`main`) | `/home/xqian/toolkit-dev/wcp-porting-img` | `da2c207d` (remote head) |
| FM workspace | `/home/xqian/work/scratch_wcgpu1/toolkit-dev/WC_FM_DINO` (docs 01–39, `dino/`, `models/`, `loader/`, `sdcc/`) | working tree |
| FM sim workspace | `/home/xqian/work/WC_FM_Sim` (via `WC_FM_DINO/WC_FM_Sim`) | working tree |
| PoLAr-MAE teacher repo | `/home/xqian/work/scratch_wcgpu1/toolkit-dev/Haiwang/PoLAr-MAE` | working tree |
| Thoughts | `/home/xqian/work/scratch_wcgpu1/toolkit-dev/wc-pr-ml-thoughts` (`3e27c43`): `GNN_Blob_Deghosting_Design.md`, `WC-PR-ML-Ideas.md`, `projective_readout/01–05`, `docs/01–09`, `WC_FM_Status.pdf`, `Wire-Cell_PR_AI_ideas.pdf`, `Wire-Cell-AI.pdf`, `main.pdf` (technote, 116 p), `slides.pdf`, `assets/20260618_MV_NPML_WireCellFM.pdf` | — |
| LArSoft-side FM configs | `/home/xqian/cffm-if/dune10kt-1x2x6/`, `/home/xqian/fdhd_dev/srcs/larwirecell/larwirecell/aiml/Labelling2D.cxx` | working tree |

Checks behind the facts of §2 (all read-only, run from `toolkit/`):

```
# workspace wires file: anode x positions, wires per plane per face, channel ranges (§2.2)
python3 - <<'EOF'
import bz2, json
st=json.load(bz2.open('/home/xqian/toolkit-dev/wire-cell-data/dune10kt-1x2x6-wires-larsoft-v1.json.bz2'))['Store']
A=[a['Anode'] for a in st['anodes']]; F=[f['Face'] for f in st['faces']]; P=[p['Plane'] for p in st['planes']]
W=[w['Wire'] for w in st['wires']]; X=[p['Point'] for p in st['points']]
for a in A[:2]+A[-1:]:
    for fi in a['faces']:
        f=F[fi]; xs=[]; npl=[]; chs=[]
        for pi in f['planes']:
            p=P[pi]; w=W[p['wires'][0]]; npl.append(len(p['wires'])); xs.append(round(X[w['tail']]['x'],1))
            cs=[W[i]['channel'] for i in p['wires']]; chs.append((min(cs),max(cs)))
        print('anode',a['ident'],'face',f['ident'],'plane x [mm]',xs,'nwires',npl,'chan range',chs)
EOF
# the two centerline definitions (§2.2)
grep -n 'local centerline' cfg/pgrapher/experiment/dune10kt-1x2x6/params.jsonnet cfg/pgrapher/experiment/dune10kt-1x2x6/simparams.jsonnet
# torch tensor conversion is 4-D only; downsample averages (§2.3)
grep -n 'dim() != 4' pytorch/src/Util.cxx; sed -n 11,19p util/src/Array.cxx
# which plugin packages are built here (§2.3)
python3 -c "exec(open('build/c4che/_cache.py').read()); print(SUBDIRS)"
# ctpc arrays (§2.3)
grep -n 'ds.add("' clus/src/PointTreeBuilding.cxx | sed -n 1,8p
# FM input scale and rebin on the LArSoft side (§2.1)
grep -n frame_scale /home/xqian/cffm-if/dune10kt-1x2x6/wcls-sim-drift-simchannel-nf-sp.jsonnet /home/xqian/cffm-if/dune10kt-1x2x6/wcls-labelling2d_sep.jsonnet
grep -n rebin_time_tick /home/xqian/cffm-if/dune10kt-1x2x6/wcls-labelling2d_sep.jsonnet
# FM normalisation constants and the deployed-feature width (§2.1)
sed -n 15,19p /home/xqian/work/scratch_wcgpu1/toolkit-dev/WC_FM_DINO/dino/transforms.py
sed -n 474,482p /home/xqian/work/scratch_wcgpu1/toolkit-dev/WC_FM_DINO/dino/train_distill.py
ls /home/xqian/work/scratch_wcgpu1/toolkit-dev/WC_FM_DINO/polarmae_checkpoints/
```

Outputs on 2026-09-24: 12 anodes, every anode's faces at x = ±(39.5, 34.8, 30.0) mm, wires per face
U 1149 / V 1148 / W 480, W channels 1600–2079 on face 1 and 2080–2559 on face 0 (anode 0);
`params.jsonnet:60` `centerline = sign*apa_cpa`, `simparams.jsonnet:73` `centerline = 0`;
`SUBDIRS` has `pytorch`, `hio`, `cuda` but not `spng`, `triton`, `zio`; `frame_scale` 0.005 (SP save)
and 50 (labelling read), `rebin_time_tick: 4`; `VIEW_NORM` U (2.77, 144977), V (2.97, 157944),
W (3.75, 83861.2); local checkpoints are the three `pm4_{U,V,W}_capstone_step80000.ckpt` teachers only.

## 1. Goal, scope, non-goals

**Goal.** Make the trained 2D foundation model (FM) available inside the toolkit as a *per-pixel
feature substrate*: for every active (channel, time-slice) pixel of every wire plane, a feature vector
that later stages can read through the same data structures they already use (the PC-tree). The first
consumer is **isochronous deghosting**: a track running parallel to the anode puts all its charge into
one or a few time slices, tiling then produces one giant blob spanning O(100–1000) wires per plane, and
none of the existing discriminants (adjacent-slice connectivity in `ChargeSolving`'s `uboone` weighting,
`InSliceDeghosting`, `ProjectionDeghosting`) can act. The plan is to break such a blob into sub-blobs,
deghost them using the FM features (the GNN of `GNN_Blob_Deghosting_Design.md`), and run pattern
recognition on the result.

**Scope of this campaign.**
1. A standalone imaging/clustering chain for the geometry the FM was trained on, DUNE FD-HD
   `dune10kt-1x2x6` ("workspace"), with an isochronous track gun and truth labels for ghosts.
2. FM inference as a toolkit stage, with two model artifacts (dense student on CPU/GPU, sparse
   student on GPU) behind one data contract.
3. Features attached to the PC-tree and reachable per blob.
4. The Phase-0 measurement the deghosting GNN needs: do FM features separate true from ghost
   sub-blobs on isochronous slices better than charge alone?
5. SBND afterwards, with the same contract.

**Non-goals here.** Training or changing the FM itself; the GNN design (it is in the GNN doc and is
referenced, not repeated); any change to PDHD/PDVD/SBND production. Every toolkit change of this
campaign ships behind a default-OFF knob with a byte-identical gate label, per `toolkit-dev/CLAUDE.md`
§1/§4. The DNN-ROI production code path (`pytorch/src/DNNROIFinding.cxx`, `Util.cxx`) is not edited; new
helpers are duplicated (rule M10).

## 2. State of play

### 2.1 The model (what "the FM" is today)

The project moved from DINO to masked reconstruction and then to distillation; `WC_FM_DINO/docs/README.md`
still describes the DINO era and indexes only docs 01–30. The current production recipe is in docs
33–38 and the technote:

| Item | Value | Source |
|---|---|---|
| Teacher | PoLAr-MAE ViT-S point transformer, 21.8M params, **one per plane** (`pm4_{U,V,W}`), semi-supervised (track/shower aux head); per-token 643-d features kNN-upsampled to "covered" pixels, PCA-64 as the distillation target; its output depends on batch composition (`MaskedLayerNorm`), so the teacher is never deployed | technote p.13, 35, 41–42, 72; doc 32 §5.5 |
| Students (deployed) | unified plane-blind: sparse "CAP" attention U-Net `kd_uni_a1_inf01` (7.2M, WarpConvNet + flash-attn, **CUDA-only**) and dense MobileNetV3-Large U-Net `kd_uni_mbv3_a1_inf01` (5.2M, torch + torchvision only, **CPU-capable**) | doc 37; `sdcc/config_kd_uni_*.json`; `models/minkunet_attention.py`, `models/mobilenetv3_unet.py`, `models/dense_mae_adapter.py` |
| **Deployed feature** | **128-d student trunk output at every active pixel**; the `Linear(128→64)` heads exist only for the KD loss and are dropped (`train_distill.py:474-482`). The "64-d" of the slides is the teacher-side PCA target, not the deployed width | doc 37; technote Fig. 21 |
| Input pixel | (anode-local channel, time slice), slice = 4 raw ticks = 2 µs; canvases U/V 800×1500, W 960×1500; sparse: only active pixels exist, N ≈ 11–62k per plane, median ≈ 3.2k | doc 19 §1, doc 17 §1, `loader/apa_sparse_dataset.py:19-23,184-189` |
| Pixel value | the **signal-processed `gauss` trace summed over the 4 ticks**, scaled 0.005 (SP save) × 50 (labelling read) = 0.25·Σ₄ in WCT-native units, i.e. numerically the 4-tick **mean** of the native `gauss` charge. To be re-verified before the C++ packing is frozen (§7 F2) | `Labelling2D.cxx` (`rebin_time_tick 4`), `cffm-if/.../wcls-sim-drift-simchannel-nf-sp.jsonnet:123`, `wcls-labelling2d_sep.jsonnet:68,109` |
| Normalisation | per plane, `FeatureLogTransform`: y = 2·(log10(x+m) − log10 m)/(log10(M+m) − log10 m) − 1, no clip; `VIEW_NORM` U (2.77, 144977), V (2.97, 157944), W (3.75, 83861.2). This is the only place plane identity enters the deployed model | `dino/transforms.py:15-51` |
| Channel layout | U = channels [0,800), V = [800,1600), W = [1600,2560) per APA, each **rebased to 0** in the image. U/V are wrapped (1149/1148 wires on 800 channels, both faces). The W image is **two faces side by side in channel order**: columns 0–479 = channels 1600–2079 = WCT face 1 (−x), columns 480–959 = channels 2080–2559 = WCT face 0 (+x). The join at column 480 is a physical discontinuity the model was trained across | `loader/wire_geometry.py`; §0 wires-file check |
| Dense-student input | per event `[1, 2, h, w]` = (normalised charge, 0/1 active mask) on the event's **tight bounding box** (floor 64); output `[1, 128, h, w]`, gathered at active pixels; `UpBlock.forward` branches on tensor shape (`mobilenetv3_unet.py:50`) | `dense_mae_adapter.py:170-224` |
| Training data | DUNE FD-HD 1x2x6 sim through LArSoft (`WC_FM_Sim/scripts/run_chain.sh`: GENIE → G4 → `wcls_sim_sp.fcl` → `wcls-labelling2d_sep.fcl` → dense (2560,1500) HDF5 per anode → sparse packs), anode 0 only, νμ/νe, 10⁵–10⁶ events; the `_20k` truth packs from `WC_FM_Sim` runs 990001–990004. Which field-response / noise files that chain used is **not verified** against the toolkit's 2026-05-18 `dune10kt-1x2x6` swap | doc 39; `WC_FM_Sim/docs/00,03,07` |
| Benchmarks (W plane, L40S / EPYC 9355) | MBV3: 17.9 ms/plane GPU, 238 → 54 ms/plane CPU at 1 → 16 threads, 0.05–0.9 GB; CAP: 5.6 ms/plane GPU at batch 1, 17–320 MiB, no CPU path; teacher 31 ms GPU / 811 ms CPU-shim. Doc 17's 6.4 ms figures are for an obsolete 0.47M backbone | doc 33 Table 2; technote Table 21 |
| Quality | probes on simulation, per plane: pid7 linear-F1 ≈ 0.55–0.57, vertex AP ≈ 0.66–0.69, instance k-NN margin ≈ +0.41–0.45, both unified students at parity with per-plane students and within ~0.05 of the teacher | doc 37 §5; technote Tables 16, 18, 19 |
| Export | **none**: no TorchScript, ONNX or `torch.export` anywhere in `WC_FM_DINO` or `PoLAr-MAE`; doc 32 "kd_export" exports teacher *features*, not a model | grep |
| Checkpoints | student checkpoints exist **only on SDCC** (`CONDOR_OUT/kd_campaign/checkpoints/...`); local disk holds only the three pm4 teachers | doc 29, 37; §0 `ls` |
| Environments | training venv torch 2.10 + cu128 + warpconvnet 1.7.8 + flash-attn 2.8.3; the toolkit's direnv Python is torch 2.5.1 + cu121; the toolkit's libtorch shim is **torch 2.8** (`/nfs/data/1/hnam/demo/pytorch/torch`) | `WC_FM_DINO/.envrc`, `docs/README.md`; `build/config.log` |

Three facts about the model that shape the design:

- **The FM has never seen a ghost and has no cross-plane objective.** Every pixel in a 2D view is real;
  ghosts exist only as wrong cross-view associations. The unified student "shares no supervision across
  planes on matched points, so the geometric consistency objective itself remains the open lever"
  (technote p.90; cross-plane consistency is queued as item #8 of doc 35 and #5 of doc 34). Whether the
  per-plane descriptors of three pixels agree more when they come from the same deposit than when they
  merely cross is therefore an *open measurement* (the Phase-0 probe of §7 F5), not a premise.
- **Dead channels are absent from the FM pipeline** (no mask anywhere in the loaders), and the FM
  input is pre-DNN-ROI-free `gauss` from a simulation with the sim's own noise model. Data/MC and
  detector-condition robustness are untested.
- **The features are pixel-local in time at 4-tick resolution** — exactly `MaskSlices`' production
  `tick_span`, so pixel k of the FM is slice k of imaging when the frame time origin is carried through
  unchanged (§2.3).

### 2.2 The workspace geometry in the toolkit (`cfg/pgrapher/experiment/dune10kt-1x2x6/`)

- 12 anodes, 2 faces each, U/V/W; `daq.nticks 6000` at 0.5 µs; drift 1.6 mm/µs (base params);
  `wires: dune10kt-1x2x6-wires-larsoft-v1.json.bz2`; field `dune-garfield-1d565.json.bz2` and noise
  `protodunehd-noise-spectra-14mVfC-v1.json.bz2` (PDHD stand-ins, swapped in 2026-05-18, `STATUS.md` §3).
- Every entry config compiles (`STATUS.md` §2); there are sim, NF and SP jobs (`wct-sim-ideal-*`,
  `wcls-*`) but **no imaging, clustering or pattern-recognition job** for this geometry.
- **Geometry bug.** The wires file puts all 12 anodes at x ≈ 0 with both faces live (§0 check: planes at
  ±39.5/34.8/30.0 mm on every anode), i.e. one central anode row drifting both ways to cathodes at
  ±3.63 m. `params.jsonnet:60` sets each anode's centerline to `sign*apa_cpa` = ±3.63 m, which does not
  match the wires; only `simparams.jsonnet:73` (`centerline = 0`) does. `STATUS.md` §1 ("2 APA columns")
  repeats the wrong picture. `wcls-sp.jsonnet` and `wcls-nf-sp.jsonnet` import both files. Consequence
  for this campaign: every volume-dependent tool (imaging `DetectorVolumes`, clustering fiducial volume,
  x from drift time) must be built from the `simparams` volumes or a corrected `params`, and PDHD's
  null-face restore (`pdhd/wct-img-all.jsonnet:35-37`) must **not** be copied — both faces are live.
- No workspace depos or frames exist in WCT format; the FM training frames live in the LArSoft HDF5
  chain.

### 2.3 The toolkit plumbing that the integration reuses

| Piece | Where | What matters for the FM |
|---|---|---|
| Inference service | `pytorch/src/TorchService.cxx` (`ITensorForward`; `model` via `WIRECELL_PATH`, `device cpu|gpu|gpuN`; `torch::jit::load`; `TorchSemaphore` count 1 — the `concurrency` key in jsonnet is never read) | reuse as is |
| Tensor conversion | `pytorch/src/Util.cxx:22-74` `to_itensor`/`from_itensor`: **4-D float only**, copies from `ten[0][0]` assuming contiguity, CUDA:0 hard-coded (`pytorch/docs/examination/bugs.org` H1/H2) | not usable for a `[1,128,h,w]` output; duplicate new helpers |
| Packing template | `pytorch/src/DNNROIFinding.cxx`: rows = `Aux::plane_channels(anode, plane)` (face 0 then face 1), `Aux::fill` → `Array::downsample(…, tick_per_slice)` (**averages**, `util/src/Array.cxx:11-19`) → `{1, ntags, nchan, nticks/4}`; reads only `output[0][0]`; chunks the channel axis without overlap | copy the frame → dense-array front end, not the back end |
| Node interface | `iface/inc/WireCellIface/IFrameTensorSet.h` (`IFunctionNode<IFrame, ITensorSet>`), used by `aux/inc/WireCellAux/FrameTensor.h` | the FM extractor's base |
| Tensor files | `sio` `TensorFileSink` (`outname` with `%d`, container by suffix: `.tar.zst` recommended — 27× faster than bz2 at equal size, `sigproc/docs/nfsp-dnnroi-perf-round1.md`) and `TensorFileSource` | the feature sidecar |
| Dtypes | `util/src/Dtype.cxx`, `aux/src/TensorDMdataset.cxx:84-102`, `hio/src/HIO.cxx:14-32`: `i1..i8, u1..u8, f4, f8, c8, c16`; **no `f2`/float16** | f16 on disk needs a new dtype or a `u2` bit-cast |
| Slicing | `img/src/MaskSlice.cxx`: slice k = ticks [k·span, (k+1)·span) from `tbin 0`, `tick_span 4` in pdhd/pdvd production (`img.jsonnet`), `nthreshold 1e-6` in production | FM pixel k == slice k |
| Pixel clouds | `clus/src/PointTreeBuilding.cxx:272-350` `add_ctpc`: per anode/face/plane `ctpc_a{apa}f{face}p{U,V,W}` with `x, y, charge, charge_err, cident, wind, slice_index` (ticks), one point per wire with activity; read by `Facade_Grouping` (`kd2d`, `is_good_point`, `get_closest_points`) | the join target |
| Point-cloud arrays | `util/inc/WireCellUtil/PointCloudArray.h`: N-D with `shape[0]` = points (`(N,128)` fine); `Dataset::add` requires equal major size; TensorDM round-trips N-D (`aux/src/TensorDMpointtree.cxx`) but **silently drops a key not present in every same-named PC** (l.88-101) | the feature array must be added to every `ctpc_*` |
| Blob footprint | `IBlob::shape().strips()[2..4]` = face-local wire index ranges `[lo, hi)`; wire → channel via `IWirePlane::wires()[wind]->channel()`; cluster-graph b–w–c edges; existing per-view projection `img/src/Projection2D.cxx:95-215` | blob → feature rows |
| Truth | `img/src/BlobDepoFill.cxx` (+ `img/docs/BlobDepoFill.org`): replaces blob charge with integrated true depo charge, exactly 0 for a blob with no depo; `img/test/depo-ssi-viz.jsonnet` splices a catcher between `BlobGrouping` and `ChargeSolving` | ghost labels |
| Re-tiling | `clus/src/retile_cluster.cxx`, `clustering_retile.cxx` (`ClusteringRetile`), `improvecluster_{1,2}.cxx` (ports of `ImprovePR3DCluster`) | the only blob re-imaging hook; nothing in `img/` splits blobs |
| Build | `./wcb configure … --with-cuda=/usr/local/cuda-12.5 --with-libtorch=…/libtorch-shim` (torch 2.8); built plugins include `pytorch`, `hio`, `cuda`; **`spng`, `triton`, `zio` are not built** | the Triton path needs a build item |
| Runner templates | `wcp-porting-img/pdhd_sim/wct-sim-check-track.jsonnet` + `run_sim_track.sh` (TrackDepos, inline `FrameFileSink`); in-tree `pdhd/{img,clus,wct-img-all,wct-clustering}.jsonnet`; `pdvd/run_{img,clus,pr}_evt.sh`, `_runlib.sh`; `abtest/{run_events.sh, ab_compare.sh, hash_archive.py}` | clone for the workspace |

## 3. Owner decisions (2026-09-24)

Asked and answered before this doc was written:

1. **Simulation source for the workspace chain:** *WCT standalone first, LArSoft later.* Start from the
   in-tree `dune10kt-1x2x6` sim with `TrackDepos` (an isochronous track gun, no LArSoft); then tap the
   `WC_FM_Sim` LArSoft chain's SP frames into a `FrameFileSink` for physics events, so the FM also sees
   the frames it was trained on.
2. **FM artifact for the first integration:** *both from the start.* The dense MBV3 unified student as
   TorchScript through `pytorch/TorchService` (CPU and GPU), and the sparse CAP student through a Triton
   Python backend (GPU), in parallel.
3. **What to store per active pixel:** *the full 128-d student trunk, f32 in memory*, sparse (active
   pixels only), f16 bits on disk.
4. **Where FM inference runs:** *a separate stage between SP and imaging*: a new job/runner reads the SP
   frames, runs the FM per plane, writes a sparse feature sidecar per anode; imaging is untouched; the
   clustering job's `PointTreeBuilding` joins the features onto the `ctpc_*` clouds by (channel, slice).

## 4. Architecture

```
SP frames ──► [FM stage: run_fm_evt.sh]  ──► fm-features-anode{N}.tar.zst   (sidecar, §4.1)
   │             FMFeatureExtract (IFrameTensorSet) ─► ITensorForward            │
   │                 TorchService (MBV3 TorchScript, cpu|gpu)                    │
   │                 TritonService (CAP, Python backend, gpu)                    │
   ▼                                                                             ▼
imaging (unchanged) ──► clusters-apa-*.tar.gz ──► clustering: PointTreeBuilding ──► ctpc_* + fm_feat
                                                     │  (join by cident, slice; default OFF)
                                                     ▼
                                          Facade::Grouping::fm_feature(...)  ──► blob / sub-blob descriptors
                                                     │
                                                     ▼
                                          iso trigger → sub-blob re-tiling → [GNN deghost, later] → PR
```

### 4.1 The feature sidecar contract (the interface everything else depends on)

One `ITensorSet` per anode per plane (three sets per anode, each spanning both faces the way the FM
image does), written by `TensorFileSink` as `work/<run6>_<evt>/fm-features-anode{N}.tar.zst` and read
back by `TensorFileSource`:

| Tensor | Shape / dtype | Content |
|---|---|---|
| `coords` | `(N, 2)` `i4` | column 0 = WCT channel ident; column 1 = slice index k (ticks [4k, 4k+4) from the frame's `tbin 0`), one row per active FM pixel |
| `feat` | `(N, 128)` `f4` in memory | the student trunk output at that pixel; on disk stored as IEEE half bits (see below) |

Set metadata (JSON): `model` (file name), `model_sha256`, `arch` (`mbv3_uni` / `cap_uni`), `plane`
(0/1/2), `anode` ident, `face_layout` (the channel → image-column map used: `"wan"`), `tick_span` (4),
`input_tag` (`gauss%d`), `input_scale` (0.25 pending F2), `view_norm` `[m, M]`, `frame_time`, `frame_tick`,
`bbox` `[ch0, ch1, t0, t1]` and `n_active`. A consumer must refuse a sidecar whose `tick_span` or
`frame_time` disagrees with the frame it is joining to.

**f16 on disk.** The toolkit has no half-float dtype. Two options, decided at F3: (a) add `f2` to
`util/src/Dtype.cxx`, `TensorDMdataset::as_array` and the numpy writer (small, generic, but touches
`util`); (b) write `feat_half (N,128) u2` bit-cast and convert on read in the FM reader only. (b) is
default-OFF-safe and local; (a) is cleaner. Start with (b); promote to (a) if a second consumer appears.

**Why this contract.** It is model-agnostic (MBV3 and CAP produce the same tensors), detector-agnostic
(channel idents + slice index), sparse by construction, and it is the same layout the FM's own packs and
`dino/model.py::match_and_gather` use (CSR of `(channel, tick)` coordinates plus features), so the
Python oracle of F2 is a direct reuse. It carries the provenance a byte-identical gate needs.

### 4.2 Frame → FM input packing: `FMFeatureExtract`

A new `IFrameTensorSet` component in `pytorch/` (`pytorch/src/FMFeatureExtract.cxx`, class
`Pytorch::FMFeatureExtract`, `WIRECELL_FACTORY`, `IConfigurable`), one instance per anode per plane,
holding a named `ITensorForward` (`forward: "TorchService:fm"` or `"TritonService:fm"`), following the
DNN-ROI node's front end:

1. **Rows.** Channels of `(anode, plane)` in WAN order via `Aux::plane_channels` — this is what the FM's
   channel rebasing assumes: U/V rows 0–799 (wrapped, both faces), W rows 0–479 = face 1, 480–959 =
   face 0 (§2.1). A `channel_order` knob records the choice; the FM training layout is the default.
2. **Time.** Traces tagged `input_tag` (`gauss%d` after SP; the L1SP/DNN-ROI question is §9) are
   `Aux::fill`ed from `tbin 0` and **summed** over `tick_span = 4` ticks (not `Array::downsample`, which
   averages), then multiplied by `input_scale` (0.25 pending F2). The active mask is `q > 0` (the packs
   keep only non-zero pixels; the exact threshold is fixed in F2 against the pack builder).
3. **Normalisation.** `VIEW_NORM` log map with the plane's `(m, M)` from config (defaults = training
   constants), applied to active pixels only; empty pixels stay 0 as in `dense_mae_adapter.py`.
4. **Canvas.** The tight bounding box of active pixels, padded to a multiple of `bbox_pad` (32, so
   `UpBlock`'s shape branch is never taken) with a floor of 64; input `[1, 2, h, w]` = (normalised
   charge, mask). When `h·w > max_dense_pixels` the time axis is tiled with an overlap `halo` (default
   64 slices) and the halo columns are discarded after gathering, so a seam never enters the stored
   features. Batch is always 1 (one plane per call).
5. **Forward and gather.** Call `forward()`; the reply is `[1, 128, h, w]`. Gather the 128 values at
   each active pixel into `feat`, write `coords`, attach metadata. The dense output is released before
   the next plane.
6. **Tensor helpers.** New `Pytorch::fm_to_itensor` / `fm_from_itensor` in a new `FMUtil.cxx`: any
   dimensionality, `.contiguous()`, device taken from the `TorchContext` rather than CUDA:0. The
   DNN-ROI-consumed `Util.cxx` is not edited (M10); the DNN-ROI gate stays trivially green.

Job/config: `wcfm/wct-fm-features.jsonnet` (top-level function: `input_prefix`, `anode_indices`,
`output_dir`, `fm_model`, `fm_device`, `fm_forward` selecting `TorchService`/`TritonService`) —
`FrameFileSource → [FMFeatureExtract × 3 planes] → TensorSetFanin → TensorFileSink` per anode — and
`wcfm/run_fm_evt.sh` (`-D cpu|gpu`, `-M model`, `WCFM_MAX_JOBS`), modelled on `pdhd/run_nf_sp_dnnroi_evt.sh`
including its VmHWM and `nvidia-smi` recording.

### 4.3 Model artifacts

- **MBV3 (TorchScript, CPU + GPU).** Export with `torch.jit.script` (not `trace`: the shape branch) from
  an environment whose torch is ≤ the toolkit's 2.8 runtime (the toolkit direnv 2.5.1 or a 2.8 venv);
  the export script lives in `WC_FM_DINO/sdcc/export_mbv3_ts.py` (new) and records the checkpoint sha,
  torch/torchvision versions and the `VIEW_NORM` constants into the model's extra files. Parity test:
  eager vs scripted on 20 events of the `_20k` packs, max |Δfeat| and cosine; the scripted model then
  goes to `wire-cell-data/fm/dune10kt-1x2x6/<name>.ts` (the DNN-ROI convention).
- **CAP (Triton, GPU).** WarpConvNet and flash-attn are not scriptable; the realistic path is a Triton
  model with a Python backend that takes `coords (N,2) i4` + `logq (N,1) f4` and returns `feat (N,128) f4`
  — i.e. the sparse contract directly, no dense canvas. Toolkit side: build `triton/` (`--with-triton`,
  gRPC/protobuf/triton-client deps; `build/config.log` shows the client libs are currently absent) and
  add a sparse packer `FMFeatureExtractSparse` (same node interface, same metadata, packs
  `(coords, logq)` instead of a canvas). Deliverables and gates are the same; only the packer and the
  forward service differ, which is what keeps the two artifacts interchangeable in jsonnet.
- Both artifacts need the SDCC checkpoints copied to this machine first (F1); until then nothing
  below the Python oracle can run.

### 4.4 Joining features onto the PC-tree (clustering, default OFF)

`clus/src/PointTreeBuilding.cxx` gains an optional third input port (or a configured file path read
through `TensorFileSource` at configure time — decided at F4 by what the Pgrapher wiring allows with the
least change) carrying the sidecar. With no input the node is byte-identical to today. With input:

- after `add_ctpc`, for every `ctpc_a{apa}f{face}p{P}` dataset, add `fm_feat (N, 128) f4`, row i = the
  sidecar row whose `coords` equal (`cident[i]`, `slice_index[i] / tick_span`); a wrapped U/V channel
  shared by two wires gets the same row on both — that *is* the projective ambiguity, and the sub-blob
  stage resolves it geometrically, not here;
- unmatched pixels get a zero row and a count; `fm_match_rate` (matched / ctpc points) is logged and
  the job aborts below `fm_min_match_rate` (default 0.99: the FM saw the same `gauss` frame that
  imaging sliced, so a mismatch is a wiring error, not physics);
- a `fm_meta` entry in the grouping metadata carries the sidecar metadata (model sha etc.) so every
  downstream file and Bee dump is traceable to the model that produced its features;
- `Facade::Grouping` gains `fm_feature(face, plane, cident, slice)` and `fm_features(face, plane)`
  returning spans into the 2-D array, mirroring `get_closest_points`;
- doctests (`clus/test/doctest_fm_join.cxx`): a synthetic grouping + sidecar round trip through TensorDM
  (2-D array survives `as_tensors`/`as_pctree`), the every-node-has-the-key rule, the match-rate abort.

Why the ctpc and not the blob `3d` clouds: the ctpc is already per (channel, slice) pixel, it exists once
per plane rather than once per blob, `ClusteringRetile` re-samples blobs (which would drop per-blob
extras), and the blob-level access of §4.5 is a lookup, not a copy. The blob clouds stay untouched.

### 4.5 Blob-level access and the first ghost feature

For a blob (or sub-blob) on `face`, slice `s`, with wire ranges `[lo_P, hi_P)` per plane P:
wires → channels (`IWirePlane::wires()[w]->channel()`, dedup) → `fm_feature(face, P, c, s)` rows →
per-view pooled descriptor (mean, and attention-free max) of the 128-d rows, plus the per-view pixel
count and summed charge. From the three pooled descriptors, the **tri-view disagreement**
(pairwise cosine distance variance, `WC-PR-ML-Ideas.md` §9.1) is a zero-label, amplitude-free ghost
score available before any classifier is trained, and the per-view descriptors are the inputs of the
Phase-0 probe (§7 F5) and of the GNN's cell nodes later.

Implementation: a `clus` helper (`clus/inc/WireCellClus/FMBlobFeatures.h`) plus an `IEnsembleVisitor`
`ClusteringFMProbe` that, when enabled, dumps per blob `{anode, face, slice, u/v/w ranges, n_pix per
view, pooled features, disagreement, BlobDepoFill truth charge if the truth cluster file is given}` to
a `TensorFileSink` set. It writes; it does not alter the tree (no gate needed beyond "off == absent").

### 4.6 The isochronous path (pointer, not a redesign)

The mechanics live in `GNN_Blob_Deghosting_Design.md` (§1.2 failure mode, §3 graph, §4 hierarchy,
§7 truth). What this campaign commits to on the toolkit side, in order:

1. **Trigger**: per slice, the maximum strip width over planes (`hi − lo`) and the fine-cell count
   estimate; iso if width > `T_wires` (start 30) or cells > `T_cells` (start 10k). Non-iso slices bypass
   everything (zero cost on ordinary events).
2. **Sub-blob generation**: re-tile the blob's own activity at coarse wire groups
   (`RayGrid::make_blobs` on an `Activity` rebinned by k wires, restricted to the blob's strips;
   `retile_cluster.cxx` shows the per-cluster re-tiling pattern), then subdivide survivors — the
   L1 → L2 → L3 ladder of the GNN doc §4 with k ∈ {12–16, 3–4, 1–3}. Sub-blobs are ordinary `IBlob`s
   with strips, so §4.5 applies unchanged.
3. **Labels**: `BlobDepoFill` on the fine level in-job (W4) and the offline multi-level relabeler on
   saved depos (GNN doc §7.4), with the fine-level parity test between the two.
4. **Write-back**: first as a *soft* penalty — a per-blob weight from the sub-blob scores mapped into
   `ChargeSolving`'s regularisation where the `uboone` adjacency weighting is blind — then, once
   validated, hard removal of sub-blobs below threshold. Soft first because a miscalibrated score
   reweights, never deletes (`WC-PR-ML-Ideas.md` §3.2).

The GNN itself is Phase 1–2 of the GNN doc and starts only after the Phase-0 probe passes.

## 5. Memory footprint plan (owner concern 2)

**Where the bytes are.**

| Object | Size | Lifetime |
|---|---|---|
| Dense FM input `[1,2,h,w]` f32 | ≤ 2·960·1500·4 B = 11.5 MB | one plane, transient |
| Dense FM output `[1,128,h,w]` f32 | worst case 128·960·1500·4 B = **614 MB** per plane; typical bbox far smaller (median event ≈ 3k active pixels) | one plane, released after the gather |
| MBV3 activations | < 1 GB at batch 1 (technote p.46; 0.05 GB GPU at batch 1, doc 33) | during forward |
| Sparse store `feat (N,128)` f32 | 512 B per active pixel → 1.5–30 MB per plane for N = 3k–60k; ×3 planes ×12 anodes worst ≈ 1 GB, typical ≈ 50–100 MB per event | per event, in the sidecar |
| Sidecar on disk (f16) | half of the above; ≈ 5–50 MB per anode typical | files |
| Clustering growth | exactly N_ctpc × 512 B (the join copies rows); ctpc has one point per wire, so wrapped U/V duplicate rows ≈ ×1.4 | clustering job RSS |

**Controls, all knobs of `FMFeatureExtract` or the runner.**

- `max_dense_pixels` (default 1.5M = a full 960×1500 canvas): above it the time axis is tiled with
  `halo` overlap, bounding the dense output to `128 × max_dense_pixels × 4 B`; with 512k pixels that is
  256 MB per tile.
- `device cpu|gpu`: on GPU the dense output lives in VRAM (0.6 GB worst case + < 1 GB activations —
  fits any 24 GB card with room for two concurrent processes); on CPU it is host RSS.
- Torch intra-op threads (`torch.set_num_threads` via `OMP_NUM_THREADS` in the runner: 16 threads is the
  measured knee, 238 → 54 ms/plane) and `WCFM_MAX_JOBS` so that jobs × threads ≤ cores (CLAUDE.md §2).
- The stage is separate, so the SP job's 2.3 GB peak (`sigproc/docs/nfsp-dnnroi-perf-round0.md`) and
  the imaging job's peak are unchanged; the FM job's own budget is **≤ +1 GB RSS on CPU** over a bare
  `FrameFileSource → TensorFileSink` job, **≤ 2 GB VRAM on GPU**, measured as VmHWM / `nvidia-smi` traces
  in `run_fm_evt.sh` exactly as the DNN-ROI runner records them.
- Feature width stays 128 (decision 3). If a later measurement needs it smaller, the projection is
  put *inside* the exported model (a fixed linear layer fit offline), so the sidecar contract only
  changes its second dimension and nothing downstream changes shape assumptions (`feat.shape[1]` is
  read, never assumed).

**What is measured before anything is adopted (F3, W5):** per-event wall and RSS/VRAM of the FM job at
1/16 CPU threads and on GPU; sidecar size; clustering RSS with and without the join on the same events;
all on the workspace iso-gun set and later on the LArSoft physics set. Numbers go into doc 02.

## 6. Workspace chain plan (the next session: W1–W5)

All new files go to `wcp-porting-img/wcfm/` (runners, jsonnet forks, `work/`), following the `pdvd/`
layout (`work/<run6>_<evt>/`, `_runlib.sh` batching, `WCFM_MAX_JOBS`). Nothing under
`cfg/pgrapher/experiment/dune10kt-1x2x6/` is edited in W1–W5 except a note in `STATUS.md`; the
corrected volumes live in `wcfm/` first and are promoted in-tree once the chain runs.

- **W1 — geometry.** `wcfm/params_workspace.jsonnet`: `simparams`-style volumes (centerline 0, both
  faces live, all 12 anodes), drift speed and readout parameters in one place; record the `params.jsonnet:60`
  vs wires-file discrepancy in `STATUS.md` §4 (not a fix of `params.jsonnet` yet — `wcls-*` import it and
  are LArSoft-facing). Drift groups for clustering: all 12 anodes via face 0 (+x) and all 12 via face 1
  (−x), replacing PDHD's `group_defs` (`pdhd/wct-clustering.jsonnet:120-123`).
- **W2 — simulation.** `wcfm/wct-sim-iso-track-nf-sp.jsonnet`, a clone of
  `pdhd_sim/wct-sim-check-track.jsonnet` with the workspace params, `TrackDepos` (`sim.tracks`) → `Drifter`
  → per-anode `DepoTransform` → noise → `Digitizer` → NF → SP (`sp.jsonnet` of the workspace) →
  `FrameFileSink` with `masks: true` (imaging's `CMMModifier`/`FrameMasking` read the "bad" cmm) and a
  `NumpyDepoSaver` for the drifted depos (needed by W4 and the offline relabeler). A generator
  `wcfm/gen_iso_tracks.py` (from `pdhd_sim/generate_tracks.py`): angle to the anode plane 0–10°, random
  y–z direction and position, lengths 0.5–5 m, 1–4 track overlays, plus ordinary cosmics as controls;
  `run_sim_evt.sh` writes `work/<run6>_<evt>/sim-frames-anode{N}.tar.bz2` and `depos-anode{N}.npz`.
- **W3 — imaging and clustering.** `wcfm/{img,clus,wct-img-all,wct-clustering}.jsonnet` cloned from the
  in-tree `pdhd/` production files (not from `pdvd/wct-img-all.jsonnet`, which is a local fork): both faces
  tiled (no null-face restore), `nthreshold 1e-6`, `tick_span 4`, drift speed from W1 (PDHD's
  `clus.jsonnet:37` hard-codes 1.576), `PointTreeBuilding` + `MultiAlgBlobClustering` as in PDHD, no
  light/Q-L (`do_qlmatch=false`, and the PR runner's "no pctree without Q/L" guard disabled for wcfm);
  `run_img_evt.sh` and `run_clus_evt.sh` with the pdvd output names (`clusters-apa-anode{N}-ms-{active,masked}.tar.gz`,
  `pctree-evt<ID>.tar.gz`, `mabc-*.zip`). Determinism check: two runs under `setarch x86_64 -R`,
  `abtest/hash_archive.py` identical member hashes (the workspace has no A/B baseline yet; this is its
  first). A `wcfm/abtest_events.txt` manifest (6 iso-gun events + 2 cosmics) is the gate manifest for
  everything later; `abtest/run_events.sh` is extended to accept a `wcfm` detector.
- **W4 — truth.** A `BlobDepoFill` catcher spliced between `BlobGrouping` and `ChargeSolving` exactly as
  `img/test/depo-ssi-viz.jsonnet` does (`ClusterFanout` via `pg.insert_node`), fed by the drifted depos;
  writes `clusters-tru-anode{N}.tar.gz` next to the reconstructed cluster file; `time_offset` aligned
  with `DepoTransform.start_time` and checked with `wirecell-img paraview-depos/paraview-blobs`
  (`img/docs/BlobDepoFill.org`). A blob with fill charge exactly 0 is a ghost. Also exported: the
  per-anode masked-channel list (the dead-channel sidecar the GNN doc §7.3 needs).
- **W5 — baseline.** On the iso-gun set: per slice the maximum strip width, blob count, ghost fraction
  (from W4) and solved-charge error, all vs track angle to the anode and vs overlay multiplicity; the
  same on the cosmic controls. This is the deficit the sub-blob + GNN stage must beat and the trigger
  thresholds' calibration (§4.6). Written up as `wcfm/docs/02_workspace-chain-and-iso-baseline.md`
  with its Repro block.

Later, not next session: the LArSoft tap. In the `WC_FM_Sim` container, add a `FrameFileSink` (masks on)
after SP in `wcls-sim-drift-simchannel-nf-sp.jsonnet` (the `pdvd/wcls-nf-sp-out.jsonnet` pattern) and a
`wclsSimDepoSetSource`/depo dump, so the physics events that the FM was trained on run through W3–W4 as
well; in the same step verify the field/noise files of that chain against the toolkit's (§2.2).

## 7. FM integration steps (F1–F6) and their gates

| Step | Work | Gate / output |
|---|---|---|
| **F1** artifacts | Copy `kd_uni_mbv3_a1_inf01` and `kd_uni_a1_inf01` checkpoints from SDCC to `WC_FM_DINO/kd_checkpoints/`; write `sdcc/export_mbv3_ts.py`; export with `torch.jit.script` under torch ≤ 2.8; record sha256 of checkpoint and `.ts` | eager-vs-scripted parity on 20 `_20k` events: max |Δ| < 1e-4 (fp32), cosine > 0.9999; `.ts` placed in `wire-cell-data/fm/dune10kt-1x2x6/` |
| **F2** oracle | `wire-cell-python` script `wcfm/scripts/fm_oracle.py`: read a `FrameFileSink` output, build the FM input exactly as the pack builder does (sum 4 ticks × scale, `VIEW_NORM`, W two-face layout), run the eager MBV3, write the sidecar layout as npz. Establish the scale constant by running the same script on a LArSoft-tapped frame and comparing with the `_20k` pack pixel values | scale and active-mask threshold fixed and written into §4.1 metadata defaults; oracle output is the reference for F3 |
| **F3** C++ stage | `FMFeatureExtract` (+ `FMUtil` helpers), `wct-fm-features.jsonnet`, `run_fm_evt.sh`; unit test `pytorch/test/doctest_fm_pack.cxx` (packing on a synthetic frame: rows, sum, log map, bbox padding, halo tiling seamlessness) | C++ vs oracle on the W3 manifest: coordinate sets identical, max |Δfeat| < 1e-4 CPU, < 1e-3 GPU; wall/RSS/VRAM table (§5); `./build/pytorch/wcdoctest-pytorch` passes; DNN-ROI untouched (`git diff` empty on `DNNROIFinding.cxx`, `Util.cxx`) |
| **F4** join | `PointTreeBuilding` sidecar input + `fm_feat` on the ctpc, `Grouping::fm_feature`, doctests | knob-off byte-identical on `abtest/events.txt` (pdhd + pdvd, `ab_compare.sh` labels reported) and on the wcfm manifest; knob-on: `fm_match_rate ≥ 0.99`, pctree TensorDM round trip preserves `(N,128)`; RSS growth = N_ctpc × 512 B ± 10 % |
| **F5** Phase-0 probe | `ClusteringFMProbe` dump on the iso-gun set (fine-level sub-blobs from a first `RetileCluster`-based generator, truth from W4), then `wcfm/scripts/crossview_probe.py`: logistic/MLP on the three pooled descriptors vs charge-only baseline, event-level splits | go/no-go for the GNN (GNN doc §10/§11 Phase 0): AP above the charge-only baseline by a pre-registered margin; if it fails, the FM needs the cross-plane objective first (doc 35 #8) and the campaign returns to the model side |
| **F6** Triton/CAP | build `triton/` (deps), Triton Python backend serving CAP with the sparse contract, `FMFeatureExtractSparse`, `fm_forward=TritonService` in jsonnet | same parity protocol against the eager CAP forward; same sidecar metadata; timing on the GPU box |

F1–F4 are sequential; F6 runs alongside F3–F4; F5 needs W1–W4 and F4.

## 8. SBND later

What changes: geometry (`sbnd` cfg, 2 TPCs, no wrapped wires, different channel counts), the
`VIEW_NORM` constants (measured on SBND frames), and the model — the FM must be retrained or fine-tuned
on SBND frames (the technote's per-detector artifact expectation, GNN doc §12.4), which needs the SBND
equivalent of the LArSoft labelling chain. What does not change: the sidecar contract (§4.1), the node
(§4.2, only `channel_order`/`view_norm`/`tick_span` differ), the join (§4.4) and the blob access (§4.5).
The SBND chain in `sbnd_xin/` already has the img/clus/QL/PR runners and the `icluster-*.npz` gate
blind spot noted in `abtest/` (hash `.npz` members too before gating there).

## 9. Risks and open questions

1. **Cross-plane association quality is unmeasured** (§2.1). The whole deghosting premise rests on F5.
2. **Per-detector model.** Features and constants are FD-HD-sim specific; PDHD/PDVD/SBND each need a
   model. The contract is built so only the artifact changes.
3. **Training-frame provenance.** The LArSoft chain's field response and noise vs the toolkit's
   2026-05-18 stand-ins; the standalone W2 sim will differ from the training distribution in noise and
   response — the iso-gun results are about geometry and ghosts, not about data/MC.
4. **Dead channels and DNN-ROI.** The FM never saw a dead-channel mask; the toolkit frames after
   `FrameMasking` do carry one. Whether to feed `gauss` or `dnnsp` (after L1SP/DNN-ROI) is decided by
   what the training packs contained (`gauss`, pre-DNN); a later FM should be trained on the production
   frame.
5. **Time alignment.** Slice k == FM pixel k only with `Reframer tbin 0` and the same frame `time`; the
   sidecar carries `frame_time` and the join checks it.
6. **Toolkit gaps to fill:** no `f2` dtype; `to_itensor` 4-D-only and CUDA:0 (bypassed by duplication);
   `TorchService` semaphore is fixed at 1 (GPU sharing between processes is by separate processes);
   `triton/` not built and its client libs absent; `spng/` (torch-native RayGrid/CellBasis/CrossViews,
   a natural GPU home for cell-level work later) not built either.
7. **Feature width and precision.** 128 f32 is the decision; f16 on disk is a lossy choice whose effect
   on the probe is checked once in F4 (probe on f32 vs f16-roundtripped features).
8. **Checkpoint custody.** Student weights exist only on SDCC; F1 copies them here and records shas.
9. **Isochronous sub-blob generation** is new code on the `RetileCluster` path and must be gated as
   default-OFF with the wcfm manifest; it is the one piece of §4.6 that touches production-shared code.

## 10. Session map

| Session | Steps | Outputs | Gate |
|---|---|---|---|
| this | design | this doc | — |
| next | W1–W3 | `wcfm/` params, sim/img/clus jsonnet + runners, first iso-gun events, determinism check | two-run `hash_archive` identity |
| +1 | W4–W5, F1 | truth catcher, baseline doc 02, checkpoints + `.ts` export + parity | parity table |
| +2 | F2–F3 | oracle, `FMFeatureExtract`, `run_fm_evt.sh`, memory/latency table | C++ vs oracle parity; RSS/VRAM budget |
| +3 | F4 | ctpc join, facade access, doctests | knob-off byte-identical (pdhd/pdvd/wcfm labels) |
| +4 | F5 (+ F6 in parallel) | sub-blob generator, probe dump, probe result | Phase-0 go/no-go |
| after | GNN Phase 1–2 (GNN doc), LArSoft tap, SBND | — | — |

References: `wc-pr-ml-thoughts/GNN_Blob_Deghosting_Design.md`; `wc-pr-ml-thoughts/WC-PR-ML-Ideas.md`
§3.1, §3.2, §9.0, §9.1; `wc-pr-ml-thoughts/projective_readout/05_recommended_architecture.md` §5
(Stages B, D); `WC_FM_DINO/docs/{17,19,32,33,35,37,39}`; technote `main.pdf` §5.4, §9, App. F/K; owner
slides `Wire-Cell_PR_AI_ideas.pdf` 13–16, 36–38 and `Wire-Cell-AI.pdf` 9, 15–18.

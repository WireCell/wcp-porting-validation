# wcfm doc 04 — F2 oracle, F3 `FMFeatureExtract`, and the Phase-0 cross-view probe on the sub-blob tier

Date 2026-09-24. Continues doc 03 (wcp `84537949`, toolkit `25edbe2a`). Three deliverables of doc 01 §7:

- **F2 (oracle):** `wcfm/scripts/fm_oracle.py` packs a `FrameFileSink` SP frame the way the FM training
  packs were built and runs the eager MBV3 student; the scale constant (0.25) and the active-mask
  threshold (pixel > 0) are *established*, not assumed (§1).
- **F3 (C++ stage):** `pytorch/FMFeatureExtract` (toolkit `c58501b8`, new files only), `wcfm/wct-fm-features.jsonnet`,
  `wcfm/run_fm_evt.sh`, doctest `pytorch/test/doctest_fm_pack.cxx`; C++ vs oracle on the 10-event
  manifest: coordinate sets identical on all 69 plane-sets, max |Δ| 9.8e-6 (§3–§4).
- **F5 (Phase-0 probe):** `wcfm/scripts/crossview_probe.py` on the doc 03 §6 sub-blob tier
  (`work/000001_*_sub/clusters-tru0-*`, 94 478 labelled sub-blobs): FM descriptors vs a charge-only
  baseline for the ghost label, leave-one-event-out, pre-registered margin +0.05 AP (§5). This runs on
  the tier offline in Python; the C++ join (F4) and `ClusteringFMProbe` are not built here.

## 0. Repro

```
cd /home/xqian/toolkit-dev/wcp-porting-img/wcfm      # toolkit direnv python (torch 2.5.1)

# --- F2: packing constants and the oracle ---------------------------------------------------
python3 scripts/fm_scale_check.py --out /home/xqian/tmp/wcfm-f3/scale_check.json      # docs/04_tables/scale_check.*
python3 scripts/fm_oracle.py work/000001_1/sim-frames-anode10.tar.bz2 \
    --ts ../../wire-cell-data/fm/dune10kt-1x2x6/kd_uni_mbv3_a1_inf01.ts \
    --out /home/xqian/tmp/wcfm-f3/oracle-000001_1-anode10.npz             # one per anode-event, 23 in all

# --- F3: the C++ stage --------------------------------------------------------------------------
# (toolkit) wcbuild ; ./build/pytorch/wcdoctest-pytorch
./run_fm_evt.sh -F -O _fmf4 1 all          # f4 store, cpu  -> work/000001_<evt>_fmf4/fm-features-anode<N>.tar.gz
./run_fm_evt.sh    -O _fm   1 all          # f16 store (default), cpu
./run_fm_evt.sh -D gpu -O _fmgpu 1 all     # f16 store, gpu ; -D gpu -F -O _fmgpuf4 for the f4 gpu arm
python3 scripts/fm_parity.py --oracle-dir /home/xqian/tmp/wcfm-f3 --suffix _fmf4 \
    --out /home/xqian/tmp/wcfm-f3/parity_fmf4.json work/000001_{1,2,3,4,5,6,7,8,9,10}   # docs/04_tables/parity*.json

# --- F5: the probe on the tru0 sub-blob tier ------------------------------------------------------
python3 scripts/crossview_probe.py --features oracle:/home/xqian/tmp/wcfm-f3 --mlp \
    --out /home/xqian/tmp/wcfm-f3/probe_oracle work/000001_{1,2,3,4,5,6,7,8,9,10}_sub   # docs/04_tables/probe_*
```

Inputs are our own products: the doc 02 SP frames (`work/000001_*/sim-frames-anode*.tar.bz2`) and the
doc 03 knob-on imaging (`work/000001_*_sub/`, BlobCutting at 20 wires, BlobDepoFill truth at 64.5 µs).
The manifest is `abtest_events.txt` (8 iso events, 2 cosmics, 23 anode-events).

## 1. F2a — the packing constants (scale 0.25, active ⇔ pixel > 0)

Doc 01 §2.1 derived "0.25·Σ₄" from reading configs and flagged it *to be re-verified*. Two independent
checks now fix it.

**Static chain (sources read this session).**

| step | where | what it does to the SP `gauss` sample |
|---|---|---|
| LArSoft SP save | `cffm-if/dune10kt-1x2x6/wcls-sim-drift-simchannel-nf-sp.jsonnet:123` `wclsFrameSaver` `frame_scale [0.005, …]`, `digitize: false` | × 0.005, stored as `recob::Wire` float |
| labelling read | `wcls-labelling2d_sep.jsonnet:68` `wclsCookedFrameSource` `frame_scale: 50` | × 50 |
| rebin | `larwirecell/aiml/Labelling2D.cxx:336-361` (`rebin_time_tick: 4`) | **sum** of the 4 ticks (not mean) |
| dense tap | `hio/HDF5FrameTap` (`scale` default 1.0, `arr = arr*scale + offset`) | unchanged, `frame_rebinned_reco (2560,1500)` |
| sparsify | `WC_FM_Sim/scripts/sparsify_to_prodjay.py:84-90` `np.nonzero(dense)` | keeps every **nonzero** pixel |

So pack pixel = 0.25 × Σ₄(native `gauss`), and the mask is "nonzero".

**The packs confirm it by themselves.** The toolkit's SP writes *integer-valued* charge (every nonzero
sample of the 23 wcfm frames is an integer; the LArSoft chain used the same OmnibusSigProc), so 0.25 × an
integer sum must be a multiple of 0.25. It is: sampling 5 M pixels per plane of
`packed_numu_truth_apa0_{U,V,W}_20k.npz`, the smallest value is exactly 0.25 in every plane and every
one of the first 2000 events, and 98.8 % of the values are exact quarter-multiples (the rest are large
values where the float32 product 0.005·q·50 is no longer exact; 24.8 % are integers, as expected for
uniformly distributed quarters). A scale of 1 (no quarters), 0.005 or 50 is excluded outright. The
smallest pack value being one quantum (0.25 = one native unit) settles the threshold: a pixel is active
iff its 4-tick sum is nonzero, which for the non-negative SP output equals **pixel > 0** — the
`active_threshold 0` default of the C++ node and the oracle.

**Distribution check (docs/04_tables/scale_check.md).** The wcfm frames × 0.25 land on the same scale as
the packs: medians U 480 vs 562, V 380 vs 563, W 538 vs 741 (ratios 0.7–0.9, where the wrong scales
would give ×4 or ×200); 2nd percentiles 2.5–3.0 vs 3.5–3.8 (the `VIEW_NORM` `m` = 2.77/2.97/3.75 was the
2nd percentile of the training mix). The wcfm sample is softer at the top (p99.999 18 k vs 113 k on U):
iso tracks and two cosmics against νμ showers, and our sim's own noise/response — a sample difference,
not a scale difference. **Written into the defaults:** `input_scale 0.25`, `active_threshold 0`,
`tick_span 4` in `FMFeatureExtract` and `wct-fm-features.jsonnet`.

## 2. F2b — the oracle (`wcfm/scripts/fm_oracle.py`)

Per anode-event it reads `frame_gauss<N>_<ident>` (2560 × 6000 f32, `tickinfo` t0 = 0, tick 500 ns,
tbin 0), and per plane:

1. rows = channel ident − first ident of the plane (U [0,800), V [800,1600), W [1600,2560) of the APA;
   the W image is therefore face 1 (channels 1600–2079) then face 0 (2080–2559) in channel order, the
   FM training layout — verified against the wires file: face-0 W wire i ↔ channel 2080+i, face-1 W
   wire i ↔ channel 2079−i);
2. pixel(row, k) = 0.25 × Σ ticks [4k, 4k+4) (float64 sum, cast to f32), active ⇔ > 0;
3. `FeatureLogTransform` with the plane's `VIEW_NORM` in float32 arithmetic;
4. canvas `[1, 2, h, w]` = (value, mask) on the tight bounding box, floored at 64 on the high side
   (`dense_mae_adapter._rasterize_one`), `--bbox-pad` optional (default 1 = training behaviour);
5. eager `FMDenseStudent` (`WC_FM_DINO/sdcc/export_mbv3_ts.py`, the wrapper doc 03 proved bit-identical
   to the `.ts`) → gather at the active pixels, rows sorted by (channel, slice).

Output npz: `coords_<P> (N,2) i4 [channel ident, slice]`, `feat_<P> (N,128) f4`, `canvas_<P>`, `bbox_<P>`,
`meta` (json: frame ident/time/tick, scale, threshold, checkpoint sha, torch version). With `--ts` the
scripted model runs on the same canvases: **0.00e+00** difference on all planes of event 1 (as in doc 03).
Event 1 anode 10: U 3102 pixels, canvas 447 × 64; V 1854, 700 × 655; W 2594, 424 × 64 — the iso track
occupies 7–9 slices, so two of three canvases hit the 64-column floor.

## 3. F3 — `Pytorch::FMFeatureExtract` (toolkit, new files only)

`pytorch/inc/WireCellPytorch/FMFeatureExtract.h`, `pytorch/src/FMFeatureExtract.cxx`: an
`IFrameTensorSet` (`IFunctionNode<IFrame, ITensorSet>`), **one instance per anode, all configured planes
in one node** (doc 01 §4.2 said one per anode-plane + `TensorSetFanin`; `TensorSetFanin` discards the
set metadata — "OUTPUT METADATA IS EMPTY" in its header — which would have dropped the sidecar
provenance, so the planes loop inside the node instead and one `ITensorSet` per frame carries
everything). Per plane:

1. **rows**: `Aux::plane_channels(anode, plane)` → sorted unique idents, checked contiguous (the FM
   rebases by the first channel; a non-contiguous plane throws at configure);
2. **time**: `Aux::fill` from `tick0` into rows × `nticks`, then Σ over `tick_span` ticks accumulated in
   double, × `input_scale` + `input_offset`, cast to f32; active ⇔ > `active_threshold`;
3. **log map** with the plane's `view_norm` `(m, M)` in float32, exactly the torch arithmetic;
4. **canvas**: tight bbox, floor `min_canvas` 64, `bbox_pad` (default 1); when `max_dense_pixels` > 0
   and h·w exceeds it, tiles along the slice axis with `halo` (64) columns on each side, and only the
   core columns of a tile are gathered (default 0 = never tile: the largest wcfm canvas is 868 k
   pixels);
5. **forward**: a `SimpleTensor {1,2,h,w} f4` in a `SimpleTensorSet` through the configured
   `ITensorForward` (`TorchService:fm`, the doc 03 `.ts`); the reply must be `{1, feature_dim, h, w} f4`
   (checked) and is read through the `ITensor` data pointer — no torch call in this node, so the
   DNN-ROI `Util.cxx` helpers are neither used nor changed and the planned `FMUtil.cxx` was not needed;
6. **output**: `coords (N,2) i4` and either `feat (N,128) f4` or, with `store_half` (default true),
   `feat_half (N,128) u2` = IEEE-half bits (doc 01 §4.1 option b; decoded in Python with
   `.view(np.float16)`); every tensor carries `{name, plane, base_channel, n_active, bbox, canvas,
   tiles, view_norm}`; the set carries frame ident/time/tick, `tick_span`, `input_tag`, scale, offset,
   threshold, `face_layout "channel"`, `feature_dim`, `store_half`, and the caller's `provenance`
   object (model file, sha256, arch, device from the jsonnet/runner).

`wcfm/wct-fm-features.jsonnet`: `FrameFileSource(gauss<N>) → FMFeatureExtract → TensorFileSink`
per anode, one shared `TorchService:fm` (`fm_model`, `fm_device`); TLAs for every packing knob and
`store_half`. Output `fm-features-anode<N>.tar.gz` (doc 01 said `.tar.zst`; `wirecell.util.ario` reads
gz/bz2/xz only, so gz). `wcfm/run_fm_evt.sh [-a N] [-D cpu|gpu] [-M model] [-F] [-O suffix] <run> <evt|all>`:
one `wire-cell` per anode under `timecmd.py` (rc/wall/maxrss), `nvidia-smi` peak-VRAM sampling for
`-D gpu`, model sha256 into the provenance, `WCFM_LIBPIN` to prepend a pinned library directory,
`WCFM_TORCH_THREADS` (default 8).

**Not taken from doc 01 §4.2:** `bbox_pad 32` as the default — padding changes the canvas the model
sees relative to training (`_rasterize_one` pads only to the 64 floor) and the parity gate is against
that behaviour; the knob exists, default 1. `channel_order` knob — rows are channel order by
construction, recorded as `face_layout: "channel"`.

### 3.1 Doctest (`pytorch/test/doctest_fm_pack.cxx`, the first doctest of the pytorch package)

A pointwise fake `ITensorForward` registered in the test binary (`WIRECELL_FACTORY` + the
`make_<name>_factory()` call of the util namedfactory test) returns `out[c] = f_c(x0, x1)`, so the
features themselves prove the gather alignment and tiling must be exact. Cases: rows/sum/scale on
traces straddling a slice boundary (0.75, 4.5, 10, 25), the log map, a plane not packed is ignored,
sorted `(channel, slice)` coords, bbox/canvas metadata (64-row floor, tight 202 columns), EOS;
tiling (`max_dense_pixels`, `halo 16`, > 2 tiles) reproduces the untiled coords and features
exactly; `store_half` gives `u2` that decodes within 1e-3. **2 cases, 3000 assertions, 0 failed.**
Revert-proven (first tick instead of the sum; gather row/col swapped; halo columns gathered).

### 3.2 Build status

A peer session's SBND campaign (doc sbnd_xin/123 arms, `~/tmp/d123-libpin`) was live in this tree for
the whole session, and the rule is *no waf target while any arm runs* (memory
`feedback_shared_tree_binary_pin`). Everything above was therefore built **outside waf** with the
package's own flags (`build/compile_commands.json`) into `/home/xqian/tmp/wcfm-f3/lib/libWireCellPytorch.so`
(the existing `build/pytorch/src/*.o` + the new object, the canonical RUNPATH, `libfmt.a`) and run
through `WCFM_LIBPIN`; the doctest binary likewise (`/home/xqian/tmp/wcfm-f3/pbuild/build.sh`).
**Canonical re-verification done in the first quiet window (21:23–21:27):** `./wcb build --notests
-p -k` + `install` (the new-symbol/doctest link trap of `feedback_new_symbol_test_link_install`, first
pass fails only on `wcdoctest-pytorch`), then `./wcb build -p` + `install -p` rc=0; freshness proof
`local/lib/libWireCellPytorch.so` 21:24:13 > source 10:51, 53 `FMFeatureExtract` symbols;
`./build/pytorch/wcdoctest-pytorch` 2 cases / 3000 assertions passed, `wcdoctest-img` 5/5; event 1
anode 10 re-run on the installed library (`libpin=none` in `work/000001_1_fmf4c/fm-provenance.txt`):
parity max |Δ| 2.62e-6, the same numbers as the private build. The manifest numbers of §4 are from
the private build of the same sources.

## 4. F3 gate — C++ vs oracle, and the cost table

**Parity (docs/04_tables/parity.md, `parity*.json`).** 10 events, 23 anode-events, 69 plane-sets,
160 807 active pixels; the coordinate sets are identical in every plane-set of every arm.

| arm | store | device | max |Δ| | median max |Δ| | median mean |Δ| | min cosine | verdict |
|---|---|---|---|---|---|---|---|
| `_fmf4` | f4 | cpu | **9.8e-6** | 2.2e-6 | 2.0e-7 | 0.9999997 | **PASS** the doc 01 gate (< 1e-4, > 0.9999) |
| `_fm` | f16 bits | cpu | 1.95e-3 | 9.8e-4 | 9.6e-5 | 0.9999996 | = half rounding (half-ulp at |feat| 4–8 is 1.95e-3); information |
| `_fmgpu` | f16 bits | gpu (RTX 4090) | 1.95e-3 | 9.8e-4 | 9.6e-5 | 0.9999997 | rounding-dominated, same as CPU f16 |
| `_fmgpuf4` | f4 | gpu | **5.1e-5** | 9.3e-6 | 7.8e-7 | 0.9999996 | **PASS** the same bar; the TF32 question of doc 03 §7 is closed (see below) |

The f4 CPU residual (1e-5) is the float32 log map and reduction-order noise; doc 03's export parity was
2e-5 against torch 2.10 for the same reason. **GPU/TF32:** doc 03 measured 9.2e-3 for the same `.ts` on
the same GPU from *Python* torch 2.5.1 and left "disable TF32 in `TorchService` behind a knob" open.
Through `TorchService` (libtorch 2.8, C++ defaults) the GPU forward agrees with the CPU one to 5e-5 on
all 69 plane-sets — the C++ context does not run the convolutions in TF32 — so the doc 01 GPU bar
(< 1e-3) is met with margin and no knob is needed. If a future libtorch flips that default, this arm
is the detector. The f16 store costs 1e-4 mean, 2e-3 max on features of
magnitude ~0.5 — 0.2 % relative — and halves the sidecar (event 1 anode 10: 2.07 MB vs 4.11 MB for
7550 pixels; the feature bytes hardly compress).

**Cost (docs/04_tables/timing.tsv; one `wire-cell` per anode-event, 8 torch threads, box shared with
the peer campaign, load 18–38).**

| arm | wall/anode-event (median, max) | peak RSS (median, max) | per-plane packing+forward ms (median, max) | canvas px (median, max) | peak VRAM Δ |
|---|---|---|---|---|---|
| cpu f4 | 3 s, 23 s | 1.19 GB, 2.01 GB | 169, 624 | 59 k, 868 k | — |
| cpu f16 | 3 s, 3 s | 1.19 GB, 2.00 GB | 156, 563 | | — |
| gpu f16 | 4 s, 4 s | 1.34 GB, 1.72 GB | 253, 664 | | ≤ 2.3 GB |
| gpu f4 | 4 s, 4 s | 1.34 GB, 1.72 GB | 260, 739 | | ≤ 2.3 GB |

Readings: (a) the process cost is libtorch itself — 1.2 GB RSS with a 14 MB model and canvases of at
most 7 MB — so the FM stage as a separate process is cheap in time and expensive in resident memory
exactly as doc 01 §5 predicted; in-job (the DNN-ROI pattern) it would share the one libtorch;
(b) the GPU numbers are cold: every process runs one frame, so each plane call pays CUDA/cuDNN warm-up
(doc 03 measured 14 ms steady-state on the same canvases) — a batch of events per process is what the
GPU path needs; (c) the 23 s outliers are four jobs starting together on the loaded box (library
load), not the forward.

**DNN-ROI untouched:** `git status` on the toolkit shows only the three new files; `DNNROIFinding.cxx`
and `Util.cxx` are byte-identical (`git diff` empty). No production config changes (the FM job is a
new wcfm entry point).

## 5. F5 — the Phase-0 cross-view probe on the sub-blob tier

**Pre-registered (in the script header and here, before the first run):** the FM probe must beat the
charge-only baseline by **≥ +0.05 pooled held-out AP** for the ghost class.

**Data.** The doc 03 knob-on tier `work/000001_*_sub/clusters-tru0-anode<N>-ms-active.tar.gz`: every
tiled + cut blob before deghosting, `val` = BlobDepoFill true charge, ghost ⇔ `val == 0`. 10 events,
23 anode-events, **94 478 sub-blobs, ghost fraction 0.634** (iso events 0.38–0.89, the cosmics 0.03 and
0.14); median 2 sub-blobs per (anode, face, slice), max 4367 (the event-8 slab). Each sub-blob's strip
wires come from the cluster file's `wnodes` (`(planeid, index) → channel`; all 1 938 406 strip wires of
the manifest are covered), the FM rows from the oracle sidecar at `(channel, slice)` (the C++ sidecar
is byte-equivalent for coordinates and 1e-5 for values, §4), pooled per view (mean and max of the
128-d rows). 7 % of the sub-blobs have a view with no FM pixel (a strip over inactive channels).

**Scores** (`scripts/crossview_probe.py`, leave-one-event-out, standardised logistic regression;
`--mlp` adds a 64-unit MLP):

| score | inputs | pooled AP | pooled AUC |
|---|---|---|---|
| zero-shot tri-view disagreement (mean pairwise cosine distance of the 3 mean-pooled vectors) | FM only, no label | 0.638 | 0.508 |
| zero-shot disagreement (variance) | FM only, no label | 0.664 | 0.540 |
| **charge-only baseline** | per view n_ch, n_pix, log Σq, log mean q; pairwise log-ratios (15 numbers) | **0.987** | 0.979 |
| **FM + charge** | the 15 + 3 × 128 mean-pooled | **0.990** | 0.983 |
| FM only | 3 × 128 | 0.966 | 0.954 |
| FM + charge, MLP | | 0.986 | 0.973 |
| charge-only, MLP | | 0.986 | 0.977 |
| prevalence | | 0.634 | |

Per event (docs/04_tables/probe_summary.md): charge-only AP 0.93–1.00 on every event; FM + charge
0.97–1.00 on the eight iso events but **0.82 / 0.78 on the two cosmics** (charge-only 0.99 / 0.93);
FM-only 0.26 on event 9 (prevalence 0.03). Restricting to the sub-blobs whose three views all have FM
pixels changes nothing (docs/04_tables/probe_hard_subset.md: 0.986 vs 0.989).

**Verdict: ΔAP = +0.003 < +0.05 → NO-GO on the pre-registered margin.** Three readings, in order of
weight:

1. **The zero-label signal is absent.** Tri-view disagreement of the FM descriptors has AUC 0.49–0.54
   on 94 k labelled cells: the per-plane features of the three views of one deposit agree no more than
   those of three views that merely cross. This is the doc 01 §2.1 caveat ("no cross-plane objective")
   turned into a number, and it is the part of the result that does not depend on the probe's design.
2. **The tier is too easy for the baseline.** Fifteen charge numbers reach AP 0.987 at prevalence 0.634:
   on BlobCutting cells the ghost label is almost a deterministic function of per-view charge
   consistency — the same information the charge solver already uses (which is why doc 03 §6 found
   the solver *deletes* the sub-blobs: it can tell most ghosts, it just cannot keep the true ones). A
   probe on this tier therefore has ~1 % of AP left to win and cannot resolve a +0.05 margin; the
   hard cases (true cells the solver drops) are a small minority the AP does not weight.
3. **The supervised FM probe does not transfer.** 384 inputs, 10 events, eight of them the same
   iso-track topology: the linear probe learns the iso events and loses 0.15–0.2 AP on the two cosmics.
   Whatever is in the descriptors is not a portable "is this cell real" feature at this sample size.

**Consequence (doc 01 §7 F5 as written):** the campaign returns to the model side — a cross-plane
objective (doc 35 #8) before a GNN is built on these descriptors. Before that, one cheaper re-run is
worth doing so the model side gets the right target: re-pose the probe on the *hard* population
(sub-blobs the doc 03 solver deleted or kept wrongly: true cells with `q_true > 0` that are absent
from `clusters-apa`, and ghosts that survived) with ≥ 100 events including cosmics and a per-slice
ranking metric, so the baseline is not at ceiling and the margin means something. F4 (the C++ join) is
not needed for that re-run; the sidecars and this script are.

## 6. Open items

- **F5 re-run** on the hard population with ≥ 100 events (§5); the probe script takes the C++
  sidecar (`--features sidecar:_fm`) or the oracle npz. Run it on a quiet box: the ten logistic folds
  took 3 h at load 78.
- **GPU path**: per-process one-event runs are warm-up dominated; batch events per process (a `-n`
  events-per-job mode of `run_fm_evt.sh` reading several frame files) before quoting GPU throughput.
  TF32: closed (§4, `_fmgpuf4` PASS at 5e-5).
- **F4** (`PointTreeBuilding` sidecar input, `Grouping::fm_feature`) is the next step of doc 01 §7;
  the probe here reads the sidecar offline, so its result does not depend on F4.
- `ClusterArrays` numpy `bwedges`: 42 % of the rows of the wcfm tru0 files are not blob→wire pairs
  (tails in the activity-node descriptor space); the probe uses `wnodes` (`(planeid, index) →
  channel`, covers all 1.94 M strip wires) instead. Pre-existing, not touched.
- The 23 s wall outliers and the 2 GB RSS ceiling should be re-measured on a quiet box.

## 7. Files

Toolkit (this commit): `pytorch/inc/WireCellPytorch/FMFeatureExtract.h`, `pytorch/src/FMFeatureExtract.cxx`,
`pytorch/test/doctest_fm_pack.cxx`. wcp (this commit): `wcfm/wct-fm-features.jsonnet`,
`wcfm/run_fm_evt.sh`, `wcfm/scripts/{fm_oracle.py, fm_scale_check.py, fm_parity.py, crossview_probe.py}`,
`docs/04_*.md`, `docs/04_tables/*`, `docs/README.md`. Work products: `work/000001_*_{fmf4,fm,fmgpu,fmgpuf4}/`
(sidecars, logs, timing), `work/000001_*_sub/` (doc 03, read only). Scratch: `/home/xqian/tmp/wcfm-f3/`
(oracle npz, parity json, probe outputs, the private build).

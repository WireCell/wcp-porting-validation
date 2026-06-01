# 3D Imaging Stage — `run_img_evt.sh` (`sbnd_xin/`)

> For per-script CLI options see **[scripts.md](scripts.md)**.
> For geometry / timing constants see **[geometry-and-timing.md](geometry-and-timing.md)**.
> For the full pipeline overview see **[sbnd.md](sbnd.md)**.

This document explains the imaging stage of the SBND standalone pipeline:
what algorithm runs, how the configuration drives it, and what the output
files contain. The imaging stage runs **no signal processing** — input is
already DNN-SP–deconvolved frames dumped from LArSoft.

---

## Driver script: `run_img_evt.sh`

```
./run_img_evt.sh [-a anode] [-s sel_tag] <idx>
```

| Option | Meaning |
|---|---|
| `<idx>` | 1-based event index (1–10); maps to event IDs below |
| `-a 0\|1` | restrict to one anode; omit for both `[0,1]` |
| `-s <tag>` | use Woodpecker-masked input from `run_select_evt.sh` |

Event mapping (`run_img_evt.sh:16`):

| idx | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| EVT_ID | 2 | 9 | 11 | 12 | 14 | 18 | 31 | 35 | 41 | 42 |

**Input path** (`run_img_evt.sh:50–56`):

| Mode | SP archive path |
|---|---|
| Normal | `work/evt<ID>/sp-frames.tar.bz2` |
| Selection (`-s <tag>`) | `work/evt<ID>_<tag>/input/sp-frames.tar.bz2` |

Both are produced upstream by `run_sp_to_magnify_evt.sh` (normal) or
`run_select_evt.sh` + `merge_sel_archives.py` (selection).

**`wire-cell` invocation** (`run_img_evt.sh:83–90`):

```sh
wire-cell \
    -l stderr \
    -l "${LOG}:debug" \
    -L debug \
    --tla-str  "input=${SP_ARCHIVE}" \
    --tla-code "anode_indices=${ANODE_CODE}" \
    --tla-str  "output_dir=${WORKDIR}" \
    -c wct-img-all.jsonnet
```

**Outputs**: `work/evt<ID>[_<tag>]/icluster-apa{0,1}-{active,masked}.npz`
**Log**: `work/evt<ID>[_<tag>]/wct_img_evt<ID>[_a<N>].log` (debug level)

---

## Top-level config: `wct-img-all.jsonnet`

### Function signature

```jsonnet
function(
  input         = 'sp-frames.tar.bz2',   // --tla-str
  anode_indices = [0, 1],                // --tla-code
  output_dir    = '',                    // --tla-str
)
```
(`wct-img-all.jsonnet:26–30`)

### Graph structure

```
FrameFileSource(tag='dnnsp')                    // reads sp-frames.tar.bz2 directly
    │
    ▼ FrameFanout  (one branch per anode)
    │   rule N: {frame: '.*'→'origN', trace: dnnsp→['gaussN','wienerN']}
    │
    ├─ [APA 0] img_maker.per_anode(anode0, 'multi-3view')
    │               ├─ port 0 → ClusterFileSink → icluster-apa0-active.npz
    │               └─ port 1 → ClusterFileSink → icluster-apa0-masked.npz
    │
    └─ [APA 1] img_maker.per_anode(anode1, 'multi-3view')
                    ├─ port 0 → ClusterFileSink → icluster-apa1-active.npz
                    └─ port 1 → ClusterFileSink → icluster-apa1-masked.npz
```

### The FrameFanout retag trick

(`wct-img-all.jsonnet:86–92`) SBND DNN-SP produces a single `dnnsp` trace
tag. The shared `experiment/sbnd/img.jsonnet` imaging graph was written for
the uboone-style pipeline that expects distinct `gauss<N>` (charge) and
`wiener<N>` (quality/threshold reference) traces per anode. The FrameFanout
rule aliases the same DNN-SP trace under both names:

```jsonnet
trace: { dnnsp: ['gauss0', 'wiener0'] }   // for APA 0
trace: { dnnsp: ['gauss1', 'wiener1'] }   // for APA 1
```

This is not needed in PDHD/PDVD, where separate noise-filtering and
signal-processing steps produce distinct gauss/wiener tags.

### The `g.intern` two-port wiring

(`wct-img-all.jsonnet:74–82`) `per_anode(...)` ends in a `g.fan.fanout`
that emits port 0 (active clusters) and port 1 (masked clusters). A plain
`g.pipeline` can only attach a single tail node, so `g.intern` is used to
wire both ports explicitly to their respective `ClusterFileSink` nodes.

### Per-anode channel selection

The redundant per-branch `ChannelSelector` pre-filter that previously
sat in front of each `per_anode(...)` has been removed. The per-anode
5638-channel restriction (`5638*N .. 5638*(N+1)-1`) is handled by
`img.jsonnet`'s internal `chsel_pipes` selector inside the shared
imaging graph.

### Plugins

(`wct-img-all.jsonnet:111–119`)
`WireCellGen, WireCellPgraph, WireCellSio, WireCellSigProc,
WireCellImg, WireCellClus, WireCellRoot`

---

## Per-anode imaging algorithm

Defined in `cfg/pgrapher/experiment/sbnd/img.jsonnet`. Called as
`img_maker.per_anode(anode, 'multi-3view', add_dump=false)` which
resolves to (`img.jsonnet:361–364`):

```
pre_proc(anode)  →  imgpipe(anode, 'multi-3view', add_dump=false)
```

### Pre-processing (`pre_proc`) — IFrame → IFrame

(`img.jsonnet:16–130`) Sequence of four IFrame-to-IFrame components:

```
ChannelSelector (chsel_pipes)
    → CMMModifier (cmm_mod)
    → FrameMasking (frame_masking)
    → ChargeErrorFrameEstimator (charge_err)
```

**1. `ChannelSelector` (chsel_pipes)** (`img.jsonnet:41–51`)

Keeps only the 5638 channels belonging to this APA and the two trace tags:

```jsonnet
channels: std.range(5638 * anode.data.ident, 5638 * (anode.data.ident + 1) - 1),
tags: ['gauss<N>', 'wiener<N>'],
```

See [geometry-and-timing.md §"Per-APA channel count"](geometry-and-timing.md)
for the history of the 5632 production bug.

**2. `CMMModifier`** (`img.jsonnet:67–91`)

Organises the `bad` channel-mask map (CMM) by expanding bad-channel ranges
using the `gauss<N>` charge frame. The boundary `org_hlimit: [3427]` ensures
the full readout window is covered (`img.jsonnet:89`).

**3. `FrameMasking`** (`img.jsonnet:118–127`)

Zeros out waveform samples on `bad` channels for both `gauss<N>` and
`wiener<N>`. Prevents bad-channel charge from leaking into the slicer.

**4. `ChargeErrorFrameEstimator`** (`img.jsonnet:26–38`)

Produces `gauss_error<N>` from `gauss<N>` using a pre-computed
`WaveformMap` loaded from `sbnd-charge-error.json.bz2`:

```jsonnet
rebin: 4,                         // rebin factor before applying waveform map
fudge_factors: [2.31, 2.31, 1.1], // per-plane (U, V, W) scale factors
time_limits: [12, 800],           // in rebin-4 ticks ≈ raw ticks 48–3200
```

The error estimate is consumed by `MaskSlices` as `error_tag` during
slicing.

> **Note**: A `MagnifySink` debug node (`img.jsonnet:53–65`) and a
> `FrameQualityTagging` node (`img.jsonnet:93–116`) are defined in the
> file but are **not part of the active pipeline** (`img.jsonnet:129`).

---

### Imaging fork — `multi-3view` mode

After pre-processing, `imgpipe` with `multi_slicing='multi-3view'` splits
into two parallel branches via `g.fan.fanout('FrameFanout', ...)`:
(`img.jsonnet:341–358`)

```
pre_proc output (IFrame)
    │
    ├─ active_fork → port 0 (ICluster, with solved charge)
    └─ masked_fork → port 1 (ICluster, geometry only)
```

---

### Active fork — `multi-3view` slicing + tiling + solving

(`img.jsonnet:347–351`, `multi_active_slicing_tiling` at line 178)

**Step 1 — slicing fanpipe** (4 branches, merged by `BlobSetMerge`)

Each branch runs one `MaskSlices` → `GridTiling` with a different plane
combination:

| Branch | `active_planes` | `masked_planes` | Coverage |
|---|---|---|---|
| 0 | [0,1,2] | [] | all three planes active |
| 1 | [0,1] | [2] | U+V only, W masked |
| 2 | [1,2] | [0] | V+W only, U masked |
| 3 | [0,2] | [1] | U+W only, V masked |

(`img.jsonnet:179–180`)

`MaskSlices` parameters (shared across all 4 branches, `img.jsonnet:133–155`):

```jsonnet
tick_span:    4,           // 4 ticks × 0.5 µs/tick = 2 µs per slice
min_tbin:     0,
max_tbin:     3427,        // full SBND readout window (was 3400)
nthreshold:   [3.6, 3.6, 3.6],   // per-plane signal threshold
wiener_tag:   'wiener<N>',
summary_tag:  'wiener<N>',
charge_tag:   'gauss<N>',
error_tag:    'gauss_error<N>',
```

`GridTiling` (`img.jsonnet:158–175`): sets `face = anode.data.ident`
(SBND-specific — one face per anode, unlike some multi-face detectors).

**Step 2 — solving** (`img.jsonnet:216–301`, active pipeline at line 300)

The "simple-solving" pipeline (the richer multi-round chain on line 299 is
commented out):

```
BlobClustering (policy='uboone')
    → BlobGrouping
    → ChargeSolving (weighting='uniform', solve_config='uboone', whiten=true)
    → LocalGeomClustering
    → ChargeSolving (weighting='uboone', solve_config='uboone', whiten=true)
    → InSliceDeghosting (config_round=1)
    → GlobalGeomClustering (policy='uboone')
```

The commented-out richer chain includes multiple rounds of
`ProjectionDeghosting` and `InSliceDeghosting` — this is a tuning knob
for future refinement.

---

### Masked fork — 2-view dummy slicing

(`img.jsonnet:352–356`, `multi_masked_2view_slicing_tiling` at line 191)

**Step 1 — slicing fanpipe** (3 branches, merged by `BlobSetMerge`)

Each branch uses one plane as a `dummy` (geometry scaffold only) and the
other two as `masked`:

| Branch | `dummy_planes` | `masked_planes` |
|---|---|---|
| 0 | [2] (W dummy) | [0,1] (U+V masked) |
| 1 | [0] (U dummy) | [1,2] (V+W masked) |
| 2 | [1] (V dummy) | [0,2] (U+W masked) |

(`img.jsonnet:192–193`)

`MaskSlices` is called with `active_planes=[]` and `span=500`
(500 ticks × 0.5 µs/tick = 250 µs per slice — much coarser than the
active fork's 4-tick span).

**Step 2 — clustering only** (`img.jsonnet:207–213`)

```
BlobClustering (spans=1.0, policy='uboone')
```

No charge solving. Output blobs carry geometry (wire-pair intersections) but
no calibrated charge.

---

## Active vs masked outputs — what they mean

| | Active (`-active.npz`) | Masked (`-masked.npz`) |
|---|---|---|
| Signal requirement | ≥2 planes with real signal above threshold | one plane treated as geometric dummy; other two "masked" |
| Slice span | 4 ticks (2 µs) | 500 ticks (250 µs) |
| Charge values | Yes — full solve including deghosting | No — geometric blobs only |
| Downstream use | Primary clustering input; Bee display | Supplements active in dead/noisy regions |

The downstream clustering step (`run_clus_evt.sh` → `MultiAlgBlobClustering`)
consumes both files together.

---

## Tags reference

| Tag | Producer | Consumer | Role |
|---|---|---|---|
| `dnnsp` | upstream `wcls-sp-dump.fcl` | `FrameFileSource`, `FrameFanout` | raw input traces from DNN-SP |
| `gauss<N>` | FrameFanout retag of `dnnsp` | `chsel_pipes`, `CMMModifier`, `FrameMasking`, `ChargeErrorFrameEstimator`, `MaskSlices.charge_tag` | per-anode charge waveforms |
| `wiener<N>` | FrameFanout retag of `dnnsp` | `chsel_pipes`, `FrameMasking`, `MaskSlices.wiener_tag`/`summary_tag` | per-anode quality/threshold reference |
| `gauss_error<N>` | `ChargeErrorFrameEstimator` | `MaskSlices.error_tag` | per-tick charge uncertainty |
| `bad` | upstream (`chanmask_bad_<EVT>`) | `CMMModifier`, `FrameMasking` | bad-channel mask (CMM) |
| `orig<N>` | FrameFanout frame rename | — | frame-level tag (not used by any component) |

---

## TLA and embedded constants

| Parameter | Value / TLA | Where set | Effect |
|---|---|---|---|
| `input` | `--tla-str input=<path>` | `wct-img-all.jsonnet:27` | SP frame archive path |
| `anode_indices` | `--tla-code anode_indices=[0,1]` | `wct-img-all.jsonnet:28` | which APAs to process |
| `output_dir` | `--tla-str output_dir=<path>` | `wct-img-all.jsonnet:29` | directory for output `.npz` files |
| channels per APA | `5638` (hard-coded) | `img.jsonnet:47` | SBND: 1984 U + 1984 V + 1670 W |
| `max_tbin` | `3427` | `img.jsonnet:145` | SBND DAQ readout window (was 3400) |
| active `tick_span` | `4` | `img.jsonnet:178`, `multi_active…` default | 2 µs per slice |
| masked `span` | `500` | `img.jsonnet:191`, `multi_masked…` default | 250 µs per slice |
| `nthreshold` | `[3.6, 3.6, 3.6]` | `img.jsonnet:150` | per-plane ADC threshold for active slicing |
| `fudge_factors` | `[2.31, 2.31, 1.1]` | `img.jsonnet:34` | U/V/W charge-error scale |
| `time_limits` | `[12, 800]` (rebin-4 ticks) | `img.jsonnet:35` | charge-error estimator tick range |

---

## Input format — `sp-frames.tar.bz2`

`FrameFileSource` reads the archive directly (no prior extraction needed).
See [sbnd.md §"Input"](sbnd.md#input) for the canonical table; summary:

| File inside archive | Shape | Tag |
|---|---|---|
| `frame_dnnsp_<EVT>.npy` | (11276, 3427) | `dnnsp` |
| `channels_dnnsp_<EVT>.npy` | (11276,) | channel IDs |
| `tickinfo_dnnsp_<EVT>.npy` | (3,) | tick0, period, nticks |
| `summary_dnnsp_<EVT>.npy` | (11276,) | per-channel SP summary |
| `chanmask_bad_<EVT>.npy` | varies | `bad` CMM |

Selection-mode archives (`work/evt<ID>_<tag>/input/sp-frames.tar.bz2`) have
the same schema with tick/channel masking already applied by
`merge_sel_archives.py`.

---

## Output format — `icluster-apa<N>-*.npz`

Produced by `ClusterFileSink` with `format: 'numpy'`
(`wct-img-all.jsonnet:62–66`). Each `.npz` is a flattened dump of the
ICluster graph for one APA, one pass (active or masked).

### Structure

The file contains one pair of arrays per cluster index `i`:

```
cluster_<i>_nodes.npy   — node descriptor table
cluster_<i>_edges.npy   — directed edge pairs (src_idx, dst_idx)
```

### Node codes and per-code columns

| Code | Meaning | Key data columns |
|---|---|---|
| `s` | slice | start tick, tick span, total charge |
| `b` | blob | face, slice index, wire-pair bounds per plane; charge value + uncertainty (active) or zero (masked) |
| `m` | measure | per-plane charge measurement (active only) |
| `w` | wire | plane index, wire index |
| `c` | channel | channel ident |

Edges encode the cluster graph: blob↔slice, blob↔measure (active),
measure↔wire, wire↔channel.

Authoritative column layout:
`<toolkit>/aux/inc/WireCellAux/ClusterArrays.h`

### Quick inspection

```python
import numpy as np
d = np.load('work/evt2/icluster-apa0-active.npz')
print(list(d.keys())[:10])   # e.g. ['cluster_0_nodes', 'cluster_0_edges', ...]
```

> **Empty file**: a run with no blobs produces a 22-byte `.npz` (zip header
> only, no arrays). Downstream scripts detect and skip these.
> See [sbnd.md §"Known gotchas"](sbnd.md#known-gotchas).

---

## Notable details and gotchas

- **`add_dump=false`** — `per_anode` is called with `add_dump=false`
  (`wct-img-all.jsonnet:37`). This suppresses the inner `img.dump` node
  (`img.jsonnet:303–313`, which writes `clusters-apa-*.tar.gz` in JSON
  format). Only the top-level numpy `ClusterFileSink` nodes fire.

- **`experiment/sbnd/img.jsonnet` is self-contained** — it does not import
  `pgrapher/common/img.jsonnet` (unlike some PDHD/PDVD configs). All
  slicing, tiling, solving, and clustering helper functions are defined
  locally in that file.

- **`face = anode.data.ident`** (`img.jsonnet:167`) — SBND uses one face per
  anode (`GridTiling` face = 0 for APA0, 1 for APA1). The generic
  multi-face loop used in other detectors is commented out.

- **Deghosting depth is now a jsonnet toggle** — `img.solving()`,
  `imgpipe()`, and `per_anode()` in
  `cfg/pgrapher/experiment/sbnd/img.jsonnet` accept `full_deghost`
  (default `false`). `false` runs the historical "simple-solving"
  (one ChargeSolving triple + one `InSliceDeghosting` round);
  `true` runs the uBooNE-matched "uboone-solving"
  (two `ProjectionDeghosting` passes + three ChargeSolving triples +
  three `InSliceDeghosting` rounds). `sbnd_xin/wct-img-all.jsonnet`
  defaults `full_deghost=true`, so the standalone pipeline matches the
  uBooNE chain. Pass `--tla-code full_deghost=false` to revert.
  Production `wcls-img-clus.jsonnet` does not pass the flag, so it
  inherits `false` and is bit-identical to the prior behavior.
  See the next section for the algorithm-level comparison.

- **`FrameQualityTagging` not in pipeline** (`img.jsonnet:93–116`) — the
  node is defined (with `min_time: 3180`, `max_time: 7870`) but is not
  connected in the active pre_proc pipeline (`img.jsonnet:129`).

- **Geometry constants** — 5638 channels per APA and 3427-tick frame
  length were both production bugs in the shared configs that have been
  fixed on this branch. See
  [geometry-and-timing.md](geometry-and-timing.md) for details.

- **2-plane active tiling was silently disabled** (`img.jsonnet:342`,
  fixed 2026-04-25) — `imgpipe()` in the shared SBND config contained
  the condition `if multi_slicing == "multi-2view"` to select the
  4-branch `multi_active_slicing_tiling` fanpipe (branches for 3-view,
  U+V, V+W, U+W). Both call sites (`sbnd_xin/wct-img-all.jsonnet:37`
  and `cfg/pgrapher/experiment/sbnd/wcls-img-clus.jsonnet:50`) pass
  `"multi-3view"`, so the condition always fell through to the `else`
  — a single 3-plane branch. The 2-plane active branches ([0,1], [1,2],
  [0,2]) never ran, making the active output blind to tracks crossing
  dead wire regions. Symptom: tracks visible in U+V but crossing a
  W-dead band were absent from the active cluster file. The bad-channel
  mask (`chanmask_bad_<evt>.npy`) was correct — all 93 dead channels
  for evt2, including the prominent 32-channel W run at channels
  4160–4191, were properly flagged. The fix extends the condition to
  `if multi_slicing == "multi-2view" || multi_slicing == "multi-3view"`.
  Active blob count for evt2 / APA0 increased from 3,114 to 4,260 after
  the fix. The identical bug was present and fixed in
  `cfg/pgrapher/experiment/dune-vd/img.jsonnet:324`.

- **Bundled Q/L chain now images its own active clusters** (fixed
  2026-06-01) — `run_clust_QL_evt.sh` historically fed yuhw's precomputed
  LArSoft `icluster-apa*-active.npz` into the live view, so it did *not*
  benefit from the 2-plane-active fix above and showed thin charge across
  W-plane dead bands (e.g. physical evt2 / APA0, Z≈251 cm). The script now
  runs `wct-img-all.jsonnet` (multi-3view + `full_deghost=true`) on the
  same assembled 10-event SP-frame bundle to (re)produce the active npz
  in-toolkit before matching; only the opflash archives stay yuhw's (the
  light-matching reference). Result for physical evt2 / APA0: charge in the
  W-dead band z∈[250,255) roughly doubles (≈1,126 → ≈2,276 points in the
  all-APA view), and the bundled `mabc.zip` for that event is byte-identical
  to the per-event toolkit run (`run_ql_evt.sh` → `work/ql_evt<ID>/`). NB:
  the bundled run labels events by ordinal 0–9 (first event in the assembled
  order = index 0), independent of the physical event id.

---

## Comparison with the uBooNE imaging chain

Reference: `/home/xqian/work/scratch_wcgpu1/toolkit-dev/wcp-porting-img/wct-uboone-img.jsonnet`
(canonical uBooNE imaging job; the `fgval/uboone-val.jsonnet` is a quick-look
variant whose `solving` block is fully commented out and just runs
`BlobClustering → GlobalGeomClustering`, so it is not the right baseline for
algorithm comparison).

This section was written in response to the question "is deghosting fully
implemented in `sbnd_xin`, and are dead channels being read in correctly?"
TL;DR:

* **Deghosting is _not_ fully implemented in the imaging chain.** SBND's
  active-fork `solving()` is `simple-solving` — no `ProjectionDeghosting`,
  one `InSliceDeghosting` round, one `ChargeSolving` triple. uBooNE's
  `uboone-solving` runs **two `ProjectionDeghosting` passes**, **three
  `ChargeSolving` triples**, and **three `InSliceDeghosting` rounds**.
* **Dead channels _are_ correctly read in** at the imaging boundary:
  the `bad` CMM tag arrives in `chanmask_bad_<EVT>.npy` inside
  `sp-frames.tar.bz2`, flows through `FrameFileSource` →
  `ChannelSelector` → `CMMModifier` → `FrameMasking`, and is consumed
  by both `MaskSlices` and the masked-fork 2-view fanpipe. `CMMModifier`
  in SBND only does the "organize" step, however — the dynamic
  veto / dead-charge augmentation knobs that uBooNE configures are all
  commented out.

### Active-fork "solving" pipeline — line-by-line

`cfg/pgrapher/experiment/sbnd/img.jsonnet:299-300`:

```jsonnet
// ret: g.pipeline([bc, gd1, cs1, ld1, gd2, cs2, ld2, cs3, ld3, gc],"uboone-solving"),
ret: g.pipeline([bc, cs1, ld1, gc],"simple-solving"),
```

where each step is built from local builders at lines 218–297. Reading the
pipeline left-to-right:

| Step alias | Component | SBND (`simple-solving`) | uBooNE (`uboone-solving`) |
|---|---|---|---|
| `bc`  | `BlobClustering`               (policy=`uboone`) | ✅ | ✅ |
| `gd1` | `ProjectionDeghosting` round 1                    | **❌ skipped** | ✅ |
| `cs1` | `BlobGrouping → ChargeSolving(uniform) → LocalGeomClustering → ChargeSolving(uboone)` round 1 | ✅ | ✅ |
| `ld1` | `InSliceDeghosting` round 1 (`config_round=1`)    | ✅ | ✅ |
| `gd2` | `ProjectionDeghosting` round 2                    | **❌ skipped** | ✅ |
| `cs2` | charge-solving triple round 2                      | **❌ skipped** | ✅ |
| `ld2` | `InSliceDeghosting` round 2 (`config_round=2`)    | **❌ skipped** | ✅ |
| `cs3` | charge-solving triple round 3                      | **❌ skipped** | ✅ |
| `ld3` | `InSliceDeghosting` round 3 (`config_round=3`)    | **❌ skipped** | ✅ |
| `gc`  | `GlobalGeomClustering` (policy=`uboone`)          | ✅ | ✅ |

Builders `global_deghosting(...)`, `local_deghosting(config_round=...)`,
`solving(...)` are all _defined_ in the SBND `img.jsonnet` (lines 266–297),
so the missing steps are present in code — they are simply not wired into
the active pipeline.

Physical meaning of the missing components:

* **`ProjectionDeghosting`** drops blobs whose 2D wire-plane projections do
  not survive a coincidence test against the deconvolved charge
  distribution. It is most effective at trimming "ghost" blobs created when
  one true track produces tile-pattern false matches across the three views.
* **`InSliceDeghosting` rounds 2 and 3** rerun the same algorithm with
  progressively tighter `config_round` settings after the charge has been
  re-solved on the surviving blobs, refining which blobs survive.

Empirically, the WCP/uBooNE production chain has shown that the
multi-round combination of Projection- and InSlice- deghosting is what
cleans up the bulk of pixelation-induced ghost blobs. The current SBND
chain therefore likely overstates active blob count and total reconstructed
charge in busy events. Switching SBND to `uboone-solving` is a one-line
flip in `img.jsonnet`; per the project convention
([CLAUDE.md / toggleable behavior changes](notes.md)), that change should
be exposed as a jsonnet toggle defaulting OFF so existing production
configs remain bit-identical until validated.

### Dead-channel pathway — full audit

End-to-end the "bad" CMM tag enters the imaging graph here:

```
1. cfg/pgrapher/experiment/sbnd/chndb-base.jsonnet:63
   ↓  bad: [546, 607, 2781, 3232..3263, 4160..4191, 4374, 4800..4805,
   ↓        5060, 5231, 5636, 5637, 7167, 7169, 8378, 8395, 8574,
   ↓        10012, 10869, 10438..10443, 11147]   (≈ 92 channels)
2. NF chain (production) writes "bad" cmm in the SP-output frame
   ↓  In sbnd_xin: this archive comes from upstream LArSoft, so the
   ↓  NF/chndb step is *bypassed*. The "bad" map is delivered in
   ↓  the tarball as `chanmask_bad_<EVT>.npy`.
3. FrameFileSource(input='sp-frames.tar.bz2', tags=['dnnsp'])
   ↓  Reads chanmask_bad_<EVT>.npy and re-attaches it to the IFrame
   ↓  as the "bad" channel-mask map.
4. FrameFanout per-anode (wct-img-all.jsonnet:86-100)
   ↓  Renames frame tag '.*' → 'orig<N>' and trace tag 'dnnsp' →
   ↓  ['gauss<N>','wiener<N>'].  The channel-mask map flows through
   ↓  unchanged (the rule list only mentions frame/trace; CMM tags
   ↓  are not renamed).
5. ChannelSelector (chsel_pipes)
   ↓  Filters traces to channels [5638*N, 5638*(N+1)); the "bad" CMM
   ↓  is preserved on the surviving channels.
6. CMMModifier      cm_tag='bad'        (img.jsonnet:67-91)
7. FrameMasking     cm_tag='bad'        (img.jsonnet:118-127)
8. MaskSlices       wiener/charge/error tags driven by the masked-channel
                    information; multi_active_slicing_tiling +
                    multi_masked_2view_slicing_tiling fanpipes both fire.
```

All four `gauss<N>`-receiving nodes (`ChannelSelector`, `CMMModifier`,
`FrameMasking`, `ChargeErrorFrameEstimator`) get the correct anode and the
correct per-APA channel range. Both the **active fork** (`multi_active_…`,
branches 0–3 covering 3-view + the three 2-view combinations) and the
**masked fork** (`multi_masked_2view_…`, three dummy/masked permutations)
consume the bad-channel state via `active_planes` / `masked_planes` /
`dummy_planes`. So at the structural level **dead channels are correctly
delivered, masked, and turned into 2-view blob hypotheses in the masked
fork**.

### `CMMModifier`: what SBND does vs. what uBooNE does

`cfg/pgrapher/experiment/sbnd/img.jsonnet:67-91` keeps **only the "organize"
step**; all other knobs are commented out. uBooNE
(`wcp-porting-img/wct-uboone-img.jsonnet:57-81`) configures the full
feature set:

| Parameter | SBND | uBooNE |
|---|---|---|
| `cm_tag` | `'bad'` | `'bad'` |
| `trace_tag` | `'gauss<N>'` | `'gauss'` |
| `start` / `end` (veto window) | _disabled_ | 0 / 9592 |
| `ncount_cont_ch`, `cont_ch_llimit`, `cont_ch_hlimit` | _disabled_ | 2; `[296, 7136]`; `[671, 7263]` (veto on continuous bad runs) |
| `ncount_veto_ch`, `veto_ch_llimit`, `veto_ch_hlimit` | _disabled_ | 1; `[3684]`; `[3699]` (hard-coded veto channels) |
| `dead_ch_ncount`, `dead_ch_charge`, `ncount_dead_ch`, `dead_ch_llimit`, `dead_ch_hlimit` | _disabled_ | 10 / 1000 / 2 / `[2160,2080]` / `[2176,2096]` (charge-based dead-channel addition) |
| `ncount_org`, `org_llimit`, `org_hlimit` | 1; `[0]`; `[3427]` | 5; `[0,1920,3840,5760,7680]`; `[1919,3839,5759,7679,9592]` |

What's missing in SBND (and what each knob does):

* **Continuous-bad-channel veto** (`cont_ch_*`): when a stretch of `cont_ch_*`
  contiguous bad channels is found within the given channel-range gates,
  the dead range is widened with a fixed margin so neighbouring borderline
  channels aren't trusted.
* **Hard-veto channels** (`veto_ch_*`): explicit channels added to the bad
  list regardless of input.
* **Dynamic-dead-channel inference** (`dead_ch_*`): channels whose
  cumulative `gauss` charge exceeds `dead_ch_charge` in `dead_ch_ncount`
  ticks within the gates `[dead_ch_llimit, dead_ch_hlimit]` are
  retroactively flagged as dead (i.e., the algorithm refuses to trust their
  charge, presumably to catch railed/saturated channels).
* **Multi-segment "organize"**: uBooNE organises bad-channel rectangles
  per 1920-tick wire-plane segment; SBND treats the whole 3427-tick
  readout as a single segment.

These are all _augmentations_ on top of the chndb-supplied static bad list.
SBND just propagates the static list as-is. For a well-tuned NF that
emits a clean, accurate `chanmask_bad` this is fine; if there are
known classes of bad behavior that the NF cannot capture (saturation,
intermittent bad-channel pickup), they will not be cleaned up here.

### Other algorithm / config differences in the imaging chain

| Setting | SBND | uBooNE | Notes |
|---|---|---|---|
| Active-fork `tick_span` | 4 (= 2 µs) | 4 | identical |
| Masked-fork `span` | 500 (= 250 µs) | 1744 (= 872 µs) | SBND uses a finer slice — gives a denser masked-fork blob set but more memory |
| `MaskSlices.max_tbin` | 3427 | 9592 | drives off the readout-window length |
| `nthreshold` (per-plane) | `[3.6, 3.6, 3.6]` | `[3.6, 3.6, 3.6]` | identical |
| `ChargeErrorFrameEstimator.rebin` | 4 | 4 | |
| `ChargeErrorFrameEstimator.fudge_factors` (U/V/W) | `[2.31, 2.31, 1.1]` | `[2.31, 2.31, 1.1]` | identical — these are the uBooNE-tuned values inherited verbatim; worth re-deriving for SBND once SP comparisons stabilise |
| `ChargeErrorFrameEstimator.time_limits` (rebin-4 ticks) | `[12, 800]` | `[12, 800]` | identical; covers ticks 48–3200 — note SBND readout extends to 3427 so the tail 227 ticks are outside the error model |
| `WaveformMap.filename` | `sbnd-charge-error.json.bz2` | `microboone-charge-error.json.bz2` | SBND-specific file (commit 2023-10-17) |
| `GridTiling.face` | `anode.data.ident` (face 0 for APA0, face 1 for APA1) | `0` only (uBooNE has one anode) | SBND has 1 face per anode hard-wired |
| `BlobClustering.policy` | `uboone` | `uboone` | identical |
| `GlobalGeomClustering.policy` | `uboone` | `uboone` | identical |
| `ChargeSolving.solve_config`, `whiten` | `uboone`, `true` | `uboone`, `true` | identical (used in the rounds that do run) |
| Image-output schema | active + masked `.npz` per APA | active + masked `.npz` per APA | identical (uBooNE has only one APA, SBND has two) |

`FrameQualityTagging` exists as a definition in both files but is not in
either active `pre_proc` pipeline; the uBooNE pipeline composition is
`[cmm_mod, frame_masking, charge_err]` and SBND is
`[chsel_pipes, cmm_mod, frame_masking, charge_err]` — the only structural
delta is the per-APA channel selector that SBND needs and uBooNE does not.

### Action items implied by this audit

1. ~~Switch the active-fork `solving()` to `uboone-solving`~~ **Done.**
   `img.jsonnet`'s `solving` / `imgpipe` / `per_anode` accept
   `full_deghost` (default `false`); `sbnd_xin/wct-img-all.jsonnet`
   defaults `full_deghost=true`, so the standalone pipeline now runs the
   full uBooNE chain. Production `wcls-img-clus.jsonnet` inherits the
   `false` default and is unchanged. Closure-test verification: per
   anode, `full_deghost=true` instantiates 2 `ProjectionDeghosting`,
   3 `BlobGrouping`, 6 `ChargeSolving`, 3 `LocalGeomClustering`, and
   3 `InSliceDeghosting` nodes (vs 0 / 1 / 2 / 1 / 1 in `false` mode);
   confirmed by `jsonnet`-level node counts on both toggle states.
2. **Re-evaluate `time_limits=[12,800]`** for `ChargeErrorFrameEstimator`
   — the SBND readout is 3427 ticks ≈ 857 rebin-4 ticks, so the upper
   bound is currently 57 rebin-ticks short of the readout end. Either
   widen it to `~855` or document the conscious choice.
3. **Re-derive `fudge_factors`** from SBND-specific SP closure data
   rather than carrying the uBooNE-tuned `[2.31,2.31,1.1]`.
4. **Decide on `CMMModifier` augmentation** — if SBND has known classes
   of intermittent bad-channel behavior (railing, pickup), wire up the
   `cont_ch_*` / `veto_ch_*` / `dead_ch_*` machinery rather than relying
   solely on the static chndb list. Otherwise leave the current minimal
   config and document explicitly that this is intentional.

# wcfm doc 03 — F1 model artifacts and the sub-blob generator (BlobCutting)

Date 2026-09-24. Continues doc 02 (wcp `33334a17`, toolkit `69515f37`). Two deliverables:

- **F1 (doc 01 §7):** the two unified-student checkpoints copied from SDCC, the dense MBV3 student
  exported to TorchScript for `pytorch/TorchService`, eager-vs-scripted parity proven, the `.ts`
  published in `wire-cell-data/fm/dune10kt-1x2x6/` (wire-cell-data `bf4fd3f`; WC_FM_DINO
  `2f7bc43` = export/parity scripts + docs/40).
- **Sub-blob generator (doc 01 §4.6, doc 02 §7):** Xuyang Ning's `BlobCutting` (branch
  `XN_apply-pointcloud`, commit `19a7f8f5`, used by the in-tree `pdhd/img_simple.jsonnet`) ported to
  `apply-pointcloud` (toolkit `25edbe2a`) with a doctest, wired into the wcfm chain behind a
  default-OFF knob, knob-off gate PASS, and a first knob-on measurement of what the *legacy*
  deghosting chain does to sub-blobs (§6): it deletes them.

## 0. Repro

```bash
# --- F1 ---------------------------------------------------------------------------------------
# SDCC -> wcgpu1: non-interactive ssh commands on dunesub01 return nothing (exit 0) and sftp hangs,
# so the copy is an rsync PUSH from the SDCC side driven through a pty (needs the owner's live
# ssh-agent, e.g. SSH_AUTH_SOCK=/tmp/ssh-XXXX5ALTJb/agent.3483520 in this session):
ssh -tt -o BatchMode=yes dunesub01 < /home/xqian/tmp/wcfm-f1/transfer_cmds.txt     # sec 1
sha256sum -c /home/xqian/toolkit-dev/WC_FM_DINO/kd_checkpoints/SHA256SUMS
cd /home/xqian/toolkit-dev/WC_FM_DINO
PY25=/home/xqian/toolkit-dev/.direnv/python-3.11.9/bin/python3                     # torch 2.5.1
$PY25 sdcc/export_mbv3_ts.py --out /home/xqian/tmp/wcfm-f1/kd_uni_mbv3_a1_inf01.ts   # sec 2
.venv-dino/bin/python sdcc/fm_eager_reference.py --out-dir /home/xqian/tmp/wcfm-f1  # torch 2.10
$PY25 sdcc/check_mbv3_ts_parity.py --ts /home/xqian/tmp/wcfm-f1/kd_uni_mbv3_a1_inf01.ts \
      --ref-dir /home/xqian/tmp/wcfm-f1 --out /home/xqian/tmp/wcfm-f1/parity.json   # sec 3
(cd /home/xqian/tmp/wcfm-f1 && ln -sf kd_uni_mbv3_a1_inf01.ts model.ts && \
     /home/xqian/toolkit-dev/toolkit/build/pytorch/check_load_tsmodel)               # libtorch 2.8
# --- BlobCutting ------------------------------------------------------------------------------
cd /home/xqian/toolkit-dev/toolkit && wcbuild && ./build/img/wcdoctest-img            # sec 4
cd /home/xqian/toolkit-dev/wcp-porting-img/wcfm
scripts/w3_gate.sh -1 wcfm-sb-off wcfm-w3-d                                          # sec 5 gate
WCFM_MAX_JOBS=3 WCFM_SETARCH=1 ./run_img_evt.sh -C -O _sub 1 all                    # sec 6 knob on
python3 scripts/iso_baseline.py work/000001_{1,2,3,4,5,6,7,8,9,10}_sub --out docs/03_tables
```

Revisions: toolkit `25edbe2a` (on `69515f37`), wire-cell-data `bf4fd3f`, WC_FM_DINO `2f7bc43`
(on `4eab794`), wcp = this commit. Scratch: `/home/xqian/tmp/wcfm-f1/` (transfer log, export and
parity logs, `parity.json`, eager references `eager_ref_{U,V,W}.npz`, gate log).

## 1. F1a — checkpoints

| run | file | bytes | sha256 (remote = local) |
|---|---|---|---|
| `kd_uni_mbv3_a1_inf01` (dense MBV3, docs/37 M2) | `checkpoint_step36000.pt` | 49 126 754 | `5ed9c290…d771e1440` |
| `kd_uni_a1_inf01` (sparse CAP, docs/37 U4) | `checkpoint_step36000.pt` | 85 040 134 | `d0feace2…24eed542` |

Source `dunesub01.sdcc.bnl.gov:/gpfs/mnt/gpfs01/dayabay/xqian/DUNE/WC_FM_DINO/CONDOR_OUT/kd_campaign/checkpoints/<run>/`
(eight checkpoints per run, step 4500…36000; 36000 = `max_steps`, the final save), plus each run's
`kd_config.json`. Destination `WC_FM_DINO/kd_checkpoints/<run>/` (gitignored; `README.md` and
`SHA256SUMS` tracked). Full shas in `SHA256SUMS`.

**Access finding.** The owner's `~/.ssh/config` reaches `dunesub01` through `ssh.sdcc.bnl.gov`
with the passphrase-protected `id_rsa`; a live agent in the owner's session holds it and the
login authenticates. On `dunesub01` an `exec` request is accepted but produces no output and exits
0 (the gateway runs commands normally; `~/.bashrc` on the shared home is the stock Fedora one, so
the cause is on the submit host, not in the dotfiles); `sftp` hangs. A pty session (`ssh -tt` with
the commands on stdin) works, and from it `rsync … xqian@wcgpu1.phy.bnl.gov:` with the SDCC-side key
(doc 09 / doc 29 route) copied 134 MB in under a minute. `rsync dunesub01:… .` from wcgpu1 cannot
work until the submit-host shell is fixed.

## 2. F1b — export

`WC_FM_DINO/sdcc/export_mbv3_ts.py` (run under the toolkit direnv python, torch 2.5.1 ≤ the
toolkit's libtorch 2.8; the training venv is torch 2.10):

- loads `models/mobilenetv3_unet.py` by file path (importing `models` pulls warpconvnet in, absent
  in the export env); unpickles the checkpoint (`KDConfig` from `dino/distill_config.py`, a
  dataclass); asserts `backbone_name == mbv3_mae`, `feature_dim == 128`; rebuilds
  `MobileNetV3_UNet(2, 128, "large", pretrained=False)` from the `net.` keys with `strict=True`
  (`charge_head.*`, `occupancy_head.*` are the only keys dropped);
- scripts `FMDenseStudent`: the arithmetic of `_forward_body` with the encoder split into the five
  stages recorded in `net._feat_indices` = `[0, 2, 4, 7, 13]` instead of the `enumerate/break`
  loop that `torch.jit.script` rejects (the DNN-ROI export in `wire-cell-data/dnnroi/pdhd/README.md`
  fell back to `trace` for that reason). Encoder layers 14–16 (1.75 M parameters) were never run by
  the original forward either, so the deployed model has **3.45 M** parameters (docs/37's 5.20 M
  counts them and the two heads). The `UpBlock` shape branch is kept, which is why `script`, not
  `trace`: a traced model would bake one canvas size;
- eager wrapper vs the original `net(x, return_logits=True)`: bit-identical on a random 96 × 130
  canvas; scripted vs eager: bit-identical; reload round trip: bit-identical;
- saved from eval mode (TorchService never calls `.eval()`), NCHW, fp32, single output, not frozen
  (`pytorch/src/Util.cxx` `to_itensor` memcpys without `.contiguous()`, so channels_last is out),
  with `fm_meta.json` as a TorchScript extra file and a sibling `<name>.meta.json`: checkpoint sha,
  `kd_config.json`, `KDConfig`, torch/torchvision versions, WC_FM_DINO git sha, `_feat_indices`,
  the contract, `VIEW_NORM`, min canvas.

Contract (unchanged from doc 01 §4.1–4.2): input `[1, 2, H, W]` = (`FeatureLogTransform(q)` with
the plane's `VIEW_NORM`, 0/1 active mask), tight bbox floored at 64; output `[1, 128, H, W]`
gathered at active pixels. No normalisation and no sigmoid inside the model.

## 3. F1c — parity, load checks, latency

Reference = the *real* forward: `DenseBackboneMAE.forward` built by
`dino.diagnostics.ab_pid_probe.build_backbone` in the training venv (torch 2.10, warpconvnet), the
plane's `VIEW_NORM` applied as `kd_probe_direct.py` does, CPU, on events 0–19 of
`packed_numu_truth_apa0_{U,V,W}_20k.npz`; the dump carries the exact canvas `_rasterize_one` built
and the gathered `[N, 128]` output (`sdcc/fm_eager_reference.py`). `sdcc/check_mbv3_ts_parity.py`
runs the `.ts` on those canvases in the export env.

| comparison (60 events) | max abs | mean abs | min cosine |
|---|---|---|---|
| (a) scripted (torch 2.5.1, CPU) vs the 2.10 adapter forward | 2.0e-5 | 2.4e-8 | 0.99999964 |
| (b) scripted vs eager `FMDenseStudent` in the export env | 0 | 0 | 0.99999964 (fp32 cosine of identical vectors) |
| scripted on GPU (RTX 4090, cuDNN defaults, TF32 conv) vs (a)'s reference | 9.2e-3 | 2.9e-4 | 0.999992 |

Per plane: U 2.0e-5 / V 2.0e-5 / W 1.6e-5 (a). Gate (doc 01 F1: max abs < 1e-4 fp32, cosine
> 0.9999): **PASS**. `|feat|` mean is 0.46–0.77, so 2e-5 is ~4e-5 relative; the GPU 1e-2 is the
usual TF32 convolution path and is informational (doc 01 F3 sets the GPU bar at 1e-3 — to be met
by disabling TF32 in TorchService or accepted at 1e-2; decision deferred to F3).

Load checks: `build/pytorch/check_load_tsmodel` (libtorch 2.8.0 shim) prints "model model.ts
loaded"; `torch.jit.load` + forward under torch 2.10 OK.

Latency of the scripted model on the 60 full-anode canvases (median 760 × 1343, max 826 × 1497;
this box under load ≈ 20): CPU 1372 / 487 / 423 ms per canvas at 1 / 8 / 16 threads; GPU 14 ms.
These are whole-anode canvases; the wcfm iso events are much smaller. Doc 33's 238 → 54 ms was on
an idle EPYC 9355 with tight per-event boxes.

Files: `wire-cell-data/fm/dune10kt-1x2x6/{kd_uni_mbv3_a1_inf01.ts, kd_uni_mbv3_a1_inf01.meta.json,
README.md, SHA256SUMS}` (`.ts` sha256 `883e4948…7087dd36`, 14.2 MB). The CAP student is
checkpoint-only (Triton, F6).

## 4. The sub-blob generator: `img/BlobCutting` (toolkit `25edbe2a`)

### 4.1 What Xuyang wrote

Branch `XN_apply-pointcloud` = one commit `19a7f8f5` ("cfg and others", 2026-04-28, 199 files) on
merge base `d4c23bc0`. `img/src/BlobCutting.cxx` + header: an `IFunctionNode<IBlobSet, IBlobSet>`
that recursively bisects any blob with a U/V/W strip wider than `length_threshold` (20) at the
mid-point of its widest wire-plane strip (`RayGrid::Blob::add` on the two half-open halves,
`drop_invalid → prune → drop_invalid`, keep the halves that are `valid()`, keep the parent if none
is), down to `max_depth` (10). `pdhd/img_simple.jsonnet` (already in-tree, identical to hers, from
the 2026-04-24 config sync) puts it between `BlobSetSync` and `BlobClustering` in its `"single"`
pipeline with `length_threshold 20, max_depth 100`; nothing imported it and the component had no
source on `apply-pointcloud`.

This is the geometric form of doc 01 §4.6 item 2 / GNN doc §4.2: because a tiling strip is a
contiguous run of *active* wires, bisecting strips is the same as re-tiling the blob's own
activity at a coarser granularity, without an `Activity` rebin. It gives the L1/L2 level in one
knob (`length_threshold` = k).

### 4.2 What changed in the port (and what did not)

Kept unchanged: the split rule, the post-processing sequence, the keep-the-parent rule, the
config keys and their defaults (`length_threshold` 20, `nudge` 0.01, `max_depth` 10), the
pass-through of uncut blobs as the same `IBlob`, the equal charge split (`value / n`).

Changed (each a defect in the original):

| original | port |
|---|---|
| `print_blob(new_blobs[0]); print_blob(new_blobs[1]);` with no size check (UB when a half has no corners) | removed; one `SPDLOG_LOGGER_TRACE` per cut blob, one `DEBUG` per blob set with `in/cut/sub/out` counts |
| unconditional `std::cout` on every split | none |
| helper functions with external linkage in `libWireCellImg` | anonymous namespace |
| `new_blob_id` restarts at **0 for every blob set** → sub-blob idents collide with retained GridTiling idents and repeat across slices (idents key unordered containers downstream, see `doctest_gridtiling_event_reset.cxx`) | per-frame counter from `ident_base` (`1<<20`, above GridTiling's per-event count), restarted at every frame boundary and at EOS, like GridTiling |
| hard-coded "widest strip ≤ 2 → stop" | `min_length` key (default 2) |
| header said "pass-through behavior" | header documents the algorithm and the non-goals |

Not taken from her branch (listed so nobody re-imports them by accident): active `std::cerr`
debug in `MultiAlgBlobClustering`, `clustering_live_dead`, `DynamicPointCloud`, `PlaneTools`,
`ClusterFileSource`, `NumpyDepoTools`, `custard_stream.hpp`; `BlobClustering.cxx` L119
(`geom_clustering`) commented out — her "single" outputs have **no blob-blob edges**, ours do;
`cfg/pgrapher/ui/wcls/nodes.jsonnet` reverting upstream `74dcc528`; `clustering_deghost2.cxx`
(+ two `.bkup` files) and the new `cfg/pgrapher/common/{pgraph,fileio,helpers}.jsonnet`; the
pdhd/protodunevd/dune10kt-* config churn. One item worth its own look later: her
`custard_stream.hpp` change also skips empty `uname`/`gname` in `write()` (a real fix, not in our
tree).

Doctest `img/test/doctest_blobcutting.cxx` (uboone test anode; 60-channel-wide activity):
sub-blobs all ≤ threshold wide, each inside its parent, unique idents ≥ `ident_base`, parent
charge conserved to 1e-4; a 12-channel slice and a 1000-wire threshold return the same `IBlob`
pointers; idents continue within a frame, restart on a new frame ident and after EOS. Build:
the stale-lib link trap (a doctest using a new symbol links against the installed old lib; broken
with `./wcb build -k` + `install -k`, then `wcbuild`); `wcdoctest-img` 195 assertions pass;
`local/lib/libWireCellImg.so` 09:38:51 > source 09:38:29.

### 4.3 The wcfm knob

`wcfm/img.jsonnet` `config.blob_cutting` (false) / `cut_length` (20) / `cut_max_depth` (10); when
on, `solving()` prepends `BlobCutting` (`blobcutting-<aname>`) ahead of `BlobClustering` on the
three-plane "active" path only (the masked 2-view fork is never cut: a dummy-plane strip would be
bisected like a real one). `wct-img-all.jsonnet` TLAs `blob_cutting`, `cut_length`,
`cut_max_depth`; `run_img_evt.sh -C [-L wires]`, provenance line `blob_cutting=`. Compiled-config
proof: knob off, `wcsonnet` output for event 1 anode 10 with and without `depos` is byte-identical
to the pre-edit compile (`cmp`); knob on, the graph has
`BlobSetMerge → BlobCutting:blobcutting-anode10-ms-active → BlobClustering`.

## 5. Gates

| gate | result |
|---|---|
| `abtest/snap/wcfm-sb-off` (this tree, knob off, `setarch -R`, 10 events img+clus) vs `wcfm-w3-d` (doc 02) | **PASS**, 178 archives identical, 24 rc=0. Note `wcfm-w3-d` (07:50) predates the 08:02 reinstall of the unchanged fix sources; this single run therefore also proves that rebuild. |
| `./build/img/wcdoctest-img` | 195 assertions, 0 failed (2 new cases, 144 assertions) |
| production byte-identity | by construction: the toolkit commit adds three new files and touches no existing one (`git show --stat 25edbe2a`); the standard PDHD/PDVD manifest is not rerun. |
| F1 parity | §3, PASS |

## 6. Knob-on: sub-blobs through the legacy chain (`work/000001_*_sub/`)

`run_img_evt.sh -C` on all 10 events (`length_threshold` 20, `max_depth` 10), same truth tiers as
doc 02 (`tru0` = every blob after `BlobClustering`, now sub-blobs; `tru` = survivors after the
full deghost/solve chain). All 23 anode jobs rc=0; wall 4–17 s (doc 02: 4–11 s; event 1 anode 10
17 s vs 6 s), RSS 461–466 MB (unchanged). Cut statistics from the logs:

| event | blob sets cut | blobs cut | sub-blobs | largest cut set (sub-blobs) |
|---|---|---|---|---|
| 1 (0°) | 11 | 107 | 2434 | 551 |
| 2 (0°) | 14 | 651 | 3746 | 582 |
| 3 (2°) | 36 | 488 | 2386 | 108 |
| 4 (5°) | 80 | 99 | 213 | 6 |
| 5 (10°) | 79 | 124 | 248 | 8 |
| 6 (×2) | 36 | 743 | 5699 | 723 |
| 7 (×3) | 62 | 1426 | 8390 | 571 |
| 8 (×4) | 107 | 1169 | 7680 | 740 |
| 9, 10 (cosmic) | 0 | 0 | 0 | — |

The cosmics are untouched (max strip 16 < 20), i.e. the threshold *is* the doc 02 `T_wires`
trigger at blob level. `docs/03_tables/iso_baseline_summary.md` (same columns as doc 02):

| event | kind | θ | ntrk | blobs tru0 | blobs final | ghost0 | ghost | max strip U/V/W | median blobs/slice | captured | q_reco/q_true−1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | iso | 0 | 1 | 20255 (17928) | 2463 (1341) | 0.890 (1.000) | 0.783 (0.998) | 28/31/35 (169/100/161) | 144 (114) | 0.272 (0.270) | +2.589 (−0.009) |
| 2 | iso | 0 | 1 | 10289 (7194) | 3694 (4550) | 0.629 (0.840) | 0.613 (0.766) | 23/20/19 (98/628/433) | 251 (364) | **0.382 (0.989)** | +1.566 (−0.008) |
| 3 | iso | 2 | 1 | 5757 (3859) | 2198 (2439) | 0.485 (0.711) | 0.450 (0.780) | 20/20/20 (69/92/78) | 53 (50) | **0.645 (0.993)** | +0.559 (+0.016) |
| 4 | iso | 5 | 1 | 596 (482) | 534 (426) | 0.594 (0.726) | 0.556 (0.697) | 10/20/19 (10/54/27) | 5 (4) | 0.981 (0.980) | +0.002 (+0.001) |
| 5 | iso | 10 | 1 | 1678 (1554) | 1408 (1329) | 0.378 (0.409) | 0.386 (0.426) | 20/19/20 (32/19/22) | 7 (7) | 0.980 (0.979) | −0.002 (−0.002) |
| 6 | iso | 1 | 2 | 11584 (6628) | 2091 (4872) | 0.753 (0.948) | 0.694 (0.967) | 30/31/37 (233/517/448) | 46 (27) | **0.270 (0.970)** | +2.559 (−0.006) |
| 7 | iso | 2 | 3 | 18367 (11403) | 4956 (6090) | 0.426 (0.627) | 0.354 (0.650) | 20/24/20 (241/203/141) | 60 (78) | **0.435 (0.805)** | +0.813 (−0.021) |
| 8 | iso | 5 | 4 | 22758 (16247) | 5587 (9379) | 0.650 (0.837) | 0.470 (0.845) | 27/28/31 (334/391/379) | 45 (29) | **0.719 (0.991)** | +0.381 (−0.004) |
| 9 | cosmic | — | 1 | 1210 (1210) | 1155 (1155) | 0.027 | 0.023 | 6/9/6 | 1 | 0.995 | +0.022 |
| 10 | cosmic | — | 2 | 1984 (1984) | 1806 (1806) | 0.143 | 0.122 | 11/16/9 | 2 | 0.984 | +0.014 |

(doc 02 values in parentheses; "captured" = Σq_true over final blobs / Σq_depo; ghost = fraction
of blobs with q_true == 0; a relative label `q_true < 0.05 q_reco` gives 0.44–0.79 on the iso
events.) Per-slice trigger statistics on the `tru0` tier, iso (601 slices) vs cosmic (1640):

| | max strip median / p90 / p99 / max | blobs per slice median / p90 / p99 / max |
|---|---|---|
| doc 02 iso | 31 / 88 / 447 / 628 | 11 / 179 / 1984 / 4367 |
| doc 03 iso, cut at 20 | 19 / 20 / 31 / 37 | 13 / 393 / 2079 / 4367 |
| cosmic (both) | 7 / 11 / 14 / 16 | 1 / 4 / 7 / 10 |

Readings:

1. **The generator does what it must.** On the `tru0` tier every iso slab is now a set of cells
   ≤ 20 wires wide (p99 31, max 37; the 21–37-wide leftovers are 2.4 % of event 1 anode 10's blobs
   and 3 % of event 8 anode 11's — parents whose two halves both fail `valid()` after `prune`, kept
   whole by design), still capturing the true charge (`captured0` 0.98–1.02 except event 7's 0.82,
   the doc 02 §6.6 uncaptured 19 %), and the `tru0` ghost fraction *drops* on every iso event
   (event 1: 1.000 → 0.890; event 2: 0.840 → 0.629; event 8: 0.837 → 0.650) because true cells
   are now separable from empty ones at 20-wire granularity. That is the fine-level labelled cell
   set the F5 probe needs, and `BlobDepoFill` labels it in-job (doc 01 §4.6 item 3). 2434–8390
   sub-blobs per event, ≤ 740 per slice, is well inside the GNN doc §4 budget.
2. **The legacy chain must not see them.** Fed to `ProjectionDeghosting`/`ChargeSolving`/
   `InSliceDeghosting`, the sub-blobs make the chain delete *more* true charge, not less: the
   captured fraction of the survivors collapses on events 2, 3, 6, 7, 8 (0.99 → 0.38, 0.99 →
   0.65, 0.97 → 0.27, 0.81 → 0.44, 0.99 → 0.72) while event 1's slab stays deleted (0.27), and the
   solved charge on the survivors overshoots their true charge by 0.4–2.6× (the measured wire
   charge is redistributed onto whatever cells remain). The deghosters' projective-consistency
   scores are calibrated on coarse blobs; many small cells sharing the same wires look like ghosts
   to them. So the sub-blob level is for the FM/GNN stage and for labels, and the write-back into
   the production path must be the soft penalty of doc 01 §4.6 item 4, not a replacement of the
   tiled blobs — exactly the ordering doc 01 prescribes, now with numbers.
3. The trigger stays: at blob level `length_threshold` 20 cuts nothing on the cosmics; at slice
   level `T_cells` 100 on the cut tier flags 191 of 601 iso slices (129 before) and 0 cosmic.
4. Cost: +0–11 s wall per anode, no RSS change. The knob is cheap enough to run on every wcfm event.

## 7. Open items

- F3 GPU parity bar: TF32 gives 1e-2 vs the 1e-3 doc 01 wrote; decide (disable TF32 in
  TorchService via `at::globalContext().setAllowTF32CuDNN(false)` behind a knob, or accept).
- F2 next: `wcfm/scripts/fm_oracle.py` on `wcfm/work/*/sim-frames-anode*.tar.bz2` reusing
  `check_mbv3_ts_parity.py`'s gather; the scale constant against the `_20k` pack pixels.
- The SDCC submit-host shell swallows non-interactive commands; every future copy goes through
  the pty push (`transfer_cmds.txt` pattern) until that is fixed on the SDCC side.
- BlobCutting: leftovers 21–37 wide (both halves invalid) — a second attempt at a different plane
  would remove them; not needed for F5. Charge split `value / n` is area-blind (moot: `value` is 0
  at tiling time). `pdhd/img_simple.jsonnet` still runs it on everything in its "single" flavor.
- The wcfm `_sub` work dirs (`work/000001_*_sub/`) are the doc 03 §6 record; the fine-level tier is
  `clusters-tru0-anode<N>-ms-active.tar.gz` there.
- Xuyang's `custard_stream.hpp` empty-`uname`/`gname` fix (§4.2) — separate change if wanted.

## 8. Files

Toolkit `25edbe2a`: `img/inc/WireCellImg/BlobCutting.h`, `img/src/BlobCutting.cxx`,
`img/test/doctest_blobcutting.cxx`. wire-cell-data `bf4fd3f`: `fm/dune10kt-1x2x6/*`. WC_FM_DINO
`2f7bc43`: `sdcc/export_mbv3_ts.py`, `sdcc/fm_eager_reference.py`, `sdcc/check_mbv3_ts_parity.py`,
`kd_checkpoints/{README.md,SHA256SUMS}`, `.gitignore`, `docs/40_torchscript_export_mbv3.md`. wcp
(this commit): `wcfm/img.jsonnet`, `wcfm/wct-img-all.jsonnet`, `wcfm/run_img_evt.sh`,
`wcfm/scripts/w3_gate.sh` (`-1` mode), `docs/03_*.md`, `docs/03_tables/*`, `docs/README.md`.
Snapshots: `abtest/snap/wcfm-sb-off/`.

# PDVD DNN-ROI run workflow

Standalone (no art/LArSoft) NF + SP + DNN-ROI for ProtoDUNE Vertical Drift,
mirroring `pdhd/run_nf_sp_dnnroi_evt.sh`. The DNN-ROI step replaces the
induction-plane signal ROIs with a MobileNetV3-UNet model output.

Full design notes: `DNN_ROI_SP/docs/pdvd_wirecell_deployment.md`.

## Files

| file | role |
|---|---|
| `run_nf_sp_dnnroi_evt.sh` | runner: NF → SP → DNN-ROI for one event |
| `wct-nf-sp-dnnroi.jsonnet` | top-level pipeline (per-anode src → resampler → NF → SP → DNN-ROI → sink) |
| `cfg/.../protodunevd/dnnroi_pp.jsonnet` | per-anode DNN-ROI subgraph (`DNNROIFinding`, per-plane sequential) |
| `wire-cell-data/dnnroi/pdvd/*.ts` | the deployed TorchScript models |

The DNN-ROI models are 6-channel (`loose_lf, mp2_roi, mp3_roi, tight_lf,
decon_charge, gauss`) and are called per-plane: U and V each feed the model
in their own (1, 6, ~476, 1600) call, serialized through one TorchService;
the W collection plane is passed through from standard SP gauss.

The PDVD MobileNetV3-UNet has 5 stride-2 levels post-rebin, so the C++
`DNNROIFinding` node is configured with `tick_pad_multiple=128`: it pads
input `nticks` up to the next 128-multiple before inference and crops the
output back to the original length. See `wire-cell-data/dnnroi/pdvd/README.md`
for the per-`nticks` padding table.

## Run DNN-ROI

```
# all 8 anodes, default best-KD model:
./run_nf_sp_dnnroi_evt.sh -D cpu 039324 0

# single anode / specific model / GPU:
./run_nf_sp_dnnroi_evt.sh -a 0 -D gpu -M dnnroi/pdvd/base_mbv3_transformer_4ch.ts 039324 0

# C++ per-call (input,output) debug dump for offline replay:
./run_nf_sp_dnnroi_evt.sh -a 0 -X dnn_debug 039324 0
```

Output (`work/<RUN_PADDED>_<EVT>/`):
`protodune-sp-dnnroi-frames-anode{N}.tar.bz2` — a standard SP-style archive.
Trace tags depend on whether L1SP is wired:

- **L1SP on (default)**: `gauss{N}` and `wiener{N}`, both from the post-DNN
  `L1SPFilterPD`. This is what the imaging/clustering chain needs.
- **L1SP off** (`-L off`): `gauss{N}` only (raw DNN-ROI output). No
  `wiener{N}`, so these frames are **not** directly imageable — imaging's
  `MaskSlice` requires matching `gauss`/`wiener` and otherwise throws
  `charge_traces.size()!=wiener_traces.size()`.

Options: `-a` anode (default all 8), `-r` data|sim, `-D` cpu|gpu, `-M` model
`.ts` (resolved via `WIRECELL_PATH`), `-X` debug-dump basename, `-N`
process|dnn|hybrid (L1SP mode, default process), `--loose-heur` (loosen L1SP
pre-filters). The QAT INT8 model is CPU-only.

## Magnify ROOT

```
./run_sp_to_magnify_evt.sh -d 039324 0
```

`-d` reads the `protodune-sp-dnnroi-frames` archives and writes the standard
`hu/hv/hw_{gauss,wiener,threshold}{N}` histograms — `gauss` is the DNN-ROI
output, `threshold` the per-channel Wiener threshold (TH1F); output ROOT name
gets a `-dnnroi` suffix.

## Validate vs standalone

`run_nf_sp_dnnroi_evt.sh -X <base>` dumps `<base>_anode{N}_call{K}.pt`. Replay
through `DNN_ROI_SP/scripts/verify_wirecell_dnn.py` to confirm the toolkit C++
node reproduces standalone PyTorch inference.

## End-to-end: DNN-ROI → imaging → clustering

The DNN runner writes `protodune-sp-**dnnroi**-frames-anode{N}.tar.bz2`, and
`run_img_evt.sh` now reads those **directly by default** (`-d on`, the standard
chain) — no `protodune-sp-frames` symlink needed. Imaging requires both `gauss`
and `wiener` tags, which the post-DNN `L1SPFilterPD` provides (L1SP on default):

```
run=039324; evt=0
./run_nf_sp_dnnroi_evt.sh -D cpu -P fp32 -r data -N hybrid --loose-heur $run $evt
./run_img_evt.sh  $run $evt     # -d on default -> reads protodune-sp-dnnroi-frames
./run_clus_evt.sh $run $evt     # -> mabc-anode{N}*.zip + mabc-all-apa.zip
```

A per-run Bee link (all events of a run in one set) is then:

```
./run_bee_combined_evt.sh $run  # builds data/<evt>/ for every event, zips, uploads, prints the URL
```

**No more silent non-DNN fallback.** `run_img_evt.sh` defaults to `-d on` and
reads the DNN-ROI frames from the work dir directly; if they are absent it
**errors loudly** (`[skip] ... run run_nf_sp_dnnroi_evt.sh first`) instead of
quietly imaging the traditional `protodune-sp-frames`. To deliberately image the
traditional/loose-ROI SP frames, pass `-d off`. (Previously the DNN flow relied
on a `protodune-sp-frames -> protodune-sp-dnnroi-frames` symlink, and a missing
symlink silently produced non-DNN clusters; the `-d on` default removes that
trap. The symlink is still honoured under `-d off` if present.)

**Gotcha — parallel `wire-cell` gets reaped.** Running ~4 DNN+L1SP `wire-cell`
processes at once on the shared box has them SIGKILLed (`Killed`) in lockstep —
not memory (a single event peaks <2 GiB), apparently an external watchdog/
process cap. Run SP serially for heavy (cosmic, `nticks=8000`) runs.

## Notes

- Anodes 0–3 (bottom drift volume) are the training corpus (crp1–4); anodes
  4–7 (top drift volume) are out-of-domain.
- L1SP-after-DNN now defaults **ON** in `run_nf_sp_dnnroi_evt.sh` (matches
  `run_nf_sp_evt.sh`); mode comes from `-N` (default `process`), pass `-L off`
  to disable. L1SP supplies the `wiener` tag the imaging chain needs; with
  `-L off` only the raw DNN `gauss` survives (used for `-X` debug input dumps).

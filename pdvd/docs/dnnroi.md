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
| `cfg/.../protodunevd/dnnroi_mp.jsonnet` | per-anode DNN-ROI subgraph (`DNNROIFindingMultiPlane`) |
| `wire-cell-data/dnnroi/pdvd/*.ts` | the 3 deployed TorchScript models |

The DNN-ROI models are 4-channel (`loose_lf, mp2_roi, mp3_roi, gauss`) and feed
the two induction planes stacked (U+V) to one model call; the W collection
plane is passed through from standard SP gauss.

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
`protodune-sp-dnnroi-frames-anode{N}.tar.bz2` — one frame tagged `dnnsp{N}`.

Options: `-a` anode (default all 8), `-r` data|sim, `-D` cpu|gpu, `-M` model
`.ts` (resolved via `WIRECELL_PATH`), `-X` debug-dump basename. The QAT INT8
model is CPU-only.

## Magnify ROOT

```
./run_sp_to_magnify_evt.sh -d 039324 0
```

`-d` reads the `protodune-sp-dnnroi-frames` archives and writes
`hu/hv/hw_dnnsp{N}` histograms; output ROOT name gets a `-dnnroi` suffix.

## Validate vs standalone

`run_nf_sp_dnnroi_evt.sh -X <base>` dumps `<base>_anode{N}_call{K}.pt`. Replay
through `DNN_ROI_SP/scripts/verify_wirecell_dnn.py` to confirm the toolkit C++
node reproduces standalone PyTorch inference.

## Notes

- Anodes 0–3 (bottom drift volume) are the training corpus (crp1–4); anodes
  4–7 (top drift volume) are out-of-domain.
- L1SP-after-DNN is not wired; L1SP inside SP is OFF in the DNN chain so the
  DNN-ROI debug input tags survive.

# wcfm — Wire-Cell foundation model (FM) integration

Runner scripts, jsonnet forks and docs for bringing the trained 2D foundation model
(`WC_FM_DINO`, DUNE FD-HD 1x2x6 "workspace" sim) into the toolkit chain. Design inputs live in
`wc-pr-ml-thoughts/` (GNN blob-deghosting design, PR-ML ideas, TriCross study, owner slides).

| Doc | Content |
|---|---|
| [01_fm-integration-campaign.md](01_fm-integration-campaign.md) | Campaign design (2026-09-24): state of the model and the toolkit, the owner's four decisions, the feature-sidecar contract, the FM stage between SP and imaging, the ctpc join, the isochronous path, the memory plan, the workspace chain (W1–W5) and FM integration steps (F1–F6) with gates |
| [02_workspace-chain-and-iso-baseline.md](02_workspace-chain-and-iso-baseline.md) | W1–W5 executed (2026-09-24): four toolkit bugs fixed (dune10kt APA centerline, empty channel-mask frame files, `BlobDepoFill` on −x faces, blob-less cluster files), the iso-track gun sim → NF → SP → imaging (+`BlobDepoFill` truth tiers) → two-face-volume clustering → pctree chain in `wcfm/`, the `time_offset` calibration (64.5 µs), the determinism gate (`wcfm-w3-c/d`), and the isochronous baseline: 43–100 % ghosts among surviving blobs on iso slices vs 2–12 % on cosmics, one whole track deleted by `ProjectionDeghosting`, triggers `T_wires 30` / `T_cells 100` validated |
| [03_fm-artifacts-and-sub-blob-generator.md](03_fm-artifacts-and-sub-blob-generator.md) | F1 done (2026-09-24): both `kd_uni_*` checkpoints copied from SDCC (pty-driven rsync push; shas), the dense MBV3 student scripted (`torch.jit.script`, 3.45 M params) with parity 2e-5 / cosine 0.9999996 against the torch-2.10 adapter forward and published in `wire-cell-data/fm/dune10kt-1x2x6/`; Xuyang's `BlobCutting` sub-blob generator ported to the toolkit (ident scheme, logging, doctest), wcfm knob `blob_cutting` default OFF (gate `wcfm-sb-off` vs `wcfm-w3-d` PASS); knob-on measurement: sub-blobs give a fine-level labelled cell set (tru0 ghost fraction drops, strips ≤ 20 wires) but the legacy deghosting chain deletes them (captured charge 0.99 → 0.27–0.72), so they are for the FM/GNN stage, not the production path |
| [04_fm-oracle-extract-and-probe.md](04_fm-oracle-extract-and-probe.md) | F2 + F3 + F5 (2026-09-24): the packing constants fixed from the packs themselves (pixel = 0.25 × Σ₄ ticks, active ⇔ > 0; every pack value is a multiple of 0.25), the Python oracle `fm_oracle.py`, the C++ stage `pytorch/FMFeatureExtract` (one node per anode, f16-bit sidecar, halo tiling, doctest) with `wct-fm-features.jsonnet` + `run_fm_evt.sh`; C++ vs oracle on the manifest: coordinate sets identical on 69 plane-sets, max Δ 9.8e-6 CPU and 5.1e-5 GPU (TF32 question closed, no knob), 3–4 s and 1.2 GB per anode-event; the Phase-0 cross-view probe on 94 478 labelled sub-blobs: **NO-GO** on the pre-registered +0.05 AP margin (charge-only 0.987, FM+charge 0.990, zero-shot tri-view disagreement AUC 0.51 = no cross-plane signal), with the ceiling caveat and the re-posed probe for the model side |

Layout of `wcfm/`: `wcfm_params.jsonnet` (campaign constants), `wct-sim-iso-track-nf-sp.jsonnet` +
`gen_iso_tracks.py` + `events/` (W2), `img.jsonnet` + `wct-img-all.jsonnet` (W3/W4), `clus.jsonnet` +
`wct-clustering.jsonnet` (W3), runners `run_{sim,img,clus}_evt.sh` + `_runlib.sh`, `scripts/`
(`iso_baseline.py`, `scan_time_offset.sh`, `w3_gate.sh`), `abtest_events.txt` (gate manifest),
`run_img_evt.sh -C` (doc 03 sub-blob knob), `wct-fm-features.jsonnet` + `run_fm_evt.sh` + `scripts/{fm_oracle,fm_scale_check,fm_parity,crossview_probe}.py` (doc 04 FM stage),
`work/` (outputs, not committed).

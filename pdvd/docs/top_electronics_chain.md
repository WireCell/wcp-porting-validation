# PDVD top-CRP (anodes 4–7) NF+SP chain

This doc summarises what is different for PDVD top (anodes 4–7) vs.
bottom (0–3).  For full per-step detail see
[nf.md](nf.md), [sp.md](sp.md), and
[nf_comparison_pdhd_pdvd.md](nf_comparison_pdhd_pdvd.md).

---

## NF stage

The pipeline for each anode runs inside a single `OmnibusNoiseFilter` pnode:

| Step | Algorithm | Top behaviour | Bottom behaviour |
|------|-----------|---------------|-----------------|
| 0 | Resampler | **Skipped** — top clock is already 500 ns | Applied (512→500 ns, data mode only) |
| 1 | `PDVDOneChannelNoise` | Same | Same |
| 2 | `PDVDCoherentNoiseSub` | Same algorithm; uses `chndb-resp-top.jsonnet` kernels | Same algorithm; uses `chndb-resp-bot.jsonnet` kernels |
| 3 | `PDVDShieldCouplingSub` | **Top only, U-plane strips only** | Not applied |

Config source: `toolkit/cfg/pgrapher/experiment/protodunevd/nf.jsonnet`

### Step 3 — Shield Coupling Removal (top anodes only)

**What it is.** The top CRP has an exposed shield plane whose capacitive
coupling induces correlated noise on the U-plane readout strips.
`PDVDShieldCouplingSub` removes this: it scales each strip's waveform by
its strip length, computes a tick-by-tick median across the group, then
subtracts and rescales back.  The idea is credited to Lardon.

**C++ class**: `WireCell::SigProc::PDVD::ShieldCouplingSub`
**Factory tag**: `PDVDShieldCouplingSub`
**Source**: `toolkit/sigproc/src/ProtoduneVD.cxx:1357`
**Header**: `toolkit/sigproc/inc/WireCellSigProc/ProtoduneVD.h:133`

**Placement**: `multigroup_chanfilters` pass inside `OmnibusNoiseFilter`,
which means it runs **inside the NF stage, before SP**, after the
per-channel (Step 1) and coherent-group (Step 2) passes.

**Gating — two layers:**

1. **jsonnet** (`nf.jsonnet:77-82`):

   ```jsonnet
   // only apply to top
   +if anode.data.ident > 3 then[
     {
       channelgroups: chndbobj.data.top_u_groups,
       filters: [wc.tn(shieldcoupling_grouped)],
     },
   ]else []
   ```

   The filter node is only added to the `OmnibusNoiseFilter` graph for
   top anodes (ident > 3); bottom anodes never instantiate it.

2. **channelgroups**: the filter is fed `chndbobj.data.top_u_groups`
   (defined in `chndb-base.jsonnet:412-417`), which restricts it to
   top-CRP U-plane strips.

Neither PDHD nor PDVD bottom has this filter.
For the full algorithm walk-through and parameter table see
[nf.md §Step 3](nf.md#step-3--pdvdshieldcouplingsub-top-anodes-only)
and the PDHD/PDVD comparison at
[nf_comparison_pdhd_pdvd.md §9](nf_comparison_pdhd_pdvd.md#9-step-3--shield-coupling-pdvd-only).

---

## SP stage

The signal-processing graph topology is identical to bottom.  The key
differences are in the electronics-response preset and L1SP mode.

**Electronics response**: top uses `JsonElecResponse` (peak ≈ 7.2 mV/fC,
postgain 1.36), vs. the SBND/PDHD-style analytic response used on bottom
(7.8 mV/fC, postgain 1.0).  This feeds into both the Wiener/Gauss
deconvolution kernels and the L1SP smearing model.

Config: `toolkit/cfg/pgrapher/experiment/protodunevd/params.jsonnet`
(`elecs[1]`); track-response table in
[nf_plot/README.md](../nf_plot/README.md#track-response--sim-overlay).

---

## L1SP on top — process mode; bottom-tuned trigger-gate overrides applied

`L1SPFilterPD` has three operational modes
(see [`toolkit/sigproc/docs/l1sp/L1SPFilterPD.md`](../../../../toolkit/sigproc/docs/l1sp/L1SPFilterPD.md)):

| Mode | Tagger fires? | LASSO runs? | gauss/wiener modified? | NPZ output? |
|------|--------------|-------------|------------------------|-------------|
| `process` | yes | yes | **yes** | optional (`-w`) |
| `dump` | yes | no | **no** | yes (`-c`) |
| `''` (off) | no | no | no | no |

Process mode now applies to **all anodes (0–7)**.  The former
`sp.jsonnet` auto-downgrade (`process → dump` for ident ≥ 4) has been
removed.  `local _eff_mode = l1sp_pd_mode;` — top and bottom are treated
identically.

Effective modes per `run_nf_sp_evt.sh` flag:

| Flag | Effective mode (all anodes 0–7) |
|------|---------------------------------|
| *(none — default)* | process |
| `-w wf_dir` | process + ROI waveform dump |
| `-c calib_dir` | dump |
| `-x` | off |

**Trigger-gate overrides** — the bottom-CRP tuned threshold set is now
applied uniformly to all anodes (`sp.jsonnet`):

| Override | Value | C++ default |
|----------|-------|-------------|
| `l1_len_long_mod` | 180 | 100 |
| `l1_len_fill_shape` | 90 | 50 |
| `l1_fill_shape_fill_thr` | 0.30 | 0.38 |
| `l1_fill_shape_fwhm_thr` | 0.25 | 0.30 |
| `l1_pdvd_track_veto_enable` | true | false |

These were tuned against bottom-CRP hand-scans and are used as the
starting point for top.  Top-CRP threshold validation against a dedicated
hand-scan is pending.  Use `-c` to produce calibration NPZs and
`sp_plot/extract_l1sp_clusters.py` to audit triggered ROIs.

To produce calibration NPZs on a top anode:

```bash
./run_nf_sp_evt.sh -a 7 -c /tmp/calib 039324 0
# NPZs land under /tmp/calib/039324_0/apa7_*.npz
python sp_plot/extract_l1sp_clusters.py --calib-dir /tmp/calib/039324_0 --anode 7 --run 39324 --event 0
```

**Regression test**: `sigproc/test/check_pdvd_anode7_nf_sp.bats` locks the
current top-CRP NF+SP+L1SP output (run 039324 evt 0, anode 7) in a
bit-exact reference fixture.  See
[L1SPFilterPD.md §Test coverage](../../../../sigproc/docs/l1sp/L1SPFilterPD.md#test-coverage)
for details.

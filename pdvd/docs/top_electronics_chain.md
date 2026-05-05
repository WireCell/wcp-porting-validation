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

## L1SP on top — tagger/calibration only; LASSO writeback gated off

`L1SPFilterPD` has three operational modes
(see [`toolkit/sigproc/docs/l1sp/L1SPFilterPD.md`](../../../../toolkit/sigproc/docs/l1sp/L1SPFilterPD.md)):

| Mode | Tagger fires? | LASSO runs? | gauss/wiener modified? | NPZ output? |
|------|--------------|-------------|------------------------|-------------|
| `process` | yes | yes | **yes** | optional (`-w`) |
| `dump` | yes | no | **no** | yes (`-c`) |
| `''` (off) | no | no | no | no |

For top anodes, **`process` is silently downgraded to `dump`** by
`sp.jsonnet:144-149`:

```jsonnet
// PDVD top electronics not yet validated for the L1SP fit; cap 'process'
// to 'dump' on top anodes (ident >= 4) so callers can request process
// mode globally without accidentally enabling LASSO writeback on top.
local _eff_mode = if anode.data.ident >= 4 && l1sp_pd_mode == 'process'
                  then 'dump'
                  else l1sp_pd_mode;
```

Effective modes per `run_nf_sp_evt.sh` flag:

| Flag | Effective mode on top | Effective mode on bottom |
|------|-----------------------|--------------------------|
| *(none — default)* | dump (tagger only) | process |
| `-w wf_dir` | dump | process + ROI waveform dump |
| `-c calib_dir` | dump | dump |
| `-x` | off | off |

**Calibration is available on top.** In dump mode the tagger still fires
and writes per-ROI asymmetry NPZs for offline analysis.  The C++ does
not load the kernel file in dump mode (`init_resp()` is guarded on
`!m_dump_mode`), so the workflow works even though top-CRP kernels exist
in `wire-cell-data/` for future enablement.

To produce calibration NPZs on a top anode:

```bash
./run_nf_sp_evt.sh 039324 0 -a 4 -c work/calib
# NPZs land under work/calib/039324_0/apa4_*.npz
```

**Why is process mode disabled?**  Top-CRP shaping (`JsonElecResponse`,
peak ≈ 7.2 mV/fC) has not yet been validated against the LASSO basis
kernels and the smearing model.  Running LASSO writeback without that
validation risks introducing a systematic bias into the gauss/wiener
output.  The cap will be lifted by removing/narrowing `sp.jsonnet:144-149`
once the top-CRP response is validated.

# PDVD electronics gain and the noise simulation

This note covers how the front-end electronics gain relates to the PDVD
electronics-**noise** simulation:

1. What electronics gain does the bottom-drift simulation use?
2. If a different gain setting is used, how does the noise simulation change?
3. The top-drift electronics response carries an extra ~1.3 `postgain` factor —
   is it decoupled from the noise simulation?
4. A gain bug that this uncovered, and how it is now fixed.

Companion to [`pdvd_noise_simulation.md`](pdvd_noise_simulation.md) (how the
noise simulation is built) and
[`noise_spectrum_comparison.md`](noise_spectrum_comparison.md) (the data-retuned
spectra).

## TL;DR

| Question | Answer |
|---|---|
| Bottom-drift sim gain | **7.8 mV/fC**, 2.2 µs shaping, `postgain` 1.0 (a `ColdElecResponse`) — the run039324 readout gain |
| Does a different gain change the simulated noise? | **Yes — by file selection.** `params.jsonnet` picks the bottom noise-spectra file from `elec.gain` (PDHD convention). `EmpiricalNoiseModel` itself does *no* gain rescaling (`chanstat` empty), so the noise change comes entirely from loading a different file |
| Is the top `postgain` 1.36 decoupled from the noise? | **Yes, completely.** `postgain` scales the signal-shaping electronics response only; the noise path never sees it |

## Signal path vs. noise path

The PDVD `splusn` ("signal plus noise") pipeline, per anode, is
(`cfg/pgrapher/experiment/protodunevd/sim.jsonnet:101`):

```
depos --> depos2traces --> Reframer --> AddNoise --> Digitizer
           \___ SIGNAL ___/             \_ NOISE _/
```

- **`depos2traces`** (`DepoTransform`) convolves the deposits with the field
  response **and the electronics response**, which carries the gain, shaping
  and `postgain` — so the **signal** is scaled by the gain here.
- **`AddNoise`** injects an independent noise waveform per channel, drawn by
  `EmpiricalNoiseModel` from the noise-spectra file. This stage receives **no
  electronics response, gain or `postgain`**.
- **`Digitizer`** converts voltage to ADC by the fullscale span (1.4 V bottom /
  2.0 V top, 14-bit) — a fixed mapping, no gain.

The electronics gain reaches only the signal stage: `tools.elec_resps`
(`cfg/pgrapher/common/tools.jsonnet:101-118`) carries `gain`/`shaping`/`postgain`
and is used **only** as a `short_response` in the field-response convolution
(`tools.jsonnet:152-154`). `make_noise_model` (`sim.jsonnet:61-74`) and
`add_noise` (`sim.jsonnet:78-89`) reference no electronics response.

## 1. Bottom-drift electronics gain

`cfg/pgrapher/experiment/protodunevd/params.jsonnet` (`elecs[0]`, the nominal
`elec`): a **`ColdElecResponse` at 7.8 mV/fC, 2.2 µs shaping, `postgain` 1.0**.
This is the gain run039324 was read out at — and the gain the data-retuned
noise spectra correspond to.

## 2. How the noise simulation changes with a different gain

The noise injected by `AddNoise` is whatever `EmpiricalNoiseModel` returns from
the **noise-spectra file**. Two facts decide how a gain change propagates:

- **Inside the model: no gain rescaling.** `EmpiricalNoiseModel` *can* rescale
  a spectrum, `amp *= ch_gain/db_gain`, where `db_gain` is the file's `gain`
  field and `ch_gain` comes from a `ChanStat` (`EmpiricalNoiseModel.cxx:366-377`).
  But it fires only when a `ChanStat` is configured; PDVD sets `chanstat: ""`
  (`sim.jsonnet:66`), so `ch_gain == db_gain` and **no rescaling happens**. The
  configured `elec.gain` does not enter the noise path at all.

- **In the config: the file is selected by gain.** `params.jsonnet` picks the
  bottom noise file from `elec.gain` with the `pdvd_bottom_noise` helper — a
  strict selector over the four cold-electronics gain settings:

  ```jsonnet
  local pdvd_bottom_noise(gain) =
      local g = gain / (wc.mV / wc.fC);
      if std.abs(g - 7.8) < 0.05 then "pdvd-bottom-noise-spectra-7d8mVfC-v1.json.bz2"
      else if std.abs(g - 14.0) < 0.05 then "pdvd-bottom-noise-spectra-14mVfC-v1.json.bz2"
      else error "...no spectra file for elec.gain = <g> mV/fC...";
  ```

So **a gain change alters the simulated noise only by loading a different
file**: at 7.8 mV/fC the 7.8 file, at 14 mV/fC the 14 file (whose amplitudes
are ~14/7.8× larger — each file is the noise *measured at its own gain*).
Within one gain regime the noise is fixed by that file and is independent of
`elec.gain`.

The selector is **strict**: the cold electronics has four valid gain settings
(4.7 / 7.8 / 14 / 25 mV/fC), spectra files exist only for 7.8 and 14, and any
other value — a setting with no file (4.7, 25) or a value that is not a valid
setting — makes the configuration **abort with an explicit error** rather than
silently fall back to a wrong-gain file. PDHD uses the identical `pdhd_noise`
selector. (This replaced an earlier binary `if elec.gain > 8 mV/fC` threshold
that always returned a filename and would silently mis-bin a third gain.)

One residual caveat: the selector trusts `elec.gain` to be the true readout
gain — it cannot verify the configured gain against the data.

The selector lives in `params.files.noises`, which **every PDVD simulation
reads** — the standalone noise-only sim (`pdvd_sim/wct-sim-noise-only.jsonnet`),
the standard sim (`cfg/pgrapher/experiment/protodunevd/sim.jsonnet`), and the
LArSoft with-track chain (`wcls-sim-drift-simchannel-splusn.jsonnet`) all take
the noise file from `params.files.noises`; no config carries a hard-coded noise
file. So the gain-based selection — and its abort-on-unsupported-gain — applies
uniformly across stand-alone, with-track and standard configurations.

## 3. The gain bug this uncovered — and the fix

Before this work the bottom config hard-wired a single file,
`pdvd-bottom-noise-spectra-v1`, whose `gain` field is `2.2430470818e-12` —
**bit-identical to the PDHD 14 mV/fC spectra file** (`14mVfC` gain field), and
exactly 1.7949× the PDHD 7.8 mV/fC value. So that file is a **14 mV/fC** noise
spectrum. It was being used, unscaled (`chanstat` empty → no rescale), in a
**7.8 mV/fC** simulation — over-stating the bottom noise by 14/7.8 = 1.79×,
which matches the ~1.65× bottom over-prediction measured in
[`noise_spectrum_comparison.md`](noise_spectrum_comparison.md).

The fix, following the PDHD per-gain-file convention:

- `pdvd-bottom-noise-spectra-v1` → **`pdvd-bottom-noise-spectra-14mVfC-v1`**
  (it *is* the 14 mV/fC file; `gain` field unchanged, correct for 14 mV/fC).
- the data-retuned 7.8 mV/fC spectra → **`pdvd-bottom-noise-spectra-7d8mVfC-v1`**,
  with the `gain` field corrected to the 7.8 mV/fC value `1.2496976598600002e-12`
  (it had inadvertently inherited the 14 mV/fC value).
- `params.jsonnet` selects between them by `elec.gain` (Section 2). At the
  nominal 7.8 mV/fC it loads the data-retuned file.

The spectra-file `gain` field is still inert in the running simulation
(`chanstat` empty), but it now correctly labels each file, and a future
`ChanStat`-based rescaling would no longer be misled.

`wire-cell-data/` now holds only the current PDVD noise-spectra files —
`pdvd-bottom-noise-spectra-7d8mVfC-v1`, `pdvd-bottom-noise-spectra-14mVfC-v1`
and `pdvd-top-noise-spectra-v3`; the superseded top files (`v2`, `v1.3`,
`v1d3`) were removed to avoid confusion over which file is current.

## 4. The top `postgain` ~1.3 factor is decoupled from the noise

The top-drift electronics response (`params.jsonnet`, `elecs[1]`) is a
`JsonElecResponse` (`dunevd-coldbox-elecresp-top-psnorm_400.json.bz2`) with
`postgain: 1.36`. `postgain` is a parameter of the **electronics-response
component** — it multiplies the response waveform
(`JsonElecResponse.cxx:79,91`, `Waveform::scale(m_wave, postgain)`). That
component is consumed only by the signal convolution; the noise path
(`EmpiricalNoiseModel` + `AddNoise`) never references it.

**Confirmed: the top `postgain` 1.36 is completely decoupled from the noise
simulation.** It scales the simulated top *signal* only; the simulated top
*noise* is set entirely by `pdvd-top-noise-spectra-v3` and the top digitizer
fullscale (2.0 V).

## `chanstat` — PDHD and PDVD

Both detectors leave it **empty**. Their `make_noise_model` functions
(`sim.jsonnet:44/61`) take an optional channel-status DB `csdb` and set
`chanstat: if csdb==null then "" else wc.tn(csdb)`; both are called as
`make_noise_model(anode)` with no `csdb`, so `chanstat: ""`. Consequently
neither detector uses the in-model `ch_gain/db_gain` rescaling — **both handle
the front-end gain by selecting a per-gain spectra file**, not by rescaling a
single file inside the model.

## Code references

| What | Where |
|------|-------|
| `splusn` pipeline (signal then noise) | `cfg/pgrapher/experiment/protodunevd/sim.jsonnet:101` |
| `EmpiricalNoiseModel` / `AddNoise` config (no gain) | `sim.jsonnet:61-74`, `:78-89` |
| `chanstat` set from optional `csdb` | `sim.jsonnet:66` (PDVD); `cfg/.../pdhd/sim.jsonnet:50` |
| bottom noise file selected by `elec.gain` | `cfg/pgrapher/experiment/protodunevd/params.jsonnet` (`files.noises`) |
| electronics gain / `postgain` / `shaping` | `params.jsonnet` (`elecs`) |
| `elec_resps` (gain) used as a signal `short_response` | `cfg/pgrapher/common/tools.jsonnet:101-118,152-154` |
| `EmpiricalNoiseModel` gain rescale (needs `ChanStat`) | `gen/src/EmpiricalNoiseModel.cxx:366-377` |
| `postgain` scales the response waveform | `gen/src/JsonElecResponse.cxx:79,91`; `ColdElecResponse.cxx:35` |

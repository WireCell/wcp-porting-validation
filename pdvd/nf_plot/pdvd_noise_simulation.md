# PDVD electronics noise simulation

How electronics noise is generated in the ProtoDUNE Vertical Drift (PDVD)
wire-cell simulation, for both the **top** and **bottom** drift volumes — the
components involved, the generation algorithm, and the input files.

For whether this simulation matches data, see
[`noise_spectrum_comparison.md`](noise_spectrum_comparison.md): the noise-spectra
files were re-derived from data (run039324) and the simulation now reproduces the
data noise spectrum to ~3 %.  The earlier
[`noise_rms_comparison.md`](noise_rms_comparison.md) study found the original
spectra mis-tuned and motivated that re-derivation.

## Where noise enters the simulation

The PDVD `splusn` ("signal plus noise") pipeline, per anode, is
(`cfg/pgrapher/experiment/protodunevd/sim.jsonnet`):

```
depos → DepoTransform (drift + field response) → Reframer → AddNoise → Digitizer
```

Noise is added in the **voltage domain**, after the field+electronics response
shaping and before digitisation. It is therefore independent of the track
signal and can be studied on its own — that is what
`pdvd_sim/wct-sim-noise-only.jsonnet` does (`SilentNoise → Reframer → AddNoise
→ Digitizer`, no depos).

## Components

| Component | Source | Role |
|-----------|--------|------|
| `EmpiricalNoiseModel` | `gen/src/EmpiricalNoiseModel.cxx` | provides the mean noise amplitude spectrum for each channel |
| `AddNoise` (= `IncoherentAddNoise`) | `gen/src/AddNoise.cxx` | draws a fluctuated noise waveform per channel and adds it to the trace |
| `Digitizer` | `gen/src/Digitizer.cxx` | converts the voltage waveform to ADC counts |

## Noise-generation algorithm

The noise model is a *sampled-spectrum* model (see `gen/docs/noise.org`):

1. `EmpiricalNoiseModel` returns, for a channel, a real-valued **mean
   amplitude spectrum** μ(f).
2. `AddNoise` converts the mean to the distribution **mode**
   σ(f) = √(2/π)·μ(f) (code: `sigmas[ind] = spec[ind]*sqrt(2/π)`).
3. Each frequency bin is *fluctuated*: a complex amplitude is drawn with real
   and imaginary parts each ∼ 𝒩(0, σ) — equivalently amplitude ∼ Rayleigh(σ),
   phase ∼ Uniform(0, 2π). The spectrum is made Hermitian-symmetric.
4. An inverse DFT produces the time-domain noise waveform, which is added to
   the channel's trace.

Each channel is drawn **independently** → this is *incoherent* noise. The
`replacement_percentage` parameter (0.02) recycles the random buffer for speed,
which introduces only a tiny residual coherency between nearby channels.

### How `EmpiricalNoiseModel` builds a per-channel spectrum

The model loads a `spectra_file` — a JSON array of *spectral objects*, each
describing the sub-sampled mean spectrum for one wire length of one plane:

| field | meaning |
|-------|---------|
| `plane` | wire-plane index (0 = U, 1 = V, 2 = W) |
| `gain` | preamplifier gain (voltage per charge) |
| `shaping` | preamplifier shaping time (ns) |
| `wirelen` | wire length the spectrum was measured for |
| `const` | constant white-noise component, in voltage |
| `freqs`, `amps` | sub-sampled mean spectrum: frequency (1/time) and amplitude (voltage) |
| `nsamples`, `period` | sampling of the stored spectrum |

For each channel the model: (1) interpolates between the two nearest
`wirelen` entries of the channel's plane (both `amps` and `const`); (2)
time-interpolates that sub-sampled spectrum to the model's configured
`nsamples`/`period`. An optional `ChanStat` gain/shaping correction is
available but **not used** for PDVD (`chanstat` is empty).

## Input files — top vs bottom

PDVD has two drift volumes with different cold electronics, so two separate
noise-spectra files are used. The choice is hard-wired on the anode ident
(`sim.jsonnet`: `params.files.noises[if anode.data.ident < 4 then 0 else 1]`).
The **bottom** file is in turn selected by the front-end gain — the PDHD
convention — so `wire-cell-data/` holds one bottom file per gain setting:

| | bottom drift | top drift |
|---|---|---|
| anodes | 0, 1, 2, 3 | 4, 5, 6, 7 |
| spectra file (`wire-cell-data/`) | `pdvd-bottom-noise-spectra-7d8mVfC-v1.json.bz2` (7.8 mV/fC, the run039324 readout gain) or `pdvd-bottom-noise-spectra-14mVfC-v1.json.bz2` (14 mV/fC) | `pdvd-top-noise-spectra-v3.json.bz2` |
| entries | 15 (U 7, V 7, W 1) for 7d8mVfC | 15 (U 7, V 7, W 1) |
| stored `nsamples` / `period` | 6400 / 500 ns | 6250 / 500 ns |
| frequency bins | 512 | 512 |
| induction (U, V) `wirelen` | 7 strip-length bins, ~150–1720 mm | 7 strip-length bins, ~150–1720 mm |
| collection (W) `wirelen` | 1 bin, 1679 mm | 1 bin, 1679 mm |
| `const` (all entries) | 0 (white-noise floor folded into `amps`) | 0 |
| `shaping` / `gain` | 2200 ns / 7.8 (or 14) mV/fC metadata | 2200 ns / inert metadata |

The `7d8mVfC` file is **data-retuned from run039324** — see
[`noise_spectrum_comparison.md`](noise_spectrum_comparison.md) — and is the
default; with the data-derived spectra the top-volume noise is the *larger* of
the two. The `14mVfC` file is the inherited earlier file (its `gain` field
identifies it as a 14 mV/fC spectrum; not independently data-validated). See
[`electronics_gain_and_noise.md`](electronics_gain_and_noise.md) for how the
gain selects the file. The model resamples both files to the simulation's
`nsamples`/`period` (`params.daq.nticks` = 10000, `params.daq.tick` = 500 ns).

### Electronics and ADC differences

Beyond the noise file, the two volumes differ in electronics and ADC
configuration (`cfg/pgrapher/experiment/protodunevd/params.jsonnet`):

| | bottom drift | top drift |
|---|---|---|
| electronics response | `RC` cold elec, 7.8 mV/fC, 2.2 µs shaping, postgain 1.0 | `JsonElecResponse` `dunevd-coldbox-elecresp-top-psnorm_400.json.bz2`, postgain 1.36 |
| field response | `protodunevd_FR_imbalance3p_260501.json.bz2` | `protodunevd_FR_imbalance3p_260501.json.bz2` |
| ADC baselines (U,V,W) | 1003.4, 1003.4, 507.7 mV | 1.0, 1.0, 1.0 V |
| ADC fullscale | [0.2, 1.6] V | [0.0, 2.0] V |
| ADC resolution | 14 bit | 14 bit |
| → ADC LSB | 1.4 V / 16384 ≈ 85.4 µV/count | 2.0 V / 16384 ≈ 122.1 µV/count |

The different LSB means a given noise voltage produces a different ADC-count
RMS in the two volumes — relevant when comparing noise RMS in ADC units.

## What is NOT simulated: coherent noise

The PDVD simulation adds noise with `AddNoise` / `IncoherentAddNoise` **only**.
There is **no `CoherentAddNoise`** node and **no `GroupNoiseModel`** in
`sim.jsonnet`. Correlated (coherent) noise across channel groups — a real
feature of the detector — is therefore **absent from PDVD Monte Carlo**.

(In data the coherent component is removed during noise filtering by
`PDVDCoherentNoiseSub`; the simulation simply never creates it.)

## Configuration and how to run

| What | Where |
|------|-------|
| noise model / adder / `splusn` pipeline | `cfg/pgrapher/experiment/protodunevd/sim.jsonnet` |
| `files.noises[]`, `elecs[]`, `adc` | `cfg/pgrapher/experiment/protodunevd/params.jsonnet` |
| simulation `nticks` (10000) | `cfg/pgrapher/experiment/protodunevd/simparams.jsonnet` |
| noise-spectra input files | `wire-cell-data/pdvd-{bottom,top}-noise-spectra-*.json.bz2` |
| noise-only run config + runner | `pdvd_sim/wct-sim-noise-only.jsonnet`, `pdvd_sim/run_sim_noise.sh` |

To produce a pure-noise frame for every anode:

```
cd pdvd_sim
./run_sim_noise.sh           # all 8 anodes -> work/noise/all/pdvd-noise-sim-anode<N>.tar.bz2
```

## References

- `gen/docs/noise.org` — WCT noise model (authoritative).
- `docs/components/AddNoise.md`, `IncoherentAddNoise.md`, `CoherentAddNoise.md`.
- [`noise_rms_comparison.md`](noise_rms_comparison.md) — data vs. simulation noise-RMS validation.

# PDVD electronics noise simulation

How electronics noise is generated in the ProtoDUNE Vertical Drift (PDVD)
wire-cell simulation, for both the **top** and **bottom** drift volumes — the
components involved, the generation algorithm, and the input files.

For whether this simulation actually matches data, see
[`noise_rms_comparison.md`](noise_rms_comparison.md) (short answer: it does not).

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
(`sim.jsonnet`: `params.files.noises[if anode.data.ident < 4 then 0 else 1]`):

| | bottom drift | top drift |
|---|---|---|
| anodes | 0, 1, 2, 3 | 4, 5, 6, 7 |
| spectra file (`wire-cell-data/`) | `pdvd-bottom-noise-spectra-v1.json.bz2` | `pdvd-top-noise-spectra-v2.json.bz2` |
| entries | 11 | 11 |
| stored `nsamples` / `period` | 6000 / 500 ns | 3000 / 500 ns |
| sub-sampled frequency bins | 201 | 81 |
| U-plane entries (`wirelen`) | 6: 52, 52, 918, 918, 1720.5, 1720.5 | 6: 52, 52, 918, 918, 1720.5, 1720.5 |
| V-plane entries (`wirelen`) | 3: 52, 918, 1720.5 | 3: 52, 918, 1720.5 |
| W-plane entries (`wirelen`) | 2: 1620.5, 1820.5 | 2: 1620.5, 1820.5 |
| `shaping` (all entries) | 2200 ns | 2200 ns |
| example U/52 `gain` | 2.243e-12 | 1.154e-12 |
| example U/52 `const` | 6.76e-9 V | 2.57e-9 V |
| example U/52 `amps[0]` | 9.40e-8 V | 3.37e-8 V |

The stored top-volume amplitudes are markedly smaller than the bottom ones.
The model resamples both files to the simulation's `nsamples`/`period`
(`params.daq.nticks` = 10000, `params.daq.tick` = 500 ns).

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

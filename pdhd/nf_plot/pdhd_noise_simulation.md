# PDHD electronics noise simulation

How electronics noise is generated in the ProtoDUNE Horizontal Drift (PDHD)
wire-cell simulation — the components involved, the generation algorithm, and
the input files.

For whether this simulation matches data, see
[`noise_rms_comparison.md`](noise_rms_comparison.md) (short answer: against the
post-NF data it is broadly consistent, within ~10% — the large pre-NF
discrepancy was coherent noise).

## Where noise enters the simulation

The PDHD `splusn` ("signal plus noise") pipeline, per APA, is
(`cfg/pgrapher/experiment/pdhd/sim.jsonnet`):

```
depos → DepoTransform (drift + field response) → Reframer → AddNoise → Digitizer
```

Noise is added in the **voltage domain**, after the field+electronics response
shaping and before digitisation. It is independent of the track signal and can
be studied on its own — that is what `pdhd_sim/wct-sim-noise-only.jsonnet` does
(`SilentNoise → Reframer → AddNoise → Digitizer`, no depos).

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
   σ(f) = √(2/π)·μ(f).
3. Each frequency bin is *fluctuated*: a complex amplitude is drawn with real
   and imaginary parts each ∼ 𝒩(0, σ) — equivalently amplitude ∼ Rayleigh(σ),
   phase ∼ Uniform(0, 2π). The spectrum is made Hermitian-symmetric.
4. An inverse DFT produces the time-domain noise waveform, which is added to
   the channel's trace.

Each channel is drawn **independently** → this is *incoherent* noise. The
`replacement_percentage` parameter (0.02) recycles the random buffer for speed,
introducing only a tiny residual coherency between nearby channels.

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
`wirelen` entries of the channel's plane; (2) time-interpolates that
sub-sampled spectrum to the model's configured `nsamples`/`period`. An optional
`ChanStat` gain/shaping correction is available but **not used** for PDHD
(`chanstat` is empty).

## Input file — the noise spectra

PDHD has 4 APAs with **identical** cold electronics, so a **single** noise
spectra file is used for all of them. Which file is chosen depends on the
front-end amplifier gain (`cfg/pgrapher/experiment/pdhd/params.jsonnet`):

```jsonnet
noise: if $.elec.gain > 8*wc.mV/wc.fC
       then "protodunehd-noise-spectra-14mVfC-v1.json.bz2"
       else "protodunehd-noise-spectra-7d8mVfC-v1.json.bz2",
```

The gain is supplied at run time via `-V elecGain=<mV/fC>`. **Run 027409 was
taken at 14 mV/fC** (`input_data/META.json`), so the 14 mV/fC file is used for
the data/sim comparison.

| | 14 mV/fC file | 7.8 mV/fC file |
|---|---|---|
| name (`wire-cell-data/`) | `protodunehd-noise-spectra-14mVfC-v1.json.bz2` | `protodunehd-noise-spectra-7d8mVfC-v1.json.bz2` |
| entries | 8 | 8 |
| stored `nsamples` / `period` | 6000 / 512 ns | 6000 / 512 ns |
| sub-sampled frequency bins | 109 | 109 |
| U-plane entries (`wirelen`) | 4: 7350, 7350, 7450, 7450 | 4: 7350, 7350, 7450, 7450 |
| V-plane entries (`wirelen`) | 2: 7350, 7450 | 2: 7350, 7450 |
| W-plane entries (`wirelen`) | 2: 5950, 6050 | 2: 5950, 6050 |
| `shaping` (all entries) | 2200 ns | 2200 ns |
| example U/7350 `gain` | 2.243e-12 | 1.250e-12 |
| example U/7350 `const` | 1.357e-8 V | 9.44e-9 V |
| example U/7350 `amps[0]` | 1.185e-7 V | 6.40e-8 V |

The same file feeds all four APAs, so the simulated noise is **APA-uniform**.
The strong APA-to-APA spread in *raw* data is coherent noise, removed by noise
filtering; the post-NF data is APA-uniform too, so on the post-NF footing this
is not a defect (see `noise_rms_comparison.md`).

## Electronics and ADC configuration

`cfg/pgrapher/experiment/pdhd/params.jsonnet`, uniform across all 4 APAs:

| | value |
|---|---|
| electronics response | cold elec, gain `-V elecGain` mV/fC (14 for run027409), 2.2 µs shaping |
| field response | `np04hd-garfield-6paths-mcmc-bestfit.json.bz2` (APA0) / `dune-garfield-1d565.json.bz2` (APA1-3) |
| ADC baselines (U,V,W) | 1003.4, 1003.4, 507.7 mV |
| ADC fullscale | [0.2, 1.6] V |
| ADC resolution | 14 bit |
| → ADC LSB | 1.4 V / 16384 ≈ 85.4 µV/count |
| daq tick | 512 ns (data) / 500 ns (simulation, `simparams.jsonnet`) |
| daq nticks | 6000 |

## What is NOT simulated: coherent noise

The PDHD simulation adds noise with `AddNoise` / `IncoherentAddNoise` **only**.
There is **no `CoherentAddNoise`** node and **no `GroupNoiseModel`** in
`sim.jsonnet`. Correlated (coherent) noise across channel groups — a real
feature of the detector — is **absent from PDHD Monte Carlo**. (In data the
coherent component is removed during noise filtering; the simulation simply
never creates it.)

## Configuration and how to run

| What | Where |
|------|-------|
| noise model / adder / `splusn` pipeline | `cfg/pgrapher/experiment/pdhd/sim.jsonnet` |
| `files.noise`, `elecs`, `adc` | `cfg/pgrapher/experiment/pdhd/params.jsonnet` |
| simulation `tick` (500 ns) / `nticks` (6000) | `cfg/pgrapher/experiment/pdhd/simparams.jsonnet` |
| noise-spectra input files | `wire-cell-data/protodunehd-noise-spectra-{14mVfC,7d8mVfC}-v1.json.bz2` |
| noise-only run config + runner | `pdhd_sim/wct-sim-noise-only.jsonnet`, `pdhd_sim/run_sim_noise.sh` |

To produce a pure-noise frame for every APA:

```
cd pdhd_sim
./run_sim_noise.sh           # all 4 APAs -> work/noise/all/pdhd-noise-sim-anode<N>.tar.bz2
./run_sim_noise.sh -g 7.8    # low-gain noise spectra instead of the 14 mV/fC default
```

## References

- `gen/docs/noise.org` — WCT noise model (authoritative).
- `docs/components/AddNoise.md`, `IncoherentAddNoise.md`, `CoherentAddNoise.md`.
- [`noise_rms_comparison.md`](noise_rms_comparison.md) — data vs. simulation noise-RMS validation.
- `../../pdvd/nf_plot/pdvd_noise_simulation.md` — the same write-up for ProtoDUNE Vertical Drift.

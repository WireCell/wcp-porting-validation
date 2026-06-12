# PDVD Top U-plane Shield Coupling Noise Filter

This document describes the `PDVDShieldCouplingSub` noise filter —
its physical motivation, algorithm, configuration, and known limitations.
For the overall NF workflow see [nf.md](nf.md).

## Physical motivation

In the PDVD top-CRP (top-electronics) TDE design the U-plane strips share
capacitive coupling to a shield grid. The coupling amplitude scales with
strip length — longer strips couple more strongly. This produces a common
per-tick additive artifact that is **pure negative polarity** on the U
waveforms (ionisation signal on U is bipolar, so the sign asymmetry is
distinctive). The algorithm estimates this artifact by building a per-tick
median across all U channels within a group (after normalizing by strip
length), then subtracts it.

This filter applies **only** to:
- **Top anodes**: `anode.data.ident > 3` (anodes 4–7)
- **U plane only**: via the `top_u_groups` channel grouping
- **Bottom anodes** (0–3): untouched — different electronics, no shield

Origin: the approach is adapted from the lardon prototype at
`lardon/noise_filter.py#L181`  
(cited in `sigproc/src/ProtoduneVD.cxx:7`).

---

## Position in the NF chain

File: `cfg/pgrapher/experiment/protodunevd/nf.jsonnet:54–93`

The `OmnibusNoiseFilter` applies three groups of filters in order:

```
channel_filters:         single      (PDVDOneChannelNoise)   — all channels
multigroup_chanfilters:
  entry 0: grouped       (PDVDCoherentNoiseSub)              — all channels
  entry 1: shieldcoupling_grouped  (PDVDShieldCouplingSub)   — top U only
```

Shield coupling subtraction therefore runs **after** per-channel and
coherent-group filtering. The coherent-group filter (`PDVDCoherentNoiseSub`)
has already removed FEMB-level correlated noise before the shield step runs.

The gate is explicit in `nf.jsonnet:77`:

```jsonnet
+if anode.data.ident > 3 then[
  {
    channelgroups: chndbobj.data.top_u_groups,
    filters: [wc.tn(shieldcoupling_grouped)],
  },
]else []
```

### Channel groups

`top_u_groups` is defined in `chndb-base.jsonnet:425–427`:

```jsonnet
top_u_groups:
  [std.range(n*3072, n*3072+475)    for n in std.range(2,3)]
  +[std.range(n*3072+476, n*3072+951) for n in std.range(2,3)]
```

This produces **four groups** (one per top-CRP face), each containing
476 contiguous U-plane channels:

| Group | Channel range | CRP |
|-------|--------------|-----|
| 0 | 6144 – 6619 | CRP2, face A |
| 1 | 6620 – 7095 | CRP2, face B |
| 2 | 9216 – 9691 | CRP3, face A |
| 3 | 9692 – 10167 | CRP3, face B |

Total: 4 × 476 = **1904 U-plane channels** covered.

Each group is processed independently — the filter sees no cross-group
information.

---

## Algorithm

Source: `sigproc/src/ProtoduneVD.cxx:1190–1456`  
Header: `sigproc/inc/WireCellSigProc/ProtoduneVD.h:133–161`

The filter is **purely time-domain** — no FFT is performed despite a
stored `m_dft` member.

### Step 1 — Strip-length normalization

`ShieldCouplingSub::apply()` at `:1369–1390`

```cpp
signal.at(ibin) /= strip_length;   // "like calib/capa in Lardon"
```

Each channel's waveform is divided by its strip length (loaded from
`PDVD_strip_length.json.bz2`). This makes the coupling amplitude
comparable across channels before computing the group median. If a
channel's length is not found in the map, an error is printed and
`strip_length = 1` is used (no-op).

### Step 2 — Signal protection (positive-only masking)

`Signal_mask_top_u()` at `:1190–1236`, called per channel.

```cpp
const double sigFactor = 4.0;
const int    padBins   = 70;

float rmsVal       = PDVD::CalcRMSWithFlags(sig);
float sigThreshold = sigFactor * rmsVal;

// flag positive excursions above threshold
if (ADCval > sigThreshold && ADCval < 16384.0)
    signalRegions[i] = true;

// expand ±padBins and mark with +200000 sentinel
sig.at(j) += 200000.0;
```

Key design choices:
- **Positive only**: only samples above `4 × RMS` are protected. The
  rationale (`:1192–1193`): "noise is pure negative". Positive excursions
  are true ionisation signal; negative ones that survive coherent NF are
  treated as coupling noise.
- **Sentinel `+200000`**: flagged samples are not zeroed — they are
  shifted up by 200000 ADC so that downstream cuts (` ≤ 100000`) exclude
  them without destroying the underlying amplitude. `RemoveFilterFlags`
  restores them later.
- **`ADCval < 16384`**: samples already carrying another flag marker
  (e.g. sticky codes) are not re-flagged.
- **`±70`-tick pad**: covers signal tails; symmetric.

### Step 3 — Per-channel RMS estimate

`CalcMedian_shieldCoupling_u()` at `:1247–1271`

```cpp
for (const auto& value : signal) {
    if (value <= 100000)           // exclude flagged (>100000) samples
        filtered_signal.push_back(value);
}
auto [mean, rms] = Waveform::mean_rms(filtered_signal);
max_rms += rms;
count_max_rms++;
```

A per-channel RMS is computed from all non-flagged samples. The group
`max_rms` is the **arithmetic mean** of all per-channel RMSes — i.e. a
single scalar that represents the typical noise level across the group.

### Step 4 — Per-tick group median

`CalcMedian_shieldCoupling_u()` at `:1273–1300`

For each tick `ibin`:

```cpp
for (int ich = 0; ich < nchannel; ich++) {
    const float cont = content[ich * nbins + ibin];
    if (cont < 5 * max_rms && fabs(cont) > 0.001)
        temp.push_back(cont);
}
medians[ibin] = (temp.size() > 0) ? Waveform::median_binned(temp) : 0.0;
```

Two filters on the pool:
- `cont < 5 * max_rms` — outlier rejection (flagged samples appear at
  `+200000`, well above this cut; true large signals that survived
  masking are also excluded here if they are sufficiently large).
- `|cont| > 0.001` — zero-tolerance (avoids including literally-zero
  channels, e.g. dead channels that have already been zeroed).

`Waveform::median_binned()` is a histogram-based median (not sort-based).
If no samples survive both cuts, the median is set to zero for that tick.

### Step 5 — Subtraction and scale restore

`ShieldCouplingSub::apply()` at `:1427–1453`

```cpp
PDVD::RemoveFilterFlags(signal);       // undo +200000 markers
float scaling = 1;
for (int i = 0; i != nbin; i++) {
    if (fabs(signal.at(i)) > 0.001)
        signal.at(i) -= medians.at(i) * scaling;
}
// then re-multiply by strip_length
signal.at(ibin) *= strip_length;
```

Notes:
- `scaling = 1` is hardcoded — no per-channel amplitude weighting beyond
  strip-length normalization.
- The zero-tolerance cut `|signal| > 0.001` means exactly-zero samples
  are not modified (consistent with dead/masked channels).
- The final multiply restores physical units.

### CMM output

`apply()` returns an empty `ChannelMaskMap` (`:1360, 1455`). The filter
does **not** contribute to the bad-channel mask — downstream nodes have
no record of which ticks were heavily perturbed.

---

## Configuration

### Jsonnet block (`nf.jsonnet:42–52`)

```jsonnet
local shieldcoupling_grouped = {
    type: 'PDVDShieldCouplingSub',
    data: {
        anode:         wc.tn(anode),
        noisedb:       wc.tn(chndbobj),
        strip_length:  params.files.strip_length,
        rms_threshold: 0.0,
    },
};
```

`params.files.strip_length` is `"PDVD_strip_length.json.bz2"` (deployed
in wirecell-data; `params.jsonnet:167`).

### C++ configure knobs (`ProtoduneVD.cxx:1319–1343`)

| Knob | Type | Used? | Purpose |
|------|------|-------|---------|
| `anode` | string TN | Yes | AnodePlane lookup |
| `strip_length` | path | Yes | Per-channel strip-length map |
| `noisedb` | string TN | Stored, not used | `m_noisedb` loaded but never queried in `apply()` |
| `dft` | string TN | Stored, not used | `m_dft` loaded but no FFT path exists |
| `rms_threshold` | float | Stored, not used | `m_rms_threshold` loaded but not referenced in `apply()` |

### Commented-out / disabled knobs

```cpp
// m_capa_weight = get<bool>(cfg, "capa_weight", ...);
// m_calibrated  = get<bool>(cfg, "calibrated",  ...);
// m_group_size  = get<int> (cfg, "group_size",  ...);
// m_min_channels= get<int> (cfg, "min_channels",...);
```

These indicate an intended richer design that was not completed.

### Hardcoded magic numbers

| Name | Value | Location | Meaning |
|------|-------|----------|---------|
| `sigFactor` | 4.0 | `:1195` | Signal-protection threshold multiplier |
| `padBins` | 70 | `:1196` | Symmetric padding around flagged samples (ticks) |
| flag sentinel | +200000 | `:1229` | In-place marker for masked samples |
| flag cutoff | 16384 | `:1207, 1228` | ADC value above which samples are already externally flagged |
| drop threshold | 100000 | `:1252` | Samples above this are excluded from RMS estimate |
| outlier factor | 5.0 | `:1278` | Per-tick pool outlier rejection: `5 × max_rms` |
| zero tolerance | 0.001 | `:1278, 1435` | Samples at or below this are treated as zero / skip |
| scaling | 1.0 | `:1433` | Median subtraction weight (no per-channel weight) |

None of these are exposed in the Jsonnet config.

---

## What is NOT in the algorithm

- **No frequency-domain processing** — despite storing `m_dft`, the
  filter is purely time-domain.
- **No per-channel coupling weight** — only strip-length normalization;
  no capacitance or calibrated coupling-coefficient weighting.
- **No iteration** — one pass per `apply()` call; no second-pass with a
  cleaner mask.
- **No CMM output** — downstream nodes receive no information about
  which ticks/channels were most heavily corrected.
- **No cross-group information** — the four shield groups are processed
  entirely independently.
- **No protection of negative-polarity signal** — only positive
  excursions are flagged; negative-going ionisation signal (induction
  plane tail) leaks into the median pool on busy events.

---

## Room for improvement

### A. Robust statistics in the group RMS estimate

**What**: Replace `mean_rms` (`:1256`) with a MAD or trimmed-mean RMS
over the non-flagged samples.

**Why**: `mean_rms` is biased upward by any unmasked non-Gaussian tail
— which inflates `max_rms`, widens the `5 × max_rms` outlier gate, and
admits genuine signal into the median pool. A robust estimator would
tighten the gate without requiring a lower threshold coefficient.

**Verify**: Check median-pool size per tick before/after on busy events
(run 039324). A correct implementation admits fewer outliers at high
occupancy.

---

### B. Bipolar signal protection

**What**: In addition to the positive-excursion mask, flag large
*negative*-going samples (below `-sigFactor × RMS`), especially when
the coherent-noise subtraction residual is substantial.

**Why**: The current rationale — "noise is pure negative" — holds for
the coupling noise in isolation. But after coherent NF the residual
induction signal on U is bipolar. On events with many tracks the
negative lobes of U responses leak into the median pool and bias the
estimated coupling artifact toward over-correction.

**Verify**: Compute the average of `median[tick]` on a busy event vs.
a quiet event. On a quiet event it should be close to 0; on a busy
event a large downward bias indicates signal contamination.

---

### C. Make the magic numbers configurable

**What**: Promote `sigFactor`, `padBins`, `outlier_factor` (5.0), and
`zero_tolerance` (0.001) into JSON-configurable fields.

**Why**: None of these are physically derived constants — they were
hand-tuned. Making them configurable:
- Enables the filter-tune viewer A/B pattern (already used for HF/LF
  filters) to compare values on real data without recompiling.
- Makes the tuning history auditable via Jsonnet diffs.

The disabled knobs `capa_weight`, `calibrated`, `group_size`,
`min_channels` are already named in `default_configuration()` —
completing them requires wiring the values into `apply()`.

---

### D. Calibrated coupling-amplitude weighting

**What**: Introduce a per-channel coupling coefficient (beyond bare
strip length) as the `capa_weight` / `calibrated` design was intended
to do.

**Why**: If shield-to-strip coupling is non-uniform along a strip (e.g.
due to varying strip-to-shield gap, geometry deviations, or FEMB
position), normalizing by strip length alone leaves a residual
amplitude spread across the group. A per-channel coefficient fit from
noise-only data would reduce this spread and lower the median variance.

**Verify**: Examine the spread of `content[ich][ibin]` values across
channels at quiet ticks (no signal). After strip-length normalization,
this should be approximately flat; systematic channel-to-channel
scatter indicates uncalibrated coupling variation.

---

### E. Iterative two-pass subtraction

**What**: Run the algorithm twice: first pass with the raw waveform →
produces a coarse subtracted result → use that result to re-derive a
tighter signal mask → second pass with fewer signal-contaminated
samples in the pool.

**Why**: The first-pass RMS estimate is computed on the raw waveform
where the coupling artifact is still present — this can over-inflate
`max_rms` and under-protect signal samples near the `4 × RMS`
boundary. A second pass on the corrected waveform would see cleaner
noise statistics.

---

### F. Guard on minimum survivors per tick

**What**: Implement the `min_channels` knob: if fewer than
`min_channels` samples survive the outlier cut at a given tick, set
`median[tick] = 0` (i.e. skip subtraction for that tick) and optionally
add the group channels to the CMM for that tick.

**Why**: Currently when the pool is empty the median defaults to 0 —
no correction applied — but the caller has no way to distinguish
"median was reliably estimated to be zero" from "median could not be
estimated". On pathological ticks (saturated group, dense tracks) a
median based on 2–3 survivors can do more harm than good.

---

### G. Optional CMM contribution for unreliable ticks

**What**: When a tick's median pool is too small (< `min_channels`), emit
those tick ranges on all channels of the group into the returned
`ChannelMaskMap`.

**Why**: Downstream SP nodes (ROI finding, imaging) cannot currently
distinguish ticks where the shield correction failed from normal ticks.
A CMM contribution would allow them to apply a wider ROI exclusion or
lower-weight those samples.

---

### H. Frequency-domain hybrid step

**What**: After time-domain median subtraction, apply a per-group
narrow-band filter to suppress coupling residuals at known harmonic
frequencies (e.g. power-line fundamentals visible in the noise
spectrum). The stored `m_dft` member and `noisedb` handle provide a
ready hook.

**Why**: Capacitive coupling noise typically has known spectral lines.
The time-domain median step removes the average waveform shape, but
per-tick variance and spectral lines at non-integer-tick periods survive.
A frequency-domain notch (similar to the existing `freqmasks` in
`PDVDOneChannelNoise`) applied to the coupling residual — without
touching the signal band — could further reduce the post-filter power.

---

### I. Performance: eliminate per-tick heap allocation

**What**: Replace the `temp.push_back()` inner loop
(`CalcMedian_shieldCoupling_u:1275`) with a pre-allocated buffer
reused across ticks.

**Why**: Currently a `realseq_t temp` vector is constructed and
destroyed for every tick × every invocation. With 476 channels and
6000+ ticks this is ~3 million small heap allocs per call. A scratch
buffer allocated once per `apply()` call and cleared per tick would
eliminate this overhead. This matches the existing note in
`sigproc/docs/examination/08-efficiency-summary.md:47` (EFF-DET-5/6).

---

## Validation approach

1. **Before/after channel dump**: Run NF with the existing NF framework
   diagnostic hooks to dump raw waveforms before and after
   `PDVDShieldCouplingSub` on top-anode U channels. Inspect with
   Magnify or the filter-tune viewer.

2. **Quiet vs busy event comparison**: On run 039324 (already processed),
   compare the tick-by-tick median waveform (`medians[]`) on a quiet
   event vs. one with many tracks. If medians are significantly
   non-zero on quiet events only where coupling noise is expected, the
   estimate is clean. If they are non-zero in track regions, signal
   contamination is present.

3. **Residual coherent power**: After NF, compute the mean power
   spectrum across the four `top_u_groups` separately. The coupling
   noise bands should be suppressed relative to the pre-shield
   spectrum. Any surviving coherent structure points to an incomplete
   subtraction.

4. **SP output quality**: Compare `wiener<N>` on top-anode U (N=4–7)
   before and after any algorithm change using the viewer at
   `http://localhost:5007/filter_tune_viewer`.

---

## Source file index

| File | Purpose |
|------|---------|
| `toolkit/sigproc/src/ProtoduneVD.cxx:1190–1456` | `Signal_mask_top_u`, `CalcMedian_shieldCoupling_u`, `ShieldCouplingSub` |
| `toolkit/sigproc/inc/WireCellSigProc/ProtoduneVD.h:133–161` | `ShieldCouplingSub` class declaration |
| `cfg/pgrapher/experiment/protodunevd/nf.jsonnet:42–93` | Jsonnet wiring; top-only gate; chain order |
| `cfg/pgrapher/experiment/protodunevd/chndb-base.jsonnet:425–428` | `top_u_groups` channel-group definition |
| `cfg/pgrapher/experiment/protodunevd/params.jsonnet:167` | `strip_length` file path |
| `PDVD_strip_length.json.bz2` | Per-channel strip-length calibration (wirecell-data) |
| `toolkit/sigproc/docs/examination/05-detector-specific.md:60,177–187,373–379` | Brief prior notes (capability matrix, EFF-DET-5/6, 5-step summary) |

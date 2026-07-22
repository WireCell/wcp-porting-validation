# PDVD light reco: raw / decon / flash-time check — a +0.75 µs self-trigger offset

**Status:** investigation, **no code and no config is changed by this document.**
The figures are diagnostics; the recommended correction at the end is
owner-gated because it would move reconstructed light times unconditionally.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/pdvd
python3 pd_plot/light_timing_check.py          # ~3 min, writes docs/pics/pdvd_light_timing_*.png
# optional: skip the 15-event raw-ROOT scan (figure 5)
PDVD_LT_MULTI=0 python3 pd_plot/light_timing_check.py
```

Inputs, all pre-existing products of our own chain (no borrowed files):

| what | path |
|---|---|
| raw + decon dense frames (3 branches) | `work/039252_light298567_keep/light-frames-{cathode,membrane,pmt}.tar.bz2` |
| flashes / ophits | `work/039252_light298567_keep/opflash_pdvd-wct.tar.gz` |
| raw ROOT (figure 5, runs 039252/039253/039349) | `input_data_light/np02vd_raw_run*_rawwf.root` |

The frames were produced by `wct-light-frames.jsonnet` (see doc
`06_pdvd-light-chain.md`); the flashes by `run_light_evt.sh`.

## The one time axis

Everything below is plotted in **microseconds relative to the chain tick
origin `t0`** of `PDVDOpWaveformSource` (`root/src/PDVDOpWaveformSource.cxx:105`):

```
t0 = min over ALL records of   llround(timestamp_us * 62.5)
                               - (nsamp <= 1024 ? trig_sample : 0)      [16 ns ticks]
```

The `min` deliberately ignores the `opch` selection so the three population
branches (cathode / membrane / PMT) share one origin.

* raw and decon frames: `tickinfo = [time, tick_ns, tbin0]`; frame column `i`
  is absolute tick `tbin0 + i`.
* flash time: **opflash column 0, in NANOSECONDS** relative to the same `t0`
  (`OpFlashFinder` works in WCT ns). A µs/ns slip here inverts every
  conclusion in this document — see `09_pdvd-qlmatching.md`.

## 1. Raw, decon, flash time — the overview

![overview](pics/pdvd_light_timing_overview.png)

Run 39252 event 298567, cathode XA `opch 1010` (full-stream, 468800 samples
≈ 7.5 ms). Top: raw ADC minus pedestal. Second: the WI-deconvolved waveform
of the same channel. Third: all 414 reconstructed flashes as a total-PE stem
plot. Bottom: a 150 µs zoom of the 16 cathode channels summed, with every
flash in the window marked.

Every visible light pulse has a flash on it, and the flash density (~414 per
7.5 ms, ~55/ms) matches the pulse density — no wholesale duplication and no
wholesale loss.

## 2. Zoom: one flash across the three populations

![zoom](pics/pdvd_light_timing_zoom.png)

Flash #222, 4113.898 µs, 2774 PE — deliberately **not** the brightest flash of
the event, because the brightest ones rail the 14-bit ADC and the decon then
shows saturation-recovery undershoot that has nothing to do with timing.
Each column is one population; the raw panel and the decon panel show the
**same** channel. Samples outside the record are exactly 0 in the dense frame
and are masked, so the record boundary is not drawn as a signal edge.

The dashed line is the flash time. The cathode decon peaks **−90 ns** from it;
the membrane XA peaks **+678 ns** and the PMT **+598 ns** *later*. That gap is
the subject of §4.

## 3. The flash time itself is correct (in the light domain)

![residual](pics/pdvd_light_timing_residual.png)

For each of the 102 flashes with total PE > 300, the peak of the summed
cathode decon within ±2 µs, minus the flash time:

| quantity | value |
|---|---|
| median | **−40 ns** |
| IQR | −117 … −3 ns |
| linear fit vs flash time | slope **−0.0042 ns/µs**, intercept −74.5 ns |

The slope means the offset grows by 31 ns across the whole 7.5 ms window —
i.e. it is **flat**. A units error (ns vs µs) or a clock-rate error would show
as a slope of order the offset itself; a wrong `t0` would show as a large
constant. Neither is present. The −40 ns is the known effect that the flash
time is the accumulator-bin time of the brightest hits while the decon peak
sits a fraction of a tick later; it reproduces the −96 ns quoted from the
python-vs-C++ validation in `06_pdvd-light-chain.md`.

**So: the light-domain flash time is sound to a few tens of ns, and this
figure is what rules out the "obvious" candidates.**

## 4. What is actually off: self-trigger records are +0.75 µs late

![population](pics/pdvd_light_timing_population.png)

Same flashes, but now the **raw** 20 %-of-maximum rise time of the loudest
channel of each population (raw, not decon, so no template or filter can be
blamed), relative to the flash time:

| population | readout | median rise − flash | IQR |
|---|---|---|---|
| cathode XA | full stream, 468800 samples | **−45 ns** | −130 … −5 |
| membrane XA | self-trigger snippet, 1024 samples | **+678 ns** | +613 … +737 |
| PMT | self-trigger snippet, 1024 samples | **+744 ns** | +666 … +786 |

The distribution is cleanly bimodal and, in the right-hand panel, **flat in
flash PE over two decades** (300 PE → 40 000 PE). That is the discriminating
test:

* **not late light / scattering** — the Ar triplet and reflected light would
  give a brightness- and topology-dependent, broad, asymmetric lag, not a
  ±70 ns core independent of PE;
* **not threshold walk** — a self-trigger firing later on dim pulses would
  give a strong PE dependence, which is absent;
* **not the decon** — it is measured on the raw samples.

It splits by **readout mode**, not by device: membrane XA (SiPM, slow) and PMT
(fast, completely different front end) sit together at +0.7 µs, and the only
thing they share is that both are read out as **self-trigger snippets** while
the cathode is a **full stream**.

### It is not event- or run-specific

![multievent](pics/pdvd_light_timing_multievent.png)

Figure 5 repeats the measurement with **no WCT product at all** — straight
from the raw ROOT, replicating `t0` in python, taking the 20 %-rise of every
stream pulse in the summed cathode channels and of every self-trigger record
at the tick where `PDVDOpWaveformSource` places it:

| population | run 039252 | run 039253 | run 039349 |
|---|---|---|---|
| membrane | +752 ns (n=2143) | +768 ns (n=1740) | +752 ns (n=1109) |
| PMT | +816 ns (n=2150) | +816 ns (n=1751) | +816 ns (n=1698) |

IQR ≈ ±40 ns everywhere. **The offset is a constant of the data**:
≈ **+0.75 µs (47 ticks) for membrane XA, +0.82 µs (51 ticks) for PMT**, stable
across 3 runs and 15 events.

### Reconciliation with doc 05, which measured the opposite

`05_pdvd-flash-dt.md` reports `pmt − cathode` median **+0.000 µs** with a 68 %
half-width of 32 ns, and concludes "per-population systematic offsets are
≤ 64 ns". That is a direct contradiction of +816 ns and has to be settled
rather than stepped around.

Doc 05 uses the same hit-time convention as the chain (its §"common clock",
line 23: snippets → `timestamp + (decon spike − 64) × 16 ns`, stream →
`timestamp + sample × 16 ns`), so the discrepancy is **not** a convention
difference. It is the **estimator**: doc 05 measures *nearest-neighbour* Δt of
each hit to the closest cathode hit ≥ 3 PE. That statistic is blind to a
common offset whenever the reference population has many hits per physical
pulse — and it does. On the chain's own ophits for event 298567:

| quantity | value |
|---|---|
| cathode ophits ≥ 3 PE in the event | 6261 |
| median gap between consecutive cathode hits | **272 ns** |
| fraction of gaps < 800 ns | **66 %** |
| doc-05 estimator reproduced (nearest-neighbour PMT − cathode) | **−64 ns**, 68 % half-width 266 ns |
| leading-edge estimator, same event (§4) | **+744 ns** |

A PMT hit sitting 816 ns after the *leading edge* of a cathode pulse simply
finds a different cathode hit — from the tail of the same pulse — a few tens
of ns away, and the estimator returns ≈ 0. Reproducing doc 05's method on our
ophits gives −64 ns, i.e. doc 05's number is correct **for what it measures**;
it just cannot see this offset. Measuring the 20 %-of-maximum **rise** of the
pulse instead removes the ambiguity, because a pulse has exactly one leading
edge.

Two consistency checks that survive: doc 05's snippet↔snippet pair
(`membrane_bot − pmt`, +64 ns) agrees with the 64 ns membrane↔PMT difference
here — the common offset cancels between two self-trigger populations, which
is precisely why only the snippet↔**stream** pairs are affected. And doc 05's
own follow-up note ("`TRIG_SAMPLE = 64` … shifts all snippet hits by a common
constant and only matters for the absolute light↔charge alignment") is exactly
the class of effect this document measures.

**Doc 05's conclusions are not withdrawn** — `bin_width = 1000 ns` and the
single all-PD flash stand — but its statement "no per-type time-offset
correction is needed" should be read as scoped to snippet↔snippet pairs.

### `trig_sample = 64`: internal structure is self-consistent, absolute placement is not testable here

The natural suspect is `trig_sample`, which `PDVDOpWaveformSource` subtracts
from a snippet's timestamp and which `06_pdvd-light-chain.md` flagged as
"assumed 64, confirm at chain assembly". Inside the 1024-sample snippets of
event 298567 (pulses above 100 ADC):

| population | in-snippet 20 %-rise sample | peak sample |
|---|---|---|
| membrane | median **64** (p5 56) | 89 |
| PMT | median **74** (p5 70) | 77 |

The pulse begins essentially *at* sample 64, exactly as a 64-sample
pre-trigger predicts. If the true pre-trigger were 64 + 47 ≈ 111 (the value
that would absorb the offset), the rise would sit at sample 111, not 64.

This is a statement about the snippet's **internal** structure only: it shows
where the pulse sits *within the recorded window*, and is self-consistent with
`trig_sample = 64`. It says nothing about where that window sits on the
absolute axis — the **absolute placement is set entirely by the timestamp**,
and cannot be validated from the snippet alone. The decomposition is therefore:
in-snippet structure consistent, absolute placement ≈ 0.75–0.82 µs late
relative to the full-stream sample counter. That residual lives in the
timestamp path of the two DAPHNE readout modes, not in our reconstruction —
and for the same reason (no internal absolute reference) the sign cannot be
resolved here.

**Caveat, stated plainly:** this measurement is *relative*. It shows the two
readout modes disagree by 0.75 µs; it cannot say which one is right in
absolute terms, because the light data contains no third reference. There is
no cathode channel read out in both modes (checked: event 298567 has 16
records of 468800 samples, all `opch 10xx`, and 2121 records of 1024 samples,
all `20xx`/`30xx`), so the readout-mode and the PD-population hypotheses
cannot be separated *within* the light stream. Resolving the sign needs
either the DAPHNE firmware pre-trigger/timestamp definition or an external
reference (the charge trigger).

## 5. What this does and does not explain

**Does not explain a light↔charge offset.** A raw/decon/flash figure lives
entirely inside the light domain. The known-open item there is different: the
legacy scalar `offset_us` stays **0** for PDVD (inert, `flash.jsonnet:203`),
and the real per-crate offsets are stamped as `offset_bot_us`/`offset_top_us`
by `run_light_evt.sh` and consumed as `trigger_offset` /
`trigger_offset_top` in `qlmatching.jsonnet`. For this event they are
−2500.24 µs (BDE/bottom) and −2486.75 µs (TDE/top). If the reported symptom
is "flash times are ~2.5 ms off the charge", that is this bookkeeping, not
§4 — check that the consumer actually reads the per-crate keys and not
`offset_us`.

**Impact of §4 on current output — modest, and mostly not on the flash time.**
The cathode carries ~89 % of the PE (`06_pdvd-light-chain.md`), so the flash
time is cathode-anchored and inherits the cathode clock; §3 measures it as
correct. What the offset costs is the *booking* of wall-XA and PMT hits: with
`bin_width = 1 µs`, a 0.75–0.82 µs shift moves them to the edge of, and
sometimes past, the bin holding the cathode hits of the same physical flash.
In event 298567 the gross symptom is small — only 5 of 414 flashes are
wall/PMT-dominated (cathode < 5 % of PE), carrying 0.3 % of the event's PE —
but this is the plausible mechanism behind the "the light **is** in the
waveform, the flash PE lost it" cases catalogued for wall XAs in the
wall-XA usability study (`qlmatch/`, `pd_plot`-side probe
`docs/qlmatch/scripts/wall_xa_wf_probe.py`).

## 6. Recommended next step (owner decision required)

The correction is a single number per self-trigger population — subtract
≈ 47 ticks (membrane) / 51 ticks (PMT) from the snippet start tick, or
equivalently configure a per-population effective `trig_sample` of ≈ 111 / 115.

This **cannot** be applied silently: it moves every membrane/PMT OpHit time
and would change flash composition and PE, so per §1 of `CLAUDE.md` it must
ship as a knob that defaults OFF and leaves the current path byte-identical.
Proposed shape, not implemented here:

* `PDVDOpWaveformSource`: new `snippet_time_offset_ticks` (default `0`,
  absent key ⇒ no-op), applied to the start tick of records with
  `nsamp <= snippet_nsamp`, alongside the existing `trig_sample`.
* `flash.jsonnet`: threaded per branch with the key-suppression idiom so the
  compiled JSON is byte-identical when off.
* Gate: `abtest` hash comparison of `opflash_pdvd-wct.tar.gz` with the knob
  off; knob-on smoke run showing the §4 medians collapse to ≈ 0.

Before implementing, two things should be settled:

1. **Sign.** Confirm against the DAPHNE self-trigger timestamp definition
   (or the charge trigger) whether the snippets are late or the stream is
   early. Correcting the wrong one leaves the light↔charge offset wrong by
   0.75 µs even though the light↔light offset closes.
2. **Whether it matters.** 0.75 µs of drift is ~1.2 mm of drift distance at
   1.586 mm/µs — negligible for Q/L T0. The case for fixing it is hit
   *booking* (§5), not T0 accuracy.

## Summary

| claim | evidence |
|---|---|
| flash times sit on the light pulses | fig 1, 414 flashes |
| flash time vs cathode decon peak: −40 ns, no drift | fig 3, slope −0.004 ns/µs |
| no ns/µs units error, no wrong `t0` | fig 3 (flat residual) |
| `trig_sample = 64` is internally self-consistent (absolute placement untestable from the snippet) | in-snippet rise at sample 64 |
| membrane/PMT records are **+0.75 / +0.82 µs late** vs the cathode stream | fig 4, fig 5; 3 runs, 15 events, IQR ±40 ns |
| the offset is a DAQ readout-mode property, not late light or the decon | flat in PE over 2 decades; measured on raw samples; splits by readout mode not by device |
| doc 05's "offsets ≤ 64 ns" does not contradict this | its nearest-neighbour estimator is blind to a common offset (66 % of cathode hit gaps < 800 ns); reproduced: −64 ns vs +744 ns leading-edge |
| a light↔charge offset cannot be seen in these figures | §5 |

## Files

* `pd_plot/light_timing_check.py` — produces all five figures and prints
  every number quoted above.
* `docs/pics/pdvd_light_timing_overview.png`
* `docs/pics/pdvd_light_timing_zoom.png`
* `docs/pics/pdvd_light_timing_residual.png`
* `docs/pics/pdvd_light_timing_population.png`
* `docs/pics/pdvd_light_timing_multievent.png`

Related: `06_pdvd-light-chain.md` (chain assembly, python-vs-C++ validation),
`05_pdvd-flash-dt.md` (per-type Δt, bin width), `03_pdvd-spe-template.md`
(kernel peak@10 convention), `09_pdvd-qlmatching.md` (offset consumption).

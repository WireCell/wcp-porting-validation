# PDVD PD mapping investigation: suspected channel↔module swap → mapping verified correct; real culprit = saturation veto

**Status: investigation complete.** The photon-detector channel↔module mapping
is **correct end-to-end** (all documentation sources agree, and the raw
waveforms prove it from data). The measured-vs-predicted light-shape
discrepancy that motivated the suspicion is caused by the **DAPHNE 14-bit ADC
saturation veto** in our light chain: bright cathode-crosser flashes rail the
X-Arapucas nearest the track, the veto removes every hit within ±16.4 µs of a
rail, and those channels enter Q-L matching as **exactly 0 PE** — the measured
pattern is hole-punched at precisely its physical peak. A fix decision (veto
policy / matcher-side masking) is a separate follow-up; see §7.

## Repro

```
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
python3 docs/qlmatch/scripts/pd_mapping_audit.py            # parts 1-3 (documentation + dump-level tests)
python3 docs/qlmatch/scripts/pd_mapping_audit.py --raw      # + parts 4-5 (raw-waveform adjudication, prevalence)
```

Inputs: `work/039252_<i>_pull2c2/`, `work/0392{53,349}_<i>/` calib dumps (the
existing shield-FV/ctoff + pull2c2 records — nothing regenerated);
`input_data_light_rawwf/*.root` (jjo's raw-waveform trees);
`/cvmfs/.../duneprototypes/v10_09_00d00/config_data/PDVD_PDS_Mapping_v04152025.json`;
`photlib/pdvd-photlib-vis-v5-128nm.json`, `photlib/pdvd-photlib-chanmap.json`.

## 1. Symptom

On the ql_scan display (port 5019, tag `candles-pull2c2`), run 039252
evt298567: the cathode-crossing pairs matched to bright flashes show measured
and predicted cathode-XA patterns that disagree **in shape** (not just
normalization):

| flash (PE) | measured bright (jjo labels) | predicted bright | predicted-bright channels measure |
|---|---|---|---|
| gid37 (5466) | C3 2027, C8 1580, C2 1124 | C5 578, C6 255, C7 133 | **exactly 0** |
| gid41 (5679) | C3 1959, C8 1706, C2 836 | C5 478/297, C7 215/578 | **exactly 0** |
| gid57 (6485) | C4 2683, C7 1139, C8 1052 | C2 762–777, C1 373 | **exactly 0** |

Suspicion: a DAPHNE offline-channel ↔ module mapping error (e.g. the
documented open item that jjo's assignment and the official PDS map differ by
the Arapuca y-mirror partner), or a coordinate-system mismatch between the
LArSoft-derived photon library and the toolkit.

## 2. Documentation three-way comparison — all sources agree

Sources compared per offline channel (`pd_mapping_audit.py` part 1):

- **A — jjo's rawwf trees** (`flashopdet/opdet_geo` + the `opchannel`/`opdet`
  columns of `rawdump/raw_waveform`): identical to the toolkit chain's
  `pdvd-opch-map.json` ∘ `pdvd-opdet-geom.json` (the chain was derived from
  these files, so this is consistency, not independent confirmation).
- **B — official `PDVD_PDS_Mapping_v04152025.json`**: geometry-channel
  numbering is v4-era; decoding gc → position **literally against the v4
  GDML** implies every Arapuca sits at the y-mirror partner of jjo's
  assignment. But the v4 GDML Arapuca layout is itself the known y-mirrored
  (wrong) layout (`08_pdvd-photon-model.md` §2), so the literal reading is
  ambiguous by exactly one y-mirror — this is the previously documented open
  item, and it cannot be settled from paperwork.
- **C — colleague's independent mapping**: the case statement names
  (1010/1011 = C1 … 2080/2081 = M8) match jjo's names channel-by-channel, and
  the cathode layout picture (readable in the higher-resolution screenshot;
  top panel = view from the top) places **C1, C4, C2, C3 on the NO-TCO half
  and C5, C8, C6, C7 on the TCO half**, in the exact staircase arrangement of
  jjo's positions. Orientation pinned by the v5 3D panel in the same figure:
  opdets 1/3/13/19 (all y = −417.6 cm) are drawn on the **TCO side ⇒ TCO =
  −y**, and the membrane picture (M1/M2/M5/M6 on No-TCO, M3/M4/M7/M8 on TCO)
  then also matches jjo's walls.

**Verdict: A ≡ C ≡ toolkit; B agrees too once its v4-mirror ambiguity is
resolved the physical way. There is no documented mapping conflict.**

## 3. Coordinate systems (LArSoft photon library vs toolkit) — cleared

Full trace (previous audit + this round, `01_pdvd-photon-models.md` §5,
`sample_ann.py`, `PhotonLibraryModel.cxx`):
ANN inputs are cm in the GDML world frame == WCT frame (cathode x=0); the
10 cm grid `origin_cm=[-350,-340,-10]`, `n=[71,69,33]`, C-order
`[nx,ny,nz,40]` matches the C++ voxel arithmetic axis-by-axis
(`PhotonLibraryModel.cxx:80-90`); v5 as-built GDML positions == raw-data
positions to 0.0 mm; library column i ≡ `m_opdets[i]` is enforced at load
time via `chan_pos_cm` (`QLMatching.cxx:489-503`). **No coordinate
transform or offset discrepancy exists.**

## 4. Why the data *looked* like a mapping problem — and why it is not

Dump-level tests over ~180 strict geometric crossers (bright ≥1500 PE flash
dominating its ±30 µs neighbourhood; both halves' drift ends within 3 cm of
the cathode at the flash time; endpoints coincident within 20 cm; harvested
from 78 events across runs 039252/039253/039349 — `pd_mapping_audit.py`
parts 2–3):

- **Permutation fit vs predictions**: exhaustively scoring all 8! = 40320
  relabelings of the cathode block, the documented mapping ranks **dead
  last**. Taken alone this "confirms" a mapping bug.
- **BUT the adjacency test refutes every permutation**: channel–channel
  correlations of √PE over ~14k flashes (prediction-free and invariant to
  per-channel gain) match the **documented** module distances near the top of
  all 40320 orderings (Spearman ≈ 0.71, rank ~16), while the permutation-fit
  "winner" scores ≈ −0.1 (rank ~29000). No single relabeling passes both
  tests.
- **No coordinate transform survives either**: the measured-light centroid of
  strict crossers is *uncorrelated* with the track's charge centroid in y
  (corr ≈ 0) and *anti*-correlated in z (≈ −0.5) — impossible for any fixed
  relabeling or affine map (predictions track the charge at corr ≈ +0.97, as
  they must).

The paradox — internally position-consistent labels whose per-flash patterns
ignore the track — is resolved by the raw waveforms.

## 5. Raw-waveform adjudication (bypasses our entire light chain)

From jjo's `rawdump/raw_waveform` streams directly
(`pd_mapping_audit.py --raw`, part 4):

- **Ganging is correct**: the two DAPHNE sub-channels of each cathode XA
  (10X0/10X1) correlate at 0.976–0.998 in 1 µs-binned signal, always far above
  any cross-module correlation.
- **Raw adjacency matches the documented layout**: e.g. the {1010,1020,1040}
  (C1,C2,C4, +y) and {1050…1081} (C5–C8, −y) groups, with C3 (the most
  central module) bridging — exactly the documented geometry.
- **At the hand-confirmed flash instants the raw streams agree with the
  prediction, not with the dump.** Flash gid37 (evt298567): raw pulse areas
  peak on **C5 (5.6M ADC·samples), C7 (3.5M), C6 (3.2M)** — the modules
  nearest the track, exactly the prediction's bright channels — while the
  dump records **0.0 PE** on those three. The un-railed channels agree with
  the dump almost channel-by-channel (raw area ≈ dump PE within ~20%):

  | module | raw area (10X0+10X1) | rail samples | dump PE |
  |---|---|---|---|
  | C1 | 353766 | 0 | 317 |
  | C2 | 1059707 | 0 | 1124 |
  | C3 | 2069279 | 0 | 2027 |
  | C4 | 335952 | 0 | 418 |
  | **C5** | **5607450** | **151+176** | **0.0** |
  | **C6** | **3206004** | **3+19** | **0.0** |
  | **C7** | **3525841** | **8+18** | **0.0** |
  | C8 | 1087399 | 0 | 1580 |

  The zeroed channels are **exactly the ones that rail** at the DAPHNE
  14-bit ceiling (ADC = 16383). Flashes gid41 and gid57 behave identically
  (gid41: C5/C6/C7 railed 319/86/24 samples, all dump-zero; gid57: C1/C2/C3/C5
  railed 21/231/119/204, all dump-zero; every un-railed channel's raw area
  tracks its dump PE).

## 6. Root cause

Our PDVD light chain runs with the saturation machinery **on** for all three
populations (`pdvd/wct-light-reco.jsonnet:58-74`): OpDecon
`detect_saturation=true` flags tick ranges at `saturation_adc=16383`
(≥1 sample), padded by `saturation_pad=1024` samples = **±16.4 µs**; OpHitFinder
`veto_saturation=true` then drops every hit overlapping a flagged range
(`OpHitFinder.cxx:373`). Consequence: a channel that rails **anywhere within
±16.4 µs of a flash contributes exactly 0 PE to it**. Bright cathode-crosser
flashes (≥ a few 1000 PE) rail the 1–4 modules nearest the track, so their
measured patterns lose precisely their peak channels.

Downstream effects on Q-L matching:

- The measured/predicted **shape mismatch** on the display (the original
  symptom): predicted peaks sit on measured zeros.
- chi2/KS of the **true** (flash, cluster) pairs explode (e.g. gid37↔top:60
  chi2 = 1867), because zero-with-large-prediction channels dominate; **dim
  flashes never rail**, so their complete patterns can outscore the true
  bright flash — a second, independent mechanism (besides the cathode
  containment gate fixed in `pdvd-cathode-containment-flash-demotion.md`)
  pushing crossers onto dim flashes.
- Normalization: the predicted totals are also globally low by ~2.5–10×;
  that is the separate, known provisional cathode PE scale / QtoL issue, not
  addressed here.

Prevalence (part 5, over the 120 events with both raw waveforms and calib
dumps — runs 039252/039253/039349): **3303 of 7797 (42%) bright (≥1000 PE)
flashes have at least one cathode channel vetoed to zero by saturation**
(~15–50 affected bright flashes and ~50–220 rail intervals per event). All
three hand-scanned evt298567 flashes are affected.

## 7. Fix directions (decision pending — owner's call)

The veto itself was a deliberate data-quality choice (PDHD run-29107 round:
railed pulses deconvolve badly). Options, in increasing order of principle
and effort:

1. **Turn the veto off for the cathode branch** (`veto_saturation=false`):
   railed pulses then reconstruct a clipped, *underestimated* PE instead of
   zero. Cheap (config knob), biased low on the brightest channels, but far
   less wrong than zero for shape metrics.
2. **Shrink `saturation_pad`** (1024 → ~64 samples): unrelated hits within
   the ±16.4 µs window survive; the railed pulse itself is still zeroed, so
   the flash-shape hole remains. Not sufficient alone.
3. **Propagate a per-flash per-channel saturation flag** into the opflash
   tensor and have QLMatching treat flagged channels as *unknown* (excluded
   from chi2/KS/LASSO, like a per-flash mask) rather than as measured-zero.
   The principled fix; needs C++ + tensor-schema work in
   OpHitFinder/OpFlashFinder/FlashTensorToOpticalPCs/QLMatching.
4. (Orthogonal) saturation-aware PE **recovery** (fit the un-railed rising /
   falling edges) — largest effort, restores brightness information.

Any of these changes light-reco output ⇒ **not byte-identical** ⇒ default-OFF
knob + revalidation per the house rules.

### 7.1 Option-1 probe on evt298567 (done, tags `_satoff`/`satoff`)

Knobs added (all default = legacy, compiled configs byte-identical when off —
`cmp` proof done):

- `wct-light-reco.jsonnet` tla `veto_saturation=true`; runner env
  `PDVD_VETO_SATURATION=0` (run_light_evt.sh) disables the veto.
- `run_clus_evt.sh` env `PDVD_LIGHT_SUFFIX` points Q-L matching at a tagged
  light variant dir (`work/<RUN6>_light<EVENTNO><SUFFIX>`).

Repro:

```
PDVD_VETO_SATURATION=0 ./run_light_evt.sh -s _satoff 39252 298567
# imaging symlinked from work/039252_0 into work/039252_0_satoff, then
PDVD_LIGHT_SUFFIX=_satoff ./run_clus_evt.sh -s satoff -calib 039252 0
```

Result on the three hand-scanned flashes (old = pull2c2 record):

| flash | total PE old→new | restored channels (old 0 → new PE) | match kept? | KS |
|---|---|---|---|---|
| gid37 | 5466 → 13924 | C5 2468, C6 3546, C7 2444 | yes (both halves) | 0.19/0.23 |
| gid41 | 5679 → 17327 | C5 2380, C6 6047, C7 3221 | yes | 0.17/0.23 |
| gid57 | 6485 → 25327 | C1 3761, C2 7933, C3 5157, C5 1992 | yes | 0.10/0.25 |

The measured **shapes** now agree with the predictions (the previously
zero/predicted-bright contradiction is gone; the restored PE are clipped
underestimates of the raw areas, as expected for railed pulses). Absolute
chi2 remains large and even grows — that is the separate global
normalization (QtoL / provisional cathode PE scale) issue now acting on
more bright channels, not a shape problem. Visual check: ql_scan on port
5020, tag `satoff-check` (5019/`candles-pull2c2` untouched).

## 8. What was checked and cleared along the way

- jjo trees ≡ toolkit maps ≡ colleague names/pictures (this doc §2).
- LArSoft-library ↔ toolkit coordinate frames identical (§3).
- Sub-channel ganging (10X0/10X1 pairing) correct (§5).
- The official-map v4-literal mirror reading is an artifact of the mirrored
  v4 GDML, not evidence of a data-side swap; with the mapping now confirmed
  from raw data, the `08_pdvd-photon-model.md` §2 open item is **closed**
  (jjo's assignment is the physical one).
- The Ar-blind-channel / Xe-vs-Ar analyses that consumed dump PE are
  unaffected by any mapping issue (there is none), but analyses of **bright**
  flashes should note the saturation holes.

## 9. Follow-ups

- Owner decision on §7; then implement as default-OFF knob(s), A/B-gate the
  legacy path, and revalidate evt298567 (the display's flashes 37/41/57
  should show measured peaks on C5/C6/C7 or C2 once handled).
- Re-examine the QL high-consistency ladder / chi2 tuning after the fix: the
  current values were tuned on hole-punched patterns.
- The per-channel gain spread suggested by the crosser fits (×2–7 on the
  cathode) feeds the known provisional-PE-scale follow-up.

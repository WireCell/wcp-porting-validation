# PDHD pre-flip coherent-noise grouping override (run-027409 etc.)

## Why

The toolkit's `chndb-base.jsonnet` derives the coherent-noise groups directly from
`PD2HDChannelMap_WIBEth_visiblewires_v1.txt` (duneprototypes `c8f43809`), which
carries the **2025-06-30 sign-flip** of the visible-wire ±3 rotation. That is the
correct grouping for frames **decoded with the post-flip channel map**.

Our **run-027409** frames on disk were decoded with the **pre-flip** map. Fingerprint
(APA0, U-plane, first FEMB group):

| | first U group | signature |
|---|---|---|
| pre-flip (our files) | offline `3,4,…,42` (contiguous, no wrap) | what `coh_group_shift=3` reproduces |
| post-flip (toolkit default) | `388,390,392,…` (FEMB-membership, wraps `…799,797,798,0,2,4,6`) | map-derived |

For pre-flip files the **old** grouping (`coh_group_shift=3` programmatic blocks +
`femb-negpulse-groups-shifted_v2.jsonnet`) is the matching one; the new map would
**mis-group U/V channels by 3 at each FEMB boundary** during coherent-noise removal.
Scope is narrow: a 3-channel cyclic misassignment at U/V FEMB edges only — **W and
the wire geometry are untouched**. The failure mode is degraded coherent subtraction
on edge channels, not a global mislabel.

> The convention lives in the **decode**, not the frames: pre- vs post-flip is not
> readable from the raw `.npy` (only the channel labels assigned at decode differ).
> The check is the duneprototypes version/date used at decode time (before
> 2025-06-30 ⇒ pre-flip). So the override is an explicit, per-running-directory
> choice, not auto-detected from data.

## Measured impact (run-027409 evt0, APA0, DNN-ROI chain)

Old (`1b6de3ea`) vs current (HEAD) DNN-ROI U-plane, and the same-binary isolation:

| comparison | U gauss/wiener turnover | what it shows |
|---|---|---|
| current new-groups vs old-groups (same HEAD binary) | **1.99 %** (max\|Δ\|≈4299); raw ADC 99.85 % | the regrouping perturbs NF on ~all U channels; ~2 % of DNN-kept pixels move |
| full old-build vs HEAD-binary+old-config | **0.0000 %** (byte-identical) | all other 1b6de3ea→HEAD changes are exactly U-neutral |

⇒ **100 % of the U-plane DNN change between the two commits is the coherent-group
redefinition** (`fd12265b`). Nothing else (W-tune, roi_plane2layer revert) touches U.

## The mechanism (local, default-off, colleagues bit-identical)

The toolkit cfg is left at the **latest (post-flip)** default. The override lives
entirely in this running directory:

- **`pdhd-coh-groups-preflip.jsonnet`** — reproduces the old grouping exactly
  (the `chndb-base` body from `1b6de3ea`): `groups(n)` = `coh_group_shift=3`
  programmatic U/V/W groups; `negpulse_groups` = `femb-negpulse-groups-shifted_v2`.
- **`wct-nf-sp.jsonnet` / `wct-nf-sp-dnnroi.jsonnet`** — new toggle
  `coh_groups_preflip` (default **false**). When true, the chndb data overrides
  `groups` and `femb_negpulse_groups` with the pre-flip versions:
  `base(...) { dft } + (if coh_groups_preflip then { groups, femb_negpulse_groups } else {})`.
  Default false ⇒ `+ {}` ⇒ **bit-identical** for colleagues.
- **`run_nf_sp_evt.sh` / `run_nf_sp_dnnroi_evt.sh`** — auto-set
  `--tla-code coh_groups_preflip=true` **iff** the local sentinel
  **`pdhd/.coh_preflip`** exists (git-ignored).

## Usage

```sh
# Process run-027409 (pre-flip) with the matching old grouping:
touch pdhd/.coh_preflip            # one time; sticky, git-ignored
./run_nf_sp_dnnroi_evt.sh -a 0 27409 0      # log prints "[coh] .coh_preflip ... PRE-FLIP"
./run_nf_sp_evt.sh 27409 0                  # same for the SP/imaging chain

# When files are re-decoded with a post-flip duneprototypes, revert to latest:
rm pdhd/.coh_preflip
```

Colleagues (no sentinel) always get the toolkit's latest post-flip grouping.

## Verification (2026-06-08)

- **Sentinel ON** ⇒ DNN-ROI U byte-identical to the full `1b6de3ea` old build
  (`_preflip_wtune` vs `_old`: 0.0000 %, max\|Δ\|=0 — W-tune is W-only, U matches).
- **Sentinel OFF** ⇒ DNN-ROI U byte-identical to the unmodified current chain
  (`_cur2` vs `_cur`: 0.0000 %) — colleagues' default unchanged.
- Isolated jsonnet check: `preflip.groups(0)[0] == [3..42]` (pre-flip signature),
  60 groups/anode, 160 negpulse groups; differs from the map-derived default.

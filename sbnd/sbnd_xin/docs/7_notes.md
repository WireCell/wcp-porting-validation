# Working notes

Things that are easy to forget when picking the work back up later.

## LAr parameters in reco (DL / DT / lifetime)

The three LAr parameters synced to `DL=4.0 cm²/s`, `DT=8.8 cm²/s`,
`lifetime=35 ms` (see [2_geometry-and-timing.md](2_geometry-and-timing.md))
are simulation inputs. Their role in the reco chain is much smaller than
in sim:

### `DL` / `DT` — only in `TrackFitting`

The only reco consumer is `clus/src/TrackFitting.cxx`. Used in the
trajectory + dQ/dx fit to set the per-point charge-cloud σ from each
point's drift time:

```
sigma_long = sqrt(2 * DL * drift_time)
sigma_tran = sqrt(2 * DT * drift_time)
```

These enter the covariance terms of the least-squares dQ/dx fit. Call
sites:

- `dQ_dx_multi_fit()` — lines 6085-6086, 6118-6119
- `dQ_dx_fit()` — lines 6444-6445
- `single_fit()` — lines 7156-7157, 7187-7188

Defaults in `clus/inc/WireCellClus/TrackFitting.h:37-38` are
`DL = 6.4 cm²/s`, `DT = 9.8 cm²/s` — those are the historical
trajectory-fit smearing values that previously also appeared in
`sbnd_xin/wct-clustering.jsonnet` and `run_clus_evt.sh`.

**Important — the sim/reco DL/DT are not the same config knob.**
`TrackFitting` does **not** read `params.lar.DL/DT`. It loads its own
JSON via `TaggerCheckNeutrino::load_trackfitting_config()`
(`clus/src/TaggerCheckNeutrino.cxx:493`). So the cfg/`sbnd_xin/` sync
to `4.0 / 8.8` only affects **simulation diffusion**; the
**trajectory-fit smearing model** still uses whatever the TrackFitting
JSON holds. When you get to dQ/dx work later, decide whether the fit
should use the same `(DL, DT)` as the sim and update that JSON too if
so.

**2026-07-25 — the fit's coefficients were set to the PHYSICAL values on owner
instruction; the sim-vs-physical question above is STILL OPEN.**
`sbnd_xin/sbnd_track_fitting.json` now carries `DL = 6.5781 cm²/s`
(`6.5781e-07`) and `DT = 13.1349 cm²/s` (`1.31349e-06`), the SBND transport
coefficients, replacing the 6.2 / 9.8 placeholders. The **simulation** was left
at 4.0 / 8.8 (that mirrors sbndcode's own `wcsimsp_sbnd.fcl`, so changing it is
an SBND production decision, not ours). Consequence: on MC the fit assumes ~15 %
more transverse smearing than the waveforms actually contain. **Not
bit-identical**: fitted `dQ`/`dx` change, so STM/PR numbers in docs 41–46
predate it.

Whether that MC divergence is acceptable is an **owner decision that has not
been taken** — the instruction was to set the coefficients, not to decide this.
The two readings, both defensible:
- **A (what is in the tree now):** the fit models the *real detector*; MC
  divergence is accepted and capped, and MC-based fit validation (docs 44/46)
  inherits a known ~15 % transverse bias.
- **B:** the fit should match whatever produced the waveforms it is fitting —
  physical 6.5781/13.1349 for data, sim 4.0/8.8 for MC. Cheap to implement: the
  runners and `wct-pr-perevt.jsonnet` already carry a `reality=sim|data` TLA, so
  this is a second JSON or a reality-switched pair of values, not new machinery.
Quantified per plane and per drift distance in
`47_stm-bragg-reference-sbnd-retune.md` §6a.

### `lifetime` — not in toolkit reco yet

Grep finds no live uses of `lifetime` / `electron_lifetime` / `tau` /
`attenuation` in `clus/`, `img/`, or the reco jsonnet glue. The only
plumbed-in use is the sim path
(`cfg/pgrapher/common/sim/nodes.jsonnet:28` → drifter).

The one informative reference is a comment at
`clus/src/PRSegmentFunctions.cxx:1196-1198`:

> "For multi-APA detectors (SBND, DUNE) a per-face electron-lifetime
> correction and a per-face recombination model would be needed;
> callers must pre-apply lifetime corrections to dQ before this
> function is called."

So in the current toolkit reco the assumption is that charge has
already been lifetime-corrected before it reaches the trajectory fit.
The physics chain you'd expect — `Q/L matching → trajectory fit →
dQ/dx lifetime correction` — is the right shape; the third step just
isn't ported from the wcp prototype yet.

**When you port the dQ/dx-correction step later**, that's the place
that needs a `params.lar.lifetime` plumb-through (and the SBND per-face
caveat from the comment above).

### Anything else?

- **SP / NF**: no DL/DT/lifetime use (only field response, electronic
  response, noise spectrum).
- **Imaging / blob building**: no DL/DT/lifetime use; only
  `drift_speed` and `tick` for time→x.
- **Q/L (flash) matching**: not yet wired up in the toolkit reco path
  in this tree. In the prototype it carries a lifetime/recombination
  factor; that piece isn't ported here.

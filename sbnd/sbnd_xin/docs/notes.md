# Working notes

Things that are easy to forget when picking the work back up later.

## LAr parameters in reco (DL / DT / lifetime)

The three LAr parameters synced to `DL=4.0 cm²/s`, `DT=8.8 cm²/s`,
`lifetime=35 ms` (see [geometry-and-timing.md](geometry-and-timing.md))
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

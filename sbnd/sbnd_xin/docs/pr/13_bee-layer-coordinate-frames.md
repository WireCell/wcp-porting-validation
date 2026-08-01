# doc pr/13 — why `img-global` does not sit on top of the PR layers

**Status:** investigation only. **No code and no config is changed by this doc** —
nothing to A/B, nothing to revalidate. §6 lists options for a later round; the choice
is the owner's.

**Question (owner, on the pr/12 Bee sets):** *"there seems to be some offsets between
img-global and the shower_track-global … I guess this may be due to the position offset
that I added for the clustering step after QLMatching?"*

**Answer:** yes for the transverse part, and there is a second, larger part. The
accurate statement is not "the PR layers are offset" but **`img-global` is the only
raw-frame layer in the zip**. Every other point layer — `clustering-global`,
`track_fit-global`, `shower_track-global`, `vertices-global` — is emitted in the
post-QLMatching *corrected* frame `(x_t0cor, y_cor, z_cor)`. Two things separate the
two frames, and both are working as designed:

| component | size | applies to |
|---|---|---|
| per-cluster T0 drift correction `x_t0cor − x` | **0 … ±121 cm** in evt 280972; 10 of its 12 clusters move >10 cm, only **31 %** of points move <1 cm | every cluster, x only |
| per-TPC transverse `pos_offset` | Δy `∓0.11`, Δz `±0.67` cm — **1.34 cm relative in z across the cathode** | every point, y and z |

The premise "`clustering-global` is consistent with `img-global`" holds **only for the
in-beam neutrino candidate**, which is what one looks at in these events: in evt 280972
the candidate (img cluster 19) has `Δx = +0.21 cm`, while the cosmics around it are
displaced by −59 … +121 cm. So on the candidate the whole visible mismatch is the
transverse `pos_offset` the owner suspected; on everything else in the frame the drift
correction dominates.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# sections A (frame fingerprint), B (img<->clustering delta), D (t0 through the tarball)
python3 bee_frame_probe.py -q work-mcp1kall-d59k -p work-mcp1kall-cath01 \
        -e 280972 -e 292384 -s ABD

# section C (the decisive track_fit test), pooled over the pr/12 spanned set
python3 bee_frame_probe.py -q work-mcp1kall-d59k -p work-mcp1kall-cath01 \
        --index docs/pr/cath_spanned.index.txt -n 20 -s C
```

Inputs are the pr/12 arms: Q/L `work-mcp1kall-d59k`, PR `work-mcp1kall-cath01`
(toolkit HEAD `3fe65876`). The probe is read-only and touches no `work-*` content.

## 1. Where each layer's coordinates come from

`pos_offset` is a **data-only** calibration — the measured SBND per-TPC transverse
cathode offset, split symmetrically (`cfg/pgrapher/experiment/sbnd/clus.jsonnet:70-71`):

```jsonnet
local pos_offset_a0 = [0, -0.11 * wc.cm, 0.67 * wc.cm];   // TPC0 (East, x<0)
local pos_offset_a1 = [0, 0.11 * wc.cm, -0.67 * wc.cm];   // TPC1 (West, x>=0)
```

It reaches the points through `T0Correction::forward()`
(`clus/src/PCTransforms.cxx:91-102`), which does the drift shift in x **and** adds
`(dy, dz)`; `backward()` (`:105-116`) inverts both. Presence of the key flips the
corrected scope's array names to `{x_t0cor, y_cor, z_cor}` (`:216, :223`), and the
jsonnet keeps lockstep through `common_corr_coords(pos_offset_on)`
(`clus.jsonnet:73-74`).

| Bee layer | dumped where | coordinates | frame |
|---|---|---|---|
| `img-global` | Q/L job, **pre-pipeline** (`MultiAlgBlobClustering.cxx:2407-2416`) | `['x','y','z']` (`clus.jsonnet:457-463`) | **raw** |
| `channel-deadarea-*` | Q/L job, same pre-pipeline point (`:2394-2402`) | `(x, z)` polygons | **raw** |
| `clustering-global` | Q/L job, end of `cm_pipeline` (`:2513-2524`) | `common_corr_coords` (`clus.jsonnet:465-475`) | corrected |
| `track_fit-global` | PR job, after `TaggerCheckNeutrino` | PRGraph `fit.point` (`:927`) | corrected |
| `shower_track-global` | PR job, after `TaggerCheckNeutrino` | `dpc->point3d(ip)` (`:893`) | corrected |
| `vertices-global` | PR job, after `TaggerCheckNeutrino` | `vertex->fit().point` (`:962`, `:1012`) | corrected |

`img` is dumped **before** the clustering pipeline runs, i.e. before
`ClusteringSwitchScope:all` creates the corrected arrays — so at that moment the
corrected coordinates do not exist yet. That, not a choice about display, is why the
layer is raw.

### A trap worth writing down

The three PR sets carry `coords: ['x','y','z']` in the jsonnet with the comment *"PRGraph
fit points are already T0-corrected, hence plain x/y/z"* (`clus.jsonnet:1272-1275`). That
key is **inert**: `fill_bee_points_from_pr_graph` appends the PRGraph's own point objects
and never reads `config.coords` (`MultiAlgBlobClustering.cxx:806-1012`). Editing it changes
nothing. The routing that sends those sets down the PRGraph path is also not
config-driven — it keys on whether the grouping already holds a PR graph
(`:2462-2476`), so *any* visitor-keyed set on the live grouping after
`TaggerCheckNeutrino` takes that path regardless of what it asks for.

## 2. Evidence

### A — frame fingerprint (`-s A`, evt 280972 / 292384)

`shower_track` re-emits points that already exist in the tree, so its coordinates can be
matched **exactly**:

```
shower_track-global   8060 uniq  exact-in-img 0  exact-in-clustering 3652 | NN med to img 0.3786  to clustering 0.0010
track_fit-global      1338 uniq  exact-in-img 0  exact-in-clustering    0 | NN med to img 0.3426  to clustering 0.3750
vertices-global         84 uniq  exact-in-img 0  exact-in-clustering    0 | NN med to img 0.3895  to clustering 0.2581
```

Every `shower_track` point lands on the **corrected** cloud (median nearest-neighbour
distance 0.0010 cm — the residue is 4-decimal rounding, not displacement) and **none**
lands on the raw cloud (0.38 cm away). `track_fit` and `vertices` are newly fitted
points, so exact matching cannot settle them — §2C does.

### B — the two frames compared point by point (`-s B`, evt 280972)

Points matched by their unique `q` value, so the same physical point is compared in both
dumps (46 562 of 49 165 matched):

```
TPC0 x<0 : n=32280  dy -0.110  dz +0.670
TPC1 x>=0: n=14282  dy +0.110  dz -0.670
```

exactly `pos_offset_a0` / `pos_offset_a1`, with the sign flip at the cathode. The drift
term is per cluster:

```
img_cluster   npts   median dx (cm)
        10   11145        52.60
        19   10805         0.21     <- the neutrino candidate
         9    4545       -41.84
        11    3789       121.03
         8    3524       -41.84
         5    3483        -0.21
         7    3333       -59.18
         6    2495        52.60
         2    2206       -41.84
        22     648       -42.88
        20     271       -42.88
        18     218        88.73
clusters >=50 pts: 12   |dx|<1cm: 2   |dx|>10cm: 10   points with |dx|<1cm: 31%
```

### C — the decisive `track_fit` test (`-s C`, 20 spanned events)

Fitted points are new points, and 0.67 cm is smaller than the img point spacing, so any
single nearest-neighbour estimator is dominated by blob geometry. The test is therefore
run **paired**: for each fitted point take the centroid of the charge within 2 cm and
subtract, against *both* clouds. Against the cloud the fit was made from the residual
must vanish and must show no step at the cathode; against the other frame it must carry
the `pos_offset` sign flip.

```
vs raw img                x<0  n= 6156  (+0.090, +0.110, -0.208)
vs raw img                x>0  n= 5136  (-0.308, -0.071, +0.330)
vs raw img                step across the cathode:  dy -0.181  dz +0.538
vs corrected clustering   x<0  n= 6155  (+0.002, -0.001, +0.023)
vs corrected clustering   x>0  n= 5153  (-0.015, -0.002, +0.021)
vs corrected clustering   step across the cathode:  dy -0.001  dz -0.002
```

Against the corrected cloud the residual is **zero to 0.02 cm on both sides with no
step**; against raw it flips sign at the cathode in both y and z. `track_fit` — and with
it `vertices`, which comes from the same `fit()` objects — is in the corrected frame.
(The raw-side magnitudes, −0.21/+0.33 in z, are diluted rather than the full ∓0.67
because the raw cloud is the *whole event*: neighbouring cosmic charge sits at its
uncorrected x and enters the 2 cm ball. The step, not the magnitude, is the measurement.)

### D — `cluster_t0` survives the tarball (`-s D`, evt 280972)

The PR job reloads the persisted point tree and re-runs `switch_scope` at the head of
its pipeline. Its own `clustering-global` and the Q/L job's agree to **0.005 cm**
(46 562 matched points, float round-trip only). So no part of the offset comes from the
PR job re-deriving the correction differently — one hypothesis eliminated.

## 3. What this does *not* affect

- **Nothing in the physics.** All of pattern recognition — clustering input, the PR
  graph, TrackFitting, the taggers, `T_rec_charge` — works consistently inside the
  corrected frame, and `backward()` returns to raw before any 3D→2D wire mapping. This
  is a display-overlay question only.
- **The pr/12 numbers.** The census reads `tracking-pr.root T_rec_charge` and the
  `clustering-global` layer of the Q/L zip; both are corrected-frame, so the cathode
  distances, the 1.08 cm `dyz` and the dQ/dx notch are internally consistent. The only
  cross-frame comparison in pr/12 is the *charge-crossing* test, which uses
  `clustering-global` — corrected — against fitted points — corrected. Unaffected.
- **`op` and `mc`.** `op` carries flash times and PE, no 3D charge positions;
  `mc` is a jsTree of particles built from the same PR objects (corrected).

## 4. What *is* affected, in display terms

Overlaying `img-global` (or the dead-area polygons) on any PR layer shows:

1. every out-of-time cosmic displaced along x by its own `−t0 · v_drift` — tens of cm,
   the visually dominant effect in a busy event;
2. the in-beam candidate displaced by `(≈0, ∓0.11, ±0.67)` cm, with the z sign flipping
   at the cathode, i.e. a **1.34 cm relative z step between the two halves of a
   cathode-crossing track** — precisely the scale of the effect pr/12 is about, which is
   why it is worth removing before the next round of cathode hand-scans.

## 5. Options (nothing implemented — for the owner to choose)

**Option 0 — use `clustering-global` as the charge reference. No change at all.**
`clustering-global` already *is* `img-global` in the corrected frame: same point cloud
(49 150 vs 49 165 points in evt 280972 — the difference is the scope filter, `filter: 1`),
same q, corrected coordinates. It is already in every pr/12 zip. For overlaying the PR
layers it is the correct base layer, and `img-global` becomes what it honestly is: the
pre-correction view. Cost: zero. Downside: the 15 filtered points, and `img-global`'s
per-cluster colouring is the pre-merge one.

**Option 1 — add an `img_cor` layer (jsonnet only, default OFF).** A bee set keyed to the
switch-scope visitor dumps the full cloud right after the correction is applied and
before any merging:

```jsonnet
{ name: 'img_cor', visitor: 'ClusteringSwitchScope:all', grouping: 'live',
  algorithm: 'img_cor', pcname: '3d',
  coords: common_corr_coords(pos_offset_on), filter: 0, individual: false }
```

Feasibility checked, not yet run: sets carrying a `visitor` are dispatched at
`MultiAlgBlobClustering.cxx:2452-2476`, and because no PR graph exists in the Q/L job the
`else` branch is taken — `fill_bee_points` honours `config.coords`. Guarded by the same
key-suppression idiom the file already uses, the compiled config is byte-identical with
the knob off. Open items: confirm the visitor string is the component `type:name`
(`ClusteringSwitchScope:all`, `cfg/pgrapher/common/clus.jsonnet:986-992`, prefix `all`),
and that the bee3 viewer accepts the new algorithm name.

**Option 2 — correct the display at merge time in `make_pr_bee.py`.** The transverse part
is a per-TPC constant, invertible offline from the sign of x; the drift part can be
recovered per cluster by the same unique-`q` match this probe uses. Emits a shifted
`img-global` without touching the toolkit. Ranked below Option 0 because it
reconstructs, with a fragile fingerprint, information that `clustering-global` already
holds; worth it only if the *unfiltered, pre-merge* img point set is specifically wanted.

**Option 3 — dump the PR layers in raw coordinates** (`backward()` per point behind a
knob). Ranked last: it would align the PR layers with `img-global` but break them away
from `clustering-global` and from `T_rec_charge`, moving the inconsistency rather than
removing it.

## 6. Caveats

- All numbers are from SBND **data** arms. `pos_offset` is gated on `reality == 'data'`
  (`clus.jsonnet:1387`), so on MC the transverse component is absent and only the drift
  component separates the frames.
- §2C pools 20 of the 44 pr/12 spanned events; the per-side medians are stable across
  the subset but no per-event spread is quoted.
- The dead-area polygons were read as raw-frame from their dump site, not measured
  against a corrected reference — they carry no z_cor by construction, but the
  consequence for an overlay has not been quantified.

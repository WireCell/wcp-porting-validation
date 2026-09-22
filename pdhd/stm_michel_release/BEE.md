# The Bee event display zip

Each event ships `events/<run6>_<evtid>/bee_<det>_<run6>_<evtid>.zip`, the 3-D display input
of the Wire-Cell "Bee" viewer for the pattern-recognition stage this release comes from.

## How to look at it

1. Open Bee: https://www.phy.bnl.gov/twister/bee (any modern browser; no installation).
2. Use the site's **Upload** function and give it the zip as is (do not unpack it).  Bee
   reads the `data/0/*.json` members and lists each one as a selectable **data layer** for
   event 0 of the upload.
3. Pick the geometry `protodunehd` (`protodunevd` for the PDVD twin); Bee reads it from the
   JSON (`geom` key) and normally selects it by itself.
4. Layers you will use (the drop-down at the top of Bee):

| layer | what it draws |
|---|---|
| `clustering` | every 3-D space point of the clustering stage, coloured by cluster; the cluster number shown when you click a point is the same `cluster_id` as in the ROOT file |
| `stm` | the space points of the clusters the cosmic taggers flagged as stopping-muon candidates (the `T_stm` rows) |
| `stm_fit` | the tagger's own fitted trajectory of those candidates (one point per fitted position; `q` = dQ/dx) |
| `track_fit` | the pattern-recognition fit of every in-scope cluster (what `T_stm_fit` holds for the candidates) |
| `shower_track` | the same fit with the track / shower topology flag |
| `steiner_graph`, `steiner_terminals` | the Steiner-tree skeleton the fit ran on |
| `vertices` | the PR vertices |
| `channel-deadarea-*` | dead channel regions per APA / face |

5. To find one candidate: read its `cluster_id` from `T_stm` (or from
   `scripts/list_candidates.py`), select the `clustering` or `stm` layer, and use Bee's cluster
   filter / click on a point of that cluster — the cluster id is printed in the info box.
   The stop position (`T_stm.stop_x/y/z`, cm) and the Michel start (`michel_start_x/y/z`)
   let you navigate there directly with Bee's coordinate box.

## What is inside the zip

```
data/0/0-clustering-global.json      x, y, z [cm], q, cluster_id, real_cluster_id, runNo, subRunNo, eventNo, geom, type
data/0/0-stm-global.json             same keys, the candidate clusters only
data/0/0-stm_fit-global.json         the tagger fit points
data/0/0-track_fit-global.json       the PR fit points
data/0/0-shower_track-global.json
data/0/0-steiner_graph-global.json, 0-steiner_terminals-global.json
data/0/0-vertices-global.json
data/0/0-channel-deadarea-apa<N>-face<F>.json
data/0/0-mc.json                     placeholder (no truth on data)
```

The JSON files are plain and can be read with `json.load` if you want the clustering-stage
space points outside Bee; `cluster_id` there joins to `T_stm.cluster_id` (the `stm` layer's
ids are exactly the `T_stm` rows).  Bee draws only what is in the zip: there is no flash /
light layer in this stage's zip — use `T_flash` / `T_opwf` for the light.

## Coordinates

Bee and the ROOT trees share the detector coordinate system: x is the drift direction
(PDHD: APA 0/2 at x < 0 ... the anode at |x| = 352.1 cm is the readout plane, the cathode at
x = 0), y vertical, z along the beam; all in cm.  The drift coordinate x is placed from the
matched flash time (`T_stm.t0_us` = the flash time the cluster was matched to).

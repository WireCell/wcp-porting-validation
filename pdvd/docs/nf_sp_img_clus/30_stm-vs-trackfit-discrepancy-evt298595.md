# 30 — Event 298595, cluster 86: why the STM-fit point reads "cluster 0" in Bee, and why `track_fit_global` looks like a different event there

## Repro block

```
# work dir (fresh v450-production smoke-test run, doc pdvd/29):
W=/home/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039252_2_v450

# Bee dump: stm_fit / track_fit / clustering(raw) / vertices global layers
unzip -j $W/mabc-pr.zip \
  "data/0/0-stm_fit-global.json" "data/0/0-track_fit-global.json" \
  "data/0/0-clustering-global.json" "data/0/0-vertices-global.json" -d <outdir>

# PR log (verdict + fit provenance lines quoted below)
grep -n "cluster 86" $W/wct_pr_039252_2.log

# calib dump (per-candidate main_vertex / kine)
python3 -c "import json; d=json.load(open('$W/calib-pr-evt298595.json')); print(d['candidates'])"
```

Run 39252, event 298595, cluster 86 (production defaults, doc pdvd/29: E=450 V/cm,
DL/DT=4.1307/7.9135, wires v7-uvwfit).

## Question 1: point (306.5, -243.4, 102.0) reads `real_cluster_id = 0` in Bee — why?

**Not a per-point or per-event anomaly.** The queried point matches
`stm_fit-global` index 4507 almost exactly (306.551, -243.426, 101.918 —
0.1 mm off), with `cluster_id = 86` but `real_cluster_id = 0`. Checking
*every* point in this event's `stm_fit-global` layer (7308 points, 22
distinct `cluster_id` values) shows `real_cluster_id == 0` for **all** of
them — not just cluster 86, not just this point.

Root cause is in the writer, `MultiAlgBlobClustering::fill_bee_points_from_cluster`
(`clus/src/MultiAlgBlobClustering.cxx:2859-2876`), the `pcname == "stm_fit"`
branch:

```cxx
for (size_t i = 0; i < fx.size(); ++i) {
    bpts.append(Point(fx[i], fy[i], fz[i]), fdQ[i]*dQdx_scale + dQdx_offset, clid, 0);
}
```

`Bee::Points::append(point, q, clid, real_clid)` (`util/inc/WireCellUtil/Bee.h:127`)
is called with the literal `0` as the fourth argument for every STM-fit point,
regardless of `clid`. The genuine `cluster_id` (86) IS written correctly and
is what the raw (unfit) `clustering-global` layer also reports at this exact
coordinate (121 raw points within 10 cm, all `cluster_id = 86`, nearest raw
point 0.42 cm away) — so "cluster 86" is the real answer; "0" is a Bee
display-field the `stm_fit` writer never populates. Compare: the `track_fit`
branch of the same function (general scoped-view path, further down in the
same function) *does* propagate a real `real_cluster_id` when the cluster
carries a `"real_cluster_id"/"perblob"` pcarray (see cluster 86's
`track_fit-global` points below: `86002`, `86003`, `86005`, `86009`,
`86010`, `-1` — genuinely varying). The `stm_fit` branch is the only one of
the three (`stm_fit` / `steiner_pc` / everything-else) that hard-codes it.

This is a **display-layer gap in the diagnostic `save_stm_fit` dump**, not a
reconstruction defect — `save_stm_fit` is default OFF in production (doc
pdvd/25 sec 7b) and exists to let a human inspect *why* TaggerCheckSTM tagged
a cluster, so nothing downstream consumes `stm_fit`'s `real_cluster_id`. Not
fixed here (read-only investigation); flagged for whoever next touches that
function.

## Question 2: this point looks like an STM, but `track_fit_global` shows something else there, and its Michel trajectory doesn't match

### The two layers are two independently-fit, independently-fired algorithms

`stm_fit-global` and `track_fit-global` are Bee dumps of **two different
grouping-level `TrackFitting` slots**, filled by **two different
components**, dumped at **two different points** in the PR pipeline
(`cfg/pgrapher/experiment/protodunevd/pr.jsonnet`'s `bee_points_sets`):

| Bee layer | slot | filled by | dumped after (`visitor:`) |
|---|---|---|---|
| `stm_fit` | `"stm"` (named) | `TaggerCheckSTM`'s own internal trial fit (`m_track_fitter`), `clus/src/TaggerCheckSTM.cxx:614-634` | `TaggerCheckSTM:pr` |
| `track_fit` | default/unnamed | `TaggerCheckNeutrino`'s PR-graph fit for its own candidate | `TaggerCheckNeutrino:pr` |

Nothing in the pipeline requires (or checks) that TaggerCheckNeutrino's PR
graph, built independently afterward, reproduces the trajectory
TaggerCheckSTM used to justify its STM tag.

### What each one actually did with cluster 86 (log evidence, `wct_pr_039252_2.log`)

```
component_extreme_wcps: cluster 86 3 component(s), 1 above 10.0 cm -> 6 extreme group(s)
check_tgm: cluster 86 CASE-B pair (1,5) rejected: rescued end, straight chord 57.7 cm has an unsupported run > 30.0 cm
visit: TaggerCheckTGM: cluster 86 -> TGM=false
check_other_tracks: cluster 86 seg 1/1: len=3.5cm medQ=0.38MIP lenThr=0.0cm straight=0.989 front=(292.8,-206.4,45.3)cm
visit: TaggerCheckSTM: cluster 86 -> STM=1 TGM=0
persist_stm_fit: cluster 86 stmfit pass=0 status=0 kink=97 exit_L=61.8 left_L=28.2 npts=141
...
TaggerCheckNeutrino: [nu_per_bundle] gid 192: candidate main cluster 86 (t0 6484.400 us, L 70.6 cm, 0 associated)
dual_chain: OFF pass (snap) vertex=true (290.69,-231.98,80.53) cm cluster 86 5 candidates, 0 voxels
sgp guard: cluster 86 VETO maxsep=4.119 cap=3.000 n_gap=135 n_base=128 detour=6.446 base_cm=129.831
  first=(283.45,-228.11,93.75) last=(343.28,-253.19,92.73)     <- repeats several times, same pair
TaggerCheckNeutrino: [nu_per_bundle] ROW 1 gid 192 cluster 86 vertex (293.0316,-205.2276,45.2573) cm Enu 909.5364 acts 1
```

Cluster 86's raw (pre-fit) 3D imaging footprint (`clustering-global`, 598
points) is a single connected but **branchy** object: 3 components / 6
"extreme" (far) points, bounding box x=[281.7,343.3] y=[-253.8,-205.4]
z=[44.8,104.5], with a dense (~200-point) cluster right at x=[282,285] —
a junction — and two "arms" reaching away from it: one toward
(344.0,-252.9,92.8), the other toward (293.0,-205.2,45.3) — a very
different direction (drops 48 cm in z).

* **TaggerCheckSTM** treated the junction as the muon-stop / Michel-decay
  point and fit a single, smooth, densely-sampled (141 pts, ~0.4 cm
  spacing, `q` ~2000-4000, MIP-like, no obvious Bragg rise) trajectory
  along the "toward-344" arm. The queried point (306.5,-243.4,102.0) sits
  mid-way along that arm, ~25 cm past the junction. This is the fit that
  produced `STM=1`. Its own second-track check (`check_other_tracks`) saw
  only a short, low-charge 3.5 cm stub near the *other* arm's end
  (292.8,-206.4,45.3) and did not treat it as competing with the main fit.
* **TaggerCheckNeutrino**, run afterward as an independent candidate search
  (`gid 192`, `nu_index 1` in the calib dump), explored a vertex right at
  the same junction (`dual_chain` snap, 290.7,-232.0,80.5 — matches STM's
  anchor to within a few cm) but ended up **selecting the opposite arm's
  end** as this candidate's `main_vertex`: (293.03,-205.23,45.26) — exactly
  matching `check_other_tracks`'s "irrelevant" 3.5 cm stub and matching
  vertex marker `86003` in `vertices-global`. Its own Steiner-graph path
  builder logged repeated `sgp guard: ... VETO` lines while trying to
  connect the junction to the "toward-344" endpoint (`maxsep=4.119 cm`
  against a `3.000 cm` cap, `n_gap=135` of `n_base=128` edges flagged gap)
  — i.e. it repeatedly found that route badly supported by its own graph
  and fell back to a different-flavor path rather than accept it outright.
* **TaggerCheckTGM**, a third, independent algorithm, separately flagged
  a straight chord near this same cluster as having "an unsupported run
  > 30.0 cm" and rejected a through-going-muon classification on that
  basis.

Net effect on the Bee dumps: `track_fit-global`'s allocation for cluster 86
is only 41 points (vs. 293 for the event's actual selected main-vertex
cluster, 40) — nearly all of them packed into the 2 cm-wide junction region,
plus a handful of isolated markers at the two arm extremities. **None are
within 20 cm of the queried point**, and the low-z arm end (the "Michel-like"
short stub) that *does* survive into `track_fit_global` sits ~63 cm away
from, and in a visually unrelated direction from, the junction/kink that
`stm_fit`'s trajectory treats as the decay point. That mismatch is exactly
what looks like "the Michel trajectory in track_fit_global is not consistent
with the STM fit."

### Is the STM leg real charge, or a fit artifact?

The raw `clustering-global` points are not gapped in 3D along x (5 cm bins
from x=280 to x=340 all carry 21-42 points, no >30 cm empty run), so the STM
leg is not fitted through empty space in the crude sense. But `chord_has_charge`
(TGM) and the Steiner `sgp` guard (NeutrinoID) both operate on stricter
graph/2D-view continuity, not raw 3D-point density in 5 cm bins, and both
independently flagged something about this cluster's long-range
connectivity that `TaggerCheckSTM`'s own fitter — which has no equivalent
continuity check — did not. **This is presented as the leading hypothesis,
not a proven code-path trace**: I have not walked `chord_has_charge`'s or
the Steiner graph's exact edge-construction logic far enough to say with
certainty that the *same* stretch (junction -> x=337, the STM leg) is what
both guards independently rejected, versus a different extreme-point pairing
across the whole branchy cluster (e.g. junction -> low-z arm, which
genuinely does bend 48 cm in z and would legitimately fail a straight-chord
support test). Either reading is consistent with the STM=1 / sparse-track_fit
outcome; they imply different follow-ups (see below).

## Status

Investigation only — nothing changed. Byte-identical (no code or config
touched).

## Recommendation / next steps

1. **Question 1** (display bug) is unambiguous and easy: thread the same
   `"real_cluster_id"/"perblob"` lookup used by the general scoped-view
   branch of `fill_bee_points_from_cluster` into the `stm_fit` branch, or at
   minimum stamp `clid` instead of the literal `0` — `save_stm_fit` is a
   diagnostic knob, so this is a UX fix, not a physics change, but worth
   doing before anyone else hand-scans STM fits in Bee and is misled by it.
2. **Question 2** is a genuine open question about whether cluster 86's
   junction->344 arm is one real track or a merge artifact — that decides
   whether the STM=1 tag or the neutrino-vertex placement is the more
   trustworthy read of this cluster, and I don't think this investigation
   settles it. Two concrete next steps that would: (a) dump and inspect the
   Steiner graph's actual edge list / gap locations for cluster 86 (the
   `sgp guard` log lines above have the endpoints and gap counts already;
   the missing piece is *where along the path* the flagged edges sit), and
   (b) look at the raw 2D wire signals (U/V/W ADC, not the 3D imaging
   points) for the junction and the x=293-337 stretch for a second,
   overlapping in-time activity that dual-view 3D imaging could be
   mis-merging into one cluster (a known PDVD failure mode; see doc
   pdvd/25's iso-overcluster investigations, docs 25-26 in this directory).
3. Not recommending a fix without (2): per project policy, a physics number
   that looks inconsistent is reported, not silently tuned to "look right."

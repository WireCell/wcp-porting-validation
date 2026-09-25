# 06 — The isochronous slab: why the doc 05 chain cannot solve it, the missing link, and the redesign

*2026-09-25. Analysis only: no toolkit change, no new simulation, no retraining. Everything below is measured on the
fixed-truth (`_sub4f`) tiers and graphs of doc 05 §9 and on the toolkit / prototype source as of toolkit `1e6b2905`,
wcp `ee99f237`.*

## 0. Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/wcfm
python3 scripts/iso_slab_probe.py --out docs/06_tables        # graphs_f + ablf_charge scores + _sub4f archives, ~3 min
# writes docs/06_tables/{slab_census,consistency,fm_similarity,gnn_by_end_distance,line_rule,measures}.md
#        docs/06_tables/view-000001_<evt>-anode<N>.png   (one per single-0-deg anode-event)
```

Inputs: the fixed-truth sub-blob graphs `/home/xqian/tmp/wcfm-gnn/graphs_f/graph-000001_<evt>-anode<N>.npz`
(doc 05 §9.4, `gnn_dataset.py`), the charge-arm scores `/home/xqian/tmp/wcfm-gnn/ablf_charge/scores_charge_s{0,1,2}.npz`
(doc 05 §9.4, `gnn_train.py`), the `work/000001_<evt>_sub4f/clusters-{tru0,apa}-anode<N>-ms-active.tar.gz` archives
(cell centres from the tru0 `bnodes` corners; the legacy solver's measure nodes from the apa `mnodes`/`bmedges`).
Population: the 7 single-track 0° events of the doc 05 sample (1, 2, 25, 27, 51, 60, 64 → 13 anode-events) and,
as the contrast, the 3 single-track 0.5° events (19, 48, 95 → 5 anode-events).

## 1. The question

The target of this round is the isochronous (ISO) case: a track parallel to the anode plane puts all its charge into a
few 2 µs slices, and the 3-D imaging built on merged wires is then maximally ambiguous. The owner's design intent had
three information sources that should, combined, resolve the sub-blob ambiguity:

1. **sub-blob deghosting** — decide which wire-crossing combinations inside a merged blob are true and which are ghosts;
2. **charge matching** — a dQ/dx fluctuation at a point of the track is seen by all three views at the corresponding
   wires; where the three views agree the track passes, where they do not the region is ambiguous;
3. **FM topology** — the foundation-model features carry what kind of thing is at a pixel (Michel, a different track,
   a shower), which should help decide which crossings are real.

Doc 05 §9.5 found the opposite: on the fixed truth the charge-only GNN keeps 0.81 of the true charge overall but
**0.25 at exactly 0°**, the FM arm is *worse* there (0.22), the legacy chain keeps 0.06, and the FM ablation is a NO-GO
on the pre-registered gate. This note asks *why*, source by source, and what has to change.

The short answer: **all three sources were never given a chance.** Deghosting an isochronous slab is a *cross-view
association* problem (which U pixel belongs with which V pixel and which W pixel). Sources 2 and 3 can only answer it
where pixels are distinguishable along the track — by texture or landmarks — and the sample has neither, by
construction. Where they are indistinguishable the only remaining information is global geometry (the track is a line
through the ambiguity region), which neither the factor graph, nor the message passing, nor the per-cell loss, nor the
legacy L1 solve can express. §2 shows what a 0° slab is; §3 measures each source; §4 shows what the model and the legacy
solver actually compute; §5 states the missing link; §6 gives the redesign with go/no-go criteria.

## 2. What a 0° slab is

A 0° track of length L in the y–z plane at drift x₀ arrives in the slices around x₀/v; diffusion spreads it over 2–4
slices which are *copies* of one another (same cells, same candidates). In each slice the U, V and W planes each see one
contiguous interval of active channels, and the tiling produces every (u, v, w) triple that is geometrically consistent:
the candidate set is the intersection of three wire strips — a hexagon in y–z — and the track is one chord of it.
The number of candidates scales with the hexagon area, the number of true cells with L.

Census of the single-track 0° anode-events (`docs/06_tables/slab_census.md`; "candidates" = all sub-blobs of the slice
on the fixed `tru0` tier, "real" = q_true > 0):

| event / anode | φ_yz [°] | L [cm] | slices with true charge | real cells per slice | candidates per slice | fan-in median (max) U / V / W |
|---|---|---|---|---|---|---|
| 1 / 10 | −160 | 300 | 3 | 69 / 69 / 69 | 989 / 989 / 989 | 25 (30) / 47 (84) / 22 (75) |
| 2 / 2 | 46 | 500 | 4 | 129 / 129 / 129 / 572 | 1151 / 1151 / 1151 / 8418 | 257 (272) / 47 (64) / 64 (93) |
| 25 / 3 | −95 | 250 | 4 | 112 × 4 | 893 × 4 | 16 (27) / 18 (27) / 76 (162) |
| 27 / 9 | 148 | 400 | 3 | 77 × 3 | 1182 × 3 | 21 (26) / 60 (87) / 22 (77) |
| 51 / 10 | 3 | 300 | 4 | 46 × 4 | 920 × 4 | 33 (41) / 33 (61) / 18 (107) |
| 60 / 6 | −93 | 200 | 2 | 144 / 156 | 727 / 693 | 11 (17) / 13 (19) / 109 (158) |
| 64 / 2 | −2 | 300 | 4 | 34 × 4 | 908 × 4 | 32 (39) / 34 (62) / 18 (107) |

Fan-in = number of candidate cells covering one wire. The ratio candidates : real is 4–30 per slice. The three slices of
a slab are literally the same picture — the cross-slice `bb` edges of the graph (the only blob–blob edges it has, §4.1)
connect every cell to its own copy.

**The contrast that matters: 0.5° is no less ambiguous per slice.** At 0.5° a 300 cm track spans 2.6 cm in x = 8–14
slices, each holding a 25–40 cm segment:

| event / anode | θ | slices | real per slice | candidates per slice | charge GNN recall in the slab |
|---|---|---|---|---|---|
| 19 / 11 | 0.5° | 8 | 42–89 | 811–1515 | 0.76 |
| 95 / 2 | 0.5° | 12 | 26–89 | 715–3261 | 0.91 |
| 95 / 3 | 0.5° | 7 | 15–87 | 691–1220 | 0.90 |

The per-slice ratio (10–60) is *worse* than at 0°, yet the charge GNN keeps 0.76–0.91 of the real cells there and
0.00–0.19 at 0°. What differs is not the ambiguity inside a slice but that consecutive slices hold *different* segments,
so the cross-slice edges carry the continuity of the track and the wire-charge pattern changes from slice to slice
(§3.1). At 0° that information does not exist.

## 3. The three sources, measured

### 3.1 Charge matching: the sample has no texture to match

`gen_iso_tracks.py:39` sets `CHARGE_PER_STEP = -500` electrons per 0.1 mm step for every step of every track.
`gen/src/TrackDepos.cxx:75-103` turns that into one depo per step with exactly that charge and uses no random number.
Downstream the only fluctuation is counting statistics: binomial absorption in `gen/src/Drifter.cxx:169-174` and
binomial per-bin sampling in `gen/src/GaussianDiffusion.cxx:185-200` (`fluctuate` true through
`cfg/pgrapher/experiment/dune10kt-1x2x6/simparams.jsonnet:182`). Nothing in `gen/` — nor in the python `depo-lines`
/ `linegen` generators (`wirecell/gen/depogen.py:83`) — produces Landau straggling, delta rays or any other along-track
structure. A 0° track in this sample is a line of constant dQ/dx read out through a constant response.

Measured consequence. For every candidate cell take the mean packed wire charge of its wires in each plane, normalise
by the slice-plane median, and score the cell by minus the variance of the three log-ratios (a cell whose three views
"agree" scores high). AP of that statistic for real vs ghost (`docs/06_tables/consistency.md`):

| event / anode | θ | candidates | real | prior | AP of the 3-view consistency statistic | wire-charge CV along the track U / V / W |
|---|---|---|---|---|---|---|
| 1 / 10 | 0° | 989 | 69 | 0.070 | 0.087 | 0.12 / 0.09 / 0.06 |
| 2 / 4 | 0° | 3748 | 375 | 0.100 | 0.104 | 0.20 / 0.53 / 0.22 |
| 25 / 3 | 0° | 893 | 112 | 0.125 | 0.170 | 0.13 / 0.17 / 0.20 |
| 27 / 9 | 0° | 1182 | 77 | 0.065 | 0.087 | 0.09 / 0.11 / 0.03 |
| 51 / 10 | 0° | 920 | 46 | 0.050 | 0.081 | 0.27 / 0.31 / 0.17 |
| 64 / 2 | 0° | 908 | 34 | 0.037 | 0.092 | 0.16 / 0.22 / 0.11 |
| 19 / 11 | 0.5° | 1515 | 89 | 0.059 | 0.180 | 0.47 / 0.50 / 0.52 |
| 95 / 2 | 0.5° | 1565 | 89 | 0.057 | 0.128 | 0.42 / 0.42 / 0.60 |
| 95 / 3 | 0.5° | 981 | 87 | 0.089 | 0.263 | 0.45 / 0.45 / 0.61 |

At 0° the statistic is at the prior (AP / prior 1.0–1.6 on 12 of the 13 anode-events, 2.5 on the smallest, 34 real cells): every wire under the track carries the
same charge up to SP noise (CV 0.03–0.56, mostly ≤ 0.3), so the three wire charges of a ghost — which are the wire charges of three
*different* real cells — agree exactly as well as those of a real cell. **There is nothing for charge matching to
match.** At 0.5° the same statistic is already 2–7× the prior, from a trivial "texture": the track's charge is split
between neighbouring slices differently along its length (CV 0.4–0.6). That is the whole reason the 0.5° stratum is
solvable and the 0° one is not — not a property of the model.

What real texture would look like: a MIP's Landau straggling over a 4-wire (2 cm) cell has FWHM/MPV ≈ 20–30 % with a
long tail, i.e. CV ≈ 0.2–0.3 against the 0.05–0.15 noise floor measured here; delta rays above a few mm occur several
times per metre and are unmistakable landmarks; in real events kinks, vertices, Michels and track crossings add more.
Even then, source 2 is not a *pointwise* test: each wire's charge is the sum over the cells on it, and a single ratio
of three wire charges is ambiguous; the information is in the three *sequences* q_U(u), q_V(v), q_W(w), which are the
same dQ/dx profile sampled at three rates. Matching them is a 1-D alignment problem (stereo correspondence), and its
solution *is* the u↔v↔w association — the line.

### 3.2 FM topology: a per-view descriptor cannot associate views unless it carries position

A projective ghost is made of real pixels: the ghost (u, v′, w′) pools exactly the U pixel of the real cell (u, v, w),
the V pixel of another real cell and the W pixel of a third. Any per-view feature — charge or a 128-d FM descriptor —
is therefore identical for the ghost and for the real cells it borrows from, *unless the descriptor differs along the
track*, so that "the U descriptor at u" and "the V descriptor at v′" can be recognised as belonging to different
places on the track.

Measured (`docs/06_tables/fm_similarity.md`): cosine similarity of the FM wire descriptor (`w_fm`, the 128-d student
output at the pixel, doc 04) between on-track wires of the same plane at increasing channel separation, 0° slabs:

| event / anode | plane | on-track wires | cos-sim at 1 / 5 / 20 / 100 channels |
|---|---|---|---|
| 1 / 10 | U / V / W | 447 / 262 / 424 | 0.998 / 0.989 / 0.982 / 0.941 · 0.999 / 0.992 / 0.970 / 0.797 · 0.999 / 0.993 / 0.988 / 0.942 |
| 25 / 3 | U / V / W | 275 / 352 / 48 | 0.999 / 0.988 / 0.973 / 0.899 · 1.000 / 0.996 / 0.981 / 0.776 · 0.999 / 0.982 / 0.946 / – |
| 51 / 10 | U / V / W | 202 / 219 / 251 | 0.998 / 0.984 / 0.959 / 0.730 · 0.997 / 0.983 / 0.962 / 0.903 · 0.999 / 0.994 / 0.986 / 0.887 |

The descriptor is translation-invariant along the line: ≥ 0.997 between neighbouring wires, ≥ 0.96 at 5 wires, still
0.90–0.99 at 20 wires (10 cm) and 0.52–0.98 at 100 wires, on every plane of every 0° anode-event (39 plane rows). It does distinguish on-track from off-track
pixels (cos 0.22–0.53 where the slice has off-track wires) — the "track vs noise" signal that the leaky truth of doc 05
§4 rewarded — but it cannot say *where along the track* a pixel is, so it cannot associate views. **FM (source 3) can
contribute to deghosting only at landmarks**: a delta ray, kink, Michel, vertex or crossing gives the pixels near it a
descriptor that differs from the rest of the line in all three views at once, and matching those descriptors across
views pins the association there. The sample has no landmarks except the overlay crossings — and the overlays that
contain a 0° track are exactly where the 0° charge recall rises from 0.25 to 0.54–0.61 (doc 05 §9.5 angle table).

### 3.3 The information that is left: the geometry of the region

With texture and landmarks absent, what remains in a 0° slab is the shape of the candidate region. The intervals of
active channels in the three planes are the projections of the track segment, so *both ends of the track are extreme in
all three pitch coordinates at once*; the track is the chord of the hexagon joining the two vertices with that property.
No other candidate cell has it: the four other hexagon vertices are extreme in two planes only, and interior cells in
none. This is a global statement about the region, not a property of any cell or wire.

A truth-free rule that uses only this (`iso_slab_probe.line_rule`): take the wire-connected component of a slice's
candidates, compute the three pitch coordinates of every cell centre (perpendicular to the U 54°, V 126°, W 90° wire
directions; channel numbers cannot be used — on the wrapped U/V planes they are not a position, a 4 m track on anode 9
of event 27 "covers" all 800 U channels), take the 3 lowest and 3 highest cells per coordinate as endpoint candidates,
choose the pair whose separation covers all three intervals best, keep the cells within r of that chord. Result on the
whole slab (`docs/06_tables/line_rule.md`; the GNN and the legacy chain evaluated on the same cells):

| event / anode | slab cells | real | line rule r < 3 cm: cell recall / precision / **charge recall** | charge GNN P ≥ 0.5: recall / precision / charge recall | legacy chain: recall / precision / charge recall |
|---|---|---|---|---|---|
| 1 / 10 | 2967 | 207 | 0.44 / 0.91 / **0.60** | 0.11 / 0.12 / 0.18 | 0.02 / 0.36 / 0.06 |
| 1 / 8 | 2529 | 201 | 0.94 / 0.67 / **0.95** | 0.04 / 0.50 / 0.13 | 0.03 / 0.22 / 0.07 |
| 2 / 4 | 6490 | 796 | 0.96 / 0.66 / **0.95** | 0.01 / 1.00 / 0.06 | 0.05 / 0.18 / 0.08 |
| 25 / 3 | 3572 | 448 | 0.98 / 0.49 / **0.98** | 0.02 / 0.28 / 0.14 | 0.06 / 0.27 / 0.06 |
| 27 / 11 | 2031 | 252 | 0.96 / 0.62 / **0.95** | 0.08 / 0.63 / 0.24 | 0.05 / 0.33 / 0.06 |
| 27 / 7 | 1743 | 246 | 0.96 / 0.62 / **0.95** | 0.13 / 0.67 / 0.18 | 0.05 / 0.15 / 0.05 |
| 51 / 10 | 3680 | 184 | 0.67 / 0.67 / **0.86** | 0.08 / 1.00 / 0.24 | 0.05 / 0.33 / 0.06 |
| 51 / 8 | 5554 | 250 | 0.72 / 0.72 / **0.63** | 0.19 / 0.96 / 0.87 | 0.05 / 0.41 / 0.10 |
| 60 / 6 | 1420 | 300 | 1.00 / 0.42 / **1.00** | 0.05 / 0.41 / 0.05 | 0.10 / 0.27 / 0.12 |
| 64 / 2 | 3632 | 136 | 0.85 / 0.62 / **0.99** | 0.03 / 1.00 / 0.09 | 0.15 / 0.28 / 0.16 |
| 2 / 2 | 11871 | 959 | 0.04 / 0.06 / 0.04 | 0.00 / 1.00 / 0.05 | 0.05 / 0.12 / 0.04 |
| 27 / 9 | 3546 | 231 | 0.01 / 0.02 / 0.04 | 0.14 / 0.18 / 0.37 | 0.00 / 0.00 / 0.00 |
| 64 / 0 | 6766 | 246 | 0.46 / 0.36 / 0.02 | 0.08 / 1.00 / 0.56 | 0.05 / 0.17 / 0.06 |

Ten of the thirteen 0° anode-events are solved by a 40-line geometric rule that uses no charge, no FM and no training
(charge recall 0.60–1.00, median 0.95; the GNN's median on the same slabs is 0.18, the legacy chain's 0.06). The three
failures are the slabs whose candidate region is *not* one track's hexagon: in 2 / 2 and 64 / 0 the region reaches the
full APA height (y −600…−2 cm for a track at y −536…−322) because the wrapped U/V wires tile ghosts across the whole
face, and in 27 / 9 the component mixes both faces; there the region's vertices are not the track's ends and the chord
is wrong. Those failures are the reason a global *learned* method is needed rather than this rule — but the rule settles
the question the doc 05 result raised: **the 0° slab is solvable from information that is in the data; the doc 05 chain
cannot represent that information.** See `docs/06_tables/view-000001_25-anode3.png` (truth / GNN keep set / rule keep
set) for the picture.

## 4. What the doc 05 chain actually computes on a 0° slab

### 4.1 The graph has no edge that could carry continuity or "this is an end"

- Blob–blob edges come only from `Img::geom_clustering` (`img/src/GeomClusteringUtil.cxx:41-105`), which compares a
  blobset only with *later* blobsets (`for (auto test = next; …)`, line 83) and skips a relative time difference of 0
  (line 85). `BlobCutting` builds no edges at all (`img/inc/WireCellImg/BlobCutting.h:1-27`); `BlobGrouping` relates
  in-slice blobs only through shared measure nodes (`img/src/BlobGrouping.cxx:131-166`); no other `add_edge` in `img/`
  creates a b–b edge. `gnn_dataset.py:134` asserts that every `bb` edge of the dataset is cross-slice.
- The graph has no wire–wire edge either. A wire node cannot know that it is the first or last active channel of its
  plane's interval, so a cell cannot know it sits at a hexagon vertex — the one local fact from which §3.3 follows.
- At 0° every cross-slice edge joins a cell to its copy in the neighbouring slice (§2). The message passing therefore
  has, inside the slab, only the factor edges (cell ↔ its 4 wires per plane), and those connect real cells and ghosts
  symmetrically: a ghost is *defined* as a cell whose wires are all real.

### 4.2 The wire update averages over the whole strip and has no conservation channel

`FactorGNN` (doc 05 §3; `gnn_train.py`) updates a wire from the *mean and max* of the embeddings of the blobs that
cover it plus log-degree, and a blob from the per-plane mean over its wires plus the mean over its `bb` neighbours. In a
0° slab a wire is covered by 11–257 candidates (fan-in above), of which 1–4 are real: the mean is the strip's average
and the max is whichever cell happens to have the largest activation. Nothing in the layer computes the quantity charge
matching is about — the **residual of the wire**, q_w − Σ_{b ∋ w} p_b q̂_b, i.e. whether the cells the model is
currently keeping explain the wire's measured charge. Without that channel a wire cannot tell the cells "you are
over-explaining me" or "someone must still explain me", and the model cannot propagate a keep decision along the track
through charge conservation.

### 4.3 The loss asks for the marginal, and in a degenerate slab the marginal is the prior

The readout is one logit per cell trained with per-cell BCE (`gnn_train.py:236`). In a slab where every support that
places the same charge on every wire is equally consistent with the data, the Bayes-optimal *marginal* P(real | data)
is ≈ (real cells / candidates) for every interior cell. That is exactly what the trained network outputs
(`docs/06_tables/gnn_by_end_distance.md`, charge arm, 3-seed mean, whole slab):

| event / anode | slab cells | real | GNN kept (real) | legacy kept (real) | mean P(real): real / ghost | recall of real cells at < 10 / 10–30 / 30–80 / > 80 cm from a track end |
|---|---|---|---|---|---|---|
| 1 / 10 | 2967 | 207 | 190 (23) | 11 (4) | 0.13 / 0.07 | 1.00 / 0.50 / 0.05 / 0.08 |
| 2 / 4 | 6490 | 796 | 6 (6) | 229 (40) | 0.08 / 0.05 | 0.14 / 0.00 / 0.00 / 0.01 |
| 25 / 3 | 3572 | 448 | 32 (9) | 105 (28) | 0.22 / 0.14 | 0.00 / 0.00 / 0.05 / 0.01 |
| 27 / 7 | 1743 | 246 | 48 (32) | 78 (12) | 0.32 / 0.15 | 0.38 / 0.04 / 0.12 / – |
| 51 / 10 | 3680 | 184 | 14 (14) | 27 (9) | 0.21 / 0.05 | 0.88 / 0.03 / 0.00 / 0.11 |
| 60 / 6 | 1420 | 300 | 34 (14) | 112 (30) | 0.28 / 0.24 | 0.25 / 0.02 / 0.02 / 0.05 |
| 64 / 2 | 3632 | 136 | 4 (4) | 71 (20) | 0.17 / 0.06 | 0.38 / 0.00 / 0.00 / 0.08 |
| 19 / 11 (0.5°) | 10293 | 559 | 588 (425) | 425 (39) | 0.60 / 0.04 | 0.87 / 0.72 / 0.75 / 0.78 |
| 95 / 2 (0.5°) | 22260 | 681 | 960 (618) | 859 (42) | 0.72 / 0.03 | 0.94 / 0.98 / 0.88 / 0.91 |

At 0° the real cells sit at P(real) 0.05–0.32 and the ghosts at 0.02–0.24 — a weak, correct ranking that never crosses
0.5 except within 10 cm of a track end, where the degree features let a cell see that its strips end (recall 0–1.00
in the first bin, ≤ 0.13 beyond 30 cm on 11 of 13 anode-events). At 0.5° the same network puts real cells at 0.60–0.72 and ghosts at 0.03–0.04.
The network has not failed to learn; it has learned the marginal, and the marginal is uninformative. **The question
"is this cell real?" has no per-cell answer in a degenerate slab; only "which configuration explains the slab?" has
one**, and a per-cell BCE cannot ask it.

### 4.4 The legacy solver sums the wires away before it solves

The legacy chain fails for a related but distinct reason. `ChargeSolving` does not solve on wires: its measurement rows
are the `BlobGrouping` measures, one per *connected group of channels* per slice and plane (`BlobGrouping.cxx:131-166`,
`CSGraph.cxx:108-134`). In an isochronous slab the track's channels form one contiguous run per plane, so the whole slab
collapses to a handful of rows (`docs/06_tables/measures.md`, from the apa archives):

| event / anode | slice | blobs the solver kept | measure rows the solver had |
|---|---|---|---|
| 1 / 10 | 748 / 749 / 750 | 187 / 17 / 147 | 4 / 4 / 4 |
| 25 / 3 | 888–891 | 152 / 69 / 94 / 157 | 12 / 9 / 9 / 11 |
| 27 / 9 | 342 / 343 / 344 | 210 / 19 / 149 | 6 / 6 / 6 |
| 51 / 10 | 745–748 | 63 / 19 / 12 / 76 | 5 / 4 / 4 / 4 |
| 64 / 2 | 539–542 | 67 / 42 / 49 / 70 | 6 / 6 / 6 / 7 |
| 19 / 11 (0.5°) | 510–517 | 87–258 | 14–29 |

Four to twelve rows for 12–210 blobs in the well-formed slabs (120 rows for 628 blobs in the split slice of event 2). The lasso (`solve_config: uboone`, `CSGraph.cxx:144-149`:
λ = 3/(2 Q_tot)·scale, non-negative, zero initialisation, coordinate descent in blob-ident order `CSGraph.cxx:17-31`,
soft threshold `util/src/LassoModel.cxx:221`) is then maximally degenerate — every support with the right sums has the
same objective — and the tie is broken by the per-blob weights (`uboone` strategy: 9, 3 or 1 depending on cross-slice
neighbours with value ≥ 300, `ChargeSolving.cxx:54-131`; uniform inside a slab of copies) and by the sweep order. The
result is the arbitrary support of doc 05: 0–44 real cells among 11–364 kept per slab in the §4.3 table. This is the owner's
"merged wires" point 1 made quantitative: the per-wire charge, which is the only place source 2 could live, is discarded
by the row definition before the solve starts.

### 4.5 The prototype already has the likelihood, but for a known path

The WCP prototype's `dQ_dx_fit` (`prototype_base/wire-cell/pid/src/PR3DCluster_dQ_dx_fit.h:368-1045`, ported as
`clus/src/TrackFitting.cxx:8777`) is three-view charge matching in the sense of source 2: it minimises
Σ_planes ‖M^{½}(d − R Q)‖² + ‖F Q‖² where d are the per-pixel 2-D charges, R integrates a diffusion-width Gaussian of
each path point over each pixel (`cal_gaus_integral_seg`), M down-weights pixels shared by many path points
(`cal_compact_matrix`, lines 3-140) and F is a second-difference smoothness along the path (lines 866-921,
λ = 0.0005). It *assumes the path* (from the Steiner/trajectory stage) and fits the charge along it. The ISO problem is
the converse: find the path whose fitted projections reproduce the three 2-D charges. The prototype has no imaging-level
isochronous logic (the only `isochronous` code is vertex/segment repair in `NeutrinoID_proto_vertex.h:1410-1462`); it
handles a 0° track by leaving the merged blob fat and letting the trajectory + dQ/dx fit sort it out downstream.

### 4.6 Power and the gate

Seven single-0° events, 4 456 real cells, were the whole 0° stratum; the pre-registered gate of doc 05 was pooled
hard-population AP, which the tilted and cosmic cells dominate and where charge alone reaches 0.947 — a ceiling with no
room for the FM to show anything even if it could. The 0° stratum was never a pre-registered endpoint.

## 5. The missing link

Deghosting an isochronous slab is a **cross-view association** problem. It is solvable by two kinds of information:

- **(a) distinguishability along the track** — charge texture (Landau, delta rays) and landmarks (kinks, vertices,
  Michels, crossings) — exploited as *sequence alignment* of the three views under *per-wire charge conservation*
  (source 2), with per-view descriptors (source 3) giving each landmark a signature that can be matched across views;
- **(b) a continuity / straightness prior over in-slice adjacency** for the featureless stretches between landmarks,
  anchored by the global geometry of the region (§3.3).

The doc 05 chain has neither: the simulation removed (a) by construction (§3.1, §3.2); the legacy solver sums the per-wire
rows away (§4.4); the graph cannot express (b) (§4.1); the message passing has no conservation channel (§4.2); and the
loss asks a per-cell question that has no per-cell answer (§4.3). The doc 05 NO-GO is a verdict on the sample and the
formulation, not on the idea. What the doc 05 result *does* establish stands: with cross-slice continuity available
(θ ≥ 0.5°) the charge factor graph alone solves the ambiguity, and the FM adds nothing *there* — as expected, since
there is no landmark for it to see.

## 6. Redesign

Three stages, each with its own go/no-go, in the order in which the information is needed.

### E1 — a sample with texture and landmarks (simulation)

New events 101 and up, generated by a Python depo writer (`wirecell.gen` depo npz) fed through `DepoFileSource` —
already the input path of `wcfm/img.jsonnet:327` — with a sim job forked from `wct-sim-iso-track-nf-sp.jsonnet`
(new file; no production config touched; events 1–100 are never rewritten):

1. per-step charge drawn from a Landau-like distribution (MPV 5 000 e/mm, CV ≈ 0.2 over 2 cm, tailed);
2. delta rays: a few per metre, 1–10 cm, random direction in 3-D, so that some are in-plane and some drift out of the slab;
3. landmark topologies at θ = 0° and 0.5°: stopping muon + Michel, a kinked track, a two-track vertex, and two-track
   crossings at controlled angles;
4. ≥ 20 single-track 0° events at random φ_yz, so that the 0° stratum has power on its own.

**Go for E1 alone** (no model needed): on the 0° slabs the three-view consistency statistic of §3.1 rises from ≈ prior
to > 2× prior (the 0.5° value it already has), and a 1-D alignment probe (cross-correlation / DTW of q_U(u), q_V(v),
q_W(w)) recovers the u↔v↔w correspondence of single tracks to within one 4-wire cell over ≥ 80 % of the length. If the
texture is there and the alignment recovers the line, source 2 works and the rest is engineering; if not, no model
will find what the data does not carry.

### E2 — a graph and a model that can express conservation and continuity

Graph (extension of `gnn_dataset.py`): keep the factor edges; add **in-slice blob–blob adjacency** (cells that touch in
y–z) and **wire–wire adjacency** (channel ±1, same plane and slice; wire *index* on wrapped planes); define the
**slab** (the slices of one connected activity region) as the unit of training and evaluation.

Model (`gnn_train.py` successor): an unrolled charge-conservation channel — every layer computes the wire residual
r_w = q_w − Σ_{b ∋ w} p_b q̂_b from the current keep probabilities p_b and a predicted cell charge q̂_b, and passes r_w
back to the cells (a deep-unfolded non-negative L1 / ISTA with learned steps); the blob update also aggregates over its
in-slice neighbours. This makes charge matching an explicit computation and continuity representable.

Loss: BCE + a **projection loss** (the predicted support's projections must reproduce the three wire-charge sequences)
+ a **continuity term** over in-slice adjacency (penalise isolated kept cells and breaks). The output is a configuration
of the slab, not a marginal.

Controls that the learned model must beat on the 0° stratum before a stage is built: (i) the geometric rule of §3.3;
(ii) a path search scored by the prototype dQ/dx-fit residual of §4.5 restricted to the slab (the classical answer:
"find the path whose projections fit"); (iii) the legacy solve with *wires* instead of measures as rows and a graph-TV
continuity penalty (the smallest change to the production solver).

**Go for E2**: on E1's single-track 0° slabs charge recall ≥ 0.9 at ghost fraction ≤ 0.2 (the rule already gets 0.95
on the well-formed slabs, so the bar is "solve the slabs the rule cannot, without losing the ones it can"), and on the
non-isochronous strata no regression against the doc 05 charge GNN.

### E3 — the FM re-test, on landmarks, with the 0° stratum as the endpoint

Only after E1 and E2: the FM arm vs the charge arm on the landmark topologies of E1, pre-registered endpoint = charge
recall at fixed ghost fraction on the 0° stratum (≥ 20 events), secondary = the same on crossings, kinks and Michels.
The FM's job in this design is precise and testable: give the pixels around a landmark a signature that matches across
views, so that the association is pinned there and the continuity prior carries it along the featureless stretches.
Pooled hard-population AP is not used again as a gate.

### Stage decision

Doc 05 §9.5 still holds for the non-isochronous regime: a charge-only sub-blob stage on the factor graph. The ISO stage
is a *different component* — a slab-level global solve with learned unaries — behind its own default-OFF knob, triggered
by the doc 02 isochronous criteria (a slab whose activity spans ≥ N channels in ≤ M slices). Neither should be built
before E1 has shown that the texture is there and E2 that the model can use it.

## 7. What carries over from doc 05, unchanged

- The fixed truth tiers (`_sub4f`), the dataset builder, the physics metrics and the y–z views.
- The finding that cross-slice continuity plus per-wire factor structure solves θ ≥ 0.5° (charge recall 0.85–1.00).
- The BlobDepoFill fix (toolkit `1e6b2905`).
- The negative FM result *on this sample*, now with its cause.

## 8. Files

- `scripts/iso_slab_probe.py` — this note's probe (Repro block); `pitch_coords`, `components`, `line_rule` are the
  geometric rule of §3.3.
- `docs/06_tables/slab_census.md`, `consistency.md`, `fm_similarity.md`, `gnn_by_end_distance.md`, `line_rule.md`,
  `measures.md` — the full tables (all 13 + 5 anode-events).
- `docs/06_tables/view-000001_<evt>-anode<N>.png` — truth / GNN keep set / rule keep set of the busiest slice of every
  single-0° anode-event.

Code cited: `wcfm/gen_iso_tracks.py:39`; toolkit `gen/src/TrackDepos.cxx:75-103`, `gen/src/Drifter.cxx:169-174`,
`gen/src/GaussianDiffusion.cxx:185-200`, `img/src/GeomClusteringUtil.cxx:41-105`, `img/src/BlobGrouping.cxx:131-166`,
`img/src/CSGraph.cxx:17-31,108-134,144-149`, `img/src/ChargeSolving.cxx:54-131`, `util/src/LassoModel.cxx:221`,
`clus/src/TrackFitting.cxx:8777`; prototype `wire-cell/pid/src/PR3DCluster_dQ_dx_fit.h:3-140,368-1045,866-921`,
`wire-cell/pid/src/NeutrinoID_proto_vertex.h:1410-1462`; `wcfm/scripts/gnn_train.py:236`, `wcfm/scripts/gnn_dataset.py:134`.

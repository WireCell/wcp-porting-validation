# Steiner terminal determination: what the algorithm does, where it is uBooNE-shaped, and a redesign

Continues [28_steiner-terminal-charge-pdvd-vs-sbnd.md](28_steiner-terminal-charge-pdvd-vs-sbnd.md)
(which decomposed the 500-vs-4000 e floor into a wire-crossing geometric
mismatch) and answers doc 26 §8 item 1, which sent 039349/14 to "a separate
Steiner-terminals campaign". Doc 26 §8 also ordered over-clustering first and
Steiner terminals after; the owner has opened this thread directly, which
supersedes that ordering. Doc 26's *other* case, 039349/53, was closed by
doc 27 as a stale-geometry face swap, not over-clustering.

**Scope: this round changes no code and no config.** It examines the current
algorithm, designs an alternative, and reports an initial feasibility study.
No A/B gate is owed because nothing that can alter reconstruction output was
touched.

---

**TL;DR.** Two independent findings, and they point in different directions.

1. **On the flagship event the terminal criterion is not the binding
   constraint** (§4). In the 111 cm below V that has no Steiner coverage, the
   sampler places 721 3-D points, **420 of them pass the terminal candidacy
   gate** at PDVD's 500 e floor, and they occupy **263 distinct blobs**. Terminal
   finding runs per blob, so Phase 1 must emit **≥263 terminals** there. One
   survives into the tree. The control half of the same track has **72**
   candidate-bearing blobs and keeps **40** terminals. So the filters between
   Phase 1 and the tree keep **56 %** of terminals on the covered half and
   **0.4 %** on the starved half. **Whatever redesign of the charge criterion we
   ship, it cannot fix this event** — the terminals are already being made and
   then thrown away.
2. **A separate, previously unrecorded defect** (§5): along that same stretch
   the V-plane per-point charge is **zero on 98.6 % of the sampled points while
   the 2-D (wire, slice) charge map holds real charge — median 4979 e — at the
   point's own cell**, for 677 of 721 points. U and W match the map exactly, so
   this is not a face-attribution error. Mechanism not established.

Both reproduce unchanged on doc 27's fresh, fully self-consistent v7 arm, so
neither is an artifact of the v6/v7 mixing doc 27 found.

The redesign (§6) therefore rests on doc 28's *population* evidence, not on
039349/14. Its core idea — combine nearby wires over a fixed **physical**
aperture instead of reading one snapped wire — is measured in §7 and works as
intended: it removes essentially all of the peak-vs-losing candidate asymmetry
that doc 28 identified as PDVD's dominant mechanism (losing/peak pass-rate ratio
0.77 → 0.92-1.00 on PDVD, 0.68 → 0.93-1.00 on SBND).

---

## 0. Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd

# Sections 4 and 5 -- stage attribution and the V-plane defect.
# d27fresh is doc 27's self-consistent v7 arm (imaging+clustering+PR on one
# geometry); d25r13fix is the older mixed arm, kept as a cross-check.
python3 docs/nf_sp_img_clus/scripts/steiner_terminal_attribution.py work/039349_14_d27fresh
python3 docs/nf_sp_img_clus/scripts/steiner_terminal_attribution.py work/039349_14_d25r13fix

# Section 7 -- aperture feasibility, PDVD and the SBND control.
python3 docs/nf_sp_img_clus/scripts/steiner_aperture_feasibility.py \
    work/039252_2_stm1/pctree-evt298595.tar.gz 500
python3 docs/nf_sp_img_clus/scripts/steiner_aperture_feasibility.py \
    /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-dbg25a-d97off/ql_evt16/pctree-evt16.tar.gz 4000
```

The Steiner coverage and gap metric of §4 are section 6 of that same script.
They come from the calib dump's `steiner` section, which carries per-cluster
`x/y/z` **and** `flag_terminal`. The cluster is resolved by **ownership of the
control region**, never by a hardcoded id — re-clustering renumbers it (36 on
`d25r13fix`, 34 on `d27fresh`), and a hardcoded id would silently select nothing.

Four traps that each produced a wrong number first; all four are commented in
`steiner_terminal_attribution.py` and enforced by assertions where possible:

- Compare against PR-log / calib coordinates using **`x_t0cor`**, not `x`. With
  `x` the nearest sampled point to V is 16 cm away and every selection is empty.
  `x_t0cor` is meaningless (order 1e9) for points whose cluster has no t0, so
  never min/max it globally.
- The flat `live/.../namedpcs/3d` cloud is a **blob-ordered concatenation**:
  `np.repeat(arange(n_blobs), scalar/npoints)` gives exact blob attribution.
  `sum(npoints) == n_points` is the assertion that this still holds.
- The Bee `clustering-global` layer has the same point *count* as the 3-D cloud
  but **not the same order** (max |y_bee·10 − y_pc| = 5266 mm). Select
  geometrically in Bee's own (y, z) cm coordinates; never index one with the
  other's mask.
- Identify the `ctpc_a<A>f<F>p<P>` dataset by **charge-matching** on points with
  known nonzero charge, not by decoding `wpid` by eye. The by-eye guess gave
  `a2f1` and 0/140 agreement; the truth is `a4f0` at 194/194.
  `slice_index = (t/500)` floored to a multiple of 4. (The two agree once
  decoded properly: `wpid = layer | face<<3 | apa<<4`,
  `iface/src/WirePlaneId.cxx:5-7,35-36`, so 71 → apa 4, face 0.)

---

## 1. What the algorithm does today

`Steiner::Grapher::create_steiner_tree` (`clus/src/SteinerGrapher.cxx:22-143`),
called from `CreateSteinerGraph.cxx:283` as

```cpp
sg.create_steiner_tree(src, path_point_indices, "ctpc_ref_pid", "steiner_graph", false, "steiner_pc");
```

so `disable_dead_mix_cell = false` — which selects the second branch of
`calc_charge_wcp` throughout. Five phases:

| phase | what | source |
|---|---|---|
| 1 | `find_steiner_terminals` — per-blob charge-peak finding | `:38`, `:599-622` |
| 2 | `filter_by_reference_cluster` — drop terminals not contained in the reference cluster's wire ranges | `:50-59`, `:149-206` |
| 3 | `filter_by_path_constraints` — drop terminals far from the supplied path | `:65-70`, `:230-319` |
| 4 | extreme points inserted **unconditionally**, after both filters | `:75-79` |
| 5 | `create_enhanced_steiner_graph` — PAAL Voronoi tree over the survivors | `:111-113` |

### 1.1 Terminals are found one blob at a time

```cpp
for (const auto& [blob_node_idx, point_indices] : cell_points_map) {
    auto blob_peaks = find_peak_point_indices(point_indices, graph_name, disable_dead_mix_cell);
    steiner_terminals.insert(blob_peaks.begin(), blob_peaks.end());
}
```
(`SteinerGrapher.cxx:611-616`; prototype `PR3DCluster_steiner.h:719-726`.)

Each blob is an independent peak-finding universe, so **every blob holding at
least one candidate contributes at least one terminal**. That property is what
makes §4's measurement possible.

### 1.2 The charge quantity

`Cluster::calc_charge_wcp` (`Facade_Cluster.cxx:1031-1112`; prototype
`PR3DCluster_steiner.h:955-1025`) reads, per plane, the charge of the **single
wire nearest the point**, stamped onto the point at sampling time by
`BlobSampler` (`BlobSampler.cxx:315-370` → `ucharge_val`/`vcharge_val`/
`wcharge_val`). With `disable_dead_mix_cell = false`:

```
flag_p  = (charge_p > cut) || (charge_p == 0)      // zero passes: no signal is not held against you
charge  = sqrt( sum(charge_p^2 over planes with charge_p != 0) / n_nonzero ),  0 unless n_nonzero > 1
quality = flag_u && flag_v && flag_w
```

### 1.3 Candidacy and local-maximum suppression

```cpp
const double charge_threshold = m_config.terminal_charge_threshold;   // 4000 default
auto [charge_quality, charge] = m_cluster.calc_charge_wcp(point_idx, charge_threshold, disable_dead_mix_cell);
if (charge > charge_threshold && charge_quality) candidates_set.insert({charge, point_idx});
```
(`SteinerGrapher.cxx:432-442`.) Note the same cut is applied **twice** — once
per plane inside `calc_charge_wcp`, once to the plane-RMS.

Candidates are then walked in descending charge order and suppressed against
their **1-hop** graph neighbours (`nlevel = 1`, `SteinerGrapher.h:190,193`), and
directly edge-connected surviving peaks are merged by connected components,
keeping the peak nearest each component's centroid (`:459-593`).

One subtlety worth recording: `map_index_charge` is built from **one blob's**
points while the BFS runs over the **whole cluster graph**, so cross-blob
neighbours are silently skipped by the two
`map_index_charge.find(...) == end() continue;` guards (`:496`, `:513`).
"Locally highest" therefore means *highest among graph-adjacent points of the
same blob*. Ties never veto (only strict `<` clears `flag_insert`). Both match
the prototype (`PR3DCluster_steiner.h:820`, `:834`).

---

## 2. Where the algorithm is uBooNE-shaped

The philosophy — *the backbone is where the charge is locally highest* — is
sound and detector-independent. Four implementation choices are not.

| # | choice | why it is uBooNE-shaped | PDVD number |
|---|---|---|---|
| 1 | **absolute floor of 4000 e**, per plane *and* on the RMS | an electron count tuned to 3 mm pitch; charge per wire scales with pitch, and the absolute scale also depends on a charge calibration PDVD does not yet have (doc 28 §4.2) | PDVD runs 500 e; W-plane per-point median ~1400 e (doc 25 §13.4 item 8) |
| 2 | **AND over exactly three planes** | with uBooNE's symmetric ±60°/0° *and equal* 3 mm pitch, at most one plane is unlucky for any track direction; PDVD keeps the angles but not the pitches | U/V 7.65 mm, W 5.10 mm — 2.55× and 1.70× coarser |
| 3 | **"local" = 1 graph hop** | a hop's physical size depends on point density, which depends on the crossing ambiguity | PDVD emits a median of 4 points per (U,V) crossing vs SBND's 1, so a hop is *shorter* on PDVD exactly where the sampling is worst |
| 4 | **single snapped wire per plane** | one wire is a fair estimate of the local charge only when the pitch is fine compared with the deposit's transverse spread | transverse diffusion is 2.01 mm = **0.39 of the W pitch**, 0.26 of U/V (doc 25 §8): the charge lands on one or two wires and the observable is adjacent-wire *sharing* |

Choices 2 and 4 are the ones the owner's brief targets, and the toolkit itself
already documents the geometric assumption behind them. `BlobSampler.cxx:817-824`,
on the mid-plane pitch correction used to place every sampled point:

> This gives a relative pitch distance measured in the "mid" view that is half
> the distance between crossing point of the 0-rays and the 1-rays in the other
> two views. In general, this is **NOT** the same as the magnitude of "adjust" /
> "ac" vector above as that diagonal of the min/max parallelogram is not
> necessarily parallel to the pitch direction in the third, "mid" view. The two
> directions are **accidentally coincident for symmetric wire patterns like in
> MicroBooNE**.

That is an in-repo, source-level statement that the geometry the sampler assumes
is exact for MicroBooNE and approximate elsewhere — stronger evidence for the
owner's hypothesis than any measurement added here.

### 2.1 `ChargeStepped` is not already this

The tree contains an unused second sampler, `ChargeStepped`
(`BlobSampler.cxx:914-1272`), a port of WCP's `calc_sampling_points()`. Neither
detector uses it: `protodunevd/clus.jsonnet:197` and `sbnd/clus.jsonnet:206`
both select `stepped`. It *does* filter candidate wires by charge before forming
a crossing (thresholds 4000/4000/4000, `:938-940`), which would change the
ambiguity population — worth a config-only experiment on its own. But it is
**not** the "combine nearby wires" idea: at `BlobSampler.cxx:1214` it still
resolves the third plane to one wire,

```cpp
coordinate_t cother{smid.layer, static_cast<int>(std::round(pitch_relative))};
```

and reads that single wire's charge. So the estimator this doc proposes to
change is unchanged in `ChargeStepped`.

---

## 3. What has already been excluded

- **The two known port divergences are already on in PDVD.** `steiner_terminal_wire_tol=1`
  (the prototype's ±1 wire of slack in the terminal filter, doc pr/29 D1) and
  `steiner_terminal_adjacent_slice=true` (D12's dead t±1 branch) are set in
  `wcp-porting-img/pdvd/wct-pr-perevt.jsonnet:373-374`, committed `a61fa097`
  on 2026-09-02 15:56 — before every arm used here. SBND still runs both **off**.
  pr/29 measured that with both off the toolkit discards 47.7 % of all Steiner
  terminals; PDVD is already paying the reduced 20.0 % version, and §4's drop is
  on top of that.
- **Stale geometry is not the explanation.** Doc 27 found that arms built from a
  v6 point tree and run against v7 anodes put clusters on anodes 2/3/6/7 one
  face height off in y. Cluster 36/34 here is on **anode 4, face 0**, which doc 27
  lists as unaffected — and every number in §4 and §5 was re-measured on doc 27's
  fresh self-consistent `d27fresh` arm and is unchanged.
- **It is not a cluster-boundary artifact.** The charge below V belongs to the
  same cluster: 735 of 795 Bee points (92 %) carry the track's own
  `real_cluster_id`, and 100 % do on the control half.

---

## 4. The stage attribution: terminal selection is not what starves 039349/14

Cluster 34 (`d27fresh`; 36 on `d25r13fix`) is one straight cosmic. Doc 26 §7.5
established that its Steiner cloud stops at the vertex V (x 273, z 87): coverage
on the half above, none along the 111 cm from V to A. Both halves of the *same
track* in the *same cluster* therefore form a controlled comparison.

Region selection is geometric: points within 3 cm of the V→A line (below) or the
V→U line (above), endpoints excluded.

| | **below V** (starved) | **above V** (control) |
|---|---|---|
| Bee points, and share owned by this cluster | 795, **92 %** | 180, **100 %** |
| sampled 3-D points | **721** | 195 |
| pass `calc_charge_wcp` + 500 e floor | **420 (58.3 %)** | 109 (55.9 %) |
| distinct blobs holding ≥1 candidate | **263** | 72 |
| ⇒ terminals Phase 1 must emit (lower bound) | **≥263** | ≥72 |
| terminals actually in the tree | **1** | 40 |
| steiner points actually in the tree | **5** | 258 |
| **survival, Phase 1 → tree** | **0.4 %** | **56 %** |
| largest steiner-free gap along the line | **108.5 cm** | 65.6 cm |

(`d25r13fix`, the older mixed arm, gives the same picture: 738 points, 419
candidates in 259 blobs, 1 terminal, 111.1 cm gap.)

The starved half has **3.7× more** candidate-bearing blobs than the half that
works, and ends with 1 terminal instead of ≥263. **The charge floor, the charge
estimator and the peak finder all did their job here.** The loss is entirely
between Phase 1 and the tree.

That is a negative result for the redesign in §6 as a fix *for this event*, and
it is stated first deliberately: doc 30 shipped a knob on a plausible-but-unmeasured
causal chain and it turned out to fire 93 times and change nothing. The
surviving suspects, in the order they run:

- **`filter_by_reference_cluster`** (`:149-206`). Tests each terminal for
  containment in the reference cluster's blob wire ranges via
  `check_wire_ranges_match`, which requires **all three** planes to match. §5's
  V-plane defect makes the V information along exactly this stretch unreliable,
  which is the one place the two findings could couple — worth testing first.
- **`filter_by_path_constraints`** (`:230-319`). Removes a terminal whose 2-D
  distance to the path is < 1.8 cm on **any two of three** planes while its 3-D
  distance exceeds 6 cm. Self-reinforcing: the path is built from the previous
  skeleton, so a cloud that already stops at V supplies a path that stops at V.
- **The retile** (`ImproveCluster_2` / `steiner_refresh`), which can hand
  `create_steiner_tree` a different cluster than the one measured here.

**Round 1 of the next round is a per-phase terminal count** — a log line
reporting `steiner_terminals.size()` after each of phases 1, 2, 3 and 4, which
is instrumentation, not an algorithm. Success criterion: it names the phase that
takes 263 terminals to 1.

---

## 5. A separate defect: V-plane charge present in the 2-D map, zero on the point

Along the same stretch, cross-referencing each sampled point's stored per-plane
charge against the 2-D `ctpc_a4f0p*` (wire, slice) charge map from the same dump:

| plane | below V (n=721) | above V (n=195) |
|---|---|---|
| stored charge nonzero | **1.4 %** | 99.5 % |
| stored value == map value at the point's own cell | **0 / 721** | 194 / 195 |
| **stored 0 while the map holds charge there** | **677 (median 4979 e)** | 1 |
| `charge_unc == 0` (not the 1e10 dead sentinel) | **98.6 %** | 0.5 % |

U (709/721 exact) and W (709/721) match the map, so the point's (apa, face) and
wire indices are right — this is specific to V, and specific to this stretch.
The uncertainty is 0 rather than the dead sentinel, so `calc_charge_wcp` treats
V as "no signal, don't hold it against the point" (§1.2) rather than as dead:
the point keeps a healthy U/W RMS and still passes, which is why §4's candidate
count is high despite a whole plane being blank. The defect is silent by
construction.

**Mechanism NOT established.** The attractive hypothesis is PDVD's wrapped
induction strips: 1568 channels carry two segments whose lengths sum to exactly
1720.04 mm, split at the CRU boundary at y = ±1685 mm, U and V only — W is never
wrapped — and this track runs y −1189 → −1678, straight at that boundary, on the
plane that is wrapped. That is circumstantial. The discriminating test, which
this round does not do: are the affected V channels the two-segment ones, and
was the `ctpc` row that holds the charge written from the *other* face's
activity?

A converse class also exists, at the few-percent level: points whose stored
charge is nonzero but which do not match the map at their own cell (PDVD U,
10296 of 137264 nonzero points event-wide; SBND 0 of 23182). The expected
explanation — not verified here — is that `PointTreeBuilding::add_ctpc` drops
rows whose uncertainty exceeds the dead threshold (`PointTreeBuilding.cxx:296-299`)
while `BlobSampler` stores the value regardless, making the map a subset of what
the sampler saw. Whatever it is, it is the opposite direction from the V finding
and does not explain it. The apparent PDVD/SBND contrast here is **not**
decomposed and should not be read as a result: the event-wide counts were never
split into "absent from the map" versus "present but different", and SBND is a
single-anode dump where the (apa, face) attribution is trivial, so the two sides
are not measured the same way.

---

## 6. The redesign

The reframe first, because it is the whole argument:

> The stated philosophy is *"the charge is highest **locally**"*. The code
> implements *"above 4000 e globally **and** highest locally"*. The absolute
> term is the detector-dependent half, and it is the half that does not
> transfer.

Six elements, in descending confidence. Blast radius is deliberately confined to
`Steiner::Grapher::Config` and `calc_charge_wcp`'s estimator — **not**
`BlobSampler::stepped`, which every detector binds (SBND, uBooNE, PDHD) and
which doc 28 showed is not where the win is: the crossing ambiguity is
structural to PDVD's pitch, and the v7-uvwfit geometry made it slightly *worse*,
not better.

**(a) Aperture-matched charge — the owner's "combine nearby wires".** Replace
the single snapped wire with a charge integral over a window whose half-width is
a fixed **physical** distance, so the same configured number means the same thing
at 3 mm and 7.65 mm pitch. This is the element §7 measures.

The primitive already exists and is production-exercised:
`Grouping::get_ave_charge(point, apa, face, pind, radius)`
(`Facade_Grouping.cxx:640-670`, used by `NeutrinoVertexFinder.cxx:414,469` and,
summed over planes, `get_ave_3d_charge` in `NeutrinoShowerClustering.cxx:2945`).
Four caveats to settle before adopting it, none fatal:

- it returns the **mean** over in-radius cells, not the sum. A mean is roughly
  pitch-independent by construction, which is what a *relative* criterion wants;
  a sum is the dQ/dx-like quantity. §7 measures both — say which one the
  criterion uses rather than inheriting the choice by accident;
- the radius is Euclidean in (x_drift, y_pitch), so a circular aperture spans
  ~2.6× more slices than wires on PDVD's 2.96 mm × 7.65 mm cell. That may be
  desirable; it should be a stated choice;
- it re-fetches `local_pcs.at(ds_name).get("charge")` on **every call**,
  uncached, while the `kd2d` beside it is memoized. At ~160 k points × 3 planes
  that is ~0.5 M map lookups per cluster pass. The last Steiner-adjacent round
  was a 27-minute perf problem (doc 25 §13.11); hoist the array fetch, or use
  `Grouping::wire_charge_row(apa, face, plane, time_slice)`
  (`Facade_Grouping.cxx:1026-1035`), which returns a whole slice row;
- it returns `0.0` when the dataset is missing, which is indistinguishable from
  "no charge" — and §1.2 shows `calc_charge_wcp` then reads zero as *"don't hold
  it against the point"*. Given §5, that conflation is not hypothetical.

**(b) A relative floor, not no floor.** Dropping the absolute cut without a
replacement puts terminals on noise. `{u,v,w}charge_unc` is already stored per
point, so an SNR floor (`charge/unc`), or a floor set as a percentile of the
cluster's own charge distribution, is detector-independent by construction where
4000 e and 500 e are both calibration-coupled. This is what makes (a)
defensible rather than reckless, and it is what makes one configured number
correct on uBooNE, SBND and PDVD at once.

**(c) Combine the planes instead of ANDing them.** `calc_charge_wcp` ANDs three
fixed cuts and returns an RMS that *rewards* a single loud plane. With
`charge_unc` available, a χ² that the three planes see a **common** charge
penalises exactly the "losing" candidate doc 28 found PDVD manufactures 4× more
of — one plane's charge without the others' — and dead handling (`unc > 1e10`)
falls out of the weighting instead of `disable_dead_mix_cell`'s two ad-hoc
branches (whose divergence from the prototype is pr/29 D2, still open).

**(d) Neighbourhood in cm, not hops**, for the local-max step, removing coupling
#3 of §2. The cluster k-d tree makes this available at no new cost.

**(e) Listed, but explicitly not load-bearing: direction-aware normalisation.**
Dividing each plane's charge by the expected path length in that plane's pitch
cell, `Q_p / L_p(θ)`, is the principled way to make three planes comparable. It
needs a local track direction from PCA, which is density-dependent and
**undefined on the half of the track that has no points** — i.e. undefined
exactly where the symptom is. Keep it in the design, do not build on it.

**(f) A backstop, labelled as one.** A coverage guarantee — at least one terminal
per X cm along the cluster's principal axis, taking the local best even if it
misses the floor — would bound the failure mode directly. It is make-it-work,
not mechanism; it belongs last and only if (a)-(d) leave a residue.

### 6.1 The metric to judge any of this by

**Largest terminal-free gap along the cluster's principal axis.** It is what was
actually wrong on 039349/14 (108.5 cm), it is detector-comparable, and it is
computable offline from the calib dump's `steiner`/`flag_terminal` arrays. The
AND-gate pass rate is a poor substitute: doc 28 moved it 17.4 % → 20.2 % with the
v7 geometry and the symptom did not move at all.

---

## 7. Feasibility: does combining nearby wires do what it should?

Measured on doc 28's PDVD event and its SBND control, since 039349/14 does not
exercise the estimator (§4). Aperture half-width **10 mm, physical**, converted
per detector and per plane from its own pitch and slice width: PDVD ±2 wires
(U/V/W) and ±4 slices of 2.96 mm; SBND ±4 wires and ±4 slices of 3.20 mm. Charge
is read from the `ctpc_*` maps. Candidacy uses the exact `calc_charge_wcp`
semantics; the aperture rows use a floor set as a fraction of that event's own
median estimate, which is element (b).

The single-wire lookup reproduces the stored `*charge_val` exactly — SBND
23182/23182 (U), 22840/22840 (V), 22009/22009 (W) — which validates the whole
2-D cross-reference before any conclusion rests on it.

| | PDVD 039252/2 (160930 pts) | SBND evt 16 (23930 pts) |
|---|---|---|
| median candidates per (U,V) crossing | **4** | **1** |
| crossings that are ambiguous | **83.6 %** | 21.0 % |
| points that are a losing candidate | **80.4 %** | 21.8 % |

reproducing doc 28 §4.3 exactly. Then, candidacy split by peak vs losing:

| estimator | floor [e] | PDVD cand / peak / losing / **ratio** | SBND cand / peak / losing / **ratio** |
|---|---|---|---|
| single wire (production) | 500 / 4000 | 0.611 / 0.750 / 0.577 / **0.77** | 0.241 / 0.259 / 0.176 / **0.68** |
| aperture SUM, rel 0.2 | 45141 / 46982 | 0.898 / 0.914 / 0.895 / **0.98** | 0.980 / 0.989 / 0.948 / **0.96** |
| aperture SUM, rel 0.5 | 112852 / 117455 | 0.741 / 0.793 / 0.729 / **0.92** | 0.845 / 0.858 / 0.798 / **0.93** |
| aperture MEAN, rel 0.2 | 1895 / 1349 | 0.925 / 0.923 / 0.925 / **1.00** | 1.000 / 1.000 / 0.999 / **1.00** |
| aperture MEAN, rel 0.5 | 4737 / 3371 | 0.768 / 0.793 / 0.762 / **0.96** | 0.930 / 0.934 / 0.917 / **0.98** |

`ratio` = losing/peak pass rate. **1.00 means a point's candidacy no longer
depends on whether it won its wire crossing.**

**The idea works, and for the stated reason.** A physical aperture removes almost
all of the peak-vs-losing asymmetry on **both** detectors (PDVD 0.77 → 0.92-1.00,
SBND 0.68 → 0.93-1.00), because both members of an ambiguous group sit inside
one aperture of the same charge. That is the mechanism doc 28 identified as
PDVD's dominant one, addressed directly rather than compensated for by lowering
a threshold.

Three honest qualifications:

1. **Removing the asymmetry is not the same as picking the right point.** Making
   the losing candidate pass at the peak's rate means candidacy stops
   *discriminating* on crossing ambiguity — it does not resolve the ambiguity.
   The discrimination has to come back from the local-max step (§6d) and from
   plane consistency (§6c), and neither is measured here. A criterion that
   admits everything is not obviously better than one that admits the wrong 60 %.
   This is the main open question for round 2.
2. **The floors are not tuned.** The relative floors here are illustrative; at
   rel 0.2 nearly everything passes on both detectors. The operating point has to
   be chosen against §6.1's gap metric, not against a pass rate.
3. **The aperture is a whole-event average here**, applied uniformly. It does not
   test the case §4 is about, and it cannot: on 039349/14 the candidates already
   exist.

---

## 8. What this round did not do, and what round 2 is

Not done: no C++ change, no config change, no gate (none is owed). The
mechanism of §5 is not established. §7's aperture is measured for its effect on
the doc 28 ambiguity population only, not end-to-end through the tree.

Round 2, in order:

1. **The per-phase terminal census on 039349/14** (§4). A log line, not an
   algorithm. It names which of `filter_by_reference_cluster`,
   `filter_by_path_constraints` or the retile takes 263 terminals to 1. Nothing
   in §6 should be built before this answers, because §4 shows §6 cannot fix
   this event.
2. **The §5 discriminating test** — are the zero-V channels the wrapped
   two-segment ones, and which face's activity wrote the `ctpc` row. Independent
   of 1; likely a defect in its own right.
3. **Only then** the §6 redesign, gated on §6.1's gap metric across a manifest,
   with SBND and uBooNE byte-identity gates because `calc_charge_wcp` and
   `Steiner::Grapher` are shared. `steiner_terminal_wire_tol` /
   `steiner_terminal_adjacent_slice` being off in SBND but on in PDVD (§3) is a
   separate owner call that this campaign should surface, not silently resolve.

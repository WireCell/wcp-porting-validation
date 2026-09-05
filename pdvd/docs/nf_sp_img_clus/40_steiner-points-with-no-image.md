# doc pdvd/40 — Steiner-graph points where there is no 3D image

**Status.** Diagnosis only. **No code and no config is changed by this
document**; the production PDVD chain is exactly what it was. Two named
points are traced to two different causes, one of which is a knob the owner
flipped ON on 2026-09-04 and which this document does **not** flip back —
§10 states why that decision is not free and belongs to the owner.

**Question (owner, 2026-09-04), on the doc pdvd/39 round-2 Bee set for
039252/2 = art event 298595
(`https://www.phy.bnl.gov/twister/bee/set/01f28648-c628-4f70-89bd-031fcccd865c/`):**

> for some of the Steiner-Graph, there are some points that are strange, see,
> (x, y, z) = (39.0, 75.7, 201.8), or (x, y, z) = (275.5, 12.1, 8.8). These are
> clearly no image associated with them. I wonder why these were part of
> Steiner Graph? … Is it due to our change in CTPC flip? Or something else?

**Short answer.** Both points are real: they are Steiner points 25.2 cm and
17.7 cm from the nearest live 3D point *of any cluster in the event*, and the
two layers are in the same coordinate frame, so this is not a display offset
(§2). They have **different causes**:

| | point | cluster | cause |
|---|---|---|---|
| **P1** | (39.0, 75.7, 201.8) | 119 | **yes — the ctpc flip.** `ctpc_aniso_metric`, ON in PDVD production since the owner's 2026-09-04 flip. Turning it off deletes P1 and all 618 fabricated points of cluster 119, and every point beyond 30 cm in the event (§5). |
| **P2** | (275.5, 12.1, 8.8) | 84 | **no.** Bit-for-bit unchanged with the metric off. It is a 52.9 cm Steiner bridge between two disconnected live components of one cluster — the uncapped-bridge residual doc pdvd/39 round 2 §14 named and did not fix (§7). |

Two further results that qualify the first one:

* Across **21 events** the metric accounts for only **8 %** of the
  fabricated points beyond 3 cm (5.06 % → 4.58 % of all Steiner points) and
  **25 %** of those beyond 30 cm. Event 298595 is an outlier, not the norm
  (§6).
* Turning the metric off on those 21 events **turns over the STM tag set**:
  96 of 127 cluster ids move, for a net 127 → 99. It is not a display-only
  fix (§6, §10).

A third, unrelated defect surfaced in the same layer and is reported here
because it is in the `clustering` layer the owner looks at: **29 clusters /
225 live points are drawn 1 480 km from the detector** (§8).

**Round 2 (§11-§13), answering the owner's follow-up.** P1's cause cannot occur
on SBND — the anisotropic metric is the *identity* there by geometry, not by
configuration. P2's machinery is live on SBND, and was measured: across 67 SBND
events **not one** Steiner point is more than 10 cm from live charge, against
10 255 on PDVD, because PDVD has 15x more multi-component clusters and worse
gaps within them. And the retiler already has its own anti-ghost filter,
`remove_bad_blobs`, which deletes 31 533 blobs on this event and still lets the
column through — **the fix is to strengthen a filter, not to add one**.

---

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd

# the two arms for the demo event: same pctree, same pipeline, one knob apart
./run_pr_evt.sh -unmerge -s d40rep    039252 2
PDVD_PR_TLA="-S ctpc_aniso_metric=false" \
./run_pr_evt.sh -unmerge -s d40aniso0 039252 2

# frame check, probes, void census, far-point groups, chance floor, sentinel-T0
docs/nf_sp_img_clus/scripts/d40_steiner_void_census.py \
    work/039252_2_d40rep work/039252_2_d40aniso0

# the 21-event A/B (base arm d39r2base already exists from doc pdvd/39 round 2)
while read ev; do run=${ev%_*}; evt=${ev#*_}
  PDVD_PR_TLA="-S ctpc_aniso_metric=false" ./run_pr_evt.sh -stm -s d40a0 $run $evt &
done < <(grep -o '^########## [0-9_]*' stm/gates/d39r2_unmerge_gate.txt | awk '{print $2}'); wait
docs/nf_sp_img_clus/scripts/d40_aniso_arm_summary.py <events.txt> d39r2base d40a0
```

```bash
# round 2 (sec 11-12): the cross-detector census and the filter measurement
while read ev; do run=${ev%_*}; evt=${ev#*_}
  ./run_pr_evt.sh -nu -s d40nu $run $evt &          # -nu writes the calib dump
done < <(...21-event manifest...); wait
docs/nf_sp_img_clus/scripts/d40_steiner_void_xdet.py --route-control \
    "PDVD:$PWD/work/*_d40nu" \
    "SBNDncpi0:../sbnd/sbnd_xin/work-ncpi0-doc25d38new/pr_evt*" \
    "SBNDnuecc:../sbnd/sbnd_xin/work-nuecc48-doc25d38new/pr_evt*"

PDVD_LOG_LEVEL=trace ./run_pr_evt.sh -unmerge -s d40trace 039252 2   # remove_bad_blobs counts
grep "blobs removed for apa" work/039252_2_d40trace/wct_pr_039252_2.log
```

Gate records: `pdvd/stm/gates/d40_aniso_ghost_gate.txt` (rounds 1),
`pdvd/stm/gates/d40r2_xdet_gate.txt` (round 2).
Pinned library for every arm: `local/lib/libWireCellClus.so`, 2026-09-04
18:54:31 — older than the first arm and unchanged across all of them, and
byte-identical (`cmp`) to `/home/xqian/tmp/d39r2_libpin/libWireCellClus.so`,
the copy the §6 base arm `d39r2base` ran against.

---

## 1. The two points, located

`steiner_graph` dumps a cluster's `steiner_pc` local point cloud
(`protodunevd/pr.jsonnet`, layer `steiner_graph`, `pcname: 'steiner_pc'`,
`coords: t0cor_coords`). Both named points are in it, to 0.06 cm:

```
probe P1 (39.0, 75.7, 201.8)   nearest steiner pt 0.056 cm  cid=119 at (39.02,75.74,201.84)
probe P2 (275.5, 12.1, 8.8)    nearest steiner pt 0.052 cm  cid= 84 at (275.46,12.14, 8.80)
```

Distance from each to the nearest **live 3D point of any of the event's 493
clusters**: **25.2 cm** (P1) and **17.7 cm** (P2). The owner's reading — no
image there — is correct.

## 2. The premise: the layers share a coordinate frame

Everything below is a distance between `steiner_graph` and `clustering`, so a
frame mismatch (raw drift-x vs `x_t0cor`) would fake the whole result.
`steiner_terminals` is a charge-selected **subset** of the same Steiner cloud,
so its distance to the nearest `clustering` point is the frame probe:

| layer | n | median | p90 | max |
|---|---:|---:|---:|---:|
| `steiner_terminals` | 9 535 | **0.100 cm** | 0.835 | 1.85 |
| `stm` | 53 628 | 0.000 | 0.000 | 0.00 |
| `stm_tagged` | 2 284 | 0.000 | 0.000 | 0.00 |
| `stm_fit` | 4 051 | 0.511 | 2.231 | 30.85 |
| `steiner_graph` | 52 983 | 0.388 | 1.530 | **70.45** |

`stm`/`stm_tagged` are exact subsets of the live cloud (0.000 everywhere) and
the terminals sit on it to a tenth of a millimetre. **The frames agree.** The
70 cm tail on `steiner_graph` is physical content of that layer, not an offset.

(`stm_fit`'s 30.85 cm tail is the fitted trajectory, which is a polyline and is
*expected* to leave the point cloud; it is not part of this investigation.)

## 3. Why these points are visible only now

They are not new. They became **displayable** in doc pdvd/39 round 2, which
re-scoped `stm`/`steiner_graph`/`steiner_terminals` from the STM *verdict* set
to the *fitted* set (`require_pc: 'stm_fit'`). Clusters 119 and 84 are fitted
but not tagged, so before round 2 they were not in the layer at all:

```
arm          zip mtime        d(P1)    d(P2)   nSteiner  >10cm    pct
d39stm2      09-04 17:48    216.341  188.912      5569    179   3.21%   <- verdict-scoped: 119/84 absent
d39lean      09-04 17:49    216.341  188.912      3553    179   5.04%
d39nu        09-04 17:50    216.341  188.912      5569    179   3.21%
d39r2base    09-04 19:02      0.005    9.936     56843   1351   2.38%   <- fit-scoped
d39r2unm     09-04 19:02      0.005    0.005     52983    838   1.58%
d40aniso0    09-04 19:46     25.191    0.005     52366    228   0.44%
```

Arms older than doc 39 round 1 (`d27fresh` … `d38flip20`) carry no
`steiner_graph` layer at all, so they cannot be bisected on this symptom; the
knob A/B in §5 replaces that bisect.

## 4. What the fabricated points look like

Grouped at 3 cm, against the cluster's own live connected components (6 cm
link), in the published arm:

```
cluster 119: 2219 live pts / 2 components ; 618 far steiner pts
   grp 0 n= 416 span=122.4 cm  x[  2.3,124.6] y[74.9,78.4] z[200.3,204.9]  between comps [0,1]  worst 70.4 cm
   grp 1 n= 202 span=106.4 cm  x[ 18.3,124.6] y[74.4,77.9] z[207.2,208.7]  off comp  [1]        worst 67.6 cm
cluster  84: 38599 live pts / 17 components ; 191 far steiner pts
   grp 4 n= 152 span= 52.9 cm  x[268.7,316.6] y[ 8.2,17.6] z[  5.7, 25.9]  between comps [1,6]  worst 36.6 cm
   (+ four groups of 3-14 points, 3.4-9.3 cm span)
cluster  87:  144 live pts /  3 components ;  25 far steiner pts
cluster  79:  331 live pts /  2 components ;   4 far steiner pts
```

Cluster 119's live cloud is two components — `x[-149.3,-4.2]` and
`x[5.2,148.5]`, i.e. one on each side of the cathode — and the 618 fabricated
points form two long columns running along the drift axis at nearly fixed
(y, z). P1 sits on `grp 0`.

## 5. P1: the ctpc anisotropic metric

`ctpc_aniso_metric` is **ON** in the compiled PR config of the published arm:

```
$ grep -o '"ctpc_aniso_metric"[^,}]*' work/039252_2_d39r2unm/.wct-pr_d39r2unm.json
"ctpc_aniso_metric" : true
```

It is set in `pdvd/wct-pr-perevt.jsonnet:3023` (`ctpc_aniso_metric = true`) by
toolkit `38245d18`, *"cfg/protodunevd: retire the good_point_pitch_frac floor,
PDVD PR moves to the anisotropic ctpc metric (doc pdvd/36 sec 11, owner
decision 2026-09-04)"*. The C++ default is still `false`
(`protodunevd/pr.jsonnet:1130`), so this is a PDVD production value, not a
toolkit default. **This is the change the owner suspected, and for P1 the
suspicion is right.**

Single-event A/B, same pctree, same `-unmerge` pipeline, one knob apart:

| | metric ON (`d40rep`) | metric OFF (`d40aniso0`) |
|---|---:|---:|
| Steiner points | 52 983 | 52 366 |
| > 3 cm from any live pt | 1 579 (2.98 %) | **900 (1.72 %)** |
| > 10 cm | 838 (1.58 %) | **228 (0.44 %)** |
| > 30 cm | 401 (0.76 %) | **0 (0.00 %)** |
| worst distance | 70.45 cm | 29.09 cm |
| cluster 119 far points | **618 (22.0 % of its cloud)** | **0** |
| cluster 84 far points | 191 | 191 (identical) |
| cluster 87 far points | 25 | 33 |
| TGM / FC verdict sets | 39 / 33 | identical / identical |
| STM verdict set | 5 | 6 (loses 109, gains 101 and 103) |
| wall, run alone | 38 s | 37 s |

**P1 is gone with the metric off** (nearest Steiner point 25.2 cm away), and so
is cluster 119's entire fabricated content.

`d40rep` is a repeat of the published `d39r2unm` arm and reproduces it
**member-hash identical** (`abtest/hash_archive.py --members`, aggregate
`e728abfe…`), so the PR stage is bit-reproducible on this event and the
differences above are attributable to the knob.

### 5.1 How the metric reaches the Steiner cloud

Traced by call graph, and confirmed only by the A/B above — the individual
edge decisions were **not** instrumented:

1. `ctpc_aniso_metric` changes exactly two Grouping queries,
   `get_closest_points` and `has_closest_point`
   (`clus/src/Facade_Grouping.cxx:699` and `:723`), by scaling the pitch axis
   of the ctpc lattice. The run logs the scale it used:
   `drift_step 2.9615 mm, pitch U/V/W 7.6500/7.6500/5.1000 mm, yscale U/V/W
   0.3871/0.3871/0.5807` — i.e. a query of a given radius reaches **2.58×
   further along the U/V pitch axis** than the isotropic circle it replaces,
   so a test that used to miss now hits. `d36_ctpc_caller_census.py` counts
   **4 external caller files / 10 calls** of the changed query.
2. `Grouping::is_good_point` (`Facade_Grouping.cxx:542`) and
   `is_good_point_wc` (`:569`) are built on `has_closest_point` (`:556`,
   `:583`).
3. `connect_graph_ctpc.cxx` calls `is_good_point` at six sites (`:118`,
   `:218`, `:274`, `:699`, `:735`, `:770`) to decide whether a step may be
   taken. This is the builder of `ctpc_ref_pid`. `connect_graph.cxx`, which
   builds `basic_pid`, calls neither — so the metric acts on the ctpc graph
   only.
4. `ctpc_ref_pid` is used twice in this chain: `ImproveCluster_2` takes a
   whole-cluster shortest path through it (`improvecluster_2.cxx:166`,
   `:176-179`) and hands it to `hack_activity_improved`, which interpolates
   the path at 0.3 cm and writes synthetic activity into all three planes over
   a ±3 slice / ±3 wire disc (`improvecluster_1.cxx:451`, insertion at
   `:562-592`); and `CreateSteinerGraph` rebuilds it on the retiled copy
   (`CreateSteinerGraph.cxx:288`) as the graph the Steiner tree is extracted
   from. (The *other* path `ImproveCluster_2` hacks with,
   `improvecluster_2.cxx:125-128`, runs on `basic_pid` and is not affected.)
5. `make_iblobs_improved` tiles that activity into blobs, the blobs are
   sampled, and those points become `steiner_pc` — the Bee layer.

So a more permissive good-point test lets the path cross a gap, and everything
downstream faithfully manufactures a track along it. The ±3 slice write is
only ≈ ±0.9 cm at PDVD's 0.2962 cm drift step, so the hack does **not** paint
a 122 cm column by itself; the column is the tiling of a path that already ran
through the void.

## 6. The 21-event A/B — the metric is not the whole story

Same construction on the doc pdvd/39 round-2 manifest: one pctree per event,
two `-stm` PR passes over it, only `ctpc_aniso_metric` differing.
(`stm/gates/d40_aniso_ghost_gate.txt`.)

**The pipeline differs from §5.** §5's demo pair runs `-unmerge` (the round-2
chain, which is what the owner's Bee set shows); this table runs `-stm` (the
production chain, which is what `d39r2base` is). So 039252_2 appears twice
with different numbers — 1 579 → 900 fabricated > 3 cm under `-unmerge`,
2 428 → 1 405 under `-stm`. Do not diff one against the other; each pair is
internally controlled.

Both arms ran against the **same** `libWireCellClus.so`: the doc-39 round-2
pin `/home/xqian/tmp/d39r2_libpin/libWireCellClus.so` and today's
`local/lib/libWireCellClus.so` are byte-identical (`cmp`, 421 627 544 bytes,
2026-09-04 18:54:31).

| | metric ON (production) | metric OFF |
|---|---:|---:|
| Steiner points, 21 events | 488 262 | 492 820 |
| fabricated > 3 cm | 24 691 (**5.06 %**) | 22 585 (**4.58 %**) |
| fabricated > 10 cm | 11 394 (2.33 %) | 10 126 (2.05 %) |
| fabricated > 30 cm | 1 655 (0.34 %) | 1 244 (0.25 %) |
| TGM tagged | 388 | 387 (7 ids move) |
| **STM tagged** | **127** | **99 (96 ids move)** |
| FC tagged | 377 | 377 (0 ids move) |

Verdict totals cover **20 of the 21 events**: 039252_2's base arm kept its Bee
zip but not its `wct_pr` log, so its verdicts come from the `d40rep` /
`d40aniso0` pair in §5 instead (TGM and FC identical, STM 5 → 6).

Two things follow, and they point in opposite directions:

* Event 298595 is **not representative**. On it the metric is responsible for
  43 % of the fabricated points beyond 3 cm (1 579 → 900) and 100 % of those
  beyond 30 cm (401 → 0);
  across 21 events it is 8 % and 25 %. **Most fabricated Steiner points in
  PDVD are not caused by this knob.**
* The **STM tag population turns over** when the metric is removed: 96 of the
  127 base ids move in one direction or the other, for a net 127 → 99. The
  count alone ("28 tags lost") badly understates that — most of the tagged set
  is not the same set. TGM moves 7 ids for a net −1; FC does not move at all.
  What this does **not** say is *why*: nothing here shows the fabricated points
  are what moves the STM verdicts (§9). It says only that removing the metric
  is a large, mostly-lateral change to the STM output, not a display-only
  cleanup.

The mean wall in the table's source run (25.1 s vs 48.3 s) is **not a
measurement**: the two batches ran at different concurrency. Run alone on the
same event the two arms are 38 s and 37 s — no material cost either way.

## 7. P2: not the metric — an uncapped bridge in a 17-component cluster

Cluster 84's 191 fabricated points are **bit-identical** with the metric off,
including P2. Its live cloud is 38 599 points in **17 components at a 6 cm
link**, and the group P2 belongs to is a 152-point, 52.9 cm span running
between components 1 and 6, reaching 36.6 cm from any live point.

This is the residual doc pdvd/39 round 2 §14 item 1 already named and left
open: the component-bridging MST in `connect_graph.cxx:20-26, :91` has **no
distance cut**, so once a cluster is disconnected in space the Steiner build
will bridge it however far it must. Round 2's `unmerge_assoc` undoes the
`clustering_isolated` share of that over-merging — and cluster 84 is
un-merged in this very arm — but its 17 components come from the long-range
merge family of docs pdvd/25 and 26, which the un-merge does not address.

Cluster 87 shows the same shape (3 components, bridges of 11–20 cm) and gets
slightly **worse** with the metric off (25 → 33 far points), which is a second
reason the metric is not the single lever here.

## 8. A separate defect in the same Bee set: 29 clusters 1 480 km away

Found while building the kd-tree for §1 — the `clustering` layer's x extent is
`[-1.48074e8, +1.48074e8]` cm.

* **225 live points in 29 clusters**, at |x| = 1.48073–1.48074 × 10⁸ cm.
* No cluster is mixed: each affected cluster moves **entirely**. Sizes 1–20
  points (median 6), i.e. all small.
* Sign split 161 positive / 64 negative — the two drift directions.

`QLMatching.cxx:1351` stamps every cluster `set_cluster_t0(-1e12)` in an init
pass and overwrites it only for flash-matched clusters
(`match/docs/qlmatching-code.md:590`). A cluster that never matches keeps the
sentinel, and `x_t0cor = x_raw − dirx·(t0+offset)·v_drift` with
v_drift = 1.48073 mm/µs puts it 1.48073 × 10⁹ mm away. The same phenomenon is
already documented inside clus for SBND —
`clustering_cathode_bundle_rescue.cxx:981`, *"materialized with the sentinel T0
(56463: x_t0cor off by 1.5e6 m)"* — but nothing filters it out of the Bee dump
on PDVD.

Consequences: those 29 clusters are invisible in Bee (drawn off-detector), and
any offline consumer that takes the layer's bounding box gets a meaningless
one. It is **not** a reconstruction error — the reconstruction never used
`x_t0cor` for these clusters — it is a display leak. A fix belongs in the Bee
point dump (drop, or fall back to raw x, for clusters at the sentinel T0),
behind a default-OFF knob like everything else.

## 9. What is *not* established

* **The internal mechanism of §5.1 is a call-graph trace, not an
  instrumented one.** No per-edge census was run; the causal claim rests on
  the knob A/B, which is decisive about *whether* but not about *which edge*.
* **A tempting statistic that turned out to be noise, recorded so it is not
  re-derived:** 97 % of cluster 119's fabricated points have a live point
  within 2 cm in the (y, z) plane, which reads like "same wires, wrong drift
  time". The chance floor in the same (y, z) window is **0.906** — the
  observation is at noise. `d40_steiner_void_census.py` prints the null
  alongside the rate for exactly this reason. Do not quote the match rate.
* **The sign of the STM change disagrees with doc pdvd/36's own number**
  (that round reported the metric *reducing* STM tags, 688 → 657, on its
  120-event manifest; here it *adds* 28 on 21 events). The two arms are
  different manifests and different cfg epochs — doc pdvd/37 and 38 both
  flipped between them — so they are not comparable, and neither is evidence
  against the other. Resolving it needs a fresh 120-event pair, not a
  cross-quote.
* **Why cluster 87 gets worse with the metric off** is unexplained.
* Whether the un-fabricated fraction is *harmful* to the STM verdict has not
  been measured. §6 shows the metric moves STM tags; it does not show the
  fabricated points are what moves them.

## 10. Recommendation

**Do not flip `ctpc_aniso_metric` back off.** It answers the owner's P1 and
would make event 298595's picture clean, but §6 says it would move 96 of 127
STM-tagged cluster ids across 21 events (net 127 → 99) for an event-wide
reduction of 0.5 percentage points in fabricated Steiner points, and it does
nothing for P2 or for cluster 87. That
trade is the owner's to make, not this document's; the knob is one TLA away
(`PDVD_PR_TLA="-S ctpc_aniso_metric=false"`) whenever the owner wants an arm.

Three leads, in the order this document would rank them:

1. **Cap the component bridge.** P2, cluster 87 and the majority of the
   21-event fabricated population are the uncapped MST in
   `connect_graph.cxx:20-26, :91` bridging live components across arbitrary
   distance. A distance cut there is the one change that addresses the
   metric-independent share, and it is the same lead doc pdvd/39 round 2 §14
   left open. Default-OFF knob + the standard gate.
2. **Make the display honest instead of the reconstruction quieter.** The
   owner's complaint is about a *picture*. `steiner_graph` could drop points
   with no live 3D support within a configurable radius — a Bee-layer filter,
   no reconstruction effect at all, byte-identical when off. This would answer
   the original question without touching a physics knob, and would keep the
   fabricated points out of every future hand scan.
3. **Filter the sentinel-T0 points out of the Bee dump** (§8), same shape of
   fix.

## 11. Round 2 — is this PDVD-only? (owner question, 2026-09-04)

### 11.1 P1's cause cannot occur on SBND, for two independent reasons

1. **Configuration.** `ctpc_aniso_metric` appears nowhere in
   `cfg/pgrapher/experiment/sbnd/` nor in `sbnd_xin`'s drivers. SBND has never
   run it.
2. **Structure — the stronger reason.**
   `ctpc_yscale = min(1, drift_step / pitch)`
   (`CtpcAnisoMetric.h:74`). Doc pdvd/34 §2 measured the lattice constants:

   | detector | drift step | pitch U/V/W | pitch/drift | yscale |
   |---|---:|---:|---:|---:|
   | SBND | 3.126 mm | 3.000 / 3.000 / 3.000 | 0.96 | **1.000 — identity** |
   | PDVD | 2.9615 mm | 7.650 / 7.650 / 5.100 | 2.58 / 2.58 / 1.72 | 0.387 / 0.387 / 0.581 |

   On SBND the drift step is already *coarser* than the pitch, so the metric
   clamps to 1 and is the identity function. Even if someone enabled the knob
   on SBND, not one query result would change. Doc 34 §2 is titled for exactly
   this: *"Why SBND never saw this and PDVD cannot avoid it."*

### 11.2 P2's machinery IS live on SBND, so it had to be measured

`CreateSteinerGraph` / `ImproveCluster_2` are bound by
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` and
`sbnd/wct-pr-perevt.jsonnet` — the same retile-and-tile code that fabricates
the blobs. Nothing about §11.1 protects SBND from that.

**Measurement.** SBND dumps no Steiner Bee layer, so both sides are read from
the **PrDisplayDump calib JSON**, whose `steiner` block carries every cluster
that has a `steiner_pc` (87 on PDVD evt 298595 against the Bee layer's 9–15).
No detector config was touched. Two controls, both passing:

* **route** — on PDVD, where both sources exist, the calib `steiner` points and
  the Bee `steiner_graph` layer are the same points to **0.0007 cm** on all 15
  shared clusters. The calib route *is* the §5 route, with a bigger population.
* **frame** — `flag_terminal` terminals sit on the live cloud at median
  **0.252 cm** (PDVD), **0.173** (SBND ncpi0), **0.001** (SBND nuecc).

Both sides run `unmerge_assoc` (PDVD arm `d40nu`'s pipeline line names it;
the SBND arm logs `ClusteringUnmergeBundle:prassoc`), so the comparison is not
confounded by the doc pdvd/39 round-2 flip.

| | PDVD, 21 evt (cosmics) | SBND ncpi0, 19 evt | SBND nuecc, 48 evt |
|---|---:|---:|---:|
| clusters with a `steiner_pc` | 5 503 | 495 | 1 351 |
| Steiner points | 1 322 557 | 69 953 | 242 187 |
| unsupported > 3 cm | 1.84 % | 0.06 % | 0.04 % |
| **unsupported > 10 cm** | **0.78 % (10 255 pts)** | **0.00 % (0)** | **0.00 % (0)** |
| unsupported > 30 cm | 0.29 % (3 787) | 0 | 0 |
| worst unsupported group span | 230.7 cm | 6.9 cm | 19.2 cm |
| **multi-component clusters** | **7.1 %** (max **25** comps) | 0.4 % (max 2) | 0.5 % (max 3) |
| *conditioned on multi-component:* > 3 cm | **3.12 %** | 0.59 % | 0.58 % |
| *conditioned:* > 10 cm | **1.37 %** | **0.00 %** | **0.00 %** |
| 1-component clusters: > 3 cm | 0.21 % | 0.02 % | 0.00 % |

**Answer: PDVD-specific in practice, shared in code.** Not one SBND Steiner
point in 67 events is more than 10 cm from live charge, against 10 255 on
PDVD. The gap has two independent factors, and the conditioning separates them:

* **Exposure — 15×.** 7.1 % of PDVD clusters are multi-component against
  0.4–0.5 % on SBND, and PDVD reaches 25 components where SBND caps at 3. The
  mechanism can only fire on a cluster that is disconnected in space.
* **Severity — 5× on top.** Even restricted to multi-component clusters, PDVD
  is 3.12 % against SBND's 0.58 %, and 1.37 % against 0.00 % beyond 10 cm.
  PDVD's gaps are simply longer.

Note that SBND is not at zero: 11 unsupported groups exist across the 67
events, the worst spanning 19.2 cm. The defect is present, just never large.

**The confound that remains.** SBND arms are neutrino MC, PDVD arms are
cosmics. Conditioning on component count controls the cluster *shape*, but
SBND's sample contains no 25-component cathode-crossing muon at all, so part of
"SBND is clean" is "SBND's events do not produce the input." A cosmic-rich SBND
arm would test that; none was run here.

## 12. What `remove_bad_blobs` already does — a filter that is too weak

The retiler has its **own anti-ghost filter**, and it is not a missing feature:
`ImproveCluster_1::remove_bad_blobs` (`improvecluster_1.cxx:626`), called from
`improvecluster_2.cxx:267` after every retile. Measured on 039252/2 with
`PDVD_LOG_LEVEL=trace` (arm `d40trace`):

```
filter invoked 1192 times (per cluster per apa/face); blobs removed = 31533
clusters where it removed anything: 61 of 494 retiled
   ('main', 119) apa 3 face 0: removed   20, remaining  1254
   ('main', 119) apa 7 face 0: removed    0, remaining  2484   <- the column lives here
   ('main',  84) apa 5 face 0: removed  741, remaining  1915
   ('main',  84) apa 6 face 1: removed    0, remaining  8482
```

It is doing real work — 31 533 blobs deleted on one event — and it still lets
the 618-point column through.

Its criterion: build a graph over the **new** blobs with adjacent-slice
`overlap_fast` edges; **if there is more than one connected component**,
validate each by whether *one representative blob* of it overlaps an
**original** blob within ±1 slice; drop whole components that fail. Two holes:

* **(a)** with `num_components == 1` it removes nothing at all, however much of
  the single component is fabricated;
* **(b)** a component is judged by one blob, so a fabricated column that
  touches the real track anywhere inherits that blob's verdict.

**Which hole lets cluster 119 through is NOT established.** apa 7 face 0
removed 0, which is consistent with either. Establishing it needs one log line
(`num_components` and the per-component vote) and is the first step of any fix,
not an afterthought.

### 12.1 The run-length evidence for a threshold

A per-blob support test would also delete the *intended* fills — bridging short
dead/inefficiency gaps is what the retile is for. The bound has to be on how
**long** an unsupported run may be, and the distribution says where:

| unsupported-group span | PDVD (1 727 groups) | SBND (11 groups) |
|---|---:|---:|
| p50 | 1.4 cm | 3.0 / 3.3 cm |
| p75 | 5.4 cm | 4.2 / 3.9 cm |
| p90 | 14.1 cm | 5.8 / 11.6 cm |
| p95 | 23.7 cm | 6.3 / 15.4 cm |
| max | **230.7 cm** | 6.9 / 19.2 cm |

Groups longer than 20 cm are **6.3 % of PDVD's groups but 63.3 % of its
fabricated points** (15 407 of 24 323), and **SBND has none at all**. A bound
somewhere in 20–30 cm therefore removes about two thirds of the fabricated
content, leaves the sub-10 cm fills that the feature exists for, and would not
touch a single SBND group in these 67 events. That is an argument for the
*shape* of the threshold; the value still has to be scanned, and 21 cosmic
events is a thin basis for it.

## 13. Revised recommendation

§10 stands — do not flip `ctpc_aniso_metric`. Ranked, with what supports each:

1. **Bound the unsupported run inside `remove_bad_blobs`** (§12). The filter,
   the support test and the call site all already exist; this changes a
   component-level vote into a per-blob one with a run-length bound. It is the
   only proposal that attacks the metric-independent majority of the defect,
   it is in shared code so SBND is protected too, and §12.1 gives the
   threshold its evidence. Default-OFF knob, standard gate. **Blocked on the
   one-line measurement in §12 that says which hole is operating.**
2. **Cap the component bridge** in `connect_graph.cxx:20-26, :91` — the same
   defect one stage earlier: stop the path crossing the gap instead of deleting
   the blobs it caused. More invasive, because that graph feeds the taggers, so
   it will move verdicts where (1) only moves display and Steiner content.
3. **A "no live support" filter on the Bee `steiner_graph` layer.** No
   reconstruction effect whatever, byte-identical when off, and it answers the
   owner's actual complaint — the picture — immediately. It would have kept
   these points out of the hand scan in the first place. Independent of 1 and 2
   and the cheapest thing on this list.
4. **Filter the sentinel-T0 points out of the Bee dump** (§8).

## 14. Related

* `39_cosmic-only-chain-and-stm-bee-layers.md` — round 2 re-scoped the STM
  layers (why these points became visible) and named the uncapped bridge as
  the residual this document re-measures.
* `36_ctpc-anisotropic-metric-implementation.md` — the metric, its
  implementation and the 120-event measurement that preceded the owner's flip;
  its own §on ghost extensions is the earlier sighting of this symptom.
* `34_ctpc-anisotropic-distance-metric.md` — why the ctpc is a 2.58:1
  anisotropic lattice on PDVD.
* `25_clustering-iso-overcluster-39324.md`, `26_…-overseparation.md` — the
  long-range merge family that makes cluster 84 a 17-component object.
* `stm/gates/d40_aniso_ghost_gate.txt` — the full gate record.
* Scripts: `scripts/d40_steiner_void_census.py`,
  `scripts/d40_aniso_arm_summary.py`, `scripts/d40_steiner_void_xdet.py`,
  `scripts/d36_ctpc_caller_census.py`.
* `34_ctpc-anisotropic-distance-metric.md` §2 — the lattice table that makes
  the metric the identity on SBND.

# doc pdvd/40 — Steiner-graph points where there is no 3D image

**Status.** Rounds 1–2 (§1–§13) are diagnosis only. **Round 3 (§15) ships
the fix** behind two default-OFF knobs on the retiler (`bad_blob_max_run`,
`bad_blob_report`); with both absent every job config compiles and runs
byte-identically (§15.5); the PDVD flip is the owner's decision and
seven OFF/ON Bee sets are built for it (§15.10). Rounds 1–2 changed no code and no config; the production PDVD chain is exactly what it was. Two named
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

> **The ranking below is SUPERSEDED by §13**, written before the §12
> `remove_bad_blobs` measurement and the §12.1 run-length distribution existed.
> The "do not flip the metric" verdict stands; the ordering of the leads does
> not. Read §13.

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

Both sides run `unmerge_assoc`, verified from the logs rather than from the
runner (a peer flipped it into `PIPE_STM`/`PIPE_NU` during this session, so the
script is not evidence for what an older arm did):
`grep -c 'ClusteringUnmergeBundle":"prassoc"'` gives **2** on every `d40nu`
log and the SBND arm configures the same component. The §5-§6 arms are the
other side of that flip — `d39r2base` and `d40a0` both give **0**, so they ran
the pre-flip `PIPE_STM` and that A/B remains single-variable. The two tables
therefore sit on opposite sides of the un-merge and must not be diffed against
each other.

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

Groups longer than 20 cm are **6.3 % of PDVD's groups but 63.3 % of the points
that sit in unsupported groups** (15 407 of 24 323 — note this denominator is
the unsupported population, not the 1 322 557-point Steiner cloud of §11.2),
and **SBND has none at all**. A bound
somewhere in 20–30 cm therefore removes about two thirds of the fabricated
content, leaves the sub-10 cm fills that the feature exists for, and would not
touch a single SBND group in these 67 events. That is an argument for the
*shape* of the threshold; the value still has to be scanned, and 21 cosmic
events is a thin basis for it.

## 13. Revised recommendation

§10 stands — do not flip `ctpc_aniso_metric`. Ranked, with what supports each:

1. **Bound the unsupported run inside `remove_bad_blobs`** (§12). **DONE in
   round 3 (§15)** — and the measurement it was blocked on found a third
   hole, a stale cache, that mattered more than either of the two named here. The filter,
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

## 15. Round 3 — the fix (owner request, 2026-09-04)

> Can you provide a fix to improve the remove_bad_blobs filter function that
> leads to these ghost connections? Please design and use this event to debug.
> Please also run the 120 events and focus on this kind of similar issues.

**Short answer.** Two default-OFF knobs on the retiler
(`ImproveCluster_1`/`_2`, shared with SBND): `bad_blob_max_run` (a length; a
connected run of retiled blobs with no original-blob support longer than this
is removed whole) and `bad_blob_report` (a log-only census). On 039252/2 at
20 cm both named points are gone — the nearest surviving Steiner point to P1
is 25.2 cm away and to P2 17.6 cm, i.e. exactly the live charge doc 40 §1
measured them against — cluster 119's fabricated points go 698 → 7, and the
event has no Steiner point beyond 30 cm of live charge (484 → 0). The
instrumentation the fix was blocked on (§12) found that **neither of the two
holes named in §12 is the main one**: the filter never ran at all on the face
where the column lives, because the shadow cluster's cache goes stale between
faces (§15.2). The 120-event result, the threshold scan and the verdict-set
check are in §15.6–§15.8; the knobs ship OFF and the PDVD flip is the owner's
decision (§15.9).

### 15.1 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
# toolkit HEAD before this round a448708f; wcp-porting-img 4c170fd3.  Pins in
# /home/xqian/tmp/d41_libpin/{ref,new3,new6}/ (md5 in stm/gates/d40r3_bad_blob_gate.txt).
# Work tags of this round are d41*; the docs/scripts/gate carry the d40r3 prefix
# (a peer opened doc pdvd/41 the same evening).

# 99 provenance pctrees for the manifest events the doc-39 round-2 set lacked:
for e in <run idx from stm/events.txt without work/<run6>_<idx>_d39r2prov>; do
  ./scripts/stage_ql_tag.sh $run $idx d41prov
  PDVD_LIGHT_SUFFIX=_keep PDVD_MAX_JOBS=8 ./run_clus_evt.sh -save-pctree -s d41prov $run all
done
# the arms (each: -nu chain, -S dl_weights='', one pin, fresh tags):
ARM=d41ref   PIN=ref  JOBS=8 ./docs/nf_sp_img_clus/scripts/run_d40r3_arms.sh
ARM=d41base2 PIN=new3 JOBS=8 EXTRA="-S retile_bad_blob_report=true" ./docs/nf_sp_img_clus/scripts/run_d40r3_arms.sh
ARM=d41rep   PIN=new6 JOBS=3 EXTRA="-S retile_bad_blob_report=true" ./docs/nf_sp_img_clus/scripts/run_d40r3_arms.sh
ARM=d41fixNN PIN=new6 EVENTS=<21-event list> EXTRA="-S retile_bad_blob_max_run=NN -S retile_bad_blob_report=true" ...   # NN = 10, 20, 30
ARM=d41fix20x PIN=new6 JOBS=8 EXTRA="-S retile_bad_blob_max_run=20 -S retile_bad_blob_report=true" ...  # the graded 120
# gates and censuses:
docs/nf_sp_img_clus/scripts/d40r3_hash_gate.py d41ref d41base2          # OFF path + report neutrality
docs/nf_sp_img_clus/scripts/d40r3_hash_gate.py d41base2 d41rep          # final binary, noise floor
docs/nf_sp_img_clus/scripts/d40r3_bad_blob_cases.py --runs-from d41fix20x work/*_d41base2
docs/nf_sp_img_clus/scripts/d40_steiner_void_xdet.py "PDVD:work/*_d41base2" "PDVD:work/*_d41fix20x"
docs/nf_sp_img_clus/scripts/d40r3_grade.py d41base2 d41fix20x
# the single-event debug loop (trace level dumps one line per retiled blob):
./scripts/stage_pr_tag.sh 39252 2 d41dbg4 d39r2prov
PDVD_LOG_LEVEL=trace PDVD_PR_TLA="-S retile_bad_blob_report=true -S dl_weights=''" ./run_pr_evt.sh -nu -s d41dbg4 39252 2
```

### 15.2 What the instrumentation found: a third hole, and it is the one that matters

`bad_blob_report=true` makes `remove_bad_blobs` print one `BADBLOB` line per
(cluster, apa, face) call — blob counts, component counts, per-blob support,
what the legacy vote removes, and the connected **runs** of unsupported blobs
with their span — and, at trace level, one `BADBLOBPT` line per retiled blob.
Read on 039252/2 (arms `d41dbg2`/`d41dbg4`), three things came out, in the
order they were understood:

1. **The log's cluster ids are not the Bee/calib ids.** `Cluster::ident()`
   (= `get_cluster_id()`) at retile time is not the `cluster_id` the dump
   writes at the end of the chain: for this event Bee cluster 119 is retile
   ident 120, Bee 84 is a union of several retile idents (104, 108, 325, …).
   Ids drift as later stages split and re-home clusters. Every join in this
   round is therefore **geometric** — runs and 3D groups are matched on (y, z),
   the coordinate the raw drift frame of the blob centers shares with the
   corrected frame of the dump. The census script says so in its header so it
   is not re-derived.
2. **The historical vote is an any-blob vote, not a first-blob vote.** The loop
   at `improvecluster_1.cxx:710-765` does `if (good.count(comp)) continue;` —
   it skips only components *already found good*, so a component's blobs are
   tested in order until one is supported. §12's hole (b) as written ("one
   representative blob speaks for the component") is not quite what the code
   does; the real shape of the hole is that a fabricated column *attached to a
   real track* sits in a component that has supported blobs and is kept
   whole. My first pure reimplementation read the loop as first-blob and the
   120-event hash gate caught it (arm `d41base`, 120/120 FAIL, kept as the
   record); the corrected core (`d41base2`) is identical to the old binary on
   all 120 events (§15.5).
3. **The filter never ran on the column's face.** The `BADBLOBRM` census
   (points before/after removal) and the missing second-face `BADBLOB` lines
   showed that `Cluster::npoints()` and `time_blob_map()` on the shadow cluster
   do not change when blobs are inserted or removed. The shadow cluster's
   `ClusterCache` (`Mixins::Cached`) is filled on first use and **nothing
   invalidates it on child insert/remove** — `Cluster::on_insert/on_remove`
   clear only the `sv3d` memo. The caller (`improvecluster_2.cxx:195-274`)
   inserts one face's blobs, filters, removes, inserts the next face's blobs,
   filters again — and from the second face on, `time_blob_map()` has no
   entry for the face being filtered, `all_new_blobs` is empty, and the
   function returns nothing. On 039252/2, 80 of 493 retiled clusters span more
   than one (apa, face) and **not one of them has a second-face `BADBLOB`
   line in the knob-OFF arm**. Cluster 119 (retile ident 120) spans apa 3 face 0
   and apa 7 face 0; the fabricated column is a 1047-blob / 1035-blob pair of
   runs along the drift direction (one blob per slice, ~310 cm in raw x), one
   on each face, and the apa-7 half was never examined. This is the "apa 7
   face 0 — removed 0" of §12, now explained: not a weak vote but a filter
   that did not run.

The stale cache is a pre-existing defect of the historical path (it is present
in `ImproveCluster_1::mutate` too, same loop shape). It is **fixed only under
the knob** (`if (m_bad_blob_max_run > 0) shad_cluster.invalidate_cache();` at
the top of `remove_bad_blobs`), because fixing it unconditionally changes the
legacy output for every multi-face cluster — §5.1 stop-and-ask territory,
reported here rather than done. Under the knob the second faces are filtered
by the same legacy vote plus the run bound.

### 15.3 The fix, as shipped

`clus/inc/WireCellClus/BadBlobRuns.h` holds the decision core as a pure
function over indices (components, the historical any-blob vote, runs of
unsupported blobs inside kept components, bounding-box span, removal list);
`remove_bad_blobs_runs` in `improvecluster_1.cxx` builds its inputs from the
two blob maps and is the only path taken when either knob is set. Semantics:

* adjacency = the historical adjacent-slice `overlap_fast(·,1)` **plus
  same-slice overlap**, so a column at fixed drift time (many blobs in one
  slice) is one run rather than N singletons;
* per-blob support = the historical ±1-slice overlap with an *original* blob,
  applied to every blob;
* component vote = the historical any-blob vote, unchanged (asserted equal to
  a literal transcription in the doctest);
* run bound = connected pieces of unsupported blobs inside kept components; a
  run whose bounding-box diagonal of blob centers exceeds `bad_blob_max_run`
  is removed whole. Short unsupported fills — the dead-region bridging the
  retile exists for — survive;
* `bad_blob_max_run <= 0` (the C++ default) takes the historical code path
  textually unchanged; `bad_blob_report` alone routes through the new function
  but returns exactly the legacy vote (gated, §15.5).

This is a **deliberate divergence from the prototype** (M15):
`Improve_PR3DCluster_2` (`ImprovePR3DCluster.cxx:515-620`) has the same
`num > 1` guard and the same component vote, and the toolkit port was
faithful. The divergence is licensed by this request and confined to the
knob-ON path.

Config: `cm.improve_cluster_2(bad_blob_max_run=null, bad_blob_report=false)`
in `cfg/pgrapher/common/clus.jsonnet` with the key-suppression idiom;
`protodunevd/pr.jsonnet` params `retile_bad_blob_max_run` (cm) /
`retile_bad_blob_report`; `pdvd/wct-pr-perevt.jsonnet` TLAs of the same names
(`PDVD_PR_TLA="-S retile_bad_blob_max_run=20 -S retile_bad_blob_report=true"`).
SBND's job config compiles byte-identically (§15.5); no SBND file is touched.

Tests: `clus/test/doctest_bad_blob_runs.cxx` — the 30 cm middle removed at
20 and kept at 40; a detached unsupported component removed at any bound;
bound 0 equals the legacy vote; the any-blob vote (a component whose only
supported blob is its last is kept); same-slice adjacency joining a
fixed-time column; longest-first reporting. `wcdoctest-clus`: 22 671
assertions pass.

### 15.4 039252/2 before and after

Same pctree (`d39r2prov`), same chain (`-nu`, `dl_weights=''`), knob-OFF arm
`d41base2` vs knob-ON `d41fix20b` (20 cm, pin `new5`, the same code as the
final pin `new6` except a census line):

| | knob OFF | 20 cm bound |
|---|---:|---:|
| nearest Steiner point to P1 (39.02, 75.74, 201.84) | 0.01 cm | **25.19 cm** |
| nearest Steiner point to P2 (275.46, 12.14, 8.80) | 0.01 cm | **17.58 cm** |
| cluster 119: Steiner points / fabricated > 3 / > 10 / > 30 cm | 2808 / 698 / 618 / 401 | 2112 / **7 / 0 / 0** |
| cluster 84: Steiner points / fabricated > 3 / > 10 / > 30 cm | 40 382 / 766 / 191 / 0 | 36 958 / 196 / 20 / 0 |
| event: fabricated > 3 / > 10 / > 30 cm | 3574 (2.52 %) / 1516 / 484 | **1150 (0.85 %) / 173 / 0** |
| retiled blobs removed (all calls) | 31 600 (legacy vote) | 57 742 |
| TGM / STM / FC tagged sets | 39 / 5 / 33 | identical |

The 25.2 cm and 17.6 cm are the distances §1 measured from P1 and P2 to the
nearest *live* point: what remains nearest is real charge. The points that
survive beyond 3 cm in cluster 84 are short fills (the largest surviving group
is 20 points beyond 10 cm; nothing beyond 30 cm in the event).

A knob-ON arm at 20 cm **without** the cache refresh (`d41fix20`, pin `new3`)
left both clusters bit-identical to knob-OFF while removing 13 220 blobs
elsewhere — the run bound alone cannot reach a face the filter never visits.
That arm is the proof that §15.2 item 3 is the operative defect for P1 and P2,
not a side finding.

### 15.5 Gates

Everything below is in `stm/gates/d40r3_bad_blob_gate.txt` with the pin
checksums.

* **Compiled config.** `wct-pr-perevt.jsonnet` at HEAD vs the working tree
  with both knobs absent: identical JSON. Knobs on: the only differing node is
  `ImproveCluster_2:pr`, keys `bad_blob_max_run` and `bad_blob_report`. SBND's
  `sbnd_xin/wct-pr-perevt.jsonnet` compiles byte-identically against HEAD cfg
  and the working tree (267 112 bytes).
* **OFF path and report neutrality, 120 events.** `d41ref` (pre-round binary,
  no knobs) vs `d41base2` (new binary, `bad_blob_report=true`), `mabc-pr.zip`
  by member content plus the calib dump with the `*_ms` timers stripped:
  **PASS 120 / FAIL 0**. One gate proves both that the new code's default path
  is the old code and that the census knob touches only the log.
* **Final binary, noise floor, 120 events.** `d41base2` (pin `new3`) vs
  `d41rep` (pin `new6`, same knobs): **PASS 120 / FAIL 0** — the final binary's default path is build 4's, and the run-to-run noise floor of this chain is zero.
* **Doctests.** `wcdoctest-clus` 22 671 assertions, six new cases.
* **The gate that failed.** The first census arm (`d41base`, pin `new`) failed
  120/120 against `d41ref` because the pure core mis-read the historical loop
  as a first-blob vote (§15.2 item 2). A reading of a loop is a hypothesis;
  the hash gate is the test. Kept in the gate file.

### 15.6 Threshold scan, 21 events

The 21 doc-39 round-2 events (`d39r2prov` pctrees), knob OFF vs 10 / 20 /
30 cm, same pin `new6`, same chain. "Fabricated" as everywhere in this doc: a
Steiner point with no live 3D point of any cluster in the event within N cm
(`d40_steiner_void_xdet.py`, frame control terminals median 0.252 cm on every
arm):

| 21 events, 5 503 clusters, | OFF | 10 cm | 20 cm | 30 cm |
|---|---:|---:|---:|---:|
| Steiner points | 1 322 557 | 1 289 110 | 1 295 872 | 1 298 573 |
| > 3 cm from live | 24 393 (1.84 %) | 4 151 (0.32 %) | 6 425 (0.50 %) | 8 149 (0.63 %) |
| > 10 cm | 10 255 (0.78 %) | 226 (0.02 %) | 269 (0.02 %) | 581 (0.04 %) |
| > 30 cm | 3 787 (0.29 %) | 3 | 3 | 3 |
| 1-component clusters, > 10 cm | 89 | 0 | 0 | 0 |
| TGM / FC tagged sets vs OFF | — | identical | identical | identical |
| STM tagged set vs OFF | 115 ids | +1, −0 | +1, −0 | +1, −0 |
| wall, sum of 21 | 754 s | 811 s | 789 s | 811 s |

Reading it:

* the bound removes **97–98 % of the points beyond 10 cm at any of the three
  values**, and everything beyond 30 cm except three points. The choice of
  value trades the 3–10 cm band: 10 cm removes 2 300 more points there than
  20 cm does. Those are exactly the short unsupported fills §12.1 argued the
  retile exists for (dead-region bridging), and this census cannot tell a
  legitimate fill from a short ghost — "unsupported" is the definition of a
  fill. So the value is set by §12.1's argument, not by this table: **20 cm**,
  above every SBND group ever measured (max 19.2 cm, §11.2) and below PDVD's
  p95 (22.3 cm).
* the live `clustering` layer is identical in all arms (asserted, 21/21), so
  the id-keyed verdict comparison is valid; the taggers move by **one STM tag
  on one event** (039252/10 cluster 70, gained, same at all three values —
  §15.7), TGM and FC not at all;
* the cost is +5–8 % wall on the PR job. Two sources: the second faces are now
  actually filtered (they were skipped before), and the same-slice adjacency
  is O(n²) per slice. Peak RSS unchanged (3.0 GB).

### 15.7 The one verdict that moves: 039252/10 cluster 70 gains STM

Same at 10, 20 and 30 cm, so it is the cache refresh (second face now
filtered) rather than the bound value. Cluster 70 is a 62 cm, 214-live-point
track lying almost in a drift plane (live x 261.8–267.1 cm). Knob OFF its
Steiner graph carried a fringe of points 2–10 cm from the charge and the STM
fit **failed** — `persist_stm_fit: pass=0 status=2 kink=30 exit_L=19.9
left_L=108.5 npts=200`. Knob ON the graph is re-drawn (107 of 141 Steiner
points replaced; 132 remain), the fit runs the length of the track —
`status=0 kink=226 exit_L=144.4 left_L=0.0 npts=226` — and the verdict passes
the cathode and readout-edge guards: `STM=1`. Reported, not adjudicated: it
reads as a fit that was being spoiled by ghost points, but one event is one
event. TGM and FC sets do not move on any of the 21 events.

### 15.8 The 120-event manifest: cases, the fix, and what it costs

**The cases.** With the knob-ON arm's run reports joined to the knob-OFF arm's
3D groups (`d40r3_bad_blob_cases.py --runs-from d41fix20x`; the OFF arm cannot
report its own hidden faces), the 1 527 Steiner groups more than 10 cm from
live charge on the 119 events with a calib dump classify as:

| class | groups | Steiner points | meaning |
|---|---:|---:|---|
| (b) mixed component | 1 065 | 65 842 | the run sits in a component that also holds real blobs — attached to a track, inherits its vote |
| (c) wire-supported | 373 | 20 244 | the group's blobs overlap an *original* blob in wire space within ±1 slice, yet the sampled 3D points are far — the per-blob support test itself is loose here |
| (a) single component | 49 | 3 338 | the retile of that face is one component, never voted on |
| (d) no run within 15 cm | 40 | 1 069 | the census's own blind spot |

Read against the OFF arm's *own* run reports (first faces only: b 772 / c 475
/ d 235), about 300 of the (b) groups and most of the (d) groups were on faces
the filter never visited. The worst cases are all of one shape: a single run
of one blob per slice along the drift direction, 250–310 cm long, attached to
a cathode-crossing muon (039349/52 cluster 58, 039253/8 cluster 99, 039349/22
cluster 67, …; table in `/home/xqian/tmp/d41_cases_120.txt`, reproducible from
the Repro block). That is the same object as §5's column, and it is what
`hack_activity_improved` draws when the whole-cluster shortest path crosses
between drift volumes.

**The fix, 120 events, 20 cm** (`d41base2` vs `d41fix20x`, same pctrees, same
pin, same chain; `d40_steiner_void_xdet.py`):

| 119 events, 30 109 clusters with a `steiner_pc` | knob OFF | 20 cm bound |
|---|---:|---:|
| Steiner points | 7 164 712 | 6 999 099 (−2.3 %) |
| > 3 cm from live | 123 140 (1.72 %) | **41 185 (0.59 %)** |
| > 10 cm | 38 652 (0.54 %) | **2 040 (0.03 %)** |
| > 30 cm | 8 982 (0.13 %) | **24** |
| 1-component clusters, > 10 cm | 220 | 9 |
| ≥ 5-component clusters, > 10 cm | 9 975 (0.95 %) | 1 066 (0.11 %) |
| 3D groups > 10 / > 20 / > 30 cm | 1 527 / 649 / 355 | 527 / 81 / 26 |
| group span p95 / p99 / max | 22.3 / 58.7 / 244.4 cm | 11.1 / 19.9 / 64.4 cm |
| wall, sum | 4 071 s | 3 619 s |
| peak RSS, max | 3.20 GB | 3.30 GB |

95 % of the points beyond 10 cm and 99.7 % of those beyond 30 cm are gone. The
residual beyond 10 cm is mostly class **(c)** — 135 of the 527 surviving
groups, and the four longest (40–64 cm) — which the run bound cannot reach by
construction: those blobs are "supported" in wire space. That is the lead for
any further round, not a tighter bound. (The wall time went *down* on the
120; on the 21 it went up 5–8 %. Both arms ran on a shared box; the honest
statement is "no cost visible above the noise".)

**Verdict sets** (`d40r3_grade.py`; the live `clustering` layer is asserted
identical, points and cluster-id multiset, before any by-id comparison):

| | TGM | STM | FC |
|---|---:|---:|---:|
| tagged ids, OFF → ON (119 comparable events) | 2 586 → 2 586 | 583 → 582 | 2 462 → 2 462 |
| only-OFF / only-ON | 0 / 0 | **7 / 6** | 0 / 0 |

TGM and FC do not move. **STM turns over 13 ids on 13 events** for a net −1,
and every one of them is a fit-status change on the same fitted track
(`persist_stm_fit`, status codes: 0 accepted, 2 long leftover past the kink,
3 dQ/dx KS rejection, 4 extra-tracks veto):

| direction | n | status OFF → ON | what changed |
|---|---:|---|---|
| gained | 4 | 3 → 0 | same fit (npts, exit_L within a few %), the dQ/dx KS test now passes |
| gained | 1 | 2 → 0 | the fit itself was repaired: 039252/10 cluster 70 (§15.7) |
| lost | 5 | 0 → 3 | same fit, the dQ/dx KS test now fails |
| lost | 2 | 0 → 2 | **same fit**, the kink relocates to point 3 / 6 so 134–186 cm becomes “leftover” (039349/23 cluster 59, 039349/69 cluster 26) — see the correction below |
| lost | 1 | 0 → 4 | extra-tracks veto (039349/48 cluster 15) |

The 120th event, 039349/7, is excluded from the by-id table because its live
layer differs: knob ON gains STM on cluster 32, `protect_bundle` therefore
opens one more bundle (20 → 21 extra clusters), and the live cluster ids
downstream shift. That is a consequence of a verdict change, not the retile
touching the live grouping. Counting it, the turnover is 7 lost / 7 gained.

Two readings, both stated because they pull in opposite directions: the
dQ/dx-KS flips (9 of 13) are the tagger reacting to a Steiner graph with
fewer ghost points along a *real* track — end-charge profiles change when
fabricated points near the end disappear — and are as likely to be
corrections as losses; the two `0 → 2` losses are the same *kink-placement*
fragility as the one gain, running the other way (below). Neither is
adjudicated here; both are named so the flip decision is made with them in
view.

**Correction (2026-09-05, from the Bee sets of §15.10).** An earlier version
of the row above said the two `0 → 2` losses were the fitter losing a
crossing the fabricated bridge used to give it. The `stm_fit` layer says
otherwise: the trajectory is unchanged in both arms —

| case | OFF | ON |
|---|---|---|
| 039349/23 cluster 59 | `status=0 kink=215 exit_L=136.0 left_L=0.0 npts=215`, fit path 135.4 cm | `status=2 kink=3 exit_L=2.1 left_L=133.8 npts=216`, fit path 135.3 cm, **same two ends** |
| 039349/69 cluster 26 | `status=0 kink=294 exit_L=190.2 left_L=0.0 npts=294`, fit path 190.5 cm | `status=2 kink=6 exit_L=3.5 left_L=186.3 npts=293`, fit path 190.0 cm, **same two ends** |
| 039252/10 cluster 70 (the gain, §15.7) | `status=2 kink=30 exit_L=19.9 left_L=108.5 npts=200` | `status=0 kink=226 exit_L=144.4 left_L=0.0 npts=226` |

so nothing breaks and no crossing is lost: what moves is where
`TaggerCheckSTM` puts the kink. In both losses it jumps to the third or sixth
trajectory point — 2.1 and 3.5 cm from the end — and the whole track past it
is then counted as leftover. In both cases the removed ghost was attached at
exactly that end (039349/23 cluster 59: a 134.8 cm removed group running from
(315.8, 44.8, 272.1) to (448.5, 66.0, 283.6), off the fit end at
(313.1, 67.6, 273.4)). 039252/10 cluster 70 is the same mechanism with the
signs swapped: OFF puts the kink at point 30 and rejects, ON finds none.
The kink test near a track end is therefore fragile to the charge sitting
just beyond that end, in both directions; the flip changes which tracks it
bites, not whether the tracks are real. That is a lead for its own round
(the kink finder, not the retiler) and it lowers the weight of the `0 → 2`
column in the decision below.

### 15.9 Recommendation

1. **Ship as is, default OFF** (done). The code path with both keys absent is
   the pre-round binary on 120/120 events; SBND's config is untouched and
   compiles identically.
2. **The PDVD flip** — `retile_bad_blob_max_run = 20` in
   `pdvd/wct-pr-perevt.jsonnet` — is the owner's decision (§5.1). For it:
   the two named points and 95 % of everything like them are gone, the
   cosmic taggers TGM/FC do not move, and the STM set turns over 13 of ~585
   ids for a net −1 with 9 of 13 being the dQ/dx test reacting to a cleaner
   graph and 3 more (2 lost, 1 gained) being kink placement moving within an
   unchanged trajectory. Against it: the STM tag set is what the PDVD
   stopping-muon campaign (doc 25) is built on, so a turnover of 2 % is not
   free even when no fit is damaged. A hand scan of the 13 would settle which
   way each goes; the Bee sets of §15.10 are built for it.
3. **The residual lead is class (c)**, not a smaller bound: blobs that overlap
   an original blob in wire space but sample far from it in 3D. A per-blob 3D
   support test against the original cluster's points (the `time_blob_map`
   overlap helper already exists on `Cluster`) would replace the wire-space
   test; it is a different knob and a different round.
4. **The stale cache is worth fixing in the legacy path too**, as a
   default-ON behaviour change with its own gate: every multi-face retile on
   every detector that binds `ImproveCluster_2` has been running with its
   second faces unfiltered since the port. That is a §5.1 decision.
5. Two doc-40 items stay open: the uncapped `connect_graph` bridge (§7, §13.2)
   and the sentinel-T0 leak (§8).

### 15.10 Bee sets for the flip decision (uploaded 2026-09-05)

Seven sets, one per event, **each holding two Bee events on the same
geometry: event `0` = knob OFF (`d41base2`), event `1` = knob ON at 20 cm
(`d41fix20x`)**, so the arms toggle in one tab. Built by re-indexing
`work/<evt>_<arm>/mabc-pr.zip` `data/0/0-*.json` into `data/0` and `data/1`
(the `run_bee_combined_evt.sh` idiom) and uploaded with `upload-to-bee.sh`;
each UUID was content-verified (HTTP 200, both events listed). Repro:

```
docs/nf_sp_img_clus/scripts/d40r3_bee_compare.sh \
    039252_2 039252_4 039252_10 039349_23 039349_69 039349_48 039349_52
```

| event | set | what to look at |
|---|---|---|
| 039252/2 | `386e9ea2-37b6-4740-a8bb-81e175f21bc5` | the origin. P1 (39.0, 75.7, 201.8) and P2 (275.5, 12.1, 8.8): 73 / 50 `steiner_graph` points within 8 cm in event 0, **0** in event 1. Cluster 119's two columns removed whole: (−9.8, 74.9, 200.0)→(124.6, 78.4, 204.9) 134.6 cm / 456 pts and (−2.1, 74.4, 207.2)→(124.6, 77.9, 208.7) 126.8 cm / 237 pts |
| 039252/4 | `63988137-54cb-47e4-aad3-b541e7ebdc01` | the largest removal in the 120: 8 160 of 59 282 `steiner_graph` points (13.8 %), 49 groups > 10 cm. Biggest at (226.0, −51.6, 204.5) 103.7 cm and (100.5, −35.7, 192.8) 98.1 cm |
| 039252/10 | `70bc1140-d5fd-4d33-bca7-0cce0d799f9a` | the STM **gain**. `stm_tagged` cluster 70: absent in event 0, 214 points in event 1; `stm_fit` 127.8 → 143.8 cm as the fit reaches the true end. The ghost that caused it: 123.8 cm / 623 pts at (−56.1, 210.6, 213.9), removed |
| 039349/23 | `d24d7369-ca59-4e36-ba88-9bc6389bd5b5` | both signs in one event. Removed: 223.2 cm / 760 pts at (79.3, 267.0, 111.5) and 208.6 cm / 709 pts at (86.6, 272.0, 106.0). Lost: `stm_tagged` cluster 59 (1 056 pts in event 0, 0 in event 1) — its `stm_fit` is unchanged in both (135.4 / 135.3 cm, same ends), only the kink moves, and the 134.8 cm ghost at (381.7, 54.8, 277.8) that used to sit off its end is gone |
| 039349/69 | `454c7698-8abe-44cf-9a96-bf1292e59edb` | the marginal cost: only 39 points removed event-wide, one 18.7 cm group at (320.9, 80.2, 145.2), and `stm_tagged` cluster 26 (1 283 pts) is lost on it. `stm_fit` again unchanged (190.5 / 190.0 cm) |
| 039349/48 | `55f18fde-522f-4fa9-811f-c93dc9cf89ca` | the `0 → 4` loss (extra-tracks veto), `stm_tagged` cluster 15. Removed: two ~22 cm groups at (150.9, 223.6, 286.5) |
| 039349/52 | `e95684e0-d29e-4158-81a5-952d5347e8ca` | the **residual**. The census's worst 3D group (cluster 58, 244.4 cm, 798 pts) is still there in event 1: only 26 points are removed event-wide, none in a group > 10 cm. This is what the flip does *not* buy — the class-(c) lead of §15.9 item 3 |

Reading notes: `clustering` (live charge) and `stm` are **byte-identical**
between the two events of every set — they are the backdrop, not the signal.
`steiner_graph` carries only two distinct `cluster_id` values event-wide, so
identify the ghosts by the coordinates above, not by cluster. `stm_tagged`
shows the verdict (cluster present = tagged) and `stm_fit` the trajectory;
both loss clusters keep their `stm_fit` in event 1, so the fit can be
compared directly. The 9 dQ/dx-KS flips are not 3D-visible and no set is
built for them.

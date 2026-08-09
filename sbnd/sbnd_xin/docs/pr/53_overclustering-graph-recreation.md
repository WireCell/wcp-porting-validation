# doc pr/53 — SBND overclustering diagnosis: two distinct mechanisms, neither fixed by naive threshold tightening

Status: investigation only. No C++ or jsonnet changed. Fixes are proposed
below as default-OFF knobs for a future session; nothing in this doc is
shipped.

**Round 2 correction (below, §13):** Finding 2's original claim -- that
`clustering_isolated`'s absorb permanently over-clusters the two 18255-71372
pairs and that "no amount of tightening `protect_overclustering` or
`connect_graph_relaxed` touches these two pairs" -- is **wrong**. There is a
designed, already-production unmerge pass (`ClusteringUnmergeBundle:prassoc`,
the `unmerge_assoc` pipeline stage) that undoes exactly this merge, very early
in the PR chain, and it was empirically confirmed to work for this event: see
§13. Read §13 before acting on §4/§6's family-B conclusions.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin   # symlink into wcp-porting-img/sbnd/sbnd_xin
S=scripts/analysis/pr53/oc53_probe.py
python3 "$S" work-ncpi0-cb0805/ql_evt21073   -36.5  32.0 367.5  -31.2  28.8 369.6  "18345-21073"
python3 "$S" work-nuecc48-cb0805/ql_evt422851 -107.9 -55.1 344.6 -110.4 -61.9 350.0 "18255-422851"
python3 "$S" work-ncpi0-cb0805/ql_evt521075  -88.0 -33.5 456.2  -84.9 -33.3 450.8  "18255-521075"
python3 "$S" work-ncpi0-cb0805/ql_evt71372  -165.2 -129.9 226.4 -155.3 -103.1 229.0 "18255-71372 p1"
python3 "$S" work-ncpi0-cb0805/ql_evt71372  -161.5 -152.1 258.5 -159.8 -144.0 287.9 "18255-71372 p2"

# Round 2: 21073 gap figure
python3 scripts/analysis/pr53/plot_21073_gap.py

# Round 2: does unmerge_assoc undo the family-B merge before the taggers run?
./run_pr_chain_batch.sh work-ncpi0-cb0805 work-oc53-71372 data 71372
grep ClusteringUnmergeBundle:prassoc work-oc53-71372/pr_evt71372/wct_pr_evt71372.log
```

Verified against `work-ncpi0-cb0805` / `work-nuecc48-cb0805` (existing QL
products, M11 — no re-imaging done) at toolkit `apply-pointcloud` HEAD
`ba5bbe59`. Full output table: `docs/pr/53_pairs.tsv`. Scripts:
`scripts/analysis/pr53/{oc53_probe.py,plot_21073_gap.py}`. Round-2 PR-chain
output: `work-oc53-71372/` (fresh out_root, M13). Round-2 Bee link:
`bee/oc53-71372/oc53-71372.url`.

Code cited below:
```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
sed -n '349,381p' clus/src/clustering_protect_overclustering.cxx     # Separate_overclustering path test
sed -n '150,303p' clus/src/connect_graph_relaxed.cxx                 # connect_graph_relaxed path test + branches
sed -n '536,616p;696,732p' clus/src/Facade_Grouping.cxx              # is_good_point/test_good_point/get_closest_dead_chs
sed -n '290,340p;584,601p' clus/src/clustering_isolated.cxx          # small->big absorb, save_assoc_id
sed -n '245,330p' cfg/pgrapher/experiment/sbnd/clus.jsonnet          # SBND per-face pipeline order
cfg/pgrapher/experiment/sbnd/dead_regions.jsonnet                    # hand-declared W dead-gap column
```

---

## 1. Symptom (owner hand-scan, 2026-08-09)

Five point pairs across four events, flagged as over-clustering:

| event | A (x,y,z) cm | B (x,y,z) cm | owner note |
|---|---|---|---|
| 18255-422851 | (-107.9, -55.1, 344.6) | (-110.4, -61.9, 350.0) | "getting worse (overclustering), not connected" |
| 18345-21073  | (-36.5, 32.0, 367.5)   | (-31.2, 28.8, 369.6)   | "over clustering" |
| 18255-71372  | (-165.2, -129.9, 226.4)| (-155.3, -103.1, 229.0)| "over clustering gaps" |
| 18255-71372  | (-161.5, -152.1, 258.5)| (-159.8, -144.0, 287.9)| "over clustering gaps" |
| 18255-521075 | (-88.0, -33.5, 456.2)  | (-84.9, -33.3, 450.8)  | "over clustering gaps" |

Owner's framing: the SBND PR chain inherits MicroBooNE's `Protect_Over_Clustering`
graph-recreation pass. It is reasonable, but two of its assumptions do not
transfer: (1) MicroBooNE ran coherent-noise removal, which breaks isochronous
(perpendicular-to-drift) tracks, so its graph was tuned to bridge those breaks
generously; SBND does not run coherent-noise removal, so isochronous breaks
are rare. (2) SBND still has genuine U/V (induction) inefficiency for
prolonged (parallel-to-wire) topologies — but *not* on W (collection) — plus
occasional dead channels and SP mistakes. A pure-connectivity graph is
therefore also wrong; the fix needs to know *why* a gap exists, not just how
wide it is.

## 2. Method

Per pair: confirm both points are same-cluster in current-HEAD QL output
(`img-global`/`clustering-global` `cluster_id`, pctree `real_cluster_id`);
classify which mechanism drew the join via the `perblob` `assoc_cluster_id`/
`assoc_cluster_main` provenance (`ClusteringIsolated::save_assoc_id`,
`clus/src/clustering_isolated.cxx:596`); for graph-family pairs, find the true
closest-approach point pair between the two 1.5 cm-proximity components
containing A and B (not A/B themselves, which sit deep inside their own
component) and replay `Grouping::test_good_point`'s per-1cm-step scoring
exactly, using **that event's own** `ctpc_a<apa>f0p<U|V|W>` and
`dead_winds_a<apa>f0p<U|V|W>` point clouds pulled straight from its pctree
archive (not a generic radius search — see Finding 1 for why that distinction
matters). Full method and caveats are documented in the script's docstring.

## 3. Finding 1 — two mechanisms, not one, separated by `assoc_cluster_main`

| case | A assoc_id/main | B assoc_id/main | mechanism |
|---|---|---|---|
| 18345-21073  | 26/1 | 26/1 | A — connectivity graph |
| 18255-422851 | 59/1 | 59/1 | A — connectivity graph |
| 18255-521075 |  7/1 |  7/1 | A — connectivity graph |
| 18255-71372 p1 | 63/1 | 56/**0** | B — `ClusteringIsolated` |
| 18255-71372 p2 | 63/1 | 51/**0** | B — `ClusteringIsolated` |

All five are genuine over-clustering (same `real_cluster_id` in current-HEAD
output) — none is a false alarm.

## 4. Finding 2 — family B (18255-71372, both pairs): not a graph problem at all

The far endpoint of both 71372 pairs sits in a distinct *associated*
sub-cluster (`assoc_cluster_main = 0`): a 14-point and a 15-point fragment
absorbed into the 5448-point main body by `clustering_isolated`, which runs
**after** `protect_overclustering` in the SBND per-face pipeline
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet:300` vs `:320`). Its small→big
absorb is:

```cpp
double small_big_dis_cut = 80 * units::cm;
...
if (min_dis < small_big_dis_cut) { /* merge, no angle test, no charge test, no path test */ }
```
(`clus/src/clustering_isolated.cxx:297,325`). `iso_cathode_guard` — the one
existing guard on this pass (doc pr/19) — defaults FALSE in SBND production,
so nothing protects it here.

Measured (exact `test_good_point` replay on the true closest-approach pair
between the main body and each fragment):

| pair | true neck | branch | steps | all-3-planes-bad | relaxed-bad (U/V/W) | would the graph reject this? |
|---|---|---|---|---|---|---|
| p1 | 27.33 cm | strong | 28 | 27/28 | 22 (27/18/19) | **yes** |
| p2 | 10.70 cm | prolonged-U | 11 | 7/11 | 5 (1/4/5) | **yes** (`num_bad[W]>=3` veto fires) |

Both bridges would be refused by `connect_graph_relaxed`'s own test — the
graph is not too permissive here, `clustering_isolated`'s absorb bypasses it
entirely at the QL stage. **§13 (round 2) found that this merge does not
survive to the final PR/Bee output: a dedicated unmerge pass reverses it
before any tagger runs.** The claim that no graph fix touches these pairs is
therefore only true of the QL-stage snapshot analyzed below, not of what the
owner would see in a Bee link generated from the full PR chain — read §13
before drawing conclusions from this section.

## 5. Which pass actually drew the join, for family A

The joins are already present at QL output (`img-global`/`real_cluster_id`),
i.e. before the PR-stage `ClusteringProtectBundle` ever runs. So the pass that
first failed to split them is `clustering_protect_overclustering.cxx`'s
`Separate_overclustering` — see Finding 7 below, it is a materially weaker
duplicate of `connect_graph_relaxed`. `protect_bundle`/`relaxed` is the second
net downstream; it was re-run fresh only for 21073 (current-HEAD PR output,
still same-cluster), not for the other three — that re-run is left to the fix
session.

Both `protect_*` passes are **splitters**: neither can create a join. The
upstream merge pass that actually drew each edge (`connect1`, `close`,
`regular`, `extend`, or `isolated` — all run before `protect_overclustering`,
`cfg/pgrapher/experiment/sbnd/clus.jsonnet:268-300`) was not identified in
this session for the three family-A events; that trace is an open item for
the fix session, not assumed here.

Before trusting a "bridged across empty space" story for family A, the direct
`connect_graph_closely` short-range test was replayed on the *actual* blob
pair nearest A and B — not the 1.5 cm proximity-component proxy used for the
family classification above — using each blob's own
`u/v/w_wire_index_min/max` and `slice_index_min/max` scalars and
`overlap_fast(offset=2)` (`clus/src/Facade_Blob.cxx:151-163`):

| case | blobA slice | blobB slice | Δslice | overlap_fast |
|---|---|---|---|---|
| 422851 | 1616–1620 | 1580–1584 | 40 | **False** (U, W miss; only V overlaps) |
| 521075 | 1868–1872 | 1912–1916 | 40 | **False** (U, W miss; only V overlaps) |
| 21073  | 2528–2532 | 2596–2600 | 64 | **False** (U, W miss; only V overlaps) |

Δslice of 40–64 (≫ the ±1/2-slice window `connect_graph_closely` uses for
inter-blob linking) confirms these are genuine long-range bridges, not an
artifact of the 1.5 cm proximity radius merging two already-close blobs. (This
checks only the direct endpoint pair, not every intermediate blob along the
full chain between them — a complete trace is left to the fix session.)

## 6. Finding 3 — family A: the path test has an arithmetic blind spot below ~3 cm

`connect_graph_relaxed.cxx` (and its weaker duplicate,
`clustering_protect_overclustering.cxx:349-381`) samples the candidate bridge
in 1 cm steps and rejects only when

```cpp
double step_dis = 1.0 * units::cm;
int num_steps = dis/step_dis + 1;
...
if (num_bad > 7 || (num_bad > 2 && num_bad >= 0.75 * num_steps)) invalidate();
```

The final step always lands on the far endpoint, which is by construction
good, so `num_bad <= num_steps - 1`. For `dis < 3 cm`, `num_steps = 3`, so
`num_bad <= 2` — **the `num_bad > 2` floor can never fire, regardless of how
empty the gap is.** (The unconditional `< 3*units::cm` MST override at
`connect_graph_relaxed.cxx:534` then re-arms any such pair a second time, past
spanning-tree sparsification, even if some other branch had rejected it.) From
3–9 cm rejection needs nearly every interior step bad; the absolute `>7` term
only starts to matter beyond that.

Measured (exact `test_good_point` replay, true closest-approach pair between
proximity components):

| case | true neck | branch | steps | all-3-planes-bad | relaxed-bad (U/V/W) | rejected? |
|---|---|---|---|---|---|---|
| 422851 | 2.33 cm | strong | 3 | 2/3 | 0 (2/0/0) | no — arithmetically impossible |
| 521075 | 2.84 cm | strong | 3 | 2/3 | 2 (2/0/2) | no — arithmetically impossible |

Cross-check that family B never reaches this graph at all (§4's numbers,
repeated for contrast): both 71372 necks *would* be rejected by this exact
test, yet they are joined — by `clustering_isolated`, downstream.

## 7. Finding 4 — 18345-21073 is a third, distinct class: charge-contiguous

A and B are in the **same** 1.5 cm-proximity component — there is no bridge to
find. A shortest-charge-path search (breadth-first over a 0.99 cm bottleneck
radius) finds a 14.5 cm geodesic connecting them for a 6.54 cm straight-line
separation, detouring through a high-charge corner near (-33, 26.4, 365) cm.
No graph tightening — closer or relaxed — can separate two points that are
already connected by continuous charge; this is an imaging (ghost charge) or
genuine-topology question, not a graph-connectivity one, and needs the
owner's screenshot to classify further. A fresh Bee link for this event
already exists from an unrelated request earlier this session (current-HEAD
`ba5bbe59` reprocessing): `sbnd_xin/bee/bee0809/bee0809.url` (bee index 0 =
21073) — use it as the visual reference instead of generating a new one.

**Why this is "no gap", precisely (round 2 detail, per owner request).** Two
different paths between A and B give two different answers:

- The **straight line A→B** (6.5 cm, 7 one-cm steps) does cross a real gap:
  4 of 7 steps have *neither* live W charge *nor* a W dead channel — a
  genuine hole in the collection plane along the direct line (exact
  `test_good_point` replay).
- The **actual charge path** that puts A and B in the same `real_cluster_id`
  — the geodesic through the 0.99 cm-bottleneck proximity graph, 14.5 cm long,
  detouring through a high-charge corner near (-33, 26.4, 365) cm — has **zero**
  W gaps: all 20 points along it have live W charge. (V is frequently on a
  dead channel there — this event has 33 dead V wires vs 2 U / 43 W — but W
  itself is continuously live along the detour.)

So A and B are not joined by a connectivity test tolerating a hole; they are
joined by real, unbroken charge that simply does not run in a straight line.
The open question is whether that detour is genuine track topology or an
imaging/ghost-charge artifact — a visual call, not a further connectivity
check. See the figure below.

![18345-21073: straight line crosses a real W gap; the actual connecting path does not](53_21073_gap.png)

*(y,z) and (x,z) projections of the img-global charge in this cluster near
the pair. Red dashed = straight line A→B, with red X marking the 4 steps that
are a true W gap (open red circles: the 3 steps that are W-ok). Solid green =
the actual 14.5 cm charge-contiguous path connecting A and B, every point
W-live. Generated by `scripts/analysis/pr53/plot_21073_gap.py`; repro command
in the block below.*

## 8. Why the MicroBooNE tuning does not transfer — quantified

SBND's dead-channel inventory, read from event 71372's own
`dead_winds_a{0,1}f0p{U,V,W}` (a per-event list, not a fixed detector
constant — quoted here as a single measured data point):

| | U | V | W |
|---|---|---|---|
| apa0 | 2 | 33 | 43 |
| apa1 | 2 | 3 | 10 |

93 dead wires out of 11276 (1984+1984+1670 wires/apa × 2 apas) = **0.82 %** in
this event. The owner's framing states MicroBooNE ran with roughly an order of
magnitude more dead/masked channels (this session did not pin that figure to a
`prototype_base/` channel-status file — flagged as unverified, but it is the
working assumption behind the existing `>7` / `0.75` tuning and is directionally
consistent with the coherent-noise-removal history). Under that assumption the
free-bad-steps allowance and the length ratio were sized for a regime SBND is
not in.

The hand-declared SBND dead-gap column (`dead_gap_a{0,1}f0pW`, W winds
832–837, all-drift x) sits at **z ≈ 249.75–251.25 cm**
(`cfg/pgrapher/experiment/sbnd/dead_regions.jsonnet`). It declares all three
planes dead across the full vertical column when a step's W projection lands
there. None of the five gaps above is near it — it is not implicated in this
scan — but it is an unconditional connectivity loosener worth naming for the
fix session's constant inventory.

## 9. Constant inventory

| constant | value | where | MicroBooNE rationale | survives in SBND? |
|---|---|---|---|---|
| `step_dis` | 1.0 cm | `connect_graph_relaxed.cxx:161` | sampling granularity | yes, but interacts badly with short necks (§6) |
| `num_bad` floor | `> 2` | `connect_graph_relaxed.cxx:731` | avoid single-step false positives | too permissive under 3 cm — see §6 |
| `num_bad` cap | `> 7` | same | tolerate ~10% dead channels | oversized at SBND's 0.82% (§8) |
| bad-step ratio | `0.75` | same | same | oversized, same reason |
| MST short-link override | `< 3 cm` | `:534` | always trust very-short bridges | compounds the floor blind spot |
| prolonged wire-angle tol (closest-pair branch) | 12.5° all 3 planes | `:253` | generic prolonged test | internally inconsistent with the 7.5° below |
| prolonged wire-angle tol (directional branches) | U/V 12.5°, W 7.5° | `:378-392,469-472` | tighter W test | inconsistent origin, not obviously SBND-motivated |
| prolonged-U/V W-veto | `num_bad[3] >= 3` | `:283,291` | — | correctly plane-aware (owner's rule 2) |
| prolonged-W veto | none (drops W, uses U+V) | `:298-301` | — | **backwards** under owner's rule 2 — see Finding 7 |
| dead-gap column | z 249.75–251.25 cm, all-drift, all-3-planes | `dead_regions.jsonnet` | n/a (SBND-specific hand patch) | correct by construction, unrelated to these 5 pairs |
| small→big absorb distance | 80 cm, no angle/charge/path test | `clustering_isolated.cxx:297` | n/a (MicroBooNE-style tail stage) | directly responsible for family B (§4) |

## 10. Other issues flagged, not fixed here

- `Separate_overclustering` (`clustering_protect_overclustering.cxx:47-49`)
  claims to mirror `connect_graph_relaxed()`. It does not: no
  `test_good_point`, no per-plane `num_bad[]`, no 12.5°/7.5° prolonged
  branches, no `flag_strong_check`. It is strictly more permissive, and it is
  the QL-stage pass that actually ran on all five pairs (§5).
- `num_bad2[]` (per-plane dead-step counts, `connect_graph_relaxed.cxx:191-193`)
  is computed and never read anywhere in the function.

## 11. Proposed fixes (future session — none implemented here)

Every knob below is default-OFF; off-path leaves the compiled config and the
graph byte-identical. Gate: `abtest/events.txt` plus the SBND 48/50-event
member+nusel manifests used in pr/48–pr/51.

**F0 — `relaxed_edge_census` (instrumentation; do this first).** Debug knob on
`connect_graph_relaxed`/`Separate_overclustering` printing, per accepted
inter-component edge: endpoints, distance, branch, `num_steps`,
`num_bad[0..3]`, `num_bad1[0]`, the four angles, component sizes. Same shape
as the existing `[iso-cathode-guard]` marker
(`clustering_isolated.cxx:337`). Turns every proposal below into a measured
detector-wide census instead of a five-event sample, and is what would resolve
21073's classification with certainty.

**F1 — `relaxed_min_interior_steps` (family A, the sub-3-cm blind spot).**
Two variants: (a) make `step_dis` configurable (SBND candidate 0.5 cm) —
simple, but **global**: doubles `num_steps` everywhere, silently rescaling
both the floor and the ratio for the long prolonged bridges SBND legitimately
needs; (b) surgical — enforce a minimum interior sample count (e.g. ≥5)
independent of `dis`, so short necks get resolved without touching long-bridge
behavior. Recommend (b), record (a) as fallback.

**F2 — `relaxed_min_bad` (the floor, not the cap) + `relaxed_bad_ratio`.** The
binding constraint at short distances is the `> 2` **floor**, not the `> 7`
cap — lowering the cap alone changes nothing for a 3-step neck. Measured
`num_bad[0]`: 521075 = 2/3, 422851 = 0/3. A floor of `>= 2` with ratio ~0.5
rejects 521075 and leaves 422851 standing — the physically correct split,
since 422851's neck has V+W live throughout with only U missing, a legitimate
induction hole under the owner's own rule 2. **F1+F2 together do not close
422851** — say so; it may need a different (ghost/topology) route.

**F3 — plane-aware W authority (owner's rule 2) — the strongest finding here.**
W has no prolonged inefficiency, so a step lacking both W charge and a W dead
channel is genuinely empty. Today the W veto (`num_bad[3] >= 3`) fires only in
the prolonged-U/V branches; the prolonged-**W** branch instead computes
`sum_bad = num_bad[2] + num_bad[1]`, dropping the W check entirely, and the
strong-check branch never examines W at all. Two readings — present both per
the escalation rule on prototype/toolkit divergence, do not pick silently:
  - *(recommended)* Backwards for SBND: extend the W veto (knob-gated, default
    off) to the strong-check and prolonged-W branches.
  - *(counter)* For a track prolonged along W, the 3-D point's projection onto
    W is geometrically degenerate, so a W miss there is not informative — a
    legitimate reason the original code drops it. If this reading is chosen,
    scope the veto to the strong-check branch only.
  Fold the 12.5°/7.5° W-angle inconsistency (closest-pair vs directional
  branches) into the same knob.

**F4 — `iso_small_big_path_check` and/or `iso_small_big_dis_cut` (family B).**
The 80 cm angle-less small→big absorb directly causes both 71372 joins. (a)
require the absorb to pass the same good-point path test the graph uses (27/28
and 7/11 bad steps ⇒ both declined) — preferred, matches the owner's "gap
provenance, not distance" framing exactly; (b) simply lower
`small_big_dis_cut` (SBND candidate 10–15 cm) — cheaper, less targeted.
`iso_cathode_guard`'s `cathode_guard_xcut` on this same absorb (doc pr/19) is
the precedent for the knob shape.

**F5 — escalation item, not a proposal.** Reconciling `Separate_overclustering`
with `connect_graph_relaxed` cannot be made knob-off-byte-identical without
full-file duplication (fork-by-duplication convention). Record as an owner
decision; do not fold into F1–F4.

## 12. Open items for the fix session

- Trace which upstream merge pass (`connect1`, `close`, `regular`, `extend`,
  `isolated`) actually drew the family-A edges for 422851, 521075, and 21073 —
  not identified here (§5).
- Re-run `protect_bundle`/`relaxed` fresh through the PR chain for 422851 and
  521075 (only 21073 was re-run this session) to confirm whether the PR-stage
  net independently catches or misses them.
- Classify 21073 with F0's census once implemented, since it cannot be
  resolved by a neck/path argument.
- Confirm the ~10% MicroBooNE dead-channel figure against a real
  `prototype_base/` source, or drop the numeric comparison to qualitative.

---

## 13. Round 2 (2026-08-09) — `unmerge_assoc` already undoes family B; Finding 2's conclusion was premature

The owner asked how `clustering_isolated` relates to the prototype's group
concept: *"since there is no cluster group concept in the toolkit ... we keep
the association inside the cluster, so that we can separate them later. Your
first mechanism indicate that we did not separate them and then run some
other algorithm right?"*

That framing is correct about the design, and it is *not* what happened —
there **is** a separation step, and it is wired into SBND production. The
comment at `cfg/pgrapher/experiment/sbnd/clus.jsonnet:716-733` states the
intent directly:

> Second, INNER un-merge: undo the per-APA isolated GROUPING
> (`clustering_isolated` `save_assoc_id`), which merges a main cluster with
> the small clusters that are near it but NOT connected to it. The prototype
> only groups these (`Clustering_isolated` returns `main -> [(assoc, dis)]`
> and leaves `live_clusters` untouched); **the toolkit physically merges
> them**, so the STM/PR endpoint finder walks into a detached clump across
> empty space (docs 50, 51). ... Order matters: this runs AFTER
> `unmerge_bundle` so the flash grouping is undone first (outer) and the
> isolated grouping second (inner).

This is instantiated as a second `ClusteringUnmergeBundle` (name `assoc`,
reading `assoc_cluster_id`/`assoc_cluster_main` instead of the default
`real_cluster_id`) and is in SBND's default `pipeline_names`, positioned
`['switch_scope', 'unmerge_bundle', 'unmerge_assoc', 'steiner', ...]` —
**second stage of the whole PR chain**, before `steiner`, before every tagger,
before `protect_bundle`. It is gated on `save_assoc=true`, which has been the
SBND default since doc 68 specifically so `unmerge_assoc` is not a silent
no-op (`wct-clus-matching-perevt.jsonnet:153-166`).

**Empirical check.** §4/§6 above were QL-stage-only — the QL zip never runs
`unmerge_assoc` (it only runs once, inside the PR chain). Re-running event
71372 through the full current-HEAD PR chain
(`run_pr_chain_batch.sh work-ncpi0-cb0805 work-oc53-71372 data 71372`, bare =
production) and grepping its log:

```
ClusteringUnmergeBundle:prassoc cluster 19: 1828 blobs -> main 1666 + 24 associated cluster(s) holding 162 (real mode)
```

QL real_cluster_id 19 is exactly the cluster both 71372 pairs came from (§3's
table). In the **final** PR output (`mabc-pr.zip`'s `clustering-global`,
written after every tagger and `protect_bundle` has run):

| point | QL real_cluster_id | final PR `cluster_id` |
|---|---|---|
| p1 A (-165.2,-129.9,226.4) | 19 | 92 |
| p1 B (-155.3,-103.1,229.0) | 19 | **69** |
| p2 A (-161.5,-152.1,258.5) | 19 | 19 |
| p2 B (-159.8,-144.0,287.9) | 19 | **64** |

Both pairs are in **different** clusters by the time the chain finishes — the
same split confirmed as the log line above. A fresh Bee link for this
current-HEAD PR run: `sbnd_xin/bee/oc53-71372/oc53-71372.url`
(`https://www.phy.bnl.gov/twister/bee/set/94ce7def-814b-4fc1-80e1-8d77c55e47c2/event/list/`).

**Revised conclusion.** §4/§6's mechanism (the angle-less 80 cm absorb) is
real and correctly described at the QL stage, but it is not what the owner
would see in a Bee link built from the full PR chain — `unmerge_assoc` already
reverses it, well before any tagger. If 18255-71372 still shows as
over-clustered on the owner's own Bee view, the cause is **not** identified in
this doc and needs fresh investigation: candidates are (a) the owner's scan
predating `save_assoc=true`/`unmerge_assoc` (doc 68), (b) a view built from the
QL `img-global` layer rather than the PR `clustering-global` layer (the raw
QL charge layer never reflects PR-stage splits), or (c) a real bug in how
`unmerge_assoc`'s split interacts with a later stage that was not traced here.
This replaces §4/§6's implicit "nothing here needs the graph fix, needs the
`clustering_isolated` fix instead" with "confirm the symptom survives the
current-HEAD Bee link before spending F4 effort on it."

**F4 status downgraded.** Given the above, F4 (§11) should not be scheduled
ahead of confirming the symptom actually reaches a current-HEAD Bee link.
Family A (F1/F2/F3) and 21073 remain unaffected by this correction — neither
goes through `clustering_isolated`'s `assoc` grouping (both endpoints share
`assoc_cluster_main=1` in every family-A/charge-contiguous case, §3).

## Verification

No build, no A/B gate — no C++ or jsonnet was changed in this session.
Round 1's numbers come from read-only analysis of existing QL products
(`work-ncpi0-cb0805`, `work-nuecc48-cb0805`) plus a stand-alone Python replay
of the toolkit's own path-test arithmetic. Round 2 additionally ran the
production PR chain once, unmodified (`run_pr_chain_batch.sh ... data 71372`,
bare = production, freshness-proofed against HEAD `ba5bbe59` per M1 in the
earlier Bee-link task this session), to check a claim empirically rather than
by code-reading alone. Re-run the Repro block to reproduce every number in
this doc, including both rounds, from a clean shell.

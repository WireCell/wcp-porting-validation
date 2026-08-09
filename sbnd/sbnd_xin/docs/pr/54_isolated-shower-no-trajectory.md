# doc pr/54 — 18255-142421: a separated EM shower gets no fitted trajectory (silently discarded, not silently skipped)

Status: investigation only. No C++ or jsonnet changed. Fixes are proposed
below as default-OFF knobs for a future session; nothing in this doc is
shipped.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin   # symlink into wcp-porting-img/sbnd/sbnd_xin
python3 scripts/analysis/pr54/oc54_probe.py \
    work-ncpi0-cb0805/pr_evt142421 work-ncpi0-cb0805/ql_evt142421 7 108.6 -71.2 220.9
python3 scripts/analysis/pr54/plot_142421_blob.py

# The PR30AUDIT line quoted in §5 is already in the existing production logs,
# no re-run needed:
grep PR30AUDIT work-bee-0809/pr_evt142421/wct_pr_evt142421.log      # bare, current HEAD
grep PR30AUDIT work-ncpi0-cb0805/pr_evt142421/wct_pr_evt142421.log  # bare, Aug-5 HEAD

# The "Isolated residual segment" trace lines in §5 come from an existing
# TRACE-level arm made earlier this session for a different investigation
# (doc pr/51); its knobs do not touch clus/src/NeutrinoOtherSegments.cxx
# (grep confirms no hits), so its find_other_segments behaviour for cluster 7
# is representative of production. No new run was made for this doc.
grep -n "Cluster 7 .*tracks\|Isolated residual segment\|Added segment" \
    work-pr51-trace142421e/pr_evt142421/wct_pr_evt142421.log
```

Verified against `work-bee-0809/pr_evt142421` (current HEAD `288f5c1d`, bare
config, owner's own Bee batch) and `work-ncpi0-cb0805/pr_evt142421` (Aug-5
HEAD, bare config, the only arm with `calib-pr-evt142421.json` /
`PR_EXTRA_STAGES=pr_display`, needed for the `steiner` list). Both are
existing products (M11 — no re-imaging, no fresh PR run). Scripts:
`scripts/analysis/pr54/{oc54_probe.py,plot_142421_blob.py}`. Figure:
`docs/pr/54_142421_blob.png`.

Code cited below:
```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit
sed -n '2374,2500p' clus/src/NeutrinoPatternBase.cxx    # find_proto_vertex driver
sed -n '31,146p' clus/src/NeutrinoOtherSegments.cxx      # Step 1: tagging
sed -n '253,424p' clus/src/NeutrinoOtherSegments.cxx     # Steps 3-8: terminal graph, MST, quality cut
sed -n '451,620p' clus/src/NeutrinoOtherSegments.cxx     # Step 9: fit, accept_proto/accept_relaxed
sed -n '620,713p' clus/src/NeutrinoOtherSegments.cxx     # Step 9 else-branch: isolated-residual discard
sed -n '1391,1440p' clus/src/TaggerCheckNeutrino.cxx     # PR30AUDIT line (oseg_proto/relaxed/reject)
sed -n '875,990p' clus/src/MultiAlgBlobClustering.cxx    # Bee shower_track/track_fit writers
```

---

## 1. Symptom (owner hand-scan, 2026-08-09)

Run 18255, event 142421: a visually separate EM shower carries no fitted
trajectory. Owner's point: `(x, y, z) = (108.6, -71.2, 220.9)`, labelled
"cluster 7021" in Bee. Owner's reading: either the basic fit was never
performed on this isolated piece, or the trajectory was produced and then
dropped.

"Cluster 7021" needs unpacking first: in the Bee `shower_track-global` layer,
`real_cluster_id` is `cluster_id*1000 + start_segment_id`
(`MultiAlgBlobClustering.cxx:889-909`), so `7021` names a **shower**, not an
image cluster — it is shower whose start segment has graph index 21, inside
PR **image cluster 7**. All numbers below are for image cluster 7.

## 2. Method

Read-only replay against the event's own products, no code run:
`clustering-global` / `track_fit-global` (Bee bee zip), the `steiner` list in
`calib-pr-evt142421.json` (only produced with `PR_EXTRA_STAGES=pr_display`),
and an existing TRACE-level log (`work-pr51-trace142421e`, made earlier this
session for an unrelated investigation, doc pr/51) whose knobs do not touch
`NeutrinoOtherSegments.cxx` (grep confirms zero hits for its knob names in
that file), so its `find_other_segments` trace for cluster 7 is representative
of the production path for this specific mechanism.

## 3. Finding 1 — the owner's point sits in a spatially separated component of cluster 7, with no fitted point anywhere near it

Connected components of cluster 7's 11611 image points at 1.6 cm link radius:
**12 components**. The owner's point lands in a **2930-point** component
(bbox `x[84,110] y[-84,-84] z[219,259]` cm) whose closest approach to the rest
of cluster 7 is **2.03 cm** — it stays a separate component up to 2.5 cm link
radius and only merges into the main body at 3.0 cm.

Distance from this 2930-point component to the nearest `track_fit-global`
point of *any* of cluster 7's 11 final segments: **min 7.30 cm, median
24.37 cm** — every single point in the component is more than 5 cm from a
fitted point (`n>5cm frac = 1.000`). See `docs/pr/54_142421_blob.png`: the
orange component sits right at the base of a "V"-shaped fitted track (blue),
close in image-charge space but never itself painted blue.

## 4. Finding 2 — this is not "steiner missing" and not "cluster never processed"

Cluster 7's `steiner_pc` (the skeleton `find_proto_vertex` builds its
candidates from) has **5357** points; of those, **2029 lie inside the
2930-point component, and 465 of those are steiner terminals**. The
algorithm's own skeleton says this region is structure worth fitting — it is
not a case of "no steiner points here" (`NeutrinoPatternBase.cxx:2381,2384`)
or "cluster dropped by the beam-window gate" (`CreateSteinerGraph.cxx:131,
135,154` — see §7 below for the clusters that *are* dropped this way).
`find_proto_vertex` did run on cluster 7 (it is the main cluster), and
`find_other_segments` — the residual-coverage pass — ran the full
`nrounds_find_other_tracks=2` rounds on it.

## 5. Finding 3 — the mechanism: candidates from this region were formed, fit, and discarded, with almost no logging

`find_other_segments` (`NeutrinoOtherSegments.cxx:31`) is a nine-step
pipeline. Steps 1-8 partition the cluster's *untagged* residual points into
terminal-graph components via a Voronoi+MST construction and apply a quality
cut (`:415-423`) before a component is even allowed to become a fit
candidate. Step 9 then, for each surviving candidate **in size order**,
computes a real trajectory fit (`track_fitter.do_single_tracking(new_seg, ...)`
at `:473`) — and only *after* that fit exists decides whether to keep it. Two
independent discard paths exist past this point, **both fired for cluster 7 in
this exact event**:

**(a) The isolated-residual discard (`:620-712`).** When neither endpoint of a
freshly-fit candidate touches the existing graph (`!v1_existed && !v2_existed`
— true whenever the candidate sits, like our component, well outside the
`search_range` of every existing segment), the code tries an isochronous
parallel-track correction; if that fails (or the candidate is too short for
the correction to even attempt: `dir_mag <= 10cm` and not `>8cm && length>13cm`),
it is logged as **"Isolated residual segment"** and discarded — `remove_vertex`
is called on both endpoints, the segment is **never added to the PR graph**,
and **no counter records this at all**. From the TRACE arm, for cluster 7:

```
round 0: existing_segments=4, N=5357, num_tagged=1483, terminals=1187 -> 21 components
  Added segment with 102 points -> [do_single_tracking fit computed] -> "Cluster 7 Middle tracks" -> "Cluster 7 Isolated residual segment" (DISCARDED)
  Added segment with 250 points -> accepted ("Other tracks")
  Added segment with 100 points -> accepted ("Other tracks")
  Added segment with 4 points   -> accepted ("Other tracks")
round 1: existing_segments=8, N=5357, num_tagged=2498, terminals=1187 -> 13 components
  Added segment with 4 points -> "Isolated residual segment" (DISCARDED)
  Added segment with 3 points -> "Isolated residual segment" (DISCARDED)
  Added segment with 10 points -> accepted ("Other tracks")
```

The candidate sizes here (102, 250, 100, 4, 4, 3, 10 points, summing to well
under the 2029 steiner points in the region) show the region was **fragmented
by the terminal-graph partition into many small pieces** — not one 2029-point
candidate — and at least 3 of those pieces were explicitly logged as
isolated-residual discards for cluster 7 across the two rounds. A real fit
(`do_single_tracking`) was computed for the 102-point candidate before it was
thrown away — literally the owner's second hypothesis, "the trajectory was
produced and then dropped."

**(b) The `accept_proto`/`accept_relaxed` reject (`:585-607`), for candidates
that *do* touch the existing graph.** Even when a candidate connects
(`v1_existed || v2_existed`, the "Other tracks" branch), it is fit and then
kept only if `length > 30cm`, or `(direct_length < 0.78*length && length >
10cm && dQ/dx > 1.6*MIP)` (the prototype's own clause), or, if the toolkit-only
widening `m_other_seg_relaxed_accept` (default true) permits it,
`(direct_length < 0.72*length && length > 15cm && dQ/dx > 1.05*MIP)`. This
path *does* have a per-event audit counter, emitted at **INFO level in every
arm** (`TaggerCheckNeutrino.cxx:1391-1440`, "PR30AUDIT"), so it needs no new
run:

```
work-bee-0809/pr_evt142421      (current HEAD, bare): oseg_proto=2 oseg_relaxed_only=0 oseg_reject=4
work-ncpi0-cb0805/pr_evt142421  (Aug-5 HEAD,   bare): oseg_proto=2 oseg_relaxed_only=0 oseg_reject=5
```

4-5 candidates that reached the fit stage in this event and connected to the
existing graph were rejected outright — a majority of the ones evaluated
(2 accepted vs 4-5 rejected). The counter is process-wide but this SBND runner
is one event per process, so it is this event's total.

**Caveat, stated plainly:** neither discard path is logged with coordinates.
The per-point Step-1 tagging trace that would let a coordinate be tied to a
specific discard is commented out
(`NeutrinoOtherSegments.cxx:117-126`), and Step 8's quality-cut loop
(`:257-424`) has zero logging at all. So this doc cannot claim with certainty
that the *specific* 102-point (or any single) discarded candidate is made of
points from the owner's exact component, only that: (i) the mechanism is real
and demonstrably fires for cluster 7 in this event, multiple times, through
two independent paths; (ii) the region's own point-count budget (2029 in-blob
steiner points) is consistent with having been fragmented across several of
the ~21/13 terminal-graph components that round 0/1 produced; and (iii) a
Step-1 tagging replay (below) rules out the one remaining alternative
explanation.

### Step-1 tagging replay — ruling out "the region was silently marked already-covered"

Before the fragmentation-and-discard mechanism above, there is a simpler
possible explanation: that Step 1 tagged the region's steiner points as
"already covered" by the existing segments (`u_ok && v_ok && w_ok`,
`:102-114`, where each plane can pass via 2D proximity **or** the
`get_closest_dead_chs` dead-channel disjunct), so it never became a candidate
component at all. Replaying that test in Python for the 2029 in-blob steiner
points against cluster 7's final fitted-segment clouds (`oc54_probe.py`):

```
tagged (u_ok and v_ok and w_ok) = 0 / 2029  (0.0%)
plane U: ok=0/2029   (0.0%), dead-channel-only=0
plane V: ok=293/2029 (14.4%), dead-channel-only=293
plane W: ok=178/2029 (8.8%), dead-channel-only=178
```

Zero points fully tag. This rules the silent-tagging explanation **out**: the
region genuinely reaches Steps 3-9 as candidate material, consistent with
Finding 3 above (the fragmentation-and-discard story), not with a Step-1
short-circuit.

## 6. Checked and explained — not the reported bug

Clusters 2, 10, 11, 12, 14, 16, 41, 42, 89 (up to 5462 points each) have **no
steiner entry at all** in the calib json's 24-entry `steiner` list. This is
`CreateSteinerGraph`'s beam-window / `require_beam_flash` gating
(`CreateSteinerGraph.cxx:131,135,154`) — expected behaviour for
non-beam-window cosmic clusters, not a bug, and not the owner's case (which is
inside main cluster 7). Recorded here so the fix session does not chase it.

## 7. Constant inventory

| constant | value | file:line | role |
|---|---|---|---|
| `search_range` | 1.5 cm | `NeutrinoOtherSegments.cxx:31` default arg | Step-1 tagging base radius |
| `scaling_2d` | 0.8 | `NeutrinoOtherSegments.cxx:31` default arg | 2D-proximity tagging threshold = `scaling_2d*search_range` = 1.2 cm |
| `nrounds_find_other_tracks` | 2 | `NeutrinoPatternBase.cxx:2374` default arg, called with default at `:2477` | how many Step-1..9 rounds run per cluster |
| length-adjust floor | 3 cm | `:385` | if `special_A`-`special_B` length < 3cm, walk `special_A` back one MST hop |
| quality cut: single-point | `number_points==1` | `:415` | always dropped |
| quality cut: no-support absolute | `length < 3.5cm` when `number_not_faked==0` | `:417` | short + zero real-2D-support candidates dropped |
| quality cut: sparse-support | `number_not_faked < 0.25*N` (any length) or `< 0.4*N` (`length<7cm`) | `:418-419` | |
| quality cut: max-2D-gap ceiling | `max_dis_u,v,w < 3cm` each, sum `< 6cm` | `:420-421` | only within this envelope does the sparse-support clause apply |
| isochronous-snap trigger | `dir_mag > 10cm` or (`>8cm` and `length>13cm`) | `:633-634` | below this, isolated candidates skip straight to discard |
| isochronous-snap radii | 6 / 10 / 18 cm at widening tiers (18/36 cm track length) | `:652-706` | |
| accept: prototype clause | `length>30cm` or (`direct<0.78*length && length>10cm && dQdx>1.6*MIP`) | `:586-588` | |
| accept: toolkit widening | `direct<0.72*length && length>15cm && dQdx>1.05*MIP` | `:589-591`, gated by `m_other_seg_relaxed_accept` (default true) | |

## 8. Proposed fixes — NOT implemented this session

Every one is a default-OFF knob whose off-path leaves the compiled config and
graph byte-identical; gate = the SBND 48/50-event manifests used by pr/48-51.

**F0 — visibility first.** Uncomment the per-point trace at
`NeutrinoOtherSegments.cxx:117-126` behind a debug knob, and add one
`SPDLOG_LOGGER_DEBUG` line at the isolated-residual discard (`:711`, before
`remove_vertex`) printing the candidate's point count, `v1_fit_pt`/`v2_fit_pt`,
and `dir_mag` — mirroring the existing `oseg_reject` audit shape
(`TaggerCheckNeutrino.cxx:1391`) but for the isolated-residual path, which
today has **no counter at all**. This is what turns "probably fragmented and
discarded" into a measured, coordinate-attributable census, and is the
prerequisite for judging any of F1-F3 below.

**F1 — raise the isochronous-snap trigger's floor, or relax its radius, for
main-cluster candidates.** The 102-point discarded candidate never even
attempted a snap because it was short. A knob widening `10cm`/`8cm+13cm`
(`:633-634`) or the 6 cm snap radius (`:652,656`) specifically for the main
cluster would let more of these short residual pieces attach to the existing
graph instead of being thrown away outright — but risks pulling in genuinely
disconnected noise; needs the F0 census to size.

**F2 — separate "isolated but well-supported" from "isolated and sparse" at
the discard point.** A residual component with many steiner terminals (like
this one's 465) but no nearby existing segment is different from a
low-terminal noise fragment; a knob adding a terminal-count floor before
allowing the isolated-residual discard (instead of relying only on
`dir_mag`/`length`) would target exactly the owner's "gaps due to dead
channels / SP mistakes leave real structure isolated" framing from doc pr/53's
opening context, applied here to the fit stage rather than the connectivity
stage.

**F3 — loosen `accept_relaxed`/`accept_proto` for main-cluster candidates that
fail no other test.** Lower priority: this event's 4-5 `oseg_reject`s are a
different (and already-audited) path; F0's census should confirm whether any
of them, not just the isolated-residual discards, are attributable to this
component before touching these thresholds.

Recommend F0 first (pure instrumentation, closest to risk-free), then F2.
Record F1 and F3 as candidates, not commitments — none is sized without F0's
census.

## 9. Open items

- Coordinate-level attribution of the specific discarded candidate(s) to the
  owner's exact component requires F0's instrumentation; not available this
  session (would be a code change).
- Whether this class (separated, well-steiner-supported, terminal-graph-
  fragmented residual EM showers near a main cluster) recurs across other
  SBND events is unmeasured — F0's census is also what would answer that.
- Not investigated: whether the owner's "missing gammas" framing implies a
  downstream physics effect (missed shower energy in `kine_reco_Enu` etc.) —
  out of scope for this doc, which is about the fit-stage mechanism only.

## Verification

No build, no A/B gate — nothing executable changed. `oc54_probe.py`'s numbers
are reproducible from the Repro block above; the TRACE-log lines and PR30AUDIT
line are already-existing products, quoted verbatim with their source files.

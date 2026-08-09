# doc pr/54 — 18255-142421: a separated EM shower gets no fitted trajectory (silently discarded, not silently skipped)

Status: fix SHIPPED as the default-OFF knob `other_seg_keep_isolated`
(round 3, §12 below) + unconditional visibility counters; off-path proven
byte-identical on both SBND manifests; production flip pending owner review
of the Bee before/after pair. §§1-11 are the original investigation record,
unchanged.

**Round 2 (below, §10):** this is one bug, not three. §8's F1-F3 are three
untested hypotheses for *which* of two discard paths caused this event's
component to be dropped, not three separate defects — F0 (visibility) is what
would actually distinguish them. Separately, the same "fit computed, then
silently discarded" pattern recurs in **at least six more places** across the
fit stage, one of them (`init_first_segment`) more severe than either path
found in §5, since its failure can silently zero out an entire cluster's PR
output. See §10 for the full census.

**Round 3 (below, §12):** the fix. §8's F0 (visibility counters, shipped
unconditional) + a F2-shaped keep knob `other_seg_keep_isolated`, DEFAULT
OFF. M15 check: the prototype *also* discards these candidates — its
`residual_segment_candidates` accumulator is write-only, never consumed
anywhere — so this is a toolkit-only extension of an unfinished prototype
feature, not a parity fix. On 142421 the knob recovers the component (fit
distance min 3.65 → 0.05 cm) and, in this event, is a strict superset: 44
other clusters byte-identical, every pre-existing cluster-7 fit point
preserved exactly, nusel unchanged. Bee before/after links in §12.

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

# Round 2, §10 census -- fit-then-discard sites elsewhere in the fit stage:
sed -n '1100,1161p' clus/src/NeutrinoPatternBase.cxx     # init_first_segment: uncounted, unlogged discard
sed -n '1928,2125p' clus/src/NeutrinoVertexFinder.cxx    # eliminate_short_vertex_activities
sed -n '2745,2765p' clus/src/TaggerCheckSTM.cxx          # search_other_tracks
sed -n '3080,3095p' clus/src/TaggerCheckSTM.cxx          # check_stm_conditions forward pass
sed -n '8615,8645p' clus/src/TrackFitting.cxx            # do_multi_tracking point-level drop
sed -n '9015,9045p' clus/src/TrackFitting.cxx            # do_single_tracking fit-output size-mismatch guard
sed -n '39,68p' clus/inc/WireCellClus/PRGraph.h          # PortAuditCounters -- full existing counter list

# Round 3 -- the fix (toolkit built with wcbuild, freshness-proven):
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
SBND_OTHER_SEG_KEEP_ISOLATED=true PR_JOBS=1 \
    ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr54-on142421 data 142421
grep -a "pr54 keep-isolated\|pr54 isolated-residual drop\|PR30AUDIT" \
    work-pr54-on142421/pr_evt142421/wct_pr_evt142421.log
python3 scripts/analysis/pr54/oc54_after.py         # recovery + superset check
python3 scripts/analysis/pr54/plot_142421_after.py  # after figure

# Round 3 off-gates:
PR_JOBS=5 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr54-off48a data
PR_JOBS=6 ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr54-off50a data \
    $(awk 'NR>1{print $2}' docs/pr/mcp1k-50-cb0805.index.txt)
python3 scripts/analysis/pr49/on_compare.py work-pr51-off48e work-pr54-off48a
python3 scripts/analysis/pr49/on_compare.py work-pr51-off50e work-pr54-off50a

# Round 3 Bee pair:
python3 scripts/bee/make_pr_bee.py -q work-ncpi0-cb0805 -p work-bee-0809       -o bee/pr54/pr54-before.zip 142421
python3 scripts/bee/make_pr_bee.py -q work-ncpi0-cb0805 -p work-pr54-on142421  -o bee/pr54/pr54-after.zip  142421
./upload-to-bee.sh bee/pr54/pr54-before.zip   # -> set/36539378-7352-4ba0-ae76-6a0f0afc03a6
./upload-to-bee.sh bee/pr54/pr54-after.zip    # -> set/d8639dc6-f6b6-4ddd-9a9a-79cbd8ee1eb0

# Round 3 prototype dead-accumulator proof (M15):
grep -rn "residual_segment_candidates" /nfs/data/1/xqian/prototype-dev/wire-cell/   # 3 hits, no consumer
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

This is **one bug** — real structure gets a trajectory fit computed and then
discarded with no way to see or count it — manifesting through two discard
points found in §5. F1-F3 below are not three separate defects; they are
three untested hypotheses for which of the two paths (or both) is responsible
for *this* event's component, offered because attribution could not be proven
without new logging (§5's caveat). F0 is what actually settles the question;
F1-F3 should not be picked without it. Every knob below is default-OFF, its
off-path leaving the compiled config and graph byte-identical; gate = the
SBND 48/50-event manifests used by pr/48-51.

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

## 10. Round 2 (2026-08-09) — the fit-then-discard pattern recurs at (at least) six more sites

The owner asked, after §5-§8 above: is this really more than one bug, and does
this class of silent discard happen anywhere else? Answer to the first: no,
one bug (see the note added to §8). Answer to the second: yes. A read-only
survey of every other call site of `TrackFitting::do_single_tracking` /
`do_multi_tracking` in `clus/src/` (`grep -rln "do_single_tracking\|
do_multi_tracking" clus/src/*.cxx`) found six more sites with the same shape
— a real fit computed, then the candidate conditionally thrown away based on
a quality/length/isolation test, with little or no way to see it happened.
None of them is covered by any of the 16 existing `PortAuditCounters`
(`clus/inc/WireCellClus/PRGraph.h:39-68`: `oov_isochronous/dead_scan/
unique_scan`, `add_segment_calls/reentry`, `endpoint_mismatch/refused`,
`pca_refine_calls/moved/move_um_sum/max`, `oseg_accept_proto/accept_relaxed/
reject`, `selfloop_segment`, `edge_aliased`) — confirmed by direct read of
each header field.

| # | site | trigger | visibility |
|---|---|---|---|
| 1 | `init_first_segment`, `NeutrinoPatternBase.cxx:1122-1159` | `fine_path.size() <= 1` after `do_single_tracking` → `remove_segment`+`remove_vertex`×2; comment reads `// Tracking failed, clean up` | **zero log, zero counter.** Worst of the set: this is the *first* segment fit attempted for a cluster (`init_first_segment` is called from `find_proto_vertex` before `find_other_segments` ever runs, `NeutrinoPatternBase.cxx:2388-2391`), and there is no fallback for main/long clusters at the `TaggerCheckNeutrino.cxx:991,1029` call sites — so a failure here can silently zero out an entire cluster's PR output with no trace of why. |
| 2 | `eliminate_short_vertex_activities`, `NeutrinoVertexFinder.cxx:1928-2125` | 4 length/isolation cuts (`<0.36cm`, `<0.5cm && degree>3`, `<0.1cm` near main vertex, isolated-vertex `n_good==0`) on stub segments just refit via `do_multi_tracking` (`:2724`) | zero log, zero counter |
| 3 | `search_other_tracks`, `TaggerCheckSTM.cxx:2758-2762` | `fits().size() <= 1` after `do_single_tracking` → never added | zero log, zero counter |
| 4 | `check_stm_conditions` forward pass, `TaggerCheckSTM.cxx:3087-3093` | `fits().size() <= 3` → candidate dropped, function returns false | DEBUG log only (cluster id + fit size), no counter |
| 5 | `do_multi_tracking` point drop, `TrackFitting.cxx:8624-8642` | zero-plane-quantity trajectory points dropped (point-level, not whole-segment) | comment says "count so it is never silent" — only a DEBUG line (`n_fits_before/after`) actually exists, **no counter despite the comment's intent** |
| 6 | `do_single_tracking` output guard, `TrackFitting.cxx:9028-9042` | fit-output size mismatch clears the fit (feeds sites 1 and 3's size checks) | WARN log (best of the set) — still no counter, so occurrences can't be aggregated across an event |

`NeutrinoGraphAudit.cxx` and `NeutrinoStructureExaminer.cxx`'s `do_multi_tracking`
call sites were checked and are a **different** shape — they re-fit *after* a
topology decision is already made, and their removals are separately counted
(`n_op1/2/3`) — not the same pattern, not added to this table.

**Read on F0:** generalizes directly. The same shape (candidate fit, then
conditionally discarded) recurs at 8 total sites now (2 in §5 + 6 here); a
single shared counter/logging convention — not eight bespoke ones — is the
right shape for a future fix session, rather than patching each site
independently.

## 11. Open items

- Coordinate-level attribution of the specific discarded candidate(s) to the
  owner's exact component requires F0's instrumentation; not available this
  session (would be a code change).
- Whether this class (separated, well-steiner-supported, terminal-graph-
  fragmented residual EM showers near a main cluster) recurs across other
  SBND events is unmeasured — F0's census is also what would answer that.
- Not investigated: whether the owner's "missing gammas" framing implies a
  downstream physics effect (missed shower energy in `kine_reco_Enu` etc.) —
  out of scope for this doc, which is about the fit-stage mechanism only.
- §10's census is from a read-only grep-and-read survey (one Explore pass),
  not an exhaustive line-by-line audit of every `remove_vertex`/`remove_segment`
  call in `clus/src/` — it is scoped to call sites reachable from the trajectory
  fitter. A broader audit of every silent-discard shape in `clus/` (not just
  fit-adjacent ones) was not attempted.

## 12. Round 3 (2026-08-09) — the fix: `other_seg_keep_isolated`, DEFAULT OFF

The owner asked for the fix, a Bee demonstration on 142421, and an answer to
"does this only add fitted points, without changing other clusters' fits?".

### M15 check first — the prototype has the same discard, unfinished

The prototype's isolated-residual branch
(`NeutrinoID_proto_vertex.h:1470-1475`) does not simply delete the candidate:
it pushes `(cluster, v1_wcpt, v2_wcpt)` into `residual_segment_candidates` —
and that accumulator is **write-only**: 3 grep hits in the whole prototype
(declaration `NeutrinoID.h:2022`, the push, one doc line), zero consumers. So
WCP discards these candidates too ("missing gammas" is a prototype behaviour
as well), the toolkit port is faithful, and this fix is a toolkit-only
extension of an unfinished prototype feature — not a parity correction. Cited
inline at the branch.

### The knob

`other_seg_keep_isolated` (C++ default **false**) with floors
`other_seg_keep_isolated_min_points` (default 25, terminal-graph component
points) and `other_seg_keep_isolated_min_length` (default 3 cm, fitted track
length). When ON and both floors pass, the isolated candidate is added to the
cluster's graph as its own disconnected piece (its two endpoint vertices) and
the cluster is refit jointly — exactly the shape of the accepted isochronous
branch, including the same curvature/dQdx classification into the
break-segments lists. Below either floor (the 3/4-point noise fragments), or
with the knob off, the legacy discard runs unchanged. Keep decision is the
doctest-covered free predicate `other_seg_keep_isolated_ok`
(`PRSegmentFunctions.h`, defined in `NeutrinoOtherSegments.cxx`).

§8's F0 ships alongside, unconditional (pure log/counter, cannot change
outputs): new `PortAuditCounters` fields `oseg_isolated_drop` /
`oseg_isolated_keep` on the PR30AUDIT line, an INFO sentinel
(`pr54 keep-isolated:`) at every keep, and a DEBUG line
(`pr54 isolated-residual drop:`) with point count / length / endpoints at
every discard — closing §5's "no counter at all" gap and giving the
detector-wide census for free in every future arm. §8's F1 and F3 were not
taken (nothing in the demo needed them).

Threading: `TaggerCheckNeutrino.{h,cxx}` → `NeutrinoPatternBase.h` members →
`NeutrinoOtherSegments.cxx`; jsonnet through `pgrapher/common/clus.jsonnet`'s
`tagger_check_neutrino` builder (key-suppression idiom),
`sbnd/clus.jsonnet` (both blocks), `sbnd/wct-pr-perevt.jsonnet` (TLA), and
the `SBND_OTHER_SEG_KEEP_ISOLATED[_MIN_POINTS,_MIN_LENGTH]` runner envs in
`run_pr_chain_batch.sh`.

**Commit provenance note.** A concurrent session committed pr/51 round 4
(toolkit `a46a0b22`, wcp `39b52fd`, 10:27) while this round's edits were in
flight, and — both knobs thread through the same files — swept this round's
already-edited plumbing (`TaggerCheckNeutrino.{h,cxx}`,
`NeutrinoPatternBase.h` members, the three jsonnet threading files, the
runner env hook) into those commits. The pr/54 logic itself — counters,
`other_seg_keep_isolated_ok` predicate, the keep/drop branch, the doctest —
is toolkit `091815c4`. Nothing was lost or double-committed; the validated
binary == `a46a0b22` + `091815c4`.

### Demonstration on 18255-142421 (arm `work-pr54-on142421`, knob ON)

The knob fires twice for cluster 7 — sentinels:

```
pr54 keep-isolated: cluster 7 n_points=664 length=98.05 cm v1=(114.9,-73.2,213.2) v2=(66.4,-83.7,274.1) cm
pr54 keep-isolated: cluster 7 n_points=47  length=10.78 cm v1=(94.5,-74.4,235.1)  v2=(88.0,-74.4,238.1) cm
```

and still drops the sparse fragments (5 drops, all 3-4 points; PR30AUDIT:
`oseg_iso_drop=5 oseg_iso_keep=2`). Note the round-0 partition itself changes
once earlier keeps re-tag the residual set: the bare arm's discarded
102-point candidate becomes a kept 664-point / 98 cm candidate spanning the
region.

Recovery (`oc54_after.py`, vs the §3 baseline):

| metric | bare (`work-bee-0809`) | knob ON |
|---|---|---|
| component→fit min | 3.65 cm | **0.05 cm** |
| component→fit median | 9.39 cm | **1.25 cm** |
| component frac >5 cm from any fit | 0.974 | **0.005** |
| owner point→fit min | 8.61 cm | **0.91 cm** |

Figures: `54_142421_blob.png` (before) / `54_142421_blob_after.png` (after).

### The superset question, answered empirically

Per-cluster diff of `track_fit-global` bare vs knob-on (`oc54_after.py`):
**44 of 46 cluster ids byte-unchanged**; only cluster 7 (808 → 1256 fit
points) and the id −1 bucket (102 → 123) change, and in both **every
pre-existing point survives at 0.000 cm displacement** — pure additions, in
this event. `clustering-global` identical; `nusel-evt142421.tsv` identical.
Caveat stated honestly: this strict-superset outcome is *not* a structural
guarantee — the joint `do_multi_tracking` refit and the downstream
vertex/shower stages may move existing cluster-internal fits in other events
(other clusters stay per-cluster-isolated by construction). The sample-wide
knob-on census (deferred, below) is what sizes that.

### Bee before/after (the owner's requested demonstration)

- before (bare, `work-bee-0809`):
  <https://www.phy.bnl.gov/twister/bee/set/36539378-7352-4ba0-ae76-6a0f0afc03a6/event/list/>
- after (`other_seg_keep_isolated=true`):
  <https://www.phy.bnl.gov/twister/bee/set/d8639dc6-f6b6-4ddd-9a9a-79cbd8ee1eb0/event/list/>

### Verification (round 3)

- Compiled-config proofs (runner TLA set incl. full `pipeline_names`,
  `reality=data`): knob-off compile byte-identical (`cmp`) to pristine-HEAD
  cfg; knob-on differs in exactly the one key (`other_seg_keep_isolated`
  appears once, zero when off).
- `wcdoctest-clus` 1284/1284 (133 cases; new:
  `doctest_other_seg_keep_isolated.cxx`, 2 cases / 9 assertions — OFF-default
  pin, floor logic, inclusive boundaries).
- Off-gates (knob compiled in, OFF; freshness-proven binary):
  `work-pr54-off48a` vs `work-pr51-off48e` — **0/48** archives differ, nusel
  0/48 both granularities; `work-pr54-off50a` vs `work-pr51-off50e` —
  **0/50**, nusel 0/50. (References are pr/51 round-3's validated baselines,
  themselves 0/48 & 0/50 against `work-pr50-snap48a`/`snap50a`.)
- Demo run `work-pr54-on142421`: batch ok 1/0 failed, sentinels + PR30AUDIT
  quoted above.

### Open items (round 3)

- Sample-wide knob-on census (48+50 manifests) deferred until after the owner
  hand-scans the Bee pair — staged-small-group validation. The
  `oseg_iso_drop` counter now measures the discard rate detector-wide in
  every arm for free.
- The 25-point / 3 cm floors are first-cut: they cleanly separate this
  event's 664/47-point keeps from its 3-4-point drops, but the census should
  confirm the margin on other events before any flip.
- §10's six other fit-then-discard sites remain uncounted (only the
  find_other_segments pair got counters this round).

## Verification

Rounds 1-2 (investigation, §§1-11): no build, no A/B gate — nothing
executable changed; `oc54_probe.py`'s numbers are reproducible from the Repro
block above; the TRACE-log lines and PR30AUDIT line are already-existing
products, quoted verbatim with their source files.
Round 3 (the fix): see §12's Verification subsection — off-gates 0/48 + 0/50
byte-identical, doctests, compiled-config proofs, demo arm.

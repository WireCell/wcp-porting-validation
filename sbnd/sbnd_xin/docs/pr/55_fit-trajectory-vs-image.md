# doc pr/55 — fitted trajectory vs. image: three over-clustering symptoms the doc pr/53 round-7 fix does not reach

Status: **diagnosis only, this session — no fix, no new knob, no default
change.** Owner's instruction: "let's worry about fix in the next session,
but we should really understand the issue in this session." Two DEBUG-only
sentinel log lines were added (§2) to confirm the root cause at runtime;
proven log-only (§2, hash + nusel comparison). Every symptom below is
measured and case-cataloged (`docs/pr/pr55-cases.tsv`, 293 rows, 46 figures
in `pics/55_*.png`) so a future fix session can start from numbers, not a
re-scan.

**Headline finding**: the owner asked, directly, whether the PR/fitting chain
actually runs on the tightened membership graph doc pr/53 shipped
(`relaxed_strict_img`). **It does not.** The trajectory router
(`do_rough_path`) runs on `"steiner_graph"` — a graph descended from
`ctpc_ref_pid`, unrelated to `protect_bundle`'s `graph_name` knob, and
strictly *more* permissive (it carries the uncapped-MST `connect_graph`
bridges doc pr/53's whole family of fixes exists to avoid). Measured directly
on the owner's own flagged case (269774, cid 13 seg 13042): replaying
*today's shipped* `relaxed_strict_img` predicate on this exact gap says
**kill** (`kill_base=True, kill_s5=True`, §4.2) — the membership rule would
reject this connection outright — yet the production fitted trajectory
crosses it anyway, because the fitter never asks the membership graph at all.
Doc pr/53 tightened *which blobs merge*; none of the three symptoms below
live in that layer.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# metrics + case catalogue (reads the existing production arms, no rerun needed
# for the numbers -- see docs/pr/pr55-arms.txt for exactly which):
python3 scripts/analysis/pr55/pr55_metrics.py docs/pr --label production

# figures (46 total: 3 event overviews + headline per-case 3-panel figures):
python3 scripts/analysis/pr55/pr55_figures.py pics

# the two new DEBUG sentinels (needs the instrumented rebuild + a fresh run --
# see docs/pr/pr55-arms.txt for the exact commands and existing arm labels):
grep -a "pr55 do_rough_path" work-pr55-instr19b/pr_evt142421/wct_pr_evt142421.log | head -3
grep -a "pr55 shower_track layer" work-pr55-instr19b/pr_evt142421/wct_pr_evt142421.log
```

## 1. The owner's report

Hand-scan of the doc pr/53 round-7 Bee production set
(`bee/pr53r7/pr53r7-after.zip`, current SBND default `relaxed_strict_img`):

1. **142421**, `(89.5, -70.0, 243.7)`: a large EM shower with **no fitted
   track trajectory in it**; also a **long track connecting a small isolated
   piece to that shower, with visibly high dQ/dx**
   (`Screenshot 2026-08-09 at 6.06.06 PM.png`).
2. **269774**: in the vertex region, **three separate image blobs joined by
   a fitted trajectory** — "the previous fix is supposed to avoid this?"
   (`Screenshot 2026-08-09 at 6.05.47 PM.png`).
3. **71372**: **many fitted points not backed by image**, near the neutrino
   vertex (`Screenshot 2026-08-09 at 6.06.48 PM.png`).

The owner's own comparison is explicitly **`track_fit` vs. `shower_track`**
— the fitted trajectory layer against the *associated-points* layer, not
against the raw 3D image (`clustering-global`). That distinction turned out
to matter (§3).

## 2. Q1 — which graph does the PR chain actually run on?

Traced in full (`clus/src/NeutrinoPatternBase.cxx`, `CreateSteinerGraph.cxx`,
`Facade_Cluster.cxx`, `SteinerGrapher.{h,cxx}`, `TrackFitting.cxx`):

- `ClusteringProtectBundle` really does perform a structural split —
  `Grouping::separate()` physically moves cut blobs into new `Cluster`
  objects and purges every cached graph product on the retained cluster
  (`take_graph`/`remove_graph_algorithms`, `steiner_pc` erased). Nothing
  downstream can reuse a pre-split graph; caches are keyed by flavor name.
- But `do_rough_path` — the only trajectory router in the PR chain — never
  asks for `relaxed`, `relaxed_strict`, or `relaxed_strict_img`. It runs
  Dijkstra on `"steiner_graph"` (`NeutrinoPatternBase.cxx:111`), built by
  `CreateSteinerGraph.cxx:227-262` as a Steiner reduction of
  `"ctpc_ref_pid"` = `closely_pid` + `connect_graph_ctpc_with_reference` +
  **`connect_graph_with_reference`** (the uncapped MST). The fitter's
  neighbor-blob gather (`TrackFitting.cxx:2167`) uses `"basic_pid"`, also
  MST-bearing. Neither has ever been touched by any doc pr/53 round.
  `init_point_segment` (short ≤6cm clusters only) genuinely does use the
  legacy `"relaxed_pid"` flavor (`NeutrinoPatternBase.cxx:2666`).
- A second leak path, above the cluster layer entirely: `TaggerCheckNeutrino`
  runs every associated cluster — including `protect_bundle`'s own split-off
  fragments — into **one** `PR::Graph`
  (`TaggerCheckNeutrino.cxx:782,1045-1049`), and `NeutrinoShowerClustering`
  re-attaches foreign-cluster segments to main-cluster showers on purely
  geometric/angular criteria (`NeutrinoShowerClustering.cxx:723,801,
  860-864,1221-1253`), with no graph flavor involved. A membership split
  survives as separate *segments*, but those pieces can still be re-absorbed
  into the neutrino as shower daughters.

**Runtime confirmation** (new sentinel, `NeutrinoPatternBase.cxx`, DEBUG,
`"pr55 do_rough_path"`): fired 104/249/278 times on 142421/71372/269774
respectively, e.g.

```
[15:39:51.110] D [clus.NeutrinoPattern] pr55 do_rough_path: cluster 7 flavor=steiner_graph
  n_vertices=4954 n_edges=12010 first=(979.8,-1106.5,4266.8) last=(1179.9,-697.8,2085.8)
```

confirming `steiner_graph` (not any `relaxed*` flavor) is what every routed
path in these three events actually used.

A second sentinel (`MultiAlgBlobClustering.cxx`, DEBUG, `"pr55 shower_track
layer"`) targets a previously-silent path relevant to §5: a segment whose
`associate_points` dpcloud is null or empty contributes **zero** points to
`shower_track` with no log at all before this change. It fired once on
142421 (seg 7020, §5) and once on 71372 — see §5 for why this matters.

**Log-only proof** (both sentinels, all three events): `hash_archive.py`
match on `mabc-pr.zip` and `pctree-pr-evt<N>.tar.gz` between the
pre-instrumentation production arms and a fresh instrumented rerun, plus a
byte-identical `nusel-evt<N>.tsv` diff. All six comparisons SAME (`docs/pr/
pr55-arms.txt` has the exact commands and hashes are reproducible from
there). `wcdoctest-clus` 141/141 test cases, 1505/1505 assertions.

## 3. The metrics (Families A–D)

All three point layers are read from each event's own **PR-stage**
`pr_evt<N>/mabc-pr.zip` (unremapped `cluster_id` — *not* the Bee-upload copy,
whose `cluster_id` is rewritten into img-global's id space, per
`make_pr_bee.py`'s own docstring):

- **IMG** = `0-clustering-global.json`, the 3D image points.
- **ASC** = `0-shower_track-global.json`, points *associated* to each PR
  segment (`q=0` painted track, `q=15000` painted shower).
- **FIT** = `0-track_fit-global.json`, the *fitted* trajectory
  (`q = dQ·0.1 − 1000`, a dQ/dx proxy). Both ASC/FIT carry
  `real_cluster_id = cluster*1000 + segment` (called `seg` throughout, per
  `fitgap_exam.py`'s precedent, to avoid confusion with the image cluster
  id).

**Family A — ghost run.** A maximal contiguous stretch of one segment's
fitted polyline (storage order = trajectory order) farther than **1.0 cm**
(same radius as the shipped `relaxed_strict_img::m_img_radius`) from the
nearest IMG point of *that fit point's own final cluster*. Case ID
`<evt>-G<k>`. Per case: gap length/point count/max distance, `dead_frac`
(fraction of samples excused by a dead channel — separates a legitimate
dead-region bridge from a real void), `q_ratio` (median dQ/dx inside the run
÷ outside it — the owner's "high dQ/dx" observation, made measurable),
`foreign_frac`/`foreign_cids` (does the run's nearest-ANY-cluster neighbor
belong to a *different* final cluster — the direct "is this trajectory
secretly tracking a foreign blob" signal), and `strict_verdict` (§4.2).

**Family B — uncovered shower.** A segment's ASC points sit far from the
FITTED polyline of **the segment's own final cluster** (union of every
segment's fit points in that cluster — see the correction note below).
Flagged when `uncov_frac3 > 0.3` (fraction of ASC points >3cm from any fit
point) and `assoc_pts >= 100`. Case ID `<evt>-U<k>`.

**Family C — phantom segment.** A segment has `fit_pts > 0` but
`assoc_pts == 0` under its own id — a trajectory drawn with (apparently)
nothing associated under it. Case ID `<evt>-P<k>`. "Material" (flagged for a
figure) at `fit_len_cm >= 10` or `n_pts >= 20`; the full census (most
material-false, short leftover stubs) stays in the TSV only.

**Family D — honesty label**, applied to every A/C case:
- `REAL-VOID` — no image nearby, not dead-excused.
- `DEAD-BRIDGE` — dead-channel excused.
- `IMG-ELSEWHERE` — image exists nearby but belongs to a *different* final
  cluster (an addition beyond the plan's original three labels, made because
  it is a materially different situation from a true void).
- `PAINT-ONLY` — image is present within ~1.5cm of the segment's own cluster
  but not associated into that segment's ASC points.
- `MISATTRIBUTED-SHOWER-ID` — Family C only, see the correction below.

### Two corrections, found empirically, that changed the whole picture

Both Family B and (especially) Family C, defined naively as "compare ASC(seg)
to FIT(same seg)", are **contaminated by a display-layer id-reassignment
convention** in `MultiAlgBlobClustering.cxx`'s `shower_track` writer: a
shower *member* segment's `associate_points` are filed under its **shower's
start segment's id**, not the member's own id (`"Shower membership is the
authoritative shower-vs-track answer..."`, `MultiAlgBlobClustering.cxx:858`).
Caught by cross-checking the new sentinel against the naive Family C count:
of 142421's four id-match phantom candidates (segs 7011/7018/7110/7020),
**the sentinel fires only once** (seg 7020) — the other three are 100%
covered by ASC points filed under seg 7109 (their shower's start segment),
confirmed directly:

```
seg 7011  fit_pts 35   n(assoc within 1cm, ANY seg)=35   all attributed to real_cluster_id 7109
seg 7018  fit_pts 14   n(assoc within 1cm, ANY seg)=14   all attributed to real_cluster_id 7109
seg 7110  fit_pts 11   n(assoc within 1cm, ANY seg)=11   attributed to 7109 (8) / 7019 (3)
seg 7020  fit_pts 109  n(assoc within 1cm, ANY seg)=7    mostly UNCOVERED -- matches the sentinel firing
```

Family C's classifier now checks coverage by **any** ASC point (any seg id),
not just the exact id match, and labels accordingly (`attribution` column).
Result across all three events: **97% of Family C's 260 id-mismatch cases
are `MISATTRIBUTED-SHOWER-ID`** — a display convention, not a reconstruction
defect — leaving only **6 genuine `PAINT-ONLY`/`REAL-VOID` phantom segments**
total (2 per event). Family B has the mirror problem (a shower's pooled ASC
points compared against only its *start segment's own tiny local fit*, not
the whole shower's fitted trajectory) and is corrected the same way: the FIT
reference is the union of every segment's fit points in the **same final
cluster**, not just the same seg id (verified on 71372 seg 19065: naive
comparison gave `cov_med=49.6cm, uncov_frac3=0.99`; cluster-level comparison
gives `cov_med=3.4cm, uncov_frac3=0.52` — still a real, substantial gap, but
now a fair one).

**This correction is itself a finding worth flagging for a future session**:
the display layer's shower-id convention makes a Bee hand-scan systematically
over-report "fit with nothing under it" — most of what looks like a phantom
trajectory in a raw viewer is charge correctly associated to the shower, just
labeled under a different segment's id than the one drawing the line nearby.

## 4. Case catalogue — summary

293 cases total (`docs/pr/pr55-cases.tsv`); 46 figures in `pics/55_*.png` (3
event overviews + every Family A/B case + every owner-tagged
non-misattributed case + every material Family C case, capped at 2
illustrative `MISATTRIBUTED-SHOWER-ID` examples per event — that class is one
explained mechanism, not 260 individual defects; see `pr55_figures.py`'s
`headline_cases()`).

| event | Family A (ghost) | Family B (uncovered) | Family C (phantom) | material C | owner-tagged |
|---|---|---|---|---|---|
| 142421 | 3 | 4 | 47 | 8 | 3 |
| 269774 | 6 | 3 | 123 | 15 | 39 |
| 71372  | 14 | 3 | 90 | 15 | 11 |

Family D label counts: `IMG-ELSEWHERE` 21, `REAL-VOID` 1, `DEAD-BRIDGE` 1
(all Family A); `MISATTRIBUTED-SHOWER-ID` 254, `PAINT-ONLY` 6 (all Family C).
Family A's `strict_verdict` (§4.2, would today's shipped `relaxed_strict_img`
kill a direct edge here?): **9/23 = True** — for those 9, the membership rule
would already reject the connection if it were ever tested as a graph edge;
it never is, because these gaps sit *inside* the fitted trajectory of a
single already-formed segment (§2's leak, not a graph-membership question at
all for most of them).

### 4.1 142421 — "shower with no fitted trajectory" + "high-dQ/dx connector"

**Both owner complaints are one object.** The owner's point `(89.5, -70.0,
243.7)` is 7.1cm from the centroid of Family B case **142421-U1** (seg 7010,
cid 7): 3372 associated points, `cov_med=6.6cm`, `uncov_frac3=0.80` — the
fitted trajectory (104 points, 64.0cm) covers barely any of the associate
cloud (`pics/55_142421-U1.png`). That same segment's fit contains Family A
case **142421-G1**: a 22.2cm ghost run, `d_max=10.8cm`, whose `q` (dQ/dx
proxy) rises **3765 → 14387** almost exactly in step with the distance to
image (`pics/55_142421-G1.png` — the profile panel overlays both curves and
they visibly track each other). Mechanism, directly from the code
(`TrackFitting.cxx` dQ/dx solve): a trajectory point with no plane
association is constrained only by the regularizer, which couples neighbors
to each other with no absolute charge target — an unconstrained run inherits
whatever its charged neighbors solve to, producing the inflated value the
owner correctly flagged as "very strange."

**142421-G1's `foreign_cids=[53]`**: a few of this run's samples' nearest
image point belongs to **cluster 53** — the "small isolated piece" the owner
describes the track connecting to. cid 7 and cid 53 are separate final
clusters both before and after doc pr/53 round 7 (matches that doc's own
residual note); this is §2's leak path 3 (PR::Graph-level segment behavior,
independent of cluster membership) operating exactly as traced.

**Provenance**: segment 7010's endpoints (`v1=(108.9,-54.6,248.9)
v2=(68.3,-79.5,261.4)`, length 64.03cm) match the production log line `pr54
keep-isolated: cluster 7 n_points=564 length=64.03cm` exactly — this is the
`other_seg_keep_isolated` residual the owner flipped ON the same day as doc
pr/53 round 7 (doc pr/54). The kept segment's own fit does not follow the
shower it was kept for.

### 4.2 269774 — "three isolated blobs joined by a trajectory"

Owner point pair A `(95.1,16.6,137.0)`/`(84.8,10.3,136.4)` → Family A case
**269774-G1** (cid 13, seg 13042): an 8.4cm ghost run, `d_max=4.9cm`
(`pics/55_269774-G1.png`) — the fitted trajectory visibly spikes off the
image into empty space and returns, matching the screenshot exactly.

**This is the case that answers the owner's direct question with a number**:
replaying today's shipped `relaxed_strict_img` predicate on this run's own
endpoints gives **`kill_base=True` (round-6 S1-S3) AND `kill_s5=True`
(round-7 S5, `max_ghost_run=8 >= 4`, `dis=8.79cm < 15cm`)** — the membership
rule, as shipped, says this connection should be rejected. It still appears
in production because, per §2, nothing in the trajectory-fitting chain ever
asks the membership graph.

Owner pair B `(45.7,-43.8,154.0)`/`(39.5,-59.1,163.7)` → Family B case
**269774-U2**, and (in the round-6 comparison arm only, §5) a second Family A
ghost run that round 7's S5 rule *did* close (§5) — matching doc pr/53's own
residual note that pair B was one of the routed/fixed cases while pair A was
not.

### 4.3 71372 — "many fitted points not backed by image"

The owner's three named points sit in a dense cluster of **14 Family A ghost
runs** across 8 segments near the neutrino vertex (cid 19/27/101/113/124),
each individually small (0.6–5.4cm) but visually numerous —
`pics/55_71372_overview.png` shows the congestion matching the screenshot.
Five of these route within 7cm of owner point p3 (segs 19017/19019/19038,
`pics/55_71372-G{2,3,4,7,10}.png`); three have `kill_any=True` (the strict
rule would reject a direct edge there too). The single largest ghost run in
the whole event, case **71372-G1** (seg 101170, 5.4cm, `d_max=2.9cm`,
`kill_any=True`), sits ~82cm away in a different part of the same final
cluster — not one of the owner's three points, but the same mechanism,
included in the catalogue for completeness. This matches the owner's separate
note that 71372 has "~4 clusters over-clustered near the vertex" more broadly
than the 3 points they listed — consistent with doc pr/53 round 7's own
residual §18.7, which reported 4 of 18 fit-gap runs in this event as
unrouted to any membership edge (the same §2 mechanism, not a new class).

## 5. Attribution check — do doc pr/53's own knobs move any of this?

Using the **existing** round-6 (`relaxed_strict`, S5 off,
`work-pr53r7-off19`/`off48`) and round-7 (`relaxed_strict_img`, current
production, `work-pr53r7-on19`/`on48`) arms already on disk (no rerun) —
re-deriving Family A/B/C cases in both and comparing every owner-tagged case:

```
142421 owner-tagged cases: IDENTICAL in both arms (G2/U1/P15≈P16, same L, same label)
269774 owner-tagged (pairA): IDENTICAL (G1, L=7.8→8.4cm -- small fit re-solve shift, same mechanism)
269774 owner-tagged (pairB): round-6 arm ALSO has a Family A ghost run (G2, L=6.6cm) that
   is ABSENT from round-7 production -- the one case doc pr/53 round 7 actually closed,
   matching that doc's own routed-case claim for this pair
71372 owner-tagged (p1/p2/p3): IDENTICAL in both arms (same 5 ghost runs, same lengths)
```

Confirms §2 empirically, independent of the code trace: doc pr/53's
membership-graph tightening (rounds 6 and 7 both) leaves essentially every
owner-flagged symptom in this doc unchanged, with the single exception of the
one case (269774 pair B) that doc pr/53 round 7 itself already claimed to
have fixed.

## 6. Not fixed this session — candidates for the next one

Per the owner's explicit scope, no fix was attempted. Candidate levers,
listed without picking one:

- **A trajectory-side gap check**, analogous to `relaxed_img_bad` but applied
  along the *fitted path itself* (not a membership graph edge) — would reach
  §4.1's seg 7010 and §4.3's small vertex-region runs, which never go through
  `protect_bundle`'s graph at all.
- **Threading a membership-aware penalty into `steiner_graph`/`ctpc_ref_pid`**
  (Design A `steiner_gap_penalty` from doc pr/51 round 4, not shipped) — would
  reach §4.2's 269774-G1 directly, since it is a genuine `do_rough_path`
  Dijkstra routing decision.
- **Re-scoping `NeutrinoShowerClustering`'s foreign-segment absorption**
  (§2 leak path 3) to respect `protect_bundle`'s split — larger, riskier
  change; touches every shower-formation site, not just these three events.
- **A visible counter for `MultiAlgBlobClustering`'s empty-`fits()`/empty-
  `associate_points` paths** (this session added log lines only, per owner's
  "printout" request) — promoting them to a `PortAuditCounters`-style field
  would make §5's misattribution class and true phantom segments countable
  at population scale instead of read from a log grep.
- No scope, cap, or threshold judgment was made on any of these — reported
  for the next session's decision, per the escalation rules.

## 7. Method notes / gotchas (carried forward for the next session)

- **Bee-upload zip vs. PR-stage zip**: `bee/pr53r7/*.zip`'s `cluster_id` is
  remapped into img-global's space (`make_pr_bee.py`); `real_cluster_id`
  (segment id) is not. All metrics here read the **PR-stage**
  `pr_evt<N>/mabc-pr.zip` directly to avoid this.
- **The shower-start-segment id convention** (§3) is the single biggest
  confound in any segment-id-keyed analysis of these three layers — always
  cross-check against ALL-seg coverage, not just same-id coverage, before
  calling something a phantom.
- `oc53_probe.Loader`/`walk_and_score` (doc pr/53) reused as-is for the
  `strict_verdict` replay and dead-channel scoring — same event, same ctpc/
  dead arrays, so the Python replay matches what the shipped C++ predicate
  actually sees.
- New scripts: `scripts/analysis/pr55/pr55_metrics.py` (case catalogue),
  `scripts/analysis/pr55/pr55_figures.py` (figures). Both import
  `oc53_probe`/`split_exam` from `scripts/analysis/pr53/` — not duplicated.

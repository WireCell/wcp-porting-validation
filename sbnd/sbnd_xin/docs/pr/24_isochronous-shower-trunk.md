# doc pr/24 — isochronous EM showers: diagnosis, two rejected fixes, and the endpoint fix that became the SBND default

**Status: `iso_endpoint` is the SBND PRODUCTION DEFAULT (owner 2026-08-03,
§16).** The fix lives in first-segment ENDPOINT FINDING, per the owner's
round-2 direction. Round 2 (§9-§14) built it; **round 3 (§15) is what made it
flippable** — it repairs a mid-track break-point regression the owner found in
the round-2 Bee scan (evts 284794, 59899) by taking the untrimmed axial extreme
(no tip discarded) and adding a sheet-aspect gate that keeps 1-D tracks out of
the branch entirely. One number moved the wrong way and is reported, not tuned:
§15.6/§16.2, evt 271851 on the diagnostic geometric arm. Round 1 below stands
as the diagnosis record: both of its attempts were REJECTED and removed.

**Round 4 (§17, 2026-08-06) — DIAGNOSIS ONLY, no code change, `iso_endpoint`
default unchanged at `true`.** The owner reported the trunk still "does not
cover the entire image" on three post-flip events (42280, 271851, 350186) and
suspected the knob was off. It was not — round 4 shows the seed and the very
first fit reach the image edge correctly, and the shortfall is manufactured
**downstream**, in the universal (non-`iso_endpoint`-gated) multi-round
refinement every cluster in every detector goes through. Same shape as round
1: real defect found, root mechanism not yet safe to fix in one sitting,
owner direction needed before touching shared code. See §17.10 for why this
round stops at diagnosis.

**Round 1 status: NO FIX ADOPTED.** Both attempts were implemented, measured and then
**removed at the owner's direction** (2026-08-02). Nothing from this round is
in the chain: the toolkit is back to its pre-round state and the compiled PR
job config is byte-identical to the doc pr/23 §9 baseline
(`scripts/cfg/compile_prjob_cfg.sh` vs `scratchpad/v6flip/postflip-prjob.json`, `cmp`
clean). What survives is this diagnosis, the measurement instrument
(`scripts/analysis/pr24/pr24_iso_probe.py`), one runner convenience (`SBND_DL_WEIGHTS`), and the
next-session plan in §7.

Trigger: owner hand-scan of SBND run 18255 evt 271851 (nueCC48 sample) —
"isochronous shower, not very good shower trunk identification … instead of a
straight track it found some weird two tracks at the beginning of the shower …
the current one does not reach the end of the image (low Y)"; then, on the Bee
pair — "there is a large isochronous blob (img_global), but then there are two
separate tracks from the PR graph, one at the very top and one at the bottom …
this is supposedly an EM shower, from the corner of the blob and then gradually
grow"; and the truth vertex **(−156.3, 6.4, 314.7)**.

> **Arms retired 2026-08-03.** The `work-*` arms this doc cites no longer exist
> on disk. Their record layer (logs, tsv, `tracking-*.root`; no pctree/Bee/npz/
> calib/opflash) is in `archive/records/pr23-25-era-20260803/<group>/<tag>.tar.gz`
> with a `.links.txt` and `.manifest.tsv` beside it. The **valfast** arms
> (`work-vf*-*`, `work-*-vf*`) were dropped with **no** record layer, per the
> transient-arm contract in `valfast/README.md` — the compare summaries quoted
> below are the only surviving record of those gates. See
> `docs/work-tags.md`, "RETIREMENT ROUND 2026-08-03".

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# --- attribution: 2x2 {protect_bundle} x {DL vs geometric vertex}, evt 271851 ---
export SBND_WCT_LOGLEVEL=trace          # the DL rerank score table is TRACE
./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24-a1            data 271851
SBND_PROTECT_BUNDLE=0 \
  ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24-a2poff      data 271851
SBND_DL_WEIGHTS= \
  ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24-a2geom      data 271851
SBND_PROTECT_BUNDLE=0 SBND_DL_WEIGHTS= \
  ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24-a2poffgeom  data 271851

# --- measurements (the instrument that survives this round) ---
python3 scripts/analysis/pr24/pr24_iso_probe.py work-pr24-a1    --detail 271851   # segments + 2-D overlap
python3 scripts/analysis/pr24/pr24_iso_probe.py work-pr24-a2poff --detail 271851
python3 scripts/analysis/pr24/pr24_iso_probe.py work-vfmcp1k-prodon --jobs 12 \
        --out /home/xqian/tmp/pr24_prodon.tsv               # main-cluster swap census
```

The two removed fixes are reproducible only from this doc's description; their
arms (`work-pr24-b1on`, `work-pr24-c3iso`, `work-vf*-pr24off/on`) are KEPT as
records of the measurements quoted below.

## 1. What the event is, measured

* **One filled 2-D charge sheet.** 9468 img-global points, 121 cm long, y-z
  diagonal 137 cm, with 98 % of the points inside a **15.0 cm drift slab**
  (2 %-trimmed quantile extent in the T0-corrected frame; a raw min-max extent
  reads 38.5 cm because of a 98-point, 1 %, tail 25 cm away). Two of its
  segments span 46 and 50 cm of path inside **4.3 and 9.6 time ticks**.
* **The PR graph traces the sheet's boundary, not its axis.** In the cluster's
  PCA frame, in every axis bin where both exist, segment 23010 sits at
  transverse **+10.2 … +12.1 cm** and 23012 at **−9.0 … −9.6 cm**, against a
  blob transverse span of about ±12 cm. This is the owner's "one at the very
  top and one at the bottom". Nothing runs down the middle. Mechanically this
  is expected: the graph's cheapest paths between the sheet's extreme points
  *are* its edges.
* **The neutrino vertex lands on an unrelated stub.** Production puts it at
  (−161.8, 19.3, 342.2) — **30.9 cm** from the owner's truth vertex — inside
  cluster 116, a 71-point **8.1 cm** stub whose two segments (4.8 and 3.3 cm)
  are the only things attached to the reconstructed vertex.
* **`nue_score` = −15**, the "never evaluated" sentinel. Pre-doc-pr/23-flip it
  was +4.30.

## 2. Attribution — 2 × 2 (distances to the owner's truth vertex)

| arm | protect_bundle | vertex algo | main cluster | neutrino vertex | dist | segs at vertex | nue | Enu (MeV) |
|---|---|---|---|---|---|---|---|---|
| `a1` (**production default**) | ON | DL | **116, 8.1 cm** | (−161.8, 19.3, 342.2) | **30.9 cm** | 4.8 + 3.3 cm | −15 | 1445.8 |
| `a2geom` | ON | geometric | 23, 144.5 cm | (−156.9, −3.3, 330.3) | 18.4 cm | one 50.4 cm | −3.84 | 1434.6 |
| `a2poff` | OFF | DL | 23, 345.1 cm | (−155.9, 5.7, 319.2) | **4.6 cm** | one 18.9 cm | +4.30 | 1469.8 |
| `a2poffgeom` | OFF | geometric | 23, 345.1 cm | (−155.9, 5.7, 319.2) | 4.6 cm | one 18.9 cm | +4.30 | 1469.8 |

Two separate defects stack on this event:

1. **`protect_bundle` splits the shower.** 1108 blobs → 837 + 5 fragments, i.e.
   121.0 → 70.8 cm of cluster length. The fit's charge coverage Σq_pred/Σq
   drops from **0.477 / 0.545 / 0.586** (U/V/W, unsplit) to **0.197 / 0.446 /
   0.459**. The owner's own reading: with protect_over_clustering out, the
   result "is still not ideal, but closer to the true situation".
2. **The DL (SCN) vertex hands the neutrino to a stub** — §3.2. Independent of
   the topology; costs the event its nue evaluation entirely.

## 3. Root causes

### 3.1 A component split inside a filled sheet is tomography, not physics

`ClusteringProtectBundle` splits a beam-bundle cluster at the connected
components of the relaxed graph (the port of the prototype's second graph
examination). That test asks whether charge is continuous *along a 1-D object*.
In an isochronous slab the tiling of one time slice yields a 2-D sheet, so a
component boundary inside the sheet does not mean two objects. uBooNE's beam
geometry made this rare; SBND sees it often — **98 of 428** PR-evaluated mcp1k
events have an isochronous longest cluster by the §5 census predicate.

### 3.2 A noise-level DL score, multiplied by 1000, outranks geometry

`determine_overall_main_vertex_DL` takes `Facade::Cluster*& main_cluster` and,
when the accepted DL vertex belongs to another cluster, calls
`swap_main_cluster` (`clus/src/NeutrinoVertexFinder.cxx:3650`). The composite
re-rank is `s_dl = dl_score * dl_vtx_score_scale` (SBND: 1000) plus geometric
terms capped at `W_MAIN 2.0 + W_CLEN 2.0 + W_FV 0.5 − W_SNAP 2.0` (:3535-3543):

```
DL top-5 scores in [0.0053, 0.0197]   DL score stats: mean=0.0083 std=0.0064 regime=uncertain
cand [voxel 0] cluster=116 L=8.1cm   snap=11.15cm | dl=+19.68 snap=-2.000 clen=+0.270 isol=+0.000 main=+0.000 fv=+0.500 | TOTAL=+18.435
cand [voxel 4] cluster= 23 L=144.5cm snap= 0.22cm | dl= +5.33 snap=-0.043 clen=+2.000 isol=+0.000 main=+2.000 fv=+0.500 | TOTAL= +9.778
rerank selected cluster=116 ... composite_score=18.4348
```

The network is in its own "uncertain" regime (the code labels < 0.02 uncertain,
:3457), so a 0.014 difference — noise at that scale — becomes a 14-point lead.
The stub clears the only existing guard: `W_ISOL` penalises non-main hosts
below `W_ISOL_L = 6 cm` (:3539) and cluster 116 is 8.1 cm. The comment at
:3452 claims that with small scores "the geometric terms below dominate the
final ranking — which is intentional"; at `dl_vtx_score_scale = 1000` that is
false.

## 4. The two attempts, and why they were rejected

### 4.1 `protect_skip_iso_xext` — exempt isochronous clusters from the split (REJECTED)

Implemented as a default-OFF knob in `ClusteringProtectBundle::process_cluster`:
skip the split when `length >= 80 cm` and the 2 %-trimmed quantile drift-x
extent of the default-scope points is `< max(skip_iso_xext, 0.18 * length)`.

Measured on evt 271851 at `skip_iso_xext = 25 cm`: fires
(`cluster 23: isochronous (121.0 cm long, drift-x extent 15.0 cm) -- split
skipped`) and reproduces the protect-OFF arm **exactly** — vertex (−155.9, 5.7,
319.2), 4.6 cm from truth; `nue_score` +4.30; charge coverage 0.53.

**Owner verdict: do not adopt.** Turning the stage off for a whole class of
clusters — 98/428 mcp1k events carry an isochronous main — gives up
protect_over_clustering to buy one event. Removed; not committed anywhere.

Worth keeping from it: the *measurement* details, because any future predicate
on "is this cluster isochronous" hits the same two traps —
`iso_band_like()`'s raw blob-centre min-max reads **38.4 cm** on this cluster
(the blobs carry more than one (apa, face) and the per-face transform inflates
it) and a min-max on corrected points still reads 38.5 cm because of a 1 %
tail; the honest number, 15.0 cm, needs the corrected frame *and* a quantile.

### 4.2 `dl_vtx_swap_min_len` / `dl_vtx_swap_min_frac` — DL main-cluster swap guard (REJECTED)

Implemented as default-OFF knobs in `determine_overall_main_vertex_DL`: reject
the DL pick outright (falling back to the traditional vertex) when it would
move the main cluster to a host shorter than 10 cm or than 25 % of the
incumbent.

Measured: evt 271851 vertex 30.9 → 18.4 cm from truth, `nue_score` −15 →
−3.84 — i.e. the vertex stops landing on the stub but still picks the wrong end
of the shower, and the event remains unselected. Population (fresh arms at the
post-flip default, `pr24off`/`pr24on`): mcp1k **25/572** events move (11 gain
`nu_evaluated`), nueCC48 1/47 (evt 122660, `nue_score` −15 → +1.91).

**Owner verdict: also does not work well; remove.** Removed from C++, from the
jsonnet wiring (the wiring had been swept into commit `75c703da` by a
concurrent session and is reverted by this round's commit), and from the runner.

Worth keeping from it: the swap census below — the pathology is real,
pre-existing, and not caused by the doc pr/23 flip.

## 5. Census (instrument: `scripts/analysis/pr24/pr24_iso_probe.py`)

The main-cluster swap is detected exactly: compare `TaggerCheckNeutrino`'s
"selected main cluster N" log line with the cluster hosting the final vertex.

| arm | events | swaps | onto a host < 25 % of the incumbent | isochronous main |
|---|---|---|---|---|
| `work-vfmcp1k-prodon` (post-flip) | 428 | 49 | **10 (2.3 %)** | 98 |
| `work-vfmcp1k-prodoff` (pre-flip) | 428 | 56 | 12 (2.8 %) | 105 |
| `work-poc48c-on3` (nueCC48, post-flip) | 47 | 5 | 2 (incl. 271851) | 28 |
| `work-poc48-off` (nueCC48, pre-flip) | 47 | 3 | 1 | 26 |

Worst cases hand a 140-350 cm main cluster to hosts of 0.4-1.4 cm. Same rate
before and after the doc pr/23 flip, so this is not a flip regression.

## 6. Bee sets (evt 271851)

| arm | Bee set |
|---|---|
| production default | https://www.phy.bnl.gov/twister/bee/set/ebd21c9d-b0d6-4274-8516-8f208aa0c34c/event/list/ |
| swap guard (rejected fix 4.2) | https://www.phy.bnl.gov/twister/bee/set/59e14584-ca98-4f5d-80dc-cd3e061cf855/event/list/ |
| isochronous exemption (rejected fix 4.1) | https://www.phy.bnl.gov/twister/bee/set/45ac3f02-a8c0-439c-883c-a2f0533b9701/event/list/ |

## 7. What to try next session

The owner's chosen direction is **leave `protect_bundle` alone and fix the
PR / track fit for this topology**. Open questions to settle first (they change
what gets built):

1. **Target output.** For a sheet like this: one axial trunk from the narrow
   corner plus a shower cloud, with the boundary segments gone? Or several
   segments, but forced onto charge-weighted interior paths instead of edges?
2. **Where the fix lives.** The fan comes from the graph/steiner path search
   using extreme points as terminals — in a filled sheet the cheapest corner-
   to-corner paths *are* the edges. Either (a) terminal/path selection changes
   for sheet-like clusters (principled, riskier, touches every cluster's PR),
   or (b) a later pass collapses a boundary fan onto the charge-weighted axis
   (contained, cosmetic-first).
3. **Success metric.** Proposed: distance from the reconstructed vertex to a
   hand-scanned truth vertex, trunk direction vs the charge-weighted axis, and
   Σq_pred/Σq coverage — on nueCC48 plus the isochronous subset of mcp1k. A
   non-isochronous control cohort still has to be measured; the 0.35/0.53
   coverage numbers above are absolute, with no reference yet.
4. **The truncation stays** while `protect_bundle` is untouched: this shower
   remains cut 121 → 70.8 cm and its `nue_score` stays −15. Any trunk fix has
   to work on the truncated object, or the split question has to be reopened
   on its own terms (not as an exemption).

Two leads recorded but not pursued:

* **Projection-overlap ghost veto** — sibling of
  `examine_partial_identical_segments`
  (`clus/src/NeutrinoStructureExaminer.cxx:2109`, which today only catches
  3-D-coincident pairs within 0.3 cm). In the split arm, segments 23011 and
  23012 share **u 0.38, v 0.67, w 0.80** of their 2-D pixels and 23011 carries
  median dQ/dx 234 against neighbours' 3000-5000 — a charge-starved ghost. The
  two 50 cm edge segments share only 0.28-0.38 in two planes, so this test
  alone would not collapse the fan.
* **DL score scaling** — §3.2's real cause is `dl_vtx_score_scale = 1000`
  amplifying an uncertain network. Scaling the DL term down (or zeroing it)
  when `dl_score_max` is in the uncertain regime would restore the ranking the
  code comment describes, but it changes an existing default for every event
  and needs its own round (CLAUDE.md §5 rule 1).

## 8. What this round leaves behind

* `sbnd_xin/scripts/analysis/pr24/pr24_iso_probe.py` — per-event isochrony / vertex-host / segment /
  2-D projection-overlap / charge-coverage probe, plus the population census.
* `sbnd_xin/run_pr_chain_batch.sh` — `SBND_DL_WEIGHTS` (unset ⇒ cfg default;
  set-empty ⇒ geometric vertex), the attribution arm of §2.
* toolkit: a revert commit removing the swap-guard jsonnet wiring that a
  concurrent session's commit `75c703da` had swept in. Compiled config
  byte-identical to the doc pr/23 §9 baseline; `wcdoctest-clus` 49/49 PASS.
* Arms kept as records: `work-pr24-a1`, `-a2poff`, `-a2geom`, `-a2poffgeom`,
  `-b1on`, `-c3iso`, `work-vf{mcp1k,nuecc48}-pr24{off,on}`, `bee-pr24/`.

---

# Round 2 (2026-08-03) — fix the ENDPOINTS, not the split: `iso_endpoint`

**Status: IMPLEMENTED, default OFF, small-set validated.** Owner direction for
this round: `protect_over_clustering` is good and stays; the suspected defects
are (a) the big split of the end tracks (Steiner graph?) and (b) incorrect
end-point finding for the isochronous case, where the pre-split image had "a
more reasonable (though still not exactly correct)" endpoint. Reference image
of a good isochronous track (owner): the fitted trajectory runs down the
sheet's interior axis and ends at one clean corner point.

## 9. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# knob-on single event, production DL vertex and geometric-vertex arms
SBND_ISO_ENDPOINT=1 ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24r2-on1     data 271851
SBND_ISO_ENDPOINT=1 SBND_DL_WEIGHTS= \
  ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24r2-on1geom data 271851

# knob-off byte-identity gate (6 events) + cohorts (valfast pinned hubs)
./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24r2-off6 data 271851 10550 111412 116962 122660 131357
valfast/run_valfast.sh pr24r2off -j 24 mcp1k nuecc48
SBND_ISO_ENDPOINT=1 valfast/run_valfast.sh pr24r2on -j 24 mcp1k nuecc48

python3 scripts/analysis/pr24/pr24_iso_probe.py work-pr24r2-on1geom --detail 271851
```

## 10. Symptom / Root cause / Why it hid / Fix / Verification

**Symptom** (owner, §0): two ~50 cm PR tracks hugging the top and bottom edges
of the isochronous charge sheet, forking from a vertex at the wrong (high-y,
high-z) corner; the reconstruction "does not reach the end of the image".

**Root cause — it is the endpoint selection, not the Steiner graph and not the
fit.** The first PR segment's endpoints come from
`init_first_segment` → `get_two_boundary_steiner_graph_idx` →
`get_two_boundary_wcps` (`clus/src/Facade_Cluster.cxx`), whose non-cosmic
score (`calculate_boundary_metric`, a faithful port of prototype
`PR3DCluster_path.h:530-536`) is

```
|dx|/one_tick + 0.0*|dU| + live_U + 0.0*|dV| + live_V + 0.0*|dW| + live_W
```

The wire-separation terms carry `*0.0` coefficients (prototype parity), so on
an isochronous sheet (`|dx| ~ 0`) the score degenerates to "count of live
wires spanned in U+V+W": the winner is the widest wire-footprint pair — two
corners on the sheet's EDGES, not the tip of its axis. The Steiner graph's
role is only obliging: between two same-edge corners the cheapest paths ARE
the boundary edges (charge-scaled weights 0.8-1.2x, intra-blob chords
0.8-0.9x). Two aggravators:

1. A TOOLKIT-ONLY local-PCA endpoint refinement
   (`clus/src/NeutrinoPatternBase.cxx`, commit `1eb097a9`, no prototype
   counterpart) power-iterates a 10 cm neighbourhood; on a filled 2-D patch
   that local PCA is degenerate in the sheet plane and drags each endpoint
   further along the edge.
2. `find_other_segments`' far endpoint `special_B` is the plain farthest
   Euclidean point (`clus/src/NeutrinoOtherSegments.cxx:306`) — from any
   corner that is the opposite corner, so every fan segment repeats the
   degeneracy. (NOT touched this round; see §13.)

**Why it hid.** uBooNE's beam is along z, so beam-related objects almost never
lie perpendicular to drift and the `*0.0` metric was adequate there; the
prototype's own iso endpoint recipe (`adjust_wcpoints_parallel`,
`data/src/PR3DCluster.cxx:428` — drop the narrowest plane, re-pick by
wire-space separation, 30 cm leash) exists only in cluster separation, and
`search_for_connection_isochronous` (`pid/src/PR3DCluster_graph.h:1445`) has
its only call site commented out. SBND sees drift-perpendicular showers
routinely (98/428 iso mains, §5). The owner's observation that the PRE-split
endpoint was "more reasonable" is the same mechanism at a different scale:
the unsplit 121 cm object includes charge that stretches the live-wire
footprint toward the true corner, the truncated 70.8 cm sheet does not.

**Fix — `iso_endpoint` (C++ default false, byte-identical off).**
`PatternAlgorithms::find_iso_first_segment_endpoints`
(`clus/src/NeutrinoPatternBase.cxx`), called at the top of
`init_first_segment`; on gate failure the legacy block runs unchanged.

* Gate: cluster `length >= iso_endpoint_min_length` (40 cm) AND the
  2 %-quantile-trimmed drift-x extent of the charge-qualified default-scope
  points (same `is_point_excluded` skip and blob-charge >= 1500 cut as the
  legacy search) `< iso_endpoint_max_xext` (25 cm) AND
  `< iso_endpoint_xext_frac * length` (0.35). The trimmed corrected-frame
  extent is the one honest isochrony measure (§4.1 traps).
* Endpoints: principal axis of the qualified points (power iteration seeded
  with the global PCA axis-0); axis projections quantile-trimmed at the same
  2 %; each end band (3 cm) contributes the point nearest its band CENTROID —
  an interior pick, not a corner — then snaps to the nearest Steiner terminal
  exactly like the legacy path. Same-terminal or degenerate-axis outcomes
  fall back to legacy.
* The local-PCA refinement is SKIPPED on this branch (aggravator 1).
* Knobs: `iso_endpoint`, `iso_endpoint_min_length`, `iso_endpoint_max_xext`,
  `iso_endpoint_xext_frac`, `iso_endpoint_xext_quantile`, threaded
  TaggerCheckNeutrino -> PatternAlgorithms; jsonnet key-suppression in
  `cfg/pgrapher/common/clus.jsonnet`, `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
  `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` (SHIPS OFF); runner
  escape `SBND_ISO_ENDPOINT=1`. Porting-divergence entry added to
  `clus/docs/porting/porting_dictionary.md` (do not "fix" the `*0.0`
  coefficients in `calculate_boundary_metric` itself).

**Verification — knob-off gates (all PASS):**

* Compiled PR-job config `cmp`-identical to the pre-change compile at the same
  HEAD (`/home/xqian/tmp/pr24r2/base-prjob.json` vs `off-prjob.json`; the
  production `pipeline_names` list, not the vacuous `[]`).
* Knob-on compiled config differs by exactly one key: `"iso_endpoint": true`.
* `./build/clus/wcdoctest-clus`: 49/49 cases, 565/565 assertions.
* 6-event A/B (`work-pr24r2-base` pre-change build vs `work-pr24r2-off6`
  post-change build, knob off): 12/12 archives member-hash identical
  (`hash_archive.py` FIELD 1), nusel tables identical. Events: 271851, 10550,
  111412, 116962, 122660, 131357.
* Freshness proof: `local/lib/libWireCellClus.so` reinstalled after the last
  source edit before every arm.

## 11. Measured effect, evt 271851 (all arms at current HEAD, protect_bundle ON)

Gate fires once, on the target cluster only:

```
iso endpoint: fired (L=70.8 cm, xext=7.5 cm) A=(-160.7,31.1,374.5) B=(-156.4,7.1,316.7)
```

Endpoint B is **2.1 cm** from the owner's truth vertex (−156.3, 6.4, 314.7).

| arm | vertex | dist to truth | vertex host | nue | Enu (MeV) |
|---|---|---|---|---|---|
| `pr24r2-base` (production DL) | (−161.8, 19.3, 342.2) | 30.9 cm | **stub 116, 8.1 cm** | −15.0 never evaluated | 1445.8 |
| `pr24r2-on1` (iso ON, DL) | (−156.7, 13.7, 334.8) | 21.4 cm | shower 23 | +4.30 | 1543.0 |
| `pr24r2-on1geom` (iso ON, geometric) | (−156.6, 7.6, 315.8) | **1.7 cm** | shower 23 | +4.30 | 1484.5 |

* **The two-edge-track V is gone.** The base arm's 50.4 + 46.0 cm edge pair is
  replaced by a connected trunk chain from the wide corner
  (−160.6, 31.7, 374.5) down to (−156.6, 7.6, 315.8) — the narrow corner where
  the shower starts — with a fan of short shower segments around it. No
  segment pair sits at opposite sheet edges.
* **The fit runs down the interior.** Fit-point transverse offset vs the
  sheet's principal axis (7807-point img cloud, span ±13 cm): |median|
  **7.8 → 1.6 cm**, p90 **11.1 → 4.2 cm** (base → on1geom).
* **The DL stub-swap disappears on this event**: with a real trunk passing
  near the DL voxel, the rerank keeps cluster 23. The DL arm's remaining
  21.4 cm residual is the SCN network's own mid-trunk voxel choice (§3.2's
  uncertain-regime scaling, unchanged this round, out of scope).
* Charge coverage Σq_pred/Σq: 0.198/0.446/0.459 → 0.053/0.598/0.558 (U/V/W;
  totals 0.353 → 0.372). V and W rise; **U drops** — the trunk's U projection
  overlaps the sheet's own U footprint less than the edge tracks did. Watch
  item, not tuned (§5 rule 7).
* Both fixes the owner rejected in round 1 stay out; `protect_bundle` is
  untouched and the cluster stays split at 70.8 cm.

## 12. Small-set cohorts (owner-authorized 572-event sample, 24 jobs)

Arms `work-vf{mcp1k,nuecc48}-pr24r2{off,on}` from the pinned valfast hubs
(`run_valfast.sh pr24r2off|pr24r2on -j 24 mcp1k nuecc48`); all 1238 jobs
rc=0. CAVEAT (valfast/README.md): the pinned pctrees predate the pr/14-pr/20
clustering defaults — valid for off-vs-on A/B, not a reproduction of today's
clustering. In particular evt 271851 is cosmic-tagged in the pinned nueCC48
hub (both arms, no firing); its recovery in §11 is measured on the
current-clustering `work-nuecc48-poc0` hub.

* **Fire rate**: mcp1k **80/572** events (14 %), nueCC48 **20/47** — 105
  cluster firings over the 100 events (a few events gate two clusters).
  Consistent with the §5 iso-main census (98/428).
* **Containment**: all **519 non-firing events are archive-identical**
  (`hash_archive.py` member hashes, pctree + mabc, 0 diffs) between off and
  on. The knob touches gated clusters only.
* **Firing-event deltas** (off → on): **0 event_label moves, 0 nu_evaluated
  moves** in both samples. mcp1k: 72/80 move `numu_score`, 52/80 move the
  vertex by > 0.5 cm — mostly a few cm, but 10 events move it 27-173 cm
  (trunk-end relocations; e.g. 168614 173 cm with numu −0.16 → −2.94, 68648
  114 cm with +2.26 → −0.23). `numu_score` moves are MIXED in sign, as
  expected when trunks re-route; there is no truth on mcp1k (doc pr/20), so
  these are recorded, not judged. `nue_score` flips are between the −15.0 /
  ±4.30 sentinels (doc pr/25 gotcha: quantized, not continuous).
* **nueCC48**: 18/20 move `numu_score`; the standout is **evt 122660** —
  `nue_score` −15.0 → **+4.30** with an 85 cm vertex move, the same event
  round 1's rejected swap guard could only take to −3.84. Owner scan of the
  20 knob-on events is the natural next step before any flip talk.
* **DL stub-swap census unchanged** (probe on the mcp1k arms): swaps 50 → 49,
  swaps onto a host < 25 % of the incumbent 11 → 11. Expected — this round
  does not touch the §3.2 DL scoring; on 271851 the swap disappears only
  because the improved trunk puts real fit points near the DL voxel.

## 13. Bee sets (evt 271851, round 2)

| arm | Bee set |
|---|---|
| production default (current HEAD) | https://www.phy.bnl.gov/twister/bee/set/1c639f1c-4413-4d06-be65-af7d53b80edd/event/list/ |
| iso_endpoint ON + DL vertex | https://www.phy.bnl.gov/twister/bee/set/98850353-e0f1-4980-92cd-8084bc00d32a/event/list/ |
| iso_endpoint ON + geometric vertex | https://www.phy.bnl.gov/twister/bee/set/e6021737-741f-4180-b8c4-6eed5d81718c/event/list/ |

Cohort off-vs-on pairs for the owner scan (same event order in the off and on
set of each pair, so Bee index N is the same event in both):

| set | events (Bee index order) | off | on |
|---|---|---|---|
| nueCC48, all 20 firing | 388, 30504, 38856, 42280, 52672, 74544, 81597, 90055, 111412, 122660, 168596, 196649, 246579, 256587, 269774, 350186, 360535, 437699, 447477, 469665 | https://www.phy.bnl.gov/twister/bee/set/54ec60af-3626-46d4-bb5e-c8ae5fb796b1/event/list/ | https://www.phy.bnl.gov/twister/bee/set/86633e49-9819-4d37-802e-a47e0c80692b/event/list/ |
| mcp1k, 17 biggest movers (vertex > 25 cm or nue flip) | 48367, 55595, 58345, 59899, 67746, 68648, 168614, 172942, 282204, 284794, 285564, 292577, 321525, 393608, 395060, 400504, 402880 | https://www.phy.bnl.gov/twister/bee/set/9c9056d1-f24d-4e01-be1e-230104e14161/event/list/ | https://www.phy.bnl.gov/twister/bee/set/bbcefa2e-2972-44ba-b7ba-dd7b81720026/event/list/ |

## 14. Residuals and the next decision

* `special_B` in `find_other_segments` still picks the farthest point; with an
  axial first trunk the observed fan is short shower segments (expected for an
  EM shower), so it was left prototype-faithful this round. Reopen only if the
  knob-on census shows surviving >=30 cm opposite-edge pairs.
* The DL mid-trunk vertex (21.4 cm) is the §3.2 uncertain-regime scaling,
  unchanged. The geometric arm shows what the trunk itself supports (1.7 cm).
* The U-plane coverage drop (§11) is unexplained and recorded.
* Default flip is the owner's call (escalation rule 1) after scanning the
  knob-on Bee sets and the §12 census.

## 15. Round 3 — the mid-track break-point regression

**Status: SHIPPED default OFF** (toolkit `iso_endpoint` unchanged at `false`;
two new knobs `iso_endpoint_tube_radius` = 4 cm diagnostic-only and
`iso_endpoint_min_aspect` = 0.12). Knob-off path proven byte-identical to
round 2. **One physics number moved the wrong way and is reported, not tuned**
(§15.6, evt 271851 geometric arm).

Owner trigger, after scanning the §13 knob-on sets: *"the new algorithm seems
to have more break points in the middle of the track"* — evt **284794** and
**59899**, each a straight track that acquired an interior vertex.

### 15.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# --- off-path gates ---
./scripts/cfg/compile_prjob_cfg.sh /nfs/data/1/xqian/toolkit-dev/toolkit/cfg /home/xqian/tmp/pr24r3/off-prjob.json
cmp /home/xqian/tmp/pr24r3/v0-base-prjob.json /home/xqian/tmp/pr24r3/off-prjob.json
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24r3-off6 data \
    271851 10550 111412 116962 122660 131357
# vs work-pr24r2-off6, hash_archive.py FIELD 1 ONLY

# --- aspect harvest (Fix B disabled, Fix A live) ---
SBND_REQUIRE_WASMAIN=0 SBND_ISO_ENDPOINT=1 SBND_ISO_MIN_ASPECT=0 PR_JOBS=11 \
  ./run_pr_chain_batch.sh work-nuecc48-nuf  work-vfnuecc48-pr24r3b data <20 evts of §13>
SBND_REQUIRE_WASMAIN=0 SBND_ISO_ENDPOINT=1 SBND_ISO_MIN_ASPECT=0 PR_JOBS=11 \
  ./run_pr_chain_batch.sh work-mcp1kall-d59k work-vfmcp1k-pr24r3b  data <17 evts of §13>

# --- final arm (shipped defaults) ---
SBND_REQUIRE_WASMAIN=0 SBND_ISO_ENDPOINT=1 PR_JOBS=11 \
  ./run_pr_chain_batch.sh work-nuecc48-nuf  work-vfnuecc48-pr24r3 data <same 20>
SBND_REQUIRE_WASMAIN=0 SBND_ISO_ENDPOINT=1 PR_JOBS=11 \
  ./run_pr_chain_batch.sh work-mcp1kall-d59k work-vfmcp1k-pr24r3  data <same 17>
SBND_ISO_ENDPOINT=1                ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24r3-on1     data 271851
SBND_ISO_ENDPOINT=1 SBND_DL_WEIGHTS= ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24r3-on1geom data 271851

# --- the new regression detector ---
python3 scripts/analysis/pr24/pr24_iso_probe.py work-vfmcp1k-pr24r3 --junctions --vs work-vfmcp1k-pr24r2off
```

### 15.1 Symptom

Two mcp1k events gained an interior vertex under round 2:

| evt | round-2 segments | junction turn | legacy trajectory bend there |
|---|---|---|---|
| 284794 | 260.1 + 15.7 cm | **0.9°** | straight |
| 59899 | 27.6 + 8.6 cm | **8.0°** (probe: 7.2°) | 1.6-2.8° |

A 0.9° junction is a vertex planted in a straight track. 59899 additionally
lost 3 cm of tip (charge coverage 0.873 → 0.738).

### 15.2 Root cause

Round 2 picked each endpoint as *the band point nearest the centroid of a 3 cm
band at the **quantile-trimmed** axial extreme*. Both steps pull inward, and
the trim scales with the point count: on the 7123-point 284794 cluster the 2 %
trim alone is 142 points, and the endpoints landed **8.4 cm (A) / 15.1 cm (B)**
short of the true tip (59899: 1.7 / 1.5 cm). The perpendicular error was only
0.05-0.20 cm — the defect was purely axial.

Nothing downstream of `init_first_segment` is modified by the knob (the helper
is called at `NeutrinoPatternBase.cxx:430`, near the top of that function, and
only replaces the two seed points). So `find_other_segments` did exactly its
job and claimed the uncovered tip as a segment of its own, and `break_segments`
split 59899 at a marginal wobble on the perturbed path.

### 15.3 Why it hid

The round-2 census reported 0 `event_label` moves, 0 `nu_evaluated` moves, and
vertex/score deltas — **none of which can see a vertex inside a straight
track**. The pathology only surfaced in the owner's Bee scan. The gate itself
was also blind: it tested "long and thin in drift", which an ordinary track
satisfies, and never tested 2-D sheet-ness.

### 15.4 Fix — two mechanisms, each fixing a different subset

**A. Endpoint = axial extreme first, lateral centring second** (all in
`find_iso_first_segment_endpoints`). The quantile trim now serves only the
*gate* measurements. The endpoint is the **untrimmed** axial extreme over all
charge-qualified points, walked inward only past isolated spikes (≥ 3 qualified
points within 3 cm, itself included); then, within the 3 cm end band, the point
**nearest the axis line** is taken — which is what still keeps the pick off a
sheet's edge corner. The endpoint can never move inward of the object's extent,
so "leaves no stub" holds by construction.

*Design note — the tube filter was measured and rejected.* The first draft
followed the owner's proposal literally: filter to a 4 cm tube around the axis
line and take that subset's extreme, on the premise that anything thinner than
the tube degenerates to the plain tip. **The premise is false on real
clusters**: the axis line is straight, so a long or curved object leaves the
tube near its tips. Measured over the 38 gated clusters, the tube held only
383/2097 points on evt 282204 and its extreme sat **28.6 cm inside** the round-2
endpoint (284794 A: −20.8 cm) — re-introducing the very inward bias this round
removes. **16 of 76 shipped endpoints lie farther than 4 cm from the axis
line** (max 29.5 cm). `iso_endpoint_tube_radius` is therefore retained as a
**diagnostic only**: the DEBUG line reports whether the chosen endpoint is
laterally central (`intube=true/false`). Under the shipped rule all 38 axial
gains are **positive** (min +0.7, max +22.7 cm) and no spike-guard walk-in ever
occurred (`walk=0.00` on all 76 endpoints).

**B. Sheet-aspect gate** `iso_endpoint_min_aspect` (default **0.12**): the
trimmed transverse-to-axial extent ratio, from the second PCA axis obtained by
deflating the covariance with the Rayleigh quotient λ₀ = v₀ᵀSv₀. Below it the
cluster is handed back to the legacy path, so the branch is *provably inert* on
1-D track-like clusters rather than merely harmless.

The C++ aspect reproduces the offline probe exactly: **271851 = 0.347,
122660 = 0.184, 284794 = 0.068, 59899 = 0.069**. 0.12 keeps both recoveries and
rejects both regressions. 0.25 would have killed the 122660 recovery.

Which mechanism fixed what, honestly: **the aspect gate is what removes the
owner's two named events** (they never reach the endpoint code again). **Fix A
is what cleans up the two further round-2 cases the detector found** —
evt 282204 (4.7°) and evt 48367 (1.1°) — which still fire under the gate.

### 15.5 The regression detector (new)

`scripts/analysis/pr24/pr24_iso_probe.py --junctions [--vs OTHER_ARM]`: for the vertex-host cluster
it pairs segments ≥ 5 cm whose endpoints meet within 2 cm, and reports the
**deviation from straight-through** (0° = the two segments continue each other
exactly). Direction at a junction is taken over 10 cm of arc so one jittery fit
point cannot set it.

Validation of the tool itself: run against round-2's own arm it flags
**4/17 mcp1k events** — the owner's 284794 (2.5°) and 59899 (7.2°) plus 282204
(4.7°) and 48367 (1.1°), which had not been spotted by hand.

| arm | mcp1k events gaining a straight-through junction vs OFF |
|---|---|
| round 2 ON | **4/17** |
| round 3, Fix A only (no aspect gate) | **0/17** |
| round 3 shipped | **0/17** |

On nueCC48 the count is 6/20 (round 2) → 5/20 (round 3). These are EM showers,
where near-collinear branch meetings are physical and mostly short (7.1/5.5,
6.2/6.2, 11.2/13.0 cm pairs) — **the metric discriminates on tracks and not on
showers**, so the count is recorded, not chased (§5 rule 7). The one
substantial case is evt 469665 (24.2 + 36.0 cm meeting at 1.6°), present in
both round 2 and round 3; it is a residual, not a round-3 regression.

### 15.6 Evt 271851 — the motivating event, mixed and REPORTED

Fired: `L=70.8 cm, xext=7.5 cm, aspect=0.347, n=7696,
A=(−156.7,32.0,377.9) B=(−155.7,4.1,317.6), gain=6.5/3.8 cm, walk=0.00/0.00,
dperp=0.2/0.1 cm`. The trunk reaches 6.5 / 3.8 cm further than round 2 and both
endpoints are laterally central.

| arm | vertex → truth (−156.3, 6.4, 314.7) | nue_score |
|---|---|---|
| production default (knob off) | 30.87 cm | −15.00 |
| round 2, DL vertex | 21.38 cm | +4.30 |
| **round 3, DL vertex** | **2.36 cm** | **+4.30** |
| round 2, geometric vertex | 1.66 cm | +4.30 |
| **round 3, geometric vertex** | **6.60 cm** | **−3.49** |

**The DL arm — the owner's official chain (confirmed 2026-08-03) — improves by
19 cm. The diagnostic geometric arm gets worse and loses the nue selection.**
So on the chain that actually runs, round 3 takes 271851 from 21.4 cm to
2.4 cm; the geometric regression is a diagnostic-arm result, and the geometric
arm's role here is to show what the trunk geometry alone supports. Mechanism, from the segment table: the
extra 3.8 cm of reach at the shower's narrow end is real but low-quality charge
— it produces a new 15.6 cm segment `23011` with chord/path **0.53** and median
dQ/dx **467** (vs 3244 on its neighbour), and the traditional vertex algorithm
picks the *far* end of that segment (y = 12.9) instead of the near end
(y = 4.6, which is 2.4 cm from truth and is what the DL arm picks). Charge
coverage also slips, 0.372 → 0.333 all-plane.

Note what that means: **segment `23011` did not exist before this change** —
the extra 3.8 cm of reach manufactured it, and the vertex finder was then
handed a segment whose far end it prefers. The vertex finder did not develop a
new flaw.

No parameter was tuned to make this look better. Two readings are available for
the owner and this doc does not choose between them: (i) the shipped endpoint
rule is right and the residual is the geometric vertex-finder's end choice on a
segment it should not terminate at; (ii) the endpoint needs a charge-quality
requirement, because the only quality filter today is the blob ≥ 1500 e cut
inherited from the legacy boundary search, which a dQ/dx-467 tail clears
easily — the concrete lever would be a minimum median dQ/dx (or a minimum
point density) on the end band before the extreme is accepted.
**No aspect threshold can protect this event**: 271851 gates at 0.347, far
above any cut that still rejects 284794/59899 at 0.068/0.069.

Weighting, given the DL arm is the official chain: this is a **watch item on a
diagnostic arm**, not a blocker on the production path. It still matters as
evidence about the endpoint rule itself — the geometric arm is the one that
reads the trunk geometry without a network prior, so its disagreement is the
cleanest signal that the last 3.8 cm of reach is low-quality charge. Note the SBND production default *is*
the DL vertex (doc pr/4), which is the arm that improved.

### 15.7 Cohort results (20 nueCC48 + 17 mcp1k of §13)

The §13 mover list is **17**, not 20 — that is every mcp1k event over the
round-2 "vertex > 25 cm or nue flip" bar.

* **Fire rate under the gate**: nueCC48 **19/20** events (437699 now rejected at
  aspect 0.111), mcp1k **7/17** — the gate hands 10 of the 17 movers straight
  back to the legacy path.
* **Containment**: all **11 non-firing events are archive-identical to the OFF
  arm** (`hash_archive.py` member hashes on `mabc-pr.zip` + `pctree`, 0 diffs).
  In particular **284794 and 59899 are byte-identical to legacy** — the two
  events the owner reported are fully reverted. Round 2's 519/519 non-firing
  containment carries over a fortiori: the round-3 gate is a strict subset of
  round 2's, so it was not re-run.
* **Recoveries held**: evt **122660** keeps `nue_score` −15.00 → **+4.30**.
* **Score census** (no truth on either sample — recorded, not judged): among the
  19 firing nueCC48 events, 6 flip `nue_score` sign vs OFF — 4 upward
  (111412 −0.34→+1.75, 350186 −4.30→+2.01, 469665 −1.32→+4.13, 122660) and
  2 downward (38856 +1.89→−1.21, 52672 +0.10→−2.92). Round 2 flipped a
  different 6. `nue_score` is quantized at the ±4.30 / −15.0 sentinels.
* **Aspect distribution** over the 38 gated clusters: min 0.039, p10 0.069,
  median 0.200, p90 0.361, max 0.554. **Six clusters sit within ±0.02 of the
  0.12 cut** (0.103, 0.104, 0.111, 0.118, 0.128, 0.137), so 0.12-vs-0.15 is a
  3-cluster decision on this set (27/38 vs 24/38 fire). This set is biased — it
  was selected as round-2 movers — so these are not population rates.

### 15.8 Gates

| gate | result |
|---|---|
| compiled config, knob off | `cmp`-identical to the pre-change baseline (`/home/xqian/tmp/pr24r3/v0-base-prjob.json`) |
| compiled config, knob on | differs by exactly the new keys |
| `./build/clus/wcdoctest-clus` | 49/49 cases, 565/565 assertions |
| freshness proof | `libWireCellClus.so` 07:58:56 > `NeutrinoPatternBase.cxx` 07:58:03 |
| knob-off A/B, 6 events × 2 archives | **12/12 identical** to the round-2 off arm (`work-pr24r3-off6` vs `work-pr24r2-off6`) |
| non-firing containment, 37-event cohort | **11/11 identical** to OFF |

### 15.9 Bee sets (round 3)

Same event order as the §13 tables, so Bee index N is the same event across the
off / round-2 / round-3 sets.

| set | fired under the gate | Bee |
|---|---|---|
| evt 271851, iso_endpoint ON + DL vertex | yes | https://www.phy.bnl.gov/twister/bee/set/a73f7989-e3c4-4f9f-b60f-29c99c34c356/event/list/ |
| evt 271851, iso_endpoint ON + geometric | yes | https://www.phy.bnl.gov/twister/bee/set/18c57992-1c9a-4b77-9040-6ebce4f47996/event/list/ |
| nueCC48, all 20 | 19/20 (437699 no) | https://www.phy.bnl.gov/twister/bee/set/aa8ce702-c1df-45f2-a77f-0c63f83e6f9d/event/list/ |
| mcp1k, 17 movers | 7/17 | https://www.phy.bnl.gov/twister/bee/set/9a54cba3-7d19-4664-9d2d-9f9618649e67/event/list/ |

Fired / not-fired per event, mcp1k (not-fired ⇒ byte-identical to the OFF set,
no scan effort needed): **fired** 48367, 168614, 172942, 282204, 285564,
400504, 402880 — **not fired** 55595, 58345, 59899, 67746, 68648, 284794,
292577, 321525, 393608, 395060.

### 15.10 Residuals and the next decision

* The **271851 geometric-arm regression** (§15.6) is the open item; it is the
  owner's call whether it blocks anything, since the production arm improved.
* `iso_endpoint_tube_radius` is now diagnostic-only. It could be deleted, but
  the `intube` flag is what makes the "endpoint stranded off-axis" cases
  (282204 at 29.5 cm) visible in a log, so it is kept.
* nueCC48 evt 469665's 24.2 + 36.0 cm junction at 1.6° predates round 3.
* The U-plane coverage drop from §11 persists (0.198 → 0.060 on 271851).
* Full valfast (572 + 47) is **still deferred** — the owner decides after this
  scan.


## 16. Production flip (owner 2026-08-03)

Owner, after the §15.9 Bee scan: *"This is better, please make this our default
production running for SBND."*

**`iso_endpoint = true` in `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`.**
Per doc 68 the SBND operating point lives in that file only — `clus.jsonnet`'s
`clus_pr()`/`pr()` function defaults stay `false`, and the **C++ default stays
`false`**, so no other experiment and no bare C++ consumer is affected. The six
sizing knobs stay `null` = the C++ defaults (40 cm min length, 25 cm max drift
extent, 0.35 frac, 0.02 quantile, 4 cm diagnostic tube radius, 0.12 min sheet
aspect).

**NOT bit-identical** to the pre-flip chain — a deliberate production-behaviour
change, same bar as every other SBND default flip.

### 16.1 What the flip buys (all measured in §15, on the DL vertex arm — the
owner's official chain)

* evt **271851**: vertex **30.9 → 2.4 cm** from truth, `nue_score` −15 → **+4.30**.
* evt **122660**: `nue_score` −15 → **+4.30**.
* **0/17** mcp1k events gain a straight-through segment junction (round 2: 4/17);
  evts 284794 and 59899 are byte-identical to legacy because the aspect gate
  rejects them at 0.068/0.069.
* Every non-firing event stays archive-identical to the legacy arm.

### 16.2 Caveat carried into production

§15.6: on the **diagnostic geometric-vertex arm** evt 271851 goes 1.7 → 6.6 cm
and loses the nue selection, because the endpoint's extra 3.8 cm of reach
manufactures a low-quality tail segment (chord/path 0.53, median dQ/dx 467)
whose far end the traditional vertex finder prefers. Accepted by the owner on
the strength of the DL-arm result; **this is the first thing to check if the
next valfast census surprises**. The lever, if it ever needs one, is a
charge-quality requirement on the end band — today's only filter is the
inherited blob ≥ 1500 e cut.

Also on record: among the 19 firing nueCC48 events, 6 flip `nue_score` sign vs
legacy, 4 up and 2 down (38856 +1.89 → −1.21, 52672 +0.10 → −2.92). Neither
sample carries truth, so these are recorded, not judged.

### 16.3 Verification of the flip itself

| check | result |
|---|---|
| bare compile (no TLA) | adds exactly one key, `"iso_endpoint": true` |
| `--tla-code iso_endpoint=false` | the key drops out entirely — legacy restorable |
| bare production run == the validated knob-on arm | **byte-identical** on evts 271851 (fires, DL), 168614 (fires), 122660 (fires), 284794 (gate rejects) — `mabc-pr.zip` + `pctree`, `hash_archive.py` member hashes |
| bare run of a gate-rejected event == legacy | evt 284794 identical to the OFF arm |
| `SBND_ISO_ENDPOINT=0` | evt 271851 identical to the pre-flip production base |

Runner: `SBND_ISO_ENDPOINT=0` now restores the legacy wire-footprint boundary
endpoints for an A/B; `=1` is a retained no-op so older invocations still mean
what they said.

```bash
# repro of the flip verification
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./scripts/cfg/compile_prjob_cfg.sh /nfs/data/1/xqian/toolkit-dev/toolkit/cfg /home/xqian/tmp/pr24r3/flip-prjob.json
PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24r3-flip1 data 271851
SBND_ISO_ENDPOINT=0 PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-poc0 work-pr24r3-flipoff1 data 271851
```

---

# Round 4 (2026-08-06) — the trunk still stops short: a downstream regression, not an endpoint-pick bug

**Status: DIAGNOSIS ONLY. No code changed. `iso_endpoint` stays `true` at its
current tuning.** Owner report on three events served on the port-5017 PR
display: "the end point of the track trajectory does not cover the entire
image... it may be due to the knob was not on for the processing, but the
problem can very well be deeper." The knob-off hypothesis is disproven; the
"deeper" hypothesis is confirmed, and localized to code the knob does not
gate.

## 17.0 Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# freshness proof, then the byte-identity + A/B pair (clean HEAD binary, df40b2a4-era):
ls -la /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so
PR_JOBS=3 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr24r4-on3  data 42280 271851 350186
PR_JOBS=3 SBND_ISO_ENDPOINT=0 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr24r4-off3 data 42280 271851 350186

# the premise check
grep -h "iso endpoint" work-nuecc48-cb0805/pr_evt{42280,271851,350186}/wct_pr_evt*.log
```

`work-pr24r4-on3` reproduces `work-nuecc48-cb0805` (the arm served on 5017)
byte-for-byte on `clustering-global`/`track_fit-global`/`vertices-global`
inside `mabc-pr.zip` for all three events; only `0-mc.json` differs, because
`work-nuecc48-cb0805` carries the extra `pr_display` pipeline stage — a
harmless dump-only addition, not a reconstruction difference. So the coverage
numbers below, measured on `work-nuecc48-cb0805`, are exactly what 5017 shows.

## 17.1 The premise check: the knob was on

`iso_endpoint = true` has been the SBND production default since `12d65f1d`
(§16); the runner leaves it on unless `SBND_ISO_ENDPOINT=0`
(`run_pr_chain_batch.sh:331`); and the gate **fired exactly once in each of
the three events**, in the arm 5017 serves:

```
evt42280  iso endpoint: fired (L=125.4 cm, xext=10.6 cm, aspect=0.308, n=15121) A=(14.9,-12.7,91.7)   B=(25.6,14.9,211.6)  gain=11.6/22.7 cm walk=0.00/0.00 cm dperp=0.7/2.0 cm intube=true/true
evt271851 iso endpoint: fired (L= 70.8 cm, xext= 7.5 cm, aspect=0.347, n= 7696) A=(-157.6,31.9,378.4) B=(-155.7,4.1,317.6) gain= 6.5/3.8 cm walk=0.00/0.00 cm dperp=0.2/0.1 cm intube=true/true
evt350186 iso endpoint: fired (L= 49.0 cm, xext=12.8 cm, aspect=0.269, n= 2782) A=(-24.9,83.5,377.2)  B=(-14.0,83.8,335.3) gain= 1.3/4.4 cm walk=0.00/0.00 cm dperp=0.5/0.3 cm intube=true/true
```
(`work-nuecc48-cb0805/pr_evt<ID>/wct_pr_evt<ID>.log:196/168/185`; no
`rejected by aspect` line anywhere; byte-identical in the newer
`work-vfnuecc48-vf0805{a,b,c}`.) No log-level blind spot: DEBUG is on
(`run_pr_evt.sh` invokes `-L debug`), and 280-310 other `D [...]` lines fire
per event.

Owner's "cluster 10" for 350186 does not match any id in the served output —
the point sits in `cluster_id = real_cluster_id = 6`. Likely a 5017-display
indexing artifact (Bee-side list position vs. PR cluster id); flagged, not
chased this round.

## 17.2 The real symptom: `iso_endpoint` ON is *worse* than legacy at the failing end

Measured from `mabc-pr.zip`'s `0-clustering-global.json` (the img cloud) vs.
`0-track_fit-global.json` (the delivered trajectory), same clean binary, same
input, only `SBND_ISO_ENDPOINT` toggled:

| evt | cluster | axis | image extent | **ON** fit extent | **OFF** fit extent | ON undershoot | OFF undershoot |
|---|---|---|---|---|---|---|---|
| 42280  | 8  | z | 88.4 … 212.2 | **99.2** … 211.9 | **89.0** … 211.9 | 10.8 cm | 0.6 cm |
| 271851 | 23 | z | 315.4 … 379.0 | **323.8** … 378.3 | **315.4** … 378.5 | 8.4 cm | **0.0 cm** |
| 350186 | 6  | z | 334.4 … 378.4 | **345.3** … 376.3 | **334.6** … 377.5 | 10.9 cm | 0.2 cm |

This directly contradicts the round-2/3 framing that legacy "picks the widest
wire-footprint pair, two corners on the sheet's edges" and that
`iso_endpoint`'s job is to do better than that. On these three events legacy
reaches the image edge almost exactly, and `iso_endpoint` — the production
default — reaches 8-11 cm *less far*, over well-charged image points (evt
42280's uncovered z∈[88,100) band alone: 357 cluster-8 points, median charge
11-17k; 25.4% of the whole cluster's img points sit >5 cm from any delivered
fit point, of any cluster). The owner's "the problem can very well be
deeper" instinct was right on both counts: it is deeper than the endpoint
pick, and on these events the endpoint pick is not even the wrong direction —
downstream processing is what undoes it.

## 17.3 Ruling out the two nearest suspects

**Not the sheet-aspect gate.** The gate's `aspect` (2nd-PCA transverse over
1st-PCA axial extent, both 2%-trimmed, on the SAME charge-qualified point set
the legacy search uses) was independently recomputed offline from the raw img
cloud (numpy power iteration, same algorithm) for all three clusters:

| evt | C++ aspect (log) | independent recompute | C++ xext (log) | independent xext |
|---|---|---|---|---|
| 42280  | 0.308 | 0.313 | 10.6 cm | 10.6 cm |
| 271851 | 0.347 | 0.349 | 7.5 cm | 7.5 cm |
| 350186 | 0.269 | 0.269 | 12.8 cm | 12.8 cm |

Matches to 2%. The independent axial/transverse extents in absolute terms:
42280's cluster is 94-128 cm long (trimmed/full) and **29 cm wide** in its
own transverse PCA direction — a genuine 2-D sheet by any reasonable
definition, not a track the gate mis-admitted. (The `pr24_iso_probe.py`
"iso-ratio" quoted elsewhere in this doc, e.g. cluster 8 = 0.028, measures a
*different* object at a *different* time — the final ~406 cm merged main
cluster *after* all downstream clustering, not the ~125 cm graph-cluster
`iso_endpoint` gated *before* it. The two numbers were never comparable; the
gate itself checks out.)

**Not `iso_endpoint`'s own walk-in/qualification logic.** All three `fired`
lines report `walk=0.00/0.00` — the axial-extreme search never had to retreat
from spike-support filtering; the raw extreme was accepted on the first try.
The 3-cm gap between the seed and the true image edge (e.g. 42280's A =
z 91.7 vs. image z-min 88.4) is the *lateral-centering* step doing its job:
among points within a 3 cm band of the axial extreme, it picks the one
closest to the principal axis line, not the absolute-extreme point — exactly
the trade this branch exists to make (round 2's whole motivation was to stop
picking sheet-edge corners). That is a small, deliberate, few-cm effect. It
does not explain an *additional* 7-11 cm loss on top of it.

## 17.4 Localizing the loss: it happens after a correct first fit, downstream of `init_first_segment`

Temporary `SPDLOG_LOGGER_DEBUG` checkpoints were added (not committed; source
reverted to HEAD after this measurement, binary rebuilt clean and freshness-
proven again before the §17.0 gate arms) at three points in
`init_first_segment` (`NeutrinoPatternBase.cxx`): the raw endpoint pick, the
Dijkstra rough-path front/back, and the post-`do_single_tracking` fit
endpoints (`v1->fit().point` / `v2->fit().point`). On evt 42280 (single
target cluster, cleanest case):

```
iso endpoint: fired ... A=(14.94,-12.71,91.73) B=(25.56,14.92,211.58)
checkpoint 1  raw endpoints:   A=(14.94,-12.71,91.73) B=(25.56,14.92,211.58)
checkpoint 2  Dijkstra rough:  front=(25.56,14.92,211.58) back=(14.94,-12.71,91.73) npts=79
checkpoint 3  post-fit:        v1=(25.59,14.84,211.96)    v2=(14.57,-12.11,91.56)   fine_n=215
```

**The seed and the very first fit both land within 0.3-0.5 cm of the iso
pick, i.e. ~3.3 cm of the true image edge — matching the legacy arm's own
result to within the small, deliberate lateral-centering offset from §17.3.**
Nothing is wrong yet. `init_first_segment` does its job correctly on this
event, for both the endpoint pick and the trajectory fit.

The loss appears afterward, in the do-multi-tracking / break / find-other-
segments refinement loop every cluster (main or not, every detector) goes
through after its first segment is built (`find_proto_vertex`,
`NeutrinoPatternBase.cxx:2014` → `break_segments` → `find_other_segments` ×
`nrounds_find_other_tracks` → `examine_structure_3` → final
`do_multi_tracking`). Instrumenting `organize_segments_path`'s per-round
end-vertex handling (the function `do_multi_tracking` calls to re-derive fit
points from each segment's *rough* `wcpts()`, not its already-fitted path)
shows the trunk's low-end vertex holding steady at the checkpoint-3 position
for ~30 refinement rounds, then being **replaced outright** — a new `Vertex`
object, a freshly re-run Dijkstra rough path (`wcpts_n` jumps from 18 to 74)
— landing at z≈99, coincident with the moment `find_other_segments`
processes newly-identified untagged fragments elsewhere in the same cluster
(satellite clusters 26/27/30/32 get their own `init_first_segment` calls in
the same log window). The identical instrumentation on the legacy (OFF) arm
shows the *same kind* of replacement event, but landing within ~1 cm of its
own (already-good) seed. So the replacement mechanism itself is shared and
not obviously buggy — what differs is how far it retreats the two arms'
seeds, and that difference (~1 cm legacy vs. ~7.5 cm iso) is the open
question.

**This mechanism is not gated by `iso_endpoint`.** `organize_segments_path`,
`do_multi_tracking`, `break_segments`, and `find_other_segments` run
identically regardless of the knob; the only thing the knob touches is where
`init_first_segment` starts (§17.1's checkpoint 1). The regression this round
found therefore cannot be fixed inside `find_iso_first_segment_endpoints` —
that function is already doing what it is designed to do.

## 17.5 Why this round stops at diagnosis

Continuing to a fix would mean instrumenting and modifying `break_segments`
and/or `find_other_segments` — prototype-ported, universally shared code that
every cluster in every detector's PR chain runs through, with a kink-search
loop (`segment_search_kink`, `proto_extend_point`) whose exact trigger for
this asymmetric retreat is not yet identified. That is squarely the CLAUDE.md
bar for stopping rather than iterating into a guess (§5 rule 5: an
unexplained divergence gets reported, not chased into a fix) and for the
fork-vs-generalize caution around production components with other live
consumers (§2 Code): a same-session, not-fully-understood change to
`break_segments` risks moving physics output for every event in every
detector, silently. Round 1 of this same doc set this precedent — diagnosis
without fix, explicit owner direction requested for the next round — and it
applies again here.

## 17.6 What's confirmed vs. what's open

**Confirmed:**
* The premise ("knob was off") is false; `iso_endpoint` fired correctly.
* `iso_endpoint`'s endpoint pick and the immediate post-fit trajectory both
  land close (~3 cm, by design) to the true edge — round 2/3's fix works as
  intended at the point it operates.
* The 8-11 cm shortfall on the display is manufactured **later**, in the
  shared do-multi-tracking/break/find-other-segments loop, which replaces the
  trunk's low-end vertex with one that has retreated much further than the
  legacy arm's equivalent replacement.
* The sheet-aspect gate is measuring correctly (independently recomputed);
  this is not a gate-tuning problem.

**Open, for the next round:**
* Why does the shared refinement loop retreat the iso-derived seed ~7x
  farther than the legacy seed on these events? Candidates to instrument
  next: `segment_search_kink`'s dQ/dx-median test at the tip (does the small
  iso-vs-legacy seed difference interact with a *nearby* find-other-segments
  fragment to manufacture a kink where none exists on the legacy path?), and
  whether the replacement is a genuine `break_segments` split (with the
  91-99 cm piece surviving as an orphan that's later dropped) or a full
  re-derivation via a different code path.
* This may be the same failure family as round 3's own residual (§15.6,
  271851 geometric arm): the endpoint rule manufacturing a low-quality tail
  segment that a later stage mishandles. Worth checking together.
* 350186's owner-quoted "cluster 10" display-id mismatch (§17.1).

## 17.7 Gates run this round

| check | result |
|---|---|
| freshness proof, clean HEAD binary | `libWireCellClus.so` rebuilt after reverting all diagnostic instrumentation; size byte-identical to the pre-instrumentation build |
| `work-pr24r4-on3` vs `work-nuecc48-cb0805` (the 5017 arm) | `clustering-global`/`track_fit-global`/`vertices-global` identical, all 3 events; only `mc.json` differs (extra `pr_display` stage in the served arm) |
| `work-pr24r4-on3` vs `work-pr24r4-off3` | `clustering-global` identical (upstream imaging unaffected, as expected); `track_fit-global`/`shower_track-global`/`vertices-global`/`mc.json` differ, all 3 events — confirms the knob has a real, measurable downstream effect |

No production code changed this round, so no knob-off byte-identity gate on
the 48-event manifest is required or was run.

## 17.8 What this round leaves behind

* This doc section (§17).
* No code, no cfg, no new knob.
* Scratch arms `work-pr24r4-on3`, `work-pr24r4-off3` (clean-binary, cited
  above) and `work-pr24r4-dbgon2`/`dbgoff2` (instrumented-binary,
  diagnostic-only — the `wct_pr_evt42280.log` files back the §17.4 quotes;
  not gate evidence, not hand-scan records, safe to discard whenever).

## 17.9 Runner hygiene note (found in passing, not fixed this round)

`run_pr_chain_batch.sh:336-341`'s comment claims `SBND_ISO_MIN_ASPECT` /
`SBND_ISO_TUBE_R` are inert unless `SBND_ISO_ENDPOINT=1`. Since the §16 flip
that is stale: both TLAs are emitted whenever the corresponding env var is
set, independent of `SBND_ISO_ENDPOINT`. Consequence: `SBND_ISO_ENDPOINT=0`
plus one of them set produces `iso_endpoint=false` **and**
`iso_endpoint_min_aspect=X` in the compiled config — not byte-for-byte the
pre-`iso_endpoint` config, even though the intent was "legacy A/B". Also, any
value of `SBND_ISO_ENDPOINT` other than `0`/`1` (typo, `true`, `yes`) falls
through to the else-branch and silently means ON. Neither affects this
round's arms (`SBND_ISO_MIN_ASPECT`/`SBND_ISO_TUBE_R` were not set).

## 17.10 Owner decision needed

Same shape as round 1 (§4): a real defect, found and localized to two
specific stack frames, in code too central and too poorly understood (this
session) to touch safely. Before a round 5 attempts a fix, the owner's call
is which of §17.6's open items to chase first — the kink-interaction
hypothesis is the most actionable, but it means instrumenting
`break_segments`/`find_other_segments` directly, which is shared production
code with no per-detector knob boundary to hide behind while iterating.

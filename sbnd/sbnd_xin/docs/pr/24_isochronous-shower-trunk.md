# doc pr/24 — isochronous EM showers: diagnosis, two rejected fixes, and what to try next

**Status: NO FIX ADOPTED.** Both attempts were implemented, measured and then
**removed at the owner's direction** (2026-08-02). Nothing from this round is
in the chain: the toolkit is back to its pre-round state and the compiled PR
job config is byte-identical to the doc pr/23 §9 baseline
(`compile_prjob_cfg.sh` vs `scratchpad/v6flip/postflip-prjob.json`, `cmp`
clean). What survives is this diagnosis, the measurement instrument
(`pr24_iso_probe.py`), one runner convenience (`SBND_DL_WEIGHTS`), and the
next-session plan in §7.

Trigger: owner hand-scan of SBND run 18255 evt 271851 (nueCC48 sample) —
"isochronous shower, not very good shower trunk identification … instead of a
straight track it found some weird two tracks at the beginning of the shower …
the current one does not reach the end of the image (low Y)"; then, on the Bee
pair — "there is a large isochronous blob (img_global), but then there are two
separate tracks from the PR graph, one at the very top and one at the bottom …
this is supposedly an EM shower, from the corner of the blob and then gradually
grow"; and the truth vertex **(−156.3, 6.4, 314.7)**.

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
python3 pr24_iso_probe.py work-pr24-a1    --detail 271851   # segments + 2-D overlap
python3 pr24_iso_probe.py work-pr24-a2poff --detail 271851
python3 pr24_iso_probe.py work-vfmcp1k-prodon --jobs 12 \
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

## 5. Census (instrument: `pr24_iso_probe.py`)

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

* `sbnd_xin/pr24_iso_probe.py` — per-event isochrony / vertex-host / segment /
  2-D projection-overlap / charge-coverage probe, plus the population census.
* `sbnd_xin/run_pr_chain_batch.sh` — `SBND_DL_WEIGHTS` (unset ⇒ cfg default;
  set-empty ⇒ geometric vertex), the attribution arm of §2.
* toolkit: a revert commit removing the swap-guard jsonnet wiring that a
  concurrent session's commit `75c703da` had swept in. Compiled config
  byte-identical to the doc pr/23 §9 baseline; `wcdoctest-clus` 49/49 PASS.
* Arms kept as records: `work-pr24-a1`, `-a2poff`, `-a2geom`, `-a2poffgeom`,
  `-b1on`, `-c3iso`, `work-vf{mcp1k,nuecc48}-pr24{off,on}`, `bee-pr24/`.

# doc pr/104 — main-vertex bias at multi-prong junctions: `vertex_junction_snap`

**Status: SHIPPED — SBND PRODUCTION ON 2026-08-21** (toolkit: knobs DEFAULT OFF
commit + cfg flip commit, hashes in §9; validation §4–§6; Bee §7).

Owner request (2026-08-21, after accepting doc pr/103):

> The improve is good.  At the same time based on my scan I also noticed the
> residual issue.  Essentially for some of the vertex (>= 3-prong junction)
> they are a bit off from the vertex.  One example is my 18255-405707, now it
> is 3-track vertex, but the vertex position is biased because of the
> short-cut.  Now, I wonder if this is something that you can aim to fix in
> this new round?  I wonder if a vertex fitting procedure is what's needed to
> fix this?

Same contract as pr/103: default-OFF knobs, byte-identical OFF gate, validate
on nueCC48 + NCpi0 + mcp1k (1000) + the mcp2k subset, flip ON for SBND
production if validation passes (pre-authorized), Bee before/after + movers.

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# toolkit @ <this round's commits>; wcbuild; ./build/clus/wcdoctest-clus

# offline rule census against the vtx100 labels (read-only, any pr_display arm)
python3 scripts/pr104_junction_census.py work-pr104-bare-mcp1k work-vtx100-base-mcp2k \
        work-pr104-bare-nuecc48 work-pr104-bare-ncpi0 --r 5 --tsv docs/pr/104_census-labelled-r5.tsv
python3 scripts/pr104_junction_census.py work-pr104-bare-mcp1k --r 5 --unlabelled   # footprint

# HEAD baselines (binary == toolkit 5b6b289c source, post-pr/103-flip config)
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-pr104-bare-mcp1k data
# knob-off gate arms (new binary, no env) and knob-on arms:
#   SBND_VERTEX_JUNCTION_SNAP=1 PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh ... work-pr104-on-<sample> data
python3 scripts/pr85_hash_gate.py <bare> <off>; echo rc=$?
python3 scripts/pr90_movers.py <bare> <on> --tags vtx100
python3 scripts/pr103_click_topology_ab.py <bare> <on> --tags vtx100
```

## 1. Why this is not a vertex-fit problem

(filled in §1 below once the round closes; summary of the evidence collected
before any code was written)

- `fit_vertex` (`NeutrinoVertexFinder.cxx:2450`, prototype
  `NeutrinoID_improve_vertex.h:11-42`) is a transverse line-intersection of the
  incident prongs' PCA axes from fit points 1.5–6 cm out, with a
  `sqrt(npoints)/0.43 cm` pull to the current position (`MyFCN.cxx:496-508`;
  relaxed to 1 cm only for 2-leg `mvfit_robust` substitutions, `:453`) and a
  charge veto.  Its gate (`MyFCN.cxx:445`) needs two legs > 15° apart — a
  collinear pass-through (405707 pre-op0: muon halves + the 2.97 cm connector)
  never opens it.  Measured motion scale (docs pr/28, pr/51): 0–1 cm.  The bias
  here is 2–4 cm.
- Nothing re-fits the vertex *position* after `main_vertex_graph_audit`:
  `improve_vertex` runs at `TaggerCheckNeutrino.cxx:2281`, mvga at `:2297`, and
  no `fit_vertex`/`MyFCN` call exists between `:2297` and the publish at
  `:2655` (`grep` of `NeutrinoGraphAudit.cxx`: none).
- **The junction the owner wants already exists as a well-fitted candidate
  vertex.**  `work-pr103-on2-mcp1k` dumps: 65289 J=18004 (deg 3), 66712
  J=11004 (deg 3), 282072 J=7002 (deg 3), 345633 J=8011 (deg 4) — all
  `main_candidate=True`, same cluster as the main vertex, `fit_distance`
  0.15–0.50 cm.  In 65289/66712/282072 the *traditional* per-cluster scorer
  had `trad_winner=True` on J; the DL composite (`route=dl-rerank-accept`)
  overrode it.  345633: DL picked 8010 at 1.63 cm from the click and the
  refinement chain moved it to 8029 at 3.69 cm.
- 405707 is the same shape one stage earlier: M=17026 is a deg-2 point *on*
  the muon (pre-op0: the 2.69 cm stub to the third prong + the 2.97 cm
  connector to J), J=17012 carries muon + proton 2.1 cm back along the muon.
  pr/103's op0 now faithfully re-anchors the prongs onto M — right topology,
  wrong point.

So the error is *which graph vertex carries the main flag*; the fix is a
pointer re-point (the seat `vertex_kink_snap`, doc pr/50, already occupies),
not a fitter.  A line fit does enter — but as the arbiter between two
candidate vertices that both own prongs (tier B below), never as a free
position.

## 2. The rule and its offline census (before any C++)

`scripts/pr104_junction_census.py` evaluates the rule on the calib dumps and
joins the vtx100 labels (600 with truth: mcp1k 342, mcp2k 189, nueCC48 37,
NCpi0 16, delta 16).  Definitions (mirrored 1:1 in the C++):

- **J candidates**: vertices of the main cluster reachable from M through a
  segment chain of total path ≤ R; the chain is "the M–J path" and is
  excluded from both prong counts.
- **prong(X)**: track-like (not shower-flagged; the provisional pdg is not consulted, §3.0) incident segment
  with ≥ 3 cm of path; a shorter stub whose far vertex has degree 2 and
  continues into such a track counts, with the direction along stub +
  continuation.
- **strength(X)**: number of distinct prong *direction classes* (PCA axis of
  the fit points 1–8 cm out); two prongs folding past 150° are one class — a
  track passing through X.
- **tier A**: strength(M) = 0 and strength(J) ≥ 2 — M is a stub end or a
  pass-through point.
- **tier B**: strength(M) ≥ 1, strength(J) ≥ 1, together ≥ 3: joint
  least-squares intersection of *all* prong lines of M and J; snap iff the
  fit point is nearer J than M by > 0.5 cm with RMS transverse residual
  < 1.0 cm.
- Best J = max strength; ties → joint fit of the prongs of M and all tied candidates (§3.0), shortest path only as the fallback.

Census on the HEAD dumps (`docs/pr/104_census-labelled-r5.tsv`, arms
`work-pr104-bare-{mcp1k,nuecc48,ncpi0}` + `work-vtx100-base-mcp2k`; identical
numbers on `work-pr103-on2-*`, same binary/config):

| R (cm) | events with a J | tier A FIX | tier A "BREAK" | tier B FIX | BREAK | MISS (sJ ≤ 1) |
|---|---|---|---|---|---|---|
| 4 | 123 | 1 (282072) | 1 (405707†) | 2 (65289, 66712) | 0 | 33 |
| 5 | 141 | 2 (+345633, path 4.7) | 1 (405707†) | 3 (+38856) | 0 | 34 |
| 6 | 173 | 2 | 1 | 3 | 0 | 43 |
| 8 | 221 | 2 | 1 | 3 | 0 | 53 |

† 405707's carried vtx100 label sits on M (it was carried by position join,
TOL = 1 cm, onto the then-main vertex); the owner's 2026-08-21 statement is
that J is the vertex — the only "BREAK" in the table is the label, not the
rule.  38856 (nueCC48) is a fix-direction move of a vertex that is 24 cm off
either way.

Every MISS row has sJ ≤ 1: the click sits at a stub tip or an elbow (deg 1–2),
a different failure class that no junction rule can address without breaking
the fired set.  **Operating point: R = 5 cm** (needed for 345633's 4.7 cm
path; R = 6–8 adds nothing).  Unlabelled footprint: mcp1k 4/445 dumps
(0.9 %), nueCC48+NCpi0 1/67 (38856).

Caveat recorded: the dumps are post-mvga, so junctions mvga has already eaten
(405707's, after pr/103's op0) are invisible here; the live count is ≥ this.

## 3. Mechanism — `snap_main_vertex_to_junction` (C++ default OFF)

### 3.0 What the first smoke run changed

The first binary honoured `vertex_kink_snap`'s `kProtectedBreak` guard (G1,
as pr/50) and fired on 282072 only.  The trace of the other four exhibits
(`work-pr104-smoke-*`, `SBND_WCT_LOGLEVEL=trace`) showed why, and reshaped
the mechanism:

```
405707  snap_main_vertex_to_kink: SNAP cluster 17 old=(-20.72,169.20,214.42) new=(-19.78,167.03,214.57) turn=55.8 deg arc=3.67 cm
65289   snap_main_vertex_to_kink: SNAP cluster 18 old=(-80.25,65.88,307.57) new=(-80.25,67.27,308.77) turn=97.0 deg arc=1.83 cm
345633  snap_main_vertex_to_kink: SNAP cluster 8  old=(79.57,-115.76,187.43) new=(76.76,-115.68,186.08) turn=42.3 deg arc=4.40 cm
        vjs: cluster N declined (main vertex kProtectedBreak)          <- all three
66712   (no vjs line: the 3.2 cm M-J connector does not exist yet at this stage)
```

**In 3 of the 5 exhibits the main vertex is a product of the kink snap**:
DL selected a vertex 0.4–1.6 cm from the click, `snap_main_vertex_to_kink`
re-seated it onto an image corner of one arm 1.8–4.4 cm away, and stamped
it `kProtectedBreak` — the same bit the pr/48 two-end dQ/dx break uses as a
physics claim.  Hence, in order of discovery:

1. **`VertexFlags::kKinkSnap`** (new bit, set on the kink snap's product next
   to `kProtectedBreak`; nothing serialises the raw flags word, so it is
   inert in every output) + knob **`vjs_override_kink_snap`**: a kink-snap
   product may be arbitrated by the junction snap; two-end breaks and
   `kink_break_protect` vertices stay protected.
2. **Euclidean candidates** (66712): a junction within `vjs_radius` of M in
   the same cluster is a candidate even when no ≤ R graph path exists yet —
   the connector is built later by `improve_vertex`/mvga.
3. **Fit-based tie-break** (65289): two tier-A candidates of equal strength
   (the DL vertex at 0.93 cm path and the owner's junction at 1.87 cm); the
   shortest path is not physics, so a joint fit of the prongs of M and of all
   tied candidates decides (fit_d 0.28 cm vs 2.68 cm → the owner's junction).
4. **EM guard = `kShowerTopology` or pdg 11, never `kShowerTrajectory`, and
   exempt for a kink-snap product** (345633's 3.7 cm break-half inherits the
   broken segment's classification).
5. **Prong counting ignores `particle_info` pdg** (345633's 47 cm pion and
   14 cm muon both still carry pdg 11 when the snap runs; the shower flags are
   the topology verdict).

And two more from the first full validation round (`work-pr104-on3-*`, the
binary with 1–5; §5.0):

6. **`vjs_min_move` = 1.0 cm**: nueCC48 400474 — the kink snap had the vertex
   exactly on the click with all three prongs; a 0.84 cm tier-B move to the
   DL vertex took it off (the one genuine ADVERSE of that round).  Two
   candidates closer than the pr/78 label tolerance are the same point.
7. **Junction-ambiguity arbitration** (replaces the tie-break of item 3):
   mcp1k 281837 — the DL vertex is a deg-1 stub end with TWO partial deg-3
   junctions of a 6-prong star within 5 cm (the baseline's `improve_vertex`
   vertex-activity search assembles the star; the snap pre-empted it on one
   side, 2.87→5.43 cm, Enu 1397→990).  Now, whenever more than one junction
   (degree ≥ 3) is in reach, a joint fit of the prongs of M and of every
   junction must converge (rms < `vjs_fit_rms`) and its nearest junction
   must be a qualified candidate; otherwise decline — never the
   shortest-path fallback.  65289 (two junctions, rms 0.29 cm) still passes.

### 3.0.1 Final-binary smoke (`work-pr104-smoke7-*`, items 1–7, lib 08:10)

```
405707  SNAP tier=A kink_snap_product=true  dist=1.56 cm sM=0 sJ=2 degJ=3 pick=single
65289   arbitrate J1 fit_d=0.28 cm / J2 fit_d=2.68 cm (rms 0.29, 4 lines) -> SNAP tier=A J1 pick=fit
66712   SNAP tier=B  fit_dM=1.97 fit_dJ=0.72 rms=0.74 (euclidean candidate)
282072  SNAP tier=A  dist=2.66 cm sM=0 sJ=2 degJ=3
345633  declined (no candidate junction)                                   RESIDUAL
281837  declined (2 junctions in reach, joint fit does not converge rms=2.87 cm)   item 7 veto
400474  declined (no candidate junction: the 0.84 cm move is below vjs_min_move)   item 6 veto
```

### 3.1 Round-3 ledger (binary with items 1–5, before 6–7)

OFF3 hash gates vs `work-pr104-bare-*`: PASS 30/30, 38/38, 96/96, 2000/2000
(binary #1, items 1–2 absent, also PASS 30/38/96/2000).

| sample | fires | movers (`pr90_movers.py --tags vtx100`) | verdicts |
|---|---|---|---|
| mcp2k (15) | 1 | 405707 2.09 cm "ADVERSE" vs its stale label | the intended move onto J (owner 2026-08-21) |
| NCpi0 (19) | 0 | — | unchanged |
| nueCC48 (48) | 4 | 400474 ADVERSE 0→1.06 cm; 163543 ADVERSE 33→37 cm; 38856 toward; 423981 unlabelled | 400474 → item 6; 163543 far off either way, nue never selected; 423981 nue −1.5→+3.4 (selected; tier A on a kink product whose arms were all still shower-flagged — right direction, weak reason) |
| mcp1k (1000) | 10 | 65289 2.38→0.00, 282072 1.98→0.00, 66712 2.62→0.61, 278046 toward (14.9→14.2); 62459 on (0.32); **281837 ADVERSE 2.87→5.43** | 281837 → item 7; tiers: A 7 (3 kink products), B 3 (1 kink product); all moves 1.9–4.3 cm |


Final smoke (`work-pr104-smoke5-*`, binary of the shipped round):

| event | sentinel | click→main before → after |
|---|---|---|
| 405707 | `SNAP tier=A kink_snap_product=true … dist=1.56 cm sM=0 sJ=2 degJ=3` | 0.00 → 2.09 (onto J; the carried label sits on the kink-snap M the owner overruled) |
| 65289 | `SNAP tier=A kink_snap_product=true … ncand=2 pick=fit` (fit_d 0.28 vs 2.68) | 2.38 → 0.00 |
| 66712 | `SNAP tier=B … euclidean … fit_dM=1.97 fit_dJ=0.72 rms=0.74` | 2.62 → 0.61 |
| 282072 | `SNAP tier=A … dist=2.66 cm sM=0 sJ=2 degJ=3` | 1.98 → 0.00 |
| 345633 | `declined (no candidate junction)` — at snap time J shows sJ=1 (its other arms are still shower-flagged that early) | 3.69 → 3.69 (RESIDUAL) |


`clus/src/NeutrinoVertexFinder.cxx` (new pass next to
`snap_main_vertex_to_kink`), called from `TaggerCheckNeutrino.cxx` **after**
the kink snap and **before** `improve_vertex`, so the vertex fit polishes the
junction with its real prongs and mvga never has to re-anchor prongs onto the
wrong point.  Pointer move only: no segment is edited.  Guards: main vertex
`kProtectedBreak` (G1, as vks); a main vertex that owns a shower prong ≥ 3 cm
is an EM vertex — declined (owner: EM out of scope); J must be in the main
cluster.  Pure helpers `vjs_direction_classes` / `vjs_joint_fit`
(`PRSegmentFunctions.h`, implemented in `MyFCN.cxx`) are doctested in
`clus/test/doctest_vertex_junction_snap.cxx`.

| knob | C++ default | meaning |
|---|---|---|
| `vertex_junction_snap` | false | master switch; false ⇒ the pass returns immediately ⇒ byte-identical |
| `vjs_radius` | 5.0 cm | graph-path reach from M |
| `vjs_min_arm` | 3.0 cm | minimum prong path length |
| `vjs_min_prongs` | 2 | direction classes J must carry (tier A) |
| `vjs_collinear` | 150° | fold threshold for a pass-through pair |
| `vjs_fit_margin` | 0.5 cm | tier B: fit point nearer J than M by this |
| `vjs_fit_rms` | 1.0 cm | tier B: max RMS transverse residual |
| `vjs_override_kink_snap` | false | arbitrate a `kKinkSnap` main vertex too (never a two-end break) |

Plumbing (8 knobs): `TaggerCheckNeutrino.h/.cxx` (configure / default_configuration /
visit cm→units), `NeutrinoPatternBase.h` mirrors,
`doctest_clus_knob_defaults.cxx` pins, `cfg/pgrapher/common/clus.jsonnet`
(key-suppression), `cfg/pgrapher/experiment/sbnd/clus.jsonnet`,
`wct-pr-perevt.jsonnet`, runner env `SBND_VERTEX_JUNCTION_SNAP` /
`SBND_VJS_*`.  Sentinels: `vjs: eval …` (TRACE, one per candidate J),
`vjs: SNAP cluster … tier=A|B M=(…) -> J=(…) dist= path= sM= sJ= degJ=`
(DEBUG), decline reasons at TRACE.

Compiled-config proof: knob-off render of `wct-pr-perevt.jsonnet` md5
`d71d19b9fe714b64e0f15f147c641398` for both the HEAD `cfg/` tree and the
working tree (re-checked after every plumbing addition); knob-on render
carries `vertex_junction_snap`/`vjs_radius`/`vjs_override_kink_snap` exactly
once.  `wcdoctest-clus`: 2347/2347 (225 cases; `doctest_vertex_junction_snap.cxx`
adds 7 cases on `vjs_direction_classes` / `vjs_joint_fit`, incl. the 65289
two-vertex shape).

## 4. Gates (final binary, lib 08:10; `work-pr104-off4-*` vs `work-pr104-bare-*`)

`python3 scripts/pr85_hash_gate.py <bare> <off4>; echo rc=$?` — per sample:

| sample | archives | result | log |
|---|---|---|---|
| mcp2k (15) | 30 | PASS 30/30 | `/home/xqian/tmp/pr104_gate_off4_mcp2k.log` |
| NCpi0 (19) | 38 | PASS 38/38 | `…_ncpi0.log` |
| nueCC48 (48) | 96 | PASS 96/96 | `…_nuecc48.log` |
| mcp1k (1000) | 2000 | PASS 2000/2000 | `…_mcp1k.log` |

Earlier binaries of this round gated identically (binary #1 and #5 — §3.1),
as expected: every change sits inside the knob-on branch or the plumbing.
Compiled config: §3.  Doctests: 2349/2349.  Runtime (mcp1k, 1000 events,
`.time.meta`): wall median 20.0 s both arms (mean 19.3 → 18.9 s), peak RSS
median 1.18 GB both.

## 5. Knob-ON validation (`work-pr104-on4-*`: `SBND_VERTEX_JUNCTION_SNAP=true SBND_VJS_OVERRIDE_KINK_SNAP=true`, numerics at C++ defaults)

### 5.1 Movers (`pr90_movers.py --tags vtx100`, sidecars `104_movers4-<sample>.tsv`)

| sample | labels compared | fires | movers | toward | on | ADVERSE |
|---|---|---|---|---|---|---|
| mcp1k | 356 | 9 | 5 | 4 (65289 2.38→0.00, 282072 1.98→0.00, 66712 2.62→0.61, 278046 14.9→14.2) | 1 (62459, 0.32) | **0** |
| mcp2k (15) | 4 | 1 | 1 | — | — | 405707 "ADVERSE" vs its stale label only (the intended move onto J; owner 2026-08-21) |
| nueCC48 | 39 | 3 | 2 | 38856 (24.5→22.8) | — | 163543 33→37 cm (wrong-object event, nue never selected either way) |
| NCpi0 | 16 | 0 | 0 | — | — | 0 |

Every fire of the final binary (13 events) is listed in §7's Bee index with a
verdict.  Fires by tier: A 9 (kink-snap products 5), B 4 (kink-snap products 1);
all moves 1.56–4.36 cm (the `vjs_min_move` floor never bit — 400474 is the
only < 1 cm candidate and it is declined before evaluation).

### 5.2 Scores (`pr83r3_scores_ab.py`, sidecars `104_scores4-<sample>.tsv`)

- nueCC48: 3 movers; nue-selected (nue > 0) **39 → 40 / 48** (423981 −1.54 → +3.40,
  owner: "looks like a nueCC"); no nue loss.
- NCpi0: 0 movers, byte-identical outputs.
- mcp1k: 9 movers, no numu sign flip in the adverse direction (282072 3.47→4.30,
  66712 2.75→3.55, 65289 2.11→1.36 still selected, 278046 0.57→−0.31 was a
  wrong-object vertex 14 cm off); Enu moves follow the prong ownership change.
- mcp2k: 405707 numu 4.30→3.99, still selected.

### 5.3 Click topology (`pr103_click_topology_ab.py`, `104_clicktopo4-mcp1k.tsv`)

mcp1k 355 labels: LOST-CLICK 0, main-moved 5, ctrk+0 / ctrk−1 (278046, the
14 cm wrong-object event), sc+1/sc−1.  nueCC48: no topology change at any
click except the three movers above.

## 6. Flip

Owner pre-authorization (2026-08-21): "for the rest, it is same as before"
(pr/103: "if the validation of the improvements pass, you can turn them on by
default for SBND production").  Validation passed (§4–§5) → production in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`:

```
vertex_junction_snap = true,      // SBND PRODUCTION ON 2026-08-21 (doc pr/104)
vjs_override_kink_snap = true,    // SBND PRODUCTION ON 2026-08-21 (doc pr/104)
```
numerics at the C++ defaults (5 cm / 3 cm / 2 / 150° / 0.5 cm / 1.0 cm /
min_move 1.0 cm).  Rollback: `-A vertex_junction_snap=false` or the runner env
`SBND_VERTEX_JUNCTION_SNAP=false`.

Flip-equivalence (`work-pr104-flipchk-*`, post-flip config, NO env, vs
`work-pr104-on4-*`): PASS 30/30 (mcp2k), 38/38 (NCpi0), 96/96 (nueCC48),
52/52 (26 mcp1k events: the 9 fires + 17 pr/85–103 exhibits; the list also
named 38856, a nueCC48 event, which the runner reported missing — harmless).
Forced-off (`work-pr104-floff-*`, post-flip config + `SBND_VERTEX_JUNCTION_SNAP=false`)
vs `work-pr104-bare-*`: PASS 38/38 + 96/96.  Compiled config: post-flip default
render md5 `92b472fec292a01d8d45812d50c4beb7` == pre-flip render with the two
TLAs; forced-off render == HEAD render `d71d19b9…`.

## 7. Bee

Annotated index `bee/pr104/pr104.index.txt` (16 events; before = `work-pr104-bare-*`,
after = `work-pr104-on4-*`):

- before: https://www.phy.bnl.gov/twister/bee/set/4db17705-9b64-4978-9e7f-86b8948f4461/event/list/
- after:  https://www.phy.bnl.gov/twister/bee/set/e87695c6-8920-4e14-9e69-4a81212242f3/event/list/

Preview sets shown to the owner mid-round (`bee/pr104/pr104pre.index.txt`):
before a1120446…, after (final-binary smoke) ae7fad8d…, round-3 (binary
before items 6–7, showing the two adverse cases) 1f6a5506….  Owner verdicts
received on the preview: "Things look pretty good. 423981 looks like a nueCC".

Please look: idx 0 (405707), 1 (65289), 2 (66712), 5 (172266), 7 (288287);
idx 14–15 are the two vetoed round-3 adverse cases (identical before/after).

## 8. Residuals and follow-ups

- **345633**: the owner's deg-4 junction is 4.7 cm of path from the kink-snap
  vertex, but when the snap runs its 47 cm pion / 14 cm muon arms are still
  shower-flagged (`kShowerTrajectory`, early shower clustering) and only one
  prong counts.  A second snap opportunity after `improve_vertex` (when the
  flags have settled and the connector exists) would catch it — not done
  here because the second call would have to re-audit the kink snap's
  `kProtectedBreak` contract a second time.
- **Why the kink snap misplaces these vertices**: in 5 of the 13 final fires
  the main vertex was the kink snap's own product (405707, 65289, 61461,
  285753, 288287) — the snap re-seats the DL vertex onto an image corner of
  one arm even when a multi-prong junction sits 2–4 cm away.  A junction-aware
  veto *inside* `snap_main_vertex_to_kink` (decline when a deg≥3 junction is
  within its scan radius) is the cleaner long-term form; the override +
  arbitration here is the reversible version.
- **Early-stage `particle_info` pdg and shower flags are not verdicts** at the
  snap's position in the chain (§3.0 items 4–5); any future pass placed before
  `improve_vertex` must not read them as such.
- **Two `run_pr_chain_batch.sh` incidents** (mcp1k baseline and on3-mcp1k):
  editing the runner while a batch executes it breaks the batch's post-loop
  bookkeeping (bash reads the script incrementally).  Per-event outputs were
  unaffected (all rc=0); the `nusel-table.tsv` merges were redone by hand
  (`./nusel_extract.py --merge`).  Rule recorded in memory: no runner edits
  while any batch is alive.
- The DL re-rank override of the traditional scorer (65289/66712/282072 had
  `trad_winner=True` on the junction) is the pr/100 input this round
  measured but did not act on.

## 9. Ship record

- toolkit `apply-pointcloud`: `a07222e2` (knobs, ALL DEFAULT OFF) + `c550541f` (SBND production flip)
- wcp-porting-img `main`: the commit carrying this doc (this doc, `scripts/pr104_junction_census.py`,
  runner env block, `bee/pr104/*`, `docs/work-tags.md`, sidecar TSVs
  `docs/pr/104_{census,movers4,scores4,clicktopo4}*.tsv`).
- memory: `project_pr104_junction_snap.md`.

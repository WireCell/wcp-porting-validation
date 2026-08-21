# doc pr/103 — near-vertex PR revisited after the fit_exclusion flip: the pass-through junction

**Status: SHIPPED — `mvga_passthru=4`, `mvga_interposed_fallback=true`, `mvga_interposed_fallback_min_angle=45` SBND PRODUCTION ON (2026-08-21, owner pre-authorized on validation PASS); C++ defaults stay OFF.**  Round ran 2026-08-20 22:30 → 08-21 00:40.  Owner request: revisit the
pattern recognition near busy (multi-track, non-EM) neutrino vertices now that
`fit_exclusion` (doc pr/98) is SBND production ON — doc pr/85's near-vertex
rounds predate it — starting from two post-flip scan events, find more such
events in the numu PR sample, build default-OFF improvements, validate on
nueCC48 + NCpi0 + mcp1k tonight, flip on PASS (pre-authorized), Bee links for
the owner.

Owner report (scan set `3617888b`, `bee/scan-prodflip/`):

> 0-405707  vertex PR 3-track … one of the track did not actually go to neutrino
> vertex, and take a short cut.
> 18255-283713  vertex PR not consistent with image?  … the 3-track vertex, but
> the long track seems to deviate a little bit from the image.  Not too bad though.

("0-" is slot notation — doc pr/99 §1; 405707 is run 18255 in **mcp2k**,
283713 in mcp1k.)

## 0. Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# toolkit @ <this round's commits>; wcbuild; ./wcb --target=wcdoctest-clus && ./build/clus/wcdoctest-clus

# near-vertex graph printer (read-only, any pr_display arm)
python3 scripts/analysis/pr103/pr103_vtx_graph.py work-pr103-bare2-mcp2k 405707 --r 5

# busy-vertex census at the reco main vertex (no labels needed)
python3 scripts/pr103_busy_vertex_census.py work-pr103-bare-mcp1k --tsv docs/pr/103_census-bare-mcp1k.tsv

# baselines (legacy binary, HEAD b4670d9b config) -- see sec 5 for the 204-event patch arm
PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-mcp1k-ql0819 work-pr103-bare-mcp1k data
# knob-off gate arms (new binary, no env) and knob-on arms:
#   SBND_MVGA_PASSTHRU=4 ./run_pr_chain_batch.sh ...   (see sec 5/6)
python3 scripts/pr85_hash_gate.py <bare> <off>; echo rc=$?
python3 scripts/pr90_movers.py <bare> <on> --tags vtx100
```

## 1. The two owner events at HEAD

### 1.1 18255-405707 (mcp2k) — the prong that "takes a shortcut": a pass-through junction, and op1 deletes the connector

Graph at the main vertex M = 17026 (production, `work-scan-prodflip-mcp2k`;
`pr103_vtx_graph.py`):

```
 vtx 17026 d_main=0.00 deg=2   (M, the owner's vtx100 click, 0.00 cm)
    seg 17037 len= 23.7  sh=1 pid=11    far=17007(deg1)      <- third prong (EM-flagged)
    seg 17038 len=  2.1  sh=0 pid=211   far=17012(deg3, 2.09 cm)   <- 4-point STUB to J
 vtx 17012 d_main=2.09 deg=3   (J)
    seg 17020 len= 34.9  pid=2212  ang(dir, J->M)=69 deg     <- the proton: ends on J, not on M
    seg 17026 len= 41.2  pid=13    ang(dir, J->M)=14 deg     <- the muon: leaves J TOWARD M and passes
    seg 17038 (the stub)                                         0.6 cm from M (census miss=0.64 cm)
```

The muon's fitted trajectory runs J → (through M) → 37 cm beyond: the
junction J sits 2 cm *behind* the vertex on the muon's back-extension, and the
proton attaches there.  The 2-D overlay (`pr85_panels2d.py`, vertex 17026)
shows the muon's charge passing straight through the star in all three planes.

Trace rerun (`work-pr103-tr-mcp2k`, `SBND_WCT_LOGLEVEL=trace`) — the mvga
sequence that produces the final graph:

```
mvga: op1 dup-merge cluster=17 removed seg len=2.97cm sumdQ=1.04e+05 overlap=1.00@14.0mm
                     vs survivor len=41.16cm sumdQ=1.95e+06 reconnects=0
mvga: op3 stub-interposed cluster=17 len=2.69cm vf_deg=2 carried=1 far_angle=140.4deg   (the third prong)
mvga: op3 decline cluster=17 anchor=sat len=34.84cm reason=ceiling
mvga: op3 decline cluster=17 anchor=sat len=41.16cm reason=ceiling
mvga: op3.5 decline cluster=17 d=2.09cm chord=30.51cm reason=chord-cap
```

1. At mvga time the M–J connector exists (2.97 cm, 1.0e5 dQ).  Because the
   muon's fit passes over it, **op1 reads the connector as a duplicate of the
   muon** (overlap 1.00) and deletes it.  `reconnects=0`: the reconnect plan
   tests "already linked" with `find_segment(le, target)`, which returns the
   loser itself (M–J *is* the loser), so no reconnect is planned.  J is now cut
   off from M.
2. op3 has no interposed stub left to splice; op3.5 declines J on the pr/99
   chord cap (30.51 > 30 cm).  Even with the connector kept, op3's
   collinearity gate would decline: the connector continues no prong at J
   (muon 14°, proton 69° from the J→M direction — the connector *is* the
   muon's back-extension, not a continuation).
3. `stitch_disconnected_main_cluster` later re-adds a 4-point stub (17038,
   pid 211) so the cluster is connected again — the proton still ends on J.

### 1.2 18255-283713 (mcp1k) — "the long track deviates a little from the image"

Post-flip the graph is a clean 3-prong vertex (M = 17003, deg 3: muon 246 cm,
protons 22.8 + 26.0 cm; the owner's vtx100 click is 17003, 0.00 cm).  Pre-flip
(`work-mcp1k-prod0819`) M was 17089 (deg 2) with a 0.5 cm stub to 17003 — the
flip *cleaned* the topology.  The complaint is fit quality: the muon's first
2–3 cm bend through the isochronous proton's charge band (`pr98_fit_panels.py
--box 6`, X-Z).  Perpendicular deviation of the muon's fit points from the
straight line fitted to its own 8–25 cm stretch:

| arc from M | 0.0 | 0.5 | 1.0 | 1.6 | 2.2 | 2.8 | 3.4 | 4.0 |
|---|---|---|---|---|---|---|---|---|
| pre-flip (cm)  | 1.41 | 1.08 | 0.74 | 0.44 | 0.29 | 0.12 | 0.15 | 0.04 |
| post-flip (cm) | 1.31 | 1.36 | 1.42 | 1.05 | 1.05 | 0.72 | 0.39 | 0.36 |

With exclusion ON the muon stays ~1–1.4 cm off its own line for the first
2.5 cm (pre-flip it had relaxed to 0.3 cm by 2.2 cm).  `snap_main_vertex_to_kink`
declines three candidates (turn 27–30° vs bendV 28–30°), so the vertex itself
is right; this is the vertex-adjacent fit, where `update_association`
arbitrates interior points only (the vertex point and its own association are
never arbitrated, `TrackFitting.cxx:3567-3700`).  Measured and parked as a
residual (§9) — the owner's own verdict was "not too bad", and no lever short
of the fitter itself addresses it.

## 2. Population at HEAD (legacy binary, post-flip config) — what fit_exclusion changed

Baselines: `work-pr103-bare-mcp1k` (796 events) + `work-pr103-bare2-mcp1k`
(204 events re-run with the same legacy binary after a build race killed them
— §5), symlink-merged as `work-pr103-baremerged-mcp1k`; `work-pr103-bare2-
{nuecc48,ncpi0,mcp2k}` (48 / 19 / 15 numu50 events).  All `pr_display`.

### 2.1 pr/85 and pr/86 censuses re-measured (mcp1k, vtx100 labels)

`PR85_DUMP_ROOT=work-pr103-baremerged-mcp1k python3 pr85_near_vertex_census.py`
(406 labels with a dump, 348 scorable at 1 cm):

| | pr/85 §2.2 (pre-everything, 462 scorable) | pr/85 §10.7 flip (57-evt census) | **HEAD tonight (348 scorable)** |
|---|---|---|---|
| mode 1a-VIA (prong reaches the click only through stubs) | 21 | 17 interposed | **11** |
| mode 1a-CUT | 8 | 9 | **6** (4 cross-cluster) |
| mode 1b STRADDLE | 4 | 3 | **2** |
| mode 2 (≥2 sub-3 cm segments at the click) | 35 (7.6 %) | 42 stubs | **7 events (2.0 %)** |

`PR86_DUMP_ARMS=… python3 pr86_orphan_census.py`: orphans {VIA 37, CUT 29,
BENIGN 11} over 445 reco anchors; op3 reachability: anchor hop < 2.5 cm 37,
≥ 2.5 cm 11, CUT 29.  The pr/85/86 machinery is doing its job post-flip —
the stub population is a third of what it was; the residual is the *shape*
this round is about, which neither census names.

### 2.2 The busy-vertex census (`scripts/pr103_busy_vertex_census.py`)

Scored at the reco main vertex (no label needed), EM-excluded when the
longest segment touching the vertex is shower-flagged and fewer than two
track-like segments touch it:

| sample | dumps | EM-excl | evaluated | **passthru** | **shortcut** | orphan | stub (<2.5 cm) | busy (≥3 track arms) |
|---|---|---|---|---|---|---|---|---|
| mcp1k (1000) | 445 | 12 | 433 | 0 | **26** (6.0 %) | 10 | 63 | 67 |
| nueCC48 | 48 | 6 | 42 | 0 | 1 | 1 | 8 | 12 |
| NCpi0 | 19 | 2 | 17 | 0 | 2 | 1 | 3 | 5 |
| mcp2k (numu50 15) | 13 | 1 | 12 | **1** (405707) | 2 | 2 | 4 | 4 |

TSVs: `docs/pr/103_census-bare-{mcp1k,nuecc48,ncpi0,mcp2k}.tsv`.

- **passthru** (405707's exact shape — a non-incident prong passing within
  1 cm of M with its end vertex J within 4 cm carrying another prong) is
  rare: 1 in ~500 evaluated events.
- **shortcut** (a ≥3 cm track prong ending on a non-main vertex J within 4 cm
  of M) is 6 % of mcp1k.  Six were re-run at trace level
  (`work-pr103-tr2-mcp1k`: 65289 66712 345633 282072 287517 400856): in
  **every one** op3's interposed splice evaluated the M–J stub and declined
  on the **far-angle gate** (`far_angle` 118.5 / 76.0 / 126.2 / 33.9 / 64.1 /
  122.4° < `mvga_interposed_angle` 130) — the stub (0.6–3.9 cm) continues no
  prong at J; op3.5 declined the same junctions on the pr/99 chord cap
  (chord 88–138 cm > 30).  Two sub-classes, separated in §4.2 by where the
  owner's click is.

## 3. What was built (all C++ default OFF; jsonnet key-suppression; runner envs)

Both live in `main_vertex_graph_audit` (`clus/src/NeutrinoGraphAudit.cxx`);
doctest pins in `clus/test/doctest_clus_knob_defaults.cxx`; TLAs in
`cfg/pgrapher/{common,experiment/sbnd}/clus.jsonnet` and
`wct-pr-perevt.jsonnet`; runner envs `SBND_MVGA_PASSTHRU`,
`SBND_MVGA_PASSTHRU_TOL`, `SBND_MVGA_INTERPOSED_FALLBACK`.

| knob | default | what it does |
|---|---|---|
| `mvga_passthru` (cm) + `mvga_passthru_tol` (cm, 1.0) | 0 = off | **op0 pass-through split**, before op1.  For a non-main, non-protected vertex J of degree ≥ 2 within `mvga_passthru` of the main vertex M, find an incident prong S whose wcpt **polyline** passes within `passthru_tol` of M's wcpt, 0.3 cm ≤ arc-from-J ≤ radius+tol, with ≥ 3 cm of S beyond M, and J keeping another prong.  S is split there: its remainder is re-anchored on M (SegmentPtr identity kept, stale fits trimmed to the remainder), and the J→M piece becomes the connecting stub (an existing M–J segment is reused).  The stub goes into `passthru_stubs`: exempt from op1/op1-post/op1-proj dup-merge (the production failure of §1.1), and admitted by op3 in pass 0 through the **created-splice path** — each remaining prong of J is carried onto M only if its straight chord from M (stub + `mvga_splice_straighten` = 5 cm) passes the es2 charge veto (`straight_steiner_chain`, radius `mvga_straighten_radius` 1.0 cm), then straightened; stub and J are removed when J is left with degree 1.  Sentinels `mvga: op0 passthru-split`, `mvga: op3 created-splice … kind=passthru`, `mvga: op0 fired`. |
| `mvga_interposed_fallback` (bool) | false | **op3 far-angle fallback**: at the main anchor, when an interposed stub is declined by the collinearity gate (`far_angle < mvga_interposed_angle`) **and the far vertex has degree 2** (stub + one prong — a single prong's elbow), take the same per-prong charge-verified straighten splice instead of declining.  Far vertices carrying ≥ 2 prongs are genuine junctions and are left alone (§4.2).  Sentinel `mvga: op3 created-splice … kind=angle-fallback far_angle=…`. |

The wcpt-polyline test is load-bearing: 405707's muon chain has a **5.8 cm
first hop** out of J, so a nearest-wcpt test never sees the pass-through
(first implementation; `work-pr103-on{A,B,C}-mcp2k` are those failed smokes).
The fit trimming and the op1 exemption are equally load-bearing: without
them op1 deletes the connector again from S's stale fits
(`work-pr103-onD-mcp2k`: J left disconnected, proton dangling).

Gate 0: `wcdoctest-clus` 2311/2311; compiled production config with both
knobs off **byte-identical** to the pre-change tree (`md5 b28f2386…` both,
`/home/xqian/tmp/pr103/cfgproof2/{before,after}.json`); knobs on ⇒
`"mvga_passthru" : 4`, `"mvga_interposed_fallback" : true` appear once each.

## 4. Stage A — the smoke set

### 4.1 405707 (op0 + passthru splice), `work-pr103-onE-mcp2k`

```
mvga: op0 passthru-split cluster=17 dJ=2.09cm miss=0.27cm arc=1.76cm rem=42.28cm J_deg=3 stub=existing stub_npts=2
mvga: op3 created-splice cluster=17 stub_arc=2.05cm carried=1 vf_kept=0 kind=passthru far_angle=73.3deg
mvga: fired cluster=17 op1=1 op2=0 op3=2 (refit done)
```
Final graph: M = 17026 **degree 3** — muon 39.3 cm (the remainder, starting at
M), proton 34.8 cm (carried through the stub, straightened, charge-verified),
third prong 22.0 cm; J removed; no stub; the op1 dup-merge of §1.1 no longer
happens (the connector is exempt and the muon's fits are trimmed).  Click
topology (`scripts/pr103_click_topology_ab.py`): `cdeg 2->3 ctrk 0->2 sc 2->0`,
main vertex unmoved (0.00 cm from the owner's click).

### 4.2 The six mcp1k shortcut events (fallback, unrestricted) — and why the fallback is restricted to degree-2 far vertices

`work-pr103-onE-mcp1k` (fallback on, NO degree restriction) vs
`work-pr103-tr2-mcp1k`, scored at the owner's vtx100 click:

| event | far_angle | far deg | click → main (cm) | click topology A→B | verdict |
|---|---|---|---|---|---|
| 287517 | 64.1 | 2 | 0.64 → 0.32 | cdeg 2→3, ctrk 1→2, sc 3→0 | **better** |
| 400856 | 122.4 | 2 | 0.00 → 0.18 | ctrk 1→2, sc 1→0 | **better** |
| 282072 | 33.9 | 2 (chain M–V–J) | 1.98 → 1.84 | two stubs → one (op2 bridge-removal then reconnects) | neutral |
| 65289 | 118.5 | **3** | 2.38 → 2.39 | cdeg 3→1, ctrk 2→1, sc 3→4 | **worse** |
| 66712 | 76.0 | **3** | 2.62 → 2.62 | cdeg 3→2, ctrk 3→2, sc 2→3 | **worse** |
| 345633 | 126.2 | **4** | 3.69 → 3.69 | cdeg 4→2, ctrk 3→2, sc 0→2 | **worse** |

The split is clean and it is not about the angle: in the three "worse" events
**the owner's click is J** (the 3–4-prong junction 2.4–3.7 cm from the reco
main vertex), so collapsing J into M pulls every prong away from the owner's
vertex.  That is a vertex-*placement* error (the DL/rerank vertex sits 2–4 cm
along a prong or, in 282072, on a charge-less bridge that op2 later removes),
and no topology edit can repair it — only move the damage around.  In the
"better" events the far vertex is a **degree-2** elbow of a single prong and
the click is M.  Hence the shipped gate `degree(vf) == 2` (§3): the fallback
is op3.5's approach-collapse shape with a 5 cm charge-verified straighten
instead of the capped whole-chain chord.  The ≥3-prong junction class is
recorded in §9 for the vertex-selection work (doc pr/100).

## 5. Gates (labels; every PASS re-checkable)

- **Gate 0** (§3): `wcdoctest-clus` 2311/2311 at the shipping binary
  (`local/lib/libWireCellClus.so` 23:33 > last edit); compiled production
  config knobs-off byte-identical (md5 `b28f2386356ee64a745d3832c48f0241`
  both), knobs-on keys present once each.
- **Gate 1** (knob-off byte identity, `scripts/pr85_hash_gate.py`, legacy-binary
  baselines vs new-binary off arms, `mabc-pr.zip` + pctree per event):
  - mcp1k: `work-pr103-baremerged-mcp1k` vs `work-pr103-off-mcp1k` — **PASS 2000/2000**
  - nueCC48: `work-pr103-bare2-nuecc48` vs `work-pr103-off-nuecc48` — **PASS 96/96**
  - NCpi0: `work-pr103-bare2-ncpi0` vs `work-pr103-off-ncpi0` — **PASS 38/38**
  - mcp2k (numu50 15): `work-pr103-bare2-mcp2k` vs `work-pr103-off-mcp2k` — **PASS 30/30**
  - the `off` arms ran on the binary one edit before shipping (the degree-2
    restriction sits inside `if (m_mvga_interposed_fallback && …)`, knob-off
    control flow untouched); final-binary re-gate `work-pr103-off2-{nuecc48,ncpi0}`: see below.
- **Incident (recorded, CLAUDE.md M3/M5 family)**: rebuilding `wcbuild` while
  the first mcp1k baseline batch was running overwrote `libWireCellClus.so`
  under 204 in-flight jobs ("failed to load plugin: WireCellClus", rc=1).
  Those 204 events were re-run with the legacy binary restored from a
  `git stash` (`work-pr103-bare2-mcp1k`) and symlink-merged; no later build
  overlapped a run.  Lesson for the runner docs: never install while a batch
  is in flight.

## 6. Knob-ON round 1 (`work-pr103-on-*`: passthru=4, fallback, no angle floor) — the ledger that set the floor

Arms on the shipping-minus-one binary (degree-2 restriction in, angle floor
not yet): mcp1k 1000 / nueCC48 / NCpi0 / mcp2k-15, all rc=0.

- **Footprint** (`pr85_hash_gate.py` bare vs on, mabc member hash): mcp1k
  **17/1000**, nueCC48 3/48, NCpi0 2/19, mcp2k 3/15 events differ; pctree
  never.  Fires: op0 pass-through **3** mcp1k (281837, 316025, 57883) + 38856
  + 405707; fallback 15 mcp1k + 3 + 2 + 2.
- **Movers** (`pr90_movers.py --tags vtx100`): mcp1k 356 compared, 11 movers
  > 0.05 cm — **ADVERSE 0**, toward 4, on 5, away 2 (389588 4.42→4.58 and
  313781 40.88→40.94 cm, both already-lost vertices); nueCC48 / NCpi0 / mcp2k
  **0 movers**.  The ops never move the main-vertex pointer; the sub-cm moves
  are op4-refit polish.
- **Click topology** (`pr103_click_topology_ab.py`, 355 mcp1k labels):
  ctrk+ 6, ctrk− 3, sc− 5, LOST-CLICK 1 (389588, click 4.4 cm from the main
  vertex in both arms).  nueCC48 39 / NCpi0 16 labels: no class changes.
- **Scores** (`pr83r3_scores_ab.py`): nueCC48 — **no nue change on any
  event**; numu moves on 138009 (0.70→−0.45) and 235435 (0.98→0.07, Enu
  818→555).  NCpi0 — 180801 nue −15→−3.2 (still negative), numu 0.64→1.36.
  mcp1k — 16 movers, numu sign flips **only positive**: 284794 −1.28→**+3.30**
  (the pr/102 hand-check event: its 276.7 cm muon now starts at the vertex
  instead of 1.6 cm away through a stub), 292533 2.47→4.30 with **cosmict
  1→0** (0.8 cm stub between the 270 cm muon and the 66 cm proton dissolved
  — a clean 2-track vertex), 316729 1.78→3.96, 400856 2.45→3.81; 405707 (mcp2k)
  numu 1.46→4.30.
- **Census** (`pr103_busy_vertex_census.py`, mcp1k 433 evaluated): shortcut
  **26→20**, orphans **10→3**, stubs 63→57; mcp2k shortcut 2→0, orphans 2→0.

**Per-fire adjudication by measured `far_angle`** (the created-splice
sentinel carries it):

| far_angle band | fires | outcome |
|---|---|---|
| **0.0° (direction unmeasurable, fit-less prong)** | 235435 ×2, 38856? | **ADVERSE**: 235435's two 10–11 cm prongs at vertex 2014 cut off the main vertex, Enu 818→555 MeV — the legacy gate declines unmeasured directions and the fallback must too |
| < 45° (hairpin / back-fold: prong leaves J nearly parallel to the stub) | 389588 10.9, 235435 8.7, 38856 13.4, 282072 33.9, 394532 38.4, 314705 41.5 | tracks shortened by the carry + op1 re-merge (389588 Enu 82→70, 314705 117→100, 394532 305→283); 282072 neutral |
| 45–130° | 284794 68.2, 292533 128.3, 316729 88.5, 400856 122.4, 315167 107.4, 390182 108.1, 55595, 400636, 138009 125.7, 180801, 285567 ×3, 475140, 281165, 235435 47.1, 349945 ×2, 313781 | neutral-to-**better** (the four numu gains above; 285567 degree 5 unchanged; 349945 nue −4.3→−15 on a numu event) |

Decision: a third knob `mvga_interposed_fallback_min_angle` (deg; C++ default
0 = "measured only" — `best_angle > 0` is now always required), production
value **45**; round 2 (§7) re-runs all four samples with it.

## 7. Knob-ON round 2 (`work-pr103-on2-*`, shipping binary, passthru=4 + fallback + min_angle=45) — PASS, flipped

All four samples rc=0 (1000 / 48 / 19 / 15).

- **Fires**: op0 pass-through 3 mcp1k (281837, 316025, 57883) + 38856 + 405707;
  fallback 11 mcp1k + 2 nueCC48 + 2 NCpi0 + 2 mcp2k — the three sub-45° and
  the 0.0° round-1 fires are gone (389588, 314705, 394532, 282072, and
  235435's three bad fires).
- **Footprint** (mabc member hash): mcp1k **13/1000**, nueCC48 3/48, NCpi0
  2/19, mcp2k 3/15; pctree never differs.
- **Movers** (`pr90_movers.py --tags vtx100`; `docs/pr/103_movers2-*.tsv`):
  mcp1k 356 compared, 8 movers > 0.05 cm — **ADVERSE 0**, toward 3, on 4,
  away 1 (313781, 40.9 cm lost vertex, +0.06 cm); nueCC48 39 / NCpi0 16 /
  mcp2k 4: **0 movers**.
- **Click topology** (`docs/pr/103_clicktopo2-*.tsv`): mcp1k ctrk+ 6, ctrk− 1
  (313781), sc− 5, **LOST-CLICK 0**; nueCC48 / NCpi0 no class change.
- **Scores** (`docs/pr/103_scores2-*.tsv`): nueCC48 **no nue change**; numu
  on 138009 0.70→−0.45, 235435 0.98→0.69 (Enu 818→**878**, the round-1 555 is
  gone), 38856 −0.34→+1.24.  NCpi0: 180801 nue −15→−3.2 (negative both arms).
  mcp1k 12 movers; numu sign flips **only +**: 284794 −1.28→+3.30;
  281837 / 390182 stay positive (1.14→0.76, 1.44→0.84); 292533 cosmict 1→0.
- **Census** (mcp1k 433 evaluated): shortcut **26→21**, orphans **10→4**,
  stubs 63→57; mcp2k shortcut 2→0, orphans 2→0.
- **Runtime** (`pr_scores_table.py`, 987 paired mcp1k events): wall median
  12.8→13.4 s, p90 15.9→17.0 s, RSS median 1.18→1.18 GB — the knobs touch
  1.4 % of events; the difference is box load (the bare arm ran under a
  lighter concurrent load).

**Flip** (owner pre-authorized on PASS): `wct-pr-perevt.jsonnet`
`mvga_passthru = 4`, `mvga_interposed_fallback = true`,
`mvga_interposed_fallback_min_angle = 45` (`mvga_passthru_tol` rides the C++
1.0 cm).  Compiled-config proofs: flip-bare == explicit-on (`cmp` equal);
forced-off == pre-flip modulo the two zero-valued keys; each key appears
exactly once.  Flip-equivalence arms: post-flip config, NO env, `work-pr103-flipchk-{mcp1k(14 fired events),nuecc48,ncpi0,mcp2k}` vs `work-pr103-on2-*` — **PASS 28/28, 96/96, 38/38, 30/30**; post-flip forced-off (`SBND_MVGA_PASSTHRU=0 SBND_MVGA_INTERPOSED_FALLBACK=0 SBND_MVGA_INTERPOSED_FALLBACK_MIN_ANGLE=0`) `work-pr103-floff-{nuecc48,ncpi0}` vs the legacy baselines — **PASS 96/96, 38/38**; `"mvga_passthru" : 4` present in 19/19 flipchk compiled configs, `: 0` in 19/19 floff ones.  Rollback at any time: `-A mvga_passthru=0 -A mvga_interposed_fallback=false`.

## 8. Bee A/B and what to look at

- before (legacy binary, HEAD config): https://www.phy.bnl.gov/twister/bee/set/16ef8189-905a-453f-8018-f28bb10f49e1/event/list/
- after (shipped operating point):   https://www.phy.bnl.gov/twister/bee/set/a1369848-8909-4e11-ada6-a2ec6a9c8d46/event/list/
- annotated index: `bee/pr103/pr103.index.txt` (26 events: owner's two first,
  then every event a knob fired on, then three pr/85 exhibits).

Owner look list: **idx 0 (405707)** the fix; **idx 2 (284794)** and **idx 3
(292533)** the two numu rescues; hand-checks on **7 (281837)**, **10 (38856)**,
**12 (138009)**, **14 (180801)**, **20 (390182)** — all score movers whose
topology change is a single re-anchored prong.

## 9. Residuals and follow-ups

- **283713**: the muon's vertex-adjacent 1–1.4 cm fit deviation (§1.2) — a
  fitter-level question (vertex point / endpoint association never
  arbitrated by `update_association`); no lever in this round.
- **The ≥3-prong far-vertex class** (65289, 66712, 345633 …): the owner's
  click is the junction J 2.4–3.7 cm from the reco main vertex; deliberately
  NOT collapsed (degree-2 restriction).  This is a vertex-placement failure
  (DL/rerank lands on a prong or a charge-less bridge) — input for doc
  pr/100's rerank tuning, where "a multi-prong junction within 4 cm" is a
  candidate feature.  Count: 3 of the 6 traced shortcut events.
- op1's reconnect plan cannot reconnect through the loser itself
  (`find_segment(le, target)` returns the loser): left as is — the correct
  repair is op0 (a rough-path reconnect would re-create the same duplicate).
- Hairpin / back-fold prongs (far_angle < 45°): untouched by design; 389588,
  314705, 394532 keep their stubs.
- `pr103_busy_vertex_census.py`'s `kink` column (first-3 cm deviation from
  the 5–20 cm line) flags ~23 % of vertices and is not a defect metric as
  built — kept as a screen only.

## 10. Ship record

- toolkit `apply-pointcloud`: (hashes in the commit log) knobs DEFAULT OFF +
  SBND production flip.
- wcp-porting-img `main`: this doc, `scripts/pr103_busy_vertex_census.py`,
  `scripts/pr103_click_topology_ab.py`, `scripts/analysis/pr103/pr103_vtx_graph.py`,
  `vtx_rules/vtx_io.py` (`TAGS_VTX100`), `scripts/pr90_movers.py` (`--tags vtx100`),
  runner envs, `docs/pr/103_*.tsv` sidecars, `bee/pr103/` sidecars, `docs/work-tags.md`.

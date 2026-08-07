# doc pr/38 — Particle flow silently drops fitted tracks (SBND run 18255, 7 events)

## Repro block

```bash
# baseline (pre-fix binary), knob-off (post-fix binary, knobs forced 0),
# knob-on (post-fix binary, cfg defaults = ON) arms, this box (NOT wcgpu1):
cd sbnd_xin
PR_JOBS=4 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-nuecc48-cb0805 work-pr38-base7   sim 219295 234638 447477 489330
PR_JOBS=3 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-mcp1k-cb0805   work-pr38-base7mc sim 52657 55715 56243
SBND_PF_BARRIER_SEGMENT_VERTICES=0 SBND_PF_ORPHAN_TRACK_ROOTS=0 \
  PR_JOBS=4 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-nuecc48-cb0805 work-pr38-off7    sim 219295 234638 447477 489330   # + mcp1k -> work-pr38-off7mc
WCT_BEE_PF_PRINT=1 PR_JOBS=4 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
    work-nuecc48-cb0805 work-pr38-on7     sim 219295 234638 447477 489330   # + mcp1k -> work-pr38-on7mc
# gate + recovery check:
python3 pr38_cmp.py
```

## Symptom

Owner report (2026-08-05, hand scan of the port-5017 PR displays): the
particle flow (Bee `mc.json` tree) sometimes misses tracks that are plainly
present in the fitted trajectory + dQ/dx display.  Screenshot
(`docs/pics/Screenshot 2026-08-05 at 10.09.36 PM.png`, evt 489330): two
tracks hang off the pi+ endpoint; one shows (as a mis-ID'd e-, separate
issue), the other is absent from the flow entirely.  Affected events, run
18255 (owner coordinates, cluster ids from the wcgpu1 cb0805 displays):

| evt | (x,y,z) cm | owner id | sample |
|---|---|---|---|
| 219295 | (27.8, 41.6, 157.1) | 15001 | nueCC48 |
| 234638 | (95.3, -58.3, 414.3) | 10028 | nueCC48 |
| 447477 | (117.9, -28.8, 377.7) | 11027* | nueCC48 |
| 489330 | (-46.9, -107.3, 151.2) | 4044 | nueCC48 |
| 52657 | (-33.7, -54.1, 257.8) | 23000* | mcp1k |
| 55715 | (-33.6, -25.8, 485.3) | 15006 | mcp1k |
| 56243 | (130.6, -4.0, 125.2) | 9001 | mcp1k |

\* actual missing segment at those coordinates: 447477 → seg 11063 (the
owner id 11027 is the e- shower there), 52657 → seg 23006 (23000 is the
cluster).  In every event the missing object is a main-cluster,
non-shower (`shower_id == -1`), PID'd **proton** segment (plus one 3-point
muon stub in 55715) with a full fitted trajectory and dQ/dx.

**Correction to the report**: the "p 4.5 dQ/dx" row the owner matched to the
missing 219295 track is a *different* object — a 1.0 cm proton blip, shower
49053 at (138.2, 42.6, 173.8), `kine_dQdx` 4.54 MeV.  The missing proton
seg 15001 is absent from the kine energy list too (see "Residual" below).

## Root cause

Two stacked toolkit-vs-prototype divergences in the particle-flow assembly
(`MultiAlgBlobClustering::fill_bee_pf_tree`):

1. **Over-wide shower-vertex barrier.**  The F2 barrier (doc pr/34 §10.3,
   `pf_shower_vertex_barrier`, SBND ON) pre-seeds the track BFS's visited
   set from every shower's vertex set via `Shower::fill_sets`.  The toolkit
   shower *view* contains the shower's **start vertex**
   (`Shower::set_start_vertex` → `add_vertex`, `PRShower.cxx:94`); the
   prototype's `WCShower::fill_sets` fills from `map_vtx_segs` — vertices
   incident to the shower's *own segments* only (`WCShower.cxx:596-599`;
   every `map_vtx_segs` insert is member-segment-keyed, `:547/:688/:698`;
   `set_start_vertex` stores the bare pointer, `:529-532`).  For a detached
   conn-type-2/3 shower the start vertex is a junction **on the main track
   tree**.  Blocking it stops the BFS cold: in evt 219295 the 9.5 MeV e-
   shower 64068 attaches (conn 2) at junction vtx 15002 = the pi+ far
   endpoint, so the BFS never expands there and proton seg 15001
   (13 fit points, (27.8,41.5,157.8)→(30.1,40.9,150.9)) is never claimed.

2. **Dropped orphan segments.**  The prototype's `fill_particle_tree` gives
   EVERY non-shower main-cluster segment a node with `mc_mother=0` in a flat
   loop *before* mother assignment (`NeutrinoID.cxx:1485-1489`), so a
   BFS-unreached segment stays in its tree as a root-level node.  The
   toolkit built the tree top-down from the BFS only and skipped
   disconnected segments entirely ("to avoid adding zero-energy orphaned
   particles" — the zero came from reading endpoints out of the BFS-only
   `seg_endpoints` map, not from the segment, which has perfectly good fit
   points).

## Why it hid

The pr/34 A/B arms measured F2 (the barrier) as moving mc.json in only 5/48
nueCC events, all reviewed as topology *re-parenting*; a node vanishing
entirely reads as "shower absorbed the segment" unless cross-checked
against the segment table, and nothing did that automatically.  The kine
energy list is assembled by a *separate* BFS with the same barrier, so the
proton is missing there too — the display gives no contradiction to notice
unless one counts fitted trajectories against flow nodes, which is exactly
what the owner's hand scan did.

## Fix

Toolkit commit (this round), two independent `bee_pf` knobs, C++ defaults
**false** (legacy path byte-identical), SBND production defaults **ON** in
`wct-pr-perevt.jsonnet`:

- `pf_barrier_segment_vertices` (F1): build the F2 barrier from vertices
  incident to each shower's own segments (prototype `map_vtx_segs`
  semantics) instead of the full view — the detached conn-2/3 start vertex
  no longer blocks the track BFS.
- `pf_orphan_track_roots` (F2): after the root-track loop, emit every
  BFS-unreached, non-shower, non-conn4, main-cluster segment as a
  root-level leaf: prototype node conventions (endpoints from the segment's
  own fit points oriented by dirsign, `NeutrinoID.cxx:1217-1239`;
  `dirsign==0` not plotted, `:1215`), KeepMC floors applied as everywhere.
  Emission sorted by encoded id (prototype order is pointer-map order —
  irreproducible; stable order chosen deliberately).

Runner tri-state env overrides added to `run_pr_chain_batch.sh`:
`SBND_PF_BARRIER_SEGMENT_VERTICES`, `SBND_PF_ORPHAN_TRACK_ROOTS`.

## Verification

Freshness proof done; `wcdoctest-clus` 984/984.  All arms this box (the
wcgpu1 cb0805 outputs are cross-machine FP-shifted and not byte-comparable;
both symptom-bearing FP paths reproduce locally in 5/7 events, below).

- **Compiled config**: knobs-off compile == pre-change HEAD compile,
  byte-identical (`cmp` PASS); both keys present when on.
- **Knob-off gate PASS**: `work-pr38-base7`+`work-pr38-base7mc` (pre-fix
  binary) vs `work-pr38-off7`+`work-pr38-off7mc` (post-fix binary, both
  knobs forced 0): 7/7 events × {mabc-pr.zip (hash_archive), pctree-pr
  (hash_archive), calib-pr JSON, nusel TSV} all identical.
- **Knob-on** (`work-pr38-on7`, `work-pr38-on7mc`): moves ONLY
  `mabc-pr.zip::data/0/0-mc.json`, and only in the 3 events with recovered
  nodes; pctree/calib/nusel byte-identical 7/7 (display-only confirmed).
  - evt 219295: **proton 97 MeV recovered** as daughter of pi+ 238 MeV
    (F1 path, `parent=15003`); the 21 MeV gamma re-parents from root to
    under the proton (its attachment vertex is the proton's far end —
    prototype `find_incoming_segment` behavior).
  - evt 489330 (the screenshot event): **proton 115 MeV recovered** at root
    (F2 orphan path — genuinely unreachable even with the corrected
    barrier).
  - evt 56243: **proton 173 MeV recovered** as daughter of proton 124 MeV
    (F1 path, `parent=9003`); the 6 MeV gamma re-parents under it.
  - evt 447477 / 52657: the missing stubs (3-point / 2-point protons) are
    now **claimed by the BFS** (`ADD track-node` at 3 MeV / 8 MeV) but
    remain hidden by the KeepMC nucleon display floor (`np_ke_min` 10 MeV
    — prototype `WCReader::KeepMC` parity, a pre-existing config knob, not
    this bug).  Lower `np_ke_min` if they should show.
  - evt 234638 / 55715: the local FP path differs from wcgpu1's (55715's
    segs 15006/15037 are shower members locally, `shower_id 15005`;
    234638's seg 10028 does not exist locally) — no missing main-cluster
    track locally, nothing to recover.  On wcgpu1 both had `shower_id -1`,
    i.e. the same claimed-or-orphan mechanisms apply there.

## Residual (NOT fixed here — flagged for a follow-up round)

`fill_kine_tree` (NeutrinoKinematics) runs its own BFS with the SAME
over-wide `fill_sets` barrier, so these protons are missing from
`kine_energy_particle` and from `kine_reco_Enu` as well: **~97-173 MeV of
real proton KE per affected event is absent from the reconstructed neutrino
energy** (numu-BDT var 69, nue reader input — doc pr/35 territory).  Fixing
it is the same endpoint-semantics change in `NeutrinoKinematics.cxx` but is
NOT display-only, so it needs its own knob, gate and score-shift review.

A second residual divergence: a shower whose start vertex sits on an orphan
segment attaches to root here; the prototype would attach it to the orphan
via `find_incoming_segment`.  Display-only, cosmetic, out of scope.

---

# Round 2 (2026-08-05 evening) — owner regressions, corrected in place

## Repro block

```bash
cd sbnd_xin
# arm A: 43f1fe25 binary; arm B: corrected binary; both barrier OFF, same cfg
SBND_PF_SHOWER_VERTEX_BARRIER=0 PR_JOBS=4 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr38b-off{A,B} sim 219295 234638 447477 489330   # + mcp1k -> off{A,B}mc
# ON arm (cfg defaults):
WCT_BEE_PF_PRINT=1 PR_JOBS=4 PR_EXTRA_STAGES=pr_display \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr38b-on7 sim 219295 234638 447477 489330        # + mcp1k -> on7mc
python3 pr38_cmp.py work-pr38b-offA work-pr38b-offB work-pr38b-on7
# H1 (geometric vertex): SBND_DL_WEIGHTS= ./run_pr_chain_batch.sh ... 234638 / 55715
```

## Owner report (5 problems) and what each turned out to be

1. **52657 8-MeV vertex proton absent** — claimed by the BFS, hidden by the
   10-MeV KeepMC nucleon floor.  FIXED by the floor change (below).
2. **447477 3.2-MeV vertex proton absent** — same.  FIXED.
3. **234638 "totally messed up, no pi/protons"** — NOT the pr/38 knobs:
   mc.json is byte-identical across base7/off7/on7 for this event.  See
   "Session divergence" below.
4. **489330 proton at root instead of under the pi+** — a genuine round-1
   implementation error.  FIXED (below); the proton 115 MeV is now claimed by
   the BFS with `parent=4019` (the pi+ 186 MeV).
5. **55715 "clustered as one EM shower"** — NOT the pr/38 knobs (byte-identical
   across arms); segs 15006 (proton) / 15037 (mu) carry `shower_id 15005` in
   this session's reconstruction.  See "Session divergence".

## Round-1 error and the corrected semantics (owner: NO new knobs)

The round-1 barrier ("member-segment endpoint vertices") was WRONG for
conn-1 showers: the prototype's barrier source `map_vtx_segs` NEVER holds the
shower's start vertex — every write path skips it
(`if (vtx == start_vertex) continue;`, WCShower.cxx:547 two-arg
set_start_segment, and both loops of complete_structure_with_start_segment
:708-716/:733-745) — *including* the conn-1 case where the start vertex is an
endpoint of the start segment.  Round 1 re-included those, so 489330's
junction 4017 (pi+ end = proton start = conn-1 e- 64 MeV start) stayed
blocked.  Round 1's `find_vertices`-based build also silently skipped
stale-descriptor shower segments (PRGraph.cxx:252 returns {}), weakening the
barrier unpredictably.

Corrected implementation, folded under the EXISTING pr/34 F2 knob (both
round-1 knobs `pf_barrier_segment_vertices` / `pf_orphan_track_roots`
REMOVED from C++, jsonnet and the runner):

- `Shower::fill_sets` gains a defaulted `exclude_start_vertex` parameter
  (PRShower.{h,cxx}); default false = view semantics, all other callers
  byte-identical.  `fill_bee_pf_tree` passes
  `exclude_start_vertex = cfg.pf_shower_vertex_barrier` — no graph lookups,
  interiors block, attachment junctions traverse.
- The orphan root-leaf safety net is now gated by `pf_shower_vertex_barrier`
  as its no-silent-drop complement.
- SBND `np_ke_min` 10 → **3 MeV** (owner decision 2026-08-05): sub-10-MeV
  protons attached at the neutrino vertex must show.  `em_ke_min` 5 MeV
  unchanged.  Side effect (expected): small conn-3 proton showers ≥3 MeV now
  render (e.g. 219295's 4.5-MeV blip 49053; 234638's 3.3/5.5-MeV pairs).

## Verification

- `wcdoctest-clus` 984/984; freshness proof done.
- **Binary gate PASS**: work-pr38b-off{A,Amc} (43f1fe25 binary) vs
  work-pr38b-off{B,Bmc} (corrected binary), same compiled config, barrier
  forced OFF: 7/7 events × {mabc-pr.zip, pctree, calib-pr, nusel} identical.
- **Compiled-config diff vs HEAD** (full-pipeline TLA): exactly
  `np_ke_min 10→3` + the two removed pr/38 keys.  Nothing else.
- **ON arm** (work-pr38b-on7{,mc}) vs the pr/34-legacy ON (work-pr38-base7):
  ONLY `data/0/0-mc.json` differs (6/7 events; 55715 unchanged);
  calib-pr and nusel TSVs byte-identical 7/7 — display-only confirmed.
  Trees: 219295 proton 97 MeV under pi+ 238; **489330 proton 115 MeV under
  pi+ 186** (`ADD track-node seg=4044 parent=4019`); 56243 proton 173 MeV
  under proton 124; 52657 proton 8 MeV and 447477 proton 3 MeV at the
  neutrino vertex.
- Note: on these 7 events the corrected-barrier-ON output coincides
  byte-for-byte with the barrier-OFF output — no track BFS path crosses a
  shower interior here; the barrier's only firings were the (wrong) junction
  blocks the correction removed.

## Session divergence (234638 / 55715) — measured, NOT a display bug

> **RETRACTED by Round 3 (2026-08-05 night, below).** The divergence is the
> `reality` TLA (pos_offset transverse calibration), not DL-vertex session
> instability. The DL vertex is consumed only at stage 11
> (TaggerCheckNeutrino) and cannot move the stage-4 CreateSteinerGraph
> counts quoted here; the H1 "afternoon DL == geometric" result was a
> sim==sim comparison. The byte-evidence in this section (base==off==on for
> these two events; single-round comparison rule) still stands.

`scratch_wcgpu1` is a symlink to this NFS tree: everything ran on ONE host.
The owner's cb0805 morning round (10:43-11:22, reproducible ×4 incl. the
vf0805a/b/c repeats) and every afternoon run (19:53+, reproducible ×3) use
the same binary lineage, the same compiled config components (42==42), the
same pr/33 knob states (PR33AUDIT), and the same input pctrees (mtimes
predate the morning PR runs) — yet reconstruct differently: first log
divergence at CreateSteinerGraph (79 vs 80 clusters, 1 vs 2 in-window mains,
cluster-4 scope flip), and the main vertex moves:

- 234638: (102.82,-6.68,363.97) morning vs (103.47,-5.01,365.97) afternoon.
  H1 measured: the afternoon DL-mode vertex EQUALS the pure-geometric vertex
  (SBND_DL_WEIGHTS= rerun is byte-compatible), i.e. the DL re-rank changed
  nothing in the afternoon session but moved the vertex 2.3 cm in the
  morning one.  Downstream, the pi+ 120/protons 123+262/pi0 156 structure
  becomes two e- showers (proton seg absorbed with `shower_id 10067`).
- 55715: DL engaged in both sessions ((-48.00,-38.21,482.13) vs
  (-47.99,-38.10,481.43), 0.7 cm apart — M4's documented SCN
  non-bit-stability) and that drift flips shower clustering: the pi+ 38 MeV
  with an e- child becomes one e- 105 MeV shower absorbing segs 15006/15037.

Consequence (procedural, owner-facing): the DL/SCN vertex is not stable
across sessions, and near-threshold events flip topology wholesale.  PF
displays and PF-tree validation must compare arms from a SINGLE production
round; re-running "the same" chain later is not a byte-level reference.  The
5017 display should be re-pointed at one consistent round (e.g. the
work-pr38b-on7{,mc} arms for these 7 events, or a fresh full production).

## Still open

- kine_reco_Enu residual (round 1 §Residual) — the `fill_sets(...,
  exclude_start_vertex)` parameter added here is now actually threaded, in
  doc pr/43 (`kine_shower_vertex_barrier`, `NeutrinoKinematics.cxx`
  `fill_kine_tree`), which mirrors this round's `pf_shower_vertex_barrier`
  fix + orphan safety net on the kine-BFS side. Mechanism-complete but
  **not flipped**: pr/43's G3/G4 census showed the fix (combined with four
  other PID knobs from the same round) moves `kine_reco_Enu` on 42/48
  nueCC48 events, held for owner review rather than auto-flipped.
- Whether SBND production PR should pin the geometric vertex for
  reproducibility (or accept per-round DL variance) — owner decision.

# Round 3 (2026-08-05 night) — real root cause of the 234638/55715 "mess": the `reality` TLA. Round-2 DL claim RETRACTED

## Repro block

```
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
# discriminator (logs already on disk; 12/12 + 7/7 across 3 binary builds):
for d in work-*/; do m=$d.batch_pr_evt234638.log; [ -f "$m" ] || continue; \
  printf '%-28s %s ' "${d%/}" "$(grep -o 'reality=[a-z]*' $m)"; \
  grep -ho 'kept [0-9]* of [0-9]* cluster(s) ([0-9]* in-window[^)]*)' \
    ${d}pr_evt234638/stdout.log | head -1; done
# decisive experiment (HEAD df40b2a4 binary, cb0805 input pctrees):
PR_JOBS=4 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr38c-data7   data 219295 234638 447477 489330
PR_JOBS=3 ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr38c-data7mc data 52657 55715 56243
# gate: member-hash tsv/pctree/zip-sans-mc.json vs the cb0805 arms -> 7/7 SAME
```

## Retraction

Round 2's "Session divergence" section attributed the 234638/55715
reconstruction flips to DL/SCN-vertex session instability. That is wrong,
on two independent grounds:

1. **Pipeline order.** `dl_weights` is consumed only by TaggerCheckNeutrino
   (stage 11; TaggerCheckNeutrino.cxx:189/:803 → NeutrinoVertexFinder.cxx:3499).
   The first divergence is at CreateSteinerGraph (stage 4) — 79 vs 80
   clusters, 1 vs 2 in-window mains — which no vertex-stage difference can
   produce.
2. **The H1 experiment was vacuous.** The `SBND_DL_WEIGHTS=` rerun that
   "matched the afternoon lineage byte-for-byte" also ran `reality=sim`, so
   it compared sim against sim. It said nothing about the DL vertex.

## Root cause

`run_pr_chain_batch.sh` takes `reality` as a required positional argument.
The owner's production rounds (`work-nuecc48-cb0805`, `work-mcp1k-cb0805`,
`work-vfnuecc48-vf0805{a,b,c}` — the arms behind the Bee uploads) passed
`data`; every Claude-session validation arm (pr33-*, pr38-*, pr38b-*,
h1geo-*) passed `sim`. `reality` gates the SBND **pos_offset transverse
cathode calibration** (clus.jsonnet:93-102, gated in 6587ed51): corrected
points shift by `pos_offset_a0 = (0,-0.11,+0.67) cm` (TPC0) /
`a1 = (0,+0.11,-0.67) cm` (TPC1) in the PR job's switch_scope. Note 0.67 cm
in z ≈ 2.2 W-wire pitches: the same charge cloud lands on different wires.

Evidence (all on-disk, re-checkable via the Repro block):

- The `reality` marker predicts the good/bad outcome 12/12 (234638) and 7/7
  (55715) across three binary builds spanning 14 h; binary vintage predicts
  nothing.
- Good−bad point/vertex deltas equal the pos_offset vectors sign-for-sign
  per TPC (e.g. 55715 "After improve vertex" (-480.017,-382.064,4821.32) mm
  vs (-479.897,-381.022,4814.32) mm — exactly a0).
- Both lineages read the *same* pctree file (md5 verified), itself produced
  by the `reality=data` cb0805 QL round — so the sim-PR arms were
  **mixed-lineage** (data-derived pctree, sim PR correction).
- **Decisive**: rerunning all 7 events with `reality=data` at HEAD
  (work-pr38c-data7{,mc}) reproduces the cb0805 reference **byte-identically**
  on nusel TSV + pctree + every mabc-pr.zip member except mc.json, whose
  diffs are exactly the intended pr/38 changes (round-2 recoveries confirmed
  in the production lineage: 234638 +10028 under 10059; 447477 +11063;
  489330 +4044 under 4019; 52657 +23006; 55715 +15006/+15037 under 15005).
  Sole exception: 219295's TSV differs in ONE derived label (stmfit
  eval→contained) because cb0805's wct log line was torn mid-write (known
  log-tearing artifact) and nusel_extract.py could not match
  "fully contained"; the physics artifacts are byte-identical.

## Mechanism of the visible damage (sim arms, forensics)

**234638 — the "red track connecting two tracks from two sides".** In the
sim lineage the tiny cluster 4 (1.9 cm) survives scope → 2nd in-window main
+ 2nd beam flash group → 44/80 kept at Steiner → extra TGM → different
candidate handling; on the shifted main cluster the vertex candidate moves
2.3 cm ((102.82,-6.68,363.97) → (103.47,-5.01,365.97) — a different
candidate wins; the shift is nonlinear amplification, not the 0.68-cm offset
itself). The initial PR then builds segment 35 (calib id 10035): a 50-point,
38.2-cm "S_traj" path from the pi+ tail (97.3,-21.3,389.6) to the shower
trunk (111.1,7.5,378.0) — the closing chord of the vertex "V". 28/50 of its
points carry essentially no charge (middle-third mean dQ ≈ 1000 vs ≈ 37000
for the real pi+): a trajectory drawn through empty space — the red line.
Downstream the whole main cluster is absorbed into e- showers 10067/10068
(pi+ 211 disappears; the pr/38-recovered proton blip rides inside 10067).

**55715 — "entire structure clustered as an EM shower".** One decision
flips. Both lineages seed a candidate shower at the 11-point pi+ vertex stub
(seg 15005) inside `shower_clustering_connecting_to_main_vertex`
(NeutrinoShowerClustering.cxx:246; the pdg-211 skip gate at :341 needs
median dQ/dx > 2.0×MIP and both arms sit at 1.66/1.67). The accept/reject
logic (:387-449 good-track veto + topology cuts, FP-knife-edge on the
shifted fits) rejects the candidate in the data lineage and accepts it in
sim; :469 `set_pdg(11)` then coerces the pi+ (139.57 MeV, 38.5 MeV KE) to
e- (0.511, 15.4 MeV) — the flip measured in print_segs_info at "After
shower clustering with NV" — and `complete_structure_with_start_segment`
(:349, prototype-parity unbounded connectivity BFS) has swept proton 15006,
muon 15037 and shower 15007 into shower 15005. The owner's guess ("later EM
shower PR code") named the right stage; the trigger is the input shift, not
a code regression — the same binary byte-reproduces the good result under
`reality=data`.

## Procedural rule + guardrail

PR arms MUST pass the same `reality` as the QL round that produced the
input pctree. `run_pr_chain_batch.sh` now writes a `.lineage_reality`
marker into its out_root and refuses (overridable warning) when the ql_root
carries a marker with a different value; the cb0805/vf0805 rounds predate
the marker, so the runner also greps the ql_root's own `.batch_*` markers
as a fallback.

## Owner decisions (report only)

- `reality=data` on these MC samples contradicts the clus.jsonnet:96
  comment (pos_offset is a data-measured calibration; "applying it to MC
  would inject a spurious shift") but is the continuity lineage of every
  reference, Bee upload and hand-scan. Whether MC rounds should ever move
  to `sim` is an operating-point decision — not changed here.
- The 5017 display should serve the data-lineage arms
  (work-pr38c-data7{,mc}) for like-for-like comparison with the Bee sets.
- Round 2's still-open items (kine_reco_Enu residual; vertex-mode pinning)
  are unchanged, except the "DL not bit-stable across sessions" premise is
  withdrawn — no session-instability evidence survives this round.

---

# Round 3 (2026-08-07) — orphan parentage: flat roots → graph-faithful attachment

Shared Repro block, gate arms and flip record: [doc pr/44](44_vertex-muon-misid-as-shower.md)
(the two knobs shipped together).  This section carries the Part A mechanism.

## Symptom

SBND 18255 evt 142421 (owner report, Bee set 73cd68ed idx 0): mu- 207 MeV
(seg 7018), pi+ 196 MeV (7011) and proton 159 MeV (7012) all draw as
top-level nodes at the neutrino vertex, though the graph chains 7012 off
7011 (shared v7014) and hangs 7011/7018 off interior vertices of shower
7023's arm (v7013/v7020).  Owner principle: the PF tree must mirror the
image's segment-graph logic, strictly.

## Root cause

Three earlier fixes compose:

1. pr/40 r6 F12 (`shower_absorb_track_guard`, SBND ON) correctly keeps the
   three tracks out of shower 7023 — but the flood-fill still claims the
   junction vertices into the shower's view (PRShower.cxx add_vertex before
   the guard test).
2. `pf_shower_vertex_barrier` (pr/34 F2, SBND ON) pre-seeds those
   shower-view vertices into `visited_vtxs`; the pop-time check means the
   BFS never expands there, so the three segments are never claimed.
3. This doc's Round-1 orphan safety net then emits each unreached segment
   as a **flat root with no parent and no children** (prototype
   `mc_mother=0` parity) — even the 7011→7012 link is dropped.

The state is UNREACHABLE in the prototype: with no F12, such tracks are
absorbed into the shower (which was the pr/40 complaint).  F12 created a
new topology class the flat net was never designed for.  Meanwhile the
stage-1 fixed point had already computed the correct parent for every
blocked vertex (`vtx_incoming_seg` / `vtx_to_parent_shower`) — and threw
it away.

## Fix

`pf_orphan_track_parentage` (bee_pf block, C++ default **false**; inert
unless `pf_shower_vertex_barrier` is also on).  A new anchoring pass runs
after the shower-attachment loop, before assembly
(`MultiAlgBlobClustering.cxx`, gated), over the same orphan selection as
the flat net, in (display-id, graph-index) order:

- endpoint vertex carries `vtx_incoming_seg` → child of that claimed TRACK
  segment (inserted into `seg_parent`/`seg_children`/`seg_endpoints`, so
  `build_seg_node`'s recursion renders it);
- else endpoint in `vtx_to_parent_shower` → child of that SHOWER's
  displayed leaf via a new `shower_child_segs` map, rendered inside
  `make_shower_leaf` (the single construction site for all three leaf
  shapes: direct leaf, pseudo-gamma wrapper, pi0 grouping) with the same
  KeepMC `keep_node` convention as track children;
- an anchored orphan is inserted into `used_segs` (bars the flat net from
  re-emitting it) and extends `vtx_incoming_seg` at its far vertex, so
  orphan-of-orphan chains (pi+ → proton) anchor in a later round of the
  fixed point;
- orphans with no anchor at all fall to the untouched flat net exactly as
  before.

Determinism: all new containers are Index-comparator keyed; the orphan
sort tie-breaks equal display ids by `get_graph_index()` (PRGraph.h:315
caveat).  One shared-text touch: `build_seg_node` is now forward-declared
as a `std::function` next to `append_showers` and assigned at its original
definition site (declaration mechanics only; first call happens at
assembly, long after assignment).

Anchored orphans switch endpoint convention from fits-front/back+dirsign
to vertex fit points (`seg_endpoints`/`get_vtx_pt`) — intended: they are
now claimed segments and render like every other claimed segment.

Rule-order note: tracks resolve TRACK-anchor before SHOWER-anchor, the
opposite of `pf_shower_parent_precedence`'s shower-first order *for
showers* — deliberate asymmetry: a track continuing a track is the
stronger topological statement, and the precedence knob's motivation
(prototype `map_vertex_in_shower`-first reads) is specific to shower
parent resolution.

## Verification

Single-event G2a (SBND 18255-142421, knob on, B off): exact strict tree —
7011 under shower-leaf 7023 with 7012 chained beneath it, 7018 under
shower-leaf 7023; three ANCHOR probe lines; no flat orphan roots; no
duplicate emission (root loop skips seg_parent non-null entries;
shower-anchored orphans never enter seg_parent; the flat net re-tests
used_segs).  Gate results: doc pr/44.

## Residual

The Round-1/2 residual stands: `fill_kine_tree` (NeutrinoKinematics) has
its own BFS + barrier and got no orphan-parentage treatment (pr/43's F4
attempt at the kine side was rolled back with pr/43); display and energy
accounting can still disagree for barrier-orphaned tracks.

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

- kine_reco_Enu residual (round 1 §Residual) — unchanged; the
  `fill_sets(..., exclude_start_vertex)` parameter added here is the intended
  vehicle for that follow-up.
- Whether SBND production PR should pin the geometric vertex for
  reproducibility (or accept per-round DL variance) — owner decision.

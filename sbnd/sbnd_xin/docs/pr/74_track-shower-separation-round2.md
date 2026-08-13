# doc pr/74 — track/shower separation: the fixes (round 2)

Round 1 (`74_track-shower-separation-round1.md`) diagnosed the four owner
cases. This round implements the fixes, validates on the full 117-event
manifest (48 nueCC + 19 NC-π⁰ + 50 PR data), and flips all four knobs to
SBND production. Toolkit `96054e1e` (knobs + probes) + `2638faa8` (K4
iteration) + `064824c1` (flip); wcp `bf0016d` + `19e8ce9` (runner env).

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
M50=$(awk 'NR>1{print $2}' docs/pr/mcp1k-50-cb0805.index.txt)

# --- (0) Phase A attribution rerun (sentinels are env-gated, log-only)
SBND_SHOWER_TOPO_DQDX_GUARD=0 WCT_SHOWER_TOPO_DEBUG=1 WCT_PID_WRITE_DEBUG=2 \
  PR_JOBS=1 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74r2-attrD48 data 90055
# stderr record: grep "TOPO_FLAG\|PID_WRITE_DEBUG" work-pr74r2-attrD48/pr_evt90055/stdout.log

# --- (1) per-knob smokes (Phase C)
SBND_SHOWER_IN_CASCADE_GUARD=1  SBND_WCT_LOGLEVEL=debug PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805  work-pr74r2-k1s50 data 53361
SBND_MICHEL_STEM_MICHEL_CHECK=1 SBND_WCT_LOGLEVEL=debug PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74r2-k2s48 data 90055
SBND_SHOWER_STEM_BACKFILL=1     SBND_WCT_LOGLEVEL=debug PR_JOBS=2 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74r2-k4s48 data 90055 469665
SBND_SHOWER_CONN3_UNREACHABLE=1 SBND_WCT_LOGLEVEL=debug PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805  work-pr74r2-k5s19 data 142421

# --- (2) off-gate (Phase D): 0/117 byte-identical
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74r2-off48 data
PR_JOBS=32 ./run_pr_chain_batch.sh work-ncpi0-cb0805   work-pr74r2-off19 data
PR_JOBS=32 ./run_pr_chain_batch.sh work-mcp1k-cb0805   work-pr74r2-off50 data $M50
python3 scripts/analysis/pr49/on_compare.py work-pr51r7-on48 work-pr74r2-off48   # x3
# + hash_archive.py loop over pctree-pr-evt*.tar.gz (on_compare covers mabc only)

# --- (3) on-arms + census (Phase E)
export SBND_SHOWER_IN_CASCADE_GUARD=1 SBND_MICHEL_STEM_MICHEL_CHECK=1 \
       SBND_SHOWER_STEM_BACKFILL=1 SBND_SHOWER_CONN3_UNREACHABLE=1
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74r2-on48 data     # + 19, 50
python3 scripts/analysis/pr49/on_compare.py  work-pr74r2-off48 work-pr74r2-on48  # x3
python3 scripts/analysis/pr51/nuvtx_census.py work-pr74r2-off48 work-pr74r2-on48 # x3

# --- (4) compiled-config proof (M6; production pipeline_names TLA REQUIRED)
PL=$(grep -m1 '^PIPELINE=' run_pr_chain_batch.sh | sed 's/PIPELINE="//;s/"//')
wcsonnet ... --tla-code "pipeline_names=['switch_scope',...]" \
  [--tla-code shower_in_cascade_guard=true ...] wct-pr-perevt.jsonnet | grep <key>
```

## What shipped (all C++ default OFF, key-suppressed — off = byte-identical)

| knob | case | mechanism | tunables (C++ default) |
|---|---|---|---|
| `shower_in_cascade_guard` (P1) | 53361 | refuse `examine_direction`'s `flag_shower_in` e⁻ relabel (the `\|pdg\|==13\|\|pdg==0` branch, `NeutrinoVertexFinder.cxx`) for a segment BOTH long AND MIP-like; predicate `segment_shower_in_cascade_vetoed` | `shower_in_max_len` 40 cm, `shower_in_mip_hi` 1.3 |
| `michel_stem_michel_check` (P2) | 90055 label | F14 Michel rescue additionally requires the graph beyond the stem's far vertex to be Michel-sized; helper `segment_far_subtree_track_length` | `michel_stem_max_far_len` 40 cm |
| `shower_stem_backfill` (K4) | 90055 trunk + 469665 stem | post-pass in `shower_clustering_with_nv` (after `in_other_clusters`, before the 2nd kine pass): walk from each substantial EM shower's attach vertex back toward the main vertex, absorbing short, not-charge-hot chain segments; a Bragg proton stops the walk | `stem_backfill_max_len` 30 cm, `stem_backfill_mip_hi` 3.5, `stem_backfill_min_shower_len` 40 cm |
| `shower_conn3_unreachable` (K5 = pr/65 rung 2) | 142421 | extend `shower_clustering_in_other_clusters`' leftover-cluster branch (the prototype's `connection_type=3` pseudo-gamma path) to graph-unreachable, unclaimed main-cluster segments; reuses `unreachable_segments` from pr/65 rung 1 | `conn3_unreachable_min_len` 10 cm |

No knob for K3 — see the retraction below. P4 (low-score defer) and P5
(demote_len review) remain unimplemented per round 1's recommendation.

## Round-1 §3.1 RETRACTED: there is no "second demotion" on 90055

The new attribution probes (`TOPO_FLAG` at every `kShowerTopology` set/unset
site, `gidx` added to the `shower_topo dbg` lines, a POST-GUARD truth line
after the F3 veto, `TOPO_REEXAM` in `improve_vertex`'s topology re-exam) give
a complete, single-stream (stderr) transition record. On the
`SBND_SHOWER_TOPO_DQDX_GUARD=0` rerun of 90055:

- The `L 14.4cm ... branch 1 ... final_shower true` stage-3 evaluation that
  round 1 attributed to the trunk belongs to **seg 11056** (gidx 56) — a
  shower-body segment that ends `e-` (score 0.078) in production anyway.
- The trunk **11045** (gidx 45) measures L 18.3 cm geometric / 14.38 cm fit
  and **fails the topology test on pure geometry** (branch 0, max_spread
  0.61 cm < 0.7 cut) — before the dQ/dx guard is ever consulted. Its flag
  was never set, so nothing ever demoted it.
- The improve_vertex re-exam block (the recon suspect) never executed for
  any segment in this event (`TOPO_REEXAM` silent).
- `work-pr74-nodqdx48` hashes **identical** to production
  (`94746edd58e6...` both) — the guard ablation has zero footprint on this
  event, which also retracts round 1's claim that the guard "kills the
  stage-3 topology flag" of the trunk with observable effect.

Round 1's conflation came from the `shower_topo dbg` line printing
`seg->id()` — which is `-1` for every segment. The probes now print `gidx`.
The trunk's exclusion is purely the structural BFS gap (prototype-shared),
which is exactly what K4 closes.

## Per-knob smoke evidence (Phase C)

- **K1 / 53361**: `pr74 shower_in_cascade_guard: veto e- relabel gidx=4
  L=113.9cm pdg=13`. PF root goes from a single `e- 405 MeV` to
  `e- 28 MeV (27001) → mu- 280 MeV (27004)` — the owner's muon is a muon.
  The short head segment keeps its legitimate trajectory-test electron label
  (it is what arms `flag_shower_in`).
- **K2 / 90055**: `pr74 michel_stem_michel_check: veto mu- rescue gidx=45
  far_len 41.1cm > 40cm` (early-exit value; the true far subtree is the
  155 cm shower spine). Trunk reads `proton 147 MeV` instead of `mu- 60 MeV`
  — same as the round-1 F14-off ablation, now surgical.
- **K4 / 90055**: `pr74 stem_backfill: shower(start gidx=44 conn=1) chain
  gidx=45 len 14.4cm dqdx 3.21x -> absorb`. The trunk's true median is
  3.21× MIP (round 1's "2.75×" was part of the same identity confusion).
  PF: `e- 2020 MeV (11044)` directly at the vertex, no separate trunk node;
  paint: 11045's 345 points join the shower (11044 7831→8176 pts, bbox
  reaches the vertex).
- **K4 / 469665**: chain `gidx=1 2.2cm 1.72x -> absorb`,
  `gidx=3 27.6cm 1.11x -> absorb`, `gidx=4 3.8cm 3.71x -> stop` — the two
  mu- stem segments join the 322 MeV shower and the walk stops exactly at
  the vertex Bragg proton (owner: "after the initial proton, the entire
  thing should be an EM shower"). NOTE the margins around the 3.5 default:
  absorb at 3.21, stop at 3.71.
- **K5 / 142421**: `pr74 conn3_unreachable: promote gidx=13 len 41.9cm
  conn=3 anchor_dis 0.0cm`; the pr/65 audit line now reads **0 unclaimed
  segment(s)**; the owner's point (96.1,−74.5,232.3) paints SHOWER and the
  segment enters PF as an `e-` node. Both halves of the owner's complaint
  (painted track + missing from PF) closed by one mechanism.
- The fragmentation half of 469665 (3 root gammas from 5 clusters) is out
  of scope per the owner's round-2 decision (stem fix only).

## Gates

- **Off-gate (Phase D) PASS**: `work-pr74r2-off{48,19,50}` vs
  `work-pr51r7-on{48,19,50}` — ARCHIVE-LEVEL 0/48, 0/19, 0/50 (mabc member
  hashes via `on_compare.py`); pctree member hashes 0/117; `nusel-events` /
  `nusel-table` 0/117. One-event pre-check: bare 90055 on the new binary
  hashes `94746edd58e6...` == production.
- **Doctests**: `./build/clus/wcdoctest-clus` **2004/2004 pass** (new:
  `doctest_pr74_track_shower.cxx` pins the P1 veto and the P2 far-subtree
  helper on synthetic graphs; 12 knob-default lines added).
- **Compiled-config proof (M6)**: with the production `pipeline_names` TLA,
  all 11 new keys ABSENT from the compiled JSON with knobs off, present
  with the TLAs on (checked for both the P1/P2 and K4/K5 groups).
- **On-census (Phase E)**, `work-pr74r2-onb{48,19,50}` vs
  `work-pr74r2-off{48,19,50}`:
  - archive movers **7/117**, every one attributed: the 4 owner targets +
    138009 (K4 absorbs a 0.7 cm MIP crumb, cosmetic) + 350186 / 506746 (K2
    vetoes two more Michel rescues, far subtrees 45.6 / 49.2 cm > 40 cap).
  - **nusel event labels 0/117 flips** (`nusel-events`/`nusel-table`
    identical).
  - **PF-orphan sweep over all 117 events**: the ONLY orphan-set change is
    142421 losing seg 7013 (the K5 fix). Zero stranded orphans anywhere —
    this metric drove the K4 iteration (first on-census stranded orphans in
    3 events: 268067 a 595 MeV proton branch, 285567 442 MeV of protons,
    56982 a fragment; the junction guard + `stem_backfill_mip_lo` closed
    all three, re-verified byte-identical-to-off on those events).
  - `nuvtx_census`: ν-vertex movers **0/117** (no move > 0.01 cm at all);
    |ΔEnu| > 100 MeV on 4 events, all knob-acted and physical: 142421
    **+795 MeV** (the formerly PF-invisible EM charge now counted), 53361
    −376 MeV (muon hypothesis replaces the 405 MeV electron), 469665
    −199 MeV, 506746 +169 MeV.
  - BDT scores (`pr_scores_table` + `pr20_scores_diff`): cells differ only
    on the 6 acted events; 53361's nue_score drops to the −15 sentinel (no
    longer evaluated as a νe candidate — the desired direction for a
    hand-scanned muon).
  - 2 events (116962, 52613) have no selected candidate in either arm
    (pre-existing; archives identical).
- **Flip gates (Phase G)** — flip commit `064824c1` sets the four bools
  `true` in `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` (the only
  file the flip touches; scalars stay at the C++ defaults):
  - **FLIP-PROOF**: bare `work-pr74r2-flip{48,19,50}` == the validated
    on-arms `work-pr74r2-onb{48,19,50}` — 0/117 archives + nusel.
  - **ESCAPE-PROOF**: `SBND_SHOWER_IN_CASCADE_GUARD=0
    SBND_MICHEL_STEM_MICHEL_CHECK=0 SBND_SHOWER_STEM_BACKFILL=0
    SBND_SHOWER_CONN3_UNREACHABLE=0` → `work-pr74r2-esc{48,19,50}` == the
    off-arms `work-pr74r2-offb{48,19,50}` — 0/117 archives + nusel.

## Bee links

All 7 on-census movers, identical event ordering, annotated index
`docs/pr/pr74r2-bee.index.txt`:

- **before** (production):
  <https://www.phy.bnl.gov/twister/bee/set/5d9f306c-1988-45ec-bfa2-ff93b42f17b9/event/list/>
- **after** (all four knobs on):
  <https://www.phy.bnl.gov/twister/bee/set/a8264f65-b41d-488e-bdbb-08bd71fc195f/event/list/>

Order: 90055, 138009, 350186, 469665, 142421, 506746, 53361. The four owner
cases are idx 0 (90055), 3 (469665), 4 (142421), 6 (53361).

## Status / open items

- **All four owner cases fixed and SBND PRODUCTION ON** (owner-delegated
  flip conditional on validation; toolkit `96054e1e` + `2638faa8` +
  `064824c1`, wcp `bf0016d` + `19e8ce9`). Bare production now reproduces
  the "after" Bee set exactly.
- Round-1 §3.1's "second demotion" is **retracted** (identity confusion in
  an id-less log line; see the attribution section). The topo dbg lines now
  print `gidx`, so this class of confusion cannot recur.
- 469665's 5-cluster fragmentation (3 root gammas) remains out of scope by
  the owner's round-2 decision — a clustering-family follow-up.
- K4's `stem_backfill_mip_hi=3.5` sits between the measured absorb (3.21×,
  90055 trunk) and the measured Bragg-proton stop (3.71×, 469665). Margins
  are real but not wide; the parameter is env-tunable
  (`SBND_STEM_BACKFILL_MIP_HI`) if a future event lands in the gap.
- The stranded-PF-orphan class K4's iteration closed is ultimately a
  display-layer limitation (orphans anchor only via `vtx_incoming_seg` /
  shower ATTACH vertices; interior shower vertices don't re-anchor). A
  future `pf_orphan_anchor_shower_members` extension in
  `fill_bee_pf_tree` would let those absorptions proceed instead of being
  blocked — noted, not needed for this round's bar.
- P4 (low-score defer) and P5 (demote_len review) remain open as round-1
  proposals.

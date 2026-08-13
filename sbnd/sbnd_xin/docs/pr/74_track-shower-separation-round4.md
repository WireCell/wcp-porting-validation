# doc pr/74 — track/shower separation, round 4: the muon+Michel reconstructed as one electron

Round 3 (`74_track-shower-separation-round3.md`, toolkit `40651cb2`, wcp
`376413b`+`7daefa3`) closed the owner's findings 1 and 2 and left finding 3
diagnosed only:

> **18255-506746** — the `107 MeV electron` is really a **muon + Michel**.

Round 3 recorded a proposal (`shower_traj_mip_chain_guard`) and an explicit
"owner call whether it becomes round 4". The owner called it. This round
implements the fix as **K6 `shower_traj_michel_stem`** and, on a clean gate
sweep, flips it ON for SBND production.

**The round-3 proposal is NOT what shipped.** Its home — the separation stage
`segment_is_shower_trajectory` — turned out to be the wrong one, and its
central discriminant (an all-MIP chain) turned out to be unmeasurable there.
§ 3 says why. What shipped is narrower, topological, and evidence-driven.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
M50=$(awk 'NR>1{print $2}' docs/pr/mcp1k-50-cb0805.index.txt)

# --- Phase A: the go/no-go probe (log only, byte-identical, still in the tree)
WCT_MICHEL_STEM_PROBE=1 PR_JOBS=32 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr74r4-probe19 data
grep -h MICHEL_STEM_PROBE work-pr74r4-probe19/pr_evt*/stdout.log

# --- the fix, on the owner's event
SBND_SHOWER_TRAJ_MICHEL_STEM=1 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr74r4-smoke2 data 506746
grep -h "pr74r4 shower_traj_michel_stem" work-pr74r4-smoke2/pr_evt506746/*.log

# --- the PF-tree attachment trace (why the Michel hangs where it does)
WCT_BEE_PF_PRINT=1 SBND_SHOWER_TRAJ_MICHEL_STEM=1 PR_JOBS=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 work-pr74r4-pfprint data 506746
grep -h "PROPAGATE-OVER-TRACK\|SEGMENT-attached" work-pr74r4-pfprint/pr_evt506746/stdout.log

# --- off-gate (0/117) and the round-4 arms
export SBND_SHOWER_TRAJ_MICHEL_STEM=0
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74r4-off48 data   # + 19, 50
export SBND_SHOWER_TRAJ_MICHEL_STEM=1
PR_JOBS=32 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr74r4-on48  data   # + 19, 50
python3 scripts/analysis/pr49/on_compare.py   work-pr74r3-on48 work-pr74r4-off48
python3 scripts/analysis/pr74/pr74_pf_roots.py work-pr74r4-off48 work-pr74r4-on48
python3 scripts/analysis/pr51/nuvtx_census.py work-pr74r4-off48 work-pr74r4-on48
```

Two reproduction notes, both cost time this round:

- **The pr/75 runner collision.** A concurrent session added a
  `vertex_scoreboard` TLA to the shared `run_pr_chain_batch.sh`, defaulted ON
  whenever `PR_EXTRA_STAGES` names `pr_display`. The toolkit side of that knob
  is not on `apply-pointcloud`, so every `pr_display` run here dies with
  `RUNTIME ERROR: function has no parameter vertex_scoreboard`. It cannot be
  suppressed by environment (`: "${SBND_VERTEX_SCOREBOARD:=true}"` fires on
  empty as well as unset). Bare runs are unaffected, so the whole campaign
  used the shared runner; only the two calib-dump runs used a private copy
  with that one line neutered (deleted afterwards). **The shared runner's
  own content was not edited** -- but note the reverse also happened: this
  round's `SBND_SHOWER_TRAJ_MICHEL_STEM` env block, added to the shared
  `run_pr_chain_batch.sh`, was swept into the pr/75 session's commit
  `dac67b2` because both sessions share the working tree. It is in the repo
  and correct; only the authorship is not what it looks like.
- **The waf link deadlock.** Adding a function to `PRSegmentFunctions.cxx` and
  calling it from a doctest fails to link forever: `wcdoctest-clus` resolves
  against the *installed* `local/lib/libWireCellClus.so`, which `wcbuild`
  only refreshes after the build step that the test link aborts. Break it once
  with `cp build/clus/libWireCellClus.so local/lib/`, then `wcbuild` normally.
  (Followed here by a clean full `wcbuild` + freshness proof.)

---

## 1. What the event is

Main cluster 21, main vertex **21038** (degree 4) at (53.8, −11.8, 40.6) cm.
The "107 MeV electron" is not one segment but a linear three-segment chain,
all three carrying `kShowerTrajectory`, forced pdg 11 and the sentinel
`particle_score 100`:

| seg | L (cm) | median dQ/dx ÷ 48000 | vertices | role |
|---|---|---|---|---|
| 21048 | 20.39 | **1.57×** | 21037 ← **21038 (ν)** | shower start segment |
| 21056 | 7.99 | 1.14× | 21034 ← 21037 | shower member |
| 21051 | 9.49 | 0.93× | 21034 → 21040 (degree 1) | shower member |

Three measurements settle it, and all three say the owner is right:

1. **21048's charge profile is a stopping muon.** In fifths from the v21037
   end: 1.61, 2.05, 1.36, 1.44, 1.52 × MIP — elevated at the far end, falling
   toward the neutrino vertex. A mean of 1.57× over 20.4 cm is what a muon
   with ~20 cm of residual range gives (KE ≈ 47 MeV). **v21037 is where it
   stops.**
2. **There is a 93° kink at v21037** (15 cm tangents; 103° by vertex chords).
   A Michel electron is emitted at a random angle off the stopping point. An
   electron trunk continuing into its own cascade goes near-forward.
3. **Everything past v21037 is terminal**: 17.5 cm total, ending at a
   degree-1 vertex, and v21037 itself has degree exactly 2 — one stop, one
   Michel, no branching. A real cascade branches.

Round 3 quoted these dQ/dx ratios against a different normalisation
(1.38/1.00/0.82). The ratios above are `segment_median_dQ_dx / 48000`, the
same quantity the code compares — use these.

## 2. Why the reconstruction cannot see it

`segment_is_shower_trajectory` (`PRSegmentFunctions.cxx:1761`) is a
*wiggliness* test with no dQ/dx cross-check strong enough to save a scattering
muon; once it sets `kShowerTrajectory`,
`segment_determine_shower_direction_trajectory` forces pdg 11 with the
sentinel score, and shower clustering absorbs muon, Michel and blob into one
"107 MeV electron" starting at the neutrino vertex.

**The ordinary track PID cannot arbitrate — measured, not assumed.**
`separate_track_shower` never runs it on a shower-flagged segment
(`NeutrinoTrackShowerSep.cxx:298-317`). The Phase A probe ran it anyway, with
the store forced (`track_pid_persist_dqdx` + `_4mom`, electron guard off — the
plain call would silently keep the stale pinfo, because `free_end_dir` is
false for any segment whose two endpoints both have degree > 1,
`PRSegmentFunctions.cxx:2662`). On 21048 it **abstains**: `stored=0`,
`pdg_code == 0`, score and KE unchanged at the trajectory sentinel
100.000 / 63.0 MeV. Only 2 of the 20 probed candidates abstain; 18 store a
verdict, so this is a real property of 21048, not a broken probe.

**Consequence for the design: the predicate must be topological, not
template-based.** That killed gate 8 of the plan (accept on a confident track
verdict) before any wiring was done — which is what the Phase A probe existed
to find out.

It also disposes of the round-3 proposal's home. `segment_is_shower_trajectory`
has no main vertex, no graph and no downstream context — it cannot see a kink,
a far-vertex degree, or a terminal subtree — and it runs on every cluster of
every event. The evidence that identifies this class only exists *after*
`determine_main_vertex`.

## 3. The fix: K6 `shower_traj_michel_stem`

`PatternAlgorithms::override_shower_traj_michel_stem`
(`NeutrinoPatternBase.cxx`), called from the tail of `examine_direction`
(`NeutrinoVertexFinder.cxx`) — the same "last word before
`shower_clustering_with_nv`" call site F8 and F14 use, but **only when
`flag_final`**. That is true for exactly one call, `TaggerCheckNeutrino.cxx:1418`;
the two internal callers (`NVF:3374`, `:4471`) pass false. So the pass runs
once, on the neutrino main cluster, on the final graph, immediately before
shower clustering consumes the shower flags.

A **separate function**, not a branch inside `override_michel_stem_muon`: F14
is SBND production ON and its body stays byte-for-byte untouched (§ 2 Code,
fork by duplication).

For each segment incident on the main vertex, all of:

| # | test | knob | default | 21048 |
|---|---|---|---|---|
| 1 | `kShowerTrajectory` and **not** `kShowerTopology`, pdg 11 | — | — | ✓ |
| 2 | `min_len` ≤ L ≤ `max_len` | `michel_stem_traj_min/max_len` | 15 / 45 cm | 20.39 |
| 3 | median dQ/dx ÷ MIP median ≥ `mip_lo` | `michel_stem_traj_mip_lo` | 1.3 | 1.57 |
| 4 | far-vertex degree **exactly 2** | — | — | 2 |
| 5 | track length beyond that vertex < `max_far_len` | `michel_stem_traj_max_far_len` | 40 cm | 17.5 |
| 6 | the single sibling there is shower-like | — | — | ✓ |
| 7 | kink between them ≥ `min_kink_deg` | `michel_stem_traj_min_kink_deg` | 40° | 93.1 |

On accept: pdg → 13 (a Michel parent is a muon — the same reasoning F14
encodes when it relabels a Bragg-fitted proton stem), 4-momentum recomputed
with `m_mip_dqdx` as F14 does, `particle_score(100.0)`, both shower flags
cleared, and the new `SegmentFlags::kMuonStemGuard` set.

`michel_stem_traj_max_far_len` is deliberately **its own member**, not P2's
`michel_stem_max_far_len`: same 40 cm today, but P2 uses it as a *veto*
ceiling and this uses it as an *accept* ceiling. Sharing it would move one
pass silently when the owner tunes the other.

### Two flag-keyed guards, neither needing a knob

`kMuonStemGuard` is written **only** by this default-OFF pass, so anything
reading it is byte-identical when K6 is off, by construction. Nothing
serialises the raw segment-flags word — only named bits are ever tested
(checked across `PrDisplayDump`, `SbndPrMagnifyTrackingVisitor`, MABC and
`PatternDebugIO`, whose `"flags"` are *cluster* flags) — so a new bit is inert
in every output. Precedent: `kTwoEndBreakArm` (doc pr/48).

**(a) `stem_backfill` must not absorb the muon straight back.** K4 walks stems
toward the main vertex with no pdg test, so it could undo the fix in one step.
On this event it cannot (`stem_backfill_min_shower_len` 40 cm > the 17.5 cm
Michel shower, **verified from the log: K4 emits no line for that shower**),
but the guard is unconditional in `NeutrinoShowerClustering.cxx`.

**(b) The Michel must hang off the muon in the particle-flow tree.** This one
was found by inspection, not by a gate, and it is the same defect class the
owner has now filed three times. `fill_bee_pf_tree` builds `vtx_incoming_seg`
by walking track-only segments out of the neutrino vertex; separately, every
root-anchored shower's *vertex set* is propagated into `root_reachable_vtxs`,
and pr/34 F3's parent-shower precedence (`MultiAlgBlobClustering.cxx:1419`,
SBND production ON) then attaches anything anchored there to the **shower**
rather than to the track. Once K6 makes 21048 a track, the BFS reaches v21037
— but the neighbouring 102 MeV shower's vertex set contains v21037 too and
claimed it first:

```
[fill_bee_pf_tree] PROPAGATE-OVER-TRACK  vtx_gidx=37
                   claimed_by_shower_ke=102.128  over_incoming_seg_gidx=48
```

so the 64 MeV Michel rendered as a daughter of that shower instead of the muon
that produced it. The fix declines the claim when the vertex was reached
through a `kMuonStemGuard` segment. Deliberately narrow: the general rule
"track BFS beats shower set" would restructure the tree at every vertex where
the two disagree, and that is a separate change with its own census.

After it, the trace reads `SEGMENT-attached shower conn_type=1 parent_seg=48
ke=64.4429 MeV` — the Michel on the muon.

## 4. Result on 18255-506746

```
BEFORE (round-3 production)          AFTER (K6 on)
21048  e-  107 MeV   (root)          21048  mu-  62 MeV   (root, ν vertex -> 45.8,-26.7,32.5)
                                       21056  e-   64 MeV   (the Michel, at the stopping point)
                                       8  gamma 17 MeV
```

`21048` is `pid 13`, `flag_shower False`, `shower_id −1` — out of every
shower. The Michel is a new 2-segment shower (`21056`+`21051`, 17.48 cm,
`start_vtx 21037`, `conn 1`).

**The π⁰ reconstruction improves sharply, and that is checkable truth on an
NC-π⁰ event:**

| | OFF | ON |
|---|---|---|
| `kine_pio_energy_1` | 146.11 | 102.13 |
| `kine_pio_angle` | 131.07° | **31.18°** |
| `kine_pio_mass` | **734.04 MeV** | **150.8 MeV** |

The fake 107 MeV electron at the vertex was distorting shower clustering: the
21050 shower had absorbed distant unrelated clusters (9 segments, out to
(105,64,121) cm), which is what gave the π⁰ hypothesis a 131° opening angle
and a nonsense 734 MeV mass. With the fake removed the shower is 6 segments,
and the π⁰ mass lands at **150.8 MeV against a true 135**.

**`nusel-evt506746.tsv` is byte-identical** between the two arms — no
selection flip, in either direction.

### Two numbers the owner will ask about

- **"64 MeV Michel" is above the 52.8 MeV Michel endpoint.** That is the
  estimator, not new charge. For this shower `kine_charge` = 72.5,
  `kine_range` = 64.4, `kine_dQdx` = **38.7**; `kine_best` takes the range
  estimate, as it did for the combined 107 MeV object before. The dQ/dx
  estimator, 38.7 MeV, is squarely a Michel. The spread is this
  reconstruction's shower-energy scale on a 17 cm object, unchanged by K6.
- **113 MeV of EM moves onto a distant anchor.** Shrinking the 21050 shower
  releases clusters 76/77, which re-anchor as conn-2/3 pseudo-gammas ~103 cm
  from their parent. That *reads* alarming, so it was censused: in today's
  round-3 production **54 of 113 events already carry a pseudo-gamma node
  whose anchor gap exceeds 50 cm — 102 such nodes, median 101 cm, max
  329 cm.** A long conn-2/3 anchor is the normal behaviour of this rendering,
  not something K6 exposed. Worth its own round; not this one's bug.

### Post-fix the vertex carries two muon-typed segments

21050 (9.95 cm, 1.67× MIP, KE 47.4) off the same neutrino vertex is already
muon-typed at `examine_direction` time. Stated explicitly so it is not a
surprise in the scan.

## 5. Pre-census: how many events can this touch?

Phase A's probe was run over the **full 117-event manifest** with the same
`flag_final` gate the knob uses, printing every candidate discriminant. Twenty
segments were candidates; the predicate accepts exactly one.

| arm | evt | gidx | L | dQ/dx | far_deg | far_len | kink | verdict |
|---|---|---|---|---|---|---|---|---|
| 48 | 131357 | 30 | 13.09 | 1.07 | 3 | 99.7 | 129.0 | L, degree |
| 48 | 137238 | 58 | 14.47 | 1.04 | 1 | 0.0 | — | degree |
| 48 | 196649 | 40 | 6.62 | 1.26 | 3 | 332.9 | 166.0 | L, degree |
| 48 | 268784 | 3 | 4.02 | 1.50 | 1 | 0.0 | — | L, degree |
| 48 | 268784 | 54 | 8.15 | 1.14 | 3 | 134.5 | 81.3 | L, dQ/dx, degree |
| 48 | 388 | 31 | 23.45 | 1.10 | 1 | 0.0 | — | dQ/dx, degree |
| 48 | 46363 | 60 | 37.19 | 1.19 | 3 | 361.5 | 101.8 | dQ/dx, degree |
| 19 | 180801 | 14 | 19.92 | 0.87 | 4 | 106.6 | 148.8 | dQ/dx, degree |
| 19 | 21073 | 8 | 10.40 | 0.95 | 3 | 110.3 | 86.9 | L, dQ/dx, degree |
| 19 | 285567 | 55 | 3.19 | 1.21 | 2 | 84.8 | 88.8 | L, dQ/dx, far_len |
| 19 | 359980 | 72 | 24.60 | 0.96 | 2 | 16.4 | 176.1 | dQ/dx |
| 19 | 399860 | 56 | 14.60 | 1.08 | 2 | 2.7 | 46.6 | L, dQ/dx |
| 19 | 463565 | 27 | 9.45 | 0.30 | 3 | 135.5 | 120.2 | L, dQ/dx, degree |
| **19** | **506746** | **48** | **20.39** | **1.57** | **2** | **17.5** | **93.1** | **ACCEPT** |
| 19 | 71372 | 123 | 13.28 | 0.95 | 1 | 0.0 | — | L, dQ/dx, degree |
| 50 | 48367 | 49 | 4.18 | 0.92 | 3 | 23.1 | 89.2 | L, dQ/dx, degree |
| 50 | 55715 | 1 | 17.10 | 1.07 | 1 | 0.0 | — | dQ/dx, degree |
| 50 | 57485 | 3 | 5.57 | 1.49 | 2 | 0.7 | 88.9 | L |
| 50 | 58607 | 5 | 11.56 | 0.29 | 1 | 0.0 | — | L, dQ/dx, degree |
| 50 | 59085 | 7 | 6.26 | 2.00 | 3 | 25.2 | 124.8 | L, degree |

**Nothing on the nueCC48 manifest comes close** — every one of its
vertex-rooted candidates fails on length or degree or charge, and the two with
L ≥ 15 cm sit at 1.10 and 1.19× MIP.

**No threshold is a knife edge.** Among candidates that pass length and
degree, dQ/dx runs 0.96, 1.08 → then **1.57**: the 1.3 cut sits in the middle
of a wide empty gap, not next to a near miss. Same for length (20.39 accepted,
nearest lower candidate 14.60, which also fails dQ/dx) and far_len (17.5
accepted against a 40 cm ceiling).

## 6. Gates

All on the shipped round-4 binary; freshness proof before every A/B
(`local/lib/libWireCellClus.so` 07:25 > last source edit 07:24).

- **Compiled-config gate PASS** — with the knob off, `wcsonnet` output is
  **byte-identical** (md5 `bec1e355cd970f569c7baaba6676c2d3`) to the same
  command compiled against `git archive HEAD cfg`. With the knob on, the key
  appears. Key-suppression idiom verified both ways (M6).
- **Doctests** — `./build/clus/wcdoctest-clus` **208 cases / 2056 assertions,
  rc=0** (from 204 / 2037). Four new cases pin the `segment_pair_kink_deg`
  sign convention — straight = 0, right angle = 90, fold-back = 180,
  unmeasurable = −1 and never 0, which is the one mistake that would silently
  invert the Michel test.
- **Off-gate PASS** — `work-pr74r4-off{48,19,50}` vs `work-pr74r3-on{48,19,50}`
  (= the production point this round started from): ARCHIVE-LEVEL **0/48,
  0/19, 0/50**; pctree member hashes **0/48, 0/19, 0/50**;
  `nusel-events` / `nusel-table` **0/117**. The round-4 code is byte-identical
  with the knob off. (pctree compared on **column 1 only** — `hash_archive.py`
  prints `<hash> <nmembers> <path>` and the path differs between arms by
  construction; the pr/57 trap.)
- **On-census**, `work-pr74r4-off*` → `work-pr74r4-on*` — the total knob
  footprint on the standard manifest:

  | arm | archive movers | members that changed |
  |---|---|---|
  | nueCC 48 | **0/48** | — |
  | NC-π⁰ 19 | **1/19** | 506746: `0-mc.json`, `0-shower_track-global.json` |
  | PR data 50 | **0/50** | — |

  **1/117 events, and only the particle-flow tree and the paint layer inside
  it.** `vertices-global` is unchanged even on the mover. **nusel event
  labels 0/117 flips**, on every arm.
- **PF-root gate PASS** — dangling roots vs the `T_tagger` ν vertex,
  `work-pr74r4-off*` → `work-pr74r4-on*`: **0 gained** on all three arms, and
  the dangling-root list is unchanged (`changed : 0`) everywhere. This is the
  gate round 3 built precisely because it is the metric this class of change
  can break; K6 does not move it. (4 events unmeasurable — no `T_tagger`, no
  selected neutrino candidate; pre-existing.)
- **ν-vertex census PASS** — `nu-vtx > 10 cm`: **0/117**. No sub-threshold
  movers either (`0.01 < dvtx ≤ 10 cm`: 0). **No vertex anywhere moves.**
- **Energy** — `|ΔEnu| > 100 MeV`: **1/117**, the mover itself:
  506746 `kine_reco_Enu` 2265.3 → 2427.3 MeV (**+162.0**), of which
  +105.7 is `kine_reco_add_energy` (296.3 → 402.0) — the clusters released by
  the shrinking 21050 shower being counted, with the 22058/76129 pair going
  32.2 → 68.0 + 46.2 MeV. Reported, not tuned away: this is the honest
  consequence of removing a fake 107 MeV primary electron, and it arrives
  together with the π⁰ mass landing on 150.8 MeV (§ 4).

### 6.1 Flip gates

Byte-exact both ways, so the flip is exactly the knob and nothing else:

- **Flip gate PASS** — bare `work-pr74r4-flip{48,19,50}` (no environment
  override, i.e. what production now does) vs the knob-on arms
  `work-pr74r4-on{48,19,50}`: **0/48, 0/19, 0/50**, nusel 0/117. Includes
  506746, so the bare production run reproduces the fix.
- **Escape gate PASS** — `SBND_SHOWER_TRAJ_MICHEL_STEM=0`
  `work-pr74r4-esc{48,19,50}` vs the knob-off arms
  `work-pr74r4-off{48,19,50}`: **0/48, 0/19, 0/50**, nusel 0/117. The escape
  hatch returns the old production point exactly.

## 6.2 Bee

- **before** (production before the flip; byte-identical to round-3
  production): https://www.phy.bnl.gov/twister/bee/set/38a3f6b7-f96d-422e-9a9f-7d344c927ea7/event/list/
- **after** (round-4 production): https://www.phy.bnl.gov/twister/bee/set/b31f462b-e91b-49d2-9d50-f53be7cd71ad/event/list/

One event, because 506746 **is** the round's entire footprint. Annotated
index: `docs/pr/pr74r4-bee.index.txt`.

## 7. Status

- **K6 `shower_traj_michel_stem` — SBND PRODUCTION ON**, owner flip
  2026-08-13 (`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` only; the
  five scalar tunables stay `null` = the C++ defaults).
- Sentinel `WCT_MICHEL_STEM_PROBE` retained in the tree: it is the only way to
  see what the track PID would say about a shower-flagged segment, and it is
  what this round's design turned on.

## 8. Open items

- **Long conn-2/3 pseudo-gamma anchors** (§ 4): 54/113 production events carry
  one over 50 cm, median 101 cm. Pre-existing, unexamined, and the reason a
  released shower fragment can land ~100 cm from its rendered parent. Its own
  round.
- **`shower_traj_mip_chain_guard`** as originally proposed in round 3 § 3 is
  **withdrawn** — superseded by K6 and shown in § 2 to be unimplementable at
  the site it named.
- The pr/75 `vertex_scoreboard` runner/toolkit skew (Repro block) will keep
  breaking `pr_display` on this branch until that toolkit change lands.

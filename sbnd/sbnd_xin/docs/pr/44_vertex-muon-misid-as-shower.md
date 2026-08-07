# doc pr/44 — vertex long muon mis-ID'd as e-/pi0 arm (18255-142421) + PF orphan parentage

Owner report on Bee set `73cd68ed` (pr43-after, idx 0 = SBND 18255 evt
142421), two layers:

1. The pi0's "gamma 163 → e- 163" arm continues (in the image) into a
   mu- 207 MeV — but that muon draws attached to the neutrino vertex in the
   particle flow.  Owner physics truth: **the whole arm is one long muon
   from the beginning**; its mis-ID'd first part got paired into the pi0.
   Expected vertex topology: THREE prongs — a long muon, pi+ 196 → proton
   159, pi+ 499 → proton 174 (the last already correct).
2. The proton 159 that (in the image) hangs off the pi+ 196 also draws
   attached to the neutrino vertex.

Owner principle governing both: *the PF tree must mirror the image's
segment-graph logic — "if inconsistency, it would be very difficult to
maintain the two."*

Layer 1 is this doc (Part B fix, `shower_long_muon_keep_type`).  Layer 2 is
[doc pr/38](38_pf-missing-tracks.md) Round 4 (Part A fix,
`pf_orphan_track_parentage`) — the two shipped together (same commit, same
gate set); this doc carries the shared Repro block and gates.

## Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
# HEAD for every number here: 225d7e7e (pr/43 rollback; source byte-identical
# to 18936f16, verified `git diff 18936f16 225d7e7e` empty).

# B0 trace (which pass writes the electron labels):
WCT_PID_WRITE_DEBUG=2 WCT_BEE_PF_PRINT=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 /home/xqian/tmp/pr44/trace0 data 142421
grep "PID_WRITE_DEBUG" /home/xqian/tmp/pr44/trace0/pr_evt142421/stdout.log | grep "clus=7 "

# Single-event demos (G2):
./run_pr_chain_batch.sh work-ncpi0-cb0805 /home/xqian/tmp/pr44/off1 data 142421
SBND_PF_ORPHAN_TRACK_PARENTAGE=1 WCT_BEE_PF_PRINT=1 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 /home/xqian/tmp/pr44/onA data 142421
SBND_PF_ORPHAN_TRACK_PARENTAGE=1 SBND_SHOWER_LONG_MUON_KEEP_TYPE=1 \
  WCT_BEE_PF_PRINT=1 WCT_PID_WRITE_DEBUG=2 \
  ./run_pr_chain_batch.sh work-ncpi0-cb0805 /home/xqian/tmp/pr44/onAB data 142421

# 48-event nueCC gate arms:
PR_JOBS=6 ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr44-off48 data
SBND_PF_ORPHAN_TRACK_PARENTAGE=1 PR_JOBS=6 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr44-onA48 data
SBND_PF_ORPHAN_TRACK_PARENTAGE=1 SBND_SHOWER_LONG_MUON_KEEP_TYPE=1 PR_JOBS=6 \
  ./run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr44-onAB48 data
# G1 reference: /home/xqian/tmp/pr43_cleanhead_ref48b (pr/43's git-stash
# clean-HEAD arm at 18936f16 == this HEAD's source; its freshness was itself
# gated in doc pr/43 against a run-to-run reproducibility check).
# Compare with abtest/hash_archive.py member hashes, never md5sum (M2).
```

## Symptom

Bee mc.json at HEAD (all current SBND production knobs, incl. pr/40 r6):

```
12: pi0  117 MeV
  13: gamma  163 MeV
    7023: e-  163 MeV      <- actually the muon's first 64 cm
  14: gamma  26 MeV ...
7011: pi+  196 MeV          <- flat root at nu vertex
7012: proton  159 MeV       <- flat root; image: daughter of 7011
7018: mu-  207 MeV          <- flat root; image: continuation of the "e-" arm
```

Cluster-7 graph ground truth (calib dump): main vertex v7003 → seg 7023
(1.2 cm stem) → v7013 → {seg 7024 (63.4 cm) → v7020 → seg 7018 (79.5 cm,
pdg 13); seg 7011 (pi+ 68 cm) → v7014 → seg 7012 (proton 17.5 cm)}.  The
chain 7023→7024→7018 is collinear end-to-end (unit directions
(0.599,0.477,0.642) vs (0.581,0.476,0.661), dot ≈ 0.9997 — one ~143 cm
line) and shower 7023's own `stem_dqdx` (20 samples) sits at 1.0–1.4× MIP,
median ≈ 1.1: a single MIP, not an EM shower.  Shower 7023 is paired into
the pi0 (`pio_id 0`, `pio_mass 117.8`).

## Root cause (B0 trace, `WCT_PID_WRITE_DEBUG=2`)

The chain was **already correctly labelled muon** before shower clustering:
7023 `0→13` (NeutrinoVertexFinder.cxx:1480), 7024 `13` (reaffirmed :1589),
7018 `13` (track PID).  Then, in order:

1. `shower_clustering_with_nv_in_main_cluster` seeds a **long-muon
   pseudo-shower** with start segment 7024 (in `segments_in_long_muon`;
   cached shower type recorded 13 at the seed — the correct muon-reassembly
   mechanism, NeutrinoShowerClustering.cxx:130-132).
2. The `update_particle_type` call that follows completion
   (NeutrinoShowerClustering.cxx:163) runs **unconditionally** — and its
   majority vote counts every non-proton member (muons included!) as
   `shower_length`, so a pure muon chain always trips
   `shower_length > track_length` and the start segment is relabelled
   **13 → 11** (PRShower.cxx:842).  The call's own comment claims
   "long-muon start segments retain PDG=13" — false for any MULTI-segment
   muon shower (`update_particle_type` early-returns only on ≤1 edges).
   **This call is a toolkit-only addition** (commit `18f09178`,
   2026-03-31 "fixed a bug"); the prototype completes the structure and
   goes straight to the deliberate long-muon → EM reclass loop
   (`NeutrinoID_shower_clustering.h:1709-1717`) — it never re-types a
   long-muon start segment there.
3. Everything downstream then works "correctly" on the poisoned label:
   `shower_clustering_connecting_to_main_vertex` candidate 7023 (1.2 cm
   stub — too short for the r6 F10 straightness veto, pdg 13 passes every
   candidate skip) builds a shower whose F12 absorb guard correctly
   excludes the still-muon 7018 and the pi+ 7011 but absorbs the
   now-"electron" 7024; the shower = {7023, 7024, 7019, 7082} passes the
   `n_multi_vtx > 0` acceptance, force-sets stem 7023 → 11 (probe tag
   `connecting_to_main_vertex`), and id_pi0 pairs the fake 163 MeV "gamma"
   with the real 26 MeV gamma (pio_mass 117.8).  The muon pseudo-shower is
   then deleted as a conflicting shower (start vtx inside the new shower's
   view).

## Why it hid

The relabel only bites MULTI-segment long-muon pseudo-showers whose chain
passes near more graph structure — a single-segment long muon (the common
case) is skipped by `update_particle_type`'s `edges() <= 1` early-return,
which is exactly the case its comment was written against.  For the broken
case, the display shows a plausible EM arm merged into a pi0 with a
"correctly" separated muon at the end — only the owner's image-vs-flow
consistency scan caught it.

## Fix

`shower_long_muon_keep_type` (TaggerCheckNeutrino → PatternAlgorithms
`m_shower_long_muon_keep_type`, C++ default **false** = legacy =
byte-identical).  When on, a shower whose cached `particle_type` is ±13
(recorded at the in_main_cluster seed) **skips the update_particle_type
call at that one site** — prototype parity.  Every other
update_particle_type site is untouched; the deliberate long-muon → EM
reclass loop below (n_others ≥ 2·n_muons …) still runs; the PDG==0
fixup after the vote is naturally inert for a pdg-13 start segment.

With the knob on, the whole cascade unwinds without further code: the
still-muon 7024 is guard-excluded (F12) from the 7023-stub candidate
shower, which shrinks to {7023, 7082}, fails `n_multi_vtx > 0`, and is
never accepted — no fake e-, no pi0 pairing, muon chain intact.

Part A (`pf_orphan_track_parentage`, doc pr/38 Round 4) independently makes
the PF assembly attach barrier-orphaned tracks by graph topology instead of
flat roots — see that doc for its mechanism.  The two compose: in this
event with B on, the muon shower's start vertex (v7013) is barrier-exempt,
the track BFS reaches everything, and A has nothing to do (0 ANCHOR
lines); with B off (e.g. a genuine EM shower with a guard-excluded track),
A is what fixes the display.

## Demonstration (G2)

`off1` reproduces the symptom byte-identically to the stored clean-HEAD
artifact (`work-pr43-head-ncpi0`, mabc member hashes identical).

**G2a — A only** (`onA`): strict graph topology, all three ANCHOR lines:

```
12: pi0  117 MeV
  13: gamma  163 MeV
    7023: e-  163 MeV
      7011: pi+  196 MeV
        7012: proton  159 MeV      <- pi+ -> proton restored
      7018: mu-  207 MeV           <- muon on the arm it continues
[fill_bee_pf_tree] ANCHOR orphan seg=7011 -> parent=shower:7023
[fill_bee_pf_tree] ANCHOR orphan seg=7012 -> parent=7011
[fill_bee_pf_tree] ANCHOR orphan seg=7018 -> parent=shower:7023
```

**G2b — A+B** (`onAB`): the owner-truth three-prong vertex:

```
7023: mu-  4 MeV                   <- muon stem from the nu vertex
  7024: mu-  332 MeV               <- muon body (long-muon shower node:
  7011: pi+  196 MeV                  segs 7024+7018+stubs, ~148 cm)
    7012: proton  159 MeV
  7082: mu-  3 MeV
7021: pi+  499 MeV
  7022: e-  23 MeV
  7013: proton  174 MeV
7081: proton  8 MeV
```

No `13 → 11` write fires anywhere in the trace; the pi0 node dissolves
(its 163 MeV partner was the muon) and the real 26 MeV gamma arm stands
alone.  The mu- 332 MeV node is the standard reassembled-long-muon shower
leaf (kine over the whole chain), exactly how long muons display
everywhere else.

## Gates

- **G0 freshness**: `local/lib/libWireCellClus.so` rebuilt+installed after
  the last source edit (build rc=0, 1m10s + install).
- **G4 unit tests**: `./build/clus/wcdoctest-clus` 1053/1053 assertions
  PASS, including two new cases: `shower_long_muon_keep_type` pinned false
  in the TaggerCheckNeutrino defaults, and a new
  `MultiAlgBlobClustering::BeePFConfig` case pinning all seven pf switches
  false (closes the gap where none of the pf_* defaults were pinned).
- **G5 compiled-config**: knobs-off compile of `wct-pr-perevt.jsonnet`
  (runner TLA set incl. the full `pipeline_names`) byte-identical (`cmp`)
  to the clean-HEAD compile; knobs-on compile carries
  `shower_long_muon_keep_type` in the TaggerCheckNeutrino node and
  `pf_orphan_track_parentage` in the MABC `bee_pf[0]` block — each key in
  exactly one node.
- **G1 knobs-off byte-identical (48-evt nueCC48)**: `work-pr44-off48` (new
  binary, both knobs off) vs `/home/xqian/tmp/pr43_cleanhead_ref48b`
  (pr/43's validated git-stash clean-HEAD reference at the same source
  point): **48/48 events PASS, 96/96 archives** (mabc-pr.zip +
  pctree-pr-evt*.tar.gz member-content hashes) **+ nusel-table.tsv
  identical**.  Single-event cross-check: `off1`'s mabc member hashes ==
  the stored clean-HEAD `work-pr43-head-ncpi0` artifact (10550's mabc hash
  `139918f0…` also matches the pr/40-r6/pr/43 clean-HEAD fingerprint).
- **G3 population (48-evt nueCC48)**:
  - A-only arm `work-pr44-onA48` vs `work-pr44-off48`: pctree
    **48/48 byte-identical**, nusel-table **0-diff** (display-only proof);
    mabc moved in **exactly 1/48 events** (239794), member-level diff
    confined to `data/0/0-mc.json`; census: **27 REPARENTs** (former flat
    orphan roots gaining their graph parent, no text/pdg/KE changes) +
    **1 ADD** (`e- 0 MeV` shower leaf below the 5 MeV `em_ke_min` KeepMC
    floor that now survives because it gained orphan children — the
    documented hierarchy-preservation rule).  Fully attributed.
  - A+B arm `work-pr44-onAB48`: **byte-identical to the A-only arm on all
    48 events** (mabc 48/48) and pctree identical to off 48/48, nusel
    **0-diff** — `shower_long_muon_keep_type` fires on NO nueCC48 event;
    its only observed firing is the motivating ncpi0 event 142421.
  - ncpi0 19-event pair (`work-pr44-off19n` vs `work-pr44-onAB19n`, both
    knobs on): nusel-table **0-diff**, pctree **19/19 byte-identical**;
    census **4/19 events**, all attributed:
    - **142421** — the intended restructure: 7023 `e- 163` → `mu- 4`
      (RETEXT), muon body `7024 mu- 332` + stub `7082 mu- 3` appear under
      it, `7018 mu- 207` disappears as a separate node (now inside the
      muon shower leaf), `7011 pi+ → 7012 proton` chain under the stem;
      the pi0 dissolves and the surviving real gammas re-pair, which
      re-shuffles the sequentially-allocated pseudo-gamma wrapper ids
      (the DEL/ADD/RETEXT entries on ids 8-20 are wrapper bookkeeping —
      the real leaves 43025/59041/64046/81063 only re-hang).
    - **21073** (2), **285567** (4), **521075** (1) — pure Part A
      REPARENTs: former flat-root mu-/proton nodes gain their graph
      parent (e.g. `mu- 56` under `proton 425` under 11010; four tracks
      under 8020).  No text changes.

## Flip

Both knobs flipped **SBND PRODUCTION DEFAULT ON** in
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` (flip bar met: zero
nusel verdict flips on nueCC48 *and* ncpi0-19; every mc.json move
attributed).  Flip verified cfg-only with bare runs (no env overrides):
142421 mabc member hashes == the gated `work-pr44-onAB19n` arm
(`d3330604…`); 239794 mabc member hashes == the gated
`work-pr44-onA48`/`onAB48` arm (`4c01d890…`).

## Scope and not-claimed

- Part B touches exactly one `update_particle_type` call site (the
  in_main_cluster seeding loop); the sites in
  `shower_clustering_in_other_clusters`, `examine_showers`, etc. are
  unchanged — long-muon pseudo-showers are only seeded at the fixed site.
- The kine/energy side of 142421 (`kine_reco_Enu`; pr/43 F4's territory,
  rolled back with pr/43) is NOT addressed here: with B on the muon chain
  enters the energy accounting through the normal muon path, but no
  kine-specific knob is (re)introduced.
- The prototype's own flat orphan emission (`mc_mother=0`) remains the
  legacy/knob-off behavior; Part A is a designed divergence documented in
  `clus/docs/porting/porting_dictionary.md` (the prototype cannot reach the
  orphan-behind-shower state because it has no F12 absorb guard).

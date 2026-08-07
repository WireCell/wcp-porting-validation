# doc pr/46 — evt 18255-55595: long-muon formation robustness (stub-blocked chain) — `long_muon_stub_bridge`

STATUS: SHIPPED — `long_muon_stub_bridge` SBND PRODUCTION ON (flip verified); class (d) documented as follow-up (§9)

## 0. Repro block

```bash
# Phase-0 diagnostic arm (206 evts = 200-evt mcp1k subset ∪ 10 survey candidates),
# temp WCT_PR46_DIAG build at 15e6d983 (diag code NOT committed):
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
env WCT_PR46_DIAG=1 SBND_MAX_JOBS=6 ./run_pr_chain_batch.sh work-mcp1k-cb0805 \
    work-pr46-diag0 data $(tr '\n' ' ' < /home/xqian/tmp/pr46/diag_events.txt)
# PF-tree survey of the full 1000-evt production arm:
python3 /home/xqian/tmp/pr46/survey_stub_long.py work-mcp1k-cb0805
# Gate arms: see §6.
```

## 1. The owner's case, precisely

Evt 18255-55595, nu-candidate cluster 8 (in-beam flash, 188.1 cm, 3303 pts,
`nu-candidate` in nusel). Production PF (work-pr45-m200on == current
production knobs):

```
[8007] pi+   5 MeV  L~2.4cm   (-177.4,-52.7,195.6) -> (-178.6,-54.7,196.1)   <- vertex stub
  [8005] mu- 451 MeV L~181.5cm (-178.6,-54.7,196.1) -> (-158.9,-193.3,311.6) <- the muon
    [8009] e-  10 MeV L~4.1cm                                                <- Michel
[8011] proton 121 MeV L~10.8cm
[8019] mu-  48 MeV  L~2.7cm                                                  <- "the muon" per slot logic
```

Owner: this is much more likely ONE long muon from the neutrino vertex broken
into two pieces; the long-muon category (`segments_in_long_muon` → type-13
pseudo-shower) should hold it. Owner's physics criteria for the refinement:

- downstream muon SHORT + large-angle kink ⇒ genuine pi→mu, keep;
- short track → MULTIPLE outgoing tracks ⇒ genuine hadronic pion, keep;
- downstream track VERY LONG + junction angle consistent + branching at most
  a delta-ray electron / tiny fragment ⇒ single long muon (the delta ray is a
  positive muon signature, must not veto);
- veto only when large angle AND multiple substantial outgoing tracks;
- junction angle from FITTED local directions, not raw endpoints (a 2.4 cm
  stub's endpoint direction is noise).

## 2. Symptom

A broken long muon whose first piece is a short (<6 cm) vertex stub is never
assembled into the long-muon category. Downstream consequences on 55595: the
muon-slot competition (`examine_direction`, NeutrinoVertexFinder.cxx:1601-1750)
only considers vertex-incident pdg-13 arms, so the 2.7 cm sibling stub 8019
wins "the muon" and the gateway stub 8007 is demoted to pi+ — producing the
absurd `pi+ 5 MeV → mu- 451 MeV` parentage with a 2.7 cm "muon" beside it.
No long-muon pseudo-shower forms, so the PF keeps two pieces and kine/taggers
see a 181 cm "daughter" muon hanging off a 2.4 cm pion.

## 3. Root cause (Phase-0 diagnostic, measured)

The long-muon set is populated at exactly ONE site: `examine_direction`'s
"Find long muon candidates" block (clus/src/NeutrinoVertexFinder.cxx:1541-1599):
seed loop over main-vertex out-edges (seed gate median dQ/dx ≤ 1.3×MIP), then
a `find_cont_muon_segment` chain walk (:1095-1191) with continuation criteria
`angle < 10°` (15° when the incoming segment is < 6 cm) measured between
15 cm fitted directions, `ratio < 1.3`, acceptance
`total > 45 cm && max > 35 cm && size > 1` (:1583). Prototype parity:
NeutrinoID_track_shower.h:2137-2190 + 2304-2369, byte-same thresholds.

Temp env-gated diagnostic (WCT_PR46_DIAG, removed before commit) on 55595:

```
PR46DIAG seedloop: cluster=8 flag_final=1 flag_fill=1 vtx=(-177.4,-52.7,195.6) is_main=1
PR46DIAG seed: cluster=8 seg=7 len=2.47cm dqdx_ratio=0.486 pass=1     <- stub SEEDS fine
PR46DIAG cont: sg=7 len=2.47cm -> cand sg2=5 len=192.92cm angle=32.7 angle1=35.4
               ratio=1.086 angle_ok=0 ratio_ok=1                      <- muon is MIP, ANGLE kills it
PR46DIAG chain: cluster=8 seed_seg=7 nsegs=1 total=2.47cm max=2.47cm accept=0
```

The failing gate is the **junction angle**: the 2.2-2.5 cm stub's fitted
direction is dominated by vertex-region noise, reading 30.5-35.4° against the
muon's clean 15/50 cm directions — over the 15° short-segment tolerance. Seed
dQ/dx passes (0.486), the muon is MIP (1.086), there is no once-per-cluster
freeze at the final call, and the 45/35/size acceptance would pass. Everything
but the angle already agrees this is one muon.

## 4. Why it hid

The angle tolerance (10°/15°) is a faithful port of the prototype and is
correct for junctions between two well-fit long pieces; it was never
revisited for the stub case where one side's direction estimate carries no
information. On uBooNE-scale samples the broken-first-piece topology is rare
enough that the category's other entry paths (well-aligned breaks) masked it.
The pr/43-r2 K2 knob (`single_muon_long_muon_claim`) sits directly downstream
but only fires when the chain already exists — on 55595 it never could.

## 5. Sample-wide statistics (Phase 0, 206 events)

PF-tree survey of the full 1000-evt production arm: 421/1000 events have a
neutrino-candidate PF; 11 rows show a short root stub (<8 cm) feeding a long
(>50 cm) track. Diagnostic classification of every candidate + the 200-evt
subset:

| evt | topology (production PF) | diag verdict |
|---|---|---|
| **55595** | pi+ 5 MeV 2.4cm → mu- 451 MeV 181.5cm | **(b) angle 30.5-35.4°** — the fix target |
| **61461** | (18002 class) 3.1cm stub → 94.1cm MIP | **(b) angle 16.4°**, call-dependent (final-call seed dqdx 2.52) |
| 66118 | pi+ 2 MeV 2.2cm → mu- 193 MeV 70cm | angle 70-82° — genuine pion per owner criteria + pr/43-r2 K1 design; **must NOT bridge** |
| 54175 | mu- 7 MeV 2.3cm → mu- 428 MeV 180cm | (d) muon direct at examine-time vertex (181.5cm, single segment, `size>1` rejects); stub-root topology arises post-formation |
| 173234 | proton 21 MeV 1.0cm → mu- 546 MeV | (d) same (235.7cm direct) |
| 59247 | proton 77 MeV 4.8cm → mu- 376 MeV | (d) same (158.6cm direct; stub dqdx 4.09 = plausible real proton) |
| 287555 | proton 12 MeV 0.9cm → mu- 290 MeV | (d) same (118.3cm direct) |
| 315497 | neutron 3.2cm → mu- 244 MeV | (d) same (335.9cm direct) |
| 64921 | neutron 7.0cm → mu- 85.9cm | (d) same (89.2cm direct) |
| 391766 | mu- 6 MeV → two long mu- children | chain 105.6+176.9cm accept=1 at a per-cluster call — already in category |
| 284145 | proton 95 MeV 5.8cm → proton 366 + mu- 530 | **negative control**: junction candidates fail MIP (1.65-3.4) and angle (85°+); multiplicity veto would also fire. Correctly left a hadronic vertex |

Class (b) — the angle-blocked stub bridge — is what this round fixes.
Class (d) is a DIFFERENT weak point (see §9): prototype-parity, majority
population, deliberately NOT changed this round.

## 6. Fix: `long_muon_stub_bridge` (C++ default false; SBND status: see §7)

One knob-gated extra acceptance disjunct inside `find_cont_muon_segment`
(clus/src/NeutrinoVertexFinder.cxx), reached ONLY through a new default-false
parameter `flag_stub_bridge` passed from the formation walk in
`examine_direction` — the other two call sites (`examine_main_vertices_local`,
NuMu tagger track extension) keep legacy behavior even when the knob is on.

Bridge conditions (ALL must hold, evaluated only when the legacy `angle_ok`
already failed):

| condition | value | rationale |
|---|---|---|
| incoming segment length | < 6 cm | the existing short-segment threshold; only stubs whose fitted direction carries no information |
| candidate length | > 35 cm | matches the chain acceptance `max_length` bar; a SHORT downstream muon keeps genuine pi→mu decay kinematics (owner criterion 1) |
| candidate median dQ/dx | < 1.3 × MIP | unchanged legacy MIP test |
| fitted junction angle | < 45° (either 15 cm or 50 cm direction) | Phase-0 separation: 55595 needs ≥ 35.4°, genuine-pion 66118 measures 70-82° and must not merge |
| junction multiplicity veto | no OTHER out-edge > 10 cm that is track-like (not shower-flagged, pdg ≠ ±11) | owner criterion 3: multiple substantial outgoing tracks = hadronic vertex; a delta-ray electron or tiny fragment does NOT veto |

The seed dQ/dx gate (≤ 1.3), the once-per-cluster freeze and the
`total > 45 cm && max > 35 cm && size > 1` acceptance are all UNCHANGED. Once
the chain forms, existing production machinery does the rest: segments
force-typed 13 and exempt from pion demotion, K2 (`single_muon_long_muon_claim`,
ON) claims the vertex muon slot with the summed chain length, the type-13
pseudo-shower wraps the chain into ONE PF node, pr/44 keep-type (ON) protects
the cached type, pr/45 paint (ON) renders it as track.

Threading (identical pattern to pr/43-45): `NeutrinoPatternBase.h` member
`m_long_muon_stub_bridge{false}` with full docstring;
`TaggerCheckNeutrino.{h,cxx}` member + configure + cfg echo + forward;
`cfg/pgrapher/common/clus.jsonnet` signature arg + key-suppression
(`+ (if long_muon_stub_bridge then { long_muon_stub_bridge: true } else {})`);
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` 4 sites;
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` TLA + forwarding; runner
`run_pr_chain_batch.sh` tri-state env `SBND_LONG_MUON_STUB_BRIDGE`
(unset = cfg default, 1 = on, 0 = off); default pin in
`clus/test/doctest_clus_knob_defaults.cxx`.

## 7. Gates and flip decision

Binary lineage: base arms at clean HEAD 15e6d983 (build 10:58), all later
arms on the knob build (11:04); freshness proofs done before each launch.

| gate | arms | result |
|---|---|---|
| G1 knob-off byte-identical (nueCC48) | work-pr46-off48 vs work-pr46-base48 | **PASS 48/48 evts, 96/96 archives** (hash_archive member content) |
| G1 knob-off byte-identical (ncpi0) | work-pr46-off19n vs work-pr46-base19n | **PASS 19/19, 38/38** |
| G4 unit tests | ./build/clus/wcdoctest-clus | **PASS 1063/1063** (incl. new default pin) |
| G5 compiled config | /home/xqian/tmp/pr46/cfg_{off,head,on}.json (runner pipeline_names) | **PASS**: off byte-identical to clean HEAD; `long_muon_stub_bridge: true` appears exactly once when on |
| G2 case checks | work-pr46-oncase (7 evts, knob on) | see §7.1 |
| G3 census | work-pr46-{on48 vs off48, on19n vs off19n, m1konb vs m1koffb} | PENDING |

Note: work-pr46-m1koff is an INVALID PARTIAL (first 1k launch at 6 jobs,
stopped at 11:13 to relaunch at owner-authorized 24-way parallelism; a few
events possibly killed mid-write). Left in place, superseded by
work-pr46-m1koffb / work-pr46-m1konb.

### 7.1 G2 case checks (knob on, work-pr46-oncase)

- **55595 — the owner's ask, delivered**: PF root becomes ONE node
  `[8007] mu- 455 MeV L~183.2cm` spanning vertex (-177.4,-52.7,195.6) →
  (-158.9,-193.3,311.6) (2.4 cm stub + 181.5 cm muon assembled in the
  long-muon category; the Michel is absorbed into the pseudo-shower
  structure, prototype-parity for this category). The 2.7 cm former "muon"
  demotes to `pi+ 48 MeV`. nusel row IDENTICAL to current production
  (verdict nu-candidate unchanged) — the change is purely PF assembly.
- **66118 (must-not-merge)**: not merged — 9004 stub keeps its child; the
  70-82° junction fails the 45° cap. (Its 9003 label pi+ is the pr/43-r2 K1
  designed relabel, present in knob-off production too.)
- **61461**: not merged — junction multiplicity veto (a second 11.5 cm
  track-like arm feeding protons). Angle alone (16.4°) would have bridged;
  the veto encodes the owner's multiple-outgoing-tracks criterion
  conservatively (it fires regardless of angle).
- **59247**: unchanged (class (d), bridge cannot reach).
- **54175 / 173234 / 284145 / 59247 / 66118 / 61461 — ALL PF-IDENTICAL
  on↔off** at current HEAD (work-pr46-oncase vs work-pr46-m1koffb): every
  apparent difference vs work-mcp1k-cb0805 quoted in early exploration was
  pr/43-r2 + pr/45 flip effects postdating that arm, not the bridge. The
  bridge's case-event footprint is 55595 alone.

### 7.2 G3 census and attribution

- nueCC48: PF census **0/48 moved**, nusel table IDENTICAL, and knob-on is
  **ARCHIVE-identical** (96/96 member-content hashes on48 vs off48) — the
  bridge never fires on the standard nueCC manifest.
- ncpi0: **0/19 moved**, nusel IDENTICAL, archives 38/38 identical.
- Full 1000-evt MC sample (work-pr46-m1konb vs work-pr46-m1koffb, both
  1000/1000 rc=0): PF census **3/1000 events moved**, nusel table **0-diff
  over all 1000 events**. Every mover is the designed stub-bridge assembly,
  chain-formation trace confirmed `nsegs=2` on each:

| evt | before (knob off) | after (knob on) | bridged stub |
|---|---|---|---|
| **55595** | pi+ 5 MeV 2.4cm → mu- 451 MeV 181.5cm (+Michel); sibling stub "mu- 48 MeV" | **single `mu- 455 MeV` 183.2cm root**; sibling demotes pi+ 48 MeV | 2.4 cm (seed dqdx 0.486, angle 32.7°) |
| **73004** | pi+ 32 MeV 4.3cm → mu- 235 MeV 88.8cm | **single `mu- 226 MeV` 91.9cm root**; sibling stub 15008 mu-→pi+; gammas re-hang (two under proton 15011) | 4.4 cm (total 96.76, max 92.38) |
| **349945** | e- 171 MeV 38.3cm stem with mu- 415 MeV 172.8cm child | **single `mu- 464 MeV` 175.6cm root**; gammas re-hang | 3.1 cm (total 179.88, max 176.75; the 38.3 cm stem is absorbed downstream by the pseudo-shower, not by the bridge) |

- Archive-level footprint on the 1k (member-content hashes, both archives
  per event): **3/1000 events differ, the same three movers, and only
  `mabc-pr.zip`** — `pctree-pr` is hash-identical even on the movers, and
  no bee-layer-only movers exist (/home/xqian/tmp/pr46/hashcmp_m1k.txt).

### 7.3 Flip decision

The owner's standing "flip if clean" bar: zero nusel verdict flips (0-diff
on 1000 + 48 + 19) AND every mover attributed (3/3, all the designed
topology) AND must-not-merge controls hold (66118 angle cap, 61461
multiplicity veto, 284145 hadronic vertex, class (d) untouched). **FLIPPED
SBND PRODUCTION ON** in cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet
(`long_muon_stub_bridge = true`).

Flip-verify (bare arm work-pr46-flipver vs forced-on work-pr46-m1konb),
full-archive member-content hashes, **6/6 MATCH**:

```
55595  mabc 55c4fd8d39b9b5e47eb1992fa2251058  pctree 37ac8fb67ac5006264869d1aecdb6887
73004  mabc 4124d9a33d1384ee2cb57271237acfed  pctree 4c4268630ca999f9b96bb6941be7a68e
349945 mabc ff33a7880005f4511e32e1d1813e8fb6  pctree 49b0dff9b827424156c52a1030852281
```

To veto the flip: set `long_muon_stub_bridge = false` in
wct-pr-perevt.jsonnet (single line) or export SBND_LONG_MUON_STUB_BRIDGE=0
in any runner invocation.

## 8. Bee evidence

`bee/pr46/pr46-{before,after}.zip` (+ index/prid-map), idx 0 = 55595,
1 = 73004, 2 = 349945; before = work-pr46-m1koffb (knob off at HEAD),
after = work-pr46-flipver (bare production post-flip). Upload (owner):

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./upload-to-bee.sh bee/pr46/pr46-before.zip
./upload-to-bee.sh bee/pr46/pr46-after.zip
```

## 9. Class (d) — the direct-single-segment exclusion (documented follow-up, NOT fixed here)

At examine time, six of the candidate events have the long muon as a single
segment directly at the (then) main vertex; the chain walk finds no
continuation and `acc_segments.size() > 1` rejects a single-segment "chain"
(NeutrinoVertexFinder.cxx:1583; prototype NeutrinoID_track_shower.h:2172
identical). The stub-root PF topology (proton/mu 1-7 cm root → long muon
child) appears only in the final PF tree — post-formation restructuring
(e.g. 54175: examine-time main vertex (-75.2,139.7,55.8), muon 181.47 cm
unsplit through the "After shower clustering with NV" print; final PF vertex
(-72.5,139.2,54.1) with a 2.3 cm root stub). The category is frozen by then.

Why not fixed this round: dropping `size>1` would sweep EVERY ordinary
vertex-attached CC muon into the pseudo-shower category (energy path switches
to `calculate_kinematics_long_muon`, tagger features change) — a huge,
un-narrow footprint; and the behavior is prototype-parity (M15: an
undocumented divergence would need an owner decision anyway). The right
follow-up is likely a late re-formation pass after the stage that creates the
stub-root topology, or category entry at PF-fill time for a stub+long-muon
root chain. Needs its own round with its own census.

## 10. Verification

- Byte-identical status: knob-off proven byte-identical to the clean source
  at 15e6d983 (G1, member-content hashes, 96/96 + 38/38); knob-on is NOT
  bit-identical on exactly 3/1000 MC events + their bee layers — that is
  the fix, all three attributed, nusel 0-diff everywhere.
- wcdoctest-clus 1063/1063 on the final binary; G5 compiled-config proof
  off/on; freshness proofs before every arm.
- Determinism note: work-pr46-oncase (6-job launch) and work-pr46-m1konb
  (24-job relaunch) agree byte-for-byte on their overlapping knob-on
  events, and work-pr46-flipver reproduces work-pr46-m1konb — three
  independent runs of the movers are hash-identical.
- Invalid partials (M13, left in place, superseded): work-pr46-m1koff
  (first 1k launch at 6 jobs, stopped at 11:13 for the 24-way relaunch).
- Owner speed-up note: this runner's parallelism env is **PR_JOBS** (default
  6), not SBND_MAX_JOBS — the m1koffb arm ran at 6 jobs before this was
  caught; m1konb/flipver ran at PR_JOBS=24 (load ~7.5/64, ~45 -> ~100
  evts/min).

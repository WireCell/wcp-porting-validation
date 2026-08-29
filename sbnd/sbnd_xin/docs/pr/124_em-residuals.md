# doc pr/124 — EM residuals round: gap band, recognition, pass3_cone

**Status: IN PROGRESS (opened 2026-08-28 night, owner directive).**

Follow-on to doc pr/123 (pass4 over-reach round, closed VALIDATED/ON). The
owner picked three fronts from the post-pr/123 residual assessment for tonight,
"the rest ... the same as before" — i.e. the standing pr/117–123 bar:
measure-first, default-OFF knobs, dual-manifest byte-identical gates
(`scripts/pr85_hash_gate.py`), census + movers adjudication, Bee A/B, flip
only on validation, commit + push.

Owner wording (2026-08-28): start work on

1. *"The 25–40 cm gap band (over-clustering, round-3 candidate). G=40 was
   deliberately conservative. ... The offline prune-scan can sweep G with
   qualifiers (component charge, PID mix, direction w.r.t. the body) against
   the existing labels with zero new arms — the question is whether a
   qualifier tames the 1:1 collateral that killed G=25."*
2. *"Recognition / fake showers. 489327, 69232, 54332's 16014, plus the
   171143/277298 fakes — the root itself is wrong, so no membership knob
   helps. ... a seed-time re-classification campaign is its own measure-first
   round. Related housekeeping: the score-100 sentinel class (rule-assigned
   PIDs) is systematically excluded from every 'confident' gate — worth an
   audit."*
3. *"pass3_cone — now the largest untouched absorber (20 OUT marks on the
   141-set). ... the final-body prune is absorber-agnostic, so pass3's
   detached mistakes are already partially handled; what remains is its
   contiguous share. Same probe-census choreography as pass4 would settle
   it."*

The neutrino-vertex identification thread stays closed (owner, pr/123).

## 0. Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin

# Front A -- gap-band qualifier scan (offline, zero new arms):
./scripts/pr124_gapband_scan.py 'work-pr123r1-r21flip141-*'

# Front B -- sentinel audit is grep-only (sec B.1); seed census after dbg arms:
#   WCT_SHOWER_ABSORB_DEBUG=1 WCT_SHOWER_TOPO_DEBUG=1 dbg arms, then
./scripts/pr124_seed_census.py 'work-pr124r1-dbg-*' 'work-pr124r1-dbg98-*'

# Front C -- pass3 absorber census from the same dbg arms:
./scripts/pr124_pass3_census.py 'work-pr124r1-dbg-*' 'work-pr124r1-dbg98-*'
```

Baselines: 98-set `work-d84r2-prod98-*`; 141-set `work-pr123r1-r21flip141-*`
(current production, incl. pr/123 r2.1 pseudo-neutron correction). Labels
(read-only, M13): `em_labels/emscan-0827` (98-manifest) and
`em_labels/emscan-0828-agent5` (141-manifest).

**Concurrency note (opening night).** The doc-84 session's 3000-event census
(`work-d84r3-cens-mcp{1,2}k`, full current production config) was running when
this round opened; per its request, no builds/installs or shared-jsonnet edits
until it completes. Phase 1 here is purely offline. Once landed,
`work-d84r3-cens-*` becomes the shared full-sample baseline and — logs
permitting — a free 3000-event out-of-sample input for the front B/C censuses
(read-only over its dirs, no reruns).

## A. Front A — the 25–40 cm gap band (prune round 3 candidate)

pr/123 shipped the final-body single-linkage prune at G=40 cm
(`shower_pass4_prune_detached` / `shower_pass4_prune_gap`, SBND ON). G=25/30
were rejected because the collateral was 1:1 — every additional OUT member
pruned cost roughly one TARGET member (the 54332-122091 / 281325 satellite
class; the owner later ruled those two rows vertex-bad, but the bar for a
tighter G remains zero collateral on correctly-vertexed showers).

Worst post-pr/123 residual rows, all sitting in the 25–40 band:

| event | qF1 (r21flip141) | note |
|---|---|---|
| 406125 | 0.097 | q_extra 2.6e6; all four OUT marks pruned at G=25 in the pr/123 offline scan |
| 94392  | 0.221 | |
| 175896 | 0.327 | |
| 286655 | low | band component(s) |
| 283515 | partial | parts in band |
| 278420 | — | contiguous far chain, first link 23.9 cm — inside even G=25; NOT a gap-band target, listed as a known non-catch |

**Question under measurement:** with G tightened into the band (25/30/35),
does a per-component qualifier — summed charge (absolute and as a fraction of
the shower), PID mix (track-like member 13/211/2212), summed length, component
direction w.r.t. the kept body axis, gap distance — separate the OUT
components from the TARGET (collateral) components on BOTH label sets?

Tool: `scripts/pr124_gapband_scan.py` (fork of `pr123_prune_scan.py`; same
final-body single-linkage machinery, adds per-component qualifier columns and
per-qualifier collateral tables). Runs on the `work-pr123r1-r21flip141-*`
calib dumps — production arms, so components already surviving the G=40 prune.

Guard rows (must NOT newly regress owner-approved outcomes): 54332-122091,
281325, 348471 (0.912, sheds only unlabeled 31011 today), and the 8 rows at
1.000 from pr/123.

Outcome gate: a qualifier reaching zero collateral on both manifests graduates
to a default-OFF second-tier knob (`shower_pass4_prune_gap2` + qualifier
param); anything short of that ships as a measurement section only (the
pr/118-P2 / pr/119 precedent), or as a morning decision table if the trade-off
is a genuine owner call.

### A.1 Measurement (offline scan, both manifests)

`scripts/pr124_gapband_scan.py --seg-join` over `work-pr123r1-r21flip141-*` +
`work-d84r2-prod98-*` (dedup on overlap). Marks are joined at SEGMENT level to
each seg's *current* owner shower: a mark recorded for the owner keeps its
verdict; IN-for-another-shower while owned here counts OUT (the seg belongs
elsewhere); OUT-of-elsewhere only is uninformative (unlabeled). The plain
shower-key join (pr/123 style) is retained as the default mode; the seg-join
matters because the pr/123 flip restructured several labeled showers.

Detached components by class per G (BAD = only OUT marks, COL = holds IN
marks, UNL = unlabeled only):

```
  G_cm   BAD   COL   MIX   UNL
    25    15     3     0     7
    30     6     2     0     3
    35     1     2     0     1
    40     0     1     0     0
```

The 25-40 band (detached at G=25, still attached at the production G=40)
holds 24 components: 15 BAD, 2 COL, 7 UNL — full rows in
`docs/pr/pr124-gapband-components.tsv`. The 142421 38-member component
(33 inferred-OUT: they are IN-marks of the *other* pi0 photon) is the pi0
collinear split — owner-gated thread, excluded from the knob stats below and
handled in its own round if opened.

**Owner's worst-row reclassification (the key finding).** Of the five rows
named in the directive, only ONE is actually gap-band-detached on current
production:

| event | scan verdict on r21flip141 |
|---|---|
| 406125 | band component (4 marks, gap 33.7, **q_frac 0.949**, mdqdx **2.86 MIP**, ang 4.8°) — heavily-ionizing on-axis blob, caught by the dQ/dx qualifier |
| 94392  | NOT band: both OUT muons (29.8 + 46.8 cm, pdg 13) are **contiguous** members — under the 50 cm pass4 track guard; member-level residual |
| 175896 | NOT band: OUT protons (5.6 + 17.6 cm, pdg 2212) contiguous — same class |
| 286655 | stale label join: labeled shower 79023 now empty (post-flip restructuring); OUT marks live under shower 19006 — needs re-adjudication |
| 283515 | stale label join: labeled shower 23009 empty; 13 OUT marks under 24014 — needs re-adjudication |

### A.2 Qualifier cut search (core band, pi0 excluded: 14 BAD, 2 COL, 7 UNL)

The two COL components: 469665-15003 (13 segs, 78 cm, q_frac 0.551, ang 7.4°
— genuine on-axis EM continuation) and 168596-85107 (single 0.3 cm stub,
q_frac 0.003). Charge-weighted results (BAD charge total 5.19e6):

```
  cut                       BAD caught (charge)   COL hit          UNL hit
  ang_body>30                5/14 (33%)           0                4
  mdqdx>2.0                  1/14 (50%)           0                0
  ang>30 or mdqdx>2.0        6/14 (83%)           0                4
  ang>40 or mdqdx>2.0        6/14 (83%)           0                1
  len_sum<20                14/14 (100%)          1 (qf 0.003)     6
  nmem<=4 & (trk|ang>30)    13/14 (98%)           1 (qf 0.003)     6
```

**Zero-collateral separator: `ang_body > 40 deg OR mdqdx > 2.0 MIP`** — 6/14
BAD components, 83% of BAD charge, zero IN-collateral on both label sets; the
single UNL hit is 54332's shw-16014 component at 88.7° (the front-B fake
shower — pruning it is neutral-to-good). Margins: BAD-side minimum ang 40.3°
(415278) against a COL maximum of 16.7°; mdqdx catch at 2.86 against COL max
1.33 / UNL max 1.91. The aggressive cuts (len_sum, nmem) reach 98-100% but
all hit the 0.3%-charge COL stub, which is shape-identical to a BAD stub
(409634) — member-count zero collateral is impossible beyond the angle+dQ/dx
pair; that pair is the knob.

### A.3 Knob (implemented default OFF, tier-2 prune)

- `shower_pass4_prune_gap2` (double cm, **0 = off**): re-split the tier-1
  KEPT body at this gap; a sub-component not holding the start segment is a
  tier-2 candidate. Operating point from A.2: 25.
- `shower_pass4_prune2_ang` (deg, default 40): prune candidate if the angle
  at the shower start vertex between the candidate's charge centroid and the
  core component's charge centroid exceeds this.
- `shower_pass4_prune2_mdqdx` (MIP, default 2.0): ... OR its median point
  dQ/dx exceeds this many MIP.
- Disposition identical to tier 1: `Shower::detach_member_set` + re-seed as
  own shower (conn 3/4). Validation + flip decision follow the standing bar.

## B. Front B — recognition / fake showers + the score-100 sentinel audit

Cases where the shower ROOT is wrong, so no membership knob helps: 489327,
69232, 54332's seg 16014, and the 171143 / 277298 fakes. The pr/122 finding
stands: long seeds carry inherited, never re-validated `kShowerTopology`
flags; the `in_main_cluster` seeder accepts the flag with no further test.

Existing instrumentation (reused, no new probes expected for round 1):

- `SHOWER_SEED` line (`pr122_probe_seed`, NeutrinoShowerClustering.cxx:735,
  under `WCT_SHOWER_ABSORB_DEBUG`): per accepted seed — which disjunct
  (traj/topo/pdg11), length, median dQ/dx, in_long_muon.
- `WCT_SHOWER_TOPO_DEBUG` (PRSegmentFunctions.cxx:4112-4124): evaluation-time
  branch + features for every `segment_is_shower_topology` verdict.
- Existing default-OFF knobs at the choke points: `shower_topo_demote_len`,
  `shower_topo_dqdx_guard`, `shower_traj_straight_guard`.

Census (`scripts/pr124_seed_census.py`): every accepted seed × disjunct ×
seed-time features × label verdict on the resulting shower (fake/real), both
manifests. Measured question: does any existing knob threshold — or a
seeder-side re-validation (re-running the topology classifier on seed-time
geometry) — separate the fake-seed class from real stems with zero collateral?
pr/122 measured the *classifier-side* features interleaved; the new angle is
seed-time staleness (flag written on early geometry, segment since regrown).
The fake-root ground truth is the labels' own signal: a `marks_by_shower`
entry keyed by the seed segment with the seed itself marked `out` (the
54332-16014 shape).

### B.2 Preliminary census (pre-flip dbg arms, work-pr123r1-dbg{A2,141v2}-*)

285 accepted in_main_cluster seeds across both manifests; 3 FAKE_ROOT vs 10
GOOD_ROOT (13 MARKED_OTHER ambiguous, rest unlabeled):

| fake root | disjunct | len cm | mdqdx MIP | straight |
|---|---|---|---|---|
| 54332 seg 16014 | topo+pdg11 | 32.3 | 1.67 | 1 |
| 489327 seg 19005 | traj+pdg11 | 22.6 | 1.29 | 0 |
| 69232 seg 20021 | **pdg11 only** | 27.5 | 0.87 | 1 |

GOOD_ROOT: len med 24.9 (p90 35.4), mdqdx med 1.33 (p10 0.99). **The three
fakes fire on three different disjuncts and interleave with good roots on
every seed-time feature** — killing the inherited kShowerTopology flag alone
would fix only 54332; 69232's fake needs no flag at all (it seeds on its
pdg-11 assignment, a straight 27 cm MIP-flat "electron"). The measured
statement: the defect is upstream in the PID write (the pr/122 + sentinel-
audit thread), not in the seeder — a seeder-side guard is measured dead at
this statistics level (pr/118-P2/pr/119 precedent). Final numbers from the
post-flip dbg arms may move this; the conclusion stands unless FAKE_ROOT
count grows a separating tail. 171143/277298 notes confirm "PID verdict,
clustering is fine" — same class.

### B.1 The score-100 sentinel audit (housekeeping, read-only)

Rule-assigned PIDs carry `particle_score = 100`; every "confident PID" gate
written as `particle_score < 1.0` (or similar) silently excludes them. The
pr/123 lost-muon defect was one instance (`segment_orphan_confident_track`).
This section enumerates every such site in `clus/` with a verdict:
load-bearing (the gate SHOULD skip rule-assigned PIDs) vs accidental (the
sentinel class was simply forgotten). Audit only — no code change in this
round without owner sign-off per site.

Sentinel vocabulary found: `100` = rule-assigned PID (default-constructed and
every explicit rule write, e.g. NeutrinoPatternBase.cxx:412/439/470/645,
NeutrinoVertexFinder.cxx:3748/3782); `200` = "score really bad, forced to
shower" (PRSegmentFunctions.cxx:3046). All comparison gates in `clus/`:

| site | gate | verdict |
|---|---|---|
| PRSegmentFunctions.cxx:1700 `segment_bragg_spares_electron_reclass` (pr/40 r10) | `score < 1.0` | **accidental** — a rule-assigned >20 cm track is NOT spared from e- reclassification; same defect shape as the pr/123 lost muon |
| PRSegmentFunctions.cxx:1708 `segment_confident_nonelectron_pid` (pr/93 Cause B) | `score < 1.0` | **accidental** — the pr/123 lost-muon root cause; worked around for the guard-freed set only via `kPass4GuardFreed` |
| PRSegmentFunctions.cxx `segment_orphan_confident_track` (pr/93 r4) | inherits the above | **accidental** (inherited) |
| PRSegmentFunctions.cxx:3043 | `1.0 < score < 100` before force-to-shower | load-bearing — deliberately protects rule-assigned PIDs from the "really bad score" demotion |
| NeutrinoShowerClustering.cxx:3849 | `pdg==2212 && score < 0.3` force-to-e- | load-bearing — a rule-assigned proton correctly keeps its label |
| NeutrinoVertexFinder.cxx:1792 | `score <= 100` before muon reclass | load-bearing (prototype-matched) — blocks the score-200 forced-shower class; note the asymmetry: score-100 rule-assigned e- IS eligible |
| NeutrinoVertexFinder.cxx:3736-3737 (knob-gated safety net) | `2212 && score<0.09` / `13 && score<0.06` | intended-as-written but shares the pattern: a rule-assigned track is never "confident", so it takes the pdg-11 escape |
| NeutrinoTrackShowerSep.cxx:1806 | `score != 100` in score averaging | load-bearing — deliberate sentinel-aware mean |

**Pattern statement for the owner:** the three pr/40/93-era "confident PID"
helpers are the accidental class — every rule-assigned (score-100) PID is
invisible to them, so protections keyed on "confident non-electron" silently
skip exactly the tracks the rules were most sure about. A candidate future
knob would treat `score == 100 && pdg != 0 && |pdg| != 11` as confident in
those three helpers; that is a behavior change with wide reach (owner
decision, not taken tonight).

## C. Front C — pass3 absorbers (largest untouched)

pass3 has three absorb sites (NeutrinoShowerClustering.cxx): `pass3_proximity`
(:1388), `pass3_cone` (:1538), `pass3_cluster_map` (:1583). pass3_cone carries
20 OUT marks on the 141-set — the largest absorber not yet measured with the
pr/120/123 choreography. Notes:

- The pr/123 final-body prune is absorber-agnostic: pass3's *detached*
  mistakes are already (partially) reclaimed. The open share is contiguous.
- Existing instrumentation (reused): `P120_P3CONE` line (~:1490, pr/120
  admission census: seg, pdg, len, site angle, dist, scan-frame ang15/ang60,
  angle_offset) + `SHOWER_ABSORB DIRECT site=pass3_*` tags, all under
  `WCT_SHOWER_ABSORB_DEBUG`.
- An existing decline knob already guards one class:
  `shower_cone_absorb_guard` (pr/93 Cause D) for confidently-PID'd
  straight-long non-electrons.

Census (`scripts/pr124_pass3_census.py`, fork of `pr123_pass4_census.py`):
every pass3 absorb joined to labels; classify contiguous-vs-detached against
the final body (detached = already covered by the prune); feature table
(site_ang, dist, ang15/ang60, pdg, len, dQ/dx) for OUT vs IN absorbs. Outcome
gate as in front A: zero-collateral separator → default-OFF guard knob;
otherwise the census ships as measurement.

### C.1 Preliminary census (pre-flip dbg arms — knobs-OFF binary; the tier-1
prune has since fixed part of this)

2186 pass3 absorbs over 239 events. Labeled: pass3_cone 22 OUT / 24 IN,
pass3_cluster_map 4 OUT / 0 IN, pass3_proximity 0/0. In the (pre-flip) final
dumps all 22 cone OUT stayed kept — but 12 of them are 282979 (0.541→1.000
after the pr/123 flip: they were far-satellite, dist 96-118 cm, and the
tier-1 prune reclaims them). The suggestive pre-flip pattern: OUT cone
absorbs cluster at LARGE dist (13/22 above 60 cm) with small site angles,
while the IN absorbs 76346's live at 55-83 cm too — and the contiguous OUT
share (94392's 16/44 cm muons at ~10 deg, 175896's protons at 33 cm) sits
well inside the IN range on geometry alone; their distinguishing feature is
pdg/track-likeness, i.e. the same track-prong logic as the pass4 guard, at a
site the pass4 guard does not cover (pass3_cone runs earlier; the existing
`shower_cone_absorb_guard` covers only confidently-PID'd straight-long
non-electrons). **The definitive OUT/IN table must come from post-flip dbg
arms** (what tier-1 already reclaims must not be double-counted); rerun in
Phase 3 before any knob design.

## Plan of record (tonight)

1. Doc opened, committed, pushed (this commit).
2. Front A offline scan on existing dumps → §A tables.
3. §B.1 sentinel audit → table.
4. After the d84r3 census window clears: dbg arms
   `work-pr124r1-dbg-{mcp1k,mcp2k}` (141) + `work-pr124r1-dbg98-*` (98),
   production knobs + probe envs, PR_JOBS ≤ 16, hash-gated vs baselines
   (probes are stderr-only ⇒ byte-identical required).
5. Front B/C censuses → §B/§C tables; 3000-event read-only pass over
   `work-d84r3-cens-*` if logs permit.
6. Knobs only where a separator measures zero-collateral on both label sets;
   ambiguous thresholds → morning decision table for the owner. Existing-knob
   default flips are never made overnight.

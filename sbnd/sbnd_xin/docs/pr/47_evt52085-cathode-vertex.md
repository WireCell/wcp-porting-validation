# doc pr/47 — evt 18255-52085: a true neutrino vertex ON the cathode

**Status: FIX SHIPPED (2026-08-07, same day, follow-up session).** §1-§7 are
the original analysis round (unchanged below; they described the tree at
`03ccaaf3` with no code change). **§8 implements O1** as the
`cathode_wide_kink_angle` knob — a fifth `segment_search_kink` accept path at
cathode-crossing fit indices keyed on the skirt-excluded wide-baseline PCA
turn angle — C++ default **OFF** (0 deg), **SBND production ON at 25 deg**.
Knob-off gate `work-pr47f-base48` vs `work-pr47f-off48` byte-identical 48/48;
knob-on footprint on the full 1000-event sample in §8.4. 52085's vertex is
recovered 0.65 cm from truth.

## The owner's question, restated

> Event 18255-52085's true vertex is inside the cathode, near
> (x,y,z) = (-0.9, 112.5, 81.3) cm. Is there a vertex candidate identified
> there? If not, why? Since the cathode region is tricky — through-going
> tracks have consistent directions overall, but the *local* point-to-point
> direction is unreliable near the cathode due to distortion — is there a
> **safe way to allow a vertex at the cathode without breaking (genuinely
> through-going) tracks**?

## Headline

| | |
|---|---|
| True vertex | (-0.9, 112.5, 81.3) cm |
| Nearest reconstructed graph vertex | **31.4 cm away** (a 0.4 cm stub, unrelated cluster) |
| The neutrino-candidate track | ONE unbroken 137.6 cm segment, straight through the true vertex |
| Nearest fit point to truth | 0.06 cm (segment interior, arc 35.4 cm from one end) |
| Real kink at the crossing (skirt-excluded PCA, distortion-tolerant) | **33-38 deg**, stable across skirt 0-5 cm x baseline 10-30 cm |
| Same kink, the shipped index-windowed statistic (`segment_search_kink`'s `refl_angle`) | **never exceeds ~23 deg** anywhere near the crossing |
| Binding blocker | **the accept-test thresholds, NOT the cathode-band veto** — disabling the veto (`cathode_kink_xcut=0`) reproduces this event's `mabc-pr.zip` byte-for-byte (§3) |
| Closest miss | C4 criterion: `sum_angles` 18.79 vs required > 19 — misses by **0.21**, all three other C4 sub-conditions already pass |
| Sample-wide cathode-crossing population (445/1000 events with a dump) | 53 crossing segments / 51 events; median turn 3.9 deg, p90 8.2 deg; only **2** events at turn >= 20 deg |
| `SBND_CATHODE_KINK_XCUT=0` full-1000-event footprint | **11/1000 archive movers, 0/1000 nusel-verdict diffs** (§6.2) |
| **§8 fix**: `cathode_wide_kink_angle=25` (SBND ON) | 52085 vertex recovered **0.65 cm** from truth; proton 243 MeV + mu- 253 MeV two-prong replaces the bogus "proton 560 MeV"; knob-off gate 48/48 byte-identical; footprint §8.4 |

## Repro block

```bash
# HEAD used throughout this doc:
git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD
#   03ccaaf31244d993ee638dc496575dcaee1918cd  (doc pr/46, long_muon_stub_bridge)

# The existing production PR-chain dump for 52085 (used for all "as
# reconstructed today" numbers in sec 1-2):
unzip -p work-mcp1k-cb0805/pr_evt52085/mabc-pr.zip data/0/0-vertices-global.json
cat work-mcp1k-cb0805/pr_evt52085/calib-pr-evt52085.json | python3 -m json.tool | less

# sec 3 diagnostic (SBND_CATHODE_KINK_XCUT is an EXISTING runner knob --
# run_pr_chain_batch.sh:174-177 -- no rebuild needed for this one):
cd sbnd/sbnd_xin
mkdir -p work-pr47-xcut0-case
SBND_CATHODE_KINK_XCUT=0 SBND_CATHODE_X=0 PR_JOBS=4 \
  bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47-xcut0-case data 52085
python3 -c "
import sys; sys.path.insert(0,'../../abtest')
import hash_archive as ha
a=dict(ha.members('work-mcp1k-cb0805/pr_evt52085/mabc-pr.zip'))
b=dict(ha.members('work-pr47-xcut0-case/pr_evt52085/mabc-pr.zip'))
print('SAME' if a==b else 'DIFF')"
#   -> SAME  (the veto changes nothing for this event)

# sec 3's break-time diagnostic used a TEMP env-gated print (WCT_PR47_DIAG)
# inserted into segment_search_kink (PRSegmentFunctions.cxx:284, right
# before the cathode-veto `continue`), rebuilt, run once on 52085, then
# fully reverted (git checkout -- clus/src/PRSegmentFunctions.cxx) and
# rebuilt again before any further arm ran.  grep count after revert:
grep -c WCT_PR47_DIAG /nfs/data/1/xqian/toolkit-dev/toolkit/clus/src/PRSegmentFunctions.cxx
#   -> 0

# sec 4: single-knob-off reruns bisecting the main-vertex flip (all against
# the SAME work-mcp1k-cb0805/ql_evt52085 pctree, so clustering/QL is held
# fixed; only the PR-stage jsonnet/C++ differs):
SBND_OTHER_SEG_EMPTY_2D_GUARD=0                                    bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47-flip-a data 52085
SBND_PSEUDO_SHOWER_TRACK_PAINT=0                                   bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47-flip-b data 52085
SBND_SHOWER_LONG_MUON_KEEP_TYPE=0                                  bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47-flip-c data 52085
SBND_SINGLE_MUON_PROTON_CHAIN_VETO=0 SBND_SINGLE_MUON_LONG_MUON_CLAIM=0 SBND_PID_FLAG_RECONCILE=0 \
                                                                     bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47-flip-d data 52085
SBND_PF_ORPHAN_TRACK_PARENTAGE=0                                   bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47-flip-e data 52085
grep "After improve vertex" work-pr47-flip-{a,b,c,d,e}/pr_evt52085/stdout.log

# sec 6 census (no runs, no rebuild -- reads the existing 445 calib-pr dumps):
python3 scripts/analysis/pr47/cathode_junction_census.py

# sec 6.2 full-1000-event veto-off arm:
mkdir -p work-pr47-xcut0-1k
PR_JOBS=24 SBND_CATHODE_KINK_XCUT=0 SBND_CATHODE_X=0 \
  bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47-xcut0-1k data $(ls work-mcp1k-cb0805 | grep '^pr_evt' | sed 's/pr_evt//' | sort -n)

# ---- sec 8 (fix round, toolkit HEAD = 20098cbf, the pr/47-fix commit) ------
# Pre-edit baseline with the clean 03ccaaf3 binary, then the knob-off gate
# with the new binary (SBND_CATHODE_WIDE_KINK_ANGLE=0 forces the C++ OFF
# path; the runner knob is run_pr_chain_batch.sh, CATH_TLA block):
PR_JOBS=6 bash run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr47f-base48 data
# (edit + wcbuild + freshness proof + ./build/clus/wcdoctest-clus here)
SBND_CATHODE_WIDE_KINK_ANGLE=0 PR_JOBS=6 \
  bash run_pr_chain_batch.sh work-nuecc48-cb0805 work-pr47f-off48 data
# member-hash mabc-pr.zip + pctree per event (hash_archive.py) + byte-cmp the
# merged nusel tsvs -> 48/48 artifacts identical, both tsvs identical (sec 8.3).

# Knob-on smoke (bare run = production = ON at 25 deg):
PR_JOBS=4 bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47f-case data 52085

# Knob-on full-1000-event footprint arm (sec 8.4):
PR_JOBS=24 bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47f-on1k data
python3 scripts/analysis/pr47/on1k_compare.py   # movers + vertex + nusel diffs

# sec 8.4 mover explanation: both-direction turn recompute over the 445
# dumps (the census's skirt_turn_angle only measured neg->pos crossings),
# and the 289559 threshold bracketing:
python3 scripts/analysis/pr47/turn_bothdir.py
for A in 0 35 60 179; do
  SBND_CATHODE_WIDE_KINK_ANGLE=$A PR_JOBS=2 \
    bash run_pr_chain_batch.sh work-mcp1k-cb0805 work-pr47f-b289a$A data 289559
done
# member-hash each vs work-pr46-m1konb/pr_evt289559 -> SAME at 0 and 179,
# DIFF at 35 and 60.

# Compiled-config proof (both directions; wcsonnet needs the runner's
# pipeline_names TLA or the tagger node is absent and the check is vacuous --
# the pr/46 G5 gotcha):
#   bare compile        -> TaggerCheckNeutrino data has cathode_wide_kink_angle: 25
#   -A ..._angle=null   -> key absent, compiled JSON byte-identical to the
#                          pre-change HEAD compile (verified via git worktree)
```

## 1. What is reconstructed: one segment sails straight through the true vertex

`work-mcp1k-cb0805/pr_evt52085/calib-pr-evt52085.json` (production PR-chain
dump): **9 segments, 16 vertices, 7 showers, 3 steiner clusters.** The
neutrino candidate is cluster 6, with exactly three segments:

| segment | pdg | n pts | length | endpoints |
|---|---|---|---|---|
| 6001 | 2212 (proton) | 228 | **137.6 cm** | (-18.45,99.48,108.75) -> (79.60,167.34,53.25) |
| 6002 | 211/13 | 4 | 2.0 cm | stub at (79.60,167.34,53.25) |
| 6003 | 13/11 | 5 | 2.2 cm | stub at (79.60,167.34,53.25) |

Segment 6001's 228 fit points run **monotonically** in x from -18.45 to
+79.60 cm (60 points at x<0, 168 at x>0); the largest consecutive step is
1.80 cm (the crossing itself: point 59 at x=-0.938 to point 60 at x=+0.591,
a 0.95 cm transverse jump too). There is **no segment boundary, vertex, or
kink anywhere in the interior** — the whole 137.6 cm object, cathode
included, is one fitted track.

Point 59, at (-0.938, 112.50, 81.34), is **0.06 cm from the stated true
vertex** — i.e. the fitted trajectory passes essentially through the true
interaction point. (Caveat: if the "true vertex" position in the owner's
question was itself read off this reconstruction, the 0.06 cm is a
tautology; the robust statement is that the fitted track passes within
~1 cm of the stated point, and that point is the cathode crossing.)

**Every one of the 16 graph vertices in this event sits at or near a track
END, not at this crossing.** Sorted by distance to the true point, the
nearest is 31.4 cm away (cluster 25, an unrelated 0.4 cm stub); the nearest
belonging to cluster 6 is 35.1 cm away, at the -x end of segment 6001. The
neutrino "vertex" the reconstruction reports is one of the two ends of this
one straight track — see §4 for which end, and why that is itself unstable.

## 2. The junction has a real, distortion-tolerant kink signature

The crossing point is not featureless. Two statistics, computed excluding a
skirt around the gap (so the ~1.25 cm physical hole and the ~1 cm
data-scale transverse misalignment near it — docs/14, docs/18 — cannot
contaminate the measurement):

- **Turn angle**: PCA direction of each arm, fit over a 15 cm baseline
  starting 3 cm from the crossing, angle between the two directions
  (oriented consistently along the direction of travel). **36.8 deg**,
  stable at 33-38 deg across skirt in {0,1.5,3,5} cm x baseline in
  {10,15,20,30} cm (max spread 5 deg across all 16 combinations).
- **dQ/dx asymmetry**: median dQ/dx of each arm over the 10 cm adjacent to
  the crossing (same 3 cm skirt), in units of the pattern-recognition MIP
  scale (43000 e/cm, `NeutrinoPatternBase.h:120`). **2.49 MIP on the 35.4 cm
  arm (proton-like) vs 1.31 MIP on the 102.2 cm arm** (still above 1 but far
  lower — consistent with a proton stub feeding a longer, more MIP-like
  continuation, exactly the topology the owner described).

This reads as a genuine two-prong vertex — a ~35 cm proton-like stub and a
~102 cm continuation — reconstructed as one "proton 560 MeV" track because
no vertex was ever placed at the junction.

## 3. Which gate blocks it: the accept-test thresholds, NOT the cathode veto

Two independent checks agree.

**(a) Disabling the veto entirely changes nothing for this event.**
`SBND_CATHODE_KINK_XCUT=0` reruns segment_search_kink
(`PRSegmentFunctions.cxx:193-420`, called only from `break_segments`,
`NeutrinoPatternBase.cxx:1617,1679`) with the cathode band's `continue`
(`:306-307`) never firing. The resulting `mabc-pr.zip` for 52085 is
**member-hash identical** to the production (veto-ON) arm — every one of
`0-clustering-global.json`, `0-mc.json`, `0-vertices-global.json`,
`0-shower_track-global.json`, `0-track_fit-global.json` matches. **The
veto was never the reason this event's kink went unfound.**

**(b) A break-time diagnostic shows the accept criteria simply never
qualify.** A temporary env-gated print (`WCT_PR47_DIAG`, reverted before
any commit — see Repro block) captured every index `segment_search_kink`
actually evaluates for this segment during `break_segments`, in internal
units (mm; divide by 10 for cm below). Selected rows across the crossing
(cathode band is |x|<5 cm; `cathode_skip=1` marks indices the shipped veto
removes):

| i | x (cm) | cathode_skip | refl | para | sum (C1/C3 stat) | sum1 (C2 stat) | ave dQ/dx (MIP) | max dQ/dx (MIP) |
|---|---|---|---|---|---|---|---|---|
| 53 | -1.96 | 1 | 12.08 | 33.61 | 12.21 | 12.21 | 2.38 | 2.67 |
| 55 | -1.43 | 1 | 22.51 | 51.46 | 17.91 | 17.91 | 1.90 | 2.58 |
| 56 | -0.93 | 1 | 18.56 | 52.37 | 18.83 | 18.83 | 1.53 | 2.67 |
| **57** | **0.84** | **1** | **23.16** | **48.96** | **18.79** | 18.79 | 1.23 | 2.33 |
| 58 | 1.46 | 1 | 17.76 | 49.71 | 16.62 | 16.62 | 1.01 | 1.25 |
| 65 | 5.05 | **0** | 6.48 | 57.15 | 4.47 | 4.47 | 1.85 | 2.84 |

(MIP columns here divide the printed internal-unit dQ/dx by the same
43000 e/cm scale converted to internal units, i.e. x4300; consistent with
§2's numbers.)

None of the four accept criteria (`PRSegmentFunctions.cxx:348-361`) fire
anywhere in this table, **on either side of the cathode band** (i=53/i=65
are both outside it and equally fail): `refl` tops out around 23 deg, never
reaching the 27-30 deg the C1/C3 tests require. The single closest approach
is **i=57, the C4 criterion**: `para`>15 (yes, 48.96), `refl`>22 (yes,
23.16), dQ/dx sub-conditions both satisfied (max 2.33 > 1.5, ave 1.23 > 1) —
**everything passes except `sum_angles`>19, which reads 18.79**. A miss of
0.21 on a global constant.

**Why the wide statistic and the narrow one disagree.** §2's turn angle
(36.8 deg, a single PCA fit over a fixed 15 cm baseline with the gap
skirted) is a stable, distortion-tolerant measurement of the same physical
kink that `refl_angle` (min over six index-offset window scales, 1.2-7.2 cm,
computed at every single fit index) is trying to detect. The min-over-scale
design is deliberately conservative — precisely because near the cathode
gap the point spacing and local direction are noisy (docs/14 measures
~0.35-1.4 cm transverse jitter there), any one of the six window scales
landing on a locally-straight-looking sub-stretch drags the whole index's
`refl_angle` down. That is a sound design *away* from the cathode; at the
cathode it throws away exactly the signal the wide statistic recovers.

**Conclusion: for 52085, the cathode-band veto is not the blocker at all —
the shipped kink test is simply too conservative at this junction, veto or
no veto.** Any option that only narrows or removes the cathode band (i.e.
touches `cathode_kink_xcut`) is a no-op for this event.

## 4. The main-vertex position is unstable, and predates the recent PR rounds

Independent of §3, the *reported* main vertex for 52085 has moved between
arms produced at different times, both on the SAME clustering/QL input
(`work-mcp1k-cb0805/ql_evt52085`, held fixed throughout):

| arm | main vertex | distance to truth |
|---|---|---|
| `work-mcp1k-cb0805` (2026-08-05 production) | (-18.45, 99.48, 108.75) | 35.1 cm |
| every arm at current HEAD (`work-pr46-m1konb`, `work-pr46-m1koffb`, `work-scan0807-set1`, `work-pr47-xcut0-case`) | (79.60, 167.34, 53.25) | 101.4 cm |

Both are simply the two ends of the same one-segment track (§1) — the
choice is a coin flip once no vertex exists at the real junction, so this
instability is itself evidence for §1's diagnosis, not a separate bug in
its own right.

**Attribution attempted, not completed.** The doc pr/46 knob is excluded
directly (its on/off `mc.json` for this event are byte-identical). Forcing
OFF, individually and then all together, every other graph-affecting knob
shipped in the pr/43-round-2 through pr/46 window —
`other_seg_empty_2d_guard`, `pseudo_shower_track_paint`,
`shower_long_muon_keep_type`, `pf_orphan_track_parentage`,
`single_muon_proton_chain_veto`, `single_muon_long_muon_claim`,
`pid_flag_reconcile`, `long_muon_stub_bridge` — **does not** move the
vertex back to the 2026-08-05 position; it stays at (79.60,167.34,53.25) in
every combination tried (see Repro block). **The flip predates the
pr/43-round-2 through pr/46 window.** Attributing it further would mean
bisecting the pr/33-through-pr/42 commit series (roughly ten more knob
rounds) landed between the cb0805 build and now — out of scope for this
doc. Flagged as an open item; not fixed, not chased further here.

## 5. Every other blocker a cathode vertex faces (code map, no runs)

Beyond `segment_search_kink`'s accept tests (§3), the active volume has a
**hole at |x| < 0.45 cm** (`params.jsonnet:24-25`, `cpa_left`/`cpa_right`
= +-0.45 cm), and several independent code paths treat "inside this hole"
as "this position cannot host a vertex":

| mechanism | file:line | effect |
|---|---|---|
| `MyFCN::UpdateInfo` | `MyFCN.cxx:316-320` | rejects any FITTED vertex position landing in the slab outright |
| `examine_structure_4` line walk | `NeutrinoStructureExaminer.cxx:603-611` | rejects any candidate branch whose straight line crosses the slab |
| Steiner-terminal vertex creation | `NeutrinoVertexFinder.cxx:193,258`; `NeutrinoStructureExaminer.cxx:544-545` | skips any Steiner terminal inside the slab as a vertex-activity candidate |
| FV score in main-vertex selection | `NeutrinoVertexFinder.cxx:1054-1063` (per-cluster), `:3388-3403` (global, softened by a main-cluster escape), `:3928-3933` (DL rerank) | withholds a +0.5 fiducial-volume score (four topology units) from any in-slab candidate |
| `examine_vertices_1p` | `NeutrinoStructureExaminer.cxx:1184-1188` | no-ops the whole pairwise vertex examination if either vertex is in the slab |
| `merge_nearby_vertices` | `NeutrinoPatternBase.cxx:1898-1923` | 0.1 cm merge threshold, vs a ~1.25-1.6 cm physical+distortion gap — cannot merge the two faces of a genuine crossing |
| DL/SCN vertex | `NeutrinoVertexFinder.cxx:3613,3706-3717` | can only SNAP to an existing graph vertex; cannot invent a position |

**For 52085 specifically, none of this matters**: the true junction sits at
|x| = 0.938 cm, already clear of the 0.45 cm slab. A vertex placed at the
nearest live fit point there (point 59, 0.06 cm from truth) would pass
every one of these tests as written. The slab-hole machinery is a real
constraint for options that need to place a vertex closer than 0.45 cm to
x=0, but it is not why 52085 has no vertex — that is purely §3's accept
thresholds. Relevant to nue `gap_identification` (`NeutrinoTaggerNuE.cxx`)
which has a standing watch item for exactly this failure mode
(pr/21 §5.7, n=0 observed so far).

## 6. Sample-wide statistics

### 6.1 Offline census (445/1000 events with a calib-pr dump; no new runs)

`scripts/analysis/pr47/cathode_junction_census.py`, applied to every segment
in `work-mcp1k-cb0805` with fit points on both sides of x=0. **Caveats
inherited from `kink_probe.py` (pr/20)**: this is the FINAL, post-fit point
cloud, not necessarily what `segment_search_kink` saw at break time (§3(b)
directly cross-checked 52085 itself and found the same qualitative
conclusion; the other 52 crossings were not individually cross-checked);
the accept-criteria replay does not model `flag_check`'s stateful walk
condition, so "criteria fire somewhere in the segment" is a superset of
"the real walk actually accepted this junction" — treat it as an upper
bound on how many crossings the shipped test *could* accept, not a
prediction of which ones it did.

- **53 cathode-crossing segments, 51 distinct events** (47 on the
  neutrino-candidate/main cluster).
- **Turn angle (skirt=3 cm, L=15 cm) distribution** (23/53 crossings have
  enough points on both sides for this measurement — short arms drop out):

  | bin (deg) | count |
  |---|---|
  | [0,2) | 4 |
  | [2,5) | 12 |
  | [5,10) | 5 |
  | [10,20) | 0 |
  | [20,30) | 0 |
  | **[30,45)** | **1** (52085, 36.8 deg) |
  | [45,90) | 0 |
  | [90,180) | 1 (349549, 107.8 deg — see below) |

  median 3.9 deg, p90 8.2 deg. **The distribution is bimodal**: a tight bulk
  at 0-10 deg (through-going tracks whose crossing is, correctly, not a
  kink) and exactly **two** outliers at >=20 deg. This bimodality is
  itself the case for a discriminator: a wide-baseline turn-angle test
  would touch only the tail, not the bulk.
- **The shipped accept criteria already fire on 6/53 crossings** (proxy,
  see caveats above): 286400, 315497, 349549, 353223, 409634 (twice, two
  segments). **Three of these five events are already independently
  documented elsewhere**: 286400 and 315497 are pr/20 Part VI movers (the
  veto's own measured effect), 409634 is pr/20's noted residual cathode
  stub (a break survives B0, built by a mechanism other than
  `segment_search_kink`). 349549 is documented within this doc itself
  (its own outlier below); 353223 (turn 5.0 deg, i.e. a modest, plausibly
  genuine kink) is not otherwise documented. The overlap with pr/20's own
  findings on 3/5 events, with no hand-tuning, is evidence the census
  methodology is sound.
- **The two >=20 deg outliers**: 349549 (turn 107.8 deg, dQ/dx 2.14/0.92
  MIP, already fires C1/C3, sits 4.6 cm from the event's main vertex — this
  crossing is essentially already recognized) and **52085** (turn 36.8 deg,
  does not fire, 35.0 cm from the reported main vertex — the owner's case).
  **Population estimate: genuine, currently-unrecognized cathode-junction
  kinks of 52085's kind are rare, roughly 1 per ~450 events in this
  sample** (52085 is the only wide-turn, non-firing crossing found).

### 6.2 Full-1000-event `SBND_CATHODE_KINK_XCUT=0` arm

`work-pr47-xcut0-1k` (`PR_JOBS=24`, all other knobs at production default,
same HEAD `03ccaaf3`, same `work-mcp1k-cb0805` ql/clustering input) vs
`work-pr46-m1konb` (current production baseline, same HEAD). Compared with
`hash_archive.py` member-content hashing (never raw zip bytes, M2).

- **Archive-level: 11/1000 events differ** — `172794, 286400, 287654,
  289559, 315497, 349549, 353751, 386948, 395060, 409634, 410008`. **52085
  itself is NOT in this list** — confirming §3(a) at full-sample scale, not
  just for the single-event check.
- Every mover's diff is confined to `0-mc.json`, `0-shower_track-global.json`,
  `0-track_fit-global.json`, `0-vertices-global.json` — clustering and
  imaging layers (`0-clustering-global.json`, dead-area layers) are
  untouched in all 1000 events, as expected (the veto only ever affects
  `segment_search_kink`'s accept step).
- **Every mover gains or loses at least one vertex inside the cathode band
  (|x|<5 cm)** — e.g. 172794: 8->9 vertices, new band vertices at x=-4.49
  and -4.65 cm; 315497: 18->20, new band vertices at x=1.74 and -2.26 cm.
  This is the veto doing exactly what it is designed to do, in both
  directions (some events gain a spurious-looking near-band vertex when the
  veto is off, consistent with pr/12 §7's population).
- **Five of the eleven movers are already independently documented**:
  172794 is pr/20 Part VI's own worked example of the reorder mechanism (an
  invented break 21 cm from the cathode when the veto is removed); 286400
  and 315497 are pr/20 Part VI's neutrino-vertex-relocation movers; 349549
  and 409634 are §6.1's own cross-checks. The other six (287654, 289559,
  353751, 386948, 395060, 410008) are new to this doc — not investigated
  further here, flagged as available for a future census if the veto's
  behavior needs closer study.
- **nusel verdict: 0/1000 diffs**, at both the event-level table
  (`nusel-events.tsv`: `event_label` etc.) and the per-bundle table
  (`nusel-table.tsv`: `label`, `tgm`, `stm`, `fc`, `stmfit`, `lm`, all
  other columns) — every one of the 11 archive-level movers still lands on
  the identical selection verdict. **Fully disabling the cathode-band veto
  moves PR-graph vertex structure in 11/1000 events and changes zero
  selection outcomes.** This is the practical floor for "how much does
  touching this veto cost" — any of §7's options, which touch far less than
  the whole veto, should cost no more than this.
- Precision/recall against §7's proposed O1 discriminator: of the 11
  movers, only 349549 and 409634 already appear in §6.1's turn-angle/dQ/dx
  census (both already recognized by the LEGACY criteria at some index —
  409634's own residual is pr/20's documented case of a break surviving via
  a DIFFERENT mechanism than `segment_search_kink`, so it is not actually
  caused by removing the veto's suppression of this test). The other 9
  movers' segments were not all captured by §6.1's calib-pr-dump coverage
  (445/1000) — a full precision/recall table would need the census rerun
  against this arm's own dumps, not attempted here (scope: this doc reports
  a discriminator design and its cost floor, not a fully quantified
  precision/recall — flagged for the fix-session round).

## 7. Options (mechanism, insertion point, measured footprint, risk)

*(Written in the analysis round, kept verbatim.  **O1 was subsequently
implemented — see §8.**  O2-O6 remain unimplemented.)*  Ranked by how
directly each targets §3's actual finding (the accept-test thresholds, not
the band).

**O1 — change the test, not the band (recommended starting point).** Add a
cathode-band-scoped *additional* accept path inside `segment_search_kink`
(`PRSegmentFunctions.cxx`, alongside the four existing criteria at
`:348-361`), keyed on the §2/§6.1 statistics: the skirt-excluded
long-baseline turn angle and the dQ/dx asymmetry between the two arms,
computed once per candidate junction rather than per fit index. Leave the
four legacy criteria and the existing `cathode_kink_xcut` band completely
untouched — this is an OR, not a replacement.

Explicitly **against** the two changes that look cheaper:
- *Narrowing `cathode_kink_xcut`* (5 cm -> e.g. 1.5 cm) does not help
  (§3(a): the veto already changes nothing for 52085) and actively risks
  regressing pr/20's own result — pr/12 §7 found the 13-of-44 spurious
  cathode vertices concentrated *closest* to the gap, exactly the region a
  narrower band re-admits.
- *Nudging the `sum_angles > 19` constant* (e.g. to 18.5) would flip 52085
  (miss margin 0.21) but is a global constant tuned on one event with no
  general justification — precisely what the owner asked NOT to do.

Bimodality (§6.1) argues this is safe in principle: the wide-baseline
turn-angle statistic separates a tight 0-10 deg bulk from a two-event tail
at >=20 deg, with a wide gap between them (10-30 deg empty in this sample)
to set a threshold in. §6.2 gives the measured cost of doing nothing but
widening the band (i.e. `cathode_kink_xcut=0`) as a sanity floor.

**O2 — "B0-stop" (never measured, pr/20 Part VI §6).** Independent of O1:
when a cathode-band candidate is suppressed, return no kink at that index
instead of letting the scan continue to the next qualifying index. Today,
suppressing an index just lets the FIRST *other* qualifying index win,
which is not always a no-op (pr/20 Part VI: evt 172794 gained an unrelated
break 21 cm from the cathode). This would make the veto strictly more
conservative (ON accepts subset of OFF accepts) and is a reasonable
companion to O1, but is orthogonal — it doesn't help 52085 by itself,
since 52085's problem is that nothing ever fires, cathode or not.

**O3 — a dedicated post-`break_segments` cathode-junction pass.** Run the
O1-style two-variable test once per cathode-crossing segment, after the
normal `break_segments` walk has run and found nothing, and create the
vertex directly at the nearest LIVE fit point (|x| >= 0.45 cm, i.e.
outside the slab described in §5) rather than trying to make
`segment_search_kink`'s stateful index walk reach the right spot. This
sidesteps every one of §5's slab-hole blockers by construction, at the
cost of being a second code path to reason about rather than one extra
disjunct in an existing one. For 52085, point 59 (0.938 cm from x=0, 0.06
cm from truth) already satisfies every §5 test as written.

**O4 — decide at clustering time.** Don't let `cathode_connect`
(`clus.jsonnet:490`) join the two halves of a crossing when the O1-style
junction test says "this looks like a vertex, not a through-track". This
is the cheapest option to reason about in isolation, but note the tension
honestly: pr/20's A1/A2 (`tip_touch_cut=4cm`, `crosser_pca_angle=20`) were
built and validated in the OPPOSITE direction — to join MORE crossings, not
fewer — because most crossings genuinely are one track. A clustering-time
veto would need its own false-positive census against that population
before it could ship.

**O5 — the FV/slab side.** A cathode-band exemption for the +0.5 FV score
(§5) and the in-slab hard rejects, needed only if a future case's junction
lands INSIDE |x|<0.45 cm (52085's does not). Secondary to O1/O3 for this
doc; cross-references the standing nue `gap_identification` watch item
(pr/21 §5.7, n=0 observed).

**O6 — calibration (recorded, not proposed).** Fix the underlying
transverse misalignment (docs/14, docs/18) rather than working around it.
pr/20 already judged this: the kink-test margins involved (there, 30.8 vs
30, 27.4 vs 27) are too thin to fix by recalibration alone, and it would do
nothing for pr/20's class A (halves that never join at all). Not proposed.

**Explicitly rejected**: fabricating charge or interpolated dQ/dx inside
the 0.45 cm slab (pr/21's standing rule — the ionization there is
genuinely lost in all three planes, not recoverable); any change to the
`cathode_kink_xcut` band's width (§3(a) makes it moot for this case, and
O1 above gives the actual reason it's moot).

**Blocked on evidence that does not currently exist**: whether O1's
discriminator, tuned on this sample's 2-event tail, generalizes to data
(where the transverse distortion is ~3x larger, docs/14 §Result) — would
need a data cathode-crosser census, which does not exist yet (docs/18 is
the closest, MC-baseline pipeline only).

## 8. Fix (this round): `cathode_wide_kink_angle` — O1 implemented

**Owner's framing for this round (2026-08-07):** the broad angle change is
large but the local small-window angle change is not, presumably because of
the cathode gap; if everything were connected, the local deflection would
very likely be big too. So fix the *measurement* at the cathode — key the
accept on the (wide-baseline) angle deflection — which (1) fixes these
events and (2) leaves every non-cathode case alone, since the local angle
deflection machinery is a major algorithm of the whole PR chain.

### 8.1 Design

A **fifth accept path** inside `segment_search_kink`, active only at
cathode-crossing fit indices, keyed on §2's statistic:

- New free helper `segment_cathode_wide_kink_accepts(fits, cathode_x,
  angle_cut_deg, skirt, baseline)` (`clus/src/PRSegmentFunctions.cxx`,
  declared in `PRSegmentFunctions.h` so it is doctest-able). For every
  sign-change crossing of consecutive fit points across `x = cathode_x`
  (both directions — note §6.1's census only measured neg→pos; the helper
  handles both): collect each arm's points with arclength from the crossing
  in `[skirt, skirt+baseline]` on that arm's own side (≥ 3 points per arm or
  the crossing never fires), PCA each arm (`Facade::calc_pca_dir`, centroid
  passed in, axis re-oriented along its own arm's chord = direction of
  travel), and take the angle between the two directions: ~0 deg for a
  straight through-going track. A pure transverse cathode offset *translates*
  one arm but does not rotate a PCA axis, so the statistic is
  distortion-tolerant by construction — the property that makes it safer on
  data (~3x larger offset) than any local-window statistic. Same arithmetic
  as the census script's `skirt_turn_angle` (post-orientation-fix); no
  prototype counterpart (new algorithm, not a port).
- If turn ≥ `angle_cut_deg`: accept at the crossing-adjacent index with the
  larger |x − cathode_x|, stepped outward up to 2 indices to clear the
  |x| < 0.5 cm slab margin (§5's in-slab blockers), clamped to the caller
  contract `0 < save_i < fits.size()-1`.
- In the scan loop the accept sits inside the `flag_check` block,
  **immediately before the pr/20 cathode-veto `continue`** — so it works
  whether or not the veto is on, inherits the legacy 1-cm end guards and the
  `flag_check` latch, and the accepted index flows through the identical
  post-accept machinery (direction averaging, straightness/`flag_switch`,
  local dQ/dx) as the four legacy criteria. Everything is gated on
  `cathode_wide_kink_angle > 0`: knob off ⇒ no precompute, no lookup, no
  arithmetic touched.

**Threshold 25 deg, angle-only.** Any value inside §6.1's measured empty gap
(10-36 deg) separates the two-event tail from the 0-10 deg bulk; 25 leaves
~12 deg margin below 52085 (36.8) and ~17 deg above the bulk p90 (8.2). No
dQ/dx-asymmetry term: the distribution separates on angle alone, and an
asymmetry requirement would reject genuine same-species kinks (a scattered
muon). All three parameters are config knobs if retuning is ever needed.

### 8.2 Knobs and threading (the `cathode_kink_xcut` value-knob pattern)

| knob | C++ default | SBND production |
|---|---|---|
| `cathode_wide_kink_angle` (deg) | **0 = OFF** | **25** |
| `cathode_wide_kink_skirt` (cm) | 3 | (default) |
| `cathode_wide_kink_baseline` (cm) | 15 | (default) |

`NeutrinoPatternBase.h` members (internal units) → `TaggerCheckNeutrino.{h,cxx}`
(configure + `default_configuration()` echo + cm→internal forward) →
`segment_search_kink` trailing params (both `break_segments` call sites,
`NeutrinoPatternBase.cxx:1617/:1680`) → `cfg/pgrapher/common/clus.jsonnet`
signature + null-suppression → `cfg/pgrapher/experiment/sbnd/clus.jsonnet`
(4 sites, ON value beside `cathode_kink_xcut=5`) →
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` (TLA default 25 +
forward) → runner `run_pr_chain_batch.sh` env
`SBND_CATHODE_WIDE_KINK_ANGLE` (empty = ON, `0` = force off, `null` = omit
key). Default pins in `clus/test/doctest_clus_knob_defaults.cxx`; five new
synthetic-track doctests in `clus/test/doctest_prsegment.cxx` (straight
never fires; 35-deg kink fires at 25 not 40 with the accept index clear of
the slab; a 1.5 cm pure transverse offset does NOT fake a kink — the
distortion-tolerance claim, tested; bends inside the skirt are invisible;
short arm cannot fire; angle 0 disables).

### 8.3 Gates

- **Unit**: `./build/clus/wcdoctest-clus` — 106/106 cases, 1080 assertions
  (5 new pr47 cases, 11 assertions).
- **Compiled-config**: bare compile carries `cathode_wide_kink_angle: 25` in
  the `pr` TaggerCheckNeutrino node; `-A cathode_wide_kink_angle=null`
  removes the key and the compiled JSON is **byte-identical to the
  pre-change HEAD compile** (git-worktree cross-compile).
- **Knob-off output gate**: `work-pr47f-base48` (clean `03ccaaf3` binary,
  production env) vs `work-pr47f-off48` (new binary,
  `SBND_CATHODE_WIDE_KINK_ANGLE=0`): **48/48 events, mabc-pr.zip + pctree
  member-hash identical (`hash_archive.py`), merged nusel tsvs
  byte-identical.**
- **Knob-on smoke** (`work-pr47f-case`, bare production run of 52085):
  a new vertex at **(-0.56, 112.68, 80.77) = 0.65 cm from truth**
  (baseline: nearest vertex 31.4 cm). The 137.6 cm "proton 560 MeV" track
  becomes a genuine two-prong at the junction — **proton 243 MeV (the
  35 cm arm, 2.49 MIP) + mu- 253 MeV (the 102 cm arm, 1.31 MIP)** — and the
  main vertex moves from the far +x track end (101.4 cm from truth) to the
  junction. This also collapses §4's end-to-end degeneracy for this event:
  once the true junction vertex exists, the main-vertex choice is no longer
  a coin flip between two endpoint candidates.

### 8.4 Full-1000-event footprint (`work-pr47f-on1k` vs `work-pr46-m1konb`)

Full 1000-event knob-on arm at the fix HEAD (`PR_JOBS=24`, run ok 1000 /
failed 0) vs the production baseline `work-pr46-m1konb` (same clean-source
`03ccaaf3` lineage, knob absent), member-hash per `hash_archive.py`:

- **4/1000 archive-level movers** — every one confined to `mabc-pr.zip`'s
  vertex/fit/PF layers (`0-mc.json`, `0-shower_track-global.json`,
  `0-track_fit-global.json`, `0-vertices-global.json`); pctree and all other
  members identical everywhere.
- **0/1000 nusel diffs at BOTH granularities** (`nusel-events.tsv` and the
  bundle-level `nusel-table.tsv`): the fix changes no selection outcome.
- Mover-by-mover, against §6.1's census + a both-direction turn recompute
  (the census's `skirt_turn_angle` only measured neg→pos crossings; 30 of
  the 53 were unmeasured `nan`):
  - **52085** — the target. New vertex at x = −0.56 (band), 0.65 cm from
    truth; measured turn 36.8 deg. See §8.3.
  - **349549** — turn 107.8 deg (census). Already fired legacy C1/C3
    4.6 cm from its main vertex; the wide accept now also fires AT the
    crossing (new band vertex x = 1.84), reorganizing 68→67 vertices.
  - **409634** — TWO steep crossings, turn 62.1 and 96.7 deg
    (both-direction recompute; census had `nan` — they are pos→neg). New
    band vertex x = 1.48. This is pr/20's evt with the residual stub.
  - **289559** — final vertex GEOMETRY byte-identical (all 11 positions
    equal at full precision); only segment-id renumbering
    (`real_cluster_id` 14002→14003 …) plus track_fit point jitter — the
    signature of an extra break-and-refit that converges to the same
    geometry. Threshold bracketing (single-event reruns
    `work-pr47f-b289a{0,35,60,179}`): byte-identical to baseline at
    angle=0 and angle=179, moves at 35 and at 60 ⇒ the break-time cloud
    carries a genuine **≥ 60 deg** wide-baseline crossing kink that the
    census's post-fit proxy cloud cannot see (its final-cloud second arm is
    a 4.8 cm remnant, too short for the offline test). Zero net effect on
    vertices, PF content (same particles/energies, tree order only) and
    nusel.
- Predicted-but-unmoved: 315497 (27.4 deg on the proxy cloud) and 406796
  (20.2 deg, below the 25 cut) — consistent with the proxy-cloud caveat
  (§6.1: `flag_check` walk state and break-time vs post-fit clouds).
- Cost comparison: the pr/20 veto-off experiment (§6.2) moved 11/1000;
  this accept path moves 4/1000, strictly fewer, all four explained by
  measured ≥ 25 deg crossings, with zero selection-verdict changes — the
  "does it break through-going tracks" answer is **no**: no mover is a
  through-going track split (the bulk of the crossing population sits at
  0-10 deg, far below the 25 deg cut).

## Caveats

- Sample is MC. Data cathode-region transverse distortion runs ~1.2-1.4 cm
  (docs/14, docs/18) vs this sample's MC ~0.35-0.48 cm — every separation
  measured in §2/§6.1 is optimistic relative to data.
- §6.1's calib-pr coverage is 445/1000 events; the remaining 555 were
  produced by an earlier sweep that did not request this dump format. No
  reason to expect a coverage-related bias, but it is not proven.
- §6.1's accept-criteria replay is a proxy (does not model `flag_check`'s
  walk state) — see the caveat block at the head of §6.1 and the identical
  caveat in `kink_probe.py` (pr/20, evt 169824).
- §4's vertex-flip attribution is incomplete by design (scoped to the
  pr/43-round-2 through pr/46 window; not chased further).

## Related

- `docs/pr/12_cathode-crossing-neutrino-pr.md` — the crossing survives as
  one track 44/45 times; §7's 13-of-44 spurious cathode vertices (this doc's
  §3(a) shows the OPPOSITE failure mode: a vertex that should exist but
  doesn't); §7a: genuine neutrino vertices land near the cathode at chance
  rate (7 observed vs 5.5 expected), no attraction, no repulsion.
- `docs/pr/20_split-cosmics-cathode-and-demoted-main.md` Part II — B0
  (`cathode_kink_xcut`, SBND ON `fe6b7d90`) and its measured effect (21/1000
  movers, reorders rather than only removes); the never-measured B0-stop
  variant this doc's O2 revives; 409634's residual stub (this doc's §6.1
  independently reproduces it).
- `docs/pr/21_cathode-dead-gap-dqdx-notch-impact.md` — why the dQ/dx notch
  looks smaller than expected; the standing nue `gap_identification` watch
  item (this doc's §5); the "compensate where recoverable, guard where not,
  never repaint" rule this doc's rejected option follows.
- `docs/14_cathode-crossing-diagnostic.md`, `docs/18_cathode-distortion-map.md`
  — the transverse-offset measurements quoted throughout.
- `sbnd_xin/scripts/analysis/cathode/kink_probe.py` — the same
  segment_search_kink-replay technique, applied to evt 169824 in pr/20; this
  doc's census script generalizes it to a sample-wide sweep.
- [[project_pr46_long_muon_stub_bridge]] — the previous round, whose
  `work-pr46-m1konb`/`m1koffb` arms are the current-HEAD baselines used in
  §4's bisection.

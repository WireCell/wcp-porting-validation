# doc pr/90 — unbroken kink, wrong neutrino vertex (mcp1k 320865, 172832, 61681)

**Status: §§0–7 are the original investigation (unchanged below); §8 is
round 2, the owner-requested implementation + validation of the §6
proposals — COMPLETE.** Outcome: both knobs implemented default-OFF
(`teb_turn_min_arm_frac`, `teb_second_max`); knobs-off path gate-proven
byte-identical on all 1067 events (mcp1k 1000 + nueCC48 48 + NCpi0 19) on
the shipping binary; `teb_turn_min_arm_frac = 0.4` (v2 two-tier preference
semantics, §8.7) validated live — the single mover in 1067 events is the
target 320865, toward, 37.3 → 1.2 cm, zero ADVERSE — and is **SBND
PRODUCTION ON** (§8.9). `teb_second_max` confirmed §3a's mechanism but
measured NEGATIVE on its own motivating events (§8.5) and **stays OFF**;
172832/61681 are unchanged in production (§8.10). A first-round hard-filter
semantics for knob 1 FAILED live adjudication (5 ADVERSE, §8.6) and was
replaced — the v1/v2 arms are both retained. **§9 is round 3** (owner
physics input + instrumented profiles): the residual mechanisms are now
fully resolved — 172832's junction is vertex-activity + a 23.5° local turn
(invisible to the dip/wide-turn routes), 61681's junction floats 4.4 cm
past the true vertex on a charge-less fit bridge, and the keep/kill
discriminator for near-end breaks is Bragg-consistency (hot-extent vs
peak) — with four designed fixes (D1–D4, §9.5) for the next session.

## 0. Repro

Bee link built this session (all 8 events requested that day; this doc covers
indices 0/1/2):

```
https://www.phy.bnl.gov/twister/bee/set/cd1f81ad-383d-4757-b2d7-4c4c18c05fa4/event/list/
```

built with (from `sbnd_xin/`):

```bash
python3 scripts/bee/make_pr_bee.py \
  -q work-mcp1k-cb0805 -q work-nuecc48-cb0805 -q work-mcp2k-cb0816 \
  -p work-mcp1k-pr89base -p work-nuecc48-pr89base -p work-mcp2k-pr89base \
  -o /home/xqian/tmp/bee_request/user-8evt.zip \
  320865 172832 61681 138009 54629 279955 405707 70084
./upload-to-bee.sh /home/xqian/tmp/bee_request/user-8evt.zip
```

All geometry/topology tables below come from files already on disk under
`work-mcp1k-pr89base/pr_evt<ID>/` — `mabc-pr.zip` (Bee point-cloud layers) and
`calib-pr-evt<ID>.json` (`PrDisplayDump`, segment/vertex graph). Nothing under
`work*/` was regenerated, written, or deleted for this study. Representative
extraction (repeat per event, `rc_t` = the long unbroken segment's
`real_cluster_id`, from `0-track_fit-global.json`):

```python
import zipfile, json, os, math
def load(p):
    z = zipfile.ZipFile(p); out = {}
    for n in z.namelist():
        b = os.path.basename(n)
        if b.startswith('0-') and b.endswith('.json'):
            out[b[2:-5]] = json.loads(z.read(n))
    return out

d = load('work-mcp1k-pr89base/pr_evt320865/mabc-pr.zip')
tf, vx = d['track_fit-global'], d['vertices-global']
nu = [(x,y,z) for x,y,z,q in zip(vx['x'],vx['y'],vx['z'],vx['q']) if q >= 15000][0]
P = [(x,y,z) for x,y,z,rc in zip(tf['x'],tf['y'],tf['z'],tf['real_cluster_id'])
     if rc == 13002]
# cumulative arclength, then a 10cm-baseline direction-change scan over P
# gives the turn profile; see full script in the harv3-delta scan family
# (pr86_kink_census.py-style) for the wide-baseline (skirt=3,baseline=35cm)
# PCA version used in §4/§5.
```

Hand-scan truth used throughout is
`vertex_labels/vtxscan-{harv3-delta,prod0813-mcp1k}/labels-evt<ID>.json` (all
six labels: `not_a_candidate: true`, manual pick at `cluster_id: -1` — the
scanner rejected every reco candidate). Kink census cross-reference:
`docs/pr/86_r2-kink-after.json` (different arm, `work-mcp1k-pr87ion3`, same
three events, corroborating turn magnitudes independently).

### 0.1 Instrumentation used to resolve the 320865 mechanism (§3b)

The first round of this investigation (below) could not explain, from static
JSON dumps alone, why `segment_two_end_break_scan`'s turn route picked fit
index 8 for the break on 320865 instead of an index near the independently
measured true corner. That called for actually looking at the live
`seg->fits()` array and the live `segment_wide_turn_angle` scan, not more
inference from `calib-pr` JSON. Toolkit repo (`/nfs/data/1/xqian/toolkit-dev/toolkit`,
branch `apply-pointcloud`):

```diff
--- a/clus/src/PRSegmentFunctions.cxx
+++ b/clus/src/PRSegmentFunctions.cxx
@@ -10,6 +10,7 @@
 #include <chrono>
 #include <cmath>
 #include <cstdlib>
+#include <fstream>
 #include <list>
 #include <numeric>
 #include <set>
@@ -593,6 +594,27 @@ namespace WireCell::Clus::PR {
         res.idx_dip = k_dip;
         res.idx_turn = k_turn;
         res.turn_deg = turn_max;
+
+        // TEMPORARY diagnostic dump for doc sbnd_xin/docs/pr/90 (evt320865
+        // k_turn mystery).  Off unless WCT_TEB_DUMP is set to an output path;
+        // byte-identical to legacy when unset.  To be reverted after use.
+        if (const char* dumpenv = std::getenv("WCT_TEB_DUMP")) {
+            std::ofstream ofs(dumpenv, std::ios::app);
+            ofs << "# SEG N=" << N << " L=" << L/units::cm
+                << "cm k_dip=" << k_dip << " k_turn=" << k_turn
+                << " turn_max=" << turn_max << "\n";
+            for (size_t k = 0; k < N; k++) {
+                const double t = (opt.turn_angle > 0)
+                    ? segment_wide_turn_angle(fits, k, opt.turn_skirt, opt.turn_baseline)
+                    : 0.0;
+                ofs << k << " " << cum[k]/units::cm << " "
+                    << fits[k].point.x()/units::cm << " "
+                    << fits[k].point.y()/units::cm << " "
+                    << fits[k].point.z()/units::cm << " "
+                    << dqdx[k] << " " << t << " " << (arm_ok(k)?1:0) << "\n";
+            }
+        }
+
         if (k_dip < 0 && k_turn < 0) return res;
```

Gated purely on an unset-by-default `getenv`, matching the existing
`WCT_DET_DEBUG` convention elsewhere in `clus/`; byte-identical to legacy
when the env var is unset, so this is not a behavior change. Build + freshness
proof (M1):

```bash
cd toolkit && wcbuild
ls -la --time-style=full-iso clus/src/PRSegmentFunctions.cxx ../local/lib/libWireCellClus.so
# lib mtime 2026-08-17 06:10:29 > source edit mtime 2026-08-17 06:09:54 -- fresh
```

Single-event rerun into a fresh out_root, `reality=data` (M9/M13-adjacent —
new scratch dir, nothing existing touched):

```bash
cd sbnd_xin
rm -f /home/xqian/tmp/teb_dump_320865.txt
WCT_TEB_DUMP=/home/xqian/tmp/teb_dump_320865.txt PR_JOBS=1 SBND_WCT_LOGLEVEL=debug \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-mcp1k-kink90 data 320865
```

After capturing `/home/xqian/tmp/teb_dump_320865.txt` (§3b), the diff was
reverted (`git checkout -- clus/src/PRSegmentFunctions.cxx`) and `wcbuild`
rerun to restore the clean production `libWireCellClus.so` — the toolkit
working tree carries no trace of this round. The scratch out_root
`work-mcp1k-kink90/` (2.2 MB, untracked, same convention as every other
`work-*` arm) is left in place for reproducibility.

---

## 1. Symptom

Owner's framing: each of these three events shows a track with a clear turn
that pattern recognition keeps as one segment, so the neutrino vertex — which
should be at the break — is wrong.

That reading is correct and the geometry confirms it precisely:

| evt | offending segment (`real_cluster_id`) | length | max local turn (10cm baseline) | turn location |
|---|---|---|---|---|
| 320865 | 13002 | 193.5 cm | 27.9° | s=42.6 cm (22% along) |
| 172832 | 21001 | 127.2 cm | 23.3° | s=105.6 cm (83% along) |
| 61681  | 2001  | 109.5 cm | 28.0° (up to 53° at a wider baseline) | s=101.4 cm (93% along) |

All three turns are 4–8× the local baseline elsewhere on the same segment
(2–6°), so this is a real, isolated corner in the *fitted* trajectory — not
the pr/73 failure mode where the fit smooths a corner the image shows. All
three segments are pure track (`shower_track-global` q=0 for every point of
the offending segment); no other segment or graph vertex attaches at the
turn (nearest other segment endpoint is 6–24 cm away and itself a
0.3–11 cm stub); dQ/dx steps across the turn in all three (e.g. 61681:
~1700 → 3300 xMIP-scale ADC/cm), consistent with two different particles
meeting at a vertex that clustering strung into one trajectory.

## 2. The headline number: the turn point IS the truth vertex

The independent hand-scan truth (a human manually placed the vertex, having
rejected every reco candidate) sits almost exactly on the geometric turn
point measured from the fit — while the reco main vertex, parked at a free
end of the same track, is 4–39 cm away:

| evt | dist(turn point, hand-scan truth) | dist(reco main vertex, truth) |
|---|---|---|
| 320865 | **1.2–1.6 cm** (two truth labels, harv3-delta / prod0813 arms) | 39.0 cm |
| 172832 | **0.0–0.6 cm** | 20.4 cm |
| 61681  | **1.7–2.0 cm** | 4.4 cm |

This is the strongest evidence in the investigation: it makes the
segmentation reading (not a vertex-*refinement* miss — see pr/89's rounds 4/5
conclusion that DL-vertex refinement lands 1.0–1.5 cm off at the acceptance
boundary) unarguable for 320865 and 172832. 61681 is the borderline case —
see §5.

## 3. Why the graph vertex is not at the turn: three different mechanisms

This is not one bug. Log evidence (`grep -a "break_two_end_dqdx:\|TaggerCheckNeutrino: selected"
work-mcp1k-pr89base/pr_evt<ID>/wct_pr_evt<ID>.log`) splits the three events into
two distinct failure classes.

### 3a. 172832 and 61681 — `break_two_end_dqdx` never runs

`break_two_end_dqdx` (`clus/src/NeutrinoPatternBase.cxx:2965`, scan
implementation `segment_two_end_break_scan` at `clus/src/PRSegmentFunctions.cxx:440`,
knob `two_end_break`, **SBND production ON** via
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet:1340`) is the pass that
exists precisely for "one long track, no branch, vertex should split it
mid-segment" (doc pr/48). Its entry gate at `NeutrinoPatternBase.cxx:2988`:

```cpp
if (n_long != 1 || !cand) return false;   // exactly ONE segment > teb_stub_max (4cm)
```

Both logs have **zero** `break_two_end_dqdx:` lines. Both events' final main
cluster already has **two** segments longer than the 4 cm stub floor:

| evt | main cluster segments (length) |
|---|---|
| 172832 | 21001 = 127.2 cm, 21021 = 13.1 cm |
| 61681  | 2001 = 109.5 cm, 2002 = 11.2 cm |

`n_long = 2` → the gate declines before any angle or dQ/dx is examined. The
turn sits inside the 127 cm / 109 cm long prong, which this pass never looks
at once a second long-ish prong exists anywhere else in the cluster.

**Caveat:** this is inferred from the *final* segment inventory and the
absence of a log line; the gate is evaluated earlier, inside
`find_proto_vertex`, so the pre-gate topology at that moment is not directly
observed here. Unlike §3b, this was not instrumented this round — a
`WCT_DET_DEBUG=2` single-event rerun (same method as §0.1, applied to
172832/61681 instead) would confirm the pre-gate topology directly and is
the natural next step before implementing the §6 gate-widening proposal.

### 3b. 320865 — the break DOES fire, and instrumentation now shows exactly why it lands at the wrong point

Unlike the other two, the log shows the pass firing and succeeding:

```
break_two_end_dqdx: cluster 13 seg len 198.6cm k*=8 (dip 67 turn 8) arms 5.3/193.3cm
  J=0.221 s15=(0.187f,0.034F) rise=(2.30,3.54) absmed=(2.30,3.54)xMIP
  turn=38.1deg routes=(false,true) found=true
  cand idx=8 m3=2.23xMIP sA=0.187f sB=0.034F accepted=true
  BROKE cluster 13 at fit idx 8 (42.77,-18.63,95.35)cm route 2
```

Route 2 (the wide-baseline turn route, `teb_turn_angle=25°`) won over route 1
(the dQ/dx-dip route, candidate at index 67 — much closer to the independently
measured true corner). The break landed ~5 cm from the segment front — 2.4 cm
from the current (wrong) reco main vertex — not near the true corner ~40–53 cm
in.

**First-round attempt (superseded below).** Reconstructing the pre-break fit
array by concatenating the two post-break segments' points from
`calib-pr-evt320865.json` did not reproduce index 8 as the turn-angle argmax,
so that round of the doc left the mechanism unresolved rather than guess. The
`calib-pr` JSON's `segments[].points` turned out not to be a reliable stand-in
for the live `seg->fits()` array (§0.1) — the fix was to look at the real
array directly.

**Resolved via instrumentation (§0.1).** The dump of the live `fits()` array
and the live `segment_wide_turn_angle(k)` value at every index (production
`teb_turn_skirt=3cm`, `teb_turn_baseline=35cm`) shows the mechanism directly.
Two local maxima compete for the `k_turn` argmax — a genuine one near the
independently-measured true corner, and a spurious one right at the front of
the segment, at fit index 8, that happens to score *higher*:

| idx | s (cm) | turn (deg) | arm A: n pts, span (cm) | arm B: n pts, span (cm) |
|---|---|---|---|---|
| 7  | 4.73  | 36.31 | 3, **1.46** | 56, 34.22 |
| **8**  | **5.34**  | **38.06** | **4, 1.94** | **57, 34.76** |
| 9  | 5.96  | 33.97 | 5, 2.49 | 57, 34.74 |
| 84 | 52.35 | 33.10 | 57, 34.22 | 55, 34.68 |
| 85 | 52.89 | 33.04 | 57, 34.19 | 54, 34.00 |
| 86 | 53.45 | 33.02 | 57, 34.30 | 55, 34.16 |
| 87 | 54.04 | 33.01 | 56, 33.79 | 55, 34.16 |
| 88 | 54.64 | 32.95 | 56, 34.11 | 56, 34.70 |

At the true corner (idx 84–88, s≈52–55 cm — matching the independently
measured turn region and the hand-scan truth vertex to within a few cm) both
PCA arms are well-formed: ~56 points spanning ~34 cm, close to the requested
35 cm baseline, and they agree stably at ~33°.

At the winning index 8, arm A is **degenerate**: only 4 points spanning
**1.94 cm** — because the segment simply doesn't have 35 cm of track in front
of index 8 (the front of the segment is only 5.3 cm from index 8), so
`segment_wide_turn_angle`'s window-collection loop (`PRSegmentFunctions.cxx:335-376`)
silently accepts whatever points fall in the truncated `[skirt, skirt+baseline]`
range rather than requiring the window be full. A 4-point PCA direction over
1.94 cm of track is dominated by local fit wiggle, not by the track's true
heading, and this particular wiggle happens to read 38.06° against the (well-
formed) arm B — 5° *higher* than the genuine 33° corner 47 cm downstream. The
argmax search (`k_turn`, a plain `if (t > turn_max)` scan with no arm-quality
weighting, `PRSegmentFunctions.cxx:583-591`) has no way to prefer the
well-supported candidate over the noisy one, and the noisy one wins by
2–5° of accumulated PCA jitter.

This confirms the general failure mode identified independently in §4's
407-event census (spurious near-end PCA turns) as the *specific, live*
mechanism for 320865, with exact numbers rather than inference.

### 3c. Ruled out: `vertex_kink_snap` reach

`vertex_kink_snap` (pr/50, SBND-ON) exists to move the main vertex onto a
nearby true kink, but its search window (`m_vks_radius = 5 cm`,
`NeutrinoPatternBase.h:402`) is far short of the 6.2–40.3 cm distance between
the current (wrong) main vertex and the true turn in all three events. It is
correctly out of reach here — not a contributing or fixable lever for this
investigation.

## 4. Negative result: a bare wide-turn-angle threshold is not shippable

The natural first idea — flag any segment with a large
`segment_wide_turn_angle` (the helper already used by route R2 above,
`clus/src/PRSegmentFunctions.cxx:335`) — was tested against a 407-event
census of the `work-mcp1k-pr89base` arm (segments ≥ 20 cm, turn evaluated
≥ 8 cm from both segment ends):

| threshold | segments firing | events with ≥1 firing |
|---|---|---|
| 15° | 232/605 (38.3%) | 183/407 (45.0%) |
| 20° | 155/605 (25.6%) | 126/407 (31.0%) |
| 25° | 101/605 (16.7%) | 87/407 (21.4%) |
| 30° | 61/605 (10.1%) | 55/407 (13.5%) |

Even at 30° this fires on 1 in 7 segments — far too broad for a blanket rule.

Adding a minimum-arm-span requirement (each PCA arm must span a minimum
fraction of the requested baseline, not just meet the existing 3-point
floor) reduces the footprint (14.5% of segments at `min_span=15cm`) but
**does not uniformly fix the argmax**: it fixes 172832 (wrong-point argmax at
14.6 cm from truth → correct region at 3.9 cm) but *actively worsens* 61681
(3.2 cm → 7.1 cm → 11.8 cm from truth as the span requirement tightens).
The reason is structural: 61681's true corner sits only ~2 cm from the
segment's own end (arm span on one side is genuinely 2.1 cm), which is
exactly the geometry a span guard is built to suppress as a *spurious*
near-end PCA artifact. **Arc-length span cannot distinguish a true end-
adjacent corner from a spurious one; this is not a usable discriminator on
its own**, and no blanket wide-turn-angle knob is proposed as a result.

§3b's instrumented evidence sharpens this rather than contradicting it: the
spurious peak that actually won on 320865 has an arm span of **1.94 cm**
against a 35 cm baseline (5.5% of requested) — nowhere close to the 61681
true corner's already-short-but-real 2.1 cm arm on a *shorter* production
baseline in that case. A guard scoped to *how starved the PCA window is
relative to what it asked for*, evaluated only inside the existing
`segment_two_end_break_scan` caller (not as a new blanket rule over every
segment in the event, which is what this section's census tested and
rejected), is a narrower and better-targeted lever — see §6.

## 5. 61681 is refinement-scale; 320865/172832 are true segmentation failures

The DL-vertex rerank composite (`NeutrinoVertexFinder.cxx:4725-4855`,
`dl_vtx_min_accept_score = 10.0` in SBND production) shows a real difference
in kind between 61681 and the other two, from each event's
`vertex_scoreboard` block in `calib-pr-evt<ID>.json`:

| evt | route | `dl_best_score` | dist(DL rank-0 voxel, truth) |
|---|---|---|---|
| 320865 | `dl-rerank-reject` | 3.88 | 16.9 cm (best candidate still far off) |
| 172832 | `dl-rerank-reject` | 7.90 | 12.2 cm |
| 61681  | `dl-rerank-accept` | 84.47 | **0.13 cm** |

For 61681 the DL vertex prior is already essentially exact; the 4.4 cm reco
error is refinement-scale (consistent with pr/89's finding that the
production DL/topology pipeline is close to optimal in this regime) and not
really a segmentation bug in the same sense as the other two. For
320865/172832 the DL model has no good candidate near truth either — because
the clustering never separated the two prongs, there is no voxel evidence at
the true vertex for DL to rank highly. These two are genuine upstream
segmentation failures; fixing them is a prerequisite for the DL vertex to
even have a chance, not an alternative to fixing them.

## 6. Proposed solution (not implemented — proposals + required gates)

Given three distinct mechanisms and a measured-negative on the generic fix,
the proposal is narrow and per-mechanism, not one knob.

**For the 172832/61681 class (gate declines, `n_long != 1`):**
Widen `break_two_end_dqdx`'s entry gate to tolerate a second long prong when
it is geometrically disjoint from the turn region under examination in the
candidate long segment — i.e. don't require the *whole cluster* to have only
one long segment, only that the *candidate's own* two-end analysis isn't
confused by an unrelated distant prong. This would need to ship as a new
default-OFF knob (e.g. `teb_allow_second_prong`), gated by:
- byte-identical A/B on `abtest/events.txt` with the knob off, then
- a smoke run confirming the break now fires on 172832/61681 with the knob on
  and lands within a few cm of the hand-scan truth, then
- a hand-scan-labeled sample (per pr/79: no net claim without a live A/B)
  before any default flip, since loosening this gate could admit false
  breaks elsewhere in the 45% of events that already have some segment
  crossing a naive turn threshold (§4).

**For the 320865 class (break fires, wrong location):** the mechanism is now
confirmed (§3b, §0.1): `segment_wide_turn_angle`'s PCA window silently accepts
a truncated arm near a segment end instead of requiring it be (close to) the
full requested baseline, so a 4-point/1.94 cm degenerate PCA reading can
outscore a well-formed 56-point/34 cm one by a few degrees of pure fit
jitter. The proposed fix is scoped to the `k_turn` argmax eligibility test
inside `segment_two_end_break_scan` (`PRSegmentFunctions.cxx:583-591`), not a
blanket rule over all segments (§4 already rejected that as too broad):

- Add a new `TwoEndBreakOptions` field, e.g. `turn_min_arm_frac` (fraction of
  `turn_baseline` each PCA arm must span to be eligible), **default 0 = off,
  current unguarded behavior, byte-identical**.
- When set (e.g. to 0.7, i.e. require ≥24.5 cm of the 35 cm baseline on both
  arms), index 8's 1.94 cm arm A (5.5% of baseline) would be excluded from
  the `k_turn` argmax search, leaving the well-formed true-corner region
  (idx 84–88, ~97% of baseline on both arms) as the winner.
- This directly and only affects the R2/turn route already in
  `segment_two_end_break_scan`, so it can only change behavior on events
  where `break_two_end_dqdx` already fires — it does not touch 172832/61681
  (§3a, gate never reached) and does not touch `segment_search_kink`'s
  independent accept ladder (§3c note below on `flag_switch`).
- Required before any default flip: byte-identical A/B on `abtest/events.txt`
  with `turn_min_arm_frac=0` (must be a no-op), then a targeted census of
  every event where `break_two_end_dqdx` currently fires — how many of those
  breaks currently land on a short-arm candidate like idx 8, and does
  `turn_min_arm_frac≈0.7` move all of them into better agreement with
  hand-scan truth without ever changing an otherwise-correct break — then a
  hand-scan-labeled sample per pr/79 before any default flip. This last step
  is not done in this doc; §0.1's dump only covers the one event.

**Explicitly not proposed:**
- Relaxing the four hard-coded `segment_search_kink` accept thresholds
  (`PRSegmentFunctions.cxx:859-877`) — verified byte-faithful to
  `prototype_base/pid/src/ProtoSegment.cxx:838-850`; this is M15 territory
  (a deliberate value, not a bug) and none of these three events' turns pass
  even the loosest criterion (para_angle gate fails all three: 8.5°/42.0°/42.7°
  vs the 7.5–15° floor combined with the other per-criterion conditions), so
  relaxing them would be a much broader behavior change than this
  investigation supports.
- Widening `vks_radius` alone — ruled out in §3c, the snap isn't the failing
  mechanism here.
- Shipping the bare wide-turn-angle threshold from §4 as a blanket knob over
  all segments — the footprint (16–45% of events depending on threshold) and
  the span-guard's event-dependent sign flip (§4) make that unsafe without
  much more work. `turn_min_arm_frac` above is deliberately scoped narrower
  (inside the existing R2 route, only where `break_two_end_dqdx` already
  runs) to avoid this failure mode.

One additional untested but plausible contributor, flagged per M15 rather
than picked silently: the WCT-only `flag_switch` stability guards in
`segment_search_kink` (`PRSegmentFunctions.cxx:920-932`, requiring
`num_p1>=9` points and `length2_1>3cm`) are a documented divergence from the
prototype that make WCT strictly less likely to break near a segment end —
and 61681's true corner is ~2 cm from an end. This was not tested this
session; a targeted single-event instrumentation of `segment_search_kink`
itself (separate from `break_two_end_dqdx`) would be the way to check it.

## 7. Summary table

(Round-2 note: the "proposed next step" column below is round 1's; the
executed outcome per event is §8.10.)

| evt | mechanism | status | proposed next step |
|---|---|---|---|
| 320865 | `two_end_break` fires, breaks at a spurious 4-pt/1.94cm-arm PCA turn 47cm short of the true corner (confirmed via instrumentation, §0.1/§3b) | segmentation failure, mechanism resolved — **FIXED in §8, SBND ON** | `turn_min_arm_frac` knob (default 0/off) + targeted census + A/B (§6) |
| 172832 | `two_end_break` gate declines (`n_long != 1`) — §3a inference CONFIRMED live in §8.5 | segmentation failure — gate-widening measured NEGATIVE (§8.5), unchanged in production | new default-OFF gate-widening knob + its own A/B (§6) |
| 61681  | `two_end_break` gate declines; but DL vertex already near-exact | refinement-scale, not segmentation | lower priority; possibly moot once DL/topology tuning (pr/89) improves acceptance near this geometry |

## 8. Round 2 — implementation + validation (owner-requested)

The owner asked for the §6 fix to be implemented and validated on the
48-event nueCC + 19-event NCpi0 + 1000-event data samples (mcp2k excluded),
with every other changed event checked fix-vs-regression, and the knob
flipped ON for SBND if validation passes.

### 8.0 Repro

```bash
# toolkit: both knobs added (see 8.1), built with wcbuild; freshness proofs:
#   v1 binary: libWireCellClus.so 2026-08-17 06:35:41 > last source edit 06:33:45
#   v2 binary: libWireCellClus.so 2026-08-17 07:12:46 > last source edit 07:11:35
# unit tests: ./build/clus/wcdoctest-clus -> 210/210 cases PASS on both binaries
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export PR_JOBS=32 PR_EXTRA_STAGES=pr_display SBND_DL_VTX_HARVEST=true
# knobs-off arms (byte-identity gate legs), all three samples:
./run_pr_chain_batch.sh work-mcp1k-cb0805   work-mcp1k-pr90off   data
./run_pr_chain_batch.sh work-nuecc48-cb0805 work-nuecc48-pr90off data
./run_pr_chain_batch.sh work-ncpi0-cb0805   work-ncpi0-pr90off   data
python3 scripts/analysis/pr64/pr64_gate.py work-<sample>-harv3 work-<sample>-pr90off
# round-1 (v1 filter semantics, §8.6 — FAILED adjudication, arms retained):
SBND_TEB_TURN_MIN_ARM_FRAC=0.4 ./run_pr_chain_batch.sh work-<sample>-cb0805 work-<sample>-pr90on data
python3 scripts/analysis/pr64/pr64_gate.py work-<sample>-pr90off work-<sample>-pr90on
python3 scripts/pr90_movers.py work-mcp1k-pr90off work-mcp1k-pr90on   # --tags harv3 default
# round-2 (v2 two-tier semantics, §8.7-8.8 — the shipped binary):
./run_pr_chain_batch.sh work-<sample>-cb0805 work-<sample>-pr90off2 data
SBND_TEB_TURN_MIN_ARM_FRAC=0.4 ./run_pr_chain_batch.sh work-<sample>-cb0805 work-<sample>-pr90on2 data
python3 scripts/analysis/pr64/pr64_gate.py work-<sample>-harv3  work-<sample>-pr90off2
python3 scripts/analysis/pr64/pr64_gate.py work-<sample>-pr90off2 work-<sample>-pr90on2
python3 scripts/pr90_movers.py work-mcp1k-pr90off2 work-mcp1k-pr90on2
# smoke arms: work-mcp1k-pr90smoke1 (v1, frac=0.45 + secmax=15, 3 evts),
# work-mcp1k-pr90smoke2 (v1, frac=0.4, 3 evts),
# work-mcp1k-pr90smoke3 (v2, frac=0.4, 13 evts = the 11 v1 movers + 172832 + 61681)
```

### 8.1 What was implemented (toolkit, both default-OFF)

**Knob 1 — `teb_turn_min_arm_frac`** (dimensionless fraction of
`teb_turn_baseline`, C++ default 0 = legacy). In route R2's `k_turn` argmax
(`clus/src/PRSegmentFunctions.cxx`, inside `segment_two_end_break_scan`),
an index's arms are "well-formed" when BOTH PCA arms' achievable arclength
(bounded by the segment end, beyond the skirt) reaches
`frac * turn_baseline` — closed-form from cumulative arclength
(`cum[k] - skirt` / `L - cum[k] - skirt`), no PCA re-collection needed.

The knob went through two semantics this round; the SHIPPED one is the
second:

- **v1 (§8.6, tested and rejected)**: hard eligibility filter — starved
  indices simply excluded from the argmax.
- **v2 (§8.7, shipped)**: two-tier preference — a first argmax pass runs
  over well-formed indices only, and its winner is kept ONLY if it clears
  `teb_turn_angle` on its own; otherwise the legacy unrestricted argmax
  (starved candidates included) stands unchanged.

Either way only R2's argmax inside this scan is affected; R1 (dip),
`segment_search_kink`, and the shared `segment_wide_turn_angle` helper are
untouched (§4's rejection of any blanket rule stands).

**Knob 2 — `teb_second_max`** (cm, C++ default 0 = legacy). The §6
gate-widening: `break_two_end_dqdx`'s entry gate
(`clus/src/NeutrinoPatternBase.cxx`) tolerates additional long
(`> teb_stub_max`) segments as long as exactly ONE segment exceeds this cap
(that one becomes the candidate). 0 = legacy strict `n_long != 1` gate.

Threading follows the pr/86 five-layer pattern: `TwoEndBreakOptions` field
(knob 1) / caller-only member (knob 2, like `teb_stub_max`);
`TaggerCheckNeutrino` members + `configure()` get + `default_configuration()`
echo + cm→internal copy; `PatternAlgorithms` mirrors;
`doctest_clus_knob_defaults.cxx` default-OFF checks (both keys); jsonnet
args + key-suppression in `cfg/pgrapher/common/clus.jsonnet`, both SBND
`clus.jsonnet` layers, and `wct-pr-perevt.jsonnet` TLA (default null).
Runner escapes `SBND_TEB_TURN_MIN_ARM_FRAC` / `SBND_TEB_SECOND_MAX` added to
`run_pr_chain_batch.sh`. The existing `break_two_end_dqdx` debug log line
gains `nlong=/armfrac=/secmax=` fields (log-only, not gated content).

### 8.2 Off-path proofs

- **Compiled-config**: full 15-stage runner pipeline compiled pre-change vs
  post-change with knobs off — `cmp` **byte-identical**; the
  `abtest/compile_all_cfg.sh` + `cmp_cfg.sh` sweep over all 16 live
  SBND/PDHD/PDVD jobs also PASSes (0 normdiff everywhere). With the knob
  TLAs set, exactly the two new keys appear, once each. (Note:
  `compile_all_cfg.sh`'s own `sbnd_pr` slice omits the
  `tagger_check_neutrino` stage from `pipeline_names`, so the full-pipeline
  compile above is the one that actually exercises the new keys.)
- **Unit tests**: `./build/clus/wcdoctest-clus` 210/210 cases PASS,
  including the two new default-OFF knob checks.
- **Byte-identity gate (the §6 requirement)**: `pr64_gate.py`
  (mabc-pr.zip + pctree tar member-content hashes + exact-byte nusel tsv)
  vs the harv3 production arms, knobs off: v1 binary **1067/1067 identical,
  0 movers** (arms `work-{mcp1k,nuecc48,ncpi0}-pr90off`); shipping v2
  binary gate in §8.8 (arms `work-*-pr90off2`). The full-pipeline
  compiled-config `cmp` was re-verified byte-identical after the v2 edits.

### 8.3 Census of currently-firing events (harv3 arms, 1067 logs)

`grep -a "BROKE cluster" work-*-harv3/pr_evt*/wct_pr_evt*.log`:
**38 events fire** `break_two_end_dqdx` in production (37 mcp1k, 1 NCpi0,
0 nueCC48). By route: 24 route-1 (dQ/dx dip) — knob 1 cannot touch these
(R1 is evaluated first and its dip index is the break) — and 14 route-2
(turn). Of the 14 route-2 breaks, **11 sit on a starved arm** (shorter arm
4.3–6.1 cm — the same signature as 320865's idx 8): evts 283905, 291064,
281214, 285443, 59261, 59247, 319611, 320865, 64921, 64503, 72586. The
three healthy route-2 breaks have shorter arms of 35.9 cm (278420),
18.4 cm (172942) and 69.0 cm (349461).

**Operating point** chosen from this table: `frac = 0.4` (required
achievable span ≥ 14 cm of the 35 cm baseline, i.e. break index ≥ 17 cm
from both ends). This excludes every starved-arm break (≤ 6.1 cm, margin
> 2×) while keeping the borderline-but-genuine 172942 break (18.4 cm arm =
15.4 cm span; the §6 illustrative 0.7 — and even 0.45 — would have clipped
it) and admitting 172832's true corner (21.6 cm from its far end). The §6
worry that 0.7 was too tight was real.

### 8.4 Smoke, shipping config (`frac=0.4` only) — `work-mcp1k-pr90smoke2`

| evt | d(main vtx, hand-scan truth) harv3 | pr90smoke2 | note |
|---|---|---|---|
| 320865 | 37.29 cm | **1.22 cm** | break idx 8 → **84** (s=52.4 cm, turn 33.1°, arms 52.4/146.3 cm) — the §3b true-corner region |
| 172832 | 20.35 cm | 20.35 cm (byte-identical) | gate still declines (knob 2 off) — knob independence confirmed |
| 61681 | 4.36 cm | 4.36 cm (byte-identical) | same |

### 8.5 Knob 2 (`teb_second_max=15`) smoke — NEGATIVE, stays OFF

`work-mcp1k-pr90smoke1` (frac=0.45 + secmax=15) did confirm §3a's inferred
mechanism directly: with the cap set, both 172832 and 61681 now enter the
pass (`nlong=2` in the new log field) — the gate WAS the blocker. But after
admission the scan's own route selection does not find the true corner:

- **172832**: R2's wide-baseline turn at the eligible indices tops out at
  18.3° < the 25° accept, so route R1 wins with a dQ/dx dip at fit idx 140
  (84 cm from the front; truth corner is ~106 cm in). Main vertex moves
  20.35 → **21.65 cm** from truth — **ADVERSE** by the pr/78/79 1 cm bar
  (+1.30 cm).
- **61681**: R1 dip at idx 166 (6.1 cm from the far end); 4.36 → 4.64 cm —
  churn within the bar, no benefit (§5 already called this event
  refinement-scale).

Per the pr/81 precedent (measured-negative, no live A/B spent), knob 2
ships implemented but **default OFF and NOT flipped**; fixing the
172832 class needs different point-selection physics after admission (the
dip route dominates and its deepest dip is not the corner), not just the
gate. The knob 1 live A/B below therefore runs with `frac=0.4` ONLY.

### 8.6 Live A/B round 1 — v1 filter semantics FAIL adjudication

Arms `work-{mcp1k,nuecc48,ncpi0}-pr90on` (v1 binary, `frac=0.4`), gated
against the pr90off arms with `pr64_gate.py`: **11 movers, all mcp1k, all
inside the §8.3 starved-arm census set** (nueCC48 48/48 and NCpi0 19/19
identical — footprint containment exactly as predicted). But the per-event
outcome was NOT "the break moves to the healthy corner": for 10 of the 11
the restricted argmax topped out below the 25° accept, so the v1 filter
**removed the break entirely**; only 320865 got a moved break (idx 84,
33.1°).

`pr90_movers.py` vs the harv3-epoch labels (7 of 11 labelled):

| evt | moved (cm) | click→main off → on | verdict | note |
|---|---|---|---|---|
| 291064 | 159.36 | 159.36 → **0.00** | toward | removing the spurious starved break (idx 7, 27.4°) let the main vertex land exactly on the click; numu 1.75→3.17 |
| 320865 | 38.43 | 37.29 → **1.22** | toward | the §3b fix proper |
| 64503 | 43.61 | 0.00 → 43.61 | **ADVERSE** | b1=0.00 — owner-approved vertex WAS the starved break (idx 68, arm 5.5 cm) |
| 319611 | 3.31 | 0.00 → 3.31 | **ADVERSE** | b1=0.00, and cosmict_flag flips 0→1 |
| 59247 | 1.12 | 0.00 → 1.12 | **ADVERSE** | b1=0.00, marginal (+0.12 cm over the bar) |
| 59261 | 4.47 | 83.34 → 87.68 | **ADVERSE** | both far off truth either way |
| 72586 | 3.53 | 297.40 → 300.35 | **ADVERSE** | both hopeless (cosmic-scale) |

Unlabelled movers 281214/283905/285443/64921: break removed, main vertex
unmoved (dvtx = 0.0), Enu shifts up to 210 MeV.

**Verdict: FAIL** (5 ADVERSE vs the zero-ADVERSE bar). The decisive
evidence is the three b1=0.00 rows: **genuine, owner-approved corners DO
sit 4–5.5 cm from a segment end with starved PCA arms** — indistinguishable
by span from 320865's spurious idx 8 (5.3 cm). This is §4's 61681 lesson
recurring in live data: span is not a validity discriminator. The v1 arms
are retained (`work-*-pr90on`) as the record of this negative.

### 8.7 v2: two-tier preference (shipped semantics)

What actually separates the fixable class from the harmful one in the §8.6
table is not the starved candidate itself but **whether a well-formed
competitor above threshold exists**: 320865's healthy-arm argmax reaches
33.1° ≥ 25°, while for all five ADVERSE events (and 291064) the healthy-arm
argmax tops out at 4.3–20.8° < 25°. So v2 keeps the starved-arm candidates
as the fallback and only PREFERS the well-formed winner when it clears
`teb_turn_angle` on its own:

```cpp
if (opt.turn_min_arm_frac > 0) {
    // pass 1: argmax over indices with both arms >= frac * baseline achievable
    ...
    if (turn_max < opt.turn_angle) { k_turn = -1; turn_max = 0; }  // no well-formed corner
}
if (k_turn < 0) {
    // pass 2: legacy unrestricted argmax (byte-identical when knob off)
    ...
}
```

Consequences, verified on the 13-event smoke `work-mcp1k-pr90smoke3`
(11 v1 movers + 172832 + 61681, v2 binary, `frac=0.4`): **12/13
byte-identical to pr90off; the only mover is 320865** (break idx 8 → 84).
All three owner-approved near-end breaks (64503/319611/59247) and 291064's
fallback break are preserved bit-for-bit. The cost, stated explicitly:
**291064's §8.6 fix is forgone** — its class (spurious starved break with
NO healthy corner above threshold) cannot be told apart from
64503/319611/59247's class (genuine starved break, same signature) by any
geometry this round measured; killing those breaks is off the table, so
291064 stays at production behavior (159 cm off, as today).

### 8.8 Live A/B round 2 — v2, `frac=0.4`, 1067 events: PASS

Arms `work-{mcp1k,nuecc48,ncpi0}-pr90off2` / `-pr90on2`, both on the
shipping v2 binary, `pr64_gate.py` throughout:

- **Knobs-off byte-identity vs harv3 production**: mcp1k 1000/1000,
  nueCC48 48/48, NCpi0 19/19 — **1067/1067 identical, 0 movers**. The v2
  restructure of the argmax loop leaves the knob-off path bit-exact.
- **Knob-on vs knobs-off**: **1066/1067 identical; the single mover IS the
  target event** (mcp1k 320865, `mabc-pr.zip`; nueCC48 and NCpi0 fully
  untouched).
- **Adjudication** (`pr90_movers.py`, harv3-epoch labels, 407 compared):
  `evt 320865 moved 38.43 cm, click→main 37.29 → 1.22, toward` —
  **1 mover, 0 ADVERSE**. Exit 0.
- **Score deltas** (`pr83_ab_compare.py`): only 320865 changes — numu
  2.83→3.48, nue −15.00→−4.30, Enu 674→754 MeV, cosmict unchanged.

### 8.9 Flip decision: `teb_turn_min_arm_frac = 0.4` SBND PRODUCTION ON

All §6-required gates held (byte-identical off on the full 1067-event
manifest, live A/B with the mover set fully adjudicated, zero ADVERSE, the
one mover is the target event moving onto the click), so per the owner's
request the knob is ON in SBND production:
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet` TLA default
`teb_turn_min_arm_frac = 0.4` (cfg-only commit; escapes
`SBND_TEB_TURN_MIN_ARM_FRAC` / `-A` restore legacy).

- Compiled-config proof: full-pipeline compile of the flipped config vs the
  pre-change tree differs by exactly one line —
  `+ "teb_turn_min_arm_frac": 0.4`.
- Bare-config verification: evt 320865 rerun with NO env overrides
  (`work-mcp1k-pr90bare`) is member-hash-identical to the validated
  `pr90on2` arm (mabc-pr.zip rollup `3e0c2ccf…`, pctree 425 members
  `a3455f2a…`, nusel byte-identical) — bare run == production == the
  validated arm (doc 68 single-source rule).

**NOT flipped**: `teb_second_max` stays OFF (§8.5 negative). Its
production default is C++ 0 / jsonnet null — byte-identical, escape-only.

### 8.10 Where this leaves the three motivating events, and the residual class

| evt | production before | production after the flip | status |
|---|---|---|---|
| 320865 | vertex 37.3 cm off truth (spurious idx-8 break) | **1.2 cm** (break at the true corner) | **FIXED, validated live** |
| 172832 | 20.4 cm off (gate declines) | unchanged | NOT fixed — §8.5: admission alone breaks at the wrong dip (ADVERSE); needs point-selection physics after admission |
| 61681 | 4.4 cm off (gate declines; DL near-exact) | unchanged | refinement-scale (§5), deliberately untouched |

### 8.11 Bee links (owner review, built 2026-08-17)

Zips built with `scripts/bee/make_pr_bee.py` from the §8.8 arms
(`/home/xqian/tmp/pr90_bee/pr90-{before,after,variants}.zip` + index/prid-map
files alongside):

- **before** (production = `work-mcp1k-pr90off2`), events in Bee-index order
  320865, 172832, 61681, 291064, 64503, 319611, 59247, 172942:
  `https://www.phy.bnl.gov/twister/bee/set/bcdf1c77-c5c0-413d-80d2-4f9a3d0c4ae9/event/list/`
- **after** (flipped production = `work-mcp1k-pr90on2`), same 8 events:
  `https://www.phy.bnl.gov/twister/bee/set/0ba37dff-6f79-48ec-bfc5-4a5251c139fb/event/list/`
  Only index 0 (320865) differs from "before" — the §8.8 single-mover claim,
  inspectable. Indices 3–7 are the preserved-break events: 291064 (the
  residual spurious starved break, unchanged), 64503/319611/59247 (the
  owner-approved b1=0 near-end breaks v1 would have destroyed), 172942 (the
  18.4 cm-arm genuine break the 0.4 operating point deliberately keeps).
- **variants** (the two REJECTED designs, NOT production): 172832 + 61681
  from `work-mcp1k-pr90smoke1` (knob 2 ON — the gate admits and breaks at
  the wrong dip, §8.5) and 291064 + 64503 + 319611 from `work-mcp1k-pr90on`
  (v1 filter — removing the starved breaks fixes 291064 but destroys
  64503/319611, §8.6):
  `https://www.phy.bnl.gov/twister/bee/set/d4bdce6e-0108-4b49-bb28-bfb9943d222b/event/list/`

Residual classes for a future round, with the §8.6 live evidence attached:
(a) 172832-class — gate-widening admits the segment but R1's deepest dip
is not the corner and R2's wide turn is below threshold there; (b)
291064-class — a spurious starved-arm break with NO healthy corner above
threshold; killing it fixed 291064 completely (159 → 0 cm) but the same
kill breaks the owner-approved b1=0 vertices of 64503/319611/59247, so a
kill rule needs a discriminator beyond arm span (dQ/dx template quality at
the candidate? DL-vertex confirmation?). Both classes are documented
negatives, not proposals. **Superseded by §9**, which resolves both
classes with owner physics input and instrumented profiles, and designs
the fixes.

## 9. Round 3 — owner physics input, instrumented profiles, fix designs (2026-08-17; implementation deferred to the next session)

### 9.0 Owner input (verbatim intent, 2026-08-17)

1. **172832 is a muon → Michel decay**, not a back-to-back particle pair:
   the Michel at the muon's end is what creates the extra prong. The whole
   cluster is still a *line* — there is no 3-track vertex — so the gate
   should tolerate it.
2. **61681 is a clear 2-track topology**; "something weird happened near
   the short track region."
3. Of the §8.11 preserved-break events, **291064 AND 64503 should NOT be
   broken**; 319611, 59247 and 172942 are OK. The discriminator is
   **consistency between a Bragg peak and the dQ/dx** — a small high-dQ/dx
   spot at a track end can be ordinary *vertex activity* at the neutrino
   vertex, and must not be read as a stopping-particle Bragg.

Point 3 corrects the §8.6 adjudication: 64503's `b1=0.00` label was
reco-anchored (the pr/85 caveat in action — the owner clicked the displayed
break vertex, and now overrides it). So of §8.6's five ADVERSE, 64503 was
actually a *correct* kill by v1, and the true regressions were
319611/59247 (+ the two noise rows). The v1 filter still fails (it cannot
tell those apart), and the shipped v2 preference remains the right call —
but the future kill rule (§9.4/D4) now has a clean target set:
kill {291064, 64503}, keep {319611, 59247, 172942}.

### 9.1 Repro (instrumented rerun, same method as §0.1)

The §0.1 `WCT_TEB_DUMP` diagnostic was temporarily re-added to
`segment_two_end_break_scan` (identical shape, one extra column: the
10 cm-baseline `segment_wide_turn_angle` `t10` next to the production
35 cm `t35`), built with `wcbuild` (freshness: lib 08:06:35 > edit
08:06:04), and run:

```bash
cd sbnd_xin
# batch A -- production defaults (12 currently-firing events):
WCT_TEB_DUMP=/home/xqian/tmp/pr90_dumps/batchA.txt PR_JOBS=1 \
  PR_EXTRA_STAGES=pr_display SBND_DL_VTX_HARVEST=true \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-mcp1k-kink90c data \
  320865 291064 64503 319611 59247 172942 59261 72586 281214 283905 285443 64921
# batch B -- gate-widened so 172832/61681 reach the scan:
WCT_TEB_DUMP=/home/xqian/tmp/pr90_dumps/batchB.txt PR_JOBS=1 \
  PR_EXTRA_STAGES=pr_display SBND_DL_VTX_HARVEST=true SBND_TEB_SECOND_MAX=15 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-mcp1k-kink90d data 172832 61681
```

Dumps at `/home/xqian/tmp/pr90_dumps/batch{A,B}.txt` (one block per scan
call, matched to events by segment length; format: idx, s(cm), x, y, z,
dqdx, t35, t10, arm_ok). Topology/graph numbers below come from the
production `work-mcp1k-pr90off2/pr_evt<ID>/calib-pr-evt<ID>.json` dumps.
The instrumentation was reverted from the source tree immediately after
capture (`git status` clean for this file). NOTE on the binary: a
concurrent session had in-flight edits under `clus/` at revert time, so
`local/lib` was deliberately NOT rebuilt — the installed lib still carries
the env-gated dump (byte-identical with `WCT_TEB_DUMP` unset; the §8
gates all used earlier clean binaries, 06:35/07:12); the next routine
`wcbuild` restores exact-HEAD binaries.

### 9.2 172832 anatomy: the junction is vertex activity + a local turn, not a dip

Production graph (calib JSON): a pure chain — vtx 21000 (deg 1, far end)
—127.2 cm seg 21001 (pid 13)— vtx 21002 (deg 2, **current main vertex**,
20.35 cm from the click) —13.1 cm seg 21021 (pid 211, the Michel)— vtx
21001 (deg 1). Every vertex has degree ≤ 2: the cluster is a line, exactly
as the owner said. **The click sits ON seg 21001 at fit idx 177/213,
0.20 cm off the trajectory** — i.e. the true topology is
[~104 cm prong] → **nu vertex** → [~21 cm muon] → Michel.

The scan-side dump (batch B, N=214, L=127.7 cm) shows why both existing
routes miss it:

- **R1 (dip)**: the winning "deepest dip" at idx 140 (s=84.0 cm) is
  q = 0.77×MIP — one of half a dozen equal-depth downward fluctuations of
  an ordinary MIP track (0.72–0.79 at idx 134/139/160/169/174-175). It is
  not a junction signature; the two-Bragg valley model does not apply.
- **The true junction is BRIGHT, not a valley**: idx 176–179 read 1.27,
  1.81, **2.50**, 2.00 ×MIP — vertex activity exactly at the click
  (idx 178, d = 0.19 cm), just as the owner's note predicts.
- **The local turn finds it**: `t10` (10 cm baseline) plateaus at
  **19–23.5°** over s ≈ 101–110 cm, centered on the click, against a
  5–7° mid-track baseline. The production `t35` tops out at 18.3° there
  (< the 25° accept) because the 35 cm arms average the corner away —
  and t35's *unrestricted* argmax is yet another starved near-end artifact
  (26.5° at idx 206, 3.7 cm from the Michel end).

### 9.3 61681 anatomy: the junction vertex floats past the true vertex on a charge-less bridge

Production graph: chain vtx 2000 —109.5 cm seg 2001 (pid 13)— vtx 2002
(deg 2, **main vertex**, 4.36 cm from click) —11.2 cm seg 2002 (pid 2212,
Bragg 6.3×MIP at its far end vtx 2001)— a genuine 2-track topology. The
"weird thing near the short track region", from the fit-point profiles:

- The muon's trajectory passes **through the click at idx 173/183
  (0.36 cm)** and keeps going: idx 174–178 read 1.7–3.1×MIP (the vertex
  activity region), then idx 179–181 read **14.4k–17.2k ≈ 0.25–0.32×MIP —
  essentially charge-less** — and the junction vertex 2002 sits at the end
  of that empty tail, 4.36 cm past the click.
- The proton mirrors it: its first 4 points (leaving the junction) are the
  same sub-MIP bridge (0.32–0.63×MIP), then the same activity spike
  (2.2–2.8×MIP), then a clean MIP proton to its Bragg.
- No existing pass can see this: `fit_distance` = 1.32 (healthy), the
  kink-accept ladder fails on its angle criteria (§3c/§6), and dQ/dx is
  never consulted for vertex *placement*. The scan-side dump confirms the
  turn signature: `t10` climbs 18→24→31→38→44→**54°** approaching the
  click (t10's window can no longer form inside the last ~2 cm).

### 9.4 The Bragg-vs-vertex-activity discriminator, measured

End profiles of the break-adjacent segment end (q in MIP units, s = cm
from that end; from batch A), with the owner's keep/kill labels:

| evt | class | end q̄(0–1 cm) | peak | hot extent (>1.5×MIP, contiguous from end) | shape |
|---|---|---|---|---|---|
| 291064 | KILL | 2.8 | 3.0 | **> 8.2 cm** (never returns to MIP) | dim, extended plateau — not a Bragg |
| 64503 | KILL | 2.75 | 2.9 | 3.6 cm | dim, moderate — not proton-bright |
| 319611 | KEEP | 3.3 | **4.4** | 2.8 cm | bright, compact — proton Bragg |
| 59247 | KEEP | 2.0 | 2.5 | 0.9 cm | very compact stub Bragg |
| 172942 | KEEP | 1.0 | — | 0 (end at MIP; mid-segment break) | n/a |

Candidate rule that separates all five labelled events:
**hot-extent (cm) > peak (×MIP) ⇒ vertex-activity/overlap, veto the
break** (a genuine Bragg concentrates its charge: bright relative to its
length). KILL: 291064 (8.2 > 3.0), 64503 (3.6 > 2.9). KEEP: 319611
(2.8 < 4.4), 59247 (0.9 < 2.5), 172942 (no hot end). Closest margin is
64503-vs-319611 (~25%); with only five labels this is a *candidate*, not
an operating point. The six unlabelled §8.6 movers under the same rule:
281214 (2.9 peak / ~8 cm — kill-like), 64921 (2.4 / ~8 cm — kill-like),
285443 (end at MIP; its 1.29 "rise" is marginal — kill-like), 283905
(4.2 / ~2.9 cm + a 1.6–1.8 plateau — ambiguous), 59261 (3.5 / ~2.5 cm +
plateau — ambiguous), 72586 (5.9 spike, compact — keep-like). These need
an owner micro-scan before any threshold is frozen.

Note the simple metrics that do NOT work, measured: the scan's own
template scores (64503's short arm scores *better*, 0.55, than
59247's 0.708), the rise ratios (kill 1.35/1.73 vs keep 1.60/1.70/1.94 —
interleaved), and arm span (§8.6).

**Micro-scan Bee links for the six (built 2026-08-17, zips
`/home/xqian/tmp/pr90_bee/pr90-micro-{break,nobreak}.zip`).** Same six
events, same Bee-index order 281214, 64921, 285443, 283905, 59261, 72586:

- **break present** (production, `work-mcp1k-pr90off2`):
  `https://www.phy.bnl.gov/twister/bee/set/e27d3c67-4778-4200-b213-f59f94bed640/event/list/`
- **break removed** (the §8.6 v1 arm, `work-mcp1k-pr90on`):
  `https://www.phy.bnl.gov/twister/bee/set/a99f2a95-abc6-4f41-8096-4bac22960a58/event/list/`

Per-event break coordinates (all route-2, near-end):
281214 (−30.3, −32.4, 418.9) 5.4 cm from one end of a 77.8 cm track;
64921 (60.3, 169.3, 164.4) 4.4 cm / 88.7 cm;
285443 (−81.9, 47.8, 495.9) 4.6 cm / 72.7 cm;
283905 (−136.8, 109.3, 243.0) 4.9 cm / 112.9 cm;
59261 (97.3, 129.8, 176.9) 4.9 cm / 89.5 cm;
72586 (−60.7, −195.0, 58.3) 6.1 cm / 311.5 cm.
The scan question per event: is the short piece beyond the break a real
second particle (own Bragg to its tip ⇒ break correct, KEEP), or is the
end brightness vertex activity / overlap on a single track (⇒ break
spurious, KILL, and where is the true vertex)? 59261 and 72586 do carry
harv3 labels but the clicks are 83 / 297 cm away from the break region and
do not adjudicate it. The recorded verdicts become the calibration labels
for D4's threshold.

### 9.4b Owner micro-scan verdicts (2026-08-17) and the calibrated D4 rule

Owner verdicts on the six: **281214 not to break; 64921 OK to break;
285443 no need to break (though); 283905 should not break; 59261 no need
to break (though); 72586 no need to break, but OK.** Encoded as: hard KILL
= 281214, 283905 (joining 291064, 64503); KEEP = 64921 (joining 319611,
59247, 172942); tolerable-either-way = 285443, 59261, 72586.

Full 11-event calibration (turn = the accepted route-2 turn at the
production break, from the §8.3 census log lines; peak / hot-extent from
the §9.4 profiles):

| evt | owner | turn (°) | end peak (×MIP) | hot extent (cm) | shape-only rule (§9.4) | turn<30° | combined |
|---|---|---|---|---|---|---|---|
| 291064 | KILL | 27.4 | 3.0 | 8.2 | veto ✓ | veto ✓ | veto ✓ |
| 64503 | KILL | 26.7 | 2.9 | 3.6 | veto ✓ | veto ✓ | veto ✓ |
| 281214 | KILL | 26.5 | 3.5 | 7.7 | veto ✓ | veto ✓ | veto ✓ |
| 283905 | KILL | 26.8 | 4.2 | 6.9 | veto ✓ | veto ✓ | veto ✓ |
| 285443 | no-need | 25.5 | 0.9 | 0 | keep (ok) | veto (ok) | veto (ok, preferred) |
| 59261 | no-need | 30.0 | 3.5 | 2.5 | keep (ok) | keep (ok) | keep (ok) |
| 72586 | no-need, OK | 33.2 | 5.9 | 1.3 | keep (ok) | keep (ok) | keep (ok) |
| 64921 | KEEP | 43.6 | 2.5 | 7.8 | **veto ✗ FAILS** | keep ✓ | keep ✓ |
| 319611 | KEEP | 65.8 | 4.4 | 2.8 | keep ✓ | keep ✓ | keep ✓ |
| 59247 | KEEP | 32.5 | 2.5 | 0.9 | keep ✓ | keep ✓ | keep ✓ |
| 172942 | KEEP | 34.9 | 1.0 | 0 | keep ✓ | keep ✓ | keep ✓ |

Two findings:

1. **The §9.4 shape-only candidate (hot-extent > peak) is refuted**:
   64921's genuine junction (owner: OK to break) has the same dim-extended
   end profile (2.5×MIP over 7.8 cm) as the hard kills 281214/291064 —
   dQ/dx end shape alone cannot separate them. (This mirrors §4's and
   §8.6's lesson: no single geometric/charge scalar has been sufficient.)
2. **The accepted turn angle separates perfectly**: every hard KILL
   clusters just above the 25° accept threshold (26.5–27.4°), every KEEP
   is ≥ 32.5° (and the shipped 320865 fix breaks at 33.1°). A genuine
   back-to-back junction turns hard; the spurious accepts are
   threshold-hugging soft bends.

**Calibrated D4 rule** (supersedes the §9.5 D4 draft): veto an accepted
route-2 break iff

```
turn_at_break < teb_bragg_veto_turn (≈ 30°)
AND NOT bragg_consistent(short-arm end)      # peak ≥ 2.0×MIP AND hot-extent(cm) ≤ peak(×MIP)
```

Scores 4/4 hard kills, 4/4 keeps, and lands all three tolerables on their
preferred side (285443 vetoed, 59261/72586 kept). Margins at 30°: highest
kill 27.4° (−8.7%), lowest keep 32.5° (+8.3%); the one boundary event
(59261, 30.0°, tolerable either way) sits exactly at threshold. The
Bragg-consistency conjunct exists so that a future sub-30° break with a
genuinely bright-compact stopping stub (none in this sample) is NOT
vetoed — the turn cut alone would take it. Route-1 (dip) breaks are
untouched. Vetoing 291064 restores its §8.6 fix (main vertex 159 → 0 cm);
the other vetoes (64503/281214/283905/285443) change segmentation and, per
the §8.6 v1 arm, move no main vertex except through downstream score
changes — the full A/B + movers gate re-checks that at implementation
time.

### 9.5 Fix designs for the next session (all default-OFF; not implemented here)

**D1 — `teb_chain_topology` (bool): line-topology gate admission.**
Replace the rejected bare length cap (`teb_second_max`, §8.5) with the
owner's actual criterion: when `n_long > 1`, admit iff the main cluster's
segment graph is a **simple path** (every vertex degree ≤ 2 — "still a
line, no 3-track vertex") and the candidate is the unique longest
segment. 172832 (deg 1,2,1) and 61681 (deg 1,2,1) both qualify;
a genuine multi-prong vertex (any degree-3 vertex) never does.

**D2 — `vertex_bridge_retract` (new examiner pass): charge-supported
junction retraction — the 61681 fix.** For a degree-2 junction on the
main cluster whose incident fit tails are BOTH sub-MIP (measured
signature: ≥ 2–3 consecutive points < 0.5×MIP spanning ≥ 1.5 cm on each
side), retract the junction along the shared trajectory through the
charge-less bridge and the contiguous vertex-activity blob to the first
MIP-supported point, preferring the local `t10` maximum inside the
retraction window. Measured target: 61681's junction moves 4.36 →
≤ ~0.4 cm from the click (the retraction path idx 182→173 crosses
exactly the 0.25–0.32×MIP bridge then the 1.7–3.1× activity blob to the
54°-turn point). Fires on plain production topology — no gate change
needed — and is inert wherever both tails carry track-level charge.

**D3 — `teb_r3_*`: a turn+activity route for chain-admitted candidates
ONLY — the 172832 fix.** For candidates admitted via D1 (never for the
legacy `n_long == 1` path — zero footprint on the 38 existing breaks by
construction), replace the dip/wide-turn routes with the signature both
events actually show at their true vertex: break at the argmax of `t10`
(10 cm baseline, windows well-formed) subject to
`t10 ≥ teb_r3_turn` (measured: 23.5° and 54° at the clicks vs 5–7°
mid-track; propose ~18°) AND a local vertex-activity corroboration
`max q within ±2 cm ≥ teb_r3_hot` (measured: 2.50× and 2.31×; propose
~1.8×MIP), with the break index refined to the activity maximum.
Measured landing: 172832 → 0.2–1.6 cm from the click (vs 21.7 cm via the
dip route, §8.5); 61681 → ≤ ~1.2 cm (D2 is the preferred fix there; D3
covers it if D2 is not adopted).

**D4 — `teb_bragg_veto`: the keep/kill rule for near-end R2 breaks.**
CALIBRATED in §9.4b against all 11 owner verdicts (the original
shape-only draft here was refuted there by 64921): veto an accepted R2
break iff `turn_at_break < teb_bragg_veto_turn (≈30°)` AND the short-arm
end is not Bragg-consistent (`peak ≥ 2.0×MIP` and
`hot-extent(cm) ≤ peak(×MIP)` over the ~8 cm end window). Kills 291064
(recovering the §8.6 forgone 159 → 0 cm fix), 64503, 281214, 283905
(+285443, preferred) while preserving 319611/59247/172942/64921/72586.
Remaining prerequisites: the standard full-manifest A/B + movers gates at
implementation time.

Validation bar for all four (per §8 precedent): byte-identical off-gates
on the 1067-event manifest, targeted smoke on the named events, full
off-vs-on A/B with every mover adjudicated against harv3-epoch labels
(zero unexplained ADVERSE), owner micro-scan where labels are missing.

### 9.6 Status after round 3

| evt | production today | mechanism (final) | designed fix |
|---|---|---|---|
| 320865 | **FIXED** (1.2 cm, §8) | starved-arm PCA jitter outbidding the true corner | shipped (`teb_turn_min_arm_frac=0.4` ON) |
| 172832 | 20.4 cm off | muon→Michel line; gate declines; junction is activity+turn, invisible to dip/wide-turn routes | D1 + D3 |
| 61681 | 4.4 cm off | junction vertex floats 4.4 cm past the click on a charge-less (0.25–0.32×MIP) fit bridge | D2 (or D1+D3) |
| 291064 | 159 cm off (spurious break kept) | end "rise" is an extended dim plateau, not a Bragg; turn 27.4° (threshold-hugging) | D4 (veto) |
| 64503 / 281214 / 283905 | broken (owner: should not be) | vertex activity / overlap ends; turns 26.5–26.8° | D4 (veto) |
| 285443 / 59261 / 72586 | broken (owner: no need, tolerable) | marginal accepts | D4 lands them veto/keep/keep — all on the tolerated side |
| 319611 / 59247 / 172942 / 64921 | correct breaks kept (owner-confirmed) | genuine junctions, turns ≥ 32.5° | unaffected by D1–D4 |

## 10. Round 4 — D1/D3/D4 implementation + validation (2026-08-17, owner-requested)

### 10.0 Scope: D1+D3+D4 implemented; D2 deferred

Owner request: "implement the designed fix, validate with the 3 samples,
if validated turn them on for SBND as default."  Implemented as three
default-OFF knobs (five-layer pattern, same as round 2):

- **`teb_chain_topology`** (bool, D1): when the entry gate sees
  `n_long > 1`, admit iff the cluster's segment graph is a **simple path**
  (every vertex degree ≤ 2 AND a single connected chain,
  n_vertices = n_edges + 1 — the owner's "still a line, no 3-track
  vertex") and the candidate is the **strictly unique longest** segment.
  Chain-admitted candidates are scanned by route R3 ONLY (the legacy dip
  route on this class breaks at an ordinary MIP fluctuation, §8.5), so
  admission additionally requires both R3 knobs.
- **`teb_r3_turn` / `teb_r3_hot`** (deg / ×MIP-median, D3): route R3 =
  `segment_chain_turn_break_scan` (new, `PRSegmentFunctions.cxx`): break
  at the largest 10 cm-baseline turn (`t10`, production 3 cm skirt) that
  carries a vertex-activity spot ≥ `teb_r3_hot` × MIP-median within
  ±2 cm, refined to the ±2 cm activity argmax (subject to arm_ok), with a
  **well-formed preference tier** (§10.3).  Both > 0 enables.
- **`teb_bragg_veto_turn`** (deg, D4): inside the standard scan, an
  accepted **route-2** break with `turn < teb_bragg_veto_turn` is vetoed
  unless its SHORT-arm end is Bragg-consistent — peak ≥ 2.0×MIP-median
  AND contiguous >1.5×MIP hot extent from that end ≤ (peak − 1) cm/MIP
  over an 8 cm window (§10.3 recalibration of the §9.4b sketch).  R1
  (dip) accepts untouched.

**D2 (`vertex_bridge_retract`) is deferred**, per §9.5's own fallback
clause ("D3 covers it if D2 is not adopted"): D2 and D3 both ON would
double-modify the same 61681 junction, and the D3 smoke landing (§10.5:
61681 final main vertex 4.36 → 2.97 cm) covers the target.  If the owner
wants the last ~2.5 cm, D2 remains fully specified in §9.5 for a future
round.

### 10.1 Repro

```bash
cd toolkit && wcbuild          # freshness: local/lib/libWireCellClus.so 09:01 > last edit
./build/clus/wcdoctest-clus    # 210/210 cases, 2100 assertions

# compiled-config proof (full 15-stage pipeline + pr_display; the
# compile_all_cfg.sh sbnd_pr slice omits tagger_check_neutrino):
#   pre-change (cfg stashed) vs post-change knobs-off -> cmp BYTE_IDENTICAL
#   knobs-on (-S teb_chain_topology=true -S teb_r3_turn=18.0
#             -S teb_r3_hot=1.6 -S teb_bragg_veto_turn=30.0)
#   -> exactly the 4 new keys appear (once each)

cd sbnd_xin
# smoke (14 calibration/target events), knobs ON:
PR_JOBS=14 PR_EXTRA_STAGES=pr_display SBND_DL_VTX_HARVEST=true \
  SBND_TEB_CHAIN_TOPOLOGY=true SBND_TEB_R3_TURN=18.0 SBND_TEB_R3_HOT=1.6 \
  SBND_TEB_BRAGG_VETO_TURN=30.0 \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-mcp1k-pr90r4smoke2 data \
  172832 61681 320865 291064 64503 281214 283905 285443 59261 72586 \
  319611 59247 172942 64921
# full arms (same env; OFF arm drops the four SBND_TEB_* r4 envs):
#   work-{mcp1k,nuecc48,ncpi0}-pr90r4offb   (knobs off, gate vs harv3)
#   work-{mcp1k,nuecc48,ncpi0}-pr90r4on     (knobs on)
# gates: scripts/analysis/pr64/pr64_gate.py <a> <b>
# movers: scripts/pr90_movers.py work-*-pr90r4offb work-*-pr90r4on --tags harv3
```

(A first OFF arm `work-mcp1k-pr90r4off` was killed ~10 min in when the
§10.3 recalibrations forced a rebuild; its partial dir is dead weight —
`rm` was blocked by session permissions — and was replaced by
`work-mcp1k-pr90r4offb` on the shipping binary.)

### 10.2 Implementation notes

- `TwoEndBreakOptions` gains `bragg_veto_turn`, `r3_turn`, `r3_hot`;
  `TwoEndBreakResult` gains `route3`, `bragg_vetoed`, `veto_peak`,
  `veto_extent`.  The caller debug line now prints
  `routes=(r1,r2,r3) ... chain={} vetoed={} vpeak={}xMIP vext={}cm`.
- D1's degree census maps (pointer-valued) vertex descriptors but is
  never iterated — only insertion-order-independent aggregates (size,
  running max) are read.  Unique-longest uses strict `>`; ties decline.
- The R3 scan is pure measurement; the caller's break_segment path
  (cluster association, `kTwoEndBreakArm` arm flags, `kProtectedBreak`)
  is shared with R1/R2 unchanged.
- Config plumbing: `common/clus.jsonnet` (args + key-suppression),
  `sbnd/clus.jsonnet` (4 sites), `sbnd/wct-pr-perevt.jsonnet` (TLA args +
  passthrough), runner escapes `SBND_TEB_CHAIN_TOPOLOGY` /
  `SBND_TEB_R3_TURN` / `SBND_TEB_R3_HOT` / `SBND_TEB_BRAGG_VETO_TURN`.
- `doctest_clus_knob_defaults` checks all four keys' OFF defaults.

### 10.3 Two recalibrations forced by in-code measurement (v1 smoke → v2)

The first smoke (`work-mcp1k-pr90r4smoke1`, thresholds exactly as §9.4b/
§9.5 sketched: hot=1.8, veto extent ≤ peak) failed two events, both for
measurement-definition reasons, and both fixes replicate offline on the
§9.1 batch A/B scan-side dumps (`/home/xqian/tmp/pr90_dumps/`, dqdx
column in e/mm, MIP-median = 4300 e/mm):

1. **64503 escaped the D4 veto by 4%.**  The in-code profile (8 cm
   window from the short-arm end, contiguous >1.5×MIP extent) reads
   peak 3.44×MIP / extent 3.3 cm — `extent ≤ peak` holds, unlike the
   §9.4 offline sketch numbers (2.9/3.6).  Recalibrated to
   **`extent ≤ peak − 1 cm`**: all five sub-30° owner verdicts veto
   (64503 with 16% margin), and every bright-compact Bragg keeps ≥ 10%
   slack under the in-code definition (320865 3.14/1.94, 319611
   6.65/2.78, 59247 3.09/0.60, 172942 2.60/0.00, 72586 6.62/1.35) —
   though all keeps are turn-protected (≥ 30°) and never consult it.
   In-code sub-30° table: 291064 3.24/7.5, 64503 3.44/3.3, 281214
   4.13/7.3, 283905 4.80/7.8, 285443 1.71/0.0 — five vetoes.
2. **172832's R3 broke at the starved near-Michel t10 spike** (fit idx
   207, 3.5 cm from the Michel end, t10 = 27.4° — the same §3b
   starved-window jitter at 10 cm scale — outbidding the true 19–23.5°
   junction plateau).  The activity corroboration cannot guard it: the
   Michel's own EM charge reads 2.3–2.4×MIP there.  Fix = the same
   two-tier idiom as the shipped `turn_min_arm_frac`: a **well-formed
   preference tier** (both t10 windows fully achievable: ≥ skirt +
   10 cm = 13 cm from each end) runs first; only if it is empty does the
   unrestricted tier run.  172832's plateau is well-formed → tier 1
   picks t10=23.5° at s=108.6, refined to the 3.0×MIP activity spot at
   idx 178.  61681's genuine 54° corner sits 4.9 cm from the junction
   end (tier 1 empty: mid-track t10 is 5–7°) and correctly falls through
   to tier 2 — but its ±2 cm activity window only catches the 1.76×MIP
   *edge* of the activity blob, so the **operating point** (not code)
   moves `teb_r3_hot` 1.8 → **1.6**; the refine then lands idx 171,
   1.25 cm past the click (vs the 4.36 cm production junction).

### 10.4 Off-path proofs

- `wcdoctest-clus` 210/210 (2100 assertions) on the shipping binary.
- Compiled-config, full 15-stage(+pr_display) pipeline: pre-change vs
  post-change knobs-off `cmp` **BYTE_IDENTICAL**; knobs-on adds exactly
  `teb_chain_topology, teb_r3_turn, teb_r3_hot, teb_bragg_veto_turn`.
- Byte-identity run gate on the 1067-event manifest: §10.6.

### 10.5 Smoke, shipping code + operating point (chain=true, r3_turn=18, r3_hot=1.6, bragg_veto=30) — `work-mcp1k-pr90r4smoke2`

Main-vertex distance to the owner click (harv3 labels; 14/14 events ok):

| evt | class | production (harv3) | knobs ON | mechanism observed |
|---|---|---|---|---|
| 172832 | fix target | 20.35 cm | **0.49 cm** | chain admit, R3 tier 1: break idx 178 = the 3.0×MIP activity spot, 0.20 cm from the click |
| 61681 | fix target | 4.36 cm | **2.97 cm** | chain admit, R3 tier 2: break idx 171, 1.25 cm past the click |
| 291064 | D4 kill | 159.36 cm | **0.00 cm** | vetoed (27.4°, 3.24/7.5) — the §8.6 forgone fix recovered |
| 64503 | D4 kill | 0.00* | 43.61* | vetoed (26.7°, 3.44/3.3); *label is the reco-anchored click the owner ruled WRONG (§9.0) — the move off it is the sanctioned correction, Bee below |
| 281214 | D4 kill | 0.00 | 0.00 | vetoed (26.5°, 4.13/7.3); main vertex stays |
| 283905 | D4 kill | 0.00 | 0.00 | vetoed (26.8°, 4.80/7.8); main vertex stays |
| 285443 | D4 kill (preferred) | 0.00 | 0.00 | vetoed (25.5°, peak 1.71 < 2); main vertex stays |
| 59261 | tolerable keep | 83.34 | 83.34 | turn 30.0 = threshold, kept (owner: either way) |
| 72586 | keep | 297.40 | 297.40 | 33.2° ≥ 30, kept (labels far from break region, §9.4) |
| 320865 | round-2 fix | 1.22 | 1.22 | 33.1° ≥ 30, kept — round-2 result preserved |
| 319611 / 59247 / 64921 | keeps | 0.00 | 0.00 | 65.8/32.5/43.6° ≥ 30, kept |
| 172942 | keep | 29.61 | 29.61 | 34.9° ≥ 30, kept |

Every §9.4b owner verdict is honored; the only main-vertex movers are the
three intended fixes (172832, 61681, 291064) and the owner-sanctioned
64503 correction.

**QL-epoch caveat on 291064** (flagged by the concurrent
cathode-bundle-rescue session, doc 73 round 3): every number in this
round is on QL epoch `cb0805` (frozen root `work-mcp1k-cb0805`).  291064's
*current Q/L input contains a false cathode merge* (≈1370 APA1 points
re-materialised 33 cm inside TPC0 by a mis-matched cosmic T0) that their
upcoming containment veto will remove — so 291064's "0.00 cm" is a claim
about this QL epoch, and a future re-production can move it without any
teb_* knob changing.  None of the other 13 calibration events overlaps
that session's touched set (398115, 237798, 281165, 65289, 78242, 65053,
51128, 317427, 319913, 486907 — checked).

### 10.6 Full A/B round A (all three knob families ON) — gates PASS, movers FAIL for D1+D3

Arms (all on the shipping binary; the 16 ON-arm events that died in the
concurrent session's 09:20:41 build window — `libWireCellClus.so: file
too short`, evts 386794–388972 — were re-run on the codegen-identical
new lib and the nusel tsv re-merged):

- `work-mcp1k-pr90r4offb` / `work-mcp1k-pr90r4on` (1000/1000 each)
- `work-nuecc48-pr90r4off` / `-on` (48/48), `work-ncpi0-pr90r4off` / `-on` (19/19)

**Byte-identity gates (knobs off) — ALL PASS:**

- mcp1k: vs `work-mcp1k-harv3` 999/1000 (single mover = 320865
  `mabc-pr.zip`, exactly the shipped round-2 fix harv3 predates); vs
  `work-mcp1k-pr90on2` (= current production) **1000/1000 identical**.
- nueCC48 vs harv3: 48/48.  NCpi0 vs harv3: 19/19.

**Off-vs-on movers (pr90_movers, harv3 labels) — FAIL for the R3 family:**
47 movers > 0.05 cm in mcp1k (labels 473, compared 407): **21 ADVERSE, 8
toward, 3 away, 17 on**; nueCC48 0 movers; NCpi0 1 benign (18625,
R3 break elsewhere on the cluster, main vertex 0.00 → 0.35 cm).
Attribution:

- **D4-caused**: toward 291064 (159.36 → 0.00 — the §8.6 forgone fix,
  headline); ADVERSE 349461 (0.00 → 122.49! §10.7) and 64503
  (0.00 → 43.61 — *whitelisted*: its label is the reco-anchored click the
  owner ruled wrong, §9.0; the §8.6 v1 arm showed the same move).
- **R3-caused**: toward 59335 (66.00 → 0.00 AND cosmict 1→0 — the pr/48
  motivating event, un-tagged as cosmic), 172832 (20.35 → 0.49), 175808
  (9.39 → 0.93), 348515 (56.18 → 46.43), 285665 (89.14 → 85.68), 61681
  (4.36 → 2.97); **ADVERSE 19**, of which five are large relocations of
  owner-approved b1=0 vertices — 285531 (0 → 141.37), 283091 (0 →
  110.61), 281505 (0 → 94.06), 391260 (0 → 87.52), 66118 (0 → 66.60) —
  plus 170814 (0 → 49.57), 486247 (14.38 → 58.56), 314507, 349835, and
  ten 1–7 cm movers, including two cosmict 0→1 flips (487853, 405234).

**Root cause of the R3 failure** (measured, `/home/xqian/tmp/
pr90r4_r3_anatomy.txt`): the D1 admission is satisfied by *every*
simple-path chain with a second >4 cm prong — 126 clusters in 1000
events, 64 of which carry some interior spot with t10 ≥ 18° and ≥1.6×MIP
charge within 2 cm (delta rays, scatters, hairpins).  The ADVERSE and
toward classes overlap completely in every local scalar (t10 19–154°
both; 2–3-segment chains both; second-prong 3–99 cm both; short-arm
3–21 cm both): what separates them is only whether production was
already right (b1 = 0) — which the algorithm cannot know.  The break
itself is often harmless (17 "on" events), but the kProtectedBreak
vertex competes in main-vertex determination and sometimes wins.  This
is the §8.6/§4 lesson a third time, now for D3: *no single local
geometric/charge signature separates a true interior junction from an
energetic delta ray*; selection (not refinement) is again the wrong
lever (pr/89 round 4/5 finding).

### 10.7 Decisions after round A

1. **D1 (`teb_chain_topology`) and D3 (`teb_r3_*`) stay DEFAULT OFF —
   not flipped.**  Net-negative live: 19 ADVERSE vs 6 toward on labels.
   The knobs remain in the code (byte-identical off, gate-proven) for a
   future vertex-anchored redesign (§9.5 D2 — which cannot create
   competing far vertices by construction — or an R3 gated by the
   scorer/DL side rather than admission-side).  172832 and 61681 stay
   unfixed in production for now; the R3 toward evidence (59335!) is
   recorded for the owner.
2. **D4 (`teb_bragg_veto_turn`) needs a near-end scope before it can
   flip**: unscoped, it vetoed 349461's healthy mid-track break (turn
   29.2°, shorter arm 69 cm — one of §8.3's three "healthy" route-2
   breaks) and moved its owner-anchored vertex 122 cm.  The §9.4b
   calibration set is exclusively near-end breaks (short arms
   4.3–6.1 cm), where "is the short piece a real second particle with
   its own Bragg to its tip?" is the question the end profile answers; a
   69 cm-arm break's distant end profile says nothing about its
   junction.  Fix: the veto is evaluated only when the break's shorter
   arm < **15 cm** (2.5× above the largest calibration kill arm, below
   172942's 18.4 cm — itself turn-protected).  Revalidation as a
   D4-only ON arm: §10.8.

### 10.8 D4-only revalidation (near-end scope, `teb_bragg_veto_turn=30` alone) — PASS

Rebuild (lib `build/clus/libWireCellClus.so` 09:42:58 — note the M1
correction now in shared memory: jobs map the BUILD tree, freshness
proofs stat `build/<pkg>/`, and `./wcb build` alone swaps the lib under
a running campaign), `wcdoctest-clus` 210/210.

- **Smoke** (`work-mcp1k-pr90r4smoke3`, 14 events): five kills vetoed
  (unchanged from §10.5); **349461 break RESTORED** (out of scope, main
  vertex 0.00) and 278420 untouched; every keep unchanged.
- **Full arms** (`work-{mcp1k,nuecc48,ncpi0}-pr90r4on2`, all clean):
  nueCC48 48/48 and NCpi0 19/19 **byte-identical** off-vs-on (zero
  footprint); mcp1k **995/1000 identical, movers EXACTLY the five
  vetoed events** {281214, 283905, 285443, 291064, 64503}, all
  `mabc-pr.zip` only.
- **Asserted invariants** (per the concurrent session's "check the
  object the code modifies" lesson): (1) every veto line shows a
  near-end candidate (turn indices 7/8 of 70–165 cm segments; 64503 idx
  68/50.7 cm — calibration arms 4.6–5.5 cm); (2) the complement — every
  event with NO veto line is byte-identical to OFF, **including 349461**
  (the §10.7 regression, now among the 995); zero route-3 breaks, zero
  chain admissions.
- **Label-scored movers: exactly 2.** 291064 toward, 159.36 → 0.00 cm
  (numu 1.749 → 3.171); 64503 0.00 → 43.61 cm against its
  owner-invalidated label (§9.0) — the sanctioned correction (numu
  −0.495 → 0.841, Enu 420 → 256).
- **Non-events actively tested**: 281214/283905/285443 main vertex
  0.00 → 0.00 (numu 1.864→1.831 / −2.888→−2.131 / 1.079→1.750; Enu
  728→678 / 408→389 / 839→629 — the segmentation change feeds
  kinematics; 285443's −25% Enu is the largest and its veto is the
  owner's preferred side, §9.4b).

### 10.9 Flip decision

**`teb_bragg_veto_turn = 30.0` SBND PRODUCTION ON** (owner request:
"if validated, turn them on for SBND as default" — D4 is the validated
subset).  `teb_chain_topology` / `teb_r3_turn` / `teb_r3_hot` ship
DEFAULT OFF, not flipped (§10.6/§10.7 net-negative live).  Toolkit
commits: fbcf068c (knobs, DEFAULT OFF) + the cfg flip commit; the
`common/clus.jsonnet` plumbing rode in the concurrent session's
17a9929a (attribution note in fbcf068c).

### 10.8b Kinematic verdicts on the vetoed five (concurrent-session cold review item)

A vertex that does not move is not an event that does not change: D4
merges segments, so PID/energy move with the vertex sitting still.  Per
event, from the `kine` blocks (off → on2):

- **285443** (Enu 839 → 629, −25%): the change is one number — the
  muon, 408 → 198 MeV — and its `kine_energy_info` flips 0 → 1: with
  the spurious break removed, the 72.7 cm track is treated as a
  contained stopping muon and gets the RANGE-based energy, and 198 MeV
  is exactly the range expectation for 72.7 cm.  The 408 was the
  broken-topology artifact.  The −25% IS the fix.
- **281214** (Enu 728 → 678): the 258 MeV "electron" (the broken-off
  fragment read as an EM shower, which paired into a FAKE pi0:
  `pio_flag` 1 → 0, `pio_mass` 132 → 0) becomes a 207 MeV muon of the
  unbroken track.  Removes a spurious pi0 from the record.
- **283905** (Enu 408 → 389): two muon pieces (272 + 30) become one
  283 MeV muon; −19 MeV is merge bookkeeping on the owner-ruled ("should
  not break") topology.
- **291064** (Enu 615 → 541): vertex relocation to the owner click
  (159.36 → 0.00) re-anchors the energy sums.
- **64503** (Enu 420 → 256): the 35 MeV "pion" (the broken stub) is
  absorbed into the muon (136 → 147) and `kine_reco_add_energy` drops
  245 → 106 with the 43.6 cm vertex relocation.  **Reframed per the
  cold review: this event is UNSCORED, not adverse** — its harv3 label
  is the reco-anchored click the owner ruled wrong (§9.0), so the
  ledger reads 1 toward + 1 unscored-resting-on-hand-scan, not 1/1.
  Note its numu score crosses zero (−0.50 → +0.84): under a
  0-threshold convention that is a selection-state change (it does not
  cross the WCP-conventional 0.9); flagged for the owner's Bee pass.

**Correction to the §10.5 QL-epoch caveat**: the concurrent session
hash-verified that `work-mcp1k-cb0805`'s 291064 Q/L input is CLEAN
(member hash identical to their knobs-OFF arm; worst cathode overshoot
−0.10 cm) — the false merge existed only in their experimental knobs-ON
arm.  The 159.36 → 0.00 result stands on its own; the caveat is
conditional strictly on a FUTURE re-production adopting their
containment veto.

Base-size note (also from the cold review): the flip evidence is 2
scored movers out of 407 labels + 3 actively-tested non-events + the
kinematic attributions above — a judgement call on a thin base, stated
as such, not a measurement.

The general principle both sessions converged on today (record it once,
symmetric): **a metric moving is not evidence of harm any more than a
metric staying still is evidence of safety — both need the mechanism.**
dvtx = 0.0 hid three real kinematic changes here; Enu −25% looked like a
regression and was the fix; and the concurrent session's four passing
gates coexisted with a cluster sitting 33 cm inside the wrong TPC.
Neither a moved nor an unmoved number means anything until the question
"what physically happened?" has a per-event answer.

# doc pr/90 — unbroken kink, wrong neutrino vertex (mcp1k 320865, 172832, 61681)

**Status: investigation only. Nothing is implemented, no knob is added, no config
is touched, nothing was rerun.** The owner reviewed a Bee link of these three
events and asked for a root-cause investigation and a proposed solution. §6
lists the proposals and the gate each would need before any code change.

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
observed here. A `WCT_DET_DEBUG=2` single-event rerun (§6) would confirm it
directly.

### 3b. 320865 — the break DOES fire, but lands at the wrong point

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

**An attempt to explain the mechanism did not succeed and is reported
honestly rather than asserted.** Reconstructing the pre-break fit array by
concatenating the two post-break segments' points from `calib-pr-evt320865.json`
(`segments[13001].points` + `segments[13002].points[1:]`, oriented via their
shared `start_vertex_id`/`end_vertex_id` = vertex 13002, the current wrong
main vertex) and re-running the exact `segment_wide_turn_angle` scan with
production defaults (`teb_turn_skirt=3cm`, `teb_turn_baseline=35cm`,
`arm_ok` gate `min_arm=1.8cm`/`min_arm_pts=4`) gives a **global argmax at
s≈53 cm with angle≈33°** — correctly near the true corner, not at index 8
(which scores only ≈12° in this reconstruction). The reconstruction also
doesn't reproduce the log's reported `turn=38.1deg` value at any index.

This means either (a) the `calib-pr` JSON's `segments[].points` is not
exactly the same array as the live `seg->fits()` TrackFitting used at that
point, or (b) the acceptance-tier stopping-template scoring
(`do_track_comp` over `accept_range`, not the raw turn angle) is what
actually selected index 8 over the higher-turn candidates near the true
corner, with the turn-angle argmax only entering as one of several rejected
proposals. **Distinguishing these needs an instrumented single-event rerun
(§6), not further static analysis** — the doc stops here rather than
guessing.

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

**For the 320865 class (break fires, wrong location):** the mechanism itself
is unresolved (§3b). The concrete next step is an instrumented rerun before
any fix is designed:

```bash
cd sbnd_xin
WCT_DET_DEBUG=2 PR_EXTRA_STAGES=pr_display PR_JOBS=1 SBND_WCT_LOGLEVEL=trace \
  ./run_pr_chain_batch.sh work-mcp1k-cb0805 work-mcp1k-kink90 data 320865
```

(fresh `out_root`, `reality=data`, per the pr/83 precedent for this exact
instrumentation). That would show whether `do_track_comp`'s acceptance-tier
scoring — not the raw turn angle — is what actually selects candidate index 8
over the higher-turn candidates near the true corner, which is the live
hypothesis from §3b.

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
- Shipping the bare wide-turn-angle threshold from §4 as a knob — the
  footprint (16–45% of events depending on threshold) and the span-guard's
  event-dependent sign flip make it unsafe without much more work.

One additional untested but plausible contributor, flagged per M15 rather
than picked silently: the WCT-only `flag_switch` stability guards in
`segment_search_kink` (`PRSegmentFunctions.cxx:920-932`, requiring
`num_p1>=9` points and `length2_1>3cm`) are a documented divergence from the
prototype that make WCT strictly less likely to break near a segment end —
and 61681's true corner is ~2 cm from an end. This was not tested this
session; a targeted single-event instrumentation of `segment_search_kink`
itself (separate from `break_two_end_dqdx`) would be the way to check it.

## 7. Summary table

| evt | mechanism | status | proposed next step |
|---|---|---|---|
| 320865 | `two_end_break` fires, breaks at wrong index (mechanism unresolved) | segmentation failure | `WCT_DET_DEBUG=2` rerun (§6) |
| 172832 | `two_end_break` gate declines (`n_long != 1`) | segmentation failure | new default-OFF gate-widening knob + its own A/B (§6) |
| 61681  | `two_end_break` gate declines; but DL vertex already near-exact | refinement-scale, not segmentation | lower priority; possibly moot once DL/topology tuning (pr/89) improves acceptance near this geometry |

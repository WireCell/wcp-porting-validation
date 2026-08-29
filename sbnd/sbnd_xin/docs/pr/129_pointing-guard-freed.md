# doc pr/129 — the pointing test: does this track aim back at the neutrino vertex?

**Status: IMPLEMENTED, DEFAULT OFF — validation in progress.**
Toolkit: `clus/inc/WireCellClus/PRSegmentFunctions.h`, `clus/src/PRSegmentFunctions.cxx`,
`clus/src/NeutrinoKinematics.cxx`, `clus/inc/WireCellClus/{NeutrinoPatternBase,TaggerCheckNeutrino}.h`,
`clus/src/TaggerCheckNeutrino.cxx`, `clus/test/doctest_clus_knob_defaults.cxx`,
`cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet`.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# the three-event decision set, knob wide open (behaviour = production, tape printed)
./scripts/pr129_pointing_arms.sh meas
# the operating point
IMPACT=20 MISS=30 ./scripts/pr129_pointing_arms.sh on
# dual-manifest gates
./scripts/pr129_arms.sh 141 off 0 ; ./scripts/pr129_arms.sh 98 off 0
python3 scripts/pr85_hash_gate.py work-pr128r1-on141-<s> work-pr129r1-off-<s>
```

## 1. Where this came from

doc pr/128 shipped four completeness knobs. Chasing its deferred items produced
a Bee probe (`bee/pr129probe/`) on three events whose class-A objects the
shipped continuation predicate rejects. The owner's verdict, 2026-08-29:

> "all three should be reject, overclustering, so not counting in the enr
> eneryg, OK tobe in PR, I guess."

That killed the vertex-proximity arm outright (399118's 481 MeV proton sits
4.9 cm from the ν vertex and was still rejected) and drew a line worth keeping:
**this class may appear in the PR tree, but must not enter `kine_reco_Enu`.**

It also exposed a **pre-existing defect**. 393505's rejected cluster 15 is
partly *already counted*: 268.70 MeV of it enters Enu through
`kine_count_guard_freed` (doc pr/123 r2, SBND ON since 2026-08-28). The
segments chain end-to-end at 0.0 cm into one continuous **295.0 cm** object
that never comes within 68.9 cm of the vertex — twice the extent of 72786's
confirmed cosmic. Not caused by pr/128; found by it.

## 2. The population is three objects

`kine_count_guard_freed` counts **exactly 3 objects across all 239 events of
both manifests, 710.66 MeV**. There is nothing else to find:

| event | KE (MeV) | len (cm) | owner verdict (Bee `bee/pr129gf/`) |
|---|---|---|---|
| 171572 | 304.75 | 125.1 | **KEEP** — "784.9 MeV should be the right energy" |
| 393505 | 268.70 | 108.5 | **DROP** — "should be the lower energy" |
| 94392 | 137.21 | 46.8 | either — "is OK" |

**94392 costs nothing either way.** With the pool off, the identical segment is
counted instead by pr/128's `kine_count_near_cross_cluster` (kink 13.3°, gap
0.00) and Enu is unchanged at 1142.5. The two pools are mutually exclusive by
construction, so the legitimate continuation class is already covered by a
predicate the owner approved.

Because 171572 must be KEPT, **a blanket flip of `kine_count_guard_freed` is
ruled out** — it would cost 410.4 MeV the owner just called correct.

## 3. Two things that do NOT work (measured, negative)

**A geometric bound.** Cluster extent and ν-vertex distance, against every
object with a known owner verdict:

| object | verdict | extent (cm) | d_vtx (cm) |
|---|---|---|---|
| 55740 | APPROVED | 131.3 | 48.1 |
| 392901 | APPROVED | 106.5 | 34.2 |
| 105074 | APPROVED | 442.8 | 0.5 |
| 393505 | REJECTED | 296.7 | 68.4 |
| 399118 | REJECTED | 102.6 | 4.3 |
| 318769 | REJECTED | 44.7 | 113.4 |

Neither column separates them: the rejected 399118 is *smaller* in extent than
the approved 55740 and *closer* to the vertex than anything approved; the
rejected 318769 has the smallest extent in the table. The segment-to-extent
ratio fails the same way (approved 392901 2.77 vs rejected 393505 2.73). Same
shape as pr/128's proximity finding: **over-clustering is a judgement, not a
distance.**

**A vacuous config proof.** A bare `wcsonnet <file>` compile of
`wct-pr-perevt.jsonnet` never instantiates the tagger node — without the
`pipeline_names` TLA the output contains *none* of the kine keys, including
long-shipped ones, so an off-vs-off byte-diff of that output proves nothing.
Confirmed by counting: `kine_count_orphan_tracks` = 0 occurrences. The proof
that counts here is the output-level hash gate (§6).

## 4. What does work — the owner's discriminator

> "the key difference is the direction, if the direction of the track is point
> to the main vertex, it is more likely to be part of neutrino. For
> overclustering they are generally not point to neutrino vertex."
> — owner, 2026-08-29

`segment_vertex_pointing(seg, vtx, dir_window = 15 cm)` measures, at the
candidate's own end nearest the vertex, using the **same windowed inward
direction** as `segment_continuation_geometry` (so a scattered track is judged
by its direction *near the vertex*, not by a chord across its whole length):

- **`impact`** — perpendicular distance from the vertex to the infinite line
  through that end. "By how much does the track's line miss the vertex."
- **`miss_deg`** — 0 when the track runs straight *out* of the vertex, 90 when
  it runs across, **>90 when the vertex is IN FRONT** of the near end, i.e. the
  track heads toward it rather than away — which a daughter never does.
- `d_vtx` — min distance from any fit point to the vertex (reported, not cut
  on: a track brushing the vertex with its *middle* has a small `d_vtx`, and
  the two terms above are what reject it).

Physical basis: a daughter leaves the vertex and travels outward, so even when
only a *fragment* survives far from the vertex, that fragment's line still aims
back at the vertex and has the vertex behind it. An over-clustered cosmic that
merely passes nearby does neither.

### Measured, from the C++ itself (`work-pr129pt-meas-mcp2k`)

| event | KE | d_vtx | **impact** | **miss°** | owner |
|---|---|---|---|---|---|
| 171572 | 304.75 | 20.36 | **4.16** | **11.8** | KEEP |
| 94392 | 137.21 | 44.49 | **6.59** | **8.5** | keep |
| 393505 | 268.70 | 74.37 | **68.67** | **67.4** | DROP |

A **factor-10.4 margin** on impact (6.59 → 68.67) and 5.7× on the angle
(11.8 → 67.4). The measurement arm runs the knob wide open (impact 1000 cm,
miss 180°) so every candidate is still counted: Enu is identical to production
on all three events, which is what makes the tape trustworthy — it reports
without deciding.

Independent confirmation the thresholds were not fitted to: **399118**, from a
different pool, has impact 5.2 cm — on impact alone it looks like a daughter —
but `miss_deg` 151.6 says the vertex is *in front* of it. The geometry and the
owner's rejection agree, on a row the cuts were not tuned against.

**Operating point: `impact ≤ 20 cm` and `miss_deg ≤ 30°`** (30° is the same
kink tolerance pr/128 shipped). 3.0× headroom above the largest keep, 3.4×
below the drop.

### Scope, and why this is a fix and not a carve-out

The test is applied to the **guard-freed pool only**. 55740, which the owner
approved, has impact 56.3 / miss 65.8 and would fail it — but 55740 is admitted
by the *continuation* rule, in a different pool with its own geometric test.
The guard-freed pool is **the only pool with no geometric admission test at
all**: its predicate is the `kPass4GuardFreed` flag and nothing else. This
gives it the admission test every other pool already has. It is deliberately
not applied globally; §3 is why.

## 5. The knobs

| key | type | C++ default | meaning |
|---|---|---|---|
| `kine_guard_freed_impact` | double (cm) | **0** | 0 ⇒ no test ⇒ byte-identical. >0 arms the pointing test. |
| `kine_guard_freed_miss_deg` | double (deg) | 90 | only consulted when the impact cut is armed |

Legacy path textually untouched: the test sits behind
`if (m_kine_guard_freed_impact > 0 && main_vertex)`. jsonnet uses the
key-suppression idiom gated on `!= 0`. Runner seats `SBND_KINE_GF_IMPACT` /
`SBND_KINE_GF_MISS_DEG` (empty ⇒ no TLA ⇒ job default). The PF side is
untouched — `pf_orphan_guard_freed` stays ON, so the object still appears in
the PR tree, which is exactly "OK to be in PR".

## 6. Validation

- [x] `wcbuild` clean; freshness proof (lib 16:12 > newest source 16:11).
- [x] `./build/clus/wcdoctest-clus` — 235 cases / 2528 assertions pass,
      including the two new default pins.
- [x] Knob ON reproduces **all three owner verdicts exactly**
      (`work-pr129pt-on-mcp2k`): 171572 784.9 (kept), 393505 940.4 → **566.1**
      (dropped), 94392 1142.5 (unchanged).
- [ ] Dual-manifest OFF gate (`work-pr129r1-off-*` vs `work-pr128r1-on141-*`),
      `pr85_hash_gate.py`, counts to be quoted.
- [ ] Dual-manifest ON run: blast radius expected to be **exactly 393505**,
      since the pool's entire population is three objects and two of them pass.
- [ ] Movers / nusel / sentinels.

## 7. Open

- The pointing test is scoped to one pool by evidence (§3), not by principle.
  Whether the other pools would benefit is unmeasured and deliberately out of
  scope here.
- `clus/src/NeutrinoKinematics.cxx` also carries an **uncommitted** env-gated
  probe (`WCT_KINE_GUARDFREED_PROBE`) from another session, referencing a doc
  pr/130. Not authored here and not committed by this round.

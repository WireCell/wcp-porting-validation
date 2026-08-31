# doc pr/119 — EM clustering round 3: over-clustered bucket, foreign-cluster member expel

**Status: CLOSED, MEASURED-ONLY (2026-08-28). The expel predicate is killed
by measurement (the pr/118-P2 precedent): no knob ships, no behavior
changes. The byte-neutral probe + census ship as measurement
infrastructure; §4 records the negative result and §7 what it proves.**

Owner directive (2026-08-28): after pr/117 (admission) and pr/118 (merge)
only ever ADDED charge to showers, run the mirror round — a measure-first
member-removal pass for the over-clustered bucket, "for the rest validation,
same as previous round" (byte-neutral probe → truth-joined census →
default-OFF knob → pr/118 validation bar; production flip pre-authorized if
validation passes; Bee A/B links for any doubt), update the md, commit and
push.

## 0. Repro block

```bash
# toolkit build (probe + knobs), freshness proof, doctests
cd /home/xqian/toolkit-dev && direnv exec . bash -c \
  'cd toolkit && ./wcb build --notests -p && ./wcb install --notests -p'
ls -la local/lib/libWireCellClus.so   # mtime must postdate the last edit
toolkit/build/clus/wcdoctest-clus

# arms (98 events, 4 samples; fresh names, M13)
/home/xqian/tmp/pr119_arms.sh off0 0                     # new binary, no env
/home/xqian/tmp/pr119_arms.sh dbgA 1                     # WCT_SHOWER_EXPEL_DEBUG=1

# probe byte-neutrality + binary-off gates
python3 scripts/pr85_hash_gate.py work-pr118r1-flipchk-<s> work-pr119r1-off0-<s>; echo rc=$?
python3 scripts/pr85_hash_gate.py work-pr119r1-off0-<s>   work-pr119r1-dbgA-<s>; echo rc=$?

# sidecars + census
python3 em_display/prep_pr117.py --tag 119dbgA work-pr119r1-dbgA-{mcp1k,mcp2k,ncpi0,nuecc48}
./scripts/pr119_expel_census.py --prepdir em_display/emprep-119dbgA \
    --groups-tsv docs/pr/pr119-expel-groups.tsv \
    work-pr119r1-dbgA-mcp1k work-pr119r1-dbgA-mcp2k \
    work-pr119r1-dbgA-ncpi0 work-pr119r1-dbgA-nuecc48
```

Baselines: Q/L roots `work-{mcp1k,mcp2k,ncpi0,nuecc48}-grp0825` (14/17/19/48
= 98 events, `em_display/em114-manifest.tsv`); production PR arms
`work-pr118r1-flipchk-<sample>` (pr/118 flip = current production); score
baseline `docs/pr/pr118-onT-score.tsv` + `emprep-118onT`.

## 1. The target, measured before any design

The hand scan (emscan-0827) carries **29 OUT marks — segments a shower
wrongly HOLDS — over 28 distinct segments in just 6 events**:

| event | sample | OUT | IN | scanner note |
|---|---|---|---|---|
| 142421 | ncpi0 | 10 | 33 | "OUT segments can form a separate gamma, and then they could be a pi0" |
| 314838 | ncpi0 | 7 | 12 | "The OUT segments should be a separate gamma cluster, then form a pi0 likely" |
| 269774 | nuecc48 | 5 | 9 | — (all pass4_angle, 24–27° at 49–65 cm: the only pure leaf-expel event) |
| 76346 | mcp2k | 4 | 16 | — (one OUT is shower 40030's own ROOT) |
| 47212 | mcp2k | 2 | 0 | seg 2103 marked OUT of two showers; root of one, stem_backfill'd into the other at ~147° |
| 54332 | mcp2k | 1 | 17 | "overclustering a track ... 16014" (32.3 cm track, q_extra 2.89e6) |

Admitting-pass distribution over the 29 (labels `marks_detail.absorbed_by`):
pass4_angle 13, from_vertices(walk) 7, pass3_cone 3, shower-root 3,
stem_backfill 2, examine_shower_1_tmp 1.

Three structural facts drive the design:

1. **Seed-out dominates.** 5 of the 6 events have the shower's own root on
   the wrong side — a leaf-member drop (the pr/99 `drop_ghost_member` shape)
   covers only evt269774.
2. **Cross-cluster is the boundary.** On all 6 events the wanted and
   unwanted sides live in different imaging clusters (pr/115 §16.4: "what
   those 6 need is a guard on the cross-cluster absorb"). And evt314838's
   two pass4 OUT marks sit at 2.7°/4.3° ON the shower axis — a collinear
   second gamma that no angle cut can separate; cluster grouping can.
3. **Ledger and ceiling.** Under the em117 cross-run scorer the wrongly-held
   charge is Σq_extra = 1.449e7 over 8 impure shower rows; a perfect expel
   moves the over-clustered bucket median 0.741 → 0.862 (both-bucket 0.740
   → 0.817). Per-event ceilings: 47212 0.970→1.000, 54332 0.686→0.799,
   142421 0.614→0.701, 314838 0.741→0.862, 76346 0.596→0.817, 269774
   0.920→0.952.

**Do-not-touch constraint:** evt469665's two "extra" segments (and the
merged fragments of 423981/281485) are pr/118 `shower_merge_relax_continuity`
products, not truth — an expel pass that undoes them undoes a validated win.
The predicate must retain them by construction (§3).

## 2. Design: split at cluster boundaries, anchored at the root

Reframed from "expel bad members" to **"split a multi-cluster shower along
upstream-cluster boundaries, anchored at the component containing the start
segment"**:

- **Anchor group** = the view-connected component holding the start segment,
  restricted to the start segment's cluster C0. Never expelled ⇒ no
  re-rooting machinery; one code path covers both the leaf case (269774)
  and the seed-out case (in those events S keeps the OUT side and the IN
  side spawns as a new shower — the charge-weighted cross-run scorer
  credits the truth gamma to whichever reco shower holds its charge).
- **Foreign group** = each remaining (cluster, view-component) member set.
  A surviving foreign group is removed from S's view (clouds rebuilt — they
  are add-only and feed kine_charge) and **spawned as a NEW shower** (no
  graph deletes: the charge is real, only the ownership is wrong; and a
  view-removed cross-cluster segment that joins no shower can vanish from
  the PF tree entirely). The pass seat is after the pr/99 ghost block and
  before the pi0 finders, so a spawned gamma is automatically a
  pi0-pairing candidate — exactly the scanner's ask on 142421/314838.
- 47212 dedup: a group already held by another shower's view is expelled
  but not spawned. 54332's absorbed track: same predicate; the spawn keeps
  its member-vote PID (pr/93 accept_pid_guard), so it comes back as a
  track-typed object rather than a forced e−.

## 3. Phase A — byte-neutral probe (`WCT_SHOWER_EXPEL_DEBUG`)

Toolkit commit: (pending). `clus/src/NeutrinoShowerClustering.cxx`:
`pr119_partition` (deterministic (cluster × view-component) partition,
stable vertex-index adjacency, shared verbatim with the future pass),
`pr119_group_endpoints` (exact group↔anchor junction argmin over fit
points), `pr119_probe_expel_groups` (per-shower `EXPEL_SHOWER`, per-group
`EXPEL_GROUP`, per-member `EXPEL_MEMBER` stderr lines), called at the exact
future knob seat — after detach/ghost, before hadronic tag, final kine
recompute and the pi0 finders. Group features: length/charge/qfrac,
median dQ/dx (MIP-normalized), max segment length, confident-non-e PID
count, multi-shower ownership, main-vertex touch, view links to the rest of
the shower, exact junction gap + the pr/118 continuity walk
(`Pr118Connector`) against the anchor, shower-axis angles at 30/100 cm to
the junction, group development angle, distance to main vertex, and a
dry-run of the spawn anchoring recipe (nearest non-group vertex,
0.8×main-dis preference, ≤80 cm conn-3).

Census: `scripts/pr119_expel_census.py` (forked from
`pr118_probe_census.py`). Two structural changes: the unit is the
(shower, group), and the truth join is charge-weighted cross-run matching
(the `em117_score.match_shower` recipe) — the pr/118 exact-key join returns
UNKNOWN on evt142421 (label key 7010, reco node 108104). Member classes
OUT / IN / OTHER / HOLD / UNKNOWN; group rollup OUT (out-charge ≥ 0.5) /
IN (≤ 0.05 with IN charge) / MIXED (predicate failure, reported).
The operating grid has **two axes only** (anti-overfit; 6 positive events):
`min_len` × retention-guard {on,off}, where the retention guard is the
pr/118 continuity test at its FROZEN production numerics (T1 gap ≤ 1 cm ∧
axis < 7.5°; T2 gap ≤ 8 cm ∧ axis < 7.5° ∧ qfrac ≥ 1.0 ∧ qmed > 5000) — the
mechanism that protects the pr/118 merge products.

### 3.1 Gate ledger (Phase A)

| gate | arms | result |
|---|---|---|
| binary-off (probe code in, env unset) | work-pr118r1-flipchk-* vs work-pr119r1-off0-* | PASS 196/196 (28+34+38+96); nusel cmp 0 diffs / 98 |
| probe byte-neutrality (env set) | work-pr119r1-off0-* vs work-pr119r1-dbgA-* | PASS 196/196 (28+34+38+96) |
| doctests | build/clus/wcdoctest-clus | 2442/2442 PASS (probe build) |

Sidecars: the dbgA arm is hash-identical to the pr/118 onT/flipchk arms
(both gates above), so the existing `emprep-118onT` sidecars describe this
reconstruction exactly and are what the census joins against (the arm ran
without `WCT_SHOWER_CONTENT_DEBUG`, so `emprep-119dbgA` is empty).

## 4. Phase A measurement — the predicate is dead

98 events, 1598 showers, 2385 groups, 4248 member rows
(`docs/pr/pr119-expel-groups.tsv`; census log repro in §0). Member truth:
24 OUT + 3 WRONGOWNER (76346's cross-owned marks: out of one scanned
shower, in another, both merged into reco 14059 — the doc 118 §7 hardest
class, now with its own census label) + 561 IN + 3622 HOLD. All 6 OUT-mark
events are seen by the probe. Two OUT marks are invisible to any grouping:
47212's seg 2103 and 54332's seg 16014 are their showers' own anchor roots.

**4a. The imaging-cluster granularity is far finer than the gamma boundary.**
2130 non-anchor groups (2056 foreign-cluster) across 98 events — big
cascades routinely span dozens of clusters (evt142421's scanned shower:
43 segments over 29 clusters; evt269774's: 64 segments over 35; evt423981's
shower 28 alone holds 15 foreign truth-IN groups). 257 foreign groups are
truth-IN: cross-cluster membership is the *normal* anatomy of a correctly
reconstructed SBND cascade, not a defect signature. The "split at cluster
boundaries" model therefore shatters real showers.

**4b. OUT-charge coverage caps at 3 of 6 events, and partially.**
q_out in foreign groups / total: 269774 **100%** (1.15e6), 142421 **33%**
(0.93e6 of 2.79e6 — the rest sits in the anchor: the root itself is OUT),
314838 **18%** (0.55e6 of 3.11e6 — same reason), 47212 **0%** and 54332
**0%** (anchor roots; 54332's absorbed 32.3 cm track is ncls=1 — same
cluster as the EM part, the boundary does not exist), 76346 wrong-owner
(two scanned gammas merged into one reco shower; no expel semantics apply).

**4c. No operating point separates.** Grid (min_len × retention-guard, the
plan's two sanctioned axes), fired counts OUT/WO/IN/HOLD:

| min_len | OUT | WO | IN | HOLD | OUT events covered |
|---|---|---|---|---|---|
| 0 cm | 10 | 1 | 252 | 1680 | 142421, 269774, 314838 |
| 2 cm | 3 | 1 | 95 | 533 | same 3 |
| 5 cm | 3 | 1 | 58 | 231 | same 3 |
| 10 cm | 2 | 1 | 33 | 113 | 142421, 314838 |

Even the tightest point fires on 33 truth-IN + 113 unscanned-HOLD groups
across 64 events for 2 true positives. The plan's structural downscope is
vacuous: **all 2056 foreign groups have nlinks=0** (cross-cluster absorption
never shares a view vertex), so view-connectivity discriminates nothing.

**4d. Within a single shower, OUT and IN groups are feature-identical.**
evt314838's scanned shower holds its one OUT group (len 12.1, gap 8.1 cm,
axis 13.1°, dqdx 0.87 MIP) alongside six truth-IN groups spanning
len 14.8–46.4, gap 11.8–57.2, axis 13.8–19.4°, dqdx 0.65–1.01 — every
feature interval overlaps. evt423981's truth-IN groups reach axis 29–30°
and gap 95 cm (the scanner wants them held). The local features cannot
carry the decision; the scanner's own notes on both π0 events decide by a
*global* hypothesis ("the OUT segments should be a separate gamma cluster,
then form a pi0 likely"), not by local geometry.

**4e. The retention guard cannot even protect the pr/118 wins.** Of the
pr/118 merge products, 281485's stub and 469665's 12.1 cm group pass the
frozen-continuity retention test, but 469665's second merged stub (grp 1:
gap 12.2 cm *to the anchor*, though it was merged at ≤8 cm to the nearest
member) fails it — the group→anchor junction is not the fragment→shower
junction. A shipped version would risk undoing a validated merge.

**Verdict: killed by measurement** — plan kill criteria (a) granularity,
(b) IN false-expels at every covering point, (c) HOLD churn ≫ O(10), and
(e) retention-guard breakage all fire. Per the plan's anti-overfit rule
("if the census cannot separate with those two axes, that is the kill
signal — do not add axes"), no further feature-fishing was done.

## 5. Phase B — not built

No knob ships. The planned `shower_expel_foreign` family was never coded:
unlike pr/118's P2 (which was already built when measurement killed it and
therefore shipped OFF/"not selected"), pr/119's measurement completed
before Phase B started, so there is nothing to ship. The toolkit change of
this round is the byte-neutral probe only (`pr119_partition`,
`pr119_group_endpoints`, `pr119_probe_expel_groups` under
`WCT_SHOWER_EXPEL_DEBUG`, seated after the pr/99 ghost block), plus this
census. Production output is untouched — both §3.1 gates PASS.

## 6. Validation

Not applicable — no behavior change exists to validate. The §3.1 gate
ledger (196/196 twice, nusel 0/98 diffs, doctests 2442/2442) is the
complete claim: with or without the probe env, output is byte-identical to
pr/118 production.

## 7. What this round proves, and where the residual actually lives

pr/118 measured (from the under-clustering side) that the detached-fragment
flanks fail every local pairwise test. pr/119 now measures the mirror (from
the over-clustering side): **wrongly-held member groups are locally
indistinguishable from correctly-held ones** — same gaps, same axis angles,
same dQ/dx, same connectivity (§4d) — and half the OUT charge is not even
separable in principle by any membership operation because it sits at the
shower's own root (§4b). Three routes remain, in recommended order:

1. **π0-hypothesis-guided split** (covers 142421 + 314838, the two largest
   remaining over-clustering ceilings, 0.614→0.701 and 0.741→0.862): both
   scanner notes justify the OUT side as "a separate gamma, then a pi0".
   The probe seat sits directly before `id_pi0_with_vertex`; a candidate
   split could be accepted only when the resulting γγ pair lands in a π0
   mass window — global context replacing the dead local features. This
   crosses into the π⁰ thread the owner has deferred, so it is
   **owner-gated**, and with only 2 positive events it needs the expanded
   scan set first.
2. **More truth** (the ~141-event unscanned c-set): every discriminator
   this campaign has killed died on 2–45 positives. A second scan tranche
   both de-risks the pr/118 threshold overfit and gives any cascade-level
   model (tree assignment or a trained pair/group classifier) enough
   statistics to be honest.
3. **The wrong-owner / seed classes** (76346, 47212, 54332, plus doc 118's
   409634/415278): these need the shower-*construction* passes to stop
   absorbing across gamma boundaries (a guard at pass3_cone/pass4_angle/
   stem_backfill admission time, judged by the same future cascade
   context), not a post-hoc membership edit — the post-hoc edit provably
   cannot see them (§4b).

Open ledger otherwise unchanged from doc 118 §7 (54332's uncharged-connector
stub, 122660's long fragments, CC formulation, residual under-clustering on
the pr/118 movers).

**Follow-on**: doc pr/120 executed the admission-time arm of route 3 — the
backward-stem guard (`stem_backfill_back_guard`, SBND ON) rescues 47212 and
releases 281567's scan-noted stem; its census also showed the pass3_cone
"backward" label angles were scanner-start-override artifacts (the
wrong-owner class stays cascade-context territory) and moved 54332 to the
recognition thread (kShowerTopology mis-flag on a straight track).

## 8. Files

- toolkit (`apply-pointcloud`): `clus/src/NeutrinoShowerClustering.cxx` —
  pr/119 probe block (file-statics after the pr/118 probe; call site after
  the pr/99 ghost seat). Byte-neutral, env-gated, no knob.
- wcp-porting-img (`main`): this doc; `scripts/pr119_expel_census.py`;
  `docs/pr/pr119-expel-groups.tsv` (2385 group rows with truth);
  `docs/pr/118_em-clustering-round2.md` §7 forward pointer.
- Arms on disk (untracked): `work-pr119r1-{off0,dbgA}-<sample>`; arm
  launcher `/home/xqian/tmp/pr119_arms.sh`; census logs
  `/home/xqian/tmp/pr119_census{1,2}.log`.

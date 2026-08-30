# pr/130 item 1b part 6 — the 98-set's own 22 condemned segments

**Status: MEASURED. Scoring and census reading only — no arm launched, no
knob, no C++, no config, nothing shipped.** Closes the open item
"*The 98-set's own 22 condemned segments, never analyzed*" left by
[`pr130-qextra-refresh.md`](pr130-qextra-refresh.md).

Part 1 of that doc computed the affirmative q_extra pool on **both** standard
manifests, then analysed only the 141-set's 22 segments and disposed of the
other side in one sentence:

> The 98-set adds a further 22 segments / 7.056e6 on its own labels. Those 44
> segments are the target list.

Nobody ever opened the second 22. This doc opens them, and asks of the 98-set
the same questions Parts 1–3 asked of the 141-set. The short answer is that
they are **not the same failure**, and that treating the 44 as one target list
was wrong.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./scripts/pr130_qextra98.py > docs/pr/pr130-qextra-98set.txt
```

Reads the two item-1 score tables, `em_labels/emscan-0827` and
`emscan-0828-agent5` (read-only, M13), the calib dumps, the absorb-census
`stdout.log`s of the arms already on disk (`work-pr130r1-probe{98,141}-*`) and
the Part-4 flip arm pair (`work-pr130r1-{g1off,gs1on}-*`). Nothing was
written over; both output files are new.

---

## 0. What the pool is, and how small its denominator is

**22 segments / 7.056e6 of charge, over three events** — 314838 (ncpi0),
142421 (ncpi0), 269774 (nuecc48). Affirmative q_extra means: the scanner put
an explicit **OUT** mark on the segment for that shower, and the
reconstruction still holds it.

Two caveats belong before any number, because the 141-set write-up made the
matching statements about itself and this pool is the weaker of the two:

- **`emscan-0827` carries 29 OUT marks in the whole scan** (against 246 IN
  marks — 11% OUT, the IN-heavy habit Part 1 documented). So these 22 are
  ~76% of the scan's entire OUT population, not a sample of one. This is the
  incidental by-product of a scan that was asked to list what *belongs*.
- **Concentration here is vacuous.** With three contributing events, "top-4
  holds 100%" is true by construction — the same empty statistic Part 1 threw
  out for the 141-set's "top-10 = 100%". There is no concentration claim in
  this doc.

What survives both caveats is the absolute pool and what is *in* it.

## 1. The finding: this is EM-to-EM mis-partition, not track contamination

The scanner wrote a note on two of the three events. They are the primary
evidence and they say the same thing:

| event | sample | q_aff | the scanner's own words |
|---|---|---|---|
| 314838 | ncpi0 | 3.112e6 | *"Overclustering, The OUT segments should be a separate gamma cluster, then forma pi0 likely."* |
| 142421 | ncpi0 | 2.794e6 | *"The OUT segments can form a separate gamma, and then they could be a pi0"* |
| 269774 | nuecc48 | 1.149e6 | (no note — but all five of its OUT segments carry an explicit **IN** mark for sibling shower 97197 in the same event, §4) |

All three events are **one EM object absorbing a second EM object**. Not one
of them is a track absorbed into a shower — which is the 141-set's leading
item (a 110 cm pdg-13 muon, 34.5% of that pool by itself) and is what the
whole pr/123 → pr/130 guard family was built to decline.

Bee links for the scan views are in `pr130-qextra-98set.txt`.

## 2. The 22 segments, with live-run attribution

`marks_detail[shower]["marked"][seg]["absorbed_by"]` is the label store's
record of which absorber placed each segment. Part 2 checked that against a
live run **for the 141-set only** and got 20/22. Same check, 98-set:

**22 of 22 agree.** No segment changed its route between the scan and today's
census arms.

| event | shower | seg | cl | len | pdg | charge | d | ang | absorber (label = live) | tier |
|---|---|---|---|---|---|---|---|---|---|---|
| 314838 | 110088 | 110089 | 110 | 10.9 | 11 | 1.122e6 | 6 | 6 | `from_vertices` (walk_add) | — |
| 314838 | 110088 | **110088** | 110 | 7.6 | 11 | 7.992e5 | 0 | 0 | *own root* | — |
| 314838 | 110088 | 111091 | 111 | 8.3 | 11 | 3.729e5 | 26 | 3 | `pass4_angle` | 1 |
| 314838 | 110088 | 110084 | 110 | 3.5 | 11 | 2.725e5 | 17 | 11 | `from_vertices` (walk_add) | — |
| 314838 | 110088 | 110085 | 110 | 5.0 | 11 | 2.173e5 | 17 | 11 | `from_vertices` (walk_add) | — |
| 314838 | 110088 | 111092 | 111 | 3.8 | 11 | 1.730e5 | 33 | 4 | `pass4_angle` | 1 |
| 314838 | 110088 | 110083 | 110 | 2.1 | 11 | 1.553e5 | 16 | 7 | `from_vertices` (walk_add) | — |
| 142421 | 7010 | 108105 | 108 | 9.3 | 11 | 5.770e5 | 4 | 79 | `from_vertices` (walk_add) | — |
| 142421 | 7010 | **108104** | 108 | 11.5 | 11 | 4.609e5 | 8 | 137 | *own root* | — |
| 142421 | 7010 | 107101 | 107 | 6.1 | 13 | 4.529e5 | 43 | 20 | `pass4_angle` | 1 |
| 142421 | 7010 | 108106 | 108 | 6.8 | 11 | 4.248e5 | 4 | 79 | `from_vertices` (walk_add) | — |
| 142421 | 7010 | 108107 | 108 | 4.0 | 11 | 3.972e5 | 4 | 79 | `from_vertices` (walk_add) | — |
| 142421 | 7010 | 107100 | 107 | 7.2 | 11 | 3.572e5 | 43 | 20 | `pass4_angle` | 1 |
| 142421 | 7010 | 50026 | 50 | 0.7 | 13 | 4.266e4 | 53 | 22 | `pass4_angle` | 1 |
| 142421 | 7010 | 48024 | 48 | 0.2 | 13 | 3.433e4 | 56 | 23 | `pass4_angle` | 1 |
| 142421 | 7010 | 56032 | 56 | 0.2 | 13 | 3.204e4 | 14 | 29 | `pass4_angle` | 1 |
| 142421 | 7010 | 79055 | 79 | 0.2 | 13 | 1.530e4 | 56 | 22 | `pass4_angle` | 1 |
| 269774 | 13237 | 96190 | 96 | 3.3 | 11 | 3.526e5 | 55 | 25 | `pass4_angle` | 1 |
| 269774 | 13237 | 96189 | 96 | 5.1 | 13 | 3.336e5 | 50 | 25 | `pass4_angle` | 1 |
| 269774 | 13237 | 67079 | 67 | 1.9 | 2212 | 1.977e5 | 59 | 25 | `pass4_angle` | 1 |
| 269774 | 13237 | 68080 | 68 | 0.6 | 13 | 1.876e5 | 65 | 27 | `pass4_angle` | 1 |
| 269774 | 13237 | 100213 | 100 | 0.5 | 2212 | 7.740e4 | 49 | 24 | `pass4_angle` | 1 |

(`len` in cm, `d`/`ang` are the label store's dist/angle. Segment ids encode `cluster*1000 + index`, so id adjacency means
*different clusters* — Part 1's correction applies here too.)

**Note on 142421's shower key.** The label key is 7010 but the reconstructed
shower is rooted at **108104**; `em117_score.py --cross-run` recorded the
match. Looking the census up by the label key returns "not in census" for
every row of a re-rooted shower — the failure mode the qmiss doc warned about
in a different guise. The script uses the `matched` column.

### absorber shares, and how little the two pools have in common

| 98-set absorber | charge | share | segs | | 141-set absorber | charge | share | segs |
|---|---|---|---|---|---|---|---|---|
| `from_vertices` (walk_add) | 3.166e6 | **44.9%** | 7 | | `pass4_proximity` | 5.973e6 | 34.5% | 1 |
| `pass4_angle` (direct) | 2.629e6 | **37.3%** | 13 | | `pass3_cone` | 4.987e6 | 28.8% | 3 |
| *own root / extent* | 1.260e6 | 17.9% | 2 | | *own root / extent* | 2.970e6 | 17.2% | 2 |
| | | | | | `pass4_angle` | 2.244e6 | 13.0% | 14 |
| | | | | | `conn3_unreachable` | 6.351e5 | 3.7% | 1 |
| | | | | | `pass3_cluster_map` | 5.054e5 | 2.9% | 1 |

`pass4_angle` is the **only** absorber the two pools share. The 98-set's
largest site, `from_vertices` walk-add, does not appear in the 141-set at all;
the 141-set's two largest, `pass4_proximity` and `pass3_cone` — the two seats
pr/130 Part 4 built guards for and Part 5 flipped ON — do not appear here at
all. Two disjoint mechanisms, matched only by their segment count.

Every `pass4_angle` admission in this pool is **tier 1**
(`angle_v1 < 25° && (pair_dis < 80 cm || body_dis < 25 cm)`), and none is near
the boundary: from the census, `angle_v1` runs 6.6–21.5° with `pair_dis`
28–91 cm. Part 1's two 141-set leads — backward admission at 137–150° and
tier-2 far admission at 98–125 cm — have **no counterpart here**.

## 3. Length: why no shipped guard can reach any of them

Every over-clustering decline this campaign has shipped is length-gated *and*
exempts electrons:

| knob | seat | floor | pdg-11 |
|---|---|---|---|
| `shower_pass4_track_guard_len` (pr/123) | `pass4_angle` | > 50 cm | exempt unless MIP-flat |
| `shower_pass4_prox_guard_len` (pr/130) | `pass4_proximity` | > 50 cm | exempt unless MIP-flat |
| `shower_pass3_backfill_guard_len` (pr/130) | `pass3_cone` | > 15 cm | exempt unless MIP-flat |
| `shower_absorb_track_guard` (pr/40 r6) | every walk_add | > 10 cm (`segment_is_straight_long_track`) | **early-out on \|pdg\|==11** unless `em_straight_min_len > 0`, which only `examine_shower_1` passes (`PRShower.cxx:826-834`) |

| | 98-set | 141-set |
|---|---|---|
| max length | **11.5 cm** | **110.3 cm** |
| median length | 3.8 cm | 1.5 cm |
| n > 10 cm | 2 | 6 |
| n > 15 cm | **0** | 6 |
| n > 50 cm | **0** | 1 |
| charge carried by pdg-11 segments | **5.68e6 = 80.5%** | 8.34e6 = 48.2% |
| largest single item | 10.9 cm e⁻, 15.9% of pool | 110.3 cm mu⁻, 34.5% of pool |
| **reachable by a shipped guard at its own seat** | **0 of 22** | 1 of 22 (evt 100222 seg 14003) |

The one 98-set segment that clears any floor — 314838 seg 110089, 10.9 cm at a
walk-add seat — is pdg 11 and therefore exempt there by construction.

**Do not read the pdg-13 rows as tracks.** Seven of the 22 carry a track PDG
on a segment **shorter than 2 cm** (three of them 0.2 cm). A PID on a 2 mm
segment is not a particle claim, and the guards' `abs(pdg) in {13,211,2212}` test
is meaningless at that length. By charge the pool is 80.5% electron.

### This is not an argument from thresholds — it is measured

The census arm that produced these 22 segments **already had every shipped
over-clustering guard and prune ON** (compiled config, not the jsonnet:
`work-pr130r1-probe98-ncpi0/pr_evt314838/.wct-cfg-evt314838.json`) —
`shower_absorb_track_guard`, `shower_cone_absorb_guard`,
`shower_pass3_cone_guard_len=15`, `shower_pass4_prune_detached`,
`shower_pass4_prune_gap2=25`, `shower_pass4_track_guard_len=50`, and thirteen
more.

The only two knobs *not* on in that arm are the pair Part 4 shipped and Part 5
flipped SBND production ON the next day. Diffing the flip arms on these three
events (`work-pr130r1-g1off-*` vs `work-pr130r1-gs1on-*`, all dump sections,
excluding only the `vertex_scoreboard.dual_chain.off_ms` wall-clock field —
a raw `cmp` reports a difference that is only a timer, M2 in miniature):

```
evt 314838 (ncpi0)  : IDENTICAL
evt 142421 (ncpi0)  : IDENTICAL
evt 269774 (nuecc48): IDENTICAL
```

Consistent with Part 4's measured 10-event blast radius, which names none of
these three — but now diffed rather than cited.

> **The 98-set's pool is exactly what the campaign's entire shipped
> over-clustering machinery leaves behind.** Every guard was on when it was
> measured, and the two newest change nothing on any of its events.

## 4. Part of it is not "extra" charge at all — it is a re-home

`em117_score.py` scores each labelled shower independently, so a segment the
scanner moved from shower A to shower B lands in **A's `extra` and B's
`miss`**. The same charge appears on both sides of the ledger, and a "decline
the absorb" fix would orphan it rather than place it.

Run on both sets so the comparison is like-for-like:

| | segments also in a sibling shower's `miss` | charge | share of pool |
|---|---|---|---|
| **98-set** | **5 of 22** (all 5 with an explicit IN mark there) | 1.149e6 | **16.3%** |
| 141-set | **0 of 22** | 0 | 0% |

All five are evt 269774: segs 67079, 68080, 96189, 96190, 100213 are OUT of
shower 13237 and explicitly IN for shower 97197. So 269774 is not an
over-cluster — it is a **swap between two showers**, and its whole 1.149e6
contribution is double-counted in the affirmative q_extra total.

Corrected: the 98-set's genuinely-extra affirmative charge is **5.907e6, not
7.056e6**, over **two** events.

The 141-set control at zero says this is a property of *this* pool, not of the
scorer. The fix shape for 269774 is a **re-home** — the pr/121 ex1-dedup
shape — not a decline.

## 5. Two of three showers are rooted on a condemned segment

| | events whose reco shower root is itself condemned |
|---|---|
| 98-set | **2 of 3** — evt 142421 root 108104, evt 314838 root 110088 |
| 141-set | 2 of 10 — evt 69232 root 20021, evt 489327 root 19005 |

This is mis-*seeding*, not over-*reach*: the shower was started on the wrong
gamma and grew from there. In 142421 the root 108104 sits at **137°** to the
axis — the same backward signature `stem_backfill_back_guard` (pr/120)
declines at 110°, but at a seeding site no guard watches. No admission-time
predicate at any absorber can fix a shower that was seeded wrong; the
condemned charge is not being *admitted*, it is where the object started.

## 6. What the merge costs: the pi0

Two of the three events are `ncpi0` and the scanner's note on both says the
condemned charge is a second gamma. The reconstructed pi0 pairing is in the
dump, so the consequence is readable with no truth file:

| event | `kine_pio_mass` | pairing | |
|---|---|---|---|
| 314838 | **130.6 MeV** | 693.4 MeV = shower 110088 (**the over-clustered shower**) + 118.6 MeV = shower 13010 | shower 13010 is rooted on a segment the scanner marked **IN** for 110088 — the reco's "second gamma" is a fragment of the *first* |
| 142421 | **42.4 MeV** | 583.7 MeV = shower 108104 (**the over-clustered shower**) + 9.4 MeV = shower 53029 | the real second gamma (2.794e6 of charge) is inside the 583.7 MeV shower; the partner is a 9 MeV fragment |
| 269774 | **1445.8 MeV** | 1150.3 MeV = shower 87134 + 916.7 MeV = shower 13237 (**the over-clustered shower**) at 88.4° | shower 97197, the one the scanner says wants 13237's charge, is **not in the pairing at all** |

314838's 130.6 MeV is close to the pi0 mass **for the wrong reason** — it
pairs a blob containing both gammas with a piece of one of them. A mass near
135 is not evidence the event reconstructed correctly, and a pi0 selection
would accept it.

This is a second named pi0 blocker alongside pr/126's finding that PID is the
top one: **the two gammas merging into one shower**. pr/126 measured the mass
scale on events where two showers exist; these are events where they do not.

## 7. Where this leaves it

Measured, in order:

1. The 98-set's affirmative q_extra pool is **22 segments / 7.056e6 in three
   events**, ~76% of an IN-heavy scan's entire OUT population. No
   concentration claim is available or made.
2. All three events are **EM-to-EM mis-partition** — two of them with the
   scanner's own note saying "should be a separate gamma → pi0". None is a
   track absorbed into a shower.
3. Label-store absorber attribution holds **22/22** against a live run
   (the 141-set's was 20/22).
4. The two pools share **one** absorber (`pass4_angle`) out of eight, and the
   141-set's top two sites do not appear here at all.
5. **Zero of 22 are reachable by any shipped guard** — max length 11.5 cm
   against floors of 15 and 50 cm, and 80.5% of the charge is electron-PID'd
   where the guards exempt. Confirmed, not argued: every shipped guard was ON
   in the measuring arm, and the two flipped on 2026-08-29 leave all three
   events byte-identical.
6. **16.3% of the pool is double-counted** (evt 269774, five segments also
   explicit-IN for sibling shower 97197); 141-set control **0%**. The
   genuinely-extra affirmative charge is 5.907e6 over two events.
7. **2 of 3 showers are rooted on a condemned segment** — a seeding failure
   no admission-time predicate can reach.
8. The cost is the pi0: masses of 130.6 (right number, wrong pairing), 42.4
   and 1445.8 MeV.

**No knob is proposed and none should be inferred.** The campaign has five
consecutive rounds (pr/119, pr/128, pr/129, pr/130 Parts 1–3 and Part 5) that
came back measured-dead on admission-time geometry, and §2 shows this pool's
`pass4_angle` admissions are nowhere near a tier boundary — a threshold move
that reached them would be a large one, fitted to three events, which is what
§5.7 forbids.

What this doc establishes is that the item was mis-scoped from the start: the
"44 segments are the target list" line in Part 1 joined two pools that share
one absorber, no length scale, and no failure mode. **The 141-set's half has
been worked through Part 5; the 98-set's half has never been addressed by
anything, and is a different problem — EM-vs-EM partition and shower seeding,
not track admission.**

### What is NOT established

- **That EM-to-EM merging is common.** Three events, from a scan whose OUT
  marks are incidental. The rate is unmeasured; it needs an OUT-protocol scan
  on both manifests, which is the same instrument the 141-set/98-set
  confound in Part 1 already asked for.
- **That the pi0 masses are wrong against truth.** Section 6 reads the reco's
  own pairing and the scanner's note. No truth file was opened; 314838's
  130.6 MeV is called out as suspect on the *pairing*, not on the number.
- **That the seeding is what fails first** in 314838/142421. The root being
  condemned is consistent with mis-seeding and with a correct seed that grew
  the wrong way and was later re-rooted; separating those needs the seeding
  census, which does not exist.
- **The corrected 5.907e6 does not propagate** to Part 1's cross-set split
  (82.9 / 17.1). Whether the 141-set's q_miss side carries a symmetric
  double-count was not checked — only its q_extra side was, and that is 0.

Related: [`pr130-qextra-refresh.md`](pr130-qextra-refresh.md) Parts 1–5,
[`pr130-qmiss-refresh.md`](pr130-qmiss-refresh.md),
[`126_pi0-audit-and-em-charge-scale.md`](126_pi0-audit-and-em-charge-scale.md),
[`130_guard-freed-overcount.md`](130_guard-freed-overcount.md).

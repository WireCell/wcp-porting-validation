# pr/130 item C — the `pass4_angle` remainder: measured dead, and a correction

**Status: MEASURED. Negative. No knob proposed.** This is the sixth
consecutive round in which admission-time geometry fails to encode the
scanner's judgement, and it is measured on the largest labelled set yet.

Item C is what items 1b/B cannot reach: the affirmative q_extra segments that
`pass4_angle` placed and **no guard had declined**, so the "overruled guard"
fix has no purchase on them.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./scripts/pr130_arms.sh 98 vtxcen 1 ; ./scripts/pr130_arms.sh 141 vtxcen141 1
./scripts/pr130_pass4angle_census.py > docs/pr/pr130-pass4angle-census.txt
```

Census arms are byte-neutral and at the **2026-08-29 production point** (both
guard seats flipped ON), so the pool is measured after items 1b/Part 5, not
before.

## A correction to Part 1

Part 1 of `pr130-qextra-refresh.md` said *"`pass4_angle` admits backward — all
four of 286655's segments come in at 137–150°, beyond the 110° that
`stem_backfill_back_guard` declines on"*, and the same claim went into commit
`32c6cedc`. **It is wrong, and the mechanism of the error is worth recording.**

The 137–150° figure is the **label store's** `angle` field — the scanner-display
angle of the marked segment relative to the shower axis at scan time. The
absorber's own admission angles are `angle_v1` / `angle_v2`, and the census
reads them at **7.7–14.1°** and **4.8–12.4°** for those same four segments.

Two different quantities with the same name. `pass4_angle` is **not** admitting
backward; it admits these at small forward angles, and it is the scanner's axis
that makes them look backward. Any knob built on the earlier reading would have
been gated on a threshold the code never evaluates.

## The measurement

**1590 `pass4_angle` admissions over 154 events; 14 scanner-condemned, 1576
not.**

| event | seg | len_cm | angle_v1 | angle_v2 | pair_dis | body_dis | tier |
|---|---|---|---|---|---|---|---|
| 72786 | 9009 | 2.3 | 5.1 | 6.7 | 53.4 | 31.6 | 1 |
| 72786 | 31033 | 0.7 | 8.8 | 10.4 | 85.0 | 32.1 | 3 |
| 278420 | 67033 | 0.2 | 4.5 | 5.2 | 102.7 | 23.9 | 1 |
| 278420 | 68034 | 0.2 | 10.2 | 10.9 | 113.3 | 15.7 | 1 |
| 278420 | 69035 | 0.4 | 10.2 | 10.7 | 115.1 | 9.9 | 1 |
| 278420 | 70036 | 0.3 | 5.5 | 5.5 | 117.3 | 12.9 | 1 |
| 278420 | 73039 | 0.7 | 2.2 | 2.2 | 130.2 | 14.7 | 1 |
| 278420 | 75041 | 4.8 | 7.0 | 7.9 | 106.8 | 7.9 | 1 |
| 278420 | 78044 | 0.3 | 1.1 | 1.3 | 110.0 | 9.7 | 1 |
| 286655 | 67016 | 1.5 | 14.1 | 11.5 | 73.7 | 17.3 | 1 |
| 286655 | 69018 | 0.4 | 8.6 | 12.4 | 88.2 | 22.9 | 1 |
| 286655 | 81062 | 0.7 | 7.7 | 4.8 | 92.4 | 17.5 | 1 |
| 286655 | 82063 | 20.6 | 9.4 | 7.2 | 77.6 | 7.8 | 1 |
| 400504 | 21003 | 0.8 | 21.5 | 21.2 | 30.2 | 20.0 | 1 |

**Every one of the twelve taped features interleaves** with the legitimate
population — none comes close to the non-overlap bar this campaign uses:

| feature | condemned | legitimate | legit rows inside the condemned range |
|---|---|---|---|
| `len_cm` | 0.23 – 20.57 | 0.00 – 37.70 | 1467 / 1576 |
| `angle_v1` | 1.05 – 21.46 | 0.19 – 45.97 | 1399 / 1576 |
| `angle_v2` | 1.34 – 21.17 | 0.12 – 29.96 | 1252 / 1576 |
| `pair_dis_cm` | 30.22 – 130.17 | 9.11 – 440.97 | 1361 / 1576 |
| `body_dis_cm` | 7.80 – 32.11 | 0.00 – 95.66 | 691 / 1576 |
| `front_dis_cm` | 27.64 – 116.31 | 0.06 – 157.61 | 1116 / 1576 |
| `snap_dis_cm` | 17.27 – 114.00 | 0.06 – 154.78 | 1003 / 1576 |
| `med_dqdx_mip` | 0.39 – 2.87 | 0.00 – 5.84 | 1518 / 1576 |
| `cur_len_cm` | 12.40 – 60.20 | 1.00 – 538.60 | 751 / 1576 |
| `cur_nseg` | 1 – 16 | 1 – 92 | 1170 / 1576 |
| `tier` | 1, 3 | 1, 2, 3, 4 | 1562 / 1576 |
| `divert` | 0 | 0 – 1 | 1547 / 1576 |

The "far chain" reading of 278420 does not survive either: its segments sit at
`pair_dis` 102–130 cm, but legitimate admissions run out to **441 cm**.

Note also what the condemned segments actually are — **twelve of fourteen are
under 5 cm long** (0.2–4.8 cm), i.e. crumbs. There is no length, angle, or
distance at which one can be told from the 1576 crumbs that belong.

## Verdict

**No admission-time predicate at the `pass4_angle` seat separates this pool.**
Ratio is the point: 14 condemned inside 1590 admissions, interleaved on every
axis. Any threshold tight enough to catch the 14 would take hundreds of
legitimate segments with it.

This closes item C negatively and, with it, the admission-side family:

| round | question | result |
|---|---|---|
| pr/119 | local separator for the expel predicate | dead |
| pr/128 | is proximity continuation | dead (needed a second term) |
| pr/129 | is over-clustering a distance | dead (needed direction) |
| pr/130 p4 | six features on the back guard | dead |
| pr/130 p5 | four vertex-relative features | dead on margin |
| **pr/130 C** | **twelve features on pass4_angle, 1590 admissions** | **dead** |

What *has* worked in this round is not a better predicate but **applying an
existing, owner-approved predicate at a seat that was missing it** (items 1b
and B). That is the pattern worth carrying forward; item C has no such seat to
offer, because no guard declined these segments in the first place.

## What this leaves

The affirmative q_extra pool splits cleanly by what is reachable:

- **~51%** (the two laundered segments) — fixed by item 1b, SBND ON.
- The back guard's two errors — fixed by item B, flip proposed.
- **~49%** — this pool. **Not reachable by any admission-time rule**, and the
  peer session's Part 6 reached the same conclusion independently on the
  98-set's own 22 segments (EM-vs-EM mis-partition and mis-seeding, "no shipped
  guard can reach it").

Both halves now point at the same place: the remaining over-clustering is a
**partition/seeding** problem, not an admission problem. A downstream coherence
test (Part 4 option 3) is the only untried family, and two of the peer's three
98-set showers being *rooted* on a condemned segment says the seeding side may
matter more than the absorbing side.

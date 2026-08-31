# pr/130 item B — fixing the back guard's two owner-ruled errors

**Status: census MEASURED, knob implemented DEFAULT OFF, gate + ON arm
running.** Closes Part 4's option 1 with a *positive* result, reversing Part
5's rejection — for a reason Part 5 could not see.

Item B is the owner's ranked-1: `stem_backfill_back_guard` is right on 6 of
the 8 candidates it can reach, and wrong on two that are still in production —
**292643** (declines an absorb that should happen, −234.0 MeV lost) and
**179369** (the decline leaves a spurious pi0, +376.0 MeV gained). Both are
failures of the pr/128 metric.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./scripts/pr130_arms.sh 98 vtxcen 1 ; ./scripts/pr130_arms.sh 141 vtxcen141 1
./scripts/pr130_vtx_census.py > docs/pr/pr130-vtx-census.txt
```

Census arms are byte-neutral (stderr only) and were run at the **2026-08-29
production point**, i.e. after the two guard seats flipped ON — not at the
pr/129 point Part 5 used.

## What Part 5 got wrong

Part 5 measured `dvtx_start` (shower start → neutrino vertex) on the guard's 8
candidates, found it separates the owner's verdicts **8/8** with a 2.50 cm
margin, and **rejected it** on the grounds that the boundary might be "an
accident of nine points". The deferred check was to widen the population.

Widening it settles the question the other way, because of a control-flow fact
neither Part 4 nor Part 5 checked:

```cpp
if (!ok) break;                       // NeutrinoShowerClustering.cxx:610
...
if (m_stem_backfill_back_guard) { ... }   // the guard, line 622
```

`ok` is the MIP-window acceptance (`len < max_len && ratio in [lo, hi)`). **The
guard only ever sees chains the MIP window already accepted.** So its
reachable population is not "whatever the angle condition matches" — it is the
`ok=1` subset.

## The census: the population is CLOSED, and it is exactly the labelled set

27 candidate rows over 25 events. **18** meet the guard's angle condition
(`ang15 > 110°`) — but only **9 of those carry `ok=1`, in exactly 8 events, and
those 8 events are precisely the ones the owner adjudicated.**

| event | ok | dvtx_start | ang15 | owner verdict |
|---|---|---|---|---|
| 289246 | 0 | 130.16 | 170.4 | *unreachable* |
| 172266 | 0 | 119.71 | 173.1 | *unreachable* |
| **179369** | **1** | **88.11** | 161.9 | **ABSORB WANTED** |
| 179611 | 0 | 72.22 | 177.6 | *unreachable* |
| 280159 | 0 | 49.77 | 139.6 | *unreachable* |
| **292643** | **1** | **46.84** | 172.8 | **ABSORB WANTED** |
| 283515 | 1 | 44.34 | 148.4 | decline ok |
| 98844 | 0 | 44.16 | 148.7 | *unreachable* |
| 67394 | 1 | 43.46 | 174.4 | decline ok |
| 98844 | 0 | 40.97 | 169.9 | *unreachable* |
| 286655 | 1 | 27.88 | 154.2 | decline ok |
| 286655 | 1 | 26.66 | 144.1 | decline ok |
| 282909 | 0 | 21.84 | 159.7 | *unreachable* |
| 281165 | 0 | 17.05 | 111.7 | *unreachable* |
| 37112 | 0 | 15.86 | 163.2 | *unreachable* |
| 347824 | 1 | 11.54 | 167.9 | decline ok |
| 281567 | 1 | 8.91 | 150.4 | decline ok |
| 47212 | 1 | 5.01 | 150.2 | decline ok |

Two consequences, and the second is the one that matters:

1. **`dvtx_start` separates the complete reachable population 8/8.**
   Absorb-wanted: 46.84 and 88.11. Every decline-ok: ≤ 44.34. The interval
   (44.34, 46.84) is empty across all 27 rows.
2. **A `dvtx_start` condition on this guard cannot touch an unadjudicated
   candidate.** Every `ok=0` row — including the four with dvtx_start above the
   boundary (289246, 172266, 179611, 280159) — breaks out before the guard
   runs. The blast radius of the knob is a subset of the 8 events the owner has
   already ruled on, and by the numbers above it is exactly the 2 he ruled
   wrong.

That is a materially different situation from "an accident of nine points".
The nine points **are the population**; there is no larger sample for the
boundary to be an accident of. The residual risk is generalisation to future
events, not mis-attribution on today's.

**Stated against the round's own bar, honestly**: the margin is 2.50 cm, and
`dvtx_start` still equals `dist_cm` for 7 of the 8 — 292643 is the only row
where the two differ, so it alone supplies the discrimination on the hard
case. This knob is weaker in *margin* than pr/128's 25° or pr/130 Part 1's
99°. What makes it shippable where Part 5's version was not is the closed,
fully-adjudicated blast radius, which no amount of margin would have given.

## The knob

`stem_backfill_back_dvtx` (cm), C++ **default 0 = off**, key-suppressed in the
sbnd jsonnet. When > 0 and the shower start is further than that from the
neutrino vertex, the backward decline is **suppressed** and the chain falls
through to the normal absorb path. Proposed operating point **45 cm** — between
44.34 and 46.84.

Emits `pr130 stem_backfill_back_dvtx: suppress decline seg=… dvtx_start=…`
and, under the census probe, `SHOWER_ABSORB P130_BACK_DVTX …`.

## Validation status

- `./build/clus/wcdoctest-clus`: **2534 assertions passed, 0 failed** (incl.
  the new default pin).
- Knob-off gate vs the new production (`work-pr130r1-gs1on{,141}-*`) and the
  ON arm at 45 cm: **running** — results appended below when they land.

## Validation — complete

- **Knob-off gate: PASS 478/478 archives byte-identical** over all 239 events
  of both manifests, vs the current production point
  `work-pr130r1-gs1on{,141}-*`. Labels `work-pr130r1-boff-{mcp1k,mcp2k,ncpi0,nuecc48}`
  and `work-pr130r1-boff141-{mcp1k,mcp2k}`.
- `./build/clus/wcdoctest-clus`: 2534 assertions passed, 0 failed.
- **ON blast radius at 45 cm: exactly 2 events — 292643 and 179369.** Nothing
  else moves in 239. The knob fires on three chain steps (292643 segs 18008 and
  18010 at dvtx_start 46.8 cm; 179369 seg 17002 at 88.1 cm).

| event | effect |
|---|---|
| **179369** | the spurious `pi0 138` and its `gamma 216` are **gone**; `gamma 38`, `pi+ 56` gone, `e- 36` appears. 43 → 39 PF nodes. This is the +376.0 MeV of spurious energy the owner ruled against. |
| **292643** | the declined stem is absorbed: `e- 227` + `mu- 59` give way to `e- 154`/`gamma 154`, `pi+ 65`, `pi+ 91`, `pi0 150`; 10 → 13 nodes, 7 → 13 showers. |

## The strongest part: the owner has already scanned both outputs

The knob does not produce a new reconstruction to be adjudicated. On **both**
events the resulting PF tree is **byte-for-byte identical to the guard-OFF
shape the owner reviewed in `bee/pr130r2` and ruled better**:

```
292643   work-pr130-292643-guardoff   vs  work-pr130r1-bon141-mcp1k   IDENTICAL (13 nodes)
179369   work-pr130r1-gd141off-mcp2k  vs  work-pr130r1-bon141-mcp2k   IDENTICAL (39 nodes)
```

So the knob reproduces the owner's preferred shape on exactly the two events he
preferred it on, and leaves the other six — where he preferred the guard's
decline — untouched. The pr/130 Part 4 verdict table is satisfied 8 of 8.

**Proposed: `stem_backfill_back_dvtx = 45` SBND PRODUCTION ON.** Awaiting the
owner; the knob is committed DEFAULT OFF and the legacy path is gated
byte-identical either way. On flip, register a sentinel per event (179369:
`pf_absent pi0 138`-shaped; 292643: the suppress-decline log line) and close
`stem_backfill_back_guard`'s standing "second side unasserted" exposure from
Part 7 — the whole 8-candidate population becomes adjudicated *and* correct.

---

## SHIPPED — `stem_backfill_back_dvtx = 45` SBND PRODUCTION ON (2026-08-29)

Owner: *"For B, flip on for SBND production."*

- **Compiled-config proof** with no env overrides: `stem_backfill_back_dvtx =
  45` in the `TaggerCheckNeutrino` data block of
  `.wct-cfg-evt{179369,292643}.json`.
- **Flip equivalence**: the flipped default reproduces the TLA-driven ON arm
  **byte-identically** — PASS 2/2 archives on each of
  `work-pr130-bflipchk-mcp2k` vs `work-pr130r1-bon141-mcp2k` and
  `work-pr130-bflipchk-mcp1k` vs `work-pr130r1-bon141-mcp1k`. So
  `work-pr130r1-bon{,141}-*` are the labels for the new production point.
- Blast radius unchanged from the ON measurement: **2 of 239 events.**

### Sentinels — the knob's whole population is now guarded

**Registry at the new production point: 33 PASS, 0 FAIL, 0 SKIP.** Negative
control on `work-pr130r1-boff*`: **both new entries FAIL**, so neither can pass
vacuously.

| event | assertions | measured |
|---|---|---|
| **292643** | `pf_node_ge pi0 100`, `pf_node_lt e- 200`, the suppress-decline line for seg 18008 | pre-flip: mu- 59 + 441, **no pi0 and no pi+ at all**, e- max 227, 10 nodes. post-flip: mu- 441, pi0 150, pi+ 91 + 65, e- max 154, 13 nodes. |
| **179369** | `pf_absent pi0`, the suppress-decline line for seg 17002 | pre-flip: pi0 138 + pi+ 56 fabricated by the decline, 43 nodes. post-flip: **no pi0 and no pi+ anywhere**, 39 nodes. `pf_absent` is exact here, not a threshold. |

Both are binary assertions, not thresholds on a drifting energy — they survive
the `kine_shower_fudge_factor` re-baseline warning at the top of the registry.

### The Part 7 exposure is closed

The registry's "DELIBERATELY UNASSERTED" block held 292643 and 179369 back
because a sentinel written for either would have pinned a state the owner had
just condemned. That block is now resolved:
**`stem_backfill_back_guard`'s entire 8-candidate population is adjudicated
*and* correct** — 47212 and 281567 guard the declines, 292643 and 179369 guard
the escapes. Nothing about this knob is unasserted any more.

Two standing exposures remain from Part 7, both unrelated to this knob:
`pr/128 class A` (55740 masked) and `long_muon_stub_bridge_len` (66366 masked).

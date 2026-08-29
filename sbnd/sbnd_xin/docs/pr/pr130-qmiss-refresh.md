# pr/130 item 1 — refreshed q_miss ranking at the true production point

**Status: MEASURED. Go/no-go answered — GO on the hand-scan criterion, with one
material qualifier that changes what the round should be aimed at.**

This closes "item 3, q_miss hand-look", deferred by the pr/128 hand-off and
shelved-as-scoped in `130_guard-freed-overcount.md`. It is scoring only: no
knob, no C++, no config, nothing shipped.

## Repro

```bash
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
# 1. sidecars + manifests from the probe arms' OWN dumps and stdout probes
cd em_display
./prep_pr117.py --tag pr130q98  work-pr130r1-probe98-{mcp1k,mcp2k,ncpi0,nuecc48}
./prep_pr121.py --tag pr130q141 work-pr130r1-probe141-{mcp1k,mcp2k}
# 2. score against the hand-scan labels (cross-run: showers re-root after a knob)
./em117_score.py --tag emscan-0827 --manifest em117-pr130q98-manifest.tsv \
    --prepdir emprep-pr130q98  --cross-run --tsv ../docs/pr/pr130-98-score-prod.tsv
./em117_score.py --tag emscan-0828-agent5 --manifest em114c-pr130q141-manifest.tsv \
    --prepdir emprep-pr130q141 --cross-run --tsv ../docs/pr/pr130-141-score-prod.tsv
# 3. ranking + both go/no-go criteria
cd .. && scripts/pr130_qmiss_rank.py
```

Fresh tags throughout (`pr130q98`, `pr130q141`); `em_labels/` read-only; no
existing prepdir, manifest or score table was written over (M13).

## The arms ARE at the true production point — established, not assumed

The probe arms `work-pr130r1-probe{98,141}-*` were gated SAME against
`work-pr128r1-on{98,141}-*`, i.e. the **pr/128** production point. `4f894afb`
(pr/129 pointing test SBND ON) landed afterwards, so the arms are nominally one
flip behind production, and pr/129's knob-on footprint — **393505** — *is* in
the 141-manifest (`em114c-manifest.tsv` line 120). That had to be checked, not
waved past.

`em117_score.py` reads exactly two things from a calib dump:
`dump["segments"]` (line 129) and the sidecar's `showers` (lines 144, 405). It
never reads `kine`, `tagger` or `vertex_scoreboard`.

Comparing `segments` + `showers` between each probe arm and the pr/129-ON arms
`work-pr129r1-on{141,98}-*`:

```
compared 239 events: SAME 239  DIFF 0  (unpaired 0)
```

On 393505 the dumps *do* differ, and the difference is confined to the kine
block — exactly pr/129's documented footprint:

| key | probe arm (pr/128 point) | production (pr/129 ON) |
|---|---|---|
| `kine_reco_Enu` | 940.448 | 566.088 |
| `kine_energy_particle` | …, **268.702** | (that entry removed) |
| `kine_particle_type` | 13, 11, 11, 11, **13** | 13, 11, 11, 11 |
| `segments`, `showers` | — | **identical** |

So for everything this scorer touches, the probe arms sit at current
production. Stating it because the masked-knob finding (66366) in this same
round is what happens when that step is skipped.

`--cross-run` is confirmed to have fired on both sets (98-set: membership moved
on 177 segment slots over 25 events; 141-set: 259 slots over 55 events). A
silent exact-key join would have scored every re-rooted shower as a
catastrophe and still looked plausible.

## Adjudicated rows crossed off

An owner-ruled event cannot motivate a new hand-scan, so it leaves the pool:

| event | set | q_miss | why |
|---|---|---|---|
| 318769 | 141 | 2.733e6 | pr/129 owner reject — was 141-rank-1, 11.6% |
| 415278 |  98 | 5.909e6 | pr/124 declined trade-off |
| 283515 | 141 | 1.965e6 | pr/130 Part 4 — owner "ON better" |
| 179369 | 141 | 0.915e6 | pr/130 Part 4 — owner "OFF better" |

**179369 and 283515 are a judgment call, flagged rather than decided.** Both
are `stem_backfill_back_guard` movers. The arms are at guard-ON production, so
the knob *is* held fixed and there is no mis-attribution across the ranking —
but the owner has now ruled on both, and on 179369 he ruled the *production*
shape wrong with no fix yet shipped. Its q_miss is therefore measured against a
reconstruction that is already known to be wrong and already scheduled to
change. Scoring it as a fresh target would double-count a settled item. If the
owner prefers them left in, the 141 top-10 concentration rises, so this choice
is conservative in the direction of NO-GO.

## Result — the two criteria disagree

|  | 98-set | 141-set |
|---|---|---|
| events with marked rows | 25 | 55 |
| total q_miss | 4.777e7 | 2.362e7 |
| total q_extra | 1.792e7 | 2.514e7 |
| **(a) q_miss share of charge error** | **72.7% PASS** | **48.4% FAIL** |
| q_miss after crossing off | 4.186e7 | 1.801e7 |
| **(a) re-checked on the kept pool** | **74.1% PASS** | **41.7% FAIL** |
| **(b) top-10 share of q_miss** | **81.9% PASS** | **78.7% PASS** |
| top-10 as fraction of the KEPT pool | 42% (of 24) | 19% (of 52) |

Both criteria keep their verdict when recomputed on the kept pool alone, so
neither result is an artifact of which rows were crossed off.

The two sets are **disjoint** (0 overlapping events), so (b) passing on both is
two independent confirmations, not one sample seen twice. The 141-set number is
the stronger of the two: its top-10 is 19% of scored events, whereas the
98-set's top-10 is 42% of its 24 kept events, which inflates concentration almost
by construction (this is the same weakness that made the old "top-25 = 100%"
figure meaningless — there were only 33 rows).

**Go/no-go as asked — (b), the hand-scan criterion: GO.** Top-10 holds >70% of
q_miss on both sets after adjudication. A top-10 hand-scan is worth a scanner's
time.

**Read (a) on the 98-set as marginal, not comfortable.** 72.7% clears a >70%
bar by 2.7 points on 25 events / 33 rows, where a single event (415278, 12.4%
of that set's q_miss) moves it materially. It is a PASS, but "72.7% PASS /
48.4% FAIL" is a cleaner-looking split than 25 events can actually support.
The direction is robust — the 141-set fails by a wide margin either way — the
98-set's comfort is not.

**The qualifier — (a) fails out of sample, and it changes the aim.** The
premise "75% of charge error is q_miss" is confirmed **98-set only** (72.7%
here). On the 141-set q_miss is **48.4%** — q_extra (2.514e7) is still the
larger half. An under-clustering round addresses less than half the charge
error on the out-of-sample manifest. The round is worth running on
concentration grounds, but it should not be sold as fixing most of the charge
error, and its gate must watch q_extra: the 141-set is where over-clustering
still dominates.

### Top-10 targets (adjudicated removed)

- **141-set** (recommended, out-of-sample): 54341, 284206, 397630, 181050,
  52044, 54453, 408304, 395597, 294174, 281325
- **98-set**: 463565, 122660, 142421, 54332, 314838, 444187, 169626, 105946,
  76346, 21073

## Side-finding — the 141-set q_extra fell 19% and it is two events

Scoring the same 55 events / 57 rows with the same labels as `pr125-141-score-ond.tsv`:

| table | q_miss | q_extra | q_miss share |
|---|---|---|---|
| pr123-141-score-off | 2.377e7 | 4.861e7 | 32.8% |
| pr124-141-score-onA / pr125-off | 2.454e7 | 2.413e7 | 50.4% |
| pr125-141-score-ond | 2.362e7 | 3.115e7 | 43.1% |
| **pr130-141-score-prod (now)** | **2.362e7** | **2.514e7** | **48.4%** |

q_miss is identical to pr/125 to four significant figures and **moved on zero
events**. The entire −6.01e6 q_extra change is **two events**: 94392
(4.284e6 → 2.116e4) and 52693 (1.859e6 → 1.123e5).

Both are pr/124's named targets (`a9545660`: "SBND 406125/94392/52693"), which
is consistent with a pr/124-or-later shower-side flip landing after the
`pr125-ond` table was written. **Attribution NOT verified** — confirming it
needs a knob-off arm, which this item did not run. Recorded because it means
the 43% figure carried in the pr/130 notes is stale; 48.4% is the current
number.

Related: [`130_guard-freed-overcount.md`](130_guard-freed-overcount.md) Part 4
for the back-guard verdicts that put 179369/283515 in the adjudicated list.

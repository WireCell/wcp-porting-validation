# Round-3 machinery check — result

Repro:

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 <scratch>/pick_mix60.py /home/xqian/tmp/scan-mix60-20260815
python3 vtx_rules/selfscan.py prepare --kit new --workers 4 \
    --only-manifest /home/xqian/tmp/scan-mix60-20260815/selection.json \
    --out /home/xqian/tmp/scan-mix60-20260815
# four scanner subagents, doc pr/80 sec 11 step 2, scan_prompt.md verbatim
python3 vtx_rules/selfscan.py review --dir /home/xqian/tmp/scan-mix60-20260815
python3 vtx_rules/selfscan.py score  --dir /home/xqian/tmp/scan-mix60-20260815
```

Kit at `f9c6efa`, working tree clean on `vtx_rules/` and `pr_display/`,
`scankit.py selftest` 25 dumps / 0 failures. Sample: 20 nueCC / 12 NCpi0 /
28 BNB-inclusive, none previously shown to an AI scanner, all on the `-ma10`
arm. Bar declared in `PREDECLARED.md` before any pick was scored.

## Verdict: the machinery works. Primary bar PASSES, secondary bar FAILS.

**Primary — scanner vs reconstruction, same 60 events, same metric:**

```
scanner  47/60 (78.3%)   @3cm 49/60 (81.7%)
reco     46/60 (76.7%)
```

Declared band was "within ~2 events = working". It came in **+1**. Answered
60/60, no abstentions.

**Secondary — `certain` ≥ 90%: FAILS at 87.5% (21/24), coverage 40.0%.**
Tiers order, but the top two are nearly flat:

```
certain  24/60 (40.0%)  21 correct (87.5%)   by vertex id 20 (83.3%)
likely   26/60 (43.3%)  22 correct (84.6%)   by vertex id 19 (73.1%)
unclear  10/60 (16.7%)   4 correct (40.0%)   by vertex id  4 (40.0%)
```

§10.5 held-out had `certain` at 95.5% over 36.7% coverage. 21/24 vs 21/22 is not
a distinguishable difference at this n — but the bar was declared as a threshold,
and 87.5% is under it. Reported as a fail, not explained away.

## The failure does not reach the deployed workflow

Every `certain`-tier error was a **confident disagreement** with the
reconstruction, so all three were routed to REVIEW FIRST rather than
auto-accepted:

```
auto-accept                             n=19   scanner 19/19   reco 19/19
REVIEW FIRST (confident disagreement)   n= 5   scanner  2/ 5   reco  1/ 5
REVIEW                                  n=36   scanner 26/36   reco 26/36
```

**Auto-accept was 19/19 on this sample** at 32% coverage — consistent with the
~95% prior, and it does not contradict the 87.5% `certain`-tier miss.

**But it is not an independent measurement, and must not be quoted as one.**
Auto-accept is *defined* as agreeing with the reconstruction, so its accuracy is
the reconstruction's accuracy on the subset the scanner felt certain about — the
§4.1 selection effect, arriving here exactly as it did in §10.5. A `certain`-tier
error can only escape into auto-accept if the scanner and the reconstruction are
wrong *together*; that class exists (evt56982, evt59335) and was routed to REVIEW
FIRST only because the two wrong answers happened to differ from each other. The
19/19 is therefore closer to a tautology than to a robustness finding.

The five confident disagreements:

| event | scanner | reco | dist |
|---|---|---|---|
| evt280774 | **right** | wrong | 0.01 |
| evt423981 | **right** | wrong | 0.00 |
| evt360535 | wrong | right | 1.27 |
| evt56982 | wrong | wrong | 12.25 |
| evt59335 | wrong | wrong | 64.78 |

Two genuine reconstruction errors caught in a 5-event pile. One case where the
scanner broke a correct reco, and it is a 1.27 cm quibble that is correct at the
3 cm tolerance. Two where both are wrong — no worse than the status quo.

Error enrichment: the reco is wrong on 23.3% of all 60 but on 66.7% of the 15 the
scanner declined or disagreed with — **×2.86**.

## Metric note: distance 47 vs vertex-id 43

The gap is entirely four events where the scanner picked a **co-located twin
vertex** — a different id within 0.76 cm of the click:

```
evt138009  picked 12117, truth 12002, 0.76 cm
evt314838  picked 18020, truth 18018, 0.30 cm
evt353223  picked 15004, truth 15009, 0.36 cm
evt359980  picked 70078, truth 69076, 0.41 cm
```

No label had a missing `truth_vid`, and there were no cases of matching id at
>1 cm. So on this sample vertex-id scoring is *stricter* than distance, the
opposite of §10 where the cross-arm refit made it looser. **Neither metric
dominates**: distance forgives a co-located twin, vertex id forgives a refit.
The like-for-like comparison against the reconstruction is the distance column,
because `reco_ok` is computed by distance — comparing scanner-by-vertex-id
against reco-by-distance would be two different questions.

## By channel — the composition question

```
channel     n        scanner            reco          certain tier
BNBincl    28     21 (75.0%)      19 (67.9%)       6/7  (85.7%)
NCpi0      12      9 (75.0%)       9 (75.0%)       1/2  (50.0%)
nueCC      20     17 (85.0%)      18 (90.0%)      14/15 (93.3%)
```

This mostly explains the tier flattening as **sample difficulty, not drift**:

- On nueCC the `certain` tier is 93.3% over 75% coverage — §10-like behaviour,
  on the cleanest topology.
- On NCpi0 the scanner claimed `certain` on only 2 of 12 events. It correctly
  senses that NC pi0 is hard; there is simply not enough certain-tier population
  there to be accurate in.
- The scanner and the reconstruction move together across channels (both worst on
  their respective hard class), which is what "harder sample" looks like. The one
  place the scanner is clearly ahead is BNB-inclusive, 21 vs 19.

n=2 and n=7 certain-tier cells: no per-channel tier claim is supportable. This is
a description of where the pooled 87.5% comes from, not a measurement.

## Operational findings

1. **The wait-for-completion rule paid off on its first use.** Worker 1's
   `picks-1.json` read 4 certain / 10 likely mid-run and 3 / 11 at completion —
   it demoted one pick at 14:20:47, after an earlier inspection. Had this been
   scored on file presence, §10.9 would have repeated.
2. **Tier usage varies by scanner.** certain counts were 8 / 3 / 6 / 7 across the
   four workers on 15 events each. The pooled `certain` rate mixes four
   calibrations; per-worker tier discipline is a real source of variance and
   should be reported alongside any pooled tier number.
3. **`--only-manifest` is the right entry point for labelled events.** §11's
   `--dumps` path discards the label join key and disables `score`. Feeding a
   manifest-shaped selection file preserves it. The manifest was verified to
   contain exactly the 60 selected events with no cross-tag twins pulled in by
   the event-name join.

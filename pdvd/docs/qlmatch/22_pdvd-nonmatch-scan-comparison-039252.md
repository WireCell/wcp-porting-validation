# 22 — Non-matched long clusters vs the hand/AI scan record (run 039252, 18 evts)

**Question (owner, 2026-07-17):** after the nm3+nm4b adoptions some long
clusters are still non-matched. The 18 events have hand/AI scan results —
did the scans match most of them? Compare case by case, explain WHY each one
was not matched, and say what additional mechanisms could reduce the count.

**Scope:** read-only analysis of the existing nm4b dumps and the frozen doc-19
scan truth. No code changed, no reprocessing, no tuning.

## Repro

```bash
cd pdvd
# per-cluster join of census + scan verdicts + dump candidates (this doc's numbers)
python ql_display/nonmatch_explain.py --tag nm4b
#   -> work/ql_scores/nm4b/nonmatch_explain.{md,json}   (91 rows, all metrics)
# deep-dive evidence renders (18 PNGs)
python ql_display/render_groups.py work/039252_0_nm4b/calib-evt298567.json \
  --outdir ql_display/png-nonmatch-nm4b/evt298567 \
  --context ql_display/png-nonmatch-nm4b/evt298567/context.jsonl \
  --mode none --groups 60,82,84,99,100,119,122,136,141,142,146,147,151,166,188
# likewise evt298581 --groups 78 (idx 1), evt298651 --groups 85 (idx 6),
#          evt298693 --groups 118 (idx 9)
```

Inputs: `work/039252_<idx>_nm4b/calib-evt*.json` (the adopted nm4b operating
point), truth = gold `work/ql_labels/wfresc/labels-evt298567.json` + AI
`ql_display/decisions-cathxa/decisions-evt*.jsonl` (objective tiers
gold/high/med, long = bbox diag >= 25 cm or >= 100 pts, tol 0.5 µs). Sanity
gates in the script all PASS: 91 missed reproduced (= census 87 C + 4 D),
every missed cluster joined exactly one positive scan line.

## 1. Answer to "did the scans match most of them?" — two different populations

The scorer's **91 "missed"** = scan positives the matcher did not reproduce.
But they are NOT all Bee non-matches:

| matcher outcome | n | where you see it in Bee |
|---|---|---|
| **wrong flash** | **66** | cluster IS matched, at a different flash (median offset 419 µs) |
| truly unmatched | 25 | the Bee "non-match" button |
| anchored at wrong time | 0 | — |

Conversely, the Bee **non-match button population** (all long clusters with no
auto bundle and no anchor ride) is **131** clusters, and the scans matched only
**25** of them (19%):

| scan verdict on the 131 Bee non-matches | n |
|---|---|
| scan matched it (gold/high/med positive) | 25 |
| low-confidence positive only | 20 |
| scanner REJECTED every candidate pairing | 80 |
| no scan verdict | 6 |

So the premise "the scans matched most of the non-matched ones" is **no longer
true at the nm4b operating point**: 61% of the remaining truly-unmatched long
clusters were also left unmatched by the scanners (they examined the candidate
flashes and rejected them all). The scan-backed deficit has largely moved from
"cluster has no flash" to "cluster is on the WRONG flash" — which is invisible
under the non-match button.

## 2. Why each one failed — failure stages over the 91

| stage | n | mechanism |
|---|---|---|
| WRONG_FLASH | 66 | matched elsewhere; truth flash lost the light-metric ranking |
| GATE_FAR_FAIL | 16 | unmatched; truth-time candidate fails rescue gates by >= 20% |
| GATE_NEAR_MISS | 4 | unmatched; fails the relaxed gates by < 20% on its closest gate |
| PASSES_UNADOPTED | 4 | unmatched; candidate passes ALL gates but the rescue never saw the cluster (blind spot, §4) |
| WRONGTIME_NO_BUNDLE | 1 | no contained candidate within 0.5 µs of the scan time |

Key facts about the dominant WRONG_FLASH class (66):

- **Who chose the wrong flash:** the LASSO itself for 48 (strength > 0),
  the relaxed rescue tier for 7 (the doc-21 low-confidence adoptions),
  other rescue paths for 11.
- **It is not a near-miss in time:** median |Δt| = 419 µs; only 2/66 are
  within 5 µs (split-flash twins). These are genuinely different flashes.
- **Brightness is the recurring casualty:** in 43/66 the scan's flash is
  BRIGHTER than the chosen one — chi2/KS are PE-scale-blind and the LASSO
  uses fractional PE, so a dim flash with a tidier shape beats a bright
  flash the model under-predicts (same mechanism as the doc-15 containment
  demotion story).
- **The truth pairing is usually light-plausible:** 46/66 truth-time
  candidates pass even the relaxed rescue gates (median pred/meas 0.98).
  The problem is RANKING among ~190 flashes, not absence of a viable
  candidate — confirming the doc-20 §8 class-C reading, now with the
  sharper statement that in 2/3 of the cases the matcher actively parked
  the cluster elsewhere, where no rescue can ever reconsider it.

The 16 GATE_FAR_FAIL truth-time candidates fail mostly on amplitude
(pred/meas from 0.05 up to 6.95, both directions), i.e. photon-model
error at that topology, not gate mis-tuning.

## 3. Deep dives (renders in `ql_display/png-nonmatch-nm4b/`)

All nine gold (owner-scanned) evt298567 misses plus one exemplar per
non-WRONG_FLASH stage. `grpNNN_gidM.png` = viewer group NNN.

**evt298567 (gold, 9 missed — 8 wrong-flash, 1 far-fail):**

| uid | len | scan flash (grp, PE) | matcher chose (grp, PE, Δt) | why the truth flash lost |
|---|---|---|---|---|
| 18 | 301 | 142, 23.7k | 122, 11.2k, −517 µs (relaxed-rescue adoption) | truth cand passes gates (ks 0.157, χ²/n 0.7) but ratio 0.46 (model 2× under-predicts the bright flash); wrong flash's ratio 1.44 scores better on \|log ratio\| |
| 141 | 270 | 136, 4.4k | 141, 6.1k, +103 µs | truth cand ratio 0.18 (5× under-pred) fails ratio_lo |
| 78 | 172 | 60, 8.4k | 84, 1.1k, +1051 µs (LASSO, strength 0.55) | wrong flash ks 0.090 beats truth 0.116; truth cand passes gates — pure ranking loss |
| 4000067 | 153 | 166, 12.0k | 151, 3.1k, −384 µs | truth cand c2ndf 28, ratio 0.15 — under-prediction |
| 4000079 | 141 | 147, 1.6k | 146, 1.2k, −31 µs | truth cand PASSES (ks 0.069, χ²/n 0.1); neighbor flash won the LASSO |
| 4000101 | 131 | 100, 4.6k | 119, 0.7k, +684 µs | truth cand ratio 0.25 just under ratio_lo 0.3 |
| 80 | 116 | 82, 5.7k | unmatched | GATE_FAR_FAIL: ratio 0.05, χ²/n 285 — model predicts 20× too dim at this topology |
| 64 | 114 | 188, 5.5k | 147, 1.6k, −1265 µs | truth cand ratio 0.09 — under-prediction |
| 97 | 36 | 99, 17.5k | 100, 4.6k, +14.5 µs | truth cand ks 0.285/χ²/n 271/ratio 0.01 — shortest gold miss, severe under-pred |

Note the displacement chains: uid64 stole uid4000079's truth flash (grp 147),
uid97 sits on uid4000101's truth flash (grp 100). Wrong-flash errors cascade —
one under-predicted bright flash can knock two clusters off their homes.

**evt298693 uid4000076 (PASSES_UNADOPTED — the rescue blind spot, §4).**
AI scan: *"the event's brightest flash (39266 PE, 6 railed channels) had NO
auto: c4000076 fits at ks 0.105, chi2/ndf 1.1, pred/meas 0.90, cathode+z_hi
contacts at this T0."* The candidate passes every gate; render grp118.

**evt298651 uid33 (GATE_NEAR_MISS).** AI scan: *"doc-06 golden anode-cathode
crosser, bottom half: at t=300us corrected x[−332,−5] spans the full drift
anode→cathode exactly; overprediction (3.5×) explained by 4 railed-to-zero
cathode XAs (nsat3) plus the 13.9k-PE sibling flash gid83 1.3us earlier."*
Truth cand ks 0.041, χ²/n 0.9, ratio 3.51 — fails ratio_hi 3.0 by 17%. The
overprediction is largely SATURATION on the measured side: the rails zero the
brightest channels, so pred/meas inflates. `chi2_sat_inflate` widens chi2
errors but the rescue RATIO has no saturation awareness.

**evt298581 uid4000032 (WRONGTIME_NO_BUNDLE).** 412 cm shower top,
xtpc-cathode-rescued in the cathxa round (scan: pred/meas 0.81 at 205.8 µs,
55.9k PE flash). In nm4b no contained bundle exists within 0.5 µs — the
candidate never formed at this T0 (containment/admission at build time), the
only case new gates cannot reach.

## 4. The rescue blind spot (PASSES_UNADOPTED, 4 clusters — confirmed mechanism)

evt298693 uid4000076, evt298735 uid163 & uid94, evt298777 uid4000245: all four
are fully unmatched, their truth-time candidate is contained, in the pre-cull
pool, and passes BOTH rescue gate tiers — yet neither tier logged an adoption
(`QLclusrescue` absent for these idents).

Cause, verified in the dumps: each cluster has exactly one OTHER bundle with
LASSO **strength ≈ 0.9 but auto_selected = false** at a WRONG flash (Δt 27–985
µs, ks 0.32–0.51, consistent=false — the post-fit quality gates correctly
refused to select it). That bundle is still live in `flash_bundles_map`, and
`rescue_unmatched_clusters` builds its off-limits set from ALL live bundles,
not auto-selected ones (`QLMatching.cxx:2927-2929`):

```cpp
std::set<Cluster*> matched;
for (auto& kv : run.flash_bundles_map)
    for (auto& b : kv.second) matched.insert(b->get_main_cluster());
```

So a cluster whose only LASSO weight landed on a quality-rejected pairing is
invisible to the rescue forever, even when a gate-passing candidate exists at
the scan's flash. A knob that counts only auto-selected bundles in this test
(default OFF = byte-identical) would put these 4 clusters — including the
39.3-kPE brightest-flash case — back in rescue reach.

## 5. What else could reduce non-matches — ranked by measured impact

1. **Wrong-flash ranking (sizes 66/91, plus its Bee-invisible cascades).**
   The single dominant mechanism, and the doc-20 recall ceiling is exactly
   this: per-bundle light scores cannot rank the true flash among ~190
   rivals. Two levers, both non-byte-identical (knob + rescan validation):
   - *Brightness-aware selection.* 43/66 truth flashes are brighter than the
     chosen one, and the under-prediction direction (truth-cand ratio < 1 in
     33/66, extreme in the gold cases) is the known bright-cathode model
     deficit (doc 12/14: railed-channel meas/pred 3.66). Candidates: an
     absolute-PE term or an asymmetric ratio penalty (punish pred>>meas more
     than pred<<meas at bright flashes) in the LASSO weight / rescue score.
   - *Let the joint LASSO arbitrate instead of pre-culling* (doc 20 §8 lever
     1, still unbuilt): keep the correct-time bundle alive into the fit so
     brightness competition happens inside one solve rather than in the
     per-bundle score.
2. **Rescue blind-spot fix (§4; 4 clusters + interacts with #1).** Small,
   sharply defined, byte-identical-when-off knob on the matched test in
   `rescue_unmatched_clusters`. Cheapest win available.
3. **Saturation-aware rescue ratio (1–3 clusters now, more at every future
   operating point).** uid33-style: compute the rescue pred/meas over
   unsaturated channels only (or inflate ratio_hi when the flash carries
   rails), mirroring what `chi2_sat_inflate` already does for chi2.
4. **Photon model at specific topologies (16 GATE_FAR_FAIL + the ratio-fail
   part of WRONG_FLASH).** The far-fail ratios (0.05–6.95 both directions)
   are model error, not tuning; gold uid80 (ratio 0.05) is the cleanest
   benchmark case. This is the long-term physics floor identified in the
   perf round — no gate change fixes it.
5. **Not worth pursuing:** further gate loosening (doc 21 nm4c was strictly
   bad; the census what-if showed ~1 wrong-flash per real recovery at every
   step) and the single WRONGTIME_NO_BUNDLE case (needs deferred-visibility
   C++ for one cluster).

A useful reframe for the owner: at nm4b, "reduce non-matches" and "reduce
wrong-flash matches" are now the SAME problem — 66 of the 91 scan-backed
misses are already matched somewhere, and pulling harder on rescue recall
(more adoptions) without fixing ranking will convert unmatched into
wrong-flash, not into agreement. The scanners' own verdict on the residual
non-match button population (80/131 rejected everything) says most of what
remains there is genuinely unmatchable light — the recoverable signal is on
the wrong flashes.

## 6. Records

- Full per-cluster metrics: `work/ql_scores/nm4b/nonmatch_explain.{md,json}`
  (this doc's appendix is the compact view).
- Renders: `ql_display/png-nonmatch-nm4b/evt{298567,298581,298651,298693}/`.
- The 15 unlabeled nm4b relaxed adoptions (doc 21 rescan queue) are a separate
  pending item and were not scanned here.

## Appendix — all 91 missed long clusters (compact)

Full metrics (ks/χ²/ratio/strength, flags, rival counts, scan reason text) in
`nonmatch_explain.{md,json}`.

| evt | uid | len_cm | conf | truth t_us | truth PE | outcome | stage | truth-cand fails |
|---|---|---|---|---|---|---|---|---|
| 298567 | 18 | 301 | gold | 3222.1 | 23741 | wrong flash (-517 us, 11182 PE) | WRONG_FLASH | passes |
| 298567 | 141 | 270 | gold | 3112.4 | 4419 | wrong flash (+103 us, 6090 PE) | WRONG_FLASH | ratio 0.18<=0.3 |
| 298567 | 78 | 172 | gold | -113.8 | 8433 | wrong flash (+1051 us, 1064 PE) | WRONG_FLASH | passes |
| 298567 | 4000067 | 153 | gold | 4001.0 | 11983 | wrong flash (-384 us, 3066 PE) | WRONG_FLASH | c2ndf 28.4>=15.0; ratio 0.15<=0.3 |
| 298567 | 4000079 | 141 | gold | 3472.2 | 1644 | wrong flash (-31 us, 1192 PE) | WRONG_FLASH | passes |
| 298567 | 4000101 | 131 | gold | 1960.7 | 4612 | wrong flash (+684 us, 732 PE) | WRONG_FLASH | ratio 0.25<=0.3 |
| 298567 | 80 | 116 | gold | 884.2 | 5693 | unmatched | GATE_FAR_FAIL | c2ndf 285.1>=15.0; ratio 0.05<=0.3 |
| 298567 | 64 | 114 | gold | 4737.0 | 5530 | wrong flash (-1265 us, 1644 PE) | WRONG_FLASH | ratio 0.09<=0.3 |
| 298567 | 97 | 36 | gold | 1946.2 | 17525 | wrong flash (+14 us, 4612 PE) | WRONG_FLASH | ks 0.285>=0.25; c2ndf 270.7>=15.0; ratio 0.01<=0.3 |
| 298581 | 4000032 | 412 | med | 205.8 | 55928 | unmatched | WRONGTIME_NO_BUNDLE | no candidate |
| 298581 | 4000023 | 398 | med | 501.7 | 3082 | unmatched | GATE_FAR_FAIL | ks 0.473>=0.25; ratio 6.95>=3.0 |
| 298581 | 168 | 248 | med | -985.8 | 13886 | wrong flash (+135 us, 8092 PE) | WRONG_FLASH | passes |
| 298581 | 4000348 | 181 | med | 1334.2 | 1209 | unmatched | GATE_FAR_FAIL | ks 0.349>=0.25 |
| 298581 | 4000240 | 86 | high | 2150.4 | 2526 | wrong flash (+842 us, 1610 PE) | WRONG_FLASH | passes |
| 298581 | 4000127 | 37 | med | 682.0 | 316 | unmatched | GATE_FAR_FAIL | ratio 0.19<=0.3 |
| 298595 | 32 | 402 | med | 1429.0 | 6082 | unmatched | GATE_FAR_FAIL | ratio 4.50>=3.0 |
| 298595 | 4000076 | 367 | med | 1429.0 | 6082 | wrong flash (-1 us, 17444 PE) | WRONG_FLASH | passes |
| 298609 | 102 | 335 | med | 2899.7 | 24384 | wrong flash (-207 us, 18967 PE) | WRONG_FLASH | passes |
| 298609 | 4000064 | 108 | med | -1865.9 | 23136 | wrong flash (-17 us, 1438 PE) | WRONG_FLASH | ks 0.305>=0.25; c2ndf 23.3>=15.0 |
| 298623 | 4000017 | 538 | med | 1959.2 | 13966 | unmatched | GATE_FAR_FAIL | ks 0.313>=0.25 |
| 298623 | 133 | 475 | med | 1959.2 | 13966 | unmatched | GATE_FAR_FAIL | ks 0.383>=0.25 |
| 298623 | 4000006 | 472 | med | 2403.7 | 66323 | unmatched | GATE_FAR_FAIL | ratio 3.74>=3.0 |
| 298623 | 5 | 258 | med | -614.1 | 7784 | wrong flash (-1 us, 17342 PE) | WRONG_FLASH | ks 0.529>=0.25 |
| 298623 | 4000016 | 161 | med | -614.1 | 7784 | wrong flash (+1650 us, 1938 PE) | WRONG_FLASH | ratio 3.80>=3.0 |
| 298623 | 4000082 | 78 | med | 2284.6 | 1198 | wrong flash (+6 us, 939 PE) | WRONG_FLASH | passes |
| 298623 | 128 | 17 | med | -1750.0 | 1377 | unmatched | GATE_FAR_FAIL | ks 0.586>=0.25 |
| 298637 | 59 | 159 | med | 1373.5 | 37249 | unmatched | GATE_FAR_FAIL | ks 0.307>=0.25 |
| 298637 | 156 | 128 | med | 1391.7 | 37171 | wrong flash (+67 us, 12012 PE) | WRONG_FLASH | no candidate |
| 298651 | 33 | 408 | med | 300.1 | 7382 | unmatched | GATE_NEAR_MISS | ratio 3.51>=3.0 |
| 298651 | 4000067 | 398 | med | -784.4 | 9691 | wrong flash (-1188 us, 6996 PE) | WRONG_FLASH | passes |
| 298651 | 4000031 | 386 | med | 3709.2 | 22894 | wrong flash (-78 us, 9043 PE) | WRONG_FLASH | passes |
| 298651 | 67 | 189 | med | -490.2 | 4739 | wrong flash (+1121 us, 1586 PE) | WRONG_FLASH | passes |
| 298651 | 4000432 | 176 | med | -1195.0 | 4756 | wrong flash (+1384 us, 1133 PE) | WRONG_FLASH | c2ndf 16.8>=15.0 |
| 298651 | 15 | 121 | med | -1669.4 | 6544 | wrong flash (+1294 us, 5660 PE) | WRONG_FLASH | c2ndf 21.2>=15.0 |
| 298651 | 4000519 | 77 | med | 2700.8 | 20018 | wrong flash (+1900 us, 1991 PE) | WRONG_FLASH | ks 0.323>=0.25; c2ndf 38.2>=15.0 |
| 298651 | 37 | 69 | med | -1972.8 | 6996 | wrong flash (+1454 us, 1180 PE) | WRONG_FLASH | ks 0.354>=0.25; c2ndf 21.2>=15.0 |
| 298651 | 85 | 63 | high | 2875.0 | 1587 | wrong flash (+389 us, 3676 PE) | WRONG_FLASH | passes |
| 298665 | 4000260 | 331 | med | 1392.4 | 9382 | wrong flash (-112 us, 19810 PE) | WRONG_FLASH | passes |
| 298679 | 4000028 | 265 | high | 546.7 | 22646 | unmatched | GATE_NEAR_MISS | ks 0.271>=0.25 |
| 298679 | 4000167 | 172 | med | 1973.0 | 5034 | wrong flash (+1560 us, 2464 PE) | WRONG_FLASH | ks 0.341>=0.25; ratio 4.22>=3.0 |
| 298679 | 5 | 171 | high | 546.7 | 22646 | wrong flash (+1012 us, 3867 PE) | WRONG_FLASH | passes |
| 298679 | 120 | 171 | high | -405.9 | 4091 | wrong flash (-394 us, 34905 PE) | WRONG_FLASH | passes |
| 298679 | 4000257 | 131 | med | 3532.6 | 2464 | wrong flash (-816 us, 9173 PE) | WRONG_FLASH | passes |
| 298693 | 4000245 | 331 | med | 3156.2 | 14542 | wrong flash (-399 us, 15810 PE) | WRONG_FLASH | passes |
| 298693 | 4000076 | 317 | med | 1630.9 | 39266 | unmatched | PASSES_UNADOPTED | passes |
| 298693 | 4000174 | 296 | med | -55.2 | 4233 | wrong flash (-30 us, 5855 PE) | WRONG_FLASH | passes |
| 298693 | 4000270 | 211 | med | 101.5 | 1661 | wrong flash (-30 us, 1636 PE) | WRONG_FLASH | passes |
| 298693 | 129 | 146 | med | -1671.1 | 26690 | wrong flash (+1616 us, 4233 PE) | WRONG_FLASH | passes |
| 298693 | 4000170 | 52 | high | -1100.4 | 1029 | wrong flash (+197 us, 1228 PE) | WRONG_FLASH | passes |
| 298707 | 5 | 273 | med | -892.8 | 19084 | unmatched | GATE_FAR_FAIL | ks 0.500>=0.25 |
| 298707 | 4000004 | 260 | med | -892.8 | 19084 | unmatched | GATE_FAR_FAIL | ks 0.606>=0.25 |
| 298707 | 4000098 | 180 | med | 3593.9 | 684 | wrong flash (+472 us, 1117 PE) | WRONG_FLASH | passes |
| 298721 | 4000303 | 268 | med | 2890.8 | 29848 | unmatched | GATE_NEAR_MISS | ks 0.370>=0.25; c2ndf 20.6>=15.0; ratio 0.27<=0.3 |
| 298721 | 23 | 254 | med | -630.0 | 7521 | wrong flash (-680 us, 2790 PE) | WRONG_FLASH | passes |
| 298721 | 63 | 199 | med | 3697.1 | 5889 | wrong flash (-440 us, 6350 PE) | WRONG_FLASH | passes |
| 298721 | 36 | 138 | med | 3689.7 | 3175 | wrong flash (+378 us, 3990 PE) | WRONG_FLASH | passes |
| 298721 | 100 | 40 | med | -2138.6 | 24133 | wrong flash (+35 us, 19332 PE) | WRONG_FLASH | no candidate |
| 298735 | 4000030 | 300 | med | -1152.9 | 10154 | wrong flash (-46 us, 10901 PE) | WRONG_FLASH | passes |
| 298735 | 126 | 287 | med | 2911.0 | 13795 | unmatched | GATE_NEAR_MISS | ks 0.355>=0.25; c2ndf 15.8>=15.0 |
| 298735 | 163 | 220 | med | -1172.2 | 6856 | unmatched | PASSES_UNADOPTED | passes |
| 298735 | 4000148 | 188 | med | 1226.2 | 6605 | wrong flash (-707 us, 4558 PE) | WRONG_FLASH | passes |
| 298735 | 4000045 | 180 | med | 2420.5 | 1328 | wrong flash (-276 us, 10231 PE) | WRONG_FLASH | passes |
| 298735 | 3 | 162 | med | 81.1 | 1267 | wrong flash (+438 us, 4558 PE) | WRONG_FLASH | passes |
| 298735 | 130 | 145 | med | 278.7 | 1081 | wrong flash (+142 us, 3278 PE) | WRONG_FLASH | passes |
| 298735 | 110 | 127 | med | 1226.2 | 6605 | wrong flash (-707 us, 4558 PE) | WRONG_FLASH | passes |
| 298735 | 4000314 | 127 | med | -1199.3 | 10901 | wrong flash (-23 us, 688 PE) | WRONG_FLASH | passes |
| 298735 | 94 | 92 | med | 3833.1 | 2439 | unmatched | PASSES_UNADOPTED | passes |
| 298735 | 133 | 20 | med | -2239.7 | 1351 | wrong flash (+1346 us, 355 PE) | WRONG_FLASH | no candidate |
| 298749 | 9 | 278 | med | 60.7 | 12824 | unmatched | GATE_FAR_FAIL | ks 0.469>=0.25 |
| 298749 | 4000002 | 76 | med | 60.7 | 12824 | wrong flash (+1545 us, 639 PE) | WRONG_FLASH | ks 0.674>=0.25 |
| 298763 | 4000006 | 316 | med | 1312.7 | 7657 | wrong flash (-240 us, 11194 PE) | WRONG_FLASH | passes |
| 298763 | 63 | 154 | high | 1515.9 | 2275 | wrong flash (+321 us, 2460 PE) | WRONG_FLASH | passes |
| 298763 | 4000354 | 112 | med | 3963.9 | 21831 | wrong flash (-529 us, 10766 PE) | WRONG_FLASH | ratio 0.14<=0.3 |
| 298777 | 4000033 | 405 | med | 1700.2 | 24889 | wrong flash (+502 us, 29849 PE) | WRONG_FLASH | passes |
| 298777 | 4000245 | 340 | med | 523.1 | 10076 | unmatched | PASSES_UNADOPTED | passes |
| 298777 | 67 | 337 | med | 49.9 | 22838 | wrong flash (+626 us, 7846 PE) | WRONG_FLASH | passes |
| 298777 | 125 | 128 | med | -2131.6 | 9382 | wrong flash (+36 us, 1593 PE) | WRONG_FLASH | passes |
| 298777 | 4000250 | 109 | high | -1384.3 | 6108 | wrong flash (+1180 us, 4922 PE) | WRONG_FLASH | passes |
| 298777 | 77 | 83 | med | -1799.0 | 6994 | wrong flash (+1433 us, 1130 PE) | WRONG_FLASH | ratio 3.17>=3.0 |
| 298791 | 4000021 | 582 | med | 61.9 | 46524 | wrong flash (+685 us, 19595 PE) | WRONG_FLASH | passes |
| 298791 | 189 | 393 | med | 1820.1 | 17401 | wrong flash (+158 us, 13559 PE) | WRONG_FLASH | passes |
| 298791 | 16 | 365 | med | 2581.0 | 21448 | unmatched | GATE_FAR_FAIL | ks 0.542>=0.25 |
| 298791 | 1 | 355 | med | 2735.3 | 28695 | unmatched | GATE_FAR_FAIL | c2ndf 51.3>=15.0 |
| 298791 | 4000613 | 340 | med | 1978.1 | 13559 | wrong flash (-190 us, 19044 PE) | WRONG_FLASH | passes |
| 298791 | 85 | 260 | med | 746.8 | 19595 | wrong flash (+812 us, 9577 PE) | WRONG_FLASH | ks 0.436>=0.25 |
| 298791 | 4000551 | 94 | med | -1484.3 | 6315 | wrong flash (+552 us, 1984 PE) | WRONG_FLASH | passes |
| 298805 | 4000033 | 391 | med | 2841.2 | 23066 | unmatched | GATE_FAR_FAIL | ks 0.353>=0.25 |
| 298805 | 4000120 | 235 | med | 4365.0 | 6520 | wrong flash (-106 us, 910 PE) | WRONG_FLASH | passes |
| 298805 | 4000214 | 218 | med | -983.2 | 6973 | wrong flash (+590 us, 2349 PE) | WRONG_FLASH | passes |
| 298805 | 4000286 | 191 | med | 1119.1 | 1926 | wrong flash (+178 us, 3961 PE) | WRONG_FLASH | passes |
| 298805 | 4000263 | 116 | med | 985.0 | 1720 | wrong flash (+312 us, 3961 PE) | WRONG_FLASH | passes |

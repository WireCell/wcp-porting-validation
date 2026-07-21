# AI hand scan of the cathxa round — findings (run 039252, idx 1-17)

Full AI (Claude) hand scan of 17 of the 18 events of the cathode-XA
operating-point reprocess (`work/039252_<idx>_cathxa/`, doc 18), performed
2026-07-16/17 with the pipeline of `ql-scan-method.md` and the rubric of
`ql-scan-criteria.md`. Event idx 0 (evt298567) was not re-scanned — it
carries the owner's gold scan (doc 17) remapped into tag `cathxa`; it served
as the calibration example. One scanner agent per event, 6 in parallel,
three waves.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
PY=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/python

# evidence per event (idx 1..17, evt = 298567 + 14*idx)
$PY ql_display/render_groups.py work/039252_<idx>_cathxa/calib-evt<ID>.json \
    --outdir ql_display/png-cathxa/evt<ID> \
    --context ql_display/context-cathxa/context-evt<ID>.jsonl

# decisions (committed record) -> viewer labels, per event
$PY ql_display/make_labels.py work/039252_<idx>_cathxa/calib-evt<ID>.json \
    ql_display/decisions-cathxa/decisions-evt<ID>.jsonl --tag claude-cathxa --check

# review interactively (picks auto-restore per event)
./ql_scan/serve_ql_scan.sh 5021 --tag claude-cathxa work/039252_*_cathxa/calib-evt*.json

# headline table + keep/reject metric quantiles
# (aggregate_scan.py from the session scratchpad; numbers below)
```

All `make_labels --check` runs returned rc=0 (labels + scan_state
round-trip). Existing records (`work/ql_labels/cathxa/`, `wfresc/`, older
`decisions-*` families) were not touched; the scan lives entirely under the
new decisions dir `ql_display/decisions-cathxa/` and the new label tag
`claude-cathxa` (M13).

## Headline: per-event verdicts

"judged" = auto-selected bundles with main cluster > 5 cm (each carries an
explicit keep/reject); short autos below the viewer filter are implicit
keeps, as in the human scans. agree = keep / judged.

| idx | event     | judged | keep | reject | add | agree% |
|----:|-----------|-------:|-----:|-------:|----:|-------:|
|  1  | evt298581 |   71   |  53  |   18   |  8  |  74.6  |
|  2  | evt298595 |   82   |  58  |   24   |  1  |  70.7  |
|  3  | evt298609 |   69   |  43  |   26   |  6  |  62.3  |
|  4  | evt298623 |   84   |  52  |   32   |  3  |  61.9  |
|  5  | evt298637 |   76   |  57  |   19   |  7  |  75.0  |
|  6  | evt298651 |   83   |  55  |   28   | 12  |  66.3  |
|  7  | evt298665 |   69   |  42  |   27   |  5  |  60.9  |
|  8  | evt298679 |   63   |  37  |   26   |  6  |  58.7  |
|  9  | evt298693 |   76   |  41  |   35   |  8  |  53.9  |
| 10  | evt298707 |   66   |  50  |   16   |  7  |  75.8  |
| 11  | evt298721 |   71   |  45  |   26   |  8  |  63.4  |
| 12  | evt298735 |   76   |  50  |   26   | 14  |  65.8  |
| 13  | evt298749 |   56   |  43  |   13   |  4  |  76.8  |
| 14  | evt298763 |   73   |  51  |   22   |  8  |  69.9  |
| 15  | evt298777 |   87   |  68  |   19   | 10  |  78.2  |
| 16  | evt298791 |   98   |  57  |   41   |  9  |  58.2  |
| 17  | evt298805 |   79   |  58  |   21   |  6  |  73.4  |
| **all** | | **1279** | **860** | **419** | **122** | **67.2** |

Confidence mix over all 1401 decision lines: 691 high / 525 med / 185 low.
(Low-confidence and at_cathode decisions are the ones to re-check first.)

For reference, the 10-event geomfix round ran at 58.7% agree; the cathxa
operating point is a net improvement (67.2%), with the remaining
disagreement dominated by a few identifiable pathologies (§4).

## Blind cross-check on evt298581

The owner saved a scan of evt298581 under tag `cathxa` shortly before this
round (34 picks, 21:28). The AI scanned the event blind (agent forbidden to
open those files):

- **Cluster level: 30 of the owner's 32 picked clusters kept (94%).**
- Pair level: 19/34 identical (flash_gid, uid) pairs. 13 of the 15
  differences keep the same cluster on an **adjacent gid** (flash pairs
  1-7 us apart: 16→15, 47→48, 76→77, 79→80, 105→106, 124→125, 126→121,
  147→148, 22→26, 85→116) — the flash-splitting mode of §4.3, where the
  remap's nearest-time carry can also land on the wrong twin.
- Outright drops: uid 47 (AI kept only its top partner 4000041, on gid 26)
  and uid 109 (AI rejected: pred/meas 0.00, chi2/ndf 158 at gid 192 — a
  deliberate disagreement worth review).

## Kept vs rejected: metric quantiles (q25/50/75, judged autos only)

| metric | kept (n=860) | rejected (n=419) |
|---|---|---|
| ks_dis            | 0.060 / 0.088 / 0.184 | 0.088 / 0.218 / 0.431 |
| chi2/ndf          | 0.6 / 1.9 / 7.4       | 2.6 / 12.4 / 44.9     |
| pred/meas         | 0.45 / 0.75 / 1.12    | 0.07 / 0.36 / 1.30    |

Same qualitative separation as the PDHD human scan (amplitude first, shape
second), with the PDVD caveats intact: kept crossers run up to ks ~0.5 when
cathode XAs are railed, so none of these were used as hard ceilings.

## Systematic findings

1. **xtpc pin/rescue is the #1 error vector.** Across the 17 events, roughly
   a third of pinned/rescued pairs are phantoms: the pin binds a genuine
   (or fake) crosser to a dim accidental flash with joint overprediction
   x4-x450 (worst: 273+260 cm pair on an 81 PE flash, evt298707 grp49; a
   26 PE phantom at evt298791 gid117; p/m ~150 at evt298735 gid163; 409x
   rail-tail capture at evt298749 gid81). Recurrent mechanisms:
   - **no amplitude sanity gate** on pin/rescue candidacy — strength-0
     LASSO members latch onto any time-compatible dim flash;
   - **equal-drift-fade fake crossers** (evt298777): two unrelated tracks
     whose charge fades at equal raw drift length fake a cathode meeting at
     the T0 that puts the fade points on the cathode;
   - **rescue admits bulk-OOB clusters**: rescued bundles with 20-99% of
     charge beyond the cathode at the rescue T0 (evt298763 gid65: 99%;
     evt298581 gid126: 76%; evt298805 gid170 stubs: 20-57%);
   - **partner-less rescues** (evt298665 gid87, evt298749 gid147).
   Good pins meet at dyz 2-9 cm; phantoms at 13-93 cm. **Meet distance +
   an unsaturated-channel amplitude term + end-direction collinearity would
   kill nearly all phantoms** — direct input for the rescue-knob census
   (doc 18 open item). Genuine finder-verified crossers were essentially
   always confirmed (e.g. 8/8 on evt298665, 13/13 on evt298805).

2. **One-cluster-many-flashes accidentals remain the dominant reject mode**
   (as in the previous AI round): amplitude-starved bundles (pred/meas
   ≤ 0.1) of T0-degenerate cathode stubs or floaters on bright flashes.
   Several re-homings followed one template: cluster auto'd/pinned onto a
   nearby-in-time flash while a brighter unmatched flash fits at p/m ≈ 1
   and cos ≈ 1.

3. **Flash splitting produces twin flashes 1-7 us apart** (rail tails,
   afterpulses, or genuine reco splits: evt298749 gid79/80/81, evt298707
   gid22/23/24, evt298791 gid101/102, evt298805 gid197/198, several in
   evt298581/298623). Consequences: duplicate autos of one cluster on both
   twins, pins landing on the dim twin (apparent 3-5x overprediction that
   is 1.2-1.8x against the combined light), and sparse one-channel
   fragments (evt298791 gid102 ch7-only) attracting phantom matches. A
   flash-merge or twin-veto in admission would remove a whole error class.

4. **Saturation handling decides the hard cases.** Railed cathode XAs
   frequently read ZERO (not clipped) on bright flashes, destroying ks/cc
   for genuinely correct matches; the unsaturated-channel amplitude ratio
   was the single most decisive discriminator all round. Two artifacts to
   look at: sat flags appearing on dim flashes (17-71 PE channels flagged,
   evt298595/298665/298679) and rail floors varying between flashes on the
   same channel (evt298791). On several big flashes every unsaturated
   cathode channel reads exactly 0 (per-flash auto-mask), leaving no
   amplitude ruler at all (evt298777 gid19/82/137).

5. **Masked-light amplitude dilution** (evt298665 gid111 is the clean demo):
   with wall XAs masked from the fit, a flash whose total_PE is 92% masked
   wall-XA light let a bundle pass the amplitude gate at 0.84 total while
   overpredicting the trusted cathode ruler x12. Amplitude/total_PE terms
   (and reject_overpred) should be computed on unmasked channels only at
   this operating point.

6. **Detached-point tails make bbox extents unreliable** (clus-34 family,
   now measured at scale): kept clusters routinely carry 0.1-5% of points
   20-79 cm past the anode/cathode; two containment-gate leaks were
   auto-selected with drift-corrected bulk 15-79 cm outside the anode
   (evt298707, evt298735). At the extreme, clusters whose apparent x-span
   exceeds the 337 cm drift depth get **zero bundles anywhere** — silent
   orphans (evt298763 uid1: 553 cm, 3291 pts, almost certainly the source
   of the unmatched 60 kPE flash gid161; evt298791 uid4000005; evt298707
   uid14). Robust-percentile endpoints (the default-OFF `robust_endpoint*`
   knob family) would address both directions; this is the strongest
   argument yet for enabling that census.

7. **Window truncation behaves lawfully and is exploitable.** Clusters
   raw-clipped at tick 0/10000 carry a T0-independent edge signature
   (±2 cm), which cleanly separates truncation-excused floaters from
   genuine no-entry rejects — but it excuses the anode-side end only
   (evt298721 gid18 counter-example). Since wtrunc biases the prediction
   LOW, **overprediction under wtrunc is a reliable reject signal**; the
   matcher repeatedly did the opposite, pinning window-clipped tracks to
   dim accidentals at x4-54 (evt298805). Late-window (t ≳ 3 ms) auto
   quality collapses (evt298609: 10 of 21 late autos rejected).

8. **Unexplained bright flashes** persist after the scan in most events —
   three classes: charge outside the readout window (early/late, expected),
   railed-to-zero flashes with no usable ruler, and a residual of genuinely
   sourceless 5-60 kPE flashes (e.g. evt298623 gid19, evt298693 gid140
   bimodal, evt298735 gid95, evt298791 gid136) worth a light-reco look.

9. **Bookkeeping oddities:** ident=-1 clusters appear as bundle main
   clusters (uid 3999999 collides — three distinct clusters share it on
   evt298651; scan_state restore is ambiguous there); the matcher still
   emits one-cluster-two-flashes duplicates both above and below the 5 cm
   filter (make_labels warns); `render_groups` evidence for idx 0-6 from
   the 21:37 batch was truncated mid-write (killed job) — agents fell back
   to calib-dump enumeration (make_labels --check enforces completeness),
   and the evidence files were regenerated to full length afterwards.

## Adds (122 total)

Every add is a `verdict:"add"` line with its reason in
`decisions-cathxa/decisions-evt<ID>.jsonl`. By category:

- **LASSO-stranded long tracks with an obvious unmatched flash** (~35):
  e.g. 421 cm anode-to-cathode through-goer → gid70 (evt298581); event's
  biggest cluster 22861 pts → gid90 (evt298651); 582 cm horizontal → gid72
  (evt298791); brightest-flash owner 4000076 → gid117 (evt298693).
- **Re-homings off phantom pins/rescues onto the true flash** (~45): the
  §4.1 rescues — e.g. crosser pair 5+4000004 → gid49 (evt298707), pair
  27+4000005 → gid155 (evt298777), 144 → gid118 (evt298791), window-clipped
  stubs 4000214→37 / 79→59 (evt298805).
- **Crosser partner halves** (~15): second volume's half added where only
  one half was auto (rubric criterion; e.g. 4000021+102 → gid163 pair,
  evt298609; 4000026 → gid117, evt298637).
- **Split-flash reassignments to the bright twin** (~10) and
  **multi-cluster flash completions** (~15) (e.g. 4000110 → gid44 closes
  the budget 0.55→0.92, evt298707; 4000354 → gid204, evt298763).

## Per-event notes (most interesting case each)

- **evt298581**: five re-homings share the one-cluster-many-flashes +
  pin/rescue template; u4000281 auto had 76% of charge below the cathode.
- **evt298595**: gid106 "overprediction" was a measurement artifact — six
  of eight cathode XAs read exactly 0 on a 6 kPE flash (only railed ch5/9
  read out).
- **evt298609**: three-way untangle at gid52/53/59; cathode-rescue produced
  pred/meas 1583 on a 29 PE flash.
- **evt298623**: pin bound a 538+475 cm through-crosser to a 367 PE flash
  31 us after its true 14 kPE flash (94x/60x ignored by the pin).
- **evt298637**: matcher split one physical crosser across two flashes 18 us
  apart (gid117/118); reunified on meeting geometry.
- **evt298651**: doc-06 golden crosser confirmed at t=300.1 us; its 3.5x
  apparent overprediction is 4 railed-to-zero XAs + a sibling flash 1.3 us
  earlier.
- **evt298665**: masked-light amplitude leak (§4.5); anode-hugger autos are
  systematically amplitude-starved on the cathode ruler (model predicts
  almost nothing for near-anode tracks — worth a photon-model look).
- **evt298679**: phantom pin chose a 1.2 kPE flash over the true 22.6 kPE
  flash 21.5 us earlier; meet distance *improves* at the right flash.
- **evt298693**: gid157 pin pair sums to exactly 1.00 on unsaturated
  channels while nominal p/m looked 1.9x — rails again.
- **evt298707**: readout-truncation edge signature (§4.7) proved decisive;
  worst mis-pin of the round (81 PE flash, p/m ≈ 470).
- **evt298721**: three phantom pins in one event; systematic 30-50 cm anode
  overshoot tails on ≥6 tracks (§4.6).
- **evt298735**: two-cosmic pattern swap (ch4/6 vs ch8/10 inversion) that
  amplitude alone could not see; 14 adds — most of the round.
- **evt298749**: cleanest event (76.8%); gid81 rail-tail flash captured a
  real crosser pair whose true flash is 1.9 us earlier.
- **evt298763**: 553 cm cluster with zero bundles anywhere (silent orphan,
  §4.6) next to the event-ending unmatched 60 kPE flash.
- **evt298777**: equal-drift-fade phantom crossers (new pathology, §4.1);
  both cathode-rescue firings wrong.
- **evt298791**: busiest event (98 autos); a genuine crosser was pinned to a
  26 PE accidental *because* a 3.3 cm detached-point anode overshoot made
  the true flash look uncontained — containment preference beat amplitude.
- **evt298805**: window-clipped tracks pinned to dim accidentals at x4-54
  while their bright flashes sat orphaned; 13/13 finder crossers clean.

## Caveats

- The scan is AI judgment, not human-verified. Low-confidence lines (185)
  and everything flagged "for review" in the reasons deserve first pass.
- at_cathode bundles remain weakly T0-constrained (cathode XAs see both
  volumes); those keeps are conservative and often med/low confidence.
- The per-channel PE scale is still uncalibrated at x-few; amplitude
  arguments were made against the round's own norm (pair overpredictions
  of x1.1-1.8 on unsaturated channels are common for good crossers —
  possibly a QtoL or rail-floor effect worth refitting on these labels).

## Next steps proposed

1. Owner review of the `claude-cathxa` tag in the viewer (port 5021
   command above), starting from low-confidence + at_cathode picks and the
   evt298581 uid47/uid109 disagreements.
2. Feed §4.1 into the rescue-knob census: add amplitude-sanity +
   collinearity + meet-distance gates to xtpc pin/rescue candidacy.
3. Robust-percentile endpoint census (§4.6) — the silent-orphan clusters
   are unmatchable at any T0 without it.
4. Flash-splitting: twin-veto or merge in flash admission (§4.3).
5. Compute amplitude/reject_overpred on unmasked channels only (§4.5).

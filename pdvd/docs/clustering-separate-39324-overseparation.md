# Run 39324: band over-separation, crossing segment swap, 102k giant (round 4)

User-flagged issues on the post-FV-round bee set (dc7c47f6), all in run 39324
(`work/039324_N`, art event = 339850 + 20·N).  Toolkit-side mechanisms and
thresholds in `clus/docs/clustering-separate-refine.md`; this note pins the
per-event evidence.

## 1. Iso-band over-separation (6 flagged pairs, group4567)

| evt | flagged pair | diagnosis |
|---|---|---|
| 339870 | (−109.3,−200.7,113.6) / (−109.7,−217.2,53.0) | one band hatched into 2 interleaved clusters |
| 339930 | (−141.3,−310.0,187.2) / (−139.4,−233.4,157.1) | band hatched + a ~70 cm chunk fused into a crossing track member |
| 339990 | (293.2,−45.5,6.0) / (290.0,−68.3,98.3) | hatched (this one pre-dates the FV round) |
| 340010 | three pairs (73/56, 53/52, 69/51 in the old ids) | hatched |

Every pair is ONE physical isochronous band (common x-slab, verified
visually): `Separate_1`'s path carve — newly firing on band topologies after
the FV insets — cuts a wide band into interleaved alternating chunks, and
NOTHING re-assembled them.  Decisive negative: in the whole 11-event run
neither `band_recarve` nor `track_recarve` ever fired (zero stdout lines in
the batch logs — note stdout goes to `.batch_clus_*.log` / the per-event run
log, NOT the spdlog `wct_clus_*.log`).  The "special two-band separation"
algorithm was NOT the splitter; 4 of 5 events were one cluster in
`bak-pre-fv`.

Fix: `band_merge_back` (anchor-grouped pool merge + band interior steal).
The discriminators were measured offline before implementation:

| case | anchor offset gap | bimodality Δμ/(σᵢ+σⱼ) | outer/inner rms |
|---|---|---|---|
| hatched single bands (merge) | ≤ 67 cm | 0.10–1.26 | ≤ 1.48 |
| evt 339850 two parallel touching bands (keep!) | 130 cm | 2.27 | ~1.0 |
| two bands crossing as X (keep) | — | ~0 | ~2.0 |

Iteration history (all caught by the instrumented per-case matrix):
single-linkage over all pieces chained 339850's two parallel bands through
overlap debris → anchor-based grouping; a 65 cm gap missed one 340010 piece at
66.8 cm → 85 cm; the 339930 chunk (width 5.3 cm) failed the ≥6 cm pool width
gate → thin-debris tier (width ≥2 cm), which then ate PDHD 40908's 318/503 cm
thin iso-aligned TRACKS → 100 cm debris length cap.

## 2. Crossing-track segment swap (339990 group0123)

(27.7,159.1,253.6) + (6.6,173.9,260.2) belong to one crosser,
(90.4,126.5,268.7) + (27.1,151.5,225.5) to the other; the first point rode in
the wrong cluster — a 48.3 cm mid-track chunk (1.0° collinear with its true
track, 2 cm off its line; 20–40 cm off the host's) fused into the host at the
49.6° crossing.  Fix: `track_repartition` (pairwise k=2 3D refit with a
moved-run no-op guard).  On PDHD the guard skips 5 of 8 candidate pairs
(crossing-region dribble) and the 3 that fire move genuine 73–110 cm runs;
all five PDHD 27409 separation invariants still pass.

## 3. 339890 group4567 giant (4 iso bands + 3 vertical tracks, never separated)

The 102,129-pt cluster sat 2 % over the `max_hull_points=100000` cap —
`get_hull` returned empty and Dec_2 silently skipped it (exactly one
`(cap 100000)` warning in the log).  Config fix: cap raised to 1,000,000 on
PDVD+PDHD.  After the raise the event separates (largest cluster 44.7k);
verticals and bands come apart.

**Known residual**: one 44.7k piece still holds two same-yz-slope bands at
DIFFERENT x-slabs bridged by a crossing track.  None of the refinements is
x-aware (merge_back works in the y-z projection; the two bands' offset means
sit 96 cm apart, past the 85 cm anchor gate, so it correctly refused to
merge them with the rest).  Splitting it would need an x-slab-aware split —
left for a future round if flagged.

## Verification

- Per-case matrix (`/home/xqian/tmp/check_39324_round4.py`): 11/11 PASS —
  six merges, the 0123 swap (both groups whole, mutually distinct), 339850's
  parallel bands distinct, giant gone.
- 39324 evt 0 invariants (three pairs, band complex = 2, vertical whole and
  separate): PASS.
- PDHD 27409 five separation events (40900/40904/40908/40912/40924
  user-point distinctness, `/home/xqian/tmp/check_27409_sep.py`): PASS.
- OFF-check: new binary + stashed (old) configs on 39324 evt 3 + 27409 evt 0:
  all mabc zips content-identical to `bak-pre-mergeback`.
- Full regression vs `bak-pre-mergeback`: PDVD 39324 0–10 + 39252/39253,
  PDHD 027409 + 027380 — flow analysis: separation-family reorganizations
  only.

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

**Known residual** (round 4): one 44.7k piece still holds two same-yz-slope
bands at DIFFERENT x-slabs bridged by a crossing track.  None of the round-4
refinements is x-aware.  **RESOLVED in round 5** by `iso_slab_split` — see
below.

# Round 5 (user scan of the full-run bee sets 74d1b7cb / ed8ac098)

Six new cases; diagnosis + fixes in toolkit d40bbe25 (doc
`clus/docs/clustering-separate-refine.md`):

| case | diagnosis | fix |
|---|---|---|
| 27409 40900 under-sep (35k cl) | band_merge_back REGRESSION: two distinct same-slab 14.8k complexes merged as "4 band pieces" with union rms 58 cm (bands read 17–43) | union-rms cap ≤50 cm |
| 27409 40924 over-sep | one straight cosmic in two touching pieces (6.5°, 0.3 cm); nothing rejoins ≥50 cm members | new `collinear_member_merge` |
| 27409 40920 over-cluster | downstream connect1 merged two cosmics across carve-fragment debris; rejoining the fragments into their own tracks first removes the bait (deterministic, verified two identical reruns) | `collinear_member_merge` (side benefit) |
| 339890 cl47 (the round-4 residual) | 2 iso bands at x-slabs 50 cm apart (24k @ x∈[59,67], 14k @ [115,123]) + 3 drift tracks chaining them under 5 cm connectivity; y-z mechanisms are x-blind | new `iso_slab_split`: 4 tracks + 2 bands, all five flagged structures distinct |
| 339990 cl41/42 | band-interior-steal REGRESSION: 193 cm run spanning 66 cm in x (a real drift track) stolen into the band; second fusion was in the carve itself | steal x-gate (run x-extent ≤20 cm) + `iso_slab_split` |
| 340010 75/87 merge | already merged in the full-run reprocess (the scanned set predated it) | none needed |

Tuning history worth keeping: the first `collinear_member_merge` cut
(union rms ≤12 cm) admitted a fork-adjacent pair in 27409 evt 40904
(8.1 cm) whose rejoin made downstream connect1 fuse the two full fork
prongs — the round-1 verified split.  All 12 genuine rejoins across the
suite read ≤5.7 cm, so the gate is pinned at 7 cm.

Round-5 verification: new checker `/home/xqian/tmp/check_round5.py` 6/6;
round-4 matrix 11/11; PDHD five-event invariants 5/5; 39324 evt-0
invariants; OFF-check content-identical; regression flows confined to
separation families plus group-stage deghost/isolated re-judgment of
newly-freed structures (largest: 27380 evt 1 drops a 5.5k single-slab
ghost-like sheet; user-scanned runs change ≤0.2%).

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

# Scan brief — the wider per-part label set  (tag `splitscan-0903-wide`, 32 objects)

**Why you are being asked for these.** Three separate measurements in doc pr/139
have now hit the same wall, and it is not a physics wall — it is a sample-size
wall in the label set:

- **§13.3** — on the `max_parts = 3` arm, three instruments gave three answers
  (merged target better, π⁰ census worse, boundary agreement worse). The
  tie-breaker is boundary agreement against your per-part labels, and it rests on
  **19 confirmed cuts and 29 hand parts.**
- **§17** — the seed cap `max_seeds = 4` binds on **76 %** of everything the
  splitter fires on, and on **all four** objects you cut into k ≥ 3. Raising it is
  now a knob, but grading the result needs labels on seed-capped objects and
  there are **four**.
- **§18** — the last false fire, 278420/61027, sits inside the confirmed-cut
  range on every tape feature. Three of its four "outlier" margins are under 1 %,
  and one is smaller than the instrument's own precision.

Roughly thirty more per-part verdicts would about double the resolving power of
all three at once.

## What to do

```
./split_display/serve_split_display.sh 5022 \
    --scan-tag splitscan-0903-wide --set docs/pr/pr140-scan-set.tsv
ssh -o ServerAliveInterval=30 -L 5022:localhost:5022 <user>@wcgpu1.phy.bnl.gov
#   then http://localhost:5022/split_viewer
```

The tag is fresh; the viewer refuses to write into a tag holding labels it did
not create. All 32 objects were verified to load before this brief was written.

**Per object, the same two questions as before**, and they are independent:

1. **KEEP or SPLIT** — does this object hold charge belonging to more than one
   thing?
2. **If SPLIT: where, and into how many parts** — drag each segment into its
   part. **The `k` matters as much as the boundary**: §17 exists because you
   labelled 396222 at k=7 and the kernel can only ever propose 4.

## The four strata, and what each one decides

| stratum | n | why these objects |
|---|---|---|
| **S3-unjudged-fire** | 8 | the splitter actually peels these today and nobody has said whether it should. Direct trigger purity. |
| **S1-seed-capped** | 8 | `n_seed` is pinned at the cap of 4. These are the objects a `max_seeds` arm will change, and they are unlabelled. |
| **S2-bound-region** | 8 | impact parameter `b` in 15–60 cm — where the `b ≤ 30` bound's three suppressions and the surviving false fire all live. |
| **S4-control** | 8 | seeded-random (seed 20260901), chosen **independently of every feature above**, so the set is not selected by the hypotheses it will test. |

The strata are in the `stratum` column of the set TSV, and the viewer does not
show them to you while you judge — the sheet stays blind.

## What will be done with the answers

- re-run §13's `max_parts` and the new §17 `max_seeds` arms and grade both on the
  enlarged boundary set, with SPLIT2 non-degradation as a pre-registered bar;
- re-price the `b ≤ 30` bound on a bound-region sample that was not chosen by `b`;
- re-ask §18's separability question with more than one negative.

**Nothing will be flipped on these labels without showing you the arm first.**

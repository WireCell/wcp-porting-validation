# Round-3 machinery check — bar declared BEFORE any pick was scored

Written after `prepare` and before the scanners returned. Kit is at the
committed state `f9c6efa` (`git status` clean on `vtx_rules/`, `pr_display/`;
`scankit.py selftest` = 25 dumps, 0 failures).

## What this run is

An execution of doc pr/80 §11 on a **fresh, differently-composed 60**: 20 nueCC
(`vtxscan-prod0813`), 12 NCpi0 (`-ncpi0`), 28 BNB-inclusive (`-mcp1k`, a
numuCC-dominated proxy — there is no per-event truth channel in the dumps).
None of the 60 has been shown to any AI scanner before. Purpose is to confirm
the machinery runs, not to produce a new headline.

Single arm, new kit, four scanners. No noise-floor arm and no old-kit arm: the
§10 five-arm structure existed for a tooling A/B and does not apply to a smoke
test.

## The bar, declared now

**§10's 45/60 is NOT the bar.** The composition is deliberately different —
nueCC/NCpi0-enriched against §10's mcp1k-dominated sample — and those topologies
may well score differently for both the scanner and the reconstruction. Quoting
the old absolute number against this sample would be comparing two things that
differ in more than one way.

The composition-independent check is **the scanner against the reconstruction on
these same 60 events**:

| observation | reading |
|---|---|
| scanner within ~2 events of the reco | machinery working; same relationship as §10 (42 vs 43, 45 vs 46) |
| scanner 3–9 events behind | inconclusive at n=60; report, do not diagnose |
| scanner ≥10 events behind the reco | something drifted — investigate before trusting the kit on new files |

Secondary, and it is the property the owner's workflow depends on:

- confidence tiers must still **order** (certain > likely > unclear);
- `certain`-tier accuracy **≥ 90%**, reported beside its **coverage**.

Scored by **vertex id**, not distance: labels were taken on `-prod0813` and these
dumps are `-ma10`, and `improve_vertex` refits after the vertex is chosen, so the
same vertex id moves between arms (§10.9 / finding F2).

`review` is run first and is the primary exercise — it is the step a future
new-files session actually hits, and it is the only step available when there is
no truth. `score` is a bonus these 60 happen to allow because they are labelled.

## Cost being paid, recorded here rather than discovered later

17 of the 60 come from the **test half** (9 nueCC, 8 NCpi0), listed in
`test-half-consumed.txt`. They are no longer held out. This was forced, not
chosen: only 11 unused dev-half nueCC and 4 unused dev-half NCpi0 events exist
after the §10 arms, so the requested mixture cannot be built from the dev half.
All 28 BNB-inclusive events are dev-half; the test half's 214 unused mcp1k events
are untouched.

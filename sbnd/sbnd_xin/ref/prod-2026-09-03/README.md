# ref/prod-2026-09-03 — SBND production operating point

**What moved from `prod-2026-09-02`, and on whose word.**

Owner, 2026-09-02, after confirming the contamination audit:
*"please turn on this knob for SBND production."*

`stm_entry_rise_guard` now defaults **true** in
`cfg/pgrapher/experiment/sbnd/{clus,wct-pr-perevt}.jsonnet`. The C++ defaults
in `clus/src/TaggerCheckSTM.cxx` stay `false / 1.3 / 5 / 60 / 70 / 22`, so no
other detector moves.

## The drift — six keys on one component, nothing else

```
DRIFT   : bare_prjob.json, prod_prjob.json, sbnd_pr.json
  ADDED [17].data.entry_rise_guard      = True
  ADDED [17].data.guard_entry_frac      = 1.3
  ADDED [17].data.guard_entry_min_cm    = 5
  ADDED [17].data.guard_entry_max_cm    = 60
  ADDED [17].data.guard_entry_min_len_cm= 70
  ADDED [17].data.guard_entry_kink_deg  = 22
```

`[17]` is `TaggerCheckSTM:pr`. The other **18 of 21** artifacts — uboone, the
six pdhd jobs, the five pdvd jobs, the four other SBND jobs, `prod.standalone`
and `prod.wcls` — are byte-identical to `prod-2026-09-02`.

`prod-2026-09-02` is left byte-untouched (M13); this is a new generation
because the change is **not** byte-identical and is a deliberate physics
change.

## What the guard does

Vetoes an STM accept whose fit carries a **contiguous elevated dQ/dx run
anchored at the boundary end** (`shoulder` between `min_cm` and `max_cm`,
threshold `frac × max(body, 1 MIP)`) **AND** whose path **turns** by at least
`kink_deg` somewhere along the muon. Two particles sharing a stretch near the
boundary, one of which left the detector — the owner's signature, doc 94
§0.3 and §13.

Both conditions are required and neither works alone: 42.3% of the evaluated
STM population clears the 22° kink bar, and the shoulder alone released one of
the owner's hand-adjudicated STMs.

## Validation behind the flip (doc 94 §13)

| | |
|---|---|
| recovers | 827-27-4 — the one of the owner's five that the round-1 `vertex_hadron_guard` could not |
| all 3067 SBND data events | 34,825 bundles identical, **2 flipped**, 0 one-arm-only |
| the two flips | `164466:7` and `350099:15` — the owner adjudicated **both as neutrinos**, so measured contamination is **zero** |
| bundles gaining a cosmic tag | **0**; 414 of 416 STM bundles keep their tag |
| hand-adjudicated bundles | right on **10 of 11**; the one error is a MISS (`707-18-12`, §14), not a false release |
| probe arm, cut disabled | 3067 of 3067 events reproduce the baseline |
| causal negative control | `frac` 1.3 → 5.0 (no elevated run can form) ⇒ 0 fires, every verdict back to baseline |
| production default == validated arm | `TaggerCheckSTM` data block differs in **0 of 34 keys** from `work-mcp1k-r3entry` |
| `wcdoctest-clus` | 237 cases pass |

Statistical limit: zero cosmic admissions observed ⇒ 90% CL upper limit
2.3 events ⇒ **< 0.075%** of events could gain a cosmic.

## Reproducing the previous behaviour

A/B escape, no runner edit needed:

```
PR_EXTRA_TLA=<file containing>  stm_entry_rise_guard=false
```

That returns the chain to `prod-2026-09-02`. Setting `stm_entry_min_cm=1000`
instead makes the guard a **pure probe**: it prints the shoulder, its
no-first-point twin, the excess and the kink on every STM-evaluated bundle and
moves no verdict.

## Still absent, and must stay absent

`descent_guard` — doc 94 §4 measured it dead; nothing anchored on the fit's
`pts[kink]` can work, because that point is often a clustering truncation.

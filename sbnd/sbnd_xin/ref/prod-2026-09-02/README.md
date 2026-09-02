# prod-2026-09-02 — the SBND production operating point, pinned (doc 94)

Supersedes `ref/prod-2026-09-01c` (doc 88), which superseded
`ref/prod-2026-09-01b` (doc 87) and `ref/prod-2026-09-01` (doc 77 round 4).
`prod_cfg_gate.py` with no `--ref` uses the **newest** `ref/prod-*`, so this
one is now the default.  `prod-2026-09-01c` is left byte-untouched.

## What moved, and on whose word

Three keys, all on one component, all on SBND:

```
DRIFT     : bare_prjob.json, prod_prjob.json, sbnd_pr.json
  ADDED   [17].data.vertex_hadron_guard   = True
  ADDED   [17].data.guard_hadron_len_cm   = 12
  ADDED   [17].data.guard_hadron_mip      = 1.5
```

Component `[17]` is `TaggerCheckSTM:pr`.  The other **18 of 21** artifacts —
`uboone.json`, the six `pdhd_*`, the five `pdvd_*`, `sbnd_{clus,img,ql,simcheck}`,
`prod.standalone`, `prod.wcls` — are byte-identical to `prod-2026-09-01c`.
The C++ defaults stay `false / 12 / 1.5`, so no detector but SBND is affected.

Owner, 2026-09-02:

> *"By the way, we can turn this knob on for SBND default running with all
> these validation."*

## What the knob does

`vertex_hadron_guard` vetoes an STM accept whose fitted main carries a prong
longer than 12 cm at more than 1.5 MIP, **with no straightness requirement**.
`check_other_tracks` was ported to find a second *muon*: every acceptance
clause there wants straightness > 0.975/0.99 or MIP-band charge, and
`TaggerCheckSTM.cxx:2909` skips a hot *curved* segment by name.  A proton from
a neutrino vertex is short, hot and curved, so it fell through all of it.

Full write-up: `sbnd_xin/docs/94_stm-neutrino-recovery.md`.
Toolkit commits: `26716eb5` (the guards), plus the flip.

## The validation behind the flip

| | |
|---|---|
| owner-adjudicated neutrinos recovered | **3 of 4** — 966-2-22, 304-6-28, 146-60-31 all flip STM → nu-candidate |
| measured A/B, all **3067** SBND data events | **1 bundle flips of 34,827**; 0 one-arm-only bundles |
| that one bundle, `64475:23` | owner-adjudicated a **neutrino** ⇒ **measured cost zero** |
| owner's 36 hand-adjudicated correct STMs broken | **0** (doc 62, `scan-d59k/stm-baseline.tsv`; join validated 68/72 on length **and** t0 first) |
| causal negative control | length bar raised to 30 cm ⇒ 0 fires, 8/8 verdicts back to baseline |
| `./build/clus/wcdoctest-clus` | 236 cases, all pass |

**Not byte-identical** — this is a deliberate physics change, which is why it
gets its own reference generation.  Set `stm_vertex_hadron_guard=false` (a
`PR_EXTRA_TLA` line) to reproduce the old behaviour for an A/B.

The production default was checked against the arm that validated it: the
compiled `TaggerCheckSTM` data block of `work-mcp1k-d94hadron` (which forced
the knob on via `PR_EXTRA_TLA`) and of this reference differ in **0 of 28
keys**.  The production path is the validated path.

## Still OFF, and must stay OFF

`descent_guard` is present in the tree and absent from this config.  It is the
instrument that measured doc 94 §4, where it is recorded as **dead**: 246 of
401 STM bundles and 16 of 30 resolved owner-CORRECT STMs sit above any usable
cut, because the fitted "stop" is frequently a clustering truncation rather
than a Bragg stop.  Do not flip it on.

## Event manifests

`gate308-{mcp1k,ncpi0,nuecc48}.txt` carry over unchanged from
`prod-2026-09-01c` (241 + 19 + 48 = 308 events).

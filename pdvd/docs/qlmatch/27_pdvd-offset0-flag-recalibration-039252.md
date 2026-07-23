# 27 — Offset-0 flag-window recalibration campaign (run 039252)

Status: **IN PROGRESS** (2026-07-22). Owner-directed follow-up to doc 26:
recover the ~34 net scan-agreed pairs lost when the 13.507 µs pull is removed
(doc 26 step 6, decision point 2), by recalibrating the position-dependent
flag windows and quality gates at the physical (offset-0) frame.

Frame under recalibration (= doc 26 rtp1, all sweeps below inherit it):
`PDVD_QL_EXTRA_OFFSET_US=0`, `PDVD_DRIFT_SPEED_{BOT,TOP}_MMUS=1.4794`,
`PDVD_QL_ANODE_MARGIN_CM=1.0`, `PDVD_QL_CATHODE_EXT1_CM=2.0`,
`PDVD_LIGHT_SUFFIX=_tmerge` (tail-merged light). Truth scoring:
`ql_agree_score.py --truth-time-map work/ql_scores/tm0/time_map.json
--truth-time-shift -13.507 --truth-uid-map-tag tm0k`.

Baselines (identical scoring machinery):
| point | agree | agree% | phantom | missed | missed% |
|---|---|---|---|---|---|
| tm0 (offset 13.507, production) | 752 | 86.6% | 116 | 91 | 10.8% |
| rtp1 (offset 0, doc 26 best) | 718 | 88.2% | 96 | 125 | 14.8% |

Target: close the missed gap (125 → ~91) without phantom inflation; adoption
per the doc-23 rule (a metric improves, none regresses) vs the tm0 row.

## Phase 0 — forensics: where the 59 newly-missed pairs go

Of the 59 rtp1-vs-tm0 newly-missed: **41 mis-picks** (cluster matched a
DIFFERENT flash), **16 no-bundle-at-truth-flash**, **2 unmatched with a good
bundle present**. Truth-bundle frame comparison (29 present in both frames):
ks/χ² essentially UNCHANGED (e.g. 0.33→0.33, 174/10→177/10) while LASSO
`strength` collapses 0.9x→0.00 — the bundle's quality is intact; the PICK
machinery flipped. Dominant discrete flips on the truth bundle:
`-xtpc_pin` ×7, `-consistent` ×6 (several at ks ≈ 0.10 = exactly the
hc_good tier edge), `±at_x_boundary` ×6; 9 crosser-flagged bundles
(at_cathode + xtpc_* in the old frame) are GONE from candidacy entirely.

Mechanism map (from `QLMatching.cxx` / `TimingTPCBundle.cxx` reading):
- `cathode_in = u_cathode + cathode_ext1` is simultaneously the containment
  ceiling, the `at_cathode`/`at_x_boundary` flag ceiling (`:4669`), and the
  xtpc cathode-rescue reference (`:4702-4730`). Real late-charge in-cathode
  tails (doc 16) sit 2 cm deeper at offset 0, crossing it.
- Losing `at_x_boundary` breaks the xtpc candidate admission (`:4123`), so
  the greedy pin (`:4287`) never forms; a pinned cluster otherwise has all
  rivals culled (`cull_inconsistent:2137`) and is strength-cutoff-exempt
  (`:1981`) — pin loss re-opens the LASSO competition and the truth bundle
  gets zeroed with unchanged metrics. The `xtpc_cathode_tol` rescue
  (PDVD_QL_XTPC_CATHODE_TOL_CM=10, ks ≤ 0.32) is the designed re-admission
  path, tuned in the pulled frame.
- `consistent` comes from the highconsist ladder (B1-B4,
  `TimingTPCBundle.cxx:292-311`); B2 `hc_good_ks=0.10` and the B4 miss
  branch (requires a boundary flag) both sit exactly where the frame shift
  lands marginal bundles.

## Phase 1 — single-mechanism diagnostics (this commit)

Four one-knob variants on top of the rtp1 frame, each isolating one
mechanism (18 evts each, tags rc1..rc4):

| tag | change vs rtp1 | probes |
|---|---|---|
| rc1 | `PDVD_QL_XTPC_CATHODE_TOL_CM=14` | xtpc cathode re-admission depth |
| rc2 | `PDVD_QL_CATHODE_EXT1_CM=2.5` | ceiling itself (diagnostic — quantifies the recoverable pool; adoption would prefer mechanism-specific windows per the owner's maintain-or-reduce guidance) |
| rc3 | `PDVD_QL_HC_GOOD_KS=0.12 PDVD_QL_HC_MISS_KS=0.10` | highconsist tier edges |
| rc4 | `PDVD_QL_POSTCULL=0` | post-fit cull share (diagnostic only) |

Results: pending.

# prod-2026-09-25b — the QLXTPC scenario-1 light gate + over-prediction ceiling ON in SBND's standalone Q/L job

## Why this generation exists

> "Flip the light gate on for production" — the owner, 2026-09-25

Doc `sbnd_xin/docs/123_sbnd-ophit-flash-reco-campaign.md` §19, on the measurements of §18 and doc 124.
In `cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet` (the in-tree SBND per-event Q/L job
that `sbnd_xin/wct-clus-matching-perevt.jsonnet` re-exports), two TLA defaults move from `null` (key
omitted) to the measured operating point:

| TLA | before | now | C++ (`QLMatching`, unchanged) |
|---|---|---|---|
| `xtpc_sc1_light_gate` | null | **true** | false |
| `xtpc_sc1_overpred_max` | null | **2.9** | 0 (not tested) |
| `xtpc_sc1_ks_max`, `xtpc_sc1_c2n_max` | null | null (the C++ 0.3 / 50, as measured) | 0.3 / 50 |

A bundle keeps the cross-TPC scenario-1 crosser flags only if its own light fits: KS ≤ 0.3,
χ²/ndf ≤ 50, and predicted light ≤ 2.9 × measured. It also culls the cluster's other bundles only then.
Measured: mcp1k νμ > 0.9 271 → 273, 3000 data events 661 → 664 νμ (FV), 0 lost, MC cv purity 86.8 → 87.0 %.

Previous generation `ref/prod-2026-09-25` is kept. **Exactly one of its 28 artifacts moved,
`sbnd_ql.json`, by exactly the two keys** `xtpc_sc1_light_gate: true` and `xtpc_sc1_overpred_max: 2.9` on the
`QLMatching` node. `sbnd_larsoft_1step.json` did not move: the LArSoft 1-step chain builds its Q/L node from
`qlmatching.jsonnet` directly and is not covered by this flip.

Runner escape: `SBND_XTPC_SC1_GATE=0` (in `run_chain_group.sh` and `run_ql_evt.sh`) passes null for both,
so the keys are omitted and the pre-flip Q/L graph compiles byte for byte.

Against this generation the tree is **PASS 28/28**.

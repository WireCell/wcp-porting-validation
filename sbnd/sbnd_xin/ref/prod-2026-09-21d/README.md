# prod-2026-09-21d — SBND's two chains run one operating point again

## Why this generation exists

> "Yes, I grant you the edit permission. Please commit and push."
> — the owner, 2026-09-21

Doc `sbnd_xin/docs/120_cfg-consolidation-and-two-chain-drift.md` §4. The SBND production PR
operating point moves **into `pr()`'s own defaults** in
`cfg/pgrapher/experiment/sbnd/clus.jsonnet` — 44 named argument defaults plus `tcn_knobs`, which
goes from `{}` to the 219-key production bag — and `sbnd/pr-operating-point.jsonnet` is **deleted**.

Previous generation `ref/prod-2026-09-21c` is kept. **Exactly one of the 26 artifacts moved**:
`sbnd_larsoft_1step.json`. The other 25 are bit-identical, so the local 2-step chain, uBooNE, PDHD,
PDVD, the three runtime fit JSONs and the runner allocator block are all untouched.

## What that one artifact gained, and why it is a restoration

The LArSoft 1-step chain was running a **different PR operating point from the local 2-step chain**,
and had been since 2026-09-14. `two_chain_gate.py` found **25 keys** present locally and absent
under LArSoft:

| class | n | effect |
|---|--:|---|
| Bee particle-flow tree (`pf_*`) | 17 | the PF tree was built without every production PF flip (docs pr/34, 38, 40, 65, 84, 117-123) |
| Bee display | 1 | `pseudo_shower_track_paint` |
| `tracking-pr.root` record | 6 | `fix_cluster_flags`, `provenance`, `rec_charge_provenance`, `nu_provenance` ×3 |
| **physics** | **1** | **`nu_bundle_flash_group`** |

`nu_bundle_flash_group` merges the two neutrino bundles one physical flash makes when their charge
touches (doc 109 rev 4 §9.7, owner-directed, `b93673ef`). C++ default `false`
(`TaggerCheckNeutrino.h:813`), and the key was absent from what LArSoft compiled — so that chain was
**not merging them** while the local chain was.

**The mechanism was omission, not staleness**, and that was measured rather than assumed: the
mirror's 242 knob names intersect *empty* with everything that changed since its pin, and its values
differ from the job's current defaults only in formatting (`2.0` vs `2`, `15 * wc.cm` vs `150`). It
was exact for what it carried and missing what was added after its generator last ran — a generator
(`gen-pr-operating-point.py`, `compile-both.sh`, `resync-operating-point.sh`) that exists in neither
repository. A currency check on the mirror passes on an omission. So did `prod_cfg_gate` PASS 26/26,
every day, throughout.

## Gate evidence

- **V5, the two chains agree** (`docs/120_figs/120_two_chain_final.txt`): `two_chain_gate.py` **PASS**
  — 40 shared components, key for key, with 17 differences forgiven as structural and each naming
  the `clus.jsonnet` line that creates it (shared Bee zip, per-event identity, output directory).
- **V6, the local chain did not move** (`120_v6_local_unmoved.txt`): `compile_prjob_cfg.sh` against
  this tree is sha256-identical to `ref/prod-2026-09-21c/prod_prjob.json`. That is also the
  empirical proof of the semantics the whole design rests on — a caller-supplied `tcn_knobs` bag
  **replaces** the 219-key default rather than merging with it.
- **Blast radius** (`120_gate_drift.txt`): 1 of 26 artifacts moved, and it is the intended one.
- Against this generation the tree is **PASS 26/26** (`120_gate_post.txt`).

## Two defects this round found in passing

- **`iso_endpoint=true` could never have compiled.** The LArSoft entry point passed it to
  `clus_maker.pr()`, which has no such parameter — it is a `tcn_knobs` bag key. It survived because
  it sat in the `preflip` branch, which production never took and jsonnet therefore never evaluated.
  The value now arrives through `pr()`'s bag default (`clus.jsonnet:2107`).
- **The `preflip` A/B arm was never frozen.** It reached its values by *inheriting* `pr()`'s
  defaults, so the doc-118 trajectory flip (09-20) and doc 119's `proj_pad` flip (09-21) had already
  moved it. Reproducing the issue-16 / issue-18 campaigns means checking out the tree at that date.
  The mode is retired; the fcl that sets it still compiles and now gets production.

## Scope

- `wcls-img-clus-matching-xin.jsonnet` is now **in-tree**
  (`cfg/pgrapher/experiment/sbnd/`), with a 1-line re-export left at the work-dir path so every fcl
  and `setup-ap.sh` keeps working. `compile_consumers.sh` step (g) names the in-tree path and no
  longer puts `wcp-porting-img/sbnd` on `WIRECELL_PATH` — nothing in this chain resolves out of the
  working repo any more.
- **PDHD and PDVD are untouched** by §4; their `pr.jsonnet` forks get a comment correction only,
  because their "SBND defaults kept verbatim" note is no longer true of SBND's file.

## Reproduce

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 $SX/scripts/cfg/two_chain_gate.py --verbose          # V5 -- PASS
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21c   # 1 of 26 drifts
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21d   # PASS 26/26
```

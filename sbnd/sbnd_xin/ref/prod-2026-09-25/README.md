# prod-2026-09-25 — SBND's standalone chain matches to flashes rebuilt from the OpHits

## Why this generation exists

> "Let's flip this hits on for SBND production, and keep the cathode rescue on."
> — the owner, 2026-09-25

Doc `sbnd_xin/docs/123_sbnd-ophit-flash-reco-campaign.md` §17, on the campaign of §10–§16.
`sbnd_xin/wct-reco1-dump.jsonnet`'s `flash_source` default moves `'reco1'` → `'hits'`: the reco1
dump — the first stage of `run_chain_group.sh`, the job that decides which flashes the whole
standalone chain matches its charge to — now writes to `opflash_apa<N>.tar.gz` the flashes that
`SBNDReco1OpHitSource` → `SBNDOpFlashFinder` (toolkit `flash/`, code by xning, commit `c2b578fe`)
build from SBND's reco1 PMT OpHits, per TPC, instead of SBND's own `recob::OpFlash`. The Q/L job,
the PR job and their operating points are untouched. The cathode-bundle rescue
(`cathode_rescue`, `cathode_rescue_unmatched` and the round-2/3 extensions) stays **ON** — inert
under hit flashes (3 of 1000 events, §15) and covering the residual.

Previous generation `ref/prod-2026-09-21d` is kept. **None of its 26 artifacts moved.** The dump
job was in none of them — the same shape of hole as the runtime fit JSONs (doc 118, step f) and the
runners' allocator block (doc 119, step h): a production operating point the tripwire could not
see. `compile_consumers.sh` step (i) now compiles it with the runner's exact data and MC TLA lists,
and the two artifacts are kept in full so a drift is named by key:

| artifact | what | 
|---|---|
| `sbnd_dump_data.json` | `wct-reco1-dump.jsonnet` as `run_chain_group.sh` compiles it for a data file (`caf_offset_mode=product`, group of 16) |
| `sbnd_dump_mc.json` | the same with `--mc` (DetSim product names, `caf_offset_mode=none`, group of 1000) |

Against this generation the tree is **PASS 28/28**.

## What the flip rests on (doc 123 §17 has the labels)

- **Proof A** — the pre-flip file compiled with the measured arms' TLA (`flash_source=hits`) vs the
  flipped file compiled bare: **0 lines**, for the data, `--fsproduct` and `--mc` TLA lists. The
  production graph is exactly what every `work-*-d123hits` arm ran.
- **Proof B** — the pre-flip file compiled bare vs the flipped file with `flash_source=reco1`:
  **0 lines**, same three lists. `SBND_FLASH_SOURCE=reco1` in `run_chain_group.sh` is the pre-flip
  graph byte for byte (the control arm; the off path is not orphaned).
- **Output gate** — the flipped runner on the production libraries (no pin) on nueCC-48:
  `work-nuecc48-d123flip` vs the campaign's `work-nuecc48-d123hits`, and `work-nuecc48-d123flipoff`
  (`SBND_FLASH_SOURCE=reco1`) vs `work-nuecc48-d123base`, member by member through stage A and stage
  B (`scripts/d123/flip_gate.py`); one MC file (`work-r3cv-d123flip/f000`) the same way. Results in §17.
- **The measurement** — data 3000 mcp events: νμ > 0.9 candidates 784 → 784 (23 lost, 23 gained),
  26 % of matched clusters change flash, 63 % of those to flashes reco1 never had; MC inclusive BNB:
  νμCC efficiency **70.0 → 71.8 %** at purity 86.7 → 86.8 %, vertex-matched candidates 82.0 → 84.7 %;
  MC exclusive νe: νeCC 41.7 → 41.8 %, purity 97.1 → 96.9 %; beam-off fake νμ per gate 0.9 → 1.3 %.

## What this generation does NOT change

- The **LArSoft 1-step chain** (`sbnd/wcls-img-clus-matching-xin.jsonnet`, artifact
  `sbnd_larsoft_1step.json`) still takes `recob::OpFlash` through `wclsOpFlashSource`: there is no
  `wclsOpHitSource` in larwirecell yet (doc 123 §16 is its design). Until it exists, the two chains
  run **different flashes** on purpose. `two_chain_gate.py` compares PR components and does not see
  this; doc 120's two-chain agreement is on the PR operating point, not on the light.
- The Q/L job, the PR job, the rescue knobs, the fit JSONs, the allocator block: bit-identical to
  `prod-2026-09-21d`.

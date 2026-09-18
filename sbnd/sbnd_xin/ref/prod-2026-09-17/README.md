# ref/prod-2026-09-17 — SBND production operating point

**Why this generation exists.** Doc 109 rev 3 turned `root_point_ids` ON in the SBND
production PR job (owner's word, 2026-09-17; toolkit `12798c4f`), so `tracking-pr.root`'s
`T_rec_charge` joins to the neutrino candidate that owns its points. The previous
generation, `ref/prod-2026-09-14`, still matched production exactly at the **unmodified**
HEAD `d2777286` — `prod_cfg_gate.py --ref ref/prod-2026-09-14` was **PASS 21/21** there — so
unlike the 09-14 generation this one has **no inherited drift to attribute**. The flip moves
one artifact by one key.

Toolkit commit: `12798c4f`. Gate:
`scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17` → **PASS, 21/21 artifacts**
(cfg tree = the working tree at `12798c4f`).

`prod_prjob.json` is committed with `git add -f` (`*.json` is gitignored).

## The drift from `prod-2026-09-14` — 1 of 21 artifacts

```
DRIFT     : prod_prjob.json

SBND PR job, key by key (reference -> current tree):
  ADDED   [24].data.rec_charge_provenance = True
```

Component `[24]` is `SbndPrMagnifyTrackingVisitor`. That is the whole diff: no other
artifact, no other key. The other 20 consumers (bare PR job, lar 1-step imaging+clustering,
standalone Q/L, PDHD, PDVD, sim checks, uBooNE) are byte-identical to pristine `d2777286`.

## What the key does

`rec_charge_provenance` is observation-only — it changes what `tracking-pr.root` records and
nothing the reconstruction produces:

- `T_rec_charge.cluster_id` comes from the candidate's own `TaggerInfo::cluster_id` (what
  `T_tagger`/`T_kine` carry) instead of a `Flags::main_cluster` scan of the candidate's PR
  graph, which returned `-1` on a demoted-main candidate and the pre-swap cluster on a
  vertex-moved row;
- `nu_index` and `point_cluster_id` are added to `T_rec_charge`;
- `T_rec_charge` and `T_proj_data` are booked even when empty, so the file's tree set no
  longer varies with the event.

Gate evidence: doc 109 sec 8.8 — knob-off vs HEAD PASS on 267 events (233 055 ROOT branches
identical), knob-on leaves `mabc-pr.zip`, the pctree, nusel and the calib dump identical, and
content checks C1–C17 have 0 failures.

**`nu_dedup_flash_group` is NOT in this generation.** It is built, measured (doc 109 sec
8.7.1) and left `false` everywhere, because it removes `T_tagger` rows.

## Reproduce

```bash
cd wcp-porting-img/sbnd/sbnd_xin
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17          # PASS 21/21
# the previous generation at the pre-flip tree:
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-14 --cfg <git archive d2777286 cfg>
```

**`ref/prod-2026-09-14` is kept.** Nothing in this round makes it stale as a record, and
removing a generation is its own decision.

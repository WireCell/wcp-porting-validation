# ref/prod-2026-09-17b — SBND production operating point

**Why this generation exists.** Doc 109 rev 4 turned `nu_bundle_flash_group` ON in the SBND
production PR job (owner's word, 2026-09-17: "if things pass, please turn this knob on";
toolkit `b93673ef` on top of the knob commit `d1caf178`), so one physical beam flash seen by
both drift volumes yields **one** neutrino candidate when the two volumes' bundles touch, the
other side's clusters joining the PR pass as companions. The previous generation,
`ref/prod-2026-09-17`, matched production exactly at the unmodified HEAD `12798c4f`, so this
one has **no inherited drift to attribute**: the flip moves one artifact by one key.

Toolkit commit: `b93673ef`. Gate:
`scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b` → **PASS, 21/21 artifacts**
(cfg tree = the working tree at `b93673ef`).

`prod_prjob.json` is committed with `git add -f` (`*.json` is gitignored).

## The drift from `prod-2026-09-17` — 1 of 21 artifacts

```
DRIFT     : prod_prjob.json

SBND PR job, key by key (reference -> current tree):
  ADDED   [21].data.nu_bundle_flash_group = True
```

Component `[21]` is `TaggerCheckNeutrino`. That is the whole diff: no other artifact, no
other key. The other 20 consumers (bare PR job, lar 1-step imaging+clustering, standalone
Q/L, PDHD, PDVD, sim checks, uBooNE) are byte-identical to pristine `12798c4f`.

## What the key does

Two in-window neutrino bundles that share a flash group (one physical flash, two opflash
gids, one per TPC) and lie on different TPCs are merged into one candidate when the closest
points of some cluster pair, one from each bundle, are within `nu_bundle_flash_group_gap`
(20 cm). The merged candidate is the longer of the two bundles' own selected activities; the
other side's associated clusters and mains join as companions (`skip_cosmic_companions`
applies). A merge needs a winner at or above `nu_per_bundle_min_length`; bundles that share
the light but do not touch stay two candidates.

Gate evidence: doc 109 sec 9.5–9.6 — knob-off vs HEAD PASS on 267 events (236 251 ROOT
branches identical), 253 non-eligible manifest events bit-identical with the knob on apart
from `Trun.op_config_sha256` (216 749 branches), all 3 067 events rc 0 with C1–C17 clean,
116 merges on the 142 eligible events, every removed row a vertex-less placeholder, no
surviving row lost its vertex.

**`nu_dedup_flash_group` stays OFF** (superseded by the merge; a WARN fires if both are on).

## Reproduce

```bash
cd wcp-porting-img/sbnd/sbnd_xin
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17b          # PASS 21/21
# the previous generation at the pre-flip tree:
scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-17 --cfg <git archive 12798c4f cfg>
```

**`ref/prod-2026-09-17` is kept.** Nothing in this round makes it stale as a record, and
removing a generation is its own decision.

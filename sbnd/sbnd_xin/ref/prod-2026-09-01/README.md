# prod-2026-09-01 — the SBND production operating point, pinned

Created by **doc 77 §13.4** (round 4, 2026-09-01) at toolkit `7542532f`
on `apply-pointcloud`.

## What this is, and the gap it closes

The SBND PR job carries **484 top-level arguments**, of which **315 hold a
production value**. Before this reference, no test guarded a single one of
them:

- `clus/test/doctest_clus_knob_defaults.cxx` pins the **C++ defaults** and says
  so in its own header — *"A green run here does NOT mean production is on the
  legacy path."*
- The compiled-config proof (`compile_consumers.sh` + `cmp_consumers.sh`)
  compares two freshly compiled trees, so it catches a change made **during** a
  round and nothing between rounds.

Neither answers the question a production tree has to answer: *is the operating
point today still the one that was validated?* doc pr/127 is what the gap
costs — a shipped fix died silently for ten days.

## Contents

| file | role |
|---|---|
| `consumers.sha256` | 21 compiled artifacts, one line each — **detects** any drift (SBND pr/img/clus/QL, PDHD ×6, PDVD ×5, uBooNE MABC, the bare PR job, the standalone/wcls jobs) |
| `prod_prjob.json` | the SBND PR job compiled at the production operating point, in full (266,930 B) — so a drift can be **named**, key by key, not merely detected |

## Use

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
scripts/cfg/prod_cfg_gate.py                    # rc 0 = matches; rc 1 = drift
scripts/cfg/prod_cfg_gate.py --cfg <other/cfg>  # check another tree
```

On drift the gate prints the differing artifacts and, for the PR job, the
changed keys:

```
DRIFT     : prod_prjob.json
SBND PR job, key by key (reference -> current tree):
  CHANGED [21].data.pi0_attached_partner_min_mev : 29 -> 28
```

## Deliberately not automatic

This is a **tripwire the owner fires on purpose**, not a build step. It is
wired into no build, runner, or hook: a stale reference breaking someone's
build is worse than no reference at all.

## Refreshing it — on an intended flip only

```bash
scripts/cfg/prod_cfg_gate.py --refresh
```

Then, in the same commit, record in the round doc **which knob moved and on
whose word**. A refresh with no such record is indistinguishable from the drift
this reference exists to catch.

Start a **new** `ref/prod-<date>/` rather than refreshing this one when the
operating point changes substantially — these are records (M13), and
`prod_cfg_gate.py` with no `--ref` uses the newest.

## Provenance

Both controls were run when this reference was created (doc 77 §13.4):

- positive — the tree as shipped: `PASS`, rc 0, 21 artifacts;
- negative — a **copy** of the cfg tree with `pi0_attached_partner_min_mev`
  29 → 28: rc 1, and the key named.

A gate that cannot fail proves nothing.

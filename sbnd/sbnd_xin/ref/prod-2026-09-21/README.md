# prod-2026-09-21 — `proj_pad_wire` / `proj_pad_time` enter the SBND production fit

## Why this generation exists

> "you can flip round 2 ones, since the output did not change."
> — the owner, 2026-09-21

Doc `sbnd_xin/docs/119_pr-profiling-rounds.md` §6; toolkit commit for the flip is the one that adds
`proj_pad_wire: 3` and `proj_pad_time: 3` to
`cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json`. Previous generation `ref/prod-2026-09-20` is
kept and still describes the pre-flip operating point (it correctly reports
`DRIFT: sbnd_track_fitting.json` against the current tree).

## Read this before quoting the flip

**On PDHD this knob bought −28.8 % core-s and −56 % peak RSS. On SBND it buys neither.** Doc 119 §6
measured it on the 62-event gate manifest and **recommended against flipping**; the owner flipped it
anyway, and the reason recorded is **cross-detector consistency**, not a saving:

| | PDHD (doc 30 round 3) | SBND (doc 119 round 2) |
|---|---|---|
| proj cells kept | 2.4 – 2.9 % | **87.9 %** |
| core-s / arm TICK | −28.8 % | +2.2 / +1.3 / +1.5 % — **inside** the null-pair floor (+0.2 / −1.6 / −2.6 %) |
| worst-event peak RSS | 5.00 → 2.19 GB (−56 %) | 1.428 → 1.424 GiB (−0.3 %) |

SBND's fitted-charge map holds 7 968 cells per event; PDHD's, *with the knob already on*, holds
9 153. SBND without the knob was already tighter than PDHD with it, so there is nothing for the
filter to remove. **Do not cite this key for a CPU or memory benefit on SBND — there is none.**

**And the flip is not free.** The physics products do not move, but `T_proj_data` and the calib
dump's `proj` block lose **12.1 %** of their cells (7 968 → 7 002 per event). That is the 2-D scan
display, and it is a real loss.

## Gate evidence

- `prod_cfg_gate.py --ref ref/prod-2026-09-20` was **PASS 25/25 at unmodified HEAD** before the
  first edit of the round (`docs/119_figs/119_gate_pre.txt`), so this drift has no inherited
  component. Against this generation the tree is **PASS 25/25**
  (`docs/119_figs/119_gate_post.txt`).
- **Output gate, 62 events** (`docs/119_figs/119_gate_pad.txt`): `T_rec_charge`, every tagger tree,
  `mabc-pr.zip`, the pctree, `nusel` and the non-`proj` calib dump are **all identical**; only
  `T_proj_data`, the `proj` block and the fit-JSON provenance move. The same gate on a **null pair**
  (one configuration run twice) returns 10/24/1 where this returns 28/48/21 — sensitive as well as
  specific (`119_gate_nullpair.txt`).
- **G2, fit-JSON key identity** (`119_gate_tfkeys.txt`): after stripping `_`-prefixed comment keys,
  which `load_trackfitting_config` skips, the production file and the measured file carry **all 49
  live keys identical in name and value**.
- **G1, the flipped default reproduces the measured arm** (`119_gate_flip.txt`): a fresh 62-event
  arm run with **no `SBND_TRACKFIT_JSON` at all**, gated against the measured arm at
  provenance-only allowance — **0 differences outside provenance**, and a direct tree comparison
  confirms `T_proj_data` itself is identical (only `Trun` differs). This is what carries the
  measurement onto production: the measured arm read an override path, production reads the
  in-tree file.

## The drift from prod-2026-09-20 — 1 of 25 artifacts

```
DRIFT     : sbnd_track_fitting.json
```

`sbnd_track_fitting.json` is consumer artifact 22, added to the set by doc 118 precisely because
this family of change never enters a compiled config — it is read at **runtime** by
`load_trackfitting_config` with a plain `ifstream`. The tripwire catching this flip is that hole
working as designed.

**Every other artifact is byte-identical**, including uBooNE (a frozen reference), PDHD and PDVD:
`TrackFitting.cxx` is shared, and each detector's own `*_track_fitting.json` is what selects the
behaviour. No jsonnet changed in this round, so the other 24 hashes are unchanged from
`prod-2026-09-20`.

## What the keys do

```json
"proj_pad_wire": 3,
"proj_pad_time": 3
```

C++ default `-1` = OFF. When `proj_pad_wire >= 0`, `fill_fitted_charge_2d` stores a cell only if it
lies within `proj_pad_wire` wires **and** `proj_pad_time` **slices** (not ticks) of a cell whose raw
`R*pos_3D` prediction is nonzero, in the same (apa, face, plane). The seed is the raw prediction
rather than `pred_charge`, so a cell the fit predicts on a dead or below-threshold channel is not
dropped. **Removing these two keys restores the pre-flip display exactly.**

Read at runtime by `TaggerCheckSTM` and `TaggerCheckNeutrino`. Every reader of the trimmed map is a
dump writer — `PrDisplayDump.cxx:1173` states the blast radius is diagnostic-only; no tagger
verdict, Bee layer or pctree tensor depends on it.

## Reproduce

```bash
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
python3 $SX/scripts/cfg/prod_cfg_gate.py --ref ref/prod-2026-09-21     # PASS 25/25
python3 $SX/scripts/d119/tf_key_gate.py                                # G2, 49 live keys
python3 $SX/scripts/d119/lever_gate.py --vs pad flip                   # G1
python3 $SX/scripts/d119/lever_gate.py pad                             # the 62-event output gate
python3 $SX/scripts/d119/lever_cost.py                                 # incl. the null-pair floor
```

To re-measure the trade if SBND's fitted-charge map ever grows (a larger readout window, a wider
fit, a denser sampler), `docs/119_figs/119_tf_sbnd_pad.json` plus
`scripts/d119/stageB_lever.sh LEVER=pad` re-runs the whole thing in about three minutes; the number
to watch is the keep fraction, 87.9 % today.

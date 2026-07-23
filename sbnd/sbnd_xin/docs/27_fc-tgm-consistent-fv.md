# 27 — FC now uses the same fiducial volume as TGM

Resolves the open question of doc 25 §5. `TaggerCheckFC` gains `fiducial` +
`fv_tolerance` config and SBND passes it the **same** `BoxFiducial:sbnd_pr_fv`
and `[-2,-2,-2.5,-2.5,-3,-3] cm` margins that `TaggerCheckTGM` already uses, so
"contained" means one thing across both verdicts.

**Result: the TGM∧FC contradiction is gone — 11 → 0.** FC counts move
27 / 47 → **3 / 71**, and an independent geometric cross-check shows the new
verdicts are right and the old ones were not (§3).

> This changes the `fc` column only. `tgm`, `stm`, `label`, the row set and the
> per-event summary are **unchanged field-for-field** (§4). Knob-off compiled
> config is byte-identical, and `TaggerCheckSTM` / `TaggerCheckNeutrino` are
> untouched.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild
cd sbnd_xin
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all
```

## 1. The problem (doc 25 §5 recap)

FC and TGM tested containment against different objects, so they were not
mutually exclusive — 11 TGM-tagged bundles read FC=1, including 477 cm and
462 cm tracks:

| | volume | margin |
|---|---|---|
| `TaggerCheckTGM` | `BoxFiducial:sbnd_pr_fv`, one box across **both** TPCs | `fv_tolerance` = −2 / −2.5 / −3 cm insets |
| `cluster_fc_check` (FC) | `FiducialUtils` → `DetectorVolumes`, union of per-face sensitive volumes | **none**, and holed at the CPA slab (\|x\| < 0.45 cm) |

FC's volume was more permissive at every wall (no inset) *and* holed at the
cathode.

## 2. The change

`cluster_fc_check` takes two new **optional** parameters
(`clus/src/Clustering_Util.cxx:75`):

```cpp
FCCheckResult cluster_fc_check(Cluster& cluster, IDetectorVolumes::pointer dv,
                               IFiducial::pointer fiducial = nullptr,
                               const std::vector<double>& fv_tolerance = {});
```

- `fiducial == nullptr` (the default) → the direct containment tests call
  `fiducial_utils->inside_fiducial_volume(p)` exactly as before. **Every
  existing caller keeps its behavior bit-for-bit**: `TaggerCheckSTM:1890` and
  `TaggerCheckNeutrino:519` still pass two arguments.
- `fiducial != nullptr` → the same three direct tests run against that
  `IFiducial` with `fv_tolerance` applied, using the identical inset logic as
  `TaggerCheckTGM::inside_fv` (negative tolerance ⇒ the point shifted *outward*
  by \|tol\| must still be contained).

Only the **three direct** FV tests are redirected (`Clustering_Util.cxx:146`,
`:238`, `:239`). The dead-region and signal-processing checks keep using
`FiducialUtils`' per-(apa,face) logic — exactly the split `TaggerCheckTGM`
already documents.

`TaggerCheckFC` grows `fiducial` / `fv_tolerance` keys, both absent by default.
One detail worth noting: `NeedFiducial::configure` is invoked **only** when the
key is present, because it falls back to looking up the type-only name
`"DetectorVolumes"`, which is not an instantiated component in the SBND PR
config (it is `dv-apa0-1`) and would throw.

SBND wires FC to the same objects as TGM
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet`), and `sbnd_pr_fv` is added to
`tagger_uses` when `tagger_check_fc` is in the pipeline.

## 3. Result — the verdicts are now geometrically sound

| | before (FiducialUtils FV) | after (shared TGM FV) |
|---|---:|---:|
| FC | 27 | **3** |
| not-FC | 47 | **71** |
| **TGM ∧ FC=1** | **11** | **0** |
| missing verdicts | 0 | 0 |

The three surviving FC bundles:

| evt | main | len_cm | in_beam | label |
|---|---:|---:|---:|---|
| 286065 | 1 | 114.3 | 0 | not-tagged |
| 286065 | 4 | 32.4 | 0 | not-tagged |
| 286241 | 11 | 303.6 | 1 | nu-candidate |

### 3.1 Independent cross-check

Rather than trust the tagger, the per-bundle 3-D point cloud was read straight
from the pctree (`x_t0cor/y_cor/z_cor`) and its extent compared against the
inset FV `x ∈ [−199.05, 199.05]`, `y ∈ [−196.812, 196.812]`,
`z ∈ [3.85, 497.15]` cm. A cluster with any point outside that box cannot be
fully contained, whatever the tagger says.

- **New table: 74 / 74 bundles consistent. Zero cases of FC=1 with points
  outside the FV.**
- **Old table: of the 27 bundles that read FC=1, 24 had points outside** — i.e.
  ~89% of the old FC=1 verdicts were geometrically impossible under TGM's
  volume. The 3 that survive are exactly the 3 that are genuinely contained.

That is the independent confirmation that this change fixes real errors rather
than merely making two numbers agree.

### 3.2 In-beam population

| evt | main | len_cm | tgm | stm | fc | label |
|---|---:|---:|---:|---:|---:|---|
| 284349 | 11 | 202.1 | 1 | 0 | 0 | TGM |
| 284657 | 5 | 96.9 | 0 | 0 | 0 | nu-candidate |
| 285185 | 21 | 141.9 | 0 | 0 | 0 | nu-candidate |
| 285999 | 22 | 38.5 | 0 | 0 | 0 | nu-candidate |
| 286065 | 3 | 237.8 | 0 | 0 | 0 | nu-candidate |
| 286241 | 11 | 303.6 | 0 | 0 | **1** | nu-candidate |
| 286329 | 14 | 113.0 | 0 | 0 | 0 | nu-candidate |
| 286527 | 16 | 119.0 | 0 | 0 | 0 | nu-candidate |

1 of 7 nu-candidates is fully contained, down from 4 of 7. Given SBND is a
surface detector on a cosmic-dominated sample, and the FV now insets every wall
by 2–3 cm, a low FC fraction is the expected outcome.

## 4. Verification

**G0 — freshness (M1).** `libWireCellClus.so` mtime `1784824212` >
`TaggerCheckFC.cxx` `1784824150` > `Clustering_Util.cxx` `1784824073`.
`wcdoctest-clus`: 41/41 cases, 518/518 assertions, rc=0.

**G1 — inert when off (M6).** `wcsonnet` on `wct-pr-perevt.jsonnet` without
`tagger_check_fc`, before vs after this change: **byte-identical**. With it on,
the two taggers now report the same fiducial:

```
TaggerCheckTGM -> {"fiducial": "BoxFiducial:sbnd_pr_fv", "fv_tolerance": [-20,-20,-25,-25,-30,-30]}
TaggerCheckFC  -> {"fiducial": "BoxFiducial:sbnd_pr_fv", "fv_tolerance": [-20,-20,-25,-25,-30,-30]}
```

**G2 — only `fc` moved.** Row set identical (76 rows); **every non-`fc`
column** (`tgm`, `stm`, `main_id`, `n_bundle`, `npts_*`, `len_main_cm`,
`n_frag`, `label`, …) is unchanged, and `nusel-events.tsv` is identical. This
also demonstrates `TaggerCheckSTM` was unaffected by the `cluster_fc_check`
signature change, as the default argument requires.

**G3 — no missing verdicts.** `tgm`, `stm`, `fc` all have 0 rows reading −1.

## 5. Two extractor bugs surfaced on the way

Both are log-parsing robustness, not physics.

### 5.1 Log tearing is DETERMINISTIC — re-running does not fix it

evt285999 crashed `nusel_extract.py` with `KeyError: 'fal'`. Its FC line is torn
mid-word:

```
TaggerCheckFC: cluster 20 → FC=fal[09:33:15.265] D [  clus  ] <Mul…
```

Doc 23/25 treated this as an interleaving race. It is not: re-running the event
reproduced the tear **at the identical byte** (`FC=fal[09:35:14.250] D [ cl…`).
It is a write-buffer boundary, deterministic given deterministic log content, so
"re-run for a clean log" is not a recovery strategy.

Fix: the verdict is captured as its **first character** (`VERDICT =
r'([tf01])\w*'`). The only possible values are `true`/`false`/`1`/`0`, so the
leading character disambiguates all four and a fragment like `fal` is read
losslessly. A tear landing before any value character still fails to match — the
correct outcome, leaving −1 plus the existing "no tagger verdict" warning
instead of a crash.

### 5.2 The two taggers spell the verdict differently

`TaggerCheckTGM` / `TaggerCheckFC` log a plain bool (`TGM=false`), while
`TaggerCheckSTM` logs the flag getter as an int (`STM=0 TGM=0`). The original
`as_int` map quietly accepted both; an intermediate strict-`(true|false)` regex
silently dropped **all 40** STM verdicts to −1. Caught by the G2 gate, which is
why that gate compares every column rather than eyeballing the new one.

## 6. Files changed

| repo | file | change |
|---|---|---|
| toolkit | `clus/inc/WireCellClus/ClusteringFuncs.h` | `cluster_fc_check` gains 2 defaulted params |
| toolkit | `clus/src/Clustering_Util.cxx` | `inside_fv` lambda; 3 direct FV call sites redirected |
| toolkit | `clus/src/TaggerCheckFC.cxx` | `NeedFiducial` + `fv_tolerance`; conditional `NeedFiducial::configure` |
| toolkit | `cfg/pgrapher/common/clus.jsonnet` | `tagger_check_fc(fiducial, fv_tolerance)` |
| toolkit | `cfg/pgrapher/experiment/sbnd/clus.jsonnet` | pass `sbnd_pr_fv` + margins; `tagger_uses` |
| wcp | `sbnd_xin/nusel_extract.py` | `VERDICT` first-char capture; both spellings |
| wcp | `sbnd_xin/work-mcp10/` | retabled |

Doc 25 §5's open question is now closed; its option 3 (changing
`cluster_fc_check` for **all** callers, which would move STM and `match_isFC`)
was **not** taken — the default-null parameter leaves those two untouched.

# 25 — An FC (fully-contained) flag alongside TGM/STM, and the 10-event retable

> **Superseded in part by doc 26.** The label counts in §2 (TGM 33 / STM 1) were
> taken with `check_neutrino_candidate` OFF, which is no longer this chain's
> default; the current counts are TGM 34 / STM 0. The FC counts (27/47), the
> in-beam FC column, all gates in §3 and the open question in §5 are unaffected.

**What was added.** A `TaggerCheckFC` visitor that records the fully-contained
verdict as the cluster flag **`FC`** — the tagger-computed sibling of `TGM` and
`STM`. This closes the FC half of the gap found in doc 24: the verdict was
already being computed for every bundle the STM tagger touched and then thrown
away. It now reaches `work-mcp10/nusel-table.tsv` as an `fc` column.

**Status.**
- Knob-off compiled config **byte-identical**; TGM/STM verdicts on all 10
  events **byte-identical** with FC on (§3). Gates below.
- `work-mcp10` retabled: 74 bundles, **27 FC / 47 not-FC**, every bundle
  carries a verdict (no `-1`).
- ⚠ **Open physics question, not tuned (§5):** FC and TGM are evaluated
  against *different* fiducial volumes, so they are not mutually exclusive.
  11 TGM bundles read FC=1, including 476 cm and 462 cm tracks. Needs an
  owner decision.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && wcbuild        # then M1 freshness proof
cd sbnd_xin
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all
# -> work-mcp10/nusel-table.tsv (per bundle, with the fc column)
#    work-mcp10/nusel-events.tsv (per event)
```

## 1. What was built

| piece | file | note |
|---|---|---|
| flag name | `clus/inc/WireCellClus/ClusteringFuncs.h` — `Flags::FC = "FC"` | next to `Flags::TGM` / `Flags::STM`. Deliberately **not** the lowercase `fully_contained`, which stays reserved for uBooNE verdicts imported via `ClusteringTaggerFlagTransfer` (doc 24 §3) |
| component | `clus/src/TaggerCheckFC.cxx` (new) | iterates main clusters, calls `Facade::cluster_fc_check`, sets `Flags::FC`, logs at INFO |
| builder | `cfg/pgrapher/common/clus.jsonnet` — `tagger_check_fc()` | new function; no existing builder touched |
| SBND pipeline entry | `cfg/pgrapher/experiment/sbnd/clus.jsonnet` — `cm_by_name.tagger_check_fc` | `require_in_scope=true`, matching TGM/STM |
| runner | `sbnd_xin/run_nusel_evt.sh` | `PIPELINE=…,tagger_check_stm,tagger_check_fc` |
| table | `sbnd_xin/nusel_extract.py` | new `fc` column + `RE_FC` |

**The FC algorithm is not re-implemented.** `TaggerCheckFC` is a thin recorder
around the existing `Facade::cluster_fc_check`
(`clus/src/Clustering_Util.cxx:75`) — the same function `TaggerCheckSTM`
(`:1890`) and `TaggerCheckNeutrino` (`:519`) already act on. So the new flag
means exactly what STM already believes about containment, with no new physics
introduced. Consistency check: the one STM-tagged bundle reads FC=0, as it must
(STM returns early on FC clusters).

### Two design choices worth stating

**A separate visitor rather than promoting STM's internal result.**
`TaggerCheckSTM` skips mains already flagged TGM (`:144`), so its FC
computation never runs for through-going bundles — surfacing it from there
would leave a third of the population unlabelled. Cost: non-TGM mains have
`cluster_fc_check` run twice per event.

**FC runs LAST in the pipeline.** `cluster_fc_check` takes `Cluster&` and
lazily populates PCA / hough / steiner-boundary caches. Position does not
affect FC's own coverage (it evaluates every in-scope main regardless of
TGM/STM flags), so running it last leaves TGM's and STM's inputs pristine
rather than relying on those caches being order-independent (M4).

**`fc` does not feed `label`.** In the prototype FC is an independent
eval-tree variable feeding the BDTs, not a cosmic veto
(`prod-wire-cell-matching-nusel-port.cxx:1002-1007`). The `label` column keeps
its doc-23 semantics exactly.

## 2. Results — MCP2025C reco1, 10 events (run 18255), `work-mcp10`

74 qualifying bundles + 2 no-bundle rows. **Every bundle got a verdict** (`fc`
is never `-1`).

| label | n | FC | not-FC |
|---|---:|---:|---:|
| TGM | 33 | 11 | 22 |
| STM | 1 | 0 | 1 |
| nu-candidate | 7 | 4 | 3 |
| not-tagged | 33 | 12 | 21 |
| **TOTAL** | **74** | **27** | **47** |

The in-beam-window population — the selection-relevant one:

| evt | main | len_cm | npts | tgm | stm | fc | label |
|---|---:|---:|---:|---:|---:|---:|---|
| 284349 | 11 | 202.1 | 2176 | 0 | 1 | 0 | STM |
| 284657 | 5 | 96.9 | 1568 | 0 | 0 | **1** | nu-candidate |
| 285185 | 21 | 141.9 | 1211 | 0 | 0 | 0 | nu-candidate |
| 285999 | 22 | 38.5 | 279 | 0 | 0 | **1** | nu-candidate |
| 286065 | 3 | 237.8 | 4407 | 0 | 0 | **1** | nu-candidate |
| 286241 | 11 | 303.6 | 4383 | 0 | 0 | **1** | nu-candidate |
| 286329 | 14 | 113.0 | 573 | 0 | 0 | 0 | nu-candidate |
| 286527 | 16 | 119.0 | 392 | 0 | 0 | 0 | nu-candidate |

4 of the 7 nu-candidates are fully contained. The per-event summary
(`nusel-events.tsv`) is unchanged from doc 23 — FC adds no event-level verdict.

## 3. Verification

Three gates, all PASS. Labels are the scratch snapshots named below.

**G0 — freshness (M1).** `local/lib/libWireCellClus.so` mtime `1784822697` >
`TaggerCheckFC.cxx` `1784822626` > `ClusteringFuncs.h` `1784822599`;
`nm -DC` finds 45 `TaggerCheckFC` symbols. `./build/clus/wcdoctest-clus`:
41/41 cases, 518/518 assertions, rc=0.

**G1 — inert when off (compiled-config proof, M6).** `wcsonnet` on
`wct-pr-perevt.jsonnet` with the doc-23 pipeline, before vs after the cfg
change: `md5 67f8326a79d392dede4601484470a1e4` both →
**byte-identical**. With `tagger_check_fc` appended, `TaggerCheckFC` appears
(2 hits) and the MABC pipeline reads
`[…, TaggerCheckTGM:pr, TaggerCheckSTM:pr, TaggerCheckFC:pr]`.

**G2 — non-perturbing when on.** Full 10-event run on the *fresh* binary with
FC off (`snap: fcoff-fresh/`), then with FC on. Dropping the new `fc` column
from the FC-on table gives a file **byte-identical** to the FC-off table;
`nusel-events.tsv` likewise. So no `tgm`, `stm`, `main_id`, `n_bundle`,
`len_main_cm` or in-scope-set value moved.

**G3 — no binary drift from the interim commit.** The FC-off run on the fresh
binary is byte-identical to the pre-existing doc-23-era `nusel-table.tsv`
(`snap: baseline-pre-fc/`), confirming `f1780d50`
(check_neutrino_candidate into TGM) is indeed inert with its knob off, and that
G2's baseline is the same table doc 23 reports.

### Incidental fix: a torn log line

G2 initially showed one spurious diff — evt285185 cluster 5 read `stm=-1`
instead of `0`. Cause: the log line was torn mid-word by an interleaved write,

```
… TaggerCheckSTM: cluster 5 already TGM; ski[09:08:06.159] D [  clus  ] <MultiAlgBlobClustering…
```

the failure mode doc 23 already records (seen there on evt286329). FC did not
cause it — it only changed the interleaving timing. `RE_SKIP` in
`nusel_extract.py` now matches the message **prefix** (`already TGM`) instead of
requiring `already TGM; skipping`: this message carries its full meaning
(`tgm=1, stm=0`) before the tear point, so a prefix match recovers it
losslessly. The verdict regexes cannot be hardened the same way — their
information is the trailing `=true`/`=false`, which a tear destroys.

## 4. Cross-check against the prototype

| prototype | toolkit, as of this change |
|---|---|
| `event_type` bit 2, set from `check_fully_contained` (`ToyFiducial.cxx:816`) | `Flags::FC` set from `Facade::cluster_fc_check` |
| `match_isFC` eval-tree variable / BDT input | `tagger_info.match_isFC` (`TaggerCheckNeutrino.cxx:519`) — unchanged, still only on the `tagger_check_neutrino` path |
| `_fc_breakdown` FV/SP/DC failure mask → `match_notFC_FV/_SP/_DC` | **still unported** — `FCCheckResult` carries no failure reason |
| retry against the original pre-merge main (`flag=2`) | **not ported** |
| endpoints from `get_extreme_wcps()`, `offset_x` from flash time | steiner boundary, `offset_x=0` (points already T0-corrected by `switch_scope`) |

So `FC` is *an* FC definition — the one STM and TaggerCheckNeutrino already
act on — and is **not bit-comparable to WCP's**.

## 5. Open: FC and TGM use different fiducial volumes

**Reported, not tuned** (CLAUDE.md §5.7).

11 of the 33 TGM-tagged bundles read FC=1. Physically that is a contradiction —
a through-going muon exits the detector — and the lengths make it stark:

| evt | main | len_cm | npts |
|---|---:|---:|---:|
| 285185 | 15 | **476.8** | 4822 |
| 284657 | 23 | **461.6** | 5877 |
| 285185 | 1 | 425.7 | 15729 |
| 284657 | 25 | 287.2 | 3639 |
| 285999 | 19 | 276.3 | 4924 |
| 286021 | 12 | 213.9 | 2231 |
| 286329 | 18 | 217.4 | 1664 |
| 285185 | 20 | 173.0 | 1896 |
| 285999 | 12 | 76.8 | 1025 |
| 284349 | 5 | 65.3 | 640 |
| 286021 | 1 | 53.0 | 412 |

A 477 cm track cannot be fully contained in SBND on any sensible margin. All 11
are out-of-beam-window, so **no in-window verdict in §2 is affected** — but the
FC column is misleading for cosmics as it stands.

**Root cause, from the compiled config — the two taggers test containment
against different objects:**

| | containment test | volume | margin |
|---|---|---|---|
| `TaggerCheckTGM` | its own `IFiducial` (`inside_fv`) | `BoxFiducial:sbnd_pr_fv`, one box spanning **both** TPCs, x ∈ [−201.05, 201.05] cm | `fv_tolerance` = **−2 / −2.5 / −3 cm** insets |
| `cluster_fc_check` → FC | `fiducial_utils->inside_fiducial_volume(p)` | `DetectorVolumes:dv-apa0-1` (per-APA/face active volumes) | called with **no tolerance vector** ⇒ `m_sd.fiducial->contained(p)`, **zero margin** |

FC's volume is therefore both a different shape *and* strictly more permissive
at every wall. A track ending 1 cm inside the wall is "outside" for TGM and
"inside" for FC. A second contributing mechanism: TGM case B deliberately tags
ends that *are* geometrically inside when a prolonged-signal artefact or dead
region explains them (`TaggerCheckTGM.cxx:5-7`), so a case-B TGM can be
geometrically FC by construction.

Note this divergence is **pre-existing**, not introduced here — `TaggerCheckSTM`
and `TaggerCheckNeutrino` have always used this FC definition against the
per-APA volume while TGM used the box. Making the FC flag visible is what
exposed it.

**Options for the owner** (none applied):

1. **Leave as is.** FC keeps meaning exactly what STM/`match_isFC` mean.
   Self-consistent, but FC≠¬TGM stays surprising.
2. **Give `TaggerCheckFC` a `fiducial` + `fv_tolerance` config** mirroring
   `tagger_check_tgm`, so both taggers can be pointed at `sbnd_pr_fv` with the
   same margins. Cheap and default-OFF-able, but FC would then diverge from
   the STM/`match_isFC` definition unless those are moved too.
3. **Change `cluster_fc_check`** to take the margins. Would alter STM and
   `match_isFC` — a behavior change needing a knob and a full gate.

Option 2 is the smallest step that makes the two columns comparable; option 3
is the one that makes the toolkit self-consistent. Both are physics calls, so
neither was taken.

## 6. Not addressed

**Light mismatch** remains unported (doc 24 §3): its inputs live on
`TimingTPCBundle` during the `QLMatching` visit and reach only the `calib_dump`
JSON, never the pctree the PR job reloads. Nothing in this change moves that.

# Doc 55 — run the PR taggers only on the beam-coincident bundle

The post-Q/L PR tail (`steiner` → `tagger_check_tgm` → `tagger_check_stm` →
`tagger_check_fc`) used to build a Steiner graph and compute TGM/STM/FC
verdicts for **every** matched bundle in the event.  That was the right thing
while the taggers were being ported and validated — the out-of-time bundles
were the sample that exercised them.  With TGM validated (docs 29–39) the
verdicts that matter are the ones for the **beam-coincident** bundle: an
out-of-time bundle is a cosmic by construction, and nothing downstream reads
its TGM/STM/FC flags.

On the 30-event MCP2025C scan that is 27 in-window bundles out of **329**.

**Change (owner request, 2026-07-26): the beam-window gate becomes the SBND
default.**  New knob `beam_window_only` on `CreateSteinerGraph`,
`TaggerCheckTGM`, `TaggerCheckSTM`, `TaggerCheckFC` — C++ default **false**
(so the compiled config with the knob off is byte-identical to the pre-doc-55
one and every other detector is untouched), turned **on** in
`cfg/pgrapher/experiment/sbnd/clus.jsonnet`, i.e. for SBND production.

Window: `[0.2, 2.2) us` on `cluster_t0` (= the matched flash time) — the same
number already used by `beam_pref_tlow/thigh` in SBND Q/L matching, by
`TaggerCheckTGM`'s in-beam protection, by `TaggerCheckNeutrino`'s bundle
selection and by the label table's `in_beam` column.  No new constant.

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# knob-off gate arm (30 events, sequential, setarch x86_64 -R, timecmd metering)
./run_perf54_nusel.sh off p56 -no-bwonly     # -> work-*-p56off
python3 p54_ab_report.py --base-tag p55opt --opt-tag p56off

# new production arm (gate ON = the new default; no flag needed)
./run_perf54_nusel.sh bw d56                 # -> work-*-d56bw
python3 p54_ab_report.py --base-tag p55opt --opt-tag d56bw   # timing table

# hand-scan display (port 5011), previous tag as the side-by-side baseline.
# GOTCHA: serve_nusel_scan.sh forwards the work roots to bokeh VERBATIM, so
# relative paths resolve against the SERVER's cwd, not the script's directory.
# Give ABSOLUTE paths -- a wrong relative root discovers zero events and the
# page comes up empty with no error in the log.
SB=$PWD
nusel_display/serve_nusel_scan.sh 5011 --tag d56bw --charge-src pr \
    --prev $SB/work-mcp10-d55ton:d55ton --prev $SB/work-mcp1000-d55ton:d55ton \
    --prev $SB/work-mcp1000b-d55ton:d55ton \
    $SB/work-mcp10-d56bw $SB/work-mcp1000-d56bw $SB/work-mcp1000b-d56bw
```

Compiled-config proof (before touching any jsonnet, the pre-change config was
saved; `bwcfg.sh` in the session scratchpad compiles the d55ton flag set):

```
cfg_before.json vs beam_window_only=false : BYTE-IDENTICAL (cmp)
beam_window_only=true : beam_window_{only,low,high} present in exactly
    CreateSteinerGraph:pr, TaggerCheckTGM:pr, TaggerCheckSTM:pr, TaggerCheckFC:pr
    (low=200, high=2200 in internal ns = 0.2/2.2 us)
```

## The gate

`cluster_t0` is the matched flash time, stamped on the bundle by Q/L matching
and **inherited by every companion** the un-merge visitors split off
(`separate()` → `from()` copies `cluster_scalar`).  Verified on evt 285185:
all 64 clusters carry their bundle's `cluster_t0` and `matched_flash_gid`
(e.g. gid 1000003 / t0 1.546 us: mains 14, 21 + companions 25, 63, 64).

- **TGM / STM / FC** — the main-cluster loop skips mains whose `cluster_t0` is
  outside `[low, high)`.  Nothing else changes: each main's verdict is computed
  from itself plus its own `matched_flash_gid` companions, so a surviving main's
  verdict cannot depend on which other mains were in the loop.  STM's
  `associated_all` list is deliberately **not** gated — the per-main loop
  already narrows it by the surviving main's gid.
- **steiner** — keeps the in-window mains **plus the companions sharing their
  `matched_flash_gid`** (the same pairing rule STM uses).  The surviving list is
  a subsequence of the ungated one, so the processing order of the kept clusters
  is unchanged, and the gid set is keyed by `int`, never by pointer.
- Empty window (`low >= high`) disables the gate and logs a warning, so
  `beam_window=[0,0]` can never silently blank an event's tagger output.

Each component logs its gate once per event, e.g.

```
visit: TaggerCheckSTM: beam_window_only [0.200, 2.200) us: 1 main(s) evaluated, 11 out of window
CreateSteinerGraph: beam_window_only [0.200, 2.200) us: kept 4 of 24 cluster(s) (1 in-window main(s), 1 flash group(s))
```

### Label table

`nusel_extract.py` reads that log line (`parse_bwonly`) and reports
`tgm/stm/fc = -1` + `stmfit = '-'` for gated-out bundles: the post-PR tree's
`flag_TGM/STM/FC` read 0 for them only because flags default to 0, and printing
0 would claim "evaluated, clean" for a bundle no tagger ever looked at.  Their
`label` stays `not-tagged`.  On an ungated (pre-doc-55) log `parse_bwonly`
returns `None` and the extractor is inert — verified by reproducing the
committed d55ton TSVs for evts 285185 and 287759 byte-for-byte.

## LM is not gated (and why)

The LM (light-mismatch) tagger is **not** part of the PR tail: it runs inside
`QLMatching` in the Q/L step, per bundle, as part of the matching itself
(doc 34).  Gating it would change Q/L matching, not the PR tail, so it is out
of scope here; its `lm_flag` is still stamped on every bundle and the table's
`lm` column is unchanged.

## What the gate removes (30 events)

329 matched bundles, of which **27 are in the beam window** (26 events have
exactly one, evt 286681 has two, and 4 events have none; evt 286197 has an
in-window flash with no bundle).  So 302 of 329 bundle evaluations go away:

| verdict | kept (in-window) | removed (out-of-window) |
|---|---:|---:|
| TGM | 4 | 117 |
| STM | 6 | 46 |
| FC  | 5 | 47 |

The removed TGM tags are the out-of-time cosmics the tagger was validated on;
nothing downstream reads them.  In the label table those bundles go from
`TGM`/`STM`/`not-tagged` to `not-tagged` with `-1/-1/-1` verdicts, and the
scan display renders them "n/a".

## Cost (30 events, sequential, `setarch x86_64 -R`)

Summed `MABC timing` per step (`mabc_step_totals.py p55opt p56off d56bw`):

| step | p55opt (ungated) | p56off (knob off) | d56bw (gate ON) |
|---|---:|---:|---:|
| CreateSteinerGraph:pr | 36.88 | 36.25 | **3.10** |
| TaggerCheckSTM:pr | 16.72 | 16.50 | **1.26** |
| TaggerCheckTGM:pr | 4.41 | 4.29 | **0.59** |
| TaggerCheckFC:pr | 0.44 | 0.43 | **0.04** |
| SbndMagnifyTrackingVisitor:pr | 4.17 | 4.14 | 4.14 |
| PR pipeline TOTAL | 68.61 | 67.51 | **14.75** |

Whole-job wall **146 s → 94 s** (−36%; the remainder is per-event process spawn,
config compile and tensor IO, not the pipeline).  Peak RSS **625 MB → 517 MB**
(−17%): the steiner stage no longer retiles 11 clusters per event, and the
pctree load that used to set the ceiling (doc 54) is now the only large
allocation.  For reference, docs 54's two code rounds bought 164 s → 146 s;
this scope change is 4× that, because it removes work rather than speeding it up.

## Verification

**Knob-off gate (byte-identicality).**  `run_perf54_nusel.sh off p56 -no-bwonly`
vs the round-2 arm `p55opt`, same scope as doc 54 (mabc-pr.zip + pctree-pr member
hashes, TSV text, tracking-stm.root branch content):

```
p54_ab_report.py --base-tag p55opt --opt-tag p56off
GATE: 120/120 comparisons identical -> PASS
TOTAL wall: 146 s -> 146 s      PEAK RSS: 625 MB -> 625 MB
```

`./build/clus/wcdoctest-clus`: 565/565 assertions pass.

**ON-path check** (`bwgate_report.py --base-tag p55opt --gated-tag d56bw`) — the
gate must remove work, never change it.  Baseline is `p55opt`, the SAME-BINARY
ungated arm: `d55ton` predates the `real_cluster_id_global` re-stamp taking
effect in the PR job, so comparing archive layers against it mixes this change
with everything since (it disagrees on 47330 companion `real_cluster_id` values
and on nothing else; `p55opt` vs `d56bw` agrees on all of them).  Verdict
columns are identical in both baselines, the doc-54 gates having proven
d55ton ≡ p55opt for those.

```
events compared           : 30  (30 clean)
in-window bundles checked : 26
tracking-stm.root FIT blocks (T_rec_charge/T_stm_pass/T_stm_eval) of
    gated-arm fitted clusters: 16 compared, 16 bitwise equal
tags KEPT     (in-window) : tgm 4  stm 6  fc 5
tags REMOVED  (out-of-win): tgm 117  stm 46  fc 47
ON-PATH CHECK: PASS
```

i.e. the bundle list per event is unchanged, every in-window bundle's
tgm/stm/fc/stmfit/label reproduces the ungated arm exactly, the post-PR tree
flags agree for in-window clusters, and every surviving STM trajectory fit is
bitwise identical.

**The display's charge is untouched.**  `mabc-pr.zip` keeps the same four layers
with the same names (`0-clustering-global`, `0-stm_fit-global`, two
`channel-deadarea`), and over all 30 events the drawn clustering layer
(`x, y, z, q, cluster_id, type, real_cluster_id`) is **identical** to `p55opt`.
Only `0-stm_fit-global` shrinks (58 kB → 8 kB on evt 285185: one fitted bundle
instead of four).  This matters because steiner now creates 1 temporary child
per event instead of 11 and Bee layers are labelled by order — that ordering did
not shift.

### Two things the ON path DID change (both understood)

1. **`T_proj_data` cells of the surviving bundle** differ in 13 of 30 events —
   its per-cell `charge_err`/`pred_charge` come from `TrackFitting`'s shared 2-D
   accumulator, and `assemble_fitted_charge_2d()` is **last-writer-wins across
   clusters**.  Ungated, an out-of-window cosmic's fit could overwrite the beam
   bundle's cells (leaving `charge_err = 8000`, `pred_charge = 0`); gated, the
   bundle's own fit values survive (e.g. evt 284657 block 50: 3476 of 3725 cells
   go from 8000 to real errors, 2852 gain a non-zero prediction).  The
   trajectory itself is untouched — this is the Magnify 2-D panel getting *more*
   faithful for the beam bundle.  NB (pre-existing, not touched here):
   `assemble_fitted_charge_2d` iterates a **pointer-keyed** map, so which
   cluster won a shared cell was never deterministic in the ungated path.
2. **A torn log line stopped being read as "no verdict".**  With one evaluated
   main per event the log's write-buffer boundary moved onto the FC verdict line
   and 7 of 26 events logged `aggerCheckFC: cluster 5 → FC=false` (head split
   off).  `nusel_extract`'s regexes required the tagger name, so those bundles
   read `fc = -1`.  The regexes now key on the `cluster N <arrow> KEY=` tail
   instead (the tagger name is never the distinguishing token, and \S+ still
   keeps the STM line from matching the TGM one).  Regression: all 30 d55ton
   TSVs regenerate **byte-identically** with the new regexes, so pre-doc-55
   tables are unaffected.

## Hand-scan display

Served on **:5011** with `--tag d56bw` and the previous production tag as the
side-by-side baseline (`--prev ../work-*-d55ton:d55ton`, then the older d52ron /
d49son chain so earlier scan labels still carry forward).  Every out-of-window
bundle will show as *changed* (amber) with `n/a` verdicts — expected, since it is
no longer evaluated.  The beam-window bundle of each event is the one to scan.

## Scope notes

- **`-no-bwonly` is now the tagger-VALIDATION path, not just a legacy escape
  hatch.**  With the gate on, the 30-event scan exercises the taggers on 4 TGM /
  6 STM / 5 FC verdicts instead of 121 / 52 / 52, so a regression in a tagger
  would be nearly invisible on it.  Any future change to TGM/STM/FC must be
  A/B'd with `-no-bwonly` (or `SBND_BW_ONLY=0`) to get a population back; use the
  gate for production throughput, not for validating the taggers themselves.
- The log-regex loosening was regression-checked on **30 logs**, not proven in
  general: it now keys on the `cluster N <arrow> KEY=` tail, so a differently
  torn line could in principle still be missed (or, in the pathological case of
  two spliced messages, misread).  The post-PR tree flags remain the primary
  source; the log is the fallback.
- Other detectors are untouched: the C++ knobs default off and only
  `cfg/pgrapher/experiment/sbnd/clus.jsonnet` turns them on.
- `tagger_check_neutrino` already selected the beam bundle itself (its own
  `beam_window_low/high`), so it needed no change.
- Not gated: `switch_scope`, `unmerge_bundle`, `unmerge_assoc`, the Bee dump and
  the pctree save all still cover every cluster — the event display and the
  bundle table are unchanged in scope.


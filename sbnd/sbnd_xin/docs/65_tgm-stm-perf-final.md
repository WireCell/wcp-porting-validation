# Doc 65 — TGM/STM tagger performance, finalized

The STM tagger is done (doc 63 closed the improvement campaign; doc 64 made
the production operating point the config default), so this doc repeats the
doc-54 perf study once more to record the **final** cost of the PR tagger
stage on the standard 30-event MCP2025C scan.

**Scope: measurement only — no code and no config is changed by this study.**
The arm is the current production binary (toolkit `d1bf5782`) with the
production flag set; the doc-54 arms (`p54base`, `p54opt`, `p55opt`) are the
recorded reference points on the same 30 events, same inputs, same box.

**Result: the whole PR tagger stage now costs ~0.5 s/event of MABC step time
(15 s / 30 events).  TaggerCheckSTM 29.7 s (doc-54 baseline) → 1.26 s
(−96%), CreateSteinerGraph 42.5 → 3.1 s, TaggerCheckTGM 4.5 → 0.6 s,
TaggerCheckFC 1.5 → 0.04 s.  Whole-job wall 164 → 94 s, peak RSS
626 → 517 MB.**  Most of the step-time reduction beyond doc 54's rounds 1–2
comes from the doc-56 beam-window gate, not from further code speed; the
doc-63 STM guard rounds added no visible cost.

## Repro

```bash
cd wcp-porting-img/sbnd/sbnd_xin

# current binary (toolkit d1bf5782, libWireCellClus.so mtime-verified newer
# than every clus source), 30 events, sequential, setarch x86_64 -R,
# per-event wall/RSS via abtest/timecmd.py -- identical harness to doc 54
# sanity first: (cd toolkit && ./build/clus/wcdoctest-clus)  # 565/565 pass
./scripts/runners/run_perf54_nusel.sh fin p65                     # -> work-*-p65fin

python3 scripts/analysis/misc/mabc_step_totals.py p54base p54opt p55opt p65fin   # step table below
python3 scripts/perf/p54_ab_report.py --base-tag p55opt --opt-tag p65fin \
    > /home/xqian/tmp/p65_ab_report.txt           # timing + output-diff census

# profiles (no setarch -- SIGPROF; config precompiled with wcsonnet, M17)
EVT=287517 ROOT=work-mcp1000-p65fin  OUTDIR=/home/xqian/tmp/p65prof_e12 \
    ./scripts/perf/profile_pr65.sh
EVT=288859 ROOT=work-mcp1000b-p65fin OUTDIR=/home/xqian/tmp/p65prof_e20 \
    ./scripts/perf/profile_pr65.sh
google-pprof --text $(which wire-cell) <OUTDIR>/pr65_evt<EVT>.prof
```

`scripts/runners/run_perf54_nusel.sh` reruns ONLY the nusel (PR tagger) stage; imaging and
Q/L products are symlinked from `work-*-d55ton` (M11/M13, identical inputs to
every doc-54 arm).  Flag set unchanged from doc 54: the d52 NUF set +
`-unmerge-assoc` (chord/rescue/rescue-chord, main-pair-real, fvz 5 / fvzi 3 /
fvx 2.5 / fvy 3, `-stm-fit`, `-mip 56000`).  The doc-56 gate and the doc-63
guards need no flags — they are SBND defaults ON.

## Per-step MABC time, 30 events (sequential, setarch x86_64 -R)

| step | p54base | p54opt (r1) | p55opt (r2) | **p65fin** | vs p54base |
|---|---:|---:|---:|---:|---:|
| CreateSteinerGraph:pr | 42.50 | 42.10 | 36.88 | **3.10** | −93% |
| TaggerCheckSTM:pr | 29.73 | 17.49 | 16.72 | **1.26** | −96% |
| TaggerCheckTGM:pr | 4.47 | 4.44 | 4.41 | **0.59** | −87% |
| SbndMagnifyTrackingVisitor:pr | 4.18 | 4.13 | 4.17 | 4.16 | ±0 |
| TaggerCheckFC:pr | 1.46 | 1.46 | 0.44 | **0.04** | −97% |
| everything else (load/save/switch_scope/unmerge/bee) | ~6 | ~6 | ~6 | ~5.7 | ±0 |
| **MABC total** | **88.3** | **75.6** | **68.6** | **14.8** | **−83%** |

Whole-job wall (config compile + tensor IO + python table included):
164 s (p54base) → 146 s (p55opt) → **94 s** (p65fin).  Peak RSS 626 → 625 →
**517 MB** (still evt 285185 = i3).

Former heaviest STM events (s): 287517 3.49 → 1.68 (r1) → **0.003**;
289849 2.98 → **<0.001**; 284657 2.45 → **0.10**; 285185 2.08 → **0.09**.
New heaviest: 288859 at **0.25** (was 2.19).  The 287517/289849 collapse is
the gate skipping their (out-of-window) heavy cosmics entirely.

## Where the reduction comes from (attribution)

The p55opt → p65fin step drops are **not** further code optimization — they
are the intended production behavior changes shipped since doc 54:

1. **Doc-56 beam-window gate** (SBND default ON) does almost all of it.
   Steiner/TGM/STM/FC run only for bundles with cluster T0 inside
   [0.2, 2.2) us.  Over the 30 events (p65fin logs):
   - `CreateSteinerGraph` builds graphs for **179 of 1361** clusters (13%);
   - each tagger evaluates **26** in-window mains and skips **302**
     out-of-window (plus out-of-scope skips).
   The −92% tagger workload matches the STM/TGM step-time drops exactly;
   steiner's −92% time on −87% clusters just says the skipped clusters (long
   cosmics) are heavier than average.
2. **Doc-63 guard rounds 1–5 + doc-49 consistent FV cost nothing visible**:
   per-main STM cost is 1.26 s / 26 = **48 ms** vs p55opt's
   16.72 s / 328 = **51 ms** — unchanged within noise, because the guards
   (second-track, deficit, vertex-kink, proton-muon, cathode, anode-fix)
   piggyback on the existing STM fit and add no new heavy pass.
3. RSS −108 MB: fewer steiner graphs/point clouds resident, same pctree load.

**Outputs intentionally differ from p55opt** — this is a timing measurement,
not an identity gate.  `scripts/perf/p54_ab_report.py --base-tag p55opt --opt-tag p65fin`
finds 118/120 comparisons changed (report:
`/home/xqian/tmp/p65_ab_report.txt`), exactly the docs-49/56/63 behavior:
out-of-window bundles now carry `tgm/stm/fc = -1 -1 -1, stmfit -` (not
evaluated) instead of verdicts, in-window bundles keep theirs (e.g. evt
288859: the sole in-beam bundle at 1.823 us is `nu-candidate` in both arms).
The finalized *physics* baseline is doc 63's (3 residual errors on 923
bundles); this doc only finalizes the cost.

## Profiles (gperftools, 250 Hz, current binary)

Evt 287517 — doc 54's profile event (1579 samples at the doc-54 baseline,
1121 after round 1): now **292 samples**, and its former hotspots are gone —
`TaggerCheckSTM::visit` 1 sample (was 810 = 51%), steiner 7.5%.  The job is
IO/plumbing-bound: `TensorFileSource::load` 13.4%, `TensorDM::as_pctree`
~12%, `SbndMagnifyTrackingVisitor` 11.6%, remainder startup/allocator/
runtime.

Evt 288859 — heaviest remaining (427 samples): `CreateSteinerGraph::visit`
18.7%, `TaggerCheckSTM::visit` 12.6% (of which `do_single_tracking` 11.9%,
`dQ_dx_fit` 5.2%), `TaggerCheckTGM::visit` 7.7%, Magnify 7.5%, tensor
load/decode ~10%.  No single physics hotspot ≥20%.

## Next levers (none worth taking)

The MABC steps sum to ~15 s of the 94 s whole-job wall; the rest is
per-event process spawn, jsonnet config compile, tensor IO and the python
table — a runner/workflow concern (batch daemon, precompiled config), not a
clus code round.  Within the steps, the single largest is now
`SbndMagnifyTrackingVisitor` (4.16 s, ahead of steiner's 3.10) — a
diagnostic output writer, only present because the production flag set keeps
`-stm-fit`.  The tagger campaign closes here.

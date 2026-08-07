# pr/41 — PR chain: how to run it, what it writes, how Bee links are made

A concise map of the full SBND pattern-recognition (PR) chain for anyone new
to this tree: input → stages → output files → Bee display. Not a reference on
any single knob (those are in the numbered docs it links to) — start here,
then follow a link when you need depth.

## 1. What "the PR chain" means here

The PR chain is the second half of SBND reconstruction, run on an existing
Q/L (charge/light matching) output. It never re-runs imaging or Q/L (M11) —
it reads a `ql_evt<ID>/` directory and adds: bundle splitting/rejoin
(`unmerge_bundle`, `steiner`, `protect_bundle`), the cosmic taggers (TGM/
STM/FC), the neutrino tagger (`tagger_check_neutrino`, includes the DL/SCN
vertex), the two BDT scorers (numu/nue), track fitting, and the T_tagger/
T_kine writer. This is the stage that produces `numu_score`/`nue_score`/
`T_kine` — nothing upstream of it does.

Driver: **`run_pr_chain_batch.sh <ql_root> <out_root> <data|sim> [evt ...]`**
(doc pr/11). Fork-by-duplication of `run_nusel_evt.sh`'s `process_event()`
(M10 — that production script stays untouched); this one appends the 5
neutrino-PR stages every other runner stops short of.

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
./run_pr_chain_batch.sh work-nuecc48-cb0805 work-myrun data 388 74544
```

- `ql_root` — a directory of `ql_evt<ID>/{pctree-evt<ID>.tar.gz,
  opflash_apa0.tar.gz, mabc-all-apa.zip}` from a prior Q/L run. Directories
  are keyed by **event id only**; run number is read from the Q/L job's own
  opflash metadata, not from the path.
- `out_root` — must be fresh (script refuses to reuse a dir it didn't create
  — M13).
- `data|sim` — reality TLA. **Must match the `ql_root`'s own lineage** (doc
  pr/38 round 3: `switch_scope`'s `pos_offset` correction is reality-gated,
  so a mismatch silently flips borderline scope/PID decisions). The script
  checks this against `ql_root/.lineage_reality` and refuses to run on a
  mismatch unless `SBND_ALLOW_REALITY_MISMATCH=1`.
- `[evt ...]` — optional explicit event-id subset; default is every
  `ql_evt<ID>` found under `ql_root`.
- `PR_JOBS` (default 6, M5) controls parallelism; check
  `cut -d' ' -f1 /proc/loadavg` after launching a large batch.

The chain itself is one `wire-cell -c wct-pr-perevt.jsonnet` invocation per
event (see the script for the full TLA list — mostly default-OFF knobs from
the numbered docs; a bare run with no env overrides reproduces the SBND
production operating point).

## 2. Per-event output — where things land

`out_root/pr_evt<ID>/`:

| file | what it is |
|---|---|
| `mabc-pr.zip` | Bee-format zip (from `MultiAlgBlobClustering`): the PR-stage clustering, written as Bee JSON layers under `data/<n>/`. **This is the file you upload to Bee.** |
| `tracking-pr.root` | ROOT file with the **tagger information**: `T_tagger` (TGM/STM/FC/neutrino verdicts, `numu_score`, `nue_score`, the BDT sub-scores) and `T_kine` (per-particle kinematics/energy), plus the Magnify-style tracking tree (`SbndPrMagnifyTrackingVisitor`). Written by `tagger_output` (`UbooneTaggerOutputVisitor`, reused from uBooNE as-is — pure `TaggerInfo`/`KineInfo` dump, no geometry) in `UPDATE` mode, so it must run *after* `tracking_visitor` and *after* both BDT scorers in the pipeline order. |
| `pctree-pr-evt<ID>.tar.gz` | the point-cloud tree tensors after the PR stages (`save_tensors`) — the PR-side counterpart of the Q/L `pctree-evt<ID>.tar.gz`. |
| `nusel-evt<ID>.tsv` | one row per qualifying bundle (main-flagged + in-scope), built by `nusel_extract.py` from `mabc-pr.zip` + the PR log + the Q/L pctree. Columns include the TGM/STM/FC/LM/neutrino-candidate label — this is the fast per-event summary; read `T_tagger`/`T_kine` in the ROOT file for the full score/kinematics detail. |
| `wct_pr_evt<ID>.log`, `stdout.log`, `rc.txt`, `.time.meta` | run log, driver stdout, wire-cell exit code, `timecmd.py` timing. |
| `calib-pr-evt<ID>.json` | **only if `PR_EXTRA_STAGES=pr_display`** (see §4) — the standalone dump the Bokeh PR event display reads. Read-only stage; does not change any other output. |

Batch-level: `out_root/nusel-table.tsv` + `nusel-events.tsv` (merged across
all events in the run, same shape as `run_nusel_evt.sh`'s `all` output).

## 3. Bee links — how they're generated

Two independent Bee-format zips exist per event:

- `ql_evt<ID>/mabc-all-apa.zip` — the Q/L-stage Bee dump (pre-PR).
- `pr_evt<ID>/mabc-pr.zip` — the PR-stage Bee dump (post-PR; this is
  normally the one you want, since it has the tagger-scoped clustering).

Either is uploaded the same way, with **`upload-to-bee.sh <path-to-zip>`**
(top level of this repo): it logs into `https://www.phy.bnl.gov/twister/bee`,
POSTs the zip, and prints a URL of the form
`.../bee/set/<uuid>/event/list/` — open that (or let `$BROWSER` do it) to get
the per-event Bee viewer links. No local server involved; this talks
directly to the BNL Bee instance.

```bash
./upload-to-bee.sh work-myrun/pr_evt388/mabc-pr.zip
```

For **local, no-upload** inspection instead, there's the Bokeh PR event
display on port 5017 (doc pr/26/§4 below) — it reads `calib-pr-evt<ID>.json`
directly, no Bee round-trip.

## 4. The PR event display (local viewer, no Bee upload)

```bash
PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh <ql_root> <out_root> <data|sim> <evt ...>
./pr_display/serve_pr_display.sh 5017 <out_root>/pr_evt<ID>/calib-pr-evt<ID>.json ...
# then:  ssh -L 5017:localhost:5017 wcgpu1.phy.bnl.gov
#        open http://localhost:5017/pr_display_viewer
```

`pr_display` (`PrDisplayDump`) is a default-OFF, read-only stage: it must run
*after* `tagger_check_neutrino` (it reads the `TrackFitting` slot and PR
graph that stage fills) and never mutates state, so an arm with it hashes
identically to one without on every other output. Full write-up: doc pr/26.

The viewer also has a **dQ/dx panel** (doc pr/42): click a particle-flow row
to plot its measured dQ/dx against the muon/proton/pion/kaon reference
curves (tracks, residual range) or the 1x/2x MIP lines (showers, distance
from the stem). The six 2-D wire-plane panels doc pr/26 built are hidden by
default now (`--wire-planes` on the `bokeh serve` command restores them).

## 5. Trained-model loading — read this before trusting a score

**Neither model was trained on SBND data.** Both are the stock uBooNE
artifacts, loaded by path from `wire-cell-data`:

- **DL (SCN) neutrino vertex**: `dl_weights` (default
  `uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth`, doc pr/4 — this has been
  the SBND production default since `e3d46c91`). Needs `libpython` preloaded
  `RTLD_GLOBAL` (`LD_PRELOAD=libpython3.11.so.1.0`) or the SCN import fails
  **silently** at the C++ level — the only symptom is a
  `"DL vertex failed: ..."` WARN in the log (grep for it; the driver does
  this automatically per event) and a fallback to the geometric vertex. If
  you ever see `numu_score`/`nue_score` looking off for a whole batch, check
  this first. Set `SBND_DL_WEIGHTS=` (empty, not unset) to force the
  geometric vertex and isolate a DL-vertex effect from a PR-structure one.
- **NuMu / nue BDT scorers**: `UbooneNumuBDTScorer` / `UbooneNueBDTScorer`,
  geometry-free `TaggerInfo` consumers, loading ~35 TMVA/XGBoost XML files
  from `wire-cell-data/uboone/weights/` (same files the qlport gate job
  books). **These weights are uBooNE-trained and unretrained on SBND** — the
  scores are usable for availability and relative ranking only, not as
  calibrated probabilities. SBND retraining is an open gap (doc pr/2 §G1).

Both are wired through `dl_weights=`/`bdt_weights_dir=` jsonnet params in
`clus.jsonnet` / `wct-pr-perevt.jsonnet` (search those names to change the
path), not hardcoded — but changing the path changes reconstruction output,
so treat it like any other CLAUDE.md §1 knob.

## 6. Where to go next

- Batch driver internals + every `SBND_*` knob it exposes: `run_pr_chain_batch.sh` header comments (each knob cites its doc).
- Pipeline stage ordering rationale (why protect_bundle sits where it does, etc.): doc pr/23 §9, doc pr/11.
- PR event display, field-by-field: doc pr/26; stage map: doc pr/27.
- `nusel-evt<ID>.tsv` column semantics: `nusel_extract.py` module docstring.
- Reality/lineage mismatch mechanics: doc pr/38 §round 3.
- Muon/pion/shower PID mis-ID fixes (chain-aware proton veto, stale Shower-type cache, shower-trajectory dQ/dx guard, kine-tree PF-tree barrier parity): doc pr/40 (rounds 1-6), doc pr/43.

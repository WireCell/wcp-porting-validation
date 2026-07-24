# 26 — `check_neutrino_candidate` ON by default, and column-aligned tables

Two changes to the SBND neutrino-selection chain, both in `sbnd_xin` only:

1. **`-nucand` is now the default.** `run_nusel_evt.sh` passes
   `tgm_neutrino_candidate=true`, so `TaggerCheckTGM` runs the ported
   `check_neutrino_candidate` Dijkstra path-topology veto and in-beam-window
   bundles become taggable. `-no-nucand` / `SBND_TGM_NUCAND=0` restores the
   old behavior.
2. **`nusel-table.tsv` / `nusel-events.tsv` are written column-aligned**, so
   they are readable as-is instead of only through `column -t`.

> ⚠ **(1) is a reconstruction-behavior change, not a byte-identical one.** It
> is a deliberate default flip requested by the owner. Its full effect on the
> 10-event sample is **one row** (§2). The C++ and jsonnet defaults are
> untouched — see §1.1.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
SBND_MAX_JOBS=5 SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all
# old behavior for comparison:  ./run_nusel_evt.sh data -no-nucand all
```

## 1. The default flip

`check_neutrino_candidate` is the prototype veto that decides whether an
in-beam-window bundle may be tagged TGM. With it off, `TaggerCheckTGM` is
conservative by construction: the protected branches **never** tag an in-window
bundle, so a genuine in-time through-going muon could not be called one. That
is the behavior doc 23/25 measured. Turning it on is what makes the in-window
population actually taggable.

### 1.1 Scope of the flip — deliberately narrow

Only the runner opts in:

| layer | default | after this change |
|---|---|---|
| C++ `TaggerCheckTGM::m_check_neutrino_candidate` | `false` | **unchanged, still `false`** |
| jsonnet `tagger_check_tgm(check_neutrino_candidate=false)` and `wct-pr-perevt.jsonnet` TLA | `false`, key suppressed when off | **unchanged** |
| `run_nusel_evt.sh` `NUCAND` | `0` | **`1`** — passes `tgm_neutrino_candidate=true` explicitly |

So every other config, experiment and detector compiles and runs exactly as
before; the knob-off compiled JSON still omits the key entirely. Only this
chain's output moves. Widening the default into the C++ or jsonnet layer would
change other people's production output and was **not** done.

**Compiled-config proof (M6).** `wcsonnet` on `wct-pr-perevt.jsonnet` with the
production pipeline:

```
tgm_neutrino_candidate=true  -> TaggerCheckTGM.check_neutrino_candidate = True
tgm_neutrino_candidate=false -> TaggerCheckTGM.check_neutrino_candidate = <key absent>
```

## 2. Effect on the 10-event sample — exactly one row

Row set identical (76 rows). Fields changed on **1 of 74** bundles:

| evt | main | in_beam | len_cm | change |
|---|---:|---:|---:|---|
| 284349 | 11 | 1 | 202.1 | `tgm 0→1`, `stm 1→0`, `label STM→TGM` |

This is precisely the event the knob was ported for (toolkit `f1780d50`: *"SBND
evt 284349 1.555 µs in-beam anode→downstream crosser was untaggable"*). The
`stm` flip to 0 is mechanical — `TaggerCheckSTM` skips mains already flagged
TGM. Log confirms: `TaggerCheckTGM: cluster 11 → TGM=true`.

Calling a 202 cm anode→downstream crosser a through-going muon rather than a
stopped muon is the intended correction, not a regression.

### 2.1 Label counts, `work-mcp10` (74 bundles + 2 no-bundle rows)

| label | n | in-window | FC |
|---|---:|---:|---:|
| TGM | 34 | 1 | 11 |
| STM | 0 | 0 | 0 |
| nu-candidate | 7 | 7 | 4 |
| not-tagged | 33 | 0 | 12 |
| no-bundle | 2 | 2 | — |

FC totals are unchanged: **27 FC / 47 not-FC**. The FC verdict is independent
of the cosmic taggers, so the flip moved no `fc` value.

**The neutrino selection is unaffected on this sample.** All 7 nu-candidates
survive unchanged, and `nusel-events.tsv` is **identical field-for-field** to
the `-no-nucand` run — evt284349 was already `cosmic-tagged` via STM, so
re-classifying it as TGM does not move any event-level verdict.

The STM count is now **0** across all 10 events. That is a consequence of the
single flip, not evidence that STM is broken: it had exactly one tag before.

> Supersedes the label counts in doc 25 §2 (TGM 33 / STM 1) and doc 23 §3.
> The FC counts and all doc-25 gates still stand.

## 3. Column-aligned tables

`nusel-table.tsv` and `nusel-events.tsv` were genuine tab-separated files:
`column -t` (which the runner pipes through for display) made them look aligned,
but the files themselves were ragged when read directly.

They are now written space-padded so each column lines up, by a shared
`write_table()` in `nusel_extract.py`.

```
run    subrun  event   main_id  flash_gid  flash_apa  flash_grp  flash_time_us  ...  tgm  stm  fc  label
18255  1       284349  8        1000000    1          6          -238.770       ...  1    0    0   TGM
18255  1       284349  11       4          0          7          1.555          ...  1    0    0   TGM
```

**Still machine-readable.** No field ever contains a space — ids and counts are
numeric and every label is hyphenated (`nu-candidate`, `no-bundle`,
`not-tagged`, `cosmic-tagged`, `no-beam-flash`) — so consumers split on runs of
whitespace: `awk '{print $19}'`, `sort -k3`, `str.split()`, or
`pandas.read_csv(..., sep=r'\s+')`.

**Backward compatible.** The new `read_table()` accepts *both* layouts, so
`--merge` still reads pre-existing tab-separated per-event tables. Verified:
re-merging the old tab-separated `nusel-evt*.tsv` files through the new code
reproduces the previous table **field-for-field** (77 lines) and the previous
event summary (11 lines).

Two deliberate details:

- `--no-header` output stays **tab-separated**. It exists for concatenation,
  where per-file column widths would not line up anyway.
- The `.tsv` filenames are kept. The paths appear in doc 23/25, the runner and
  the merge step; renaming them would churn more than it clarifies. The
  extension is now a slight misnomer — the files are whitespace-aligned, not
  strictly tab-separated.

## 4. Files changed

`sbnd_xin/` only — no toolkit C++ or cfg change in this round, so toolkit
commit `959f9e02` (TaggerCheckFC) is untouched.

| file | change |
|---|---|
| `run_nusel_evt.sh` | `NUCAND` default `0`→`1`; new `-no-nucand`; help + header updated |
| `nusel_extract.py` | `write_table()` (aligned writer), `read_table()` (accepts both layouts); merge/one-event/events writers use them |
| `work-mcp10/` | retabled: `nusel-table.tsv`, `nusel-events.tsv`, per-event `nusel-evt*.tsv`, PR logs |

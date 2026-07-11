# AI hand scan of the PDVD Q/L matching — method

How the AI (Claude) performs the Q/L hand scan of the first 10 events of run
039252 (idx 0-9, art events 298567-298693) headlessly, producing labels that
load in the standard `pdvd/ql_scan` viewer. Companion docs:
`ql-scan-criteria.md` (the judgment rubric, inferred from the human SBND/PDHD
scans) and `ql-scan-findings.md` (results).

## Pipeline

```
calib-evt<ID>.json  (work/039252_<idx>/, from run_clus_evt.sh -calib)
   |
   |  render_groups.py            one composite PNG per scan group with the
   v                              viewer's evidence panels + context JSONL
png/evt<ID>/grp###_gid<N>.png     (all nav groups) with per-candidate
context/context-evt<ID>.jsonl     metrics + compare-table evidence
   |
   |  AI judgment (vision + numerics, rubric = ql-scan-criteria.md)
   v
decisions/decisions-evt<ID>.jsonl one line per judged bundle:
   |                              {"event","group","flash_gid","apa",
   |                               "main_cluster_uid","main_cluster_ident",
   |                               "verdict":"keep|reject|add",
   |                               "auto_selected","confidence","reason"}
   |
   |  make_labels.py --tag claude
   v
work/ql_labels/claude/labels-evt<ID>.json      viewer-compatible Save output
work/ql_labels/claude/.scan_state-evt<ID>.json viewer autosave (restores the
                                               selection on event load)
```

The PNGs and label JSONs are gitignored (`*.png`, `*.json`); the committed
scientific record is `decisions/*.jsonl` (+ these docs), from which the
labels are deterministically regenerable.

## Formula / semantics provenance (replicated from ql_scan_viewer.py)

- Scan group = one shared flash, numbered by ascending flash time
  (`Event.__init__`).
- Drift correction: `x_display = x_raw + sign_offset * (t_flash +
  trigger_offset) * drift_speed` (`Event.dx_cm`); the dumps carry
  `trigger_offset = 0.0` because the per-crate offsets are already applied by
  the matcher — `trigger_offsets_us` is NOT re-applied.
- Cluster length = 3-D bbox diagonal (`Event.cluster_length`); the viewer's
  navigation length filter (`MIN_CLUS_LEN` = 5 cm) defines the scan universe.
- Light-map side membership by opdet x: cathode XAs (|x| <= 10 cm) on both
  volumes (`Event.od_side`).
- Label files reproduce `_match_entry` / `on_save` / `save_state` exactly
  (key order, sorted-bundle-index iteration, `indent=1`).

## Scan protocol

- **Pass A (review)**: every group holding >= 1 auto-selected bundle whose
  main cluster passes the 5 cm filter gets a rendered composite; each such
  auto bundle receives an explicit keep/reject; non-auto candidates in those
  groups may be promoted (`add`), e.g. the second volume's half of a cathode
  crosser.
- **Auto bundles hidden by the length filter** (main cluster <= 5 cm,
  ~25-50/event) default to keep without a decision line — the human scanner
  never sees them either (the viewer navigation skips them).
- **Pass B (sweep)**: the context JSONL covers ALL nav groups; groups without
  auto matches are escalated (rendered with `--groups`) only when the
  numerics show a compelling non-auto candidate for a still-unmatched long
  cluster. Adds from Pass B are conservative and tagged low/medium
  confidence.
- Duplicate assignments (one cluster auto-kept on two flashes — the matcher
  does produce these; `make_labels.py` warns) are resolved during Pass A via
  the compare-table evidence.

## Reviewing the AI scan

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
./ql_scan/serve_ql_scan.sh 5017 --tag claude work/039252_{0,1,2,3,4,5,6,7,8,9}/calib-evt*.json
```

The viewer auto-restores the AI selection per event (scan_state); tick/untick
to correct it and Save under the same tag (or work in your own tag and diff
the `labels-*.json`). The per-decision reasons live in
`ql_display/decisions/decisions-evt<ID>.jsonl`.

## Commands (reproduce)

```bash
cd pdvd/ql_display
PY=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/python
$PY analyze_labels.py                       # criteria study (stdout)
$PY render_groups.py ../work/039252_0/calib-evt298567.json \
    --outdir png/evt298567 --context context/context-evt298567.jsonl
$PY make_labels.py ../work/039252_0/calib-evt298567.json \
    decisions/decisions-evt298567.jsonl --tag claude --check
```

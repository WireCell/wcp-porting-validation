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

## Xe/175 nm re-scan round (2026-07-11)

After the data-driven library verdict (`../../docs/pdvd-questions-dune.md`
§3) the QL default switched to the Xe/175 nm library (toolkit `0adb15fa`:
175 nm grid, `eff_Xe`, ch 13/29/39 unmasked, QtoL 0.082) and the 10 events
were reprocessed (the 128 nm dumps are kept as
`work/039252_*/128nm-calib-evt*.json`). The scan was REDONE from scratch on
the new dumps with the same rubric:

- evidence: `png-xe/evt<ID>/` + `context-xe/context-evt<ID>.jsonl`
- decisions: `decisions-xe/decisions-evt<ID>.jsonl`
- labels: tag **`claude-xe`** (the 128 nm round stays under tag `claude`,
  meaningful only against the archived 128 nm dumps)
- scanner deltas vs round 1: ch 13/29/39 now appear in the light maps and
  channel panels; the auto set differs (~+5-10 bundles/event at the new
  QtoL); amplitude bands, nearPD logic and duplicate resolution unchanged.

Review:
`./ql_scan/serve_ql_scan.sh 5017 --tag claude-xe work/039252_{0,1,2,3,4,5,6,7,8,9}/calib-evt*.json`

## Geometry-fix round (2026-07-11)

A PDVD-specific active-volume Y-truncation bug in `QLMatching::compute_geometry`
(toolkit `565ccd62`, see `pdvd-tpc-geometry-fiducial.md` and
`ql-scan-findings.md` §8) was found and fixed while reviewing the display's
detector-box drawing. It affected both the drawn box and the light-
prediction inclusion gate, so all 120 PDVD `-calib` dumps were reprocessed
and QtoL refit (0.082 -> 0.070, `cfg/.../protodunevd/qlmatching.jsonnet`).
The 10 scan events were rescanned from scratch on the corrected dumps:

- evidence: `png-geomfix/evt<ID>/` + `context-geomfix/context-evt<ID>.jsonl`
- decisions: `decisions-geomfix/decisions-evt<ID>.jsonl`
- labels: tag **`claude-geomfix`** (both `claude` and `claude-xe` stay valid
  only against their own archived pre-fix dumps)

Review:
`./ql_scan/serve_ql_scan.sh 5017 --tag claude-geomfix work/039252_{0,1,2,3,4,5,6,7,8,9}/calib-evt*.json`

## Cathode-XA operating-point round (2026-07-16/17) — full 18-event set

After the cathode-XA operating point was adopted (doc 18: wall-XA mask,
cathode flash admission 5PE/2ch, wall flags off, cathode-scoped
reject_overpred) and all 18 events reprocessed into
`work/039252_<idx>_cathxa/`, the scan was extended to the FULL event set:
idx 1-17 scanned from scratch with the same rubric (idx 0 = evt298567 keeps
the owner's doc-17 gold scan, remapped into tag `cathxa`; evt298581 was
scanned blind and cross-checked against the owner's same-day save — 94%
cluster-level agreement).

- evidence: `png-cathxa/evt<ID>/` + `context-cathxa/context-evt<ID>.jsonl`
- decisions: `decisions-cathxa/decisions-evt<ID>.jsonl` (17 events)
- labels: tag **`claude-cathxa`** (valid against the cathxa dumps only)
- findings: `docs/ql-scan-findings-cathxa.md`
- scanner deltas vs the geomfix round: one scanner agent per event (6 in
  parallel); pins/rescues (`xtpc_pin`/`xtpc_cathode_rescued`) explicitly
  verified by cathode-meeting distance + unsaturated-channel amplitude;
  amplitude judged on unsaturated, unmasked channels throughout.
- harness note: a killed 21:37 render batch left idx 0-6 context/PNGs
  truncated mid-event; scanners fell back to calib-dump enumeration
  (`make_labels --check` enforces verdict completeness) and the evidence
  files were re-rendered to full length afterwards.

Review:
`./ql_scan/serve_ql_scan.sh 5021 --tag claude-cathxa work/039252_*_cathxa/calib-evt*.json`

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

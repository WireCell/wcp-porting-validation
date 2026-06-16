# PDHD waveform hand-scan viewer

A small Bokeh display for hand-scanning the per-channel waveform overview plots
`pics/pd/wf_ch<NNN>.png` (one PNG per optical channel, 3 rows small/medium/large
× 2 cols raw/decon, written by `../pd_plot/spe_waveform_examples.py`). It pages
through the images and records a **Good / Bad** verdict plus an optional comment
per channel, so the verdicts can drive later per-channel decisions.

> **Status:** machinery only. The `wf_ch*.png` plots are still being refined; this
> is the tool to scan them once they are final.

## Serve

```bash
# default port 5016 (img_plot owns 5013, pd_plot 5014, ql_scan 5015)
./serve_wf_scan.sh 5016
# or name the scanner / point at a different image set:
./serve_wf_scan.sh 5016 --scanner xin ../pics/pd
```

From a workstation:

```bash
ssh -L 5016:localhost:5016 <host>
# then open http://localhost:5016/wf_scan_viewer
```

## Scan workflow

1. The current channel's PNG is shown inline. Use the **Channel** dropdown or
   **Prev / Next** to page through (wraps around).
2. Type an optional note in **Comment** (used mostly for Bad, but saved for either).
3. Click **Good** or **Bad** — the verdict is written to disk immediately and the
   view auto-advances to the next channel.
4. **Undo** reverts the last write (session-only, one deep).

The scan is resumable: re-launching reloads `scan_results.json`, pre-fills each
channel's saved comment, and shows its current verdict in the status line. A
re-scan of a channel overwrites its entry. The bottom table lists everything
scanned so far; the status line shows running good/bad/unscanned counts.

## Result file

Written beside the images as `pics/pd/scan_results.json`, a single object keyed by
channel:

```jsonc
{
  "ch000": {
    "verdict": "good",
    "comment": "",
    "scanner": "anon",
    "image": "wf_ch000.png",
    "scanned_at": "2026-06-15T17:02:11"
  },
  "ch005": {
    "verdict": "bad",
    "comment": "decon ringing",
    "scanner": "xin",
    "image": "wf_ch005.png",
    "scanned_at": "2026-06-15T17:03:40"
  }
}
```

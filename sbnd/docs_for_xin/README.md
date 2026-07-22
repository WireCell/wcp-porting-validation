# SBND imaging + clustering + Q/L matching — run instructions

Two ways to run the SBND WireCell imaging + clustering + charge-light (Q/L)
matching over an artROOT (reco1) input, both inside the SL7 apptainer with the
local builds (`/exp/sbnd/app/users/yuhw/opt`, via `sbnd/setup-ap.sh`):

1. **[1-step: one larwirecell job](1-run-1step-larwirecell-artroot.md)** —
   a single `lar` job (WireCellToolkit module) reads the artROOT products and
   runs imaging → clustering → joint Q/L matching in one process.  Simplest;
   use this for normal running.

2. **[2-step: dump then standalone wire-cell](2-run-2step-dump-standalone.md)** —
   step 1 uses larwirecell to *dump* the WireCell inputs (image clusters,
   opflash, SP frames) to standalone files; step 2 runs a pure `wire-cell`
   (no LArSoft) job on those files.  Use this to develop/debug the WireCell
   graph without the LArSoft round-trip, or to hand the dumped files to
   someone without a LArSoft setup.

Both produce the same Bee display (`mabc.zip`: `img`/`clustering`/`op`/`tgm`
sets) and are kept in sync (the standalone graph mirrors the 1-step graph).

## Common notes

- **Environment (both)**: run inside the SL7 container and `source
  sbnd/setup-ap.sh` first.  See `sbnd/docs/1-run-tests-sl7-local-builds-sbnd.md`
  for the container invocation and env details.
- **sim vs data**: the entry configs take `reality` = `"sim"` or `"data"`.
  It selects a grouped reco config in the toolkit clus maker
  (`sim` → `use_sce=true`, `pos_offset_on=false`; `data` → `use_sce=false`,
  `pos_offset_on=true`) and, for the 1-step, whether the truth labeler emits
  truth.
- **DATA needs frameshift first**: for **Gen2 real data**, if the input
  artROOT file does not already contain the FrameShift product, run
  `run_frameshift.fcl` on it first (`lar -c run_frameshift.fcl -s <in>.root`
  → `<in>_frameshift.root`) and use that as the input.  See
  `sbnd/samples/docs/gen2-data-frameshift.md`.
- **Upload to Bee**: `BROWSER=echo bash sbnd/sbnd_xin/upload-to-bee.sh <mabc.zip>`.

# 2-step: dump (larwirecell) then standalone wire-cell

Split the chain in two: **step 1** uses larwirecell (`lar`) to *dump* the
WireCell inputs from the artROOT products into standalone files, and **step 2**
runs a pure `wire-cell` job (no LArSoft) on those files.  The step-2 graph is
kept in sync with the 1-step (`wcls-img-clus-matching-xin.jsonnet`), so the Bee
output is identical (verified below).

All commands run inside the SL7 apptainer after
`source sbnd/setup-ap.sh`.  The dump fcls and the standalone jsonnet live in
`sbnd/standalone-sample/`, so put it on the search paths:

```bash
source /nashome/y/yuhw/.bashrc
source /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/setup-ap.sh
SS=/exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/standalone-sample
export FHICL_FILE_PATH=$SS:$FHICL_FILE_PATH        # dump fcls
export WIRECELL_PATH=$SS:$WIRECELL_PATH            # dump + standalone jsonnets
cd $SS   # or any working dir; the dumps + mabc.zip land in cwd
```

## Step 1 — dump (larwirecell -> standalone files)

Dump all three WireCell inputs.  Each is a separate `lar` job over the same
artROOT input; they are independent and can run in parallel (up to 8 cores).

**DATA needs frameshift first** — if the input file has no FrameShift product,
run `run_frameshift.fcl -s <in>.root` and use the `_frameshift.root`
(`sbnd/samples/docs/gen2-data-frameshift.md`).

sim (MC):
```bash
lar -n 1 -c wcls-img-dump.fcl        -s <mc>.root --no-output   # -> icluster-apa{0,1}-{active,masked}.npz
lar -n 1 -c wcls-flash-dump.fcl      -s <mc>.root --no-output   # -> opflash_apa{0,1}.tar.gz
lar -n 1 -c wcls-frame-dump.fcl      -s <mc>.root --no-output   # -> frames-dnn.tar.bz2  (SP frames)
```

data:
```bash
lar -n 1 -c wcls-img-dump-data.fcl   -s <..._frameshift>.root --no-output   # sptpc2d tags
lar -n 1 -c wcls-flash-dump.fcl      -s <..._frameshift>.root --no-output   # opflashtpc0/1 labels (same as sim)
lar -n 1 -c wcls-frame-dump-data.fcl -s <..._frameshift>.root --no-output   # sptpc2d tags
```

What each dumps:
- **img** (`wcls-img-dump[-data].fcl`) → `icluster-apa{0,1}-active.npz` +
  `-masked.npz`: the imaged blob clusters (live=active, dead=masked).  The
  imaging is kept in sync with the 1-step (`multi-3view` + `full_deghost`), so
  these blobs are exactly what the 1-step feeds to clustering.
- **flash** (`wcls-flash-dump.fcl`) → `opflash_apa{0,1}.tar.gz`: the opflash
  optical tensors (labels `opflashtpc0:`/`opflashtpc1:`, same for sim/data).
- **sp / frame** (`wcls-frame-dump[-data].fcl`) → `frames-dnn.tar.bz2`: the
  DNN signal-ROI `recob::Wire` frames (viewable with `wirecell-plot frame`).
  Not consumed by step 2 in this recipe (we feed the dumped `icluster`
  directly), but part of the standard dump set (e.g. to re-image standalone
  via `sbnd_xin/wct-img-all.jsonnet`).

## Step 2 — standalone wire-cell (clustering + Q/L matching)

Reads the dumped `icluster` + `opflash` and runs the SAME per-APA clustering →
joint QLMatching → all-APA MABC as the 1-step, writing `mabc.zip`.

```bash
wire-cell -l stdout -L info \
    -V reality=data \        # or sim
    -V input=. \             # dir holding icluster-*.npz + opflash_*.tar.gz
    -c wct-clus-matching-standalone.jsonnet
BROWSER=echo bash /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/sbnd_xin/upload-to-bee.sh mabc.zip
```

- `reality` picks the SAME grouped reco config as the 1-step (`data` →
  `use_sce=false`, `pos_offset_on=true`).
- No `wclsTensorSetLabeler`: it is a larwirecell component needing the art
  event, so the standalone produces the reco Bee sets only (no truth / no
  nugraph HDF5).

## Sync with the 1-step

`wct-clus-matching-standalone.jsonnet` mirrors
`sbnd/wcls-img-clus-matching-xin.jsonnet`: it imports the SAME toolkit
`clus.jsonnet` + `qlmatching.jsonnet`, uses `FlashTensorToOpticalPCs` +
joint `matching_joint` (premerged all-APA MABC), and appends
`cathode_fv.configs` (the `cpa-exclusion` CompositeFiducial the matcher
references by tn).  Only the sources differ (ClusterFileSource /
TensorFileSource here vs art sources there).  If the 1-step graph changes, keep
this file and `wcls-img-dump.jsonnet` (the imaging) in step.

## Validation

Data (1 event, `..._eventidfiltered_frameshift.root`): the 2-step `mabc.zip`
is **byte-identical** to the 1-step across all Bee sets (clustering-apa0/apa1/
global, img-global, op, tgm-global, channel-deadarea — MD5 match).

- 1-step BEE: https://www.phy.bnl.gov/twister/bee/set/7582c18b-1c0a-4eae-86d4-43ff56de3909/event/list/
- 2-step BEE: https://www.phy.bnl.gov/twister/bee/set/2345684d-d233-4711-9c93-4b2768488cb1/event/list/

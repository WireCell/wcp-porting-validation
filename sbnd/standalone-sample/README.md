# standalone-sample

## check SP results
```bash
lar -n 10 -c wcls-sp-dump.fcl -s 2025f-mc.root --no-output
wirecell-plot frame -t dnnsp -o sp-frames.pdf sp-frames.tar.bz2

python plot_simchannels.py --input 2025f-mc.root --entry 0 --channel-min 0 --channel-max 1983 --vmax-percentile 80 --out-prefix simchannels_entry0
python plot_simchannels.py --input 2025f-mc.root --entry 1 --channel-min 7250 --tdc-min 4000 --tdc-max 5000 --channel-max 7600 --vmax-percentile 80 --out-prefix simchannels_entry1
python plot_simchannels.py \
  --input 2025f-mc.root \
  --entry 1 \
  --channel-min 0 \
  --channel-max 1900 \
  --interactive \
  --initial-channel 1000
```

## dump dnn recob::Wire frames

Dump the DNN signal-ROI `recob::Wire` products into WCT frame files
(float32 `.tar.bz2`, viewable with `wirecell-plot frame`).
The fcl defaults to MC tags (`simtpc2d:dnnsp`); for data, swap the active
`structs` block for the commented `sptpc2d:dnnsp` block.

```bash
# MC (as committed): simtpc2d:dnnsp
lar -n 10 -c wcls-frame-dump.fcl -s 2025f-mc.root --no-output
wirecell-plot frame -t dnnsp -o dnn-frames.pdf dnn-frames.tar.bz2

# data: uncomment the sptpc2d structs block in wcls-frame-dump.fcl, then
lar -n 10 -c wcls-frame-dump.fcl -s 2025f-data.root --no-output
```

## standalone files

mc
```bash
# Part A: dump image clusters
lar -n 1 -c wcls-img-dump.fcl -s 2025f-mc.root --no-output
# Part B: dump opflash data
lar -n 1 -c wcls-flash-dump.fcl -s 2025f-mc.root --no-output

data
```bash
# Part A: dump image clusters: icluster*.npz
lar -n 10 -c wcls-img-dump.fcl -s 2025f-data.root --no-output
# Part B: dump opflash data: opflash*.tar.gz
lar -n 10 -c wcls-flash-dump.fcl -s 2025f-data.root --no-output
```

## matching:

refactored lar-matching:
```bash
# clustering + QL matching + all-APA clustering (needs LArSoft env)
lar -n 1 -c wct-clus-matching.fcl --no-output
```

standalone wct-matching:
```bash
wire-cell -l stdout -L info \
-V reality=sim \
-V DL=6.2 -V DT=9.8 -V lifetime=6 -V driftSpeed=1.565 \
-V input=input-10evt-data \
-V semimodel_file=sbnd/photodet/semi-analytical-sbnd.json \
-c wct-clus-matching-standalone.jsonnet
```

## upload to bee:


## Re-sim
```bash
lar -n 1 -c standard_detsim_sbnd-dump.fcl -s 2025f-mc.root -o 2025f-mc-resim.root
```


## run Xin's
  cd /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/standalone-sample

  # jsonnet + semimodel search paths (semimodel is referenced by bare name)
  path-prepend /exp/sbnd/app/users/yuhw/wire-cell-toolkit/cfg               WIRECELL_PATH
  path-prepend /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/sbnd_xin       WIRECELL_PATH
  path-prepend /exp/sbnd/app/users/yuhw/wire-cell-data/sbnd/photodet       WIRECELL_PATH

  # 1) image the active clusters from the SP-frames bundle -> icluster-apa{0,1}-active.npz
  wire-cell -l stderr -l wct-img.log:debug -L debug \
      --tla-str  "input=2025f-mc-sp-frames.tar.bz2" \
      --tla-code "anode_indices=[0,1]" \
      --tla-str  "output_dir=." \
      -c ../sbnd_xin/wct-img-all.jsonnet

  # 2) the standalone clustering+matching graph
  wire-cell -l stderr -l wct-clus-matching.log:debug -L debug \
      -V reality=sim \
      -V input=. \
      -V frames=2025f-mc-sp-frames.tar.bz2 \
      -V semimodel_file=semi-analytical-sbnd.json \
      -C DL=6.2 -C DT=9.8 -C lifetime=6 \
      -C joint=true -C pmt_nl=true \
      -c ../sbnd_xin/wct-clus-matching-standalone.jsonnet

BROWSER=echo bash ../sbnd_xin/upload-to-bee.sh mabc.zip
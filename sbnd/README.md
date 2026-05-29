# SBND Test

## Setup

```bash
source setup-local-opt.sh
```

## Run

```bash
rm -rf data-sep
time lar --nskip 0 -n 1 -c wcls-img-clus-matching.fcl -s standalone-sample/2025f-mc.root --no-output >& wcls-img-clus-matching.log
```

## Upload to Bee

```bash
./bee-upload.sh
```

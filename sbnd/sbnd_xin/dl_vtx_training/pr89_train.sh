#!/bin/bash
# doc pr/89 Arm B -- launch one training config, 6 folds in parallel.
# Usage: ./pr89_train.sh <name> [extra train.py flags...]
#
# The base recipe is pr/81-B hr3 VERBATIM (runs/hr3/config.json) on the
# sealed pr89_pool: freeze none, lr0 1e-5, bn-freeze, min-cloud 16, clip
# 5.0, cands + cand-softmax 1.0, scale-anchor 1.0, dense-weight 0.1, 18
# epochs, 6-fold, seed 20260814, CPU (hr3 trained on CPU; these clouds are
# small enough that GPU launch overhead loses).  Extra flags append, so
#   ./pr89_train.sh hr4                                   # B0 dose replay
#   ./pr89_train.sh hr4-hum  --source-weight ai:0.0       # 700-human twin
#   ./pr89_train.sh hr4-maxa --scale-anchor 0.0 --max-anchor 1.0
set -u
NAME=$1; shift
cd "$(dirname "$0")"
mkdir -p runs
for k in 0 1 2 3 4 5; do
  OMP_NUM_THREADS=1 python3 train.py \
    --data data/pr89_pool --cands data/pr89_pool-cands --name "$NAME" \
    --fold $k --kfold 6 --epochs 18 --lr0 1e-5 --freeze none \
    --bn-freeze --min-cloud 16 --clip 5.0 \
    --cand-softmax 1.0 --scale-anchor 1.0 --dense-weight 0.1 \
    --device cpu "$@" \
    > "runs/${NAME}-f$k.log" 2>&1 &
done
wait
echo "== $NAME folds done =="
for k in 0 1 2 3 4 5; do
  tail -2 "runs/${NAME}-f$k.log" | head -1
done

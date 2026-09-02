#!/bin/bash
# doc 94 -- the whole round-1 measurement, in one command, over the completed
# 3067-event probe arms.  Read-only.
set -u
cd -P "$(dirname "$0")/.." || exit 1
O=products/doc94; mkdir -p $O
ARMS="--arm work-mcp1k-d94probe --arm work-mcp2k-d94probe --arm work-ncpi0-d94probe --arm work-nuecc48-d94probe"
echo "############ 1. descent feature (cos_y) + owner baseline ############"
python3 scripts/doc94_descent_census.py $ARMS --baseline --cut -0.25 --out $O/descent-census.tsv
echo
echo "############ 2. other-track prong feature ############"
python3 scripts/doc94_prong_census.py $ARMS --out $O/prong-census.tsv
echo
echo "############ 3. proton_muon_guard fire census ############"
python3 scripts/doc94_pmguard_census.py $ARMS --out $O/pmguard-census.tsv
echo
echo "############ 4. verdict identity: probe arm vs prod0901b ############"
python3 scripts/doc94_identity_check.py

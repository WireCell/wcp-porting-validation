#!/bin/bash
# doc pdvd/47 -- run d47_sim_transverse_profile.py over every (detector, arm, tag) of the xtrack study
# (outputs under /home/xqian/tmp/xtrack/<det>/ana/, one "<det> <label> rc=<n>" line per job in ana_rc.txt).
# run the estimator over (det, arm, tag) and append one summary line per (est=share, plane joint)
S=$(dirname "$0")/d47_sim_transverse_profile.py
run() {  # det truth archive tag label extra
    local det=$1 truth=$2 arc=$3 tag=$4 label=$5; shift 5
    local O=/home/xqian/tmp/xtrack/$det/ana; mkdir -p $O
    python3 $S --det $det --truth $truth --frames $arc --tag $tag --nboot 100 "$@" --out $O/$label > $O/$label.log 2>&1
    echo "$det $label rc=$?" >> /home/xqian/tmp/xtrack/ana_rc.txt
}
export -f run; export S
X=/home/xqian/tmp/xtrack
{
for det in pdhd pdvd sbnd; do
  case $det in pdhd) a=1;; *) a=0;; esac
  T=$X/$det/truth_${det}_a$a.json
  echo "run $det $T $X/$det/S0-anode$a-splat.tar.bz2 auto S0_splat --kernel"
  echo "run $det $T $X/$det/S0n5-anode$a-splat.tar.bz2 auto S0n5_splat"
  for arm in S1 S2 S3 S5 nsig5; do
    for tag in gauss wiener rawdecon; do
      k=""; [ "$arm" = S3 ] && k="--kernel"
      echo "run $det $T $X/$det/$arm-anode$a-sp.tar.bz2 $tag ${arm}_$tag $k"
    done
    echo "run $det $T $X/$det/$arm-anode$a-raw.tar.bz2 raw ${arm}_raw"
  done
done
echo "run pdhd $X/pdhd/truth_pdhd_a1.json $X/pdhd/S6a-anode1-sp.tar.bz2 gauss S6a_gauss"
echo "run pdhd $X/pdhd/truth_pdhd_a0.json $X/pdhd/S6b-anode0-sp.tar.bz2 gauss S6b_gauss"
echo "run pdhd $X/pdhd/truth_pdhd_a0.json $X/pdhd/S6b-anode0-sp.tar.bz2 rawdecon S6b_rawdecon"
echo "run pdvd $X/pdvd/truth_pdvd_a4.json $X/pdvd/top-anode4-sp.tar.bz2 gauss top_gauss"
echo "run pdvd $X/pdvd/truth_pdvd_a4.json $X/pdvd/top-anode4-sp.tar.bz2 rawdecon top_rawdecon"
} > $X/ana_jobs.txt
rm -f $X/ana_rc.txt; wc -l $X/ana_jobs.txt
cat $X/ana_jobs.txt | xargs -P 12 -I{} bash -c "{}"
echo ALLDONE >> $X/ana_rc.txt

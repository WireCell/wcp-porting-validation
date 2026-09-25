#!/bin/bash
# doc pdvd/119 sec 8 private A/B build: arm A3 = clean 1e6b2905, arm B3 = 1e6b2905 + op_cluster_anodes patch.
L=/home/xqian/tmp/d119s_build.log
export ROOTSYS=/wcwc/opt/builtin/linux-debian12-x86_64/gcc-12.2.0/root-6.32.02-5umcjrnuqhwwinuhofro6w4ofwb3e2yk
cd /home/xqian/tmp/d119wt || exit 1
step() { echo "== $* ($(date +%T))" >> $L; "$@" >> $L 2>&1; local rc=$?; echo "rc=$rc :: $*" >> $L; [ $rc -eq 0 ] || { echo "ABORT" >> $L; exit $rc; }; }
: > $L
step git checkout -- util/inc/WireCellUtil/Bee.h util/src/Bee.cxx clus/inc/WireCellClus/MultiAlgBlobClustering.h clus/src/MultiAlgBlobClustering.cxx
step rm -f util/test/doctest_bee_flashes_beam.cxx
step git checkout --detach 1e6b2905
step git status --short
step ./wcb build --notests -p -j 24
step ./wcb install --notests -p -j 24
rm -rf /home/xqian/tmp/d119inst/A3
step cp -a /home/xqian/tmp/d119inst/live /home/xqian/tmp/d119inst/A3
step git apply /home/xqian/tmp/d119s_toolkit.patch
step mv util/test/doctest_bee_flashes_beam.cxx /home/xqian/tmp/d119s_doctest_hold.cxx
step ./wcb build --notests -p -j 24
step ./wcb install --notests -p -j 24
rm -rf /home/xqian/tmp/d119inst/B3
step cp -a /home/xqian/tmp/d119inst/live /home/xqian/tmp/d119inst/B3
step mv /home/xqian/tmp/d119s_doctest_hold.cxx util/test/doctest_bee_flashes_beam.cxx
step ./wcb build -p -j 24 --targets=wcdoctest-clus
echo "ALL DONE" >> $L

#!/bin/bash
# doc qlmatch/32 -- content hashes (abtest/hash_archive.py) of every opflash_pdvd-wct.tar.gz of one light arm, in the
# d31/gate/hashes_<arm>.txt format: "<run6>_light<evt> <sha256> <nmembers>".
#   d32_gate_hash.sh <arm-suffix-without-underscore> > ../d32/gate/hashes_<arm>.txt
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
arm=$1
grep -v '^#' $PDVD/stm/events.txt | awk 'NF>=3 {printf "%06d %s\n", $1, $3}' | while read r6 evt; do
    f=$PDVD/work/${r6}_light${evt}_$arm/opflash_pdvd-wct.tar.gz
    if [ -s "$f" ]; then
        set -- $(python3 $PDVD/../abtest/hash_archive.py "$f")
        echo "${r6}_light${evt} $1 $2"
    else
        echo "${r6}_light${evt} MISSING 0"
    fi
done

#!/usr/bin/env python3
"""doc qlmatch/35 amendment 1 -- the blind scan record, with V2 read against the truth the calibration was drawn from.

d103_scan_record.py (imported, untouched) builds the record and the V2 calibration check, but reads its "existing
record" for V2 from the doc 103 PDVD lineage record only.  The doc 35 calibration items were drawn (d35_scan_items.py)
from the full grade truth -- d116_grade.truth() with own116v and smx116 folded -- so a calibration key need not be in
the older record (039253_6/124 is not).  This swaps old_truth() for that union, (verdict, source) per key, and runs
d103_scan_record.main() unchanged otherwise (same > 25 % stop rule).

    python3 d35_scan_record.py --det pdvd --round /home/xqian/tmp/p35scan/round_pdvd \
        --items /home/xqian/tmp/p35scan/items/pdvd_items.tsv --tag smx35 \
        --record-out pdvd/docs/scan/pdvd_stm_michel_smx35_verdicts.json
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d35_stm_grade as S           # noqa: E402  (puts nf_sp_img_clus/scripts on the path)
import d103_scan_record as R        # noqa: E402


def old_truth(det):
    T, _ = S.G16.truth(det, S.EXTRA, S.OWNER)
    return {k: (v[0], v[2]) for k, v in T.items()}


if __name__ == "__main__":
    R.old_truth = old_truth
    R.main()

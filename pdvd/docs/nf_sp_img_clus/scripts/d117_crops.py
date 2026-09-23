#!/usr/bin/env python3
"""doc pdvd/117 -- rebuild doc 98's latest-arm PDVD Michel crops from regenerated SP frames.

    python3 d117_crops.py --det pdvd --run ../d117/latest --pdvd-arm p98vonq \
        --pdvd-record ../../scan/pdvd_stm_michel_p98vonq_carried_sw99_verdicts.json -j 8

p98von's SP frames were removed by a disk-cleanup round.  Production-default SP reproduces them (doc pdvd/100 F2), and
run_sp.sh wrote that SP to work/<evt>_d117sp; d117_cell_identity.py checks those frames against the cell charges
p98vonq's PR tree recorded.  This wrapper runs d98_michel_crops.py unchanged except that the SP directory is the _d117sp
one; everything else (selection, record, labels, masks, cut) is doc 98's code.
"""
import os, sys
import d98_michel_crops as C

_orig = C.sp_dir


def sp_dir(det, evt):
    d = f"{C.IMG}/{det}/work/{evt}_d117sp"
    return d if os.path.isdir(d) else None


C.sp_dir = sp_dir

if __name__ == "__main__":
    sys.exit(C.main())

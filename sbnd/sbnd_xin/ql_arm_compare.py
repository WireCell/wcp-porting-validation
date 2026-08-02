#!/usr/bin/env python3
"""Compare QL-stage products (mabc-all-apa.zip + pctree tarball) of two arms.

usage: ql_arm_compare.py <armA> <armB> <evt> [<evt> ...]
Arms are sbnd_xin work-* roots holding ql_evt<ID>/.
"""
import sys, subprocess, os

HASH = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest/hash_archive.py"

def h(path):
    if not os.path.exists(path):
        return "MISSING " + path
    out = subprocess.run([sys.executable, HASH, path], capture_output=True, text=True)
    # output lines are "<content-hash> <nmembers> <abs path>"; drop the path
    return [line.split()[0:2] for line in out.stdout.strip().splitlines()]

armA, armB = sys.argv[1], sys.argv[2]
nz = np = 0
evts = sys.argv[3:]
for evt in evts:
    da, db = f"{armA}/ql_evt{evt}", f"{armB}/ql_evt{evt}"
    zi = h(f"{da}/mabc-all-apa.zip") == h(f"{db}/mabc-all-apa.zip")
    pi = h(f"{da}/pctree-evt{evt}.tar.gz") == h(f"{db}/pctree-evt{evt}.tar.gz")
    nz += zi; np += pi
    print(f"{'OK ' if zi and pi else 'DIFF'} evt {evt}  zip={'=' if zi else '≠'} pctree={'=' if pi else '≠'}")
print(f"SUMMARY {len(evts)} events | zip identical {nz}/{len(evts)} | pctree {np}/{len(evts)}")

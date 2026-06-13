#!/usr/bin/env python
import sys, os

# Usage: python wct-img-2-bee.py <run> <subrun> <event> <idx0>:<path0> [<idx1>:<path1> ...]
# Each pair specifies an anode index (0-7) and the corresponding
# clusters-apa-anodeN-ms-active.tar.gz path.  Anodes 0-3 use bottom-drift
# speed/x0; anodes 4-7 use top-drift.
#
# By default the drift-side anodes are merged into two Bee instances
# (anodes 0-3 -> group0123, anodes 4-7 -> group4567).  Set PDVD_BEE_GROUP=0 for
# the legacy one-instance-per-anode output.

# Drift-side display groups: (name, geo-representative idx, member anode idxs).
GROUPS = [("imaging-group0123", 0, [0, 1, 2, 3]),
          ("imaging-group4567", 4, [4, 5, 6, 7])]

def anode_args(idx):
    if idx <= 3:
        return '--speed "-1.57*mm/us" --t0 "0*us" --x0 "-341.5*cm"'
    else:
        return '--speed "1.57*mm/us" --t0 "0*us" --x0 "341.5*cm"'

def run_bee_blobs(rse, density, geo_idx, out_name, files):
    cmd = ('wirecell-img bee-blobs -g protodunevd -s uniform -d %f %s %s'
           ' -o data/0/0-%s.json %s'
           % (density, rse, anode_args(geo_idx), out_name, " ".join(files)))
    print(cmd)
    os.system(cmd)

def main(run, subrun, event, pairs):
    if os.path.exists('data/0'):
        print('found old data, removing ...')
        os.system('rm -rf data')
    if os.path.exists('upload.zip'):
        os.system('rm -f upload.zip')
    os.system('mkdir -p data/0')

    density = 1
    rse = '--rse %s %s %s' % (run, subrun, event)
    pmap = dict(pairs)  # idx -> filepath

    if os.environ.get("PDVD_BEE_GROUP", "1") == "0":
        for idx, fp in pairs:
            run_bee_blobs(rse, density, idx, "apa%d" % idx, [fp])
    else:
        for gname, geo_idx, idxs in GROUPS:
            files = [pmap[i] for i in idxs if i in pmap]
            if files:
                run_bee_blobs(rse, density, geo_idx, gname, files)

    os.system('zip -r upload data')

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("usage: python wct-img-2-bee.py <run> <subrun> <event> <idx0>:<path0> [<idx1>:<path1> ...]")
        sys.exit(1)
    run, subrun, event = sys.argv[1], sys.argv[2], sys.argv[3]
    pairs = []
    for arg in sys.argv[4:]:
        idx_str, fp = arg.split(':', 1)
        pairs.append((int(idx_str), fp))
    main(run, subrun, event, pairs)

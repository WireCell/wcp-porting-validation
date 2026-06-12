#!/usr/bin/env python
import sys, os

# Usage: python wct-img-2-bee.py <run> <subrun> <event> <idx0>:<path0> [<idx1>:<path1> ...]
# Each pair specifies an APA index (0-3) and the corresponding
# clusters-apa-apa<N>-ms-active.tar.gz path.  Even APAs (0, 2) drift in the
# -x direction; odd APAs (1, 3) drift in the +x direction.
#
# By default the drift-side pairs are merged into two Bee instances
# (APA0+APA2 -> group02, APA1+APA3 -> group13).  Set PDHD_BEE_GROUP=0 for the
# legacy one-instance-per-APA output.

# Drift-side display groups: (name, geo-representative idx, member APA idxs).
GROUPS = [("imaging-group02", 0, [0, 2]), ("imaging-group13", 1, [1, 3])]

def anode_args(idx):
    if idx % 2 == 0:
        return '--speed "-1.6*mm/us" --t0 "0*us" --x0 "-358*cm"'
    else:
        return '--speed "1.6*mm/us" --t0 "0*us" --x0 "358*cm"'

def run_bee_blobs(rse, density, geo_idx, out_name, files):
    cmd = ('wirecell-img bee-blobs -g protodunehd -s uniform -d %f %s %s'
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

    if os.environ.get("PDHD_BEE_GROUP", "1") == "0":
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

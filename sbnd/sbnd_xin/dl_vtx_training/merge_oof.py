#!/usr/bin/env python3
'''doc pr/78 -- merge per-fold oof_fold<k>.tsv (from fold-parallel train.py
--fold launches) into the single oof_d_argmax.tsv the eval tools expect.'''
import glob
import os
import sys

run_dir = sys.argv[1]
oof = {}
for path in sorted(glob.glob(os.path.join(run_dir, 'oof_fold*.tsv'))):
    with open(path) as fh:
        next(fh)
        for line in fh:
            evt, d = line.split()
            oof[int(evt)] = d
out = os.path.join(run_dir, 'oof_d_argmax.tsv')
with open(out, 'w') as fh:
    fh.write('evt\td_argmax_cm\n')
    for evt in sorted(oof):
        fh.write('%d\t%s\n' % (evt, oof[evt]))
print('merged %d events -> %s' % (len(oof), out))

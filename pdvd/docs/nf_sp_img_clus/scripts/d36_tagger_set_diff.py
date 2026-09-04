#!/usr/bin/env python3
"""Per-event COSMIC-TAGGER SET difference between two PDVD PR arms.

A count-only census reads flat when the same number of clusters is tagged but
they are different clusters, and the per-bundle PR that follows runs on the
tagged set, not on its size.  This reports, per event, the symmetric difference
of the STM-tagged and TGM-tagged cluster id sets.

  STM  from "TaggerCheckSTM: cluster N -> STM=1 ..."   (the verdict line; the
       same verdict is echoed twice more by ClusteringProtectBundle, so anchor
       on the line that carries the id)
  TGM  from "TaggerCheckTGM: cluster N -> TGM=true"

Usage:
  cd <repo>/pdvd
  python3 docs/nf_sp_img_clus/scripts/d36_tagger_set_diff.py d36off d36on
"""
import glob, os, re, sys
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))          # <repo>/pdvd
W = os.path.join(PDVD, 'work')
STM = re.compile(r"TaggerCheckSTM: cluster (\d+) . STM=1 ")
TGM = re.compile(r"TaggerCheckTGM: cluster (\d+) . TGM=true")
A, B = sys.argv[1], sys.argv[2]

def sets(tag, base):
    logs = glob.glob(os.path.join(W, base + '_' + tag, 'wct_pr_*.log'))
    done = glob.glob(os.path.join(W, base + '_' + tag, 'pr_resource_*.txt'))
    if not logs or not done:          # a log exists from the moment a job starts
        return None                   # -- completion is the resource file
    text = open(logs[0], errors='replace').read()
    return (set(int(x) for x in STM.findall(text)),
            set(int(x) for x in TGM.findall(text)))

bases = sorted({os.path.basename(d)[:-(len(A)+1)] for d in glob.glob(os.path.join(W, '*_'+A))})
rows = []
for base in bases:
    sa, sb = sets(A, base), sets(B, base)
    if sa is None or sb is None:
        continue
    rows.append((base, len(sa[0] ^ sb[0]), len(sa[1] ^ sb[1]),
                 len(sa[0]), len(sb[0]), len(sa[1]), len(sb[1])))
n = len(rows)
for k, name in ((1, 'STM'), (2, 'TGM')):
    d = [r for r in rows if r[k] > 0]
    tot = sum(r[k] for r in rows)
    print('%s set differs on %3d / %3d events (%4.1f %%); mean symmetric difference %.2f clusters'
          % (name, len(d), n, 100.0*len(d)/n, float(tot)/n))
print('%s totals: STM %d -> %d, TGM %d -> %d (distinct tagged clusters over %d events)'
      % (A + ' -> ' + B, sum(r[3] for r in rows), sum(r[4] for r in rows),
         sum(r[5] for r in rows), sum(r[6] for r in rows), n))
print('\nworst events by STM symmetric difference:')
for r in sorted(rows, key=lambda r: -r[1])[:8]:
    print('   %-12s STM %2d changed (%2d -> %2d)   TGM %2d changed (%2d -> %2d)'
          % (r[0], r[1], r[3], r[4], r[2], r[5], r[6]))

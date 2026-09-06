#!/usr/bin/env python3
"""pdhd/docs/01_steiner-wrapped-planes.md sec 5: the per-arm log counters of the
stm-tagger-chain.md sec 12.5 table (a tracked replacement for the scratch
harvest_scan.py that produced it), plus the STM-tagged SET per event so two
arms can be compared the way sec 13.1 did (on the (event, cluster) key, not on
the count -- feedback: count the tags, miss the objects).

Counters per arm, summed over its logs (strings are the log's own):
  fewterm    'steiner terminal(s) found'              -- create_steiner_tree bailed (<2 terminals)
  nosteiner  'no steiner_pc'                          -- STM fit skipped for want of a Steiner cloud
  thin_calls / term_in / term_out   'steiner_thin: nterm_in=A nterm_out=B'
  keep       'TaggerCheckSTM: cluster N -> STM=1'     -- STM tags kept
  n_verdicts 'TaggerCheckSTM: cluster'                -- all STM verdicts

Usage (from wcp-porting-img/pdhd):
  python3 docs/scripts/d01_harvest_counters.py stm0 stmw stmwc [--run 029107]
  python3 docs/scripts/d01_harvest_counters.py stm0 stmwc --churn   # STM set diff, first vs each other arm
"""
import argparse
import glob
import re
from collections import Counter

THIN_RE = re.compile(r'steiner_thin: nterm_in=(\d+) nterm_out=(\d+)')
STM_RE = re.compile(r'TaggerCheckSTM: cluster (\d+) .*STM=(\d)')


def harvest(tag, run):
    c = Counter()
    tags = {}   # event -> set of STM-tagged cluster ids
    files = sorted(glob.glob(f'work/{run}_*_{tag}/wct_pr_{run}_*.log'))
    for f in files:
        evt = int(f.split(f'{run}_')[1].split('_')[0])
        tagged = set()
        with open(f, errors='replace') as fh:
            for line in fh:
                if 'steiner terminal(s) found' in line:
                    c['fewterm'] += 1
                elif 'no steiner_pc' in line:
                    c['nosteiner'] += 1
                elif 'steiner_thin: nterm_in=' in line:
                    m = THIN_RE.search(line)
                    c['thin_calls'] += 1
                    c['term_in'] += int(m.group(1))
                    c['term_out'] += int(m.group(2))
                elif 'TaggerCheckSTM: cluster' in line:
                    m = STM_RE.search(line)
                    if not m:
                        continue
                    c['n_verdicts'] += 1
                    if m.group(2) == '1':
                        c['keep'] += 1
                        tagged.add(int(m.group(1)))
        tags[evt] = tagged
    c['events'] = len(files)
    return c, tags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--run', default='029107')
    ap.add_argument('--churn', action='store_true')
    a = ap.parse_args()
    cols = ['events', 'fewterm', 'nosteiner', 'thin_calls', 'term_in', 'term_out', 'keep', 'n_verdicts']
    print('arm\t' + '\t'.join(cols))
    res = {}
    for tag in a.tags:
        c, tags = harvest(tag, a.run)
        res[tag] = (c, tags)
        print(tag + '\t' + '\t'.join(str(c[k]) for k in cols))
    if a.churn and len(a.tags) > 1:
        base = a.tags[0]
        for tag in a.tags[1:]:
            t0, t1 = res[base][1], res[tag][1]
            evts = sorted(set(t0) & set(t1))
            gain = sum(len(t1[e] - t0[e]) for e in evts)
            lose = sum(len(t0[e] - t1[e]) for e in evts)
            both = sum(len(t0[e] & t1[e]) for e in evts)
            moved = sum(1 for e in evts if t0[e] != t1[e])
            print(f'churn {base} -> {tag}: events {len(evts)}, events whose STM set differs {moved}, '
                  f'tags kept by both {both}, gained {gain}, lost {lose}')


if __name__ == '__main__':
    main()

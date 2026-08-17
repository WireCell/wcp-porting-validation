#!/usr/bin/env python3
"""docs/73 round 2: did the missing half actually JOIN the in-beam main?

PR-FREE BY CONSTRUCTION.  The defect happens *at* Q/L matching and PR only ever
sees what Q/L passed, so everything needed is in one Q/L event dir:

    ql_evt<ID>/mabc-all-apa.zip   0-img-global.json        RAW x, img cluster ids
                                  0-clustering-global.json the final all-APA clusters
    ql_evt<ID>/mabc-apa{0,1}-face0.zip                     img cid -> APA

No nusel-table.tsv, no PR chain (owner, docs/73 round 2: "evaluate based on QL
match result, no need to run PR chain").

METHOD.  doc 72 sec A already identified, per signal event, the in-beam main
(`main_id`, a clustering-global id in the BASELINE arm) and the partner img
cluster (`p_cid`) that continues it across the cathode.  Per arm:

  1. q-join img-global <-> clustering-global.  Join on `q` ALONE: the T0
     correction also carries a transverse shift whose sign FLIPS between the
     TPCs, so a (y,z,q) key returns zero matches (doc 72).  q collides for a few
     % of points; those drop out and the majority vote absorbs it.
  2. In the BASELINE arm, resolve main_id -> the set of img cids feeding it.
  3. In the TEST arm, find the clustering-global cluster holding those img cids
     (majority) and the one holding p_cid, and ask whether they are the SAME.
  4. Cross-check against the component's own log: a `rescue round` line naming a
     destination gid whose t0 is inside the beam window is what makes the merge a
     FIX rather than merely a merge -- the a/b/c/d direction rule can hand the
     joined crosser to the COSMIC bundle, which is worse for PR than leaving it
     split (docs/73 F4).

Verdicts: JOINED-BEAM (the fix), JOINED-COSMIC (merged into the wrong bundle),
NOT-JOINED (unchanged).

Repro:
  cd .../sbnd/sbnd_xin
  python3 scripts/analysis/cathode/cbr_join_metric.py \
      --baseline work-mcp1k-cb0805 --baseline work-mcp2k-cb0816 \
      --test work-cbr2-on2 \
      --rows products/beam-cathode-split-mcp1k.tsv \
      --rows products/beam-cathode-split-mcp2k.tsv
"""
import argparse
import collections
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cathode_halfmatch import load_event   # noqa: E402

BEAM_LO, BEAM_HI = 0.2, 2.2      # us, SBND production beam window


def qmap(ql_dir):
    """(img cid -> clustering-global cid) by q-join, plus the img point counts."""
    got = load_event(ql_dir)
    if got is None:
        return None
    img, clus, _op, _cid2apa = got
    key = {}
    for i, q in enumerate(np.round(img['q'], 4).tolist()):
        key[q] = -1 if q in key else i
    icid = np.array(img['cluster_id'], int)
    vote = collections.defaultdict(collections.Counter)   # clus cid -> img cids
    rvote = collections.defaultdict(collections.Counter)  # img cid -> clus cids
    for q, c in zip(np.round(clus['q'], 4).tolist(), clus['cluster_id']):
        i = key.get(q, -1)
        if i >= 0:
            vote[int(c)][int(icid[i])] += 1
            rvote[int(icid[i])][int(c)] += 1
    return {c: v.most_common(1)[0][0] for c, v in rvote.items()}, \
           {c: set(v) for c, v in vote.items()}


def moves(root, evt):
    """The rescue's own decisions for this event: (dest_gid, dest_t0_us, rule)."""
    log = os.path.join(root, 'ql_evt%d' % evt, 'wct_ql_evt%d.log' % evt)
    out = []
    if not os.path.exists(log):
        return out
    for line in open(log, errors='replace'):
        if 'rescue round' not in line:
            continue
        m = re.search(r'\+ c(\d+) \(gid (\d+), t0 ([-\d.]+) us.*-> gid (\d+) \((\S+?);', line)
        # destination t0 is whichever of the two halves owns the destination gid
        t0s = [float(x) for x in re.findall(r't0 ([-\d.]+) us', line)]
        g = re.findall(r'gid (\d+)', line)
        if not m or len(t0s) < 2 or len(g) < 3:
            # pass 2 ("unmatched rescue round") names one gid and one t0: the
            # orphan has neither, and the destination is always the beam half.
            if 'unmatched rescue round' in line and t0s:
                out.append((int(g[0]) if g else None, t0s[0], 'unmatched-adopt'))
            else:
                out.append((None, None, 'unparsed'))
            continue
        dest = int(g[2])
        dt0 = t0s[0] if int(g[0]) == dest else t0s[1]
        out.append((dest, dt0, m.group(5)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, action='append')
    ap.add_argument('--test', required=True)
    ap.add_argument('--rows', required=True, action='append')
    ap.add_argument('--kind', default='MISSING')
    a = ap.parse_args()

    rows = []
    for f in a.rows:
        rows += [r for r in csv.DictReader(open(f), delimiter='\t') if r['kind'] == a.kind]
    # one row per event: the longest missing piece
    best = {}
    for r in sorted(rows, key=lambda r: -float(r['p_ext'])):
        best.setdefault(int(r['event']), r)

    def find(root_list, evt):
        for w in root_list:
            d = os.path.join(w, 'ql_evt%d' % evt)
            if os.path.isdir(d):
                return d
        return None

    print('%-9s %-6s %-6s %-16s %-9s %s' %
          ('event', 'missing', 'fired', 'verdict', 'dest_t0', 'rule'))
    tally = collections.Counter()
    for evt in sorted(best, key=lambda e: -float(best[e]['p_ext'])):
        r = best[evt]
        bdir = find(a.baseline, evt)
        tdir = find([a.test], evt)
        if bdir is None or tdir is None:
            print('%-9d %-6s  -- arm missing --' % (evt, r['p_ext']))
            continue
        bi2c, bc2i = qmap(bdir)
        ti2c, tc2i = qmap(tdir)
        kimg = bc2i.get(int(r['main_id']), set())
        pcid = int(r['p_cid'])
        # where do the main's img cids and the partner's land in the TEST arm?
        kdest = collections.Counter(ti2c[c] for c in kimg if c in ti2c)
        pdest = ti2c.get(pcid)
        joined = bool(kdest) and pdest is not None and pdest in kdest

        mv = moves(a.test, evt)
        fired = len(mv)
        if joined:
            beam = any(t is not None and BEAM_LO <= t < BEAM_HI for _g, t, _r in mv)
            verdict = 'JOINED-BEAM' if (beam or not mv) else 'JOINED-COSMIC'
        else:
            verdict = 'NOT-JOINED'
        tally[verdict] += 1
        dt0 = ('%.3f' % mv[0][1]) if mv and mv[0][1] is not None else '-'
        print('%-9d %-6.0f %-6d %-16s %-9s %s' %
              (evt, float(r['p_ext']), fired, verdict, dt0,
               ','.join(m[2] for m in mv) or '-'))
    print('\n%s' % dict(tally))


if __name__ == '__main__':
    main()

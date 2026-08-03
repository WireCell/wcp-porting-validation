#!/usr/bin/env python3
"""Parse ClusteringCathodeConnect's own per-pair tracers (CC_FEATURE_DUMP +
CATHODE_CONNECT_DEBUG) out of a run_full1k_nusel.sh arm and evaluate candidate
relaxations OFFLINE -- the "analytical accept-log" method used by earlier
cathode_connect rounds.  No binary is rebuilt: every number below is what the
SHIPPED binary computed for that pair; only the accept/reject predicate is
re-evaluated here.

Emission order per candidate pair inside is_cathode_crossing_pair():
    [cc]   (only when dis<10cm and min(len)>=10cm)     -- idents, t0s
    [feat] (every cross-APA pair with dis<max_dis)     -- full feature vector
    [cc]   CLOSE both-long ...                          -- only in that branch
    [ccx]  <reason> ... -> ACCEPT|reject                -- the actual verdict
so a [feat] line and the NEXT [ccx] line describe the same pair.
"""
import sys, os, re, glob, collections

FEAT = re.compile(
    r'\[feat\] dis=(?P<dis>[-\d.]+) dX=(?P<dX>[-\d.]+) tip1=(?P<tip1>[-\d.]+) '
    r'tip2=(?P<tip2>[-\d.]+) ttH=(?P<ttH>[-\d.]+) ttP=(?P<ttP>[-\d.]+) '
    r'ccH=(?P<ccH>[-\d.]+) ccP=(?P<ccP>[-\d.]+) len1=(?P<len1>[-\d.]+) '
    r'len2=(?P<len2>[-\d.]+) p1=(?P<p1>[-\d.,]+) p2=(?P<p2>[-\d.,]+) apa=(?P<apa>\d+/\d+)')
CCX = re.compile(
    r'\[ccx\] c(?P<c1>\d+)<->c(?P<c2>\d+) (?P<why>\S+)\s+dis=(?P<dis>[-\d.]+).*-> (?P<verdict>ACCEPT|reject)')
CC = re.compile(r'\[cc\] c(?P<c1>\d+)<->c(?P<c2>\d+) dis=.*dt0=(?P<dt0>[-\d.]+)us')

# the SBND shipped operating point (cfg/pgrapher/experiment/sbnd/clus.jsonnet:428)
ANGLE_CUT, CONN_FAR, DIS_CUT, SHORT_DIR_LEN = 10.0, 30.0, 5.0, 25.0


def parse(arm):
    rows = []
    for lg in sorted(glob.glob(os.path.join(arm, '.log_e*.log'))):
        evt = None
        pend = None
        for line in open(lg, errors='replace'):
            if evt is None and 'rse=(' in line:
                evt = line.split(',')[-1].split(')')[0].strip()
            m = FEAT.search(line)
            if m:
                pend = {k: v for k, v in m.groupdict().items()}
                continue
            m = CCX.search(line)
            if m and pend is not None:
                r = dict(pend)
                for k in ('dis', 'dX', 'tip1', 'tip2', 'ttH', 'ttP', 'ccH', 'ccP',
                          'len1', 'len2'):
                    r[k] = float(r[k])
                r.update(evt=evt, log=os.path.basename(lg),
                         c1=int(m.group('c1')), c2=int(m.group('c2')),
                         why=m.group('why'), acc=(m.group('verdict') == 'ACCEPT'))
                rows.append(r)
                pend = None
    return rows


def close_regime(r):
    """Does the pair reach the close-regime branches at all?  (hard gates first)"""
    return (r['tip1'] < 5.0 and r['tip2'] < 5.0 and r['dX'] < 8.0
            and r['dis'] < DIS_CUT)


def rule_tip_touch(r, cut, pca_ang=ANGLE_CUT):
    """acc_pca with the cc_pca term dropped because the tips touch."""
    if not close_regime(r):
        return False
    if r['ttH'] < ANGLE_CUT:
        return False                     # already accepted by close_primary
    if min(r['len1'], r['len2']) < SHORT_DIR_LEN:
        return False                     # both-long branch not reached
    if r['dis'] >= cut:
        return False
    return r['ttP'] < pca_ang


def rule_tip_touch_hough(r, cut, ang):
    """acc_hough: tips touch AND the local charge-weighted Hough arms agree."""
    if not close_regime(r):
        return False
    if r['ttH'] < ANGLE_CUT:
        return False
    if min(r['len1'], r['len2']) < SHORT_DIR_LEN:
        return False
    return r['dis'] < cut and r['ttH'] < ang


def main(arm):
    rows = parse(arm)
    nev = len(set(r['log'] for r in rows))
    print(f'arm {arm}')
    print(f'  {len(rows)} cross-APA candidate pairs (dis<25cm) over {nev} event logs')
    acc = [r for r in rows if r['acc']]
    print(f'  currently ACCEPTED: {len(acc)}')
    print('  reject reasons: ' + ', '.join(
        f'{k}={v}' for k, v in collections.Counter(
            r['why'] for r in rows if not r['acc']).most_common()))
    print('  accept reasons: ' + ', '.join(
        f'{k}={v}' for k, v in collections.Counter(
            r['why'] for r in acc).most_common()))

    print('\n  close-regime fall-throughs (the population the relaxations act on):')
    ft = [r for r in rows if not r['acc'] and close_regime(r)
          and min(r['len1'], r['len2']) >= SHORT_DIR_LEN and r['ttH'] >= ANGLE_CUT]
    print(f'    {len(ft)} pairs; of these ttP<10: '
          f'{sum(1 for r in ft if r["ttP"] < ANGLE_CUT)}')
    for r in sorted(ft, key=lambda r: r['dis']):
        print(f'      evt {r["evt"]:>7} c{r["c1"]}<->c{r["c2"]} dis={r["dis"]:5.2f} '
              f'dX={r["dX"]:4.2f} tips={r["tip1"]:4.2f}/{r["tip2"]:4.2f} '
              f'ttH={r["ttH"]:5.1f} ttP={r["ttP"]:5.1f} ccP={r["ccP"]:5.1f} '
              f'len={r["len1"]:6.1f}/{r["len2"]:6.1f}  p1={r["p1"]}')

    print('\n  candidate relaxations (new edges beyond the shipped binary):')
    for cut in (2.0, 3.0, 4.0, 5.0):
        new = [r for r in rows if not r['acc'] and rule_tip_touch(r, cut)]
        print(f'    tip_touch_cut={cut:4.1f}cm (ttP<10, cc dropped): +{len(new)} edges'
              + (('  -> ' + ', '.join(f'evt{r["evt"]}:c{r["c1"]}-c{r["c2"]}@{r["dis"]:.2f}'
                                      for r in new)) if new else ''))
    for cut, ang in ((4.0, 12.0), (4.0, 20.0), (4.0, 30.0)):
        new = [r for r in rows if not r['acc'] and rule_tip_touch_hough(r, cut, ang)]
        print(f'    tip_touch_cut={cut:4.1f} tip_touch_angle_cut={ang:4.1f} (Hough): '
              f'+{len(new)} edges'
              + (('  -> ' + ', '.join(f'evt{r["evt"]}:c{r["c1"]}-c{r["c2"]}@{r["dis"]:.2f}'
                                      for r in new)) if new else ''))
    for pa in (15.0, 20.0):
        new = [r for r in rows if not r['acc'] and rule_tip_touch(r, 4.0, pa)]
        print(f'    tip_touch_cut= 4.0 crosser_pca_angle={pa:4.1f}: +{len(new)} edges'
              + (('  -> ' + ', '.join(f'evt{r["evt"]}:c{r["c1"]}-c{r["c2"]}@{r["dis"]:.2f}'
                                      for r in new)) if new else ''))
    return rows


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-mcp1kall-ccfeat300')

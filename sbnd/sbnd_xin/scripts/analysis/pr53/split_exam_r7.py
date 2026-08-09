#!/usr/bin/env python3
"""doc pr/53 round 7: causal validation of every cluster split introduced by
the "relaxed_strict_img" S5 3D-image OR-kill, same bar as round 6's
split_exam.py -- every split must trace to an edge the new flavor killed
that the old one didn't, with a figure showing a real void.

Unlike round 6 (legacy "relaxed" vs strict "relaxed_strict", two DIFFERENT
census prefixes OC53CENSUS/OC53CENSUS-S), round 7 compares "relaxed_strict"
(off) against "relaxed_strict_img" (on) -- both protect_bundle-only, both
logged under the SAME OC53CENSUS-S prefix. So the "removed" set here is
computed from two OC53CENSUS-S reruns (cenoff = image_check off, cenon =
image_check on) matched by cluster signature, reusing split_exam.py's
parse_blocks() as-is.

Usage:
  split_exam_r7.py <off_arm> <on_arm> <cenoff_arm> <cenon_arm> <ql_root> <outdir> <evt> [...]

e.g.
  python3 scripts/analysis/pr53/split_exam_r7.py \\
      work-pr53r7-off48 work-pr53r7-on48 work-pr53r7-cenoff48 work-pr53r7-cen48 \\
      work-nuecc48-cb0805 pics 269774
"""
import sys, os
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from oc53_probe import Loader, walk_and_score  # noqa: E402
from split_exam import parse_blocks, load_partition, make_plot, SB  # noqa: E402


def removed_edges_r7(evt, cenoff_arm, cenon_arm):
    """(image_check=off-emitted - image_check=on-emitted) edges, both under
    the OC53CENSUS-S prefix (protect_bundle-only by construction either
    way), matched by cluster signature -- same matching convention as round
    6's removed_edges(), but same-prefix on both sides since round 7 does
    not change which flavor NAME protect_bundle logs under."""
    logoff = os.path.join(SB, cenoff_arm, 'pr_evt%s' % evt, 'wct_pr_evt%s.log' % evt)
    logon = os.path.join(SB, cenon_arm, 'pr_evt%s' % evt, 'wct_pr_evt%s.log' % evt)
    boff = parse_blocks(logoff, 'OC53CENSUS-S')
    bon = parse_blocks(logon, 'OC53CENSUS-S')
    out = []
    for bn in bon:
        cands = [b for b in boff if b['sig'] == bn['sig']]
        if not cands:
            print('  WARNING: no off-flavor census block with signature %s' % (bn['sig'],))
            continue
        bo = cands[-1]
        for jk, e in bo['edges'].items():
            if jk in bn['edges']:
                continue
            rec = dict(e)
            rec['jk'] = jk
            rec['sig'] = bo['sig']
            c_on = bn['closest'].get(jk)
            c_off = bo['closest'].get(jk)
            rec['closest'] = c_on or c_off
            rec['flip_closest'] = bool(c_on and c_on['killed'] and c_off and not c_off['killed'])
            out.append(rec)
    return out


def exam_event(off_arm, on_arm, cenoff_arm, cenon_arm, ql_root, outdir, evt):
    Poff, Coff = load_partition(off_arm, evt)
    Pon, Con = load_partition(on_arm, evt)
    removed = removed_edges_r7(evt, cenoff_arm, cenon_arm)

    t_on = cKDTree(Pon)
    dd, jj = t_on.query(Poff, k=1)
    matched = dd < 0.05
    print('[%s] %d/%d OFF points matched in ON arm; %d removed protect_bundle edge(s)'
          % (evt, matched.sum(), len(Poff), len(removed)))

    ld = None
    results = []
    for cid in np.unique(Coff):
        sel = (Coff == cid) & matched
        if sel.sum() < 2:
            continue
        on_cids, counts = np.unique(Con[jj[sel]], return_counts=True)
        material = [(c, n) for c, n in zip(on_cids, counts) if n >= 5]
        if len(material) < 2:
            continue
        material.sort(key=lambda cn: -cn[1])
        main_cid = material[0][0]
        Xmain = Pon[Con == main_cid]
        if ld is None:
            ld = Loader(os.path.join(SB, on_arm, 'pr_evt%s' % evt))
        for frag_cid, npts in material[1:]:
            Xfrag = Pon[Con == frag_cid]
            tf = cKDTree(Xfrag)
            best, best_d = None, 1e9
            for rec in removed:
                c = rec['closest']
                if c is None:
                    continue
                d = min(tf.query(c['p1'])[0], tf.query(c['p2'])[0])
                if d < best_d:
                    best, best_d = rec, d
            if best is not None and best_d < 3.0:
                c = best['closest']
                r = walk_and_score(ld, c['p1'], c['p2'])
                tag = 'FLIP-CLOSEST' if best['flip_closest'] else 'REMOVED-OTHER'
                out = os.path.join(SB, outdir,
                                   '53_r7_split_%s_off%d_frag%d.png' % (evt, cid, frag_cid))
                make_plot(Xmain, Xfrag, c['p1'], c['p2'], r, c,
                          '%s: off-cluster %d -> fragment %d (%d pts) [%s] (round 7 relaxed_strict_img)'
                          % (evt, cid, frag_cid, npts, tag), out)
                results.append((evt, cid, frag_cid, npts, c['dis'], c['nsteps'],
                                c['nb1'][0], tag, out))
                print('  off cid %d -> frag %d (%d pts): removed edge j=%d k=%d dis=%.2fcm '
                      'nsteps=%d nb=%s nb1=%s strong=%s %s'
                      % (cid, frag_cid, npts, best['jk'][0], best['jk'][1], c['dis'],
                         c['nsteps'], c['nb'], c['nb1'], c['strong'], tag))
            else:
                results.append((evt, cid, frag_cid, npts, None, None, None,
                                'NO-CAUSAL-MATCH', None))
                print('  off cid %d -> frag %d (%d pts): NO removed edge within 3cm '
                      '(nearest %.2fcm) -- manual review' % (cid, frag_cid, npts, best_d))
    if ld is not None:
        ld.cleanup()
    if not results:
        print('  no material split found (mover differs some other way -- e.g. '
              'downstream fit shifts only)')
    return results


if __name__ == '__main__':
    off_arm, on_arm, cenoff_arm, cenon_arm, ql_root, outdir = sys.argv[1:7]
    allres = []
    for evt in sys.argv[7:]:
        allres += exam_event(off_arm, on_arm, cenoff_arm, cenon_arm, ql_root, outdir, evt)
    unmatched = [x for x in allres if x[7] == 'NO-CAUSAL-MATCH']
    flips = [x for x in allres if x[7] == 'FLIP-CLOSEST']
    print('\n%d split(s) examined: %d FLIP-CLOSEST, %d REMOVED-OTHER, %d NO-CAUSAL-MATCH'
          % (len(allres), len(flips), len(allres) - len(flips) - len(unmatched), len(unmatched)))

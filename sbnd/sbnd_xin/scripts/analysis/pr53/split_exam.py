#!/usr/bin/env python3
"""doc pr/53 round 6: validate every cluster split introduced by the
"relaxed_strict" protect_bundle graph (S1+S2+S3) on a mover event.

Causal attribution, not nearest-guess: two census-enabled reruns of the same
event (both byte-identical to their arms, proven by hash) give the exact
component-level edge list protect_bundle's graph emitted under the legacy
"relaxed" flavor (OC53CENSUS lines) and under "relaxed_strict" (OC53CENSUS-S
lines).  The removed set (legacy-emitted minus strict-emitted, per cluster,
matched block-by-block -- the component indexing is identical because the
closely-graph input is identical) is the complete causal explanation of every
split.  Each ON-arm fragment is matched to its removed edge(s) by endpoint
proximity, the C++'s own closest-pair counters for that edge are quoted, and
the per-1cm-step per-plane path test is replayed at the exact p1/p2 the C++
tested (oc53_probe.Loader) for the figure.

A fragment with no removed edge within 3cm is flagged NO-CAUSAL-MATCH for
manual review.

Usage:
  split_exam.py <off_arm> <on_arm> <cenoff_arm> <cenon_arm> <ql_root> <outdir> <evt> [...]

e.g.
  python3 scripts/analysis/pr53/split_exam.py \
      work-pr53-off48a work-pr53-on48a work-pr53-cenoff48 work-pr53-cen48 \
      work-nuecc48-cb0805 pics 422851
"""
import sys, os, re, json, zipfile
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from oc53_probe import Loader, walk_and_score

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'

PLANE_STYLE = {
    0: dict(marker='^', color='tab:orange', label='U gap'),
    1: dict(marker='s', color='tab:green', label='V gap'),
    2: dict(marker='D', color='tab:red', label='W gap'),
}


def _res(pfx):
    return dict(
        cluster=re.compile(pfx + r" cluster nblobs=(\d+) npoints=(\d+) ncomp=(\d+)"),
        closest=re.compile(
            pfx + r" closest j=(\d+) k=(\d+) "
            r"p1=\(([-\d.]+),([-\d.]+),([-\d.]+)\) p2=\(([-\d.]+),([-\d.]+),([-\d.]+)\) "
            r"dis=([-\d.]+)cm nsteps=(\d+) strong=(\w+) "
            r"nb=\[(\d+),(\d+),(\d+),(\d+)\] nb1=\[(\d+),(\d+),(\d+),(\d+)\] killed=(\w+)"),
        edge=re.compile(pfx + r" edge j=(\d+) k=(\d+) dis=([-\d.]+)cm "
                        r"mst=(\w+) dir1=(\w+) dir2=(\w+) bridge=(\w+)"))


def parse_blocks(log_path, prefix):
    """Census blocks (one per graph build).  prefix: 'OC53CENSUS' (legacy
    relaxed -- logged by every relaxed build: unmerge_bundle, TaggerCheckTGM,
    ...) or 'OC53CENSUS-S' (relaxed_strict -- protect_bundle only)."""
    if prefix.endswith('-S'):
        R = _res(re.escape(prefix))
    else:
        R = _res(re.escape(prefix) + r"(?!-)")  # OC53CENSUS but not OC53CENSUS-S
    blocks = []
    with open(log_path) as f:
        for line in f:
            m = R['cluster'].search(line)
            if m:
                blocks.append(dict(sig=tuple(int(x) for x in m.groups()),
                                   closest={}, edges={}))
                continue
            if not blocks:
                continue
            m = R['closest'].search(line)
            if m:
                g = m.groups()
                blocks[-1]['closest'][(int(g[0]), int(g[1]))] = dict(
                    p1=np.array([float(g[2]), float(g[3]), float(g[4])]),
                    p2=np.array([float(g[5]), float(g[6]), float(g[7])]),
                    dis=float(g[8]), nsteps=int(g[9]), strong=(g[10] == 'true'),
                    nb=[int(x) for x in g[11:15]], nb1=[int(x) for x in g[15:19]],
                    killed=(g[19] == 'true'))
                continue
            m = R['edge'].search(line)
            if m:
                g = m.groups()
                blocks[-1]['edges'][(int(g[0]), int(g[1]))] = dict(
                    dis=float(g[2]), mst=g[3] == 'true',
                    dir1=g[4] == 'true', dir2=g[5] == 'true', bridge=g[6] == 'true')
    return blocks


def removed_edges(evt, cenoff_arm, cenon_arm):
    """(legacy-emitted - strict-emitted) edges with the strict run's closest
    counters attached, across all protect_bundle clusters of the event.

    The strict blocks (-S prefix) are protect_bundle's by construction.  In
    the legacy run protect_bundle reuses the cached "relaxed" graph built
    earlier on the identical cluster (by TaggerCheckTGM / unmerge stages), so
    the matching legacy block is the LAST plain-census block with the same
    (nblobs, npoints, ncomp) signature."""
    logoff = os.path.join(SB, cenoff_arm, 'pr_evt%s' % evt, 'wct_pr_evt%s.log' % evt)
    logon = os.path.join(SB, cenon_arm, 'pr_evt%s' % evt, 'wct_pr_evt%s.log' % evt)
    boff = parse_blocks(logoff, 'OC53CENSUS')
    bon = parse_blocks(logon, 'OC53CENSUS-S')
    out = []
    for bn in bon:
        cands = [b for b in boff if b['sig'] == bn['sig']]
        if not cands:
            print('  WARNING: no legacy census block with signature %s' % (bn['sig'],))
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


def load_partition(arm, evt):
    z = zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%s' % evt, 'mabc-pr.zip'))
    d = json.loads(z.read('data/0/0-clustering-global.json'))
    P = np.c_[d['x'], d['y'], d['z']]
    C = np.array(d['real_cluster_id'])
    return P, C


def exam_event(off_arm, on_arm, cenoff_arm, cenon_arm, ql_root, outdir, evt):
    Poff, Coff = load_partition(off_arm, evt)
    Pon, Con = load_partition(on_arm, evt)
    removed = removed_edges(evt, cenoff_arm, cenon_arm)

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
            ld = Loader(os.path.join(SB, ql_root, 'ql_evt%s' % evt))
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
                                   '53_r6_%s_off%d_frag%d.png' % (evt, cid, frag_cid))
                make_plot(Xmain, Xfrag, c['p1'], c['p2'], r, c,
                          '%s: off-cluster %d -> fragment %d (%d pts) [%s]'
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


def make_plot(Xmain, Xfrag, p1, p2, r, cen, label, out):
    pts = np.array([p1 + (p2 - p1) / r['nst'] * (i + 1) for i in range(r['nst'])])
    gap = {pi: np.array([s[pi] == 0 and s[pi + 3] == 0 for s in r['rows']]) for pi in range(3)}
    any_gap = gap[0] | gap[1] | gap[2]
    mid = (p1 + p2) / 2
    rad = max(14.0, r['dis'] * 0.6 + 6)
    nm = np.linalg.norm(Xmain - mid, axis=1) < rad
    nf = np.linalg.norm(Xfrag - mid, axis=1) < rad

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, (i0, i1, l0, l1) in zip(axes, [(2, 1, 'z (cm)', 'y (cm)'), (0, 2, 'x (cm)', 'z (cm)')]):
        ax.scatter(Xmain[nm][:, i0], Xmain[nm][:, i1], s=10, c='0.8', label='retained cluster')
        ax.scatter(Xfrag[nf][:, i0], Xfrag[nf][:, i1], s=12, c='tab:blue', label='split fragment')
        ax.plot(pts[:, i0], pts[:, i1], 'k--', lw=1.1, alpha=0.6,
                label='removed edge (%.2fcm, C++ census)' % cen['dis'])
        ax.scatter(pts[~any_gap][:, i0], pts[~any_gap][:, i1], marker='o', s=35,
                   facecolors='none', edgecolors='0.4', linewidths=1.2, label='step: all 3 planes ok')
        for pi, style in PLANE_STYLE.items():
            selp = gap[pi]
            if selp.any():
                ax.scatter(pts[selp][:, i0], pts[selp][:, i1], s=90, linewidths=2.0,
                           facecolors='none', edgecolors=style['color'], marker=style['marker'],
                           label='step: %s (no live, no dead ch)' % style['label'])
        ax.set_xlabel(l0); ax.set_ylabel(l1)
    axes[0].set_title('(y,z) view'); axes[1].set_title('(x,z) view -- drift direction')
    axes[0].legend(fontsize=7, loc='best')
    plt.suptitle('%s\nC++ census: nsteps=%d strong=%s nb=%s nb1=%s -- '
                 'per-plane replay: %d/%d U-gap %d/%d V-gap %d/%d W-gap'
                 % (label, cen['nsteps'], cen['strong'], cen['nb'], cen['nb1'],
                    gap[0].sum(), r['nst'], gap[1].sum(), r['nst'], gap[2].sum(), r['nst']))
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print('  wrote', out)


if __name__ == '__main__':
    off_arm, on_arm, cenoff_arm, cenon_arm, ql_root, outdir = sys.argv[1:7]
    allres = []
    for evt in sys.argv[7:]:
        allres += exam_event(off_arm, on_arm, cenoff_arm, cenon_arm, ql_root, outdir, evt)
    unmatched = [x for x in allres if x[7] == 'NO-CAUSAL-MATCH']
    flips = [x for x in allres if x[7] == 'FLIP-CLOSEST']
    print('\n%d split(s) examined: %d FLIP-CLOSEST, %d REMOVED-OTHER, %d NO-CAUSAL-MATCH'
          % (len(allres), len(flips), len(allres) - len(flips) - len(unmatched), len(unmatched)))

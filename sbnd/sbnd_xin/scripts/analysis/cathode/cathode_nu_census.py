#!/usr/bin/env python3
"""doc pr/12: how the SBND PR chain treats a neutrino candidate that crosses the cathode.

SBND's two drift volumes meet at x = 0; active charge stops at x ~ -+0.45 cm, so a
particle crossing the cathode leaves a ~0.9 cm dead gap in its trajectory, on top of
the ~1.4 cm transverse TPC0/TPC1 misalignment measured in data by doc 14.  Clustering
already joins the two halves (`cathode_connect`, doc 53: 28/28 crossers retained
whole).  This script asks what the *pattern recognition* stage then does with them.

Read-only.  Inputs are what run_pr_chain_batch.sh already writes per event under
<root>/pr_evt<ID>/:

  wct_pr_evt<ID>.log   "TaggerCheckNeutrino: selected main cluster ..." -- emitted iff
                       a candidate was actually chosen (TaggerCheckNeutrino.cxx:345),
                       which is the ONLY reliable nu_evaluated signal (pr_scores_table.py
                       docstring); also carries its t0 / length / n-associated.
  tracking-pr.root     T_rec_charge -- the fitted trajectory points of the candidate,
                       and T_tagger nu_x/nu_y/nu_z.
  mabc-pr.zip          0-clustering-global.json -- the CHARGE the fit was made from,
                       used to tell "the fit split" from "the fit never got there".

Encodings, confirmed in source rather than inferred (doc pr/12 sec 1):

  T_rec_charge.cluster_id        = reco_mother_cluster_id, i.e. the MAIN cluster of the
                                   event, repeated on every row -- not a per-row id
                                   (SbndPrMagnifyTrackingVisitor.cxx:375).
  T_rec_charge.sub_cluster_id    = seg->cluster()->get_cluster_id()*1000 +
    == .real_cluster_id             seg->get_graph_index()  (:378-379, :522) -- the two
                                   branches are the SAME variable.  A change of this
                                   value therefore means A DIFFERENT FITTED SEGMENT.
                                   Vertex rows carry -1 (:484) because the visitor also
                                   appends the PR-graph vertex fit points.
  T_rec_charge.q / .nq           = dQ*dQdx_scale + dQdx_offset  and  dx in cm (:363-364,
                                   :497-498).  Trun holds the scale/offset (0.1/-1000
                                   on SBND), so dQ/dx = (q - offset)/scale / nq.
  vertices-global (mabc-pr.zip)  = every PR graph vertex; q = 15000 iff kNeutrinoVertex
                                   (MultiAlgBlobClustering.cxx:1011).

*** The two exclusions that decide the headline number ***

  1. sub_cluster_id == -1 rows MUST be dropped before the continuity test.  They are
     vertex fit points, they occur on both sides of the cathode, and leaving them in
     makes the negative-side and positive-side segment sets share -1 in EVERY event --
     i.e. every crosser looks "spanned" and the split rate reads a spurious 0.
  2. T_tagger nu_(x,y,z) == (0,0,0) exactly is an unfilled sentinel, not a vertex at
     the cathode.  50 of the 629 evaluated events in the doc pr/11 census carry it.

Classification (per event, on the candidate's fitted trajectory):

  tiny        fewer than --min-pts fitted segment points
  no-contact  no fitted point within --near cm of the cathode on either side
  one-sided   fitted points near the cathode on one side only
  spanned     nearest cross-cathode fit-point pair is closer than --gap cm in 3D AND
              both points belong to the SAME segment -- the crossing survived
  split       same proximity, but the two points belong to DIFFERENT segments -- the
              track is reconstructed as two tracks meeting at the cathode
  eroded      points exist on both sides but the nearest pair is farther than --gap cm:
              the fit does not reach the cathode from at least one side

For `one-sided` and `eroded` the script then asks whether the CHARGE was there anyway:
it counts clustering-global points on the far side within --radius cm in (y,z) of the
fit's cathode-end point.  Charge present + fit absent is the strongest form of the
question -- half the trajectory dropped rather than split.

Usage:
  cathode_nu_census.py --root work-mcp1kall-pr11v3 --sample mcp1k --out rows.tsv
  cathode_nu_census.py --merge rows1.tsv rows2.tsv --out census.tsv --summary
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np
import uproot

RE_SELECTED = re.compile(
    r'TaggerCheckNeutrino: selected main cluster (\S+) \(t0 ([-0-9.]+) us, '
    r'L ([-0-9.]+) cm, (\d+) associated\)')

COLUMNS = [
    'sample', 'run', 'subrun', 'event', 'cls',
    'main_cid', 'n_fit', 'n_seg', 'sel_len_cm', 'n_assoc',
    # how close the fit gets to the cathode, and how much of it sits near it
    'fit_absx', 'n_near_neg', 'n_near_pos',
    # nearest cross-cathode pair of FITTED points
    'gap3d_cm', 'dyz_cm', 'x_neg', 'x_pos', 'seg_neg', 'seg_pos',
    'shower_neg', 'shower_pos', 'pid_neg', 'pid_pos',
    # the same pair on the CHARGE (clustering-global)
    'qgap3d_cm', 'q_n6_neg', 'q_n6_pos', 'q_farside_n',
    'q_extrap_dperp', 'q_extrap_n',
    # were the two halves ONE cluster in the Q/L output the PR job is fed?
    'ql_joined', 'ql_cid_near', 'ql_cid_far',
    # PR graph vertex nearest the cathode
    'vtx_absx', 'vtx_y', 'vtx_z', 'vtx_is_main', 'vtx_nseg', 'vtx_seg_sides',
    # neutrino vertex
    'nu_x', 'nu_y', 'nu_z', 'nu_sentinel', 'nu_absx',
    # dQ/dx through the gap, in e/cm
    'dqdx_in3', 'dqdx_3to15', 'dqdx_ratio',
]

BLANK = ''


def read_log(path):
    """nu_evaluated + the selected candidate's t0/length/n-associated."""
    if not os.path.isfile(path):
        return None
    with open(path, errors='replace') as f:
        for line in f:
            m = RE_SELECTED.search(line)
            if m:
                return dict(main=m.group(1), t0=float(m.group(2)),
                            length=float(m.group(3)), nassoc=int(m.group(4)))
    return None


def bee_layer(zf, layer):
    for name in zf.namelist():
        if os.path.basename(name) == f'0-{layer}.json':
            return json.loads(zf.read(name))
    return None


def nearest_pair(P, Q):
    """Index of the closest (P_i, Q_j) pair and its distance, chunked to bound memory."""
    best = (None, None, np.inf)
    for lo in range(0, len(P), 4096):
        blk = P[lo:lo + 4096]
        d = np.linalg.norm(blk[:, None, :] - Q[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < best[2]:
            best = (lo + i, j, float(d[i, j]))
    return best


def census_event(root, sample, evt, args):
    d = os.path.join(root, f'pr_evt{evt}')
    sel = read_log(os.path.join(d, f'wct_pr_evt{evt}.log'))
    if sel is None:
        return None                      # no candidate selected -> nothing to say
    rpath = os.path.join(d, 'tracking-pr.root')
    if not os.path.isfile(rpath):
        return None

    f = uproot.open(rpath)
    trun = f['Trun'].arrays(library='np')
    row = dict.fromkeys(COLUMNS, BLANK)
    row.update(sample=sample, event=evt,
               run=int(trun['runNo'][0]), subrun=int(trun['subRunNo'][0]),
               sel_len_cm=round(sel['length'], 1), n_assoc=sel['nassoc'])

    t = f['T_rec_charge'].arrays(library='np')
    if len(t['x']) == 0:
        row['cls'] = 'tiny'
        row['n_fit'] = 0
        return row
    row['main_cid'] = int(t['cluster_id'][0])

    # T_tagger neutrino vertex, with the (0,0,0) sentinel called out.
    if 'T_tagger' in [k.split(';')[0] for k in f.keys()]:
        tg = f['T_tagger'].arrays(['nu_x', 'nu_y', 'nu_z'], library='np')
        nx, ny, nz = float(tg['nu_x'][0]), float(tg['nu_y'][0]), float(tg['nu_z'][0])
        sentinel = int(nx == 0.0 and ny == 0.0 and nz == 0.0)
        row.update(nu_x=round(nx, 2), nu_y=round(ny, 2), nu_z=round(nz, 2),
                   nu_sentinel=sentinel,
                   nu_absx=BLANK if sentinel else round(abs(nx), 2))

    scale, offset = float(trun['dQdx_scale'][0]), float(trun['dQdx_offset'][0])

    seg_rows = t['sub_cluster_id'] != -1          # exclusion 1 (see module docstring)
    x, y, z = t['x'][seg_rows], t['y'][seg_rows], t['z'][seg_rows]
    sub = t['sub_cluster_id'][seg_rows]
    shower, pid = t['flag_shower'][seg_rows], t['particle_id'][seg_rows]
    dq = (t['q'][seg_rows] - offset) / scale
    dx = t['nq'][seg_rows]
    row['n_fit'] = int(len(x))
    row['n_seg'] = int(len(set(sub.tolist())))
    if len(x) < args.min_pts:
        row['cls'] = 'tiny'
        return row

    neg = x < 0
    pos = x > 0
    near_neg = neg & (x > -args.near)
    near_pos = pos & (x < args.near)
    row.update(fit_absx=round(float(np.min(np.abs(x))), 2),
               n_near_neg=int(near_neg.sum()), n_near_pos=int(near_pos.sum()))
    if not near_neg.any() and not near_pos.any():
        row['cls'] = 'no-contact'
        return row

    P = np.stack([x[neg], y[neg], z[neg]], 1)
    Q = np.stack([x[pos], y[pos], z[pos]], 1)
    if len(P) == 0 or len(Q) == 0:
        row['cls'] = 'one-sided'
        i_end = int(np.argmin(np.abs(x)))
        cathode_pt = np.array([x[i_end], y[i_end], z[i_end]])
    else:
        i, j, gap = nearest_pair(P, Q)
        pn, pp = P[i], Q[j]
        si, sj = int(sub[neg][i]), int(sub[pos][j])
        row.update(gap3d_cm=round(gap, 2),
                   dyz_cm=round(float(np.hypot(pn[1] - pp[1], pn[2] - pp[2])), 2),
                   x_neg=round(float(pn[0]), 2), x_pos=round(float(pp[0]), 2),
                   seg_neg=si, seg_pos=sj,
                   shower_neg=int(shower[neg][i]), shower_pos=int(shower[pos][j]),
                   pid_neg=int(pid[neg][i]), pid_pos=int(pid[pos][j]))
        if gap > args.gap:
            row['cls'] = 'eroded'
        elif si == sj:
            row['cls'] = 'spanned'
        else:
            row['cls'] = 'split'
        cathode_pt = (pn + pp) / 2.0

    # Probe points for the charge test: the fit's cathode-end point on each side that
    # has one.  For `eroded` the two ends can be 12+ cm apart, so a single tube on
    # their midpoint would sit on neither track -- probe each end separately and keep
    # the closest charge crossing found.
    probes = []
    for side in (neg, pos):
        if side.any():
            k = int(np.argmin(np.abs(x[side])))
            probes.append((np.array([x[side][k], y[side][k], z[side][k]]),
                           int(sub[side][k])))
    if not probes:
        probes = [(cathode_pt, None)]

    # --- PR graph vertex nearest the cathode, and what touches it ----------------
    zpath = os.path.join(d, 'mabc-pr.zip')
    clus = None
    chosen_pt = probes[0][0]      # replaced below by the probe the charge test picks
    if os.path.isfile(zpath):
        with zipfile.ZipFile(zpath) as zf:
            vtx = bee_layer(zf, 'vertices-global')
            clus = bee_layer(zf, 'clustering-global')
        if vtx and vtx['x']:
            vx = np.asarray(vtx['x'])
            k = int(np.argmin(np.abs(vx)))
            row.update(vtx_absx=round(float(abs(vx[k])), 2),
                       vtx_y=round(float(vtx['y'][k]), 1),
                       vtx_z=round(float(vtx['z'][k]), 1),
                       vtx_is_main=int(vtx['q'][k] >= 15000.0))
            vp = np.array([vx[k], vtx['y'][k], vtx['z'][k]])
            near = np.linalg.norm(np.stack([x, y, z], 1) - vp, axis=1) < args.vtx_radius
            segs = sorted(set(sub[near].tolist()))
            row['vtx_nseg'] = len(segs)
            # which side of the cathode each incident segment lives on
            sides = []
            for s in segs:
                sx = x[sub == s]
                sides.append('-' if sx.max() <= 0 else ('+' if sx.min() >= 0 else '0'))
            row['vtx_seg_sides'] = ''.join(sides) if sides else BLANK

    # --- was the CHARGE there when the fit was not? -------------------------------
    if clus and clus['x']:
        cxyz = np.stack([np.asarray(clus['x']), np.asarray(clus['y']),
                         np.asarray(clus['z'])], 1)
        cx = cxyz[:, 0]
        best = None
        for pt, seg in probes:
            tube = np.hypot(cxyz[:, 1] - pt[1], cxyz[:, 2] - pt[2]) < args.radius
            cn = cxyz[tube & (cx < 0)]
            cp = cxyz[tube & (cx > 0)]
            if not len(cn) or not len(cp):
                continue
            g = nearest_pair(cn, cp)[2]
            if best is None or g < best[0]:
                best = (g, tube, pt, seg)
        if best is not None:
            g, tube, pt, seg = best
            chosen_pt = pt
            row['qgap3d_cm'] = round(g, 2)
            row['q_n6_neg'] = int((tube & (cx < 0) & (cx > -args.near)).sum())
            row['q_n6_pos'] = int((tube & (cx > 0) & (cx < args.near)).sum())
            far = (cx > 0) if pt[0] < 0 else (cx < 0)
            row['q_farside_n'] = int((tube & far & (np.abs(cx) < args.near)).sum())

            # Direction test: is the far-side charge ALONG the fit's extrapolation, or
            # merely inside the cylinder?  A different particle passing within --radius
            # in (y,z) would satisfy the tube test but not this one.  Local direction
            # from the PCA of the cathode-end segment's points nearest that end.
            if seg is not None:
                sp = np.stack([x, y, z], 1)[sub == seg]
                if len(sp) >= 3:
                    order = np.argsort(np.linalg.norm(sp - pt, axis=1))[:10]
                    loc = sp[order]
                    u, s_, vt = np.linalg.svd(loc - loc.mean(0), full_matrices=False)
                    dvec = vt[0] / np.linalg.norm(vt[0])
                    if np.dot(pt - loc.mean(0), dvec) < 0:
                        dvec = -dvec          # orient outward, toward the cathode
                    fc = cxyz[tube & far & (np.abs(cx) < args.near)]
                    if len(fc):
                        rel = fc - pt
                        proj = rel @ dvec
                        perp = np.linalg.norm(rel - np.outer(proj, dvec), axis=1)
                        ahead = proj > 0
                        if ahead.any():
                            row['q_extrap_dperp'] = round(float(perp[ahead].min()), 2)
                            row['q_extrap_n'] = int((perp[ahead] < 3.0).sum())
                        else:
                            row['q_extrap_dperp'] = BLANK
                            row['q_extrap_n'] = 0

    # --- did the PR job even RECEIVE the two halves as one cluster? ----------------
    # The PR job runs on the post-Q/L pctree; ql_evt<ID>/mabc-all-apa.zip's
    # clustering-global records that same state, i.e. AFTER cathode_connect
    # (cfg/pgrapher/experiment/sbnd/clus.jsonnet:84,376).  Without this test a fit that
    # stops at the cathode looks like a tracking failure even when the far half was
    # never in the candidate's cluster -- which is what it usually is (doc pr/12 sec 6).
    if args.ql_root:
        pt = chosen_pt           # the same tube the charge test found the crossing in
        qz = next((os.path.join(w, f'ql_evt{evt}', 'mabc-all-apa.zip')
                   for w in args.ql_root
                   if os.path.isfile(os.path.join(w, f'ql_evt{evt}',
                                                  'mabc-all-apa.zip'))), None)
        if qz:
            with zipfile.ZipFile(qz) as zf:
                q = bee_layer(zf, 'clustering-global')
            if q and q['x']:
                qx = np.asarray(q['x']); qy = np.asarray(q['y']); qz_ = np.asarray(q['z'])
                qc = np.asarray(q['cluster_id'])
                tube = np.hypot(qy - pt[1], qz_ - pt[2]) < args.radius
                inx = tube & (np.abs(qx) < args.near)
                a = collections.Counter(qc[inx & (qx < 0)].tolist())
                b = collections.Counter(qc[inx & (qx > 0)].tolist())
                if a and b:
                    row['ql_joined'] = int(bool(set(a) & set(b)))
                    row['ql_cid_near'] = a.most_common(1)[0][0]
                    row['ql_cid_far'] = b.most_common(1)[0][0]
                    if pt[0] > 0:            # name them relative to the fit's side
                        row['ql_cid_near'], row['ql_cid_far'] = \
                            row['ql_cid_far'], row['ql_cid_near']

    # --- dQ/dx through the gap ----------------------------------------------------
    # Only meaningful where the fitted trajectory actually crosses: elsewhere the two
    # bands average unrelated segments.
    if row['cls'] in ('spanned', 'split'):
        good = dx > 0
        if good.any():
            dqdx = dq[good] / dx[good]
            ax = np.abs(x[good])
            near3 = ax < 3.0
            band = (ax >= 3.0) & (ax < 15.0)
            if near3.any():
                row['dqdx_in3'] = int(np.median(dqdx[near3]))
            if band.any():
                row['dqdx_3to15'] = int(np.median(dqdx[band]))
            if near3.any() and band.any() and np.median(dqdx[band]) > 0:
                row['dqdx_ratio'] = round(float(np.median(dqdx[near3]) /
                                                np.median(dqdx[band])), 3)
    return row


def write_rows(rows, out, header=True):
    fh = open(out, 'w') if out else sys.stdout
    if header:
        fh.write('\t'.join(COLUMNS) + '\n')
    for r in rows:
        fh.write('\t'.join(str(r.get(c, BLANK)) for c in COLUMNS) + '\n')
    if out:
        fh.close()


def summarize(rows):
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r['sample']][r['cls']] += 1
    order = ['spanned', 'split', 'eroded', 'one-sided', 'no-contact', 'tiny']
    print('\n== class x sample (events with a selected nu candidate) ==')
    print('\t'.join(['class'] + sorted(per) + ['all']))
    tot = collections.Counter()
    for c in order:
        line = [c]
        for s in sorted(per):
            line.append(str(per[s][c]))
            tot[c] += per[s][c]
        line.append(str(tot[c]))
        print('\t'.join(line))
    print('\t'.join(['TOTAL'] + [str(sum(per[s].values())) for s in sorted(per)] +
                    [str(sum(tot.values()))]))

    crossers = [r for r in rows if r['cls'] in ('spanned', 'split')]
    if crossers:
        print(f'\n== the {len(crossers)} events whose fit reaches the cathode from both '
              f'sides ==')
        nsplit = sum(1 for r in crossers if r['cls'] == 'split')
        print(f'split into two segments at the cathode: {nsplit}/{len(crossers)} '
              f'({100.0*nsplit/len(crossers):.0f}%)')
        for tag in ('spanned', 'split'):
            sub = [r for r in crossers if r['cls'] == tag]
            if not sub:
                continue
            g = [float(r['gap3d_cm']) for r in sub]
            dyz = [float(r['dyz_cm']) for r in sub]
            vx = [float(r['vtx_absx']) for r in sub if r['vtx_absx'] != BLANK]
            print(f'  {tag:8s} n={len(sub):3d}  gap3d med {np.median(g):.2f} cm  '
                  f'dyz med {np.median(dyz):.2f} cm  '
                  f'nearest-vertex |x| med {np.median(vx):.2f} cm')

    # The discriminating case: the CHARGE runs continuously across the cathode
    # (small qgap3d) while the FIT does not reach across it.  Far-side charge alone
    # is not enough -- it is usually a different track a few cm away in y-z.
    lost = [r for r in rows if r['cls'] in ('one-sided', 'eroded')
            and r['qgap3d_cm'] != BLANK and float(r['qgap3d_cm']) <= 4.0]
    aligned = [r for r in lost if r['q_extrap_n'] != BLANK and int(r['q_extrap_n']) > 0]
    print(f'\n== charge continuous across the cathode but the fit is not: '
          f'{len(lost)} events, of which {len(aligned)} have the far-side charge ON '
          f'the fit extrapolation ==')
    for r in sorted(lost, key=lambda r: -float(r['sel_len_cm'] or 0))[:20]:
        tag = 'ALIGNED' if r in aligned else 'off-axis'
        print(f"  {r['sample']:8s} evt {r['event']:>7s}  cls {r['cls']:9s} "
              f"L {r['sel_len_cm']:>6} cm  charge gap {r['qgap3d_cm']:>5} cm  "
              f"fit gap {r['gap3d_cm'] or 'n/a':>6} cm  "
              f"charge pts -/+ {r['q_n6_neg']}/{r['q_n6_pos']}  "
              f"extrap dperp {r['q_extrap_dperp']:>5} n {r['q_extrap_n']:>4}  {tag}")

    joined = [r for r in rows if r['ql_joined'] != BLANK]
    if joined:
        print(f'\n== join test: were the two halves ONE cluster in the PR job\'s input? '
              f'({len(joined)} events with charge on both sides) ==')
        print('class\tjoined\tnot-joined')
        for c in ('spanned', 'split', 'eroded', 'one-sided'):
            sub = [r for r in joined if r['cls'] == c]
            if sub:
                y = sum(1 for r in sub if r['ql_joined'] == '1')
                print(f'{c}\t{y}\t{len(sub) - y}')

    # The population the Bee sets should carry: the candidate's CHARGE crosses the
    # cathode, whatever the fit did.
    crossing_charge = [r for r in rows if r['qgap3d_cm'] != BLANK
                       and float(r['qgap3d_cm']) <= 4.0]
    print(f'\n== candidates whose CHARGE crosses the cathode (qgap <= 4 cm): '
          f'{len(crossing_charge)} ==')
    print('   by class: ' + '  '.join(
        f'{c}={sum(1 for r in crossing_charge if r["cls"] == c)}'
        for c in ('spanned', 'split', 'eroded', 'one-sided')))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', help='PR arm directory holding pr_evt<ID>/')
    ap.add_argument('--ql-root', action='append',
                    help='Q/L arm holding ql_evt<ID>/mabc-all-apa.zip; enables the '
                         'join test (were the two halves one cluster in the PR job\'s '
                         'input?).  Repeatable; first root that has the event wins')
    ap.add_argument('--sample', default='?', help='label for the sample column')
    ap.add_argument('--events', nargs='*', help='restrict to these event ids')
    ap.add_argument('--merge', nargs='*', help='merge previously written row files')
    ap.add_argument('--out', help='output TSV (default stdout)')
    ap.add_argument('--no-header', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--near', type=float, default=6.0,
                    help='cm from the cathode counted as "contact" (default 6)')
    ap.add_argument('--gap', type=float, default=4.0,
                    help='max 3D gap for a fit to count as reaching across (default 4)')
    ap.add_argument('--radius', type=float, default=10.0,
                    help='y-z radius for the charge-vs-fit comparison (default 10)')
    ap.add_argument('--vtx-radius', type=float, default=3.0,
                    help='radius for segments incident on the cathode vertex (default 3)')
    ap.add_argument('--min-pts', type=int, default=10,
                    help='minimum fitted segment points to classify (default 10)')
    args = ap.parse_args()

    if args.merge:
        rows = []
        for path in args.merge:
            with open(path) as fh:
                for line in fh:
                    p = line.rstrip('\n').split('\t')
                    if p[0] == 'sample':
                        continue
                    rows.append(dict(zip(COLUMNS, p)))
        rows.sort(key=lambda r: (r['sample'], int(r['event'])))
        write_rows(rows, args.out, not args.no_header)
        if args.summary:
            summarize(rows)
        return 0

    if not args.root:
        ap.error('--root or --merge is required')
    if args.events:
        evts = list(args.events)
    else:
        evts = sorted(os.path.basename(p)[len('pr_evt'):]
                      for p in glob.glob(os.path.join(args.root, 'pr_evt*')))
    rows = []
    for evt in evts:
        r = census_event(args.root, args.sample, evt, args)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: int(r['event']))
    write_rows(rows, args.out, not args.no_header)
    if args.summary:
        summarize(rows)
    print(f'{args.sample}: {len(rows)} events with a selected nu candidate '
          f'(of {len(evts)} run)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())

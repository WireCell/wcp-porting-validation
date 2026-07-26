#!/usr/bin/env python3
"""Audit: does ClusteringUnmergeBundle tear apart a cathode-crossing merge?

Question (owner, doc 52 follow-up).  The all-APA post-Q/L clustering stage
joins the two halves of a cathode crosser (clustering_cathode_connect, and the
generic extend/regular/parallel_prolong/close passes running with
use_flash_t0=true).  Those are REAL merges: one particle, two TPCs.  The two
ClusteringUnmergeBundle instances in the PR job then undo the two BOOKKEEPING
merges -- the flash-time bundle merge (real_cluster_id/real_cluster_main) and
the per-APA isolated grouping (assoc_cluster_id/assoc_cluster_main).  A real
merge must survive both.

Method (pure file parsing, no re-run, no writes into any tag dir):

  * "pre-merge unit" = the set of blobs of one Q/L-stage cluster that share the
    same real_cluster_id, i.e. one member of the flash-time merge.  A unit whose
    blobs sit in BOTH SBND TPCs (blob wpid>>4 == 0 and == 1) is, by
    construction, the product of a cross-cathode merge -- nothing else in the
    chain can put charge from both drift volumes into one cluster.  This is the
    crosser definition used here; cathode_connect itself logs no pair list.

  * The two un-merges are replayed exactly as ClusteringUnmergeBundle does it:
      pass 1 (instance "pr"):      retained main = blobs with real_cluster_main != 0
                                   each other real_cluster_id -> one associated cluster
      pass 2 (instance "prassoc"): runs ONLY on the retained main (the split-off
                                   parts get flag_main_cluster cleared, and
                                   visit() skips clusters without it)
                                   retained main = blobs with assoc_cluster_main != 0
                                   each other assoc_cluster_id -> one associated cluster
    So the final main is {real_main != 0} AND {assoc_main != 0} -- the "union
    rule": every main-marked member is retained together.

  * Each cross-cathode unit is bucketed:
      (i)   MAIN-WHOLE      unit is the flash representative and the final main
                            still spans both TPCs                      <- good
      (ii)  ASSOC-WHOLE     unit is not the representative: it leaves whole as a
                            single associated cluster (intact partition, but
                            check_stm only COUNTS companions, never fits them)
      (iii) TORN            unit's blobs end up in more than one final cluster
                            with the two TPC sides separated            <- the bug
      (iv)  NO-MAIN-FLAG    the Q/L cluster carries no flag_main_cluster, so
                            visit() skips it entirely (crosser safe, but its own
                            fragments are never split off either)

  * Scatter check (doc 52 s11, the blob-pointer-keying defect): every
    assoc_cluster_main == 0 group and every non-representative real_cluster_id
    group must live in ONE TPC.  A group straddling the cathode means the
    provenance rows got attached to the wrong blobs.

  * END-STATE check, independent of the replay: the carved provenance survives
    into the PR job's own output tree (pctree-pr-evt*.tar.gz), so the delivered
    product can be measured directly.  For every cross-cathode unit, its
    SURVIVING MAIN CONTENT -- real_cluster_main != 0 AND assoc_cluster_main != 0
    -- must still span both TPCs.  NB real_cluster_id is unique only WITHIN a
    cluster (a cluster that was never flash-merged gets the default fill
    rid = its own ident, which can collide with a merged neighbour's pre-merge
    member id; evt288397 and evt288727 both do), so units are keyed by
    (Q/L cluster ident, rid).  The mapping is exact because a retained main keeps
    its Q/L ident and main content exists nowhere else.

  * Cross-check: the replay's final main blob count is compared against the PR
    tree per cluster ident.  Disagreements are all one-sided -- the PR tree
    keeps MORE blobs than the replay predicts -- because visit() skips clusters
    that fail require_in_scope (detector-edge shards: z outside [0,501] cm etc.)
    and process_cluster() skips nb<2.  The replay is therefore an UPPER BOUND on
    splitting: whatever it says survives whole, survives whole in reality too.
    The audit fails if any disagreement goes the other way.

  * Also counted, as a separate observation and NOT a defect: Q/L clusters that
    spanned both TPCs only because the flash-time bundle put two DIFFERENT
    pre-merge clusters together (different real_cluster_id).  Undoing that is the
    whole point of the visitor -- but if one of those pairs is really one
    particle that cathode_connect failed to join, that is a cathode_connect
    recall question, so they are listed for hand-scan.

Repro:

    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
    python3 unmerge_crosser_audit.py work-mcp10-d52ron work-mcp1000-d52ron \
                                     work-mcp1000b-d52ron
    # add -v for the per-unit table
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile
from collections import defaultdict

import numpy as np

# SBND: two drift volumes, cathode at x = 0.  Blob wpid packs the apa index in
# bits >>4 (observed values 7 = apa0/TPC0/x<0, 23 = apa1/TPC1/x>0).
TPC_OF_WPID = lambda w: int(w) >> 4

# Screening thresholds for "was this really one particle cathode_connect missed?"
NEAR_CATHODE_CM = 20.0     # only blobs this close to the cathode can bridge it
MISSED_CROSSER_CM = 10.0   # closest approach of the two halves at the cathode


def load_tensors(fname):
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    return {md['datapath']: (md, arrays.get(b))
            for b, md in metas.items() if 'datapath' in md}


def parse_clusters(fname):
    """[{ident, main, assoc, t0, gid, nblob, tpc[], rid[], rmain[], aid[], amain[], x[]}]

    Per-blob arrays only; `x` is the blob's mean corrected drift x (cm) for
    reporting, never for the TPC decision (which uses wpid).
    """
    bp = load_tensors(fname)
    live = [p for p in bp if re.fullmatch(r'pointtrees/\d+/live', p)]
    if len(live) != 1:
        raise RuntimeError(f'{fname}: expected one live tree, got {live}')
    live = live[0]
    event = int(live.split('/')[1])
    md = bp[live][0]
    items = bp[md['pointclouds']][0]['items']
    lpc = bp[md['lpcmaps']][0]['arrays']

    def arr(pcname, aname):
        ds = bp[items[pcname]][0]['arrays']
        return bp[ds[aname]][1] if aname in ds else None

    def has(pcname, aname):
        return pcname in items and aname in bp[items[pcname]][0]['arrays']

    ident = arr('cluster_scalar', 'ident').astype(int)
    fmain = arr('cluster_scalar', 'flag_main_cluster').astype(int)
    fassoc = (arr('cluster_scalar', 'flag_associated_cluster').astype(int)
              if has('cluster_scalar', 'flag_associated_cluster')
              else np.zeros_like(ident))
    t0 = arr('cluster_scalar', 'cluster_t0')
    gid = (arr('cluster_scalar', 'matched_flash_gid').astype(int)
           if has('cluster_scalar', 'matched_flash_gid') else np.full_like(ident, -1))

    # Node-order maps: a cluster node owns one cluster_scalar row; the blob
    # nodes that follow own one 'perblob'/'scalar' row and their 3d points.
    map_cs = bp[lpc['cluster_scalar']][1].astype(int)
    map_pb = bp[lpc['perblob']][1].astype(int) if 'perblob' in lpc else None
    map_3d = bp[lpc['3d']][1].astype(int)

    if map_pb is None:
        raise RuntimeError(f'{fname}: no perblob lpcmap (tree saved without provenance)')

    wpid_b = arr('scalar', 'wpid').astype(int)
    rid = arr('perblob', 'real_cluster_id')
    rmain = arr('perblob', 'real_cluster_main')
    aid = arr('perblob', 'assoc_cluster_id')
    amain = arr('perblob', 'assoc_cluster_main')
    x3 = arr('3d', 'x_t0cor')
    if x3 is None:
        x3 = arr('3d', 'x')
    y3 = arr('3d', 'y_cor')
    z3 = arr('3d', 'z_cor')
    if y3 is None:
        y3, z3 = arr('3d', 'y'), arr('3d', 'z')

    # Walk nodes: accumulate per-cluster blob-row spans and per-blob point spans.
    cl_blobs = defaultdict(list)     # cluster index -> [blob row index]
    blob_pts = []                    # blob row index -> (start, n) into 3d
    ci, brow, ppos = -1, 0, 0
    for n in range(len(map_cs)):
        if map_cs[n]:
            ci += 1
        if map_pb[n]:
            for _ in range(int(map_pb[n])):
                if ci >= 0:
                    cl_blobs[ci].append(brow)
                brow += 1
        if map_3d[n]:
            blob_pts.append((ppos, int(map_3d[n])))
            ppos += int(map_3d[n])

    out = []
    for i in range(len(ident)):
        rows = np.array(cl_blobs.get(i, []), dtype=int)
        if rows.size == 0:
            out.append(dict(ident=int(ident[i]), main=int(fmain[i]), assoc=int(fassoc[i]),
                            t0=float(t0[i]), gid=int(gid[i]), nblob=0,
                            tpc=np.zeros(0, int), rid=np.zeros(0, int),
                            rmain=np.zeros(0, int), aid=np.zeros(0, int),
                            amain=np.zeros(0, int), x=np.zeros(0),
                            y=np.zeros(0), z=np.zeros(0)))
            continue
        xs = np.array([x3[blob_pts[r][0]:blob_pts[r][0] + blob_pts[r][1]].mean()
                       if blob_pts[r][1] else np.nan for r in rows])
        ys = np.array([y3[blob_pts[r][0]:blob_pts[r][0] + blob_pts[r][1]].mean()
                       if blob_pts[r][1] else np.nan for r in rows])
        zs = np.array([z3[blob_pts[r][0]:blob_pts[r][0] + blob_pts[r][1]].mean()
                       if blob_pts[r][1] else np.nan for r in rows])
        out.append(dict(
            y=ys / 10.0, z=zs / 10.0,
            ident=int(ident[i]), main=int(fmain[i]), assoc=int(fassoc[i]),
            t0=float(t0[i]), gid=int(gid[i]), nblob=int(rows.size),
            tpc=np.array([TPC_OF_WPID(w) for w in wpid_b[rows]], dtype=int),
            rid=rid[rows].astype(int), rmain=rmain[rows].astype(int),
            aid=aid[rows].astype(int), amain=amain[rows].astype(int),
            x=xs / 10.0))   # internal mm -> cm for the report
    return event, out


def audit_event(qlfile, prfile, verbose=False):
    event, cls = parse_clusters(qlfile)
    res = dict(event=event, ncl=len(cls),
               crossers=0, main_whole=0, assoc_whole=0, torn=0, no_main_flag=0,
               scatter_real=0, scatter_assoc=0, bundle_only=0, bundle_near=0,
               end_crossers=0, end_split=0, absent=0, untouched=0,
               detail=[], replay_mismatch=[])

    for c in cls:
        if c['nblob'] < 1:
            continue
        rid, rmain, aid, amain, tpc = c['rid'], c['rmain'], c['aid'], c['amain'], c['tpc']

        # --- cross-cathode pre-merge units (groups of one real_cluster_id) ---
        for u in np.unique(rid):
            sel = rid == u
            sides = set(tpc[sel].tolist())
            if len(sides) < 2:
                continue
            res['crossers'] += 1
            is_rep = bool((rmain[sel] != 0).any())
            # rmain is constant within a member by construction; flag if not.
            if is_rep and not (rmain[sel] != 0).all():
                res['replay_mismatch'].append(
                    f'evt{event} cl{c["ident"]} rid={u}: real_cluster_main mixed inside one member')
            span_cm = float(np.nanmax(c['x'][sel]) - np.nanmin(c['x'][sel]))
            info = dict(evt=event, cl=c['ident'], rid=int(u), nb=int(sel.sum()),
                        span=span_cm, gid=c['gid'], t0=c['t0'])

            if c['main'] == 0:
                res['no_main_flag'] += 1
                info['bucket'] = 'NO-MAIN-FLAG'
            elif not is_rep:
                # pass 1 moves the whole member out as one associated cluster
                res['assoc_whole'] += 1
                info['bucket'] = 'ASSOC-WHOLE'
            else:
                # pass 1 keeps it; pass 2 keeps only amain != 0
                keep = sel & (amain != 0)
                kept_sides = set(tpc[keep].tolist())
                if kept_sides == sides:
                    res['main_whole'] += 1
                    info['bucket'] = 'MAIN-WHOLE'
                    info['kept'] = int(keep.sum())
                else:
                    res['torn'] += 1
                    info['bucket'] = 'TORN'
                    info['kept'] = int(keep.sum())
                    info['kept_sides'] = sorted(kept_sides)
            res['detail'].append(info)

        # --- scatter check: a split-off group must live in ONE TPC ---
        for u in np.unique(rid):
            sel = (rid == u) & (rmain == 0)
            if sel.sum() and len(set(tpc[sel].tolist())) > 1:
                res['scatter_real'] += 1
                res['detail'].append(dict(evt=event, cl=c['ident'], rid=int(u),
                                          nb=int(sel.sum()), bucket='SCATTER-real',
                                          span=float(np.nanmax(c['x'][sel]) - np.nanmin(c['x'][sel])),
                                          gid=c['gid'], t0=c['t0']))
        if c['main']:
            keep1 = rmain != 0
            for u in np.unique(aid[keep1]) if keep1.any() else []:
                sel = keep1 & (aid == u) & (amain == 0)
                if sel.sum() and len(set(tpc[sel].tolist())) > 1:
                    res['scatter_assoc'] += 1
                    res['detail'].append(dict(evt=event, cl=c['ident'], rid=int(u),
                                              nb=int(sel.sum()), bucket='SCATTER-assoc',
                                              span=float(np.nanmax(c['x'][sel]) - np.nanmin(c['x'][sel])),
                                              gid=c['gid'], t0=c['t0']))

    # --- observation: cross-cathode content that came from the flash bundle ---
    # (a Q/L cluster spanning both TPCs whose per-real_cluster_id units are each
    # single-TPC -- the visitor correctly separates them; listed for hand-scan in
    # case cathode_connect should have joined a pair)
    for c in cls:
        if c['nblob'] < 1 or not c['main']:
            continue
        if len(set(c['tpc'].tolist())) < 2:
            continue
        if any(len(set(c['tpc'][c['rid'] == u].tolist())) > 1 for u in np.unique(c['rid'])):
            continue                      # has a genuine cross-cathode unit
        res['bundle_only'] += 1
        # Screening: could this pair actually be ONE particle that
        # cathode_connect failed to join?  Then both sides would reach the
        # cathode and meet there.  d_min = smallest blob-centre separation
        # between the TPC0 and TPC1 halves, restricted to blobs within
        # NEAR_CATHODE_CM of the cathode on each side.
        a, b = c['tpc'] == 0, c['tpc'] == 1
        near = np.abs(c['x']) < NEAR_CATHODE_CM
        pa, pb = a & near, b & near
        dmin = np.inf
        if pa.any() and pb.any():
            A = np.stack([c['x'][pa], c['y'][pa], c['z'][pa]], axis=1)
            B = np.stack([c['x'][pb], c['y'][pb], c['z'][pb]], axis=1)
            dmin = float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min())
        flag = dmin < MISSED_CROSSER_CM
        if flag:
            res['bundle_near'] += 1
        res['detail'].append(dict(evt=event, cl=c['ident'], rid=-1, nb=c['nblob'],
                                  bucket='BUNDLE-NEAR' if flag else 'BUNDLE-ONLY',
                                  span=(dmin if np.isfinite(dmin) else -1.0),
                                  gid=c['gid'], t0=c['t0']))

    # --- END-STATE check + replay cross-check against the PR output tree ---
    if prfile and os.path.exists(prfile):
        _, prcls = parse_clusters(prfile)

        # END STATE, read off the delivered PR tree.
        #
        # Keying: real_cluster_id is the pre-flash-merge cluster ident, unique
        # only WITHIN a cluster -- two different Q/L clusters can both carry a
        # unit labelled 11 (a cluster that was never flash-merged gets the
        # default fill rid = its own current ident, which can collide with a
        # merged neighbour's pre-merge member ident; seen in evt288397 and
        # evt288727).  So a unit is keyed by (Q/L cluster ident, rid).
        #
        # The mapping to the PR tree is exact: the retained main KEEPS the Q/L
        # cluster's ident, and blobs with real_cluster_main != 0 AND
        # assoc_cluster_main != 0 exist ONLY in a retained main (pass 1 splits
        # off real_cluster_main == 0, pass 2 splits off assoc_cluster_main == 0).
        # So the unit's surviving main content is exactly what PR cluster
        # <ident> holds for that rid, and it must still span both TPCs.
        pr_by_ident = {c['ident']: c for c in prcls if c['main'] and not c['assoc']}
        for c in cls:
            if c['nblob'] < 1 or not c['main']:
                continue
            for u in np.unique(c['rid']):
                if len(set(c['tpc'][c['rid'] == u].tolist())) < 2:
                    continue
                p = pr_by_ident.get(c['ident'])
                if p is None or p['nblob'] == 0:
                    continue      # cluster never reached the PR product
                keep = (p['rid'] == u) & (p['rmain'] != 0) & (p['amain'] != 0)
                if not keep.any():
                    continue
                res['end_crossers'] += 1
                sides = set(p['tpc'][keep].tolist())
                if len(sides) < 2:
                    res['end_split'] += 1
                    res['detail'].append(
                        dict(evt=event, cl=c['ident'], rid=int(u), nb=int(keep.sum()),
                             bucket='END-SPLIT', gid=c['gid'], t0=c['t0'],
                             span=float(np.nanmax(p['x'][keep]) - np.nanmin(p['x'][keep]))))

        # replay cross-check (one-sided: see the module docstring)
        pred = {c['ident']: int(((c['rmain'] != 0) & (c['amain'] != 0)).sum())
                for c in cls if c['main'] and c['nblob']}
        got = {c['ident']: c['nblob'] for c in prcls if c['main'] and not c['assoc']}
        for k, v in sorted(pred.items()):
            if k not in got:
                res['absent'] += 1           # ident not in the PR tree at all
            elif got[k] > v:
                res['untouched'] += 1        # visitor left it alone (require_in_scope / nb<2)
            elif got[k] < v:
                res['replay_mismatch'].append(
                    f'evt{event} cl{k}: PR tree main has {got[k]} blobs, FEWER than the '
                    f'replay upper bound {v} -- the visitor split MORE than modelled')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+', help='work-*-<tag> directories')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    grand = defaultdict(int)
    all_detail = []
    mismatches = []
    print(f'{"work dir":22s} {"evt":>7s} {"cl":>4s} {"cross":>6s} {"main":>5s} '
          f'{"assoc":>6s} {"torn":>5s} {"noflag":>7s} {"scat":>5s} {"endX":>5s} '
          f'{"endsp":>6s} {"bundl":>6s}')
    for d in args.dirs:
        sub = defaultdict(int)
        for qlf in sorted(glob.glob(os.path.join(d, 'ql_evt*', 'pctree-evt*.tar.gz'))):
            evtdir = os.path.basename(os.path.dirname(qlf))
            evt = evtdir[len('ql_evt'):]
            prf = os.path.join(d, f'nusel_evt{evt}', f'pctree-pr-evt{evt}.tar.gz')
            try:
                r = audit_event(qlf, prf, args.verbose)
            except Exception as e:
                print(f'{d:26s} {evt:>7s}  ERROR: {e}')
                continue
            scat = r['scatter_real'] + r['scatter_assoc']
            print(f'{d:22s} {r["event"]:>7d} {r["ncl"]:>4d} {r["crossers"]:>6d} '
                  f'{r["main_whole"]:>5d} {r["assoc_whole"]:>6d} {r["torn"]:>5d} '
                  f'{r["no_main_flag"]:>7d} {scat:>5d} {r["end_crossers"]:>5d} '
                  f'{r["end_split"]:>6d} {r["bundle_only"]:>6d}')
            for k in ('ncl', 'crossers', 'main_whole', 'assoc_whole', 'torn',
                      'no_main_flag', 'scatter_real', 'scatter_assoc',
                      'bundle_only', 'bundle_near', 'end_crossers', 'end_split',
                      'absent', 'untouched'):
                sub[k] += r[k]
                grand[k] += r[k]
            sub['evts'] += 1
            grand['evts'] += 1
            all_detail += r['detail']
            mismatches += r['replay_mismatch']
        print(f'{"  -> " + d:22s} {sub["evts"]:>7d} {sub["ncl"]:>4d} {sub["crossers"]:>6d} '
              f'{sub["main_whole"]:>5d} {sub["assoc_whole"]:>6d} {sub["torn"]:>5d} '
              f'{sub["no_main_flag"]:>7d} {sub["scatter_real"] + sub["scatter_assoc"]:>5d} '
              f'{sub["end_crossers"]:>5d} {sub["end_split"]:>6d} {sub["bundle_only"]:>6d}')
        print()

    print('=' * 78)
    print(f'TOTAL over {grand["evts"]} events, {grand["ncl"]} Q/L clusters')
    print(f'  cross-cathode pre-merge units : {grand["crossers"]}')
    print(f'    (i)   MAIN-WHOLE  kept whole as the fitted main : {grand["main_whole"]}')
    print(f'    (ii)  ASSOC-WHOLE kept whole, demoted to assoc  : {grand["assoc_whole"]}')
    print(f'    (iii) TORN        split across the cathode      : {grand["torn"]}   <-- must be 0')
    print(f'    (iv)  NO-MAIN-FLAG cluster skipped by visit()   : {grand["no_main_flag"]}')
    print(f'  scatter check (group straddling the cathode)')
    print(f'    real_cluster_id groups  : {grand["scatter_real"]}   <-- must be 0')
    print(f'    assoc_cluster_id groups : {grand["scatter_assoc"]}   <-- must be 0')
    print(f'  END STATE, measured on the PR output tree')
    print(f'    cross-cathode units in the delivered product : {grand["end_crossers"]}')
    print(f'    of those, spread over >1 PR cluster          : {grand["end_split"]}   <-- must be 0')
    print(f'  replay is an UPPER BOUND on splitting')
    print(f'    clusters the visitor left alone (require_in_scope / nb<2) : '
          f'{grand["untouched"]}')
    print(f'    clusters whose ident is absent from the PR tree           : '
          f'{grand["absent"]}')
    print(f'    clusters split MORE than the replay predicts              : '
          f'{len(mismatches)}   <-- must be 0')
    print(f'    NB every one of the {grand["end_crossers"]} cross-cathode units WAS located by'
          f' ident in the PR tree,')
    print(f'       so none of the above fell into the silently-unsplit path.')
    for m in mismatches[:20]:
        print('    ' + m)
    print(f'  observation (not a defect)')
    print(f'    Q/L clusters spanning both TPCs only via the flash bundle : '
          f'{grand["bundle_only"]}')
    print(f'      of those, the two halves meet at the cathode (<{MISSED_CROSSER_CM:.0f} cm,'
          f' i.e. possibly one particle cathode_connect missed) : {grand["bundle_near"]}')

    if args.verbose or grand['torn'] or grand['scatter_real'] or grand['scatter_assoc'] \
            or grand['end_split']:
        print()
        print(f'{"bucket":14s} {"evt":>7s} {"cl":>5s} {"rid":>5s} {"nblob":>6s} '
              f'{"dx_cm":>8s} {"gid":>5s} {"t0_us":>10s}')
        for i in sorted(all_detail, key=lambda d: (d['bucket'], -d['nb'])):
            if not (args.verbose or i['bucket'] != 'MAIN-WHOLE'):
                continue
            print(f'{i["bucket"]:14s} {i["evt"]:>7d} {i["cl"]:>5d} {i["rid"]:>5d} '
                  f'{i["nb"]:>6d} {i["span"]:>8.1f} {i["gid"]:>5d} {i["t0"]/1000.0:>10.2f}')

    return 1 if (grand['torn'] or grand['scatter_real'] or grand['scatter_assoc']
                 or grand['end_split'] or mismatches) else 0


if __name__ == '__main__':
    sys.exit(main())

'''
Rebuild the DL net input cloud from calib-pr-evt<ID>.json, and load
hand-scan labels.

The cloud mirrors NeutrinoVertexFinder.cxx:4147-4179 (doc pr/52 S1):
  - every PR-graph vertex fit point, q = dQ*dQdx_scale + dQdx_offset;
  - every segment INTERIOR fit point (i = 1..n-2; endpoints duplicate
    vertex positions and are skipped by the C++);
  - units cm, float32; whole event (all clusters).

Known approximation (doc pr/77): the calib dump is written AFTER
snap_main_vertex_to_kink + improve_vertex + shower reclustering, i.e. the
post-DL refit graph, while the net saw the pre-refit graph.  parity_check.py
quantifies this per event.  Invalid-fit vertices (C++ used wcpt() + dQ=0):
not recorded in the dump; the proxy dQ==-1 && reduced_chi2==-1 (PRCommon.h
Fit defaults) maps them to the C++ q = 0*scale + offset.
'''
import json
import os
import glob
import numpy as np


def load_calib(path):
    with open(path) as fh:
        return json.load(fh)


def rebuild_cloud(calib):
    """calib (parsed calib-pr-evt JSON) -> (xyz (N,3) float32 cm,
    q (N,) float32 transformed feature, info dict)."""
    meta = calib.get('meta', {})
    scale = float(meta.get('dQdx_scale', 0.1))
    offset = float(meta.get('dQdx_offset', -1000.0))

    xs, ys, zs, qs = [], [], [], []
    n_invalid = 0
    for v in calib.get('vertices', []):
        fit = v.get('fit') or {}
        dQ = float(fit.get('dQ', -1.0))
        if dQ == -1.0 and float(fit.get('reduced_chi2', -1.0)) == -1.0:
            dQ = 0.0  # invalid-fit proxy: C++ feeds wcpt() point with dQ=0
            n_invalid += 1
        xs.append(fit['x']); ys.append(fit['y']); zs.append(fit['z'])
        qs.append(dQ * scale + offset)
    n_vtx = len(xs)

    for s in calib.get('segments', []):
        pts = s.get('points', [])
        for p in pts[1:-1]:
            xs.append(p['x']); ys.append(p['y']); zs.append(p['z'])
            qs.append(float(p['dQ']) * scale + offset)

    xyz = np.stack([np.asarray(xs, dtype=np.float32),
                    np.asarray(ys, dtype=np.float32),
                    np.asarray(zs, dtype=np.float32)], axis=1)
    q = np.asarray(qs, dtype=np.float32)
    info = dict(n_vtx_points=n_vtx, n_seg_points=len(qs) - n_vtx,
                n_invalid_fit=n_invalid, dQdx_scale=scale, dQdx_offset=offset,
                runNo=meta.get('runNo'), subRunNo=meta.get('subRunNo'),
                eventNo=meta.get('eventNo'))
    return xyz, q, info


def load_label(path):
    """labels-evt<ID>.json -> dict with truth = rank-1 pick."""
    with open(path) as fh:
        doc = json.load(fh)
    picks = sorted(doc.get('picks', []), key=lambda p: p.get('rank', 99))
    if not picks:
        raise ValueError('label has no picks: %s' % path)
    p1 = picks[0]
    return dict(
        eventNo=doc['eventNo'], runNo=doc.get('runNo'), subRunNo=doc.get('subRunNo'),
        arm=doc.get('arm'), scan_tag=doc.get('scan_tag'), saved_utc=doc.get('saved_utc'),
        source=doc.get('source'), confidence=doc.get('confidence'),
        not_a_candidate=bool(doc.get('not_a_candidate', False)),
        truth_xyz=np.array([p1['x'], p1['y'], p1['z']], dtype=np.float32),
        pick_kind=p1.get('kind'), pick_vertex_id=p1.get('vertex_id'),
        dis_to_main=p1.get('dis_to_main'),
        main_vertex=doc.get('main_vertex'), n_picks=len(picks),
        label_path=path)


def iter_labels(sbnd_root, tags):
    """Yield label dicts for every labels-evt*.json under the given tags.
    Read-only: never writes into vertex_labels/ (M13)."""
    for tag in tags:
        d = os.path.join(sbnd_root, 'vertex_labels', tag)
        files = sorted(glob.glob(os.path.join(d, 'labels-evt*.json')))
        if not files:
            raise FileNotFoundError('no labels under %s' % d)
        for f in files:
            yield load_label(f)


def calib_path_for_label(sbnd_root, label):
    """Resolve the calib JSON for a label: trust the recorded `source` if it
    exists, else <sbnd_root>/<arm>/pr_evt<ID>/calib-pr-evt<ID>.json."""
    src = label.get('source')
    if src and os.path.exists(src):
        return src
    evt = label['eventNo']
    return os.path.join(sbnd_root, label['arm'], 'pr_evt%d' % evt,
                        'calib-pr-evt%d.json' % evt)


def load_scores_tsv(sbnd_root, name='mcp1k',
                    fname='products/prod0813/%s-scores-prod0813.tsv'):
    """eventNo -> row dict from the pr_scores_table.py TSV (numu_score,
    event_label, ...).  numu_score parsed to float, blank -> None.
    Caution (doc pr/77 round 2): nue_score is the sentinel -15.0 for most
    events -- use numu_score directly, never numu>nue."""
    import csv
    path = os.path.join(sbnd_root, fname % name)
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            evt = int(r['event'])
            ns = r.get('numu_score', '')
            r['numu_score'] = float(ns) if ns not in ('', None) else None
            out[evt] = r
    return out


def is_corrective(label, tol=1.0):
    """Scanner disagreed with production: rank-1 pick moved by > tol cm from
    the production main vertex, or the pick isn't a graph candidate at all."""
    if label.get('not_a_candidate') or label.get('pick_kind') == 'manual':
        return True
    dis = label.get('dis_to_main')
    return dis is not None and float(dis) > tol


def sample_of_label(label, numu_events=None):
    """Coarse sample name for reporting: nuecc / ncpi0 / mcp1k (data);
    'numu50' if the event is in the provided numu_events set."""
    tag = label.get('scan_tag') or ''
    if 'ncpi0' in tag:
        return 'ncpi0'
    if 'mcp1k' in tag:
        if numu_events and label['eventNo'] in numu_events:
            return 'numu50'
        return 'mcp1k'
    return 'nuecc'


def default_sbnd_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def default_weights():
    """The production uBooNE checkpoint, resolved like WIRECELL_PATH would."""
    for base in os.environ.get('WIRECELL_PATH', '').split(':'):
        if not base:
            continue
        p = os.path.join(base, 'uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth')
        if os.path.exists(p):
            return p
    return '/nfs/data/1/xqian/toolkit-dev/wire-cell-data/uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth'

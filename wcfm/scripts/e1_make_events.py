#!/usr/bin/env python3
"""wcfm E1 (doc 07): build workspace events 101+ from the extracted WC_FM_Sim pilot depos.

Input:  wcfm/work/e1/depos-<flavor>.tar.bz2 (e1/run_extract.sh): depo_data_N / depo_info_N per
        art event (N 0-based), un-drifted G4 depos in WCT units (t ns, q electrons < 0, xyz mm),
        info = (G4 track id, pdg, gen, child).
Output: events/000001_<evt>.json (same keys the chain reads: kind, tracks[0].{tail,head,charge,
        angle_deg,length_cm}, anodes, seed, step_mm; plus source/flavor/depo_file) and
        work/000001_<evt>/e1-depos-in.tar.bz2 (one depo_data_0 / depo_info_0 pair, gen 0).
Strata: 101-120 numu natural, 121-140 nue natural (kind 'nu'), 201-220 numu rotated so that the
        primary muon is exactly parallel to the anode (0 deg), 301-320 numu rotated to 0.5 deg
        (kind 'nurot').  The rotation is rigid, about the primary vertex, about the axis
        x_hat x d_mu (the minimal rotation that removes the muon's drift-direction component);
        G4 physics is rotation invariant, so the texture and landmarks are real.
Never overwrites an existing event json; events < 101 are refused.
"""
import argparse, io, json, os, sys, tarfile
import numpy as np

WCFM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = 1
# active volume of the FD-HD 1x2x6 (cm): wires x 3.00155 cm from the APA centre line, cathode
# 362.916 cm; y +-600.019; z -0.876 .. 1393.46 (dune10kt-1x2x6/simparams.jsonnet, wcfm_params).
X_MIN_ABS, X_MAX = 3.0, 362.5
Y_MAX = 600.0
Z_MIN, Z_MAX = 0.0, 1393.0
Z_PITCH_CM = 232.39           # wcfm_params z_pitch 2323.9 mm; ident = 2*col + row(y>0)
ANODE_QMIN = 5e5              # electrons an APA must hold to be simulated (~10 cm of MIP)

STRATA = [  # (first event, flavor, rotation target in deg or None, kind)
    (101, 'numu', None, 'nu'),
    (121, 'nue', None, 'nu'),
    (201, 'numu', 0.0, 'nurot'),
    (301, 'numu', 0.5, 'nurot'),
]


def read_depo_file(path):
    sets = {}
    with tarfile.open(path, 'r:*') as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if not name.endswith('.npy'):
                continue
            parts = name[:-4].split('_')      # depo, data|info, N
            n = int(parts[-1]); what = parts[-2]
            sets.setdefault(n, {})[what] = np.load(io.BytesIO(tf.extractfile(m).read()))
    return {n: (v['data'], v['info']) for n, v in sorted(sets.items())}


def write_depo_file(path, data, info):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, 'w:bz2') as tf:
        for name, arr in (('depo_data_0.npy', data.astype(np.float32)),
                          ('depo_info_0.npy', info.astype(np.int32))):
            b = io.BytesIO(); np.save(b, arr); b.seek(0)
            ti = tarfile.TarInfo(name); ti.size = len(b.getvalue())
            tf.addfile(ti, b)


def anode_ident(y_cm, z_cm):
    return 2 * np.floor(z_cm / Z_PITCH_CM).astype(int) + (y_cm > 0).astype(int)


def primary(info, data, pdgs):
    """track id of the primary among |pdg| in pdgs: the smallest G4 id (primaries come first)."""
    sel = np.isin(np.abs(info[:, 1]), pdgs) & (info[:, 0] > 0)   # negative ids = dropped EM daughters
    if not sel.any():
        return None
    ids = info[sel, 0]
    return int(ids.min())


def track_geometry(data, info, tid):
    """tail/head (cm) along time order, path length (cm), unit direction, charge."""
    sel = info[:, 0] == tid
    d = data[sel]
    order = np.argsort(d[:, 0], kind='stable')
    xyz = d[order, 2:5].astype(np.float64) / 10.0
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    length = float(seg.sum())
    tail, head = xyz[0], xyz[-1]
    v = head - tail
    n = np.linalg.norm(v)
    dhat = v / n if n > 0 else np.array([0.0, 0.0, 1.0])
    return tail, head, length, dhat, float(np.abs(d[:, 1]).sum())


def rodrigues(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def rotation_to_anode_angle(dhat, target_deg):
    """R such that asin(|(R dhat)_x|) = target_deg, minimal rotation, axis x_hat x dhat."""
    xhat = np.array([1.0, 0.0, 0.0])
    axis = np.cross(xhat, dhat)
    if np.linalg.norm(axis) < 1e-9:      # muon along the drift axis: pick any perpendicular
        axis = np.array([0.0, 0.0, 1.0])
    theta = np.arcsin(np.clip(dhat[0], -1, 1))
    want = np.sign(dhat[0] if dhat[0] != 0 else 1.0) * np.radians(target_deg)
    best = None
    for ang in (theta - want, want - theta, -(theta - want), -(want - theta)):
        R = rodrigues(axis, ang)
        got = np.arcsin(np.clip((R @ dhat)[0], -1, 1))
        err = abs(got - want)
        if best is None or err < best[0]:
            best = (err, R, ang)
    err, R, ang = best
    assert err < 1e-6, (err, np.degrees(theta), target_deg)
    return R, float(np.degrees(ang))


def contained(xyz_cm):
    x, y, z = xyz_cm[:, 0], xyz_cm[:, 1], xyz_cm[:, 2]
    return (np.abs(x) >= X_MIN_ABS) & (np.abs(x) <= X_MAX) & (np.abs(y) <= Y_MAX) & (z >= Z_MIN) & (z <= Z_MAX)


def build(evt, flavor, art_index, data, info, target_deg, kind, args):
    pdgs = (13,) if flavor == 'numu' else (11,)
    tid = primary(info, data, pdgs)
    if tid is None:
        return None, 'no primary %s' % pdgs
    tail, head, length, dhat, qmu = track_geometry(data, info, tid)
    theta_nat = float(np.degrees(np.arcsin(abs(dhat[0]))))
    d2 = data.copy()
    rot = None
    if target_deg is not None:
        R, ang = rotation_to_anode_angle(dhat, target_deg)
        v = tail                                            # vertex = first primary depo
        p = data[:, 2:5].astype(np.float64) / 10.0
        p2 = (p - v) @ R.T + v
        d2[:, 2:5] = (p2 * 10.0).astype(np.float32)
        rot = {'axis': 'xhat x d_mu', 'angle_deg': ang, 'about_cm': [float(c) for c in v]}
        tail2, head2, length2, dhat2, _ = track_geometry(d2, info, tid)
        angle = float(np.degrees(np.arcsin(abs(dhat2[0]))))
        assert abs(angle - target_deg) < 1e-3, (angle, target_deg)
    else:
        angle = theta_nat
    keep = contained(d2[:, 2:5] / 10.0) & (d2[:, 1] != 0)
    n_all, n_keep = len(d2), int(keep.sum())
    mu_sel = info[:, 0] == tid
    q_mu_all = float(np.abs(d2[mu_sel, 1]).sum())
    q_mu_keep = float(np.abs(d2[mu_sel & keep, 1]).sum())
    kept_frac = q_mu_keep / q_mu_all if q_mu_all > 0 else 0.0
    d3, i3 = d2[keep], info[keep].copy()
    i3[:, 2] = 0; i3[:, 3] = 0
    if len(d3) == 0:
        return None, 'nothing contained'
    # track summary after rotation/containment (primary only, kept depos)
    tail_k, head_k, length_k, dhat_k, _ = track_geometry(d3, i3, tid) if (i3[:, 0] == tid).any() else (tail, head, 0.0, dhat, 0)
    # anodes with enough charge, plus every anode the primary touches
    ids = anode_ident(d3[:, 3] / 10.0, d3[:, 4] / 10.0)
    qa = {}
    for a, q in zip(ids, np.abs(d3[:, 1])):
        qa[int(a)] = qa.get(int(a), 0.0) + float(q)
    anodes = sorted(a for a, q in qa.items() if q >= ANODE_QMIN or a in set(int(x) for x in ids[i3[:, 0] == tid]))
    anodes = [a for a in anodes if 0 <= a < 12]
    t_us = d3[:, 0] / 1000.0
    pdg_counts = {int(k): int(v) for k, v in zip(*np.unique(i3[:, 1], return_counts=True))}
    ev = {
        'run': RUN, 'event': evt, 'seed': 1000 * RUN + evt, 'kind': kind, 'step_mm': None,
        'flavor': flavor,
        'source': {'file': os.path.relpath(args.indir, WCFM) + '/depos-%s.tar.bz2' % flavor,
                   'set_index': art_index, 'art_event': art_index + 1,
                   'rotation': rot, 'theta_natural_deg': round(theta_nat, 4)},
        'tracks': [{
            'tail': [round(float(c), 3) for c in tail_k], 'head': [round(float(c), 3) for c in head_k],
            'charge': None, 'angle_deg': (round(angle, 4) if flavor == 'numu' else None),
            'length_cm': round(length_k, 2), 'length_full_cm': round(length, 2),
            'pdg': int(info[mu_sel, 1][0]), 'track_id': tid, 'kept_frac': round(kept_frac, 4),
        }],
        'anodes': anodes,
        'ndepos': n_keep, 'ndepos_extracted': n_all,
        'q_sum': float(d3[:, 1].sum()),
        't_range_us': [round(float(t_us.min()), 3), round(float(t_us.max()), 3)],
        'pdg_counts': pdg_counts,
        'depo_file': 'work/%06d_%d/e1-depos-in.tar.bz2' % (RUN, evt),
    }
    return (ev, d3, i3), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', default=os.path.join(WCFM, 'work', 'e1'))
    ap.add_argument('--out-table', default=os.path.join(WCFM, 'docs', '07_tables', 'events.md'))
    ap.add_argument('--dry-run', action='store_true', help='print the table, write nothing')
    ap.add_argument('--strata', default='all', help='comma list of first-event numbers to build')
    args = ap.parse_args()
    sets = {}
    rows = []
    for first, flavor, target, kind in STRATA:
        if args.strata != 'all' and str(first) not in args.strata.split(','):
            continue
        if flavor not in sets:
            sets[flavor] = read_depo_file(os.path.join(args.indir, 'depos-%s.tar.bz2' % flavor))
        for k, (n, (data, info)) in enumerate(sorted(sets[flavor].items())):
            evt = first + k
            assert evt >= 101, evt
            js = os.path.join(WCFM, 'events', '%06d_%d.json' % (RUN, evt))
            if os.path.exists(js) and not args.dry_run:
                print('refusing to overwrite', js, file=sys.stderr); sys.exit(2)
            built, why = build(evt, flavor, n, data, info, target, kind, args)
            if built is None:
                rows.append((evt, flavor, kind, target, 'SKIP: ' + why)); continue
            ev, d3, i3 = built
            tr = ev['tracks'][0]
            rows.append((evt, flavor, kind, target, len(d3), ev['q_sum'], ev['t_range_us'],
                         ev['source']['theta_natural_deg'], tr['angle_deg'], tr['length_cm'],
                         tr['kept_frac'], ev['anodes'], round(ev['source']['rotation']['angle_deg'], 3) if ev['source']['rotation'] else 0.0))
            if not args.dry_run:
                write_depo_file(os.path.join(WCFM, ev['depo_file']), d3, i3)
                with open(js, 'w') as f:
                    json.dump(ev, f, indent=1)
    lines = ['| event | flavor | kind | rot target | ndepos kept | q sum (e) | t range (us) | theta natural (deg) | angle_deg | primary length kept (cm) | primary kept frac | anodes | rotation applied (deg) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if len(r) == 5:
            lines.append('| %d | %s | %s | %s | %s | | | | | | | | |' % r)
        else:
            lines.append('| %d | %s | %s | %s | %d | %.3g | %s | %.3f | %s | %.1f | %.3f | %s | %.3f |' % r)
    txt = '\n'.join(lines) + '\n'
    print(txt)
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.out_table), exist_ok=True)
        open(args.out_table, 'w').write(txt)


if __name__ == '__main__':
    main()

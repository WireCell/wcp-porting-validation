#!/usr/bin/env python3
"""pdhd/docs/01_steiner-wrapped-planes.md: the plane-multiplicity census of the
RETILED cluster the Steiner stage reads, on any detector, from the env-gated
probe WCT_STEINER_PHASE_DUMP=1 (clus/src/SteinerGrapher.cxx, `steiner_phase_pt`
lines with phase=P0_cluster: every point of every >1000-point retiled cluster
with its three per-plane charges).

Why this exists.  Cluster::calc_charge_wcp (Facade_Cluster.cxx:1031-1112,
disable_dead_mix_cell=false, the only Steiner call site) returns charge 0
unless MORE THAN ONE plane carries non-zero charge, so `ncharge > 1` is the
ceiling of the terminal population at ANY threshold.  stm-tagger-chain.md sec 12
measured ncharge=3 == 0.000 on PDHD with an untracked scratch parser; this is
the tracked, three-detector replacement.

What it reports per label (an arm = one or more logs):
  * the ncharge histogram (0/1/2/3) and the seven plane combinations;
  * per plane: fraction with q == 0, median of the non-zero charges;
  * --thr T [--validate]: reproduces calc_charge_wcp's candidate predicate at
    threshold T and compares the count with the run's own `steiner_p1_blobs:
    ncand_pt` counter -- the sec 12.2 positive control (must be exact);
  * --wires FILE --det pdhd: the nearest-wire SEGMENT of each point's U and V
    wire, computed from the wires file (the dump carries no wire index before
    the round-2 probe extension), so P(q != 0 | segment) and the joint
    (segU, segV) table can be read -- the discriminating test for the wrapped-
    plane mechanism -- plus the uniform-face geometric prior of (segU, segV);
  * when the extended probe fields are present (wu wv ww uu uv uw ts nu nv nw,
    appended by the round-2 SteinerGrapher.cxx change): the split of q == 0
    into dead (unc > 1e10) / neighbour-recoverable (max ctpc charge within +-k
    wires in the same slice > 0) / empty, per plane.

Usage (from wcp-porting-img/pdhd):
  python3 docs/scripts/d01_steiner_plane_multiplicity.py \
      "PDHD prod:work/029107_*_phdump/wct_pr_*.log" \
      "PDVD:../pdvd/work/039252_2_d31r6e2e/wct_pr_039252_2.log" \
      "SBND:../sbnd/sbnd_xin/work-d31r7probe2/pr_evt*/wct_pr_evt*.log" \
      --thr 500 --validate --tsv docs/figs/d01_plane_multiplicity.tsv
  python3 docs/scripts/d01_steiner_plane_multiplicity.py \
      "retiler on:work/029107_0_phdumpw/wct_pr_029107_0.log" \
      --wires ../../wire-cell-data/protodunehd-wires-larsoft-v1.json.bz2 --det pdhd

Face assignment for --det pdhd: the one SENSITIVE face per APA (params.jsonnet
nulls the wall face): even anode ident -> face 0, odd -> face 1; the anode is
chosen by sign(x) (x < 0 -> anodes 0/2) and z (z < 231.5 cm -> anodes 0/1).
The nearest wire is the argmin of |pitch coordinate difference|, the same rule
BlobSampler uses (pimpos->closest), without its +0.1 mm tie shift.
"""
import argparse
import bz2
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict

import numpy as np

PT_RE = re.compile(
    r'steiner_phase_pt: npts=(\d+) nterm=(\d+) phase=(\S+) x=(\S+) y=(\S+) z=(\S+)'
    r' cu=(\S+) cv=(\S+) cw=(\S+)')
EXT_RE = re.compile(
    r' wu=(-?\d+) wv=(-?\d+) ww=(-?\d+) uu=(\S+) uv=(\S+) uw=(\S+) ts=(-?\d+)'
    r' nu=(\S+) nv=(\S+) nw=(\S+)')
P1_RE = re.compile(r'steiner_p1_blobs: nblob=(\d+) ncand_blob=(\d+) nterm=(\d+) npt=(\d+) ncand_pt=(\d+)')
COMP_RE = re.compile(r'<([A-Za-z0-9_]+:[A-Za-z0-9_]+)>')   # spdlog component tag, e.g. <CreateSteinerGraph:pr>
DEAD_UNC = 1e10   # Facade_Cluster.cxx:1036 dead_threshold


def calc_charge_wcp(cu, cv, cw, thr):
    """Facade_Cluster.cxx:1031-1112, disable_dead_mix_cell=false branch."""
    flags = [(q > thr) or (q == 0) for q in (cu, cv, cw)]
    nz = [q for q in (cu, cv, cw) if q != 0]
    charge = math.sqrt(sum(q * q for q in nz) / len(nz)) if len(nz) > 1 else 0.0
    return all(flags) and charge > thr


def parse_logs(files):
    """Return (points, p1) where points is a list of dicts (P0_cluster only)
    and p1 the list of (npt, ncand_pt) counters."""
    pts = []
    p1 = []       # (component, npt, ncand_pt) -- ImproveCluster_1 and CreateSteinerGraph both emit it
    comps = set() # components that produced P0_cluster lines
    for f in files:
        with open(f, errors='replace') as fh:
            for line in fh:
                m = PT_RE.search(line)
                if m:
                    if m.group(3) != 'P0_cluster':
                        continue
                    c = COMP_RE.search(line)
                    comps.add(c.group(1) if c else '')
                    d = dict(x=float(m.group(4)), y=float(m.group(5)), z=float(m.group(6)),
                             cu=float(m.group(7)), cv=float(m.group(8)), cw=float(m.group(9)))
                    e = EXT_RE.search(line, m.end())
                    if e:
                        d.update(wu=int(e.group(1)), wv=int(e.group(2)), ww=int(e.group(3)),
                                 uu=float(e.group(4)), uv=float(e.group(5)), uw=float(e.group(6)),
                                 ts=int(e.group(7)),
                                 nu=float(e.group(8)), nv=float(e.group(9)), nw=float(e.group(10)))
                    pts.append(d)
                    continue
                m = P1_RE.search(line)
                if m:
                    c = COMP_RE.search(line)
                    rec = (c.group(1) if c else '', int(m.group(4)), int(m.group(5)))
                    # CreateSteinerGraph runs find_steiner_terminals TWICE per
                    # cluster with the same flags (establish_same_blob_steiner_edges
                    # at CreateSteinerGraph.cxx:299, then create_steiner_tree at
                    # :323), so the counter line repeats verbatim; count it once.
                    if p1 and p1[-1] == rec:
                        continue
                    p1.append(rec)
    # Keep only the counters of the component(s) whose points were dumped.
    p1 = [(npt, nc) for comp, npt, nc in p1 if comp in comps]
    return pts, p1


# ---------------------------------------------------------------- geometry
class Wires:
    """Per (anode, face, plane): wires sorted by pitch coordinate with their
    segment and channel; the pitch axis is the in-plane normal to the wire
    direction (y-z plane), the same axis pimpos uses."""

    def __init__(self, path):
        s = json.load(bz2.open(path))['Store']
        A = [a['Anode'] for a in s['anodes']]
        F = [f['Face'] for f in s['faces']]
        P = [p['Plane'] for p in s['planes']]
        W = [w['Wire'] for w in s['wires']]
        X = [p['Point'] for p in s['points']]
        pt = lambda i: np.array([X[i]['x'], X[i]['y'], X[i]['z']]) / 10.0   # mm -> cm
        self.planes = {}
        self.anode_ident = [a['ident'] for a in A]
        for ai, a in enumerate(A):
            for fi, fid in enumerate(a['faces']):
                for pi, pid in enumerate(F[fid]['planes']):
                    ws = [W[i] for i in P[pid]['wires']]
                    t = np.array([pt(w['tail']) for w in ws])
                    h = np.array([pt(w['head']) for w in ws])
                    mid = (t + h) / 2
                    d = (h - t)[0].copy(); d[0] = 0; d /= np.linalg.norm(d)
                    pitch = np.array([0.0, -d[2], d[1]])
                    pc = mid @ pitch
                    o = np.argsort(pc)
                    self.planes[(ai, fi, pi)] = dict(
                        pitch=pitch, pc=pc[o],
                        seg=np.array([w['segment'] for w in ws])[o],
                        ch=np.array([w['channel'] for w in ws])[o],
                        y0=float(min(t[:, 1].min(), h[:, 1].min())), y1=float(max(t[:, 1].max(), h[:, 1].max())),
                        z0=float(min(t[:, 2].min(), h[:, 2].min())), z1=float(max(t[:, 2].max(), h[:, 2].max())))

    def nearest(self, key, y, z):
        pl = self.planes[key]
        q = pl['pitch'][1] * y + pl['pitch'][2] * z
        i = int(np.searchsorted(pl['pc'], q))
        i = min(max(i, 0), len(pl['pc']) - 1)
        if i > 0 and abs(pl['pc'][i - 1] - q) < abs(pl['pc'][i] - q):
            i -= 1
        return i


def pdhd_anode_face(x, z):
    """Sensitive face per APA (pdhd params.jsonnet: even anode -> face 0)."""
    ai = (0 if x < 0 else 1) + (0 if z < 231.5 else 2)
    return ai, (0 if ai % 2 == 0 else 1)


# ---------------------------------------------------------------- reports
def census(label, pts, p1, thr, validate, ext_k):
    n = len(pts)
    if n == 0:
        print(f'== {label}: no P0_cluster points')
        return None
    nc = Counter(); combo = Counter(); q0 = Counter(); nzq = defaultdict(list)
    for d in pts:
        z = [d['cu'] != 0, d['cv'] != 0, d['cw'] != 0]
        nc[sum(z)] += 1
        combo[''.join(l for l, f in zip('UVW', z) if f) or '-'] += 1
        for l, k in zip('UVW', ('cu', 'cv', 'cw')):
            if d[k] == 0:
                q0[l] += 1
            else:
                nzq[l].append(d[k])
    print(f'== {label}: n={n} points, {len(p1)} create_steiner_tree calls with p1 counters')
    print('   ncharge : ' + '  '.join(f'{k}={nc[k]/n:.3f}' for k in range(4)) + f'   eligible(>1)={(nc[2]+nc[3])/n:.3f}')
    print('   combos  : ' + '  '.join(f'{k}={v/n:.3f}' for k, v in combo.most_common()))
    print('   q==0    : ' + '  '.join(f'{l}={q0[l]/n:.3f}' for l in 'UVW')
          + '   median nonzero: ' + '  '.join(f'{l}={np.median(nzq[l]):.0f}' if nzq[l] else f'{l}=nan' for l in 'UVW'))
    row = dict(label=label, n=n, **{f'nc{k}': nc[k] / n for k in range(4)},
               eligible=(nc[2] + nc[3]) / n, **{f'q0_{l}': q0[l] / n for l in 'UVW'},
               **{f'combo_{k}': combo[k] / n for k in ('UVW', 'UV', 'UW', 'VW', 'U', 'V', 'W', '-')})
    if thr is not None:
        ncand = sum(calc_charge_wcp(d['cu'], d['cv'], d['cw'], thr) for d in pts)
        row['cand_frac'] = ncand / n
        msg = f'   cand@{thr:g}: {ncand} = {ncand/n:.3f} of dumped points'
        if validate:
            # The dump is restricted to >1000-point clusters; the p1 counters cover
            # every call.  Compare on the calls whose npt is > 1000 only.
            big = [(npt, nc_) for npt, nc_ in p1 if npt > 1000]
            npt_sum = sum(a for a, _ in big); nc_sum = sum(c for _, c in big)
            # The dump prints charges with 0.1 e precision, so a point within
            # 0.05 e of the threshold can flip offline; the counter itself is exact.
            delta = ncand - nc_sum
            ok = (npt_sum == n and abs(delta) <= max(1, int(1e-3 * max(nc_sum, 1))))
            msg += (f' | control: same-component ncand_pt over calls with npt>1000 = {nc_sum}'
                    f' on {npt_sum} points, offline - counter = {delta:+d} -> {"EXACT" if delta == 0 and npt_sum == n else ("OK (print precision)" if ok else "MISMATCH")}')
            row['control_delta'] = delta; row['control_npt_match'] = int(npt_sum == n)
        print(msg)
    if ext_k is not None and 'wu' in pts[0]:
        print(f'   extended probe: q==0 split per plane.  sentinel = unc>{DEAD_UNC:g} (the retiler wrote (0,1e12): a dead'
              f' channel or a path-forced wire, improvecluster_1.cxx); nb = a ctpc wire within |dw|<={ext_k} in the same slice has charge')
        for l, qk, uk, nk in (('U', 'cu', 'uu', 'nu'), ('V', 'cv', 'uv', 'nv'), ('W', 'cw', 'uw', 'nw')):
            zero = [d for d in pts if d[qk] == 0]
            if not zero:
                continue
            nz = len(zero)
            s_nb = sum(d[uk] > DEAD_UNC and d[nk] > 0 for d in zero)
            s_no = sum(d[uk] > DEAD_UNC and d[nk] <= 0 for d in zero)
            l_nb = sum(d[uk] <= DEAD_UNC and d[nk] > 0 for d in zero)
            l_no = nz - s_nb - s_no - l_nb
            print(f'     {l}: zero={nz/n:.3f}  sentinel+nb={s_nb/nz:.3f}  sentinel-no-nb={s_no/nz:.3f}'
                  f'  live+nb={l_nb/nz:.3f}  live-no-nb(empty)={l_no/nz:.3f}')
            row[f'zero_{l}'] = nz / n; row[f'sent_nb_{l}'] = s_nb / nz; row[f'sent_nonb_{l}'] = s_no / nz
            row[f'live_nb_{l}'] = l_nb / nz; row[f'empty_{l}'] = l_no / nz
        # Upper bound of a "nearby wires" fallback: a zero plane takes its
        # neighbour max (sentinel or not) -- how many points become eligible?
        elig0 = sum(((d['cu'] != 0) + (d['cv'] != 0) + (d['cw'] != 0)) > 1 for d in pts)
        elig1 = sum(((d['cu'] or d['nu']) != 0) + ((d['cv'] or d['nv']) != 0) + ((d['cw'] or d['nw']) != 0) > 1 for d in pts)
        print(f'     eligible (ncharge>1): {elig0/n:.3f} -> {elig1/n:.3f} if every zero plane took its +-{ext_k}-wire ctpc neighbour max')
        row['eligible_nb'] = elig1 / n
    return row


def segment_attribution(label, pts, wires, det):
    if det != 'pdhd':
        print(f'   (segment attribution implemented for pdhd only; skipped for {det})')
        return
    n = len(pts)
    segU = Counter(); nzU = Counter(); segV = Counter(); nzV = Counter(); joint = Counter(); jointnz = Counter()
    for d in pts:
        ai, fi = pdhd_anode_face(d['x'], d['z'])
        su = int(wires.planes[(ai, fi, 0)]['seg'][wires.nearest((ai, fi, 0), d['y'], d['z'])])
        sv = int(wires.planes[(ai, fi, 1)]['seg'][wires.nearest((ai, fi, 1), d['y'], d['z'])])
        segU[su] += 1; segV[sv] += 1; joint[(su, sv)] += 1
        if d['cu'] != 0: nzU[su] += 1
        if d['cv'] != 0: nzV[sv] += 1
        if d['cu'] != 0 and d['cv'] != 0: jointnz[(su, sv)] += 1
    print(f'   segment attribution ({label}):')
    print('     U: ' + '  '.join(f'seg{k}: share={segU[k]/n:.3f} P(cu!=0)={nzU[k]/segU[k]:.3f}' for k in sorted(segU)))
    print('     V: ' + '  '.join(f'seg{k}: share={segV[k]/n:.3f} P(cv!=0)={nzV[k]/segV[k]:.3f}' for k in sorted(segV)))
    print('     joint (segU,segV): ' + '  '.join(f'{k}: {joint[k]/n:.3f}/{jointnz[k]/joint[k]:.3f}' for k in sorted(joint)))
    # Geometric prior: uniform points on anode 0's sensitive face.
    rng = np.random.default_rng(1)
    pl = wires.planes[(0, 0, 0)]
    ys = rng.uniform(pl['y0'], pl['y1'], 100000); zs = rng.uniform(pl['z0'], pl['z1'], 100000)
    jc = Counter()
    for y, z in zip(ys, zs):
        jc[(int(wires.planes[(0, 0, 0)]['seg'][wires.nearest((0, 0, 0), y, z)]),
            int(wires.planes[(0, 0, 1)]['seg'][wires.nearest((0, 0, 1), y, z)]))] += 1
    print('     uniform-face prior (segU,segV): ' + '  '.join(f'{k}: {v/100000:.3f}' for k, v in sorted(jc.items())))
    return dict(label=label, **{f'U{k}_share': segU[k] / n for k in sorted(segU)},
                **{f'U{k}_nz': nzU[k] / segU[k] for k in sorted(segU)},
                **{f'V{k}_share': segV[k] / n for k in sorted(segV)},
                **{f'V{k}_nz': nzV[k] / segV[k] for k in sorted(segV)},
                **{f'J{a}{b}_share': joint[(a, b)] / n for (a, b) in sorted(joint)},
                **{f'J{a}{b}_bothnz': jointnz[(a, b)] / joint[(a, b)] for (a, b) in sorted(joint)},
                **{f'prior{a}{b}': v / 100000 for (a, b), v in sorted(jc.items())})


def pair_coverage(label, pts, wires, det):
    """Tier-2 feasibility (doc sec 6/7): for each dumped point whose U (V)
    wire belongs to a channel with TWO segments on the sensitive face, is the
    channel's OTHER segment also under a dumped point in the same slice?  If it
    is, the ident-resolved lookup credits both points with the whole channel
    activity (the ghost ambiguity is live); if not, the activity is unambiguous.
    Dumped points come only from >1000-point clusters, so this is a LOWER bound
    on the coverage rate.  Needs the extended probe fields (wu wv ts)."""
    if det != 'pdhd' or not pts or 'wu' not in pts[0]:
        return None
    # (anode, face, plane) -> {wire index: (channel, partner wire index or None)}
    partner = {}
    for key, pl in wires.planes.items():
        by_ch = defaultdict(list)
        for i, ch in enumerate(pl['ch'].tolist()):
            by_ch[ch].append(i)
        partner[key] = {i: (ch, [j for j in by_ch[ch] if j != i]) for ch, idxs in by_ch.items() for i in idxs}
    occupied = defaultdict(set)   # (anode, face, plane, ts) -> wire indices under a dumped point
    recs = []
    for d in pts:
        ai, fi = pdhd_anode_face(d['x'], d['z'])
        for plane, wk in ((0, 'wu'), (1, 'wv')):
            occupied[(ai, fi, plane, d['ts'])].add(d[wk])
            recs.append((ai, fi, plane, d['ts'], d[wk], d['cu' if plane == 0 else 'cv']))
    out = {}
    for plane, letter in ((0, 'U'), (1, 'V')):
        n = two = covered = covered_nz = 0
        for ai, fi, pl, ts, w, q in recs:
            if pl != plane:
                continue
            n += 1
            ch, others = partner[(ai, fi, plane)].get(w, (None, []))
            if not others:
                continue
            two += 1
            if any(o in occupied[(ai, fi, plane, ts)] for o in others):
                covered += 1
                covered_nz += (q != 0)
        print(f'   two-segment coverage ({label}) {letter}: points={n} on a two-segment channel={two/n:.3f}'
              f' partner also under a dumped point in the same slice={covered/n:.3f} (of which q!=0: {covered_nz/max(covered,1):.3f})')
        out[f'two_seg_{letter}'] = two / n; out[f'pair_covered_{letter}'] = covered / n
    return out


def geometry_census(wires):
    """The wrap topology the doc's sec 3 quotes: per plane the segment stripes
    in pitch order, and per APA how many U/V channels have 0/1/2 segments on the
    SENSITIVE face (the wall face is nulled in pdhd params.jsonnet)."""
    print('geometry census (segments in pitch order as (segment, run length)):')
    for key in sorted(wires.planes):
        seg = wires.planes[key]['seg']; runs = []; cur = int(seg[0]); n = 0
        for sg in seg:
            if int(sg) == cur:
                n += 1
            else:
                runs.append((cur, n)); cur = int(sg); n = 1
        runs.append((cur, n))
        nch = len(set(wires.planes[key]['ch'].tolist()))
        print(f'  anode{key[0]} face{key[1]} plane{key[2]}: nwires={len(seg)} nchannels={nch} runs={runs}')
    anodes = sorted(set(k[0] for k in wires.planes))
    for ai in anodes:
        fi = 0 if ai % 2 == 0 else 1
        for pi in (0, 1):
            segs_total = Counter(); segs_sens = Counter()
            for f in (0, 1):
                for ch, sg in zip(wires.planes[(ai, f, pi)]['ch'], wires.planes[(ai, f, pi)]['seg']):
                    segs_total[int(ch)] += 1
                    if f == fi:
                        segs_sens[int(ch)] += 1
            hist = Counter((segs_total[c], segs_sens.get(c, 0)) for c in segs_total)
            print(f'  anode{ai} sensitive face{fi} plane{pi}: channels by (segments total, on sensitive face): '
                  + '  '.join(f'{k}:{v}' for k, v in sorted(hist.items())))


def write_tsv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w') as f:
        f.write('\t'.join(keys) + '\n')
        for r in rows:
            f.write('\t'.join(f'{r[k]:.4f}' if isinstance(r.get(k), float) else str(r.get(k, '')) for k in keys) + '\n')
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arms', nargs='*', help='"label:glob" (glob may match several logs)')
    ap.add_argument('--thr', type=float, default=None, help='terminal_charge_threshold of the arm (e)')
    ap.add_argument('--validate', action='store_true', help='compare with steiner_p1_blobs ncand_pt')
    ap.add_argument('--wires', default=None, help='wires file (json.bz2) for the segment attribution')
    ap.add_argument('--det', default='pdhd')
    ap.add_argument('--ext-k', type=int, default=2, help='neighbour half-width the probe used')
    ap.add_argument('--tsv', default=None)
    ap.add_argument('--seg-tsv', default=None)
    ap.add_argument('--geometry', action='store_true', help='print the wrap-topology census of --wires and continue')
    a = ap.parse_args()
    wires = Wires(a.wires) if a.wires else None
    if a.geometry and wires:
        geometry_census(wires)
    rows = []; seg_rows = []
    for arm in a.arms:
        label, pat = arm.split(':', 1)
        files = sorted(glob.glob(pat))
        if not files:
            print(f'== {label}: no files match {pat}', file=sys.stderr)
            continue
        pts, p1 = parse_logs(files)
        row = census(label, pts, p1, a.thr, a.validate, a.ext_k)
        if row:
            rows.append(row)
        if wires and pts:
            sr = segment_attribution(label, pts, wires, a.det)
            pc = pair_coverage(label, pts, wires, a.det)
            if sr:
                if pc:
                    sr.update(pc)
                seg_rows.append(sr)
    if a.tsv and rows:
        write_tsv(a.tsv, rows)
    if a.seg_tsv and seg_rows:
        write_tsv(a.seg_tsv, seg_rows)


if __name__ == '__main__':
    main()

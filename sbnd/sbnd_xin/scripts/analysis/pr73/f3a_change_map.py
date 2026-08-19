#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73 round 2 (F3a) -- what CHANGED, per event, in the
48-event sgp_max_sep scan set.

Read-only.  For every event of the scan set it joins three sources that are
already on disk and answers "what should I look at in Bee":

  1. the ON arm's `sgp guard: cluster N VETO ...` lines -- WHERE the guard
     acted (cluster ident, route endpoints, route length, excursion size);
  2. `nusel-evt<E>.tsv` in BOTH arms -- WHICH bundle is the nu candidate
     (main_id) and its taggers/label, used to detect a candidate SWAP (a
     dvtx between two different objects is not a vertex move at all);
  3. `tracking-pr.root` in both arms -- T_tagger nu_{x,y,z}, T_kine
     kine_reco_Enu, the per-particle kine_energy_particle /
     kine_particle_type / kine_energy_included vectors, the pi0 block
     (kine_pio_mass / kine_pio_flag, the physics observable of the NCpi0
     manifest), and T_rec_charge -- the nu candidate's OWN fitted trajectory.

The `cluster.ident() == nusel main_id` join is NOT trusted: idents are
re-enumerated after every visitor (doc 53), so a mid-chain log line and the
end-of-chain nusel dump can be different epochs.  It disagreed with geometry
on 7 of 48 events here.  Everything the classification rests on is instead
geometric against T_rec_charge, which is epoch-free.

Usage:
  f3a_change_map.py                 # whole scan set, classified
  f3a_change_map.py --movers        # the >1 cm vertex movers, slide vs relocate
  f3a_change_map.py --evt 46363     # one event, verbose decomposition
  f3a_change_map.py --tsv OUT.tsv   # dump the raw per-event columns
"""
import os, re, sys, glob
from collections import Counter
import numpy as np
import uproot

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
MAN = {'48': 'nueCC48', '19': 'NCpi0-19', '50': 'mcp1k-50'}
BEE = '/home/xqian/tmp/pr73f3a-bee'

# Movement thresholds.  "quiet" = below every one of these.
VTX_QUIET = 0.1     # cm
ENU_QUIET = 20.0    # MeV
PIO_QUIET = 1.0     # MeV



# doc pr/94 Phase 5: T_tagger/T_kine hold ONE ROW PER IN-BEAM-WINDOW BUNDLE when
# the nu_per_bundle knob is on, so a hard [0] silently reports whichever bundle
# was enumerated first.  primary_index() reproduces the legacy meaning of "the
# candidate" (longest selected main activity) and falls back to 0 for pre-pr/94
# and knob-off files.
import os as _pr94_os, sys as _pr94_sys
_pr94_sys.path.insert(0, _pr94_os.path.join(
    _pr94_os.path.dirname(_pr94_os.path.abspath(__file__)), "../.."))
from pr94_rows import primary_index  # noqa: E402

def read_nusel(arm, evt):
    p = os.path.join(SB, arm, 'pr_evt%s' % evt, 'nusel-evt%s.tsv' % evt)
    if not os.path.exists(p):
        return None, [], None
    lines = [l for l in open(p).read().splitlines() if l.strip()]
    hdr = lines[0].split()
    rows = [dict(zip(hdr, l.split())) for l in lines[1:]]
    cand = [r for r in rows if r.get('label') == 'nu-candidate']
    return (cand[0]['main_id'] if cand else None,
            [r['main_id'] for r in rows],
            cand[0] if cand else None)


FIRE_RE = re.compile(
    r"sgp guard: cluster (\d+) VETO maxsep=([\d.]+) cap=[\d.]+ "
    r"n_gap=(\d+) n_base=(\d+) detour=([-\d.]+) base_cm=([\d.]+) "
    r"first=\(([-\d.]+),([-\d.]+),([-\d.]+)\) last=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")


def read_fires(arm, evt):
    p = os.path.join(SB, arm, 'pr_evt%s' % evt, 'wct_pr_evt%s.log' % evt)
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p, errors='replace'):
        m = FIRE_RE.search(line)
        if m:
            out.append(dict(cluster=m.group(1), maxsep=float(m.group(2)),
                            n_gap=int(m.group(3)), n_base=int(m.group(4)),
                            detour=float(m.group(5)), base_cm=float(m.group(6)),
                            first=np.array([float(m.group(i)) for i in (7, 8, 9)]),
                            last=np.array([float(m.group(i)) for i in (10, 11, 12)])))
    return out


def read_root(arm, evt):
    p = os.path.join(SB, arm, 'pr_evt%s' % evt, 'tracking-pr.root')
    if not os.path.exists(p):
        return None
    try:
        f = uproot.open(p)
        t, k = f['T_tagger'], f['T_kine']
        _i = primary_index(t)
        # T_rec_charge is the nu candidate's OWN fitted trajectory -- the thing
        # drawn in Bee.  Using it instead of the nusel main_id join makes the
        # "did the guard touch the neutrino" question epoch-proof: cluster
        # idents are re-enumerated after every visitor (doc 53), so the ident
        # in a mid-chain log line need not be the ident nusel dumps at the end.
        rc = f['T_rec_charge'].arrays(
            ['x', 'y', 'z', 'q', 'flag_shower', 'particle_id'], library='np')
        d = dict(
            vtx=np.array([t['nu_x'].array()[_i], t['nu_y'].array()[_i], t['nu_z'].array()[_i]]),
            enu=float(k['kine_reco_Enu'].array()[_i]),
            add=float(k['kine_reco_add_energy'].array()[_i]),
            ep=np.asarray(k['kine_energy_particle'].array()[_i], dtype=float),
            pt=np.asarray(k['kine_particle_type'].array()[_i], dtype=int),
            inc=np.asarray(k['kine_energy_included'].array()[_i], dtype=int),
            info=np.asarray(k['kine_energy_info'].array()[_i], dtype=int),
            pio_mass=float(k['kine_pio_mass'].array()[_i]),
            pio_flag=int(k['kine_pio_flag'].array()[_i]),
            pio_e1=float(k['kine_pio_energy_1'].array()[_i]),
            pio_e2=float(k['kine_pio_energy_2'].array()[_i]),
            pio_ang=float(k['kine_pio_angle'].array()[_i]),
            traj=np.stack([rc['x'], rc['y'], rc['z']], axis=1),
            tq=np.asarray(rc['q'], dtype=float),
            tshw=np.asarray(rc['flag_shower'], dtype=int),
            tpid=np.asarray(rc['particle_id'], dtype=int),
        )
        return d
    except Exception as ex:
        print('  [skip] %s evt %s: %s' % (arm, evt, ex), file=sys.stderr)
        return None


PDG = {11: 'e', -11: 'e+', 13: 'mu', -13: 'mu+', 22: 'gamma', 211: 'pi+',
       -211: 'pi-', 111: 'pi0', 2212: 'p', 2112: 'n', 321: 'K+', -321: 'K-', 0: '-'}


def pname(p):
    return PDG.get(int(p), str(int(p)))


SIG_MEV = 20.0   # below this a kine_energy_particle entry is a shower fragment,
                 # not a particle a hand scan can see -- counting them makes
                 # every shower event look like a multiplicity change.


def sig_list(d):
    """Included particles above SIG_MEV, as (pdg, energy), energy-sorted."""
    v = [(int(t), float(e)) for t, e, i in zip(d['pt'], d['ep'], d['inc'])
         if i and float(e) >= SIG_MEV]
    return sorted(v, key=lambda x: -x[1])


def lead_str(d):
    v = sig_list(d)
    return '%s %.0f' % (pname(v[0][0]), v[0][1]) if v else '-'


def particle_diff(a, b):
    """Human summary of the >=20 MeV particle content change."""
    na = Counter(p for p, _ in sig_list(a))
    nb = Counter(p for p, _ in sig_list(b))
    parts = []
    for pdg in sorted(set(na) | set(nb), key=lambda p: -abs(p)):
        if na[pdg] != nb[pdg]:
            parts.append('%s %d->%d' % (pname(pdg), na[pdg], nb[pdg]))
    if parts:
        return ', '.join(parts)
    # same content: report the biggest single-particle energy shift
    shifts = []
    for pdg in sorted(set(na)):
        ea = sorted([e for p, e in sig_list(a) if p == pdg])
        eb = sorted([e for p, e in sig_list(b) if p == pdg])
        for x, y in zip(ea, eb):
            if abs(y - x) > 5.0:
                shifts.append((abs(y - x), '%s %.0f->%.0f MeV' % (pname(pdg), x, y)))
    if shifts:
        shifts.sort(reverse=True)
        return ', '.join(s for _, s in shifts[:2])
    return ''


def endpoint_flip(r):
    """True when the vertex jumped from one end of a vetoed route to the other
    -- i.e. the guard re-decided which end of a long track the neutrino is on.
    """
    a, b = r.get('a'), r.get('b')
    if not a or not b or r.get('dvtx', 0) < 10.0:
        return False
    # cluster-agnostic on purpose: the ident in the log line and the ident in
    # nusel can belong to different enumeration epochs (doc 53), and geometry
    # does not care.
    for f in r['fires']:
        for p, q in ((f['first'], f['last']), (f['last'], f['first'])):
            if (np.linalg.norm(a['vtx'] - p) < 3.0
                    and np.linalg.norm(b['vtx'] - q) < 3.0):
                return True
    return False


def load_order():
    order = []
    for n in ('48', '19', '50'):
        p = os.path.join(BEE, 'evts%s.txt' % n)
        if os.path.exists(p):
            order += [(e, n) for e in open(p).read().split()]
    if not order:   # fall back to the committed index
        p = os.path.join(SB, 'docs/pr/pr73f3a-movers.index.txt')
        for line in open(p):
            m = re.match(r"^(\d+)\s+(\d+)\s+(\S+)\s", line)
            if m:
                inv = {v: k for k, v in MAN.items()}
                order.append((m.group(2), inv[m.group(3)]))
    return order


def collect():
    rows = []
    for idx, (e, n) in enumerate(load_order()):
        off, on = 'work-pr73f3a-off%s' % n, 'work-pr73f3a-on%s' % n
        coff, alloff, roff = read_nusel(off, e)
        con, allon, ron = read_nusel(on, e)
        fs = read_fires(on, e)
        a, b = read_root(off, e), read_root(on, e)
        c = Counter(f['cluster'] for f in fs)
        onnu = [f for f in fs if f['cluster'] == con]
        offnu = [f for f in fs if f['cluster'] != con]
        r = dict(idx=idx, evt=e, man=MAN[n], cand_off=coff, cand_on=con,
                 n_fire=len(fs), n_fire_nu=len(onnu), n_fire_other=len(offnu),
                 other_clusters=sorted(set(f['cluster'] for f in offnu)),
                 orphan=[k for k in c if k not in allon],
                 maxsep_nu=max([f['maxsep'] for f in onnu], default=0.0),
                 maxsep_other=max([f['maxsep'] for f in offnu], default=0.0),
                 base_cm=[min([f['base_cm'] for f in onnu], default=0.0),
                          max([f['base_cm'] for f in onnu], default=0.0)],
                 row_off=roff, row_on=ron, a=a, b=b, fires=fs)
        if a and b:
            r['dvtx'] = float(np.linalg.norm(a['vtx'] - b['vtx']))
            r['denu'] = b['enu'] - a['enu']
            r['dpio'] = (b['pio_mass'] - a['pio_mass']
                         if (a['pio_flag'] or b['pio_flag']) else 0.0)
            r['pio_flag'] = (a['pio_flag'], b['pio_flag'])
            r['pdiff'] = particle_diff(a, b)
            r['d_vtx_fire'] = min(
                [min(np.linalg.norm(f['first'] - b['vtx']),
                     np.linalg.norm(f['last'] - b['vtx'])) for f in fs],
                default=float('nan'))
            r.update(traj_delta(a, b, fs))
        rows.append(r)
    return rows


TOUCH_CM = 5.0    # a vetoed route endpoint this close to the nu candidate's own
                  # fitted trajectory means the guard acted ON the neutrino.
MOVE_CM = 1.5     # a trajectory point this far from every point of the other arm
                  # has genuinely moved (Bee draws ~cm-scale detail).


def traj_delta(a, b, fires):
    """Compare the two arms' nu-candidate trajectories, and ask whether any
    vetoed route actually touches that trajectory (epoch-proof, geometric)."""
    from scipy.spatial import cKDTree
    out = {}
    ta, tb = a['traj'], b['traj']
    out['npts'] = (len(ta), len(tb))
    out['npart'] = (len(set(a['tpid'].tolist())), len(set(b['tpid'].tolist())))
    out['qsum'] = (float(a['tq'].sum()), float(b['tq'].sum()))
    if len(ta) and len(tb):
        ka, kb = cKDTree(ta), cKDTree(tb)
        out['moved_pct'] = 100.0 * float((kb.query(ta)[0] > MOVE_CM).mean())
        out['moved_pct_rev'] = 100.0 * float((ka.query(tb)[0] > MOVE_CM).mean())
        # does any veto touch the neutrino's own trajectory?
        d = []
        for f in fires:
            pts = np.stack([f['first'], f['last']])
            d.append(float(min(kb.query(pts)[0].min(), ka.query(pts)[0].min())))
        out['d_fire_traj'] = min(d) if d else float('nan')
        out['n_fire_touch'] = sum(1 for x in d if x <= TOUCH_CM)
        # Did the vertex SLIDE along a trajectory both arms agree on, or land
        # on a different structure?  Old vertex vs NEW trajectory and new
        # vertex vs OLD trajectory: both small => the drawn object is the same
        # and only the vertex position along it changed.  That is a very
        # different scan question from "is this the right object at all".
        out['d_old_on'] = float(kb.query(a['vtx'][None])[0][0])
        out['d_new_off'] = float(ka.query(b['vtx'][None])[0][0])
        out['slide'] = out['d_old_on'] < 3.0 and out['d_new_off'] < 3.0
    else:
        out['moved_pct'] = out['moved_pct_rev'] = float('nan')
        out['d_fire_traj'] = float('nan')
        out['n_fire_touch'] = 0
        out['d_old_on'] = out['d_new_off'] = float('nan')
        out['slide'] = False
    return out


def nusel_delta(r):
    """What moved in the nu-candidate bundle row."""
    if not r['row_off'] or not r['row_on']:
        return ''
    out = []
    for k, fmt in (('npts_main', '%s'), ('len_main_cm', '%s'), ('n_frag', '%s'),
                   ('n_bundle', '%s'), ('tgm', '%s'), ('stm', '%s'), ('fc', '%s'),
                   ('stmfit', '%s'), ('lm', '%s'), ('label', '%s')):
        x, y = r['row_off'].get(k), r['row_on'].get(k)
        if x != y:
            out.append('%s %s->%s' % (k, x, y))
    return ', '.join(out)


def classify(r):
    """Classify by what a scan would SEE, not by where the guard fired.
    (Where it fired is an annotation -- see n_fire_touch / d_fire_traj.)

    Deliberately NOT part of the quiet test: traj_moved_pct.  The fitted point
    list also changes LENGTH between arms (npts moves by up to ~5% on events
    whose every physics observable is static), so a few percent of "moved"
    points is re-sampling of the same track, not a track that bent -- and it
    is not something a physics scan can adjudicate.  It is reported as an
    annotation instead.
    """
    quiet = (r.get('dvtx', 9) <= VTX_QUIET and abs(r.get('denu', 9)) < ENU_QUIET
             and abs(r.get('dpio', 9)) < PIO_QUIET)
    if quiet:
        return 'D. NO PHYSICS OBSERVABLE MOVED -- SKIP'
    if r.get('dvtx', 0) > 10.0:
        return 'A. VERTEX RELOCATED (>10 cm)'
    if r.get('dvtx', 0) > 1.0:
        return 'B. VERTEX NUDGED (1-10 cm)'
    return 'C. SAME VERTEX, DIFFERENT ENERGY / PID'


def main():
    rows = collect()
    if '--evt' in sys.argv:
        want = sys.argv[sys.argv.index('--evt') + 1]
        for r in rows:
            if r['evt'] == want:
                verbose(r)
        return
    if '--pi0' in sys.argv:
        print('pi0 block, events whose kine_pio_mass moves by more than 30 MeV')
        print('with the tag set in BOTH arms.  RE-PAIRED means the photon pair')
        print('itself changed (E2 or the opening angle moved a lot), so the mass')
        print('is a different quantity -- not the same measurement gone worse.')
        print()
        print('%-8s %-9s %12s %12s %12s %10s  %s'
              % ('evt', 'manifest', 'mass', 'E1', 'E2', 'angle', 'reading'))
        rep = same = 0
        for r in rows:
            a, b = r['a'], r['b']
            if not (a['pio_flag'] and b['pio_flag']):
                continue
            if abs(a['pio_mass'] - b['pio_mass']) <= 30:
                continue
            repaired = (abs(b['pio_ang'] - a['pio_ang']) > 20
                        or abs(b['pio_e2'] - a['pio_e2']) > 0.5 * max(a['pio_e2'], 1))
            rep += repaired
            same += not repaired
            print('%-8s %-9s %5.0f->%-5.0f %5.0f->%-5.0f %5.0f->%-5.0f %4.0f->%-4.0f  %s'
                  % (r['evt'], r['man'], a['pio_mass'], b['pio_mass'],
                     a['pio_e1'], b['pio_e1'], a['pio_e2'], b['pio_e2'],
                     a['pio_ang'], b['pio_ang'],
                     'RE-PAIRED' if repaired else 'same pair, re-measured'))
        print()
        print('  re-paired: %d    same pair re-measured: %d' % (rep, same))
        flips = [(r['evt'], r['a']['pio_flag'], r['b']['pio_flag']) for r in rows
                 if r['a']['pio_flag'] != r['b']['pio_flag']]
        for e, x, y in flips:
            print('  evt %s: pi0 TAG %d -> %d' % (e, x, y))
        return
    if '--movers' in sys.argv:
        print('%-4s %-8s %8s %9s %9s %10s %8s  %s'
              % ('idx', 'evt', 'dvtx', 'oldV->new', 'newV->old', 'route cm',
                 'maxsep', 'reading'))
        for r in sorted([x for x in rows if x.get('dvtx', 0) > 1.0],
                        key=lambda x: -x['dvtx']):
            L = [f['base_cm'] for f in r['fires']]
            print('%-4d %-8s %8.2f %9.2f %9.2f %10s %8.2f  %s'
                  % (r['idx'], r['evt'], r['dvtx'], r['d_old_on'], r['d_new_off'],
                     '%.0f-%.0f' % (min(L), max(L)),
                     max(f['maxsep'] for f in r['fires']),
                     'vertex SLID along a track both arms agree on'
                     if r['slide'] else 'vertex landed on a DIFFERENT structure'))
        return
    if '--tsv' in sys.argv:
        out = sys.argv[sys.argv.index('--tsv') + 1]
        with open(out, 'w') as fh:
            cols = ('idx evt man class n_fire n_fire_touch d_fire_traj maxsep '
                    'route_lo route_hi dvtx d_old_on d_new_off slide traj_moved_pct '
                    'denu pio_mass_off pio_mass_on npart_off npart_on '
                    'nusel_delta particle_delta').split()
            fh.write('\t'.join(cols) + '\n')
            for r in rows:
                L = [f['base_cm'] for f in r['fires']] or [0.0]
                a, b = r['a'], r['b']
                fh.write('\t'.join(str(x) for x in (
                    r['idx'], r['evt'], r['man'], classify(r).split('.')[0],
                    r['n_fire'], r['n_fire_touch'],
                    '%.2f' % r['d_fire_traj'],
                    '%.3f' % max([f['maxsep'] for f in r['fires']], default=0.0),
                    '%.1f' % min(L), '%.1f' % max(L),
                    '%.2f' % r.get('dvtx', 0),
                    '%.2f' % r['d_old_on'], '%.2f' % r['d_new_off'],
                    int(bool(r['slide'])),
                    '%.1f' % max(r.get('moved_pct', 0), r.get('moved_pct_rev', 0)),
                    '%.1f' % r.get('denu', 0),
                    '%.1f' % a['pio_mass'], '%.1f' % b['pio_mass'],
                    r['npart'][0], r['npart'][1],
                    nusel_delta(r) or '-', r.get('pdiff', '') or '-')) + '\n')
        print('wrote %s (%d rows)' % (out, len(rows)))
        return

    swap = [r for r in rows if r['cand_off'] != r['cand_on']]
    touch = [r for r in rows if r['n_fire_touch'] > 0]
    idagree = [r for r in rows
               if (r['n_fire_nu'] > 0) == (r['n_fire_touch'] > 0)]
    print('INTEGRITY')
    print('  nu-candidate bundle main_id changes off->on in %d/%d events'
          % (len(swap), len(rows)))
    for r in swap:
        print('     evt %s: %s -> %s' % (r['evt'], r['cand_off'], r['cand_on']))
    print('  a vetoed route touches the nu candidate\'s own fitted trajectory'
          ' (<= %.0f cm) in %d/%d events' % (TOUCH_CM, len(touch), len(rows)))
    print('  the nusel main_id join agrees with that geometric test in %d/%d --'
          ' the %d disagreements' % (len(idagree), len(rows), len(rows) - len(idagree)))
    print('  are cluster-ident epoch skew (doc 53), so the geometric test is the one used below.')
    print()

    order = ['A. VERTEX RELOCATED (>10 cm)', 'B. VERTEX NUDGED (1-10 cm)',
             'C. SAME VERTEX, DIFFERENT ENERGY / PID',
             'D. NO PHYSICS OBSERVABLE MOVED -- SKIP']
    groups = {}
    for r in rows:
        groups.setdefault(classify(r), []).append(r)
    for g in order:
        rs = groups.get(g)
        if not rs:
            continue
        print('=== %s : %d events ===' % (g, len(rs)))
        print('%-4s %-8s %-9s %5s %8s %8s %12s %13s  %s'
              % ('idx', 'evt', 'manifest', 'fires', 'dvtx', 'dEnu', 'pi0 mass',
                 'npts off->on', 'what changed'))
        for r in sorted(rs, key=lambda r: -r.get('dvtx', 0)):
            what = r.get('pdiff', '')
            if endpoint_flip(r):
                what = 'VERTEX SWAPPED TO THE OTHER END OF THE VETOED ROUTE; ' + what
            nd = nusel_delta(r)
            if nd:
                what = (what + ' | ' if what else '') + nd
            a, b = r['a'], r['b']
            pio = ('%4.0f->%-4.0f' % (a['pio_mass'], b['pio_mass'])
                   if (a['pio_flag'] or b['pio_flag']) else '     -      ')
            print('%-4d %-8s %-9s %2d/%-2d %8.2f %8.1f %12s %6d->%-6d %s'
                  % (r['idx'], r['evt'], r['man'], r['n_fire_touch'], r['n_fire'],
                     r.get('dvtx', 0), r.get('denu', 0), pio,
                     r['npts'][0], r['npts'][1], what[:88]))
        print()


def verbose(r):
    print('evt %s (%s)  idx %d   class %s' % (r['evt'], r['man'], r['idx'], classify(r)))
    print('  nu candidate main_id: off=%s on=%s' % (r['cand_off'], r['cand_on']))
    print('  fires: %d total, %d on the candidate cluster, %d elsewhere %s'
          % (r['n_fire'], r['n_fire_nu'], r['n_fire_other'], r['other_clusters'] or ''))
    for f in r['fires']:
        tag = 'CAND' if f['cluster'] == r['cand_on'] else 'other'
        print('    cluster %-4s %-5s maxsep=%6.3f base=%7.2f cm  (%.1f,%.1f,%.1f) -> (%.1f,%.1f,%.1f)'
              % (f['cluster'], tag, f['maxsep'], f['base_cm'],
                 f['first'][0], f['first'][1], f['first'][2],
                 f['last'][0], f['last'][1], f['last'][2]))
    a, b = r['a'], r['b']
    print('  nu vertex  off=(%.2f,%.2f,%.2f)  on=(%.2f,%.2f,%.2f)  |d|=%.2f cm'
          % (tuple(a['vtx']) + tuple(b['vtx']) + (r['dvtx'],)))
    print('  Enu %.1f -> %.1f MeV  (d=%+.1f);  add_energy %.1f -> %.1f'
          % (a['enu'], b['enu'], r['denu'], a['add'], b['add']))
    print('  pi0: flag %d->%d  mass %.1f->%.1f  E1 %.1f->%.1f  E2 %.1f->%.1f  angle %.3f->%.3f'
          % (a['pio_flag'], b['pio_flag'], a['pio_mass'], b['pio_mass'],
             a['pio_e1'], b['pio_e1'], a['pio_e2'], b['pio_e2'], a['pio_ang'], b['pio_ang']))
    for nm, d in (('OFF', a), ('ON ', b)):
        inc = [(pname(t), float(e)) for t, e, i in zip(d['pt'], d['ep'], d['inc']) if i]
        exc = [(pname(t), float(e)) for t, e, i in zip(d['pt'], d['ep'], d['inc']) if not i]
        print('  %s included (%d): %s' % (nm, len(inc),
              ', '.join('%s %.1f' % x for x in inc)))
        if exc:
            print('  %s excluded (%d): %s' % (nm, len(exc),
                  ', '.join('%s %.1f' % x for x in exc)))
    print('  nusel candidate row: %s' % (nusel_delta(r) or 'unchanged'))


if __name__ == '__main__':
    main()

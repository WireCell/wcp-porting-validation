#!/usr/bin/env python3
"""doc 86 -- pick showcase events for the reconstruction video from `prod0830`.

Reads ONLY existing prod0830 products (no arm is re-run):

  products/prod0830/<sample>-scores-prod0830.tsv   pr_scores_table.py output
  work-<sample>-prod0830/pr_evt<ID>/tracking-pr.root      T_kine, every row
  work-<sample>-prod0830/pr_evt<ID>/mabc-pr.zip           track_fit-global + mc

and writes per-category pick lists + one wide feature table under docs/86_video/.

THREE THINGS THIS SCRIPT IS CAREFUL ABOUT
-----------------------------------------
1. There is NO TRUTH.  All four samples are SBND *data* (doc 85 sec 1; doc 21
   line 51: reco1 carries no MC truth).  Every category below is a
   *reconstruction* topology on a *selected* sample -- "nueCC" names the
   selection and the reco final state, never the interaction.

2. `kine_pio_flag == 1` is NOT "a pi0 was found", and is NOT "the gammas point
   at the vertex".  NeutrinoShowerClustering.cxx:6046-6100 fills the whole
   pio_kine block for the HIGHEST-ENERGY shower pair in the event whatever it
   is, as a BDT feature.  973 of 1458 prod0830 rows have flag==1 with a median
   mass of 5 MeV and median gamma energies of 2.7 MeV -- shower fragments.
   The real pi0 gate is on the KINEMATICS (mass + both gamma energies), and
   "pointing back at the vertex" is `kine_pio_vtx_dis` (pi0 decay vertex ->
   main vertex), NOT `dis_1`/`dis_2`, which are the photon CONVERSION GAPS and
   for a real pi0 are expected to be LARGE.

3. Cathode crossing is measured from the fitted trajectory, not inferred.
   SBND's cathode is at x = -+0.45 cm (doc 2), so a crosser has track_fit
   points on both sides with margin.

Usage:  python3 scripts/analysis/d86_video_picks.py
"""
import glob
import json
import os
import re
import sys
import math
import zipfile
from collections import OrderedDict

import numpy as np
import uproot

SX = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(SX, 'docs', '86_video')
SAMPLES = ['ncpi0', 'nuecc48', 'mcp1k', 'mcp2k']
ARM = 'prod0830'

RE_SEL = re.compile(r'TaggerCheckNeutrino: selected main cluster ')
RE_NUNODE = re.compile(r'^nu (\d+) \(gid (-?\d+), cluster (-?\d+)\)')
RE_PART = re.compile(r'^\s*(\S+)\s+([0-9.]+)\s*MeV')

# --- gates -------------------------------------------------------------------
NUE_SEL, NUMU_SEL = 7.0, 0.9          # MicroBooNE working points (doc 85 sec 7)
NUE_UNFILLED = -15.0                  # br_filled != 1 sentinel; test EQUALITY
PI0_MASS_LO, PI0_MASS_HI = 90.0, 180.0    # generous; ranking is on |m-135|
PI0_GAMMA_MIN = 40.0                  # MeV, each gamma
PI0_GAMMA_MAX = 800.0                 # MeV; above this the "gamma" is the whole
                                      # event's shower paired with a fragment
PI0_ASYM_MAX = 0.70                   # |E1-E2|/(E1+E2): a visually balanced pair
PI0_ANGLE_MIN = 15.0                  # deg, opening angle (two distinct showers)
PI0_VTXDIS_MAX = 5.0                  # cm, pi0 decay vertex -> nu vertex
CATH_MARGIN = 10.0                    # cm either side of x=0
MU_LEN_MIN = 50.0                     # cm, muon-candidate track length
# SBND active volume (doc 2) with a 20 cm margin.
FV = dict(x=(-201.45, 201.45), y=(-200.0, 200.0), z=(0.0, 500.0))
FV_MARGIN = 20.0


def in_fv(x, y, z, m=FV_MARGIN):
    return (FV['x'][0] + m < x < FV['x'][1] - m and
            FV['y'][0] + m < y < FV['y'][1] - m and
            FV['z'][0] + m < z < FV['z'][1] - m)


def flag_is(v, want):
    """Tagger flags reach the table as floats-as-strings ('0.0'/'1.0'), so a
    string compare against '0'/'1' silently matches NOTHING.  Compare numbers."""
    f = fnum(v)
    return f is not None and int(f) == want


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_scores():
    """(sample, event) -> dict of the pr_scores_table.py row."""
    out = {}
    for s in SAMPLES:
        p = os.path.join(SX, 'products', ARM, f'{s}-scores-{ARM}.tsv')
        with open(p) as f:
            hdr = f.readline().rstrip('\n').split('\t')
            for line in f:
                v = line.rstrip('\n').split('\t')
                d = dict(zip(hdr, v))
                out[(s, int(d['event']))] = d
    return out


def read_mc_tree(zf):
    """Parse the PR particle-flow jsTree: one entry per neutrino candidate."""
    for n in zf.namelist():
        if os.path.basename(n) == '0-mc.json':
            return json.loads(zf.read(n))
    return []


def summarize_nus(mc):
    """One entry per top-level `nu N (gid G, cluster C)` node, each carrying its
    particle list with the per-particle start/end the jsTree records."""
    nus = []
    for node in mc:
        txt = node.get('text', '')
        m = RE_NUNODE.match(txt)
        if not m:
            continue
        parts = []
        stack = list(node.get('children', []))
        while stack:
            c = stack.pop(0)
            pm = RE_PART.match(c.get('text', ''))
            if pm:
                dd = c.get('data', {}) or {}
                parts.append(dict(name=pm.group(1), mev=float(pm.group(2)),
                                  start=dd.get('start'), end=dd.get('end')))
            stack.extend(c.get('children', []))
        nus.append(dict(nu=int(m.group(1)), cluster=int(m.group(3)),
                        start=(node.get('data', {}) or {}).get('start'),
                        parts=parts))
    return nus


def final_state(nus):
    """'mu- 1138 + proton 3 + neutron 3' per neutrino candidate, for the doc."""
    out = []
    for nu in nus:
        ps = sorted(nu['parts'], key=lambda q: -q['mev'])
        out.append(' + '.join(f"{q['name']} {q['mev']:.0f}" for q in ps[:6])
                   or '(none)')
    return ' || '.join(out)


def longest_muon_x(nus):
    """(x_start, x_end) of the highest-energy reconstructed muon, or None."""
    best = None
    for nu in nus:
        for p in nu['parts']:
            # The jsTree names particles 'mu-' / 'mu+' / 'e-' / 'proton' ...
            # (NOT 'muon' -- PrDisplayDump writes the PDG name).
            if p['name'] not in ('mu-', 'mu+') or not p.get('start') or not p.get('end'):
                continue
            if best is None or p['mev'] > best['mev']:
                best = p
    if best is None:
        return None
    return (float(best['start'][0]), float(best['end'][0]))


def longest_muon_pt(nus):
    """(start, end) 3-points of the highest-energy muon -- containment check."""
    best = None
    for nu in nus:
        for p in nu['parts']:
            if p['name'] not in ('mu-', 'mu+') or not p.get('start') or not p.get('end'):
                continue
            if best is None or p['mev'] > best['mev']:
                best = p
    return None if best is None else (best['start'], best['end'])


def track_geom(zf):
    """x/y/z extent of the fitted trajectory + cathode-crossing evidence."""
    for n in zf.namelist():
        if os.path.basename(n) == '0-track_fit-global.json':
            d = json.loads(zf.read(n))
            x = np.asarray(d.get('x', []), dtype=float)
            y = np.asarray(d.get('y', []), dtype=float)
            z = np.asarray(d.get('z', []), dtype=float)
            if x.size == 0:
                return None
            nneg = int((x < -CATH_MARGIN).sum())
            npos = int((x > CATH_MARGIN).sum())
            return dict(npts=int(x.size),
                        xmin=float(x.min()), xmax=float(x.max()),
                        ymin=float(y.min()), ymax=float(y.max()),
                        zmin=float(z.min()), zmax=float(z.max()),
                        n_xneg=nneg, n_xpos=npos,
                        cathode_cross=bool(nneg >= 5 and npos >= 5),
                        diag=float(np.sqrt((x.max() - x.min()) ** 2 +
                                           (y.max() - y.min()) ** 2 +
                                           (z.max() - z.min()) ** 2)))
    return None


KINE_BR = ['kine_reco_Enu', 'kine_pio_mass', 'kine_pio_flag', 'kine_pio_vtx_dis',
           'kine_pio_energy_1', 'kine_pio_dis_1', 'kine_pio_energy_2',
           'kine_pio_dis_2', 'kine_pio_angle', 'kine_energy_particle',
           'kine_particle_type', 'kine_energy_excluded',
           'kine_energy_excluded_main', 'kine_energy_excluded_other',
           'kine_n_excluded', 'kine_energy_flagged', 'nu_index']


def build():
    scores = read_scores()
    rows = []
    for s in SAMPLES:
        root = os.path.join(SX, f'work-{s}-{ARM}')
        for d in sorted(glob.glob(os.path.join(root, 'pr_evt*'))):
            evt = int(os.path.basename(d)[6:])
            log = os.path.join(d, f'wct_pr_evt{evt}.log')
            if not os.path.isfile(log):
                continue
            with open(log, errors='replace') as f:
                if not any(RE_SEL.search(l) for l in f):
                    continue            # nu_evaluated == 0: scores are defaults
            # This arm records the exit code in rc.txt; the .time.meta the
            # score table reads was not written, so rc/wall_s/maxrss_kb are
            # blank there.  Read the real source.
            rcv = ''
            rcp = os.path.join(d, 'rc.txt')
            if os.path.isfile(rcp):
                rcv = open(rcp).read().strip().replace('rc=', '')
            rp = os.path.join(d, 'tracking-pr.root')
            zp = os.path.join(d, 'mabc-pr.zip')
            if not (os.path.isfile(rp) and os.path.isfile(zp)):
                continue
            try:
                tk = uproot.open(rp)['T_kine']
                if tk.num_entries == 0:
                    continue
                a = tk.arrays(KINE_BR, library='np')
            except Exception as e:                       # noqa: BLE001
                print(f'ERR kine {s} {evt}: {e}', file=sys.stderr)
                continue
            try:
                with zipfile.ZipFile(zp) as zf:
                    nus = summarize_nus(read_mc_tree(zf))
                    geom = track_geom(zf)
            except Exception as e:                       # noqa: BLE001
                print(f'ERR zip {s} {evt}: {e}', file=sys.stderr)
                continue

            sc = scores.get((s, evt), {})
            for i in range(tk.num_entries):
                ptypes = list(map(int, a['kine_particle_type'][i]))
                penes = [float(x) for x in a['kine_energy_particle'][i]]
                by = {}
                for t, e in zip(ptypes, penes):
                    by.setdefault(abs(t), []).append(e)
                rows.append(dict(
                    sample=s, event=evt, krow=i, nkrow=int(tk.num_entries),
                    run=sc.get('run', ''), subrun=sc.get('subrun', ''),
                    rc=rcv, dl_warn=sc.get('dl_warn', ''),
                    event_label=sc.get('event_label', ''),
                    n_inbeam_bundle=sc.get('n_inbeam_bundle', ''),
                    numu_score=fnum(sc.get('numu_score')),
                    nue_score=fnum(sc.get('nue_score')),
                    cosmic_flag=sc.get('cosmic_flag', ''),
                    cosmict_flag=sc.get('cosmict_flag', ''),
                    nu_x=fnum(sc.get('nu_x_cm')), nu_y=fnum(sc.get('nu_y_cm')),
                    nu_z=fnum(sc.get('nu_z_cm')),
                    nu_len=fnum(sc.get('nu_sel_len_cm')),
                    wall_s=fnum(sc.get('wall_s')), maxrss=fnum(sc.get('maxrss_kb')),
                    # (both blank on this arm -- see rc.txt note above)
                    Enu=float(a['kine_reco_Enu'][i]),
                    exc=float(a['kine_energy_excluded'][i]),
                    exc_main=float(a['kine_energy_excluded_main'][i]),
                    exc_other=float(a['kine_energy_excluded_other'][i]),
                    n_exc=int(a['kine_n_excluded'][i]),
                    flagged=float(a['kine_energy_flagged'][i]),
                    pio_flag=int(a['kine_pio_flag'][i]),
                    pio_mass=float(a['kine_pio_mass'][i]),
                    pio_vtxdis=float(a['kine_pio_vtx_dis'][i]),
                    g1=float(a['kine_pio_energy_1'][i]), gd1=float(a['kine_pio_dis_1'][i]),
                    g2=float(a['kine_pio_energy_2'][i]), gd2=float(a['kine_pio_dis_2'][i]),
                    pio_angle=float(a['kine_pio_angle'][i]),
                    n_part=len(ptypes),
                    e_mu=max(by.get(13, [0]) or [0]), n_mu=len(by.get(13, [])),
                    e_e=max(by.get(11, [0]) or [0]), n_e=len(by.get(11, [])),
                    n_p=len(by.get(2212, [])), n_pi=len(by.get(211, [])),
                    n_nu_nodes=len(nus), nus=nus, geom=geom,
                    mu_seg=longest_muon_x(nus), fstate=final_state(nus),
                    mu_end=longest_muon_pt(nus)))
    return rows


def mu_crosses_cathode(r):
    """Does the reconstructed MUON itself span the cathode?  The track_fit
    layer covers every segment of the candidate, so 'some point either side of
    x=0' can be two unrelated prongs.  The mc jsTree gives each particle its
    own start/end, so this asks the question that matters."""
    seg = r.get('mu_seg')
    if not seg:
        return False
    (x0, x1) = seg
    return (min(x0, x1) < -CATH_MARGIN) and (max(x0, x1) > CATH_MARGIN)


def pio_closure(r):
    """(recomputed mass, recomputed/reported).

    m = sqrt(4.E1.E2.sin^2(theta/2)) is how the finder builds `mass`
    (NeutrinoShowerClustering.cxx:6034), but the E's and theta it STORES are
    not the ones it used:

      * `mass` uses `local_dirs[sh]`, which is the shower's own `get_init_dir()`
        when the shower is attached to the candidate pi0 vertex (:5964, :5978)
        and the vertex->start vector when it was associated by angle (:5997);
      * `kine_pio_angle` is recomputed in the fill loop (:6078-6084) from a
        DIFFERENT rule -- a fresh 15 cm direction fit when the conversion gap
        is < 3 cm, the vertex->start vector otherwise.

    The two rules pick the same source for most rows and a different one for
    the rest, so the reported triple reproduces the reported mass 25 times out
    of 35 in the pi0 pool and misses by up to 20% on the others.  A pick whose
    triple does NOT close would put a mass on screen that nobody can re-derive
    from the numbers printed beside it, so closure is a DISPLAY gate here --
    not a physics quality claim.
    """
    e1, e2, a, m = r['g1'], r['g2'], r['pio_angle'], r['pio_mass']
    if m <= 0 or e1 <= 0 or e2 <= 0:
        return (0.0, float('nan'))
    rec = math.sqrt(4.0 * e1 * e2) * math.sin(math.radians(a) / 2.0)
    return (rec, rec / m)


def frac_exc(r):
    return r['exc'] / r['Enu'] if r['Enu'] > 0 else float('nan')


def degenerate(r):
    """doc 85 sec 1.1: Enu == 0 at vertex (0,0,0) -- an evaluated-but-empty row."""
    return (r['Enu'] == 0.0 and r['nu_x'] is not None
            and abs(r['nu_x']) < 1e-6 and abs(r['nu_y']) < 1e-6 and abs(r['nu_z']) < 1e-6)


def base_ok(r):
    """The gates every SHOWCASE event must pass (failure cases skip these)."""
    return (r['rc'] == '0' and r['dl_warn'] in ('0', '') and not degenerate(r)
            and r['Enu'] > 0 and r['nu_x'] is not None
            and in_fv(r['nu_x'], r['nu_y'], r['nu_z'])
            and r['geom'] is not None)


def good_pi0(r):
    """A pi0 whose two gammas back-project to the neutrino vertex.  See the
    module docstring: mass + both gamma energies carry the physics, and
    vtx_dis is the pointing.  The asymmetry and opening-angle gates are for
    the DISPLAY -- a 1.7 GeV shower paired with a 4 MeV fragment can land on
    135 MeV by accident and looks like nothing on screen."""
    if r['pio_flag'] != 1:
        return False
    e1, e2 = r['g1'], r['g2']
    if not (PI0_GAMMA_MIN < e1 < PI0_GAMMA_MAX and PI0_GAMMA_MIN < e2 < PI0_GAMMA_MAX):
        return False
    if abs(e1 - e2) / (e1 + e2) > PI0_ASYM_MAX:
        return False
    return (PI0_MASS_LO < r['pio_mass'] < PI0_MASS_HI
            and r['pio_angle'] > PI0_ANGLE_MIN
            and r['pio_vtxdis'] < PI0_VTXDIS_MAX)


def pi0_closes(r, tol=0.005):
    return abs(pio_closure(r)[1] - 1.0) < tol


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = build()
    print(f'rows (nu_evaluated, kine row) = {len(rows)}')

    cols = ['sample', 'event', 'krow', 'nkrow', 'run', 'subrun', 'rc', 'numu_score',
            'nue_score', 'cosmic_flag', 'cosmict_flag', 'event_label',
            'n_inbeam_bundle', 'nu_x', 'nu_y', 'nu_z', 'nu_len', 'Enu', 'exc',
            'exc_main', 'exc_other', 'n_exc', 'flagged', 'pio_flag', 'pio_mass',
            'pio_vtxdis', 'g1', 'gd1', 'g2', 'gd2', 'pio_angle', 'n_part',
            'n_mu', 'e_mu', 'n_e', 'e_e', 'n_p', 'n_pi', 'n_nu_nodes',
            'wall_s', 'maxrss']
    with open(os.path.join(OUT, 'd86-features.tsv'), 'w') as f:
        f.write('\t'.join(cols + ['exc_frac', 'cathode_cross', 'mu_cathode_cross',
                                  'pio_mass_recomputed', 'pio_mass_closure',
                                  'x_min', 'x_max', 'trk_diag', 'trk_npts',
                                  'final_state']) + '\n')
        for r in rows:
            g = r['geom'] or {}
            f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\t'
                    + f"{frac_exc(r):.4f}\t{int(bool(g.get('cathode_cross')))}\t"
                    + f"{int(mu_crosses_cathode(r))}\t"
                    + f"{pio_closure(r)[0]:.2f}\t{pio_closure(r)[1]:.4f}\t"
                    + f"{g.get('xmin', ''):}\t{g.get('xmax', ''):}\t"
                    + f"{g.get('diag', ''):}\t{g.get('npts', '')}\t"
                    + f"{r.get('fstate', '')}\n")

    ok = [r for r in rows if base_ok(r)]
    print(f'base_ok rows = {len(ok)}')

    picks = OrderedDict()
    used = set()                       # (sample, event) -- one category each

    def take(cands, key, n):
        cands.sort(key=key)
        out = []
        for r in cands:
            if (r['sample'], r['event']) in used:
                continue
            out.append(r)
            used.add((r['sample'], r['event']))
            if len(out) >= n:
                break
        return out

    # --- 1. nueCC candidate: a reconstructed primary electron, high nue_score
    c = [r for r in ok if r['sample'] == 'nuecc48'
         and r['nue_score'] is not None and r['nue_score'] > NUE_SEL
         and r['nue_score'] != NUE_UNFILLED
         and r['n_e'] >= 1 and r['e_e'] > 100
         and flag_is(r['cosmict_flag'], 0)
         and frac_exc(r) < 0.05]
    picks['nuecc'] = take(c, lambda r: (-r['nue_score'], frac_exc(r)), 3)

    # --- 2. numuCC candidate: long muon, high numu_score.  The owner asked for
    #        a CATHODE CROSSER, taken first so it is guaranteed a slot, and
    #        judged on the MUON's own endpoints, not the whole candidate.
    c = [r for r in ok if r['sample'] in ('mcp1k', 'mcp2k')
         and r['numu_score'] is not None and r['numu_score'] > NUMU_SEL
         and r['n_mu'] >= 1 and r['e_mu'] > 200
         and r['nu_len'] is not None and r['nu_len'] > MU_LEN_MIN
         and flag_is(r['cosmict_flag'], 0)
         and r['n_nu_nodes'] == 1
         and frac_exc(r) < 0.05]
    cross = [r for r in c if mu_crosses_cathode(r)]
    # Prefer the visually CLEAN crosser (fewest spurious prongs), not the
    # longest -- both have zero excluded energy, so e_mu is not the tiebreak.
    picks['numucc_cathode'] = take(cross, lambda r: (frac_exc(r), r['n_part'],
                                                     -r['e_mu']), 2)
    plain = [r for r in c if not mu_crosses_cathode(r)]
    picks['numucc'] = take(plain, lambda r: (frac_exc(r), -r['e_mu']), 2)

    # --- 3/4. pi0 with both gammas back-projecting to the vertex.  CC vs NC
    #          splits on a reconstructed primary muon -- the only handle there
    #          is without truth -- and the numu BDT must AGREE with that split,
    #          so the label is not resting on one number.
    p = [r for r in ok if good_pi0(r)]
    print(f'good_pi0 rows = {len(p)}  '
          f'(ncpi0 sideband: {sum(1 for r in p if r["sample"] == "ncpi0")})')
    ccp = [r for r in p if r['n_mu'] >= 1 and r['e_mu'] > 150
           and r['numu_score'] is not None and r['numu_score'] > NUMU_SEL]
    ncp = [r for r in p if r['n_mu'] == 0
           and r['numu_score'] is not None and r['numu_score'] < NUMU_SEL]
    print(f'  ccpi0 pool = {len(ccp)}   ncpi0 pool (mu-free + BDT agrees) = {len(ncp)}')
    print(f'  of those, triple closes: cc {sum(map(pi0_closes, ccp))}/{len(ccp)}'
          f'  nc {sum(map(pi0_closes, ncp))}/{len(ncp)}')
    picks['ccpi0'] = take(ccp, lambda r: (not pi0_closes(r),
                                          abs(r['pio_mass'] - 135.0),
                                          frac_exc(r)), 2)
    # Two NC pi0 picks with DIFFERENT claims, because no single event has both:
    #  (a) self-consistent -- no reconstructed muon AND the numu BDT agrees;
    #  (b) from the NC pi0 SIDEBAND sample itself, muon-free, whatever the BDT
    #      says.  The sideband's only muon-free good pi0 is numu-LIKE to the
    #      BDT, and that tension is reported rather than gated away.
    picks['ncpi0'] = take(ncp, lambda r: (not pi0_closes(r),
                                          abs(r['pio_mass'] - 135.0)), 2)
    side = [r for r in p if r['sample'] == 'ncpi0' and r['n_mu'] == 0]
    picks['ncpi0_sideband'] = take(side, lambda r: abs(r['pio_mass'] - 135.0), 1)

    # --- 5. cosmic-like: cosmict_flag is the tagger's cosmic verdict
    #        (cosmic_flag is NOT -- it is 1.0 on every nuecc48 event).  Rank by
    #        how cosmic the numu BDT also thinks it is, over long tracks.
    c = [r for r in rows if r['rc'] == '0' and not degenerate(r)
         and flag_is(r['cosmict_flag'], 1) and r['geom'] is not None
         and r['geom']['diag'] > 300 and r['numu_score'] is not None]
    picks['cosmiclike'] = take(c, lambda r: (r['numu_score'], -r['geom']['diag']), 2)

    # --- 6. more than one neutrino candidate reconstructed in one event
    c = [r for r in ok if r['n_nu_nodes'] >= 2 and r['krow'] == 0
         and r['nkrow'] >= 2]
    picks['multinu'] = take(c, lambda r: (-r['n_nu_nodes'], frac_exc(r)), 2)

    # --- F1. busy event: many reconstructed particles AND much charge that the
    #         Enu sum could not account for (the doc-85 sec 9 variable).
    c = [r for r in rows if r['rc'] == '0' and not degenerate(r) and r['Enu'] > 0
         and r['n_part'] >= 12 and r['n_exc'] >= 15]
    picks['fail_busy'] = take(c, lambda r: (-r['n_exc'], -r['n_part']), 2)

    # --- F2. EM-shower clustering failure: a nueCC-SELECTED event the nue BDT
    #         lost.  Two mechanisms (doc 85 sec 9.7): scored-and-lost near
    #         misses, and rows the nue BDT never ran on at all (== -15).
    c = [r for r in rows if r['sample'] == 'nuecc48' and r['rc'] == '0'
         and r['nue_score'] is not None and r['nue_score'] < NUE_SEL
         and r['nue_score'] != NUE_UNFILLED and not degenerate(r)]
    picks['fail_em_nearmiss'] = take(c, lambda r: -r['nue_score'], 2)
    c = [r for r in rows if r['sample'] == 'nuecc48' and r['rc'] == '0'
         and r['nue_score'] == NUE_UNFILLED and not degenerate(r)]
    picks['fail_em_unscored'] = take(c, lambda r: -r['Enu'], 1)

    for k, v in picks.items():
        with open(os.path.join(OUT, f'd86-{k}.txt'), 'w') as f:
            for r in v:
                f.write(f"{r['event']}\n")
        print(f'\n== {k} ({len(v)}) ==')
        for r in v:
            g = r['geom'] or {}
            print(f"  {r['sample']:8s} {r['event']:>7d} row{r['krow']} "
                  f"nue={r['nue_score']} numu={r['numu_score']} "
                  f"Enu={r['Enu']:.0f} exc={r['exc']:.1f}({frac_exc(r)*100:.1f}%) "
                  f"n_exc={r['n_exc']} mu={r['n_mu']}/{r['e_mu']:.0f} "
                  f"e={r['n_e']}/{r['e_e']:.0f} p={r['n_p']} "
                  f"pi0={r['pio_mass']:.1f}({r['g1']:.0f},{r['g2']:.0f},"
                  f"vd={r['pio_vtxdis']:.1f}) nnu={r['n_nu_nodes']} "
                  f"cath_evt={int(bool(g.get('cathode_cross')))} "
                  f"cath_mu={int(mu_crosses_cathode(r))} "
                  f"diag={g.get('diag', 0):.0f}\n           FS: {r.get('fstate', '')}"
                  f"  mu_end={r.get('mu_end')}")

    # ---- the FINAL video sets ------------------------------------------
    # Chosen from the pools printed above; the reason for each is in doc 86.
    # (sample, event) pairs, IN THE ORDER THEY ARE ADDED TO THE BEE ZIP -- Bee
    # numbers events by upload order, not by event id, so this list IS the
    # event/<i>/ mapping and must not be reordered without rebuilding.
    FINAL = OrderedDict([
        ('nuecc',          [('nuecc48', 81597), ('nuecc48', 267597)]),
        ('numucc-cathode', [('mcp2k', 283591), ('mcp1k', 313979)]),
        ('numucc',         [('mcp2k', 290718), ('mcp2k', 94293)]),
        # pi0 picks are the CLOSING ones (see pio_closure): 400504, the earlier
        # CC pick, reports mass 138.9 from 64.4/146.2 MeV at 73.0 deg, which
        # recomputes to 115.4 -- unusable on screen.  It is kept in doc 86
        # sec 1.1 as the worked example instead.
        ('ccpi0',          [('mcp2k', 99838), ('mcp2k', 242726)]),
        # index 2 is the NC pi0 SIDEBAND sample's only muon-free pi0; it is
        # last and caveated (triple closes to only 0.912, numu_score +1.36,
        # and its second photon has no conversion gap at all).
        ('ncpi0',          [('mcp2k', 57709), ('mcp2k', 176986),
                            ('ncpi0', 180801)]),
        ('cosmiclike',     [('mcp2k', 180698), ('mcp2k', 99563)]),
        ('multinu',        [('mcp1k', 487303), ('mcp2k', 174661)]),
        ('fail-busy',      [('nuecc48', 389538), ('mcp2k', 67868)]),
        ('fail-em',        [('nuecc48', 69314), ('nuecc48', 138009),
                            ('nuecc48', 271851)]),
    ])
    by_key = {}
    for r in rows:
        by_key.setdefault((r['sample'], r['event']), []).append(r)

    missing = [k for v in FINAL.values() for k in v if k not in by_key]
    if missing:
        print(f'!! FINAL references rows that do not exist: {missing}')

    with open(os.path.join(OUT, 'd86-final.tsv'), 'w') as f:
        f.write('\t'.join(['set', 'bee_index', 'sample', 'run', 'subrun', 'event',
                           'krow', 'nkrow', 'numu_score', 'nue_score',
                           'cosmic_flag', 'cosmict_flag', 'Enu_MeV',
                           'excluded_MeV', 'exc_frac',
                           'n_excluded', 'pio_mass', 'pio_mass_recomputed',
                           'pio_mass_closure', 'pio_g1', 'pio_g2',
                           'pio_angle', 'pio_gap_1', 'pio_gap_2',
                           'pio_vtx_dis', 'nu_x', 'nu_y', 'nu_z',
                           'n_nu_nodes', 'mu_cathode_cross', 'final_state']) + '\n')
        for setname, keys in FINAL.items():
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, f'd86-set-{setname}.txt'), 'w') as pf:
                for i, k in enumerate(keys):
                    pf.write(f'{k[1]}\n')
            for i, k in enumerate(keys):
                for r in by_key.get(k, []):
                    f.write('\t'.join(str(x) for x in [
                        setname, i, r['sample'], r['run'], r['subrun'], r['event'],
                        r['krow'], r['nkrow'], r['numu_score'], r['nue_score'],
                        r['cosmic_flag'], r['cosmict_flag'],
                        f"{r['Enu']:.1f}", f"{r['exc']:.1f}",
                        f'{frac_exc(r):.4f}', r['n_exc'], f"{r['pio_mass']:.1f}",
                        f"{pio_closure(r)[0]:.1f}", f"{pio_closure(r)[1]:.4f}",
                        f"{r['g1']:.1f}", f"{r['g2']:.1f}", f"{r['pio_angle']:.1f}",
                        f"{r['gd1']:.1f}", f"{r['gd2']:.1f}",
                        f"{r['pio_vtxdis']:.2f}", r['nu_x'], r['nu_y'], r['nu_z'],
                        r['n_nu_nodes'], int(mu_crosses_cathode(r)),
                        r.get('fstate', '')]) + '\n')
    json.dump({k: v for k, v in FINAL.items()},
              open(os.path.join(OUT, 'd86-final.json'), 'w'), indent=1)
    print(f'\nFINAL sets written: {len(FINAL)}, '
          f'{sum(len(v) for v in FINAL.values())} events')

    json.dump({k: [(r['sample'], r['event'], r['krow']) for r in v]
               for k, v in picks.items()},
              open(os.path.join(OUT, 'd86-picks.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

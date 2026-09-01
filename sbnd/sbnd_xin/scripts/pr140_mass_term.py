#!/usr/bin/env python3
"""doc pr/139 sec 24 -- does the EM-shower SPLIT perturb kine_reco_Enu through
the rest-MASS term?  The owner's question, 2026-09-03:

    "For the split of EM shower, I want to make sure that we do not have large
     perturbation to the neutrino energy reconstruction.  This is mostly coming
     from potential mass term of the particle.  EM shower is essentially
     electron with almost no mass contribution, but other particle will have
     mass contribution."

The concern is exactly right and the mechanism is in
NeutrinoKinematics.cxx:102-111 (`rest_term_rules`, doc pr/101 K2, SBND
PRODUCTION ON via kine_mass_rules):

    |pdg| == 11              -> 0                 (electron: no rest term)
    |pdg| in {2212, 2112}    -> 8.6 MeV           (nucleon binding energy)
    |pdg| in {13, 211, 321}  -> full rest mass    (105.66 / 139.57 / 493.68 MeV)
    otherwise                -> full rest mass

and every shower pushes one such term (`push_shower_kine`, :223-236), which is
summed into kine_reco_add_energy and thence into kine_reco_Enu (:889-893).

**So a split adds one MORE object, and the perturbation is ZERO if and only if
the daughter is typed 11.**  This measures how often that holds.

    python3 scripts/pr140_mass_term.py                       # shipped config
    python3 scripts/pr140_mass_term.py --pair work-pr138r2-c90off work-pr138r2-c90on

READ-ONLY.
"""
import argparse, collections, glob, json, os, re, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEEL = re.compile(r'SHOWER_SPLIT peel shower=(\d+) part=(\d+) new_start=(\d+) nseg=(\d+)')
SHED = re.compile(r'SHOWER_SPLIT shed shower=(\d+)')
SAMPLES = ('mcp1k', 'mcp2k', 'ncpi0', 'nuecc48')
BIND = 8.6


def rest_term(pdg, mass_mev):
    a = abs(int(pdg))
    if a == 11:
        return 0.0
    if a in (2212, 2112):
        return BIND
    return mass_mev


# PDG rest masses in MeV, for the species the rule names.
MASS = {11: 0.511, 13: 105.658, 211: 139.570, 321: 493.677, 2212: 938.272, 2112: 939.565,
        22: 0.0, 111: 134.977, 0: 0.0}


def dumps(arm):
    for s in SAMPLES:
        for p in sorted(glob.glob(os.path.join(SX, '%s-%s' % (arm, s), 'pr_evt*',
                                               'calib-pr-evt*.json'))):
            ev = int(re.search(r'calib-pr-evt(\d+)\.json', p).group(1))
            yield ev, p


def peels(arm):
    out = collections.defaultdict(list)
    nshed = 0
    for lg in sorted(glob.glob(os.path.join(SX, arm) + '-*/pr_evt*/stdout.log')):
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = PEEL.search(line)
            if m:
                out[ev].append(int(m.group(3)))
            elif SHED.search(line):
                nshed += 1
    return out, nshed


def shipped(arm):
    P, nshed = peels(arm)
    tot = 0
    by = collections.Counter()
    mass_added = 0.0
    offenders = []
    nev = 0
    for ev, p in dumps(arm):
        nev += 1
        d = json.load(open(p))
        byid = {}
        for s in (d.get('showers') or ()):
            byid[int(s['id'])] = s
        for start in P.get(ev, ()):
            tot += 1
            sh = byid.get(start)
            if sh is None:
                by['<not a shower in the dump>'] += 1
                continue
            pdg = sh.get('particle_id')
            pdg = int(pdg) if pdg is not None else 0
            by[pdg] += 1
            rt = rest_term(pdg, MASS.get(abs(pdg), 0.0))
            mass_added += rt
            if rt > 0:
                offenders.append((ev, start, pdg, rt, sh.get('kine_best')))
    print("=== the SHIPPED configuration: %s  (%d events) ===" % (arm, nev))
    print("  daughters created by a peel : %d      shed components (no daughter): %d"
          % (tot, nshed))
    print("  their PDG                   : %s" % dict(sorted(by.items(), key=lambda kv: str(kv[0]))))
    print()
    print("  TOTAL rest-mass term added by split daughters : %.1f MeV over %d events"
          % (mass_added, nev))
    print("  i.e. %.2f MeV per event averaged over the manifest" % (mass_added / max(nev, 1)))
    if offenders:
        print()
        print("  every daughter that adds a non-zero rest term:")
        for ev, st, pdg, rt, kb in sorted(offenders, key=lambda t: -t[3]):
            print("    evt%-8d start_seg %-7d pdg=%-6d adds %7.2f MeV   (its own kine_best %.1f MeV)"
                  % (ev, st, pdg, rt, kb if kb is not None else float('nan')))
    else:
        print("\n  NO split daughter adds any rest-mass term at all.")
    return mass_added, nev


def pair(a, b):
    """Δ kine_reco_Enu and Δ kine_reco_add_energy between two arms, per event."""
    A = {ev: p for ev, p in dumps(a)}
    B = {ev: p for ev, p in dumps(b)}
    ev_both = sorted(set(A) & set(B))
    dE, dM, rows = [], [], []
    for ev in ev_both:
        ka = json.load(open(A[ev])).get('kine') or {}
        kb = json.load(open(B[ev])).get('kine') or {}
        if 'kine_reco_Enu' not in ka or 'kine_reco_Enu' not in kb:
            continue
        de = kb['kine_reco_Enu'] - ka['kine_reco_Enu']
        dm = (kb.get('kine_reco_add_energy') or 0) - (ka.get('kine_reco_add_energy') or 0)
        dE.append(de); dM.append(dm)
        if abs(dm) > 0.05:
            rows.append((ev, de, dm, ka.get('kine_reco_Enu'), kb.get('kine_reco_Enu')))
    import statistics
    print("\n=== %s  ->  %s   (%d events in both) ===" % (a, b, len(dE)))
    print("  events where the MASS term changed at all : %d of %d" % (len(rows), len(dE)))
    print("  sum   d(kine_reco_add_energy) : %+.1f MeV" % sum(dM))
    print("  sum   d(kine_reco_Enu)        : %+.1f MeV" % sum(dE))
    if dE:
        print("  median d(Enu) %+.3f MeV   mean %+.3f   max |d(Enu)| %.1f"
              % (statistics.median(dE), statistics.fmean(dE), max(abs(x) for x in dE)))
    for ev, de, dm, ea, eb in sorted(rows, key=lambda r: -abs(r[2])):
        print("    evt%-8d d(mass) %+8.2f MeV   d(Enu) %+9.2f MeV   Enu %.1f -> %.1f"
              % (ev, dm, de, ea, eb))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='work-pr140r2-off')
    ap.add_argument('--pair', nargs=2, metavar=('OFF', 'ON'))
    a = ap.parse_args()
    if a.pair:
        return pair(*a.pair)
    shipped(a.arm)
    return 0


if __name__ == '__main__':
    sys.exit(main())

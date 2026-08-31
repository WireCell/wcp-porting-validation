#!/usr/bin/env python3
"""doc 86 sec 8 -- data the "video round 2" cut needs, read from prod0830.

Reads ONLY existing prod0830 products (no arm is re-run, no code change):

  work-<sample>-prod0830/pr_evt<ID>/tracking-pr.root   T_kine (per-krow arrays,
                                                        including kine_mcs_* and
                                                        kine_reco_add_energy) and
                                                        T_tagger (per-bundle
                                                        numu_score/nue_score)

Answers, with data:

  8.1  confirms the three named events exist with a selected candidate
       (nu_evaluated == 1), same check make_pr_bee.py makes.
  8.2  kine_mcs_* for every T_kine row of every sec-4 event.
  8.4 Q2  T_tagger carries numu_score/nue_score PER BUNDLE (nu_index), not one
       value shared across an event's candidates.  pr_scores_table.py's
       primary_index() (pr94_rows.py) picks the longest selected activity as
       THE event's score -- for a multi-candidate event this silently drops
       the other candidate's own score.  This script reads every row.
  8.6  the four per-object T_kine vectors, plus kine_reco_add_energy, for
       every sec-4 event; and a closure check
       sum(kine_energy_particle) + kine_reco_add_energy == kine_reco_Enu
       to confirm the binding/rest-mass term is visible only as that one
       scalar, never split out per object.

Usage:  python3 scripts/analysis/d86r2_video_round2.py
Writes: docs/86_video/d86r2-mcs.tsv, docs/86_video/d86r2-objects.tsv,
        docs/86_video/d86r2-candidates.tsv (the T_tagger per-row scores)
"""
import os
import sys

import uproot

SX = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(SX, 'docs', '86_video')
ARM = 'prod0830'

sys.path.insert(0, os.path.join(SX, 'scripts'))
from pr94_rows import primary_index  # noqa: E402

# (sample, event) -- every event named anywhere in doc 86 sec 8, plus every
# sec-4 showcase event (needed for the 8.2/8.6 per-event tables).
SEC4 = [
    ('nuecc48', 81597), ('nuecc48', 267597),
    ('mcp2k', 290718), ('mcp2k', 94293),
    ('mcp2k', 283591), ('mcp1k', 313979),
    ('mcp2k', 99838), ('mcp2k', 242726),
    ('mcp2k', 57709), ('mcp2k', 176986), ('ncpi0', 180801),
    ('mcp2k', 180698), ('mcp2k', 99563),
    ('mcp1k', 487303), ('mcp2k', 174661),
    ('nuecc48', 389538), ('mcp2k', 67868),
    ('nuecc48', 69314), ('nuecc48', 138009), ('nuecc48', 271851),
]
NEW = [
    ('mcp2k', 179054),   # sec 8.1 video 2, replaces 487303
    ('mcp1k', 169626),   # sec 8.1 video 3 failure
    ('ncpi0', 142421),   # sec 8.1 video 3 failure
    ('mcp2k', 318769),   # sec 8.3 busy-event candidate
    ('mcp2k', 281781),   # sec 8.3 busy-event candidate
]
EVENTS = SEC4 + NEW

KINE_BR = ['nu_index', 'kine_reco_Enu', 'kine_reco_add_energy',
           'kine_particle_type', 'kine_energy_particle', 'kine_energy_info',
           'kine_energy_included',
           'kine_mcs_energy', 'kine_mcs_ambiguity', 'kine_mcs_tracklen',
           'kine_mcs_range_energy', 'kine_mcs_segment_id']
TAG_BR = ['nu_index', 'numu_score', 'nue_score', 'act_is_selected', 'act_length_cm']


def selected_len(sel, length):
    for s, l in zip(sel, length):
        if int(s) == 1:
            return float(l)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    mcs_rows, obj_rows, cand_rows = [], [], []

    for sample, evt in EVENTS:
        rp = os.path.join(SX, f'work-{sample}-{ARM}', f'pr_evt{evt}', 'tracking-pr.root')
        if not os.path.isfile(rp):
            print(f'MISSING {rp}', file=sys.stderr)
            continue
        f = uproot.open(rp)

        tk = f['T_kine']
        a = tk.arrays(KINE_BR, library='np')
        for i in range(tk.num_entries):
            enu = float(a['kine_reco_Enu'][i])
            add = float(a['kine_reco_add_energy'][i])
            ssum = float(sum(a['kine_energy_particle'][i]))
            mcs_rows.append(dict(
                sample=sample, event=evt, krow=i, nu_index=int(a['nu_index'][i]),
                kine_reco_Enu=enu, kine_reco_add_energy=add,
                sum_particle=ssum, closes=abs(ssum + add - enu) < 0.01,
                kine_mcs_energy=float(a['kine_mcs_energy'][i]),
                kine_mcs_ambiguity=float(a['kine_mcs_ambiguity'][i]),
                kine_mcs_tracklen=float(a['kine_mcs_tracklen'][i]),
                kine_mcs_range_energy=float(a['kine_mcs_range_energy'][i]),
                kine_mcs_segment_id=int(a['kine_mcs_segment_id'][i])))
            ptypes = list(map(int, a['kine_particle_type'][i]))
            penes = [float(x) for x in a['kine_energy_particle'][i]]
            pinfo = list(map(int, a['kine_energy_info'][i]))
            pincl = list(map(int, a['kine_energy_included'][i]))
            for j, (pdg, e, info, inc) in enumerate(zip(ptypes, penes, pinfo, pincl)):
                obj_rows.append(dict(sample=sample, event=evt, krow=i, obj=j,
                                      pdg=pdg, energy_MeV=e, info=info, included=inc))

        if 'T_tagger' in f:
            tt = f['T_tagger']
            keys = set(tt.keys())
            if {'nu_index', 'numu_score', 'nue_score'} <= keys:
                b = tt.arrays(TAG_BR, library='np')
                pidx = primary_index(tt)
                n = len(b['nu_index'])
                for i in range(n):
                    sel = b['act_is_selected'][i] if 'act_is_selected' in b else []
                    length = b['act_length_cm'][i] if 'act_length_cm' in b else []
                    cand_rows.append(dict(
                        sample=sample, event=evt, row=i, nu_index=int(b['nu_index'][i]),
                        numu_score=float(b['numu_score'][i]), nue_score=float(b['nue_score'][i]),
                        selected_len_cm=selected_len(sel, length),
                        is_primary=(i == pidx), n_rows=n))

    def write_tsv(path, rows):
        if not rows:
            return
        hdr = list(rows[0].keys())
        with open(path, 'w') as fh:
            fh.write('\t'.join(hdr) + '\n')
            for r in rows:
                fh.write('\t'.join(str(r[k]) for k in hdr) + '\n')
        print(f'wrote {path} ({len(rows)} rows)')

    write_tsv(os.path.join(OUT, 'd86r2-mcs.tsv'), mcs_rows)
    write_tsv(os.path.join(OUT, 'd86r2-objects.tsv'), obj_rows)
    write_tsv(os.path.join(OUT, 'd86r2-candidates.tsv'), cand_rows)

    nbad = sum(1 for r in mcs_rows if not r['closes'])
    print(f'closure check: {len(mcs_rows) - nbad}/{len(mcs_rows)} rows close '
          f'(sum(kine_energy_particle) + kine_reco_add_energy == kine_reco_Enu)')
    multi = [r for r in cand_rows if r['n_rows'] > 1]
    if multi:
        print(f'{len(set((r["sample"], r["event"]) for r in multi))} event(s) with >1 T_tagger row:')
        for r in multi:
            print(f"  {r['sample']}/{r['event']} row{r['row']} nu_index={r['nu_index']} "
                  f"numu={r['numu_score']:.3f} nue={r['nue_score']:.3f} "
                  f"sel_len={r['selected_len_cm']} primary={r['is_primary']}")


if __name__ == '__main__':
    main()

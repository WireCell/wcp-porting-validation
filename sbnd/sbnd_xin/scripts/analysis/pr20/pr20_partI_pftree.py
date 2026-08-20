#!/usr/bin/env python3
"""doc pr/20 Part I S10: read the T_kine particle-flow tree of both arms.

The score table's `kine_reco_Enu_MeV` says an event moved; it does not say why,
and on this population "why" is the whole question -- a NEGATIVE energy change
and a POSITIVE one turned out to be the same defect seen from opposite sides
(Part IX 6). This dumps, per dropped-companion event:

  dvtx   how far the reconstructed neutrino vertex moved (kine_nu_*_corr)
  dEnu   the energy change
  -[..] / +[..]   the (pdg, energy) pairs that left and entered the PF tree

`dvtx` is the column that matters: it separates "a shower node was removed and
nothing else changed" (the intended fix; 10 of 14 events, dvtx < 2.2 cm) from
"the pattern recognition re-solved the event" (dvtx 27-116 cm), which needs a
scan before anyone trusts it.

Repro:
  ./pr20_partI_pftree.py work-pi5cens-pr0 work-pi5cens-prB \\
      --drops /home/xqian/tmp/pr20exec/s10_mcp1k_B_drops.tsv
"""
import argparse
import sys

import uproot

PDG = {13: 'mu-', -13: 'mu+', 11: 'e-', -11: 'e+', 22: 'gamma', 211: 'pi+',
       -211: 'pi-', 2212: 'p', 2112: 'n', 111: 'pi0'}



# doc pr/94 Phase 5: T_tagger/T_kine hold ONE ROW PER IN-BEAM-WINDOW BUNDLE when
# the nu_per_bundle knob is on, so a hard [0] silently reports whichever bundle
# was enumerated first.  primary_index() reproduces the legacy meaning of "the
# candidate" (longest selected main activity) and falls back to 0 for pre-pr/94
# and knob-off files.
import os as _pr94_os, sys as _pr94_sys
_pr94_sys.path.insert(0, _pr94_os.path.join(
    _pr94_os.path.dirname(_pr94_os.path.abspath(__file__)), "../.."))
from pr94_rows import primary_index  # noqa: E402

def kine(root, arm, evt):
    f = uproot.open("%s/pr_evt%d/tracking-pr.root" % (arm, evt))
    i = primary_index(f["T_tagger"]) if "T_tagger" in f else 0
    a = f["T_kine"].arrays(library="np")
    vtx = [float(a['kine_nu_%s_corr' % c][i]) for c in "xyz"]
    parts = [(int(t), float(e)) for t, e in
             zip(a['kine_particle_type'][i], a['kine_energy_particle'][i])]
    return float(a['kine_reco_Enu'][i]), vtx, parts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("off_root")
    ap.add_argument("on_root")
    ap.add_argument("--drops", required=True,
                    help="pr20_partI_census.py's _drops.tsv (gives event and L_cm)")
    args = ap.parse_args()

    rows = [l.split('\t') for l in open(args.drops).read().splitlines()]
    hdr = rows[0]
    ie, iL = hdr.index('event'), hdr.index('L_cm')
    items = sorted((float(r[iL]), int(r[ie])) for r in rows[1:])

    print("%8s %7s  %-8s %-9s %s" % ("evt", "L_cm", "dvtx_cm", "dEnu", "PF-tree change"))
    for L, evt in items:
        try:
            e0, v0, p0 = kine(args.off_root, args.off_root, evt)
            e1, v1, p1 = kine(args.on_root, args.on_root, evt)
        except Exception as err:
            print("%8d %7.1f  %s" % (evt, L, err))
            continue
        dv = sum((x - y) ** 2 for x, y in zip(v0, v1)) ** 0.5
        lost = ["%s %.0f" % (PDG.get(t, t), e) for t, e in p0 if (t, e) not in p1]
        gain = ["%s %.0f" % (PDG.get(t, t), e) for t, e in p1 if (t, e) not in p0]
        print("%8d %7.1f  %-8.1f %-9.1f -[%s] +[%s]"
              % (evt, L, dv, e1 - e0, ", ".join(lost), ", ".join(gain)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

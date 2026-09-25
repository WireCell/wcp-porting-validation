#!/usr/bin/env python3
"""doc sbnd_xin/125 sec 9.3: the sentinel suite re-baselined on the CURRENT SBND production
(work-*-d123lgoppr + work-*-d123lgflippr), to replace the pr150s0 reference.

Runs scripts/pr127_sentinels.py UNCHANGED (M10), through the pr/149 wrapper's PF tolerance, and adds
eight entries to its KNOWN_OPEN registry.  Each is a shipped fix whose named event lost the fixed
behaviour on the PDHD/PDVD trajectory, which doc 118 flipped into SBND production (`tfull`, toolkit
675fd266, owner override of the frozen HOLD).  The loss was MEASURED BEFORE the flip:
docs/pr/150_figs/150_s3_sentinels_tfull.txt fails exactly these eight plus 66366 (which production now
passes), and doc 116 line 454 warned the flip would "re-open every downstream sentinel".  The d123
hit-flash / light-gate changes add none (pr150csp3bw, the same trajectory family on the old flash, fails
7 of the same set).  They stay OPEN -- tracked, not green -- until the vertex-choice retune pr/150 sec 1
names re-closes them; a NEW FAIL still fails the run.

Usage:  python3 scripts/analysis/d125/sentinels_prod.py --arms 'work-*-d123lgoppr' 'work-*-d123lgflippr'
Expected on 2026-09-25: 13 PASS, 0 FAIL, 10 OPEN, 7 INERT, 0 SKIP.
"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'pr127_sentinels.py')
spec = importlib.util.spec_from_file_location('pr127_sentinels', SRC)
mod = importlib.util.module_from_spec(spec)
sys.argv[0] = SRC
spec.loader.exec_module(mod)

_orig = mod.pf_texts
def pf_texts(arm, event):          # the doc pr/149 tolerance (analysis/pr149/sentinels_tolerant.py)
    try:
        return _orig(arm, event)
    except KeyError:
        print(f'# pr149 tolerance: {arm} evt {event} has no data/0/0-mc.json -> no PF node', file=sys.stderr)
        return []
mod.pf_texts = pf_texts

WHY = "doc 118 tfull trajectory flip -- recorded before it in docs/pr/150_figs/150_s3_sentinels_tfull.txt; "
D118 = {
    (69314,  "pr/125 K5"):      WHY + "calib showers 19 -> 22 (window [14, 20])",
    (171572, "pr/123 r2"):      WHY + "pf-orphan-guard-freed no longer fires; the muon IS a PF root (mu- 325 MeV)",
    (315167, "pr/93 r4"):       WHY + "the 150.7 cm proton (613 MeV on pr150s0) is not a PF node",
    (72786,  "pr/128 class A"): WHY + "control: cosmics still OUT, pass4_prox_guard declines 4 -> 3",
    (393505, "pr/129"):         WHY + "Enu 546.3 in window; the 267 MeV muon reads 262",
    (497311, "doc 84 r1"):      WHY + "the unbroken muon is split (459 + 749 MeV); range fallback not reached",
    (292643, "pr/130 B"):       WHY + "outcome holds (no pi0, e- < 200); the dvtx suppress line does not fire",
    (179369, "pr/130 B"):       WHY + "no 112 MeV pi0; the dvtx suppress line does not fire; mu- 1042 -> e- 1437",
}
clash = set(D118) & set(mod.KNOWN_OPEN_D144)
if clash:
    sys.exit(f"REFUSING: {clash} already KNOWN_OPEN in pr127_sentinels.py")
mod.KNOWN_OPEN_D144.update(D118)
sys.exit(mod.main())

#!/usr/bin/env python3
"""doc pr/149: run scripts/pr127_sentinels.py unchanged, except that an event whose mabc-pr.zip
carries no data/0/0-mc.json (the PF tree is absent because the arm lost the neutrino candidate)
reads as "no PF node" instead of aborting the whole run with KeyError.

The production script is imported, not edited (CLAUDE.md M10).  Its pf_texts() is wrapped:
KeyError on the missing member -> [] (which every pf_* assertion already treats as not seen).
Usage: identical to pr127_sentinels.py (--arms ...).
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'pr127_sentinels.py')
spec = importlib.util.spec_from_file_location('pr127_sentinels', SRC)
mod = importlib.util.module_from_spec(spec)
sys.argv[0] = SRC
spec.loader.exec_module(mod)
_orig = mod.pf_texts


def pf_texts(arm, event):
    try:
        return _orig(arm, event)
    except KeyError:
        print(f'# pr149 wrapper: {arm} evt {event} has no data/0/0-mc.json (no PF tree) -> no PF node',
              file=sys.stderr)
        return []


mod.pf_texts = pf_texts
sys.exit(mod.main())

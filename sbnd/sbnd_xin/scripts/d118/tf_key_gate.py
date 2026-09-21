#!/usr/bin/env python3
"""doc sbnd_xin/118 gate G2: the flipped production fit JSON is the measured arm's, key for key.

Two of the three components of this flip live in a file that is read at RUNTIME and never enters
the compiled config, so prod_cfg_gate.py is blind to them (the file says so itself, in
_comment_diffusion).  This is the instrument that is not blind: it compares the production
sbnd_track_fitting.json against the doc-116 `tfull` arm's 149_tf_sbnd_kf.json after dropping the
'_'-prefixed keys, which is exactly what load_trackfitting_config
(TaggerCheckSTM.cxx:1114, TaggerCheckNeutrino.cxx:4803) skips.

Exit 0 = the two files configure the fitter identically.

Usage: python3 scripts/d118/tf_key_gate.py [prod.json] [arm.json]
"""
import json, sys, os

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROD = sys.argv[1] if len(sys.argv) > 1 else \
    "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json"
ARM = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SX, "docs/pr/149_figs/149_tf_sbnd_kf.json")


def live(path):
    with open(path) as fh:
        d = json.load(fh)
    return {k: v for k, v in d.items() if not k.startswith("_")}


a, b = live(PROD), live(ARM)
print(f"# doc 118 G2 -- live (non '_') keys of the fit JSON")
print(f"prod : {PROD}  ({len(a)} live keys)")
print(f"arm  : {ARM}  ({len(b)} live keys)")
bad = 0
for k in sorted(set(a) | set(b)):
    if k not in a:
        print(f"  MISSING in prod : {k} = {b[k]!r}"); bad += 1
    elif k not in b:
        print(f"  EXTRA   in prod : {k} = {a[k]!r}"); bad += 1
    elif a[k] != b[k]:
        print(f"  DIFFERS         : {k}  prod={a[k]!r}  arm={b[k]!r}"); bad += 1
print("PASS -- identical" if not bad else f"FAIL -- {bad} difference(s)")
sys.exit(0 if not bad else 1)

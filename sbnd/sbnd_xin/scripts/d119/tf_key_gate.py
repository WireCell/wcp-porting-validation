#!/usr/bin/env python3
"""doc sbnd_xin/119 round 2 gate G2: is the FLIPPED production fit JSON the configuration that was
actually measured?

Fork by duplication (M10) of scripts/d118/tf_key_gate.py, which stays untouched as doc 118's
record; only the two files compared differ.

WHY THIS GATE EXISTS.  The 62-event output gate (scripts/d119/lever_gate.py pad) compared the
`pad` arm against the `ctl` arm, and the `pad` arm read its knobs from
docs/119_figs/119_tf_sbnd_pad.json via SBND_TRACKFIT_JSON.  Production after the flip reads
cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json instead.  Those are two different files, so
the output gate says nothing about production unless the two files carry the SAME live keys.  A
compiled-config hash cannot help: this file is read at RUNTIME by load_trackfitting_config with a
plain ifstream and never enters the compiled config at all (doc sbnd_xin/118 sec 6).

`_`-prefixed keys are skipped by load_trackfitting_config, so the comment blocks -- which DO differ,
deliberately -- are stripped before comparing.

Usage: python3 scripts/d119/tf_key_gate.py
"""
import collections
import json
import sys

PROD = "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json"
MEASURED = ("/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/"
            "docs/119_figs/119_tf_sbnd_pad.json")


def live(path):
    d = json.load(open(path), object_pairs_hook=collections.OrderedDict)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main():
    a, b = live(PROD), live(MEASURED)
    print("# doc sbnd_xin/119 G2 -- production fit JSON vs the JSON the 62-event gate measured")
    print(f"# production : {PROD}")
    print(f"# measured   : {MEASURED}")
    print(f"# live keys  : {len(a)} production, {len(b)} measured "
          f"(underscore-prefixed comment keys stripped -- they differ by design)")
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    differ = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    for label, keys in (("only in production", only_a), ("only in the measured file", only_b)):
        if keys:
            print(f"  {label}: {keys}")
    for k in differ:
        print(f"  VALUE DIFFERS {k}: production {a[k]!r} vs measured {b[k]!r}")
    rc = bool(only_a or only_b or differ)
    if not rc:
        print(f"\nPASS -- all {len(a)} live keys identical in name and value.  The 62-event output")
        print("       gate therefore applies to production as it now runs.")
        print(f"       proj_pad_wire={a.get('proj_pad_wire')} proj_pad_time={a.get('proj_pad_time')}")
    else:
        print("\nFAIL -- production is NOT the configuration that was gated")
    return rc


if __name__ == "__main__":
    sys.exit(main())

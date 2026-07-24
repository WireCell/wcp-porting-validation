#!/usr/bin/env python3
"""Build a Bee upload zip that carries the STM fit layer next to the usual ones.

The nusel chain writes two separate Bee trees per event:

  ql_evt<ID>/mabc-all-apa.zip     img-global, clustering-global, deadareas, op
  nusel_evt<ID>/mabc-pr.zip       clustering-global, stm_fit-global (knob on),
                                  deadareas

Both are dumped from the same T0-corrected frame (verified: the two
clustering-global layers are point-for-point the same range/count), so they
can be merged into one Bee event.  This script takes the QL zip as the base
and adds the PR zip's `stm_fit` layer, whose q column carries the fitted
dQ/dx as q = dQ*0.1 - 1000 (the uBooNE `track_fit` convention).

Usage:
  python3 make_stmfit_bee.py -w work-mcp10-stmon -o upload_286241.zip 286241
  ./upload-to-bee.sh upload_286241.zip
"""
import argparse
import os
import sys
import zipfile

PR_LAYERS = ("stm_fit-global",)   # layers taken from mabc-pr.zip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--work-root', required=True,
                    help='work root holding ql_evt<ID>/ and nusel_evt<ID>/')
    ap.add_argument('-o', '--output', required=True, help='output upload zip')
    ap.add_argument('events', nargs='+', help='event ids in Bee-index order')
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"ERROR: {args.output} exists; refusing to overwrite")

    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as out:
        for bee_idx, evt in enumerate(args.events):
            ql = os.path.join(args.work_root, f'ql_evt{evt}', 'mabc-all-apa.zip')
            pr = os.path.join(args.work_root, f'nusel_evt{evt}', 'mabc-pr.zip')
            if not os.path.isfile(ql):
                sys.exit(f"ERROR: missing {ql}")
            if not os.path.isfile(pr):
                sys.exit(f"ERROR: missing {pr}")

            n = 0
            with zipfile.ZipFile(ql) as zin:
                for name in zin.namelist():
                    base = os.path.basename(name)
                    if not base.startswith('0-'):
                        continue
                    out.writestr(f'data/{bee_idx}/{bee_idx}-{base[2:]}', zin.read(name))
                    n += 1

            got = []
            with zipfile.ZipFile(pr) as zin:
                for name in zin.namelist():
                    base = os.path.basename(name)
                    layer = base[2:].rsplit('.json', 1)[0]
                    if layer not in PR_LAYERS:
                        continue
                    out.writestr(f'data/{bee_idx}/{bee_idx}-{base[2:]}', zin.read(name))
                    got.append(layer)
                    n += 1
            missing = [l for l in PR_LAYERS if l not in got]
            if missing:
                print(f"WARNING: evt {evt}: {pr} has no {missing} "
                      f"(was the run made with -stm-fit?)", file=sys.stderr)
            print(f"event {evt} -> bee index {bee_idx}: {n} layer(s) {got}")

    print(f"wrote {args.output}")


if __name__ == '__main__':
    main()

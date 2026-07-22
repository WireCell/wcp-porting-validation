#!/usr/bin/env python3
"""Merge per-event mabc-all-apa.zip Bee dumps into one multi-event upload zip.

Each work root ql_evt<ID>/mabc-all-apa.zip holds a single-event Bee tree
data/0/0-<layer>.json (layers: clustering-global, img-global,
channel-deadarea-apa*-face*, op).  Bee identifies events by the numeric
directory/file prefix, so this script re-keys event i (in the given order)
to data/<i>/<i>-<layer>.json and bundles everything into one zip for a
single upload-to-bee.sh call.

Usage:
  python3 merge_mabc_bee.py -w <work_root> -o <out.zip> <evt_id> [<evt_id> ...]

The event-id order given on the command line defines the Bee event index
(use the run_ql_evt.sh idx order for consistency with the samples).
"""
import argparse
import os
import sys
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--work-root', required=True,
                    help='work root holding ql_evt<ID>/mabc-all-apa.zip')
    ap.add_argument('-o', '--output', required=True, help='output upload zip')
    ap.add_argument('events', nargs='+', help='event ids in Bee-index order')
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"ERROR: {args.output} exists; refusing to overwrite")

    nlayers = 0
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as out:
        for bee_idx, evt in enumerate(args.events):
            src = os.path.join(args.work_root, f'ql_evt{evt}', 'mabc-all-apa.zip')
            if not os.path.isfile(src):
                sys.exit(f"ERROR: missing {src}")
            with zipfile.ZipFile(src) as zin:
                for name in zin.namelist():
                    # data/0/0-<layer>.json -> data/<bee_idx>/<bee_idx>-<layer>.json
                    parts = name.split('/')
                    if len(parts) != 3 or parts[0] != 'data':
                        sys.exit(f"ERROR: unexpected member {name} in {src}")
                    stem = parts[2]
                    layer = stem.split('-', 1)[1]
                    newname = f'data/{bee_idx}/{bee_idx}-{layer}'
                    out.writestr(newname, zin.read(name))
                    nlayers += 1
            print(f'  bee idx {bee_idx:2d} <- evt {evt}')
    print(f'{args.output}: {len(args.events)} events, {nlayers} members')


if __name__ == '__main__':
    main()

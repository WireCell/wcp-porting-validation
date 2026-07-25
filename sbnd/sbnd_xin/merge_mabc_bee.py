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
    ap.add_argument('-w', '--work-root', required=True, action='append',
                    help='work root holding the per-event Bee zips.  Repeatable: '
                         'the 30-event scan lives in three roots '
                         '(work-{mcp10,mcp1000,mcp1000b}-<tag>) and the first root '
                         'that has the event wins.')
    ap.add_argument('--src', default='ql_evt{evt}/mabc-all-apa.zip',
                    help='per-event source zip, relative to a work root, with '
                         '{evt} for the event id.  Default is the Q/L dump, which '
                         'is PRE-un-merge; pass '
                         '"nusel_evt{evt}/mabc-pr.zip" for the post-un-merge '
                         'geometry the taggers actually saw (doc 50 Q1).')
    ap.add_argument('-o', '--output', required=True, help='output upload zip')
    ap.add_argument('events', nargs='+', help='event ids in Bee-index order')
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"ERROR: {args.output} exists; refusing to overwrite")

    nlayers = 0
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as out:
        for bee_idx, evt in enumerate(args.events):
            rel = args.src.format(evt=evt)
            src = next((c for c in (os.path.join(w, rel) for w in args.work_root)
                        if os.path.isfile(c)), None)
            if src is None:
                sys.exit("ERROR: no work root has " + rel + " (tried: "
                         + ", ".join(args.work_root) + ")")
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

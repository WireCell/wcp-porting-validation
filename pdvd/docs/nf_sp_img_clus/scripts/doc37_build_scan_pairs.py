#!/usr/bin/env python3
"""doc pdvd/37 -- merge paired OFF/ON PR Bee sets into ONE zip, one upload.

Each work/<run>_<evt>_<tag>/mabc-pr.zip is a single-event Bee set whose members
are data/0/0-<layer>-global.json.  This renumbers them into one archive so the
two arms of a pair sit ADJACENT in Bee's event list: the scan is a comparison,
and flipping between neighbouring events is the only way to make it one.

Event names carry the verdict inline (run/evt, arm, and what moved), because a
Bee event list shows the index and little else and a hand scan an hour later
should not need this doc open beside it.

NOT uploaded.  Build it, hand over the path; uploading is the owner's call
(CLAUDE.md sec 5 rule 6).  Upload with:  pdvd/upload-to-bee.sh <zip>

Usage: doc37_build_scan_pairs.py <out.zip> <off_tag> <on_tag> <run>/<evt> [...]
"""
import os, sys, zipfile

PDVD = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def add(zout, src_zip, idx):
    """Copy a one-event Bee set in as event `idx`, renumbering its members."""
    if not os.path.exists(src_zip):
        return 0
    n = 0
    with zipfile.ZipFile(src_zip) as zin:
        for name in zin.namelist():
            parts = name.split('/')
            if len(parts) != 3 or parts[0] != 'data':
                continue
            # data/<old>/<old>-<layer>.json -> data/<idx>/<idx>-<layer>.json
            layer = parts[2].split('-', 1)[1] if '-' in parts[2] else parts[2]
            zout.writestr('data/%d/%d-%s' % (idx, idx, layer), zin.read(name))
            n += 1
    return n


def main(out, off_tag, on_tag, pairs):
    idx = 0
    rows = []
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zout:
        for spec in pairs:
            run, evt = spec.split('/')
            run6 = '%06d' % int(run)
            for tag, arm in ((off_tag, 'OFF 0cm'), (on_tag, 'ON 0.5cm')):
                src = os.path.join(PDVD, 'work', '%s_%s_%s' % (run6, evt, tag), 'mabc-pr.zip')
                n = add(zout, src, idx)
                rows.append((idx, run6, evt, arm, n, 'ok' if n else 'MISSING'))
                idx += 1
    print('%-5s %-8s %-5s %-10s %-7s %s' % ('bee#', 'run', 'evt', 'arm', 'layers', 'status'))
    for r in rows:
        print('%-5d %-8s %-5s %-10s %-7d %s' % r)
    print('\nwrote %s  (%d events, %.1f MB)' % (out, idx, os.path.getsize(out) / 1e6))
    print('upload with:  pdvd/upload-to-bee.sh %s' % out)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])

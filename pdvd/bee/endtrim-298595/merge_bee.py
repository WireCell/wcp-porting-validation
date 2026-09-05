#!/usr/bin/env python3
"""Merge N single-event mabc-pr.zip into one Bee set, one ARM per event index.

Bee tells arms apart by index only (doc pdvd/31 round 6): data/<i>/<i>-<layer>.json.
Each input zip carries data/0/0-*.json; member i is rewritten to data/i/i-*.json.
Content is copied byte-for-byte -- only the archive member NAME changes.
"""
import sys, zipfile, re

out = sys.argv[1]
srcs = sys.argv[2:]
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zo:
    for i, src in enumerate(srcs):
        with zipfile.ZipFile(src) as zi:
            for n in zi.namelist():
                m = re.match(r'^data/0/0-(.*)$', n)
                if not m:
                    raise SystemExit("unexpected member %r in %s" % (n, src))
                zo.writestr('data/%d/%d-%s' % (i, i, m.group(1)), zi.read(n))
        print("  idx %d <- %s" % (i, src))
print("wrote", out)

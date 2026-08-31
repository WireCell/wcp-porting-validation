#!/usr/bin/env python3
# doc pr/138 A1.2 -- resolve "which uploaded Bee set holds this event, at what index".
"""Bee links for the split scan, built from the records already in bee/.

Every uploaded round leaves a pair behind: `<name>.url` holding the set URL, and
`<name>.index.txt` mapping bee_idx -> event.  That pair is all a per-event deep
link needs, because Bee addresses an event by its INDEX IN THE SET, not by run
and event number:

    https://www.phy.bnl.gov/twister/bee/set/<uuid>/event/<bee_idx>/

So this module scans bee/*/ once, joins each .url to its index, and answers
"give me every link that shows event N".  Nothing is uploaded here and nothing is
fetched -- it is a read over committed sidecars.

The consequence worth stating: a set that was never uploaded has no .url and
therefore contributes nothing, which is why the purpose-built pr137r2 package
supplies no links until the owner authorises its upload (CLAUDE.md sec 5.6).
"""
import os, glob, collections

BEE_ROOT = 'bee'
EVENT_URL = "%sevent/%d/"


def _index_events(path):
    """{event: bee_idx} from an annotated index file"""
    out = {}
    for line in open(path, errors='replace'):
        if line.startswith('#'):
            continue
        f = line.rstrip('\n').split('\t')
        if len(f) < 2:
            continue
        try:
            idx, ev = int(f[0]), int(f[1])
        except ValueError:
            continue
        out.setdefault(ev, idx)
    return out


def scan(root=BEE_ROOT):
    """{event: [(set_name, url, bee_idx), ...]}, newest set first."""
    by_event = collections.defaultdict(list)
    for url_path in sorted(glob.glob(os.path.join(root, '*', '*.url')),
                           key=lambda p: -os.path.getmtime(p)):
        d, base = os.path.dirname(url_path), os.path.basename(url_path)[:-4]
        try:
            url = open(url_path).read().strip().split()[0]
        except Exception:
            continue
        if '/set/' not in url:
            continue
        # exact-basename index first; a directory's single index is the fallback
        cand = os.path.join(d, base + '.index.txt')
        if not os.path.exists(cand):
            others = glob.glob(os.path.join(d, '*.index.txt'))
            cand = others[0] if len(others) == 1 else None
        if not cand or not os.path.exists(cand):
            continue
        stem = url if url.endswith('/') else url + '/'
        stem = stem[:stem.rindex('event/')] if 'event/' in stem else stem
        for ev, idx in _index_events(cand).items():
            by_event[ev].append((base, EVENT_URL % (stem, idx), idx))
    return by_event


def links_html(by_event, event, limit=6):
    """one line of <a> tags for the viewer, or a note if the event is in no set."""
    hits = by_event.get(event, [])
    if not hits:
        return ("<span style='color:#a33'>no uploaded Bee set holds evt%d</span>"
                " &nbsp;<span style='color:#888'>(bee/pr137r2 is built but not "
                "uploaded)</span>" % event)
    out = []
    for name, url, idx in hits[:limit]:
        out.append("<a href='%s' target='_blank' rel='noopener' "
                   "style='margin-right:10px'>%s [%d]</a>" % (url, name, idx))
    extra = '' if len(hits) <= limit else " <span style='color:#888'>+%d more</span>" % (len(hits) - limit)
    return "".join(out) + extra

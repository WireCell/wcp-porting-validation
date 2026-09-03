#!/usr/bin/env python3
"""doc 99 round 3 -- what did the WRITE knob actually move inside the stage-A archive?

The pctree tarball names its members by TENSOR INDEX
(clustering_tensor_<evt>_<n>_{metadata.json,array.npy}), and adding point clouds
renumbers every tensor after the insertion point.  So a member-name diff reports
"everything moved" and answers nothing.  This keys on each tensor's own
`datapath` instead -- the stable, meaningful name of what the tensor IS -- and
reports ADDED / REMOVED / CHANGED / same datapaths.

The question it exists to answer is a stop-and-ask one (CLAUDE.md 5.5): the
merge at QLMatching.cxx:1109 runs BEFORE matching at :1351, so if carrying every
input's flash PCs hands the matcher more candidate flashes, MATCHING ITSELF
moves and the write knob is a reconstruction change, not an archive-content
change.  Identical cluster/blob datapaths are what license "optical only".

Usage: d99r3_pctree_datapath_diff.py <arm-a-root> <arm-b-root> [--stage ql|pr] [max-events]
"""
import hashlib, json, os, re, sys, tarfile
from multiprocessing import Pool


def load(path):
    """-> {datapath: sha256(metadata + array payload)}, plus the top-level index map."""
    out = {}
    with tarfile.open(path) as tf:
        infos = {m.name: m for m in tf.getmembers() if m.isfile()}
        # index -> (meta bytes, array bytes)
        blobs = {}
        for name in sorted(infos):
            body = tf.extractfile(infos[name]).read()
            if name.endswith('_metadata.json'):
                blobs.setdefault(name[:-len('_metadata.json')], {})['m'] = body
            elif name.endswith('_array.npy'):
                blobs.setdefault(name[:-len('_array.npy')], {})['a'] = body
        for _, b in blobs.items():
            m = b.get('m', b'{}')
            try:
                md = json.loads(m)
            except Exception:
                md = {}
            dp = md.get('datapath')
            if dp is None:
                continue
            h = hashlib.sha256()
            # 'pointclouds'/'lpcmaps'/'items' hold TENSOR INDICES, which renumber
            # whenever a PC is added.  Hash them separately so a pure renumber is
            # not reported as a content change.
            structural = {k: md[k] for k in ('pointclouds', 'lpcmaps', 'items') if k in md}
            content = {k: v for k, v in md.items() if k not in ('pointclouds', 'lpcmaps', 'items')}
            h.update(json.dumps(content, sort_keys=True).encode())
            h.update(b.get('a', b''))
            # strip the event id so counts aggregate across events
            dpn = re.sub(r'^pointtrees/\d+/', 'pointtrees/<evt>/', dp)
            out[dpn] = (h.hexdigest(), bool(structural))
    return out


def one(job):
    a, b, evt = job
    A, B = load(a), load(b)
    ka, kb = set(A), set(B)
    added = sorted(kb - ka)
    removed = sorted(ka - kb)
    changed = sorted(k for k in (ka & kb) if A[k][0] != B[k][0])
    return evt, added, removed, changed, len(ka), len(kb)


def main():
    ra, rb = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    stage = 'ql'
    if '--stage' in rest:
        i = rest.index('--stage'); stage = rest[i + 1]; del rest[i:i + 2]
    lim = int(rest[0]) if rest else 0
    pfx = 'ql_evt' if stage == 'ql' else 'pr_evt'
    arch = 'pctree-evt%s.tar.gz' if stage == 'ql' else 'pctree-pr-evt%s.tar.gz'
    evts = sorted(d[len(pfx):] for d in os.listdir(ra) if d.startswith(pfx))
    if lim:
        evts = evts[:lim]
    jobs = []
    for e in evts:
        pa = os.path.join(ra, pfx + e, arch % e)
        pb = os.path.join(rb, pfx + e, arch % e)
        if os.path.exists(pa) and os.path.exists(pb):
            jobs.append((pa, pb, e))
    if not jobs:
        print('REFUSE: 0 event pairs -- nothing compared'); return 1
    import collections
    cadd, crem, cchg = collections.Counter(), collections.Counter(), collections.Counter()
    nev = 0
    with Pool(8) as p:
        for evt, added, removed, changed, na, nb in p.imap_unordered(one, jobs):
            nev += 1
            for k in added:   cadd[k] += 1
            for k in removed: crem[k] += 1
            for k in changed: cchg[k] += 1
    print('%d event pairs compared (%s vs %s)' % (nev, os.path.basename(ra), os.path.basename(rb)))
    def show(title, c):
        print('\n%s: %d distinct datapaths' % (title, len(c)))
        for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:60]:
            print('  %5d/%d  %s' % (v, nev, k))
    show('ADDED in B', cadd)
    show('REMOVED in B', crem)
    show('CHANGED content', cchg)
    return 0


if __name__ == '__main__':
    sys.exit(main())

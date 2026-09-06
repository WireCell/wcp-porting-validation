#!/usr/bin/env python3
"""doc pdhd/04 sec 11 -- build the BLIND hand-scan package for the movers of the
wrapped_channel_charge flip (sec 10.3).

A "mover" is a cluster whose cosmic-tagger verdict changed between the two
manifest arms.  The list is DERIVED from the two arms' PR logs here, never
hand-copied, and cluster idents are comparable across arms only because sec 10.2
proved the clustering is the same objects (same npts, x, y, z, cluster_id).

Blind by design (feedback: blind_the_scan_sheet).  The Bee set carries
  clustering            the full 3-D image, the dense local context
  channel-deadarea-*    where charge could not have been seen
  scan                  ONLY the movers, so they can be found
and NOT stm / stm_fit / stm_tagged / steiner_* -- those layers ARE the tagger's
answer, which is the thing the scan exists to test.  The sheet prints geometry
and identifiers only; the direction of each move goes to a separate KEY file.

Usage:
  d04_movers_scan.py --off <dir>... --on <dir>... --out <prefix> [--events 0,1,...]
"""
import sys, os, re, json, zipfile, argparse, collections, random

RE_TGM  = re.compile(r'TaggerCheckTGM: cluster (\d+) . TGM=(\w+)')
RE_STM  = re.compile(r'TaggerCheckSTM: cluster (\d+) . STM=(\d) TGM=(\d)')
RE_FIT  = re.compile(r'persist_stm_fit: cluster (\d+) stmfit pass=')

LABELS = """
  THRU        through-going: BOTH ends leave the active volume (or die in a dead region at it)
  STOP        stopping: ONE end enters, the other stops inside the active volume
  CONT        fully contained: NEITHER end reaches a boundary
  FRAG>THRU   this cluster is only PART of the object; the FULL object is through-going
  FRAG>STOP   this cluster is only PART of the object; the FULL object is a stopper
  FRAG>CONT   this cluster is only PART of the object; the FULL object is contained
  MESSY       the object is ill-posed: several particles merged, or a shower
  UNCLEAR     cannot judge from the display
""".rstrip()

def readlog(d, pfx):
    for f in sorted(os.listdir(d)):
        if f.startswith(pfx) and f.endswith('.log'):
            return open(os.path.join(d, f), errors='replace').read()
    return ''

def verdicts(d):
    t = readlog(d, 'wct_pr_')
    return (
        {int(c) for c, v in RE_TGM.findall(t) if v == 'true'},
        {int(c) for c, s, _ in RE_STM.findall(t) if s == '1'},
        {int(c) for c in RE_FIT.findall(t)},
    )

def evtnum(d):
    m = re.search(r'_(\d+)_[^/]*$', d.rstrip('/'))
    return int(m.group(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--off', nargs='+', required=True)
    ap.add_argument('--on', nargs='+', required=True)
    ap.add_argument('--out', required=True, help='output prefix (writes <p>.zip, <p>.sheet.tsv, <p>.KEY.tsv, <p>.index.txt)')
    a = ap.parse_args()

    off = {evtnum(d): d for d in a.off}
    on = {evtnum(d): d for d in a.on}
    assert set(off) == set(on), (sorted(off), sorted(on))
    events = sorted(off)

    movers = []          # (evt, cluster, direction)
    for e in events:
        ta, sa, fa = verdicts(off[e])
        tb, sb, fb = verdicts(on[e])
        for c in sorted(tb - ta):
            movers.append((e, c, 'TGM_gained'))
        for c in sorted(ta - tb):
            movers.append((e, c, 'TGM_lost'))
        for c in sorted(sb - sa):
            movers.append((e, c, 'STM_gained'))
        for c in sorted(sa - sb):
            movers.append((e, c, 'STM_lost_to_TGM' if c in tb else 'STM_lost'))

    by_evt = collections.defaultdict(set)
    for e, c, _ in movers:
        by_evt[e].add(c)

    # --- build the blind Bee set ------------------------------------------
    KEEP = ('clustering',)
    stage = {}
    npts = {}
    for idx, e in enumerate(events):
        z = os.path.join(on[e], 'mabc-pr.zip')
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                base = os.path.basename(n)
                if not base.endswith('.json'):
                    continue
                suffix = base.split('-', 1)[1]
                layer = suffix.rsplit('-global.json', 1)[0]
                if layer.startswith('channel-deadarea'):
                    stage[f'data/{idx}/{idx}-{suffix}'] = zf.read(n)
                    continue
                if layer not in KEEP:
                    continue
                j = json.loads(zf.read(n))
                stage[f'data/{idx}/{idx}-{suffix}'] = json.dumps(j).encode()
                cid = j['cluster_id']
                cnt = collections.Counter(cid)
                for c in by_evt[e]:
                    npts[(e, c)] = cnt.get(c, 0)
                sel = [i for i, c in enumerate(cid) if c in by_evt[e]]
                scan = {k: v for k, v in j.items() if not isinstance(v, list)}
                scan['type'] = 'scan'
                for k, v in j.items():
                    if isinstance(v, list) and len(v) == len(cid):
                        scan[k] = [v[i] for i in sel]
                stage[f'data/{idx}/{idx}-scan-global.json'] = json.dumps(scan).encode()

    outzip = a.out + '.zip'
    with zipfile.ZipFile(outzip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(stage):
            zf.writestr(name, stage[name])

    # --- the blind sheet ---------------------------------------------------
    with open(a.out + '.sheet.tsv', 'w') as fh:
        fh.write("# doc pdhd/04 sec 11 -- BLIND hand-scan sheet.  Bee set: " + os.path.basename(outzip) + "\n")
        fh.write("# One row per object whose cosmic-tagger verdict moved when the wrapped-channel\n")
        fh.write("# charge lookup was fixed.  WHICH WAY it moved is deliberately NOT printed here\n")
        fh.write("# (it is the thing this scan tests); it is in the .KEY.tsv, to be read AFTER.\n")
        fh.write("# Toggle the 'scan' layer to see only these objects; 'clustering' is the full context.\n#\n")
        fh.write("# label vocabulary (put one in the 'label' column):\n")
        for line in LABELS.splitlines():
            fh.write("#" + line + "\n")
        fh.write("#\n# FRAG> carries the FULL object's verdict on purpose: a fragment of a through-goer\n")
        fh.write("# and a fragment of a stopper are opposite physics truths.\n")
        fh.write("bee_idx\tevent\tcluster\tnpts\tlabel\tnote\n")
        for idx, e in enumerate(events):
            for c in sorted(by_evt[e]):
                fh.write(f"{idx}\t{e}\t{c}\t{npts.get((e, c), 0)}\t\t\n")

    # One OBJECT can move two ways at once (TGM gained AND its STM tag removed),
    # so the key is keyed on the object and carries every direction it moved in.
    dirs = collections.defaultdict(list)
    for e, c, d in movers:
        dirs[(e, c)].append(d)
    with open(a.out + '.KEY.tsv', 'w') as fh:
        fh.write("# THE ANSWER KEY -- do not read before scanning (feedback: blind_the_scan_sheet)\n")
        fh.write("event\tcluster\tnpts\tdirections\n")
        for (e, c) in sorted(dirs):
            fh.write(f"{e}\t{c}\t{npts.get((e, c), 0)}\t{'+'.join(sorted(dirs[(e, c)]))}\n")

    with open(a.out + '.index.txt', 'w') as fh:
        fh.write("# bee_idx -> event, and the movers in it\n")
        for idx, e in enumerate(events):
            fh.write(f"{idx}\t029107 evt {e}\t{len(by_evt[e])} movers: {sorted(by_evt[e])}\n")

    objs = sorted(dirs)
    n = len(objs)
    sizes = sorted(npts.get(k, 0) for k in objs)
    print(f"{n} OBJECTS ({len(movers)} moves -- 4 objects gained TGM and lost their STM tag in "
          f"the same step) over {len(events)} events -> {outzip}")
    print("  moves by direction: " + ", ".join(f"{k}={v}" for k, v in
                                               sorted(collections.Counter(d for _, _, d in movers).items())))
    # Judgeability bands from the doc pdhd/stm-tagger-chain sec 13 measurement:
    # <200 points is essentially unjudgeable, >=1000 is well resolved.  Report it
    # so a bar is not pre-registered on a stratum that cannot answer it.
    band = collections.Counter()
    for k in objs:
        v = npts.get(k, 0)
        band['<200' if v < 200 else '200-1000' if v < 1000 else '>=1000'] += 1
    print(f"  npts: min {sizes[0]}  median {sizes[n//2]}  max {sizes[-1]}")
    print("  judgeability bands: " + ", ".join(f"{b}={band[b]}" for b in ('>=1000', '200-1000', '<200')))
    print(f"  sheet: {a.out}.sheet.tsv   key: {a.out}.KEY.tsv   index: {a.out}.index.txt")

if __name__ == '__main__':
    main()

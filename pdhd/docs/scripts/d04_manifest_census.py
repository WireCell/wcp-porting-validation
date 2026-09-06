#!/usr/bin/env python3
"""doc pdhd/04 sec 10 -- end-to-end census of the 6-event manifest, one arm per
call, for the wrapped_channel_charge A/B.

Repro:
  python3 docs/scripts/d04_manifest_census.py work/029107_{0,1,12,16,20,22}_d05mOFF
  python3 docs/scripts/d04_manifest_census.py work/029107_{0,1,12,16,20,22}_d05mON

Reads, per event directory:
  wct_clus_*.log   the Q/L stage: one row per matched (flash, cluster) bundle
                   and the per-flash best bundle strength
  mabc-all-apa.zip the clustering output the Q/L job wrote
  wct_pr_*.log     the cosmic taggers: TGM / STM / FC verdicts

Cluster IDENTS are NOT compared across arms -- the clustering itself moves, so
an ident means different objects in the two arms (feedback:
a_retile_ident_is_not_the_bee_cluster_id).  Only COUNTS and distributions are
reported, plus the per-flash strength, which is keyed on the flash.
"""
import sys, re, os, json, zipfile, collections, statistics

RE_BUNDLE = re.compile(r'bundle_flags: flash id (\d+) cluster ident (\d+) gidx (\d+)')
RE_BEST   = re.compile(r'best bundle strength ([0-9.eE+-]+) for flash (\d+)')
RE_TGM    = re.compile(r'TaggerCheckTGM: cluster (\d+) . TGM=(\w+)')
RE_STM    = re.compile(r'TaggerCheckSTM: cluster (\d+) . STM=(\d) TGM=(\d)')
RE_SKIP   = re.compile(r'TaggerCheckSTM: cluster (\d+) already TGM')
RE_FC     = re.compile(r'TaggerCheckFC: cluster (\d+) . FC=(\w+)')
RE_FIT    = re.compile(r'persist_stm_fit: cluster (\d+) stmfit pass=')

def readlog(d, pfx):
    for f in sorted(os.listdir(d)):
        if f.startswith(pfx) and f.endswith('.log'):
            return open(os.path.join(d, f), errors='replace').read()
    return ''

def nclusters(zpath):
    if not os.path.exists(zpath):
        return None
    ids = set()
    with zipfile.ZipFile(zpath) as zf:
        for n in zf.namelist():
            if n.endswith('.json') and 'deadarea' not in n:
                try:
                    j = json.loads(zf.read(n))
                except Exception:
                    continue
                if isinstance(j, dict) and 'cluster_id' in j:
                    ids |= set(j['cluster_id'])
    return len(ids)

hdr = ("event", "nclus", "bundles", "flashes", "clus_matched", "med_strength",
       "tgm_eval", "tgm_true", "stm_eval", "stm_skip", "stm_fitted", "stm_tagged",
       "fc_eval", "fc_true")
rows = []
for d in sys.argv[1:]:
    ev = os.path.basename(d.rstrip('/'))
    clus = readlog(d, 'wct_clus_')
    pr = readlog(d, 'wct_pr_')
    bundles = RE_BUNDLE.findall(clus)
    best = [(int(f), float(s)) for s, f in RE_BEST.findall(clus)]
    tgm = {int(c): v for c, v in RE_TGM.findall(pr)}
    stm = {int(c): int(s) for c, s, _ in RE_STM.findall(pr)}
    skip = {int(c) for c in RE_SKIP.findall(pr)}
    fc = {int(c): v for c, v in RE_FC.findall(pr)}
    fitted = {int(c) for c in RE_FIT.findall(pr)}
    rows.append(dict(zip(hdr, (
        ev,
        nclusters(os.path.join(d, 'mabc-all-apa.zip')),
        len(bundles),
        len({b[0] for b in bundles}),
        len({b[1] for b in bundles}),
        round(statistics.median([s for _, s in best]), 4) if best else None,
        len(tgm), sum(1 for v in tgm.values() if v == 'true'),
        len(stm), len(skip), len(fitted), sum(1 for v in stm.values() if v == 1),
        len(fc), sum(1 for v in fc.values() if v == 'true'),
    ))))

print("\t".join(hdr))
for r in rows:
    print("\t".join("" if r[k] is None else str(r[k]) for k in hdr))
tot = {k: sum(r[k] for r in rows if isinstance(r[k], int)) for k in hdr[1:]}
tot["med_strength"] = round(statistics.median(
    [r["med_strength"] for r in rows if r["med_strength"] is not None]), 4) \
    if any(r["med_strength"] is not None for r in rows) else ""
print("\t".join(["TOTAL"] + [str(tot.get(k, "")) for k in hdr[1:]]))

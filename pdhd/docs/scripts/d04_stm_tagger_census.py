#!/usr/bin/env python3
"""doc pdhd/04 -- per-event census of the STM TAGGER (TaggerCheckSTM) itself.

Repro:
  python3 docs/scripts/d04_stm_tagger_census.py work/029107_{12,1,16,20,22,0}_d04bee

Reads only the PR log (`wct_pr_*.log`) plus the Bee zip, and reports, per event:
  mains evaluated / skipped, why each main got NO stm fit, how many mains
  recorded a fit (= the population the four STM Bee layers draw), and how many
  the tagger TAGGED (flag_STM).  Bee-layer cluster counts come from the zip,
  never from a log-line count (a persist line is per fit PASS, not per cluster).
"""
import sys, re, os, json, zipfile, collections

RE_VERDICT = re.compile(r'TaggerCheckSTM: cluster (\d+) . STM=(\d) TGM=(\d)')
RE_SKIPTGM = re.compile(r'TaggerCheckSTM: cluster (\d+) already TGM')
RE_NOFIT   = re.compile(r'check_stm_conditions: cluster (\d+) no STM fit: (.*)$', re.M)
RE_PERSIST = re.compile(r'persist_stm_fit: cluster (\d+) stmfit pass=(\d+) status=(\S+) kink=(\S+) exit_L=(\S+) left_L=(\S+) npts=(\d+)')

def norm(reason):
    r = re.sub(r'\b\d+\b', 'N', reason.strip())
    return re.sub(r'\d+\.\d+', 'N.N', r)

rows = []
for d in sys.argv[1:]:
    logs = [f for f in os.listdir(d) if f.startswith('wct_pr_') and f.endswith('.log')]
    if not logs:
        print(f"[skip] {d}: no wct_pr_*.log", file=sys.stderr); continue
    txt = open(os.path.join(d, logs[0]), errors='replace').read()
    verdicts = {int(m.group(1)): int(m.group(2)) for m in RE_VERDICT.finditer(txt)}
    skipped_tgm = {int(m.group(1)) for m in RE_SKIPTGM.finditer(txt)}
    nofit = collections.Counter()
    nofit_cl = set()
    for m in RE_NOFIT.finditer(txt):
        nofit[norm(m.group(2))] += 1; nofit_cl.add(int(m.group(1)))
    passes = [m.groups() for m in RE_PERSIST.finditer(txt)]
    fitted = sorted({int(p[0]) for p in passes})
    tagged = sorted([c for c, s in verdicts.items() if s == 1])

    # the object set the Bee layers actually draw
    bee = {}
    z = os.path.join(d, 'mabc-pr.zip')
    if os.path.exists(z):
        zf = zipfile.ZipFile(z)
        for n in zf.namelist():
            if n.endswith('.json') and 'deadarea' not in n:
                j = json.loads(zf.read(n))
                if isinstance(j, dict) and 'cluster_id' in j:
                    bee[n.split('/')[-1].split('-', 1)[1].replace('-global.json', '')] = \
                        sorted(set(j['cluster_id']))
    rows.append((d, verdicts, skipped_tgm, nofit, fitted, tagged, bee, passes))

for d, verdicts, skipped_tgm, nofit, fitted, tagged, bee, passes in rows:
    ev = os.path.basename(d)
    print(f"\n=== {ev} ===")
    print(f"  mains with a TaggerCheckSTM verdict line : {len(verdicts)}"
          f"  (skipped, already TGM: {len(skipped_tgm)})")
    print(f"  mains that recorded a fit (Bee stm/stm_fit/steiner_* population): "
          f"{len(fitted)}  -> {fitted}")
    print(f"  mains TAGGED STM (flag_STM, Bee stm_tagged)              : "
          f"{len(tagged)}  -> {tagged}")
    print(f"  fit passes persisted: {len(passes)} over {len(fitted)} clusters")
    if nofit:
        print("  no-fit reasons:")
        for r, n in nofit.most_common():
            print(f"      {n:4d}  {r}")
    if bee:
        print("  Bee layers (nclusters): " +
              "  ".join(f"{k}={len(v)}" for k, v in sorted(bee.items())))
        for k, v in sorted(bee.items()):
            if k in ('stm', 'stm_fit', 'steiner_graph', 'steiner_terminals') and sorted(v) != fitted:
                print(f"      NOTE: layer {k} draws {sorted(v)} != fitted set from the log")

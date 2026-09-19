#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 4 -- the blind vertex scan of the > 10 cm movers, BOTH arms shown, arm hidden.

Fork by duplication (CLAUDE.md M10) of vtx_rules/selfscan.py prepare/score (untouched), changed in three ways:
  * an ITEM is (event, arm): every mover is rendered twice, once from each arm's calib dump, under an opaque
    item id (itemNNN) so the scanner cannot pair the two renderings or read the arm from a path; the item ->
    (sample, event, arm, role) key lives in <out>/KEY.json, which the scanner is never told about;
  * the dump the worker drives `scankit.py zoom` with is a COPY of that arm's calib dump at <out>/items/itemNNN/dump.json
    (no arm in the path);
  * scoring resolves a pick against that ITEM's candidate pool (every PR-graph vertex of that arm), then measures its
    distance to (1) the vtx105 click, (2) this arm's own main vertex, (3) the OTHER arm's main vertex, and classifies:
        CONFIRMS-LABEL      pick within TOL of the vtx105 click
        A0-WAS-WRONG        an 'away' mover (B's main vertex farther from the click than A's) whose picks on BOTH arms
                            land within TOL of B's main vertex and not within TOL of the click
        REGRESSION          an 'away' mover whose pick on the A item confirms the click (or A's main vertex) and whose
                            pick on the B item does not land on B's main vertex
        UNRESOLVED          anything else (abstentions, unclear, inconsistent picks)
    Calibration items (non-movers with vtx105 truth, s0 rendering only) report the plain doc pr/80 agreement.
Labels are written ONLY under <out>/ (scratch) and, when --record is given, to vertex_labels/<tag>/ as a NEW tag
(refused if the tag exists; vtx105 is never written to).

Usage:
  pr150_vtx_scan.py prepare --a pr150s0 --b pr150csp3bw --movers <tsv from vertex_tolerance --movers-tsv> \\
        --out /home/xqian/tmp/pr150/vtxscan --workers 6 [--calib-frac 0.10] [--seed 150]
  pr150_vtx_scan.py score --out /home/xqian/tmp/pr150/vtxscan [--tol 1.0]
"""
import argparse, csv, glob, json, os, random, shutil, sys, time
SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
sys.path.insert(0, os.path.join(SX, 'vtx_rules'))
import scankit, vtx_io  # noqa: E402


def dump_path(arm, sample, event):
    return f'{SX}/work-{sample}-{arm}/pr_evt{event}/calib-pr-evt{event}.json'


def candidates(raw):
    d = scankit.sanitize(raw)
    return [dict(vertex_id=v['id'], cluster_id=v['cluster_id'], degree=v.get('degree', 0), aliases=list(v.get('aliases') or []),
                 x=round(p[0], 2), y=round(p[1], 2), z=round(p[2], 2)) for v in scankit.candidates(d) for p in [scankit.vertex_xyz(v)]]


def prepare(a):
    if os.path.exists(a.out):
        sys.exit(f'REFUSING: {a.out} exists (a scan set is never rebuilt in place)')
    labs = {L['event']: L for L in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105)}
    movers = []
    for r in csv.DictReader(open(a.movers), delimiter='\t'):
        movers.append((r['sample'], int(r['event']), r.get('direction', ''), float(r.get('dA', 'nan')), float(r.get('dB', 'nan'))))
    pool = sorted({(s, e) for s, e, *_ in movers})
    rng = random.Random(a.seed)
    # calibration: labelled non-movers of the same samples with a calib dump in A
    samples = sorted({s for s, _ in pool})
    cal_pool = []
    for s in samples:
        for d in sorted(glob.glob(f'{SX}/work-{s}-{a.a}/pr_evt*')):
            e = int(os.path.basename(d)[6:])
            if f'evt{e}' in labs and (s, e) not in set(pool) and os.path.exists(dump_path(a.a, s, e)):
                cal_pool.append((s, e))
    ncal = max(1, int(round(a.calib_frac * len(pool))))
    cal = sorted(rng.sample(cal_pool, min(ncal, len(cal_pool))))
    items = [(s, e, a.a, 'mover') for s, e in pool] + [(s, e, a.b, 'mover') for s, e in pool] + [(s, e, a.a, 'calib') for s, e in cal]
    rng.shuffle(items)
    os.makedirs(f'{a.out}/items')
    key, manifest = [], []
    for i, (s, e, arm, role) in enumerate(items, 1):
        iid = f'item{i:03d}'
        src = dump_path(arm, s, e)
        if not os.path.exists(src):
            key.append(dict(item=iid, sample=s, event=e, arm=arm, role=role, missing=True)); continue
        ed = f'{a.out}/items/{iid}'
        os.makedirs(ed)
        shutil.copyfile(src, f'{ed}/dump.json')
        raw = json.load(open(src))
        cands = candidates(raw)
        made = scankit.prepare(f'{ed}/dump.json', ed, title=iid)
        manifest.append(dict(event=iid, dump=f'{ed}/dump.json', dir=ed, files=made, candidates=cands, scannable=bool(cands)))
        key.append(dict(item=iid, sample=s, event=e, arm=arm, role=role, missing=False))
    json.dump(manifest, open(f'{a.out}/manifest.json', 'w'), indent=1)
    json.dump(dict(a=a.a, b=a.b, seed=a.seed, movers=a.movers, items=key), open(f'{a.out}/KEY.json', 'w'), indent=1)
    todo = [m for m in manifest if m['scannable']]
    nw = max(1, min(a.workers, len(todo))); per = (len(todo) + nw - 1) // nw
    for w in range(nw):
        chunk = todo[w * per:(w + 1) * per]
        with open(f'{a.out}/worklist-{w}.txt', 'w') as fh:
            for m in chunk:
                fh.write(f"{m['event']}\t{m['dir']}\t{m['dump']}\n")
    print(f'prepared {len(todo)} scannable items ({len(pool)} movers x 2 arms + {len(cal)} calibration) in {a.out}, {nw} worklists; '
          f'{sum(1 for k in key if k["missing"])} items missing a dump; {sum(1 for m in manifest if not m["scannable"])} without candidates')


def resolve(m, pick):
    for c in m['candidates']:
        if pick.get('vertex_id') in [c['vertex_id']] + list(c.get('aliases') or []):
            return (c['x'], c['y'], c['z'])
    return None


def score(a):
    key = json.load(open(f'{a.out}/KEY.json'))
    manifest = {m['event']: m for m in json.load(open(f'{a.out}/manifest.json'))}
    picks, stamps = {}, {}
    for f in sorted(glob.glob(f'{a.out}/picks-*.json')):
        stamps[os.path.basename(f)] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))
        for p in json.load(open(f)):
            picks[p['event']] = p
    labs = {L['event']: L for L in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105)}
    items = {k['item']: k for k in key['items'] if not k['missing']}
    need = [i for i in items if i in manifest and manifest[i]['scannable']]
    missing = [i for i in need if i not in picks]
    print(f'# pr150 vertex scan score: {len(need)} scannable items, picks for {len(need) - len(missing)}, missing {len(missing)}; picks written at {stamps}')
    if missing:
        print(f'REFUSING to score: picks missing for {missing[:10]}...'); return 1
    by_ev = {}
    for i, k in items.items():
        if i not in manifest:
            continue
        p = picks[i]; m = manifest[i]
        pos = resolve(m, p)
        raw = json.load(open(m['dump']))
        mv = vtx_io.xyz(raw.get('main_vertex'))
        L = labs.get(f"evt{k['event']}")
        truth = L['truth'] if L else None
        rec = dict(item=i, arm=k['arm'], role=k['role'], vid=p.get('vertex_id'), conf=p.get('confidence', '-'), why=p.get('why', ''), pos=pos, main=mv,
                   d_truth=vtx_io.dist(truth, pos) if truth and pos else None, d_main=vtx_io.dist(mv, pos) if mv and pos else None,
                   reco_truth=vtx_io.dist(truth, mv) if truth and mv else None)
        by_ev.setdefault((k['sample'], k['event'], k['role']), {})[k['arm']] = rec
    ok = lambda d: d is not None and d <= a.tol
    rows, cls = [], {}
    print('sample\tevent\trole\tclass\tA_pick\tA_conf\tA_d_truth\tA_d_main\tB_pick\tB_conf\tB_d_truth\tB_d_main\tB_d_Amain\tA_d_Bmain\treco_A\treco_B')
    for (s, e, role), R in sorted(by_ev.items()):
        A, B = R.get(key['a']), R.get(key['b'])
        if role == 'calib':
            c = 'CALIB-AGREE' if ok(A['d_truth']) else ('CALIB-ABSTAIN' if A['vid'] is None else 'CALIB-DISAGREE')
            rows.append((s, e, role, c, A, None)); cls[c] = cls.get(c, 0) + 1; continue
        if A is None or B is None:
            c = 'UNRESOLVED-missing-arm'
        else:
            away = (B['reco_truth'] or 0) > (A['reco_truth'] or 0)
            a_on_click, b_on_click = ok(A['d_truth']), ok(B['d_truth'])
            b_on_bmain = ok(B['d_main']); a_on_bmain = ok(vtx_io.dist(B['main'], A['pos']) if B['main'] and A['pos'] else None)
            a_on_amain = ok(A['d_main'])
            if a_on_click and b_on_click:
                c = 'CONFIRMS-LABEL'
            elif away and b_on_bmain and a_on_bmain and not a_on_click and not b_on_click:
                c = 'A0-WAS-WRONG'
            elif away and (a_on_click or a_on_amain) and not b_on_bmain:
                c = 'REGRESSION'
            elif away and (a_on_click or a_on_amain) and b_on_bmain:
                c = 'REGRESSION-b-agrees'        # the scanner follows B's choice on B's rendering but the click on A's
            elif not away and b_on_click and not a_on_click:
                c = 'IMPROVEMENT-CONFIRMED'
            else:
                c = 'UNRESOLVED'
        rows.append((s, e, role, c, A, B)); cls[c] = cls.get(c, 0) + 1
    f = lambda x: '' if x is None else (f'{x:.2f}' if isinstance(x, float) else str(x))
    for s, e, role, c, A, B in rows:
        ab = (B['d_truth'] if B else None)
        print('\t'.join([s, str(e), role, c, f(A['vid']), A['conf'], f(A['d_truth']), f(A['d_main']),
                         f(B['vid']) if B else '', B['conf'] if B else '', f(ab), f(B['d_main']) if B else '',
                         f(vtx_io.dist(A['main'], B['pos']) if B and A['main'] and B['pos'] else None), f(vtx_io.dist(B['main'], A['pos']) if B and B['main'] and A['pos'] else None),
                         f(A['reco_truth']), f(B['reco_truth']) if B else '']))
    print('# classes: ' + ', '.join(f'{k} {v}' for k, v in sorted(cls.items())))
    json.dump(dict(scored_at=time.strftime('%Y-%m-%d %H:%M:%S'), picks_written=stamps, tol=a.tol,
                   rows=[dict(sample=s, event=e, role=role, cls=c, A={k: v for k, v in A.items() if k != 'why'} | {'why': A['why']},
                              B=({k: v for k, v in B.items()} if B else None)) for s, e, role, c, A, B in rows]),
              open(f'{a.out}/scored.json', 'w'), indent=1, default=str)
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--a', required=True); p.add_argument('--b', required=True); p.add_argument('--movers', required=True)
    p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=6); p.add_argument('--calib-frac', type=float, default=0.10); p.add_argument('--seed', type=int, default=150)
    q = sub.add_parser('score'); q.add_argument('--out', required=True); q.add_argument('--tol', type=float, default=1.0)
    a = ap.parse_args()
    sys.exit(prepare(a) if a.cmd == 'prepare' else score(a))

#!/usr/bin/env python3
"""Build a Bee zip for the colleague STM/TGM-tagger feedback sample (doc 93).

Fork of make_pr_bee.py (CLAUDE.md M10: make_pr_bee.py is live for the pr/NN
A/B rounds and stays byte-untouched).  Two differences, both forced by what
this sample is -- 8 events of which 7 are cosmic-tagged (TGM/STM) so
TaggerCheckNeutrino never selects a candidate and mabc-pr.zip carries no
point layers at all:

  1. make_pr_bee.py SKIPS the PR job's own `clustering-global` ("kept: the
     PR job's own copy is skipped").  Here it is carried, RENAMED to
     `clustering-pr-global` -- the post-PR in-scope cluster set.  That is
     the layer the colleague's own Bee set shows, and on a cosmic-tagged
     event it is the ONLY thing the PR job contributes, so dropping it would
     leave 7 of 8 events Q/L-only.
  2. An event with no selected nu candidate is NOT refused.  make_pr_bee.py
     refuses because the three PR *point* layers degenerate to a whole-event
     dump when nothing was selected -- but that is a statement about a zip
     that HAS those layers.  Here they are simply absent (verified: evt 4 and
     evt 25 mabc-pr.zip hold only deadarea + clustering-global), so there is
     no degenerate layer to ship.  Refusal is replaced by a per-event note.

Per event the output Bee tree is

  from <ql_root>/ql_evt<ID>/mabc-all-apa.zip
      img-global            the raw imaged blobs, per-APA post-QL cluster ids
      clustering-global     the Q/L clustering
      op                    flash <-> cluster match, drives flash navigation
      channel-deadarea-*    dead channel bands
  from <pr_root>/pr_evt<ID>/mabc-pr.zip
      clustering-pr-global  the post-PR in-scope set  (ALL events)
      track_fit-global      fitted trajectory, q = dQ*0.1 - 1000   (nu events)
      shower_track-global   track/shower split                     (nu events)
      vertices-global       PR graph vertices                      (nu events)
      mc                    the PR particle-flow jsTree            (nu events)

`clustering-pr-global` keeps its PR cluster ids on purpose: the remap bridge
is BUILT from that layer, so remapping it would recolour it as img-global and
destroy the PR clustering it exists to show.  Only the point layers are
remapped, exactly as in make_pr_bee.py.

Usage:
  python3 make_stmfb_bee.py -q work-stmfb8-ql -p work-stmfb8-pr \
      -o bee/stmfb8/stmfb8.zip 4 17 22 28 12 31 10 25
"""
import argparse
import collections
import json
import os
import re
import sys
import zipfile

# Layers taken from mabc-pr.zip.  'mc' is the particle-flow jsTree, not a point layer.
PR_POINT_LAYERS = ('track_fit-global', 'shower_track-global', 'vertices-global')
PR_LAYERS = PR_POINT_LAYERS + ('mc',)

CLUS_LAYER = 'clustering-global'
IMG_LAYER = 'img-global'
OP_LAYER = 'op'

# The PR job's own clustering-global, carried under this name so it sits beside
# the Q/L one instead of colliding with it.  Same name the colleague's set uses.
PR_CLUS_OUT = 'clustering-pr-global'

RE_SELECTED = re.compile(r'TaggerCheckNeutrino: selected main cluster ')


def _member(zf, layer):
    """Return the parsed JSON of the single-event member 0-<layer>.json, or None."""
    for name in zf.namelist():
        if os.path.basename(name) == f'0-{layer}.json':
            return json.loads(zf.read(name))
    return None


def nu_evaluated(pr_dir, evt):
    """True iff TaggerCheckNeutrino actually selected a candidate this event."""
    log = os.path.join(pr_dir, f'wct_pr_evt{evt}.log')
    if not os.path.isfile(log):
        return None                      # unknown -- caller decides
    with open(log, errors='replace') as fh:
        return any(RE_SELECTED.search(line) for line in fh)


def build_remap(img, op, pr_clus, pr_ids, evt):
    """PR cluster id -> img-global cluster id, over the union of the PR layers' ids."""
    q2ids = collections.defaultdict(set)
    for q, cid in zip(img['q'], img['cluster_id']):
        q2ids[q].add(cid)
    q2id = {q: next(iter(s)) for q, s in q2ids.items() if len(s) == 1}
    nambig = len(q2ids) - len(q2id)

    matched = set()
    for cids in (op.get('op_cluster_ids') or []):
        matched.update(cids)

    votes = collections.defaultdict(collections.Counter)
    unmapped = collections.Counter()
    for q, cid in zip(pr_clus['q'], pr_clus['cluster_id']):
        img_id = q2id.get(q)
        if img_id is None:
            unmapped[cid] += 1
        else:
            votes[cid][img_id] += 1

    mapping, rows = {}, []
    for cid in sorted(pr_ids):
        counter = votes.get(cid)
        if not counter:
            rows.append((cid, None, [], unmapped[cid], 'NO-BRIDGE'))
            continue
        ranked = counter.most_common()
        pick = next((i for i, _ in ranked if i in matched), None)
        note = 'flash-matched'
        if pick is None:
            pick = ranked[0][0]
            note = 'NO-FLASH-MATCH (fallback: dominant)'
        mapping[cid] = pick
        rows.append((cid, pick, ranked, unmapped[cid], note))
    nobridge = sum(1 for r in rows if r[4] == 'NO-BRIDGE')
    if nobridge:
        print(f"WARNING: evt {evt}: {nobridge}/{len(rows)} PR cluster ids have no "
              f"counterpart in the PR {CLUS_LAYER} layer; those keep their PR id "
              f"(they will not follow the flash filter)", file=sys.stderr)
    return mapping, rows, nambig


def remap_points(layer_json, mapping):
    """Relabel cluster_id only.  real_cluster_id (cluster*1000+seg) is preserved."""
    out = dict(layer_json)
    out['cluster_id'] = [mapping.get(c, c) for c in layer_json['cluster_id']]
    assert len(out['cluster_id']) == len(layer_json['cluster_id'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-q', '--ql-root', required=True, action='append',
                    help='root holding ql_evt<ID>/mabc-all-apa.zip; repeatable, '
                         'first root that has the event wins')
    ap.add_argument('-p', '--pr-root', required=True, action='append',
                    help='root holding pr_evt<ID>/mabc-pr.zip; repeatable')
    ap.add_argument('-o', '--output', required=True, help='output upload zip')
    ap.add_argument('-no-remap', '--no-remap', dest='remap', action='store_false',
                    help='keep the PR cluster ids (legacy behaviour)')
    ap.add_argument('--allow-unevaluated', action='store_true',
                    help='do not refuse events with no selected nu candidate')
    ap.add_argument('events', nargs='+', help='event ids in Bee-index order')
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"ERROR: {args.output} exists; refusing to overwrite")

    map_path = args.output.rsplit('.zip', 1)[0] + '.prid-map.txt'
    idx_path = args.output.rsplit('.zip', 1)[0] + '.index.txt'
    map_lines, idx_lines = [], []

    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as out:
        for bee_idx, evt in enumerate(args.events):
            qroot = next((w for w in args.ql_root
                          if os.path.isfile(os.path.join(w, f'ql_evt{evt}',
                                                         'mabc-all-apa.zip'))), None)
            if qroot is None:
                sys.exit(f"ERROR: no ql root has ql_evt{evt}/mabc-all-apa.zip "
                         f"(tried: {', '.join(args.ql_root)})")
            proot = next((w for w in args.pr_root
                          if os.path.isfile(os.path.join(w, f'pr_evt{evt}',
                                                         'mabc-pr.zip'))), None)
            if proot is None:
                sys.exit(f"ERROR: no pr root has pr_evt{evt}/mabc-pr.zip "
                         f"(tried: {', '.join(args.pr_root)})")
            pr_dir = os.path.join(proot, f'pr_evt{evt}')
            ok = nu_evaluated(pr_dir, evt)
            if ok is False:
                print(f"  evt {evt}: no nu candidate selected (cosmic-tagged) -- "
                      f"PR side contributes {PR_CLUS_OUT} only")
            if ok is None:
                print(f"WARNING: evt {evt}: no {pr_dir}/wct_pr_evt{evt}.log; cannot "
                      f"confirm a candidate was selected", file=sys.stderr)

            ql = os.path.join(qroot, f'ql_evt{evt}', 'mabc-all-apa.zip')
            pr = os.path.join(pr_dir, 'mabc-pr.zip')

            n = 0
            with zipfile.ZipFile(ql) as zin:
                for name in zin.namelist():
                    base = os.path.basename(name)
                    if not base.startswith('0-'):
                        continue
                    out.writestr(f'data/{bee_idx}/{bee_idx}-{base[2:]}', zin.read(name))
                    n += 1
                img = _member(zin, IMG_LAYER) if args.remap else None
                op = _member(zin, OP_LAYER) if args.remap else None

            got = []
            with zipfile.ZipFile(pr) as zin:
                pr_clus = _member(zin, CLUS_LAYER) if args.remap else None
                # One remap per event, over the union of the point layers' ids.
                mapping = {}
                if args.remap:
                    ids = set()
                    for lay in PR_POINT_LAYERS:
                        j = _member(zin, lay)
                        if j:
                            ids.update(j['cluster_id'])
                    if img is None or op is None or pr_clus is None:
                        print(f"WARNING: evt {evt}: cannot remap ({IMG_LAYER}/{OP_LAYER}/"
                              f"PR {CLUS_LAYER} missing); keeping PR cluster ids",
                              file=sys.stderr)
                    elif ids:
                        mapping, rows, nambig = build_remap(img, op, pr_clus, ids, evt)
                        print(f"  evt {evt}: PR cluster_id -> img-global space "
                              f"({len(ids)} ids, {nambig} ambiguous q fingerprints)")
                        for cid, pick, ranked, unm, note in rows:
                            comp = ', '.join(f'{i}:{c}' for i, c in ranked) or '-'
                            map_lines.append(
                                f"{evt}\t{bee_idx}\t{cid}\t{pick}\t{note}\t{comp}")

                for name in zin.namelist():
                    base = os.path.basename(name)
                    layer = base[2:].rsplit('.json', 1)[0]
                    if layer == CLUS_LAYER:
                        # the PR in-scope set, renamed so it does not collide
                        # with the Q/L clustering-global already written above
                        out.writestr(f'data/{bee_idx}/{bee_idx}-{PR_CLUS_OUT}.json',
                                     zin.read(name))
                        got.append(PR_CLUS_OUT)
                        n += 1
                        continue
                    if layer not in PR_LAYERS:
                        continue
                    payload = zin.read(name)
                    if mapping and layer in PR_POINT_LAYERS:
                        payload = json.dumps(
                            remap_points(json.loads(payload), mapping)).encode()
                    out.writestr(f'data/{bee_idx}/{bee_idx}-{base[2:]}', payload)
                    got.append(layer)
                    n += 1

            if PR_CLUS_OUT not in got:
                print(f"WARNING: evt {evt}: {pr} has no {CLUS_LAYER}", file=sys.stderr)
            missing = [l for l in PR_LAYERS if l not in got]
            if missing and ok is not False:
                print(f"WARNING: evt {evt}: {pr} has no {missing}", file=sys.stderr)
            print(f"event {evt} -> bee index {bee_idx}: {n} layer(s), PR {got}")
            idx_lines.append(f"{bee_idx}\t{evt}")

    print(f"wrote {args.output}")
    with open(idx_path, 'w') as fh:
        fh.write("# bee_idx\tevent\n")
        fh.write('\n'.join(idx_lines) + '\n')
    print(f"wrote {idx_path}")
    if map_lines:
        with open(map_path, 'w') as fh:
            fh.write(f"# PR cluster_id remap for {args.output}\n")
            fh.write("# event\tbee_idx\tpr_cluster_id\timg_cluster_id\tnote\t"
                     "img_constituents(id:npts)\n")
            fh.write('\n'.join(map_lines) + '\n')
        print(f"wrote {map_path}")


if __name__ == '__main__':
    main()

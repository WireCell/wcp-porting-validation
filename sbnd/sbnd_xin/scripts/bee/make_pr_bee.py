#!/usr/bin/env python3
"""Build a Bee upload zip carrying the full neutrino-PR layer set next to img-global.

Fork of make_stmfit_bee.py (CLAUDE.md M10: fork by duplication -- make_stmfit_bee.py
is live for make_scan_bee.sh and stays byte-untouched).  Two differences: the PR side
comes from an at-HEAD `run_pr_chain_batch.sh` arm (`pr_evt<ID>/mabc-pr.zip`, a
different directory from the Q/L arm), and FOUR PR layers are carried instead of one.

Per event the output Bee tree is

  from <ql_root>/ql_evt<ID>/mabc-all-apa.zip
      img-global            the raw imaged blobs, per-APA post-QL cluster ids
      clustering-global     the Q/L clustering (kept: the PR job's own copy is skipped)
      op                    flash <-> cluster match, drives Bee's flash navigation
      channel-deadarea-*    dead channel bands
  from <pr_root>/pr_evt<ID>/mabc-pr.zip
      track_fit-global      the fitted trajectory; q = dQ*0.1 - 1000, i.e. dQ/dx
      shower_track-global   track/shower separation: q = 0 track, q = 15000 shower,
                            real_cluster_id = cluster*1000 + segment  (per-particle
                            sub-clustering colour)
      vertices-global       every PR graph vertex; q = 15000 on the neutrino vertex
      mc                    the PR particle-flow jsTree

Cluster-id remapping (default ON; -no-remap to disable)
-------------------------------------------------------
Same bridge as make_stmfit_bee.py, applied to the three PR *point* layers: their
cluster_id lives in the PR job's id space, which is unrelated to the img-global ids the
`op` layer stores in op_cluster_ids, so without the remap the PR layers stay invisible
while stepping through flashes.  The bridge is the per-point charge q, which the img and
PR clustering dumps compute identically from the same blobs.

  PR layer cluster C -> PR clustering-global cluster C -> its points' q
                     -> img-global cluster ids -> pick a flash-matched representative

`real_cluster_id` is deliberately NOT rewritten: `cluster*1000 + segment` is the
per-particle sub-clustering that Bee colours by (it prefers real_cluster_id over
cluster_id when positive), and that is the thing this display exists to show.  `mc` is
a jsTree structure with no cluster ids and is copied verbatim.

Events with no selected neutrino candidate are REFUSED.  When TaggerCheckNeutrino
selects nothing, the three PR point layers degenerate into the same whole-event dump
(seen on evt 167904: 50481 identical points in all three), which would ship as a
meaningless display.  The signal is the log line
"TaggerCheckNeutrino: selected main cluster ..." (TaggerCheckNeutrino.cxx:345).

Usage:
  python3 make_pr_bee.py -q work-mcp1kall-d59k -p work-cathnu-mcp1k \
      -o /home/xqian/tmp/cath_a.zip 315497 286241 ...
  ./upload-to-bee.sh cath_a.zip
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
            if ok is False and not args.allow_unevaluated:
                sys.exit(f"ERROR: evt {evt}: TaggerCheckNeutrino selected no candidate; "
                         f"its PR layers are a degenerate whole-event dump.  Drop the "
                         f"event or pass --allow-unevaluated.")
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
                    if layer not in PR_LAYERS:
                        continue
                    payload = zin.read(name)
                    if mapping and layer in PR_POINT_LAYERS:
                        payload = json.dumps(
                            remap_points(json.loads(payload), mapping)).encode()
                    out.writestr(f'data/{bee_idx}/{bee_idx}-{base[2:]}', payload)
                    got.append(layer)
                    n += 1

            missing = [l for l in PR_LAYERS if l not in got]
            if missing:
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

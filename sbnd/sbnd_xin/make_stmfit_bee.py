#!/usr/bin/env python3
"""Build a Bee upload zip that carries the STM fit layer next to the usual ones.

The nusel chain writes two separate Bee trees per event:

  ql_evt<ID>/mabc-all-apa.zip     img-global, clustering-global, deadareas, op
  nusel_evt<ID>/mabc-pr.zip       clustering-global, stm_fit-global (knob on),
                                  deadareas

Both are dumped from the same T0-corrected frame (verified: the two
clustering-global layers are point-for-point the same range/count), so they
can be merged into one Bee event.  This script takes the QL zip as the base
and adds the PR zip's `stm_fit` layer, whose q column carries the fitted
dQ/dx as q = dQ*0.1 - 1000 (the uBooNE `track_fit` convention).

Cluster-id remapping (default ON; -no-remap to disable)
-------------------------------------------------------
The three layers live in *different* cluster-id spaces:

  img-global   ids of the per-APA post-QL clusters, dumped by clus_all_apa
               BEFORE its clustering pipeline.  These are the ids the `op`
               layer stores in op_cluster_ids, so Bee's flash "<" / ">"
               navigation with "show matching cluster" filters img-global
               correctly.
  clustering-global / stm_fit-global
               ids after the all-APA merge + examine_bundles flash-bundle
               collapse (and, in the PR job, after switch_scope).  A bundle
               id is unrelated to the img ids of the clusters it absorbed,
               so the flash filter never matches an stm_fit point and the fit
               layer stays invisible while stepping through flashes.

This script rewrites stm_fit-global's cluster_id into img-global's space so
the fit trajectories appear/disappear together with the QLMatch bundle the
flash arrows select.  The bridge is the per-point charge q, which both the
img and clustering dumps compute identically from the same blobs (the two
layers differ only in coordinates: img is raw x, clustering is T0-corrected
x plus the pos_offset y/z shift).  So:

  stm_fit cluster C  ->  PR clustering-global cluster C  ->  its points' q
                     ->  img-global cluster ids  ->  pick one representative

The representative is the highest-point-count img constituent that actually
appears in some flash's op_cluster_ids; only if none of the constituents is
flash-matched does it fall back to the overall dominant one (with a warning),
because an unmatched id would leave the fit invisible under every flash.

Side benefit: Bee colors a point by real_cluster_id when >0 else cluster_id,
and img-global has real_cluster_id == cluster_id, so after the remap each fit
renders in the same color as the img cluster it belongs to.

The PR cluster id is what tracking-stm.root encodes as cid*10+pass and what
the nusel TSV reports, so the PR->img table is printed for cross-reference.

Usage:
  python3 make_stmfit_bee.py -w work-mcp10-stmon -o upload_286241.zip 286241
  ./upload-to-bee.sh upload_286241.zip
"""
import argparse
import collections
import json
import os
import sys
import zipfile

PR_LAYERS = ("stm_fit-global",)   # layers taken from mabc-pr.zip

STM_LAYER = 'stm_fit-global'
CLUS_LAYER = 'clustering-global'
IMG_LAYER = 'img-global'
OP_LAYER = 'op'


def _member(zf, layer):
    """Return the parsed JSON of the single-event member 0-<layer>.json, or None."""
    for name in zf.namelist():
        if os.path.basename(name) == f'0-{layer}.json':
            return json.loads(zf.read(name))
    return None


def build_remap(img, op, pr_clus, stm, evt):
    """PR/stm cluster id -> img-global cluster id.

    Returns (mapping, report_rows).  Ids with no usable bridge are absent from
    the mapping and keep their original label.
    """
    # q fingerprint -> img cluster id, dropping q values shared by >1 img cluster.
    q2ids = collections.defaultdict(set)
    for q, cid in zip(img['q'], img['cluster_id']):
        q2ids[q].add(cid)
    q2id = {q: next(iter(s)) for q, s in q2ids.items() if len(s) == 1}
    nambig = len(q2ids) - len(q2id)

    # img ids that some flash is matched to (union over all flashes).
    matched = set()
    for cids in (op.get('op_cluster_ids') or []):
        matched.update(cids)

    # Per PR cluster: how many of its points fingerprint to each img cluster.
    votes = collections.defaultdict(collections.Counter)
    unmapped = collections.Counter()
    for q, cid in zip(pr_clus['q'], pr_clus['cluster_id']):
        img_id = q2id.get(q)
        if img_id is None:
            unmapped[cid] += 1
        else:
            votes[cid][img_id] += 1

    stm_ids = sorted(set(stm['cluster_id']))
    mapping, rows = {}, []
    for cid in stm_ids:
        counter = votes.get(cid)
        if not counter:
            print(f"WARNING: evt {evt}: stm_fit cluster {cid} has no counterpart in "
                  f"the PR {CLUS_LAYER} layer (or none of its points fingerprinted); "
                  f"left unlabelled", file=sys.stderr)
            rows.append((cid, None, [], unmapped[cid], 'NO-BRIDGE'))
            continue
        ranked = counter.most_common()
        pick = next((i for i, _ in ranked if i in matched), None)
        note = 'flash-matched'
        if pick is None:
            pick = ranked[0][0]
            note = 'NO-FLASH-MATCH (fallback: dominant)'
            print(f"WARNING: evt {evt}: stm_fit cluster {cid} -> img {pick}: none of its "
                  f"img constituents {[i for i, _ in ranked]} is in any op_cluster_ids; "
                  f"the fit will not show under the flash filter", file=sys.stderr)
        mapping[cid] = pick
        rows.append((cid, pick, ranked, unmapped[cid], note))

    # Two PR bundles can land on the same img constituent if the all-APA pipeline
    # split one img cluster across them; the two fits would then be one color and
    # inseparable under the flash filter.
    dupes = [i for i, c in collections.Counter(mapping.values()).items() if c > 1]
    for i in dupes:
        print(f"WARNING: evt {evt}: img id {i} is the label for several stm_fit "
              f"clusters {sorted(c for c, v in mapping.items() if v == i)}; those fits "
              f"are indistinguishable in Bee", file=sys.stderr)
    return mapping, rows, nambig


def remap_stm(stm, mapping):
    """Return stm with cluster_id relabelled; every other array untouched."""
    out = dict(stm)
    out['cluster_id'] = [mapping.get(c, c) for c in stm['cluster_id']]
    assert len(out['cluster_id']) == len(stm['cluster_id'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--work-root', required=True, action='append',
                    help='work root holding ql_evt<ID>/ and nusel_evt<ID>/.  '
                         'Repeatable (as in merge_mabc_bee.py): the 30-event '
                         'scan lives in three roots and the first root that '
                         'has the event wins.')
    ap.add_argument('-o', '--output', required=True, help='output upload zip')
    ap.add_argument('-no-remap', '--no-remap', dest='remap', action='store_false',
                    help='keep the PR cluster ids in stm_fit-global (legacy behaviour)')
    ap.add_argument('events', nargs='+', help='event ids in Bee-index order')
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"ERROR: {args.output} exists; refusing to overwrite")

    # PR cluster id <-> img cluster id, persisted next to the zip: the PR id is
    # what tracking-stm.root encodes as cid*10+pass and what the nusel TSV
    # reports, so a scan that starts from a Bee color needs the way back.
    map_path = args.output.rsplit('.zip', 1)[0] + '.stmid-map.txt'
    map_lines = []

    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as out:
        for bee_idx, evt in enumerate(args.events):
            root = next((w for w in args.work_root
                         if os.path.isfile(os.path.join(w, f'ql_evt{evt}',
                                                        'mabc-all-apa.zip'))), None)
            if root is None:
                sys.exit(f"ERROR: no work root has ql_evt{evt}/mabc-all-apa.zip "
                         f"(tried: {', '.join(args.work_root)})")
            ql = os.path.join(root, f'ql_evt{evt}', 'mabc-all-apa.zip')
            pr = os.path.join(root, f'nusel_evt{evt}', 'mabc-pr.zip')
            if not os.path.isfile(pr):
                sys.exit(f"ERROR: missing {pr}")

            n = 0
            with zipfile.ZipFile(ql) as zin:
                for name in zin.namelist():
                    base = os.path.basename(name)
                    if not base.startswith('0-'):
                        continue
                    out.writestr(f'data/{bee_idx}/{bee_idx}-{base[2:]}', zin.read(name))
                    n += 1

                # Parsed here, inside the per-event loop: the remap is per event.
                img = _member(zin, IMG_LAYER) if args.remap else None
                op = _member(zin, OP_LAYER) if args.remap else None

            got = []
            with zipfile.ZipFile(pr) as zin:
                pr_clus = _member(zin, CLUS_LAYER) if args.remap else None
                for name in zin.namelist():
                    base = os.path.basename(name)
                    layer = base[2:].rsplit('.json', 1)[0]
                    if layer not in PR_LAYERS:
                        continue
                    payload = zin.read(name)
                    if args.remap and layer == STM_LAYER:
                        if img is None or op is None or pr_clus is None:
                            print(f"WARNING: evt {evt}: cannot remap ({IMG_LAYER}/{OP_LAYER}"
                                  f"/PR {CLUS_LAYER} missing); keeping PR cluster ids",
                                  file=sys.stderr)
                        else:
                            stm = json.loads(payload)
                            mapping, rows, nambig = build_remap(img, op, pr_clus, stm, evt)
                            payload = json.dumps(remap_stm(stm, mapping)).encode()
                            print(f"  evt {evt}: stm_fit cluster_id -> img-global space "
                                  f"({len(stm['cluster_id'])} pts, "
                                  f"{nambig} ambiguous q fingerprints)")
                            for cid, pick, ranked, unm, note in rows:
                                comp = ', '.join(f'{i}:{c}' for i, c in ranked) or '-'
                                line = (f"    PR clus {cid:>4} -> img {str(pick):>4}"
                                        f"   [{note}]  constituents {{{comp}}}"
                                        f"  unmapped {unm}")
                                print(line)
                                map_lines.append(
                                    f"{evt}\t{bee_idx}\t{cid}\t{pick}\t{note}\t{comp}")
                    out.writestr(f'data/{bee_idx}/{bee_idx}-{base[2:]}', payload)
                    got.append(layer)
                    n += 1
            missing = [l for l in PR_LAYERS if l not in got]
            if missing:
                print(f"WARNING: evt {evt}: {pr} has no {missing} "
                      f"(was the run made with -stm-fit?)", file=sys.stderr)
            print(f"event {evt} -> bee index {bee_idx}: {n} layer(s) {got}")

    print(f"wrote {args.output}")
    if map_lines:
        with open(map_path, 'w') as fh:
            fh.write("# stm_fit-global cluster_id remap for %s\n" % args.output)
            fh.write("# event\tbee_idx\tpr_cluster_id\timg_cluster_id\tnote\t"
                     "img_constituents(id:npts)\n")
            fh.write('\n'.join(map_lines) + '\n')
        print(f"wrote {map_path}")


if __name__ == '__main__':
    main()

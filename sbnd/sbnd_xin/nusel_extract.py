#!/usr/bin/env python3
"""Extract the per-bundle neutrino-selection table for one event.

Inputs (driven by run_nusel_evt.sh):
  --pctree  post-QL point-cloud tree (run_ql_evt.sh -save-pctree): bundle
            structure from the per-cluster scalars QLMatching writes (ident,
            cluster_t0, matched_flash_gid, flag_main/associated_cluster), the
            per-cluster 3d points, and the root opflash PC.
  --prbee   the PR job's mabc-pr.zip: its clustering layer contains exactly the
            clusters that PASS switch_scope's active-volume filter, so its
            cluster_id set is the in-scope set.  scope_filter is runtime state
            and is NOT persisted in the pctree, so this zip is the only way to
            recover it (and it is more robust than the log, whose lines can be
            torn by interleaved writes -- seen on evt286329).
  --prlog   the PR-job log: TaggerCheckTGM / TaggerCheckSTM / TaggerCheckFC
            verdicts.

A row is emitted per QUALIFYING BUNDLE, defined as a cluster that is
  (a) flagged flag_main_cluster  -- QLMatching's matched main, and
  (b) in scope                   -- passes the active-volume filter,
i.e. exactly the population the taggers evaluate (clus_pr require_in_scope).
Main-flagged clusters that fail (b) are non-physical shards switch_scope split
off; they are counted and reported, not emitted.

Flashes are deduplicated across APAs first: SBND reconstructs light per TPC, so
one physical flash yields one flash object per APA (APA1 gids offset +1000000).
Flashes within FLASH_GROUP_NS of each other on opposite sides are one physical
flash and are counted once -- otherwise a bright beam flash double-counts.

Labels: TGM | STM | LM (in-window, light mismatch: the bundle's charge cannot
        explain its matched flash -- QLMatching lm_tagger, doc 34) |
        nu-candidate (in-window, untagged) | not-tagged (out-of-window,
        untagged) | no-bundle (in-window physical flash with no qualifying
        bundle).

The 'lm' column is the raw LM verdict from the cluster scalar "lm_flag" the
Q/L step stamps when run with -lm (0 pass / 1 low-energy / 2 light mismatch;
-1 = scalar absent, i.e. a pre-LM tree or a knob-off run).  Unlike fc it DOES
enter 'label', with priority TGM > STM > LM, and only for in-beam bundles.

The 'fc' column is the fully-contained verdict (TaggerCheckFC -> cluster flag
"FC"), the toolkit counterpart of the prototype's event_type bit 2 / match_isFC.
It is an ORTHOGONAL property, not a cosmic veto: like the prototype (where FC is
an independent eval-tree variable feeding the BDTs) it does NOT enter 'label'.
-1 = no verdict in the log (tagger_check_fc absent from the pipeline).

Usage:
  nusel_extract.py --pctree QL.tar.gz --prbee mabc-pr.zip --prlog PR.log \
      --out row.tsv [--beam-window 0.2,2.2] [--run R --subrun S] [--no-header]
  nusel_extract.py --merge row1.tsv ... --out table.tsv [--events-out ev.tsv]
"""
import argparse
import io
import json
import os
import re
import sys
import tarfile
import zipfile

import numpy as np

US = 1000.0            # WCT time unit is ns
MM = 10.0              # WCT length unit is mm
FLASH_GROUP_NS = 80.0  # cross-APA coincidence window (= MABC flash_group_window)

COLUMNS = ['run', 'subrun', 'event', 'main_id', 'flash_gid', 'flash_apa', 'flash_grp',
           'flash_time_us', 'flash_pe', 'flash_pe_grp', 'in_beam', 'n_bundle',
           'npts_main', 'npts_bundle', 'len_main_cm', 'n_frag', 'tgm', 'stm', 'fc',
           'stmfit',
           'lm', 'label']


def load_pctree(fname):
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    return {md['datapath']: (md, arrays.get(b))
            for b, md in metas.items() if 'datapath' in md}


def parse_pctree(fname):
    """(event, clusters, flashes) from the post-QL tree."""
    bp = load_pctree(fname)
    live = [p for p in bp if re.fullmatch(r'pointtrees/\d+/live', p)]
    if len(live) != 1:
        sys.exit(f'ERROR: {fname}: expected one live tree, got {live}')
    live = live[0]
    event = int(live.split('/')[1])
    md = bp[live][0]
    items = bp[md['pointclouds']][0]['items']
    lpc = bp[md['lpcmaps']][0]['arrays']

    def ds(pcname):
        return bp[items[pcname]][0]['arrays']

    def arr(pcname, aname):
        return bp[ds(pcname)[aname]][1]

    cs = ds('cluster_scalar')
    ident = arr('cluster_scalar', 'ident').astype(int)
    t0 = arr('cluster_scalar', 'cluster_t0')
    gid = arr('cluster_scalar', 'matched_flash_gid').astype(int)
    main = arr('cluster_scalar', 'flag_main_cluster').astype(int)
    assoc = (arr('cluster_scalar', 'flag_associated_cluster').astype(int)
             if 'flag_associated_cluster' in cs else np.zeros_like(ident))
    # LM (light-mismatch) verdict, stamped by QLMatching's lm_tagger knob
    # (doc 34): 0 pass / 1 low-energy / 2 light mismatch.  Absent on pre-LM
    # trees and knob-off runs => every cluster reads -1 (no verdict).
    lm = (arr('cluster_scalar', 'lm_flag').astype(int)
          if 'lm_flag' in cs else np.full_like(ident, -1))

    # Per-cluster point slices: the lpcmaps arrays give, per tree node in
    # serialization order, how many rows that node contributed to each named PC.
    # A cluster node contributes its cluster_scalar row; the 3d rows of the
    # following nodes (its blobs) belong to that cluster.
    map_cs = bp[lpc['cluster_scalar']][1].astype(int)
    map_3d = bp[lpc['3d']][1].astype(int)
    d3 = ds('3d')
    xn = 'x_t0cor' if 'x_t0cor' in d3 else 'x'
    yn = 'y_cor' if 'y_cor' in d3 else 'y'
    zn = 'z_cor' if 'z_cor' in d3 else 'z'
    x, y, z = arr('3d', xn), arr('3d', yn), arr('3d', zn)

    starts, ci, pos = {}, -1, 0
    for n in range(len(map_cs)):
        if map_cs[n]:
            ci += 1
            starts[ci] = [pos, 0]
        if map_3d[n]:
            if ci >= 0:
                starts[ci][1] += int(map_3d[n])
            pos += int(map_3d[n])

    clusters = []
    for i in range(len(ident)):
        s, n = starts.get(i, (0, 0))
        # End-to-end (farthest-pair) length, NOT the bounding-box diagonal: a
        # flash-merged cluster can carry a far-away fragment, which inflates the
        # bbox diagonal into a fictitious "cathode crosser" (evt284349 cluster 11
        # read 447 cm across a 3-point speck; the real track is 202 cm).
        length_cm = 0.0
        if n > 1:
            P = np.c_[x[s:s+n], y[s:s+n], z[s:s+n]]
            a = int(np.argmax(((P - P[0]) ** 2).sum(1)))
            b = int(np.argmax(((P - P[a]) ** 2).sum(1)))
            length_cm = float(np.linalg.norm(P[a] - P[b])) / MM
        clusters.append(dict(ident=int(ident[i]), t0_us=float(t0[i]) / US,
                             gid=int(gid[i]), main=int(main[i]),
                             assoc=int(assoc[i]), lm=int(lm[i]),
                             npts=int(n), len_cm=length_cm))

    fl_gid = arr('opflash', 'gid').astype(int)
    fl_t, fl_pe = arr('opflash', 'time'), arr('opflash', 'pe')
    fl_apa = arr('opflash', 'apa').astype(int)
    flashes = {}
    for g, t, p, a in zip(fl_gid, fl_t, fl_pe, fl_apa):
        g = int(g)
        if g not in flashes:
            flashes[g] = {'t_us': float(t) / US, 'pe': 0.0, 'apa': int(a)}
        flashes[g]['pe'] += float(p)
    return event, clusters, flashes


def group_flashes(flashes):
    """Assign each flash gid a physical-flash group id.

    Single-linkage in time with FLASH_GROUP_NS, restricted to one flash per APA
    per group: the same scintillation flash is reconstructed once per TPC, but
    two same-side flashes that close in time are genuinely distinct.
    """
    order = sorted(flashes, key=lambda g: (flashes[g]['t_us'], g))
    grp, next_grp = {}, 0
    cur, cur_sides = [], set()
    for g in order:
        t = flashes[g]['t_us']
        side = flashes[g]['apa']
        if cur and (t - flashes[cur[-1]]['t_us']) * US <= FLASH_GROUP_NS \
                and side not in cur_sides:
            cur.append(g)
            cur_sides.add(side)
        else:
            next_grp += 1
            cur, cur_sides = [g], {side}
        grp[g] = next_grp
    return grp


# The verdict token is matched STRICTLY as true|false, never as \w+.  Log lines
# can be torn mid-word by interleaved writes (doc 23/25), and a torn "FC=false"
# leaves the fragment "FC=fal": with \w+ that fragment MATCHES and then blows up
# the int conversion (crashed evt285999).  Requiring the whole word means a torn
# line simply fails to match, the verdict stays -1, and the caller's
# "no tagger verdict" warning reports it instead of the run dying.
# The verdict token is captured as its FIRST CHARACTER only, for two reasons:
#
#  1. The taggers spell it differently -- TaggerCheckTGM/FC log a plain bool
#     ("TGM=false") while TaggerCheckSTM logs the flag getter as an int
#     ("STM=0 TGM=0") -- and t/f/1/0 covers every spelling.
#  2. Log lines get torn mid-word by interleaved writes, and the tear is
#     DETERMINISTIC (a write-buffer boundary, not a race): evt285999's
#     "FC=fal[09:33:15.265] D [ clus ] <Mul..." reproduces byte-for-byte on a
#     re-run, so re-running does NOT recover it.  Since the only possible
#     values are true/false/1/0, the leading character disambiguates all four,
#     and a fragment like "fal" is read losslessly.
#
# A tear landing before any value character ("FC=") still fails to match, which
# is the correct outcome: the verdict stays -1 and the caller warns.
VERDICT = r'([tf01])\w*'
RE_TGM = re.compile(r'TaggerCheckTGM: cluster (\d+) \S+ TGM=' + VERDICT + r'\b')
RE_STM = re.compile(r'TaggerCheckSTM: cluster (\d+) \S+ STM=' + VERDICT
                    + r' TGM=' + VERDICT + r'\b')
# NB: matches the message PREFIX only.  Log lines can be torn by interleaved
# writes (doc 23; seen again on evt285185 cluster 5, where another line was
# spliced in mid-word after "...already TGM; ski").  This message carries its
# whole meaning before the tear point -- tgm=1, stm=0 -- so a prefix match
# recovers it losslessly.  The verdict regexes above cannot do the same: their
# information is the trailing =true/=false, which a tear destroys.
RE_SKIP = re.compile(r'TaggerCheckSTM: cluster (\d+) already TGM')
RE_FC = re.compile(r'TaggerCheckFC: cluster (\d+) \S+ FC=' + VERDICT + r'\b')
# "cluster N no STM fit: <reason>" -- the pre-fit exits of check_stm_conditions.
RE_STM_SKIP = re.compile(r'check_stm_conditions: cluster (\d+) no STM fit: (.+)')


def parse_prlog(fname):
    verdicts, as_int = {}, {'t': 1, 'f': 0, '1': 1, '0': 0}
    with open(fname, errors='replace') as f:
        for line in f:
            m = RE_TGM.search(line)
            if m:
                verdicts.setdefault(int(m.group(1)), {})['tgm'] = as_int[m.group(2)]
                continue
            m = RE_STM.search(line)
            if m:
                v = verdicts.setdefault(int(m.group(1)), {})
                v['stm'], v['tgm'] = as_int[m.group(2)], as_int[m.group(3)]
                continue
            m = RE_FC.search(line)
            if m:
                verdicts.setdefault(int(m.group(1)), {})['fc'] = as_int[m.group(2)]
                continue
            m = RE_SKIP.search(line)
            if m:
                v = verdicts.setdefault(int(m.group(1)), {})
                v.setdefault('tgm', 1)
                v['stm'] = 0
                v['stm_skip_tgm'] = 1
    return verdicts


def parse_prtree_flags(fname):
    """{cluster_ident: {'tgm','stm','fc'}} from a POST-PR pctree's cluster_scalar.

    This is the authoritative verdict source and it exists because the taggers
    stamp flag_TGM / flag_STM / flag_FC onto every cluster
    (MultiAlgBlobClustering::normalize_cluster_flags), which the saved tree
    carries.  Preferred over parse_prlog because a log line can be TORN by
    interleaved writes and then simply fails to match: evt287517 cluster 8
    logged "TaggerCheckSTM: c" on one line and "luster 8 -> STM=1 TGM=0"
    seventy lines later, so the regex read no verdict and the bundle was
    written out as not-tagged when the tagger had in fact said STM=1.  The tear
    is deterministic (a write-buffer boundary), so re-running does not fix it,
    and it moves when unrelated logging changes -- turning -stm-fit on was
    enough to create it.  Flags cannot tear.

    Requires run_nusel_evt.sh -save-pr-tree.  Returns {} if the file is absent
    or predates the taggers (no flag_* arrays), and the caller falls back to
    the log.
    """
    if not fname or not os.path.isfile(fname):
        return {}
    try:
        bp = load_pctree(fname)
    except Exception as e:
        print(f'WARNING: cannot read {fname}: {e}', file=sys.stderr)
        return {}
    live = [q for q in bp if re.fullmatch(r'pointtrees/\d+/live', q)]
    if len(live) != 1:
        return {}
    md = bp[live[0]][0]
    items = bp[md['pointclouds']][0]['items']
    if 'cluster_scalar' not in items:
        return {}
    cs = bp[items['cluster_scalar']][0]['arrays']
    # MultiAlgBlobClustering::normalize_cluster_flags only normalizes flags that
    # at least ONE cluster carries, and a tagger sets its flag only on a cluster
    # it tagged.  So an ABSENT flag_X array on a post-tagger tree means "no
    # cluster in this event got X", i.e. verdict 0 everywhere -- not unknown.
    # (evt289145: all_flags=[TGM,...] with no STM and no FC, because nothing was
    # tagged either way; evt287517 has all three.)  If none of the three are
    # present we cannot tell a post-tagger tree from a pre-tagger one, so give
    # up and let the caller fall back to the log.
    present = {'flag_TGM', 'flag_STM', 'flag_FC'} & set(cs)
    if not present:
        return {}

    def a(n):
        return bp[cs[n]][1]

    ident = a('ident').astype(int)
    out = {}
    for i, cid in enumerate(ident):
        v = {}
        for key, arrname in (('tgm', 'flag_TGM'), ('stm', 'flag_STM'), ('fc', 'flag_FC')):
            v[key] = int(a(arrname)[i]) if arrname in cs else 0
        out[int(cid)] = v
    return out


# "cluster N no STM fit: <reason>" -> a short column code for the scan table.
# Order matters: first substring match wins.
STMFIT_CODES = [
    ('fully contained',      'contained'),   # Mid Point A: cluster_fc_check says no FV exit
    ('Mid Point B',          'midkink'),     # 1 exit point + >120 deg mid-track kink
    ('Mid Point C',          'nexits'),      # != 1 candidate exit point
    ('no steiner_pc',        'nosteiner'),
    ('no fiducialutils',     'nofidu'),
    ('round-1 forward fit',  'shortfit'),    # <=3 fit points
    ('no pass recorded',     'postfit'),     # exited after round-1, nothing persisted
]


def stmfit_code(reason):
    for frag, code in STMFIT_CODES:
        if frag in reason:
            return code
    return 'skip'


def parse_stm_skips(fname):
    """{cluster_ident: reason} for clusters the STM tagger exited before fitting.

    Reads the "cluster N no STM fit: <reason>" DEBUG lines from
    TaggerCheckSTM::check_stm_conditions.  Answers "why is there no dQ/dx fit
    for this bundle?" -- previously only knowable from SPDLOG_LOGGER_TRACE
    lines, which are compiled out of this build.
    """
    skips = {}
    if not fname or not os.path.isfile(fname):
        return skips
    with open(fname, errors='replace') as f:
        for line in f:
            m = RE_STM_SKIP.search(line)
            if m:
                # FIRST reason wins: check_stm_conditions logs the specific
                # pre-fit exit, and the visit loop then logs the generic
                # "no pass recorded" catch-all for the same cluster.  Keeping
                # the specific one is what makes the column diagnostic.
                skips.setdefault(int(m.group(1)),
                                 m.group(2).strip().split(' ../')[0])
    return skips


def parse_qlbee(fname):
    """{cluster_id: (len_cm_of_largest_merge_component, n_components)}.

    examine_bundles' flash merge can graft a far-away fragment onto a cluster
    (evt284349 cluster 11 = a 202 cm TPC0 track + a 3-point speck 300 cm away in
    the other TPC).  Any whole-cluster extent -- bounding box OR farthest pair --
    then reports the gap, not the track (447 / 426 cm here).  The Bee clustering
    layer carries per-point real_cluster_id = the pre-merge cluster each blob
    came from, so measure within the dominant component and report how many
    components were merged.  (real_cluster_id is written only on merged clusters,
    so it is NOT uniform per cluster and does not survive into the pctree --
    the zip is the only place it exists.)

    NB: Bee coordinates are already in cm, unlike the pctree arrays which are in
    WCT internal units (mm).  The returned length is in cm.
    """
    with zipfile.ZipFile(fname) as z:
        names = [n for n in z.namelist() if n.endswith('-clustering-global.json')]
        if not names:
            return {}
        d = json.loads(z.read(names[0]))
    if 'real_cluster_id' not in d:
        return {}
    x, y, z_ = (np.array(d[k], float) for k in ('x', 'y', 'z'))
    cid = np.array(d['cluster_id'], int)
    rid = np.array(d['real_cluster_id'], int)
    out = {}
    for c in set(cid.tolist()):
        m = cid == c
        comps = sorted(set(rid[m].tolist()))
        best_n, best_len = -1, 0.0
        for r in comps:
            s_ = m & (rid == r)
            n = int(s_.sum())
            if n <= best_n:
                continue
            P = np.c_[x[s_], y[s_], z_[s_]]
            if len(P) > 1:
                a = int(np.argmax(((P - P[0]) ** 2).sum(1)))
                b = int(np.argmax(((P - P[a]) ** 2).sum(1)))
                best_len = float(np.linalg.norm(P[a] - P[b]))
            else:
                best_len = 0.0
            best_n = n
        out[c] = (best_len, len(comps))
    return out


def parse_prbee(fname):
    """Cluster ids present in the PR Bee clustering layer = the in-scope set."""
    with zipfile.ZipFile(fname) as z:
        names = [n for n in z.namelist() if n.endswith('-clustering-global.json')]
        if not names:
            sys.exit(f'ERROR: {fname}: no clustering layer')
        d = json.loads(z.read(names[0]))
    return set(int(c) for c in d['cluster_id'])


def write_table(fh, header, rows):
    """Write a column-ALIGNED table: each field space-padded to its column width.

    The files are still trivially machine-readable -- no field ever contains a
    space (ids and counts are numeric, labels are hyphenated: nu-candidate,
    no-bundle, not-tagged, cosmic-tagged, no-beam-flash), so any consumer can
    split on runs of whitespace: awk, `sort -k`, str.split(), or pandas with
    sep=r'\s+'.  read_table() below accepts both this layout and the older
    tab-separated one.
    """
    grid = [list(header)] + [[str(v) for v in r] for r in rows]
    width = [max(len(g[c]) for g in grid) for c in range(len(header))]
    for g in grid:
        # rstrip so the last column carries no trailing padding
        print('  '.join(f'{v:<{w}}' for v, w in zip(g, width)).rstrip(), file=fh)


def read_table(fname, first_col):
    """(header, rows) from an aligned OR tab-separated table; header may be absent."""
    with open(fname) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    if not lines:
        return None, []
    header = None
    if lines[0].split() and lines[0].split()[0] == first_col:
        header, lines = lines[0].split(), lines[1:]
    return header, [ln.split() for ln in lines]


def label_of(tgm, stm, in_beam, lm=-1):
    if tgm == 1:
        return 'TGM'
    if stm == 1:
        return 'STM'
    # LM demotes an otherwise-untagged IN-BEAM bundle (owner decision, doc 34):
    # a beam-window bundle whose matched flash its charge cannot explain is a
    # cosmic-flash accidental, not a nu-candidate (evt286021 main 8).  Priority
    # TGM > STM > LM; out-of-beam labels are unchanged by LM.
    if in_beam and lm == 2:
        return 'LM'
    return 'nu-candidate' if in_beam else 'not-tagged'


def one_event(args):
    lo, hi = (float(v) for v in args.beam_window.split(','))
    event, clusters, flashes = parse_pctree(args.pctree)
    verdicts = parse_prlog(args.prlog)
    in_scope = parse_prbee(args.prbee)
    qlgeom = parse_qlbee(args.qlbee) if args.qlbee else {}
    fgrp = group_flashes(flashes)
    pe_grp = {}
    for g, f in flashes.items():
        pe_grp[fgrp[g]] = pe_grp.get(fgrp[g], 0.0) + f['pe']

    # Qualifying bundles: main-flagged AND in scope.
    mains = [c for c in clusters if c['main']]
    bundles = [c for c in mains if c['ident'] in in_scope]
    dropped = [c for c in mains if c['ident'] not in in_scope]
    for c in bundles:
        if c['ident'] not in verdicts:
            print(f'WARNING: evt{event} in-scope main {c["ident"]} has no tagger '
                  f'verdict in {args.prlog}', file=sys.stderr)
    for c in dropped:
        if c['ident'] in verdicts:
            print(f'WARNING: evt{event} out-of-scope main {c["ident"]} WAS evaluated '
                  f'by the taggers -- is require_in_scope on?', file=sys.stderr)

    # Verdicts: prefer the post-PR tree's flags, fall back to the log per key.
    # See parse_prtree_flags for why the log alone is not trustworthy.
    flagv = parse_prtree_flags(getattr(args, 'prtree', None))
    if flagv:
        # Only report on clusters the log DID mention: the log never logs a
        # cluster the taggers skipped, so "log has no line" is normal and not
        # worth a note.  A genuine contradiction (log says 0, flag says 1, or
        # vice versa) means a torn line and is worth seeing.
        bad = [(cid, k, verdicts[cid][k], fv[k])
               for cid, fv in flagv.items() if cid in verdicts
               for k in ('tgm', 'stm', 'fc')
               if k in verdicts[cid] and k in fv and verdicts[cid][k] != fv[k]]
        if bad:
            print(f'NOTE: evt{event}: {len(bad)} log verdict(s) CONTRADICTED by '
                  f'the post-PR tree flags (torn log line); flags win: '
                  + ', '.join(f'cluster {c} {k} log={l} flag={f}'
                              for c, k, l, f in bad), file=sys.stderr)

    skips = parse_stm_skips(getattr(args, 'prlog', None))

    rows = []
    for c in sorted(bundles, key=lambda c: c['t0_us']):
        peers = [o for o in clusters if o['gid'] == c['gid'] and o is not c]
        v = dict(verdicts.get(c['ident'], {}))
        v.update({k: val for k, val in flagv.get(c['ident'], {}).items()})
        tgm, stm, fc = v.get('tgm', -1), v.get('stm', -1), v.get('fc', -1)
        # stmfit: did this bundle's main cluster reach the STM trajectory fit,
        # and if not, which pre-fit exit stopped it.  'fit' means a dQ/dx fit
        # exists (the event display's STM panel will have curves); anything else
        # names the reason there is none.
        if c['ident'] in skips:
            stmfit = stmfit_code(skips[c['ident']])
        elif v.get('stm_skip_tgm'):
            stmfit = 'tgm'          # already TGM, STM never ran
        elif stm >= 0:
            # Evaluated with no logged exit.  Called 'eval' rather than 'fit'
            # because only the dQ/dx dump proves a trajectory was persisted --
            # the event display upgrades this to 'fit' when tracking-stm.root
            # actually has points for the cluster.
            stmfit = 'eval'
        else:
            stmfit = '-'            # never evaluated
        in_beam = int(lo <= c['t0_us'] < hi)
        fl = flashes.get(c['gid'])
        g = fgrp.get(c['gid'], -1)
        glen, nfrag = qlgeom.get(c['ident'], (c['len_cm'], 1))   # both in cm
        rows.append([args.run, args.subrun, event, c['ident'], c['gid'],
                     fl['apa'] if fl else -1, g,
                     f'{c["t0_us"]:.3f}', f'{fl["pe"]:.1f}' if fl else '-1',
                     f'{pe_grp.get(g, 0.0):.1f}', in_beam, 1 + len(peers),
                     c['npts'], c['npts'] + sum(o['npts'] for o in peers),
                     f'{glen:.1f}', nfrag, tgm, stm, fc, stmfit, c['lm'],
                     label_of(tgm, stm, in_beam, c['lm'])])

    # In-window PHYSICAL flashes (deduped) with no qualifying bundle.
    bundle_grps = {fgrp.get(c['gid'], -1) for c in bundles
                   if lo <= c['t0_us'] < hi}
    seen = set()
    for g in sorted(flashes, key=lambda g: flashes[g]['t_us']):
        f = flashes[g]
        gr = fgrp[g]
        if not (lo <= f['t_us'] < hi) or gr in bundle_grps or gr in seen:
            continue
        seen.add(gr)
        rows.append([args.run, args.subrun, event, -1, g, f['apa'], gr,
                     f'{f["t_us"]:.3f}', f'{f["pe"]:.1f}', f'{pe_grp[gr]:.1f}',
                     1, 0, 0, 0, '0.0', 0, -1, -1, -1, '-', -1, 'no-bundle'])

    with open(args.out, 'w') if args.out else sys.stdout as fh:
        if args.no_header:
            # Headerless output is meant for concatenation; column widths would
            # be per-file and would not line up, so keep it tab-separated.
            for r in rows:
                print('\t'.join(str(v) for v in r), file=fh)
        else:
            write_table(fh, COLUMNS, rows)
    if dropped:
        print(f'[evt {event}] {len(dropped)} main-flagged cluster(s) dropped as '
              f'out-of-scope (idents {sorted(c["ident"] for c in dropped)})',
              file=sys.stderr)


def merge(args):
    rows, header = [], None
    for fname in args.merge:
        hdr, rs = read_table(fname, COLUMNS[0])
        if hdr:
            header = hdr
        rows += rs

    i = {c: n for n, c in enumerate(COLUMNS)}
    rows.sort(key=lambda r: (int(r[i['run']]), int(r[i['subrun']]),
                             int(r[i['event']]), float(r[i['flash_time_us']])))
    with open(args.out, 'w') if args.out else sys.stdout as fh:
        write_table(fh, header or COLUMNS, rows)

    if not args.events_out:
        return
    ev = {}
    for r in rows:
        key = (r[i['run']], r[i['subrun']], r[i['event']])
        e = ev.setdefault(key, {'bundles': 0, 'inbeam_bundles': [],
                                'inbeam_grps': set()})
        is_bundle = r[i['main_id']] != '-1'
        if is_bundle:
            e['bundles'] += 1
        if r[i['in_beam']] == '1':
            e['inbeam_grps'].add(r[i['flash_grp']])
            if is_bundle:
                e['inbeam_bundles'].append(r[i['label']])
    ev_cols = ['run', 'subrun', 'event', 'n_bundles', 'n_inbeam_flash',
               'n_inbeam_bundle', 'event_label']
    ev_rows = []
    for key in sorted(ev, key=lambda k: tuple(int(v) for v in k)):
        e = ev[key]
        labels = e['inbeam_bundles']
        if any(l not in ('TGM', 'STM', 'LM') for l in labels):
            elabel = 'nu-candidate'      # an in-window bundle survived all taggers
        elif labels:
            elabel = 'cosmic-tagged'     # every in-window bundle is TGM/STM/LM
        elif e['inbeam_grps']:
            elabel = 'no-bundle'
        else:
            elabel = 'no-beam-flash'
        ev_rows.append([*key, e['bundles'], len(e['inbeam_grps']),
                        len(labels), elabel])
    with open(args.events_out, 'w') as fh:
        write_table(fh, ev_cols, ev_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pctree', help='post-QL pctree tarball (bundle structure)')
    ap.add_argument('--prbee', help="PR job's mabc-pr.zip (in-scope cluster set)")
    ap.add_argument('--qlbee', help="QL job's mabc-all-apa.zip (per-merge-component "
                                    "geometry via real_cluster_id)")
    ap.add_argument('--prlog', help='PR-job log (tagger verdict FALLBACK + '
                                    'STM no-fit reasons)')
    ap.add_argument('--prtree', help="post-PR pctree tarball (authoritative "
                                     "tagger verdicts from flag_TGM/STM/FC; "
                                     "needs run_nusel_evt.sh -save-pr-tree)")
    ap.add_argument('--beam-window', default='0.2,2.2',
                    help='low,high in us on cluster_t0 (default 0.2,2.2)')
    ap.add_argument('--run', type=int, default=0)
    ap.add_argument('--subrun', type=int, default=0)
    ap.add_argument('--out', help='output TSV (default stdout)')
    ap.add_argument('--no-header', action='store_true')
    ap.add_argument('--merge', nargs='+', help='merge per-event TSVs instead')
    ap.add_argument('--events-out', help='with --merge: per-event summary TSV')
    args = ap.parse_args()

    if args.merge:
        merge(args)
    elif args.pctree and args.prbee and args.prlog:
        one_event(args)
    else:
        ap.error('need --pctree + --prbee + --prlog, or --merge')


if __name__ == '__main__':
    main()

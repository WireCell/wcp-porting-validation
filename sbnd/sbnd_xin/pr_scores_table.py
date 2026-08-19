#!/usr/bin/env python3
"""doc pr/11: per-event score + runtime + memory table for the full PR chain.

Reads what run_pr_chain_batch.sh writes per event under <out_root>/pr_evt<ID>/:
  nusel-evt<ID>.tsv    label / TGM / STM / FC / LM / bundle info (nusel_extract.py,
                       UNMODIFIED -- this script does not duplicate its logic)
  tracking-pr.root     T_tagger / T_kine / Trun (UbooneTaggerOutputVisitor) --
                       numu_score / nue_score / cosmic-side quantities / kine
  wct_pr_evt<ID>.log   "Timer: Total W wall-sec, C core-sec" (sub-second,
                       always emitted, no perf flag needed) + the
                       "DL vertex failed" WARN if the SCN import/inference hit
                       an error this event
  .time.meta           timecmd.py: rc / wall_s (integer -- kept only as a
                       cross-check) / maxrss_kb (RUSAGE_CHILDREN peak)

One row per event, keyed on (run, subrun, event) -- event ids collide across
the samples in doc pr/11 (e.g. evt 12 is both a round1-qlmatch and a
round2-patrec MC event), so run/subrun disambiguate.

IMPORTANT -- cosmict_score: NeutrinoTaggerInfo.h / UbooneNumuBDTScorer.cxx
never write it (only the struct default and the T_tagger branch exist); it is
reported here as-is (always 0) rather than papered over.  The live cosmic-side
quantities are cosmic_flag / cosmict_flag / cosmict_10_score.

IMPORTANT -- the T_tagger row is NOT joinable to a nusel-evt<ID>.tsv main_id.
unmerge_bundle/unmerge_assoc (production knobs, both ON here) run BEFORE
tagger_check_neutrino and can split a pre-PR bundle into new, PR-job-internal
cluster idents; TaggerCheckNeutrino picks its main_cluster from THOSE idents,
which do not correspond to nusel_extract's (pre-PR) main_id numbering. Seen
directly on evt 285201 (work-mcp1kall-d59k): nusel-evt285201.tsv lists exactly
one in-window bundle (main_id 2, 10.0 cm, STM); the log's
"selected main cluster 1 (t0 1.444 us, L 2.5 cm, 0 associated)" is a
DIFFERENT, unmerge-split 2.5 cm cluster with no row in nusel-evt285201.tsv at
all.  So this script does NOT try to match a score row to a specific nusel
bundle row.  Instead:
  - the EVENT's cosmic/nu verdict comes from nusel-evt<ID>.tsv's rows via the
    same priority rule nusel_extract.py's merge() uses for its per-event
    summary (any in-beam bundle not TGM/STM/LM => nu-candidate; elif any
    in-beam bundle => cosmic-tagged; elif any in-beam flash => no-bundle;
    else no-beam-flash) -- reproduced here as event_label, not re-derived from
    a single row.
  - whether the SCORES are a real evaluation comes directly from the PR log:
    "TaggerCheckNeutrino: selected main cluster ..." is emitted iff
    main_cluster was non-null (TaggerCheckNeutrino.cxx:345-346), i.e. iff
    TaggerInfo/KineInfo actually got filled with real features. Scores are
    reported only when that line is present (nu_evaluated=1); otherwise the
    T_tagger row (if tracking-pr.root even exists) holds only NeutrinoTaggerInfo
    struct DEFAULTS (cosmic_flag=1, nue_score=-15 "not filled" sentinel, ...)
    run through the BDTs -- meaningless, and blanked here rather than reported.

Usage:
  pr_scores_table.py --root OUTROOT [--sample NAME] --out rows.tsv
  pr_scores_table.py --merge rows1.tsv ... --out table.tsv [--summary]
"""
import argparse
import glob
import os
import re
import sys

COLUMNS = ['sample', 'run', 'subrun', 'event', 'rc', 'wall_s', 'core_s', 'timecmd_wall_s',
           'maxrss_kb', 'dl_warn', 'event_label', 'n_bundle', 'n_inbeam_bundle',
           'n_inbeam_flash', 'nu_evaluated', 'n_cosmic_skipped', 'nu_sel_t0_us',
           'nu_sel_len_cm', 'nu_sel_n_assoc', 'nu_x_cm', 'nu_y_cm', 'nu_z_cm',
           'numu_score', 'nue_score', 'cosmic_flag', 'cosmict_flag', 'cosmict_10_score',
           'cosmict_score', 'kine_reco_Enu_MeV']

RE_TIMER = re.compile(r'Timer: Total ([0-9.]+) wall-sec, ([0-9.]+) core-sec')
RE_SELECTED = re.compile(
    r'TaggerCheckNeutrino: selected main cluster \S+ \(t0 ([-0-9.]+) us, '
    r'L ([-0-9.]+) cm, (\d+) associated\)')
RE_NOMAIN = re.compile(r'TaggerCheckNeutrino: no main cluster selected')
RE_COSMIC_SKIP = re.compile(r'TaggerCheckNeutrino: in-window cluster .* cosmic-tagged .*; '
                             r'skipping \(nu_skip_cosmic\)')



# doc pr/94 Phase 5: T_tagger/T_kine hold ONE ROW PER IN-BEAM-WINDOW BUNDLE when
# the nu_per_bundle knob is on, so a hard [0] silently reports whichever bundle
# was enumerated first.  primary_index() reproduces the legacy meaning of "the
# candidate" (longest selected main activity) and falls back to 0 for pre-pr/94
# and knob-off files.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from pr94_rows import primary_index, n_rows  # noqa: E402

def read_time_meta(path):
    d = {}
    if os.path.isfile(path):
        for line in open(path):
            if '=' in line:
                k, v = line.strip().split('=', 1)
                d[k] = v
    return d


def read_timer(logpath):
    """Whole-job Timer total, the DL-vertex hard-failure WARN, and the three
    TaggerCheckNeutrino selection signals (see module docstring): whether it
    filled TaggerInfo for real (and on what candidate), and how many in-window
    candidates it skipped as cosmic-tagged."""
    wall = core = None
    dl_warn = 0
    nu_evaluated = 0
    nu_t0 = nu_len = nu_nassoc = ''
    n_skip = 0
    if not os.path.isfile(logpath):
        return wall, core, dl_warn, nu_evaluated, nu_t0, nu_len, nu_nassoc, n_skip
    with open(logpath, errors='replace') as f:
        for line in f:
            m = RE_TIMER.search(line)
            if m:
                wall, core = float(m.group(1)), float(m.group(2))
            if 'DL vertex failed' in line:
                dl_warn = 1
            m = RE_SELECTED.search(line)
            if m:
                nu_evaluated = 1
                nu_t0, nu_len, nu_nassoc = m.group(1), m.group(2), m.group(3)
            if RE_COSMIC_SKIP.search(line):
                n_skip += 1
    return wall, core, dl_warn, nu_evaluated, nu_t0, nu_len, nu_nassoc, n_skip


# nusel_extract.py's default (--out, no --no-header) output is a
# space-PADDED table for human readability, not raw tabs (write_table());
# only its --no-header form is tab-separated ("meant for concatenation").
# Whitespace-split (like nusel_extract.py's own read_table()) parses both.
NUSEL_COLUMNS = ['run', 'subrun', 'event', 'main_id', 'flash_gid', 'flash_apa',
                  'flash_grp', 'flash_time_us', 'flash_pe', 'flash_pe_grp', 'in_beam',
                  'n_bundle', 'npts_main', 'npts_bundle', 'len_main_cm', 'n_frag',
                  'tgm', 'stm', 'fc', 'stmfit', 'lm', 'label']
COSMIC_LABELS = ('TGM', 'STM', 'LM')


def read_nusel_rows(tsv_path):
    if not os.path.isfile(tsv_path):
        return []
    with open(tsv_path) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    if not lines:
        return []
    if lines[0].split() and lines[0].split()[0] == NUSEL_COLUMNS[0]:
        lines = lines[1:]
    return [dict(zip(NUSEL_COLUMNS, ln.split())) for ln in lines]


def event_summary(rows):
    """Reproduce nusel_extract.py's merge()/--events-out per-event summary
    (run/subrun/event, n_bundles, n_inbeam_flash, n_inbeam_bundle, event_label)
    for a SINGLE event's rows, so run_pr_chain_batch.sh does not need a second
    nusel_extract.py invocation per event.  Same priority rule, same source."""
    n_bundle = sum(1 for r in rows if r.get('main_id') != '-1')
    inbeam_grps = {r['flash_grp'] for r in rows if r.get('in_beam') == '1'}
    inbeam_labels = [r['label'] for r in rows
                      if r.get('in_beam') == '1' and r.get('main_id') != '-1']
    if any(l not in COSMIC_LABELS for l in inbeam_labels):
        label = 'nu-candidate'
    elif inbeam_labels:
        label = 'cosmic-tagged'
    elif inbeam_grps:
        label = 'no-bundle'
    else:
        label = 'no-beam-flash'
    run = rows[0]['run'] if rows else ''
    subrun = rows[0]['subrun'] if rows else ''
    return {'run': run, 'subrun': subrun, 'label': label, 'n_bundle': n_bundle,
            'n_inbeam_bundle': len(inbeam_labels), 'n_inbeam_flash': len(inbeam_grps)}


RE_RSE = re.compile(r'rse=\((\d+), (\d+), \d+\)')


def read_rse_fallback(batch_log_path):
    """run/subrun from run_pr_chain_batch.sh's own '[evt N] rse=(R, S, N) ...'
    line (its out_root/.batch_pr_evt<N>.log), for events whose nusel-evt<N>.tsv
    was never written (e.g. a crash before nusel_extract.py could run) -- so
    the failure census can still name real (run, subrun, event)."""
    if not os.path.isfile(batch_log_path):
        return '', ''
    with open(batch_log_path, errors='replace') as f:
        for line in f:
            m = RE_RSE.search(line)
            if m:
                return m.group(1), m.group(2)
    return '', ''


def read_tagger_root(root_path):
    out = {}
    if not os.path.isfile(root_path):
        return out
    try:
        import uproot
    except ImportError:
        sys.exit('ERROR: uproot not importable')
    try:
        f = uproot.open(root_path)
    except Exception as e:
        print(f'WARN: cannot open {root_path}: {e}', file=sys.stderr)
        return out
    row = 0
    if 'T_tagger' in f:
        t = f['T_tagger']
        row = primary_index(t)
        out['nu_row'] = row
        out['n_nu_rows'] = n_rows(t)
        for k in ('numu_score', 'nue_score', 'cosmic_flag', 'cosmict_flag',
                  'cosmict_10_score', 'cosmict_score', 'nu_x', 'nu_y', 'nu_z'):
            if k in t:
                arr = t[k].array()
                if len(arr) > row:
                    out[k] = arr[row]
    if 'T_kine' in f:
        ki = f['T_kine']
        if 'kine_reco_Enu' in ki:
            arr = ki['kine_reco_Enu'].array()
            if len(arr) > row:
                out['kine_reco_Enu'] = arr[row]
    return out


def one_root(args):
    root = os.path.abspath(args.root)
    sample = args.sample or os.path.basename(root)
    evtdirs = sorted(glob.glob(os.path.join(root, 'pr_evt*')),
                      key=lambda p: int(re.search(r'pr_evt(\d+)$', p).group(1)))
    out = open(args.out, 'w') if args.out else sys.stdout
    if not args.no_header:
        out.write('\t'.join(COLUMNS) + '\n')
    for d in evtdirs:
        evt = int(re.search(r'pr_evt(\d+)$', d).group(1))
        meta = read_time_meta(os.path.join(d, '.time.meta'))
        logpath = os.path.join(d, f'wct_pr_evt{evt}.log')
        (wall, core, dl_warn, nu_eval, nu_t0, nu_len, nu_nassoc,
         n_skip) = read_timer(logpath)
        nrows = read_nusel_rows(os.path.join(d, f'nusel-evt{evt}.tsv'))
        es = event_summary(nrows)
        if not nrows and meta.get('rc', '0') != '0':
            # nusel_extract.py could not run (e.g. the wire-cell job crashed
            # before writing mabc-pr.zip's clustering layer) -- do not let the
            # empty-rows default ('no-beam-flash') masquerade as a real verdict,
            # and still recover run/subrun so the failure census names the event.
            es['label'] = 'crashed/no-extract'
            es['run'], es['subrun'] = read_rse_fallback(
                os.path.join(root, f'.batch_pr_evt{evt}.log'))
        # Scores are only a real evaluation when the log's own selection line
        # fired (see module docstring) -- gate on nu_eval, not on any nusel
        # main_id (which is not joinable to the PR job's post-unmerge idents).
        sc = read_tagger_root(os.path.join(d, 'tracking-pr.root')) if nu_eval else {}
        row = {
            'sample': sample,
            'run': es.get('run', ''), 'subrun': es.get('subrun', ''), 'event': evt,
            'rc': meta.get('rc', ''), 'wall_s': f'{wall:.3f}' if wall is not None else '',
            'core_s': f'{core:.3f}' if core is not None else '',
            'timecmd_wall_s': meta.get('wall_s', ''),
            'maxrss_kb': meta.get('maxrss_kb', ''), 'dl_warn': dl_warn,
            'event_label': es.get('label', ''), 'n_bundle': es.get('n_bundle', ''),
            'n_inbeam_bundle': es.get('n_inbeam_bundle', ''),
            'n_inbeam_flash': es.get('n_inbeam_flash', ''),
            'nu_evaluated': nu_eval, 'n_cosmic_skipped': n_skip,
            'nu_sel_t0_us': nu_t0, 'nu_sel_len_cm': nu_len, 'nu_sel_n_assoc': nu_nassoc,
            'nu_x_cm': sc.get('nu_x', ''), 'nu_y_cm': sc.get('nu_y', ''),
            'nu_z_cm': sc.get('nu_z', ''),
            'numu_score': sc.get('numu_score', ''), 'nue_score': sc.get('nue_score', ''),
            'cosmic_flag': sc.get('cosmic_flag', ''), 'cosmict_flag': sc.get('cosmict_flag', ''),
            'cosmict_10_score': sc.get('cosmict_10_score', ''),
            'cosmict_score': sc.get('cosmict_score', ''),
            'kine_reco_Enu_MeV': sc.get('kine_reco_Enu', ''),
        }
        out.write('\t'.join(str(row[c]) for c in COLUMNS) + '\n')
    if args.out:
        out.close()


def merge(args):
    rows = []
    for fn in args.merge:
        with open(fn) as f:
            lines = [ln.rstrip('\n') for ln in f if ln.strip()]
        # Tolerate header-or-not, same as nusel_extract.py's read_table():
        # --no-header inputs must not silently lose their first data row.
        if lines and lines[0].split('\t')[0] == COLUMNS[0]:
            lines = lines[1:]
        rows.extend(lines)
    out = open(args.out, 'w') if args.out else sys.stdout
    out.write('\t'.join(COLUMNS) + '\n')
    for r in rows:
        out.write(r + '\n')
    if args.out:
        out.close()
    if args.summary:
        print_summary(rows)


def print_summary(rows):
    from collections import Counter
    n = len(rows)
    rc_ctr = Counter()
    label_ctr = Counter()
    scored = 0
    dl_warn_ctr = 0
    for r in rows:
        v = dict(zip(COLUMNS, r.split('\t')))
        rc_ctr[v.get('rc', '?')] += 1
        label_ctr[v.get('event_label', '?')] += 1
        if v.get('nu_evaluated') == '1':
            scored += 1
        if v.get('dl_warn') == '1':
            dl_warn_ctr += 1
    print(f'\n===== summary: {n} events =====', file=sys.stderr)
    print('rc:', dict(rc_ctr), file=sys.stderr)
    print('event_label:', dict(label_ctr), file=sys.stderr)
    print(f'nu_evaluated=1 (real TaggerInfo, scores meaningful): {scored}/{n}', file=sys.stderr)
    print(f'DL vertex failed WARN: {dl_warn_ctr}/{n}', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', help='out_root written by run_pr_chain_batch.sh')
    ap.add_argument('--sample', help='sample name column (default: basename of --root)')
    ap.add_argument('--out', help='output TSV (default stdout)')
    ap.add_argument('--no-header', action='store_true')
    ap.add_argument('--merge', nargs='+', help='merge per-sample TSVs instead')
    ap.add_argument('--summary', action='store_true', help='with --merge: print a census to stderr')
    args = ap.parse_args()
    if args.merge:
        merge(args)
    elif args.root:
        one_root(args)
    else:
        ap.error('need --root, or --merge')


if __name__ == '__main__':
    main()

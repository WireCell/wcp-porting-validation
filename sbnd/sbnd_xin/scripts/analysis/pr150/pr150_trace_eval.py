#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 3.2 -- run the doc pdvd/113-115 STM-fit / Steiner-seed instruments on SBND arms.

The pdvd scripts (pdvd/docs/nf_sp_img_clus/scripts/d11{3,4,5}_*.py, d111s_common.py, d111_stage_attrib.py) stay
UNTOUCHED.  They expect <IMG>/<det>/work/<pre>_<arm>/{tracking-stm.root,mabc-pr.zip,wct_pr_*.log} and a trace log
<logd>/evt_<pre>.log.gz, and read the plane bases from d111_stage_attrib.BASE[det].  This wrapper
  1. builds a shim layout /home/xqian/tmp/pr150/tr/sbnd/work/<sample>-<event>_<arm>/ of symlinks into
     work-<sample>-<arm>/pr_evt<event>/ (only events that have tracking-stm.root), and gzips each event's stdout.log
     (where the WCT_STEINER_GRAPH_DUMP / WCT_STM_PATH_DEBUG lines land) into <logd>/evt_<pre>.log.gz;
  2. imports the pdvd modules, sets d111s_common.IMG to the shim root and BASE['sbnd'] = (0, 3968, 7936) (the
     calib-dump meta 'base'; d42_proj2d_resid.py:40), and calls the instrument's main() with the given arguments.
Instruments: support (d114_support_census), resid (d115_proj_resid), dqdx (d115_dqdx_compare with
--ref nusel_display/stm_ref_dqdx.json), seed (d113_steiner_census).
Usage:
  pr150_trace_eval.py shim --arm pr150ts0 --samples mcp1k mcp2k
  pr150_trace_eval.py support --arm pr150ts0 --out /home/xqian/tmp/pr150/trace/support_ts0 [--jobs 8]
  pr150_trace_eval.py resid   --base pr150ts0 --arm pr150tcsp3bw --out ... [--jobs 8]
  pr150_trace_eval.py dqdx    --base pr150ts0 --arm pr150tcsp3bw --out ...
  pr150_trace_eval.py seed    --arm pr150ts0 --out ...
  pr150_trace_eval.py fig     --event mcp2k-68748 --cluster 11 --arms pr150ts0:s0,pr150tcsp3bw:csp3bw --out <png stem>
"""
import argparse, glob, gzip, os, shutil, sys
import uproot
SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
PDVD = '/home/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts'
ROOT = '/home/xqian/tmp/pr150/tr'
REF = f'{SX}/nusel_display/stm_ref_dqdx.json'
sys.path.insert(0, PDVD)


def shim(a):
    n = skipped = 0
    for s in a.samples:
        for d in sorted(glob.glob(f'{SX}/work-{s}-{a.arm}/pr_evt*')):
            ev = os.path.basename(d)[6:]
            if not os.path.exists(f'{d}/tracking-stm.root'):
                continue
            try:                                   # SBND writes the file even with no STM candidate: then no
                f = uproot.open(f'{d}/tracking-stm.root')   # T_proj_data / T_rec_charge rows, which the pdvd
                if 'T_proj_data' not in f or f['T_rec_charge'].num_entries == 0:   # instruments assume present
                    skipped += 1; continue
            except Exception:  # noqa: BLE001
                skipped += 1; continue
            pre = f'{s}-{ev}'
            t = f'{ROOT}/sbnd/work/{pre}_{a.arm}'
            os.makedirs(t, exist_ok=True)
            for f in ('tracking-stm.root', 'tracking-pr.root', 'mabc-pr.zip', f'calib-pr-evt{ev}.json', f'wct_pr_evt{ev}.log'):
                if os.path.exists(f'{d}/{f}') and not os.path.lexists(f'{t}/{f}'):
                    os.symlink(f'{d}/{f}', f'{t}/{f}')
            lg = f'{ROOT}/logs_{a.arm}/evt_{pre}.log.gz'
            os.makedirs(os.path.dirname(lg), exist_ok=True)
            if not os.path.exists(lg) and os.path.exists(f'{d}/stdout.log'):
                with open(f'{d}/stdout.log', 'rb') as fi, gzip.open(lg, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
            n += 1
    print(f'shim {a.arm}: {n} events with an STM fit under {ROOT}/sbnd/work ({skipped} with tracking-stm.root but no STM candidate skipped), logs under {ROOT}/logs_{a.arm}')


def patched():
    import d111s_common as C
    import d111_stage_attrib as A
    C.IMG = ROOT
    A.BASE['sbnd'] = (0, 3968, 7936)
    return C, A


def run(mod_name, argv):
    C, A = patched()
    import importlib
    M = importlib.import_module(mod_name)
    if hasattr(M, 'IMG'):
        M.IMG = ROOT
    if hasattr(M, 'REF') and isinstance(M.REF, dict):
        M.REF['sbnd'] = REF
    if hasattr(M, 'BASE') and isinstance(M.BASE, dict):
        M.BASE['sbnd'] = (0, 3968, 7936)
    sys.argv = [mod_name] + argv
    return M.main()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('shim'); p.add_argument('--arm', required=True); p.add_argument('--samples', nargs='+', required=True)
    for c in ('support', 'seed'):
        q = sub.add_parser(c); q.add_argument('--arm', required=True); q.add_argument('--out', required=True); q.add_argument('--jobs', type=int, default=8); q.add_argument('--events')
    q = sub.add_parser('fig'); q.add_argument('--event', required=True); q.add_argument('--cluster', type=int, required=True); q.add_argument('--arms', required=True)
    q.add_argument('--out', required=True); q.add_argument('--xr', type=float, nargs=2)
    for c in ('resid', 'dqdx'):
        q = sub.add_parser(c); q.add_argument('--base', required=True); q.add_argument('--arm', required=True); q.add_argument('--out', required=True); q.add_argument('--jobs', type=int, default=8)
        q.add_argument('--events'); q.add_argument('--status', default='0')
    a = ap.parse_args()
    if a.cmd == 'shim':
        shim(a)
    elif a.cmd == 'support':
        run('d114_support_census', ['--det', 'sbnd', '--arm', a.arm, '--logd', f'{ROOT}/logs_{a.arm}', '--jobs', str(a.jobs), '--out', a.out] + (['--events', a.events] if a.events else []))
    elif a.cmd == 'seed':
        run('d113_steiner_census', ['--det', 'sbnd', '--arm', a.arm, '--logd', f'{ROOT}/logs_{a.arm}', '--jobs', str(a.jobs), '--out', a.out])
    elif a.cmd == 'resid':
        run('d115_proj_resid', ['--det', 'sbnd', '--base', a.base, '--arm', a.arm, '--logd-base', f'{ROOT}/logs_{a.base}', '--logd-arm', f'{ROOT}/logs_{a.arm}',
                                '--jobs', str(a.jobs), '--out', a.out] + (['--events', a.events] if a.events else []))
    elif a.cmd == 'fig':
        run('d115r2_cluster_fig', ['--det', 'sbnd', '--event', a.event, '--cluster', str(a.cluster), '--arms', a.arms, '--out', a.out] + (['--xr', str(a.xr[0]), str(a.xr[1])] if a.xr else []))
    elif a.cmd == 'dqdx':
        run('d115_dqdx_compare', ['--det', 'sbnd', '--base', a.base, '--arm', a.arm, '--out', a.out, '--ref', REF, '--status', a.status])

#!/usr/bin/env python3
"""doc 77 round 2 -- per-TLA plumbing gate for the SBND PR job.

The production compile only exercises the TLAs that are ON; a knob left null or
false emits no key, so a broken forward for it would pass every output gate
silently.  This compiles the PR job against TWO cfg trees with ONE TLA at a
time set to a distinctive non-default value and requires the two results to be
identical -- including identical failure text where the probe value is illegal.

Usage: tla_probe_gate.py <cfgrootA> <cfgrootB> [--jobs N]
Exit 0 = every probe agrees.
"""
import re, sys, subprocess, argparse
from concurrent.futures import ThreadPoolExecutor

WCS = '/nfs/data/1/xqian/toolkit-dev/local/bin/wcsonnet'
DATA = '/nfs/data/1/xqian/toolkit-dev/wire-cell-data'
JOBS = {'pr': 'pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet',
        'ql': 'pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet'}
JOB = JOBS['pr']
PIPELINE = ("['switch_scope','unmerge_bundle','unmerge_assoc','steiner','fiducialutils',"
            "'tagger_check_tgm','tagger_check_stm','tagger_check_fc','protect_bundle',"
            "'steiner_refresh','tagger_check_neutrino','numu_bdt_scorer','nue_bdt_scorer',"
            "'tracking_visitor','tagger_output']")
# what the base invocation already supplies (probing these would duplicate a flag)
BASE_KEYS = {'input', 'output_dir', 'run', 'subrun', 'event', 'reality',
             'pipeline_names', 'save_tensors', 'dl_weights'}
QLBASE = ['-A', 'input=in.tar.gz', '-A', 'output_dir=out',
          '-S', 'run=18253', '-S', 'subrun=1', '-S', 'event=172230',
          '-A', 'reality=data', '-S', 'anode_indices=[0,1]']
QLBASE_KEYS = {'input', 'output_dir', 'run', 'subrun', 'event', 'reality', 'anode_indices'}
BASE = ['-A', 'input=in.tar.gz', '-A', 'output_dir=out',
        '-S', 'run=18253', '-S', 'subrun=1', '-S', 'event=172230',
        '-A', 'reality=data', '-S', 'pipeline_names=' + PIPELINE,
        '-A', 'save_tensors=out.tar.gz', '-A', 'dl_weights=']
LIST_PROBE = {'anode_indices': '[0]', 'muon_dqdx_curve': '[0.11,0.22,3,0.44]',
              'beam_window_us': '[0.3,2.3]'}


def tlas(cfgroot):
    """(name, default-literal) for every TLA of the PR job, in signature order."""
    src = open(f'{cfgroot}/{JOB}').read().split('\n')
    beg = next(i for i, l in enumerate(src) if l.startswith('function('))
    end = next(i for i, l in enumerate(src) if i > beg and l.startswith(')'))
    out = []
    for l in src[beg:end]:
        m = re.match(r'^    ([A-Za-z_]\w*)\s*=\s*(.+?),\s*(//.*)?$', l)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def probe_arg(name, default):
    """A distinctive, type-appropriate override for one TLA."""
    if name in LIST_PROBE:
        return ['-S', f'{name}={LIST_PROBE[name]}']
    if default in ('true', 'false'):
        return ['-S', f'{name}={"false" if default == "true" else "true"}']
    if default.startswith(("'", '"')):
        return ['-A', f'{name}=zzprobe']
    return ['-S', f'{name}=4243']          # null, numeric, and `N * wc.cm`


def compile_one(cfgroot, extra):
    env = {'WIRECELL_PATH': f'{cfgroot}:{DATA}:{DATA}/sbnd/photodet',
           'PATH': '/usr/bin:/bin', 'HOME': '/home/xqian'}
    r = subprocess.run([WCS] + BASE + extra + [f'{cfgroot}/{JOB}'],
                       capture_output=True, env=env)
    # strip the cfg root so the two trees' paths cannot themselves differ
    norm = lambda b: b.decode('utf-8', 'replace').replace(cfgroot, '@CFG@')
    return r.returncode, norm(r.stdout), norm(r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfgA'); ap.add_argument('cfgB')
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--job', choices=('pr', 'ql'), default='pr')
    a = ap.parse_args()
    global JOB, BASE, BASE_KEYS
    JOB = JOBS[a.job]
    if a.job == 'ql':
        BASE, BASE_KEYS = QLBASE, QLBASE_KEYS

    names = tlas(a.cfgA)
    namesB = dict(tlas(a.cfgB))
    lost = [n for n, _ in names if n not in namesB]
    gained = [n for n in namesB if n not in dict(names)]
    if lost or gained:
        print(f'FAIL TLA surface moved: lost={lost} gained={gained}')
        return 1

    def one(item):
        name, default = item
        if name in BASE_KEYS:
            return (name, 'base', True)
        extra = probe_arg(name, default)
        ra = compile_one(a.cfgA, extra)
        rb = compile_one(a.cfgB, extra)
        return (name, 'ok' if ra[0] == 0 else f'rc{ra[0]}', ra == rb)

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(one, names))

    bad = [r for r in res if not r[2]]
    nerr = sum(1 for r in res if r[1].startswith('rc'))
    nbase = sum(1 for r in res if r[1] == 'base')
    print(f'# TLAs {len(res)}  supplied-by-base {nbase}  '
          f'probe-compiles-clean {len(res)-nerr-nbase}  '
          f'probe-rejected-identically {nerr}')
    if bad:
        print(f'FAIL {len(bad)} probes differ between the two trees:')
        for n, s, _ in bad[:20]:
            print(f'  {n} ({s})')
        return 1
    print(f'PASS {len(res)}/{len(res)} probes identical')
    return 0


sys.exit(main())

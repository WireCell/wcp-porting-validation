#!/usr/bin/env python3
"""Doc 66 follow-up: blast radius of candidate detect_proton cut strengthenings.

Parses the TRACE logs of work-stmcamp-d66newtrace0 (all events with a
status-0 = accepted-STM bundle, new-diffusion arm) and re-evaluates the
end-proton decision cascade of TaggerCheckSTM::detect_proton
(clus/src/TaggerCheckSTM.cxx lines 1753-1788) under modified cuts.

TRACE line formats (s_log "clus.NeutrinoPattern"):
  main : detect_proton: End proton detection: ks1 ks2 r1 r2 ks3 r3 comb peak tail
  guard: detect_proton: proton_muon_guard: ... (bundle immune: returns false)
  d1   : detect_proton: End proton detection1: tm peak ks3 r3

A detect_proton call-group is associated with the NEXT
"persist_stm_fit: cluster N ... status=S" line; only statuses 0/5 ever call
detect_proton.  Torn log lines (known WCT spdlog tearing) are reported.
"""
import glob
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-stmcamp-d66newtrace0'

R_MAIN = re.compile(r'detect_proton: End proton detection: ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) (\d+) ?$')
R_GUARD = re.compile(r'detect_proton: proton_muon_guard:')
R_D1 = re.compile(r'detect_proton: End proton detection1: ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)$')
R_PERSIST = re.compile(r'persist_stm_fit: cluster (\d+) stmfit pass=(\d+) status=(\d+)')


def detect_proton(v, tm_cut=1.0, b_ks2=0.05, c_ks3=0.06, c_peak43=4.3):
    """Re-evaluate lines 1753-1788 from the logged discriminants.
    v: dict with ks1 ks2 r1 r2 ks3 r3 comb peak tail guarded tm(:opt)."""
    if v['guarded']:
        return False
    ks1, ks2, r2 = v['ks1'], v['ks2'], v['r2']
    ks3, r3, comb, peak, tail = v['ks3'], v['r3'], v['comb'], v['peak'], v['tail']
    # Block B (1769)
    if comb > 0.02 and peak > 2.3 and (tail <= 3 or (ks2 < b_ks2 and tail <= 12)):
        if tail <= 1 and peak > 2.5 and ks2 < 0.035 and abs(r2 - 1) < 0.1:
            return True
        if tail <= 1 and ((peak < 3.0 and ((ks1 < 0.06 and ks2 > 0.03) or (ks1 < 0.065 and ks2 > 0.04)))
                          or (ks1 < 0.035 and peak < 4.0)):
            return False
        if comb > 0.027:
            return True
    # Block C (1783) — needs tm; if the original run exited inside B (no d1
    # line), C never ran and cannot run under these cut changes either
    # (B's exit path 'return false' at B2 is unmodified).
    tm = v.get('tm')
    if tm is None:
        return False
    if tm < tm_cut and peak > 3.5:
        if (ks3 > c_ks3 and r3 > 1.1 and ks1 > 0.045) or (ks3 > 0.1 and ks2 < 0.19) or (r3 > 1.3):
            return True
        if (ks2 < 0.045 and ks3 > 0.03) or (peak > c_peak43 and ks3 > 0.03):
            return True
    elif tm < tm_cut and peak > 3.0:
        if ks3 > 0.12 and ks1 > 0.03:
            return True
    return False


def parse_event(path):
    """-> list of (cluster, pass, status, values-dict|'TORN')"""
    groups = []          # detect_proton call groups awaiting a persist line
    out = []
    cur = None
    for line in open(path, errors='replace'):
        m = R_MAIN.search(line)
        if m:
            cur = dict(zip(('ks1', 'ks2', 'r1', 'r2', 'ks3', 'r3', 'comb', 'peak'),
                           map(float, m.groups()[:8])))
            cur['tail'] = int(m.group(9)); cur['guarded'] = False
            groups.append(cur)
            continue
        if 'End proton detection: ' in line and not m:
            groups.append('TORN'); cur = None
            continue
        if R_GUARD.search(line):
            if cur is not None: cur['guarded'] = True
            continue
        m = R_D1.search(line)
        if m:
            if cur is not None:
                cur['tm'] = float(m.group(1))
            continue
        m = R_PERSIST.search(line)
        if m:
            cid, pss, st = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if st in (0, 5):
                v = groups.pop(0) if groups else None
                out.append((cid, pss, st, v))
            cur = None
    if groups:
        out.append((-1, -1, -1, 'ORPHAN'))
    return out


def main():
    # Prefer the batch stderr sink (.log_<evt>.log): the per-event file sink
    # tears these TRACE lines deterministically against a MultiAlgBlobClustering
    # timing line; the stderr sink writes the same records intact.
    logs = sorted(glob.glob(f'{ROOT}/.log_*.log'))
    if not logs:
        logs = sorted(glob.glob(f'{ROOT}/nusel_evt*/wct_nusel_evt*.log'))
    print(f"logs: {len(logs)}")
    bundles = []         # (evt, cid, status, v)
    torn = []
    for lp in logs:
        evt = lp.split('nusel_evt')[1].split('/')[0]
        for cid, pss, st, v in parse_event(lp):
            if v in ('TORN', 'ORPHAN') or v is None:
                torn.append((evt, cid, st, v)); continue
            # sanity: original decision must reproduce the recorded status
            orig = detect_proton(v)
            if orig != (st == 5):
                print(f"MISMATCH {evt}:{cid} status={st} but recompute proton={orig} {v}")
                continue
            bundles.append((evt, cid, st, v))
    print(f"parsed detect_proton bundles: {len(bundles)}  (torn/unparsed: {len(torn)} {torn})")
    n0 = sum(1 for b in bundles if b[2] == 0)
    ng = sum(1 for b in bundles if b[2] == 0 and b[3]['guarded'])
    print(f"  status-0 (accepted STM) reaching end-proton: {n0}  of which muon-guard-protected: {ng}")

    props = [
        ('P3  tm 1.0->1.05',            dict(tm_cut=1.05)),
        ('P4  B ks2 0.05->0.055',       dict(b_ks2=0.055)),
        ('P5a C ks3 0.06->0.058',       dict(c_ks3=0.058)),
        ('P5b C peak43 4.3->4.15',      dict(c_peak43=4.15)),
        ('P3+P5a',                      dict(tm_cut=1.05, c_ks3=0.058)),
        ('P3+P5b',                      dict(tm_cut=1.05, c_peak43=4.15)),
        ('P3+P4+P5a',                   dict(tm_cut=1.05, b_ks2=0.055, c_ks3=0.058)),
        ('P3+P4+P5b',                   dict(tm_cut=1.05, b_ks2=0.055, c_peak43=4.15)),
    ]
    for name, kw in props:
        flips = [(e, c) for e, c, st, v in bundles
                 if st == 0 and detect_proton(v, **kw)]
        print(f"{name:<26}: {len(flips)} accepted STMs newly vetoed  {[f'{e}:{c}' for e, c in flips]}")


if __name__ == '__main__':
    main()

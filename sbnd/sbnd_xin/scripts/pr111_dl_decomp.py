#!/usr/bin/env python3
"""doc pr/111 sec 2 -- decompose "exclusion OFF wins the neutrino vertex" into the
mechanisms that actually produce the win.

The DL vertex is chosen in two stages and the two must never be conflated:

  voxels[].dl_score   what the NET said   -- the sparse-conv response at a voxel,
                      range ~[-1,+1], read straight off PRVertexScoreboard.
  rows[].total        what the SELECTOR chose -- s_dl(=1000*dl_score) + s_snap +
                      s_clen + s_main + s_fv, argmax'd, admitted at min_accept=4.0.
                      The winner's voxel need not be rank 0.

Because s_clen(2) + s_main(2) + s_fv(0.5) alone clear min_accept=4.0, a candidate
on the main cluster in the FV is admitted even when the net is silent.  So
"the DL vertex was accepted" is NOT "the net found the vertex".

Classes (only defined on events whose target-metric verdict flips OFF-better):
  M-a  net-blind -> responsive : OFF accepted, ON net silent, OFF net response up
  M-b  confident, peak moves   : OFF accepted, ON net already confident
  M-c  the DL gets out of the way : OFF REJECTED -- the fallback vertex wins.
       Not a DL improvement at all.

Repro:
  python3 scripts/pr111_dl_decomp.py
"""
import json, os, sys, csv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM_ON  = os.path.join(HERE, 'work-vtx106-harv-base-nuecc48')    # fit_exclusion ON  (production)
ARM_OFF = os.path.join(HERE, 'work-vtx106-harv-nofitx-nuecc48')  # fit_exclusion OFF
TSV_ON  = os.path.join(HERE, 'docs/pr/106_events-prod-orig.tsv')
TSV_OFF = os.path.join(HERE, 'docs/pr/106_events-nofitx.tsv')

# net-blind cut: the per-voxel background floor sits at ~0.005; a net that has
# actually localized something runs 0.1-1.0.  0.05 is an order of magnitude above
# the floor and an order of magnitude below a real response.
BLIND = 0.05


def board(arm, evt):
    p = os.path.join(arm, f'pr_evt{evt}', f'calib-pr-evt{evt}.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get('vertex_scoreboard') or None


def summarize(b):
    """what the net said, what the selector chose."""
    vox = b.get('voxels') or []
    rows = b.get('rows') or []
    win = next((r for r in rows if r.get('dl_winner')), None)
    return dict(
        route=b.get('route', ''),
        accepted=bool(b.get('dl_accepted')),
        net_top=(vox[0]['dl_score'] if vox else float('nan')),   # what the NET said
        composite=b.get('dl_best_score', 0.0),                   # what the SELECTOR scored
        win_id=(win['vertex_id'] if win else None),
        win_dl=(win['s_dl'] / 1000.0 if win else float('nan')),
        win_snap=(win.get('snap_dis') if win else None),
        trad_id=b.get('hv_trad_main_vertex_id'),
        final_id=b.get('final_vertex_id'),
        ncloud=len((b.get('hv_cloud') or {}).get('x', []) or []),
    )


def read_tsv(p):
    out = {}
    with open(p) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r.get('sample') != 'nuecc48':
                continue
            out[r['evt']] = r
    return out


def main():
    t_on, t_off = read_tsv(TSV_ON), read_tsv(TSV_OFF)
    evts = sorted(set(t_on) & set(t_off), key=int)
    recs = []
    for e in evts:
        bo, bf = board(ARM_ON, e), board(ARM_OFF, e)
        if not bo or not bf:
            print(f'# evt {e}: missing board, skipped', file=sys.stderr)
            continue
        so, sf = summarize(bo), summarize(bf)
        hon = int(t_on[e].get('hit_M3') or 0)
        hoff = int(t_off[e].get('hit_M3') or 0)
        flip = hoff - hon                      # +1 = OFF fixes, -1 = OFF breaks
        cls = ''
        if flip > 0:
            if not sf['accepted']:
                cls = 'M-c'
            elif so['net_top'] < BLIND:
                cls = 'M-a'
            else:
                cls = 'M-b'
        elif flip < 0:
            cls = 'BREAK'
        recs.append(dict(evt=e, flip=flip, cls=cls, hon=hon, hoff=hoff,
                         tgt_on=t_on[e].get('target'), tgt_off=t_off[e].get('target'),
                         on=so, off=sf))

    w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    w.writerow(['evt', 'flip', 'class', 'hit_ON', 'hit_OFF', 'target_ON', 'target_OFF',
                'net_top_ON', 'net_top_OFF', 'composite_ON', 'composite_OFF',
                'route_ON', 'route_OFF', 'accept_ON', 'accept_OFF',
                'win_ON', 'win_OFF', 'trad_ON', 'trad_OFF', 'final_ON', 'final_OFF',
                'ncloud_ON', 'ncloud_OFF'])
    for r in recs:
        o, f = r['on'], r['off']
        w.writerow([r['evt'], f"{r['flip']:+d}", r['cls'], r['hon'], r['hoff'],
                    r['tgt_on'], r['tgt_off'],
                    f"{o['net_top']:.5f}", f"{f['net_top']:.5f}",
                    f"{o['composite']:.3f}", f"{f['composite']:.3f}",
                    o['route'], f['route'], int(o['accepted']), int(f['accepted']),
                    o['win_id'], f['win_id'], o['trad_id'], f['trad_id'],
                    o['final_id'], f['final_id'], o['ncloud'], f['ncloud']])

    # ---- summary to stderr so the TSV stays clean
    p = lambda *a: print(*a, file=sys.stderr)
    p('')
    p(f'nueCC48 events scored in both arms: {len(recs)}')
    p(f"target-metric M3:  ON {sum(r['hon'] for r in recs)}  OFF {sum(r['hoff'] for r in recs)}")
    fixes = [r for r in recs if r['flip'] > 0]
    breaks = [r for r in recs if r['flip'] < 0]
    p(f'OFF fixes {len(fixes)}: ' + ' '.join(r['evt'] for r in fixes))
    p(f'OFF breaks {len(breaks)}: ' + ' '.join(r['evt'] for r in breaks))
    p('')
    for c in ('M-a', 'M-b', 'M-c'):
        sel = [r for r in fixes if r['cls'] == c]
        p(f'  {c}: {len(sel)}  ' + ' '.join(r['evt'] for r in sel))
    p('')
    blind_on = [r for r in recs if r['on']['net_top'] < BLIND]
    blind_off = [r for r in recs if r['off']['net_top'] < BLIND]
    p(f'net-blind (top voxel dl_score < {BLIND}):  ON {len(blind_on)}/{len(recs)}   OFF {len(blind_off)}/{len(recs)}')
    acc_on = sum(r['on']['accepted'] for r in recs)
    acc_off = sum(r['off']['accepted'] for r in recs)
    p(f'DL accepted:  ON {acc_on}/{len(recs)}   OFF {acc_off}/{len(recs)}')
    silent_acc_on = sum(1 for r in recs if r['on']['accepted'] and r['on']['net_top'] < BLIND)
    silent_acc_off = sum(1 for r in recs if r['off']['accepted'] and r['off']['net_top'] < BLIND)
    p(f'  of which the NET was silent (dl_score < {BLIND}):  ON {silent_acc_on}   OFF {silent_acc_off}')
    import statistics as st
    for lab, k in (('ON ', 'on'), ('OFF', 'off')):
        v = sorted(r[k]['net_top'] for r in recs)
        p(f'  net_top {lab}: median={st.median(v):.4f} mean={st.mean(v):.4f} '
          f'min={v[0]:.4f} max={v[-1]:.4f}')


if __name__ == '__main__':
    main()

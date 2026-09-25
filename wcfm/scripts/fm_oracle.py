#!/usr/bin/env python3
"""fm_oracle.py -- the F2 Python oracle of the FM feature stage (wcfm/docs/01 sec 7 F2, docs/04).

Reads one FrameFileSink output (sim-frames-anode<N>.tar.bz2, the wct-sim-iso-track-nf-sp.jsonnet
product), builds the FM input EXACTLY as the training pack builder did (WC_FM_Sim: LArSoft SP
saved x0.005, read back x50, Labelling2D rebin_time_tick 4 = SUM of 4 ticks, HDF5FrameTap scale 1,
sparsify_to_prodjay.py keeps nonzero pixels; wcfm/docs/04 sec 1), i.e.

    pixel(channel, k) = input_scale * sum_{t in [4k, 4k+4)} gauss(channel, t),   active <=> pixel > 0
    x -> 2 (log10(x + m) - log10 m) / (log10(M + m) - log10 m) - 1     with the plane's VIEW_NORM (m, M)

one canvas per plane on the tight bounding box of the active pixels (floored at 64 x 64 on the
high side, dense_mae_adapter._rasterize_one), rows = channel ident - first channel of the plane
(U [0,800), V [800,1600), W [1600,2560) per APA: the W image is face 1 then face 0 in channel
order, the FM training layout), columns = 4-tick slice index k from the frame's tbin 0, and
runs the EAGER FMDenseStudent (WC_FM_DINO/sdcc/export_mbv3_ts.py, the wrapper the TorchScript was
proven bit-identical to in doc 03) on it.  Output: the sidecar layout as npz, per plane

    coords_<P>  (N, 2) int32   [channel ident, slice k]   sorted by (channel, slice)
    feat_<P>    (N, 128) f32   the student trunk output at that pixel
    canvas_<P>  (2, h, w) f32  the exact model input (for the C++ parity)
    bbox_<P>    [row0, row1, k0, k1]  (inclusive bounds, rows relative to the plane base)

    python3 scripts/fm_oracle.py work/000001_1/sim-frames-anode10.tar.bz2 --out /home/xqian/tmp/wcfm-f3/oracle-000001_1-anode10.npz
Runs with the toolkit direnv python (torch 2.5.1); --ts also runs the TorchScript on the same
canvases and reports its max |d| vs eager (informational: doc 03 proved 0 in this env).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

WCFM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WCT_BASE = os.path.dirname(os.path.dirname(WCFM_DIR)) if os.path.basename(os.path.dirname(WCFM_DIR)) == 'wcp-porting-img' else '/home/xqian/toolkit-dev'
FM_DINO = os.path.join(WCT_BASE, 'WC_FM_DINO')
DEFAULT_CKPT = os.path.join(FM_DINO, 'kd_checkpoints/kd_uni_mbv3_a1_inf01/checkpoint_step36000.pt')

# The training normalisation constants (WC_FM_DINO/dino/transforms.py VIEW_NORM), plane index order.
VIEW_NORM = {0: (2.77, 144977.0), 1: (2.97, 157944.0), 2: (3.75, 83861.2)}
PLANE_NAME = {0: 'U', 1: 'V', 2: 'W'}
PLANE_CH = {0: (0, 800), 1: (800, 1600), 2: (1600, 2560)}   # channel offsets within one APA
CH_PER_APA = 2560
MIN_CANVAS = 64


def log_norm(x, m, M):
    """FeatureLogTransform on float32 arrays (dino/transforms.py:15-51), no clip."""
    y0 = math.log10(m)
    scale = 2.0 / (math.log10(M + m) - y0)
    return ((np.log10(x.astype(np.float32) + np.float32(m)) - np.float32(y0)) * np.float32(scale) - np.float32(1.0)).astype(np.float32)


def load_frame(path, tag):
    from wirecell.util import ario
    f = ario.load(path)
    keys = [k for k in f.keys() if k.startswith('frame_' + tag)]
    if len(keys) != 1:
        sys.exit(f'{path}: expected one frame_{tag}* array, found {keys}')
    fk = keys[0]
    suffix = fk[len('frame_'):]           # e.g. gauss10_100
    frame = np.asarray(f[fk], dtype=np.float32)
    chans = np.asarray(f['channels_' + suffix]).astype(np.int64)
    tinfo = np.asarray(f['tickinfo_' + suffix], dtype=np.float64)   # (t0, tick, tbin)
    ident = int(suffix.rsplit('_', 1)[1])
    return frame, chans, tinfo, ident


def pack_plane(frame, chans, anode, plane, tick_span, scale, offset, threshold):
    """Return (q4 (nrows, nslices) f32, rows, base channel) for one plane.  Sum in float64 then
    cast, as the C++ does."""
    lo, hi = PLANE_CH[plane]
    base = anode * CH_PER_APA + lo
    nrows = hi - lo
    nticks = frame.shape[1]
    if nticks % tick_span:
        sys.exit(f'nticks {nticks} not a multiple of tick_span {tick_span}')
    rows = np.zeros((nrows, nticks), dtype=np.float32)
    sel = (chans >= base) & (chans < base + nrows)
    rows[chans[sel] - base] = frame[sel]
    q = rows.reshape(nrows, nticks // tick_span, tick_span).astype(np.float64).sum(axis=2)
    q = (q * scale + offset).astype(np.float32)
    return q, base


def canvas_of(q, m, M, threshold, min_canvas=MIN_CANVAS, bbox_pad=1):
    r, k = np.nonzero(q > threshold)
    if r.size == 0:
        return None
    r0, r1, k0, k1 = int(r.min()), int(r.max()), int(k.min()), int(k.max())
    h = max(r1 - r0 + 1, min_canvas)
    w = max(k1 - k0 + 1, min_canvas)
    if bbox_pad > 1:
        h = ((h + bbox_pad - 1) // bbox_pad) * bbox_pad
        w = ((w + bbox_pad - 1) // bbox_pad) * bbox_pad
    canvas = np.zeros((2, h, w), dtype=np.float32)
    canvas[0, r - r0, k - k0] = log_norm(q[r, k], m, M)
    canvas[1, r - r0, k - k0] = 1.0
    order = np.lexsort((k, r))          # sorted by (row, slice) == (channel, slice)
    return canvas, (r0, r1, k0, k1), r[order], k[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames', help='sim-frames-anode<N>.tar.bz2')
    ap.add_argument('--tag', default='gauss', help='trace tag prefix (default gauss; the anode ident is appended by the sim job)')
    ap.add_argument('--anode', type=int, default=-1, help='anode ident (default: from the file name)')
    ap.add_argument('--tick-span', type=int, default=4)
    ap.add_argument('--input-scale', type=float, default=0.25)
    ap.add_argument('--input-offset', type=float, default=0.0)
    ap.add_argument('--active-threshold', type=float, default=0.0, help='active <=> pixel > threshold')
    ap.add_argument('--bbox-pad', type=int, default=1)
    ap.add_argument('--planes', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--ts', default='', help='also run this TorchScript on the canvases (informational)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import torch
    sys.path.insert(0, os.path.join(FM_DINO, 'sdcc'))
    from export_mbv3_ts import build_wrapper  # noqa: E402
    torch.set_num_threads(args.threads)

    anode = args.anode
    if anode < 0:
        anode = int(os.path.basename(args.frames).split('anode')[1].split('.')[0])
    frame, chans, tinfo, ident = load_frame(args.frames, f'{args.tag}{anode}')
    t0, tick, tbin = float(tinfo[0]), float(tinfo[1]), int(tinfo[2])
    if tbin != 0:
        sys.exit(f'tbin {tbin} != 0: the oracle assumes the frame starts at its tbin 0')
    nneg = int((frame < 0).sum())
    print(f'[oracle] {args.frames}: frame ident={ident} shape={frame.shape} channels=[{chans.min()},{chans.max()}] '
          f't0={t0} tick={tick} negative samples={nneg}', flush=True)

    wrapper, _net, _ck, _dropped = build_wrapper(args.ckpt)
    wrapper = wrapper.to(args.device).eval()
    ts = None
    if args.ts:
        ts = torch.jit.load(args.ts, map_location=args.device).eval()

    out = {}
    meta = dict(frames=os.path.abspath(args.frames), frame_ident=ident, anode=anode, frame_time=t0, frame_tick=tick,
                tick_span=args.tick_span, input_tag=f'{args.tag}{anode}', input_scale=args.input_scale,
                input_offset=args.input_offset, active_threshold=args.active_threshold, bbox_pad=args.bbox_pad,
                min_canvas=MIN_CANVAS, face_layout='channel', ckpt=os.path.abspath(args.ckpt),
                ckpt_sha256=hashlib.sha256(open(args.ckpt, 'rb').read()).hexdigest(), torch=torch.__version__,
                device=args.device, planes={})
    for P in args.planes:
        m, M = VIEW_NORM[P]
        q, base = pack_plane(frame, chans, anode, P, args.tick_span, args.input_scale, args.input_offset, args.active_threshold)
        packed = canvas_of(q, m, M, args.active_threshold, bbox_pad=args.bbox_pad)
        name = PLANE_NAME[P]
        if packed is None:
            print(f'[oracle] plane {name}: no active pixel', flush=True)
            out[f'coords_{name}'] = np.zeros((0, 2), np.int32)
            out[f'feat_{name}'] = np.zeros((0, 128), np.float32)
            meta['planes'][name] = dict(plane=P, n_active=0, view_norm=[m, M])
            continue
        canvas, bbox, r, k = packed
        x = torch.from_numpy(canvas)[None].to(args.device)
        tt = time.time()
        with torch.no_grad():
            y = wrapper(x)
        dt = (time.time() - tt) * 1e3
        feat = y[0].permute(1, 2, 0)[torch.as_tensor(r - bbox[0]).long(), torch.as_tensor(k - bbox[2]).long(), :].cpu().numpy().astype(np.float32)
        d_ts = None
        if ts is not None:
            with torch.no_grad():
                yt = ts(x)
            d_ts = float((yt - y).abs().max())
        out[f'coords_{name}'] = np.stack([r + base, k], axis=1).astype(np.int32)
        out[f'feat_{name}'] = feat
        out[f'canvas_{name}'] = canvas
        out[f'bbox_{name}'] = np.array(bbox, dtype=np.int64)
        meta['planes'][name] = dict(plane=P, base_channel=int(base), n_active=int(r.size), view_norm=[m, M],
                                    bbox=[int(b) for b in bbox], canvas=[int(canvas.shape[1]), int(canvas.shape[2])],
                                    q_min=float(q[r, k].min()), q_max=float(q[r, k].max()), forward_ms=dt,
                                    ts_vs_eager_max_abs=d_ts)
        print(f'[oracle] plane {name}: N={r.size} bbox rows [{bbox[0]},{bbox[1]}] slices [{bbox[2]},{bbox[3]}] '
              f'canvas={tuple(canvas.shape[1:])} q=[{q[r, k].min():.3g},{q[r, k].max():.3g}] |feat| mean={np.abs(feat).mean():.4f} '
              f'forward {dt:.0f} ms' + (f' ts-eager max|d|={d_ts:.2e}' if d_ts is not None else ''), flush=True)
    out['meta'] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f'[oracle] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()

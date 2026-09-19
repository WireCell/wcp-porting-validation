#!/usr/bin/env python3
"""doc sbnd_xin/113 -- shared loaders for the 2-D-vs-3-D missing-charge instrument.

Everything is read from the production stage-A / stage-B products on disk (no wire-cell rerun):
  L0  SP charge      work-<s>-d102m/g<K>/frames-dnn.tar.bz2 :: frame_dnnsp_<evt>.npy (11276 ch x 3427 ticks),
                     summary_dnnsp (per-channel wiener RMS -> slicing threshold), chanmask_bad, tickinfo
  L1  activity       pctree ctpc_a<A>f0p<P> (charge, cident, wind, slice_index): the post-threshold
                     (channel x 4-tick slice) map the tiling saw  (aux/src/SamplingHelpers.cxx add_ctpc)
                     and, where kept, the imaging npz 'a' nodes (nuecc48 / ncpi0 only)
  L2' blobs          pctree per-blob 'scalar' PC (u/v/w_wire_index_min/max, slice_index_min/max, wpid, charge)
                     under each cluster node; the dead tree's 'scalar' holds the masked-fork (dead-region) blobs
  L3  scope          NOT readable offline (every stage-A cluster carries a t0/gid; see PCTree.in_scope) -> level 3 == level 2
  L4  PR bundle      tracking-pr.root T_proj_data (cluster_id -> channel rank / time_slice / charge) for the
                     bundle's stage clusters (union rule of pr/150), main id from calib main_vertex.cluster_id
                     or T_rec_charge.cluster_id

Conventions are NOT assumed: `d113_missing2d.py selftest` derives and checks every one of them on two whole
groups (see the doc, sec 1) before the census is allowed to run.
"""
import collections, io, json, os, re, tarfile
import numpy as np

SX = '/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
TMP = '/home/xqian/tmp/d113'
SAMPLES = ('nuecc48', 'ncpi0', 'mcp1k', 'mcp2k')
NAPA = 2
NCH_APA = 5638                       # channels per APA (TPC): U 1984 + V 1984 + W 1670
NWIRE = (1984, 1984, 1670)           # wires per plane per APA
PLANE_OFF = (0, 1984, 3968)          # channel offset of plane p inside an APA (verified by selftest 2)
BASE = (0, 3968, 7936)               # T_proj_data global channel-rank plane bases (calib meta 'base')
TICK_SPAN = 4                        # MaskSlices tick_span (active fork); slice = 4 ticks = 2 us
NTICK = 3427
NTHRESHOLD = (3.6, 3.6, 3.6)         # img.jsonnet MaskSlices nthreshold
DEFAULT_THRESHOLD = (5.87819e+02 * 4.0, 8.36644e+02 * 4.0, 5.67974e+02 * 4.0)   # MaskSlices.h:79 (zero-summary fallback)
T0_UNSET = -1e12                      # QLMatching.cxx:1351 sentinel (cluster with no matched flash)
LAYER_BIT = {1: 0, 2: 1, 4: 2}        # WirePlaneId layer bits -> plane index (wpid & 7)


def plane_of_channel(ch):
    """global channel -> (apa, plane, wire-in-plane), per the PLANE_OFF convention (checked by selftest 2)."""
    ch = np.asarray(ch, dtype=np.int64)
    apa = ch // NCH_APA
    loc = ch - apa * NCH_APA
    plane = np.where(loc < PLANE_OFF[1], 0, np.where(loc < PLANE_OFF[2], 1, 2))
    wip = loc - np.take(np.array(PLANE_OFF), plane)
    return apa, plane, wip


def channel_of(apa, plane, wip):
    return apa * NCH_APA + PLANE_OFF[plane] + wip


# ----------------------------------------------------------------------------------------------- frames (L0)
def group_dirs(sample):
    root = f'{SX}/work-{sample}-d102m'
    gs = sorted(glob_groups(root), key=lambda p: int(re.search(r'/g(\d+)$', p).group(1)))
    return gs


def glob_groups(root):
    import glob
    return [g for g in glob.glob(f'{root}/g*') if re.search(r'/g\d+$', g) and os.path.exists(f'{g}/frames-dnn.tar.bz2')]


def group_events(gdir):
    return [int(x) for x in open(f'{gdir}/events.txt').read().split()]


def iter_frames(gdir, wanted=None):
    """Yield (evt, dict(frame, channels, tickinfo, summary, chanmask)) sequentially from the group's bz2 archive.
    Members come in the order frame_, channels_, tickinfo_, summary_, chanmask_ per event; the archive is read
    once, streaming, so one event (~155 MB) is resident at a time."""
    cur, cur_ev = {}, None
    with tarfile.open(f'{gdir}/frames-dnn.tar.bz2', 'r:bz2') as tf:
        for m in tf:
            mm = re.match(r'(frame|channels|tickinfo|summary|chanmask)_(?:dnnsp|bad)_(\d+)\.npy$', m.name)
            if not mm:
                continue
            kind, ev = mm.group(1), int(mm.group(2))
            if cur_ev is not None and ev != cur_ev:
                if len(cur) == 5 and (wanted is None or cur_ev in wanted):
                    yield cur_ev, cur
                cur = {}
            cur_ev = ev
            if wanted is not None and ev not in wanted:
                continue
            cur[kind] = np.load(io.BytesIO(tf.extractfile(m).read()))
        if cur_ev is not None and len(cur) == 5 and (wanted is None or cur_ev in wanted):
            yield cur_ev, cur


def mask_frame(fr):
    """Zero the bad-channel (channel, tick range) mask in place, the way FrameMasking does before slicing.
    Returns the per-channel bad flag (any masked tick)."""
    F, ch = fr['frame'], fr['channels']
    row = {int(c): i for i, c in enumerate(ch.tolist())}
    bad = np.zeros(len(ch), bool)
    for c, t0, t1 in fr['chanmask'].tolist():
        i = row.get(int(c))
        if i is None:
            continue
        F[i, max(0, int(t0)):min(F.shape[1], int(t1))] = 0.0
        bad[i] = True
    return bad


def recompute_activity(fr, nthreshold=NTHRESHOLD, default_threshold=DEFAULT_THRESHOLD):
    """numpy port of Img::MaskSliceBase::thresholding (img/src/MaskSlice.cxx:173-215, :319-350) on the dnnsp
    frame (gauss == wiener == the same trace).  Returns (nch, nslice) float32 activity = sum of the gauss charge
    over the ACTIVE ticks of each 4-tick slice (0 where no tick is active), plus the per-channel threshold."""
    F, ch, summ = fr['frame'], fr['channels'], fr['summary']
    nch, nq = F.shape
    _, plane, _ = plane_of_channel(ch)
    thr = np.array(nthreshold)[plane] * summ
    zero = thr == 0
    thr = np.where(zero, np.array(default_threshold)[plane], thr)
    nsl = (nq + TICK_SPAN - 1) // TICK_SPAN
    # per-slice mean of the wiener charge (count = ticks actually present in the slice)
    pad = nsl * TICK_SPAN - nq
    Fp = np.pad(F, ((0, 0), (0, pad)))
    cnt = np.full(nsl, TICK_SPAN, dtype=np.float32); cnt[-1] = TICK_SPAN - pad if pad else TICK_SPAN
    M = Fp.reshape(nch, nsl, TICK_SPAN).sum(axis=2) / cnt
    # q_next[b] = M[b+1] if (b+1)*span < nq else 0 ; q_prev[b] = M[b-1] if (b-1)*span > 0 (b >= 2) else 0
    qnext = np.zeros_like(M); qnext[:, :-1] = M[:, 1:]
    last_ok = np.arange(nsl) + 1
    qnext[:, (last_ok * TICK_SPAN) >= nq] = 0.0
    qprev = np.zeros_like(M); qprev[:, 2:] = M[:, 1:-1]
    T = thr[:, None]
    active_slice_next = qnext > T
    active_slice_prev = qprev > T
    Fq = Fp.reshape(nch, nsl, TICK_SPAN)
    a1 = Fq > T[:, :, None]
    a2 = (Fq > qnext[:, :, None] / 3.0) & active_slice_next[:, :, None]
    a3 = (Fq > qprev[:, :, None] / 3.0) & active_slice_prev[:, :, None]
    act = a1 | a2 | a3
    A = np.where(act, Fq, 0.0).sum(axis=2).astype(np.float32)
    return A, thr, zero


# ----------------------------------------------------------------------------------------------- pctree (L1-L3)
def load_tensors(fname):
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    return {md['datapath']: (md, arrays.get(b)) for b, md in metas.items() if 'datapath' in md}


class PCTree:
    """One event's stage-A pctree: clusters (ident, t0, flags), blobs (per-plane wire ranges, slice range,
    cluster index), the ctpc activity map per (apa, plane), the dead-tree blobs."""

    def __init__(self, fname, ev):
        bp = load_tensors(fname)
        self.bp = bp
        P = f'pointtrees/{ev}/live'
        self.ev = ev
        A = lambda n: bp[P + '/pointclouds/namedpcs/' + n][1]
        m_cs = bp[P + '/lpcmaps/arrays/cluster_scalar'][1]
        m_sc = bp[P + '/lpcmaps/arrays/scalar'][1]
        # cluster nodes carry one cluster_scalar row; blob nodes one scalar row; tree walked depth first
        cl_of_blob = []
        ci = -1
        for i in range(len(m_cs)):
            if m_cs[i]:
                ci += 1
            if m_sc[i]:
                cl_of_blob.extend([ci] * int(m_sc[i]))
        self.blob_cluster = np.array(cl_of_blob, dtype=np.int64)
        cs = 'cluster_scalar/arrays/'
        self.cl_ident = A(cs + 'ident').astype(np.int64)
        self.cl_t0 = A(cs + 'cluster_t0').astype(np.float64)
        self.cl_main = A(cs + 'flag_main_cluster').astype(np.int64)
        self.cl_assoc = A(cs + 'flag_associated_cluster').astype(np.int64)
        self.cl_gid = A(cs + 'matched_flash_gid').astype(np.int64)
        sc = 'scalar/arrays/'
        self.b_wmin = np.stack([A(sc + f'{p}_wire_index_min') for p in 'uvw'], axis=1).astype(np.int64)
        self.b_wmax = np.stack([A(sc + f'{p}_wire_index_max') for p in 'uvw'], axis=1).astype(np.int64)
        self.b_smin = A(sc + 'slice_index_min').astype(np.int64)
        self.b_smax = A(sc + 'slice_index_max').astype(np.int64)
        self.b_wpid = A(sc + 'wpid').astype(np.int64)
        self.b_apa = (self.b_wpid >> 4) & 0xff          # WirePlaneId: layer | face<<3 | apa<<4 (iface/src/WirePlaneId.cxx)
        self.b_charge = A(sc + 'charge').astype(np.float64)
        self.b_npts = A(sc + 'npoints').astype(np.int64)
        assert len(self.blob_cluster) == len(self.b_smin), (len(self.blob_cluster), len(self.b_smin))
        self.ctpc = {}
        for apa in range(NAPA):
            for p, pn in enumerate('UVW'):
                key = f'ctpc_a{apa}f0p{pn}/arrays/'
                if P + '/pointclouds/namedpcs/' + key + 'charge' not in bp:
                    continue
                self.ctpc[(apa, p)] = dict(charge=A(key + 'charge').astype(np.float64),
                                           cident=A(key + 'cident').astype(np.int64),
                                           wind=A(key + 'wind').astype(np.int64),
                                           slice_index=A(key + 'slice_index').astype(np.int64),
                                           x=A(key + 'x').astype(np.float64))
        # dead tree (masked fork) blobs
        D = f'pointtrees/{ev}/dead'
        self.dead = None
        if D + '/pointclouds/namedpcs/scalar/arrays/wpid' in bp:
            Ad = lambda n: bp[D + '/pointclouds/namedpcs/' + n][1]
            self.dead = dict(wmin=np.stack([Ad(sc + f'{p}_wire_index_min') for p in 'uvw'], axis=1).astype(np.int64),
                             wmax=np.stack([Ad(sc + f'{p}_wire_index_max') for p in 'uvw'], axis=1).astype(np.int64),
                             smin=Ad(sc + 'slice_index_min').astype(np.int64), smax=Ad(sc + 'slice_index_max').astype(np.int64),
                             wpid=Ad(sc + 'wpid').astype(np.int64))
            self.dead['apa'] = (self.dead['wpid'] >> 4) & 0xff
        # opflash (root PC)
        self.opflash = None
        for k, (md, arr) in bp.items():
            pass
        of = P + '/pointclouds/namedpcs/opflash/arrays/'
        if of + 'time' in bp:
            self.opflash = dict(time=bp[of + 'time'][1], pe=bp[of + 'pe'][1], gid=bp[of + 'gid'][1])

    def in_scope(self):
        # Every stage-A cluster carries a t0 and a matched_flash_gid in the pctree (unmatched ones get a
        # 1000000+ rescue gid), and the Bee img/clustering layers draw the same points to within ~100, so the
        # PR-stage scope exclusion (clustering_switch_scope.cxx) is NOT readable from these products.  Level 3
        # therefore equals level 2 in the census; imaged-but-not-in-candidate charge is the BUNDLE class and
        # Part 2 splits it with the PR log / T_cluster.in_scope where needed.
        return np.ones(len(self.cl_t0), dtype=bool)


# ----------------------------------------------------------------------------------------------- npz (L1/L2 where kept)
class ImgNpz:
    """icluster-apa<A>-active.npz (aux/src/ClusterArrays.cxx schema).  Node rows: [desc, ident, ...].
    a: [desc, ident=channel, value, unc, index=wip, wpid]; s: [desc, ident=slicebin, sigv, sigu, frameid, start, span];
    b: [desc, ident, sigv, sigu, faceid, sliceid, start, span, u0,u1,v0,v1,w0,w1, nc, corners...]  (code, not the comment);
    w: [desc, ident, wip, seg, ch, plane, tail xyz, head xyz]."""

    def __init__(self, fname, ev):
        z = np.load(fname)
        k = lambda n: z[f'cluster_{ev}_{n}']
        self.a, self.s, self.b, self.w = k('anodes'), k('snodes'), k('bnodes'), k('wnodes')
        self.asedges, self.awedges, self.bsedges, self.bwedges = k('asedges'), k('awedges'), k('bsedges'), k('bwedges')
        # the edge rows are [edge-desc, tail-row, head-row] with rows indexing the node arrays of the two types
        # (ClusterArrays.cxx: edges are stored per (code1, code2) pair; tail is the first code)


# ----------------------------------------------------------------------------------------------- stage B (L4)
def load_bundle(prdir, ev):
    """Return (main_cluster_id, union ids, cells) where cells[(apa, plane)] = (wip array, slice array, charge)
    for the PR bundle's stage clusters; None when the event has no PR candidate."""
    import uproot
    root = f'{prdir}/tracking-pr.root'
    if not os.path.exists(root):
        return None
    f = uproot.open(root)
    cid = None
    calib = f'{prdir}/calib-pr-evt{ev}.json'
    if os.path.exists(calib):
        try:
            mv = json.load(open(calib)).get('main_vertex') or {}
            cid = mv.get('cluster_id')
        except Exception:
            cid = None
    rc = f['T_rec_charge'].arrays(['cluster_id', 'real_cluster_id'], library='np') if 'T_rec_charge' in f else None
    if cid is None and rc is not None and len(rc['cluster_id']):
        vals, cnts = np.unique(rc['cluster_id'], return_counts=True)
        cid = int(vals[np.argmax(cnts)])
    if cid is None:
        return None
    U = {int(cid)}
    if rc is not None and len(rc['cluster_id']):
        m = rc['cluster_id'] == cid
        U |= {int(r) // 1000 for r in rc['real_cluster_id'][m].tolist() if r > 0}
    pd = f['T_proj_data'].arrays(library='np')
    cells = collections.defaultdict(lambda: ([], [], []))
    if len(pd['cluster_id']) == 0:
        return dict(main=int(cid), union=sorted(U), cells={})
    for c, ch, ts, q in zip(pd['cluster_id'][0], pd['channel'][0], pd['time_slice'][0], pd['charge'][0]):
        if int(c) not in U:
            continue
        ch = np.asarray(ch).astype(np.int64); ts = np.asarray(ts).astype(np.int64); q = np.asarray(q, float)
        m = q > 0
        for apa, p, w, s, qq in zip(*rank_to_wire(ch[m]), ts[m], q[m]):
            L = cells[(int(apa), int(p))]
            L[0].append(int(w)); L[1].append(int(s)); L[2].append(float(qq))
    cells = {k: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for k, v in cells.items()}
    return dict(main=int(cid), union=sorted(U), cells=cells)


def rank_to_wire(rank):
    """T_proj_data global channel rank -> (apa, plane, wip): rank = BASE[p] + apa*NWIRE[p] + wip
    (calib meta base=[0,3968,7936], nch=[1984,1984,1670]); verified by selftest 5 against the ctpc charge."""
    rank = np.asarray(rank, dtype=np.int64)
    p = np.searchsorted(np.array(BASE[1:]), rank, side='right')
    loc = rank - np.take(np.array(BASE), p)
    n = np.take(np.array(NWIRE), p)
    apa = loc // n
    wip = loc - apa * n
    return apa, p, wip

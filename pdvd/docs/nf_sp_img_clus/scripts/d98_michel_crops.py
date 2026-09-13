#!/usr/bin/env python3
"""doc pdvd/98 -- select the clean STM+Michel electrons of PDHD (h28prod) and PDVD (p96vprod) and cut, for each,
the W-plane (collection) 256-channel x 1024-tick crop the DUNE-VD diffusion drift regressor
(DNN_ROI_SP/simulation/dunevd_singlep/diffusion_t0/ml/runs/m3-200k-w) was trained on, with the STM muon removed,
plus the Michel's TRUE drift distance from the Q-L matching t0.

    python3 d98_michel_crops.py --census            # selection counts + scan/d98/candidates.tsv, no frames read
    python3 d98_michel_crops.py [-j 8]              # + crops -> /home/xqian/tmp/d98/crops_<det>.npz, label_check.txt

READ-ONLY on every arm.  Nothing under work/ is written.

Inputs per candidate (all production products, nothing recomputed):
  T_stm_michel      (tracking-pr.root)  the chain's verdict row; hand truth joined as in d25_bragg_michel.py
  T_stm_michel_2d   (tracking-pr.root)  the per-cell table CheckSTM_Michel.cxx:2570-2588 writes when
                    michel_q2d_cells is on (both production bags): `channel` is the raw LArSoft ident, `time` the
                    absolute start tick of a 4-tick slice, and charge == frame_gauss[row(channel), time:time+4].sum()
                    exactly (checked 10/10 PDHD, 6/6 PDVD).  role 3 = Michel cells, 1 = the STM muon footprint
                    within michel_q2d_stm_window_cm (30 cm) of the stop, pred_mu = the muon fit's prediction there.
  T_stm_michel_pts + T_rec_charge (tracking-pr.root)  the fit points (role 1 muon, 3 Michel) and their
                    (pw, pt) wire/slice coordinates (every pts row is a T_rec_charge row, prep_stm_michel_scan.py:305).
                    pw is a ChanScheme rank; -> LArSoft ident through the sorted W channel list of chan_rank().
  SP gauss W frame  <SP arm>/proto*-sp-dnnroi-frames-anode<apa>.tar.bz2, reached through the production dir's
                    pctree symlink -> clustering arm -> clusters-*.tar.gz symlink -> SP arm.  Tick-level (0.5 us),
                    electrons, the same `gauss` tag and |frame| the training crops used (make_crops.load_frame).
  T_cluster.cluster_t0_us + pctree-evt*.tlas (trigger_offset_us, drift_speed_mmus; PDVD per volume) for the label.

Crop construction (tick-level, on the anode's W band):
  keep  = role-3 W cells U projected role-3 fit points, dilated +-KEEP_CH channels, +-KEEP_TK ticks
  muon  = role-1 W cells U projected role-1 fit points (whole chain), dilated +-MUON_CH, +-MUON_TK
          (tighter than keep: measured on 30 candidates BEFORE any prediction was run, dilating the muon like the
          keep band removed a median 48 % of the keep charge, the undilated cells 20 %, while the Michel's OWN cells
          lose only a median 8 % of their charge to the undilated muon cells.  +-1 ch / +-4 ticks fills the slice
          gaps between consecutive fit points and costs a median 39 % of the keep charge -- mostly muon charge that
          the keep dilation had swept in.  frac_lost below is the cleaner loss measure and is what tier A uses.)
  Z (primary)  |gauss| * keep * ~muon            everything that is not the Michel is exactly zero
  S            like Z, but an overlap pixel that sits on a role-3 table cell keeps the fraction
               max(charge - pred_mu, 0)/charge of its charge instead of being zeroed
  M (control)  |gauss| * keep                    the muon left in
  Zw           Z with the wider dilation KEEP_CH_W / KEEP_TK_W (sensitivity)
  All variants are cut with make_crops.cut() at the charge centroid of Z (the training anchor rule:
  make_crops.py:21-26 -- the crop is anchored on the observed charge, never on absolute time).

Label (two routes, both stored; route (a) is THE label, (b) the gate):
  (a) tick route, charge-weighted over the Z pixels:  drift_cm = (tick*0.5 - cluster_t0_us - trigger_offset_us)
      * drift_speed_mmus / 10   (Aux::time2drift SamplingHelpers.cxx:247-256 then PCTransforms.cxx:78-115, with
      the event's own .tlas values; PDVD anodes 0-3 use the *_bot_* pair, 4-7 the *_top_* pair)
  (b) fit route: x_anode(collection plane) - |x| averaged over the role-3 fit points, x_anode 353.10 PDHD /
      341.55 PDVD (d44_sigma_fit.py) -- no t0 arithmetic, so agreement between (a) and (b) checks the arithmetic.

Selection (pre-registered, every rule counted in the census print):
  hand verdict STM_MICHEL with michel_kind attached/both; chain is_stm==1 & topology_cleared_bits==0 (Bragg-path
  accept) & michel_found==1; michel_conn_type in {1 attached, 2 bridged}; n_retreat==0 & n_split==0 &
  michel_near_arm==0; 0 < michel_ke_best <= 52.8; the Michel's W cells in ONE (apa, face); products present.
  Tier A = Bragg ratio >= 0.8 & michel_q2d_mu_w/raw_w < 0.2 & n_stop_gammas == 0 & frac_lost < 0.2, where
  frac_lost = charge of the Michel's own (undilated) cells that falls inside the muon mask / their total charge;
  frac_overlap = charge removed from the keep band / keep-band charge (what separates M from Z) is stored too.
  Tier B = everything selected.  inrange = 80 <= drift(a) <= 340 (the model saturates below ~85 cm).
"""
import argparse, collections, csv, glob, io, json, multiprocessing as mp, os, sys, tarfile, time
import numpy as np
import uproot
from scipy.spatial import cKDTree

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
ML = "/home/xqian/toolkit-dev/DNN_ROI_SP/simulation/dunevd_singlep/diffusion_t0/ml"
OUT_TMP = "/home/xqian/tmp/d98"
SCAN = IMG + "/pdvd/docs/scan/d98"
os.environ.setdefault("STM_SCAN_RECORD", IMG + "/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json")
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
sys.path.insert(0, ML)
import d25_bragg_michel as m                      # noqa: E402
import prep_stm_michel_scan as prep               # noqa: E402  chan_rank()
import make_crops                                 # noqa: E402  cut(), NCHAN, NTICK

m.BR += ["t0_us", "michel_start_x", "michel_start_y", "michel_start_z", "michel_dis_cm", "michel_kink_deg",
         "michel_near_arm", "near_arm_dist_cm", "n_retreat", "n_split", "n_stop_gammas", "michel_q2d_raw_w",
         "michel_q2d_mu_w", "michel_q2d_n_w", "michel_ke_q2d_region", "michel_seg_id", "n_dots", "michel_far_len"]

ARM = {"pdhd": "h28prod", "pdvd": "p96vprod"}
POP = {"pdhd": "strict", "pdvd": None}            # PDHD headline population is APA0-strict (doc pdhd/23 sec 7)
WIRES = {"pdhd": "protodunehd-wires-larsoft-v1.json.bz2", "pdvd": "protodunevd-wires-larsoft-v7-uvwfit.json.bz2"}
FRAME_PFX = {"pdhd": "protodunehd-sp-dnnroi-frames-anode%d.tar.bz2", "pdvd": "protodune-sp-dnnroi-frames-anode%d.tar.bz2"}
X_ANODE_W = {"pdhd": 353.10, "pdvd": 341.55}      # collection-plane x TrackFitting reads (d44_sigma_fit.py)
ENDPOINT_MEV = 52.8
TICKS_PER_SLICE = 4
TICK_US = 0.5
KEEP_CH, KEEP_TK = 2, 10                          # dilation of a cell / fit point: channels, ticks
KEEP_CH_W, KEEP_TK_W = 3, 16                      # the wider variant
MUON_CH, MUON_TK = 1, 4                           # dilation of the muon mask (see the docstring)
TIER_A_RATIO, TIER_A_OVERLAP = 0.8, 0.2
DRIFT_LO, DRIFT_HI = 80.0, 340.0
D97_PICKS = IMG + "/pdvd/docs/scan/d97/picks.tsv"

COLS = ["det", "key", "kind", "source", "conn_type", "ratio", "muon_len", "michel_len", "ke_best", "ke_region",
        "kink_deg", "q2d_raw_w", "q2d_mu_w", "q2d_ovl", "n_stop_gammas", "n_dots", "t0_us", "stop_x", "stop_y", "stop_z",
        "apa", "face", "n_apaface", "d97", "selected", "why"]
CROP_COLS = ["det", "key", "apa", "face", "trigger_offset_us", "drift_speed_mmus", "cluster_t0_us",
             "drift_tick", "drift_fit", "drift_fit_rms", "drift_tick_sd", "q_keep", "q_overlap", "frac_overlap",
             "frac_lost", "n_pix", "n_cells3", "n_pts3", "n_pts1", "c_center", "t_center", "c0", "t0", "inrange"]


# ---------------------------------------------------------------- products
def prod_dir(det, evt):
    return f"{IMG}/{det}/work/{evt}_{ARM[det]}"


def sp_dir(det, evt):
    """the SP arm holding the frames, through the production dir's symlink chain"""
    pcs = glob.glob(prod_dir(det, evt) + "/pctree-evt*.tar.gz")
    if not pcs:
        return None
    clus = os.path.dirname(os.path.realpath(pcs[0]))
    cls = glob.glob(clus + "/clusters-apa-*-ms-active.tar.gz")
    if not cls:
        return None
    return os.path.dirname(os.path.realpath(cls[0]))


def read_tlas(det, evt):
    out = {}
    for tl in glob.glob(prod_dir(det, evt) + "/pctree-evt*.tlas"):
        for line in open(tl):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                out[k] = v
    return out


def tlas_pair(det, tl, apa):
    """(trigger_offset_us, drift_speed_mmus) of the drift volume the anode reads"""
    if det == "pdhd":
        return float(tl["trigger_offset_us"]), float(tl["drift_speed_mmus"])
    side = "bot" if apa < 4 else "top"
    return float(tl[f"trigger_offset_{side}_us"]), float(tl[f"drift_speed_{side}_mmus"])


def load_gauss(path):
    """(channels [LArSoft ids], |frame| float32 (nch, ntick), tbin) -- make_crops.load_frame's contract"""
    with tarfile.open(path, "r:bz2") as tf:
        names = tf.getnames()

        def arr(pfx):
            n = next(x for x in names if os.path.basename(x).startswith(pfx))
            return np.load(io.BytesIO(tf.extractfile(n).read()))
        frame = arr("frame_gauss")
        chans = arr("channels_gauss")
        tick = arr("tickinfo_gauss")
    assert abs(float(tick[1]) - 500.0) < 1e-6, tick
    return np.asarray(chans).astype(int), np.ascontiguousarray(np.abs(frame)).astype(np.float32), int(tick[2])


_W_IDENT = {}


def w_ident(det):
    """{ChanScheme W rank: LArSoft channel ident} (inverse of prep.chan_rank on plane 2)"""
    if det not in _W_IDENT:
        _W_IDENT[det] = {r: c for c, (p, r) in prep.chan_rank(det, WIRES[det]).items() if p == 2}
    return _W_IDENT[det]


# ---------------------------------------------------------------- census
def d97_keys():
    if not os.path.exists(D97_PICKS):
        return {}
    rows = csv.DictReader((l for l in open(D97_PICKS) if not l.startswith("#")), delimiter="\t")
    return {(r["det"], r["key"]): r["cls"] for r in rows}


def cells_apaface(det, evt, cid):
    """Counter of (apa, face) over the candidate's role-3 W cells, or None if the table is absent"""
    f = uproot.open(prod_dir(det, evt) + "/tracking-pr.root")
    if "T_stm_michel_2d" not in f:
        return None
    c = f["T_stm_michel_2d"].arrays(["cluster_id", "plane", "role", "apa", "face"], library="np")
    s = (c["cluster_id"] == cid) & (c["plane"] == 2) & (c["role"] == 3)
    return collections.Counter(zip(c["apa"][s].tolist(), c["face"][s].tolist()))


def candidates(det):
    its, missing = m.items(det, ARM[det], POP[det])
    d97 = d97_keys()
    n = collections.OrderedDict()
    n["hand STM_MICHEL, kind attached/both"] = 0
    n["+ Bragg-path accept & michel_found"] = 0
    n["+ conn_type attached(1)/bridged(2)"] = 0
    n["+ no retreat/split, no near-arm"] = 0
    n["+ 0 < ke_best <= 52.8 MeV"] = 0
    n["+ Michel W cells in one (apa,face)"] = 0
    n["+ frames present"] = 0
    rows = []
    for key, verdict, kind, src, d in its:
        if verdict != "STM_MICHEL" or kind not in m.MICHEL_KINDS:
            continue
        evt, cid = key.split("/"); cid = int(cid)
        r = dict(det=det, key=key, kind=kind, source=src, conn_type=int(d["michel_conn_type"]),
                 ratio=(d["contrast"] / d["contrast_expected"] if d["contrast_expected"] > 0 else 0.0),
                 muon_len=d["muon_len"], michel_len=d["michel_len"], ke_best=d["michel_ke_best"],
                 ke_region=d.get("michel_ke_q2d_region", -1), kink_deg=d["michel_kink_deg"],
                 q2d_raw_w=d["michel_q2d_raw_w"], q2d_mu_w=d["michel_q2d_mu_w"],
                 q2d_ovl=(d["michel_q2d_mu_w"] / d["michel_q2d_raw_w"] if d["michel_q2d_raw_w"] > 0 else -1),
                 n_stop_gammas=int(d["n_stop_gammas"]), n_dots=int(d["n_dots"]), t0_us=d["t0_us"],
                 stop_x=d["stop_x"], stop_y=d["stop_y"], stop_z=d["stop_z"], apa=-1, face=-1, n_apaface=0,
                 d97=d97.get((det, key), ""), selected=0, why="")
        rows.append(r)
        n["hand STM_MICHEL, kind attached/both"] += 1
        if not (int(d["is_stm"]) == 1 and int(d["topology_cleared_bits"]) == 0 and int(d["michel_found"]) == 1):
            r["why"] = "not Bragg-path accept / michel_found"; continue
        n["+ Bragg-path accept & michel_found"] += 1
        if r["conn_type"] not in (1, 2):
            r["why"] = "conn_type %d" % r["conn_type"]; continue
        n["+ conn_type attached(1)/bridged(2)"] += 1
        if not (int(d["n_retreat"]) == 0 and int(d["n_split"]) == 0 and int(d["michel_near_arm"]) == 0):
            r["why"] = "retreat/split/near_arm"; continue
        n["+ no retreat/split, no near-arm"] += 1
        if not (0 < r["ke_best"] <= ENDPOINT_MEV):
            r["why"] = "ke_best %.1f" % r["ke_best"]; continue
        n["+ 0 < ke_best <= 52.8 MeV"] += 1
        af = cells_apaface(det, evt, cid)
        if not af:
            r["why"] = "no role-3 W cells"; continue
        r["n_apaface"] = len(af)
        (r["apa"], r["face"]), _ = af.most_common(1)[0]
        if len(af) != 1:
            r["why"] = "W cells straddle %s" % sorted(af); continue
        n["+ Michel W cells in one (apa,face)"] += 1
        sd = sp_dir(det, evt)
        if sd is None or not os.path.exists(os.path.join(sd, FRAME_PFX[det] % r["apa"])):
            r["why"] = "frames missing"; continue
        n["+ frames present"] += 1
        r["selected"] = 1
    print(f"[{det} {ARM[det]}] population {len(its)} (record items missing a candidate: {missing})")
    for k, v in n.items():
        print(f"   {v:4d}  {k}")
    return rows


def write_tsv(path, rows, cols, header):
    with open(path, "w") as f:
        f.write(header)
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")


# ---------------------------------------------------------------- crops
def dilate(mask, dch, dtk):
    out = np.zeros_like(mask)
    nch, ntk = mask.shape
    for a in range(-dch, dch + 1):
        for b in range(-dtk, dtk + 1):
            src = mask[max(0, -a):nch - max(0, a), max(0, -b):ntk - max(0, b)]
            out[max(0, a):nch - max(0, -a), max(0, b):ntk - max(0, -b)] |= src
    return out


def build_one(args):
    det, row = args
    evt, cid = row["key"].split("/"); cid = int(cid)
    apa = row["apa"]
    f = uproot.open(prod_dir(det, evt) + "/tracking-pr.root")
    c = f["T_stm_michel_2d"].arrays(["cluster_id", "plane", "role", "apa", "channel", "time", "charge", "pred_mu"],
                                    library="np")
    cs = (c["cluster_id"] == cid) & (c["plane"] == 2) & (c["apa"] == apa)
    rc = f["T_rec_charge"].arrays(["x", "y", "z", "pw", "pt", "sub_cluster_id"], library="np")
    rs = (rc["sub_cluster_id"] // 1000) == cid
    P = np.c_[rc["x"][rs], rc["y"][rs], rc["z"][rs]]
    tree = cKDTree(P)
    pts = f["T_stm_michel_pts"].arrays(["cluster_id", "role", "x", "y", "z"], library="np")
    ps = pts["cluster_id"] == cid
    tc = f["T_cluster"].arrays(["cluster_id", "cluster_t0_us", "flash_time_us"], library="np")
    ti = np.nonzero(tc["cluster_id"] == cid)[0]
    assert len(ti) == 1, (det, row["key"], "T_cluster rows", len(ti))
    t0_us = float(tc["cluster_t0_us"][ti[0]])
    tl = read_tlas(det, evt)
    trig_us, v_mmus = tlas_pair(det, tl, apa)
    ident = w_ident(det)

    chans, frame, tbin = load_gauss(os.path.join(sp_dir(det, evt), FRAME_PFX[det] % apa))
    cidx = {int(ch): i for i, ch in enumerate(chans)}
    nch, ntk = frame.shape
    seed = {1: np.zeros((nch, ntk), bool), 3: np.zeros((nch, ntk), bool)}
    frac3 = np.ones((nch, ntk), np.float32)         # variant S: Michel fraction on overlap cells
    ncell3 = 0
    for role in (1, 3):
        sel = cs & (c["role"] == role)
        for ch, t, q, pmu in zip(c["channel"][sel], c["time"][sel], c["charge"][sel], c["pred_mu"][sel]):
            i = cidx.get(int(ch))
            if i is None:
                continue
            t = int(t)
            seed[role][i, max(0, t):min(ntk, t + TICKS_PER_SLICE)] = True
            if role == 3:
                ncell3 += 1
                if pmu > 0 and q > 0:
                    frac3[i, max(0, t):min(ntk, t + TICKS_PER_SLICE)] = max(q - pmu, 0.0) / q
    npts = {}
    xs3 = pts["x"][ps & (pts["role"] == 3)]          # the fit route needs no wire join
    for role in (1, 3):
        sel = ps & (pts["role"] == role)
        Q = np.c_[pts["x"][sel], pts["y"][sel], pts["z"][sel]]
        npts[role] = int(sel.sum())
        if not len(Q):
            continue
        d, j = tree.query(Q)
        for jj in j[d < 0.05]:
            ch = ident.get(int(np.floor(rc["pw"][rs][jj])))
            i = cidx.get(ch) if ch is not None else None
            if i is None:
                continue
            t = int(rc["pt"][rs][jj]) * TICKS_PER_SLICE
            seed[role][i, max(0, t):min(ntk, t + TICKS_PER_SLICE)] = True
    keep = dilate(seed[3], KEEP_CH, KEEP_TK)
    muon = dilate(seed[1], MUON_CH, MUON_TK)
    keep_w = dilate(seed[3], KEEP_CH_W, KEEP_TK_W)
    Z = frame * (keep & ~muon)
    S = frame * keep * np.where(muon, frac3, 1.0)
    M = frame * keep
    Zw = frame * (keep_w & ~muon)
    q_keep = float(M.sum()); q_ovl = float((frame * (keep & muon)).sum())
    q_own = float((frame * seed[3]).sum()); q_lost = float((frame * (seed[3] & muon)).sum())
    if Z.sum() <= 0:
        return None
    # centroid of Z (training anchor rule), then cut every variant there
    rows_i, cols_i = np.nonzero(Z)
    w = Z[rows_i, cols_i]
    c_center = float((chans[rows_i] * w).sum() / w.sum())
    t_center = float(((cols_i + tbin) * w).sum() / w.sum())
    crops = {}
    for name, img in (("Z", Z), ("S", S), ("M", M), ("Zw", Zw)):
        crops[name], c0, t0 = make_crops.cut(chans, img, tbin, c_center, t_center)
    # labels
    tk = cols_i + tbin
    drift_px = (tk * TICK_US - t0_us - trig_us) * v_mmus / 10.0
    drift_tick = float((drift_px * w).sum() / w.sum())
    drift_tick_sd = float(np.sqrt(((drift_px - drift_tick) ** 2 * w).sum() / w.sum()))
    dfit = X_ANODE_W[det] - np.abs(np.asarray(xs3, float)) if len(xs3) else np.array([np.nan])
    out = dict(det=det, key=row["key"], apa=apa, face=row["face"], trigger_offset_us=trig_us, drift_speed_mmus=v_mmus,
               cluster_t0_us=t0_us, drift_tick=drift_tick, drift_fit=float(np.mean(dfit)),
               drift_fit_rms=float(np.std(dfit)), drift_tick_sd=drift_tick_sd, q_keep=q_keep, q_overlap=q_ovl,
               frac_overlap=(q_ovl / q_keep if q_keep > 0 else -1), frac_lost=(q_lost / q_own if q_own > 0 else -1),
               n_pix=int(len(w)), n_cells3=ncell3,
               n_pts3=npts[3], n_pts1=npts[1], c_center=c_center, t_center=t_center, c0=c0, t0=t0,
               inrange=int(DRIFT_LO <= drift_tick <= DRIFT_HI))
    return out, crops


def main():
    global SCAN, OUT_TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", action="store_true", help="selection only; no frames read")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--det", default="pdhd,pdvd")
    # doc pdvd/98 sec 11 (the latest configuration, doc pdvd/99): a named run reads another PDVD arm with its own record
    # and writes scan/d98/<run>/ + /home/xqian/tmp/d98/<run>/.  No --run = round 1's paths, arms and record, unchanged.
    ap.add_argument("--run", default="")
    ap.add_argument("--pdvd-arm", default=ARM["pdvd"])
    ap.add_argument("--pdvd-record", default=m.VD_REC)
    a = ap.parse_args()
    if a.run:
        SCAN, OUT_TMP = f"{SCAN}/{a.run}", f"{OUT_TMP}/{a.run}"
    elif (a.pdvd_arm, a.pdvd_record) != (ARM["pdvd"], m.VD_REC):
        sys.exit("--pdvd-arm / --pdvd-record need --run (round 1's records are never overwritten)")
    ARM["pdvd"], m.VD_REC = a.pdvd_arm, a.pdvd_record
    os.makedirs(SCAN, exist_ok=True); os.makedirs(OUT_TMP, exist_ok=True)
    allrows = []
    for det in a.det.split(","):
        allrows += candidates(det)
    hdr = ("# doc pdvd/98 -- STM+Michel candidates for the drift-regressor validation (d98_michel_crops.py --census)\n"
           f"# PDHD {ARM['pdhd']} pop {POP['pdhd']}, record {os.environ['STM_SCAN_RECORD']}; PDVD {ARM['pdvd']} record {m.VD_REC}\n"
           "# selected=1 rows are cropped; `why` names the first failed rule; d97 = doc pdvd/97 showcase class\n")
    write_tsv(SCAN + "/candidates.tsv", allrows, COLS, hdr)
    print("wrote", SCAN + "/candidates.tsv", "selected:", sum(r["selected"] for r in allrows))
    if a.census:
        return 0
    todo = [(r["det"], r) for r in allrows if r["selected"]]
    t1 = time.time()
    with mp.Pool(a.j) as pool:
        res = pool.map(build_one, todo, chunksize=1)
    print(f"built {sum(r is not None for r in res)}/{len(todo)} crops in {time.time() - t1:.0f} s")
    lab = []
    for det in a.det.split(","):
        keys, meta, cr = [], [], {k: [] for k in ("Z", "S", "M", "Zw")}
        for (d, r), out in zip(todo, res):
            if d != det or out is None:
                continue
            info, crops = out
            keys.append(r["key"]); meta.append(info)
            for k in cr:
                cr[k].append(crops[k])
            lab.append(info)
        if not keys:
            continue
        np.savez_compressed(f"{OUT_TMP}/crops_{det}.npz", keys=np.array(keys),
                            **{k: np.stack(v).astype(np.float32) for k, v in cr.items()},
                            meta=json.dumps(meta))
        print(f"wrote {OUT_TMP}/crops_{det}.npz ({len(keys)} candidates)")
    hdr = ("# doc pdvd/98 -- per-candidate crop diagnostics and the two drift labels (d98_michel_crops.py)\n"
           "# drift_tick = (tick*0.5 - cluster_t0_us - trigger_offset_us)*drift_speed/10, charge-weighted over the Z pixels;\n"
           "# drift_fit = x_anode(W) - |x| over the role-3 fit points (353.10 PDHD / 341.55 PDVD)\n")
    write_tsv(SCAN + "/crops.tsv", lab, CROP_COLS, hdr)
    with open(SCAN + "/label_check.txt", "w") as f:
        f.write("# doc pdvd/98 -- tick-route vs fit-route drift label, per detector (cm)\n")
        for det in a.det.split(","):
            L = [r for r in lab if r["det"] == det]
            if not L:
                continue
            dd = np.array([r["drift_tick"] - r["drift_fit"] for r in L])
            dd = dd[np.isfinite(dd)]
            f.write(f"{det}: n={len(L)} (finite {len(dd)}) median(tick-fit)={np.median(dd):+.2f} mean={dd.mean():+.2f} "
                    f"sd={dd.std():.2f} max|d|={np.abs(dd).max():.2f}  "
                    f"gate median|d|<3: {'PASS' if abs(np.median(dd)) < 3 else 'FAIL'}\n")
            fl = np.array([r["frac_lost"] for r in L]); fo = np.array([r["frac_overlap"] for r in L])
            f.write(f"   frac_lost q10/50/90 = {np.quantile(fl, [.1, .5, .9]).round(2).tolist()}, n<0.2: {(fl < 0.2).sum()};"
                    f"  frac_overlap q10/50/90 = {np.quantile(fo, [.1, .5, .9]).round(2).tolist()}\n")
            tls = sorted({(r['trigger_offset_us'], r['drift_speed_mmus']) for r in L})
            f.write(f"   (trigger_offset_us, drift_speed_mmus) pairs seen: {tls[:6]}{' ...' if len(tls) > 6 else ''}\n")
    print(open(SCAN + "/label_check.txt").read())
    return 0


if __name__ == "__main__":
    sys.exit(main())

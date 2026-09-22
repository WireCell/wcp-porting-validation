#!/usr/bin/env python3
"""stm_release.py -- reader helpers for the STM + Michel release files.

One ROOT file per event, plain TTrees (readable with uproot or ROOT):
    events/<run6>_<evtid>/stm_michel_<det>_<run6>_<evtid>.root

    import stm_release as sr
    ev = sr.open_event("events/029107_1135/stm_michel_pdhd_029107_1135.root")
    cand = sr.candidates(ev)                  # dict of numpy arrays, one entry per STM candidate
    pts  = sr.points(ev, cluster_id=30)       # trajectory + dQ/dx rows of that candidate
    fl   = sr.flash(ev, flash_id=94)          # the flash row (per-PD PE vector included)
    wf   = sr.waveform(ev, flash_id=94, channel=143)   # raw / decon window of one PD

See ROOTFILE.md for every tree and branch, LIGHT.md for the optical side.
Only numpy + uproot (+ awkward, which uproot pulls in) are needed.
"""
import csv
import glob
import json
import os

import numpy as np
import uproot

# ---------------------------------------------------------------- code tables

#: T_stm.reject_bits -- is_stm == 1 iff reject_bits == 0
#: (toolkit clus/inc/WireCellClus/StmMichelFunctions.h, enum StmMichelReject)
REJECT_BITS = {
    0: "no_chain",            # no stm_pass/stm_fit PC, or no route entry -> stop
    1: "stop_unmatched",      # no graph vertex near the tagger's stop; chain walked greedily
    2: "no_bragg",            # contrast < bragg_contrast_min * expected
    3: "shape_flat",          # ks_mu + ks_margin >= ks_flat
    4: "not_muon_pid",        # proton or electron template beats muon
    5: "continuation",        # a collinear MIP track leaves the stop
    6: "stop_near_boundary",  # stop outside the fiducial inset
    7: "vertex_hadron",       # a long heavily-ionizing prong off the body
    8: "short",               # fewer than min_chain_points profile points
    9: "profile_sparse",      # too few LIVE points to judge
    10: "plateau_off_mip",    # plateau_med / mip_dqdx outside the MIP window
    11: "stop_into_dead",     # the visible end walks into a dead region
    12: "cluster_not_track",  # too few cluster points lie on the reconstructed track
    13: "profile_geometry",   # coiled end, or a long fitted segment the charge does not support
    14: "readout_edge",       # stop at the readout-window edge and no Michel object
}

#: T_stm_pts.role and T_michel_2d.role
ROLE = {
    0: "unclaimed cell (2-D table only: in the region, claimed by no segment)",
    1: "muon (the stopping-muon chain, entry -> stop)",
    2: "delta ray off the muon body",
    3: "Michel electron (the object at the stop)",
    4: "gamma / dot attached to the Michel",
    5: "unfitted dot cluster near the stop",
    6: "survey segment (knob-only)",
    7: "other arm at the stop (knob-only)",
    8: "segment census (knob-only)",
}

#: T_stm.michel_conn_type
MICHEL_CONN = {0: "none", 1: "attached", 2: "bridged", 3: "unfit dots"}


def reject_names(bits):
    """Decode a reject_bits value into the list of set bit names."""
    bits = int(bits)
    return [name for b, name in REJECT_BITS.items() if bits & (1 << b)]


# ---------------------------------------------------------------- files

def release_dir(path=None):
    """The release directory: the argument, or the parent of this scripts/ dir."""
    if path:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def list_events(rel=None):
    """Rows of index.csv as dicts (run, idx, evtid, n_candidates, n_stm, n_michel, root, bee)."""
    rel = release_dir(rel)
    with open(os.path.join(rel, "index.csv")) as fh:
        return list(csv.DictReader(fh))


def event_files(rel=None):
    """Sorted list of every per-event ROOT file in the release."""
    rel = release_dir(rel)
    return sorted(glob.glob(os.path.join(rel, "events", "*", "stm_michel_*.root")))


def open_event(path):
    """uproot ReadOnlyDirectory of one per-event file."""
    return uproot.open(path)


def event_info(ev):
    """The single T_event row as a dict (strings decoded)."""
    t = ev["T_event"].arrays(library="ak")[0]
    return {k: (t[k] if not hasattr(t[k], "tolist") else t[k].tolist()) for k in t.fields}


def dqdx_ref(rel=None):
    """The dQ/dx expectation tables (e/cm vs residual range cm) shipped with the release:
    returns (rr_cm, {particle: dqdx_e_per_cm})."""
    rel = release_dir(rel)
    d = json.load(open(os.path.join(rel, "dqdx_ref.json")))
    g = d["grid"]
    rr = g["start"] + g["step"] * np.arange(g["n"])
    return rr, {k: np.asarray(v, float) for k, v in d.items() if isinstance(v, list)}


# ---------------------------------------------------------------- charge side

def candidates(ev, branches=None):
    """T_stm as a dict of numpy arrays (all branches by default)."""
    return ev["T_stm"].arrays(branches, library="np")


def points(ev, cluster_id=None, role=None):
    """T_stm_pts rows, optionally for one candidate cluster and one role.
    q is dQ/dx in electrons per cm, rr the residual range in cm from the chain's stop
    (-1 on non-muon rows), L the distance along the chain from the entry."""
    p = ev["T_stm_pts"].arrays(library="np")
    m = np.ones(len(p["cluster_id"]), bool)
    if cluster_id is not None:
        m &= p["cluster_id"] == cluster_id
    if role is not None:
        m &= p["role"] == role
    return {k: v[m] for k, v in p.items()}


def fit_points(ev, cluster_id=None):
    """T_stm_fit rows (the PR fit of the candidate cluster: 3-D points with the
    projected wire-rank / time-slice coordinates pu, pv, pw, pt and dqdx in e/cm)."""
    p = ev["T_stm_fit"].arrays(library="np")
    if cluster_id is None:
        return p
    m = p["cluster_id"] == cluster_id
    return {k: v[m] for k, v in p.items()}


def cells_2d(ev, cluster_id=None, plane=None):
    """T_stm_2d rows: the fitted 2-D charge cells of the candidate cluster
    (one row per (plane, wire-rank, time-slice) cell)."""
    if "T_stm_2d" not in ev:
        return {}
    c = ev["T_stm_2d"].arrays(library="np")
    m = np.ones(len(c["cluster_id"]), bool)
    if cluster_id is not None:
        m &= c["cluster_id"] == cluster_id
    if plane is not None:
        m &= c["plane"] == plane
    return {k: v[m] for k, v in c.items()}


def michel_cells(ev, cluster_id=None, role=None):
    """T_michel_2d rows: the Michel / gamma / muon-footprint / region cells with
    measured charge and the muon-only and full fit predictions."""
    if "T_michel_2d" not in ev:
        return {}
    c = ev["T_michel_2d"].arrays(library="np")
    m = np.ones(len(c["cluster_id"]), bool)
    if cluster_id is not None:
        m &= c["cluster_id"] == cluster_id
    if role is not None:
        m &= c["role"] == role
    return {k: v[m] for k, v in c.items()}


def michel_region_sum(ev, cluster_id, region_cm=10.0):
    """Re-derive the production Michel charge estimator per plane from T_michel_2d with the
    C++ rule (CheckSTM_Michel michel_q2d_region, scope 1): every own-cluster cell within
    region_cm of the stop contributes measured - max(pred_mu, 0); on a cross-shared cell the
    measurement is not attributable and max(pred_all - pred_mu, 0) is taken instead.
    Returns ([q_u, q_v, q_w] in electrons, [n_u, n_v, n_w]); compare with
    T_stm.michel_q2d_region_{u,v,w} and michel_q2d_region_n_{u,v,w}."""
    C = michel_cells(ev, cluster_id)
    if not C:
        return [0.0, 0.0, 0.0], [0, 0, 0]
    s = (C["d_stop_cm"] >= 0) & (C["d_stop_cm"] <= region_cm) & (C["own_blob"] != 0)
    pmu = np.maximum(C["pred_mu"][s], 0.0)
    xs = C["xshared"][s] == 1
    contrib = np.where(xs, np.maximum(C["pred_all"][s] - pmu, 0.0), C["charge"][s] - pmu)
    pl = C["plane"][s]
    return [float(contrib[pl == p].sum()) for p in range(3)], [int((pl == p).sum()) for p in range(3)]


# ---------------------------------------------------------------- light side

def flashes(ev):
    """T_flash as awkward arrays (pe / sat / cov are per-OpDet vectors)."""
    return ev["T_flash"].arrays(library="ak")


def flash(ev, flash_id):
    """One T_flash row as a dict; 'pe' is the per-OpDet PE vector (numpy)."""
    t = ev["T_flash"].arrays(library="ak")
    i = int(np.flatnonzero(np.asarray(t["flash_id"]) == flash_id)[0])
    row = t[i]
    out = {}
    for k in row.fields:
        v = row[k]
        out[k] = np.asarray(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
    return out


def ophits(ev, flash_id=None, channel=None):
    """T_ophit rows (times in us on the light axis), optionally for one flash / channel."""
    h = ev["T_ophit"].arrays(library="np")
    m = np.ones(len(h["flash_id"]), bool)
    if flash_id is not None:
        m &= h["flash_id"] == flash_id
    if channel is not None:
        m &= h["channel"] == channel
    return {k: v[m] for k, v in h.items()}


def waveforms(ev, flash_id=None, channel=None, opdet=None):
    """T_opwf rows as awkward arrays: raw / decon (/ decon_roi) sample windows, one row per
    (flash, readout channel).  t0_us is the window start on the light time axis, tick_ns the
    sample spacing; sample i is at t0_us + i * tick_ns / 1000."""
    t = ev["T_opwf"].arrays(library="ak")
    m = np.ones(len(t), bool)
    if flash_id is not None:
        m &= np.asarray(t["flash_id"]) == flash_id
    if channel is not None:
        m &= np.asarray(t["channel"]) == channel
    if opdet is not None:
        m &= np.asarray(t["opdet"]) == opdet
    return t[m]


def waveform(ev, flash_id, channel):
    """One (flash, channel) window as a dict of numpy arrays plus a time axis 't_us'."""
    t = waveforms(ev, flash_id=flash_id, channel=channel)
    if len(t) == 0:
        raise KeyError(f"no waveform for flash {flash_id} channel {channel}")
    r = t[0]
    out = {k: (np.asarray(r[k]) if hasattr(r[k], "__len__") else r[k]) for k in r.fields}
    n = len(out["raw"])
    out["t_us"] = float(out["t0_us"]) + np.arange(n) * float(out["tick_ns"]) / 1000.0
    return out

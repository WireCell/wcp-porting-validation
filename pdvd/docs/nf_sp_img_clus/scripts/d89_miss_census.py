#!/usr/bin/env python3
"""doc pdvd/89 -- the missed stoppers, by the check that rejects each (doc 88 sec 9.5 item 1).

Read-only over one production prep dir, that arm's tracking-stm.root (T_stm_pass /
T_stm_eval), the compiled production config and the merged hand-scan record.  It forks
d77_review.py's plumbing; doc 77's record script stays untouched.  Sections:

  0  the thresholds each check uses, and where every number comes from: the compiled
     config, or (key absent there) the C++ member initializer, parsed from the source
  1  the two denominators: judged items with a candidate (census_score's population)
     and all judged items (an item with no candidate counts as is_stm 0)
  2  every missed stopper with a candidate: the set bits, each bit's margin to its
     threshold, the P1 state (Michel object, the floor it misses), the stop against the
     scanner's pin
  3  the judged items with no candidate: the tagger's pass status; for status 3, the first
     condition of TaggerCheckSTM::eval_stm_core_impl that each T_stm_eval row fails, and
     the best row per item; the THRU status-3 items are the negative control
  4  the partition: (a) the chain can act, (b) re-judge first / no lever, (c) closed by the
     owner's preference, (d) the tagger's, before a candidate exists
  5  levers: exact where the setting has one consumer and its inputs are recorded; a
     setting the geometric re-read (CheckSTM_Michel.cxx, doc 75's P1b) also consumes gives
     an exact LOWER bound on gains, with the anchored items that could add named
  6  movement against an older production arm on the same record

Usage: STM_SCAN_RECORD=<smx1a+smx3+smx4+smx5+smx6 record> d89_miss_census.py --prep DIR \
           --arm LABEL --cfg post.json [--prep-old DIR --arm-old LABEL] [--json OUT]
"""
import argparse, collections, glob, json, math, os, re, sys

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
TK = "/home/xqian/toolkit-dev/toolkit/clus/src"
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the smx1a+smx3+smx4+smx5+smx6 record (doc 88 sec 9.2)")
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--cfg", required=True, help="the arm's compiled config (wcsonnet output)")
ap.add_argument("--prep-old", default=None)
ap.add_argument("--arm-old", default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args()
out = {}


def load_prep(d):
    P = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("smprep-") and f.endswith(".json"):
            P[f[7:-5].replace("-c", "/")] = json.load(open(os.path.join(d, f)))["verdict"]
    return P


R = C.load_record()
J = {k: r for k, r in R.items() if C.judged(r)}
V = load_prep(a.prep)
print("record %s: %d items, %d judged; arm %s: %d candidates, %d on judged items" % (
    os.path.basename(C.REC), len(R), len(J), a.arm, len(V), sum(1 for k in J if k in V)))


# ---------------------------------------------------------------- 0 thresholds
def cfg_node(cfg, typ):
    stack = [cfg]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if x.get("type") == typ:
                return x.get("data", {})
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    sys.exit("no %s node in %s" % (typ, a.cfg))


SRC = {f: open(os.path.join(TK, f)).read() for f in ("CheckSTM_Michel.cxx", "TaggerCheckSTM.cxx")}


def cpp_default(member, f):
    m = re.search(r"\b%s\{([^}]*)\}" % re.escape(member), SRC[f])
    if not m:
        sys.exit("no initializer for %s in %s" % (member, f))
    s = m.group(1).strip()
    if s in ("true", "false"):
        return s == "true"
    scale = 10.0 if "units::cm" in s else 1.0          # WCT internal length unit is mm
    return float(s.split("*")[0]) * scale


cfg = json.load(open(a.cfg))
CK, TG = cfg_node(cfg, "CheckSTM_Michel"), cfg_node(cfg, "TaggerCheckSTM")
TH, THSRC = {}, {}


def th(key, node, nodename, f, member=None):
    if key in node:
        TH[key], THSRC[key] = node[key], "compiled %s" % nodename
    else:
        TH[key], THSRC[key] = cpp_default(member or "m_" + key, f), "C++ default (%s)" % f
    return TH[key]


for k in ("ks_margin", "plateau_mip_lo", "plateau_mip_hi", "mip_dqdx", "bragg_contrast_min", "topology_michel_ke_min",
          "topology_michel_len_min_cm", "continuation_max_angle_deg", "continuation_min_len_cm", "continuation_mip_lo",
          "continuation_mip_hi", "bragg_anchor_rise_min"):
    th(k, CK, "CheckSTM_Michel", "CheckSTM_Michel.cxx")
th("tagger_mip_dqdx", TG, "TaggerCheckSTM", "TaggerCheckSTM.cxx", "m_mip_dqdx")
if "mip_dqdx" in TG:
    TH["tagger_mip_dqdx"], THSRC["tagger_mip_dqdx"] = TG["mip_dqdx"], "compiled TaggerCheckSTM"
th("michel_res_length_cut", TG, "TaggerCheckSTM", "TaggerCheckSTM.cxx", "m_michel_res_len_cut")
th("accept_guards", TG, "TaggerCheckSTM", "TaggerCheckSTM.cxx")
th("guard_ratio2_max", TG, "TaggerCheckSTM", "TaggerCheckSTM.cxx")
print("\n=== 0. thresholds, with their source ===")
for k in TH:
    print("  %-28s %-12s %s" % (k, TH[k], THSRC[k]))
print("  fiducial %s, fv_tolerance %s (compiled; in_fv is a yes/no on the stop, no distance is recorded)" % (
    CK.get("fiducial"), CK.get("fv_tolerance")))
out["thresholds"] = {k: [TH[k], THSRC[k]] for k in TH}


# ---------------------------------------------------------------- helpers
def stm(k, P=V):
    return int(P[k]["is_stm"]) if k in P else 0


def rv(k):
    r = R[k]; return "%s/%s/%s" % (r["verdict"], r.get("michel_kind"), r.get("confidence"))


def rates(keys, pred):
    tp = sum(1 for k in keys if pred(k) and C.is_stopper(R[k])); fp = sum(1 for k in keys if pred(k) and not C.is_stopper(R[k]))
    fn = sum(1 for k in keys if not pred(k) and C.is_stopper(R[k])); tn = len(keys) - tp - fp - fn
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1); f1 = 2 * pu * ef / max(pu + ef, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, purity=round(pu, 3), efficiency=round(ef, 3), f1=round(f1, 3))


def fmt(d):
    return "%3d / %2d / %3d / %3d  pur %.3f eff %.3f F1 %.3f" % (d["tp"], d["fp"], d["fn"], d["tn"], d["purity"], d["efficiency"], d["f1"])


# ---------------------------------------------------------------- 1 denominators
print("\n=== 1. the two denominators (is_stm TP / FP / FN / TN) ===")
WITH = sorted(k for k in J if k in V)
ALL = sorted(J)
r_with, r_all = rates(WITH, stm), rates(ALL, stm)
print("  judged items with a candidate  n %3d: %s   <- census_score's population (doc 88 sec 9.3)" % (len(WITH), fmt(r_with)))
print("  all judged items               n %3d: %s   <- no candidate counts as 0 (doc 77's)" % (len(ALL), fmt(r_all)))
out["denominators"] = {"with_candidate": r_with, "all_judged": r_all}

# ---------------------------------------------------------------- 2 per bit
SHAPE = ("no_bragg", "shape_flat", "plateau_off_mip", "profile_sparse")
PREF = ("stop_near_boundary", "continuation", "vertex_hadron")


def margins(x):
    """one string per set bit: how far the recorded value sits beyond the threshold."""
    b, s = x["reject_names"], []
    for n in b:
        if n == "no_bragg":
            e = x["contrast_expected"] or 0
            s.append("no_bragg c/(%.1f e)=%.2f" % (TH["bragg_contrast_min"], (x["contrast"] or 0) / (TH["bragg_contrast_min"] * e) if e > 0 else float("nan")))
        elif n == "shape_flat":
            s.append("shape_flat ks_mu%+.2f-ks_flat=%+.3f" % (TH["ks_margin"], x["ks_mu"] + TH["ks_margin"] - x["ks_flat"]))
        elif n == "plateau_off_mip":
            s.append("plateau %.2f MIP not in [%.1f,%.1f]" % ((x["plateau_med"] or 0) / TH["mip_dqdx"], TH["plateau_mip_lo"], TH["plateau_mip_hi"]))
        elif n == "continuation":
            s.append("continuation %.1f deg<%g len %.1f>%g cm mip %.2f in [%g,%g]" % (
                x["cont_angle_deg"], TH["continuation_max_angle_deg"], x["cont_len"], TH["continuation_min_len_cm"],
                x["cont_mip"], TH["continuation_mip_lo"], TH["continuation_mip_hi"]))
        elif n == "stop_near_boundary":
            s.append("out of FV at (%.0f, %.0f, %.0f)" % (x["stop_x"], x["stop_y"], x["stop_z"]))
        elif n == "vertex_hadron":
            s.append("vertex_hadron n_body_hadron %d" % x["n_body_hadron"])
        else:
            s.append(n)
    return s


def p1_state(x):
    if not x["michel_found"]:
        return "no Michel object"
    miss = []
    if x["michel_conn_type"] not in (1, 2): miss.append("conn %d" % x["michel_conn_type"])
    if x["michel_ke_best"] < TH["topology_michel_ke_min"]: miss.append("ke<%g" % TH["topology_michel_ke_min"])
    if x["michel_len"] < TH["topology_michel_len_min_cm"]: miss.append("len<%g" % TH["topology_michel_len_min_cm"])
    return "Michel conn %d %.1f MeV %.1f cm%s" % (x["michel_conn_type"], x["michel_ke_best"], x["michel_len"],
                                                   " (P1 misses: %s)" % ", ".join(miss) if miss else " (P1 passes)")


def pin_col(k, x):
    r = R[k]; p = r.get("pin") or {}
    if p.get("placed"):
        d = math.dist((p["x"], p["y"], p["z"]), (x["stop_x"], x["stop_y"], x["stop_z"]))
        return "pin |stop-pin| %.1f cm" % d
    if r.get("pin_rr") is not None:
        return "pin_rr %.1f (anchor shift %.1f)" % (r["pin_rr"], x["bragg_anchor_shift_cm"])
    return ""


FNC = sorted(k for k in WITH if C.is_stopper(J[k]) and not stm(k))
print("\n=== 2. missed stoppers WITH a candidate (%d), by the deciding bits; each bit's margin ===" % len(FNC))
groups = collections.defaultdict(list)
for k in FNC:
    b = V[k]["reject_names"]
    g = ("shape only: " if all(n in SHAPE for n in b) else "") + "+".join(b)
    groups[g].append(k)
for g, ks in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    print("  -- %s: %d" % (g, len(ks)))
    for k in ks:
        x = V[k]
        print("     %-14s %-30s %s | %s | %s" % (k, rv(k), "; ".join(margins(x)), p1_state(x), pin_col(k, x)))
out["fn_with_candidate"] = {g: ks for g, ks in groups.items()}
# margin summaries (the pre-registered P2 / P3)
sf = sorted(V[k]["ks_mu"] + TH["ks_margin"] - V[k]["ks_flat"] for k in FNC if "shape_flat" in V[k]["reject_names"])
nb = sorted((V[k]["contrast"] or 0) / (TH["bragg_contrast_min"] * V[k]["contrast_expected"]) for k in FNC
            if "no_bragg" in V[k]["reject_names"] and (V[k]["contrast_expected"] or 0) > 0)
q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")
print("  shape_flat on %d: ks_mu%+.2f-ks_flat min %+.3f p25 %+.3f median %+.3f max %+.3f; beyond +0.02: %d of %d" % (
    len(sf), TH["ks_margin"], sf[0], q(sf, .25), q(sf, .5), sf[-1], sum(1 for v in sf if v > 0.02), len(sf)))
print("  no_bragg on %d: contrast/(%.1f expected) min %.2f p25 %.2f median %.2f max %.2f; under 0.9: %d of %d" % (
    len(nb), TH["bragg_contrast_min"], nb[0], q(nb, .25), q(nb, .5), nb[-1], sum(1 for v in nb if v < 0.9), len(nb)))
print("  (bits are read on the ANCHORED profile; bragg_anchor_fallback 1 would mean the geometric reading stood: %d of %d)" % (
    sum(1 for k in FNC if V[k].get("bragg_anchor_fallback")), len(FNC)))
out["margins"] = {"shape_flat": sf, "no_bragg": nb}

# ---------------------------------------------------------------- 3 no candidate
STATUS = {0: "accepted", 1: "TGM", 2: "long leftover past kink", 3: "dQ/dx eval (KS) rejected", 4: "extra-tracks veto",
          5: "proton endpoint", 6: "fitted, no decision", 7: "accept_guards / cathode / second-track / deficit / vertex-kink", 8: "round-2 fit empty"}
_ev = {}


def stm_eval(ev, arm):
    fn = "%s/%s_%s/tracking-stm.root" % (C.WORK, ev, arm)
    if fn not in _ev:
        t = None
        if os.path.exists(fn):
            import uproot
            try:
                t = uproot.open(fn)["T_stm_eval"].arrays(library="np")
            except Exception:
                t = None
        _ev[fn] = t
    return _ev[fn]


STAGES = ["F0 ratio2 cap", "F1 ks1-ks2>=0", "F2 near-flat", "F3 straight residual", "F4 residual not Michel-like", "F5 no accept branch", "accept"]


def row_stage(r, waive_f1_below=None):
    """(stage index, F3 excluded?) of one eval row, TaggerCheckSTM.cxx eval_stm_core_impl in code order.
    T_stm_eval lengths are in cm: peak_range reads 40 / 20 and com_range 35 / 15, the code's constants.
    F3 needs res_length1 / res_dis1, which T_stm_eval does not record: it is excluded exactly only when
    ave_res/mip <= 0.9, or res_length <= 8 cm and ave_res/mip <= 2.3 (both of its clauses then fail).
    waive_f1_below: the lever section 3 sizes -- F1 does not reject a row whose comb is below it."""
    ks1, ks2, r1, r2 = r["ks1"], r["ks2"], r["ratio1"], r["ratio2"]
    resl, resq = r["res_length"], r["ave_res_dqdx"] / TH["tagger_mip_dqdx"]
    comb = ks1 - ks2 + (abs(r1 - 1) - abs(r2 - 1)) / 1.5 * 0.3
    if TH["accept_guards"] and r2 > TH["guard_ratio2_max"]: return 0, True
    if ks1 - ks2 >= 0.0 and not (waive_f1_below is not None and comb < waive_f1_below): return 1, True
    if math.hypot(ks2 / 0.06, (r2 - 1) / 0.06) < 1.4 and comb > -0.02: return 2, True
    f3_excl = resq <= 0.9 or (resl <= 8.0 and resq <= 2.3)
    cut = TH["michel_res_length_cut"] / 10.0        # compiled in internal units (mm)
    q = r["ave_res_dqdx"]; mip = TH["tagger_mip_dqdx"]
    if ((resl > 20 and resq > 1.2 and comb > -0.02) or (resl > 16 and q > 1.45 * mip) or
            (resl > 10 and q > 1.45 * mip and comb > -0.05) or (resl > 10 and q > 1.7 * mip) or
            (resl > cut and q > 1.85 * mip) or (resl > cut and q > 1.45 * mip and comb > -0.05) or
            (resl > 4 and resq > 1.4 and comb > 0.02) or (resl > 2 and resq > 4.5)):
        return 4, f3_excl
    if not r["strong"]:
        acc = (ks1 - ks2 < -0.02 and ((ks2 > 0.09 and abs(r2 - 1) > 0.1) or r2 > 1.5 or ks2 > 0.2)) or comb < 0
    else:
        acc = ((ks1 - ks2 < -0.02 and (ks2 > 0.09 or r2 > 1.5) and ks1 < 0.05 and abs(r1 - 1) < 0.1) or
               (comb < 0 and ks1 < 0.05 and abs(r1 - 1) < 0.1))
    return (6 if acc else 5), f3_excl


def eval_rows(k, arm):
    ev, c = k.split("/")
    t = stm_eval(ev, arm); p = C.stm_pass(ev, arm)
    if t is None or p is None:
        return None, []
    sp = p["cluster_id"] == int(c)
    st = sorted(set(int(s) for s in p["status"][sp]))
    rej_pass = set(int(x) for x, s in zip(p["pass"][sp], p["status"][sp]) if int(s) == 3)
    rows = []
    for i in range(len(t["cluster_id"])):
        if int(t["cluster_id"][i]) != int(c) or int(t["pass"][i]) not in rej_pass:
            continue
        r = {n: float(t[n][i]) for n in ("ks1", "ks2", "ratio1", "ratio2", "res_length", "ave_res_dqdx", "peak_range", "offset_length", "com_range")}
        r.update(strong=int(t["strong"][i]), verdict=int(t["verdict"][i]), pass_=int(t["pass"][i]))
        r["stage"], r["f3_excl"] = row_stage(r)
        r["comb"] = r["ks1"] - r["ks2"] + (abs(r["ratio1"] - 1) - abs(r["ratio2"] - 1)) / 1.5 * 0.3
        rows.append(r)
    return st, rows


def best_row(rows):
    # the deepest stage reached, then the smallest ks1 - ks2: the row closest to accepting
    return sorted(rows, key=lambda r: (-r["stage"], r["ks1"] - r["ks2"]))[0] if rows else None


def row_str(r):
    tag = STAGES[r["stage"]]
    if r["stage"] >= 4 and not r["f3_excl"]:
        tag = "F3 or " + tag.split(" ", 1)[0]
    if r["stage"] == 6 and r["verdict"] == 0:
        tag = "F3 (inferred: every recorded test accepts)" if not r["f3_excl"] else "MISMATCH"
    return "%-34s ks1-ks2 %+.3f comb %+.3f ks2 %.3f r2 %.2f res %.1f cm %.2f MIP | pr %.0f off %.0f com %.0f cm" % (
        tag, r["ks1"] - r["ks2"], r["comb"], r["ks2"], r["ratio2"], r["res_length"], r["ave_res_dqdx"] / TH["tagger_mip_dqdx"],
        r["peak_range"], r["offset_length"], r["com_range"])


NC = sorted(k for k in J if k not in V)
print("\n=== 3. judged items with NO candidate (%d): the tagger's pass status; status 3 by eval row ===" % len(NC))
nc_stat, s3 = {}, collections.defaultdict(list)
mism = 0
for k in NC:
    st, rows = eval_rows(k, a.arm)
    nc_stat[k] = st
    if st and 3 in st:
        s3["stopper" if C.is_stopper(J[k]) else "other"].append((k, rows))
    mism += sum(1 for r in rows if (r["stage"] == 6) != (r["verdict"] == 1) and not (r["stage"] == 6 and not r["f3_excl"]))
print("  status by verdict: %s" % dict(collections.Counter(("%s: %s" % (J[k]["verdict"], "+".join(str(s) for s in (nc_stat[k] or ["none"])))) for k in NC)))
print("  eval rows whose recomputed verdict disagrees with the recorded one (a transcription check; 0 expected): %d" % mism)
for k in NC:
    if C.is_stopper(J[k]) and not (nc_stat[k] and 3 in nc_stat[k]):
        print("  %-14s %-30s status %s (%s)" % (k, rv(k), nc_stat[k], "; ".join(STATUS.get(s, "?") for s in (nc_stat[k] or []))))
for grp in ("stopper", "other"):
    print("  -- status 3, record %s: %d" % ("STM_*" if grp == "stopper" else "not a stopper", len(s3[grp])))
    for k, rows in s3[grp]:
        cnt = collections.Counter(STAGES[r["stage"]].split(" ")[0] for r in rows)
        print("     %-14s %-30s %d rows %s | best: %s" % (k, rv(k), len(rows), dict(sorted(cnt.items())), row_str(best_row(rows))))
# the separation question: can one best-row number admit the stoppers without the negatives?
print("  -- separation on the best row (admit if value < t): stoppers admitted vs negatives admitted")
for name, f in (("ks1-ks2", lambda r: r["ks1"] - r["ks2"]), ("comb", lambda r: r["comb"])):
    sv = sorted(f(best_row(rows)) for _, rows in s3["stopper"] if rows)
    tv = sorted(f(best_row(rows)) for _, rows in s3["other"] if rows)
    line = []
    for i, t in enumerate(sv):
        line.append("%d/%d at t>%+.3f: %d of %d negatives" % (i + 1, len(sv), t, sum(1 for v in tv if v <= t), len(tv)))
    print("     %-8s %s" % (name, " | ".join(line)))
    print("              stoppers %s ; negatives %s" % (["%+.3f" % v for v in sv], ["%+.3f" % v for v in tv]))
fs = collections.Counter(STAGES[best_row(rows)["stage"]].split(" ")[0] for _, rows in s3["stopper"] if rows)
fo = collections.Counter(STAGES[best_row(rows)["stage"]].split(" ")[0] for _, rows in s3["other"] if rows)
print("     best-row stage: stoppers %s ; negatives %s" % (dict(sorted(fs.items())), dict(sorted(fo.items()))))
out["no_candidate"] = {k: nc_stat[k] for k in NC}


# The lever the separation points at: F1 (ks1 - ks2 >= 0) waived on a row whose combined score is
# below t; the row must then pass F2..F5 as written.  Exact on the tagger's eval where F3 is excluded;
# "unless F3" otherwise.  It only hands a cluster on: the chain's own verdict on it needs an arm.
def waived(rows, t):
    st = [row_stage(r, t) for r in rows]
    return any(s == 6 and ex for s, ex in st), any(s == 6 for s, ex in st)


def rclass(k):
    if k not in R: return "unjudged"
    if not C.judged(R[k]): return "MESSY/UNCLEAR"
    return "stopper" if C.is_stopper(R[k]) else "not a stopper"


POP = {}
for fn in sorted(glob.glob("%s/*_%s/tracking-stm.root" % (C.WORK, a.arm))):
    ev = os.path.basename(os.path.dirname(fn))[: -len(a.arm) - 1]
    p = C.stm_pass(ev, a.arm)
    if p is None: continue
    for c in sorted(set(int(x) for x, s in zip(p["cluster_id"], p["status"]) if int(s) == 3)):
        k = "%s/%d" % (ev, c)
        if k in V: continue
        POP[k] = eval_rows(k, a.arm)[1]
print("  -- the waiver 'F1 does not reject a row with comb < t', over EVERY status-3 cluster without a candidate "
      "(%d in %d events), by record class: exact accepts [+ accepts unless the unrecorded F3 fires]" % (
          len(POP), len(set(k.split("/")[0] for k in POP))))
print("     classes: %s" % dict(collections.Counter(rclass(k) for k in POP)))
out["waiver"] = {}
for t in (-0.02, -0.015, -0.01, -0.005, 0.0):
    ex = collections.Counter(); mb = collections.Counter(); names = []
    for k, rows in POP.items():
        e, m = waived(rows, t)
        if e: ex[rclass(k)] += 1
        if m and not e: mb[rclass(k)] += 1
        if m: names.append("%s[%s%s]" % (k, rclass(k)[:7], "" if e else ",F3?"))
    print("     t %+.3f: %s + %s | %s" % (t, dict(ex), dict(mb), " ".join(names)))
    out["waiver"]["%+.3f" % t] = names


# ---------------------------------------------------------------- 5 levers (computed before 4, printed after)
SFb, NBb, PSb, PMb = 1 << 3, 1 << 2, 1 << 9, 1 << 10


def score(sel):
    tp = sum(1 for k in J if sel(k) and C.is_stopper(J[k])); fp = sum(1 for k in J if sel(k) and not C.is_stopper(J[k]))
    fn = sum(1 for k in J if not sel(k) and C.is_stopper(J[k]))
    g = sorted(k for k in J if sel(k) and not stm(k)); l = sorted(k for k in J if not sel(k) and stm(k))
    return (tp, fp, fn), g, l


def p1_sel(K, L):
    def sel(k):
        x = V.get(k)
        if not x: return False
        b = int(x["reject_bits"] or 0) | int(x.get("topology_cleared_bits") or 0)
        if x["michel_found"] and x["michel_conn_type"] in (1, 2) and x["michel_ke_best"] >= K and x["michel_len"] >= L:
            b &= ~(NBb | SFb | PSb)
        return b == 0
    return sel


def pm_sel(lo, hi):
    def sel(k):
        x = V.get(k)
        if not x: return False
        b = int(x["reject_bits"] or 0)
        if b & ~PMb: return False
        if not (b & PMb): return True
        pm = (x["plateau_med"] or 0) / TH["mip_dqdx"]
        return lo <= pm <= hi
    return sel


def ks_sel(m):
    def sel(k):
        x = V.get(k)
        if not x: return False
        b = int(x["reject_bits"] or 0); tcb = int(x.get("topology_cleared_bits") or 0)
        if b & ~SFb: return False
        if tcb & SFb: return True
        return not (x["ks_mu"] + m >= x["ks_flat"])
    return sel


def bc_sel(f):
    def sel(k):
        x = V.get(k)
        if not x: return False
        b = int(x["reject_bits"] or 0)
        if b & ~NBb: return False
        if not (b & NBb): return True
        return (x["contrast"] or 0) >= f * (x["contrast_expected"] or 0)
    return sel


LEVERS = []
for K, L in ((10, 3), (10, 2), (8, 3), (5, 3), (3, 3), (0, 3), (5, 2), (3, 2), (0, 0)):
    LEVERS.append(("P1 floors ke>=%g len>=%g" % (K, L), "exact", p1_sel(K, L)))
for lo, hi in ((0.6, 1.6), (0.6, 2.0), (0.5, 1.6), (0.5, 2.0), (0.0, 99.0)):
    LEVERS.append(("plateau window [%.2f, %.2f]" % (lo, hi), "exact lower bound", pm_sel(lo, hi)))
for m in (-0.02, -0.03, -0.04):
    LEVERS.append(("ks_margin %+.2f" % m, "exact lower bound", ks_sel(m)))
for f in (0.6, 0.5, 0.4, 0.3):
    LEVERS.append(("bragg_contrast_min %.1f" % f, "indicative (two set sites)", bc_sel(f)))


def bundle_sel(k):
    """P1 floors 3 MeV / 3 cm AND the plateau window [0.6, 2.0], together: the two 0-FP settings as one."""
    x = V.get(k)
    if not x: return False
    b = int(x["reject_bits"] or 0) | int(x.get("topology_cleared_bits") or 0)
    if b & PMb and 0.6 <= (x["plateau_med"] or 0) / TH["mip_dqdx"] <= 2.0:
        b &= ~PMb
    if x["michel_found"] and x["michel_conn_type"] in (1, 2) and x["michel_ke_best"] >= 3 and x["michel_len"] >= 3:
        b &= ~(NBb | SFb | PSb)
    return b == 0


LEVERS.append(("bundle: P1 3/3 + plateau [0.6, 2.0]", "exact lower bound", bundle_sel))
LEV = []
for name, kind, sel in LEVERS:
    (tp, fp, fn), g, l = score(sel)
    LEV.append((name, kind, tp, fp, fn, g, l))
_b = LEV[-1]
zero_fp_gain = set(_b[5]) if not [k for k in _b[5] if not C.is_stopper(J[k])] and not _b[6] else set()
anchored = sorted(k for k in FNC if V[k]["bragg_anchor_shift_cm"] > 0 and any(n in SHAPE for n in V[k]["reject_names"]))

# ---------------------------------------------------------------- 4 partition
FN_ALL = sorted(k for k in J if C.is_stopper(J[k]) and not stm(k))
part = collections.OrderedDict((p, []) for p in ("d", "c", "a1", "a2", "b1", "b2", "other"))
for k in FN_ALL:
    x = V.get(k)
    if x is None: part["d"].append(k); continue
    b = x["reject_names"]
    if any(n in PREF for n in b): part["c"].append(k); continue
    if not all(n in SHAPE for n in b): part["other"].append(k); continue
    if k in zero_fp_gain: part["a1"].append(k); continue
    if x["michel_found"] and x["michel_conn_type"] in (1, 2): part["a2"].append(k); continue
    part["b1" if J[k].get("confidence") in ("medium", "low") else "b2"].append(k)
LAB = {"d": "(d) the tagger's: no candidate", "c": "(c) closed by the owner's preference: boundary / continuation / hadron",
       "a1": "(a1) the chain can act: the 0-FP bundle (P1 floors 3 MeV / 3 cm + plateau_mip_hi 2.0) clears it",
       "a2": "(a2) a Michel object under the floors that the bundle does not clear",
       "b1": "(b1) re-judge first: shape only, no Michel object, medium / low confidence",
       "b2": "(b2) shape only, no Michel object, high / owner confidence: no lever on this record", "other": "other bits"}
print("\n=== 4. the partition of all %d missed stoppers ===" % len(FN_ALL))
for p, ks in part.items():
    if not ks and p == "other": continue
    print("  -- %s: %d  (STM_MICHEL %d, STM_ONLY %d)" % (LAB[p], len(ks), sum(C.is_michel(J[k]) for k in ks), sum(not C.is_michel(J[k]) for k in ks)))
    for k in ks:
        x = V.get(k)
        print("     %-14s %-30s %s" % (k, rv(k), ("; ".join(margins(x)) + " | " + p1_state(x)) if x else "status %s" % nc_stat.get(k)))
hi = [k for k in FN_ALL if J[k].get("confidence") == "high"]
print("  high-confidence misses (%d): %s" % (len(hi), " ".join("%s[%s]" % (k, "+".join(V[k]["reject_names"]) if k in V else "no cand") for k in hi)))
out["partition"] = part

print("\n=== 5. levers on all %d judged items (no candidate = 0), gains and losses by name ===" % len(J))
for name, kind, tp, fp, fn, g, l in LEV:
    print("  %-32s %-26s TP/FP/FN %d/%d/%d | gained %s | lost %s" % (name, kind, tp, fp, fn,
          " ".join("%s[%s,%s]" % (k, J[k]["verdict"][:9], J[k].get("confidence")) for k in g) or "-", " ".join(l) or "-"))
print("  items the geometric re-read could ADD under a looser plateau / ks / contrast setting (anchored, shape bits): %d: %s" % (
    len(anchored), " ".join(anchored)))
out["levers"] = [dict(name=n, kind=kd, tp=tp, fp=fp, fn=fn, gained=g, lost=l) for n, kd, tp, fp, fn, g, l in LEV]

# ---------------------------------------------------------------- 6 movement
if a.prep_old:
    VO = load_prep(a.prep_old)
    FO = set(k for k in J if C.is_stopper(J[k]) and not stm(k, VO))
    FN_set = set(FN_ALL)
    print("\n=== 6. movement %s -> %s on the same record: missed %d -> %d ===" % (a.arm_old, a.arm, len(FO), len(FN_set)))
    for lab, ks in (("left the missed set", sorted(FO - FN_set)), ("entered the missed set", sorted(FN_set - FO))):
        print("  -- %s: %d" % (lab, len(ks)))
        for k in ks:
            f = lambda P: ("is_stm %d bits %s" % (P[k]["is_stm"], "+".join(P[k]["reject_names"]) or "none")) if k in P else "no candidate"
            print("     %-14s %-30s %s: %s -> %s: %s" % (k, rv(k), a.arm_old, f(VO), a.arm, f(V)))
    moved = sorted(k for k in J if C.is_stopper(J[k]) is False and stm(k, VO) != stm(k))
    print("  non-stoppers whose is_stm moved: %s" % (" ".join("%s %d->%d" % (k, stm(k, VO), stm(k)) for k in moved) or "none"))
    out["movement"] = {"left": sorted(FO - FN_set), "entered": sorted(FN_set - FO)}

if a.json:
    json.dump(out, open(a.json, "w"), indent=1, default=str)
    print("wrote", a.json)

#!/usr/bin/env python3
"""doc pdvd/90 -- the offline twin of doc 89's 0-FP bundle, written BEFORE the arms run:
topology_michel_ke_min 3.0 (P1's energy floor, C++ default 10) and plateau_mip_hi 2.0
(compiled 1.6), alone and together, over EVERY candidate of the production payloads --
judged, MESSY / UNCLEAR and unjudged -- not only the record's items.

What is exact and what is not (doc 89 sec 5):
  * P1 (stm_michel_topology_clear, CheckSTM_Michel.cxx:3989) runs last, on recorded inputs
    (michel_found, michel_conn_type, michel_ke_best, michel_len): its reject_bits,
    topology_cleared_bits and is_stm are exact.  A lower floor can also clear shape bits on
    a candidate that keeps another bit: reject_bits / topology_cleared_bits move while is_stm
    stays 0 -- listed as branch movers.
  * The plateau test (:2729) reads the recorded ANCHORED plateau, so every gain it names is
    exact.  Two things are not recorded and are listed, not predicted:
      - doc 75's geometric re-read (:2803) runs only while the anchored reading still carries
        a shape bit; a wider window can let it clear more (anchored items with a shape bit
        left: possible extra gains);
      - on a candidate where the re-read already stood (bragg_anchor_fallback 1), the recorded
        bragg fields are the geometric ones; if the anchored reading's only failure was a
        plateau in (1.6, 2.0], the anchored reading now stands instead -- is_stm stays 1 but
        contrast / ks / plateau / bragg_anchor_fallback change (possible branch movers).
  * An is_stm 0 -> 1 candidate un-withholds its capture gammas (stop_gamma_require_stm,
    doc 85), so its event's zip / calib / T_rec_charge may move.

Usage: STM_SCAN_RECORD=<smx6 record> d90_twin.py --prep DIR [--json OUT]
"""
import argparse, collections, json, os, sys

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the smx1a+smx3+smx4+smx5+smx6 record (doc 88 sec 9.2)")
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--log-arm", default=None, help="production arm whose DEBUG anchor_geo_fallback lines name the re-reads")
a = ap.parse_args()

R = C.load_record()
V = {}
for f in sorted(os.listdir(a.prep)):
    if f.startswith("smprep-") and f.endswith(".json"):
        V[f[7:-5].replace("-c", "/")] = json.load(open(os.path.join(a.prep, f)))["verdict"]
NB, SF, PS, PM = 1 << 2, 1 << 3, 1 << 9, 1 << 10
SHAPE = NB | SF | PS | PM
MIP, LO = 55000.0, 0.6
PROD = dict(ke=10.0, hi=1.6)
ARMS = {"p90vb": dict(ke=3.0, hi=2.0), "p90vk": dict(ke=3.0, hi=1.6), "p90vp": dict(ke=10.0, hi=2.0)}


def verdict(x, ke, hi):
    """(reject_bits, topology_cleared_bits) under the floors / window."""
    b = int(x["reject_bits"] or 0) | int(x.get("topology_cleared_bits") or 0)     # the bits before P1
    if b & PM and x.get("bragg_valid") and LO <= (x["plateau_med"] or 0) / MIP <= hi:
        b &= ~PM
    clr = 0
    if x["michel_found"] and x["michel_conn_type"] in (1, 2) and x["michel_ke_best"] >= ke and x["michel_len"] >= 3.0:
        clr = b & (NB | SF | PS)
    return b & ~clr, clr


def cls(k):
    if k not in R: return "unjudged"
    if not C.judged(R[k]): return "MESSY/UNCLEAR"
    return "stopper" if C.is_stopper(R[k]) else "not a stopper"


def names(b):
    return "+".join(n for i, n in enumerate(C.STM_BITS) if b >> i & 1) or "STM"


GEO = {}   # key -> (anchored bits, geometric bits, which reading stood), from production's own log
if a.log_arm:
    import glob, re
    for d in sorted(glob.glob("%s/*_%s" % (C.WORK, a.log_arm))):
        ev = os.path.basename(d)[: -len(a.log_arm) - 1]
        for f in glob.glob(d + "/wct_pr_*.log"):
            for line in open(f, errors="replace"):
                m = re.search(r"anchor_geo_fallback: cluster (\d+) .*anchored contrast [^|]* bits (\d+) \| "
                              r"geometric contrast [^|]* bits (\d+) -> (\w+) stands", line)
                if m:
                    GEO["%s/%s" % (ev, m.group(1))] = (int(m.group(2)), int(m.group(3)), m.group(4))
    print("production's geometric re-read ran on %d candidates (DEBUG anchor_geo_fallback lines of %s)" % (len(GEO), a.log_arm))

# the twin must reproduce production itself before it predicts anything
bad = [k for k, x in V.items() if verdict(x, **PROD) != (int(x["reject_bits"] or 0), int(x.get("topology_cleared_bits") or 0))]
print("twin vs production on %d candidates: %d mismatches %s" % (len(V), len(bad), bad[:5]))
J = [k for k in R if C.judged(R[k])]
out = {}
for arm, p in ARMS.items():
    mv_stm, mv_br = [], []
    for k in sorted(V):
        x = V[k]
        b0, c0 = int(x["reject_bits"] or 0), int(x.get("topology_cleared_bits") or 0)
        b1, c1 = verdict(x, **p)
        if (b1 == 0) != (b0 == 0):
            mv_stm.append(k)
        elif (b1, c1) != (b0, c0):
            mv_br.append(k)
    stm = lambda k: (verdict(V[k], **p)[0] == 0) if k in V else False
    tp = sum(1 for k in J if stm(k) and C.is_stopper(R[k])); fp = sum(1 for k in J if stm(k) and not C.is_stopper(R[k]))
    fn = sum(1 for k in J if not stm(k) and C.is_stopper(R[k]))
    print("\n=== %s %s: is_stm on the 576 judged %d / %d / %d" % (arm, p, tp, fp, fn))
    print("  is_stm movers (%d): %s" % (len(mv_stm), " ".join("%s[%s: %s -> %s]" % (
        k, cls(k), names(int(V[k]["reject_bits"] or 0)), names(verdict(V[k], **p)[0])) for k in mv_stm) or "-"))
    print("  branch-only movers, is_stm unchanged (%d): %s" % (len(mv_br), " ".join("%s[%s: %s -> %s]" % (
        k, cls(k), names(int(V[k]["reject_bits"] or 0)), names(verdict(V[k], **p)[0])) for k in mv_br) or "-"))
    geo_add = fb_move = []
    if p["hi"] != PROD["hi"]:
        # The re-read runs only while the ANCHORED reading still carries a shape bit, and can newly
        # stand only where its own sole failure in production was the plateau test (bits == 1024 on
        # its DEBUG line; the geometric plateau value itself is not logged, so these are possible,
        # not predicted).  Where the re-read stood, the anchored reading takes over instead when its
        # sole failure was the plateau (anchored bits == 1024) and its plateau is now inside.
        def anch_left(k, ab):
            if ab & PM and LO <= (V[k]["plateau_med"] or 0) / MIP <= p["hi"]:
                ab &= ~PM
            return ab & SHAPE
        geo_add = sorted(k for k, (ab, gb, st) in GEO.items() if k in V and st == "anchored" and gb == PM and anch_left(k, ab))
        fb_move = sorted(k for k, (ab, gb, st) in GEO.items() if st == "geometric" and ab == PM)
        other = lambda k: names(int(V[k]["reject_bits"] or 0) & ~SHAPE) if int(V[k]["reject_bits"] or 0) & ~SHAPE else "no other bit"
        print("  NOT predicted, possible: gains through the geometric re-read (its sole production failure the plateau): %d: %s"
              % (len(geo_add), " ".join("%s[%s, %s]" % (k, cls(k), other(k)) for k in geo_add) or "-"))
        print("  NOT predicted, possible: branch movers (is_stm stays 1) where the anchored reading would take over: %d: %s"
              % (len(fb_move), " ".join("%s[%s]" % (k, cls(k)) for k in fb_move) or "-"))
    out[arm] = dict(census=[tp, fp, fn], is_stm_movers=mv_stm, branch_movers=mv_br, possible_geo=geo_add, possible_fb=fb_move)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
    print("wrote", a.json)

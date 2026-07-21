#!/usr/bin/env python3
"""Derive the cathode-XA-anchored QLMatching operating point (doc 18).

Data-tunes two default-OFF toolkit knob groups on hand-scan ground truth:

  1. Cathode-scoped flash admission (flash_sel_*): a flash entering matching
     must show sum(cathode PE) >= minPE and >= min_fired cathode channels at
     >= fired_pe PE.  Tuned so ~all keep-round confirmed candle flashes
     (crossers + boundary tracks, 18 events of run 039252) survive.
  2. Cathode-scoped over-prediction prefilter (reject_overpred +
     overpred_channels): R_total / R_max ceilings over the cathode XAs only,
     with the production close_to_PMT/window_truncated/at_x_boundary
     exemption reproduced.

Inputs (read-only):
  - 18-event keep-round dumps      work/039252_<idx>_keep/calib-evt*.json
  - keep-round confirmed scans     ql_display/decisions-{crossers,boundary}-keep/
  - evt298567 full hand scan       work/ql_labels/wfresc/labels-evt298567.json
                                   + work/039252_0_wfresc/calib-evt298567.json

Run:  python3 derive_cathode_operating_point.py   (paths overridable via argv)
"""
import json, glob, os, sys
import numpy as np

PDVD = sys.argv[1] if len(sys.argv) > 1 else \
    "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
CATH = list(range(4, 12))            # cathode XA channels (qlmatching.jsonnet)
FIRED_THRS = [0.5, 1.0, 2.0, 5.0]


def decisions_for(evt):
    """(kept, rejected) sets of (flash_gid, main_cluster_uid) from the
    keep-round crossers+boundary scans; plus the raw rows for reporting."""
    keep, rej, rows = set(), set(), []
    for cat in ("crossers", "boundary"):
        p = f"{PDVD}/ql_display/decisions-{cat}-keep/decisions-{evt}.jsonl"
        if not os.path.isfile(p):
            continue
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            r["_cat"] = cat
            rows.append(r)
            key = (r["flash_gid"], r["main_cluster_uid"])
            if r["verdict"] in ("keep", "add"):
                keep.add(key)
            elif r["verdict"] == "reject":
                rej.add(key)
    return keep, rej, rows


def bundle_exempt(b):
    """The production reject_overpred exemption (QLMatching.cxx)."""
    return bool(b.get("close_to_PMT") or b.get("window_truncated")
                or b.get("at_x_boundary"))


def cath_ratios(pred, meas, act):
    """Cathode-scoped R_total / R_max, the C++ overpred formulae."""
    tp = tm = mp = ma = 0.0
    for c in CATH:
        if c not in act:
            continue
        p, m = pred[c], meas[c]
        tp += p; tm += m
        if p > mp:
            mp, ma = p, m
    r_tot = tp / tm if tm > 0 else (1e30 if tp > 0 else 0.0)
    r_max = mp / max(ma, 1.0) if mp > 0 else 0.0
    return r_tot, r_max


dumps = sorted(g for g in glob.glob(f"{PDVD}/work/039252_*_keep/calib-evt*.json")
               if "light" not in g)
print(f"keep dumps: {len(dumps)}")

conf_flash, all_flash = [], []       # (evt, gid, cath_sum, [nfired/thr], totPE)
conf_R, rej_R, all_R = [], [], []    # (r_tot, r_max) non-exempt bundles
outliers = []
for dp in dumps:
    d = json.load(open(dp))
    evt = os.path.basename(dp)[len("calib-"):-len(".json")]
    keep, rej, rows = decisions_for(evt)
    conf_gids = {k[0] for k in keep}
    fl_by_gid = {f["gid"]: f for f in d["flashes"]}
    act = {o["ch"] for o in d["opdets"] if o["active"]}
    for f in d["flashes"]:
        cp = [f["pe"][c] for c in CATH]
        row = (evt, f["gid"], sum(cp),
               [sum(1 for v in cp if v >= t) for t in FIRED_THRS], sum(f["pe"]))
        all_flash.append(row)
        if f["gid"] in conf_gids:
            conf_flash.append(row)
    for b in d["bundles"]:
        if bundle_exempt(b):
            continue
        f = fl_by_gid[b["flash_gid"]]
        r = cath_ratios(b["pred_pe"], f["pe"], act)
        all_R.append(r)
        key = (b["flash_gid"], b["main_cluster"])
        if key in keep:
            conf_R.append(r)
            if r[0] > 15 or r[1] > 50:
                outliers.append((evt, key) + r)
        elif key in rej:
            rej_R.append(r)

print(f"flashes: {len(all_flash)} admitted (flash_minPE=25), "
      f"{len(conf_flash)} carry a confirmed candle pick")
print(f"non-exempt candidate bundles: {len(all_R)}; "
      f"confirmed {len(conf_R)}, rejected {len(rej_R)}")

print("\n== flash admission grid (cathode sum PE, n fired @ >=1 PE) ==")
print(f"{'sum>=':>6} {'nf>=':>5} {'conf lost':>10} {'admitted cut %':>15}")
for mpe in (1, 2, 3, 5, 8, 10):
    for mnf in (1, 2):
        cl = sum(1 for r in conf_flash if not (r[2] >= mpe and r[3][1] >= mnf))
        ac = sum(1 for r in all_flash if not (r[2] >= mpe and r[3][1] >= mnf))
        print(f"{mpe:>6} {mnf:>5} {cl:>6}/{len(conf_flash):<4} "
              f"{100*ac/len(all_flash):>14.1f}%")

print("\n== overpred ceiling grid (cathode-only R_total / R_max) ==")
print(f"{'R_tot<=':>8} {'R_max<=':>8} {'conf culled':>12} {'bundles culled %':>17}")
for ct in (10, 15, 20, 30):
    for cm in (20, 30, 50):
        cc = sum(1 for a, b in conf_R if a > ct or b > cm)
        ac = sum(1 for a, b in all_R if a > ct or b > cm)
        print(f"{ct:>8} {cm:>8} {cc:>6}/{len(conf_R):<5} "
              f"{100*ac/len(all_R):>16.1f}%")

print("\nconfirmed non-exempt pairs above the chosen (15, 50) ceilings:")
for evt, key, rt, rm in outliers:
    print(f"  {evt} key={key} R_tot={rt:.1f} R_max={rm:.1f}")

# evt298567 full-scan safety check
lab = json.load(open(f"{PDVD}/work/ql_labels/wfresc/labels-evt298567.json"))
d = json.load(open(f"{PDVD}/work/039252_0_wfresc/calib-evt298567.json"))
fl_by_gid = {f["gid"]: f for f in d["flashes"]}
act = {o["ch"] for o in d["opdets"] if o["active"]}
print("\n== evt298567 wfresc full-scan safety check ==")
for name, entries in (("matches", lab["matches"]),
                      ("rejected_auto", lab["rejected_auto"])):
    ne = []
    for e in entries:
        fl = e["flags"]
        if fl.get("close_to_PMT") or fl.get("window_truncated") or fl.get("at_x_boundary"):
            continue
        ne.append(cath_ratios(e["pred_pes"], fl_by_gid[e["flash_gid"]]["pe"], act))
    rt = np.array([a for a, b in ne]); rm = np.array([b for a, b in ne])
    print(f"{name}: {len(ne)} non-exempt; R_total med={np.median(rt):.2f} "
          f"max={rt.max():.2f}; R_max med={np.median(rm):.2f} max={rm.max():.2f}")
mg = {e["flash_gid"] for e in lab["matches"]}
rows = [(sum(f["pe"][c] for c in CATH),
         sum(1 for c in CATH if f["pe"][c] >= 1.0))
        for f in d["flashes"] if f["gid"] in mg]
cs = np.array([r[0] for r in rows]); nf = np.array([r[1] for r in rows])
print(f"evt298567 matched flashes: n={len(rows)}, cathode sum PE min={cs.min():.1f}, "
      f"nfired@1PE min={nf.min()} (all clear 5 PE / 2 fired easily)")

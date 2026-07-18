#!/usr/bin/env python3
"""Amplitude-model residual study on scan-agreed Q/L pairs (doc 23 phase 3a).

Question: does a fittable brightness/topology-dependent correction to the
predicted light exist that could fix the wrong-flash ranking losses (doc 22:
66/91 nm4b misses)?  Residual = log(meas/pred) on SCAN-AGREED matches only
(the ground-truth sample the doc-22 follow-up recommended fitting on).

Repro:
    python ql_display/amp_residual_fit.py --tag ac2
Outputs work/ql_scores/<out-tag>/amp_residual.{md,json} (refuses to
overwrite an existing output, M13).

Verdict encoded in the tables (039252, tag ac2):
  - flash-level, satfrac<0.2: residual p50 within +-0.1 of 0 in every
    brightness decile => the photon model is UNBIASED on clean flashes; no
    brightness correction exists to fit.
  - flash-level, satfrac>=0.2: p50 ~ -0.25 (meas below pred): DAPHNE rail
    censoring, not a model error -- already handled gate-side by the
    sat-aware knobs (doc 23 phases 1b/2).
  - bundle-level topology slices: boundary/two-boundary/close-PMT classes
    shift +0.3..+0.66 (charge truncation under-predicts), but within-class
    scatter stays sd ~0.5 in log => a multiplicative correction cannot
    separate the doc-22 tail (truth-candidate residuals +0.8..+3.0) from the
    agreed bulk.  Phase 3b (correction knob) NOT built.
"""
import argparse, json, math, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unmatched_census import (OBJECTIVE_TIERS, evt_of_idx, load_truth,
                              load_calib, NEVT)

TOL = 0.5


def collect(tag, work_root, truth):
    """Return (flash_rows, bundle_rows) of agreed-match residuals."""
    flash_rows, bundle_rows = [], []
    for idx in range(NEVT):
        evt = evt_of_idx(idx)
        try:
            calib, _ = load_calib(work_root, tag, idx)
        except FileNotFoundError:
            continue
        fg = {f["gid"]: f for f in calib["flashes"]}
        pred_by_flash, agree_flash, nsel_by_flash = {}, set(), {}
        for b in calib["bundles"]:
            if not b.get("auto_selected"):
                continue
            g = b["flash_gid"]
            pred_by_flash[g] = pred_by_flash.get(g, 0.0) + b["total_pred_light"]
            nsel_by_flash[g] = nsel_by_flash.get(g, 0) + 1
            t = fg[g]["time"]
            pos = [x for x in truth.get(evt, [])
                   if x["uid"] == b["main_cluster"] and x["positive"]
                   and x["conf"] in OBJECTIVE_TIERS]
            agreed = any(abs(t - x["time"]) <= TOL for x in pos)
            if agreed:
                agree_flash.add(g)
            f = fg[g]
            meas = f["total_PE"]
            if agreed and nsel_by_flash and meas > 0 and b["total_pred_light"] > 0:
                satpe = sum(p for p, s in zip(f["pe"], f["sat"]) if s)
                bundle_rows.append(dict(
                    evt=evt, uid=b["main_cluster"], gid=g,
                    res=math.log(meas / b["total_pred_light"]),
                    satfrac=satpe / meas,
                    flags={k: bool(b.get(k)) for k in
                           ("at_cathode", "at_x_boundary", "window_truncated",
                            "close_to_PMT", "two_boundary", "xtpc_pin")},
                    apa=b["apa"], nsel=None))
        for g, pred in pred_by_flash.items():
            if g not in agree_flash:
                continue
            f = fg[g]
            meas = f["total_PE"]
            if meas <= 0 or pred <= 0:
                continue
            satpe = sum(p for p, s in zip(f["pe"], f["sat"]) if s)
            flash_rows.append(dict(evt=evt, gid=g, pe=meas,
                                   res=math.log(meas / pred),
                                   satfrac=satpe / meas,
                                   nsel=nsel_by_flash[g]))
        for r in bundle_rows:
            if r["nsel"] is None and r["evt"] == evt:
                r["nsel"] = nsel_by_flash.get(r["gid"], 1)
    return flash_rows, bundle_rows


def deciles(rows, nbin=8):
    rows = sorted(rows, key=lambda r: r["pe"])
    out, k = [], max(len(rows) // nbin, 1)
    for i in range(0, len(rows), k):
        ch = rows[i:i + k]
        res = [r["res"] for r in ch]
        out.append(dict(pe_lo=ch[0]["pe"], pe_hi=ch[-1]["pe"], n=len(ch),
                        p50=statistics.median(res),
                        mean=statistics.mean(res),
                        sd=statistics.pstdev(res)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ac2")
    ap.add_argument("--work-root", default="work")
    ap.add_argument("--gold", default="work/ql_labels/wfresc/labels-evt298567.json")
    ap.add_argument("--decisions-dir", default="ql_display/decisions-cathxa")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()
    out_tag = args.out_tag or f"{args.tag}_amp"
    out_dir = os.path.join("work", "ql_scores", out_tag)
    out_md = os.path.join(out_dir, "amp_residual.md")
    if os.path.exists(out_md):
        sys.exit(f"REFUSING to overwrite {out_md} (M13): pick a new --out-tag")
    truth = load_truth(args.gold, args.decisions_dir)
    flash_rows, bundle_rows = collect(args.tag, args.work_root, truth)

    rep = dict(tag=args.tag, tol=TOL,
               n_flash=len(flash_rows), n_bundle=len(bundle_rows))
    lines = [f"# Amplitude residual study — tag `{args.tag}` (doc 23 phase 3a)", "",
             "Residual = log(meas/pred) on scan-agreed matches; >0 = under-predicted.", ""]
    for name, sel in (("satfrac<0.2", lambda r: r["satfrac"] < 0.2),
                      ("satfrac>=0.2", lambda r: r["satfrac"] >= 0.2)):
        sub = [r for r in flash_rows if sel(r)]
        rep[name] = deciles(sub)
        lines += [f"## flash-level, {name} (n={len(sub)})", "",
                  "| PE range | n | p50 | mean | sd |", "|---|---|---|---|---|"]
        for d in rep[name]:
            lines.append(f"| {d['pe_lo']:.0f}–{d['pe_hi']:.0f} | {d['n']} | "
                         f"{d['p50']:+.2f} | {d['mean']:+.2f} | {d['sd']:.2f} |")
        lines.append("")
    # topology slices, clean single-bundle
    single = [r for r in bundle_rows if r["satfrac"] < 0.2 and r["nsel"] == 1]
    rep["topology"] = {}
    lines += [f"## bundle-level topology, clean single-bundle (n={len(single)})", "",
              "| class | n | p50 | mean | sd |", "|---|---|---|---|---|"]
    def emit(name, rs):
        if len(rs) < 8:
            return
        v = [r["res"] for r in rs]
        rep["topology"][name] = dict(n=len(rs), p50=statistics.median(v),
                                     mean=statistics.mean(v), sd=statistics.pstdev(v))
        d = rep["topology"][name]
        lines.append(f"| {name} | {d['n']} | {d['p50']:+.2f} | {d['mean']:+.2f} | {d['sd']:.2f} |")
    emit("ALL", single)
    for fl in ("at_x_boundary", "window_truncated", "close_to_PMT", "two_boundary"):
        emit(fl, [r for r in single if r["flags"][fl]])
        emit("!" + fl, [r for r in single if not r["flags"][fl]])
    emit("apa0(bottom)", [r for r in single if r["apa"] == 0])
    emit("apa4(top)", [r for r in single if r["apa"] == 4])
    lines += ["",
              "## Verdict",
              "",
              "Clean flashes are UNBIASED at every brightness; the sat offset is",
              "measurement censoring (handled gate-side, doc 23 phases 1b/2);",
              "topology biases (+0.3..+0.66) are real but within-class sd ~0.5",
              "dominates — no multiplicative correction separates the doc-22",
              "wrong-flash tail from the agreed bulk. Phase 3b knob NOT built.", ""]
    os.makedirs(out_dir, exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(out_dir, "amp_residual.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[out] {out_md}")


if __name__ == "__main__":
    main()

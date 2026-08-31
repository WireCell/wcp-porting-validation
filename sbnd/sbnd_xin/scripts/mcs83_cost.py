#!/usr/bin/env python3
"""doc 83 Part 1 -- how fast is MCS, and is it negligible against the PR stage?

Three independent numbers, all from data already on disk plus one clean
micro-benchmark:

  (a) in-situ bracket: every ON-arm log line carries an [HH:MM:SS.mmm]
      timestamp, and the MCS sentinel is emitted right after the
      "TaggerCheckNeutrino timing: fill_kine_tree took" perf line with only a
      trivial field-stamp block between -- the delta brackets the whole call
      (upper bound: includes that stamp block + spdlog formatting; cannot see
      early-exit calls, which are cheaper still).
  (b) a clean micro-benchmark: mcs_probe bench, replaying REAL harvested
      per-muon clouds (the actual muon_segments-mode point counts, not the
      round-0 whole-cluster fixtures) N times each, split into
      trim_trajectory / form_segs / estimate_energy.
  (c) the noise floor: OFFB vs REF are two independently-built knob-OFF
      binaries already gate-PROVEN to produce byte-identical output (M2) --
      so their whole per-event work is provably identical, and their
      per-event wall-time spread IS the measurement noise floor a same-arm
      comparison lives in.  The ON-vs-OFFB paired delta is reported against
      that floor, not as a signed cost.

Usage: mcs83_cost.py --out DIR --on ARM_ON --offb ARM_OFFB --ref ARM_REF
                     [--bench-clouds DIR_OF_TXT_CLOUDS]
Writes into DIR (must not pre-exist unless --out-exist-ok, M13).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "..", "mcs_upstream", "dumper", "mcs_probe")

TS = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\.(\d\d\d)\]")
FILL_KINE = re.compile(r"fill_kine_tree took")
SENT_OK = re.compile(
    r"mcs: source=(?P<source>\S+) nseg=(?P<nseg>\d+) npoints=(?P<npoints>\d+) "
    r"len=(?P<len>[\d.]+)cm.*\(nsegs14=(?P<nsegs14>\d+)")
SENT_ANY = re.compile(r"\bmcs:")
SENT_SHORT = re.compile(r"mcs: selected muon too short")
MABC_LINE = re.compile(r"MABC timing: (\S+) took ([\d.]+) ms \(cumulative ([\d.]+) ms\)")
TIMER_TOTAL = re.compile(r"Timer: Total ([\d.]+) wall-sec, ([\d.]+) core-sec")


def ts_ms(m):
    h, mi, s, ms = (int(x) for x in m.groups())
    return ((h * 60 + mi) * 60 + s) * 1000 + ms


def scan_on_arm(arm):
    brackets = []   # ms per completed invocation
    npoints_of = []
    nsegs14_of = []
    mabc_cum = []          # per event, PR-stage ms (ALL events)
    mabc_cum_active = []   # per event, PR-stage ms (events with >=1 completed MCS call)
    n_ok = n_short = n_any = 0
    n_events = 0
    for prdir in sorted(glob.glob(os.path.join(arm, "pr_evt*"))):
        logs = glob.glob(os.path.join(prdir, "wct_pr_evt*.log"))
        if not logs:
            continue
        n_events += 1
        prev_ts = None
        last_cum = None
        n_ok_this_event = 0
        with open(logs[0], errors="replace") as fh:
            for line in fh:
                m = TS.match(line)
                cur_ts = ts_ms(m) if m else None
                if FILL_KINE.search(line) and cur_ts is not None:
                    prev_ts = cur_ts
                mo = SENT_OK.search(line)
                if mo and cur_ts is not None:
                    n_ok += 1
                    n_ok_this_event += 1
                    if prev_ts is not None:
                        brackets.append(cur_ts - prev_ts)
                        npoints_of.append(int(mo.group("npoints")))
                        nsegs14_of.append(int(mo.group("nsegs14")))
                    prev_ts = None
                elif SENT_SHORT.search(line):
                    n_short += 1
                elif SENT_ANY.search(line):
                    n_any += 1
                mm = MABC_LINE.search(line)
                if mm:
                    last_cum = float(mm.group(3))
        if last_cum is not None:
            mabc_cum.append(last_cum)
            if n_ok_this_event > 0:
                mabc_cum_active.append(last_cum)
    return dict(brackets=brackets, npoints=npoints_of, nsegs14=nsegs14_of,
               mabc_cum=mabc_cum, mabc_cum_active=mabc_cum_active,
               n_ok=n_ok, n_short=n_short, n_any_line=n_any + n_ok + n_short,
               n_events=n_events)


def scan_wall(arm):
    """Per-event wall-sec from the job epilogue Timer: Total line, keyed by
    event id, for the noise-floor comparison (needs event-matched pairing)."""
    out = {}
    for prdir in sorted(glob.glob(os.path.join(arm, "pr_evt*"))):
        evt = int(os.path.basename(prdir).rsplit("pr_evt", 1)[1])
        logs = glob.glob(os.path.join(prdir, "wct_pr_evt*.log"))
        if not logs:
            continue
        wall = core = None
        with open(logs[0], errors="replace") as fh:
            for line in fh:
                m = TIMER_TOTAL.search(line)
                if m:
                    wall, core = float(m.group(1)), float(m.group(2))
        if wall is not None:
            out[evt] = (wall, core)
    return out


def pctl(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-exist-ok", action="store_true")
    ap.add_argument("--on", required=True)
    ap.add_argument("--offb", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--bench-clouds", default=None,
                    help="dir of *.txt clouds (mcs_probe format) to micro-benchmark; "
                         "default: reuse docs/83_mcs/clouds if present")
    ap.add_argument("--bench-reps", type=int, default=300)
    args = ap.parse_args()
    if os.path.exists(args.out) and not args.out_exist_ok:
        sys.exit(f"refusing to write into existing {args.out} (M13); use a fresh dir")
    os.makedirs(args.out, exist_ok=True)

    summary = []

    # ---- (a) in-situ bracket ----
    on = scan_on_arm(args.on)
    br = np.array(on["brackets"], dtype=float)
    summary.append(f"(a) in-situ bracket: N_completed={on['n_ok']} N_too_short={on['n_short']} "
                   f"N_events_with_any_mcs_line={on['n_any_line']} / N_events={on['n_events']}")
    if len(br):
        summary.append("    per-invocation ms: p0=%.1f p25=%.1f p50=%.1f p75=%.1f p90=%.1f "
                       "p99=%.1f max=%.1f  mean=%.2f  sum=%.1fms over %d calls"
                       % (pctl(br, 0), pctl(br, 25), pctl(br, 50), pctl(br, 75), pctl(br, 90),
                          pctl(br, 99), br.max(), br.mean(), br.sum(), len(br)))
    mabc = np.array(on["mabc_cum"], dtype=float)
    mabc_active = np.array(on["mabc_cum_active"], dtype=float)
    if len(mabc):
        per_event_ms = br.sum() / max(on["n_events"], 1)
        summary.append("    PR-stage (MABC cumulative) median over ALL %d events = %.0f ms/event; "
                       "MCS amortized over ALL events = %.2f ms/event = %.3f%% of that"
                       % (on["n_events"], np.median(mabc), per_event_ms,
                          100.0 * per_event_ms / np.median(mabc)))
    if len(mabc_active):
        per_active_ms = br.sum() / max(len(mabc_active), 1)
        summary.append("    PR-stage median over the %d events with >=1 completed MCS call = %.0f ms/event "
                       "(these events do more -- BDT scoring etc -- hence bigger than the all-events "
                       "median above); MCS cost as a fraction of THAT = %.3f%%"
                       % (len(mabc_active), np.median(mabc_active),
                          100.0 * per_active_ms / np.median(mabc_active)))

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(br):
        ax.hist(br, bins=40, histtype="step", color="tab:blue")
    ax.set_xlabel("per-invocation bracket [ms]")
    ax.set_ylabel("MCS calls")
    ax.set_title(f"MCS per-call cost, in situ (N={len(br)})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "cost_hist.png"), dpi=150)

    with open(os.path.join(args.out, "cost_brackets.tsv"), "w") as fh:
        fh.write("bracket_ms\tnpoints\tnsegs14\n")
        for b, p, n in zip(on["brackets"], on["npoints"], on["nsegs14"]):
            fh.write(f"{b}\t{p}\t{n}\n")

    # ---- (b) clean micro-benchmark ----
    cloud_dir = args.bench_clouds or os.path.join(args.out, "..", "83_mcs", "clouds")
    cloud_dir = os.path.normpath(cloud_dir)
    clouds = sorted(glob.glob(os.path.join(cloud_dir, "*.txt"))) if os.path.isdir(cloud_dir) else []
    bench_rows = []
    if clouds and os.path.exists(PROBE):
        # spread across the npoints range rather than every cloud, for speed
        sizes = []
        for c in clouds:
            with open(c) as fh:
                n = sum(1 for _ in fh) - 2
            sizes.append((n, c))
        sizes.sort()
        picks = sizes[::max(1, len(sizes) // 12)][:12]
        for n, c in picks:
            out_json = os.path.join(args.out, "bench_" + os.path.basename(c).replace(".txt", ".json"))
            r = subprocess.run([PROBE, "bench", c, str(args.bench_reps), out_json],
                               capture_output=True, text=True)
            if r.returncode != 0:
                continue
            d = json.load(open(out_json))
            bench_rows.append(dict(npoints=n, **d["median_us"]))
        with open(os.path.join(args.out, "bench_summary.json"), "w") as fh:
            json.dump(bench_rows, fh, indent=1)
        if bench_rows:
            fig, ax = plt.subplots(figsize=(6, 4))
            xs = [r["npoints"] for r in bench_rows]
            for key, color in [("run", "k"), ("trim_trajectory", "tab:red"),
                               ("form_segs", "tab:green"), ("estimate_energy", "tab:blue")]:
                ax.plot(xs, [r[key] for r in bench_rows], "o-", color=color, label=key)
            ax.set_xlabel("cloud size [points]")
            ax.set_ylabel("median time [us]")
            ax.set_yscale("log")
            ax.set_title("MCS micro-benchmark, real harvested clouds")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "bench_scaling.png"), dpi=150)
            worst = max(bench_rows, key=lambda r: r["run"])
            summary.append(f"(b) micro-benchmark: {len(bench_rows)} real clouds, "
                           f"{args.bench_reps} reps each; run() median {min(r['run'] for r in bench_rows):.0f}"
                           f"-{worst['run']:.0f} us (worst at npoints={worst['npoints']}); "
                           f"split at that size: trim={worst['trim_trajectory']:.0f}us "
                           f"form_segs={worst['form_segs']:.0f}us "
                           f"estimate_energy={worst['estimate_energy']:.0f}us")
    else:
        summary.append("(b) micro-benchmark: SKIPPED (no cloud dir / probe found; "
                       "run mcs83_outliers.py first to harvest clouds, or pass --bench-clouds)")

    # ---- (c) noise floor: OFFB vs REF (byte-identical output, so identical work) ----
    wall_offb = scan_wall(args.offb)
    wall_ref = scan_wall(args.ref)
    wall_on = scan_wall(args.on)
    shared = sorted(set(wall_offb) & set(wall_ref))
    if shared:
        d = np.array([wall_offb[e][1] - wall_ref[e][1] for e in shared])  # core-sec
        base = np.array([wall_ref[e][1] for e in shared])
        summary.append(f"(c) noise floor (OFFB-REF, byte-identical output, N={len(shared)} events): "
                       f"core-sec delta median={np.median(d)*1000:.1f}ms "
                       f"({100*np.median(d/base):.2f}%) p90={pctl(np.abs(d),90)*1000:.1f}ms "
                       f"max|delta|={np.max(np.abs(d))*1000:.1f}ms")
    shared2 = sorted(set(wall_on) & set(wall_offb))
    if shared2:
        d2 = np.array([wall_on[e][1] - wall_offb[e][1] for e in shared2])
        summary.append(f"(c) paired ON-OFFB core-sec delta (N={len(shared2)} shared events): "
                       f"median={np.median(d2)*1000:+.1f}ms mean={np.mean(d2)*1000:+.1f}ms "
                       f"sd={np.std(d2)*1000:.1f}ms  -- compare to the noise floor above, "
                       f"NOT to zero: this is expected to be indistinguishable from noise")

    with open(os.path.join(args.out, "cost_summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

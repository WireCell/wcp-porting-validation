#!/usr/bin/env python3
"""Plot opflash.AbsTime() before and after applying frame_apply_at_caf.

Reads debug.csv produced by larwirecell::OpFlashSource (columns:
event,flashid,tpcid,opflash_time_us,opflash_abstime_us,frame_apply_at_caf_us).
Writes opflash_abstime_hist.pdf in the same dir as the input.
"""
import argparse, csv, os, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="path to debug.csv")
    p.add_argument("--out", default=None, help="output pdf path (default: alongside csv)")
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--range", default=None, metavar="LO,HI",
                   help="histogram x-range in us (default: data-driven, equal for both panels)")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = []
    abstime = []
    abstime_shifted = []
    with open(args.csv) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            t  = float(row["opflash_time_us"])
            at = float(row["opflash_abstime_us"])
            fc = float(row["frame_apply_at_caf_us"])
            time.append(t)
            abstime.append(at)
            abstime_shifted.append(at + fc)

    if not abstime:
        sys.exit(f"no rows in {args.csv}")

    if args.range:
        lo, hi = [float(x) for x in args.range.split(",")]
    else:
        lo = min(min(time), min(abstime), min(abstime_shifted))
        hi = max(max(time), max(abstime), max(abstime_shifted))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    axes[0].hist(time,            bins=args.bins, range=(lo, hi), color="C2", alpha=0.85)
    axes[0].set_title(f"opflash.Time()  ({len(time)} flashes)")
    axes[0].set_xlabel("time [us]")
    axes[0].set_ylabel("flashes / bin")

    axes[1].hist(abstime,         bins=args.bins, range=(lo, hi), color="C0", alpha=0.85)
    axes[1].set_title(f"opflash.AbsTime()")
    axes[1].set_xlabel("time [us]")

    axes[2].hist(abstime_shifted, bins=args.bins, range=(lo, hi), color="C1", alpha=0.85)
    axes[2].set_title(f"AbsTime() + frame_apply_at_caf")
    axes[2].set_xlabel("time [us]")

    fig.suptitle(os.path.basename(args.csv))
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)) or ".",
                                   "opflash_abstime_hist.pdf")
    fig.savefig(out)
    print(f"wrote {out}  bins={args.bins}  range=[{lo:.3f},{hi:.3f}] us")

    # --- Per (event, tpc) "closest-to-0" reduction ---------------------------
    # For each (event, tpc) pick the single flash whose AbsTime() has the
    # smallest |.|; preserve sign so the histogram shows whether the closest
    # flash typically arrives just before or just after t=0.
    closest = {}  # (event, tpc) -> (abs_abstime, abstime, fcaf_us)
    with open(args.csv) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            key = (row["event"], row["tpcid"])
            at = float(row["opflash_abstime_us"])
            fc = float(row["frame_apply_at_caf_us"])
            entry = (abs(at), at, fc)
            if key not in closest or entry[0] < closest[key][0]:
                closest[key] = entry
    closest_abstime         = [v[1]         for v in closest.values()]
    closest_abstime_shifted = [v[1] + v[2]  for v in closest.values()]

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes2[0].hist(closest_abstime, bins=args.bins, range=(lo, hi), color="C0", alpha=0.85)
    axes2[0].set_title(f"closest-to-0 AbsTime() per (event,APA)  ({len(closest_abstime)} entries)")
    axes2[0].set_xlabel("time [us]")
    axes2[0].set_ylabel("entries / bin")
    axes2[1].hist(closest_abstime_shifted, bins=args.bins, range=(lo, hi), color="C1", alpha=0.85)
    axes2[1].set_title("closest-to-0 AbsTime() + frame_apply_at_caf")
    axes2[1].set_xlabel("time [us]")
    fig2.suptitle(os.path.basename(args.csv))
    fig2.tight_layout()
    out2 = os.path.join(os.path.dirname(out) or ".", "opflash_abstime_closest_hist.pdf")
    fig2.savefig(out2)
    print(f"wrote {out2}")
    # Quick stats for the log.
    import statistics
    print(f"  Time():          mean={statistics.mean(time):.3f}  "
          f"stdev={statistics.pstdev(time):.3f}  min={min(time):.3f}  max={max(time):.3f}  us")
    print(f"  AbsTime():       mean={statistics.mean(abstime):.3f}  "
          f"stdev={statistics.pstdev(abstime):.3f}  min={min(abstime):.3f}  max={max(abstime):.3f}  us")
    print(f"  AbsTime()+fcaf:  mean={statistics.mean(abstime_shifted):.3f}  "
          f"stdev={statistics.pstdev(abstime_shifted):.3f}  min={min(abstime_shifted):.3f}  max={max(abstime_shifted):.3f}  us")
    if closest_abstime:
        print(f"  closest AbsTime():       mean={statistics.mean(closest_abstime):.3f}  "
              f"stdev={statistics.pstdev(closest_abstime):.3f}  min={min(closest_abstime):.3f}  max={max(closest_abstime):.3f}  us")
        print(f"  closest AbsTime()+fcaf:  mean={statistics.mean(closest_abstime_shifted):.3f}  "
              f"stdev={statistics.pstdev(closest_abstime_shifted):.3f}  min={min(closest_abstime_shifted):.3f}  max={max(closest_abstime_shifted):.3f}  us")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Doc 66: per-bundle tagger-verdict flip report between two diffusion arms.

Both arms run the SAME binary over the SAME Q/L pctrees (symlinked from
work-mcp1kall-d59k) with the SAME production flag set, and differ ONLY in the
TrackFitting parameter JSON's DL/DT -- so every difference reported here is
caused by the diffusion constants and nothing else.

A bundle is keyed by (event, main_id), which is the identity the nusel table
itself uses.  Bundles that exist in only one arm are reported separately: the
un-merge and the tagger tail can in principle change which mains survive, and
silently dropping those would understate the change.

Usage:
  ./d66_flip_report.py work-stmcamp-d66old work-stmcamp-d66new [--beam-only]
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TAGGERS = ("tgm", "stm", "fc", "lm")


def read_arm(root):
    """(event, main_id) -> row dict, over every nusel-evt<ID>.tsv in the arm."""
    root = root if os.path.isabs(root) else os.path.join(HERE, root)
    out = {}
    events = set()
    for d in sorted(os.listdir(root)):
        if not d.startswith("nusel_evt"):
            continue
        evt = d[len("nusel_evt"):]
        tsv = os.path.join(root, d, f"nusel-evt{evt}.tsv")
        if not os.path.exists(tsv):
            continue
        events.add(evt)
        with open(tsv) as f:
            hdr = f.readline().split()
            for line in f:
                parts = line.split()
                if len(parts) != len(hdr):
                    continue
                r = dict(zip(hdr, parts))
                out[(evt, r["main_id"])] = r
    return out, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--beam-only", action="store_true",
                    help="restrict to in_beam==1 bundles (the ones the scan judges)")
    args = ap.parse_args()

    old, ev_old = read_arm(args.old)
    new, ev_new = read_arm(args.new)

    print(f"arm OLD {args.old}: {len(ev_old)} events, {len(old)} bundles")
    print(f"arm NEW {args.new}: {len(ev_new)} events, {len(new)} bundles")

    # Only events BOTH arms produced can be compared.  Without this the
    # bundle-level "only in one arm" counts are dominated by whichever arm has
    # more events on disk (e.g. mid-run), which would read as a physics result
    # and is not one.
    ev_common = ev_old & ev_new
    if ev_old != ev_new:
        print(f"  events only in OLD: {len(ev_old - ev_new)} "
              f"{sorted(ev_old - ev_new)[:10]}{' ...' if len(ev_old - ev_new) > 10 else ''}")
        print(f"  events only in NEW: {len(ev_new - ev_old)} "
              f"{sorted(ev_new - ev_old)[:10]}{' ...' if len(ev_new - ev_old) > 10 else ''}")
        print(f"  --> comparing the {len(ev_common)} events present in BOTH arms")

    old = {k: v for k, v in old.items() if k[0] in ev_common}
    new = {k: v for k, v in new.items() if k[0] in ev_common}

    keys_both = set(old) & set(new)
    only_old = set(old) - set(new)
    only_new = set(new) - set(old)

    def keep(k, arm):
        return (not args.beam_only) or arm[k].get("in_beam") == "1"

    common = sorted(k for k in keys_both if keep(k, old) or keep(k, new))
    scope = "in-beam bundles" if args.beam_only else "all bundles"
    print(f"\n{scope}: {len(common)} present in both arms; "
          f"{len(only_old)} only-OLD, {len(only_new)} only-NEW")

    if only_old or only_new:
        print("  (bundle-set differences, first 20 of each)")
        for k in sorted(only_old)[:20]:
            print(f"    only-OLD evt{k[0]} main {k[1]}")
        for k in sorted(only_new)[:20]:
            print(f"    only-NEW evt{k[0]} main {k[1]}")

    # --- per-tagger confusion -------------------------------------------------
    print(f"\n{'tagger':<8} {'unchanged':>10} {'flipped':>8}  transitions (old->new)")
    total_flipped_keys = set()
    for t in TAGGERS:
        conf = collections.Counter()
        flips = 0
        for k in common:
            a, b = old[k].get(t, "?"), new[k].get(t, "?")
            if a == b:
                conf[(a, b)] += 1
            else:
                conf[(a, b)] += 1
                flips += 1
                total_flipped_keys.add(k)
        trans = ", ".join(f"{a}->{b}:{n}" for (a, b), n in sorted(conf.items())
                          if a != b) or "none"
        print(f"{t:<8} {len(common) - flips:>10} {flips:>8}  {trans}")

    # --- the final label ------------------------------------------------------
    lab = collections.Counter()
    lab_flips = []
    for k in common:
        a, b = old[k].get("label", "?"), new[k].get("label", "?")
        lab[(a, b)] += 1
        if a != b:
            lab_flips.append((k, a, b))
    print(f"\nlabel   {len(common) - len(lab_flips):>10} {len(lab_flips):>8}")
    for (a, b), n in sorted(lab.items()):
        if a != b:
            print(f"    {a} -> {b}: {n}")

    if lab_flips:
        print("\nevery label flip (event, main, old -> new):")
        for (k, a, b) in sorted(lab_flips):
            r = new[k]
            print(f"    evt{k[0]:<8} main {k[1]:<4} in_beam={r.get('in_beam')} "
                  f"len={r.get('len_main_cm'):>7}  {a} -> {b}")

    print(f"\nbundles with ANY tagger flip: {len(total_flipped_keys)} / {len(common)}"
          f" ({100.0 * len(total_flipped_keys) / max(1, len(common)):.2f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

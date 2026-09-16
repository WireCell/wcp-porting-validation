#!/usr/bin/env python3
"""doc pdvd/111 sec 7 (reported, not gating) -- STM tags of a knob arm against its base: the (event, cluster) pairs whose
TaggerCheckSTM verdict line reads STM=1 in each arm's wct_pr log, and how many are lost / gained.  Read-only.

Usage: d111_tag_churn.py --det pdhd --base d111hoff --arms d111hsr6,d111hsr10,d111hsr15
"""
import argparse, glob, re

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
R = re.compile(r"TaggerCheckSTM: cluster (\d+) \S+ STM=(\d) TGM=(\d)")


def tagged(det, arm):
    out = set()
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = d.split("/")[-1][:-len(arm) - 1]
        for lg in glob.glob(f"{d}/wct_pr_*.log"):
            for c, s, _t in R.findall(open(lg, errors="replace").read()):
                if s == "1":
                    out.add((ev, int(c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", required=True)
    a = ap.parse_args()
    base = tagged(a.det, a.base)
    print(f"# doc pdvd/111 STM tag churn ({a.det}): base {a.base} STM-tagged {len(base)}")
    for arm in a.arms.split(","):
        k = tagged(a.det, arm)
        print(f"{arm}: tagged {len(k)}, lost {len(base - k)} ({100*len(base - k)/max(len(base),1):.1f} %), "
              f"gained {len(k - base)} ({100*len(k - base)/max(len(base),1):.1f} %)")


if __name__ == "__main__":
    main()

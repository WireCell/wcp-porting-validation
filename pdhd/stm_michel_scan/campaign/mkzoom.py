#!/usr/bin/env python3
"""doc pdhd/18 -- write h_dqdx_zoom.png beside every g_dqdx.png in a shots dir.

    mkzoom.py SHOTSDIR [--scale 3]

g_dqdx.png ships at ~514x338, and both fit-failure modes (the overshoot's
collapse, the undershoot's hot tip) live in the last few centimetres.  Every
PDVD tranche-2 scanner cropped and upscaled it by hand, each with a private
helper (doc pdvd/55; rubric display trap 1) -- five agents x ~50 chunks of
duplicated tool calls.  This does it once, the same way for every item: the
whole panel, Lanczos-upscaled, nothing cropped (so no scale or axis label can be
lost), nothing re-rendered (so it is the same picture, only larger).

Idempotent; an existing zoom newer than its source is kept.
"""
import argparse, os, sys
from PIL import Image


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("shotsdir")
    ap.add_argument("--scale", type=float, default=3.0)
    a = ap.parse_args(argv)
    n = kept = missing = 0
    for d in sorted(os.listdir(a.shotsdir)):
        src = os.path.join(a.shotsdir, d, "g_dqdx.png")
        if not os.path.isdir(os.path.join(a.shotsdir, d)):
            continue
        if not os.path.exists(src):
            missing += 1
            continue
        dst = os.path.join(a.shotsdir, d, "h_dqdx_zoom.png")
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            kept += 1
            continue
        im = Image.open(src).convert("RGB")
        w, h = im.size
        im.resize((int(w * a.scale), int(h * a.scale)), Image.LANCZOS).save(dst + ".tmp.png")
        os.replace(dst + ".tmp.png", dst)
        n += 1
    print("mkzoom: wrote %d, kept %d, %d item dir(s) without g_dqdx.png" % (n, kept, missing))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

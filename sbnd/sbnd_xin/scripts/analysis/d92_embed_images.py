#!/usr/bin/env python3
"""doc 92 -- embed (or refresh) the figures of the production guide as base64 data URIs.

The guide is ONE self-contained HTML file: every <img> carries a data-src="NAME.png"
attribute naming the figure d92_prodguide_plots.py wrote, and its src is the base64 of
that file.  Re-running this after regenerating the PNGs refreshes every image in place;
the HTML never references a file on disk, so it can be mailed or moved alone.

Usage:
  scripts/analysis/d92_embed_images.py docs/92_production-running-and-validation-guide.html <pngdir>
Exit 1 if any data-src names a file that does not exist.
"""
import base64
import os
import re
import sys

RE_IMG = re.compile(r'<img([^>]*?)\sdata-src="([^"]+)"([^>]*)>', re.S)


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    html_path, pngdir = sys.argv[1:]
    t = open(html_path).read()
    missing, n = [], 0

    def sub(m):
        nonlocal n
        pre, name, post = m.group(1), m.group(2), m.group(3)
        p = os.path.join(pngdir, name)
        if not os.path.exists(p):
            missing.append(name); return m.group(0)
        b64 = base64.b64encode(open(p, "rb").read()).decode("ascii")
        pre = re.sub(r'\ssrc="[^"]*"', "", pre); post = re.sub(r'\ssrc="[^"]*"', "", post)
        n += 1
        return f'<img{pre} data-src="{name}" src="data:image/png;base64,{b64}"{post}>'

    t2 = RE_IMG.sub(sub, t)
    if missing:
        print("MISSING:", " ".join(missing), file=sys.stderr); sys.exit(1)
    open(html_path, "w").write(t2)
    print(f"embedded {n} images -> {html_path} ({os.path.getsize(html_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

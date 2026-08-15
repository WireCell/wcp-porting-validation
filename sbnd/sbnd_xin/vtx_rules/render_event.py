"""Render one event so an AI scanner can actually LOOK at it (doc pr/80).

An agent session has no browser, so every "see the Bragg rise", "see which way
the shower grows" step of the scan procedure is unexecutable without this.  The
PNG is written to disk and opened with the Read tool, which displays images.

Same conventions as the port-5017 display, on purpose -- an image that coloured
differently from the interactive viewer would teach the wrong reflexes:
  * three projections, X-Y / Y-Z / X-Z, positions in cm;
  * fitted track points coloured by measured dQ/dx on a FIXED 0..150000 e/cm
    ramp (MIP = 43000), never autoscaled per event;
  * points with no measurement (dx <= 0 or dQ < 0) in neutral grey, never at
    the bottom of the ramp.

  cd sbnd_xin
  python3 vtx_rules/render_event.py work-mcp1k-ma10/pr_evt59335/calib-pr-evt59335.json \\
      --out /home/xqian/tmp/evt59335.png
  python3 vtx_rules/render_event.py <dump> --out <png> --zoom 40   # around the pick
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtx_geom as G                                             # noqa: E402
import vtx_io                                                    # noqa: E402
import vtx_rules                                                 # noqa: E402

DQDX_LO, DQDX_HI = 0.0, 150000.0        # e/cm, the display's fixed ramp
PROJ = [("x", "y", "X-Y"), ("z", "y", "Z-Y"), ("z", "x", "Z-X")]


def render(dump, out, zoom=None, title="", blind=False):
    res = vtx_rules.decide(dump)
    cid = vtx_io.main_cluster_id(dump)
    # blind=True hides BOTH the reconstructed vertex and the engine's pick.
    # For a self-scan that is the whole point: the owner sees the reco star, but
    # an AI that anchors on it produces labels that only confirm the reco, which
    # is precisely the bias that makes a generated label set worse than none.

    meas = {"x": [], "y": [], "z": [], "c": []}
    grey = {"x": [], "y": [], "z": []}
    for seg in dump.get("segments", []):
        for p in G.seg_points(seg):
            tgt = meas if G.valid_dqdx(p) else grey
            for k in "xyz":
                tgt[k].append(p[k])
            if tgt is meas:
                meas["c"].append(G.dqdx(p))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    sc = None
    for ax, (ha, hb, name) in zip(axes, PROJ):
        if grey["x"]:
            ax.scatter(grey[ha], grey[hb], s=6, c="#9e9e9e", alpha=0.6,
                       linewidths=0, zorder=2)
        if meas["x"]:
            sc = ax.scatter(meas[ha], meas[hb], s=9, c=meas["c"],
                            cmap="turbo", vmin=DQDX_LO, vmax=DQDX_HI,
                            linewidths=0, zorder=3)

        # Candidate vertices of the main cluster, numbered by vertex id so the
        # picture can be talked about in the same language as the tables.
        for v in dump.get("vertices", []):
            p = vtx_io.vertex_xyz(v)
            if p is None:
                continue
            d = dict(zip("xyz", p))
            main = v.get("cluster_id") == cid
            ax.plot(d[ha], d[hb], marker="o", ms=6 if main else 3,
                    mfc="none", mec="#111111" if main else "#999999",
                    mew=1.1, zorder=4)
            if main:
                ax.annotate(str(v["id"]), (d[ha], d[hb]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points",
                            color="#111111", zorder=6)

        mv = None if blind else vtx_io.xyz(dump.get("main_vertex"))
        if mv:
            d = dict(zip("xyz", mv))
            ax.plot(d[ha], d[hb], marker="*", ms=17, mfc="#e377c2",
                    mec="#7b2d6b", mew=1.3, zorder=7, label="reco vertex")

        pick = None
        if blind:
            pass
        elif res["decision"] == "answer":
            pick = (res["x"], res["y"], res["z"])
        elif res.get("guess_x") is not None:
            pick = (res["guess_x"], res["guess_y"], res["guess_z"])
        if pick:
            d = dict(zip("xyz", pick))
            ax.plot(d[ha], d[hb], marker="D", ms=9, mfc="none",
                    mec="#2ca02c", mew=2.0, zorder=8,
                    label="rules %s" % ("pick" if res["decision"] == "answer"
                                        else "guess (abstained)"))
            if zoom:
                ax.set_xlim(d[ha] - zoom, d[ha] + zoom)
                ax.set_ylim(d[hb] - zoom, d[hb] + zoom)

        ax.set_xlabel(ha + " [cm]")
        ax.set_ylabel(hb + " [cm]")
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=7, loc="best")

    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, orientation="horizontal",
                          fraction=0.045, pad=0.12)
        cb.set_label("track-fit dQ/dx [e/cm]   (MIP 43000, 2x MIP 86000; "
                     "fixed range, not per-event)", fontsize=8)

    if blind:
        head = "%s   |   BLIND: no reco vertex, no engine pick shown" % title
    else:
        head = "%s   |   %s / %s / conf=%s" % (title, res["branch"], res["rule"],
                                               res["confidence"])
        if res.get("reco_dis") is not None:
            head += "   |   rules-vs-reco %.1f cm" % res["reco_dis"]
    fig.suptitle(head, fontsize=10)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blind", action="store_true",
                    help="hide the reco vertex and the engine pick (self-scan)")
    ap.add_argument("--zoom", type=float,
                    help="half-width in cm around the rules' pick")
    args = ap.parse_args()
    with open(args.dump) as fh:
        dump = json.load(fh)
    res = render(dump, args.out, args.zoom,
                 title=os.path.basename(args.dump), blind=args.blind)
    print("wrote %s" % args.out)
    for note in res["notes"]:
        print("   %s" % note)
    return 0


if __name__ == "__main__":
    sys.exit(main())

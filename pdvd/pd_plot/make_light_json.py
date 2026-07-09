#!/usr/bin/env python3
"""Generate the PDVD light-chain data JSONs for the WCT toolkit cfg:

  pdvd-spe-templates.json  per-DAPHNE-channel SPE templates (raw ADC,
                           kernel-trimmed to peak@sample-10, 16 ns ticks)
                           for Flash::OpDecon.  Template choice mirrors
                           spe_compare.py main(): own harvest template
                           (cathode tail-repaired) when present, else the
                           peak-normalized population mean x the channel
                           1-PE mode.
  pdvd-opdet-geom.json     40-OpDet x/y/z [mm] from the raw_waveform TTree
                           (cm x 10) for Flash::OpFlashFinder.
  pdvd-opch-map.json       DAPHNE OpChannel -> OpDet ganging map for
                           OpFlashFinder channel_map_file (XAs have two
                           channels per OpDet).

Usage: make_light_json.py [run] [--exclude ch,ch,...]
       (default run 39252; --exclude drops channels from the template
        JSON so OpDecon skips them -- none excluded by default)
Output dir: toolkit cfg/pgrapher/experiment/protodunevd/
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdvd_light as pl
import spe_build as sb
import spe_compare as sc

# pdvd/ is a symlink into the wcp-porting-img repo, so resolve the toolkit
# cfg dir explicitly (override with PDVD_CFG_DIR if the layout differs).
CFG_DIR = os.environ.get(
    "PDVD_CFG_DIR",
    "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/protodunevd")


def final_templates(harvest):
    """Per-channel final template, exactly as spe_compare.main()."""
    with_tmpl = {ch: h for ch, h in harvest.items() if "tmpl" in h}
    popavg = {}
    for pop, lohi in (("cathode", 1000), ("membrane", 2000), ("pmt", 3000)):
        shapes = [h["tmpl"] / max(h["tmpl"].max(), 1e-9)
                  for ch, h in with_tmpl.items() if lohi <= ch < lohi + 1000]
        if shapes:
            popavg[pop] = np.mean(shapes, axis=0)
    out = {}
    for ch, h in sorted(harvest.items()):
        if "tmpl" in h:
            out[ch] = (np.asarray(h["tmpl"], dtype=float), False)
        elif "mode" in h:
            pop = ("cathode" if ch < 2000 else
                   "membrane" if ch < 3000 else "pmt")
            out[ch] = (popavg[pop] * float(h["mode"]), True)
    return out


def main():
    args = [a for a in sys.argv[1:]]
    exclude = set()
    if "--exclude" in args:
        i = args.index("--exclude")
        exclude = {int(c) for c in args[i + 1].split(",")}
        del args[i:i + 2]
    run = int(args[0]) if args else 39252

    harvest = sc.load_harvest(run)
    src = [p for p in pl.data_files() if f"run{run:06d}" in p][0]
    fname = os.path.basename(src)

    # ---------------- SPE templates
    tmpls = final_templates(harvest)
    templates, channels, tindex = [], [], []
    for ch, (tmpl, fallback) in tmpls.items():
        if ch in exclude:
            print(f"  ch{ch}: EXCLUDED")
            continue
        kern = tmpl[sc.TRIM:]          # peak at sample 10 (NP04 convention)
        templates.append({
            "name": f"SPE_PDVD_r{run:06d}_ch{ch}"
                    + ("_popavg" if fallback else ""),
            "values": [round(float(v), 4) for v in kern],
        })
        channels.append(ch)
        tindex.append(len(templates) - 1)
    spe = {
        "comment": ("PDVD single-p.e. response templates (ADC, 16 ns ticks, "
                    "peak at sample 10) and the DAPHNE OpChannel -> template "
                    "map for Flash::OpDecon.  Data-driven per-channel "
                    "templates from the run-{run} harvest (spe_build.py: 1-PE "
                    "averages; cathode tails exp-repaired, no AC undershoot); "
                    "channels named *_popavg carry the population-average "
                    "fallback scaled to the channel 1-PE mode.  See "
                    "pdvd/docs/pdvd-spe-template.md.").replace("{run}", str(run)),
        "source": fname,
        "templates": templates,
        "channels": channels,
        "template_index": tindex,
    }
    out = os.path.join(CFG_DIR, "pdvd-spe-templates.json")
    with open(out, "w") as fp:
        json.dump(spe, fp, indent=1)
    print(f"wrote {out}: {len(channels)} channels")

    # ---------------- OpDet geometry + opch map from the TTree
    f = pl.LightFile(src)
    geom = {}       # opdet -> (x,y,z) mm
    opchmap = {}    # opch -> opdet
    for evt in f.events[:1]:
        for r in f.records(evt):
            geom[int(r["opdet"])] = (10.0 * r["x"], 10.0 * r["y"], 10.0 * r["z"])
            opchmap[int(r["opch"])] = int(r["opdet"])

    jgeom = {
        "comment": ("PDVD optical detector positions [mm] from the "
                    "raw_waveform TTree (x/y/z branches, cm).  40 OpDets; "
                    "dead OpDets (24/27/28/34) absent from the data are "
                    "omitted.  Cathode XA OpDets 4-11 (double-sided, x~0), "
                    "membrane XA 0-3/12/13/18/19, PMTs the rest."),
        "source": fname,
        "opdets": [
            {"opdet": od, "x": round(x, 3), "y": round(y, 3), "z": round(z, 3)}
            for od, (x, y, z) in sorted(geom.items())
        ],
    }
    out = os.path.join(CFG_DIR, "pdvd-opdet-geom.json")
    with open(out, "w") as fp:
        json.dump(jgeom, fp, indent=1)
    print(f"wrote {out}: {len(geom)} opdets")

    jmap = {
        "comment": ("DAPHNE offline OpChannel -> OpDet ganging map for "
                    "Flash::OpFlashFinder channel_map_file.  X-ARAPUCAs "
                    "(10xx cathode, 20xx membrane) have two channels per "
                    "OpDet; PMTs (30xx) one."),
        "source": fname,
        "channels": [{"opch": ch, "opdet": od}
                     for ch, od in sorted(opchmap.items())],
    }
    out = os.path.join(CFG_DIR, "pdvd-opch-map.json")
    with open(out, "w") as fp:
        json.dump(jmap, fp, indent=1)
    print(f"wrote {out}: {len(opchmap)} channels")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pdhd/22 sec 5.1 -- trace one candidate's verdict quantities across arms.

    python3 d22_item_trace.py <event>/<cluster> [...]      (default: this round's two FPs)

Why: 028084_3/72 appeared BOTH as h22c's unpredicted interaction FP and as a PREDICTED
P1 FP.  Two different mechanisms cannot both be the whole story, and sec 5's hold on the
wide anchor rested on attributing it to the interaction.  This reconciles them from the
primary source: the knob-only branches (bragg_wide_fired, topology_cleared_bits) say
WHICH path cleared the candidate in each arm.
"""
import uproot, glob, os, sys
BITS = ["no_chain","stop_unmatched","no_bragg","shape_flat","not_muon_pid","continuation",
        "stop_near_boundary","vertex_hadron","short","profile_sparse","plateau_off_mip",
        "stop_into_dead","cluster_not_track","profile_geometry"]
names = lambda rb: "|".join(n for i,n in enumerate(BITS) if int(rb)>>i&1) or "-STM-"
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
WANT = set(sys.argv[1:]) or {"028084_3/72", "029107_10/44"}
WANT = {tuple(w.split("/")) for w in WANT}
COLS = ["cluster_id","is_stm","reject_bits","contrast","ks_mu","ks_flat",
        "bragg_wide_fired","bragg_wide_shift_cm","topology_cleared_bits"]
for tag in ["h22base","h22w","h22g","h22c","h22p1"]:
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{tag}/tracking-pr.root")):
        evt = os.path.basename(os.path.dirname(f)).replace("_"+tag,"")
        if evt not in {k[0] for k in WANT}: continue
        t = uproot.open(f)["T_stm_michel"]; keys = set(t.keys())
        cols = [c for c in COLS if c in keys]; a = t.arrays(cols, library="np")
        for i,c in enumerate(a["cluster_id"]):
            if (evt,str(int(c))) not in WANT: continue
            d = {k:a[k][i] for k in cols}
            ex = ""
            if "bragg_wide_fired" in d:
                ex += f" wide_fired={int(d['bragg_wide_fired'])} shift={d.get('bragg_wide_shift_cm',0):.2f}"
            if "topology_cleared_bits" in d:
                ex += f" topo_cleared={names(d['topology_cleared_bits']) if d['topology_cleared_bits'] else 'none'}"
            print(f"{evt}/{int(c):<4} {tag:8s} is_stm={int(d['is_stm'])} "
                  f"bits={names(d['reject_bits']):34s} contrast={d.get('contrast',0):6.3f} "
                  f"ks_margin={d.get('ks_mu',0)-d.get('ks_flat',0):+.4f}{ex}")
    print()

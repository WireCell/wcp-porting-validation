#!/usr/bin/env python3
"""doc pr/141 session 2, item 4 -- the mu-typed PID scan set, with a
PRE-REGISTERED predictor.

READ-ONLY.  Reads work-pr140r2-off-* (the shipped production arm, 239 events)
and writes only its --tsv / --manifest.

WHY.  doc pr/141 sec 9.5 recorded one specimen: 122660's shower 54071, an
owner-called ELECTRON typed `pdg = 13`.  A mu-typed object's `kine_charge` is
priced under the TRACK hypothesis, so if it is really EM its energy enters
`kine_reco_Enu` low by the exact global factor

    r = (recom * fudge) / (shower_recom * shower_fudge)
      = (0.87 * 0.95) / (0.58 * 0.86) = 1.657          [SBND production]

i.e. the missing energy is (r - 1)/r = 39.7% of the true shower energy, or
0.657 * kine_charge added on top of what is stored.  That is the impact metric
this script ranks by.

THE PREDICTOR, pre-registered before the scan (sec 16.2).  On the six objects
the owner has already typed by hand (3 GAMMA / 3 TRACK, sec 9.5) two candidate
screens were tried:

    kine_charge / kine_range   GAMMA 0.85 1.55 0.53 | TRACK 0.60 0.29 0.66
                               -> OVERLAPS, rejected
    mean segment length L/nseg GAMMA 10.7 5.2 35.1  | TRACK 40.3 78.2 42.4
                               -> separates at ~38 cm/seg

so the prediction is **mean segment length < 40 cm => EM (gamma/electron)**.
n = 6 sets it, which is far too few to ship anything; the point of the scan is
to test it on 14 unseen objects, not to trust it.  A muon is one long segment;
a shower branches, so its segments are short.  Length ALONE does not work --
the highest-energy mu-typed objects are 300-500 cm cosmics with 3-6 kink
segments, which is why the density and not the count is the screen.

The served set is 18 objects ranked by kine_charge DESCENDING and is BLIND:
the predicted class is not printed on the sheet or in the manifest
(feedback_blind_the_scan_sheet).  Four of the 18 are predicted-TRACK controls.

    python3 scripts/pr141_pidset.py --tsv docs/pr/pr141-pidset.tsv \
        --manifest em_display/em141-pidmu18-manifest.tsv
"""
import argparse, glob, json, os

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["work-pr140r2-off-mcp1k", "work-pr140r2-off-mcp2k",
        "work-pr140r2-off-ncpi0", "work-pr140r2-off-nuecc48"]
HYP = 1.657          # shower-hypothesis / track-hypothesis energy, exact global
E_FLOOR = 50.0       # MeV, track hypothesis -- below this the Enu error is < 33 MeV
CPS_CUT = 40.0       # cm per segment; < CPS_CUT => predicted EM
# sec 9.5's six, already typed by the owner -- excluded from the served set.
DONE = {(283713, 23011): "GAMMA", (350354, 18009): "GAMMA",
        (122660, 54071): "GAMMA", (392901, 23017): "TRACK",
        (280159, 90098): "TRACK", (294174, 25030): "TRACK"}
N_CTRL = 4           # predicted-TRACK controls, highest kine_charge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    rows, nev = [], 0
    for arm in ARMS:
        samp = arm.rsplit("-", 1)[-1]
        for f in sorted(glob.glob(os.path.join(SX, arm, "pr_evt*", "calib-pr-evt*.json"))):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            nev += 1
            m = d.get("meta") or {}
            ev = int(m.get("eventNo") or 0)
            for s in (d.get("showers") or ()):
                if abs(int(s.get("particle_id") or 0)) != 13:
                    continue
                kc = float(s.get("kine_charge") or 0.0)
                if kc <= E_FLOOR:
                    continue
                nseg = int(s.get("num_segments") or 0)
                L = float(s.get("total_length") or 0.0)
                cps = L / nseg if nseg > 0 else 1e9
                rows.append(dict(
                    sample=samp, run=int(m.get("runNo") or 0),
                    subrun=int(m.get("subRunNo") or 0), event=ev,
                    obj=int(s["id"]), kine_charge=round(kc, 1),
                    dE_if_EM=round((HYP - 1.0) * kc, 1),
                    kine_range=round(float(s.get("kine_range") or 0.0), 1),
                    nseg=nseg, length=round(L, 1), cm_per_seg=round(cps, 1),
                    conn=int(s.get("start_connection_type") or -1),
                    predicted="EM" if cps < CPS_CUT else "TRACK",
                    already_typed=DONE.get((ev, int(s["id"])), ""),
                    dump=os.path.relpath(f, SX)))
    rows.sort(key=lambda r: -r["kine_charge"])

    fresh = [r for r in rows if not r["already_typed"]]
    em = [r for r in fresh if r["predicted"] == "EM"]
    tr = [r for r in fresh if r["predicted"] == "TRACK"][:N_CTRL]
    served = sorted(em + tr, key=lambda r: -r["kine_charge"])

    print("events read                      : %d" % nev)
    print("mu-typed objects > %.0f MeV       : %d" % (E_FLOOR, len(rows)))
    print("  already typed by hand (sec 9.5): %d" % sum(1 for r in rows if r["already_typed"]))
    print("  predicted EM   (< %.0f cm/seg)  : %d" % (CPS_CUT, sum(1 for r in rows if r["predicted"] == "EM")))
    print("  predicted TRACK                : %d" % sum(1 for r in rows if r["predicted"] == "TRACK"))
    print("SERVED: %d objects (%d predicted-EM + %d controls)" % (len(served), len(em), len(tr)))
    print("energy at stake if every predicted-EM object is EM: %.0f MeV over %d objects"
          % (sum(0.657 * r["kine_charge"] for r in em), len(em)))
    print()
    hdr = ["sample", "event", "obj", "kine_charge", "kine_range", "nseg",
           "length", "cm_per_seg", "conn", "predicted"]
    print("\t".join(hdr))
    for r in served:
        print("\t".join(str(r[h]) for h in hdr))

    if args.tsv:
        cols = hdr + ["run", "subrun", "already_typed", "dump"]
        with open(os.path.join(SX, args.tsv), "w") as fh:
            fh.write("# doc pr/141 sec 16 -- mu-typed PID set, arm work-pr140r2-off-*\n")
            fh.write("# predictor pre-registered: cm_per_seg < %.0f => EM.  n=6 basis, sec 16.2\n" % CPS_CUT)
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
        print("\nwrote %s (%d rows, ALL mu-typed objects)" % (args.tsv, len(rows)))
    if args.manifest:
        # BLIND: no predicted class, no ranking hint beyond energy order.
        with open(os.path.join(SX, args.manifest), "w") as fh:
            fh.write("sample\trun\tsubrun\tevent\tdump\n")
            for r in served:
                fh.write("%s\t%d\t%d\t%d\t%s\n" % (r["sample"], r["run"], r["subrun"],
                                                   r["event"], r["dump"]))
        print("wrote %s (%d events)" % (args.manifest, len(served)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc pr/94 Phase 5b -- enumerate and characterise what per-bundle mode ADDS.

Two disjoint categories, because they need different owner judgements:

  NEW      the OFF arm reconstructed no vertex anywhere in the event and the
           ON arm reconstructed one.  These are the "newly identified
           neutrinos" -- the events a cut on the legacy output would have
           thrown away entirely.  Scan these first.

  EXTRA    the OFF arm already had a vertex and the ON arm added a SECOND
           reconstructed row (a second in-beam bundle).  The legacy primary
           row is unchanged (pr94_primary_gate.py proves that separately), so
           the question here is only whether the added row is a real second
           activity or a fragment.

For each row we print what the owner needs to judge it without opening a
file: the selected cluster, its length, the reconstructed Enu and the two BDT
scores, plus the per-activity cosmic-flag block that motivated doc pr/94 --
`act_*` shows WHY the legacy chain stayed silent (which sibling was
TGM/STM/LM-tagged, and whether that sibling was a main or a demoted main;
only NON-demoted tagged mains convict a bundle in the legacy chain, see
TaggerCheckNeutrino.cxx:1002).

`nu_x/y/z == (0,0,0)` means the chain selected a candidate but reconstructed
no vertex for it -- counted as "no vertex", never as a gain.

Usage: pr94_gained.py <off_arm> <on_arm> [--sample NAME] [--ids-only]
"""
import argparse
import os
import re
import sys

import uproot

BR = ["cluster_id", "matched_flash_gid", "nu_x", "nu_y", "nu_z",
      "numu_score", "nue_score"]
ACT = ["act_cluster_id", "act_length_cm", "act_is_selected", "act_is_demoted",
       "act_tgm", "act_stm", "act_fc", "act_lm", "act_evaluated"]


def has_vtx(a, i):
    return (float(a["nu_x"][i]), float(a["nu_y"][i]),
            float(a["nu_z"][i])) != (0.0, 0.0, 0.0)


def read(arm, evt):
    """-> (rows, present).  rows = list of dicts; present=False if no T_tagger."""
    p = os.path.join(arm, "pr_evt%d" % evt, "tracking-pr.root")
    if not os.path.exists(p):
        return [], False
    with uproot.open(p) as f:
        keys = set(k.split(";")[0] for k in f.keys())
        if "T_tagger" not in keys:
            return [], False
        t = f["T_tagger"]
        have = set(t.keys())
        want = [b for b in BR if b in have] + [b for b in ACT if b in have]
        a = t.arrays(want, library="np")
        enu = (f["T_kine"].arrays(["kine_reco_Enu"], library="np")["kine_reco_Enu"]
               if "T_kine" in keys else None)
        out = []
        for i in range(len(a[want[0]])):
            r = {b: a[b][i] for b in want}
            r["vtx"] = has_vtx(a, i)
            r["Enu"] = float(enu[i]) if enu is not None and i < len(enu) else float("nan")
            out.append(r)
        return out, True


def events(arm):
    return set(int(m.group(1)) for m in
               (re.match(r"pr_evt(\d+)$", d) for d in os.listdir(arm)) if m)


def fmt_acts(r):
    if "act_cluster_id" not in r:
        return "    (no act_* block -- OFF-arm schema)"
    lines = []
    for j in range(len(r["act_cluster_id"])):
        tags = []
        if r["act_tgm"][j]:
            tags.append("TGM")
        if r["act_stm"][j]:
            tags.append("STM")
        if r["act_fc"][j]:
            tags.append("FC")
        if r["act_lm"][j] > 0:
            tags.append("LM%d" % r["act_lm"][j])
        lines.append("    act cid %4d  L %7.1f cm  %-8s %-9s %s"
                     % (r["act_cluster_id"][j], r["act_length_cm"][j],
                        "SELECTED" if r["act_is_selected"][j] else "",
                        "demoted" if r["act_is_demoted"][j] else "main",
                        "+".join(tags) if tags else "-"))
    return "\n".join(lines)


def fmt_row(r):
    return ("  cluster %-4d gid %-8d Enu %8.1f MeV  numu %7.3f  nue %7.3f  vtx (%.1f, %.1f, %.1f)"
            % (r.get("cluster_id", -1), r.get("matched_flash_gid", -1), r["Enu"],
               r.get("numu_score", float("nan")), r.get("nue_score", float("nan")),
               r["nu_x"], r["nu_y"], r["nu_z"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off_arm")
    ap.add_argument("on_arm")
    ap.add_argument("--sample", default=None)
    ap.add_argument("--ids-only", action="store_true",
                    help="print just the NEW event ids, space separated "
                         "(feed straight to make_pr_bee.py)")
    args = ap.parse_args()

    ks = sorted(events(args.off_arm) & events(args.on_arm))
    new, extra = [], []
    for e in ks:
        ro, _ = read(args.off_arm, e)
        rn, _ = read(args.on_arm, e)
        no = sum(1 for r in ro if r["vtx"])
        nn = sum(1 for r in rn if r["vtx"])
        if no == 0 and nn > 0:
            new.append((e, ro, rn))
        elif nn > no:
            extra.append((e, ro, rn))

    if args.ids_only:
        print(" ".join(str(e) for e, _, _ in new))
        return 0

    name = args.sample or os.path.basename(args.on_arm)
    print("== %s : what per-bundle mode ADDS (%d events compared) ==" % (name, len(ks)))
    print("NEW   (OFF had no vertex anywhere, ON reconstructed one): %d" % len(new))
    print("EXTRA (OFF had a vertex, ON added a second one):          %d" % len(extra))
    print()
    for tag, group in (("NEW", new), ("EXTRA", extra)):
        if not group:
            continue
        print("=" * 72)
        print("%s events" % tag)
        print("=" * 72)
        for e, ro, rn in group:
            print("evt %d" % e)
            print("  OFF: %s" % ("no tagger output / no vertex" if not ro
                                 else "%d row(s)" % len(ro)))
            for r in ro:
                print("   OFF" + fmt_row(r)[2:])
            for i, r in enumerate(rn):
                print("   ON row %d%s%s" % (i, "" if r["vtx"] else " [NO VERTEX]",
                                            fmt_row(r)[2:]))
                print(fmt_acts(r))
            print()
    print("NEW ids: %s" % " ".join(str(e) for e, _, _ in new))
    print("EXTRA ids: %s" % " ".join(str(e) for e, _, _ in extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

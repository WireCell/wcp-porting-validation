#!/usr/bin/env python3
"""doc pr/94 §7 -- the per-activity sidecar, and the corrected event label.

`nusel-table.tsv` is one row per QUALIFYING BUNDLE and roughly twenty scripts
parse it, so its row cardinality and column set are deliberately left alone.
This writes two NEW files next to an arm instead:

  nusel-mains.tsv   one row per MAIN ACTIVITY evaluated inside an in-beam
                    -window bundle, keyed (run, subrun, event, bundle_gid,
                    main_id), carrying that activity's own TGM/STM/FC/LM
                    verdicts, whether it was the bundle's selected candidate,
                    and whether it was evaluated at all.

  nusel-events-pr94.tsv
                    one row per event with a CORRECTED `event_label`.

The label is what doc pr/94 exists to fix.  `nusel_extract.py`'s `merge()`
derives it from bundle-level labels alone, so an event where every in-window
BUNDLE is TGM/STM/LM comes out `cosmic-tagged` -- even when the PR chain went on
to select and fully reconstruct a clean neutrino candidate from a demoted main
inside one of those bundles.  That was 68 of 629 `cosmic-tagged` events on the
2000-event mcp2k arm (10.8 %), and SBND 18255/395148 is the case study.  Here the
label is derived from what the chain ACTUALLY produced:

  nu-candidate    at least one row reconstructed a vertex
  cosmic-tagged   in-beam bundles exist, every evaluated activity in them is
                  TGM/STM/LM-convicted, and no row reconstructed a vertex
  opened-no-vertex
                  an untagged activity was selected but PR found no vertex
  no-bundle       no in-beam bundle produced a row at all

Source is `tracking-pr.root`'s `T_tagger` act_* block, which is authoritative --
it is written by the same component that made the decision.  Needs an arm run
with `SBND_NU_PER_BUNDLE=1`; a knob-off arm has no act_* branches and is
reported as such rather than guessed at.

Usage: pr94_mains_sidecar.py <arm> [--outdir DIR]
"""
import argparse
import os
import re
import sys

import uproot

MAIN_COLS = ["run", "subrun", "event", "bundle_gid", "nu_index", "main_id",
             "len_cm", "selected", "demoted", "tgm", "stm", "fc", "lm",
             "evaluated"]
EV_COLS = ["run", "subrun", "event", "n_inbeam_bundle", "n_activity",
           "n_with_vertex", "event_label"]


def rse(prdir, evt):
    """run/subrun from Trun; the filename already fixes the event number."""
    try:
        with uproot.open(os.path.join(prdir, "tracking-pr.root")) as f:
            t = f["Trun"].arrays(library="np")
            return int(t["runNo"][0]), int(t["subRunNo"][0])
    except Exception as err:                              # noqa: BLE001
        # Loud on purpose: a silent (-1, -1) would put a wrong run number on
        # every row of the sidecar and nothing downstream would notice.
        print("WARN: %s: cannot read Trun runNo/subRunNo (%s)" % (prdir, err),
              file=sys.stderr)
        return -1, -1


def write_table(path, cols, rows):
    w = [len(c) for c in cols]
    for r in rows:
        for j, v in enumerate(r):
            w[j] = max(w[j], len(str(v)))
    with open(path, "w") as fh:
        print("  ".join(c.ljust(w[j]) for j, c in enumerate(cols)), file=fh)
        for r in rows:
            print("  ".join(str(v).ljust(w[j]) for j, v in enumerate(r)), file=fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or args.arm

    evts = sorted(int(m.group(1)) for d in os.listdir(args.arm)
                  if (m := re.match(r"pr_evt(\d+)$", d)))
    main_rows, ev_rows, nolabel = [], [], 0

    for e in evts:
        prdir = os.path.join(args.arm, "pr_evt%d" % e)
        rp = os.path.join(prdir, "tracking-pr.root")
        if not os.path.exists(rp):
            continue
        with uproot.open(rp) as f:
            t = f["T_tagger"]
            if "act_cluster_id" not in t.keys():
                nolabel += 1
                continue
            a = t.arrays(library="np")
        run, subrun = rse(prdir, e)
        nrow = len(a["nu_index"])
        nact = nwith = 0
        any_untagged_selected = False
        for i in range(nrow):
            has_vtx = (float(a["nu_x"][i]), float(a["nu_y"][i]),
                       float(a["nu_z"][i])) != (0.0, 0.0, 0.0)
            nwith += int(has_vtx)
            sel_id = int(a["cluster_id"][i])
            for j in range(len(a["act_cluster_id"][i])):
                cid = int(a["act_cluster_id"][i][j])
                selected = int(a["act_is_selected"][i][j])
                tgm = int(a["act_tgm"][i][j])
                stm = int(a["act_stm"][i][j])
                lm = int(a["act_lm"][i][j])
                if selected and not (tgm or stm or lm > 0):
                    any_untagged_selected = True
                main_rows.append([run, subrun, e,
                                  int(a["matched_flash_gid"][i]),
                                  int(a["nu_index"][i]), cid,
                                  "%.1f" % float(a["act_length_cm"][i][j]),
                                  selected, int(a["act_is_demoted"][i][j]),
                                  tgm, stm, int(a["act_fc"][i][j]), lm,
                                  int(a["act_evaluated"][i][j])])
                nact += 1
            _ = sel_id
        if nrow == 0:
            label = "no-bundle"
        elif nwith:
            label = "nu-candidate"
        elif any_untagged_selected:
            label = "opened-no-vertex"
        else:
            label = "cosmic-tagged"
        ev_rows.append([run, subrun, e, nrow, nact, nwith, label])

    write_table(os.path.join(outdir, "nusel-mains.tsv"), MAIN_COLS, main_rows)
    write_table(os.path.join(outdir, "nusel-events-pr94.tsv"), EV_COLS, ev_rows)
    counts = {}
    for r in ev_rows:
        counts[r[-1]] = counts.get(r[-1], 0) + 1
    print("wrote %s/nusel-mains.tsv (%d activity rows) and "
          "nusel-events-pr94.tsv (%d events)" % (outdir, len(main_rows), len(ev_rows)))
    print("event_label:", ", ".join("%s=%d" % kv for kv in sorted(counts.items())))
    if nolabel:
        print("NOTE: %d event(s) had no act_* branches (knob-off arm); skipped."
              % nolabel, file=sys.stderr)


if __name__ == "__main__":
    main()

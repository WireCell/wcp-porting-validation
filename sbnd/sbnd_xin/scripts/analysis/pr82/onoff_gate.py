#!/usr/bin/env python3
"""doc pr/82 sec 3 gate 1 -- the harvest ON == OFF gate, over a whole arm.

pr/79 sec 10c ran this by hand on a handful of events and doc pr/82 sec 0b
(smoke test 4) ran it on exactly one.  Neither left a script behind, and
`scripts/retire/gate_cmp_arms.py` deliberately EXCLUDES calib-pr-evt*.json from
its comparison ("would make an on-vs-off comparison fail by construction"), so
it cannot be reused.  This is the missing piece.

The claim being gated: turning `dl_vtx_harvest` on is *recording only*.  Strip
every harvest-added key from the ON dump and what remains must be identical to
the OFF dump -- same vertices, same fits, same main_vertex, same scoreboard
rows, same everything.  If that holds, then

  (a) the pr/79 sec 10 byte-identity claim holds at this binary, and
  (b) doc pr/82 sec 1's carry-forward numbers, which were measured on the
      harvest-OFF pr87ion3 arms, transfer to the harvest-ON arms unchanged --
      which is what licenses carry_labels.py to join against the harvest arms
      (doc pr/82 sec 4.0).

The strip is deliberately written as a PREFIX rule (`hv_*`) plus the single
literal `harvest`, not as a hand-listed key set: a future harvest field would
silently fail an exact-list gate for the wrong reason.  The key names as of
toolkit 771f075b are, for the record --
  board: harvest, hv_trad_main_vertex_id, hv_single_candidate_ids,
         hv_all_showers_winner_ids, hv_cloud, hv_global
  rows[]: hv_filled, hv_n_proton_in, hv_n_proton_out, hv_z_prior, hv_n_tracks,
          hv_n_showers, hv_in_fv, hv_conflicts, hv_reduced_chi2
(clus/src/PrDisplayDump.cxx:720-788).

Comparison is on the canonical JSON serialisation (sort_keys, separators fixed)
of the stripped document, NOT on file bytes -- key order in the writer is not
part of the contract and CLAUDE.md M2 is the standing warning about comparing
containers rather than content.

Repro:
  python3 scripts/analysis/pr82/onoff_gate.py \
      --on work-nuecc48-harv3 --off work-nuecc48-pr87ion3
  python3 scripts/analysis/pr82/onoff_gate.py \
      --on work-mcp1k-harv3 --off work-mcp1k-pr87ion3 --tsv /tmp/mcp1k.tsv

Exit 0 = every common event identical AND no event missing on either side.
Exit 1 = any mismatch or any coverage gap.  A coverage gap is a FAIL, not a
warning: a short arm is the M5 truncation signature.
"""
import argparse
import json
import os
import re
import sys


def strip_harvest(obj):
    """Recursively drop `harvest` and every `hv_*` key.

    Applied to the whole document rather than just `vertex_scoreboard` so that
    the rows[] entries (which live one level down) are covered by the same
    rule, and so a harvest field added elsewhere in a future round is stripped
    rather than reported as a spurious diff.
    """
    if isinstance(obj, dict):
        return {k: strip_harvest(v) for k, v in obj.items()
                if not (k == "harvest" or k.startswith("hv_"))}
    if isinstance(obj, list):
        return [strip_harvest(v) for v in obj]
    return obj


def canon(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def events_of(arm):
    """event id -> calib dump path, for every pr_evt<N>/calib-pr-evt<N>.json."""
    out = {}
    if not os.path.isdir(arm):
        return out
    for name in os.listdir(arm):
        m = re.fullmatch(r"pr_evt(\d+)", name)
        if not m:
            continue
        ev = int(m.group(1))
        p = os.path.join(arm, name, "calib-pr-evt%d.json" % ev)
        if os.path.isfile(p):
            out[ev] = p
    return out


def first_diff(a, b):
    """A short, human-usable description of where two stripped docs differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return "only-in-OFF:%s" % k
            if k not in b:
                return "only-in-ON:%s" % k
            d = first_diff(a[k], b[k])
            if d:
                return "%s.%s" % (k, d)
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return "len %d!=%d" % (len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y)
            if d:
                return "[%d].%s" % (i, d)
        return None
    return None if a == b else "%r!=%r" % (a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, help="harvest-ON arm dir")
    ap.add_argument("--off", required=True, help="harvest-OFF reference arm dir")
    ap.add_argument("--tsv", default=None, help="optional per-event result TSV")
    args = ap.parse_args()

    on, off = events_of(args.on), events_of(args.off)
    common = sorted(set(on) & set(off))
    only_on = sorted(set(on) - set(off))
    only_off = sorted(set(off) - set(on))

    rows, bad, no_harvest = [], [], []
    for ev in common:
        with open(on[ev]) as fh:
            don = json.load(fh)
        with open(off[ev]) as fh:
            doff = json.load(fh)

        # The ON arm must actually BE a harvest arm; an inert knob would make
        # this gate pass vacuously, which is the M1 failure shape.
        sb = don.get("vertex_scoreboard") or {}
        if not sb.get("harvest"):
            no_harvest.append(ev)

        son, soff = strip_harvest(don), strip_harvest(doff)
        ok = canon(son) == canon(soff)
        why = "" if ok else (first_diff(soff, son) or "canonical-form")
        if not ok:
            bad.append((ev, why))
        rows.append((ev, "OK" if ok else "DIFF", why))

    if args.tsv:
        with open(args.tsv, "w") as fh:
            fh.write("evt\tverdict\tfirst_diff\n")
            for ev, v, why in rows:
                fh.write("%d\t%s\t%s\n" % (ev, v, why))

    print("ON  %s: %d dumps" % (args.on, len(on)))
    print("OFF %s: %d dumps" % (args.off, len(off)))
    print("common %d, identical-after-strip %d, DIFF %d"
          % (len(common), len(common) - len(bad), len(bad)))
    if no_harvest:
        print("!! %d ON dumps have no harvest flag: %s"
              % (len(no_harvest), no_harvest[:10]))
    if only_on:
        print("!! %d events only in ON: %s" % (len(only_on), only_on[:10]))
    if only_off:
        print("!! %d events only in OFF: %s" % (len(only_off), only_off[:10]))
    for ev, why in bad[:20]:
        print("   DIFF evt%d  %s" % (ev, why))

    fail = bool(bad or only_on or only_off or no_harvest)
    print("GATE:", "FAIL" if fail else "PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

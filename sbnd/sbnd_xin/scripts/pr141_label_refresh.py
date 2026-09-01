#!/usr/bin/env python3
"""doc pr/141 recommendation 3 -- how much of the pi0 census is LABEL STALENESS?

READ-ONLY.  doc pr/141 sec 3 found that two of the nine "absent-on-arm" gamma
slots were id RENAMES -- the census reading a stale label as a missing particle.
The owner's 2026-08-31 rescan (tag pi0mass-0904-owner) re-paired four of the
nine mass-failure events, so the same question can be asked directly of the
census: if the newest hand pairing wins instead of the oldest, does the
reconstruction agree with more of them?

`pr132_pi0_census.py` resolves labels BASE-WINS (emscan-0827 / emscan-0828-agent5
beat the pi0scan overlay).  That precedence is right for an overlay that only
EXTENDS the denominator; it is wrong for a rescan that CORRECTS an earlier call.
This script does not change the census -- it asks, per rescanned event, whether
the reconstruction's own accepted pi0 matches the OLD hand pair or the NEW one.

    python3 scripts/pr141_label_refresh.py --arm work-pr141r1-off \
        --tsv docs/pr/pr141-label-refresh.tsv
"""
import argparse, glob, json, os, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_TAG = "pi0mass-0904-owner"
OLD_TAGS = ["emscan-0827", "emscan-0828-agent5", "pi0scan-0829-agent"]
SAMPLE = {21073: "ncpi0", 168432: "mcp1k", 280159: "mcp2k", 286655: "mcp1k",
          348691: "mcp1k", 397630: "mcp2k", 409634: "mcp1k", 71872: "mcp2k",
          283713: "mcp1k"}


def pair_of(rec):
    p = (rec or {}).get("pio")
    if not p:
        return None
    g = p.get("gammas") or {}
    try:
        return tuple(sorted(int(g[s]["shower"]) for s in ("1", "2")))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="work-pr141r1-off")
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()

    rows, flips = [], 0
    for ev, samp in sorted(SAMPLE.items()):
        new = pair_of(json.load(open(os.path.join(
            SX, "em_labels", NEW_TAG, "labels-evt%d.json" % ev))))
        old = None
        for t in OLD_TAGS:
            p = os.path.join(SX, "em_labels", t, "labels-evt%d.json" % ev)
            if os.path.exists(p):
                q = pair_of(json.load(open(p)))
                if q:
                    old = q
                    break
        dump = json.load(open(os.path.join(
            SX, "%s-%s" % (args.arm, samp), "pr_evt%d" % ev,
            "calib-pr-evt%d.json" % ev)))
        groups = {}
        for s in (dump.get("showers") or ()):
            pid = s.get("pio_id")
            if pid is not None and int(pid) >= 0:
                groups.setdefault(int(pid), []).append(int(s["id"]))
        reco = {tuple(sorted(v)) for v in groups.values() if len(v) == 2}

        m_old = old in reco if old else False
        m_new = new in reco if new else False
        verdict = ("FLIPS TO EXACT under the refreshed label" if (m_new and not m_old)
                   else "already agreed" if m_old
                   else "no reco pair matches either" if not m_new
                   else "-")
        if m_new and not m_old:
            flips += 1
        rows.append(dict(event=ev, sample=samp,
                         old_pair="%s+%s" % old if old else "",
                         new_pair="%s+%s" % new if new else "",
                         changed=bool(old and new and old != new),
                         reco_pairs=";".join("%d+%d" % p for p in sorted(reco)) or "none",
                         reco_matches_old=m_old, reco_matches_new=m_new,
                         verdict=verdict))
        print("evt %-7d old %-14s new %-14s reco %-28s %s"
              % (ev, rows[-1]["old_pair"] or "-", rows[-1]["new_pair"] or "-",
                 rows[-1]["reco_pairs"], verdict))

    print("\n=== %d of %d rescanned events flip to EXACT on the label alone ==="
          % (flips, len(rows)))
    if args.tsv:
        hdr = ["event", "sample", "old_pair", "new_pair", "changed", "reco_pairs",
               "reco_matches_old", "reco_matches_new", "verdict"]
        with open(args.tsv, "w") as fh:
            fh.write("\t".join(hdr) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[h]) for h in hdr) + "\n")
        print("wrote %s" % args.tsv)


if __name__ == "__main__":
    main()

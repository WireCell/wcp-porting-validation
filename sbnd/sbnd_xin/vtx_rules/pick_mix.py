"""Select 60 never-AI-scanned events, mixed nueCC / NCpi0 / BNB-inclusive.

Round-3 machinery check for doc pr/80 sec 11.  The composition the owner asked
for (nueCC + NCpi0 + numuCC) cannot be met from the dev half alone: after the
sec 10 arms there are 11 unused dev-half nueCC and 4 unused dev-half NCpi0
events in existence.  So the two small strata are topped up from the test half
and the consumed test-half ids are written out, because those events stop being
held out the moment a scanner reads them.

Sample -> channel mapping (stated, not assumed):
  vtxscan-prod0813        nuecc48 production sample   -> nueCC
  vtxscan-prod0813-ncpi0  NC pi0 sample               -> NCpi0
  vtxscan-prod0813-mcp1k  BNB inclusive MC, 1k events -> numuCC-DOMINATED, not pure
The third is a proxy: there is no per-event truth channel in the dumps, so a
"numuCC" bucket can only be an inclusive BNB bucket.  Reported as such.

NOTE ON REPRODUCIBILITY.  This tool selects events NOT yet shown to any AI
scanner, by reading every `vtx_rules/runs/*/manifest.json`.  Once a run is
committed its events become "used", so re-running this will produce a DIFFERENT
sixty -- by design.  The selection actually scanned is committed beside the run
as `selection.tsv`; that file, not a re-run, is the record.

Run:  cd sbnd_xin && python3 vtx_rules/pick_mix.py <outdir>
"""
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.dirname(HERE))

import make_split          # noqa: E402
import vtx_io              # noqa: E402
import baselines           # noqa: E402

SEED = 20260816
QUOTA = [("vtxscan-prod0813",       "nueCC",   20),
         ("vtxscan-prod0813-ncpi0", "NCpi0",   12),
         ("vtxscan-prod0813-mcp1k", "BNBincl", 28)]


def used_events():
    """Every event any AI scanner has already been shown, from run manifests."""
    seen = set()
    for p in (glob.glob("vtx_rules/runs/*/manifest.json")
              + glob.glob("vtx_rules/runs/*/*/manifest.json")):
        with open(p) as fh:
            for m in json.load(fh):
                seen.add(m["event"])
    return seen


def main(outdir):
    used = used_events()
    split = make_split.load_split()
    half_of = {}
    for h in ("dev", "test"):
        for k in split[h]:
            half_of[k] = h

    pool = {}
    for L in vtx_io.load_labels():
        if L["event"] in used:
            continue
        if not baselines.deployed_dump_path(L):
            continue
        h = half_of.get(L["key"])
        if h is None:
            continue
        pool.setdefault((L["tag"], h), []).append(L)

    rng = random.Random(SEED)
    chosen, burned = [], []
    for tag, chan, want in QUOTA:
        got = []
        # Dev half first -- it is already spent as a development half.  Only
        # then take from the test half, and record what that costs.
        for h in ("dev", "test"):
            if len(got) >= want:
                break
            cand = sorted(pool.get((tag, h), []), key=lambda L: L["event"])
            rng.shuffle(cand)
            take = cand[:want - len(got)]
            got += take
            if h == "test":
                burned += [L["event"] for L in take]
        if len(got) < want:
            raise SystemExit("only %d unused events for %s (wanted %d)"
                             % (len(got), tag, want))
        for L in got:
            chosen.append(dict(event=L["event"], tag=tag, channel=chan,
                               half=half_of[L["key"]]))

    chosen.sort(key=lambda c: c["event"])
    os.makedirs(outdir, exist_ok=True)
    # manifest-shaped, so `selfscan.py prepare --only-manifest` consumes it and
    # keeps the label join key (which `--dumps` throws away, disabling `score`).
    with open(os.path.join(outdir, "selection.json"), "w") as fh:
        json.dump(chosen, fh, indent=1)
    with open(os.path.join(outdir, "selection.tsv"), "w") as fh:
        fh.write("event\tchannel\ttag\thalf\n")
        for c in chosen:
            fh.write("%s\t%s\t%s\t%s\n"
                     % (c["event"], c["channel"], c["tag"], c["half"]))
    with open(os.path.join(outdir, "test-half-consumed.txt"), "w") as fh:
        fh.write("# these test-half events are no longer held out: an AI "
                 "scanner read them in this run\n")
        for e in sorted(burned):
            fh.write("%s\n" % e)

    n = {}
    for c in chosen:
        n[(c["channel"], c["half"])] = n.get((c["channel"], c["half"]), 0) + 1
    for k in sorted(n):
        print("%-8s %-4s %d" % (k[0], k[1], n[k]))
    print("total %d, unique %d, test-half consumed %d"
          % (len(chosen), len({c["event"] for c in chosen}), len(burned)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

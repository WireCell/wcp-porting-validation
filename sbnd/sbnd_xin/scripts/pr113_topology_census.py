#!/usr/bin/env python3
"""doc pr/113 -- reconstruction-defined topology census for the EM-shower /
pi0 / long-muon validation sample.

There is NO MC truth in this chain: every prod0825 arm is SBND *data*
(the art inputs carry only artdaq::Fragments / CRT / PMT raw products, no
simb::MCTruth -- see doc 83 and doc pr/113 sec 6).  So every label below is a
RECONSTRUCTION verdict, not truth.  Purity is unknown and unknowable here; the
lists are for exercising and eyeballing the shower/pi0/long-muon code, NOT for
efficiency or purity measurements.

Per event we read `calib-pr-evt<ID>.json` (written only under
PR_EXTRA_STAGES=pr_display, which prod0825 used) and join `nusel-events.tsv`
for run/subrun and the pre-PR bundle verdict.

Two pool denominators are reported because they are NOT joinable and do not
agree (doc pr/113 sec 6):
  n_calib      events with calib-pr-evt*.json  == TaggerCheckNeutrino actually
               evaluated a main cluster.  THIS is the census denominator.
  n_nucand     events whose nusel-events.tsv event_label == 'nu-candidate'
               -- a PRE-PR bundle verdict from nusel_extract.py, taken before
               unmerge_bundle runs, on a different numbering scheme.

Topology predicates (all reco), evaluated on segments incident on the main
vertex ("primaries") and on showers[]:
  em_max        max kine_best over showers[] with particle_id == 11
  has_mu        a pdg-13 primary at least --mu-floor cm long (default 30)
  has_mu_any    the same with NO length floor -- kept only to show how bad the
                unfloored predicate is (1025/1366 vs 791/1366)
  has_e_nonpio  a primary e-rooted shower that is NOT pi0-paired
  has_pio_pair  >= 2 showers carry pio_id >= 0, i.e. a real gamma PAIR

Priority ladder, so the three lists are disjoint by construction.  All three
carry the SAME >= 100 MeV "sizable EM shower" floor:
  1. numuCC_em   has_mu and em_max >= 100 MeV      <-- the owner's ask
  2. NCpi0       (not has_mu) and has_pio_pair and em_max >= 100
  3. nueCC       (not has_mu) and (not has_pio_pair) and has_e_nonpio
                 and em_max >= 100

`has_e` (a primary e-like segment that roots ANY shower) is reported as a
column for continuity but is NOT part of any verdict -- the ladder uses
has_e_nonpio, which additionally requires the shower not be pi0-paired.

The two curated arms (nuecc48, ncpi0) enter their own list wholesale as
origin=curated; the unbiased beam arms (mcp1k, mcp2k) contribute origin=reco
additions.  numuCC_em is mined from the beam arms only.

Usage:
  pr113_topology_census.py <arm_dir>[:label] ... --out FILE.tsv --outdir DIR
"""
import argparse
import glob
import json
import os
import sys

EM_TIERS = (50.0, 100.0, 200.0)
# minimum length of a pdg-13 primary for it to count as a muon-like primary
MU_FLOOR_CM = 30.0


def pio_id_of(shower):
    """`showers[].pio_id`, with -1 for "not pi0-paired".

    ROUND 2 CORRECTION -- do not collapse this back into the `or -1` idiom used
    everywhere else in this file.  `pio_id` is allocated from **0**, and `0` is
    falsy in Python, so `(sh.get("pio_id", -1) or -1)` silently reports the
    FIRST pi0 group of every event as unpaired.  Since almost every event that
    has a group at all has group 0, round 1 undercounted paired events by 10x:
    7 of 1433 reported, **70** actual.  The idiom is safe for the other fields
    it is used on (a 0 energy or a 0 score maps to 0 either way) and is left
    alone there; it is only ever wrong for a value whose valid domain includes
    zero.  See doc pr/113 sec 10.
    """
    v = shower.get("pio_id")
    return -1 if v is None else v


def evtid(path):
    b = os.path.basename(path)
    return b[len("calib-pr-evt"):-len(".json")]


def load_nusel(arm):
    """event -> (run, subrun, event_label).  Missing file is not fatal."""
    out = {}
    p = os.path.join(arm, "nusel-events.tsv")
    if not os.path.exists(p):
        return out
    # NOTE: despite the .tsv name these files are SPACE-padded, not
    # tab-separated -- split on runs of whitespace.
    with open(p) as f:
        hdr = f.readline().split()
        idx = {c: i for i, c in enumerate(hdr)}
        for line in f:
            r = line.split()
            if len(r) < len(hdr):
                continue
            out[r[idx["event"]]] = (r[idx["run"]], r[idx["subrun"]],
                                    r[idx["event_label"]])
    return out


def census_event(path, sample, nusel, mu_floor=MU_FLOOR_CM):
    j = json.load(open(path))
    e = evtid(path)

    main_ids = [v["id"] for v in (j.get("vertices") or []) if v.get("is_main")]
    main_vtx = main_ids[0] if main_ids else None
    # 49/1433 events have a calib dump but main_vertex == null and empty
    # showers/segments/vertices -- PR ran and found no main cluster.  They are
    # kept as rows with has_main=0 so the denominator stays honest.
    mv = j.get("main_vertex") or {}
    main_cluster = mv.get("cluster_id", -1)

    segs = j.get("segments") or []
    showers = j.get("showers") or []
    kine = j.get("kine") or {}
    tag = j.get("tagger") or {}

    # --- primaries: segments incident on the main vertex -------------------
    prim = [s for s in segs
            if main_vtx is not None
            and (s.get("start_vertex_id") == main_vtx
                 or s.get("end_vertex_id") == main_vtx)]
    # A muon-like PRIMARY needs a length floor.  Without one, any pdg-13 stub
    # at the vertex counts: a hand check of nuecc48 111412 found has_mu set by
    # a 2.4 cm segment, and the raw flag fires on 1025/1366 beam events (5th
    # pct 3.2 cm).  With the 30 cm floor it is 791/1366.  See the sensitivity
    # table in doc pr/113 sec 6 -- the floor is reported, not hidden.
    mu_prim = [s for s in prim if abs(s.get("particle_id", 0)) == 13]
    mu_len = max((s["length"] for s in mu_prim), default=0.0)
    has_mu_any = int(len(mu_prim) > 0)
    has_mu = mu_len >= mu_floor

    # A primary e-like segment that actually ROOTS a shower.  Join key: the
    # dump's `showers[].id` IS the start segment's id (cluster*1000+seg);
    # `showers[].shower_id` is the internal sequential get_shower_id() and does
    # NOT join to segments.  (`segments[].shower_id` -> `showers[].id` is the
    # same join pr93_shower_composition.py uses.)
    shower_start_segs = {sh.get("id") for sh in showers
                         if sh.get("particle_id") == 11}
    e_prim = [s for s in prim
              if abs(s.get("particle_id", 0)) == 11 and s["id"] in shower_start_segs]
    has_e = len(e_prim) > 0

    # --- EM shower content --------------------------------------------------
    em = [sh for sh in showers if sh.get("particle_id") == 11]
    em_sorted = sorted(em, key=lambda sh: -(sh.get("kine_best", 0.0) or 0.0))
    em_E = [sh.get("kine_best", 0.0) or 0.0 for sh in em_sorted]
    em_max = em_E[0] if em_E else 0.0
    em_sum = sum(em_E)
    # Does the event's DOMINANT EM object belong to a reconstructed pi0?
    # This is the discriminator kine_pio_flag cannot be: the flag merely says
    # "a pi0 pair was found somewhere" and fires on 37/48 of the curated
    # nueCC arm (doc pr/113 sec 6) -- exactly the failure pr/93 sec 3 warned of.
    lead_pio = int(bool(em_sorted) and pio_id_of(em_sorted[0]) >= 0)
    # a primary e-rooted shower that is NOT pi0-paired == an electron candidate
    e_prim_ids = {s["id"] for s in prim if abs(s.get("particle_id", 0)) == 11}
    has_e_nonpio = int(any(pio_id_of(sh) < 0
                           for sh in em if sh.get("id") in e_prim_ids))
    em_tier = 0
    for t in EM_TIERS:
        if em_max >= t:
            em_tier = int(t)

    # --- pi0 ----------------------------------------------------------------
    pio_flag = int(kine.get("kine_pio_flag", 0) or 0)
    pio_mass = float(kine.get("kine_pio_mass", -1.0) or -1.0)
    n_pio_showers = sum(1 for sh in showers if pio_id_of(sh) >= 0)

    run, subrun, label = nusel.get(e, ("", "", ""))

    row = dict(
        sample=sample, run=run, subrun=subrun, evt=e,
        event_label=label, has_main=int(main_vtx is not None),
        main_cluster=main_cluster,
        n_seg=len(segs), n_shower=len(showers), n_em_shower=len(em),
        em_max=round(em_max, 2), em_sum=round(em_sum, 2), em_tier=em_tier,
        has_mu=int(has_mu), has_mu_any=has_mu_any,
        mu_len=round(mu_len, 2),
        has_e=int(has_e), n_prim=len(prim),
        pio_flag=pio_flag, pio_mass=round(pio_mass, 2),
        n_pio_showers=n_pio_showers, lead_pio=lead_pio,
        has_e_nonpio=has_e_nonpio,
        pio_e1=round(float(kine.get("kine_pio_energy_1", -1) or -1), 2),
        pio_e2=round(float(kine.get("kine_pio_energy_2", -1) or -1), 2),
        pio_angle=round(float(kine.get("kine_pio_angle", -1) or -1), 2),
        Enu=round(float(kine.get("kine_reco_Enu", 0.0) or 0.0), 2),
        nue_score=round(float(tag.get("nue_score", 0.0) or 0.0), 4),
        numu_score=round(float(tag.get("numu_score", 0.0) or 0.0), 4),
        pio_2_score=round(float(tag.get("pio_2_score", 0.0) or 0.0), 4),
        cosmic_flag=int(tag.get("cosmic_flag", 0) or 0),
        match_isFC=int(tag.get("match_isFC", 0) or 0),
    )

    # --- the three reco verdicts, as a PRIORITY LADDER so they are disjoint --
    #   1. muon-like primary               -> numuCC family (EM-tagged >=100 MeV)
    #   2. else an actual pi0 gamma PAIR   -> NCpi0
    #   3. else a non-pi0 primary e shower -> nueCC
    #
    # NCpi0 uses n_pio_showers >= 2 (both gammas carry a pio_id), NOT
    # kine_pio_flag and NOT lead_pio:
    #   - kine_pio_flag > 0 fires on 37/48 of the curated nueCC arm -- useless
    #     as a topology cut, exactly as pr/93 sec 3 warned;
    #   - lead_pio (the highest-energy EM shower is pi0-paired) is far too
    #     strict: pairing picks whatever clears the mass window, so over the 67
    #     curated events 36 showers carry a pio_id (18 pairs) yet the LEADING
    #     shower is paired in only 1 event.
    # The SAME >=100 MeV EM floor applies to all three reco verdicts, not just
    # numuCC.  Without it the nueCC additions are dominated by stubs: a hand
    # check of the delivered list found mcp1k 166804 ("e- 10.2 MeV", a 3.0 cm
    # segment) and 167684 ("e- 41.7 MeV"), against a curated nuecc48 arm whose
    # em_max median is 1173 MeV with 47/48 above 100.  Flooring takes the reco
    # nueCC additions 129 -> 50 and makes the three lists mutually comparable.
    # (No-op for NCpi0: all 4 reco additions are already >= 336 MeV.)
    sizable = em_max >= 100.0
    has_pio_pair = int(n_pio_showers >= 2)
    row["has_pio_pair"] = has_pio_pair
    row["is_numucc_em"] = int(has_mu and sizable)
    row["is_ncpi0_reco"] = int((not has_mu) and has_pio_pair and sizable)
    row["is_nuecc_reco"] = int((not has_mu) and (not has_pio_pair)
                               and has_e_nonpio and sizable)
    return row


COLS = ["sample", "run", "subrun", "evt", "event_label", "has_main",
        "main_cluster",
        "n_seg", "n_shower", "n_em_shower", "em_max", "em_sum", "em_tier",
        "has_mu", "has_mu_any", "mu_len", "has_e", "n_prim",
        "pio_flag", "pio_mass", "n_pio_showers", "has_pio_pair", "lead_pio",
        "has_e_nonpio", "pio_e1", "pio_e2",
        "pio_angle", "Enu", "nue_score", "numu_score", "pio_2_score",
        "cosmic_flag", "match_isFC",
        "is_nuecc_reco", "is_ncpi0_reco", "is_numucc_em"]

# The two curated arms are topology samples by upstream construction, so they
# enter their list wholesale; the unbiased beam arms contribute reco-selected
# additions.  origin is carried in the index files so the two never blur.
CURATED = {"nuecc48": "nuecc", "ncpi0": "ncpi0"}
BEAM = ("mcp1k", "mcp2k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="arm_dir[:sample_label]")
    ap.add_argument("--out", required=True, help="master TSV")
    ap.add_argument("--outdir", required=True, help="dir for the .index.txt lists")
    ap.add_argument("--mu-floor", type=float, default=MU_FLOOR_CM,
                    help="min length (cm) of a pdg-13 primary to count as a "
                         "muon-like primary (default %(default)s)")
    args = ap.parse_args()

    rows = []
    pool = []
    for spec in args.arms:
        arm, label = spec.split(":", 1) if ":" in spec else (
            spec, os.path.basename(spec.rstrip("/")))
        nusel = load_nusel(arm)
        n_evt_dirs = len(glob.glob(os.path.join(arm, "pr_evt*")))
        calibs = sorted(glob.glob(os.path.join(arm, "pr_evt*",
                                               "calib-pr-evt*.json")))
        n_nucand = sum(1 for v in nusel.values() if v[2] == "nu-candidate")
        for p in calibs:
            try:
                rows.append(census_event(p, label, nusel, args.mu_floor))
            except Exception as ex:            # noqa: BLE001 - report, don't die
                print(f"WARN {p}: {ex}", file=sys.stderr)
        pool.append((label, n_evt_dirs, len(calibs), n_nucand))

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLS) + "\n")

    def pick(curated_label, reco_key):
        sel = []
        for r in rows:
            if CURATED.get(r["sample"]) == curated_label:
                sel.append((r, "curated"))
            elif r["sample"] in BEAM and r[reco_key]:
                sel.append((r, "reco"))
        return sel

    lists = {
        "pr113-nuecc.index.txt": pick("nuecc", "is_nuecc_reco"),
        "pr113-ncpi0.index.txt": pick("ncpi0", "is_ncpi0_reco"),
        # the owner's genuinely new ask: mined from the unbiased beam arms only
        "pr113-numucc-emshower.index.txt": [
            (r, "reco") for r in rows
            if r["sample"] in BEAM and r["is_numucc_em"]],
    }
    for name, sel in lists.items():
        with open(os.path.join(args.outdir, name), "w") as f:
            f.write("# sample\torigin\trun\tsubrun\tevent\tem_max_MeV\t"
                    "em_tier\tEnu_MeV\tpio_mass\n")
            for r, origin in sorted(sel, key=lambda t: (t[0]["sample"],
                                                        int(t[0]["evt"]))):
                f.write(f"{r['sample']}\t{origin}\t{r['run']}\t{r['subrun']}"
                        f"\t{r['evt']}\t{r['em_max']}\t{r['em_tier']}"
                        f"\t{r['Enu']}\t{r['pio_mass']}\n")

    # ---- report both denominators, and the overlap between the lists -------
    print("pool (arm: pr_evt dirs / calib-pr json / nusel nu-candidate):")
    tot_dirs = tot_calib = tot_nucand = 0
    for label, nd, nc, nn in pool:
        print(f"  {label:26s} {nd:5d} {nc:5d} {nn:5d}")
        tot_dirs += nd
        tot_calib += nc
        tot_nucand += nn
    print(f"  {'TOTAL':26s} {tot_dirs:5d} {tot_calib:5d} {tot_nucand:5d}")
    n_main = sum(1 for r in rows if r["has_main"])
    print(f"census denominator = n_calib = {tot_calib}, of which "
          f"{n_main} have a main vertex ({tot_calib - n_main} have a calib "
          f"dump but main_vertex==null / empty showers)")
    print(f"  (nu-candidate {tot_nucand} is a PRE-PR bundle verdict from "
          f"nusel_extract.py -- different numbering, NOT joinable)")
    print(f"wrote {len(rows)} rows to {args.out}")
    for name, sel in lists.items():
        print(f"  {name}: {len(sel)}")
    keys = {n: {(r["sample"], r["evt"]) for r, _ in s} for n, s in lists.items()}
    names = list(keys)
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            ov = keys[names[i]] & keys[names[k]]
            print(f"  overlap {names[i]} x {names[k]}: {len(ov)}")


if __name__ == "__main__":
    main()

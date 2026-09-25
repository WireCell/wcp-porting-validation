#!/usr/bin/env python3
"""doc sbnd_xin/124 -- the anatomy of a numu > 0.9 flip between two PR arms that share their imaging.

For every event whose per-event verdict (highest numu_score candidate, as scripts/d123/r3_pr_compare.py
picks it) is > CUT in one arm only, compare what the PR stage was given and what it did, arm A vs arm B.

Cross-arm key.  PR cluster ids are NOT stable across arms (they are assigned after the Q/L stage, whose
clustering merges on flash t0), and one id can hold different points.  The imaging is shared, so a point
is keyed by (y, z, q) from the PR job's own clustering-global Bee layer (pr_evt<E>/mabc-pr.zip): y and z
do not move with the flash time (only x = x_raw - t0 * v_drift does), and q is the blob charge.  A PR
cluster is its key set.

Per arm, from tracking-pr.root: T_tagger (the selected candidate's active roster act_*, flash group,
numu flags), T_cluster (every cluster's matched flash gid/time/TPC, main/associated, tagger flags),
T_flash (every flash with its flash group), T_kine (particle list).

Classes (first match wins; every flag is also written):
  QL-main       the main cluster's key set differs majorly (Jaccard < 0.5) or one arm has no candidate
  pairing       the PR input sets differ by clusters that entered through the cross-TPC flash group
                (their matched flash is the other TPC's flash of the candidate's group) in one arm only
  t0-merge      the main's own key set differs (0.5 <= Jaccard < 0.98): imaging clusters merged into it
                in one arm only
  fragment      same main, the PR input sets differ by separate clusters on the candidate's own flash
  PR-only       identical PR input key sets; only the flash time / PE differ
Sub-flags: vertex move (cm), particle-list change, changed BDT-input groups.

usage: flip_anatomy.py --a <pr_root_A> --b <pr_root_B> --pa <products_A> --pb <products_B>
                       [--label A,B] [--cut 0.9] [--events ev.txt] [--tsv out.tsv] [--detail out.txt]
       (pr roots may hold pr_evt* directly or per-file f*/pr_evt*; MC events are keyed by run/subrun/event)
"""
import argparse, csv, glob, json, math, os, re, sys, zipfile
from collections import Counter, defaultdict
import numpy as np
import uproot

TREES = ("T_tagger", "T_cluster", "T_flash", "T_kine", "T_bundle")
PDG = {11: "e", -11: "e", 13: "mu", -13: "mu", 211: "pi", -211: "pi", 2212: "p", 22: "g", 111: "pi0", 321: "K", 2112: "n", 0: "?"}


def best_rows(prod):
    """(run, subrun, event) -> the candidate row with the highest numu_score (r3_pr_compare.py's pick)."""
    out = {}
    for r in csv.DictReader(open(os.path.join(prod, "candidates.tsv")), delimiter="\t"):
        k = (int(r["run"]), int(r["subrun"]), int(r["event"]))
        if k not in out or float(r["numu_score"]) > float(out[k]["numu_score"]):
            out[k] = r
    return out


def index_pr(root):
    """(run, subrun, event) -> pr_evt dir, from each dir's nusel-evt<E>.tsv (MC event numbers repeat across files)."""
    idx = {}
    for d in glob.glob(os.path.join(root, "pr_evt*")) + glob.glob(os.path.join(root, "f*", "pr_evt*")):
        e = os.path.basename(d)[6:]
        ns = os.path.join(d, "nusel-evt%s.tsv" % e)
        rse = None
        if os.path.exists(ns):
            with open(ns) as f:
                f.readline(); l = f.readline().split()
                if len(l) >= 3:
                    rse = (int(l[0]), int(l[1]), int(l[2]))
        if rse is None:
            continue
        idx[rse] = d
    return idx


def load_arm(d, row):
    ev = os.path.basename(d)[6:]
    f = uproot.open(os.path.join(d, "tracking-pr.root"))
    T = {t: f[t].arrays(library="np") for t in TREES if t in f}
    # key sets per PR cluster id
    z = zipfile.ZipFile(os.path.join(d, "mabc-pr.zip"))
    n = [x for x in z.namelist() if x.endswith("-clustering-global.json")][0]
    cg = json.loads(z.read(n))
    keys = defaultdict(set)
    for y, zz, q, c in zip(cg["y"], cg["z"], cg["q"], cg["cluster_id"]):
        keys[c].add((round(y, 1), round(zz, 1), round(q, 0)))
    tc = T["T_cluster"]
    clus = {int(c): {k: tc[k][i] for k in tc} for i, c in enumerate(tc["cluster_id"])}
    tf = T["T_flash"]
    flashes = {int(g): {k: tf[k][i] for k in tf} for i, g in enumerate(tf["gid"])}
    tag = None
    if row is not None:
        tt = T["T_tagger"]
        for i in range(len(tt["cluster_id"])):
            if int(tt["cluster_id"][i]) == int(row["cluster_id"]) or int(tt["nu_index"][i]) == int(row["nu_index"]):
                tag = {k: tt[k][i] for k in tt}
                if int(tt["cluster_id"][i]) == int(row["cluster_id"]):
                    break
    kine = None
    if row is not None and "T_kine" in T:
        tk = T["T_kine"]
        for i in range(len(tk["cluster_id"])):
            if int(tk["cluster_id"][i]) == int(row["cluster_id"]):
                kine = {k: tk[k][i] for k in tk}
    return dict(dir=d, row=row, tag=tag, kine=kine, keys=keys, clus=clus, flashes=flashes, bundle=T.get("T_bundle"))


def jacc(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def pr_input(arm):
    """PR input cluster ids of the selected candidate: act_in_pr == 1, and the selected main."""
    t = arm["tag"]
    if t is None:
        return set()
    ids = set(int(c) for c, p in zip(t["act_cluster_id"], t["act_in_pr"]) if p == 1)
    ids.add(int(arm["row"]["cluster_id"])); ids.add(int(arm["row"]["sel_cluster_id"]))
    return ids


def keyset(arm, ids):
    s = set()
    for c in ids:
        s |= arm["keys"].get(c, set())
    return s


def owner(arm, key_set):
    """Which clusters of this arm hold these keys: [(cluster_id, nkeys)] by count."""
    cnt = Counter()
    inv = arm.setdefault("_inv", None)
    if inv is None:
        inv = {}
        for c, ks in arm["keys"].items():
            for k in ks:
                inv[k] = c
        arm["_inv"] = inv
    for k in key_set:
        c = inv.get(k)
        if c is not None:
            cnt[c] += 1
    return cnt.most_common()


def cdesc(arm, c, nk=None):
    x = arm["clus"].get(c)
    if x is None:
        return "c%d(?)" % c
    fl = []
    for f in ("tgm", "stm", "lm"):
        if int(x[f]) == 1:
            fl.append(f.upper())
    role = "main" if int(x["is_main"]) else "assoc"
    return "c%d[%s %.1fcm %dpt t=%.3fus tpc%d%s%s%s]" % (
        c, role, x["length_cm"], x["npoints"], x["flash_time_us"], x["flash_tpc"],
        " beam" if int(x["beam_flash"]) else "", (" " + "/".join(fl)) if fl else "", (" n=%d" % nk) if nk is not None else "")


def beam_flashes(arm):
    return sorted([(int(v["tpc"]), float(v["time_us"]), float(v["pe"]), int(v["flash_group"]), g)
                   for g, v in arm["flashes"].items() if int(v["in_window"]) == 1])


def particles(arm):
    k = arm["kine"]
    if k is None:
        return ""
    out = []
    for t, e, inc in zip(k["kine_particle_type"], k["kine_energy_particle"], k["kine_energy_included"]):
        out.append("%s%.0f" % (PDG.get(int(t), str(int(t))), e))
    return " ".join(out)


GROUPS = ("cosmict_", "numu_cc_", "shw_sp_", "br", "mip", "stem", "pio", "lol", "tro", "hol", "vis", "cme", "anc", "gap", "spt", "stw", "sig", "lem", "brm", "numu_")


def changed_groups(ta, tb):
    if ta is None or tb is None:
        return ""
    g = Counter()
    for k in ta:
        if k.startswith("act_") or k in ("cluster_id", "matched_flash_gid", "nu_index", "flash_time_us", "flash_pe", "flash_group", "flash_tpc"):
            continue
        va, vb = ta[k], tb.get(k)
        try:
            fa = np.ravel(np.asarray(va, dtype=float)); fb = np.ravel(np.asarray(vb, dtype=float))
            same = fa.shape == fb.shape and np.allclose(fa, fb, equal_nan=True, rtol=1e-5, atol=1e-6)
        except Exception:
            same = True
        if not same:
            p = next((x for x in GROUPS if k.startswith(x)), k.split("_")[0] + "_")
            g[p] += 1
    return ",".join("%s%d" % (k, v) for k, v in g.most_common())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--pa", required=True); ap.add_argument("--pb", required=True)
    ap.add_argument("--label", default="A,B"); ap.add_argument("--cut", type=float, default=0.9)
    ap.add_argument("--events"); ap.add_argument("--tsv"); ap.add_argument("--detail")
    ap.add_argument("--sample", default="")
    a = ap.parse_args()
    LA, LB = a.label.split(",")
    RA, RB = best_rows(a.pa), best_rows(a.pb)
    IA, IB = index_pr(a.a), index_pr(a.b)
    keys = sorted(set(RA) | set(RB) | (set(IA) & set(IB)))
    def sc(R, k):
        return float(R[k]["numu_score"]) if k in R else -999.0
    flips = [k for k in keys if (sc(RA, k) > a.cut) != (sc(RB, k) > a.cut)]
    if a.events:
        want = set(int(x.split()[0]) for x in open(a.events) if x.strip())
        flips = [k for k in flips if k[2] in want]
    det = open(a.detail, "w") if a.detail else sys.stdout
    rows = []
    for k in flips:
        if k not in IA or k not in IB:
            print("WARN no pr dir for %s" % (k,), file=sys.stderr); continue
        A = load_arm(IA[k], RA.get(k)); B = load_arm(IB[k], RB.get(k))
        kind = "LOST" if sc(RA, k) > a.cut else "GAINED"      # relative to B (new)
        rec = dict(sample=a.sample, run=k[0], subrun=k[1], event=k[2], kind=kind,
                   numu_A=sc(RA, k), numu_B=sc(RB, k))
        for L, X in ((LA, A), (LB, B)):
            r = X["row"]
            rec["Enu_" + L] = float(r["reco_Enu"]) if r else 0.0
            rec["vtx_" + L] = "%.1f,%.1f,%.1f" % (float(r["nu_x"]), float(r["nu_y"]), float(r["nu_z"])) if r else ""
            rec["main_" + L] = int(r["sel_cluster_id"]) if r else -1     # the selected activity (before a vertex-driven swap)
            rec["vmoved_" + L] = int(r["vertex_moved_cluster"]) if r else ""
            rec["mainlen_" + L] = float(r["sel_length_cm"]) if r else 0.0
            rec["flash_" + L] = "%s/%s/tpc%s" % (r["flash_time_us"], r["flash_pe"], r["flash_tpc"]) if r else ""
            rec["vdef_" + L] = r["vertex_default"] if r else ""
            rec["nt_" + L] = r["neutrino_type"] if r else ""
            rec["pid_" + L] = particles(X)
            bf = beam_flashes(X)
            rec["beamfl_" + L] = ";".join("tpc%d:%.3f/%.0f/g%d" % (t, tt, pe, g) for t, tt, pe, g, _ in bf)
            t0 = [x for x in bf if x[0] == 0]; t1 = [x for x in bf if x[0] == 1]
            rec["dt01_" + L] = ("%.0f" % (1000 * min((abs(x[1] - y[1]) for x in t0 for y in t1)))) if t0 and t1 else ""
            rec["paired_" + L] = int(any(x[3] == y[3] for x in t0 for y in t1)) if t0 and t1 else 0
        # vertex move
        if A["row"] and B["row"]:
            va = np.array([float(A["row"][c]) for c in ("nu_x", "nu_y", "nu_z")])
            vb = np.array([float(B["row"][c]) for c in ("nu_x", "nu_y", "nu_z")])
            rec["dvtx_cm"] = round(float(np.linalg.norm(va - vb)), 1)
        else:
            rec["dvtx_cm"] = ""
        # main and PR-input key sets
        kmA = A["keys"].get(rec["main_" + LA], set()) if A["row"] else set()
        kmB = B["keys"].get(rec["main_" + LB], set()) if B["row"] else set()
        inA, inB = pr_input(A), pr_input(B)
        kA, kB = keyset(A, inA), keyset(B, inB)
        rec["main_jacc"] = round(jacc(kmA, kmB), 3) if (kmA or kmB) else ""
        rec["input_jacc"] = round(jacc(kA, kB), 3) if (kA or kB) else ""
        rec["npts_in_A"] = len(kA); rec["npts_in_B"] = len(kB)
        onlyA, onlyB = kA - kB, kB - kA
        rec["onlyA"] = len(onlyA); rec["onlyB"] = len(onlyB)
        # how the differing points entered: per owning cluster in the arm that has them
        def entry(X, only):
            t = X["tag"]; gid = int(t["matched_flash_gid"]) if t is not None else None
            grp = int(t["flash_group"]) if t is not None else None
            out = []
            for c, nk in owner(X, only):
                x = X["clus"].get(c)
                if x is None:
                    out.append(("?", c, nk)); continue
                if c in (int(X["row"]["cluster_id"]), int(X["row"]["sel_cluster_id"])):
                    via = "main"
                elif gid is not None and int(x["matched_flash_gid"]) == gid:
                    via = "own"
                else:
                    fg = X["flashes"].get(int(x["matched_flash_gid"]), {}).get("flash_group")
                    via = "group" if (fg is not None and grp is not None and int(fg) == grp) else "other"
                out.append((via, c, nk))
            return out
        eA = entry(A, onlyA) if A["row"] else []
        eB = entry(B, onlyB) if B["row"] else []
        vias = Counter(v for v, _, n in eA + eB for _ in [0] if n >= 5)
        # classification
        if not A["row"] or not B["row"] or (rec["main_jacc"] != "" and rec["main_jacc"] < 0.5):
            cls = "QL-main"
        elif vias.get("group"):
            cls = "pairing"
        elif rec["main_jacc"] < 0.98:
            cls = "t0-merge"
        elif vias.get("own") or vias.get("other"):
            cls = "fragment"
        elif rec["onlyA"] + rec["onlyB"] == 0:
            cls = "PR-only"
        else:
            cls = "other"
        rec["class"] = cls
        rec["vias"] = ",".join("%s%d" % kv for kv in sorted(vias.items()))
        rec["bdt_changed"] = changed_groups(A["tag"], B["tag"])
        rows.append(rec)
        # detail sheet
        print("=" * 110, file=det)
        print("%s %d/%d/%d  %s  class=%s  numu %s %.3f -> %s %.3f  Enu %.0f -> %.0f  dvtx %s cm" % (
            a.sample, k[0], k[1], k[2], kind, cls, LA, rec["numu_A"], LB, rec["numu_B"], rec["Enu_" + LA], rec["Enu_" + LB], rec["dvtx_cm"]), file=det)
        for L, X, kmX, only, eX, other, kother in ((LA, A, kmA, onlyA, eA, B, kmB), (LB, B, kmB, onlyB, eB, A, kmA)):
            r = X["row"]
            if r is None:
                print("  [%s] no candidate.  where the other arm's main points are here:" % L, file=det)
                own = owner(X, kother)
                nin = sum(n for _, n in own)
                print("        %d of %d keys present in this arm's PR scope%s" % (nin, len(kother), "" if nin else " (cluster out of scope: its t0 puts it outside the drift volume, or unmatched)"), file=det)
                for c, nk in own[:4]:
                    print("        %s" % cdesc(X, c, nk), file=det)
                bf = beam_flashes(X)
                print("        beam-window flashes: %s" % "; ".join("tpc%d t=%.3f pe=%.0f grp=%d gid=%d" % f for f in bf), file=det)
                if X["bundle"] is not None:
                    tb = X["bundle"]
                    for i in range(len(tb["gid"])):
                        print("        T_bundle gid=%d t=%.3f pe=%.0f n_main=%d n_dem=%d n_comp=%d rej_cosmic=%d rej_stm=%d rej_floor=%d reason=%d sel=%d" % (
                            tb["gid"][i], tb["flash_time_us"][i], tb["flash_pe"][i], tb["n_main"][i], tb["n_demoted"][i], tb["n_companion"][i],
                            tb["n_rej_cosmic"][i], tb["n_rej_stm_only"][i], tb["n_rej_floor"][i], tb["reason"][i], tb["sel_cluster_id"][i]), file=det)
                continue
            t = X["tag"]
            print("  [%s] numu %s Enu %s vtx (%s) nt %s vdef %s  main %s  flash %s grp %s  pid: %s" % (
                L, r["numu_score"], r["reco_Enu"], rec["vtx_" + L], r["neutrino_type"], r["vertex_default"],
                cdesc(X, int(r["sel_cluster_id"])) + ("" if r["cluster_id"] == r["sel_cluster_id"] else " vertex-moved-to " + cdesc(X, int(r["cluster_id"]))), rec["flash_" + L], int(t["flash_group"]) if t is not None else "?", rec["pid_" + L]), file=det)
            print("        beam-window flashes: %s" % rec["beamfl_" + L], file=det)
            print("        PR input: %d clusters, %d keys; only here: %d keys in:" % (len(pr_input(X)), len(keyset(X, pr_input(X))), len(only)), file=det)
            for via, c, nk in eX[:8]:
                if nk < 3:
                    continue
                # where are these keys in the other arm?
                oth = owner(other, X["keys"][c] & only) if c in X["keys"] else []
                print("          via=%-5s %s  -> in other arm: %s" % (via, cdesc(X, c, nk),
                      ", ".join(cdesc(other, oc, on) for oc, on in oth[:2]) or "not present"), file=det)
        print("  BDT-input groups changed: %s" % rec["bdt_changed"], file=det)
    cols = ["sample", "run", "subrun", "event", "kind", "class", "numu_A", "numu_B", "Enu_" + LA, "Enu_" + LB, "dvtx_cm",
            "main_jacc", "input_jacc", "npts_in_A", "npts_in_B", "onlyA", "onlyB", "vias",
            "mainlen_" + LA, "mainlen_" + LB, "flash_" + LA, "flash_" + LB, "dt01_" + LA, "dt01_" + LB, "paired_" + LA, "paired_" + LB,
            "vmoved_" + LA, "vmoved_" + LB, "nt_" + LA, "nt_" + LB, "vdef_" + LA, "vdef_" + LB, "vtx_" + LA, "vtx_" + LB, "pid_" + LA, "pid_" + LB,
            "beamfl_" + LA, "beamfl_" + LB, "bdt_changed"]
    if a.tsv:
        with open(a.tsv, "w") as f:
            w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    c = Counter((r["kind"], r["class"]) for r in rows)
    print("flips: %d (%s)" % (len(rows), ", ".join("%s/%s %d" % (k[0], k[1], v) for k, v in sorted(c.items()))), file=sys.stderr)


if __name__ == "__main__":
    main()

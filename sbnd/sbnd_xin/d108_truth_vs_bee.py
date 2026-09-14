#!/usr/bin/env python3
"""doc 108 sec 3.6: validate the bare-ROOT reco1 truth (d108_reco1_truth_vectors.C
output) against the production Bee mc.json truth nodes of the same events.

Usage:
  python3 d108_truth_vs_bee.py docs/108_logs/reco1_truth_vectors.txt [--bee /nfs/data/1/xqian/sbnd_data/run/bee]

Per interaction: flavour (|pdg|), CC/NC, mode name, Etot (MeV), vertex (cm),
T (us) and Edep (MeV).  The generator MCTruth product id is taken as the one
whose Assns keys reach n_nu-1 in an event with n_nu > 1 (the corsika product
holds a single MCTruth); its Edep per key is compared with Bee's Edep.
Bee lists an interaction only if it has a particle above 10 MeV KE, so a
reco1 interaction absent from Bee is reported, not counted as a mismatch.
"""
import argparse
import json
import os
import re
import zipfile
import zlib

MODE = {0: "QE", 1: "RES", 2: "DIS", 3: "COH", 10: "MEC"}
FLAV = {12: "nue", 14: "numu", 16: "nutau"}


def parse(log):
    ev, cur = {}, None
    for line in open(log):
        m = re.match(r"(r\d+_s\d+_e\d+) entry (\d+): n_nu=(\d+)", line)
        if m:
            cur = m.group(1)
            ev[cur] = {"nu": [], "edep": {}}
            continue
        m = re.match(r"\s+NU idx=(\d+) pdg=(\S+) ccnc=(\S+) mode=(\S+) int_type=(\S+) E_GeV=(\S+) "
                     r"vtx_cm=\((\S+),(\S+),(\S+)\) t_ns=(\S+)", line)
        if m and cur:
            g = m.groups()
            ev[cur]["nu"].append(dict(idx=int(g[0]), pdg=int(float(g[1])), ccnc=int(float(g[2])),
                                      mode=int(float(g[3])), E=float(g[5]) * 1e3,
                                      vtx=[float(g[6]), float(g[7]), float(g[8])], T=float(g[9]) / 1e3))
            continue
        m = re.match(r"\s+EDEP_MeV by \(MCTruth product id, key\):(.*)unmatched=(\S+)", line)
        if m and cur:
            for pid, key, val in re.findall(r"\((\d+),(\d+)\)=(\S+)", m.group(1)):
                ev[cur]["edep"][(int(pid), int(key))] = float(val)
            ev[cur]["unmatched"] = float(m.group(2))
    return ev


def bee_truth(bee, tag):
    z = zipfile.ZipFile(os.path.join(bee, f"bee_{tag}.zip"))
    mc = json.loads(z.read([n for n in z.namelist() if n.endswith("mc.json")][0]))
    out = {}
    for n in (mc if isinstance(mc, list) else [mc]):
        if 9000000 <= n["id"] < 9100000:
            m = re.match(r"\d+ (\S+) (\S+) (CC|NC) Etot (\S+) MeV Edep (\S+) MeV T (\S+) us", n["text"])
            out[n["id"] - 9000000] = dict(flav=m.group(1), mode=m.group(2), cc=m.group(3), E=float(m.group(4)),
                                          Edep=float(m.group(5)), T=float(m.group(6)), vtx=n["data"]["start"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--bee", default="/nfs/data/1/xqian/sbnd_data/run/bee")
    a = ap.parse_args()
    ev = parse(a.log)
    # art ProductID = CRC32 of the product's branch name (checked: generator
    # 3130632973, corsika 647339619 on these files)
    gen = zlib.crc32(b"simb::MCTruths_generator__GenieGen.") & 0xffffffff
    print("generator MCTruth product id = crc32('simb::MCTruths_generator__GenieGen.') =", gen,
          "| ids seen:", sorted({pid for e in ev.values() for (pid, _k) in e["edep"]}))
    tot = dict(nu=0, in_bee=0, flav=0, cc=0, mode=0, E=0, vtx=0, T=0, edep=0)
    for tag, e in ev.items():
        B = bee_truth(a.bee, tag)
        for nu in e["nu"]:
            tot["nu"] += 1
            b = B.get(nu["idx"])
            ed = e["edep"].get((gen, nu["idx"]), 0.0) if gen is not None else float("nan")
            if b is None:
                print(f"{tag} nu{nu['idx']}: not in Bee (reco1 Edep {ed:.1f} MeV)")
                continue
            tot["in_bee"] += 1
            ok = dict(flav=FLAV.get(abs(nu["pdg"])) == b["flav"], cc=("CC" if nu["ccnc"] == 0 else "NC") == b["cc"],
                      mode=MODE.get(nu["mode"], str(nu["mode"])) == b["mode"], E=abs(nu["E"] - b["E"]) < 0.11,
                      vtx=max(abs(p - q) for p, q in zip(nu["vtx"], b["vtx"])) < 0.02, T=abs(nu["T"] - b["T"]) < 0.001,
                      edep=abs(ed - b["Edep"]) < 0.11)
            for k, v in ok.items():
                tot[k] += int(v)
            print(f"{tag} nu{nu['idx']}: pdg {nu['pdg']:+d} | Bee {b['flav']} {b['mode']} {b['cc']} "
                  f"E {b['E']} vs {nu['E']:.1f} | Edep Bee {b['Edep']} vs reco1 {ed:.1f} | "
                  f"{'ALL OK' if all(ok.values()) else 'MISMATCH ' + str([k for k, v in ok.items() if not v])}")
    print("summary", tot)


if __name__ == "__main__":
    main()

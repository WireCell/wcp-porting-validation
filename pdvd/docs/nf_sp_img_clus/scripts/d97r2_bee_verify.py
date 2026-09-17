#!/usr/bin/env python3
"""doc pdvd/97 round 2 -- check an uploaded Bee set against the local zip, and write scan/d97r2/bee_sets.txt.

    python3 d97r2_bee_verify.py <set url .../event/list/> [--uploaded "2026-09-16 (owner request)"]

An HTTP 200 alone proves nothing on Bee: a missing layer also answers 200 with a short "does not exist" body
(memory: bee layer url 200 when missing).  So:
  (1) the set's event/list/ page must link exactly events 0..N-1, N = the events in the zip;
  (2) per event, the layer routes event/<i>/mc/ and event/<i>/track_fit-global/ must download at exactly the size of
      the zip member data/<i>/<i>-mc.json and data/<i>/<i>-track_fit-global.json.
curl runs under `env -i` (the direnv shell breaks its OpenSSL paths).  rc 0 only if every check passes.
"""
import argparse, datetime, os, re, subprocess, sys, zipfile

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
ZIP = "/home/xqian/tmp/d97r2/bee-d97r2-pdvd.zip"
IDX = IMG + "/pdvd/docs/scan/d97r2/bee-d97r2-pdvd.index.txt"
OUT = IMG + "/pdvd/docs/scan/d97r2/bee_sets.txt"


def curl(url, body=False):
    cmd = ["env", "-i", "PATH=/usr/bin:/bin", "curl", "-sk", "-w", "\n%{http_code} %{size_download}"]
    r = subprocess.run(cmd + [url], capture_output=True, check=True)
    text, _, tail = r.stdout.rpartition(b"\n")
    code, size = tail.decode().split()
    return int(code), int(size), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--uploaded", default=datetime.date.today().isoformat())
    a = ap.parse_args()
    base = a.url.split("/event/list")[0]
    z = zipfile.ZipFile(ZIP)
    nev = len({n.split("/")[1] for n in z.namelist() if n.startswith("data/") and n.count("/") >= 2 and n.split("/")[1]})
    classes = [l.rstrip("\n").split("\t") for l in open(IDX) if not l.startswith("#")]
    bad = 0
    code, size, page = curl(a.url)
    listed = sorted({int(m) for m in re.findall(rb"event/(\d+)/", page)})
    ok_list = code == 200 and listed == list(range(nev))
    bad += not ok_list
    L = [f"# doc pdvd/97 round 2 -- Bee showcase set (built by scripts/d97r2_build_bee.sh from production d103vflip;",
         "# every member sha256-identical to the production zips; checked by scripts/d97r2_bee_verify.py)",
         "# det   arm        events  uploaded      url",
         f"pdvd    d103vflip  {nev}      {a.uploaded}  {a.url}",
         "",
         f"# event/list/: http {code}; linked events {listed[:1]}..{listed[-1:]} ({len(listed)}); zip events {nev} -> "
         f"{'OK' if ok_list else 'MISMATCH'}",
         "# per event: layer route downloaded, size compared with the zip member (a missing layer also returns 200)",
         "# event  class          key             layer             http  size_download  zip_member_size  ok"]
    for i, cls, key, *_ in classes:
        for layer, member in (("mc", f"data/{i}/{i}-mc.json"), ("track_fit-global", f"data/{i}/{i}-track_fit-global.json")):
            code, size, _ = curl(f"{base}/event/{i}/{layer}/")
            want = z.getinfo(member).file_size
            ok = code == 200 and size == want
            bad += not ok
            L.append(f"{i:<6} {cls:14s} {key:15s} {layer:17s} {code}   {size:<13}  {want:<15}  {'yes' if ok else 'NO'}")
    L.append(f"# {2 * len(classes) - sum(1 for l in L if l.endswith(' NO'))}/{2 * len(classes)} layer sizes match; "
             f"overall {'PASS' if not bad else 'FAIL'}")
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

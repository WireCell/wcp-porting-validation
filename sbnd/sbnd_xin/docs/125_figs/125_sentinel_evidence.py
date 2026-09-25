import sys, os, glob, re, json, importlib.util, collections
SX="/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
spec=importlib.util.spec_from_file_location("s",f"{SX}/scripts/pr127_sentinels.py"); s=importlib.util.module_from_spec(spec); sys.argv=["x"]; spec.loader.exec_module(s)
TAGS=["satellite","pf-orphan-guard-freed","orphan","pr128 pf-orphan-near-cross-cluster","pr130 pass4_prox_guard: decline seg=","long_muon_range","stem_backfill_back_dvtx","suppress decline seg="]
EV={69314:"mcp2k",171572:"mcp2k",315167:"mcp1k",72786:"mcp2k",393505:"mcp2k",497311:"mcp2k",292643:"mcp1k",179369:"mcp2k"}
def nurow(arm,e):
    p=f"{arm}/pr_evt{e}/nusel-evt{e}.tsv"
    if not os.path.exists(p): return "no nusel"
    rows=[l.rstrip("\n").split("\t") for l in open(p)]
    h=[x.strip() for x in rows[0]]; out=[]
    for r in rows[1:]:
        r=[x.strip() for x in re.split(r"\s{2,}|\t",("  ".join(r)))] if len(r)<len(h) else [x.strip() for x in r]
        d=dict(zip(h,r))
        if d.get("in_beam")=="1" or d.get("label","not-tagged")!="not-tagged":
            out.append(f"main={d.get('main_id')} t={d.get('flash_time_us')} pe={d.get('flash_pe')} npts={d.get('npts_main')} len={d.get('len_main_cm')} lab={d.get('label')}")
    return "; ".join(out) or "no in-beam/tagged row"
for e,samp in EV.items():
    print(f"===== {e} ({samp})")
    for tag in ("pr150s0","d123lgoppr"):
        arm=f"{SX}/work-{samp}-{tag}"
        c=s.calib(arm,e) or {}
        kin={k:v for k,v in c.items() if k in ("kine_reco_Enu",)} if isinstance(c,dict) else {}
        pf=s.pf_texts(arm,e) or []
        logs="".join(open(f,errors="replace").read() for f in glob.glob(f"{arm}/pr_evt{e}/*.log"))
        cnt={t:logs.count(t) for t in TAGS if logs.count(t)}
        sh=len(c.get("showers",[])) if isinstance(c,dict) else None
        print(f"  {tag:10s} Enu={kin.get('kine_reco_Enu')} showers={sh} nPF={len(pf)}")
        print(f"     nu: {nurow(arm,e)}")
        print(f"     PF: {[t.strip() for t in pf][:18]}")
        print(f"     tags: {cnt}")
        for t in ("suppress decline seg=","pass4_prox_guard: decline seg=","long_muon_range"):
            ls=[l.strip()[:150] for l in logs.splitlines() if t in l][:4]
            if ls: print(f"     [{t}] {ls}")

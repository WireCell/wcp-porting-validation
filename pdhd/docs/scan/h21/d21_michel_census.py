import glob,os,json,csv,sys
import uproot
IMG="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
X=IMG+"/pdhd/docs/scan"
base=lambda v: v[5:] if v and v.startswith("FRAG_") else v
rd=lambda p: list(csv.DictReader([l for l in open(p) if not l.startswith("#")],delimiter="\t"))
POP={"%s/%s"%(r["event"],r["cluster"]) for r in rd(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv")}
rec=json.load(open(f"{X}/pdhd_stm_michel_smx22_verdicts.json"))
def truth(r):
    if r.get("owner_review"):
        o=r["owner_review"]; return base(o["verdict"]), o.get("michel_kind"), "owner_review"
    if r.get("owner_smx1"):
        o=r["owner_smx1"]; return base(o.get("choice") or o.get("label")), o.get("michel_kind"), "owner"
    return base(r["verdict"]), r["michel_kind"], "agent"
def arm(t):
    o={}
    for f in sorted(glob.glob(f"{IMG}/pdhd/work/*_{t}/tracking-pr.root")):
        e=os.path.basename(os.path.dirname(f)).replace("_"+t,"")
        c=uproot.open(f)["T_stm_michel"].arrays(["cluster_id","michel_found"],library="np")
        for i in range(len(c["cluster_id"])):
            o[f"{e}/{int(c['cluster_id'][i])}"]=int(c["michel_found"][i])
    return o
print("michel_found graded on hand stoppers only; truth = michel_kind in (attached, both);")
print("an owner_review stopper with no kind is EXCLUDED, never scored 'no Michel'.\n")
for t in sys.argv[1:]:
    A=arm(t)
    if not A: print(f"  {t:10s} NO OUTPUT"); continue
    tp=fp=fn=tn=nk=0
    for r in rec:
        k=r["key"]
        if k not in POP or k not in A: continue
        v,kind,src=truth(r)
        if v not in ("STM_MICHEL","STM_ONLY"): continue
        if src=="owner_review" and kind is None: nk+=1; continue
        hm=kind in ("attached","both"); cm=A[k]==1
        if hm and cm: tp+=1
        elif hm: fn+=1
        elif cm: fp+=1
        else: tn+=1
    p=tp/max(1,tp+fp); e=tp/max(1,tp+fn)
    print(f"  {t:10s} TP {tp:3d} FP {fp:3d} FN {fn:3d} TN {tn:3d} | purity {p:.3f} efficiency {e:.3f} (excluded {nk})")

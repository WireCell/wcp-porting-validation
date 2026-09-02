#!/bin/bash
# doc 97 -- run the sep_fv_point decomposition arms (see d97_fv_decomp.py).
set -u
cd -P "$(dirname "$0")/.." || exit 1
export LD_LIBRARY_PATH=$HOME/tmp/d97b-libsnap:${LD_LIBRARY_PATH:-}
export WIRECELL_PATH=/nfs/data/1/xqian/toolkit-dev/toolkit/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:/nfs/data/1/xqian/toolkit-dev/wire-cell-data/sbnd/photodet
R=${D97_DECOMP_OUT:-/home/xqian/tmp/d97/decomp}
for evt in "$@"; do
    for arm in off inset farpoint dec1 inset+far all4; do
        setarch x86_64 -R wire-cell -l "$R/$arm/ql_evt$evt/wct.log:info" -L info \
            -c "$R/$arm/evt$evt.json" > "$R/$arm/ql_evt$evt/stdout.log" 2>&1
        rc=$?
        n=$(python3 - "$R/$arm/ql_evt$evt/mabc-all-apa.zip" <<'PY'
import sys, zipfile, json
try:
    z = zipfile.ZipFile(sys.argv[1])
    nm = [x for x in z.namelist() if 'clustering' in x and x.endswith('.json')]
    d = json.loads(z.read(nm[0]))
    ids = d.get('cluster_id') or d.get('real_cluster_id') or []
    from collections import Counter
    c = Counter(ids)
    print(f"{len(c)} clusters, top5 {sorted(c.values(), reverse=True)[:5]}")
except Exception as e:
    print('n/a', e)
PY
)
        echo "evt$evt $arm rc=$rc recarve=$(grep -ac 'Separate track_recarve' "$R/$arm/ql_evt$evt/stdout.log" 2>/dev/null) | $n"
    done
done

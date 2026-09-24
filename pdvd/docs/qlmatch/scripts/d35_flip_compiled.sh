#!/bin/bash
# doc qlmatch/35 F1(a) -- compiled-config proof of the Q/L runner-default flip (the doc 29 flip_compiled_config form).
#   bash d35_flip_compiled.sh pre     BEFORE the run_clus_evt.sh edit: bare, and with the two Q/L env assignments
#   bash d35_flip_compiled.sh post    AFTER the edit: bare, and with the escape assignments; then the comparison
# run_clus_evt.sh -s q35cfg -calib -save-pctree <run> <idx>, PDVD_CLUS_COMPILE_ONLY=1 PDVD_KEEP_CFG=1,
# PDVD_LIGHT_SUFFIX=_q32ti for every variant (the light path is in the config), other PDVD_* unset.
# Compiled JSONs are kept under /home/xqian/tmp/p35/cfg/<event>_<variant>.json; the report goes to stdout.
set -u
PHASE=${1:?pre or post}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
OUT=/home/xqian/tmp/p35/cfg; mkdir -p "$OUT"
EVS="039252_0 039253_15 039349_7"
declare -A VAR
if [ "$PHASE" = pre ]; then
    VAR[pre_bare]=""
    VAR[pre_env]="PDVD_QL_LASSO_W_UNRAILED=1 PDVD_QL_KS_SAT_TOL=0.3075"
else
    VAR[post_bare]=""
    VAR[post_esc]="PDVD_QL_LASSO_W_UNRAILED=0 PDVD_QL_KS_SAT_TOL="
fi
for e in $EVS; do
    run=${e%_*}; idx=${e#*_}
    [ -d "$PDVD/work/${e}_q35cfg" ] || (cd "$PDVD" && ./scripts/stage_ql_tag.sh "$((10#$run))" "$idx" q35cfg > /dev/null) || exit 2
    for v in "${!VAR[@]}"; do
        (cd "$PDVD" && env -u PDVD_READOUT_NTICKS -u PDVD_QTOL -u PDVD_CLUS_WIRES ${VAR[$v]} PDVD_LIGHT_SUFFIX=_q32ti \
            PDVD_CLUS_COMPILE_ONLY=1 PDVD_KEEP_CFG=1 ./run_clus_evt.sh -s q35cfg -calib -save-pctree "$((10#$run))" "$idx") \
            > "$OUT/${e}_$v.log" 2>&1 || { echo "compile FAIL $e $v"; exit 2; }
        cp "$PDVD/work/${e}_q35cfg/.wct-clus.json" "$OUT/${e}_$v.json"
    done
done
[ "$PHASE" = pre ] && { echo "pre-flip compiles written to $OUT"; exit 0; }
echo "# doc qlmatch/35 F1(a) -- compiled-config proof of the PDVD_QL_LASSO_W_UNRAILED=1 / PDVD_QL_KS_SAT_TOL=0.3075 runner defaults"
echo "# run_clus_evt.sh -s q35cfg -calib -save-pctree <run> <idx>, PDVD_CLUS_COMPILE_ONLY=1 PDVD_LIGHT_SUFFIX=_q32ti, other PDVD_* unset"
echo "# pre_bare/pre_env = before the edit (no env / the two assignments); post_bare/post_esc = after (no env / W_UNRAILED=0 KS_SAT_TOL= empty)"
fail=0
for e in $EVS; do
    m() { md5sum < "$OUT/${e}_$1.json" | cut -c1-12; }
    echo "$e   pre_bare=$(m pre_bare) pre_env=$(m pre_env) post_bare=$(m post_bare) post_esc=$(m post_esc)"
    cmp -s "$OUT/${e}_post_bare.json" "$OUT/${e}_pre_env.json" && echo "  post_bare == pre_env PASS" || { echo "  post_bare == pre_env FAIL"; fail=1; }
    cmp -s "$OUT/${e}_post_esc.json" "$OUT/${e}_pre_bare.json" && echo "  post_esc == pre_bare PASS" || { echo "  post_esc == pre_bare FAIL"; fail=1; }
    grep -q 'ks_sat_tol' "$OUT/${e}_post_bare.json" && echo "  ks_sat_tol present in post_bare" || { echo "  ks_sat_tol ABSENT in post_bare"; fail=1; }
done
e=039252_0
echo "# leaves that differ pre_bare -> pre_env ($e):"
python3 - "$OUT/${e}_pre_bare.json" "$OUT/${e}_pre_env.json" <<'EOF'
import json, sys
def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, f"{p}/{i}")
    else:
        yield p, x
a, b = dict(leaves(json.load(open(sys.argv[1])))), dict(leaves(json.load(open(sys.argv[2]))))
for k in sorted(set(a) | set(b)):
    if a.get(k, "<absent>") != b.get(k, "<absent>"):
        print(f"  {k}: {a.get(k, '<absent>')} -> {b.get(k, '<absent>')}")
EOF
echo "VERDICT F1(a): $([ $fail = 0 ] && echo PASS || echo FAIL)"
